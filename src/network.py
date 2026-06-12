"""
network.py — Phase 2

Defines the StrikeNet MLP and the training loop.
Trains on data/dataset/strike_dataset.npy and saves the best model to models/strategy_net.pth.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

class StrikeNet(nn.Module):
    def __init__(self, variant: str = "legacy"):
        super().__init__()
        assert variant in ("legacy", "structured")
        self.variant = variant
        out_dim = 5 if variant == "legacy" else 3   # legacy: T,x,y,sin,cos ; structured: T,sin,cos
        self.net = nn.Sequential(
            nn.Linear(7, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )
        
        self.register_buffer('input_mean', torch.zeros(7))
        self.register_buffer('input_std', torch.ones(7))
        self.register_buffer('output_mean', torch.zeros(out_dim))
        self.register_buffer('output_std', torch.ones(out_dim))
        
    def forward(self, x):
        # x shape: (batch_size, 7)
        # Normalize input; the network predicts z-scored outputs
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x_norm)
    
    def predict(self, x):
        self.eval()
        device = self.input_mean.device
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32, device=device)
            else:
                x = x.to(device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            out = self(x) * (self.output_std + 1e-8) + self.output_mean
            T = out[:, 0].cpu().numpy()
            if self.variant == "legacy":
                x_s = out[:, 1].cpu().numpy(); y_s = out[:, 2].cpu().numpy()
                theta = np.arctan2(out[:, 3].cpu().numpy(), out[:, 4].cpu().numpy())
                preds = np.column_stack([T, x_s, y_s, theta])   # (N,4)
            else:
                theta = np.arctan2(out[:, 1].cpu().numpy(), out[:, 2].cpu().numpy())
                preds = np.column_stack([T, theta])             # (N,2)
            return preds[0] if preds.shape[0] == 1 else preds

    @classmethod
    def load(cls, path, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        out_dim = sd["output_mean"].numel()
        variant = "legacy" if out_dim == 5 else "structured"
        model = cls(variant=variant)
        model.load_state_dict(sd)
        model.eval()
        return model

def train(data_path: str, model_path: str, log_path: str, seed: int = 42, variant: str = "legacy"):
    # Set seeds for research reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    print(f"Loading data from {data_path}...")
    dataset_np = np.load(data_path)
    
    inputs = dataset_np[:, :7]
    outputs = dataset_np[:, 7:11]
    
    # Process outputs: replace theta with sin(theta), cos(theta)
    T_s = outputs[:, 0]
    x_s = outputs[:, 1]
    y_s = outputs[:, 2]
    theta_s = outputs[:, 3]
    
    if variant == "legacy":
        outputs_transformed = np.column_stack([
            T_s, x_s, y_s, np.sin(theta_s), np.cos(theta_s)
        ])
    else:
        outputs_transformed = np.column_stack([
            T_s, np.sin(theta_s), np.cos(theta_s)
        ])
    
    # Create tensors
    X = torch.tensor(inputs, dtype=torch.float32)
    Y = torch.tensor(outputs_transformed, dtype=torch.float32)
    
    # Split train/test
    dataset = TensorDataset(X, Y)
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # Initialize model
    model = StrikeNet(variant=variant)
    
    # Calculate input/output normalization from the training split only
    train_indices = train_dataset.indices
    X_train = X[train_indices]
    Y_train = Y[train_indices]
    model.input_mean.copy_(X_train.mean(dim=0))
    model.input_std.copy_(X_train.std(dim=0))
    model.output_mean.copy_(Y_train.mean(dim=0))
    model.output_std.copy_(Y_train.std(dim=0))
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training Loop
    epochs = 200
    patience = 20
    best_test_loss = float('inf')
    epochs_no_improve = 0
    
    log_data = []
    
    print(f"Starting training on {device}...")
    from tqdm import tqdm
    pbar = tqdm(range(1, epochs + 1), desc="Training StrikeNet")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            target = (batch_y - model.output_mean) / (model.output_std + 1e-8)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Eval
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                target = (batch_y - model.output_mean) / (model.output_std + 1e-8)
                preds = model(batch_x)
                loss = criterion(preds, target)
                test_loss += loss.item() * batch_x.size(0)
            test_loss /= len(test_loader.dataset)
            
        log_data.append({'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss})
        
        pbar.set_postfix(train_loss=f"{train_loss:.4f}", test_loss=f"{test_loss:.4f}", best=f"{best_test_loss if best_test_loss != float('inf') else test_loss:.4f}")
            
        # Early stopping and saving best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                pbar.write(f"Early stopping at epoch {epoch}! Best Test Loss: {best_test_loss:.4f}")
                break
                
    # Save log
    import pandas as pd
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pd.DataFrame(log_data).to_csv(log_path, index=False)
    print(f"Training complete. Best model saved to {model_path}.")
    print(f"Training log saved to {log_path}.")

if __name__ == "__main__":
    import argparse
    from src.data_layout import STRIKE_DATASET, model_path_for_variant, training_log_for_variant
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["legacy", "structured", "both"], default="both")
    a = p.parse_args()
    variants = ["legacy", "structured"] if a.variant == "both" else [a.variant]
    for v in variants:
        print(f"\n=== Training StrikeNet variant: {v} ===")
        train(str(STRIKE_DATASET), str(model_path_for_variant(v)),
              str(training_log_for_variant(v)), variant=v)
