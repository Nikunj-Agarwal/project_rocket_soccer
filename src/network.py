"""
network.py — Phase 2

Defines the StrikeNet MLP and the training loop.
Trains on data/dataset/strike_dataset.npy and saves the best model to models/strategy_net.pth.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

class StrikeNet(nn.Module):
    def __init__(self):
        super(StrikeNet, self).__init__()
        # 7 inputs: ball_x, ball_y, ball_vx, ball_vy, car_x, car_y, car_theta
        # 5 outputs: T_strike, x_strike, y_strike, sin(theta_strike), cos(theta_strike)
        self.net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        # We will save normalization parameters in the model state
        self.register_buffer('input_mean', torch.zeros(7))
        self.register_buffer('input_std', torch.ones(7))
        
    def forward(self, x):
        # x shape: (batch_size, 7)
        # Normalize input
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        return self.net(x_norm)
    
    def predict(self, x):
        """
        Convenience method for inference.
        x: tensor or numpy array of shape (7,) or (N, 7)
        Returns: numpy array of [T_strike, x_strike, y_strike, theta_strike]
        """
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # Forward pass
            out = self.forward(x) # (N, 5)
            
            # Extract outputs
            T = out[:, 0].cpu().numpy()
            x_s = out[:, 1].cpu().numpy()
            y_s = out[:, 2].cpu().numpy()
            sin_t = out[:, 3].cpu().numpy()
            cos_t = out[:, 4].cpu().numpy()
            
            # Reconstruct theta from sin and cos
            theta = np.arctan2(sin_t, cos_t)
            
            # Shape (N, 4)
            preds = np.column_stack([T, x_s, y_s, theta])
            return preds[0] if preds.shape[0] == 1 else preds

def train(data_path: str, model_path: str, log_path: str):
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
    
    outputs_transformed = np.column_stack([
        T_s, x_s, y_s, np.sin(theta_s), np.cos(theta_s)
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
    model = StrikeNet()
    
    # Calculate input normalization from training set
    # Gather all train inputs
    train_indices = train_dataset.indices
    X_train = X[train_indices]
    model.input_mean.copy_(X_train.mean(dim=0))
    model.input_std.copy_(X_train.std(dim=0))
    
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
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
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
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                test_loss += loss.item() * batch_x.size(0)
            test_loss /= len(test_loader.dataset)
            
        log_data.append({'epoch': epoch, 'train_loss': train_loss, 'test_loss': test_loss})
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
            
        # Early stopping and saving best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}! Best Test Loss: {best_test_loss:.4f}")
                break
                
    # Save log
    pd.DataFrame(log_data).to_csv(log_path, index=False)
    print(f"Training complete. Best model saved to {model_path}.")
    print(f"Training log saved to {log_path}.")

if __name__ == "__main__":
    from src.data_layout import STRIKE_DATASET, TRAINING_LOG

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(project_root, "models", "strategy_net.pth")

    train(str(STRIKE_DATASET), model_path, str(TRAINING_LOG))
