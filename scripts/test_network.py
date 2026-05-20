"""
test_network.py — Phase 2 sanity check

Loads the trained StrikeNet and evaluates it on a few samples
from the dataset to ensure the predictions are reasonable.
"""

import os
import sys
import numpy as np
import torch

# Add project root to path so we can import src.*
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.network import StrikeNet

def main():
    data_path = os.path.join(project_root, "data", "strike_dataset.npy")
    model_path = os.path.join(project_root, "models", "strategy_net.pth")
    
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Data or model not found. Run data_generator.py and network.py first.")
        return 1
        
    # Load data
    dataset = np.load(data_path)
    # Pick 10 random samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), 10, replace=False)
    samples = dataset[indices]
    
    inputs = samples[:, :7]
    gt_outputs = samples[:, 7:11]
    
    # Load model
    model = StrikeNet()
    # Handle map_location for CPU if CUDA not available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Predict
    preds = model.predict(inputs)
    
    # Compare
    print("="*60)
    print(f"  StrikeNet Inference Sanity Check")
    print("="*60)
    
    for i in range(10):
        print(f"Sample {i+1}:")
        print(f"  Inputs (ball_x, ball_y, vx, vy, car_x, car_y, car_theta):")
        print(f"    {inputs[i]}")
        print(f"  Ground Truth (T, x, y, theta):")
        print(f"    {gt_outputs[i]}")
        print(f"  Prediction (T, x, y, theta):")
        print(f"    {preds[i]}")
        
        # Calculate errors
        T_err = abs(preds[i][0] - gt_outputs[i][0])
        pos_err = np.linalg.norm(preds[i][1:3] - gt_outputs[i][1:3])
        # Heading error accounting for wrap around
        h_err = abs(np.arctan2(
            np.sin(preds[i][3] - gt_outputs[i][3]),
            np.cos(preds[i][3] - gt_outputs[i][3])
        ))
        
        print(f"  Errors: T: {T_err:.3f}s | pos: {pos_err:.3f}m | heading: {h_err:.3f}rad")
        print("-" * 60)
        
    # Calculate overall stats on the 10 samples
    T_errs = np.abs(preds[:, 0] - gt_outputs[:, 0])
    pos_errs = np.linalg.norm(preds[:, 1:3] - gt_outputs[:, 1:3], axis=1)
    heading_errs = np.abs(np.arctan2(
        np.sin(preds[:, 3] - gt_outputs[:, 3]),
        np.cos(preds[:, 3] - gt_outputs[:, 3])
    ))
    
    print(f"Average Errors over 10 samples:")
    print(f"  T       : {np.mean(T_errs):.3f} s")
    print(f"  Pos     : {np.mean(pos_errs):.3f} m")
    print(f"  Heading : {np.mean(heading_errs):.3f} rad")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
