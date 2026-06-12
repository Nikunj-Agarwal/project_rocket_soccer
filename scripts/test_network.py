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

from src.data_layout import STRIKE_DATASET, model_path_for_variant
from src.network import StrikeNet

def test_variant(variant: str, dataset: np.ndarray):
    model_path = str(model_path_for_variant(variant))
    
    if not os.path.exists(model_path):
        print(f"Model {variant} not found at {model_path}. Run `python -m src.network --variant {variant}` first.")
        return 1
        
    # Pick 10 random samples
    np.random.seed(42)
    indices = np.random.choice(len(dataset), 10, replace=False)
    samples = dataset[indices]
    
    inputs = samples[:, :7]
    gt_outputs = samples[:, 7:11]
    
    # Load model
    model = StrikeNet.load(model_path)
    
    # Predict
    preds = model.predict(inputs)
    
    print("="*60)
    print(f"  StrikeNet Inference Sanity Check ({variant})")
    print("="*60)
    
    for i in range(10):
        print(f"Sample {i+1}:")
        print(f"  Inputs (ball_x, ball_y, vx, vy, car_x, car_y, car_theta):")
        print(f"    {inputs[i]}")
        print(f"  Ground Truth (T, x, y, theta):")
        print(f"    {gt_outputs[i]}")
        if variant == "legacy":
            print(f"  Prediction (T, x, y, theta):")
        else:
            print(f"  Prediction (T, theta):")
        print(f"    {preds[i]}")
        
        # Calculate errors
        T_err = abs(preds[i][0] - gt_outputs[i][0])
        if variant == "legacy":
            pos_err = np.linalg.norm(preds[i][1:3] - gt_outputs[i][1:3])
            pred_theta = preds[i][3]
        else:
            pos_err = float('nan')
            pred_theta = preds[i][1]
            
        # Heading error accounting for wrap around
        h_err = abs(np.arctan2(
            np.sin(pred_theta - gt_outputs[i][3]),
            np.cos(pred_theta - gt_outputs[i][3])
        ))
        
        if variant == "legacy":
            print(f"  Errors: T: {T_err:.3f}s | pos: {pos_err:.3f}m | heading: {h_err:.3f}rad")
        else:
            print(f"  Errors: T: {T_err:.3f}s | heading: {h_err:.3f}rad")
        print("-" * 60)
        
    T_errs = np.abs(preds[:, 0] - gt_outputs[:, 0])
    if variant == "legacy":
        pos_errs = np.linalg.norm(preds[:, 1:3] - gt_outputs[:, 1:3], axis=1)
        pred_thetas = preds[:, 3]
    else:
        pos_errs = np.full(len(preds), np.nan)
        pred_thetas = preds[:, 1]
        
    heading_errs = np.abs(np.arctan2(
        np.sin(pred_thetas - gt_outputs[:, 3]),
        np.cos(pred_thetas - gt_outputs[:, 3])
    ))
    
    print(f"Average Errors over 10 samples:")
    print(f"  T       : {np.mean(T_errs):.3f} s")
    if variant == "legacy":
        print(f"  Pos     : {np.mean(pos_errs):.3f} m")
    print(f"  Heading : {np.mean(heading_errs):.3f} rad")
    print("="*60)
    
    return 0

def main():
    data_path = str(STRIKE_DATASET)
    if not os.path.exists(data_path):
        print("Data not found. Run data_generator.py first.")
        return 1
    dataset = np.load(data_path)
    
    test_variant("legacy", dataset)
    test_variant("structured", dataset)
    return 0

if __name__ == "__main__":
    sys.exit(main())
