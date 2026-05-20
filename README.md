# Robot Soccer Striker — Motion Planning

Closed-loop interception: **StrikeNet** predicts when/where to strike; **NMPC** drives the car; **World** simulates car + bouncing ball on a 10×6 m field.

## Setup

```powershell
conda activate striker
pip install -r requirements.txt
pip install imageio imageio-ffmpeg   # for simulation.mp4 output
```

## Quick start

```powershell
# Train (if dataset/model missing)
python -m src.data_generator --num_samples 50000
python -m src.network

# Integration test + videos per seed
python scripts/test_main.py

# Manual demo run
python src/main.py --seed 10 --save-video

# Report figures
python scripts/generate_plots.py
```

## Project layout

| Path | Role |
|------|------|
| `src/simulator.py` | World dynamics + rendering |
| `src/ball_physics.py` | Shared inelastic wall bounce |
| `src/nmpc_solver.py` | CasADi shrinking-horizon MPC |
| `src/network.py` | StrikeNet MLP |
| `src/main.py` | Closed-loop simulation |
| `scripts/` | Tests and `generate_plots.py` |
| `models/strategy_net.pth` | Trained weights |
| `data/` | Datasets, runs, tests, plots — see [`data/README.md`](data/README.md) |

## Phases

1. Simulator + NMPC (static target)  
2. Dataset + StrikeNet  
3. Real-time interception loop  
3.6. Full-pipeline ball bounce  
4. Report plots  

Details: `data/phase_archives/`
