# Robot Soccer Striker — Motion Planning

Closed-loop striker: **StrikeNet** predicts when/where to intercept the ball; **NMPC** drives the car; **World** simulates the kinematic bicycle and bouncing ball, detecting elastic collisions to redirect the ball into the goal.

**Documentation:** [docs/README.md](docs/README.md) — system overview, physics/constraints, pipeline logic, data & report plots, and phase updates.

## Setup

```powershell
conda activate striker
pip install -r requirements.txt
pip install imageio imageio-ffmpeg
```

### GPU PyTorch (RTX 40-series / CUDA 12.6 driver)

Plain `pip install torch` often installs **CPU-only** (`2.x.x+cpu`). For NVIDIA GPU training:

```powershell
conda activate striker
pip uninstall torch torchvision torchaudio -y
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu126
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expect `2.12.0+cu126` and `CUDA: True`. You do **not** need the full CUDA Toolkit — only an up-to-date NVIDIA driver.

Use the **striker** env for all commands below (`conda activate striker` or full path to `striker\python.exe`).

## Full pipeline from scratch

```powershell
conda activate striker
cd D:\SNU\Semester_6\motion_planning\project_retry

# 1) Dataset (scoring-aware labels, 1D theta sweep)
python -m src.data_generator --num_samples 100000

# 2) Train StrikeNet (uses GPU if CUDA available)
python -m src.network

# 3) Integration test — 50 seeds (100–149), each with trajectory + simulation.mp4 + metadata
python scripts/test_main.py

# 4) Report plots → data/reports/plots/integration/{batch_id}/seed_{N}/
python scripts/generate_plots.py
python scripts/generate_plots.py --batch 20260522_035708
```

Custom seeds:

```powershell
python scripts/test_main.py --seeds 10 21 32 43 54 7 14 28 35 42
```

Single manual demo:

```powershell
python src/main.py --seed 10 --save-video
```

## Project layout

| Path | Role |
|------|------|
| `src/simulator.py` | World dynamics (RK4 car, ball propagation) + rendering |
| `src/ball_physics.py` | Shared inelastic wall bounce and elastic car-ball collision models |
| `src/goal.py` | Goal mouth segment definition and scoring check |
| `src/nmpc_solver.py` | CasADi shrinking-horizon MPC with pursuit-based warm-start and silenced solver logging |
| `src/network.py` | StrikeNet MLP (7-D input to 5-D output) |
| `src/main.py` | Closed-loop two-phase simulation (interception and braking/ball flight) |
| `src/data_layout.py` | Canonical `data/` paths |
| `scripts/` | Tests and `generate_plots.py` |
| `models/strategy_net.pth` | Trained StrikeNet weights |
| `data/` | Datasets, runs, tests, plots — see [`data/README.md`](data/README.md) |

## Phases

1. Simulator + NMPC (static target)  
2. Dataset + StrikeNet  
3. Real-time interception loop  
3.6. Full-pipeline ball bounce  
4. Report plots  
5. **Strike & Score** (elastic bumper collisions, goal line scoring checks, active braking, target offset)

Details: `data/phase_archives/`
