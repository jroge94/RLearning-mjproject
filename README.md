# CSC4444-Project

## Git Bash (Windows)

From Git Bash in the project root:

```bash
./install.sh
```

This installs Python dependencies and creates `logs`, `ckpt`, and `imgs`. Install MuJoCo for Windows manually and set `MUJOCO_PATH` or `LD_LIBRARY_PATH` if required.

### Running experiments and plots (CLI)

Use the main script `run.py`:

```bash
# Run all experiments (algos × envs from config)
python run.py experiment

# Run a subset
python run.py experiment --algos ppo --envs Hopper-v3
python run.py experiment --algos ppo,impala --envs all --rounds 10

# List available algos and envs
python run.py experiment --list

# Generate plots from logs
python run.py plot
python run.py plot --logs-dir logs --imgs-dir imgs --envs "Hopper-v3,Walker2d-v3"
```

Legacy entry points still work: `python run_experiment.py`, `python plot.py`.
