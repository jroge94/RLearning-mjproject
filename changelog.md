# Project improvements 

## Refactors applied

- **config.py**: Added `LOGS_DIR`, `IMGS_DIR`, `CKPT_DIR`, and `PROJECT_ROOT` so paths are defined in one place. Added `get_env_names()` and `get_algo_names()` so experiments and plotting use the same env/algo lists.
- **utils.py**: Type hints; `log_csv_path()` for consistent log filenames; `_get_learner_time_ms_impala` and `_get_learner_loss()` to simplify `process_train_results`; `export_csv` uses `config.LOGS_DIR` and creates the directory; safer handling when learner keys are missing.
- **run_experiment.py**: Removed unused import; introduced `CSV_HEADER` and `_append_round_row()`; logging instead of raw prints; `trainer.stop()` in a `finally` block; `main()` with try/finally for `ray.shutdown()`; loop simplified to `range(1, stop_max_round + 1)`.
- **plot.py**: Uses `config.LOGS_DIR`, `config.IMGS_DIR`, `config.get_env_names()`, `config.get_algo_names()`; execution wrapped in `if __name__ == "__main__": main()`; plot functions take `logs_dir`/`img_dir`; uses `utils.log_csv_path()` for filenames.
- **requirements.txt**: Removed duplicate `pygame`; added `matplotlib` explicitly (used by `plot.py`).

---

## TODOs

### 1. CLI for experiments
Add `argparse` (or `click`) to `run_experiment.py` 
shell, e.g.:
```bash
python run_experiment.py --algos ppo --envs Hopper-v3
python run_experiment.py --algos ppo,impala --envs all
```

### 2. Checkpointing and resume
- Save trainer checkpoints under `config.CKPT_DIR` (e.g. every N rounds or at the end).
- Add a `--resume` flag to load from a checkpoint and continue training (RLlib supports this).

### 3. Plot script CLI
Allow overriding paths and which envs/models to plot:
```bash
python plot.py --logs-dir logs --imgs-dir imgs
```

### 4. Tests
- **Unit**: `process_train_results` with minimal fake `train_results` dicts; `log_csv_path` and CSV row shape.
- **Integration**: One short run (e.g. 1 round, 1 worker) to ensure Ray + env + CSV export work.

### 5. Logging to file
Optionally log experiment progress to a file (e.g. `logs/experiment.log`) in addition to stdout, and rotate by run or size.

### 6. Config validation
Validate `config.envs` and `config.algos` at import or in `main()` (e.g. env names exist in Gymnasium, algo names are in a known set) to fail fast.

### 7. Dependencies
- Pin `torch` to a version compatible with your Ray/RLlib and CUDA (if any).
- Consider `pip-compile` or a lockfile for reproducible installs.

### 8. Optional: Hydra or YAML config
For many hyperparameters, a YAML config (or Hydra) can make it easier to launch multiple runs (e.g. different `stop_max_round`, workers, envs) without editing Python.
