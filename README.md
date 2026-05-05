# Robust Object Detection in Satellite Images under Distribution Shift

Research framework for training and evaluating object detectors across climate zones and disaster scenarios using RWDS benchmark datasets.

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Download datasets
python scripts/download_data.py --dataset xview --output data/raw/

# Preprocess
python scripts/preprocess_data.py --input data/raw/xview --output data/processed/

# Train
python scripts/train.py --config configs/rwds_cz/single_source/tropical.yaml --gpu 0

# Evaluate
python scripts/evaluate.py --checkpoint results/experiments/exp_id/checkpoint_best.pth --domains tropical arid temperate
```

## Project Structure

See `docs/superpowers/specs/` for architecture design.

## Citation

Coming soon.
