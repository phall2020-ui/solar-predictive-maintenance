# Quick Start Guide

## Installation

```bash
cd solar-predictive-maintenance
pip install -e .
```

## Basic Workflow

### 1. Generate Data (30 seconds - 2 minutes)

```bash
python -m src.interfaces.cli generate-data --days 90 --sites 3
```

This creates `data/telemetry.csv` and `data/labels.csv`.

### 2. Train a Model (1-5 minutes)

```bash
python -m src.interfaces.cli train \
    --task inverter_failure \
    --data-dir data \
    --model-output models \
    --model-type xgboost
```

### 3. Evaluate Model

```bash
python -m src.interfaces.cli evaluate \
    --task inverter_failure \
    --model-path models/inverter_failure_xgboost.pkl \
    --data-dir data \
    --output-dir reports
```

Check `reports/` for metrics JSON and plots.

## Using Jupyter Notebooks

1. Start Jupyter:
```bash
jupyter notebook notebooks/
```

2. Run `01_eda.ipynb` for data exploration
3. Run `02_training_example.ipynb` for training examples

## Python API

```python
from src.config.settings import DEFAULT_CONFIG
from src.data_generation.simulate_digital_twin import simulate_digital_twin
from src.features.feature_engineering import engineer_features
from src.models.training import train_inverter_failure_model

# Generate data
config = DEFAULT_CONFIG
telemetry, labels = simulate_digital_twin(config, seed=42)

# Engineer features
X, y, metadata = engineer_features(
    telemetry, labels, "inverter_failure", config.features
)

# Train
model_dict = train_inverter_failure_model(
    X, y, metadata, config.models, "xgboost"
)

# Predict
from src.models.inference import predict
predictions = predict(model_dict, X)
```

## All Available Tasks

- `inverter_failure` - Predict inverter failures
- `string_underperformance` - Detect underperforming strings
- `sensor_drift` - Detect sensor drift
- `soiling` - Predict soiling losses

## Troubleshooting

**Import errors**: Make sure you've installed the package: `pip install -e .`

**Memory issues**: Reduce `--days` or `--sites` when generating data

**Slow training**: Use `--model-type random_forest` instead of `xgboost` for faster training

