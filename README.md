# Solar Predictive Maintenance

A comprehensive Python project for predictive maintenance of solar assets using machine learning. This project generates synthetic "digital twin" data for solar inverters and strings, engineers features, and trains models to predict various failure modes and performance issues.

## Overview

This project implements predictive maintenance models for four key prediction tasks:

1. **Inverter Failures** (Binary Classification): Predict inverter failures in the next 24 hours
2. **String Underperformance** (Classification/Regression): Detect strings that are underperforming relative to peers
3. **Sensor Drift** (Binary Classification): Detect when irradiance sensors are drifting and providing biased readings
4. **Soiling Losses** (Regression): Predict soiling-related performance losses and expected gains from cleaning

## Features

- **Synthetic Data Generation**: Realistic simulation of solar asset telemetry with internal consistency
- **Feature Engineering**: Comprehensive feature extraction including rolling statistics, lag features, peer comparisons, and performance metrics
- **Multiple ML Models**: Support for XGBoost, Random Forest, and Logistic Regression
- **Time-Aware Splitting**: Prevents data leakage by using time-based train/validation/test splits
- **Comprehensive Evaluation**: Metrics, plots, and reports for model performance
- **CLI Interface**: Easy-to-use command-line interface for all operations
- **Extensible Design**: Modular structure for easy integration with real telemetry data

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Setup

1. Clone or navigate to the project directory:
```bash
cd solar-predictive-maintenance
```

2. Install dependencies:
```bash
pip install -e .
```

Or using uv:
```bash
uv pip install -e .
```

3. Install development dependencies (optional):
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Generate Synthetic Data

Generate synthetic solar asset telemetry data:

```bash
python -m src.interfaces.cli generate-data --days 365 --sites 5 --output-dir data
```

This creates:
- `data/telemetry.csv`: Time-series telemetry data
- `data/labels.csv`: Labels for all prediction tasks

### 2. Train a Model

Train a model for inverter failure prediction:

```bash
python -m src.interfaces.cli train \
    --task inverter_failure \
    --data-dir data \
    --model-output models \
    --model-type xgboost
```

Available tasks:
- `inverter_failure`
- `string_underperformance`
- `sensor_drift`
- `soiling`

Available model types:
- `xgboost` (recommended)
- `random_forest`
- `logistic`

### 3. Evaluate Model

Generate evaluation reports and plots:

```bash
python -m src.interfaces.cli evaluate \
    --task inverter_failure \
    --model-path models/inverter_failure_xgboost.pkl \
    --data-dir data \
    --output-dir reports
```

### 4. Make Predictions

Run predictions on new data:

```bash
python -m src.interfaces.cli predict \
    --task inverter_failure \
    --model-path models/inverter_failure_xgboost.pkl \
    --input-data new_data.csv \
    --output predictions.csv
```

## Project Structure

```
solar-predictive-maintenance/
├── src/
│   ├── config/
│   │   └── settings.py          # Configuration management
│   ├── data_generation/
│   │   └── simulate_digital_twin.py  # Synthetic data generation
│   ├── features/
│   │   └── feature_engineering.py    # Feature extraction
│   ├── models/
│   │   ├── training.py          # Model training pipelines
│   │   ├── inference.py         # Prediction interface
│   │   └── evaluation.py        # Metrics and plotting
│   └── interfaces/
│       └── cli.py               # Command-line interface
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   └── 02_training_example.ipynb  # Training example
├── tests/
│   ├── test_data_generation.py
│   ├── test_feature_engineering.py
│   └── test_models.py
├── pyproject.toml              # Project configuration
└── README.md
```

## Usage Examples

### Python API

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
    telemetry, labels, 
    task="inverter_failure",
    config=config.features
)

# Train model
model_dict = train_inverter_failure_model(
    X, y, metadata, 
    config.models, 
    model_type="xgboost"
)

# Make predictions
from src.models.inference import predict
predictions = predict(model_dict, X)
```

### Jupyter Notebooks

See the `notebooks/` directory for interactive examples:
- `01_eda.ipynb`: Data generation and exploratory analysis
- `02_training_example.ipynb`: End-to-end training example

## Configuration

Configuration is managed through `src/config/settings.py`. Key parameters:

### Simulation Parameters
- `days`: Number of days to simulate
- `num_sites`: Number of solar sites
- `inverters_per_site`: Inverters per site
- `strings_per_inverter`: Strings per inverter
- `granularity_minutes`: Time step (default: 15 minutes)

### Feature Engineering
- `rolling_windows`: Window sizes for rolling features (hours)
- `lag_hours`: Lag periods for lag features
- `feature_window_hours`: History window for features

### Model Training
- `test_size`: Proportion for test set
- `val_size`: Proportion for validation set
- `use_time_split`: Use time-aware splitting (recommended)

## Prediction Tasks

### 1. Inverter Failure Prediction

**Task Type**: Binary Classification  
**Horizon**: 24 hours  
**Features**: Power patterns, temperature, status flags, fault counts, power instability  
**Use Case**: Proactive maintenance scheduling, reducing downtime

### 2. String Underperformance Detection

**Task Type**: Classification or Regression  
**Horizon**: 1 week  
**Features**: Peer comparison, relative performance, degradation trends  
**Use Case**: Identify strings needing inspection or replacement

### 3. Sensor Drift Detection

**Task Type**: Binary Classification  
**Horizon**: 30 days  
**Features**: Power mismatch (expected vs actual), peer sensor comparison, mismatch trends  
**Use Case**: Maintain data quality, prevent incorrect performance assessments

### 4. Soiling Loss Prediction

**Task Type**: Regression  
**Horizon**: 1 week  
**Features**: Performance ratio trends, yield ratios, days since cleaning  
**Use Case**: Optimize cleaning schedules, estimate revenue impact

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## Data Format

### Telemetry CSV

Required columns:
- `timestamp`: ISO format datetime
- `site_id`: Site identifier
- `inverter_id`: Inverter identifier
- `string_id`: String identifier
- `irradiance_w_m2`: Plane of array irradiance
- `ac_power_kw`: AC power output
- `dc_power_kw`: DC power input
- `energy_kwh`: Energy yield
- `module_temp_c`: Module temperature
- `ambient_temp_c`: Ambient temperature
- `inverter_status`: Status (OK/FAULT)

### Labels CSV

Required columns (matching telemetry timestamps and IDs):
- `inverter_failure_label`: Binary (0/1)
- `string_underperformance_label`: Binary (0/1)
- `sensor_drift_label`: Binary (0/1)
- `soiling_loss_kwh`: Continuous (kWh loss)

## Integration with Real Data

To use with real telemetry data:

1. **Data Format**: Ensure your telemetry matches the expected CSV format
2. **Feature Engineering**: The feature engineering module will work with any data matching the schema
3. **Labels**: You'll need to create labels based on your actual failure/event data
4. **Configuration**: Adjust simulation parameters in config to match your data characteristics

Example integration:

```python
# Load your real data
telemetry = pd.read_csv("your_telemetry.csv")
labels = pd.read_csv("your_labels.csv")

# Use existing feature engineering
X, y, metadata = engineer_features(
    telemetry, labels, 
    task="inverter_failure",
    config=config.features
)

# Train and use models as before
```

## Model Performance

Typical performance on synthetic data:

- **Inverter Failure**: ROC-AUC ~0.85-0.95 (depends on failure frequency)
- **String Underperformance**: F1 ~0.75-0.85
- **Sensor Drift**: ROC-AUC ~0.80-0.90
- **Soiling Loss**: R² ~0.70-0.85, MAE ~0.5-2.0 kWh

*Note: Performance on real data will vary based on data quality and label accuracy.*

## System Design

### Data Flow

```
Simulation/Real Data
    ↓
Feature Engineering
    ↓
Time-Aware Split
    ↓
Model Training
    ↓
Evaluation & Reports
    ↓
Inference on New Data
```

### Integration Architecture

This system can be integrated into a solar asset management platform:

1. **Data Ingestion**: Connect to telemetry APIs/databases
2. **Feature Pipeline**: Run feature engineering on streaming or batch data
3. **Model Serving**: Deploy models via API (FastAPI/Flask) or batch jobs
4. **Alerting**: Trigger alerts based on predictions
5. **Monitoring**: Track model performance and drift over time

## Contributing

This is a foundational implementation. Areas for extension:

- Real-time streaming predictions
- Model retraining pipelines
- Additional prediction tasks
- Advanced feature engineering (e.g., spectral analysis)
- Deep learning models (LSTM, Transformer)
- Integration with digital twin platforms
- API endpoints (FastAPI)

## License

This project is provided as-is for demonstration and development purposes.

## References

- Solar asset management best practices
- Predictive maintenance methodologies
- Time-series feature engineering
- Imbalanced classification techniques

## Support

For questions or issues, please refer to the code documentation or create an issue in the repository.

