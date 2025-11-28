# Fouling Detection Module

## Overview

The `fouling_detection.py` module provides comprehensive functionality for detecting and quantifying soiling (fouling) on solar PV systems using operational data. The module implements advanced POA-based normalisation to remove seasonal effects, enabling accurate fouling detection throughout the year.

## Key Features

### Core Functionality
- **Performance Ratio (PR) Calculation**: Computes PR using AC power, POA irradiance, and nameplate capacity
- **POA-Based Seasonality Normalisation**: Bins data by POA levels (100 W/m² bins) to compare performance at same irradiance levels, effectively removing seasonal effects
- **Clean Reference Period Identification**: Automatically identifies periods with stable high PR as baseline
- **Fouling Index Calculation**: Derives 0-100% fouling index from actual/expected power ratio
- **Classification System**: Maps fouling to 4 categories:
  - Clean (0-5% loss)
  - Light Soiling (5-10% loss)
  - Moderate (10-20% loss)
  - Severe (>20% loss)
- **Energy Loss Estimation**: Calculates daily kWh losses due to fouling
- **Cleaning Event Detection**: Identifies PR improvements indicating cleaning

### Robust Design
- Handles missing data gracefully (temperature, DC power optional)
- Filters out low POA readings (<200 W/m²) to avoid noise
- Uses POA bin matching for robust seasonality handling
- Comprehensive error handling and edge case management

## Quick Start

### Basic Usage

```python
import pandas as pd
from src.fouling_detection import run_fouling_analysis

# Load your data
df = pd.read_csv("sample_site_data.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp")

# Run analysis
results = run_fouling_analysis(
    df,
    ac_power_col="ac_power_kw",
    poa_col="poa_w_m2",
    nameplate_dc_capacity=1000.0
)

# View results
print(f"Fouling Index: {results['fouling_index']:.1%}")
print(f"Fouling Level: {results['fouling_level']}")
print(f"Energy Loss: {results['energy_loss_kwh_per_day']:.2f} kWh/day")
```

### Required Data Format

Your dataframe must include:
- `timestamp`: DateTime column (ISO format recommended)
- `ac_power_kw`: AC power output in kW
- `poa_w_m2`: Plane-of-array irradiance in W/m²

Optional columns:
- `dc_power_kw`: DC power input in kW
- `module_temp_c`: Module temperature in °C

## Technical Details

### POA-Based Seasonality Normalisation

The core innovation of this module is POA-based normalisation, which addresses a critical challenge in fouling detection: distinguishing between seasonal variations and actual soiling.

**The Problem:**
- Solar irradiance varies throughout the year due to sun angle changes
- Performance naturally varies with irradiance and temperature
- Naive comparisons of power output over time confuse seasonal effects with fouling

**The Solution:**
- Bin data by POA level (default: 100 W/m² bins)
- For each POA bin, calculate expected "clean" performance from historical clean periods
- Compare current performance to historical performance at the **same POA level**
- This removes the confounding effect of seasonal irradiance changes

See full documentation in the module docstrings for API reference and examples.
