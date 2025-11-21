"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np

from src.config.settings import FeatureConfig
from src.features.feature_engineering import (
    create_time_features,
    create_rolling_features,
    engineer_features,
)


@pytest.fixture
def sample_telemetry():
    """Create sample telemetry data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="15min")
    return pd.DataFrame({
        "timestamp": dates,
        "site_id": ["site_1"] * 100,
        "inverter_id": ["inv_1"] * 100,
        "string_id": ["str_1"] * 100,
        "irradiance_w_m2": np.random.uniform(0, 1000, 100),
        "ac_power_kw": np.random.uniform(0, 50, 100),
        "dc_power_kw": np.random.uniform(0, 55, 100),
        "energy_kwh": np.random.uniform(0, 10, 100),
        "module_temp_c": np.random.uniform(20, 60, 100),
        "ambient_temp_c": np.random.uniform(15, 35, 100),
        "inverter_status": ["OK"] * 100,
        "soiling_loss_factor": [0.0] * 100,
        "sensor_bias_factor": [0.0] * 100,
        "availability_factor": [1.0] * 100,
        "underperformance_factor": [0.0] * 100,
    })


@pytest.fixture
def sample_labels():
    """Create sample labels."""
    dates = pd.date_range("2020-01-01", periods=100, freq="15min")
    return pd.DataFrame({
        "timestamp": dates,
        "site_id": ["site_1"] * 100,
        "inverter_id": ["inv_1"] * 100,
        "string_id": ["str_1"] * 100,
        "inverter_failure_label": [0] * 100,
        "string_underperformance_label": [0] * 100,
        "sensor_drift_label": [0] * 100,
        "soiling_loss_kwh": [0.0] * 100,
    })


def test_time_features(sample_telemetry):
    """Test time feature creation."""
    df = create_time_features(sample_telemetry)
    
    assert "hour" in df.columns
    assert "day_of_week" in df.columns
    assert "month" in df.columns
    assert "hour_sin" in df.columns
    assert "hour_cos" in df.columns
    
    # Check hour values are in range
    assert (df["hour"] >= 0).all()
    assert (df["hour"] < 24).all()


def test_rolling_features(sample_telemetry):
    """Test rolling feature creation."""
    df = create_rolling_features(
        sample_telemetry,
        value_cols=["ac_power_kw"],
        windows_hours=[1, 6],
        group_cols=["inverter_id"],
        granularity_minutes=15
    )
    
    assert "ac_power_kw_rolling_mean_1h" in df.columns
    assert "ac_power_kw_rolling_mean_6h" in df.columns
    assert "ac_power_kw_rolling_std_1h" in df.columns
    
    # Check no NaN in first row (should be filled)
    assert not df["ac_power_kw_rolling_mean_1h"].iloc[0:5].isna().any()


def test_engineer_features(sample_telemetry, sample_labels):
    """Test main feature engineering function."""
    config = FeatureConfig()
    
    X, y, metadata = engineer_features(
        sample_telemetry,
        sample_labels,
        task="inverter_failure",
        config=config,
        granularity_minutes=15
    )
    
    assert len(X) == len(sample_telemetry)
    assert len(y) == len(sample_telemetry)
    assert len(metadata) == len(sample_telemetry)
    
    # Check no NaN values
    assert not X.isna().any().any()
    
    # Check no inf values
    assert not np.isinf(X.values).any()
    
    # Check feature count is reasonable
    assert len(X.columns) > 10


def test_all_tasks(sample_telemetry, sample_labels):
    """Test feature engineering for all tasks."""
    config = FeatureConfig()
    
    tasks = ["inverter_failure", "string_underperformance", "sensor_drift", "soiling"]
    
    for task in tasks:
        X, y, metadata = engineer_features(
            sample_telemetry,
            sample_labels,
            task=task,
            config=config,
            granularity_minutes=15
        )
        
        assert len(X) > 0
        assert len(y) > 0
        assert len(X.columns) > 0

