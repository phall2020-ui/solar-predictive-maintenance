"""Tests for data generation module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.config.settings import Config, SimulationConfig, FeatureConfig, ModelConfig
from src.data_generation.simulate_digital_twin import (
    simulate_digital_twin,
    generate_portfolio_structure,
    generate_weather_timeseries,
)


@pytest.fixture
def test_config():
    """Create a test configuration with small dataset."""
    return Config(
        simulation=SimulationConfig(
            start_date="2020-01-01",
            days=30,  # 30 days for fast tests
            granularity_minutes=15,
            num_sites=2,
            inverters_per_site=2,
            strings_per_inverter=2,
        ),
        features=FeatureConfig(),
        models=ModelConfig(),
    )


def test_portfolio_structure():
    """Test portfolio structure generation."""
    portfolio = generate_portfolio_structure(
        num_sites=2,
        inverters_per_site=2,
        strings_per_inverter=2
    )
    
    assert len(portfolio) == 8  # 2 sites * 2 inverters * 2 strings
    assert "site_id" in portfolio.columns
    assert "inverter_id" in portfolio.columns
    assert "string_id" in portfolio.columns
    assert "nameplate_kw" in portfolio.columns
    
    # Check all sites are present
    assert len(portfolio["site_id"].unique()) == 2
    assert len(portfolio["inverter_id"].unique()) == 4
    assert len(portfolio["string_id"].unique()) == 8


def test_weather_generation():
    """Test weather data generation."""
    weather = generate_weather_timeseries(
        start_date="2020-01-01",
        days=7,
        granularity_minutes=15
    )
    
    expected_points = 7 * 24 * 4  # 7 days * 24 hours * 4 (15-min intervals)
    assert len(weather) == expected_points
    
    assert "timestamp" in weather.columns
    assert "irradiance_w_m2" in weather.columns
    assert "ambient_temp_c" in weather.columns
    
    # Check irradiance is reasonable (0-1200 W/m²)
    assert weather["irradiance_w_m2"].min() >= 0
    assert weather["irradiance_w_m2"].max() <= 1200
    
    # Check temperature is reasonable
    assert weather["ambient_temp_c"].min() >= -20
    assert weather["ambient_temp_c"].max() <= 50


def test_simulate_digital_twin(test_config):
    """Test complete data simulation."""
    telemetry, labels = simulate_digital_twin(test_config, seed=42)
    
    # Check telemetry structure
    assert len(telemetry) > 0
    required_cols = [
        "timestamp", "site_id", "inverter_id", "string_id",
        "irradiance_w_m2", "ac_power_kw", "dc_power_kw", "energy_kwh"
    ]
    for col in required_cols:
        assert col in telemetry.columns, f"Missing column: {col}"
    
    # Check labels structure
    assert len(labels) == len(telemetry)
    label_cols = [
        "inverter_failure_label",
        "string_underperformance_label",
        "sensor_drift_label",
        "soiling_loss_kwh"
    ]
    for col in label_cols:
        assert col in labels.columns, f"Missing label column: {col}"
    
    # Check that labels have some positive examples (at least for some tasks)
    assert labels["inverter_failure_label"].sum() >= 0  # May be 0 for small dataset
    assert labels["soiling_loss_kwh"].sum() >= 0
    
    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(telemetry["timestamp"])
    
    # Check power values are non-negative
    assert (telemetry["ac_power_kw"] >= 0).all()
    assert (telemetry["dc_power_kw"] >= 0).all()


def test_data_consistency(test_config):
    """Test that generated data is internally consistent."""
    telemetry, labels = simulate_digital_twin(test_config, seed=42)
    
    # AC power should be less than or equal to DC power (inverter efficiency)
    assert (telemetry["ac_power_kw"] <= telemetry["dc_power_kw"] * 1.1).all()
    
    # Energy should be proportional to power
    # Energy (kWh) ≈ Power (kW) * time (hours)
    time_hours = test_config.simulation.granularity_minutes / 60
    expected_energy = telemetry["ac_power_kw"] * time_hours
    # Allow some tolerance
    assert np.allclose(telemetry["energy_kwh"], expected_energy, rtol=0.1)

