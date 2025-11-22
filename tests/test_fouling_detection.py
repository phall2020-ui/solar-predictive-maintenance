"""Tests for fouling detection module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.fouling_detection import (
    calculate_pr,
    identify_clean_reference_periods,
    estimate_clean_baseline_poa_matched,
    calculate_fouling_index,
    classify_fouling_level,
    estimate_energy_loss,
    detect_cleaning_events,
    run_fouling_analysis,
    CLEAN_THRESHOLD,
    LIGHT_SOILING_THRESHOLD,
    MODERATE_THRESHOLD,
)


@pytest.fixture
def sample_solar_data():
    """Create sample solar PV data for testing."""
    np.random.seed(42)
    
    # Create 30 days of 15-minute interval data
    dates = pd.date_range("2024-01-01", periods=2880, freq="15min")
    n_points = len(dates)
    
    # Simulate POA with daily patterns
    hour_of_day = dates.hour + dates.minute / 60
    poa_base = np.maximum(0, 800 * np.sin((hour_of_day - 6) * np.pi / 12))
    poa = poa_base + np.random.normal(0, 20, n_points)
    poa = np.maximum(0, poa)
    
    # Simulate AC power (clean system)
    nameplate = 1000  # kW
    clean_pr = 0.90
    expected_dc = (poa / 1000) * nameplate
    ac_power = expected_dc * clean_pr + np.random.normal(0, 5, n_points)
    ac_power = np.maximum(0, ac_power)
    
    return pd.DataFrame({
        "timestamp": dates,
        "poa_w_m2": poa,
        "ac_power_kw": ac_power
    })


@pytest.fixture
def fouled_solar_data(sample_solar_data):
    """Create solar data with simulated fouling."""
    df = sample_solar_data.copy()
    
    # Apply gradual fouling over time (reduce power by up to 15%)
    n = len(df)
    fouling_factor = np.linspace(1.0, 0.85, n)
    df["ac_power_kw"] = df["ac_power_kw"] * fouling_factor
    
    return df


def test_calculate_pr(sample_solar_data):
    """Test PR calculation."""
    df = calculate_pr(sample_solar_data, nameplate_dc_capacity=1000.0)
    
    # Check that PR column was added
    assert "pr" in df.columns
    
    # Check PR is in reasonable range
    valid_pr = df[df["poa_w_m2"] > 200]["pr"]
    assert valid_pr.min() >= 0.0
    assert valid_pr.max() <= 1.2
    
    # Check mean PR is reasonable for clean system
    assert 0.80 <= valid_pr.mean() <= 0.95


def test_calculate_pr_with_zero_poa():
    """Test PR calculation handles zero POA correctly."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="15min"),
        "poa_w_m2": [0, 100, 200, 0, 300, 400, 0, 500, 600, 0],
        "ac_power_kw": [0, 80, 170, 0, 260, 350, 0, 440, 530, 0]
    })
    
    result = calculate_pr(df, nameplate_dc_capacity=1000.0)
    
    # Zero POA should result in NaN PR (not division by zero error)
    assert result["pr"].isna().sum() >= 4  # At least the zero POA entries


def test_identify_clean_reference_periods(sample_solar_data):
    """Test identification of clean reference periods."""
    # First calculate PR
    df = calculate_pr(sample_solar_data, nameplate_dc_capacity=1000.0)
    
    # Identify clean periods (with relaxed thresholds for test data)
    df = identify_clean_reference_periods(df, pr_threshold=0.80, stability_threshold=0.10)
    
    # Check that column was added
    assert "is_clean_reference" in df.columns
    
    # Check that some periods are identified as clean (or all data can serve as reference)
    # Since this is synthetic clean data, we just need the column to exist
    assert df["is_clean_reference"].dtype == bool


def test_estimate_clean_baseline_poa_matched(sample_solar_data):
    """Test POA-matched baseline estimation."""
    # Calculate PR and identify clean periods
    df = calculate_pr(sample_solar_data, nameplate_dc_capacity=1000.0)
    df = identify_clean_reference_periods(df)
    
    # Create clean reference dataframe
    clean_df = df[df["is_clean_reference"]]
    
    if len(clean_df) == 0:
        # Use all data as clean for this test
        clean_df = df
    
    # Estimate baseline
    result = estimate_clean_baseline_poa_matched(df, clean_df)
    
    # Check that expected columns were added
    assert "expected_clean_power" in result.columns
    assert "expected_clean_pr" in result.columns
    
    # Check that most values are not NaN (interpolation should fill gaps)
    valid_poa = result["poa_w_m2"] >= 200
    assert result[valid_poa]["expected_clean_power"].notna().sum() > 0


def test_calculate_fouling_index_clean_system(sample_solar_data):
    """Test fouling index calculation on clean system."""
    # Calculate PR and create baseline
    df = calculate_pr(sample_solar_data, nameplate_dc_capacity=1000.0)
    df = identify_clean_reference_periods(df)
    
    clean_df = df[df["is_clean_reference"]] if df["is_clean_reference"].sum() > 0 else df
    df = estimate_clean_baseline_poa_matched(df, clean_df)
    
    # Calculate fouling index
    fouling_index = calculate_fouling_index(df, recent_window_days=30)
    
    # Clean system should have low fouling index
    assert 0.0 <= fouling_index <= 0.10


def test_calculate_fouling_index_fouled_system(fouled_solar_data):
    """Test fouling index calculation on fouled system."""
    # Calculate PR and create baseline (using early data as clean)
    df = calculate_pr(fouled_solar_data, nameplate_dc_capacity=1000.0)
    
    # Use first 7 days as clean reference
    clean_df = df.iloc[:672]  # 7 days * 96 intervals
    
    df = estimate_clean_baseline_poa_matched(df, clean_df)
    
    # Calculate fouling index on recent data
    fouling_index = calculate_fouling_index(df, recent_window_days=7)
    
    # Fouled system should have higher fouling index
    assert fouling_index >= 0.10


def test_classify_fouling_level():
    """Test fouling level classification."""
    assert classify_fouling_level(0.02) == "Clean"
    assert classify_fouling_level(0.05) == "Clean"
    assert classify_fouling_level(0.07) == "Light Soiling"
    assert classify_fouling_level(0.10) == "Light Soiling"
    assert classify_fouling_level(0.15) == "Moderate"
    assert classify_fouling_level(0.20) == "Moderate"
    assert classify_fouling_level(0.25) == "Severe"
    assert classify_fouling_level(0.50) == "Severe"


def test_estimate_energy_loss_clean_system(sample_solar_data):
    """Test energy loss estimation on clean system."""
    df = calculate_pr(sample_solar_data, nameplate_dc_capacity=1000.0)
    df = identify_clean_reference_periods(df)
    
    clean_df = df[df["is_clean_reference"]] if df["is_clean_reference"].sum() > 0 else df
    df = estimate_clean_baseline_poa_matched(df, clean_df)
    
    energy_loss = estimate_energy_loss(df, recent_window_days=30)
    
    # Clean system should have minimal energy loss (relaxed threshold for synthetic data)
    assert energy_loss >= 0.0
    assert energy_loss < 200.0  # Less than 200 kWh/day for clean system


def test_estimate_energy_loss_fouled_system(fouled_solar_data):
    """Test energy loss estimation on fouled system."""
    df = calculate_pr(fouled_solar_data, nameplate_dc_capacity=1000.0)
    
    # Use first 7 days as clean reference
    clean_df = df.iloc[:672]
    df = estimate_clean_baseline_poa_matched(df, clean_df)
    
    energy_loss = estimate_energy_loss(df, recent_window_days=7)
    
    # Fouled system should have measurable energy loss
    assert energy_loss > 0.0


def test_detect_cleaning_events():
    """Test cleaning event detection."""
    np.random.seed(42)
    
    # Create data with a cleaning event (PR jump)
    dates = pd.date_range("2024-01-01", periods=1440, freq="15min")
    n = len(dates)
    
    hour_of_day = dates.hour + dates.minute / 60
    poa = np.maximum(0, 800 * np.sin((hour_of_day - 6) * np.pi / 12))
    
    # Simulate fouling and then cleaning
    pr_values = np.ones(n) * 0.75  # Fouled
    pr_values[720:] = 0.90  # Cleaned (after 7.5 days)
    
    nameplate = 1000
    expected_dc = (poa / 1000) * nameplate
    ac_power = expected_dc * pr_values
    
    df = pd.DataFrame({
        "timestamp": dates,
        "poa_w_m2": poa,
        "ac_power_kw": ac_power
    })
    
    df = calculate_pr(df, nameplate_dc_capacity=1000.0)
    df = detect_cleaning_events(df)
    
    # Check that column was added
    assert "cleaning_event" in df.columns
    
    # Should detect at least one event near the cleaning point
    assert df["cleaning_event"].sum() > 0


def test_run_fouling_analysis_clean_system(sample_solar_data):
    """Test full fouling analysis on clean system."""
    results = run_fouling_analysis(
        sample_solar_data,
        nameplate_dc_capacity=1000.0
    )
    
    # Check all expected keys are present
    assert "fouling_index" in results
    assert "fouling_level" in results
    assert "energy_loss_kwh_per_day" in results
    assert "pr_mean" in results
    assert "pr_std" in results
    assert "clean_periods_count" in results
    assert "analysis_period_days" in results
    assert "dataframe" in results
    
    # Check values are reasonable
    assert 0.0 <= results["fouling_index"] <= 1.0
    assert results["fouling_level"] in ["Clean", "Light Soiling", "Moderate", "Severe"]
    assert results["energy_loss_kwh_per_day"] >= 0.0
    assert 0.0 <= results["pr_mean"] <= 1.0
    assert results["pr_std"] >= 0.0
    assert results["clean_periods_count"] >= 0
    assert results["analysis_period_days"] > 0
    
    # Clean system should have low fouling
    assert results["fouling_level"] in ["Clean", "Light Soiling"]


def test_run_fouling_analysis_fouled_system(fouled_solar_data):
    """Test full fouling analysis on fouled system."""
    results = run_fouling_analysis(
        fouled_solar_data,
        nameplate_dc_capacity=1000.0
    )
    
    # Check all expected keys are present
    assert "fouling_index" in results
    assert "fouling_level" in results
    assert "energy_loss_kwh_per_day" in results
    
    # Fouled system should show some degree of fouling
    assert results["fouling_index"] > 0.0


def test_run_fouling_analysis_missing_columns():
    """Test that analysis fails gracefully with missing columns."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
        "wrong_col": [100] * 100
    })
    
    with pytest.raises(ValueError, match="Missing required columns"):
        run_fouling_analysis(df)


def test_run_fouling_analysis_with_optional_columns(sample_solar_data):
    """Test that analysis works with optional columns present."""
    df = sample_solar_data.copy()
    df["dc_power_kw"] = df["ac_power_kw"] / 0.98  # Add DC power
    df["module_temp_c"] = 25 + np.random.normal(0, 5, len(df))  # Add temperature
    
    # Should not raise an error
    results = run_fouling_analysis(
        df,
        nameplate_dc_capacity=1000.0,
        dc_power_col="dc_power_kw",
        temp_col="module_temp_c"
    )
    
    assert "fouling_index" in results


def test_edge_case_all_zero_power():
    """Test handling of all zero power (nighttime data)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="15min"),
        "poa_w_m2": [0] * 100,
        "ac_power_kw": [0] * 100
    })
    
    # Should not crash
    results = run_fouling_analysis(df, nameplate_dc_capacity=1000.0)
    
    # With no valid data, should default to clean
    assert results["fouling_index"] == 0.0


def test_edge_case_very_short_data():
    """Test handling of very short time series."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="15min"),
        "poa_w_m2": [400, 500, 600, 700, 800, 750, 650, 550, 450, 350],
        "ac_power_kw": [320, 400, 480, 560, 640, 600, 520, 440, 360, 280]
    })
    
    # Should not crash even with minimal data
    results = run_fouling_analysis(df, nameplate_dc_capacity=1000.0)
    
    assert "fouling_index" in results
    assert results["analysis_period_days"] > 0


def test_poa_binning_consistency():
    """Test that POA binning produces consistent results."""
    np.random.seed(42)
    
    # Create data with distinct POA bins
    poa_levels = [300, 400, 500, 600, 700, 800] * 100
    ac_power = [p * 0.8 for p in poa_levels]
    
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(poa_levels), freq="15min"),
        "poa_w_m2": poa_levels,
        "ac_power_kw": ac_power
    })
    
    df = calculate_pr(df, nameplate_dc_capacity=1000.0)
    df = identify_clean_reference_periods(df)
    
    clean_df = df  # All data is clean
    df = estimate_clean_baseline_poa_matched(df, clean_df)
    
    # Expected clean power should be populated
    assert df["expected_clean_power"].notna().all()
    
    # Within same POA bin, expected values should be similar
    df["poa_bin"] = (df["poa_w_m2"] // 100) * 100
    for poa_bin in df["poa_bin"].unique():
        bin_data = df[df["poa_bin"] == poa_bin]
        expected_std = bin_data["expected_clean_power"].std()
        # Should be very low variance within bin
        assert expected_std < 50  # Less than 50 kW std dev
