"""
Fouling Detection Module for Solar PV Systems.

This module determines the level of soiling (fouling) on a solar PV system
using operational data, with POA-based normalisation to remove seasonal effects.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Fouling classification thresholds (percentage loss)
CLEAN_THRESHOLD = 0.05  # 0-5% loss
LIGHT_SOILING_THRESHOLD = 0.10  # 5-10% loss
MODERATE_THRESHOLD = 0.20  # 10-20% loss
# >20% is Severe

# Minimum POA threshold to avoid noise
MIN_POA_THRESHOLD = 200  # W/m²

# POA bin size for grouping similar irradiance levels
POA_BIN_SIZE = 100  # W/m²


@dataclass
class FoulingAnalysisResult:
    """Results from fouling analysis."""
    fouling_index: float  # 0-1 (0-100%)
    fouling_level: str  # Classification category
    energy_loss_kwh_per_day: float  # Estimated daily energy loss
    pr_mean: float  # Mean performance ratio
    pr_std: float  # PR standard deviation
    clean_periods_count: int  # Number of clean reference periods identified
    analysis_period_days: float  # Duration of analysis period


def calculate_pr(
    df: pd.DataFrame,
    ac_power_col: str = "ac_power_kw",
    poa_col: str = "poa_w_m2",
    nameplate_dc_capacity: float = 1000.0,
    dc_power_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute performance ratio (PR) using AC power, POA, and nameplate DC capacity.
    
    PR = (AC Power / Expected DC Power) where Expected DC Power = POA * Nameplate / 1000
    
    Args:
        df: Input dataframe with power and irradiance data
        ac_power_col: Column name for AC power (kW)
        poa_col: Column name for plane-of-array irradiance (W/m²)
        nameplate_dc_capacity: Nameplate DC capacity in kW (default 1000 kW)
        dc_power_col: Optional DC power column name
    
    Returns:
        DataFrame with added 'pr' column
    """
    df = df.copy()
    
    # Calculate expected DC power based on POA
    # Expected DC Power (kW) = (POA / 1000) * Nameplate DC Capacity
    expected_dc_power = (df[poa_col] / 1000.0) * nameplate_dc_capacity
    
    # Calculate PR: Actual AC Power / Expected DC Power
    # Avoid division by zero
    df["pr"] = np.where(
        expected_dc_power > 0,
        df[ac_power_col] / expected_dc_power,
        np.nan
    )
    
    # Cap PR at reasonable values (0 to 1.2 to handle measurement errors)
    df["pr"] = df["pr"].clip(lower=0, upper=1.2)
    
    return df


def identify_clean_reference_periods(
    df: pd.DataFrame,
    pr_col: str = "pr",
    poa_col: str = "poa_w_m2",
    window_days: int = 7,
    pr_threshold: float = 0.85,
    stability_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Identify historical "clean" periods based on stable high PR.
    
    Clean periods are identified as periods where:
    - PR is above a threshold (indicating good performance)
    - PR is stable (low standard deviation)
    - POA is above minimum threshold
    
    Args:
        df: Input dataframe with PR and POA columns
        pr_col: Column name for performance ratio
        poa_col: Column name for POA irradiance
        window_days: Rolling window size in days for stability analysis
        pr_threshold: Minimum PR for clean period (default 0.85)
        stability_threshold: Maximum PR std dev for stability (default 0.05)
    
    Returns:
        DataFrame with added 'is_clean_reference' boolean column
    """
    df = df.copy()
    
    # Filter out low POA data
    valid_poa = df[poa_col] >= MIN_POA_THRESHOLD
    
    # Calculate rolling statistics for PR
    # Assume timestamps are sorted and regular
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    
    # Estimate window size in rows (assuming regular intervals)
    if len(df) > 0 and "timestamp" in df.columns:
        time_diff = df["timestamp"].diff().median()
        if pd.notna(time_diff):
            rows_per_day = pd.Timedelta(days=1) / time_diff
            window_size = int(window_days * rows_per_day)
        else:
            window_size = window_days * 96  # Assume 15-min intervals
    else:
        window_size = window_days * 96  # Assume 15-min intervals
    
    # Calculate rolling mean and std of PR
    rolling_pr_mean = df[pr_col].rolling(window=window_size, min_periods=1).mean()
    rolling_pr_std = df[pr_col].rolling(window=window_size, min_periods=1).std()
    
    # Identify clean periods: high PR, stable PR, valid POA
    is_clean = (
        valid_poa &
        (rolling_pr_mean >= pr_threshold) &
        (rolling_pr_std <= stability_threshold) &
        (df[pr_col].notna())
    )
    
    df["is_clean_reference"] = is_clean
    
    return df


def estimate_clean_baseline_poa_matched(
    df: pd.DataFrame,
    clean_df: pd.DataFrame,
    poa_col: str = "poa_w_m2",
    ac_power_col: str = "ac_power_kw",
    pr_col: str = "pr",
    poa_bin_size: float = POA_BIN_SIZE
) -> pd.DataFrame:
    """
    Estimate clean baseline using POA-based normalisation.
    
    This is the core seasonality normalisation function. It bins data by POA level
    and estimates expected "clean" performance from historical clean periods at
    the same POA level, removing the effect of seasonal irradiance changes.
    
    Args:
        df: Full dataframe to add baseline estimates to
        clean_df: Dataframe containing only clean reference periods
        poa_col: Column name for POA irradiance
        ac_power_col: Column name for AC power
        pr_col: Column name for performance ratio
        poa_bin_size: Size of POA bins in W/m² (default 100)
    
    Returns:
        DataFrame with added 'expected_clean_power' and 'expected_clean_pr' columns
    """
    df = df.copy()
    
    # Create POA bins
    df["poa_bin"] = (df[poa_col] // poa_bin_size) * poa_bin_size
    clean_df = clean_df.copy()
    clean_df["poa_bin"] = (clean_df[poa_col] // poa_bin_size) * poa_bin_size
    
    # Calculate expected clean power and PR for each POA bin from clean periods
    clean_baseline = clean_df.groupby("poa_bin").agg({
        ac_power_col: "mean",
        pr_col: "mean"
    }).reset_index()
    clean_baseline.columns = ["poa_bin", "expected_clean_power", "expected_clean_pr"]
    
    # Merge baseline expectations back to full dataframe
    df = df.merge(clean_baseline, on="poa_bin", how="left")
    
    # For POA bins without clean reference data, use linear interpolation
    # based on neighboring bins
    if df["expected_clean_power"].isna().any():
        # Sort by POA bin for interpolation
        df = df.sort_values("poa_bin")
        df["expected_clean_power"] = df["expected_clean_power"].interpolate(method="linear")
        df["expected_clean_pr"] = df["expected_clean_pr"].interpolate(method="linear")
        
        # Fill any remaining NaNs with forward/backward fill
        df["expected_clean_power"] = df["expected_clean_power"].ffill().bfill()
        df["expected_clean_pr"] = df["expected_clean_pr"].ffill().bfill()
    
    # Drop the temporary poa_bin column
    df = df.drop(columns=["poa_bin"])
    
    return df


def calculate_fouling_index(
    df: pd.DataFrame,
    ac_power_col: str = "ac_power_kw",
    expected_power_col: str = "expected_clean_power",
    poa_col: str = "poa_w_m2",
    recent_window_days: int = 30
) -> float:
    """
    Calculate fouling index from ratio of actual to expected power.
    
    Fouling index = 1 - median(actual_power / expected_power) over recent window
    
    Args:
        df: Dataframe with actual and expected power columns
        ac_power_col: Column name for actual AC power
        expected_power_col: Column name for expected clean power
        poa_col: Column name for POA irradiance
        recent_window_days: Number of recent days to analyze (default 30)
    
    Returns:
        Fouling index between 0 and 1 (0 = clean, 1 = completely fouled)
    """
    # Filter to recent period and valid POA
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        latest_date = df["timestamp"].max()
        cutoff_date = latest_date - pd.Timedelta(days=recent_window_days)
        recent_df = df[df["timestamp"] >= cutoff_date]
    else:
        # If no timestamp, use last N rows
        rows_per_day = 96  # Assume 15-min intervals
        n_rows = recent_window_days * rows_per_day
        recent_df = df.tail(n_rows)
    
    # Filter for valid POA and non-missing values
    valid_data = recent_df[
        (recent_df[poa_col] >= MIN_POA_THRESHOLD) &
        (recent_df[ac_power_col].notna()) &
        (recent_df[expected_power_col].notna()) &
        (recent_df[expected_power_col] > 0)
    ]
    
    if len(valid_data) == 0:
        return 0.0  # No valid data, assume clean
    
    # Calculate actual/expected ratio
    power_ratio = valid_data[ac_power_col] / valid_data[expected_power_col]
    
    # Fouling index = 1 - median ratio (clipped to [0, 1])
    median_ratio = power_ratio.median()
    fouling_index = 1.0 - median_ratio
    fouling_index = np.clip(fouling_index, 0.0, 1.0)
    
    return float(fouling_index)


def classify_fouling_level(fouling_index: float) -> str:
    """
    Map fouling index to classification category.
    
    Args:
        fouling_index: Fouling index between 0 and 1
    
    Returns:
        Classification string: "Clean", "Light Soiling", "Moderate", or "Severe"
    """
    if fouling_index <= CLEAN_THRESHOLD:
        return "Clean"
    elif fouling_index <= LIGHT_SOILING_THRESHOLD:
        return "Light Soiling"
    elif fouling_index <= MODERATE_THRESHOLD:
        return "Moderate"
    else:
        return "Severe"


def estimate_energy_loss(
    df: pd.DataFrame,
    ac_power_col: str = "ac_power_kw",
    expected_power_col: str = "expected_clean_power",
    poa_col: str = "poa_w_m2",
    recent_window_days: int = 30
) -> float:
    """
    Estimate daily energy loss due to fouling.
    
    Args:
        df: Dataframe with actual and expected power
        ac_power_col: Column name for actual AC power
        expected_power_col: Column name for expected clean power
        poa_col: Column name for POA irradiance
        recent_window_days: Number of recent days to analyze
    
    Returns:
        Average daily energy loss in kWh
    """
    # Filter to recent period
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        latest_date = df["timestamp"].max()
        cutoff_date = latest_date - pd.Timedelta(days=recent_window_days)
        recent_df = df[df["timestamp"] >= cutoff_date]
    else:
        rows_per_day = 96
        n_rows = recent_window_days * rows_per_day
        recent_df = df.tail(n_rows)
    
    # Filter for valid data
    valid_data = recent_df[
        (recent_df[poa_col] >= MIN_POA_THRESHOLD) &
        (recent_df[ac_power_col].notna()) &
        (recent_df[expected_power_col].notna())
    ]
    
    if len(valid_data) == 0:
        return 0.0
    
    # Calculate energy difference
    # Energy = Power * Time (assuming regular intervals)
    power_loss = valid_data[expected_power_col] - valid_data[ac_power_col]
    power_loss = power_loss.clip(lower=0)  # Only count losses, not gains
    
    # Estimate time interval in hours
    if "timestamp" in valid_data.columns and len(valid_data) > 1:
        time_diff = valid_data["timestamp"].diff().median()
        if pd.notna(time_diff):
            interval_hours = time_diff.total_seconds() / 3600
        else:
            interval_hours = 0.25  # Default 15 minutes
    else:
        interval_hours = 0.25  # Default 15 minutes
    
    # Total energy loss in kWh
    total_energy_loss = (power_loss * interval_hours).sum()
    
    # Calculate actual number of days in the period
    if "timestamp" in valid_data.columns and len(valid_data) > 0:
        time_span = valid_data["timestamp"].max() - valid_data["timestamp"].min()
        actual_days = max(time_span.total_seconds() / (24 * 3600), 1)
    else:
        actual_days = recent_window_days
    
    # Average daily loss
    daily_loss = total_energy_loss / actual_days
    
    return float(daily_loss)


def detect_cleaning_events(
    df: pd.DataFrame,
    pr_col: str = "pr",
    poa_col: str = "poa_w_m2",
    improvement_threshold: float = 0.05,
    window_days: int = 3
) -> pd.DataFrame:
    """
    Detect sharp improvements in PR that likely indicate cleaning events.
    
    Args:
        df: Input dataframe with PR data
        pr_col: Column name for performance ratio
        poa_col: Column name for POA irradiance
        improvement_threshold: Minimum PR improvement to flag as cleaning (default 0.05)
        window_days: Window for before/after comparison
    
    Returns:
        DataFrame with added 'cleaning_event' boolean column
    """
    df = df.copy()
    
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
    
    # Estimate window size in rows
    if len(df) > 0 and "timestamp" in df.columns:
        time_diff = df["timestamp"].diff().median()
        if pd.notna(time_diff):
            rows_per_day = pd.Timedelta(days=1) / time_diff
            window_size = int(window_days * rows_per_day)
        else:
            window_size = window_days * 96
    else:
        window_size = window_days * 96
    
    # Filter for valid POA
    valid_poa = df[poa_col] >= MIN_POA_THRESHOLD
    
    # Calculate rolling PR before and after each point
    pr_before = df[pr_col].rolling(window=window_size, min_periods=1).mean()
    pr_after = df[pr_col].rolling(window=window_size, min_periods=1).mean().shift(-window_size)
    
    # Detect significant improvements
    pr_improvement = pr_after - pr_before
    cleaning_event = valid_poa & (pr_improvement >= improvement_threshold)
    
    df["cleaning_event"] = cleaning_event.fillna(False)
    
    return df


def run_fouling_analysis(
    df: pd.DataFrame,
    ac_power_col: str = "ac_power_kw",
    poa_col: str = "poa_w_m2",
    nameplate_dc_capacity: float = 1000.0,
    dc_power_col: Optional[str] = None,
    temp_col: Optional[str] = None,
    recent_window_days: int = 30
) -> Dict:
    """
    High-level orchestrator for fouling analysis.
    
    This function:
    1. Calculates PR
    2. Identifies clean reference periods
    3. Builds POA-matched clean baseline (core seasonality normalisation)
    4. Computes fouling index and classification
    5. Estimates energy loss
    
    Args:
        df: Input dataframe with required columns
        ac_power_col: Column name for AC power (kW)
        poa_col: Column name for POA irradiance (W/m²)
        nameplate_dc_capacity: Nameplate DC capacity in kW
        dc_power_col: Optional DC power column
        temp_col: Optional temperature column
        recent_window_days: Window for recent analysis
    
    Returns:
        Dictionary with analysis results
    """
    # Validate input data
    required_cols = [ac_power_col, poa_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Make a copy to avoid modifying original
    df_work = df.copy()
    
    # Step 1: Calculate PR
    df_work = calculate_pr(
        df_work,
        ac_power_col=ac_power_col,
        poa_col=poa_col,
        nameplate_dc_capacity=nameplate_dc_capacity,
        dc_power_col=dc_power_col
    )
    
    # Step 2: Identify clean reference periods
    df_work = identify_clean_reference_periods(df_work, pr_col="pr", poa_col=poa_col)
    
    # Step 3: Build POA-matched clean baseline
    clean_df = df_work[df_work["is_clean_reference"]]
    
    if len(clean_df) == 0:
        # No clean periods identified, use top quartile as reference
        pr_75th = df_work["pr"].quantile(0.75)
        clean_df = df_work[df_work["pr"] >= pr_75th].copy()
        # Mark these as clean reference periods for consistency
        clean_df["is_clean_reference"] = True
    
    df_work = estimate_clean_baseline_poa_matched(
        df_work,
        clean_df,
        poa_col=poa_col,
        ac_power_col=ac_power_col,
        pr_col="pr"
    )
    
    # Step 4: Calculate fouling index
    fouling_index = calculate_fouling_index(
        df_work,
        ac_power_col=ac_power_col,
        expected_power_col="expected_clean_power",
        poa_col=poa_col,
        recent_window_days=recent_window_days
    )
    
    # Step 5: Classify fouling level
    fouling_level = classify_fouling_level(fouling_index)
    
    # Step 6: Estimate energy loss
    energy_loss = estimate_energy_loss(
        df_work,
        ac_power_col=ac_power_col,
        expected_power_col="expected_clean_power",
        poa_col=poa_col,
        recent_window_days=recent_window_days
    )
    
    # Step 7: Detect cleaning events (optional)
    df_work = detect_cleaning_events(df_work, pr_col="pr", poa_col=poa_col)
    
    # Calculate summary statistics
    valid_pr = df_work[df_work["pr"].notna() & (df_work[poa_col] >= MIN_POA_THRESHOLD)]
    pr_mean = valid_pr["pr"].mean() if len(valid_pr) > 0 else 0.0
    pr_std = valid_pr["pr"].std() if len(valid_pr) > 0 else 0.0
    
    # Calculate analysis period
    if "timestamp" in df_work.columns and len(df_work) > 0:
        time_span = df_work["timestamp"].max() - df_work["timestamp"].min()
        analysis_days = time_span.total_seconds() / (24 * 3600)
    else:
        analysis_days = len(df_work) / 96  # Assume 15-min intervals
    
    # Compile results
    result = FoulingAnalysisResult(
        fouling_index=fouling_index,
        fouling_level=fouling_level,
        energy_loss_kwh_per_day=energy_loss,
        pr_mean=float(pr_mean),
        pr_std=float(pr_std),
        clean_periods_count=int(clean_df["is_clean_reference"].sum()),
        analysis_period_days=float(analysis_days)
    )
    
    # Return as dictionary
    return {
        "fouling_index": result.fouling_index,
        "fouling_level": result.fouling_level,
        "energy_loss_kwh_per_day": result.energy_loss_kwh_per_day,
        "pr_mean": result.pr_mean,
        "pr_std": result.pr_std,
        "clean_periods_count": result.clean_periods_count,
        "analysis_period_days": result.analysis_period_days,
        "dataframe": df_work  # Include processed dataframe for further analysis
    }


if __name__ == "__main__":
    # Example usage
    print("Fouling Detection Module - Example Usage")
    print("=" * 50)
    
    # Generate sample data for demonstration
    np.random.seed(42)
    
    # Create synthetic time series data
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="15min")
    n_points = len(dates)
    
    # Simulate POA with daily patterns
    hour_of_day = dates.hour + dates.minute / 60
    poa_base = np.maximum(0, 800 * np.sin((hour_of_day - 6) * np.pi / 12))
    
    # Add seasonal variation
    day_of_year = dates.dayofyear
    seasonal_factor = 0.8 + 0.4 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
    poa = poa_base * seasonal_factor + np.random.normal(0, 20, n_points)
    poa = np.maximum(0, poa)
    
    # Simulate clean performance initially, then gradual fouling
    clean_pr = 0.90
    fouling_factor = np.ones(n_points)
    
    # Apply gradual fouling after day 180
    fouling_start = n_points // 2
    fouling_progression = np.linspace(1.0, 0.85, n_points - fouling_start)
    fouling_factor[fouling_start:] = fouling_progression
    
    # Calculate AC power based on POA and fouling
    nameplate = 1000  # kW
    expected_dc = (poa / 1000) * nameplate
    ac_power = expected_dc * clean_pr * fouling_factor + np.random.normal(0, 5, n_points)
    ac_power = np.maximum(0, ac_power)
    
    # Create dataframe
    sample_df = pd.DataFrame({
        "timestamp": dates,
        "poa_w_m2": poa,
        "ac_power_kw": ac_power
    })
    
    print(f"\nGenerated sample data:")
    print(f"  - Date range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")
    print(f"  - Number of records: {len(sample_df)}")
    print(f"  - POA range: {sample_df['poa_w_m2'].min():.1f} - {sample_df['poa_w_m2'].max():.1f} W/m²")
    print(f"  - AC power range: {sample_df['ac_power_kw'].min():.1f} - {sample_df['ac_power_kw'].max():.1f} kW")
    
    # Run fouling analysis
    print("\nRunning fouling analysis...")
    results = run_fouling_analysis(
        sample_df,
        ac_power_col="ac_power_kw",
        poa_col="poa_w_m2",
        nameplate_dc_capacity=1000.0
    )
    
    # Display results
    print("\n" + "=" * 50)
    print("FOULING ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Fouling Index: {results['fouling_index']:.1%}")
    print(f"Fouling Level: {results['fouling_level']}")
    print(f"Energy Loss: {results['energy_loss_kwh_per_day']:.2f} kWh/day")
    print(f"Mean PR: {results['pr_mean']:.3f}")
    print(f"PR Std Dev: {results['pr_std']:.3f}")
    print(f"Clean Reference Periods: {results['clean_periods_count']}")
    print(f"Analysis Period: {results['analysis_period_days']:.1f} days")
    print("=" * 50)
    
    # Provide recommendations
    print("\nRECOMMENDATIONS:")
    if results['fouling_level'] == "Clean":
        print("  ✓ System is performing well. Continue monitoring.")
    elif results['fouling_level'] == "Light Soiling":
        print("  ⚠ Light soiling detected. Schedule cleaning within 2-4 weeks.")
    elif results['fouling_level'] == "Moderate":
        print("  ⚠ Moderate soiling detected. Schedule cleaning within 1-2 weeks.")
    else:  # Severe
        print("  ⚠⚠ Severe soiling detected. Schedule immediate cleaning to recover lost energy.")
    
    print(f"\n  Estimated monthly revenue loss: ${results['energy_loss_kwh_per_day'] * 30 * 0.10:.2f}")
    print(f"  (assuming $0.10/kWh electricity rate)")
