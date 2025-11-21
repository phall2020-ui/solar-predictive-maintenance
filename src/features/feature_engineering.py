"""Feature engineering for solar asset predictive maintenance."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from src.config.settings import FeatureConfig


def create_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Create time-based features from timestamp.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
    
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    timestamps = pd.to_datetime(df[timestamp_col])
    
    df["hour"] = timestamps.dt.hour
    df["day_of_week"] = timestamps.dt.dayofweek
    df["day_of_year"] = timestamps.dt.dayofyear
    df["month"] = timestamps.dt.month
    df["is_weekend"] = (timestamps.dt.dayofweek >= 5).astype(int)
    
    # Cyclical encoding for hour and day_of_year
    df["hour_sin"] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
    df["day_of_year_sin"] = np.sin(2 * np.pi * timestamps.dt.dayofyear / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * timestamps.dt.dayofyear / 365)
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    value_cols: List[str],
    windows_hours: List[int],
    group_cols: List[str],
    timestamp_col: str = "timestamp",
    granularity_minutes: int = 15
) -> pd.DataFrame:
    """
    Create rolling window features (mean, std, min, max) for specified columns.
    
    Args:
        df: Input DataFrame
        value_cols: Columns to compute rolling features for
        windows_hours: List of window sizes in hours
        group_cols: Columns to group by (e.g., ['inverter_id'])
        timestamp_col: Name of timestamp column
        granularity_minutes: Time granularity in minutes
    
    Returns:
        DataFrame with added rolling features
    """
    df = df.copy()
    df = df.sort_values(group_cols + [timestamp_col])
    
    for col in value_cols:
        for window_hours in windows_hours:
            window_size = int(window_hours * 60 / granularity_minutes)
            
            for group_key, group_df in df.groupby(group_cols):
                group_indices = group_df.index
                
                # Rolling statistics
                rolling_mean = group_df[col].rolling(window=window_size, min_periods=1).mean()
                rolling_std = group_df[col].rolling(window=window_size, min_periods=1).std()
                rolling_min = group_df[col].rolling(window=window_size, min_periods=1).min()
                rolling_max = group_df[col].rolling(window=window_size, min_periods=1).max()
                
                df.loc[group_indices, f"{col}_rolling_mean_{window_hours}h"] = rolling_mean.values
                df.loc[group_indices, f"{col}_rolling_std_{window_hours}h"] = rolling_std.values
                df.loc[group_indices, f"{col}_rolling_min_{window_hours}h"] = rolling_min.values
                df.loc[group_indices, f"{col}_rolling_max_{window_hours}h"] = rolling_max.values
    
    return df


def create_lag_features(
    df: pd.DataFrame,
    value_cols: List[str],
    lag_hours: List[int],
    group_cols: List[str],
    timestamp_col: str = "timestamp",
    granularity_minutes: int = 15
) -> pd.DataFrame:
    """
    Create lag features (values from previous time steps).
    
    Args:
        df: Input DataFrame
        value_cols: Columns to create lag features for
        lag_hours: List of lag periods in hours
        group_cols: Columns to group by
        timestamp_col: Name of timestamp column
        granularity_minutes: Time granularity in minutes
    
    Returns:
        DataFrame with added lag features
    """
    df = df.copy()
    df = df.sort_values(group_cols + [timestamp_col])
    
    for col in value_cols:
        for lag_hour in lag_hours:
            lag_size = int(lag_hour * 60 / granularity_minutes)
            
            for group_key, group_df in df.groupby(group_cols):
                group_indices = group_df.index
                lagged_values = group_df[col].shift(lag_size)
                df.loc[group_indices, f"{col}_lag_{lag_hour}h"] = lagged_values.values
    
    return df


def calculate_performance_ratio(
    df: pd.DataFrame,
    energy_col: str = "energy_kwh",
    irradiance_col: str = "irradiance_w_m2",
    nameplate_col: Optional[str] = None
) -> pd.Series:
    """
    Calculate Performance Ratio (PR).
    
    PR = (Actual Energy) / (Expected Energy from Irradiance)
    
    Args:
        df: Input DataFrame
        energy_col: Column name for actual energy
        irradiance_col: Column name for irradiance
        nameplate_col: Optional column name for nameplate capacity
    
    Returns:
        Series with PR values
    """
    # Expected energy from irradiance (simplified model)
    # Assuming STC: 1000 W/m², 25°C
    stc_irradiance = 1000.0
    
    if nameplate_col:
        expected_energy = df[nameplate_col] * (df[irradiance_col] / stc_irradiance)
    else:
        # Use actual energy as proxy (will normalize)
        expected_energy = df[energy_col] * (df[irradiance_col] / stc_irradiance)
    
    # Avoid division by zero
    pr = df[energy_col] / (expected_energy + 1e-6)
    
    # Cap PR at reasonable values
    pr = np.clip(pr, 0, 2.0)
    
    return pr


def create_peer_comparison_features(
    df: pd.DataFrame,
    value_cols: List[str],
    peer_group_col: str = "inverter_id",
    entity_col: str = "string_id"
) -> pd.DataFrame:
    """
    Create features comparing entity performance to peers.
    
    Args:
        df: Input DataFrame
        value_cols: Columns to compare
        peer_group_col: Column defining peer group (e.g., 'inverter_id')
        entity_col: Column defining entity (e.g., 'string_id')
    
    Returns:
        DataFrame with added peer comparison features
    """
    df = df.copy()
    
    for col in value_cols:
        # Calculate peer statistics
        peer_stats = df.groupby([peer_group_col, "timestamp"])[col].agg([
            "mean", "median", "std", "min", "max"
        ]).reset_index()
        
        peer_stats.columns = [peer_group_col, "timestamp", 
                             f"{col}_peer_mean", f"{col}_peer_median",
                             f"{col}_peer_std", f"{col}_peer_min", f"{col}_peer_max"]
        
        # Merge back
        df = df.merge(peer_stats, on=[peer_group_col, "timestamp"], how="left")
        
        # Calculate relative performance
        df[f"{col}_vs_peer_mean"] = df[col] / (df[f"{col}_peer_mean"] + 1e-6)
        df[f"{col}_vs_peer_median"] = df[col] / (df[f"{col}_peer_median"] + 1e-6)
        df[f"{col}_percentile_in_peer"] = df.groupby([peer_group_col, "timestamp"])[col].transform(
            lambda x: pd.qcut(x, q=10, labels=False, duplicates="drop")
        ) / 10.0
    
    return df


def create_inverter_failure_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    granularity_minutes: int = 15
) -> pd.DataFrame:
    """
    Create features specific to inverter failure prediction.
    
    Args:
        df: Input DataFrame with telemetry
        config: Feature configuration
        granularity_minutes: Time granularity
    
    Returns:
        DataFrame with added failure prediction features
    """
    df = df.copy()
    
    # Group by inverter
    group_cols = ["inverter_id"]
    
    # Power-related features
    power_cols = ["ac_power_kw", "dc_power_kw", "energy_kwh"]
    df = create_rolling_features(
        df, power_cols, config.rolling_windows, group_cols, granularity_minutes=granularity_minutes
    )
    df = create_lag_features(
        df, power_cols, config.lag_hours, group_cols, granularity_minutes=granularity_minutes
    )
    
    # Temperature features
    temp_cols = ["module_temp_c", "ambient_temp_c"]
    df = create_rolling_features(
        df, temp_cols, config.rolling_windows, group_cols, granularity_minutes=granularity_minutes
    )
    
    # Status flag features
    df["status_fault"] = (df["inverter_status"] == "FAULT").astype(int)
    df["status_ok"] = (df["inverter_status"] == "OK").astype(int)
    
    # Count faults in rolling window
    window_size = int(config.feature_window_hours * 60 / granularity_minutes)
    for group_key, group_df in df.groupby(group_cols):
        group_indices = group_df.index
        fault_count = group_df["status_fault"].rolling(window=window_size, min_periods=1).sum()
        df.loc[group_indices, "fault_count_recent"] = fault_count.values
    
    # Power instability (coefficient of variation)
    for window_hours in config.rolling_windows:
        window_size = int(window_hours * 60 / granularity_minutes)
        for group_key, group_df in df.groupby(group_cols):
            group_indices = group_df.index
            rolling_mean = group_df["ac_power_kw"].rolling(window=window_size, min_periods=1).mean()
            rolling_std = group_df["ac_power_kw"].rolling(window=window_size, min_periods=1).std()
            cv = rolling_std / (rolling_mean + 1e-6)
            df.loc[group_indices, f"power_cv_{window_hours}h"] = cv.values
    
    return df


def create_string_underperformance_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    granularity_minutes: int = 15
) -> pd.DataFrame:
    """
    Create features specific to string underperformance prediction.
    
    Args:
        df: Input DataFrame
        config: Feature configuration
        granularity_minutes: Time granularity
    
    Returns:
        DataFrame with added underperformance features
    """
    df = df.copy()
    
    # Peer comparison features (string vs other strings on same inverter)
    df = create_peer_comparison_features(
        df,
        value_cols=["energy_kwh", "dc_power_kw"],
        peer_group_col="inverter_id",
        entity_col="string_id"
    )
    
    # Rolling features for string performance
    group_cols = ["string_id"]
    perf_cols = ["energy_kwh", "dc_power_kw"]
    df = create_rolling_features(
        df, perf_cols, config.rolling_windows, group_cols, granularity_minutes=granularity_minutes
    )
    
    # Long-term degradation signal (trend over longer window)
    long_window = max(config.rolling_windows) * 2  # 2x longest window
    for group_key, group_df in df.groupby(group_cols):
        group_indices = group_df.index
        # Simple linear trend
        x = np.arange(len(group_df))
        y = group_df["energy_kwh"].values
        if len(y) > 10:
            # Calculate rolling slope
            window_size = int(long_window * 60 / granularity_minutes)
            slopes = []
            for i in range(len(y)):
                start_idx = max(0, i - window_size)
                window_y = y[start_idx:i+1]
                window_x = x[start_idx:i+1]
                if len(window_y) > 1:
                    slope = np.polyfit(window_x, window_y, 1)[0] if len(window_y) > 1 else 0
                else:
                    slope = 0
                slopes.append(slope)
            df.loc[group_indices, "energy_trend"] = slopes
        else:
            df.loc[group_indices, "energy_trend"] = 0
    
    return df


def create_sensor_drift_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    granularity_minutes: int = 15
) -> pd.DataFrame:
    """
    Create features specific to sensor drift detection.
    
    Args:
        df: Input DataFrame
        config: Feature configuration
        granularity_minutes: Time granularity
    
    Returns:
        DataFrame with added sensor drift features
    """
    df = df.copy()
    
    # Calculate expected power from physics model
    # Expected DC power = f(irradiance, temperature, nameplate)
    # Use simplified model
    df["expected_dc_power"] = (
        df["irradiance_w_m2"] / 1000.0 * df.get("nameplate_kw", 100)
    )
    
    # Mismatch between expected and actual
    df["power_mismatch"] = df["dc_power_kw"] - df["expected_dc_power"]
    df["power_mismatch_ratio"] = df["dc_power_kw"] / (df["expected_dc_power"] + 1e-6)
    
    # Peer-based expected power (compare to other strings on same inverter)
    df = create_peer_comparison_features(
        df,
        value_cols=["irradiance_w_m2"],
        peer_group_col="inverter_id",
        entity_col="string_id"
    )
    
    # Irradiance deviation from peers
    df["irradiance_vs_peer"] = (
        df["irradiance_w_m2"] / (df["irradiance_w_m2_peer_median"] + 1e-6)
    )
    
    # Rolling features for mismatch
    group_cols = ["string_id"]
    mismatch_cols = ["power_mismatch", "power_mismatch_ratio", "irradiance_vs_peer"]
    df = create_rolling_features(
        df, mismatch_cols, config.rolling_windows, group_cols, granularity_minutes=granularity_minutes
    )
    
    # Trend in mismatch (increasing mismatch suggests drift)
    for group_key, group_df in df.groupby(group_cols):
        group_indices = group_df.index
        x = np.arange(len(group_df))
        y = group_df["power_mismatch_ratio"].values
        if len(y) > 10:
            window_size = int(config.feature_window_hours * 60 / granularity_minutes)
            trends = []
            for i in range(len(y)):
                start_idx = max(0, i - window_size)
                window_y = y[start_idx:i+1]
                window_x = x[start_idx:i+1]
                if len(window_y) > 1:
                    slope = np.polyfit(window_x, window_y, 1)[0]
                else:
                    slope = 0
                trends.append(slope)
            df.loc[group_indices, "mismatch_trend"] = trends
        else:
            df.loc[group_indices, "mismatch_trend"] = 0
    
    return df


def create_soiling_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    granularity_minutes: int = 15
) -> pd.DataFrame:
    """
    Create features specific to soiling loss prediction.
    
    Args:
        df: Input DataFrame
        config: Feature configuration
        granularity_minutes: Time granularity
    
    Returns:
        DataFrame with added soiling features
    """
    df = df.copy()
    
    # Calculate Performance Ratio
    df["performance_ratio"] = calculate_performance_ratio(
        df, energy_col="energy_kwh", irradiance_col="irradiance_w_m2"
    )
    
    # Rolling PR features
    group_cols = ["string_id"]
    pr_cols = ["performance_ratio"]
    df = create_rolling_features(
        df, pr_cols, config.rolling_windows, group_cols, granularity_minutes=granularity_minutes
    )
    
    # Long-term PR trend (degradation/soiling)
    long_window = max(config.rolling_windows) * 3
    for group_key, group_df in df.groupby(group_cols):
        group_indices = group_df.index
        x = np.arange(len(group_df))
        y = group_df["performance_ratio"].values
        if len(y) > 10:
            window_size = int(long_window * 60 / granularity_minutes)
            trends = []
            for i in range(len(y)):
                start_idx = max(0, i - window_size)
                window_y = y[start_idx:i+1]
                window_x = x[start_idx:i+1]
                if len(window_y) > 1:
                    slope = np.polyfit(window_x, window_y, 1)[0]
                else:
                    slope = 0
                trends.append(slope)
            df.loc[group_indices, "pr_trend"] = trends
        else:
            df.loc[group_indices, "pr_trend"] = 0
    
    # Days since last cleaning (simplified: use soiling_loss_factor)
    # If soiling_loss_factor is low, assume recent cleaning
    df["days_since_cleaning_est"] = df["soiling_loss_factor"] * 90  # Rough estimate
    
    # Expected vs actual yield
    df["yield_ratio"] = df["energy_kwh"] / (df["irradiance_w_m2"] / 1000.0 + 1e-6)
    
    return df


def engineer_features(
    telemetry: pd.DataFrame,
    labels: pd.DataFrame,
    task: str,
    config: FeatureConfig,
    granularity_minutes: int = 15
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Main feature engineering function for a specific prediction task.
    
    Args:
        telemetry: Telemetry DataFrame
        labels: Labels DataFrame
        task: Task name ('inverter_failure', 'string_underperformance', 'sensor_drift', 'soiling')
        config: Feature configuration
        granularity_minutes: Time granularity
    
    Returns:
        Tuple of (feature_df, labels_series, metadata_df)
    """
    # Merge telemetry and labels
    df = telemetry.merge(
        labels,
        on=["timestamp", "site_id", "inverter_id", "string_id"],
        how="inner"
    )
    
    # Create time features
    df = create_time_features(df)
    
    # Task-specific feature engineering
    if task == "inverter_failure":
        df = create_inverter_failure_features(df, config, granularity_minutes)
        label_col = "inverter_failure_label"
    elif task == "string_underperformance":
        df = create_string_underperformance_features(df, config, granularity_minutes)
        label_col = "string_underperformance_label"
    elif task == "sensor_drift":
        df = create_sensor_drift_features(df, config, granularity_minutes)
        label_col = "sensor_drift_label"
    elif task == "soiling":
        df = create_soiling_features(df, config, granularity_minutes)
        label_col = "soiling_loss_kwh"
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Select feature columns (exclude metadata and labels)
    exclude_cols = [
        "timestamp", "site_id", "inverter_id", "string_id",
        "inverter_status", "inverter_failure_label",
        "string_underperformance_label", "sensor_drift_label",
        "soiling_loss_kwh", "soiling_loss_factor", "sensor_bias_factor",
        "availability_factor", "underperformance_factor"
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Extract features and labels
    X = df[feature_cols].copy()
    y = df[label_col].copy()
    metadata = df[["timestamp", "site_id", "inverter_id", "string_id"]].copy()
    
    # Handle NaN values (fill with 0 or median)
    X = X.fillna(0)
    
    # Replace inf values
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, y, metadata

