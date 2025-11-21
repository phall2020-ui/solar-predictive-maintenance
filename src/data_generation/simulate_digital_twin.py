"""Generate synthetic time-series data for solar assets (digital twin simulation)."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random

from src.config.settings import SimulationConfig, FeatureConfig, Config


def generate_weather_timeseries(
    start_date: str,
    days: int,
    granularity_minutes: int,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic weather data (irradiance and temperature).
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        days: Number of days to simulate
        granularity_minutes: Time step in minutes
        
    Returns:
        DataFrame with columns: timestamp, irradiance_w_m2, ambient_temp_c
    """
    np.random.seed(seed)
    random.seed(seed)
    
    start = pd.to_datetime(start_date)
    timestamps = pd.date_range(
        start=start,
        periods=int(days * 24 * 60 / granularity_minutes),
        freq=f"{granularity_minutes}min"
    )
    
    # Generate daily irradiance pattern (sine wave with noise)
    n_points = len(timestamps)
    hours_of_day = timestamps.hour + timestamps.minute / 60
    
    # Base irradiance: sine wave peaking at noon
    base_irradiance = np.maximum(0, np.sin((hours_of_day - 6) * np.pi / 12)) * 1000
    
    # Add seasonal variation (higher in summer)
    day_of_year = timestamps.dayofyear
    seasonal_factor = 0.7 + 0.3 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
    base_irradiance *= seasonal_factor
    
    # Add weather variability (clouds, etc.)
    weather_noise = np.random.normal(1.0, 0.15, n_points)
    weather_noise = np.clip(weather_noise, 0.1, 1.2)
    
    irradiance = base_irradiance * weather_noise
    irradiance = np.clip(irradiance, 0, 1200)  # Cap at realistic max
    
    # Temperature: base + daily cycle + seasonal
    base_temp = 20 + 10 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
    daily_cycle = 5 * np.sin((hours_of_day - 6) * np.pi / 12)
    temp_noise = np.random.normal(0, 2, n_points)
    
    ambient_temp = base_temp + daily_cycle + temp_noise
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "irradiance_w_m2": irradiance,
        "ambient_temp_c": ambient_temp,
    })


def calculate_ideal_power(
    irradiance: np.ndarray,
    nameplate_kw: float,
    temperature: np.ndarray,
    temp_coeff: float = -0.004
) -> np.ndarray:
    """
    Calculate ideal DC power from irradiance using simple physics model.
    
    Args:
        irradiance: Irradiance in W/m²
        nameplate_kw: Nameplate capacity in kW
        temperature: Module temperature in °C
        temp_coeff: Temperature coefficient (%/°C)
    
    Returns:
        Ideal DC power in kW
    """
    # Standard test conditions: 1000 W/m², 25°C
    stc_irradiance = 1000.0
    
    # Power proportional to irradiance
    power_ratio = irradiance / stc_irradiance
    
    # Temperature derating
    temp_derating = 1 + temp_coeff * (temperature - 25)
    
    ideal_dc_power = nameplate_kw * power_ratio * np.maximum(temp_derating, 0.5)
    
    return np.clip(ideal_dc_power, 0, nameplate_kw * 1.1)


def generate_portfolio_structure(
    num_sites: int,
    inverters_per_site: int,
    strings_per_inverter: int
) -> pd.DataFrame:
    """
    Generate portfolio structure (sites, inverters, strings).
    
    Returns:
        DataFrame with columns: site_id, inverter_id, string_id, nameplate_kw
    """
    portfolio = []
    
    for site_id in range(1, num_sites + 1):
        for inv_id in range(1, inverters_per_site + 1):
            inverter_id = f"site{site_id}_inv{inv_id}"
            # Vary nameplate capacity slightly (e.g., 100-120 kW)
            nameplate = np.random.uniform(100, 120)
            
            for str_id in range(1, strings_per_inverter + 1):
                string_id = f"{inverter_id}_str{str_id}"
                portfolio.append({
                    "site_id": f"site_{site_id}",
                    "inverter_id": inverter_id,
                    "string_id": string_id,
                    "nameplate_kw": nameplate / strings_per_inverter,  # Per string
                })
    
    return pd.DataFrame(portfolio)


def simulate_soiling_curve(
    timestamps: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    max_loss: float,
    cleaning_frequency_days: int,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate soiling loss over time with periodic cleaning events.
    
    Returns:
        Array of soiling loss factors (0 = no loss, 1 = max loss)
    """
    np.random.seed(seed)
    
    n_points = len(timestamps)
    # Convert to pandas Series if needed for .days access
    if isinstance(timestamps, np.ndarray):
        timestamps = pd.DatetimeIndex(timestamps)
    days_elapsed = (timestamps - start_date).days
    
    # Initialize soiling factor (0 = clean, 1 = max soiling)
    soiling_factor = np.zeros(n_points)
    
    # Simulate gradual accumulation and periodic cleaning
    current_soiling = 0.0
    next_cleaning = np.random.uniform(cleaning_frequency_days * 0.7, cleaning_frequency_days * 1.3)
    
    for i, days in enumerate(days_elapsed):
        if days >= next_cleaning:
            # Cleaning event: reset soiling
            current_soiling = 0.0
            next_cleaning = days + np.random.uniform(
                cleaning_frequency_days * 0.7,
                cleaning_frequency_days * 1.3
            )
        else:
            # Gradual accumulation (exponential approach to max)
            accumulation_rate = max_loss / (cleaning_frequency_days * 2)
            current_soiling = min(max_loss, current_soiling + accumulation_rate * np.random.uniform(0.5, 1.5))
        
        soiling_factor[i] = current_soiling
    
    return soiling_factor


def simulate_sensor_drift(
    timestamps: pd.DatetimeIndex,
    drift_start_idx: int,
    max_bias: float,
    drift_rate: float = 0.001
) -> np.ndarray:
    """
    Simulate sensor drift (gradual bias increase).
    
    Returns:
        Array of bias multipliers (1.0 = no bias, >1.0 = positive bias)
    """
    n_points = len(timestamps)
    bias = np.ones(n_points)
    
    if drift_start_idx < n_points:
        # Gradual drift after start
        drift_duration = n_points - drift_start_idx
        for i in range(drift_start_idx, n_points):
            days_drifted = (i - drift_start_idx) / (24 * 4)  # Assuming 15-min intervals
            bias[i] = 1.0 + min(max_bias, days_drifted * drift_rate)
    
    return bias


def simulate_inverter_failure(
    timestamps: pd.DatetimeIndex,
    failure_start_idx: int,
    failure_duration_hours: int,
    granularity_minutes: int
) -> np.ndarray:
    """
    Simulate inverter failure (output drops to near zero).
    
    Returns:
        Array of availability factors (1.0 = normal, 0.0 = failed)
    """
    n_points = len(timestamps)
    availability = np.ones(n_points)
    
    if failure_start_idx < n_points:
        failure_duration_points = int(failure_duration_hours * 60 / granularity_minutes)
        end_idx = min(n_points, failure_start_idx + failure_duration_points)
        availability[failure_start_idx:end_idx] = 0.0
    
    return availability


def generate_telemetry(
    config: SimulationConfig,
    portfolio: pd.DataFrame,
    weather: pd.DataFrame,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate complete telemetry dataset for all assets.
    
    Returns:
        Long-format DataFrame with telemetry data
    """
    np.random.seed(seed)
    random.seed(seed)
    
    timestamps = weather["timestamp"].values
    n_timestamps = len(timestamps)
    
    # Initialize lists for telemetry
    telemetry_rows = []
    
    # Pre-generate event schedules for each asset
    # Soiling: which sites have soiling issues
    sites_with_soiling = random.sample(
        portfolio["site_id"].unique().tolist(),
        k=int(len(portfolio["site_id"].unique()) * config.soiling_occurrence_rate)
    )
    
    # Sensor drift: which sensors drift
    num_sensors = len(portfolio["site_id"].unique())
    sensors_with_drift = random.sample(
        portfolio["site_id"].unique().tolist(),
        k=int(num_sensors * config.sensor_drift_rate)
    )
    
    # Inverter failures: schedule failures
    inverter_failures = {}  # inverter_id -> list of (start_idx, duration_hours)
    num_failures = int(len(portfolio["inverter_id"].unique()) * config.inverter_failure_rate)
    failed_inverters = random.sample(
        portfolio["inverter_id"].unique().tolist(),
        k=num_failures
    )
    
    for inv_id in failed_inverters:
        # Random failure start time
        failure_start_idx = random.randint(int(n_timestamps * 0.1), int(n_timestamps * 0.9))
        failure_duration = random.uniform(24, 720)  # 1 day to 30 days
        inverter_failures[inv_id] = (failure_start_idx, failure_duration)
    
    # String underperformance: which strings underperform
    num_underperforming = int(len(portfolio) * config.string_underperformance_rate)
    underperforming_strings = random.sample(
        portfolio["string_id"].tolist(),
        k=num_underperforming
    )
    underperformance_factor = {s: random.uniform(0.7, 0.85) for s in underperforming_strings}
    
    # Generate telemetry for each asset
    for _, asset in portfolio.iterrows():
        site_id = asset["site_id"]
        inverter_id = asset["inverter_id"]
        string_id = asset["string_id"]
        nameplate_kw = asset["nameplate_kw"]
        
        # Get weather for this site (add some spatial variation)
        site_weather = weather.copy()
        site_irradiance = site_weather["irradiance_w_m2"].values * np.random.uniform(0.95, 1.05)
        site_temp = site_weather["ambient_temp_c"].values + np.random.uniform(-2, 2)
        
        # Module temperature (higher than ambient due to solar heating)
        module_temp = site_temp + site_irradiance / 1000 * 30
        
        # Ideal DC power
        ideal_dc_power = calculate_ideal_power(
            site_irradiance,
            nameplate_kw,
            module_temp
        )
        
        # Apply soiling if this site has soiling issues
        if site_id in sites_with_soiling:
            soiling_factor = simulate_soiling_curve(
                timestamps,
                pd.to_datetime(config.start_date),
                config.soiling_max_loss,
                config.cleaning_frequency_days,
                seed=seed + hash(site_id) % 1000
            )
            soiling_loss = 1 - soiling_factor
        else:
            soiling_loss = np.ones(n_timestamps)
        
        # Apply sensor drift if this site's sensor drifts
        if site_id in sensors_with_drift:
            drift_start = random.randint(int(n_timestamps * 0.2), int(n_timestamps * 0.8))
            sensor_bias = simulate_sensor_drift(
                timestamps,
                drift_start,
                config.drift_max_bias
            )
            # Bias affects irradiance reading (but not actual generation)
            measured_irradiance = site_irradiance * sensor_bias
        else:
            measured_irradiance = site_irradiance
            sensor_bias = np.ones(n_timestamps)
        
        # Apply inverter failure
        if inverter_id in inverter_failures:
            failure_start_idx, failure_duration = inverter_failures[inverter_id]
            availability = simulate_inverter_failure(
                timestamps,
                failure_start_idx,
                failure_duration,
                config.granularity_minutes
            )
        else:
            availability = np.ones(n_timestamps)
        
        # Apply string underperformance
        if string_id in underperforming_strings:
            perf_factor = underperformance_factor[string_id]
        else:
            perf_factor = 1.0
        
        # Calculate actual DC power with all effects
        actual_dc_power = (
            ideal_dc_power
            * soiling_loss
            * availability
            * perf_factor
            * np.random.uniform(0.98, 1.02, n_timestamps)  # Small random noise
        )
        
        # AC power (inverter efficiency ~97%)
        inverter_efficiency = 0.97
        ac_power = actual_dc_power * inverter_efficiency
        
        # Energy yield (kWh) - integrate power over time interval
        energy_kwh = ac_power * (config.granularity_minutes / 60)
        
        # Inverter status
        inverter_status = np.where(availability > 0.5, "OK", "FAULT")
        
        # Build rows
        for i in range(n_timestamps):
            telemetry_rows.append({
                "timestamp": timestamps[i],
                "site_id": site_id,
                "inverter_id": inverter_id,
                "string_id": string_id,
                "irradiance_w_m2": measured_irradiance[i],
                "ambient_temp_c": site_temp[i],
                "module_temp_c": module_temp[i],
                "dc_power_kw": actual_dc_power[i],
                "ac_power_kw": ac_power[i],
                "energy_kwh": energy_kwh[i],
                "inverter_status": inverter_status[i],
                "soiling_loss_factor": 1 - soiling_loss[i] if site_id in sites_with_soiling else 0.0,
                "sensor_bias_factor": sensor_bias[i] - 1.0 if site_id in sensors_with_drift else 0.0,
                "availability_factor": availability[i],
                "underperformance_factor": 1.0 - perf_factor if string_id in underperforming_strings else 0.0,
            })
    
    return pd.DataFrame(telemetry_rows)


def generate_labels(
    telemetry: pd.DataFrame,
    config: SimulationConfig,
    feature_config: FeatureConfig
) -> pd.DataFrame:
    """
    Generate labels for all prediction tasks from telemetry.
    
    Returns:
        DataFrame with label columns for each task
    """
    labels = telemetry[["timestamp", "site_id", "inverter_id", "string_id"]].copy()
    
    # Inverter failure label (1 if failure in next N hours)
    horizon_points = int(feature_config.failure_horizon_hours * 60 / config.granularity_minutes)
    
    labels["inverter_failure_label"] = 0
    for inv_id in telemetry["inverter_id"].unique():
        inv_mask = telemetry["inverter_id"] == inv_id
        inv_data = telemetry[inv_mask].sort_values("timestamp")
        
        # Find failure periods
        failure_mask = inv_data["availability_factor"] < 0.5
        failure_indices = inv_data[failure_mask].index
        
        # Label points before failures
        for fail_idx in failure_indices:
            fail_pos = inv_data.index.get_loc(fail_idx)
            label_start = max(0, fail_pos - horizon_points)
            label_end = fail_pos
            label_indices = inv_data.iloc[label_start:label_end].index
            labels.loc[label_indices, "inverter_failure_label"] = 1
    
    # String underperformance label
    labels["string_underperformance_label"] = (
        telemetry["underperformance_factor"] > 0.1
    ).astype(int)
    
    # Sensor drift label
    labels["sensor_drift_label"] = (
        np.abs(telemetry["sensor_bias_factor"]) > 0.02
    ).astype(int)
    
    # Soiling loss (regression target)
    labels["soiling_loss_kwh"] = (
        telemetry["soiling_loss_factor"] * telemetry["energy_kwh"]
    )
    
    return labels


def simulate_digital_twin(config: Config, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to generate complete synthetic dataset.
    
    Args:
        config: Full configuration object
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (telemetry_df, labels_df)
    """
    # Generate portfolio structure
    portfolio = generate_portfolio_structure(
        config.simulation.num_sites,
        config.simulation.inverters_per_site,
        config.simulation.strings_per_inverter
    )
    
    # Generate weather data
    weather = generate_weather_timeseries(
        config.simulation.start_date,
        config.simulation.days,
        config.simulation.granularity_minutes,
        seed=seed
    )
    
    # Generate telemetry
    telemetry = generate_telemetry(config.simulation, portfolio, weather, seed=seed)
    
    # Generate labels
    labels = generate_labels(telemetry, config.simulation, config.features)
    
    return telemetry, labels

