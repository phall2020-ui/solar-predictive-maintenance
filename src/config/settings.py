"""Configuration settings for the solar predictive maintenance system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class SimulationConfig:
    """Configuration for data simulation."""
    
    # Time parameters
    start_date: str = "2020-01-01"
    days: int = 365 * 3  # 3 years
    granularity_minutes: int = 15
    
    # Portfolio structure
    num_sites: int = 5
    inverters_per_site: int = 10
    strings_per_inverter: int = 4
    
    # Event frequencies (events per year)
    inverter_failure_rate: float = 0.1  # 10% of inverters fail per year
    string_underperformance_rate: float = 0.05  # 5% of strings underperform
    sensor_drift_rate: float = 0.02  # 2% of sensors drift per year
    soiling_occurrence_rate: float = 0.3  # 30% of sites have soiling issues
    
    # Soiling parameters
    soiling_max_loss: float = 0.15  # Max 15% performance loss
    cleaning_frequency_days: int = 90  # Cleaning every 90 days on average
    
    # Sensor drift parameters
    drift_max_bias: float = 0.10  # Max 10% bias in sensor readings
    
    # Output paths
    output_dir: Path = Path("data")
    telemetry_file: str = "telemetry.csv"
    labels_file: str = "labels.csv"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Rolling window sizes (in hours)
    rolling_windows: list[int] = None
    
    # Lag features (in hours)
    lag_hours: list[int] = None
    
    # Prediction horizons (in hours)
    failure_horizon_hours: int = 24
    underperformance_horizon_hours: int = 168  # 1 week
    sensor_drift_horizon_hours: int = 720  # 30 days
    soiling_horizon_hours: int = 168  # 1 week
    
    # Feature window (how much history to use)
    feature_window_hours: int = 24
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [1, 6, 24, 168]  # 1h, 6h, 1d, 1w
        if self.lag_hours is None:
            self.lag_hours = [1, 6, 12, 24]


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    # Train/test split
    test_size: float = 0.2
    val_size: float = 0.1
    
    # Time-aware split (use date cutoff instead of random)
    use_time_split: bool = True
    split_date: Optional[str] = None  # If None, use test_size
    
    # Model hyperparameters
    random_state: int = 42
    
    # XGBoost defaults
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # Random Forest defaults
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    
    # Output paths
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")


@dataclass
class Config:
    """Main configuration class."""
    
    simulation: SimulationConfig
    features: FeatureConfig
    models: ModelConfig
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls(
            simulation=SimulationConfig(**data.get("simulation", {})),
            features=FeatureConfig(**data.get("features", {})),
            models=ModelConfig(**data.get("models", {})),
        )
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        data = {
            "simulation": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.simulation.__dict__.items()
            },
            "features": self.features.__dict__,
            "models": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in self.models.__dict__.items()
            },
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


# Default configuration
DEFAULT_CONFIG = Config(
    simulation=SimulationConfig(),
    features=FeatureConfig(),
    models=ModelConfig(),
)

