"""Tests for model training."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.config.settings import ModelConfig
from src.models.training import (
    train_inverter_failure_model,
    train_soiling_model,
    save_model,
    load_model,
)


@pytest.fixture
def sample_features():
    """Create sample feature matrix."""
    n_samples = 200
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    return X


@pytest.fixture
def sample_labels_classification(sample_features):
    """Create binary classification labels."""
    # Create imbalanced labels
    y = pd.Series(np.random.choice([0, 1], size=len(sample_features), p=[0.9, 0.1]))
    return y


@pytest.fixture
def sample_labels_regression(sample_features):
    """Create regression labels."""
    y = pd.Series(np.random.uniform(0, 10, size=len(sample_features)))
    return y


@pytest.fixture
def sample_metadata(sample_features):
    """Create sample metadata."""
    dates = pd.date_range("2020-01-01", periods=len(sample_features), freq="15min")
    return pd.DataFrame({
        "timestamp": dates,
        "site_id": ["site_1"] * len(sample_features),
        "inverter_id": ["inv_1"] * len(sample_features),
        "string_id": ["str_1"] * len(sample_features),
    })


@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(
        test_size=0.2,
        val_size=0.1,
        random_state=42,
    )


def test_train_inverter_failure_model(
    sample_features, sample_labels_classification, sample_metadata, model_config
):
    """Test inverter failure model training."""
    model_dict = train_inverter_failure_model(
        sample_features,
        sample_labels_classification,
        sample_metadata,
        model_config,
        model_type="random_forest"
    )
    
    assert "model" in model_dict
    assert "scaler" in model_dict
    assert "train_metrics" in model_dict
    assert "test_metrics" in model_dict
    
    # Check metrics exist
    assert "accuracy" in model_dict["test_metrics"]
    assert "f1" in model_dict["test_metrics"]


def test_train_soiling_model(
    sample_features, sample_labels_regression, sample_metadata, model_config
):
    """Test soiling model training."""
    model_dict = train_soiling_model(
        sample_features,
        sample_labels_regression,
        sample_metadata,
        model_config,
        model_type="random_forest"
    )
    
    assert "model" in model_dict
    assert "scaler" in model_dict
    assert "test_metrics" in model_dict
    
    # Check regression metrics
    assert "mae" in model_dict["test_metrics"]
    assert "rmse" in model_dict["test_metrics"]
    assert "r2" in model_dict["test_metrics"]


def test_save_load_model(
    sample_features, sample_labels_classification, sample_metadata, model_config, tmp_path
):
    """Test model saving and loading."""
    model_dict = train_inverter_failure_model(
        sample_features,
        sample_labels_classification,
        sample_metadata,
        model_config,
        model_type="random_forest"
    )
    
    model_path = tmp_path / "test_model.pkl"
    save_model(model_dict, model_path)
    
    assert model_path.exists()
    
    # Load model
    loaded_dict = load_model(model_path)
    
    assert "model" in loaded_dict
    assert "scaler" in loaded_dict
    assert loaded_dict["task"] == "inverter_failure"

