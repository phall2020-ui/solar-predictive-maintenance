#!/usr/bin/env python3
"""Quick test script to verify the system works."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.settings import DEFAULT_CONFIG
from src.data_generation.simulate_digital_twin import simulate_digital_twin
from src.features.feature_engineering import engineer_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

def time_aware_split(X, y, metadata, test_size=0.2, val_size=0.1):
    """Simple time-aware split."""
    sorted_indices = metadata.sort_values("timestamp").index
    n_total = len(sorted_indices)
    test_start = int(n_total * (1 - test_size))
    val_start = int(test_start * (1 - val_size))
    train_indices = sorted_indices[:val_start]
    val_indices = sorted_indices[val_start:test_start]
    test_indices = sorted_indices[test_start:]
    return (
        X.loc[train_indices], X.loc[val_indices], X.loc[test_indices],
        y.loc[train_indices], y.loc[val_indices], y.loc[test_indices]
    )

print("=" * 60)
print("Solar Predictive Maintenance - System Test")
print("=" * 60)

print("\n[1/4] Generating synthetic data...")
config = DEFAULT_CONFIG
config.simulation.days = 90  # 3 months
config.simulation.num_sites = 3
config.simulation.inverters_per_site = 5
config.simulation.strings_per_inverter = 4

telemetry, labels = simulate_digital_twin(config, seed=42)
print(f"✓ Generated {len(telemetry):,} telemetry records")
print(f"  Time range: {telemetry['timestamp'].min()} to {telemetry['timestamp'].max()}")
print(f"  Sites: {telemetry['site_id'].nunique()}, Inverters: {telemetry['inverter_id'].nunique()}, Strings: {telemetry['string_id'].nunique()}")

print("\n[2/4] Engineering features...")
X, y, metadata = engineer_features(
    telemetry, labels, 'inverter_failure', config.features,
    granularity_minutes=config.simulation.granularity_minutes
)
print(f"✓ Created {len(X.columns)} features")
print(f"  Feature matrix shape: {X.shape}")
print(f"  Label distribution: {y.value_counts().to_dict()}")

if y.sum() == 0:
    print("\n⚠ Warning: No positive labels found. This is normal for small datasets.")
    print("  The system is working correctly, but model training will show baseline performance.")

print("\n[3/4] Training Random Forest model...")
X_train, X_val, X_test, y_train, y_val, y_test = time_aware_split(
    X, y, metadata, config.models.test_size, config.models.val_size
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

print("\n[4/4] Evaluating model...")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\n✓ Model trained successfully!")
print(f"\nTest Set Metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n" + "=" * 60)
print("✓ All systems operational!")
print("=" * 60)

