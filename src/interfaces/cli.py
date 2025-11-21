"""Command-line interface for solar predictive maintenance."""

import typer
from pathlib import Path
from typing import Optional
import pandas as pd

from src.config.settings import Config, DEFAULT_CONFIG
from src.data_generation.simulate_digital_twin import simulate_digital_twin
from src.features.feature_engineering import engineer_features
from src.models.training import (
    train_inverter_failure_model,
    train_string_underperformance_model,
    train_sensor_drift_model,
    train_soiling_model,
    save_model,
    time_aware_split,
)
from src.models.inference import predict_from_file, predict
from src.models.evaluation import generate_evaluation_report

app = typer.Typer(help="Solar Predictive Maintenance CLI")


@app.command()
def generate_data(
    output_dir: str = typer.Option("data", help="Output directory for generated data"),
    days: int = typer.Option(365, help="Number of days to simulate"),
    sites: int = typer.Option(5, help="Number of sites"),
    granularity_mins: int = typer.Option(15, help="Time granularity in minutes"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Generate synthetic solar asset telemetry data."""
    typer.echo(f"Generating data for {days} days, {sites} sites...")
    
    # Update config
    config = DEFAULT_CONFIG
    config.simulation.days = days
    config.simulation.num_sites = sites
    config.simulation.granularity_minutes = granularity_mins
    config.simulation.output_dir = Path(output_dir)
    
    # Generate data
    telemetry, labels = simulate_digital_twin(config, seed=seed)
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    telemetry_path = output_path / config.simulation.telemetry_file
    labels_path = output_path / config.simulation.labels_file
    
    telemetry.to_csv(telemetry_path, index=False)
    labels.to_csv(labels_path, index=False)
    
    typer.echo(f"✓ Telemetry saved to {telemetry_path}")
    typer.echo(f"✓ Labels saved to {labels_path}")
    typer.echo(f"  Generated {len(telemetry)} telemetry records")


@app.command()
def train(
    task: str = typer.Option(..., help="Task: inverter_failure, string_underperformance, sensor_drift, or soiling"),
    data_dir: str = typer.Option("data", help="Directory with telemetry and labels CSV files"),
    model_output: str = typer.Option("models", help="Output directory for trained models"),
    model_type: str = typer.Option("xgboost", help="Model type: xgboost, random_forest, or logistic"),
    seed: int = typer.Option(42, help="Random seed"),
):
    """Train a predictive model for a specific task."""
    typer.echo(f"Training {task} model...")
    
    # Load data
    data_path = Path(data_dir)
    telemetry = pd.read_csv(data_path / "telemetry.csv")
    labels = pd.read_csv(data_path / "labels.csv")
    
    typer.echo(f"  Loaded {len(telemetry)} telemetry records")
    
    # Engineer features
    typer.echo("  Engineering features...")
    config = DEFAULT_CONFIG
    X, y, metadata = engineer_features(
        telemetry, labels, task, config.features,
        granularity_minutes=config.simulation.granularity_minutes
    )
    
    typer.echo(f"  Created {len(X.columns)} features")
    
    # Train model
    typer.echo("  Training model...")
    if task == "inverter_failure":
        model_dict = train_inverter_failure_model(X, y, metadata, config.models, model_type)
    elif task == "string_underperformance":
        model_dict = train_string_underperformance_model(X, y, metadata, config.models, model_type)
    elif task == "sensor_drift":
        model_dict = train_sensor_drift_model(X, y, metadata, config.models, model_type)
    elif task == "soiling":
        model_dict = train_soiling_model(X, y, metadata, config.models, model_type)
    else:
        typer.echo(f"Error: Unknown task {task}")
        raise typer.Exit(1)
    
    # Save model
    model_output_path = Path(model_output)
    model_output_path.mkdir(parents=True, exist_ok=True)
    model_file = model_output_path / f"{task}_{model_type}.pkl"
    save_model(model_dict, model_file)
    
    typer.echo(f"✓ Model saved to {model_file}")
    typer.echo(f"\nTest Metrics:")
    for metric, value in model_dict["test_metrics"].items():
        typer.echo(f"  {metric}: {value:.4f}")


@app.command()
def evaluate(
    task: str = typer.Option(..., help="Task name"),
    model_path: str = typer.Option(..., help="Path to trained model (.pkl)"),
    data_dir: str = typer.Option("data", help="Directory with test data"),
    output_dir: str = typer.Option("reports", help="Output directory for evaluation reports"),
):
    """Evaluate a trained model and generate reports."""
    typer.echo(f"Evaluating {task} model...")
    
    from src.models.training import load_model
    
    # Load model
    model_dict = load_model(Path(model_path))
    
    # Load test data
    data_path = Path(data_dir)
    telemetry = pd.read_csv(data_path / "telemetry.csv")
    labels = pd.read_csv(data_path / "labels.csv")
    
    # Engineer features
    config = DEFAULT_CONFIG
    X, y, metadata = engineer_features(
        telemetry, labels, task, config.features,
        granularity_minutes=config.simulation.granularity_minutes
    )
    
    # Make predictions
    y_pred = predict(model_dict, X)
    y_proba = predict(model_dict, X, return_proba=True) if task != "soiling" else None
    
    # Generate report
    output_path = Path(output_dir)
    generate_evaluation_report(
        model_dict, output_path, metadata, y.values, y_pred, y_proba
    )
    
    typer.echo(f"✓ Evaluation report saved to {output_path}")


@app.command()
def predict_cmd(
    task: str = typer.Option(..., help="Task name"),
    model_path: str = typer.Option(..., help="Path to trained model (.pkl)"),
    input_data: str = typer.Option(..., help="Path to input CSV file with features"),
    output: str = typer.Option("predictions.csv", help="Output CSV file for predictions"),
):
    """Run predictions on new data."""
    typer.echo(f"Running predictions for {task}...")
    
    results = predict_from_file(Path(model_path), Path(input_data), Path(output))
    
    typer.echo(f"✓ Predictions saved to {output}")
    typer.echo(f"  Generated {len(results)} predictions")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
