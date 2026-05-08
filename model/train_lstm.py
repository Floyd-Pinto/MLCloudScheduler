"""
model/train_lstm.py
-------------------
Training script for the multi-resource LSTM forecaster.

Generates three-dimensional workload data (CPU, memory, network I/O),
trains the LSTM with a window of 20 time steps and a 5-step forecast
horizon, and reports per-resource evaluation metrics.

Usage:
    python model/train_lstm.py

Outputs:
    model/saved_models/lstm_model.pt
    model/saved_models/lstm_scaler.pkl
    model/saved_models/lstm_meta.json
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.workload_generator import generate_multivariate
from model.lstm_model import LSTMForecaster

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


def build_training_data(seed: int = 42) -> np.ndarray:
    """
    Construct a diverse multi-resource training corpus.

    Concatenates multiple workload patterns with varying parameters
    to expose the LSTM to a wide range of temporal dynamics.
    """
    segments = [
        generate_multivariate("gradual",  steps=500, seed=seed),
        generate_multivariate("spike",    steps=500, seed=seed + 1),
        generate_multivariate("periodic", steps=500, seed=seed + 2),
        generate_multivariate("combined", steps=600, seed=seed + 3),
        # Additional spike variations for robustness
        generate_multivariate("spike",    steps=400, seed=seed + 10),
        generate_multivariate("spike",    steps=400, seed=seed + 11),
        generate_multivariate("combined", steps=500, seed=seed + 7),
        generate_multivariate("gradual",  steps=400, seed=seed + 4),
        generate_multivariate("periodic", steps=400, seed=seed + 6),
    ]
    return np.vstack(segments)


def train(data: np.ndarray | None = None, verbose: bool = True) -> dict:
    """Train LSTM and return per-resource metrics."""
    if data is None:
        if verbose:
            print("Generating multi-resource training data for LSTM...")
        data = build_training_data()

    if verbose:
        print(f"  Training data: {data.shape[0]} steps × {data.shape[1]} features")

    model = LSTMForecaster()
    metrics = model.fit(data, verbose=verbose)
    model.save()

    if verbose:
        print(f"\n  LSTM Results:")
        for key in ["cpu_r2", "memory_r2", "network_r2", "r2"]:
            if key in metrics:
                label = key.replace("_", " ").title()
                print(f"    {label}: {metrics[key]:.4f}")

    return metrics


if __name__ == "__main__":
    result = train()
    if result:
        print(f"\nLSTM training complete. Overall R² = {result['r2']:.4f}")
