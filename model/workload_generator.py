"""
model/workload_generator.py
---------------------------
Generates synthetic cloud workload patterns:
  - gradual  : steady linear growth
  - spike    : sudden burst (flash sale / viral event)
  - periodic : daily sinusoidal cycle
  - combined : mix of gradual + periodic + spike
"""

import numpy as np
import pandas as pd


def generate_gradual(steps: int = 200, base: float = 20.0, slope: float = 0.3,
                     noise_std: float = 2.0, seed: int = 42) -> np.ndarray:
    """Linearly increasing workload with Gaussian noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps)
    load = base + slope * t + rng.normal(0, noise_std, steps)
    return np.clip(load, 0, 100)


def generate_spike(steps: int = 200, base: float = 30.0,
                   spike_at: int = 100, spike_height: float = 60.0,
                   spike_width: int = 20, noise_std: float = 2.0,
                   seed: int = 42) -> np.ndarray:
    """Stable load with a sudden spike (e.g., flash sale)."""
    rng = np.random.default_rng(seed)
    load = np.full(steps, base, dtype=float)
    spike = spike_height * np.exp(
        -0.5 * ((np.arange(steps) - spike_at) / (spike_width / 2.5)) ** 2
    )
    load += spike + rng.normal(0, noise_std, steps)
    return np.clip(load, 0, 100)


def generate_periodic(steps: int = 200, base: float = 40.0,
                      amplitude: float = 25.0, period: float = 50.0,
                      noise_std: float = 2.5, seed: int = 42) -> np.ndarray:
    """Sinusoidal daily traffic cycle."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps)
    load = base + amplitude * np.sin(2 * np.pi * t / period)
    load += rng.normal(0, noise_std, steps)
    return np.clip(load, 0, 100)


def generate_combined(steps: int = 300, seed: int = 42) -> np.ndarray:
    """Mix of gradual trend + periodic cycle + a mid-run spike."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps)
    trend = 20 + 0.15 * t
    cycle = 15 * np.sin(2 * np.pi * t / 60)
    spike = 35 * np.exp(-0.5 * ((t - 150) / 10) ** 2)
    noise = rng.normal(0, 2.5, steps)
    return np.clip(trend + cycle + spike + noise, 0, 100)


PATTERN_GENERATORS = {
    "gradual":  generate_gradual,
    "spike":    generate_spike,
    "periodic": generate_periodic,
    "combined": generate_combined,
}


def generate(pattern: str = "combined", steps: int = 200, seed: int = 42) -> np.ndarray:
    """High-level entry point for API use."""
    fn = PATTERN_GENERATORS.get(pattern)
    if fn is None:
        raise ValueError(f"Unknown pattern '{pattern}'. Choose from {list(PATTERN_GENERATORS)}")
    if pattern == "combined":
        return fn(steps=steps, seed=seed)
    return fn(steps=steps, seed=seed)


def to_dataframe(workload: np.ndarray, pattern_name: str = "workload") -> pd.DataFrame:
    return pd.DataFrame({
        "time_step": np.arange(len(workload)),
        "workload":  workload.round(4),
        "pattern":   pattern_name,
    })
