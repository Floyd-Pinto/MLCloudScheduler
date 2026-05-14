"""
model/workload_generator.py
---------------------------
Generates synthetic cloud workload patterns with three correlated resource
signals: CPU utilisation, memory consumption, and network I/O throughput.

Supported patterns:
  - gradual  : steady linear growth
  - spike    : sudden burst (flash sale / viral event)
  - periodic : daily sinusoidal cycle
  - combined : mix of gradual + periodic + spike

Each time step produces:
  cpu_usage     – primary compute signal, 0–100 %
  memory_usage  – correlated with CPU (lag ≈ 3 steps, ρ ≈ 0.75)
  network_io    – bursty, proportional to workload with random spikes

Backward Compatibility:
  Functions still return np.ndarray of cpu_usage when called with
  ``multivariate=False`` (the default for the original single-signal API).
  Set ``multivariate=True`` to receive a (steps, 3) array.
"""

import numpy as np
import pandas as pd


# ── Single-signal generators (backward compatible) ───────────────────────────

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
    """High-level entry point for API use (returns cpu_usage only)."""
    fn = PATTERN_GENERATORS.get(pattern)
    if fn is None:
        raise ValueError(f"Unknown pattern '{pattern}'. Choose from {list(PATTERN_GENERATORS)}")
    if pattern == "combined":
        return fn(steps=steps, seed=seed)
    return fn(steps=steps, seed=seed)


# ── Real data loaders ────────────────────────────────────────────────────────

def generate_from_real_data(source: str = "google", steps: int = 600,
                            start_offset: int = 0) -> np.ndarray:
    """
    Load real-world cluster trace data and return a (steps, 3) slice.

    Parameters
    ----------
    source : str
        One of 'google' or 'alibaba'.
    steps : int
        Number of time steps to return.
    start_offset : int
        Starting index within the loaded data.

    Returns
    -------
    np.ndarray
        Shape (steps, 3) with columns [cpu_usage, memory_usage, network_io].
    """
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    if source == "google":
        npy_path = os.path.join(data_dir, "google_trace_sample.npy")
        if not os.path.exists(npy_path):
            from model.data.google_trace_loader import download_and_process
            download_and_process(npy_path)
        data = np.load(npy_path)
    elif source == "alibaba":
        npy_path = os.path.join(data_dir, "alibaba_trace_sample.npy")
        if not os.path.exists(npy_path):
            from model.data.alibaba_loader import download_and_process
            download_and_process(npy_path)
        data = np.load(npy_path)
    else:
        raise ValueError(f"Unknown source '{source}'. Choose 'google' or 'alibaba'.")

    # Ensure we have enough data
    end = start_offset + steps
    if end > len(data):
        # Wrap around or truncate
        end = len(data)
        start_offset = max(0, end - steps)

    return data[start_offset:start_offset + steps].copy()


# ── Multi-resource signal derivation ─────────────────────────────────────────

def derive_memory_usage(cpu_usage: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Derive memory utilisation from CPU signal.

    Memory exhibits temporal lag relative to CPU (approximately 3 steps)
    with a Pearson correlation coefficient of approximately 0.75, reflecting
    the empirical observation that memory allocation follows compute demand.

    Formula:  memory(t) = clip(0.7 * cpu(t-3) + 0.3 * cpu(t) + N(0, 5), 0, 100)
    """
    steps = len(cpu_usage)
    memory = np.zeros(steps)
    for t in range(steps):
        lagged = cpu_usage[max(0, t - 3)]
        memory[t] = 0.7 * lagged + 0.3 * cpu_usage[t] + rng.normal(0, 5)
    return np.clip(memory, 0, 100)


def derive_network_io(cpu_usage: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Derive network I/O throughput from the workload intensity signal.

    Network traffic is proportional to workload but exhibits stochastic
    bursty behaviour modelled via a Poisson process with random-amplitude
    spikes, reflecting real-world packet-burst phenomena.

    Formula:  network(t) = clip(workload(t) * 0.9 + spike_term + N(0, 8), 0, 100)
      where spike_term = Poisson(λ=0.05) * Uniform(15, 40)
    """
    steps = len(cpu_usage)
    network = np.zeros(steps)
    for t in range(steps):
        spike_count = rng.poisson(0.05)
        spike_term = spike_count * rng.uniform(15, 40) if spike_count > 0 else 0.0
        network[t] = cpu_usage[t] * 0.9 + spike_term + rng.normal(0, 8)
    return np.clip(network, 0, 100)


# ── Multi-resource workload generation ───────────────────────────────────────

def generate_multivariate(pattern: str = "combined", steps: int = 200,
                          seed: int = 42) -> np.ndarray:
    """
    Generate a multivariate workload signal with three correlated resources.

    Parameters
    ----------
    pattern : str
        One of 'gradual', 'spike', 'periodic', 'combined', 'google_trace', 'alibaba_trace'.
    steps : int
        Number of time steps to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (steps, 3) with columns [cpu_usage, memory_usage, network_io].
    """
    if pattern == "google_trace":
        return generate_from_real_data("google", steps=steps, start_offset=seed)
    if pattern == "alibaba_trace":
        return generate_from_real_data("alibaba", steps=steps, start_offset=seed)

    cpu_usage = generate(pattern=pattern, steps=steps, seed=seed)
    rng = np.random.default_rng(seed + 1000)  # separate seed for derived signals
    memory_usage = derive_memory_usage(cpu_usage, rng)
    network_io = derive_network_io(cpu_usage, rng)
    return np.column_stack([cpu_usage, memory_usage, network_io])


def generate_multivariate_records(pattern: str = "combined", steps: int = 200,
                                  seed: int = 42) -> list[dict]:
    """
    Generate multi-resource workload as a list of per-step dictionaries.

    Each dictionary contains:
      cpu_usage, memory_usage, network_io, workload, timestamp
    """
    if pattern in ["google_trace", "alibaba_trace"]:
        data = generate_multivariate(pattern=pattern, steps=steps, seed=seed)
        records = []
        actual_steps = len(data)
        for t in range(actual_steps):
            records.append({
                "timestamp":    t,
                "cpu_usage":    round(float(data[t, 0]), 4),
                "memory_usage": round(float(data[t, 1]), 4),
                "network_io":   round(float(data[t, 2]), 4),
                "workload":     round(float(data[t, 0]), 4), # alias workload to cpu_usage
            })
        return records

    cpu_usage = generate(pattern=pattern, steps=steps, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    memory_usage = derive_memory_usage(cpu_usage, rng)
    network_io = derive_network_io(cpu_usage, rng)

    records = []
    for t in range(steps):
        records.append({
            "timestamp":    t,
            "cpu_usage":    round(float(cpu_usage[t]), 4),
            "memory_usage": round(float(memory_usage[t]), 4),
            "network_io":   round(float(network_io[t]), 4),
            "workload":     round(float(cpu_usage[t]), 4),
        })
    return records


def to_dataframe(workload: np.ndarray, pattern_name: str = "workload") -> pd.DataFrame:
    """Convert workload array to a pandas DataFrame (backward compatible)."""
    if workload.ndim == 1:
        return pd.DataFrame({
            "time_step": np.arange(len(workload)),
            "workload":  workload.round(4),
            "pattern":   pattern_name,
        })
    # Multivariate case: (steps, 3)
    return pd.DataFrame({
        "time_step":    np.arange(len(workload)),
        "cpu_usage":    workload[:, 0].round(4),
        "memory_usage": workload[:, 1].round(4),
        "network_io":   workload[:, 2].round(4),
        "pattern":      pattern_name,
    })
