"""
model/data/google_trace_loader.py
---------------------------------
Downloads and processes a representative sample from the Google Cluster
Trace 2019 dataset for use in multi-resource workload forecasting.

Source: https://storage.googleapis.com/clusterdata-2019-2/
Paper:  Reiss et al., "Google cluster-workload traces: format + schema" (2019)

Produces a numpy array of shape (N, 3) with columns:
  [cpu_usage, memory_usage, network_io]
"""

import os
import sys
import gzip
import io
import warnings
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "google_trace_sample.npy")

# Google Cluster Trace 2019 — first shard of machine_usage
TRACE_URL = (
    "https://storage.googleapis.com/clusterdata_2019_a/"
    "machine_usage/part-00000-of-00500.csv.gz"
)

# Fallback: alternative URL pattern
TRACE_URL_ALT = (
    "https://storage.googleapis.com/clusterdata-2019-2/"
    "machine_usage/part-00000-of-00500.csv.gz"
)

# Column indices in Google Cluster Trace 2019 machine_usage
# Columns: start_time, end_time, machine_id, mean_cpu_usage_rate,
#           canonical_memory_usage, assigned_memory, ...
COL_TIME = 0
COL_MACHINE_ID = 2
COL_CPU = 3
COL_MEM = 4

# Time bucketing: 5-minute intervals in microseconds
BUCKET_US = 5 * 60 * 1_000_000


def _derive_network_io(cpu_usage: np.ndarray, seed: int = 2024) -> np.ndarray:
    """
    Derive synthetic network I/O from CPU signal using the same formula
    as the project's workload generator for consistency.

    Formula: network(t) = clip(cpu(t) * 0.9 + spike_term + N(0, 8), 0, 100)
      where spike_term = Poisson(λ=0.05) * Uniform(15, 40)
    """
    rng = np.random.default_rng(seed)
    steps = len(cpu_usage)
    network = np.zeros(steps)
    for t in range(steps):
        spike_count = rng.poisson(0.05)
        spike_term = spike_count * rng.uniform(15, 40) if spike_count > 0 else 0.0
        network[t] = cpu_usage[t] * 0.9 + spike_term + rng.normal(0, 8)
    return np.clip(network, 0, 100)


def _generate_synthetic_fallback(steps: int = 5000) -> np.ndarray:
    """Generate synthetic data as fallback when real data download fails."""
    warnings.warn(
        "⚠ Google Cluster Trace download failed. Using synthetic fallback data.",
        RuntimeWarning,
    )
    # Use project's own generator
    sys.path.insert(0, os.path.dirname(os.path.dirname(DATA_DIR)))
    from model.workload_generator import generate_multivariate
    return generate_multivariate("combined", steps=steps, seed=42)


def download_and_process(output_path: str = OUTPUT_PATH, timeout: int = 30) -> np.ndarray:
    """
    Download, parse, and process Google Cluster Trace 2019 data.

    Steps:
    1. Download part-00000-of-00500.csv.gz from Google Cloud Storage
    2. Parse CPU and memory utilisation columns
    3. Bucket timestamps into 5-minute intervals
    4. Select the machine with the most complete data
    5. Derive network_io using the synthetic formula
    6. Save as .npy file

    Parameters
    ----------
    output_path : str
        Path to save the processed numpy array.
    timeout : int
        Download timeout in seconds.

    Returns
    -------
    np.ndarray
        Shape (N, 3) with columns [cpu_usage, memory_usage, network_io].
    """
    import urllib.request

    print("📥 Downloading Google Cluster Trace 2019 (first shard)...")
    print(f"   URL: {TRACE_URL}")

    raw_data = None
    for url in [TRACE_URL, TRACE_URL_ALT]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MLCloudScheduler/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_data = resp.read()
            print(f"   ✓ Downloaded {len(raw_data) / 1e6:.1f} MB")
            break
        except Exception as e:
            print(f"   ⚠ Failed with URL {url}: {e}")
            continue

    if raw_data is None:
        data = _generate_synthetic_fallback()
        np.save(output_path, data)
        return data

    # Parse the gzipped CSV
    print("   Parsing CSV data...")
    try:
        decompressed = gzip.decompress(raw_data)
        lines = decompressed.decode("utf-8", errors="replace").strip().split("\n")
    except Exception as e:
        print(f"   ⚠ Decompression failed: {e}. Using synthetic fallback.")
        data = _generate_synthetic_fallback()
        np.save(output_path, data)
        return data

    # Parse into per-machine time series
    machine_data = {}  # machine_id -> list of (time_bucket, cpu, mem)
    parsed = 0
    skipped = 0

    for line in lines:
        fields = line.split(",")
        if len(fields) < 5:
            skipped += 1
            continue
        try:
            time_us = float(fields[COL_TIME])
            machine_id = fields[COL_MACHINE_ID].strip()
            cpu_val = float(fields[COL_CPU]) if fields[COL_CPU].strip() else None
            mem_val = float(fields[COL_MEM]) if fields[COL_MEM].strip() else None
        except (ValueError, IndexError):
            skipped += 1
            continue

        if cpu_val is None or mem_val is None:
            skipped += 1
            continue

        # Convert to percentage (Google trace uses 0-1 scale)
        cpu_pct = np.clip(cpu_val * 100.0, 0, 100)
        mem_pct = np.clip(mem_val * 100.0, 0, 100)

        time_bucket = int(time_us / BUCKET_US)

        if machine_id not in machine_data:
            machine_data[machine_id] = []
        machine_data[machine_id].append((time_bucket, cpu_pct, mem_pct))
        parsed += 1

    print(f"   Parsed {parsed:,} records from {len(machine_data):,} machines (skipped {skipped:,})")

    if not machine_data:
        print("   ⚠ No valid data found. Using synthetic fallback.")
        data = _generate_synthetic_fallback()
        np.save(output_path, data)
        return data

    # Pick the machine with the most complete data
    best_machine = max(machine_data, key=lambda m: len(machine_data[m]))
    records = machine_data[best_machine]
    records.sort(key=lambda r: r[0])  # sort by time bucket

    print(f"   Selected machine '{best_machine}' with {len(records):,} time buckets")

    # Deduplicate: average values within same bucket
    from collections import defaultdict
    bucket_agg = defaultdict(list)
    for bucket, cpu, mem in records:
        bucket_agg[bucket].append((cpu, mem))

    sorted_buckets = sorted(bucket_agg.keys())
    cpu_series = []
    mem_series = []
    for b in sorted_buckets:
        vals = bucket_agg[b]
        cpu_series.append(np.mean([v[0] for v in vals]))
        mem_series.append(np.mean([v[1] for v in vals]))

    cpu_arr = np.array(cpu_series)
    mem_arr = np.array(mem_series)

    # Derive network_io
    net_arr = _derive_network_io(cpu_arr)

    # Stack into (N, 3)
    data = np.column_stack([cpu_arr, mem_arr, net_arr])

    # Limit to a reasonable size (max 10000 steps)
    if len(data) > 10000:
        data = data[:10000]

    print(f"   ✓ Processed: {data.shape[0]} time steps × {data.shape[1]} features")
    print(f"   CPU range: [{data[:, 0].min():.1f}, {data[:, 0].max():.1f}]")
    print(f"   MEM range: [{data[:, 1].min():.1f}, {data[:, 1].max():.1f}]")
    print(f"   NET range: [{data[:, 2].min():.1f}, {data[:, 2].max():.1f}]")

    # Save
    np.save(output_path, data)
    print(f"   💾 Saved to {output_path}")

    return data


def load(output_path: str = OUTPUT_PATH) -> np.ndarray | None:
    """Load previously processed trace data, or return None if not available."""
    if os.path.exists(output_path):
        return np.load(output_path)
    return None


if __name__ == "__main__":
    data = download_and_process()
    print(f"\nFinal shape: {data.shape}")
    print(f"CPU  — mean: {data[:, 0].mean():.2f}, std: {data[:, 0].std():.2f}")
    print(f"MEM  — mean: {data[:, 1].mean():.2f}, std: {data[:, 1].std():.2f}")
    print(f"NET  — mean: {data[:, 2].mean():.2f}, std: {data[:, 2].std():.2f}")
