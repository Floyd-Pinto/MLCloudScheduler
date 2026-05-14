"""
model/data/alibaba_loader.py
-----------------------------
Downloads and processes Alibaba Cluster Trace v2018 data for use
in multi-resource workload forecasting.

Source: https://github.com/alibaba/clusterdata

Produces a numpy array of shape (N, 3) with columns:
  [cpu_usage, memory_usage, network_io]
"""

import os
import sys
import gzip
import tarfile
import io
import warnings
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "alibaba_trace_sample.npy")

# Alibaba Cluster Trace v2018 — machine usage data
TRACE_URL = (
    "https://raw.githubusercontent.com/alibaba/clusterdata/"
    "master/cluster-trace-v2018/trace_2018.md"
)

# Direct CSV download (smaller subset for practical use)
TRACE_CSV_URL = (
    "https://raw.githubusercontent.com/alibaba/clusterdata/"
    "master/cluster-trace-v2018/data/machine_usage.tar.gz"
)


def _derive_network_io(cpu_usage: np.ndarray, seed: int = 2025) -> np.ndarray:
    """
    Derive synthetic network I/O from CPU signal using the project's
    consistent formula.
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
        "⚠ Alibaba Cluster Trace download failed. Using synthetic fallback data.",
        RuntimeWarning,
    )
    sys.path.insert(0, os.path.dirname(os.path.dirname(DATA_DIR)))
    from model.workload_generator import generate_multivariate
    return generate_multivariate("combined", steps=steps, seed=99)


def download_and_process(output_path: str = OUTPUT_PATH, timeout: int = 30) -> np.ndarray:
    """
    Download, parse, and process Alibaba Cluster Trace v2018.

    The Alibaba trace machine_usage CSV has columns:
      machine_id, time_stamp, cpu_util_percent, mem_util_percent,
      mem_gps, mpki, net_in, net_out, disk_io_percent

    Steps:
    1. Download machine_usage data
    2. Parse CPU and memory utilisation
    3. Select the container/machine with the most data
    4. Derive network_io from CPU
    5. Save as .npy file

    Returns
    -------
    np.ndarray
        Shape (N, 3) with columns [cpu_usage, memory_usage, network_io].
    """
    import urllib.request

    print("📥 Downloading Alibaba Cluster Trace v2018...")
    print(f"   URL: {TRACE_CSV_URL}")

    raw_data = None
    try:
        req = urllib.request.Request(
            TRACE_CSV_URL,
            headers={"User-Agent": "MLCloudScheduler/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_data = resp.read()
        print(f"   ✓ Downloaded {len(raw_data) / 1e6:.1f} MB")
    except Exception as e:
        print(f"   ⚠ Download failed: {e}")

    if raw_data is None:
        data = _generate_synthetic_fallback()
        np.save(output_path, data)
        return data

    # Try to extract from tar.gz
    print("   Extracting and parsing...")
    lines = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw_data), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and "machine_usage" in member.name:
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8", errors="replace")
                        lines.extend(content.strip().split("\n"))
                        # Only read the first file for efficiency
                        break
    except Exception as e:
        print(f"   ⚠ Tar extraction failed: {e}")
        # Try as plain gzipped CSV
        try:
            decompressed = gzip.decompress(raw_data)
            lines = decompressed.decode("utf-8", errors="replace").strip().split("\n")
        except Exception:
            print("   ⚠ Cannot decompress. Using synthetic fallback.")
            data = _generate_synthetic_fallback()
            np.save(output_path, data)
            return data

    if not lines:
        print("   ⚠ No data lines found. Using synthetic fallback.")
        data = _generate_synthetic_fallback()
        np.save(output_path, data)
        return data

    # Parse: machine_id, time_stamp, cpu_util_percent, mem_util_percent, ...
    from collections import defaultdict
    machine_data = defaultdict(list)
    parsed = 0
    skipped = 0

    for line in lines:
        fields = line.split(",")
        if len(fields) < 4:
            skipped += 1
            continue
        try:
            machine_id = fields[0].strip()
            timestamp = int(float(fields[1]))
            cpu_val = float(fields[2]) if fields[2].strip() else None
            mem_val = float(fields[3]) if fields[3].strip() else None
        except (ValueError, IndexError):
            skipped += 1
            continue

        if cpu_val is None or mem_val is None:
            skipped += 1
            continue

        # Alibaba trace may already be in percentage form
        # If values are > 1, assume already percentage; otherwise scale
        if cpu_val <= 1.0:
            cpu_val *= 100.0
        if mem_val <= 1.0:
            mem_val *= 100.0

        cpu_pct = np.clip(cpu_val, 0, 100)
        mem_pct = np.clip(mem_val, 0, 100)

        machine_data[machine_id].append((timestamp, cpu_pct, mem_pct))
        parsed += 1

    print(f"   Parsed {parsed:,} records from {len(machine_data):,} machines")

    if not machine_data:
        print("   ⚠ No valid data. Using synthetic fallback.")
        data = _generate_synthetic_fallback()
        np.save(output_path, data)
        return data

    # Pick the machine with the most data
    best_machine = max(machine_data, key=lambda m: len(machine_data[m]))
    records = machine_data[best_machine]
    records.sort(key=lambda r: r[0])

    print(f"   Selected machine '{best_machine}' with {len(records):,} time points")

    # Deduplicate by timestamp
    bucket_agg = defaultdict(list)
    for ts, cpu, mem in records:
        bucket_agg[ts].append((cpu, mem))

    sorted_ts = sorted(bucket_agg.keys())
    cpu_series = [np.mean([v[0] for v in bucket_agg[t]]) for t in sorted_ts]
    mem_series = [np.mean([v[1] for v in bucket_agg[t]]) for t in sorted_ts]

    cpu_arr = np.array(cpu_series)
    mem_arr = np.array(mem_series)

    # Derive network_io
    net_arr = _derive_network_io(cpu_arr)

    data = np.column_stack([cpu_arr, mem_arr, net_arr])

    # Limit to 10000 steps max
    if len(data) > 10000:
        data = data[:10000]

    print(f"   ✓ Processed: {data.shape[0]} time steps × {data.shape[1]} features")
    print(f"   CPU range: [{data[:, 0].min():.1f}, {data[:, 0].max():.1f}]")
    print(f"   MEM range: [{data[:, 1].min():.1f}, {data[:, 1].max():.1f}]")
    print(f"   NET range: [{data[:, 2].min():.1f}, {data[:, 2].max():.1f}]")

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
