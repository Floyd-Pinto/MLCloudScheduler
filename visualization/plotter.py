"""
visualization/plotter.py
------------------------
Produces comparison plots between reactive and predictive schedulers.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_workload_vs_capacity(
    reactive_df: pd.DataFrame,
    predictive_df: pd.DataFrame,
    pattern_name: str = "workload",
):
    """Side-by-side workload and allocated capacity over time."""
    _ensure_output_dir()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for ax, df, label, color in zip(
        axes,
        [reactive_df, predictive_df],
        ["Reactive Scheduler", "Predictive Scheduler"],
        ["tab:orange", "tab:blue"],
    ):
        ax.plot(df["time_step"], df["workload"], label="Workload",
                color="grey", linewidth=1.2, alpha=0.8)
        ax.plot(df["time_step"], df["capacity"], label="Capacity",
                color=color, linewidth=1.8)
        overloads = df[df["overloaded"]]
        ax.scatter(overloads["time_step"], overloads["workload"],
                   color="red", s=18, zorder=5, label="Overload")
        ax.set_ylabel("Value")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(f"Workload vs Capacity — {pattern_name}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{pattern_name}_capacity.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_cpu_usage(
    reactive_df: pd.DataFrame,
    predictive_df: pd.DataFrame,
    pattern_name: str = "workload",
    overload_threshold: float = 80.0,
):
    """CPU utilisation comparison."""
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(reactive_df["time_step"], reactive_df["cpu_usage"],
            label="Reactive", color="tab:orange", linewidth=1.5)
    ax.plot(predictive_df["time_step"], predictive_df["cpu_usage"],
            label="Predictive", color="tab:blue", linewidth=1.5)
    ax.axhline(overload_threshold, color="red", linestyle="--",
               linewidth=1.2, label=f"Overload threshold ({overload_threshold}%)")

    ax.set_xlabel("Time Step")
    ax.set_ylabel("CPU Usage (%)")
    ax.set_title(f"CPU Utilisation — {pattern_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{pattern_name}_cpu.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_comparison(summaries: dict[str, dict], pattern_name: str = "workload"):
    """Bar chart of key metrics side-by-side."""
    _ensure_output_dir()
    metrics = ["overload_events", "avg_cpu_%", "avg_capacity", "total_cost"]
    labels = list(summaries.keys())
    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:orange", "tab:blue"]
    for i, (label, summary) in enumerate(summaries.items()):
        values = [summary[m] for m in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar([xi + offset for xi in x], values, width=width,
                      label=label, color=colors[i], alpha=0.85)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(["Overload\nEvents", "Avg CPU %",
                        "Avg Capacity", "Total Cost"], fontsize=10)
    ax.set_title(f"Scheduler Comparison Summary — {pattern_name}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{pattern_name}_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")
