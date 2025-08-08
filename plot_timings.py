"""
Timing analysis plots for GPT-2 models.

This script creates various plots to analyze timing performance (INF, FWD, TS)
as a function of model parameters, sequence length, and batch size.
NO averaging across models is performed to avoid distortion.
"""

import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for better-looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load and clean timing data from CSV file.

    Args:
        csv_path: Path to the CSV file containing timing data.

    Returns:
        Cleaned DataFrame with timing data.

    """
    df = pd.read_csv(csv_path)

    # Filter out failed runs (out_of_memory)
    df = df[df["status"] == "success"].copy()

    # Convert timing columns to numeric, handling missing values
    timing_cols = ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]
    for col in timing_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows with missing timing data
    df = df.dropna(subset=timing_cols)

    return df


def create_timing_vs_params_plot(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """Create plot showing timing vs model parameters for different batch sizes.

    Args:
        df: DataFrame containing timing data.
        save_path: Optional path to save the plot.

    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    timing_cols = ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]
    titles = ["Inference Time", "Forward Pass Time", "Total Time"]

    # Get unique batch sizes for color coding
    batch_sizes = sorted(df["batch_size"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(batch_sizes)))

    for i, (col, title) in enumerate(zip(timing_cols, titles)):
        ax = axes[i]

        for j, batch_size in enumerate(batch_sizes):
            batch_data = df[df["batch_size"] == batch_size]
            if len(batch_data) > 0:
                ax.scatter(
                    batch_data["total_params_millions"],
                    batch_data[col],
                    c=[colors[j]],
                    s=50,
                    alpha=0.7,
                    label=f"Batch Size {batch_size}",
                )

        ax.set_xlabel("Model Parameters (Millions)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_timing_vs_batch_size_plot(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """Create plot showing timing vs batch size for different models.

    Args:
        df: DataFrame containing timing data.
        save_path: Optional path to save the plot.

    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    timing_cols = ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]
    titles = ["Inference Time", "Forward Pass Time", "Total Time"]

    # Get unique models for color coding
    models = sorted(df["model_name"].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for i, (col, title) in enumerate(zip(timing_cols, titles)):
        ax = axes[i]

        for j, model in enumerate(models):
            model_data = df[df["model_name"] == model]
            if len(model_data) > 0:
                # Group by batch size and take mean for each sequence length
                grouped = model_data.groupby("batch_size")[col].mean()
                ax.plot(
                    grouped.index,
                    grouped.values,
                    marker="o",
                    c=colors[j],
                    label=f"{model}",
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_timing_vs_seq_length_plot(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """Create plot showing timing vs sequence length for different models.

    Args:
        df: DataFrame containing timing data.
        save_path: Optional path to save the plot.

    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    timing_cols = ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]
    titles = ["Inference Time", "Forward Pass Time", "Total Time"]

    # Get unique models for color coding
    models = sorted(df["model_name"].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    for i, (col, title) in enumerate(zip(timing_cols, titles)):
        ax = axes[i]

        for j, model in enumerate(models):
            model_data = df[df["model_name"] == model]
            if len(model_data) > 0:
                # Group by sequence length and take mean for each batch size
                grouped = model_data.groupby("sequence_length")[col].mean()
                ax.plot(
                    grouped.index,
                    grouped.values,
                    marker="s",
                    c=colors[j],
                    label=f"{model}",
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Time (ms)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_heatmap_plot(
    df: pd.DataFrame, timing_col: str, save_path: Optional[str] = None
) -> None:
    """Create heatmap showing timing as function of batch size and sequence length.

    This version creates separate heatmaps for each model to avoid averaging
    across different model sizes.

    Args:
        df: DataFrame containing timing data.
        timing_col: Column name for timing metric to plot.
        save_path: Optional path to save the plot.

    """
    # Get unique models
    models = sorted(df["model_name"].unique())

    # Create subplots for each model
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, model in enumerate(models):
        if i >= len(axes):
            break

        model_data = df[df["model_name"] == model]

        if len(model_data) == 0:
            continue

        # Create pivot table for this specific model
        pivot_data = model_data.pivot_table(
            values=timing_col,
            index="batch_size",
            columns="sequence_length",
            aggfunc="mean",
        )

        # Create heatmap for this model
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": f"{timing_col} (ms)"},
            ax=axes[i],
        )
        axes[i].set_title(f"{model} - {timing_col}")
        axes[i].set_xlabel("Sequence Length")
        axes[i].set_ylabel("Batch Size")

    # Hide unused subplots
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_model_specific_analysis(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """Create detailed analysis showing timing patterns for each model separately.

    Args:
        df: DataFrame containing timing data.
        save_path: Optional path to save the plot.

    """
    models = sorted(df["model_name"].unique())
    timing_cols = ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]

    fig, axes = plt.subplots(len(models), 3, figsize=(18, 4 * len(models)))

    for i, model in enumerate(models):
        model_data = df[df["model_name"] == model]

        for j, timing_col in enumerate(timing_cols):
            ax = axes[i, j] if len(models) > 1 else axes[j]

            # Create pivot table for this model and timing metric
            pivot_data = model_data.pivot_table(
                values=timing_col,
                index="batch_size",
                columns="sequence_length",
                aggfunc="mean",
            )

            # Create heatmap
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                cbar_kws={"label": f"{timing_col} (ms)"},
                ax=ax,
            )

            ax.set_title(f"{model} - {timing_col}")
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Batch Size")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_3d_scatter_plot(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Create 3D scatter plot showing timing vs parameters and batch size.

    Args:
        df: DataFrame containing timing data.
        save_path: Optional path to save the plot.

    """
    fig = plt.figure(figsize=(15, 5))

    timing_cols = ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]
    titles = ["Inference Time", "Forward Pass Time", "Total Time"]

    for i, (col, title) in enumerate(zip(timing_cols, titles)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")

        scatter = ax.scatter(
            df["total_params_millions"],
            df["batch_size"],
            df[col],
            c=df["sequence_length"],
            cmap="viridis",
            s=50,
            alpha=0.7,
        )

        ax.set_xlabel("Model Parameters (Millions)")
        ax.set_ylabel("Batch Size")
        ax.set_zlabel("Time (ms)")
        ax.set_title(title)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label("Sequence Length")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_efficiency_analysis(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """Create efficiency analysis plots (time per token, throughput).

    Args:
        df: DataFrame containing timing data.
        save_path: Optional path to save the plot.

    """
    # Calculate tokens per second (throughput)
    df["tokens_per_second"] = (df["batch_size"] * df["sequence_length"]) / (
        df["TS_time_ms"] / 1000
    )

    # Calculate time per token
    df["time_per_token_ms"] = df["TS_time_ms"] / (
        df["batch_size"] * df["sequence_length"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Throughput vs batch size
    for model in sorted(df["model_name"].unique()):
        model_data = df[df["model_name"] == model]
        grouped = model_data.groupby("batch_size")["tokens_per_second"].mean()
        axes[0, 0].plot(
            grouped.index, grouped.values, marker="o", label=model, linewidth=2
        )

    axes[0, 0].set_xlabel("Batch Size")
    axes[0, 0].set_ylabel("Tokens per Second")
    axes[0, 0].set_title("Throughput vs Batch Size")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")

    # Time per token vs model size
    for batch_size in sorted(df["batch_size"].unique()):
        batch_data = df[df["batch_size"] == batch_size]
        if len(batch_data) > 0:
            grouped = batch_data.groupby("total_params_millions")[
                "time_per_token_ms"
            ].mean()
            axes[0, 1].scatter(
                grouped.index,
                grouped.values,
                label=f"Batch {batch_size}",
                s=50,
                alpha=0.7,
            )

    axes[0, 1].set_xlabel("Model Parameters (Millions)")
    axes[0, 1].set_ylabel("Time per Token (ms)")
    axes[0, 1].set_title("Time per Token vs Model Size")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")

    # Memory efficiency (tokens per second per million parameters)
    df["efficiency"] = df["tokens_per_second"] / df["total_params_millions"]

    for model in sorted(df["model_name"].unique()):
        model_data = df[df["model_name"] == model]
        grouped = model_data.groupby("batch_size")["efficiency"].mean()
        axes[1, 0].plot(
            grouped.index, grouped.values, marker="s", label=model, linewidth=2
        )

    axes[1, 0].set_xlabel("Batch Size")
    axes[1, 0].set_ylabel("Efficiency (tokens/sec/million params)")
    axes[1, 0].set_title("Memory Efficiency vs Batch Size")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_yscale("log")

    # FWD vs INF ratio
    df["fwd_inf_ratio"] = df["FWD_time_ms"] / df["INF_time_ms"]

    for model in sorted(df["model_name"].unique()):
        model_data = df[df["model_name"] == model]
        grouped = model_data.groupby("batch_size")["fwd_inf_ratio"].mean()
        axes[1, 1].plot(
            grouped.index, grouped.values, marker="^", label=model, linewidth=2
        )

    axes[1, 1].set_xlabel("Batch Size")
    axes[1, 1].set_ylabel("FWD/INF Time Ratio")
    axes[1, 1].set_title("Forward/Inference Time Ratio vs Batch Size")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale("log")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_model_comparison_plots(
    df: pd.DataFrame, save_path: Optional[str] = None
) -> None:
    """Create side-by-side comparison plots for each model.

    Args:
        df: DataFrame containing timing data.
        save_path: Optional path to save the plot.

    """
    models = sorted(df["model_name"].unique())
    timing_cols = ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]

    # Create separate plots for each timing metric
    for timing_col in timing_cols:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for i, model in enumerate(models):
            if i >= len(axes):
                break

            model_data = df[df["model_name"] == model]

            if len(model_data) == 0:
                continue

            # Create pivot table for this model
            pivot_data = model_data.pivot_table(
                values=timing_col,
                index="batch_size",
                columns="sequence_length",
                aggfunc="mean",
            )

            # Create heatmap
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt=".1f",
                cmap="YlOrRd",
                cbar_kws={"label": f"{timing_col} (ms)"},
                ax=axes[i],
            )

            axes[i].set_title(f"{model}\n{timing_col}")
            axes[i].set_xlabel("Sequence Length")
            axes[i].set_ylabel("Batch Size")

        # Hide unused subplots
        for i in range(len(models), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f"{timing_col} Comparison Across Models", fontsize=16)
        plt.tight_layout()

        if save_path:
            # Create filename with timing column
            filename = save_path.replace(".png", f"_{timing_col}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.show()


def analyze_strange_patterns(df: pd.DataFrame) -> None:
    """Analyze and explain the strange patterns in the timing data.

    Args:
        df: DataFrame containing timing data.

    """
    print("=== ANALYSIS OF STRANGE PATTERNS IN TIMING DATA ===\n")

    # 1. Check for non-monotonic behavior in sequence length
    print("1. NON-MONOTONIC BEHAVIOR IN SEQUENCE LENGTH:")
    print(
        "   Looking for cases where timing decreases with increasing sequence length..."
    )

    for model in sorted(df["model_name"].unique()):
        model_data = df[df["model_name"] == model]
        print(f"\n   Model: {model}")

        for batch_size in sorted(model_data["batch_size"].unique()):
            batch_data = model_data[model_data["batch_size"] == batch_size]
            if len(batch_data) < 2:
                continue

            # Sort by sequence length
            sorted_data = batch_data.sort_values("sequence_length")

            for timing_col in ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]:
                times = sorted_data[timing_col].values
                seq_lengths = sorted_data["sequence_length"].values

                # Check for decreases
                for i in range(1, len(times)):
                    if times[i] < times[i - 1]:
                        print(
                            f"     Batch {batch_size}, {timing_col}: "
                            f"{seq_lengths[i-1]}→{seq_lengths[i]}: "
                            f"{times[i-1]:.1f}ms → {times[i]:.1f}ms "
                            f"(decrease of {times[i-1]-times[i]:.1f}ms)"
                        )

    # 2. Analyze memory pressure effects
    print("\n2. MEMORY PRESSURE ANALYSIS:")
    print(
        "   Looking for sudden jumps in timing that might indicate memory pressure..."
    )

    for model in sorted(df["model_name"].unique()):
        model_data = df[df["model_name"] == model]
        print(f"\n   Model: {model}")

        for batch_size in sorted(model_data["batch_size"].unique()):
            batch_data = model_data[model_data["batch_size"] == batch_size]
            if len(batch_data) < 2:
                continue

            sorted_data = batch_data.sort_values("sequence_length")

            for timing_col in ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]:
                times = sorted_data[timing_col].values
                seq_lengths = sorted_data["sequence_length"].values

                # Check for large jumps (>50% increase)
                for i in range(1, len(times)):
                    increase_ratio = times[i] / times[i - 1]
                    if increase_ratio > 1.5:
                        print(
                            f"     Batch {batch_size}, {timing_col}: "
                            f"{seq_lengths[i-1]}→{seq_lengths[i]}: "
                            f"{times[i-1]:.1f}ms → {times[i]:.1f}ms "
                            f"({increase_ratio:.1f}x increase)"
                        )

    # 3. Model size effects
    print("\n3. MODEL SIZE EFFECTS:")
    print("   Analyzing how different model sizes affect timing patterns...")

    for batch_size in sorted(df["batch_size"].unique()):
        batch_data = df[df["batch_size"] == batch_size]
        if len(batch_data) == 0:
            continue

        print(f"\n   Batch Size: {batch_size}")

        for seq_length in sorted(batch_data["sequence_length"].unique()):
            seq_data = batch_data[batch_data["sequence_length"] == seq_length]
            if len(seq_data) < 2:
                continue

            print(f"     Sequence Length: {seq_length}")
            for _, row in seq_data.iterrows():
                print(
                    f"       {row['model_name']}: "
                    f"INF={row['INF_time_ms']:.1f}ms, "
                    f"FWD={row['FWD_time_ms']:.1f}ms, "
                    f"TS={row['TS_time_ms']:.1f}ms"
                )


def main():
    """Main function to generate all timing analysis plots."""
    # Load data
    csv_path = "timing_results_20250808_160512.csv"
    df = load_and_clean_data(csv_path)

    print(f"Loaded {len(df)} successful timing measurements")
    print(f"Models: {sorted(df['model_name'].unique())}")
    print(f"Batch sizes: {sorted(df['batch_size'].unique())}")
    print(f"Sequence lengths: {sorted(df['sequence_length'].unique())}")

    # Create plots directory
    import os

    os.makedirs("plots", exist_ok=True)

    # Analyze strange patterns first
    analyze_strange_patterns(df)

    # Generate all plots (NO averaging across models)
    print("\nGenerating timing analysis plots (no model averaging)...")

    # 1. Timing vs model parameters (scatter plot - no averaging)
    create_timing_vs_params_plot(df, "plots/timing_vs_params.png")

    # 2. Timing vs batch size (separate lines for each model)
    create_timing_vs_batch_size_plot(df, "plots/timing_vs_batch_size.png")

    # 3. Timing vs sequence length (separate lines for each model)
    create_timing_vs_seq_length_plot(df, "plots/timing_vs_seq_length.png")

    # 4. Heatmaps - separate for each model (NO averaging)
    for timing_col in ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]:
        create_heatmap_plot(df, timing_col, f"plots/heatmap_{timing_col}_by_model.png")

    # 5. Model-specific detailed analysis
    create_model_specific_analysis(df, "plots/model_specific_analysis.png")

    # 6. Model comparison plots
    create_model_comparison_plots(df, "plots/model_comparison.png")

    # 7. 3D scatter plots (no averaging)
    create_3d_scatter_plot(df, "plots/3d_scatter_timing.png")

    # 8. Efficiency analysis (separate lines for each model)
    create_efficiency_analysis(df, "plots/efficiency_analysis.png")

    print("All plots generated successfully in the 'plots' directory!")
    print("Note: NO averaging across models was performed to avoid distortion.")


if __name__ == "__main__":
    main()
