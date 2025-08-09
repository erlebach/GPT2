# /Users/erlebach/src/2025/GPT2/time_line_plots.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterator, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

try:
    from jaxtyping import Float, Int  # type: ignore
except Exception:  # pragma: no cover
    Float = Any  # fallback
    Int = Any  # fallback


def find_latest_timing_csv(root: str | Path = ".") -> Path:
    """Find the most recent timing CSV file (timing_results_*.csv).

    Args:
        root: Directory to search from (recursively).

    Returns:
        Path to the most recent timing CSV.

    Raises:
        FileNotFoundError: If no matching file is found.

    """
    root = Path(root)
    candidates: list[Path] = []
    for p in root.rglob("timing_results_*.csv"):
        if p.is_file():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError("No timing CSV found matching 'timing_results_*.csv'.")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_and_clean_data(csv_path: str | Path) -> pd.DataFrame:
    """Load timing CSV, filter successes, and coerce numeric timings.

    Follows the same rules as `stacked_bar_charts.py`:
    - Keep only rows where status == "success".
    - Coerce INF/FWD/TS time columns to numeric and drop NaNs.

    Args:
        csv_path: Absolute or relative path to the CSV file.

    Returns:
        Cleaned dataframe.

    """
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "success"].copy()

    for col in ["INF_time_ms", "FWD_time_ms", "TS_time_ms"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["INF_time_ms", "FWD_time_ms", "TS_time_ms"])

    # Normalize dtypes for grouping/plotting
    df["model_name"] = df["model_name"].astype(str)
    df["batch_size"] = pd.to_numeric(df["batch_size"], errors="coerce").astype(int)
    df["sequence_length"] = pd.to_numeric(
        df["sequence_length"], errors="coerce"
    ).astype(int)

    return df


def pick_medium_model(df: pd.DataFrame, preferred: str = "medium1024") -> str:
    """Pick a single medium model name, preferring a given name if present.

    Args:
        df: Source DataFrame including 'model_name'.
        preferred: Preferred 'medium*' model name.

    Returns:
        Chosen model name.

    Raises:
        ValueError: If no 'medium*' model is present.

    """
    mediums = sorted(
        df.loc[df["model_name"].str.contains("medium", case=False), "model_name"]
        .dropna()
        .unique()
        .tolist()
    )
    if not mediums:
        raise ValueError("No model_name containing 'medium' found in the CSV.")
    if preferred in mediums:
        return preferred
    return mediums[0]


def _cycled_styles() -> Iterator[tuple[str, str]]:
    """Yield a cycle of (linestyle, marker) pairs for distinguishable lines.

    Returns:
        Iterator over style tuples.

    """
    linestyles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    styles = [(ls, mk) for ls in linestyles for mk in markers]
    while True:
        for s in styles:
            yield s


def plot_time_vs_batch_for_seq_lens(
    df: pd.DataFrame,
    model_name: str,
    out_path: str | Path,
) -> Path:
    """Plot TS/FWD/INF as function of batch_size for different seq_len.

    - All seq_len overlays on the same axes.
    - TS/FWD/INF use three distinct colors.
    - Different seq_len use different linestyles/markers.

    Args:
        df: Filtered DataFrame for a single model.
        model_name: Name of the medium model being plotted.
        out_path: Output image path.

    Returns:
        Path to the saved figure.

    """
    metrics = {
        "TS_time_ms": ("TS", "#d62728"),  # red
        "FWD_time_ms": ("FWD", "#1f77b4"),  # blue
        "INF_time_ms": ("INF", "#2ca02c"),  # green
    }

    seq_lens = sorted(df["sequence_length"].unique().tolist())
    style_iter = _cycled_styles()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    handles_metric: dict[str, Any] = {}
    handles_seq: list[Any] = []
    for sl in seq_lens:
        dsl = df[df["sequence_length"] == sl].sort_values("batch_size")
        if dsl["batch_size"].nunique() < 2:
            continue

        linestyle, marker = next(style_iter)
        for mcol, (mlab, color) in metrics.items():
            ax.plot(
                dsl["batch_size"].to_numpy(),
                dsl[mcol].to_numpy(),
                label=f"{mlab} @ seq={sl}",
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                alpha=0.9,
            )
            if mlab not in handles_metric:
                handles_metric[mlab] = Line2D([0], [0], color=color, lw=2, label=mlab)

        handles_seq.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                lw=2,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                label=f"seq_len={sl}",
            )
        )

    ax.set_title(f"{model_name}: Time vs Batch Size for Different Sequence Lengths")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    leg1 = ax.legend(
        handles=list(handles_metric.values()),
        title="Metric",
        loc="upper left",
    )
    ax.add_artist(leg1)
    ax.legend(handles=handles_seq, title="Sequence length", loc="upper right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_time_vs_seq_for_batch_sizes(
    df: pd.DataFrame,
    model_name: str,
    out_path: str | Path,
) -> Path:
    """Plot TS/FWD/INF as function of seq_len for different batch_size.

    - All batch_size overlays on the same axes.
    - TS/FWD/INF use three distinct colors.
    - Different batch_size use different linestyles/markers.

    Args:
        df: Filtered DataFrame for a single model.
        model_name: Name of the medium model being plotted.
        out_path: Output image path.

    Returns:
        Path to the saved figure.

    """
    metrics = {
        "TS_time_ms": ("TS", "#d62728"),  # red
        "FWD_time_ms": ("FWD", "#1f77b4"),  # blue
        "INF_time_ms": ("INF", "#2ca02c"),  # green
    }

    batch_sizes = sorted(df["batch_size"].unique().tolist())
    style_iter = _cycled_styles()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    handles_metric: dict[str, Any] = {}
    handles_bs: list[Any] = []
    for bs in batch_sizes:
        dbs = df[df["batch_size"] == bs].sort_values("sequence_length")
        if dbs["sequence_length"].nunique() < 2:
            continue

        linestyle, marker = next(style_iter)
        for mcol, (mlab, color) in metrics.items():
            ax.plot(
                dbs["sequence_length"].to_numpy(),
                dbs[mcol].to_numpy(),
                label=f"{mlab} @ batch={bs}",
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                alpha=0.9,
            )
            if mlab not in handles_metric:
                handles_metric[mlab] = Line2D([0], [0], color=color, lw=2, label=mlab)

        handles_bs.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                lw=2,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                label=f"batch={bs}",
            )
        )

    ax.set_title(f"{model_name}: Time vs Sequence Length for Different Batch Sizes")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Time (ms)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    leg1 = ax.legend(
        handles=list(handles_metric.values()),
        title="Metric",
        loc="upper left",
    )
    ax.add_artist(leg1)
    ax.legend(handles=handles_bs, title="Batch size", loc="upper right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ratios_vs_batch_for_seq_lens(
    df: pd.DataFrame,
    model_name: str,
    out_path: str | Path,
) -> Path:
    """Plot INF/TS and FWD/TS vs batch_size for different sequence lengths.

    - Ratios use two distinct colors.
    - Different seq_len use different linestyles/markers.

    Args:
        df: Filtered DataFrame for a single model.
        model_name: Name of the medium model being plotted.
        out_path: Output image path.

    Returns:
        Path to the saved figure.

    """
    ratio_defs = {
        "INF_over_TS": ("INF/TS", "#2ca02c"),  # green
        "FWD_over_TS": ("FWD/TS", "#1f77b4"),  # blue
    }

    # Prepare ratios
    df = df.copy()
    ts = df["TS_time_ms"].replace(0, np.nan)
    df["INF_over_TS"] = df["INF_time_ms"] / ts
    df["FWD_over_TS"] = df["FWD_time_ms"] / ts
    df = df.dropna(subset=["INF_over_TS", "FWD_over_TS"])

    seq_lens = sorted(df["sequence_length"].unique().tolist())
    style_iter = _cycled_styles()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    handles_ratio: dict[str, Any] = {}
    handles_seq: list[Any] = []
    for sl in seq_lens:
        dsl = df[df["sequence_length"] == sl].sort_values("batch_size")
        if dsl["batch_size"].nunique() < 2:
            continue

        linestyle, marker = next(style_iter)
        for col, (lab, color) in ratio_defs.items():
            ax.plot(
                dsl["batch_size"].to_numpy(),
                dsl[col].to_numpy(),
                label=f"{lab} @ seq={sl}",
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                alpha=0.9,
            )
            if lab not in handles_ratio:
                handles_ratio[lab] = Line2D([0], [0], color=color, lw=2, label=lab)

        handles_seq.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                lw=2,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                label=f"seq_len={sl}",
            )
        )

    ax.set_title(f"{model_name}: Ratios vs Batch Size for Different Sequence Lengths")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Ratio")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    leg1 = ax.legend(
        handles=list(handles_ratio.values()), title="Ratio", loc="upper left"
    )
    ax.add_artist(leg1)
    ax.legend(handles=handles_seq, title="Sequence length", loc="upper right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_ratios_vs_seq_for_batch_sizes(
    df: pd.DataFrame,
    model_name: str,
    out_path: str | Path,
) -> Path:
    """Plot INF/TS and FWD/TS vs sequence_length for different batch sizes.

    - Ratios use two distinct colors.
    - Different batch sizes use different linestyles/markers.

    Args:
        df: Filtered DataFrame for a single model.
        model_name: Name of the medium model being plotted.
        out_path: Output image path.

    Returns:
        Path to the saved figure.

    """
    ratio_defs = {
        "INF_over_TS": ("INF/TS", "#2ca02c"),  # green
        "FWD_over_TS": ("FWD/TS", "#1f77b4"),  # blue
    }

    # Prepare ratios
    df = df.copy()
    ts = df["TS_time_ms"].replace(0, np.nan)
    df["INF_over_TS"] = df["INF_time_ms"] / ts
    df["FWD_over_TS"] = df["FWD_time_ms"] / ts
    df = df.dropna(subset=["INF_over_TS", "FWD_over_TS"])

    batch_sizes = sorted(df["batch_size"].unique().tolist())
    style_iter = _cycled_styles()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    handles_ratio: dict[str, Any] = {}
    handles_bs: list[Any] = []
    for bs in batch_sizes:
        dbs = df[df["batch_size"] == bs].sort_values("sequence_length")
        if dbs["sequence_length"].nunique() < 2:
            continue

        linestyle, marker = next(style_iter)
        for col, (lab, color) in ratio_defs.items():
            ax.plot(
                dbs["sequence_length"].to_numpy(),
                dbs[col].to_numpy(),
                label=f"{lab} @ batch={bs}",
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                linewidth=1.8,
                alpha=0.9,
            )
            if lab not in handles_ratio:
                handles_ratio[lab] = Line2D([0], [0], color=color, lw=2, label=lab)

        handles_bs.append(
            Line2D(
                [0],
                [0],
                color="#555555",
                lw=2,
                linestyle=linestyle,
                marker=marker,
                markersize=5,
                label=f"batch={bs}",
            )
        )

    ax.set_title(f"{model_name}: Ratios vs Sequence Length for Different Batch Sizes")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Ratio")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)

    leg1 = ax.legend(
        handles=list(handles_ratio.values()), title="Ratio", loc="upper left"
    )
    ax.add_artist(leg1)
    ax.legend(handles=handles_bs, title="Batch size", loc="upper right")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_plots(
    csv_path: str | Path | None = None,
    model_name: str | None = None,
    output_dir: str | Path = "plots",
) -> dict[str, Path]:
    """Generate both plots for the selected medium model.

    Args:
        csv_path: Path to the timing CSV. If None, use latest found.
        model_name: Exact model_name to plot. If None, auto-pick a 'medium*'.
        output_dir: Directory to save the resulting figures.

    Returns:
        Mapping of figure identifiers to saved paths.

    """
    if csv_path is None:
        csv_path = find_latest_timing_csv(".")
    df = load_and_clean_data(csv_path)

    if model_name is None:
        model_name = pick_medium_model(df, preferred="medium1024")
    dfm = df[df["model_name"] == model_name].copy()
    if dfm.empty:
        raise ValueError(
            f"No rows found for model_name='{model_name}'. "
            "Check the CSV or choose a different model."
        )

    output_dir = Path(output_dir)
    out1 = output_dir / f"{model_name}_time_vs_batch_by_seq.png"
    out2 = output_dir / f"{model_name}_time_vs_seq_by_batch.png"
    out3 = output_dir / f"{model_name}_ratios_vs_batch_by_seq.png"
    out4 = output_dir / f"{model_name}_ratios_vs_seq_by_batch.png"

    p1 = plot_time_vs_batch_for_seq_lens(dfm, model_name, out1)
    p2 = plot_time_vs_seq_for_batch_sizes(dfm, model_name, out2)
    p3 = plot_ratios_vs_batch_for_seq_lens(dfm, model_name, out3)
    p4 = plot_ratios_vs_seq_for_batch_sizes(dfm, model_name, out4)
    return {
        "time_vs_batch_by_seq": p1,
        "time_vs_seq_by_batch": p2,
        "ratios_vs_batch_by_seq": p3,
        "ratios_vs_seq_by_batch": p4,
    }


def _build_synthetic_csv(path: Path) -> None:
    """Create a small synthetic CSV compatible with timing schema.

    Args:
        path: Where to write the synthetic CSV file.

    """
    rows = []
    models = ["small512", "medium1024"]
    seqs = [256, 512, 1024]
    bss = [1, 16, 64]

    rng = np.random.default_rng(0)
    for m in models:
        for sl in seqs:
            for bs in bss:
                base = {
                    "INF_time_ms": 0.06 * sl + 0.7 * bs,
                    "FWD_time_ms": 0.08 * sl + 0.9 * bs,
                    "TS_time_ms": 0.12 * sl + 1.2 * bs,
                }
                noise = rng.normal(0.0, 2.0, size=3)
                rows.append(
                    {
                        "model_name": m,
                        "batch_size": bs,
                        "sequence_length": sl,
                        "total_params_millions": 100.0,
                        "INF_time_ms": base["INF_time_ms"] + float(noise[0]),
                        "INF_time_s": 0.0,
                        "FWD_time_ms": base["FWD_time_ms"] + float(noise[1]),
                        "FWD_time_s": 0.0,
                        "TS_time_ms": base["TS_time_ms"] + float(noise[2]),
                        "TS_time_s": 0.0,
                        "status": "success",
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _run_module_tests(tmp_dir: Path) -> None:
    """Run basic self-tests for this module.

    Args:
        tmp_dir: Temporary directory for artifacts.

    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_dir / "timing_results_test.csv"
    _build_synthetic_csv(csv_path)
    print(f"Synthetic CSV created at: {csv_path}")

    df = load_and_clean_data(csv_path)
    assert not df.empty, "DataFrame should not be empty after loading."
    print("Test 1 passed: CSV loading produced non-empty DataFrame.")

    model = pick_medium_model(df, preferred="medium1024")
    assert model == "medium1024", "Preferred medium1024 should be selected."
    print("Test 2 passed: pick_medium_model selects preferred model.")

    out_dir = tmp_dir / "plots"
    results = run_plots(csv_path=csv_path, model_name=model, output_dir=out_dir)
    p1 = results["time_vs_batch_by_seq"]
    p2 = results["time_vs_seq_by_batch"]
    assert p1.exists() and p1.stat().st_size > 0, "First plot not saved."
    print("Test 3 passed: time_vs_batch_by_seq plot saved and non-empty.")
    assert p2.exists() and p2.stat().st_size > 0, "Second plot not saved."
    print("Test 4 passed: time_vs_seq_by_batch plot saved and non-empty.")

    p3 = out_dir / "medium1024_ratios_vs_batch_by_seq.png"
    p4 = out_dir / "medium1024_ratios_vs_seq_by_batch.png"
    assert p3.exists() and p3.stat().st_size > 0, "Ratios (batch) plot not saved."
    print("Test 5 passed: ratios_vs_batch_by_seq plot saved and non-empty.")
    assert p4.exists() and p4.stat().st_size > 0, "Ratios (seq) plot not saved."
    print("Test 6 passed: ratios_vs_seq_by_batch plot saved and non-empty.")


def _loglog_fit(
    x: Float[np.ndarray, " n"], y: Float[np.ndarray, " n"]
) -> tuple[float, float, float]:
    """Fit y ≈ c * x^a in log–log space; return (a, log_c, mse).

    Args:
        x: Positive x values.
        y: Positive y values.

    Returns:
        Slope a, intercept log_c, and mean squared error in log space.

    """
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    a, b = np.polyfit(lx, ly, 1)
    pred = a * lx + b
    mse = float(np.mean((ly - pred) ** 2))
    return float(a), float(b), mse


def _precompute_segment_sse(
    x: Float[np.ndarray, " n"], y: Float[np.ndarray, " n"], min_points: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Precompute slope/intercept SSE for all valid segments in log–log space.

    Args:
        x: Batch sizes (positive), sorted ascending.
        y: TS times (positive).
        min_points: Minimum points per segment.

    Returns:
        (sse, slope) arrays of shape [n, n] for segments [i, j] inclusive.
    """
    n = len(x)
    lx = np.log(x)
    ly = np.log(y)
    sse = np.full((n, n), np.inf, dtype=float)
    slope = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(i + min_points - 1, n):
            xx = lx[i : j + 1]
            yy = ly[i : j + 1]
            a, b = np.polyfit(xx, yy, 1)
            pred = a * xx + b
            sse[i, j] = float(np.sum((yy - pred) ** 2))
            slope[i, j] = float(a)
    return sse, slope


def piecewise_loglog_fit(
    x: Sequence[float],
    y: Sequence[float],
    max_segments: int = 3,
    min_points: int = 2,
) -> dict:
    """Fit up to max_segments piecewise lines in log–log; DP for best breakpoints.

    Args:
        x: Positive, increasing batch sizes.
        y: Positive TS times.
        max_segments: Maximum number of segments to fit.
        min_points: Minimum points per segment.

    Returns:
        Dict with keys:
          - 'segments': list of dicts with 'start', 'end', 'slope', 'indices'
          - 'break_batches': list of batch sizes where breaks occur (excluding start)
          - 'total_sse': total SSE in log–log fit

    """
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    assert np.all(xv > 0) and np.all(yv > 0)

    n = len(xv)
    sse_tbl, slope_tbl = _precompute_segment_sse(xv, yv, min_points=min_points)

    K = min(max_segments, max(1, n // min_points))
    dp = np.full((K + 1, n), np.inf, dtype=float)
    prev = np.full((K + 1, n), -1, dtype=int)

    # 1 segment base case
    for j in range(min_points - 1, n):
        dp[1, j] = sse_tbl[0, j]

    for k in range(2, K + 1):
        for j in range(min_points * k - 1, n):
            best_cost = np.inf
            best_i = -1
            # last segment is [i, j]
            i_min = (k - 1) * min_points - 1
            for i in range(i_min, j - (min_points - 1)):
                cost = dp[k - 1, i] + sse_tbl[i + 1, j]
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
            dp[k, j] = best_cost
            prev[k, j] = best_i

    # choose best K' <= K
    best_k = 1
    best_val = dp[1, n - 1]
    for k in range(2, K + 1):
        if dp[k, n - 1] < best_val:
            best_k = k
            best_val = dp[k, n - 1]

    # backtrack
    segs: list[dict] = []
    j = n - 1
    k = best_k
    while k >= 1:
        i = prev[k, j]
        start = 0 if i < 0 else i + 1
        end = j
        a = slope_tbl[start, end]
        segs.append(
            {
                "start": start,
                "end": end,
                "slope": float(a),
                "indices": list(range(start, end + 1)),
            }
        )
        j = i
        k -= 1
    segs.reverse()

    break_batches = [xv[seg["start"]] for seg in segs[1:]]
    return {
        "segments": segs,
        "break_batches": break_batches,
        "total_sse": float(best_val),
    }


def analyze_ts_scaling_by_batch(
    df: pd.DataFrame,
    model_name: str,
    output_dir: str | Path = "plots/analysis",
    max_segments: int = 3,
    min_points: int = 2,
    save_plots: bool = True,
) -> None:
    """Analyze TS vs batch_size per seq_len: fit piecewise log–log segments.

    Args:
        df: Cleaned dataframe filtered to a single model.
        model_name: Model name (for titles).
        output_dir: Where to save diagnostic figures.
        max_segments: Upper bound on number of segments.
        min_points: Minimum points per segment.
        save_plots: Whether to save per-seq_len plots.

    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for sl in sorted(df["sequence_length"].unique()):
        d = df[df["sequence_length"] == sl].sort_values("batch_size")
        if len(d) < max(2 * min_points, 3):
            continue

        bs = d["batch_size"].to_numpy(dtype=float)
        ts = d["TS_time_ms"].to_numpy(dtype=float)
        fit = piecewise_loglog_fit(
            bs, ts, max_segments=max_segments, min_points=min_points
        )

        # Print summary
        print(f"\n[TS scaling] {model_name} @ seq_len={sl}")
        for idx, seg in enumerate(fit["segments"], 1):
            b0 = int(bs[seg["start"]])
            b1 = int(bs[seg["end"]])
            a = seg["slope"]
            print(
                f"  Segment {idx}: batch {b0}..{b1}  slope a={a:.2f}  n={len(seg['indices'])}"
            )
        if fit["break_batches"]:
            brks = ", ".join(str(int(b)) for b in fit["break_batches"])
            print(f"  Breakpoints near batch sizes: {brks}")

        if save_plots:
            fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
            ax.plot(bs, ts, "ko-", label="TS data", alpha=0.7)
            # overlay fits in linear coords for readability
            lx = np.log(bs)
            for seg in fit["segments"]:
                i0, i1 = seg["start"], seg["end"]
                a, b = _loglog_fit(bs[i0 : i1 + 1], ts[i0 : i1 + 1])[:2]
                xs = bs[i0 : i1 + 1]
                ys = np.exp(b) * (xs**a)
                ax.plot(xs, ys, "-", lw=2, label=f"fit a={a:.2f}")
            ax.set_title(
                f"{model_name} TS vs batch (seq={sl}) — piecewise log–log fits"
            )
            ax.set_xlabel("Batch size")
            ax.set_ylabel("TS time (ms)")
            ax.grid(True, ls="--", alpha=0.3)
            ax.legend()
            fig.savefig(
                out_dir / f"{model_name}_TS_vs_batch_seq{sl}_analysis.png", dpi=150
            )
            plt.close(fig)


def main() -> None:
    """CLI entry point to produce requested timing plots."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate timing line plots for TS/FWD/INF vs batch size or "
            "sequence length for the medium model."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="timing_results_20250808_220829.csv",
        help="Path to timing_results_*.csv.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="medium1024",
        help="Exact model_name to plot (default: medium1024).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="plots",
        help="Directory to save figures (default: plots).",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run internal module tests with synthetic data.",
    )
    parser.add_argument(
        "--analyze-ts",
        action="store_true",
        help="Analyze TS vs batch_size per seq_len with piecewise log–log fits.",
    )
    parser.add_argument(
        "--max-segments", type=int, default=3, help="Max segments for piecewise fits."
    )
    parser.add_argument(
        "--min-points", type=int, default=2, help="Min points per segment."
    )
    parser.add_argument(
        "--analysis-outdir",
        type=str,
        default="plots/analysis",
        help="Directory for analysis plots.",
    )
    args = parser.parse_args()

    if args.run_tests:
        _run_module_tests(Path(".") / "tmp_time_line_plots")
        return

    results = run_plots(
        csv_path=args.csv, model_name=args.model, output_dir=args.outdir
    )
    print(f"Saved: {results['time_vs_batch_by_seq']}")
    print(f"Saved: {results['time_vs_seq_by_batch']}")

    if args.analyze_ts:
        df_all = load_and_clean_data(args.csv)
        dfm_all = df_all[df_all["model_name"] == args.model].copy()
        analyze_ts_scaling_by_batch(
            dfm_all,
            args.model,
            output_dir=args.analysis_outdir,
            max_segments=args.max_segments,
            min_points=args.min_points,
            save_plots=True,
        )


if __name__ == "__main__":
    main()
