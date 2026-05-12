#!/usr/bin/env python3

import argparse
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "Sign-Iteration",
    "Box-PGA",
    "LR-SDP r=2",
    "LR-SDP r=4",
    "SDP+GW",
]

COLOR_MAP = {
    "Sign-Iteration": "#4E79A7",
    "Box-PGA": "#F28E2B",
    "LR-SDP r=2": "#59A14F",
    "LR-SDP r=4": "#B07AA1",
    "SDP+GW": "#8C564B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot ablation study figures from solver and rounding CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "results" / "ablation_studies",
        help="Directory containing solver_results.csv and rounding_trace.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "ablation_studies",
        help="Directory for PNG and summary CSV outputs.",
    )
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--group-count", type=int, default=8)
    parser.add_argument("--group-max-size", type=int, default=32)
    parser.add_argument("--rounding-group-size", type=int, default=8)
    parser.add_argument("--sampling-seed", type=int, default=20260512)
    return parser.parse_args()


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )


def read_inputs(input_dir):
    solver_path = input_dir / "solver_results.csv"
    rounding_path = input_dir / "rounding_trace.csv"
    if not solver_path.exists():
        raise FileNotFoundError(f"Missing {solver_path}")
    if not rounding_path.exists():
        raise FileNotFoundError(f"Missing {rounding_path}")

    solver = pd.read_csv(solver_path)
    rounding = pd.read_csv(rounding_path)

    for frame in (solver, rounding):
        for col in frame.columns:
            if col in {
                "T",
                "d",
                "rank",
                "seed_index",
                "seed",
                "binary_objective",
                "continuous_objective",
                "relaxed_objective",
                "solve_seconds",
                "rounding_seconds",
                "total_seconds",
                "iterations",
                "best_iteration",
                "best_restart",
                "grad_norm",
                "round_restarts",
                "maxrss_mb",
                "rounding_index",
                "sample_objective",
                "best_objective",
                "rounding_elapsed_seconds",
            }:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    solver = solver[~solver["status"].astype(str).str.startswith("failed:")].copy()
    return solver, rounding


def sdp_reference(solver, T):
    rows = solver[(solver["T"] == T) & (solver["method_label"] == "SDP+GW")]
    if rows.empty:
        raise ValueError(f"No SDP+GW solver row found for T={T}")
    return float(rows["binary_objective"].max())


def common_seed_indices(solver, T, labels):
    seed_sets = []
    subset = solver[solver["T"] == T]
    for label in labels:
        rows = subset[subset["method_label"] == label]
        if rows.empty:
            raise ValueError(f"No rows found for {label} at T={T}")
        seed_sets.append(set(rows["seed_index"].dropna().astype(int)))
    common = sorted(set.intersection(*seed_sets))
    if not common:
        raise ValueError("No common seed indices across requested methods")
    return np.array(common, dtype=int)


def random_groups(seed_indices, group_count, group_size, rng):
    needed = group_count * group_size
    if needed > len(seed_indices):
        raise ValueError(
            f"Need {needed} seeds for {group_count} groups of size {group_size}, "
            f"but only {len(seed_indices)} are available."
        )
    sampled = rng.choice(seed_indices, size=needed, replace=False)
    return sampled.reshape(group_count, group_size)


def values_by_seed(frame, value_col):
    return (
        frame.sort_values("seed_index")
        .set_index("seed_index")[value_col]
        .astype(float)
        .to_dict()
    )


def group_max_stats(frame, groups, value_col):
    lookup = values_by_seed(frame, value_col)
    maxima = np.array(
        [max(lookup[int(seed)] for seed in group) for group in groups],
        dtype=float,
    )
    std = float(maxima.std(ddof=1)) if len(maxima) > 1 else 0.0
    return float(maxima.mean()), std, maxima


def set_quality_ylim(ax, values):
    finite = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if len(finite) == 0:
        return
    ymin = max(0.0, np.floor((finite.min() - 0.01) * 100) / 100)
    ymax = min(1.02, np.ceil((finite.max() + 0.01) * 100) / 100)
    if ymax - ymin < 0.04:
        center = 0.5 * (ymin + ymax)
        ymin = max(0.0, center - 0.02)
        ymax = min(1.02, center + 0.02)
    ax.set_ylim(ymin, ymax)


def make_group_size_ablation(
    solver, output_dir, T, group_count, max_group_size, sampling_seed
):
    labels = ["Sign-Iteration", "Box-PGA", "LR-SDP r=2", "LR-SDP r=4"]
    subset = solver[solver["T"] == T]
    ref = sdp_reference(solver, T)
    seeds = common_seed_indices(solver, T, labels)
    rng = np.random.default_rng(sampling_seed)
    group_sizes = np.arange(1, max_group_size + 1)

    groups_by_size = {
        int(size): random_groups(seeds, group_count, int(size), rng)
        for size in group_sizes
    }

    records = []
    all_quality_values = []
    fig, ax = plt.subplots(figsize=(11, 6.4))

    for label in labels:
        rows = subset[subset["method_label"] == label]
        means = []
        stds = []
        for size in group_sizes:
            groups = groups_by_size[int(size)]
            mean_obj, std_obj, maxima = group_max_stats(
                rows, groups, "binary_objective"
            )
            quality = mean_obj / ref
            quality_std = std_obj / ref
            means.append(quality)
            stds.append(quality_std)
            all_quality_values.extend([quality - quality_std, quality + quality_std])
            records.append(
                {
                    "T": T,
                    "method_label": label,
                    "group_size": int(size),
                    "groups_used": int(group_count),
                    "sampling_seed": int(sampling_seed),
                    "sampled_seed_indices": " ".join(
                        str(int(seed)) for seed in groups.ravel()
                    ),
                    "mean_group_max_objective": mean_obj,
                    "std_group_max_objective": std_obj,
                    "quality_vs_sdp_gw": quality,
                    "quality_std_vs_sdp_gw": quality_std,
                }
            )

        means = np.array(means)
        stds = np.array(stds)
        color = COLOR_MAP[label]
        ax.fill_between(
            group_sizes,
            means - stds,
            means + stds,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(group_sizes, means, label=label, color=color, linewidth=2.2, alpha=0.8)

    summary = pd.DataFrame(records)
    summary_path = output_dir / "group_size_ablation.csv"
    summary.to_csv(summary_path, index=False)

    ax.set_title(f"T={T} Group-Size Ablation")
    ax.set_xlabel("Seed group size")
    ax.set_ylabel("Quality")
    ax.set_xlim(1, max_group_size)
    ax.set_xticks([x for x in [1, 4, 8, 16, 24, 32] if x <= max_group_size])
    set_quality_ylim(ax, all_quality_values)
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")

    output_path = output_dir / "group_size_ablation.png"
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return summary_path, output_path


def rounding_groups(solver, T, labels, group_size, sampling_seed):
    seeds = common_seed_indices(solver, T, labels)
    group_count = len(seeds) // group_size
    if group_count < 1:
        raise ValueError(f"Not enough seeds for rounding group size {group_size}")
    rng = np.random.default_rng(sampling_seed)
    sampled = rng.choice(seeds, size=group_count * group_size, replace=False)
    return sampled.reshape(group_count, group_size)


def make_rounding_ablation(
    solver, rounding, output_dir, T, group_size, sampling_seed
):
    labels = ["LR-SDP r=2", "LR-SDP r=4"]
    ref = sdp_reference(solver, T)
    round_subset = rounding[
        (rounding["T"] == T)
        & (rounding["method_label"].isin(labels + ["SDP+GW"]))
    ]
    max_round = int(round_subset["rounding_index"].max())
    rounds = np.arange(1, max_round + 1)
    groups = rounding_groups(solver, T, labels, group_size, sampling_seed)

    records = []
    all_quality_values = []
    fig, ax = plt.subplots(figsize=(11, 6.4))

    for label in labels:
        rows = rounding[(rounding["T"] == T) & (rounding["method_label"] == label)]
        means = []
        stds = []
        for k in rounds:
            rows_k = rows[rows["rounding_index"] == k]
            mean_obj, std_obj, maxima = group_max_stats(
                rows_k, groups, "best_objective"
            )
            quality = mean_obj / ref
            quality_std = std_obj / ref
            means.append(quality)
            stds.append(quality_std)
            all_quality_values.extend([quality - quality_std, quality + quality_std])
            records.append(
                {
                    "T": T,
                    "method_label": label,
                    "group_size": group_size,
                    "groups_used": int(groups.shape[0]),
                    "roundings": int(k),
                    "sampling_seed": int(sampling_seed),
                    "mean_group_max_objective": mean_obj,
                    "std_group_max_objective": std_obj,
                    "quality_vs_sdp_gw": quality,
                    "quality_std_vs_sdp_gw": quality_std,
                }
            )

        means = np.array(means)
        stds = np.array(stds)
        color = COLOR_MAP[label]
        ax.fill_between(
            rounds,
            means - stds,
            means + stds,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(rounds, means, label=label, color=color, linewidth=2.2, alpha=0.8)

    sdp = rounding[(rounding["T"] == T) & (rounding["method_label"] == "SDP+GW")]
    sdp_line = (
        sdp.sort_values("rounding_index")
        .drop_duplicates("rounding_index", keep="last")
        .set_index("rounding_index")
        .reindex(rounds)
    )
    sdp_quality = sdp_line["best_objective"].to_numpy(dtype=float) / ref
    all_quality_values.extend(sdp_quality.tolist())
    ax.plot(
        rounds,
        sdp_quality,
        label="SDP+GW",
        color=COLOR_MAP["SDP+GW"],
        linewidth=2.4,
        alpha=0.8,
    )
    for k, quality in zip(rounds, sdp_quality):
        records.append(
            {
                "T": T,
                "method_label": "SDP+GW",
                "group_size": np.nan,
                "groups_used": 1,
                "roundings": int(k),
                "sampling_seed": int(sampling_seed),
                "mean_group_max_objective": quality * ref,
                "std_group_max_objective": 0.0,
                "quality_vs_sdp_gw": quality,
                "quality_std_vs_sdp_gw": 0.0,
            }
        )

    summary = pd.DataFrame(records)
    summary_path = output_dir / f"rounding_ablation_group{group_size}.csv"
    summary.to_csv(summary_path, index=False)

    ax.set_title(f"T={T} GW Rounding Ablation")
    ax.set_xlabel("Number of GW roundings")
    ax.set_ylabel("Quality")
    ax.set_xlim(1, 64)
    set_quality_ylim(ax, all_quality_values)
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")

    output_path = output_dir / f"rounding_ablation_group{group_size}.png"
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return summary_path, output_path


def main():
    args = parse_args()
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    solver, rounding = read_inputs(args.input_dir)

    print(f"Loaded solver rows: {len(solver):,}")
    print(f"Loaded rounding rows: {len(rounding):,}")

    outputs = []
    outputs.extend(
        make_group_size_ablation(
            solver,
            args.output_dir,
            args.T,
            args.group_count,
            args.group_max_size,
            args.sampling_seed,
        )
    )
    outputs.extend(
        make_rounding_ablation(
            solver,
            rounding,
            args.output_dir,
            args.T,
            args.rounding_group_size,
            args.sampling_seed,
        )
    )

    print("Wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
