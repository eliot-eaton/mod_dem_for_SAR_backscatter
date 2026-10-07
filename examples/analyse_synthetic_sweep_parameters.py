#!/usr/bin/env python3
"""
Analyse synthetic DEM sweep metadata together with continuous-edge inversion scores.

Inputs
------
1. A directory containing per-model JSON metadata files, e.g.
       synthetic_sweep/4053.json
2. The peak_model_ranking.csv produced by the inversion.

The script:
- flattens model metadata into one table;
- joins metadata to inversion scores by model ID;
- characterises the best-performing models;
- plots the distribution of all tried models vs good models;
- plots centre and semi-axis parameter coverage;
- plots parameter value vs inversion RMSE;
- maps pairwise sampling density to expose sparse / untried regions;
- writes pairwise bin tables that can guide the next sweep.

Only numpy, pandas and matplotlib are required.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CORE_PARAMETERS = [
    "center_x_m",
    "center_y_m",
    "center_z_m",
    "semi_axis_x_m",
    "semi_axis_y_m",
    "semi_axis_z_m",
    "depth_below_reference_surface_m",
]

PAIR_PARAMETERS = [
    ("center_x_m", "center_y_m"),
    ("semi_axis_x_m", "semi_axis_y_m"),
    ("semi_axis_x_m", "semi_axis_z_m"),
    ("semi_axis_y_m", "semi_axis_z_m"),
    ("center_z_m", "semi_axis_z_m"),
    ("depth_below_reference_surface_m", "semi_axis_z_m"),
]


def normalise_id(value) -> str:
    """Return a join-safe model ID."""
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except Exception:
        return text


def _triplet(values, prefix: str) -> Dict[str, float]:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return {
            f"{prefix}_x_m": np.nan,
            f"{prefix}_y_m": np.nan,
            f"{prefix}_z_m": np.nan,
        }
    return {
        f"{prefix}_x_m": float(values[0]),
        f"{prefix}_y_m": float(values[1]),
        f"{prefix}_z_m": float(values[2]),
    }


def flatten_metadata(path: Path, shape_index: int = 0) -> Dict[str, object]:
    payload = json.loads(path.read_text())

    record: Dict[str, object] = {
        "run_id": normalise_id(payload.get("id", path.stem)),
        "metadata_json": str(path),
        "source_dem": payload.get("source_dem"),
        "output_dem": payload.get("output_dem"),
        "working_projected_crs": payload.get("working_projected_crs"),
    }

    shapes = payload.get("shapes", [])
    record["n_shapes"] = len(shapes)

    if not shapes or shape_index >= len(shapes):
        record["shape_type"] = None
        record["interaction"] = None
        return record

    shape = shapes[shape_index]
    record["shape_type"] = shape.get("type")
    record["interaction"] = shape.get("interaction")

    center = shape.get("center_xyz_m")
    if isinstance(center, (list, tuple)) and len(center) >= 3:
        record["center_x_m"] = float(center[0])
        record["center_y_m"] = float(center[1])
        record["center_z_m"] = float(center[2])
    else:
        record.update(
            {
                "center_x_m": np.nan,
                "center_y_m": np.nan,
                "center_z_m": np.nan,
            }
        )

    axes = shape.get("semi_axes_m")
    if isinstance(axes, (list, tuple)) and len(axes) >= 3:
        record["semi_axis_x_m"] = float(axes[0])
        record["semi_axis_y_m"] = float(axes[1])
        record["semi_axis_z_m"] = float(axes[2])
    else:
        record.update(
            {
                "semi_axis_x_m": np.nan,
                "semi_axis_y_m": np.nan,
                "semi_axis_z_m": np.nan,
            }
        )

    rotation = shape.get("rotation_deg")
    if isinstance(rotation, (list, tuple)) and len(rotation) >= 3:
        record["rotation_x_deg"] = float(rotation[0])
        record["rotation_y_deg"] = float(rotation[1])
        record["rotation_z_deg"] = float(rotation[2])

    sweep = shape.get("sweep_parameters", {}) or {}
    for key in (
        "reference_x_m",
        "reference_y_m",
        "reference_surface_z_m",
        "depth_below_reference_surface_m",
        "x_m",
        "y_m",
        "z_m",
    ):
        if key in sweep:
            try:
                record[key] = float(sweep[key])
            except Exception:
                record[key] = sweep[key]

    surface = shape.get("surface_intersection", {}) or {}
    for key in ("changed_pixels_projected", "modifies_surface"):
        if key in surface:
            record[key] = surface[key]

    validation = payload.get("validation", {}) or {}
    for key in (
        "changed_pixels",
        "changed_fraction",
        "changed_percent",
        "min_change_m",
        "max_change_m",
        "mean_change_m",
        "nan_pixels",
    ):
        if key in validation:
            record[f"validation_{key}"] = validation[key]

    return record


def read_metadata_directory(
    metadata_dir: Path,
    *,
    shape_index: int = 0,
) -> pd.DataFrame:
    paths = sorted(metadata_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(
            f"No JSON files found in {metadata_dir}"
        )

    records: List[Dict[str, object]] = []
    failures = []

    for path in paths:
        try:
            records.append(
                flatten_metadata(path, shape_index=shape_index)
            )
        except Exception as exc:
            failures.append((path.name, str(exc)))

    if not records:
        raise RuntimeError("No metadata JSON files could be read.")

    df = pd.DataFrame(records)

    if failures:
        print(
            f"WARNING: {len(failures)} metadata files could not be read."
        )
        for name, message in failures[:10]:
            print(f"  {name}: {message}")
        if len(failures) > 10:
            print("  ...")

    return df


def read_ranking(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"run_id": str})

    if "run_id" not in df.columns:
        raise ValueError(
            f"{path} has no 'run_id' column."
        )
    if "rmse_filtered_m" not in df.columns:
        raise ValueError(
            f"{path} has no 'rmse_filtered_m' column."
        )

    df["run_id"] = df["run_id"].map(normalise_id)
    df["rmse_filtered_m"] = pd.to_numeric(
        df["rmse_filtered_m"],
        errors="coerce",
    )

    return df


def select_rankable(df: pd.DataFrame) -> pd.DataFrame:
    mask = np.isfinite(df["rmse_filtered_m"])

    if "status" in df.columns:
        mask &= df["status"].astype(str).eq("ok")

    return df.loc[mask].copy()


def select_good_models(
    df: pd.DataFrame,
    *,
    top_n: Optional[int],
    top_fraction: Optional[float],
    rmse_max: Optional[float],
) -> pd.DataFrame:
    rankable = select_rankable(df).sort_values(
        "rmse_filtered_m",
        ascending=True,
    )

    if rmse_max is not None:
        good = rankable[
            rankable["rmse_filtered_m"] <= rmse_max
        ].copy()
    elif top_fraction is not None:
        if not (0 < top_fraction <= 1):
            raise ValueError(
                "--good-fraction must be in (0, 1]."
            )
        n = max(
            1,
            int(math.ceil(len(rankable) * top_fraction)),
        )
        good = rankable.head(n).copy()
    else:
        n = 100 if top_n is None else int(top_n)
        if n < 1:
            raise ValueError("--good-top-n must be >= 1.")
        good = rankable.head(n).copy()

    return good


def available_parameters(df: pd.DataFrame) -> List[str]:
    out = []
    for parameter in CORE_PARAMETERS:
        if parameter not in df.columns:
            continue
        values = pd.to_numeric(df[parameter], errors="coerce")
        if np.isfinite(values).sum() >= 2:
            out.append(parameter)
    return out


def characterise_parameters(
    all_models: pd.DataFrame,
    good_models: pd.DataFrame,
    parameters: Sequence[str],
) -> pd.DataFrame:
    rows = []

    for p in parameters:
        all_v = pd.to_numeric(
            all_models[p],
            errors="coerce",
        ).dropna()
        good_v = pd.to_numeric(
            good_models[p],
            errors="coerce",
        ).dropna()

        if all_v.empty:
            continue

        all_std = float(all_v.std(ddof=0))
        all_median = float(all_v.median())
        good_median = (
            float(good_v.median())
            if not good_v.empty
            else np.nan
        )

        rows.append(
            {
                "parameter": p,
                "n_all": len(all_v),
                "n_good": len(good_v),
                "all_min": float(all_v.min()),
                "all_q10": float(all_v.quantile(0.10)),
                "all_q25": float(all_v.quantile(0.25)),
                "all_median": all_median,
                "all_q75": float(all_v.quantile(0.75)),
                "all_q90": float(all_v.quantile(0.90)),
                "all_max": float(all_v.max()),
                "good_min": (
                    float(good_v.min())
                    if not good_v.empty
                    else np.nan
                ),
                "good_q10": (
                    float(good_v.quantile(0.10))
                    if not good_v.empty
                    else np.nan
                ),
                "good_q25": (
                    float(good_v.quantile(0.25))
                    if not good_v.empty
                    else np.nan
                ),
                "good_median": good_median,
                "good_q75": (
                    float(good_v.quantile(0.75))
                    if not good_v.empty
                    else np.nan
                ),
                "good_q90": (
                    float(good_v.quantile(0.90))
                    if not good_v.empty
                    else np.nan
                ),
                "good_max": (
                    float(good_v.max())
                    if not good_v.empty
                    else np.nan
                ),
                "good_minus_all_median": (
                    good_median - all_median
                    if np.isfinite(good_median)
                    else np.nan
                ),
                "good_median_shift_all_sd": (
                    (good_median - all_median) / all_std
                    if np.isfinite(good_median) and all_std > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def make_histograms(
    all_models: pd.DataFrame,
    good_models: pd.DataFrame,
    parameters: Sequence[str],
    output: Path,
    bins: int,
):
    n = len(parameters)
    if n == 0:
        return

    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.6 * ncols, 3.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, p in zip(axes.ravel(), parameters):
        all_v = pd.to_numeric(
            all_models[p],
            errors="coerce",
        ).dropna()
        good_v = pd.to_numeric(
            good_models[p],
            errors="coerce",
        ).dropna()

        ax.hist(
            all_v,
            bins=bins,
            alpha=0.55,
            label=f"All ({len(all_v)})",
        )
        if not good_v.empty:
            ax.hist(
                good_v,
                bins=bins,
                alpha=0.55,
                label=f"Good ({len(good_v)})",
            )

        ax.set_xlabel(p)
        ax.set_ylabel("Number of models")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(
        "Parameter distributions: all tried models vs good models"
    )
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def scatter_all_good(
    all_models: pd.DataFrame,
    good_models: pd.DataFrame,
    xcol: str,
    ycol: str,
    output: Path,
    title: str,
):
    if xcol not in all_models or ycol not in all_models:
        return

    x_all = pd.to_numeric(
        all_models[xcol],
        errors="coerce",
    )
    y_all = pd.to_numeric(
        all_models[ycol],
        errors="coerce",
    )
    x_good = pd.to_numeric(
        good_models[xcol],
        errors="coerce",
    )
    y_good = pd.to_numeric(
        good_models[ycol],
        errors="coerce",
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 6.2),
        constrained_layout=True,
    )

    ax.scatter(
        x_all,
        y_all,
        s=12,
        alpha=0.25,
        label="All tried models",
    )
    ax.scatter(
        x_good,
        y_good,
        s=30,
        alpha=0.85,
        label="Good models",
    )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)

    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_semi_axes_plot(
    all_models: pd.DataFrame,
    good_models: pd.DataFrame,
    output: Path,
):
    pairs = [
        ("semi_axis_x_m", "semi_axis_y_m"),
        ("semi_axis_x_m", "semi_axis_z_m"),
        ("semi_axis_y_m", "semi_axis_z_m"),
    ]

    if not all(
        a in all_models.columns and b in all_models.columns
        for a, b in pairs
    ):
        return

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        constrained_layout=True,
    )

    for ax, (a, b) in zip(axes, pairs):
        ax.scatter(
            pd.to_numeric(all_models[a], errors="coerce"),
            pd.to_numeric(all_models[b], errors="coerce"),
            s=10,
            alpha=0.22,
            label="All",
        )
        ax.scatter(
            pd.to_numeric(good_models[a], errors="coerce"),
            pd.to_numeric(good_models[b], errors="coerce"),
            s=30,
            alpha=0.85,
            label="Good",
        )
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.grid(alpha=0.18)

    axes[0].legend(frameon=False)
    fig.suptitle(
        "Ellipsoid semi-axis parameter space"
    )
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def make_parameter_vs_rmse(
    rankable: pd.DataFrame,
    good_models: pd.DataFrame,
    parameters: Sequence[str],
    output: Path,
):
    n = len(parameters)
    if n == 0:
        return

    ncols = 3
    nrows = int(math.ceil(n / ncols))

    good_ids = set(good_models["run_id"].astype(str))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 3.6 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, p in zip(axes.ravel(), parameters):
        x = pd.to_numeric(rankable[p], errors="coerce")
        y = pd.to_numeric(
            rankable["rmse_filtered_m"],
            errors="coerce",
        )

        good_mask = rankable["run_id"].astype(str).isin(
            good_ids
        )

        ax.scatter(
            x[~good_mask],
            y[~good_mask],
            s=10,
            alpha=0.25,
            label="Other ranked models",
        )
        ax.scatter(
            x[good_mask],
            y[good_mask],
            s=28,
            alpha=0.85,
            label="Good models",
        )

        ax.set_xlabel(p)
        ax.set_ylabel("RMSE (m)")
        ax.grid(alpha=0.18)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    axes.ravel()[0].legend(frameon=False)
    fig.suptitle(
        "Model parameters versus continuous-edge inversion RMSE"
    )
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def pairwise_bin_table(
    df: pd.DataFrame,
    good_ids: set,
    xcol: str,
    ycol: str,
    bins: int,
) -> pd.DataFrame:
    work = df[
        ["run_id", xcol, ycol, "rmse_filtered_m"]
    ].copy()

    for col in (xcol, ycol, "rmse_filtered_m"):
        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

    work = work.dropna()

    if work.empty:
        return pd.DataFrame()

    x_edges = np.linspace(
        work[xcol].min(),
        work[xcol].max(),
        bins + 1,
    )
    y_edges = np.linspace(
        work[ycol].min(),
        work[ycol].max(),
        bins + 1,
    )

    # Handle constant parameters.
    if np.allclose(x_edges[0], x_edges[-1]) or np.allclose(
        y_edges[0], y_edges[-1]
    ):
        return pd.DataFrame()

    work["x_bin"] = pd.cut(
        work[xcol],
        bins=x_edges,
        include_lowest=True,
        duplicates="drop",
    )
    work["y_bin"] = pd.cut(
        work[ycol],
        bins=y_edges,
        include_lowest=True,
        duplicates="drop",
    )
    work["is_good"] = work["run_id"].astype(str).isin(
        good_ids
    )

    rows = []

    for (xb, yb), group in work.groupby(
        ["x_bin", "y_bin"],
        observed=False,
        dropna=False,
    ):
        if pd.isna(xb) or pd.isna(yb):
            continue

        if len(group):
            median_rmse = float(
                group["rmse_filtered_m"].median()
            )
            best_rmse = float(
                group["rmse_filtered_m"].min()
            )
            good_count = int(group["is_good"].sum())
        else:
            median_rmse = np.nan
            best_rmse = np.nan
            good_count = 0

        rows.append(
            {
                "x_parameter": xcol,
                "y_parameter": ycol,
                "x_low": float(xb.left),
                "x_high": float(xb.right),
                "x_mid": float(
                    (xb.left + xb.right) / 2
                ),
                "y_low": float(yb.left),
                "y_high": float(yb.right),
                "y_mid": float(
                    (yb.left + yb.right) / 2
                ),
                "model_count": int(len(group)),
                "good_model_count": good_count,
                "median_rmse_m": median_rmse,
                "best_rmse_m": best_rmse,
            }
        )

    return pd.DataFrame(rows)


def make_occupancy_plot(
    rankable: pd.DataFrame,
    good_models: pd.DataFrame,
    pairs: Sequence[Tuple[str, str]],
    output: Path,
    bins: int,
):
    usable = [
        (a, b)
        for a, b in pairs
        if a in rankable.columns
        and b in rankable.columns
        and pd.to_numeric(
            rankable[a],
            errors="coerce",
        ).nunique() > 1
        and pd.to_numeric(
            rankable[b],
            errors="coerce",
        ).nunique() > 1
    ]

    if not usable:
        return

    ncols = 2
    nrows = int(math.ceil(len(usable) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 5.5 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, (xcol, ycol) in zip(
        axes.ravel(),
        usable,
    ):
        x = pd.to_numeric(
            rankable[xcol],
            errors="coerce",
        ).to_numpy()
        y = pd.to_numeric(
            rankable[ycol],
            errors="coerce",
        ).to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)

        h = ax.hist2d(
            x[valid],
            y[valid],
            bins=bins,
        )
        fig.colorbar(
            h[3],
            ax=ax,
            label="Models tried",
        )

        ax.scatter(
            pd.to_numeric(
                good_models[xcol],
                errors="coerce",
            ),
            pd.to_numeric(
                good_models[ycol],
                errors="coerce",
            ),
            s=24,
            facecolors="none",
            linewidths=0.9,
            label="Good models",
        )

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(alpha=0.12)
        ax.legend(frameon=False)

    for ax in axes.ravel()[len(usable):]:
        ax.axis("off")

    fig.suptitle(
        "Sampling density: sparse / empty areas show potentially unexplored parameter space"
    )
    fig.savefig(output, dpi=250, bbox_inches="tight")
    plt.close(fig)


def promising_sparse_cells(
    pair_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Find occupied but relatively sparse pairwise cells that contain at least
    one good model. These are useful candidates for local refinement.

    Empty cells are also written separately by the main program because an
    empty cell may reflect either an unexplored region or an invalid /
    intentionally excluded parameter combination.
    """
    if pair_table.empty:
        return pair_table.copy()

    pieces = []

    for _, group in pair_table.groupby(
        ["x_parameter", "y_parameter"]
    ):
        occupied = group[
            group["model_count"] > 0
        ]
        if occupied.empty:
            continue

        sparse_limit = max(
            1.0,
            float(
                occupied["model_count"].quantile(0.25)
            ),
        )

        piece = group[
            (group["model_count"] > 0)
            & (group["model_count"] <= sparse_limit)
            & (group["good_model_count"] > 0)
        ].copy()

        pieces.append(piece)

    if not pieces:
        return pd.DataFrame(
            columns=pair_table.columns
        )

    out = pd.concat(
        pieces,
        ignore_index=True,
    )

    return out.sort_values(
        [
            "best_rmse_m",
            "model_count",
        ],
        ascending=[
            True,
            True,
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Join synthetic-sweep JSON metadata with inversion ranking "
            "and analyse sampled vs high-performing parameter space."
        )
    )

    parser.add_argument(
        "metadata_dir",
        type=Path,
        help="Directory containing model JSON metadata files.",
    )
    parser.add_argument(
        "ranking_csv",
        type=Path,
        help="peak_model_ranking.csv from the inversion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "model_parameter_analysis"
        ),
    )
    parser.add_argument(
        "--shape-index",
        type=int,
        default=0,
        help=(
            "Shape entry to analyse when a metadata JSON contains "
            "multiple shapes. Default: 0."
        ),
    )

    good = parser.add_mutually_exclusive_group()
    good.add_argument(
        "--good-top-n",
        type=int,
        default=None,
        help=(
            "Number of lowest-RMSE ranked models treated as 'good'. "
            "Default: 100 when no other good-model criterion is supplied."
        ),
    )
    good.add_argument(
        "--good-fraction",
        type=float,
        default=None,
        help=(
            "Lowest-RMSE fraction treated as good, e.g. 0.05 for 5%%."
        ),
    )
    good.add_argument(
        "--good-rmse-max",
        type=float,
        default=None,
        help=(
            "Treat models with RMSE <= this value in metres as good."
        ),
    )

    parser.add_argument(
        "--hist-bins",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--space-bins",
        type=int,
        default=12,
        help=(
            "Number of bins per dimension for pairwise occupancy maps. "
            "Default: 12."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[1/6] Reading model metadata")
    metadata = read_metadata_directory(
        args.metadata_dir,
        shape_index=args.shape_index,
    )
    metadata["run_id"] = metadata["run_id"].map(
        normalise_id
    )

    print(
        f"      metadata models: {len(metadata)}"
    )

    print("[2/6] Reading inversion ranking")
    ranking = read_ranking(
        args.ranking_csv
    )
    print(
        f"      ranking rows: {len(ranking)}"
    )

    # Avoid duplicated metadata columns from ranking.
    ranking_only = ranking.copy()

    joined = metadata.merge(
        ranking_only,
        on="run_id",
        how="left",
        suffixes=("", "_ranking"),
    )

    joined_csv = (
        args.output_dir
        / "model_metadata_with_inversion_scores.csv"
    )
    joined.to_csv(
        joined_csv,
        index=False,
    )

    matched = joined[
        np.isfinite(
            pd.to_numeric(
                joined["rmse_filtered_m"],
                errors="coerce",
            )
        )
    ]

    print(
        f"      metadata/ranking matches with RMSE: "
        f"{len(matched)}/{len(joined)}"
    )

    print("[3/6] Selecting and characterising good models")
    good_models = select_good_models(
        joined,
        top_n=args.good_top_n,
        top_fraction=args.good_fraction,
        rmse_max=args.good_rmse_max,
    )

    good_csv = (
        args.output_dir
        / "good_models.csv"
    )
    good_models.to_csv(
        good_csv,
        index=False,
    )

    parameters = available_parameters(
        joined
    )

    characteristics = characterise_parameters(
        joined,
        good_models,
        parameters,
    )

    characteristics_csv = (
        args.output_dir
        / "good_model_characteristics.csv"
    )
    characteristics.to_csv(
        characteristics_csv,
        index=False,
    )

    print(
        f"      good models selected: {len(good_models)}"
    )

    if len(good_models):
        print(
            "      good-model RMSE range: "
            f"{good_models['rmse_filtered_m'].min():.2f} to "
            f"{good_models['rmse_filtered_m'].max():.2f} m"
        )

    print("[4/6] Plotting parameter distributions")
    make_histograms(
        joined,
        good_models,
        parameters,
        args.output_dir
        / "parameter_distributions_all_vs_good.png",
        bins=args.hist_bins,
    )

    scatter_all_good(
        joined,
        good_models,
        "center_x_m",
        "center_y_m",
        args.output_dir
        / "center_xy_all_vs_good.png",
        "Ellipsoid centre locations: all tried vs good models",
    )

    make_semi_axes_plot(
        joined,
        good_models,
        args.output_dir
        / "semi_axes_all_vs_good.png",
    )

    rankable = select_rankable(
        joined
    )

    make_parameter_vs_rmse(
        rankable,
        good_models,
        parameters,
        args.output_dir
        / "parameter_vs_rmse.png",
    )

    print("[5/6] Mapping sampled and sparse parameter space")
    good_ids = set(
        good_models["run_id"].astype(str)
    )

    pair_tables = []

    for xcol, ycol in PAIR_PARAMETERS:
        if (
            xcol not in rankable.columns
            or ycol not in rankable.columns
        ):
            continue

        table = pairwise_bin_table(
            rankable,
            good_ids,
            xcol,
            ycol,
            bins=args.space_bins,
        )

        if not table.empty:
            pair_tables.append(table)

    if pair_tables:
        pair_table = pd.concat(
            pair_tables,
            ignore_index=True,
        )

        pair_table.to_csv(
            args.output_dir
            / "pairwise_parameter_space_bins.csv",
            index=False,
        )

        empty = pair_table[
            pair_table["model_count"] == 0
        ].copy()

        empty.to_csv(
            args.output_dir
            / "empty_pairwise_parameter_cells.csv",
            index=False,
        )

        promising = promising_sparse_cells(
            pair_table
        )

        promising.to_csv(
            args.output_dir
            / "promising_sparse_parameter_cells.csv",
            index=False,
        )
    else:
        pair_table = pd.DataFrame()

    make_occupancy_plot(
        rankable,
        good_models,
        PAIR_PARAMETERS,
        args.output_dir
        / "parameter_space_occupancy.png",
        bins=args.space_bins,
    )

    print("[6/6] Finished")
    print("\nKey outputs:")
    print(
        "  "
        + str(
            args.output_dir
            / "model_metadata_with_inversion_scores.csv"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / "good_model_characteristics.csv"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / "parameter_distributions_all_vs_good.png"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / "center_xy_all_vs_good.png"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / "semi_axes_all_vs_good.png"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / "parameter_vs_rmse.png"
        )
    )
    print(
        "  "
        + str(
            args.output_dir
            / "parameter_space_occupancy.png"
        )
    )

    if pair_tables:
        print(
            "  "
            + str(
                args.output_dir
                / "promising_sparse_parameter_cells.csv"
            )
        )
        print(
            "  "
            + str(
                args.output_dir
                / "empty_pairwise_parameter_cells.csv"
            )
        )


if __name__ == "__main__":
    main()

