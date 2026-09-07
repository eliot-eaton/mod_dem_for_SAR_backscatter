#!/usr/bin/env python3
"""Test inversion using both sides of the radar shadow.

Two range-direction edges are measured on every azimuth row:

1. pre-shadow edge: the bright peak immediately before the shadow trough;
2. post-shadow edge: the bright peak immediately after the shadow.

The MLI is median filtered before picking. SimSAR is median filtered with a
small 2-D kernel (3 x 3 by default). SimSAR candidates are tracked across
azimuth so isolated layover pixels cannot force large row-to-row jumps.

This is intentionally a compact test script. It does not plot raw MLI data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


# Profile geometry used by the Sinabung inversion.
PROFILE_X1 = 775
PROFILE_X2 = 900
PROFILE_ROWS = (2030, 2010, 1990)
PROFILE_LABELS = ("A", "B", "C")
SHADOW_START_XS = (812, 823, 826)
RANGE_PIXEL_SPACING_M = 2.728212

DEFAULT_AZ_MIN = min(PROFILE_ROWS)
DEFAULT_AZ_MAX = max(PROFILE_ROWS)

IMAGE_X1 = 750
IMAGE_X2 = 930
IMAGE_PAD_Y = 20

EXCAVATION_INTERACTIONS = {"excavate_to_lower", "subtract_thickness"}
KNOWN_INTERACTIONS = EXCAVATION_INTERACTIONS | {
    "fill_to_upper",
    "add_thickness",
}


@dataclass(frozen=True)
class Peak:
    x: float
    db: float
    prominence: float


@dataclass(frozen=True)
class EdgeMetrics:
    rmse_m: float
    mae_m: float
    bias_m: float
    coverage: float
    matched: int
    observed: int


def finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return []
    padded = np.r_[False, mask, False]
    change = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(change == 1)
    stops = np.flatnonzero(change == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def interp_finite(values: np.ndarray) -> Optional[np.ndarray]:
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if good.sum() < 3:
        return None
    x = np.arange(values.size, dtype=float)
    out = values.copy()
    out[~good] = np.interp(x[~good], x[good], values[good])
    return out


def log_intensity(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    out = np.full(data.shape, np.nan, dtype=np.float32)
    good = np.isfinite(data) & (data > 0)
    out[good] = 10.0 * np.log10(data[good])
    return out


def interpolated_shadow_starts(rows: Sequence[int]) -> Dict[int, float]:
    anchor_rows = np.asarray(PROFILE_ROWS, dtype=float)
    anchor_x = np.asarray(SHADOW_START_XS, dtype=float)
    order = np.argsort(anchor_rows)
    anchor_rows = anchor_rows[order]
    anchor_x = anchor_x[order]
    q = np.asarray(rows, dtype=float)
    out = np.interp(q, anchor_rows, anchor_x)

    left = q < anchor_rows[0]
    if np.any(left):
        slope = (anchor_x[1] - anchor_x[0]) / (anchor_rows[1] - anchor_rows[0])
        out[left] = anchor_x[0] + slope * (q[left] - anchor_rows[0])

    right = q > anchor_rows[-1]
    if np.any(right):
        slope = (anchor_x[-1] - anchor_x[-2]) / (anchor_rows[-1] - anchor_rows[-2])
        out[right] = anchor_x[-1] + slope * (q[right] - anchor_rows[-1])

    return {int(r): float(x) for r, x in zip(q, out)}


def read_corridor(path: Path, rows: Sequence[int], pad: int = 0) -> Tuple[np.ndarray, int, int, Tuple[int, int]]:
    row_min = min(rows)
    row_max = max(rows)
    with rasterio.open(path) as src:
        if row_max >= src.height or PROFILE_X2 >= src.width:
            raise ValueError(f"Requested profile corridor is outside {path.name}")
        row0 = max(0, row_min - pad)
        row1 = min(src.height, row_max + 1 + pad)
        col0 = max(0, PROFILE_X1 - pad)
        col1 = min(src.width, PROFILE_X2 + 1 + pad)
        arr = src.read(
            1,
            window=Window(col0, row0, col1 - col0, row1 - row0),
            masked=True,
        )
        shape = (src.height, src.width)

    data = arr.filled(np.nan).astype(np.float32) if np.ma.isMaskedArray(arr) else np.asarray(arr, dtype=np.float32)
    return data, row0, col0, shape


def extract_profiles(data: np.ndarray, row0: int, col0: int, rows: Sequence[int]) -> Dict[int, np.ndarray]:
    x0 = PROFILE_X1 - col0
    x1 = PROFILE_X2 - col0 + 1
    return {
        int(row): np.asarray(data[int(row) - row0, x0:x1], dtype=np.float32)
        for row in rows
    }


def prepare_mli(path: Path, rows: Sequence[int], median_size: int) -> Tuple[Dict[int, np.ndarray], Tuple[int, int]]:
    if median_size < 1 or median_size % 2 == 0:
        raise ValueError("--median-size must be a positive odd integer")

    pad = median_size // 2
    data, row0, col0, shape = read_corridor(path, rows, pad=pad)
    good = np.isfinite(data) & (data > 0)
    if not np.any(good):
        raise ValueError("MLI corridor contains no finite positive pixels")

    work = data.copy()
    work[~good] = float(np.median(data[good]))
    filt = median_filter(work, size=median_size, mode="nearest").astype(np.float32)
    filt[~good] = np.nan
    profiles = extract_profiles(filt, row0, col0, rows)
    return {row: log_intensity(p) for row, p in profiles.items()}, shape


def prepare_simsar(path: Path, rows: Sequence[int], expected_shape: Tuple[int, int], median_size: int) -> Dict[int, np.ndarray]:
    if median_size < 1 or median_size % 2 == 0:
        raise ValueError("--simsar-median-size must be a positive odd integer")

    pad = median_size // 2
    data, row0, col0, shape = read_corridor(path, rows, pad=pad)
    if shape != expected_shape:
        raise ValueError(f"Raster shape mismatch: SimSAR {shape}, MLI {expected_shape}")

    good = np.isfinite(data) & (data > 0)
    if median_size > 1:
        work = np.where(good, data, 0.0).astype(np.float32)
        filt = median_filter(work, size=(median_size, median_size), mode="nearest").astype(np.float32)
        filt[~good] = np.nan
    else:
        filt = data.copy()
        filt[~good] = np.nan

    profiles = extract_profiles(filt, row0, col0, rows)
    return {row: log_intensity(p + 1e-12) for row, p in profiles.items()}


def smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(values, dtype=float)
    return gaussian_filter1d(np.asarray(values, dtype=float), sigma=sigma, mode="nearest")


def find_trough_x(profile_db: np.ndarray, shadow_start_x: float, sigma: float, half_window: int) -> Optional[float]:
    x = np.arange(PROFILE_X1, PROFILE_X2 + 1, dtype=float)
    y = interp_finite(profile_db)
    if y is None:
        return None
    y = smooth(y, sigma)
    keep = (x >= shadow_start_x - half_window) & (x <= shadow_start_x + half_window)
    if keep.sum() < 3:
        return None
    local = np.flatnonzero(keep)
    return float(x[local[int(np.argmin(y[keep]))]])


def peak_candidates(segment_x: np.ndarray, segment_y: np.ndarray, *, sigma: float, prominence_db: float, distance: int, allow_left_boundary: bool = False, allow_right_boundary: bool = False) -> List[Peak]:
    if segment_x.size < 2:
        return []
    y = smooth(segment_y, sigma)
    peaks, props = find_peaks(y, prominence=prominence_db, distance=max(1, int(distance)))
    out = [
        Peak(float(segment_x[int(i)]), float(y[int(i)]), float(props["prominences"][j]))
        for j, i in enumerate(peaks)
    ]

    window = min(y.size, max(3, 1 + 2 * max(1, int(distance))))
    if allow_left_boundary and y.size >= 2 and y[0] > y[1]:
        prom = max(0.0, float(y[0]) - float(np.min(y[1:window])))
        out.append(Peak(float(segment_x[0]), float(y[0]), prom))
    if allow_right_boundary and y.size >= 2 and y[-1] > y[-2]:
        prom = max(0.0, float(y[-1]) - float(np.min(y[-window:-1])))
        out.append(Peak(float(segment_x[-1]), float(y[-1]), prom))
    return out


def first_internal_gap_after(profile_db: np.ndarray, shadow_start_x: float) -> Optional[Tuple[int, int]]:
    x = np.arange(PROFILE_X1, PROFILE_X2 + 1, dtype=float)
    start = int(np.searchsorted(x, shadow_start_x, side="left"))
    runs = finite_runs(np.isfinite(profile_db[start:]))
    if len(runs) < 2:
        return None
    first_stop = start + runs[0][1]
    second_start = start + runs[1][0]
    return first_stop, second_start


def post_shadow_candidates(profile_db: np.ndarray, shadow_start_x: float, *, sigma: float, prominence_db: float, distance: int, use_gap: bool) -> List[Peak]:
    x = np.arange(PROFILE_X1, PROFILE_X2 + 1, dtype=float)

    if use_gap:
        gap = first_internal_gap_after(profile_db, shadow_start_x)
        if gap is not None:
            _, second_start = gap
            runs = finite_runs(np.isfinite(profile_db[second_start:]))
            if not runs:
                return []
            a, b = runs[0]
            idx = np.arange(second_start + a, second_start + b)
            return sorted(
                peak_candidates(x[idx], profile_db[idx], sigma=sigma, prominence_db=prominence_db, distance=distance, allow_left_boundary=True),
                key=lambda p: p.x,
            )

    keep = x >= shadow_start_x
    y = interp_finite(profile_db[keep])
    if y is None:
        return []
    return sorted(
        peak_candidates(x[keep], y, sigma=sigma, prominence_db=prominence_db, distance=distance, allow_left_boundary=True),
        key=lambda p: p.x,
    )


def pre_shadow_candidates(profile_db: np.ndarray, shadow_start_x: float, *, sigma: float, prominence_db: float, distance: int, use_gap: bool, search_pixels: int, trough_half_window: int) -> List[Peak]:
    """Find bright returns immediately before the shadow/trough."""
    x = np.arange(PROFILE_X1, PROFILE_X2 + 1, dtype=float)

    if use_gap:
        gap = first_internal_gap_after(profile_db, shadow_start_x)
        if gap is not None:
            first_stop, _ = gap
            right = first_stop - 1
            left_x = max(PROFILE_X1, x[right] - search_pixels)
            left = int(np.searchsorted(x, left_x, side="left"))
            idx = np.arange(left, right + 1)
            idx = idx[np.isfinite(profile_db[idx])]
            if idx.size >= 2:
                # Nearer-to-shadow candidates come first.
                return sorted(
                    peak_candidates(x[idx], profile_db[idx], sigma=sigma, prominence_db=prominence_db, distance=distance, allow_right_boundary=True),
                    key=lambda p: p.x,
                    reverse=True,
                )

    trough_x = find_trough_x(profile_db, shadow_start_x, sigma, trough_half_window)
    if trough_x is None:
        return []
    keep = (x >= trough_x - search_pixels) & (x < trough_x)
    y = interp_finite(profile_db[keep])
    if y is None:
        return []
    return sorted(
        peak_candidates(x[keep], y, sigma=sigma, prominence_db=prominence_db, distance=distance, allow_right_boundary=True),
        key=lambda p: p.x,
        reverse=True,
    )


def track_edge(candidates_by_row: Mapping[int, List[Peak]], rows: Sequence[int], *, max_jump: float, continuity_penalty: float, missing_penalty: float = 1.5, rank_penalty: float = 0.35) -> Dict[int, Optional[Peak]]:
    """Choose a continuous path through peak candidates without using MLI positions."""
    states = [(0.0, None, None, {})]

    for row in rows:
        next_states = []
        for cost, last_x, last_row, picks in states:
            skipped = dict(picks)
            skipped[row] = None
            next_states.append((cost + missing_penalty, last_x, last_row, skipped))

            for rank, peak in enumerate(candidates_by_row[row]):
                jump_cost = 0.0
                if last_x is not None and last_row is not None:
                    row_gap = max(1, row - last_row)
                    jump = abs(peak.x - last_x)
                    if jump > max_jump * row_gap:
                        continue
                    jump_cost = continuity_penalty * jump
                chosen = dict(picks)
                chosen[row] = peak
                next_states.append((cost + jump_cost + rank_penalty * rank, peak.x, row, chosen))

        best = {}
        for state in next_states:
            key = (state[1], state[2])
            if key not in best or state[0] < best[key][0]:
                best[key] = state
        states = list(best.values())

    if not states:
        return {int(row): None for row in rows}
    path = min(states, key=lambda s: s[0])[3]
    return {int(row): path.get(row) for row in rows}


def pick_mli_post(profiles: Mapping[int, np.ndarray], rows: Sequence[int], starts: Mapping[int, float], *, sigma: float, prominence_db: float, distance: int) -> Dict[int, Optional[Peak]]:
    out = {}
    for row in rows:
        candidates = post_shadow_candidates(profiles[row], starts[row], sigma=sigma, prominence_db=prominence_db, distance=distance, use_gap=False)
        out[row] = candidates[0] if candidates else None
    return out


def pick_mli_pre(profiles: Mapping[int, np.ndarray], rows: Sequence[int], starts: Mapping[int, float], *, sigma: float, prominence_db: float, distance: int, search_pixels: int, trough_half_window: int, max_jump: float, continuity_penalty: float) -> Dict[int, Optional[Peak]]:
    candidates = {
        row: pre_shadow_candidates(
            profiles[row], starts[row], sigma=sigma, prominence_db=prominence_db,
            distance=distance, use_gap=False, search_pixels=search_pixels,
            trough_half_window=trough_half_window,
        )
        for row in rows
    }
    return track_edge(candidates, rows, max_jump=max_jump, continuity_penalty=continuity_penalty)


def pick_simsar_edges(profiles: Mapping[int, np.ndarray], rows: Sequence[int], starts: Mapping[int, float], *, sigma: float, prominence_db: float, distance: int, use_gap: bool, search_pixels: int, trough_half_window: int, max_jump: float, continuity_penalty: float) -> Tuple[Dict[int, Optional[Peak]], Dict[int, Optional[Peak]]]:
    pre_candidates = {}
    post_candidates = {}
    for row in rows:
        pre_candidates[row] = pre_shadow_candidates(
            profiles[row], starts[row], sigma=sigma, prominence_db=prominence_db,
            distance=distance, use_gap=use_gap, search_pixels=search_pixels,
            trough_half_window=trough_half_window,
        )
        post_candidates[row] = post_shadow_candidates(
            profiles[row], starts[row], sigma=sigma, prominence_db=prominence_db,
            distance=distance, use_gap=use_gap,
        )
    pre = track_edge(pre_candidates, rows, max_jump=max_jump, continuity_penalty=continuity_penalty)
    post = track_edge(post_candidates, rows, max_jump=max_jump, continuity_penalty=continuity_penalty)
    return pre, post


def edge_metrics(observed: Mapping[int, Optional[Peak]], simulated: Mapping[int, Optional[Peak]], rows: Sequence[int]) -> Tuple[EdgeMetrics, List[Dict[str, float]]]:
    obs_rows = [row for row in rows if observed[row] is not None]
    residuals = []
    errors_m = []

    for row in rows:
        obs = observed[row]
        sim = simulated[row]
        error_px = np.nan
        error_m = np.nan
        if obs is not None and sim is not None:
            error_px = sim.x - obs.x
            error_m = error_px * RANGE_PIXEL_SPACING_M
            errors_m.append(error_m)
        residuals.append({
            "azimuth_row": row,
            "mli_x_px": np.nan if obs is None else obs.x,
            "simsar_x_px": np.nan if sim is None else sim.x,
            "error_px": error_px,
            "error_m": error_m,
        })

    matched = len(errors_m)
    coverage = matched / len(obs_rows) if obs_rows else 0.0
    if errors_m:
        arr = np.asarray(errors_m, dtype=float)
        metrics = EdgeMetrics(
            rmse_m=float(np.sqrt(np.mean(arr ** 2))),
            mae_m=float(np.mean(np.abs(arr))),
            bias_m=float(np.mean(arr)),
            coverage=coverage,
            matched=matched,
            observed=len(obs_rows),
        )
    else:
        metrics = EdgeMetrics(np.nan, np.nan, np.nan, coverage, 0, len(obs_rows))
    return metrics, residuals


def read_interaction(run_id: str, provenance_dir: Optional[Path], pattern: str) -> Tuple[str, bool]:
    if provenance_dir is None:
        return "unknown", False
    path = provenance_dir / pattern.format(id=run_id)
    if not path.exists():
        return "unknown", False
    payload = json.loads(path.read_text())
    interactions = []
    for shape in payload.get("shapes", []):
        value = shape.get("interaction")
        if value and value not in interactions:
            interactions.append(str(value))
    unknown = [x for x in interactions if x not in KNOWN_INTERACTIONS]
    if unknown:
        raise ValueError(f"Unknown interaction in {path.name}: {unknown}")
    return "+".join(interactions) if interactions else "unknown", any(x in EXCAVATION_INTERACTIONS for x in interactions)


def expand_ids(start: str, end: str) -> List[str]:
    a, b = int(start), int(end)
    if b < a:
        raise ValueError("id_end must be >= id_start")
    width = max(len(start), len(end))
    return [f"{i:0{width}d}" for i in range(a, b + 1)]


def read_image_db(path: Path, row_min: int, row_max: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    y1 = max(0, row_min - IMAGE_PAD_Y)
    y2 = row_max + IMAGE_PAD_Y
    with rasterio.open(path) as src:
        y2 = min(src.height - 1, y2)
        arr = src.read(1, window=Window(IMAGE_X1, y1, IMAGE_X2 - IMAGE_X1 + 1, y2 - y1 + 1), masked=True)
    data = arr.filled(np.nan).astype(np.float32) if np.ma.isMaskedArray(arr) else np.asarray(arr, dtype=np.float32)
    return log_intensity(data), (IMAGE_X1, IMAGE_X2, y1, y2)


def edge_arrays(picks: Mapping[int, Optional[Peak]]) -> Tuple[np.ndarray, np.ndarray]:
    pts = [(p.x, row) for row, p in picks.items() if p is not None]
    if not pts:
        return np.array([]), np.array([])
    pts.sort(key=lambda t: t[1])
    return np.asarray([p[0] for p in pts]), np.asarray([p[1] for p in pts])


def plot_best_edges(mli_path: Path, sim_path: Path, mli_pre: Mapping[int, Optional[Peak]], mli_post: Mapping[int, Optional[Peak]], sim_pre: Mapping[int, Optional[Peak]], sim_post: Mapping[int, Optional[Peak]], rows: Sequence[int], output: Path, run_id: str) -> None:
    mli_img, extent = read_image_db(mli_path, min(rows), max(rows))
    sim_img, _ = read_image_db(sim_path, min(rows), max(rows))
    x1, x2, y1, y2 = extent

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, image, title, pre, post in [
        (axes[0], mli_img, "Filtered MLI picks", mli_pre, mli_post),
        (axes[1], sim_img, f"SimSAR P.{run_id}", sim_pre, sim_post),
    ]:
        ax.imshow(image, cmap="gray", vmin=-30, vmax=0, extent=(x1, x2, y2, y1), aspect="auto")
        pre_x, pre_y = edge_arrays(pre)
        post_x, post_y = edge_arrays(post)
        if pre_x.size:
            ax.plot(pre_x, pre_y, "-", linewidth=1.4, label="pre-shadow edge")
        if post_x.size:
            ax.plot(post_x, post_y, "-", linewidth=1.4, label="post-shadow edge")
        ax.set_title(title)
        ax.set_xlabel("range pixel")
        ax.set_ylabel("azimuth row")
        ax.legend(loc="best")
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_profile_check(mli_profiles: Mapping[int, np.ndarray], sim_profiles: Mapping[int, np.ndarray], mli_pre: Mapping[int, Optional[Peak]], mli_post: Mapping[int, Optional[Peak]], sim_pre: Mapping[int, Optional[Peak]], sim_post: Mapping[int, Optional[Peak]], output: Path, run_id: str) -> None:
    x = np.arange(PROFILE_X1, PROFILE_X2 + 1)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for ax, row, label in zip(axes, PROFILE_ROWS, PROFILE_LABELS):
        ax.plot(x, mli_profiles[row], linewidth=1.3, label="filtered MLI")
        ax.plot(x, sim_profiles[row], linewidth=1.1, label=f"SimSAR P.{run_id}")
        for peak, style, name in [
            (mli_pre[row], "--", "MLI pre"),
            (mli_post[row], ":", "MLI post"),
            (sim_pre[row], "--", "Sim pre"),
            (sim_post[row], ":", "Sim post"),
        ]:
            if peak is not None:
                ax.axvline(peak.x, linestyle=style, linewidth=1.0, label=name)
        ax.set_title(f"Profile {label}, azimuth row {row}")
        ax.set_ylabel("dB")
    axes[-1].set_xlabel("range pixel")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), ncol=3, fontsize=8)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    rows = list(range(args.azimuth_min, args.azimuth_max + 1))
    starts = interpolated_shadow_starts(rows)
    run_ids = expand_ids(args.id_start, args.id_end)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mli_profiles, mli_shape = prepare_mli(args.mli_tif, rows, args.median_size)
    mli_post = pick_mli_post(
        mli_profiles, rows, starts,
        sigma=args.peak_sigma, prominence_db=args.peak_prominence_db,
        distance=args.peak_distance_pixels,
    )
    mli_pre = pick_mli_pre(
        mli_profiles, rows, starts,
        sigma=args.peak_sigma, prominence_db=args.pre_shadow_prominence_db,
        distance=args.peak_distance_pixels,
        search_pixels=args.pre_shadow_search_pixels,
        trough_half_window=args.shadow_trough_half_window,
        max_jump=args.simsar_max_jump_pixels,
        continuity_penalty=args.simsar_continuity_penalty,
    )

    print(f"MLI pre-shadow picks:  {sum(p is not None for p in mli_pre.values())}/{len(rows)}")
    print(f"MLI post-shadow picks: {sum(p is not None for p in mli_post.values())}/{len(rows)}")

    ranking_rows = []
    residual_rows = []
    cached = {}

    for i, run_id in enumerate(run_ids, start=1):
        sim_path = args.simsar_dir / args.simsar_pattern.format(id=run_id)
        print(f"[{i:>4}/{len(run_ids)}] {run_id}: ", end="", flush=True)
        if not sim_path.exists():
            print("missing")
            continue

        interaction, use_gap = read_interaction(run_id, args.provenance_dir, args.provenance_pattern)
        try:
            sim_profiles = prepare_simsar(sim_path, rows, mli_shape, args.simsar_median_size)
            sim_pre, sim_post = pick_simsar_edges(
                sim_profiles, rows, starts,
                sigma=args.peak_sigma,
                prominence_db=args.peak_prominence_db,
                distance=args.peak_distance_pixels,
                use_gap=use_gap,
                search_pixels=args.pre_shadow_search_pixels,
                trough_half_window=args.shadow_trough_half_window,
                max_jump=args.simsar_max_jump_pixels,
                continuity_penalty=args.simsar_continuity_penalty,
            )
            pre_metrics, pre_res = edge_metrics(mli_pre, sim_pre, rows)
            post_metrics, post_res = edge_metrics(mli_post, sim_post, rows)

            ok_pre = pre_metrics.coverage >= args.min_coverage and np.isfinite(pre_metrics.rmse_m)
            ok_post = post_metrics.coverage >= args.min_coverage and np.isfinite(post_metrics.rmse_m)
            status = "ok" if ok_pre and ok_post else "insufficient_coverage"

            squared = []
            if ok_pre:
                squared.extend([r["error_m"] ** 2 for r in pre_res if np.isfinite(r["error_m"])])
            if ok_post:
                squared.extend([r["error_m"] ** 2 for r in post_res if np.isfinite(r["error_m"])])
            combined = float(np.sqrt(np.mean(squared))) if ok_pre and ok_post and squared else np.nan

            ranking_rows.append({
                "run_id": run_id,
                "status": status,
                "interaction": interaction,
                "rmse_pre_shadow_m": pre_metrics.rmse_m if ok_pre else np.nan,
                "rmse_post_shadow_m": post_metrics.rmse_m if ok_post else np.nan,
                "rmse_both_edges_m": combined,
                "coverage_pre_shadow": pre_metrics.coverage,
                "coverage_post_shadow": post_metrics.coverage,
                "matched_pre_shadow": pre_metrics.matched,
                "matched_post_shadow": post_metrics.matched,
                "bias_pre_shadow_m": pre_metrics.bias_m,
                "bias_post_shadow_m": post_metrics.bias_m,
            })

            for edge_name, records in (("pre", pre_res), ("post", post_res)):
                for record in records:
                    residual_rows.append({"run_id": run_id, "edge": edge_name, **record})

            cached[run_id] = (sim_path, sim_profiles, sim_pre, sim_post)
            print(f"pre {pre_metrics.rmse_m:.2f} m, post {post_metrics.rmse_m:.2f} m, combined {combined:.2f} m")
        except Exception as exc:
            ranking_rows.append({"run_id": run_id, "status": f"error: {exc}", "interaction": interaction})
            print(f"failed: {exc}")

    ranking = pd.DataFrame(ranking_rows)
    if ranking.empty:
        raise RuntimeError("No SimSAR models were processed")

    rank_column = {
        "pre": "rmse_pre_shadow_m",
        "post": "rmse_post_shadow_m",
        "combined": "rmse_both_edges_m",
    }[args.rank_by]
    ranking = ranking.sort_values([rank_column, "run_id"], na_position="last").reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    ranking_csv = args.output_dir / "dual_shadow_edge_ranking.csv"
    residual_csv = args.output_dir / "dual_shadow_edge_residuals.csv"
    ranking.to_csv(ranking_csv, index=False)
    pd.DataFrame(residual_rows).to_csv(residual_csv, index=False)

    valid = ranking[(ranking["status"] == "ok") & np.isfinite(ranking[rank_column])]
    if valid.empty:
        print("\nNo model met coverage for both edges. Inspect the CSVs before changing thresholds.")
        return

    print("\nTop test models:")
    cols = ["rank", "run_id", "rmse_pre_shadow_m", "rmse_post_shadow_m", "rmse_both_edges_m", "coverage_pre_shadow", "coverage_post_shadow"]
    print(valid.head(args.top_n)[cols].to_string(index=False))

    best_id = str(valid.iloc[0]["run_id"])
    sim_path, sim_profiles, sim_pre, sim_post = cached[best_id]
    plot_best_edges(
        args.mli_tif, sim_path, mli_pre, mli_post, sim_pre, sim_post, rows,
        args.output_dir / f"best_{best_id}_dual_edges.png", best_id,
    )
    plot_profile_check(
        mli_profiles, sim_profiles, mli_pre, mli_post, sim_pre, sim_post,
        args.output_dir / f"best_{best_id}_ABC_profiles.png", best_id,
    )

    print(f"\nRanking:    {ranking_csv}")
    print(f"Residuals:  {residual_csv}")
    print(f"Edge image: {args.output_dir / f'best_{best_id}_dual_edges.png'}")
    print(f"Profiles:   {args.output_dir / f'best_{best_id}_ABC_profiles.png'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Test pre- and post-shadow edge fitting on MLI and SimSAR")
    p.add_argument("mli_tif", type=Path)
    p.add_argument("simsar_dir", type=Path)
    p.add_argument("id_start")
    p.add_argument("id_end")
    p.add_argument("--output-dir", type=Path, default=Path("dual_shadow_edge_test"))
    p.add_argument("--simsar-pattern", default="P.{id}.sim_sar.radar.tif")
    p.add_argument("--median-size", type=int, default=15)
    p.add_argument("--simsar-median-size", type=int, default=3)
    p.add_argument("--peak-sigma", type=float, default=1.0)
    p.add_argument("--peak-prominence-db", type=float, default=5.0, help="Post-shadow peak prominence")
    p.add_argument("--pre-shadow-prominence-db", type=float, default=3.0, help="Pre-shadow peak prominence; lower default because this edge is usually less isolated")
    p.add_argument("--peak-distance-pixels", type=int, default=3)
    p.add_argument("--pre-shadow-search-pixels", type=int, default=30, help="How far left of the shadow trough to search for the pre-shadow peak")
    p.add_argument("--shadow-trough-half-window", type=int, default=12, help="Half-width around the expected shadow start used to locate the MLI/finite SimSAR trough")
    p.add_argument("--simsar-max-jump-pixels", type=float, default=4.0)
    p.add_argument("--simsar-continuity-penalty", type=float, default=0.25)
    p.add_argument("--min-coverage", type=float, default=0.2)
    p.add_argument("--azimuth-min", type=int, default=DEFAULT_AZ_MIN)
    p.add_argument("--azimuth-max", type=int, default=DEFAULT_AZ_MAX)
    p.add_argument("--provenance-dir", type=Path, default=None)
    p.add_argument("--provenance-pattern", default="{id}.json")
    p.add_argument("--rank-by", choices=("pre", "post", "combined"), default="combined")
    p.add_argument("--top-n", type=int, default=10)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.mli_tif = args.mli_tif.resolve()
    args.simsar_dir = args.simsar_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.provenance_dir is not None:
        args.provenance_dir = args.provenance_dir.resolve()
    run(args)


if __name__ == "__main__":
    main()
