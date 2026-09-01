#!/usr/bin/env python3
"""
Dense peak-position inversion of modified-DEM SimSAR models against one MLI.

The inversion can be evaluated on a configurable azimuth corridor.  The
original A/B/C profiles remain the only rows shown in the profile diagnostic:

    A: y=2030, shadow-start x=812
    B: y=2010, shadow-start x=823
    C: y=1990, shadow-start x=826

By default the inversion uses rows 1990..2030.  Use --azimuth-min and
--azimuth-max to extend the inversion beyond A/C.  Shadow-search start x is
piecewise-linear between A/B/C and linearly extrapolated beyond the end
anchors using the nearest anchor-pair slope.

Ranking is based ONLY on peak-position error relative to the MEDIAN-FILTERED
MLI.  Raw MLI peaks are retained as diagnostics but never affect ranking.

Interaction-aware SimSAR no-data handling
------------------------------------------
No-data/NaN pixels are shown as -40 dB in plots.  The special
"data -> no-data -> data: search only the final data segment" rule is applied
ONLY to excavation/lowering geometries:

    excavate_to_lower
    subtract_thickness

For filling/raising geometries (fill_to_upper, add_thickness), the first data
section is NOT discarded and the normal peak picker is used.

The interaction can be read per model from the existing run JSON via
--provenance-dir (payload field shapes[*].interaction), or forced for all
models with --interaction.

Expected SimSAR filenames
-------------------------
    P.{ID}.sim_sar.radar.tif

Main outputs
------------
    observed_dense_peak_positions.csv
    dense_peak_model_residuals.csv
    peak_model_ranking.csv
    observed_mli_filtered_vs_unfiltered.png
    top5_peak_profile_comparison.png
    top5_3x5_image_mli_profile_comparison.png

Example
-------
python invert_simsar_peak_models_extended_azimuth_interaction.py \
    ./20201226.mli.tif \
    ./sim_sar \
    1 100 \
    --provenance-dir ./mod_dem/synthetic_sweep \
    --azimuth-min 1970 \
    --azimuth-max 2050 \
    --output-dir ./peak_inversion_20201226
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks


# =============================================================================
# GEOMETRY
# =============================================================================

PROFILE_X1 = 775
PROFILE_X2 = 900

PLOT_PROFILE_LABELS = ("A", "B", "C")
PLOT_PROFILE_ROWS = (2030, 2010, 1990)
PLOT_SHADOW_START_XS = (812, 823, 826)

DEFAULT_INVERSION_ROW_MIN = min(PLOT_PROFILE_ROWS)
DEFAULT_INVERSION_ROW_MAX = max(PLOT_PROFILE_ROWS)

# These globals are configured at run time by _configure_inversion_geometry().
# Keeping them module-level means the rest of the existing dense-profile code
# can continue to use one authoritative inversion corridor.
INVERSION_ROW_MIN = DEFAULT_INVERSION_ROW_MIN
INVERSION_ROW_MAX = DEFAULT_INVERSION_ROW_MAX
INVERSION_ROWS = tuple(range(INVERSION_ROW_MIN, INVERSION_ROW_MAX + 1))

RANGE_PIXEL_SPACING_M = 2.728212

# Local image crop for the 3 x 5 diagnostic.  The y limits expand
# automatically if --azimuth-min/--azimuth-max extend beyond this region.
IMAGE_X1 = 750
IMAGE_X2 = 930
DEFAULT_IMAGE_Y1 = 1960
DEFAULT_IMAGE_Y2 = 2060
IMAGE_Y1 = DEFAULT_IMAGE_Y1
IMAGE_Y2 = DEFAULT_IMAGE_Y2
IMAGE_AZIMUTH_PAD = 150

DISPLAY_VMIN_DB = -30.0
DISPLAY_VMAX_DB = 0.0

# SimSAR no-data pixels are displayed as a deliberately low return.
# Peak picking still preserves the original no-data mask so gaps cannot
# manufacture false peaks.
SIMSAR_NODATA_DB = -40.0


@dataclass(frozen=True)
class PeakPick:
    x_pixel: float
    intensity_db: float
    prominence_db: float


def plot_label_for_row(row: int) -> str:
    """Return A/B/C for the three display rows, otherwise the row number."""
    mapping = dict(zip(PLOT_PROFILE_ROWS, PLOT_PROFILE_LABELS))
    return mapping.get(int(row), str(int(row)))


def _interp_extrapolate_piecewise(
    rows: np.ndarray,
    anchor_rows: np.ndarray,
    anchor_starts: np.ndarray,
) -> np.ndarray:
    """Linear interpolation between anchors and linear extrapolation outside."""
    order = np.argsort(anchor_rows)
    xp = np.asarray(anchor_rows, dtype=float)[order]
    fp = np.asarray(anchor_starts, dtype=float)[order]
    x = np.asarray(rows, dtype=float)

    if xp.size < 2:
        raise ValueError("At least two shadow-start anchors are required.")

    out = np.interp(x, xp, fp)

    left = x < xp[0]
    if np.any(left):
        slope_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
        out[left] = fp[0] + slope_left * (x[left] - xp[0])

    right = x > xp[-1]
    if np.any(right):
        slope_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        out[right] = fp[-1] + slope_right * (x[right] - xp[-1])

    return out


def dense_shadow_start_map() -> Dict[int, float]:
    """Return row-specific shadow-search starts for the active corridor."""
    anchor_rows = np.asarray(PLOT_PROFILE_ROWS, dtype=float)
    anchor_starts = np.asarray(PLOT_SHADOW_START_XS, dtype=float)
    rows = np.asarray(INVERSION_ROWS, dtype=float)
    starts = _interp_extrapolate_piecewise(rows, anchor_rows, anchor_starts)

    return {
        int(row): float(start)
        for row, start in zip(rows, starts)
    }


def _configure_inversion_geometry(azimuth_min: int, azimuth_max: int) -> None:
    """Configure the dense azimuth corridor and row-specific search starts."""
    global INVERSION_ROW_MIN, INVERSION_ROW_MAX, INVERSION_ROWS
    global SHADOW_START_BY_ROW, IMAGE_Y1, IMAGE_Y2

    azimuth_min = int(azimuth_min)
    azimuth_max = int(azimuth_max)

    if azimuth_min < 0:
        raise ValueError("azimuth_min must be >= 0.")
    if azimuth_max < azimuth_min:
        raise ValueError("azimuth_max must be >= azimuth_min.")

    INVERSION_ROW_MIN = azimuth_min
    INVERSION_ROW_MAX = azimuth_max
    INVERSION_ROWS = tuple(range(INVERSION_ROW_MIN, INVERSION_ROW_MAX + 1))
    SHADOW_START_BY_ROW = dense_shadow_start_map()

    # Ensure the dense picked-edge line remains visible in the image panels.
    IMAGE_Y1 = min(DEFAULT_IMAGE_Y1, INVERSION_ROW_MIN - IMAGE_AZIMUTH_PAD)
    IMAGE_Y1 = max(0, IMAGE_Y1)
    IMAGE_Y2 = max(DEFAULT_IMAGE_Y2, INVERSION_ROW_MAX + IMAGE_AZIMUTH_PAD)


SHADOW_START_BY_ROW = dense_shadow_start_map()


# =============================================================================
# MODEL INTERACTION / PROVENANCE
# =============================================================================

EXCAVATION_INTERACTIONS = {"excavate_to_lower", "subtract_thickness"}
FILL_INTERACTIONS = {"fill_to_upper", "add_thickness"}
KNOWN_INTERACTIONS = EXCAVATION_INTERACTIONS | FILL_INTERACTIONS


def _read_interactions_from_run_json(path: Path) -> List[str]:
    """Read unique shapes[*].interaction values from one run provenance JSON."""
    payload = json.loads(Path(path).read_text())
    interactions: List[str] = []

    for shape in payload.get("shapes", []):
        value = shape.get("interaction")
        if value is not None:
            value = str(value)
            if value not in interactions:
                interactions.append(value)

    return interactions


def resolve_model_interaction(
    run_id: str,
    *,
    simsar_dir: Path,
    interaction: str = "auto",
    provenance_dir: Optional[Path] = None,
    provenance_pattern: str = "{id}.json",
) -> Tuple[str, bool, Optional[Path]]:
    """
    Resolve model interaction and whether excavation-specific gap handling applies.

    Returns
    -------
    interaction_label, use_excavation_gap_rule, provenance_path
    """
    if interaction != "auto":
        if interaction not in KNOWN_INTERACTIONS:
            raise ValueError(f"Unknown interaction override: {interaction}")
        return interaction, interaction in EXCAVATION_INTERACTIONS, None

    candidates: List[Path] = []
    if provenance_dir is not None:
        candidates.append(Path(provenance_dir) / provenance_pattern.format(id=run_id))
    else:
        # Convenient fallbacks only.  For the repository's usual layout,
        # --provenance-dir should point to the synthetic DEM/run-JSON directory.
        candidates.extend([
            Path(simsar_dir) / provenance_pattern.format(id=run_id),
            Path(simsar_dir).parent / provenance_pattern.format(id=run_id),
        ])

    json_path = next((p for p in candidates if p.exists()), None)
    if json_path is None:
        # Unknown is deliberately NON-excavation: do not discard the first
        # valid data segment unless provenance proves the model was excavated.
        return "unknown", False, None

    interactions = _read_interactions_from_run_json(json_path)
    if not interactions:
        return "unknown", False, json_path

    unknown = [x for x in interactions if x not in KNOWN_INTERACTIONS]
    if unknown:
        raise ValueError(
            f"Unknown interaction(s) in {json_path.name}: {', '.join(unknown)}"
        )

    use_excavation_gap_rule = any(x in EXCAVATION_INTERACTIONS for x in interactions)
    return "+".join(interactions), use_excavation_gap_rule, json_path


# =============================================================================
# RASTER / INTENSITY HELPERS
# =============================================================================


def log_intensity(data: np.ndarray) -> np.ndarray:
    """Convert positive linear intensity to dB using 10*log10(intensity)."""
    data = np.asarray(data, dtype=np.float32)
    out = np.full(data.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(data) & (data > 0)
    out[valid] = 10.0 * np.log10(data[valid])
    return out


def _validate_geometry(shape: Tuple[int, int]) -> None:
    nrows, ncols = shape
    if INVERSION_ROW_MAX >= nrows:
        raise ValueError(
            f"Inversion row {INVERSION_ROW_MAX} is outside raster with {nrows} rows."
        )
    if PROFILE_X2 >= ncols:
        raise ValueError(
            f"Profile x={PROFILE_X2} is outside raster with {ncols} columns."
        )


def _read_dense_corridor(
    path: Path,
    *,
    pad: int = 0,
) -> Tuple[np.ndarray, int, int, Tuple[int, int]]:
    """Read the corridor containing all inversion rows and profile range."""
    path = Path(path)

    with rasterio.open(path) as src:
        full_shape = (src.height, src.width)
        _validate_geometry(full_shape)

        row0 = max(0, INVERSION_ROW_MIN - pad)
        row1 = min(src.height, INVERSION_ROW_MAX + 1 + pad)
        col0 = max(0, PROFILE_X1 - pad)
        col1 = min(src.width, PROFILE_X2 + 1 + pad)

        arr = src.read(
            1,
            window=Window(
                col_off=col0,
                row_off=row0,
                width=col1 - col0,
                height=row1 - row0,
            ),
            masked=True,
        )

    if np.ma.isMaskedArray(arr):
        data = arr.filled(np.nan).astype(np.float32)
    else:
        data = np.asarray(arr, dtype=np.float32)

    return data, row0, col0, full_shape


def _extract_dense_profiles(
    data: np.ndarray,
    row0: int,
    col0: int,
) -> Dict[int, np.ndarray]:
    """Extract x=PROFILE_X1..PROFILE_X2 for every inversion azimuth row."""
    local_x1 = PROFILE_X1 - col0
    local_x2 = PROFILE_X2 - col0
    expected = PROFILE_X2 - PROFILE_X1 + 1

    profiles: Dict[int, np.ndarray] = {}

    for row in INVERSION_ROWS:
        local_row = row - row0
        profile = data[local_row, local_x1:local_x2 + 1]
        if profile.size != expected:
            raise ValueError(
                f"Row {row} has {profile.size} samples; expected {expected}."
            )
        profiles[row] = np.asarray(profile, dtype=np.float32)

    return profiles


def read_radar_crop_db(
    path: Path,
    *,
    expected_shape: Optional[Tuple[int, int]] = None,
    add_epsilon: bool = False,
) -> np.ndarray:
    """Read the local diagnostic image crop and return dB values."""
    path = Path(path)

    with rasterio.open(path) as src:
        shape = (src.height, src.width)
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(f"Raster shape differs from MLI: {shape} != {expected_shape}")

        if IMAGE_X2 >= src.width or IMAGE_Y2 >= src.height:
            raise ValueError("Requested diagnostic crop falls outside raster.")

        arr = src.read(
            1,
            window=Window(
                col_off=IMAGE_X1,
                row_off=IMAGE_Y1,
                width=IMAGE_X2 - IMAGE_X1 + 1,
                height=IMAGE_Y2 - IMAGE_Y1 + 1,
            ),
            masked=True,
        )

    if np.ma.isMaskedArray(arr):
        linear = arr.filled(np.nan).astype(np.float32)
    else:
        linear = np.asarray(arr, dtype=np.float32)

    if add_epsilon:
        linear = linear + 1e-12

    return log_intensity(linear)


# =============================================================================
# MLI / SIMSAR PREPARATION
# =============================================================================


def prepare_mli_dense_profiles(
    mli_tif: Path,
    *,
    median_size: int = 15,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Tuple[int, int]]:
    """
    Return raw and median-filtered dB profiles for every inversion row.

    Median filtering is 2-D and performed in LINEAR intensity space.
    """
    if median_size < 1 or median_size % 2 == 0:
        raise ValueError("median_size must be a positive odd integer.")

    pad = median_size // 2
    mli_linear, row0, col0, full_shape = _read_dense_corridor(
        Path(mli_tif), pad=pad
    )

    valid = np.isfinite(mli_linear) & (mli_linear > 0)
    if not np.any(valid):
        raise ValueError("MLI inversion corridor contains no finite positive pixels.")

    work = mli_linear.copy()
    fill_value = float(np.median(mli_linear[valid]))
    work[~valid] = fill_value

    filtered_linear = median_filter(
        work,
        size=median_size,
        mode="nearest",
    ).astype(np.float32)
    filtered_linear[~valid] = np.nan

    raw_linear = _extract_dense_profiles(mli_linear, row0, col0)
    filtered_linear_profiles = _extract_dense_profiles(filtered_linear, row0, col0)

    raw_db = {row: log_intensity(values) for row, values in raw_linear.items()}
    filtered_db = {
        row: log_intensity(values)
        for row, values in filtered_linear_profiles.items()
    }

    return raw_db, filtered_db, full_shape


def read_simsar_dense_profiles(
    simsar_tif: Path,
    *,
    expected_shape: Tuple[int, int],
) -> Dict[int, np.ndarray]:
    """Read one SimSAR and return dB profiles for all inversion rows."""
    data, row0, col0, full_shape = _read_dense_corridor(Path(simsar_tif), pad=0)

    if full_shape != expected_shape:
        raise ValueError(
            "sim_sar and MLI raster shapes differ: "
            f"sim_sar={full_shape}, MLI={expected_shape}. No resampling is performed."
        )

    profiles_linear = _extract_dense_profiles(data, row0, col0)
    return {
        row: log_intensity(values + 1e-12)
        for row, values in profiles_linear.items()
    }


# =============================================================================
# PEAK PICKING
# =============================================================================


def _interpolate_finite_1d(values: np.ndarray) -> Optional[np.ndarray]:
    """Interpolate finite values for MLI-only peak picking."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return None

    x = np.arange(values.size, dtype=float)
    out = values.copy()
    out[~finite] = np.interp(x[~finite], x[finite], values[finite])
    return out


def _finite_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return contiguous True runs as half-open (start, stop) index pairs."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return []

    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def has_internal_nodata_gap(
    profile_db: np.ndarray,
    *,
    shadow_start_x: float,
) -> bool:
    """True when post-shadow data contain finite -> no-data -> finite structure."""
    x_pixels = np.arange(PROFILE_X1, PROFILE_X2 + 1, dtype=float)
    y = np.asarray(profile_db, dtype=float)[x_pixels >= float(shadow_start_x)]
    return len(_finite_runs(np.isfinite(y))) >= 2


def find_post_shadow_peak(
    profile_db: np.ndarray,
    *,
    shadow_start_x: float,
    peak_sigma: float = 1.5,
    prominence_db: float = 2.0,
    min_distance_pixels: int = 3,
    peak_mode: str = "first",
    respect_internal_nodata_gaps: bool = False,
    min_segment_pixels: int = 5,
) -> Optional[PeakPick]:
    """
    Find a significant peak after the row-specific shadow search start.

    For an EXCAVATED SimSAR, set ``respect_internal_nodata_gaps=True``.  If
    the profile contains data -> no-data -> data after the shadow start, peak
    picking is restricted to the FINAL contiguous finite-data segment.  Therefore a peak
    in the first data section can never be selected merely because the signal
    falls into a no-data gap.

    No-data is *not* interpolated for SimSAR.  It is preserved as a mask for
    picking and is only replaced by SIMSAR_NODATA_DB for plotting.
    """
    if peak_mode not in {"first", "most_prominent"}:
        raise ValueError("peak_mode must be 'first' or 'most_prominent'.")

    x_pixels = np.arange(PROFILE_X1, PROFILE_X2 + 1, dtype=float)
    profile_db = np.asarray(profile_db, dtype=float)

    if profile_db.size != x_pixels.size:
        raise ValueError(
            f"Profile has {profile_db.size} samples; expected {x_pixels.size}."
        )

    search = x_pixels >= float(shadow_start_x)
    x = x_pixels[search]
    y = profile_db[search]

    if respect_internal_nodata_gaps:
        runs = _finite_runs(np.isfinite(y))
        if not runs:
            return None

        # Critical behaviour: if there is data -> no-data -> data, discard
        # every earlier data section and search ONLY the final data segment.
        start, stop = runs[-1]
        if (stop - start) < max(3, int(min_segment_pixels)):
            return None

        x = x[start:stop]
        y_work = np.asarray(y[start:stop], dtype=float)
    else:
        y_work = _interpolate_finite_1d(y)
        if y_work is None:
            return None

    if peak_sigma > 0:
        y_smooth = gaussian_filter1d(y_work, sigma=peak_sigma, mode="nearest")
    else:
        y_smooth = y_work

    peaks, properties = find_peaks(
        y_smooth,
        prominence=prominence_db,
        distance=max(1, int(min_distance_pixels)),
    )

    if peaks.size == 0:
        return None

    if peak_mode == "first":
        chosen = 0
    else:
        chosen = int(np.argmax(properties["prominences"]))

    idx = int(peaks[chosen])
    return PeakPick(
        x_pixel=float(x[idx]),
        intensity_db=float(y_smooth[idx]),
        prominence_db=float(properties["prominences"][chosen]),
    )


def pick_dense_profile_set(
    profiles_db: Mapping[int, np.ndarray],
    *,
    peak_sigma: float,
    prominence_db: float,
    min_distance_pixels: int,
    peak_mode: str,
    respect_internal_nodata_gaps: bool = False,
) -> Dict[int, Optional[PeakPick]]:
    """Pick a post-shadow peak on every inversion azimuth row."""
    picks: Dict[int, Optional[PeakPick]] = {}

    for row in INVERSION_ROWS:
        picks[row] = find_post_shadow_peak(
            profiles_db[row],
            shadow_start_x=SHADOW_START_BY_ROW[row],
            peak_sigma=peak_sigma,
            prominence_db=prominence_db,
            min_distance_pixels=min_distance_pixels,
            peak_mode=peak_mode,
            respect_internal_nodata_gaps=respect_internal_nodata_gaps,
        )

    return picks


# =============================================================================
# SCORING / TABLES
# =============================================================================


def rmse(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(arr ** 2)))


def mae(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.mean(np.abs(arr)))


def observed_dense_peak_table(
    raw_picks: Mapping[int, Optional[PeakPick]],
    filtered_picks: Mapping[int, Optional[PeakPick]],
) -> pd.DataFrame:
    rows = []

    for row in INVERSION_ROWS:
        raw = raw_picks[row]
        filt = filtered_picks[row]

        raw_x = np.nan if raw is None else raw.x_pixel
        filt_x = np.nan if filt is None else filt.x_pixel
        delta_px = filt_x - raw_x if np.isfinite(raw_x) and np.isfinite(filt_x) else np.nan

        rows.append({
            "azimuth_row": row,
            "plot_profile": plot_label_for_row(row) if row in PLOT_PROFILE_ROWS else "",
            "shadow_start_x": SHADOW_START_BY_ROW[row],
            "raw_peak_x_px": raw_x,
            "raw_peak_prominence_db": np.nan if raw is None else raw.prominence_db,
            "filtered_peak_x_px": filt_x,
            "filtered_peak_prominence_db": np.nan if filt is None else filt.prominence_db,
            "filtered_minus_raw_px": delta_px,
            "filtered_minus_raw_m": delta_px * RANGE_PIXEL_SPACING_M if np.isfinite(delta_px) else np.nan,
        })

    return pd.DataFrame(rows)


def score_dense_model(
    run_id: str,
    sim_picks: Mapping[int, Optional[PeakPick]],
    raw_mli_picks: Mapping[int, Optional[PeakPick]],
    filtered_mli_picks: Mapping[int, Optional[PeakPick]],
    *,
    min_coverage: float = 1.0,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Score one model using ALL rows with valid filtered-MLI peaks.

    Ranking uses filtered MLI peak positions only.  By default the model must
    produce a SimSAR peak on 100% of rows where the filtered MLI has a valid
    peak.  --min-coverage can relax this if needed.
    """
    if not (0 < min_coverage <= 1.0):
        raise ValueError("min_coverage must be in (0, 1].")

    residual_rows: List[Dict[str, object]] = []

    filtered_errors_m: List[float] = []
    raw_errors_m: List[float] = []

    observed_valid_rows = [
        row for row in INVERSION_ROWS
        if filtered_mli_picks[row] is not None
    ]

    if not observed_valid_rows:
        return ({
            "run_id": run_id,
            "status": "no_filtered_mli_peaks",
            "n_rows_total": len(INVERSION_ROWS),
            "n_rows_observed_valid": 0,
            "n_rows_matched": 0,
            "coverage": 0.0,
            "rmse_filtered_m": np.nan,
            "mae_filtered_m": np.nan,
            "bias_filtered_m": np.nan,
            "rmse_raw_m": np.nan,
        }, residual_rows)

    matched = 0

    for row in INVERSION_ROWS:
        sim = sim_picks[row]
        raw = raw_mli_picks[row]
        filt = filtered_mli_picks[row]

        obs_valid = filt is not None
        sim_valid = sim is not None

        filtered_error_px = np.nan
        filtered_error_m = np.nan
        raw_error_px = np.nan
        raw_error_m = np.nan

        if obs_valid and sim_valid:
            assert filt is not None
            assert sim is not None
            filtered_error_px = sim.x_pixel - filt.x_pixel
            filtered_error_m = filtered_error_px * RANGE_PIXEL_SPACING_M
            filtered_errors_m.append(filtered_error_m)
            matched += 1

            if raw is not None:
                raw_error_px = sim.x_pixel - raw.x_pixel
                raw_error_m = raw_error_px * RANGE_PIXEL_SPACING_M
                raw_errors_m.append(raw_error_m)

        residual_rows.append({
            "run_id": run_id,
            "azimuth_row": row,
            "plot_profile": plot_label_for_row(row) if row in PLOT_PROFILE_ROWS else "",
            "shadow_start_x": SHADOW_START_BY_ROW[row],
            "filtered_mli_peak_x_px": np.nan if filt is None else filt.x_pixel,
            "filtered_mli_peak_prominence_db": np.nan if filt is None else filt.prominence_db,
            "raw_mli_peak_x_px": np.nan if raw is None else raw.x_pixel,
            "sim_peak_x_px": np.nan if sim is None else sim.x_pixel,
            "sim_peak_prominence_db": np.nan if sim is None else sim.prominence_db,
            "error_filtered_px": filtered_error_px,
            "error_filtered_m": filtered_error_m,
            "error_raw_px": raw_error_px,
            "error_raw_m": raw_error_m,
        })

    n_obs = len(observed_valid_rows)
    coverage = matched / n_obs

    status = "ok" if coverage >= min_coverage else "insufficient_peak_coverage"

    filtered_arr = np.asarray(filtered_errors_m, dtype=float)

    summary: Dict[str, object] = {
        "run_id": run_id,
        "status": status,
        "n_rows_total": len(INVERSION_ROWS),
        "n_rows_observed_valid": n_obs,
        "n_rows_matched": matched,
        "coverage": coverage,
        "rmse_filtered_m": rmse(filtered_errors_m) if status == "ok" else np.nan,
        "mae_filtered_m": mae(filtered_errors_m) if status == "ok" else np.nan,
        "bias_filtered_m": float(np.mean(filtered_arr)) if status == "ok" and filtered_arr.size else np.nan,
        "rmse_raw_m": rmse(raw_errors_m) if raw_errors_m else np.nan,
    }

    # Keep the three plotted-profile errors in the summary table for quick QA.
    for label, row in zip(PLOT_PROFILE_LABELS, PLOT_PROFILE_ROWS):
        sim = sim_picks[row]
        filt = filtered_mli_picks[row]
        if sim is not None and filt is not None:
            err_px = sim.x_pixel - filt.x_pixel
            summary[f"error_filtered_{label}_px"] = err_px
            summary[f"error_filtered_{label}_m"] = err_px * RANGE_PIXEL_SPACING_M
        else:
            summary[f"error_filtered_{label}_px"] = np.nan
            summary[f"error_filtered_{label}_m"] = np.nan

    return summary, residual_rows


# =============================================================================
# PLOTTING HELPERS — ONLY A/B/C ARE DISPLAYED
# =============================================================================


def subset_plot_profiles(
    dense_profiles: Mapping[int, np.ndarray],
) -> Dict[str, np.ndarray]:
    return {
        label: np.asarray(dense_profiles[row], dtype=float)
        for label, row in zip(PLOT_PROFILE_LABELS, PLOT_PROFILE_ROWS)
    }


def subset_plot_picks(
    dense_picks: Mapping[int, Optional[PeakPick]],
) -> Dict[str, Optional[PeakPick]]:
    return {
        label: dense_picks[row]
        for label, row in zip(PLOT_PROFILE_LABELS, PLOT_PROFILE_ROWS)
    }


def _display_shift_to_filtered_mli(
    sim_profiles: Mapping[str, np.ndarray],
    filtered_mli_profiles: Mapping[str, np.ndarray],
) -> float:
    sim = np.concatenate([np.asarray(sim_profiles[k], dtype=float) for k in PLOT_PROFILE_LABELS])
    obs = np.concatenate([np.asarray(filtered_mli_profiles[k], dtype=float) for k in PLOT_PROFILE_LABELS])
    common = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(common):
        return 0.0
    return float(np.median(obs[common]) - np.median(sim[common]))


def _profile_value_at_pick(profile: np.ndarray, pick: PeakPick) -> float:
    idx = int(round(pick.x_pixel - PROFILE_X1))
    if idx < 0 or idx >= len(profile):
        return np.nan
    value = float(profile[idx])
    return value if np.isfinite(value) else np.nan


def plot_observed_mli_profiles(
    raw_profiles: Mapping[str, np.ndarray],
    filtered_profiles: Mapping[str, np.ndarray],
    raw_picks: Mapping[str, Optional[PeakPick]],
    filtered_picks: Mapping[str, Optional[PeakPick]],
    output_png: Path,
    *,
    mli_name: str,
) -> None:
    """Plot raw vs filtered MLI only for the original A/B/C rows."""
    x = np.arange(PROFILE_X1, PROFILE_X2 + 1)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.6), sharex=True, constrained_layout=True)

    for ax, label, shadow_start in zip(
        axes, PLOT_PROFILE_LABELS, PLOT_SHADOW_START_XS
    ):
        ax.plot(x, raw_profiles[label], linewidth=0.9, alpha=0.55, label="MLI raw")
        ax.plot(x, filtered_profiles[label], linewidth=1.35, label="MLI median filtered")

        raw_pick = raw_picks[label]
        filt_pick = filtered_picks[label]

        ax.axvline(shadow_start, linestyle=":", linewidth=0.9,
                   label="Shadow-search start" if label == "A" else None)

        if raw_pick is not None:
            ax.axvline(raw_pick.x_pixel, linestyle="--", linewidth=0.9, alpha=0.6,
                       label="Raw peak" if label == "A" else None)
        if filt_pick is not None:
            ax.axvline(filt_pick.x_pixel, linestyle="-.", linewidth=1.1,
                       label="Filtered peak" if label == "A" else None)

        ax.set_ylabel(f"{label}\nIntensity (dB)")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_xlim(PROFILE_X1, PROFILE_X2)

    axes[-1].set_xlabel("Range pixel")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle(f"Unfiltered vs median-filtered MLI peak positions: {mli_name}")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top_models(
    top_models: pd.DataFrame,
    sim_profiles_by_id: Mapping[str, Mapping[str, np.ndarray]],
    raw_mli_profiles: Mapping[str, np.ndarray],
    filtered_mli_profiles: Mapping[str, np.ndarray],
    raw_mli_picks: Mapping[str, Optional[PeakPick]],
    filtered_mli_picks: Mapping[str, Optional[PeakPick]],
    sim_picks_by_id: Mapping[str, Mapping[str, Optional[PeakPick]]],
    output_png: Path,
    *,
    mli_name: str,
) -> None:
    """A/B/C-only profile plot for the top-ranked dense-inversion models."""
    if top_models.empty:
        raise RuntimeError("No valid models are available to plot.")

    nmodels = len(top_models)
    fig, axes = plt.subplots(
        3, nmodels, figsize=(3.1 * nmodels, 8.2), sharex=True,
        sharey="row", squeeze=False,
    )
    x = np.arange(PROFILE_X1, PROFILE_X2 + 1)

    for col, (_, model) in enumerate(top_models.iterrows()):
        run_id = str(model["run_id"])
        sim_profiles = sim_profiles_by_id[run_id]
        sim_picks = sim_picks_by_id[run_id]
        display_shift = _display_shift_to_filtered_mli(sim_profiles, filtered_mli_profiles)

        for r, (label, shadow_start) in enumerate(zip(PLOT_PROFILE_LABELS, PLOT_SHADOW_START_XS)):
            ax = axes[r, col]
            raw = raw_mli_profiles[label]
            filt = filtered_mli_profiles[label]
            sim_raw = np.asarray(sim_profiles[label], dtype=float)
            sim = np.where(
                np.isfinite(sim_raw),
                sim_raw + display_shift,
                SIMSAR_NODATA_DB,
            )

            ax.plot(x, raw, linewidth=0.8, alpha=0.45,
                    label="MLI raw" if (r == 0 and col == 0) else None)
            ax.plot(x, filt, linewidth=1.35,
                    label="MLI median filtered" if (r == 0 and col == 0) else None)
            ax.plot(x, sim, linewidth=1.15, linestyle="--",
                    label="SimSAR (display shifted)" if (r == 0 and col == 0) else None)
            ax.axvline(shadow_start, linewidth=0.8, linestyle=":", alpha=0.7)

            raw_pick = raw_mli_picks[label]
            filt_pick = filtered_mli_picks[label]
            sim_pick = sim_picks[label]

            if raw_pick is not None:
                ax.axvline(raw_pick.x_pixel, linewidth=0.75, linestyle="--", alpha=0.5)
            if filt_pick is not None:
                ax.axvline(filt_pick.x_pixel, linewidth=0.95, linestyle="--", alpha=0.8)
            if sim_pick is not None:
                ax.axvline(sim_pick.x_pixel, linewidth=1.15, linestyle="-.", alpha=0.9)

            ax.grid(alpha=0.18, linewidth=0.5)
            ax.set_xlim(PROFILE_X1, PROFILE_X2)
            if col == 0:
                ax.set_ylabel(f"Profile {label}\nIntensity (dB)")
            if r == 2:
                ax.set_xlabel("Range pixel")

        axes[0, col].set_title(
            f"Rank {int(model['rank_selected'])}: P.{run_id}.dem\n"
            f"Dense filtered RMSE = {model['rmse_filtered_m']:.2f} m\n"
            f"matched {int(model['n_rows_matched'])}/{int(model['n_rows_observed_valid'])} rows",
            fontsize=9,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.935))
    fig.suptitle(
        f"Observed MLI vs top {nmodels} models: {mli_name}\n"
        f"Ranking uses every azimuth row {INVERSION_ROW_MIN}..{INVERSION_ROW_MAX}; "
        "only A/B/C are plotted",
        fontsize=11, y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_top5_image_profile_summary(
    top_models: pd.DataFrame,
    simsar_dir: Path,
    simsar_pattern: str,
    mli_tif: Path,
    expected_shape: Tuple[int, int],
    sim_profiles_by_id: Mapping[str, Mapping[str, np.ndarray]],
    raw_mli_profiles: Mapping[str, np.ndarray],
    filtered_mli_profiles: Mapping[str, np.ndarray],
    raw_mli_picks: Mapping[str, Optional[PeakPick]],
    filtered_mli_picks: Mapping[str, Optional[PeakPick]],
    sim_picks_by_id: Mapping[str, Mapping[str, Optional[PeakPick]]],
    filtered_mli_picks_dense: Mapping[int, Optional[PeakPick]],
    sim_picks_dense_by_id: Mapping[str, Mapping[int, Optional[PeakPick]]],
    output_png: Path,
) -> None:
    """3 x 5 figure with dense picked-edge lines; A/B/C profiles remain the only profile curves shown."""
    if top_models.empty:
        raise RuntimeError("No valid models are available for the 3 x 5 plot.")

    models = top_models.head(5).copy()
    nmodels = len(models)
    mli_crop_db = read_radar_crop_db(mli_tif, expected_shape=expected_shape)

    fig, axes = plt.subplots(3, nmodels, figsize=(4.2 * nmodels, 11.2), squeeze=False)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(cycle) < 3:
        cycle = [f"C{i}" for i in range(3)]
    colors = {label: cycle[i] for i, label in enumerate(PLOT_PROFILE_LABELS)}
    x = np.arange(PROFILE_X1, PROFILE_X2 + 1)

    # Build the observed dense MLI picked edge before the per-model loop.
    # The SimSAR panel uses these coordinates before the middle MLI panel
    # is drawn, so they must already exist here.
    mli_edge_y = []
    mli_edge_x = []
    for row_y in INVERSION_ROWS:
        pick = filtered_mli_picks_dense.get(row_y)
        if pick is not None:
            mli_edge_y.append(row_y)
            mli_edge_x.append(pick.x_pixel)

    if mli_edge_x:
        order = np.argsort(mli_edge_y)
        mli_edge_y = np.asarray(mli_edge_y)[order]
        mli_edge_x = np.asarray(mli_edge_x)[order]

    for col, (_, model) in enumerate(models.iterrows()):
        run_id = str(model["run_id"])
        simsar_tif = Path(simsar_dir) / simsar_pattern.format(id=run_id)
        sim_crop_db = read_radar_crop_db(
            simsar_tif, expected_shape=expected_shape, add_epsilon=True
        )

        sim_profiles = sim_profiles_by_id[run_id]
        sim_picks = sim_picks_by_id[run_id]
        display_shift = _display_shift_to_filtered_mli(sim_profiles, filtered_mli_profiles)
        sim_crop_display = np.where(
            np.isfinite(sim_crop_db),
            sim_crop_db + display_shift,
            SIMSAR_NODATA_DB,
        )

        # Top row: SimSAR image with the full dense picked edge.
        ax = axes[0, col]
        ax.imshow(
            sim_crop_display, cmap="gray", vmin=DISPLAY_VMIN_DB, vmax=DISPLAY_VMAX_DB,
            origin="upper",
            extent=[IMAGE_X1, IMAGE_X2 + 1, IMAGE_Y2 + 1, IMAGE_Y1],
            interpolation="nearest",
        )

        sim_dense_picks = sim_picks_dense_by_id[run_id]
        sim_edge_y = []
        sim_edge_x = []
        for row_y in INVERSION_ROWS:
            pick = sim_dense_picks.get(row_y)
            if pick is not None:
                sim_edge_y.append(row_y)
                sim_edge_x.append(pick.x_pixel)

        if sim_edge_x:
            order = np.argsort(sim_edge_y)
            sim_edge_y = np.asarray(sim_edge_y)[order]
            sim_edge_x = np.asarray(sim_edge_x)[order]
            ax.plot(
                sim_edge_x, sim_edge_y,
                linewidth=2.2,
                marker=".", markersize=2.5,
                label="SimSAR picked edge",
                color="orange",
                zorder=7,
            )
            ax.plot(
                mli_edge_x, mli_edge_y,
                linewidth=2.2,
                marker=".", markersize=2.5,
                label="Filtered MLI picked edge",
                color="blue",
                zorder=7,
            )

        for label, row_y in zip(PLOT_PROFILE_LABELS, PLOT_PROFILE_ROWS):
            color = colors[label]
            ax.plot([PROFILE_X1, PROFILE_X2], [row_y, row_y], linewidth=0.8, color=color, alpha=0.65)
            ax.text(PROFILE_X1 + 2, row_y - 2, label, color=color, fontsize=8,
                    fontweight="bold", va="bottom")
        ax.set_xlim(IMAGE_X1, IMAGE_X2)
        ax.set_ylim(IMAGE_Y2, IMAGE_Y1)
        ax.set_aspect("equal")
        ax.set_title(
            f"Rank {int(model['rank_selected'])}: P.{run_id}.dem\n"
            f"Dense RMSE = {model['rmse_filtered_m']:.2f} m",
            fontsize=9,
        )
        if col == 0:
            ax.set_ylabel("SimSAR\nAzimuth line")
        else:
            ax.set_yticklabels([])

        # Middle row: observed MLI with the full filtered-MLI picked edge.
        ax = axes[1, col]
        ax.imshow(
            mli_crop_db, cmap="gray", vmin=DISPLAY_VMIN_DB, vmax=DISPLAY_VMAX_DB,
            origin="upper",
            extent=[IMAGE_X1, IMAGE_X2 + 1, IMAGE_Y2 + 1, IMAGE_Y1],
            interpolation="nearest",
        )

        mli_edge_y = []
        mli_edge_x = []
        for row_y in INVERSION_ROWS:
            pick = filtered_mli_picks_dense.get(row_y)
            if pick is not None:
                mli_edge_y.append(row_y)
                mli_edge_x.append(pick.x_pixel)

        if mli_edge_x:
            order = np.argsort(mli_edge_y)
            mli_edge_y = np.asarray(mli_edge_y)[order]
            mli_edge_x = np.asarray(mli_edge_x)[order]
            ax.plot(
                mli_edge_x, mli_edge_y,
                linewidth=2.2,
                marker=".", markersize=2.5,
                label="Filtered MLI picked edge",
                color="blue",
                zorder=7,
            )
            ax.plot(
                    sim_edge_x, sim_edge_y,
                    linewidth=2.2,
                    marker=".", markersize=2.5,
                    label="SimSAR picked edge",
                    color="orange",
                    zorder=7,
                )

        for label, row_y in zip(PLOT_PROFILE_LABELS, PLOT_PROFILE_ROWS):
            color = colors[label]
            ax.plot([PROFILE_X1, PROFILE_X2], [row_y, row_y], linewidth=0.8, color=color, alpha=0.65)
            ax.text(PROFILE_X1 + 2, row_y - 2, label, color=color, fontsize=8,
                    fontweight="bold", va="bottom")
        ax.set_xlim(IMAGE_X1, IMAGE_X2)
        ax.set_ylim(IMAGE_Y2, IMAGE_Y1)
        ax.set_aspect("equal")
        ax.set_xlabel("Range pixel")
        if col == 0:
            ax.set_ylabel("Observed MLI\nAzimuth line")
        else:
            ax.set_yticklabels([])

        # Bottom row: A/B/C profile curves only.
        ax = axes[2, col]
        error_lines = []

        for label in PLOT_PROFILE_LABELS:
            color = colors[label]
            raw = np.asarray(raw_mli_profiles[label], dtype=float)
            filt = np.asarray(filtered_mli_profiles[label], dtype=float)
            sim_raw = np.asarray(sim_profiles[label], dtype=float)
            sim = np.where(
                np.isfinite(sim_raw),
                sim_raw + display_shift,
                SIMSAR_NODATA_DB,
            )
            raw_pick = raw_mli_picks[label]
            filt_pick = filtered_mli_picks[label]
            sim_pick = sim_picks[label]

            ax.plot(x, raw, color=color, linewidth=0.75, linestyle=":", alpha=0.45)
            ax.plot(x, filt, color=color, linewidth=1.25, linestyle="-")
            ax.plot(x, sim, color=color, linewidth=1.05, linestyle="--")

            if raw_pick is not None:
                yv = _profile_value_at_pick(raw, raw_pick)
                if np.isfinite(yv):
                    ax.scatter(raw_pick.x_pixel, yv, s=24, marker="o", facecolors="none",
                               edgecolors=color, linewidths=1.0, zorder=5)
            if filt_pick is not None:
                yv = _profile_value_at_pick(filt, filt_pick)
                if np.isfinite(yv):
                    ax.scatter(filt_pick.x_pixel, yv, s=28, marker="x", color=color,
                               linewidths=1.2, zorder=6)
            if sim_pick is not None:
                yv = _profile_value_at_pick(sim, sim_pick)
                if np.isfinite(yv):
                    ax.scatter(sim_pick.x_pixel, yv, s=34, marker="s", facecolors="none",
                               edgecolors=color, linewidths=1.2, zorder=7)

            if filt_pick is not None and sim_pick is not None:
                err = (sim_pick.x_pixel - filt_pick.x_pixel) * RANGE_PIXEL_SPACING_M
                error_lines.append(f"{label}: {err:+.1f} m")
            else:
                error_lines.append(f"{label}: NA")

        ax.set_xlim(PROFILE_X1, PROFILE_X2)
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_xlabel("Range pixel")
        if col == 0:
            ax.set_ylabel("Profile intensity (dB)")
        ax.text(
            0.02, 0.03, "A/B/C peak error vs filtered MLI\n" + "  ".join(error_lines),
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    legend_handles = [
        Line2D([0], [0], linestyle="-", linewidth=2.2, marker=".", label="Dense picked edge (image panels)"),
        Line2D([0], [0], linestyle=":", linewidth=1.0, label="MLI raw profile"),
        Line2D([0], [0], linestyle="-", linewidth=1.3, label="MLI median-filtered profile"),
        Line2D([0], [0], linestyle="--", linewidth=1.1, label="SimSAR profile (display shifted)"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="none", label="Raw MLI peak (A/B/C)"),
        Line2D([0], [0], marker="x", linestyle="None", label="Filtered MLI peak (A/B/C)"),
        Line2D([0], [0], marker="s", linestyle="None", markerfacecolor="none", label="SimSAR peak (A/B/C)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.962), fontsize=8)
    fig.suptitle(
        f"Top-five dense peak-position inversion\n"
        f"Ranking and image-edge lines use rows {INVERSION_ROW_MIN}..{INVERSION_ROW_MAX}; bottom profiles show A/B/C only",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.0, w_pad=0.6)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN WORKFLOW
# =============================================================================


def build_run_ids(start: str, end: str, *, minimum_width: int = 4) -> List[str]:
    start_i = int(start)
    end_i = int(end)
    if end_i < start_i:
        raise ValueError("id_end must be >= id_start.")
    width = max(minimum_width, len(str(start)), len(str(end)))
    return [f"{i:0{width}d}" for i in range(start_i, end_i + 1)]


def run_peak_inversion(
    mli_tif: Path,
    simsar_dir: Path,
    run_ids: Iterable[str],
    output_dir: Path,
    *,
    simsar_pattern: str = "P.{id}.sim_sar.radar.tif",
    median_size: int = 15,
    peak_sigma: float = 1.5,
    peak_prominence_db: float = 2.0,
    peak_distance_pixels: int = 3,
    peak_mode: str = "first",
    min_coverage: float = 1.0,
    top_n: int = 5,
    azimuth_min: int = DEFAULT_INVERSION_ROW_MIN,
    azimuth_max: int = DEFAULT_INVERSION_ROW_MAX,
    interaction: str = "auto",
    provenance_dir: Optional[Path] = None,
    provenance_pattern: str = "{id}.json",
) -> Dict[str, object]:
    _configure_inversion_geometry(azimuth_min, azimuth_max)

    mli_tif = Path(mli_tif).resolve()
    simsar_dir = Path(simsar_dir).resolve()
    output_dir = Path(output_dir).resolve()
    provenance_dir = (
        None if provenance_dir is None else Path(provenance_dir).resolve()
    )

    if not mli_tif.exists():
        raise FileNotFoundError(mli_tif)
    if not simsar_dir.exists():
        raise FileNotFoundError(simsar_dir)
    if provenance_dir is not None and not provenance_dir.exists():
        raise FileNotFoundError(provenance_dir)
    if not (0 < min_coverage <= 1.0):
        raise ValueError("min_coverage must be in (0, 1].")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_ids = [str(x) for x in run_ids]
    if not run_ids:
        raise ValueError("No run IDs supplied.")

    print("\n" + "=" * 80)
    print("DENSE PEAK-POSITION SimSAR MODEL INVERSION")
    print("=" * 80)
    print(f"MLI:                    {mli_tif}")
    print(f"sim_sar directory:      {simsar_dir}")
    print(f"models requested:       {len(run_ids)}")
    print(f"inversion rows:         {INVERSION_ROW_MIN}..{INVERSION_ROW_MAX} ({len(INVERSION_ROWS)} rows)")
    print(f"plot rows only:         {PLOT_PROFILE_ROWS}")
    print(f"profile x range:        {PROFILE_X1}..{PROFILE_X2}")
    print(f"median filter:          {median_size} x {median_size}")
    print(f"peak sigma:             {peak_sigma} px")
    print(f"peak prominence:        {peak_prominence_db} dB")
    print(f"peak mode:              {peak_mode}")
    print(f"minimum model coverage: {min_coverage:.0%}")
    print("ranking basis:          filtered MLI peak positions ONLY")
    print(f"interaction mode:       {interaction}")
    if provenance_dir is not None:
        print(f"provenance directory:   {provenance_dir}")
    print(
        "excavation gap rule:    ONLY excavate_to_lower/subtract_thickness; "
        "fill/addition keep the first data section"
    )

    # ------------------------------------------------------------------
    # Observed MLI
    # ------------------------------------------------------------------
    print("\n[1/4] Preparing dense observed MLI profiles")

    raw_mli_dense, filtered_mli_dense, mli_shape = prepare_mli_dense_profiles(
        mli_tif, median_size=median_size
    )
    raw_mli_picks_dense = pick_dense_profile_set(
        raw_mli_dense,
        peak_sigma=peak_sigma,
        prominence_db=peak_prominence_db,
        min_distance_pixels=peak_distance_pixels,
        peak_mode=peak_mode,
    )
    filtered_mli_picks_dense = pick_dense_profile_set(
        filtered_mli_dense,
        peak_sigma=peak_sigma,
        prominence_db=peak_prominence_db,
        min_distance_pixels=peak_distance_pixels,
        peak_mode=peak_mode,
    )

    n_filtered_obs = sum(p is not None for p in filtered_mli_picks_dense.values())
    if n_filtered_obs == 0:
        raise RuntimeError(
            "No filtered MLI peaks were detected on any inversion row. "
            "Try lowering --peak-prominence-db or changing --peak-mode."
        )

    print(f"  filtered MLI valid peaks: {n_filtered_obs}/{len(INVERSION_ROWS)} rows")
    if n_filtered_obs < len(INVERSION_ROWS):
        print("  WARNING: rows without an observed filtered peak are attempted and written to CSV, but cannot constrain the inversion.")

    observed_df = observed_dense_peak_table(raw_mli_picks_dense, filtered_mli_picks_dense)
    observed_csv = output_dir / "observed_dense_peak_positions.csv"
    observed_df.to_csv(observed_csv, index=False)

    raw_plot_profiles = subset_plot_profiles(raw_mli_dense)
    filtered_plot_profiles = subset_plot_profiles(filtered_mli_dense)
    raw_plot_picks = subset_plot_picks(raw_mli_picks_dense)
    filtered_plot_picks = subset_plot_picks(filtered_mli_picks_dense)

    observed_png = output_dir / "observed_mli_filtered_vs_unfiltered.png"
    plot_observed_mli_profiles(
        raw_plot_profiles,
        filtered_plot_profiles,
        raw_plot_picks,
        filtered_plot_picks,
        observed_png,
        mli_name=mli_tif.name,
    )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    print("\n[2/4] Scoring models on every inversion row")

    score_rows: List[Dict[str, object]] = []
    all_residual_rows: List[Dict[str, object]] = []

    # Only retain A/B/C subsets in memory for valid models because plotting
    # does not need the other 38 profile arrays after scoring.
    sim_plot_profiles_by_id: Dict[str, Dict[str, np.ndarray]] = {}
    sim_plot_picks_by_id: Dict[str, Dict[str, Optional[PeakPick]]] = {}
    sim_dense_picks_by_id: Dict[str, Dict[int, Optional[PeakPick]]] = {}

    for index, run_id in enumerate(run_ids, start=1):
        simsar_tif = simsar_dir / simsar_pattern.format(id=run_id)
        model_interaction, use_excavation_gap_rule, provenance_path = (
            resolve_model_interaction(
                run_id,
                simsar_dir=simsar_dir,
                interaction=interaction,
                provenance_dir=provenance_dir,
                provenance_pattern=provenance_pattern,
            )
        )
        print(
            f"  [{index:>4}/{len(run_ids)}] {run_id} "
            f"[{model_interaction}]: ",
            end="",
            flush=True,
        )

        if not simsar_tif.exists():
            print("missing sim_sar")
            score_rows.append({
                "run_id": run_id,
                "interaction": model_interaction,
                "excavation_gap_rule_applied": bool(use_excavation_gap_rule),
                "provenance_json": None if provenance_path is None else str(provenance_path),
                "status": "missing_file",
                "n_rows_total": len(INVERSION_ROWS),
                "n_rows_observed_valid": n_filtered_obs,
                "n_rows_matched": 0,
                "coverage": 0.0,
                "rmse_filtered_m": np.nan,
                "mae_filtered_m": np.nan,
                "bias_filtered_m": np.nan,
                "rmse_raw_m": np.nan,
                "n_rows_internal_nodata_gap": np.nan,
            })
            continue

        try:
            sim_dense = read_simsar_dense_profiles(simsar_tif, expected_shape=mli_shape)
            internal_gap_rows = sum(
                has_internal_nodata_gap(
                    sim_dense[row],
                    shadow_start_x=SHADOW_START_BY_ROW[row],
                )
                for row in INVERSION_ROWS
            )

            sim_picks_dense = pick_dense_profile_set(
                sim_dense,
                peak_sigma=peak_sigma,
                prominence_db=peak_prominence_db,
                min_distance_pixels=peak_distance_pixels,
                peak_mode=peak_mode,
                respect_internal_nodata_gaps=use_excavation_gap_rule,
            )

            summary, residuals = score_dense_model(
                run_id,
                sim_picks_dense,
                raw_mli_picks_dense,
                filtered_mli_picks_dense,
                min_coverage=min_coverage,
            )
            summary["interaction"] = model_interaction
            summary["excavation_gap_rule_applied"] = bool(use_excavation_gap_rule)
            summary["provenance_json"] = (
                None if provenance_path is None else str(provenance_path)
            )
            summary["n_rows_internal_nodata_gap"] = int(internal_gap_rows)
            score_rows.append(summary)
            all_residual_rows.extend(residuals)

            if summary["status"] == "ok":
                sim_plot_profiles_by_id[run_id] = subset_plot_profiles(sim_dense)
                sim_plot_picks_by_id[run_id] = subset_plot_picks(sim_picks_dense)
                sim_dense_picks_by_id[run_id] = dict(sim_picks_dense)
                print(
                    f"dense filtered RMSE {float(summary['rmse_filtered_m']):.2f} m "
                    f"({int(summary['n_rows_matched'])}/{int(summary['n_rows_observed_valid'])} rows; "
                    f"{int(summary['n_rows_internal_nodata_gap'])} rows with internal no-data gaps; "
                    f"gap-rule={'ON' if use_excavation_gap_rule else 'OFF'})"
                )
            else:
                print(
                    f"{summary['status']} "
                    f"({int(summary['n_rows_matched'])}/{int(summary['n_rows_observed_valid'])} rows)"
                )

        except Exception as exc:
            print(f"failed: {exc}")
            score_rows.append({
                "run_id": run_id,
                "interaction": model_interaction,
                "excavation_gap_rule_applied": bool(use_excavation_gap_rule),
                "provenance_json": None if provenance_path is None else str(provenance_path),
                "status": f"error: {exc}",
                "n_rows_total": len(INVERSION_ROWS),
                "n_rows_observed_valid": n_filtered_obs,
                "n_rows_matched": 0,
                "coverage": 0.0,
                "rmse_filtered_m": np.nan,
                "mae_filtered_m": np.nan,
                "bias_filtered_m": np.nan,
                "rmse_raw_m": np.nan,
                "n_rows_internal_nodata_gap": np.nan,
            })

    ranking = pd.DataFrame(score_rows)
    residuals_df = pd.DataFrame(all_residual_rows)

    residuals_csv = output_dir / "dense_peak_model_residuals.csv"
    residuals_df.to_csv(residuals_csv, index=False)

    valid_mask = (
        (ranking["status"] == "ok")
        & np.isfinite(ranking["rmse_filtered_m"])
    )

    if not np.any(valid_mask):
        ranking_csv = output_dir / "peak_model_ranking.csv"
        ranking.to_csv(ranking_csv, index=False)
        raise RuntimeError(
            "No model met the dense peak-coverage requirement. "
            f"Partial ranking written to {ranking_csv}. Consider lowering --min-coverage if appropriate."
        )

    ranking["rank_selected"] = np.nan
    ranking.loc[valid_mask, "rank_selected"] = (
        ranking.loc[valid_mask, "rmse_filtered_m"].rank(method="min", ascending=True)
    )

    ranking = ranking.sort_values(
        by=["rmse_filtered_m", "run_id"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)

    ranking_csv = output_dir / "peak_model_ranking.csv"
    ranking.to_csv(ranking_csv, index=False)

    valid_ranking = ranking[
        (ranking["status"] == "ok") & np.isfinite(ranking["rmse_filtered_m"])
    ].copy()
    top_models = valid_ranking.head(max(1, int(top_n))).copy()

    # ------------------------------------------------------------------
    # Plot only A/B/C
    # ------------------------------------------------------------------
    print("\n[3/4] Plotting top models (A/B/C only)")

    top_png = output_dir / "top5_peak_profile_comparison.png"
    plot_top_models(
        top_models,
        sim_plot_profiles_by_id,
        raw_plot_profiles,
        filtered_plot_profiles,
        raw_plot_picks,
        filtered_plot_picks,
        sim_plot_picks_by_id,
        top_png,
        mli_name=mli_tif.name,
    )

    top_3x5_png = output_dir / "top5_3x5_image_mli_profile_comparison.png"
    plot_top5_image_profile_summary(
        top_models,
        simsar_dir,
        simsar_pattern,
        mli_tif,
        mli_shape,
        sim_plot_profiles_by_id,
        raw_plot_profiles,
        filtered_plot_profiles,
        raw_plot_picks,
        filtered_plot_picks,
        sim_plot_picks_by_id,
        filtered_mli_picks_dense,
        sim_dense_picks_by_id,
        top_3x5_png,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n[4/4] Best-fitting dense models")
    show_cols = [
        "rank_selected", "run_id", "rmse_filtered_m", "mae_filtered_m",
        "bias_filtered_m", "coverage", "n_rows_matched", "n_rows_observed_valid",
    ]
    print(top_models[show_cols].to_string(index=False))

    best = top_models.iloc[0]
    print("\nBEST FITTING MODEL — filtered MLI dense peak inversion")
    print(f"  P.{best['run_id']}.dem")
    print(f"  RMSE:      {best['rmse_filtered_m']:.3f} m")
    print(f"  MAE:       {best['mae_filtered_m']:.3f} m")
    print(f"  bias:      {best['bias_filtered_m']:+.3f} m")
    print(f"  coverage:  {best['coverage']:.1%}")
    print(f"  rows used: {int(best['n_rows_matched'])}/{int(best['n_rows_observed_valid'])}")

    print("\nOutputs:")
    for path in [observed_csv, residuals_csv, ranking_csv, observed_png, top_png, top_3x5_png]:
        print(f"  {path}")

    return {
        "best_run_id": str(best["run_id"]),
        "best_rmse_filtered_m": float(best["rmse_filtered_m"]),
        "ranking": ranking,
        "residuals": residuals_df,
        "observed_peaks": observed_df,
        "ranking_csv": ranking_csv,
        "residuals_csv": residuals_csv,
        "top5_plot": top_png,
        "top5_3x5_plot": top_3x5_png,
    }


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rank modified DEM models by filtered-MLI post-shadow peak positions "
            "over a configurable azimuth corridor, while plotting only A/B/C."
        )
    )
    parser.add_argument("mli_tif", type=Path)
    parser.add_argument("simsar_dir", type=Path)
    parser.add_argument("id_start")
    parser.add_argument("id_end")
    parser.add_argument("--output-dir", type=Path, default=Path("peak_model_inversion_dense"))
    parser.add_argument(
        "--simsar-pattern",
        default="P.{id}.sim_sar.radar.tif",
        help="Filename pattern containing {id}.",
    )
    parser.add_argument("--median-size", type=int, default=15)
    parser.add_argument("--peak-sigma", type=float, default=1.5)
    parser.add_argument("--peak-prominence-db", type=float, default=2.0)
    parser.add_argument("--peak-distance-pixels", type=int, default=3)
    parser.add_argument(
        "--peak-mode", choices=("first", "most_prominent"), default="first"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help=(
            "Fraction of filtered-MLI-valid rows on which a model must also "
            "produce a peak to be ranked. Default 1.0 (100%%)."
        ),
    )
    parser.add_argument(
        "--azimuth-min",
        type=int,
        default=DEFAULT_INVERSION_ROW_MIN,
        help=(
            f"First azimuth row used in inversion. Default: {DEFAULT_INVERSION_ROW_MIN}. "
            "Can extend below profile C."
        ),
    )
    parser.add_argument(
        "--azimuth-max",
        type=int,
        default=DEFAULT_INVERSION_ROW_MAX,
        help=(
            f"Last azimuth row used in inversion. Default: {DEFAULT_INVERSION_ROW_MAX}. "
            "Can extend above profile A."
        ),
    )
    parser.add_argument(
        "--interaction",
        choices=(
            "auto",
            "excavate_to_lower",
            "subtract_thickness",
            "fill_to_upper",
            "add_thickness",
        ),
        default="auto",
        help=(
            "Interaction handling. 'auto' reads each model run JSON; otherwise "
            "force one interaction for all requested IDs."
        ),
    )
    parser.add_argument(
        "--provenance-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing run JSON files (normally the synthetic DEM "
            "directory). Required for reliable per-model --interaction auto."
        ),
    )
    parser.add_argument(
        "--provenance-pattern",
        default="{id}.json",
        help="Run-JSON filename pattern containing {id}. Default: {id}.json",
    )
    parser.add_argument("--top-n", type=int, default=5)

    args = parser.parse_args()

    if "{id}" not in args.simsar_pattern:
        parser.error("--simsar-pattern must contain '{id}'.")
    if "{id}" not in args.provenance_pattern:
        parser.error("--provenance-pattern must contain '{id}'.")

    run_ids = build_run_ids(args.id_start, args.id_end)

    run_peak_inversion(
        mli_tif=args.mli_tif,
        simsar_dir=args.simsar_dir,
        run_ids=run_ids,
        output_dir=args.output_dir,
        simsar_pattern=args.simsar_pattern,
        median_size=args.median_size,
        peak_sigma=args.peak_sigma,
        peak_prominence_db=args.peak_prominence_db,
        peak_distance_pixels=args.peak_distance_pixels,
        peak_mode=args.peak_mode,
        min_coverage=args.min_coverage,
        top_n=args.top_n,
        azimuth_min=args.azimuth_min,
        azimuth_max=args.azimuth_max,
        interaction=args.interaction,
        provenance_dir=args.provenance_dir,
        provenance_pattern=args.provenance_pattern,
    )


if __name__ == "__main__":
    main()