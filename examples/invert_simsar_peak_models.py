
"""
Rank modified DEM models by post-shadow SAR peak position.

This script compares one observed MLI against a range of radar-coordinate
simulated SAR GeoTIFFs named, by default:

    P.{ID}.sim_sar.radar.tif

It uses the three fixed horizontal profiles from Radar_Shadows_new_xarray.ipynb:

    Profile A: azimuth row 2030, shadow start x=812
    Profile B: azimuth row 2010, shadow start x=823
    Profile C: azimuth row 1990, shadow start x=826
    Profile range: x=775..900 inclusive
    Range spacing: 2.728212 m/pixel

For the observed MLI it makes two versions:

1. Unfiltered MLI (linear intensity -> dB)
2. 2-D median-filtered MLI, filtered in linear intensity then converted to dB

Peak picking is then performed on the same A/B/C profiles for:

- unfiltered observed MLI
- filtered observed MLI
- each simulated SAR model

The model score is based ONLY on peak POSITION, not peak amplitude.  A small
1-D Gaussian smoothing is used inside the peak picker to suppress pixel-scale
noise.  The underlying unfiltered/filtered profiles are still kept separately.

For each model the script reports:

- RMSE against unfiltered MLI peak positions
- RMSE against filtered MLI peak positions
- RMSE against filtered MLI peak positions
- an optional combined RMSE using all 6 residuals (3 profiles x 2 MLI forms)

By default the inversion is ranked by the FILTERED MLI RMSE.  This is more
defensible than treating raw and filtered versions of the same observation as
independent data.  Use --rank-by raw or --rank-by combined for sensitivity
tests.  The top five models under the selected ranking are plotted in a 3 x 5
profile figure.
The simulated curves are median-shifted in dB FOR DISPLAY ONLY; this shift does
not affect peak locations or model scores.

Dependencies
------------
numpy, pandas, matplotlib, rasterio, scipy

Example
-------
python invert_simsar_peak_models.py \
    ./radar_mli/20201226.mli.tif \
    ./sim_sar \
    1 100 \
    --output-dir ./peak_inversion_20201226 \
    --median-size 15 \
    --peak-prominence-db 2.0 \
    --peak-sigma 1.5 \
    --peak-mode first
"""

from __future__ import annotations

import argparse
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


# =============================================================================
# FIXED PROFILE GEOMETRY FROM Radar_Shadows_new_xarray.ipynb
# =============================================================================

PROFILE_X1 = 775
PROFILE_X2 = 900

PROFILE_LABELS = ("A", "B", "C")
PROFILE_ROWS = (2030, 2010, 1990)
SHADOW_START_XS = (812, 823, 826)

RANGE_PIXEL_SPACING_M = 2.728212


@dataclass(frozen=True)
class PeakPick:
    """One detected peak along a radar-coordinate range profile."""

    x_pixel: float
    intensity_db: float
    prominence_db: float


# =============================================================================
# BASIC RASTER / INTENSITY HELPERS
# =============================================================================


def log_intensity(data: np.ndarray) -> np.ndarray:
    """Convert positive linear intensity to dB using 10*log10(intensity)."""

    data = np.asarray(data, dtype=np.float32)

    out = np.full(
        data.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = np.isfinite(data) & (data > 0)

    out[valid] = 10.0 * np.log10(data[valid])

    return out


def raster_shape(path: Path) -> Tuple[int, int]:
    """Return raster shape as (rows, columns)."""

    with rasterio.open(path) as src:
        return src.height, src.width


def _validate_profile_geometry(shape: Tuple[int, int]) -> None:
    """Check that the fixed A/B/C profiles fit inside a raster."""

    nrows, ncols = shape

    if max(PROFILE_ROWS) >= nrows:
        raise ValueError(
            f"Profile row {max(PROFILE_ROWS)} is outside raster with {nrows} rows."
        )

    if PROFILE_X2 >= ncols:
        raise ValueError(
            f"Profile x={PROFILE_X2} is outside raster with {ncols} columns."
        )


def _read_corridor(
    path: Path,
    *,
    pad: int = 0,
) -> Tuple[np.ndarray, int, int, Tuple[int, int]]:
    """
    Read the smallest rectangular raster corridor containing all three profiles.

    pad is added on every side.  For the MLI median filter, pad should be at
    least median_size//2 so the extracted profile pixels see the same local
    neighbourhood they would see in a full-image median filter.

    Returns
    -------
    data : ndarray
        Linear-intensity raster values for the corridor.
    row0, col0 : int
        Absolute raster index represented by data[0, 0].
    shape : tuple
        Full source raster shape.
    """

    path = Path(path)

    with rasterio.open(path) as src:

        full_shape = (src.height, src.width)
        _validate_profile_geometry(full_shape)

        row0 = max(0, min(PROFILE_ROWS) - pad)
        row1 = min(src.height, max(PROFILE_ROWS) + 1 + pad)

        col0 = max(0, PROFILE_X1 - pad)
        col1 = min(src.width, PROFILE_X2 + 1 + pad)

        window = Window(
            col_off=col0,
            row_off=row0,
            width=col1 - col0,
            height=row1 - row0,
        )

        arr = src.read(
            1,
            window=window,
            masked=True,
        )

    if np.ma.isMaskedArray(arr):
        data = arr.filled(np.nan).astype(np.float32)
    else:
        data = np.asarray(arr, dtype=np.float32)

    return data, row0, col0, full_shape


def _extract_profiles_from_corridor(
    data: np.ndarray,
    row0: int,
    col0: int,
) -> Dict[str, np.ndarray]:
    """Extract A/B/C arrays from a corridor using absolute radar indices."""

    local_x1 = PROFILE_X1 - col0
    local_x2 = PROFILE_X2 - col0

    profiles: Dict[str, np.ndarray] = {}

    for label, row in zip(PROFILE_LABELS, PROFILE_ROWS):

        local_row = row - row0

        profile = data[
            local_row,
            local_x1:local_x2 + 1,
        ]

        expected = PROFILE_X2 - PROFILE_X1 + 1

        if profile.size != expected:
            raise ValueError(
                f"Profile {label} has {profile.size} samples; expected {expected}."
            )

        profiles[label] = np.asarray(profile, dtype=np.float32)

    return profiles


# =============================================================================
# MLI PREPARATION
# =============================================================================


def prepare_mli_profiles(
    mli_tif: Path,
    *,
    median_size: int = 15,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Tuple[int, int],
]:
    """
    Extract unfiltered and median-filtered MLI A/B/C profiles in dB.

    The 2-D median filter is applied in LINEAR intensity space, matching the
    existing comparison approach in the repository.  Only a padded profile
    corridor is filtered because no pixels outside these A/B/C profiles are
    needed for this inversion.
    """

    if median_size < 1 or median_size % 2 == 0:
        raise ValueError("median_size must be a positive odd integer.")

    pad = median_size // 2

    mli_linear, row0, col0, full_shape = _read_corridor(
        Path(mli_tif),
        pad=pad,
    )

    valid = np.isfinite(mli_linear) & (mli_linear > 0)

    if not np.any(valid):
        raise ValueError("MLI profile corridor contains no finite positive pixels.")

    # Same invalid-value strategy as the existing compare_simsar_mli.py:
    # fill invalid pixels with a representative median during filtering, then
    # restore invalid source pixels afterwards.
    work = mli_linear.copy()
    fill_value = float(np.median(mli_linear[valid]))
    work[~valid] = fill_value

    filtered_linear = median_filter(
        work,
        size=median_size,
        mode="nearest",
    ).astype(np.float32)

    filtered_linear[~valid] = np.nan

    raw_profiles_linear = _extract_profiles_from_corridor(
        mli_linear,
        row0,
        col0,
    )

    filtered_profiles_linear = _extract_profiles_from_corridor(
        filtered_linear,
        row0,
        col0,
    )

    raw_profiles_db = {
        label: log_intensity(values)
        for label, values in raw_profiles_linear.items()
    }

    filtered_profiles_db = {
        label: log_intensity(values)
        for label, values in filtered_profiles_linear.items()
    }

    return raw_profiles_db, filtered_profiles_db, full_shape


def read_simsar_profiles(
    simsar_tif: Path,
    *,
    expected_shape: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """Read one simulated SAR model and return A/B/C profiles in dB."""

    data, row0, col0, full_shape = _read_corridor(
        Path(simsar_tif),
        pad=0,
    )

    if full_shape != expected_shape:
        raise ValueError(
            "sim_sar and MLI raster shapes differ: "
            f"sim_sar={full_shape}, MLI={expected_shape}. "
            "No resampling is performed."
        )

    profiles_linear = _extract_profiles_from_corridor(
        data,
        row0,
        col0,
    )

    return {
        label: log_intensity(values + 1e-12)
        for label, values in profiles_linear.items()
    }


# =============================================================================
# PEAK PICKING
# =============================================================================


def _interpolate_finite_1d(values: np.ndarray) -> Optional[np.ndarray]:
    """Fill internal NaNs by 1-D interpolation; return None if too few values."""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)

    if finite.sum() < 3:
        return None

    x = np.arange(values.size, dtype=float)

    out = values.copy()
    out[~finite] = np.interp(
        x[~finite],
        x[finite],
        values[finite],
    )

    return out


def find_post_shadow_peak(
    profile_db: np.ndarray,
    *,
    shadow_start_x: int,
    peak_sigma: float = 1.5,
    prominence_db: float = 2.0,
    min_distance_pixels: int = 3,
    peak_mode: str = "first",
) -> Optional[PeakPick]:
    """
    Find a significant intensity peak after the known shadow start.

    Parameters
    ----------
    profile_db
        One A/B/C profile covering PROFILE_X1..PROFILE_X2 inclusive.
    shadow_start_x
        Absolute radar range-column where the post-shadow search begins.
    peak_sigma
        Gaussian sigma (pixels) used only inside the peak detector. Set to 0
        to disable this small 1-D smoothing.
    prominence_db
        Minimum scipy.signal.find_peaks prominence in dB.
    min_distance_pixels
        Minimum separation between candidate peaks.
    peak_mode
        "first" chooses the first significant post-shadow peak.
        "most_prominent" chooses the candidate with the largest prominence.

    Returns
    -------
    PeakPick or None
    """

    if peak_mode not in {"first", "most_prominent"}:
        raise ValueError("peak_mode must be 'first' or 'most_prominent'.")

    x_pixels = np.arange(
        PROFILE_X1,
        PROFILE_X2 + 1,
        dtype=float,
    )

    profile_db = np.asarray(profile_db, dtype=float)

    if profile_db.size != x_pixels.size:
        raise ValueError(
            f"Profile has {profile_db.size} samples; expected {x_pixels.size}."
        )

    search = x_pixels >= shadow_start_x

    x = x_pixels[search]
    y = profile_db[search]

    y_work = _interpolate_finite_1d(y)

    if y_work is None:
        return None

    if peak_sigma > 0:
        y_smooth = gaussian_filter1d(
            y_work,
            sigma=peak_sigma,
            mode="nearest",
        )
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


def pick_profile_set(
    profiles_db: Mapping[str, np.ndarray],
    *,
    peak_sigma: float,
    prominence_db: float,
    min_distance_pixels: int,
    peak_mode: str,
) -> Dict[str, Optional[PeakPick]]:
    """Detect A/B/C post-shadow peaks for one image/profile set."""

    picks: Dict[str, Optional[PeakPick]] = {}

    for label, shadow_start in zip(PROFILE_LABELS, SHADOW_START_XS):

        picks[label] = find_post_shadow_peak(
            profiles_db[label],
            shadow_start_x=shadow_start,
            peak_sigma=peak_sigma,
            prominence_db=prominence_db,
            min_distance_pixels=min_distance_pixels,
            peak_mode=peak_mode,
        )

    return picks


def _require_observed_peaks(
    picks: Mapping[str, Optional[PeakPick]],
    name: str,
) -> None:
    """Fail early if peak settings do not find all observed A/B/C peaks."""

    missing = [
        label
        for label in PROFILE_LABELS
        if picks[label] is None
    ]

    if missing:
        raise RuntimeError(
            f"No {name} MLI peak detected for profile(s): {', '.join(missing)}. "
            "Try lowering --peak-prominence-db, changing --peak-sigma, or "
            "using --peak-mode most_prominent."
        )


# =============================================================================
# MODEL SCORING
# =============================================================================


def rmse(values: Sequence[float]) -> float:
    """Root-mean-square of finite values."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(arr ** 2)))


def score_model(
    run_id: str,
    sim_picks: Mapping[str, Optional[PeakPick]],
    raw_mli_picks: Mapping[str, Optional[PeakPick]],
    filtered_mli_picks: Mapping[str, Optional[PeakPick]],
) -> Dict[str, object]:
    """
    Score one model against unfiltered and filtered MLI peak positions.

    A model is considered valid only if all three simulated A/B/C peaks were
    detected.  This prevents a model with missing constraints from appearing
    artificially good because its RMSE used fewer profiles.
    """

    row: Dict[str, object] = {
        "run_id": run_id,
        "status": "ok",
    }

    missing_sim = [
        label
        for label in PROFILE_LABELS
        if sim_picks[label] is None
    ]

    if missing_sim:
        row["status"] = "missing_sim_peak_" + "_".join(missing_sim)
        row["rmse_raw_m"] = np.nan
        row["rmse_filtered_m"] = np.nan
        row["rmse_combined_m"] = np.nan
        return row

    raw_errors_m: List[float] = []
    filtered_errors_m: List[float] = []

    for label in PROFILE_LABELS:

        sim = sim_picks[label]
        obs_raw = raw_mli_picks[label]
        obs_filtered = filtered_mli_picks[label]

        # Observed peaks are checked before model scoring, so these are safe.
        assert sim is not None
        assert obs_raw is not None
        assert obs_filtered is not None

        raw_error_px = sim.x_pixel - obs_raw.x_pixel
        filtered_error_px = sim.x_pixel - obs_filtered.x_pixel

        raw_error_m = raw_error_px * RANGE_PIXEL_SPACING_M
        filtered_error_m = filtered_error_px * RANGE_PIXEL_SPACING_M

        raw_errors_m.append(raw_error_m)
        filtered_errors_m.append(filtered_error_m)

        row[f"sim_peak_{label}_px"] = sim.x_pixel
        row[f"sim_peak_{label}_prominence_db"] = sim.prominence_db

        row[f"raw_mli_peak_{label}_px"] = obs_raw.x_pixel
        row[f"filtered_mli_peak_{label}_px"] = obs_filtered.x_pixel

        row[f"error_raw_{label}_px"] = raw_error_px
        row[f"error_raw_{label}_m"] = raw_error_m

        row[f"error_filtered_{label}_px"] = filtered_error_px
        row[f"error_filtered_{label}_m"] = filtered_error_m

    row["rmse_raw_m"] = rmse(raw_errors_m)
    row["rmse_filtered_m"] = rmse(filtered_errors_m)

    combined_errors = raw_errors_m + filtered_errors_m
    row["rmse_combined_m"] = rmse(combined_errors)

    return row


# =============================================================================
# OUTPUT TABLES
# =============================================================================


def observed_peak_table(
    raw_picks: Mapping[str, Optional[PeakPick]],
    filtered_picks: Mapping[str, Optional[PeakPick]],
) -> pd.DataFrame:
    """Build a compact table comparing raw and filtered observed MLI peaks."""

    rows = []

    for label, row_y, shadow_start in zip(
        PROFILE_LABELS,
        PROFILE_ROWS,
        SHADOW_START_XS,
    ):

        raw = raw_picks[label]
        filt = filtered_picks[label]

        assert raw is not None
        assert filt is not None

        delta_px = filt.x_pixel - raw.x_pixel

        rows.append({
            "profile": label,
            "azimuth_row": row_y,
            "shadow_start_x": shadow_start,
            "raw_peak_x_px": raw.x_pixel,
            "raw_peak_prominence_db": raw.prominence_db,
            "filtered_peak_x_px": filt.x_pixel,
            "filtered_peak_prominence_db": filt.prominence_db,
            "filtered_minus_raw_px": delta_px,
            "filtered_minus_raw_m": delta_px * RANGE_PIXEL_SPACING_M,
        })

    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================


def _display_shift_to_filtered_mli(
    sim_profiles: Mapping[str, np.ndarray],
    filtered_mli_profiles: Mapping[str, np.ndarray],
) -> float:
    """
    Median dB shift for plotting the simulated profiles on the MLI scale.

    This is deliberately NOT used for peak picking or model scoring.
    """

    sim = np.concatenate([
        np.asarray(sim_profiles[label], dtype=float)
        for label in PROFILE_LABELS
    ])

    obs = np.concatenate([
        np.asarray(filtered_mli_profiles[label], dtype=float)
        for label in PROFILE_LABELS
    ])

    common = np.isfinite(sim) & np.isfinite(obs)

    if not np.any(common):
        return 0.0

    return float(
        np.median(obs[common])
        - np.median(sim[common])
    )


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
    ranking_metric: str,
    ranking_label: str,
) -> None:
    """Plot A/B/C profile comparisons for the top-ranked models."""

    if top_models.empty:
        raise RuntimeError("No valid models are available to plot.")

    nmodels = len(top_models)

    fig, axes = plt.subplots(
        len(PROFILE_LABELS),
        nmodels,
        figsize=(3.1 * nmodels, 8.2),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    x = np.arange(PROFILE_X1, PROFILE_X2 + 1)

    for col, (_, model) in enumerate(top_models.iterrows()):

        run_id = str(model["run_id"])

        sim_profiles = sim_profiles_by_id[run_id]
        sim_picks = sim_picks_by_id[run_id]

        display_shift = _display_shift_to_filtered_mli(
            sim_profiles,
            filtered_mli_profiles,
        )

        for row, (label, shadow_start) in enumerate(
            zip(PROFILE_LABELS, SHADOW_START_XS)
        ):

            ax = axes[row, col]

            raw = raw_mli_profiles[label]
            filt = filtered_mli_profiles[label]
            sim = sim_profiles[label] + display_shift

            ax.plot(
                x,
                raw,
                linewidth=0.8,
                alpha=0.45,
                label="MLI raw" if (row == 0 and col == 0) else None,
            )

            ax.plot(
                x,
                filt,
                linewidth=1.35,
                label="MLI median filtered" if (row == 0 and col == 0) else None,
            )

            ax.plot(
                x,
                sim,
                linewidth=1.15,
                linestyle="--",
                label="sim_sar (display shifted)" if (row == 0 and col == 0) else None,
            )

            ax.axvline(
                shadow_start,
                linewidth=0.8,
                linestyle=":",
                alpha=0.7,
            )

            raw_pick = raw_mli_picks[label]
            filt_pick = filtered_mli_picks[label]
            sim_pick = sim_picks[label]

            assert raw_pick is not None
            assert filt_pick is not None
            assert sim_pick is not None

            ax.axvline(
                raw_pick.x_pixel,
                linewidth=0.75,
                linestyle="--",
                alpha=0.5,
            )

            ax.axvline(
                filt_pick.x_pixel,
                linewidth=0.95,
                linestyle="--",
                alpha=0.8,
            )

            ax.axvline(
                sim_pick.x_pixel,
                linewidth=1.15,
                linestyle="-.",
                alpha=0.9,
            )

            # Put detected peaks directly on the displayed curves.
            raw_idx = int(round(raw_pick.x_pixel - PROFILE_X1))
            filt_idx = int(round(filt_pick.x_pixel - PROFILE_X1))
            sim_idx = int(round(sim_pick.x_pixel - PROFILE_X1))

            if 0 <= raw_idx < raw.size and np.isfinite(raw[raw_idx]):
                ax.scatter(raw_pick.x_pixel, raw[raw_idx], s=16, zorder=5)

            if 0 <= filt_idx < filt.size and np.isfinite(filt[filt_idx]):
                ax.scatter(filt_pick.x_pixel, filt[filt_idx], s=18, zorder=5)

            if 0 <= sim_idx < sim.size and np.isfinite(sim[sim_idx]):
                ax.scatter(sim_pick.x_pixel, sim[sim_idx], s=20, zorder=5)

            ax.grid(
                alpha=0.18,
                linewidth=0.5,
            )

            ax.set_xlim(PROFILE_X1, PROFILE_X2)

            if col == 0:
                ax.set_ylabel(
                    f"Profile {label}\nIntensity (dB)"
                )

            if row == len(PROFILE_LABELS) - 1:
                ax.set_xlabel("Range pixel")

        axes[0, col].set_title(
            f"Rank {int(model['rank_selected'])}: P.{run_id}.dem\n"
            f"{ranking_label} RMSE = {model[ranking_metric]:.2f} m\n"
            f"raw={model['rmse_raw_m']:.2f}, filtered={model['rmse_filtered_m']:.2f} m",
            fontsize=9,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.935),
    )

    fig.suptitle(
        f"Observed MLI vs top {nmodels} simulated DEM models: {mli_name}\n"
        f"Ranking: {ranking_label} A/B/C post-shadow peak-position RMSE; "
        "sim_sar dB shifts are for display only",
        fontsize=11,
        y=0.985,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_observed_mli_profiles(
    raw_profiles: Mapping[str, np.ndarray],
    filtered_profiles: Mapping[str, np.ndarray],
    raw_picks: Mapping[str, Optional[PeakPick]],
    filtered_picks: Mapping[str, Optional[PeakPick]],
    output_png: Path,
    *,
    mli_name: str,
) -> None:
    """Plot raw vs filtered observed MLI and their detected peak positions."""

    x = np.arange(PROFILE_X1, PROFILE_X2 + 1)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 6.6),
        sharex=True,
        constrained_layout=True,
    )

    for ax, label, shadow_start in zip(
        axes,
        PROFILE_LABELS,
        SHADOW_START_XS,
    ):

        ax.plot(
            x,
            raw_profiles[label],
            linewidth=0.9,
            alpha=0.55,
            label="MLI raw",
        )

        ax.plot(
            x,
            filtered_profiles[label],
            linewidth=1.35,
            label="MLI median filtered",
        )

        raw_pick = raw_picks[label]
        filt_pick = filtered_picks[label]

        assert raw_pick is not None
        assert filt_pick is not None

        ax.axvline(
            shadow_start,
            linestyle=":",
            linewidth=0.9,
            label="Shadow-search start" if label == "A" else None,
        )

        ax.axvline(
            raw_pick.x_pixel,
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            label="Raw peak" if label == "A" else None,
        )

        ax.axvline(
            filt_pick.x_pixel,
            linestyle="-.",
            linewidth=1.1,
            label="Filtered peak" if label == "A" else None,
        )

        ax.set_ylabel(f"{label}\nIntensity (dB)")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_xlim(PROFILE_X1, PROFILE_X2)

    axes[-1].set_xlabel("Range pixel")
    axes[0].legend(frameon=False, ncol=2)

    fig.suptitle(
        f"Unfiltered vs median-filtered MLI peak positions: {mli_name}"
    )

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN INVERSION WORKFLOW
# =============================================================================


def build_run_ids(
    start: str,
    end: str,
    *,
    minimum_width: int = 4,
) -> List[str]:
    """Build an inclusive zero-padded integer ID range."""

    start_i = int(start)
    end_i = int(end)

    if end_i < start_i:
        raise ValueError("id_end must be >= id_start.")

    width = max(
        minimum_width,
        len(str(start)),
        len(str(end)),
    )

    return [
        f"{i:0{width}d}"
        for i in range(start_i, end_i + 1)
    ]


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
    top_n: int = 5,
    rank_by: str = "filtered",
) -> Dict[str, object]:
    """Run the full one-MLI peak-position inversion."""

    mli_tif = Path(mli_tif).resolve()
    simsar_dir = Path(simsar_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not mli_tif.exists():
        raise FileNotFoundError(mli_tif)

    if not simsar_dir.exists():
        raise FileNotFoundError(simsar_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    run_ids = [str(run_id) for run_id in run_ids]

    if not run_ids:
        raise ValueError("No run IDs were supplied.")

    print("\n" + "=" * 78)
    print("PEAK-POSITION SimSAR MODEL INVERSION")
    print("=" * 78)

    print(f"MLI:                 {mli_tif}")
    print(f"sim_sar directory:   {simsar_dir}")
    print(f"models requested:    {len(run_ids)}")
    print(f"median filter:       {median_size} x {median_size}")
    print(f"peak sigma:          {peak_sigma} pixels")
    print(f"peak prominence:     {peak_prominence_db} dB")
    if rank_by not in {"filtered", "raw", "combined"}:
        raise ValueError("rank_by must be 'filtered', 'raw', or 'combined'.")

    print(f"peak mode:           {peak_mode}")
    print(f"ranking basis:       {rank_by}")
    print(f"profile x range:     {PROFILE_X1}..{PROFILE_X2}")
    print(f"profile rows:        {PROFILE_ROWS}")
    print(f"shadow starts:       {SHADOW_START_XS}")

    # -------------------------------------------------------------------------
    # Observed MLI
    # -------------------------------------------------------------------------

    print("\n[1/4] Preparing observed MLI profiles")

    raw_mli_profiles, filtered_mli_profiles, mli_shape = prepare_mli_profiles(
        mli_tif,
        median_size=median_size,
    )

    raw_mli_picks = pick_profile_set(
        raw_mli_profiles,
        peak_sigma=peak_sigma,
        prominence_db=peak_prominence_db,
        min_distance_pixels=peak_distance_pixels,
        peak_mode=peak_mode,
    )

    filtered_mli_picks = pick_profile_set(
        filtered_mli_profiles,
        peak_sigma=peak_sigma,
        prominence_db=peak_prominence_db,
        min_distance_pixels=peak_distance_pixels,
        peak_mode=peak_mode,
    )

    _require_observed_peaks(raw_mli_picks, "unfiltered")
    _require_observed_peaks(filtered_mli_picks, "filtered")

    observed_df = observed_peak_table(
        raw_mli_picks,
        filtered_mli_picks,
    )

    observed_csv = output_dir / "observed_mli_peak_positions.csv"
    observed_df.to_csv(observed_csv, index=False)

    observed_png = output_dir / "observed_mli_filtered_vs_unfiltered.png"
    plot_observed_mli_profiles(
        raw_mli_profiles,
        filtered_mli_profiles,
        raw_mli_picks,
        filtered_mli_picks,
        observed_png,
        mli_name=mli_tif.name,
    )

    print("\nObserved peak positions:")
    print(observed_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------

    print("\n[2/4] Scoring simulated SAR models")

    score_rows: List[Dict[str, object]] = []

    sim_profiles_by_id: Dict[str, Dict[str, np.ndarray]] = {}
    sim_picks_by_id: Dict[str, Dict[str, Optional[PeakPick]]] = {}

    for index, run_id in enumerate(run_ids, start=1):

        simsar_tif = simsar_dir / simsar_pattern.format(id=run_id)

        print(
            f"  [{index:>4}/{len(run_ids)}] {run_id}: ",
            end="",
            flush=True,
        )

        if not simsar_tif.exists():
            print("missing sim_sar")
            score_rows.append({
                "run_id": run_id,
                "status": "missing_file",
                "rmse_raw_m": np.nan,
                "rmse_filtered_m": np.nan,
                "rmse_combined_m": np.nan,
            })
            continue

        try:
            sim_profiles = read_simsar_profiles(
                simsar_tif,
                expected_shape=mli_shape,
            )

            sim_picks = pick_profile_set(
                sim_profiles,
                peak_sigma=peak_sigma,
                prominence_db=peak_prominence_db,
                min_distance_pixels=peak_distance_pixels,
                peak_mode=peak_mode,
            )

            score = score_model(
                run_id,
                sim_picks,
                raw_mli_picks,
                filtered_mli_picks,
            )

            score_rows.append(score)

            if score["status"] == "ok":
                sim_profiles_by_id[run_id] = sim_profiles
                sim_picks_by_id[run_id] = sim_picks

                print(
                    "combined RMSE "
                    f"{float(score['rmse_combined_m']):.2f} m"
                )
            else:
                print(score["status"])

        except Exception as exc:
            print(f"failed: {exc}")
            score_rows.append({
                "run_id": run_id,
                "status": f"error: {exc}",
                "rmse_raw_m": np.nan,
                "rmse_filtered_m": np.nan,
                "rmse_combined_m": np.nan,
            })

    ranking = pd.DataFrame(score_rows)

    # Ensure numeric score columns exist even if all models fail.
    for col in ("rmse_raw_m", "rmse_filtered_m", "rmse_combined_m"):
        if col not in ranking:
            ranking[col] = np.nan

    valid_mask = (
        (ranking["status"] == "ok")
        & np.isfinite(ranking["rmse_combined_m"])
    )

    if not np.any(valid_mask):
        failed_csv = output_dir / "peak_model_ranking.csv"
        ranking.to_csv(failed_csv, index=False)
        raise RuntimeError(
            "No model produced valid A/B/C peaks. "
            f"Partial results were written to {failed_csv}."
        )

    # Ranks are computed only over valid models.
    ranking["rank_raw"] = np.nan
    ranking["rank_filtered"] = np.nan
    ranking["rank_combined"] = np.nan

    ranking.loc[valid_mask, "rank_raw"] = (
        ranking.loc[valid_mask, "rmse_raw_m"]
        .rank(method="min", ascending=True)
    )

    ranking.loc[valid_mask, "rank_filtered"] = (
        ranking.loc[valid_mask, "rmse_filtered_m"]
        .rank(method="min", ascending=True)
    )

    ranking.loc[valid_mask, "rank_combined"] = (
        ranking.loc[valid_mask, "rmse_combined_m"]
        .rank(method="min", ascending=True)
    )

    metric_map = {
        "filtered": ("rmse_filtered_m", "Filtered MLI"),
        "raw": ("rmse_raw_m", "Raw MLI"),
        "combined": ("rmse_combined_m", "Combined raw+filtered"),
    }

    ranking_metric, ranking_label = metric_map[rank_by]

    selected_rank_map = {
        "filtered": "rank_filtered",
        "raw": "rank_raw",
        "combined": "rank_combined",
    }

    ranking["rank_selected"] = ranking[selected_rank_map[rank_by]]

    secondary = [
        col
        for col in ("rmse_filtered_m", "rmse_raw_m", "rmse_combined_m")
        if col != ranking_metric
    ]

    ranking = ranking.sort_values(
        by=[ranking_metric] + secondary,
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)

    ranking_csv = output_dir / "peak_model_ranking.csv"
    ranking.to_csv(ranking_csv, index=False)

    valid_ranking = ranking[
        (ranking["status"] == "ok")
        & np.isfinite(ranking[ranking_metric])
    ].copy()

    top_models = valid_ranking.head(max(1, int(top_n))).copy()

    # -------------------------------------------------------------------------
    # Top-model plot
    # -------------------------------------------------------------------------

    print("\n[3/4] Plotting top models")

    top_png = output_dir / "top5_peak_profile_comparison.png"

    plot_top_models(
        top_models,
        sim_profiles_by_id,
        raw_mli_profiles,
        filtered_mli_profiles,
        raw_mli_picks,
        filtered_mli_picks,
        sim_picks_by_id,
        top_png,
        mli_name=mli_tif.name,
        ranking_metric=ranking_metric,
        ranking_label=ranking_label,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("\n[4/4] Best-fitting models")

    show_cols = [
        "rank_selected",
        "run_id",
        ranking_metric,
        "rmse_raw_m",
        "rmse_filtered_m",
        "rmse_combined_m",
    ]

    print(
        top_models[show_cols]
        .to_string(index=False)
    )

    best = top_models.iloc[0]

    print(f"\nBEST FITTING MODEL — ranked by {ranking_label}")
    print(f"  P.{best['run_id']}.dem")
    print(f"  selected peak RMSE: {best[ranking_metric]:.3f} m")
    print(f"  raw MLI peak RMSE:  {best['rmse_raw_m']:.3f} m")
    print(f"  filt MLI peak RMSE: {best['rmse_filtered_m']:.3f} m")
    print(f"  combined RMSE:      {best['rmse_combined_m']:.3f} m")

    print("\nOutputs:")
    print(f"  {observed_csv}")
    print(f"  {observed_png}")
    print(f"  {ranking_csv}")
    print(f"  {top_png}")

    return {
        "best_run_id": str(best["run_id"]),
        "rank_by": rank_by,
        "best_selected_rmse_m": float(best[ranking_metric]),
        "best_combined_rmse_m": float(best["rmse_combined_m"]),
        "ranking": ranking,
        "observed_peaks": observed_df,
        "ranking_csv": ranking_csv,
        "top5_plot": top_png,
    }


# =============================================================================
# COMMAND LINE
# =============================================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Rank modified DEM models by A/B/C post-shadow peak positions "
            "in radar-coordinate simulated SAR versus one observed MLI."
        )
    )

    parser.add_argument(
        "mli_tif",
        type=Path,
        help="Observed MLI GeoTIFF in the same radar grid as sim_sar outputs.",
    )

    parser.add_argument(
        "simsar_dir",
        type=Path,
        help="Directory containing P.{ID}.sim_sar.radar.tif files.",
    )

    parser.add_argument(
        "id_start",
        help="First modified-DEM ID, inclusive (e.g. 1 or 0001).",
    )

    parser.add_argument(
        "id_end",
        help="Last modified-DEM ID, inclusive (e.g. 100 or 0100).",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("peak_model_inversion"),
        help="Output directory. Default: ./peak_model_inversion",
    )

    parser.add_argument(
        "--simsar-pattern",
        default="P.{id}.sim_sar.radar.tif",
        help=(
            "Filename pattern inside simsar_dir. It must contain {id}. "
            "Default: P.{id}.sim_sar.radar.tif"
        ),
    )

    parser.add_argument(
        "--median-size",
        type=int,
        default=15,
        help=(
            "Odd 2-D median-filter size for the filtered MLI comparison. "
            "Filtering is done in linear intensity. Default: 15"
        ),
    )

    parser.add_argument(
        "--peak-sigma",
        type=float,
        default=1.5,
        help=(
            "1-D Gaussian sigma in pixels used only by the peak detector. "
            "Set 0 to disable. Default: 1.5"
        ),
    )

    parser.add_argument(
        "--peak-prominence-db",
        type=float,
        default=2.0,
        help="Minimum post-shadow peak prominence in dB. Default: 2.0",
    )

    parser.add_argument(
        "--peak-distance-pixels",
        type=int,
        default=3,
        help="Minimum separation between candidate peaks. Default: 3 pixels",
    )

    parser.add_argument(
        "--peak-mode",
        choices=("first", "most_prominent"),
        default="first",
        help=(
            "Choose the first significant post-shadow peak or the most "
            "prominent one. Default: first"
        ),
    )

    parser.add_argument(
        "--rank-by",
        choices=("filtered", "raw", "combined"),
        default="filtered",
        help=(
            "Primary model-ranking metric. 'filtered' is recommended because "
            "raw and filtered MLI are two treatments of the same observation, "
            "not independent observations. Default: filtered"
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of best models to plot. Default: 5",
    )

    args = parser.parse_args()

    if "{id}" not in args.simsar_pattern:
        parser.error("--simsar-pattern must contain '{id}'.")

    run_ids = build_run_ids(
        args.id_start,
        args.id_end,
    )

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
        top_n=args.top_n,
        rank_by=args.rank_by,
    )


if __name__ == "__main__":
    main()
