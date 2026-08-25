#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def read_single_band_tif(path):
    path = Path(path)

    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

        metadata = {
            "shape": data.shape,
            "transform": src.transform,
            "crs": src.crs,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
        }

    return data, metadata


def log_intensity(data):
    """
    Convert positive intensity values to log10 space.

    Non-positive and non-finite values become NaN.
    """
    out = np.full(data.shape, np.nan, dtype=np.float32)

    valid = np.isfinite(data) & (data > 0)

    out[valid] = 10.0 * np.log10(data[valid])

    return out


def robust_standardize(data):
    """
    Robust normalization for visual comparison.

    Uses median and MAD so the comparison is less sensitive
    to very bright SAR pixels.
    """
    valid = np.isfinite(data)

    values = data[valid]

    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad == 0:
        raise ValueError("MAD is zero; cannot standardize raster.")

    scale = 1.4826 * mad

    result = np.full(data.shape, np.nan, dtype=np.float32)

    result[valid] = (data[valid] - median) / scale

    return result
def make_intensity_histogram(
    mli,
    simsar,
    output_png,
    *,
    median_size=9,
    bins=150,
    title=None,
):
    """
    Compare observed and simulated SAR intensity distributions.

    Processing
    ----------
    1. Remove invalid / non-positive pixels.
    2. Median-filter the observed MLI in linear intensity space.
    3. Convert all data to dB using 10*log10(intensity).
    4. Shift sim_sar in dB so its median matches the median-filtered MLI.
    5. Plot normalized histograms.
    """

    output_png = Path(output_png)

    # ------------------------------------------------------------
    # Valid data
    # ------------------------------------------------------------

    mli_linear = np.asarray(mli, dtype=np.float32)
    sim_linear = np.asarray(simsar, dtype=np.float32)

    mli_valid = (
        np.isfinite(mli_linear)
        & (mli_linear > 0)
    )

    sim_valid = (
        np.isfinite(sim_linear)
        & (sim_linear > 0)
    )

    # ------------------------------------------------------------
    # Median-filter MLI
    #
    # Fill invalid pixels with the global median temporarily so
    # extreme nodata values do not enter the filter.
    # ------------------------------------------------------------

    mli_work = mli_linear.copy()

    global_median = np.median(
        mli_work[mli_valid]
    )

    mli_work[~mli_valid] = global_median

    mli_filtered = median_filter(
        mli_work,
        size=median_size,
        mode="nearest",
    )

    # Restore invalid source pixels to NaN
    mli_filtered[~mli_valid] = np.nan

    # ------------------------------------------------------------
    # Log intensity
    # ------------------------------------------------------------

    mli_db = log_intensity(mli_linear)
    mli_filtered_db = log_intensity(mli_filtered)
    sim_db = log_intensity(sim_linear)

    # ------------------------------------------------------------
    # Use pixels valid in all relevant datasets
    # ------------------------------------------------------------

    common_valid = (
        np.isfinite(mli_db)
        & np.isfinite(mli_filtered_db)
        & np.isfinite(sim_db)
    )

    if not np.any(common_valid):
        raise ValueError(
            "No common valid pixels for histogram comparison."
        )

    mli_values = mli_db[common_valid]
    filtered_values = mli_filtered_db[common_valid]
    sim_values = sim_db[common_valid]

    # ------------------------------------------------------------
    # Median alignment
    # ------------------------------------------------------------

    filtered_median = float(
        np.median(filtered_values)
    )

    sim_median = float(
        np.median(sim_values)
    )

    median_shift_db = (
        filtered_median - sim_median
    )

    sim_scaled_values = (
        sim_values + median_shift_db
    )

    print("\n[CHECK] MLI median filtering")
    print(
        f"        Median filter: {median_size} x {median_size} pixels"
    )
    print(
        "        Purpose: suppress small-scale SAR speckle/outliers "
        "at approximately DEM-scale resolution."
    )

    print("\n[CHECK] Median intensity alignment")
    print(
        "        Median filtered MLI:",
        filtered_median,
        "dB",
    )
    print(
        "        Median original sim_sar:",
        sim_median,
        "dB",
    )
    print(
        "        sim_sar shift:",
        median_shift_db,
        "dB",
    )
    print(
        "        Median scaled sim_sar:",
        float(np.median(sim_scaled_values)),
        "dB",
    )

    # ------------------------------------------------------------
    # Common histogram limits
    #
    # Use robust percentiles so isolated extreme values do not
    # determine the plot extent.
    # ------------------------------------------------------------

    all_values = np.concatenate([
        mli_values,
        filtered_values,
        sim_values,
        sim_scaled_values,
    ])

    hist_min = float(
        np.nanpercentile(all_values, 0.5)
    )

    hist_max = float(
        np.nanpercentile(all_values, 99.5)
    )

    bin_edges = np.linspace(
        hist_min,
        hist_max,
        bins + 1,
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        mli_values,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=1.2,
        label="Original MLI",
    )

    ax.hist(
        filtered_values,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=2.0,
        label=f"MLI median filtered ({median_size}×{median_size})",
    )

    ax.hist(
        sim_values,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=1.2,
        label="Original sim_sar",
    )

    ax.hist(
        sim_scaled_values,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="Median-aligned sim_sar",
    )

    # Median markers
    ax.axvline(
        filtered_median,
        linestyle="--",
        linewidth=1,
        label="Filtered MLI / scaled sim_sar median",
    )

    ax.axvline(
        sim_median,
        linestyle=":",
        linewidth=1,
        label="Original sim_sar median",
    )

    ax.set_xlabel("Log intensity (dB)")
    ax.set_ylabel("Probability density")

    if title is None:
        title = "Observed and simulated SAR intensity distributions"

    ax.set_title(title)
    ax.legend()

    ax.grid(alpha=0.2)

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_png,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n[PASS] Intensity histogram written:")
    print("      ", output_png)

    return {
        "median_filter_size": median_size,
        "mli_filtered_median_db": filtered_median,
        "simsar_original_median_db": sim_median,
        "simsar_shift_db": median_shift_db,
        "simsar_scaled_median_db": float(
            np.median(sim_scaled_values)
        ),
    }

def compare_simsar_mli(
    simsar_tif,
    mli_tif,
    output_png,
    *,
    title=None,
):
    simsar_tif = Path(simsar_tif)
    mli_tif = Path(mli_tif)
    output_png = Path(output_png)

    simsar, sim_meta = read_single_band_tif(simsar_tif)
    mli, mli_meta = read_single_band_tif(mli_tif)


    histogram_png = output_png.with_name(
        output_png.stem + "_histogram.png"
    )

    hist_stats = make_intensity_histogram(
        mli=mli,
        simsar=simsar,
        output_png=histogram_png,
        median_size=9,
        bins=150,
        title="MLI vs simulated SAR intensity distributions",
    )
    print("\n[CHECK] Input raster geometry")
    print("        sim_sar shape:", simsar.shape)
    print("        MLI shape:    ", mli.shape)

    if simsar.shape != mli.shape:
        raise ValueError(
            "sim_sar and MLI shapes differ. "
            "No resampling has been performed because radar geometry "
            "should already match."
        )

    print("\n[CHECK] Pixel dimensions agree.")
    print(
        "        We deliberately do not resample either image before "
        "comparison."
    )

    # ------------------------------------------------------------
    # Convert both to log intensity
    # ------------------------------------------------------------

    sim_log = log_intensity(simsar)
    mli_log = log_intensity(mli)

    common_valid = (
        np.isfinite(sim_log)
        & np.isfinite(mli_log)
    )

    n_common = int(np.count_nonzero(common_valid))

    print("\n[CHECK] Common finite positive pixels:", n_common)

    if n_common == 0:
        raise ValueError(
            "No common positive finite pixels between sim_sar and MLI."
        )

    # ------------------------------------------------------------
    # Robust normalization
    #
    # This compares spatial structure rather than absolute amplitude.
    # ------------------------------------------------------------

    sim_norm = robust_standardize(sim_log)
    mli_norm = robust_standardize(mli_log)

    difference = sim_norm - mli_norm

    difference[~common_valid] = np.nan

    # ------------------------------------------------------------
    # Simple diagnostic statistics
    # ------------------------------------------------------------

    sim_values = sim_log[common_valid]
    mli_values = mli_log[common_valid]

    correlation = np.corrcoef(
        sim_values,
        mli_values,
    )[0, 1]

    print("\n[CHECK] Comparison statistics")
    print(
        "        Pearson correlation in log intensity:",
        float(correlation),
    )

    print(
        "        Median observed MLI (dB):",
        float(np.median(mli_values)),
    )

    print(
        "        Median simulated SAR (dB):",
        float(np.median(sim_values)),
    )

    # ------------------------------------------------------------
    # Common display ranges
    # ------------------------------------------------------------

    combined = np.concatenate([
        mli_log[common_valid],
        sim_log[common_valid],
    ])

    vmin = np.nanpercentile(combined, 2)
    vmax = np.nanpercentile(combined, 98)

    diff_valid = difference[np.isfinite(difference)]

    diff_limit = np.nanpercentile(
        np.abs(diff_valid),
        98,
    )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6),
        constrained_layout=True,
    )

    im0 = axes[0].pcolormesh(
        mli_log,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title("Observed MLI")
    axes[0].set_xlabel("Range pixel")
    axes[0].set_ylabel("Azimuth line")

    fig.colorbar(
        im0,
        ax=axes[0],
        label="Intensity (dB)",
    )

    im1 = axes[1].pcolormesh(
        sim_log,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_title("Simulated SAR")
    axes[1].set_xlabel("Range pixel")
    axes[1].set_ylabel("Azimuth line")

    fig.colorbar(
        im1,
        ax=axes[1],
        label="Intensity (dB)",
    )

    im2 = axes[2].pcolormesh(
        difference,
        shading="auto",
        vmin=-diff_limit,
        vmax=diff_limit,
        cmap="RdBu_r",
    )

    axes[2].set_title("Normalized difference")
    axes[2].set_xlabel("Range pixel")
    axes[2].set_ylabel("Azimuth line")

    fig.colorbar(
        im2,
        ax=axes[2],
        label="Robust normalized difference",
    )

    # Radar rasters normally have azimuth line 0 at the top.
    for ax in axes:
        ax.invert_yaxis()

    if title is not None:
        fig.suptitle(title)

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_png,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\n[PASS] Comparison plot written:")
    print("      ", output_png)


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare a GAMMA simulated SAR GeoTIFF "
            "with an observed MLI GeoTIFF."
        )
    )

    parser.add_argument(
        "simsar_tif",
        help="P.{ID}.sim_sar.radar.tif",
    )

    parser.add_argument(
        "mli_tif",
        help="Observed MLI GeoTIFF",
    )

    parser.add_argument(
        "output_png",
        help="Output comparison PNG",
    )

    parser.add_argument(
        "--title",
        default=None,
    )

    args = parser.parse_args()

    compare_simsar_mli(
        args.simsar_tif,
        args.mli_tif,
        args.output_png,
        title=args.title,
    )


if __name__ == "__main__":
    main()