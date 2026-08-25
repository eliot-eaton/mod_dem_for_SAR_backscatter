#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import median_filter


def read_single_band_tif(path):
    """
    Read band 1 from a GeoTIFF.
    """
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
    Convert positive linear intensity to dB using:

        10 * log10(intensity)

    Non-positive or non-finite values become NaN.
    """
    data = np.asarray(data)

    out = np.full(
        data.shape,
        np.nan,
        dtype=np.float32,
    )

    valid = (
        np.isfinite(data)
        & (data > 0)
    )

    out[valid] = (
        10.0 * np.log10(data[valid])
    )

    return out


def prepare_filtered_mli_and_scaled_simsar(
    mli,
    simsar,
    *,
    median_size=9,
):
    """
    Prepare MLI and simulated SAR for comparison.

    Steps
    -----
    1. Median-filter MLI in linear intensity space.
    2. Convert filtered MLI to dB.
    3. Convert sim_sar to dB.
    4. Shift sim_sar in dB so its median matches the filtered MLI.

    Returns
    -------
    mli_filtered : ndarray
        Median-filtered MLI in linear intensity.

    mli_db : ndarray
        Original MLI in dB.

    mli_filtered_db : ndarray
        Median-filtered MLI in dB.

    sim_db : ndarray
        Original sim_sar in dB.

    sim_scaled_db : ndarray
        sim_sar shifted in dB so its median matches filtered MLI.

    median_shift_db : float
        Additive dB shift applied to sim_sar.
    """

    mli = np.asarray(
        mli,
        dtype=np.float32,
    )

    simsar = np.asarray(
        simsar,
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # Identify valid pixels
    # ------------------------------------------------------------

    mli_valid = (
        np.isfinite(mli)
        & (mli > 0)
    )

    sim_valid = (
        np.isfinite(simsar)
        & (simsar > 0)
    )

    if not np.any(mli_valid):
        raise ValueError(
            "MLI contains no finite positive pixels."
        )

    if not np.any(sim_valid):
        raise ValueError(
            "sim_sar contains no finite positive pixels."
        )

    # ------------------------------------------------------------
    # Median-filter MLI in linear intensity space
    #
    # Invalid pixels are temporarily filled with the global valid
    # median so they do not inject extreme values into the filter.
    # ------------------------------------------------------------

    mli_work = mli.copy()

    mli_fill_value = float(
        np.median(mli[mli_valid])
    )

    mli_work[~mli_valid] = mli_fill_value

    mli_filtered = median_filter(
        mli_work,
        size=median_size,
        mode="nearest",
    )

    # Restore invalid original pixels
    mli_filtered[~mli_valid] = np.nan

    # ------------------------------------------------------------
    # Convert to dB
    # ------------------------------------------------------------

    mli_db = log_intensity(mli)

    mli_filtered_db = log_intensity(
        mli_filtered
    )

    sim_db = log_intensity(simsar)

    # ------------------------------------------------------------
    # Compare medians only where both filtered MLI and sim_sar
    # contain valid values.
    # ------------------------------------------------------------

    common_valid = (
        np.isfinite(mli_filtered_db)
        & np.isfinite(sim_db)
    )

    if not np.any(common_valid):
        raise ValueError(
            "No common valid pixels between "
            "filtered MLI and sim_sar."
        )

    mli_filtered_median_db = float(
        np.median(
            mli_filtered_db[common_valid]
        )
    )

    sim_median_db = float(
        np.median(
            sim_db[common_valid]
        )
    )

    median_shift_db = (
        mli_filtered_median_db
        - sim_median_db
    )

    sim_scaled_db = (
        sim_db + median_shift_db
    )

    print("\n[CHECK] MLI filtering and sim_sar scaling")

    print(
        f"        Median filter: "
        f"{median_size} x {median_size} pixels"
    )

    print(
        "        Filtered MLI median:",
        f"{mli_filtered_median_db:.3f} dB",
    )

    print(
        "        Original sim_sar median:",
        f"{sim_median_db:.3f} dB",
    )

    print(
        "        sim_sar shift:",
        f"{median_shift_db:+.3f} dB",
    )

    print(
        "        Scaled sim_sar median:",
        f"{np.median(sim_scaled_db[common_valid]):.3f} dB",
    )

    return {
        "mli_filtered": mli_filtered,
        "mli_db": mli_db,
        "mli_filtered_db": mli_filtered_db,
        "sim_db": sim_db,
        "sim_scaled_db": sim_scaled_db,
        "median_shift_db": median_shift_db,
        "common_valid": common_valid,
    }


def plot_histogram_comparison(
    prepared,
    output_png,
    *,
    bins=150,
    title=None,
):
    """
    Plot log-intensity histograms of:

    - original MLI
    - median-filtered MLI
    - original sim_sar
    - median-aligned sim_sar
    """

    output_png = Path(output_png)

    mli_db = prepared["mli_db"]
    mli_filtered_db = prepared[
        "mli_filtered_db"
    ]
    sim_db = prepared["sim_db"]
    sim_scaled_db = prepared[
        "sim_scaled_db"
    ]

    common_valid = prepared[
        "common_valid"
    ]

    mli_values = mli_db[common_valid]

    filtered_values = (
        mli_filtered_db[common_valid]
    )

    sim_values = sim_db[common_valid]

    scaled_values = (
        sim_scaled_db[common_valid]
    )

    # ------------------------------------------------------------
    # Robust histogram range
    # ------------------------------------------------------------

    all_values = np.concatenate([
        mli_values,
        filtered_values,
        sim_values,
        scaled_values,
    ])

    hist_min = float(
        np.nanpercentile(
            all_values,
            0.5,
        )
    )

    hist_max = float(
        np.nanpercentile(
            all_values,
            99.5,
        )
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
        label="MLI median filtered",
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
        scaled_values,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="Median-aligned sim_sar",
    )

    filtered_median = float(
        np.median(filtered_values)
    )

    original_sim_median = float(
        np.median(sim_values)
    )

    ax.axvline(
        filtered_median,
        linestyle="--",
        linewidth=1,
        label="Filtered MLI / scaled sim_sar median",
    )

    ax.axvline(
        original_sim_median,
        linestyle=":",
        linewidth=1,
        label="Original sim_sar median",
    )

    ax.set_xlabel(
        "Log intensity (dB)"
    )

    ax.set_ylabel(
        "Probability density"
    )

    if title is None:
        title = (
            "Observed and simulated SAR "
            "intensity distributions"
        )

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

    print(
        "\n[PASS] Histogram comparison written:"
    )

    print(
        "      ",
        output_png,
    )


def plot_spatial_comparison(
    prepared,
    output_png,
    *,
    title=None,
):
    """
    Spatial comparison using:

    - 9x9 median-filtered observed MLI
    - median-aligned simulated SAR
    - scaled sim_sar - filtered MLI
    """

    output_png = Path(output_png)

    mli_filtered_db = prepared[
        "mli_filtered_db"
    ]

    sim_scaled_db = prepared[
        "sim_scaled_db"
    ]

    median_shift_db = prepared[
        "median_shift_db"
    ]

    common_valid = prepared[
        "common_valid"
    ]

    difference_db = (
        sim_scaled_db
        - mli_filtered_db
    )

    difference_db = difference_db.copy()

    difference_db[
        ~common_valid
    ] = np.nan

    # ------------------------------------------------------------
    # Shared intensity colour scale
    # ------------------------------------------------------------

    combined = np.concatenate([
        mli_filtered_db[
            common_valid
        ],
        sim_scaled_db[
            common_valid
        ],
    ])

    vmin = -30

    vmax = 5

    # ------------------------------------------------------------
    # Symmetric difference scale
    # ------------------------------------------------------------

    difference_values = (
        difference_db[
            common_valid
        ]
    )

    diff_limit = float(
        np.nanpercentile(
            np.abs(
                difference_values
            ),
            98,
        )
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

    # Filtered MLI
    im0 = axes[0].pcolormesh(
        mli_filtered_db,
        shading="auto",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title(
        "MLI — 9×9 median filtered"
    )

    axes[0].set_xlabel(
        "Range pixel"
    )

    axes[0].set_ylabel(
        "Azimuth line"
    )

    fig.colorbar(
        im0,
        ax=axes[0],
        label="Intensity (dB)",
    )

    # Median-aligned simulated SAR
    im1 = axes[1].pcolormesh(
        sim_scaled_db,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap="gray",
    )

    axes[1].set_title(
        "sim_sar — median aligned "
        f"({median_shift_db:+.2f} dB)"
    )

    axes[1].set_xlabel(
        "Range pixel"
    )

    axes[1].set_ylabel(
        "Azimuth line"
    )

    fig.colorbar(
        im1,
        ax=axes[1],
        label="Intensity (dB)",
    )

    # Difference
    im2 = axes[2].pcolormesh(
        difference_db,
        shading="auto",
        vmin=-diff_limit,
        vmax=diff_limit,
        cmap="RdBu_r"
    )

    axes[2].set_title(
        "Adjusted sim_sar − filtered MLI"
    )

    axes[2].set_xlabel(
        "Range pixel"
    )

    axes[2].set_ylabel(
        "Azimuth line"
    )

    fig.colorbar(
        im2,
        ax=axes[2],
        label="Difference (dB)",
    )

    # GAMMA radar rasters normally have azimuth line 0 at top
    for ax in axes:
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlim([750,1150])
        ax.set_ylim([1950,2100])
    if title is None:
        title = (
            "Filtered observed MLI vs "
            "median-aligned simulated SAR"
        )

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

    print(
        "\n[PASS] Spatial comparison written:"
    )

    print(
        "      ",
        output_png,
    )


def compare_simsar_mli(
    simsar_tif,
    mli_tif,
    output_prefix,
    *,
    median_size=15,
    bins=150,
):
    """
    Full comparison workflow.

    Outputs
    -------
    {prefix}_histogram.png
    {prefix}_spatial.png
    """

    simsar_tif = Path(
        simsar_tif
    )

    mli_tif = Path(
        mli_tif
    )

    output_prefix = Path(
        output_prefix
    )

    # ------------------------------------------------------------
    # Read input rasters
    # ------------------------------------------------------------

    simsar, sim_meta = (
        read_single_band_tif(
            simsar_tif
        )
    )

    mli, mli_meta = (
        read_single_band_tif(
            mli_tif
        )
    )

    print(
        "\n[CHECK] Input raster geometry"
    )

    print(
        "        sim_sar shape:",
        simsar.shape,
    )

    print(
        "        MLI shape:    ",
        mli.shape,
    )

    # ------------------------------------------------------------
    # Do NOT silently resample
    # ------------------------------------------------------------

    if simsar.shape != mli.shape:
        raise ValueError(
            "sim_sar and MLI shapes differ. "
            "No resampling has been performed because "
            "the radar-coordinate grids should already match."
        )

    print(
        "\n[PASS] Raster dimensions match."
    )

    print(
        "       No resampling has been performed."
    )

    # ------------------------------------------------------------
    # Prepare comparison products
    # ------------------------------------------------------------

    prepared = (
        prepare_filtered_mli_and_scaled_simsar(
            mli,
            simsar,
            median_size=median_size,
        )
    )

    # ------------------------------------------------------------
    # Output paths
    # ------------------------------------------------------------

    histogram_png = Path(
        str(output_prefix)
        + "_histogram.png"
    )

    spatial_png = Path(
        str(output_prefix)
        + "_spatial.png"
    )

    # ------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------

    plot_histogram_comparison(
        prepared,
        histogram_png,
        bins=bins,
    )

    # ------------------------------------------------------------
    # Spatial comparison
    # ------------------------------------------------------------

    plot_spatial_comparison(
        prepared,
        spatial_png,
    )

    print(
        "\n[DONE] Comparison complete."
    )

    print(
        "       Histogram:",
        histogram_png,
    )

    print(
        "       Spatial:  ",
        spatial_png,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare radar-coordinate simulated SAR "
            "with an observed MLI GeoTIFF."
        )
    )

    parser.add_argument(
        "simsar_tif",
        help=(
            "Radar-coordinate simulated SAR GeoTIFF, "
            "e.g. P.001.sim_sar.radar.tif"
        ),
    )

    parser.add_argument(
        "mli_tif",
        help=(
            "Observed MLI GeoTIFF"
        ),
    )

    parser.add_argument(
        "output_prefix",
        help=(
            "Output prefix. "
            "Creates *_histogram.png and *_spatial.png"
        ),
    )

    parser.add_argument(
        "--median-size",
        type=int,
        default=15,
        help=(
            "Median-filter window size in pixels. "
            "Default: 15"
        ),
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=150,
        help=(
            "Number of histogram bins. "
            "Default: 150"
        ),
    )

    args = parser.parse_args()

    compare_simsar_mli(
        simsar_tif=args.simsar_tif,
        mli_tif=args.mli_tif,
        output_prefix=args.output_prefix,
        median_size=args.median_size,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()