#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

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