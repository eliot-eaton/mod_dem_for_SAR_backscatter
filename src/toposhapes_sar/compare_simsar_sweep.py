#!/usr/bin/env python3
"""
Batch compare simulated SAR GeoTIFFs with one observed MLI.

Runs in the normal geo-py environment after GAMMA processing is complete.

Installed command:
    toposhapes-compare-simsar-batch

Example:
toposhapes-compare-simsar-batch \
    ./sim_sar \
    ./mli_tifs/2020-2021/20201226.mli.tif \
    0001 0002 0003 0004 0005
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .compare_simsar_mli import compare_simsar_mli

def expand_run_ids(run_ids):
    """
    Expand run IDs and zero-padded ranges.

    Examples
    --------
    ["0040-0050"]
        -> ["0040", "0041", ..., "0050"]

    ["0001", "0040-0043", "0050"]
        -> ["0001", "0040", "0041", "0042", "0043", "0050"]
    """

    # Make sure we always have a list
    if isinstance(run_ids, (str, int)):
        run_ids = [run_ids]

    expanded = []

    for item in run_ids:

        item = str(item)

        if "-" in item:

            start_str, end_str = item.split("-", 1)

            start = int(start_str)
            end = int(end_str)

            if end < start:
                raise ValueError(
                    f"Invalid run-ID range '{item}': "
                    f"end must be >= start."
                )

            # Preserve zero padding.
            # max() also safely handles e.g. 0098-0102.
            pad_width = max(
                len(start_str),
                len(end_str),
            )

            expanded.extend(
                f"{run_id:0{pad_width}d}"
                for run_id in range(start, end + 1)
            )

        else:

            expanded.append(item)

    return expanded


def compare_simsar_sweep(
    simsar_dir,
    mli_tif,
    run_ids,
    *,
    output_dir=None,
    overwrite=False,
    median_size=15,
    bins=150,
):
    simsar_dir = Path(simsar_dir).resolve()
    mli_tif = Path(mli_tif).resolve()

    if output_dir is None:
        output_dir = simsar_dir
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not simsar_dir.exists():
        raise FileNotFoundError(simsar_dir)
    if not mli_tif.exists():
        raise FileNotFoundError(mli_tif)

    run_ids = expand_run_ids(run_ids)
    mli_date = mli_tif.name.split(".")[0]

    successful = []
    skipped = []
    failed = []

    print("\n[SETUP] sim_sar comparison batch")
    print(f"        sim_sar directory: {simsar_dir}")
    print(f"        MLI:               {mli_tif}")
    print(f"        MLI date:          {mli_date}")
    print(f"        output directory:  {output_dir}")
    print(f"        IDs:               {' '.join(run_ids)}")

    for index, run_id in enumerate(run_ids, start=1):
        print("\n" + "=" * 72)
        print(f"[COMPARE {run_id}] {index}/{len(run_ids)}")
        print("=" * 72)

        simsar_tif = simsar_dir / f"P.{run_id}.sim_sar.radar.tif"
        output_prefix = output_dir / f"P.{run_id}.sim_sar.{mli_date}"

        histogram_png = Path(str(output_prefix) + "_histogram.png")
        spatial_png = Path(str(output_prefix) + "_spatial.png")

        if not simsar_tif.exists():
            print(f"[FAIL] Missing sim_sar: {simsar_tif}")
            failed.append(run_id)
            continue

        if (
            histogram_png.exists()
            and spatial_png.exists()
            and not overwrite
        ):
            print("[SKIP] Comparison outputs already exist")
            skipped.append(run_id)
            successful.append(run_id)
            continue

        try:
            compare_simsar_mli(
                simsar_tif=simsar_tif,
                mli_tif=mli_tif,
                output_prefix=output_prefix,
                median_size=median_size,
                bins=bins,
            )
            successful.append(run_id)

        except Exception as exc:
            failed.append(run_id)
            print(f"[FAIL] {run_id}: {exc}")

    print("\n" + "=" * 72)
    print("[DONE] Comparison batch summary")
    print("=" * 72)
    print(f"        successful: {len(successful)}")
    print(f"        skipped:    {len(skipped)}")
    print(f"        failed:     {len(failed)}")

    if failed:
        print("        failed IDs: " + " ".join(failed))

    return {
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a list of P.{ID}.sim_sar.radar.tif files "
            "with one observed MLI GeoTIFF."
        )
    )
    parser.add_argument("simsar_dir")
    parser.add_argument("mli_tif")
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--median-size", type=int, default=15)
    parser.add_argument("--bins", type=int, default=150)

    args = parser.parse_args()

    compare_simsar_sweep(
        simsar_dir=args.simsar_dir,
        mli_tif=args.mli_tif,
        run_ids=args.ids,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        median_size=args.median_size,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
