#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from .compare_simsar_mli import compare_simsar_mli
from .gamma_processing import run_gamma_processing


def run_batch_simsar_compare(
    dem_dir,
    simsar_dir,
    mli_par,
    mli_tif,
    run_ids,
    *,
    overwrite_gamma=False,
    overwrite_comparison=False,
    lat_ovr=1,
    lon_ovr=1,
    median_size=15,
    bins=150,
):
    """
    Generate sim_sar for a list of synthetic DEMs and compare every
    successful result with one observed MLI.

    Expected DEM inputs
    -------------------
    P.dem_par
    P.{ID}.dem

    Persistent GAMMA outputs
    ------------------------
    P.mapped.{ID}.dem_par
    P.{ID}.sim_sar.radar.tif

    Comparison outputs
    ------------------
    P.{ID}.sim_sar.{DATE}_histogram.png
    P.{ID}.sim_sar.{DATE}_spatial.png
    """

    dem_dir = Path(dem_dir).resolve()
    simsar_dir = Path(simsar_dir).resolve()
    mli_par = Path(mli_par).resolve()
    mli_tif = Path(mli_tif).resolve()

    simsar_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Check common inputs
    # ------------------------------------------------------------

    required = [
        dem_dir / "P.dem_par",
        mli_par,
        mli_tif,
    ]

    print("\n[CHECK] Common batch inputs")

    for path in required:

        print(f"        {path}")

        if not path.exists():
            raise FileNotFoundError(path)

    run_ids = [
        str(run_id)
        for run_id in run_ids
    ]

    # e.g. 20201226.mli.tif -> 20201226
    mli_date = mli_tif.name.split(".")[0]

    print("\n[SETUP] Batch sim_sar generation + comparison")

    print(
        f"        DEM directory:     {dem_dir}"
    )

    print(
        f"        sim_sar directory: {simsar_dir}"
    )

    print(
        f"        MLI par:           {mli_par}"
    )

    print(
        f"        MLI tif:           {mli_tif}"
    )

    print(
        f"        MLI date:          {mli_date}"
    )

    print(
        f"        IDs:               {' '.join(run_ids)}"
    )

    gamma_ok = []
    gamma_failed = []

    compare_ok = []
    compare_failed = []

    # ============================================================
    # STAGE 1
    # Generate simulated SAR
    # ============================================================

    print("\n" + "=" * 72)
    print("[STAGE 1] Generate sim_sar for requested IDs")
    print("=" * 72)

    for index, run_id in enumerate(
        run_ids,
        start=1,
    ):

        print("\n" + "-" * 72)

        print(
            f"[GAMMA {run_id}] "
            f"{index}/{len(run_ids)}"
        )

        print("-" * 72)

        dem_path = (
            dem_dir
            / f"P.{run_id}.dem"
        )

        simsar_tif = (
            simsar_dir
            / f"P.{run_id}.sim_sar.radar.tif"
        )

        if not dem_path.exists():

            print(
                f"[FAIL] DEM not found: {dem_path}"
            )

            gamma_failed.append(
                run_id
            )

            continue

        # --------------------------------------------------------
        # Restart-friendly behaviour
        # --------------------------------------------------------

        if (
            simsar_tif.exists()
            and not overwrite_gamma
        ):

            print(
                "[SKIP] sim_sar already exists:"
            )

            print(
                f"       {simsar_tif}"
            )

            gamma_ok.append(
                run_id
            )

            continue

        # --------------------------------------------------------
        # Run GAMMA directly through the installed package
        # --------------------------------------------------------

        try:

            run_gamma_processing(
                input_dir=dem_dir,
                run_id=run_id,
                mli_par_path=mli_par,
                output_dir=simsar_dir,
                lat_ovr=lat_ovr,
                lon_ovr=lon_ovr,
            )

            if not simsar_tif.exists():

                raise RuntimeError(
                    "GAMMA processing returned successfully "
                    "but did not create "
                    f"{simsar_tif}"
                )

            gamma_ok.append(
                run_id
            )

            print(
                f"[PASS] sim_sar created for {run_id}"
            )

        except Exception as exc:

            gamma_failed.append(
                run_id
            )

            print(
                "[FAIL] GAMMA processing failed "
                f"for {run_id}: {exc}"
            )

            # Do not stop the entire sweep.
            continue

    # ============================================================
    # STAGE 2
    # Compare every successful sim_sar with the same MLI
    # ============================================================

    print("\n" + "=" * 72)

    print(
        "[STAGE 2] Compare successful sim_sar "
        "outputs with observed MLI"
    )

    print("=" * 72)

    for index, run_id in enumerate(
        gamma_ok,
        start=1,
    ):

        print("\n" + "-" * 72)

        print(
            f"[COMPARE {run_id}] "
            f"{index}/{len(gamma_ok)}"
        )

        print("-" * 72)

        simsar_tif = (
            simsar_dir
            / f"P.{run_id}.sim_sar.radar.tif"
        )

        output_prefix = (
            simsar_dir
            / f"P.{run_id}.sim_sar.{mli_date}"
        )

        histogram_png = Path(
            str(output_prefix)
            + "_histogram.png"
        )

        spatial_png = Path(
            str(output_prefix)
            + "_spatial.png"
        )

        # --------------------------------------------------------
        # Restart-friendly comparison
        # --------------------------------------------------------

        if (
            histogram_png.exists()
            and spatial_png.exists()
            and not overwrite_comparison
        ):

            print(
                "[SKIP] comparison outputs already exist"
            )

            compare_ok.append(
                run_id
            )

            continue

        try:

            compare_simsar_mli(
                simsar_tif=simsar_tif,
                mli_tif=mli_tif,
                output_prefix=output_prefix,
                median_size=median_size,
                bins=bins,
            )

            compare_ok.append(
                run_id
            )

            print(
                "[PASS] comparison complete "
                f"for {run_id}"
            )

        except Exception as exc:

            compare_failed.append(
                run_id
            )

            print(
                "[FAIL] comparison failed "
                f"for {run_id}: {exc}"
            )

            continue

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "=" * 72)
    print("[DONE] Batch summary")
    print("=" * 72)

    print(
        f"        requested IDs:      {len(run_ids)}"
    )

    print(
        f"        GAMMA successful:   {len(gamma_ok)}"
    )

    print(
        f"        GAMMA failed:       {len(gamma_failed)}"
    )

    print(
        f"        compare successful: {len(compare_ok)}"
    )

    print(
        f"        compare failed:     {len(compare_failed)}"
    )

    if gamma_failed:

        print(
            "        GAMMA failed IDs:   "
            + " ".join(gamma_failed)
        )

    if compare_failed:

        print(
            "        compare failed IDs: "
            + " ".join(compare_failed)
        )

    return {
        "gamma_successful": gamma_ok,
        "gamma_failed": gamma_failed,
        "compare_successful": compare_ok,
        "compare_failed": compare_failed,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate GAMMA simulated SAR for a list of "
            "P.{ID}.dem files and compare every successful "
            "result with one observed MLI."
        )
    )

    parser.add_argument(
        "dem_dir",
        help=(
            "Directory containing P.{ID}.dem files "
            "and shared P.dem_par"
        ),
    )

    parser.add_argument(
        "simsar_dir",
        help=(
            "Directory for GAMMA and comparison outputs"
        ),
    )

    parser.add_argument(
        "mli_par",
        help=(
            "GAMMA MLI parameter file used by gc_map2"
        ),
    )

    parser.add_argument(
        "mli_tif",
        help=(
            "Observed MLI GeoTIFF used for every comparison"
        ),
    )

    parser.add_argument(
        "ids",
        nargs="+",
        help=(
            "Synthetic DEM IDs, "
            "e.g. 0001 0002 0003"
        ),
    )

    parser.add_argument(
        "--overwrite-gamma",
        action="store_true",
        help=(
            "Regenerate sim_sar even if the output "
            "GeoTIFF already exists"
        ),
    )

    parser.add_argument(
        "--overwrite-comparison",
        action="store_true",
        help=(
            "Regenerate comparison plots even if "
            "they already exist"
        ),
    )

    parser.add_argument(
        "--lat-ovr",
        type=int,
        default=1,
        help="gc_map2 latitude oversampling factor",
    )

    parser.add_argument(
        "--lon-ovr",
        type=int,
        default=1,
        help="gc_map2 longitude oversampling factor",
    )

    parser.add_argument(
        "--median-size",
        type=int,
        default=15,
        help=(
            "Median-filter window used for the MLI "
            "comparison"
        ),
    )

    parser.add_argument(
        "--bins",
        type=int,
        default=150,
        help="Number of histogram bins",
    )

    args = parser.parse_args()

    run_batch_simsar_compare(
        dem_dir=args.dem_dir,
        simsar_dir=args.simsar_dir,
        mli_par=args.mli_par,
        mli_tif=args.mli_tif,
        run_ids=args.ids,
        overwrite_gamma=args.overwrite_gamma,
        overwrite_comparison=args.overwrite_comparison,
        lat_ovr=args.lat_ovr,
        lon_ovr=args.lon_ovr,
        median_size=args.median_size,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()