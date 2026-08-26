#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():

    parser = argparse.ArgumentParser(
        description=(
            "For a list of DEM IDs: "
            "1) generate sim_sar with gamma_processing.py, "
            "2) compare each sim_sar with one observed MLI."
        )
    )

    parser.add_argument(
        "dem_dir",
        type=Path,
        help=(
            "Directory containing P.{ID}.dem and shared P.dem_par"
        ),
    )

    parser.add_argument(
        "simsar_dir",
        type=Path,
        help=(
            "Directory where GAMMA sim_sar outputs will be written"
        ),
    )

    parser.add_argument(
        "mli_par",
        type=Path,
        help=(
            "GAMMA MLI parameter file used for gc_map2"
        ),
    )

    parser.add_argument(
        "mli_tif",
        type=Path,
        help=(
            "Observed MLI GeoTIFF used for all comparisons"
        ),
    )

    parser.add_argument(
        "ids",
        nargs="+",
        help=(
            "Run IDs to process, e.g. "
            "0001 0002 0003"
        ),
    )

    parser.add_argument(
        "--gamma-script",
        type=Path,
        default=Path(
            "src/toposhapes_sar/gamma_processing.py"
        ),
        help=(
            "Path to gamma_processing.py"
        ),
    )

    parser.add_argument(
        "--compare-script",
        type=Path,
        default=Path(
            "src/toposhapes_sar/compare_simsar_mli.py"
        ),
        help=(
            "Path to compare_simsar_mli.py"
        ),
    )

    parser.add_argument(
        "--overwrite-gamma",
        action="store_true",
        help=(
            "Re-run GAMMA even if "
            "P.{ID}.sim_sar.radar.tif already exists"
        ),
    )

    parser.add_argument(
        "--overwrite-comparison",
        action="store_true",
        help=(
            "Re-run comparison even if output PNGs already exist"
        ),
    )

    args = parser.parse_args()

    dem_dir = args.dem_dir.resolve()
    simsar_dir = args.simsar_dir.resolve()
    mli_par = args.mli_par.resolve()
    mli_tif = args.mli_tif.resolve()

    gamma_script = args.gamma_script.resolve()
    compare_script = args.compare_script.resolve()

    simsar_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------

    required = [
        dem_dir / "P.dem_par",
        mli_par,
        mli_tif,
        gamma_script,
        compare_script,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    # Example:
    # 20201226.mli.tif -> 20201226
    mli_date = mli_tif.name.split(".")[0]

    print("\n[SETUP] Batch sim_sar + comparison")
    print(f"        DEM directory:      {dem_dir}")
    print(f"        sim_sar directory:  {simsar_dir}")
    print(f"        MLI par:            {mli_par}")
    print(f"        MLI tif:            {mli_tif}")
    print(f"        MLI date:           {mli_date}")
    print(f"        IDs:                {args.ids}")

    gamma_passed = []
    gamma_failed = []

    compare_passed = []
    compare_failed = []

    # ============================================================
    # STAGE 1: GENERATE SIM_SAR FOR EACH ID
    # ============================================================

    print("\n" + "=" * 72)
    print("[STAGE 1] Generate sim_sar for requested IDs")
    print("=" * 72)

    for i, run_id in enumerate(
        args.ids,
        start=1,
    ):

        print("\n" + "-" * 72)
        print(
            f"[GAMMA {run_id}] "
            f"{i}/{len(args.ids)}"
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
            gamma_failed.append(run_id)
            continue

        if (
            simsar_tif.exists()
            and not args.overwrite_gamma
        ):
            print(
                "[SKIP] sim_sar already exists:"
            )
            print(
                f"       {simsar_tif}"
            )

            gamma_passed.append(run_id)
            continue

        command = [
            sys.executable,
            str(gamma_script),
            str(dem_dir),
            str(run_id),
            str(mli_par),
            "--output-dir",
            str(simsar_dir),
        ]

        print("[RUN]")
        print(" ".join(command))

        try:

            subprocess.run(
                command,
                check=True,
            )

            if not simsar_tif.exists():

                raise RuntimeError(
                    "gamma_processing.py returned successfully "
                    "but sim_sar GeoTIFF was not found"
                )

            print(
                f"[PASS] sim_sar created: "
                f"{simsar_tif.name}"
            )

            gamma_passed.append(
                run_id
            )

        except Exception as exc:

            print(
                f"[FAIL] GAMMA processing failed "
                f"for {run_id}: {exc}"
            )

            gamma_failed.append(
                run_id
            )

            # Continue to next ID
            continue

    # ============================================================
    # STAGE 2: COMPARE SUCCESSFUL SIM_SAR FILES TO ONE MLI
    # ============================================================

    print("\n" + "=" * 72)
    print("[STAGE 2] Compare sim_sar outputs to observed MLI")
    print("=" * 72)

    for i, run_id in enumerate(
        gamma_passed,
        start=1,
    ):

        print("\n" + "-" * 72)
        print(
            f"[COMPARE {run_id}] "
            f"{i}/{len(gamma_passed)}"
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

        if (
            histogram_png.exists()
            and spatial_png.exists()
            and not args.overwrite_comparison
        ):
            print(
                "[SKIP] comparison outputs already exist"
            )

            compare_passed.append(
                run_id
            )

            continue

        command = [
            sys.executable,
            str(compare_script),
            str(simsar_tif),
            str(mli_tif),
            str(output_prefix),
        ]

        print("[RUN]")
        print(" ".join(command))

        try:

            subprocess.run(
                command,
                check=True,
            )

            print(
                f"[PASS] comparison complete for {run_id}"
            )

            compare_passed.append(
                run_id
            )

        except subprocess.CalledProcessError as exc:

            print(
                f"[FAIL] comparison failed "
                f"for {run_id}: "
                f"return code {exc.returncode}"
            )

            compare_failed.append(
                run_id
            )

            continue

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n" + "=" * 72)
    print("[DONE] Batch processing summary")
    print("=" * 72)

    print(
        f"Requested IDs:       {len(args.ids)}"
    )

    print(
        f"GAMMA successful:    {len(gamma_passed)}"
    )

    print(
        f"GAMMA failed:        {len(gamma_failed)}"
    )

    print(
        f"Compare successful:  {len(compare_passed)}"
    )

    print(
        f"Compare failed:      {len(compare_failed)}"
    )

    if gamma_failed:
        print(
            "\nGAMMA failed IDs:"
        )
        print(
            " ".join(gamma_failed)
        )

    if compare_failed:
        print(
            "\nComparison failed IDs:"
        )
        print(
            " ".join(compare_failed)
        )


if __name__ == "__main__":
    main()