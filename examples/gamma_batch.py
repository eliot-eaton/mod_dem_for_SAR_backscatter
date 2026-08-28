#!/usr/bin/env python3
"""
Standalone batch GAMMA runner.

Runs in the fixed GAMMA/py_gamma environment and intentionally does not
import the installed toposhapes_sar package.

Usage:
python gamma_batch.py \
    ./mod_dem/synthetic_sweep \
    ./sim_sar \
    ./slcs/20201226M/20201226.mli.par \
    0001 0002 0003 0004 0005

Existing P.{ID}.sim_sar.radar.tif outputs are skipped unless --overwrite
is supplied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gamma_processing import run_gamma_processing


from pathlib import Path

from pathlib import Path

def run_gamma_batch(
    input_dir,
    output_dir,
    mli_par,
    run_ids,
    *,
    overwrite=False,
    lat_ovr=1,
    lon_ovr=1,
):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    mli_par = Path(mli_par).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not (input_dir / "P.dem_par").exists():
        raise FileNotFoundError(input_dir / "P.dem_par")
    if not mli_par.exists():
        raise FileNotFoundError(mli_par)

    # Handle zero-padded string ranges like "0025-0073"
    if isinstance(run_ids, str) and "-" in run_ids:
        start_str, end_str = run_ids.split("-")
        
        # Capture padding width (e.g., 4 for "0025")
        pad_width = len(start_str) 
        
        start = int(start_str)
        end = int(end_str)
        
        # Generate range and apply the original padding width
        run_ids = [f"{num:0{pad_width}d}" for num in range(start, end + 1)]
        
    # Handle standard lists, tuples, or single inputs
    else:
        if isinstance(run_ids, (int, str)):
            run_ids = [run_ids]
        elif isinstance(run_ids, tuple) and len(run_ids) == 2:
            run_ids = list(range(run_ids[0], run_ids[1] + 1))
            
        # Convert all standard items to strings
        run_ids = [str(run_id) for run_id in run_ids]
    


  

    successful = []
    skipped = []
    failed = []

    print("\n[SETUP] GAMMA batch")
    print(f"        input DEM directory: {input_dir}")
    print(f"        output directory:    {output_dir}")
    print(f"        MLI parameter file:  {mli_par}")
    print(f"        IDs:                 {' '.join(run_ids)}")

    for index, run_id in enumerate(run_ids, start=1):
        print("\n" + "=" * 72)
        print(f"[GAMMA {run_id}] {index}/{len(run_ids)}")
        print("=" * 72)

        dem_path = input_dir / f"P.{run_id}.dem"
        output_tif = output_dir / f"P.{run_id}.sim_sar.radar.tif"

        if not dem_path.exists():
            print(f"[FAIL] Missing DEM: {dem_path}")
            failed.append(run_id)
            continue

        if output_tif.exists() and not overwrite:
            print(f"[SKIP] Existing output: {output_tif}")
            skipped.append(run_id)
            successful.append(run_id)
            continue

        try:
            run_gamma_processing(
                input_dir=input_dir,
                run_id=run_id,
                mli_par_path=mli_par,
                output_dir=output_dir,
                lat_ovr=lat_ovr,
                lon_ovr=lon_ovr,
            )
            successful.append(run_id)

        except Exception as exc:
            failed.append(run_id)
            print(f"[FAIL] {run_id}: {exc}")

    print("\n" + "=" * 72)
    print("[DONE] GAMMA batch summary")
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
        description="Generate sim_sar GeoTIFFs for a list of P.{ID}.dem files."
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("mli_par")
    parser.add_argument("ids", nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--lat-ovr", type=int, default=1)
    parser.add_argument("--lon-ovr", type=int, default=1)

    args = parser.parse_args()

    run_gamma_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mli_par=args.mli_par,
        run_ids=args.ids,
        overwrite=args.overwrite,
        lat_ovr=args.lat_ovr,
        lon_ovr=args.lon_ovr,
    )


if __name__ == "__main__":
    main()
