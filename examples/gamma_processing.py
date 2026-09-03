#!/usr/bin/env python3
"""
Standalone GAMMA processing for one synthetic DEM.

This file is deliberately kept outside the installed Python package so that
it can run inside an older/fixed GAMMA + py_gamma environment without
installing the geo-py dependency stack.

Required Python dependencies:
- Python standard library
- py_gamma

Usage:
python gamma_processing.py \
    ./mod_dem/synthetic_sweep \
    0001 \
    ./slcs/20201226M/20201226.mli.par \
    --output-dir ./sim_sar

Persistent outputs are committed atomically: GAMMA writes to uniquely named
staging files first, and those files replace the final output paths only after
successful validation. This is especially useful when several IDs are being
processed concurrently by gamma_batch.py.
"""

from __future__ import annotations

import argparse
import os
import shutil
import uuid
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import py_gamma as pg


def _required_par_int(par_file, key, path):
    """Read a required integer key and give a useful corruption error."""
    value = par_file.get_value(key)
    if value is None:
        raise RuntimeError(
            f"Parameter file is incomplete or missing {key!r}: {path}"
        )
    try:
        return int(value)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Invalid {key!r} value {value!r} in parameter file: {path}"
        )


def _wait_for_stable_file(path, *, checks=3, delay_s=0.15):
    """Require a file to exist and have a stable, non-zero size."""
    path = Path(path)
    previous = None
    stable = 0
    for _ in range(max(checks * 3, 3)):
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == previous:
                stable += 1
                if stable >= checks:
                    return size
            else:
                stable = 0
            previous = size
        time.sleep(delay_s)
    if not path.exists():
        raise RuntimeError(f"Expected GAMMA output was not created: {path}")
    raise RuntimeError(f"GAMMA output did not reach a stable non-zero size: {path}")


def _atomic_copy(source, destination):
    """Copy source to a same-directory staging file, then atomically replace."""
    source = Path(source)
    destination = Path(destination)
    token = uuid.uuid4().hex
    staging = destination.parent / f".{destination.name}.{token}.tmp"

    try:
        shutil.copy2(source, staging)
        os.replace(staging, destination)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def run_gamma_processing(
    input_dir,
    run_id,
    mli_par_path,
    *,
    output_dir=None,
    lat_ovr=1,
    lon_ovr=1,
):
    input_dir = Path(input_dir).resolve()
    mli_par_path = Path(mli_par_path).resolve()

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    dem_path = input_dir / f"P.{run_id}.dem"
    dem_par_path = input_dir / "P.dem_par"

    required = [dem_path, dem_par_path, mli_par_path]

    print("\n[CHECK] Required GAMMA inputs")
    for path in required:
        print(f"        {path}")
        if not path.exists():
            raise FileNotFoundError(path)

    mapped_dem_par_out = output_dir / f"P.mapped.{run_id}.dem_par"
    sim_sar_tif_out = output_dir / f"P.{run_id}.sim_sar.radar.tif"

    # data2tiff writes directly to a unique staging TIFF in the output
    # directory. Keeping staging and final files on the same filesystem makes
    # os.replace() atomic.
    token = uuid.uuid4().hex
    sim_sar_tif_stage = (
        output_dir
        / f".P.{run_id}.sim_sar.radar.{token}.tmp.tif"
    )

    try:
        with TemporaryDirectory(prefix=f"gamma_{run_id}_") as tmp:
            work = Path(tmp)

            print("\n[CHECK] Temporary GAMMA workspace")
            print(f"        {work}")

            map_dem_par = work / "mapped.dem_par"
            map_dem = work / "mapped.dem"
            lookup = work / "lookup.lt"
            ls_map = '-'
            ls_map_rdc = '-'
            incidence = '-'
            resolution = '-'
            offnadir = '-'
            sim_sar = work / "sim_sar"
            u = '-'
            v ='-'
            psi = '-'
            pix = '-'

            print("\n[STEP 1] Running gc_map2")
            print(f"         DEM oversampling: lat={lat_ovr}, lon={lon_ovr}")

            pg.gc_map2(
                str(mli_par_path),
                str(dem_par_path),
                str(dem_path),
                str(map_dem_par),
                str(map_dem),
                str(lookup),
                lat_ovr,
                lon_ovr,
                str(ls_map),
                str(ls_map_rdc),
                str(incidence),
                str(resolution),
                str(offnadir),
                str(sim_sar),
                str(u),
                str(v),
                str(psi),
                str(pix),
            )

            _wait_for_stable_file(map_dem_par)
            _wait_for_stable_file(sim_sar)

            mapped_par = pg.ParFile(str(map_dem_par))
            map_width = _required_par_int(mapped_par, "width", map_dem_par)
            map_lines = _required_par_int(mapped_par, "nlines", map_dem_par)

            mli_par = pg.ParFile(str(mli_par_path))
            range_samples = _required_par_int(
                mli_par, "range_samples", mli_par_path
            )
            azimuth_lines = _required_par_int(
                mli_par, "azimuth_lines", mli_par_path
            )

            expected_sim_sar_bytes = map_width * map_lines * 4
            actual_sim_sar_bytes = sim_sar.stat().st_size
            if actual_sim_sar_bytes != expected_sim_sar_bytes:
                raise RuntimeError(
                    "sim_sar size does not match mapped DEM dimensions: "
                    f"expected {expected_sim_sar_bytes} bytes, "
                    f"got {actual_sim_sar_bytes} bytes"
                )

            sim_sar_radar = work / "sim_sar.radar"

            print("\n[STEP 2] Converting sim_sar to radar geometry")

            pg.geocode(
                str(lookup),
                str(sim_sar),
                map_width,
                str(sim_sar_radar),
                range_samples,
                azimuth_lines,
                2,
                0,
            )

            _wait_for_stable_file(sim_sar_radar)

            expected_radar_bytes = range_samples * azimuth_lines * 4
            actual_radar_bytes = sim_sar_radar.stat().st_size
            if actual_radar_bytes != expected_radar_bytes:
                raise RuntimeError(
                    "Radar-coordinate sim_sar size is unexpected: "
                    f"expected {expected_radar_bytes} bytes, "
                    f"got {actual_radar_bytes} bytes"
                )

            print("\n[STEP 3] Writing radar-coordinate GeoTIFF")

            pg.data2tiff(
                str(sim_sar_radar),
                range_samples,
                2,
                str(sim_sar_tif_stage),
            )

            if not sim_sar_tif_stage.exists():
                raise RuntimeError("data2tiff did not create output GeoTIFF")
            if sim_sar_tif_stage.stat().st_size == 0:
                raise RuntimeError("data2tiff created an empty output GeoTIFF")

            # Commit both persistent outputs only after all GAMMA processing
            # and validation has succeeded.
            _atomic_copy(map_dem_par, mapped_dem_par_out)
            os.replace(sim_sar_tif_stage, sim_sar_tif_out)

    finally:
        # Remove an incomplete staging TIFF after errors/interruption.
        try:
            sim_sar_tif_stage.unlink()
        except FileNotFoundError:
            pass

    print("\n[PASS] GAMMA processing completed")
    print(f"       {mapped_dem_par_out}")
    print(f"       {sim_sar_tif_out}")

    return {
        "mapped_dem_par": mapped_dem_par_out,
        "sim_sar_radar_tif": sim_sar_tif_out,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate radar-coordinate sim_sar for one P.{ID}.dem."
    )
    parser.add_argument("input_dir")
    parser.add_argument("run_id")
    parser.add_argument("mli_par")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lat-ovr", type=int, default=1)
    parser.add_argument("--lon-ovr", type=int, default=1)

    args = parser.parse_args()

    run_gamma_processing(
        input_dir=args.input_dir,
        run_id=args.run_id,
        mli_par_path=args.mli_par,
        output_dir=args.output_dir,
        lat_ovr=args.lat_ovr,
        lon_ovr=args.lon_ovr,
    )


if __name__ == "__main__":
    main()
