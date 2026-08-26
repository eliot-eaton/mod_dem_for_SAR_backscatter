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
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import py_gamma as pg


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

    with TemporaryDirectory(prefix=f"gamma_{run_id}_") as tmp:
        work = Path(tmp)

        print("\n[CHECK] Temporary GAMMA workspace")
        print(f"        {work}")

        map_dem_par = work / "mapped.dem_par"
        map_dem = work / "mapped.dem"
        lookup = work / "lookup.lt"
        ls_map = work / "ls_map"
        ls_map_rdc = work / "ls_map_rdc"
        incidence = work / "inc"
        resolution = work / "res"
        offnadir = work / "offnadir"
        sim_sar = work / "sim_sar"
        u = work / "u"
        v = work / "v"
        psi = work / "psi"
        pix = work / "pix"

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

        if not map_dem_par.exists():
            raise RuntimeError("gc_map2 did not create mapped DEM parameter file")
        if not sim_sar.exists():
            raise RuntimeError("gc_map2 did not create sim_sar")

        shutil.copy2(map_dem_par, mapped_dem_par_out)

        mapped_par = pg.ParFile(str(map_dem_par))
        map_width = int(mapped_par.get_value("width"))
        map_lines = int(mapped_par.get_value("nlines"))

        mli_par = pg.ParFile(str(mli_par_path))
        range_samples = int(mli_par.get_value("range_samples"))
        azimuth_lines = int(mli_par.get_value("azimuth_lines"))

        expected_sim_sar_bytes = map_width * map_lines * 4
        if sim_sar.stat().st_size != expected_sim_sar_bytes:
            raise RuntimeError("sim_sar size does not match mapped DEM dimensions")

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

        if not sim_sar_radar.exists():
            raise RuntimeError("geocode did not create sim_sar.radar")

        expected_radar_bytes = range_samples * azimuth_lines * 4
        if sim_sar_radar.stat().st_size != expected_radar_bytes:
            raise RuntimeError("Radar-coordinate sim_sar size is unexpected")

        print("\n[STEP 3] Writing radar-coordinate GeoTIFF")

        pg.data2tiff(
            str(sim_sar_radar),
            range_samples,
            2,
            str(sim_sar_tif_out),
        )

        if not sim_sar_tif_out.exists():
            raise RuntimeError("data2tiff did not create output GeoTIFF")

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
