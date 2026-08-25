#!/usr/bin/env python3

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
    """
    Process one synthetic GAMMA DEM through:

        P.{ID}.dem
            -> gc_map2
            -> sim_sar
            -> radar geometry
            -> GeoTIFF

    Persistent outputs
    ------------------
    P.mapped.{ID}.dem_par
        Parameter file for the DEM segment created internally by gc_map2.

    P.{ID}.sim_sar.radar.tif
        Simulated SAR image converted into radar coordinates and written
        as GeoTIFF.

    All other GAMMA products are temporary.
    """

    input_dir = Path(input_dir).resolve()
    mli_par_path = Path(mli_par_path).resolve()

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Required inputs
    # ------------------------------------------------------------

    dem_path = input_dir / f"P.{run_id}.dem"
    dem_par_path = input_dir / "P.dem_par"

    required = [
        dem_path,
        dem_par_path,
        mli_par_path,
    ]

    print("\n[CHECK] Required GAMMA inputs")

    for path in required:
        print(f"        {path}")

        if not path.exists():
            raise FileNotFoundError(path)

    # ------------------------------------------------------------
    # Persistent outputs
    # ------------------------------------------------------------

    mapped_dem_par_out = (
        output_dir / f"P.mapped.{run_id}.dem_par"
    )

    sim_sar_tif_out = (
        output_dir / f"P.{run_id}.sim_sar.radar.tif"
    )

    # ------------------------------------------------------------
    # Everything else is temporary
    # ------------------------------------------------------------

    with TemporaryDirectory(prefix=f"gamma_{run_id}_") as tmp:

        work = Path(tmp)

        print("\n[CHECK] Temporary GAMMA workspace")
        print(f"        {work}")
        print(
            "        Intermediate gc_map2/geocode files will "
            "be deleted after processing."
        )

        # gc_map2 products
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

        # ========================================================
        # 1. gc_map2
        # ========================================================

        print("\n[STEP 1] Running gc_map2")
        print(
            "         Input DEM is the synthetic P.{ID}.dem."
        )
        print(
            "         DEM oversampling factors:"
            f" lat={lat_ovr}, lon={lon_ovr}"
        )

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
            raise RuntimeError(
                "gc_map2 did not create mapped DEM parameter file"
            )

        if not sim_sar.exists():
            raise RuntimeError(
                "gc_map2 did not create sim_sar"
            )

        print("\n[PASS] gc_map2 completed")
        print(
            f"       sim_sar size: {sim_sar.stat().st_size} bytes"
        )

        # ========================================================
        # 2. Save mapped DEM parameter file
        # ========================================================

        shutil.copy2(
            map_dem_par,
            mapped_dem_par_out,
        )

        print("\n[CHECK] Preserving mapped DEM parameter file")
        print(f"        {mapped_dem_par_out}")

        # ========================================================
        # 3. Read map and radar dimensions
        # ========================================================

        mapped_par = pg.ParFile(str(map_dem_par))

        map_width = int(
            mapped_par.get_value("width")
        )

        map_lines = int(
            mapped_par.get_value("nlines")
        )

        mli_par = pg.ParFile(str(mli_par_path))

        range_samples = int(
            mli_par.get_value("range_samples")
        )

        azimuth_lines = int(
            mli_par.get_value("azimuth_lines")
        )

        print("\n[CHECK] Raster dimensions")

        print(
            "        sim_sar/map geometry:"
            f" width={map_width}, lines={map_lines}"
        )

        print(
            "        radar geometry:"
            f" range={range_samples},"
            f" azimuth={azimuth_lines}"
        )

        # sim_sar is REAL*4
        expected_sim_sar_bytes = (
            map_width * map_lines * 4
        )

        print(
            "        expected sim_sar bytes:",
            expected_sim_sar_bytes,
        )

        print(
            "        actual sim_sar bytes:  ",
            sim_sar.stat().st_size,
        )

        if sim_sar.stat().st_size != expected_sim_sar_bytes:
            raise RuntimeError(
                "sim_sar file size does not match "
                "mapped DEM dimensions"
            )

        # ========================================================
        # 4. Map geometry -> radar geometry
        # ========================================================

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
            raise RuntimeError(
                "geocode did not create sim_sar.radar"
            )

        expected_radar_bytes = (
            range_samples
            * azimuth_lines
            * 4
        )

        print(
            "        expected radar bytes:",
            expected_radar_bytes,
        )

        print(
            "        actual radar bytes:  ",
            sim_sar_radar.stat().st_size,
        )

        if sim_sar_radar.stat().st_size != expected_radar_bytes:
            raise RuntimeError(
                "Radar-coordinate sim_sar size is unexpected"
            )

        # ========================================================
        # 5. Radar binary -> GeoTIFF
        # ========================================================

        print("\n[STEP 3] Writing radar-coordinate GeoTIFF")

        pg.data2tiff(
            str(sim_sar_radar),
            range_samples,
            2,
            str(sim_sar_tif_out),
        )

        if not sim_sar_tif_out.exists():
            raise RuntimeError(
                "data2tiff did not create output GeoTIFF"
            )

        print("\n[PASS] GAMMA processing completed")

        print("\nPersistent outputs:")
        print(f"    {mapped_dem_par_out}")
        print(f"    {sim_sar_tif_out}")

    # TemporaryDirectory is deleted here.

    print("\n[CHECK] Temporary GAMMA products removed.")

    return {
        "mapped_dem_par": mapped_dem_par_out,
        "sim_sar_radar_tif": sim_sar_tif_out,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate radar-coordinate simulated SAR from "
            "a synthetic P.{ID}.dem."
        )
    )

    parser.add_argument(
        "input_dir",
        help=(
            "Directory containing P.{ID}.dem and P.dem_par"
        ),
    )

    parser.add_argument(
        "run_id",
        help="Synthetic DEM ID, e.g. 001",
    )

    parser.add_argument(
        "mli_par",
        help="Input GAMMA MLI parameter file",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to input_dir."
        ),
    )

    parser.add_argument(
        "--lat-ovr",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--lon-ovr",
        type=int,
        default=1,
    )

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