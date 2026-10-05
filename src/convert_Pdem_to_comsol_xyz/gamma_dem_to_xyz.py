#!/usr/bin/env python3
"""
gamma_dem_to_xyz.py

Convert one modified GAMMA DEM to projected-metre XYZ.

Processing order:
    1. Read P.<ID>.dem on its original geographic GAMMA grid.
    2. Crop to a rectangular latitude/longitude box.
    3. Project the cropped DEM to a locally appropriate metre CRS.
    4. Replace residual NaNs with the chosen base elevation.
    5. Add a linear fade to the base elevation.
    6. Add an outer flat-elevation pad.
    7. Export X,Y,Z pixel centres as comma-separated text.

Usage:
    python gamma_dem_to_xyz.py input.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toposhapes_sar import project_dem_nearest, read_gamma_dem


def read_input_file(path: Path) -> dict:
    """Read parameters from a JSON input file."""
    with path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    if not isinstance(params, dict):
        raise ValueError("Top-level JSON value must be an object.")

    return params


def required(params, key):
    if key not in params:
        raise ValueError(f"Missing required input parameter: {key}")
    return params[key]


def as_int(params, key):
    return int(required(params, key))


def as_float(params, key):
    return float(required(params, key))


def optional_float(params, key, default=None):
    value = params.get(key, default)
    if value is None:
        return default
    return float(value)


def linear_fade_pad_2d(img: np.ndarray, n: int, target: float) -> np.ndarray:
    """Pad n pixels per side while linearly fading DEM edges to target."""
    img = np.asarray(img, dtype=float)

    if n == 0:
        return img.copy()
    if n < 0:
        raise ValueError("FADE_PIXELS must be >= 0.")

    padded = np.pad(img, n, mode="constant", constant_values=target)
    ramp = np.linspace(0.0, 1.0, n, endpoint=False)

    padded[:n, n:-n] = target + ramp[:, None] * (img[0] - target)
    padded[-n:, n:-n] = target + ramp[::-1, None] * (img[-1] - target)

    padded[n:-n, :n] = (
        target + ramp[None, :] * (img[:, 0][:, None] - target)
    )
    padded[n:-n, -n:] = (
        target + ramp[::-1][None, :] * (img[:, -1][:, None] - target)
    )

    yy, xx = np.meshgrid(ramp, ramp, indexing="ij")

    padded[:n, :n] = target + np.minimum(yy, xx) * (img[0, 0] - target)
    padded[:n, -n:] = np.flip(
        target + np.minimum(yy, xx) * (img[0, -1] - target), axis=1
    )
    padded[-n:, :n] = (
        target + np.minimum(yy[::-1], xx) * (img[-1, 0] - target)
    )
    padded[-n:, -n:] = np.flip(
        target + np.minimum(yy[::-1], xx) * (img[-1, -1] - target), axis=1
    )

    return padded


def extend_pixel_centres(coords, before, after):
    """Extend a regularly spaced projected pixel-centre coordinate vector."""
    coords = np.asarray(coords, dtype=float)

    if coords.ndim != 1 or coords.size < 2:
        raise ValueError("Need >=2 coordinates to determine grid spacing.")

    step = float(np.median(np.diff(coords)))

    if not np.allclose(
        np.diff(coords),
        step,
        rtol=1e-7,
        atol=max(abs(step) * 1e-7, 1e-9),
    ):
        raise ValueError("Projected coordinate vector is not regularly spaced.")

    prefix = coords[0] - step * np.arange(before, 0, -1)
    suffix = coords[-1] + step * np.arange(1, after + 1)

    return np.concatenate((prefix, coords, suffix))


def crop_geographic_rectangle(dem, lon_min, lon_max, lat_min, lat_max):
    """Crop directly on the original geographic pixel-centre grid."""
    x = np.asarray(dem.x.values, dtype=float)
    y = np.asarray(dem.y.values, dtype=float)

    xidx = np.flatnonzero((x >= lon_min) & (x <= lon_max))
    yidx = np.flatnonzero((y >= lat_min) & (y <= lat_max))

    if xidx.size < 2 or yidx.size < 2:
        raise ValueError(
            "Requested lat/lon rectangle contains fewer than two DEM "
            "pixel centres in one or both dimensions."
        )

    return dem.isel(
        x=slice(int(xidx[0]), int(xidx[-1]) + 1),
        y=slice(int(yidx[0]), int(yidx[-1]) + 1),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Crop, project, pad and export one GAMMA DEM as XYZ."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="JSON input parameter file",
    )
    args = parser.parse_args()

    params = read_input_file(args.input_file)

    data_dir = Path(required(params, "DATA"))
    run_dir = data_dir / required(params, "RUN_DIR")
    run_id = required(params, "ID")
    par_path = data_dir / required(params, "PAR")

    dem_path = run_dir / f"P.{run_id}.dem"

    output_value = params.get("OUTPUT_XYZ", f"P.{run_id}.xyz")
    xyz_path = Path(output_value)
    if not xyz_path.is_absolute():
        xyz_path = run_dir / xyz_path

    lat_min = as_float(params, "CROP_LAT_MIN")
    lat_max = as_float(params, "CROP_LAT_MAX")
    lon_min = as_float(params, "CROP_LON_MIN")
    lon_max = as_float(params, "CROP_LON_MAX")

    fade = as_int(params, "FADE_PIXELS")
    flat = as_int(params, "FLAT_PAD_PIXELS")
    decimals = int(params.get("XYZ_DECIMALS", "2"))
    requested_base = optional_float(params, "BASE_ELEVATION_M", None)

    if lat_min >= lat_max:
        raise ValueError("CROP_LAT_MIN must be less than CROP_LAT_MAX.")
    if lon_min >= lon_max:
        raise ValueError("CROP_LON_MIN must be less than CROP_LON_MAX.")
    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError("Latitude crop limits must lie between -90 and 90.")
    if not (-180 <= lon_min < lon_max <= 180):
        raise ValueError("Longitude crop limits must lie between -180 and 180.")
    if fade < 0 or flat < 0:
        raise ValueError("Padding widths must be >= 0.")

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    if not par_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {par_path}")

    print(f"\nInput file: {args.input_file}")
    print(f"Run ID:     {run_id}")

    # 1. Read geographic DEM.
    print("\n[1/6] Reading GAMMA DEM")
    dem_geo, meta = read_gamma_dem(dem_path, par_path)

    print(f"      DEM:             {dem_path}")
    print(f"      geographic CRS:  {dem_geo.rio.crs}")
    print(f"      original shape:  {dem_geo.shape}")

    # 2. Crop rectangle in lat/lon before projection.
    print("\n[2/6] Cropping geographic rectangle")
    print(f"      longitude:       {lon_min:.8f} to {lon_max:.8f}")
    print(f"      latitude:        {lat_min:.8f} to {lat_max:.8f}")

    dem_geo_crop = crop_geographic_rectangle(
        dem_geo,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
    )

    print(f"      cropped shape:   {dem_geo_crop.shape}")
    print(
        f"      actual lon:      {float(dem_geo_crop.x.min()):.8f} to "
        f"{float(dem_geo_crop.x.max()):.8f}"
    )
    print(
        f"      actual lat:      {float(dem_geo_crop.y.min()):.8f} to "
        f"{float(dem_geo_crop.y.max()):.8f}"
    )

    # 3. Project cropped DEM to metres.
    print("\n[3/6] Projecting cropped DEM to metres")
    dem_m = project_dem_nearest(dem_geo_crop)

    projected_crs = dem_m.rio.crs
    x = np.asarray(dem_m.x.values, dtype=float)
    y = np.asarray(dem_m.y.values, dtype=float)
    z = np.asarray(dem_m.values, dtype=float)

    if z.ndim != 2 or x.size < 2 or y.size < 2:
        raise ValueError("Projected crop is too small or not 2-D.")

    print(f"      projected CRS:   {projected_crs}")
    print(f"      resolution:      {dem_m.rio.resolution()} m")
    print(f"      projected shape: {z.shape}")

    # 4. Base elevation / NaNs.
    print("\n[4/6] Preparing projected DEM")
    finite = np.isfinite(z)

    if not np.any(finite):
        raise ValueError("Projected cropped DEM contains no finite elevations.")

    if requested_base is None:
        base = float(np.nanmin(z))
        base_source = "minimum finite projected elevation"
    else:
        base = requested_base
        base_source = "input file"

    nan_count = int(np.count_nonzero(~finite))
    z = z.copy()
    z[~finite] = base

    print(f"      base elevation:  {base:.3f} m ({base_source})")
    print(f"      NaNs filled:     {nan_count}")

    # 5. Fade and flat padding.
    print("\n[5/6] Adding padding")

    z_out = linear_fade_pad_2d(z, fade, base)

    if flat:
        z_out = np.pad(
            z_out,
            flat,
            mode="constant",
            constant_values=base,
        )

    total_pad = fade + flat
    x_out = extend_pixel_centres(x, total_pad, total_pad)
    y_out = extend_pixel_centres(y, total_pad, total_pad)

    if z_out.shape != (y_out.size, x_out.size):
        raise RuntimeError("Padded coordinate and DEM dimensions disagree.")

    print(f"      fade:            {fade} pixels/side")
    print(f"      flat:            {flat} pixels/side")
    print(f"      final shape:     {z_out.shape}")

    # 6. XYZ export.
    print("\n[6/6] Writing XYZ")

    rows, cols = z_out.shape
    xyz = np.column_stack(
        (
            np.tile(x_out, rows),
            np.repeat(y_out, cols),
            z_out.ravel(order="C"),
        )
    )

    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"%.{decimals}f"

    np.savetxt(
        xyz_path,
        xyz,
        delimiter=",",
        fmt=(fmt, fmt, fmt),
    )

    print(f"      points written:  {xyz.shape[0]:,}")
    print(f"      output:          {xyz_path.resolve()}")

    print("\n[DONE]")
    print(f"      projected CRS:   {projected_crs}")
    print(f"      base elevation:  {base:.3f} m")


if __name__ == "__main__":
    main()
