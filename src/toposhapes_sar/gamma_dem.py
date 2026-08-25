from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import xarray as xr
from rasterio.transform import from_origin


@dataclass(frozen=True)
class GammaDEMMetadata:
    title: str
    projection: str
    data_format: str
    width: int
    nlines: int
    corner_lat: float
    corner_lon: float
    post_lat: float
    post_lon: float
    ellipsoid_name: str | None = None
    datum_name: str | None = None
    hgt_offset: float = 0.0
    scale: float = 1.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.nlines, self.width


def _parse_first_token(lines: list[str], key: str, cast=str, default=None):
    prefix = f"{key}:"
    for line in lines:
        if line.strip().startswith(prefix):
            value = line.split(":", 1)[1].strip().split()[0]
            return cast(value)
    if default is not None:
        return default
    raise ValueError(f"Required key {key!r} not found in GAMMA DEM parameter file")


def parse_gamma_dem_par(par_path: str | Path) -> GammaDEMMetadata:
    par_path = Path(par_path)
    lines = par_path.read_text(errors="replace").splitlines()
    title = next((line.split(":",1)[1].strip() for line in lines if line.strip().startswith("title:")), "")
    projection = _parse_first_token(lines, "DEM_projection")
    data_format = _parse_first_token(lines, "data_format")
    if data_format != "REAL*4":
        raise NotImplementedError(f"Only REAL*4 is currently supported; got {data_format}")
    if projection != "EQA":
        raise NotImplementedError(f"Only geographic EQA DEMs are currently supported; got {projection}")
    return GammaDEMMetadata(
        title=title,
        projection=projection,
        data_format=data_format,
        width=_parse_first_token(lines, "width", int),
        nlines=_parse_first_token(lines, "nlines", int),
        corner_lat=_parse_first_token(lines, "corner_lat", float),
        corner_lon=_parse_first_token(lines, "corner_lon", float),
        post_lat=_parse_first_token(lines, "post_lat", float),
        post_lon=_parse_first_token(lines, "post_lon", float),
        ellipsoid_name=next((line.split(":",1)[1].strip() for line in lines if line.strip().startswith("ellipsoid_name:")), None),
        datum_name=next((line.split(":",1)[1].strip() for line in lines if line.strip().startswith("datum_name:")), None),
        hgt_offset=_parse_first_token(lines, "DEM_hgt_offset", float, 0.0),
        scale=_parse_first_token(lines, "DEM_scale", float, 1.0),
    )


def read_gamma_dem(
    dem_path: str | Path,
    par_path: str | Path,
    *,
    byte_order: str = "big",
) -> tuple[xr.DataArray, GammaDEMMetadata]:
    """Read an original GAMMA EQA REAL*4 DEM as the authoritative geographic grid."""
    import rioxarray  # noqa: F401
    dem_path = Path(dem_path)
    meta = parse_gamma_dem_par(par_path)
    dtype = {"big": ">f4", "little": "<f4"}.get(byte_order)
    if dtype is None:
        raise ValueError("byte_order must be 'big' or 'little'")

    expected = meta.width * meta.nlines * 4
    actual = dem_path.stat().st_size
    if actual != expected:
        raise ValueError(f"Binary byte count mismatch: expected {expected}, found {actual}")

    values = np.fromfile(dem_path, dtype=dtype).reshape(meta.shape)
    # Native float32 is deliberate: GDAL/rasterio warping can mis-handle non-native endian arrays.
    values = values.astype(np.float32, copy=False)

    x = meta.corner_lon + np.arange(meta.width, dtype=np.float64) * meta.post_lon
    y = meta.corner_lat + np.arange(meta.nlines, dtype=np.float64) * meta.post_lat
    da = xr.DataArray(values, dims=("y", "x"), coords={"x": x, "y": y}, name="elevation")

    datum_text = f"{meta.datum_name or ''} {meta.ellipsoid_name or ''}".upper()
    if "WGS 84" not in datum_text and "WGS84" not in datum_text:
        raise NotImplementedError("Current CRS mapping only supports WGS84 EQA DEMs")
    da = da.rio.write_crs("EPSG:4326")

    west = meta.corner_lon - meta.post_lon / 2.0
    north = meta.corner_lat - meta.post_lat / 2.0
    transform = from_origin(west, north, abs(meta.post_lon), abs(meta.post_lat))
    da = da.rio.write_transform(transform)
    da.attrs.update({
        "gamma_source_dtype": dtype,
        "gamma_data_format": meta.data_format,
        "gamma_parameter_file": str(par_path),
    })
    return da, meta


def write_gamma_dem(dem_geo: xr.DataArray, out_path: str | Path, *, byte_order: str = "big") -> Path:
    """Write a validated geographic DEM as headerless GAMMA REAL*4 binary."""
    out_path = Path(out_path)
    dtype = {"big": ">f4", "little": "<f4"}.get(byte_order)
    if dtype is None:
        raise ValueError("byte_order must be 'big' or 'little'")
    if np.isnan(dem_geo.values).any():
        raise ValueError("Cannot write GAMMA DEM containing NaN values")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(dem_geo.values, dtype=dtype).tofile(out_path)
    return out_path
