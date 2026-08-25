from __future__ import annotations

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401
from rasterio.enums import Resampling


def project_dem_nearest(dem_geo_original: xr.DataArray, projected_crs=None) -> xr.DataArray:
    """Create a metre-grid working DEM with nearest-neighbour sampling only."""
    if projected_crs is None:
        projected_crs = dem_geo_original.rio.estimate_utm_crs()
    # Native endian is required for robust rasterio/GDAL warping.
    src = dem_geo_original.astype(np.float32)
    return src.rio.reproject(projected_crs, resampling=Resampling.nearest, nodata=np.nan)


def _fresh_displacement(dz_m: xr.DataArray) -> xr.DataArray:
    """Remove inherited NaN nodata metadata and represent unmodified pixels as exact 0."""
    values=np.asarray(dz_m.values,dtype=np.float32)
    values=np.where(np.isfinite(values),values,0.0).astype(np.float32)
    out=xr.DataArray(values,dims=("y","x"),coords={"y":dz_m.y,"x":dz_m.x},name="dz")
    out=out.rio.write_crs(dz_m.rio.crs)
    out=out.rio.write_transform(dz_m.rio.transform())
    out=out.rio.write_nodata(0.0)
    return out


def transfer_displacement_to_original_grid(
    dem_geo_original: xr.DataArray,
    dz_m: xr.DataArray,
) -> tuple[xr.DataArray,xr.DataArray]:
    """Transfer only intentional displacement back to the exact original grid.

    This avoids round-trip topographic artifacts: the DEM itself is never accepted
    from the UTM->geographic warp. Instead, nearest-neighbour maps dz, with zero
    meaning unchanged, then dz is added to the immutable geographic original.
    """
    src=_fresh_displacement(dz_m)
    dz_geo=src.rio.reproject_match(dem_geo_original,resampling=Resampling.nearest,nodata=0.0)
    dz_geo=dz_geo.assign_coords(x=dem_geo_original.x,y=dem_geo_original.y)
    dz_geo=dz_geo.rio.write_crs(dem_geo_original.rio.crs)
    dz_geo=dz_geo.rio.write_transform(dem_geo_original.rio.transform())
    dz_values=np.where(np.isfinite(dz_geo.values),dz_geo.values,0.0).astype(np.float32)
    dz_geo.values[:]=dz_values

    # xr.where ensures exact source values are copied at zero-displacement pixels.
    changed=dz_geo != 0
    candidate=(dem_geo_original.astype(np.float32)+dz_geo.astype(np.float32))
    modified=xr.where(changed,candidate,dem_geo_original)
    modified=modified.assign_coords(x=dem_geo_original.x,y=dem_geo_original.y)
    modified=modified.rio.write_crs(dem_geo_original.rio.crs)
    modified=modified.rio.write_transform(dem_geo_original.rio.transform())
    return modified,dz_geo


def validate_original_grid(original: xr.DataArray, modified: xr.DataArray) -> dict:
    diff=modified.astype(np.float32)-original.astype(np.float32)
    changed=np.asarray(diff.values)!=0
    values=np.asarray(diff.values)
    stats={
        "shape_exact": modified.shape==original.shape,
        "x_exact": bool(np.array_equal(modified.x.values,original.x.values)),
        "y_exact": bool(np.array_equal(modified.y.values,original.y.values)),
        "transform_exact": modified.rio.transform()==original.rio.transform(),
        "crs_exact": modified.rio.crs==original.rio.crs,
        "nan_pixels": int(np.isnan(modified.values).sum()),
        "changed_pixels": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "changed_percent": float(100*changed.mean()),
        "unchanged_pixels_exact": bool(np.array_equal(modified.values[~changed],original.values[~changed])),
    }
    if changed.any():
        stats.update(min_change_m=float(values[changed].min()), max_change_m=float(values[changed].max()), mean_change_m=float(values[changed].mean()))
    else:
        stats.update(min_change_m=0.0,max_change_m=0.0,mean_change_m=0.0)
    return stats
