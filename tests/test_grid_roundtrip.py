import numpy as np
import xarray as xr
import pytest
rioxarray=pytest.importorskip("rioxarray")
from rasterio.transform import from_origin
from toposhapes_sar.grid import project_dem_nearest, transfer_displacement_to_original_grid, validate_original_grid
from toposhapes_sar.shapes import RotatedEllipsoid
from toposhapes_sar.modify import apply_shape


def make_geo_dem():
    width=120; height=100; dx=0.0001; dy=-0.0001; lon=98.3; lat=3.2
    x=lon+np.arange(width)*dx; y=lat+np.arange(height)*dy
    yy,xx=np.meshgrid(np.arange(height),np.arange(width),indexing="ij")
    values=(1000+0.5*xx+0.2*yy).astype(np.float32)
    da=xr.DataArray(values,dims=("y","x"),coords={"x":x,"y":y},name="elevation")
    da=da.rio.write_crs("EPSG:4326")
    da=da.rio.write_transform(from_origin(lon-dx/2,lat-dy/2,dx,abs(dy)))
    return da


def test_only_intentional_displacement_returns_to_original_grid():
    print("\n[CHECK] Geographic output uses the original grid as authoritative.")
    print("        We warp only dz back, not the full DEM, so untouched pixels are copied exactly.")
    original=make_geo_dem(); projected=project_dem_nearest(original,"EPSG:32647")
    cx=float(projected.x.values[len(projected.x)//2]); cy=float(projected.y.values[len(projected.y)//2]); cz=float(projected.sel(x=cx,y=cy))-5
    shape=RotatedEllipsoid(center=(cx,cy,cz),semi_axes=(30,30,20))
    modified_m,dz_m=apply_shape(projected,shape,interaction="fill_to_upper")
    modified_geo,dz_geo=transfer_displacement_to_original_grid(original,dz_m)
    stats=validate_original_grid(original,modified_geo)
    print("        Validation:",stats)
    assert stats["shape_exact"] and stats["x_exact"] and stats["y_exact"]
    assert stats["transform_exact"] and stats["crs_exact"]
    assert stats["unchanged_pixels_exact"]
    assert stats["nan_pixels"]==0
    assert stats["min_change_m"]>=0
