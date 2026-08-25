import numpy as np
import xarray as xr
from toposhapes_sar.shapes import RotatedEllipsoid, RotatedCuboid
from toposhapes_sar.modify import apply_shape


def flat_dem(size=401, spacing=10.0, z=0.0):
    half=(size//2)*spacing
    x=np.linspace(-half,half,size); y=np.linspace(half,-half,size)
    return xr.DataArray(np.full((size,size),z,dtype=np.float32),dims=("y","x"),coords={"y":y,"x":x},name="elevation")


def test_add_sphere_only_raises_surface():
    print("\n[CHECK] 'fill_to_upper' can only raise terrain; it never creates negative changes.")
    dem=flat_dem(); sphere=RotatedEllipsoid(center=(0,0,-50),semi_axes=(100,100,100))
    modified,dz=apply_shape(dem,sphere,interaction="fill_to_upper")
    assert np.nanmin(dz.values)>=0
    assert float(modified.sel(x=0,y=0))==50.0


def test_deep_sphere_excavates_vertical_shaft_to_lower_hemisphere():
    print("\n[CHECK] Deep subtraction semantics: a buried sphere can still define an excavation.")
    print("        Flat DEM z=0 m; sphere centre z=-1000 m; radius=100 m.")
    print("        'excavate_to_lower' lowers every footprint pixel to the sphere's lower surface,")
    print("        producing vertical shaft walls and a hemispherical lower floor.")
    dem=flat_dem(); sphere=RotatedEllipsoid(center=(0,0,-1000),semi_axes=(100,100,100))
    modified,dz=apply_shape(dem,sphere,interaction="excavate_to_lower")
    centre=float(modified.sel(x=0,y=0))
    edge=float(modified.sel(x=100,y=0))
    outside=float(modified.sel(x=110,y=0))
    print(f"        centre={centre:.1f} m, footprint edge={edge:.1f} m, outside={outside:.1f} m")
    assert centre==-1100.0
    assert edge==-1000.0
    assert outside==0.0
    assert np.nanmax(dz.values)<=0


def test_subtract_cuboid():
    print("\n[CHECK] Cuboids support the same explicit excavation semantics.")
    dem=flat_dem(); box=RotatedCuboid(center=(0,0,-20),size=(100,80,40))
    modified,dz=apply_shape(dem,box,interaction="excavate_to_lower")
    assert float(modified.sel(x=0,y=0))==-40.0
    assert np.nanmax(dz.values)<=0
