"""Transparent end-to-end checks for a real P.dem/P.dem_par.

Run with:
  python examples/check_real_dem.py /path/P.dem /path/P.dem_par
"""
import sys
import numpy as np
from toposhapes_sar import read_gamma_dem, project_dem_nearest

if len(sys.argv)!=3:
    raise SystemExit("usage: check_real_dem.py P.dem P.dem_par")

dem_path,par_path=sys.argv[1:]
print("\n[CHECK 1] Load the original GAMMA binary without changing its grid.")
dem_geo,meta=read_gamma_dem(dem_path,par_path,byte_order="big")
print("  shape:",dem_geo.shape,"dtype for processing:",dem_geo.dtype,"CRS:",dem_geo.rio.crs)
print("  source binary convention: REAL*4 big-endian")
print("  transform:",repr(dem_geo.rio.transform()))

print("\n[CHECK 2] Reproject to metres with nearest-neighbour, not interpolation.")
dem_m=project_dem_nearest(dem_geo)
finite=dem_m.values[np.isfinite(dem_m.values)]
print("  projected CRS:",dem_m.rio.crs,"shape:",dem_m.shape,"resolution:",dem_m.rio.resolution())
print("  finite elevation range:",float(finite.min()),float(finite.max()))
print("  Purpose: shapes are specified in physical metre units while original geographic DEM remains immutable.")
