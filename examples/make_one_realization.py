"""Example realization; edit coordinates/geometry for your site."""
from pathlib import Path
import numpy as np
from toposhapes_sar import (
    read_gamma_dem, project_dem_nearest, RotatedEllipsoid, apply_shape,
    transfer_displacement_to_original_grid, write_gamma_dem, write_run_json,
)
from toposhapes_sar.grid import validate_original_grid

# EDIT THESE
DATA=Path("data")
DEM=DATA/"P.dem"; PAR=DATA/"P.dem_par"; RUN_ID="001"
OUT=DATA/"synthetic"; OUT.mkdir(parents=True,exist_ok=True)

dem_geo_original,meta=read_gamma_dem(DEM,PAR)
dem_m_original=project_dem_nearest(dem_geo_original)

# Fixed xyz placement: z is chosen explicitly outside the shape constructor.
shape_x=432450.0; shape_y=350500.0
surface=float(dem_m_original.sel(x=shape_x,y=shape_y,method="nearest"))
shape_z=surface-10.0
shape=RotatedEllipsoid(center=(shape_x,shape_y,shape_z),semi_axes=(50,50,50),rotation_deg=(30,0,0))

dem_m_modified,dz_m=apply_shape(dem_m_original,shape,interaction="fill_to_upper")
dem_geo_modified,dz_geo=transfer_displacement_to_original_grid(dem_geo_original,dz_m)
stats=validate_original_grid(dem_geo_original,dem_geo_modified)
print("Validation:",stats)
assert stats["unchanged_pixels_exact"] and stats["nan_pixels"]==0

dem_out=OUT/f"P.{RUN_ID}.dem"; json_out=OUT/f"{RUN_ID}.json"
write_gamma_dem(dem_geo_modified,dem_out)
record=shape.to_dict(); record["interaction"]="fill_to_upper"
write_run_json(json_out,run_id=RUN_ID,source_dem=DEM,source_dem_par=PAR,output_dem=dem_out,dem_geo_original=dem_geo_original,shape_records=[record],validation=stats,projected_crs=dem_m_original.rio.crs)
print("Wrote",dem_out,"and",json_out)
