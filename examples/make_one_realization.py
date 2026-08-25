"""Create one real DEM realization, validate it, write GAMMA output + QA plots.

Edit the paths and shape parameters below for your site. The important workflow is:

P.dem (immutable geographic original)
    -> nearest-neighbour projected metre grid
    -> apply one or more explicit 3-D shape interactions
    -> save visual QA plots on the projected grid
    -> transfer ONLY dz back to the exact original geographic grid
    -> validate unchanged pixels are exactly original
    -> write P.{ID}.dem + {ID}.json
"""

from pathlib import Path

from toposhapes_sar import (
    RotatedEllipsoid,
    apply_shape,
    project_dem_nearest,
    read_gamma_dem,
    save_qa_plots,
    transfer_displacement_to_original_grid,
    write_gamma_dem,
    write_run_json,
)
from toposhapes_sar.grid import validate_original_grid


# -----------------------------------------------------------------------------
# EDIT THESE INPUTS
# -----------------------------------------------------------------------------
DATA = Path("data")
DEM = DATA / "P.dem"
PAR = DATA / "P.dem_par"
RUN_ID = "001"

OUT = DATA / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

# Visual QA: 500 m of context outside every side of the changed footprint.
QA_PADDING_M = 500.0


# -----------------------------------------------------------------------------
# 1. Read the authoritative original GAMMA DEM
# -----------------------------------------------------------------------------
print("\n[STEP 1] Reading original P.dem without modifying it.")
dem_geo_original, meta = read_gamma_dem(DEM, PAR)
print(f"         shape={dem_geo_original.shape}, CRS={dem_geo_original.rio.crs}")
print("         The original geographic grid remains authoritative for final output.")


# -----------------------------------------------------------------------------
# 2. Create projected metre-grid working DEM using nearest-neighbour only
# -----------------------------------------------------------------------------
print("\n[STEP 2] Reprojecting to a metre CRS using nearest-neighbour sampling.")
dem_m_original = project_dem_nearest(dem_geo_original)
print(f"         projected CRS={dem_m_original.rio.crs}")
print(f"         resolution={dem_m_original.rio.resolution()} metres")
print("         Metre coordinates let shape sizes, depths and positions be physical units.")


# -----------------------------------------------------------------------------
# 3. Define a fixed 3-D shape placement
# -----------------------------------------------------------------------------
# Example coordinates: EDIT for your DEM.
shape_x = 432450.0
shape_y = 350500.0

# z is calculated ONCE outside the shape constructor. Moving x/y later will not
# silently move the shape vertically.
surface = float(dem_m_original.sel(x=shape_x, y=shape_y, method="nearest"))
shape_z = surface - 10.0

shape = RotatedEllipsoid(
    center=(shape_x, shape_y, shape_z),
    semi_axes=(50.0, 50.0, 50.0),
    rotation_deg=(30.0, 0.0, 0.0),
)
interaction = "fill_to_upper"

print("\n[STEP 3] Applying synthetic geometry on the full projected DEM.")
print(f"         centre=(x={shape_x:.3f}, y={shape_y:.3f}, z={shape_z:.3f}) m")
print(f"         interaction={interaction}")

dem_m_modified, dz_m = apply_shape(
    dem_m_original,
    shape,
    interaction=interaction,
)

print(f"         projected changed pixels={int((dz_m != 0).sum())}")
print(f"         min dz={float(dz_m.min()):.3f} m, max dz={float(dz_m.max()):.3f} m")


# -----------------------------------------------------------------------------
# 4. Save visual QA while coordinates are still metres
# -----------------------------------------------------------------------------
print("\n[STEP 4] Saving cropped visual QA plots around the modified footprint.")
qa_outputs = save_qa_plots(
    dem_m_modified,
    dz_m,
    OUT,
    RUN_ID,
    padding_m=QA_PADDING_M,
)


# -----------------------------------------------------------------------------
# 5. Transfer only intentional dz back to the exact original geographic grid
# -----------------------------------------------------------------------------
print("\n[STEP 5] Mapping only intentional dz back to the original P.dem grid.")
dem_geo_modified, dz_geo = transfer_displacement_to_original_grid(
    dem_geo_original,
    dz_m,
)

print("         The full modified UTM DEM is NOT round-tripped back.")
print("         Zero-dz pixels are copied exactly from the immutable original DEM.")


# -----------------------------------------------------------------------------
# 6. Validate the geographic result before GAMMA binary output
# -----------------------------------------------------------------------------
print("\n[STEP 6] Validating exact original-grid preservation and changed pixels.")
stats = validate_original_grid(dem_geo_original, dem_geo_modified)
for key, value in stats.items():
    print(f"         {key}: {value}")

assert stats["shape_exact"]
assert stats["x_exact"]
assert stats["y_exact"]
assert stats["transform_exact"]
assert stats["crs_exact"]
assert stats["nan_pixels"] == 0
assert stats["unchanged_pixels_exact"]


# -----------------------------------------------------------------------------
# 7. Write GAMMA binary and provenance JSON
# -----------------------------------------------------------------------------
dem_out = OUT / f"P.{RUN_ID}.dem"
json_out = OUT / f"{RUN_ID}.json"

print("\n[STEP 7] Writing GAMMA-compatible REAL*4 big-endian DEM.")
write_gamma_dem(dem_geo_modified, dem_out)

record = shape.to_dict()
record["interaction"] = interaction

write_run_json(
    json_out,
    run_id=RUN_ID,
    source_dem=DEM,
    source_dem_par=PAR,
    output_dem=dem_out,
    dem_geo_original=dem_geo_original,
    shape_records=[record],
    validation=stats,
    projected_crs=dem_m_original.rio.crs,
    qa_outputs=qa_outputs,
)

print("\n[DONE] Realization complete:")
print(f"       DEM:        {dem_out}")
print(f"       provenance: {json_out}")
print(f"       DEM QA:     {qa_outputs['modified_dem_plot']}")
print(f"       dz QA:      {qa_outputs['difference_plot']}")
