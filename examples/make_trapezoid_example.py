#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from toposhapes_sar import (
    RotatedTrapezoidalPrism,
    apply_shape,
    project_dem_nearest,
    read_gamma_dem,
    save_qa_plots,
    transfer_displacement_to_original_grid,
    write_gamma_dem,
    write_run_json,
)
from toposhapes_sar.grid import (
    validate_original_grid,
)


# -------------------------------------------------------------------------
# Inputs
# -------------------------------------------------------------------------

DATA = Path("data")

DEM = DATA / "P.dem"
PAR = DATA / "P.dem_par"

RUN_ID = "trapezoid_test"

OUT = DATA / "synthetic"
OUT.mkdir(
    parents=True,
    exist_ok=True,
)

QA_PADDING_M = 500.0


# -------------------------------------------------------------------------
# Read original DEM
# -------------------------------------------------------------------------

print(
    "\n[STEP 1] Reading authoritative original DEM."
)

dem_geo_original, meta = (
    read_gamma_dem(
        DEM,
        PAR,
    )
)


# -------------------------------------------------------------------------
# Project to metres
# -------------------------------------------------------------------------

print(
    "\n[STEP 2] Projecting DEM to metre coordinates."
)

dem_m_original = (
    project_dem_nearest(
        dem_geo_original
    )
)


# -------------------------------------------------------------------------
# Explicit shape position
# -------------------------------------------------------------------------

shape_x = 432450+250.0
shape_y = 350500+500.0

surface_z = float(
    dem_m_original.sel(
        x=shape_x,
        y=shape_y,
        method="nearest",
    )
)

# Fixed absolute vertical placement
shape_z = (
    surface_z - 20.0
)


# -------------------------------------------------------------------------
# Define trapezoidal prism
# -------------------------------------------------------------------------

shape = RotatedTrapezoidalPrism(
    center=(
        shape_x,
        shape_y,
        shape_z,
    ),

    # Wider at the bottom than at the top
    bottom_length=70.0,
    top_length=150.0,

    width=500.0,
    height=80.0,

    rotation_deg=(
        90.0,   # yaw
        0.0,    # pitch
        45.0,    # roll
    ),
)

interaction = (
    "excavate_to_lower"
)

print(
    "\n[STEP 3] Applying trapezoidal prism."
)

print(
    f"         centre={shape.center}"
)

print(
    "         bottom length="
    f"{shape.bottom_length} m"
)

print(
    "         top length="
    f"{shape.top_length} m"
)

print(
    f"         width={shape.width} m"
)

print(
    f"         height={shape.height} m"
)

print(
    f"         interaction={interaction}"
)


# -------------------------------------------------------------------------
# Apply shape
# -------------------------------------------------------------------------

dem_m_modified, dz_m = (
    apply_shape(
        dem_m_original,
        shape,
        interaction=interaction,
    )
)

changed_pixels = int(
    (dz_m != 0).sum()
)

print(
    f"         changed pixels="
    f"{changed_pixels}"
)


# -------------------------------------------------------------------------
# Volume metadata
# -------------------------------------------------------------------------

dx, dy = (
    dem_m_original.rio.resolution()
)

pixel_area_m2 = abs(
    dx * dy
)

dz_values = dz_m.values

added_volume_m3 = max(
    0.0,
    float(
        np.sum(
            dz_values[
                np.isfinite(dz_values)
                & (dz_values > 0)
            ]
        )
        * pixel_area_m2
    ),
)

removed_volume_m3 = max(
    0.0,
    float(
        -np.sum(
            dz_values[
                np.isfinite(dz_values)
                & (dz_values < 0)
            ]
        )
        * pixel_area_m2
    ),
)

net_volume_change_m3 = float(
    np.nansum(dz_values)
    * pixel_area_m2
)

shape_volume_m3 = (
    shape.volume_m3()
)

print(
    "\n         Volume diagnostics:"
)

print(
    "             geometric prism volume: "
    f"{shape_volume_m3:.1f} m³"
)

print(
    "             material added: "
    f"{added_volume_m3:.1f} m³"
)

print(
    "             material removed: "
    f"{removed_volume_m3:.1f} m³"
)

print(
    "             net DEM volume change: "
    f"{net_volume_change_m3:+.1f} m³"
)


# -------------------------------------------------------------------------
# QA plots
# -------------------------------------------------------------------------

if changed_pixels > 0:

    qa_outputs = (
        save_qa_plots(
            dem_m_modified,
            dz_m,
            OUT,
            RUN_ID,
            padding_m=QA_PADDING_M,
        )
    )

else:

    qa_outputs = {
        "modified_dem_plot": None,
        "difference_plot": None,
    }


# -------------------------------------------------------------------------
# Return intentional dz to original grid
# -------------------------------------------------------------------------

dem_geo_modified, dz_geo = (
    transfer_displacement_to_original_grid(
        dem_geo_original,
        dz_m,
    )
)


# -------------------------------------------------------------------------
# Validate exact original-grid behavior
# -------------------------------------------------------------------------

stats = validate_original_grid(
    dem_geo_original,
    dem_geo_modified,
)

print(
    "\n[STEP 4] Validation:"
)

for key, value in stats.items():

    print(
        f"         {key}: {value}"
    )

assert stats["shape_exact"]
assert stats["x_exact"]
assert stats["y_exact"]
assert stats["transform_exact"]
assert stats["crs_exact"]
assert stats["nan_pixels"] == 0
assert stats["unchanged_pixels_exact"]


# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------

dem_out = (
    OUT
    / f"P.{RUN_ID}.dem"
)

json_out = (
    OUT
    / f"{RUN_ID}.json"
)

write_gamma_dem(
    dem_geo_modified,
    dem_out,
)


# -------------------------------------------------------------------------
# Provenance
# -------------------------------------------------------------------------

record = shape.to_dict()

record[
    "interaction"
] = interaction

record[
    "volume"
] = {
    "geometric_shape_volume_m3":
        float(shape_volume_m3),

    "added_material_m3":
        float(added_volume_m3),

    "removed_material_m3":
        float(removed_volume_m3),

    "net_volume_change_m3":
        float(net_volume_change_m3),

    "pixel_area_m2":
        float(pixel_area_m2),

    "method":
        "sum(dz * projected_pixel_area)",
}


write_run_json(
    json_out,

    run_id=RUN_ID,

    source_dem=DEM,

    source_dem_par=PAR,

    output_dem=dem_out,

    dem_geo_original=
        dem_geo_original,

    shape_records=[
        record
    ],

    validation=stats,

    projected_crs=
        dem_m_original.rio.crs,

    qa_outputs=
        qa_outputs,
)


print(
    "\n[DONE]"
)

print(
    f"       DEM: {dem_out}"
)

print(
    f"       JSON: {json_out}"
)