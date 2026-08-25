#!/usr/bin/env python3

from __future__ import annotations

from itertools import product
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
import numpy as np

# =============================================================================
# USER SETTINGS
# =============================================================================

DATA = Path("mod_dem")

DEM = DATA / "P.dem"
PAR = DATA / "P.dem_par"

OUT = DATA / "synthetic_sweep"
OUT.mkdir(parents=True, exist_ok=True)

QA_PADDING_M = 500.0


# -----------------------------------------------------------------------------
# Geometry sweep
# -----------------------------------------------------------------------------

X_VALUES = [
    432400.0,
    432450.0,
    432500.0,
]

Y_VALUES = [
    350450.0,
    350500.0,
    350550.0,
]

Z_VALUES = [
    2335,
    2325,
    2315,
    2305,
]
# Semi-axes (a, b, c), metres
SEMI_AXES_VALUES = [
    (50.0, 50.0, 50.0),
    (75.0, 50.0, 40.0),
    (100.0, 60.0, 30.0),
]

# yaw, pitch, roll in degrees
ROTATION_VALUES = [
    (0.0, 0.0, 0.0),
    (30.0, 0.0, 0.0),
    (60.0, 0.0, 0.0),
]

# Vertical placement:
#
# We calculate one absolute z reference ONCE.
# Moving x/y therefore does NOT silently change z.
REFERENCE_X = 432450.0
REFERENCE_Y = 350500.0

DEPTH_BELOW_REFERENCE_SURFACE_M = 10.0

INTERACTION = "fill_to_upper"


# =============================================================================
# 1. READ ORIGINAL DEM ONCE
# =============================================================================

print("\n[SETUP] Reading authoritative original P.dem.")

dem_geo_original, meta = read_gamma_dem(
    DEM,
    PAR,
)

print(
    f"        shape={dem_geo_original.shape}, "
    f"CRS={dem_geo_original.rio.crs}"
)


# =============================================================================
# 2. PROJECT ORIGINAL DEM ONCE
# =============================================================================

print("\n[SETUP] Creating projected metre-grid DEM.")

dem_m_original = project_dem_nearest(
    dem_geo_original
)

print(
    f"        projected CRS={dem_m_original.rio.crs}"
)

print(
    f"        resolution={dem_m_original.rio.resolution()} m"
)


# =============================================================================
# 3. FIX ABSOLUTE Z ONCE
# =============================================================================

reference_surface_z = float(
    dem_m_original.sel(
        x=REFERENCE_X,
        y=REFERENCE_Y,
        method="nearest",
    )
)
print(
    f"        reference surface z = "
    f"{reference_surface_z:.3f} m"
)

# shape_z = (
#     reference_surface_z
#     - DEPTH_BELOW_REFERENCE_SURFACE_M
# )

print("\n[SETUP] Fixed absolute ellipsoid z.")

print(
    f"        reference x/y = "
    f"({REFERENCE_X:.3f}, {REFERENCE_Y:.3f})"
)

print(
    f"        reference surface z = "
    f"{reference_surface_z:.3f} m"
)

# print(
#     f"        fixed shape centre z = "
#     f"{shape_z:.3f} m"
# )

# print(
#     "        Changing x/y during the sweep will not "
#     "change this z value."
# )


# =============================================================================
# 4. BUILD PARAMETER COMBINATIONS
# =============================================================================

parameter_sets = list(
    product(
        X_VALUES,
        Y_VALUES,
        Z_VALUES,
        SEMI_AXES_VALUES,
        ROTATION_VALUES,
    )
)

print("\n[SETUP] Parameter sweep")

print(
    f"        total realizations = "
    f"{len(parameter_sets)}"
)


# =============================================================================
# 5. RUN REALIZATIONS
# =============================================================================

for i, (
    shape_x,
    shape_y,
    shape_z,
    semi_axes,
    rotation_deg,
) in enumerate(
    parameter_sets,
    start=1,
):

    run_id = f"{i:04d}"

    print("\n" + "=" * 72)
    print(
        f"[RUN {run_id}] "
        f"{i}/{len(parameter_sets)}"
    )
    print("=" * 72)

    print(
        f"        centre xyz = "
        f"({shape_x:.3f}, "
        f"{shape_y:.3f}, "
        f"{shape_z:.3f}) m"
    )

    print(
        f"        semi_axes = "
        f"{semi_axes} m"
    )

    print(
        f"        rotation = "
        f"{rotation_deg} deg"
    )

    print(
        f"        interaction = "
        f"{INTERACTION}"
    )

    # -------------------------------------------------------------------------
    # Shape
    # -------------------------------------------------------------------------

    shape = RotatedEllipsoid(
        center=(
            shape_x,
            shape_y,
            shape_z,
        ),
        semi_axes=semi_axes,
        rotation_deg=rotation_deg,
    )

    # -------------------------------------------------------------------------
    # Apply to a fresh copy of the SAME original projected DEM
    # -------------------------------------------------------------------------

    dem_m_modified, dz_m = apply_shape(
        dem_m_original,
        shape,
        interaction=INTERACTION,
    )
    # -------------------------------------------------------------------------
    # Volume diagnostics
    # -------------------------------------------------------------------------


    dx, dy = dem_m_original.rio.resolution()
    pixel_area_m2 = abs(dx * dy)

    dz_values = dz_m.values

    valid_change = np.isfinite(dz_values)

    # Signed volume:
    # positive = material added
    # negative = material removed
    net_volume_change_m3 = float(
        np.sum(dz_values[valid_change]) * pixel_area_m2
    )

    # Report addition/removal independently as positive magnitudes
    added_volume_m3 = float(
        np.sum(dz_values[valid_change & (dz_values > 0)])
        * pixel_area_m2
    )

    removed_volume_m3 = float(
        -np.sum(dz_values[valid_change & (dz_values < 0)])
        * pixel_area_m2
    )

    # Complete mathematical ellipsoid volume
    a, b, c = semi_axes

    ellipsoid_volume_m3 = float(
        (4.0 / 3.0) * np.pi * a * b * c
    )

    print("\n        Volume diagnostics:")
    print(
        f"            pixel area: "
        f"{pixel_area_m2:.3f} m²"
    )
    print(
        f"            full ellipsoid volume: "
        f"{ellipsoid_volume_m3:.1f} m³"
    )
    print(
        f"            material added: "
        f"{added_volume_m3:.1f} m³"
    )
    print(
        f"            material removed: "
        f"{removed_volume_m3:.1f} m³"
    )
    print(
        f"            net DEM volume change: "
        f"{net_volume_change_m3:+.1f} m³"
    )
    changed_m = int(
        (dz_m != 0).sum()
    )

    print(
        f"        projected changed pixels = "
        f"{changed_m}"
    )

    if changed_m == 0:
        print(
            "        WARNING: this geometry produced no "
            "surface modification."
        )

    # -------------------------------------------------------------------------
    # QA plots
    # -------------------------------------------------------------------------

    qa_outputs = save_qa_plots(
        dem_m_modified,
        dz_m,
        OUT,
        run_id,
        padding_m=QA_PADDING_M,
    )

    # -------------------------------------------------------------------------
    # Transfer ONLY dz to original geographic grid
    # -------------------------------------------------------------------------

    dem_geo_modified, dz_geo = (
        transfer_displacement_to_original_grid(
            dem_geo_original,
            dz_m,
        )
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    stats = validate_original_grid(
        dem_geo_original,
        dem_geo_modified,
    )

    print("\n        Validation:")

    for key, value in stats.items():
        print(
            f"            {key}: {value}"
        )

    assert stats["shape_exact"]
    assert stats["x_exact"]
    assert stats["y_exact"]
    assert stats["transform_exact"]
    assert stats["crs_exact"]
    assert stats["nan_pixels"] == 0
    assert stats["unchanged_pixels_exact"]

    # -------------------------------------------------------------------------
    # Write DEM
    # -------------------------------------------------------------------------

    dem_out = (
        OUT / f"P.{run_id}.dem"
    )

    json_out = (
        OUT / f"{run_id}.json"
    )

    write_gamma_dem(
        dem_geo_modified,
        dem_out,
    )

    # -------------------------------------------------------------------------
    # Provenance record
    # -------------------------------------------------------------------------

    record = shape.to_dict()

    record["interaction"] = INTERACTION

    record["sweep_parameters"] = {
        "x_m": float(shape_x),
        "y_m": float(shape_y),
        "z_m": float(shape_z),

        "semi_axes_m": [
            float(v)
            for v in semi_axes
        ],

        "rotation_deg": [
            float(v)
            for v in rotation_deg
        ],

        "reference_x_m": float(
            REFERENCE_X
        ),

        "reference_y_m": float(
            REFERENCE_Y
        ),

        "reference_surface_z_m": float(
            reference_surface_z
        ),

        "depth_below_reference_surface_m": float(
            DEPTH_BELOW_REFERENCE_SURFACE_M
        ),
    }
    

    write_run_json(
        json_out,
        run_id=run_id,
        source_dem=DEM,
        source_dem_par=PAR,
        output_dem=dem_out,
        dem_geo_original=dem_geo_original,
        shape_records=[record],
        validation=stats,
        projected_crs=dem_m_original.rio.crs,
        qa_outputs=qa_outputs,
    )

    print("\n        Outputs:")
    print(
        f"            {dem_out}"
    )
    print(
        f"            {json_out}"
    )
    print(
        f"            "
        f"{qa_outputs['modified_dem_plot']}"
    )
    print(
        f"            "
        f"{qa_outputs['difference_plot']}"
    )


print("\n[DONE] Parameter sweep complete.")

print(
    f"       Created {len(parameter_sets)} "
    f"realizations in:"
)

print(
    f"       {OUT.resolve()}"
)