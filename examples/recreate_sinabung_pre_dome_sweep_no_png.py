#!/usr/bin/env python3

from __future__ import annotations

from itertools import product
from pathlib import Path
import re

import numpy as np

from toposhapes_sar import (
    RotatedEllipsoid,
    apply_shape,
    project_dem_nearest,
    read_gamma_dem,
    transfer_displacement_to_original_grid,
    write_gamma_dem,
    write_run_json,
)
from toposhapes_sar.grid import validate_original_grid


def get_next_run_id(output_dir: Path) -> int:
    """Return one more than the largest existing P.{ID}.dem run ID."""
    output_dir = Path(output_dir)
    pattern = re.compile(r"^P\.(\d+)\.dem$")
    existing_ids = []

    for path in output_dir.glob("P.*.dem"):
        match = pattern.match(path.name)
        if match is not None:
            existing_ids.append(int(match.group(1)))

    if not existing_ids:
        return 1
    return max(existing_ids) + 1


# =============================================================================
# USER SETTINGS
# =============================================================================

DATA = Path("mod_dem_crater")

DEM = DATA / "P.dem"
PAR = DATA / "P.dem_par"

OUT = DATA / "synthetic_sweep"
OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Geometry sweep
# -----------------------------------------------------------------------------


X_VALUES = [
    432480.0,
    432500.0,
    432520.0,
    432540.0,


]

Y_VALUES = [
    350440.0,

    350420.0,

    350400.0,

    350380,

    350360,


    
]

Z_VALUES = [
    2340,
    2320,
    2300,
    2280,
  
]
# Semi-axes (a, b, c), metres
SEMI_AXES_VALUES = []

for a in np.arange(60, 160, 20):
    for b in np.arange(100, 160, 20):
        for c in np.arange(60, 160, 20):
            SEMI_AXES_VALUES.append((a, b, c))

# yaw, pitch, roll in degrees
ROTATION_VALUES = [
    (0, 0.0, 0.0),
]

# Vertical placement reference retained for provenance.
REFERENCE_X = 432450.0
REFERENCE_Y = 350500.0
DEPTH_BELOW_REFERENCE_SURFACE_M = 10.0

INTERACTION = "excavate_to_lower"


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

dem_m_original = project_dem_nearest(dem_geo_original)

print(f"        projected CRS={dem_m_original.rio.crs}")
print(f"        resolution={dem_m_original.rio.resolution()} m")


# =============================================================================
# 3. REFERENCE SURFACE HEIGHT
# =============================================================================

reference_surface_z = float(
    dem_m_original.sel(
        x=REFERENCE_X,
        y=REFERENCE_Y,
        method="nearest",
    )
)

print(f"        reference surface z = {reference_surface_z:.3f} m")
print("\n[SETUP] Fixed absolute ellipsoid z values supplied by Z_VALUES.")
print(
    f"        reference x/y = "
    f"({REFERENCE_X:.3f}, {REFERENCE_Y:.3f})"
)
print(f"        reference surface z = {reference_surface_z:.3f} m")


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
print(f"        total realizations = {len(parameter_sets)}")


# =============================================================================
# 5. RUN REALIZATIONS
# =============================================================================

start_id = get_next_run_id(OUT)
print(f"        first new run ID = {start_id:04d}")

for i, (
    shape_x,
    shape_y,
    shape_z,
    semi_axes,
    rotation_deg,
) in enumerate(parameter_sets):
    numeric_id = start_id + i
    run_id = f"{numeric_id:04d}"

    print("\n" + "=" * 72)
    print(f"[RUN {run_id}] {i + 1}/{len(parameter_sets)}")
    print("=" * 72)
    print(
        f"        centre xyz = "
        f"({shape_x:.3f}, {shape_y:.3f}, {shape_z:.3f}) m"
    )
    print(f"        semi_axes = {semi_axes} m")
    print(f"        rotation = {rotation_deg} deg")
    print(f"        interaction = {INTERACTION}")

    # -------------------------------------------------------------------------
    # Shape
    # -------------------------------------------------------------------------

    shape = RotatedEllipsoid(
        center=(shape_x, shape_y, shape_z),
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

    net_volume_change_m3 = float(
        np.sum(dz_values[valid_change]) * pixel_area_m2
    )
    added_volume_m3 = float(
        np.sum(dz_values[valid_change & (dz_values > 0)]) * pixel_area_m2
    )
    removed_volume_m3 = float(
        -np.sum(dz_values[valid_change & (dz_values < 0)]) * pixel_area_m2
    )

    a, b, c = semi_axes
    ellipsoid_volume_m3 = float((4.0 / 3.0) * np.pi * a * b * c)

    print("\n        Volume diagnostics:")
    print(f"            pixel area: {pixel_area_m2:.3f} m²")
    print(f"            full ellipsoid volume: {ellipsoid_volume_m3:.1f} m³")
    print(f"            material added: {added_volume_m3:.1f} m³")
    print(f"            material removed: {removed_volume_m3:.1f} m³")
    print(f"            net DEM volume change: {net_volume_change_m3:+.1f} m³")

    changed_m = int((dz_m != 0).sum())
    print(f"        projected changed pixels = {changed_m}")

    if changed_m == 0:
        print(
            "        WARNING: this geometry produced no "
            "surface modification."
        )

    # -------------------------------------------------------------------------
    # QA plots intentionally disabled
    # -------------------------------------------------------------------------
    # Keep the provenance fields so existing JSON consumers see the same keys,
    # but do not create any PNGs. This saves substantial I/O and storage during
    # large parameter sweeps.

    qa_outputs = {
        "modified_dem_plot": None,
        "difference_plot": None,
    }

    # -------------------------------------------------------------------------
    # Transfer ONLY dz to original geographic grid
    # -------------------------------------------------------------------------

    dem_geo_modified, dz_geo = transfer_displacement_to_original_grid(
        dem_geo_original,
        dz_m,
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
        print(f"            {key}: {value}")

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

    dem_out = OUT / f"P.{run_id}.dem"
    json_out = OUT / f"{run_id}.json"

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
        "semi_axes_m": [float(v) for v in semi_axes],
        "rotation_deg": [float(v) for v in rotation_deg],
        "reference_x_m": float(REFERENCE_X),
        "reference_y_m": float(REFERENCE_Y),
        "reference_surface_z_m": float(reference_surface_z),
        "depth_below_reference_surface_m": float(
            DEPTH_BELOW_REFERENCE_SURFACE_M
        ),
    }
    record["surface_intersection"] = {
        "changed_pixels_projected": int(changed_m),
        "modifies_surface": bool(changed_m > 0),
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
    print(f"            {dem_out}")
    print(f"            {json_out}")
    print("            QA PNGs: disabled")


print("\n[DONE] Parameter sweep complete.")
print(f"       Created {len(parameter_sets)} realizations in:")
print(f"       {OUT.resolve()}")
