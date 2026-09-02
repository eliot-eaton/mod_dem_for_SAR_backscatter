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


def volume_diagnostics(dz_m, pixel_area_m2: float) -> dict[str, float | int]:
    """Calculate signed/add/remove volume and changed-pixel diagnostics."""
    values = dz_m.values
    valid = np.isfinite(values)

    net = float(np.sum(values[valid]) * pixel_area_m2)
    added = float(np.sum(values[valid & (values > 0)]) * pixel_area_m2)
    removed = float(-np.sum(values[valid & (values < 0)]) * pixel_area_m2)
    changed = int(np.count_nonzero(valid & (values != 0)))

    return {
        "net_volume_change_m3": net,
        "added_volume_m3": added,
        "removed_volume_m3": removed,
        "changed_pixels_projected": changed,
    }


def print_volume_diagnostics(label: str, stats: dict[str, float | int]) -> None:
    print(f"\n        {label}:")
    print(f"            changed pixels: {stats['changed_pixels_projected']}")
    print(f"            material added: {stats['added_volume_m3']:.1f} m³")
    print(f"            material removed: {stats['removed_volume_m3']:.1f} m³")
    print(f"            net volume change: {stats['net_volume_change_m3']:+.1f} m³")


# =============================================================================
# USER SETTINGS
# =============================================================================

DATA = Path("mod_dem_Dome")
DEM = DATA / "P.dem"
PAR = DATA / "P.dem_par"

# Use a separate output folder so this combined sweep cannot collide with the
# single-geometry sweep.
OUT = DATA / "synthetic_sweep_excavate_fill"
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GEOMETRY 1: EXCAVATE TO LOWER
# =============================================================================
# Every value below is a list so you can sweep any parameter independently.
# With one entry in each list, geometry 1 contributes one configuration.

EXCAVATE_X_VALUES = [
    432500.0,
]

EXCAVATE_Y_VALUES = [
    350450.0,
]

EXCAVATE_Z_VALUES = [
    2355.0,
]

# Semi-axes (a, b, c), metres.
EXCAVATE_SEMI_AXES_VALUES = [
    (100.0, 120.0, 100.0),
]

# yaw, pitch, roll, degrees.
EXCAVATE_ROTATION_VALUES = [
    (0.0, 0.0, 0.0),
]

EXCAVATE_INTERACTION = "excavate_to_lower"


# =============================================================================
# GEOMETRY 2: FILL TO UPPER
# =============================================================================
# This geometry is applied AFTER the excavation to the already modified DEM.

FILL_X_VALUES = [
    432480.0,
]

FILL_Y_VALUES = [
    350500.0,
]

FILL_Z_VALUES = [
    2320.0,
]

# Semi-axes (a, b, c), metres.
FILL_SEMI_AXES_VALUES = [
    (75.0, 75.0, 75.0),
]

# yaw, pitch, roll, degrees.
FILL_ROTATION_VALUES = [
    (0.0, 0.0, 0.0),
]

FILL_INTERACTION = "fill_to_upper"


# Reference retained for provenance / inspection.
REFERENCE_X = 432450.0
REFERENCE_Y = 350500.0


# =============================================================================
# 1. READ ORIGINAL DEM ONCE
# =============================================================================

print("\n[SETUP] Reading authoritative original P.dem.")
dem_geo_original, meta = read_gamma_dem(DEM, PAR)

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

reference_surface_z = float(
    dem_m_original.sel(
        x=REFERENCE_X,
        y=REFERENCE_Y,
        method="nearest",
    )
)
print(f"        reference surface z = {reference_surface_z:.3f} m")


dx, dy = dem_m_original.rio.resolution()
pixel_area_m2 = abs(dx * dy)


# =============================================================================
# 3. BUILD TWO-GEOMETRY PARAMETER COMBINATIONS
# =============================================================================

excavate_parameter_sets = list(
    product(
        EXCAVATE_X_VALUES,
        EXCAVATE_Y_VALUES,
        EXCAVATE_Z_VALUES,
        EXCAVATE_SEMI_AXES_VALUES,
        EXCAVATE_ROTATION_VALUES,
    )
)

fill_parameter_sets = list(
    product(
        FILL_X_VALUES,
        FILL_Y_VALUES,
        FILL_Z_VALUES,
        FILL_SEMI_AXES_VALUES,
        FILL_ROTATION_VALUES,
    )
)

# Each excavation configuration is paired with each fill configuration.
parameter_sets = list(product(excavate_parameter_sets, fill_parameter_sets))

print("\n[SETUP] Combined two-geometry sweep")
print(f"        excavation configurations = {len(excavate_parameter_sets)}")
print(f"        fill configurations       = {len(fill_parameter_sets)}")
print(f"        total realizations        = {len(parameter_sets)}")
print(f"        geometry 1 interaction    = {EXCAVATE_INTERACTION}")
print(f"        geometry 2 interaction    = {FILL_INTERACTION}")


# =============================================================================
# 4. RUN REALIZATIONS
# =============================================================================

start_id = get_next_run_id(OUT)
print(f"        first new run ID = {start_id:06d}")

for i, (excavate_params, fill_params) in enumerate(parameter_sets, start=1):
    numeric_id = start_id + i - 1
    run_id = f"{numeric_id:06d}"

    (
        excavate_x,
        excavate_y,
        excavate_z,
        excavate_semi_axes,
        excavate_rotation,
    ) = excavate_params

    (
        fill_x,
        fill_y,
        fill_z,
        fill_semi_axes,
        fill_rotation,
    ) = fill_params

    print("\n" + "=" * 78)
    print(f"[RUN {run_id}] {i}/{len(parameter_sets)}")
    print("=" * 78)

    print("\n        Geometry 1 — excavation")
    print(
        f"            centre xyz = "
        f"({excavate_x:.3f}, {excavate_y:.3f}, {excavate_z:.3f}) m"
    )
    print(f"            semi_axes = {excavate_semi_axes} m")
    print(f"            rotation = {excavate_rotation} deg")
    print(f"            interaction = {EXCAVATE_INTERACTION}")

    print("\n        Geometry 2 — fill")
    print(
        f"            centre xyz = "
        f"({fill_x:.3f}, {fill_y:.3f}, {fill_z:.3f}) m"
    )
    print(f"            semi_axes = {fill_semi_axes} m")
    print(f"            rotation = {fill_rotation} deg")
    print(f"            interaction = {FILL_INTERACTION}")

    # -------------------------------------------------------------------------
    # Geometry 1: excavate original surface to the lower ellipsoid surface
    # -------------------------------------------------------------------------

    excavate_shape = RotatedEllipsoid(
        center=(excavate_x, excavate_y, excavate_z),
        semi_axes=excavate_semi_axes,
        rotation_deg=excavate_rotation,
    )

    dem_after_excavate, dz_excavate = apply_shape(
        dem_m_original,
        excavate_shape,
        interaction=EXCAVATE_INTERACTION,
    )

    # -------------------------------------------------------------------------
    # Geometry 2: fill the excavated DEM to the upper ellipsoid surface
    # -------------------------------------------------------------------------

    fill_shape = RotatedEllipsoid(
        center=(fill_x, fill_y, fill_z),
        semi_axes=fill_semi_axes,
        rotation_deg=fill_rotation,
    )

    dem_m_final, dz_fill = apply_shape(
        dem_after_excavate,
        fill_shape,
        interaction=FILL_INTERACTION,
    )

    # IMPORTANT: dz_fill is relative to dem_after_excavate.  For transfer back
    # to the original geographic GAMMA grid we need the NET displacement from
    # the original DEM after BOTH interactions.
    dz_total = dem_m_final - dem_m_original

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    excavate_stats = volume_diagnostics(dz_excavate, pixel_area_m2)
    fill_stats = volume_diagnostics(dz_fill, pixel_area_m2)
    total_stats = volume_diagnostics(dz_total, pixel_area_m2)

    print_volume_diagnostics("Geometry 1 excavation diagnostics", excavate_stats)
    print_volume_diagnostics("Geometry 2 fill diagnostics", fill_stats)
    print_volume_diagnostics("Combined final DEM diagnostics", total_stats)

    # -------------------------------------------------------------------------
    # QA PNGs intentionally disabled
    # -------------------------------------------------------------------------

    qa_outputs = {
        "modified_dem_plot": None,
        "difference_plot": None,
    }

    # -------------------------------------------------------------------------
    # Transfer NET displacement to original geographic grid
    # -------------------------------------------------------------------------

    dem_geo_modified, dz_geo = transfer_displacement_to_original_grid(
        dem_geo_original,
        dz_total,
    )

    # -------------------------------------------------------------------------
    # Validate exact original-grid compatibility
    # -------------------------------------------------------------------------

    validation = validate_original_grid(
        dem_geo_original,
        dem_geo_modified,
    )

    print("\n        Validation:")
    for key, value in validation.items():
        print(f"            {key}: {value}")

    assert validation["shape_exact"]
    assert validation["x_exact"]
    assert validation["y_exact"]
    assert validation["transform_exact"]
    assert validation["crs_exact"]
    assert validation["nan_pixels"] == 0
    assert validation["unchanged_pixels_exact"]

    # -------------------------------------------------------------------------
    # Write final combined DEM
    # -------------------------------------------------------------------------

    dem_out = OUT / f"P.{run_id}.dem"
    json_out = OUT / f"{run_id}.json"

    write_gamma_dem(dem_geo_modified, dem_out)

    # -------------------------------------------------------------------------
    # Provenance: record BOTH geometries in application order
    # -------------------------------------------------------------------------

    excavate_record = excavate_shape.to_dict()
    excavate_record["interaction"] = EXCAVATE_INTERACTION
    excavate_record["application_order"] = 1
    excavate_record["role"] = "excavation"
    excavate_record["sweep_parameters"] = {
        "x_m": float(excavate_x),
        "y_m": float(excavate_y),
        "z_m": float(excavate_z),
        "semi_axes_m": [float(v) for v in excavate_semi_axes],
        "rotation_deg": [float(v) for v in excavate_rotation],
    }
    excavate_record["surface_intersection"] = {
        "changed_pixels_projected": int(
            excavate_stats["changed_pixels_projected"]
        ),
        "modifies_surface": bool(
            excavate_stats["changed_pixels_projected"] > 0
        ),
    }
    excavate_record["volume_diagnostics"] = excavate_stats

    fill_record = fill_shape.to_dict()
    fill_record["interaction"] = FILL_INTERACTION
    fill_record["application_order"] = 2
    fill_record["role"] = "fill"
    fill_record["sweep_parameters"] = {
        "x_m": float(fill_x),
        "y_m": float(fill_y),
        "z_m": float(fill_z),
        "semi_axes_m": [float(v) for v in fill_semi_axes],
        "rotation_deg": [float(v) for v in fill_rotation],
    }
    fill_record["surface_intersection"] = {
        "changed_pixels_projected": int(fill_stats["changed_pixels_projected"]),
        "modifies_surface": bool(fill_stats["changed_pixels_projected"] > 0),
    }
    fill_record["volume_diagnostics"] = fill_stats

    # Store common/reference information on both records for easy standalone
    # interpretation of either provenance entry.
    for record in (excavate_record, fill_record):
        record["reference"] = {
            "reference_x_m": float(REFERENCE_X),
            "reference_y_m": float(REFERENCE_Y),
            "reference_surface_z_m": float(reference_surface_z),
        }

    # Also retain the combined net-volume diagnostics in both entries so the
    # final DEM result can be reconstructed/checked from the JSON alone.
    excavate_record["combined_final_diagnostics"] = total_stats
    fill_record["combined_final_diagnostics"] = total_stats

    write_run_json(
        json_out,
        run_id=run_id,
        source_dem=DEM,
        source_dem_par=PAR,
        output_dem=dem_out,
        dem_geo_original=dem_geo_original,
        shape_records=[excavate_record, fill_record],
        validation=validation,
        projected_crs=dem_m_original.rio.crs,
        qa_outputs=qa_outputs,
    )

    print("\n        Outputs:")
    print(f"            {dem_out}")
    print(f"            {json_out}")
    print("            QA PNGs: disabled")


print("\n[DONE] Combined excavate + fill sweep complete.")
print(f"       Created {len(parameter_sets)} realizations in:")
print(f"       {OUT.resolve()}")
