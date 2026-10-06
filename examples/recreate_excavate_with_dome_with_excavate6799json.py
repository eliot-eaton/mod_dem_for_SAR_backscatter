#!/usr/bin/env python3

"""
Recreate all existing excavation + fill models using excavation model 6799.

For each existing combined-model JSON:

    1. Read its existing fill geometry.
    2. Ignore its existing excavation geometry.
    3. Apply fixed excavation model 6799 to the original DEM.
    4. Apply the original fill geometry.
    5. Write a new DEM and JSON using the same run ID.

The existing fill parameter space is therefore reproduced exactly, while
only the excavation geometry is changed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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


# =============================================================================
# SETTINGS
# =============================================================================

DATA = Path("mod_dem_Dome")

DEM = DATA / "P.dem"
PAR = DATA / "P.dem_par"


# Existing excavation + fill models whose fills we want to preserve
SOURCE_MODELS = (
    DATA
    / "synthetic_sweep_excavate_fill"
)


# New fixed excavation
EXCAVATION_JSON = Path(
    "mod_dem_crater/synthetic_sweep/006799.json"
)


# New output directory
OUT = (
    DATA
    / "synthetic_sweep_excavate6799_existing_fill"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


EXCAVATE_INTERACTION = "excavate_to_lower"
FILL_INTERACTION = "fill_to_upper"


REFERENCE_X = 432450.0
REFERENCE_Y = 350500.0


# =============================================================================
# HELPERS
# =============================================================================

def volume_diagnostics(
    dz_m,
    pixel_area_m2,
):

    values = np.asarray(
        dz_m.values,
        dtype=float,
    )

    valid = np.isfinite(values)

    changed = (
        valid
        & (values != 0.0)
    )

    net = float(
        np.sum(values[changed])
        * pixel_area_m2
    )

    added = float(
        np.sum(
            values[
                changed
                & (values > 0)
            ]
        )
        * pixel_area_m2
    )

    removed = float(
        -np.sum(
            values[
                changed
                & (values < 0)
            ]
        )
        * pixel_area_m2
    )

    changed_pixels = int(
        np.count_nonzero(changed)
    )

    return {
        "net_volume_change_m3": net,
        "added_volume_m3": added,
        "removed_volume_m3": removed,
        "changed_pixels_projected": changed_pixels,
    }


def find_shape(
    metadata,
    interaction,
):

    for shape in metadata.get(
        "shapes",
        [],
    ):

        if (
            shape.get("interaction")
            == interaction
        ):
            return shape

    return None


def shape_geometry(
    shape,
):

    center = shape["center_xyz_m"]
    axes = shape["semi_axes_m"]

    rotation = shape.get(
        "rotation_deg",
        [0.0, 0.0, 0.0],
    )

    return {
        "x": float(center[0]),
        "y": float(center[1]),
        "z": float(center[2]),
        "semi_axes": tuple(
            float(v)
            for v in axes
        ),
        "rotation": tuple(
            float(v)
            for v in rotation
        ),
    }


# =============================================================================
# READ EXCAVATION 6799
# =============================================================================

if not EXCAVATION_JSON.exists():

    raise FileNotFoundError(
        f"Excavation JSON not found: "
        f"{EXCAVATION_JSON}"
    )


with EXCAVATION_JSON.open() as f:

    excavation_metadata = json.load(f)


excavation_shape_json = find_shape(
    excavation_metadata,
    EXCAVATE_INTERACTION,
)


if excavation_shape_json is None:

    raise RuntimeError(
        "Could not find an "
        "excavate_to_lower geometry in "
        f"{EXCAVATION_JSON}"
    )


EXCAVATION = shape_geometry(
    excavation_shape_json
)


EXCAVATION_RUN_ID = str(
    excavation_metadata.get(
        "id",
        EXCAVATION_JSON.stem,
    )
)


print()
print("=" * 78)
print("FIXED EXCAVATION")
print("=" * 78)

print(
    f"Excavation model: "
    f"{EXCAVATION_RUN_ID}"
)

print(
    "Centre: "
    f"({EXCAVATION['x']}, "
    f"{EXCAVATION['y']}, "
    f"{EXCAVATION['z']})"
)

print(
    f"Semi-axes: "
    f"{EXCAVATION['semi_axes']}"
)


# =============================================================================
# FIND EXISTING FILL JSON FILES
# =============================================================================

json_pattern = re.compile(
    r"^(\d+)\.json$"
)


source_jsons = []


for path in SOURCE_MODELS.glob(
    "*.json"
):

    match = json_pattern.match(
        path.name
    )

    if match is not None:

        source_jsons.append(
            (
                int(match.group(1)),
                path,
            )
        )


source_jsons.sort(
    key=lambda item: item[0]
)


if not source_jsons:

    raise RuntimeError(
        f"No ID.json files found in "
        f"{SOURCE_MODELS}"
    )


print()
print(
    f"Found {len(source_jsons)} "
    "existing model JSON files."
)


# =============================================================================
# READ ORIGINAL DEM
# =============================================================================

dem_geo_original, meta = read_gamma_dem(
    DEM,
    PAR,
)


dem_m_original = project_dem_nearest(
    dem_geo_original
)


reference_surface_z = float(

    dem_m_original.sel(
        x=REFERENCE_X,
        y=REFERENCE_Y,
        method="nearest",
    )
)


dx, dy = (
    dem_m_original.rio.resolution()
)

pixel_area_m2 = abs(
    dx * dy
)


# =============================================================================
# APPLY EXCAVATION 6799 ONCE
# =============================================================================

excavate_shape = RotatedEllipsoid(

    center=(
        EXCAVATION["x"],
        EXCAVATION["y"],
        EXCAVATION["z"],
    ),

    semi_axes=(
        EXCAVATION["semi_axes"]
    ),

    rotation_deg=(
        EXCAVATION["rotation"]
    ),
)


dem_after_excavate, dz_excavate = (
    apply_shape(
        dem_m_original,
        excavate_shape,
        interaction=EXCAVATE_INTERACTION,
    )
)


excavate_stats = volume_diagnostics(
    dz_excavate,
    pixel_area_m2,
)


print()
print(
    "Excavation removed volume: "
    f"{excavate_stats['removed_volume_m3']:.1f} m3"
)


# =============================================================================
# REPLAY EVERY EXISTING FILL
# =============================================================================

created = 0
skipped = 0


for numeric_id, source_json in source_jsons:

    run_id = f"{numeric_id:06d}"

    dem_out = (
        OUT
        / f"P.{run_id}.dem"
    )

    json_out = (
        OUT
        / f"{run_id}.json"
    )


    # -------------------------------------------------------------------------
    # Do not silently overwrite completed models
    # -------------------------------------------------------------------------

    if (
        dem_out.exists()
        and json_out.exists()
    ):

        print(
            f"[{run_id}] already exists "
            "- skipping"
        )

        skipped += 1
        continue


    print()
    print("=" * 78)

    print(
        f"[{run_id}] "
        f"reading fill from "
        f"{source_json}"
    )

    print("=" * 78)


    with source_json.open() as f:

        source_metadata = json.load(f)


    source_fill_shape = find_shape(
        source_metadata,
        FILL_INTERACTION,
    )


    if source_fill_shape is None:

        print(
            f"[{run_id}] no fill_to_upper "
            "geometry found - skipping"
        )

        skipped += 1
        continue


    fill_geometry = shape_geometry(
        source_fill_shape
    )


    print(
        "Fill centre: "
        f"({fill_geometry['x']}, "
        f"{fill_geometry['y']}, "
        f"{fill_geometry['z']})"
    )

    print(
        f"Fill axes: "
        f"{fill_geometry['semi_axes']}"
    )


    # =========================================================================
    # CREATE ORIGINAL FILL SHAPE
    # =========================================================================

    fill_shape = RotatedEllipsoid(

        center=(
            fill_geometry["x"],
            fill_geometry["y"],
            fill_geometry["z"],
        ),

        semi_axes=(
            fill_geometry[
                "semi_axes"
            ]
        ),

        rotation_deg=(
            fill_geometry[
                "rotation"
            ]
        ),
    )


    # =========================================================================
    # APPLY ORIGINAL FILL TO NEW EXCAVATION
    # =========================================================================

    dem_m_final, dz_fill = (
        apply_shape(
            dem_after_excavate,
            fill_shape,
            interaction=FILL_INTERACTION,
        )
    )


    # Total displacement relative to original DEM
    dz_total = (
        dem_m_final
        - dem_m_original
    )


    fill_stats = volume_diagnostics(
        dz_fill,
        pixel_area_m2,
    )


    total_stats = volume_diagnostics(
        dz_total,
        pixel_area_m2,
    )


    # =========================================================================
    # TRANSFER BACK TO ORIGINAL GAMMA GRID
    # =========================================================================

    dem_geo_modified, dz_geo = (
        transfer_displacement_to_original_grid(
            dem_geo_original,
            dz_total,
        )
    )


    validation = (
        validate_original_grid(
            dem_geo_original,
            dem_geo_modified,
        )
    )


    assert validation["shape_exact"]
    assert validation["x_exact"]
    assert validation["y_exact"]
    assert validation["transform_exact"]
    assert validation["crs_exact"]

    assert (
        validation["nan_pixels"]
        == 0
    )

    assert validation[
        "unchanged_pixels_exact"
    ]


    # =========================================================================
    # WRITE DEM
    # =========================================================================

    if (
        dem_out.exists()
        or json_out.exists()
    ):

        raise FileExistsError(
            "Partial output already exists "
            f"for run {run_id}. "
            "Refusing to overwrite."
        )


    write_gamma_dem(
        dem_geo_modified,
        dem_out,
    )


    # =========================================================================
    # EXCAVATION PROVENANCE
    # =========================================================================

    excavate_record = (
        excavate_shape.to_dict()
    )

    excavate_record[
        "interaction"
    ] = EXCAVATE_INTERACTION

    excavate_record[
        "application_order"
    ] = 1

    excavate_record[
        "role"
    ] = "excavation"

    excavate_record[
        "source_excavation_run_id"
    ] = EXCAVATION_RUN_ID

    excavate_record[
        "source_excavation_json"
    ] = str(
        EXCAVATION_JSON
    )

    excavate_record[
        "volume_diagnostics"
    ] = excavate_stats


    # =========================================================================
    # FILL PROVENANCE
    # =========================================================================
    #
    # We recreate the record from the actual geometry but also explicitly retain
    # the source JSON / source run ID so you can trace exactly where the fill
    # came from.
    # =========================================================================

    fill_record = (
        fill_shape.to_dict()
    )

    fill_record[
        "interaction"
    ] = FILL_INTERACTION

    fill_record[
        "application_order"
    ] = 2

    fill_record[
        "role"
    ] = "fill"

    fill_record[
        "source_fill_run_id"
    ] = str(
        source_metadata.get(
            "id",
            run_id,
        )
    )

    fill_record[
        "source_fill_json"
    ] = str(
        source_json
    )

    fill_record[
        "volume_diagnostics"
    ] = fill_stats


    # Keep any useful original fill sweep metadata
    if (
        "sweep_parameters"
        in source_fill_shape
    ):

        fill_record[
            "sweep_parameters"
        ] = source_fill_shape[
            "sweep_parameters"
        ]


    # =========================================================================
    # COMMON REFERENCE / DIAGNOSTICS
    # =========================================================================

    for record in (
        excavate_record,
        fill_record,
    ):

        record[
            "reference"
        ] = {

            "reference_x_m":
                float(REFERENCE_X),

            "reference_y_m":
                float(REFERENCE_Y),

            "reference_surface_z_m":
                float(reference_surface_z),
        }

        record[
            "combined_final_diagnostics"
        ] = total_stats


    qa_outputs = {
        "modified_dem_plot": None,
        "difference_plot": None,
    }


    # =========================================================================
    # WRITE JSON
    # =========================================================================

    write_run_json(

        json_out,

        run_id=run_id,

        source_dem=DEM,

        source_dem_par=PAR,

        output_dem=dem_out,

        dem_geo_original=dem_geo_original,

        shape_records=[
            excavate_record,
            fill_record,
        ],

        validation=validation,

        projected_crs=(
            dem_m_original.rio.crs
        ),

        qa_outputs=qa_outputs,
    )


    created += 1


    print(
        f"[{run_id}] written:"
    )

    print(
        f"    {dem_out}"
    )

    print(
        f"    {json_out}"
    )


# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 78)
print("FINISHED")
print("=" * 78)

print(
    f"Created: {created}"
)

print(
    f"Skipped: {skipped}"
)

print(
    f"Output directory: {OUT}"
)