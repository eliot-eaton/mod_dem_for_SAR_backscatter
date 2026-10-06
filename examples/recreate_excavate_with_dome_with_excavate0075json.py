#!/usr/bin/env python3

"""
Generate a targeted Sinabung excavation + fill DEM sweep.

EXCAVATION
----------
The excavation geometry is read directly from:

    mod_dem_crater/synthetic_sweep/0075.json

This excavation is fixed for every model.

FILL
----
Approximately 5000 targeted fill geometries are generated.

The search is deliberately NOT one enormous Cartesian product.

Four batches are used:

    1. core_low_z_c
       Dense exploration of lower Z and smaller C.

    2. xy_extension
       Extend farther in X and toward lower Y.

    3. large_equal_ab
       Explore larger A=B fill geometries.

    4. elongated_x
       Explore A > B, meaning the x-direction semi-axis is longer
       than the y-direction semi-axis.

Candidate count before duplicate removal:

    core_low_z_c   = 2400
    xy_extension   = 1344
    large_equal_ab =  640
    elongated_x    =  600

    TOTAL          = 4984


DUPLICATE PROTECTION
--------------------
Existing excavate+fill JSON files are read before any DEM is created.

A complete model is considered identical only when BOTH geometries match:

    excavation:
        x, y, z, A, B, C, rotation, interaction

    fill:
        x, y, z, A, B, C, rotation, interaction

Duplicates are removed:

    1. against models already present in the output directory
    2. between the new batches themselves

Run IDs are allocated only when a genuinely new DEM is written.

Both DEM and JSON filenames are checked when finding the next available ID.


DRY RUN
-------
To inspect the proposed sweep without creating DEMs:

    python generate_excavate_fill_0075_sweep.py --dry-run

To actually generate the DEMs:

    python generate_excavate_fill_0075_sweep.py
"""

from __future__ import annotations

import argparse
import json
import re

from itertools import product
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
# COMMAND-LINE ARGUMENTS
# =============================================================================

parser = argparse.ArgumentParser(
    description=(
        "Generate targeted excavation + fill DEMs "
        "using excavation model 0075."
    )
)

parser.add_argument(
    "--dry-run",
    action="store_true",
    help=(
        "Build and deduplicate the parameter sweep, print the model-space "
        "summary, then exit without creating any DEMs."
    ),
)

args = parser.parse_args()

DRY_RUN = args.dry_run


# =============================================================================
# USER SETTINGS
# =============================================================================

# Original DEM used for the combined excavation + fill models
DATA = Path("mod_dem_Dome")

DEM = DATA / "P.dem"
PAR = DATA / "P.dem_par"


# -----------------------------------------------------------------------------
# Excavation model to use
# -----------------------------------------------------------------------------

EXCAVATION_JSON = Path(
    "mod_dem_crater/synthetic_sweep/0075.json"
)


# -----------------------------------------------------------------------------
# Output directory
# -----------------------------------------------------------------------------

OUT = (
    DATA
    / "synthetic_sweep_excavate_fill"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------------------------------
# Interactions
# -----------------------------------------------------------------------------

EXCAVATE_INTERACTION = "excavate_to_lower"

FILL_INTERACTION = "fill_to_upper"


# -----------------------------------------------------------------------------
# Fill rotation
# -----------------------------------------------------------------------------

FILL_ROTATION = (
    0.0,
    0.0,
    0.0,
)


# -----------------------------------------------------------------------------
# Reference location retained for provenance
# -----------------------------------------------------------------------------

REFERENCE_X = 432450.0
REFERENCE_Y = 350500.0


# =============================================================================
# FILL MODEL SPACE
# =============================================================================
#
# The new search is divided into four targeted batches.
#
# This avoids wasting thousands of models on combinations that are currently
# less interesting.
#
# In particular:
#
#     * more effort goes into LOW Z
#     * more effort goes into SMALL C
#     * large Z is NOT combined with large C
#     * X is extended both lower and higher
#     * Y is extended lower
#     * larger A=B models are tested
#     * some A>B models are tested
#
# =============================================================================


# =============================================================================
# BATCH 1
#
# DENSE SEARCH AROUND LOWER Z AND SMALLER C
# =============================================================================

BATCH_1_NAME = "core_low_z_c"


BATCH_1_X_VALUES = [
    432475.0,
    432500.0,
    432525.0,
    432550.0,
    432575.0,
]


BATCH_1_Y_VALUES = [
    350425.0,
    350450.0,
    350475.0,
    350500.0,
    350525.0,
]


# -----------------------------------------------------------------------------
# Explicit Z/C pairs
#
# IMPORTANT:
#
# Z and C are NOT independently crossed.
#
# The current optimum occurs at the smallest tested Z and C, so much more
# effort is placed below the old boundary.
#
# Higher Z values are retained as a bridge to the previous experiment, but
# they are not combined with the largest C.
# -----------------------------------------------------------------------------

BATCH_1_Z_C_PAIRS = [

    # Strong extension below previous model space
    (2225.0, 40.0),
    (2250.0, 40.0),
    (2250.0, 50.0),

    (2275.0, 40.0),
    (2275.0, 50.0),
    (2275.0, 60.0),

    # Around / just below previous lower boundary
    (2300.0, 50.0),
    (2300.0, 60.0),
    (2300.0, 75.0),

    # Bridge to previous space
    (2325.0, 75.0),
    (2325.0, 100.0),

    # Retain one high-Z / small-C comparison
    (2375.0, 75.0),
]


# -----------------------------------------------------------------------------
# Primarily A = B
#
# Includes the old general range and extends larger.
# -----------------------------------------------------------------------------

BATCH_1_AB_VALUES = [
    (90.0, 90.0),
    (100.0, 100.0),
    (110.0, 110.0),
    (120.0, 120.0),
    (130.0, 130.0),
    (140.0, 140.0),
    (150.0, 150.0),
    (160.0, 160.0),
]


# Count:
#
# 5 X
# x 5 Y
# x 12 Z/C pairs
# x 8 A/B pairs
#
# = 2400


# =============================================================================
# BATCH 2
#
# X / Y EXTENSION
# =============================================================================

BATCH_2_NAME = "xy_extension"


# -----------------------------------------------------------------------------
# Extend both directions in X.
#
# Old sweep:
#     432500 -> 432575
#
# New values include:
#     smaller X
#     larger X
# -----------------------------------------------------------------------------

BATCH_2_X_VALUES = [
    432425.0,
    432450.0,
    432475.0,
    432500.0,
    432550.0,
    432600.0,
    432625.0,
    432650.0,
]


# -----------------------------------------------------------------------------
# Extend substantially toward lower Y.
#
# Old sweep:
#     350450 -> 350525
# -----------------------------------------------------------------------------

BATCH_2_Y_VALUES = [
    350350.0,
    350375.0,
    350400.0,
    350425.0,
    350450.0,
    350475.0,
    350500.0,
]


# -----------------------------------------------------------------------------
# Focus this positional experiment on promising low-Z / low-C geometries.
# -----------------------------------------------------------------------------

BATCH_2_Z_C_PAIRS = [
    (2225.0, 40.0),
    (2250.0, 40.0),
    (2250.0, 50.0),
    (2275.0, 50.0),
    (2300.0, 60.0),
    (2300.0, 75.0),
]


# -----------------------------------------------------------------------------
# Representative equal A/B geometries.
#
# We do not need to test every horizontal size here because Batch 2 is mainly
# asking where the fill centre should be.
# -----------------------------------------------------------------------------

BATCH_2_AB_VALUES = [
    (100.0, 100.0),
    (120.0, 120.0),
    (140.0, 140.0),
    (160.0, 160.0),
]


# Count:
#
# 8 X
# x 7 Y
# x 6 Z/C
# x 4 A/B
#
# = 1344


# =============================================================================
# BATCH 3
#
# LARGE EQUAL A AND B
# =============================================================================

BATCH_3_NAME = "large_equal_ab"


# Keep position relatively constrained while testing larger horizontal size.

BATCH_3_X_VALUES = [
    432500.0,
    432525.0,
    432550.0,
    432575.0,
]


BATCH_3_Y_VALUES = [
    350400.0,
    350425.0,
    350450.0,
    350475.0,
]


# -----------------------------------------------------------------------------
# Again concentrate on lower Z/C.
# -----------------------------------------------------------------------------

BATCH_3_Z_C_PAIRS = [
    (2225.0, 40.0),
    (2250.0, 40.0),
    (2250.0, 50.0),
    (2275.0, 50.0),
    (2275.0, 60.0),
    (2300.0, 60.0),
    (2300.0, 75.0),
    (2325.0, 75.0),
]


# -----------------------------------------------------------------------------
# Larger A = B values.
#
# The previous sweep stopped at approximately 130 m.
# -----------------------------------------------------------------------------

BATCH_3_AB_VALUES = [
    (170.0, 170.0),
    (180.0, 180.0),
    (190.0, 190.0),
    (200.0, 200.0),
    (220.0, 220.0),
]


# Count:
#
# 4 X
# x 4 Y
# x 8 Z/C
# x 5 A/B
#
# = 640


# =============================================================================
# BATCH 4
#
# ELONGATED IN X DIRECTION: A > B
# =============================================================================

BATCH_4_NAME = "elongated_x"


BATCH_4_X_VALUES = [
    432475.0,
    432500.0,
    432525.0,
    432550.0,
    432575.0,
]


BATCH_4_Y_VALUES = [
    350400.0,
    350425.0,
    350450.0,
    350475.0,
    350500.0,
]


# -----------------------------------------------------------------------------
# Only four targeted Z/C pairs here.
#
# This batch is primarily testing the horizontal shape.
# -----------------------------------------------------------------------------

BATCH_4_Z_C_PAIRS = [
    (2225.0, 40.0),
    (2250.0, 50.0),
    (2275.0, 60.0),
    (2300.0, 75.0),
]


# -----------------------------------------------------------------------------
# A > B
#
# A is the x-direction semi-axis.
# B is the y-direction semi-axis.
#
# These test increasingly elongated east-west fill geometries.
# -----------------------------------------------------------------------------

BATCH_4_AB_VALUES = [
    (130.0, 110.0),
    (140.0, 110.0),
    (150.0, 120.0),
    (160.0, 120.0),
    (180.0, 130.0),
    (200.0, 140.0),
]


# Count:
#
# 5 X
# x 5 Y
# x 4 Z/C
# x 6 A/B
#
# = 600


# =============================================================================
# HELPER: VOLUME DIAGNOSTICS
# =============================================================================

def volume_diagnostics(
    dz_m,
    pixel_area_m2,
):

    values = np.asarray(
        dz_m.values,
        dtype=float,
    )

    valid = np.isfinite(
        values
    )

    changed = (
        valid
        & (values != 0.0)
    )


    net = float(
        np.sum(
            values[changed]
        )
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
        np.count_nonzero(
            changed
        )
    )


    return {
        "net_volume_change_m3": net,
        "added_volume_m3": added,
        "removed_volume_m3": removed,
        "changed_pixels_projected": changed_pixels,
    }


# =============================================================================
# HELPER: PRINT VOLUME DIAGNOSTICS
# =============================================================================

def print_volume_diagnostics(
    label,
    stats,
):

    print()
    print(f"        {label}")

    print(
        f"            changed pixels: "
        f"{stats['changed_pixels_projected']}"
    )

    print(
        f"            material added: "
        f"{stats['added_volume_m3']:.1f} m3"
    )

    print(
        f"            material removed: "
        f"{stats['removed_volume_m3']:.1f} m3"
    )

    print(
        f"            net volume change: "
        f"{stats['net_volume_change_m3']:+.1f} m3"
    )


# =============================================================================
# HELPER: GEOMETRY KEY
# =============================================================================

def geometry_key(
    x,
    y,
    z,
    semi_axes,
    rotation,
    interaction,
):

    a, b, c = semi_axes

    r1, r2, r3 = rotation

    return (
        float(x),
        float(y),
        float(z),
        float(a),
        float(b),
        float(c),
        float(r1),
        float(r2),
        float(r3),
        str(interaction),
    )


# =============================================================================
# HELPER: COMPLETE TWO-GEOMETRY MODEL KEY
# =============================================================================

def model_key(
    excavate_geometry,
    fill_geometry,
):

    return (
        geometry_key(
            excavate_geometry["x"],
            excavate_geometry["y"],
            excavate_geometry["z"],
            excavate_geometry["semi_axes"],
            excavate_geometry["rotation"],
            EXCAVATE_INTERACTION,
        ),
        geometry_key(
            fill_geometry["x"],
            fill_geometry["y"],
            fill_geometry["z"],
            fill_geometry["semi_axes"],
            fill_geometry["rotation"],
            FILL_INTERACTION,
        ),
    )


# =============================================================================
# HELPER: NEXT RUN ID
# =============================================================================

def get_next_run_id(
    output_dir,
):

    """
    Return one more than the largest existing DEM OR JSON run ID.
    """

    dem_pattern = re.compile(
        r"^P\.(\d+)\.dem$"
    )

    json_pattern = re.compile(
        r"^(\d+)\.json$"
    )

    existing_ids = []


    for path in output_dir.iterdir():

        if not path.is_file():
            continue


        match = dem_pattern.match(
            path.name
        )

        if match is not None:

            existing_ids.append(
                int(match.group(1))
            )

            continue


        match = json_pattern.match(
            path.name
        )

        if match is not None:

            existing_ids.append(
                int(match.group(1))
            )


    if not existing_ids:
        return 1


    return max(existing_ids) + 1


# =============================================================================
# HELPER: WRITE RUN-ID MANIFEST
# =============================================================================

def write_run_id_manifest(
    path,
    run_ids,
):

    unique_ids = sorted(
        {
            str(run_id)
            for run_id in run_ids
        },
        key=lambda value: int(value),
    )


    path.write_text(
        "".join(
            f"{run_id}\n"
            for run_id in unique_ids
        )
    )


# =============================================================================
# HELPER: BUILD ONE TARGETED BATCH
# =============================================================================

def make_batch(
    batch_name,
    x_values,
    y_values,
    z_c_pairs,
    ab_values,
):

    parameter_sets = []


    for (
        x,
        y,
        z_c,
        ab,
    ) in product(
        x_values,
        y_values,
        z_c_pairs,
        ab_values,
    ):

        z, c = z_c
        a, b = ab


        fill_geometry = {

            "batch_name":
                batch_name,

            "x":
                float(x),

            "y":
                float(y),

            "z":
                float(z),

            "semi_axes": (
                float(a),
                float(b),
                float(c),
            ),

            "rotation":
                FILL_ROTATION,
        }


        parameter_sets.append(
            fill_geometry
        )


    return parameter_sets


# =============================================================================
# 1. READ EXCAVATION MODEL 0075
# =============================================================================

print()
print("=" * 78)
print("READING FIXED EXCAVATION MODEL")
print("=" * 78)


if not EXCAVATION_JSON.exists():

    raise FileNotFoundError(
        f"Excavation JSON not found: "
        f"{EXCAVATION_JSON}"
    )


with open(EXCAVATION_JSON) as f:

    excavation_metadata = json.load(
        f
    )


excavation_shape = None


for shape in excavation_metadata.get(
    "shapes",
    [],
):

    if (
        shape.get("interaction")
        == EXCAVATE_INTERACTION
    ):

        excavation_shape = shape
        break


if excavation_shape is None:

    raise RuntimeError(
        "Could not find an excavate_to_lower "
        "geometry in 0075.json"
    )


excavation_center = (
    excavation_shape[
        "center_xyz_m"
    ]
)

excavation_axes = (
    excavation_shape[
        "semi_axes_m"
    ]
)

excavation_rotation = (
    excavation_shape.get(
        "rotation_deg",
        [0.0, 0.0, 0.0],
    )
)


EXCAVATION = {

    "source_run_id":
        str(
            excavation_metadata.get(
                "id",
                EXCAVATION_JSON.stem,
            )
        ),

    "x":
        float(excavation_center[0]),

    "y":
        float(excavation_center[1]),

    "z":
        float(excavation_center[2]),

    "semi_axes": tuple(
        float(v)
        for v in excavation_axes
    ),

    "rotation": tuple(
        float(v)
        for v in excavation_rotation
    ),
}


print(
    f"Source excavation run ID: "
    f"{EXCAVATION['source_run_id']}"
)

print(
    f"Centre xyz: "
    f"({EXCAVATION['x']:.1f}, "
    f"{EXCAVATION['y']:.1f}, "
    f"{EXCAVATION['z']:.1f})"
)

print(
    f"Semi-axes: "
    f"{EXCAVATION['semi_axes']}"
)

print(
    f"Rotation: "
    f"{EXCAVATION['rotation']}"
)


# =============================================================================
# 2. BUILD FOUR FILL BATCHES
# =============================================================================

batch_1 = make_batch(
    BATCH_1_NAME,
    BATCH_1_X_VALUES,
    BATCH_1_Y_VALUES,
    BATCH_1_Z_C_PAIRS,
    BATCH_1_AB_VALUES,
)


batch_2 = make_batch(
    BATCH_2_NAME,
    BATCH_2_X_VALUES,
    BATCH_2_Y_VALUES,
    BATCH_2_Z_C_PAIRS,
    BATCH_2_AB_VALUES,
)


batch_3 = make_batch(
    BATCH_3_NAME,
    BATCH_3_X_VALUES,
    BATCH_3_Y_VALUES,
    BATCH_3_Z_C_PAIRS,
    BATCH_3_AB_VALUES,
)


batch_4 = make_batch(
    BATCH_4_NAME,
    BATCH_4_X_VALUES,
    BATCH_4_Y_VALUES,
    BATCH_4_Z_C_PAIRS,
    BATCH_4_AB_VALUES,
)


candidate_fill_geometries = (
    batch_1
    + batch_2
    + batch_3
    + batch_4
)


print()
print("=" * 78)
print("CANDIDATE FILL MODEL SPACE")
print("=" * 78)

print(
    f"{BATCH_1_NAME:25s}: "
    f"{len(batch_1)}"
)

print(
    f"{BATCH_2_NAME:25s}: "
    f"{len(batch_2)}"
)

print(
    f"{BATCH_3_NAME:25s}: "
    f"{len(batch_3)}"
)

print(
    f"{BATCH_4_NAME:25s}: "
    f"{len(batch_4)}"
)

print("-" * 78)

print(
    f"{'TOTAL':25s}: "
    f"{len(candidate_fill_geometries)}"
)


# =============================================================================
# 3. READ EXISTING EXCAVATE + FILL MODELS
# =============================================================================

print()
print("=" * 78)
print("READING EXISTING EXCAVATE + FILL MODELS")
print("=" * 78)


existing_model_keys = set()

existing_key_to_run_id = {}

existing_json_files = sorted(
    OUT.glob("*.json")
)


unreadable_jsons = 0


for json_file in existing_json_files:

    try:

        with open(json_file) as f:

            metadata = json.load(f)

    except Exception as exc:

        unreadable_jsons += 1

        print(
            f"WARNING: could not read "
            f"{json_file.name}: {exc}"
        )

        continue


    existing_excavation = None
    existing_fill = None


    for shape in metadata.get(
        "shapes",
        [],
    ):

        role = shape.get(
            "role"
        )

        interaction = shape.get(
            "interaction"
        )


        if (
            role == "excavation"
            or interaction
            == EXCAVATE_INTERACTION
        ):

            center = shape.get(
                "center_xyz_m"
            )

            axes = shape.get(
                "semi_axes_m"
            )

            rotation = shape.get(
                "rotation_deg",
                [0.0, 0.0, 0.0],
            )


            if (
                center is not None
                and axes is not None
            ):

                existing_excavation = {

                    "x":
                        float(center[0]),

                    "y":
                        float(center[1]),

                    "z":
                        float(center[2]),

                    "semi_axes":
                        tuple(
                            float(v)
                            for v in axes
                        ),

                    "rotation":
                        tuple(
                            float(v)
                            for v in rotation
                        ),
                }


        if (
            role == "fill"
            or interaction
            == FILL_INTERACTION
        ):

            center = shape.get(
                "center_xyz_m"
            )

            axes = shape.get(
                "semi_axes_m"
            )

            rotation = shape.get(
                "rotation_deg",
                [0.0, 0.0, 0.0],
            )


            if (
                center is not None
                and axes is not None
            ):

                existing_fill = {

                    "x":
                        float(center[0]),

                    "y":
                        float(center[1]),

                    "z":
                        float(center[2]),

                    "semi_axes":
                        tuple(
                            float(v)
                            for v in axes
                        ),

                    "rotation":
                        tuple(
                            float(v)
                            for v in rotation
                        ),
                }


    if (
        existing_excavation is None
        or existing_fill is None
    ):

        continue


    key = model_key(
        existing_excavation,
        existing_fill,
    )


    existing_model_keys.add(
        key
    )


    if key not in existing_key_to_run_id:

        existing_key_to_run_id[
            key
        ] = str(
            metadata.get(
                "id",
                json_file.stem,
            )
        )


print(
    f"Existing JSON files: "
    f"{len(existing_json_files)}"
)

print(
    f"Existing complete two-shape models: "
    f"{len(existing_model_keys)}"
)

if unreadable_jsons:

    print(
        f"Unreadable JSON files: "
        f"{unreadable_jsons}"
    )


# =============================================================================
# 4. REMOVE DUPLICATES
# =============================================================================

new_fill_geometries = []

seen_new_keys = set()

skipped_existing = []

skipped_between_batches = []


for fill_geometry in candidate_fill_geometries:

    key = model_key(
        EXCAVATION,
        fill_geometry,
    )


    # -------------------------------------------------------------------------
    # Already exists on disk
    # -------------------------------------------------------------------------

    if key in existing_model_keys:

        skipped_existing.append(
            (
                fill_geometry,
                existing_key_to_run_id.get(
                    key,
                    "unknown",
                ),
            )
        )

        continue


    # -------------------------------------------------------------------------
    # Already requested by another new batch
    # -------------------------------------------------------------------------

    if key in seen_new_keys:

        skipped_between_batches.append(
            fill_geometry
        )

        continue


    seen_new_keys.add(
        key
    )

    new_fill_geometries.append(
        fill_geometry
    )


# =============================================================================
# 5. DEDUPLICATION SUMMARY
# =============================================================================

print()
print("=" * 78)
print("DEDUPLICATION SUMMARY")
print("=" * 78)


print(
    f"Candidate models:                "
    f"{len(candidate_fill_geometries)}"
)

print(
    f"Already exist:                   "
    f"{len(skipped_existing)}"
)

print(
    f"Duplicates between new batches:  "
    f"{len(skipped_between_batches)}"
)

print("-" * 78)

print(
    f"GENUINELY NEW MODELS:             "
    f"{len(new_fill_geometries)}"
)


# =============================================================================
# 6. COUNT NEW MODELS BY BATCH
# =============================================================================

print()
print("New models by batch:")


for batch_name in [
    BATCH_1_NAME,
    BATCH_2_NAME,
    BATCH_3_NAME,
    BATCH_4_NAME,
]:

    count = sum(

        fill_geometry[
            "batch_name"
        ]
        == batch_name

        for fill_geometry
        in new_fill_geometries
    )


    print(
        f"    {batch_name:25s}: "
        f"{count}"
    )


# =============================================================================
# 7. PRINT FINAL MODEL-SPACE RANGES
# =============================================================================

if new_fill_geometries:

    all_x = [
        model["x"]
        for model in new_fill_geometries
    ]

    all_y = [
        model["y"]
        for model in new_fill_geometries
    ]

    all_z = [
        model["z"]
        for model in new_fill_geometries
    ]

    all_a = [
        model["semi_axes"][0]
        for model in new_fill_geometries
    ]

    all_b = [
        model["semi_axes"][1]
        for model in new_fill_geometries
    ]

    all_c = [
        model["semi_axes"][2]
        for model in new_fill_geometries
    ]


    print()
    print("=" * 78)
    print("FINAL NEW FILL MODEL SPACE")
    print("=" * 78)


    print(
        f"X: "
        f"{min(all_x):.1f} -> "
        f"{max(all_x):.1f} m"
    )

    print(
        f"Y: "
        f"{min(all_y):.1f} -> "
        f"{max(all_y):.1f} m"
    )

    print(
        f"Z: "
        f"{min(all_z):.1f} -> "
        f"{max(all_z):.1f} m"
    )

    print(
        f"A: "
        f"{min(all_a):.1f} -> "
        f"{max(all_a):.1f} m"
    )

    print(
        f"B: "
        f"{min(all_b):.1f} -> "
        f"{max(all_b):.1f} m"
    )

    print(
        f"C: "
        f"{min(all_c):.1f} -> "
        f"{max(all_c):.1f} m"
    )


    print()
    print("Unique tested values:")

    print(
        f"    X: {sorted(set(all_x))}"
    )

    print(
        f"    Y: {sorted(set(all_y))}"
    )

    print(
        f"    Z: {sorted(set(all_z))}"
    )

    print(
        f"    A: {sorted(set(all_a))}"
    )

    print(
        f"    B: {sorted(set(all_b))}"
    )

    print(
        f"    C: {sorted(set(all_c))}"
    )


# =============================================================================
# 8. SHOW FIRST EXISTING DUPLICATES
# =============================================================================

if skipped_existing:

    print()
    print(
        "First existing models skipped:"
    )


    for (
        fill_geometry,
        existing_id,
    ) in skipped_existing[:10]:

        print(
            f"    existing run "
            f"{existing_id}: "
            f"batch="
            f"{fill_geometry['batch_name']}, "
            f"x={fill_geometry['x']}, "
            f"y={fill_geometry['y']}, "
            f"z={fill_geometry['z']}, "
            f"axes="
            f"{fill_geometry['semi_axes']}"
        )


    if len(skipped_existing) > 10:

        print(
            f"    ... plus "
            f"{len(skipped_existing) - 10} "
            f"more"
        )


# =============================================================================
# 9. DRY-RUN EXIT
# =============================================================================

if DRY_RUN:

    print()
    print("=" * 78)
    print("DRY RUN COMPLETE")
    print("=" * 78)

    print(
        "No DEMs or JSON files were created."
    )

    print(
        f"The real run would attempt to create "
        f"{len(new_fill_geometries)} new DEMs."
    )

    print()
    print(
        "To generate the models, run again "
        "without --dry-run."
    )

    raise SystemExit(0)


# =============================================================================
# 10. NOTHING TO CREATE
# =============================================================================

if not new_fill_geometries:

    print()
    print(
        "All requested models already exist."
    )

    raise SystemExit(0)


# =============================================================================
# 11. READ ORIGINAL DEM
# =============================================================================

print()
print("=" * 78)
print("READING ORIGINAL DEM")
print("=" * 78)


dem_geo_original, meta = read_gamma_dem(
    DEM,
    PAR,
)


print(
    f"Original shape: "
    f"{dem_geo_original.shape}"
)

print(
    f"Original CRS: "
    f"{dem_geo_original.rio.crs}"
)


# =============================================================================
# 12. PROJECT ORIGINAL DEM
# =============================================================================

print()
print(
    "[SETUP] Creating projected metre-grid DEM"
)


dem_m_original = project_dem_nearest(
    dem_geo_original
)


print(
    f"Projected CRS: "
    f"{dem_m_original.rio.crs}"
)

print(
    f"Resolution: "
    f"{dem_m_original.rio.resolution()} m"
)


# =============================================================================
# 13. REFERENCE SURFACE HEIGHT
# =============================================================================

reference_surface_z = float(

    dem_m_original.sel(
        x=REFERENCE_X,
        y=REFERENCE_Y,
        method="nearest",
    )
)


print(
    f"Reference surface z: "
    f"{reference_surface_z:.3f} m"
)


# =============================================================================
# 14. PIXEL AREA
# =============================================================================

dx, dy = (
    dem_m_original.rio.resolution()
)

pixel_area_m2 = abs(
    dx * dy
)


# =============================================================================
# 15. CREATE FIXED EXCAVATION ONCE
# =============================================================================
#
# Every new model uses exactly the same excavation.
#
# Therefore there is no reason to calculate the excavation thousands of times.
# This is an important efficiency improvement.
# =============================================================================

print()
print("=" * 78)
print("CREATING FIXED 0075 EXCAVATION")
print("=" * 78)


excavate_shape = RotatedEllipsoid(

    center=(
        EXCAVATION["x"],
        EXCAVATION["y"],
        EXCAVATION["z"],
    ),

    semi_axes=(
        EXCAVATION[
            "semi_axes"
        ]
    ),

    rotation_deg=(
        EXCAVATION[
            "rotation"
        ]
    ),
)


dem_after_excavate, dz_excavate = (
    apply_shape(
        dem_m_original,
        excavate_shape,
        interaction=EXCAVATE_INTERACTION,
    )
)


excavate_stats = (
    volume_diagnostics(
        dz_excavate,
        pixel_area_m2,
    )
)


print_volume_diagnostics(
    "Fixed excavation diagnostics",
    excavate_stats,
)


# =============================================================================
# 16. FIRST AVAILABLE RUN ID
# =============================================================================

next_numeric_id = (
    get_next_run_id(
        OUT
    )
)


print()
print(
    f"First available run ID: "
    f"{next_numeric_id:06d}"
)


# =============================================================================
# 17. CREATE NEW MODELS
# =============================================================================

newly_created_run_ids = []


created_by_batch = {

    BATCH_1_NAME: 0,
    BATCH_2_NAME: 0,
    BATCH_3_NAME: 0,
    BATCH_4_NAME: 0,
}


for model_number, fill_geometry in enumerate(
    new_fill_geometries,
    start=1,
):

    run_id = (
        f"{next_numeric_id:06d}"
    )


    fill_x = (
        fill_geometry["x"]
    )

    fill_y = (
        fill_geometry["y"]
    )

    fill_z = (
        fill_geometry["z"]
    )

    fill_semi_axes = (
        fill_geometry[
            "semi_axes"
        ]
    )

    fill_rotation = (
        fill_geometry[
            "rotation"
        ]
    )

    batch_name = (
        fill_geometry[
            "batch_name"
        ]
    )


    print()
    print("=" * 78)

    print(
        f"[RUN {run_id}] "
        f"{model_number}/"
        f"{len(new_fill_geometries)}"
    )

    print("=" * 78)

    print(
        f"        batch = "
        f"{batch_name}"
    )

    print(
        f"        fill centre xyz = "
        f"({fill_x:.1f}, "
        f"{fill_y:.1f}, "
        f"{fill_z:.1f})"
    )

    print(
        f"        fill semi-axes = "
        f"{fill_semi_axes}"
    )


    # =========================================================================
    # FILL SHAPE
    # =========================================================================

    fill_shape = RotatedEllipsoid(

        center=(
            fill_x,
            fill_y,
            fill_z,
        ),

        semi_axes=(
            fill_semi_axes
        ),

        rotation_deg=(
            fill_rotation
        ),
    )


    # =========================================================================
    # APPLY FILL TO THE FIXED EXCAVATED DEM
    # =========================================================================

    dem_m_final, dz_fill = (
        apply_shape(
            dem_after_excavate,
            fill_shape,
            interaction=FILL_INTERACTION,
        )
    )


    # =========================================================================
    # NET DISPLACEMENT RELATIVE TO ORIGINAL DEM
    # =========================================================================
    #
    # This is important.
    #
    # dz_fill is relative to the excavated DEM.
    #
    # For transfer back to the original geographic GAMMA grid we need the
    # TOTAL displacement after excavation + fill relative to the original DEM.
    # =========================================================================

    dz_total = (
        dem_m_final
        - dem_m_original
    )


    # =========================================================================
    # VOLUME DIAGNOSTICS
    # =========================================================================

    fill_stats = (
        volume_diagnostics(
            dz_fill,
            pixel_area_m2,
        )
    )


    total_stats = (
        volume_diagnostics(
            dz_total,
            pixel_area_m2,
        )
    )


    print_volume_diagnostics(
        "Fill diagnostics",
        fill_stats,
    )

    print_volume_diagnostics(
        "Combined final DEM diagnostics",
        total_stats,
    )


    # =========================================================================
    # TRANSFER NET DISPLACEMENT TO ORIGINAL GRID
    # =========================================================================

    dem_geo_modified, dz_geo = (
        transfer_displacement_to_original_grid(
            dem_geo_original,
            dz_total,
        )
    )


    # =========================================================================
    # VALIDATE ORIGINAL GRID
    # =========================================================================

    validation = (
        validate_original_grid(
            dem_geo_original,
            dem_geo_modified,
        )
    )


    assert validation[
        "shape_exact"
    ]

    assert validation[
        "x_exact"
    ]

    assert validation[
        "y_exact"
    ]

    assert validation[
        "transform_exact"
    ]

    assert validation[
        "crs_exact"
    ]

    assert (
        validation[
            "nan_pixels"
        ]
        == 0
    )

    assert validation[
        "unchanged_pixels_exact"
    ]


    # =========================================================================
    # OUTPUT PATHS
    # =========================================================================

    dem_out = (
        OUT
        / f"P.{run_id}.dem"
    )

    json_out = (
        OUT
        / f"{run_id}.json"
    )


    # =========================================================================
    # FINAL ID COLLISION CHECK
    # =========================================================================

    if (
        dem_out.exists()
        or json_out.exists()
    ):

        raise FileExistsError(
            "Refusing to overwrite an existing "
            f"run ID: {run_id}"
        )


    # =========================================================================
    # WRITE DEM
    # =========================================================================

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
    ] = EXCAVATION[
        "source_run_id"
    ]


    excavate_record[
        "source_excavation_json"
    ] = str(
        EXCAVATION_JSON
    )


    excavate_record[
        "sweep_parameters"
    ] = {

        "x_m":
            float(
                EXCAVATION["x"]
            ),

        "y_m":
            float(
                EXCAVATION["y"]
            ),

        "z_m":
            float(
                EXCAVATION["z"]
            ),

        "semi_axes_m": [
            float(v)
            for v in EXCAVATION[
                "semi_axes"
            ]
        ],

        "rotation_deg": [
            float(v)
            for v in EXCAVATION[
                "rotation"
            ]
        ],
    }


    excavate_record[
        "surface_intersection"
    ] = {

        "changed_pixels_projected":
            int(
                excavate_stats[
                    "changed_pixels_projected"
                ]
            ),

        "modifies_surface":
            bool(
                excavate_stats[
                    "changed_pixels_projected"
                ]
                > 0
            ),
    }


    excavate_record[
        "volume_diagnostics"
    ] = excavate_stats


    # =========================================================================
    # FILL PROVENANCE
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
        "sweep_parameters"
    ] = {

        "batch_name":
            batch_name,

        "x_m":
            float(fill_x),

        "y_m":
            float(fill_y),

        "z_m":
            float(fill_z),

        "semi_axes_m": [
            float(v)
            for v in fill_semi_axes
        ],

        "rotation_deg": [
            float(v)
            for v in fill_rotation
        ],
    }


    fill_record[
        "surface_intersection"
    ] = {

        "changed_pixels_projected":
            int(
                fill_stats[
                    "changed_pixels_projected"
                ]
            ),

        "modifies_surface":
            bool(
                fill_stats[
                    "changed_pixels_projected"
                ]
                > 0
            ),
    }


    fill_record[
        "volume_diagnostics"
    ] = fill_stats


    # =========================================================================
    # COMMON REFERENCE INFORMATION
    # =========================================================================

    for record in (
        excavate_record,
        fill_record,
    ):

        record[
            "reference"
        ] = {

            "reference_x_m":
                float(
                    REFERENCE_X
                ),

            "reference_y_m":
                float(
                    REFERENCE_Y
                ),

            "reference_surface_z_m":
                float(
                    reference_surface_z
                ),
        }


    # =========================================================================
    # FINAL COMBINED DIAGNOSTICS
    # =========================================================================

    excavate_record[
        "combined_final_diagnostics"
    ] = total_stats


    fill_record[
        "combined_final_diagnostics"
    ] = total_stats


    # =========================================================================
    # QA OUTPUTS
    # =========================================================================

    qa_outputs = {

        "modified_dem_plot":
            None,

        "difference_plot":
            None,
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


    # =========================================================================
    # UPDATE IN-MEMORY DUPLICATE LOOKUP
    # =========================================================================

    key = model_key(
        EXCAVATION,
        fill_geometry,
    )


    existing_model_keys.add(
        key
    )

    existing_key_to_run_id[
        key
    ] = run_id


    # =========================================================================
    # RECORD NEW ID
    # =========================================================================

    newly_created_run_ids.append(
        run_id
    )


    created_by_batch[
        batch_name
    ] += 1


    print()
    print("        Outputs:")

    print(
        f"            {dem_out}"
    )

    print(
        f"            {json_out}"
    )


    # =========================================================================
    # NEXT ID
    # =========================================================================

    next_numeric_id += 1


# =============================================================================
# 18. WRITE SIMSAR MANIFEST
# =============================================================================

manifest = (
    OUT
    / "0075_fill_exploration_new_models_for_simsar.txt"
)


write_run_id_manifest(
    manifest,
    newly_created_run_ids,
)


# =============================================================================
# 19. FINAL SUMMARY
# =============================================================================

print()
print("=" * 78)
print("FINISHED")
print("=" * 78)


print(
    f"Created "
    f"{len(newly_created_run_ids)} "
    f"new DEMs."
)


print()
print("Created by batch:")


for (
    batch_name,
    count,
) in created_by_batch.items():

    print(
        f"    {batch_name:25s}: "
        f"{count}"
    )


print()
print(
    f"SimSAR run-ID manifest:"
)

print(
    f"    {manifest}"
)
