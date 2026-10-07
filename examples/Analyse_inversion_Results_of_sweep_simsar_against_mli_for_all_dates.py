#!/usr/bin/env python3

from pathlib import Path
import json

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# PLOT STYLE
# =============================================================================

mpl.rcParams.update({
    "figure.figsize": (10.0, 8.0),
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# SETTINGS
# =============================================================================

BASE_DIR = Path(
    "/scratch/ee16eme/sinabung_asc_tsx/new_dense_for_each_date"
)

MODEL_DIR = Path(
    "/scratch/ee16eme/sinabung_asc_tsx/"
    "mod_dem_Dome/"
    "synthetic_sweep_excavate6799_existing_fill"
)

DATES = [
    "20201021",
    "20201101",
    "20201226",
    "20210106",
    "20210117",
    "20210128",
    "20210208",
    "20210219",
]

# Number of lowest-RMSE models shown at each date
N_BEST = 5

OUTPUT_FIGURE = (
    BASE_DIR
    / "top5_model_evolution.png"
)

OUTPUT_PDF = (
    BASE_DIR
    / "top5_model_evolution.pdf"
)

OUTPUT_CSV = (
    BASE_DIR
    / "top5_model_evolution.csv"
)


# =============================================================================
# 1. READ MODEL JSON METADATA
# =============================================================================

rows = []

json_files = sorted(
    MODEL_DIR.glob("*.json")
)

print(
    f"Found {len(json_files)} JSON files"
)


for json_file in json_files:

    with open(json_file) as f:
        metadata = json.load(f)

    # -------------------------------------------------------------------------
    # Run ID
    # -------------------------------------------------------------------------

    run_id = str(
        metadata.get(
            "id",
            json_file.stem,
        )
    ).zfill(6)


    # -------------------------------------------------------------------------
    # Find excavation and fill records
    # -------------------------------------------------------------------------

    excavation = None
    fill = None

    for shape in metadata.get(
        "shapes",
        [],
    ):

        role = shape.get("role")

        interaction = shape.get(
            "interaction"
        )

        if (
            role == "excavation"
            or interaction == "excavate_to_lower"
        ):
            excavation = shape

        elif (
            role == "fill"
            or interaction == "fill_to_upper"
        ):
            fill = shape


    if excavation is None or fill is None:

        print(
            f"Skipping {json_file.name}: "
            f"missing excavation or fill"
        )

        continue


    # =========================================================================
    # EXCAVATION GEOMETRY
    # =========================================================================
    #
    # Older JSONs may contain:
    #
    #     excavation["sweep_parameters"]
    #
    # Newer fixed-excavation JSONs may instead contain the geometry directly:
    #
    #     excavation["center_xyz_m"]
    #     excavation["semi_axes_m"]
    #
    # Support both formats.
    # =========================================================================

    if "sweep_parameters" in excavation:

        exc = excavation[
            "sweep_parameters"
        ]

        exc_x = float(
            exc["x_m"]
        )

        exc_y = float(
            exc["y_m"]
        )

        exc_z = float(
            exc["z_m"]
        )

        exc_a, exc_b, exc_c = [
            float(v)
            for v in exc[
                "semi_axes_m"
            ]
        ]

    else:

        exc_x, exc_y, exc_z = [
            float(v)
            for v in excavation[
                "center_xyz_m"
            ]
        ]

        exc_a, exc_b, exc_c = [
            float(v)
            for v in excavation[
                "semi_axes_m"
            ]
        ]


    # =========================================================================
    # FILL GEOMETRY
    # =========================================================================
    #
    # Again support both formats.
    # =========================================================================

    if "sweep_parameters" in fill:

        fil = fill[
            "sweep_parameters"
        ]

        fill_x = float(
            fil["x_m"]
        )

        fill_y = float(
            fil["y_m"]
        )

        fill_z = float(
            fil["z_m"]
        )

        fill_a, fill_b, fill_c = [
            float(v)
            for v in fil[
                "semi_axes_m"
            ]
        ]

    else:

        fill_x, fill_y, fill_z = [
            float(v)
            for v in fill[
                "center_xyz_m"
            ]
        ]

        fill_a, fill_b, fill_c = [
            float(v)
            for v in fill[
                "semi_axes_m"
            ]
        ]


    # =========================================================================
    # VOLUME DIAGNOSTICS
    # =========================================================================

    exc_volume = excavation[
        "volume_diagnostics"
    ]

    fill_volume = fill[
        "volume_diagnostics"
    ]


    # -------------------------------------------------------------------------
    # Final modified DEM relative to the original DEM
    #
    # This is the volume quantity we want to use for the final added volume.
    # -------------------------------------------------------------------------

    final_volume = fill[
        "combined_final_diagnostics"
    ]


    # =========================================================================
    # SOURCE MODEL INFORMATION
    # =========================================================================

    source_excavation_run_id = (
        excavation.get(
            "source_excavation_run_id"
        )
    )

    source_fill_run_id = (
        fill.get(
            "source_fill_run_id"
        )
    )


    # =========================================================================
    # STORE MODEL
    # =========================================================================

    rows.append(
        {
            "run_id":
                run_id,

            # -----------------------------------------------------------------
            # Source models
            # -----------------------------------------------------------------

            "source_excavation_run_id":
                source_excavation_run_id,

            "source_fill_run_id":
                source_fill_run_id,

            # -----------------------------------------------------------------
            # Excavation geometry
            # -----------------------------------------------------------------

            "excavation_x_m":
                exc_x,

            "excavation_y_m":
                exc_y,

            "excavation_z_m":
                exc_z,

            "excavation_a_m":
                exc_a,

            "excavation_b_m":
                exc_b,

            "excavation_c_m":
                exc_c,

            # -----------------------------------------------------------------
            # Fill geometry
            # -----------------------------------------------------------------

            "fill_x_m":
                fill_x,

            "fill_y_m":
                fill_y,

            "fill_z_m":
                fill_z,

            "fill_a_m":
                fill_a,

            "fill_b_m":
                fill_b,

            "fill_c_m":
                fill_c,

            # -----------------------------------------------------------------
            # Excavation volume
            # -----------------------------------------------------------------

            "excavation_removed_volume_m3":
                float(
                    exc_volume[
                        "removed_volume_m3"
                    ]
                ),

            # -----------------------------------------------------------------
            # Fill-operation volume
            #
            # This is only what the fill operation itself added.
            # -----------------------------------------------------------------

            "fill_added_volume_m3":
                float(
                    fill_volume[
                        "added_volume_m3"
                    ]
                ),

            # -----------------------------------------------------------------
            # Final DEM volumes
            #
            # These describe the final combined excavation + fill DEM relative
            # to the original DEM.
            # -----------------------------------------------------------------

            "final_added_volume_m3":
                float(
                    final_volume[
                        "added_volume_m3"
                    ]
                ),

            "final_removed_volume_m3":
                float(
                    final_volume[
                        "removed_volume_m3"
                    ]
                ),

            "final_net_volume_m3":
                float(
                    final_volume[
                        "net_volume_change_m3"
                    ]
                ),
        }
    )


# =============================================================================
# 2. CREATE MODEL METADATA DATAFRAME
# =============================================================================

models = pd.DataFrame(
    rows
)


# -----------------------------------------------------------------------------
# Give a useful error if no models could be loaded.
# -----------------------------------------------------------------------------

if models.empty:

    raise RuntimeError(
        "No usable excavation + fill models "
        f"were read from:\n{MODEL_DIR}"
    )


models["run_number"] = (
    models["run_id"]
    .astype(int)
)


models = (
    models
    .sort_values(
        "run_number"
    )
    .reset_index(
        drop=True
    )
)


# -----------------------------------------------------------------------------
# Final added volume in millions of cubic metres
# -----------------------------------------------------------------------------

models[
    "final_added_volume_Mm3"
] = (
    models[
        "final_added_volume_m3"
    ]
    / 1e6
)


print(
    f"Models loaded: {len(models)}"
)


# -----------------------------------------------------------------------------
# Print basic information about the fixed excavation
# -----------------------------------------------------------------------------

print()
print(
    "Unique excavation geometries:"
)

unique_excavations = (
    models[
        [
            "excavation_x_m",
            "excavation_y_m",
            "excavation_z_m",
            "excavation_a_m",
            "excavation_b_m",
            "excavation_c_m",
        ]
    ]
    .drop_duplicates()
)


print(
    unique_excavations.to_string(
        index=False
    )
)


# =============================================================================
# 3. READ INVERSION RESULTS
# =============================================================================

all_results = []


for date_string in DATES:

    inversion_dir = (
        BASE_DIR
        / f"peak_model_inversion_dense_{date_string}"
    )

    ranking_file = (
        inversion_dir
        / "peak_model_ranking.csv"
    )


    print()
    print(
        f"Reading {ranking_file}"
    )


    if not ranking_file.exists():

        raise FileNotFoundError(
            f"Ranking file not found:\n"
            f"{ranking_file}"
        )


    ranking = pd.read_csv(
        ranking_file,
        dtype={
            "run_id": str,
        },
    )


    # -------------------------------------------------------------------------
    # Normalise IDs
    # -------------------------------------------------------------------------

    ranking["run_id"] = (
        ranking["run_id"]
        .str.strip()
        .str.zfill(6)
    )


    # -------------------------------------------------------------------------
    # Keep valid inversion results only
    # -------------------------------------------------------------------------

    ranking = ranking[
        (
            ranking["status"]
            == "ok"
        )
        &
        ranking[
            "rmse_filtered_m"
        ].notna()
    ].copy()


    print(
        f"    valid inversion models: "
        f"{len(ranking)}"
    )


    # -------------------------------------------------------------------------
    # Lowest RMSE first
    # -------------------------------------------------------------------------

    ranking = (
        ranking
        .sort_values(
            "rmse_filtered_m"
        )
        .reset_index(
            drop=True
        )
    )


    # -------------------------------------------------------------------------
    # Keep N lowest-RMSE models
    # -------------------------------------------------------------------------

    ranking = (
        ranking
        .head(N_BEST)
        .copy()
    )


    # -------------------------------------------------------------------------
    # Rank 1 = best-fitting model
    # -------------------------------------------------------------------------

    ranking["rank"] = range(
        1,
        len(ranking) + 1,
    )


    # -------------------------------------------------------------------------
    # Acquisition date
    # -------------------------------------------------------------------------

    ranking["date"] = (
        pd.to_datetime(
            date_string,
            format="%Y%m%d",
        )
    )


    # -------------------------------------------------------------------------
    # Join inversion results to model metadata
    # -------------------------------------------------------------------------

    ranking = ranking.merge(
        models,
        on="run_id",
        how="left",
        validate="many_to_one",
    )


    all_results.append(
        ranking
    )


# =============================================================================
# 4. COMBINE ALL DATES
# =============================================================================

if not all_results:

    raise RuntimeError(
        "No inversion results were loaded."
    )


top5 = pd.concat(
    all_results,
    ignore_index=True,
)


top5 = (
    top5
    .sort_values(
        [
            "date",
            "rank",
        ]
    )
    .reset_index(
        drop=True
    )
)


# -----------------------------------------------------------------------------
# Rank 1 models only
# -----------------------------------------------------------------------------

best = (
    top5[
        top5["rank"] == 1
    ]
    .sort_values(
        "date"
    )
    .copy()
)


# =============================================================================
# 5. CHECK FOR MISSING METADATA
# =============================================================================

missing = top5[
    top5[
        "final_added_volume_m3"
    ].isna()
]


if len(missing) > 0:

    print()
    print("=" * 100)

    print(
        "WARNING: SOME RUN IDs DID NOT "
        "MATCH JSON METADATA"
    )

    print("=" * 100)

    print(
        missing[
            [
                "date",
                "run_id",
                "rmse_filtered_m",
            ]
        ].to_string(
            index=False
        )
    )


# =============================================================================
# 6. PRINT RESULTS
# =============================================================================

print()
print("=" * 120)

print(
    "FIVE LOWEST-RMSE MODELS FOR EACH DATE"
)

print("=" * 120)


print(
    top5[
        [
            "date",
            "rank",
            "run_id",

            "source_excavation_run_id",
            "source_fill_run_id",

            "fill_a_m",
            "fill_b_m",
            "fill_c_m",

            "fill_z_m",
            "fill_x_m",
            "fill_y_m",

            "final_added_volume_m3",

            "rmse_filtered_m",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# 7. PRINT BEST MODEL FOR EACH DATE
# =============================================================================

print()
print("=" * 120)

print(
    "BEST-FITTING MODEL FOR EACH DATE"
)

print("=" * 120)


print(
    best[
        [
            "date",
            "run_id",

            "fill_a_m",
            "fill_b_m",
            "fill_c_m",

            "fill_z_m",
            "fill_x_m",
            "fill_y_m",

            "final_added_volume_m3",

            "rmse_filtered_m",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# 8. SAVE RESULTS TABLE
# =============================================================================

top5.to_csv(
    OUTPUT_CSV,
    index=False,
)


print()
print(
    f"Saved table: {OUTPUT_CSV}"
)


# =============================================================================
# 9. PARAMETERS TO PLOT
# =============================================================================

plot_parameters = [

    # -------------------------------------------------------------------------
    # Row 1
    # -------------------------------------------------------------------------

    (
        "fill_a_m",
        "A",
        "Semi-axis A (m)",
        1,
    ),

    (
        "fill_b_m",
        "B",
        "Semi-axis B (m)",
        1,
    ),

    (
        "fill_c_m",
        "C",
        "Semi-axis C (m)",
        1,
    ),

    # -------------------------------------------------------------------------
    # Row 2
    # -------------------------------------------------------------------------

    (
        "fill_z_m",
        "z",
        "z (m)",
        1,
    ),

    (
        "fill_x_m",
        "x",
        "x (m)",
        1,
    ),

    (
        "fill_y_m",
        "y",
        "y (m)",
        1,
    ),

    # -------------------------------------------------------------------------
    # Row 3
    # -------------------------------------------------------------------------

    (
        "final_added_volume_m3",
        "Final added volume",
        "Volume ($10^6$ m$^3$)",
        1e6,
    ),

    (
        "rmse_filtered_m",
        "Model fit",
        "RMSE (m)",
        1,
    ),
]


# =============================================================================
# 10. CREATE 3 x 3 FIGURE
# =============================================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(10, 8),
    sharex=True,
)


axes = axes.flatten()


# =============================================================================
# 11. PLOT PARAMETERS
# =============================================================================

for ax, (
    column,
    title,
    ylabel,
    scale,
) in zip(
    axes,
    plot_parameters,
):


    # -------------------------------------------------------------------------
    # Five lowest-RMSE models
    # -------------------------------------------------------------------------

    ax.scatter(
        top5["date"],
        top5[column] / scale,
        s=24,
        alpha=0.35,
        label="Five lowest RMSE",
        zorder=2,
    )


    # -------------------------------------------------------------------------
    # Best-fitting model
    # -------------------------------------------------------------------------

    ax.plot(
        best["date"],
        best[column] / scale,
        marker="o",
        markersize=4.5,
        linewidth=1.3,
        label="Best fit",
        zorder=3,
    )


    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------

    ax.set_title(
        title
    )

    ax.set_ylabel(
        ylabel
    )


    # -------------------------------------------------------------------------
    # Light horizontal grid only
    # -------------------------------------------------------------------------

    ax.grid(
        axis="y",
        alpha=0.2,
        linewidth=0.6,
    )


    # -------------------------------------------------------------------------
    # Remove unnecessary borders
    # -------------------------------------------------------------------------

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


# =============================================================================
# 12. DATE FORMATTING
# =============================================================================

for ax in axes[:8]:

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(
            interval=2
        )
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%b\n%Y"
        )
    )


# =============================================================================
# 13. NINTH PANEL = LEGEND
# =============================================================================

legend_ax = axes[8]

legend_ax.axis(
    "off"
)


handles, labels = (
    axes[0]
    .get_legend_handles_labels()
)


legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
    fontsize=9,
)


legend_ax.text(
    0.5,
    0.28,
    "Points show the five models\n"
    "with the lowest RMSE at each date.\n\n"
    "The line follows the\n"
    "lowest-RMSE model.",
    ha="center",
    va="center",
    transform=legend_ax.transAxes,
    fontsize=8,
)


# =============================================================================
# 14. PANEL LABELS
# =============================================================================

panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
    "(e)",
    "(f)",
    "(g)",
    "(h)",
]


for ax, label in zip(
    axes[:8],
    panel_labels,
):

    ax.text(
        0.02,
        0.96,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


# =============================================================================
# 15. FINAL FORMATTING
# =============================================================================

fig.suptitle(
    "Evolution of the five lowest-RMSE models",
    fontsize=11,
    y=0.99,
)


fig.tight_layout(
    rect=[
        0,
        0,
        1,
        0.97,
    ],
    h_pad=1.3,
    w_pad=1.5,
)


# =============================================================================
# 16. SAVE FIGURES
# =============================================================================

fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
)


fig.savefig(
    OUTPUT_PDF,
)


print()
print(
    f"Saved PNG: {OUTPUT_FIGURE}"
)

print(
    f"Saved PDF: {OUTPUT_PDF}"
)


# =============================================================================
# 17. SHOW
# =============================================================================

plt.show()