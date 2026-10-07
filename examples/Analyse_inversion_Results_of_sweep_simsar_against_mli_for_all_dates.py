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

BASE_DIR = Path("/scratch/ee16eme/sinabung_asc_tsx/new_dense_for_each_date")

MODEL_DIR = (
    BASE_DIR
    / "mod_dem_Dome"
    / "synthetic_sweep_excavate6799_existing_fill"
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

OUTPUT_FIGURE = BASE_DIR / "top5_model_evolution.png"
OUTPUT_PDF = BASE_DIR / "top5_model_evolution.pdf"
OUTPUT_CSV = BASE_DIR / "top5_model_evolution.csv"


# =============================================================================
# 1. READ MODEL JSON METADATA
# =============================================================================

rows = []

json_files = sorted(MODEL_DIR.glob("*.json"))

print(f"Found {len(json_files)} JSON files")


for json_file in json_files:

    with open(json_file) as f:
        metadata = json.load(f)

    run_id = json_file.stem

    excavation = None
    fill = None

    # Find the excavation and fill records
    for shape in metadata["shapes"]:

        if shape.get("role") == "excavation":
            excavation = shape

        elif shape.get("role") == "fill":
            fill = shape

    if excavation is None or fill is None:
        print(f"Skipping {json_file.name}: missing excavation or fill")
        continue

    # Geometry
    exc = excavation["sweep_parameters"]
    fil = fill["sweep_parameters"]

    exc_a, exc_b, exc_c = exc["semi_axes_m"]
    fill_a, fill_b, fill_c = fil["semi_axes_m"]

    # Volume diagnostics
    exc_volume = excavation["volume_diagnostics"]
    fill_volume = fill["volume_diagnostics"]

    # Final modified DEM relative to original DEM
    final_volume = fill["combined_final_diagnostics"]

    rows.append(
        {
            "run_id": run_id,

            # Excavation geometry
            "excavation_x_m": exc["x_m"],
            "excavation_y_m": exc["y_m"],
            "excavation_z_m": exc["z_m"],
            "excavation_a_m": exc_a,
            "excavation_b_m": exc_b,
            "excavation_c_m": exc_c,

            # Fill geometry
            "fill_x_m": fil["x_m"],
            "fill_y_m": fil["y_m"],
            "fill_z_m": fil["z_m"],
            "fill_a_m": fill_a,
            "fill_b_m": fill_b,
            "fill_c_m": fill_c,

            # Volume diagnostics
            "excavation_removed_volume_m3":
                exc_volume["removed_volume_m3"],

            "fill_added_volume_m3":
                fill_volume["added_volume_m3"],

            "final_added_volume_m3":
                final_volume["added_volume_m3"],

            "final_removed_volume_m3":
                final_volume["removed_volume_m3"],

            "final_net_volume_m3":
                final_volume["net_volume_change_m3"],
        }
    )


# =============================================================================
# 2. CREATE MODEL METADATA DATAFRAME
# =============================================================================

models = pd.DataFrame(rows)

models["run_number"] = models["run_id"].astype(int)

models = (
    models
    .sort_values("run_number")
    .reset_index(drop=True)
)

# Final added volume in millions of cubic metres
models["final_added_volume_Mm3"] = (
    models["final_added_volume_m3"] / 1e6
)

print(f"Models loaded: {len(models)}")


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

    print(f"Reading {ranking_file}")

    ranking = pd.read_csv(
        ranking_file,
        dtype={"run_id": str},
    )

    # Keep six-digit IDs
    ranking["run_id"] = (
        ranking["run_id"]
        .str.zfill(6)
    )

    # Keep valid models only
    ranking = ranking[
        (ranking["status"] == "ok")
        & ranking["rmse_filtered_m"].notna()
    ].copy()

    # Lowest RMSE first
    ranking = ranking.sort_values(
        "rmse_filtered_m"
    )

    # Keep the lowest five RMSE models
    ranking = ranking.head(N_BEST).copy()

    # Rank 1 = best-fitting model
    ranking["rank"] = range(
        1,
        len(ranking) + 1,
    )

    # Acquisition date
    ranking["date"] = pd.to_datetime(
        date_string,
        format="%Y%m%d",
    )

    # Join inversion results to model metadata
    ranking = ranking.merge(
        models,
        on="run_id",
        how="left",
    )

    all_results.append(ranking)


# =============================================================================
# 4. COMBINE DATES
# =============================================================================

top5 = pd.concat(
    all_results,
    ignore_index=True,
)

top5 = top5.sort_values(
    ["date", "rank"]
)

# Rank 1 models only
best = (
    top5[
        top5["rank"] == 1
    ]
    .sort_values("date")
    .copy()
)


# =============================================================================
# 5. CHECK FOR MISSING METADATA
# =============================================================================

missing = top5[
    top5["final_added_volume_m3"].isna()
]

if len(missing) > 0:

    print("\nWARNING: Some run IDs did not match JSON metadata:")

    print(
        missing[
            [
                "date",
                "run_id",
                "rmse_filtered_m",
            ]
        ]
    )


# =============================================================================
# 6. PRINT RESULTS
# =============================================================================

print("\n")
print("=" * 100)
print("FIVE LOWEST-RMSE MODELS FOR EACH DATE")
print("=" * 100)

print(
    top5[
        [
            "date",
            "rank",
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
    ].to_string(index=False)
)


# =============================================================================
# 7. SAVE RESULTS TABLE
# =============================================================================

top5.to_csv(
    OUTPUT_CSV,
    index=False,
)

print(f"\nSaved table: {OUTPUT_CSV}")


# =============================================================================
# 8. PARAMETERS TO PLOT
# =============================================================================

plot_parameters = [

    # Row 1
    ("fill_a_m", "A", "Semi-axis A (m)", 1),
    ("fill_b_m", "B", "Semi-axis B (m)", 1),
    ("fill_c_m", "C", "Semi-axis C (m)", 1),

    # Row 2
    ("fill_z_m", "z", "z (m)", 1),
    ("fill_x_m", "x", "x (m)", 1),
    ("fill_y_m", "y", "y (m)", 1),

    # Row 3
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
# 9. CREATE 3 x 3 FIGURE
# =============================================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(10, 8),
    sharex=True,
)

# Convert to a simple one-dimensional list
axes = axes.flatten()


# =============================================================================
# 10. PLOT PARAMETERS
# =============================================================================

for ax, (column, title, ylabel, scale) in zip(
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

    ax.set_title(title)

    ax.set_ylabel(ylabel)

    # Light horizontal grid only
    ax.grid(
        axis="y",
        alpha=0.2,
        linewidth=0.6,
    )

    # Remove unnecessary top/right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# 11. DATE FORMATTING
# =============================================================================

for ax in axes[:8]:

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=2)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b\n%Y")
    )


# =============================================================================
# 12. NINTH PANEL = LEGEND
# =============================================================================

legend_ax = axes[8]

legend_ax.axis("off")


# Create legend using handles from first plot
handles, labels = axes[0].get_legend_handles_labels()

legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
    fontsize=9,
)


# Add short explanation
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
# 13. PANEL LABELS
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
# 14. FINAL FORMATTING
# =============================================================================

fig.suptitle(
    "Evolution of the five lowest-RMSE models",
    fontsize=11,
    y=0.99,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97],
    h_pad=1.3,
    w_pad=1.5,
)


# =============================================================================
# 15. SAVE
# =============================================================================

fig.savefig(
    OUTPUT_FIGURE,
    dpi=300,
)

fig.savefig(
    OUTPUT_PDF,
)


print(f"\nSaved PNG: {OUTPUT_FIGURE}")
print(f"Saved PDF: {OUTPUT_PDF}")


# =============================================================================
# 16. SHOW
# =============================================================================

plt.show()