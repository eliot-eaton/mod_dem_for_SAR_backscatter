
#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =============================================================================
# PLOT STYLE
# =============================================================================

mpl.rcParams.update({
    "figure.figsize": (10.0, 8.0),
    "font.family": "DejaVu Sans",
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
    "mod_dem_Dome/synthetic_sweep_excavate6799_existing_fill"
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

# Maximum number of selected models per date
N_BEST = 10

# Only consider models with RMSE strictly below this value
RMSE_THRESHOLD = 25.0

# Volume penalty:
# Score = RMSE (m) + LAMBDA_VOLUME * fill volume (million m3)
#
# 0.0 = rank only by RMSE
# 0.5 = weak preference for smaller fill volumes
# 1.0 = initial volume-aware ranking
# 2.0 = stronger preference for smaller fill volumes
#
# This is a ranking preference, not a physical conversion.
LAMBDA_VOLUME = 1.0

OUTPUT_FIGURE = BASE_DIR / "top10_volume_prioritised_evolution.png"
OUTPUT_PDF = BASE_DIR / "top10_volume_prioritised_evolution.pdf"
OUTPUT_CSV = BASE_DIR / "top10_volume_prioritised_evolution.csv"

# Additional CSV containing all acceptable models, before top-20 selection
OUTPUT_ALL_ACCEPTABLE_CSV = (
    BASE_DIR / "all_models_rmse_below_20.csv"
)


# =============================================================================
# HELPER: READ GEOMETRY FROM EITHER JSON FORMAT
# =============================================================================

def get_geometry(shape):
    """
    Return x, y, z, A, B, C for an ellipsoid.

    Supports:
      - shape["sweep_parameters"]
      - shape["center_xyz_m"] and shape["semi_axes_m"]
    """

    params = shape.get("sweep_parameters")

    if params is not None:
        x = float(params["x_m"])
        y = float(params["y_m"])
        z = float(params["z_m"])

        a, b, c = [
            float(v) for v in params["semi_axes_m"]
        ]

    else:
        x, y, z = [
            float(v) for v in shape["center_xyz_m"]
        ]

        a, b, c = [
            float(v) for v in shape["semi_axes_m"]
        ]

    return x, y, z, a, b, c


# =============================================================================
# 1. READ MODEL JSON METADATA
# =============================================================================

rows = []

json_files = sorted(MODEL_DIR.glob("*.json"))

print(f"Found {len(json_files)} JSON files")

if not json_files:
    raise FileNotFoundError(
        f"No model JSON files found in:\n{MODEL_DIR}"
    )


for json_file in json_files:

    with open(json_file) as f:
        metadata = json.load(f)

    run_id = str(
        metadata.get("id", json_file.stem)
    ).zfill(6)

    excavation = None
    fill = None

    # Find excavation and fill shapes
    for shape in metadata.get("shapes", []):

        role = shape.get("role")
        interaction = shape.get("interaction")

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
            "missing excavation or fill"
        )
        continue

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------

    (
        exc_x, exc_y, exc_z,
        exc_a, exc_b, exc_c
    ) = get_geometry(excavation)

    (
        fill_x, fill_y, fill_z,
        fill_a, fill_b, fill_c
    ) = get_geometry(fill)

    # -------------------------------------------------------------------------
    # Volume diagnostics
    # -------------------------------------------------------------------------

    exc_volume = excavation["volume_diagnostics"]
    fill_volume = fill["volume_diagnostics"]

    # IMPORTANT:
    #
    # Use the volume added by the fill_to_upper operation.
    #
    # This is measured relative to the excavated crater,
    # NOT relative to the original unmodified DEM.
    #
    # Do not use combined_final_diagnostics["added_volume_m3"]
    # for selection or ranking.

    fill_added_volume_m3 = float(
        fill_volume["added_volume_m3"]
    )

    # Store metadata
    rows.append({
        "run_id": run_id,

        "source_excavation_run_id":
            excavation.get("source_excavation_run_id"),

        "source_fill_run_id":
            fill.get("source_fill_run_id"),

        # Excavation geometry
        "excavation_x_m": exc_x,
        "excavation_y_m": exc_y,
        "excavation_z_m": exc_z,
        "excavation_a_m": exc_a,
        "excavation_b_m": exc_b,
        "excavation_c_m": exc_c,

        # Fill geometry
        "fill_x_m": fill_x,
        "fill_y_m": fill_y,
        "fill_z_m": fill_z,
        "fill_a_m": fill_a,
        "fill_b_m": fill_b,
        "fill_c_m": fill_c,

        # Excavation volume (informational only)
        "excavation_removed_volume_m3": float(
            exc_volume["removed_volume_m3"]
        ),

        # The ONLY volume used for model selection
        "fill_added_volume_m3": fill_added_volume_m3,
    })


# =============================================================================
# 2. CREATE MODEL METADATA DATAFRAME
# =============================================================================

models = pd.DataFrame(rows)

if models.empty:
    raise RuntimeError(
        f"No usable excavation + fill models found in:\n{MODEL_DIR}"
    )

if models["run_id"].duplicated().any():
    raise RuntimeError("Duplicate run IDs found in model metadata.")

models["run_number"] = models["run_id"].astype(int)

models = (
    models
    .sort_values("run_number")
    .reset_index(drop=True)
)

models["fill_added_volume_Mm3"] = (
    models["fill_added_volume_m3"] / 1e6
)

print(f"Models loaded: {len(models)}")

print("\nUnique excavation geometries:")

print(
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
    .to_string(index=False)
)


# =============================================================================
# 3. READ AND RANK INVERSION RESULTS FOR EACH DATE
# =============================================================================

all_results = []
all_acceptable_results = []

for date_string in DATES:

    inversion_dir = (
        BASE_DIR
        / f"peak_model_inversion_dense_{date_string}"
    )

    ranking_file = (
        inversion_dir
        / "peak_model_ranking.csv"
    )

    print(f"\nReading {ranking_file}")

    if not ranking_file.exists():
        raise FileNotFoundError(
            f"Ranking file not found:\n{ranking_file}"
        )

    ranking = pd.read_csv(
        ranking_file,
        dtype={"run_id": str},
    )

    ranking["run_id"] = (
        ranking["run_id"]
        .str.strip()
        .str.zfill(6)
    )

    ranking["rmse_filtered_m"] = pd.to_numeric(
        ranking["rmse_filtered_m"],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Filter by acceptable RMSE
    # -------------------------------------------------------------------------

    ranking = ranking[
        (ranking["status"] == "ok")
        & np.isfinite(ranking["rmse_filtered_m"])
        & (ranking["rmse_filtered_m"] < RMSE_THRESHOLD)
    ].copy()

    n_acceptable_before_merge = len(ranking)

    # -------------------------------------------------------------------------
    # Join model metadata BEFORE volume-based ranking
    # -------------------------------------------------------------------------

    ranking = ranking.merge(
        models,
        on="run_id",
        how="inner",
        validate="many_to_one",
    )

    n_missing_metadata = (
        n_acceptable_before_merge - len(ranking)
    )

    if n_missing_metadata:
        print(
            f"  WARNING: {n_missing_metadata} acceptable "
            "ranking rows have no matching model JSON."
        )

    # -------------------------------------------------------------------------
    # Exclude missing or invalid fill volumes
    # -------------------------------------------------------------------------

    ranking = ranking[
        np.isfinite(ranking["fill_added_volume_Mm3"])
        & (ranking["fill_added_volume_Mm3"] >= 0)
    ].copy()

    # -------------------------------------------------------------------------
    # Calculate volume-aware score
    #
    # Lower is better.
    # -------------------------------------------------------------------------

    ranking["score"] = (
        ranking["rmse_filtered_m"]
        + LAMBDA_VOLUME * ranking["fill_added_volume_Mm3"]
    )

    # Sort using volume-aware score
    ranking = ranking.sort_values(
        [
            "score",
            "rmse_filtered_m",
            "fill_added_volume_Mm3",
            "run_id",
        ]
    ).reset_index(drop=True)

    # Add acquisition date
    ranking["date"] = pd.to_datetime(
        date_string,
        format="%Y%m%d",
    )

    # Save all acceptable models before top-20 selection
    all_acceptable_results.append(ranking.copy())

    # -------------------------------------------------------------------------
    # Select 20 preferred models
    # -------------------------------------------------------------------------

    selected = ranking.head(N_BEST).copy()

    selected["rank"] = range(
        1,
        len(selected) + 1,
    )

    all_results.append(selected)

    print(
        f"  Models with RMSE < {RMSE_THRESHOLD}: "
        f"{n_acceptable_before_merge}"
    )

    print(
        f"  Models with usable metadata: {len(ranking)}"
    )

    print(
        f"  Selected: {len(selected)}"
    )

    if not selected.empty:
        print(
            "  Best volume-aware model: "
            f"{selected.iloc[0]['run_id']}"
        )

        print(
            "  RMSE: "
            f"{selected.iloc[0]['rmse_filtered_m']:.3f} m"
        )

        print(
            "  Fill volume: "
            f"{selected.iloc[0]['fill_added_volume_Mm3']:.3f} "
            "million m3"
        )


# =============================================================================
# 4. COMBINE DATES
# =============================================================================

top20 = pd.concat(
    all_results,
    ignore_index=True,
)

all_acceptable = pd.concat(
    all_acceptable_results,
    ignore_index=True,
)

if top20.empty:
    raise RuntimeError(
        "No models satisfy the RMSE threshold "
        "and have usable fill-volume metadata."
    )

top20 = (
    top20
    .sort_values(["date", "rank"])
    .reset_index(drop=True)
)

# Rank 1 is now the lowest volume-penalised SCORE,
# not necessarily the lowest raw RMSE.
best = (
    top20[top20["rank"] == 1]
    .sort_values("date")
    .copy()
)


# =============================================================================
# 5. PRINT RESULTS
# =============================================================================

print("\n" + "=" * 120)
print("20 VOLUME-PRIORITISED MODELS FOR EACH DATE")
print("=" * 120)

print(
    top20[
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
            "fill_added_volume_Mm3",
            "rmse_filtered_m",
            "score",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 120)
print("BEST VOLUME-ADJUSTED MODEL FOR EACH DATE")
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
            "fill_added_volume_Mm3",
            "rmse_filtered_m",
            "score",
        ]
    ].to_string(index=False)
)


# =============================================================================
# 6. SAVE RESULTS TABLES
# =============================================================================

BASE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

top20.to_csv(
    OUTPUT_CSV,
    index=False,
)

all_acceptable.to_csv(
    OUTPUT_ALL_ACCEPTABLE_CSV,
    index=False,
)

print(f"\nSaved top-20 table: {OUTPUT_CSV}")

print(
    "Saved all acceptable models: "
    f"{OUTPUT_ALL_ACCEPTABLE_CSV}"
)


# =============================================================================
# 7. PARAMETERS TO PLOT
# =============================================================================

plot_parameters = [

    ("fill_a_m", "A", "Semi-axis A (m)", 1),
    ("fill_b_m", "B", "Semi-axis B (m)", 1),
    ("fill_c_m", "C", "Semi-axis C (m)", 1),

    ("fill_z_m", "z", "z (m)", 1),
    ("fill_x_m", "x", "x (m)", 1),
    ("fill_y_m", "y", "y (m)", 1),

    (
        "fill_added_volume_m3",
        "Fill volume",
        "Added volume ($10^6$ m$^3$)",
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
# 8. CREATE 3 x 3 FIGURE
# =============================================================================

fig, axes = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(10, 8),
    sharex=True,
)

axes = axes.flatten()


# =============================================================================
# 9. PLOT PARAMETERS
# =============================================================================

for ax, (column, title, ylabel, scale) in zip(
    axes,
    plot_parameters,
):

    # All 20 selected models at each date
    ax.scatter(
        top20["date"],
        top20[column] / scale,
        s=24,
        alpha=0.35,
        label="20 volume-prioritised models",
        zorder=2,
    )

    # Best volume-adjusted model
    ax.plot(
        best["date"],
        best[column] / scale,
        marker="o",
        markersize=4.5,
        linewidth=1.3,
        label="Best volume-adjusted score",
        zorder=3,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ax.grid(
        axis="y",
        alpha=0.2,
        linewidth=0.6,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# 10. DATE FORMATTING
# =============================================================================

for ax in axes[:8]:

    ax.xaxis.set_major_locator(
        mdates.MonthLocator(interval=2)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%b\n%Y")
    )


# =============================================================================
# 11. NINTH PANEL = LEGEND
# =============================================================================

legend_ax = axes[8]
legend_ax.axis("off")

handles, labels = axes[0].get_legend_handles_labels()

legend_ax.legend(
    handles,
    labels,
    loc="center",
    frameon=False,
    fontsize=9,
)

legend_ax.text(
    0.5,
    0.26,
    f"Only models with RMSE < {RMSE_THRESHOLD:g} m.\n"
    f"Up to {N_BEST} models per date.\n\n"
    "Score = RMSE + "
    f"{LAMBDA_VOLUME:g} × fill volume\n"
    "(million cubic metres).\n\n"
    "Volume is measured relative\n"
    "to the excavated crater.",
    ha="center",
    va="center",
    transform=legend_ax.transAxes,
    fontsize=8,
)


# =============================================================================
# 12. PANEL LABELS
# =============================================================================

panel_labels = [
    "(a)", "(b)", "(c)",
    "(d)", "(e)", "(f)",
    "(g)", "(h)",
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
# 13. FINAL FORMATTING
# =============================================================================

fig.suptitle(
    "Evolution of volume-prioritised models (RMSE < 20 m)",
    fontsize=11,
    y=0.99,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.97],
    h_pad=1.3,
    w_pad=1.5,
)


# =============================================================================
# 14. SAVE FIGURES
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
# 15. SHOW
# =============================================================================

plt.show()
