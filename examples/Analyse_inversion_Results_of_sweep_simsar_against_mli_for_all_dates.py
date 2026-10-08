#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path("/scratch/ee16eme/sinabung_asc_tsx/new_dense_for_each_date")
MODEL_DIR = Path("/scratch/ee16eme/sinabung_asc_tsx/mod_dem_Dome/synthetic_sweep_excavate6799_existing_fill")
DATES = ["20201021", "20201101", "20201226", "20210106", "20210117", "20210128", "20210208", "20210219"]
mpl.rcParams.update({"figure.figsize": (10, 8), "font.size": 9,
                     "axes.labelsize": 9, "axes.titlesize": 10,
                     "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42})

def geometry(shape):
    sweep = shape.get("sweep_parameters")
    if sweep is not None:
        xyz = [float(sweep[k]) for k in ("x_m", "y_m", "z_m")]
        axes = [float(v) for v in sweep["semi_axes_m"]]
    else:
        xyz = [float(v) for v in shape["center_xyz_m"]]
        axes = [float(v) for v in shape["semi_axes_m"]]
    return xyz + axes

rows = []
for json_file in sorted(MODEL_DIR.glob("*.json")):
    with json_file.open() as f:
        metadata = json.load(f)
    excavation = None
    fill = None
    for shape in metadata.get("shapes", []):
        if shape.get("role") == "excavation" or shape.get("interaction") == "excavate_to_lower":
            excavation = shape
        elif shape.get("role") == "fill" or shape.get("interaction") == "fill_to_upper":
            fill = shape
    if excavation is None or fill is None:
        print(f"Skipping {json_file.name}: missing excavation or fill")
        continue
    ex = geometry(excavation)
    fi = geometry(fill)
    final = fill["combined_final_diagnostics"]
    row = {"run_id": str(metadata.get("id", json_file.stem)).zfill(6),
           "source_excavation_run_id": excavation.get("source_excavation_run_id"),
           "source_fill_run_id": fill.get("source_fill_run_id"),
           "excavation_removed_volume_m3": float(excavation["volume_diagnostics"]["removed_volume_m3"]),
           "fill_added_volume_m3": float(fill["volume_diagnostics"]["added_volume_m3"]),
           "final_added_volume_m3": float(final["added_volume_m3"]),
           "final_removed_volume_m3": float(final["removed_volume_m3"]),
           "final_net_volume_m3": float(final["net_volume_change_m3"])}
    for prefix, vals in (("excavation", ex), ("fill", fi)):
        row.update({f"{prefix}_{name}_m": value for name, value in zip(("x", "y", "z", "a", "b", "c"), vals)})
    rows.append(row)
models = pd.DataFrame(rows)
if models.empty:
    raise RuntimeError(f"No usable excavation + fill models were read from: {MODEL_DIR}")
models["run_number"] = models["run_id"].astype(int)
models = models.sort_values("run_number").reset_index(drop=True)
models["final_added_volume_Mm3"] = models["final_added_volume_m3"] / 1e6
print(f"Models loaded: {len(models)}")
print("Unique excavation geometries:")
print(models[[f"excavation_{name}_m" for name in ("x", "y", "z", "a", "b", "c")]].drop_duplicates().to_string(index=False))

import numpy as np

N_BEST = 20
# Larger values make the ensemble weights more nearly uniform.
# The spread in the best 20 RMSE values is the per-date temperature.
WEIGHT_TEMPERATURE_FRACTION = 0.5
OUTPUT_FIGURE = BASE_DIR / "top20_weighted_model_evolution.png"
OUTPUT_PDF = BASE_DIR / "top20_weighted_model_evolution.pdf"
OUTPUT_CSV = BASE_DIR / "top20_weighted_model_evolution.csv"
OUTPUT_SUMMARY_CSV = BASE_DIR / "top20_weighted_model_summary.csv"

plot_parameters = [
    ("fill_a_m", "A", "Semi-axis A (m)", 1),
    ("fill_b_m", "B", "Semi-axis B (m)", 1),
    ("fill_c_m", "C", "Semi-axis C (m)", 1),
    ("fill_z_m", "z", "z (m)", 1),
    ("fill_x_m", "x", "x (m)", 1),
    ("fill_y_m", "y", "y (m)", 1),
    ("final_added_volume_m3", "Final added volume", "Volume ($10^6$ m$^3$)", 1e6),
    ("rmse_filtered_m", "Model fit", "RMSE (m)", 1),
]


def weighted_quantile(values, weights, probs=(0.05, 0.25, 0.75, 0.95)):
    """Weighted inverse-CDF quantiles, robust to missing parameter values."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.full(len(probs), np.nan)
    v = values[mask]
    w = weights[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return v[np.searchsorted(cdf, probs, side="left").clip(max=len(v) - 1)]


def misfit_weights(rmse, fraction=WEIGHT_TEMPERATURE_FRACTION):
    """Relative-RMSE weights; avoids temperature depending on absolute RMSE."""
    rmse = np.asarray(rmse, dtype=float)
    if not np.isfinite(rmse).all():
        raise ValueError("Misfit weights require finite RMSE values")
    excess = rmse - np.min(rmse)
    spread = np.max(excess)
    if spread <= 1e-12:
        return np.full(len(rmse), 1.0 / len(rmse))
    tau = max(fraction * spread, 1e-12)
    scores = -0.5 * (excess / tau) ** 2
    scores -= np.max(scores)
    weights = np.exp(scores)
    return weights / weights.sum()


all_results = []
for date_string in DATES:
    ranking_file = (BASE_DIR / f"peak_model_inversion_dense_{date_string}"
                    / "peak_model_ranking.csv")
    if not ranking_file.exists():
        raise FileNotFoundError(f"Ranking file not found: {ranking_file}")
    ranking = pd.read_csv(ranking_file, dtype={"run_id": str})
    ranking["run_id"] = ranking["run_id"].str.strip().str.zfill(6)
    ranking["rmse_filtered_m"] = pd.to_numeric(
        ranking["rmse_filtered_m"], errors="coerce")
    ranking = ranking.loc[
        ranking["status"].eq("ok") & np.isfinite(ranking["rmse_filtered_m"])
    ].copy()
    # Avoid selecting duplicate model IDs for the same date.
    ranking = (ranking.sort_values("rmse_filtered_m")
               .drop_duplicates(subset="run_id", keep="first"))
    ranking = ranking.merge(models, on="run_id", how="inner", validate="many_to_one")
    if len(ranking) < N_BEST:
        print(f"WARNING: {date_string}: only {len(ranking)} valid ranked models with metadata")
    ranking = ranking.nsmallest(N_BEST, "rmse_filtered_m").copy()
    if ranking.empty:
        print(f"WARNING: {date_string}: no matched valid models; skipping")
        continue
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["date"] = pd.to_datetime(date_string, format="%Y%m%d")
    ranking["weight"] = misfit_weights(ranking["rmse_filtered_m"].to_numpy())
    all_results.append(ranking)

if not all_results:
    raise RuntimeError("No valid models with metadata could be matched to inversion rankings")

top20 = pd.concat(all_results, ignore_index=True).sort_values(["date", "rank"])
top20.to_csv(OUTPUT_CSV, index=False)

summaries = []
for date, group in top20.groupby("date", sort=True):
    row = {"date": date, "n_models": len(group),
           "effective_n": 1.0 / np.sum(group["weight"].to_numpy() ** 2),
           "best_run_id": group.iloc[0]["run_id"],
           "minimum_rmse_m": group["rmse_filtered_m"].min()}
    for column, _, _, _ in plot_parameters:
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
        weights = group["weight"].to_numpy(float)
        valid = np.isfinite(values)
        row[f"{column}_best"] = values[0]
        row[f"{column}_mean"] = (np.average(values[valid], weights=weights[valid])
                                    if valid.any() else np.nan)
        q05, q25, q75, q95 = weighted_quantile(values, weights)
        row.update({f"{column}_{p}": v for p, v in
                    zip(("q05", "q25", "q75", "q95"),
                        (q05, q25, q75, q95))})
    summaries.append(row)

summary = pd.DataFrame(summaries).sort_values("date")
summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

fig, axes = plt.subplots(3, 3, figsize=(12, 8.7), sharex=True)
axes = axes.ravel()
dates = pd.to_datetime(summary["date"])

for idx, (ax, (column, title, ylabel, scale)) in enumerate(
        zip(axes, plot_parameters)):
    series = lambda suffix: summary[f"{column}_{suffix}"].to_numpy(float) / scale
    ax.fill_between(dates, series("q05"), series("q95"), color="C0", alpha=0.14,
                    label="Weighted 5–95%", linewidth=0)
    ax.fill_between(dates, series("q25"), series("q75"), color="C0", alpha=0.30,
                    label="Weighted 25–75%", linewidth=0)
    ax.scatter(top20["date"], top20[column] / scale, s=9, color="0.35",
               alpha=0.17, linewidths=0, label="Top-20 models", zorder=2)
    ax.plot(dates, series("mean"), "o-", color="C0", lw=1.7, ms=4.2,
            label="Misfit-weighted estimate", zorder=4)
    ax.plot(dates, series("best"), "--", color="C3", lw=0.9,
            alpha=0.7, label="Single best model", zorder=3)
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="y", alpha=0.2, lw=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.96, f"({chr(97 + idx)})", transform=ax.transAxes,
            ha="left", va="top", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

axes[8].axis("off")
handles, labels = axes[0].get_legend_handles_labels()
axes[8].legend(handles, labels, loc="upper center", frameon=False, fontsize=9)
axes[8].text(0.5, 0.32,
             "20 lowest-RMSE models per date\n"
             "Shaded bands: misfit-weighted quantiles\n"
             "Bands show ensemble spread, not formal confidence limits",
             ha="center", va="center", transform=axes[8].transAxes, fontsize=8)
fig.suptitle("Evolution of misfit-weighted top-20 inversion models", y=0.995, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.3, w_pad=1.3)
fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
fig.savefig(OUTPUT_PDF, bbox_inches="tight")
print(f"Saved: {OUTPUT_FIGURE}\nSaved: {OUTPUT_PDF}\n"
      f"Saved: {OUTPUT_CSV}\nSaved: {OUTPUT_SUMMARY_CSV}")
plt.show()
