#!/usr/bin/env python3
"""Analyse the 20 best volume-aware, acceptable-fit models per acquisition date."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path('/scratch/ee16eme/sinabung_asc_tsx/new_dense_for_each_date')
MODEL_DIR = Path('/scratch/ee16eme/sinabung_asc_tsx/mod_dem_Dome/synthetic_sweep_excavate6799_existing_fill')
DATES = ['20201021', '20201101', '20201226', '20210106',
         '20210117', '20210128', '20210208', '20210219']

N_BEST = 20
RMSE_THRESHOLD_M = 25.0  # Strictly less than 25 m
VOLUME_PENALTY_M_PER_MM3 = 3.0  # Score = RMSE (m) + lambda * fill volume (million m3)
WEIGHT_TEMPERATURE_FRACTION = 0.5

OUTPUT_FIGURE = BASE_DIR / 'top20_volume_weighted_model_evolution.png'
OUTPUT_PDF = BASE_DIR / 'top20_volume_weighted_model_evolution.pdf'
OUTPUT_CSV = BASE_DIR / 'top20_volume_weighted_model_evolution.csv'
OUTPUT_SUMMARY_CSV = BASE_DIR / 'top20_volume_weighted_model_summary.csv'
OUTPUT_ELIGIBLE_CSV = BASE_DIR / 'all_eligible_models_rmse_below_20.csv'

mpl.rcParams.update({
    'figure.figsize': (10, 8), 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 10,
    'savefig.dpi': 300, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def geometry(shape):
    sweep = shape.get('sweep_parameters')
    if sweep is not None:
        xyz = [float(sweep[k]) for k in ('x_m', 'y_m', 'z_m')]
        axes = [float(v) for v in sweep['semi_axes_m']]
    else:
        xyz = [float(v) for v in shape['center_xyz_m']]
        axes = [float(v) for v in shape['semi_axes_m']]
    return xyz + axes


def weighted_quantile(values, weights, probs=(0.05, 0.25, 0.75, 0.95)):
    """Weighted inverse-CDF quantiles; not formal confidence limits."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.full(len(probs), np.nan)
    v, w = values[mask], weights[mask]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return v[np.searchsorted(cdf, probs, side='left').clip(max=len(v) - 1)]


def score_weights(scores, fraction=WEIGHT_TEMPERATURE_FRACTION):
    """Heuristic weights based on differences in combined RMSE-volume score."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or len(scores) == 0:
        raise ValueError('Weights require nonempty, finite scores')
    excess = scores - np.min(scores)
    spread = np.max(excess)
    if spread <= 1e-12:
        return np.full(len(scores), 1.0 / len(scores))
    tau = max(fraction * spread, 1e-12)
    log_weights = -0.5 * (excess / tau) ** 2
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / weights.sum()


# Read model metadata
rows = []
json_files = sorted(MODEL_DIR.glob('*.json'))
print(f'Found {len(json_files)} JSON files')
for json_file in json_files:
    with json_file.open() as f:
        metadata = json.load(f)
    excavation = fill = None
    for shape in metadata.get('shapes', []):
        if shape.get('role') == 'excavation' or shape.get('interaction') == 'excavate_to_lower':
            excavation = shape
        elif shape.get('role') == 'fill' or shape.get('interaction') == 'fill_to_upper':
            fill = shape
    if excavation is None or fill is None:
        print(f'Skipping {json_file.name}: missing excavation or fill')
        continue
    ex, fi = geometry(excavation), geometry(fill)
    # The fill operation's added volume is relative to the *excavated* DEM.
    # Never use combined_final_diagnostics for the ranking or volume plot.
    fill_added = float(fill['volume_diagnostics']['added_volume_m3'])
    row = {
        'run_id': str(metadata.get('id', json_file.stem)).zfill(6),
        'source_excavation_run_id': excavation.get('source_excavation_run_id'),
        'source_fill_run_id': fill.get('source_fill_run_id'),
        'excavation_removed_volume_m3': float(excavation['volume_diagnostics']['removed_volume_m3']),
        'fill_added_volume_m3': fill_added,
    }
    for prefix, vals in (('excavation', ex), ('fill', fi)):
        row.update({f'{prefix}_{name}_m': value
                    for name, value in zip(('x', 'y', 'z', 'a', 'b', 'c'), vals)})
    rows.append(row)

models = pd.DataFrame(rows)
if models.empty:
    raise RuntimeError(f'No usable excavation + fill models in {MODEL_DIR}')
if models['run_id'].duplicated().any():
    raise RuntimeError('Duplicate run IDs in model JSON metadata')
models['run_number'] = models['run_id'].astype(int)
models = models.sort_values('run_number').reset_index(drop=True)
models['fill_added_volume_Mm3'] = models['fill_added_volume_m3'] / 1e6
print(f'Models loaded: {len(models)}')
print('Unique excavation geometries:')
print(models[[f'excavation_{name}_m' for name in ('x', 'y', 'z', 'a', 'b', 'c')]]
      .drop_duplicates().to_string(index=False))

plot_parameters = [
    ('fill_a_m', 'A', 'Semi-axis A (m)', 1),
    ('fill_b_m', 'B', 'Semi-axis B (m)', 1),
    ('fill_c_m', 'C', 'Semi-axis C (m)', 1),
    ('fill_z_m', 'z', 'z (m)', 1),
    ('fill_x_m', 'x', 'x (m)', 1),
    ('fill_y_m', 'y', 'y (m)', 1),
    ('fill_added_volume_m3', 'Fill added to excavated crater', 'Volume ($10^6$ m$^3$)', 1e6),
    ('rmse_filtered_m', 'Model fit', 'RMSE (m)', 1),
]

# Select and weight models for each date
all_results = []
all_eligible = []
for date_string in DATES:
    ranking_file = BASE_DIR / f'peak_model_inversion_dense_{date_string}' / 'peak_model_ranking.csv'
    if not ranking_file.exists():
        raise FileNotFoundError(f'Ranking file not found: {ranking_file}')
    ranking = pd.read_csv(ranking_file, dtype={'run_id': str})
    ranking['run_id'] = ranking['run_id'].str.strip().str.zfill(6)
    ranking['rmse_filtered_m'] = pd.to_numeric(ranking['rmse_filtered_m'], errors='coerce')
    ranking = ranking.loc[
        ranking['status'].eq('ok')
        & np.isfinite(ranking['rmse_filtered_m'])
        & (ranking['rmse_filtered_m'] < RMSE_THRESHOLD_M)
    ].copy()
    # Retain only the lowest RMSE entry per run ID, if repeated.
    ranking = (ranking.sort_values('rmse_filtered_m')
               .drop_duplicates(subset='run_id', keep='first'))
    n_before_merge = len(ranking)
    ranking = ranking.merge(models, on='run_id', how='inner', validate='many_to_one')
    if len(ranking) != n_before_merge:
        print(f'WARNING: {date_string}: {n_before_merge - len(ranking)} eligible models lack JSON metadata')
    ranking = ranking.loc[
        np.isfinite(ranking['fill_added_volume_Mm3'])
        & (ranking['fill_added_volume_Mm3'] >= 0)
    ].copy()
    ranking['score'] = (
        ranking['rmse_filtered_m']
        + VOLUME_PENALTY_M_PER_MM3 * ranking['fill_added_volume_Mm3']
    )
    ranking = ranking.sort_values(
        ['score', 'rmse_filtered_m', 'fill_added_volume_Mm3', 'run_id']
    ).reset_index(drop=True)
    ranking['date'] = pd.to_datetime(date_string, format='%Y%m%d')
    all_eligible.append(ranking.copy())
    selected = ranking.head(N_BEST).copy()
    if selected.empty:
        print(f'WARNING: {date_string}: no eligible models; skipping')
        continue
    if len(selected) < N_BEST:
        print(f'WARNING: {date_string}: only {len(selected)} eligible models')
    selected['rank'] = np.arange(1, len(selected) + 1)
    selected['weight'] = score_weights(selected['score'].to_numpy())
    all_results.append(selected)
    print(f'{date_string}: {len(ranking)} eligible, {len(selected)} selected; '
          f'best={selected.iloc[0]["run_id"]}, '
          f'RMSE={selected.iloc[0]["rmse_filtered_m"]:.2f} m, '
          f'fill={selected.iloc[0]["fill_added_volume_Mm3"]:.3f} Mm3')

if not all_results:
    raise RuntimeError('No eligible models matched ranking CSVs and JSON metadata')
BASE_DIR.mkdir(parents=True, exist_ok=True)
top20 = pd.concat(all_results, ignore_index=True).sort_values(['date', 'rank'])
top20.to_csv(OUTPUT_CSV, index=False)
if all_eligible:
    pd.concat(all_eligible, ignore_index=True).to_csv(OUTPUT_ELIGIBLE_CSV, index=False)

# Weighted summaries, retaining original quantile and effective-N methodology
summaries = []
for date, group in top20.groupby('date', sort=True):
    weights = group['weight'].to_numpy(float)
    row = {
        'date': date,
        'n_models': len(group),
        'effective_n': 1.0 / np.sum(weights ** 2),
        'best_run_id': group.iloc[0]['run_id'],
        'minimum_rmse_m': group['rmse_filtered_m'].min(),
        'best_score': group['score'].min(),
    }
    for column, _, _, _ in plot_parameters:
        values = pd.to_numeric(group[column], errors='coerce').to_numpy(float)
        valid = np.isfinite(values)
        row[f'{column}_best'] = values[0]
        row[f'{column}_mean'] = (
            np.average(values[valid], weights=weights[valid]) if valid.any() else np.nan
        )
        q05, q25, q75, q95 = weighted_quantile(values, weights)
        row.update({f'{column}_{p}': v for p, v in
                    zip(('q05', 'q25', 'q75', 'q95'), (q05, q25, q75, q95))})
    summaries.append(row)

summary = pd.DataFrame(summaries).sort_values('date')
summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)

# Plot original weighted-ensemble visualization with updated score/volume
fig, axes = plt.subplots(3, 3, figsize=(12, 8.7), sharex=True)
axes = axes.ravel()
dates = pd.to_datetime(summary['date'])
for idx, (ax, (column, title, ylabel, scale)) in enumerate(zip(axes, plot_parameters)):
    def series(suffix):
        return summary[f'{column}_{suffix}'].to_numpy(float) / scale

    ax.fill_between(dates, series('q05'), series('q95'), color='C0', alpha=0.14,
                    label='Weighted 5–95%', linewidth=0)
    ax.fill_between(dates, series('q25'), series('q75'), color='C0', alpha=0.30,
                    label='Weighted 25–75%', linewidth=0)
    ax.scatter(top20['date'], top20[column] / scale, s=9, color='0.35',
               alpha=0.17, linewidths=0, label='Selected 20 models', zorder=2)
    ax.plot(dates, series('mean'), 'o-', color='C0', lw=1.7, ms=4.2,
            label='Score-weighted estimate', zorder=4)
    ax.plot(dates, series('best'), '--', color='C3', lw=0.9,
            alpha=0.7, label='Best score model', zorder=3)
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis='y', alpha=0.2, lw=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.02, 0.96, f'({chr(97 + idx)})', transform=ax.transAxes,
            ha='left', va='top', fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

axes[8].axis('off')
handles, labels = axes[0].get_legend_handles_labels()
axes[8].legend(handles, labels, loc='upper center', frameon=False, fontsize=9)
axes[8].text(
    0.5, 0.30,
    f'Up to {N_BEST} models per date; RMSE < {RMSE_THRESHOLD_M:g} m\n'
    f'Score = RMSE + {VOLUME_PENALTY_M_PER_MM3:g} × fill volume (Mm³)\n'
    'Fill volume measured relative to excavated crater\n'
    'Shaded bands: score-weighted quantiles\n'
    'Bands show ensemble spread, not formal confidence limits',
    ha='center', va='center', transform=axes[8].transAxes, fontsize=8,
)
fig.suptitle('Evolution of volume-and-misfit-weighted top-20 inversion models',
             y=0.995, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.3, w_pad=1.3)
fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_PDF, bbox_inches='tight')
print(f'Saved: {OUTPUT_FIGURE}\nSaved: {OUTPUT_PDF}\n'
      f'Saved: {OUTPUT_CSV}\nSaved: {OUTPUT_SUMMARY_CSV}\n'
      f'Saved: {OUTPUT_ELIGIBLE_CSV}')
plt.show()
