from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _changed_crop(
    dem: xr.DataArray,
    dz: xr.DataArray,
    *,
    padding_m: float = 500.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Crop a projected DEM and displacement field around all changed pixels.

    Cropping is for visual QA only. The full-size DEM and displacement arrays are
    never modified. ``padding_m`` is added on every side of the changed footprint.
    """
    if dem.dims != ("y", "x") or dz.dims != ("y", "x"):
        raise ValueError("DEM and dz dimensions must be exactly ('y', 'x')")
    if dem.shape != dz.shape:
        raise ValueError("DEM and dz must have the same shape")
    if padding_m < 0:
        raise ValueError("padding_m must be non-negative")

    changed = np.isfinite(dz.values) & (dz.values != 0)
    if not np.any(changed):
        raise ValueError("No modified pixels found; cannot make a modification QA plot")

    rows, cols = np.where(changed)
    x_changed = dem.x.values[cols]
    y_changed = dem.y.values[rows]

    xmin = float(np.min(x_changed) - padding_m)
    xmax = float(np.max(x_changed) + padding_m)
    ymin = float(np.min(y_changed) - padding_m)
    ymax = float(np.max(y_changed) + padding_m)

    x_increasing = bool(dem.x.values[-1] > dem.x.values[0])
    y_increasing = bool(dem.y.values[-1] > dem.y.values[0])

    x_slice = slice(xmin, xmax) if x_increasing else slice(xmax, xmin)
    y_slice = slice(ymin, ymax) if y_increasing else slice(ymax, ymin)

    return dem.sel(x=x_slice, y=y_slice), dz.sel(x=x_slice, y=y_slice)


def save_modified_dem_plot(
    dem_m_modified: xr.DataArray,
    dz_m: xr.DataArray,
    output_path: str | Path,
    *,
    padding_m: float = 500.0,
    contour_levels: int = 40,
    dpi: int = 200,
) -> Path:
    """Save a cropped colour+contour QA plot of the modified projected DEM."""
    dem_plot, dz_plot = _changed_crop(dem_m_modified, dz_m, padding_m=padding_m)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    dem_plot.plot.pcolormesh(
        ax=ax,
        x="x",
        y="y",
        shading="auto",
        cbar_kwargs={"label": "Elevation (m)"},
    )

    dem_plot.plot.contour(
        ax=ax,
        x="x",
        y="y",
        levels=contour_levels,
        colors="k",
        linewidths=0.45,
    )

    # Outline only the intentionally changed footprint.
    changed_plot = xr.where(dz_plot != 0, 1.0, 0.0)
    changed_plot.plot.contour(
        ax=ax,
        x="x",
        y="y",
        levels=[0.5],
        colors=["red"],
        linewidths=1.5,
        add_colorbar=False,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("Projected x / easting (m)")
    ax.set_ylabel("Projected y / northing (m)")
    ax.set_title("Modified DEM — visual QA")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(
        "[CHECK] Saved modified-DEM QA plot.\n"
        "        Colour and black contours show the final projected topography.\n"
        "        Red outline shows pixels intentionally modified.\n"
        f"        Plot includes {padding_m:g} m of context on each side.\n"
        f"        Output: {output_path}"
    )
    return output_path


def save_difference_plot(
    dz_m: xr.DataArray,
    output_path: str | Path,
    *,
    padding_m: float = 500.0,
    contour_levels: int = 12,
    dpi: int = 200,
) -> Path:
    """Save a cropped QA plot of intentional elevation change (modified-original)."""
    # dz itself is the authoritative projected-grid difference.
    dz_plot, dz_plot_again = _changed_crop(dz_m, dz_m, padding_m=padding_m)
    del dz_plot_again

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    changed_values = dz_plot.values[np.isfinite(dz_plot.values) & (dz_plot.values != 0)]
    max_abs = float(np.max(np.abs(changed_values)))
    if max_abs == 0:
        raise ValueError("No non-zero displacement available for plotting")

    fig, ax = plt.subplots(figsize=(8, 6))

    dz_plot.plot.pcolormesh(
        ax=ax,
        x="x",
        y="y",
        shading="auto",
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        cbar_kwargs={"label": "Elevation change, modified − original (m)"},
    )

    # Contours only where there is a real modification; zero background is hidden.
    changed_only = dz_plot.where(dz_plot != 0)
    finite_changed = np.isfinite(changed_only.values)
    if np.count_nonzero(finite_changed) > 3:
        values = changed_only.values[finite_changed]
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmax > vmin:
            levels = np.linspace(vmin, vmax, contour_levels)
            changed_only.plot.contour(
                ax=ax,
                x="x",
                y="y",
                levels=levels,
                colors="k",
                linewidths=0.4,
            )

    footprint = xr.where(dz_plot != 0, 1.0, 0.0)
    footprint.plot.contour(
        ax=ax,
        x="x",
        y="y",
        levels=[0.5],
        colors="k",
        linewidths=1.2,
        add_colorbar=False,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("Projected x / easting (m)")
    ax.set_ylabel("Projected y / northing (m)")
    ax.set_title("DEM modification — modified − original")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(
        "[CHECK] Saved DEM-difference QA plot.\n"
        "        This plot shows only the intentional projected-grid displacement dz.\n"
        "        Positive values add topography; negative values excavate/subtract.\n"
        f"        Plot includes {padding_m:g} m of context on each side.\n"
        f"        Output: {output_path}"
    )
    return output_path


def save_qa_plots(
    dem_m_modified: xr.DataArray,
    dz_m: xr.DataArray,
    output_dir: str | Path,
    run_id: str,
    *,
    padding_m: float = 500.0,
) -> dict[str, Path]:
    """Create both standard visual-QA PNGs for one DEM realization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dem_path = output_dir / f"{run_id}.dem_check.png"
    difference_path = output_dir / f"{run_id}.difference_check.png"

    save_modified_dem_plot(
        dem_m_modified,
        dz_m,
        dem_path,
        padding_m=padding_m,
    )
    save_difference_plot(
        dz_m,
        difference_path,
        padding_m=padding_m,
    )

    return {
        "modified_dem_plot": dem_path,
        "difference_plot": difference_path,
    }
