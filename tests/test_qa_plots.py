from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import xarray as xr

from toposhapes_sar.qa import save_qa_plots


def test_visual_qa_outputs_are_created(tmp_path: Path):
    x = np.arange(0.0, 1000.0, 10.0)
    y = np.arange(1000.0, 0.0, -10.0)
    xx, yy = np.meshgrid(x, y)

    original = xr.DataArray(
        1000.0 + 0.01 * xx + 0.02 * yy,
        dims=("y", "x"),
        coords={"y": y, "x": x},
    )
    dz = xr.zeros_like(original, dtype=np.float32)
    dz.loc[dict(x=slice(450.0, 550.0), y=slice(550.0, 450.0))] = 25.0
    modified = original + dz

    print("\n[CHECK] Each realization saves two cropped visual QA PNGs.")
    print("        One checks modified topography + contours; the other checks dz directly.")

    outputs = save_qa_plots(
        modified,
        dz,
        tmp_path,
        "plot_test",
        padding_m=100.0,
    )

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
