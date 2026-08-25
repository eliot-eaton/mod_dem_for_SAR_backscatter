from __future__ import annotations

import json
from pathlib import Path


def write_run_json(
    path,
    *,
    run_id,
    source_dem,
    source_dem_par,
    output_dem,
    dem_geo_original,
    shape_records,
    validation,
    projected_crs,
    qa_outputs=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "id": str(run_id),
        "source_dem": str(source_dem),
        "source_dem_par": str(source_dem_par),
        "output_dem": Path(output_dem).name,
        "binary_format": {
            "data_format": "REAL*4",
            "numpy_dtype": ">f4",
            "byte_order": "big-endian",
            "shape": list(dem_geo_original.shape),
        },
        "original_grid": {
            "crs": str(dem_geo_original.rio.crs),
            "resolution": [float(v) for v in dem_geo_original.rio.resolution()],
            "bounds": [float(v) for v in dem_geo_original.rio.bounds()],
            "transform": [
                float(v)
                for v in tuple(dem_geo_original.rio.transform())[:6]
            ],
        },
        "working_projected_crs": str(projected_crs),
        "shapes": shape_records,
        "validation": validation,
    }

    if qa_outputs:
        payload["qa_outputs"] = {
            key: Path(value).name
            for key, value in qa_outputs.items()
        }

    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
