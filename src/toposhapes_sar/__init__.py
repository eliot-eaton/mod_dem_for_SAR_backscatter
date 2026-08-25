"""Synthetic DEM modification for SAR/InSAR experiments."""

from .shapes import RotatedEllipsoid, RotatedCuboid
from .modify import apply_shape, apply_specs

__all__ = ["RotatedEllipsoid", "RotatedCuboid", "apply_shape", "apply_specs"]

try:
    from .gamma_dem import GammaDEMMetadata, read_gamma_dem, write_gamma_dem
    from .grid import project_dem_nearest, transfer_displacement_to_original_grid
    from .provenance import write_run_json
    from .qa import save_modified_dem_plot, save_difference_plot, save_qa_plots
except ModuleNotFoundError as exc:
    if exc.name != "rioxarray":
        raise
else:
    __all__ += [
        "GammaDEMMetadata", "read_gamma_dem", "write_gamma_dem",
        "project_dem_nearest", "transfer_displacement_to_original_grid",
        "write_run_json",
        "save_modified_dem_plot", "save_difference_plot", "save_qa_plots",
    ]
