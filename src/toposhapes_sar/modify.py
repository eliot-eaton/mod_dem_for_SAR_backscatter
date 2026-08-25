from __future__ import annotations

from typing import Iterable, Literal
import numpy as np
import xarray as xr

from .shapes import VerticalSolid

Interaction = Literal[
    "fill_to_upper",
    "excavate_to_lower",
    "add_thickness",
    "subtract_thickness",
]


def _window(coords, lower, upper):
    idx=np.flatnonzero((coords>=min(lower,upper)) & (coords<=max(lower,upper)))
    return None if idx.size==0 else slice(int(idx[0]),int(idx[-1])+1)


def apply_shape(dem: xr.DataArray, shape: VerticalSolid, *, interaction: Interaction) -> tuple[xr.DataArray,xr.DataArray]:
    """Apply a solid to a projected metre-grid DEM and return (modified_dem, dz).

    Interaction semantics are deliberately explicit:
    - fill_to_upper: raise terrain to the solid's upper surface where upper > DEM.
    - excavate_to_lower: lower terrain to the solid's lower surface where lower < DEM.
      This creates a vertical excavation from the surface down to a buried solid's
      lower surface, even if the solid does not intersect the original DEM.
    - add_thickness / subtract_thickness: legacy thickness operations.
    """
    if interaction not in {"fill_to_upper","excavate_to_lower","add_thickness","subtract_thickness"}:
        raise ValueError(f"Unknown interaction {interaction!r}")
    if dem.dims != ("y","x"):
        raise ValueError("DEM dimensions must be exactly ('y','x')")

    out=dem.copy(deep=True)
    dz=xr.zeros_like(dem,dtype=np.float32); dz.name="dz"; dz.attrs["units"]="m"
    xmin,xmax,ymin,ymax=shape.bounds_xy()
    cs=_window(dem.x.values,xmin,xmax); rs=_window(dem.y.values,ymin,ymax)
    if cs is None or rs is None: return out,dz

    xx,yy=np.meshgrid(dem.x.values[cs],dem.y.values[rs])
    lower,upper,solid=shape.vertical_interval(xx,yy)
    terrain=np.asarray(dem.values[rs,cs],dtype=np.float32)
    finite=np.isfinite(terrain)
    new=terrain.copy()

    if interaction=="fill_to_upper":
        mask=solid & finite & (upper>terrain); new[mask]=upper[mask]
    elif interaction=="excavate_to_lower":
        mask=solid & finite & (lower<terrain); new[mask]=lower[mask]
    else:
        thickness=upper-lower
        mask=solid & finite
        sign=1.0 if interaction=="add_thickness" else -1.0
        new[mask]=terrain[mask]+sign*thickness[mask]

    local_dz=(new-terrain).astype(np.float32)
    out.values[rs,cs]=new.astype(out.dtype,copy=False)
    dz.values[rs,cs]=local_dz
    return out,dz


def apply_specs(dem: xr.DataArray, specs: Iterable[tuple[VerticalSolid,Interaction]]) -> tuple[xr.DataArray,xr.DataArray]:
    """Apply several shapes sequentially to one realization, returning cumulative dz."""
    out=dem.copy(deep=True); total=xr.zeros_like(dem,dtype=np.float32)
    for shape,interaction in specs:
        out,local=apply_shape(out,shape,interaction=interaction)
        total=total+local
    total.name="dz"; total.attrs["units"]="m"
    return out,total
