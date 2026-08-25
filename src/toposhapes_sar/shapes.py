from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np


def rotation_matrix(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0) -> np.ndarray:
    yaw, pitch, roll = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    rz = np.array([[cy,-sy,0.0],[sy,cy,0.0],[0.0,0.0,1.0]])
    ry = np.array([[cp,0.0,sp],[0.0,1.0,0.0],[-sp,0.0,cp]])
    rx = np.array([[1.0,0.0,0.0],[0.0,cr,-sr],[0.0,sr,cr]])
    return rz @ ry @ rx


class VerticalSolid(Protocol):
    def bounds_xy(self) -> tuple[float,float,float,float]: ...
    def vertical_interval(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]: ...
    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class RotatedEllipsoid:
    """A true 3-D ellipsoid with fixed centre (x,y,z), all distances in metres."""
    center: tuple[float,float,float]
    semi_axes: tuple[float,float,float]
    rotation_deg: tuple[float,float,float] = (0.0,0.0,0.0)

    def __post_init__(self):
        if len(self.center) != 3 or not np.all(np.isfinite(self.center)):
            raise ValueError("center must be finite (x,y,z)")
        if len(self.semi_axes) != 3 or any(float(v) <= 0 for v in self.semi_axes):
            raise ValueError("semi_axes must contain three positive values")
        if len(self.rotation_deg) != 3 or not np.all(np.isfinite(self.rotation_deg)):
            raise ValueError("rotation_deg must contain three finite values")

    @property
    def rotation(self):
        return rotation_matrix(*self.rotation_deg)

    def bounds_xy(self):
        cx, cy, _ = map(float, self.center)
        axes = np.asarray(self.semi_axes, dtype=float)
        r = self.rotation
        hx = float(np.sqrt(np.sum((r[0,:]*axes)**2)))
        hy = float(np.sqrt(np.sum((r[1,:]*axes)**2)))
        return cx-hx, cx+hx, cy-hy, cy+hy

    def vertical_interval(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x,float), np.asarray(y,float))
        cx, cy, cz = map(float, self.center)
        dx, dy = x-cx, y-cy
        rt = self.rotation.T
        axes = np.asarray(self.semi_axes,float)
        inv = 1.0/(axes**2)
        q0 = [rt[i,0]*dx + rt[i,1]*dy for i in range(3)]
        vertical = rt[:,2]
        A = float(np.sum(vertical**2 * inv))
        B = 2.0*sum(q0[i]*vertical[i]*inv[i] for i in range(3))
        C = sum(q0[i]**2*inv[i] for i in range(3)) - 1.0
        disc = B**2 - 4*A*C
        valid = disc >= 0.0
        lower = np.full(x.shape, np.nan, dtype=float)
        upper = np.full(x.shape, np.nan, dtype=float)
        root = np.sqrt(np.maximum(disc[valid], 0.0))
        lower[valid] = cz + (-B[valid]-root)/(2*A)
        upper[valid] = cz + (-B[valid]+root)/(2*A)
        return lower, upper, valid

    def to_dict(self):
        return {"type":"ellipsoid", "center_xyz_m":list(map(float,self.center)), "semi_axes_m":list(map(float,self.semi_axes)), "rotation_deg":list(map(float,self.rotation_deg))}


@dataclass(frozen=True)
class RotatedCuboid:
    """A true 3-D cuboid with fixed centre (x,y,z), all distances in metres."""
    center: tuple[float,float,float]
    size: tuple[float,float,float]
    rotation_deg: tuple[float,float,float] = (0.0,0.0,0.0)

    def __post_init__(self):
        if len(self.center) != 3 or not np.all(np.isfinite(self.center)):
            raise ValueError("center must be finite (x,y,z)")
        if len(self.size) != 3 or any(float(v) <= 0 for v in self.size):
            raise ValueError("size must contain three positive values")

    @property
    def rotation(self): return rotation_matrix(*self.rotation_deg)
    @property
    def half_sizes(self): return np.asarray(self.size,float)/2.0

    def bounds_xy(self):
        cx, cy, _ = map(float,self.center); h=self.half_sizes; r=self.rotation
        hx=float(np.sum(np.abs(r[0,:])*h)); hy=float(np.sum(np.abs(r[1,:])*h))
        return cx-hx,cx+hx,cy-hy,cy+hy

    def vertical_interval(self, x, y):
        x,y=np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
        cx,cy,cz=map(float,self.center); dx=x-cx; dy=y-cy
        rt=self.rotation.T; h=self.half_sizes
        lower=np.full(x.shape,-np.inf); upper=np.full(x.shape,np.inf); valid=np.ones(x.shape,bool)
        eps=1e-12
        for axis in range(3):
            q0=rt[axis,0]*dx+rt[axis,1]*dy; v=float(rt[axis,2])
            if abs(v)<eps:
                valid &= np.abs(q0)<=h[axis]; continue
            z1=(-h[axis]-q0)/v; z2=(h[axis]-q0)/v
            lower=np.maximum(lower,np.minimum(z1,z2)); upper=np.minimum(upper,np.maximum(z1,z2))
        valid &= upper>=lower
        lo=np.full(x.shape,np.nan); hi=np.full(x.shape,np.nan)
        lo[valid]=cz+lower[valid]; hi[valid]=cz+upper[valid]
        return lo,hi,valid

    def to_dict(self):
        return {"type":"cuboid", "center_xyz_m":list(map(float,self.center)), "size_m":list(map(float,self.size)), "rotation_deg":list(map(float,self.rotation_deg))}
