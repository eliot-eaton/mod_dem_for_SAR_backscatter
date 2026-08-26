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
@dataclass(frozen=True)
class RotatedTrapezoidalPrism:
    """
    A true 3-D trapezoidal prism with fixed centre (x, y, z).

    The unrotated shape is:

        - trapezoidal in local x-z
        - constant width along local y

    Parameters
    ----------
    center
        Fixed world centre (x, y, z) in metres.

    bottom_length
        Full x-length at the lower face.

    top_length
        Full x-length at the upper face.

    width
        Full prism width in local y.

    height
        Full vertical height in local z.

    rotation_deg
        (yaw, pitch, roll) in degrees.
    """

    center: tuple[float, float, float]

    bottom_length: float
    top_length: float
    width: float
    height: float

    rotation_deg: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    def __post_init__(self):

        if (
            len(self.center) != 3
            or not np.all(np.isfinite(self.center))
        ):
            raise ValueError(
                "center must be finite (x, y, z)"
            )

        dimensions = [
            self.bottom_length,
            self.top_length,
            self.width,
            self.height,
        ]

        if (
            not np.all(np.isfinite(dimensions))
            or any(float(v) <= 0 for v in dimensions)
        ):
            raise ValueError(
                "bottom_length, top_length, width and height "
                "must all be finite and > 0"
            )

        if (
            len(self.rotation_deg) != 3
            or not np.all(np.isfinite(self.rotation_deg))
        ):
            raise ValueError(
                "rotation_deg must contain three finite values"
            )

    @property
    def rotation(self):
        return rotation_matrix(
            *self.rotation_deg
        )

    @property
    def half_height(self):
        return float(self.height) / 2.0

    @property
    def half_width(self):
        return float(self.width) / 2.0

    def _local_vertices(self):
        """
        Return the 8 vertices of the unrotated local prism.

        Bottom is at z=-height/2.
        Top is at z=+height/2.
        """

        hb = float(self.bottom_length) / 2.0
        ht = float(self.top_length) / 2.0
        hw = self.half_width
        hz = self.half_height

        return np.array(
            [
                # bottom face
                [-hb, -hw, -hz],
                [ hb, -hw, -hz],
                [ hb,  hw, -hz],
                [-hb,  hw, -hz],

                # top face
                [-ht, -hw,  hz],
                [ ht, -hw,  hz],
                [ ht,  hw,  hz],
                [-ht,  hw,  hz],
            ],
            dtype=float,
        )

    def bounds_xy(self):
        """
        Conservative world x/y bounds from the rotated vertices.
        """

        cx, cy, cz = map(
            float,
            self.center,
        )

        local = self._local_vertices()

        world = (
            local @ self.rotation.T
        )

        world[:, 0] += cx
        world[:, 1] += cy
        world[:, 2] += cz

        return (
            float(world[:, 0].min()),
            float(world[:, 0].max()),
            float(world[:, 1].min()),
            float(world[:, 1].max()),
        )

    def _inside_local(self, qx, qy, qz):
        """
        Test whether local coordinates lie inside the prism.

        At each local z, the allowed x half-length changes linearly
        from bottom_length/2 to top_length/2.
        """

        hz = self.half_height
        hw = self.half_width

        vertical_ok = (
            (qz >= -hz)
            & (qz <= hz)
        )

        width_ok = (
            np.abs(qy) <= hw
        )

        # Normalized vertical position:
        # t=0 at bottom, t=1 at top
        t = (
            (qz + hz)
            / (2.0 * hz)
        )

        half_bottom = (
            float(self.bottom_length) / 2.0
        )

        half_top = (
            float(self.top_length) / 2.0
        )

        half_length = (
            half_bottom
            + t * (
                half_top
                - half_bottom
            )
        )

        length_ok = (
            np.abs(qx)
            <= half_length
        )

        return (
            vertical_ok
            & width_ok
            & length_ok
        )

    def vertical_interval(
        self,
        x,
        y,
    ):
        """
        Return world-z lower/upper intersections of a vertical line
        with the rotated trapezoidal prism.

        This implementation solves the intersection numerically along
        each vertical world line. It is deliberately explicit and robust
        rather than relying on a special-case analytic formula.

        Returns
        -------
        lower, upper, valid
            Arrays matching x/y shape.
        """

        x, y = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )

        cx, cy, cz = map(
            float,
            self.center,
        )

        # Determine conservative possible world-z range from vertices.
        local_vertices = self._local_vertices()

        world_vertices = (
            local_vertices
            @ self.rotation.T
        )

        zmin = (
            cz
            + float(
                world_vertices[:, 2].min()
            )
        )

        zmax = (
            cz
            + float(
                world_vertices[:, 2].max()
            )
        )

        # Sample vertical intersections densely enough to bracket
        # entering/leaving the convex solid.
        #
        # This can later be replaced with an analytic half-space
        # intersection implementation if performance becomes important.
        nz = 257

        z_samples = np.linspace(
            zmin,
            zmax,
            nz,
        )

        lower = np.full(
            x.shape,
            np.nan,
            dtype=float,
        )

        upper = np.full(
            x.shape,
            np.nan,
            dtype=float,
        )

        valid = np.zeros(
            x.shape,
            dtype=bool,
        )

        rt = self.rotation.T

        for z in z_samples:

            dx = x - cx
            dy = y - cy
            dz = z - cz

            qx = (
                rt[0, 0] * dx
                + rt[0, 1] * dy
                + rt[0, 2] * dz
            )

            qy = (
                rt[1, 0] * dx
                + rt[1, 1] * dy
                + rt[1, 2] * dz
            )

            qz = (
                rt[2, 0] * dx
                + rt[2, 1] * dy
                + rt[2, 2] * dz
            )

            inside = self._inside_local(
                qx,
                qy,
                qz,
            )

            newly_inside = (
                inside
                & ~valid
            )

            lower[newly_inside] = z

            upper[inside] = z

            valid |= inside

        return (
            lower,
            upper,
            valid,
        )

    def to_dict(self):

        return {
            "type": "trapezoidal_prism",

            "center_xyz_m": list(
                map(
                    float,
                    self.center,
                )
            ),

            "bottom_length_m": float(
                self.bottom_length
            ),

            "top_length_m": float(
                self.top_length
            ),

            "width_m": float(
                self.width
            ),

            "height_m": float(
                self.height
            ),

            "rotation_deg": list(
                map(
                    float,
                    self.rotation_deg,
                )
            ),
        }

    def volume_m3(self):
        """
        Exact geometric volume of the prism.
        """

        trapezoid_area = (
            0.5
            * (
                float(self.bottom_length)
                + float(self.top_length)
            )
            * float(self.height)
        )

        return (
            trapezoid_area
            * float(self.width)
        )