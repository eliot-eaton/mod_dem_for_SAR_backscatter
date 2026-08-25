import numpy as np
from toposhapes_sar.shapes import RotatedEllipsoid, RotatedCuboid


def test_unrotated_sphere_interval():
    print("\n[CHECK] Geometry uses explicit metre-valued x, y, z coordinates.")
    s=RotatedEllipsoid(center=(0,0,-1000),semi_axes=(100,100,100))
    lo,hi,valid=s.vertical_interval(np.array([[0.0]]),np.array([[0.0]]))
    print("        Sphere centre z=-1000 m, radius=100 m -> centre-line interval [-1100,-900] m.")
    assert valid[0,0]
    assert lo[0,0]==-1100
    assert hi[0,0]==-900


def test_cuboid_interval():
    print("\n[CHECK] Cuboid lower/upper surfaces are represented independently from terrain.")
    c=RotatedCuboid(center=(0,0,10),size=(20,30,40))
    lo,hi,valid=c.vertical_interval(np.array([[0.0]]),np.array([[0.0]]))
    assert valid[0,0] and lo[0,0]==-10 and hi[0,0]==30
