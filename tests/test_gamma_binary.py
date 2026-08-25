import numpy as np
import xarray as xr
from toposhapes_sar.gamma_dem import write_gamma_dem


def test_write_big_endian_real4(tmp_path):
    print("\n[CHECK] GAMMA output preserves the original REAL*4 big-endian binary convention.")
    values=np.array([[1.25,2.5],[3.75,4.0]],dtype=np.float32)
    da=xr.DataArray(values,dims=("y","x"))
    out=tmp_path/"P.test.dem"; write_gamma_dem(da,out)
    raw=np.fromfile(out,dtype=">f4").reshape(2,2)
    print(f"        Expected bytes={values.size*4}; written bytes={out.stat().st_size}.")
    assert out.stat().st_size==values.size*4
    assert np.array_equal(raw,values)
