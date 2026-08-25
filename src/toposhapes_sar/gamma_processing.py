from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory


def gamma_dem_to_sim_sar_tiff(*, dem_path, dem_par_path, mli_par_path, output_tif, keep_workdir=None):
    """Run the user's gc_map2 -> geocode -> data2tiff sequence with explicit paths.

    Requires GAMMA's `py_gamma` in the runtime environment. The input DEM is the
    generated P.{ID}.dem; gc_map2's map DEM outputs are treated as intermediates.
    """
    try:
        import py_gamma as pg
    except ImportError as exc:
        raise RuntimeError("py_gamma is required for GAMMA processing") from exc

    dem_path=Path(dem_path); dem_par_path=Path(dem_par_path); mli_par_path=Path(mli_par_path); output_tif=Path(output_tif)
    output_tif.parent.mkdir(parents=True,exist_ok=True)

    def run(work: Path):
        map_dem_par=work/"map.dem_par"; map_dem=work/"map.dem"; lookup=work/"lookup.lt"
        ls_map=work/"ls_map"; ls_map_rdc=work/"ls_map_rdc"; inc=work/"inc"; res=work/"res"; offnadir=work/"offnadir"
        sim_sar=work/"sim_sar"; u=work/"u"; v=work/"v"; psi=work/"psi"; pix=work/"pix"
        pg.gc_map2(str(mli_par_path),str(dem_par_path),str(dem_path),str(map_dem_par),str(map_dem),str(lookup),5,5,str(ls_map),str(ls_map_rdc),str(inc),str(res),str(offnadir),str(sim_sar),str(u),str(v),str(psi),str(pix))
        mli=pg.ParFile(str(mli_par_path)); range_samples=int(mli.get_value("range_samples")); azimuth_lines=int(mli.get_value("azimuth_lines"))
        mapped=pg.ParFile(str(map_dem_par)); widthdem=int(mapped.get_value("width"))
        radar=work/"sim_sar.radar"
        pg.geocode(str(lookup),str(sim_sar),widthdem,str(radar),range_samples,azimuth_lines,2,0)
        pg.data2tiff(str(radar),range_samples,2,str(output_tif))
        return output_tif

    if keep_workdir is not None:
        work=Path(keep_workdir); work.mkdir(parents=True,exist_ok=True); return run(work)
    with TemporaryDirectory(prefix="toposhapes_gamma_") as td:
        return run(Path(td))
