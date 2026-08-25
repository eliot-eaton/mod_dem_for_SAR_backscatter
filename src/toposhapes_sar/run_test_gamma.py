from pathlib import Path
import py_gamma as pg


def run_gc_map2_test(
    dem_path,
    dem_par_path,
    mli_par_path,
    output_dir,
    run_id="test",
):
    dem_path = Path(dem_path).resolve()
    dem_par_path = Path(dem_par_path).resolve()
    mli_par_path = Path(mli_par_path).resolve()
    output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"P.{run_id}"

    # gc_map2 outputs
    map_dem_par = output_dir / f"{prefix}.map.dem_par"
    map_dem = output_dir / f"{prefix}.map.dem"

    lookup = output_dir / f"{prefix}.lt"

    ls_map = output_dir / f"{prefix}.ls_map"
    ls_map_rdc = output_dir / f"{prefix}.ls_map_rdc"
    inc = output_dir / f"{prefix}.inc"
    res = output_dir / f"{prefix}.res"
    offnadir = output_dir / f"{prefix}.offnadir"

    sim_sar = output_dir / f"{prefix}.sim_sar"

    u = output_dir / f"{prefix}.u"
    v = output_dir / f"{prefix}.v"
    psi = output_dir / f"{prefix}.psi"
    pix = output_dir / f"{prefix}.pix"

    print("\n[CHECK] GAMMA gc_map2 input files")
    print("        DEM:      ", dem_path)
    print("        DEM par:  ", dem_par_path)
    print("        MLI par:  ", mli_par_path)

    for path in [dem_path, dem_par_path, mli_par_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("\n[CHECK] Running gc_map2")
    print("        Purpose: generate SAR geometry from P.{ID}.dem")
    print("        Key product: simulated SAR backscatter (sim_sar)")

    pg.gc_map2(
        str(mli_par_path),
        str(dem_par_path),
        str(dem_path),
        str(map_dem_par),
        str(map_dem),
        str(lookup),
        1,
        1,
        str(ls_map),
        str(ls_map_rdc),
        str(inc),
        str(res),
        str(offnadir),
        str(sim_sar),
        str(u),
        str(v),
        str(psi),
        str(pix),
    )

    print("\n[CHECK] gc_map2 outputs")

    outputs = {
        "map_dem_par": map_dem_par,
        "map_dem": map_dem,
        "lookup": lookup,
        "ls_map": ls_map,
        "ls_map_rdc": ls_map_rdc,
        "inc": inc,
        "res": res,
        "offnadir": offnadir,
        "sim_sar": sim_sar,
        "u": u,
        "v": v,
        "psi": psi,
        "pix": pix,
    }

    for name, path in outputs.items():
        exists = path.exists()
        size = path.stat().st_size if exists else 0

        print(
            f"        {name:12s} "
            f"exists={exists} "
            f"size={size}"
        )

    if not sim_sar.exists():
        raise RuntimeError("gc_map2 completed but sim_sar was not created")

    if sim_sar.stat().st_size == 0:
        raise RuntimeError("sim_sar exists but is empty")

    print("\n[PASS] sim_sar was successfully generated")
    print("       ", sim_sar)

    return outputs


outputs = run_gc_map2_test(
    dem_path="./mod_dem/P.001.dem",
    dem_par_path="./mod_dem/P.dem_par",
    mli_par_path="./slcs/20210208M/20210208.mli.par",
    output_dir="./sim_sar",
    run_id="001",
)    
