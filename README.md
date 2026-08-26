# toposhapes-sar-dem

A small, explicit Python package for generating synthetic DEM realizations for GAMMA SAR/InSAR experiments.

The design is based on the supplied `modify_P_dem.ipynb`, but removes notebook-state dependencies and makes the important invariants testable.

## Core rules

1. **The input `P.dem` geographic grid is authoritative and immutable.**
2. GAMMA `REAL*4` binary format and big-endian output are preserved explicitly.
3. The working DEM is reprojected to a local metre CRS with **nearest-neighbour** sampling.
4. Shapes have fixed 3-D `(x, y, z)` centres. `z` is not silently recomputed when `x/y` move.
5. Shape geometry and terrain interaction are separate concepts.
6. Only the intentional **displacement field (`dz`)** is mapped back to the original geographic grid. Unchanged pixels are copied exactly from the original DEM.
7. Each realization is written as `P.{ID}.dem` plus `{ID}.json` provenance.

## Terrain interaction semantics

The package intentionally does not use a vague `add/subtract` flag.

- `fill_to_upper`: raise the DEM to the shape's upper surface wherever that surface is above terrain.
- `excavate_to_lower`: lower the DEM to the shape's lower surface across the shape footprint. This represents a vertical excavation from the existing surface down to the lower surface of a buried solid.
- `add_thickness` / `subtract_thickness`: legacy full-thickness operations retained for experiments where that is actually desired.

### Deep buried sphere example

For a flat DEM at `z=0`, a sphere with centre `z=-1000 m` and radius `100 m`, using `excavate_to_lower`:

- centre of cavity floor = `-1100 m`
- footprint edge = `-1000 m`
- outside the 100 m radius footprint = unchanged `0 m`

So the result is a vertical-sided excavation with a hemispherical lower floor, even though the sphere itself does not intersect the original surface.

## Install

```bash
python -m pip install -e ".[test]"
```
If you would like to use the interactive notebook tool to play with modifying the DEM, please install in interactive mode:
```bash
pip install -e ".[interactive,test]"
```
## Run tests with explanatory output

```bash
pytest -s
```

The tests print *why* each check exists: original binary preservation, metre-unit geometry, add/subtract semantics, exact unchanged pixels, and round-trip grid invariants.

## Real-data check

```bash
python examples/check_real_dem.py /path/to/P.dem /path/to/P.dem_par
```

Then edit and run `examples/make_one_realization.py`.

## GAMMA handoff

`toposhapes_sar.gamma_processing.gamma_dem_to_sim_sar_tiff()` wraps the current sequence:

`P.{ID}.dem + P.dem_par + mli.par -> gc_map2 -> sim_sar -> geocode -> data2tiff`

All `gc_map2` map DEM / lookup / geometry products are temporary unless `keep_workdir=` is supplied. `py_gamma` is deliberately not a package dependency because it is supplied by a GAMMA installation.

## Reference notebook

The supplied notebook is preserved at `docs/modify_P_dem_reference.ipynb` for traceability.

## Visual QA plots for every realization

`examples/make_one_realization.py` now writes two cropped PNGs automatically from the projected metre-grid DEM:

- `{ID}.dem_check.png` — modified DEM shown as colour plus elevation contours; a red outline marks the intentionally changed footprint.
- `{ID}.difference_check.png` — the authoritative projected displacement `dz = modified - original`, so positive additions and negative excavations are immediately visible.

By default the crop contains every changed pixel plus **500 m of context on each side**. Change `QA_PADDING_M` in the example workflow if a tighter or wider view is useful. Cropping is visualization-only; the DEM itself always remains full size.

For a run with `RUN_ID = "001"`, the persistent realization outputs are therefore:

```text
P.001.dem
001.json
001.dem_check.png
001.difference_check.png
```

The JSON also records the QA filenames.
