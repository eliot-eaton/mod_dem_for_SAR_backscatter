# Examples

This directory demonstrates the recommended `toposhapes-sar-dem` workflow.

The workflow is deliberately split across **two Python environments**:

1. **geo-py / normal scientific Python**
   - inspect the DEM;
   - use the interactive geometry viewer;
   - create modified `P.{ID}.dem` files;
   - compare simulated SAR with observed MLI data.

2. **GAMMA / py_gamma environment**
   - run `gc_map2`;
   - convert `sim_sar` to radar coordinates;
   - write `P.{ID}.sim_sar.radar.tif`.

The GAMMA environment does **not** need the `toposhapes-sar-dem` package to be
installed. This is intentional because GAMMA installations often live in fixed
Python environments that should not be modified.

---

## 1. Install in the normal geo-py environment

From the repository root:

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e ".[test]"
```

For the interactive viewer:

```bash
python -m pip install -e ".[interactive]"
```

For development with both:

```bash
python -m pip install -e ".[test,interactive]"
```

The normal package dependencies include NumPy, xarray, rasterio, rioxarray,
pyproj, Matplotlib and SciPy.

The `interactive` extra adds Plotly, ipywidgets and JupyterLab.

---

## 2. Start with the interactive geometry viewer

Use the interactive notebook first to decide:

- geometry type;
- x/y position;
- absolute z position;
- dimensions;
- yaw, pitch and roll;
- interaction mode.

The viewer works on the projected DEM in metres and should show:

- original or modified 3-D DEM;
- 50 m elevation contours;
- east-west and north-south profiles through the shape centre;
- changed-pixel count;
- added/removed/net volume.

Once a geometry looks sensible, use those explicit values in a
single-realization or parameter-sweep script.

---

## 3. Create modified DEMs in geo-py

A sweep directory may look like:

```text
mod_dem/synthetic_sweep/
├── P.dem_par
├── P.0001.dem
├── 0001.json
├── P.0002.dem
├── 0002.json
└── ...
```

Each `P.{ID}.dem` is GAMMA REAL*4 big-endian.

The original `P.dem` grid remains authoritative. Only the intentional
modification is transferred back to that grid, so unchanged pixels can remain
exactly equal to the source DEM.

---

# GAMMA environment

The GAMMA-side scripts are standalone:

```text
examples/gamma_processing.py
examples/gamma_batch.py
```

They intentionally do **not** import `toposhapes_sar`.

They require only:

- Python standard library;
- `py_gamma`.

---

## 4. Process one DEM with GAMMA

Activate the existing GAMMA environment, then run:

```bash
python ../mod_dem_for_SAR_backscatter/examples/gamma_processing.py     ./mod_dem/synthetic_sweep     0001     ./slcs/20201226M/20201226.mli.par     --output-dir ./sim_sar
```

Inputs:

```text
./mod_dem/synthetic_sweep/P.0001.dem
./mod_dem/synthetic_sweep/P.dem_par
./slcs/20201226M/20201226.mli.par
```

Persistent outputs:

```text
./sim_sar/P.mapped.0001.dem_par
./sim_sar/P.0001.sim_sar.radar.tif
```

All other GAMMA products are temporary.

---

## 5. Process many IDs with GAMMA

```bash
python ../mod_dem_for_SAR_backscatter/examples/gamma_batch.py     ./mod_dem/synthetic_sweep     ./sim_sar     ./slcs/20201226M/20201226.mli.par     0001 0002 0003 0004 0005
```

Arguments:

```text
gamma_batch.py     DEM_INPUT_DIRECTORY     SIMSAR_OUTPUT_DIRECTORY     MLI_PAR     ID [ID ...]
```

Existing `P.{ID}.sim_sar.radar.tif` files are skipped.

To regenerate them:

```bash
python ../mod_dem_for_SAR_backscatter/examples/gamma_batch.py     ./mod_dem/synthetic_sweep     ./sim_sar     ./slcs/20201226M/20201226.mli.par     0001 0002 0003     --overwrite
```

After this finishes, leave the GAMMA environment.

---

# Back in geo-py

## 6. Compare one sim_sar with one MLI

```bash
toposhapes-compare-simsar     ./sim_sar/P.0001.sim_sar.radar.tif     ./mli_tifs/2020-2021/20201226.mli.tif     ./sim_sar/P.0001.sim_sar.20201226
```

This writes:

```text
P.0001.sim_sar.20201226_histogram.png
P.0001.sim_sar.20201226_spatial.png
```

---

## 7. Compare many sim_sar products with one MLI date

```bash
toposhapes-compare-simsar-batch     ./sim_sar     ./mli_tifs/2020-2021/20201226.mli.tif     0001 0002 0003 0004 0005
```

Arguments:

```text
toposhapes-compare-simsar-batch     SIMSAR_DIRECTORY     MLI_TIF     ID [ID ...]
```

For each ID, it reads:

```text
P.{ID}.sim_sar.radar.tif
```

and writes:

```text
P.{ID}.sim_sar.20201226_histogram.png
P.{ID}.sim_sar.20201226_spatial.png
```

Existing comparison pairs are skipped.

To regenerate:

```bash
toposhapes-compare-simsar-batch     ./sim_sar     ./mli_tifs/2020-2021/20201226.mli.tif     0001 0002 0003     --overwrite
```

---

## Recommended order

```text
geo-py
  |
  | interactive viewer
  | generate P.{ID}.dem + {ID}.json
  v

GAMMA / py_gamma environment
  |
  | examples/gamma_batch.py
  v

P.{ID}.sim_sar.radar.tif
  |
  v

geo-py
  |
  | toposhapes-compare-simsar-batch
  v

comparison PNGs
```

The environment split is intentional: the GAMMA environment stays fixed and
does not need the scientific Python package installed into it.
