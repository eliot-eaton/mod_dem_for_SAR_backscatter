# Examples

This directory contains example workflows for modifying a GAMMA `P.dem`
with synthetic 3-D geometries and then using the modified DEM to generate
simulated SAR backscatter with GAMMA.

The recommended workflow is:

1. **Explore the DEM interactively** to decide where a geometry should be
   placed and how it should be oriented.
2. **Create a modified DEM** using the chosen geometry parameters.
3. **Inspect the DEM QA outputs** to check that the terrain modification is
   behaving as intended.
4. **Run GAMMA processing** in a `py_gamma` environment to generate
   `sim_sar`.
5. Compare the simulated SAR response with the observed SAR data if required.

The original `P.dem` should be treated as the authoritative DEM. The package
is designed so that pixels outside the modification footprint remain exactly
equal to the original DEM.


## 1. Explore geometry placement interactively

Start here.

The interactive viewer is intended for choosing:

- geometry type;
- easting and northing;
- absolute elevation/depth;
- geometry dimensions;
- yaw, pitch and roll;
- interaction with the existing terrain.

For example, the available interaction modes include:

- `fill_to_upper`
- `excavate_to_lower`
- `add_thickness`
- `subtract_thickness`

The viewer displays the original and modified terrain and provides
east-west and north-south profiles through the centre of the geometry.
This makes it much easier to understand how a 3-D object intersects the
existing topography before generating a large parameter sweep.

### Input files

Place the original GAMMA DEM and parameter file in:

```text
./data/P.dem
./data/P.dem_par