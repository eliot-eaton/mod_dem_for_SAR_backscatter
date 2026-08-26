Examples

This directory contains example workflows for modifying a GAMMA P.dem
with synthetic 3-D geometries and then using the modified DEM to
generate simulated SAR backscatter with GAMMA.

Recommended workflow

1.  Explore the DEM interactively to decide where a geometry should be
    placed and how it should be oriented.
2.  Create a modified DEM using the chosen geometry parameters.
3.  Inspect the DEM QA outputs to check that the terrain modification is
    behaving as intended.
4.  Run GAMMA processing in a py_gamma environment to generate sim_sar.
5.  Compare the simulated SAR response with the observed SAR data if
    required.

The original P.dem should be treated as the authoritative DEM. The
package is designed so that pixels outside the modification footprint
remain exactly equal to the original DEM.

1. Explore geometry placement interactively

Start here.

The interactive viewer is intended for choosing: - geometry type; -
easting and northing; - absolute elevation/depth; - geometry
dimensions; - yaw, pitch and roll; - interaction with the existing
terrain.

Available interaction modes include: - fill_to_upper -
excavate_to_lower - add_thickness - subtract_thickness

The viewer displays the original and modified terrain and provides
east-west and north-south profiles through the centre of the geometry.
This makes it much easier to understand how a 3-D object intersects the
existing topography before generating a large parameter sweep.

Input files

Place the original GAMMA DEM and parameter file in:

./data/P.dem ./data/P.dem_par

P.dem is expected to be a GAMMA REAL*4 big-endian DEM.

Run the interactive example in Jupyter or VS Code’s notebook interface,
for example:

conda activate geo-py jupyter lab

Open the interactive-viewer notebook and first select a relatively small
area of the projected DEM.

The viewer works in a projected CRS so that geometry dimensions and
positions are specified directly in metres. Nearest-neighbour
reprojection is deliberately used when constructing the working
projected DEM. This avoids interpolating the original elevation values
unnecessarily.

What to record

Once a useful geometry has been found, record its parameters, for
example:

centre = (432450.0, 350500.0, 2335.0) semi_axes = (75.0, 50.0, 50.0)
rotation_deg = (20.0, 0.0, 0.0) interaction = “fill_to_upper”

These parameters can then be transferred directly into a DEM-generation
script.

2. Produce a modified P.dem

The next step is to apply the selected geometry to the full DEM.

The interactive viewer is only used to help select the geometry. The
final modification should be applied to the full-resolution DEM, not
just the cropped plotting window.

A typical realization produces:

P.001.dem 001.json

where P.001.dem is the modified GAMMA DEM and 001.json records how that
DEM was produced.

For a parameter sweep this becomes:

P.001.dem 001.json P.002.dem 002.json P.003.dem 003.json …

The JSON metadata should make each realization reproducible and includes
information such as geometry type, centre coordinates, dimensions,
rotation, interaction mode, number of modified pixels, added material
volume, removed material volume, net DEM volume change, and validation
information.

DEM preservation

A key design principle is:

Only pixels affected by the synthetic geometry should change.

The geographic P.dem grid is therefore retained as the authoritative
output grid.

The geometry is evaluated in metres on the projected grid, but the
modification (dz) is transferred back to the original geographic grid.
The complete modified DEM is not blindly resampled back and forth.

This allows unchanged pixels in the final DEM to remain exactly equal to
their values in the original P.dem.

The output is written using the original GAMMA binary convention:

REAL*4 big-endian

QA

Before running GAMMA, inspect the QA plot produced for the realization.
It should show a small region around the modification, including
original topography, modified topography, elevation difference and
contours.

Also check the printed validation information. A successful realization
should preserve:

shape_exact x_exact y_exact transform_exact crs_exact
unchanged_pixels_exact

and should not introduce unexpected NaN values.

3. Generate simulated SAR backscatter with GAMMA

The next stage requires an environment containing GAMMA and py_gamma.

For example:

conda activate

The important inputs are:

P.001.dem P.dem_par .mli.par

P.001.dem is the modified DEM. P.dem_par describes the input geographic
DEM. .mli.par contains the SAR geometry required by gc_map2.

Conceptually:

P.001.dem + P.dem_par + MLI parameter file | v gc_map2 | v P.001.sim_sar

gc_map2 may create an internal/map DEM with a different extent and shape
from the input P.dem. This is expected.

The input P.dem was originally constructed with its own frame and
oversampling choices. Running gc_map2 again determines the DEM segment
required for the supplied SAR acquisition. The resulting mapped DEM
should therefore not be interpreted as a replacement for the
authoritative P.001.dem.

The key product for this workflow is:

P.001.sim_sar

Radar coordinates

The sim_sar generated by gc_map2 is initially associated with the mapped
DEM geometry. The GAMMA processing example then uses the lookup table
generated by gc_map2 to produce the corresponding radar-coordinate
product.

The final simulated SAR can then be exported as a GeoTIFF for subsequent
analysis and comparison.

4. Suggested workflow for many realizations

Once one realization has been tested end-to-end, the same process can be
used for a parameter sweep.

Useful parameters to sweep include: - x position; - y position; -
absolute z position; - ellipsoid semi-axes; - trapezoid dimensions; -
yaw; - pitch; - roll; - interaction mode.

It is strongly recommended to test a small number of realizations
through the complete DEM -> GAMMA -> sim_sar workflow before launching a
large parameter sweep.

5. Recommended order

A. Interactive geometry viewer Goal: decide where the geometry goes and
what it should look like.

B. Single DEM realization Goal: generate one P.{ID}.dem, its JSON
metadata and QA products, and confirm that the terrain modification is
correct.

C. GAMMA processing Goal: move to a py_gamma environment and produce
sim_sar for the synthetic topography.

D. SAR comparison Goal: compare the resulting simulated SAR with the
MLI.

E. Parameter sweep Goal: only after the complete workflow works for a
single DEM, generate many realizations by varying geometry, position and
orientation.

Important concepts

Geometry coordinates are in metres

Geometry construction is performed in a projected CRS. Therefore x, y,
shape dimensions and vertical geometry parameters are expressed in
metres.

Geometry z is absolute

The geometry centre elevation can be calculated separately from the
shape function. This is useful when changing x/y position while
deliberately keeping the geometry at a fixed absolute z.

The geometry and DEM are separate objects

A geometry exists independently of the terrain surface. Its interaction
mode determines how it modifies the DEM.

For example, a deeply buried geometry can still be used with an
excavation interaction to define a cavity even if the original geometric
surface does not intersect the terrain.

Preserve the original DEM

The original P.dem is authoritative. Projection is used to make geometry
calculations physically meaningful in metres. It should not cause
unnecessary interpolation of terrain outside the modified region.

Keep the JSON metadata

Do not discard the {ID}.json files when running parameter sweeps. They
provide the link between:

geometry parameters | v P.{ID}.dem | v P.{ID}.sim_sar

Suggested example layout

examples/ |– README.md |– 01_interactive_geometry_viewer.ipynb |–
02_make_one_realization.py |– 03_make_parameter_sweep.py |–
04_gamma_processing.py `– 05_compare_simsar_mli.py
