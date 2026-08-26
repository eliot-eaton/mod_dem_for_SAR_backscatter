#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray
from rasterio.transform import from_origin
from rasterio.enums import Resampling

import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output

from toposhapes_sar import (
    RotatedEllipsoid,
    RotatedCuboid,
    apply_shape,
)


# =============================================================================
# 1. INPUT FILES
# =============================================================================

DATA_DIR = Path("./data")

DEM_PATH = DATA_DIR / "P.dem"
PAR_PATH = DATA_DIR / "P.dem_par"

assert DEM_PATH.exists(), f"Missing DEM: {DEM_PATH}"
assert PAR_PATH.exists(), f"Missing DEM parameter file: {PAR_PATH}"

print("[SETUP] Input files")
print("        DEM:    ", DEM_PATH.resolve())
print("        DEM par:", PAR_PATH.resolve())


# =============================================================================
# 2. READ GAMMA DEM PARAMETER FILE
# =============================================================================

def read_gamma_dem_par(path):
    """
    Parse the small subset of GAMMA DEM parameter metadata needed here.
    """

    values = {}

    with open(path, "r") as f:

        for line in f:

            if ":" not in line:
                continue

            key, value = line.split(":", 1)

            key = key.strip()
            value = value.strip()

            if not value:
                continue

            # First token is sufficient for the numeric fields we need.
            token = value.split()[0]

            values[key] = token

    required = [
        "data_format",
        "width",
        "nlines",
        "corner_lat",
        "corner_lon",
        "post_lat",
        "post_lon",
    ]

    for key in required:
        if key not in values:
            raise ValueError(
                f"Required GAMMA parameter '{key}' not found in {path}"
            )

    return {
        "data_format": values["data_format"],
        "width": int(values["width"]),
        "nlines": int(values["nlines"]),
        "corner_lat": float(values["corner_lat"]),
        "corner_lon": float(values["corner_lon"]),
        "post_lat": float(values["post_lat"]),
        "post_lon": float(values["post_lon"]),
        "projection": values.get("DEM_projection", None),
        "ellipsoid": values.get("ellipsoid_name", None),
        "datum": values.get("datum_name", None),
    }


meta = read_gamma_dem_par(PAR_PATH)

print("\n[SETUP] Parsed GAMMA metadata")

for key, value in meta.items():
    print(f"        {key}: {value}")

if meta["data_format"] != "REAL*4":
    raise ValueError(
        f"Expected GAMMA REAL*4, got {meta['data_format']}"
    )


# =============================================================================
# 3. READ ORIGINAL GAMMA BINARY
# =============================================================================

width = meta["width"]
nlines = meta["nlines"]

expected_bytes = width * nlines * 4

actual_bytes = DEM_PATH.stat().st_size

print("\n[CHECK] GAMMA binary size")
print("        expected:", expected_bytes)
print("        actual:  ", actual_bytes)

if actual_bytes != expected_bytes:
    raise ValueError(
        "P.dem file size does not match width * nlines * 4."
    )

# GAMMA source DEM is big-endian REAL*4
dem_values = np.fromfile(
    DEM_PATH,
    dtype=">f4",
).reshape(
    nlines,
    width,
)

# Convert to native float32 for numerical processing.
dem_values = dem_values.astype(
    np.float32,
    copy=False,
)

print("\n[CHECK] Original DEM")
print("        shape:", dem_values.shape)
print("        dtype:", dem_values.dtype)
print(
    "        elevation range:",
    float(np.nanmin(dem_values)),
    "to",
    float(np.nanmax(dem_values)),
    "m",
)


# =============================================================================
# 4. CREATE AUTHORITATIVE GEOGRAPHIC XARRAY
# =============================================================================

corner_lat = meta["corner_lat"]
corner_lon = meta["corner_lon"]

post_lat = meta["post_lat"]
post_lon = meta["post_lon"]

# GAMMA corner_lat/lon are pixel-centre coordinates.
x_geo = (
    corner_lon
    + np.arange(width) * post_lon
)

y_geo = (
    corner_lat
    + np.arange(nlines) * post_lat
)

dem_geo_original = xr.DataArray(
    dem_values,
    dims=("y", "x"),
    coords={
        "x": x_geo,
        "y": y_geo,
    },
    name="elevation",
)

dem_geo_original = (
    dem_geo_original
    .rio.write_crs("EPSG:4326")
)

west = (
    corner_lon
    - post_lon / 2.0
)

north = (
    corner_lat
    - post_lat / 2.0
)

geo_transform = from_origin(
    west,
    north,
    post_lon,
    abs(post_lat),
)

dem_geo_original = (
    dem_geo_original
    .rio.write_transform(
        geo_transform
    )
)

print("\n[CHECK] Geographic DEM")
print("        CRS:", dem_geo_original.rio.crs)
print("        shape:", dem_geo_original.shape)
print("        resolution:", dem_geo_original.rio.resolution())
print("        bounds:", dem_geo_original.rio.bounds())


# =============================================================================
# 5. PROJECT TO METRES
# =============================================================================

projected_crs = (
    dem_geo_original
    .rio
    .estimate_utm_crs()
)

print("\n[SETUP] Projecting DEM")
print("        projected CRS:", projected_crs)
print("        resampling: nearest neighbour")

dem_m_original = (
    dem_geo_original
    .rio
    .reproject(
        projected_crs,
        resampling=Resampling.nearest,
        nodata=np.nan,
    )
)

print("\n[CHECK] Projected DEM")
print("        shape:", dem_m_original.shape)
print("        CRS:", dem_m_original.rio.crs)
print("        resolution:", dem_m_original.rio.resolution())
print("        bounds:", dem_m_original.rio.bounds())


# =============================================================================
# 6. CHOOSE INTERACTIVE VIEW REGION
# =============================================================================

# Default to the centre of the projected DEM.
VIEW_X = float(
    dem_m_original.x.values[
        len(dem_m_original.x) // 2
    ]
)

VIEW_Y = float(
    dem_m_original.y.values[
        len(dem_m_original.y) // 2
    ]
)

# Half-width of displayed region in metres.
VIEW_RADIUS_M = 500.0

dem_view_original = (
    dem_m_original.sel(
        x=slice(
            VIEW_X - VIEW_RADIUS_M,
            VIEW_X + VIEW_RADIUS_M,
        ),
        # projected y is descending
        y=slice(
            VIEW_Y + VIEW_RADIUS_M,
            VIEW_Y - VIEW_RADIUS_M,
        ),
    )
)

print("\n[SETUP] Interactive DEM crop")
print("        centre x:", VIEW_X)
print("        centre y:", VIEW_Y)
print("        shape:", dem_view_original.shape)

print(
    "        x range:",
    float(dem_view_original.x.min()),
    "to",
    float(dem_view_original.x.max()),
)

print(
    "        y range:",
    float(dem_view_original.y.min()),
    "to",
    float(dem_view_original.y.max()),
)


# =============================================================================
# 7. INITIAL FIXED Z
# =============================================================================

reference_surface_z = float(
    dem_m_original.sel(
        x=VIEW_X,
        y=VIEW_Y,
        method="nearest",
    )
)

INITIAL_Z = reference_surface_z

print("\n[SETUP] Initial shape z")
print("        local DEM surface:", INITIAL_Z, "m")


# =============================================================================
# 8. INTERACTIVE CONTROLS
# =============================================================================

shape_dropdown = widgets.Dropdown(
    options=[
        "ellipsoid",
        "cuboid",
    ],
    value="ellipsoid",
    description="Shape",
)

interaction_dropdown = widgets.Dropdown(
    options=[
        "fill_to_upper",
        "excavate_to_lower",
        "add_thickness",
        "subtract_thickness",
    ],
    value="fill_to_upper",
    description="Interaction",
)

display_dropdown = widgets.Dropdown(
    options=[
        "modified DEM",
        "original DEM",
        "difference",
    ],
    value="modified DEM",
    description="Display",
)


x_slider = widgets.FloatSlider(
    value=VIEW_X,
    min=VIEW_X - 350.0,
    max=VIEW_X + 350.0,
    step=5.0,
    description="x (m)",
    continuous_update=False,
)

y_slider = widgets.FloatSlider(
    value=VIEW_Y,
    min=VIEW_Y - 350.0,
    max=VIEW_Y + 350.0,
    step=5.0,
    description="y (m)",
    continuous_update=False,
)

z_slider = widgets.FloatSlider(
    value=INITIAL_Z,
    min=INITIAL_Z - 300.0,
    max=INITIAL_Z + 300.0,
    step=5.0,
    description="z (m)",
    continuous_update=False,
)


a_slider = widgets.FloatSlider(
    value=75.0,
    min=10.0,
    max=250.0,
    step=5.0,
    description="a / length",
    continuous_update=False,
)

b_slider = widgets.FloatSlider(
    value=50.0,
    min=10.0,
    max=250.0,
    step=5.0,
    description="b / width",
    continuous_update=False,
)

c_slider = widgets.FloatSlider(
    value=50.0,
    min=10.0,
    max=250.0,
    step=5.0,
    description="c / height",
    continuous_update=False,
)


yaw_slider = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=360.0,
    step=5.0,
    description="yaw",
    continuous_update=False,
)

pitch_slider = widgets.FloatSlider(
    value=0.0,
    min=-90.0,
    max=90.0,
    step=5.0,
    description="pitch",
    continuous_update=False,
)

roll_slider = widgets.FloatSlider(
    value=0.0,
    min=-90.0,
    max=90.0,
    step=5.0,
    description="roll",
    continuous_update=False,
)


output = widgets.Output()


# =============================================================================
# 9. SHAPE FACTORY
# =============================================================================

def make_current_shape(
    shape_type,
    x,
    y,
    z,
    a,
    b,
    c,
    yaw,
    pitch,
    roll,
):

    center = (
        float(x),
        float(y),
        float(z),
    )

    rotation = (
        float(yaw),
        float(pitch),
        float(roll),
    )

    if shape_type == "ellipsoid":

        return RotatedEllipsoid(
            center=center,
            semi_axes=(
                float(a),
                float(b),
                float(c),
            ),
            rotation_deg=rotation,
        )

    elif shape_type == "cuboid":

        return RotatedCuboid(
            center=center,
            size=(
                float(a) * 2.0,
                float(b) * 2.0,
                float(c) * 2.0,
            ),
            rotation_deg=rotation,
        )

    raise ValueError(
        f"Unknown shape type: {shape_type}"
    )


# =============================================================================
# 10. INTERACTIVE UPDATE FUNCTION
# =============================================================================

def update_explorer(
    shape_type,
    interaction,
    x,
    y,
    z,
    a,
    b,
    c,
    yaw,
    pitch,
    roll,
    display_mode,
):

    shape = make_current_shape(
        shape_type,
        x,
        y,
        z,
        a,
        b,
        c,
        yaw,
        pitch,
        roll,
    )

    # Visualization only:
    # apply shape to cropped projected DEM.
    dem_modified, dz = apply_shape(
        dem_view_original,
        shape,
        interaction=interaction,
    )

    dz_values = dz.values

    changed_mask = (
        np.isfinite(dz_values)
        & (dz_values != 0)
    )

    changed_pixels = int(
        np.count_nonzero(
            changed_mask
        )
    )

    dx, dy = (
        dem_view_original
        .rio
        .resolution()
    )

    pixel_area_m2 = abs(
        dx * dy
    )

    added_volume_m3 = max(
        0.0,
        float(
            np.sum(
                dz_values[
                    np.isfinite(dz_values)
                    & (dz_values > 0)
                ]
            )
            * pixel_area_m2
        ),
    )

    removed_volume_m3 = max(
        0.0,
        float(
            -np.sum(
                dz_values[
                    np.isfinite(dz_values)
                    & (dz_values < 0)
                ]
            )
            * pixel_area_m2
        ),
    )

    net_volume_m3 = float(
        np.nansum(
            dz_values
        )
        * pixel_area_m2
    )

    # -------------------------------------------------------------------------
    # Select plot surface
    # -------------------------------------------------------------------------

    if display_mode == "original DEM":

        z_plot = (
            dem_view_original.values
        )

        title = (
            "Original DEM"
        )

        colorbar_title = (
            "Elevation (m)"
        )

    elif display_mode == "difference":

        z_plot = dz.values

        title = (
            f"DEM difference — {interaction}"
        )

        colorbar_title = (
            "Δz (m)"
        )

    else:

        z_plot = (
            dem_modified.values
        )

        title = (
            f"Modified DEM — {interaction}"
        )

        colorbar_title = (
            "Elevation (m)"
        )

    # -------------------------------------------------------------------------
    # Plotly figure
    # -------------------------------------------------------------------------

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=dem_view_original.x.values,
            y=dem_view_original.y.values,
            z=z_plot,
            colorscale="Viridis",
            colorbar=dict(
                title=colorbar_title
            ),
            name="DEM",
        )
    )

    # Shape centre marker
    fig.add_trace(
        go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode="markers",
            marker=dict(
                size=6,
                color="red",
            ),
            name="Shape centre",
        )
    )

    fig.update_layout(
        title=title,

        width=900,
        height=700,

        scene=dict(
            xaxis_title="UTM easting (m)",
            yaxis_title="UTM northing (m)",
            zaxis_title=colorbar_title,
            aspectmode="data",
        ),

        margin=dict(
            l=0,
            r=0,
            b=0,
            t=50,
        ),
    )

    # -------------------------------------------------------------------------
    # Output diagnostics
    # -------------------------------------------------------------------------

    with output:

        clear_output(
            wait=True
        )

        print(
            "[INTERACTIVE SHAPE EXPLORER]"
        )

        print(
            f"Shape:           {shape_type}"
        )

        print(
            f"Interaction:     {interaction}"
        )

        print(
            "Centre xyz:      "
            f"{x:.1f}, "
            f"{y:.1f}, "
            f"{z:.1f} m"
        )

        print(
            "Size parameters: "
            f"{a:.1f}, "
            f"{b:.1f}, "
            f"{c:.1f} m"
        )

        print(
            "Rotation:        "
            f"yaw={yaw:.1f}, "
            f"pitch={pitch:.1f}, "
            f"roll={roll:.1f}"
        )

        print(
            f"Changed pixels:  {changed_pixels}"
        )

        print(
            f"Added volume:    {added_volume_m3:,.1f} m³"
        )

        print(
            f"Removed volume:  {removed_volume_m3:,.1f} m³"
        )

        print(
            f"Net volume:      {net_volume_m3:+,.1f} m³"
        )

        display(
            fig
        )


# =============================================================================
# 11. BUILD UI
# =============================================================================

controls = widgets.VBox(
    [
        widgets.HBox(
            [
                shape_dropdown,
                interaction_dropdown,
                display_dropdown,
            ]
        ),

        widgets.HTML(
            "<b>Position</b>"
        ),

        x_slider,
        y_slider,
        z_slider,

        widgets.HTML(
            "<b>Size</b>"
        ),

        a_slider,
        b_slider,
        c_slider,

        widgets.HTML(
            "<b>Rotation</b>"
        ),

        yaw_slider,
        pitch_slider,
        roll_slider,
    ]
)


interactive_output = widgets.interactive_output(
    update_explorer,
    {
        "shape_type": shape_dropdown,
        "interaction": interaction_dropdown,

        "x": x_slider,
        "y": y_slider,
        "z": z_slider,

        "a": a_slider,
        "b": b_slider,
        "c": c_slider,

        "yaw": yaw_slider,
        "pitch": pitch_slider,
        "roll": roll_slider,

        "display_mode": display_dropdown,
    },
)


display(
    widgets.HBox(
        [
            controls,
            output,
        ]
    )
)