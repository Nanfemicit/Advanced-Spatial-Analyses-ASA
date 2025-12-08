"""
Advanced Spatial Analysis - Optimizing the Dutch energy grid using weather data – Assignment 2.2
Author: Ruth Femicit Dala

Aim and expected outcomes
- Combine Dutch high voltage network data with weather station data from KNMI to analyse how temperature and wind speed affect available grid capacity.
- Quantify and visualise how capacity factors vary in space (across power lines) and how forest cover modifies wind speed and thus line capacity.
- Produce maps, plots and summary statistics that could support operational decisions for grid management under different weather conditions.

Input data

1. Map of weather stations
    File format: vector file (e.g. .shp / .gpkg)
    Raster/vector: vector point data
    Variable: station ID, station name, coordinates, possibly metadata
    Units: coordinates in projected CRS (e.g. meters)
    Dimensions: collection of points (N stations)
    Resolution (space): point locations (no grid)
    Projection: to be read from file (will reproject to Dutch CRS, e.g. EPSG:28992)
    Other: used to spatially link KNMI station time series to lines

2. High voltage network (TenneT)  
    File format: vector file (e.g. .shp / .gpkg)
    Raster/vector: vector line data
    Variable: power line geometries, IDs, voltage level, possibly capacity attributes
    Units: coordinates in projected CRS (e.g. meters)
    Dimensions: polyline features (N lines)
    Resolution (space): line representation of the network
    Projection: to be read from file (likely Dutch national CRS)
    Other: main geometry over which capacity and nearest station are computed

3. Forest cover raster
    File format: raster (e.g. .tif)
    Raster/vector: raster
    Variable: forest presence / forest fraction / land cover class
    Units: dimensionless (0/1, fraction, or class code)
    Dimensions: rows × columns (x × y)
    Resolution (space): high spatial resolution (e.g. 10–25 m, to be confirmed)
    Resolution (time): static snapshot (no time dimension)
    Projection: to be read from file (must match or be reprojected to network CRS)
    Other: used to determine where wind speed should be reduced (forested areas)

4. KNMI weather data via API (get_knmi_data function)
    File format: JSON (downloaded text, then parsed to Python dict / DataFrame)
    Raster/vector: tabular time series per station (later converted to point GeoDataFrame)
    Variable: hourly temperature, wind speed, and other meteorological variables
    Units: as defined by KNMI (e.g. temperature in 0.1 °C, wind speed in 0.1 m/s, etc.)
    Dimensions: station × time (24 hours for a single date)
    Resolution (time): hourly (24 entries per station per day)
    Resolution (space): at station locations (discrete points)
    Projection: not spatial yet; coordinates are added by merging with station map
    Other: downloaded and cached locally to avoid repeated API calls

Output data
- GeoDataFrame of weather stations with one selected hour of temperature and wind speed (vector, point).
- Map of weather stations visualising spatial patterns in temperature and wind speed (PNG).
- High voltage network map (base map, PNG).
- High voltage network with lines coloured by temperature of nearest station (PNG).
- High voltage network with lines coloured by wind speed of nearest station (PNG).
- Capacity factor per power line based on temperature and wind speed (added attribute in line GeoDataFrame).
- Map of line capacity factors (PNG).
- Aggregated forest raster at coarser resolution, suitable for identifying larger forest areas (raster, GeoTIFF, plus PNG visualisation).
- Network capacity map after reducing wind speed by 30% in forested areas (PNG) and updated capacity factor attribute.

Processing steps
- Set up Python environment and import required packages (geopandas, pandas, numpy, shapely, rasterio, rasterstats, matplotlib, requests, json, pathlib).
- Use get_knmi_data(date, station_id="ALL") to download KNMI weather data for a chosen date; cache the JSON response.
- Parse the JSON string to a Python dict using json.loads(...) and convert to a pandas DataFrame with pd.DataFrame.from_dict(...).
- Explore the KNMI variables and units; select one specific hour of the day and subset the DataFrame to that hour only.
- Load the weather station location map (vector file) and merge it with the KNMI DataFrame on station ID to create a GeoDataFrame with geometry and weather attributes.
- Reproject the weather station GeoDataFrame to the same CRS as the high voltage network (and forest raster if needed).
- Make spatial plots of weather stations, coloured by temperature and by wind speed; check if the patterns are realistic.
- Load the high voltage network vector data (power lines) and plot the network to confirm geometry and extent.
- For each power line, find the closest weather station (e.g. using geometry.distance and idxmin) and join the corresponding temperature and wind speed values to the line attributes.
- Create maps of the power network with lines coloured by temperature and by wind speed of the nearest station; verify that all stations are used and that CRS settings are correct.
- Implement the capacity factor formula for power lines using temperature and wind speed (from lecture slides) and store the resulting capacity factor as a new attribute in the power line GeoDataFrame.
- Plot line capacity factors over the network to visualise spatial differences in available capacity.
- Load the high-resolution forest raster using rasterio; choose an appropriate coarser resolution and aggregation method (e.g. mean or majority) to emphasise larger forest patches.
- Aggregate (resample) the forest raster to the chosen resolution; justify the choice of method and resolution; plot the aggregated forest raster.
- Sample the aggregated forest raster at or along the power lines (e.g. using rasterstats or rasterio.sample) to determine which lines (or line segments) are influenced by forest cover.
- Reduce wind speed by 30% where forest is present, compute a forest-modified wind speed, and recompute the capacity factor; add this as a new attribute to the power line GeoDataFrame.
- Produce a new capacity map that includes the forest-reduced wind speed scenario and compare it visually and statistically to the original capacity map.
- Interpret and describe: how do temperature, wind speed, and forest cover affect grid capacity; discuss assumptions, possible errors (distance to stations, temporal resolution, interpolation), and limitations.
"""

# Imports
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shapely
import rasterio
from rasterstats import zonal_stats
import json
from pathlib import Path
import requests
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import rowcol

# Setup directory structure
try:
    BASE_DIR = Path(__file__).parent       # works in .py scripts
except:
    BASE_DIR = Path().resolve()           # works in Jupyter notebooks

data_dir = BASE_DIR / "data"
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

# KNMI API function 
def get_knmi_data(date, station_id="ALL", cache=True):
    """
    Download KNMI weather data from their API.
    Returns a JSON string.
    """

    cache_folder = BASE_DIR / "cache"
    cache_folder.mkdir(exist_ok=True)
    cache_file = f"{date}_{station_id}.json"
    cache_path = cache_folder / cache_file

    if cache and cache_path.exists():
        with open(cache_path, "r") as f:
            return f.read()

    url = "https://www.daggegevens.knmi.nl/klimatologie/uurgegevens"
    params = {
        "start": date,
        "end": date,
        "vars": "ALL",
        "stns": station_id,
        "fmt": "json",
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.text
        if cache:
            with open(cache_path, "w") as f:
                f.write(data)
        return data
    else:
        response.raise_for_status()

# Download and inspect KNMI data for  selected date: 2019-08-25
date_str = "20190825"

raw_json = get_knmi_data(date_str, station_id="ALL", cache=True)

# parse JSON string to Python object
data_obj = json.loads(raw_json)

print("Top-level JSON type:", type(data_obj))

# convert JSON to pandas DataFrame (robust to two common formats)
if isinstance(data_obj, list):
    knmi_df = pd.DataFrame.from_records(data_obj)
elif isinstance(data_obj, dict) and "data" in data_obj:
    knmi_df = pd.DataFrame.from_records(data_obj["data"])
else:
    knmi_df = pd.DataFrame.from_dict(data_obj)

print("KNMI dataframe columns:")
print(knmi_df.columns)
print(knmi_df.head())

target_hour = 14  # 14:00 (2 PM)
knmi_hour = knmi_df[knmi_df["hour"] == target_hour].copy()

print(f"\nSelected records for hour = {target_hour}:00")
print(knmi_hour.head())
print("Number of rows for this hour:", len(knmi_hour))

# Convert KNMI units to something readable
# T: 0.1 °C -> °C
# FF: 0.1 m/s -> m/s

knmi_hour["T_C"] = knmi_hour["T"] / 10.0
knmi_hour["FF_ms"] = knmi_hour["FF"] / 10.0

print("\nWith converted units (T_C and FF_ms):")
print(knmi_hour[["station_code", "hour", "T", "T_C", "FF", "FF_ms"]].head())

# Load weather station locations 
stations_path = data_dir / "weather_stations.gpkg"

if not stations_path.exists():
    raise FileNotFoundError(f"Station file not found: {stations_path}")

print("\nUsing station file:", stations_path)

stations_gdf = gpd.read_file(stations_path)
print("Station GeoDataFrame columns:")
print(stations_gdf.columns)
print(stations_gdf.head())
print("Station CRS:", stations_gdf.crs)

# Detect join column between stations and KNMI
knmi_codes = set(knmi_hour["station_code"].astype(str).unique())

join_col = None
for col in stations_gdf.columns:
    if col == stations_gdf.geometry.name:
        continue
    values = stations_gdf[col].astype(str)
    overlap = values.isin(knmi_codes).sum()
    if overlap > 0:
        join_col = col
        print(f"Potential join column found: {col} (overlap = {overlap})")
        break

if join_col is None:
    raise ValueError(
        "Could not find a matching join column between station shapefile and KNMI station_code. "
        "Inspect stations_gdf and knmi_hour and set join_col manually."
    )

print("Using join column:", join_col)

# Merge KNMI hourly weather onto station locations
stations_weather = stations_gdf.merge(
    knmi_hour,
    left_on=join_col,
    right_on="station_code",
    how="inner",
)

print("\nMerged station-weather GeoDataFrame:")
print(stations_weather[[join_col, "station_code", "T_C", "FF_ms"]].head())
print("Number of stations with data:", len(stations_weather))

# spatial plots of weather stations
# keep only stations with non-missing T_C and FF_ms
stations_plot = stations_weather.dropna(subset=["T_C", "FF_ms"]).copy()

print("\nStations used for plotting (non-missing T_C and FF_ms):", len(stations_plot))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# plot temperature
stations_plot.plot(
    column="T_C",
    cmap="coolwarm",
    legend=True,
    ax=axes[0],
    markersize=40,
)
axes[0].set_title("KNMI stations – temperature (°C) at 14:00, 25 Aug 2019")
axes[0].set_xlabel("longitude")
axes[0].set_ylabel("latitude")

# plot wind speed
stations_plot.plot(
    column="FF_ms",
    cmap="viridis",
    legend=True,
    ax=axes[1],
    markersize=40,
)
axes[1].set_title("KNMI stations – wind speed (m/s) at 14:00, 25 Aug 2019")
axes[1].set_xlabel("longitude")
axes[1].set_ylabel("latitude")

plt.tight_layout()

# save figure
stations_plot_path = output_dir / "stations_temperature_wind_20190825_14h.png"
plt.savefig(stations_plot_path, dpi=150)
plt.close()

print("Saved station temperature/wind map to:", stations_plot_path)

# Load high-voltage network
hv_path = data_dir / "high_voltage_net.gpkg"
print("\nLoading high-voltage network from:", hv_path)

power_lines = gpd.read_file(hv_path)
print("Power line columns:", power_lines.columns)
print("CRS of power lines:", power_lines.crs)
print(power_lines.head())

# Reproject weather stations to network CRS
if stations_weather.crs != power_lines.crs:
    print("Reprojecting stations to match power line CRS...")
    stations_weather = stations_weather.to_crs(power_lines.crs)

# Plot the network base map
plt.figure(figsize=(8, 10))
power_lines.plot(color="black", linewidth=1)
plt.title("Dutch High-Voltage Network – Base Map")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")

base_map_path = output_dir / "network_base_map.png"
plt.savefig(base_map_path, dpi=150)
plt.close()

print("Saved network base map to:", base_map_path)

# assign closest weather station to each power line
weather_data = stations_weather.dropna(subset=["T_C", "FF_ms"]).copy()
print("\nNumber of stations with valid T and FF:", len(weather_data))

def get_closest_weather_station(power_line_geom):
    """Return the index of the closest weather station to this power line geometry."""
    return weather_data.geometry.distance(power_line_geom).idxmin()

# compute index of closest station for each power line
power_lines["closest_station_idx"] = power_lines.geometry.apply(get_closest_weather_station)

print("example closest_station_idx values:")
print(power_lines["closest_station_idx"].value_counts().head())
print("unique stations used for all lines:", power_lines["closest_station_idx"].nunique())

# join station weather attributes onto power lines using the index
power_lines = power_lines.join(
    weather_data[["T_C", "FF_ms"]],
    on="closest_station_idx",
    rsuffix="_station",
)

print("\npower lines with attached weather columns:")
print(power_lines[["closest_station_idx", "T_C", "FF_ms"]].head())

# plots: power lines coloured by temperature and wind

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# temperature on lines
power_lines.plot(
    column="T_C",
    cmap="coolwarm",
    linewidth=1.2,
    legend=True,
    ax=axes[0],
)
axes[0].set_title("power line temperature (°C) – closest station")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

# wind speed on lines
power_lines.plot(
    column="FF_ms",
    cmap="viridis",
    linewidth=1.2,
    legend=True,
    ax=axes[1],
)
axes[1].set_title("power line wind speed (m/s) – closest station")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.tight_layout()
network_weather_path = output_dir / "network_temperature_wind_closest_station.png"
plt.savefig(network_weather_path, dpi=150)
plt.close()

print("Saved network temperature/wind plot to:", network_weather_path)

print(power_lines["closest_station_idx"].value_counts().head())
print("unique stations used for all lines:", power_lines["closest_station_idx"].nunique())

# capacity factor based on temp and wind speed
print("\n=== Calculating capacity factor for power lines ===")

# reference values
T_ref = 20.0   # °C
V_ref = 5.0    # m/s

def compute_capacity_factor(T_C, V_ms):
    """
    Capacity model from lecture:
    - CF decreases 1% per °C above 20
    - CF increases 2% per m/s above 5
    - CF limited between 0.5 and 1.5
    """
    cf = 1.0 - 0.01 * (T_C - T_ref) + 0.02 * (V_ms - V_ref)
    return np.clip(cf, 0.5, 1.5)

# only compute where there is valid data
valid = power_lines["T_C"].notna() & power_lines["FF_ms"].notna()

power_lines["capacity_factor"] = np.nan
power_lines.loc[valid, "capacity_factor"] = compute_capacity_factor(
    power_lines.loc[valid, "T_C"],
    power_lines.loc[valid, "FF_ms"]
)

# summary
print("\nCapacity factor summary:")
print(power_lines["capacity_factor"].describe())

# plot capacity factor map
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
power_lines.plot(
    ax=ax,
    column="capacity_factor",
    cmap="viridis",
    linewidth=1,
    legend=True,
)
ax.set_title("Power Line Capacity Factor (Temperature + Wind)")
ax.set_axis_off()

cap_map_path = output_dir / "network_capacity_factor_temperature_wind.png"
plt.tight_layout()
plt.savefig(cap_map_path, dpi=200)
plt.close(fig)

print(f"Saved capacity factor map to: {cap_map_path}")

# -------------------------------------------------------------------
# forest raster: aggregate to coarser resolution and plot
# -------------------------------------------------------------------

forest_path = data_dir / "forest_NL.tif"

with rasterio.open(forest_path) as src:
    forest = src.read(1)
    forest_profile = src.profile
    forest_transform = src.transform
    forest_crs = src.crs
    forest_nodata = src.nodata

print("\nForest raster info:")
print("  shape (rows, cols):", forest.shape)
print("  CRS:", forest_crs)
print("  original transform:", forest_transform)

# I use a simple block-mean aggregation (equivalent to average resampling)
# with an aggregation factor of 10. This groups 10×10 original pixels into
# one coarser cell and takes the mean. Because the forest raster is binary
# (0 = no forest, 1 = forest), the mean represents the fraction of forest
# in each coarse cell. This filters out tiny forest fragments and highlights
# larger patches that are more relevant for reducing wind speed along power lines.

agg_factor = 10  # factor 10 means 10x10 original pixels -> 1 new pixel

rows, cols = forest.shape
new_rows = rows // agg_factor
new_cols = cols // agg_factor

# trim to a multiple of agg_factor to avoid shape issues
forest_trim = forest[: new_rows * agg_factor, : new_cols * agg_factor]

# aggregate by mean (gives forest fraction in each bigger cell)
forest_agg = forest_trim.reshape(
    new_rows, agg_factor, new_cols, agg_factor
).mean(axis=(1, 3))

# update transform: pixels are agg_factor times larger
forest_agg_transform = forest_transform * Affine.scale(agg_factor, agg_factor)

print("Aggregated forest shape:", forest_agg.shape)
print("Aggregated transform:", forest_agg_transform)

# plot aggregated forest
fig, ax = plt.subplots(1, 1, figsize=(6, 8))
im = ax.imshow(forest_agg, cmap="Greens")
ax.set_title(f"Aggregated forest raster (factor {agg_factor})")
plt.colorbar(im, ax=ax, label="Forest cover (mean of block)")
ax.set_axis_off()

forest_agg_png = output_dir / f"forest_aggregated_factor{agg_factor}.png"
plt.tight_layout()
plt.savefig(forest_agg_png, dpi=200)
plt.close(fig)

print(f"Saved aggregated forest raster plot to: {forest_agg_png}")

# -------------------------------------------------------------------
# Sample aggregated forest for each power line and adjust wind speed
# -------------------------------------------------------------------

# Compute line centroids and reproject to forest CRS
power_centroids_forest_crs = power_lines.to_crs(forest_crs).geometry.centroid

# Helper function: get forest fraction from aggregated raster
def sample_forest_fraction(x, y, arr, transform):
    """
    Sample the aggregated forest raster at coordinate (x, y).

    Returns the forest fraction in [0,1] or np.nan if outside.
    """
    try:
        r, c = rowcol(transform, x, y)
    except Exception:
        return np.nan

    if (0 <= r < arr.shape[0]) and (0 <= c < arr.shape[1]):
        return arr[r, c]
    else:
        return np.nan

# Create a new column with forest fraction at each line centroid
forest_frac_list = []
for geom in power_centroids_forest_crs:
    x, y = geom.x, geom.y
    frac = sample_forest_fraction(x, y, forest_agg, forest_agg_transform)
    forest_frac_list.append(frac)

power_lines["forest_fraction"] = forest_frac_list

print("\nForest fraction statistics at line centroids:")
print(power_lines["forest_fraction"].describe())

# Threshold for "forested" lines (forest fraction >= 0.3)
forest_threshold = 0.3
is_forested = power_lines["forest_fraction"] >= forest_threshold

# New wind speed column with forest reduction
power_lines["FF_ms_forest"] = power_lines["FF_ms"].copy()
power_lines.loc[is_forested, "FF_ms_forest"] = (
    power_lines.loc[is_forested, "FF_ms"] * 0.7  # 30% reduction
)

print("\nNumber of lines affected by forest wind reduction:", is_forested.sum())

# -------------------------------------------------------------------
# Capacity factor with forest-reduced wind
# -------------------------------------------------------------------

valid_forest_mask = (
    power_lines["T_C"].notna()
    & power_lines["FF_ms_forest"].notna()
)

power_lines["capacity_factor_forest"] = np.nan
power_lines.loc[valid_forest_mask, "capacity_factor_forest"] = (
    compute_capacity_factor(
        power_lines.loc[valid_forest_mask, "T_C"],
        power_lines.loc[valid_forest_mask, "FF_ms_forest"],
    )
)

print("\nCapacity factor with forest reduction statistics:")
print(power_lines["capacity_factor_forest"].describe())

# Difference in capacity (forest scenario minus original)
capacity_diff = (
    power_lines["capacity_factor_forest"]
    - power_lines["capacity_factor"]
)
print("\nChange in capacity due to forest (forest - original):")
print(capacity_diff.describe())

# -------------------------------------------------------------------
# Plot: capacity with and without forest wind reduction
# -------------------------------------------------------------------

vmin = min(
    power_lines["capacity_factor"].min(),
    power_lines["capacity_factor_forest"].min(),
)
vmax = max(
    power_lines["capacity_factor"].max(),
    power_lines["capacity_factor_forest"].max(),
)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
ax1, ax2 = axes

# Original capacity
power_lines.plot(
    column="capacity_factor",
    ax=ax1,
    legend=True,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
ax1.set_title("Capacity factor (no forest adjustment)")
ax1.set_axis_off()

# Forest-adjusted capacity
power_lines.plot(
    column="capacity_factor_forest",
    ax=ax2,
    legend=True,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax,
)
ax2.set_title("Capacity factor (wind reduced by 30% in forests)")
ax2.set_axis_off()

plt.tight_layout()
cap_forest_png = output_dir / "network_capacity_factor_with_forest_reduction.png"
plt.savefig(cap_forest_png, dpi=200)
plt.close(fig)

print("Saved forest-adjusted capacity map to:", cap_forest_png)

"""
Interpreting the Results
------------------------

1. Weather station plots
   - The station maps for 25 August 2019 at 14:00 show how temperature (T_C) and
     wind speed (FF_ms) vary across the Netherlands. Warmer stations cluster in
     some areas, while wind speeds vary along the coast and inland.
   - These patterns indicate that even for a single hour of one day, there is
     spatial variability in cooling conditions for the power grid.

2. Power lines with nearest-station temperature and wind
   - By assigning each power line segment the temperature and wind speed of the
     closest KNMI station, I obtain a first approximation of local weather
     conditions for the network.
   - The plots coloured by T_C and FF_ms show that coastal and some inland lines
     experience higher wind speeds, while certain inland regions are relatively
     warm with lower wind, which is less favourable for line cooling.

3. Capacity factor based on temperature and wind
   - The capacity factor map (without forest adjustment) reflects the simple
     model used: capacity decreases by 1% per °C above 20 °C and increases by
     2% per m/s above 5 m/s, clipped between 0.5 and 1.5.
   - On this specific summer day and hour, many lines have capacity factors
     below 1, driven by temperatures above 20 °C and moderate winds. Lines in
     relatively cooler and windier locations achieve slightly higher capacity.

4. Aggregated forest raster
   - After aggregation with a factor of 10, the forest raster emphasises larger,
     contiguous forest patches instead of very small fragments. The mean of
     the original binary pixels gives a forest fraction per coarse cell.
   - Areas with high forest fraction (> 0.3) indicate locations where wind
     speed is likely reduced by canopy effects over a relevant spatial scale.

5. Capacity with forest-reduced wind
   - By reducing wind speed by 30% in cells where the forest fraction exceeds
     0.3, the forest-adjusted capacity factor is lower than the original
     capacity factor for those lines.
   - The summary statistics and the comparison map show that the reduction
     is modest but systematic: forested lines lose some cooling benefit from
     wind, which translates into a small loss of thermal capacity.
   - The difference distribution (capacity_factor_forest - capacity_factor)
     quantifies this impact and can be used to identify especially sensitive
     parts of the network.

Assumptions and Limitations
---------------------------

- Single-day, single-hour analysis:
  The analysis uses weather data for one specific day (2019-08-25) at one
  hour (14:00). This captures a realistic snapshot, but it does not represent
  seasonal or long-term variability in grid capacity.

- Nearest-station assignment:
  Each power line is assigned weather conditions from the nearest KNMI station.
  This assumes that temperature and wind fields are locally homogeneous around
  the station, and it ignores small-scale gradients and local effects along
  the line itself.

- Simple capacity model:
  The capacity factor model is linear and highly simplified: it assumes a fixed
  reference temperature and wind speed and uniform sensitivity for all lines.
  In reality, capacity depends on voltage level, conductor type, sag limits,
  solar radiation, and safety margins from operators like TenneT.

- Forest representation and threshold:
  The forest raster is aggregated by block mean, and a threshold of 0.3 on
  forest_fraction is used to decide whether a line is "in forest". This is a
  pragmatic choice to focus on substantial forest patches. Real aerodynamic
  effects depend on stand height, density, and upwind fetch, which are not
  included here.

- Line representation by centroids:
  Forest influence is sampled at the line centroid, not along the full
  geometry. This can miss cases where only part of a line crosses a forested
  area or where forest is upwind but not directly underneath the line.

- Static forest and weather:
  The analysis treats forest as static and weather as an instantaneous field.
  It does not consider time lags, dynamic stability constraints, or temporal
  averaging that a grid operator would apply in practice.

"""
