'''
Advanced Spatial Analysis - Climate Stripes – Assignment 2.1
Author: Ruth Femicit Dala

PLAN
Aim & Expected Outcomes:
- Calculate daily and yearly mean temperatures for selected countries.
- Compute anomalies relative to 1981–2014.
- Visualise anomalies as climate stripes for historical and future scenarios.

Input Data:
GADM – Country Boundaries
    File format: .gpkg
    Vector polygons containing national boundaries
    Units: N/A
    Dimensions: polygons
    Projection: read from dataset
    Other: global administrative dataset

ISIMIP Temperature Data (tas)
    File format: NetCDF (.nc)
    Raster gridded climate data
    Variable: tas (air temperature)
    Units: Kelvin
    Dimensions: time × lat × lon
    Resolution: daily time steps, ~0.5° spatial grid
    Period: 1981–2014 (historical), 2015–2050 (future SSP scenarios)

Output Data:
- Daily mean temperature plot (PNG)
- Yearly temperature anomaly plot (PNG)
- Climate stripes (PNG)
- Multi-country region stripes (PNG)
- Climate stripes including future SSP1–2.6 and SSP5–8.5 projections
- CSV files containing yearly means + anomalies

Processing Steps:
- Load GADM boundaries
- Select target countries
- Load historical temperature files
- Ensure CRS alignment between raster and vector data
- Clip raster to country boundaries
- Compute daily means, yearly means, baseline, anomalies
- Generate climate stripes
- Wrap workflow in functions
- Process multiple countries & multi-country regions
- Load future scenario data
- Compute anomalies using fixed 1981–2014 baseline
- Generate combined historical + future stripes
'''

# ---------------------------------------------
# Imports
# ---------------------------------------------

import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from pathlib import Path

# ---------------------------------------------
# Setup base directories and load GADM
# ---------------------------------------------

BASE_DIR = Path(__file__).parent

gadm_path = BASE_DIR / "data" / "gadm" / "gadm_410_0.gpkg"
print("Loading GADM:", gadm_path)

gdf = gpd.read_file(gadm_path)
print("GADM loaded. CRS:", gdf.crs)

print("Columns:", gdf.columns)

# Detect country name column (Netherlands search)
country_col = None
for col in gdf.columns:
    if gdf[col].dtype == "object" and gdf[col].astype(str).str.contains("Netherlands").any():
        country_col = col
        break

print("IDENTIFIED COUNTRY COLUMN:", country_col)

# ---------------------------------------------
# Directories and colormap
# ---------------------------------------------

plots_dir = BASE_DIR / "outputs" / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

processed_dir = BASE_DIR / "outputs" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

hist_dir = BASE_DIR / "data" / "historical_temp"
hist_files = sorted(hist_dir.glob("*.nc"))

print("Historical files found:")
for f in hist_files:
    print(" -", f.name)

stripe_cmap = ListedColormap(
    [
        "#08306b", "#08519c", "#2171b5", "#4292c6",
        "#6baed6", "#9ecae1", "#c6dbef", "#deebf7",
        "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a",
        "#ef3b2c", "#cb181d", "#a50f15", "#67000d"
    ]
)

# ---------------------------------------------
# Function: historical processing
# ---------------------------------------------

def process_country_or_region(country_names, label=None):

    if isinstance(country_names, str):
        country_names = [country_names]

    if label is None:
        label = "_".join(country_names).lower().replace(" ", "_")

    print(f"\n=== HISTORICAL PROCESSING: {country_names} ===")

    subset = gdf[gdf[country_col].isin(country_names)]
    geom = subset.geometry.values

    clipped = []
    for f in hist_files:
        print("Clipping:", f.name)
        ds = xr.open_dataset(f)
        ds = ds.rio.write_crs("EPSG:4326")
        ds = ds.rio.clip(geom, subset.crs)
        clipped.append(ds)

    # Combine 1981–2014 dataset
    ds_all = xr.concat(clipped, dim="time")

    # Daily mean
    daily = ds_all["tas"].mean(dim=["lat", "lon"])
    daily_df = daily.to_dataframe().reset_index()
    daily_df.rename(columns={"tas": "temp"}, inplace=True)

    # Daily plot
    plt.figure(figsize=(10,4))
    plt.plot(daily_df["time"], daily_df["temp"])
    plt.title(f"Daily Mean Temperature – {label} (1981–2014)")
    plt.xlabel("date")
    plt.ylabel("temperature (K)")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{label}_daily_temp_1981_2014.png", dpi=150)
    plt.close()

    # Yearly means & anomalies
    daily_df["year"] = daily_df["time"].dt.year
    annual = daily_df.groupby("year")["temp"].mean().reset_index()
    baseline = annual["temp"].mean()
    annual["anomaly"] = annual["temp"] - baseline

    annual.to_csv(
        processed_dir / f"{label}_annual_temps_anomalies_1981_2014.csv",
        index=False
    )

    # Yearly anomaly plot
    plt.figure(figsize=(8,4))
    plt.axhline(0, color="black")
    plt.bar(annual["year"], annual["anomaly"])
    plt.title(f"{label} – Yearly Temperature Anomalies (1981–2014)")
    plt.xlabel("year")
    plt.ylabel("temperature anomaly (K)")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{label}_yearly_anomalies_1981_2014.png", dpi=150)
    plt.close()

    # Climate stripes (historical)
    anomalies = annual["anomaly"].values
    n_boxes = len(anomalies)

    fig = plt.figure(figsize=(10, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    rects = [Rectangle((i, 0), 1, 1) for i in range(n_boxes)]
    collection = PatchCollection(rects)
    collection.set_array(anomalies)
    collection.set_cmap(stripe_cmap)
    collection.set_clim(-3, 3)

    ax.add_collection(collection)
    ax.set_xlim(0, n_boxes)
    ax.set_ylim(0, 1)

    plt.savefig(
        plots_dir / f"{label}_climate_stripes_1981_2014.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

    return annual, baseline

# ---------------------------------------------
# Function: future scenario processing
# ---------------------------------------------

def process_future_scenario(country_names, label, scenario_folder, scenario_name, baseline_temp, annual_hist):

    if isinstance(country_names, str):
        country_names = [country_names]

    print(f"\n=== FUTURE SCENARIO {scenario_name.upper()} for {label} ===")

    subset = gdf[gdf[country_col].isin(country_names)]
    geom = subset.geometry.values

    scen_dir = BASE_DIR / "data" / scenario_folder
    scen_files = sorted(scen_dir.glob("*.nc"))

    clipped = []
    for f in scen_files:
        print("Processing:", f.name)
        ds = xr.open_dataset(f)
        ds = ds.rio.write_crs("EPSG:4326")
        ds = ds.rio.clip(geom, subset.crs)
        clipped.append(ds)

    ds_future = xr.concat(clipped, dim="time")

    # Daily → yearly
    daily = ds_future["tas"].mean(dim=["lat", "lon"])
    df_daily = daily.to_dataframe().reset_index()
    df_daily.rename(columns={"tas": "temp"}, inplace=True)
    df_daily["year"] = df_daily["time"].dt.year

    annual_future = df_daily.groupby("year")["temp"].mean().reset_index()

    # anomalies using historical baseline
    annual_future["anomaly"] = annual_future["temp"] - baseline_temp

    annual_future.to_csv(
        processed_dir / f"{label}_{scenario_name}_annual_temps_anomalies_2015_2050.csv",
        index=False
    )

    # future anomaly plot
    plt.figure(figsize=(8,4))
    plt.axhline(0, color="black")
    plt.bar(annual_future["year"], annual_future["anomaly"])
    plt.title(f"{label} – {scenario_name.upper()} Anomalies (2015–2050)")
    plt.xlabel("year")
    plt.ylabel("anomaly (K)")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{label}_{scenario_name}_yearly_anomalies_2015_2050.png", dpi=150)
    plt.close()

    # combined historical + future stripes
    hist_anom = annual_hist["anomaly"].values
    fut_anom = annual_future["anomaly"].values
    all_anom = np.concatenate([hist_anom, fut_anom])

    n_boxes = len(all_anom)

    fig = plt.figure(figsize=(12, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    rects = [Rectangle((i, 0), 1, 1) for i in range(n_boxes)]
    collection = PatchCollection(rects)
    collection.set_array(all_anom)
    collection.set_cmap(stripe_cmap)
    collection.set_clim(-3, 3)

    ax.add_collection(collection)
    ax.set_xlim(0, n_boxes)
    ax.set_ylim(0, 1)

    plt.savefig(
        plots_dir / f"{label}_climate_stripes_{scenario_name}_1981_2050.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

    return annual_future

# ---------------------------------------------
# RUN PROCESSING FOR ALL COUNTRIES & REGION
# ---------------------------------------------

targets = {
    "netherlands": ["Netherlands"],
    "belgium": ["Belgium"],
    "spain": ["Spain"],
    "sweden": ["Sweden"],
    "alpine_region": ["Switzerland", "Austria", "Italy"],
}

all_targets = {}

# Historical first
for label, countries in targets.items():
    hist_df, baseline = process_country_or_region(countries, label=label)
    all_targets[label] = {"names": countries, "hist": hist_df, "baseline": baseline}

# Future scenarios
for label, info in all_targets.items():
    process_future_scenario(
        country_names=info["names"],
        label=label,
        scenario_folder="scenario-SSP1-2.6",
        scenario_name="ssp126",
        baseline_temp=info["baseline"],
        annual_hist=info["hist"],
    )
    process_future_scenario(
        country_names=info["names"],
        label=label,
        scenario_folder="scenario-SSP5-8.5",
        scenario_name="ssp585",
        baseline_temp=info["baseline"],
        annual_hist=info["hist"],
    )

print("\n=== ALL PROCESSING COMPLETED SUCCESSFULLY ===\n")
