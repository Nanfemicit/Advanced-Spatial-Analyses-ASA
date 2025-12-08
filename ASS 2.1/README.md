# ASS 2.1 – Climate Stripes

This assignment creates **climate stripes** to visualise how temperature has changed over time for different countries, using historical and future climate projections. The work is done in Python using gridded temperature data and country boundary data.


## Code structure

Main file:

- `climatestripes_dala.py`  
  Contains the code to:
  - Load and clip temperature data to selected country polygons.
  - Compute daily country-average temperature.
  - Aggregate to yearly means and compute anomalies relative to 1981–2014.
  - Plot climate stripes using Matplotlib and a custom colour map.

The script is organised into reusable functions so that:
- The same workflow can be applied to different countries.
- The same functions can be reused for historical and future scenarios.
- The code can handle either a single country or a list of countries.

---

## Skills and tools demonstrated

- Working with **NetCDF** climate data using `xarray`, `rioxarray`, and `dask`.
- Spatial operations and clipping with **GeoPandas** and GADM boundaries.
- Numerical calculations with **NumPy**.
- Building reusable analysis functions in Python.
- Plotting custom visualisations (climate stripes) with **Matplotlib**.

---

## How to run

From inside the `ASS 2.1` folder:

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # on Windows
