# ASS 2.2 – Optimizing the Dutch Energy Grid

This assignment analyses how weather conditions influence the capacity of the Dutch high-voltage power grid. The work combines weather station data with power line infrastructure to assess how temperature and wind speed affect transmission capacity.

## Code structure

Main workflow:

- Load weather data from the KNMI API and convert it into a usable format  
- Merge weather station data with spatial coordinates  
- Link each power line to the nearest weather station  
- Calculate capacity factors based on temperature and wind speed  
- Adjust wind speed using forest cover data  
- Visualise spatial variation in grid capacity  

## Skills and tools demonstrated

- Working with API data and JSON in Python  
- Data processing using pandas and NumPy  
- Spatial analysis with GeoPandas  
- Nearest-neighbour spatial matching  
- Raster processing and aggregation  
- Visualisation using Matplotlib  
