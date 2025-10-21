# check_shapefile.py
import geopandas as gpd

gdf = gpd.read_file('data/SD-point.shp')
print("Columns in SD-point.shp:", gdf.columns.tolist())
print("Sample data:\n", gdf.head())
print("Bounds:", gdf.total_bounds)