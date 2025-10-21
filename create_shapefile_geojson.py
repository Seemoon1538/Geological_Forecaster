# create_shapefile_geojson.py
import pandas as pd
import geopandas as gpd

# Load CSV
df = pd.read_csv('data/drill_points.csv')

# Create GeoDataFrame (x=lat, y=lon)
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['y'], df['x']),  # lon,lat
    crs="EPSG:4326"
)

# Save to Shapefile
gdf.to_file('data/drill_points.shp')
print("Created data/drill_points.shp")

# Save to GeoJSON
gdf.to_file('data/drill_points.geojson', driver='GeoJSON')
print("Created data/drill_points.geojson")