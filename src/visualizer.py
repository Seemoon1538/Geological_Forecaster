"""Visualization module."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
import geopandas as gpd
from typing import List, Tuple, Optional, Callable
import logging
from rasterio.io import MemoryFile
import rasterio
from folium.raster_layers import ImageOverlay

logger = logging.getLogger(__name__)

class Visualizer:
    """Handles plotting clusters and interpolation heatmap."""

    @classmethod
    def plot_clusters(cls, coords: np.ndarray, labels: np.ndarray, clusters: List[Tuple], output_dir: str):
        """Plot scatter with clusters and save PNG.

        Args:
            coords: (N,2) x,y (lat,lon for geo data).
            labels: Cluster labels.
            clusters: List of (center, pred_vol).
            output_dir: Output path.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='viridis', s=50, label='Points')
        
        if clusters:
            for i, (center, pred_vol) in enumerate(clusters):
                ax.scatter(center[0], center[1], c='red', s=100, marker='x', label=f'Cluster {i}: {pred_vol:.2f}')
        
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Longitude')
        ax.set_title('Geological Clusters (DBSCAN)')
        plt.colorbar(scatter, ax=ax, label='Cluster Label')
        if clusters:
            plt.legend()
        plt.savefig(f"{output_dir}/clusters.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved clusters.png")

    @classmethod
    def plot_heatmap(cls, interpolator: Callable, gdf: gpd.GeoDataFrame, output_dir: str):
        """Plot RBF interpolation heatmap.

        Args:
            interpolator: Fitted RBFInterpolator.
            gdf: GeoDataFrame with x, y columns (x=lat, y=lon).
            output_dir: Output path.
        """
        coords = gdf[['x', 'y']].to_numpy()
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid_coords = np.c_[xx.ravel(), yy.ravel()]
        z = interpolator(grid_coords).reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        sns.heatmap(z, xticklabels=False, yticklabels=False, cmap='YlOrRd')
        plt.title('Volume Interpolation Heatmap (RBF)')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.savefig(f"{output_dir}/heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved heatmap.png")

    @classmethod
    def plot_interactive(cls, gdf: gpd.GeoDataFrame, json_data: dict, output_dir: str, raster: Optional[MemoryFile] = None):
        """Plot interactive map with Folium.

        Args:
            gdf: GeoDataFrame with points (x=lat, y=lon).
            json_data: Forecast JSON with clusters, deposit_center, deposit_ore_type.
            output_dir: Output path.
            raster: Optional WMS raster for map background.
        """
        if gdf.empty and json_data['deposit_center'] == [0.0, 0.0]:
            logger.error("No data or clusters to plot on map")
            return
        
        # Log coordinates for debugging
        logger.debug(f"GeoDataFrame coords: x_mean={gdf['x'].mean()}, y_mean={gdf['y'].mean()}")
        logger.debug(f"Deposit center: {json_data['deposit_center']}")

        # Initialize map at mean coordinates or deposit center
        if gdf.empty:
            center = [json_data['deposit_center'][1], json_data['deposit_center'][0]]  # lat,lon
        else:
            center = [gdf['x'].mean(), gdf['y'].mean()]  # lat,lon
        logger.debug(f"Map center: {center}")
        m = folium.Map(location=center, zoom_start=12, tiles='OpenStreetMap')

        # Add points
        for _, row in gdf.iterrows():
            logger.debug(f"Point: [lat={row['x']}, lon={row['y']}]")
            folium.CircleMarker(
                [row['x'], row['y']],  # lat,lon
                radius=5,
                popup=f"Volume: {row['volume']:.2f}, Type: {row['ore_type']}",
                color='blue',
                fill=True,
                fill_opacity=0.6
            ).add_to(m)

        # Add deposit center
        logger.debug(f"Deposit marker: [lat={json_data['deposit_center'][1]}, lon={json_data['deposit_center'][0]}]")
        folium.Marker(
            [json_data['deposit_center'][1], json_data['deposit_center'][0]],  # lat,lon
            popup=f"Deposit: {json_data['predicted_total_volume']:.2f} tons, {json_data['deposit_ore_type']}",
            icon=folium.Icon(color='red', icon='star')
        ).add_to(m)

        # Add WMS raster
        if raster:
            with rasterio.open(raster) as dataset:
                img_array = dataset.read(1)  # Read first band
                bounds = dataset.bounds  # (minx, miny, maxx, maxy)
                img_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]  # [[miny, minx], [maxy, maxx]]
                ImageOverlay(
                    image=img_array,
                    bounds=img_bounds,
                    opacity=0.6
                ).add_to(m)
            logger.info("Added WMS raster to map")

        m.save(f"{output_dir}/interactive_map.html")
        logger.info("Saved interactive_map.html")