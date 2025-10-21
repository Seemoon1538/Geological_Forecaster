"""Main orchestration script."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

from .config import Config, logger
from .data_loader import DataLoader
from .gis_loader import load_gis_data
from .forecaster import Forecaster
from .visualizer import Visualizer
import geopandas as gpd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def main():
    """Run the forecaster pipeline."""
    parser = argparse.ArgumentParser(description="Geological Forecaster")
    parser.add_argument("--input", required=True, help="Input path (CSV, Shapefile, GeoJSON, or WMS URL)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--wms-layer", default=None, help="WMS layer name (optional)")
    parser.add_argument("--bbox", nargs=4, type=float, default=None, help="Bounding box: minx miny maxx maxy (optional)")
    parser.add_argument("--eps", type=float, default=0.01, help="DBSCAN eps distance (degrees, ~1km)")
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN min samples per cluster")
    args = parser.parse_args()

    # Config
    config_dict = {
        "input_path": args.input,
        "output_dir": args.output,
        "eps": args.eps,
        "min_samples": args.min_samples,
        "rbf_kernel": "thin_plate_spline",
        "rf_estimators": 100,
        "wms_layer": args.wms_layer,
        "bbox": tuple(args.bbox) if args.bbox else None,
        "crs": "EPSG:4326"
    }
    config = Config.from_dict(config_dict)
    Path(config.output_dir).mkdir(exist_ok=True)

    # Load data
    if args.input.endswith(('.csv', '.txt')):
        df = DataLoader.load_data(config.input_path, normalize=False)
        gdf = DataLoader.to_geodataframe(df, crs=config.crs)
        raster = None
    else:
        gdf, raster = load_gis_data(config.input_path, wms_layer=config.wms_layer, bbox=config.bbox)
        if gdf.empty and raster:
            logger.warning("No vector data loaded from WMS, only raster available")
            # Optionally load points from CSV for WMS background
            if config.input_path.startswith(('http', 'https')):
                logger.info("Loading points from CSV for WMS")
                df = DataLoader.load_data('data/drill_points.csv', normalize=False)
                gdf = DataLoader.to_geodataframe(df, crs=config.crs)

    # Validate GeoDataFrame
    if gdf.empty:
        raise ValueError("No valid data loaded for forecasting")

    # Forecast
    forecaster = Forecaster(
        eps=config.eps,
        min_samples=config.min_samples,
        rbf_kernel=config.rbf_kernel,
        rf_estimators=config.rf_estimators
    )
    clusters, labels, deposit_center, deposit_ore_type = forecaster.fit_predict(gdf)

    # Save JSON
    json_data = forecaster.to_json(clusters, deposit_center, deposit_ore_type)
    output_json = f"{config.output_dir}/forecast.json"
    with open(output_json, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Saved {output_json} with {len(clusters)} clusters, deposit at {deposit_center}")

    # Visualize
    visualizer = Visualizer()
    cluster_centers = [(c.center, c.predicted_volume) for c in clusters]
    visualizer.plot_clusters(gdf[['x', 'y']].to_numpy(), labels, cluster_centers, config.output_dir)
    if forecaster.interpolator:
        visualizer.plot_heatmap(forecaster.interpolator, gdf, config.output_dir)
    visualizer.plot_interactive(gdf, json_data, config.output_dir, raster=raster)

if __name__ == "__main__":
    main()