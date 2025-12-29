"""GIS data loading module."""

import geopandas as gpd
import pandera.pandas as pa
from .data_loader import DataLoader
from owslib.wms import WebMapService
from rasterio.io import MemoryFile
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def load_gis_data(input_path: str, wms_layer: Optional[str] = None, bbox: Optional[Tuple[float, float, float, float]] = None) -> Tuple[gpd.GeoDataFrame, Optional[MemoryFile]]:
    """Load GIS data from Shapefile, GeoJSON, or WMS.

    Args:
        input_path: Path to file (Shapefile, GeoJSON) or WMS URL.
        wms_layer: WMS layer name (optional).
        bbox: Bounding box (minx, miny, maxx, maxy) for WMS or filtering (optional).

    Returns:
        Tuple of (GeoDataFrame, optional raster).
    """
    try:
        if input_path.endswith(('.shp', '.geojson')):
            gdf = gpd.read_file(input_path)
           
            column_mapping = {
                'GRADE_AU': 'volume',
                'TONNAGE': 'volume',
                'DEPOSIT_TYPE': 'ore_type',
                'COMMODITY': 'ore_type',
            }
            
            available_columns = set(gdf.columns)
            applied_mapping = {k: v for k, v in column_mapping.items() if k in available_columns}
            gdf = gdf.rename(columns=applied_mapping)
            
            
            expected_columns = {'volume', 'ore_type'}
            missing_columns = expected_columns - set(gdf.columns)
            if missing_columns:
                logger.warning(f"Missing columns in GIS data: {missing_columns}. Adding synthetic data.")
                
                if 'volume' in missing_columns:
                    import numpy as np
                    gdf['volume'] = np.random.uniform(7.0, 10.0, size=len(gdf))  
                if 'ore_type' in missing_columns:
                    gdf['ore_type'] = 'vein'  
                    logger.info(f"Added synthetic columns: {missing_columns}")
            
            
            homestake_bbox = (-103.8, 44.3, -103.7, 44.4)  
            if not gdf.empty:
                gdf = gdf.cx[homestake_bbox[0]:homestake_bbox[2], homestake_bbox[1]:homestake_bbox[3]]
                logger.info(f"Filtered to Homestake Mine bbox {homestake_bbox}: {len(gdf)} rows")
                if gdf.empty:
                    logger.warning("No points found within Homestake Mine bbox")
            
            
            gdf['x'] = gdf.geometry.y  
            gdf['y'] = gdf.geometry.x  
            logger.info(f"Loaded GIS data from {input_path}, {len(gdf)} rows")
            logger.debug(f"GIS bounds: {gdf.total_bounds}")
            return gdf, None
        elif input_path.startswith(('http', 'https')):
            wms = WebMapService(input_path)
            if wms_layer and bbox:
                img = wms.getmap(layers=[wms_layer], srs='EPSG:4326', bbox=bbox, size=(256, 256), format='image/png')
                raster = MemoryFile(img.read())
                logger.info(f"Loaded WMS raster from {input_path}, layer={wms_layer}, bbox={bbox}")
                
                df = DataLoader.load_data('data/drill_points.csv', normalize=False)
                gdf = DataLoader.to_geodataframe(df, crs='EPSG:4326')
                return gdf, raster
            else:
                logger.error("WMS requires layer and bbox")
                raise ValueError("WMS requires layer and bbox")
        else:
            logger.error(f"Unsupported file format: {input_path}")
            raise ValueError(f"Unsupported file format: {input_path}")
    except Exception as e:
        logger.error(f"Error loading GIS data: {e}")
        raise