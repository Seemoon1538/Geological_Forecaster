"""Data loading and validation module."""

import pandas as pd
import numpy as np
import pandera.pandas as pa
from pandera import DataFrameSchema, Column, Check
import geopandas as gpd
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and cleaning of geological data."""

    SCHEMA = DataFrameSchema({
        "x": Column(float, nullable=False),  
        "y": Column(float, nullable=False),  
        "volume": Column(float, Check.gt(0), nullable=False),
        "ore_type": Column(str, nullable=False)
    })

    @classmethod
    def load_data(cls, input_path: str, normalize: bool = False) -> pd.DataFrame:
        """Load CSV data and validate.

        Args:
            input_path: Path to CSV file (x=lat, y=lon).
            normalize: Whether to normalize x,y to [0,1] (default: False for lat/lon).

        Returns:
            Cleaned DataFrame.

        Raises:
            ValueError: If validation fails or data is empty.
        """
        try:
            df = pd.read_csv(input_path)
            if isinstance(df, pd.Series):
                df = df.to_frame()
            assert isinstance(df, pd.DataFrame), "Loaded data is not a DataFrame"
            logger.info(f"Loaded {len(df)} rows from {input_path}")
            logger.debug(f"Raw coordinates: lat_min={df['x'].min()}, lat_max={df['x'].max()}, lon_min={df['y'].min()}, lon_max={df['y'].max()}")
        except Exception as e:
            logger.error(f"Error loading {input_path}: {e}")
            raise

        
        df = df.dropna()
        if df.empty:
            logger.error("DataFrame is empty after dropna")
            raise ValueError("DataFrame is empty after cleaning")

        z_scores = np.abs((df['volume'] - df['volume'].mean()) / df['volume'].std())
        df = df[z_scores < 3]
        logger.info(f"After cleaning: {len(df)} rows")

        
        try:
            validated_df: pd.DataFrame = cls.SCHEMA.validate(df)
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise ValueError("Data does not match schema: volume > 0, ore_type str")

        
        if normalize:
            validated_df['x'] = (validated_df['x'] - validated_df['x'].min()) / (validated_df['x'].max() - validated_df['x'].min())
            validated_df['y'] = (validated_df['y'] - validated_df['y'].min()) / (validated_df['y'].max() - validated_df['y'].min())
            logger.debug("Coordinates normalized to [0,1]")

        return validated_df

    @classmethod
    def to_geodataframe(cls, df: pd.DataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """Convert DataFrame to GeoDataFrame.

        Args:
            df: DataFrame with x, y columns (x=lat, y=lon).
            crs: Coordinate Reference System (default: EPSG:4326).

        Returns:
            GeoDataFrame with geometry column.
        """
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df['y'], df['x']), 
            crs=crs
        )
        logger.debug(f"GeoDataFrame created: CRS={gdf.crs}, bounds={gdf.total_bounds}")
        return gdf

    @classmethod
    def prepare_features(cls, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for ML (legacy support).

        Args:
            df: Cleaned DataFrame (x=lat, y=lon).

        Returns:
            Tuple of (coords, volumes) as numpy arrays.
        """
        coords = df[['x', 'y']].to_numpy()  
        volumes = df['volume'].to_numpy()
        return coords, volumes