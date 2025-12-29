"""ML forecasting module: clustering, volume interpolation, and ore type classification."""

import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.interpolate import RBFInterpolator
import geopandas as gpd
import pandas as pd
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Cluster:
    center: Tuple[float, float]
    points: np.ndarray
    predicted_volume: float

class Forecaster:
    def __init__(self, eps: float = 0.5, min_samples: int = 3, rbf_kernel: str = "thin_plate_spline", rf_estimators: int = 100):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.interpolator = None
        self.label_encoder = LabelEncoder()
        self.rf_classifier = RandomForestClassifier(n_estimators=rf_estimators)
        self.eps = eps
        self.min_samples = min_samples
        self.rbf_kernel = rbf_kernel

    def fit_predict(self, gdf: gpd.GeoDataFrame) -> Tuple[List[Cluster], np.ndarray, Tuple[float, float], str]:
        """Fit DBSCAN and RF, predict clusters and ore type."""
        coords = gdf[['x', 'y']].to_numpy()  
        volumes = gdf['volume'].to_numpy()
        ore_types = gdf['ore_type'].astype(str).to_numpy()

        
        labels = self.dbscan.fit_predict(coords)
        logger.info(f"DBSCAN labels: {labels}")
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"Found {n_clusters} clusters")

        
        self.label_encoder.fit(ore_types)
        encoded_ore_types = self.label_encoder.transform(ore_types)
        self.rf_classifier.fit(coords, encoded_ore_types)

        clusters = []
        for label in set(labels) - {-1}:
            mask = labels == label
            cluster_points = coords[mask]
            if len(cluster_points) > 0:
                center = tuple(np.mean(cluster_points, axis=0)) 
                pred_volume = np.sum(volumes[mask])
                clusters.append(Cluster(center=center, points=cluster_points, predicted_volume=pred_volume))
                logger.debug(f"Cluster {label}: center={center}, volume={pred_volume}")

        
        if clusters:
            max_volume_cluster = max(clusters, key=lambda c: c.predicted_volume)
            deposit_center = max_volume_cluster.center
            cluster_X = max_volume_cluster.points
            pred_types = self.label_encoder.inverse_transform(self.rf_classifier.predict(cluster_X))
            deposit_ore_type = str(pd.Series(pred_types).mode().iloc[0])
        else:
            logger.warning("No clusters found, deposit center set to mean coordinates")
            deposit_center = (float(gdf['x'].mean()), float(gdf['y'].mean())) if not gdf.empty else (0.0, 0.0)
            deposit_ore_type = "unknown"

       
        self.interpolator = RBFInterpolator(coords, volumes, kernel=self.rbf_kernel)

        logger.debug(f"Deposit center: {deposit_center}, ore_type: {deposit_ore_type}")
        return clusters, labels, deposit_center, deposit_ore_type

    def to_json(self, clusters: List[Cluster], deposit_center: Tuple[float, float], deposit_ore_type: str) -> Dict[str, Any]:
        """Convert forecast to JSON."""
        json_data = {
            "clusters": [
                {
                    "center": [float(c.center[1]), float(c.center[0])], 
                    "points": [[float(p[1]), float(p[0])] for p in c.points.tolist()],  
                    "predicted_volume": float(c.predicted_volume)
                } for c in clusters
            ],
            "deposit_center": [float(deposit_center[1]), float(deposit_center[0])],  
            "deposit_ore_type": deposit_ore_type,
            "predicted_total_volume": float(sum(c.predicted_volume for c in clusters))
        }
        logger.debug(f"JSON data: {json_data}")
        return json_data