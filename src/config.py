"""Configuration module."""

from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class Config:
    input_path: str
    output_dir: str
    eps: float
    min_samples: int
    rbf_kernel: str
    rf_estimators: int
    wms_layer: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    crs: str = "EPSG:4326"

    @classmethod
    def from_dict(cls, config_dict: dict):
        return cls(**config_dict)