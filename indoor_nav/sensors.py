"""Sensor functions for LiDAR and ToF. Returns None until hardware is wired up."""

import numpy as np

from indoor_nav.config import DIST_NEAR_M, DIST_FAR_M

try:
    from lidar.api import Lidar
    from haptic_zones import get_haptic_zones
    _lidar_available = True
except (ImportError, ModuleNotFoundError):
    _lidar_available = False


def read_lidar_sectors():
    """Read LiDAR sectors (L, C, R) in metres. Returns [left_m, center_m, right_m] or None."""
    if not _lidar_available:
        return None
    
    try:
        lidar = Lidar()
        lidar.start()
        data = lidar.read()
        haptic = get_haptic_zones(data)
        lidar.stop()
        
        # Extract closest distance in each sector (returned in mm)
        left_dist_mm = haptic['left_distance']
        center_dist_mm = haptic['center_distance']
        right_dist_mm = haptic['right_distance']
        
        # Convert to meters (1000 mm = 1 m), default to max distance if None
        left_m = left_dist_mm / 1000.0 if left_dist_mm is not None else 3.0
        center_m = center_dist_mm / 1000.0 if center_dist_mm is not None else 3.0
        right_m = right_dist_mm / 1000.0 if right_dist_mm is not None else 3.0
        
        return [left_m, center_m, right_m]
    except Exception:
        return None


def read_tof_sectors():
    """Read ToF sectors (L, C, R) in metres. Returns None until hardware is connected."""
    # Hook up I2C ToF reads
    return None


def distance_to_risk(dist_m, near_m=DIST_NEAR_M, far_m=DIST_FAR_M):
    """Linear ramp from distance in metres to 0-1 risk score."""
    if dist_m is None:
        return 0.0
    return float(np.clip(1.0 - (dist_m - near_m) / (far_m - near_m), 0.0, 1.0))
