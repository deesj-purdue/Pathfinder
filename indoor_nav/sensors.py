"""Sensor functions for LiDAR and ToF. Returns None until hardware is wired up."""

import numpy as np

from indoor_nav.config import DIST_NEAR_M, DIST_FAR_M

# Optional LiDAR integration
try:
    from lidar.api import Lidar
    from haptic_zones import get_haptic_zones
    _lidar_available = True
except (ImportError, ModuleNotFoundError):
    _lidar_available = False


def read_lidar_sectors():
    """Read LiDAR sectors (L, C, R) in metres. Returns None until hardware is connected."""
    if not _lidar_available:
        return None
    
    try:
        lidar = Lidar()
        lidar.start()
        data = lidar.read()
        haptic = get_haptic_zones(data)
        lidar.stop()
        
        # Convert haptic vibration values to distances in meters
        # Reverse the quadratic mapping from haptic_zones
        # vibration = x^2, so x = sqrt(vibration)
        # x = (3.0 - distance) / (3.0 - 1.5)
        # distance = 3.0 - x * (3.0 - 1.5)
        left_vib = haptic['left']
        center_vib = haptic['center']
        right_vib = haptic['right']
        
        def vibration_to_distance(vib):
            """Convert vibration intensity back to distance in meters."""
            if vib == 0.0:
                return 3.0  # max distance
            x = np.sqrt(vib)
            return 3.0 - x * 1.5
        
        return [
            vibration_to_distance(left_vib),
            vibration_to_distance(center_vib),
            vibration_to_distance(right_vib),
        ]
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
