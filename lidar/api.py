import lidar.backend as backend
import threading
import time
from typing import Dict, Tuple, Optional, Iterator

LIDAR_PORT = "COM5"
LIDAR_BAUDRATE = 460800


class LidarData:
    """
    Container for LIDAR scan results.
    Dict of all angles (0-360 deg) -> (distance in mm, quality)
    """

    def __init__(self, data: Dict[float, Tuple[Optional[float], int]]):
        self.data: Dict[float, Tuple[Optional[float], int]] = {}
        for angle, (distance, quality) in sorted(data.items()):
            self.data[float(angle)] = (
                float(distance) if distance is not None else None,
                int(quality),
            )

    def __iter__(self) -> Iterator[Tuple[float, Tuple[Optional[float], int]]]:
        return iter(self.data.items())

    def __getitem__(self, angle: float) -> Tuple[Optional[float], int]:
        return self.data[angle]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()


class Lidar:
    """
    Clean API for accessing LIDAR functions from other programs.
    Usage:
        lidar = Lidar()
        lidar.start()
        data = lidar.read()  # returns LidarData
        lidar.stop()
    """

    def __init__(self, port: str = LIDAR_PORT, baudrate: int = LIDAR_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the LIDAR scanner in the background."""
        with self._lock:
            if not self._running:
                backend.start_scanner(self.port, self.baudrate)
                self._running = True

    def read(self) -> LidarData:
        """Get the most recent scan data as a LidarData object."""
        return LidarData(backend.get_latest_reads())

    def stop(self) -> None:
        """Stop the LIDAR scanner and release resources."""
        with self._lock:
            if self._running:
                backend.stop_scanner()
                self._running = False


# Example usage
if __name__ == "__main__":
    lidar = Lidar()
    lidar.start()
    try:
        while True:
            time.sleep(1)
            data = lidar.read()
            for angle, (distance, quality) in data:
                print(f"{angle:.2f}: {distance}, {quality}")
    except KeyboardInterrupt:
        lidar.stop()
