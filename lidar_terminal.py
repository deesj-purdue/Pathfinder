from lidar.api import Lidar
import time

def terminal_lidar():
    """
    Live LIDAR printing to terminal.
    """
    lidar = Lidar()
    lidar.start()
    try:
        while True:
            time.sleep(1)
            data = lidar.read()
            for angle, (distance, quality) in data:
                if distance is not None:
                    distance_str = f"{distance:10.2f}"
                else:
                    distance_str = "      None"
                print(f"{angle:6.2f} deg: {distance_str} mm, q {quality}")
                
    except KeyboardInterrupt:
        lidar.stop()


if __name__ == "__main__":
    terminal_lidar()
