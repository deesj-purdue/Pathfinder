from haptic_zones import get_haptic_zones
from esp32_handler import ESP32Handler
import time


def haptic_feedback(esp32_port="/dev/ttyUSB1"):
    """
    Reads LIDAR, gets motor values (0-100), displays to terminal, and sends to ESP32 at 5Hz.
    """
    from lidar.api import Lidar
    
    lidar = Lidar()
    lidar.start()
    
    # Initialize ESP32 connection
    esp32 = ESP32Handler(port=esp32_port, baudrate=9600)
    esp32_connected = esp32.connect()
    
    if not esp32_connected:
        print(f"Warning: Could not connect to ESP32 on {esp32_port}")
    
    try:
        while True:
            time.sleep(0.2)  # 5 Hz update rate
            
            data = lidar.read()
            motor_values = get_haptic_zones(data)
            
            left = motor_values['left']
            center = motor_values['center']
            right = motor_values['right']
            
            # Send to ESP32 if connected
            if esp32_connected:
                esp32.send_motor_command(left, center, right)
            
            # Print motor values
            print(f"Left: {left:3d} | Center: {center:3d} | Right: {right:3d}")
            
    except KeyboardInterrupt:
        lidar.stop()
        if esp32_connected:
            esp32.disconnect()
        print("\nHaptic feedback stopped.")


if __name__ == "__main__":
    haptic_feedback()
