"""Serial communication with ESP32 for haptic motor control."""

import serial
import logging

logger = logging.getLogger(__name__)


class ESP32Handler:
    """ESP32 serial communicator for motor control via message format: L,C,R,0,0\\n"""
    
    def __init__(self, port="/dev/ttyUSB1", baudrate=9600, timeout=1.0):
        """Initialize ESP32 connection."""
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self._connected = False
    
    def connect(self):
        """Establish serial connection to ESP32."""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout
            )
            self._connected = True
            logger.info(f"Connected to ESP32 on {self.port} at {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to connect to ESP32: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Close serial connection."""
        if self.serial and self._connected:
            try:
                self.serial.close()
                self._connected = False
                logger.info("Disconnected from ESP32")
            except Exception as e:
                logger.error(f"Error closing serial connection: {e}")
    
    def send_motor_command(self, left, center, right, tof_left=0, tof_right=0):
        """Send motor values (0-100 int) to ESP32."""
        if not self._connected:
            return False
        
        left = max(0, min(100, int(left)))
        center = max(0, min(100, int(center)))
        right = max(0, min(100, int(right)))
        tof_left = max(0, min(100, int(tof_left)))
        tof_right = max(0, min(100, int(tof_right)))
        
        message = f"{left},{center},{right},{tof_left},{tof_right}\n"
        
        try:
            self.serial.write(message.encode())
            logger.debug(f"Sent: {message.strip()}")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def is_connected(self):
        """Return connection status."""
        return self._connected and self.serial and self.serial.is_open
