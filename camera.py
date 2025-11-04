import cv2
import threading
import time
import subprocess
import os
import glob
from shutil import which

#!/usr/bin/env python3
"""
camera.py

USB FHD camera helper for Jetson Nano (USBFHD08S or similar USB UVC cameras).
Provides a simple threaded capture class, basic property setters (with v4l2-ctl fallback),
and a small CLI example to preview and save frames.

Dependencies:
- Python 3
- OpenCV (cv2) built with V4L2 support
- v4l2-ctl (optional, for robust property control)
"""


class Camera:
    def __init__(self, device=None, width=1920, height=1080, fps=30, backend=cv2.CAP_V4L2):
        # device: integer index or path like "/dev/video0". If None, auto-find first working device.
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend

        self._cap = None
        self._running = False
        self._thread = None
        self.frame = None
        self.timestamp = None

        if self.device is None:
            self.device = self._auto_find_device()

    def _auto_find_device(self):
        for dev in sorted(glob.glob('/dev/video*')):
            idx = int(dev.replace('/dev/video', ''))
            cap = cv2.VideoCapture(idx, self.backend)
            if cap.isOpened():
                cap.release()
                return dev
        # fallback to index 0
        return 0

    def open(self):
        if isinstance(self.device, str) and self.device.startswith('/dev/video'):
            # open by numeric index extracted from device path
            idx = int(self.device.replace('/dev/video', ''))
            self._cap = cv2.VideoCapture(idx, self.backend)
        else:
            self._cap = cv2.VideoCapture(int(self.device), self.backend)

        if not self._cap or not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device}")

        # try to set properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self._cap.set(cv2.CAP_PROP_FPS, float(self.fps))

        # warm up
        time.sleep(0.1)
        # grab one frame
        ret, frame = self._cap.read()
        if ret:
            self.frame = frame
            self.timestamp = time.time()

    def start(self):
        if self._running:
            return
        if not self._cap or not self._cap.isOpened():
            self.open()
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                # brief pause to avoid tight loop if camera disconnects
                time.sleep(0.02)
                continue
            self.frame = frame
            self.timestamp = time.time()

    def read(self):
        # Returns (frame, timestamp) or (None, None)
        return self.frame, self.timestamp

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def release(self):
        self.stop()

    # Basic property controls. Some cameras ignore cv2 settings; v4l2-ctl fallback tries to set using device controls.
    def set_property(self, prop, value):
        # prop is one of 'exposure', 'gain', 'brightness', 'contrast', 'white_balance_temperature'
        cv_props = {
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'gain': cv2.CAP_PROP_GAIN,
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'white_balance_temperature': cv2.CAP_PROP_WHITE_BALANCE_BLUE_U  # not always consistent
        }
        if prop in cv_props:
            try:
                ok = self._cap.set(cv_props[prop], float(value))
                if ok:
                    return True
            except Exception:
                pass
        # fallback to v4l2-ctl if available and device path is known
        dev_path = self._resolve_dev_path()
        if dev_path and shutil_which('v4l2-ctl'):
            ctrl_map = {
                'exposure': 'exposure_absolute',
                'gain': 'gain',
                'brightness': 'brightness',
                'contrast': 'contrast',
                'white_balance_temperature': 'white_balance_temperature'
            }
            if prop in ctrl_map:
                cmd = ['v4l2-ctl', '-d', dev_path, '--set-ctrl', f"{ctrl_map[prop]}={int(value)}"]
                try:
                    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                except subprocess.CalledProcessError:
                    return False
        return False

    def _resolve_dev_path(self):
        if isinstance(self.device, str) and self.device.startswith('/dev/video'):
            return self.device
        # try to find index -> path
        try:
            idx = int(self.device)
            path = f"/dev/video{idx}"
            if os.path.exists(path):
                return path
        except Exception:
            pass
        return None

def shutil_which(name):
    # local copy to avoid importing shutil at top-level if not needed
    return which(name)

if __name__ == "__main__":
    # Simple CLI: preview and save a frame when pressing 's'
    cam = Camera(device=None, width=1920, height=1080, fps=30)
    try:
        cam.start()
        print("Camera started. Press 's' to save a frame, 'q' to quit.")
        while True:
            frame, ts = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            # downscale for display if too large
            h, w = frame.shape[:2]
            display = frame
            if w > 1280:
                scale = 1280.0 / w
                display = cv2.resize(frame, (int(w * scale), int(h * scale)))
            cv2.imshow("Camera Preview", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                fname = f"frame_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print("Saved", fname)
            elif key == ord('q'):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()