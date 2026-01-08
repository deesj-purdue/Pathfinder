"""
Web-based Stream Detection - YOLO + MiDaS Fusion

Serves detection results over HTTP so you can view from any laptop/device on the network.
Access at: http://<server-ip>:5000

Objects are color-coded by depth:
- Green: Far objects
- Yellow: Medium distance
- Red: Close objects
"""

import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import sys
import json
from pathlib import Path
from threading import Thread, Lock
from queue import Queue
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
import base64
import io

# Setup paths and device
root = Path(__file__).resolve().parent
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 model
sys.path.insert(0, str(root / "yolov5"))
model_path = root / "yolov5s.pt"  # choose yolov5s for better accuracy on CPU while keeping speed reasonable
print("Loading YOLOv5 model...")
yolo = torch.hub.load(str(root / "yolov5"), 'custom', path=str(model_path), source='local')
yolo.conf = 0.5  # Confidence threshold
# Keep only navigation-relevant classes to reduce postprocessing cost
ALLOWED_CLASSES = {
    'person',
    'bicycle', 'motorcycle', 'car', 'bus', 'truck',
    'traffic light', 'stop sign', 'fire hydrant', 'parking meter',
    'bench', 'chair', 'couch', 'potted plant',
    'backpack', 'handbag', 'suitcase'
}

# Try to load MiDaS for depth estimation
midas_available = False
midas_model = None
midas_transform = None
midas_needs_upsample = False

def try_load_midas():
    """
    Load MiDaS either from local weights (if present) or via torch.hub (downloads weights).
    """
    global midas_available, midas_model, midas_transform, midas_needs_upsample

    # First, try local weights with provided loader
    try:
        sys.path.insert(0, str(root))
        from MiDaS.midas.model_loader import load_model

        weight_path = root / "MiDaS" / "weights" / "midas_v21_small_256.pt"
        if weight_path.exists():
            midas_model, midas_transform, _ = load_model(
                device,
                str(weight_path),
                model_type="midas_v21_small_256",
                optimize=False
            )
            midas_needs_upsample = False
            midas_available = True
            print("Loaded MiDaS (local weights).")
            return
    except Exception as e:
        print(f"Local MiDaS load failed: {e}")

    # Fallback: torch.hub download (MiDaS_small)
    try:
        print("Loading MiDaS (this may take a moment)...")
        midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = transforms.small_transform
        midas_model.to(device).eval()
        midas_needs_upsample = True
        midas_available = True
        print("Loaded MiDaS (torch.hub MiDaS_small).")
    except Exception as e:
        print(f"torch.hub MiDaS load failed: {e}")
        midas_available = False

try_load_midas()

# Flask app
app = Flask(__name__)

# Processing configuration
YOLO_EVERY_N_FRAMES = 1  # Run YOLO every frame (fast at 320)
MIDAS_EVERY_N_FRAMES = 3  # Run MiDaS every Nth frame (expensive, cache between runs)
DETECTION_PERSISTENCE_THRESHOLD = 2  # Require detection in N consecutive processed frames before alerting
INFERENCE_SIZE = 320  # YOLO inference resolution (320 is fastest, still accurate for navigation)
TEMPORAL_DEPTH_SMOOTHING = 0.3  # Blend factor for temporal depth smoothing (lower = more responsive)
USE_MIDAS = True  # Set True for accurate MiDaS depth (runs less frequently)

# Global state
processing_lock = Lock()
frame_queue = Queue(maxsize=1)
result_queue = Queue(maxsize=1)
is_paused = False
fps_counter = 0
fps_time = time.time()
frame_count = 0
processed_frame_count = 0  # Frames actually processed by YOLO+MiDaS
current_stats = {
    'fps': 0,
    'processed_fps': 0,
    'detections': 0,
    'stable_detections': 0,
    'depth_method': 'MiDaS (cached)' if (midas_available and USE_MIDAS) else 'Edge-based',
    'total_frames': 0,
    'uptime': 0
}
start_time = time.time()

# Detection persistence tracking: {label+box_hash: consecutive_frame_count}
detection_history = {}
previous_depth_map = None
last_detections = []
last_depth = None


def normalize_depth_map(depth):
    """
    Normalize a depth map to the 0-1 range per frame.
    Uses min/max so the full range maps to 0..1 each frame.
    """
    depth = depth.astype(np.float32)
    depth_min = np.nanmin(depth)
    depth_max = np.nanmax(depth)
    
    if depth_max - depth_min < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    
    depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    return np.clip(depth, 0.0, 1.0)


def get_depth_map_midas(frame, img_rgb):
    """
    Compute depth using MiDaS model.
    Returns normalized depth map where 0 = far and 1 = close.
    """
    h, w = frame.shape[:2]
    
    with torch.no_grad():
        input_batch = midas_transform(img_rgb).to(device)
        prediction = midas_model(input_batch)

        if midas_needs_upsample:
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze(1)
            depth = prediction.squeeze().cpu().numpy()
        else:
            depth = prediction.squeeze().cpu().numpy()
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    
    depth = normalize_depth_map(depth)
    return depth


def get_depth_map_edge_based(frame):
    """
    Compute depth using edge-based method (fast fallback).
    Returns normalized depth map where 0 = far and 1 = close to edges/objects.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 50, 150)
    
    depth = cv2.distanceTransform(
        cv2.bitwise_not(edges),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, kernel)
    depth = cv2.GaussianBlur(depth, (21, 21), 0)
    
    depth = normalize_depth_map(depth)
    
    return depth


def get_detection_key(det):
    """Generate a unique key for a detection based on label and approximate location."""
    x1, y1, x2, y2 = det['box']
    # Quantize box to reduce sensitivity to small movements
    cx, cy = (x1 + x2) // 2 // 50, (y1 + y2) // 2 // 50
    return f"{det['label']}_{cx}_{cy}"


def process_frame_worker():
    """
    Background thread: continuously process frames from the queue.
    Performs YOLO detection and depth estimation only on selected frames.
    Reuses previous results for intermediate frames.
    """
    global frame_count, current_stats, processed_frame_count
    global detection_history, previous_depth_map, last_detections, last_depth
    
    last_process_time = time.time()
    
    while True:
        try:
            frame_id, frame, timestamp = frame_queue.get(timeout=1)
        except:
            continue
        
        if frame is None:  # Shutdown signal
            break
        
        h, w = frame.shape[:2]
        
        # Decide whether to run YOLO and/or MiDaS on this frame
        should_run_yolo = (frame_id % YOLO_EVERY_N_FRAMES == 0)
        should_run_midas = (frame_id % MIDAS_EVERY_N_FRAMES == 0)
        
        # Compute depth map (MiDaS runs less frequently, cached between runs)
        if should_run_midas and USE_MIDAS and midas_available:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = get_depth_map_midas(frame, img_rgb)
            
            # Temporal depth smoothing
            if previous_depth_map is not None and depth.shape == previous_depth_map.shape:
                depth = (TEMPORAL_DEPTH_SMOOTHING * previous_depth_map + 
                         (1 - TEMPORAL_DEPTH_SMOOTHING) * depth)
            previous_depth_map = depth.copy()
            last_depth = depth
        elif last_depth is not None and last_depth.shape[:2] == (h, w):
            # Reuse cached depth map
            depth = last_depth
        else:
            # Fallback: edge-based or zeros
            if not USE_MIDAS:
                depth = get_depth_map_edge_based(frame)
            else:
                depth = np.zeros((h, w), dtype=np.float32)
            last_depth = depth
        
        # YOLO detection (runs every frame for low latency)
        if should_run_yolo:
            results = yolo(frame, size=INFERENCE_SIZE)
            df = results.pandas().xyxy[0]
            processed_frame_count += 1
        else:
            df = None  # Skip YOLO parsing
        
        # Process detections
        detections = []
        vis = frame.copy()
        
        if should_run_yolo and df is not None:
            # Full processing: run detection pipeline
            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                roi = depth[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                
                median_depth = np.nanmedian(roi)
                if not np.isfinite(median_depth):
                    continue
                
                label = str(row["name"])
                if label not in ALLOWED_CLASSES:
                    continue
                confidence = float(row["confidence"])
                
                detections.append({
                    'label': label,
                    'depth': median_depth,
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence
                })
            
            # Update detection persistence tracking
            current_keys = set()
            for det in detections:
                key = get_detection_key(det)
                current_keys.add(key)
                detection_history[key] = detection_history.get(key, 0) + 1
            
            # Decay/remove detections not seen this frame
            keys_to_remove = [k for k in detection_history if k not in current_keys]
            for k in keys_to_remove:
                del detection_history[k]
            
            # Mark stable detections (persisted across threshold frames)
            for det in detections:
                key = get_detection_key(det)
                det['stable'] = detection_history.get(key, 0) >= DETECTION_PERSISTENCE_THRESHOLD
            
            # Cache results for intermediate frames
            last_detections = detections
            last_depth = depth
        else:
            # Reuse cached detections from last processed frame
            detections = last_detections
        
        # Sort by depth (farthest first)
        detections = sorted(detections, key=lambda x: x['depth'], reverse=True)
        
        # Count stable detections for stats
        stable_count = sum(1 for d in detections if d.get('stable', False))
        
        # Draw detections on frame
        for det in detections:
            x1, y1, x2, y2 = det['box']
            depth_val = det['depth']
            
            depth_normalized = float(np.clip(depth_val, 0.0, 1.0))
            
            # Color gradient: green (far) -> yellow -> red (close)
            if depth_normalized < 0.5:
                r = int(255 * (depth_normalized * 2))
                g = 255
                b = 0
            else:
                r = 255
                g = int(255 * (1 - (depth_normalized - 0.5) * 2))
                b = 0
            
            color = (b, g, r)
            
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            text = "{} ({:.2f}) d:{:.2f}".format(det['label'], det['confidence'], depth_normalized)
            cv2.putText(
                vis, text, (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2
            )
        
        # Add stats to frame
        depth_method = "MiDaS" if midas_available else "Edge-based"
        status_text = "{} det ({} stable) | {} | {:.1f} FPS".format(
            len(detections), stable_count, depth_method, current_stats['processed_fps']
        )
        cv2.putText(
            vis, status_text, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        
        # Update stats
        current_stats['detections'] = len(detections)
        current_stats['stable_detections'] = stable_count
        current_stats['uptime'] = int(time.time() - start_time)
        current_stats['total_frames'] = frame_id
        
        try:
            result_queue.put_nowait((frame_id, vis, detections, depth))
        except:
            pass
        
        frame_count += 1


# Video streaming
current_frame = None
frame_lock = Lock()


def generate_frames():
    """
    Generator function that yields JPEG frames for streaming.
    """
    global current_frame, fps_counter, fps_time, current_stats, processed_frame_count
    
    while True:
        if current_frame is None:
            time.sleep(0.01)
            continue
        
        with frame_lock:
            _, buffer = cv2.imencode('.jpg', current_frame)
            frame_data = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n' + 
               frame_data + b'\r\n')
        
        # Update FPS counters
        fps_counter += 1
        if time.time() - fps_time > 1.0:
            current_stats['fps'] = fps_counter
            current_stats['processed_fps'] = processed_frame_count
            fps_time = time.time()
            fps_counter = 0
            processed_frame_count = 0
        
        time.sleep(0.01)


def main_capture_worker():
    """
    Main capture thread: continuously grab frames from webcam and process.
    """
    global current_frame, is_paused
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Webcam opened successfully!")
    
    # Start processing thread
    processor = Thread(target=process_frame_worker, daemon=True)
    processor.start()
    
    frame_id = 0
    last_result = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Send latest frame to processing queue if not paused (drop stale frames)
            if not is_paused:
                try:
                    while not frame_queue.empty():
                        frame_queue.get_nowait()
                    frame_queue.put_nowait((frame_id, frame, time.time()))
                except:
                    pass
                
                # Get latest result if available
                try:
                    last_result = result_queue.get_nowait()
                except:
                    pass
            
            # Use processed frame or raw frame
            if last_result:
                _, display_frame, _, _ = last_result
            else:
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    "Processing...",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
            
            # Store for streaming
            with frame_lock:
                current_frame = display_frame
            
            frame_id += 1
    
    except KeyboardInterrupt:
        print("\nShutdown signal received")
    
    finally:
        frame_queue.put((None, None, None))
        cap.release()
        print("Webcam closed")


# Routes
@app.route('/')
def index():
    """Main page with video stream and stats."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Stream video frames."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """Return current statistics as JSON."""
    return jsonify(current_stats)


@app.route('/pause', methods=['POST'])
def pause():
    """Pause/resume processing."""
    global is_paused
    is_paused = not is_paused
    return jsonify({'paused': is_paused})


if __name__ == "__main__":
    print("\nPathfinder Stream Detection - Web Interface")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"MiDaS Available: {midas_available}")
    print("\nStarting capture thread...")
    
    # Start capture thread
    capture_thread = Thread(target=main_capture_worker, daemon=False)
    capture_thread.start()
    
    # Give it a moment to start
    time.sleep(2)
    
    # Start Flask server
    print("\n" + "=" * 50)
    print("Web interface starting on http://0.0.0.0:5000")
    print("Access from your laptop at: http://<server-ip>:5000")
    print("=" * 50 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
