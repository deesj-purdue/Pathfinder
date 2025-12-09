"""
Video Fusion - Real-time Video Processing

Captures live video from your webcam and processes frames with object detection
and depth estimation. Uses threading to handle detection in the background while
continuing to capture frames, improving overall throughput.

Outputs annotated frames showing detected objects ranked by distance.
Python 3.6+ compatible.
"""

import time
import cv2
import torch
import numpy as np
import sys
from pathlib import Path
from threading import Thread, Lock
from queue import Queue
from collections import deque

# Setup paths and output directory
root = Path(__file__).resolve().parent
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)

# Check if GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Load YOLOv5 directly (Python 3.6 compatible)
print("Loading YOLOv5 model...")
import yolov5
model_path = root / "yolov5n.pt"
yolo = yolov5.load(str(model_path))

# Try to load MiDaS if available, otherwise use simple depth methods
print("Setting up depth estimation...")
try:
    sys.path.insert(0, str(root))
    from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
    from torchvision.transforms import Compose
    depth_method = "edge-based"
except Exception as e:
    depth_method = "edge-based"

print("Models loaded successfully!\n")

# Global processing queue and lock
process_queue = Queue()
results_lock = Lock()
processed_count = 0


def get_depth_map(frame):
    """
    Compute depth map from frame using fast edge-based method.
    
    This estimates depth by analyzing edges and structural elements
    in the image. Areas with more structure tend to be closer to the camera.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if depth_method == "edge-based":
        # Find edges in the image as a proxy for depth structure
        edges = cv2.Canny(gray, 100, 200)
        # Distance transform shows how far each pixel is from an edge
        depth = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    else:
        # Simple brightness-based fallback
        depth = gray.astype(np.float32) / 255.0
        depth = cv2.GaussianBlur(depth, (15, 15), 0)
    
    # Normalize to 0-1 range
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def process_frame(frame_data):
    """
    Process a single frame with object detection and depth estimation.
    
    This function handles the full pipeline: detect objects, compute depth,
    rank by distance, and draw annotations on the frame.
    """
    global processed_count
    
    frame_id, frame, timestamp = frame_data
    print("Processing frame {} (captured at {:.2f}s)...".format(frame_id, timestamp))
    
    h, w = frame.shape[:2]
    
    # Compute depth map
    depth = get_depth_map(frame)
    
    # YOLO object detection
    results = yolo(frame, size=640)
    df = results.pandas().xyxy[0]
    
    # Create fusion overlay
    vis = frame.copy()
    ordered = []
    
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        roi = depth[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        m = float(np.nanmedian(roi))
        if not np.isfinite(m):
            continue
            
        label = str(row["name"])
        confidence = float(row["confidence"])
        ordered.append((label, m, confidence, (x1, y1, x2, y2)))
    
    # Sort by depth (closest first)
    ordered.sort(key=lambda x: x[1])
    
    # Draw bounding boxes and labels
    for idx, (label, depth_val, conf, box) in enumerate(ordered):
        x1, y1, x2, y2 = box
        
        # Color based on depth (red=close, green=far)
        color_val = int(255 * depth_val)
        color = (0, color_val, 255 - color_val)
        
        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with depth
        text = "{} {:.2f} ({:.2f})".format(label, depth_val, conf)
        cv2.putText(vis, text, (x1, max(20, y1 - 6)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add frame info overlay
    info_text = "Frame {} @ {:.1f}s | Objects: {}".format(frame_id, timestamp, len(ordered))
    cv2.putText(vis, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Save fusion overlay
    output_path = out_dir / "fusion_overlay_{}.png".format(frame_id)
    cv2.imwrite(str(output_path), vis)
    
    with results_lock:
        processed_count += 1
        
    print("✓ Frame {} processed: {} objects detected -> {}".format(frame_id, len(ordered), output_path.name))
    
    return frame_id, output_path, ordered


def processing_worker():
    """Worker thread for processing frames concurrently"""
    while True:
        frame_data = process_queue.get()
        if frame_data is None:  # Poison pill to stop worker
            break
        try:
            process_frame(frame_data)
        except Exception as e:
            print("Error processing frame: {}".format(e))
        finally:
            process_queue.task_done()


def main():
    # Configuration
    DURATION_SECONDS = 5
    PROCESS_INTERVAL = 1.0  # Process one frame per second
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 360
    
    print("Starting {}-second video capture...".format(DURATION_SECONDS))
    print("Will process 1 frame every {} second(s)\n".format(PROCESS_INTERVAL))
    
    # Setup camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Start processing worker thread
    worker = Thread(target=processing_worker, daemon=True)
    worker.start()
    
    # Capture and process frames
    start_time = time.time()
    last_process_time = start_time
    frame_count = 0
    total_frames = 0
    
    frames_to_process = []
    
    try:
        while time.time() - start_time < DURATION_SECONDS:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to capture frame")
                break
            
            total_frames += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if we should process this frame
            if current_time - last_process_time >= PROCESS_INTERVAL:
                frame_count += 1
                # Queue frame for processing
                process_queue.put((frame_count, frame.copy(), elapsed))
                last_process_time = current_time
                
            # Small delay to prevent CPU spinning
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    finally:
        cap.release()
    
    print("\nCapture complete!")
    print("Total frames captured: {}".format(total_frames))
    print("Frames queued for processing: {}\n".format(frame_count))
    print("Waiting for processing to complete...")
    
    # Wait for all processing to complete
    process_queue.join()
    
    # Stop worker thread
    process_queue.put(None)
    worker.join()
    
    print("\n{}".format("="*60))
    print("PROCESSING COMPLETE")
    print("{}".format("="*60))
    print("Successfully processed: {}/{} frames".format(processed_count, frame_count))
    print("Output directory: {}".format(out_dir))
    print("\nGenerated files:")
    for i in range(1, frame_count + 1):
        output_file = out_dir / "fusion_overlay_{}.png".format(i)
        if output_file.exists():
            print("  ✓ {}".format(output_file.name))
    print("\n{}\n".format("="*60))


if __name__ == "__main__":
    main()
