"""
Video Fusion Proof of Concept
Captures 5-second video and processes one frame per second with YOLO + MiDaS
Outputs 5 fusion overlay images showing objects with depth information
Uses concurrent processing for real-time performance
"""

import time
import cv2
import torch
import numpy as np
from pathlib import Path
from threading import Thread, Lock
from queue import Queue
from collections import deque

# Setup paths and output directory
root = Path(__file__).resolve().parent
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models (once at startup)
print("Loading YOLO model...")
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, trust_repo=True)
if hasattr(yolo, "to"):
    yolo.to(device)

print("Loading MiDaS model...")
midas = torch.hub.load("isl-org/MiDaS", "DPT_Large", trust_repo=True).to(device).eval()
midas_transform = torch.hub.load("isl-org/MiDaS", "transforms").dpt_transform

print("Models loaded successfully!\n")

# Global processing queue and lock
process_queue = Queue()
results_lock = Lock()
processed_count = 0


def process_frame(frame_data):
    """
    Process a single frame with YOLO and MiDaS
    frame_data: tuple of (frame_id, frame, timestamp)
    """
    global processed_count
    
    frame_id, frame, timestamp = frame_data
    print(f"Processing frame {frame_id} (captured at {timestamp:.2f}s)...")
    
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # MiDaS depth estimation
    with torch.no_grad():
        inp = midas_transform(rgb).to(device)
        pred = midas(inp)
        depth = pred.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # YOLO object detection
    results = yolo(rgb, size=640)
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
        text = f"{label} {depth_val:.2f} ({conf:.2f})"
        cv2.putText(vis, text, (x1, max(20, y1 - 6)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add frame info overlay
    info_text = f"Frame {frame_id} @ {timestamp:.1f}s | Objects: {len(ordered)}"
    cv2.putText(vis, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    # Save fusion overlay
    output_path = out_dir / f"fusion_overlay_{frame_id}.png"
    cv2.imwrite(str(output_path), vis)
    
    with results_lock:
        processed_count += 1
        
    print(f"✓ Frame {frame_id} processed: {len(ordered)} objects detected -> {output_path.name}")
    
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
            print(f"Error processing frame: {e}")
        finally:
            process_queue.task_done()


def main():
    # Configuration
    DURATION_SECONDS = 5
    PROCESS_INTERVAL = 1.0  # Process one frame per second
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 360
    
    print(f"Starting {DURATION_SECONDS}-second video capture...")
    print(f"Will process 1 frame every {PROCESS_INTERVAL} second(s)\n")
    
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
    
    print(f"\nCapture complete!")
    print(f"Total frames captured: {total_frames}")
    print(f"Frames queued for processing: {frame_count}")
    print(f"\nWaiting for processing to complete...")
    
    # Wait for all processing to complete
    process_queue.join()
    
    # Stop worker thread
    process_queue.put(None)
    worker.join()
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {processed_count}/{frame_count} frames")
    print(f"Output directory: {out_dir}")
    print(f"\nGenerated files:")
    for i in range(1, frame_count + 1):
        output_file = out_dir / f"fusion_overlay_{i}.png"
        if output_file.exists():
            print(f"  ✓ {output_file.name}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
