"""Optimized Video Fusion for Real-time Performance"""

import time
import cv2
import torch
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue
from datetime import datetime
import csv

# Performance preset: "fastest" | "balanced" | "quality"
PRESET = "fastest"

PRESETS = {
    "fastest": {
        "yolo_model": "yolov5n",  # Nano - 4x faster than yolov5s
        "midas_model": "MiDaS_small",  # Small - 10x faster than DPT_Large
        "input_size": 320,  # Smaller input = faster processing
        "description": "Optimized for speed (~5-10 FPS on CPU)"
    },
    "balanced": {
        "yolo_model": "yolov5s",  # Small
        "midas_model": "DPT_Hybrid",  # Hybrid - good balance
        "input_size": 480,
        "description": "Balance of speed and quality (~2-5 FPS on CPU)"
    },
    "quality": {
        "yolo_model": "yolov5m",  # Medium
        "midas_model": "DPT_Large",  # Large - best quality
        "input_size": 640,
        "description": "Best quality, slowest (~0.5-2 FPS on CPU)"
    }
}

config = PRESETS[PRESET]
DURATION_SECONDS = 10
PROCESS_INTERVAL = 1.0

root = Path(__file__).resolve().parent
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*70}")
print(f"OPTIMIZED VIDEO FUSION")
print(f"{'='*70}")
print(f"Preset: {PRESET.upper()}")
print(f"Description: {config['description']}")
print(f"Device: {device}")
print(f"YOLO Model: {config['yolo_model']}")
print(f"MiDaS Model: {config['midas_model']}")
print(f"Input Size: {config['input_size']}")
print(f"{'='*70}\n")

# Load YOLO with specified size
yolo = torch.hub.load("ultralytics/yolov5", config["yolo_model"], pretrained=True, trust_repo=True)
if hasattr(yolo, "to"):
    yolo.to(device)
yolo.conf = 0.5

# Load MiDaS with specified size
midas = torch.hub.load("isl-org/MiDaS", config["midas_model"], trust_repo=True).to(device).eval()

transforms_module = torch.hub.load("isl-org/MiDaS", "transforms")
if "small" in config["midas_model"].lower():
    midas_transform = transforms_module.small_transform
else:
    midas_transform = transforms_module.dpt_transform

print("✓ Models loaded successfully!\n")

use_amp = device.type == "cuda"
if use_amp:
    print("✓ Using Automatic Mixed Precision (AMP) for faster GPU inference\n")

process_queue = Queue()
processed_count = 0
processing_times = []
timing_records = []
run_start_time = time.time()
run_id = f"{PRESET}_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"


def process_frame(frame_data):
    """Optimized frame processing with detailed timing"""
    global processed_count, processing_times, timing_records
    
    frame_id, frame, capture_elapsed = frame_data
    process_start = time.time()
    t_stages = {}
    
    h, w = frame.shape[:2]
    input_w, input_h = w, h
    
    # Stage 1: Resize to target size
    target_size = config["input_size"]
    if w > target_size or h > target_size:
        scale = target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        input_w, input_h = new_w, new_h
    else:
        frame_resized = frame
        new_w, new_h = w, h
    
    t1 = time.time()
    
    # Stage 2: Convert to RGB
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    t_rgb = time.time() - t1
    
    # Stage 3: MiDaS preprocessing and inference
    t2 = time.time()
    with torch.no_grad(), torch.inference_mode():
        inp = midas_transform(rgb).to(device)
        t_midas_prep = time.time() - t2
        
        t3 = time.time()
        if use_amp:
            with torch.amp.autocast('cuda'):
                pred = midas(inp)
        else:
            pred = midas(inp)
        t_midas_infer = time.time() - t3
        
        t4 = time.time()
        depth = pred.cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        
        if frame_resized.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        t_depth_post = time.time() - t4
    
    # Stage 4: YOLO inference
    t5 = time.time()
    with torch.no_grad():
        results = yolo(rgb, size=config["input_size"])
    t_yolo_infer = time.time() - t5
    df = results.pandas().xyxy[0]
    
    # Stage 5: Process detections and create overlay
    t6 = time.time()
    if frame_resized.shape[:2] != (h, w):
        scale_x = w / new_w
        scale_y = h / new_h
    else:
        scale_x = scale_y = 1.0
    
    vis = frame.copy()
    ordered = []
    
    for _, row in df.iterrows():
        x1 = int(row.xmin * scale_x)
        y1 = int(row.ymin * scale_y)
        x2 = int(row.xmax * scale_x)
        y2 = int(row.ymax * scale_y)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        roi = depth[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        
        m = float(np.mean(roi))
        if not np.isfinite(m):
            continue
        
        label = str(row["name"])
        confidence = float(row["confidence"])
        ordered.append((label, m, confidence, (x1, y1, x2, y2)))
    
    ordered.sort(key=lambda x: x[1])
    
    for idx, (label, depth_val, conf, box) in enumerate(ordered):
        x1, y1, x2, y2 = box
        
        color_val = int(255 * depth_val)
        color = (0, color_val, 255 - color_val)
        
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {depth_val:.2f}"
        cv2.putText(vis, text, (x1, max(20, y1 - 6)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    info_text = f"Frame {frame_id} @ {capture_elapsed:.1f}s | Objects: {len(ordered)} | {PRESET.upper()}"
    cv2.putText(vis, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    output_path = out_dir / f"fusion_overlay_{frame_id}_{PRESET}.png"
    cv2.imwrite(str(output_path), vis)
    t_overlay = time.time() - t6
    
    t_total = time.time() - process_start
    proc_elapsed = time.time() - run_start_time
    
    processed_count += 1
    processing_times.append(t_total)
    
    # Record timing data
    timing_records.append({
        'run_id': run_id,
        'preset': PRESET,
        'device': str(device),
        'yolo_model': config['yolo_model'],
        'midas_model': config['midas_model'],
        'input_w': input_w,
        'input_h': input_h,
        'frame': frame_id,
        't_rgb': round(t_rgb, 3),
        't_midas_prep': round(t_midas_prep, 3),
        't_midas_infer': round(t_midas_infer, 3),
        't_depth_post': round(t_depth_post, 3),
        't_yolo_infer': round(t_yolo_infer, 3),
        't_overlay': round(t_overlay, 3),
        't_total': round(t_total, 3),
        'objects': len(ordered),
        'save_path': str(output_path),
        'err': '',
        'capture_ts': round(capture_elapsed, 3),
        'proc_ts': round(proc_elapsed, 3)
    })
    
    print(f"Processing frame {frame_id}...", end=" ", flush=True)
    print(f"✓ Done in {t_total:.2f}s ({len(ordered)} objects)")
    
    return frame_id, output_path, ordered


def processing_worker():
    """Worker thread for concurrent processing"""
    while True:
        frame_data = process_queue.get()
        if frame_data is None:
            break
        try:
            process_frame(frame_data)
        except Exception as e:
            print(f"Error processing frame: {e}")
        finally:
            process_queue.task_done()


def export_timing_csv():
    """Export detailed timing data to CSV"""
    csv_path = out_dir / f"fusion_timing_{PRESET}.csv"
    
    fieldnames = [
        'run_id', 'preset', 'device', 'yolo_model', 'midas_model',
        'input_w', 'input_h', 'frame', 't_rgb', 't_midas_prep',
        't_midas_infer', 't_depth_post', 't_yolo_infer', 't_overlay',
        't_total', 'objects', 'save_path', 'err', 'capture_ts', 'proc_ts'
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_records)
    
    return csv_path


def main():
    # Setup camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Start processing worker
    worker = Thread(target=processing_worker, daemon=True)
    worker.start()
    
    print(f"Starting {DURATION_SECONDS}-second capture...\n")
    
    start_time = time.time()
    last_process_time = start_time
    frame_count = 0
    total_frames = 0
    
    try:
        while time.time() - start_time < DURATION_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            
            total_frames += 1
            current_time = time.time()
            elapsed = current_time - start_time
            
            if current_time - last_process_time >= PROCESS_INTERVAL:
                frame_count += 1
                process_queue.put((frame_count, frame.copy(), elapsed))
                last_process_time = current_time
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
    
    print(f"\nCapture complete! Waiting for processing...\n")
    
    # Wait for processing
    process_queue.join()
    process_queue.put(None)
    worker.join()
    
    # Export timing data
    csv_path = export_timing_csv()
    
    # Statistics
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Frames processed: {processed_count}/{frame_count}")
    if processing_times:
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        min_time = np.min(processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"Processing time: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
        print(f"Effective FPS: {fps:.2f}")
        print(f"Real-time capable: {'YES ✓' if fps >= 1.0 else 'NO ✗'}")
    print(f"Output: {out_dir}")
    print(f"Timing CSV: {csv_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
