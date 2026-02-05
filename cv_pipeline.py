# imports
import cv2
import torch
import numpy as np
import time
import sys
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent / "fast-scnn" / "Fast-SCNN-pytorch"))
from models.fast_scnn import FastSCNN

# configuration
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

YOLO_INTERVAL = 2
MIDAS_INTERVAL = 3
SEGMENTATION_INTERVAL = 4

# cityscapes class labels and colors for segmentation visualization
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
    'motorcycle', 'bicycle'
]

CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)


# model loading functions
def load_yolo():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    return model

def load_midas(device):
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.to(device).eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    return model, transform

def load_fast_scnn(device):
    model = FastSCNN(num_classes=19)
    
    weights_path = Path(__file__).parent / "fast-scnn" / "Fast-SCNN-pytorch" / "weights" / "fast_scnn_citys.pth"
    if weights_path.exists():
        state_dict = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device).eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    return model, mean, std


# model inference functions
def run_yolo(model, frame):
    results = model(frame, verbose=False, conf=0.4)
    return results[0] if results else None

def run_midas(model, transform, frame, device):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    
    with torch.no_grad():
        depth = model(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False
        ).squeeze()
    
    return depth.cpu().numpy()

def run_segmentation(model, frame, mean, std, device):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    img_tensor = (img_tensor - mean) / std
    
    with torch.no_grad():
        outputs = model(img_tensor)
        pred = torch.argmax(outputs[0], dim=1).squeeze(0)
    
    return pred.cpu().numpy()


# visualization functions
def draw_yolo_panel(frame, yolo_results):
    panel = frame.copy()
    
    if yolo_results is not None:
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = yolo_results.names[cls_id]
            
            cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(panel, f"{label} {conf:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(panel, "YOLO Detection", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return panel

def draw_depth_panel(frame, depth_map):
    if depth_map is not None:
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)
    else:
        depth_colored = np.zeros_like(frame)
    
    cv2.putText(depth_colored, "MiDaS Depth", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return depth_colored

def draw_segmentation_panel(frame, segmentation):
    if segmentation is not None:
        seg_colored = CITYSCAPES_COLORS[segmentation]
        seg_colored = cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR)
    else:
        seg_colored = np.zeros_like(frame)
    
    cv2.putText(seg_colored, "Fast-SCNN Segmentation", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return seg_colored

def draw_fused_panel(frame, yolo_results, depth_map, segmentation):
    fused = frame.copy()
    
    if segmentation is not None:
        seg_colored = CITYSCAPES_COLORS[segmentation]
        seg_colored = cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR)
        fused = cv2.addWeighted(fused, 0.6, seg_colored, 0.4, 0)
    
    if depth_map is not None:
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_overlay = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_MAGMA)
        fused = cv2.addWeighted(fused, 0.8, depth_overlay, 0.2, 0)
    
    if yolo_results is not None:
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = yolo_results.names[int(box.cls[0])]
            cv2.rectangle(fused, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(fused, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(fused, "Fused View", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return fused

# main function
def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    print("Loading models...")
    yolo_model = load_yolo()
    midas_model, midas_transform = load_midas(device)
    scnn_model, scnn_mean, scnn_std = load_fast_scnn(device)
    print("All models loaded!")
    
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting pipeline... Press 'q' to quit")
    
    frame_count = 0
    yolo_results = None
    depth_map = None
    segmentation = None
    
    fps_history = deque(maxlen=30)
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_count += 1
        
        # staggered model execution
        if frame_count % YOLO_INTERVAL == 0:
            yolo_results = run_yolo(yolo_model, frame)
        
        if frame_count % MIDAS_INTERVAL == 0:
            depth_map = run_midas(midas_model, midas_transform, frame, device)
        
        if frame_count % SEGMENTATION_INTERVAL == 0:
            segmentation = run_segmentation(scnn_model, frame, scnn_mean, scnn_std, device)
        
        # draw panels
        yolo_panel = draw_yolo_panel(frame, yolo_results)
        depth_panel = draw_depth_panel(frame, depth_map)
        seg_panel = draw_segmentation_panel(frame, segmentation)
        fused_panel = draw_fused_panel(frame, yolo_results, depth_map, segmentation)
        
        # combine into 2x2 grid
        top_row = np.hstack([yolo_panel, depth_panel])
        bottom_row = np.hstack([seg_panel, fused_panel])
        display = np.vstack([top_row, bottom_row])
        
        # calculate fps
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        fps_history.append(fps)
        avg_fps = sum(fps_history) / len(fps_history)
        
        cv2.putText(display, f"FPS: {avg_fps:.1f}", (display.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Pathfinder CV Pipeline", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()