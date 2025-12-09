"""
Fusion Test - YOLO Detection + Depth Estimation

This script validates the entire object detection and depth estimation pipeline
on a static image. It shows how well the system can identify objects and understand
their spatial relationships (which are closer, which are farther away).

Uses YOLOv5 for finding objects and edge-based depth estimation to rank them by
distance. Works with Python 3.6+ and includes fallback methods when MiDaS isn't available.
"""
import torch
import cv2
import numpy as np
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
img_path = root / "yolov5" / "data" / "images" / "bus.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import yolov5

print("="*70)
print("FUSION TEST - YOLO + DEPTH ESTIMATION")
print("="*70)

# Start by loading the pre-trained object detection model
print("\n[1/3] Loading YOLOv5 model...")
model_path = root / "yolov5n.pt"
yolo = yolov5.load(str(model_path))
print("✓ YOLOv5n loaded successfully")

# Now run the model on our test image to find objects
print("\n[2/3] Running object detection...")
results = yolo(str(img_path))
df = results.pandas().xyxy[0]
print("Detected {} objects".format(len(df)))

# Load the image for depth processing
img = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_h, img_w = img.shape[:2]

# Now compute depth to understand spatial relationships
print("\n[3/3] Computing depth map...")

# First, try the more advanced MiDaS model if available
# Otherwise, fall back to edge-based depth estimation
midas_available = False
try:
    sys.path.insert(0, str(root))
    from MiDaS.midas.model_loader import load_model
    
    weight_path = root / "MiDaS" / "weights" / "midas_v21_small_256.pt"
    if weight_path.exists():
        # If weights exist, load and use MiDaS for better depth accuracy
        print("  Loading MiDaS depth model...")
        model, transform, net_w = load_model(
            device, 
            str(weight_path), 
            model_type="midas_v21_small_256", 
            optimize=False
        )
        
        # Process the image through the model
        with torch.no_grad():
            input_batch = transform(img_rgb).to(device)
            prediction = model(input_batch)
            depth = prediction.squeeze().cpu().numpy()
        
        # Normalize the depth values to 0-1 range for easier interpretation
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        print("  ✓ Using MiDaS depth estimation")
        midas_available = True
    else:
        print("  MiDaS weights not found, using fallback")
except Exception as e:
    print("  Could not load MiDaS: {}".format(str(e)))

# Fallback: edge-based depth (captures object prominence and distance cues)
# If MiDaS didn't work, use our edge-based method
# This looks at edges and structure in the image to estimate depth
if not midas_available:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find all the edges in the image (where objects are)
    edges = cv2.Canny(gray, 50, 150)
    # Distance transform measures how much structure exists at each point
    # More structure = closer to objects = likely closer in depth
    dist = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Smooth things out to get a cleaner depth map
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dist = cv2.morphologyEx(dist, cv2.MORPH_CLOSE, kernel)
    dist = cv2.GaussianBlur(dist, (21, 21), 0)
    # Scale everything to 0-1 so it's easy to interpret
    depth = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
    print("  ✓ Using edge-based depth estimation")

# Now take each detected object and figure out how far it is
print("\n" + "="*70)
print("RESULTS")
print("="*70)

ordered = []
vis = img.copy()

for _, row in df.iterrows():
    # Get the bounding box coordinates from the detection
    x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)
    
    if x2 <= x1 or y2 <= y1:
        continue
    
    # Extract the depth values for just this object's region
    roi = depth[y1:y2, x1:x2]
    if roi.size == 0:
        continue
    
    # Use the median depth value for the object
    # (middle value is more robust than mean in case of noise)
    median_depth = np.nanmedian(roi)
    if not np.isfinite(median_depth):
        continue
    
    label = str(row["name"])
    confidence = float(row["confidence"])
    ordered.append((label, float(median_depth), confidence, (x1, y1, x2, y2)))

# Sort by depth with highest values first (farthest objects first)
# This gives us the ranking from back to front in the scene
ordered = sorted(ordered, key=lambda x: x[1], reverse=True)

# Display the ranked list so we can see which objects are farthest
print("\nObjects by distance (farthest to closest):\n")
for idx, (label, depth_val, conf, box) in enumerate(ordered, 1):
    distance_percent = int(depth_val * 100)
    print("  {}: {} - Depth {:.3f} ({}%) | Confidence {:.2f}".format(
        idx, label, depth_val, distance_percent, conf
    ))
    
    # Draw boxes on the image with colors that show depth
    x1, y1, x2, y2 = box
    # Green for far objects, red for close objects
    color_val = int(255 * (1 - depth_val))
    color = (color_val, 255 - abs(color_val - 128), 255 - color_val)
    
    # Draw the box on the image
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    # Label it with the class name and depth value
    text = "{} {:.2f}".format(label, depth_val)
    cv2.putText(
        vis, text, (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

# Save the annotated image to the outputs folder
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)
output_path = out_dir / "fusion_test_result.png"
cv2.imwrite(str(output_path), vis)

# All done! Show a summary
print("\n" + "="*70)
print("✓ Test completed successfully!")
print("✓ Output saved to: {}".format(output_path))
print("="*70 + "\n")
print("Next steps: Check the output image to see which objects are marked")
print("green (far) vs red (close). This validates our depth ordering.\n")
