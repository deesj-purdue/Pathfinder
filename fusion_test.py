import torch, cv2, numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent
img_path = root / "yolov5" / "data" / "images" / "bus.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True, trust_repo=True)
results = yolo(str(img_path))
df = results.pandas().xyxy[0]

midas = torch.hub.load("isl-org/MiDaS", "DPT_Large", trust_repo=True).to(device).eval()
transforms = torch.hub.load("isl-org/MiDaS", "transforms")
transform = transforms.dpt_transform
img = cv2.imread(str(img_path))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
inp = transform(img_rgb).to(device)
with torch.no_grad():
    pred = midas(inp)
depth = pred.squeeze().cpu().numpy()
depth = pred.squeeze().cpu().numpy()
depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
h, w = depth.shape
img = cv2.imread(str(img_path))
img_h, img_w = img.shape[:2]
depth = cv2.resize(depth, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
ordered = []
vis = img.copy()
for _, row in df.iterrows():
    x1, y1, x2, y2 = map(int, [row.xmin, row.ymin, row.xmax, row.ymax])
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)
    if x2 <= x1 or y2 <= y1:
        continue
    roi = depth[y1:y2, x1:x2]
    if roi.size == 0:
        continue
    m = np.nanmedian(roi)
    if not np.isfinite(m):
        continue
    ordered.append((row["name"], float(m), (x1, y1, x2, y2)))
ordered = sorted(ordered, key=lambda x: x[1])
print("Objects detected (farthest first):")
for label, d, box in ordered:
    print(f"{label} - relative depth {d:.3f}")
    x1, y1, x2, y2 = box
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"{label} {d:.2f}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
out_dir = root / "outputs"
out_dir.mkdir(exist_ok=True)
cv2.imwrite(str(out_dir / "fusion_overlay.png"), vis)
