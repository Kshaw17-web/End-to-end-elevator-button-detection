import os
import random
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

BASE       = r'c:\Users\ksr20\OneDrive\Desktop\Lastmile Robotics'
WEIGHTS    = os.path.join(BASE, 'runs', 'detect', 'button_yolov8n_fast_baseline', 'weights', 'best.pt')
TEST_IMGS  = os.path.join(BASE, 'dataset_button', 'test', 'images')
TEST_LBLS  = os.path.join(BASE, 'dataset_button', 'test', 'labels')
OUT_VIZ    = os.path.join(BASE, 'runs', 'detect', 'button_yolov8n_fast_baseline', 'test_predictions')
OUT_CROPS  = os.path.join(BASE, 'runs', 'detect', 'button_yolov8n_fast_baseline', 'test_crops')

os.makedirs(OUT_VIZ, exist_ok=True)
os.makedirs(OUT_CROPS, exist_ok=True)

CONF_THRESH = 0.25
N_SAMPLES   = 20
SEED        = 42

all_images = sorted(Path(TEST_IMGS).glob('*.jpg'))
random.seed(SEED)
selected = random.sample(all_images, min(N_SAMPLES, len(all_images)))

model = YOLO(WEIGHTS)

total_detections = 0
all_confs = []
per_image_stats = []

for img_path in selected:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    stem = img_path.stem

    results = model(img_path, conf=CONF_THRESH, device='cpu', verbose=False)
    boxes = results[0].boxes

    viz = img.copy()
    crop_count = 0

    lbl_path = Path(TEST_LBLS) / (stem + '.txt')
    gt_boxes = []
    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    gt_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 1)

    img_confs = []
    if boxes is not None and len(boxes):
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f'{conf:.2f}'
            cv2.putText(viz, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

            pad = 2
            cx1 = max(x1 - pad, 0)
            cy1 = max(y1 - pad, 0)
            cx2 = min(x2 + pad, w)
            cy2 = min(y2 + pad, h)
            crop = img[cy1:cy2, cx1:cx2]
            crop_name = f'{stem}_crop{i:03d}_conf{conf:.2f}.jpg'
            cv2.imwrite(os.path.join(OUT_CROPS, crop_name), crop)
            crop_count += 1
            img_confs.append(conf)

    n_gt = len(gt_boxes)
    n_pred = len(img_confs)
    total_detections += n_pred
    all_confs.extend(img_confs)

    cv2.imwrite(os.path.join(OUT_VIZ, stem + '_pred.jpg'), viz)

    per_image_stats.append({
        'file': img_path.name,
        'gt': n_gt,
        'pred': n_pred,
        'min_conf': round(min(img_confs), 4) if img_confs else None,
        'max_conf': round(max(img_confs), 4) if img_confs else None,
        'avg_conf': round(sum(img_confs)/len(img_confs), 4) if img_confs else None,
    })

print(f'\n=== Inference Summary ===')
print(f'Images processed   : {len(selected)}')
print(f'Total detections   : {total_detections}')
print(f'Avg detections/img : {total_detections / len(selected):.2f}')
print(f'Min confidence     : {min(all_confs):.4f}')
print(f'Max confidence     : {max(all_confs):.4f}')
print(f'Avg confidence     : {sum(all_confs)/len(all_confs):.4f}')

print(f'\n=== Per-Image Results ===')
print(f'{"File":<60} {"GT":>4} {"Pred":>5} {"MinC":>6} {"MaxC":>6} {"AvgC":>6}')
print('-' * 85)
for s in per_image_stats:
    mc = f'{s["min_conf"]}' if s["min_conf"] is not None else 'N/A'
    xc = f'{s["max_conf"]}' if s["max_conf"] is not None else 'N/A'
    ac = f'{s["avg_conf"]}' if s["avg_conf"] is not None else 'N/A'
    print(f'{s["file"]:<60} {s["gt"]:>4} {s["pred"]:>5} {mc:>6} {xc:>6} {ac:>6}')

print(f'\nVisualizations : {OUT_VIZ}')
print(f'Crops          : {OUT_CROPS}')
