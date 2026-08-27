import os
import sys
import json
import glob
import random
import csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from final_pipeline import run, FLOOR_VOCAB, DEMO_DIR, BASE

META_DIR = os.path.join(BASE, 'dataset_button', 'original_labels')
TEST_DIR = os.path.join(BASE, 'dataset_button', 'test', 'images')

test_images = sorted(glob.glob(os.path.join(TEST_DIR, '*')))

label_to_images = {}
image_to_labels = {}
for img_path in test_images:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    meta_path = os.path.join(META_DIR, stem + '.json')
    if not os.path.exists(meta_path):
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    labels = []
    for ann in meta['annotations']:
        if not ann.get('malformed', False) and ann['class_name'] in FLOOR_VOCAB:
            labels.append(ann['class_name'])
    if labels:
        image_to_labels[img_path] = labels
        for lbl in labels:
            label_to_images.setdefault(lbl, []).append(img_path)

random.seed(42)

DEMO_TARGETS = [
    ['11', '12', '13', '10', '14'],
    ['20', '22', '21', '25', '27'],
    ['19', '17', '18', '15', '16'],
    ['B1', 'B2', 'B3'],
    ['7', '9', '4', '5', '6'],
]

selected = []
used = set()
for cat in DEMO_TARGETS:
    for lbl in cat:
        candidates = [p for p in label_to_images.get(lbl, []) if p not in used]
        if candidates:
            img_path = random.choice(candidates)
            selected.append((img_path, lbl))
            used.add(img_path)
            break

while len(selected) < 5:
    extras = [(p, image_to_labels[p][0]) for p in image_to_labels if p not in used]
    random.shuffle(extras)
    for item in extras:
        selected.append(item)
        used.add(item[0])
        if len(selected) >= 5:
            break

os.makedirs(DEMO_DIR, exist_ok=True)
csv_path = os.path.join(DEMO_DIR, 'final_demo_results.csv')
fieldnames = ['image_name', 'target_floor', 'target_found', 'predicted_floor',
              'yolo_confidence', 'ocr_confidence', 'orientation', 'fallback_used', 'match']
csv_rows = []

print('=== Lastmile Robotics — 5-Image End-to-End Demo ===')
print()

for i, (img_path, gt_target) in enumerate(selected[:5]):
    print(f'[{i+1}/5] Image: {os.path.basename(img_path)}  Target: {gt_target}')
    target_found, buttons, target_button = run(img_path, gt_target, save_dir=DEMO_DIR)

    if target_button:
        predicted = target_button['floor']
        yolo_conf = target_button['yolo_confidence']
        ocr_conf = target_button['ocr_confidence']
        orient = target_button['orientation']
        fallback_used = target_button['fallback']
    elif buttons:
        best_b = max(buttons, key=lambda b: b['ocr_confidence'])
        predicted = best_b['floor']
        yolo_conf = best_b['yolo_confidence']
        ocr_conf = best_b['ocr_confidence']
        orient = best_b['orientation']
        fallback_used = best_b['fallback']
    else:
        predicted = 'NO_DETECTION'
        yolo_conf = 0.0
        ocr_conf = 0.0
        orient = 'n/a'
        fallback_used = False

    matched = predicted == gt_target
    print(f'    predicted={predicted}  yolo_conf={yolo_conf}  ocr_conf={ocr_conf}  orient={orient}  fallback={fallback_used}  match={matched}')
    print()

    csv_rows.append({
        'image_name': os.path.basename(img_path),
        'target_floor': gt_target,
        'target_found': target_found,
        'predicted_floor': predicted,
        'yolo_confidence': yolo_conf,
        'ocr_confidence': ocr_conf,
        'orientation': orient,
        'fallback_used': fallback_used,
        'match': matched,
    })

with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

matches = sum(1 for r in csv_rows if r['match'])
print(f'Demo complete: {matches}/5 predictions matched ground truth')
print('CSV: ' + csv_path)
print('Annotated images: ' + DEMO_DIR)
