import os
import re
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr

BASE        = r'c:\Users\ksr20\OneDrive\Desktop\Lastmile Robotics'
WEIGHTS     = os.path.join(BASE, 'runs', 'detect', 'button_yolov8n_fast_baseline', 'weights', 'best.pt')
TEST_IMGS   = os.path.join(BASE, 'dataset_button', 'test', 'images')
ORIG_LABELS = os.path.join(BASE, 'dataset_button', 'original_labels')
OUT_DIR     = os.path.join(BASE, 'runs', 'detect', 'button_yolov8n_fast_baseline', 'ocr_evaluation')
VIZ_DIR     = os.path.join(OUT_DIR, 'visualizations')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)

CONF_THRESH = 0.45
PAD         = 3
BRAILLE_CUT = 0.15
IOU_THRESH  = 0.5

FLOOR_LABELS = set([str(i) for i in range(1, 31)] + ['B1', 'B2', 'B3'])

ICON_CLASSES = {'up', 'down', 'alarm', 'open', 'close', 'stop', 'call',
                'bt_keyhole', 'indicator', 'led', 'fan', 'speaker', 'switch',
                'keyhole', 'light', 'fire', 'hat', 'key', 'unknown', 'blur',
                'empty', 'updown'}

reader = easyocr.Reader(['en'], gpu=False, verbose=False)
model = YOLO(WEIGHTS)

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def preprocess(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced = clahe.apply(gray)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {'gray': gray, 'clahe': enhanced, 'otsu': otsu}

def run_ocr(img_arr):
    results = reader.readtext(img_arr, detail=1, paragraph=False)
    if not results:
        return '', 0.0
    best = max(results, key=lambda r: r[2])
    return best[1], float(best[2])

def normalize_text(raw):
    t = raw.strip().upper()
    t = re.sub(r'[^\w]', '', t)
    if not t:
        return ''
    if t in ('B1', 'B2', 'B3'):
        return t
    if re.fullmatch(r'[0-9]+', t):
        return t
    if len(t) == 1:
        if t == 'O': return '0'
        if t in ('I', 'L'): return '1'
        return t
    if len(t) == 2:
        m = {'O':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'2'}
        n = ''.join(m.get(c,c) if not c.isdigit() else c for c in t)
        if re.fullmatch(r'[0-9]+', n): return n
    return t

def best_ocr_for_crop(crop_bgr):
    h = crop_bgr.shape[0]
    focus = crop_bgr[:int(h*(1.0-BRAILLE_CUT)), :]
    gray = cv2.cvtColor(focus, cv2.COLOR_BGR2GRAY)
    variants = preprocess(gray)
    flip_gray = cv2.cvtColor(cv2.flip(focus, 1), cv2.COLOR_BGR2GRAY)
    flip_variants = preprocess(flip_gray)
    results = []
    for name, img in variants.items():
        text, conf = run_ocr(img)
        results.append((text, conf, 'original', name))
    for name, img in flip_variants.items():
        text, conf = run_ocr(img)
        results.append((text, conf, 'flipped', name))
    if not any(r[1] > 0 for r in results):
        return '', 0.0, 'original', 'gray'
    return max(results, key=lambda r: r[1])

all_rows = []
viz_count = 0
VIZ_LIMIT = 20

for img_path in sorted(Path(TEST_IMGS).glob('*.jpg')):
    stem = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    h, w = img.shape[:2]
    json_path = Path(ORIG_LABELS) / (stem + '.json')
    gt_annots = []
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        for ann in data.get('annotations', []):
            cx, cy, bw, bh = ann['cx'], ann['cy'], ann['w'], ann['h']
            x1 = int((cx-bw/2)*w); y1 = int((cy-bh/2)*h)
            x2 = int((cx+bw/2)*w); y2 = int((cy+bh/2)*h)
            gt_annots.append({'class_name': ann['class_name'], 'box': (x1,y1,x2,y2), 'matched': False})
    results = model(img_path, conf=CONF_THRESH, device='cpu', verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        continue
    for det_idx, box in enumerate(boxes):
        yolo_conf = float(box.conf[0])
        px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
        gt_class = None
        best_iou = 0.0
        for ann in gt_annots:
            if ann['matched']:
                continue
            iou = compute_iou((px1,py1,px2,py2), ann['box'])
            if iou > best_iou:
                best_iou = iou
                if iou >= IOU_THRESH:
                    gt_class = ann['class_name']
                    ann['matched'] = True
        if gt_class is None:
            continue
        cx1 = max(px1-PAD,0); cy1 = max(py1-PAD,0)
        cx2 = min(px2+PAD,w); cy2 = min(py2+PAD,h)
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        raw_text, ocr_conf, orientation, pp_variant = best_ocr_for_crop(crop)
        norm_text = normalize_text(raw_text)
        gt_upper = gt_class.upper().strip()
        exact_match = (norm_text == gt_upper)
        is_floor = gt_upper in FLOOR_LABELS
        is_icon = gt_class.lower() in ICON_CLASSES
        all_rows.append({
            'image': stem, 'det_idx': det_idx,
            'yolo_conf': round(yolo_conf, 4), 'box': f'{px1},{py1},{px2},{py2}',
            'gt_class': gt_class, 'raw_ocr': raw_text, 'norm_ocr': norm_text,
            'ocr_conf': round(ocr_conf, 4), 'orientation': orientation,
            'pp_variant': pp_variant, 'exact_match': exact_match,
            'is_floor': is_floor, 'is_icon': is_icon, 'iou': round(best_iou, 4),
        })
        do_viz = viz_count < VIZ_LIMIT and (
            re.fullmatch(r'\d', gt_upper) or
            re.fullmatch(r'\d{2,}', gt_upper) or
            gt_upper in ('B1','B2','B3') or
            not exact_match or is_icon
        )
        if do_viz:
            ph, pw = crop.shape[:2]
            panel = np.zeros((max(ph, 120), pw + 320, 3), dtype=np.uint8)
            panel[:ph, :pw] = crop
            tx = pw + 8
            cv2.putText(panel, f'GT: {gt_class}',    (tx,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(panel, f'OCR: {norm_text}',  (tx,42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)
            cv2.putText(panel, f'conf:{ocr_conf:.2f}',(tx,64), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(200,200,200),1)
            cv2.putText(panel, orientation,           (tx,84), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200),1)
            col = (0,255,0) if exact_match else (0,0,255)
            cv2.putText(panel, 'MATCH' if exact_match else 'FAIL', (tx,106), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            fname = f'viz_{viz_count:03d}_gt{gt_class}_{stem[:30]}.jpg'
            cv2.imwrite(os.path.join(VIZ_DIR, fname), panel)
            viz_count += 1

csv_path = os.path.join(OUT_DIR, 'ocr_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)

matched     = [r for r in all_rows if not r['is_icon']]
floor_rows  = [r for r in matched if r['is_floor']]
single_rows = [r for r in floor_rows if re.fullmatch(r'\d', r['gt_class'].strip())]
multi_rows  = [r for r in floor_rows if re.fullmatch(r'\d{2,}', r['gt_class'].strip())]
base_rows   = [r for r in floor_rows if r['gt_class'].upper() in ('B1','B2','B3')]
orig_rows   = [r for r in matched if r['orientation'] == 'original']
flip_rows   = [r for r in matched if r['orientation'] == 'flipped']
unreadable  = sum(1 for r in matched if not r['raw_ocr'].strip())
avg_conf    = round(sum(r['ocr_conf'] for r in matched)/len(matched), 4) if matched else 0.0

def acc(rows):
    if not rows: return 0.0
    return round(sum(1 for r in rows if r['exact_match'])/len(rows)*100, 2)

def norm_acc(rows):
    if not rows: return 0.0
    return round(sum(1 for r in rows if normalize_text(r['raw_ocr'])==r['gt_class'].upper().strip())/len(rows)*100, 2)

fail_pats = {}
for r in matched:
    if not r['exact_match'] and r['gt_class'].upper() in FLOOR_LABELS:
        key = f"GT={r['gt_class']} -> OCR={r['norm_ocr'] or '(empty)'}"
        fail_pats[key] = fail_pats.get(key, 0) + 1
top_failures = sorted(fail_pats.items(), key=lambda x: -x[1])[:15]

summary = {
    'total_matched_detections': len(all_rows),
    'non_icon_detections': len(matched),
    'floor_label_instances': len(floor_rows),
    'exact_ocr_accuracy_pct': acc(matched),
    'normalized_ocr_accuracy_pct': norm_acc(matched),
    'floor_exact_accuracy_pct': acc(floor_rows),
    'single_digit_accuracy_pct': acc(single_rows),
    'multi_digit_accuracy_pct': acc(multi_rows),
    'basement_label_accuracy_pct': acc(base_rows),
    'avg_ocr_confidence': avg_conf,
    'unreadable_crops': unreadable,
    'original_orientation_accuracy_pct': acc(orig_rows),
    'flipped_orientation_accuracy_pct': acc(flip_rows),
    'original_orientation_count': len(orig_rows),
    'flipped_orientation_count': len(flip_rows),
    'top_failure_patterns': top_failures,
}

json_path = os.path.join(OUT_DIR, 'ocr_summary.json')
with open(json_path, 'w') as f:
    json.dump(summary, f, indent=2)

report = ['# OCR Evaluation Report', '## Run: button_yolov8n_fast_baseline', '',
    '## Configuration', '',
    '| Parameter | Value |', '|---|---|',
    f'| YOLO model | best.pt (epoch 21) |',
    f'| YOLO confidence threshold | {CONF_THRESH} |',
    f'| IoU threshold | {IOU_THRESH} |',
    f'| OCR engine | EasyOCR English CPU |',
    f'| Braille mask | lower {int(BRAILLE_CUT*100)}% removed |',
    f'| Preprocessing | gray, CLAHE, Otsu |',
    f'| Orientations | original + horizontally flipped |', '',
    '## Results Summary', '',
    '| Metric | Value |', '|---|---|',
    f'| Total matched detections | {summary["total_matched_detections"]} |',
    f'| Non-icon detections evaluated | {summary["non_icon_detections"]} |',
    f'| Floor label instances | {summary["floor_label_instances"]} |',
    f'| Exact OCR accuracy (all non-icon) | {summary["exact_ocr_accuracy_pct"]}% |',
    f'| Normalized OCR accuracy | {summary["normalized_ocr_accuracy_pct"]}% |',
    f'| Floor label exact accuracy | {summary["floor_exact_accuracy_pct"]}% |',
    f'| Single-digit accuracy | {summary["single_digit_accuracy_pct"]}% |',
    f'| Multi-digit accuracy | {summary["multi_digit_accuracy_pct"]}% |',
    f'| Basement label accuracy | {summary["basement_label_accuracy_pct"]}% |',
    f'| Average OCR confidence | {summary["avg_ocr_confidence"]} |',
    f'| Unreadable crops | {summary["unreadable_crops"]} |', '',
    '## Orientation Analysis', '',
    '| Orientation | Count | Accuracy |', '|---|---|---|',
    f'| Original | {summary["original_orientation_count"]} | {summary["original_orientation_accuracy_pct"]}% |',
    f'| Flipped | {summary["flipped_orientation_count"]} | {summary["flipped_orientation_accuracy_pct"]}% |', '',
    '## Top OCR Failure Patterns', '',
    '| Pattern | Count |', '|---|---|',
]
for pat, cnt in top_failures:
    report.append(f'| {pat} | {cnt} |')

report_path = os.path.join(OUT_DIR, 'ocr_evaluation_report.md')
with open(report_path, 'w') as f:
    f.write('\n'.join(report))

print('=== OCR Evaluation Complete ===')
for k, v in summary.items():
    if k != 'top_failure_patterns':
        print(f'{k}: {v}')
print('\nTop failure patterns:')
for pat, cnt in top_failures:
    print(f'  {cnt:3d}x  {pat}')
print(f'\nCSV:    {csv_path}')
print(f'JSON:   {json_path}')
print(f'Report: {report_path}')
print(f'Viz:    {VIZ_DIR}')
