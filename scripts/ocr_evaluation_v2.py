import os
import sys
import glob
import json
import time
import random
import csv
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE = r'c:\Users\ksr20\OneDrive\Desktop\Lastmile Robotics'
MODEL_PATH = os.path.join(BASE, 'runs', 'detect', 'button_yolov8n_fast_baseline', 'weights', 'best.pt')
TEST_IMG_DIR = os.path.join(BASE, 'dataset_button', 'test', 'images')
META_DIR = os.path.join(BASE, 'dataset_button', 'original_labels')
OUT_DIR = os.path.join(BASE, 'runs', 'detect', 'button_yolov8n_fast_baseline', 'ocr_evaluation_v2')
VIZ_DIR = os.path.join(OUT_DIR, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

CONF_THRESH = 0.45
STRONG_CONF = 0.75
SCALE3 = 3
SCALE4 = 4
IOU_THRESH = 0.5
FLOOR_VOCAB = set([str(i) for i in range(1, 31)] + ['B1', 'B2', 'B3'])

print('Loading EasyOCR...', flush=True)
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print('Loading YOLO...', flush=True)
model = YOLO(MODEL_PATH)
print('Models loaded.', flush=True)

def make_crop(img_bgr, x1, y1, x2, y2, pad, braille_mask, flip=False):
    h, w = img_bgr.shape[:2]
    src = cv2.flip(img_bgr, 1) if flip else img_bgr
    if flip:
        fx1, fx2 = w - x2, w - x1
    else:
        fx1, fx2 = x1, x2
    cx1 = max(0, fx1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, fx2 + pad)
    cy2 = min(h, y2 + pad)
    crop = src[cy1:cy2, cx1:cx2]
    if braille_mask and crop.shape[0] > 4:
        cut = max(1, int(crop.shape[0] * 0.85))
        crop = crop[:cut, :]
    return crop

def to_gray_scaled(crop_bgr, scale):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

def apply_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def apply_otsu(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def run_ocr(gray_img, allowlist=None):
    h, w = gray_img.shape[:2]
    kw = {'allowlist': allowlist} if allowlist else {}
    try:
        results = reader.recognize(gray_img, [[0, w, 0, h]], [], **kw)
    except Exception:
        results = reader.readtext(gray_img, **kw)
    if not results:
        return '', 0.0
    best = max(results, key=lambda r: r[2])
    return str(best[1]), float(best[2])

def run_two(gray_img):
    t_a, c_a = run_ocr(gray_img)
    t_b, c_b = run_ocr(gray_img, allowlist='0123456789B')
    return [(t_a, c_a), (t_b, c_b)]

def normalize(text):
    return ''.join(c for c in text.upper().strip() if c.isalnum())

def is_strong(text, conf):
    return normalize(text) in FLOOR_VOCAB and conf >= STRONG_CONF

def first_strong(pairs):
    for t, c in pairs:
        if is_strong(t, c):
            return normalize(t), c
    return None

def score_candidate(text, conf):
    t = normalize(text)
    if t in FLOOR_VOCAB:
        return conf + (1.0 if conf >= STRONG_CONF else 0.5)
    best_fs = 0.0
    for v in FLOOR_VOCAB:
        common = sum(a == b for a, b in zip(t, v))
        denom = len(t) + len(v)
        best_fs = max(best_fs, 2 * common / denom if denom else 0.0)
    return conf * 0.5 + best_fs * 0.3

def process_detection(img_bgr, x1, y1, x2, y2):
    calls = 0
    all_cands = []

    gray = to_gray_scaled(make_crop(img_bgr, x1, y1, x2, y2, 4, True, False), SCALE3)
    pairs = run_two(gray)
    calls += 2
    all_cands.extend(pairs)
    strong = first_strong(pairs)
    if strong:
        return strong[0], strong[1], 1, calls, 'original', 4, 'gray', all_cands

    for flip, orient, pad in [(True, 'flipped', 4), (False, 'original', 8), (True, 'flipped', 8)]:
        gray = to_gray_scaled(make_crop(img_bgr, x1, y1, x2, y2, pad, True, flip), SCALE3)
        pairs = run_two(gray)
        calls += 2
        all_cands.extend(pairs)
        strong = first_strong(pairs)
        if strong:
            return strong[0], strong[1], 2, calls, orient, pad, 'gray', all_cands

    for prep_fn, prep_name in [(apply_clahe, 'clahe'), (apply_otsu, 'otsu')]:
        for scale in [SCALE3, SCALE4]:
            for flip, orient in [(False, 'original'), (True, 'flipped')]:
                raw_gray = to_gray_scaled(make_crop(img_bgr, x1, y1, x2, y2, 4, True, flip), 1)
                proc = cv2.resize(prep_fn(raw_gray), (raw_gray.shape[1] * scale, raw_gray.shape[0] * scale),
                                  interpolation=cv2.INTER_CUBIC)
                pairs = run_two(proc)
                calls += 2
                all_cands.extend(pairs)
                strong = first_strong(pairs)
                if strong:
                    return strong[0], strong[1], 3, calls, orient, 4, prep_name, all_cands

    gray_full = to_gray_scaled(make_crop(img_bgr, x1, y1, x2, y2, 4, False, False), SCALE3)
    pairs = run_two(gray_full)
    calls += 2
    all_cands.extend(pairs)
    strong = first_strong(pairs)
    if strong:
        return strong[0], strong[1], 3, calls, 'original', 4, 'gray_full', all_cands

    non_empty = [(t, c) for t, c in all_cands if t.strip()]
    if non_empty:
        best_t, best_c = max(non_empty, key=lambda x: score_candidate(x[0], x[1]))
        nt = normalize(best_t)
        if nt in FLOOR_VOCAB and best_c > 0.0:
            return nt, best_c, 3, calls, 'original', 4, 'best_fallback', all_cands
    return 'UNREADABLE', 0.0, 3, calls, 'original', 4, 'none', all_cands

def iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

FLOOR_VOCAB_SET = set([str(i) for i in range(1, 31)] + ['B1', 'B2', 'B3'])
SINGLE_DIGIT = set(str(i) for i in range(1, 10))
MULTI_DIGIT = set(str(i) for i in range(10, 31))
BASEMENT = {'B1', 'B2', 'B3'}

test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*')))
total_imgs = len(test_images)

csv_path = os.path.join(OUT_DIR, 'ocr_results_v2.csv')
fieldnames = ['image_name','detection_index','gt_label','bbox_x1','bbox_y1','bbox_x2','bbox_y2',
              'yolo_confidence','stage_used','crop_padding','crop_type','upscale_factor',
              'preprocessing','orientation','ocr_mode','raw_ocr_text','ocr_confidence',
              'normalized_text','is_valid_floor_candidate','final_candidate','final_candidate_score',
              'raw_match','final_match','is_floor_label','is_unreadable']

csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()

rows = []
total_ocr_calls = 0
total_ocr_time = 0.0
stage_counts = {1: 0, 2: 0, 3: 0}
t_start_all = time.time()

for img_idx, img_path in enumerate(test_images):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    h_img, w_img = img_bgr.shape[:2]

    yolo_results = model.predict(img_path, conf=CONF_THRESH, verbose=False)
    detections = []
    if yolo_results and len(yolo_results[0].boxes):
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append((x1, y1, x2, y2, float(box.conf[0])))

    meta_path = os.path.join(META_DIR, stem + '.json')
    gt_anns = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for ann in meta['annotations']:
            if ann.get('malformed', False):
                continue
            cx, cy, bw, bh = ann['cx'], ann['cy'], ann['w'], ann['h']
            gx1 = int((cx - bw/2) * w_img)
            gy1 = int((cy - bh/2) * h_img)
            gx2 = int((cx + bw/2) * w_img)
            gy2 = int((cy + bh/2) * h_img)
            gt_anns.append({'label': ann['class_name'], 'box': (gx1, gy1, gx2, gy2)})

    for det_idx, (x1, y1, x2, y2, yolo_conf) in enumerate(detections):
        t0 = time.time()
        final_text, final_conf, stage, calls, orient, pad, prep, all_cands = process_detection(img_bgr, x1, y1, x2, y2)
        elapsed = time.time() - t0
        total_ocr_calls += calls
        total_ocr_time += elapsed
        stage_counts[stage] += 1

        gt_label = ''
        best_iou_val = 0.0
        for gt in gt_anns:
            iv = iou((x1, y1, x2, y2), gt['box'])
            if iv > best_iou_val:
                best_iou_val = iv
                if iv >= IOU_THRESH:
                    gt_label = gt['label']

        raw_text = all_cands[0][0] if all_cands else ''
        raw_conf_val = all_cands[0][1] if all_cands else 0.0
        is_floor = gt_label in FLOOR_VOCAB_SET
        raw_match = normalize(raw_text) == gt_label if gt_label else False
        final_match = final_text == gt_label if gt_label else False
        is_unreadable = final_text == 'UNREADABLE'

        row = {
            'image_name': os.path.basename(img_path),
            'detection_index': det_idx,
            'gt_label': gt_label,
            'bbox_x1': x1, 'bbox_y1': y1, 'bbox_x2': x2, 'bbox_y2': y2,
            'yolo_confidence': round(yolo_conf, 4),
            'stage_used': stage,
            'crop_padding': pad,
            'crop_type': 'braille_masked',
            'upscale_factor': SCALE3,
            'preprocessing': prep,
            'orientation': orient,
            'ocr_mode': 'both',
            'raw_ocr_text': raw_text,
            'ocr_confidence': round(raw_conf_val, 4),
            'normalized_text': normalize(raw_text),
            'is_valid_floor_candidate': normalize(raw_text) in FLOOR_VOCAB_SET,
            'final_candidate': final_text,
            'final_candidate_score': round(final_conf, 4),
            'raw_match': raw_match,
            'final_match': final_match,
            'is_floor_label': is_floor,
            'is_unreadable': is_unreadable,
        }
        rows.append(row)
        writer.writerow(row)
        csv_file.flush()

    elapsed_total = time.time() - t_start_all
    eta = (elapsed_total / (img_idx + 1)) * (total_imgs - img_idx - 1)
    print(f'[{img_idx+1}/{total_imgs}] {os.path.basename(img_path)} | dets={len(detections)} | '
          f'ocr_calls={total_ocr_calls} | elapsed={elapsed_total:.0f}s | ETA={eta:.0f}s', flush=True)

csv_file.close()

matched_rows = [r for r in rows if r['gt_label']]
floor_rows = [r for r in matched_rows if r['is_floor_label']]
single_rows = [r for r in floor_rows if r['gt_label'] in SINGLE_DIGIT]
multi_rows = [r for r in floor_rows if r['gt_label'] in MULTI_DIGIT]
basement_rows = [r for r in floor_rows if r['gt_label'] in BASEMENT]
orig_rows = [r for r in floor_rows if r['orientation'] == 'original']
flip_rows = [r for r in floor_rows if r['orientation'] == 'flipped']

def acc(lst, key='final_match'):
    return round(100.0 * sum(1 for r in lst if r[key]) / len(lst), 2) if lst else 0.0

raw_acc = acc(matched_rows, 'raw_match')
final_acc = acc(floor_rows, 'final_match')
single_acc = acc(single_rows)
multi_acc = acc(multi_rows)
basement_acc = acc(basement_rows)
orig_acc = acc(orig_rows)
flip_acc = acc(flip_rows)
n_det = len(rows)
unreadable_rate = round(100.0 * sum(1 for r in matched_rows if r['is_unreadable']) / len(matched_rows), 2) if matched_rows else 0.0
avg_calls = round(total_ocr_calls / n_det, 2) if n_det else 0.0
avg_time = round(total_ocr_time / n_det, 3) if n_det else 0.0

summary = {
    'total_matched_detections': len(matched_rows),
    'floor_label_instances': len(floor_rows),
    'raw_ocr_accuracy_pct': raw_acc,
    'domain_constrained_accuracy_pct': final_acc,
    'single_digit_accuracy_pct': single_acc,
    'multi_digit_accuracy_pct': multi_acc,
    'basement_label_accuracy_pct': basement_acc,
    'original_orientation_accuracy_pct': orig_acc,
    'flipped_orientation_accuracy_pct': flip_acc,
    'unreadable_rate_pct': unreadable_rate,
    'stage_1_count': stage_counts[1],
    'stage_2_count': stage_counts[2],
    'stage_3_count': stage_counts[3],
    'total_easyocr_calls': total_ocr_calls,
    'average_easyocr_calls_per_detection': avg_calls,
    'total_ocr_runtime_seconds': round(total_ocr_time, 1),
    'average_ocr_runtime_per_detection_seconds': avg_time,
}
with open(os.path.join(OUT_DIR, 'ocr_summary_v2.json'), 'w') as f:
    json.dump(summary, f, indent=2)

PRIORITY_LABELS = ['1','4','7','9','10','11','14','20','21','22','23','24','25','26','27','28','29','30','B1','B2','B3']
viz_rows = []
covered = set()
for label in PRIORITY_LABELS:
    cands = [r for r in floor_rows if r['gt_label'] == label]
    if cands:
        viz_rows.append(cands[0])
        covered.add(cands[0]['image_name'] + str(cands[0]['detection_index']))
    else:
        print(f'Label {label} not found in test set', flush=True)

remaining = [r for r in matched_rows if (r['image_name'] + str(r['detection_index'])) not in covered]
random.seed(42)
random.shuffle(remaining)
viz_rows.extend(remaining[:max(0, 30 - len(viz_rows))])

for i, row in enumerate(viz_rows[:30]):
    img_path = os.path.join(TEST_IMG_DIR, row['image_name'])
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x1, y1, x2, y2 = row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_rgb)
    ec = 'lime' if row['final_match'] else 'red'
    axes[0].add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=ec, facecolor='none'))
    axes[0].set_title('GT: ' + str(row['gt_label']), fontsize=11)
    axes[0].axis('off')
    crop = img_bgr[max(0, y1):y2, max(0, x1):x2]
    axes[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    info = (f"Pred: {row['final_candidate']}\nRaw OCR: {row['raw_ocr_text']}\n"
            f"Conf: {row['final_candidate_score']}\nStage: {row['stage_used']}  "
            f"Orient: {row['orientation']}\nMatch: {row['final_match']}")
    axes[1].set_title(info, fontsize=8, color=ec)
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, f"viz_{str(i).zfill(2)}_gt{row['gt_label']}.png"), dpi=100, bbox_inches='tight')
    plt.close()

failure_counts = {}
for r in floor_rows:
    if not r['final_match']:
        failure_counts[r['gt_label']] = failure_counts.get(r['gt_label'], 0) + 1
top_failures = sorted(failure_counts.items(), key=lambda x: -x[1])[:15]

report = f"""# OCR Evaluation v2 — Staged Escalation Pipeline

## Configuration

| Parameter | Value |
|---|---|
| YOLO model | button_yolov8n_fast_baseline/weights/best.pt |
| YOLO conf threshold | {CONF_THRESH} |
| OCR engine | EasyOCR English CPU (recognize(), skip CRAFT detection) |
| Stage 1 calls per detection | 2 (fixed) |
| Strong threshold | conf >= {STRONG_CONF} + exact vocab match |

## Dataset

| Split | Images | Detections | GT-matched | Floor instances |
|---|---|---|---|---|
| Test | 113 | {n_det} | {len(matched_rows)} | {len(floor_rows)} |

## Stage Resolution

| Stage | Detections | % |
|---|---|---|
| Stage 1 (2 calls) | {stage_counts[1]} | {round(100*stage_counts[1]/n_det,1) if n_det else 0}% |
| Stage 2 (escalated) | {stage_counts[2]} | {round(100*stage_counts[2]/n_det,1) if n_det else 0}% |
| Stage 3 (fallback) | {stage_counts[3]} | {round(100*stage_counts[3]/n_det,1) if n_det else 0}% |

## V1 vs V2 Accuracy

| Metric | V1 | V2 | Delta |
|---|---|---|---|
| Raw OCR accuracy | 65.57% | {raw_acc}% | {"+" if raw_acc > 65.57 else ""}{round(raw_acc - 65.57, 2)}% |
| Floor-label accuracy | 66.62% | {final_acc}% | {"+" if final_acc > 66.62 else ""}{round(final_acc - 66.62, 2)}% |
| Single-digit accuracy | 56.53% | {single_acc}% | {"+" if single_acc > 56.53 else ""}{round(single_acc - 56.53, 2)}% |
| Multi-digit accuracy | 83.55% | {multi_acc}% | {"+" if multi_acc > 83.55 else ""}{round(multi_acc - 83.55, 2)}% |
| B1/B2/B3 accuracy | 18.18% | {basement_acc}% | {"+" if basement_acc > 18.18 else ""}{round(basement_acc - 18.18, 2)}% |
| Original orientation | 57.75% | {orig_acc}% | {"+" if orig_acc > 57.75 else ""}{round(orig_acc - 57.75, 2)}% |
| Flipped orientation | 81.66% | {flip_acc}% | {"+" if flip_acc > 81.66 else ""}{round(flip_acc - 81.66, 2)}% |
| Unreadable rate | — | {unreadable_rate}% | — |

## Computational Cost

| Metric | Value |
|---|---|
| Total EasyOCR calls | {total_ocr_calls} |
| Average calls per detection | {avg_calls} |
| Total OCR runtime | {round(total_ocr_time, 1)} s |
| Average per detection | {avg_time} s |

## Failure Analysis

Top floor-label failures by class:

"""
for lbl, cnt in top_failures:
    report += f"- **{lbl}**: {cnt} failures\n"

improved_floor = final_acc > 66.62
improved_single = single_acc > 56.53
improved_basement = basement_acc > 18.18
cpu_practical = avg_time < 15.0

report += f"""
## Final Summary

| Question | Answer |
|---|---|
| V2 improved floor-label accuracy | {"YES (+"+str(round(final_acc-66.62,2))+"%" if improved_floor else "NO ("+str(round(final_acc-66.62,2))+"%)"} |
| V2 improved single-digit | {"YES" if improved_single else "NO"} |
| V2 improved B1/B2/B3 | {"YES" if improved_basement else "NO"} |
| Stage 1 resolved | {stage_counts[1]}/{n_det} ({round(100*stage_counts[1]/n_det,1) if n_det else 0}%) |
| Stage 2 required | {stage_counts[2]} |
| Stage 3 required | {stage_counts[3]} |
| Total EasyOCR calls | {total_ocr_calls} |
| Total runtime | {round(total_ocr_time,1)} s |
| Avg runtime per detection | {avg_time} s |
| CPU-practical | {"YES" if cpu_practical else "NO — consider reducing escalation"} |
| Use V2 in final project | {"YES" if improved_floor else "REVIEW — marginal improvement"} |
"""

with open(os.path.join(OUT_DIR, 'ocr_evaluation_report_v2.md'), 'w') as f:
    f.write(report)

print(json.dumps(summary, indent=2), flush=True)
print('Done. Outputs in: ' + OUT_DIR, flush=True)
