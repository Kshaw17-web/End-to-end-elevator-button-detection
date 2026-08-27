import argparse
import os
import sys
import cv2
import easyocr
from ultralytics import YOLO

BASE = r'c:\Users\ksr20\OneDrive\Desktop\Lastmile Robotics'
MODEL_PATH = os.path.join(BASE, 'runs', 'detect', 'button_yolov8n_fast_baseline', 'weights', 'best.pt')
DEMO_DIR = os.path.join(BASE, 'runs', 'detect', 'button_yolov8n_fast_baseline', 'final_demo')
CONF_THRESH = 0.45
STRONG_CONF = 0.75
FLOOR_VOCAB = set([str(i) for i in range(1, 31)] + ['B1', 'B2', 'B3'])

reader = easyocr.Reader(['en'], gpu=False, verbose=False)
model = YOLO(MODEL_PATH)


def normalize(text):
    return ''.join(c for c in text.upper().strip() if c.isalnum())


def ocr_recognize(gray_img, allowlist=None):
    h, w = gray_img.shape[:2]
    kw = {'allowlist': allowlist} if allowlist else {}
    try:
        res = reader.recognize(gray_img, [[0, w, 0, h]], [], **kw)
    except Exception:
        try:
            res = reader.recognize(gray_img, [], [[[0, 0], [w, 0], [w, h], [0, h]]], **kw)
        except Exception:
            return '', 0.0
    if not res:
        return '', 0.0
    best = max(res, key=lambda r: r[2])
    return str(best[1]), float(best[2])


def ocr_two_passes(gray_img):
    t_a, c_a = ocr_recognize(gray_img)
    t_b, c_b = ocr_recognize(gray_img, allowlist='0123456789B')
    return [(t_a, c_a), (t_b, c_b)]


def prepare_gray(img_bgr, x1, y1, x2, y2, pad, flip):
    h, w = img_bgr.shape[:2]
    src = cv2.flip(img_bgr, 1) if flip else img_bgr
    fx1 = w - x2 if flip else x1
    fx2 = w - x1 if flip else x2
    cx1, cy1 = max(0, fx1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, fx2 + pad), min(h, y2 + pad)
    crop = src[cy1:cy2, cx1:cx2]
    if crop.shape[0] > 4:
        crop = crop[:max(1, int(crop.shape[0] * 0.70)), :]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    if float(gray.mean()) < 127:
        gray = cv2.bitwise_not(gray)
    gh, gw = gray.shape
    return cv2.resize(gray, (gw * 3, gh * 3), interpolation=cv2.INTER_CUBIC)


def contrast_enhance(gray_img):
    enhanced = cv2.convertScaleAbs(gray_img, alpha=1.4, beta=20)
    if float(enhanced.mean()) < 127:
        enhanced = cv2.bitwise_not(enhanced)
    return enhanced


def score_candidate(text, conf, support_count):
    nt = normalize(text)
    if nt not in FLOOR_VOCAB:
        return -1.0
    return conf + support_count * 0.1


def select_best(candidates):
    valid = [(normalize(t), c) for t, c in candidates if normalize(t) in FLOOR_VOCAB]
    if not valid:
        return 'UNREADABLE', 0.0
    tally = {}
    for nt, c in valid:
        if nt not in tally:
            tally[nt] = {'max_conf': c, 'count': 0}
        tally[nt]['count'] += 1
        tally[nt]['max_conf'] = max(tally[nt]['max_conf'], c)
    ranked = sorted(tally.items(),
                    key=lambda x: score_candidate(x[0], x[1]['max_conf'], x[1]['count']),
                    reverse=True)
    best_label, best_meta = ranked[0]
    return best_label, best_meta['max_conf']


def is_strong(text, conf):
    return normalize(text) in FLOOR_VOCAB and conf >= STRONG_CONF


def process_button(img_bgr, x1, y1, x2, y2):
    all_candidates = []
    used_fallback = False

    gray = prepare_gray(img_bgr, x1, y1, x2, y2, pad=4, flip=False)
    pairs = ocr_two_passes(gray)
    all_candidates.extend(pairs)
    for t, c in pairs:
        if is_strong(t, c):
            floor, conf = select_best(all_candidates)
            return floor, conf, 'original', t, used_fallback

    gray_f = prepare_gray(img_bgr, x1, y1, x2, y2, pad=4, flip=True)
    pairs_f = ocr_two_passes(gray_f)
    all_candidates.extend(pairs_f)
    for t, c in pairs_f:
        if is_strong(t, c):
            floor, conf = select_best(all_candidates)
            return floor, conf, 'flipped', t, used_fallback

    used_fallback = True

    gray_ce = contrast_enhance(gray)
    pairs_ce = ocr_two_passes(gray_ce)
    all_candidates.extend(pairs_ce)
    for t, c in pairs_ce:
        if is_strong(t, c):
            floor, conf = select_best(all_candidates)
            return floor, conf, 'original_enhanced', t, used_fallback

    gray_cef = contrast_enhance(gray_f)
    pairs_cef = ocr_two_passes(gray_cef)
    all_candidates.extend(pairs_cef)
    for t, c in pairs_cef:
        if is_strong(t, c):
            floor, conf = select_best(all_candidates)
            return floor, conf, 'flipped_enhanced', t, used_fallback

    floor, conf = select_best(all_candidates)
    raw = next((t for t, c in all_candidates if t.strip()), '')
    return floor, conf, 'best_effort', raw, used_fallback


def run(image_path, target_floor, save_dir=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print('ERROR: cannot read image: ' + image_path)
        sys.exit(1)

    yolo_res = model.predict(image_path, conf=CONF_THRESH, verbose=False)
    detections = []
    if yolo_res and len(yolo_res[0].boxes):
        for box in yolo_res[0].boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            detections.append((x1, y1, x2, y2, float(box.conf[0])))

    buttons = []
    for x1, y1, x2, y2, yolo_conf in detections:
        floor, ocr_conf, orient, raw, used_fallback = process_button(img_bgr, x1, y1, x2, y2)
        buttons.append({
            'bbox': [x1, y1, x2, y2],
            'center': [(x1 + x2) // 2, (y1 + y2) // 2],
            'yolo_confidence': round(yolo_conf, 4),
            'floor': floor,
            'ocr_confidence': round(ocr_conf, 4),
            'raw_ocr': raw,
            'orientation': orient,
            'fallback': used_fallback,
        })

    target_button = None
    for b in buttons:
        if b['floor'] == target_floor:
            if target_button is None or b['ocr_confidence'] > target_button['ocr_confidence']:
                target_button = b

    target_found = target_button is not None
    print('target_found=' + str(target_found).lower())
    print('target_floor=' + target_floor)
    if target_found:
        tb = target_button
        print('bbox=' + str(tb['bbox']))
        print('center=' + str(tb['center']))
        print('detection_confidence=' + str(tb['yolo_confidence']))
        print('ocr_confidence=' + str(tb['ocr_confidence']))
        print('ocr_text=' + tb['floor'])
        print('orientation=' + tb['orientation'])
        print('fallback_used=' + str(tb['fallback']).lower())
    print('total_buttons_detected=' + str(len(detections)))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ann = img_bgr.copy()
        for b in buttons:
            x1, y1, x2, y2 = b['bbox']
            is_tgt = (b is target_button)
            color = (0, 220, 0) if is_tgt else (255, 140, 0)
            thick = 3 if is_tgt else 2
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, thick)
            lbl = b['floor'] + ' ' + str(round(b['ocr_confidence'], 2))
            cv2.putText(ann, lbl, (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        header = 'Target: ' + target_floor + (' [FOUND]' if target_found else ' [NOT FOUND]')
        cv2.putText(ann, header, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 255), 2)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(save_dir, stem + '_annotated.jpg'), ann)

    return target_found, buttons, target_button


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--target-floor', required=True)
    args = parser.parse_args()
    tf = args.target_floor.upper()
    run(args.image, tf, save_dir=DEMO_DIR)
