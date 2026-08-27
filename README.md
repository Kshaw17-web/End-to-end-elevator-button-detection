# Lastmile Robotics — Elevator Button Perception System

An end-to-end computer vision pipeline that detects elevator floor buttons and reads their labels, enabling an autonomous last-mile delivery robot to locate and press the correct floor button.

---

## Problem Statement

Autonomous delivery robots operating in multi-story buildings must navigate elevators autonomously. This requires identifying the correct floor button from a camera image of an elevator panel. The system must:

1. Detect all elevator buttons in the camera image.
2. Read the floor label printed on each button using OCR.
3. Locate the button matching the requested target floor and report its position.

---

## System Architecture

```
Input image
    │
    ▼
YOLOv8n button detector  (src/detector.py)
    │  bounding boxes + YOLO confidence
    ▼
Per-button crop preprocessing  (src/ocr.py)
    │  4 px padding · Braille-region masking · 3× upscale · grayscale
    ▼
EasyOCR reader.recognize()  — skips CRAFT text detection
    │  Pass A: unrestricted · Pass B: digit+B allowlist
    │  Flipped fallback · Contrast-enhanced fallback
    ▼
Floor label selection  (src/floor_selector.py)
    │  normalize → vocabulary match → confidence scoring
    ▼
Optional target-floor comparison  (detect_floor.py)
    │
    ▼
Output: target_found, bbox, center, YOLO confidence, OCR confidence
```

**Design decision — skipping CRAFT:** YOLO has already localized each button precisely. Running EasyOCR's CRAFT text-detection network on the crop is redundant and slow. The pipeline calls `reader.recognize()` with an explicit `horizontal_list` covering the entire preprocessed crop, bypassing CRAFT and running only the recognition network.

---

## Dataset

**Source:** ACC Elevator Button Dataset (Roboflow `acc-stwam/elevator-button-jinxe-qb0gs v1`, CC BY 4.0)

| Property | Value |
|---|---|
| Original images | 1,139 |
| Original classes | 38 (floors 1–30, B1–B3, Open, Close, Up, Down, Emergency) |
| Annotation style | YOLO format bounding boxes |

**Why a single `button` class?**

The first pipeline stage is button *localization* — every button is structurally equivalent regardless of its label. Collapsing all 38 semantic classes into a single class maximizes training samples, simplifies the detector, and cleanly separates localization (YOLO) from recognition (OCR). The original 38-class labels are preserved in `dataset_button/original_labels/` for evaluation purposes.

**Dataset preparation** (`dataset_button/`):

- All 38 classes remapped to class 0 (`button`)
- 2 geometrically degenerate annotations removed
- 6 orphaned label files (no matching image) excluded
- Stratified 80/10/10 split using a greedy rarity-first strategy so all 38 original classes appear in every split

| Split | Images | Annotations |
|---|---|---|
| Train | 907 | 5,258 |
| Validation | 113 | 780 |
| Test | 113 | 761 |

See `dataset_button/preparation_report.md` for full details.

---

## Model

| Property | Value |
|---|---|
| Architecture | YOLOv8n (nano) |
| Parameters | 3.0 M |
| GFLOPs | 8.2 |
| Pre-trained weights | Official `yolov8n.pt` (COCO) |
| Classes | 1 (`button`) |
| Input size | 512 × 512 |
| Trained weights | `models/best.pt` |

---

## Training

| Parameter | Value |
|---|---|
| Epochs | 30 (early stopping, patience = 8) |
| Batch size | 8 |
| Image size | 512 |
| Optimizer | AdamW (YOLOv8 default) |
| Augmentation | YOLOv8 default mosaic + geometric augmentation |
| Seed | 42 |
| Device | CPU (AMD Ryzen 5 5500U) |
| Best epoch | 21 |

**Validation metrics (best epoch):**

| Metric | Value |
|---|---|
| mAP@50 | 0.9941 |
| mAP@50-95 | 0.7241 |
| Precision | 0.9863 |
| Recall | 0.9923 |

**Test-set metrics (held-out, 113 images, 761 annotations, conf ≥ 0.45):**

| Metric | Value |
|---|---|
| Precision | **0.9882** |
| Recall | **0.9931** |
| mAP@50 | **0.9915** |
| mAP@50-95 | **0.7359** |

Full training logs and curves are in `runs/detect/button_yolov8n_fast_baseline/`.

---

## OCR Pipeline

**Engine:** EasyOCR (English, CPU)

**Per-button preprocessing:**
1. 4 px padding clipped to image boundaries
2. Lower 15% of crop removed (reduces Braille dot interference)
3. 3× upscaling (bicubic interpolation)
4. Grayscale conversion

**Recognition passes (4 total):**

| Pass | Orientation | Preprocessing |
|---|---|---|
| 1A | Original | Plain grayscale |
| 1B | Original | Plain grayscale + digit/B allowlist |
| 2A | Horizontally flipped | Plain grayscale |
| 2B | Horizontally flipped | Plain grayscale + digit/B allowlist |

If none of the above yields a high-confidence match, contrast-enhanced variants (alpha=1.4) of both orientations are tried as a fallback.

**EasyOCR API note:** `reader.recognize()` is called with `horizontal_list` constructed from the dimensions of the *post-processed crop* (after all resizing), not from the original YOLO bounding box. This ensures the bounding box passed to EasyOCR exactly matches the array being recognized.

---

## Floor Label Selection

**Valid floor vocabulary:** integers 1–30, plus `B1`, `B2`, `B3`.

**Selection logic (`src/floor_selector.py`):**

1. Normalize each OCR result: uppercase, strip non-alphanumeric characters.
2. Filter to candidates whose normalized text is in the floor vocabulary.
3. Score each valid candidate: `max_confidence + vote_count × 0.1`.
4. Return the highest-scoring candidate, or `UNREADABLE` if no valid candidate exists.

No hard-coded corrections or per-label special cases are used.

---

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Target-floor detection (main use case)

```bash
python detect_floor.py --image dataset_button/test/images/<image.jpg> --target-floor "14"
```

Example output:
```
total_buttons_detected=9
target_floor=14
target_found=true
bbox=[120, 80, 245, 205]
center=[182, 142]
detection_confidence=0.91
ocr_confidence=0.94
ocr_text=14
annotated_image=<save_dir>/<stem>_annotated.jpg
```

Save an annotated image:
```bash
python detect_floor.py --image path/to/image.jpg --target-floor "14" --save-dir output/
```

### General inference (all buttons)

```bash
python inference.py --image path/to/image.jpg [--save path/to/output.jpg]
```

### 5-image demo (existing experiment)

```bash
python scripts/demo.py
```

### Train from scratch

```bash
python train.py --epochs 30 --imgsz 512 --batch 8
```

> **Note:** The model has already been trained. `models/best.pt` contains the trained weights. Re-running `train.py` will start a new training run; it will not overwrite `models/best.pt`.

---

## Results

### Detector (test set)

| Metric | Value |
|---|---|
| Precision | 0.9882 |
| Recall | 0.9931 |
| mAP@50 | 0.9915 |
| mAP@50-95 | 0.7359 |

### OCR (experimental V1 evaluation — full test set, 113 images, 761 matched detections)

> **Note:** These results are from the V1 OCR evaluation script (`scripts/ocr_evaluation.py`), which used a different preprocessing path (CLAHE + Otsu) than the production pipeline in `src/ocr.py`. They are reported for reference. A V2 evaluation run was started (`scripts/ocr_evaluation_v2.py`) but did not produce a full-test-set summary artifact. A precise end-to-end correct-floor count (correct / total) cannot be computed from stored artifacts without re-running the pipeline; only category-level accuracy percentages are available.

| Metric | Value | Evaluated on |
|---|---|---|
| Raw OCR accuracy (all non-icon detections) | 65.57% | 700 non-icon crops |
| Floor-label accuracy | 66.62% | 689 floor-label instances |
| Single-digit floor accuracy | 56.53% | subset of 689 |
| Multi-digit floor accuracy | 83.55% | subset of 689 |
| B1/B2/B3 accuracy | 18.18% | subset of 689 |
| Original orientation accuracy | 57.75% | 471 crops |
| Flipped orientation accuracy | 81.66% | 229 crops |

OCR is the primary performance bottleneck. The detector achieves near-perfect localization (mAP@50 = 0.9915); OCR accuracy varies substantially by floor label category. B1/B2/B3 recognition remains the weakest category. The 5-image `final_demo_results.csv` (3/5 target floors correctly matched) is qualitative only and is not a substitute for a full-test-set end-to-end evaluation.

---

## Limitations

- **CPU-only:** Both YOLO inference and EasyOCR run on CPU. EasyOCR recognition takes approximately 0.5–1.5 s per button crop; the total per-image latency grows linearly with button count. YOLO detector inference is faster (sub-second on typical panels), but a per-image YOLO-only timing artifact was not separately stored. GPU acceleration would be required for real-time use.
- **Single-digit floors:** Small character size at typical crop resolutions reduces OCR confidence for floors 1–9.
- **B1/B2/B3:** Character similarity between `B`, `8`, and common OCR noise tokens makes basement floor recognition the hardest category.
- **Mirrored text:** Some elevator panels produce horizontally flipped text relative to the camera. The flipped-orientation pass mitigates but does not fully solve this.
- **Braille interference:** Braille dots on button surfaces can confuse OCR. The lower-15%-masking heuristic reduces this but is imperfect.
- **Illumination and reflections:** Specular reflections on metal buttons reduce text contrast.

---

## Future Improvements

- **Stronger OCR model or ensemble:** Replace EasyOCR with a scene-text model fine-tuned on elevator panel fonts, or ensemble EasyOCR with a secondary engine (e.g. Tesseract in single-character mode) to improve single-digit and B1/B2/B3 accuracy.
- **Improved B1/B2/B3 handling:** The basement labels are the weakest category (18.18%). Post-processing with a character-level confusion model (B vs. 8, R vs. B, etc.) could improve recognition without hard-coded corrections.
- **Better preprocessing for reflections and illumination:** Adaptive contrast methods beyond the current 3× upscale and lower-15% masking — such as polarisation-aware preprocessing or learned deglare — could reduce OCR failures on metallic reflective buttons.
- **GPU acceleration:** Running both YOLO and EasyOCR on a GPU would reduce per-image latency by an order of magnitude, enabling near-real-time panel scanning for a deployed delivery robot.
- **More diverse training data:** The dataset originates from a single elevator panel style. Collecting images from panels with different button shapes, fonts, languages, and lighting conditions would improve generalisation to unseen building deployments.

---

## Project Structure

```
Lastmile Robotics/
│
├── README.md
├── requirements.txt
│
├── detect_floor.py          # Main entry point: detect buttons + optional target-floor matching
├── inference.py             # General inference: detect all buttons and report labels
├── train.py                 # Reproducible training entry point
│
├── models/
│   └── best.pt              # Trained YOLOv8n weights (mAP@50=0.9915)
│
├── src/
│   ├── __init__.py
│   ├── detector.py          # YOLO button detector (localization only)
│   ├── ocr.py               # EasyOCR recognize() wrapper (4-pass pipeline)
│   └── floor_selector.py    # Generic floor label selection and scoring
│
├── examples/
│   ├── input/               # Sample input images
│   └── output/              # Corresponding annotated outputs
│
├── dataset_button/          # Cleaned single-class dataset (nc=1)
│   ├── data.yaml
│   ├── preparation_report.md
│   ├── train/               # 907 images
│   ├── valid/               # 113 images
│   ├── test/                # 113 images
│   └── original_labels/     # Per-image original 38-class metadata (JSON)
│
├── scripts/                 # Experiment and evaluation scripts (preserved)
│   ├── train_baseline.py
│   ├── final_pipeline.py
│   ├── demo.py
│   ├── inspect_predictions.py
│   ├── ocr_evaluation.py
│   └── ocr_evaluation_v2.py
│
└── runs/
    └── detect/
        └── button_yolov8n_fast_baseline/   # Primary training run artifacts
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── results.csv
            ├── results.png
            ├── confusion_matrix.png
            ├── BoxPR_curve.png
            ├── final_demo/
            └── ocr_evaluation_v2/
```
