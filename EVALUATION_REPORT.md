# Evaluation Report — Elevator Button Detection & Floor Recognition

**Assignment:** Lastmile Robotics — Assignment 05
**Model:** YOLOv8n (nano), trained from COCO weights
**Dataset:** ACC Elevator Button Dataset (Roboflow, CC BY 4.0) — 1,139 images, 1 class (`button`)
**Hardware:** CPU — AMD Ryzen 5 5500U
**Best checkpoint:** epoch 21 of 30 (early stopping, patience = 8)

---

## 1. Detection (YOLOv8n Button Detector)

Evaluation on the held-out **test split** (113 images, 761 ground-truth annotations, confidence threshold >= 0.45).

| Metric | Value |
|---|---|
| Precision | **0.9882** |
| Recall | **0.9931** |
| mAP@50 | **0.9915** |
| mAP@50-95 | **0.7359** |

**Validation metrics at best epoch (epoch 21):**

| Metric | Value |
|---|---|
| Precision | 0.9863 |
| Recall | 0.9923 |
| mAP@50 | 0.9941 |
| mAP@50-95 | 0.7241 |

**Inference latency:**
EasyOCR recognition (the dominant cost) takes approximately 0.5-1.5 s per button crop on CPU.
YOLO detection alone is sub-second on a typical panel but a standalone per-image YOLO-only timing artifact was not separately logged.
Full training curves and per-epoch metrics are in `runs/detect/button_yolov8n_fast_baseline/results.csv`.

---

## 2. OCR (Button Text Recognition)

Evaluation on all 113 test-set images, matching YOLO detections to ground-truth labels.
Source: `runs/detect/button_yolov8n_fast_baseline/ocr_evaluation/ocr_results.csv` (761 rows).

> Note: Results are from the V1 OCR evaluation script (`scripts/ocr_evaluation.py`), which used CLAHE + Otsu
> preprocessing. The production pipeline (`src/ocr.py`) uses a 4-pass EasyOCR approach with grayscale +
> digit/B allowlist. V2 evaluation did not produce a complete summary artifact.

### Counts

| Item | Count |
|---|---|
| Total matched YOLO detections evaluated | 761 |
| Icon buttons excluded (Open, Close, Up, Down, Emergency) | 61 |
| Non-icon detections evaluated for OCR | 700 |
| Floor-label instances | 689 |
| **Floor labels correctly recognized** | **459** |
| **Floor labels incorrectly recognized** | **230** |

### Accuracy

| Metric | Value |
|---|---|
| Raw OCR accuracy (700 non-icon crops) | 65.57% |
| **Floor-label accuracy (689 instances)** | **66.62% (459 / 689)** |
| Single-digit floor accuracy (1-9) | 56.53% |
| Multi-digit floor accuracy (10-30) | 83.55% |
| B1 / B2 / B3 accuracy | 18.18% |
| Average OCR confidence score | 0.7414 |
| Unreadable crops (no valid output) | 153 |

### Orientation Analysis

| Orientation | Crops | Accuracy |
|---|---|---|
| Original | 471 | 57.75% |
| Horizontally flipped | 229 | 81.66% |

### Top OCR Failure Patterns

| Ground truth -> OCR | Count |
|---|---|
| GT=1 -> (empty) | 33 |
| GT=7 -> (empty) | 28 |
| GT=4 -> (empty) | 19 |
| GT=9 -> (empty) | 16 |
| GT=9 -> 0 | 10 |
| GT=B1 -> RA | 6 |
| GT=B1 -> 87 | 4 |
| GT=B1 -> TA | 4 |

---

## 3. End-to-End (Floor Identification)

A full-test-set end-to-end run was not completed due to CPU time constraints (EasyOCR across 761 crops).
The available result is the 5-image qualitative demo using the final production pipeline.
Source: `runs/detect/button_yolov8n_fast_baseline/final_demo/final_demo_results.csv`

| Image | Target | Found | Predicted | Match |
|---|---|---|---|---|
| 6_mp4-0016...jpg | 11 | No | 7 | FAIL |
| 12_mp4-0021...jpg | 20 | Yes | 20 | PASS |
| 12_mp4-0020...jpg | 19 | Yes | 19 | PASS |
| 3_mp4-0009...jpg | B1 | No | 7 | FAIL |
| 18_mp4-0038...jpg | 7 | Yes | 7 | PASS |

| End-to-end metric | Value |
|---|---|
| Total test cases (qualitative demo) | 5 |
| Correct floor identifications | **3** |
| End-to-end accuracy | **60% (3 / 5)** |

**Failure analysis:**
- target=11, predicted=7 — multi-digit button misread; OCR returned 7 instead of 11
- target=B1, predicted=7 — B1 glyph on reflective surface read as 7

> The 5-image demo is qualitative only. The best available quantitative estimate of per-button recognition
> is the full-test-set floor-label OCR accuracy of 66.62% (459 / 689 instances).

---

## 4. Summary

| Stage | Key metric | Value |
|---|---|---|
| Button Detection | mAP@50 (test set, 113 images) | **99.15%** |
| Button Detection | Precision / Recall | 98.82% / 99.31% |
| OCR | Floor-label accuracy | **66.62% (459 / 689)** |
| OCR | Multi-digit accuracy | 83.55% |
| OCR | B1/B2/B3 accuracy | 18.18% |
| End-to-end | Demo accuracy (5 images) | 60% (3 / 5) |

The detector performs at near-production quality. OCR is the primary bottleneck, especially for single-digit
floors and basement labels. Replacing or augmenting EasyOCR with a domain-fine-tuned scene-text model
would be the highest-impact improvement.
