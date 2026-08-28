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

**YOLO-only inference latency** (measured — `models/best.pt`, 113 test images, CPU, imgsz=512, conf=0.45):

| Stage | Per-image (ms) |
|---|---|
| Preprocess | 2.36 ms |
| **Neural-network inference** | **72.8 ms** |
| Postprocess | 1.29 ms |
| **Total YOLO pipeline** | **76.45 ms** |

| Throughput | Value |
|---|---|
| Inference-only FPS | **13.7 FPS** |
| Full YOLO pipeline FPS | **13.1 FPS** |

> These figures cover **YOLO detection only** (no EasyOCR). Measured via `results[0].speed` from the
> Ultralytics API on a warm model (one warm-up image excluded). EasyOCR recognition adds approximately
> 0.5–1.5 s per button crop on the same hardware, making OCR the dominant latency in the full pipeline.
> Full training curves and per-epoch metrics are in `runs/detect/button_yolov8n_fast_baseline/results.csv`.
> Raw benchmark data: `runs/detect/button_yolov8n_fast_baseline/yolo_latency_benchmark.json`.

---

## 2. OCR (Button Text Recognition)

Evaluation on all 113 test-set images, matching YOLO detections to ground-truth labels.
Source: `runs/detect/button_yolov8n_fast_baseline/ocr_evaluation/ocr_results.csv` (761 rows).

> Note: Results are from the V1 OCR evaluation script (`scripts/ocr_evaluation.py`), which used CLAHE + Otsu
> preprocessing. The production pipeline (`src/ocr.py`) uses a 4-pass EasyOCR approach with grayscale +
> digit/B allowlist. 

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

> The 5-image demo is qualitative only. See Section 3b below for the quantitative evaluation.

---

## 3b. Quantitative End-to-End Evaluation (Production Pipeline)

A deterministic 20-image subset of the 113-image held-out test set was evaluated using the actual
production pipeline (`detect_floor.py` + `src/`). Images were selected with `random.sample(seed=42)`
on the sorted test image list. Three of the 20 images contained only icon buttons (Open/Close/Emergency)
with no floor labels in `dataset_button/original_labels/` and were excluded, yielding **17 valid test cases**.

For each image, the most frequent floor label in the ground-truth metadata was used as the target floor.
The target was supplied to `detect_floor.py --target-floor` and the pipeline result was compared against it.
The ground truth did not influence the prediction — it was only used to select the target argument and
judge correctness after the pipeline returned.

Source: `runs/detect/button_yolov8n_fast_baseline/end_to_end_evaluation.csv`

| Image | Target | Predicted | Correct? |
|---|---|---|---|
| 11_mp4-0004…3ea3.jpg | 4 | 4 | ✅ |
| 11_mp4-0004…b887.jpg | 4 | 4 | ✅ |
| 12_mp4-0017…jpg | B1 | B1 | ✅ |
| 12_mp4-0020…jpg | B1 | 3 | ❌ |
| 12_mp4-0021…jpg | 9 | 9 | ✅ |
| 12_mp4-0024…jpg | 29 | 29 | ✅ |
| 13_mp4-0028…jpg | B1 | 2 | ❌ |
| 16_mp4-0002…jpg | 8 | 8 | ✅ |
| 17_mp4-0000…jpg | 6 | 6 | ✅ |
| 19_mp4-0005…jpg | B1 | 8 | ❌ |
| 1_mp4-0008…jpg | B1 | B1 | ✅ |
| 3_mp4-0004…jpg | 7 | 6 | ❌ |
| 3_mp4-0018…jpg | 9 | 9 | ✅ |
| 3_mp4-0027…jpg | B1 | B1 | ✅ |
| 4_mp4-0004…jpg | 8 | 17 | ❌ |
| 4_mp4-0008…jpg | 7 | 15 | ❌ |
| 6_mp4-0008…jpg | 8 | 2 | ❌ |

| Metric | Value |
|---|---|
| Images sampled (seed=42) | 20 |
| Valid test cases (floor labels present) | **17** |
| Skipped (icon-only panels) | 3 |
| **Correct floor identifications** | **10** |
| **Incorrect floor identifications** | **7** |
| **End-to-end accuracy** | **10 / 17 = 58.8%** |

**Failure analysis:**
- B1/B2/B3 failures dominate: 3 of the 7 failures had target=B1, consistent with the 18.18% B1 OCR accuracy.
- Single-digit mismatches (7→6, 8→17, 7→15, 8→2): OCR returned a different valid floor label that scored higher.

> **Scope note:** This is a deterministic 20-image sample from the 113-image held-out test set.
> It is NOT a full-test-set end-to-end evaluation. Results are indicative of production pipeline
> performance on this hardware (CPU, AMD Ryzen 5 5500U).

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
| End-to-end | **Quantitative (20-image sample, 17 valid)** | **58.8% (10 / 17)** |

The detector performs at near-production quality. OCR is the primary bottleneck, especially for single-digit
floors and basement labels. Replacing or augmenting EasyOCR with a domain-fine-tuned scene-text model
would be the highest-impact improvement.
