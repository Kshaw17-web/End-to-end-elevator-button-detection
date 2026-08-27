# OCR Evaluation Report
## Run: button_yolov8n_fast_baseline

## Configuration

| Parameter | Value |
|---|---|
| YOLO model | best.pt (epoch 21) |
| YOLO confidence threshold | 0.45 |
| IoU threshold | 0.5 |
| OCR engine | EasyOCR English CPU |
| Braille mask | lower 15% removed |
| Preprocessing | gray, CLAHE, Otsu |
| Orientations | original + horizontally flipped |

## Results Summary

| Metric | Value |
|---|---|
| Total matched detections | 761 |
| Non-icon detections evaluated | 700 |
| Floor label instances | 689 |
| Exact OCR accuracy (all non-icon) | 65.57% |
| Normalized OCR accuracy | 65.57% |
| Floor label exact accuracy | 66.62% |
| Single-digit accuracy | 56.53% |
| Multi-digit accuracy | 83.55% |
| Basement label accuracy | 18.18% |
| Average OCR confidence | 0.7414 |
| Unreadable crops | 153 |

## Orientation Analysis

| Orientation | Count | Accuracy |
|---|---|---|
| Original | 471 | 57.75% |
| Flipped | 229 | 81.66% |

## Top OCR Failure Patterns

| Pattern | Count |
|---|---|
| GT=1 -> OCR=(empty) | 33 |
| GT=7 -> OCR=(empty) | 28 |
| GT=4 -> OCR=(empty) | 19 |
| GT=9 -> OCR=(empty) | 16 |
| GT=6 -> OCR=(empty) | 10 |
| GT=9 -> OCR=0 | 10 |
| GT=11 -> OCR=(empty) | 7 |
| GT=8 -> OCR=(empty) | 7 |
| GT=5 -> OCR=(empty) | 7 |
| GT=B1 -> OCR=RA | 6 |
| GT=3 -> OCR=(empty) | 5 |
| GT=B1 -> OCR=87 | 4 |
| GT=B1 -> OCR=TA | 4 |
| GT=6 -> OCR=0 | 4 |
| GT=29 -> OCR=(empty) | 3 |