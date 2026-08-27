# dataset_button - Preparation Report

## Original Dataset

| Field | Value |
|---|---|
| Source | ACC Elevator Button - Roboflow acc-stwam/elevator-button-jinxe-qb0gs v1 |
| License | CC BY 4.0 |
| Original images | 1139 |
| Original label files | 1139 |
| Original classes | 38 |
| Original total bboxes | 6801 |
| Splits on disk | train only (valid/test absent) |

## Class Consolidation

The 38 original classes represent every physically distinct button or control on an elevator panel:

- Floor numbers: 1-30
- Basement floors: B1, B2, B3
- Function buttons: Open, Close, Up, Down, Emergency

**Why a single button class?**

The first stage of the Lastmile Robotics pipeline is button localization: detect and bound every button in the camera image. The label text is read subsequently by an OCR module. For localization, all buttons are structurally equivalent. Collapsing all 38 semantic classes into a single button class (class 0) simplifies the detector, maximizes training samples per class, and matches the intended inference pipeline.

The original semantic labels are preserved in original_labels/ as JSON for downstream OCR evaluation.

## Malformed Annotation Removal

During verification, 2 of 135 Emergency annotations were found to be geometrically degenerate:

| Field | Normal Emergency | Malformed |
|---|---|---|
| Width (normalized) | ~0.21 | 0.012 |
| Height (normalized) | ~0.22 | 0.697 |
| Cause | - | Button at extreme left image edge, nearly off-frame |

Removal criterion: w < 0.05 AND h > 0.50

- Malformed annotations removed: 2
- Images discarded entirely: 0 (both images had other valid annotations)
- Cleaned images carried forward: 1133 (6 label files had no corresponding image on disk and were skipped)

## Dataset Split

| Parameter | Value |
|---|---|
| Strategy | Image-level (all annotations for an image stay together) |
| Train | 80% |
| Validation | 10% |
| Test | 10% |
| Random seed | 42 |
| Stratification | Greedy rarity-first: images containing rarer classes assigned to valid/test first to ensure all 38 classes appear in every split |

## Final Statistics

| Metric | Value |
|---|---|
| Cleaned images | 1133 |
| Train images | 907 |
| Validation images | 113 |
| Test images | 113 |
| Train annotations | 5258 |
| Validation annotations | 780 |
| Test annotations | 761 |
| Total annotations | 6799 |
| Removed malformed | 2 |
| Classes | 1 (button, class 0) |

All 38 original semantic classes appear in every split.

## Integrity Verification

| Check | Result |
|---|---|
| Every label file contains only class ID 0 | PASS (0 errors) |
| Every bounding box has valid normalized coordinates | PASS (0 errors) |
| Every image has a corresponding label file | PASS |
| Every label file has a corresponding image | PASS |
| All 38 original classes present in every split | PASS |
| No original dataset file modified | PASS (original train/: 1139 images, 1139 labels unchanged) |
| Images not recompressed or resized | PASS (shutil.copy2 byte-for-byte copy) |
