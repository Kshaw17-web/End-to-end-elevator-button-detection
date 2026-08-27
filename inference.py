import argparse
import os
import sys
import cv2

from src.detector import ButtonDetector
from src.ocr import recognize_button
from src.floor_selector import select


def run(image_path, save_path=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"ERROR: cannot read image: {image_path}")
        sys.exit(1)

    detector = ButtonDetector()
    detections = detector.detect(image_path)

    print(f"total_buttons_detected={len(detections)}")

    results = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        yolo_conf = det["confidence"]
        candidates = recognize_button(img_bgr, x1, y1, x2, y2)
        floor, ocr_conf = select(candidates)
        center = [(x1 + x2) // 2, (y1 + y2) // 2]
        results.append({
            "index": i,
            "floor": floor,
            "bbox": [x1, y1, x2, y2],
            "center": center,
            "yolo_confidence": yolo_conf,
            "ocr_confidence": ocr_conf,
        })
        print(
            f"button[{i}] floor={floor} bbox={[x1,y1,x2,y2]} center={center} "
            f"yolo_conf={yolo_conf} ocr_conf={ocr_conf}"
        )

    if save_path and results:
        ann = img_bgr.copy()
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(ann, (x1, y1), (x2, y2), (255, 140, 0), 2)
            cv2.putText(
                ann,
                f"{r['floor']} {r['ocr_confidence']:.2f}",
                (x1, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 140, 0),
                2,
            )
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, ann)
        print(f"annotated_image={save_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Elevator button detection and OCR inference.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--save", default=None, help="Optional path to save annotated output image.")
    args = parser.parse_args()
    run(args.image, save_path=args.save)


if __name__ == "__main__":
    main()
