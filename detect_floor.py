import argparse
import os
import sys
import cv2

from src.detector import ButtonDetector
from src.ocr import recognize_button
from src.floor_selector import select, normalize, FLOOR_VOCAB


def run(image_path, target_floor=None, save_dir=None):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"ERROR: cannot read image: {image_path}")
        sys.exit(1)

    detector = ButtonDetector()
    detections = detector.detect(image_path)

    buttons = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        yolo_conf = det["confidence"]
        candidates = recognize_button(img_bgr, x1, y1, x2, y2)
        floor, ocr_conf = select(candidates)
        buttons.append({
            "bbox": [x1, y1, x2, y2],
            "center": [(x1 + x2) // 2, (y1 + y2) // 2],
            "yolo_confidence": yolo_conf,
            "floor": floor,
            "ocr_confidence": ocr_conf,
        })

    target_button = None
    if target_floor is not None:
        tf_norm = normalize(target_floor)
        for b in buttons:
            if b["floor"] == tf_norm:
                if target_button is None or b["ocr_confidence"] > target_button["ocr_confidence"]:
                    target_button = b

    target_found = target_button is not None

    print(f"total_buttons_detected={len(detections)}")
    if target_floor is not None:
        print(f"target_floor={normalize(target_floor)}")
        print(f"target_found={str(target_found).lower()}")
        if target_found:
            tb = target_button
            print(f"bbox={tb['bbox']}")
            print(f"center={tb['center']}")
            print(f"detection_confidence={tb['yolo_confidence']}")
            print(f"ocr_confidence={tb['ocr_confidence']}")
            print(f"ocr_text={tb['floor']}")
    else:
        for i, b in enumerate(buttons):
            print(
                f"button[{i}] floor={b['floor']} bbox={b['bbox']} center={b['center']} "
                f"yolo_conf={b['yolo_confidence']} ocr_conf={b['ocr_confidence']}"
            )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        ann = img_bgr.copy()
        for b in buttons:
            x1, y1, x2, y2 = b["bbox"]
            is_tgt = b is target_button
            color = (0, 220, 0) if is_tgt else (255, 140, 0)
            thick = 3 if is_tgt else 2
            cv2.rectangle(ann, (x1, y1), (x2, y2), color, thick)
            cv2.putText(
                ann,
                f"{b['floor']} {b['ocr_confidence']:.2f}",
                (x1, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
        if target_floor is not None:
            status = "[FOUND]" if target_found else "[NOT FOUND]"
            cv2.putText(
                ann,
                f"Target: {normalize(target_floor)} {status}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 210, 255),
                2,
            )
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(save_dir, stem + "_annotated.jpg")
        cv2.imwrite(out_path, ann)
        print(f"annotated_image={out_path}")

    return target_found, buttons, target_button


def main():
    parser = argparse.ArgumentParser(
        description="Detect elevator floor buttons and optionally locate a target floor."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--target-floor", default=None, help="Target floor label (e.g. '14', 'B1').")
    parser.add_argument("--save-dir", default=None, help="Directory to save annotated output image.")
    args = parser.parse_args()
    run(args.image, target_floor=args.target_floor, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
