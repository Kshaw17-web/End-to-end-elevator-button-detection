import os
from ultralytics import YOLO

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_DEFAULT_MODEL = os.path.join(_ROOT, "models", "best.pt")
_FALLBACK_MODEL = os.path.join(
    _ROOT, "runs", "detect", "button_yolov8n_fast_baseline", "weights", "best.pt"
)


class ButtonDetector:
    def __init__(self, model_path=None, conf=0.45):
        if model_path is None:
            model_path = _DEFAULT_MODEL if os.path.exists(_DEFAULT_MODEL) else _FALLBACK_MODEL
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image_source):
        results = self.model.predict(image_source, conf=self.conf, verbose=False)
        detections = []
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": round(float(box.conf[0]), 4),
                })
        return detections
