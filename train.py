import argparse
import os

DATA_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_button", "data.yaml")
PRETRAINED = "yolov8n.pt"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "detect")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n elevator button detector.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--name", default="button_yolov8n", help="Run name inside runs/detect/.")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(PRETRAINED)
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        project=OUTPUT_DIR,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
