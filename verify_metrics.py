
from ultralytics import YOLO

DATA_YAML = (
    r"D:\research papper\rs\data\Multi-Weather Pothole Detection (MWPD)"
    r"\Multi-Weather Pothole Detection (MWPD)\MWPD\data.yaml"
)

MODELS = {
    "YOLOv8n": (
        r"D:\research papper\rs\experiments\phase1\mwpd\weights\best.pt"
    ),
    "YOLOv8m": (
        r"D:\research papper\rs\experiments\phase1\mwpd_yolov8m\weights\best.pt"
    ),
}

if __name__ == "__main__":
    for name, checkpoint in MODELS.items():
        print(f"\n{'=' * 50}")
        print(f"Evaluating {name}")
        print(f"{'=' * 50}")

        model = YOLO(checkpoint)

        metrics = model.val(
            data=DATA_YAML,
            split="test",
            imgsz=640,
            batch=1,
            conf=0.001,
            iou=0.7,
            plots=True,
            project=(
                r"D:\research papper\rs\experiments\phase1"
                r"\official_validation"
            ),
            name=name,
            exist_ok=True,
            verbose=True,
        )

        print(f"\n{name} results")
        print(f"Precision: {metrics.box.mp:.5f}")
        print(f"Recall:    {metrics.box.mr:.5f}")
        print(f"mAP50:     {metrics.box.map50:.5f}")
        print(f"mAP50-95:  {metrics.box.map:.5f}")
