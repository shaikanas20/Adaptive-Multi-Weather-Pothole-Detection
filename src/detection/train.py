from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "rdd2022.yaml"
)

MODEL_NAME = "yolov8n.pt"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "phase1"
    / "baseline"
)


def train_baseline():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("RDD2022 BASELINE TRAINING")
    print("=" * 60)

    print(f"Dataset: {DATASET_CONFIG}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(DATASET_CONFIG),

        # Fixed baseline configuration
        imgsz=640,
        batch=16,
        epochs=70,

        # Reproducibility
        seed=42,

        # Hardware
        device=0,

        # Training output
        project=str(OUTPUT_DIR),
        name="yolov8n_fixed",

        # Disable unnecessary experiment complexity
        pretrained=True,

        # Save best model
        save=True,

        # Validation during training
        val=True,

        # Workers
        workers=4,

        # Training verbosity
        verbose=True,
    )

    print("\nTraining completed.")

    print("\nBest model should be located at:")

    print(
        OUTPUT_DIR
        / "yolov8n_fixed"
        / "weights"
        / "best.pt"
    )


def train_mwpd(epochs=50):

    mwpd_config = PROJECT_ROOT / "configs" / "mwpd.yaml"
    output_dir = PROJECT_ROOT / "experiments" / "phase1" / "mwpd"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("MWPD BASELINE TRAINING")
    print("=" * 60)

    print(f"Dataset: {mwpd_config}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {output_dir}")

    # Standard pretrained YOLOv8 model - NO fine-tuned or RDD2022 checkpoints
    model = YOLO(MODEL_NAME)

    results = model.train(
        data=str(mwpd_config),

        # Baseline configuration
        imgsz=640,
        batch=16,
        epochs=50,

        # Reproducibility
        seed=42,

        # Hardware
        device=0,

        # Disable AMP to prevent CUDA launch failures on Windows
        amp=False,

        # Training output saved to experiments/phase1/mwpd
        project=str(PROJECT_ROOT / "experiments" / "phase1"),
        name="mwpd",
        exist_ok=True,

        # Standard pretrained backbone initialization
        pretrained=True,

        # Save best model
        save=True,

        # Validation during training
        val=True,

        # Workers (0 avoids Windows OpenCV multi-worker memory allocation issues)
        workers=0,

        # Training verbosity
        verbose=True,
    )

    print("\nMWPD Training completed.")

    print("\nBest model is located at:")

    print(
        output_dir
        / "weights"
        / "best.pt"
    )


# ============================================================
# MWPD YOLOv8M TRAINING
# ============================================================

def train_mwpd_yolov8m(epochs=50):

    mwpd_config = PROJECT_ROOT / "configs" / "mwpd.yaml"
    output_dir = PROJECT_ROOT / "experiments" / "phase1" / "mwpd_yolov8m"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("MWPD YOLOv8M BASELINE TRAINING")
    print("=" * 60)

    print(f"Dataset: {mwpd_config}")
    print("Model: yolov8m.pt")
    print(f"Output: {output_dir}")

    # Standard pretrained YOLOv8M model
    # This is NOT initialized from the YOLOv8n MWPD model
    model = YOLO("yolov8m.pt")

    results = model.train(
        data=str(mwpd_config),

        # Same baseline configuration
        imgsz=640,
        batch=4,
        epochs=50,

        # Reproducibility
        seed=42,

        # Hardware
        device=0,

        # Disable AMP to prevent CUDA launch failures on Windows
        amp=False,

        # Training output
        project=str(PROJECT_ROOT / "experiments" / "phase1"),
        name="mwpd_yolov8m",
        exist_ok=True,

        # Standard pretrained YOLOv8M initialization
        pretrained=True,

        # Save best model
        save=True,

        # Validation during training
        val=True,

        # Workers
        workers=0,

        # Training verbosity
        verbose=True,
    )

    print("\nMWPD YOLOv8M Training completed.")

    print("\nBest model is located at:")

    print(
        output_dir
        / "weights"
        / "best.pt"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Object Detection Training Script"
    )

    parser.add_argument(
        "--dataset",
        choices=[
            "rdd2022",
            "mwpd",
            "mwpd_yolov8m"
        ],
        default="rdd2022",
        help="Dataset/model configuration to train",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50 for MWPD, 70 for RDD2022 baseline)",
    )

    args = parser.parse_args()

    if args.dataset == "mwpd":

        train_mwpd(
            epochs=args.epochs
        )

    elif args.dataset == "mwpd_yolov8m":

        train_mwpd_yolov8m(
            epochs=args.epochs
        )

    else:

        train_baseline()