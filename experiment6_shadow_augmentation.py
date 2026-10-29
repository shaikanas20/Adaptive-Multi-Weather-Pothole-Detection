
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================
ROOT = Path(r"D:\research papper\rs")

DATA_ROOT = (
    ROOT / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "MWPD"
)

BASELINE_WEIGHTS = (
    ROOT / "experiments" / "phase1"
    / "mwpd" / "weights" / "best.pt"
)

EXPERIMENT_DIR = (
    ROOT / "experiments" / "phase1"
    / "experiment6_shadow_augmentation"
)

DATASET_DIR = EXPERIMENT_DIR / "dataset"
TRAIN_IMAGES_OUT = DATASET_DIR / "images" / "train"
TRAIN_LABELS_OUT = DATASET_DIR / "labels" / "train"
VAL_IMAGES_OUT = DATASET_DIR / "images" / "val"
VAL_LABELS_OUT = DATASET_DIR / "labels" / "val"

SEED = 42
EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Shadow augmentation
# ============================================================
def add_synthetic_shadow(image):
    """
    Add a soft, semi-transparent polygonal shadow.
    Bounding-box labels remain unchanged.

    This is a controlled synthetic augmentation, not a claim
    that the shadow perfectly reproduces real weather.
    """
    h, w = image.shape[:2]

    overlay = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    # Create a randomly oriented polygon crossing part of the image.
    cx = random.randint(int(0.2 * w), int(0.8 * w))
    cy = random.randint(int(0.2 * h), int(0.8 * h))

    length = random.uniform(0.35, 0.8) * w
    width = random.uniform(0.2, 0.6) * min(h, w)
    angle = random.uniform(0, np.pi)

    ux, uy = np.cos(angle), np.sin(angle)
    vx, vy = -uy, ux

    points = []
    for along, across in [
        (-length / 2, -width / 2),
        (length / 2, -width / 2),
        (length / 2, width / 2),
        (-length / 2, width / 2),
    ]:
        x = int(cx + along * ux + across * vx)
        y = int(cy + along * uy + across * vy)
        points.append([x, y])

    polygon = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    # Blur mask boundaries for softer shadow edges.
    blur_kernel = max(3, (min(h, w) // 20) | 1)
    mask = cv2.GaussianBlur(
        mask, (blur_kernel, blur_kernel), 0
    )

    alpha = random.uniform(0.18, 0.42)
    shadow_strength = (mask.astype(np.float32) / 255.0) * alpha

    for channel in range(3):
        overlay[:, :, channel] = (
            image[:, :, channel].astype(np.float32)
            * (1.0 - shadow_strength)
        ).clip(0, 255).astype(np.uint8)

    return overlay


def copy_image_and_label(image_path, label_path, output_stem):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    output_image = TRAIN_IMAGES_OUT / f"{output_stem}{image_path.suffix.lower()}"
    output_label = TRAIN_LABELS_OUT / f"{output_stem}.txt"

    shutil.copy2(image_path, output_image)

    if label_path.exists():
        shutil.copy2(label_path, output_label)
    else:
        # Empty label means no annotated objects in the image.
        output_label.write_text("", encoding="utf-8")


# ============================================================
# Validate inputs
# ============================================================

# ============================================================
# Train using the already-prepared dataset
# ============================================================
def main():
    if not BASELINE_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Baseline weights not found: {BASELINE_WEIGHTS}"
        )

    dataset_yaml = DATASET_DIR / "data.yaml"

    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Prepared dataset YAML not found: {dataset_yaml}"
        )

    if not any(TRAIN_IMAGES_OUT.iterdir()):
        raise RuntimeError("Prepared training images are missing.")

    if not any(VAL_IMAGES_OUT.iterdir()):
        raise RuntimeError("Prepared validation images are missing.")

    print("Reusing prepared dataset:", DATASET_DIR)
    print("Training images:", len(list(TRAIN_IMAGES_OUT.iterdir())))
    print("Validation images:", len(list(VAL_IMAGES_OUT.iterdir())))

    model = YOLO(str(BASELINE_WEIGHTS))

    model.train(
        data=str(dataset_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=10,
        seed=SEED,
        workers=0,
        device=0,
        project=str(EXPERIMENT_DIR / "training"),
        name="yolov8n_shadow_aug_retry",
        exist_ok=False,
        pretrained=True,
        deterministic=True,
        plots=True,
        save=True,
        verbose=True,
    )

    print("\nTraining completed.")
    print(
        "Check for best.pt at:",
        EXPERIMENT_DIR / "training"
        / "yolov8n_shadow_aug_retry" / "weights" / "best.pt"
    )


if __name__ == "__main__":
    main()
