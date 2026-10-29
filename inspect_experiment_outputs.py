
from pathlib import Path
import random

import cv2
import matplotlib.pyplot as plt


ROOT = Path(r"D:\research papper\rs")

ORIGINAL_IMAGES = (
    ROOT / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "MWPD" / "train" / "images"
)

ORIGINAL_LABELS = (
    ROOT / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "MWPD" / "train" / "labels"
)

EXP6_DATASET = (
    ROOT / "experiments" / "phase1"
    / "experiment6_shadow_augmentation" / "dataset"
)

SYNTHETIC_IMAGES = EXP6_DATASET / "images" / "train"
SYNTHETIC_LABELS = EXP6_DATASET / "labels" / "train"

OUTPUT = (
    ROOT / "experiments" / "phase1"
    / "experiment6_shadow_augmentation"
    / "shadow_contact_sheet.png"
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NUM_PAIRS = 5


def read_labels(label_path):
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label file: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def draw_boxes(image, label_path):
    image = image.copy()
    height, width = image.shape[:2]

    for line in read_labels(label_path):
        values = line.split()

        if len(values) != 5:
            raise ValueError(f"Invalid YOLO label: {line}")

        class_id, xc, yc, w, h = map(float, values)

        x1 = int((xc - w / 2) * width)
        y1 = int((yc - h / 2) * height)
        x2 = int((xc + w / 2) * width)
        y2 = int((yc + h / 2) * height)

        cv2.rectangle(
            image, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    for folder in [
        ORIGINAL_IMAGES,
        ORIGINAL_LABELS,
        SYNTHETIC_IMAGES,
        SYNTHETIC_LABELS,
    ]:
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

    # Match synthetic pairs using their shared filename stem.
    image_files = [
        p for p in SYNTHETIC_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    originals = {}
    shadows = {}

    for path in image_files:
        if path.stem.endswith("_original"):
            key = path.stem[:-len("_original")]
            originals[key] = path

        elif path.stem.endswith("_shadow"):
            key = path.stem[:-len("_shadow")]
            shadows[key] = path

    common_keys = sorted(set(originals) & set(shadows))

    if not common_keys:
        raise RuntimeError(
            "No matching _original/_shadow pairs found."
        )

    rng = random.Random(42)
    selected = rng.sample(
        common_keys, min(NUM_PAIRS, len(common_keys))
    )

    print("Synthetic original images:", len(originals))
    print("Synthetic shadow images:", len(shadows))
    print("Matched pairs:", len(common_keys))

    fig, axes = plt.subplots(
        len(selected), 2,
        figsize=(14, 4 * len(selected)),
        squeeze=False,
    )

    for row, key in enumerate(selected):
        original_path = originals[key]
        shadow_path = shadows[key]

        # Experiment 6 filenames may differ from the original
        # dataset filenames, so use the labels shipped with
        # the Experiment 6 dataset.
        original_label = SYNTHETIC_LABELS / f"{original_path.stem}.txt"
        shadow_label = SYNTHETIC_LABELS / f"{shadow_path.stem}.txt"

        original = cv2.imread(str(original_path))
        shadow = cv2.imread(str(shadow_path))

        if original is None or shadow is None:
            raise RuntimeError(f"Could not read pair: {key}")

        original = draw_boxes(original, original_label)
        shadow = draw_boxes(shadow, shadow_label)

        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"Original: {original_path.name}")
        axes[row, 1].imshow(shadow)
        axes[row, 1].set_title(f"Synthetic shadow: {shadow_path.name}")

        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    plt.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nContact sheet saved to:\n{OUTPUT}")


if __name__ == "__main__":
    main()
