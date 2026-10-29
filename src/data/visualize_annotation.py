from pathlib import Path
import random

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "RDD2022"
    / "raw"
    / "archive"
    / "RDD_SPLIT"
)

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Other / Repair Mark",
}


def draw_annotations(split="train", num_images=10):
    image_dir = DATASET_ROOT / split / "images"
    label_dir = DATASET_ROOT / split / "labels"

    output_dir = PROJECT_ROOT / "experiments" / "phase1" / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(image_dir.glob("*.jpg"))

    if not images:
        print(f"No images found in {image_dir}")
        return

    selected_images = random.sample(
        images,
        min(num_images, len(images))
    )

    for image_path in selected_images:

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path}")
            continue

        height, width = image.shape[:2]

        label_path = label_dir / f"{image_path.stem}.txt"

        if label_path.exists():

            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                # Convert normalized YOLO coordinates
                # to pixel coordinates.

                x_center *= width
                y_center *= height

                box_width *= width
                box_height *= height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)

                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                # Keep coordinates inside image
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width - 1, x2)
                y2 = min(height - 1, y2)

                class_name = CLASS_NAMES.get(
                    class_id,
                    f"Unknown Class {class_id}"
                )

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    class_name,
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        output_path = output_dir / f"{split}_{image_path.stem}.jpg"

        cv2.imwrite(
            str(output_path),
            image
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    draw_annotations(
        split="train",
        num_images=10
    )