from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions"
)

CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection",
]


def read_labels(label_path):
    """
    Read YOLO labels and return valid annotations.
    """

    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                labels.append(
                    (
                        class_id,
                        x,
                        y,
                        width,
                        height
                    )
                )

            except ValueError:
                continue

    return labels


def validate_label(label):

    class_id, x, y, width, height = label

    # YOLO normalized coordinates
    if not (0 <= x <= 1):
        return False

    if not (0 <= y <= 1):
        return False

    if not (0 < width <= 1):
        return False

    if not (0 < height <= 1):
        return False

    if class_id < 0:
        return False

    return True


def main():

    print("=" * 70)
    print("CONTROLLED CONDITION DATASET VERIFICATION")
    print("=" * 70)

    overall_images = 0
    overall_boxes = 0
    overall_invalid = 0

    for condition in CONDITIONS:

        image_dir = DATASET_DIR / condition / "images"
        label_dir = DATASET_DIR / condition / "labels"

        print()
        print("-" * 70)
        print(f"Condition: {condition}")
        print("-" * 70)

        if not image_dir.exists():

            print("ERROR: Image directory missing")
            continue

        images = sorted(
            image_dir.glob("*.jpg")
        )

        image_count = 0
        label_count = 0
        box_count = 0
        invalid_count = 0
        missing_labels = 0

        for image_path in images:

            image_count += 1

            # ------------------------------------------------
            # Check image
            # ------------------------------------------------

            image = cv2.imread(
                str(image_path)
            )

            if image is None:

                print(
                    f"ERROR: Cannot read {image_path.name}"
                )

                continue

            # ------------------------------------------------
            # Corresponding label
            # ------------------------------------------------

            label_path = (
                label_dir
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                missing_labels += 1
                continue

            labels = read_labels(
                label_path
            )

            label_count += 1

            for label in labels:

                box_count += 1

                if not validate_label(label):

                    invalid_count += 1

        print(f"Images           : {image_count}")
        print(f"Label files      : {label_count}")
        print(f"Missing labels   : {missing_labels}")
        print(f"Bounding boxes   : {box_count}")
        print(f"Invalid boxes    : {invalid_count}")

        overall_images += image_count
        overall_boxes += box_count
        overall_invalid += invalid_count

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(f"Total images     : {overall_images}")
    print(f"Total boxes      : {overall_boxes}")
    print(f"Invalid boxes    : {overall_invalid}")

    if overall_invalid == 0:

        print()
        print("RESULT: ALL ANNOTATIONS ARE VALID.")

    else:

        print()
        print("RESULT: INVALID ANNOTATIONS FOUND.")


if __name__ == "__main__":
    main()