from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "RDD2022"
    / "raw"
    / "archive"
    / "RDD_SPLIT"
)

SPLITS = ["train", "val", "test"]


# ============================================================
# DATASET VALIDATION
# ============================================================

def inspect_split(split_name):
    split_path = DATASET_ROOT / split_name

    image_dir = split_path / "images"
    label_dir = split_path / "labels"

    print("\n" + "=" * 70)
    print(f"{split_name.upper()} DATASET")
    print("=" * 70)

    if not image_dir.exists():
        print(f"[ERROR] Image directory not found:")
        print(image_dir)
        return None

    if not label_dir.exists():
        print(f"[ERROR] Label directory not found:")
        print(label_dir)
        return None

    images = list(image_dir.glob("*.jpg"))
    labels = list(label_dir.glob("*.txt"))

    image_names = {image.stem for image in images}
    label_names = {label.stem for label in labels}

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    class_counter = Counter()
    image_class_counter = Counter()

    invalid_labels = []
    empty_labels = []

    total_boxes = 0
    widths = []
    heights = []

    # --------------------------------------------------------
    # Inspect images
    # --------------------------------------------------------

    for image_path in images:

        try:
            with Image.open(image_path) as image:
                widths.append(image.width)
                heights.append(image.height)

        except Exception as e:
            print(f"[ERROR] Cannot read image: {image_path}")
            print(f"        {e}")

    # --------------------------------------------------------
    # Inspect annotations
    # --------------------------------------------------------

    for label_path in labels:

        try:
            content = label_path.read_text().strip()

            if not content:
                empty_labels.append(label_path.name)
                continue

            classes_in_image = set()

            for line_number, line in enumerate(content.splitlines(), start=1):

                parts = line.split()

                # YOLO format:
                # class x_center y_center width height

                if len(parts) != 5:
                    invalid_labels.append(
                        (
                            label_path.name,
                            line_number,
                            "Expected 5 values",
                            line
                        )
                    )
                    continue

                try:
                    class_id = int(parts[0])

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                except ValueError:
                    invalid_labels.append(
                        (
                            label_path.name,
                            line_number,
                            "Non-numeric value",
                            line
                        )
                    )
                    continue

                # YOLO normalized coordinates should normally be [0, 1]
                if not (
                    0 <= x_center <= 1
                    and 0 <= y_center <= 1
                    and 0 <= width <= 1
                    and 0 <= height <= 1
                ):
                    invalid_labels.append(
                        (
                            label_path.name,
                            line_number,
                            "Coordinates outside [0,1]",
                            line
                        )
                    )

                class_counter[class_id] += 1
                classes_in_image.add(class_id)

                total_boxes += 1

            for class_id in classes_in_image:
                image_class_counter[class_id] += 1

        except Exception as e:

            invalid_labels.append(
                (
                    label_path.name,
                    0,
                    "Could not read file",
                    str(e)
                )
            )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(f"\nImages                : {len(images):,}")
    print(f"Labels                : {len(labels):,}")
    print(f"Total bounding boxes  : {total_boxes:,}")

    print("\nFile matching")
    print("-" * 40)

    print(f"Images without labels : {len(missing_labels):,}")
    print(f"Labels without images : {len(missing_images):,}")

    print("\nAnnotation problems")
    print("-" * 40)

    print(f"Empty label files     : {len(empty_labels):,}")
    print(f"Invalid annotations   : {len(invalid_labels):,}")

    print("\nClass distribution")
    print("-" * 40)

    if class_counter:

        for class_id, count in sorted(class_counter.items()):

            image_count = image_class_counter[class_id]

            print(
                f"Class {class_id:<3} "
                f"boxes={count:>8,} "
                f"images={image_count:>8,}"
            )

    else:
        print("No annotations found.")

    # --------------------------------------------------------
    # Image statistics
    # --------------------------------------------------------

    print("\nImage dimensions")
    print("-" * 40)

    if widths:

        dimension_counter = Counter(
            zip(widths, heights)
        )

        for (width, height), count in dimension_counter.most_common():

            print(
                f"{width} x {height}"
                f" : {count:,} images"
            )

    # --------------------------------------------------------
    # Return statistics
    # --------------------------------------------------------

    return {
        "images": len(images),
        "labels": len(labels),
        "boxes": total_boxes,
        "missing_labels": len(missing_labels),
        "missing_images": len(missing_images),
        "empty_labels": len(empty_labels),
        "invalid_labels": len(invalid_labels),
        "classes": dict(class_counter),
        "images_per_class": dict(image_class_counter),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RDD2022 DATASET INSPECTION")
    print("=" * 70)

    print("\nDataset location:")
    print(DATASET_ROOT)

    if not DATASET_ROOT.exists():

        print("\n[ERROR] Dataset directory does not exist.")
        print("\nExpected structure:")
        print(
            "data/RDD2022/raw/archive/RDD_SPLIT/"
        )
        return

    results = {}

    for split in SPLITS:

        results[split] = inspect_split(split)

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print("\n\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    print(
        f"{'Split':<10}"
        f"{'Images':>12}"
        f"{'Labels':>12}"
        f"{'Boxes':>12}"
    )

    print("-" * 46)

    for split, stats in results.items():

        if stats is None:
            continue

        print(
            f"{split:<10}"
            f"{stats['images']:>12,}"
            f"{stats['labels']:>12,}"
            f"{stats['boxes']:>12,}"
        )

    print("\nInspection completed.")


if __name__ == "__main__":
    main()