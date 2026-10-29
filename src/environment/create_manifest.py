from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "RDD2022"
    / "controlled_conditions"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "phase1"
    / "environment"
)

OUTPUT_FILE = OUTPUT_DIR / "condition_manifest.csv"

CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection",
]

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Alligator Crack",
    3: "Pothole",
    4: "Other / Repair Mark",
}


def read_labels(label_path):

    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:
                class_id = int(parts[0])

                labels.append(class_id)

            except ValueError:
                continue

    return labels


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    normal_image_dir = (
        DATASET_DIR
        / "normal"
        / "images"
    )

    images = sorted(
        normal_image_dir.glob("*.jpg")
    )

    rows = []

    for image_path in images:

        image_name = image_path.name
        image_stem = image_path.stem

        # ----------------------------------------------------
        # Read labels from normal condition
        # ----------------------------------------------------

        label_path = (
            DATASET_DIR
            / "normal"
            / "labels"
            / f"{image_stem}.txt"
        )

        class_ids = read_labels(
            label_path
        )

        # ----------------------------------------------------
        # Class presence
        # ----------------------------------------------------

        class_set = set(class_ids)

        row = {
            "source_image": image_name,
            "number_of_objects": len(class_ids),

            "longitudinal_crack_present":
                int(0 in class_set),

            "transverse_crack_present":
                int(1 in class_set),

            "alligator_crack_present":
                int(2 in class_set),

            "pothole_present":
                int(3 in class_set),

            "other_damage_present":
                int(4 in class_set),
        }

        # ----------------------------------------------------
        # Store paths for every condition
        # ----------------------------------------------------

        for condition in CONDITIONS:

            image_file = (
                DATASET_DIR
                / condition
                / "images"
                / image_name
            )

            label_file = (
                DATASET_DIR
                / condition
                / "labels"
                / f"{image_stem}.txt"
            )

            row[
                f"{condition}_image"
            ] = str(
                image_file.relative_to(
                    PROJECT_ROOT
                )
            )

            row[
                f"{condition}_label"
            ] = str(
                label_file.relative_to(
                    PROJECT_ROOT
                )
            )

        rows.append(row)

    # --------------------------------------------------------
    # CSV columns
    # --------------------------------------------------------

    fieldnames = [
        "source_image",
        "number_of_objects",

        "longitudinal_crack_present",
        "transverse_crack_present",
        "alligator_crack_present",
        "pothole_present",
        "other_damage_present",
    ]

    for condition in CONDITIONS:

        fieldnames.extend([
            f"{condition}_image",
            f"{condition}_label",
        ])

    # --------------------------------------------------------
    # Write manifest
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 70)
    print("CONTROLLED CONDITION MANIFEST")
    print("=" * 70)

    print(f"Source scenes : {len(rows)}")
    print(f"Conditions    : {len(CONDITIONS)}")
    print(f"Output        : {OUTPUT_FILE}")

    print()
    print("Manifest created successfully.")


if __name__ == "__main__":
    main()