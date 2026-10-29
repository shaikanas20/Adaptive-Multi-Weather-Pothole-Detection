from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "MWPD"
    / "test"
    / "images"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions"
    / "source_images.txt"
)

MAX_IMAGES = 500
SEED = 42


def main():

    images = sorted(SOURCE_DIR.glob("*.jpg"))

    random.seed(SEED)
    random.shuffle(images)

    selected = images[:MAX_IMAGES]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w") as f:

        for image in selected:
            f.write(image.name + "\n")

    print("=" * 60)
    print("CONTROLLED CONDITION SOURCE LIST")
    print("=" * 60)

    print(f"Available test images : {len(images)}")
    print(f"Selected images       : {len(selected)}")
    print(f"Saved to              : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()