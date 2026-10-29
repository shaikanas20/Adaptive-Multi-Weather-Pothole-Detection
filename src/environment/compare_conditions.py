from pathlib import Path
import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "phase1"
    / "environment"
    / "visual_checks"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection"
]


def load_image(condition, filename):

    path = (
        DATASET_ROOT
        / condition
        / "images"
        / filename
    )

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(path)

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


def create_comparison(filename):

    images = []

    for condition in CONDITIONS:

        image = load_image(
            condition,
            filename
        )

        images.append(
            (condition, image)
        )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    for ax, (condition, image) in zip(
        axes.flatten(),
        images
    ):

        ax.imshow(image)
        ax.set_title(condition)
        ax.axis("off")

    plt.suptitle(
        f"Controlled Condition Comparison: {filename}",
        fontsize=16
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / f"{Path(filename).stem}_comparison.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")


def main():

    normal_dir = (
        DATASET_ROOT
        / "normal"
        / "images"
    )

    image_files = sorted(
        normal_dir.glob("*.jpg")
    )

    if not image_files:

        print("No images found.")

        return

    # Inspect the first 5 images.
    selected = image_files[:5]

    for image_path in selected:

        create_comparison(
            image_path.name
        )

    print()
    print("Visual comparison completed.")


if __name__ == "__main__":
    main()