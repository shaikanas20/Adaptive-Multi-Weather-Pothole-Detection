from pathlib import Path
import cv2
import numpy as np
import csv


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "MWPD"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "phase1"
    / "environment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Configuration
# ============================================================

# Number of images to analyze from each split
MAX_IMAGES = 1000

SPLITS = [
    "train",
    "val",
    "test"
]


# ============================================================
# Image analysis functions
# ============================================================

def calculate_brightness(image):
    """
    Calculate average grayscale brightness.
    Range: approximately 0-255.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(np.mean(gray))


def calculate_contrast(image):
    """
    Calculate grayscale standard deviation.
    Higher value generally means greater intensity variation.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(np.std(gray))


def calculate_blur(image):
    """
    Estimate image sharpness using Laplacian variance.

    Lower value -> potentially blurry
    Higher value -> sharper image
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_saturation(image):
    """
    Calculate average HSV saturation.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return float(np.mean(hsv[:, :, 1]))


def calculate_dark_pixel_ratio(image):
    """
    Percentage of pixels that are relatively dark.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dark_pixels = np.sum(gray < 50)

    total_pixels = gray.size

    return float(dark_pixels / total_pixels)


def analyze_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return None

    height, width = image.shape[:2]

    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)
    blur = calculate_blur(image)
    saturation = calculate_saturation(image)
    dark_ratio = calculate_dark_pixel_ratio(image)

    return {
        "image": image_path.name,
        "width": width,
        "height": height,
        "brightness": brightness,
        "contrast": contrast,
        "blur": blur,
        "saturation": saturation,
        "dark_pixel_ratio": dark_ratio
    }


# ============================================================
# Analyze split
# ============================================================

def analyze_split(split):

    image_dir = DATASET_ROOT /"images"

    if not image_dir.exists():

        print(f"Directory not found: {image_dir}")

        return []

    image_files = sorted(image_dir.glob("*.jpg"))

    # Limit number of images for initial analysis
    image_files = image_files[:MAX_IMAGES]

    print()
    print("=" * 60)
    print(f"Analyzing {split} dataset")
    print(f"Images: {len(image_files)}")
    print("=" * 60)

    results = []

    for index, image_path in enumerate(image_files, start=1):

        result = analyze_image(image_path)

        if result is not None:
            result["split"] = split
            results.append(result)

        if index % 100 == 0:
            print(f"Processed {index}/{len(image_files)}")

    return results


# ============================================================
# Save CSV
# ============================================================

def save_results(results):

    output_file = OUTPUT_DIR / "environment_features.csv"

    if not results:
        print("No results to save.")

        return

    fieldnames = [
        "split",
        "image",
        "width",
        "height",
        "brightness",
        "contrast",
        "blur",
        "saturation",
        "dark_pixel_ratio"
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)

    print()
    print("=" * 60)
    print("Results saved")
    print("=" * 60)

    print(output_file)


# ============================================================
# Main
# ============================================================

def main():

    all_results = []

    for split in SPLITS:

        results = analyze_split(split)

        all_results.extend(results)

    save_results(all_results)

    print()
    print(f"Total images analyzed: {len(all_results)}")


if __name__ == "__main__":
    main()