from pathlib import Path
import cv2
import numpy as np
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\research papper\rs")

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "enhanced_conditions_advanced"
)

CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection",
]

# ------------------------------------------------------------
# Enhancement parameters
# ------------------------------------------------------------

IMAGE_EXTENSIONS = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG",
]

# Maximum enhancement strength.
# Keeping these moderate is intentional because your previous
# experiment showed that aggressive enhancement reduced YOLO
# performance.
MAX_CLAHE = 2.2

# ------------------------------------------------------------
# Save optional processing statistics
# ------------------------------------------------------------

SAVE_STATS = True

STATS_FILE = OUTPUT_DIR / "enhancement_statistics.csv"


# ============================================================
# BASIC UTILITIES
# ============================================================

def clip_image(image):
    """
    Keep image values safely inside 0-255.
    """
    return np.clip(
        image,
        0,
        255
    ).astype(np.uint8)


def calculate_brightness(image):
    """
    Calculate average grayscale brightness.
    Range: 0-255
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(np.mean(gray))


def calculate_contrast(image):
    """
    Calculate grayscale standard deviation.
    Higher value = higher contrast.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(np.std(gray))


def calculate_dark_ratio(image):
    """
    Percentage of pixels that are very dark.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    ratio = np.mean(gray < 60)

    return float(ratio)


def calculate_bright_ratio(image):
    """
    Percentage of very bright pixels.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    ratio = np.mean(gray > 220)

    return float(ratio)


# ============================================================
# ADAPTIVE GAMMA
# ============================================================

def adaptive_gamma(image):
    """
    Automatically correct underexposure.

    Dark image  -> gamma < 1
    Normal image -> gamma approximately 1
    Bright image -> almost no correction
    """

    brightness = calculate_brightness(image)

    if brightness < 45:
        gamma = 0.65

    elif brightness < 60:
        gamma = 0.75

    elif brightness < 75:
        gamma = 0.85

    elif brightness < 95:
        gamma = 0.93

    else:
        gamma = 1.0

    if gamma == 1.0:
        return image

    table = np.array(
        [
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ],
        dtype=np.uint8
    )

    return cv2.LUT(
        image,
        table
    )


# ============================================================
# ADAPTIVE CLAHE
# ============================================================

def adaptive_clahe(image):
    """
    Apply CLAHE only according to the current image contrast.

    Low contrast -> stronger CLAHE
    Normal contrast -> mild CLAHE
    High contrast -> almost no CLAHE
    """

    contrast = calculate_contrast(image)

    if contrast < 25:
        clip_limit = 2.0

    elif contrast < 35:
        clip_limit = 1.7

    elif contrast < 45:
        clip_limit = 1.3

    else:
        clip_limit = 1.0

    clip_limit = min(
        clip_limit,
        MAX_CLAHE
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    enhanced = cv2.merge(
        [l, a, b]
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# ============================================================
# EDGE-PRESERVING DENOISING
# ============================================================

def edge_preserving_denoise(image, strength=1):
    """
    Bilateral filtering reduces noise while preserving
    pothole boundaries and road edges.
    """

    if strength <= 0:
        return image

    diameter = 5

    sigma_color = 30 + (strength * 10)
    sigma_space = 30 + (strength * 10)

    return cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space
    )


# ============================================================
# MILD SHARPENING
# ============================================================

def mild_sharpen(image, amount=0.15):
    """
    Very mild unsharp masking.

    Important:
    We intentionally keep sharpening weak because strong
    sharpening can amplify rain, fog, reflections and shadows.
    """

    if amount <= 0:
        return image

    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        image,
        1.0 + amount,
        blurred,
        -amount,
        0
    )

    return clip_image(
        sharpened
    )


# ============================================================
# ILLUMINATION CORRECTION
# ============================================================

def illumination_correction(image):
    """
    Estimate slow illumination changes and correct them.

    Useful for:
    - shadows
    - uneven road lighting
    - low-light scenes
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    background = cv2.GaussianBlur(
        l,
        (0, 0),
        25
    )

    background = np.maximum(
        background,
        1
    )

    corrected = (
        l.astype(np.float32)
        / background.astype(np.float32)
    )

    corrected *= 128.0

    corrected = np.clip(
        corrected,
        0,
        255
    ).astype(np.uint8)

    # Blend with original to avoid over-correction
    corrected = cv2.addWeighted(
        l,
        0.65,
        corrected,
        0.35,
        0
    )

    result = cv2.merge(
        [corrected, a, b]
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# SIMPLE DARK CHANNEL
# ============================================================

def dark_channel(image, kernel_size=15):
    """
    Approximate dark channel used for lightweight
    atmospheric haze estimation.
    """

    minimum = np.min(
        image,
        axis=2
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size)
    )

    dark = cv2.erode(
        minimum,
        kernel
    )

    return dark


# ============================================================
# LIGHTWEIGHT FOG / DEHAZING
# ============================================================

def lightweight_dehaze(image):
    """
    Lightweight atmospheric correction.

    This is intentionally conservative because aggressive
    dehazing can distort road texture and pothole boundaries.
    """

    image_float = (
        image.astype(np.float32)
        / 255.0
    )

    dark = dark_channel(
        image,
        kernel_size=15
    )

    # Estimate atmospheric strength
    dark_mean = float(
        np.mean(dark)
    )

    # If haze is weak, don't apply dehazing
    if dark_mean < 45:
        return image

    # Conservative transmission estimate
    transmission = 1.0 - (
        0.55
        * dark.astype(np.float32)
        / 255.0
    )

    transmission = np.clip(
        transmission,
        0.55,
        1.0
    )

    # Atmospheric light estimation
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightest_index = np.unravel_index(
        np.argmax(gray),
        gray.shape
    )

    atmospheric = image_float[
        brightest_index[0],
        brightest_index[1]
    ]

    atmospheric = np.maximum(
        atmospheric,
        0.65
    )

    result = np.zeros_like(
        image_float
    )

    for channel in range(3):

        result[:, :, channel] = (
            image_float[:, :, channel]
            - atmospheric[channel]
        ) / transmission

        result[:, :, channel] += (
            atmospheric[channel]
        )

    result = np.clip(
        result,
        0,
        1
    )

    result = (
        result * 255
    ).astype(np.uint8)

    # Conservative blending
    result = cv2.addWeighted(
        image,
        0.60,
        result,
        0.40,
        0
    )

    return result


# ============================================================
# RAIN PROCESSING
# ============================================================

def reduce_rain_artifacts(image):
    """
    Conservative rain processing.

    We do NOT attempt aggressive rain-streak removal because
    that can erase pothole boundaries.

    Instead:
    1. suppress high-frequency noise
    2. preserve major edges
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect high-frequency noise
    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        1.2
    )

    high_frequency = cv2.absdiff(
        gray,
        blur
    )

    noise_level = float(
        np.mean(high_frequency)
    )

    if noise_level < 8:
        return image

    # Bilateral filtering
    filtered = cv2.bilateralFilter(
        image,
        5,
        35,
        35
    )

    # Keep some original detail
    result = cv2.addWeighted(
        image,
        0.55,
        filtered,
        0.45,
        0
    )

    return result


# ============================================================
# SHADOW PROCESSING
# ============================================================

def reduce_shadow_effect(image):
    """
    Correct uneven illumination while preserving texture.
    """

    brightness = calculate_brightness(
        image
    )

    dark_ratio = calculate_dark_ratio(
        image
    )

    # Only perform stronger correction when significant
    # dark regions exist.
    if dark_ratio < 0.15:
        return image

    corrected = illumination_correction(
        image
    )

    # Conservative blend
    strength = 0.35

    if brightness < 70:
        strength = 0.45

    result = cv2.addWeighted(
        image,
        1.0 - strength,
        corrected,
        strength,
        0
    )

    return result


# ============================================================
# WATER / REFLECTION PROCESSING
# ============================================================

def enhance_water_reflection(image):
    """
    Preserve reflections while improving local visibility.

    We avoid aggressive sharpening because water reflections
    can otherwise become false edges.
    """

    contrast = calculate_contrast(
        image
    )

    if contrast < 30:

        enhanced = adaptive_clahe(
            image
        )

    else:

        enhanced = image.copy()

    # Mild denoising
    enhanced = cv2.bilateralFilter(
        enhanced,
        5,
        25,
        25
    )

    return enhanced


# ============================================================
# LOW-LIGHT PIPELINE
# ============================================================

def process_low_light(image):

    # Step 1: adaptive gamma
    result = adaptive_gamma(
        image
    )

    # Step 2: local contrast
    result = adaptive_clahe(
        result
    )

    # Step 3: edge-preserving denoise
    result = edge_preserving_denoise(
        result,
        strength=1
    )

    # Very mild sharpening
    result = mild_sharpen(
        result,
        amount=0.08
    )

    return result


# ============================================================
# FOG PIPELINE
# ============================================================

def process_fog(image):

    # Step 1: conservative dehazing
    result = lightweight_dehaze(
        image
    )

    # Step 2: only mild local contrast
    result = adaptive_clahe(
        result
    )

    # Step 3: very mild sharpening
    result = mild_sharpen(
        result,
        amount=0.06
    )

    return result


# ============================================================
# RAIN PIPELINE
# ============================================================

def process_rain(image):

    # Step 1: conservative rain artifact reduction
    result = reduce_rain_artifacts(
        image
    )

    # Step 2: adaptive contrast
    result = adaptive_clahe(
        result
    )

    # Do NOT aggressively sharpen rain images.
    return result


# ============================================================
# SHADOW PIPELINE
# ============================================================

def process_shadow(image):

    # Step 1: illumination correction
    result = reduce_shadow_effect(
        image
    )

    # Step 2: adaptive local contrast
    result = adaptive_clahe(
        result
    )

    # No aggressive sharpening
    return result


# ============================================================
# WATER PIPELINE
# ============================================================

def process_water_reflection(image):

    result = enhance_water_reflection(
        image
    )

    return result


# ============================================================
# NORMAL PIPELINE
# ============================================================

def process_normal(image):

    # IMPORTANT:
    # Normal condition remains untouched.
    return image.copy()


# ============================================================
# CONDITION DISPATCHER
# ============================================================

def enhance_image(image, condition):

    if condition == "normal":
        return process_normal(image)

    elif condition == "low_light":
        return process_low_light(image)

    elif condition == "fog":
        return process_fog(image)

    elif condition == "rain":
        return process_rain(image)

    elif condition == "shadow":
        return process_shadow(image)

    elif condition == "water_reflection":
        return process_water_reflection(image)

    else:
        raise ValueError(
            f"Unknown condition: {condition}"
        )


# ============================================================
# IMAGE STATISTICS
# ============================================================

def get_image_statistics(
    original,
    enhanced,
    condition,
    filename
):

    original_brightness = calculate_brightness(
        original
    )

    enhanced_brightness = calculate_brightness(
        enhanced
    )

    original_contrast = calculate_contrast(
        original
    )

    enhanced_contrast = calculate_contrast(
        enhanced
    )

    return {
        "condition": condition,
        "image": filename,

        "original_brightness":
            original_brightness,

        "enhanced_brightness":
            enhanced_brightness,

        "brightness_change":
            enhanced_brightness
            - original_brightness,

        "original_contrast":
            original_contrast,

        "enhanced_contrast":
            enhanced_contrast,

        "contrast_change":
            enhanced_contrast
            - original_contrast,
    }


# ============================================================
# FIND IMAGES
# ============================================================

def get_image_files(directory):

    files = []

    for extension in IMAGE_EXTENSIONS:

        files.extend(
            directory.glob(extension)
        )

    return sorted(
        set(files)
    )


# ============================================================
# PROCESS ONE CONDITION
# ============================================================

def process_condition(
    condition,
    statistics
):

    input_images = (
        INPUT_DIR
        / condition
        / "images"
    )

    input_labels = (
        INPUT_DIR
        / condition
        / "labels"
    )

    output_images = (
        OUTPUT_DIR
        / condition
        / "images"
    )

    output_labels = (
        OUTPUT_DIR
        / condition
        / "labels"
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not input_images.exists():

        print(
            f"\nWARNING: Missing image directory:"
        )

        print(input_images)

        return

    image_files = get_image_files(
        input_images
    )

    print("\n" + "=" * 70)

    print(
        f"{condition.upper()}: "
        f"{len(image_files)} images"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    processed = 0

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Could not read:"
                f"{image_path}"
            )

            continue

        # ----------------------------------------------------
        # Enhancement
        # ----------------------------------------------------

        enhanced = enhance_image(
            image,
            condition
        )

        enhanced = clip_image(
            enhanced
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        output_path = (
            output_images
            / image_path.name
        )

        success = cv2.imwrite(
            str(output_path),
            enhanced
        )

        if not success:

            print(
                f"WARNING: Failed to save:"
                f"{output_path}"
            )

            continue

        # ----------------------------------------------------
        # Copy YOLO annotation
        # ----------------------------------------------------

        label_path = (
            input_labels
            / f"{image_path.stem}.txt"
        )

        output_label_path = (
            output_labels
            / f"{image_path.stem}.txt"
        )

        if label_path.exists():

            shutil.copy2(
                label_path,
                output_label_path
            )

        else:

            print(
                f"WARNING: Missing label:"
                f"{label_path}"
            )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if SAVE_STATS:

            stats = get_image_statistics(
                image,
                enhanced,
                condition,
                image_path.name
            )

            statistics.append(
                stats
            )

        processed += 1

        if index % 25 == 0:

            print(
                f"  Processed "
                f"{index}/{len(image_files)}"
            )

    print(
        f"Completed {condition}: "
        f"{processed}/{len(image_files)}"
    )


# ============================================================
# SAVE STATISTICS
# ============================================================

def save_statistics(
    statistics
):

    if not SAVE_STATS:
        return

    if not statistics:
        return

    import csv

    with open(
        STATS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=statistics[0].keys()
        )

        writer.writeheader()

        writer.writerows(
            statistics
        )

    print(
        "\nEnhancement statistics saved:"
    )

    print(
        STATS_FILE
    )


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def print_summary(
    statistics
):

    if not statistics:
        return

    print("\n")
    print("=" * 80)
    print("ENHANCEMENT STATISTICS")
    print("=" * 80)

    conditions = sorted(
        set(
            row["condition"]
            for row in statistics
        )
    )

    for condition in conditions:

        rows = [
            row
            for row in statistics
            if row["condition"] == condition
        ]

        brightness_change = np.mean(
            [
                row["brightness_change"]
                for row in rows
            ]
        )

        contrast_change = np.mean(
            [
                row["contrast_change"]
                for row in rows
            ]
        )

        print(
            f"\n{condition}"
        )

        print(
            f"  Images: "
            f"{len(rows)}"
        )

        print(
            f"  Avg brightness change: "
            f"{brightness_change:+.2f}"
        )

        print(
            f"  Avg contrast change: "
            f"{contrast_change:+.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "ADVANCED ADAPTIVE ROAD-IMAGE ENHANCEMENT"
    )

    print("=" * 80)

    print(
        f"\nInput:"
    )

    print(
        INPUT_DIR
    )

    print(
        f"\nOutput:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nEnhancement strategy:"
    )

    print(
        "  Normal          -> Original"
    )

    print(
        "  Low light       -> Adaptive Gamma + CLAHE + Denoise"
    )

    print(
        "  Fog             -> Conservative Dehazing + CLAHE"
    )

    print(
        "  Rain            -> Rain Artifact Reduction + CLAHE"
    )

    print(
        "  Shadow          -> Illumination Correction + CLAHE"
    )

    print(
        "  Water/Reflection-> Reflection-aware Contrast"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    statistics = []

    # --------------------------------------------------------
    # Process all conditions
    # --------------------------------------------------------

    for condition in CONDITIONS:

        process_condition(
            condition,
            statistics
        )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    save_statistics(
        statistics
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print_summary(
        statistics
    )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)

    print(
        "ADVANCED ENHANCEMENT COMPLETE"
    )

    print("=" * 80)

    print(
        f"\nEnhanced dataset:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()