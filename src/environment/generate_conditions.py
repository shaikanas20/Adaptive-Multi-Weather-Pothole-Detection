from pathlib import Path
import cv2
import numpy as np
import shutil
import random
import csv


# ============================================================
# ADVANCED ADVERSE CONDITION GENERATOR
# ============================================================
#
# Purpose:
# Generate realistic but conservative adverse-condition
# images for pothole robustness evaluation.
#
# Conditions:
#   1. normal
#   2. low_light
#   3. fog
#   4. rain
#   5. shadow
#   6. water_reflection
#
# Important:
# Original bounding boxes are preserved.
#
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "MWPD"
    / "test"
)

SOURCE_IMAGE_DIR = SOURCE_DIR / "images"
SOURCE_LABEL_DIR = SOURCE_DIR / "labels"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions_advanced"
)

SOURCE_LIST = (
    PROJECT_ROOT
    / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions"
    / "source_images.txt"
)

MAX_IMAGES = 97

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def clip_uint8(image):
    """
    Safely clip image to uint8.
    """

    return np.clip(
        image,
        0,
        255
    ).astype(np.uint8)


def calculate_brightness(image):
    """
    Calculate mean grayscale brightness.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(np.mean(gray))


def calculate_contrast(image):
    """
    Calculate grayscale standard deviation.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(np.std(gray))


def adaptive_gamma(image, target_brightness=105):
    """
    Adaptive gamma correction.

    Avoids very strong gamma transformations.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    current = float(np.mean(gray))

    if current < 1:
        return image.copy()

    ratio = target_brightness / current

    # Conservative gamma range
    gamma = np.clip(
        ratio ** 0.35,
        0.80,
        1.20
    )

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


def apply_clahe_conservative(image):
    """
    Mild CLAHE.

    Strong CLAHE can amplify road texture and noise,
    so the clip limit is intentionally conservative.
    """

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=1.3,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    result = cv2.merge(
        [l, a, b]
    )

    return cv2.cvtColor(
        result,
        cv2.COLOR_LAB2BGR
    )


def preserve_colors(original, enhanced, strength=0.35):
    """
    Blend enhanced image with original.

    This prevents over-processing.
    """

    return cv2.addWeighted(
        original,
        1.0 - strength,
        enhanced,
        strength,
        0
    )


def smooth_mask(mask, kernel=31):
    """
    Smooth an illumination mask.
    """

    if kernel % 2 == 0:
        kernel += 1

    return cv2.GaussianBlur(
        mask,
        (kernel, kernel),
        0
    )


# ============================================================
# NORMAL
# ============================================================

def condition_normal(image):
    """
    No transformation.
    """

    return image.copy()


# ============================================================
# LOW LIGHT
# ============================================================

def condition_low_light(image):
    """
    Realistic low-light transformation.

    Strategy:
        1. Controlled brightness reduction
        2. Mild adaptive gamma
        3. Conservative local contrast
        4. Preserve color information
    """

    result = image.copy()

    # --------------------------------------------------------
    # 1. Moderate brightness reduction
    # --------------------------------------------------------

    factor = random.uniform(
        0.62,
        0.78
    )

    result = (
        result.astype(np.float32)
        * factor
    )

    result = clip_uint8(
        result
    )

    # --------------------------------------------------------
    # 2. Mild gamma correction
    # --------------------------------------------------------

    gamma = random.uniform(
        0.90,
        1.05
    )

    table = np.array(
        [
            ((i / 255.0) ** gamma) * 255
            for i in range(256)
        ],
        dtype=np.uint8
    )

    result = cv2.LUT(
        result,
        table
    )

    # --------------------------------------------------------
    # 3. Conservative CLAHE
    # --------------------------------------------------------

    enhanced = apply_clahe_conservative(
        result
    )

    result = preserve_colors(
        result,
        enhanced,
        strength=0.35
    )

    return result


# ============================================================
# FOG
# ============================================================

def condition_fog(image):
    """
    Conservative atmospheric fog.

    Important:
    No Gaussian blur is used.

    This preserves pothole boundaries.
    """

    result = image.copy()

    h, w = result.shape[:2]

    # --------------------------------------------------------
    # Atmospheric gradient
    # --------------------------------------------------------

    y = np.linspace(
        0,
        1,
        h
    ).reshape(h, 1)

    # Stronger toward upper/background region
    fog_strength = (
        0.05
        +
        0.18 * (1.0 - y)
    )

    # Add small image-dependent variation
    fog_strength += np.random.uniform(
        -0.015,
        0.015,
        size=(h, 1)
    )

    fog_strength = np.clip(
        fog_strength,
        0.03,
        0.24
    )

    # --------------------------------------------------------
    # Atmospheric light
    # --------------------------------------------------------

    atmospheric = np.full_like(
        result,
        random.randint(
            205,
            225
        ),
        dtype=np.float32
    )

    image_float = result.astype(
        np.float32
    )

    fogged = (
        image_float
        * (1.0 - fog_strength[:, :, None])
        +
        atmospheric
        * fog_strength[:, :, None]
    )

    fogged = clip_uint8(
        fogged
    )

    # --------------------------------------------------------
    # Very mild local contrast
    # --------------------------------------------------------

    enhanced = apply_clahe_conservative(
        fogged
    )

    result = preserve_colors(
        fogged,
        enhanced,
        strength=0.20
    )

    return result


# ============================================================
# RAIN
# ============================================================

def condition_rain(image):
    """
    Conservative rain simulation.

    No heavy blur.
    Very low rain-streak opacity.
    """

    result = image.copy()

    h, w = result.shape[:2]

    # --------------------------------------------------------
    # 1. Mild darkening
    # --------------------------------------------------------

    brightness_factor = random.uniform(
        0.88,
        0.96
    )

    result = (
        result.astype(np.float32)
        * brightness_factor
    )

    result = clip_uint8(
        result
    )

    # --------------------------------------------------------
    # 2. Mild atmospheric haze
    # --------------------------------------------------------

    haze = np.full_like(
        result,
        random.randint(
            185,
            215
        ),
        dtype=np.uint8
    )

    haze_alpha = random.uniform(
        0.025,
        0.07
    )

    result = cv2.addWeighted(
        result,
        1.0 - haze_alpha,
        haze,
        haze_alpha,
        0
    )

    # --------------------------------------------------------
    # 3. Very subtle rain streaks
    # --------------------------------------------------------

    overlay = result.copy()

    # Much fewer streaks than original implementation
    number_of_streaks = max(
        3,
        int((h * w) / 18000)
    )

    for _ in range(
        number_of_streaks
    ):

        x = random.randint(
            0,
            w - 1
        )

        y = random.randint(
            0,
            max(0, h - 25)
        )

        length = random.randint(
            8,
            18
        )

        dx = random.randint(
            -2,
            2
        )

        x2 = np.clip(
            x + dx,
            0,
            w - 1
        )

        y2 = np.clip(
            y + length,
            0,
            h - 1
        )

        intensity = random.randint(
            150,
            205
        )

        cv2.line(
            overlay,
            (x, y),
            (x2, y2),
            (
                intensity,
                intensity,
                intensity
            ),
            1
        )

    # Very weak blending
    result = cv2.addWeighted(
        result,
        0.96,
        overlay,
        0.04,
        0
    )

    return result


# ============================================================
# SHADOW
# ============================================================

def condition_shadow(image):
    """
    Soft natural-looking shadow.

    Instead of drawing a hard black polygon,
    use a smooth illumination field.
    """

    result = image.copy()

    h, w = result.shape[:2]

    # --------------------------------------------------------
    # Create smooth elliptical shadow
    # --------------------------------------------------------

    mask = np.zeros(
        (h, w),
        dtype=np.float32
    )

    center_x = random.randint(
        int(w * 0.2),
        int(w * 0.8)
    )

    center_y = random.randint(
        int(h * 0.3),
        int(h * 0.75)
    )

    axis_x = random.randint(
        int(w * 0.25),
        int(w * 0.55)
    )

    axis_y = random.randint(
        int(h * 0.15),
        int(h * 0.40)
    )

    cv2.ellipse(
        mask,
        (
            center_x,
            center_y
        ),
        (
            axis_x,
            axis_y
        ),
        random.randint(
            -25,
            25
        ),
        0,
        360,
        1.0,
        -1
    )

    mask = smooth_mask(
        mask,
        51
    )

    # --------------------------------------------------------
    # Moderate shadow strength
    # --------------------------------------------------------

    shadow_strength = random.uniform(
        0.10,
        0.22
    )

    result_float = result.astype(
        np.float32
    )

    result_float *= (
        1.0
        -
        mask[:, :, None]
        * shadow_strength
    )

    result = clip_uint8(
        result_float
    )

    # --------------------------------------------------------
    # Preserve local detail
    # --------------------------------------------------------

    enhanced = apply_clahe_conservative(
        result
    )

    result = preserve_colors(
        result,
        enhanced,
        strength=0.18
    )

    return result


# ============================================================
# WATER / WET ROAD
# ============================================================

def condition_water_reflection(image):
    """
    Conservative wet-road transformation.

    IMPORTANT:
    No full-image vertical reflection is used.

    This avoids introducing unrealistic patterns.
    """

    result = image.copy()

    h, w = result.shape[:2]

    # --------------------------------------------------------
    # Road region
    # --------------------------------------------------------

    road_start = int(
        h * 0.55
    )

    road_gradient = np.zeros(
        (h, w),
        dtype=np.float32
    )

    for y in range(
        road_start,
        h
    ):

        strength = (
            (y - road_start)
            /
            max(
                h - road_start,
                1
            )
        )

        road_gradient[y, :] = strength

    # --------------------------------------------------------
    # Wet-road darkening
    # --------------------------------------------------------

    darkening = random.uniform(
        0.04,
        0.12
    )

    result_float = result.astype(
        np.float32
    )

    result_float *= (
        1.0
        -
        road_gradient[:, :, None]
        * darkening
    )

    # --------------------------------------------------------
    # Soft wet highlights
    # --------------------------------------------------------

    highlight_mask = np.zeros(
        (h, w),
        dtype=np.float32
    )

    number_of_highlights = random.randint(
        5,
        12
    )

    for _ in range(
        number_of_highlights
    ):

        x = random.randint(
            0,
            w - 1
        )

        y = random.randint(
            int(h * 0.55),
            h - 1
        )

        axis_x = random.randint(
            15,
            max(
                20,
                int(w * 0.08)
            )
        )

        axis_y = random.randint(
            2,
            5
        )

        cv2.ellipse(
            highlight_mask,
            (
                x,
                y
            ),
            (
                axis_x,
                axis_y
            ),
            random.randint(
                -10,
                10
            ),
            0,
            360,
            random.uniform(
                0.15,
                0.40
            ),
            -1
        )

    highlight_mask = smooth_mask(
        highlight_mask,
        21
    )

    result_float += (
        highlight_mask[:, :, None]
        * 25.0
    )

    result = clip_uint8(
        result_float
    )

    # --------------------------------------------------------
    # Mild local contrast
    # --------------------------------------------------------

    enhanced = apply_clahe_conservative(
        result
    )

    result = preserve_colors(
        result,
        enhanced,
        strength=0.15
    )

    return result


# ============================================================
# IMAGE STATISTICS
# ============================================================

def image_statistics(
    original,
    transformed
):

    return {
        "original_brightness":
            calculate_brightness(
                original
            ),

        "transformed_brightness":
            calculate_brightness(
                transformed
            ),

        "original_contrast":
            calculate_contrast(
                original
            ),

        "transformed_contrast":
            calculate_contrast(
                transformed
            )
    }


# ============================================================
# LOAD FIXED SOURCE LIST
# ============================================================

def load_source_images():

    if not SOURCE_LIST.exists():

        raise FileNotFoundError(
            f"\nSource image list not found:\n"
            f"{SOURCE_LIST}\n"
        )

    with open(
        SOURCE_LIST,
        "r"
    ) as f:

        names = [
            line.strip()
            for line in f
            if line.strip()
        ]

    if len(names) != MAX_IMAGES:

        raise ValueError(
            f"\nExpected {MAX_IMAGES} "
            f"images but found "
            f"{len(names)}"
        )

    images = []

    for name in names:

        image_path = (
            SOURCE_IMAGE_DIR
            / name
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"\nImage not found:\n"
                f"{image_path}"
            )

        images.append(
            image_path
        )

    return images


# ============================================================
# PREPARE OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory(
    condition
):

    condition_dir = (
        OUTPUT_DIR
        / condition
    )

    if condition_dir.exists():

        shutil.rmtree(
            condition_dir
        )

    image_dir = (
        condition_dir
        / "images"
    )

    label_dir = (
        condition_dir
        / "labels"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    label_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return image_dir, label_dir


# ============================================================
# GENERATE CONDITION
# ============================================================

def generate_condition(
    condition_name,
    transform,
    images,
    statistics
):

    print()
    print("=" * 70)

    print(
        f"GENERATING: {condition_name}"
    )

    print("=" * 70)

    image_dir, label_dir = (
        prepare_output_directory(
            condition_name
        )
    )

    for index, image_path in enumerate(
        images,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Could not read:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        transformed = transform(
            image
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        stats = image_statistics(
            image,
            transformed
        )

        statistics.append(
            {
                "condition":
                    condition_name,

                "image":
                    image_path.name,

                **stats
            }
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        output_image = (
            image_dir
            / image_path.name
        )

        success = cv2.imwrite(
            str(output_image),
            transformed,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if not success:

            raise RuntimeError(
                f"Could not save:\n"
                f"{output_image}"
            )

        # ----------------------------------------------------
        # Copy original label
        # ----------------------------------------------------

        source_label = (
            SOURCE_LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        output_label = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        if source_label.exists():

            shutil.copy2(
                source_label,
                output_label
            )

        else:

            output_label.write_text(
                ""
            )

        if index % 25 == 0:

            print(
                f"Processed "
                f"{index}/{len(images)}"
            )

    print(
        f"Completed {condition_name}"
    )


# ============================================================
# SAVE STATISTICS
# ============================================================

def save_statistics(
    statistics
):

    output_file = (
        OUTPUT_DIR
        / "enhancement_statistics.csv"
    )

    if not statistics:
        return

    fields = [
        "condition",
        "image",
        "original_brightness",
        "transformed_brightness",
        "original_contrast",
        "transformed_contrast"
    ]

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(
            statistics
        )

    print()
    print(
        f"Statistics saved to:\n"
        f"{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "ADVANCED ROBUST ADVERSE-CONDITION "
        "DATASET GENERATION"
    )
    print("=" * 70)

    print()
    print(
        "Source:"
    )

    print(
        SOURCE_DIR
    )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Verify source
    # --------------------------------------------------------

    if not SOURCE_IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Source image directory missing:\n"
            f"{SOURCE_IMAGE_DIR}"
        )

    if not SOURCE_LABEL_DIR.exists():

        raise FileNotFoundError(
            f"Source label directory missing:\n"
            f"{SOURCE_LABEL_DIR}"
        )

    # --------------------------------------------------------
    # Load fixed images
    # --------------------------------------------------------

    images = load_source_images()

    print()
    print(
        f"Fixed source images: "
        f"{len(images)}"
    )

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    conditions = {

        "normal":
            condition_normal,

        "low_light":
            condition_low_light,

        "fog":
            condition_fog,

        "rain":
            condition_rain,

        "shadow":
            condition_shadow,

        "water_reflection":
            condition_water_reflection
    }

    statistics = []

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    for condition_name, transform in (
        conditions.items()
    ):

        generate_condition(
            condition_name,
            transform,
            images,
            statistics
        )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    save_statistics(
        statistics
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "GENERATION COMPLETE"
    )
    print("=" * 70)

    print()

    for condition in conditions:

        print(
            f"  {condition}: "
            f"{MAX_IMAGES} images"
        )

    print()
    print(
        "Original dataset was NOT modified."
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()