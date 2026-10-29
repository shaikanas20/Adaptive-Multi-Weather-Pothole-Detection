
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(r"D:\research papper\rs")

BASELINE_MODEL = (
    ROOT / "experiments" / "phase1" / "mwpd"
    / "weights" / "best.pt"
)

AUGMENTED_MODEL = (
    ROOT / "experiments" / "phase1"
    / "experiment6_shadow_augmentation"
    / "training" / "yolov8n_shadow_aug_retry"
    / "weights" / "best.pt"
)

DATA_ROOT = (
    ROOT / "data"
    / "Multi-Weather Pothole Detection (MWPD)"
)

# Controlled-condition images are in:
# controlled_conditions/<condition>/images/
CONDITION_ROOT = DATA_ROOT / "controlled_conditions"

# Ground-truth labels from the original held-out test set.
DATASET_ROOT = (
    DATA_ROOT / "Multi-Weather Pothole Detection (MWPD)" / "MWPD"
)
LABEL_DIR = DATASET_ROOT / "test" / "labels"

OUTPUT_DIR = (
    ROOT / "experiments" / "phase1"
    / "experiment6_shadow_augmentation"
    / "evaluation"
)

CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection",
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

IMAGE_SIZE = 640
PREDICTION_CONF = 0.001
NMS_IOU = 0.70
MAX_DETECTIONS = 300

EVAL_CONF = 0.25
MATCH_IOU = 0.50

BOOTSTRAP_ROUNDS = 5000
RANDOM_SEED = 42


# ============================================================
# 2. LOAD GROUND-TRUTH LABELS
# ============================================================

def load_ground_truth(label_path, image_width, image_height):
    """
    Load YOLO labels:
    class_id x_center y_center width height

    Coordinates are normalized. Convert them to pixel XYXY.
    Only class 0 (Potholes) is evaluated.
    """

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing ground-truth label: {label_path}"
        )

    boxes = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"Invalid label format at {label_path}, "
                    f"line {line_number}: {line}"
                )

            class_id, xc, yc, w, h = map(float, values)

            if int(class_id) != 0:
                continue

            x1 = (xc - w / 2) * image_width
            y1 = (yc - h / 2) * image_height
            x2 = (xc + w / 2) * image_width
            y2 = (yc + h / 2) * image_height

            boxes.append([x1, y1, x2, y2])

    if not boxes:
        return np.empty((0, 4), dtype=np.float32)

    return np.asarray(boxes, dtype=np.float32)


# ============================================================
# 3. FIND CONDITION IMAGES
# ============================================================

def get_image_files(condition):
    """
    Actual directory structure:
    controlled_conditions/
        normal/
            images/
            labels/
            labels.cache
    """

    image_folder = CONDITION_ROOT / condition / "images"

    if not image_folder.exists():
        raise FileNotFoundError(
            f"Image folder does not exist: {image_folder}"
        )

    files = sorted(
        path for path in image_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not files:
        raise RuntimeError(
            f"No images found in: {image_folder}"
        )

    print(f"{condition}: found {len(files)} images")

    return files


# ============================================================
# 4. CALCULATE IOU
# ============================================================

def calculate_iou_matrix(pred_boxes, gt_boxes):
    """Calculate IoU between every prediction and ground truth."""

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros(
            (len(pred_boxes), len(gt_boxes)),
            dtype=np.float32,
        )

    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)

    x1 = np.maximum(
        pred_boxes[:, None, 0],
        gt_boxes[None, :, 0],
    )
    y1 = np.maximum(
        pred_boxes[:, None, 1],
        gt_boxes[None, :, 1],
    )
    x2 = np.minimum(
        pred_boxes[:, None, 2],
        gt_boxes[None, :, 2],
    )
    y2 = np.minimum(
        pred_boxes[:, None, 3],
        gt_boxes[None, :, 3],
    )

    intersection = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    pred_area = (
        np.maximum(0, pred_boxes[:, 2] - pred_boxes[:, 0])
        * np.maximum(0, pred_boxes[:, 3] - pred_boxes[:, 1])
    )

    gt_area = (
        np.maximum(0, gt_boxes[:, 2] - gt_boxes[:, 0])
        * np.maximum(0, gt_boxes[:, 3] - gt_boxes[:, 1])
    )

    union = pred_area[:, None] + gt_area[None, :] - intersection

    return intersection / np.maximum(union, 1e-9)


# ============================================================
# 5. MATCH PREDICTIONS WITH GROUND TRUTH
# ============================================================

def count_detection_errors(pred_boxes, pred_scores, gt_boxes):
    """
    Filter predictions at EVAL_CONF.

    Sort predictions by descending confidence.
    Match each prediction to the best unmatched ground-truth box
    if IoU >= MATCH_IOU.

    Each ground-truth box can be matched only once.
    """

    keep = pred_scores >= EVAL_CONF

    pred_boxes = pred_boxes[keep]
    pred_scores = pred_scores[keep]

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    iou_matrix = calculate_iou_matrix(pred_boxes, gt_boxes)

    matched_gt = set()
    tp = 0
    fp = 0

    for pred_idx in range(len(pred_boxes)):

        if len(gt_boxes) == 0:
            fp += 1
            continue

        candidate_ious = iou_matrix[pred_idx].copy()

        for gt_idx in matched_gt:
            candidate_ious[gt_idx] = -1

        best_gt = int(np.argmax(candidate_ious))
        best_iou = candidate_ious[best_gt]

        if best_iou >= MATCH_IOU:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return tp, fp, fn, precision, recall, f1


# ============================================================
# 6. EXTRACT YOLO PREDICTIONS
# ============================================================

def get_prediction_arrays(result):
    """Extract predicted boxes and confidence scores for class 0."""

    if result.boxes is None or len(result.boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    keep = classes == 0

    return boxes[keep], scores[keep]


# ============================================================
# 7. EVALUATE ONE MODEL ON ALL SIX CONDITIONS
# ============================================================

def evaluate_model(model, model_name, images_by_condition, gt_cache):
    rows = []

    all_image_paths = [
        path
        for condition in CONDITIONS
        for path in images_by_condition[condition]
    ]

    print(f"\nEvaluating: {model_name}")

    # Warm-up inference. This image is also evaluated below.
    if all_image_paths:
        model.predict(
            source=str(all_image_paths[0]),
            imgsz=IMAGE_SIZE,
            conf=PREDICTION_CONF,
            iou=NMS_IOU,
            max_det=MAX_DETECTIONS,
            verbose=False,
        )

    for condition in CONDITIONS:

        image_paths = images_by_condition[condition]

        print(
            f"\n{model_name} | {condition} | "
            f"{len(image_paths)} images"
        )

        start_time = time.perf_counter()

        # Use identical inference parameters for both models.
        results = model.predict(
            source=[str(path) for path in image_paths],
            imgsz=IMAGE_SIZE,
            conf=PREDICTION_CONF,
            iou=NMS_IOU,
            max_det=MAX_DETECTIONS,
            batch=1,
            stream=False,
            verbose=False,
        )

        elapsed_seconds = time.perf_counter() - start_time

        if len(results) != len(image_paths):
            raise RuntimeError(
                f"Expected {len(image_paths)} results for "
                f"{condition}, received {len(results)}."
            )

        wall_ms_per_image = (
            elapsed_seconds * 1000 / len(image_paths)
        )

        for image_path, result in zip(image_paths, results):

            stem = image_path.stem
            cache_key = (condition, stem)

            if cache_key not in gt_cache:
                raise KeyError(
                    f"Ground truth missing for "
                    f"{condition}/{stem}.txt"
                )

            gt_boxes = gt_cache[cache_key]

            pred_boxes, pred_scores = get_prediction_arrays(result)

            tp, fp, fn, precision, recall, f1 = (
                count_detection_errors(
                    pred_boxes,
                    pred_scores,
                    gt_boxes,
                )
            )

            rows.append({
                "condition": condition,
                "model": model_name,
                "image_stem": stem,
                "gt_count": len(gt_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "wall_ms_per_image": wall_ms_per_image,
            })

    return rows


# ============================================================
# 8. CONDITION-LEVEL SUMMARY
# ============================================================

def build_condition_summary(per_image_df):
    summary_rows = []

    for (condition, model_name), group in per_image_df.groupby(
        ["condition", "model"], sort=False
    ):

        total_tp = int(group["tp"].sum())
        total_fp = int(group["fp"].sum())
        total_fn = int(group["fn"].sum())

        # Pooled metrics computed from total detections.
        pooled_precision = (
            total_tp / (total_tp + total_fp)
            if total_tp + total_fp else 0.0
        )

        pooled_recall = (
            total_tp / (total_tp + total_fn)
            if total_tp + total_fn else 0.0
        )

        pooled_f1 = (
            2 * pooled_precision * pooled_recall
            / (pooled_precision + pooled_recall)
            if pooled_precision + pooled_recall else 0.0
        )

        summary_rows.append({
            "condition": condition,
            "model": model_name,
            "num_images": len(group),
            "total_gt": int(group["gt_count"].sum()),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "pooled_precision": pooled_precision,
            "pooled_recall": pooled_recall,
            "pooled_f1": pooled_f1,
            "mean_image_precision": group["precision"].mean(),
            "mean_image_recall": group["recall"].mean(),
            "mean_image_f1": group["f1"].mean(),
            "mean_wall_ms_per_image": (
                group["wall_ms_per_image"].mean()
            ),
        })

    return pd.DataFrame(summary_rows)


# ============================================================
# 9. PAIRED BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_mean_difference(
    augmented_values,
    baseline_values,
    rng,
):
    """
    Paired bootstrap over image stems.

    Difference = augmented model metric - baseline metric.
    """

    augmented_values = np.asarray(
        augmented_values, dtype=float
    )
    baseline_values = np.asarray(
        baseline_values, dtype=float
    )

    if len(augmented_values) != len(baseline_values):
        raise ValueError(
            "Paired metric arrays must have equal lengths."
        )

    differences = augmented_values - baseline_values
    n = len(differences)

    if n == 0:
        return np.nan, np.nan, np.nan

    observed_mean = float(np.mean(differences))

    bootstrap_means = np.empty(
        BOOTSTRAP_ROUNDS, dtype=float
    )

    for i in range(BOOTSTRAP_ROUNDS):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = np.mean(differences[indices])

    ci_low, ci_high = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    return observed_mean, float(ci_low), float(ci_high)


def build_paired_comparison(per_image_df):
    rng = np.random.default_rng(RANDOM_SEED)

    baseline_name = "yolov8n_baseline"
    augmented_name = "yolov8n_shadow_augmented"

    comparison_rows = []

    for condition in CONDITIONS:

        subset = per_image_df[
            per_image_df["condition"] == condition
        ]

        baseline = (
            subset[subset["model"] == baseline_name]
            .set_index("image_stem")
            .sort_index()
        )

        augmented = (
            subset[subset["model"] == augmented_name]
            .set_index("image_stem")
            .sort_index()
        )

        common_stems = sorted(
            set(baseline.index) & set(augmented.index)
        )

        if not common_stems:
            raise RuntimeError(
                f"No paired images for condition: {condition}"
            )

        # Explicit alignment ensures each image is compared
        # against the same source image from the other model.
        baseline = baseline.loc[common_stems]
        augmented = augmented.loc[common_stems]

        row = {
            "condition": condition,
            "paired_images": len(common_stems),
        }

        for metric in ["precision", "recall", "f1"]:

            base_values = baseline[metric].to_numpy()
            aug_values = augmented[metric].to_numpy()

            mean_diff, ci_low, ci_high = (
                bootstrap_mean_difference(
                    aug_values,
                    base_values,
                    rng,
                )
            )

            row[f"baseline_mean_image_{metric}"] = float(
                np.mean(base_values)
            )

            row[f"augmented_mean_image_{metric}"] = float(
                np.mean(aug_values)
            )

            row[f"delta_{metric}"] = mean_diff
            row[f"delta_{metric}_ci95_low"] = ci_low
            row[f"delta_{metric}_ci95_high"] = ci_high

        base_f1 = baseline["f1"].to_numpy()
        aug_f1 = augmented["f1"].to_numpy()

        row["images_f1_improved"] = int(
            np.sum(aug_f1 > base_f1)
        )

        row["images_f1_unchanged"] = int(
            np.sum(aug_f1 == base_f1)
        )

        row["images_f1_decreased"] = int(
            np.sum(aug_f1 < base_f1)
        )

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify required paths before starting inference.
    required_paths = [
        BASELINE_MODEL,
        AUGMENTED_MODEL,
        LABEL_DIR,
        CONDITION_ROOT,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required path does not exist: {path}"
            )

    print("Loading baseline model...")
    baseline_model = YOLO(str(BASELINE_MODEL))

    print("Loading shadow-augmented model...")
    augmented_model = YOLO(str(AUGMENTED_MODEL))

    print("\nModels loaded successfully.")

    # IMPORTANT:
    # Images are under <condition>/images/, not directly under
    # <condition>/.
    images_by_condition = {
        condition: get_image_files(condition)
        for condition in CONDITIONS
    }

    total_images = sum(
        len(paths) for paths in images_by_condition.values()
    )

    print(f"\nTotal condition-image instances: {total_images}")

    # Verify paired image stems and load labels.
    reference_stems = {
        path.stem
        for path in images_by_condition[CONDITIONS[0]]
    }

    for condition in CONDITIONS[1:]:
        stems = {
            path.stem
            for path in images_by_condition[condition]
        }

        if stems != reference_stems:
            missing = sorted(reference_stems - stems)
            extra = sorted(stems - reference_stems)

            raise RuntimeError(
                f"Image stems differ in condition '{condition}'. "
                f"Missing examples: {missing[:5]}; "
                f"extra examples: {extra[:5]}"
            )

    gt_cache = {}

    print("\nLoading original test labels...")

    for condition, image_paths in images_by_condition.items():

        for image_path in image_paths:

            label_path = LABEL_DIR / f"{image_path.stem}.txt"

            image = cv2.imread(str(image_path))

            if image is None:
                raise RuntimeError(
                    f"Could not read image: {image_path}"
                )

            height, width = image.shape[:2]

            gt_cache[(condition, image_path.stem)] = (
                load_ground_truth(
                    label_path,
                    width,
                    height,
                )
            )

    print(f"Loaded {len(gt_cache)} image-condition label sets.")

    # Evaluate both models with identical settings.
    baseline_rows = evaluate_model(
        baseline_model,
        "yolov8n_baseline",
        images_by_condition,
        gt_cache,
    )

    augmented_rows = evaluate_model(
        augmented_model,
        "yolov8n_shadow_augmented",
        images_by_condition,
        gt_cache,
    )

    per_image_df = pd.DataFrame(
        baseline_rows + augmented_rows
    )

    per_image_path = OUTPUT_DIR / "per_image_metrics.csv"
    per_image_df.to_csv(per_image_path, index=False)

    summary_df = build_condition_summary(per_image_df)

    summary_path = OUTPUT_DIR / "condition_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    paired_df = build_paired_comparison(per_image_df)

    paired_path = OUTPUT_DIR / "paired_comparison.csv"
    paired_df.to_csv(paired_path, index=False)

    # Print condition-level metrics.
    print("\n" + "=" * 90)
    print("CONDITION SUMMARY")
    print("=" * 90)

    print(
        summary_df[
            [
                "condition",
                "model",
                "total_fp",
                "total_fn",
                "pooled_precision",
                "pooled_recall",
                "pooled_f1",
                "mean_image_recall",
                "mean_image_f1",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # Print paired differences and confidence intervals.
    print("\n" + "=" * 90)
    print("PAIRED COMPARISON: AUGMENTED MINUS BASELINE")
    print("=" * 90)

    print(
        paired_df[
            [
                "condition",
                "paired_images",
                "delta_recall",
                "delta_recall_ci95_low",
                "delta_recall_ci95_high",
                "delta_f1",
                "delta_f1_ci95_low",
                "delta_f1_ci95_high",
                "images_f1_improved",
                "images_f1_unchanged",
                "images_f1_decreased",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nEvaluation completed successfully.")
    print(f"Per-image results: {per_image_path}")
    print(f"Condition summary: {summary_path}")
    print(f"Paired comparison: {paired_path}")


if __name__ == "__main__":
    main()
