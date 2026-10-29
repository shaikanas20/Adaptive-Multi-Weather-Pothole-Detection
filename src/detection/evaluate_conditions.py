
from pathlib import Path
import csv
import math
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\research papper\rs")

CONTROLLED_DIR = (
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
    / "condition_evaluation"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "phase1"
    / "mwpd_yolov8m"
    / "weights"
    / "best.pt"
)

CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection",
]

# Expected dataset classes
EXPECTED_CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
    "Other / Repair Mark",
]

TARGET_CLASS_NAME = "Pothole"


# ============================================================
# CREATE TEMPORARY DATASET YAML
# ============================================================

def create_dataset_yaml(condition):

    condition_dir = CONTROLLED_DIR / condition

    yaml_path = OUTPUT_DIR / f"{condition}.yaml"

    yaml_content = f"""
path: "{condition_dir.as_posix()}"
train: images
val: images
test: images

names:
  0: Longitudinal Crack
  1: Transverse Crack
  2: Alligator Crack
  3: Pothole
  4: Other / Repair Mark
"""

    yaml_path.write_text(
        yaml_content.strip(),
        encoding="utf-8"
    )

    return yaml_path


# ============================================================
# F1 SCORE
# ============================================================

def calculate_f1(precision, recall):

    if not math.isfinite(precision) or not math.isfinite(recall):
        return float("nan")

    if precision + recall == 0:
        return 0.0

    return (
        2 * precision * recall
        / (precision + recall)
    )


# ============================================================
# CLASS NAME HELPERS
# ============================================================

def normalize_class_name(name):
    """
    Normalize class names so that small differences such as
    'pothole', 'Pothole', or 'POTHOLE' do not matter.
    """
    return str(name).strip().lower()


def get_model_class_mapping(model):

    names = model.names

    if isinstance(names, dict):
        return {
            int(k): str(v)
            for k, v in names.items()
        }

    return {
        i: str(name)
        for i, name in enumerate(names)
    }


def find_target_class_index(model):

    class_mapping = get_model_class_mapping(model)

    target = normalize_class_name(TARGET_CLASS_NAME)

    for class_id, class_name in class_mapping.items():

        if normalize_class_name(class_name) == target:
            return class_id

    return None


# ============================================================
# SAFE PER-CLASS METRIC EXTRACTION
# ============================================================

def get_array_value(array, index):

    if array is None:
        return None

    try:

        if hasattr(array, "tolist"):
            array = array.tolist()

        if index < 0 or index >= len(array):
            return None

        value = float(array[index])

        if not math.isfinite(value):
            return None

        return value

    except Exception:
        return None


def get_metric_for_class(box_metrics, target_class_index, metric_name):

    """
    Extract a per-class metric safely.

    For AP metrics, Ultralytics may store values according to
    ap_class_index rather than direct class ID.
    """

    # --------------------------------------------------------
    # AP / mAP metrics
    # --------------------------------------------------------

    if metric_name in ["ap50", "ap"]:

        values = getattr(box_metrics, metric_name, None)

        if values is None:
            return None

        try:

            if hasattr(values, "tolist"):
                values = values.tolist()

            class_indices = getattr(
                box_metrics,
                "ap_class_index",
                None
            )

            if hasattr(class_indices, "tolist"):
                class_indices = class_indices.tolist()

            # If AP values are mapped through ap_class_index
            if class_indices is not None:

                for position, class_id in enumerate(class_indices):

                    if int(class_id) == int(target_class_index):

                        if position < len(values):
                            value = float(values[position])

                            if math.isfinite(value):
                                return value

                            return None

                return None

            # Fallback: direct class indexing
            if target_class_index < len(values):

                value = float(values[target_class_index])

                if math.isfinite(value):
                    return value

        except Exception:
            return None

        return None

    # --------------------------------------------------------
    # Precision / Recall / F1
    # --------------------------------------------------------

    values = getattr(box_metrics, metric_name, None)

    return get_array_value(
        values,
        target_class_index
    )


# ============================================================
# EXTRACT METRICS
# ============================================================

def extract_metrics(metrics, target_class_index, condition):

    results = metrics.results_dict

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    precision = float(
        results.get("metrics/precision(B)", 0.0)
    )

    recall = float(
        results.get("metrics/recall(B)", 0.0)
    )

    map50 = float(
        results.get("metrics/mAP50(B)", 0.0)
    )

    map5095 = float(
        results.get("metrics/mAP50-95(B)", 0.0)
    )

    f1 = calculate_f1(
        precision,
        recall
    )

    # --------------------------------------------------------
    # Per-class Pothole metrics
    # --------------------------------------------------------

    box_metrics = metrics.box

    pothole_precision = get_metric_for_class(
        box_metrics,
        target_class_index,
        "p"
    )

    pothole_recall = get_metric_for_class(
        box_metrics,
        target_class_index,
        "r"
    )

    pothole_f1 = get_metric_for_class(
        box_metrics,
        target_class_index,
        "f1"
    )

    pothole_map50 = get_metric_for_class(
        box_metrics,
        target_class_index,
        "ap50"
    )

    pothole_map5095 = get_metric_for_class(
        box_metrics,
        target_class_index,
        "ap"
    )

    # --------------------------------------------------------
    # If F1 is unavailable, calculate it
    # --------------------------------------------------------

    if pothole_f1 is None:

        if (
            pothole_precision is not None
            and pothole_recall is not None
        ):

            pothole_f1 = calculate_f1(
                pothole_precision,
                pothole_recall
            )

    # --------------------------------------------------------
    # Warning if Pothole metrics are unavailable
    # --------------------------------------------------------

    pothole_values = [
        pothole_precision,
        pothole_recall,
        pothole_map50,
        pothole_map5095,
    ]

    if all(value is None for value in pothole_values):

        print(
            f"\nWARNING [{condition}]: "
            "Pothole-specific metrics could not be extracted."
        )

        print(
            "This is NOT automatically treated as 0.0."
        )

        print(
            "Check class mapping and Ultralytics metric structure."
        )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------

    speed = metrics.speed

    preprocess = float(
        speed.get("preprocess", 0.0)
    )

    inference = float(
        speed.get("inference", 0.0)
    )

    postprocess = float(
        speed.get("postprocess", 0.0)
    )

    total_latency = (
        preprocess
        + inference
        + postprocess
    )

    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    return {

        # Overall
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map5095,
        "F1": f1,

        # Pothole
        "pothole_precision": pothole_precision,
        "pothole_recall": pothole_recall,
        "pothole_mAP50": pothole_map50,
        "pothole_mAP50_95": pothole_map5095,
        "pothole_F1": pothole_f1,

        # Timing
        "preprocess_ms": preprocess,
        "inference_ms": inference,
        "postprocess_ms": postprocess,
        "total_latency_ms": total_latency,
    }


# ============================================================
# FORMAT VALUE
# ============================================================

def format_value(value, decimals=4):

    if value is None:
        return "N/A"

    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"

    return f"{value:.{decimals}f}"


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 80)
    print("CONTROLLED CONDITION ROBUSTNESS EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        print("\nERROR: Model not found.")
        print(f"Expected model:\n{MODEL_PATH}")

        print(
            "\nIf your training folder has a suffix such as "
            "'yolov8n_fixed-2', update MODEL_PATH."
        )

        return

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model:")
    print(MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Print model information
    # --------------------------------------------------------

    class_mapping = get_model_class_mapping(model)

    print("\nMODEL CLASS MAPPING")
    print("-" * 40)

    for class_id, class_name in class_mapping.items():
        print(
            f"{class_id}: {class_name}"
        )

    # --------------------------------------------------------
    # Find Pothole class dynamically
    # --------------------------------------------------------

    target_class_index = find_target_class_index(model)

    if target_class_index is None:

        print(
            "\nERROR: 'Pothole' class was not found in model.names."
        )

        print(
            "\nModel classes:"
        )

        for class_id, class_name in class_mapping.items():
            print(
                f"  {class_id}: {class_name}"
            )

        print(
            "\nEvaluation stopped because pothole-specific "
            "metrics cannot be identified safely."
        )

        return

    print(
        f"\nPothole class detected:"
        f" ID={target_class_index}, "
        f"Name={class_mapping[target_class_index]}"
    )

    # --------------------------------------------------------
    # Check expected classes
    # --------------------------------------------------------

    if len(class_mapping) != len(EXPECTED_CLASS_NAMES):

        print(
            "\nWARNING: Model has "
            f"{len(class_mapping)} classes, but the evaluation "
            f"dataset expects {len(EXPECTED_CLASS_NAMES)} classes."
        )

        print(
            "Continue only if this model was intentionally "
            "trained with a different class configuration."
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_results = []

    for condition in CONDITIONS:

        print("\n")
        print("=" * 80)
        print(f"EVALUATING: {condition.upper()}")
        print("=" * 80)

        condition_dir = (
            CONTROLLED_DIR
            / condition
        )

        images_dir = (
            condition_dir
            / "images"
        )

        labels_dir = (
            condition_dir
            / "labels"
        )

        # ----------------------------------------------------
        # Check directories
        # ----------------------------------------------------

        if not images_dir.exists():

            print(
                f"WARNING: Image directory missing:\n"
                f"{images_dir}"
            )

            continue

        if not labels_dir.exists():

            print(
                f"WARNING: Label directory missing:\n"
                f"{labels_dir}"
            )

            continue

        # ----------------------------------------------------
        # Create YAML
        # ----------------------------------------------------

        yaml_path = create_dataset_yaml(
            condition
        )

        print(
            f"Dataset YAML: {yaml_path}"
        )

        # ----------------------------------------------------
        # Validate model
        # ----------------------------------------------------

        metrics = model.val(
            data=str(yaml_path),
            split="test",
            imgsz=640,
            batch=16,
            device=0,
            workers=4,
            verbose=True,
            plots=True,
            project=str(OUTPUT_DIR),
            name=condition,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Extract metrics
        # ----------------------------------------------------

        extracted = extract_metrics(
            metrics,
            target_class_index,
            condition
        )

        # ----------------------------------------------------
        # Print condition result immediately
        # ----------------------------------------------------

        print("\nCONDITION RESULT")
        print("-" * 50)

        print(
            f"Overall Precision : "
            f"{format_value(extracted['precision'])}"
        )

        print(
            f"Overall Recall    : "
            f"{format_value(extracted['recall'])}"
        )

        print(
            f"Overall mAP50     : "
            f"{format_value(extracted['mAP50'])}"
        )

        print(
            f"Overall mAP50-95  : "
            f"{format_value(extracted['mAP50_95'])}"
        )

        print(
            f"Overall F1        : "
            f"{format_value(extracted['F1'])}"
        )

        print("\nPOTHOLE ONLY")

        print(
            f"Pothole Precision : "
            f"{format_value(extracted['pothole_precision'])}"
        )

        print(
            f"Pothole Recall    : "
            f"{format_value(extracted['pothole_recall'])}"
        )

        print(
            f"Pothole mAP50     : "
            f"{format_value(extracted['pothole_mAP50'])}"
        )

        print(
            f"Pothole mAP50-95  : "
            f"{format_value(extracted['pothole_mAP50_95'])}"
        )

        print(
            f"Pothole F1        : "
            f"{format_value(extracted['pothole_F1'])}"
        )

        print(
            f"\nLatency           : "
            f"{format_value(extracted['total_latency_ms'], 2)} ms"
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        row = {
            "condition": condition,
            "model": MODEL_PATH.name,
            **extracted,
        }

        all_results.append(row)

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not all_results:

        print("\nNo evaluations were completed.")
        return

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        OUTPUT_DIR
        / "condition_results.csv"
    )

    fieldnames = list(
        all_results[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(all_results)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 110)
    print("CONTROLLED CONDITION SUMMARY - POTHOLE")
    print("=" * 110)

    print(
        f"{'Condition':<20}"
        f"{'P':>10}"
        f"{'R':>10}"
        f"{'mAP50':>12}"
        f"{'mAP50-95':>14}"
        f"{'F1':>10}"
        f"{'Latency':>14}"
    )

    print("-" * 110)

    for row in all_results:

        print(
            f"{row['condition']:<20}"
            f"{format_value(row['pothole_precision'], 3):>10}"
            f"{format_value(row['pothole_recall'], 3):>10}"
            f"{format_value(row['pothole_mAP50'], 3):>12}"
            f"{format_value(row['pothole_mAP50_95'], 3):>14}"
            f"{format_value(row['pothole_F1'], 3):>10}"
            f"{format_value(row['total_latency_ms'], 2):>12} ms"
        )

    # ========================================================
    # FINAL SUMMARY - OVERALL
    # ========================================================

    print("\n")
    print("=" * 110)
    print("CONTROLLED CONDITION SUMMARY - ALL CLASSES")
    print("=" * 110)

    print(
        f"{'Condition':<20}"
        f"{'P':>10}"
        f"{'R':>10}"
        f"{'mAP50':>12}"
        f"{'mAP50-95':>14}"
        f"{'F1':>10}"
        f"{'Latency':>14}"
    )

    print("-" * 110)

    for row in all_results:

        print(
            f"{row['condition']:<20}"
            f"{row['precision']:>10.3f}"
            f"{row['recall']:>10.3f}"
            f"{row['mAP50']:>12.3f}"
            f"{row['mAP50_95']:>14.3f}"
            f"{row['F1']:>10.3f}"
            f"{row['total_latency_ms']:>12.2f} ms"
        )

    # ========================================================
    # SAVE LOCATION
    # ========================================================

    print("\nResults saved to:")
    print(csv_path)

    print("\nEvaluation complete.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
