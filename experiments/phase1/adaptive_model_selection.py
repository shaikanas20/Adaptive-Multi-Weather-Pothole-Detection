
from pathlib import Path
import argparse
import csv
import json
import time

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(r"D:\research papper\rs")

# YOLOv8n checkpoint
SMALL_MODEL = (
    ROOT / "experiments" / "phase1" / "mwpd" / "weights" / "best.pt"
)

# YOLOv8m checkpoint
MEDIUM_MODEL = (
    ROOT / "experiments" / "phase1" / "mwpd_yolov8m" / "weights" / "best.pt"
)

DATA_YAML = (
    ROOT / "data" / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)"
    / "MWPD" / "data.yaml"
)

CONDITIONS_DIR = (
    ROOT / "data" / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions_advanced"
)

OUTPUT_DIR = (
    ROOT / "experiments" / "phase1" / "adaptive_model_selection"
)

CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)


# ============================================================
# 2. DATASET UTILITIES
# ============================================================

def normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split(yaml_path, data, split):
    root = Path(data.get("path", yaml_path.parent))

    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    split_path = Path(data[split])

    if split_path.is_absolute():
        return split_path

    return (root / split_path).resolve()


def list_images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist:\n{folder}")

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise ValueError(f"No images found in:\n{folder}")

    return images


def find_pothole_class(model):
    names = normalize_names(model.names)

    matches = [
        class_id
        for class_id, name in names.items()
        if name.strip().lower() == "pothole"
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one Pothole class. Model classes are: {names}"
        )

    return matches[0]


def get_label_path(image, images_dir, labels_dir=None):
    if labels_dir is not None:
        return labels_dir / (image.stem + ".txt")

    # Standard YOLO layout: images/example.jpg -> labels/example.txt
    parts = list(image.parts)

    image_indices = [
        i for i, part in enumerate(parts)
        if part.lower() == "images"
    ]

    if image_indices:
        parts[image_indices[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")

    return images_dir.parent / "labels" / (image.stem + ".txt")


def load_ground_truth(
    image, images_dir, dataset_names, model_class_id, labels_dir=None
):
    label_path = get_label_path(image, images_dir, labels_dir)

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Missing label file:\n{label_path}\n"
            "For images with no potholes, create an empty label file."
        )

    names = normalize_names(dataset_names)

    pothole_ids = [
    class_id
    for class_id, name in names.items()
    if name.strip().lower() in {"pothole", "potholes"}
    ]

    if len(pothole_ids) != 1:
        raise ValueError(
            f"Dataset YAML must define exactly one Pothole/Potholes class. "
            f"Names: {names}"
        )

    # Supports both a single-class dataset (Pothole = class 0)
    # and a multiclass dataset where Pothole has another class ID.
    single_class = len(names) == 1 and pothole_ids[0] == 0

    with Image.open(image) as im:
        width, height = im.size

    rows = []
    content = label_path.read_text(encoding="utf-8").strip()

    for line in content.splitlines():
        values = line.split()

        if len(values) < 5:
            raise ValueError(f"Invalid YOLO label in {label_path}: {line}")

        class_id = int(float(values[0]))

        if single_class:
            if class_id != 0:
                continue
        elif class_id not in pothole_ids:
            continue

        xc, yc, w, h = map(float, values[1:5])

        # YOLO normalized xywh -> pixel xyxy
        x1 = (xc - w / 2) * width
        y1 = (yc - h / 2) * height
        x2 = (xc + w / 2) * width
        y2 = (yc + h / 2) * height

        rows.append([model_class_id, x1, y1, x2, y2])

    return np.asarray(rows, dtype=float).reshape(-1, 5)


# ============================================================
# 3. INFERENCE
# ============================================================

def predict_one(model, image, pothole_class_id, args):
    start = time.perf_counter()

    result = model.predict(
        source=str(image),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.predict_conf,
        iou=args.nms_iou,
        max_det=args.max_det,
        verbose=False,
        save=False,
    )[0]

    elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []

    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, score, class_id in zip(boxes, scores, classes):
            if class_id == pothole_class_id:
                detections.append([class_id, score, *box.tolist()])

    return np.asarray(detections, dtype=float).reshape(-1, 6), elapsed_ms


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    )

    area_a = (
        max(0, box[2] - box[0]) *
        max(0, box[3] - box[1])
    )

    area_b = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0]) *
        np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    return intersection / np.maximum(
        area_a + area_b - intersection, 1e-12
    )


# ============================================================
# 4. EVALUATION METRICS
# ============================================================

def average_precision(recall, precision):
    if len(recall) == 0:
        return 0.0

    values = []

    for target_recall in np.linspace(0, 1, 101):
        valid = precision[recall >= target_recall]
        values.append(float(np.max(valid)) if len(valid) else 0.0)

    return float(np.mean(values))


def calculate_metrics(predictions, ground_truths):
    total_gt = sum(len(gt) for gt in ground_truths)

    if total_gt == 0:
        raise ValueError(
            "No ground-truth potholes found. Check label class IDs and files."
        )

    # Precision, recall and F1 at confidence >= 0.25 and IoU >= 0.50
    tp = 0
    fp = 0

    for pred, gt in zip(predictions, ground_truths):
        pred = pred[pred[:, 1] >= 0.25]
        pred = pred[np.argsort(-pred[:, 1])]
        matched_gt = set()

        for detection in pred:
            class_id = int(detection[0])
            ids = np.where(gt[:, 0].astype(int) == class_id)[0]

            if len(ids) == 0:
                fp += 1
                continue

            overlaps = box_iou(detection[2:6], gt[ids, 1:5])
            found = False

            for j in np.argsort(-overlaps):
                gt_index = int(ids[j])

                if overlaps[j] >= 0.50 and gt_index not in matched_gt:
                    matched_gt.add(gt_index)
                    tp += 1
                    found = True
                    break

            if not found:
                fp += 1

    precision_value = tp / max(tp + fp, 1)
    recall_value = tp / max(total_gt, 1)

    f1_value = (
        2 * precision_value * recall_value /
        max(precision_value + recall_value, 1e-12)
    )

    # AP at IoU 0.50:0.05:0.95
    aps = []

    for iou_threshold in IOU_THRESHOLDS:
        ranked = []

        for image_index, pred in enumerate(predictions):
            for detection in pred:
                ranked.append(
                    (float(detection[1]), image_index, detection)
                )

        ranked.sort(key=lambda item: item[0], reverse=True)

        true_positive = np.zeros(len(ranked))
        false_positive = np.zeros(len(ranked))
        used = [set() for _ in ground_truths]

        for k, (_, image_index, detection) in enumerate(ranked):
            gt = ground_truths[image_index]
            class_id = int(detection[0])

            ids = np.where(gt[:, 0].astype(int) == class_id)[0]

            if len(ids) == 0:
                false_positive[k] = 1
                continue

            overlaps = box_iou(detection[2:6], gt[ids, 1:5])
            found = False

            for j in np.argsort(-overlaps):
                gt_index = int(ids[j])

                if (
                    overlaps[j] >= iou_threshold
                    and gt_index not in used[image_index]
                ):
                    used[image_index].add(gt_index)
                    true_positive[k] = 1
                    found = True
                    break

            if not found:
                false_positive[k] = 1

        if len(ranked):
            cum_tp = np.cumsum(true_positive)
            cum_fp = np.cumsum(false_positive)

            recall = cum_tp / total_gt
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

            aps.append(average_precision(recall, precision))
        else:
            aps.append(0.0)

    return {
        "precision": float(precision_value),
        "recall": float(recall_value),
        "F1": float(f1_value),
        "mAP50": float(aps[0]),
        "mAP50_95": float(np.mean(aps)),
        "ground_truth_potholes": int(total_gt),
    }


# ============================================================
# 5. FIXED MODEL EVALUATION
# ============================================================

def evaluate_fixed(
    model, files, images_dir, labels_dir, names, class_id, args
):
    predictions = []
    truths = []
    total_latency = 0.0

    for i, image in enumerate(files, 1):
        truths.append(
            load_ground_truth(
                image, images_dir, names, class_id, labels_dir
            )
        )

        pred, latency = predict_one(model, image, class_id, args)
        predictions.append(pred)
        total_latency += latency

        if i % 100 == 0 or i == len(files):
            print(f"    Processed {i}/{len(files)} images")

    result = calculate_metrics(predictions, truths)
    result["images"] = len(files)
    result["mean_latency_ms"] = total_latency / len(files)
    result["escalation_rate"] = 0.0

    return result, predictions, truths


# ============================================================
# 6. ADAPTIVE YOLOv8n -> YOLOv8m
# ============================================================

def evaluate_adaptive(
    small, medium, files, images_dir, labels_dir, names,
    small_id, medium_id, threshold, args
):
    predictions = []
    truths = []
    total_latency = 0.0
    escalated = 0

    for i, image in enumerate(files, 1):
        truths.append(
            load_ground_truth(
                image, images_dir, names, small_id, labels_dir
            )
        )

        small_pred, small_latency = predict_one(
            small, image, small_id, args
        )

        max_confidence = (
            float(np.max(small_pred[:, 1])) if len(small_pred) else 0.0
        )

        if max_confidence < threshold:
            # Low-confidence image: run YOLOv8m.
            medium_pred, medium_latency = predict_one(
                medium, image, medium_id, args
            )

            predictions.append(medium_pred)
            total_latency += small_latency + medium_latency
            escalated += 1
        else:
            # High-confidence image: retain YOLOv8n detections.
            predictions.append(small_pred)
            total_latency += small_latency

        if i % 100 == 0 or i == len(files):
            print(f"    Processed {i}/{len(files)} images")

    result = calculate_metrics(predictions, truths)
    result["images"] = len(files)
    result["mean_latency_ms"] = total_latency / len(files)
    result["escalated_images"] = escalated
    result["escalation_rate"] = escalated / len(files)

    return result


def save_csv(path, rows):
    if not rows:
        return

    columns = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--small-model", type=Path, default=SMALL_MODEL)
    parser.add_argument("--medium-model", type=Path, default=MEDIUM_MODEL)
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML)
    parser.add_argument("--conditions-dir", type=Path, default=CONDITIONS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    parser.add_argument("--device", default="0", help="GPU index or cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--predict-conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.25, 0.35, 0.45, 0.55, 0.65],
    )

    parser.add_argument("--map-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    # Verify paths before loading the checkpoints.
    for path in [
        args.small_model,
        args.medium_model,
        args.data_yaml,
        args.conditions_dir,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found:\n{path}")

    if args.small_model.resolve() == args.medium_model.resolve():
        raise ValueError(
            "YOLOv8n and YOLOv8m point to the same checkpoint. "
            "Set the two paths to different best.pt files."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading YOLOv8n and YOLOv8m...")
    small = YOLO(str(args.small_model))
    medium = YOLO(str(args.medium_model))

    print("YOLOv8n checkpoint:", args.small_model)
    print("YOLOv8m checkpoint:", args.medium_model)
    print("YOLOv8n classes:", small.names)
    print("YOLOv8m classes:", medium.names)

    small_names = {
        k: v.lower() for k, v in normalize_names(small.names).items()
    }
    medium_names = {
        k: v.lower() for k, v in normalize_names(medium.names).items()
    }

    if small_names != medium_names:
        raise ValueError(
            "Class mappings differ between the checkpoints.\n"
            f"YOLOv8n: {small_names}\nYOLOv8m: {medium_names}"
        )

    small_id = find_pothole_class(small)
    medium_id = find_pothole_class(medium)

    data = read_yaml(args.data_yaml)
    dataset_names = normalize_names(data["names"])

    val_images_dir = resolve_split(args.data_yaml, data, "val")
    val_files = list_images(val_images_dir)

    print("\nValidation directory:", val_images_dir)
    print("Validation images:", len(val_files))
    print("\nEvaluating fixed models on validation data...")

    val_n, val_n_preds, val_truths = evaluate_fixed(
        small, val_files, val_images_dir, None,
        dataset_names, small_id, args
    )

    val_m, _, _ = evaluate_fixed(
        medium, val_files, val_images_dir, None,
        dataset_names, medium_id, args
    )

    best_fixed_map = max(val_n["mAP50_95"], val_m["mAP50_95"])
    accuracy_floor = best_fixed_map - args.map_tolerance

    # Calibrate threshold on validation data only.
    calibration_rows = []

    for threshold in args.thresholds:
        adaptive_preds = []
        escalated = 0

        for image, small_pred in zip(val_files, val_n_preds):
            max_confidence = (
                float(np.max(small_pred[:, 1]))
                if len(small_pred) else 0.0
            )

            if max_confidence < threshold:
                medium_pred, _ = predict_one(
                    medium, image, medium_id, args
                )
                adaptive_preds.append(medium_pred)
                escalated += 1
            else:
                adaptive_preds.append(small_pred)

        result = calculate_metrics(adaptive_preds, val_truths)
        result.update({
            "threshold": threshold,
            "escalated_images": escalated,
            "escalation_rate": escalated / len(val_files),
            "validation_accuracy_floor": accuracy_floor,
        })

        calibration_rows.append(result)

        print(
            f"Threshold {threshold:.2f}: "
            f"val mAP50-95={result['mAP50_95']:.4f}, "
            f"escalation={result['escalation_rate']:.1%}"
        )

    eligible = [
        row for row in calibration_rows
        if row["mAP50_95"] >= accuracy_floor
    ]

    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                row["escalation_rate"], -row["mAP50_95"]
            ),
        )
        selection_note = (
            "Lowest escalation rate among thresholds meeting "
            "the validation mAP50-95 floor."
        )
    else:
        selected = max(
            calibration_rows, key=lambda row: row["mAP50_95"]
        )
        selection_note = (
            "No threshold met the validation accuracy floor; "
            "selected the threshold with the best validation mAP50-95."
        )

    threshold = float(selected["threshold"])

    save_csv(
        args.output_dir / "threshold_calibration.csv",
        calibration_rows,
    )

    print(f"\nSelected threshold: {threshold:.2f}")
    print(selection_note)

    # Evaluate the six controlled-condition test folders.
    test_rows = []

    for condition in CONDITIONS:
        condition_dir = args.conditions_dir / condition

        if (condition_dir / "test" / "images").is_dir():
            images_dir = condition_dir / "test" / "images"
            labels_dir = condition_dir / "test" / "labels"
        else:
            images_dir = condition_dir / "images"
            labels_dir = condition_dir / "labels"

        if not images_dir.is_dir():
            print(f"WARNING: Skipping {condition}; missing {images_dir}")
            continue

        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Label directory not found for {condition}:\n{labels_dir}"
            )

        files = list_images(images_dir)

        print(f"\nCondition: {condition} ({len(files)} images)")

        result_n, _, _ = evaluate_fixed(
            small, files, images_dir, labels_dir,
            dataset_names, small_id, args
        )
        result_n.update({
            "condition": condition,
            "strategy": "fixed_yolov8n",
        })
        test_rows.append(result_n)

        result_m, _, _ = evaluate_fixed(
            medium, files, images_dir, labels_dir,
            dataset_names, medium_id, args
        )
        result_m.update({
            "condition": condition,
            "strategy": "fixed_yolov8m",
        })
        test_rows.append(result_m)

        result_a = evaluate_adaptive(
            small, medium, files, images_dir, labels_dir,
            dataset_names, small_id, medium_id, threshold, args
        )
        result_a.update({
            "condition": condition,
            "strategy": "adaptive_yolov8n_to_yolov8m",
            "threshold": threshold,
        })
        test_rows.append(result_a)

        for result in [result_n, result_m, result_a]:
            print(
                f"  {result['strategy']}: "
                f"mAP50={result['mAP50']:.4f}, "
                f"mAP50-95={result['mAP50_95']:.4f}, "
                f"latency={result['mean_latency_ms']:.2f} ms, "
                f"escalation={result['escalation_rate']:.1%}"
            )

    if not test_rows:
        raise RuntimeError(
            "No condition results were produced. Verify the condition folders."
        )

    save_csv(
        args.output_dir / "heldout_test_results_by_condition.csv",
        test_rows,
    )

    summary_rows = []

    strategies = [
        "fixed_yolov8n",
        "fixed_yolov8m",
        "adaptive_yolov8n_to_yolov8m",
    ]

    for strategy in strategies:
        subset = [
            row for row in test_rows if row["strategy"] == strategy
        ]

        if not subset:
            continue

        summary = {
            "strategy": strategy,
            "conditions": len(subset),
            "aggregation": "Unweighted mean across evaluated conditions",
        }

        for key in [
            "precision", "recall", "F1", "mAP50",
            "mAP50_95", "mean_latency_ms", "escalation_rate",
        ]:
            summary[key] = float(
                np.mean([row[key] for row in subset])
            )

        summary_rows.append(summary)

    save_csv(
        args.output_dir / "heldout_test_summary_condition_mean.csv",
        summary_rows,
    )

    with open(
        args.output_dir / "experiment1_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({
            "yolov8n_checkpoint": str(args.small_model),
            "yolov8m_checkpoint": str(args.medium_model),
            "selected_threshold": threshold,
            "selection_note": selection_note,
            "validation_yolov8n": val_n,
            "validation_yolov8m": val_m,
            "validation_accuracy_floor": accuracy_floor,
            "test_results_by_condition": test_rows,
            "test_summary_condition_means": summary_rows,
            "metric_note": (
                "Pothole-only metrics using this script's custom "
                "101-point AP implementation."
            ),
            "latency_note": (
                "Wall-clock inference timing; this is not an energy measurement."
            ),
        }, f, indent=2)

    print("\nExperiment completed.")
    print("Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
