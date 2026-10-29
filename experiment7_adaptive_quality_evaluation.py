
from pathlib import Path
from datetime import datetime
import csv
import json
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# EXPERIMENT 7: ADAPTIVE DETECTION QUALITY EVALUATION
# ============================================================

ROOT = Path(r"D:\research papper\rs")

NANO_PATH = (
    ROOT / "experiments" / "phase1" / "mwpd" / "weights" / "best.pt"
)
MEDIUM_PATH = (
    ROOT / "experiments" / "phase1" / "mwpd_yolov8m" / "weights" / "best.pt"
)

CONDITIONS_ROOT = (
    ROOT / "data" / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions_advanced"
)

DATASET_ROOT = (
    ROOT / "data" / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)" / "MWPD"
)

LABELS_DIR = DATASET_ROOT / "test" / "labels"

CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Keep these settings consistent with the earlier experiments.
IMAGE_SIZE = 640
PREDICT_CONF = 0.001
NMS_IOU = 0.7
MAX_DET = 300
ADAPTIVE_THRESHOLD = 0.30

# Confidence threshold for the reported precision, recall and F1.
# AP is calculated using predictions across confidence scores.
METRIC_CONF = 0.25

IOU_METRIC = 0.50
AP_IOU_THRESHOLDS = np.arange(0.50, 0.951, 0.05)

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# A fresh output directory preserves previous experiment results.
RUN_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")
OUTPUT_DIR = (
    ROOT / "experiments" / "phase1"
    / "adaptive_quality_evaluation" / RUN_NAME
)


# ============================================================
# VALIDATION AND IMAGE DISCOVERY
# ============================================================

def validate_paths():
    required = {
        "YOLOv8n checkpoint": NANO_PATH,
        "YOLOv8m checkpoint": MEDIUM_PATH,
        "condition image root": CONDITIONS_ROOT,
        "canonical labels directory": LABELS_DIR,
    }

    for description, path in required.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{description} not found:\n{path}"
            )

    for condition in CONDITIONS:
        image_dir = CONDITIONS_ROOT / condition / "images"

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Expected condition image directory not found:\n"
                f"{image_dir}"
            )


def get_condition_images(condition):
    # Actual dataset layout:
    # controlled_conditions_advanced/<condition>/images/*.jpg
    image_dir = CONDITIONS_ROOT / condition / "images"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(
            f"No supported images found in:\n{image_dir}"
        )

    return images


# ============================================================
# GROUND-TRUTH LABEL LOADING
# ============================================================

def load_labels(image_path, image_width, image_height):
    """
    Reads YOLO-format labels from the original test/labels folder.

    Label format:
        class_id x_center y_center width height

    Coordinates are normalized to [0, 1].
    Returns ground-truth boxes in pixel xyxy format.
    """
    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing ground-truth label for image:\n"
            f"Image: {image_path}\n"
            f"Expected label: {label_path}"
        )

    boxes = []

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label at {label_path}, "
                    f"line {line_number}: {line}"
                )

            class_id, xc, yc, width, height = map(float, parts)

            # This experiment evaluates the single pothole class.
            if int(class_id) != 0:
                continue

            x1 = (xc - width / 2) * image_width
            y1 = (yc - height / 2) * image_height
            x2 = (xc + width / 2) * image_width
            y2 = (yc + height / 2) * image_height

            boxes.append([
                max(0.0, min(x1, image_width)),
                max(0.0, min(y1, image_height)),
                max(0.0, min(x2, image_width)),
                max(0.0, min(y2, image_height)),
            ])

    if not boxes:
        return np.empty((0, 4), dtype=np.float32)

    return np.asarray(boxes, dtype=np.float32)


# ============================================================
# MODEL INFERENCE
# ============================================================

def predict_one(model, image_path):
    """
    Returns predicted boxes, confidence scores and elapsed time.

    The measured time includes the prediction call and conversion
    of the returned boxes to CPU NumPy arrays.
    """
    start = time.perf_counter()

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=PREDICT_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        device=DEVICE,
        verbose=False,
        save=False,
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
    else:
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return boxes, scores, elapsed_ms


def max_detection_confidence(scores):
    if len(scores) == 0:
        return 0.0
    return float(np.max(scores))


# ============================================================
# BOX MATCHING AND METRICS
# ============================================================

def box_iou(box, boxes):
    """IoU between one xyxy box and an array of xyxy boxes."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0.0, x2 - x1)
        * np.maximum(0.0, y2 - y1)
    )

    area_box = (
        max(0.0, box[2] - box[0])
        * max(0.0, box[3] - box[1])
    )

    area_boxes = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = area_box + area_boxes - intersection

    return intersection / np.maximum(union, 1e-12)


def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold):
    """
    Greedy, confidence-ordered, one-to-one matching.

    Returns a TP/FP flag for each prediction in its original order.
    """
    number_predictions = len(pred_boxes)
    tp_flags = np.zeros(number_predictions, dtype=np.int32)

    if number_predictions == 0 or len(gt_boxes) == 0:
        return tp_flags

    order = np.argsort(-pred_scores)
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)

    for prediction_index in order:
        ious = box_iou(
            pred_boxes[prediction_index],
            gt_boxes,
        )

        if len(ious) == 0:
            continue

        # Try the highest-IoU unmatched ground-truth box.
        candidates = np.argsort(-ious)

        for gt_index in candidates:
            if (
                ious[gt_index] >= iou_threshold
                and not matched_gt[gt_index]
            ):
                tp_flags[prediction_index] = 1
                matched_gt[gt_index] = True
                break

    return tp_flags


def calculate_precision_recall_f1(
    pred_boxes,
    pred_scores,
    gt_boxes,
    confidence_threshold=METRIC_CONF,
    iou_threshold=IOU_METRIC,
):
    keep = pred_scores >= confidence_threshold

    boxes = pred_boxes[keep]
    scores = pred_scores[keep]

    tp_flags = match_predictions(
        boxes, scores, gt_boxes, iou_threshold
    )

    tp = int(tp_flags.sum())
    fp = int(len(boxes) - tp)
    fn = int(len(gt_boxes) - tp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def average_precision(tp_flags, scores, total_gt):
    """
    Computes interpolated AP from globally confidence-ranked
    predictions for one condition and one IoU threshold.
    """
    if total_gt == 0:
        return float("nan")

    if len(scores) == 0:
        return 0.0

    order = np.argsort(-scores)
    tp_sorted = tp_flags[order].astype(np.float64)
    fp_sorted = 1.0 - tp_sorted

    cumulative_tp = np.cumsum(tp_sorted)
    cumulative_fp = np.cumsum(fp_sorted)

    recalls = cumulative_tp / total_gt
    precisions = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, 1e-12
    )

    # Precision envelope.
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    changes = np.where(recalls[1:] != recalls[:-1])[0]

    return float(
        np.sum(
            (recalls[changes + 1] - recalls[changes])
            * precisions[changes + 1]
        )
    )


# ============================================================
# PER-CONDITION AGGREGATION
# ============================================================

def summarize_condition_strategy(
    condition,
    strategy,
    image_records,
    total_gt,
):
    """
    Computes pooled precision/recall/F1 and AP for one condition.

    Pooled metrics are computed from summed TP, FP and FN.
    AP is calculated by ranking predictions across all images
    in that condition.
    """
    tp = sum(record["tp"] for record in image_records)
    fp = sum(record["fp"] for record in image_records)
    fn = sum(record["fn"] for record in image_records)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    all_scores = []
    all_tp_by_iou = {
        float(threshold): []
        for threshold in AP_IOU_THRESHOLDS
    }

    for record in image_records:
        scores = record["all_scores"]
        boxes = record["all_boxes"]
        gt_boxes = record["gt_boxes"]

        all_scores.extend(scores.tolist())

        for threshold in AP_IOU_THRESHOLDS:
            matched = match_predictions(
                boxes,
                scores,
                gt_boxes,
                float(threshold),
            )
            all_tp_by_iou[float(threshold)].extend(
                matched.tolist()
            )

    scores_array = np.asarray(all_scores, dtype=np.float32)

    aps = []
    for threshold in AP_IOU_THRESHOLDS:
        threshold_key = float(threshold)
        tp_array = np.asarray(
            all_tp_by_iou[threshold_key],
            dtype=np.int32,
        )

        ap = average_precision(
            tp_array,
            scores_array,
            total_gt,
        )
        aps.append(ap)

    ap50 = aps[0]
    map50_95 = float(np.nanmean(aps)) if aps else float("nan")

    return {
        "condition": condition,
        "strategy": strategy,
        "images": len(image_records),
        "ground_truth_boxes": total_gt,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": ap50,
        "mAP50_95": map50_95,
        "mean_image_latency_ms": float(np.mean([
            record["strategy_latency_ms"]
            for record in image_records
        ])),
        "total_escalations": sum(
            int(record["escalated"])
            for record in image_records
        ),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    validate_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    print("Loading checkpoints...")
    print(f"Device: {DEVICE}")

    nano_model = YOLO(str(NANO_PATH))
    medium_model = YOLO(str(MEDIUM_PATH))

    strategies = ["fixed_n", "fixed_m", "adaptive"]

    all_condition_metrics = []
    all_image_rows = []

    # Warm up both models using a real image.
    warmup_image = None

    for condition in CONDITIONS:
        candidates = get_condition_images(condition)
        if candidates:
            warmup_image = candidates[0]
            break

    if warmup_image is None:
        raise RuntimeError(
            f"No condition images found under {CONDITIONS_ROOT}"
        )

    print(f"Warm-up image: {warmup_image.name}")

    predict_one(nano_model, warmup_image)
    predict_one(medium_model, warmup_image)

    print("\nStarting evaluation...")
    print(f"Output directory: {OUTPUT_DIR}")

    for condition in CONDITIONS:
        images = get_condition_images(condition)
        print(f"\nCondition: {condition} | Images: {len(images)}")

        # Per-image records used for pooled metrics and AP.
        records_by_strategy = {
            strategy: [] for strategy in strategies
        }

        total_gt = 0

        for image_number, image_path in enumerate(images, start=1):
            image = cv2.imread(str(image_path))

            if image is None:
                raise RuntimeError(
                    f"OpenCV could not read image:\n{image_path}"
                )

            image_height, image_width = image.shape[:2]

            gt_boxes = load_labels(
                image_path,
                image_width,
                image_height,
            )
            total_gt += len(gt_boxes)

            # Run both models so fixed-model baselines can be
            # compared on the exact same images.
            n_boxes, n_scores, n_ms = predict_one(
                nano_model, image_path
            )
            m_boxes, m_scores, m_ms = predict_one(
                medium_model, image_path
            )

            n_max_conf = max_detection_confidence(n_scores)
            m_max_conf = max_detection_confidence(m_scores)

            # Adaptive routing is determined ONLY by nano confidence.
            escalated = n_max_conf < ADAPTIVE_THRESHOLD

            if escalated:
                adaptive_boxes = m_boxes
                adaptive_scores = m_scores
                adaptive_latency_estimate = n_ms + m_ms
            else:
                adaptive_boxes = n_boxes
                adaptive_scores = n_scores
                adaptive_latency_estimate = n_ms

            selected = {
                "fixed_n": (n_boxes, n_scores, n_ms),
                "fixed_m": (m_boxes, m_scores, m_ms),
                "adaptive": (
                    adaptive_boxes,
                    adaptive_scores,
                    adaptive_latency_estimate,
                ),
            }

            for strategy, (pred_boxes, pred_scores, latency_ms) in selected.items():
                point_metrics = calculate_precision_recall_f1(
                    pred_boxes,
                    pred_scores,
                    gt_boxes,
                )

                record = {
                    "tp": point_metrics["tp"],
                    "fp": point_metrics["fp"],
                    "fn": point_metrics["fn"],
                    "all_boxes": pred_boxes,
                    "all_scores": pred_scores,
                    "gt_boxes": gt_boxes,
                    "strategy_latency_ms": latency_ms,
                    "escalated": (
                        escalated if strategy == "adaptive" else False
                    ),
                }

                records_by_strategy[strategy].append(record)

                all_image_rows.append({
                    "condition": condition,
                    "image": image_path.name,
                    "strategy": strategy,
                    "ground_truth_boxes": len(gt_boxes),
                    "predictions_above_metric_conf": int(
                        np.sum(pred_scores >= METRIC_CONF)
                    ),
                    "tp": point_metrics["tp"],
                    "fp": point_metrics["fp"],
                    "fn": point_metrics["fn"],
                    "precision": point_metrics["precision"],
                    "recall": point_metrics["recall"],
                    "f1": point_metrics["f1"],
                    "nano_max_confidence": n_max_conf,
                    "medium_max_confidence": m_max_conf,
                    "adaptive_escalated": (
                        int(escalated) if strategy == "adaptive" else 0
                    ),
                    "nano_inference_ms": n_ms,
                    "medium_inference_ms": m_ms,
                    "strategy_latency_ms": latency_ms,
                })

            if image_number % 20 == 0 or image_number == len(images):
                print(
                    f"  Processed {image_number}/{len(images)} images",
                    end="\r",
                    flush=True,
                )

        print()

        for strategy in strategies:
            summary = summarize_condition_strategy(
                condition=condition,
                strategy=strategy,
                image_records=records_by_strategy[strategy],
                total_gt=total_gt,
            )
            all_condition_metrics.append(summary)

            print(
                f"  {strategy:8s} | "
                f"P={summary['precision']:.4f} "
                f"R={summary['recall']:.4f} "
                f"F1={summary['f1']:.4f} "
                f"mAP50={summary['mAP50']:.4f} "
                f"mAP50-95={summary['mAP50_95']:.4f}"
            )

    # --------------------------------------------------------
    # Save condition/strategy metrics
    # --------------------------------------------------------

    condition_csv = OUTPUT_DIR / "condition_strategy_metrics.csv"

    condition_fields = list(all_condition_metrics[0].keys())

    with condition_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=condition_fields)
        writer.writeheader()
        writer.writerows(all_condition_metrics)

    # --------------------------------------------------------
    # Save per-image metrics and routing details
    # --------------------------------------------------------

    per_image_csv = (
        OUTPUT_DIR / "per_image_latency_and_routing.csv"
    )

    with per_image_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(all_image_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_image_rows)

    # --------------------------------------------------------
    # Six-condition macro summary
    # Each condition contributes equally to the macro average.
    # --------------------------------------------------------

    macro_rows = []

    metric_columns = [
        "precision",
        "recall",
        "f1",
        "mAP50",
        "mAP50_95",
        "mean_image_latency_ms",
    ]

    for strategy in strategies:
        rows = [
            row for row in all_condition_metrics
            if row["strategy"] == strategy
        ]

        macro = {
            "strategy": strategy,
            "conditions": len(rows),
        }

        for metric in metric_columns:
            macro[f"macro_{metric}"] = float(
                np.mean([row[metric] for row in rows])
            )

        macro["total_escalations"] = sum(
            row["total_escalations"] for row in rows
        )
        macro["total_images"] = sum(
            row["images"] for row in rows
        )
        macro["escalation_rate"] = (
            macro["total_escalations"] / macro["total_images"]
            if macro["total_images"]
            else 0.0
        )

        macro_rows.append(macro)

    macro_csv = OUTPUT_DIR / "six_condition_macro_summary.csv"

    with macro_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(macro_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(macro_rows)

    # --------------------------------------------------------
    # Save configuration and metric definitions
    # --------------------------------------------------------

    config = {
        "experiment": "Experiment 7 - Adaptive Detection Quality Evaluation",
        "run_name": RUN_NAME,
        "device": str(DEVICE),
        "nano_checkpoint": str(NANO_PATH),
        "medium_checkpoint": str(MEDIUM_PATH),
        "condition_images_root": str(CONDITIONS_ROOT),
        "labels_directory": str(LABELS_DIR),
        "conditions": CONDITIONS,
        "images_per_condition": {
            condition: len(get_condition_images(condition))
            for condition in CONDITIONS
        },
        "image_size": IMAGE_SIZE,
        "prediction_confidence": PREDICT_CONF,
        "nms_iou": NMS_IOU,
        "max_detections": MAX_DET,
        "adaptive_threshold": ADAPTIVE_THRESHOLD,
        "metric_confidence_threshold": METRIC_CONF,
        "precision_recall_f1_iou_threshold": IOU_METRIC,
        "ap_iou_thresholds": [
            float(value) for value in AP_IOU_THRESHOLDS
        ],
        "label_source": (
            "Canonical MWPD test labels matched by image filename stem"
        ),
        "metric_method": (
            "Custom single-class, greedy one-to-one IoU matching. "
            "Pooled precision/recall/F1 are computed per condition. "
            "AP is calculated by globally confidence-ranking predictions "
            "within each condition."
        ),
        "latency_caveat": (
            "The adaptive latency column is an estimated selected-path "
            "latency: nano inference time plus medium inference time only "
            "when nano confidence is below threshold. Both models are "
            "executed for every image in this evaluation to obtain the "
            "fixed-model baselines. Therefore, this is NOT a direct "
            "wall-clock measurement of a deployed adaptive pipeline."
        ),
        "important_limitations": [
            "These are custom metrics and may differ from official Ultralytics validation metrics.",
            "Condition images share canonical labels matched by filename stem.",
            "The evaluation does not establish statistical significance.",
            "The script evaluates detection quality and estimated path latency, not energy consumption.",
        ],
    }

    config_path = OUTPUT_DIR / "evaluation_config.json"

    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("EXPERIMENT 7 COMPLETED")
    print("=" * 72)

    print(f"Output directory:\n{OUTPUT_DIR}")

    print("\nSix-condition macro summary:")

    for row in macro_rows:
        print(
            f"{row['strategy']:8s} | "
            f"P={row['macro_precision']:.4f} "
            f"R={row['macro_recall']:.4f} "
            f"F1={row['macro_f1']:.4f} "
            f"mAP50={row['macro_mAP50']:.4f} "
            f"mAP50-95={row['macro_mAP50_95']:.4f} "
            f"Latency estimate={row['macro_mean_image_latency_ms']:.3f} ms "
            f"Escalations={row['total_escalations']}/"
            f"{row['total_images']}"
        )

    print("\nFiles created:")
    print(f"1. {condition_csv}")
    print(f"2. {per_image_csv}")
    print(f"3. {macro_csv}")
    print(f"4. {config_path}")


if __name__ == "__main__":
    main()
