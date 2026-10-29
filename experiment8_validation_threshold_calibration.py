
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
# EXPERIMENT 8: VALIDATION-ONLY THRESHOLD CALIBRATION
# ============================================================

ROOT = Path(r"D:\research papper\rs")

NANO_PATH = (
    ROOT / "experiments" / "phase1" / "mwpd"
    / "weights" / "best.pt"
)
MEDIUM_PATH = (
    ROOT / "experiments" / "phase1" / "mwpd_yolov8m"
    / "weights" / "best.pt"
)

DATASET_ROOT = (
    ROOT / "data" / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)" / "MWPD"
)

VAL_IMAGES = DATASET_ROOT / "valid" / "images"
VAL_LABELS = DATASET_ROOT / "valid" / "labels"

OUTPUT_ROOT = (
    ROOT / "experiments" / "phase1"
    / "validation_threshold_calibration"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IMAGE_SIZE = 640
PREDICT_CONF = 0.001
METRIC_CONF = 0.25
NMS_IOU = 0.7
MAX_DET = 300
IOU_F1 = 0.50
AP_IOUS = [0.50 + 0.05 * i for i in range(10)]

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Previous preliminary three-repeat energy estimates.
# These are approximate GPU board energy per image, not
# total laptop energy and not fresh validation measurements.
NANO_ENERGY_J_PER_IMAGE = 0.3707
MEDIUM_ENERGY_J_PER_IMAGE = 1.2604

# Predeclared quality and energy constraints.
MAX_F1_DROP = 0.01
MAX_MAP5095_DROP = 0.02
ENERGY_BUDGET_FRACTION_OF_MEDIUM = 0.50

# Sweep thresholds from 0.00 through 0.80, including the
# no-escalation baseline. Routing condition: max nano confidence
# < candidate threshold.
THRESHOLDS = [round(i * 0.05, 2) for i in range(17)]

RUN_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")
OUTPUT_DIR = OUTPUT_ROOT / RUN_NAME


# ============================================================
# VALIDATION
# ============================================================

def validate_paths():
    paths = {
        "nano checkpoint": NANO_PATH,
        "medium checkpoint": MEDIUM_PATH,
        "validation image folder": VAL_IMAGES,
        "validation label folder": VAL_LABELS,
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found:\n{path}")


def list_images(folder):
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No images found in {folder}")

    return images


def load_ground_truth(image_path, image_width, image_height):
    label_path = VAL_LABELS / f"{image_path.stem}.txt"

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Missing validation label:\n{label_path}\n"
            "Do not silently treat a missing label as an empty label."
        )

    boxes = []

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Malformed label at {label_path}, "
                    f"line {line_number}: {line}"
                )

            class_id, xc, yc, bw, bh = map(float, parts)

            # This project evaluates the single pothole class.
            if int(class_id) != 0:
                continue

            x1 = (xc - bw / 2) * image_width
            y1 = (yc - bh / 2) * image_height
            x2 = (xc + bw / 2) * image_width
            y2 = (yc + bh / 2) * image_height

            boxes.append([
                max(0.0, min(x1, image_width)),
                max(0.0, min(y1, image_height)),
                max(0.0, min(x2, image_width)),
                max(0.0, min(y2, image_height)),
            ])

    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


# ============================================================
# INFERENCE
# ============================================================

def predict_one(model, image_path):
    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    result = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=PREDICT_CONF,
        iou=NMS_IOU,
        max_det=MAX_DET,
        device=DEVICE,
        verbose=False,
        save=False,
    )[0]

    # Ensure queued GPU work is finished before ending timing.
    if DEVICE != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    if result.boxes is None or len(result.boxes) == 0:
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
    else:
        boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)

    return {
        "boxes": boxes,
        "scores": scores,
        "latency_ms": elapsed_ms,
    }


def max_confidence(prediction):
    scores = prediction["scores"]
    return float(np.max(scores)) if len(scores) else 0.0


# ============================================================
# IoU AND GREEDY MATCHING
# ============================================================

def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0.0, x2 - x1)
        * np.maximum(0.0, y2 - y1)
    )

    area_a = (
        max(0.0, box[2] - box[0])
        * max(0.0, box[3] - box[1])
    )
    area_b = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = area_a + area_b - intersection
    return intersection / np.maximum(union, 1e-12)


def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold):
    """
    Return a TP flag per prediction, preserving original prediction order.
    Matching is greedy and confidence-ordered.
    """
    flags = np.zeros(len(pred_boxes), dtype=np.int32)

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return flags

    used_gt = np.zeros(len(gt_boxes), dtype=bool)
    order = np.argsort(-pred_scores)

    for pred_index in order:
        overlaps = box_iou(pred_boxes[pred_index], gt_boxes)

        for gt_index in np.argsort(-overlaps):
            if (
                overlaps[gt_index] >= iou_threshold
                and not used_gt[gt_index]
            ):
                flags[pred_index] = 1
                used_gt[gt_index] = True
                break

    return flags


# ============================================================
# CUSTOM METRICS
# ============================================================

def ap_101(recalls, precisions):
    if len(recalls) == 0:
        return 0.0

    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    interpolated = []

    for recall_level in np.linspace(0.0, 1.0, 101):
        valid = precisions[recalls >= recall_level]
        interpolated.append(float(np.max(valid)) if len(valid) else 0.0)

    return float(np.mean(interpolated))


def evaluate_predictions(predictions, ground_truths):
    """
    Pooled precision/recall/F1 at METRIC_CONF and IoU=0.50.
    Custom interpolated AP at IoU 0.50:0.05:0.95.
    """
    total_gt = sum(len(gt) for gt in ground_truths)

    if total_gt == 0:
        raise RuntimeError("No ground-truth pothole boxes in validation data.")

    tp_total = fp_total = 0

    # Precision, recall and F1 at the fixed operating confidence.
    for prediction, gt in zip(predictions, ground_truths):
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        keep = scores >= METRIC_CONF

        boxes = boxes[keep]
        scores = scores[keep]

        flags = match_predictions(boxes, scores, gt, IOU_F1)

        tp_total += int(flags.sum())
        fp_total += int(len(boxes) - flags.sum())

    fn_total = total_gt - tp_total

    precision = (
        tp_total / (tp_total + fp_total)
        if tp_total + fp_total else 0.0
    )
    recall = tp_total / total_gt
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    # Confidence-ranked AP across the whole validation split.
    ranked = []

    for image_idx, prediction in enumerate(predictions):
        for box, score in zip(
            prediction["boxes"], prediction["scores"]
        ):
            ranked.append((float(score), image_idx, box))

    ranked.sort(key=lambda item: item[0], reverse=True)

    aps = []

    for iou_threshold in AP_IOUS:
        tp_flags = np.zeros(len(ranked), dtype=np.float64)
        fp_flags = np.zeros(len(ranked), dtype=np.float64)
        matched_gt = [set() for _ in ground_truths]

        for k, (_, image_idx, box) in enumerate(ranked):
            gt = ground_truths[image_idx]

            if len(gt) == 0:
                fp_flags[k] = 1.0
                continue

            overlaps = box_iou(box, gt)
            matched = False

            for gt_idx in np.argsort(-overlaps):
                if (
                    overlaps[gt_idx] >= iou_threshold
                    and gt_idx not in matched_gt[image_idx]
                ):
                    matched_gt[image_idx].add(int(gt_idx))
                    tp_flags[k] = 1.0
                    matched = True
                    break

            if not matched:
                fp_flags[k] = 1.0

        if len(ranked) == 0:
            aps.append(0.0)
            continue

        cumulative_tp = np.cumsum(tp_flags)
        cumulative_fp = np.cumsum(fp_flags)

        recalls = cumulative_tp / total_gt
        precisions = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, 1e-12
        )

        aps.append(ap_101(recalls, precisions))

    return {
        "ground_truth_boxes": int(total_gt),
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mAP50": float(aps[0]),
        "mAP50_95": float(np.mean(aps)),
    }


# ============================================================
# MAIN: ONE VALIDATION INFERENCE PASS, THEN OFFLINE SWEEP
# ============================================================

def main():
    validate_paths()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    files = list_images(VAL_IMAGES)
    print(f"Validation images: {len(files)}")
    print(f"Device: {DEVICE}")

    print("Loading checkpoints...")
    nano = YOLO(str(NANO_PATH))
    medium = YOLO(str(MEDIUM_PATH))

    print("Warm-up...")
    predict_one(nano, files[0])
    predict_one(medium, files[0])

    nano_preds = []
    medium_preds = []
    ground_truths = []

    for i, image_path in enumerate(files, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        h, w = image.shape[:2]
        gt = load_ground_truth(image_path, w, h)

        n_pred = predict_one(nano, image_path)
        m_pred = predict_one(medium, image_path)

        nano_preds.append(n_pred)
        medium_preds.append(m_pred)
        ground_truths.append(gt)

        if i % 50 == 0 or i == len(files):
            print(f"Processed {i}/{len(files)} validation images")

    # Fixed model baselines on the validation split.
    fixed_n_metrics = evaluate_predictions(nano_preds, ground_truths)
    fixed_m_metrics = evaluate_predictions(medium_preds, ground_truths)

    best_fixed_map95 = max(
        fixed_n_metrics["mAP50_95"],
        fixed_m_metrics["mAP50_95"],
    )

    map95_floor = best_fixed_map95 - MAX_MAP5095_DROP
    f1_floor = fixed_n_metrics["f1"] - MAX_F1_DROP
    energy_budget = (
        ENERGY_BUDGET_FRACTION_OF_MEDIUM
        * MEDIUM_ENERGY_J_PER_IMAGE
    )

    nano_times = np.asarray(
        [p["latency_ms"] for p in nano_preds], dtype=float
    )
    medium_times = np.asarray(
        [p["latency_ms"] for p in medium_preds], dtype=float
    )

    nano_max_conf = np.asarray(
        [max_confidence(p) for p in nano_preds], dtype=float
    )

    rows = []

    for threshold in THRESHOLDS:
        routed = nano_max_conf < threshold
        selected_predictions = [
            medium_preds[i] if routed[i] else nano_preds[i]
            for i in range(len(files))
        ]

        metrics = evaluate_predictions(
            selected_predictions, ground_truths
        )

        escalation_rate = float(np.mean(routed))

        # Expected cascade energy:
        # nano runs on every image; medium runs only on escalated images.
        estimated_energy = (
            NANO_ENERGY_J_PER_IMAGE
            + escalation_rate * MEDIUM_ENERGY_J_PER_IMAGE
        )

        estimated_latency = float(
            np.mean(nano_times + routed.astype(float) * medium_times)
        )

        rows.append({
            "threshold": threshold,
            **metrics,
            "escalated_images": int(routed.sum()),
            "escalation_rate": escalation_rate,
            "estimated_latency_ms": estimated_latency,
            "estimated_energy_J_per_image": estimated_energy,
            "estimated_energy_saving_vs_medium_pct": (
                100.0 * (
                    1.0 - estimated_energy / MEDIUM_ENERGY_J_PER_IMAGE
                )
            ),
            "f1_floor": f1_floor,
            "map50_95_floor": map95_floor,
            "energy_budget_J_per_image": energy_budget,
        })

    # Selection objective:
    # maximize recall while respecting all predeclared constraints.
    eligible = [
        row for row in rows
        if (
            row["f1"] >= f1_floor
            and row["mAP50_95"] >= map95_floor
            and row["estimated_energy_J_per_image"] <= energy_budget
        )
    ]

    if not eligible:
        raise RuntimeError(
            "No threshold satisfied all predeclared constraints. "
            "Inspect the saved sweep before changing any constraint."
        )

    selected = sorted(
        eligible,
        key=lambda row: (
            -row["recall"],
            -row["f1"],
            -row["mAP50_95"],
            row["estimated_energy_J_per_image"],
        ),
    )[0]

    # Save threshold sweep.
    sweep_path = OUTPUT_DIR / "validation_threshold_sweep.csv"

    with sweep_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "experiment": "Validation-only confidence threshold calibration",
        "run_name": RUN_NAME,
        "threshold_selected": selected["threshold"],
        "selection_objective": (
            "Maximize validation recall subject to: F1 no more than "
            f"{MAX_F1_DROP:.3f} below fixed nano; mAP50-95 no more than "
            f"{MAX_MAP5095_DROP:.3f} below the better fixed model; and "
            f"estimated energy no more than "
            f"{ENERGY_BUDGET_FRACTION_OF_MEDIUM:.0%} of fixed medium."
        ),
        "validation_image_count": len(files),
        "device": str(DEVICE),
        "prediction_confidence": PREDICT_CONF,
        "metric_confidence": METRIC_CONF,
        "nms_iou": NMS_IOU,
        "image_size": IMAGE_SIZE,
        "nano_checkpoint": str(NANO_PATH),
        "medium_checkpoint": str(MEDIUM_PATH),
        "fixed_nano_validation": fixed_n_metrics,
        "fixed_medium_validation": fixed_m_metrics,
        "constraints": {
            "minimum_f1": f1_floor,
            "minimum_map50_95": map95_floor,
            "maximum_estimated_energy_J_per_image": energy_budget,
        },
        "selected_validation_result": selected,
        "energy_model": {
            "nano_energy_J_per_image": NANO_ENERGY_J_PER_IMAGE,
            "medium_energy_J_per_inference": MEDIUM_ENERGY_J_PER_IMAGE,
            "formula": (
                "estimated adaptive energy = nano energy per image + "
                "escalation rate * medium energy per inference"
            ),
            "caveat": (
                "Energy costs are approximate means from the earlier "
                "three-repeat GPU measurement. They are not fresh power "
                "measurements on the validation split."
            ),
        },
        "metric_caveat": (
            "These are custom single-class validation metrics, not official "
            "Ultralytics validation metrics. Threshold selection uses only "
            "the validation split; do not use held-out test results to "
            "change the selected threshold."
        ),
    }

    summary_path = OUTPUT_DIR / "validation_threshold_selection.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 72)
    print("VALIDATION THRESHOLD SWEEP COMPLETED")
    print("=" * 72)

    print(
        f"Fixed nano: F1={fixed_n_metrics['f1']:.4f}, "
        f"Recall={fixed_n_metrics['recall']:.4f}, "
        f"mAP50-95={fixed_n_metrics['mAP50_95']:.4f}"
    )
    print(
        f"Fixed medium: F1={fixed_m_metrics['f1']:.4f}, "
        f"Recall={fixed_m_metrics['recall']:.4f}, "
        f"mAP50-95={fixed_m_metrics['mAP50_95']:.4f}"
    )

    print(f"\nSelected threshold: {selected['threshold']:.2f}")
    print(f"Escalation rate: {selected['escalation_rate']:.2%}")
    print(f"Validation recall: {selected['recall']:.4f}")
    print(f"Validation F1: {selected['f1']:.4f}")
    print(f"Validation mAP50: {selected['mAP50']:.4f}")
    print(f"Validation mAP50-95: {selected['mAP50_95']:.4f}")
    print(
        "Estimated energy/image: "
        f"{selected['estimated_energy_J_per_image']:.4f} J"
    )
    print(
        "Estimated latency/image: "
        f"{selected['estimated_latency_ms']:.3f} ms"
    )

    print("\nSaved files:")
    print(sweep_path)
    print(summary_path)
    print("\nDo not tune the threshold using the held-out condition test set.")


if __name__ == "__main__":
    main()
