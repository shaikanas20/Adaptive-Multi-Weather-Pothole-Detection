
from pathlib import Path
import csv
import yaml
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(r"D:\research papper\rs")
DATA = ROOT / "data/Multi-Weather Pothole Detection (MWPD)/Multi-Weather Pothole Detection (MWPD)/MWPD/data.yaml"
MODELS = {
    "yolov8n": ROOT / "experiments/phase1/mwpd/weights/best.pt",
    "yolov8m": ROOT / "experiments/phase1/mwpd_yolov8m/weights/best.pt",
}
WEATHER = ROOT / "data/Multi-Weather Pothole Detection (MWPD)/controlled_conditions"
OUT = ROOT / "experiments/phase1/experiment4_per_image_errors"

CONDITIONS = [
    "normal", "low_light", "fog", "rain", "shadow", "water_reflection"
]
CONF = 0.25
PREDICT_CONF = 0.001
IOU_THRESHOLD = 0.50
DEVICE = "0"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def names_dict(names):
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def read_gt(image, label, width, height):
    if not label.is_file():
        raise FileNotFoundError(f"Missing label: {label}")

    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        v = line.split()
        if len(v) < 5:
            raise ValueError(f"Invalid label row: {label}: {line}")
        if int(float(v[0])) != 0:
            continue
        xc, yc, w, h = map(float, v[1:5])
        boxes.append([
            (xc - w/2) * width, (yc - h/2) * height,
            (xc + w/2) * width, (yc + h/2) * height
        ])
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    a = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
    b = np.maximum(0, boxes[:, 2]-boxes[:, 0]) * np.maximum(
        0, boxes[:, 3]-boxes[:, 1]
    )
    return inter / np.maximum(a + b - inter, 1e-12)


def predict(model, image, class_id):
    result = model.predict(
        source=str(image), imgsz=640, device=DEVICE,
        conf=PREDICT_CONF, iou=0.7, max_det=300,
        verbose=False, save=False
    )[0]

    rows = []
    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls in zip(boxes, scores, classes):
            if cls == class_id:
                rows.append([float(score), *box.tolist()])
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def evaluate_image(preds, gt):
    # Greedy confidence-ranked matching, same operating point as existing P/R/F1.
    candidates = [
        (i, p) for i, p in enumerate(preds) if p[0] >= CONF
    ]
    candidates.sort(key=lambda x: x[1][0], reverse=True)

    unmatched = set(range(len(gt)))
    matches, false_positives = [], []

    for pred_idx, pred in candidates:
        overlaps = iou(pred[1:5], gt)
        matched_gt = next(
            (int(j) for j in np.argsort(-overlaps)
             if overlaps[j] >= IOU_THRESHOLD and int(j) in unmatched),
            None
        )
        if matched_gt is None:
            best_iou = float(overlaps.max()) if len(overlaps) else 0.0
            false_positives.append((pred_idx, pred, best_iou))
        else:
            unmatched.remove(matched_gt)
            matches.append((pred_idx, pred, matched_gt))

    tp, fp, fn = len(matches), len(false_positives), len(unmatched)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return matches, false_positives, sorted(unmatched), precision, recall, f1


def main():
    for path in [DATA, WEATHER, *MODELS.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

    cfg = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    dataset_root = Path(cfg["path"])
    test_images = dataset_root / cfg["test"]
    test_labels = dataset_root / "test/labels"

    if not test_images.is_dir() or not test_labels.is_dir():
        raise FileNotFoundError(
            f"Check test paths:\n{test_images}\n{test_labels}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    models = {}
    class_ids = {}

    for name, path in MODELS.items():
        model = YOLO(str(path))
        model_names = names_dict(model.names)
        ids = [
            k for k, v in model_names.items()
            if str(v).strip().lower() in {"pothole", "potholes"}
        ]
        if len(ids) != 1:
            raise ValueError(f"Could not identify pothole class in {name}")
        models[name] = model
        class_ids[name] = ids[0]

    errors, metrics = [], []

    for condition in CONDITIONS:
        folder = WEATHER / condition
        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTS
        )
        if not images:
            raise RuntimeError(f"No images found in {folder}")

        print(f"\n{condition}: {len(images)} images")

        for count, image in enumerate(images, 1):
            label = test_labels / f"{image.stem}.txt"

            with Image.open(image) as im:
                width, height = im.size
            gt = read_gt(image, label, width, height)

            for model_name, model in models.items():
                preds = predict(model, image, class_ids[model_name])
                matches, fps, missed, precision, recall, f1 = evaluate_image(
                    preds, gt
                )

                metrics.append({
                    "condition": condition,
                    "image": image.name,
                    "model": model_name,
                    "ground_truth": len(gt),
                    "true_positives": len(matches),
                    "false_positives": len(fps),
                    "false_negatives": len(missed),
                    "precision": precision,
                    "recall": recall,
                    "F1": f1,
                })

                pred_boxes = np.asarray(
                    [p[1:5] for p in preds if p[0] >= CONF],
                    dtype=float
                ).reshape(-1, 4)

                for gi in missed:
                    best = iou(gt[gi], pred_boxes)
                    errors.append({
                        "condition": condition, "image": image.name,
                        "model": model_name, "error_type": "false_negative",
                        "confidence": "",
                        "best_iou": float(best.max()) if len(best) else 0.0,
                        "gt_index": gi,
                        "pred_x1": "", "pred_y1": "",
                        "pred_x2": "", "pred_y2": "",
                        "gt_x1": gt[gi, 0], "gt_y1": gt[gi, 1],
                        "gt_x2": gt[gi, 2], "gt_y2": gt[gi, 3],
                    })

                for _, pred, best_iou in fps:
                    errors.append({
                        "condition": condition, "image": image.name,
                        "model": model_name, "error_type": "false_positive",
                        "confidence": pred[0], "best_iou": best_iou,
                        "gt_index": "",
                        "pred_x1": pred[1], "pred_y1": pred[2],
                        "pred_x2": pred[3], "pred_y2": pred[4],
                        "gt_x1": "", "gt_y1": "", "gt_x2": "", "gt_y2": "",
                    })

            if count % 25 == 0 or count == len(images):
                print(f"  {count}/{len(images)} processed")

    def save_csv(path, rows):
        if not rows:
            print(f"No rows to write: {path}")
            return
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    save_csv(OUT / "per_image_detection_errors.csv", errors)
    save_csv(OUT / "per_image_metrics.csv", metrics)

    counts = {}
    for row in errors:
        key = (row["condition"], row["model"], row["error_type"])
        counts[key] = counts.get(key, 0) + 1

    summary = [
        {"condition": k[0], "model": k[1],
         "error_type": k[2], "count": v}
        for k, v in sorted(counts.items())
    ]
    save_csv(OUT / "error_count_summary.csv", summary)

    print("\nExperiment 4 complete.")
    print("Results:", OUT)
    print("P/R/F1 matching: confidence >= 0.25, IoU >= 0.50")


if __name__ == "__main__":
    main()
