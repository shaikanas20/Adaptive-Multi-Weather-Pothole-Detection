from pathlib import Path
import csv
import tempfile
import yaml
from ultralytics import YOLO

ROOT = Path("D:/research papper/rs")
MODEL_PATH = ROOT / "experiments/phase1/mwpd_yolov8m/weights/best.pt"
ORIGINAL_ROOT = ROOT / "data/Multi-Weather Pothole Detection (MWPD)/controlled_conditions"
ENHANCED_ROOT = ROOT / "data/Multi-Weather Pothole Detection (MWPD)/enhanced_conditions_advanced"
OUTPUT_CSV = ROOT / "experiments/phase1/environment/e2_enhancement_comparison.csv"

CONDITIONS = ["normal", "low_light", "fog", "rain", "shadow", "water_reflection"]
CLASS_NAMES = {
    #0: "Longitudinal Crack",
    #1: "Transverse Crack",
    #2: "Alligator Crack",
    0: "Pothole",
    #4: "Other / Repair Mark",
}
POTHOLE_ID = 0

IMGSZ = 640
BATCH = 16
DEVICE = 0
WORKERS = 4
CONF = 0.001
IOU = 0.7


def make_yaml(condition_root, yaml_path):
    data = {
        "path": str(condition_root.resolve()),
        "train": "images",
        "val": "images",
        "test": "images",
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def scalar(x):
    try:
        return float(x)
    except Exception:
        return None


def class_metric(box, class_id, name):
    try:
        indices = [int(x) for x in box.ap_class_index]
        if class_id not in indices:
            return None
        return scalar(getattr(box, name)[indices.index(class_id)])
    except Exception:
        return None


def evaluate(model, root, condition, pipeline):
    images = root / "images"
    labels = root / "labels"

    image_count = len(list(images.glob("*.jpg")))
    label_count = len(list(labels.glob("*.txt")))

    print(f"\n{'=' * 70}\n{pipeline.upper()} | {condition.upper()}\n{'=' * 70}")
    print(f"Images: {image_count} | Labels: {label_count}")

    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = Path(tmp) / "data.yaml"
        make_yaml(root, data_yaml)

        metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            conf=CONF,
            iou=IOU,
            verbose=False,
            plots=False,
        )

    box = metrics.box
    p = scalar(box.mp)
    r = scalar(box.mr)
    map50 = scalar(box.map50)
    map5095 = scalar(box.map)

    f1 = 2 * p * r / (p + r) if p is not None and r is not None and p + r else None

    pp = class_metric(box, POTHOLE_ID, "p")
    pr = class_metric(box, POTHOLE_ID, "r")
    pm50 = class_metric(box, POTHOLE_ID, "ap50")
    pm5095 = class_metric(box, POTHOLE_ID, "ap")
    pf1 = 2 * pp * pr / (pp + pr) if pp is not None and pr is not None and pp + pr else None

    speed = getattr(metrics, "speed", {}) or {}
    pre = scalar(speed.get("preprocess"))
    inf = scalar(speed.get("inference"))
    post = scalar(speed.get("postprocess"))
    total = sum(x for x in [pre, inf, post] if x is not None)

    print(f"Overall: P={p:.4f} R={r:.4f} mAP50={map50:.4f} mAP50-95={map5095:.4f} F1={f1:.4f}")
    print(f"Pothole: P={pp:.4f} R={pr:.4f} mAP50={pm50:.4f} mAP50-95={pm5095:.4f} F1={pf1:.4f}")
    print(f"Latency: {total:.3f} ms/image")

    return {
        "experiment": "E2",
        "pipeline": pipeline,
        "condition": condition,
        "images": image_count,
        "labels": label_count,
        "precision": p,
        "recall": r,
        "mAP50": map50,
        "mAP50_95": map5095,
        "F1": f1,
        "pothole_precision": pp,
        "pothole_recall": pr,
        "pothole_mAP50": pm50,
        "pothole_mAP50_95": pm5095,
        "pothole_F1": pf1,
        "preprocess_ms": pre,
        "inference_ms": inf,
        "postprocess_ms": post,
        "total_latency_ms": total,
        "model": MODEL_PATH.name,
        "imgsz": IMGSZ,
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not ORIGINAL_ROOT.exists():
        raise FileNotFoundError(f"Original dataset not found: {ORIGINAL_ROOT}")
    if not ENHANCED_ROOT.exists():
        raise FileNotFoundError(f"Enhanced dataset not found: {ENHANCED_ROOT}")

    print("=" * 70)
    print("E2 — CONTROLLED IMAGE ENHANCEMENT EVALUATION")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    results = []

    for condition in CONDITIONS:
        results.append(evaluate(model, ORIGINAL_ROOT / condition, condition, "original"))

    for condition in CONDITIONS:
        results.append(evaluate(model, ENHANCED_ROOT / condition, condition, "enhanced"))

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = list(results[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("E2 COMPLETE")
    print("=" * 70)
    print(f"Cases evaluated: {len(results)} / 12")
    print(f"CSV: {OUTPUT_CSV}")

    print("\nSummary:")
    print(f"{'Pipeline':<12}{'Condition':<20}{'mAP50':>10}{'mAP50-95':>12}{'Pothole mAP50':>16}")
    print("-" * 70)
    for x in results:
        print(
            f"{x['pipeline']:<12}{x['condition']:<20}"
            f"{x['mAP50']:>10.4f}{x['mAP50_95']:>12.4f}"
            f"{x['pothole_mAP50']:>16.4f}"
        )


if __name__ == "__main__":
    main()
