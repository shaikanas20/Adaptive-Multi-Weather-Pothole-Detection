
from pathlib import Path
from datetime import datetime
import csv
import json
import math
import statistics
import yaml
from PIL import Image

ROOT = Path(r"D:\research papper\rs")

DATA_YAML = (
    ROOT / "data" / "Multi-Weather Pothole Detection (MWPD)"
    / "Multi-Weather Pothole Detection (MWPD)" / "MWPD" / "data.yaml"
)

CONDITIONS_DIR = (
    ROOT / "data" / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions_advanced"
)

OUTPUT_ROOT = (
    ROOT / "experiments" / "phase1" / "severity_index"
)

CONDITIONS = [
    "normal", "low_light", "fog", "rain", "shadow", "water_reflection"
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_labels(path):
    """Read YOLO labels and return boxes plus validation issues."""
    boxes = []
    issues = []

    if not path.exists():
        return boxes, ["missing_label_file"]

    for line_no, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5:
            issues.append(f"line_{line_no}_invalid_field_count")
            continue

        try:
            class_id = int(float(values[0]))
            xc, yc, w, h = map(float, values[1:])
        except ValueError:
            issues.append(f"line_{line_no}_invalid_numeric_value")
            continue

        if not all(math.isfinite(v) for v in (xc, yc, w, h)):
            issues.append(f"line_{line_no}_non_finite_value")
            continue

        if w <= 0 or h <= 0:
            issues.append(f"line_{line_no}_non_positive_size")
            continue

        if not all(0 <= v <= 1 for v in (xc, yc, w, h)):
            issues.append(f"line_{line_no}_coordinate_out_of_range")

        x1, y1 = xc - w / 2, yc - h / 2
        x2, y2 = xc + w / 2, yc + h / 2

        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            issues.append(f"line_{line_no}_box_outside_image")

        boxes.append({
            "label_line": line_no,
            "class_id": class_id,
            "center_x_norm": xc,
            "center_y_norm": yc,
            "width_norm": w,
            "height_norm": h,
            "area_ratio": w * h,
            "aspect_ratio": w / h,
            "x1_norm": x1,
            "y1_norm": y1,
            "x2_norm": x2,
            "y2_norm": y2,
        })

    return boxes, issues


def mean_or_none(values):
    return statistics.mean(values) if values else None


def median_or_none(values):
    return statistics.median(values) if values else None


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)

    if low == high:
        return values[low]

    return values[low] * (high - position) + values[high] * (
        position - low
    )


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    config = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path", DATA_YAML.parent))

    if not dataset_root.is_absolute():
        dataset_root = DATA_YAML.parent / dataset_root

    test_images_dir = dataset_root / config.get("test", "test/images")
    label_dir = test_images_dir.parent / "labels"

    if not label_dir.is_dir():
        raise FileNotFoundError(f"Canonical test labels not found: {label_dir}")

    run_id = datetime.now().strftime("audit_%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    image_rows = []
    box_rows = []
    condition_rows = []
    issue_rows = []

    for condition in CONDITIONS:
        image_dir = CONDITIONS_DIR / condition / "images"

        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        condition_boxes = []
        condition_areas = []
        missing_labels = 0

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"

            try:
                with Image.open(image_path) as im:
                    image_w, image_h = im.size
            except Exception as exc:
                issue_rows.append({
                    "condition": condition,
                    "image": image_path.name,
                    "issue": f"image_read_error:{type(exc).__name__}",
                })
                continue

            boxes, issues = read_labels(label_path)

            if not label_path.exists():
                missing_labels += 1

            for issue in issues:
                issue_rows.append({
                    "condition": condition,
                    "image": image_path.name,
                    "issue": issue,
                })

            image_rows.append({
                "condition": condition,
                "image": image_path.name,
                "image_width_px": image_w,
                "image_height_px": image_h,
                "label_exists": label_path.exists(),
                "box_count": len(boxes),
                "issue_count": len(issues),
            })

            for index, box in enumerate(boxes, 1):
                width_px = box["width_norm"] * image_w
                height_px = box["height_norm"] * image_h

                row = {
                    "condition": condition,
                    "image": image_path.name,
                    "box_index": index,
                    "class_id": box["class_id"],
                    "image_width_px": image_w,
                    "image_height_px": image_h,
                    "bbox_width_px": width_px,
                    "bbox_height_px": height_px,
                    "bbox_area_px": width_px * height_px,
                    "bbox_area_ratio": box["area_ratio"],
                    "bbox_width_norm": box["width_norm"],
                    "bbox_height_norm": box["height_norm"],
                    "bbox_aspect_ratio": box["aspect_ratio"],
                    "center_x_norm": box["center_x_norm"],
                    "center_y_norm": box["center_y_norm"],
                    "x1_norm": box["x1_norm"],
                    "y1_norm": box["y1_norm"],
                    "x2_norm": box["x2_norm"],
                    "y2_norm": box["y2_norm"],
                }

                box_rows.append(row)
                condition_boxes.append(row)
                condition_areas.append(box["area_ratio"])

        condition_rows.append({
            "condition": condition,
            "image_count": len(images),
            "missing_label_files": missing_labels,
            "parsed_boxes": len(condition_boxes),
            "mean_bbox_area_ratio": mean_or_none(condition_areas),
            "median_bbox_area_ratio": median_or_none(condition_areas),
            "p90_bbox_area_ratio": percentile(condition_areas, 0.90),
        })

    write_csv(out_dir / "per_image_label_audit.csv", image_rows)
    write_csv(out_dir / "per_box_geometry.csv", box_rows)
    write_csv(out_dir / "condition_geometry_summary.csv", condition_rows)
    write_csv(out_dir / "label_issues.csv", issue_rows)

    all_areas = [row["bbox_area_ratio"] for row in box_rows]

    summary = {
        "dataset_yaml": str(DATA_YAML),
        "condition_images_root": str(CONDITIONS_DIR),
        "canonical_test_labels": str(label_dir),
        "images_audited": len(image_rows),
        "boxes_parsed_across_condition_views": len(box_rows),
        "issue_records": len(issue_rows),
        "overall_bbox_area_ratio_mean": mean_or_none(all_areas),
        "overall_bbox_area_ratio_median": median_or_none(all_areas),
        "overall_bbox_area_ratio_p90": percentile(all_areas, 0.90),
        "scientific_limitation": (
            "Bounding-box geometry is a visual proxy, not physical pothole "
            "severity. These outputs do not establish validated severity classes."
        ),
    }

    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSeverity Index audit complete.")
    print(f"Output directory: {out_dir}")
    print(json.dumps(summary, indent=2))
    print("\nCondition summary:")
    for row in condition_rows:
        print(row)


if __name__ == "__main__":
    main()
