
from pathlib import Path
from datetime import datetime
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(r"D:\research papper\rs")

DATASET = ROOT / (
    r"data\Multi-Weather Pothole Detection (MWPD)"
    r"\Multi-Weather Pothole Detection (MWPD)"
    r"\MWPD"
)

CSV_PATH = (
    ROOT / "experiments" / "phase1" / "severity_index"
    / "audit_20261029_135521" / "per_box_geometry.csv"
)

IMAGE_DIR = DATASET / "test" / "images"
OUTPUT_ROOT = ROOT / "experiments" / "phase1" / "severity_index"

# Each source image must be counted only once.
df = pd.read_csv(CSV_PATH)
df = df[df["condition"] == "normal"].copy()

df = df.sort_values("bbox_area_ratio").reset_index(drop=True)

smallest = df.head(10).copy()
smallest["selection"] = "smallest"

largest = df.tail(10).sort_values(
    "bbox_area_ratio", ascending=False
).copy()
largest["selection"] = "largest"

selected = pd.concat([largest, smallest], ignore_index=True)

run_dir = OUTPUT_ROOT / (
    "visual_inspection_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
run_dir.mkdir(parents=True, exist_ok=False)

manifest = []

for rank, row in selected.iterrows():
    image_path = IMAGE_DIR / row["image"]

    if not image_path.exists():
        manifest.append({
            "image": row["image"],
            "selection": row["selection"],
            "status": "IMAGE_NOT_FOUND",
        })
        continue

    with Image.open(image_path) as im:
        image = im.convert("RGB")

    W, H = image.size

    # Coordinates are normalized to the image.
    x1 = int(round(row["x1_norm"] * W))
    y1 = int(round(row["y1_norm"] * H))
    x2 = int(round(row["x2_norm"] * W))
    y2 = int(round(row["y2_norm"] * H))

    draw = ImageDraw.Draw(image)
    draw.rectangle((x1, y1, x2, y2), outline="red", width=4)

    filename = (
        f"{row['selection']}_{rank + 1:02d}_"
        f"area_{row['bbox_area_ratio']:.4f}_"
        f"{Path(row['image']).stem}.jpg"
    )
    image.save(run_dir / filename, quality=95)

    manifest.append({
        "image": row["image"],
        "selection": row["selection"],
        "bbox_area_ratio": row["bbox_area_ratio"],
        "bbox_width_norm": row["bbox_width_norm"],
        "bbox_height_norm": row["bbox_height_norm"],
        "bbox_aspect_ratio": row["bbox_aspect_ratio"],
        "saved_image": filename,
        "status": "SAVED",
    })

pd.DataFrame(manifest).to_csv(
    run_dir / "inspection_manifest.csv", index=False
)

print("Visual inspection images saved.")
print(f"Output folder: {run_dir}")
print(f"Selected boxes: {len(selected)}")
print(f"Images saved: {sum(x['status'] == 'SAVED' for x in manifest)}")
