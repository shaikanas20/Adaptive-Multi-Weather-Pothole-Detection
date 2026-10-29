
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
ROOT = Path(r"D:\research papper\rs")

INPUT_CSV = (
    ROOT
    / "experiments"
    / "phase1"
    / "experiment4_per_image_errors"
    / "per_image_metrics.csv"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "phase1"
    / "experiment5_paired_bootstrap"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 10000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95

CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection",
]

MODELS = ["yolov8n", "yolov8m"]
METRICS = ["recall", "F1"]


def image_stem(value):
    """Extract a filename stem regardless of Windows/Linux separators."""
    return str(value).replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]


def paired_bootstrap(differences, rng):
    """
    Bootstrap the mean paired difference.

    Positive differences favor the comparison group.
    Negative differences favor the reference group.
    """
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]

    n = len(differences)

    if n == 0:
        return None

    observed_difference = float(np.mean(differences))

    # Resample paired image-level differences, not individual
    # detections or independent condition rows.
    boot_means = np.empty(N_BOOTSTRAP)

    for start in range(0, N_BOOTSTRAP, 500):
        batch_size = min(500, N_BOOTSTRAP - start)
        indices = rng.integers(
            0, n, size=(batch_size, n)
        )
        boot_means[start:start + batch_size] = (
            differences[indices].mean(axis=1)
        )

    alpha = 1 - CONFIDENCE_LEVEL

    lower = float(np.quantile(boot_means, alpha / 2))
    upper = float(np.quantile(boot_means, 1 - alpha / 2))

    # Descriptive bootstrap probability, NOT a formal p-value.
    prob_positive = float(np.mean(boot_means > 0))

    return {
        "n_pairs": n,
        "mean_difference": observed_difference,
        "ci_lower_95": lower,
        "ci_upper_95": upper,
        "bootstrap_prob_difference_gt_zero": prob_positive,
    }


# ---------------------------------------------------------
# Load and validate data
# ---------------------------------------------------------
if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

required = {
    "condition", "image", "model", "recall", "F1"
}
missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}\n"
        f"Available columns: {list(df.columns)}"
    )

df["condition"] = df["condition"].astype(str).str.strip().str.lower()
df["model"] = df["model"].astype(str).str.strip().str.lower()
df["image_id"] = df["image"].map(image_stem)

for metric in METRICS:
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

df = df[
    df["condition"].isin(CONDITIONS)
    & df["model"].isin(MODELS)
].copy()

# Each source image should have exactly one row per condition/model.
key = ["image_id", "condition", "model"]

duplicates = df.duplicated(key, keep=False)
if duplicates.any():
    print("WARNING: Duplicate image-condition-model rows found.")
    print(df.loc[duplicates, key].head(20).to_string(index=False))
    raise ValueError(
        "Resolve duplicate rows before running paired bootstrap."
    )

print("Rows loaded:", len(df))
print("Unique source images:", df["image_id"].nunique())
print("Conditions:", sorted(df["condition"].unique()))
print("Models:", sorted(df["model"].unique()))

rng = np.random.default_rng(RANDOM_SEED)
results = []

# ---------------------------------------------------------
# Analysis A:
# YOLOv8n minus YOLOv8m within each condition.
# Positive difference favors YOLOv8n.
# ---------------------------------------------------------
for condition in CONDITIONS:
    subset = df[df["condition"] == condition]

    for metric in METRICS:
        wide = subset.pivot(
            index="image_id",
            columns="model",
            values=metric,
        )

        if not all(m in wide.columns for m in MODELS):
            print(f"Skipping incomplete model pair: {condition}, {metric}")
            continue

        paired = wide[MODELS].dropna()
        differences = paired["yolov8n"] - paired["yolov8m"]

        stats = paired_bootstrap(differences, rng)
        if stats is None:
            continue

        results.append({
            "comparison_type": "model_comparison",
            "condition": condition,
            "model": "yolov8n_minus_yolov8m",
            "metric": metric,
            "mean_reference": paired["yolov8m"].mean(),
            "mean_comparison": paired["yolov8n"].mean(),
            **stats,
        })

# ---------------------------------------------------------
# Analysis B:
# Each transformed condition minus normal for each model.
# Positive difference means the metric is higher in the
# transformed condition; negative means it is lower.
# ---------------------------------------------------------
for model in MODELS:
    subset = df[df["model"] == model]

    for condition in CONDITIONS:
        if condition == "normal":
            continue

        for metric in METRICS:
            wide = subset.pivot(
                index="image_id",
                columns="condition",
                values=metric,
            )

            if "normal" not in wide.columns or condition not in wide.columns:
                print(
                    f"Skipping incomplete condition pair: "
                    f"{model}, {condition}, {metric}"
                )
                continue

            paired = wide[["normal", condition]].dropna()
            differences = paired[condition] - paired["normal"]

            stats = paired_bootstrap(differences, rng)
            if stats is None:
                continue

            results.append({
                "comparison_type": "condition_vs_normal",
                "condition": condition,
                "model": model,
                "metric": metric,
                "mean_reference": paired["normal"].mean(),
                "mean_comparison": paired[condition].mean(),
                **stats,
            })

# ---------------------------------------------------------
# Save outputs
# ---------------------------------------------------------
result_df = pd.DataFrame(results)

if result_df.empty:
    raise ValueError("No valid paired comparisons were produced.")

result_df["ci_excludes_zero"] = (
    (result_df["ci_lower_95"] > 0)
    | (result_df["ci_upper_95"] < 0)
)

output_csv = OUTPUT_DIR / "paired_bootstrap_results.csv"
result_df.to_csv(output_csv, index=False)

print("\nExperiment 5 complete.")
print("Results saved to:", output_csv)

print("\nSummary:")
columns = [
    "comparison_type",
    "condition",
    "model",
    "metric",
    "n_pairs",
    "mean_difference",
    "ci_lower_95",
    "ci_upper_95",
    "ci_excludes_zero",
]

print(
    result_df[columns]
    .round(4)
    .to_string(index=False)
)
