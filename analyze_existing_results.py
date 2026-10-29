
from pathlib import Path
import pandas as pd

ROOT = Path(r"D:\research papper\rs")

INPUT = (
    ROOT / "experiments" / "phase1"
    / "adaptive_model_selection"
    / "heldout_test_results_by_condition.csv"
)

OUTPUT = ROOT / "experiments" / "phase1" / "experiment2_analysis"
OUTPUT.mkdir(parents=True, exist_ok=True)

EXPECTED_CONDITIONS = [
    "normal", "low_light", "fog",
    "rain", "shadow", "water_reflection"
]

EXPECTED_STRATEGIES = [
    "fixed_yolov8n",
    "fixed_yolov8m",
    "adaptive_yolov8n_to_yolov8m"
]

df = pd.read_csv(INPUT)

required = {
    "condition", "strategy", "precision", "recall", "F1",
    "mAP50", "mAP50_95", "mean_latency_ms", "escalation_rate"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df.duplicated(["condition", "strategy"]).any():
    raise ValueError("Duplicate condition/strategy rows found.")

if len(df) != 18:
    raise ValueError(f"Expected 18 rows, found {len(df)}.")

if set(df["condition"]) != set(EXPECTED_CONDITIONS):
    raise ValueError("Unexpected or missing weather conditions.")

if set(df["strategy"]) != set(EXPECTED_STRATEGIES):
    raise ValueError("Unexpected or missing strategies.")

if df[list(required - {"condition", "strategy"})].isna().any().any():
    raise ValueError("Missing metric values found.")

# --------------------------------------------------
# 1. Overall descriptive summary
# Equal weight is given to each weather condition.
# These are means of condition-level metrics, not pooled mAP.
# --------------------------------------------------
metrics = [
    "precision", "recall", "F1",
    "mAP50", "mAP50_95",
    "mean_latency_ms", "escalation_rate"
]

summary = (
    df.groupby("strategy")[metrics]
    .mean()
    .reindex(EXPECTED_STRATEGIES)
)

summary.to_csv(OUTPUT / "strategy_summary.csv", float_format="%.6f")

print("\n=== Mean across six weather conditions ===")
print(summary.round(4).to_string())

# --------------------------------------------------
# 2. Condition-wise accuracy comparison
# Differences are in percentage points after multiplying
# metric differences by 100.
# --------------------------------------------------
wide = df.pivot(index="condition", columns="strategy")

condition_rows = []

for condition in EXPECTED_CONDITIONS:
    row = {"condition": condition}

    for metric in ["mAP50", "mAP50_95", "recall", "mean_latency_ms"]:
        values = wide.loc[condition, metric]

        n = values["fixed_yolov8n"]
        m = values["fixed_yolov8m"]
        adaptive = values["adaptive_yolov8n_to_yolov8m"]

        multiplier = 100 if metric != "mean_latency_ms" else 1
        suffix = "_pp" if multiplier == 100 else "_ms"

        row[f"adaptive_minus_n_{metric}{suffix}"] = (
            adaptive - n
        ) * multiplier

        row[f"m_minus_n_{metric}{suffix}"] = (
            m - n
        ) * multiplier

    condition_rows.append(row)

condition_analysis = pd.DataFrame(condition_rows)
condition_analysis.to_csv(
    OUTPUT / "condition_differences.csv",
    index=False,
    float_format="%.6f"
)

print("\n=== Adaptive minus fixed YOLOv8n ===")
print(condition_analysis.to_string(index=False))

# --------------------------------------------------
# 3. Escalation summary
# --------------------------------------------------
adaptive = df[df["strategy"] == "adaptive_yolov8n_to_yolov8m"]

print("\n=== Adaptive escalation ===")
for _, row in adaptive.iterrows():
    count = row.get("escalated_images", None)
    print(
        f"{row['condition']:18s} "
        f"rate={row['escalation_rate']:.2%} "
        f"images={count if pd.notna(count) else 'not recorded'}"
    )

print("\nSaved analysis files to:", OUTPUT)
