
from pathlib import Path

root = Path(r"D:\research papper\rs")

targets = [
    root / "data" / "Multi-Weather Pothole Detection (MWPD)",
    root / "experiments" / "phase1",
]

for target in targets:
    print(f"\n{'=' * 70}")
    print("TARGET:", target)
    print("EXISTS:", target.exists())

    if not target.exists():
        continue

    for path in list(target.rglob("*"))[:150]:
        if path.is_dir():
            print("[DIR ]", path)
        elif (
            path.name.lower() in {
                "data.yaml", "normal.yaml", "best.pt",
                "condition_results.csv"
            }
            or path.suffix.lower() in {".py", ".yaml", ".yml"}
        ):
            print("[FILE]", path)
