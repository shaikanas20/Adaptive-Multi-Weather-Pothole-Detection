
from pathlib import Path
from collections import defaultdict

ROOT = Path(
    r"D:\research papper\rs\data\Multi-Weather Pothole Detection (MWPD)"
    r"\Multi-Weather Pothole Detection (MWPD)\MWPD"
)

WEATHER_ROOT = Path(
    r"D:\research papper\rs\data\Multi-Weather Pothole Detection (MWPD)"
    r"\controlled_conditions"
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_images(folder):
    if not folder.exists():
        return []
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]


def index_by_stem(paths):
    index = defaultdict(list)
    for p in paths:
        index[p.stem.lower()].append(p)
    return index


train = get_images(ROOT / "train" / "images")
valid = get_images(ROOT / "valid" / "images")
test = get_images(ROOT / "test" / "images")

train_idx = index_by_stem(train)
valid_idx = index_by_stem(valid)
test_idx = index_by_stem(test)

print("Test images:", len(test))

for condition in sorted(WEATHER_ROOT.iterdir()):
    if not condition.is_dir():
        continue

    weather_images = get_images(condition)
    matched_test = 0
    matched_train = 0
    matched_valid = 0
    unmatched = []

    for img in weather_images:
        stem = img.stem.lower()

        if stem in test_idx:
            matched_test += 1
        elif stem in train_idx:
            matched_train += 1
        elif stem in valid_idx:
            matched_valid += 1
        else:
            unmatched.append(img.name)

    print(f"\nCondition: {condition.name}")
    print("Weather images:", len(weather_images))
    print("Filename matches in test:", matched_test)
    print("Filename matches in train:", matched_train)
    print("Filename matches in valid:", matched_valid)
    print("Unmatched filenames:", len(unmatched))
    print("Examples:", unmatched[:5])

print(
    "\nWARNING: These are filename-based matches only. "
    "Renamed images need source metadata or visual/hash matching."
)
