from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "phase1"
    / "environment"
    / "environment_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "phase1"
    / "environment"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_feature(df, feature):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df[feature],
        bins=50
    )

    plt.xlabel(feature)
    plt.ylabel("Number of Images")
    plt.title(f"Distribution of {feature}")

    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{feature}_distribution.png"

    plt.savefig(output_file, dpi=200)

    plt.close()

    print(f"Saved: {output_file}")


def main():

    df = pd.read_csv(INPUT_FILE)

    features = [
        "brightness",
        "contrast",
        "blur",
        "saturation",
        "dark_pixel_ratio"
    ]

    for feature in features:
        plot_feature(df, feature)

    print("\nVisualization completed.")


if __name__ == "__main__":
    main()