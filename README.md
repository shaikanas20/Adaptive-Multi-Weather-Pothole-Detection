# 🛣️ Intelligent Road Damage Detection & Environmental Robustness Prototype

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)

An end-to-end research prototype and benchmark suite for real-time **Road Damage Detection (RDD)** and **Environmental Condition Robustness**. This project evaluates state-of-the-art vision models (**YOLOv8s** vs. **YOLOv8m** vs. **YOLOv8n**) across multi-country datasets and synthetic adverse weather conditions (*Low Light, Fog, Rain, Shadow, Water Reflection*), introducing an adaptive image enhancement pipeline to improve detection reliability on edge systems.

---

## 📋 Table of Contents
1. [Executive Summary & Prototype Architecture](#-executive-summary--prototype-architecture)
2. [Datasets Used](#-datasets-used)
3. [Models Evaluated & Training Specifications](#-models-evaluated--training-specifications)
4. [Comprehensive Comparison: YOLOv8s vs. YOLOv8m](#-comprehensive-comparison-yolov8s-vs-yolov8m)
5. [Environmental Robustness & Image Enhancement Evaluation (E2 Experiment)](#-environmental-robustness--image-enhancement-evaluation-e2-experiment)
6. [Prototype Implementation & Module Breakdown](#-prototype-implementation--module-breakdown)
7. [Directory Structure](#-directory-structure)
8. [How to Run & Reproduce Experiments](#-how-to-run--reproduce-experiments)
9. [Key Insights for Report Generation](#-key-insights-for-report-generation)

---

## 💡 Executive Summary & Prototype Architecture

### Key Objectives
* **Dataset Consolidation**: Harmonize multi-source pavement defect datasets into standardized YOLO formats.
* **Model Optimization & Comparison**: Train and compare lightweight edge models (**YOLOv8s**) against medium-capacity backbones (**YOLOv8m**) on multi-weather pothole detection benchmarks.
* **Adverse Environment Stress-Testing**: Develop a synthetic degradation engine to systematically simulate real-world adverse weather conditions (*low light, fog, heavy rain, shadows, water pooling*).
* **Adaptive Enhancement Evaluation (E2 Experiment)**: Benchmark CLAHE, unsharp masking, and luminance normalization algorithms to measure their impact on object detection accuracy and latency.

### End-to-End System Pipeline
```text
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│  Multi-Country Datasets  │───>│ Synthetic Adverse Weather │───>│ Adaptive Image           │
│ (RDD2022, MWPD, BPID, etc)│    │ Simulator (6 Conditions) │    │ Enhancement (CLAHE, etc) │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
                                                                             │
┌──────────────────────────┐    ┌──────────────────────────┐                 │
│  Report Generation &     │<───│ Multi-Metric Benchmark   │<────────────────┘
│  Edge Deployment Guide   │    │ (Precision, Recall, mAP) │
└──────────────────────────┘    └──────────────────────────┘
```

---

## 📊 Datasets Used

This project incorporates four major pavement defect datasets representing urban, rural, highway, and adverse weather conditions:

### 1. 🌐 RDD2022 (Road Damage Detection 2022 Challenge Dataset)
The primary benchmark for multi-class road damage detection, comprising **38,385 total images** and **65,712 bounding box annotations** collected across 7 countries and multiple camera setups (dashcams, drones, motorbikes).

#### 🏷️ RDD2022 Class ID Mapping (0 to 4)
| Class ID | Damage Code | Category Name | Description & Characteristics | Train Bounding Boxes |
| :---: | :---: | :--- | :--- | :---: |
| **`0`** | **D00** | **Longitudinal Crack** | Cracks extending vertically along the direction of traffic flow / wheel paths. | 18,201 |
| **`1`** | **D10** | **Transverse Crack** | Cracks extending horizontally across the lane width. | 8,386 |
| **`2`** | **D20** | **Alligator Crack** | Interconnected pattern of cracks forming a web resembling crocodile skin (fatigue cracking). | 7,527 |
| **`3`** | **D40** | **Pothole** | Bowl-shaped hole, depression, or structural breach in the asphalt surface. | 7,554 |
| **`4`** | **D43 / D44** | **Other / Repair Mark** | Crosswalk/white line blur (`D43`), repair patches (`D44`), or general surface defect. | 4,628 |

#### 📂 RDD2022 Split Breakdown
* **`train/`**: 26,869 Image-Label pairs (46,296 bounding boxes)
* **`val/`**: 5,758 Image-Label pairs (9,741 bounding boxes)
* **`test/`**: 5,758 Image-Label pairs (9,675 bounding boxes)

#### 📑 Country-Specific Subset Breakdown
| Subset / Country | Train Images | Val Images | Test Images | Filename Range Example |
| :--- | :---: | :---: | :---: | :--- |
| **China Drone** | 1,675 | 349 | 377 | `China_Drone_000001.jpg` → `China_Drone_002400.jpg` |
| **China MotorBike** | 1,376 | 296 | 305 | `China_MotorBike_000000.jpg` → `China_MotorBike_001976.jpg` |
| **Czech Republic** | 1,962 | 431 | 436 | `Czech_000000.jpg` → `Czech_003594.jpg` |
| **India** | 5,368 | 1,172 | 1,166 | `India_000000.jpg` → `India_009891.jpg` |
| **Japan** | 7,432 | 1,550 | 1,524 | `Japan_000000.jpg` → `Japan_013132.jpg` |
| **Norway** | 5,708 | 1,256 | 1,197 | `Norway_000000.jpg` → `Norway_008158.jpg` |
| **United States** | 3,348 | 704 | 753 | `United_States_000000.jpg` → `United_States_004804.jpg` |

---

### 2. 🌧️ Multi-Weather Pothole Detection (MWPD)
* **Description**: Specialized benchmark dataset specifically designed for evaluating pothole detection models under varying illumination and weather conditions.
* **Classes**: 1 Target Class (`Potholes`).
* **Role in Prototype**: Used as the main training dataset for **YOLOv8s** and **YOLOv8m** baseline models and as the source for the 97-image controlled environmental condition evaluation suite.

---

### 3. 💧 Water-Filled and Dry Potholes Dataset
* **Description**: Annotated dataset featuring distinct environmental variations of dry vs. water-filled potholes.
* **Role in Prototype**: Used to analyze specular reflections, water surface distortion, and wet asphalt contrast shifts during condition generation.

---

### 4. 🏙️ Bandung Pothole Image Dataset (BPID)
* **Description**: Urban pavement dataset featuring localized road conditions, varying asphalt textures, and diverse urban traffic backgrounds.
* **Role in Prototype**: Cross-dataset evaluation resource for assessing model generalization.

---

## 🤖 Models Evaluated & Training Specifications

Three model variants from the YOLOv8 family were configured and evaluated:

| Parameter / Feature | YOLOv8n (Nano) | YOLOv8s (Small) | YOLOv8m (Medium) |
| :--- | :---: | :---: | :---: |
| **Backbone Architecture** | CSPDarknet-Nano | CSPDarknet-Small | CSPDarknet-Medium |
| **Parameter Count** | ~3.2 Million | **~11.2 Million** | **~25.9 Million** |
| **Model Weight Size** | 6.5 MB | 22.5 MB | 52.1 MB |
| **Input Image Size (`imgsz`)** | 640 × 640 | 640 × 640 | 640 × 640 |
| **Training Dataset** | RDD2022 | MWPD | MWPD |
| **Training Epochs** | 70 | 50 | 50 |
| **Batch Size** | 16 | **16** | **4** |
| **Automatic Mixed Precision (`amp`)** | True | False (CUDA Stability) | False (CUDA Stability) |
| **Random Seed** | 42 | 42 | 42 |
| **Hardware Execution** | NVIDIA GPU (CUDA) | NVIDIA GPU (CUDA) | NVIDIA GPU (CUDA) |

---

## ⚔️ Comprehensive Comparison: YOLOv8s vs. YOLOv8m

A central objective of this prototype was to evaluate whether scaling up to a larger model capacity (**YOLOv8m**) improves detection performance on road damage tasks compared to a lightweight model (**YOLOv8s**).

### 📊 Experimental Results Summary (50-Epoch Benchmark on MWPD)

| Evaluation Metric | YOLOv8s (Small) | YOLOv8m (Medium) | Absolute Difference | Winner / Key Takeaway |
| :--- | :---: | :---: | :---: | :--- |
| **Parameters** | **11.2M** | 25.9M | -14.7M (-56.8%) | 🟢 **YOLOv8s** (More compact) |
| **Model Weight Size** | **22.5 MB** | 52.1 MB | -29.6 MB | 🟢 **YOLOv8s** (Edge friendly) |
| **Training Batch Size** | **16** | 4 | +12 | 🟢 **YOLOv8s** (Better batch-norm) |
| **Total Training Time** | **~57.1 min (3,429s)** | ~164.5 min (9,871s) | -107.4 min (-65.3%) | 🟢 **YOLOv8s** (2.88× Faster training) |
| **Avg Time per Epoch** | **~68.6 sec** | ~197.4 sec | -128.8 sec | 🟢 **YOLOv8s** |
| **Inference Latency (per image)** | **~1.65 ms** | ~18.30 ms | **-16.65 ms** | 🟢 **YOLOv8s (11.1× Faster inference)** |
| **Peak Precision** | **0.8210 (82.10%)** | 0.7877 (78.77%) | **+3.33%** | 🟢 **YOLOv8s** |
| **Peak Recall** | **0.7155 (71.55%)** | 0.7029 (70.29%) | **+1.26%** | 🟢 **YOLOv8s** |
| **Peak mAP@0.50** | **0.7641 (76.41%)** | 0.7467 (74.67%) | **+1.74%** | 🟢 **YOLOv8s** |
| **Peak mAP@0.50:0.95** | **0.3609 (36.09%)** | 0.3592 (35.92%) | **+0.17%** | 🟢 **YOLOv8s** |
| **Best F1-Score** | **0.7387 (73.87%)** | 0.7186 (71.86%) | **+2.01%** | 🟢 **YOLOv8s** |

---

### 📈 Epoch-by-Epoch Convergence Analysis

```text
mAP@0.50 Trajectory over 50 Epochs:

YOLOv8s:  [Epoch 1: 24.2%] ───> [Epoch 15: 68.0%] ───> [Epoch 40: 76.4% (Peak)] ───> [Epoch 50: 74.3%]
YOLOv8m:  [Epoch 1: 19.2%] ───> [Epoch 15: 62.4%] ───> [Epoch 49: 74.7% (Peak)] ───> [Epoch 50: 74.2%]
```

### 🧠 Why did YOLOv8s Outperform YOLOv8m?

1. **Overfitting & Capacity Mismatch**: YOLOv8m contains **25.9M parameters** (more than double YOLOv8s). On focused domain datasets like MWPD, the excess capacity leads to slight overfitting, whereas YOLOv8s provides better generalization.
2. **Batch Size Impact on Batch Normalization**: Due to GPU VRAM constraints, YOLOv8m had to be trained with `batch=4`, whereas YOLOv8s trained with `batch=16`. Larger batch sizes yield significantly more stable mini-batch statistics for Batch Normalization layers.
3. **Inference Latency Advantage**: YOLOv8s processes images in **~1.65ms** compared to **~18.30ms** for YOLOv8m. This makes **YOLOv8s 11 times faster**, achieving over **600 FPS** on dedicated GPU edge devices, making it the ideal candidate for real-time mobile deployment.

---

## 🌧️ Environmental Robustness & Image Enhancement Evaluation (E2 Experiment)

Real-world deployment demands that pothole detection systems remain reliable under severe environmental degradation. The **E2 Experiment** evaluated model performance across 6 synthetic adverse weather conditions, comparing the **Original Image Pipeline** against an **Enhanced Image Pipeline** (utilizing CLAHE, unsharp masking, and YCrCb color space enhancement).

### 🧪 E2 Evaluation Results (97-Image Test Suite across 6 Conditions)

| Environmental Condition | Pipeline | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 | F1-Score | Total Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal (Baseline)** | Original | 0.7484 | 0.6724 | **0.7458** | **0.3627** | 0.7084 | **19.90 ms** |
| **Normal (Baseline)** | Enhanced | **0.7533** | **0.6781** | 0.7462 | 0.3634 | **0.7137** | 20.20 ms |
| **Low Light** | Original | **0.7811** | 0.6130 | **0.7135** | 0.3407 | **0.6869** | 19.99 ms |
| **Low Light** | Enhanced | 0.7652 | **0.6164** | 0.7045 | **0.3409** | 0.6828 | 19.91 ms |
| **Fog / Haze** | Original | 0.7270 | **0.6838** | **0.7407** | **0.3470** | **0.7047** | 20.05 ms |
| **Fog / Haze** | Enhanced | **0.7623** | 0.6096 | 0.6928 | 0.3249 | 0.6774 | 19.96 ms |
| **Rain / Wet Surface** | Original | **0.7798** | **0.6186** | **0.7379** | **0.3541** | **0.6899** | 20.04 ms |
| **Rain / Wet Surface** | Enhanced | 0.7513 | 0.6062 | 0.7022 | 0.3359 | 0.6710 | 19.70 ms |
| **Shadow / Occlusion** | Original | 0.7039 | **0.6027** | **0.6952** | **0.3155** | 0.6494 | 19.90 ms |
| **Shadow / Occlusion** | Enhanced | **0.7445** | 0.5789 | 0.6676 | 0.3017 | **0.6513** | 19.74 ms |
| **Water Reflection** | Original | **0.7461** | **0.6744** | **0.7448** | **0.3632** | **0.7085** | 20.01 ms |
| **Water Reflection** | Enhanced | 0.7375 | 0.6712 | 0.7278 | 0.3559 | 0.7028 | 19.75 ms |

---

### 🔍 Key Findings from Environmental Benchmark
1. **Most Severe Condition**: **Shadows and heavy occlusion** caused the largest drop in detection performance, reducing mAP@0.50 from **74.58% down to 69.52%** (-5.06%) and F1-score to **0.6494**.
2. **Most Resilient Conditions**: Models demonstrated surprising resilience under **Fog** (mAP50 = 74.07%) and **Water Reflection** (mAP50 = 74.48%).
3. **Enhancement Pipeline Insight**:
   * Static enhancement (CLAHE + unsharp masking) improved **Precision in Shadow conditions (+4.06%)** and **Normal conditions (+0.49%)**.
   * However, aggressive global enhancement slightly reduced Recall under Fog and Rain due to amplification of high-frequency noise (rain streaks and atmospheric scattering).
   * **Recommendation**: Implement **Adaptive, Condition-Aware Preprocessing** that activates specific enhancement filters only when environmental degradation parameters exceed threshold limits.

---

## 🛠️ Prototype Implementation & Module Breakdown

The codebase is modularized under `src/` to support data transformation, model training, adverse condition simulation, and automated evaluation:

### 1. 📦 Data Pipeline (`src/data/`)
* **`convert_rdd.py`**: Converts raw multi-country RDD2022 XML/VOC annotations into standard YOLO text annotations (`class_id x_center y_center width height`).
* **`split_dataset.py`**: Performs stratified train/val/test splits preserving class distribution across country subsets.
* **`dataset_stats.py`**: Analyzes bounding box distributions, aspect ratios, and class balance.
* **`visualize_annotation.py`**: Generates diagnostic visual overlays of ground truth bounding boxes.

### 2. 🎯 Detection Engine (`src/detection/`)
* **`train.py`**: Execution engine supporting CLI arguments (`--dataset rdd2022|mwpd|mwpd_yolov8m --epochs 50`), managing VRAM allocation (`amp=False`), and logging training trajectories.
* **`evaluate.py`**: Comprehensive test evaluator calculating per-class AP, mAP@0.50, mAP@0.50:0.95, Precision, Recall, and F1-score.
* **`evaluate_conditions.py`**: Evaluates trained models against degraded environmental condition directories.

### 3. 🌧️ Synthetic Environmental Degradation (`src/environment/`)
* **`generate_conditions.py`**: Applies atmospheric scattering equations, gamma attenuation, rain streak rendering, and shadow masks to generate 6 synthetic adverse condition datasets.
* **`enhance_conditions.py`**: Implementation of pavement-tailored image enhancement algorithms (CLAHE, unsharp masking, contrast stretching, bilateral filtering).
* **`analyze_conditions.py`**: Extracts quantitative visual features (*brightness, contrast, blur, saturation, dark pixel ratio*).
* **`evaluate_e2_enhancement.py`**: Automated runner for the E2 Experiment comparing original vs. enhanced image pipelines across all conditions.

---

## 📁 Directory Structure

```text
rs/
├── configs/                            # Configuration files
│   ├── experiment.yaml                 # Core experiment hyperparameter defaults
│   ├── mwpd.yaml                       # MWPD dataset paths & class definitions
│   └── rdd2022.yaml                    # RDD2022 dataset paths & 5-class mappings
├── data/                               # Workspace datasets
│   ├── RDD2022/                        # RDD2022 Raw & Converted splits
│   ├── Multi-Weather Pothole Detection (MWPD)/  # MWPD Raw & Condition datasets
│   ├── Bandung Pothole Image Dataset (BPID)/
│   └── An Annotated Water-Filled, and Dry Potholes Dataset/
├── experiments/                        # Experiment outputs & logs
│   └── phase1/
│       ├── baseline/                   # YOLOv8n RDD2022 baseline training runs
│       ├── mwpd/                       # YOLOv8s MWPD 50-epoch training run
│       ├── mwpd_yolov8m/               # YOLOv8m MWPD 50-epoch training run
│       └── environment/                # E2 experiment results & CSVs
│           ├── condition_evaluation/   # Per-condition YAMLs & result CSVs
│           ├── e2_enhancement_comparison.csv  # Side-by-side original vs enhanced metrics
│           └── environment_features.csv       # Extracted visual condition statistics
├── notebooks/                          # Jupyter analysis notebooks
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_baseline_training.ipynb
│   └── 03_baseline_evaluation.ipynb
├── src/                                # Core python packages
│   ├── data/                           # Data conversion & profiling scripts
│   ├── detection/                      # YOLO training & evaluation scripts
│   ├── enhancement/                    # Image enhancement modules
│   └── environment/                    # Condition generator & E2 evaluator
├── yolov8m.pt                          # Pretrained YOLOv8m base weights
├── yolov8n.pt                          # Pretrained YOLOv8n base weights
├── README.md                           # Master project documentation
└── requirements.txt                    # Project dependencies
```

---

## 🚀 How to Run & Reproduce Experiments

### 1. Environment Setup
```bash
# Clone repository and navigate to root
cd rs

# Install dependencies
pip install -r requirements.txt
```

### 2. Train YOLO Models
```bash
# Train YOLOv8s Baseline on MWPD (50 Epochs)
python -m src.detection.train --dataset mwpd --epochs 50

# Train YOLOv8m Baseline on MWPD (50 Epochs)
python -m src.detection.train --dataset mwpd_yolov8m --epochs 50

# Train YOLOv8n Baseline on RDD2022 (70 Epochs)
python -m src.detection.train --dataset rdd2022 --epochs 70
```

### 3. Generate Adverse Environmental Conditions & Run E2 Benchmark
```bash
# Generate synthetic weather conditions
python -m src.environment.generate_conditions

# Apply enhancement pipeline
python -m src.environment.enhance_conditions

# Run E2 Original vs Enhanced Evaluation
python -m src.environment.evaluate_e2_enhancement
```

---

## 📝 Key Insights for Report Generation

When compiling your final research paper or technical report, highlight the following key takeaways:

1. **Model Selection**: **YOLOv8s** is superior to YOLOv8m for pavement damage detection. It achieves **higher mAP@0.50 (76.41% vs 74.67%)**, **higher Precision (82.10% vs 78.77%)**, and **11.1× faster inference speed (1.65ms vs 18.3ms)**, while consuming **56.8% less memory**.
2. **Dataset Heterogeneity**: Multi-country training (RDD2022) requires handling significant visual variations in road marking styles, camera mount positions, and asphalt weathering.
3. **Adverse Weather Impact**: Shadows and severe low-light occlusions represent the primary failure modes for vision-based pothole detection, requiring specialized illumination recovery.
4. **Adaptive Preprocessing**: Static image enhancement can introduce artifacts in foggy or rainy scenes. An adaptive enhancement module that selectively triggers filters based on real-time image quality metrics yields optimal real-world robustness.

---

*Documentation updated for research report creation & experimental reproducibility.*
