
import csv
import json
import time
import threading
from datetime import datetime
from pathlib import Path

import psutil
import pynvml
import torch
from ultralytics import YOLO


# ---------------------------------------------------------
# Configuration: preserve existing checkpoints and results
# ---------------------------------------------------------
ROOT = Path(r"D:\research papper\rs")

MODEL_N_PATH = (
    ROOT / "experiments" / "phase1" / "mwpd"
    / "weights" / "best.pt"
)
MODEL_M_PATH = (
    ROOT / "experiments" / "phase1" / "mwpd_yolov8m"
    / "weights" / "best.pt"
)

CONDITIONS_DIR = (
    ROOT / "data" / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions"
)

OUTPUT_ROOT = (
    ROOT / "experiments" / "phase1"
    / "energy_resource_optimization"
)

IMG_SIZE = 640
CONF = 0.001
IOU = 0.7
MAX_DET = 300
ADAPTIVE_THRESHOLD = 0.30
WARMUP_IMAGES = 5
POWER_SAMPLE_INTERVAL = 0.5

CONDITIONS = [
    "normal",
    "low_light",
    "fog",
    "rain",
    "shadow",
    "water_reflection",
]


def get_images():
    """Collect the same controlled-condition image set for each strategy."""
    images = []

    for condition in CONDITIONS:
        image_dir = CONDITIONS_DIR / condition / "images"

        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Missing condition image directory: {image_dir}"
            )

        files = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )

        if not files:
            raise RuntimeError(f"No images found in {image_dir}")

        images.extend((condition, p) for p in files)

    if not images:
        raise RuntimeError("No test images found.")

    return images


class ResourceMonitor:
    """Sample GPU power/utilization/memory and system CPU/RAM."""

    def __init__(self, handle, output_path):
        self.handle = handle
        self.output_path = output_path
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None

    def sample(self):
        now = time.perf_counter()

        try:
            power_w = pynvml.nvmlDeviceGetPowerUsage(
                self.handle
            ) / 1000.0
        except pynvml.NVMLError:
            power_w = None

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(
                self.handle
            )
            gpu_util = util.gpu
            mem_util = util.memory
        except pynvml.NVMLError:
            gpu_util = None
            mem_util = None

        try:
            memory_mb = (
                pynvml.nvmlDeviceGetMemoryInfo(self.handle).used
                / (1024 ** 2)
            )
        except pynvml.NVMLError:
            memory_mb = None

        try:
            temperature_c = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            temperature_c = None

        self.samples.append({
            "elapsed_s": now,
            "power_w": power_w,
            "gpu_util_percent": gpu_util,
            "gpu_memory_used_mb": memory_mb,
            "gpu_memory_util_percent": mem_util,
            "gpu_temperature_c": temperature_c,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": psutil.virtual_memory().used / (1024 ** 2),
        })

    def _run(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(POWER_SAMPLE_INTERVAL)

        # Include a final sample near the end of the workload.
        self.sample()

    def start(self):
        psutil.cpu_percent(interval=None)
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.write_csv()

    def write_csv(self):
        if not self.samples:
            return

        with self.output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(self.samples[0].keys())
            )
            writer.writeheader()
            writer.writerows(self.samples)

    def summary(self, elapsed_s, image_count):
        valid = [
            s for s in self.samples
            if s["power_w"] is not None
        ]

        energy_j = 0.0
        if len(valid) >= 2:
            for left, right in zip(valid, valid[1:]):
                dt = right["elapsed_s"] - left["elapsed_s"]
                energy_j += (
                    (left["power_w"] + right["power_w"]) / 2.0
                ) * dt

        def mean(key):
            values = [
                s[key] for s in self.samples
                if s[key] is not None
            ]
            return sum(values) / len(values) if values else None

        return {
            "images_processed": image_count,
            "measured_wall_time_s": elapsed_s,
            "throughput_images_per_s": (
                image_count / elapsed_s if elapsed_s > 0 else None
            ),
            "mean_gpu_power_w": mean("power_w"),
            "estimated_gpu_energy_j": energy_j,
            "estimated_gpu_energy_j_per_image": (
                energy_j / image_count if image_count else None
            ),
            "mean_gpu_utilization_percent": mean("gpu_util_percent"),
            "mean_gpu_memory_used_mb": mean("gpu_memory_used_mb"),
            "mean_cpu_percent": mean("cpu_percent"),
            "mean_ram_used_mb": mean("ram_used_mb"),
            "gpu_power_sample_count": len(valid),
        }


def predict(model, image_path):
    """Return maximum detection confidence and elapsed inference time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    result = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DET,
        device=0,
        verbose=False,
        save=False,
    )[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    max_confidence = 0.0
    if result.boxes is not None and len(result.boxes):
        max_confidence = float(
            result.boxes.conf.max().detach().cpu().item()
        )

    return max_confidence, elapsed_ms


def warmup(model, images):
    for _, image_path in images[:WARMUP_IMAGES]:
        predict(model, image_path)


def run_strategy(name, images, model_n, model_m, output_dir):
    telemetry_path = output_dir / f"{name}_telemetry.csv"
    per_image_path = output_dir / f"{name}_per_image.csv"

    monitor = ResourceMonitor(
        GPU_HANDLE, telemetry_path
    )

    records = []
    escalated = 0

    print(f"\nStarting strategy: {name}")

    start = time.perf_counter()
    monitor.start()

    try:
        for index, (condition, image_path) in enumerate(images, 1):
            selected_model = name
            confidence, latency_ms = predict(
                model_n if name != "fixed_yolov8m" else model_m,
                image_path,
            )

            if name == "adaptive":
                if confidence < ADAPTIVE_THRESHOLD:
                    confidence_m, latency_m = predict(
                        model_m, image_path
                    )
                    confidence = confidence_m
                    latency_ms += latency_m
                    selected_model = "yolov8m"
                    escalated += 1
                else:
                    selected_model = "yolov8n"

            records.append({
                "condition": condition,
                "image": image_path.name,
                "strategy": name,
                "selected_model": selected_model,
                "max_detection_confidence": confidence,
                "latency_ms": latency_ms,
            })

            if index % 100 == 0 or index == len(images):
                print(f"  {index}/{len(images)} images processed")

    finally:
        elapsed_s = time.perf_counter() - start
        monitor.stop()

    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    summary = monitor.summary(elapsed_s, len(images))
    summary.update({
        "strategy": name,
        "mean_per_image_latency_ms": (
            sum(r["latency_ms"] for r in records) / len(records)
        ),
        "adaptive_escalated_images": escalated,
        "adaptive_escalation_rate": (
            escalated / len(images) if name == "adaptive" else None
        ),
        "telemetry_note": (
            "GPU board energy estimated by integrating sampled power. "
            "Not whole-laptop energy; readings may include other GPU use."
        ),
    })

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; expected GPU inference.")

    for model_path in (MODEL_N_PATH, MODEL_M_PATH):
        if not model_path.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {model_path}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=False, exist_ok=False)

    images = get_images()
    print(f"Total images across conditions: {len(images)}")
    print(f"Results directory: {output_dir}")

    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)

    try:
        print("Loading YOLOv8n...")
        model_n = YOLO(str(MODEL_N_PATH))

        print("Loading YOLOv8m...")
        model_m = YOLO(str(MODEL_M_PATH))

        # Warm up both models before measuring.
        warmup(model_n, images)
        warmup(model_m, images)

        summaries = []
        for strategy in (
            "fixed_yolov8n",
            "fixed_yolov8m",
            "adaptive",
        ):
            summaries.append(
                run_strategy(
                    strategy, images, model_n, model_m, output_dir
                )
            )

        summary_path = output_dir / "resource_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)

        with (output_dir / "resource_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(summaries[0].keys())
            )
            writer.writeheader()
            writer.writerows(summaries)

        print("\nResource baseline completed.")
        print(f"Summary: {summary_path}")

    finally:
        pynvml.nvmlShutdown()
