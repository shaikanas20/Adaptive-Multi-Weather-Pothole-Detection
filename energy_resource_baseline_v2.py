
import csv
import json
import random
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import pynvml
import torch
from ultralytics import YOLO


ROOT = Path(r"D:\research papper\rs")

MODEL_N_PATH = ROOT / "experiments" / "phase1" / "mwpd" / "weights" / "best.pt"
MODEL_M_PATH = ROOT / "experiments" / "phase1" / "mwpd_yolov8m" / "weights" / "best.pt"

# Match the original adaptive-model-selection experiment.
CONDITIONS_DIR = (
    ROOT / "data" / "Multi-Weather Pothole Detection (MWPD)"
    / "controlled_conditions_advanced"
)

OUTPUT_ROOT = (
    ROOT / "experiments" / "phase1"
    / "energy_resource_optimization"
)

IMG_SIZE = 640
CONF = 0.001
IOU = 0.7
MAX_DET = 300
THRESHOLD = 0.30
WARMUP_IMAGES = 5
SAMPLE_INTERVAL = 0.5
REPEATS = 3
SEED = 20261029

CONDITIONS = [
    "normal", "low_light", "fog", "rain", "shadow", "water_reflection"
]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images():
    images = []

    for condition in CONDITIONS:
        folder = CONDITIONS_DIR / condition / "images"
        if not folder.is_dir():
            raise FileNotFoundError(folder)

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        if not files:
            raise RuntimeError(f"No images in {folder}")

        images.extend((condition, p) for p in files)

    if len(images) != 582:
        raise RuntimeError(
            f"Expected 582 images, found {len(images)}. "
            "Check the dataset before running."
        )

    return images


class Monitor:
    def __init__(self, handle, output_path):
        self.handle = handle
        self.output_path = output_path
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = None
        self.t0 = None

    def sample(self):
        now = time.perf_counter()
        if self.t0 is None:
            self.t0 = now

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except pynvml.NVMLError:
            power = None

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util, mem_util = util.gpu, util.memory
        except pynvml.NVMLError:
            gpu_util, mem_util = None, None

        try:
            memory_mb = (
                pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / (1024 ** 2)
            )
        except pynvml.NVMLError:
            memory_mb = None

        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                self.handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except pynvml.NVMLError:
            temp = None

        self.samples.append({
            "elapsed_s": now - self.t0,
            "power_w": power,
            "gpu_util_percent": gpu_util,
            "gpu_memory_util_percent": mem_util,
            "gpu_memory_used_mb": memory_mb,
            "gpu_temperature_c": temp,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": psutil.virtual_memory().used / (1024 ** 2),
        })

    def loop(self):
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(SAMPLE_INTERVAL)
        self.sample()

    def start(self):
        psutil.cpu_percent(interval=None)
        self.t0 = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.save()

    def save(self):
        if not self.samples:
            return
        with self.output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.samples[0]))
            writer.writeheader()
            writer.writerows(self.samples)

    def summary(self, elapsed_s, image_count):
        valid = [s for s in self.samples if s["power_w"] is not None]
        energy_j = 0.0

        for left, right in zip(valid, valid[1:]):
            dt = right["elapsed_s"] - left["elapsed_s"]
            if dt > 0:
                energy_j += (left["power_w"] + right["power_w"]) * 0.5 * dt

        def mean(key):
            values = [s[key] for s in self.samples if s[key] is not None]
            return sum(values) / len(values) if values else None

        return {
            "images": image_count,
            "wall_time_s": elapsed_s,
            "throughput_images_per_s": image_count / elapsed_s,
            "mean_gpu_power_w": mean("power_w"),
            "estimated_gpu_energy_j": energy_j,
            "estimated_gpu_energy_per_image_j": energy_j / image_count,
            "mean_gpu_utilization_percent": mean("gpu_util_percent"),
            "mean_gpu_memory_used_mb": mean("gpu_memory_used_mb"),
            "mean_gpu_temperature_c": mean("gpu_temperature_c"),
            "mean_cpu_percent": mean("cpu_percent"),
            "mean_ram_used_mb": mean("ram_used_mb"),
            "power_samples": len(valid),
        }


def predict(model, image_path):
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

    latency_ms = (time.perf_counter() - start) * 1000.0
    confidence = 0.0

    if result.boxes is not None and len(result.boxes):
        confidence = float(result.boxes.conf.max().detach().cpu().item())

    return confidence, latency_ms


def run_strategy(strategy, images, model_n, model_m, handle, out_dir, repeat):
    stem = f"repeat_{repeat}_{strategy}"
    monitor = Monitor(handle, out_dir / f"{stem}_telemetry.csv")
    records = []
    escalated = 0

    print(f"\nRepeat {repeat}: {strategy}")
    start = time.perf_counter()
    monitor.start()

    try:
        for index, (condition, path) in enumerate(images, 1):
            n_conf = None
            m_conf = None

            if strategy == "fixed_yolov8n":
                n_conf, latency = predict(model_n, path)
                selected = "yolov8n"

            elif strategy == "fixed_yolov8m":
                m_conf, latency = predict(model_m, path)
                selected = "yolov8m"

            else:
                n_conf, latency_n = predict(model_n, path)
                latency = latency_n
                selected = "yolov8n"

                if n_conf < THRESHOLD:
                    m_conf, latency_m = predict(model_m, path)
                    latency += latency_m
                    selected = "yolov8m"
                    escalated += 1

            records.append({
                "repeat": repeat,
                "condition": condition,
                "image": path.name,
                "strategy": strategy,
                "selected_model": selected,
                "yolov8n_trigger_confidence": n_conf,
                "yolov8m_confidence": m_conf,
                "latency_ms": latency,
            })

            if index % 100 == 0 or index == len(images):
                print(f"  {index}/{len(images)}")

    finally:
        elapsed = time.perf_counter() - start
        monitor.stop()

    with (out_dir / f"{stem}_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary = monitor.summary(elapsed, len(images))
    summary.update({
        "repeat": repeat,
        "strategy": strategy,
        "mean_latency_ms": sum(r["latency_ms"] for r in records) / len(records),
        "escalated_images": escalated if strategy == "adaptive" else None,
        "escalation_rate": escalated / len(images) if strategy == "adaptive" else None,
        "threshold": THRESHOLD if strategy == "adaptive" else None,
    })

    return summary


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    for path in (MODEL_N_PATH, MODEL_M_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    images = collect_images()
    print(f"Dataset: {CONDITIONS_DIR}")
    print(f"Total images: {len(images)}")

    run_id = datetime.now().strftime("run_v2_%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    # Save exact input manifest for reproducibility.
    with (out_dir / "input_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "image_path"])
        writer.writerows((condition, str(path)) for condition, path in images)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    try:
        model_n = YOLO(str(MODEL_N_PATH))
        model_m = YOLO(str(MODEL_M_PATH))

        for _, path in images[:WARMUP_IMAGES]:
            predict(model_n, path)
            predict(model_m, path)

        summaries = []
        rng = random.Random(SEED)

        # Each repeat contains all three strategies, in a randomized order.
        strategies = ["fixed_yolov8n", "fixed_yolov8m", "adaptive"]

        for repeat in range(1, REPEATS + 1):
            order = strategies.copy()
            rng.shuffle(order)
            print(f"\nRepeat {repeat} order: {order}")

            for strategy in order:
                summaries.append(
                    run_strategy(
                        strategy, images, model_n, model_m,
                        handle, out_dir, repeat
                    )
                )

        with (out_dir / "resource_summary.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(summaries, f, indent=2)

        with (out_dir / "resource_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

        print("\nCompleted.")
        print(f"Results saved to: {out_dir}")

    finally:
        pynvml.nvmlShutdown()
