from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_bot.camera_catalog import UfanetCatalogClient
from parking_bot.config_loader import load_camera_configs
from parking_bot.demo_loader import load_demo_observations
from parking_bot.detector import ParkingSpaceDetector, VehicleDetector
from parking_bot.repository import SQLiteRepository
from parking_bot.service import CameraMonitorService
from parking_bot.settings import load_settings
from parking_bot.slot_classifier import SlotStatusClassifier
from parking_bot.types import CameraConfig, ResolvedCamera

CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.yaml"
FRAMES_DIR = PROJECT_ROOT / "runtime" / "frames"
OUTPUT_JSON_PATH = PROJECT_ROOT / "runtime" / "benchmark_presentation.json"
OUTPUT_MARKDOWN_PATH = PROJECT_ROOT / "runtime" / "benchmark_presentation.md"
RESULTS_CSV_PATH = PROJECT_ROOT / "pklot_yolov84" / "results.csv"
BENCHMARK_DB_PATH = PROJECT_ROOT / "runtime" / "benchmark_pipeline.db"
MEASURED_ROUNDS = 5
WARMUP_ROUNDS = 2
INCLUDE_DISABLED_CAMERAS = False


@dataclass(slots=True)
class CallMetrics:
    total_ms: float = 0.0
    calls: int = 0

    def record(self, elapsed_ms: float) -> None:
        self.total_ms += elapsed_ms
        self.calls += 1


@dataclass(slots=True)
class RoundReport:
    round_index: int
    full_pipeline_ms: float
    image_load_ms: float
    image_load_calls: int
    space_detector_ms: float
    space_detector_calls: int
    vehicle_detector_ms: float
    vehicle_detector_calls: int
    slot_classifier_ms: float
    slot_classifier_calls: int
    other_ms: float
    free_count: int
    occupied_count: int
    availability: str


class TimedParkingSpaceDetector:
    def __init__(self, inner: ParkingSpaceDetector) -> None:
        self._inner = inner
        self._metrics = CallMetrics()

    def reset_benchmark_metrics(self) -> None:
        self._metrics = CallMetrics()

    def benchmark_metrics_snapshot(self) -> dict[str, float | int]:
        return {
            "space_detector_ms": self._metrics.total_ms,
            "space_detector_calls": self._metrics.calls,
        }

    def detect_image(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._inner.detect_image(*args, **kwargs)
        finally:
            self._metrics.record((time.perf_counter() - started) * 1000.0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TimedVehicleDetector:
    def __init__(self, inner: VehicleDetector) -> None:
        self._inner = inner
        self._metrics = CallMetrics()

    def reset_benchmark_metrics(self) -> None:
        self._metrics = CallMetrics()

    def benchmark_metrics_snapshot(self) -> dict[str, float | int]:
        return {
            "vehicle_detector_ms": self._metrics.total_ms,
            "vehicle_detector_calls": self._metrics.calls,
        }

    def detect_image(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._inner.detect_image(*args, **kwargs)
        finally:
            self._metrics.record((time.perf_counter() - started) * 1000.0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TimedSlotStatusClassifier:
    def __init__(self, inner: SlotStatusClassifier) -> None:
        self._inner = inner
        self._metrics = CallMetrics()

    def reset_benchmark_metrics(self) -> None:
        self._metrics = CallMetrics()

    def benchmark_metrics_snapshot(self) -> dict[str, float | int]:
        return {
            "slot_classifier_ms": self._metrics.total_ms,
            "slot_classifier_calls": self._metrics.calls,
        }

    def predict_image(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._inner.predict_image(*args, **kwargs)
        finally:
            self._metrics.record((time.perf_counter() - started) * 1000.0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class BenchmarkCameraMonitorService(CameraMonitorService):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._image_load_metrics = CallMetrics()

    def reset_benchmark_metrics(self) -> None:
        self._image_load_metrics = CallMetrics()

        if hasattr(self.detector, "reset_benchmark_metrics"):
            self.detector.reset_benchmark_metrics()
        if self.vehicle_detector is not None and hasattr(
            self.vehicle_detector, "reset_benchmark_metrics"
        ):
            self.vehicle_detector.reset_benchmark_metrics()
        if self.slot_classifier is not None and hasattr(
            self.slot_classifier, "reset_benchmark_metrics"
        ):
            self.slot_classifier.reset_benchmark_metrics()

    def benchmark_metrics_snapshot(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "image_load_ms": self._image_load_metrics.total_ms,
            "image_load_calls": self._image_load_metrics.calls,
            "space_detector_ms": 0.0,
            "space_detector_calls": 0,
            "vehicle_detector_ms": 0.0,
            "vehicle_detector_calls": 0,
            "slot_classifier_ms": 0.0,
            "slot_classifier_calls": 0,
        }
        if hasattr(self.detector, "benchmark_metrics_snapshot"):
            payload.update(self.detector.benchmark_metrics_snapshot())
        if self.vehicle_detector is not None and hasattr(
            self.vehicle_detector, "benchmark_metrics_snapshot"
        ):
            payload.update(self.vehicle_detector.benchmark_metrics_snapshot())
        if self.slot_classifier is not None and hasattr(
            self.slot_classifier, "benchmark_metrics_snapshot"
        ):
            payload.update(self.slot_classifier.benchmark_metrics_snapshot())
        return payload

    def _load_frame_image(self, raw_frame_path: Path) -> Any:
        started = time.perf_counter()
        try:
            return super()._load_frame_image(raw_frame_path)
        finally:
            self._image_load_metrics.record((time.perf_counter() - started) * 1000.0)


def _build_offline_cameras(camera_configs: list[CameraConfig]) -> list[ResolvedCamera]:
    cameras: list[ResolvedCamera] = []
    for config in camera_configs:
        demo_observations = ()
        if config.source_kind == "demo":
            if config.demo_observations_path is None:
                raise RuntimeError(
                    f"Demo camera {config.id} must define demo_observations_file in the config."
                )
            demo_observations = load_demo_observations(config.demo_observations_path)

        cameras.append(
            ResolvedCamera(
                id=config.id,
                map_url=config.map_url,
                number=config.id if config.source_kind == "demo" else "offline",
                server="demo" if config.source_kind == "demo" else "offline",
                token="demo" if config.source_kind == "demo" else "offline",
                display_name=config.display_name or config.id,
                enabled=config.enabled,
                source_kind=config.source_kind,
                source_transport=config.source_transport,
                min_confidence=config.min_confidence,
                min_free_spaces=config.min_free_spaces,
                detection_image_size=config.detection_image_size,
                detection_roi=config.detection_roi,
                detection_exclude_rois=config.detection_exclude_rois,
                parking_slots=config.parking_slots,
                stable_cycles=config.stable_cycles,
                poll_interval_seconds=config.poll_interval_seconds,
                demo_observations=demo_observations,
                demo_loop=config.demo_loop,
            )
        )
    return cameras


def _find_raw_frame(camera: ResolvedCamera, frames_dir: Path) -> Path | None:
    candidate = frames_dir / f"{camera.id}_raw.jpg"
    if candidate.exists():
        return candidate

    for observation in camera.demo_observations:
        if observation.raw_frame_path is not None and observation.raw_frame_path.exists():
            return observation.raw_frame_path
    return None


def _round_ms(value: float) -> float:
    return round(value, 1)


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * ratio
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "mean_ms": _round_ms(statistics.fmean(values)),
        "median_ms": _round_ms(statistics.median(values)),
        "p95_ms": _round_ms(_percentile(values, 0.95)),
        "min_ms": _round_ms(min(values)),
        "max_ms": _round_ms(max(values)),
    }


def _mean_int(values: list[int]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def _mean_calls(values: list[int]) -> float:
    return round(statistics.fmean(values), 2) if values else 0.0


def _per_call_ms(total_ms: float, total_calls: int) -> float:
    if total_calls <= 0:
        return 0.0
    return round(total_ms / total_calls, 2)


def _build_environment_report(settings: Any) -> dict[str, object]:
    return {
        "project_root": str(PROJECT_ROOT),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or os.getenv("PROCESSOR_IDENTIFIER", ""),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "model_path": str(settings.model_path),
        "vehicle_model_path": str(settings.vehicle_model_path),
        "slot_classifier_model_path": str(settings.slot_classifier_model_path),
    }


def _camera_summary(
    camera: ResolvedCamera,
    raw_frame_path: Path,
    rounds: list[RoundReport],
) -> dict[str, object]:
    full_pipeline_values = [item.full_pipeline_ms for item in rounds]
    image_load_values = [item.image_load_ms for item in rounds]
    space_detector_values = [item.space_detector_ms for item in rounds]
    vehicle_detector_values = [item.vehicle_detector_ms for item in rounds]
    slot_classifier_values = [item.slot_classifier_ms for item in rounds]
    other_values = [item.other_ms for item in rounds]
    space_detector_calls = [item.space_detector_calls for item in rounds]
    vehicle_detector_calls = [item.vehicle_detector_calls for item in rounds]
    slot_classifier_calls = [item.slot_classifier_calls for item in rounds]

    total_space_detector_ms = sum(space_detector_values)
    total_vehicle_detector_ms = sum(vehicle_detector_values)
    total_slot_classifier_ms = sum(slot_classifier_values)
    total_space_detector_calls = sum(space_detector_calls)
    total_vehicle_detector_calls = sum(vehicle_detector_calls)
    total_slot_classifier_calls = sum(slot_classifier_calls)

    return {
        "camera_id": camera.id,
        "display_name": camera.display_name,
        "pipeline_type": "slot_pipeline" if camera.parking_slots else "space_model",
        "raw_frame_path": str(raw_frame_path),
        "slot_count": len(camera.parking_slots),
        "rounds": len(rounds),
        "availability_samples": [item.availability for item in rounds],
        "free_count_mean": _mean_int([item.free_count for item in rounds]),
        "occupied_count_mean": _mean_int([item.occupied_count for item in rounds]),
        "full_pipeline": _summarize_metric(full_pipeline_values),
        "image_load": _summarize_metric(image_load_values),
        "space_detector": {
            **_summarize_metric(space_detector_values),
            "calls_mean": _mean_calls(space_detector_calls),
            "total_ms": _round_ms(total_space_detector_ms),
            "total_calls": total_space_detector_calls,
            "per_call_ms": _per_call_ms(total_space_detector_ms, total_space_detector_calls),
        },
        "vehicle_detector": {
            **_summarize_metric(vehicle_detector_values),
            "calls_mean": _mean_calls(vehicle_detector_calls),
            "total_ms": _round_ms(total_vehicle_detector_ms),
            "total_calls": total_vehicle_detector_calls,
            "per_call_ms": _per_call_ms(total_vehicle_detector_ms, total_vehicle_detector_calls),
        },
        "slot_classifier": {
            **_summarize_metric(slot_classifier_values),
            "calls_mean": _mean_calls(slot_classifier_calls),
            "total_ms": _round_ms(total_slot_classifier_ms),
            "total_calls": total_slot_classifier_calls,
            "per_call_ms": _per_call_ms(total_slot_classifier_ms, total_slot_classifier_calls),
        },
        "other": _summarize_metric(other_values),
        "per_round": [asdict(item) for item in rounds],
    }


def benchmark_camera(
    service: BenchmarkCameraMonitorService,
    camera: ResolvedCamera,
    raw_frame_path: Path,
    *,
    warmup_rounds: int,
    rounds: int,
) -> dict[str, object]:
    for _ in range(max(0, warmup_rounds)):
        service.reset_benchmark_metrics()
        service._detect_frame(camera, raw_frame_path)

    measured_rounds: list[RoundReport] = []
    for round_index in range(1, max(1, rounds) + 1):
        service.reset_benchmark_metrics()
        started = time.perf_counter()
        summary = service._detect_frame(camera, raw_frame_path)
        full_pipeline_ms = (time.perf_counter() - started) * 1000.0
        stage_metrics = service.benchmark_metrics_snapshot()

        known_stage_ms = (
            float(stage_metrics["image_load_ms"])
            + float(stage_metrics["space_detector_ms"])
            + float(stage_metrics["vehicle_detector_ms"])
            + float(stage_metrics["slot_classifier_ms"])
        )
        measured_rounds.append(
            RoundReport(
                round_index=round_index,
                full_pipeline_ms=_round_ms(full_pipeline_ms),
                image_load_ms=_round_ms(float(stage_metrics["image_load_ms"])),
                image_load_calls=int(stage_metrics["image_load_calls"]),
                space_detector_ms=_round_ms(float(stage_metrics["space_detector_ms"])),
                space_detector_calls=int(stage_metrics["space_detector_calls"]),
                vehicle_detector_ms=_round_ms(float(stage_metrics["vehicle_detector_ms"])),
                vehicle_detector_calls=int(stage_metrics["vehicle_detector_calls"]),
                slot_classifier_ms=_round_ms(float(stage_metrics["slot_classifier_ms"])),
                slot_classifier_calls=int(stage_metrics["slot_classifier_calls"]),
                other_ms=_round_ms(max(0.0, full_pipeline_ms - known_stage_ms)),
                free_count=summary.free_count,
                occupied_count=summary.occupied_count,
                availability=summary.availability.value,
            )
        )

    return _camera_summary(camera, raw_frame_path, measured_rounds)


def _build_benchmark_service(
    settings: Any,
    cameras: list[ResolvedCamera],
) -> BenchmarkCameraMonitorService:
    repository = SQLiteRepository(BENCHMARK_DB_PATH)
    detector = TimedParkingSpaceDetector(
        ParkingSpaceDetector(settings.model_path, image_size=max(640, settings.image_width))
    )
    vehicle_detector = TimedVehicleDetector(VehicleDetector(settings.vehicle_model_path))
    slot_classifier = (
        TimedSlotStatusClassifier(
            SlotStatusClassifier(
                settings.slot_classifier_model_path,
                image_size=settings.slot_classifier_image_size,
            )
        )
        if settings.slot_classifier_model_path.exists()
        else None
    )
    catalog_client = UfanetCatalogClient(settings.catalog_url, settings.user_agent)
    return BenchmarkCameraMonitorService(
        settings=settings,
        repository=repository,
        detector=detector,
        vehicle_detector=vehicle_detector,
        slot_classifier=slot_classifier,
        catalog_client=catalog_client,
        cameras=cameras,
    )


def _load_training_metrics() -> dict[str, float] | None:
    if not RESULTS_CSV_PATH.exists():
        return None
    with RESULTS_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    last_row = rows[-1]
    try:
        return {
            "epoch": int(float(last_row["epoch"])),
            "mAP50": round(float(last_row["metrics/mAP50(B)"]), 4),
            "mAP50_95": round(float(last_row["metrics/mAP50-95(B)"]), 4),
            "precision": round(float(last_row["metrics/precision(B)"]), 4),
            "recall": round(float(last_row["metrics/recall(B)"]), 4),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _build_stage_overview(camera_reports: list[dict[str, object]]) -> dict[str, object]:
    image_load_values: list[float] = []
    space_detector_values: list[float] = []
    vehicle_detector_values: list[float] = []
    slot_classifier_values: list[float] = []
    other_values: list[float] = []
    vehicle_total_ms = 0.0
    vehicle_total_calls = 0
    slot_total_ms = 0.0
    slot_total_calls = 0
    space_total_ms = 0.0
    space_total_calls = 0

    for camera in camera_reports:
        for round_data in camera["per_round"]:
            image_load_values.append(round_data["image_load_ms"])
            space_detector_values.append(round_data["space_detector_ms"])
            vehicle_detector_values.append(round_data["vehicle_detector_ms"])
            slot_classifier_values.append(round_data["slot_classifier_ms"])
            other_values.append(round_data["other_ms"])
            vehicle_total_ms += round_data["vehicle_detector_ms"]
            vehicle_total_calls += round_data["vehicle_detector_calls"]
            slot_total_ms += round_data["slot_classifier_ms"]
            slot_total_calls += round_data["slot_classifier_calls"]
            space_total_ms += round_data["space_detector_ms"]
            space_total_calls += round_data["space_detector_calls"]

    return {
        "image_load": _summarize_metric(image_load_values),
        "space_detector": {
            **_summarize_metric(space_detector_values),
            "per_call_ms": _per_call_ms(space_total_ms, space_total_calls),
            "total_calls": space_total_calls,
        },
        "vehicle_detector": {
            **_summarize_metric(vehicle_detector_values),
            "per_call_ms": _per_call_ms(vehicle_total_ms, vehicle_total_calls),
            "total_calls": vehicle_total_calls,
        },
        "slot_classifier": {
            **_summarize_metric(slot_classifier_values),
            "per_call_ms": _per_call_ms(slot_total_ms, slot_total_calls),
            "total_calls": slot_total_calls,
        },
        "other": _summarize_metric(other_values),
    }


def _build_cold_start_summary(
    settings: Any,
    available_frames: list[tuple[ResolvedCamera, Path]],
) -> dict[str, object] | None:
    if not available_frames:
        return None

    reference_camera, raw_frame_path = min(available_frames, key=lambda item: len(item[0].parking_slots))
    service = _build_benchmark_service(settings, [reference_camera])
    try:
        service.reset_benchmark_metrics()
        started = time.perf_counter()
        summary = service._detect_frame(reference_camera, raw_frame_path)
        full_pipeline_ms = (time.perf_counter() - started) * 1000.0
        stage_metrics = service.benchmark_metrics_snapshot()
    finally:
        service._executor.shutdown(wait=False, cancel_futures=True)

    known_stage_ms = (
        float(stage_metrics["image_load_ms"])
        + float(stage_metrics["space_detector_ms"])
        + float(stage_metrics["vehicle_detector_ms"])
        + float(stage_metrics["slot_classifier_ms"])
    )
    return {
        "camera_id": reference_camera.id,
        "slot_count": len(reference_camera.parking_slots),
        "raw_frame_path": str(raw_frame_path),
        "full_pipeline_ms": _round_ms(full_pipeline_ms),
        "image_load_ms": _round_ms(float(stage_metrics["image_load_ms"])),
        "space_detector_ms": _round_ms(float(stage_metrics["space_detector_ms"])),
        "vehicle_detector_ms": _round_ms(float(stage_metrics["vehicle_detector_ms"])),
        "slot_classifier_ms": _round_ms(float(stage_metrics["slot_classifier_ms"])),
        "other_ms": _round_ms(max(0.0, full_pipeline_ms - known_stage_ms)),
        "free_count": summary.free_count,
        "occupied_count": summary.occupied_count,
        "availability": summary.availability.value,
    }


def _build_presentation_summary(
    report: dict[str, object],
) -> dict[str, object]:
    cameras = report["cameras"]
    fastest = min(cameras, key=lambda item: item["full_pipeline"]["mean_ms"])
    slowest = max(cameras, key=lambda item: item["full_pipeline"]["mean_ms"])
    stage_overview = report["stage_overview"]
    overall = report["overall"]["full_pipeline"]

    summary: dict[str, object] = {
        "camera_count": report["overall"]["camera_count"],
        "warm_full_pipeline_mean_ms": overall["mean_ms"],
        "warm_full_pipeline_median_ms": overall["median_ms"],
        "warm_full_pipeline_p95_ms": overall["p95_ms"],
        "fastest_camera": {
            "camera_id": fastest["camera_id"],
            "slot_count": fastest["slot_count"],
            "mean_ms": fastest["full_pipeline"]["mean_ms"],
        },
        "slowest_camera": {
            "camera_id": slowest["camera_id"],
            "slot_count": slowest["slot_count"],
            "mean_ms": slowest["full_pipeline"]["mean_ms"],
        },
        "vehicle_detector_per_call_ms": stage_overview["vehicle_detector"]["per_call_ms"],
        "slot_classifier_per_call_ms": stage_overview["slot_classifier"]["per_call_ms"],
        "space_detector_per_call_ms": stage_overview["space_detector"]["per_call_ms"],
    }
    if report["cold_start"] is not None:
        summary["cold_start_reference"] = {
            "camera_id": report["cold_start"]["camera_id"],
            "slot_count": report["cold_start"]["slot_count"],
            "full_pipeline_ms": report["cold_start"]["full_pipeline_ms"],
        }
    if report["training_metrics"] is not None:
        summary["training_metrics"] = report["training_metrics"]
    return summary


def print_report(report: dict[str, object]) -> None:
    environment = report["environment"]
    print("Environment:")
    print(
        f"  Python {environment['python_version']} | {environment['platform']} | "
        f"machine={environment['machine']} | cpu={environment['cpu_count']}"
    )
    if environment["processor"]:
        print(f"  processor={environment['processor']}")

    print()
    print("Camera benchmarks:")
    for camera in report["cameras"]:
        full_pipeline = camera["full_pipeline"]
        vehicle = camera["vehicle_detector"]
        slot_classifier = camera["slot_classifier"]
        space_detector = camera["space_detector"]
        other = camera["other"]
        print(
            f"  {camera['camera_id']}: full mean={full_pipeline['mean_ms']} ms "
            f"(p95={full_pipeline['p95_ms']} ms, min={full_pipeline['min_ms']} ms, "
            f"max={full_pipeline['max_ms']} ms)"
        )
        print(
            f"    slots={camera['slot_count']} | pipeline={camera['pipeline_type']} | "
            f"frame={camera['raw_frame_path']}"
        )
        print(
            f"    image_load={camera['image_load']['mean_ms']} ms | "
            f"space_detector={space_detector['mean_ms']} ms ({space_detector['calls_mean']} calls) | "
            f"vehicle_detector={vehicle['mean_ms']} ms ({vehicle['calls_mean']} calls, "
            f"{vehicle['per_call_ms']} ms/call) | "
            f"slot_classifier={slot_classifier['mean_ms']} ms ({slot_classifier['calls_mean']} calls, "
            f"{slot_classifier['per_call_ms']} ms/call) | "
            f"other={other['mean_ms']} ms"
        )

    overall = report["overall"]
    stage_overview = report["stage_overview"]
    print()
    print(
        "Overall:"
        f" full mean={overall['full_pipeline']['mean_ms']} ms"
        f" | median={overall['full_pipeline']['median_ms']} ms"
        f" | p95={overall['full_pipeline']['p95_ms']} ms"
    )
    print(
        "Stages:"
        f" image_load={stage_overview['image_load']['mean_ms']} ms"
        f" | vehicle_per_call={stage_overview['vehicle_detector']['per_call_ms']} ms"
        f" | slot_classifier_per_call={stage_overview['slot_classifier']['per_call_ms']} ms"
    )
    if report["cold_start"] is not None:
        cold = report["cold_start"]
        print(
            "Cold start:"
            f" camera={cold['camera_id']}"
            f" | slots={cold['slot_count']}"
            f" | full={cold['full_pipeline_ms']} ms"
        )
    if report["training_metrics"] is not None:
        metrics = report["training_metrics"]
        print(
            "Training:"
            f" epoch={metrics['epoch']}"
            f" | mAP50={metrics['mAP50']}"
            f" | mAP50-95={metrics['mAP50_95']}"
        )


def _build_markdown_summary(report: dict[str, object]) -> str:
    presentation = report["presentation_summary"]
    lines = [
        "# Benchmark Summary",
        "",
        "## Environment",
        (
            f"- Python {report['environment']['python_version']}, "
            f"{report['environment']['platform']}, "
            f"CPU threads: {report['environment']['cpu_count']}"
        ),
        f"- Processor: {report['environment']['processor'] or 'unknown'}",
        "",
        "## Presentation Numbers",
        f"- Warm full pipeline mean: {presentation['warm_full_pipeline_mean_ms']} ms",
        f"- Warm full pipeline median: {presentation['warm_full_pipeline_median_ms']} ms",
        f"- Warm full pipeline p95: {presentation['warm_full_pipeline_p95_ms']} ms",
        (
            f"- Fastest camera: {presentation['fastest_camera']['camera_id']} "
            f"({presentation['fastest_camera']['mean_ms']} ms, "
            f"{presentation['fastest_camera']['slot_count']} slots)"
        ),
        (
            f"- Slowest camera: {presentation['slowest_camera']['camera_id']} "
            f"({presentation['slowest_camera']['mean_ms']} ms, "
            f"{presentation['slowest_camera']['slot_count']} slots)"
        ),
        (
            f"- Vehicle detector mean per call: "
            f"{presentation['vehicle_detector_per_call_ms']} ms"
        ),
        (
            f"- Slot classifier mean per call: "
            f"{presentation['slot_classifier_per_call_ms']} ms"
        ),
    ]
    if "cold_start_reference" in presentation:
        lines.append(
            f"- Cold start reference: {presentation['cold_start_reference']['full_pipeline_ms']} ms "
            f"on {presentation['cold_start_reference']['camera_id']} "
            f"({presentation['cold_start_reference']['slot_count']} slots)"
        )
    if "training_metrics" in presentation:
        metrics = presentation["training_metrics"]
        lines.extend(
            [
                f"- mAP50: {metrics['mAP50']}",
                f"- mAP50-95: {metrics['mAP50_95']}",
                f"- Precision: {metrics['precision']}",
                f"- Recall: {metrics['recall']}",
            ]
        )

    lines.extend(["", "## Per Camera"])
    for camera in report["cameras"]:
        lines.append(
            f"- {camera['camera_id']}: mean={camera['full_pipeline']['mean_ms']} ms, "
            f"p95={camera['full_pipeline']['p95_ms']} ms, slots={camera['slot_count']}"
        )

    if report["skipped_cameras"]:
        lines.extend(["", "## Skipped"])
        for item in report["skipped_cameras"]:
            lines.append(f"- {item['camera_id']}: {item['reason']}")

    return "\n".join(lines) + "\n"


def _run_preflight_probe(label: str, code: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    signal_name = None
    if result.returncode < 0:
        try:
            import signal

            signal_name = signal.Signals(-result.returncode).name
        except Exception:
            signal_name = f"SIG{-result.returncode}"
    return {
        "label": label,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "signal": signal_name,
    }


def _run_preflight_checks() -> list[dict[str, object]]:
    return [
        _run_preflight_probe("numpy", "import numpy; print(numpy.__version__)"),
        _run_preflight_probe("cv2", "import cv2; print(cv2.__version__)"),
        _run_preflight_probe("torch", "import torch; print(torch.__version__)"),
        _run_preflight_probe("torchvision", "import torchvision; print(torchvision.__version__)"),
        _run_preflight_probe(
            "ultralytics",
            "from ultralytics import YOLO; print('ultralytics ok')",
        ),
    ]


def _find_failed_probe(probes: list[dict[str, object]]) -> dict[str, object] | None:
    for probe in probes:
        if int(probe["returncode"]) != 0:
            return probe
    return None


def _format_preflight_error(probe: dict[str, object]) -> str:
    signal_name = probe.get("signal")
    signal_suffix = f", signal={signal_name}" if signal_name else ""
    stderr = probe.get("stderr") or ""
    stdout = probe.get("stdout") or ""
    details = stderr or stdout or "no extra output"
    return (
        "Preflight dependency check failed before benchmarking.\n"
        f"Probe: {probe['label']}\n"
        f"Return code: {probe['returncode']}{signal_suffix}\n"
        f"Details: {details}\n\n"
        "This usually means a native wheel is incompatible with the current CPU/OS build. "
        "On Raspberry Pi 3 the most likely culprit is torch/torchvision."
    )


def main() -> int:
    settings = load_settings(PROJECT_ROOT, require_telegram_token=False)
    preflight_checks = _run_preflight_checks()
    failed_probe = _find_failed_probe(preflight_checks)
    if failed_probe is not None:
        raise RuntimeError(_format_preflight_error(failed_probe))

    config_path = CONFIG_PATH
    frames_dir = FRAMES_DIR
    if not frames_dir.exists():
        raise RuntimeError(f"Frames directory was not found: {frames_dir}")

    camera_configs = load_camera_configs(config_path, settings)
    offline_cameras = _build_offline_cameras(camera_configs)
    cameras = [
        camera
        for camera in offline_cameras
        if INCLUDE_DISABLED_CAMERAS or camera.enabled
    ]
    if not cameras:
        raise RuntimeError("No cameras matched the benchmark selection.")

    service = _build_benchmark_service(settings, cameras)

    camera_reports: list[dict[str, object]] = []
    skipped_cameras: list[dict[str, str]] = []
    available_frames: list[tuple[ResolvedCamera, Path]] = []
    try:
        for camera in cameras:
            raw_frame_path = _find_raw_frame(camera, frames_dir)
            if raw_frame_path is None:
                skipped_cameras.append(
                    {
                        "camera_id": camera.id,
                        "reason": f"Raw frame was not found in {frames_dir}",
                    }
                )
                continue
            available_frames.append((camera, raw_frame_path))
            print(
                f"Benchmarking {camera.id} "
                f"(warmup={WARMUP_ROUNDS}, rounds={MEASURED_ROUNDS})..."
            )
            camera_reports.append(
                benchmark_camera(
                    service,
                    camera,
                    raw_frame_path,
                    warmup_rounds=WARMUP_ROUNDS,
                    rounds=MEASURED_ROUNDS,
                )
            )
    finally:
        service._executor.shutdown(wait=False, cancel_futures=True)

    if not camera_reports:
        raise RuntimeError("No benchmark data was collected. Check camera selection and frame files.")

    camera_reports.sort(key=lambda item: item["full_pipeline"]["mean_ms"], reverse=True)
    overall_full_values = [
        round_data["full_pipeline_ms"]
        for camera in camera_reports
        for round_data in camera["per_round"]
    ]
    report = {
        "environment": _build_environment_report(settings),
        "config_path": str(config_path),
        "frames_dir": str(frames_dir),
        "warmup_rounds": WARMUP_ROUNDS,
        "rounds": MEASURED_ROUNDS,
        "cameras": camera_reports,
        "skipped_cameras": skipped_cameras,
        "training_metrics": _load_training_metrics(),
        "cold_start": _build_cold_start_summary(settings, available_frames),
        "preflight_checks": preflight_checks,
        "overall": {
            "camera_count": len(camera_reports),
            "skipped_count": len(skipped_cameras),
            "full_pipeline": _summarize_metric(overall_full_values),
        },
    }
    report["stage_overview"] = _build_stage_overview(camera_reports)
    report["presentation_summary"] = _build_presentation_summary(report)

    print()
    print_report(report)
    if skipped_cameras:
        print()
        print("Skipped cameras:")
        for item in skipped_cameras:
            print(f"  {item['camera_id']}: {item['reason']}")
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_MARKDOWN_PATH.write_text(_build_markdown_summary(report), encoding="utf-8")
    print()
    print(f"Saved benchmark JSON to: {OUTPUT_JSON_PATH}")
    print(f"Saved benchmark summary to: {OUTPUT_MARKDOWN_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
