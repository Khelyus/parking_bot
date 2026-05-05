from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import shutil
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_bot.camera_catalog import UfanetCatalogClient, resolve_cameras
from parking_bot.config_loader import load_camera_configs
from parking_bot.detector import ParkingSpaceDetector
from parking_bot.repository import SQLiteRepository
from parking_bot.service import CameraMonitorService
from parking_bot.settings import load_settings
from parking_bot.time_utils import resolve_timezone
from parking_bot.types import ResolvedCamera


logger = logging.getLogger(__name__)


def _require_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for frame archival.") from exc
    return cv2


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for frame archival.") from exc
    return np


@dataclass(slots=True)
class FrameFingerprint:
    sha256: str
    dhash: int
    mean_abs_diff_key: list[int]


@dataclass(slots=True)
class CameraArchiveState:
    fingerprint: FrameFingerprint
    saved_path: Path
    observed_at: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive training frames for enabled slot cameras and skip near-duplicates."
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root. Relative paths are resolved from the repository root, not from cwd.",
    )
    parser.add_argument("--output-dir", default="runtime/training_frames")
    parser.add_argument(
        "--metadata-path",
        default="runtime/training_frames/metadata.jsonl",
        help="JSONL log with save/skip events.",
    )
    parser.add_argument(
        "--log-file",
        default="runtime/logs/archive_training_frames.log",
        help="Path to the text log file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level, for example INFO or DEBUG.",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        help="Optional camera ids to include. By default only enabled cameras with parking_slots are used.",
    )
    parser.add_argument(
        "--include-roi-only",
        action="store_true",
        help="Also include enabled cameras that have ROI config but no parking_slots.",
    )
    parser.add_argument(
        "--day-start",
        default="08:00",
        help="Local time when daytime cadence starts.",
    )
    parser.add_argument(
        "--night-start",
        default="22:00",
        help="Local time when nighttime cadence starts.",
    )
    parser.add_argument(
        "--day-interval-minutes",
        type=int,
        default=15,
        help="Archive cadence during daytime.",
    )
    parser.add_argument(
        "--night-interval-minutes",
        type=int,
        default=30,
        help="Archive cadence during nighttime.",
    )
    parser.add_argument(
        "--dhash-threshold",
        type=int,
        default=4,
        help="Maximum Hamming distance between dHash fingerprints to treat frames as duplicates.",
    )
    parser.add_argument(
        "--mean-abs-diff-threshold",
        type=float,
        default=2.0,
        help="Maximum mean absolute pixel difference on a small grayscale thumbnail to treat frames as duplicates.",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="dHash resolution. 8 means a 64-bit perceptual hash.",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run a single archival round and exit.",
    )
    return parser.parse_args()


def _configure_logging(*, project_root: Path, log_file: str, log_level: str) -> None:
    resolved_log_path = Path(log_file)
    if not resolved_log_path.is_absolute():
        resolved_log_path = (project_root / resolved_log_path).resolve()
    resolved_log_path.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(logging, str(log_level).upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        resolved_log_path,
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logger.info("Logging initialized. log_file=%s level=%s", resolved_log_path, logging.getLevelName(level))


def _parse_hhmm(value: str) -> tuple[int, int]:
    raw = value.strip()
    try:
        hour_text, minute_text = raw.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Time must look like HH:MM, got {value!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise RuntimeError(f"Invalid time value: {value!r}")
    return hour, minute


def _is_daytime(moment: datetime, *, day_start: tuple[int, int], night_start: tuple[int, int]) -> bool:
    current_minutes = moment.hour * 60 + moment.minute
    day_minutes = day_start[0] * 60 + day_start[1]
    night_minutes = night_start[0] * 60 + night_start[1]
    if day_minutes == night_minutes:
        return True
    if day_minutes < night_minutes:
        return day_minutes <= current_minutes < night_minutes
    return current_minutes >= day_minutes or current_minutes < night_minutes


def _next_run_time(
    moment: datetime,
    *,
    day_start: tuple[int, int],
    night_start: tuple[int, int],
    day_interval_minutes: int,
    night_interval_minutes: int,
) -> datetime:
    interval_minutes = (
        day_interval_minutes
        if _is_daytime(moment, day_start=day_start, night_start=night_start)
        else night_interval_minutes
    )
    rounded = moment.replace(second=0, microsecond=0)
    minutes_since_midnight = rounded.hour * 60 + rounded.minute
    next_bucket = ((minutes_since_midnight // interval_minutes) + 1) * interval_minutes
    day_start_time = rounded.replace(hour=0, minute=0)
    return day_start_time + timedelta(minutes=next_bucket)


def _build_service(project_root: Path) -> CameraMonitorService:
    settings = load_settings(project_root, require_telegram_token=False)
    catalog_client = UfanetCatalogClient(
        settings.catalog_url,
        settings.user_agent,
        cache_path=project_root / "runtime" / "ufanet_catalog_cache.json",
        request_timeout_seconds=settings.request_timeout_seconds,
    )
    camera_configs = load_camera_configs(settings.camera_config_path, settings)
    cameras = [camera for camera in resolve_cameras(camera_configs, catalog_client) if camera.enabled]
    return CameraMonitorService(
        settings=settings,
        repository=SQLiteRepository(settings.database_path),
        detector=ParkingSpaceDetector(settings.model_path, image_size=max(640, settings.image_width)),
        vehicle_detector=None,
        slot_classifier=None,
        catalog_client=catalog_client,
        cameras=cameras,
    )


def _select_cameras(
    service: CameraMonitorService,
    *,
    selected_ids: set[str],
    include_roi_only: bool,
) -> list[ResolvedCamera]:
    selected: list[ResolvedCamera] = []
    for camera in service.cameras:
        if selected_ids and camera.id not in selected_ids:
            continue
        if camera.parking_slots:
            selected.append(camera)
            continue
        if include_roi_only and (camera.detection_roi is not None or camera.detection_exclude_rois):
            selected.append(camera)
    return selected


def _camera_timestamp(now: datetime) -> str:
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def _compute_fingerprint(image_path: Path, *, hash_size: int) -> FrameFingerprint:
    cv2 = _require_cv2()
    np = _require_numpy()
    payload = image_path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode image from {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    dhash = 0
    for bit in diff.flatten():
        dhash = (dhash << 1) | int(bool(bit))

    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return FrameFingerprint(
        sha256=hashlib.sha256(payload).hexdigest(),
        dhash=dhash,
        mean_abs_diff_key=[int(value) for value in thumb.flatten().tolist()],
    )


def _hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _mean_abs_diff(left: list[int], right: list[int]) -> float:
    if len(left) != len(right):
        return float("inf")
    total = 0
    for lhs, rhs in zip(left, right):
        total += abs(lhs - rhs)
    return total / max(1, len(left))


def _is_duplicate(
    current: FrameFingerprint,
    previous: FrameFingerprint,
    *,
    dhash_threshold: int,
    mean_abs_diff_threshold: float,
) -> tuple[bool, int, float]:
    if current.sha256 == previous.sha256:
        return (True, 0, 0.0)
    hamming = _hamming_distance(current.dhash, previous.dhash)
    mean_abs_diff = _mean_abs_diff(current.mean_abs_diff_key, previous.mean_abs_diff_key)
    is_duplicate = hamming <= dhash_threshold and mean_abs_diff <= mean_abs_diff_threshold
    return (is_duplicate, hamming, mean_abs_diff)


def _load_archive_state(
    metadata_path: Path,
    *,
    hash_size: int,
) -> dict[str, CameraArchiveState]:
    states: dict[str, CameraArchiveState] = {}
    if not metadata_path.exists():
        return states

    for raw_line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") != "saved":
            continue
        camera_id = str(payload.get("camera_id", ""))
        saved_path = Path(str(payload.get("saved_path", "")))
        if not camera_id or not saved_path.exists():
            continue
        fingerprint_payload = payload.get("fingerprint")
        if not isinstance(fingerprint_payload, dict):
            try:
                fingerprint = _compute_fingerprint(saved_path, hash_size=hash_size)
            except Exception:
                continue
        else:
            try:
                fingerprint = FrameFingerprint(
                    sha256=str(fingerprint_payload["sha256"]),
                    dhash=int(fingerprint_payload["dhash"]),
                    mean_abs_diff_key=[int(item) for item in fingerprint_payload["mean_abs_diff_key"]],
                )
            except (KeyError, TypeError, ValueError):
                try:
                    fingerprint = _compute_fingerprint(saved_path, hash_size=hash_size)
                except Exception:
                    continue
        observed_at_text = payload.get("observed_at")
        try:
            observed_at = datetime.fromisoformat(str(observed_at_text))
        except ValueError:
            observed_at = datetime.fromtimestamp(saved_path.stat().st_mtime)
        states[camera_id] = CameraArchiveState(
            fingerprint=fingerprint,
            saved_path=saved_path,
            observed_at=observed_at,
        )
    logger.info("Loaded archive state for %s cameras from %s", len(states), metadata_path)
    return states


def _append_metadata(metadata_path: Path, row: dict[str, object]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _fetch_camera_frame(
    service: CameraMonitorService,
    camera: ResolvedCamera,
) -> tuple[Path, datetime | None]:
    if camera.source_transport == "hls":
        return service._fetch_hls_frame(camera)
    raw_frame_path, _payload, source_updated_at = service._fetch_snapshot(camera)
    return raw_frame_path, source_updated_at


def _archive_round(
    *,
    service: CameraMonitorService,
    cameras: list[ResolvedCamera],
    output_dir: Path,
    metadata_path: Path,
    states: dict[str, CameraArchiveState],
    hash_size: int,
    dhash_threshold: int,
    mean_abs_diff_threshold: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=service.timezone)
    logger.info("Archival round started. camera_count=%s output_dir=%s", len(cameras), output_dir)

    for camera in cameras:
        try:
            raw_frame_path, source_updated_at = _fetch_camera_frame(service, camera)
            fingerprint = _compute_fingerprint(raw_frame_path, hash_size=hash_size)
            previous = states.get(camera.id)
            hamming = None
            mean_abs_diff = None
            if previous is not None:
                is_duplicate, hamming, mean_abs_diff = _is_duplicate(
                    fingerprint,
                    previous.fingerprint,
                    dhash_threshold=dhash_threshold,
                    mean_abs_diff_threshold=mean_abs_diff_threshold,
                )
                if is_duplicate:
                    row = {
                        "event": "skipped_duplicate",
                        "camera_id": camera.id,
                        "display_name": camera.display_name,
                        "observed_at": now.isoformat(),
                        "source_updated_at": source_updated_at.isoformat() if source_updated_at else None,
                        "raw_frame_path": str(raw_frame_path),
                        "matched_saved_path": str(previous.saved_path),
                        "dhash_distance": hamming,
                        "mean_abs_diff": mean_abs_diff,
                    }
                    _append_metadata(metadata_path, row)
                    logger.info(
                        "Skipped duplicate frame. camera_id=%s matched=%s dhash=%s mean_abs_diff=%.3f",
                        camera.id,
                        previous.saved_path.name,
                        hamming,
                        mean_abs_diff,
                    )
                    continue

            camera_dir = output_dir / camera.id
            camera_dir.mkdir(parents=True, exist_ok=True)
            target_path = camera_dir / f"{_camera_timestamp(now)}.jpg"
            shutil.copy2(raw_frame_path, target_path)
            states[camera.id] = CameraArchiveState(
                fingerprint=fingerprint,
                saved_path=target_path,
                observed_at=now,
            )
            row = {
                "event": "saved",
                "camera_id": camera.id,
                "display_name": camera.display_name,
                "observed_at": now.isoformat(),
                "source_updated_at": source_updated_at.isoformat() if source_updated_at else None,
                "saved_path": str(target_path),
                "raw_frame_path": str(raw_frame_path),
                "source_transport": camera.source_transport,
                "slot_count": len(camera.parking_slots),
                "fingerprint": {
                    "sha256": fingerprint.sha256,
                    "dhash": str(fingerprint.dhash),
                    "mean_abs_diff_key": fingerprint.mean_abs_diff_key,
                },
                "dhash_distance_to_previous": hamming,
                "mean_abs_diff_to_previous": mean_abs_diff,
            }
            _append_metadata(metadata_path, row)
            logger.info(
                "Saved frame. camera_id=%s path=%s source_transport=%s slot_count=%s",
                camera.id,
                target_path,
                camera.source_transport,
                len(camera.parking_slots),
            )
        except Exception as exc:
            row = {
                "event": "error",
                "camera_id": camera.id,
                "display_name": camera.display_name,
                "observed_at": now.isoformat(),
                "error": str(exc),
            }
            _append_metadata(metadata_path, row)
            logger.exception("Camera archival failed. camera_id=%s error=%s", camera.id, exc)


def main() -> None:
    args = parse_args()
    raw_project_root = Path(args.project_root)
    if raw_project_root.is_absolute():
        project_root = raw_project_root.resolve()
    else:
        project_root = (PROJECT_ROOT / raw_project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    metadata_path = (project_root / args.metadata_path).resolve()
    _configure_logging(
        project_root=project_root,
        log_file=args.log_file,
        log_level=args.log_level,
    )
    logger.info("Using project_root=%s", project_root)

    if args.day_interval_minutes <= 0 or args.night_interval_minutes <= 0:
        raise RuntimeError("Intervals must be positive.")
    if args.hash_size < 4:
        raise RuntimeError("hash_size must be at least 4.")

    day_start = _parse_hhmm(args.day_start)
    night_start = _parse_hhmm(args.night_start)

    service = _build_service(project_root)
    timezone = resolve_timezone(service.settings.timezone)
    selected_ids = set(args.cameras or [])
    cameras = _select_cameras(
        service,
        selected_ids=selected_ids,
        include_roi_only=bool(args.include_roi_only),
    )
    if not cameras:
        raise RuntimeError("No cameras matched the selection criteria.")

    states = _load_archive_state(metadata_path, hash_size=args.hash_size)
    logger.info("Selected cameras: %s", ", ".join(camera.id for camera in cameras))

    while True:
        started_at = datetime.now(tz=timezone)
        logger.info("Starting archival round at %s", started_at.isoformat())
        _archive_round(
            service=service,
            cameras=cameras,
            output_dir=output_dir,
            metadata_path=metadata_path,
            states=states,
            hash_size=args.hash_size,
            dhash_threshold=args.dhash_threshold,
            mean_abs_diff_threshold=args.mean_abs_diff_threshold,
        )
        if args.run_once:
            return

        next_run = _next_run_time(
            datetime.now(tz=timezone),
            day_start=day_start,
            night_start=night_start,
            day_interval_minutes=args.day_interval_minutes,
            night_interval_minutes=args.night_interval_minutes,
        )
        sleep_seconds = max(1.0, (next_run - datetime.now(tz=timezone)).total_seconds())
        logger.info(
            "Next archival round at %s (sleep %.0fs)",
            next_run.isoformat(),
            sleep_seconds,
        )
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
