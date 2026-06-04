from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_bot.camera_catalog import UfanetCatalogClient, resolve_cameras
from parking_bot.config_loader import load_camera_configs
from parking_bot.detector import ParkingSpaceDetector, VehicleDetector
from parking_bot.repository import SQLiteRepository
from parking_bot.service import CameraMonitorService
from parking_bot.settings import load_settings
from parking_bot.slot_classifier import SlotStatusClassifier


def _require_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required to export calibration crops.") from exc
    return cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pseudo-labeled slot calibration dataset from archived training frames."
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root. Relative paths are resolved from the repository root, not from cwd.",
    )
    parser.add_argument("--frames-dir", default="runtime/training_frames")
    parser.add_argument("--output-dir", default="runtime/slot_calibration_dataset")
    parser.add_argument(
        "--cameras",
        nargs="*",
        help="Optional camera ids to include. Defaults to all enabled slot cameras with archived frames.",
    )
    parser.add_argument(
        "--min-empty-probability",
        type=float,
        default=0.82,
        help="Minimum final confidence for confidently empty slot samples.",
    )
    parser.add_argument(
        "--min-occupied-probability",
        type=float,
        default=0.70,
        help="Minimum final confidence for confidently occupied slot samples.",
    )
    parser.add_argument(
        "--max-empty-vehicle-score",
        type=float,
        default=0.12,
        help="Maximum vehicle score still considered confidently empty.",
    )
    parser.add_argument(
        "--limit-per-camera",
        type=int,
        default=0,
        help="Optional cap on archived frames per camera. 0 means no cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output class folders before exporting a fresh dataset.",
    )
    return parser.parse_args()


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
        detector=ParkingSpaceDetector(settings.model_path),
        vehicle_detector=VehicleDetector(settings.vehicle_model_path),
        slot_classifier=SlotStatusClassifier(
            settings.slot_classifier_model_path,
            image_size=settings.slot_classifier_image_size,
        ),
        catalog_client=catalog_client,
        cameras=cameras,
    )


def _pseudo_label_slot(
    *,
    service: CameraMonitorService,
    image: Any,
    slot_box: tuple[int, int, int, int],
    vehicles: list[Any],
    min_empty_probability: float,
    min_occupied_probability: float,
    max_empty_vehicle_score: float,
) -> tuple[str, float, dict[str, float]] | None:
    prediction = service._predict_slot_status(image, slot_box)
    if prediction is None:
        return None

    matched_vehicle = service._match_vehicle_to_slot(slot_box, vehicles)
    vehicle_score = (
        service._slot_vehicle_match_score(slot_box, matched_vehicle.box)
        if matched_vehicle is not None
        else 0.0
    )
    local_vehicle_score = service._detect_local_vehicle_score(
        image,
        slot_box,
        base_vehicle_score=vehicle_score,
        slot_prediction=prediction,
    )
    label, confidence = service._resolve_slot_status(
        vehicle_score=vehicle_score,
        local_vehicle_score=local_vehicle_score,
        slot_prediction=prediction,
    )
    metrics = {
        "empty_probability": prediction.empty_probability,
        "occupied_probability": prediction.occupied_probability,
        "vehicle_score": vehicle_score,
        "local_vehicle_score": local_vehicle_score,
    }

    if label == "space-occupied" and confidence >= min_occupied_probability:
        return ("space-occupied", confidence, metrics)

    if (
        label == "space-empty"
        and confidence >= min_empty_probability
        and vehicle_score <= max_empty_vehicle_score
        and local_vehicle_score <= max_empty_vehicle_score
    ):
        return ("space-empty", confidence, metrics)

    return None


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if overwrite and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                for nested_dir in sorted(
                    [path for path in child.rglob("*") if path.is_dir()],
                    reverse=True,
                ):
                    nested_dir.rmdir()
                child.rmdir()
            elif child.name == "metadata.json":
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    raw_project_root = Path(args.project_root)
    if raw_project_root.is_absolute():
        project_root = raw_project_root.resolve()
    else:
        project_root = (PROJECT_ROOT / raw_project_root).resolve()
    frames_dir = (project_root / args.frames_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    if not frames_dir.exists():
        raise RuntimeError(f"Archived frames directory was not found: {frames_dir}")

    _prepare_output_dir(output_dir, overwrite=bool(args.overwrite))

    cv2 = _require_cv2()
    service = _build_service(project_root)
    selected_cameras = set(args.cameras or [])
    per_camera_limit = args.limit_per_camera if args.limit_per_camera > 0 else None

    metadata: list[dict[str, object]] = []
    exported = 0
    skipped = 0
    class_counts = {"space-empty": 0, "space-occupied": 0}

    for camera in service.cameras:
        if not camera.parking_slots:
            continue
        if selected_cameras and camera.id not in selected_cameras:
            continue

        camera_frames_dir = frames_dir / camera.id
        if not camera_frames_dir.exists():
            continue

        frame_paths = sorted(path for path in camera_frames_dir.iterdir() if path.is_file())
        if per_camera_limit is not None:
            frame_paths = frame_paths[:per_camera_limit]
        if not frame_paths:
            continue

        print(f"Processing {camera.id}: {len(frame_paths)} archived frames")
        for frame_path in frame_paths:
            image = service._load_frame_image(frame_path)
            vehicles = (
                service.vehicle_detector.detect_image(
                    image,
                    confidence=service.settings.vehicle_confidence,
                    image_size=max(1280, camera.detection_image_size or 0) or None,
                )
                if service.vehicle_detector is not None
                else []
            )

            for slot in camera.parking_slots:
                slot_box = service._resolve_slot_geometry(image.shape[:2], slot)
                labeled = _pseudo_label_slot(
                    service=service,
                    image=image,
                    slot_box=slot_box,
                    vehicles=vehicles,
                    min_empty_probability=args.min_empty_probability,
                    min_occupied_probability=args.min_occupied_probability,
                    max_empty_vehicle_score=args.max_empty_vehicle_score,
                )
                if labeled is None:
                    skipped += 1
                    continue

                label, confidence, metrics = labeled
                crop_box = service._expand_slot_crop_box(slot_box, image.shape[:2])
                crop = service._extract_slot_crop(image, slot_box)
                if crop.size == 0:
                    skipped += 1
                    continue

                class_dir = output_dir / label
                class_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{camera.id}__{slot.id}__{frame_path.stem}.jpg"
                target_path = class_dir / filename
                cv2.imwrite(str(target_path), crop)

                metadata.append(
                    {
                        "camera_id": camera.id,
                        "slot_id": slot.id,
                        "frame_path": str(frame_path),
                        "exported_path": str(target_path),
                        "label": label,
                        "confidence": confidence,
                        "slot_box": slot_box.bounds,
                        "crop_box": crop_box,
                        "angle_degrees": slot.angle_degrees,
                        **metrics,
                    }
                )
                class_counts[label] += 1
                exported += 1

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "exported": exported,
                "skipped": skipped,
                "class_counts": class_counts,
                "samples": metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Exported {exported} calibration crops to {output_dir} "
        f"(empty={class_counts['space-empty']}, occupied={class_counts['space-occupied']}, skipped={skipped})"
    )


if __name__ == "__main__":
    main()
