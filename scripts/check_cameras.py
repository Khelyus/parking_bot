from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_bot.camera_catalog import UfanetCatalogClient, resolve_cameras
from parking_bot.config_loader import load_camera_configs
from parking_bot.detector import ParkingSpaceDetector, VehicleDetector
from parking_bot.repository import SQLiteRepository
from parking_bot.service import CameraMonitorService
from parking_bot.settings import load_settings
from parking_bot.slot_classifier import SlotStatusClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a smoke-check for all configured cameras and save a JSON report."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the offline demo camera config instead of the live Ufanet cameras.",
    )
    parser.add_argument(
        "--output",
        default="runtime/camera_report.json",
        help="Where to save the JSON summary.",
    )
    return parser.parse_args()


def _copy_report_frame(
    source_path: Path | None,
    *,
    target_dir: Path,
    camera_id: str,
    suffix: str,
) -> Path | None:
    if source_path is None or not source_path.exists():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{camera_id}_{suffix}{source_path.suffix.lower() or '.jpg'}"
    shutil.copy2(source_path, target_path)
    return target_path


async def build_report(output_path: Path, *, use_demo: bool) -> None:
    settings = load_settings(PROJECT_ROOT, require_telegram_token=False)
    if use_demo:
        settings.camera_config_path = settings.project_dir / "config/cameras.demo.yaml"
    catalog_client = UfanetCatalogClient(settings.catalog_url, settings.user_agent)
    camera_configs = load_camera_configs(settings.camera_config_path, settings)
    cameras = [camera for camera in resolve_cameras(camera_configs, catalog_client) if camera.enabled]
    repository = SQLiteRepository(settings.database_path)
    detector = ParkingSpaceDetector(settings.model_path, image_size=max(640, settings.image_width))
    vehicle_detector = VehicleDetector(settings.vehicle_model_path)
    slot_classifier = (
        SlotStatusClassifier(
            settings.slot_classifier_model_path,
            image_size=settings.slot_classifier_image_size,
        )
        if settings.slot_classifier_model_path.exists()
        else None
    )
    service = CameraMonitorService(
        settings=settings,
        repository=repository,
        detector=detector,
        vehicle_detector=vehicle_detector,
        slot_classifier=slot_classifier,
        catalog_client=catalog_client,
        cameras=cameras,
    )

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    assets_dir = output_path.parent / f"{output_path.stem}_assets"

    rows: list[dict[str, object]] = []
    try:
        for camera in cameras:
            status = await service.refresh_camera(camera.id, notify=False)
            copied_annotated_path = _copy_report_frame(
                status.annotated_frame_path,
                target_dir=assets_dir,
                camera_id=camera.id,
                suffix="annotated",
            )
            copied_raw_path = _copy_report_frame(
                status.raw_frame_path,
                target_dir=assets_dir,
                camera_id=camera.id,
                suffix="raw",
            )
            rows.append(
                {
                    "camera_id": camera.id,
                    "display_name": camera.display_name,
                    "map_url": camera.map_url,
                    "current_availability": status.current_availability.value,
                    "stable_availability": status.stable_availability.value,
                    "free_count": status.free_count,
                    "occupied_count": status.occupied_count,
                    "observed_at": status.observed_at.isoformat() if status.observed_at else None,
                    "annotated_frame_path": (
                        str(copied_annotated_path) if copied_annotated_path else None
                    ),
                    "raw_frame_path": str(copied_raw_path) if copied_raw_path else None,
                    "last_error": status.last_error,
                }
            )
    finally:
        await service.stop()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved camera report to: {output_path}")
    print(f"Saved report frames to: {assets_dir}")
    for row in rows:
        print(
            f"{row['camera_id']}: {row['current_availability']}"
            f" | free={row['free_count']}"
            f" | occupied={row['occupied_count']}"
        )


def main() -> None:
    args = parse_args()
    asyncio.run(build_report(Path(args.output), use_demo=args.demo))


if __name__ == "__main__":
    main()
