from __future__ import annotations

import argparse
import logging
from pathlib import Path

from parking_bot.camera_catalog import UfanetCatalogClient, resolve_cameras
from parking_bot.config_loader import load_camera_configs
from parking_bot.detector import ParkingSpaceDetector, VehicleDetector
from parking_bot.repository import SQLiteRepository
from parking_bot.service import CameraMonitorService
from parking_bot.settings import load_settings
from parking_bot.slot_classifier import SlotStatusClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the parking notification bot.")
    parser.add_argument(
        "--config",
        help="Path to a camera config yaml relative to the project root or absolute path.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the stable demo camera subset from config/cameras.demo.yaml.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level, for example INFO or DEBUG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    settings = load_settings()
    if args.demo:
        settings.camera_config_path = settings.project_dir / "config/cameras.demo.yaml"
    elif args.config:
        candidate = Path(args.config)
        settings.camera_config_path = (
            candidate if candidate.is_absolute() else settings.project_dir / candidate
        )

    camera_configs = load_camera_configs(settings.camera_config_path, settings)
    catalog_client = UfanetCatalogClient(settings.catalog_url, settings.user_agent)
    cameras = [camera for camera in resolve_cameras(camera_configs, catalog_client) if camera.enabled]
    if not cameras:
        raise RuntimeError("No enabled cameras were found in the configuration.")
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

    try:
        from telegram import Update
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "python-telegram-bot is not installed. Run 'python -m pip install -r requirements.txt' first."
        ) from exc

    from parking_bot.telegram_bot import build_application

    application = build_application(settings, service)
    application.run_polling(allowed_updates=Update.ALL_TYPES)
