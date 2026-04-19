from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_bot.camera_catalog import UfanetCatalogClient, resolve_cameras
from parking_bot.config_loader import load_camera_configs
from parking_bot.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local project setup before launch.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Validate the demo camera subset instead of the config from .env.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(PROJECT_ROOT, require_telegram_token=False)
    if args.demo:
        settings.camera_config_path = settings.project_dir / "config/cameras.demo.yaml"

    print(f"Project root: {settings.project_dir}")
    print(f"Camera config: {settings.camera_config_path}")
    print(f"Model path: {settings.model_path} | exists={settings.model_path.exists()}")
    print(f"Frames dir: {settings.frames_dir} | exists={settings.frames_dir.exists()}")
    print(f"Database dir: {settings.database_path.parent} | exists={settings.database_path.parent.exists()}")
    print(
        "Telegram token present: "
        + ("yes" if settings.telegram_bot_token and "replace-with-your-token" not in settings.telegram_bot_token else "no")
    )

    camera_configs = load_camera_configs(settings.camera_config_path, settings)
    print(f"Configured cameras: {len(camera_configs)}")

    catalog_client = UfanetCatalogClient(settings.catalog_url, settings.user_agent)
    cameras = resolve_cameras(camera_configs, catalog_client)
    print(f"Resolved cameras: {len(cameras)}")
    for camera in cameras:
        print(f"  {camera.id} -> {camera.display_name} [{camera.source_kind}]")


if __name__ == "__main__":
    main()
