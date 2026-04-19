from pathlib import Path

from parking_bot.camera_catalog import UfanetCatalogClient, resolve_cameras
from parking_bot.config_loader import load_camera_configs
from parking_bot.settings import load_settings
from parking_bot.types import Availability


def test_demo_config_resolves_without_live_catalog_requests() -> None:
    project_root = Path(__file__).resolve().parents[1]
    settings = load_settings(project_root, require_telegram_token=False)
    settings.camera_config_path = project_root / "config/cameras.demo.yaml"

    camera_configs = load_camera_configs(settings.camera_config_path, settings)
    catalog_client = UfanetCatalogClient(settings.catalog_url, settings.user_agent)
    cameras = resolve_cameras(camera_configs, catalog_client)

    assert [camera.id for camera in cameras] == [
        "demo_alert_camera",
        "demo_busy_camera",
        "demo_free_camera",
    ]
    assert cameras[0].source_kind == "demo"
    assert cameras[0].demo_observations[0].availability == Availability.FULL
    assert cameras[0].demo_observations[-1].availability == Availability.FREE
