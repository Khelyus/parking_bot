from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_bot.camera_catalog import UfanetCatalogClient, resolve_cameras
from parking_bot.config_loader import load_camera_configs
from parking_bot.repository import SQLiteRepository
from parking_bot.service import CameraMonitorService
from parking_bot.settings import load_settings
from parking_bot.state_machine import AvailabilityStabilizer
from parking_bot.time_utils import resolve_timezone
from parking_bot.types import Availability, TimeWindow


def _assert_time_window() -> None:
    window = TimeWindow.parse("18:00-19:00")
    tz = resolve_timezone("Europe/Moscow")
    assert window.contains(datetime(2026, 4, 5, 18, 30, tzinfo=tz))
    assert not window.contains(datetime(2026, 4, 5, 19, 0, tzinfo=tz))


def _assert_stabilizer() -> None:
    stabilizer = AvailabilityStabilizer(stable_cycles=2)
    assert stabilizer.ingest(Availability.FULL) is None
    assert stabilizer.ingest(Availability.FULL) == (Availability.UNKNOWN, Availability.FULL)
    assert stabilizer.ingest(Availability.FREE) is None
    assert stabilizer.ingest(Availability.FREE) == (Availability.FULL, Availability.FREE)


def _assert_repository(tmp_dir: Path) -> None:
    repository = SQLiteRepository(tmp_dir / "bot.db")
    created_at = datetime(2026, 4, 5, 10, 0).astimezone()
    repository.upsert_subscription(
        chat_id=1,
        camera_id="cam-1",
        time_window=None,
        created_at=created_at,
    )
    repository.upsert_subscription(
        chat_id=2,
        camera_id="cam-1",
        time_window=TimeWindow.parse("18:00-19:00"),
        created_at=created_at,
    )
    repository.mark_camera_subscriptions_state(
        camera_id="cam-1",
        availability=Availability.FULL,
    )
    subscriptions = repository.list_camera_subscriptions("cam-1")
    assert len(subscriptions) == 2
    assert all(subscription.last_notified_state == Availability.FULL for subscription in subscriptions)


class _DetectorStub:
    def detect(self, *_args, **_kwargs):  # pragma: no cover - demo cameras do not call it
        raise AssertionError("Detector should not be called for demo cameras")


async def _assert_demo_service(tmp_dir: Path) -> None:
    settings = load_settings(PROJECT_ROOT, require_telegram_token=False)
    settings.database_path = tmp_dir / "demo.db"
    settings.frames_dir = tmp_dir / "frames"
    settings.camera_config_path = PROJECT_ROOT / "config/cameras.demo.yaml"
    settings.frames_dir.mkdir(parents=True, exist_ok=True)

    camera_configs = load_camera_configs(settings.camera_config_path, settings)
    catalog_client = UfanetCatalogClient(settings.catalog_url, settings.user_agent)
    cameras = resolve_cameras(camera_configs, catalog_client)

    repository = SQLiteRepository(settings.database_path)
    service = CameraMonitorService(
        settings=settings,
        repository=repository,
        detector=_DetectorStub(),
        catalog_client=catalog_client,
        cameras=cameras,
    )
    try:
        demo_camera = next(camera for camera in cameras if camera.id == "demo_alert_camera")
        statuses = []
        for _ in range(4):
            status = await service.refresh_camera(demo_camera.id, notify=False)
            statuses.append(status.current_availability)
        assert statuses == [
            Availability.FULL,
            Availability.FULL,
            Availability.FREE,
            Availability.FREE,
        ]
        final_status = repository.get_camera_status(demo_camera.id)
        assert final_status is not None
        assert final_status.stable_availability == Availability.FREE
    finally:
        await service.stop()


def main() -> None:
    tmp_dir = PROJECT_ROOT / "runtime" / "self_check_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        _assert_time_window()
        _assert_stabilizer()
        _assert_repository(tmp_dir)
        asyncio.run(_assert_demo_service(tmp_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Self-checks passed.")


if __name__ == "__main__":
    main()
