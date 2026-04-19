import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from parking_bot.repository import SQLiteRepository
from parking_bot.service import CameraMonitorService
from parking_bot.time_utils import resolve_timezone
from parking_bot.types import Availability, CameraObservation, ResolvedCamera


def _build_camera() -> ResolvedCamera:
    return ResolvedCamera(
        id="cam-1",
        map_url="http://example.com/cam-1",
        number="1",
        server="example.com",
        token="token",
        display_name="Cam 1",
        stable_cycles=2,
        poll_interval_seconds=30,
    )


def _build_settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        timezone="Europe/Moscow",
        monitor_workers=1,
        poll_interval_seconds=30,
        notification_stable_cycles=4,
        notification_cooldown_seconds=900,
        frames_dir=tmp_path / "frames",
        hls_burst_frames=1,
        hls_burst_pause_seconds=0.0,
    )


def _observation(
    camera: ResolvedCamera,
    tz_name: str,
    availability: Availability,
    second: int,
) -> CameraObservation:
    tz = resolve_timezone(tz_name)
    free_count = 1 if availability == Availability.FREE else 0
    occupied_count = 1 if availability == Availability.FULL else 0
    observed_at = datetime(2026, 4, 15, 10, 0, 0, tzinfo=tz) + timedelta(seconds=second)
    return CameraObservation(
        camera_id=camera.id,
        display_name=camera.display_name,
        observed_at=observed_at,
        source_updated_at=None,
        availability=availability,
        free_count=free_count,
        occupied_count=occupied_count,
        detections=[],
    )


def test_notification_requires_more_stable_frames_than_ui_status(tmp_path) -> None:
    async def _run() -> None:
        settings = _build_settings(tmp_path)
        settings.frames_dir.mkdir(parents=True, exist_ok=True)
        repository = SQLiteRepository(tmp_path / "bot.db")
        camera = _build_camera()
        service = CameraMonitorService(
            settings=settings,
            repository=repository,
            detector=object(),
            vehicle_detector=None,
            slot_classifier=None,
            catalog_client=object(),
            cameras=[camera],
        )
        notifications: list[Availability] = []

        async def _capture_notification(camera_arg, observation_arg) -> None:
            notifications.append(observation_arg.availability)

        service._notify_subscribers = _capture_notification  # type: ignore[method-assign]
        observations = iter(
            [
                _observation(camera, settings.timezone, Availability.FULL, 0),
                _observation(camera, settings.timezone, Availability.FULL, 10),
                _observation(camera, settings.timezone, Availability.FULL, 20),
                _observation(camera, settings.timezone, Availability.FULL, 30),
                _observation(camera, settings.timezone, Availability.FREE, 40),
                _observation(camera, settings.timezone, Availability.FREE, 50),
                _observation(camera, settings.timezone, Availability.FREE, 60),
                _observation(camera, settings.timezone, Availability.FREE, 70),
            ]
        )

        service._fetch_and_detect = lambda _camera: next(observations)  # type: ignore[method-assign]

        for _ in range(6):
            await service._poll_camera(camera, notify=True)

        status_after_two_free = repository.get_camera_status(camera.id)
        assert status_after_two_free is not None
        assert status_after_two_free.stable_availability == Availability.FREE
        assert notifications == []

        await service._poll_camera(camera, notify=True)
        assert notifications == []

        await service._poll_camera(camera, notify=True)
        assert notifications == [Availability.FREE]

        await service.stop()

    asyncio.run(_run())
