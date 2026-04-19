from datetime import datetime
from types import SimpleNamespace

from parking_bot.detector import DetectionSummary
from parking_bot.service import CameraMonitorService
from parking_bot.time_utils import resolve_timezone
from parking_bot.types import Availability, Detection


def _build_service() -> CameraMonitorService:
    service = CameraMonitorService.__new__(CameraMonitorService)
    service.settings = SimpleNamespace(timezone="Europe/Moscow")
    return service


def test_parse_source_timestamp_supports_hls_program_date_time() -> None:
    service = _build_service()

    parsed = service._parse_source_timestamp("2026-04-13T08:54:52.204Z")

    assert parsed is not None
    assert parsed.isoformat() == "2026-04-13T11:54:52.204000+03:00"


def test_parse_source_timestamp_supports_http_last_modified() -> None:
    service = _build_service()

    parsed = service._parse_source_timestamp("Mon, 13 Apr 2026 08:54:52 GMT")

    assert parsed is not None
    assert parsed.isoformat() == "2026-04-13T11:54:52+03:00"


def test_rank_detection_candidate_prefers_known_result_over_unknown() -> None:
    service = _build_service()
    tz = resolve_timezone("Europe/Moscow")
    observed_at = datetime(2026, 4, 13, 12, 0, tzinfo=tz)
    unknown = DetectionSummary(
        free_count=0,
        occupied_count=0,
        availability=Availability.UNKNOWN,
        detections=[],
        annotated_frame=None,
    )
    known = DetectionSummary(
        free_count=1,
        occupied_count=0,
        availability=Availability.FREE,
        detections=[Detection(label="space-empty", confidence=0.81, box=(1, 2, 3, 4))],
        annotated_frame=None,
    )

    unknown_score = service._rank_detection_candidate(
        unknown,
        source_updated_at=datetime(2026, 4, 13, 12, 0, 5, tzinfo=tz),
        observed_at=observed_at,
    )
    known_score = service._rank_detection_candidate(
        known,
        source_updated_at=datetime(2026, 4, 13, 12, 0, 1, tzinfo=tz),
        observed_at=observed_at,
    )

    assert known_score > unknown_score


def test_rank_detection_candidate_prefers_free_over_full() -> None:
    service = _build_service()
    tz = resolve_timezone("Europe/Moscow")
    observed_at = datetime(2026, 4, 13, 12, 0, tzinfo=tz)
    free = DetectionSummary(
        free_count=1,
        occupied_count=0,
        availability=Availability.FREE,
        detections=[Detection(label="space-empty", confidence=0.08, box=(1, 2, 3, 4))],
        annotated_frame=None,
    )
    full = DetectionSummary(
        free_count=0,
        occupied_count=4,
        availability=Availability.FULL,
        detections=[
            Detection(label="space-occupied", confidence=0.2, box=(1, 2, 3, 4)),
            Detection(label="space-occupied", confidence=0.18, box=(4, 5, 6, 7)),
        ],
        annotated_frame=None,
    )

    free_score = service._rank_detection_candidate(
        free,
        source_updated_at=datetime(2026, 4, 13, 12, 0, 1, tzinfo=tz),
        observed_at=observed_at,
    )
    full_score = service._rank_detection_candidate(
        full,
        source_updated_at=datetime(2026, 4, 13, 12, 0, 2, tzinfo=tz),
        observed_at=observed_at,
    )

    assert free_score > full_score


def test_subprocess_no_window_kwargs_hide_console_on_windows(monkeypatch) -> None:
    service = _build_service()

    class _StartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0
            self.wShowWindow = 1

    monkeypatch.setattr("parking_bot.service.os.name", "nt", raising=False)
    monkeypatch.setattr("parking_bot.service.subprocess.CREATE_NO_WINDOW", 134217728, raising=False)
    monkeypatch.setattr("parking_bot.service.subprocess.STARTF_USESHOWWINDOW", 1, raising=False)
    monkeypatch.setattr("parking_bot.service.subprocess.SW_HIDE", 0, raising=False)
    monkeypatch.setattr("parking_bot.service.subprocess.STARTUPINFO", _StartupInfo, raising=False)

    kwargs = service._subprocess_no_window_kwargs()

    assert kwargs["creationflags"] == 134217728
    assert kwargs["startupinfo"].dwFlags == 1
    assert kwargs["startupinfo"].wShowWindow == 0


def test_resolve_media_playlist_url_prefers_highest_resolution_variant() -> None:
    service = _build_service()

    playlist_url = service._resolve_media_playlist_url(
        "http://example.com/master.m3u8",
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-STREAM-INF:BANDWIDTH=900000,RESOLUTION=640x360",
                "low/index.m3u8",
                "#EXT-X-STREAM-INF:BANDWIDTH=2000000,RESOLUTION=1920x1080",
                "high/index.m3u8",
            ]
        ),
    )

    assert playlist_url == "http://example.com/high/index.m3u8"
