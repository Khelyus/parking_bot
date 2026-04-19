from datetime import datetime

from parking_bot.time_utils import resolve_timezone
from parking_bot.types import TimeWindow


def test_time_window_contains_daytime_times() -> None:
    window = TimeWindow.parse("18:00-19:00")
    tz = resolve_timezone("Europe/Moscow")
    assert window.contains(datetime(2026, 4, 5, 18, 30, tzinfo=tz))
    assert not window.contains(datetime(2026, 4, 5, 19, 0, tzinfo=tz))


def test_time_window_supports_overnight_ranges() -> None:
    window = TimeWindow.parse("23:00-02:00")
    tz = resolve_timezone("Europe/Moscow")
    assert window.contains(datetime(2026, 4, 5, 23, 30, tzinfo=tz))
    assert window.contains(datetime(2026, 4, 6, 1, 0, tzinfo=tz))
    assert not window.contains(datetime(2026, 4, 6, 3, 0, tzinfo=tz))
