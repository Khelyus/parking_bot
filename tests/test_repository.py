from datetime import datetime

from parking_bot.repository import SQLiteRepository
from parking_bot.types import Availability, TimeWindow


def test_mark_camera_subscriptions_state_updates_all_rows(tmp_path) -> None:
    repo = SQLiteRepository(tmp_path / "bot.db")
    created_at = datetime(2026, 4, 5, 10, 0).astimezone()

    repo.upsert_subscription(
        chat_id=1,
        camera_id="cam-1",
        time_window=None,
        created_at=created_at,
    )
    repo.upsert_subscription(
        chat_id=2,
        camera_id="cam-1",
        time_window=TimeWindow.parse("18:00-19:00"),
        created_at=created_at,
    )

    repo.mark_camera_subscriptions_state(
        camera_id="cam-1",
        availability=Availability.FULL,
    )

    subscriptions = repo.list_camera_subscriptions("cam-1")
    assert len(subscriptions) == 2
    assert all(subscription.last_notified_state == Availability.FULL for subscription in subscriptions)
