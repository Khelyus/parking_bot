from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from threading import Lock

from parking_bot.types import Availability, CameraStatusRecord, Subscription, TimeWindow


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _parse_path(value: str | None) -> Path | None:
    return Path(value) if value else None


class SQLiteRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self._lock = Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    camera_id TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    window_key TEXT NOT NULL,
                    window_start TEXT,
                    window_end TEXT,
                    last_notified_state TEXT,
                    last_notified_at TEXT,
                    UNIQUE(chat_id, camera_id, window_key)
                );

                CREATE TABLE IF NOT EXISTS camera_status (
                    camera_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    observed_at TEXT,
                    source_updated_at TEXT,
                    current_availability TEXT NOT NULL,
                    stable_availability TEXT NOT NULL,
                    stable_cycles INTEGER NOT NULL DEFAULT 0,
                    free_count INTEGER NOT NULL DEFAULT 0,
                    occupied_count INTEGER NOT NULL DEFAULT 0,
                    raw_frame_path TEXT,
                    annotated_frame_path TEXT,
                    last_error TEXT
                );
                """
            )
            self._ensure_camera_status_columns(connection)

    def _ensure_camera_status_columns(self, connection: sqlite3.Connection) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(camera_status)").fetchall()
        }
        if "source_updated_at" not in columns:
            connection.execute("ALTER TABLE camera_status ADD COLUMN source_updated_at TEXT")

    def upsert_subscription(
        self,
        *,
        chat_id: int,
        camera_id: str,
        time_window: TimeWindow | None,
        created_at: datetime,
    ) -> Subscription:
        window_key = time_window.key if time_window else "always"
        window_start = time_window.start.strftime("%H:%M") if time_window else None
        window_end = time_window.end.strftime("%H:%M") if time_window else None
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO subscriptions (
                    chat_id, camera_id, active, created_at, updated_at,
                    window_key, window_start, window_end
                )
                VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, camera_id, window_key) DO UPDATE SET
                    active = 1,
                    updated_at = excluded.updated_at,
                    window_start = excluded.window_start,
                    window_end = excluded.window_end
                """,
                (
                    chat_id,
                    camera_id,
                    _iso(created_at),
                    _iso(created_at),
                    window_key,
                    window_start,
                    window_end,
                ),
            )
            row = connection.execute(
                """
                SELECT *
                FROM subscriptions
                WHERE chat_id = ? AND camera_id = ? AND window_key = ?
                """,
                (chat_id, camera_id, window_key),
            ).fetchone()
        return self._row_to_subscription(row)

    def list_subscriptions(self, chat_id: int, *, active_only: bool = True) -> list[Subscription]:
        query = """
            SELECT *
            FROM subscriptions
            WHERE chat_id = ?
        """
        params: list[object] = [chat_id]
        if active_only:
            query += " AND active = 1"
        query += " ORDER BY camera_id, window_key"
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_subscription(row) for row in rows]

    def list_camera_subscriptions(self, camera_id: str) -> list[Subscription]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM subscriptions
                WHERE camera_id = ? AND active = 1
                ORDER BY updated_at
                """,
                (camera_id,),
            ).fetchall()
        return [self._row_to_subscription(row) for row in rows]

    def deactivate_subscription(self, chat_id: int, subscription_id: int) -> bool:
        with self._lock, self._connect() as connection:
            result = connection.execute(
                """
                UPDATE subscriptions
                SET active = 0, updated_at = ?
                WHERE id = ? AND chat_id = ?
                """,
                (_iso(datetime.now().astimezone()), subscription_id, chat_id),
            )
        return result.rowcount > 0

    def mark_camera_subscriptions_state(
        self,
        *,
        camera_id: str,
        availability: Availability,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE subscriptions
                SET last_notified_state = ?
                WHERE camera_id = ?
                """,
                (availability.value, camera_id),
            )

    def mark_notification(
        self,
        *,
        subscription_id: int,
        availability: Availability,
        sent_at: datetime,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE subscriptions
                SET last_notified_state = ?, last_notified_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (availability.value, _iso(sent_at), _iso(sent_at), subscription_id),
            )

    def save_camera_status(self, record: CameraStatusRecord) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO camera_status (
                    camera_id, display_name, observed_at, source_updated_at, current_availability,
                    stable_availability, stable_cycles, free_count, occupied_count,
                    raw_frame_path, annotated_frame_path, last_error
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    observed_at = excluded.observed_at,
                    source_updated_at = excluded.source_updated_at,
                    current_availability = excluded.current_availability,
                    stable_availability = excluded.stable_availability,
                    stable_cycles = excluded.stable_cycles,
                    free_count = excluded.free_count,
                    occupied_count = excluded.occupied_count,
                    raw_frame_path = excluded.raw_frame_path,
                    annotated_frame_path = excluded.annotated_frame_path,
                    last_error = excluded.last_error
                """,
                (
                    record.camera_id,
                    record.display_name,
                    _iso(record.observed_at),
                    _iso(record.source_updated_at),
                    record.current_availability.value,
                    record.stable_availability.value,
                    record.stable_cycles,
                    record.free_count,
                    record.occupied_count,
                    str(record.raw_frame_path) if record.raw_frame_path else None,
                    str(record.annotated_frame_path) if record.annotated_frame_path else None,
                    record.last_error,
                ),
            )

    def get_camera_status(self, camera_id: str) -> CameraStatusRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM camera_status
                WHERE camera_id = ?
                """,
                (camera_id,),
            ).fetchone()
        return self._row_to_status(row) if row else None

    def _row_to_subscription(self, row: sqlite3.Row) -> Subscription:
        if row["window_start"] and row["window_end"]:
            time_window = TimeWindow.parse(f"{row['window_start']}-{row['window_end']}")
        else:
            time_window = None
        return Subscription(
            id=int(row["id"]),
            chat_id=int(row["chat_id"]),
            camera_id=str(row["camera_id"]),
            active=bool(row["active"]),
            created_at=_parse_dt(row["created_at"]) or datetime.now(),
            updated_at=_parse_dt(row["updated_at"]) or datetime.now(),
            window_key=str(row["window_key"]),
            time_window=time_window,
            last_notified_state=Availability(row["last_notified_state"])
            if row["last_notified_state"]
            else None,
            last_notified_at=_parse_dt(row["last_notified_at"]),
        )

    def _row_to_status(self, row: sqlite3.Row) -> CameraStatusRecord:
        return CameraStatusRecord(
            camera_id=str(row["camera_id"]),
            display_name=str(row["display_name"]),
            observed_at=_parse_dt(row["observed_at"]),
            source_updated_at=_parse_dt(row["source_updated_at"]) if "source_updated_at" in row.keys() else None,
            current_availability=Availability(row["current_availability"]),
            stable_availability=Availability(row["stable_availability"]),
            stable_cycles=int(row["stable_cycles"]),
            free_count=int(row["free_count"]),
            occupied_count=int(row["occupied_count"]),
            raw_frame_path=_parse_path(row["raw_frame_path"]),
            annotated_frame_path=_parse_path(row["annotated_frame_path"]),
            last_error=row["last_error"],
        )
