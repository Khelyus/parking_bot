from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from pathlib import Path
import re


class Availability(StrEnum):
    FREE = "free"
    FULL = "full"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class TimeWindow:
    start: time
    end: time

    @classmethod
    def parse(cls, value: str) -> "TimeWindow":
        match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", value)
        if not match:
            raise ValueError("Time window must look like HH:MM-HH:MM")
        start_hour, start_minute, end_hour, end_minute = map(int, match.groups())
        if start_hour > 23 or end_hour > 23 or start_minute > 59 or end_minute > 59:
            raise ValueError("Time window contains an invalid time")
        return cls(
            start=time(start_hour, start_minute),
            end=time(end_hour, end_minute),
        )

    @property
    def key(self) -> str:
        return f"{self.start.strftime('%H:%M')}-{self.end.strftime('%H:%M')}"

    def contains(self, moment: datetime) -> bool:
        current = moment.timetz().replace(tzinfo=None)
        if self.start == self.end:
            return True
        if self.start < self.end:
            return self.start <= current < self.end
        return current >= self.start or current < self.end

    def format_for_humans(self) -> str:
        return self.key


@dataclass(slots=True)
class CameraConfig:
    id: str
    map_url: str
    enabled: bool = True
    display_name: str | None = None
    source_kind: str = "ufanet"
    source_transport: str = "snapshot"
    min_confidence: float = 0.05
    min_free_spaces: int = 1
    detection_image_size: int | None = None
    detection_roi: tuple[float, float, float, float] | None = None
    detection_exclude_rois: tuple[tuple[float, float, float, float], ...] = ()
    parking_slots: tuple["ParkingSlot", ...] = ()
    stable_cycles: int = 2
    poll_interval_seconds: int | None = None
    demo_observations_path: Path | None = None
    demo_loop: bool = True


@dataclass(slots=True)
class DemoObservationStep:
    availability: Availability
    free_count: int
    occupied_count: int
    raw_frame_path: Path | None = None
    annotated_frame_path: Path | None = None


@dataclass(slots=True)
class ResolvedCamera:
    id: str
    map_url: str
    number: str
    server: str
    token: str
    display_name: str
    enabled: bool = True
    source_kind: str = "ufanet"
    source_transport: str = "snapshot"
    min_confidence: float = 0.05
    min_free_spaces: int = 1
    detection_image_size: int | None = None
    detection_roi: tuple[float, float, float, float] | None = None
    detection_exclude_rois: tuple[tuple[float, float, float, float], ...] = ()
    parking_slots: tuple["ParkingSlot", ...] = ()
    stable_cycles: int = 2
    poll_interval_seconds: int | None = None
    demo_observations: tuple[DemoObservationStep, ...] = ()
    demo_loop: bool = True

    def snapshot_url(self, image_width: int) -> str:
        if self.source_kind != "ufanet":
            raise RuntimeError(f"Camera {self.id} does not use Ufanet snapshots")
        return (
            f"http://cams.ufanet.ru/api/v0/screenshots/"
            f"{self.number}~{image_width}.jpg?token={self.token}"
        )

    def hls_master_url(self) -> str:
        if self.source_kind != "ufanet":
            raise RuntimeError(f"Camera {self.id} does not use Ufanet streams")
        return f"http://{self.server}/{self.number}/index.fmp4.m3u8?token={self.token}"


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass(slots=True)
class ParkingSlot:
    id: str
    box: tuple[float, float, float, float]


@dataclass(slots=True)
class CameraObservation:
    camera_id: str
    display_name: str
    observed_at: datetime
    source_updated_at: datetime | None
    availability: Availability
    free_count: int
    occupied_count: int
    detections: list[Detection]
    raw_frame_path: Path | None = None
    annotated_frame_path: Path | None = None


@dataclass(slots=True)
class CameraStatusRecord:
    camera_id: str
    display_name: str
    observed_at: datetime | None
    source_updated_at: datetime | None
    current_availability: Availability
    stable_availability: Availability
    stable_cycles: int
    free_count: int
    occupied_count: int
    raw_frame_path: Path | None = None
    annotated_frame_path: Path | None = None
    last_error: str | None = None


@dataclass(slots=True)
class Subscription:
    id: int
    chat_id: int
    camera_id: str
    active: bool
    created_at: datetime
    updated_at: datetime
    window_key: str
    time_window: TimeWindow | None = None
    last_notified_state: Availability | None = None
    last_notified_at: datetime | None = None

    def is_active_now(self, moment: datetime) -> bool:
        if not self.active:
            return False
        return self.time_window.contains(moment) if self.time_window else True
