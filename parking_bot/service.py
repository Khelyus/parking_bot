from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, tzinfo
from email.utils import parsedate_to_datetime
import logging
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from telegram.error import Forbidden, NetworkError, RetryAfter, TelegramError, TimedOut

from parking_bot.camera_catalog import UfanetCatalogClient
from parking_bot.detector import DetectionSummary, ParkingSpaceDetector, VehicleDetector
from parking_bot.repository import SQLiteRepository
from parking_bot.settings import Settings
from parking_bot.slot_classifier import SlotPrediction, SlotStatusClassifier
from parking_bot.state_machine import AvailabilityStabilizer
from parking_bot.time_utils import resolve_timezone
from parking_bot.types import (
    Availability,
    CameraObservation,
    CameraStatusRecord,
    Detection,
    ParkingSlot,
    ResolvedCamera,
    SlotGeometry,
)

if TYPE_CHECKING:
    from telegram import Bot


logger = logging.getLogger(__name__)


def _require_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is not installed. Run 'python -m pip install -r requirements.txt' first."
        ) from exc
    return cv2


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "NumPy is not installed. Run 'python -m pip install -r requirements.txt' first."
        ) from exc
    return np


def _require_ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is not installed. Run 'python -m pip install -r requirements.txt' first."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


class CameraMonitorService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: SQLiteRepository,
        detector: ParkingSpaceDetector,
        vehicle_detector: VehicleDetector | None,
        slot_classifier: SlotStatusClassifier | None,
        catalog_client: UfanetCatalogClient,
        cameras: list[ResolvedCamera],
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.detector = detector
        self.vehicle_detector = vehicle_detector
        self.slot_classifier = slot_classifier
        self.catalog_client = catalog_client
        self._cameras = {camera.id: camera for camera in cameras}
        self._stabilizers = {
            camera.id: AvailabilityStabilizer(camera.stable_cycles) for camera in cameras
        }
        self._notification_stabilizers = {
            camera.id: AvailabilityStabilizer(
                max(camera.stable_cycles, self.settings.notification_stable_cycles)
            )
            for camera in cameras
        }
        self._next_due = {camera.id: 0.0 for camera in cameras}
        self._locks = {camera.id: asyncio.Lock() for camera in cameras}
        self._demo_indices = {
            camera.id: 0 for camera in cameras if camera.source_kind == "demo" and camera.demo_observations
        }
        self._task: asyncio.Task[None] | None = None
        worker_count = max(1, min(len(cameras), self.settings.monitor_workers))
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="parking-monitor",
        )
        self._bot: Bot | None = None
        self._telegram_notification_retry_delays = (0.0, 2.0, 5.0)

    @property
    def timezone(self) -> tzinfo:
        return resolve_timezone(self.settings.timezone)

    @property
    def cameras(self) -> list[ResolvedCamera]:
        return list(self._cameras.values())

    def get_camera(self, camera_id: str) -> ResolvedCamera:
        return self._cameras[camera_id]

    def find_camera(self, camera_id: str) -> ResolvedCamera | None:
        return self._cameras.get(camera_id)

    def get_status(self, camera_id: str) -> CameraStatusRecord | None:
        return self.repository.get_camera_status(camera_id)

    async def start(self, bot: Bot) -> None:
        self._bot = bot
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop(), name="camera-monitor-loop")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def refresh_camera(self, camera_id: str, *, notify: bool = False) -> CameraStatusRecord:
        camera = self.get_camera(camera_id)
        async with self._locks[camera_id]:
            record = await self._poll_camera(camera, notify=notify)
            next_due = asyncio.get_running_loop().time()
            self._next_due[camera_id] = next_due + self._poll_interval(camera)
            return record

    async def _run_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while True:
                due_camera_ids = [
                    camera_id
                    for camera_id, next_due in self._next_due.items()
                    if loop.time() >= next_due
                ]
                if not due_camera_ids:
                    await asyncio.sleep(1)
                    continue

                results = await asyncio.gather(
                    *(self.refresh_camera(camera_id, notify=True) for camera_id in due_camera_ids),
                    return_exceptions=True,
                )
                for camera_id, result in zip(due_camera_ids, results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Failed to process camera %s",
                            camera_id,
                            exc_info=(type(result), result, result.__traceback__),
                        )
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise

    async def _poll_camera(self, camera: ResolvedCamera, *, notify: bool) -> CameraStatusRecord:
        loop = asyncio.get_running_loop()
        try:
            observation = await loop.run_in_executor(self._executor, self._fetch_and_detect, camera)
        except Exception as exc:
            logger.warning("Camera polling failed for %s: %s", camera.id, exc)
            existing = self.repository.get_camera_status(camera.id)
            fallback = CameraStatusRecord(
                camera_id=camera.id,
                display_name=camera.display_name,
                observed_at=existing.observed_at if existing else None,
                source_updated_at=existing.source_updated_at if existing else None,
                current_availability=existing.current_availability if existing else Availability.UNKNOWN,
                stable_availability=existing.stable_availability if existing else Availability.UNKNOWN,
                stable_cycles=existing.stable_cycles if existing else 0,
                free_count=existing.free_count if existing else 0,
                occupied_count=existing.occupied_count if existing else 0,
                raw_frame_path=existing.raw_frame_path if existing else None,
                annotated_frame_path=existing.annotated_frame_path if existing else None,
                last_error=str(exc),
            )
            self.repository.save_camera_status(fallback)
            return fallback

        transition = self._stabilizers[camera.id].ingest(observation.availability)
        notification_transition = self._notification_stabilizers[camera.id].ingest(
            observation.availability
        )
        snapshot = self._stabilizers[camera.id].snapshot()
        stable_cycles = (
            snapshot.candidate_hits if snapshot.candidate_availability != Availability.UNKNOWN else 0
        )
        record = CameraStatusRecord(
            camera_id=camera.id,
            display_name=camera.display_name,
            observed_at=observation.observed_at,
            source_updated_at=observation.source_updated_at,
            current_availability=observation.availability,
            stable_availability=snapshot.stable_availability,
            stable_cycles=stable_cycles,
            free_count=observation.free_count,
            occupied_count=observation.occupied_count,
            raw_frame_path=observation.raw_frame_path,
            annotated_frame_path=observation.annotated_frame_path,
            last_error=None,
        )
        self.repository.save_camera_status(record)

        if notification_transition is not None and notification_transition[1] == Availability.FULL:
            self.repository.mark_camera_subscriptions_state(
                camera_id=camera.id,
                availability=Availability.FULL,
            )

        if notify and notification_transition == (Availability.FULL, Availability.FREE):
            await self._notify_subscribers(camera, observation)

        return record

    def _fetch_and_detect(self, camera: ResolvedCamera) -> CameraObservation:
        if camera.source_kind == "demo":
            return self._build_demo_observation(camera)

        if camera.source_transport == "hls":
            return self._fetch_and_detect_hls(camera)

        raw_frame_path, _payload, source_updated_at = self._fetch_snapshot(camera)
        detection = self._detect_frame(camera, raw_frame_path)
        return self._build_detection_observation(
            camera,
            detection,
            raw_frame_path=raw_frame_path,
            source_updated_at=source_updated_at,
            observed_at=datetime.now(tz=self.timezone),
        )

    def _fetch_and_detect_hls(self, camera: ResolvedCamera) -> CameraObservation:
        try:
            attempts = max(1, self.settings.hls_burst_frames)
            best_candidate: tuple[
                tuple[int, int, float, float],
                DetectionSummary,
                bytes,
                datetime | None,
                datetime,
            ] | None = None
            for attempt in range(attempts):
                raw_frame_path, source_updated_at = self._fetch_hls_frame(camera)
                detection = self._detect_frame(camera, raw_frame_path)
                observed_at = datetime.now(tz=self.timezone)
                raw_bytes = raw_frame_path.read_bytes()
                score = self._rank_detection_candidate(
                    detection,
                    source_updated_at=source_updated_at,
                    observed_at=observed_at,
                )
                if best_candidate is None or score > best_candidate[0]:
                    best_candidate = (
                        score,
                        detection,
                        raw_bytes,
                        source_updated_at,
                        observed_at,
                    )
                if attempt + 1 < attempts and self.settings.hls_burst_pause_seconds > 0:
                    time.sleep(self.settings.hls_burst_pause_seconds)

            if best_candidate is None:
                raise RuntimeError(f"HLS burst did not produce a usable frame for {camera.id}")

            _score, detection, raw_bytes, source_updated_at, observed_at = best_candidate
            raw_frame_path = self._frame_path(camera.id, "raw")
            raw_frame_path.write_bytes(raw_bytes)
            return self._build_detection_observation(
                camera,
                detection,
                raw_frame_path=raw_frame_path,
                source_updated_at=source_updated_at,
                observed_at=observed_at,
            )
        except Exception as exc:
            logger.warning(
                "HLS frame extraction failed for %s, falling back to snapshot API: %s",
                camera.id,
                exc,
            )
            raw_frame_path, _payload, source_updated_at = self._fetch_snapshot(camera)
            detection = self._detect_frame(camera, raw_frame_path)
            return self._build_detection_observation(
                camera,
                detection,
                raw_frame_path=raw_frame_path,
                source_updated_at=source_updated_at,
                observed_at=datetime.now(tz=self.timezone),
            )

    def _detect_frame(
        self,
        camera: ResolvedCamera,
        raw_frame_path: Path,
    ) -> DetectionSummary:
        if camera.parking_slots:
            return self._detect_frame_by_slots(camera, raw_frame_path)
        return self._detect_frame_with_space_model(camera, raw_frame_path)

    def _detect_frame_with_space_model(
        self,
        camera: ResolvedCamera,
        raw_frame_path: Path,
    ) -> DetectionSummary:
        image = self._load_frame_image(raw_frame_path)
        roi_box: tuple[int, int, int, int] | None = None
        exclusion_boxes = [
            self._resolve_roi_bounds(image.shape[:2], roi)
            for roi in camera.detection_exclude_rois
        ]
        detection_image = image
        if camera.detection_roi is not None:
            roi_box = self._resolve_roi_bounds(image.shape[:2], camera.detection_roi)
            x1, y1, x2, y2 = roi_box
            detection_image = image[y1:y2, x1:x2]

        detection = self.detector.detect_image(
            detection_image,
            confidence=camera.min_confidence,
            min_free_spaces=camera.min_free_spaces,
            image_size=camera.detection_image_size,
        )
        vehicle_detections = self._detect_vehicle_detections(
            detection_image,
            camera=camera,
            offset=(roi_box[0], roi_box[1]) if roi_box is not None else (0, 0),
        )
        vehicle_detections = self._filter_excluded_detections(
            vehicle_detections,
            exclusion_boxes=exclusion_boxes,
        )
        remapped_detections = self._remap_detections(
            detection.detections,
            offset=(roi_box[0], roi_box[1]) if roi_box is not None else (0, 0),
        )
        remapped_detections = self._filter_excluded_detections(
            remapped_detections,
            exclusion_boxes=exclusion_boxes,
        )
        filtered_detections = self._filter_space_detections(
            remapped_detections,
            vehicle_detections=vehicle_detections,
            image_shape=image.shape[:2],
        )
        return self._build_space_summary(
            image,
            detections=filtered_detections,
            min_free_spaces=camera.min_free_spaces,
            roi_box=roi_box,
            exclusion_boxes=exclusion_boxes,
        )

    def _detect_frame_by_slots(
        self,
        camera: ResolvedCamera,
        raw_frame_path: Path,
    ) -> DetectionSummary:
        if self.vehicle_detector is None and self.slot_classifier is None:
            logger.warning(
                "Camera %s has parking slots configured, but no slot detectors are available; falling back",
                camera.id,
            )
            return self._detect_frame_with_space_model(camera, raw_frame_path)

        image = self._load_frame_image(raw_frame_path)
        vehicle_detections: list[Detection] = []
        if self.vehicle_detector is not None:
            try:
                vehicle_detections = self.vehicle_detector.detect_image(
                    image,
                    confidence=self.settings.vehicle_confidence,
                    image_size=max(1280, camera.detection_image_size or 0) or None,
                )
            except Exception as exc:
                logger.warning("Vehicle-based slot detection failed for %s: %s", camera.id, exc)

        slot_detections: list[Detection] = []
        slot_geometries: list[SlotGeometry] = []
        for slot in camera.parking_slots:
            slot_geometry = self._resolve_slot_geometry(image.shape[:2], slot)
            slot_geometries.append(slot_geometry)
            matched_vehicle = self._match_vehicle_to_slot(slot_geometry, vehicle_detections)
            slot_prediction = self._predict_slot_status(image, slot_geometry)
            vehicle_score = (
                self._slot_vehicle_match_score(slot_geometry, matched_vehicle.box)
                if matched_vehicle is not None
                else 0.0
            )
            local_vehicle_score = self._detect_local_vehicle_score(
                image,
                slot_geometry,
                base_vehicle_score=vehicle_score,
                slot_prediction=slot_prediction,
            )
            label, confidence = self._resolve_slot_status(
                vehicle_score=vehicle_score,
                local_vehicle_score=local_vehicle_score,
                slot_prediction=slot_prediction,
            )
            slot_detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    box=slot_geometry.bounds,
                )
            )

        free_count = sum(1 for item in slot_detections if item.label == "space-empty")
        occupied_count = sum(1 for item in slot_detections if item.label == "space-occupied")
        if free_count >= camera.min_free_spaces:
            availability = Availability.FREE
        elif occupied_count > 0 and free_count == 0:
            availability = Availability.FULL
        else:
            availability = Availability.UNKNOWN

        return DetectionSummary(
            free_count=free_count,
            occupied_count=occupied_count,
            availability=availability,
            detections=slot_detections,
            annotated_frame=self._annotate_slot_detection(
                image,
                slot_detections,
                slot_geometries=slot_geometries,
                vehicle_detections=vehicle_detections,
            ),
        )

    def _predict_slot_status(
        self,
        image: Any,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
    ) -> SlotPrediction | None:
        if self.slot_classifier is None:
            return None
        try:
            crop = self._extract_slot_crop(image, slot_shape)
            if crop.size == 0:
                return None
            return self.slot_classifier.predict_image(crop)
        except Exception as exc:
            logger.warning("Slot classifier failed for slot %s: %s", self._slot_bounds(slot_shape), exc)
            return None

    def _detect_local_vehicle_score(
        self,
        image: Any,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
        *,
        base_vehicle_score: float,
        slot_prediction: SlotPrediction | None,
    ) -> float:
        if self.vehicle_detector is None:
            return 0.0
        if not self._should_run_local_vehicle_check(
            base_vehicle_score=base_vehicle_score,
            slot_prediction=slot_prediction,
        ):
            return 0.0

        slot_geometry = self._coerce_slot_geometry(slot_shape)
        crop_box = self._expand_slot_crop_box(
            slot_geometry,
            image.shape[:2],
            padding_scale_x=0.40,
            padding_scale_y_top=0.42,
            padding_scale_y_bottom=0.28,
        )
        x1, y1, x2, y2 = crop_box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0

        try:
            local_detections = self.vehicle_detector.detect_image(
                crop,
                confidence=max(0.02, self.settings.vehicle_confidence * 0.7),
                image_size=self._align_inference_image_size(
                    max(960, min(1600, round(max(crop.shape[:2]) * 2)))
                ),
            )
        except Exception as exc:
            logger.warning("Local vehicle check failed for crop %s: %s", slot_box, exc)
            return 0.0

        if not local_detections:
            return 0.0

        remapped_detections = self._remap_detections(local_detections, offset=(x1, y1))
        local_match = self._match_vehicle_to_slot(slot_geometry, remapped_detections)
        if local_match is not None:
            return self._slot_vehicle_match_score(slot_geometry, local_match.box)

        return max(
            (
                self._slot_box_overlap_ratio(slot_geometry, detection.box)
                for detection in remapped_detections
            ),
            default=0.0,
        )

    def _should_run_local_vehicle_check(
        self,
        *,
        base_vehicle_score: float,
        slot_prediction: SlotPrediction | None,
    ) -> bool:
        if base_vehicle_score >= 0.82:
            return False
        if slot_prediction is None:
            return base_vehicle_score < 0.55
        if slot_prediction.label == "space-occupied" and slot_prediction.confidence >= 0.72:
            return False
        return (
            slot_prediction.empty_probability >= 0.52
            or slot_prediction.occupied_probability >= 0.18
            or base_vehicle_score < 0.55
        )

    def _align_inference_image_size(self, image_size: int) -> int:
        return max(32, ((int(image_size) + 31) // 32) * 32)

    def _expand_slot_crop_box(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
        image_shape: tuple[int, int],
        *,
        padding_scale_x: float = 0.28,
        padding_scale_y_top: float = 0.28,
        padding_scale_y_bottom: float = 0.18,
    ) -> tuple[int, int, int, int]:
        expanded_geometry = self._expand_slot_geometry(
            self._coerce_slot_geometry(slot_shape),
            image_shape,
            padding_scale_x=padding_scale_x,
            padding_scale_y_top=padding_scale_y_top,
            padding_scale_y_bottom=padding_scale_y_bottom,
        )
        return expanded_geometry.bounds

    def _coerce_slot_geometry(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
    ) -> SlotGeometry:
        if isinstance(slot_shape, SlotGeometry):
            return slot_shape
        x1, y1, x2, y2 = slot_shape
        width = max(1.0, float(x2 - x1))
        height = max(1.0, float(y2 - y1))
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return SlotGeometry(
            bounds=(x1, y1, x2, y2),
            center=center,
            size=(width, height),
            angle_degrees=0.0,
        )

    def _resolve_slot_geometry(
        self,
        image_shape: tuple[int, int],
        slot: ParkingSlot,
    ) -> SlotGeometry:
        bounds = self._resolve_roi_bounds(image_shape, slot.box)
        x1, y1, x2, y2 = bounds
        width = max(1.0, float(x2 - x1))
        height = max(1.0, float(y2 - y1))
        return SlotGeometry(
            bounds=bounds,
            center=((x1 + x2) / 2, (y1 + y2) / 2),
            size=(width, height),
            angle_degrees=float(slot.angle_degrees),
        )

    def _slot_bounds(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
    ) -> tuple[int, int, int, int]:
        return self._coerce_slot_geometry(slot_shape).bounds

    def _slot_polygon(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
    ) -> Any:
        np = _require_numpy()
        slot_geometry = self._coerce_slot_geometry(slot_shape)
        center_x, center_y = slot_geometry.center
        width, height = slot_geometry.size
        half_width = width / 2
        half_height = height / 2
        radians = math.radians(slot_geometry.angle_degrees)
        cos_angle = math.cos(radians)
        sin_angle = math.sin(radians)
        corners: list[tuple[float, float]] = []
        for dx, dy in (
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ):
            corners.append(
                (
                    center_x + (dx * cos_angle) - (dy * sin_angle),
                    center_y + (dx * sin_angle) + (dy * cos_angle),
                )
            )
        return np.array(corners, dtype=np.float32)

    def _slot_area(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
    ) -> float:
        slot_geometry = self._coerce_slot_geometry(slot_shape)
        return max(1.0, slot_geometry.size[0] * slot_geometry.size[1])

    def _bounds_from_polygon(
        self,
        polygon: Any,
        image_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        image_height, image_width = image_shape
        min_x = max(0, min(image_width - 1, int(math.floor(float(polygon[:, 0].min())))))
        min_y = max(0, min(image_height - 1, int(math.floor(float(polygon[:, 1].min())))))
        max_x = max(min_x + 1, min(image_width, int(math.ceil(float(polygon[:, 0].max())))))
        max_y = max(min_y + 1, min(image_height, int(math.ceil(float(polygon[:, 1].max())))))
        return (min_x, min_y, max_x, max_y)

    def _expand_slot_geometry(
        self,
        slot_geometry: SlotGeometry,
        image_shape: tuple[int, int],
        *,
        padding_scale_x: float = 0.28,
        padding_scale_y_top: float = 0.28,
        padding_scale_y_bottom: float = 0.18,
    ) -> SlotGeometry:
        width = max(1.0, slot_geometry.size[0])
        height = max(1.0, slot_geometry.size[1])
        padding_x = max(6, round(width * padding_scale_x))
        padding_y_top = max(6, round(height * padding_scale_y_top))
        padding_y_bottom = max(6, round(height * padding_scale_y_bottom))
        expanded_width = width + padding_x * 2
        expanded_height = height + padding_y_top + padding_y_bottom
        center_x, center_y = slot_geometry.center
        center_shift_local_y = (padding_y_bottom - padding_y_top) / 2
        radians = math.radians(slot_geometry.angle_degrees)
        shifted_center = (
            center_x - (center_shift_local_y * math.sin(radians)),
            center_y + (center_shift_local_y * math.cos(radians)),
        )
        expanded = SlotGeometry(
            bounds=slot_geometry.bounds,
            center=shifted_center,
            size=(expanded_width, expanded_height),
            angle_degrees=slot_geometry.angle_degrees,
        )
        return SlotGeometry(
            bounds=self._bounds_from_polygon(self._slot_polygon(expanded), image_shape),
            center=expanded.center,
            size=expanded.size,
            angle_degrees=expanded.angle_degrees,
        )

    def _extract_slot_crop(
        self,
        image: Any,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
        *,
        padding_scale_x: float = 0.28,
        padding_scale_y_top: float = 0.28,
        padding_scale_y_bottom: float = 0.18,
    ) -> Any:
        cv2 = _require_cv2()
        expanded_geometry = self._expand_slot_geometry(
            self._coerce_slot_geometry(slot_shape),
            image.shape[:2],
            padding_scale_x=padding_scale_x,
            padding_scale_y_top=padding_scale_y_top,
            padding_scale_y_bottom=padding_scale_y_bottom,
        )
        output_width = max(1, round(expanded_geometry.size[0]))
        output_height = max(1, round(expanded_geometry.size[1]))
        if abs(expanded_geometry.angle_degrees) < 1e-3:
            x1, y1, x2, y2 = expanded_geometry.bounds
            return image[y1:y2, x1:x2]

        np = _require_numpy()
        source = self._slot_polygon(expanded_geometry).astype(np.float32)
        destination = np.array(
            [
                [0, 0],
                [output_width - 1, 0],
                [output_width - 1, output_height - 1],
                [0, output_height - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        return cv2.warpPerspective(image, matrix, (output_width, output_height))

    def _resolve_slot_status(
        self,
        *,
        vehicle_score: float,
        local_vehicle_score: float,
        slot_prediction: SlotPrediction | None,
    ) -> tuple[str, float]:
        vehicle_probability = max(
            self._vehicle_match_probability(vehicle_score),
            self._vehicle_match_probability(local_vehicle_score),
        )
        empty_probability = (
            slot_prediction.empty_probability
            if slot_prediction is not None
            else (0.70 if vehicle_probability < 0.15 else 0.0)
        )
        occupied_probability = max(
            vehicle_probability,
            slot_prediction.occupied_probability if slot_prediction is not None else 0.0,
        )

        if vehicle_probability >= 0.74:
            return ("space-occupied", vehicle_probability)
        if occupied_probability >= 0.62 and occupied_probability >= empty_probability + 0.08:
            return ("space-occupied", occupied_probability)
        if (
            empty_probability >= max(self.settings.slot_classifier_confidence, 0.72)
            and occupied_probability <= 0.30
            and vehicle_probability <= 0.12
        ):
            return ("space-empty", empty_probability)
        if vehicle_probability >= 0.38:
            return ("space-occupied", max(0.52, occupied_probability))
        if occupied_probability >= 0.32 and empty_probability < 0.88:
            return ("space-occupied", max(0.51, occupied_probability))
        if vehicle_score >= 0.55 or local_vehicle_score >= 0.55:
            return ("space-occupied", max(0.55, occupied_probability))
        if empty_probability >= 0.88 and occupied_probability <= 0.18:
            return ("space-empty", empty_probability)
        return ("space-occupied", max(0.50, occupied_probability, vehicle_probability))

    def _vehicle_match_probability(self, vehicle_score: float) -> float:
        if vehicle_score <= 0.0:
            return 0.0
        normalized = (vehicle_score - 0.55) / 1.05
        return max(0.0, min(1.0, normalized))

    def _build_detection_observation(
        self,
        camera: ResolvedCamera,
        detection: DetectionSummary,
        *,
        raw_frame_path: Path,
        source_updated_at: datetime | None,
        observed_at: datetime,
    ) -> CameraObservation:
        annotated_frame_path = self._frame_path(camera.id, "annotated")
        cv2 = _require_cv2()
        cv2.imwrite(str(annotated_frame_path), detection.annotated_frame)
        return CameraObservation(
            camera_id=camera.id,
            display_name=camera.display_name,
            observed_at=observed_at,
            source_updated_at=source_updated_at,
            availability=detection.availability,
            free_count=detection.free_count,
            occupied_count=detection.occupied_count,
            detections=detection.detections,
            raw_frame_path=raw_frame_path,
            annotated_frame_path=annotated_frame_path,
        )

    def _load_frame_image(self, raw_frame_path: Path) -> Any:
        cv2 = _require_cv2()
        np = _require_numpy()
        payload = raw_frame_path.read_bytes()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode frame from {raw_frame_path}")
        return image

    def _resolve_roi_bounds(
        self,
        image_shape: tuple[int, int],
        roi: tuple[float, float, float, float],
    ) -> tuple[int, int, int, int]:
        height, width = image_shape
        x1 = max(0, min(width - 1, int(width * roi[0])))
        y1 = max(0, min(height - 1, int(height * roi[1])))
        x2 = max(x1 + 1, min(width, int(width * roi[2])))
        y2 = max(y1 + 1, min(height, int(height * roi[3])))
        return (x1, y1, x2, y2)

    def _detect_vehicle_detections(
        self,
        image: Any,
        *,
        camera: ResolvedCamera,
        offset: tuple[int, int] = (0, 0),
    ) -> list[Detection]:
        if self.vehicle_detector is None:
            return []
        try:
            local_detections = self.vehicle_detector.detect_image(
                image,
                confidence=self.settings.vehicle_confidence,
                image_size=max(1280, camera.detection_image_size or 0) or None,
            )
        except Exception as exc:
            logger.debug("Vehicle detection failed for %s: %s", camera.id, exc)
            return []
        return self._remap_detections(local_detections, offset=offset)

    def _filter_excluded_detections(
        self,
        detections: list[Detection],
        *,
        exclusion_boxes: list[tuple[int, int, int, int]],
    ) -> list[Detection]:
        if not exclusion_boxes:
            return list(detections)
        return [
            detection
            for detection in detections
            if not self._is_detection_excluded(detection.box, exclusion_boxes)
        ]

    def _is_detection_excluded(
        self,
        box: tuple[int, int, int, int],
        exclusion_boxes: list[tuple[int, int, int, int]],
    ) -> bool:
        center_x, center_y = self._box_center(box)
        for exclusion_box in exclusion_boxes:
            x1, y1, x2, y2 = exclusion_box
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                return True
            if self._box_overlap_ratio(box, exclusion_box) >= 0.45:
                return True
        return False

    def _remap_detections(
        self,
        detections: list[Detection],
        *,
        offset: tuple[int, int],
    ) -> list[Detection]:
        offset_x, offset_y = offset
        if offset_x == 0 and offset_y == 0:
            return list(detections)
        return [
            Detection(
                label=item.label,
                confidence=item.confidence,
                box=(
                    item.box[0] + offset_x,
                    item.box[1] + offset_y,
                    item.box[2] + offset_x,
                    item.box[3] + offset_y,
                ),
            )
            for item in detections
        ]

    def _filter_space_detections(
        self,
        detections: list[Detection],
        *,
        vehicle_detections: list[Detection],
        image_shape: tuple[int, int],
    ) -> list[Detection]:
        occupied = [item for item in detections if "occupied" in item.label.lower()]
        free = [item for item in detections if "empty" in item.label.lower() or "free" in item.label.lower()]
        filtered_occupied = [
            detection
            for detection in occupied
            if self._is_supported_occupied_detection(
                detection,
                vehicle_detections=vehicle_detections,
                image_shape=image_shape,
            )
        ]
        filtered: list[Detection] = list(filtered_occupied)
        for detection in free:
            if self._is_supported_free_detection(
                detection,
                occupied_detections=filtered_occupied,
                vehicle_detections=vehicle_detections,
                image_shape=image_shape,
            ):
                filtered.append(detection)
        filtered.sort(key=lambda item: item.confidence, reverse=True)
        return filtered

    def _is_supported_occupied_detection(
        self,
        detection: Detection,
        *,
        vehicle_detections: list[Detection],
        image_shape: tuple[int, int],
    ) -> bool:
        image_height, image_width = image_shape
        x1, y1, x2, y2 = detection.box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = width * height
        image_area = image_height * image_width
        if width < 10 or height < 10:
            return False
        if y2 < image_height * 0.10:
            return False
        if area < max(180, int(image_area * 0.00012)):
            return False
        if area > image_area * 0.12:
            return False
        if not vehicle_detections:
            return True
        if self._max_vehicle_overlap_ratio(detection.box, vehicle_detections) >= 0.02:
            return True

        support_dx = max(120, width * 5)
        support_dy = max(45, int(height * 2.2))
        nearby_vehicles = self._nearby_support_count(
            detection.box,
            [vehicle.box for vehicle in vehicle_detections],
            max_dx=support_dx,
            max_dy=support_dy,
        )
        return nearby_vehicles >= 1

    def _is_supported_free_detection(
        self,
        detection: Detection,
        *,
        occupied_detections: list[Detection],
        vehicle_detections: list[Detection],
        image_shape: tuple[int, int],
    ) -> bool:
        image_height, image_width = image_shape
        x1, y1, x2, y2 = detection.box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = width * height
        image_area = image_height * image_width
        if width < 12 or height < 12:
            return False
        if y2 < image_height * 0.10:
            return False
        if area < max(220, int(image_area * 0.00016)):
            return False
        if area > image_area * 0.10:
            return False
        if self._max_vehicle_overlap_ratio(detection.box, vehicle_detections) >= 0.08:
            return False

        support_dx = max(120, width * 5)
        support_dy = max(40, int(height * 1.8))
        nearby_vehicles = self._nearby_support_count(
            detection.box,
            [vehicle.box for vehicle in vehicle_detections],
            max_dx=support_dx,
            max_dy=support_dy,
        )
        if occupied_detections:
            nearby_occupied = self._nearby_support_count(
                detection.box,
                [item.box for item in occupied_detections],
                max_dx=support_dx,
                max_dy=support_dy,
            )
            return nearby_occupied > 0 or nearby_vehicles >= 2
        return nearby_vehicles >= 2

    def _nearby_support_count(
        self,
        box: tuple[int, int, int, int],
        supports: list[tuple[int, int, int, int]],
        *,
        max_dx: int,
        max_dy: int,
    ) -> int:
        center_x, center_y = self._box_center(box)
        count = 0
        for support in supports:
            support_x, support_y = self._box_center(support)
            if abs(center_x - support_x) <= max_dx and abs(center_y - support_y) <= max_dy:
                count += 1
        return count

    def _max_vehicle_overlap_ratio(
        self,
        box: tuple[int, int, int, int],
        vehicle_detections: list[Detection],
    ) -> float:
        return max((self._box_overlap_ratio(box, item.box) for item in vehicle_detections), default=0.0)

    def _box_center(
        self,
        box: tuple[int, int, int, int],
    ) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    def _box_overlap_ratio(
        self,
        left: tuple[int, int, int, int],
        right: tuple[int, int, int, int],
    ) -> float:
        x1 = max(left[0], right[0])
        y1 = max(left[1], right[1])
        x2 = min(left[2], right[2])
        y2 = min(left[3], right[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        intersection = (x2 - x1) * (y2 - y1)
        left_area = max(1, (left[2] - left[0]) * (left[3] - left[1]))
        return intersection / left_area

    def _build_space_summary(
        self,
        image: Any,
        *,
        detections: list[Detection],
        min_free_spaces: int,
        roi_box: tuple[int, int, int, int] | None,
        exclusion_boxes: list[tuple[int, int, int, int]],
    ) -> DetectionSummary:
        free_count = sum(1 for item in detections if "empty" in item.label.lower() or "free" in item.label.lower())
        occupied_count = sum(1 for item in detections if "occupied" in item.label.lower() or "busy" in item.label.lower())
        if free_count >= min_free_spaces:
            availability = Availability.FREE
        elif occupied_count > 0 and free_count == 0:
            availability = Availability.FULL
        else:
            availability = Availability.UNKNOWN

        annotated_frame = self._annotate_space_detections(image, detections)
        if roi_box is not None or exclusion_boxes:
            cv2 = _require_cv2()
        if roi_box is not None:
            cv2.rectangle(annotated_frame, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 180, 255), 2)
        for exclusion_box in exclusion_boxes:
            cv2.rectangle(
                annotated_frame,
                (exclusion_box[0], exclusion_box[1]),
                (exclusion_box[2], exclusion_box[3]),
                (0, 0, 255),
                2,
            )
        return DetectionSummary(
            free_count=free_count,
            occupied_count=occupied_count,
            availability=availability,
            detections=detections,
            annotated_frame=annotated_frame,
        )

    def _annotate_space_detections(
        self,
        image: Any,
        detections: list[Detection],
    ) -> Any:
        cv2 = _require_cv2()
        annotated = image.copy()
        for detection in detections:
            is_free = "empty" in detection.label.lower() or "free" in detection.label.lower()
            color = (0, 200, 80) if is_free else (0, 90, 220)
            x1, y1, x2, y2 = detection.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{detection.label} {detection.confidence:.2f}"
            label_y = y1 - 10 if y1 > 20 else y1 + 20
            cv2.putText(
                annotated,
                text,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _match_vehicle_to_slot(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
        vehicles: list[Detection],
    ) -> Detection | None:
        best_match: Detection | None = None
        best_score = 0.0
        for vehicle in vehicles:
            score = self._slot_vehicle_match_score(slot_shape, vehicle.box)
            if score < 0.55:
                continue
            if (
                best_match is None
                or score > best_score
                or (abs(score - best_score) < 1e-6 and vehicle.confidence > best_match.confidence)
            ):
                best_match = vehicle
                best_score = score
        return best_match

    def _slot_vehicle_match_score(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
        vehicle_box: tuple[int, int, int, int],
    ) -> float:
        score = 0.0
        if self._slot_contains_vehicle_anchor(slot_shape, vehicle_box):
            score += 1.0

        center_x, center_y = self._box_center(vehicle_box)
        if self._slot_contains_point(slot_shape, center_x, center_y):
            score += 0.7

        overlap_slot = self._slot_box_overlap_ratio(slot_shape, vehicle_box)
        overlap_vehicle = self._vehicle_overlap_with_slot_ratio(vehicle_box, slot_shape)
        score += min(0.8, overlap_slot * 1.8)
        score += min(0.4, overlap_vehicle * 0.8)
        return score

    def _slot_contains_point(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
        point_x: float,
        point_y: float,
    ) -> bool:
        cv2 = _require_cv2()
        polygon = self._slot_polygon(slot_shape)
        return cv2.pointPolygonTest(polygon, (float(point_x), float(point_y)), False) >= 0

    def _slot_contains_vehicle_anchor(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
        vehicle_box: tuple[int, int, int, int],
    ) -> bool:
        anchor_x, anchor_y = self._vehicle_anchor_point(vehicle_box)
        return self._slot_contains_point(slot_shape, anchor_x, anchor_y)

    def _vehicle_overlap_with_slot_ratio(
        self,
        vehicle_box: tuple[int, int, int, int],
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
    ) -> float:
        vehicle_area = max(1.0, (vehicle_box[2] - vehicle_box[0]) * (vehicle_box[3] - vehicle_box[1]))
        return self._slot_box_intersection_area(slot_shape, vehicle_box) / vehicle_area

    def _slot_box_overlap_ratio(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
        box: tuple[int, int, int, int],
    ) -> float:
        return self._slot_box_intersection_area(slot_shape, box) / self._slot_area(slot_shape)

    def _slot_box_intersection_area(
        self,
        slot_shape: tuple[int, int, int, int] | SlotGeometry,
        box: tuple[int, int, int, int],
    ) -> float:
        cv2 = _require_cv2()
        np = _require_numpy()
        slot_polygon = self._slot_polygon(slot_shape).astype(np.float32)
        box_polygon = np.array(
            [
                [box[0], box[1]],
                [box[2], box[1]],
                [box[2], box[3]],
                [box[0], box[3]],
            ],
            dtype=np.float32,
        )
        try:
            area, _intersection = cv2.intersectConvexConvex(slot_polygon, box_polygon)
        except cv2.error:
            return 0.0
        return max(0.0, float(area))

    def _vehicle_anchor_point(
        self,
        vehicle_box: tuple[int, int, int, int],
    ) -> tuple[int, int]:
        x1, y1, x2, y2 = vehicle_box
        return (round((x1 + x2) / 2), round(y2 - (y2 - y1) * 0.12))

    def _annotate_slot_detection(
        self,
        image: Any,
        slot_detections: list[Detection],
        *,
        slot_geometries: list[SlotGeometry] | None,
        vehicle_detections: list[Detection],
    ) -> Any:
        cv2 = _require_cv2()
        np = _require_numpy()
        annotated = image.copy()
        overlay = image.copy()

        geometry_map: list[SlotGeometry | None]
        if slot_geometries is not None and len(slot_geometries) == len(slot_detections):
            geometry_map = list(slot_geometries)
        else:
            geometry_map = [None] * len(slot_detections)

        for detection, geometry in zip(slot_detections, geometry_map):
            is_free = detection.label == "space-empty"
            color = (0, 190, 90) if is_free else (0, 90, 220)
            if geometry is not None:
                polygon = np.round(self._slot_polygon(geometry)).astype(np.int32)
                cv2.fillConvexPoly(overlay, polygon, color)
            else:
                x1, y1, x2, y2 = detection.box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)

        for detection, geometry in zip(slot_detections, geometry_map):
            is_free = detection.label == "space-empty"
            color = (0, 190, 90) if is_free else (0, 90, 220)
            if geometry is not None:
                polygon = np.round(self._slot_polygon(geometry)).astype(np.int32)
                cv2.polylines(annotated, [polygon], isClosed=True, color=color, thickness=2)
                label_bounds = geometry.bounds
            else:
                x1, y1, x2, y2 = detection.box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label_bounds = detection.box
            text = "FREE" if is_free else "OCC"
            x1, y1, _x2, _y2 = label_bounds
            cv2.putText(
                annotated,
                text,
                (x1 + 4, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated

    def _rank_detection_candidate(
        self,
        detection: DetectionSummary,
        *,
        source_updated_at: datetime | None,
        observed_at: datetime,
    ) -> tuple[int, int, float, float]:
        availability_priority = {
            Availability.UNKNOWN: 0,
            Availability.FULL: 1,
            Availability.FREE: 2,
        }[detection.availability]
        detection_count = detection.free_count + detection.occupied_count
        total_confidence = sum(item.confidence for item in detection.detections)
        freshness = (source_updated_at or observed_at).timestamp()
        return (availability_priority, detection_count, total_confidence, freshness)

    def _build_demo_observation(self, camera: ResolvedCamera) -> CameraObservation:
        if not camera.demo_observations:
            raise RuntimeError(f"Demo camera {camera.id} does not contain observations")

        index = self._demo_indices.get(camera.id, 0)
        step = camera.demo_observations[index]
        if camera.demo_loop:
            next_index = (index + 1) % len(camera.demo_observations)
        else:
            next_index = min(index + 1, len(camera.demo_observations) - 1)
        self._demo_indices[camera.id] = next_index

        return CameraObservation(
            camera_id=camera.id,
            display_name=camera.display_name,
            observed_at=datetime.now(tz=self.timezone),
            source_updated_at=None,
            availability=step.availability,
            free_count=step.free_count,
            occupied_count=step.occupied_count,
            detections=[],
            raw_frame_path=step.raw_frame_path,
            annotated_frame_path=step.annotated_frame_path,
        )

    def _fetch_snapshot(self, camera: ResolvedCamera) -> tuple[Path, bytes, datetime | None]:
        last_error: Exception | None = None
        for attempt in range(2):
            request = Request(
                camera.snapshot_url(self.settings.image_width),
                headers={"User-Agent": self.settings.user_agent},
            )
            try:
                with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                    payload = response.read()
                    source_updated_at = self._parse_source_timestamp(response.headers.get("Last-Modified"))
                self._validate_snapshot_payload(payload, camera.id)
                raw_frame_path = self._frame_path(camera.id, "raw")
                raw_frame_path.write_bytes(payload)
                return raw_frame_path, payload, source_updated_at
            except (HTTPError, URLError, ValueError) as exc:
                last_error = exc
                if attempt == 0:
                    self._refresh_camera_credentials(camera)
                    continue
                raise

        raise RuntimeError(f"Snapshot download failed for {camera.id}") from last_error

    def _fetch_hls_frame(self, camera: ResolvedCamera) -> tuple[Path, datetime | None]:
        ffmpeg_exe = _require_ffmpeg_exe()
        raw_frame_path = self._frame_path(camera.id, "raw")
        last_error = "unknown error"
        for attempt in range(2):
            playlist_url = camera.hls_master_url()
            source_updated_at: datetime | None = None
            try:
                source_updated_at = self._fetch_hls_source_timestamp(playlist_url)
            except Exception as exc:
                logger.debug(
                    "Could not read HLS source timestamp for %s, continuing without it: %s",
                    camera.id,
                    exc,
                )
            command = [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-protocol_whitelist",
                "file,http,https,tcp,tls,crypto",
                "-allowed_extensions",
                "ALL",
                "-rw_timeout",
                "15000000",
                "-live_start_index",
                "-1",
                "-i",
                playlist_url,
                "-frames:v",
                "1",
                "-update",
                "1",
                str(raw_frame_path),
            ]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=max(30, self.settings.request_timeout_seconds * 3),
                    **self._subprocess_no_window_kwargs(),
                )
            except Exception as exc:
                last_error = str(exc) or exc.__class__.__name__
                if attempt == 0:
                    self._refresh_camera_credentials(camera)
                    time.sleep(1)
                    continue
                raise RuntimeError(
                    f"ffmpeg could not extract a frame from HLS for {camera.id}: {last_error}"
                ) from exc

            if result.returncode == 0 and raw_frame_path.exists() and raw_frame_path.stat().st_size > 0:
                return raw_frame_path, source_updated_at
            last_error = (result.stderr or "").strip() or "unknown error"
            if attempt == 0:
                self._refresh_camera_credentials(camera)
            time.sleep(1)

        raise RuntimeError(
            f"ffmpeg could not extract a frame from HLS for {camera.id}: {last_error}"
        )

    def _fetch_hls_source_timestamp(self, master_playlist_url: str) -> datetime | None:
        master_text = self._download_text(master_playlist_url)
        media_playlist_url = self._resolve_media_playlist_url(master_playlist_url, master_text)
        if media_playlist_url is None:
            return None

        media_text = self._download_text(media_playlist_url)
        first_timestamp: datetime | None = None
        total_duration_seconds = 0.0
        for raw_line in media_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#EXT-X-PROGRAM-DATE-TIME:") and first_timestamp is None:
                first_timestamp = self._parse_source_timestamp(
                    line.split(":", 1)[1].strip()
                )
            elif line.startswith("#EXTINF:"):
                value = line.split(":", 1)[1].split(",", 1)[0].strip()
                try:
                    total_duration_seconds += float(value)
                except ValueError:
                    continue

        if first_timestamp is None:
            return None
        return first_timestamp + timedelta(seconds=total_duration_seconds)

    def _download_text(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": self.settings.user_agent})
        with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="ignore")

    def _subprocess_no_window_kwargs(self) -> dict[str, Any]:
        if os.name != "nt":
            return {}

        kwargs: dict[str, Any] = {}
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if creationflags:
            kwargs["creationflags"] = creationflags

        startupinfo_factory = getattr(subprocess, "STARTUPINFO", None)
        if startupinfo_factory is None:
            return kwargs

        startupinfo = startupinfo_factory()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["startupinfo"] = startupinfo
        return kwargs

    def _resolve_media_playlist_url(self, master_playlist_url: str, master_text: str) -> str | None:
        best_candidate_url: str | None = None
        best_candidate_score: tuple[int, int] | None = None
        pending_score: tuple[int, int] | None = None

        for raw_line in master_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#EXT-X-STREAM-INF:"):
                bandwidth_match = re.search(r"BANDWIDTH=(\d+)", line)
                resolution_match = re.search(r"RESOLUTION=(\d+)x(\d+)", line)
                bandwidth = int(bandwidth_match.group(1)) if bandwidth_match else 0
                if resolution_match:
                    width = int(resolution_match.group(1))
                    height = int(resolution_match.group(2))
                    resolution_area = width * height
                else:
                    resolution_area = 0
                pending_score = (resolution_area, bandwidth)
                continue
            if line.startswith("#"):
                continue

            if pending_score is not None and (
                best_candidate_score is None or pending_score > best_candidate_score
            ):
                best_candidate_score = pending_score
                best_candidate_url = line
            elif best_candidate_url is None:
                best_candidate_url = line
            pending_score = None

        if best_candidate_url is None:
            return None
        return urljoin(master_playlist_url, best_candidate_url)

    def _validate_snapshot_payload(self, payload: bytes, camera_id: str) -> None:
        if not payload:
            raise ValueError(f"Camera {camera_id} returned an empty response")
        cv2 = _require_cv2()
        np = _require_numpy()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Camera {camera_id} returned a payload that is not a valid image")

    def _parse_source_timestamp(self, raw_value: str | None) -> datetime | None:
        if not raw_value:
            return None
        normalized = raw_value.strip()
        try:
            iso_candidate = normalized.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(iso_candidate)
        except ValueError:
            parsed = None
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=self.timezone)
            return parsed.astimezone(self.timezone)
        try:
            parsed = parsedate_to_datetime(normalized)
        except (TypeError, ValueError, IndexError):
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=self.timezone)
        return parsed.astimezone(self.timezone)

    def _refresh_camera_credentials(self, camera: ResolvedCamera) -> None:
        descriptor = self.catalog_client.resolve(camera.map_url, force_refresh=True)
        camera.number = descriptor.number
        camera.server = descriptor.server
        camera.token = descriptor.token
        camera.display_name = descriptor.name

    def _frame_path(self, camera_id: str, suffix: str) -> Path:
        safe_camera_id = re.sub(r"[^A-Za-z0-9_-]+", "_", camera_id)
        return self.settings.frames_dir / f"{safe_camera_id}_{suffix}.jpg"

    def _poll_interval(self, camera: ResolvedCamera) -> int:
        return camera.poll_interval_seconds or self.settings.poll_interval_seconds

    async def _notify_subscribers(
        self,
        camera: ResolvedCamera,
        observation: CameraObservation,
    ) -> None:
        if self._bot is None:
            return

        now = datetime.now(tz=self.timezone)
        for subscription in self.repository.list_camera_subscriptions(camera.id):
            if not subscription.is_active_now(now):
                continue
            if (
                subscription.last_notified_state == Availability.FREE
                and subscription.last_notified_at is not None
                and (now - subscription.last_notified_at).total_seconds()
                < self.settings.notification_cooldown_seconds
            ):
                continue

            caption = (
                "На парковке появилось свободное место\n\n"
                f"Камера: {camera.display_name}\n"
                f"Свободных мест: {observation.free_count}\n"
                f"Занятых мест: {observation.occupied_count}\n"
                f"Обработано: {observation.observed_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            if observation.source_updated_at is not None:
                caption += (
                    f"\nКадр источника: {observation.source_updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            if observation.annotated_frame_path and observation.annotated_frame_path.exists():
                with observation.annotated_frame_path.open("rb") as frame:
                    sent = await self._send_notification_with_retries(
                        camera=camera,
                        subscription=subscription,
                        method="send_photo",
                        chat_id=subscription.chat_id,
                        photo=frame,
                        caption=caption,
                    )
            else:
                sent = await self._send_notification_with_retries(
                    camera=camera,
                    subscription=subscription,
                    method="send_message",
                    chat_id=subscription.chat_id,
                    text=caption,
                )
            if not sent:
                continue
            self.repository.mark_notification(
                subscription_id=subscription.id,
                availability=Availability.FREE,
                sent_at=now,
            )

    async def _send_notification_with_retries(
        self,
        *,
        camera: ResolvedCamera,
        subscription: Any,
        method: str,
        **kwargs: Any,
    ) -> bool:
        if self._bot is None:
            return False

        delays = self._telegram_notification_retry_delays
        for attempt, delay in enumerate(delays, start=1):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                photo = kwargs.get("photo")
                if hasattr(photo, "seek"):
                    photo.seek(0)
                await getattr(self._bot, method)(**kwargs)
                return True
            except RetryAfter as exc:
                retry_after = min(float(exc.retry_after), 30.0)
                logger.warning(
                    "Telegram notification throttled for camera %s subscription %s chat %s "
                    "attempt %s/%s: retry after %.1f seconds",
                    camera.id,
                    subscription.id,
                    subscription.chat_id,
                    attempt,
                    len(delays),
                    retry_after,
                )
                if attempt < len(delays):
                    await asyncio.sleep(retry_after)
                continue
            except (TimedOut, NetworkError) as exc:
                logger.warning(
                    "Telegram notification failed for camera %s subscription %s chat %s "
                    "attempt %s/%s: %s: %s",
                    camera.id,
                    subscription.id,
                    subscription.chat_id,
                    attempt,
                    len(delays),
                    exc.__class__.__name__,
                    exc,
                )
                continue
            except Forbidden as exc:
                logger.warning(
                    "Telegram notification rejected for camera %s subscription %s chat %s: %s",
                    camera.id,
                    subscription.id,
                    subscription.chat_id,
                    exc,
                )
                return False
            except TelegramError as exc:
                logger.warning(
                    "Telegram notification failed for camera %s subscription %s chat %s: %s: %s",
                    camera.id,
                    subscription.id,
                    subscription.chat_id,
                    exc.__class__.__name__,
                    exc,
                )
                return False

        logger.warning(
            "Telegram notification skipped for camera %s subscription %s chat %s after %s attempts",
            camera.id,
            subscription.id,
            subscription.chat_id,
            len(delays),
        )
        return False
