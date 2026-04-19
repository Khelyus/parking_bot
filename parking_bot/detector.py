from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parking_bot.types import Availability, Detection


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


def _require_yolo() -> Any:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run 'python -m pip install -r requirements.txt' first."
        ) from exc
    return YOLO


@dataclass(slots=True)
class DetectionSummary:
    free_count: int
    occupied_count: int
    availability: Availability
    detections: list[Detection]
    annotated_frame: Any


@dataclass(slots=True)
class _DetectionCandidate:
    label: str
    confidence: float
    box: tuple[int, int, int, int]
    variant: str


class ParkingSpaceDetector:
    def __init__(self, model_path: Path, image_size: int = 640) -> None:
        self.model_path = Path(model_path)
        self.image_size = image_size
        self._model: Any | None = None
        self._free_ids: set[int] = set()
        self._occupied_ids: set[int] = set()

    def _ensure_model(self) -> Any:
        if self._model is None:
            YOLO = _require_yolo()
            self._model = YOLO(str(self.model_path))
            names = {int(idx): str(name).lower() for idx, name in self._model.names.items()}
            self._free_ids = {
                idx for idx, name in names.items() if "empty" in name or "free" in name
            }
            self._occupied_ids = {
                idx for idx, name in names.items() if "occupied" in name or "busy" in name
            }
            if not self._free_ids or not self._occupied_ids:
                raise RuntimeError(
                    "The model classes must include both free and occupied parking states"
                )
        return self._model

    def detect(
        self,
        image_path: Path,
        *,
        confidence: float,
        min_free_spaces: int,
        image_size: int | None = None,
    ) -> DetectionSummary:
        return self.detect_image(
            self._read_image(image_path),
            confidence=confidence,
            min_free_spaces=min_free_spaces,
            image_size=image_size,
        )

    def detect_image(
        self,
        image: Any,
        *,
        confidence: float,
        min_free_spaces: int,
        image_size: int | None = None,
    ) -> DetectionSummary:
        if image is None:
            raise RuntimeError("Detection image is empty")

        base_image = image.copy()
        image_height, image_width = base_image.shape[:2]
        target_size = self._resolve_target_image_size(
            image_size=image_size,
            image_width=image_width,
            image_height=image_height,
        )
        stages = self._build_detection_stages(
            base_image,
            requested_size=target_size,
            confidence=confidence,
        )
        fused_summary: DetectionSummary | None = None
        raw_candidates: list[_DetectionCandidate] = []

        for stage in stages:
            for variant_name, variant_image, variant_confidence, variant_size, remap_box in stage:
                raw_candidates.extend(
                    self._detect_candidates(
                        variant_image,
                        variant=variant_name,
                        confidence=variant_confidence,
                        image_size=variant_size,
                        remap_box=remap_box,
                    )
                )
                fused_summary = self._summarize_candidates(
                    base_image,
                    raw_candidates,
                    min_free_spaces=min_free_spaces,
                )

        if fused_summary is None:
            return DetectionSummary(
                free_count=0,
                occupied_count=0,
                availability=Availability.UNKNOWN,
                detections=[],
                annotated_frame=base_image,
            )
        return fused_summary

    def _read_image(self, image_path: Path) -> Any:
        cv2 = _require_cv2()
        np = _require_numpy()
        payload = Path(image_path).read_bytes()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode image from {image_path}")
        return image

    def _resolve_target_image_size(
        self,
        *,
        image_size: int | None,
        image_width: int,
        image_height: int,
    ) -> int:
        if image_size is not None:
            return image_size
        max_dimension = max(image_width, image_height)
        if max_dimension >= 2000:
            return 1536
        if max_dimension >= 1400:
            return 1280
        if max_dimension >= 900:
            return 960
        return max(640, self.image_size)

    def _build_detection_stages(
        self,
        image: Any,
        *,
        requested_size: int,
        confidence: float,
    ) -> list[list[tuple[str, Any, float, int, Any]]]:
        base_confidence = confidence
        relaxed_confidence = max(0.02, confidence * 0.6)
        contrast_image = self._apply_contrast_boost(image)
        stages: list[list[tuple[str, Any, float, int, Any]]] = [
            [
                ("orig", image, base_confidence, requested_size, self._identity_box),
                (
                    "bright",
                    self._apply_gamma(image, 0.85),
                    relaxed_confidence,
                    requested_size,
                    self._identity_box,
                ),
                (
                    "dark",
                    self._apply_gamma(image, 1.15),
                    relaxed_confidence,
                    requested_size,
                    self._identity_box,
                ),
            ],
            [
                (
                    "contrast",
                    contrast_image,
                    relaxed_confidence,
                    requested_size,
                    self._identity_box,
                ),
                (
                    "clahe",
                    self._apply_clahe(image),
                    relaxed_confidence,
                    requested_size,
                    self._identity_box,
                ),
                (
                    "sharp",
                    self._apply_sharpen(image),
                    relaxed_confidence,
                    requested_size,
                    self._identity_box,
                ),
            ],
        ]
        image_height, image_width = image.shape[:2]
        if max(image_width, image_height) >= 1400:
            cv2 = _require_cv2()
            proxy_width = 400
            proxy_height = max(1, round(image_height * proxy_width / image_width))
            proxy_image = cv2.resize(image, (proxy_width, proxy_height))
            remap_box = self._scale_box_mapper(
                scale_x=image_width / proxy_width,
                scale_y=image_height / proxy_height,
            )
            stages.append(
                [
                    ("proxy", proxy_image, base_confidence, 960, remap_box),
                    (
                        "proxy_bright",
                        self._apply_gamma(proxy_image, 0.85),
                        relaxed_confidence,
                        960,
                        remap_box,
                    ),
                ]
            )
            tile_size = min(1024, max(768, round(min(image_width, image_height) * 0.9)))
            tile_image_size = max(960, min(1280, requested_size))
            stages.append(
                self._build_tiled_stage(
                    image,
                    variant_prefix="tile",
                    confidence=max(0.015, confidence * 0.5),
                    image_size=tile_image_size,
                    tile_size=tile_size,
                )
            )
        return stages

    def _build_tiled_stage(
        self,
        image: Any,
        *,
        variant_prefix: str,
        confidence: float,
        image_size: int,
        tile_size: int,
    ) -> list[tuple[str, Any, float, int, Any]]:
        image_height, image_width = image.shape[:2]
        effective_tile_size = max(512, min(tile_size, max(image_width, image_height)))
        step = max(128, round(effective_tile_size * 0.72))
        x_positions = self._tile_positions(image_width, effective_tile_size, step)
        y_positions = self._tile_positions(image_height, effective_tile_size, step)
        stage: list[tuple[str, Any, float, int, Any]] = []
        for row_index, y1 in enumerate(y_positions):
            for column_index, x1 in enumerate(x_positions):
                x2 = min(image_width, x1 + effective_tile_size)
                y2 = min(image_height, y1 + effective_tile_size)
                tile = image[y1:y2, x1:x2]
                if tile.size == 0:
                    continue
                stage.append(
                    (
                        f"{variant_prefix}_{row_index}_{column_index}",
                        tile,
                        confidence,
                        image_size,
                        self._offset_box_mapper(x1, y1),
                    )
                )
        return stage

    def _detect_candidates(
        self,
        image: Any,
        *,
        variant: str,
        confidence: float,
        image_size: int,
        remap_box: Any,
    ) -> list[_DetectionCandidate]:
        model = self._ensure_model()
        result = model(
            image,
            verbose=False,
            conf=confidence,
            imgsz=image_size,
        )[0]
        candidates: list[_DetectionCandidate] = []
        if result.boxes is None:
            return candidates

        for box in result.boxes:
            class_id = int(box.cls.item())
            label = str(model.names[class_id])
            conf = float(box.conf.item())
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            mapped_box = remap_box((x1, y1, x2, y2))
            candidates.append(
                _DetectionCandidate(
                    label=label,
                    confidence=conf,
                    box=mapped_box,
                    variant=variant,
                )
            )
        return candidates

    def _summarize_candidates(
        self,
        image: Any,
        candidates: list[_DetectionCandidate],
        *,
        min_free_spaces: int,
    ) -> DetectionSummary:
        groups = self._cluster_candidates(candidates)
        detections = [
            detection
            for detection, group in groups
            if self._accept_cluster(
                label=detection.label,
                group=group,
                image_shape=image.shape[:2],
            )
        ]
        detections.sort(key=lambda item: item.confidence, reverse=True)

        free_count = sum(1 for detection in detections if self._is_free_label(detection.label))
        occupied_count = sum(1 for detection in detections if self._is_occupied_label(detection.label))
        if free_count >= min_free_spaces:
            availability = Availability.FREE
        elif occupied_count > 0 and free_count == 0:
            availability = Availability.FULL
        else:
            availability = Availability.UNKNOWN

        return DetectionSummary(
            free_count=free_count,
            occupied_count=occupied_count,
            availability=availability,
            detections=detections,
            annotated_frame=self._annotate_image(image, detections),
        )

    def _cluster_candidates(
        self,
        candidates: list[_DetectionCandidate],
    ) -> list[tuple[Detection, list[_DetectionCandidate]]]:
        groups: list[list[_DetectionCandidate]] = []
        ordered = sorted(candidates, key=lambda item: item.confidence, reverse=True)
        for candidate in ordered:
            if not self._is_box_reasonable(candidate.box):
                continue
            best_group: list[_DetectionCandidate] | None = None
            best_iou = 0.0
            for group in groups:
                if group[0].label != candidate.label:
                    continue
                iou = self._box_iou(candidate.box, self._merge_group_box(group))
                if iou >= 0.3 and iou > best_iou:
                    best_iou = iou
                    best_group = group
            if best_group is None:
                groups.append([candidate])
            else:
                best_group.append(candidate)

        merged: list[tuple[Detection, list[_DetectionCandidate]]] = []
        for group in groups:
            merged.append((self._merge_group_detection(group), group))
        return merged

    def _merge_group_detection(self, group: list[_DetectionCandidate]) -> Detection:
        total_weight = sum(item.confidence for item in group) or float(len(group))
        x1 = round(sum(item.box[0] * item.confidence for item in group) / total_weight)
        y1 = round(sum(item.box[1] * item.confidence for item in group) / total_weight)
        x2 = round(sum(item.box[2] * item.confidence for item in group) / total_weight)
        y2 = round(sum(item.box[3] * item.confidence for item in group) / total_weight)
        return Detection(
            label=group[0].label,
            confidence=max(item.confidence for item in group),
            box=(x1, y1, x2, y2),
        )

    def _merge_group_box(self, group: list[_DetectionCandidate]) -> tuple[int, int, int, int]:
        merged = self._merge_group_detection(group)
        return merged.box

    def _accept_cluster(
        self,
        *,
        label: str,
        group: list[_DetectionCandidate],
        image_shape: tuple[int, int],
    ) -> bool:
        detection = self._merge_group_detection(group)
        width = max(1, detection.box[2] - detection.box[0])
        height = max(1, detection.box[3] - detection.box[1])
        image_height, image_width = image_shape
        image_area = image_height * image_width
        area = width * height
        votes = len(group)
        max_confidence = max(item.confidence for item in group)
        avg_confidence = sum(item.confidence for item in group) / votes

        if width < 6 or height < 6:
            return False
        if area > image_area * 0.25:
            return False

        if self._is_free_label(label):
            return (
                max_confidence >= 0.12
                or (votes >= 2 and avg_confidence >= 0.035)
                or (votes >= 3 and max_confidence >= 0.03)
            )

        return (
            max_confidence >= 0.10
            or (votes >= 2 and avg_confidence >= 0.035)
            or (votes >= 3 and max_confidence >= 0.03)
        )

    def _annotate_image(self, image: Any, detections: list[Detection]) -> Any:
        cv2 = _require_cv2()
        annotated = image.copy()
        for detection in detections:
            color = (0, 200, 80) if self._is_free_label(detection.label) else (0, 90, 220)
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

    def _apply_clahe(self, image: Any) -> Any:
        cv2 = _require_cv2()
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(l_channel)
        return cv2.cvtColor(
            cv2.merge((equalized, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

    def _apply_contrast_boost(self, image: Any) -> Any:
        cv2 = _require_cv2()
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        normalized = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        boosted = clahe.apply(normalized)
        return cv2.cvtColor(
            cv2.merge((boosted, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

    def _apply_sharpen(self, image: Any) -> Any:
        cv2 = _require_cv2()
        np = _require_numpy()
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)

    def _apply_gamma(self, image: Any, gamma: float) -> Any:
        cv2 = _require_cv2()
        np = _require_numpy()
        table = np.array(
            [((index / 255.0) ** gamma) * 255 for index in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image, table)

    def _is_box_reasonable(self, box: tuple[int, int, int, int]) -> bool:
        width = box[2] - box[0]
        height = box[3] - box[1]
        return width > 2 and height > 2

    def _identity_box(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return box

    def _offset_box_mapper(self, offset_x: int, offset_y: int) -> Any:
        def remap(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            return (
                box[0] + offset_x,
                box[1] + offset_y,
                box[2] + offset_x,
                box[3] + offset_y,
            )

        return remap

    def _scale_box_mapper(self, scale_x: float, scale_y: float) -> Any:
        def remap(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
            return (
                round(box[0] * scale_x),
                round(box[1] * scale_y),
                round(box[2] * scale_x),
                round(box[3] * scale_y),
            )

        return remap

    def _tile_positions(self, length: int, tile_size: int, step: int) -> list[int]:
        if length <= tile_size:
            return [0]
        last_position = length - tile_size
        positions = list(range(0, last_position + 1, step))
        if not positions or positions[-1] != last_position:
            positions.append(last_position)
        return positions

    def _box_iou(
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
        right_area = max(1, (right[2] - right[0]) * (right[3] - right[1]))
        union = left_area + right_area - intersection
        return intersection / union if union > 0 else 0.0

    def _is_free_label(self, label: str) -> bool:
        normalized = label.lower()
        return "empty" in normalized or "free" in normalized

    def _is_occupied_label(self, label: str) -> bool:
        normalized = label.lower()
        return "occupied" in normalized or "busy" in normalized


class VehicleDetector:
    def __init__(self, model_path: str | Path, image_size: int = 1280) -> None:
        self.model_path = str(model_path)
        self.image_size = image_size
        self._model: Any | None = None
        self._vehicle_labels = {"car", "truck", "bus", "motorcycle"}

    def _ensure_model(self) -> Any:
        if self._model is None:
            YOLO = _require_yolo()
            self._model = YOLO(self.model_path)
        return self._model

    def detect_image(
        self,
        image: Any,
        *,
        confidence: float,
        image_size: int | None = None,
    ) -> list[Detection]:
        if image is None:
            raise RuntimeError("Vehicle detection image is empty")

        model = self._ensure_model()
        result = model(
            image,
            verbose=False,
            conf=confidence,
            imgsz=image_size or self.image_size,
        )[0]
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls.item())
            label = str(model.names[class_id]).lower()
            if label not in self._vehicle_labels:
                continue
            detections.append(
                Detection(
                    label=label,
                    confidence=float(box.conf.item()),
                    box=tuple(int(v) for v in box.xyxy[0].tolist()),
                )
            )
        return detections
