from types import SimpleNamespace

from parking_bot.service import CameraMonitorService
from parking_bot.slot_classifier import SlotPrediction
from parking_bot.types import Detection


def _build_service() -> CameraMonitorService:
    service = CameraMonitorService.__new__(CameraMonitorService)
    service.settings = SimpleNamespace(vehicle_confidence=0.05, slot_classifier_confidence=0.58)
    return service


def test_match_vehicle_to_slot_prefers_anchor_hit() -> None:
    service = _build_service()
    slot_box = (100, 100, 200, 260)
    overlap_only = Detection(label="car", confidence=0.95, box=(40, 120, 130, 240))
    anchor_hit = Detection(label="car", confidence=0.61, box=(120, 120, 180, 230))

    matched = service._match_vehicle_to_slot(slot_box, [overlap_only, anchor_hit])

    assert matched == anchor_hit


def test_vehicle_anchor_point_is_taken_from_the_lower_part_of_the_box() -> None:
    service = _build_service()
    vehicle_box = (100, 120, 220, 320)

    anchor = service._vehicle_anchor_point(vehicle_box)

    assert anchor == (160, 296)


def test_match_vehicle_to_slot_accepts_center_and_overlap_support() -> None:
    service = _build_service()
    slot_box = (100, 100, 200, 220)
    overlap_match = Detection(label="car", confidence=0.72, box=(95, 80, 190, 180))

    matched = service._match_vehicle_to_slot(slot_box, [overlap_match])

    assert matched == overlap_match


def test_free_detection_requires_same_row_vehicle_support() -> None:
    service = _build_service()
    detection = Detection(label="space-empty", confidence=0.31, box=(100, 60, 150, 120))
    far_below_vehicles = [
        Detection(label="car", confidence=0.9, box=(90, 260, 150, 330)),
        Detection(label="car", confidence=0.88, box=(155, 255, 215, 325)),
    ]

    supported = service._is_supported_free_detection(
        detection,
        occupied_detections=[],
        vehicle_detections=far_below_vehicles,
        image_shape=(500, 500),
    )

    assert supported is False


def test_free_detection_accepts_same_row_vehicle_support() -> None:
    service = _build_service()
    detection = Detection(label="space-empty", confidence=0.31, box=(100, 180, 150, 250))
    same_row_vehicles = [
        Detection(label="car", confidence=0.9, box=(35, 175, 95, 255)),
        Detection(label="car", confidence=0.88, box=(160, 178, 220, 258)),
    ]

    supported = service._is_supported_free_detection(
        detection,
        occupied_detections=[],
        vehicle_detections=same_row_vehicles,
        image_shape=(500, 500),
    )

    assert supported is True


def test_resolve_slot_status_prefers_classifier_when_vehicle_is_absent() -> None:
    service = _build_service()

    label, confidence = service._resolve_slot_status(
        vehicle_score=0.0,
        local_vehicle_score=0.0,
        slot_prediction=SlotPrediction(
            label="space-occupied",
            confidence=0.91,
            empty_probability=0.09,
            occupied_probability=0.91,
        ),
    )

    assert label == "space-occupied"
    assert confidence == 0.91


def test_resolve_slot_status_keeps_strong_vehicle_match() -> None:
    service = _build_service()

    label, confidence = service._resolve_slot_status(
        vehicle_score=1.45,
        local_vehicle_score=0.0,
        slot_prediction=SlotPrediction(
            label="space-empty",
            confidence=0.79,
            empty_probability=0.79,
            occupied_probability=0.21,
        ),
    )

    assert label == "space-occupied"
    assert confidence > 0.80


def test_resolve_slot_status_falls_back_to_empty_without_signals() -> None:
    service = _build_service()

    label, confidence = service._resolve_slot_status(
        vehicle_score=0.0,
        local_vehicle_score=0.0,
        slot_prediction=None,
    )

    assert label == "space-occupied"
    assert confidence >= 0.50


def test_resolve_slot_status_local_vehicle_vetoes_false_empty() -> None:
    service = _build_service()

    label, confidence = service._resolve_slot_status(
        vehicle_score=0.0,
        local_vehicle_score=1.10,
        slot_prediction=SlotPrediction(
            label="space-empty",
            confidence=0.83,
            empty_probability=0.83,
            occupied_probability=0.17,
        ),
    )

    assert label == "space-occupied"
    assert confidence >= 0.50
