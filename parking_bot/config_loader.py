from __future__ import annotations

from pathlib import Path

from parking_bot.settings import Settings
from parking_bot.types import CameraConfig, ParkingSlot
from parking_bot.simple_yaml import load_simple_yaml

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


def _load_config_payload(path: Path) -> dict[str, object]:
    if yaml is not None:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return load_simple_yaml(path)


def _parse_normalized_box(
    value: object,
    camera_id: str,
    field_name: str,
) -> tuple[float, float, float, float] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = [str(part).strip() for part in value]
    else:
        raise RuntimeError(
            f"Unsupported {field_name} for camera {camera_id!r}; expected 'x1,y1,x2,y2'"
        )

    if len(parts) != 4:
        raise RuntimeError(
            f"Camera {camera_id!r} must define {field_name} as four normalized numbers"
        )

    try:
        x1, y1, x2, y2 = [float(part) for part in parts]
    except ValueError as exc:
        raise RuntimeError(
            f"Camera {camera_id!r} contains an invalid {field_name} value"
        ) from exc

    if not all(0.0 <= part <= 1.0 for part in (x1, y1, x2, y2)):
        raise RuntimeError(
            f"Camera {camera_id!r} {field_name} values must be between 0.0 and 1.0"
        )
    if x1 >= x2 or y1 >= y2:
        raise RuntimeError(
            f"Camera {camera_id!r} {field_name} must satisfy x1 < x2 and y1 < y2"
    )
    return (x1, y1, x2, y2)


def _parse_normalized_boxes(
    value: object,
    camera_id: str,
    field_name: str,
) -> tuple[tuple[float, float, float, float], ...]:
    if value is None or value == "":
        return ()

    entries: list[object]
    if isinstance(value, str):
        entries = [entry.strip() for entry in value.split(";") if entry.strip()]
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        raise RuntimeError(
            f"Unsupported {field_name} for camera {camera_id!r}; expected a list or ';' separated string"
        )

    boxes: list[tuple[float, float, float, float]] = []
    for index, entry in enumerate(entries, start=1):
        box_source = entry
        if isinstance(entry, dict):
            box_source = entry.get("box", entry.get("bounds"))
        box = _parse_normalized_box(
            box_source,
            camera_id,
            f"{field_name}[{index}]",
        )
        if box is None:
            continue
        boxes.append(box)
    return tuple(boxes)


def _parse_parking_slots(value: object, camera_id: str) -> tuple[ParkingSlot, ...]:
    if value is None or value == "":
        return ()

    slots: list[ParkingSlot] = []
    if isinstance(value, str):
        entries = [entry.strip() for entry in value.split(";") if entry.strip()]
        for index, entry in enumerate(entries, start=1):
            slot_id = f"slot_{index}"
            box_source: object = entry
            if "=" in entry:
                raw_slot_id, raw_box = entry.split("=", 1)
                slot_id = raw_slot_id.strip() or slot_id
                box_source = raw_box.strip()
            box = _parse_normalized_box(
                box_source,
                camera_id,
                f"parking_slots[{slot_id}]",
            )
            if box is None:
                continue
            slots.append(ParkingSlot(id=slot_id, box=box, angle_degrees=0.0))
        return tuple(slots)

    if not isinstance(value, (list, tuple)):
        raise RuntimeError(
            f"Unsupported parking_slots for camera {camera_id!r}; expected a list or ';' separated string"
        )

    for index, item in enumerate(value, start=1):
        slot_id = f"slot_{index}"
        box_source: object = item
        if isinstance(item, dict):
            slot_id = str(item.get("id") or slot_id)
            box_source = item.get("box", item.get("bounds"))
            raw_angle = item.get("angle", item.get("angle_degrees", 0.0))
        else:
            raw_angle = 0.0
        box = _parse_normalized_box(
            box_source,
            camera_id,
            f"parking_slots[{slot_id}]",
        )
        if box is None:
            continue
        try:
            angle_degrees = float(raw_angle)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Camera {camera_id!r} parking_slots[{slot_id}] contains an invalid angle"
            ) from exc
        slots.append(ParkingSlot(id=slot_id, box=box, angle_degrees=angle_degrees))
    return tuple(slots)


def load_camera_configs(path: Path, settings: Settings) -> list[CameraConfig]:
    if not path.exists():
        raise RuntimeError(f"Camera config file was not found: {path}")

    raw = _load_config_payload(path)
    items = raw.get("cameras") or []
    configs: list[CameraConfig] = []

    for item in items:
        camera_id = str(item["id"])
        source_kind = str(item.get("source_kind", "ufanet")).lower()
        if source_kind not in {"ufanet", "demo"}:
            raise RuntimeError(
                f"Unsupported source_kind '{source_kind}' for camera {item.get('id', '<unknown>')}"
            )
        source_transport = str(item.get("source_transport", "snapshot")).lower()
        if source_transport not in {"snapshot", "hls"}:
            raise RuntimeError(
                f"Unsupported source_transport '{source_transport}' for camera {item.get('id', '<unknown>')}"
            )
        detection_roi = _parse_normalized_box(item.get("detection_roi"), camera_id, "detection_roi")
        detection_exclude_rois = _parse_normalized_boxes(
            item.get("detection_exclude_rois"),
            camera_id,
            "detection_exclude_rois",
        )
        parking_slots = _parse_parking_slots(item.get("parking_slots"), camera_id)
        demo_observations_path = item.get("demo_observations_file")
        resolved_demo_path = None
        if demo_observations_path:
            candidate = Path(str(demo_observations_path))
            resolved_demo_path = candidate if candidate.is_absolute() else settings.project_dir / candidate
        configs.append(
            CameraConfig(
                id=camera_id,
                map_url=str(item.get("map_url", "")),
                enabled=bool(item.get("enabled", True)),
                display_name=item.get("display_name"),
                source_kind=source_kind,
                source_transport=source_transport,
                min_confidence=float(item.get("min_confidence", settings.default_confidence)),
                min_free_spaces=int(item.get("min_free_spaces", 1)),
                detection_image_size=(
                    int(item["detection_image_size"])
                    if item.get("detection_image_size") is not None
                    else None
                ),
                detection_roi=detection_roi,
                detection_exclude_rois=detection_exclude_rois,
                parking_slots=parking_slots,
                stable_cycles=int(item.get("stable_cycles", settings.default_stable_cycles)),
                poll_interval_seconds=(
                    int(item["poll_interval_seconds"])
                    if item.get("poll_interval_seconds") is not None
                    else None
                ),
                demo_observations_path=resolved_demo_path,
                demo_loop=bool(item.get("demo_loop", True)),
            )
        )

    if not configs:
        raise RuntimeError(f"No cameras were configured in {path}")
    return configs
