from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_bot.config_loader import load_camera_configs
from parking_bot.settings import load_settings
from parking_bot.types import ParkingSlot


def _require_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("OpenCV is required for the parking slot editor.") from exc
    return cv2


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for the parking slot editor.") from exc
    return np


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


@dataclass
class EditableSlot:
    id: str
    box: tuple[int, int, int, int]
    angle_degrees: float = 0.0
    included: bool = True


@dataclass
class EditorState:
    image_paths: list[Path]
    image_index: int
    image_shape: tuple[int, int]
    slots: list[EditableSlot]
    selected_index: int = -1
    show_help: bool = True
    dirty: bool = False
    drag_mode: str | None = None
    drag_anchor: tuple[int, int] | None = None
    drag_origin_box: tuple[int, int, int, int] | None = None
    draft_box: tuple[int, int, int, int] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive OpenCV editor for parking_slots in config/cameras.yaml."
    )
    parser.add_argument("--camera-id", help="Camera id from config/cameras.yaml.")
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--frame",
        help="Optional explicit frame path. If omitted, uses runtime/training_frames/<camera_id>/ and falls back to runtime/frames/<camera_id>_raw.jpg.",
    )
    parser.add_argument(
        "--frames-dir",
        default="runtime/training_frames",
        help="Archived frames root used when --frame is omitted.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Maximum preview width.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1000,
        help="Maximum preview height.",
    )
    parser.add_argument(
        "--output-dir",
        default="runtime/slot_editor",
        help="Directory for preview images and legacy editor artifacts.",
    )
    return parser.parse_args()


def _resolve_project_root(raw_project_root: str) -> Path:
    candidate = Path(raw_project_root)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _load_camera_configs(project_root: Path) -> tuple[Any, list[Any]]:
    settings = load_settings(project_root, require_telegram_token=False)
    configs = load_camera_configs(settings.camera_config_path, settings)
    return settings, configs


def _choose_camera_interactively(project_root: Path, explicit_camera_id: str | None) -> str:
    if explicit_camera_id:
        return explicit_camera_id

    settings, configs = _load_camera_configs(project_root)
    selectable = [camera for camera in configs if camera.enabled]
    if not selectable:
        raise RuntimeError(f"No enabled cameras were found in {settings.camera_config_path}")

    print("Select camera:")
    for index, camera in enumerate(selectable, start=1):
        display_name = camera.display_name or camera.id
        print(f"  {index}. {camera.id} | {display_name} | slots={len(camera.parking_slots)}")

    while True:
        raw = input("Enter camera number or id: ").strip()
        if not raw:
            continue
        if raw.isdigit():
            selected_index = int(raw) - 1
            if 0 <= selected_index < len(selectable):
                return selectable[selected_index].id
        for camera in selectable:
            if raw == camera.id:
                return camera.id
        print("Invalid selection. Try again.")


def _load_camera(project_root: Path, camera_id: str) -> tuple[Any, list[ParkingSlot]]:
    _settings, configs = _load_camera_configs(project_root)
    for camera in configs:
        if camera.id == camera_id:
            return camera, list(camera.parking_slots)
    raise RuntimeError(f"Camera {camera_id!r} was not found in the configured cameras list")


def _editor_state_path(output_dir: Path, camera_id: str) -> Path:
    return output_dir / f"{camera_id}.editor_state.json"


def _prod_slots_path(camera_config_path: Path, camera_id: str) -> Path:
    return camera_config_path.parent / "parking_slots" / f"{camera_id}.json"


def _collect_image_paths(project_root: Path, camera_id: str, frame_arg: str | None, frames_dir_arg: str) -> list[Path]:
    if frame_arg:
        path = Path(frame_arg)
        resolved = path if path.is_absolute() else (project_root / path).resolve()
        if not resolved.exists():
            raise RuntimeError(f"Frame was not found: {resolved}")
        return [resolved]

    frames_dir = (project_root / frames_dir_arg).resolve()
    camera_dir = frames_dir / camera_id
    paths: list[Path] = []
    if camera_dir.exists():
        paths.extend(
            sorted(
                path
                for path in camera_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
        )
    raw_fallback = (project_root / "runtime" / "frames" / f"{camera_id}_raw.jpg").resolve()
    if not paths and raw_fallback.exists():
        paths.append(raw_fallback)
    if not paths:
        raise RuntimeError(
            f"No frames found for {camera_id!r}. "
            f"Looked in {camera_dir} and {raw_fallback}."
        )
    return paths


def _load_image(path: Path) -> Any:
    cv2 = _require_cv2()
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not decode image: {path}")
    return image


def _normalized_to_pixels(slot: ParkingSlot, image_width: int, image_height: int) -> EditableSlot:
    x1 = _clamp(round(slot.box[0] * image_width), 0, image_width - 1)
    y1 = _clamp(round(slot.box[1] * image_height), 0, image_height - 1)
    x2 = _clamp(round(slot.box[2] * image_width), x1 + 1, image_width)
    y2 = _clamp(round(slot.box[3] * image_height), y1 + 1, image_height)
    return EditableSlot(id=slot.id, box=(x1, y1, x2, y2), angle_degrees=slot.angle_degrees)


def _pixels_to_normalized(box: tuple[int, int, int, int], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        round(x1 / image_width, 3),
        round(y1 / image_height, 3),
        round(x2 / image_width, 3),
        round(y2 / image_height, 3),
    )


def _load_editor_slots(
    state_paths: list[Path],
    image_width: int,
    image_height: int,
) -> list[EditableSlot] | None:
    for state_path in state_paths:
        if not state_path.exists():
            continue

        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Warning: could not load saved editor state from {state_path}: {exc}")
            continue

        raw_slots = payload.get("slots")
        if not isinstance(raw_slots, list):
            continue

        slots: list[EditableSlot] = []
        for index, item in enumerate(raw_slots, start=1):
            if not isinstance(item, dict):
                continue
            slot_id = str(item.get("id") or f"slot_{index}")
            raw_box = item.get("box")
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in raw_box]
                angle_degrees = float(item.get("angle_degrees", 0.0))
            except (TypeError, ValueError):
                continue
            normalized_box = (
                max(0.0, min(1.0, x1)),
                max(0.0, min(1.0, y1)),
                max(0.0, min(1.0, x2)),
                max(0.0, min(1.0, y2)),
            )
            slot = _normalized_to_pixels(
                ParkingSlot(id=slot_id, box=normalized_box, angle_degrees=angle_degrees),
                image_width,
                image_height,
            )
            slot.included = bool(item.get("included", True))
            slots.append(slot)
        return slots
    return None


def _fit_scale(image_width: int, image_height: int, max_width: int, max_height: int) -> float:
    scale = min(max_width / image_width, max_height / image_height, 1.0)
    return max(scale, 0.1)


def _slot_area(box: tuple[int, int, int, int]) -> int:
    return max(1, box[2] - box[0]) * max(1, box[3] - box[1])


def _slot_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _rotate_local_to_world(dx: float, dy: float, angle_degrees: float) -> tuple[float, float]:
    radians = math.radians(angle_degrees)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    return (
        (dx * cos_angle) - (dy * sin_angle),
        (dx * sin_angle) + (dy * cos_angle),
    )


def _rotate_world_to_local(dx: float, dy: float, angle_degrees: float) -> tuple[float, float]:
    radians = math.radians(angle_degrees)
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)
    return (
        (dx * cos_angle) + (dy * sin_angle),
        (-dx * sin_angle) + (dy * cos_angle),
    )


def _slot_polygon(slot: EditableSlot) -> list[tuple[int, int]]:
    center_x, center_y = _slot_center(slot.box)
    width = max(1, slot.box[2] - slot.box[0])
    height = max(1, slot.box[3] - slot.box[1])
    half_width = width / 2
    half_height = height / 2
    corners: list[tuple[int, int]] = []
    for dx, dy in (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ):
        world_dx, world_dy = _rotate_local_to_world(dx, dy, slot.angle_degrees)
        corners.append((round(center_x + world_dx), round(center_y + world_dy)))
    return corners


def _slot_bounds(slot: EditableSlot) -> tuple[int, int, int, int]:
    polygon = _slot_polygon(slot)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _slot_contains(slot: EditableSlot, x: int, y: int) -> bool:
    cv2 = _require_cv2()
    np = _require_numpy()
    polygon = np.array(_slot_polygon(slot), dtype=np.float32)
    return cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0


def _find_slot_index(slots: list[EditableSlot], x: int, y: int) -> int:
    matches = [index for index, slot in enumerate(slots) if _slot_contains(slot, x, y)]
    if not matches:
        return -1
    return min(matches, key=lambda index: _slot_area(slots[index].box))


def _corner_hit(slot: EditableSlot, x: int, y: int, radius: int = 14) -> str | None:
    points = {
        "resize_tl": _slot_polygon(slot)[0],
        "resize_tr": _slot_polygon(slot)[1],
        "resize_br": _slot_polygon(slot)[2],
        "resize_bl": _slot_polygon(slot)[3],
    }
    for name, (px, py) in points.items():
        if abs(x - px) <= radius and abs(y - py) <= radius:
            return name
    return None


def _next_slot_id(slots: list[EditableSlot]) -> str:
    used = {slot.id for slot in slots}
    index = 1
    while f"slot_{index}" in used:
        index += 1
    return f"slot_{index}"


def _key_char(key: int) -> str | None:
    if 0 <= key <= 0xFF:
        char = chr(key)
        if char.isprintable():
            return char.lower()
    return None


def _arrow_key_codes() -> tuple[set[int], set[int]]:
    if sys.platform.startswith("win"):
        # On Windows `waitKeyEx` reports arrow keys as extended virtual-key values.
        # Plain letters such as `S` must remain available for editor shortcuts.
        return ({2424832}, {2555904})
    return ({81, 65361}, {83, 65363})


def _duplicate_slot(
    slot: EditableSlot,
    slots: list[EditableSlot],
    image_width: int,
    image_height: int,
) -> EditableSlot:
    x1, y1, x2, y2 = slot.box
    width = x2 - x1
    height = y2 - y1
    offset_x = max(8, round(width * 0.18))
    offset_y = max(8, round(height * 0.12))
    new_x1 = _clamp(x1 + offset_x, 0, image_width - width)
    new_y1 = _clamp(y1 + offset_y, 0, image_height - height)
    return EditableSlot(
        id=_next_slot_id(slots),
        box=(new_x1, new_y1, new_x1 + width, new_y1 + height),
        angle_degrees=slot.angle_degrees,
        included=True,
    )


def _normalize_pixel_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left = _clamp(min(x1, x2), 0, image_width - 1)
    top = _clamp(min(y1, y2), 0, image_height - 1)
    right = _clamp(max(x1, x2), left + 1, image_width)
    bottom = _clamp(max(y1, y2), top + 1, image_height)
    return (left, top, right, bottom)


def _update_slot_from_drag(
    origin_slot: EditableSlot,
    drag_mode: str,
    anchor_x: int,
    anchor_y: int,
    current_x: int,
    current_y: int,
    image_width: int,
    image_height: int,
) -> EditableSlot:
    x1, y1, x2, y2 = origin_slot.box
    dx = current_x - anchor_x
    dy = current_y - anchor_y
    if drag_mode == "move":
        width = x2 - x1
        height = y2 - y1
        new_x1 = _clamp(x1 + dx, 0, image_width - width)
        new_y1 = _clamp(y1 + dy, 0, image_height - height)
        return EditableSlot(
            id=origin_slot.id,
            box=(new_x1, new_y1, new_x1 + width, new_y1 + height),
            angle_degrees=origin_slot.angle_degrees,
            included=origin_slot.included,
        )

    center_x, center_y = _slot_center(origin_slot.box)
    half_width = max(1.0, (x2 - x1) / 2)
    half_height = max(1.0, (y2 - y1) / 2)
    left = -half_width
    right = half_width
    top = -half_height
    bottom = half_height
    local_x, local_y = _rotate_world_to_local(current_x - center_x, current_y - center_y, origin_slot.angle_degrees)
    minimum_size = 6.0
    if drag_mode == "resize_tl":
        left = min(local_x, right - minimum_size)
        top = min(local_y, bottom - minimum_size)
    elif drag_mode == "resize_tr":
        right = max(local_x, left + minimum_size)
        top = min(local_y, bottom - minimum_size)
    elif drag_mode == "resize_bl":
        left = min(local_x, right - minimum_size)
        bottom = max(local_y, top + minimum_size)
    elif drag_mode == "resize_br":
        right = max(local_x, left + minimum_size)
        bottom = max(local_y, top + minimum_size)
    else:
        raise RuntimeError(f"Unsupported drag mode: {drag_mode}")

    local_center_shift = ((left + right) / 2, (top + bottom) / 2)
    world_shift = _rotate_local_to_world(local_center_shift[0], local_center_shift[1], origin_slot.angle_degrees)
    new_center_x = center_x + world_shift[0]
    new_center_y = center_y + world_shift[1]
    new_width = right - left
    new_height = bottom - top
    new_box = _normalize_pixel_box(
        (
            round(new_center_x - new_width / 2),
            round(new_center_y - new_height / 2),
            round(new_center_x + new_width / 2),
            round(new_center_y + new_height / 2),
        ),
        image_width,
        image_height,
    )
    return EditableSlot(
        id=origin_slot.id,
        box=new_box,
        angle_degrees=origin_slot.angle_degrees,
        included=origin_slot.included,
    )


def _render(
    state: EditorState,
    image: Any,
    camera_id: str,
    scale: float,
    max_width: int,
    max_height: int,
) -> Any:
    cv2 = _require_cv2()
    np = _require_numpy()
    canvas = image.copy()
    overlay = image.copy()

    for index, slot in enumerate(state.slots):
        selected = index == state.selected_index
        if slot.included:
            color = (0, 220, 90) if selected else (0, 140, 255)
            fill_color = color
            fill_alpha = 0.12
            label = slot.id
        else:
            color = (0, 120, 255) if selected else (140, 140, 140)
            fill_color = (90, 90, 90)
            fill_alpha = 0.05
            label = f"{slot.id} [off]"
        polygon = np.array(_slot_polygon(slot), dtype=np.int32)
        tinted_overlay = overlay.copy()
        cv2.fillConvexPoly(tinted_overlay, polygon, fill_color)
        cv2.addWeighted(tinted_overlay, fill_alpha, overlay, 1.0 - fill_alpha, 0, overlay)
        cv2.polylines(canvas, [polygon], isClosed=True, color=color, thickness=2 if selected else 1)
        handle_radius = 5 if selected else 3
        for point in polygon:
            cv2.circle(canvas, point, handle_radius, color, -1)
        bounds = _slot_bounds(slot)
        label_x = bounds[0] + 4
        label_y = bounds[1] - 8 if bounds[1] > 24 else bounds[1] + 18
        cv2.putText(
            canvas,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    if state.draft_box is not None:
        x1, y1, x2, y2 = _normalize_pixel_box(state.draft_box, state.image_shape[1], state.image_shape[0])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)

    lines = [
        f"camera={camera_id}",
        f"frame={state.image_index + 1}/{len(state.image_paths)}",
        f"slots={sum(1 for slot in state.slots if slot.included)}/{len(state.slots)} active",
        "dirty=yes" if state.dirty else "dirty=no",
    ]
    if 0 <= state.selected_index < len(state.slots):
        slot = state.slots[state.selected_index]
        norm_box = _pixels_to_normalized(slot.box, state.image_shape[1], state.image_shape[0])
        lines.append(
            "selected="
            f"{slot.id} {norm_box[0]:.3f},{norm_box[1]:.3f},{norm_box[2]:.3f},{norm_box[3]:.3f}"
            f" angle={slot.angle_degrees:.1f} included={'yes' if slot.included else 'no'}"
        )

    help_lines = [
        "Mouse: drag inside slot to move, drag corners to resize, shift+drag to create",
        "Keys: left/right frame | tab next slot | c clone | x include/exclude | ,/. rotate | d delete | s save | h help | q quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26

    if state.show_help:
        y += 8
        for line in help_lines:
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2, cv2.LINE_AA)
            y += 24

    if scale != 1.0:
        resized = cv2.resize(
            canvas,
            (min(max_width, round(state.image_shape[1] * scale)), min(max_height, round(state.image_shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized
    return canvas


def _serialize_slot(slot: EditableSlot, image_width: int, image_height: int) -> dict[str, object]:
    x1, y1, x2, y2 = _pixels_to_normalized(slot.box, image_width, image_height)
    return {
        "id": slot.id,
        "box": [x1, y1, x2, y2],
        "angle_degrees": round(slot.angle_degrees, 1),
        "included": slot.included,
    }


def _serialize_prod_slots_payload(state: EditorState, camera_id: str) -> dict[str, object]:
    return {
        "camera_id": camera_id,
        "image_shape": [state.image_shape[0], state.image_shape[1]],
        "slots": [
            _serialize_slot(slot, state.image_shape[1], state.image_shape[0])
            for slot in state.slots
        ],
    }


def _format_slot_yaml_lines(slots: list[EditableSlot], image_width: int, image_height: int) -> list[str]:
    if not slots:
        return []

    lines = ["    parking_slots:"]
    for slot in slots:
        x1, y1, x2, y2 = _pixels_to_normalized(slot.box, image_width, image_height)
        lines.extend(
            [
                f"      - id: {slot.id}",
                f"        box: {x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}",
            ]
        )
        if abs(slot.angle_degrees) >= 0.05:
            lines.append(f"        angle: {slot.angle_degrees:.1f}")
    return lines


def _normalize_scalar_text(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _replace_camera_parking_slots(config_text: str, camera_id: str, slot_lines: list[str]) -> str:
    lines = config_text.splitlines()
    camera_start = -1
    camera_end = len(lines)
    for index, line in enumerate(lines):
        if not line.startswith("  - id:"):
            continue
        stripped = line.strip()
        current_camera_id = _normalize_scalar_text(stripped.split(":", 1)[1])
        if current_camera_id == camera_id:
            camera_start = index
            continue
        if camera_start >= 0:
            camera_end = index
            break

    if camera_start < 0:
        raise RuntimeError(f"Camera {camera_id!r} was not found in the YAML config")

    parking_start = -1
    parking_end = camera_end
    stable_cycles_index = -1
    for index in range(camera_start + 1, camera_end):
        line = lines[index]
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if indent == 4 and stripped == "parking_slots:":
            parking_start = index
            parking_end = index + 1
            while parking_end < camera_end:
                nested_line = lines[parking_end]
                if not nested_line.strip():
                    parking_end += 1
                    continue
                nested_indent = len(nested_line) - len(nested_line.lstrip(" "))
                if nested_indent <= 4:
                    break
                parking_end += 1
            break
        if indent == 4 and stripped.startswith("stable_cycles:") and stable_cycles_index < 0:
            stable_cycles_index = index

    new_lines = list(lines)
    if parking_start >= 0:
        new_lines[parking_start:parking_end] = slot_lines
    elif slot_lines:
        insert_at = stable_cycles_index if stable_cycles_index >= 0 else camera_end
        new_lines[insert_at:insert_at] = slot_lines

    trailing_newline = "\n" if config_text.endswith("\n") else ""
    return "\n".join(new_lines) + trailing_newline


def _save_outputs(
    *,
    state: EditorState,
    image: Any,
    camera_id: str,
    output_dir: Path,
    camera_config_path: Path,
) -> tuple[Path, Path, Path]:
    cv2 = _require_cv2()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / f"{camera_id}.preview.jpg"
    state_path = _editor_state_path(output_dir, camera_id)
    prod_slots_state_path = _prod_slots_path(camera_config_path, camera_id)

    prod_slots_state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize_prod_slots_payload(state, camera_id)
    serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    prod_slots_state_path.write_text(serialized_payload, encoding="utf-8")
    state_path.write_text(serialized_payload, encoding="utf-8")

    active_slots = [slot for slot in state.slots if slot.included]
    slot_lines = _format_slot_yaml_lines(active_slots, state.image_shape[1], state.image_shape[0])
    updated_yaml = _replace_camera_parking_slots(
        camera_config_path.read_text(encoding="utf-8"),
        camera_id,
        slot_lines,
    )
    camera_config_path.write_text(updated_yaml, encoding="utf-8")

    preview = _render(state, image, camera_id, scale=1.0, max_width=state.image_shape[1], max_height=state.image_shape[0])
    cv2.imwrite(str(preview_path), preview)
    state.dirty = False
    return prod_slots_state_path, camera_config_path, preview_path


def _install_mouse_handler(state: EditorState, scale: float) -> None:
    cv2 = _require_cv2()

    def handle(event: int, x: int, y: int, flags: int, _param: object) -> None:
        image_x = _clamp(round(x / scale), 0, state.image_shape[1] - 1)
        image_y = _clamp(round(y / scale), 0, state.image_shape[0] - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            shift_pressed = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
            slot_index = _find_slot_index(state.slots, image_x, image_y)
            if shift_pressed:
                state.selected_index = -1
                state.drag_mode = "create"
                state.drag_anchor = (image_x, image_y)
                state.draft_box = (image_x, image_y, image_x, image_y)
                return

            if slot_index >= 0:
                state.selected_index = slot_index
                hit = _corner_hit(state.slots[slot_index], image_x, image_y) or "move"
                state.drag_mode = hit
                state.drag_anchor = (image_x, image_y)
                state.drag_origin_box = state.slots[slot_index].box
                return

            state.selected_index = -1

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.drag_mode == "create" and state.drag_anchor is not None:
                anchor_x, anchor_y = state.drag_anchor
                state.draft_box = (anchor_x, anchor_y, image_x, image_y)
                return

            if (
                state.drag_mode is not None
                and state.drag_mode != "create"
                and state.drag_anchor is not None
                and state.drag_origin_box is not None
                and 0 <= state.selected_index < len(state.slots)
            ):
                state.slots[state.selected_index] = _update_slot_from_drag(
                    EditableSlot(
                        id=state.slots[state.selected_index].id,
                        box=state.drag_origin_box,
                        angle_degrees=state.slots[state.selected_index].angle_degrees,
                        included=state.slots[state.selected_index].included,
                    ),
                    state.drag_mode,
                    state.drag_anchor[0],
                    state.drag_anchor[1],
                    image_x,
                    image_y,
                    state.image_shape[1],
                    state.image_shape[0],
                )
                state.dirty = True
                return

        elif event == cv2.EVENT_LBUTTONUP:
            if state.drag_mode == "create" and state.draft_box is not None:
                box = _normalize_pixel_box(state.draft_box, state.image_shape[1], state.image_shape[0])
                if box[2] - box[0] >= 8 and box[3] - box[1] >= 8:
                    state.slots.append(
                        EditableSlot(
                            id=_next_slot_id(state.slots),
                            box=box,
                            angle_degrees=0.0,
                            included=True,
                        )
                    )
                    state.selected_index = len(state.slots) - 1
                    state.dirty = True
                state.draft_box = None

            state.drag_mode = None
            state.drag_anchor = None
            state.drag_origin_box = None

    cv2.setMouseCallback("Parking Slot Editor", handle)


def main() -> None:
    args = parse_args()
    project_root = _resolve_project_root(args.project_root)
    settings, _ = _load_camera_configs(project_root)
    camera_id = _choose_camera_interactively(project_root, args.camera_id)
    camera, camera_slots = _load_camera(project_root, camera_id)
    image_paths = _collect_image_paths(project_root, camera_id, args.frame, args.frames_dir)
    image = _load_image(image_paths[0])
    image_height, image_width = image.shape[:2]
    scale = _fit_scale(image_width, image_height, args.max_width, args.max_height)
    output_dir = (project_root / args.output_dir).resolve()
    initial_slots = _load_editor_slots(
        [
            _prod_slots_path(settings.camera_config_path, camera_id),
            _editor_state_path(output_dir, camera_id),
        ],
        image_width,
        image_height,
    )
    if initial_slots is None:
        initial_slots = [_normalized_to_pixels(slot, image_width, image_height) for slot in camera_slots]

    state = EditorState(
        image_paths=image_paths,
        image_index=0,
        image_shape=(image_height, image_width),
        slots=initial_slots,
        selected_index=0 if initial_slots else -1,
    )

    cv2 = _require_cv2()
    cv2.namedWindow("Parking Slot Editor", cv2.WINDOW_NORMAL)
    _install_mouse_handler(state, scale)

    print(
        f"Loaded camera={camera.id} display_name={camera.display_name or camera.id} "
        f"frames={len(image_paths)} slots={len(state.slots)} output_dir={output_dir}"
    )

    left_arrow_keys, right_arrow_keys = _arrow_key_codes()

    while True:
        current_path = state.image_paths[state.image_index]
        image = _load_image(current_path)
        if image.shape[:2] != state.image_shape:
            raise RuntimeError(
                f"Frame shape changed from {state.image_shape} to {image.shape[:2]} for {current_path}. "
                "Use frames from the same camera/resolution."
            )
        preview = _render(state, image, camera.id, scale, args.max_width, args.max_height)
        cv2.imshow("Parking Slot Editor", preview)
        try:
            if cv2.getWindowProperty("Parking Slot Editor", cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        key = cv2.waitKeyEx(30)
        if key < 0:
            continue
        key_char = _key_char(key)

        if key == 27 or key_char == "q":
            break
        if key_char == "h":
            state.show_help = not state.show_help
            continue
        if key in right_arrow_keys:
            state.image_index = (state.image_index + 1) % len(state.image_paths)
            continue
        if key in left_arrow_keys:
            state.image_index = (state.image_index - 1) % len(state.image_paths)
            continue
        if (key in (9, ord("\t")) or key_char == "n") and state.slots:
            state.selected_index = (state.selected_index + 1) % len(state.slots)
            continue
        if key_char == "p" and state.slots:
            state.selected_index = (state.selected_index - 1) % len(state.slots)
            continue
        if key_char == "c" and 0 <= state.selected_index < len(state.slots):
            duplicated = _duplicate_slot(
                state.slots[state.selected_index],
                state.slots,
                state.image_shape[1],
                state.image_shape[0],
            )
            state.slots.append(duplicated)
            state.selected_index = len(state.slots) - 1
            state.dirty = True
            continue
        if key_char == "x" and 0 <= state.selected_index < len(state.slots):
            state.slots[state.selected_index].included = not state.slots[state.selected_index].included
            state.dirty = True
            continue
        if key_char in {",", "<"} and 0 <= state.selected_index < len(state.slots):
            step = 5.0 if key_char == "<" else 1.0
            state.slots[state.selected_index].angle_degrees -= step
            state.dirty = True
            continue
        if key_char in {".", ">"} and 0 <= state.selected_index < len(state.slots):
            step = 5.0 if key_char == ">" else 1.0
            state.slots[state.selected_index].angle_degrees += step
            state.dirty = True
            continue
        if key_char == "d" and 0 <= state.selected_index < len(state.slots):
            del state.slots[state.selected_index]
            if not state.slots:
                state.selected_index = -1
            else:
                state.selected_index = min(state.selected_index, len(state.slots) - 1)
            state.dirty = True
            continue
        if key_char == "s":
            prod_json_path, prod_yaml_path, preview_path = _save_outputs(
                state=state,
                image=image,
                camera_id=camera.id,
                output_dir=output_dir,
                camera_config_path=settings.camera_config_path,
            )
            print(f"Saved prod JSON to {prod_json_path}")
            print(f"Updated prod YAML at {prod_yaml_path}")
            print(f"Saved preview to {preview_path}")
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
