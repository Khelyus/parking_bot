from __future__ import annotations

import json
from pathlib import Path

from parking_bot.types import Availability, DemoObservationStep


def _resolve_optional_path(base_dir: Path, raw_value: object) -> Path | None:
    if raw_value in {None, ""}:
        return None
    candidate = Path(str(raw_value))
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def load_demo_observations(path: Path) -> tuple[DemoObservationStep, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("observations", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"Demo observations file must contain a non-empty list: {path}")

    base_dir = path.parent
    steps: list[DemoObservationStep] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(f"Observation #{index} in {path} must be an object")

        raw_frame_path = _resolve_optional_path(base_dir, item.get("raw_frame_path"))
        annotated_frame_path = _resolve_optional_path(base_dir, item.get("annotated_frame_path"))
        steps.append(
            DemoObservationStep(
                availability=Availability(str(item["availability"])),
                free_count=int(item.get("free_count", 0)),
                occupied_count=int(item.get("occupied_count", 0)),
                raw_frame_path=raw_frame_path,
                annotated_frame_path=annotated_frame_path,
            )
        )

    return tuple(steps)
