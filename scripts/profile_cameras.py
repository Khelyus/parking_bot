from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parking_bot.camera_catalog import UfanetCatalogClient, resolve_cameras
from parking_bot.config_loader import load_camera_configs
from parking_bot.detector import ParkingSpaceDetector
from parking_bot.repository import SQLiteRepository
from parking_bot.service import CameraMonitorService
from parking_bot.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile all configured cameras across several rounds and rank the best ones."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the offline demo camera config instead of the live Ufanet cameras.",
    )
    parser.add_argument("--rounds", type=int, default=3, help="How many polling rounds to run.")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=5.0,
        help="Pause between rounds.",
    )
    parser.add_argument(
        "--output",
        default="runtime/camera_profile.json",
        help="Where to save the JSON profile.",
    )
    parser.add_argument(
        "--min-known-ratio",
        type=float,
        default=0.6,
        help="Minimum share of non-unknown states for a camera to be considered recommended.",
    )
    return parser.parse_args()


async def build_profile(
    *,
    use_demo: bool,
    rounds: int,
    delay_seconds: float,
    output_path: Path,
    min_known_ratio: float,
) -> None:
    settings = load_settings(PROJECT_ROOT, require_telegram_token=False)
    if use_demo:
        settings.camera_config_path = settings.project_dir / "config/cameras.demo.yaml"
    catalog_client = UfanetCatalogClient(settings.catalog_url, settings.user_agent)
    camera_configs = load_camera_configs(settings.camera_config_path, settings)
    cameras = [camera for camera in resolve_cameras(camera_configs, catalog_client) if camera.enabled]
    repository = SQLiteRepository(settings.database_path)
    detector = ParkingSpaceDetector(settings.model_path, image_size=max(640, settings.image_width))
    service = CameraMonitorService(
        settings=settings,
        repository=repository,
        detector=detector,
        catalog_client=catalog_client,
        cameras=cameras,
    )

    series: dict[str, list[dict[str, object]]] = {camera.id: [] for camera in cameras}
    try:
        for round_index in range(1, rounds + 1):
            print(f"Round {round_index}/{rounds}")
            for camera in cameras:
                status = await service.refresh_camera(camera.id, notify=False)
                row = {
                    "observed_at": status.observed_at.isoformat() if status.observed_at else None,
                    "current_availability": status.current_availability.value,
                    "stable_availability": status.stable_availability.value,
                    "free_count": status.free_count,
                    "occupied_count": status.occupied_count,
                    "last_error": status.last_error,
                }
                series[camera.id].append(row)
                print(
                    f"  {camera.id}: {status.current_availability.value}"
                    f" | free={status.free_count}"
                    f" | occupied={status.occupied_count}"
                )
            if round_index < rounds:
                await asyncio.sleep(delay_seconds)
    finally:
        await service.stop()

    profile_rows: list[dict[str, object]] = []
    for camera in cameras:
        rows = series[camera.id]
        states = Counter(row["current_availability"] for row in rows)
        known_ratio = (states.get("free", 0) + states.get("full", 0)) / max(1, len(rows))
        free_counts = [int(row["free_count"]) for row in rows]
        occupied_counts = [int(row["occupied_count"]) for row in rows]
        profile_rows.append(
            {
                "camera_id": camera.id,
                "display_name": camera.display_name,
                "map_url": camera.map_url,
                "rounds": len(rows),
                "states": dict(states),
                "known_ratio": round(known_ratio, 3),
                "average_free_count": round(statistics.fmean(free_counts), 3),
                "average_occupied_count": round(statistics.fmean(occupied_counts), 3),
                "max_free_count": max(free_counts, default=0),
                "max_occupied_count": max(occupied_counts, default=0),
                "recommended": known_ratio >= min_known_ratio,
                "samples": rows,
            }
        )

    profile_rows.sort(
        key=lambda row: (
            row["known_ratio"],
            row["max_free_count"] + row["max_occupied_count"],
            row["average_free_count"] + row["average_occupied_count"],
        ),
        reverse=True,
    )

    recommended = [row["camera_id"] for row in profile_rows if row["recommended"]]
    payload = {
        "rounds": rounds,
        "delay_seconds": delay_seconds,
        "min_known_ratio": min_known_ratio,
        "recommended_cameras": recommended,
        "profile": profile_rows,
    }

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Saved profile to: {output_path}")
    print("Recommended cameras:")
    if not recommended:
        print("  none")
    for camera_id in recommended:
        row = next(item for item in profile_rows if item["camera_id"] == camera_id)
        print(
            f"  {camera_id}"
            f" | known_ratio={row['known_ratio']}"
            f" | states={row['states']}"
        )


def main() -> None:
    args = parse_args()
    asyncio.run(
        build_profile(
            use_demo=args.demo,
            rounds=max(1, args.rounds),
            delay_seconds=max(0.0, args.delay_seconds),
            output_path=Path(args.output),
            min_known_ratio=max(0.0, min(1.0, args.min_known_ratio)),
        )
    )


if __name__ == "__main__":
    main()
