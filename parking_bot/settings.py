from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from parking_bot.env_loader import load_env_file


@dataclass(slots=True)
class Settings:
    project_dir: Path
    camera_config_path: Path
    model_path: Path
    slot_classifier_model_path: Path
    vehicle_model_path: str
    database_path: Path
    frames_dir: Path
    timezone: str
    telegram_bot_token: str
    telegram_proxy: str | None
    telegram_connect_timeout_seconds: float
    telegram_read_timeout_seconds: float
    telegram_write_timeout_seconds: float
    telegram_pool_timeout_seconds: float
    telegram_bootstrap_retries: int
    poll_interval_seconds: int
    monitor_workers: int
    image_width: int
    request_timeout_seconds: int
    hls_burst_frames: int
    hls_burst_pause_seconds: float
    default_confidence: float
    vehicle_confidence: float
    slot_classifier_confidence: float
    slot_classifier_image_size: int
    default_stable_cycles: int
    notification_stable_cycles: int
    notification_cooldown_seconds: int
    catalog_url: str
    user_agent: str


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def load_settings(
    project_dir: Path | None = None,
    *,
    require_telegram_token: bool = True,
) -> Settings:
    root = Path(project_dir or Path.cwd())
    load_env_file(root / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if require_telegram_token and not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Create .env from .env.example first.")

    default_stable_cycles = _env_int("DEFAULT_STABLE_CYCLES", 2)
    notification_stable_cycles = _env_int(
        "NOTIFICATION_STABLE_CYCLES",
        max(default_stable_cycles + 2, 4),
    )

    settings = Settings(
        project_dir=root,
        camera_config_path=root / os.getenv("CAMERA_CONFIG_PATH", "config/cameras.yaml"),
        model_path=root / os.getenv("MODEL_PATH", "models/pklot_yolov84_best.pt"),
        slot_classifier_model_path=root
        / os.getenv("SLOT_CLASSIFIER_MODEL_PATH", "models/pklot_slot_classifier.pt"),
        vehicle_model_path=os.getenv("VEHICLE_MODEL_PATH", "yolov8n.pt"),
        database_path=root / os.getenv("DATABASE_PATH", "runtime/parking_bot.db"),
        frames_dir=root / os.getenv("FRAMES_DIR", "runtime/frames"),
        timezone=os.getenv("TIMEZONE", "Europe/Moscow"),
        telegram_bot_token=token,
        telegram_proxy=os.getenv("TELEGRAM_PROXY", "").strip() or None,
        telegram_connect_timeout_seconds=_env_float("TELEGRAM_CONNECT_TIMEOUT_SECONDS", 20.0),
        telegram_read_timeout_seconds=_env_float("TELEGRAM_READ_TIMEOUT_SECONDS", 30.0),
        telegram_write_timeout_seconds=_env_float("TELEGRAM_WRITE_TIMEOUT_SECONDS", 30.0),
        telegram_pool_timeout_seconds=_env_float("TELEGRAM_POOL_TIMEOUT_SECONDS", 5.0),
        telegram_bootstrap_retries=_env_int("TELEGRAM_BOOTSTRAP_RETRIES", 3),
        poll_interval_seconds=_env_int("POLL_INTERVAL_SECONDS", 30),
        monitor_workers=_env_int("MONITOR_WORKERS", 2),
        image_width=_env_int("IMAGE_WIDTH", 400),
        request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 20),
        hls_burst_frames=_env_int("HLS_BURST_FRAMES", 5),
        hls_burst_pause_seconds=_env_float("HLS_BURST_PAUSE_SECONDS", 1.0),
        default_confidence=_env_float("DEFAULT_CONFIDENCE", 0.05),
        vehicle_confidence=_env_float("VEHICLE_CONFIDENCE", 0.05),
        slot_classifier_confidence=_env_float("SLOT_CLASSIFIER_CONFIDENCE", 0.58),
        slot_classifier_image_size=_env_int("SLOT_CLASSIFIER_IMAGE_SIZE", 224),
        default_stable_cycles=default_stable_cycles,
        notification_stable_cycles=notification_stable_cycles,
        notification_cooldown_seconds=_env_int("NOTIFICATION_COOLDOWN_SECONDS", 900),
        catalog_url=os.getenv("CATALOG_URL", "https://maps.ufanet.ru/ufa"),
        user_agent=os.getenv(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0 Safari/537.36",
        ),
    )

    ensure_runtime_layout(settings)
    ensure_default_model(settings)
    return settings


def ensure_runtime_layout(settings: Settings) -> None:
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    settings.frames_dir.mkdir(parents=True, exist_ok=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.slot_classifier_model_path.parent.mkdir(parents=True, exist_ok=True)


def ensure_default_model(settings: Settings) -> None:
    if settings.model_path.exists():
        return
    fallback = settings.project_dir / "pklot_yolov84" / "weights" / "best.pt"
    if fallback.exists():
        shutil.copy2(fallback, settings.model_path)
