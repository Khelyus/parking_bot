from pathlib import Path

from parking_bot.config_loader import load_camera_configs
from parking_bot.settings import load_settings


def test_load_camera_configs_supports_detection_tuning_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "cameras.yaml"
    config_path.write_text(
        "\n".join(
            [
                "cameras:",
                "  - id: live_cam",
                "    map_url: http://maps.ufanet.ru/ufa#001-999-111",
                "    source_transport: hls",
                "    detection_image_size: 1280",
                "    detection_roi: 0.55,0.22,0.92,0.50",
                "    detection_exclude_rois:",
                "      - 0.12,0.10,0.30,0.24",
                "      - 0.70,0.00,1.00,0.18",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(tmp_path, require_telegram_token=False)

    configs = load_camera_configs(config_path, settings)

    assert configs[0].detection_image_size == 1280
    assert configs[0].detection_roi == (0.55, 0.22, 0.92, 0.5)
    assert configs[0].detection_exclude_rois == (
        (0.12, 0.1, 0.3, 0.24),
        (0.7, 0.0, 1.0, 0.18),
    )


def test_load_camera_configs_supports_parking_slots(tmp_path: Path) -> None:
    config_path = tmp_path / "cameras.yaml"
    config_path.write_text(
        "\n".join(
            [
                "cameras:",
                "  - id: live_cam",
                "    map_url: http://maps.ufanet.ru/ufa#001-999-111",
                "    source_transport: hls",
                "    parking_slots:",
                "      - id: upper_1",
                "        box: 0.10,0.20,0.30,0.40",
                "        angle: 17.5",
                "      - box: 0.40,0.20,0.50,0.45",
            ]
        ),
        encoding="utf-8",
    )
    settings = load_settings(tmp_path, require_telegram_token=False)

    configs = load_camera_configs(config_path, settings)

    assert len(configs[0].parking_slots) == 2
    assert configs[0].parking_slots[0].id == "upper_1"
    assert configs[0].parking_slots[0].box == (0.1, 0.2, 0.3, 0.4)
    assert configs[0].parking_slots[0].angle_degrees == 17.5
    assert configs[0].parking_slots[1].id == "slot_2"
    assert configs[0].parking_slots[1].box == (0.4, 0.2, 0.5, 0.45)
    assert configs[0].parking_slots[1].angle_degrees == 0.0
