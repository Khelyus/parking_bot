from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from parking_bot.demo_loader import load_demo_observations
from parking_bot.types import CameraConfig, ResolvedCamera


MAP_HASH_RE = re.compile(r"#(?P<number>[^/?#]+)$")
CATALOG_RE = re.compile(
    r"marker\.name = '(?P<name>(?:\\'|[^'])*)';\s+"
    r"marker\.server = '(?P<server>(?:\\'|[^'])*)';\s+"
    r"marker\.number = '(?P<number>(?:\\'|[^'])*)';\s+"
    r"marker\.token = '(?P<token>(?:\\'|[^'])*)';",
    re.S,
)


@dataclass(slots=True)
class UfanetDescriptor:
    number: str
    name: str
    server: str
    token: str


def extract_camera_number_from_url(map_url: str) -> str:
    match = MAP_HASH_RE.search(map_url)
    if not match:
        raise ValueError(f"Could not find a camera hash in map url: {map_url}")
    return match.group("number")


def parse_catalog_html(html: str) -> dict[str, UfanetDescriptor]:
    cameras: dict[str, UfanetDescriptor] = {}
    for match in CATALOG_RE.finditer(html):
        number = match.group("number").replace("\\'", "'")
        cameras[number] = UfanetDescriptor(
            number=number,
            name=unescape(match.group("name").replace("\\'", "'")),
            server=match.group("server").replace("\\'", "'"),
            token=match.group("token").replace("\\'", "'"),
        )
    return cameras


class UfanetCatalogClient:
    def __init__(
        self,
        catalog_url: str,
        user_agent: str,
        cache_ttl_seconds: int = 1800,
        *,
        cache_path: str | Path | None = None,
        request_timeout_seconds: int = 30,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.catalog_url = catalog_url
        self.user_agent = user_agent
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_path = Path(cache_path) if cache_path is not None else Path.cwd() / "runtime" / "ufanet_catalog_cache.json"
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._loaded_at = 0.0
        self._cache: dict[str, UfanetDescriptor] = {}
        self._load_cache_from_disk()

    def get_catalog(self, force_refresh: bool = False) -> dict[str, UfanetDescriptor]:
        is_fresh = time.monotonic() - self._loaded_at < self.cache_ttl_seconds
        if self._cache and is_fresh and not force_refresh:
            return self._cache

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            request = Request(self.catalog_url, headers={"User-Agent": self.user_agent})
            try:
                with urlopen(request, timeout=self.request_timeout_seconds) as response:
                    charset = response.headers.get_content_charset() or "utf-8"
                    html = response.read().decode(charset, errors="ignore")
                catalog = parse_catalog_html(html)
                if not catalog:
                    raise RuntimeError("The Ufanet catalog response did not contain any cameras")
                self._cache = catalog
                self._loaded_at = time.monotonic()
                self._save_cache_to_disk(catalog)
                return self._cache
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_seconds)

        if self._cache:
            self._loaded_at = time.monotonic()
            return self._cache

        raise RuntimeError("Could not download the Ufanet camera catalog") from last_error

    def resolve(self, map_url: str, force_refresh: bool = False) -> UfanetDescriptor:
        number = extract_camera_number_from_url(map_url)
        catalog = self.get_catalog(force_refresh=force_refresh)
        if number not in catalog:
            raise KeyError(f"Camera {number} was not found in the Ufanet catalog")
        return catalog[number]

    def _load_cache_from_disk(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        raw_catalog = payload.get("cameras", payload)
        if not isinstance(raw_catalog, dict):
            return
        catalog: dict[str, UfanetDescriptor] = {}
        for number, item in raw_catalog.items():
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            server = item.get("server")
            token = item.get("token")
            if not all(isinstance(value, str) and value for value in (name, server, token)):
                continue
            catalog[str(number)] = UfanetDescriptor(
                number=str(number),
                name=name,
                server=server,
                token=token,
            )
        if catalog:
            self._cache = catalog

    def _save_cache_to_disk(self, catalog: dict[str, UfanetDescriptor]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cameras": {
                number: {
                    "name": descriptor.name,
                    "server": descriptor.server,
                    "token": descriptor.token,
                }
                for number, descriptor in catalog.items()
            }
        }
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_cameras(
    camera_configs: list[CameraConfig],
    catalog_client: UfanetCatalogClient,
) -> list[ResolvedCamera]:
    resolved: list[ResolvedCamera] = []
    for config in camera_configs:
        if config.source_kind == "demo":
            if config.demo_observations_path is None:
                raise RuntimeError(
                    f"Demo camera {config.id} must define demo_observations_file in the config"
                )
            resolved.append(
                ResolvedCamera(
                    id=config.id,
                    map_url=config.map_url,
                    number=config.id,
                    server="demo",
                    token="demo",
                    display_name=config.display_name or f"Demo camera {config.id}",
                    enabled=config.enabled,
                    source_kind=config.source_kind,
                    source_transport=config.source_transport,
                    min_confidence=config.min_confidence,
                    min_free_spaces=config.min_free_spaces,
                    detection_image_size=config.detection_image_size,
                    detection_roi=config.detection_roi,
                    detection_exclude_rois=config.detection_exclude_rois,
                    parking_slots=config.parking_slots,
                    stable_cycles=config.stable_cycles,
                    poll_interval_seconds=config.poll_interval_seconds,
                    demo_observations=load_demo_observations(config.demo_observations_path),
                    demo_loop=config.demo_loop,
                )
            )
            continue

        if not config.map_url:
            raise RuntimeError(f"Ufanet camera {config.id} must define map_url")
        descriptor = catalog_client.resolve(config.map_url)
        resolved.append(
            ResolvedCamera(
                id=config.id,
                map_url=config.map_url,
                number=descriptor.number,
                server=descriptor.server,
                token=descriptor.token,
                display_name=config.display_name or descriptor.name,
                enabled=config.enabled,
                source_kind=config.source_kind,
                source_transport=config.source_transport,
                min_confidence=config.min_confidence,
                min_free_spaces=config.min_free_spaces,
                detection_image_size=config.detection_image_size,
                detection_roi=config.detection_roi,
                detection_exclude_rois=config.detection_exclude_rois,
                parking_slots=config.parking_slots,
                stable_cycles=config.stable_cycles,
                poll_interval_seconds=config.poll_interval_seconds,
                demo_loop=config.demo_loop,
            )
        )
    return resolved
