import json
from email.message import Message

from parking_bot.camera_catalog import (
    UfanetCatalogClient,
    extract_camera_number_from_url,
    parse_catalog_html,
)


def test_extract_camera_number_from_url() -> None:
    value = extract_camera_number_from_url("http://maps.ufanet.ru/ufa#1549441947OXE898")
    assert value == "1549441947OXE898"


def test_parse_catalog_html() -> None:
    html = """
    marker.name = 'Test Camera';
    marker.server = '136.169.226.80';
    marker.number = '001-999-011';
    marker.token = 'abc123';
    """
    catalog = parse_catalog_html(html)
    assert catalog["001-999-011"].name == "Test Camera"
    assert catalog["001-999-011"].server == "136.169.226.80"
    assert catalog["001-999-011"].token == "abc123"


def test_catalog_client_writes_disk_cache_after_successful_fetch(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "ufanet_catalog_cache.json"
    html = """
    marker.name = 'Cached Camera';
    marker.server = '136.169.226.81';
    marker.number = '001-999-119';
    marker.token = 'xyz789';
    """

    class _FakeResponse:
        def __init__(self, payload: str) -> None:
            self._payload = payload.encode("utf-8")
            self.headers = Message()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self._payload

    monkeypatch.setattr(
        "parking_bot.camera_catalog.urlopen",
        lambda request, timeout: _FakeResponse(html),
    )

    client = UfanetCatalogClient(
        "https://example.test/catalog",
        "test-agent",
        cache_path=cache_path,
        max_retries=1,
    )

    catalog = client.get_catalog()

    assert catalog["001-999-119"].name == "Cached Camera"
    cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cached_payload["cameras"]["001-999-119"]["server"] == "136.169.226.81"


def test_catalog_client_uses_disk_cache_when_refresh_fails(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / "ufanet_catalog_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "cameras": {
                    "001-999-011": {
                        "name": "Offline Camera",
                        "server": "136.169.226.80",
                        "token": "cached-token",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    def _fail(*args, **kwargs):
        raise TimeoutError("catalog timeout")

    monkeypatch.setattr("parking_bot.camera_catalog.urlopen", _fail)

    client = UfanetCatalogClient(
        "https://example.test/catalog",
        "test-agent",
        cache_ttl_seconds=0,
        cache_path=cache_path,
        max_retries=1,
    )

    descriptor = client.resolve("http://maps.ufanet.ru/ufa#001-999-011")

    assert descriptor.name == "Offline Camera"
    assert descriptor.token == "cached-token"
