from __future__ import annotations

from datetime import timedelta, timezone, tzinfo
import re

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]

    class ZoneInfoNotFoundError(Exception):
        pass


_KNOWN_FALLBACKS = {
    "UTC": timezone.utc,
    "Europe/Moscow": timezone(timedelta(hours=3), name="Europe/Moscow"),
}


def _parse_utc_offset(value: str) -> tzinfo | None:
    match = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", value.strip())
    if not match:
        return None
    sign, hours, minutes = match.groups()
    offset = timedelta(hours=int(hours), minutes=int(minutes))
    if sign == "-":
        offset = -offset
    return timezone(offset, name=value)


def resolve_timezone(value: str) -> tzinfo:
    if ZoneInfo is not None:
        try:
            return ZoneInfo(value)
        except ZoneInfoNotFoundError:
            pass

    fixed_offset = _parse_utc_offset(value)
    if fixed_offset is not None:
        return fixed_offset
    return _KNOWN_FALLBACKS.get(value, timezone.utc)
