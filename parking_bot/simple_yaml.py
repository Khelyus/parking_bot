from __future__ import annotations

from pathlib import Path


def _strip_comment(text: str) -> str:
    in_quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_quote:
            if char == in_quote:
                in_quote = None
            continue
        if char in {"'", '"'}:
            in_quote = char
            continue
        if char == "#":
            return text[:index]
    return text


def _parse_scalar(value: str) -> object:
    value = value.strip()
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None

    try:
        if any(char in value for char in {".", "e", "E"}):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _split_key_value(text: str) -> tuple[str, object]:
    if ":" not in text:
        raise ValueError(f"Expected a key/value line, got: {text!r}")
    key, value = text.split(":", 1)
    return key.strip(), _parse_scalar(value)


def load_simple_yaml(path: Path) -> dict[str, object]:
    root: dict[str, object] = {}
    current_list_name: str | None = None
    current_item: dict[str, object] | None = None

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        without_comment = _strip_comment(raw_line)
        if not without_comment.strip():
            continue

        indent = len(without_comment) - len(without_comment.lstrip(" "))
        stripped = without_comment.strip()

        if indent == 0:
            current_item = None
            if stripped.endswith(":") and ":" not in stripped[:-1]:
                key = stripped[:-1].strip()
                root[key] = []
                current_list_name = key
                continue

            key, value = _split_key_value(stripped)
            root[key] = value
            current_list_name = None
            continue

        if indent == 2 and stripped.startswith("- "):
            if current_list_name is None or not isinstance(root.get(current_list_name), list):
                raise ValueError(f"Unexpected list item on line {line_number}")
            current_item = {}
            root[current_list_name].append(current_item)
            item_text = stripped[2:].strip()
            if item_text:
                key, value = _split_key_value(item_text)
                current_item[key] = value
            continue

        if indent >= 4 and current_item is not None:
            key, value = _split_key_value(stripped)
            current_item[key] = value
            continue

        raise ValueError(f"Unsupported YAML structure on line {line_number}: {raw_line!r}")

    return root
