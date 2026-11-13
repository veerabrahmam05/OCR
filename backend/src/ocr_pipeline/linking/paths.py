"""JSON paths like ``line_items[2].amount`` over extracted data."""

from __future__ import annotations

import re
from typing import Any

_SEGMENT = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
_MISSING = object()


def _is_free_form(data: dict) -> bool:
    fields = data.get("fields")
    return set(data) == {"fields"} and isinstance(fields, list) and all(
        isinstance(f, dict) and "name" in f for f in fields
    )


def resolve(data: dict, path: str) -> tuple[bool, Any]:
    """(found, value). Free-form results are addressed by entry name."""
    node: Any = data
    for key, index in _SEGMENT.findall(path):
        if key and isinstance(node, dict) and key in node:
            node = node[key]
        elif index and isinstance(node, list) and int(index) < len(node):
            node = node[int(index)]
        else:
            node = _MISSING
            break
    if node is not _MISSING:
        return True, node
    if _is_free_form(data):
        for f in data["fields"]:
            if f["name"] == path:
                return True, f.get("value")
    return False, None


def leaf_paths(data: dict) -> list[str]:
    """Paths of every non-null scalar value (free-form: names with a value)."""
    if _is_free_form(data):
        return [f["name"] for f in data["fields"] if f.get("value") is not None]
    out: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif node is not None:
            out.append(path)

    walk(data, "")
    return out


def covers(source_path: str, leaf: str) -> bool:
    """A source for ``line_items[0]`` also covers ``line_items[0].amount``."""
    return leaf == source_path or leaf.startswith(source_path + ".") or leaf.startswith(source_path + "[")
