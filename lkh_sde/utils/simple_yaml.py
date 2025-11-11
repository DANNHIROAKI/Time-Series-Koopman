"""Minimal YAML-like parser for configuration files."""
from __future__ import annotations

from typing import Any, Dict, List


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if token.startswith("0") and token != "0" and not token.startswith("0."):
            raise ValueError
        return int(token)
    except ValueError:
        try:
            return float(token)
        except ValueError:
            return token


def _parse_list(value: str) -> List[Any]:
    inner = value[1:-1].strip()
    if not inner:
        return []
    parts = [part.strip() for part in inner.split(",")]
    return [_parse_scalar(part) for part in parts]


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        key, sep, value = line.partition(":")
        if sep == "":
            raise ValueError(f"Invalid line: {raw_line}")
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            new_dict: Dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent, new_dict))
            continue
        if value.startswith("[") and value.endswith("]"):
            parent[key] = _parse_list(value)
        else:
            parent[key] = _parse_scalar(value)
    return root
