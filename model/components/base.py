"""Shared helpers for component configuration dataclasses."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, TypeVar

T = TypeVar("T")


def config_to_dict(config: Any) -> dict[str, Any]:
    return asdict(config)


def merge_dict(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def build_config(cls: type[T], data: dict[str, Any] | None, defaults: dict[str, Any]) -> T:
    merged = merge_dict(defaults, data)
    allowed = {field.name for field in fields(cls)}
    kwargs = {key: merged[key] for key in allowed if key in merged}
    return cls(**kwargs)
