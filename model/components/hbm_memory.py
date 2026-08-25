from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict


@dataclass
class HBMMemoryConfig:
    sku: str = "generic_hbm"
    capacity_gb: float = 80.0
    bandwidth_GBps: float = 3350.0
    stacks: int = 5
    channels: int = 5
    read_latency_ns: float = 100.0
    write_latency_ns: float = 100.0
    generation: str = "HBM3"
    power_w: float = 60.0


DEFAULT_HBM_MEMORY = config_to_dict(HBMMemoryConfig())


def hbm_memory_config_from_dict(data: dict[str, Any] | None) -> HBMMemoryConfig:
    return build_config(HBMMemoryConfig, data, DEFAULT_HBM_MEMORY)
