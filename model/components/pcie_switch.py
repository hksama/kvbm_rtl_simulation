from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict


@dataclass
class PCIeSwitchConfig:
    sku: str = "generic_pcie_switch"
    pcie_generation: str = "Gen5"
    port_count: int = 48
    total_lanes: int = 48
    internal_bandwidth_GBps: float = 128.0
    hop_latency_us: float = 0.15
    non_blocking: bool = True
    default_mps: int = 256


DEFAULT_PCIE_SWITCH = config_to_dict(PCIeSwitchConfig())


def pcie_switch_config_from_dict(data: dict[str, Any] | None) -> PCIeSwitchConfig:
    return build_config(PCIeSwitchConfig, data, DEFAULT_PCIE_SWITCH)
