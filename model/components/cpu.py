from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict


@dataclass
class CPUConfig:
    sku: str = "generic_cpu"
    vendor: str = "AMD"
    cores: int = 64
    threads: int = 128
    base_freq_ghz: float = 2.4
    boost_freq_ghz: float = 3.7
    l3_cache_mb: float = 384.0
    memory_channels: int = 12
    max_dram_bandwidth_GBps: float = 460.0
    pcie_generation: str = "Gen5"
    pcie_lanes: int = 128
    numa_nodes: int = 2
    tdp_w: float = 360.0


DEFAULT_CPU = config_to_dict(CPUConfig())


def cpu_config_from_dict(data: dict[str, Any] | None) -> CPUConfig:
    return build_config(CPUConfig, data, DEFAULT_CPU)
