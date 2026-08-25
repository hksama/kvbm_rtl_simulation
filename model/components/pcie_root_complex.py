from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict


@dataclass
class PCIeRootComplexConfig:
    sku: str = "cpu_integrated_rc"
    pcie_generation: str = "Gen5"
    total_lanes: int = 128
    max_link_width: str = "x16"
    port_count: int = 8
    default_mps: int = 256
    default_mrrs: int = 512
    hop_latency_us: float = 0.2
    supports_acs: bool = True
    supports_ats: bool = True


DEFAULT_PCIE_ROOT_COMPLEX = config_to_dict(PCIeRootComplexConfig())


def pcie_root_complex_config_from_dict(data: dict[str, Any] | None) -> PCIeRootComplexConfig:
    return build_config(PCIeRootComplexConfig, data, DEFAULT_PCIE_ROOT_COMPLEX)
