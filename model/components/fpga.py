from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict


@dataclass
class FPGAConfig:
    sku: str = "generic_fpga"
    vendor: str = "AMD"
    family: str = "Alveo"
    logic_cells_k: float = 1300.0
    bram_mb: float = 54.0
    uram_mb: float = 0.0
    dsp_slices: int = 6840
    hbm_capacity_gb: float = 0.0
    hbm_bandwidth_GBps: float = 0.0
    pcie_generation: str = "Gen3"
    pcie_width: str = "x16"
    dma_engines: int = 2
    max_dma_queue_depth: int = 64
    clock_freq_mhz: float = 300.0
    pipeline_depth: int = 8
    tdp_w: float = 225.0


DEFAULT_FPGA = config_to_dict(FPGAConfig())


def fpga_config_from_dict(data: dict[str, Any] | None) -> FPGAConfig:
    return build_config(FPGAConfig, data, DEFAULT_FPGA)
