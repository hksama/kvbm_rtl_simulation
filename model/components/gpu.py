from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict


@dataclass
class GPUConfig:
    sku: str = "generic_gpu"
    vendor: str = "NVIDIA"
    hbm_capacity_gb: float = 80.0
    hbm_bandwidth_GBps: float = 3350.0
    pcie_generation: str = "Gen5"
    pcie_width: str = "x16"
    nvlink_version: str = "NVLink4"
    nvlink_links: int = 18
    nvlink_bandwidth_GBps: float = 900.0
    sm_count: int = 132
    fp16_tflops: float = 990.0
    tdp_w: float = 700.0


DEFAULT_GPU = config_to_dict(GPUConfig())


def gpu_config_from_dict(data: dict[str, Any] | None) -> GPUConfig:
    return build_config(GPUConfig, data, DEFAULT_GPU)
