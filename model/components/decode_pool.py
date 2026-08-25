from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict


@dataclass
class DecodePoolConfig:
    worker_count: int = 4
    gpu_preset: str = "h100_sxm"
    workload_rate_rps: float = 2000.0
    metadata_ops_rate: float = 8000.0
    kv_blocks_owned: int = 65536
    attached_hbm_node_id: str = ""


DEFAULT_DECODE_POOL = config_to_dict(DecodePoolConfig())


def decode_pool_config_from_dict(data: dict[str, Any] | None) -> DecodePoolConfig:
    return build_config(DecodePoolConfig, data, DEFAULT_DECODE_POOL)
