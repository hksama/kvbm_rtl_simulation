from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict


@dataclass
class PrefillPoolConfig:
    worker_count: int = 4
    gpu_preset: str = "h100_sxm"
    workload_rate_rps: float = 1000.0
    metadata_ops_rate: float = 16000.0
    kv_blocks_owned: int = 65536
    attached_hbm_node_id: str = ""


DEFAULT_PREFILL_POOL = config_to_dict(PrefillPoolConfig())


def prefill_pool_config_from_dict(data: dict[str, Any] | None) -> PrefillPoolConfig:
    return build_config(PrefillPoolConfig, data, DEFAULT_PREFILL_POOL)
