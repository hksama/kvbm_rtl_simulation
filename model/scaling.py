"""Prefill GPU scaling model for checkpoints 2 and 3."""

from __future__ import annotations

import math

from model.checkpoint1 import Checkpoint1Config
from model.software import scaling_factor_n

PREFILL_GPU_OPTIONS_CP2 = [2, 4, 8, 16]
PREFILL_GPU_OPTIONS_CP3 = [32, 64, 128]


def apply_prefill_scaling(config: Checkpoint1Config) -> Checkpoint1Config:
    """
    Scale configuration for N prefill GPUs sharing one RTL engine.

    As prefill GPUs increase:
    - effective metadata entries grow (distributed KV blocks)
    - coordination overhead increases
    - software CPU contention increases
    - metadata ops/request increase (more worker state to track)
    - candidate count is capped by available GPUs
    """
    n = config.prefill_gpus
    if n <= 1:
        return config

    gpu_scale = scaling_factor_n(n, base_overhead=0.1, log_factor=0.2)
    coord_scale = 1.0 + 0.25 * math.log2(n)

    scaled = config.to_dict()
    scaled["num_kv_blocks"] = int(config.num_kv_blocks * n)
    scaled["candidate_count_k"] = min(config.candidate_count_k, n)
    scaled["coordination_latency_us"] = config.coordination_latency_us * coord_scale
    scaled["cpu_contention"] = min(1.0, config.cpu_contention + 0.05 * (n - 1))
    scaled["metadata_ops_per_request"] = int(
        math.ceil(config.metadata_ops_per_request * (1.0 + 0.1 * (n - 1)))
    )
    scaled["sw_sync_overhead_us"] = config.sw_sync_overhead_us * gpu_scale

    return Checkpoint1Config(**scaled)


def metadata_load_factor(prefill_gpus: int) -> float:
    """Metadata demand multiplier from distributed prefill workers."""
    if prefill_gpus <= 1:
        return 1.0
    return 1.0 + 0.12 * (prefill_gpus - 1)
