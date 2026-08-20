"""Software KV metadata manager latency and throughput model."""

from __future__ import annotations

import math

from model.queueing import compute_queueing, queue_delay_us
from model.types import LatencyBreakdown, QueueingMetrics, ThroughputMetrics


def scaling_factor_n(
    n: int,
    base_overhead: float = 0.0,
    log_factor: float = 0.1,
) -> float:
    """Scale latency with N using log(N) contention model."""
    if n <= 1:
        return 1.0
    return 1.0 + base_overhead + log_factor * math.log2(n)


def compute_software_latency(
    lookup_us: float,
    metadata_update_us: float,
    score_us: float,
    selection_us: float,
    coordination_us: float,
    memory_access_us: float,
    cache_hit_rate: float,
    cpu_contention: float,
    metadata_entries: int,
    candidate_k: int,
    parallelism: int,
    metadata_ops: int,
    arrival_rate: float,
    service_rate: float,
) -> tuple[LatencyBreakdown, QueueingMetrics]:
    """
    T_sw = T_lookup + T_metadata_update + T_score + T_selection
           + T_coordination + T_queue

    Latency scales with metadata_entries, candidate_k, and contention.
    Parallelism reduces effective per-request latency when concurrency > 1.
    """
    n_scale = scaling_factor_n(metadata_entries, base_overhead=0.05, log_factor=0.08)
    k_scale = scaling_factor_n(candidate_k, base_overhead=0.0, log_factor=0.15)
    contention_scale = 1.0 + cpu_contention

    # Cache hit reduces memory penalty
    mem_penalty = memory_access_us * (1.0 - cache_hit_rate) * metadata_ops

    t_lookup = lookup_us * n_scale * contention_scale
    t_update = metadata_update_us * metadata_ops * contention_scale * 0.5
    t_score = score_us * k_scale * candidate_k * 0.1 * contention_scale
    t_selection = selection_us * k_scale * contention_scale
    t_coord = coordination_us * contention_scale
    t_mem = mem_penalty

    raw_total = t_lookup + t_update + t_score + t_selection + t_coord + t_mem

    # Parallelism: effective latency reduction (not linear)
    par_factor = max(1.0, parallelism)
    effective_service = raw_total / (1.0 + 0.5 * (par_factor - 1))

    queueing = compute_queueing(arrival_rate, service_rate, effective_service)
    t_queue = queue_delay_us(queueing, effective_service)

    breakdown = LatencyBreakdown(
        lookup=t_lookup / par_factor,
        metadata_update=(t_update + t_mem) / par_factor,
        score=t_score / par_factor,
        candidate_selection=t_selection / par_factor,
        coordination=t_coord / par_factor,
        queue=t_queue,
    )

    return breakdown, queueing


def compute_software_throughput(
    cpu_cores: int,
    parallelism: int,
    per_op_latency_us: float,
    metadata_ops_per_request: int,
    requests_per_sec: float,
) -> ThroughputMetrics:
    """
    Software throughput limited by CPU cycles and parallelism.
    """
    if per_op_latency_us <= 0:
        per_op_latency_us = 0.001

    # Effective parallel ops
    effective_cores = min(cpu_cores, parallelism)
    ops_per_sec_per_core = 1e6 / per_op_latency_us
    theoretical_ops = ops_per_sec_per_core * effective_cores

    metadata_demand = requests_per_sec * metadata_ops_per_request
    sustainable_ops = theoretical_ops
    sustainable_rps = sustainable_ops / max(metadata_ops_per_request, 1)

    utilization = min(metadata_demand / max(sustainable_ops, 1), 1.0)
    achieved_rps = min(requests_per_sec, sustainable_rps)

    return ThroughputMetrics(
        ops_per_sec=min(metadata_demand, sustainable_ops),
        theoretical_ops_per_sec=theoretical_ops,
        sustainable_ops_per_sec=sustainable_ops,
        sustainable_rps=sustainable_rps,
        achieved_rps=achieved_rps,
        utilization=utilization,
        cycles_per_operation=per_op_latency_us * 1000,  # rough CPU cycles at 1GHz equiv
        operations_per_cycle=1.0 / max(per_op_latency_us * 1000, 1),
    )
