"""RTL hardware KV metadata engine latency and throughput model."""

from __future__ import annotations

import math

from model.hotness import hotness_update_cycles
from model.queueing import compute_queueing, queue_delay_us
from model.types import LatencyBreakdown, QueueingMetrics, ThroughputMetrics


def rtl_cycles_per_operation(
    pipeline_depth: int,
    metadata_entries: int,
    candidate_k: int,
    hotness_complexity: float,
    op_type: str = "lookup",
) -> float:
    """
    Estimate cycles per metadata operation based on RTL parameters.

    T_hw(N) = base_hw_latency + f(N, concurrency, metadata_size)
    """
    n_scale = 1.0 + 0.02 * math.log2(max(metadata_entries, 2))
    k_scale = 1.0 + 0.1 * math.log2(max(candidate_k, 2))

    base = {
        "lookup": 2,
        "update": 3,
        "score": 4,
        "selection": 2,
    }.get(op_type, 2)

    if op_type == "score":
        return (base + hotness_update_cycles(hotness_complexity, pipeline_depth)) * k_scale

    return base * n_scale + pipeline_depth * 0.5


def cycles_to_us(cycles: float, clock_freq_mhz: float) -> float:
    """Convert RTL cycles to microseconds."""
    if clock_freq_mhz <= 0:
        return 0.0
    return cycles / clock_freq_mhz


def compute_hardware_latency(
    clock_freq_mhz: float,
    pipeline_depth: int,
    lookup_ports: int,
    parallel_ops: int,
    candidate_k: int,
    scoring_units: int,
    metadata_entries: int,
    hotness_complexity: float,
    metadata_ops: int,
    request_arrival_us: float,
    hw_to_sw_us: float,
    coordination_us: float,
    arrival_rate: float,
    service_rate: float,
) -> tuple[LatencyBreakdown, QueueingMetrics, float]:
    """
    T_hw = T_request_arrival + T_hw_lookup + T_metadata_update + T_score
           + T_selection + T_hw_to_sw + T_coordination + T_queue
    """
    lookup_cycles = rtl_cycles_per_operation(
        pipeline_depth, metadata_entries, candidate_k, hotness_complexity, "lookup"
    )
    update_cycles = rtl_cycles_per_operation(
        pipeline_depth, metadata_entries, candidate_k, hotness_complexity, "update"
    )
    score_cycles = rtl_cycles_per_operation(
        pipeline_depth, metadata_entries, candidate_k, hotness_complexity, "score"
    )
    selection_cycles = rtl_cycles_per_operation(
        pipeline_depth, metadata_entries, candidate_k, hotness_complexity, "selection"
    )

    # Parallelism from ports and units reduces effective latency
    lookup_parallelism = max(lookup_ports, 1)
    op_parallelism = max(parallel_ops, 1)
    score_parallelism = max(scoring_units, 1)

    t_lookup = cycles_to_us(lookup_cycles / lookup_parallelism, clock_freq_mhz)
    t_update = cycles_to_us(update_cycles * metadata_ops * 0.5 / op_parallelism, clock_freq_mhz)
    t_score = cycles_to_us(
        score_cycles * candidate_k / score_parallelism, clock_freq_mhz
    )
    t_selection = cycles_to_us(selection_cycles / max(candidate_k // 2, 1), clock_freq_mhz)

    raw_service = t_lookup + t_update + t_score + t_selection + request_arrival_us + hw_to_sw_us + coordination_us

    queueing = compute_queueing(arrival_rate, service_rate, raw_service)
    t_queue = queue_delay_us(queueing, raw_service)

    breakdown = LatencyBreakdown(
        request_arrival=request_arrival_us,
        lookup=t_lookup,
        metadata_update=t_update,
        score=t_score,
        candidate_selection=t_selection,
        hw_to_sw=hw_to_sw_us,
        coordination=coordination_us,
        queue=t_queue,
    )

    cycles_per_op = (lookup_cycles + update_cycles + score_cycles + selection_cycles) / 4
    return breakdown, queueing, cycles_per_op


def compute_hardware_throughput(
    clock_freq_mhz: float,
    pipeline_depth: int,
    lookup_ports: int,
    parallel_ops: int,
    scoring_units: int,
    candidate_comparison_units: int,
    metadata_memory_bandwidth_gbps: float,
    metadata_bytes_per_block: int,
    metadata_ops_per_request: int,
    requests_per_sec: float,
    cycles_per_operation: float,
    queue_depth: int,
) -> ThroughputMetrics:
    """
    Hardware throughput from RTL parallelism and memory bandwidth.
    """
    clock_hz = clock_freq_mhz * 1e6
    parallel_engines = max(lookup_ports, parallel_ops, scoring_units, candidate_comparison_units)

    if cycles_per_operation <= 0:
        cycles_per_operation = 1

    ops_per_cycle = parallel_engines / cycles_per_operation
    theoretical_ops = (clock_hz / cycles_per_operation) * parallel_engines

    # Memory bandwidth limit
    bytes_per_op = metadata_bytes_per_block * 0.25  # ASSUMPTION: partial read per op
    bandwidth_bytes_per_sec = metadata_memory_bandwidth_gbps * 1e9
    bandwidth_limited_ops = bandwidth_bytes_per_sec / max(bytes_per_op, 1)

    sustainable_ops = min(theoretical_ops, bandwidth_limited_ops)

    # Queue depth limits burst handling
    burst_capacity_factor = 1.0 + math.log2(max(queue_depth, 2)) * 0.05
    sustainable_ops *= burst_capacity_factor

    metadata_demand = requests_per_sec * metadata_ops_per_request
    utilization = min(metadata_demand / max(sustainable_ops, 1), 1.0)
    sustainable_rps = sustainable_ops / max(metadata_ops_per_request, 1)
    achieved_rps = min(requests_per_sec, sustainable_rps)

    return ThroughputMetrics(
        ops_per_sec=min(metadata_demand, sustainable_ops),
        theoretical_ops_per_sec=theoretical_ops,
        sustainable_ops_per_sec=sustainable_ops,
        sustainable_rps=sustainable_rps,
        achieved_rps=achieved_rps,
        utilization=utilization,
        cycles_per_operation=cycles_per_operation,
        operations_per_cycle=ops_per_cycle,
    )
