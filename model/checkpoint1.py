"""Checkpoint 1: Single prefill GPU, single RTL engine simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from model.hotness import (
    compute_hotness,
    generate_hotness_distribution,
    hotness_update_cycles,
)
from model.rtl import compute_hardware_latency, compute_hardware_throughput
from model.software import compute_software_latency, compute_software_throughput
from model.types import HotnessModelType, SimulationResult


@dataclass
class Checkpoint1Config:
    """All parameters for checkpoint 1 simulation."""

    # Workload
    requests_per_sec: float = 1000.0
    concurrent_requests: int = 64
    input_tokens_per_request: int = 512
    output_tokens_per_request: int = 128
    block_size_tokens: int = 16
    kv_blocks_per_request: int = 32
    prefix_reuse_fraction: float = 0.3
    prefix_reuse_pct: float = 0.5
    block_accesses_per_request: int = 8
    metadata_ops_per_request: int = 16
    burstiness: float = 1.0

    # KV metadata
    num_kv_blocks: int = 65536
    metadata_bytes_per_block: int = 64
    metadata_lookup_ops_per_request: int = 4
    metadata_update_ops_per_request: int = 8
    access_counter_width: int = 32
    timestamp_width: int = 32
    reference_counter_width: int = 16
    hotness_complexity: float = 1.0
    candidate_count_k: int = 8

    # Software
    sw_lookup_latency_us: float = 2.0
    sw_metadata_update_latency_us: float = 1.5
    sw_score_calc_latency_us: float = 3.0
    sw_candidate_selection_latency_us: float = 1.0
    cpu_cores: int = 8
    software_parallelism: int = 4
    memory_access_latency_us: float = 0.1
    cache_hit_rate: float = 0.85
    cpu_contention: float = 0.2
    sw_sync_overhead_us: float = 0.5

    # RTL
    clock_freq_mhz: float = 500.0
    pipeline_depth: int = 8
    lookup_ports: int = 2
    metadata_banks: int = 4
    parallel_metadata_ops: int = 4
    candidate_comparison_units: int = 8
    scoring_units: int = 4
    queue_depth: int = 256
    metadata_storage_capacity: int = 131072
    metadata_memory_bandwidth_gbps: float = 32.0
    hw_to_sw_overhead_us: float = 0.3
    request_arrival_overhead_us: float = 0.1

    # Hotness
    hotness_model_type: HotnessModelType = HotnessModelType.WEIGHTED_LINEAR
    hotness_alpha: float = 1.0
    hotness_beta: float = 0.5
    hotness_gamma: float = 0.3
    hotness_decay_lambda: float = 0.01

    # Topology (fixed for checkpoint 1)
    prefill_gpus: int = 1
    rtl_engines: int = 1

    # Coordination
    coordination_latency_us: float = 1.0

    @property
    def metadata_entries(self) -> int:
        return self.num_kv_blocks

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Checkpoint1Config":
        with open(path) as f:
            data = yaml.safe_load(f)

        w = data.get("workload", {})
        kv = data.get("kv_metadata", {})
        sw = data.get("software", {})
        rtl = data.get("rtl", {})
        hot = data.get("hotness", {})
        topo = data.get("topology", {})
        coord = data.get("coordination", {})

        hotness_type_map = {
            "weighted_linear": HotnessModelType.WEIGHTED_LINEAR,
            "lru_like": HotnessModelType.LRU_LIKE,
            "lfu_like": HotnessModelType.LFU_LIKE,
            "recency_weighted": HotnessModelType.RECENCY_WEIGHTED,
            "exp_decay": HotnessModelType.EXP_DECAY,
        }

        return cls(
            requests_per_sec=w.get("requests_per_sec", {}).get("value", 1000),
            concurrent_requests=w.get("concurrent_requests", {}).get("value", 64),
            input_tokens_per_request=w.get("input_tokens_per_request", {}).get("value", 512),
            output_tokens_per_request=w.get("output_tokens_per_request", {}).get("value", 128),
            block_size_tokens=w.get("block_size_tokens", {}).get("value", 16),
            kv_blocks_per_request=w.get("kv_blocks_per_request", {}).get("value", 32),
            prefix_reuse_fraction=w.get("prefix_reuse_fraction", {}).get("value", 0.3),
            prefix_reuse_pct=w.get("prefix_reuse_pct", {}).get("value", 0.5),
            block_accesses_per_request=w.get("block_accesses_per_request", {}).get("value", 8),
            metadata_ops_per_request=w.get("metadata_ops_per_request", {}).get("value", 16),
            burstiness=w.get("burstiness", {}).get("value", 1.0),
            num_kv_blocks=kv.get("num_kv_blocks", {}).get("value", 65536),
            metadata_bytes_per_block=kv.get("metadata_bytes_per_block", {}).get("value", 64),
            metadata_lookup_ops_per_request=kv.get("metadata_lookup_ops_per_request", {}).get("value", 4),
            metadata_update_ops_per_request=kv.get("metadata_update_ops_per_request", {}).get("value", 8),
            access_counter_width=kv.get("access_counter_width", {}).get("value", 32),
            timestamp_width=kv.get("timestamp_width", {}).get("value", 32),
            reference_counter_width=kv.get("reference_counter_width", {}).get("value", 16),
            hotness_complexity=kv.get("hotness_complexity", {}).get("value", 1.0),
            candidate_count_k=kv.get("candidate_count_k", {}).get("value", 8),
            sw_lookup_latency_us=sw.get("lookup_latency_us", {}).get("value", 2.0),
            sw_metadata_update_latency_us=sw.get("metadata_update_latency_us", {}).get("value", 1.5),
            sw_score_calc_latency_us=sw.get("score_calc_latency_us", {}).get("value", 3.0),
            sw_candidate_selection_latency_us=sw.get("candidate_selection_latency_us", {}).get("value", 1.0),
            cpu_cores=sw.get("cpu_cores", {}).get("value", 8),
            software_parallelism=sw.get("parallelism", {}).get("value", 4),
            memory_access_latency_us=sw.get("memory_access_latency_us", {}).get("value", 0.1),
            cache_hit_rate=sw.get("cache_hit_rate", {}).get("value", 0.85),
            cpu_contention=sw.get("cpu_contention", {}).get("value", 0.2),
            sw_sync_overhead_us=sw.get("sync_overhead_us", {}).get("value", 0.5),
            clock_freq_mhz=rtl.get("clock_freq_mhz", {}).get("value", 500),
            pipeline_depth=rtl.get("pipeline_depth", {}).get("value", 8),
            lookup_ports=rtl.get("lookup_ports", {}).get("value", 2),
            metadata_banks=rtl.get("metadata_banks", {}).get("value", 4),
            parallel_metadata_ops=rtl.get("parallel_metadata_ops", {}).get("value", 4),
            candidate_comparison_units=rtl.get("candidate_comparison_units", {}).get("value", 8),
            scoring_units=rtl.get("scoring_units", {}).get("value", 4),
            queue_depth=rtl.get("queue_depth", {}).get("value", 256),
            metadata_storage_capacity=rtl.get("metadata_storage_capacity", {}).get("value", 131072),
            metadata_memory_bandwidth_gbps=rtl.get("metadata_memory_bandwidth_gbps", {}).get("value", 32),
            hw_to_sw_overhead_us=rtl.get("hw_to_sw_overhead_us", {}).get("value", 0.3),
            request_arrival_overhead_us=rtl.get("request_arrival_overhead_us", {}).get("value", 0.1),
            hotness_model_type=hotness_type_map.get(
                hot.get("model_type", {}).get("value", "weighted_linear"),
                HotnessModelType.WEIGHTED_LINEAR,
            ),
            hotness_alpha=hot.get("alpha", {}).get("value", 1.0),
            hotness_beta=hot.get("beta", {}).get("value", 0.5),
            hotness_gamma=hot.get("gamma", {}).get("value", 0.3),
            hotness_decay_lambda=hot.get("decay_lambda", {}).get("value", 0.01),
            prefill_gpus=topo.get("prefill_gpus", {}).get("value", 1),
            rtl_engines=topo.get("rtl_engines", {}).get("value", 1),
            coordination_latency_us=coord.get("coordination_latency_us", {}).get("value", 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def classify_speedup_region(speedup: float, tolerance: float = 0.05) -> str:
    if speedup < 1.0 - tolerance:
        return "HARDWARE WORSE"
    if speedup > 1.0 + tolerance:
        return "HARDWARE BETTER"
    return "BREAK-EVEN"


def find_crossover_rps(
    config: Checkpoint1Config,
    rps_min: float = 100.0,
    rps_max: float = 50000.0,
    steps: int = 50,
) -> float | None:
    """Binary search for RPS where speedup crosses 1.0."""
    prev_speedup = None
    crossover = None

    for i in range(steps):
        rps = rps_min + (rps_max - rps_min) * i / (steps - 1)
        cfg = Checkpoint1Config(**{**config.to_dict(), "requests_per_sec": rps})
        result = Checkpoint1Simulator(cfg).run()
        speedup = result.speedup

        if prev_speedup is not None:
            if (prev_speedup < 1.0 <= speedup) or (prev_speedup > 1.0 >= speedup):
                crossover = rps
                break
        prev_speedup = speedup

    return crossover


class Checkpoint1Simulator:
    """Deterministic simulator for checkpoint 1."""

    def __init__(self, config: Checkpoint1Config):
        self.config = config

    def run(self) -> SimulationResult:
        cfg = self.config
        metadata_ops = cfg.metadata_ops_per_request
        arrival_rate = cfg.requests_per_sec * cfg.burstiness
        metadata_demand_ops = arrival_rate * metadata_ops

        # First pass: estimate service rates without queueing
        sw_breakdown_pre, _ = compute_software_latency(
            lookup_us=cfg.sw_lookup_latency_us,
            metadata_update_us=cfg.sw_metadata_update_latency_us,
            score_us=cfg.sw_score_calc_latency_us,
            selection_us=cfg.sw_candidate_selection_latency_us,
            coordination_us=cfg.sw_sync_overhead_us + cfg.coordination_latency_us,
            memory_access_us=cfg.memory_access_latency_us,
            cache_hit_rate=cfg.cache_hit_rate,
            cpu_contention=cfg.cpu_contention,
            metadata_entries=cfg.metadata_entries,
            candidate_k=cfg.candidate_count_k,
            parallelism=cfg.software_parallelism,
            metadata_ops=metadata_ops,
            arrival_rate=arrival_rate,
            service_rate=1e12,  # no queue for service rate estimate
        )

        sw_throughput_pre = compute_software_throughput(
            cpu_cores=cfg.cpu_cores,
            parallelism=cfg.software_parallelism,
            per_op_latency_us=sw_breakdown_pre.total / max(metadata_ops, 1),
            metadata_ops_per_request=metadata_ops,
            requests_per_sec=cfg.requests_per_sec,
        )

        hw_breakdown_pre, _, hw_cycles = compute_hardware_latency(
            clock_freq_mhz=cfg.clock_freq_mhz,
            pipeline_depth=cfg.pipeline_depth,
            lookup_ports=cfg.lookup_ports,
            parallel_ops=cfg.parallel_metadata_ops,
            candidate_k=cfg.candidate_count_k,
            scoring_units=cfg.scoring_units,
            metadata_entries=cfg.metadata_entries,
            hotness_complexity=cfg.hotness_complexity,
            metadata_ops=metadata_ops,
            request_arrival_us=cfg.request_arrival_overhead_us,
            hw_to_sw_us=cfg.hw_to_sw_overhead_us,
            coordination_us=cfg.coordination_latency_us,
            arrival_rate=arrival_rate,
            service_rate=1e12,
        )

        hw_throughput_pre = compute_hardware_throughput(
            clock_freq_mhz=cfg.clock_freq_mhz,
            pipeline_depth=cfg.pipeline_depth,
            lookup_ports=cfg.lookup_ports,
            parallel_ops=cfg.parallel_metadata_ops,
            scoring_units=cfg.scoring_units,
            candidate_comparison_units=cfg.candidate_comparison_units,
            metadata_memory_bandwidth_gbps=cfg.metadata_memory_bandwidth_gbps,
            metadata_bytes_per_block=cfg.metadata_bytes_per_block,
            metadata_ops_per_request=metadata_ops,
            requests_per_sec=cfg.requests_per_sec,
            cycles_per_operation=hw_cycles,
            queue_depth=cfg.queue_depth,
        )

        # Second pass: with queueing
        sw_breakdown, sw_queueing = compute_software_latency(
            lookup_us=cfg.sw_lookup_latency_us,
            metadata_update_us=cfg.sw_metadata_update_latency_us,
            score_us=cfg.sw_score_calc_latency_us,
            selection_us=cfg.sw_candidate_selection_latency_us,
            coordination_us=cfg.sw_sync_overhead_us + cfg.coordination_latency_us,
            memory_access_us=cfg.memory_access_latency_us,
            cache_hit_rate=cfg.cache_hit_rate,
            cpu_contention=cfg.cpu_contention,
            metadata_entries=cfg.metadata_entries,
            candidate_k=cfg.candidate_count_k,
            parallelism=cfg.software_parallelism,
            metadata_ops=metadata_ops,
            arrival_rate=metadata_demand_ops,
            service_rate=sw_throughput_pre.sustainable_ops_per_sec,
        )

        hw_breakdown, hw_queueing, hw_cycles = compute_hardware_latency(
            clock_freq_mhz=cfg.clock_freq_mhz,
            pipeline_depth=cfg.pipeline_depth,
            lookup_ports=cfg.lookup_ports,
            parallel_ops=cfg.parallel_metadata_ops,
            candidate_k=cfg.candidate_count_k,
            scoring_units=cfg.scoring_units,
            metadata_entries=cfg.metadata_entries,
            hotness_complexity=cfg.hotness_complexity,
            metadata_ops=metadata_ops,
            request_arrival_us=cfg.request_arrival_overhead_us,
            hw_to_sw_us=cfg.hw_to_sw_overhead_us,
            coordination_us=cfg.coordination_latency_us,
            arrival_rate=metadata_demand_ops,
            service_rate=hw_throughput_pre.sustainable_ops_per_sec,
        )

        sw_throughput = compute_software_throughput(
            cpu_cores=cfg.cpu_cores,
            parallelism=cfg.software_parallelism,
            per_op_latency_us=sw_breakdown.total / max(metadata_ops, 1),
            metadata_ops_per_request=metadata_ops,
            requests_per_sec=cfg.requests_per_sec,
        )

        hw_throughput = compute_hardware_throughput(
            clock_freq_mhz=cfg.clock_freq_mhz,
            pipeline_depth=cfg.pipeline_depth,
            lookup_ports=cfg.lookup_ports,
            parallel_ops=cfg.parallel_metadata_ops,
            scoring_units=cfg.scoring_units,
            candidate_comparison_units=cfg.candidate_comparison_units,
            metadata_memory_bandwidth_gbps=cfg.metadata_memory_bandwidth_gbps,
            metadata_bytes_per_block=cfg.metadata_bytes_per_block,
            metadata_ops_per_request=metadata_ops,
            requests_per_sec=cfg.requests_per_sec,
            cycles_per_operation=hw_cycles,
            queue_depth=cfg.queue_depth,
        )

        sw_latency = sw_breakdown.total
        hw_latency = hw_breakdown.total
        speedup = sw_latency / hw_latency if hw_latency > 0 else 0.0

        # Hotness distribution (sample subset for plotting)
        sample_blocks = min(cfg.num_kv_blocks, 512)
        hotness_dist = generate_hotness_distribution(
            num_blocks=sample_blocks,
            model_type=cfg.hotness_model_type,
            alpha=cfg.hotness_alpha,
            beta=cfg.hotness_beta,
            gamma=cfg.hotness_gamma,
            decay_lambda=cfg.hotness_decay_lambda,
            hot_fraction=0.2,
            seed=42,
        )

        hw_cycles_hotness = hotness_update_cycles(cfg.hotness_complexity, cfg.pipeline_depth)
        hotness_update_rate = metadata_demand_ops * (
            cfg.metadata_update_ops_per_request / max(metadata_ops, 1)
        )

        return SimulationResult(
            software_latency_us=sw_latency,
            hardware_latency_us=hw_latency,
            speedup=speedup,
            software_breakdown=sw_breakdown,
            hardware_breakdown=hw_breakdown,
            software_throughput=sw_throughput,
            hardware_throughput=hw_throughput,
            software_queueing=sw_queueing,
            hardware_queueing=hw_queueing,
            rtl_utilization=hw_throughput.utilization,
            metadata_capacity=cfg.metadata_storage_capacity,
            coordination_overhead_us=cfg.coordination_latency_us + cfg.hw_to_sw_overhead_us,
            metadata_ops_per_sec=metadata_demand_ops,
            hotness_distribution=hotness_dist,
            hotness_update_rate=hotness_update_rate,
            hw_cycles_per_hotness_update=hw_cycles_hotness,
            region=classify_speedup_region(speedup),
        )

    def sweep_rps(self, rps_values: list[float]) -> list[tuple[float, SimulationResult]]:
        results = []
        for rps in rps_values:
            cfg = Checkpoint1Config(**{**self.config.to_dict(), "requests_per_sec": rps})
            results.append((rps, Checkpoint1Simulator(cfg).run()))
        return results

    def sweep_metadata_entries(self, entry_values: list[int]) -> list[tuple[int, SimulationResult]]:
        results = []
        for n in entry_values:
            cfg = Checkpoint1Config(**{**self.config.to_dict(), "num_kv_blocks": n})
            results.append((n, Checkpoint1Simulator(cfg).run()))
        return results

    def sweep_candidate_k(self, k_values: list[int]) -> list[tuple[int, SimulationResult]]:
        results = []
        for k in k_values:
            cfg = Checkpoint1Config(**{**self.config.to_dict(), "candidate_count_k": k})
            results.append((k, Checkpoint1Simulator(cfg).run()))
        return results
