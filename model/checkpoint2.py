"""Checkpoint 2: Prefill GPU scaling (2, 4, 8, 16 GPUs) with one RTL engine."""

from __future__ import annotations

from dataclasses import dataclass

from model.checkpoint1 import Checkpoint1Config, Checkpoint1Simulator
from model.scaling import PREFILL_GPU_OPTIONS_CP2, apply_prefill_scaling, metadata_load_factor
from model.types import SimulationResult


@dataclass
class PrefillScalingPoint:
    prefill_gpus: int
    result: SimulationResult


@dataclass
class PrefillScalingSweep:
    points: list[PrefillScalingPoint]

    @property
    def gpu_counts(self) -> list[int]:
        return [p.prefill_gpus for p in self.points]

    @property
    def rtl_utilizations(self) -> list[float]:
        return [p.result.rtl_utilization for p in self.points]

    @property
    def software_latencies(self) -> list[float]:
        return [p.result.software_latency_us for p in self.points]

    @property
    def hardware_latencies(self) -> list[float]:
        return [p.result.hardware_latency_us for p in self.points]

    @property
    def metadata_ops_per_sec(self) -> list[float]:
        return [p.result.metadata_ops_per_sec for p in self.points]


class Checkpoint2Simulator:
    """Simulator for checkpoint 2 prefill scaling."""

    PREFILL_GPU_OPTIONS = PREFILL_GPU_OPTIONS_CP2

    def __init__(self, config: Checkpoint1Config):
        if config.prefill_gpus not in self.PREFILL_GPU_OPTIONS:
            raise ValueError(
                f"Checkpoint 2 requires prefill_gpus in {self.PREFILL_GPU_OPTIONS}, got {config.prefill_gpus}"
            )
        if config.rtl_engines != 1:
            raise ValueError("Checkpoint 2 requires exactly 1 RTL engine")
        self.base_config = config

    def run(self) -> SimulationResult:
        scaled = apply_prefill_scaling(self.base_config)
        result = Checkpoint1Simulator(scaled).run()

        load = metadata_load_factor(self.base_config.prefill_gpus)
        result.metadata_ops_per_sec *= load

        if result.hardware_throughput.sustainable_ops_per_sec > 0:
            result.rtl_utilization = min(
                (result.metadata_ops_per_sec) / result.hardware_throughput.sustainable_ops_per_sec,
                1.0,
            )

        return result

    def sweep_prefill_gpus(
        self,
        gpu_counts: list[int] | None = None,
    ) -> PrefillScalingSweep:
        counts = gpu_counts or self.PREFILL_GPU_OPTIONS
        points: list[PrefillScalingPoint] = []

        for n in counts:
            if n not in self.PREFILL_GPU_OPTIONS:
                continue
            cfg = Checkpoint1Config(**{**self.base_config.to_dict(), "prefill_gpus": n})
            points.append(PrefillScalingPoint(prefill_gpus=n, result=Checkpoint2Simulator(cfg).run()))

        return PrefillScalingSweep(points=points)

    def sweep_rps(self, rps_values: list[float]) -> list[tuple[float, SimulationResult]]:
        results = []
        for rps in rps_values:
            cfg = Checkpoint1Config(**{**self.base_config.to_dict(), "requests_per_sec": rps})
            results.append((rps, Checkpoint2Simulator(cfg).run()))
        return results

    def sweep_metadata_entries(self, entry_values: list[int]) -> list[tuple[int, SimulationResult]]:
        results = []
        for n in entry_values:
            cfg = Checkpoint1Config(**{**self.base_config.to_dict(), "num_kv_blocks": n})
            results.append((n, Checkpoint2Simulator(cfg).run()))
        return results

    def sweep_candidate_k(self, k_values: list[int]) -> list[tuple[int, SimulationResult]]:
        results = []
        for k in k_values:
            cfg = Checkpoint1Config(**{**self.base_config.to_dict(), "candidate_count_k": k})
            results.append((k, Checkpoint2Simulator(cfg).run()))
        return results
