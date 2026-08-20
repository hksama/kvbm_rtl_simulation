"""Shared types and enums for the analytical model."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParameterLabel(str, Enum):
    ASSUMPTION = "ASSUMPTION"
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"


class HotnessModelType(str, Enum):
    WEIGHTED_LINEAR = "weighted_linear"
    LRU_LIKE = "lru_like"
    LFU_LIKE = "lfu_like"
    RECENCY_WEIGHTED = "recency_weighted"
    EXP_DECAY = "exp_decay"


@dataclass
class LatencyBreakdown:
    """Per-component latency in microseconds."""

    lookup: float = 0.0
    metadata_update: float = 0.0
    score: float = 0.0
    candidate_selection: float = 0.0
    coordination: float = 0.0
    queue: float = 0.0
    request_arrival: float = 0.0
    hw_to_sw: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.lookup
            + self.metadata_update
            + self.score
            + self.candidate_selection
            + self.coordination
            + self.queue
            + self.request_arrival
            + self.hw_to_sw
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "lookup": self.lookup,
            "metadata_update": self.metadata_update,
            "score": self.score,
            "candidate_selection": self.candidate_selection,
            "coordination": self.coordination,
            "queue": self.queue,
            "request_arrival": self.request_arrival,
            "hw_to_sw": self.hw_to_sw,
            "total": self.total,
        }


@dataclass
class ThroughputMetrics:
    """Throughput and capacity metrics."""

    ops_per_sec: float = 0.0
    theoretical_ops_per_sec: float = 0.0
    sustainable_ops_per_sec: float = 0.0
    sustainable_rps: float = 0.0
    achieved_rps: float = 0.0
    utilization: float = 0.0
    cycles_per_operation: float = 0.0
    operations_per_cycle: float = 0.0


@dataclass
class QueueingMetrics:
    """Queueing delay estimates."""

    utilization: float = 0.0
    avg_latency_us: float = 0.0
    p95_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    saturated: bool = False


@dataclass
class HotnessBlock:
    """Single block hotness state for distribution plots."""

    block_id: int
    access_count: int
    recency: float
    reference_count: int
    hotness: float
    classification: str  # "hot" or "cold"


@dataclass
class LabeledValue:
    value: Any
    label: ParameterLabel = ParameterLabel.ASSUMPTION
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "label": self.label.value, "unit": self.unit}


@dataclass
class SimulationResult:
    """Complete output of a simulation run."""

    software_latency_us: float = 0.0
    hardware_latency_us: float = 0.0
    speedup: float = 0.0
    software_breakdown: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    hardware_breakdown: LatencyBreakdown = field(default_factory=LatencyBreakdown)
    software_throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    hardware_throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    software_queueing: QueueingMetrics = field(default_factory=QueueingMetrics)
    hardware_queueing: QueueingMetrics = field(default_factory=QueueingMetrics)
    rtl_utilization: float = 0.0
    metadata_capacity: int = 0
    coordination_overhead_us: float = 0.0
    metadata_ops_per_sec: float = 0.0
    hotness_distribution: list[HotnessBlock] = field(default_factory=list)
    hotness_update_rate: float = 0.0
    hw_cycles_per_hotness_update: float = 0.0
    crossover_rps: float | None = None
    crossover_metadata_ops: float | None = None
    region: str = "BREAK-EVEN"  # HARDWARE WORSE | BREAK-EVEN | HARDWARE BETTER
