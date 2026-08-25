"""Topology node, edge, and path types (Layers 3–4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """System component types (no internal sub-modeling)."""

    CPU = "CPU"
    GPU = "GPU"
    FPGA = "FPGA"
    PREFILL_POOL = "PREFILL_POOL"
    DECODE_POOL = "DECODE_POOL"
    HBM_MEMORY = "HBM_MEMORY"
    PCIE_ROOT_COMPLEX = "PCIE_ROOT_COMPLEX"
    PCIE_SWITCH = "PCIE_SWITCH"


class InterconnectType(str, Enum):
    """Communication link types for architectural exploration."""

    PCIE = "PCIe"
    NVLINK = "NVLink"
    INFINIBAND = "InfiniBand"


NODE_TYPE_ALIASES: dict[str, str] = {
    "MEMORY": NodeType.HBM_MEMORY.value,
}


def normalize_node_type(value: str) -> str:
    return NODE_TYPE_ALIASES.get(value, value)


@dataclass
class Node:
    """A vertex in the system topology graph."""

    id: str
    type: NodeType
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A directed communication link between two topology nodes."""

    id: str
    source_id: str
    dest_id: str
    interconnect_type: InterconnectType
    bandwidth_GBps: float
    latency_us: float
    width: str = "x16"
    generation: str = "Gen4"
    serialization_enabled: bool = True
    congestion_factor: float = 1.0
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def reverse(self, new_id: str) -> Edge:
        """Return a bidirectional counterpart edge (dest -> source)."""
        return Edge(
            id=new_id,
            source_id=self.dest_id,
            dest_id=self.source_id,
            interconnect_type=self.interconnect_type,
            bandwidth_GBps=self.bandwidth_GBps,
            latency_us=self.latency_us,
            width=self.width,
            generation=self.generation,
            serialization_enabled=self.serialization_enabled,
            congestion_factor=self.congestion_factor,
            enabled=self.enabled,
            metadata=dict(self.metadata),
        )


@dataclass
class Path:
    """Ordered node/edge sequence from source to destination."""

    nodes: list[Node]
    edges: list[Edge]

    @property
    def hop_count(self) -> int:
        return len(self.edges)


@dataclass
class PathSummary:
    """Structured result of path analysis."""

    nodes: list[str]
    edges: list[str]
    hop_count: int
    bottleneck_bandwidth_GBps: float
    latency_us: float
    serialization_time_us: float
    congestion_penalty_us: float
    transfer_time_us: float
    utilization: float = 0.0
    bandwidth_label: str = "EFFECTIVE"
    latency_label: str = "ANALYTICAL ESTIMATE"
    node_hop_latency_us: float = 0.0
    memory_service_latency_us: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "hop_count": self.hop_count,
            "bottleneck_bandwidth_GBps": self.bottleneck_bandwidth_GBps,
            "latency_us": self.latency_us,
            "serialization_time_us": self.serialization_time_us,
            "congestion_penalty_us": self.congestion_penalty_us,
            "transfer_time_us": self.transfer_time_us,
            "utilization": self.utilization,
            "bandwidth_label": self.bandwidth_label,
            "latency_label": self.latency_label,
            "node_hop_latency_us": self.node_hop_latency_us,
            "memory_service_latency_us": self.memory_service_latency_us,
            "metadata": dict(self.metadata),
        }
