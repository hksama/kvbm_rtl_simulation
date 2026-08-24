"""PCIe configuration object (Part 4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PCIeGeneration(str, Enum):
    GEN3 = "Gen3"
    GEN4 = "Gen4"
    GEN5 = "Gen5"
    GEN6 = "Gen6"


class PCIeWidth(str, Enum):
    X1 = "x1"
    X2 = "x2"
    X4 = "x4"
    X8 = "x8"
    X16 = "x16"


class PCIeMPS(int, Enum):
    """Max Payload Size (bytes)."""

    B128 = 128
    B256 = 256
    B512 = 512
    B1024 = 1024


class PCIeMRRS(int, Enum):
    """Max Read Request Size (bytes)."""

    B128 = 128
    B256 = 256
    B512 = 512
    B1024 = 1024
    B2048 = 2048
    B4096 = 4096


@dataclass
class PCIeConfig:
    """Parameterized PCIe link configuration for analytical modeling."""

    generation: PCIeGeneration = PCIeGeneration.GEN4
    width: PCIeWidth = PCIeWidth.X16
    mps: PCIeMPS = PCIeMPS.B256
    mrrs: PCIeMRRS = PCIeMRRS.B512
    base_latency_us: float = 0.5
    switch_hops: int = 0
    congestion_factor: float = 1.0
    protocol_efficiency: float = 0.95
    switch_hop_latency_us: float = 0.15
    read_request_latency_us: float = 0.2
    memory_service_latency_us: float = 0.5
    endpoint_processing_us: float = 0.1
    memory_processing_us: float = 0.1
    congestion_base_penalty_us: float = 0.05

    def lane_count(self) -> int:
        return int(self.width.value.removeprefix("x"))
