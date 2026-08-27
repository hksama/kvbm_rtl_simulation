"""PCIe bandwidth analytical model (Part 5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from model.interconnect.pcie.config import PCIeConfig, PCIeGeneration, PCIeWidth

__all__ = [
    "BandwidthBreakdown",
    "BandwidthKind",
    "GENERATION_TABLE",
    "GEN6_FLIT_EFFICIENCY",
    "GEN6_FLIT_SIZE_BYTES",
    "compute_effective_bandwidth_gbps",
    "compute_pcie_bandwidth",
    "lane_count",
    "protocol_bandwidth_gbps",
    "raw_bandwidth_gbps",
]


class BandwidthKind(str, Enum):
    RAW = "RAW"
    PROTOCOL = "PROTOCOL"
    EFFECTIVE = "EFFECTIVE"


@dataclass(frozen=True)
class GenerationSpec:
    """Signaling parameters for one PCIe generation."""

    gt_per_s: float
    bits_per_symbol: int
    encoding_efficiency: float
    label: str


# Signaling table includes Gen1/Gen2 for completeness; PCIeConfig uses Gen3–Gen6.
GENERATION_TABLE: dict[str, GenerationSpec] = {
    "Gen1": GenerationSpec(
        gt_per_s=2.5,
        bits_per_symbol=1,
        encoding_efficiency=8 / 10,
        label="8b/10b NRZ",
    ),
    "Gen2": GenerationSpec(
        gt_per_s=5.0,
        bits_per_symbol=1,
        encoding_efficiency=8 / 10,
        label="8b/10b NRZ",
    ),
    "Gen3": GenerationSpec(
        gt_per_s=8.0,
        bits_per_symbol=1,
        encoding_efficiency=128 / 130,
        label="128b/130b NRZ",
    ),
    "Gen4": GenerationSpec(
        gt_per_s=16.0,
        bits_per_symbol=1,
        encoding_efficiency=128 / 130,
        label="128b/130b NRZ",
    ),
    "Gen5": GenerationSpec(
        gt_per_s=32.0,
        bits_per_symbol=1,
        encoding_efficiency=128 / 130,
        label="128b/130b NRZ",
    ),
}

# Gen6: 64 GT/s PAM4 (2 bits/symbol), 256-byte FLIT, 236-byte payload (pcie_impl.md).
GEN6_FLIT_SIZE_BYTES = 256
GEN6_USABLE_PAYLOAD_BYTES = 236
GEN6_FLIT_EFFICIENCY = GEN6_USABLE_PAYLOAD_BYTES / GEN6_FLIT_SIZE_BYTES

GENERATION_TABLE["Gen6"] = GenerationSpec(
    gt_per_s=64.0,
    bits_per_symbol=2,
    encoding_efficiency=GEN6_FLIT_EFFICIENCY,
    label="PAM4 FLIT mode (256 B FLIT, FEC/CRC overhead)",
)


def lane_count(width: PCIeWidth | str) -> int:
    value = width.value if isinstance(width, PCIeWidth) else str(width)
    return int(value.removeprefix("x"))


def raw_bandwidth_gbps(lanes: int, generation: PCIeGeneration | str) -> float:
    """Per-direction RAW bandwidth in GB/s."""
    gen_key = generation.value if isinstance(generation, PCIeGeneration) else str(generation)
    spec = GENERATION_TABLE[gen_key]
    return lanes * spec.gt_per_s * spec.bits_per_symbol / 8.0


def protocol_bandwidth_gbps(lanes: int, generation: PCIeGeneration | str) -> float:
    """Per-direction PROTOCOL-level usable bandwidth in GB/s."""
    gen_key = generation.value if isinstance(generation, PCIeGeneration) else str(generation)
    spec = GENERATION_TABLE[gen_key]
    return raw_bandwidth_gbps(lanes, gen_key) * spec.encoding_efficiency


@dataclass
class BandwidthBreakdown:
    """Distinct RAW, PROTOCOL, and EFFECTIVE bandwidth values."""

    raw_GBps: float
    protocol_GBps: float
    effective_GBps: float
    generation: str
    width: str
    lanes: int
    encoding_label: str

    def get(self, kind: BandwidthKind) -> float:
        if kind == BandwidthKind.RAW:
            return self.raw_GBps
        if kind == BandwidthKind.PROTOCOL:
            return self.protocol_GBps
        return self.effective_GBps

    def labeled_values(self) -> dict[str, float]:
        return {
            "RAW": self.raw_GBps,
            "PROTOCOL": self.protocol_GBps,
            "EFFECTIVE": self.effective_GBps,
        }

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "raw_GBps": self.raw_GBps,
            "protocol_GBps": self.protocol_GBps,
            "effective_GBps": self.effective_GBps,
            "generation": self.generation,
            "width": self.width,
            "lanes": self.lanes,
            "encoding_label": self.encoding_label,
        }


def compute_pcie_bandwidth(config: PCIeConfig) -> BandwidthBreakdown:
    """Compute RAW, PROTOCOL, and EFFECTIVE per-direction bandwidth."""
    lanes = config.lane_count()
    gen_key = config.generation.value
    spec = GENERATION_TABLE[gen_key]
    raw = raw_bandwidth_gbps(lanes, gen_key)
    protocol = raw * spec.encoding_efficiency
    effective = protocol * config.protocol_efficiency
    return BandwidthBreakdown(
        raw_GBps=raw,
        protocol_GBps=protocol,
        effective_GBps=effective,
        generation=gen_key,
        width=config.width.value,
        lanes=lanes,
        encoding_label=spec.label,
    )


def compute_effective_bandwidth_gbps(config: PCIeConfig) -> float:
    """Return EFFECTIVE per-direction bandwidth for serialization models."""
    return compute_pcie_bandwidth(config).effective_GBps
