"""Layer 3: Interconnect models."""

from model.interconnect.pcie import (
    BandwidthBreakdown,
    PCIeConfig,
    PCIeGeneration,
    PCIeLatencyResult,
    PCIeMPS,
    PCIeMRRS,
    PCIeWidth,
    compute_effective_bandwidth_gbps,
    compute_pcie_bandwidth,
    compute_read_latency_us,
    compute_transfer_time_us,
    compute_write_latency_us,
    lane_count,
)

__all__ = [
    "BandwidthBreakdown",
    "PCIeConfig",
    "PCIeGeneration",
    "PCIeLatencyResult",
    "PCIeMPS",
    "PCIeMRRS",
    "PCIeWidth",
    "compute_effective_bandwidth_gbps",
    "compute_pcie_bandwidth",
    "compute_read_latency_us",
    "compute_transfer_time_us",
    "compute_write_latency_us",
    "lane_count",
]
