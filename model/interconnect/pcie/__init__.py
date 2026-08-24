"""PCIe interconnect analytical model."""

from model.interconnect.pcie.bandwidth import (
    BandwidthBreakdown,
    compute_effective_bandwidth_gbps,
    compute_pcie_bandwidth,
    lane_count,
)
from model.interconnect.pcie.config import (
    PCIeConfig,
    PCIeGeneration,
    PCIeMPS,
    PCIeMRRS,
    PCIeWidth,
)
from model.interconnect.pcie.latency import (
    PCIeLatencyResult,
    compute_read_latency_us,
    compute_transfer_time_us,
    compute_write_latency_us,
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
