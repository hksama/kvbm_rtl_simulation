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
from model.interconnect.pcie.dma import DMAEngine, DMAOperation, DMATransferResult, compute_dma_transfer
from model.interconnect.pcie.flit import (
    FLIT_SIZE_BYTES,
    FLIT_TLP_PAYLOAD_AREA_BYTES,
    apply_flit_overhead,
    flit_on_wire_bytes,
    flit_summary,
    pack_bytes_into_flits,
)
from model.interconnect.pcie.latency import (
    PCIeLatencyResult,
    compute_read_latency_us,
    compute_transfer_time_us,
    compute_write_latency_us,
)
from model.interconnect.pcie.tlp import (
    TLP,
    TLPType,
    compute_transfer_on_wire_bytes,
    fragment_read_transfer,
    fragment_transfer,
    tlp_summary,
    write_on_wire_bytes,
)

__all__ = [
    "BandwidthBreakdown",
    "DMAEngine",
    "DMAOperation",
    "DMATransferResult",
    "FLIT_SIZE_BYTES",
    "FLIT_TLP_PAYLOAD_AREA_BYTES",
    "PCIeConfig",
    "PCIeGeneration",
    "PCIeLatencyResult",
    "PCIeMPS",
    "PCIeMRRS",
    "PCIeWidth",
    "TLP",
    "TLPType",
    "apply_flit_overhead",
    "compute_dma_transfer",
    "compute_effective_bandwidth_gbps",
    "compute_pcie_bandwidth",
    "compute_read_latency_us",
    "compute_transfer_on_wire_bytes",
    "compute_transfer_time_us",
    "compute_write_latency_us",
    "flit_on_wire_bytes",
    "flit_summary",
    "fragment_read_transfer",
    "fragment_transfer",
    "lane_count",
    "pack_bytes_into_flits",
    "tlp_summary",
    "write_on_wire_bytes",
]
