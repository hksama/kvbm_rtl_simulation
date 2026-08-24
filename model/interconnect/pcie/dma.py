"""PCIe DMA analytical model (Part 7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from model.interconnect.pcie.config import PCIeConfig
from model.interconnect.pcie.flit import apply_flit_overhead, flit_summary
from model.interconnect.pcie.latency import compute_read_latency_us, compute_write_latency_us
from model.interconnect.pcie.tlp import (
    compute_transfer_on_wire_bytes,
    fragment_read_transfer,
    fragment_transfer,
    read_on_wire_bytes,
    write_on_wire_bytes,
)

LATENCY_LABEL = "ANALYTICAL ESTIMATE"


class DMAOperation(str, Enum):
    DMA_READ = "DMA_READ"
    DMA_WRITE = "DMA_WRITE"


@dataclass
class DMAEngine:
    """DMA request generator operating through the PCIe hierarchy."""

    queue_depth: int = 64
    max_outstanding_reads: int = 16
    request_size: int = 4096
    completion_size: int = 4096
    read_latency_us: float = 0.1
    write_latency_us: float = 0.05


@dataclass
class DMATransferResult:
    operation: DMAOperation
    payload_bytes: int
    request_tlps: int
    completion_tlps: int
    tlp_on_wire_bytes: int
    serialized_bytes: int
    flit_count: int
    pcie_latency_us: float
    dma_overhead_us: float
    queue_penalty_us: float
    total_time_us: float
    label: str = LATENCY_LABEL

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "operation": self.operation.value,
            "payload_bytes": self.payload_bytes,
            "request_tlps": self.request_tlps,
            "completion_tlps": self.completion_tlps,
            "tlp_on_wire_bytes": self.tlp_on_wire_bytes,
            "serialized_bytes": self.serialized_bytes,
            "flit_count": self.flit_count,
            "pcie_latency_us": self.pcie_latency_us,
            "dma_overhead_us": self.dma_overhead_us,
            "queue_penalty_us": self.queue_penalty_us,
            "total_time_us": self.total_time_us,
            "label": self.label,
        }


def _queue_penalty_us(engine: DMAEngine, operation: DMAOperation, tlp_count: int) -> float:
    if engine.queue_depth <= 0:
        return 0.0
    utilization = min(1.0, tlp_count / engine.queue_depth)
    base = 0.02 * utilization
    if operation == DMAOperation.DMA_READ and tlp_count > engine.max_outstanding_reads:
        outstanding_penalty = 0.05 * (tlp_count - engine.max_outstanding_reads)
        return base + outstanding_penalty
    return base


def compute_dma_transfer(
    config: PCIeConfig,
    engine: DMAEngine,
    payload_bytes: int,
    operation: DMAOperation,
    *,
    requester: str = "FPGA",
    completer: str = "Memory",
) -> DMATransferResult:
    """
    Model DMA read/write through PCIe (never bypassing the link).

    DMA READ:  FPGA → MRd TLPs → PCIe → Memory → CplD → FPGA
    DMA WRITE: FPGA → MWr TLPs → PCIe → Memory
    """
    if payload_bytes <= 0:
        return DMATransferResult(
            operation=operation,
            payload_bytes=0,
            request_tlps=0,
            completion_tlps=0,
            tlp_on_wire_bytes=0,
            serialized_bytes=0,
            flit_count=0,
            pcie_latency_us=0.0,
            dma_overhead_us=0.0,
            queue_penalty_us=0.0,
            total_time_us=0.0,
        )

    if operation == DMAOperation.DMA_WRITE:
        write_tlps = fragment_transfer(
            payload_bytes,
            config.mps,
            requester=requester,
            completer=completer,
        )
        tlp_on_wire = write_on_wire_bytes(payload_bytes, config.mps)
        request_tlps = len(write_tlps)
        completion_tlps = 0
        pcie = compute_write_latency_us(config, payload_bytes)
        dma_overhead = engine.write_latency_us
    else:
        read_reqs, completions = fragment_read_transfer(
            payload_bytes,
            config.mps,
            config.mrrs.value,
            requester=requester,
            completer=completer,
        )
        tlp_on_wire = read_on_wire_bytes(payload_bytes, config.mps, config.mrrs.value)
        request_tlps = len(read_reqs)
        completion_tlps = len(completions)
        pcie = compute_read_latency_us(config, payload_bytes)
        dma_overhead = engine.read_latency_us

    serialized = apply_flit_overhead(tlp_on_wire, config.generation)
    flit_count = flit_summary(tlp_on_wire)["flit_count"] if tlp_on_wire else 0
    tlp_count = request_tlps + completion_tlps
    queue_penalty = _queue_penalty_us(engine, operation, tlp_count)
    total = pcie.transfer_time_us + dma_overhead + queue_penalty

    return DMATransferResult(
        operation=operation,
        payload_bytes=payload_bytes,
        request_tlps=request_tlps,
        completion_tlps=completion_tlps,
        tlp_on_wire_bytes=tlp_on_wire,
        serialized_bytes=int(serialized),
        flit_count=int(flit_count),
        pcie_latency_us=pcie.transfer_time_us,
        dma_overhead_us=dma_overhead,
        queue_penalty_us=queue_penalty,
        total_time_us=total,
    )
