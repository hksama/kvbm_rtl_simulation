"""PCIe latency analytical model (Part 6)."""

from __future__ import annotations

from dataclasses import dataclass

from model.interconnect.pcie.bandwidth import compute_effective_bandwidth_gbps
from model.interconnect.pcie.config import PCIeConfig
from model.interconnect.pcie.flit import apply_flit_overhead
from model.interconnect.pcie.tlp import compute_transfer_on_wire_bytes

LATENCY_LABEL = "ANALYTICAL ESTIMATE"


@dataclass
class PCIeLatencyResult:
    """Breakdown of PCIe transfer latency components."""

    transfer_time_us: float
    fixed_link_latency_us: float
    serialization_time_us: float
    hop_latency_us: float
    congestion_penalty_us: float
    endpoint_processing_us: float = 0.0
    request_latency_us: float = 0.0
    memory_service_latency_us: float = 0.0
    label: str = LATENCY_LABEL

    def to_dict(self) -> dict[str, float | str]:
        return {
            "transfer_time_us": self.transfer_time_us,
            "fixed_link_latency_us": self.fixed_link_latency_us,
            "serialization_time_us": self.serialization_time_us,
            "hop_latency_us": self.hop_latency_us,
            "congestion_penalty_us": self.congestion_penalty_us,
            "endpoint_processing_us": self.endpoint_processing_us,
            "request_latency_us": self.request_latency_us,
            "memory_service_latency_us": self.memory_service_latency_us,
            "label": self.label,
        }


def _serialized_bytes(config: PCIeConfig, payload_bytes: int, *, is_read: bool) -> int:
    """Payload + TLP headers (+ Gen6 FLIT encapsulation)."""
    tlp_bytes = compute_transfer_on_wire_bytes(config, payload_bytes, is_read=is_read)
    return apply_flit_overhead(tlp_bytes, config.generation)


def _serialization_time_us(config: PCIeConfig, payload_bytes: int, *, is_read: bool) -> float:
    serialized = _serialized_bytes(config, payload_bytes, is_read=is_read)
    effective_bw = compute_effective_bandwidth_gbps(config)
    return _bytes_to_time_us(serialized, effective_bw)


def _bytes_to_time_us(num_bytes: int, effective_bandwidth_gbps: float) -> float:
    if num_bytes <= 0:
        return 0.0
    if effective_bandwidth_gbps <= 0:
        return float("inf")
    bytes_per_us = effective_bandwidth_gbps * 1000.0
    return num_bytes / bytes_per_us


def _hop_latency_us(config: PCIeConfig) -> float:
    return config.switch_hops * config.switch_hop_latency_us


def _congestion_penalty_us(config: PCIeConfig) -> float:
    if config.congestion_factor <= 1.0:
        return 0.0
    return config.congestion_base_penalty_us * (config.congestion_factor - 1.0)


def compute_transfer_time_us(
    config: PCIeConfig,
    payload_bytes: int,
    *,
    include_endpoint_processing: bool = False,
) -> PCIeLatencyResult:
    """
    Generic one-direction transfer time:

        fixed_link_latency + serialization_time + hop_latency + congestion_penalty
    """
    serialization = _serialization_time_us(config, payload_bytes, is_read=False)
    fixed = config.base_latency_us
    hops = _hop_latency_us(config)
    congestion = _congestion_penalty_us(config)
    endpoint = config.endpoint_processing_us if include_endpoint_processing else 0.0
    total = fixed + serialization + hops + congestion + endpoint
    return PCIeLatencyResult(
        transfer_time_us=total,
        fixed_link_latency_us=fixed,
        serialization_time_us=serialization,
        hop_latency_us=hops,
        congestion_penalty_us=congestion,
        endpoint_processing_us=endpoint,
        label=LATENCY_LABEL,
    )


def compute_write_latency_us(config: PCIeConfig, payload_bytes: int) -> PCIeLatencyResult:
    """
    PCIe write analytical model:

        T_write = forward_path + endpoint/memory processing
    """
    forward = compute_transfer_time_us(
        config,
        payload_bytes,
        include_endpoint_processing=False,
    )
    memory_processing = config.memory_processing_us
    endpoint = config.endpoint_processing_us
    total = forward.transfer_time_us + memory_processing + endpoint
    return PCIeLatencyResult(
        transfer_time_us=total,
        fixed_link_latency_us=forward.fixed_link_latency_us,
        serialization_time_us=forward.serialization_time_us,
        hop_latency_us=forward.hop_latency_us,
        congestion_penalty_us=forward.congestion_penalty_us,
        endpoint_processing_us=endpoint + memory_processing,
        label=LATENCY_LABEL,
    )


def compute_read_latency_us(config: PCIeConfig, payload_bytes: int) -> PCIeLatencyResult:
    """
    PCIe read analytical model:

        T_read = request_latency + forward_path + memory_latency
                 + return_path + endpoint_processing

    Forward and return paths each incur link fixed latency, hop latency,
    congestion, and payload serialization on the completion path.
    """
    request = config.read_request_latency_us
    memory = config.memory_service_latency_us
    endpoint = config.endpoint_processing_us

    forward_fixed = config.base_latency_us + _hop_latency_us(config) + _congestion_penalty_us(config)
    return_fixed = config.base_latency_us + _hop_latency_us(config) + _congestion_penalty_us(config)
    completion_serialization = _serialization_time_us(config, payload_bytes, is_read=True)

    total = (
        request
        + forward_fixed
        + memory
        + return_fixed
        + completion_serialization
        + endpoint
    )
    return PCIeLatencyResult(
        transfer_time_us=total,
        fixed_link_latency_us=forward_fixed + return_fixed,
        serialization_time_us=completion_serialization,
        hop_latency_us=_hop_latency_us(config) * 2,
        congestion_penalty_us=_congestion_penalty_us(config) * 2,
        endpoint_processing_us=endpoint,
        request_latency_us=request,
        memory_service_latency_us=memory,
        label=LATENCY_LABEL,
    )
