"""InfiniBand edge configuration (no subnet / SM modeling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict

LATENCY_LABEL = "ANALYTICAL ESTIMATE"


@dataclass
class InfiniBandConfig:
    generation: str = "NDR"
    signaling_rate_gbps: float = 400.0
    lanes: int = 4
    raw_bandwidth_GBps: float = 50.0
    effective_bandwidth_GBps: float = 47.5
    latency_us: float = 2.0
    mtu_bytes: int = 4096
    rdma_enabled: bool = True
    protocol_efficiency: float = 0.95
    congestion_factor: float = 1.0


@dataclass
class InfiniBandBandwidthBreakdown:
    """RAW line rate and EFFECTIVE application bandwidth (ASSUMPTION)."""

    raw_GBps: float
    effective_GBps: float
    bandwidth_label: str = "EFFECTIVE"


@dataclass
class InfiniBandLatencyResult:
    transfer_time_us: float
    fixed_latency_us: float
    serialization_time_us: float
    congestion_penalty_us: float
    label: str = LATENCY_LABEL


DEFAULT_INFINIBAND: dict[str, Any] = config_to_dict(InfiniBandConfig())


def infiniband_config_from_dict(data: dict[str, Any] | None) -> InfiniBandConfig:
    config = build_config(InfiniBandConfig, data, DEFAULT_INFINIBAND)
    if config.raw_bandwidth_GBps <= 0:
        config.raw_bandwidth_GBps = config.signaling_rate_gbps / 8.0
    if config.effective_bandwidth_GBps <= 0:
        config.effective_bandwidth_GBps = config.raw_bandwidth_GBps * config.protocol_efficiency
    return config


def infiniband_config_to_dict(config: InfiniBandConfig) -> dict[str, Any]:
    return config_to_dict(config)


def raw_infiniband_bandwidth_gbps(config: InfiniBandConfig) -> float:
    if config.raw_bandwidth_GBps > 0:
        return config.raw_bandwidth_GBps
    return config.signaling_rate_gbps / 8.0


def effective_infiniband_bandwidth_gbps(config: InfiniBandConfig) -> float:
    if config.effective_bandwidth_GBps > 0:
        return config.effective_bandwidth_GBps
    return raw_infiniband_bandwidth_gbps(config) * config.protocol_efficiency


def compute_infiniband_bandwidth(config: InfiniBandConfig) -> InfiniBandBandwidthBreakdown:
    raw = raw_infiniband_bandwidth_gbps(config)
    effective = (
        config.effective_bandwidth_GBps
        if config.effective_bandwidth_GBps > 0
        else raw * config.protocol_efficiency
    )
    return InfiniBandBandwidthBreakdown(raw_GBps=raw, effective_GBps=effective)


def _congestion_penalty_us(config: InfiniBandConfig) -> float:
    if config.congestion_factor <= 1.0:
        return 0.0
    return config.latency_us * 0.1 * (config.congestion_factor - 1.0)


def _serialization_time_us(config: InfiniBandConfig, payload_bytes: int) -> float:
    if payload_bytes <= 0:
        return 0.0
    effective_bw = effective_infiniband_bandwidth_gbps(config)
    if effective_bw <= 0:
        return float("inf")
    return payload_bytes / (effective_bw * 1000.0)


def compute_infiniband_transfer_time_us(
    config: InfiniBandConfig,
    payload_bytes: int,
) -> InfiniBandLatencyResult:
    """One-direction InfiniBand transfer: fixed hop + serialization + congestion."""
    serialization = _serialization_time_us(config, payload_bytes)
    congestion = _congestion_penalty_us(config)
    fixed = config.latency_us
    total = fixed + serialization + congestion
    return InfiniBandLatencyResult(
        transfer_time_us=total,
        fixed_latency_us=fixed,
        serialization_time_us=serialization,
        congestion_penalty_us=congestion,
        label=LATENCY_LABEL,
    )
