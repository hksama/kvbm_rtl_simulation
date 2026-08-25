"""NVLink edge configuration (no internal fabric modeling)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from model.components.base import build_config, config_to_dict

LATENCY_LABEL = "ANALYTICAL ESTIMATE"


@dataclass
class NVLinkConfig:
    version: str = "NVLink4"
    links: int = 18
    bandwidth_per_link_GBps: float = 50.0
    aggregate_bandwidth_GBps: float = 900.0
    latency_us: float = 0.5
    bidirectional: bool = True
    congestion_factor: float = 1.0
    protocol_efficiency: float = 0.95


@dataclass
class NVLinkBandwidthBreakdown:
    """RAW aggregate and EFFECTIVE application bandwidth (ASSUMPTION)."""

    raw_GBps: float
    effective_GBps: float
    bandwidth_label: str = "EFFECTIVE"


@dataclass
class NVLinkLatencyResult:
    transfer_time_us: float
    fixed_latency_us: float
    serialization_time_us: float
    congestion_penalty_us: float
    label: str = LATENCY_LABEL


DEFAULT_NVLINK: dict[str, Any] = config_to_dict(NVLinkConfig())


def nvlink_config_from_dict(data: dict[str, Any] | None) -> NVLinkConfig:
    config = build_config(NVLinkConfig, data, DEFAULT_NVLINK)
    if config.aggregate_bandwidth_GBps <= 0:
        config.aggregate_bandwidth_GBps = config.links * config.bandwidth_per_link_GBps
    return config


def nvlink_config_to_dict(config: NVLinkConfig) -> dict[str, Any]:
    return config_to_dict(config)


def raw_nvlink_bandwidth_gbps(config: NVLinkConfig) -> float:
    if config.aggregate_bandwidth_GBps > 0:
        return config.aggregate_bandwidth_GBps
    return config.links * config.bandwidth_per_link_GBps


def effective_nvlink_bandwidth_gbps(config: NVLinkConfig) -> float:
    return raw_nvlink_bandwidth_gbps(config) * config.protocol_efficiency


def compute_nvlink_bandwidth(config: NVLinkConfig) -> NVLinkBandwidthBreakdown:
    raw = raw_nvlink_bandwidth_gbps(config)
    return NVLinkBandwidthBreakdown(
        raw_GBps=raw,
        effective_GBps=raw * config.protocol_efficiency,
    )


def _congestion_penalty_us(config: NVLinkConfig) -> float:
    if config.congestion_factor <= 1.0:
        return 0.0
    return config.latency_us * 0.1 * (config.congestion_factor - 1.0)


def _serialization_time_us(config: NVLinkConfig, payload_bytes: int) -> float:
    if payload_bytes <= 0:
        return 0.0
    effective_bw = effective_nvlink_bandwidth_gbps(config)
    if effective_bw <= 0:
        return float("inf")
    return payload_bytes / (effective_bw * 1000.0)


def compute_nvlink_transfer_time_us(
    config: NVLinkConfig,
    payload_bytes: int,
) -> NVLinkLatencyResult:
    """One-direction NVLink transfer: fixed hop + serialization + congestion."""
    serialization = _serialization_time_us(config, payload_bytes)
    congestion = _congestion_penalty_us(config)
    fixed = config.latency_us
    total = fixed + serialization + congestion
    return NVLinkLatencyResult(
        transfer_time_us=total,
        fixed_latency_us=fixed,
        serialization_time_us=serialization,
        congestion_penalty_us=congestion,
        label=LATENCY_LABEL,
    )
