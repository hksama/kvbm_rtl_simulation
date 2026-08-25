"""Tests for NVLink and InfiniBand analytical edge models."""

import pytest

from model.interconnect.infiniband import (
    InfiniBandConfig,
    compute_infiniband_bandwidth,
    compute_infiniband_transfer_time_us,
    effective_infiniband_bandwidth_gbps,
    infiniband_config_from_dict,
)
from model.interconnect.nvlink import (
    NVLinkConfig,
    compute_nvlink_bandwidth,
    compute_nvlink_transfer_time_us,
    effective_nvlink_bandwidth_gbps,
    nvlink_config_from_dict,
)


class TestNVLinkModel:
    def test_effective_bandwidth_uses_protocol_efficiency(self):
        config = NVLinkConfig(
            links=18,
            bandwidth_per_link_GBps=50.0,
            aggregate_bandwidth_GBps=900.0,
            protocol_efficiency=0.95,
        )
        assert effective_nvlink_bandwidth_gbps(config) == pytest.approx(855.0)

    def test_bandwidth_breakdown(self):
        config = nvlink_config_from_dict({
            "links": 12,
            "bandwidth_per_link_GBps": 50.0,
            "aggregate_bandwidth_GBps": 0.0,
        })
        breakdown = compute_nvlink_bandwidth(config)
        assert breakdown.raw_GBps == pytest.approx(600.0)
        assert breakdown.effective_GBps == pytest.approx(570.0)

    def test_transfer_time_increases_with_payload(self):
        config = NVLinkConfig(aggregate_bandwidth_GBps=900.0, latency_us=0.5)
        small = compute_nvlink_transfer_time_us(config, 4096)
        large = compute_nvlink_transfer_time_us(config, 4 * 1024 * 1024)
        assert large.transfer_time_us > small.transfer_time_us
        assert large.serialization_time_us > small.serialization_time_us


class TestInfiniBandModel:
    def test_effective_bandwidth_from_raw(self):
        config = InfiniBandConfig(
            raw_bandwidth_GBps=50.0,
            effective_bandwidth_GBps=0.0,
            protocol_efficiency=0.95,
        )
        assert effective_infiniband_bandwidth_gbps(config) == pytest.approx(47.5)

    def test_bandwidth_breakdown(self):
        config = infiniband_config_from_dict({"signaling_rate_gbps": 400.0, "lanes": 4})
        breakdown = compute_infiniband_bandwidth(config)
        assert breakdown.raw_GBps == pytest.approx(50.0)
        assert breakdown.effective_GBps == pytest.approx(47.5)

    def test_transfer_time_includes_fixed_latency(self):
        config = InfiniBandConfig(effective_bandwidth_GBps=47.5, latency_us=2.0)
        result = compute_infiniband_transfer_time_us(config, 0)
        assert result.transfer_time_us == pytest.approx(2.0)
        assert result.serialization_time_us == 0.0
