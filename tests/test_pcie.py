"""Tests for PCIe bandwidth and latency models (Parts 4–6)."""

import pytest

from model.interconnect.pcie import (
    PCIeConfig,
    PCIeGeneration,
    PCIeWidth,
    compute_effective_bandwidth_gbps,
    compute_pcie_bandwidth,
    compute_read_latency_us,
    compute_transfer_time_us,
    compute_write_latency_us,
    lane_count,
)
from model.interconnect.pcie.bandwidth import GENERATION_TABLE, protocol_bandwidth_gbps, raw_bandwidth_gbps


class TestPCIeBandwidth:
    def test_lane_scaling_x8_is_half_x16(self):
        gen4_x16 = raw_bandwidth_gbps(16, "Gen4")
        gen4_x8 = raw_bandwidth_gbps(8, "Gen4")
        assert gen4_x8 == pytest.approx(gen4_x16 * 0.5)

    def test_gen5_is_double_gen4_raw(self):
        gen4 = raw_bandwidth_gbps(16, "Gen4")
        gen5 = raw_bandwidth_gbps(16, "Gen5")
        assert gen5 == pytest.approx(gen4 * 2.0)

    def test_raw_protocol_effective_are_distinct(self):
        config = PCIeConfig(
            generation=PCIeGeneration.GEN5,
            width=PCIeWidth.X16,
            protocol_efficiency=0.9,
        )
        breakdown = compute_pcie_bandwidth(config)
        assert breakdown.raw_GBps > breakdown.protocol_GBps > breakdown.effective_GBps
        assert breakdown.labeled_values()["RAW"] == breakdown.raw_GBps
        assert breakdown.labeled_values()["PROTOCOL"] == breakdown.protocol_GBps
        assert breakdown.labeled_values()["EFFECTIVE"] == breakdown.effective_GBps

    def test_gen3_known_bandwidth(self):
        # x16 Gen3: 16 * 8 / 8 = 16 GB/s RAW per direction
        assert raw_bandwidth_gbps(16, "Gen3") == pytest.approx(16.0)
        protocol = protocol_bandwidth_gbps(16, "Gen3")
        assert protocol == pytest.approx(16.0 * (128 / 130))

    def test_gen6_flit_overhead_reduces_protocol_bandwidth(self):
        raw = raw_bandwidth_gbps(16, "Gen6")
        protocol = protocol_bandwidth_gbps(16, "Gen6")
        assert protocol < raw
        assert protocol == pytest.approx(raw * GENERATION_TABLE["Gen6"].encoding_efficiency)

    def test_effective_matches_breakdown(self):
        config = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X8)
        assert compute_effective_bandwidth_gbps(config) == pytest.approx(
            compute_pcie_bandwidth(config).effective_GBps
        )

    def test_lane_count_helper(self):
        assert lane_count(PCIeWidth.X4) == 4


class TestPCIeLatency:
    def test_serialization_scales_with_payload(self):
        config = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X16)
        small = compute_transfer_time_us(config, 4096)
        large = compute_transfer_time_us(config, 4096 * 16)
        assert large.serialization_time_us > small.serialization_time_us
        assert large.transfer_time_us > small.transfer_time_us

    def test_read_exceeds_write_for_same_payload(self):
        config = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X16)
        payload = 65536
        read = compute_read_latency_us(config, payload)
        write = compute_write_latency_us(config, payload)
        assert read.transfer_time_us > write.transfer_time_us
        assert read.label == "ANALYTICAL ESTIMATE"

    def test_switch_hops_add_latency(self):
        base = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X16, switch_hops=0)
        switched = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X16, switch_hops=2)
        assert compute_transfer_time_us(switched, 0).hop_latency_us > compute_transfer_time_us(base, 0).hop_latency_us

    def test_congestion_penalty_positive_when_factor_above_one(self):
        config = PCIeConfig(
            generation=PCIeGeneration.GEN4,
            width=PCIeWidth.X16,
            congestion_factor=1.5,
        )
        result = compute_transfer_time_us(config, 1024)
        assert result.congestion_penalty_us > 0.0

    def test_zero_payload_has_no_serialization(self):
        config = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X16)
        result = compute_transfer_time_us(config, 0)
        assert result.serialization_time_us == 0.0
