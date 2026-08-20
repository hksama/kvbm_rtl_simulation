"""Tests for throughput calculations."""

import pytest

from model.checkpoint1 import Checkpoint1Config, Checkpoint1Simulator
from model.rtl import compute_hardware_throughput
from model.software import compute_software_throughput


class TestSoftwareThroughput:
    def test_throughput_positive(self):
        t = compute_software_throughput(
            cpu_cores=8,
            parallelism=4,
            per_op_latency_us=1.0,
            metadata_ops_per_request=16,
            requests_per_sec=1000,
        )
        assert t.theoretical_ops_per_sec > 0
        assert t.sustainable_ops_per_sec > 0

    def test_utilization_capped(self):
        t = compute_software_throughput(
            cpu_cores=2,
            parallelism=1,
            per_op_latency_us=10.0,
            metadata_ops_per_request=64,
            requests_per_sec=100000,
        )
        assert t.utilization <= 1.0

    def test_more_cores_higher_throughput(self):
        low = compute_software_throughput(2, 2, 1.0, 16, 1000)
        high = compute_software_throughput(16, 8, 1.0, 16, 1000)
        assert high.theoretical_ops_per_sec > low.theoretical_ops_per_sec


class TestHardwareThroughput:
    def test_throughput_positive(self):
        t = compute_hardware_throughput(
            clock_freq_mhz=500,
            pipeline_depth=8,
            lookup_ports=2,
            parallel_ops=4,
            scoring_units=4,
            candidate_comparison_units=8,
            metadata_memory_bandwidth_gbps=32,
            metadata_bytes_per_block=64,
            metadata_ops_per_request=16,
            requests_per_sec=1000,
            cycles_per_operation=10,
            queue_depth=256,
        )
        assert t.theoretical_ops_per_sec > 0

    def test_higher_clock_higher_throughput(self):
        common = dict(
            pipeline_depth=8, lookup_ports=2, parallel_ops=4, scoring_units=4,
            candidate_comparison_units=8, metadata_memory_bandwidth_gbps=32,
            metadata_bytes_per_block=64, metadata_ops_per_request=16,
            requests_per_sec=1000, cycles_per_operation=10, queue_depth=256,
        )
        slow = compute_hardware_throughput(clock_freq_mhz=250, **common)
        fast = compute_hardware_throughput(clock_freq_mhz=1000, **common)
        assert fast.theoretical_ops_per_sec > slow.theoretical_ops_per_sec


class TestCheckpoint1Throughput:
    def test_sweep_rps_monotonic_utilization(self):
        sim = Checkpoint1Simulator(Checkpoint1Config())
        sweep = sim.sweep_rps([500, 2000, 8000, 20000])
        utils = [r[1].rtl_utilization for r in sweep]
        assert utils[-1] >= utils[0]

    def test_achieved_rps_limited_by_capacity(self):
        cfg = Checkpoint1Config(
            requests_per_sec=50_000_000,
            metadata_ops_per_request=64,
            clock_freq_mhz=250,
            parallel_metadata_ops=1,
            lookup_ports=1,
            scoring_units=1,
            metadata_memory_bandwidth_gbps=1.0,
        )
        result = Checkpoint1Simulator(cfg).run()
        assert result.hardware_throughput.achieved_rps < cfg.requests_per_sec
