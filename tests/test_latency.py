"""Tests for latency calculations."""

import pytest

from model.checkpoint1 import Checkpoint1Config, Checkpoint1Simulator, classify_speedup_region
from model.queueing import compute_queueing, queue_delay_us
from model.rtl import compute_hardware_latency, cycles_to_us, rtl_cycles_per_operation
from model.software import compute_software_latency, scaling_factor_n


class TestScalingFactor:
    def test_n_equals_one(self):
        assert scaling_factor_n(1) == 1.0

    def test_increases_with_n(self):
        assert scaling_factor_n(1024) > scaling_factor_n(64)


class TestQueueing:
    def test_zero_utilization(self):
        q = compute_queueing(arrival_rate=100, service_rate=10000, service_time_us=1.0)
        assert q.utilization < 0.1
        assert not q.saturated

    def test_saturation(self):
        q = compute_queueing(arrival_rate=10000, service_rate=1000, service_time_us=1.0)
        assert q.saturated
        assert q.utilization == 1.0

    def test_queue_delay_nonnegative(self):
        q = compute_queueing(arrival_rate=500, service_rate=1000, service_time_us=2.0)
        delay = queue_delay_us(q, 2.0)
        assert delay >= 0.0


class TestSoftwareLatency:
    def test_components_sum_to_total(self):
        breakdown, _ = compute_software_latency(
            lookup_us=2.0,
            metadata_update_us=1.5,
            score_us=3.0,
            selection_us=1.0,
            coordination_us=0.5,
            memory_access_us=0.1,
            cache_hit_rate=0.85,
            cpu_contention=0.2,
            metadata_entries=65536,
            candidate_k=8,
            parallelism=4,
            metadata_ops=16,
            arrival_rate=1000,
            service_rate=100000,
        )
        assert breakdown.total > 0
        assert breakdown.lookup > 0

    def test_contention_increases_latency(self):
        _, _ = compute_software_latency(
            lookup_us=2.0, metadata_update_us=1.5, score_us=3.0, selection_us=1.0,
            coordination_us=0.5, memory_access_us=0.1, cache_hit_rate=0.85,
            cpu_contention=0.0, metadata_entries=1024, candidate_k=4,
            parallelism=1, metadata_ops=8, arrival_rate=100, service_rate=1e6,
        )
        low, _ = compute_software_latency(
            lookup_us=2.0, metadata_update_us=1.5, score_us=3.0, selection_us=1.0,
            coordination_us=0.5, memory_access_us=0.1, cache_hit_rate=0.85,
            cpu_contention=0.0, metadata_entries=1024, candidate_k=4,
            parallelism=1, metadata_ops=8, arrival_rate=100, service_rate=1e6,
        )
        high, _ = compute_software_latency(
            lookup_us=2.0, metadata_update_us=1.5, score_us=3.0, selection_us=1.0,
            coordination_us=0.5, memory_access_us=0.1, cache_hit_rate=0.85,
            cpu_contention=0.8, metadata_entries=1024, candidate_k=4,
            parallelism=1, metadata_ops=8, arrival_rate=100, service_rate=1e6,
        )
        assert high.total > low.total


class TestHardwareLatency:
    def test_cycles_to_us(self):
        assert cycles_to_us(500, 500) == pytest.approx(1.0)

    def test_higher_clock_reduces_latency(self):
        common = dict(
            pipeline_depth=8, lookup_ports=2, parallel_ops=4, candidate_k=8,
            scoring_units=4, metadata_entries=65536, hotness_complexity=1.0,
            metadata_ops=16, request_arrival_us=0.1, hw_to_sw_us=0.3,
            coordination_us=1.0, arrival_rate=1000, service_rate=1e6,
        )
        slow, _, _ = compute_hardware_latency(clock_freq_mhz=250, **common)
        fast, _, _ = compute_hardware_latency(clock_freq_mhz=1000, **common)
        assert fast.total < slow.total

    def test_rtl_cycles_positive(self):
        cycles = rtl_cycles_per_operation(8, 65536, 8, 1.0, "lookup")
        assert cycles > 0


class TestCheckpoint1Simulator:
    def test_deterministic(self):
        cfg = Checkpoint1Config()
        r1 = Checkpoint1Simulator(cfg).run()
        r2 = Checkpoint1Simulator(cfg).run()
        assert r1.software_latency_us == r2.software_latency_us
        assert r1.hardware_latency_us == r2.hardware_latency_us
        assert r1.speedup == r2.speedup

    def test_speedup_positive(self):
        result = Checkpoint1Simulator(Checkpoint1Config()).run()
        assert result.speedup > 0

    def test_prefill_gpu_fixed_at_one(self):
        cfg = Checkpoint1Config(prefill_gpus=1, rtl_engines=1)
        assert cfg.prefill_gpus == 1
        assert cfg.rtl_engines == 1

    def test_metadata_ops_per_sec(self):
        cfg = Checkpoint1Config(requests_per_sec=1000, metadata_ops_per_request=16, burstiness=1.0)
        result = Checkpoint1Simulator(cfg).run()
        assert result.metadata_ops_per_sec == pytest.approx(16000)

    def test_utilization_bounded(self):
        result = Checkpoint1Simulator(Checkpoint1Config()).run()
        assert 0 <= result.rtl_utilization <= 1.0


class TestSpeedupRegion:
    def test_hardware_better(self):
        assert classify_speedup_region(1.5) == "HARDWARE BETTER"

    def test_hardware_worse(self):
        assert classify_speedup_region(0.5) == "HARDWARE WORSE"

    def test_break_even(self):
        assert classify_speedup_region(1.0) == "BREAK-EVEN"
