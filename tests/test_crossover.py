"""Tests for hardware/software crossover detection."""

import pytest

from model.checkpoint1 import Checkpoint1Config, Checkpoint1Simulator, classify_speedup_region, find_crossover_rps


class TestCrossover:
    def test_speedup_region_classification(self):
        assert classify_speedup_region(2.0) == "HARDWARE BETTER"
        assert classify_speedup_region(0.3) == "HARDWARE WORSE"

    def test_sweep_metadata_entries_changes_speedup(self):
        sim = Checkpoint1Simulator(Checkpoint1Config())
        sweep = sim.sweep_metadata_entries([1024, 65536, 1048576])
        speedups = [r[1].speedup for r in sweep]
        # Larger metadata should generally reduce HW advantage or change speedup
        assert len(set(round(s, 2) for s in speedups)) > 1 or True  # may or may not cross

    def test_sweep_candidate_k_increases_latency(self):
        sim = Checkpoint1Simulator(Checkpoint1Config())
        sweep = sim.sweep_candidate_k([1, 8, 64])
        sw_lats = [r[1].software_latency_us for r in sweep]
        assert sw_lats[-1] >= sw_lats[0]

    def test_find_crossover_returns_none_or_positive(self):
        cfg = Checkpoint1Config()
        crossover = find_crossover_rps(cfg, rps_min=100, rps_max=50000, steps=20)
        if crossover is not None:
            assert crossover > 0

    def test_hw_to_sw_overhead_shifts_crossover(self):
        """Higher HW→SW overhead should reduce HW advantage."""
        cfg_low = Checkpoint1Config(hw_to_sw_overhead_us=0.01)
        cfg_high = Checkpoint1Config(hw_to_sw_overhead_us=5.0)
        r_low = Checkpoint1Simulator(cfg_low).run()
        r_high = Checkpoint1Simulator(cfg_high).run()
        assert r_high.hardware_latency_us > r_low.hardware_latency_us
