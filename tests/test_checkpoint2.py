"""Tests for checkpoint 2 prefill GPU scaling."""

import pytest

from model.checkpoint1 import Checkpoint1Config, Checkpoint1Simulator
from model.checkpoint2 import Checkpoint2Simulator, PrefillScalingSweep
from model.scaling import apply_prefill_scaling, metadata_load_factor


class TestPrefillScaling:
    def test_load_factor_increases_with_gpus(self):
        assert metadata_load_factor(1) == 1.0
        assert metadata_load_factor(16) > metadata_load_factor(2)

    def test_apply_scaling_increases_metadata_entries(self):
        base = Checkpoint1Config(prefill_gpus=4, num_kv_blocks=1000)
        scaled = apply_prefill_scaling(base)
        assert scaled.num_kv_blocks == 4000

    def test_apply_scaling_noop_for_one_gpu(self):
        base = Checkpoint1Config(prefill_gpus=1)
        scaled = apply_prefill_scaling(base)
        assert scaled.num_kv_blocks == base.num_kv_blocks

    def test_candidate_k_capped_by_gpus(self):
        base = Checkpoint1Config(prefill_gpus=2, candidate_count_k=8)
        scaled = apply_prefill_scaling(base)
        assert scaled.candidate_count_k == 2


class TestCheckpoint2Simulator:
    def test_valid_gpu_counts(self):
        for n in [2, 4, 8, 16]:
            cfg = Checkpoint1Config(prefill_gpus=n)
            result = Checkpoint2Simulator(cfg).run()
            assert result.metadata_ops_per_sec > 0

    def test_invalid_gpu_count_raises(self):
        with pytest.raises(ValueError):
            Checkpoint2Simulator(Checkpoint1Config(prefill_gpus=1))

    def test_invalid_gpu_count_32_raises(self):
        with pytest.raises(ValueError):
            Checkpoint2Simulator(Checkpoint1Config(prefill_gpus=32))

    def test_utilization_increases_with_gpus(self):
        base = Checkpoint1Config(requests_per_sec=5000, metadata_ops_per_request=16)
        sweep = Checkpoint2Simulator(
            Checkpoint1Config(**{**base.to_dict(), "prefill_gpus": 2})
        ).sweep_prefill_gpus()
        utils = sweep.rtl_utilizations
        assert utils[-1] >= utils[0]

    def test_sweep_returns_all_points(self):
        cfg = Checkpoint1Config(prefill_gpus=4)
        sweep = Checkpoint2Simulator(cfg).sweep_prefill_gpus()
        assert isinstance(sweep, PrefillScalingSweep)
        assert len(sweep.points) == 4
        assert sweep.gpu_counts == [2, 4, 8, 16]

    def test_latency_increases_with_gpus(self):
        cfg = Checkpoint1Config(prefill_gpus=2, requests_per_sec=8000)
        sweep = Checkpoint2Simulator(cfg).sweep_prefill_gpus()
        assert sweep.hardware_latencies[-1] >= sweep.hardware_latencies[0]

    def test_metadata_ops_increase_with_gpus(self):
        cfg = Checkpoint1Config(prefill_gpus=2)
        sweep = Checkpoint2Simulator(cfg).sweep_prefill_gpus()
        assert sweep.metadata_ops_per_sec[-1] > sweep.metadata_ops_per_sec[0]

    def test_cp1_unchanged_at_one_gpu(self):
        cfg = Checkpoint1Config(prefill_gpus=1)
        r1 = Checkpoint1Simulator(cfg).run()
        scaled = apply_prefill_scaling(cfg)
        r2 = Checkpoint1Simulator(scaled).run()
        assert r1.software_latency_us == r2.software_latency_us
