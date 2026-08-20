"""Tests for checkpoint 3 high-scale prefill GPU scaling."""

import pytest

from model.checkpoint1 import Checkpoint1Config
from model.checkpoint2 import Checkpoint2Simulator
from model.checkpoint3 import Checkpoint3Simulator, PrefillScalingSweep
from model.scaling import PREFILL_GPU_OPTIONS_CP3


class TestCheckpoint3Simulator:
    def test_valid_gpu_counts(self):
        for n in PREFILL_GPU_OPTIONS_CP3:
            cfg = Checkpoint1Config(prefill_gpus=n)
            result = Checkpoint3Simulator(cfg).run()
            assert result.metadata_ops_per_sec > 0

    def test_invalid_gpu_count_raises(self):
        with pytest.raises(ValueError):
            Checkpoint3Simulator(Checkpoint1Config(prefill_gpus=16))

    def test_invalid_gpu_count_2_raises(self):
        with pytest.raises(ValueError):
            Checkpoint3Simulator(Checkpoint1Config(prefill_gpus=2))

    def test_sweep_returns_all_points(self):
        cfg = Checkpoint1Config(prefill_gpus=64)
        sweep = Checkpoint3Simulator(cfg).sweep_prefill_gpus()
        assert isinstance(sweep, PrefillScalingSweep)
        assert len(sweep.points) == 3
        assert sweep.gpu_counts == [32, 64, 128]

    def test_utilization_increases_with_gpus(self):
        cfg = Checkpoint1Config(prefill_gpus=32, requests_per_sec=5000, metadata_ops_per_request=16)
        sweep = Checkpoint3Simulator(cfg).sweep_prefill_gpus()
        assert sweep.rtl_utilizations[-1] >= sweep.rtl_utilizations[0]

    def test_metadata_ops_increase_with_gpus(self):
        cfg = Checkpoint1Config(prefill_gpus=32)
        sweep = Checkpoint3Simulator(cfg).sweep_prefill_gpus()
        assert sweep.metadata_ops_per_sec[-1] > sweep.metadata_ops_per_sec[0]

    def test_high_gpu_count_increases_utilization(self):
        base = Checkpoint1Config(requests_per_sec=20000, metadata_ops_per_request=32)
        r32 = Checkpoint3Simulator(Checkpoint1Config(**{**base.to_dict(), "prefill_gpus": 32})).run()
        r128 = Checkpoint3Simulator(Checkpoint1Config(**{**base.to_dict(), "prefill_gpus": 128})).run()
        assert r128.rtl_utilization > r32.rtl_utilization

    def test_cp2_rejects_cp3_gpu_counts(self):
        with pytest.raises(ValueError):
            Checkpoint2Simulator(Checkpoint1Config(prefill_gpus=64))

    def test_cp3_rejects_cp2_gpu_counts(self):
        with pytest.raises(ValueError):
            Checkpoint3Simulator(Checkpoint1Config(prefill_gpus=8))

    def test_requires_one_rtl_engine(self):
        with pytest.raises(ValueError):
            Checkpoint3Simulator(Checkpoint1Config(prefill_gpus=64, rtl_engines=2))
