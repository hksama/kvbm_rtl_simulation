"""Tests for hotness model."""

import pytest

from model.hotness import (
    classify_hotness,
    compute_hotness,
    generate_hotness_distribution,
    hotness_update_cycles,
)
from model.types import HotnessModelType


class TestComputeHotness:
    def test_weighted_linear(self):
        h = compute_hotness(10, 5.0, 3, HotnessModelType.WEIGHTED_LINEAR, alpha=1.0, beta=0.5, gamma=0.3)
        assert h == pytest.approx(10 + 2.5 + 0.9)

    def test_lfu_like(self):
        h = compute_hotness(42, 0, 0, HotnessModelType.LFU_LIKE)
        assert h == 42.0

    def test_exp_decay_decreases_with_age(self):
        young = compute_hotness(100, 1.0, 0, HotnessModelType.EXP_DECAY, decay_lambda=0.1)
        old = compute_hotness(100, 100.0, 0, HotnessModelType.EXP_DECAY, decay_lambda=0.1)
        assert young > old


class TestHotnessDistribution:
    def test_deterministic(self):
        d1 = generate_hotness_distribution(100, HotnessModelType.WEIGHTED_LINEAR, seed=42)
        d2 = generate_hotness_distribution(100, HotnessModelType.WEIGHTED_LINEAR, seed=42)
        assert [b.hotness for b in d1] == [b.hotness for b in d2]

    def test_classification(self):
        dist = generate_hotness_distribution(200, HotnessModelType.WEIGHTED_LINEAR, hot_fraction=0.2)
        hot_count = sum(1 for b in dist if b.classification == "hot")
        assert hot_count > 0
        assert hot_count < len(dist)

    def test_classify_hotness(self):
        assert classify_hotness(10.0, 5.0) == "hot"
        assert classify_hotness(3.0, 5.0) == "cold"


class TestHotnessCycles:
    def test_cycles_increase_with_complexity(self):
        low = hotness_update_cycles(1.0, 8)
        high = hotness_update_cycles(3.0, 8)
        assert high > low

    def test_cycles_positive(self):
        assert hotness_update_cycles(1.0, 4) > 0
