"""Hotness tracking model for KV metadata blocks."""

from __future__ import annotations

import math
import random

from model.types import HotnessBlock, HotnessModelType


def compute_hotness(
    access_count: int,
    recency: float,
    reference_count: int,
    model_type: HotnessModelType,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    decay_lambda: float = 0.01,
) -> float:
    """Compute hotness score for a single block."""
    if model_type == HotnessModelType.WEIGHTED_LINEAR:
        return alpha * access_count + beta * recency + gamma * reference_count

    if model_type == HotnessModelType.LRU_LIKE:
        # Lower recency (more recent) => higher hotness
        return 1.0 / (1.0 + recency)

    if model_type == HotnessModelType.LFU_LIKE:
        return float(access_count)

    if model_type == HotnessModelType.RECENCY_WEIGHTED:
        return recency * access_count

    if model_type == HotnessModelType.EXP_DECAY:
        return access_count * math.exp(-decay_lambda * recency)

    return alpha * access_count + beta * recency + gamma * reference_count


def classify_hotness(hotness: float, threshold: float) -> str:
    return "hot" if hotness >= threshold else "cold"


def generate_hotness_distribution(
    num_blocks: int,
    model_type: HotnessModelType,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.3,
    decay_lambda: float = 0.01,
    hot_fraction: float = 0.2,
    seed: int = 42,
) -> list[HotnessBlock]:
    """
    Generate a deterministic synthetic hotness distribution.

    Uses a seeded RNG so results are reproducible.
    """
    rng = random.Random(seed)
    blocks: list[HotnessBlock] = []

    for i in range(num_blocks):
        # Zipf-like access pattern: few hot blocks, many cold
        rank = i + 1
        access_count = max(1, int(1000 / (rank ** 0.8) + rng.randint(0, 10)))
        recency = rng.uniform(0, 1000)
        reference_count = max(0, int(access_count * rng.uniform(0.1, 0.5)))

        hotness = compute_hotness(
            access_count, recency, reference_count, model_type,
            alpha, beta, gamma, decay_lambda,
        )
        blocks.append(
            HotnessBlock(
                block_id=i,
                access_count=access_count,
                recency=recency,
                reference_count=reference_count,
                hotness=hotness,
                classification="cold",
            )
        )

    # Classify based on percentile threshold
    sorted_hotness = sorted(b.hotness for b in blocks)
    threshold_idx = int(len(sorted_hotness) * (1.0 - hot_fraction))
    threshold_idx = min(threshold_idx, len(sorted_hotness) - 1)
    threshold = sorted_hotness[threshold_idx]

    for block in blocks:
        block.classification = classify_hotness(block.hotness, threshold)

    return blocks


def hotness_update_cycles(complexity: float, pipeline_depth: int) -> float:
    """Estimate RTL cycles per hotness update."""
    base_cycles = 4  # ASSUMPTION: base hotness arithmetic
    return base_cycles * complexity + pipeline_depth
