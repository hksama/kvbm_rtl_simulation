"""Checkpoint 3: High prefill GPU scaling (32, 64, 128 GPUs) with one RTL engine."""

from model.prefill_scaling import PrefillScalingPoint, PrefillScalingSimulator, PrefillScalingSweep
from model.scaling import PREFILL_GPU_OPTIONS_CP3


class Checkpoint3Simulator(PrefillScalingSimulator):
    """Simulator for checkpoint 3 high-scale prefill scaling."""

    PREFILL_GPU_OPTIONS = PREFILL_GPU_OPTIONS_CP3


__all__ = ["Checkpoint3Simulator", "PrefillScalingPoint", "PrefillScalingSweep"]
