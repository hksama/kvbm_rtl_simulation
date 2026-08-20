"""Checkpoint 2: Prefill GPU scaling (2, 4, 8, 16 GPUs) with one RTL engine."""

from model.prefill_scaling import PrefillScalingPoint, PrefillScalingSimulator, PrefillScalingSweep
from model.scaling import PREFILL_GPU_OPTIONS_CP2


class Checkpoint2Simulator(PrefillScalingSimulator):
    """Simulator for checkpoint 2 prefill scaling."""

    PREFILL_GPU_OPTIONS = PREFILL_GPU_OPTIONS_CP2


__all__ = ["Checkpoint2Simulator", "PrefillScalingPoint", "PrefillScalingSweep"]
