"""Typed component configuration models.

Import registry helpers explicitly from ``model.components.registry`` to avoid
import cycles with ``model.interconnect`` edge config modules.
"""

from model.components.cpu import CPUConfig
from model.components.decode_pool import DecodePoolConfig
from model.components.fpga import FPGAConfig
from model.components.gpu import GPUConfig
from model.components.hbm_memory import HBMMemoryConfig
from model.components.prefill_pool import PrefillPoolConfig
from model.components.pcie_root_complex import PCIeRootComplexConfig
from model.components.pcie_switch import PCIeSwitchConfig

__all__ = [
    "CPUConfig",
    "DecodePoolConfig",
    "FPGAConfig",
    "GPUConfig",
    "HBMMemoryConfig",
    "PCIeRootComplexConfig",
    "PCIeSwitchConfig",
    "PrefillPoolConfig",
]
