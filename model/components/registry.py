"""Component config registry, parsing, and preset loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from model.components.cpu import DEFAULT_CPU, CPUConfig, cpu_config_from_dict
from model.components.decode_pool import (
    DEFAULT_DECODE_POOL,
    DecodePoolConfig,
    decode_pool_config_from_dict,
)
from model.components.fpga import DEFAULT_FPGA, FPGAConfig, fpga_config_from_dict
from model.components.gpu import DEFAULT_GPU, GPUConfig, gpu_config_from_dict
from model.components.hbm_memory import (
    DEFAULT_HBM_MEMORY,
    HBMMemoryConfig,
    hbm_memory_config_from_dict,
)
from model.components.prefill_pool import (
    DEFAULT_PREFILL_POOL,
    PrefillPoolConfig,
    prefill_pool_config_from_dict,
)
from model.components.pcie_root_complex import (
    DEFAULT_PCIE_ROOT_COMPLEX,
    PCIeRootComplexConfig,
    pcie_root_complex_config_from_dict,
)
from model.components.pcie_switch import (
    DEFAULT_PCIE_SWITCH,
    PCIeSwitchConfig,
    pcie_switch_config_from_dict,
)
from model.interconnect.infiniband import (
    DEFAULT_INFINIBAND,
    InfiniBandConfig,
    infiniband_config_from_dict,
    infiniband_config_to_dict,
)
from model.interconnect.nvlink import (
    DEFAULT_NVLINK,
    NVLinkConfig,
    nvlink_config_from_dict,
    nvlink_config_to_dict,
)
from model.topology.types import NodeType, normalize_node_type

COMPONENTS_DIR = Path(__file__).resolve().parents[2] / "configs" / "components"

NODE_CONFIG_DEFAULTS: dict[str, dict[str, Any]] = {
    NodeType.CPU.value: DEFAULT_CPU,
    NodeType.GPU.value: DEFAULT_GPU,
    NodeType.FPGA.value: DEFAULT_FPGA,
    NodeType.PREFILL_POOL.value: DEFAULT_PREFILL_POOL,
    NodeType.DECODE_POOL.value: DEFAULT_DECODE_POOL,
    NodeType.HBM_MEMORY.value: DEFAULT_HBM_MEMORY,
    NodeType.PCIE_ROOT_COMPLEX.value: DEFAULT_PCIE_ROOT_COMPLEX,
    NodeType.PCIE_SWITCH.value: DEFAULT_PCIE_SWITCH,
}

NODE_CONFIG_PARSERS: dict[str, Callable[[dict[str, Any] | None], Any]] = {
    NodeType.CPU.value: cpu_config_from_dict,
    NodeType.GPU.value: gpu_config_from_dict,
    NodeType.FPGA.value: fpga_config_from_dict,
    NodeType.PREFILL_POOL.value: prefill_pool_config_from_dict,
    NodeType.DECODE_POOL.value: decode_pool_config_from_dict,
    NodeType.HBM_MEMORY.value: hbm_memory_config_from_dict,
    NodeType.PCIE_ROOT_COMPLEX.value: pcie_root_complex_config_from_dict,
    NodeType.PCIE_SWITCH.value: pcie_switch_config_from_dict,
}

EDGE_CONFIG_DEFAULTS: dict[str, dict[str, Any]] = {
    "NVLink": DEFAULT_NVLINK,
    "InfiniBand": DEFAULT_INFINIBAND,
}

EDGE_CONFIG_PARSERS: dict[str, Callable[[dict[str, Any] | None], Any]] = {
    "NVLink": nvlink_config_from_dict,
    "InfiniBand": infiniband_config_from_dict,
}

EDGE_CONFIG_SERIALIZERS: dict[str, Callable[[Any], dict[str, Any]]] = {
    "NVLink": nvlink_config_to_dict,
    "InfiniBand": infiniband_config_to_dict,
}

_PRESET_CACHE: dict[str, dict[str, dict[str, Any]]] | None = None


def _load_preset_cache() -> dict[str, dict[str, dict[str, Any]]]:
    global _PRESET_CACHE
    if _PRESET_CACHE is not None:
        return _PRESET_CACHE

    cache: dict[str, dict[str, dict[str, Any]]] = {}
    if COMPONENTS_DIR.exists():
        for path in sorted(COMPONENTS_DIR.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            kind = str(raw.get("kind", path.stem.upper()))
            presets = raw.get("presets", {}) or {}
            cache[kind] = {str(key): dict(value) for key, value in presets.items()}
    _PRESET_CACHE = cache
    return cache


def list_presets(kind: str) -> list[str]:
    return sorted(_load_preset_cache().get(kind, {}).keys())


def get_preset(kind: str, preset_id: str) -> dict[str, Any]:
    presets = _load_preset_cache().get(kind, {})
    if preset_id not in presets:
        available = ", ".join(sorted(presets.keys())) or "(none)"
        raise KeyError(f"Unknown preset '{preset_id}' for kind '{kind}'. Available: {available}")
    return dict(presets[preset_id])


def default_node_config_dict(node_type: str) -> dict[str, Any]:
    normalized = normalize_node_type(node_type)
    return dict(NODE_CONFIG_DEFAULTS[normalized])


def parse_node_config(node_type: str, data: dict[str, Any] | None) -> Any:
    normalized = normalize_node_type(node_type)
    if normalized not in NODE_CONFIG_PARSERS:
        raise ValueError(f"No component config parser for node type '{node_type}'.")
    return NODE_CONFIG_PARSERS[normalized](data)


def node_config_to_dict(node_type: str, config: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(config)


def parse_edge_config(interconnect: str, data: dict[str, Any] | None) -> Any:
    if interconnect not in EDGE_CONFIG_PARSERS:
        raise ValueError(f"No edge config parser for interconnect '{interconnect}'.")
    return EDGE_CONFIG_PARSERS[interconnect](data)


def edge_config_to_dict(interconnect: str, config: Any) -> dict[str, Any]:
    if interconnect not in EDGE_CONFIG_SERIALIZERS:
        raise ValueError(f"No edge config serializer for interconnect '{interconnect}'.")
    return EDGE_CONFIG_SERIALIZERS[interconnect](config)


def attach_node_config(node_type: str, config_data: dict[str, Any] | None) -> dict[str, Any]:
    config = parse_node_config(node_type, config_data)
    return node_config_to_dict(node_type, config)


__all__ = [
    "CPUConfig",
    "DecodePoolConfig",
    "FPGAConfig",
    "GPUConfig",
    "HBMMemoryConfig",
    "InfiniBandConfig",
    "NVLinkConfig",
    "PCIeRootComplexConfig",
    "PCIeSwitchConfig",
    "PrefillPoolConfig",
    "attach_node_config",
    "default_node_config_dict",
    "edge_config_to_dict",
    "get_preset",
    "list_presets",
    "node_config_to_dict",
    "parse_edge_config",
    "parse_node_config",
]
