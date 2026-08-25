"""Architecture topology presets (Part 11 + worker pool / HBM layouts)."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from model.interconnect.infiniband import InfiniBandConfig, infiniband_config_from_dict
from model.interconnect.nvlink import NVLinkConfig, nvlink_config_from_dict
from model.interconnect.pcie import PCIeConfig
from model.topology.graph import Topology
from model.topology.types import Node, NodeType


def _default_pcie_config() -> PCIeConfig:
    return PCIeConfig()


def _default_nvlink_config() -> NVLinkConfig:
    return nvlink_config_from_dict(None)


def _default_infiniband_config() -> InfiniBandConfig:
    return infiniband_config_from_dict(None)


def _pcie(topology: Topology, edge_id: str, src: str, dst: str, config: PCIeConfig | None = None) -> None:
    topology.connect_pcie(edge_id, src, dst, config or _default_pcie_config())


def _nvlink(
    topology: Topology,
    edge_id: str,
    src: str,
    dst: str,
    config: NVLinkConfig | None = None,
) -> None:
    topology.connect_nvlink(edge_id, src, dst, config or _default_nvlink_config())


def _infiniband(
    topology: Topology,
    edge_id: str,
    src: str,
    dst: str,
    config: InfiniBandConfig | None = None,
) -> None:
    topology.connect_infiniband(edge_id, src, dst, config or _default_infiniband_config())


def _add_node(
    topology: Topology,
    node: Node,
    *,
    config: dict | None = None,
) -> None:
    from model.components.registry import default_node_config_dict

    base = default_node_config_dict(node.type.value)
    if config:
        base.update(config)
    topology.add_node(node, config=base)


PresetResult = tuple[Topology, dict[str, tuple[float, float]]]


def preset_architecture_a(pcie: PCIeConfig | None = None) -> PresetResult:
    """CPU — PCIe — FPGA."""
    topology = Topology()
    positions = {"cpu": (0.0, 120.0), "fpga": (0.0, -120.0)}
    _add_node(topology, Node(id="cpu", type=NodeType.CPU, name="CPU"))
    _add_node(topology, Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    _pcie(topology, "cpu_fpga", "cpu", "fpga", pcie)
    return topology, positions


def preset_architecture_b(pcie: PCIeConfig | None = None) -> PresetResult:
    """CPU — RC — Switch — FPGA."""
    topology = Topology()
    positions = {
        "cpu": (0.0, 220.0),
        "rc": (0.0, 100.0),
        "switch": (0.0, -20.0),
        "fpga": (0.0, -200.0),
    }
    cfg = pcie or _default_pcie_config()
    sw_cfg = replace(cfg, switch_hops=max(cfg.switch_hops, 1))

    _add_node(topology, Node(id="cpu", type=NodeType.CPU, name="CPU"))
    _add_node(topology, Node(id="rc", type=NodeType.PCIE_ROOT_COMPLEX, name="PCIe Root Complex"))
    _add_node(topology, Node(id="switch", type=NodeType.PCIE_SWITCH, name="PCIe Switch"))
    _add_node(topology, Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    _pcie(topology, "cpu_rc", "cpu", "rc", cfg)
    _pcie(topology, "rc_switch", "rc", "switch", sw_cfg)
    _pcie(topology, "switch_fpga", "switch", "fpga", sw_cfg)
    return topology, positions


def preset_architecture_c(pcie: PCIeConfig | None = None) -> PresetResult:
    """CPU — RC — GPU + FPGA."""
    topology = Topology()
    positions = {
        "cpu": (0.0, 200.0),
        "rc": (0.0, 60.0),
        "gpu": (-180.0, -120.0),
        "fpga": (180.0, -120.0),
    }
    cfg = pcie or _default_pcie_config()
    _add_node(topology, Node(id="cpu", type=NodeType.CPU, name="CPU"))
    _add_node(topology, Node(id="rc", type=NodeType.PCIE_ROOT_COMPLEX, name="PCIe Root Complex"))
    _add_node(topology, Node(id="gpu", type=NodeType.GPU, name="GPU"))
    _add_node(topology, Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    _pcie(topology, "cpu_rc", "cpu", "rc", cfg)
    _pcie(topology, "rc_gpu", "rc", "gpu", cfg)
    _pcie(topology, "rc_fpga", "rc", "fpga", cfg)
    return topology, positions


def preset_architecture_d(pcie: PCIeConfig | None = None) -> PresetResult:
    """Prefill GPU pool — FPGA — Decode GPU pool."""
    topology = Topology()
    positions = {
        "prefill_pool": (0.0, 200.0),
        "fpga": (0.0, 0.0),
        "decode_pool": (0.0, -200.0),
    }
    cfg = pcie or _default_pcie_config()
    _add_node(
        topology,
        Node(id="prefill_pool", type=NodeType.PREFILL_POOL, name="Prefill Worker Pool"),
        config={"worker_count": 4},
    )
    _add_node(topology, Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    _add_node(
        topology,
        Node(id="decode_pool", type=NodeType.DECODE_POOL, name="Decode Worker Pool"),
        config={"worker_count": 4},
    )
    _pcie(topology, "prefill_fpga", "prefill_pool", "fpga", cfg)
    _pcie(topology, "fpga_decode", "fpga", "decode_pool", cfg)
    return topology, positions


def preset_architecture_e() -> PresetResult:
    """Prefill GPUs — InfiniBand — Decode GPUs."""
    topology = Topology()
    positions = {
        "prefill_pool": (-220.0, 80.0),
        "decode_pool": (220.0, -80.0),
    }
    _add_node(
        topology,
        Node(id="prefill_pool", type=NodeType.PREFILL_POOL, name="Prefill Worker Pool"),
        config={"worker_count": 4},
    )
    _add_node(
        topology,
        Node(id="decode_pool", type=NodeType.DECODE_POOL, name="Decode Worker Pool"),
        config={"worker_count": 4},
    )
    _infiniband(topology, "prefill_decode", "prefill_pool", "decode_pool")
    return topology, positions


def preset_architecture_f(pcie: PCIeConfig | None = None) -> PresetResult:
    """
    Prefill pool — NVLink — GPU — PCIe — FPGA.

    High-level nodes only; internal NVLink mesh is not modeled.
    """
    topology = Topology()
    positions = {
        "prefill_pool": (0.0, 220.0),
        "gpu": (0.0, 40.0),
        "fpga": (0.0, -160.0),
    }
    cfg = pcie or _default_pcie_config()
    _add_node(
        topology,
        Node(id="prefill_pool", type=NodeType.PREFILL_POOL, name="Prefill Worker Pool"),
        config={"worker_count": 2},
    )
    _add_node(topology, Node(id="gpu", type=NodeType.GPU, name="GPU"))
    _add_node(topology, Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    _nvlink(topology, "prefill_gpu", "prefill_pool", "gpu")
    _pcie(topology, "gpu_fpga", "gpu", "fpga", cfg)
    return topology, positions


def preset_worker_pools_hbm(pcie: PCIeConfig | None = None) -> PresetResult:
    """
    Prefill worker pool → Prefill HBM cache
    Decode worker pool → Decode HBM cache

    No internal NVLink/NVSwitch inside worker pools.
    """
    topology = Topology()
    positions = {
        "prefill_pool": (-240.0, 120.0),
        "prefill_hbm": (-240.0, -40.0),
        "decode_pool": (240.0, 120.0),
        "decode_hbm": (240.0, -40.0),
    }
    cfg = pcie or _default_pcie_config()

    _add_node(
        topology,
        Node(id="prefill_pool", type=NodeType.PREFILL_POOL, name="Prefill Worker Pool"),
        config={"worker_count": 4, "attached_hbm_node_id": "prefill_hbm"},
    )
    _add_node(
        topology,
        Node(id="prefill_hbm", type=NodeType.HBM_MEMORY, name="Prefill HBM Cache"),
    )
    _add_node(
        topology,
        Node(id="decode_pool", type=NodeType.DECODE_POOL, name="Decode Worker Pool"),
        config={"worker_count": 4, "attached_hbm_node_id": "decode_hbm"},
    )
    _add_node(
        topology,
        Node(id="decode_hbm", type=NodeType.HBM_MEMORY, name="Decode HBM Cache"),
    )

    _pcie(topology, "prefill_pool_hbm", "prefill_pool", "prefill_hbm", cfg)
    _pcie(topology, "decode_pool_hbm", "decode_pool", "decode_hbm", cfg)
    return topology, positions


def preset_worker_pools_hbm_fpga(pcie: PCIeConfig | None = None) -> PresetResult:
    """Worker pools + HBM caches + centralized KVBM FPGA."""
    topology = Topology()
    positions = {
        "prefill_pool": (-280.0, 160.0),
        "prefill_hbm": (-280.0, 20.0),
        "fpga": (0.0, 0.0),
        "decode_pool": (280.0, 160.0),
        "decode_hbm": (280.0, 20.0),
    }
    cfg = pcie or _default_pcie_config()

    _add_node(
        topology,
        Node(id="prefill_pool", type=NodeType.PREFILL_POOL, name="Prefill Worker Pool"),
        config={"worker_count": 4, "attached_hbm_node_id": "prefill_hbm"},
    )
    _add_node(
        topology,
        Node(id="prefill_hbm", type=NodeType.HBM_MEMORY, name="Prefill HBM Cache"),
    )
    _add_node(topology, Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    _add_node(
        topology,
        Node(id="decode_pool", type=NodeType.DECODE_POOL, name="Decode Worker Pool"),
        config={"worker_count": 4, "attached_hbm_node_id": "decode_hbm"},
    )
    _add_node(
        topology,
        Node(id="decode_hbm", type=NodeType.HBM_MEMORY, name="Decode HBM Cache"),
    )

    _pcie(topology, "prefill_pool_hbm", "prefill_pool", "prefill_hbm", cfg)
    _pcie(topology, "decode_pool_hbm", "decode_pool", "decode_hbm", cfg)
    _pcie(topology, "prefill_fpga", "prefill_pool", "fpga", cfg)
    _pcie(topology, "fpga_decode", "fpga", "decode_pool", cfg)
    return topology, positions


ARCHITECTURE_PRESETS: dict[str, Callable[..., PresetResult]] = {
    "Architecture A — CPU ↔ FPGA (direct PCIe)": preset_architecture_a,
    "Architecture B — CPU → RC → Switch → FPGA": preset_architecture_b,
    "Architecture C — CPU → RC → GPU + FPGA": preset_architecture_c,
    "Architecture D — Prefill → FPGA → Decode": preset_architecture_d,
    "Architecture E — Prefill ↔ Decode (InfiniBand)": preset_architecture_e,
    "Architecture F — Prefill → NVLink → GPU → PCIe → FPGA": preset_architecture_f,
    "Worker Pools + HBM Caches (PCIe only)": preset_worker_pools_hbm,
    "Worker Pools + HBM + FPGA": preset_worker_pools_hbm_fpga,
}


def build_preset(name: str, pcie: PCIeConfig | None = None) -> PresetResult:
    factory = ARCHITECTURE_PRESETS[name]
    if name.startswith("Architecture E"):
        return factory()
    return factory(pcie)
