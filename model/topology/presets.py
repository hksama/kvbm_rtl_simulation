"""Architecture topology presets (Part 11 + worker pool / HBM layouts)."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from model.interconnect.pcie import PCIeConfig
from model.topology.graph import Topology
from model.topology.types import Edge, InterconnectType, Node, NodeType


def _default_pcie_config() -> PCIeConfig:
    return PCIeConfig()


def _pcie(topology: Topology, edge_id: str, src: str, dst: str, config: PCIeConfig | None = None) -> None:
    topology.connect_pcie(edge_id, src, dst, config or _default_pcie_config())


def _ethernet(
    topology: Topology,
    edge_id: str,
    src: str,
    dst: str,
    *,
    bandwidth_gbps: float = 100.0,
    latency_us: float = 2.0,
    bidirectional: bool = True,
) -> None:
    topology.add_edge(
        Edge(
            id=edge_id,
            source_id=src,
            dest_id=dst,
            interconnect_type=InterconnectType.ETHERNET,
            bandwidth_GBps=bandwidth_gbps,
            latency_us=latency_us,
            generation="—",
            width="—",
        )
    )
    if bidirectional:
        topology.add_edge(
            Edge(
                id=f"{edge_id}_rev",
                source_id=dst,
                dest_id=src,
                interconnect_type=InterconnectType.ETHERNET,
                bandwidth_GBps=bandwidth_gbps,
                latency_us=latency_us,
                generation="—",
                width="—",
            )
        )


PresetResult = tuple[Topology, dict[str, tuple[float, float]]]


def preset_architecture_a(pcie: PCIeConfig | None = None) -> PresetResult:
    """CPU — PCIe — FPGA."""
    topology = Topology()
    positions = {"cpu": (0.0, 120.0), "fpga": (0.0, -120.0)}
    topology.add_node(Node(id="cpu", type=NodeType.CPU, name="CPU"))
    topology.add_node(Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
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

    topology.add_node(Node(id="cpu", type=NodeType.CPU, name="CPU"))
    topology.add_node(Node(id="rc", type=NodeType.PCIE_ROOT_COMPLEX, name="PCIe Root Complex"))
    topology.add_node(Node(id="switch", type=NodeType.PCIE_SWITCH, name="PCIe Switch"))
    topology.add_node(Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
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
    topology.add_node(Node(id="cpu", type=NodeType.CPU, name="CPU"))
    topology.add_node(Node(id="rc", type=NodeType.PCIE_ROOT_COMPLEX, name="PCIe Root Complex"))
    topology.add_node(Node(id="gpu", type=NodeType.GPU, name="GPU"))
    topology.add_node(Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    _pcie(topology, "cpu_rc", "cpu", "rc", cfg)
    _pcie(topology, "rc_gpu", "rc", "gpu", cfg)
    _pcie(topology, "rc_fpga", "rc", "fpga", cfg)
    return topology, positions


def preset_architecture_d(pcie: PCIeConfig | None = None) -> PresetResult:
    """Prefill GPU pool — KVBM — Decode GPU pool."""
    topology = Topology()
    positions = {
        "prefill_pool": (0.0, 200.0),
        "kvbm": (0.0, 0.0),
        "decode_pool": (0.0, -200.0),
    }
    cfg = pcie or _default_pcie_config()
    topology.add_node(
        Node(
            id="prefill_pool",
            type=NodeType.PREFILL_POOL,
            name="Prefill Worker Pool",
            metadata={"worker_count": 4, "role": "prefill"},
        )
    )
    topology.add_node(Node(id="kvbm", type=NodeType.KVBM_ENGINE, name="KVBM FPGA"))
    topology.add_node(
        Node(
            id="decode_pool",
            type=NodeType.DECODE_POOL,
            name="Decode Worker Pool",
            metadata={"worker_count": 4, "role": "decode"},
        )
    )
    _pcie(topology, "prefill_kvbm", "prefill_pool", "kvbm", cfg)
    _pcie(topology, "kvbm_decode", "kvbm", "decode_pool", cfg)
    return topology, positions


def preset_architecture_e() -> PresetResult:
    """Prefill GPUs — network — Router — network — Decode GPUs."""
    topology = Topology()
    positions = {
        "prefill_pool": (-220.0, 80.0),
        "router": (0.0, 0.0),
        "decode_pool": (220.0, -80.0),
    }
    topology.add_node(
        Node(
            id="prefill_pool",
            type=NodeType.PREFILL_POOL,
            name="Prefill Worker Pool",
            metadata={"worker_count": 4},
        )
    )
    topology.add_node(Node(id="router", type=NodeType.ROUTER, name="Router"))
    topology.add_node(
        Node(
            id="decode_pool",
            type=NodeType.DECODE_POOL,
            name="Decode Worker Pool",
            metadata={"worker_count": 4},
        )
    )
    _ethernet(topology, "prefill_router", "prefill_pool", "router")
    _ethernet(topology, "router_decode", "router", "decode_pool")
    return topology, positions


def preset_architecture_f(pcie: PCIeConfig | None = None) -> PresetResult:
    """
    Prefill pool — NVSwitch — GPU mesh — PCIe — FPGA.

    High-level nodes only; internal NVSwitch / NVLink mesh is not modeled.
    """
    topology = Topology()
    positions = {
        "prefill_pool": (0.0, 220.0),
        "nvswitch": (0.0, 80.0),
        "gpu_a": (-140.0, -40.0),
        "gpu_b": (140.0, -40.0),
        "fpga": (0.0, -200.0),
    }
    cfg = pcie or _default_pcie_config()
    topology.add_node(
        Node(
            id="prefill_pool",
            type=NodeType.PREFILL_POOL,
            name="Prefill Worker Pool",
            metadata={"worker_count": 2},
        )
    )
    topology.add_node(Node(id="nvswitch", type=NodeType.NVSWITCH, name="NVSwitch"))
    topology.add_node(Node(id="gpu_a", type=NodeType.GPU, name="GPU A"))
    topology.add_node(Node(id="gpu_b", type=NodeType.GPU, name="GPU B"))
    topology.add_node(Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))

    topology.add_edge(
        Edge(
            id="prefill_nvswitch",
            source_id="prefill_pool",
            dest_id="nvswitch",
            interconnect_type=InterconnectType.NVLINK,
            bandwidth_GBps=900.0,
            latency_us=1.0,
            generation="—",
            width="—",
        )
    )
    topology.add_edge(
        Edge(
            id="nvswitch_gpu_a",
            source_id="nvswitch",
            dest_id="gpu_a",
            interconnect_type=InterconnectType.NVSWITCH,
            bandwidth_GBps=900.0,
            latency_us=0.5,
            generation="—",
            width="—",
        )
    )
    topology.add_edge(
        Edge(
            id="nvswitch_gpu_b",
            source_id="nvswitch",
            dest_id="gpu_b",
            interconnect_type=InterconnectType.NVSWITCH,
            bandwidth_GBps=900.0,
            latency_us=0.5,
            generation="—",
            width="—",
        )
    )
    _pcie(topology, "gpu_a_fpga", "gpu_a", "fpga", cfg)
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

    topology.add_node(
        Node(
            id="prefill_pool",
            type=NodeType.PREFILL_POOL,
            name="Prefill Worker Pool",
            metadata={"worker_count": 4, "role": "prefill"},
        )
    )
    topology.add_node(
        Node(
            id="prefill_hbm",
            type=NodeType.MEMORY,
            name="Prefill HBM Cache",
            metadata={"memory_type": "HBM", "role": "prefill_kv_cache"},
        )
    )
    topology.add_node(
        Node(
            id="decode_pool",
            type=NodeType.DECODE_POOL,
            name="Decode Worker Pool",
            metadata={"worker_count": 4, "role": "decode"},
        )
    )
    topology.add_node(
        Node(
            id="decode_hbm",
            type=NodeType.MEMORY,
            name="Decode HBM Cache",
            metadata={"memory_type": "HBM", "role": "decode_kv_cache"},
        )
    )

    _pcie(topology, "prefill_pool_hbm", "prefill_pool", "prefill_hbm", cfg)
    _pcie(topology, "decode_pool_hbm", "decode_pool", "decode_hbm", cfg)
    return topology, positions


def preset_worker_pools_hbm_kvbm(pcie: PCIeConfig | None = None) -> PresetResult:
    """Worker pools + HBM caches + centralized KVBM engine."""
    topology = Topology()
    positions = {
        "prefill_pool": (-280.0, 160.0),
        "prefill_hbm": (-280.0, 20.0),
        "kvbm": (0.0, 0.0),
        "decode_pool": (280.0, 160.0),
        "decode_hbm": (280.0, 20.0),
    }
    cfg = pcie or _default_pcie_config()

    topology.add_node(
        Node(
            id="prefill_pool",
            type=NodeType.PREFILL_POOL,
            name="Prefill Worker Pool",
            metadata={"worker_count": 4},
        )
    )
    topology.add_node(
        Node(
            id="prefill_hbm",
            type=NodeType.MEMORY,
            name="Prefill HBM Cache",
            metadata={"memory_type": "HBM", "role": "prefill_kv_cache"},
        )
    )
    topology.add_node(Node(id="kvbm", type=NodeType.KVBM_ENGINE, name="KVBM FPGA"))
    topology.add_node(
        Node(
            id="decode_pool",
            type=NodeType.DECODE_POOL,
            name="Decode Worker Pool",
            metadata={"worker_count": 4},
        )
    )
    topology.add_node(
        Node(
            id="decode_hbm",
            type=NodeType.MEMORY,
            name="Decode HBM Cache",
            metadata={"memory_type": "HBM", "role": "decode_kv_cache"},
        )
    )

    _pcie(topology, "prefill_pool_hbm", "prefill_pool", "prefill_hbm", cfg)
    _pcie(topology, "decode_pool_hbm", "decode_pool", "decode_hbm", cfg)
    _pcie(topology, "prefill_kvbm", "prefill_pool", "kvbm", cfg)
    _pcie(topology, "kvbm_decode", "kvbm", "decode_pool", cfg)
    return topology, positions


ARCHITECTURE_PRESETS: dict[str, Callable[..., PresetResult]] = {
    "Architecture A — CPU ↔ FPGA (direct PCIe)": preset_architecture_a,
    "Architecture B — CPU → RC → Switch → FPGA": preset_architecture_b,
    "Architecture C — CPU → RC → GPU + FPGA": preset_architecture_c,
    "Architecture D — Prefill → KVBM → Decode": preset_architecture_d,
    "Architecture E — Prefill → Router → Decode (network)": preset_architecture_e,
    "Architecture F — Prefill → NVSwitch → GPU → PCIe → FPGA": preset_architecture_f,
    "Worker Pools + HBM Caches (PCIe only)": preset_worker_pools_hbm,
    "Worker Pools + HBM + KVBM": preset_worker_pools_hbm_kvbm,
}


def build_preset(name: str, pcie: PCIeConfig | None = None) -> PresetResult:
    factory = ARCHITECTURE_PRESETS[name]
    if name.startswith("Architecture E"):
        return factory()
    return factory(pcie)
