"""Session-state helpers for topology / PCIe UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from model.interconnect.pcie import PCIeConfig, PCIeGeneration, PCIeMPS, PCIeMRRS, PCIeWidth
from model.topology import Edge, InterconnectType, Node, NodeType, Topology


SESSION_KEY = "kvbm_topology_bundle"


def _default_pcie_dict() -> dict[str, Any]:
    return {
        "generation": PCIeGeneration.GEN4.value,
        "width": PCIeWidth.X16.value,
        "mps": PCIeMPS.B256.value,
        "mrrs": PCIeMRRS.B512.value,
        "base_latency_us": 0.5,
        "switch_hops": 0,
        "congestion_factor": 1.0,
        "protocol_efficiency": 0.95,
        "switch_hop_latency_us": 0.15,
        "read_request_latency_us": 0.2,
        "memory_service_latency_us": 0.5,
        "endpoint_processing_us": 0.1,
        "memory_processing_us": 0.1,
        "congestion_base_penalty_us": 0.05,
    }


def pcie_config_from_dict(data: dict[str, Any]) -> PCIeConfig:
    return PCIeConfig(
        generation=PCIeGeneration(data["generation"]),
        width=PCIeWidth(data["width"]),
        mps=PCIeMPS(int(data["mps"])),
        mrrs=PCIeMRRS(int(data["mrrs"])),
        base_latency_us=float(data["base_latency_us"]),
        switch_hops=int(data["switch_hops"]),
        congestion_factor=float(data["congestion_factor"]),
        protocol_efficiency=float(data["protocol_efficiency"]),
        switch_hop_latency_us=float(data["switch_hop_latency_us"]),
        read_request_latency_us=float(data["read_request_latency_us"]),
        memory_service_latency_us=float(data["memory_service_latency_us"]),
        endpoint_processing_us=float(data["endpoint_processing_us"]),
        memory_processing_us=float(data["memory_processing_us"]),
        congestion_base_penalty_us=float(data["congestion_base_penalty_us"]),
    )


def pcie_config_to_dict(config: PCIeConfig) -> dict[str, Any]:
    return {
        "generation": config.generation.value,
        "width": config.width.value,
        "mps": config.mps.value,
        "mrrs": config.mrrs.value,
        "base_latency_us": config.base_latency_us,
        "switch_hops": config.switch_hops,
        "congestion_factor": config.congestion_factor,
        "protocol_efficiency": config.protocol_efficiency,
        "switch_hop_latency_us": config.switch_hop_latency_us,
        "read_request_latency_us": config.read_request_latency_us,
        "memory_service_latency_us": config.memory_service_latency_us,
        "endpoint_processing_us": config.endpoint_processing_us,
        "memory_processing_us": config.memory_processing_us,
        "congestion_base_penalty_us": config.congestion_base_penalty_us,
    }


def _node_to_dict(node: Node) -> dict[str, Any]:
    return {
        "id": node.id,
        "type": node.type.value,
        "name": node.name,
        "metadata": dict(node.metadata),
    }


def _node_from_dict(data: dict[str, Any]) -> Node:
    return Node(
        id=data["id"],
        type=NodeType(data["type"]),
        name=data["name"],
        metadata=dict(data.get("metadata", {})),
    )


def _edge_to_dict(edge: Edge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source_id": edge.source_id,
        "dest_id": edge.dest_id,
        "interconnect_type": edge.interconnect_type.value,
        "bandwidth_GBps": edge.bandwidth_GBps,
        "latency_us": edge.latency_us,
        "width": edge.width,
        "generation": edge.generation,
        "serialization_enabled": edge.serialization_enabled,
        "congestion_factor": edge.congestion_factor,
        "enabled": edge.enabled,
        "metadata": dict(edge.metadata),
    }


def _edge_from_dict(data: dict[str, Any]) -> Edge:
    return Edge(
        id=data["id"],
        source_id=data["source_id"],
        dest_id=data["dest_id"],
        interconnect_type=InterconnectType(data["interconnect_type"]),
        bandwidth_GBps=float(data["bandwidth_GBps"]),
        latency_us=float(data["latency_us"]),
        width=data.get("width", "x16"),
        generation=data.get("generation", "Gen4"),
        serialization_enabled=bool(data.get("serialization_enabled", True)),
        congestion_factor=float(data.get("congestion_factor", 1.0)),
        enabled=bool(data.get("enabled", True)),
        metadata=dict(data.get("metadata", {})),
    )


def bundle_to_topology(bundle: dict[str, Any]) -> Topology:
    topology = Topology()
    for node_data in bundle.get("nodes", []):
        topology.add_node(_node_from_dict(node_data))
    pcie_lookup = {
        edge_id: pcie_config_from_dict(cfg)
        for edge_id, cfg in bundle.get("pcie_configs", {}).items()
    }
    for edge_data in bundle.get("edges", []):
        edge = _edge_from_dict(edge_data)
        pcie = pcie_lookup.get(edge.id)
        topology.add_edge(edge, pcie_config=pcie)
    return topology


def topology_to_bundle(
    topology: Topology,
    positions: dict[str, tuple[float, float]],
    default_pcie: dict[str, Any],
) -> dict[str, Any]:
    return {
        "nodes": [_node_to_dict(n) for n in topology.nodes.values()],
        "edges": [_edge_to_dict(e) for e in topology.edges.values()],
        "pcie_configs": {
            edge_id: pcie_config_to_dict(cfg)
            for edge_id, cfg in topology.pcie_configs.items()
        },
        "positions": {node_id: list(pos) for node_id, pos in positions.items()},
        "default_pcie": default_pcie,
    }


def init_topology_session() -> None:
    if SESSION_KEY not in st.session_state:
        topology, positions = preset_architecture_a()
        st.session_state[SESSION_KEY] = topology_to_bundle(
            topology, positions, _default_pcie_dict()
        )


def get_bundle() -> dict[str, Any]:
    init_topology_session()
    return st.session_state[SESSION_KEY]


def save_bundle(bundle: dict[str, Any]) -> None:
    st.session_state[SESSION_KEY] = bundle


def get_topology() -> Topology:
    return bundle_to_topology(get_bundle())


def get_positions() -> dict[str, tuple[float, float]]:
    bundle = get_bundle()
    return {
        node_id: (float(coords[0]), float(coords[1]))
        for node_id, coords in bundle.get("positions", {}).items()
    }


def get_default_pcie_dict() -> dict[str, Any]:
    bundle = get_bundle()
    return dict(bundle.get("default_pcie", _default_pcie_dict()))


def set_default_pcie_dict(data: dict[str, Any]) -> None:
    bundle = get_bundle()
    bundle["default_pcie"] = data
    save_bundle(bundle)


def preset_architecture_a() -> tuple[Topology, dict[str, tuple[float, float]]]:
    topology = Topology()
    positions = {
        "cpu": (0.0, 0.0),
        "fpga": (0.0, -200.0),
    }
    topology.add_node(Node(id="cpu", type=NodeType.CPU, name="CPU"))
    topology.add_node(Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    config = pcie_config_from_dict(_default_pcie_dict())
    topology.connect_pcie("cpu_fpga", "cpu", "fpga", config)
    return topology, positions


def preset_architecture_b() -> tuple[Topology, dict[str, tuple[float, float]]]:
    topology = Topology()
    positions = {
        "cpu": (0.0, 200.0),
        "rc": (0.0, 80.0),
        "switch": (0.0, -40.0),
        "fpga": (0.0, -200.0),
    }
    topology.add_node(Node(id="cpu", type=NodeType.CPU, name="CPU"))
    topology.add_node(Node(id="rc", type=NodeType.PCIE_ROOT_COMPLEX, name="PCIe Root Complex"))
    topology.add_node(Node(id="switch", type=NodeType.PCIE_SWITCH, name="PCIe Switch"))
    topology.add_node(Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    config = pcie_config_from_dict(_default_pcie_dict())
    topology.connect_pcie("cpu_rc", "cpu", "rc", config)
    sw_config = pcie_config_from_dict({**_default_pcie_dict(), "switch_hops": 1})
    topology.connect_pcie("rc_switch", "rc", "switch", sw_config)
    topology.connect_pcie("switch_fpga", "switch", "fpga", sw_config)
    return topology, positions


def preset_architecture_c() -> tuple[Topology, dict[str, tuple[float, float]]]:
    topology = Topology()
    positions = {
        "cpu": (0.0, 200.0),
        "rc": (0.0, 60.0),
        "gpu": (-180.0, -120.0),
        "fpga": (180.0, -120.0),
    }
    topology.add_node(Node(id="cpu", type=NodeType.CPU, name="CPU"))
    topology.add_node(Node(id="rc", type=NodeType.PCIE_ROOT_COMPLEX, name="PCIe Root Complex"))
    topology.add_node(Node(id="gpu", type=NodeType.GPU, name="GPU"))
    topology.add_node(Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    config = pcie_config_from_dict(_default_pcie_dict())
    topology.connect_pcie("cpu_rc", "cpu", "rc", config)
    topology.connect_pcie("rc_gpu", "rc", "gpu", config)
    topology.connect_pcie("rc_fpga", "rc", "fpga", config)
    return topology, positions


def preset_architecture_d() -> tuple[Topology, dict[str, tuple[float, float]]]:
    topology = Topology()
    positions = {
        "prefill": (0.0, 200.0),
        "kvbm": (0.0, 0.0),
        "decode": (0.0, -200.0),
    }
    topology.add_node(Node(id="prefill", type=NodeType.PREFILL_POOL, name="Prefill GPU Pool"))
    topology.add_node(Node(id="kvbm", type=NodeType.KVBM_ENGINE, name="KVBM FPGA"))
    topology.add_node(Node(id="decode", type=NodeType.DECODE_POOL, name="Decode GPU Pool"))
    config = pcie_config_from_dict(_default_pcie_dict())
    topology.connect_pcie("prefill_kvbm", "prefill", "kvbm", config)
    topology.connect_pcie("kvbm_decode", "kvbm", "decode", config)
    return topology, positions


ARCHITECTURE_PRESETS = {
    "Architecture A — CPU ↔ FPGA (direct PCIe)": preset_architecture_a,
    "Architecture B — CPU → RC → Switch → FPGA": preset_architecture_b,
    "Architecture C — CPU → RC → GPU + FPGA": preset_architecture_c,
    "Architecture D — Prefill → KVBM → Decode": preset_architecture_d,
}


def load_preset(name: str) -> None:
    factory = ARCHITECTURE_PRESETS[name]
    topology, positions = factory()
    save_bundle(topology_to_bundle(topology, positions, get_default_pcie_dict()))


def sync_topology(topology: Topology, positions: dict[str, tuple[float, float]]) -> None:
    save_bundle(topology_to_bundle(topology, positions, get_default_pcie_dict()))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    import streamlit as st

    st.set_page_config(page_title="KVBM — module info", layout="centered")
    st.title("Not a Streamlit app")
    st.warning(
        "`topology_state.py` is a **helper module** (session state + presets). "
        "It does not render any UI on its own."
    )
    st.markdown("Run one of these instead:")
    st.code("uv run streamlit run ui/topology_app.py", language="bash")
    st.code("uv run streamlit run ui/topology_ui.py", language="bash")
    st.code("uv run streamlit run ui/app.py", language="bash")
    st.caption("The full explorer opens the same topology UI under the **Topology & PCIe** tab.")
