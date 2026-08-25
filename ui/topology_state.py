"""Session-state helpers for topology / PCIe UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from model.interconnect.infiniband import infiniband_config_from_dict, infiniband_config_to_dict
from model.interconnect.nvlink import nvlink_config_from_dict, nvlink_config_to_dict
from model.interconnect.pcie import PCIeConfig
from model.topology import Edge, InterconnectType, Node, NodeType, Topology
from model.topology.io import (
    DEFAULT_DMA,
    DEFAULT_INFINIBAND,
    DEFAULT_NVLINK,
    DEFAULT_PCIE,
    SCHEMA_VERSION,
    dump_topology_yaml_text,
    load_topology_yaml,
    load_topology_yaml_text,
    pcie_config_from_dict,
    pcie_config_to_dict,
)
from model.topology.presets import (
    ARCHITECTURE_PRESETS,
    build_preset,
    preset_architecture_a,
)

SESSION_KEY = "kvbm_topology_bundle"
DMA_SESSION_KEY = "kvbm_dma_engine"


def _default_pcie_dict() -> dict[str, Any]:
    return dict(DEFAULT_PCIE)


def _default_dma_dict() -> dict[str, Any]:
    return dict(DEFAULT_DMA)


def _default_nvlink_dict() -> dict[str, Any]:
    return dict(DEFAULT_NVLINK)


def _default_infiniband_dict() -> dict[str, Any]:
    return dict(DEFAULT_INFINIBAND)


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
    node_configs = bundle.get("node_configs", {})
    for node_data in bundle.get("nodes", []):
        config = node_configs.get(node_data["id"])
        topology.add_node(_node_from_dict(node_data), config=config)
    pcie_lookup = {
        edge_id: pcie_config_from_dict(cfg)
        for edge_id, cfg in bundle.get("pcie_configs", {}).items()
    }
    nvlink_lookup = {
        edge_id: nvlink_config_from_dict(cfg)
        for edge_id, cfg in bundle.get("nvlink_configs", {}).items()
    }
    ib_lookup = {
        edge_id: infiniband_config_from_dict(cfg)
        for edge_id, cfg in bundle.get("infiniband_configs", {}).items()
    }
    for edge_data in bundle.get("edges", []):
        edge = _edge_from_dict(edge_data)
        topology.add_edge(
            edge,
            pcie_config=pcie_lookup.get(edge.id),
            nvlink_config=nvlink_lookup.get(edge.id),
            infiniband_config=ib_lookup.get(edge.id),
        )
    return topology


def topology_to_bundle(
    topology: Topology,
    positions: dict[str, tuple[float, float]],
    default_pcie: dict[str, Any],
    default_dma: dict[str, Any] | None = None,
    *,
    schema_version: int = SCHEMA_VERSION,
    default_nvlink: dict[str, Any] | None = None,
    default_infiniband: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "nodes": [_node_to_dict(n) for n in topology.nodes.values()],
        "edges": [_edge_to_dict(e) for e in topology.edges.values()],
        "pcie_configs": {
            edge_id: pcie_config_to_dict(cfg)
            for edge_id, cfg in topology.pcie_configs.items()
        },
        "nvlink_configs": {
            edge_id: nvlink_config_to_dict(cfg)
            for edge_id, cfg in topology.nvlink_configs.items()
        },
        "infiniband_configs": {
            edge_id: infiniband_config_to_dict(cfg)
            for edge_id, cfg in topology.infiniband_configs.items()
        },
        "node_configs": {
            node_id: dict(cfg) for node_id, cfg in topology.node_configs.items()
        },
        "positions": {node_id: list(pos) for node_id, pos in positions.items()},
        "default_pcie": default_pcie,
        "default_dma": default_dma or _default_dma_dict(),
        "default_nvlink": default_nvlink or _default_nvlink_dict(),
        "default_infiniband": default_infiniband or _default_infiniband_dict(),
    }


def init_topology_session() -> None:
    if SESSION_KEY not in st.session_state:
        topology, positions = preset_architecture_a(pcie_config_from_dict(_default_pcie_dict()))
        st.session_state[SESSION_KEY] = topology_to_bundle(
            topology, positions, _default_pcie_dict(), _default_dma_dict()
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


def get_default_dma_dict() -> dict[str, Any]:
    bundle = get_bundle()
    return dict(bundle.get("default_dma", _default_dma_dict()))


def set_default_dma_dict(data: dict[str, Any]) -> None:
    bundle = get_bundle()
    bundle["default_dma"] = data
    save_bundle(bundle)


def get_default_nvlink_dict() -> dict[str, Any]:
    bundle = get_bundle()
    return dict(bundle.get("default_nvlink", _default_nvlink_dict()))


def set_default_nvlink_dict(data: dict[str, Any]) -> None:
    bundle = get_bundle()
    bundle["default_nvlink"] = data
    save_bundle(bundle)


def get_default_infiniband_dict() -> dict[str, Any]:
    bundle = get_bundle()
    return dict(bundle.get("default_infiniband", _default_infiniband_dict()))


def set_default_infiniband_dict(data: dict[str, Any]) -> None:
    bundle = get_bundle()
    bundle["default_infiniband"] = data
    save_bundle(bundle)


def load_preset(name: str) -> None:
    topology, positions = build_preset(name, pcie_config_from_dict(get_default_pcie_dict()))
    save_bundle(topology_to_bundle(topology, positions, get_default_pcie_dict(), get_default_dma_dict()))


def load_topology_from_text(yaml_text: str) -> None:
    """Replace the current session topology from YAML text."""
    save_bundle(load_topology_yaml_text(yaml_text))


def load_topology_from_file(path: str | Path) -> None:
    """Replace the current session topology from a YAML file path."""
    save_bundle(load_topology_yaml(path))


def export_topology_yaml_text() -> str:
    """Serialize the current session topology to YAML text."""
    return dump_topology_yaml_text(get_bundle())


def sync_topology(topology: Topology, positions: dict[str, tuple[float, float]]) -> None:
    bundle = get_bundle()
    save_bundle(
        topology_to_bundle(
            topology,
            positions,
            get_default_pcie_dict(),
            get_default_dma_dict(),
            schema_version=int(bundle.get("schema_version", SCHEMA_VERSION)),
            default_nvlink=dict(bundle.get("default_nvlink", _default_nvlink_dict())),
            default_infiniband=dict(bundle.get("default_infiniband", _default_infiniband_dict())),
        )
    )


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
