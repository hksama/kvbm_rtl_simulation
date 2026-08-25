"""
Topology YAML I/O — declarative input format for the simulator.

Schema version 1. Supports every field editable in the Topology UI:
nodes (id, name, type, position, metadata), links (interconnect type,
PCIe config or generic link params, bidirectional flag), and defaults
(default PCIe + DMA engine settings).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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
from model.interconnect.pcie import PCIeConfig, PCIeGeneration, PCIeMPS, PCIeMRRS, PCIeWidth
from model.interconnect.pcie import compute_effective_bandwidth_gbps
from model.topology.graph import Topology
from model.topology.types import Edge, InterconnectType, Node, NodeType, normalize_node_type

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = {1, 2}

DEFAULT_PCIE: dict[str, Any] = {
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

DEFAULT_DMA: dict[str, Any] = {
    "queue_depth": 64,
    "max_outstanding_reads": 16,
    "request_size": 4096,
    "completion_size": 4096,
    "read_latency_us": 0.1,
    "write_latency_us": 0.05,
}


class TopologyFormatError(ValueError):
    """Raised when a topology document is invalid."""


def _registry():
    from model.components.registry import attach_node_config, default_node_config_dict

    return attach_node_config, default_node_config_dict


@dataclass
class TopologyDocument:
    """Parsed topology document."""

    schema_version: int
    name: str
    description: str
    default_pcie: dict[str, Any]
    default_dma: dict[str, Any]
    default_nvlink: dict[str, Any]
    default_infiniband: dict[str, Any]
    nodes: list[dict[str, Any]]
    links: list[dict[str, Any]]


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


def _merge_pcie(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if override:
        merged.update(override)
    return merged


def _parse_position(raw: Any) -> tuple[float, float]:
    if raw is None:
        return (0.0, 0.0)
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        return (float(raw[0]), float(raw[1]))
    if isinstance(raw, dict):
        if "x" in raw and "y" in raw:
            return (float(raw["x"]), float(raw["y"]))
    raise TopologyFormatError(f"Invalid position format: {raw!r}. Use [x, y] or {{x: ..., y: ...}}.")


def _validate_node_type(value: str) -> str:
    normalized = normalize_node_type(value)
    try:
        return NodeType(normalized).value
    except ValueError as exc:
        valid = ", ".join(t.value for t in NodeType)
        raise TopologyFormatError(
            f"Unknown node type '{value}'. Valid types: {valid} (MEMORY aliases to HBM_MEMORY)."
        ) from exc


def _validate_interconnect(value: str) -> str:
    try:
        return InterconnectType(value).value
    except ValueError as exc:
        valid = ", ".join(t.value for t in InterconnectType)
        raise TopologyFormatError(
            f"Unknown interconnect type '{value}'. Valid types: {valid}"
        ) from exc


def parse_topology_document(data: dict[str, Any]) -> TopologyDocument:
    """Parse and validate a topology dict (typically loaded from YAML)."""
    if not isinstance(data, dict):
        raise TopologyFormatError("Topology document must be a mapping.")

    version = int(data.get("schema_version", 1))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise TopologyFormatError(
            f"Unsupported schema_version {version}. Supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}."
        )

    meta = data.get("topology", {})
    if not isinstance(meta, dict):
        raise TopologyFormatError("'topology' section must be a mapping.")

    defaults = data.get("defaults", {}) or {}
    default_pcie = _merge_pcie(DEFAULT_PCIE, defaults.get("pcie"))
    default_dma = {**DEFAULT_DMA, **(defaults.get("dma") or {})}
    default_nvlink = {**DEFAULT_NVLINK, **(defaults.get("nvlink") or {})}
    default_infiniband = {**DEFAULT_INFINIBAND, **(defaults.get("infiniband") or {})}

    nodes_raw = data.get("nodes")
    links_raw = data.get("links")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        raise TopologyFormatError("'nodes' must be a non-empty list.")
    if not isinstance(links_raw, list):
        raise TopologyFormatError("'links' must be a list (may be empty).")

    nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(nodes_raw):
        if not isinstance(raw, dict):
            raise TopologyFormatError(f"Node at index {index} must be a mapping.")
        node_id = raw.get("id")
        if not node_id or not isinstance(node_id, str):
            raise TopologyFormatError(f"Node at index {index} requires string 'id'.")
        if node_id in seen_ids:
            raise TopologyFormatError(f"Duplicate node id '{node_id}'.")
        seen_ids.add(node_id)
        name = raw.get("name")
        if not name:
            raise TopologyFormatError(f"Node '{node_id}' requires 'name'.")
        node_type = _validate_node_type(str(raw.get("type", "")))
        position = _parse_position(raw.get("position"))
        metadata = dict(raw.get("metadata") or {})
        config_data = raw.get("config")
        attach_node_config, default_node_config_dict = _registry()
        if config_data is None and version == 1:
            config_dict = default_node_config_dict(node_type)
            config_dict.update(metadata)
        else:
            base = default_node_config_dict(node_type)
            if isinstance(config_data, dict):
                base.update(config_data)
            config_dict = attach_node_config(node_type, base)
        nodes.append({
            "id": node_id,
            "name": str(name),
            "type": node_type,
            "position": position,
            "metadata": metadata,
            "config": config_dict,
        })

    links: list[dict[str, Any]] = []
    seen_link_ids: set[str] = set()
    for index, raw in enumerate(links_raw):
        if not isinstance(raw, dict):
            raise TopologyFormatError(f"Link at index {index} must be a mapping.")
        link_id = raw.get("id")
        if not link_id or not isinstance(link_id, str):
            raise TopologyFormatError(f"Link at index {index} requires string 'id'.")
        if link_id in seen_link_ids:
            raise TopologyFormatError(f"Duplicate link id '{link_id}'.")
        seen_link_ids.add(link_id)

        source = raw.get("source")
        dest = raw.get("dest")
        if source not in seen_ids or dest not in seen_ids:
            raise TopologyFormatError(
                f"Link '{link_id}' references unknown node(s): source={source!r}, dest={dest!r}."
            )

        interconnect = _validate_interconnect(str(raw.get("interconnect", "PCIe")))
        link_entry: dict[str, Any] = {
            "id": link_id,
            "source": source,
            "dest": dest,
            "interconnect": interconnect,
            "bidirectional": bool(raw.get("bidirectional", True)),
            "enabled": bool(raw.get("enabled", True)),
            "serialization_enabled": bool(raw.get("serialization_enabled", True)),
            "congestion_factor": float(raw.get("congestion_factor", 1.0)),
            "metadata": dict(raw.get("metadata") or {}),
        }
        if interconnect == InterconnectType.PCIE.value:
            link_entry["pcie"] = _merge_pcie(default_pcie, raw.get("pcie"))
        elif interconnect == InterconnectType.NVLINK.value:
            merged = dict(default_nvlink)
            if isinstance(raw.get("nvlink"), dict):
                merged.update(raw["nvlink"])
            link_entry["nvlink"] = merged
        elif interconnect == InterconnectType.INFINIBAND.value:
            merged = dict(default_infiniband)
            if isinstance(raw.get("infiniband"), dict):
                merged.update(raw["infiniband"])
            link_entry["infiniband"] = merged
        links.append(link_entry)

    return TopologyDocument(
        schema_version=version,
        name=str(meta.get("name", "Untitled Topology")),
        description=str(meta.get("description", "")),
        default_pcie=default_pcie,
        default_dma=default_dma,
        default_nvlink=default_nvlink,
        default_infiniband=default_infiniband,
        nodes=nodes,
        links=links,
    )


def document_to_bundle(doc: TopologyDocument) -> dict[str, Any]:
    """Convert a validated document into the internal session bundle format."""
    _, default_node_config_dict = _registry()
    topology = Topology()
    positions: dict[str, tuple[float, float]] = {}
    node_configs: dict[str, dict[str, Any]] = {}

    for node_data in doc.nodes:
        config = dict(node_data.get("config") or default_node_config_dict(node_data["type"]))
        topology.add_node(
            Node(
                id=node_data["id"],
                type=NodeType(node_data["type"]),
                name=node_data["name"],
                metadata=node_data["metadata"],
            ),
            config=config,
        )
        node_configs[node_data["id"]] = config
        positions[node_data["id"]] = node_data["position"]

    pcie_configs: dict[str, dict[str, Any]] = {}
    nvlink_configs: dict[str, dict[str, Any]] = {}
    infiniband_configs: dict[str, dict[str, Any]] = {}

    for link in doc.links:
        interconnect = InterconnectType(link["interconnect"])
        if interconnect == InterconnectType.PCIE:
            pcie = pcie_config_from_dict(link["pcie"])
            topology.connect_pcie(
                link["id"],
                link["source"],
                link["dest"],
                pcie,
                bidirectional=link["bidirectional"],
            )
            pcie_configs[link["id"]] = pcie_config_to_dict(pcie)
            if link["bidirectional"]:
                pcie_configs[f"{link['id']}_rev"] = pcie_config_to_dict(pcie)
        elif interconnect == InterconnectType.NVLINK:
            nvlink = nvlink_config_from_dict(link["nvlink"])
            topology.connect_nvlink(
                link["id"],
                link["source"],
                link["dest"],
                nvlink,
                bidirectional=link["bidirectional"],
            )
            nvlink_configs[link["id"]] = nvlink_config_to_dict(nvlink)
            if link["bidirectional"]:
                nvlink_configs[f"{link['id']}_rev"] = nvlink_config_to_dict(nvlink)
        elif interconnect == InterconnectType.INFINIBAND:
            ib = infiniband_config_from_dict(link["infiniband"])
            topology.connect_infiniband(
                link["id"],
                link["source"],
                link["dest"],
                ib,
                bidirectional=link["bidirectional"],
            )
            infiniband_configs[link["id"]] = infiniband_config_to_dict(ib)
            if link["bidirectional"]:
                infiniband_configs[f"{link['id']}_rev"] = infiniband_config_to_dict(ib)

        edge_ids = [link["id"]]
        if link["bidirectional"]:
            edge_ids.append(f"{link['id']}_rev")
        for edge_id in edge_ids:
            edge = topology.edges.get(edge_id)
            if edge is None:
                continue
            edge.enabled = link["enabled"]
            edge.serialization_enabled = link["serialization_enabled"]
            edge.congestion_factor = link["congestion_factor"]
            edge.metadata = dict(link["metadata"])

    return {
        "schema_version": doc.schema_version,
        "topology_name": doc.name,
        "topology_description": doc.description,
        "nodes": [
            {
                "id": n.id,
                "type": n.type.value,
                "name": n.name,
                "metadata": dict(n.metadata),
            }
            for n in topology.nodes.values()
        ],
        "edges": [
            {
                "id": e.id,
                "source_id": e.source_id,
                "dest_id": e.dest_id,
                "interconnect_type": e.interconnect_type.value,
                "bandwidth_GBps": e.bandwidth_GBps,
                "latency_us": e.latency_us,
                "width": e.width,
                "generation": e.generation,
                "serialization_enabled": e.serialization_enabled,
                "congestion_factor": e.congestion_factor,
                "enabled": e.enabled,
                "metadata": dict(e.metadata),
            }
            for e in topology.edges.values()
        ],
        "pcie_configs": pcie_configs,
        "nvlink_configs": nvlink_configs,
        "infiniband_configs": infiniband_configs,
        "node_configs": node_configs,
        "positions": {node_id: list(pos) for node_id, pos in positions.items()},
        "default_pcie": doc.default_pcie,
        "default_dma": doc.default_dma,
        "default_nvlink": doc.default_nvlink,
        "default_infiniband": doc.default_infiniband,
    }


def bundle_to_document(bundle: dict[str, Any]) -> TopologyDocument:
    """Convert an internal session bundle back to a topology document."""
    _, default_node_config_dict = _registry()
    topology = Topology()
    node_configs = bundle.get("node_configs", {})
    for node_data in bundle.get("nodes", []):
        config = dict(node_configs.get(node_data["id"], default_node_config_dict(node_data["type"])))
        topology.add_node(
            Node(
                id=node_data["id"],
                type=NodeType(node_data["type"]),
                name=node_data["name"],
                metadata=dict(node_data.get("metadata", {})),
            ),
            config=config,
        )
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
        edge = Edge(
            id=edge_data["id"],
            source_id=edge_data["source_id"],
            dest_id=edge_data["dest_id"],
            interconnect_type=InterconnectType(edge_data["interconnect_type"]),
            bandwidth_GBps=float(edge_data["bandwidth_GBps"]),
            latency_us=float(edge_data["latency_us"]),
            width=edge_data.get("width", "x16"),
            generation=edge_data.get("generation", "Gen4"),
            serialization_enabled=bool(edge_data.get("serialization_enabled", True)),
            congestion_factor=float(edge_data.get("congestion_factor", 1.0)),
            enabled=bool(edge_data.get("enabled", True)),
            metadata=dict(edge_data.get("metadata", {})),
        )
        topology.add_edge(
            edge,
            pcie_config=pcie_lookup.get(edge.id),
            nvlink_config=nvlink_lookup.get(edge.id),
            infiniband_config=ib_lookup.get(edge.id),
        )

    positions = {
        node_id: (float(coords[0]), float(coords[1]))
        for node_id, coords in bundle.get("positions", {}).items()
    }

    seen_forward: set[str] = set()
    links: list[dict[str, Any]] = []
    for edge in topology.edges.values():
        if edge.id.endswith("_rev"):
            continue
        rev_id = f"{edge.id}_rev"
        bidirectional = rev_id in topology.edges
        if edge.id in seen_forward:
            continue
        seen_forward.add(edge.id)

        link: dict[str, Any] = {
            "id": edge.id,
            "source": edge.source_id,
            "dest": edge.dest_id,
            "interconnect": edge.interconnect_type.value,
            "bidirectional": bidirectional,
            "enabled": edge.enabled,
            "serialization_enabled": edge.serialization_enabled,
            "congestion_factor": edge.congestion_factor,
            "metadata": dict(edge.metadata) or None,
        }
        if edge.interconnect_type == InterconnectType.PCIE:
            cfg = topology.pcie_configs.get(edge.id)
            if cfg is not None:
                link["pcie"] = pcie_config_to_dict(cfg)
        elif edge.interconnect_type == InterconnectType.NVLINK:
            cfg = topology.nvlink_configs.get(edge.id)
            if cfg is not None:
                link["nvlink"] = nvlink_config_to_dict(cfg)
        elif edge.interconnect_type == InterconnectType.INFINIBAND:
            cfg = topology.infiniband_configs.get(edge.id)
            if cfg is not None:
                link["infiniband"] = infiniband_config_to_dict(cfg)
        links.append(link)

    nodes = []
    for node in topology.nodes.values():
        pos = positions.get(node.id, (0.0, 0.0))
        entry: dict[str, Any] = {
            "id": node.id,
            "name": node.name,
            "type": node.type.value,
            "position": {"x": pos[0], "y": pos[1]},
        }
        config = topology.node_configs.get(node.id) or node.metadata.get("config")
        if config:
            entry["config"] = dict(config)
        if node.metadata:
            entry["metadata"] = {k: v for k, v in node.metadata.items() if k != "config"} or None
        nodes.append(entry)

    return TopologyDocument(
        schema_version=int(bundle.get("schema_version", SCHEMA_VERSION)),
        name=str(bundle.get("topology_name", "Exported Topology")),
        description=str(bundle.get("topology_description", "")),
        default_pcie=dict(bundle.get("default_pcie", DEFAULT_PCIE)),
        default_dma=dict(bundle.get("default_dma", DEFAULT_DMA)),
        default_nvlink=dict(bundle.get("default_nvlink", DEFAULT_NVLINK)),
        default_infiniband=dict(bundle.get("default_infiniband", DEFAULT_INFINIBAND)),
        nodes=nodes,
        links=links,
    )


def _position_to_dict(position: Any) -> dict[str, float]:
    if isinstance(position, dict):
        return {"x": float(position["x"]), "y": float(position["y"])}
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        return {"x": float(position[0]), "y": float(position[1])}
    raise TopologyFormatError(f"Invalid position in document: {position!r}")


def document_to_yaml_dict(doc: TopologyDocument) -> dict[str, Any]:
    """Serialize a TopologyDocument to a YAML-ready dict."""
    nodes_out = []
    for node in doc.nodes:
        entry: dict[str, Any] = {
            "id": node["id"],
            "name": node["name"],
            "type": node["type"],
            "position": _position_to_dict(node["position"]),
        }
        if node.get("config"):
            entry["config"] = node["config"]
        if node.get("metadata"):
            entry["metadata"] = node["metadata"]
        nodes_out.append(entry)

    links_out = []
    for link in doc.links:
        entry: dict[str, Any] = {
            "id": link["id"],
            "source": link["source"],
            "dest": link["dest"],
            "interconnect": link["interconnect"],
            "bidirectional": link["bidirectional"],
            "enabled": link["enabled"],
            "serialization_enabled": link["serialization_enabled"],
            "congestion_factor": link["congestion_factor"],
        }
        if link.get("metadata"):
            entry["metadata"] = link["metadata"]
        if link["interconnect"] == InterconnectType.PCIE.value and link.get("pcie"):
            entry["pcie"] = link["pcie"]
        elif link["interconnect"] == InterconnectType.NVLINK.value and link.get("nvlink"):
            entry["nvlink"] = link["nvlink"]
        elif link["interconnect"] == InterconnectType.INFINIBAND.value and link.get("infiniband"):
            entry["infiniband"] = link["infiniband"]
        links_out.append(entry)

    result: dict[str, Any] = {
        "schema_version": doc.schema_version,
        "topology": {
            "name": doc.name,
        },
        "defaults": {
            "pcie": doc.default_pcie,
            "dma": doc.default_dma,
            "nvlink": doc.default_nvlink,
            "infiniband": doc.default_infiniband,
        },
        "nodes": nodes_out,
        "links": links_out,
    }
    if doc.description:
        result["topology"]["description"] = doc.description
    return result


def load_topology_yaml(path: str | Path) -> dict[str, Any]:
    """Load a topology YAML file into an internal session bundle."""
    text = Path(path).read_text(encoding="utf-8")
    return load_topology_yaml_text(text)


def load_topology_yaml_text(text: str) -> dict[str, Any]:
    """Parse topology YAML text into an internal session bundle."""
    data = yaml.safe_load(text)
    if data is None:
        raise TopologyFormatError("Topology file is empty.")
    doc = parse_topology_document(data)
    return document_to_bundle(doc)


def dump_topology_yaml(bundle: dict[str, Any], path: str | Path) -> None:
    """Write an internal session bundle to a topology YAML file."""
    Path(path).write_text(dump_topology_yaml_text(bundle), encoding="utf-8")


def dump_topology_yaml_text(bundle: dict[str, Any]) -> str:
    """Serialize an internal session bundle to topology YAML text."""
    doc = bundle_to_document(bundle)
    payload = document_to_yaml_dict(doc)
    return yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
