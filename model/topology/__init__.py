"""Layer 4: Topology / system graph model."""

from model.topology.graph import Topology
from model.topology.io import (
    TopologyFormatError,
    dump_topology_yaml,
    dump_topology_yaml_text,
    load_topology_yaml,
    load_topology_yaml_text,
    parse_topology_document,
)
from model.topology.presets import ARCHITECTURE_PRESETS, build_preset
from model.topology.types import (
    Edge,
    InterconnectType,
    Node,
    NodeType,
    Path,
    PathSummary,
)

__all__ = [
    "ARCHITECTURE_PRESETS",
    "Edge",
    "InterconnectType",
    "Node",
    "NodeType",
    "Path",
    "PathSummary",
    "Topology",
    "TopologyFormatError",
    "build_preset",
    "dump_topology_yaml",
    "dump_topology_yaml_text",
    "load_topology_yaml",
    "load_topology_yaml_text",
    "parse_topology_document",
]
