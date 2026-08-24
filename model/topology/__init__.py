"""Layer 4: Topology / system graph model."""

from model.topology.graph import Topology
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
    "build_preset",
]
