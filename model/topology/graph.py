"""Topology graph with path discovery and interconnect-aware cost APIs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from model.interconnect.pcie import (
    PCIeConfig,
    compute_effective_bandwidth_gbps,
    compute_read_latency_us,
    compute_write_latency_us,
)
from model.interconnect.pcie.config import PCIeGeneration, PCIeWidth
from model.topology.types import (
    Edge,
    InterconnectType,
    Node,
    NodeType,
    Path,
    PathSummary,
)


def _edge_key(source_id: str, dest_id: str) -> tuple[str, str]:
    return (source_id, dest_id)


@dataclass
class Topology:
    """
    Layer 4 graph model exposing a clean simulation API.

    The KVBM workload model can call these methods when topology-aware
    behavior is enabled without depending on Streamlit UI objects.
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    pcie_configs: dict[str, PCIeConfig] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        to_remove = [edge_id for edge_id, edge in self.edges.items()
                     if edge.source_id == node_id or edge.dest_id == node_id]
        for edge_id in to_remove:
            self.remove_edge(edge_id)
        self.pcie_configs.pop(node_id, None)

    def add_edge(self, edge: Edge, *, pcie_config: PCIeConfig | None = None) -> None:
        if edge.interconnect_type == InterconnectType.PCIE and pcie_config is not None:
            self.pcie_configs[edge.id] = pcie_config
            bandwidth = compute_effective_bandwidth_gbps(pcie_config)
            edge = Edge(
                id=edge.id,
                source_id=edge.source_id,
                dest_id=edge.dest_id,
                interconnect_type=edge.interconnect_type,
                bandwidth_GBps=bandwidth,
                latency_us=pcie_config.base_latency_us,
                width=pcie_config.width.value,
                generation=pcie_config.generation.value,
                serialization_enabled=edge.serialization_enabled,
                congestion_factor=edge.congestion_factor,
                enabled=edge.enabled,
                metadata=dict(edge.metadata),
            )
        self.edges[edge.id] = edge

    def remove_edge(self, edge_id: str) -> None:
        self.edges.pop(edge_id, None)
        self.pcie_configs.pop(edge_id, None)

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def get_edge(self, edge_id: str) -> Edge | None:
        return self.edges.get(edge_id)

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return [edge for edge in self.edges.values()
                if edge.enabled and edge.source_id == node_id]

    def find_path(
        self,
        source: str,
        destination: str,
        *,
        undirected: bool = True,
    ) -> Path | None:
        """Breadth-first search for a minimum-hop path."""
        if source not in self.nodes or destination not in self.nodes:
            return None
        if source == destination:
            return Path(nodes=[self.nodes[source]], edges=[])

        adjacency: dict[str, list[tuple[str, Edge]]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges.values():
            if not edge.enabled:
                continue
            adjacency[edge.source_id].append((edge.dest_id, edge))
            if undirected:
                adjacency[edge.dest_id].append((edge.source_id, edge.reverse(f"{edge.id}_rev")))

        queue: deque[str] = deque([source])
        parent: dict[str, tuple[str | None, Edge | None]] = {source: (None, None)}

        while queue:
            current = queue.popleft()
            for neighbor, edge in adjacency.get(current, []):
                if neighbor in parent:
                    continue
                parent[neighbor] = (current, edge)
                if neighbor == destination:
                    queue.clear()
                    break
                queue.append(neighbor)

        if destination not in parent:
            return None

        path_nodes: list[Node] = []
        path_edges: list[Edge] = []
        walk = destination
        while True:
            path_nodes.append(self.nodes[walk])
            prev, edge = parent[walk]
            if edge is not None:
                canonical = self.edges.get(edge.id)
                path_edges.append(canonical if canonical is not None else edge)
            if prev is None:
                break
            walk = prev

        path_nodes.reverse()
        path_edges.reverse()
        return Path(nodes=path_nodes, edges=path_edges)

    def path_bandwidth(self, path: Path) -> float:
        """Bottleneck EFFECTIVE bandwidth along the path (GB/s, one direction)."""
        if not path.edges:
            return float("inf")
        return min(edge.bandwidth_GBps for edge in path.edges)

    def _edge_transfer_time_us(
        self,
        edge: Edge,
        message_size: int,
        *,
        is_read: bool = False,
    ) -> tuple[float, float, float]:
        pcie_config = self.pcie_configs.get(edge.id)
        if edge.interconnect_type == InterconnectType.PCIE and pcie_config is not None:
            if is_read:
                result = compute_read_latency_us(pcie_config, message_size)
            else:
                result = compute_write_latency_us(pcie_config, message_size)
            return (
                result.transfer_time_us,
                result.serialization_time_us,
                result.congestion_penalty_us,
            )

        if not edge.serialization_enabled or edge.bandwidth_GBps <= 0:
            fixed = edge.latency_us
            congestion = max(0.0, edge.congestion_factor - 1.0) * edge.latency_us * 0.1
            return fixed + congestion, 0.0, congestion

        bytes_per_us = edge.bandwidth_GBps * 1000.0
        serialization = message_size / bytes_per_us if message_size > 0 else 0.0
        congestion = max(0.0, edge.congestion_factor - 1.0) * edge.latency_us * 0.1
        return edge.latency_us + serialization + congestion, serialization, congestion

    def path_latency(
        self,
        path: Path,
        *,
        message_size: int = 0,
        is_read: bool = False,
    ) -> float:
        """Sum per-edge transfer times for the given payload."""
        if not path.edges:
            return 0.0
        total = 0.0
        for edge in path.edges:
            edge_time, _, _ = self._edge_transfer_time_us(edge, message_size, is_read=is_read)
            total += edge_time
        return total

    def transfer_time(self, path: Path, message_size: int, *, is_read: bool = False) -> float:
        """Alias for path_latency with explicit message size."""
        return self.path_latency(path, message_size=message_size, is_read=is_read)

    def path_summary(
        self,
        path: Path,
        message_size: int,
        *,
        is_read: bool = False,
        offered_bandwidth_gbps: float = 0.0,
    ) -> PathSummary:
        """Structured latency/bandwidth summary for a path."""
        bottleneck = self.path_bandwidth(path)
        serialization_total = 0.0
        congestion_total = 0.0
        fixed_total = 0.0

        for edge in path.edges:
            edge_time, serialization, congestion = self._edge_transfer_time_us(
                edge, message_size, is_read=is_read
            )
            fixed_total += edge_time - serialization
            serialization_total += serialization
            congestion_total += congestion

        transfer_time = fixed_total + serialization_total
        utilization = 0.0
        if bottleneck > 0 and offered_bandwidth_gbps > 0:
            utilization = offered_bandwidth_gbps / bottleneck

        return PathSummary(
            nodes=[node.name for node in path.nodes],
            edges=[edge.id for edge in path.edges],
            hop_count=path.hop_count,
            bottleneck_bandwidth_GBps=bottleneck,
            latency_us=fixed_total,
            serialization_time_us=serialization_total,
            congestion_penalty_us=congestion_total,
            transfer_time_us=transfer_time,
            utilization=min(utilization, 1.0),
            bandwidth_label="EFFECTIVE",
            latency_label="ANALYTICAL ESTIMATE",
        )

    def connect_pcie(
        self,
        edge_id: str,
        source_id: str,
        dest_id: str,
        config: PCIeConfig,
        *,
        bidirectional: bool = True,
    ) -> None:
        """Helper to attach a PCIe-modeled edge (and optional reverse link)."""
        self.add_edge(
            Edge(
                id=edge_id,
                source_id=source_id,
                dest_id=dest_id,
                interconnect_type=InterconnectType.PCIE,
                bandwidth_GBps=0.0,
                latency_us=config.base_latency_us,
                width=config.width.value,
                generation=config.generation.value,
                congestion_factor=config.congestion_factor,
            ),
            pcie_config=config,
        )
        if bidirectional:
            self.add_edge(
                Edge(
                    id=f"{edge_id}_rev",
                    source_id=dest_id,
                    dest_id=source_id,
                    interconnect_type=InterconnectType.PCIE,
                    bandwidth_GBps=0.0,
                    latency_us=config.base_latency_us,
                    width=config.width.value,
                    generation=config.generation.value,
                    congestion_factor=config.congestion_factor,
                ),
                pcie_config=config,
            )


def build_simple_cpu_fpga_topology(
    *,
    generation: PCIeGeneration = PCIeGeneration.GEN4,
    width: PCIeWidth = PCIeWidth.X16,
    switch_hops: int = 0,
) -> Topology:
    """Architecture A/B helper: CPU connected to FPGA over PCIe."""
    topology = Topology()
    topology.add_node(Node(id="cpu", type=NodeType.CPU, name="CPU"))
    topology.add_node(Node(id="fpga", type=NodeType.FPGA, name="KVBM FPGA"))
    config = PCIeConfig(generation=generation, width=width, switch_hops=switch_hops)
    topology.connect_pcie("cpu_fpga", "cpu", "fpga", config)
    return topology
