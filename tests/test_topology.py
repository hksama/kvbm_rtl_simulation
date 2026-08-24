"""Tests for topology graph and interconnect edge model (Parts 1–3)."""

import pytest

from model.interconnect.pcie import PCIeConfig, PCIeGeneration, PCIeWidth
from model.topology import Edge, InterconnectType, Node, NodeType, Topology
from model.topology.graph import build_simple_cpu_fpga_topology


class TestTopologyGraph:
    def test_add_nodes_and_find_path(self):
        topology = Topology()
        topology.add_node(Node(id="a", type=NodeType.CPU, name="CPU"))
        topology.add_node(Node(id="b", type=NodeType.PCIE_SWITCH, name="Switch"))
        topology.add_node(Node(id="c", type=NodeType.FPGA, name="FPGA"))

        config = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X16)
        topology.connect_pcie("a_b", "a", "b", config)
        topology.connect_pcie("b_c", "b", "c", config)

        path = topology.find_path("a", "c")
        assert path is not None
        assert path.hop_count == 2
        assert [node.id for node in path.nodes] == ["a", "b", "c"]

    def test_path_bandwidth_is_bottleneck(self):
        topology = Topology()
        topology.add_node(Node(id="gpu", type=NodeType.GPU, name="GPU"))
        topology.add_node(Node(id="sw", type=NodeType.PCIE_SWITCH, name="Switch"))
        topology.add_node(Node(id="fpga", type=NodeType.FPGA, name="FPGA"))

        fast = PCIeConfig(generation=PCIeGeneration.GEN5, width=PCIeWidth.X16)
        slow = PCIeConfig(generation=PCIeGeneration.GEN3, width=PCIeWidth.X4)
        topology.connect_pcie("gpu_sw", "gpu", "sw", fast)
        topology.connect_pcie("sw_fpga", "sw", "fpga", slow)

        path = topology.find_path("gpu", "fpga")
        assert path is not None
        bottleneck = topology.path_bandwidth(path)
        assert bottleneck == pytest.approx(min(edge.bandwidth_GBps for edge in path.edges))

    def test_transfer_time_increases_with_message_size(self):
        topology = build_simple_cpu_fpga_topology()
        path = topology.find_path("cpu", "fpga")
        assert path is not None
        small = topology.transfer_time(path, 4096)
        large = topology.transfer_time(path, 4 * 1024 * 1024)
        assert large > small

    def test_path_summary_structure(self):
        topology = build_simple_cpu_fpga_topology(switch_hops=1)
        path = topology.find_path("cpu", "fpga")
        assert path is not None
        summary = topology.path_summary(path, message_size=1024 * 1024, is_read=True)
        assert summary.hop_count == 1
        assert summary.bottleneck_bandwidth_GBps > 0
        assert summary.latency_label == "ANALYTICAL ESTIMATE"
        assert summary.bandwidth_label == "EFFECTIVE"
        assert "nodes" in summary.to_dict()

    def test_remove_node_removes_incident_edges(self):
        topology = build_simple_cpu_fpga_topology()
        topology.remove_node("cpu")
        assert "cpu" not in topology.nodes
        assert len(topology.edges) == 0

    def test_generic_edge_latency_model(self):
        topology = Topology()
        topology.add_node(Node(id="n1", type=NodeType.ROUTER, name="Router"))
        topology.add_node(Node(id="n2", type=NodeType.NIC, name="NIC"))
        topology.add_edge(
            Edge(
                id="net",
                source_id="n1",
                dest_id="n2",
                interconnect_type=InterconnectType.ETHERNET,
                bandwidth_GBps=25.0,
                latency_us=2.0,
            )
        )
        path = topology.find_path("n1", "n2", undirected=False)
        assert path is not None
        assert topology.path_latency(path, message_size=25000000) > 2.0
