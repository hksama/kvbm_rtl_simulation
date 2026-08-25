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
        from model.interconnect.infiniband import InfiniBandConfig

        topology = Topology()
        topology.add_node(Node(id="n1", type=NodeType.PREFILL_POOL, name="Prefill Pool"))
        topology.add_node(Node(id="n2", type=NodeType.DECODE_POOL, name="Decode Pool"))
        ib = InfiniBandConfig(effective_bandwidth_GBps=25.0, latency_us=2.0)
        topology.connect_infiniband("pool_link", "n1", "n2", ib, bidirectional=False)
        path = topology.find_path("n1", "n2", undirected=False)
        assert path is not None
        assert topology.path_latency(path, message_size=25000000) > 2.0

    def test_nvlink_path_uses_nvlink_effective_bandwidth(self):
        from model.interconnect.nvlink import NVLinkConfig, effective_nvlink_bandwidth_gbps

        topology = Topology()
        topology.add_node(Node(id="gpu_a", type=NodeType.GPU, name="GPU A"))
        topology.add_node(Node(id="gpu_b", type=NodeType.GPU, name="GPU B"))
        nvlink = NVLinkConfig(aggregate_bandwidth_GBps=900.0, protocol_efficiency=0.95)
        topology.connect_nvlink("gpu_link", "gpu_a", "gpu_b", nvlink, bidirectional=False)

        path = topology.find_path("gpu_a", "gpu_b", undirected=False)
        assert path is not None
        assert topology.path_bandwidth(path) == pytest.approx(
            effective_nvlink_bandwidth_gbps(nvlink)
        )

    def test_mixed_path_bottleneck_is_minimum_edge(self):
        from model.interconnect.nvlink import NVLinkConfig, effective_nvlink_bandwidth_gbps

        topology = Topology()
        topology.add_node(Node(id="gpu_a", type=NodeType.GPU, name="GPU A"))
        topology.add_node(Node(id="sw", type=NodeType.PCIE_SWITCH, name="Switch"))
        topology.add_node(Node(id="gpu_b", type=NodeType.GPU, name="GPU B"))

        nvlink = NVLinkConfig(aggregate_bandwidth_GBps=900.0, protocol_efficiency=0.95)
        slow_pcie = PCIeConfig(generation=PCIeGeneration.GEN3, width=PCIeWidth.X4)

        topology.connect_nvlink("a_sw", "gpu_a", "sw", nvlink, bidirectional=False)
        topology.connect_pcie("sw_b", "sw", "gpu_b", slow_pcie, bidirectional=False)

        path = topology.find_path("gpu_a", "gpu_b", undirected=False)
        assert path is not None
        assert path.hop_count == 2
        edge_bandwidths = [edge.bandwidth_GBps for edge in path.edges]
        assert topology.path_bandwidth(path) == pytest.approx(min(edge_bandwidths))
        assert topology.path_bandwidth(path) < effective_nvlink_bandwidth_gbps(nvlink)

    def test_rc_switch_hops_add_node_latency(self):
        from model.topology.presets import preset_architecture_b

        topology, _ = preset_architecture_b()
        topology.node_configs["rc"]["hop_latency_us"] = 1.0
        topology.node_configs["switch"]["hop_latency_us"] = 0.5

        path = topology.find_path("cpu", "fpga")
        assert path is not None
        summary = topology.path_summary(path, message_size=4096, is_read=False)
        assert summary.node_hop_latency_us == pytest.approx(1.5)

        baseline = build_simple_cpu_fpga_topology()
        baseline_path = baseline.find_path("cpu", "fpga")
        baseline_summary = baseline.path_summary(baseline_path, message_size=4096, is_read=False)
        assert summary.transfer_time_us > baseline_summary.transfer_time_us

    def test_hbm_destination_adds_memory_service_latency(self):
        from model.topology.presets import preset_worker_pools_hbm

        topology, _ = preset_worker_pools_hbm()
        topology.node_configs["prefill_hbm"]["read_latency_ns"] = 200.0

        path = topology.find_path("prefill_pool", "prefill_hbm")
        assert path is not None
        summary = topology.path_summary(path, message_size=4096, is_read=True)
        assert summary.memory_service_latency_us == pytest.approx(0.2)

    def test_pool_metadata_in_path_summary(self):
        from model.topology.presets import preset_architecture_e

        topology, _ = preset_architecture_e()
        path = topology.find_path("prefill_pool", "decode_pool")
        assert path is not None
        summary = topology.path_summary(path, message_size=4096)
        assert "pools" in summary.metadata
        assert len(summary.metadata["pools"]) == 2

    def test_bidirectional_link_does_not_double_bandwidth(self):
        from model.interconnect.nvlink import NVLinkConfig, effective_nvlink_bandwidth_gbps

        topology = Topology()
        topology.add_node(Node(id="a", type=NodeType.GPU, name="A"))
        topology.add_node(Node(id="b", type=NodeType.GPU, name="B"))
        nvlink = NVLinkConfig(aggregate_bandwidth_GBps=900.0, protocol_efficiency=0.95)
        topology.connect_nvlink("a_b", "a", "b", nvlink, bidirectional=True)

        path = topology.find_path("a", "b", undirected=False)
        assert path is not None
        assert len(path.edges) == 1
        assert topology.path_bandwidth(path) == pytest.approx(
            effective_nvlink_bandwidth_gbps(nvlink)
        )
