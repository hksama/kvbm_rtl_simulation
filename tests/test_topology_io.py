"""Tests for topology YAML input format."""

from pathlib import Path

import pytest
import yaml

from model.topology.io import (
    TopologyFormatError,
    document_to_bundle,
    dump_topology_yaml_text,
    load_topology_yaml,
    load_topology_yaml_text,
    parse_topology_document,
)


EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "configs" / "topologies" / "worker_pools_hbm.yaml"


class TestTopologyYAML:
    def test_load_example_file(self):
        bundle = load_topology_yaml(EXAMPLE_PATH)
        assert len(bundle["nodes"]) == 4
        assert len(bundle["edges"]) == 4  # two bidirectional PCIe links
        assert "prefill_pool" in bundle["positions"]

    def test_round_trip(self):
        original = load_topology_yaml(EXAMPLE_PATH)
        yaml_text = dump_topology_yaml_text(original)
        reloaded = load_topology_yaml_text(yaml_text)
        assert len(reloaded["nodes"]) == len(original["nodes"])
        assert len(reloaded["edges"]) == len(original["edges"])

    def test_fragmentation_fields_preserved(self):
        bundle = load_topology_yaml(EXAMPLE_PATH)
        assert bundle["default_pcie"]["mps"] == 256
        assert bundle["default_dma"]["queue_depth"] == 64

    def test_pcie_link_gets_config(self):
        bundle = load_topology_yaml(EXAMPLE_PATH)
        assert "prefill_pool_hbm" in bundle["pcie_configs"]
        assert bundle["pcie_configs"]["prefill_pool_hbm"]["generation"] == "Gen4"

    def test_invalid_node_type_raises(self):
        data = yaml.safe_load(EXAMPLE_PATH.read_text())
        data["nodes"][0]["type"] = "NOT_A_TYPE"
        with pytest.raises(TopologyFormatError):
            parse_topology_document(data)

    def test_unknown_link_endpoint_raises(self):
        data = yaml.safe_load(EXAMPLE_PATH.read_text())
        data["links"].append({
            "id": "bad",
            "source": "missing",
            "dest": "prefill_hbm",
            "interconnect": "PCIe",
        })
        with pytest.raises(TopologyFormatError):
            parse_topology_document(data)

    def test_ethernet_link(self):
        text = """
schema_version: 1
topology:
  name: Network test
defaults:
  pcie:
    generation: Gen4
    width: x16
    mps: 256
    mrrs: 512
    base_latency_us: 0.5
    switch_hops: 0
    congestion_factor: 1.0
    protocol_efficiency: 0.95
    switch_hop_latency_us: 0.15
    read_request_latency_us: 0.2
    memory_service_latency_us: 0.5
    endpoint_processing_us: 0.1
    memory_processing_us: 0.1
    congestion_base_penalty_us: 0.05
  dma:
    queue_depth: 64
    max_outstanding_reads: 16
    request_size: 4096
    completion_size: 4096
    read_latency_us: 0.1
    write_latency_us: 0.05
nodes:
  - id: a
    name: A
    type: ROUTER
    position: {x: 0, y: 0}
  - id: b
    name: B
    type: NIC
    position: {x: 100, y: 0}
links:
  - id: a_b
    source: a
    dest: b
    interconnect: Ethernet
    bidirectional: true
    bandwidth_GBps: 100.0
    latency_us: 2.0
"""
        bundle = load_topology_yaml_text(text)
        assert bundle["edges"][0]["interconnect_type"] == "Ethernet"
        assert bundle["edges"][0]["bandwidth_GBps"] == 100.0

    def test_document_to_bundle_topology_name(self):
        doc = parse_topology_document(yaml.safe_load(EXAMPLE_PATH.read_text()))
        bundle = document_to_bundle(doc)
        assert bundle["topology_name"] == "Worker Pools with HBM Caches"
