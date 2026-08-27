"""Tests for PCIe DMA, TLP, and FLIT models (Parts 7–9)."""

import pytest

from model.interconnect.pcie import (
    DMAEngine,
    DMAOperation,
    PCIeConfig,
    PCIeGeneration,
    PCIeMPS,
    PCIeWidth,
    compute_dma_transfer,
    compute_transfer_on_wire_bytes,
    fragment_transfer,
    tlp_summary,
)
from model.interconnect.pcie.flit import (
    FLIT_CRC_BYTES,
    FLIT_DLP_BYTES,
    FLIT_FEC_BYTES,
    FLIT_SIZE_BYTES,
    FLIT_TLP_PAYLOAD_AREA_BYTES,
    GEN6_FLIT_EFFICIENCY,
    apply_flit_overhead,
    flit_count_for_bytes,
    flit_summary,
)
from model.topology.presets import (
    ARCHITECTURE_PRESETS,
    build_preset,
    preset_worker_pools_hbm,
)


class TestTLP:
    def test_fragment_transfer_4096_mps256(self):
        tlps = fragment_transfer(4096, PCIeMPS.B256)
        assert len(tlps) == 16
        assert sum(t.payload_bytes for t in tlps) == 4096

    def test_tlp_header_increases_on_wire_bytes(self):
        summary = tlp_summary(4096, PCIeMPS.B256)
        assert summary["on_wire_bytes"] > 4096
        assert summary["tlp_count"] == 16

    def test_read_has_request_and_completion_overhead(self):
        config = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X16)
        write_bytes = compute_transfer_on_wire_bytes(config, 4096, is_read=False)
        read_bytes = compute_transfer_on_wire_bytes(config, 4096, is_read=True)
        assert read_bytes > write_bytes


class TestFLIT:
    def test_gen6_flit_layout_constants(self):
        assert FLIT_SIZE_BYTES == 256
        assert FLIT_TLP_PAYLOAD_AREA_BYTES == 236
        assert FLIT_DLP_BYTES == 6
        assert FLIT_CRC_BYTES == 8
        assert FLIT_FEC_BYTES == 6
        assert FLIT_TLP_PAYLOAD_AREA_BYTES + FLIT_DLP_BYTES + FLIT_CRC_BYTES + FLIT_FEC_BYTES == 256
        assert GEN6_FLIT_EFFICIENCY == pytest.approx(236 / 256)

    def test_gen6_applies_flit_overhead(self):
        config = PCIeConfig(generation=PCIeGeneration.GEN6, width=PCIeWidth.X16)
        tlp_bytes = compute_transfer_on_wire_bytes(config, 4096, is_read=False)
        flit_bytes = apply_flit_overhead(tlp_bytes, config.generation)
        assert flit_bytes >= tlp_bytes
        summary = flit_summary(tlp_bytes)
        assert summary["flit_count"] >= 1
        assert summary["flit_size_bytes"] == FLIT_SIZE_BYTES
        assert summary["eta_flit"] == pytest.approx(236 / 256)

    def test_512_byte_tlp_spans_multiple_flits(self):
        assert flit_count_for_bytes(512) == 3
        assert flit_count_for_bytes(236) == 1
        assert flit_count_for_bytes(237) == 2

    def test_non_gen6_skips_flit(self):
        assert apply_flit_overhead(5000, PCIeGeneration.GEN4) == 5000


class TestDMA:
    def test_dma_read_exceeds_write(self):
        config = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X16)
        engine = DMAEngine()
        payload = 65536
        read = compute_dma_transfer(config, engine, payload, DMAOperation.DMA_READ)
        write = compute_dma_transfer(config, engine, payload, DMAOperation.DMA_WRITE)
        assert read.total_time_us > write.total_time_us
        assert read.request_tlps > 0
        assert read.completion_tlps > 0
        assert write.completion_tlps == 0
        assert read.label == "ANALYTICAL ESTIMATE"

    def test_dma_does_not_bypass_pcie(self):
        config = PCIeConfig(generation=PCIeGeneration.GEN4, width=PCIeWidth.X16)
        engine = DMAEngine()
        result = compute_dma_transfer(config, engine, 4096, DMAOperation.DMA_WRITE)
        assert result.pcie_latency_us > 0
        assert result.tlp_on_wire_bytes > 4096


class TestPresets:
    def test_all_architecture_presets_build(self):
        for name in ARCHITECTURE_PRESETS:
            topology, positions = build_preset(name)
            assert len(topology.nodes) >= 2
            assert len(positions) == len(topology.nodes)

    def test_worker_pools_connect_to_hbm(self):
        topology, _ = preset_worker_pools_hbm()
        assert "prefill_pool" in topology.nodes
        assert "prefill_hbm" in topology.nodes
        assert "decode_pool" in topology.nodes
        assert "decode_hbm" in topology.nodes
        path_prefill = topology.find_path("prefill_pool", "prefill_hbm")
        path_decode = topology.find_path("decode_pool", "decode_hbm")
        assert path_prefill is not None
        assert path_decode is not None
