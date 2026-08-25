"""Tests for component preset YAML and registry."""

from pathlib import Path

import pytest
import yaml

from model.components.registry import (
    default_node_config_dict,
    get_preset,
    list_presets,
    parse_node_config,
)
from model.topology.types import NodeType, normalize_node_type


COMPONENTS_DIR = Path(__file__).resolve().parent.parent / "configs" / "components"


class TestComponentPresets:
    def test_all_component_yaml_files_load(self):
        for path in sorted(COMPONENTS_DIR.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert "kind" in raw
            assert "presets" in raw
            assert len(raw["presets"]) >= 2, f"{path.name} should define at least two presets"

    @pytest.mark.parametrize(
        "kind",
        [
            "CPU",
            "GPU",
            "FPGA",
            "PREFILL_POOL",
            "DECODE_POOL",
            "HBM_MEMORY",
            "PCIE_ROOT_COMPLEX",
            "PCIE_SWITCH",
        ],
    )
    def test_node_presets_have_two_skus(self, kind: str):
        presets = list_presets(kind)
        assert len(presets) >= 2
        for preset_id in presets:
            data = get_preset(kind, preset_id)
            parse_node_config(kind, data)

    @pytest.mark.parametrize("kind", ["NVLink", "InfiniBand"])
    def test_edge_presets_have_two_skus(self, kind: str):
        presets = list_presets(kind)
        assert len(presets) >= 2

    def test_default_node_configs_cover_all_node_types(self):
        for node_type in NodeType:
            config = default_node_config_dict(node_type.value)
            assert isinstance(config, dict)
            assert config

    def test_memory_alias_normalizes_to_hbm(self):
        assert normalize_node_type("MEMORY") == NodeType.HBM_MEMORY.value
        config = default_node_config_dict("MEMORY")
        assert "capacity_gb" in config
