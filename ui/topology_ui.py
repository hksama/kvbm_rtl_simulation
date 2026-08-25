"""Streamlit UI for topology design and PCIe interconnect modeling."""

from __future__ import annotations

import math
import sys
import uuid
from pathlib import Path
from typing import Any

# Allow `streamlit run ui/topology_ui.py` from the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model.interconnect.pcie import (
    DMAEngine,
    DMAOperation,
    PCIeGeneration,
    PCIeMPS,
    PCIeMRRS,
    PCIeWidth,
    compute_dma_transfer,
    compute_pcie_bandwidth,
    compute_read_latency_us,
    compute_write_latency_us,
    fragment_transfer,
    flit_summary,
    tlp_summary,
)
from model.topology import Edge, InterconnectType, Node, NodeType, Topology
from ui.topology_state import (
    ARCHITECTURE_PRESETS,
    export_topology_yaml_text,
    get_bundle,
    get_default_dma_dict,
    get_default_pcie_dict,
    get_positions,
    get_topology,
    init_topology_session,
    load_preset,
    load_topology_from_text,
    pcie_config_from_dict,
    set_default_dma_dict,
    set_default_pcie_dict,
    sync_topology,
)
from model.topology.io import TopologyFormatError


NODE_TYPE_OPTIONS = {t.value: t for t in NodeType}
INTERCONNECT_OPTIONS = [InterconnectType.PCIE.value, InterconnectType.GENERIC.value]
GEN_OPTIONS = [g.value for g in PCIeGeneration]
WIDTH_OPTIONS = [w.value for w in PCIeWidth]
MPS_OPTIONS = [m.value for m in PCIeMPS]
MRRS_OPTIONS = [m.value for m in PCIeMRRS]

NODE_COLORS = {
    NodeType.CPU.value: "#636EFA",
    NodeType.GPU.value: "#00CC96",
    NodeType.FPGA.value: "#EF553B",
    NodeType.KVBM_ENGINE.value: "#AB63FA",
    NodeType.PREFILL_POOL.value: "#19D3F3",
    NodeType.DECODE_POOL.value: "#FF6692",
    NodeType.PCIE_ROOT_COMPLEX.value: "#FECB52",
    NodeType.PCIE_SWITCH.value: "#B6E880",
    NodeType.ROUTER.value: "#FF97FF",
    NodeType.MEMORY.value: "#C4B5A0",
    NodeType.NVSWITCH.value: "#76D7C4",
}


def _slug(name: str) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip()).strip("_")
    return base or f"node_{uuid.uuid4().hex[:6]}"


def render_pcie_config_form(
    data: dict[str, Any],
    *,
    key_prefix: str,
    title: str = "PCIe Link Parameters",
) -> dict[str, Any]:
    st.markdown(f"#### {title}")
    st.caption("All latency values are **ANALYTICAL ESTIMATE** — not measured hardware.")

    col1, col2, col3 = st.columns(3)
    with col1:
        generation = st.selectbox(
            "Generation",
            GEN_OPTIONS,
            index=GEN_OPTIONS.index(data.get("generation", "Gen4")),
            key=f"{key_prefix}_gen",
        )
        width = st.selectbox(
            "Lane width",
            WIDTH_OPTIONS,
            index=WIDTH_OPTIONS.index(data.get("width", "x16")),
            key=f"{key_prefix}_width",
        )
        mps = st.selectbox(
            "MPS (Max Payload Size)",
            MPS_OPTIONS,
            index=MPS_OPTIONS.index(int(data.get("mps", 256))),
            key=f"{key_prefix}_mps",
        )
    with col2:
        mrrs = st.selectbox(
            "MRRS (Max Read Request Size)",
            MRRS_OPTIONS,
            index=MRRS_OPTIONS.index(int(data.get("mrrs", 512))),
            key=f"{key_prefix}_mrrs",
        )
        base_latency = st.number_input(
            "Base link latency (µs)",
            min_value=0.0,
            max_value=50.0,
            value=float(data.get("base_latency_us", 0.5)),
            step=0.05,
            key=f"{key_prefix}_base_lat",
        )
        switch_hops = st.number_input(
            "Switch hops on link",
            min_value=0,
            max_value=8,
            value=int(data.get("switch_hops", 0)),
            step=1,
            key=f"{key_prefix}_hops",
        )
    with col3:
        protocol_eff = st.slider(
            "Protocol → application efficiency",
            min_value=0.5,
            max_value=1.0,
            value=float(data.get("protocol_efficiency", 0.95)),
            step=0.01,
            key=f"{key_prefix}_proto_eff",
        )
        congestion = st.slider(
            "Congestion factor",
            min_value=1.0,
            max_value=3.0,
            value=float(data.get("congestion_factor", 1.0)),
            step=0.05,
            key=f"{key_prefix}_cong",
        )
        switch_hop_lat = st.number_input(
            "Switch hop latency (µs)",
            min_value=0.0,
            max_value=5.0,
            value=float(data.get("switch_hop_latency_us", 0.15)),
            step=0.05,
            key=f"{key_prefix}_hop_lat",
        )

    with st.expander("Read / write latency components", expanded=False):
        lat1, lat2, lat3 = st.columns(3)
        with lat1:
            read_req = st.number_input(
                "Read request latency (µs)",
                min_value=0.0,
                max_value=10.0,
                value=float(data.get("read_request_latency_us", 0.2)),
                step=0.05,
                key=f"{key_prefix}_read_req",
            )
            memory_service = st.number_input(
                "Memory service latency (µs)",
                min_value=0.0,
                max_value=20.0,
                value=float(data.get("memory_service_latency_us", 0.5)),
                step=0.05,
                key=f"{key_prefix}_mem_svc",
            )
        with lat2:
            endpoint = st.number_input(
                "Endpoint processing (µs)",
                min_value=0.0,
                max_value=10.0,
                value=float(data.get("endpoint_processing_us", 0.1)),
                step=0.05,
                key=f"{key_prefix}_endpoint",
            )
            memory_proc = st.number_input(
                "Memory processing (µs)",
                min_value=0.0,
                max_value=10.0,
                value=float(data.get("memory_processing_us", 0.1)),
                step=0.05,
                key=f"{key_prefix}_mem_proc",
            )
        with lat3:
            cong_penalty = st.number_input(
                "Congestion base penalty (µs)",
                min_value=0.0,
                max_value=5.0,
                value=float(data.get("congestion_base_penalty_us", 0.05)),
                step=0.01,
                key=f"{key_prefix}_cong_pen",
            )

    updated = {
        "generation": generation,
        "width": width,
        "mps": int(mps),
        "mrrs": int(mrrs),
        "base_latency_us": base_latency,
        "switch_hops": int(switch_hops),
        "congestion_factor": congestion,
        "protocol_efficiency": protocol_eff,
        "switch_hop_latency_us": switch_hop_lat,
        "read_request_latency_us": read_req,
        "memory_service_latency_us": memory_service,
        "endpoint_processing_us": endpoint,
        "memory_processing_us": memory_proc,
        "congestion_base_penalty_us": cong_penalty,
    }

    config = pcie_config_from_dict(updated)
    breakdown = compute_pcie_bandwidth(config)
    bw1, bw2, bw3 = st.columns(3)
    with bw1:
        st.metric("RAW bandwidth (per direction)", f"{breakdown.raw_GBps:.2f} GB/s")
    with bw2:
        st.metric("PROTOCOL bandwidth", f"{breakdown.protocol_GBps:.2f} GB/s")
    with bw3:
        st.metric("EFFECTIVE bandwidth", f"{breakdown.effective_GBps:.2f} GB/s")

    return updated


def topology_graph_figure(
    topology: Topology,
    positions: dict[str, tuple[float, float]],
    *,
    highlight_path_node_ids: list[str] | None = None,
    highlight_edge_ids: list[str] | None = None,
) -> go.Figure:
    highlight_nodes = set(highlight_path_node_ids or [])
    highlight_edges = set(highlight_edge_ids or [])

    fig = go.Figure()
    seen_undirected: set[tuple[str, str]] = set()

    for edge in topology.edges.values():
        if not edge.enabled:
            continue
        pair = tuple(sorted((edge.source_id, edge.dest_id)))
        if pair in seen_undirected:
            continue
        seen_undirected.add(pair)

        src_pos = positions.get(edge.source_id)
        dst_pos = positions.get(edge.dest_id)
        if src_pos is None or dst_pos is None:
            continue

        is_highlight = edge.id in highlight_edges or f"{edge.id}_rev" in highlight_edges
        color = "#FFD700" if is_highlight else "#888888"
        width = 4 if is_highlight else 2
        label = f"{edge.generation} {edge.width} PCIe"

        fig.add_trace(
            go.Scatter(
                x=[src_pos[0], dst_pos[0]],
                y=[src_pos[1], dst_pos[1]],
                mode="lines",
                line={"color": color, "width": width},
                hovertext=label,
                hoverinfo="text",
                showlegend=False,
            )
        )

        mid_x = (src_pos[0] + dst_pos[0]) / 2
        mid_y = (src_pos[1] + dst_pos[1]) / 2
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            text=label,
            showarrow=False,
            font={"size": 9, "color": "#444444"},
            bgcolor="rgba(255,255,255,0.7)",
        )

    for node in topology.nodes.values():
        pos = positions.get(node.id, (0.0, 0.0))
        color = NODE_COLORS.get(node.type.value, "#777777")
        if node.metadata.get("memory_type") == "HBM":
            color = "#D4AC0D" if node.metadata.get("role") == "prefill_kv_cache" else "#E67E22"
        if node.id in highlight_nodes:
            color = "#FFD700"
        fig.add_trace(
            go.Scatter(
                x=[pos[0]],
                y=[pos[1]],
                mode="markers+text",
                marker={
                    "size": 28 if node.id in highlight_nodes else 22,
                    "color": color,
                    "line": {"width": 2, "color": "#222222"},
                },
                text=[node.name],
                textposition="top center",
                name=node.name,
                hovertemplate=(
                    f"<b>{node.name}</b><br>"
                    f"Type: {node.type.value}<br>"
                    f"ID: {node.id}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="System Topology (PCIe links shown as edges)",
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        height=520,
        showlegend=False,
        plot_bgcolor="#FAFAFA",
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def _auto_position(node_id: str, index: int, total: int) -> tuple[float, float]:
    angle = 2 * math.pi * index / max(total, 1)
    return (220 * math.cos(angle), 180 * math.sin(angle))


def render_topology_designer_tab() -> None:
    init_topology_session()
    topology = get_topology()
    positions = get_positions()

    st.markdown("### Topology Designer")
    st.caption(
        "Build a system graph by adding nodes and PCIe (or generic) links. "
        "Load an architecture preset, then customize links and PCIe parameters."
    )

    preset_col, action_col = st.columns([2, 1])
    with preset_col:
        preset_name = st.selectbox("Architecture preset", list(ARCHITECTURE_PRESETS.keys()))
        if st.button("Load preset into editor", type="primary"):
            load_preset(preset_name)
            st.rerun()
    with action_col:
        if st.button("Reset to Architecture A"):
            load_preset("Architecture A — CPU ↔ FPGA (direct PCIe)")
            st.rerun()

    with st.expander("Topology YAML — load / export", expanded=False):
        st.caption(
            "Declarative format (`schema_version: 1`) covering nodes, links, interconnect types, "
            "PCIe config, layout positions, and default PCIe/DMA settings."
        )
        example_path = _ROOT / "configs" / "topologies" / "worker_pools_hbm.yaml"
        if example_path.exists():
            st.markdown(f"Example file: `{example_path.relative_to(_ROOT)}`")

        if st.button("Fill editor with example YAML", key="fill_topology_yaml_example"):
            if example_path.exists():
                st.session_state["topology_yaml_input"] = example_path.read_text(encoding="utf-8")
                st.rerun()
            else:
                st.error(f"Example file not found: {example_path}")

        yaml_input = st.text_area(
            "Paste topology YAML",
            height=220,
            placeholder="schema_version: 1\ntopology:\n  name: My Topology\n...",
            key="topology_yaml_input",
        )
        ycol1, ycol2 = st.columns(2)
        with ycol1:
            if st.button("Load YAML into simulator"):
                try:
                    load_topology_from_text(yaml_input)
                    st.success("Topology loaded.")
                    st.rerun()
                except TopologyFormatError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Failed to parse YAML: {exc}")
        with ycol2:
            exported = export_topology_yaml_text()
            st.download_button(
                "Download current topology YAML",
                data=exported,
                file_name="topology.yaml",
                mime="text/yaml",
            )

    graph_col, control_col = st.columns([1.4, 1])

    with control_col:
        st.markdown("#### Nodes")
        with st.form("create_node_form", clear_on_submit=True):
            node_name = st.text_input("Node name", value="GPU 0")
            node_type = st.selectbox("Node type", list(NODE_TYPE_OPTIONS.keys()))
            node_id_input = st.text_input("Node ID (optional)", value="")
            submitted = st.form_submit_button("Create node")
            if submitted:
                node_id = node_id_input.strip() or _slug(node_name)
                if node_id in topology.nodes:
                    st.error(f"Node ID `{node_id}` already exists.")
                else:
                    topology.add_node(
                        Node(id=node_id, type=NODE_TYPE_OPTIONS[node_type], name=node_name)
                    )
                    positions[node_id] = _auto_position(node_id, len(topology.nodes), len(topology.nodes))
                    sync_topology(topology, positions)
                    st.rerun()

        if topology.nodes:
            delete_node = st.selectbox(
                "Delete node",
                options=list(topology.nodes.keys()),
                format_func=lambda nid: topology.nodes[nid].name,
            )
            if st.button("Delete selected node"):
                topology.remove_node(delete_node)
                positions.pop(delete_node, None)
                sync_topology(topology, positions)
                st.rerun()

        st.markdown("#### Node layout")
        st.caption("Adjust canvas positions (drag-and-drop is not available in Streamlit; use coordinates).")
        if topology.nodes:
            layout_node = st.selectbox(
                "Edit position",
                options=list(topology.nodes.keys()),
                format_func=lambda nid: topology.nodes[nid].name,
                key="layout_node_select",
            )
            pos = positions.get(layout_node, (0.0, 0.0))
            new_x = st.number_input("X", value=float(pos[0]), step=10.0, key="layout_x")
            new_y = st.number_input("Y", value=float(pos[1]), step=10.0, key="layout_y")
            if st.button("Apply position"):
                positions[layout_node] = (new_x, new_y)
                sync_topology(topology, positions)
                st.rerun()

        st.markdown("#### Connect with PCIe / link")
        if len(topology.nodes) >= 2:
            node_ids = list(topology.nodes.keys())
            source = st.selectbox(
                "Source",
                node_ids,
                format_func=lambda nid: topology.nodes[nid].name,
                key="conn_source",
            )
            dest = st.selectbox(
                "Destination",
                [nid for nid in node_ids if nid != source],
                format_func=lambda nid: topology.nodes[nid].name,
                key="conn_dest",
            )
            interconnect = st.selectbox("Interconnect type", INTERCONNECT_OPTIONS, key="conn_type")
            bidirectional = st.checkbox("Bidirectional link", value=True)
            edge_name = st.text_input("Link ID", value=f"{source}_{dest}")

            if st.button("Create connection"):
                default_pcie = get_default_pcie_dict()
                if interconnect == InterconnectType.PCIE.value:
                    config = pcie_config_from_dict(default_pcie)
                    topology.connect_pcie(edge_name, source, dest, config, bidirectional=bidirectional)
                else:
                    config = pcie_config_from_dict(default_pcie)
                    bw = compute_pcie_bandwidth(config).effective_GBps
                    topology.add_edge(
                        Edge(
                            id=edge_name,
                            source_id=source,
                            dest_id=dest,
                            interconnect_type=InterconnectType.GENERIC,
                            bandwidth_GBps=bw,
                            latency_us=config.base_latency_us,
                        )
                    )
                sync_topology(topology, positions)
                st.rerun()
        else:
            st.info("Add at least two nodes to create a connection.")

        st.markdown("#### Links")
        if topology.edges:
            edge_rows = []
            for edge in topology.edges.values():
                src = topology.nodes.get(edge.source_id)
                dst = topology.nodes.get(edge.dest_id)
                edge_rows.append({
                    "Link ID": edge.id,
                    "From": src.name if src else edge.source_id,
                    "To": dst.name if dst else edge.dest_id,
                    "Type": edge.interconnect_type.value,
                    "Gen": edge.generation,
                    "Width": edge.width,
                    "BW (GB/s)": f"{edge.bandwidth_GBps:.2f}",
                })
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

            selected_edge = st.selectbox("Select link to edit / delete", [e.id for e in topology.edges.values()])
            if st.button("Delete selected link"):
                topology.remove_edge(selected_edge)
                rev_id = f"{selected_edge}_rev"
                if rev_id in topology.edges:
                    topology.remove_edge(rev_id)
                sync_topology(topology, positions)
                st.rerun()
        else:
            st.info("No links yet.")

    with graph_col:
        st.plotly_chart(
            topology_graph_figure(topology, positions),
            use_container_width=True,
            key="topology_graph_main",
        )

        if topology.nodes:
            st.markdown("#### Node list")
            st.dataframe(
                pd.DataFrame([
                    {"ID": n.id, "Name": n.name, "Type": n.type.value}
                    for n in topology.nodes.values()
                ]),
                use_container_width=True,
                hide_index=True,
            )


def render_interconnect_tab() -> None:
    init_topology_session()
    topology = get_topology()
    default_pcie = get_default_pcie_dict()

    st.markdown("### Interconnect / PCIe Configuration")
    st.caption(
        "Configure default PCIe parameters for **new links**. "
        "Per-link overrides are applied on the selected link below."
    )

    updated_default = render_pcie_config_form(default_pcie, key_prefix="default_pcie", title="Default PCIe parameters")
    if st.button("Save as default for new links"):
        set_default_pcie_dict(updated_default)
        st.success("Default PCIe configuration saved.")

    st.divider()
    st.markdown("#### Per-link PCIe overrides")

    pcie_edges = [
        e for e in topology.edges.values()
        if e.interconnect_type == InterconnectType.PCIE and not e.id.endswith("_rev")
    ]
    if not pcie_edges:
        st.info("No PCIe links in the current topology. Create links in the Topology Designer tab.")
        return

    edge = st.selectbox(
        "PCIe link",
        [e.id for e in pcie_edges],
        format_func=lambda eid: _edge_label(topology, eid),
    )
    bundle = get_bundle()
    current = bundle["pcie_configs"].get(edge, updated_default)
    per_link = render_pcie_config_form(
        current,
        key_prefix=f"edge_{edge}",
        title=f"Parameters for `{_edge_label(topology, edge)}`",
    )

    if st.button("Apply to selected link"):
        config = pcie_config_from_dict(per_link)
        selected = topology.edges[edge]
        topology.add_edge(selected, pcie_config=config)
        rev_id = f"{edge}_rev"
        if rev_id in topology.edges:
            topology.add_edge(topology.edges[rev_id], pcie_config=config)
        sync_topology(topology, get_positions())
        st.success("Link updated.")
        st.rerun()

    st.markdown("#### All PCIe links — bandwidth summary")
    rows = []
    for e in pcie_edges:
        cfg = topology.pcie_configs.get(e.id)
        if cfg is None:
            continue
        bw = compute_pcie_bandwidth(cfg)
        rows.append({
            "Link": _edge_label(topology, e.id),
            "Generation": cfg.generation.value,
            "Width": cfg.width.value,
            "RAW (GB/s)": f"{bw.raw_GBps:.2f}",
            "PROTOCOL (GB/s)": f"{bw.protocol_GBps:.2f}",
            "EFFECTIVE (GB/s)": f"{bw.effective_GBps:.2f}",
            "Switch hops": cfg.switch_hops,
            "MPS / MRRS": f"{cfg.mps.value} / {cfg.mrrs.value}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _edge_label(topology: Topology, edge_id: str) -> str:
    edge = topology.edges[edge_id]
    src = topology.nodes.get(edge.source_id)
    dst = topology.nodes.get(edge.dest_id)
    return f"{src.name if src else edge.source_id} → {dst.name if dst else edge.dest_id} ({edge_id})"


def render_path_inspector_tab() -> None:
    init_topology_session()
    topology = get_topology()
    positions = get_positions()

    st.markdown("### Path Inspector")
    st.caption("Select source and destination to analyze PCIe path latency and bandwidth.")

    if len(topology.nodes) < 2:
        st.warning("Add at least two nodes in the Topology Designer.")
        return

    node_ids = list(topology.nodes.keys())
    col1, col2, col3 = st.columns(3)
    with col1:
        source = st.selectbox(
            "Source",
            node_ids,
            format_func=lambda nid: topology.nodes[nid].name,
            key="path_source",
        )
    with col2:
        dest_options = [nid for nid in node_ids if nid != source]
        destination = st.selectbox(
            "Destination",
            dest_options,
            format_func=lambda nid: topology.nodes[nid].name,
            key="path_dest",
        )
    with col3:
        payload_mb = st.slider("Payload size (MB)", min_value=0.0, max_value=256.0, value=4.0, step=0.5)
        payload_bytes = int(payload_mb * 1024 * 1024)

    traffic = st.radio("Traffic type", ["PCIe Write", "PCIe Read"], horizontal=True)
    is_read = traffic == "PCIe Read"

    path = topology.find_path(source, destination)
    if path is None:
        st.error("No path found between selected nodes.")
        return

    summary = topology.path_summary(path, payload_bytes, is_read=is_read)

    st.plotly_chart(
        topology_graph_figure(
            topology,
            positions,
            highlight_path_node_ids=[n.id for n in path.nodes],
            highlight_edge_ids=[e.id for e in path.edges],
        ),
        use_container_width=True,
        key="topology_graph_path",
    )

    st.markdown("#### Path")
    for index, node in enumerate(path.nodes):
        st.markdown(f"{index + 1}. **{node.name}** `{node.type.value}`")
        if index < len(path.edges):
            edge = path.edges[index]
            st.markdown(f"   ↓ `{edge.generation}` `{edge.width}` {edge.interconnect_type.value} ({edge.bandwidth_GBps:.2f} GB/s EFFECTIVE)")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Hop count", summary.hop_count)
    with m2:
        st.metric("Bottleneck bandwidth", f"{summary.bottleneck_bandwidth_GBps:.2f} GB/s")
    with m3:
        st.metric("Transfer time", f"{summary.transfer_time_us:.2f} µs")
    with m4:
        st.metric("Label", summary.latency_label)

    m5, m6, m7 = st.columns(3)
    with m5:
        st.metric("Fixed + hop latency", f"{summary.latency_us:.2f} µs")
    with m6:
        st.metric("Serialization", f"{summary.serialization_time_us:.2f} µs")
    with m7:
        st.metric("Congestion penalty", f"{summary.congestion_penalty_us:.2f} µs")

    st.markdown("#### Per-hop PCIe detail")
    hop_rows = []
    for edge in path.edges:
        cfg = topology.pcie_configs.get(edge.id)
        if cfg is None:
            hop_rows.append({
                "Hop": _edge_label(topology, edge.id),
                "Type": edge.interconnect_type.value,
                "Effective BW": f"{edge.bandwidth_GBps:.2f} GB/s",
                "Latency": f"{edge.latency_us:.2f} µs",
            })
            continue
        if is_read:
            lat = compute_read_latency_us(cfg, payload_bytes)
        else:
            lat = compute_write_latency_us(cfg, payload_bytes)
        bw = compute_pcie_bandwidth(cfg)
        hop_rows.append({
            "Hop": _edge_label(topology, edge.id),
            "RAW / PROTO / EFF (GB/s)": f"{bw.raw_GBps:.1f} / {bw.protocol_GBps:.1f} / {bw.effective_GBps:.1f}",
            "Transfer (µs)": f"{lat.transfer_time_us:.2f}",
            "Serialization (µs)": f"{lat.serialization_time_us:.2f}",
            "Switch hops": cfg.switch_hops,
        })
    st.dataframe(pd.DataFrame(hop_rows), use_container_width=True, hide_index=True)


def render_dma_tlp_flit_tab() -> None:
    init_topology_session()
    default_pcie = get_default_pcie_dict()
    default_dma = get_default_dma_dict()
    pcie = pcie_config_from_dict(default_pcie)

    st.markdown("### DMA / TLP / FLIT Analysis")
    st.caption(
        "DMA operates through PCIe (never bypassing the link). "
        "TLP fragmentation and Gen6 FLIT packing affect serialization."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### DMA Engine")
        queue_depth = st.number_input("Queue depth", 1, 4096, int(default_dma["queue_depth"]), key="dma_qd")
        max_outstanding = st.number_input(
            "Max outstanding reads", 1, 256, int(default_dma["max_outstanding_reads"]), key="dma_mor"
        )
        request_size = st.number_input("Request size (B)", 64, 65536, int(default_dma["request_size"]), key="dma_req")
        completion_size = st.number_input(
            "Completion size (B)", 64, 65536, int(default_dma["completion_size"]), key="dma_cpl"
        )
        read_lat = st.number_input(
            "DMA read overhead (µs)", 0.0, 10.0, float(default_dma["read_latency_us"]), key="dma_rlat"
        )
        write_lat = st.number_input(
            "DMA write overhead (µs)", 0.0, 10.0, float(default_dma["write_latency_us"]), key="dma_wlat"
        )
        if st.button("Save DMA engine defaults"):
            set_default_dma_dict({
                "queue_depth": int(queue_depth),
                "max_outstanding_reads": int(max_outstanding),
                "request_size": int(request_size),
                "completion_size": int(completion_size),
                "read_latency_us": float(read_lat),
                "write_latency_us": float(write_lat),
            })
            st.success("DMA defaults saved.")

    with col2:
        st.markdown("#### Transfer under test")
        payload_mb = st.slider("Payload (MB)", 0.0, 64.0, 4.0, 0.5, key="dma_payload_mb")
        payload_bytes = int(payload_mb * 1024 * 1024)
        operation = st.radio("DMA operation", ["DMA_WRITE", "DMA_READ"], horizontal=True, key="dma_op")
        requester = st.text_input("Requester", value="KVBM FPGA", key="dma_requester")
        completer = st.text_input("Completer / Memory", value="Prefill HBM Cache", key="dma_completer")

    engine = DMAEngine(
        queue_depth=int(queue_depth),
        max_outstanding_reads=int(max_outstanding),
        request_size=int(request_size),
        completion_size=int(completion_size),
        read_latency_us=float(read_lat),
        write_latency_us=float(write_lat),
    )
    dma_op = DMAOperation.DMA_READ if operation == "DMA_READ" else DMAOperation.DMA_WRITE
    dma_result = compute_dma_transfer(
        pcie, engine, payload_bytes, dma_op, requester=requester, completer=completer
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total DMA time", f"{dma_result.total_time_us:.2f} µs")
    with m2:
        st.metric("PCIe latency", f"{dma_result.pcie_latency_us:.2f} µs")
    with m3:
        st.metric("TLP on-wire bytes", f"{dma_result.tlp_on_wire_bytes:,}")
    with m4:
        st.metric("Serialized bytes", f"{dma_result.serialized_bytes:,}")

    st.markdown("#### TLP fragmentation")
    if operation == "DMA_WRITE":
        tlps = fragment_transfer(payload_bytes, pcie.mps)
        tlp_info = tlp_summary(payload_bytes, pcie.mps)
    else:
        from model.interconnect.pcie import fragment_read_transfer
        reqs, cpls = fragment_read_transfer(payload_bytes, pcie.mps, pcie.mrrs.value)
        tlps = reqs + cpls
        tlp_info = {
            "tlp_count": len(reqs) + len(cpls),
            "on_wire_bytes": dma_result.tlp_on_wire_bytes,
            "header_overhead_bytes": dma_result.tlp_on_wire_bytes - payload_bytes,
        }

    tlp_df = pd.DataFrame([
        {"Field": "Payload bytes", "Value": payload_bytes},
        {"Field": "TLP count", "Value": tlp_info.get("tlp_count", len(tlps))},
        {"Field": "MPS", "Value": pcie.mps.value},
        {"Field": "MRRS", "Value": pcie.mrrs.value},
        {"Field": "On-wire bytes (TLP)", "Value": dma_result.tlp_on_wire_bytes},
        {"Field": "Header overhead (B)", "Value": tlp_info.get("header_overhead_bytes", 0)},
        {"Field": "Request TLPs", "Value": dma_result.request_tlps},
        {"Field": "Completion TLPs", "Value": dma_result.completion_tlps},
    ])
    st.dataframe(tlp_df, use_container_width=True, hide_index=True)

    if pcie.generation == PCIeGeneration.GEN6:
        st.markdown("#### Gen6 FLIT packing")
        flit_info = flit_summary(dma_result.tlp_on_wire_bytes)
        flit_df = pd.DataFrame([
            {"Field": "FLIT size (B)", "Value": flit_info["flit_size_bytes"]},
            {"Field": "FLIT count", "Value": flit_info["flit_count"]},
            {"Field": "FLIT on-wire bytes", "Value": flit_info["flit_on_wire_bytes"]},
            {"Field": "FLIT overhead (B)", "Value": flit_info["flit_overhead_bytes"]},
            {"Field": "FLIT efficiency", "Value": f"{flit_info['efficiency']:.3f}"},
        ])
        st.dataframe(flit_df, use_container_width=True, hide_index=True)
    else:
        st.info("Select **Gen6** in default PCIe settings to see FLIT packing analysis.")

    st.markdown("#### DMA latency breakdown")
    st.dataframe(
        pd.DataFrame([
            {"Component": "PCIe transfer", "Time (µs)": f"{dma_result.pcie_latency_us:.3f}", "Label": dma_result.label},
            {"Component": "DMA engine overhead", "Time (µs)": f"{dma_result.dma_overhead_us:.3f}", "Label": dma_result.label},
            {"Component": "Queue penalty", "Time (µs)": f"{dma_result.queue_penalty_us:.3f}", "Label": dma_result.label},
            {"Component": "Total", "Time (µs)": f"{dma_result.total_time_us:.3f}", "Label": dma_result.label},
        ]),
        use_container_width=True,
        hide_index=True,
    )


def render_topology_section() -> None:
    tab_topo, tab_pcie, tab_path, tab_dma = st.tabs([
        "Topology Designer",
        "Interconnect / PCIe",
        "Path Inspector",
        "DMA / TLP / FLIT",
    ])
    with tab_topo:
        render_topology_designer_tab()
    with tab_pcie:
        render_interconnect_tab()
    with tab_path:
        render_path_inspector_tab()
    with tab_dma:
        render_dma_tlp_flit_tab()


def main() -> None:
    st.set_page_config(
        page_title="KVBM Topology & PCIe Designer",
        page_icon="🔗",
        layout="wide",
    )
    st.title("Topology & PCIe Designer")
    st.caption(
        "Design system topologies, configure PCIe links, and inspect transfer paths. "
        "All bandwidth/latency values are analytical estimates."
    )
    render_topology_section()


if __name__ == "__main__":
    main()
