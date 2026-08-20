"""Streamlit UI for KV metadata placement engine design-space exploration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from model.checkpoint1 import Checkpoint1Config, Checkpoint1Simulator, find_crossover_rps
from model.checkpoint2 import Checkpoint2Simulator
from model.types import HotnessModelType, SimulationResult

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "assumptions.yaml"

CHECKPOINT_1 = "Checkpoint 1: Single Prefill GPU"
CHECKPOINT_2 = "Checkpoint 2: Prefill Scaling"
CHECKPOINT_OPTIONS = [CHECKPOINT_1, CHECKPOINT_2]
PREFILL_GPU_OPTIONS_CP2 = [2, 4, 8, 16]

WORKLOAD_PRESETS = {
    "Manual": None,
    "Light": {"requests_per_sec": 500, "metadata_ops_per_request": 8, "burstiness": 1.0},
    "Medium": {"requests_per_sec": 2000, "metadata_ops_per_request": 16, "burstiness": 1.0},
    "Heavy": {"requests_per_sec": 8000, "metadata_ops_per_request": 32, "burstiness": 1.2},
    "Bursty": {"requests_per_sec": 3000, "metadata_ops_per_request": 16, "burstiness": 2.5},
    "Cache-reuse heavy": {
        "requests_per_sec": 2000,
        "prefix_reuse_fraction": 0.7,
        "prefix_reuse_pct": 0.8,
        "metadata_ops_per_request": 12,
    },
    "Low cache-reuse": {
        "requests_per_sec": 2000,
        "prefix_reuse_fraction": 0.05,
        "prefix_reuse_pct": 0.1,
        "metadata_ops_per_request": 20,
    },
}

HOTNESS_OPTIONS = {
    "Weighted linear": HotnessModelType.WEIGHTED_LINEAR,
    "LRU-like": HotnessModelType.LRU_LIKE,
    "LFU-like": HotnessModelType.LFU_LIKE,
    "Recency-weighted": HotnessModelType.RECENCY_WEIGHTED,
    "Exponentially decayed": HotnessModelType.EXP_DECAY,
}

CANDIDATE_K_OPTIONS = [1, 2, 4, 8, 16, 32, 64]
CLOCK_OPTIONS = [250, 500, 750, 1000]
PARALLEL_ENGINES = [1, 2, 4, 8, 16]


def load_default_config() -> Checkpoint1Config:
    if CONFIG_PATH.exists():
        return Checkpoint1Config.from_yaml(CONFIG_PATH)
    return Checkpoint1Config()


class SimulatorProtocol(Protocol):
    def run(self) -> SimulationResult: ...
    def sweep_rps(self, rps_values: list[float]) -> list[tuple[float, SimulationResult]]: ...
    def sweep_metadata_entries(self, entry_values: list[int]) -> list[tuple[int, SimulationResult]]: ...
    def sweep_candidate_k(self, k_values: list[int]) -> list[tuple[int, SimulationResult]]: ...


def create_simulator(checkpoint: str, config: Checkpoint1Config) -> SimulatorProtocol:
    if checkpoint == CHECKPOINT_2:
        return Checkpoint2Simulator(config)
    return Checkpoint1Simulator(config)


def run_simulation(checkpoint: str, config: Checkpoint1Config) -> SimulationResult:
    return create_simulator(checkpoint, config).run()


def checkpoint_description(checkpoint: str) -> str:
    if checkpoint == CHECKPOINT_2:
        return "**Checkpoint 2:** 2–16 Prefill GPUs · 1 RTL Engine · Prefill Scaling"
    return "**Checkpoint 1:** 1 Prefill GPU · 1 RTL Engine · Hotness Tracker Focus"


def render_summary(result, config: Checkpoint1Config, crossover_rps: float | None):
    region = result.region
    region_color = {
        "HARDWARE WORSE": "🔴",
        "BREAK-EVEN": "🟡",
        "HARDWARE BETTER": "🟢",
    }.get(region, "")

    st.markdown(f"### Output Summary {region_color} **{region}**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Software ops/s", f"{result.software_throughput.ops_per_sec:,.0f}")
        st.metric("Software latency", f"{result.software_latency_us:.2f} µs")
    with col2:
        st.metric("RTL ops/s", f"{result.hardware_throughput.ops_per_sec:,.0f}")
        st.metric("RTL latency", f"{result.hardware_latency_us:.2f} µs")
    with col3:
        st.metric("Speedup", f"{result.speedup:.2f}×")
        st.metric("RTL utilization", f"{result.rtl_utilization * 100:.1f}%")
    with col4:
        st.metric("Metadata ops/s", f"{result.metadata_ops_per_sec:,.0f}")
        st.metric("Metadata capacity", f"{result.metadata_capacity:,} entries")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Coordination overhead", f"{result.coordination_overhead_us:.2f} µs")
    with col6:
        st.metric("HW cycles / hotness update", f"{result.hw_cycles_per_hotness_update:.1f}")
    with col7:
        crossover_str = f"{crossover_rps:,.0f} RPS" if crossover_rps else "N/A"
        st.metric("Hardware crossover", crossover_str)
    with col8:
        st.metric("Prefill GPUs / RTL engines", f"{config.prefill_gpus} / {config.rtl_engines}")


def latency_breakdown_chart(sw_breakdown, hw_breakdown) -> go.Figure:
    components = ["lookup", "metadata_update", "score", "candidate_selection", "coordination", "queue"]
    sw_vals = [getattr(sw_breakdown, c) for c in components]
    hw_vals = [getattr(hw_breakdown, c) for c in components]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Software", x=components, y=sw_vals, marker_color="#636EFA"))
    fig.add_trace(go.Bar(name="RTL", x=components, y=hw_vals, marker_color="#EF553B"))
    fig.update_layout(
        title="Latency Breakdown (µs)",
        barmode="group",
        yaxis_title="Latency (µs)",
        height=400,
    )
    return fig


def speedup_vs_rps_chart(sweep_data: list) -> go.Figure:
    rps = [d[0] for d in sweep_data]
    speedup = [d[1].speedup for d in sweep_data]
    sw_lat = [d[1].software_latency_us for d in sweep_data]
    hw_lat = [d[1].hardware_latency_us for d in sweep_data]
    util = [d[1].rtl_utilization * 100 for d in sweep_data]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=rps, y=speedup, name="Speedup", line=dict(color="#00CC96", width=3)))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Break-even")
    fig.add_trace(
        go.Scatter(x=rps, y=sw_lat, name="Software latency", line=dict(color="#636EFA", dash="dot")),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=rps, y=hw_lat, name="RTL latency", line=dict(color="#EF553B", dash="dot")),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=rps, y=util, name="RTL utilization %", line=dict(color="#AB63FA", dash="dash")),
        secondary_y=True,
    )
    fig.update_xaxes(title_text="Requests/sec")
    fig.update_yaxes(title_text="Speedup (SW/HW)", secondary_y=False)
    fig.update_yaxes(title_text="Latency (µs) / Utilization (%)", secondary_y=True)
    fig.update_layout(title="RPS vs Speedup / Latency / Utilization", height=450)
    return fig


def metadata_entries_speedup_chart(sweep_data: list) -> go.Figure:
    entries = [d[0] for d in sweep_data]
    speedup = [d[1].speedup for d in sweep_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=entries, y=speedup, mode="lines+markers", line=dict(color="#00CC96")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Metadata Entries vs Speedup",
        xaxis_title="Metadata entries",
        yaxis_title="Speedup",
        xaxis_type="log",
        height=400,
    )
    return fig


def candidate_k_latency_chart(sweep_data: list) -> go.Figure:
    k_vals = [d[0] for d in sweep_data]
    sw_lat = [d[1].software_latency_us for d in sweep_data]
    hw_lat = [d[1].hardware_latency_us for d in sweep_data]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k_vals, y=sw_lat, name="Software", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=k_vals, y=hw_lat, name="RTL", mode="lines+markers"))
    fig.update_layout(
        title="Candidate Count K vs Latency",
        xaxis_title="Candidate count K",
        xaxis_type="log",
        yaxis_title="Decision latency (µs)",
        height=400,
    )
    return fig


def hotness_distribution_chart(hotness_dist) -> go.Figure:
    hotness_vals = [b.hotness for b in hotness_dist]
    classifications = [b.classification for b in hotness_dist]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Hotness Distribution", "Hot/Cold Classification"),
        specs=[[{"type": "xy"}, {"type": "domain"}]],
    )

    fig.add_trace(
        go.Histogram(x=hotness_vals, nbinsx=30, marker_color="#00CC96", name="Hotness"),
        row=1,
        col=1,
    )

    hot_count = classifications.count("hot")
    cold_count = classifications.count("cold")
    fig.add_trace(
        go.Pie(
            labels=["Hot", "Cold"],
            values=[hot_count, cold_count],
            marker={"colors": ["#EF553B", "#636EFA"]},
            hole=0.35,
        ),
        row=1,
        col=2,
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def prefill_gpu_utilization_chart(rows: list[dict[str, Any]], selected_gpus: int) -> go.Figure:
    gpu_counts = [r["prefill_gpus"] for r in rows]
    utils = [r["rtl_utilization"] * 100 for r in rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gpu_counts, y=utils, mode="lines+markers", name="RTL utilization",
            line={"color": "#AB63FA", "width": 3}, marker={"size": 10},
        )
    )
    for r in rows:
        if r["prefill_gpus"] == selected_gpus:
            fig.add_trace(go.Scatter(
                x=[selected_gpus], y=[r["rtl_utilization"] * 100], mode="markers",
                name="Selected", marker={"size": 16, "color": "#EF553B", "symbol": "star"},
            ))
            break
    fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Saturation")
    fig.update_layout(
        title="Prefill GPUs vs RTL Utilization",
        xaxis={"title": "Prefill GPUs", "type": "log", "dtick": 1},
        yaxis={"title": "RTL Utilization (%)", "range": [0, 105]},
        height=420,
    )
    return fig


def prefill_gpu_latency_chart(rows: list[dict[str, Any]], selected_gpus: int) -> go.Figure:
    gpu_counts = [r["prefill_gpus"] for r in rows]
    sw_lat = [r["software_latency_us"] for r in rows]
    hw_lat = [r["hardware_latency_us"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gpu_counts, y=sw_lat, mode="lines+markers", name="Software latency", line={"color": "#636EFA"}))
    fig.add_trace(go.Scatter(x=gpu_counts, y=hw_lat, mode="lines+markers", name="RTL latency", line={"color": "#EF553B"}))
    for r in rows:
        if r["prefill_gpus"] == selected_gpus:
            fig.add_trace(go.Scatter(
                x=[selected_gpus, selected_gpus],
                y=[r["software_latency_us"], r["hardware_latency_us"]],
                mode="markers", name="Selected",
                marker={"size": 14, "color": "#00CC96", "symbol": "diamond"},
            ))
            break
    fig.update_layout(
        title="Prefill GPUs vs Decision Latency",
        xaxis={"title": "Prefill GPUs", "type": "log", "dtick": 1},
        yaxis={"title": "Decision latency (µs)"},
        height=420,
    )
    return fig


def prefill_gpu_metadata_ops_chart(rows: list[dict[str, Any]], selected_gpus: int) -> go.Figure:
    gpu_counts = [r["prefill_gpus"] for r in rows]
    ops = [r["metadata_ops_per_sec"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gpu_counts, y=ops, mode="lines+markers", name="Metadata ops/s",
        line={"color": "#00CC96", "width": 3}, fill="tozeroy",
    ))
    for r in rows:
        if r["prefill_gpus"] == selected_gpus:
            fig.add_trace(go.Scatter(
                x=[selected_gpus], y=[r["metadata_ops_per_sec"]], mode="markers",
                name="Selected", marker={"size": 16, "color": "#EF553B", "symbol": "star"},
            ))
            break
    fig.update_layout(
        title="Prefill GPUs vs Metadata Operations/sec",
        xaxis={"title": "Prefill GPUs", "type": "log", "dtick": 1},
        yaxis={"title": "Metadata ops/s"},
        height=420,
    )
    return fig


def prefill_scaling_summary_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Prefill GPUs": r["prefill_gpus"],
            "Metadata ops/s": f"{r['metadata_ops_per_sec']:,.0f}",
            "SW latency (µs)": f"{r['software_latency_us']:.2f}",
            "RTL latency (µs)": f"{r['hardware_latency_us']:.2f}",
            "Speedup": f"{r['speedup']:.2f}×",
            "RTL utilization": f"{r['rtl_utilization'] * 100:.1f}%",
            "Region": r["region"],
        }
        for r in rows
    ])


@st.cache_data(show_spinner=False)
def compute_prefill_scaling_sweep(config_key: tuple[tuple[str, Any], ...]) -> list[dict[str, Any]]:
    config = Checkpoint1Config(**dict(config_key), prefill_gpus=2)
    sweep = Checkpoint2Simulator(config).sweep_prefill_gpus()
    return [
        {
            "prefill_gpus": p.prefill_gpus,
            "rtl_utilization": p.result.rtl_utilization,
            "software_latency_us": p.result.software_latency_us,
            "hardware_latency_us": p.result.hardware_latency_us,
            "metadata_ops_per_sec": p.result.metadata_ops_per_sec,
            "speedup": p.result.speedup,
            "region": p.result.region,
        }
        for p in sweep.points
    ]


def throughput_chart(result) -> go.Figure:
    categories = ["Theoretical", "Sustainable", "Achieved"]
    sw_ops = [
        result.software_throughput.theoretical_ops_per_sec,
        result.software_throughput.sustainable_ops_per_sec,
        result.software_throughput.ops_per_sec,
    ]
    hw_ops = [
        result.hardware_throughput.theoretical_ops_per_sec,
        result.hardware_throughput.sustainable_ops_per_sec,
        result.hardware_throughput.ops_per_sec,
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Software", x=categories, y=sw_ops))
    fig.add_trace(go.Bar(name="RTL", x=categories, y=hw_ops))
    fig.update_layout(title="Throughput Comparison (ops/s)", barmode="group", height=400)
    return fig


@st.cache_data(show_spinner=False)
def compute_crossover_grid(
    config_key: tuple[tuple[str, Any], ...],
    checkpoint: str,
) -> tuple[list[float], list[int], list[list[float]]]:
    """Cache heatmap grid keyed by the full runtime configuration."""
    config = Checkpoint1Config(**dict(config_key))
    rps_range = np.logspace(2, 4.5, 15).tolist()
    meta_range = np.logspace(3, 5, 12).astype(int).tolist()
    z: list[list[float]] = []

    for n in meta_range:
        row: list[float] = []
        for rps in rps_range:
            cfg = Checkpoint1Config(**{**config.to_dict(), "num_kv_blocks": int(n), "requests_per_sec": float(rps)})
            row.append(run_simulation(checkpoint, cfg).speedup)
        z.append(row)

    return rps_range, meta_range, z


def crossover_heatmap(config: Checkpoint1Config, checkpoint: str) -> go.Figure:
    config_key = tuple(sorted((k, v) for k, v in config.to_dict().items()))
    rps_range, meta_range, z = compute_crossover_grid(config_key, checkpoint)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=rps_range,
            y=meta_range,
            z=z,
            colorscale="RdYlGn",
            zmid=1.0,
            colorbar={"title": "Speedup (SW/HW)"},
            hovertemplate=(
                "Requests/sec: %{x:.0f}<br>"
                "Metadata entries: %{y:.0f}<br>"
                "Speedup: %{z:.2f}<extra></extra>"
            ),
        )
    )

    # Break-even contour at speedup = 1
    fig.add_trace(
        go.Contour(
            x=rps_range,
            y=meta_range,
            z=z,
            contours={
                "coloring": "lines",
                "showlabels": True,
                "labelfont": {"size": 11, "color": "white"},
                "start": 1,
                "end": 1,
                "size": 1,
            },
            line={"color": "white", "width": 2, "dash": "dash"},
            showscale=False,
            name="Break-even",
        )
    )

    fig.update_layout(
        title="Hardware Crossover Heatmap — Speedup vs RPS × Metadata Entries",
        xaxis={"title": "Requests/sec", "type": "log"},
        yaxis={"title": "Metadata entries", "type": "log"},
        height=520,
        annotations=[
            dict(x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
                 text="<b>Green</b>: HARDWARE BETTER &nbsp;|&nbsp; <b>Red</b>: HARDWARE WORSE &nbsp;|&nbsp; White line: break-even",
                 showarrow=False, font={"size": 12}, bgcolor="rgba(255,255,255,0.8)"),
        ],
    )
    return fig


def flatten_assumptions(raw: dict[str, Any], prefix: str = "") -> list[dict[str, str]]:
    """Flatten nested assumptions.yaml into table rows."""
    rows: list[dict[str, str]] = []

    for key, value in raw.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and "value" in value and "label" in value:
            rows.append({
                "Parameter": path,
                "Value": str(value.get("value", "")),
                "Label": str(value.get("label", "ASSUMPTION")),
                "Unit": str(value.get("unit", "")),
            })
        elif isinstance(value, dict):
            rows.extend(flatten_assumptions(value, path))

    return rows


def build_runtime_parameter_rows(config: Checkpoint1Config, result) -> list[dict[str, str]]:
    """Build labelled rows for the active sidebar configuration."""
    runtime_fields = [
        ("workload.requests_per_sec", config.requests_per_sec, "req/s"),
        ("workload.concurrent_requests", config.concurrent_requests, ""),
        ("workload.metadata_ops_per_request", config.metadata_ops_per_request, "ops/req"),
        ("workload.burstiness", config.burstiness, ""),
        ("workload.prefix_reuse_fraction", config.prefix_reuse_fraction, ""),
        ("kv_metadata.num_kv_blocks", config.num_kv_blocks, "entries"),
        ("kv_metadata.metadata_bytes_per_block", config.metadata_bytes_per_block, "bytes"),
        ("kv_metadata.candidate_count_k", config.candidate_count_k, ""),
        ("kv_metadata.hotness_complexity", config.hotness_complexity, ""),
        ("software.lookup_latency_us", config.sw_lookup_latency_us, "µs"),
        ("software.metadata_update_latency_us", config.sw_metadata_update_latency_us, "µs"),
        ("software.score_calc_latency_us", config.sw_score_calc_latency_us, "µs"),
        ("software.cpu_cores", config.cpu_cores, ""),
        ("software.parallelism", config.software_parallelism, ""),
        ("software.cache_hit_rate", config.cache_hit_rate, ""),
        ("software.cpu_contention", config.cpu_contention, ""),
        ("software.sync_overhead_us", config.sw_sync_overhead_us, "µs"),
        ("rtl.clock_freq_mhz", config.clock_freq_mhz, "MHz"),
        ("rtl.pipeline_depth", config.pipeline_depth, "cycles"),
        ("rtl.lookup_ports", config.lookup_ports, ""),
        ("rtl.parallel_metadata_ops", config.parallel_metadata_ops, ""),
        ("rtl.scoring_units", config.scoring_units, ""),
        ("rtl.queue_depth", config.queue_depth, ""),
        ("rtl.metadata_memory_bandwidth_gbps", config.metadata_memory_bandwidth_gbps, "GB/s"),
        ("rtl.hw_to_sw_overhead_us", config.hw_to_sw_overhead_us, "µs"),
        ("hotness.model_type", config.hotness_model_type.value, ""),
        ("hotness.alpha", config.hotness_alpha, ""),
        ("hotness.beta", config.hotness_beta, ""),
        ("hotness.gamma", config.hotness_gamma, ""),
        ("hotness.decay_lambda", config.hotness_decay_lambda, ""),
        ("topology.prefill_gpus", config.prefill_gpus, ""),
        ("topology.rtl_engines", config.rtl_engines, ""),
        ("coordination.coordination_latency_us", config.coordination_latency_us, "µs"),
    ]

    rows = [
        {
            "Parameter": name,
            "Value": str(value),
            "Label": "ASSUMPTION",
            "Unit": unit,
        }
        for name, value, unit in runtime_fields
    ]

    derived_fields = [
        ("derived.software_latency_us", f"{result.software_latency_us:.3f}", "µs"),
        ("derived.hardware_latency_us", f"{result.hardware_latency_us:.3f}", "µs"),
        ("derived.speedup", f"{result.speedup:.3f}", "×"),
        ("derived.metadata_ops_per_sec", f"{result.metadata_ops_per_sec:,.0f}", "ops/s"),
        ("derived.rtl_utilization", f"{result.rtl_utilization * 100:.1f}", "%"),
        ("derived.region", result.region, ""),
    ]
    rows.extend(
        {"Parameter": name, "Value": value, "Label": "DERIVED", "Unit": unit}
        for name, value, unit in derived_fields
    )
    return rows


def render_parameter_labels_tab(config: Checkpoint1Config, result, checkpoint: str):
    st.markdown("### Configuration Assumptions")
    cp_label = "Checkpoint 2" if checkpoint == CHECKPOINT_2 else "Checkpoint 1"
    st.caption(
        f"Every parameter is tagged as **ASSUMPTION**, **MEASURED**, or **DERIVED**. "
        f"{cp_label} uses analytical assumptions only — no measured Dynamo numbers."
    )

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        assumption_rows = flatten_assumptions(raw)
        assumptions_df = pd.DataFrame(assumption_rows)
        st.markdown("#### Default assumptions file (`configs/assumptions.yaml`)")
        st.dataframe(assumptions_df, use_container_width=True, hide_index=True)
    else:
        st.warning("`configs/assumptions.yaml` not found.")

    st.markdown("#### Active runtime configuration (sidebar)")
    runtime_df = pd.DataFrame(build_runtime_parameter_rows(config, result))
    st.dataframe(runtime_df, use_container_width=True, hide_index=True)

    st.markdown("#### Label legend")
    legend_col1, legend_col2, legend_col3 = st.columns(3)
    with legend_col1:
        st.info("**ASSUMPTION** — user-configurable design-space input")
    with legend_col2:
        st.success("**MEASURED** — value from real hardware/system measurement")
    with legend_col3:
        st.warning("**DERIVED** — computed by the analytical model")

    with st.expander("Raw assumptions YAML"):
        if CONFIG_PATH.exists():
            st.code(CONFIG_PATH.read_text(), language="yaml")


def build_config_from_sidebar(defaults: Checkpoint1Config) -> tuple[str, Checkpoint1Config]:
    st.sidebar.header("Checkpoint Mode")
    checkpoint = st.sidebar.selectbox("Checkpoint", CHECKPOINT_OPTIONS, index=0)

    if checkpoint == CHECKPOINT_2:
        st.sidebar.caption("Checkpoint 2: scale prefill GPUs (2, 4, 8, 16) with 1 RTL engine.")
        prefill_gpus = st.sidebar.selectbox("Prefill GPUs", PREFILL_GPU_OPTIONS_CP2, index=1)
    else:
        st.sidebar.caption("Checkpoint 1: single prefill GPU, single RTL engine.")
        prefill_gpus = 1

    st.sidebar.header("Workload")
    preset = st.sidebar.selectbox("Workload preset", list(WORKLOAD_PRESETS.keys()))
    preset_vals = WORKLOAD_PRESETS[preset] or {}

    def _val(key, widget_fn, *args, **kwargs):
        default = preset_vals.get(key, getattr(defaults, key))
        return widget_fn(*args, value=default, **kwargs)

    requests_per_sec = st.sidebar.slider("Requests/sec", 100, 20000, int(_val("requests_per_sec", lambda **kw: kw["value"])))
    concurrent_requests = st.sidebar.slider("Concurrent requests", 1, 256, int(_val("concurrent_requests", lambda **kw: kw["value"])))
    metadata_ops = st.sidebar.slider("Metadata ops/request", 1, 64, int(_val("metadata_ops_per_request", lambda **kw: kw["value"])))
    burstiness = st.sidebar.slider("Burstiness", 0.5, 5.0, float(_val("burstiness", lambda **kw: kw["value"])), 0.1)
    prefix_reuse_fraction = st.sidebar.slider(
        "Prefix reuse fraction", 0.0, 1.0,
        float(_val("prefix_reuse_fraction", lambda **kw: kw["value"])), 0.05,
    )

    st.sidebar.header("KV Metadata")
    num_kv_blocks = st.sidebar.select_slider(
        "Metadata entries (KV blocks)",
        options=[1024, 4096, 16384, 65536, 262144, 1048576],
        value=int(_val("num_kv_blocks", lambda **kw: kw["value"])),
    )
    metadata_bytes = st.sidebar.slider("Metadata bytes/block", 16, 256, int(defaults.metadata_bytes_per_block), 16)
    candidate_k = st.sidebar.selectbox("Candidate count K", CANDIDATE_K_OPTIONS, index=CANDIDATE_K_OPTIONS.index(defaults.candidate_count_k))
    hotness_complexity = st.sidebar.slider("Hotness complexity", 0.5, 5.0, float(defaults.hotness_complexity), 0.1)

    st.sidebar.header("Software Baseline")
    sw_lookup = st.sidebar.number_input("SW lookup latency (µs)", 0.1, 20.0, float(defaults.sw_lookup_latency_us), 0.1)
    sw_update = st.sidebar.number_input("SW metadata update (µs)", 0.1, 20.0, float(defaults.sw_metadata_update_latency_us), 0.1)
    sw_score = st.sidebar.number_input("SW score calc (µs)", 0.1, 20.0, float(defaults.sw_score_calc_latency_us), 0.1)
    cpu_cores = st.sidebar.slider("CPU cores", 1, 64, int(defaults.cpu_cores))
    parallelism = st.sidebar.slider("Software parallelism", 1, 16, int(defaults.software_parallelism))
    cache_hit_rate = st.sidebar.slider("Cache hit rate", 0.0, 1.0, float(defaults.cache_hit_rate), 0.05)
    cpu_contention = st.sidebar.slider("CPU contention", 0.0, 1.0, float(defaults.cpu_contention), 0.05)
    sw_sync = st.sidebar.number_input("SW sync overhead (µs)", 0.0, 10.0, float(defaults.sw_sync_overhead_us), 0.1)

    st.sidebar.header("RTL Parameters")
    clock_freq = st.sidebar.selectbox("Clock frequency (MHz)", CLOCK_OPTIONS, index=CLOCK_OPTIONS.index(int(defaults.clock_freq_mhz)))
    pipeline_depth = st.sidebar.slider("Pipeline depth", 1, 32, int(defaults.pipeline_depth))
    lookup_ports = st.sidebar.selectbox("Lookup ports", PARALLEL_ENGINES, index=PARALLEL_ENGINES.index(defaults.lookup_ports))
    parallel_ops = st.sidebar.selectbox("Parallel metadata ops", PARALLEL_ENGINES, index=PARALLEL_ENGINES.index(defaults.parallel_metadata_ops))
    scoring_units = st.sidebar.selectbox("Scoring units", PARALLEL_ENGINES, index=PARALLEL_ENGINES.index(defaults.scoring_units))
    queue_depth = st.sidebar.slider("Queue depth", 16, 1024, int(defaults.queue_depth), 16)
    metadata_bw = st.sidebar.slider("Metadata memory BW (GB/s)", 1.0, 128.0, float(defaults.metadata_memory_bandwidth_gbps), 1.0)
    hw_to_sw = st.sidebar.number_input("HW→SW overhead (µs)", 0.0, 5.0, float(defaults.hw_to_sw_overhead_us), 0.05)

    st.sidebar.header("Hotness Model")
    hotness_name = st.sidebar.selectbox("Hotness model", list(HOTNESS_OPTIONS.keys()))
    hotness_alpha = st.sidebar.slider("Alpha (access count)", 0.0, 5.0, float(defaults.hotness_alpha), 0.1)
    hotness_beta = st.sidebar.slider("Beta (recency)", 0.0, 5.0, float(defaults.hotness_beta), 0.1)
    hotness_gamma = st.sidebar.slider("Gamma (ref count)", 0.0, 5.0, float(defaults.hotness_gamma), 0.1)
    decay_lambda = st.sidebar.slider("Decay lambda", 0.001, 0.1, float(defaults.hotness_decay_lambda), 0.001, format="%.3f")

    st.sidebar.header("Coordination")
    coord_latency = st.sidebar.number_input("Coordination latency (µs)", 0.0, 20.0, float(defaults.coordination_latency_us), 0.1)

    return checkpoint, Checkpoint1Config(
        requests_per_sec=float(requests_per_sec),
        concurrent_requests=concurrent_requests,
        metadata_ops_per_request=metadata_ops,
        burstiness=burstiness,
        prefix_reuse_fraction=prefix_reuse_fraction,
        num_kv_blocks=num_kv_blocks,
        metadata_bytes_per_block=metadata_bytes,
        candidate_count_k=candidate_k,
        hotness_complexity=hotness_complexity,
        sw_lookup_latency_us=sw_lookup,
        sw_metadata_update_latency_us=sw_update,
        sw_score_calc_latency_us=sw_score,
        cpu_cores=cpu_cores,
        software_parallelism=parallelism,
        cache_hit_rate=cache_hit_rate,
        cpu_contention=cpu_contention,
        sw_sync_overhead_us=sw_sync,
        clock_freq_mhz=float(clock_freq),
        pipeline_depth=pipeline_depth,
        lookup_ports=lookup_ports,
        parallel_metadata_ops=parallel_ops,
        scoring_units=scoring_units,
        queue_depth=queue_depth,
        metadata_memory_bandwidth_gbps=metadata_bw,
        hw_to_sw_overhead_us=hw_to_sw,
        hotness_model_type=HOTNESS_OPTIONS[hotness_name],
        hotness_alpha=hotness_alpha,
        hotness_beta=hotness_beta,
        hotness_gamma=hotness_gamma,
        hotness_decay_lambda=decay_lambda,
        coordination_latency_us=coord_latency,
        prefill_gpus=prefill_gpus,
        rtl_engines=1,
    )


def render_prefill_scaling_tab(config: Checkpoint1Config, checkpoint: str):
    st.markdown("### Prefill GPU Scaling")
    if checkpoint != CHECKPOINT_2:
        st.info("Switch to **Checkpoint 2: Prefill Scaling** in the sidebar to explore 2, 4, 8, and 16 prefill GPUs.")
        st.caption("Checkpoint 1 is fixed at 1 prefill GPU.")
        return

    config_key = tuple(sorted((k, v) for k, v in config.to_dict().items() if k != "prefill_gpus"))
    scaling_rows = compute_prefill_scaling_sweep(config_key)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(prefill_gpu_utilization_chart(scaling_rows, config.prefill_gpus), use_container_width=True)
    with col2:
        st.plotly_chart(prefill_gpu_latency_chart(scaling_rows, config.prefill_gpus), use_container_width=True)

    st.plotly_chart(prefill_gpu_metadata_ops_chart(scaling_rows, config.prefill_gpus), use_container_width=True)
    st.markdown("#### Scaling Summary")
    st.dataframe(prefill_scaling_summary_table(scaling_rows), use_container_width=True, hide_index=True)

    max_util = max(r["rtl_utilization"] for r in scaling_rows)
    if max_util >= 0.95:
        st.warning(f"RTL engine approaching saturation at {max_util * 100:.0f}% utilization with current workload.")
    else:
        st.success(f"RTL engine has headroom — max utilization {max_util * 100:.0f}% across 2–16 GPUs.")


def main():
    st.set_page_config(
        page_title="KV Metadata Engine — Design Explorer",
        page_icon="⚡",
        layout="wide",
    )

    st.title("KV Metadata & Placement Engine — Design-Space Explorer")

    defaults = load_default_config()
    checkpoint, config = build_config_from_sidebar(defaults)

    st.markdown(
        f"{checkpoint_description(checkpoint)}  \n"
        "All parameters labelled **ASSUMPTION** unless measured. This is an analytical model, not a Dynamo benchmark."
    )

    sim = create_simulator(checkpoint, config)
    result = sim.run()

    crossover_rps = find_crossover_rps(config) if checkpoint == CHECKPOINT_1 else None

    render_summary(result, config, crossover_rps)

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Latency & Throughput",
        "Crossover Analysis",
        "Hotness Tracker",
        "Parameter Labels",
        "Prefill Scaling",
    ])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(latency_breakdown_chart(result.software_breakdown, result.hardware_breakdown), use_container_width=True)
        with col_b:
            st.plotly_chart(throughput_chart(result), use_container_width=True)

        st.plotly_chart(
            speedup_vs_rps_chart(sim.sweep_rps(list(np.logspace(2, 4.5, 30)))),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            metadata_entries_speedup_chart(
                sim.sweep_metadata_entries([1024, 4096, 16384, 65536, 262144, 524288, 1048576])
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            candidate_k_latency_chart(sim.sweep_candidate_k(CANDIDATE_K_OPTIONS)),
            use_container_width=True,
        )
        with st.spinner("Computing crossover heatmap..."):
            st.plotly_chart(crossover_heatmap(config, checkpoint), use_container_width=True)

        region = result.region
        if region == "HARDWARE WORSE":
            st.warning("Under current assumptions, software outperforms RTL hardware.")
        elif region == "HARDWARE BETTER":
            st.success("Under current assumptions, RTL hardware outperforms software.")
        else:
            st.info("Current configuration is near break-even between software and RTL.")

    with tab3:
        st.plotly_chart(hotness_distribution_chart(result.hotness_distribution), use_container_width=True)
        col_h1, col_h2, col_h3 = st.columns(3)
        with col_h1:
            st.metric("Hotness update rate", f"{result.hotness_update_rate:,.0f} ops/s")
        with col_h2:
            st.metric("HW cycles / update", f"{result.hw_cycles_per_hotness_update:.1f}")
        with col_h3:
            hot_pct = sum(1 for b in result.hotness_distribution if b.classification == "hot") / max(len(result.hotness_distribution), 1) * 100
            st.metric("Hot blocks (sample)", f"{hot_pct:.1f}%")

    with tab4:
        render_parameter_labels_tab(config, result, checkpoint)

    with tab5:
        render_prefill_scaling_tab(config, checkpoint)


if __name__ == "__main__":
    main()
