# Topology YAML Input Format (schema_version: 1)
#
# Load into the simulator:
#   from model.topology.io import load_topology_yaml
#   bundle = load_topology_yaml("configs/topologies/worker_pools_hbm.yaml")
#
# Or in the UI: Topology Designer → "Topology YAML — load / export"
#
# ── Root fields ──────────────────────────────────────────────────────────────
# schema_version: 1                          # required
# topology:
#   name: "My System"                        # required
#   description: "optional notes"
#
# defaults:                                   # optional (UI default PCIe / DMA)
#   pcie: { generation, width, mps, mrrs, base_latency_us, switch_hops,
#           congestion_factor, protocol_efficiency, switch_hop_latency_us,
#           read_request_latency_us, memory_service_latency_us,
#           endpoint_processing_us, memory_processing_us,
#           congestion_base_penalty_us }
#   dma:  { queue_depth, max_outstanding_reads, request_size, completion_size,
#           read_latency_us, write_latency_us }
#
# ── Nodes ────────────────────────────────────────────────────────────────────
# nodes:
#   - id: prefill_pool                        # required, unique
#     name: Prefill Worker Pool                # required
#     type: PREFILL_POOL                      # required (see NodeType enum)
#     position: {x: -240, y: 120}             # or [x, y]
#     metadata:                               # optional free-form map
#       worker_count: 4
#
# Valid node types:
#   CPU, GPU, FPGA, PREFILL_POOL, DECODE_POOL, ROUTER, KVBM_ENGINE, MEMORY,
#   PCIE_ROOT_COMPLEX, PCIE_SWITCH, NETWORK_SWITCH, NVSWITCH, NIC
#
# ── Links ────────────────────────────────────────────────────────────────────
# links:
#   - id: prefill_pool_hbm                    # required, unique
#     source: prefill_pool                     # required (node id)
#     dest: prefill_hbm                        # required (node id)
#     interconnect: PCIe                       # required (see InterconnectType)
#     bidirectional: true                      # default true
#     enabled: true                            # default true
#     serialization_enabled: true              # default true
#     congestion_factor: 1.0                    # default 1.0
#     metadata: {}                             # optional
#     pcie:                                     # PCIe links: optional override
#       generation: Gen5                       # inherits defaults.pcie if omitted
#       width: x16
#       # ... same keys as defaults.pcie
#
# Non-PCIe link example:
#   - id: router_nic
#     source: router
#     dest: nic
#     interconnect: Ethernet
#     bandwidth_GBps: 100.0
#     latency_us: 2.0
#     bidirectional: true
#
# Valid interconnect types:
#   PCIe, NVLink, NVSwitch, Ethernet, InfiniBand, RoCE, CXL, Generic
