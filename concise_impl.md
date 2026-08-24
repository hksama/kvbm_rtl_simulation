You are modifying an EXISTING Streamlit application for research into a
hardware-assisted KV-cache metadata and placement engine for distributed LLM
inference.

IMPORTANT:

DO NOT rewrite the existing analytical KVBM model.

DO NOT replace the current Streamlit UI.

DO NOT break the existing checkpoint system.

The current application already implements Checkpoint 1:

    Single Prefill GPU
    Single RTL KVBM engine

with controls for:

    Workload
    KV metadata
    Software baseline
    RTL parameters
    Hotness model
    Coordination

and existing plots / calculations.

Your job is to EXTEND the application with:

    1. PCIe/interconnect modeling
    2. topology modeling
    3. a visual topology editor
    4. NVIDIA-Dynamo-like system architecture modeling
    5. path selection / path analysis
    6. integration of topology costs into the existing KVBM model

The application remains a DESIGN-SPACE EXPLORATION TOOL.

It is NOT intended to reproduce NVIDIA Dynamo's implementation or benchmark
Dynamo's real performance.

============================================================
PART 1 — ARCHITECTURAL PRINCIPLE
============================================================

Separate the application into four conceptual layers:

    Layer 1:
        Workload / KVBM model

    Layer 2:
        Hardware/software metadata engine model

    Layer 3:
        Interconnect model

    Layer 4:
        Topology / system graph

The existing KVBM calculations should not know about Streamlit UI objects.

The topology model should expose a clean API to the simulation.

For example:

    topology.find_path(source, destination)

    topology.path_latency(path)

    topology.path_bandwidth(path)

    topology.transfer_time(path, message_size)

    topology.path_summary(path)

The KVBM model should then call these functions when topology-aware behavior
is enabled.

============================================================
PART 2 — TOPOLOGY GRAPH
============================================================

Introduce a graph-based topology representation.

Every system component is a NODE.

Every communication connection is an EDGE.

Example:

                    ROUTER
                       |
          +------------+------------+
          |                         |
       PREFILL                    DECODE
        POOL                       POOL
          |                         |
      GPU nodes                 GPU nodes
          |
      KV metadata
          |
       KVBM FPGA


Represent nodes with:

    id
    type
    name
    metadata

Node types initially:

    CPU
    GPU
    FPGA
    PREFILL_POOL
    DECODE_POOL
    ROUTER
    KVBM_ENGINE
    MEMORY
    PCIe_ROOT_COMPLEX
    PCIe_SWITCH
    NETWORK_SWITCH
    NVSWITCH
    NIC

The architecture should be extensible so more node types can be added later.

============================================================
PART 3 — EDGE / INTERCONNECT MODEL
============================================================

Every edge represents a communication link.

Each edge should have:

    interconnect_type
    bandwidth_GBps
    latency_us
    width
    generation
    serialization_enabled
    congestion_factor
    enabled

Initial interconnect types:

    PCIe
    NVLink
    NVSwitch
    Ethernet
    InfiniBand
    RoCE
    CXL
    Generic

Do NOT initially attempt to reproduce the full protocol stack of every
interconnect.

The purpose is architectural exploration.

============================================================
PART 4 — PCIe MODEL
============================================================

PCIe should be the first fully parameterized interconnect.

Create a PCIe configuration object.

Parameters:

    generation:
        Gen3
        Gen4
        Gen5
        Gen6

    width:
        x1
        x2
        x4
        x8
        x16

    MPS:
        128
        256
        512
        1024

    MRRS:
        128
        256
        512
        1024
        2048
        4096

    base_latency_us

    switch_hops

    congestion_factor

    protocol_efficiency

The model must distinguish:

    raw bandwidth
    protocol payload bandwidth
    effective/application bandwidth

DO NOT collapse all of these into one value.

============================================================
PART 5 — PCIe BANDWIDTH MODEL
============================================================

Implement a clean PCIe generation table.

Approximate signaling parameters:

    Gen1:
        2.5 GT/s
        NRZ
        8b/10b

    Gen2:
        5 GT/s
        NRZ
        8b/10b

    Gen3:
        8 GT/s
        NRZ
        128b/130b

    Gen4:
        16 GT/s
        NRZ
        128b/130b

    Gen5:
        32 GT/s
        NRZ
        128b/130b

    Gen6:
        64 GT/s
        PAM4
        FLIT mode
        FEC
        CRC

Calculate raw per-direction bandwidth from:

    lanes
    signaling rate
    bits/symbol

Then calculate protocol-level usable bandwidth.

For Gen3-Gen5 use the appropriate encoding efficiency.

For Gen6 model:

    64 GT/s
    2 bits/symbol
    256-byte FLIT
    FLIT overhead

Do not pretend this is a transistor-level PCIe implementation.

This is an analytical model.

Every displayed bandwidth value must say whether it is:

    RAW
    PROTOCOL
    EFFECTIVE

============================================================
PART 6 — PCIe LATENCY MODEL
============================================================

Model:

    transfer_time =
        fixed_link_latency
      + serialization_time
      + hop_latency
      + congestion_penalty

For a payload of N bytes:

    serialization_time =
        N / effective_bandwidth

For PCIe reads, distinguish:

    request
    memory service
    completion

Approximate:

    T_read =
        request_latency
      + forward_path
      + memory_latency
      + return_path
      + endpoint_processing

For PCIe writes:

    T_write =
        forward_path
      + endpoint/memory processing

Allow separate:

    read_latency
    write_latency

Do NOT claim these are exact physical PCIe latencies.

Label them:

    ANALYTICAL ESTIMATE

============================================================
PART 7 — PCIe DMA MODEL
============================================================

Add a DMA model.

Represent:

    DMAEngine

with:

    queue_depth
    max_outstanding_reads
    request_size
    completion_size
    read_latency
    write_latency

Support:

    DMA_READ
    DMA_WRITE

DMA READ:

    FPGA
      |
      | Memory Read Request TLP
      v
    PCIe
      |
      v
    Memory
      |
      | Completion with Data
      v
    FPGA

DMA WRITE:

    FPGA
      |
      | Memory Write TLP
      v
    PCIe
      |
      v
    Memory

Do NOT model DMA as bypassing PCIe.

DMA is a request-generation mechanism operating through the PCIe hierarchy.

============================================================
PART 8 — TLP MODEL
============================================================

Create a lightweight TLP abstraction.

A TLP should contain:

    type
    payload_bytes
    address
    requester
    completer
    sequence_number
    tag
    mps

Types:

    MEMORY_READ
    MEMORY_WRITE
    COMPLETION
    COMPLETION_WITH_DATA

Implement a fragmentation function:

    fragment_transfer(size, MPS)

Example:

    4096 B transfer
    MPS = 256 B

produces approximately:

    16 payload TLPs

Account for:

    TLP header overhead

in bandwidth calculations.

Do not implement the complete PCIe specification.

============================================================
PART 9 — PCIe 6 FLIT MODEL
============================================================

For Gen6 introduce a lightweight FLIT abstraction.

FLIT:

    size = 256 bytes

Model:

    TLP payload area
    data-link information
    CRC
    FEC

Allow TLPs to be packed into FLITs.

A TLP may span multiple FLITs.

The purpose is to estimate:

    serialization
    overhead
    effective bandwidth

NOT to implement a real PCIe 6 controller.

============================================================
PART 10 — PATH MODEL
============================================================

Every communication request should be represented as:

    source
    destination
    message_size
    traffic_type

Examples:

    FPGA -> GPU
    GPU -> FPGA
    Prefill GPU -> Decode GPU
    Router -> KVBM engine
    KVBM engine -> Decode worker
    KVBM engine -> Memory

The topology layer should find a path:

    source -> node -> link -> node -> ... -> destination

For every path calculate:

    hop_count
    bottleneck_bandwidth
    total_fixed_latency
    serialization_time
    congestion_penalty
    total_latency

Return a structured PathResult.

Example:

    PathResult:
        nodes
        edges
        hop_count
        bottleneck_bandwidth
        latency_us
        transfer_time_us
        utilization

============================================================
PART 11 — TOPOLOGY TYPES
============================================================

Support these initial architectures.

ARCHITECTURE A:

    CPU
     |
    PCIe
     |
    FPGA


ARCHITECTURE B:

    CPU
     |
    PCIe Root Complex
     |
    PCIe Switch
     |
    FPGA


ARCHITECTURE C:

    CPU
     |
    PCIe Root
     |
    +----------+----------+
    |                     |
   GPU                    FPGA


ARCHITECTURE D:

    PREFILL GPU POOL
          |
          | PCIe / network
          |
       KVBM FPGA
          |
          | PCIe / network
          |
    DECODE GPU POOL


ARCHITECTURE E:

    PREFILL GPUs
         |
       network
         |
       Router
         |
       network
         |
    DECODE GPUs


ARCHITECTURE F:

    PREFILL GPUs
         |
      NVLink
         |
     NVSwitch
      /     \
   GPU      GPU
         |
       PCIe
         |
       FPGA


These are illustrative architectures, NOT claims about NVIDIA Dynamo's
actual physical deployment.

============================================================
PART 12 — DYNAMO-INSPIRED LOGICAL ARCHITECTURE
============================================================

Add a logical system model inspired by the concepts exposed by NVIDIA Dynamo.

Represent:

    Router
    Prefill Worker Pool
    Decode Worker Pool
    KV Metadata Manager
    KV Cache
    KVBM Accelerator

The logical architecture should be:

                 CLIENT
                    |
                    v
                  ROUTER
                 /      \
                /        \
               v          v
         PREFILL POOL   DECODE POOL
               |            |
               |            |
               +-----+------+
                     |
                 KV METADATA
                     |
                KVBM ENGINE
                     |
                KV PLACEMENT
                     |
                KV TRANSFER


Do NOT claim this is the exact physical Dynamo topology.

The purpose is to model a Dynamo-like distributed inference system.

============================================================
PART 13 — PREFILL/DECODE POOLS
============================================================

Represent worker pools explicitly.

PrefillPool:

    worker_count
    worker_ids
    workload_rate
    metadata_ops_rate

DecodePool:

    worker_count
    worker_ids
    workload_rate
    metadata_ops_rate

Each worker is a topology node.

Example:

    Prefill GPU 0
    Prefill GPU 1
    Prefill GPU 2
    ...

and:

    Decode GPU 0
    Decode GPU 1
    Decode GPU 2
    ...

The user should be able to create/remove workers through the UI.

============================================================
PART 14 — KV CACHE MODEL
============================================================

Separate:

    KV metadata
    KV tensor data

The KVBM accelerator initially manages:

    metadata
    placement decisions
    block location
    hotness
    candidate selection

It does NOT physically own the KV tensor data.

Represent KV cache state:

    block_id
    size_bytes
    owner_worker
    last_access
    access_count
    hotness
    metadata_version

Allow a KV block to be located on:

    Prefill worker
    Decode worker
    Shared memory
    External memory

============================================================
PART 15 — VISUAL TOPOLOGY EDITOR
============================================================

This is a major new UI component.

The Streamlit application should contain a:

    "Topology Designer"

tab/page.

The user should be able to:

    create node
    delete node
    connect nodes
    edit node properties
    edit edge properties
    duplicate nodes
    create worker pools
    assign interconnect types

Provide a visual graph.

Prefer a Streamlit-compatible graph visualization library such as:

    streamlit-agraph

or another appropriate graph library already compatible with the project's
dependency setup.

DO NOT introduce a massive frontend framework unless necessary.

The topology editor should remain native to the existing Streamlit app.

============================================================
PART 16 — DRAG AND DROP
============================================================

The user explicitly wants drag-and-drop experimentation.

Implement the closest reliable Streamlit-compatible interaction possible.

The user should be able to:

    drag nodes
    connect nodes
    visually inspect topology

If true arbitrary drag/drop editing is technically limited by Streamlit or the
chosen graph library, do NOT fake it.

Instead provide:

    visual graph editor
    node creation controls
    connection controls
    position persistence

with a clean abstraction so a richer frontend can be introduced later.

Do not redesign the entire application just to implement drag/drop.

============================================================
PART 17 — PATH INSPECTOR
============================================================

When the user selects:

    source node
    destination node

display:

    path

Example:

    Prefill GPU 2
        ↓
    PCIe Root Complex
        ↓
    PCIe Switch
        ↓
    KVBM FPGA
        ↓
    PCIe Switch
        ↓
    Decode GPU 4

Display:

    hop count
    bandwidth per hop
    bottleneck bandwidth
    total latency
    transfer latency
    estimated KV transfer time

Highlight the selected path in the graph.

============================================================
PART 18 — PATH COMPARISON
============================================================

Allow the user to compare paths.

Example:

    Path A:
        GPU -> PCIe -> FPGA

    Path B:
        GPU -> PCIe Switch -> FPGA

    Path C:
        GPU -> NVSwitch -> GPU -> PCIe -> FPGA

Display:

    latency
    bandwidth
    hops
    estimated transfer time

This should make topology tradeoffs visually obvious.

============================================================
PART 19 — KVBM INTEGRATION
============================================================

The existing KVBM model should now have:

    topology_aware = True / False

When FALSE:

    preserve current behavior exactly.

When TRUE:

    include topology cost.

For example:

    placement_score(worker) =
        existing_score(worker)
        - network_weight * topology_cost(worker)

Topology cost should include:

    latency
    bandwidth
    transfer_size
    congestion
    hop count

Do not hard-code the placement algorithm.

============================================================
PART 20 — PREFILL → DECODE KV TRANSFER
============================================================

Add a simulation event:

    KV_TRANSFER

with:

    source_worker
    destination_worker
    kv_size_bytes
    path
    transfer_time
    queue_delay

Calculate:

    T_KV_transfer =
        path_latency
        + KV_size / path_bandwidth
        + congestion
        + queueing

Display this separately from metadata decision latency.

This distinction is extremely important.

The KVBM engine may make a placement decision in nanoseconds/microseconds,
while the actual KV transfer may take substantially longer.

Do not merge them into a single number.

============================================================
PART 21 — TOPOLOGY CONGESTION
============================================================

Each edge should track:

    offered_bandwidth
    capacity
    utilization

For example:

    utilization =
        offered_bandwidth / link_capacity

If:

    utilization >= 1

the edge is saturated.

Add a configurable congestion model.

At minimum:

    congestion_penalty =
        base_penalty * utilization_factor

Make this model explicit and documented.

Do not claim it represents real switch queueing exactly.

============================================================
PART 22 — MULTIPLE KVBM ENGINES
============================================================

Prepare the topology model for:

    1 KVBM engine
    2
    4
    8
    16

Support:

    centralized
    sharded
    replicated
    hierarchical

The topology editor should make these architectures visually representable.

Example:

CENTRALIZED:

    Prefill GPUs
          |
          v
       KVBM-0
          |
          v
    Decode GPUs


SHARDED:

    Prefill GPUs
       / | \
      /  |  \
 KVBM0 KVBM1 KVBM2
      \  |  /
       Decode


HIERARCHICAL:

             Global KVBM
             /        \
        Rack KVBM   Rack KVBM
          /  \        /  \
        GPU GPU     GPU GPU

Do not implement all algorithms yet.

First make the architecture representable and measurable.

============================================================
PART 23 — STREAMLIT UI
============================================================

Add the following high-level tabs:

    1. Simulation
    2. Topology
    3. Interconnect
    4. Path Inspector
    5. Dynamo Architecture
    6. Scaling
    7. Results
    8. Assumptions

The current simulation controls should remain accessible.

Do not dump hundreds of controls into one page.

Use expandable sections.

============================================================
PART 24 — TOPOLOGY UI
============================================================

Topology page:

    [Create Node]
    [Create Connection]
    [Delete Selected]
    [Reset Topology]
    [Save Topology]
    [Load Topology]

Node properties:

    name
    type
    worker role
    pool
    metadata capacity
    compute capacity

Edge properties:

    interconnect
    generation
    width
    bandwidth
    latency
    congestion

Display:

    interactive topology graph

============================================================
PART 25 — DYNAMO ARCHITECTURE PRESETS
============================================================

Provide presets:

    Minimal:
        1 prefill
        1 decode
        1 KVBM

    Small:
        4 prefill
        4 decode
        1 KVBM

    Medium:
        16 prefill
        16 decode
        2 KVBM

    Large:
        64 prefill
        64 decode
        4 KVBM

These are simulation presets only.

Do NOT present them as real Dynamo deployment configurations.

Allow:

    "Load preset into topology editor"

so the user can modify the architecture.

============================================================
PART 26 — RESULTS
============================================================

The Results page should display:

    Metadata decision latency
    KV transfer latency
    End-to-end latency
    Software latency
    Hardware latency
    Speedup
    Throughput
    Link utilization
    KVBM utilization
    GPU utilization estimate
    Queue depth
    Stale metadata rate
    Avoidable miss rate
    KV wait time

Separate:

    METADATA PATH

from:

    DATA PATH

For example:

    METADATA PATH:

    Router
      ↓
    KVBM
      ↓
    placement decision


    DATA PATH:

    Prefill GPU
      ↓
    network / PCIe / NVLink
      ↓
    Decode GPU

This distinction should be visually obvious.

============================================================
PART 27 — ARCHITECTURAL QUESTIONS THE UI SHOULD ANSWER
============================================================

The UI should make these questions easy to investigate:

1.

    When does a single KVBM engine saturate?

2.

    How many prefill GPUs can one engine support?

3.

    How many decode GPUs can one engine support?

4.

    When is centralized metadata no longer scalable?

5.

    When does sharding become beneficial?

6.

    When does synchronization overhead dominate?

7.

    When does topology-aware placement become beneficial?

8.

    How much does PCIe bandwidth matter?

9.

    When does PCIe latency dominate instead of bandwidth?

10.

    When does actual KV movement dominate metadata decision latency?

11.

    How sensitive is the system to stale metadata?

12.

    How many KVBM engines are required for a given cluster size?

============================================================
PART 28 — CURRENT CHECKPOINT
============================================================

The current application is at:

    CHECKPOINT 2 / early scaling stage

Do NOT force implementation of every future checkpoint immediately.

Implement the infrastructure required for future checkpoints while ensuring
the current checkpoint continues working.

The first milestone after this change should be:

    Checkpoint 2:

        configurable prefill GPU count
        topology graph
        PCIe links
        KVBM engine
        path calculation
        topology-aware latency
        scaling plots

Then make it easy to extend to:

    distributed prefill
    decode
    distributed decode
    integrated prefill + decode

============================================================
PART 29 — DATA MODEL
============================================================

Create clean dataclasses / typed models.

Suggested conceptual structures:

    Node
    Edge
    Topology
    PCIeConfig
    DMAConfig
    TLP
    FLIT
    Path
    PathResult
    Worker
    WorkerPool
    KVBlock
    KVBMEngine
    SimulationConfig
    SimulationResult

Do not tightly couple these to Streamlit session_state.

Use session_state only for UI state.

============================================================
PART 30 — TESTING
============================================================

Add tests for:

    PCIe bandwidth calculations
    PCIe generation scaling
    lane scaling
    TLP fragmentation
    FLIT overhead
    serialization time
    DMA read latency
    DMA write latency
    path discovery
    bottleneck bandwidth
    multi-hop latency
    congestion
    topology-aware placement
    KV transfer latency
    topology presets

Important invariants:

    PCIe x8 bandwidth ~= 0.5 * PCIe x16

    PCIe Gen5 bandwidth ~= 2 * Gen4
    before protocol/application effects

    bidirectional traffic must not be incorrectly treated as twice the
    per-direction bandwidth for a one-direction transfer

    path bandwidth =
        minimum edge capacity along path

    path latency =
        sum of edge latencies
        + serialization
        + queue/congestion effects

============================================================
PART 31 — ASSUMPTIONS
============================================================

Create:

    configs/assumptions.yaml

or equivalent.

Every non-measured parameter must be documented.

Categorize parameters:

    ASSUMPTION
    DERIVED
    MEASURED

The UI should expose an "Assumptions" page.

Example:

    PCIe Gen5 x16 theoretical bandwidth
        DERIVED

    FPGA DMA fixed latency
        ASSUMPTION

    Actual FPGA synthesis timing
        NOT AVAILABLE

Never present analytical estimates as measured hardware performance.

============================================================
PART 32 — CODE QUALITY
============================================================

Before modifying anything:

    inspect the existing repository structure.

Identify:

    current simulation/model code
    Streamlit entrypoint
    existing state management
    plotting code
    checkpoint implementation
    configuration handling
    tests

Reuse existing abstractions where possible.

Do not duplicate existing calculations.

Do not create a second independent simulation model.

Do not rewrite working UI unnecessarily.

After implementation:

    run existing tests
    run new tests
    launch Streamlit
    verify existing Checkpoint 1
    verify new topology functionality
    verify a simple PCIe transfer manually

============================================================
FINAL DELIVERABLE
============================================================

The result should be a Streamlit research tool where I can visually construct:

    Router
       |
       +--------- Prefill GPU
       |
       +--------- Prefill GPU
       |
      KVBM FPGA
       |
       +--------- Decode GPU
       |
       +--------- Decode GPU

and change:

    PCIe Gen
    PCIe width
    bandwidth
    latency
    topology
    number of workers
    number of KVBM engines
    metadata workload
    synchronization
    KV transfer size

and immediately see:

    path
    bandwidth
    latency
    utilization
    queueing
    KV transfer time
    metadata decision time
    total latency
    speedup
    scalability

The long-term purpose is to evolve this into a topology-aware simulator for
hardware-assisted KV-cache management.

Do not implement RTL.

Do not claim real Dynamo performance.

Do not fabricate hardware measurements.

Build a clean analytical architecture model that can later be validated
against actual FPGA measurements.
