I am researching a hardware-assisted KV-cache metadata and placement engine
for distributed LLM inference.

The goal is NOT to simulate the GPU computation itself.

The goal is to build a parameterized architecture model that estimates when a
small specialized RTL engine for KV metadata management / placement decisions
could outperform a software implementation in latency and throughput, and
what scale/topology is required before the hardware becomes useful.

The model must be interactive, with sliders/controls for architectural and
workload parameters and live graphs.

IMPORTANT:
Do NOT assume that hardware is automatically faster.
The model must explicitly show the crossover point where hardware becomes
better, equal, or worse than software.

============================================================
1. SYSTEM BEING MODELED
============================================================

Model a disaggregated LLM inference system with:

                REQUEST
                   |
                   v
              ROUTER / META
                   |
          +--------+--------+
          |                 |
       PREFILL           DECODE
        POOL               POOL
          |                 |
       GPUs...            GPUs...
          |                 |
          +------ KV -------+
                 transfer

The hardware accelerator being investigated is a specialized KV metadata /
placement engine.

It does NOT initially own the actual KV tensor data.

It performs operations such as:

    lookup block metadata
    update access metadata
    update counters
    update timestamps
    update reference counts
    calculate hotness
    determine candidate blocks/workers
    calculate placement score
    select candidate

The actual KV data transfer is modeled separately.

============================================================
2. PRIMARY RESEARCH QUESTION
============================================================

The central question is:

"At what workload, KV-cache size, worker-pool size, request rate, topology,
and synchronization overhead does a specialized hardware KV metadata /
placement engine provide measurable benefit over a software implementation?"

The model should determine:

    - latency crossover point
    - throughput crossover point
    - maximum sustainable request rate
    - metadata operations/sec
    - hardware utilization
    - software CPU utilization estimate
    - coordination overhead
    - topology-induced latency
    - incorrect/stale placement consequences
    - scalability limits

============================================================
3. DO NOT MODEL ONLY A SINGLE 'HARDWARE TIME'
============================================================

Separate the total decision latency into components.

For SOFTWARE:

    T_sw =
        T_lookup_sw
      + T_metadata_update_sw
      + T_score_sw
      + T_candidate_selection_sw
      + T_coordination_sw
      + T_queue_sw

For HARDWARE:

    T_hw =
        T_request_arrival
      + T_hw_lookup
      + T_metadata_update
      + T_score
      + T_selection
      + T_hw_to_sw
      + T_coordination
      + T_queue

Do not assume these values are constant.

Allow a configurable model:

    T_hw(N) = base_hw_latency + f(N, concurrency, metadata_size)

and similarly:

    T_sw(N) = base_sw_latency + g(N, metadata_size, contention)

where N may represent:

    number of GPUs
    number of workers
    number of metadata entries
    number of candidates

depending on the operation.

============================================================
4. PARAMETERS
============================================================

Create sliders / numeric controls for the following.

------------------------------------------------------------
A. WORKLOAD PARAMETERS
------------------------------------------------------------

1. Requests per second
2. Concurrent requests
3. Input tokens/request
4. Output tokens/request
5. Block size in tokens
6. KV blocks/request
7. Fraction of requests with reusable prefix
8. Prefix reuse percentage
9. Block accesses/request
10. Metadata operations/request
11. Burstiness
12. Request arrival distribution

Allow workloads such as:

    light
    medium
    heavy
    bursty
    highly cache-reuse-heavy
    low-cache-reuse

Also allow manual parameter control.

------------------------------------------------------------
B. KV METADATA PARAMETERS
------------------------------------------------------------

1. Number of KV blocks
2. Metadata bytes/block
3. Number of metadata entries
4. Metadata lookup operations/request
5. Metadata update operations/request
6. Access counter width
7. Timestamp width
8. Reference counter width
9. Hotness calculation complexity
10. Candidate count K

Candidate count should be configurable:

    K = 1, 2, 4, 8, 16, 32, 64

------------------------------------------------------------
C. SOFTWARE BASELINE
------------------------------------------------------------

Model a software KV metadata manager.

Parameters:

    software lookup latency
    software metadata update latency
    software score calculation latency
    software candidate selection latency
    CPU cores available
    software parallelism
    memory access latency
    cache hit rate
    CPU contention
    synchronization overhead

Allow software throughput to be limited by:

    CPU cycles
    memory accesses
    lock contention
    synchronization

Do NOT model software as a single serial operation if concurrency is enabled.

============================================================
5. RTL PARAMETERS
============================================================

Allow:

    RTL clock frequency
    pipeline depth
    number of lookup ports
    number of metadata banks
    number of parallel metadata operations
    candidate comparison units
    scoring units
    queue depth
    metadata storage capacity
    metadata memory bandwidth

Example:

    250 MHz
    500 MHz
    750 MHz
    1 GHz

Number of parallel engines:

    1
    2
    4
    8
    16

The model should calculate:

    cycles/operation
    operations/cycle
    operations/sec
    theoretical throughput
    sustainable throughput
    queueing delay
    utilization

============================================================
6. HOTNESS MODEL
============================================================

The first implementation should focus specifically on a hotness tracker.

Model hotness using a configurable function.

Example:

    hotness =
        alpha * access_count
      + beta * recency
      + gamma * reference_count

Allow the user to modify:

    alpha
    beta
    gamma

Also allow alternative models:

    LRU-like
    LFU-like
    recency-weighted
    exponentially decayed frequency

For example:

    H(t) = access_count * exp(-lambda * age)

Allow lambda to be controlled.

Plot:

    hotness distribution
    hot/cold classification
    metadata update rate
    hardware cycles per hotness update

============================================================
7. TOPOLOGY MODEL
============================================================

Model a hierarchical datacenter topology.

                    CLUSTER
                       |
            +----------+----------+
            |                     |
          RACK 0                RACK 1
            |                     |
       +----+----+           +----+----+
       |         |           |         |
     NODE      NODE        NODE      NODE
       |
    +--+--+
    |     |
   GPU   GPU

Allow:

    GPUs/node
    nodes/rack
    racks
    total GPUs
    prefill GPUs
    decode GPUs

Each connection has:

    bandwidth
    latency
    hop count

At minimum model:

    same GPU
    same node
    same rack
    different rack

Allow custom hop latency.

============================================================
8. HOP MODEL
============================================================

Do NOT use only "hop count".

Model:

    T_network =
        base_link_latency
      + hops * hop_latency
      + serialization_time
      + congestion_penalty

Allow:

    hop_latency
    bandwidth
    message size
    congestion factor

For example:

    T_transfer =
        latency
      + message_size / bandwidth
      + congestion_penalty

The metadata message should be modeled separately from the actual KV data.

============================================================
9. COORDINATION MODEL
============================================================

This is a critical parameter.

Model coordination between:

    software router
    RTL engine
    prefill workers
    decode workers
    distributed RTL engines

Parameters:

    coordination latency
    event propagation latency
    metadata synchronization interval
    event batching size
    stale metadata window
    queueing delay
    synchronization frequency

Allow:

    synchronous coordination
    asynchronous coordination
    batched event propagation

============================================================
10. STALE METADATA / CORRECTNESS MODEL
============================================================

Do NOT define correctness simply as cache hit/miss.

Separate:

A. Metadata correctness

    Does the hardware correctly know where a block exists?

B. Placement correctness

    Did it select a worker that actually satisfies the requested KV state?

C. Staleness

    Was the metadata valid at decision time?

D. Cache hit

    Did the selected worker contain the required KV blocks?

E. Avoidable miss

    Did another eligible worker have the KV blocks, but stale metadata caused
    the engine to miss it?

F. Bad placement

    Did the engine select a valid worker but one with substantially worse
    topology / load?

G. Stall

    Did the decode worker have to wait for KV transfer?

Define the following metrics:

    true_positive_cache_hit
    true_negative_cache_miss
    false_negative_cache_lookup
    false_positive_cache_lookup
    avoidable_miss_rate
    stale_decision_rate
    bad_placement_rate
    unnecessary_transfer_rate
    KV_wait_time

Most importantly:

    avoidable_miss_rate =
        misses caused by incorrect/stale metadata
        /
        total routing opportunities

This is more useful than raw cache miss rate.

============================================================
11. PREFILL SCALING CHECKPOINTS
============================================================

Implement the simulation in checkpoints.

CHECKPOINT 1:

    1 prefill GPU
    1 RTL engine

Measure:

    metadata operations/sec
    hardware latency
    software latency
    throughput
    utilization

CHECKPOINT 2:

    2, 4, 8, 16 prefill GPUs

CHECKPOINT 3:

    32, 64, 128 prefill GPUs

CHECKPOINT 4:

    Continue increasing prefill GPUs until one RTL engine saturates.

Determine:

    maximum prefill GPUs supported per RTL engine

under a specified workload.

This is an important output.

Plot:

    prefill GPUs
        vs
    RTL utilization

    prefill GPUs
        vs
    decision latency

    prefill GPUs
        vs
    metadata operations/sec

============================================================
12. DISTRIBUTED PREFILL CHECKPOINT
============================================================

Once a single RTL engine saturates, introduce:

    multiple RTL engines

Example:

    1 engine
    2 engines
    4 engines
    8 engines
    16 engines

Compare architectures:

A. centralized metadata engine

B. sharded metadata engines

C. replicated metadata engines

D. hierarchical metadata engines

For sharding:

    prefix/block -> shard

For hierarchical:

    global metadata
          |
       rack metadata
          |
       node metadata

Measure:

    coordination traffic
    synchronization latency
    lookup latency
    scalability
    stale metadata
    throughput

Determine the scaling point where distributed hardware becomes necessary.

============================================================
13. DECODE CHECKPOINT
============================================================

Repeat the experiment for decode GPUs.

Start with:

    1 decode GPU

Then:

    2
    4
    8
    16
    32
    64
    128
    ...

Determine:

    maximum decode GPUs per RTL engine

under decode-style metadata access patterns.

Important:

Decode workloads are likely to have different access behavior from prefill.

Model:

    higher access frequency
    smaller per-request work
    more frequent metadata updates
    longer-lived active KV blocks

Do NOT assume prefill and decode have identical workloads.

============================================================
14. DISTRIBUTED DECODE CHECKPOINT
============================================================

Create a decode-only distributed topology.

Do NOT connect it to prefill yet.

Compare:

    centralized
    sharded
    replicated
    hierarchical

Measure:

    lookup latency
    synchronization traffic
    stale state
    throughput
    scaling efficiency

============================================================
15. PREFILL + DECODE CHECKPOINT
============================================================

Finally connect:

    PREFILL POOL
         |
         | KV metadata / transfer metadata
         |
         v
    DECODE POOL

Model:

    prefill worker selection
    decode worker selection
    KV availability
    KV transfer
    topology
    coordination
    stale metadata
    queueing
    transfer latency

The end-to-end latency should be approximately:

    T_total =
        T_prefill
      + T_metadata_prefill
      + T_worker_selection
      + T_coordination
      + T_KV_transfer
      + T_decode_metadata
      + T_decode_wait

Do not assume all components are sequential.

Allow overlapping operations where appropriate.

============================================================
16. TOPOLOGY-AWARE PLACEMENT
============================================================

Implement a placement score:

    Score(worker) =
        w_cache * cache_overlap
      - w_load * worker_load
      - w_network * topology_cost
      - w_staleness * metadata_age
      - w_transfer * estimated_transfer_cost

Allow all weights to be sliders.

The selected worker is:

    argmax Score(worker)

or equivalent minimum-cost formulation.

Plot how changing the weights changes:

    selected worker
    cache hit rate
    transfer distance
    latency
    load balance

============================================================
17. HARDWARE VS SOFTWARE CROSSOVER
============================================================

This is the most important visualization.

For each workload, calculate:

    Speedup =
        T_software / T_hardware

Plot:

    x-axis:
        number of GPUs

    y-axis:
        speedup

Add a horizontal line:

    speedup = 1

The crossing point is:

    hardware crossover point

Also plot:

    requests/sec vs speedup
    metadata entries vs speedup
    concurrency vs speedup
    candidate count vs speedup
    synchronization latency vs speedup
    hop latency vs speedup
    stale metadata rate vs speedup

Clearly identify the region:

    HARDWARE WORSE
    BREAK-EVEN
    HARDWARE BETTER

============================================================
18. THROUGHPUT MODEL
============================================================

Calculate:

    software_capacity
    hardware_capacity
    workload_rate

For each architecture:

    sustainable_RPS =
        min(
            metadata_processing_capacity,
            memory_bandwidth_capacity,
            coordination_capacity,
            worker_capacity
        )

Plot:

    offered RPS
        vs
    achieved RPS

Also plot utilization.

============================================================
19. QUEUEING
============================================================

Do not only calculate raw service time.

If:

    arrival_rate >= service_rate

the engine should saturate.

Model queueing.

At minimum:

    utilization = arrival_rate / service_rate

and approximate queueing latency.

Show:

    average latency
    p95 latency
    p99 latency

This is especially important because a hardware engine may have a lower
single-operation latency but still become a bottleneck when request rate
increases.

============================================================
20. RESOURCE MODEL
============================================================

Estimate hardware resources parametrically.

Inputs:

    metadata entries
    bits/entry
    number of banks
    number of lookup ports
    number of scoring units
    pipeline depth

Outputs:

    metadata storage bits
    estimated BRAM
    estimated URAM
    comparator count
    arithmetic unit count
    pipeline registers

Do NOT claim exact FPGA resource utilization.

Call these:

    analytical estimates

until actual synthesis is performed.

============================================================
21. REQUIRED GRAPHS
============================================================

Create at least these interactive plots:

1. GPU count vs software latency
2. GPU count vs RTL latency
3. GPU count vs speedup
4. GPU count vs RTL utilization
5. RPS vs speedup
6. Metadata entries vs speedup
7. Candidate count vs latency
8. Coordination latency vs speedup
9. Hop latency vs speedup
10. Synchronization frequency vs stale metadata rate
11. Synchronization frequency vs coordination overhead
12. Cache reuse vs speedup
13. Cache reuse vs avoidable miss rate
14. Workload intensity vs hardware utilization
15. Hardware engines vs aggregate throughput
16. Hardware engines vs synchronization overhead
17. Network topology vs KV wait time
18. Prefill GPU count vs maximum sustainable RPS
19. Decode GPU count vs maximum sustainable RPS
20. End-to-end prefill-to-decode latency

============================================================
22. MOST IMPORTANT GRAPH
============================================================

Create a 2D heatmap:

    X = number of GPUs
    Y = requests/sec

Cell value:

    hardware_speedup / software_speedup

Clearly mark:

    hardware worse
    break-even
    hardware better

Create another heatmap:

    X = coordination latency
    Y = metadata operations/request

Cell value:

    hardware speedup

This should expose where the hardware architecture is actually useful.

============================================================
23. CHECKPOINT MODE
============================================================

The application should have a dropdown:

    Checkpoint 1:
        Single Prefill GPU

    Checkpoint 2:
        Prefill Scaling

    Checkpoint 3:
        Distributed Prefill

    Checkpoint 4:
        Single Decode GPU

    Checkpoint 5:
        Decode Scaling

    Checkpoint 6:
        Distributed Decode

    Checkpoint 7:
        Integrated Prefill + Decode

Each checkpoint should expose only the parameters relevant to that stage.

============================================================
24. OUTPUT SUMMARY
============================================================

At the top of the application display:

    Current configuration

    Software:
        X ops/s
        Y us decision latency

    RTL:
        X ops/s
        Y us decision latency

    Speedup:
        X×

    RTL utilization:
        X%

    Metadata capacity:
        X entries

    Coordination overhead:
        X us

    Avoidable miss rate:
        X%

    KV wait time:
        X us

    Maximum GPUs supported:
        X

    Hardware crossover:
        X GPUs / X RPS

============================================================
25. IMPORTANT SCIENTIFIC CONSTRAINT
============================================================

Do not fabricate real NVIDIA Dynamo performance numbers.

Separate:

    measured values
    assumed values
    analytically derived values

Every parameter should be clearly labelled as:

    ASSUMPTION
    MEASURED
    DERIVED

Provide a configuration file containing all assumptions.

The model is a design-space exploration tool, NOT a benchmark of Dynamo.

The final objective is to tell me:

"Under these workload and topology assumptions, a single RTL engine becomes
useful above approximately X metadata operations/sec or Y workers, and
distributed hardware becomes necessary above approximately Z workers."

============================================================
26. IMPLEMENTATION
============================================================

Build this as a local interactive web application.

Use:

    Python backend/model OR TypeScript model
    Plotly for interactive graphs
    sliders for parameters
    dropdown for checkpoint
    heatmaps
    line charts
    tables

Keep the analytical model separate from the UI.

Create:

    /model
    /ui
    /configs
    /tests

The model must be deterministic.

Add unit tests for:

    latency calculation
    throughput calculation
    topology cost
    cache hit/miss classification
    stale metadata
    placement score
    hardware/software crossover
    queue saturation

Do not implement RTL yet.

The purpose of this application is to determine what RTL architecture is
worth implementing and at what scale.
