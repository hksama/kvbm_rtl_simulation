We already have an existing simplified PCIe simulator. DO NOT rewrite the architecture
from scratch. Extend the existing model with a PCIe Gen6+ latency-accurate abstraction.

Goal:
Model latency and concurrency for GPU <-> FPGA/RTL PCIe traffic, especially many
concurrent DMA streams targeting one FPGA. The model is NOT intended to simulate
analog electrical behavior or a complete PCIe controller. It should be a discrete-event
/ analytical latency model with configurable Gen6/Gen7 parameters.

PRIMARY GOAL
-----------
For N concurrent GPU/FPGA traffic streams, determine:

1. end-to-end latency
2. serialization latency
3. FLIT formation/accumulation latency
4. queueing latency
5. throughput
6. link utilization
7. P50/P95/P99/P99.9 latency
8. maximum sustainable concurrency
9. latency knee as utilization approaches saturation
10. FPGA buffer occupancy and overflow risk

PCIe GENERATION MODEL
---------------------
Support at least:

Gen5:
    32 GT/s
    NRZ
    128b/130b
    normal TLP mode

Gen6:
    64 GT/s
    PAM4
    2 bits/symbol
    256-byte FLIT
    236-byte TLP/DLP payload region
    6-byte DLP
    8-byte CRC
    6-byte FEC
    FLIT mode
    low-latency FEC
    FLIT-level ACK/NAK/replay

Gen7:
    128 GT/s
    PAM4
    retain the Gen6-style FLIT/FEC abstraction initially
    make all parameters configurable rather than hard-coding Gen6

Do not assume that every physical timing can be derived exactly from GT/s.
Use explicit PHY timing parameters so measured/specification values can be injected.

CORE BANDWIDTH FORMULAE
-----------------------

For Gen6:

R_GT = 64e9 GT/s/lane

PAM4 bits per symbol:
    b_symbol = 2

For W lanes:

R_bit = R_GT * b_symbol * W

R_raw_bytes = R_bit / 8

For x16:
    R_raw_bytes = 256e9 bytes/s

Do not confuse:
    raw line rate
with:
    useful TLP payload bandwidth.

Define:

eta_flit = flit_payload_bytes / flit_bytes

For Gen6:

flit_bytes = 256
flit_payload_bytes = 236

eta_flit = 236 / 256

Useful bandwidth should be:

B_useful =
    R_raw_bytes
    * eta_flit
    * eta_protocol

where eta_protocol accounts for TLP headers, alignment,
and any additional modeled protocol overhead.

Make eta_protocol configurable.

IMPORTANT:
PCI-SIG specifies up to 256 GB/s bidirectional for PCIe 6.0 x16.
Treat this as the raw/link-level headline value and separately model useful
payload bandwidth.

FLIT MODEL
----------

Each Gen6 FLIT is:

256 bytes total

with approximately:
236 bytes available for TLP/DLLP payload
6 bytes DLP
8 bytes CRC
6 bytes FEC

A FLIT may contain:
- multiple TLPs
- one TLP
- part of a TLP

Therefore do NOT model:
one TLP = one FLIT.

Implement a FLIT packer.

Given TLP payload size L:

N_flit =
    ceil((L + modeled_TLP_overhead) / 236)

Serialization time:

T_serial =
    N_flit * T_flit

where T_flit is an explicit parameter determined from:
    generation
    lane width
    PHY implementation

Do not derive FEC latency from an invented formula.
Use:

T_FEC_TX
T_FEC_RX

as configurable parameters.

FLIT ACCUMULATION
-----------------

Add:

T_accumulation

This is the time between TLP arrival at the TX scheduler
and availability of a complete FLIT for transmission.

For an analytical baseline, allow:

T_accumulation ~ Uniform(0, T_flit)

and therefore:

E[T_accumulation] = T_flit / 2

But for the discrete-event simulator, calculate it explicitly from
the arrival time of TLPs and the FLIT packer's current occupancy.

Do NOT always add T_flit/2 when the discrete-event model can calculate
the exact accumulation delay.

LATENCY DECOMPOSITION
---------------------

For a direct GPU -> FPGA transfer:

T_total =
    T_endpoint_GPU
  + T_queue_TX
  + T_accumulation
  + T_FEC_TX
  + T_serial
  + T_propagation
  + T_FEC_RX
  + T_endpoint_FPGA
  + T_queue_FPGA

For a PCIe switch topology:

T_total =
    T_endpoint_source
  + sum(T_queue_link_i)
  + sum(T_switch_i)
  + sum(T_accumulation_i)
  + sum(T_serial_i)
  + sum(T_propagation_i)
  + T_endpoint_destination

Keep each term separately measurable.

DO NOT collapse all latency into one number.

QUEUE MODEL
-----------

For each PCIe link define:

capacity B
queue depth Q
service time Ts
arrival rate lambda

Utilization:

rho = lambda / B

For N identical streams each generating bandwidth B_stream:

lambda_total = N * B_stream

rho = lambda_total / B

Use M/M/1 only as an analytical sanity check:

E[T_system] = Ts / (1 - rho)

E[T_queue] = rho * Ts / (1 - rho)

But the primary simulator should use a discrete-event queue rather
than relying on M/M/1 because PCIe traffic is bursty and packetized.

Also support deterministic/bursty traffic.

TRAFFIC MODEL
-------------

Each stream should support:

- source GPU ID
- destination FPGA ID
- direction
- average bandwidth
- transfer size
- burst size
- inter-arrival time
- packet/TLP size
- priority
- maximum outstanding requests
- read/write type

Support:

1. continuous streaming
2. periodic bursts
3. Poisson arrivals
4. deterministic arrivals
5. mixed workloads

TRANSFER SIZE
-------------

For a transfer of L bytes:

T_serial = effective_wire_bytes / B_raw

or, in FLIT mode:

N_flit = ceil((L + overhead) / 236)

T_serial = N_flit * T_flit

Keep both analytical and discrete-event calculations.

READ VS WRITE
-------------

Model these separately.

Posted write:

GPU -> FPGA:
    request/data
    no per-request completion required

Read:

GPU/FPGA -> remote memory:
    request
    request transmission
    remote service
    completion
    return path

For a read:

T_read =
    T_request
  + RTT
  + T_remote_service
  + T_completion

Support multiple outstanding reads.

BANDWIDTH-DELAY PRODUCT
-----------------------

Define:

BDP = B * RTT

For request payload size L_req:

N_outstanding_required =
    ceil(BDP / L_req)

This is a first-order estimate of the number of outstanding requests
required to keep the link busy.

Make this a diagnostic metric.

PCIe FLOW CONTROL / CREDITS
---------------------------

Add an optional credit model.

Track:

posted_data_credits
nonposted_data_credits
completion_data_credits
header_credits

A transfer cannot enter the transmit service stage if the required
credits are unavailable.

Credit stall latency:

T_credit_stall

must be reported separately from ordinary queueing latency.

Do not initially implement every PCIe credit rule. Build a parameterized
abstract credit model that can later be replaced with exact rules.

REPLAY / FEC
------------

Model Gen6:

raw error probability
FEC correction probability
uncorrectable FLIT probability
retry probability

Expected retry contribution:

E[T_retry] = P_retry * T_retry

For each simulation event, explicitly model whether a FLIT was:
- received correctly
- corrected by FEC
- detected as bad by CRC
- replayed

Do not treat FEC as a large queueing delay.
Use a configurable low fixed latency:

T_FEC_TX
T_FEC_RX

and allow the default values to be calibrated against specification/
measured hardware values.

TOPOLOGY
--------

Represent PCIe as a graph:

G = (V, E)

Nodes:
- Root Complex
- PCIe Switch
- GPU
- FPGA/RTL

Edges:
- PCIe links

Each edge has:

generation
width
bandwidth
FLIT size
FLIT transmission time
propagation latency
PHY latency
FEC latency
queue depth
credit capacity

A flow has a path:

P_f = {e1, e2, ..., ek}

End-to-end latency:

T_f =
    T_endpoint
  + sum(T_serial,e)
  + sum(T_queue,e)
  + sum(T_switch,e)
  + sum(T_phy,e)
  + sum(T_propagation,e)

BOTTLENECK DETECTION
--------------------

For every simulation run identify:

- most utilized link
- maximum queue
- queueing delay contribution
- serialization contribution
- FLIT accumulation contribution
- credit stall contribution
- endpoint processing contribution

Report:

rho_max

and the bottleneck edge.

FPGA BUFFER MODEL
-----------------

For each FPGA RX/TX queue define:

buffer_size
arrival_rate
consumption_rate

If:

B_in > B_out

buffer growth rate:

dQ/dt = B_in - B_out

Approximate overflow time:

T_overflow =
    buffer_size / (B_in - B_out)

For discrete-event simulation, calculate actual queue occupancy.

LATENCY DISTRIBUTION
--------------------

For every workload report:

mean
P50
P95
P99
P99.9
max

Also report:

T_fixed
T_serial
T_accumulation
T_queue
T_credit
T_retry
T_endpoint

as separate distributions where possible.

CONCURRENCY SWEEP
-----------------

Automatically sweep:

N =
1, 2, 4, 8, 16, 32, 64, 128, ...

For every N report:

aggregate offered bandwidth
aggregate achieved bandwidth
utilization
P50 latency
P99 latency
P99.9 latency
maximum queue depth
FPGA buffer occupancy

Calculate:

N_theoretical =
    floor(B_link / B_stream)

and:

N_safe =
    floor(rho_target * B_link / B_stream)

where rho_target is configurable, e.g. 0.7 or 0.8.

LATENCY KNEE
------------

Find the concurrency/utilization at which:

P99 latency exceeds a configurable threshold.

For example:

latency_target = 2 us

Find:

N_knee =
    first N such that P99_latency(N) > latency_target

Also find:

rho_knee

This is one of the main outputs of the simulator.

MULTI-GPU -> ONE FPGA
---------------------

Support:

GPU0 ─┐
GPU1 ─┤
GPU2 ─┤
GPU3 ─┼── PCIe Switch ── FPGA
GPU4 ─┤
GPU5 ─┤
GPU6 ─┤
GPU7 ─┘

All flows share the upstream bottleneck.

Aggregate load:

lambda = sum(lambda_i)

Utilization:

rho = lambda / B_bottleneck

Implement configurable scheduling:

FIFO
round-robin
weighted round-robin
strict priority
deadline/latency aware

Report per-GPU latency and fairness.

IMPORTANT:
Do not assume 8 GPUs means 8x bandwidth to the FPGA.
The physical PCIe topology determines the actual bottleneck.

NVLINK SHOULD BE MODELED SEPARATELY
------------------------------------

Add an optional NVLink fabric model later.

Do not merge NVLink and PCIe into one generic link.

Represent:

GPU <-> GPU:
    NVLink / NVSwitch

GPU <-> FPGA:
    PCIe

GPU <-> HBM:
    local GPU memory subsystem

For now, NVLink can be an independent graph with:
bandwidth
hop latency
switch latency
contention

METRICS
-------

Generate plots:

1. latency vs number of concurrent streams
2. P99 latency vs utilization
3. throughput vs concurrency
4. queue depth vs concurrency
5. FPGA buffer occupancy vs time
6. latency breakdown
7. bandwidth utilization per PCIe link
8. per-GPU latency
9. fairness across GPUs
10. latency CDF

VALIDATION
----------

Create analytical unit tests for:

1. one stream, no queueing
2. N streams below saturation
3. N streams exactly at saturation
4. N streams above saturation
5. small 64B TLP
6. 236B payload
7. 512B TLP spanning multiple FLITs
8. 4KB transfer
9. 1MB transfer
10. Gen5 x16 vs Gen6 x16
11. Gen6 x4/x8/x16
12. one GPU -> FPGA
13. eight GPUs -> one FPGA through switch
14. read vs write
15. retry event
16. FPGA RX buffer overflow

Keep the implementation modular.

DO NOT:
- implement a full PCIe protocol stack
- implement analog PAM4 simulation
- simulate Maxwell equations
- invent exact PHY/FEC timing
- hard-code one GPU or topology
- replace the existing simulator architecture

The goal is an architecture-level Gen6+ PCIe latency/concurrency simulator
with specification-grounded parameters and explicit latency decomposition.
