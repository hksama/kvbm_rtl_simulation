Tentative Problems Statements
1. Dynamic KV-cache manager / specialized allocator
What is it solving?
Dynamic management of KV-cache blocks during LLM inference.
Instead of treating KV memory as generic memory, the allocator uses properties such as:
block hotness
request lifetime
prefix sharing
reference count
prefill vs decode phase
memory pressure
possibly block size
to decide how KV blocks should be allocated, reused, retained, or evicted.
Why is it needed?
KV cache grows with context and concurrency.
Traditional approaches have real problems:
reserving maximum sequence length wastes memory
paging reduces fragmentation but introduces block-management overhead
smaller blocks reduce fragmentation but increase metadata/management overhead
shared prefixes create opportunities for reuse
long-context and multi-turn workloads create much larger KV footprints
This is the area where PagedAttention, vAttention, RadixAttention/SGLang, InfiniGen, etc. provide the existing foundation.
Main bottlenecks
​
The important question is:

    Can a KV-specific memory manager make better decisions than a generic/fixed block-management policy without adding more overhead than it saves?

Output
Your implementation:
​
2. RTL + Verilator KV-management engine
What is it solving?
The expensive/repetitive metadata decision-making involved in KV-cache management.
Instead of Rust doing every decision:
​
you investigate whether some of this bookkeeping can be performed by dedicated hardware.
Why is it needed?
KV management can involve huge numbers of tiny operations.
For example:
​
These operations aren't computationally complicated.
They're high-frequency metadata operations.
That makes them interesting for hardware acceleration.
Main bottlenecks
The hardware itself isn't doing matrix multiplication.
The bottleneck you're investigating is:
​
You want to know:

    Can a small specialized hardware engine process KV metadata decisions faster/cheaper than software?

What goes into the RTL?
For example:

```
                RTL ENGINE

       ┌─────────────────────┐
       │ KV metadata table   │
       ├─────────────────────┤
       │ hotness tracker     │
       │ refcount tracker    │
       │ access counter      │
       │ candidate selector  │
       │ scoring unit        │
       └──────────┬──────────┘
                  │
             decision
                  ↓
             KEEP/MOVE/EVICT
```

### What is Verilator doing?

You don't initially need an FPGA.

You can compile the SystemVerilog into a cycle-accurate-ish software simulation:

```
Rust
 │
 │ commands/events
 ▼
Verilator
``````
 │
 ▼
SystemVerilog RTL
 │
 ▼
placement decision
```

You measure:

```
cycles/decision
throughput
latency
area after synthesis
memory required
```
