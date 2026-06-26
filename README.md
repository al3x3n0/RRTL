# RRTL

**A data-parallel RTL simulator in Rust** — plus a typed RTL frontend, a
SystemVerilog importer/exporter, and a distributed runtime to drive it at scale.

RRTL compiles a design once and runs it through a ladder of simulation backends
that trade latency for throughput — scalar oracle, SIMD CPU batch, Cranelift
JIT, AOT C, bit-parallel / bit-sliced batch engines, and GPU — every one
differentially checked bit-for-bit against the others.

## Why RRTL

The differentiator is the **batch axis**: many independent instances of one
design (fuzzing, Monte-Carlo, design-space exploration, fault simulation).
Verilator has no batch axis — N instances is N processes — so the lane-parallel
engines pull far ahead there, while the JIT/AOT keep single-instance latency
within a small constant of Verilator. All numbers are bit-exact and reproducible
([methodology](docs/rrtl-optimizations.md)):

- **~96× a Verilator instance/core** (and ~186× on 8 cores) on gate-level
  *batch* simulation, identical netlist.
- **~1.7–3.2× Verilator** on single-instance designs via the Cranelift JIT;
  the AOT C backend narrows the remaining gap to ~2.8× on picorv32.
- **~93× a CPU core** for large GPU batches on Apple M3.
- The SystemVerilog frontend lowers, JITs, and correctly executes the full
  YosysHQ **picorv32** RV32I core.

→ **[docs/simulation.md](docs/simulation.md)** for the backend ladder, the
performance table, and how to reproduce every figure.

## Quick start

```rust
use rrtl_core::{lit_u, mux, uint, Design, Simulator};

let mut design = Design::new();
{
    let mut m = design.module("Counter");
    let clk = m.input("clk", uint(1));
    let rst = m.input("rst", uint(1));
    let en = m.input("en", uint(1));
    let out = m.output("out", uint(8));
    let count = m.reg("count", uint(8));

    m.clock(count, clk);
    m.reset(count, rst, 0);
    m.next(count, mux(en, count + lit_u(1, 8), count));
    m.assign(out, count);
}

design.validate().unwrap();
let compiled = design.compile().unwrap();

// Simulate (the scalar oracle every accelerated backend is checked against).
let mut sim = Simulator::new(&design, "Counter").unwrap();
sim.set(en, 1);
sim.tick();

// Or emit SystemVerilog.
println!("{}", rrtl_sv::emit(&design).unwrap());
```

## Crates

- `rrtl-ir` — core IR, identifiers, expressions, modules, diagnostics.
- `rrtl-core` — builder API, validation passes, scalar cycle simulation.
- `rrtl-sim-ir` — packed simulation IR and the accelerated CPU backends (SIMD
  batch, Cranelift JIT, AOT C, bit-parallel, bit-slice).
- `rrtl-gpu-sim` — batched GPU simulation research backend (`wgpu`).
- `rrtl-runtime` — distributed CPU/GPU lane runtime and partition planning.
- `rrtl-sv` / `rrtl-sv-frontend` — SystemVerilog export and import.
- `rrtl-riscv-asm` — RV32I assembler for simulation harnesses.
- `rrtl-macros` — procedural macros for concise structured RTL.
- `rrtl-pyrtl` — PyRTL block importer and `pyrtl2rrtl` CLI.
- `rrtl-surrogate` — RTL-derived surrogate model scaffold.

## Documentation

- [Simulation engines](docs/simulation.md) — the backend ladder and performance.
- [Optimization deep-dive](docs/rrtl-optimizations.md) — the full methodology and
  research write-up (paper draft).
- [RTL authoring](docs/authoring.md) — typed construction, the macro DSL, VCD
  tracing, assertions and covers.
- [Experimental GPU simulation](docs/gpu.md) — the `wgpu` batch path and autotune.
- [Distributed runtime](docs/runtime.md) — sharding huge batches across workers,
  checkpoints, and transports.
- [PyRTL bridge](docs/pyrtl.md) — import, corpus migration, and benchmarking.
- [RTL-derived surrogates](docs/surrogate.md) — surrogate ingestion and FAST
  orchestration.
