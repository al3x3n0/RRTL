# Simulation Engines

RRTL is, first and foremost, a **data-parallel RTL simulator**. The same
compiled design can be run through a ladder of backends that trade latency for
throughput, all differentially checked bit-for-bit against a scalar reference
oracle.

The thesis is the **batch axis**: many independent instances of one design
(fuzzing, Monte-Carlo, design-space exploration, fault simulation, constrained-
random verification). Verilator has no batch axis — N instances means N
processes — so on that workload RRTL's lane-parallel engines pull far ahead.
For *single-instance latency*, Verilator remains strong; RRTL's JIT/AOT close
that gap to a small constant.

Full methodology, derivations, and the honest negatives are in
[`rrtl-optimizations.md`](rrtl-optimizations.md).

## The backend ladder

| Backend | Crate / feature | Regime | Role |
| --- | --- | --- | --- |
| Scalar cycle simulator | `rrtl-core` | single instance | deterministic oracle; correctness reference for everything below |
| Packed SIMD CPU batch | `rrtl-sim-ir` | batch (lanes) | interpreted multi-lane execution over packed SSA IR |
| Cranelift JIT | `rrtl-sim-ir` (`jit`) | single + batch | native straight-line code, no external toolchain |
| AOT C backend | `rrtl-sim-ir` (`aot`) | single + batch | emits C → `clang -O3` → `dlopen`; best single-instance latency |
| Bit-parallel gate-level | `rrtl-sim-ir` | gate-level batch | 1 bit/lane packed 64 lanes/word, bitwise-only |
| Bit-slice multi-bit | `rrtl-sim-ir` (`aot`) | multi-bit batch | W bit-planes/signal — every width at full lane density |
| GPU batch | `rrtl-gpu-sim` (`wgpu`) + CUDA/OpenCL codegen | huge batch | thousands of lanes on the GPU |

The packed simulation IR (`rrtl-sim-ir`) is the shared substrate: it flattens
hierarchy, lays scalar state out as `u32` limbs, and schedules combinational and
tick work into packet streams that every accelerated backend consumes.

## Performance highlights

All figures are bit-exact against the scalar oracle and reproducible from the
example commands below. The accelerated backends are research-grade and
validated on specific designs (noted per number); treat them as the measured
state of the engine, not a turnkey guarantee.

- **Gate-level batch — the moat.** The bit-parallel AOT runs a gate netlist at
  ~96× a single Verilator instance per core, and ~186× with 8-core rayon, on the
  identical netlist (`bench/sv/gate`, Verilator 5.030 measured at 19.7 Mcyc/s).
  The win is clang auto-vectorizing the packed `uint64_t` word loop.
- **Multi-bit batch.** The bit-slice AOT generalizes that packing to W-bit
  signals (W bit-planes, every width at full 64-lane/word density) and runs
  ~1.55× the vector JIT on `crc32` — a 32-bit datapath where the vector engine's
  fixed width-class wastes lanes on the many 1-bit control signals.
- **Single-instance latency.** The Cranelift JIT is ~1.7–3.2× Verilator on
  single-instance designs and ~20× the interpreter; the AOT C backend adds
  ~1.84× over the JIT on picorv32, narrowing the Verilator gap to ~2.8×.
- **GPU batch.** The CUDA/OpenCL per-design codegen runs bit-exact on Apple M3
  (crc32, a memory design, tensor-core-linear blocks) at ~93× a CPU core for
  large batches.
- **Real cores.** The SystemVerilog frontend lowers and JITs the full YosysHQ
  picorv32 and executes real RV32I correctly.

### Reproduce

```sh
# Gate-level bit-parallel: interp / SIMD-CPU / scalar-JIT / AOT / vector-JIT
cargo run --release --features "aot jit" -p rrtl-sim-ir --example bitparallel_bench

# Multi-bit bit-slice AOT vs the vector JIT (crc32), bit-exact + throughput
cargo run --release --features "aot jit" -p rrtl-sim-ir --example bitslice_aot_bench

# GPU batch vs packed CPU vs scalar lanes
cargo run --release -p rrtl-gpu-sim --example batch_bench
```

## Scalar reference

The deterministic cycle simulator in `rrtl-core` is the oracle every other
backend is checked against:

```rust
use rrtl_core::Simulator;

let mut sim = Simulator::new(&design, "Counter").unwrap();
sim.set(en, 1);
sim.tick();
let count = sim.get(out);
```

See [`gpu.md`](gpu.md) for the GPU batch path and
[`runtime.md`](runtime.md) for sharding huge batches across CPU/GPU workers.
