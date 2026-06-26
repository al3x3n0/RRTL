# AGENTS.md — guide for coding agents working in RRTL

This file orients an automated coding agent (OpenAI Codex and friends) inside the
RRTL repository. Read it before making changes. For the human-facing overview see
[`README.md`](README.md); for the simulator architecture and performance see
[`docs/simulation.md`](docs/simulation.md).

## What RRTL is

RRTL is a **data-parallel RTL (hardware) simulator** written in Rust, plus a
typed RTL frontend, a SystemVerilog importer/exporter, and a distributed runtime.
A design is compiled once and run through a *ladder* of simulation backends that
trade latency for throughput — scalar oracle → SIMD CPU batch → Cranelift JIT →
AOT C → bit-parallel / bit-sliced batch → GPU. The project's thesis is the
**batch axis** (many independent instances of one design: fuzzing, Monte-Carlo,
design-space exploration, fault simulation), where Verilator — which has no batch
axis — is left far behind.

## Workspace layout

Cargo workspace, Rust **edition 2021**, resolver 2. Crates and their dependency
direction (lower depends on higher):

| Crate | Role |
| --- | --- |
| `rrtl-ir` | core IR: identifiers, expressions, modules, diagnostics |
| `rrtl-core` | builder API, validation passes, **scalar cycle simulator** (the correctness oracle) |
| `rrtl-macros` | proc-macros (`bundle!`, `interface!`, `signals!`, `logic!`, …) |
| `rrtl-sv` | SystemVerilog **export** |
| `rrtl-sv-frontend` | SystemVerilog **import** (parser/lower); split into `ast/parser/lower/preprocess` modules |
| `rrtl-sim-ir` | **packed simulation IR** + accelerated CPU backends (SIMD batch, JIT, AOT, bit-parallel, bit-slice) |
| `rrtl-gpu-sim` | batched GPU simulation via `wgpu` (research backend) |
| `rrtl-runtime` | distributed CPU/GPU lane runtime, partition planning, checkpoints |
| `rrtl-pyrtl` | PyRTL block importer + `pyrtl2rrtl` CLI |
| `rrtl-surrogate` | RTL-derived surrogate-model scaffold |
| `rrtl-riscv-asm` | RV32I assembler for simulation harnesses |
| `rrtl` (root `.`) | umbrella crate: `pub use rrtl_core::*`, `rrtl::runtime`, `rrtl::sv`, `rrtl::ir`, macros |

The **packed simulation IR** in `rrtl-sim-ir` is the shared substrate every
accelerated backend consumes. Key type: `PackedMachineProgram` with streams
(`async_reset_comb`, `comb`, `tick_next`, `tick_commit`), each a `PackedBlock` of
packets (`instrs` + `effects`). Lower a compiled design with
`lower_to_packed_program` then `lower_to_machine_program`.

## Build & test

```sh
cargo build                      # whole workspace, default features
cargo test                       # workspace tests, default features
cargo test -p rrtl-sim-ir        # one crate

# The accelerated CPU backends are behind feature flags. Test them with:
cargo test -p rrtl-sim-ir --features "aot jit"
```

Feature flags that matter:

- `rrtl-sim-ir/jit` — Cranelift JIT backend (no external toolchain).
- `rrtl-sim-ir/aot` — AOT C backend. **Emits C and shells out to `clang -O3`**,
  then `dlopen`s the result. Requires `clang` on PATH (override with `CC=...`).
- `rrtl-pyrtl/onnx-ort`, `rrtl-surrogate/onnx-ort` — ONNX execution via `ort`.
- `rrtl-gpu-sim` always pulls `wgpu`; GPU tests no-op gracefully when no adapter
  is present.

Default-feature builds must stay warning-clean. When you add a method used only
under a feature, gate it (`#[cfg(feature = "aot")]`) so the default build does
not warn about dead code.

## Conventions (follow these)

1. **Differential testing is the contract.** Every accelerated backend is
   validated *bit-exact* against the scalar `rrtl_core::Simulator` oracle (or the
   `SimdCpuSimulator` batch oracle). A new backend or codegen path is not "done"
   until a test/example shows it matches the oracle on every output of every lane.
   Add a `#[cfg(test)]` unit test next to the engine.

2. **Examples are the benchmark/validation harnesses.** `rrtl-sim-ir` alone has
   ~67. Naming convention:
   - `*_check.rs` — bit-exact validation of a backend/feature.
   - `*_probe.rs` — **measure-first** sizing: estimate the payoff *before*
     building the engine (op-mix, cost models). Don't build a big engine without
     a probe that justifies it.
   - `*_bench.rs` — throughput comparison across backends.

   Run one with, e.g.,
   `cargo run --release --features "aot jit" -p rrtl-sim-ir --example bitslice_aot_bench`.

3. **Measure first; document honest negatives.** This codebase deliberately
   records optimizations that *did not* pay off (clang -O3 subsumes most
   value-level op tricks; the generic macro-op superoptimizer was neutral on
   GPU). When a probe refutes an idea, say so in the example output and in
   `docs/rrtl-optimizations.md` rather than shipping a non-win.

4. **Keep the paper in sync.** `docs/rrtl-optimizations.md` is an arXiv paper
   draft. Performance claims in README/docs must be reproducible from a cited
   example and match the paper. State the batch-vs-single-instance regime
   honestly (RRTL wins batch throughput; Verilator is strong on single-instance
   latency).

5. **Match surrounding style.** Mirror the comment density, naming, and idioms of
   the file you are editing. Reference code as `path:line`.

## Gotchas / sharp edges

- **`aot` needs a C compiler.** The AOT backend literally writes `.c`, runs
  `clang -O3 -shared -fPIC`, and `dlopen`s the `.dylib`/`.so`. No clang → the
  backend errors out (callers fall back). The bit-parallel/bit-slice AOT wins
  come specifically from clang **auto-vectorizing** an inner word loop; don't
  expect the Cranelift JIT to match it.
- **GPU watchdog.** Long single `tick_many` GPU dispatches can be killed by the
  OS GPU watchdog (Metal on macOS). Chunk long runs and `synchronize` between
  chunks.
- **x86 AVX2 paths can't be built on Apple Silicon.** Some SIMD paths are
  `x86_64`-gated; verifying them requires an x86 build host. Don't assume an
  arm64 dev machine compiled them.
- **Single-clock assumption.** Much of the engine assumes one clock. Multi-clock
  support is partial (a gold-oracle edge-gated slice plus some engine plumbing);
  check before relying on multi-domain behavior.
- **Backend scope limits (enforced at `new()`/`compile_lanes`, caller falls
  back):** bit-parallel = pure gate-level (every signal 1-bit, bitwise ops only);
  bit-slice = ≤128-bit, **no memory, no multiply**. GPU interp rejects external
  modules, inout, assertions, covers.
- **SV frontend coercion gap.** Signed↔unsigned assignment coercion is a known
  gap; `bench/sv/cpu.sv` is the unsigned variant for that reason.
- **Benchmark designs and Verilator harnesses** live in `bench/` (`bench/sv/*.sv`
  + `obj_dir/` Verilator builds, which are git-ignored). `picorv32.v` is the full
  YosysHQ core; `crc32.sv` is the GF(2)-linear block; `gate/` is the gate-level
  netlist used for the Verilator comparison.

## Git / PR conventions

- Branch off `main`; don't commit directly to the default branch.
- Commit and push **only when asked**.
- End commit messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
  (adjust the author line to your own agent identity as appropriate).
- Keep Verilator/C build artifacts out of commits — `.gitignore` already excludes
  `obj_dir/`, `*.o`, `target/`.

## Where to look first

- A new simulation optimization → `crates/rrtl-sim-ir/src/` (`lib.rs` for the
  packed IR; `jit.rs`, `aot.rs`, `bitparallel.rs`, `bitslice.rs`,
  `specialize.rs` for backends) and a matching `examples/*_probe.rs`.
- RTL authoring/semantics → `crates/rrtl-core/src/lib.rs`.
- SystemVerilog parsing → `crates/rrtl-sv-frontend/src/{parser,lower}.rs`.
- GPU → `crates/rrtl-gpu-sim/src/interp.rs` (design-as-data interpreter + WGSL).
- The full design rationale and measured results → `docs/rrtl-optimizations.md`.
