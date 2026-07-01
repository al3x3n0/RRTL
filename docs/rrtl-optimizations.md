# RRTL: Optimization Techniques for a Data-Parallel RTL Simulator

> Working notes toward an ArXiv paper on RRTL. Each section is written to map onto
> a paper section: the *idea*, the *mechanism* (how RRTL implements it), the
> *measured result* on Apple Silicon (M-series, wgpu/Metal, unified memory), and
> *honest limitations*. Numbers are best-of-N wall-clock with warm-up and cooldown;
> "Mlc/s" = million lane-cycles per second (lanes × steps / seconds).

## Abstract (draft)

Verilator compiles one RTL design into a large oblivious 2-state C++ program that
evaluates all logic every cycle. This wins on single-instance latency but scales
poorly in two regimes: (a) *throughput* across many stimulus instances, where the
data-parallelism is left on the table, and (b) *compile/iteration*, where huge
designs produce multi-hour C++ builds and multi-GB objects. RRTL targets both with
a **packed SSA simulation IR** executed by **data-parallel lane engines** — a
SIMD-vectorized CPU backend (NEON/AVX2 across stimulus lanes) and a **design-as-data
GPU interpreter** (a fixed O(1) WGSL kernel that interprets a design uploaded as
buffers, instead of generating a per-design shader). On a mixed-width datapath the
GPU interpreter sustains ~20–25× the CPU-SIMD lane throughput while remaining
bit-exact, and sidesteps the shader-compile wall entirely. Where peak throughput
matters more than iteration speed, we additionally emit a **compiled per-design GPU
kernel** (OpenCL/CUDA, the GPU analogue of an AOT backend) that runs bit-exact on a
real Apple M3 GPU at **~93× a single fast CPU core** in batch, and we show that
GF(2)-linear cones (CRC/FEC/crypto) reduce to a **1-bit matrix product** that maps
directly onto tensor-core BMMA — offloadable selectively, per register cone, within
a mixed design. The same linear structure also admits a **temporal leap**: the
N-cycle evolution of a linear block is `M^N`, so its state can be jumped N cycles in
**O(log N)** matrix mults (e.g. 2⁴⁰ cycles in 41), and a single varying-input stream
folds into P independent segments — an *algorithmic* speedup, not a constant factor. For **gate-level netlists** — the regime where beating Verilator at
scale matters most — a **bit-parallel** engine packs 64 lanes per machine word and
evaluates each gate as one bitwise op; compiled by clang -O3 (auto-vectorized to
128 lanes/op) and run multicore, it sustains **~186× a single Verilator instance** in
batch, bit-exact. We further describe a register-cone partitioner for single-design
multi-core latency, and two instructive negative results: heterogeneous CPU‖GPU
lane-splitting is counterproductive on unified-memory systems, and an automatic
fused-superoperator (a Souper-style RTL superoptimizer) is value-traffic-neutral on
the interpreter — fusion is best delegated to the compiled backends, which
register-allocate single-use intermediates for free.

---

## 1. Background: where Verilator's model leaves room

- **Oblivious 2-state cycle model.** Verilator evaluates the full cone every cycle.
  Excellent constant factor for one instance; no exploitation of *batch*
  data-parallelism (many stimuli / seeds / parameter points).
- **Compile-time blowup.** A single huge design lowers to one enormous C++ program;
  build times reach hours and object files reach gigabytes. Iteration suffers.
- **Single-instance serial by default.** `--threads` adds static partitioning, but
  the base model is one design, one stimulus, advancing serially.

RRTL's thesis: represent the design *as data* over a compact IR, then exploit the
two axes Verilator ignores — **lane (stimulus) parallelism** and **avoiding the
giant per-design compile**.

## 2. The packed simulation IR

The core is `rrtl-sim-ir`: an SSA-like packed machine program (`PackedMachineProgram`)
of `PackedInstr`/`PackedEffect` grouped into `PackedBlock`s over four ordered streams:

1. `async_reset_comb` — combinational logic feeding async resets
2. `comb` — the main combinational cone
3. `tick_next` — next-state computation for sequential elements
4. `tick_commit` — commit of next-state into state

This separation makes a cycle a deterministic sequence of stream evaluations and
gives a clean place to express both register semantics and reset priority. The IR is
**oblivious by construction today** (evaluates every packet each cycle) — activity
skipping (ESSENT-style sparse-conditional evaluation) is the obvious future axis and
is explicitly *not yet* implemented.

### 2.1 Storage layout: WordMajor / lane-coalesced

State and value words are laid out **WordMajor**: address `(offset + limb) * lanes + lane`.
All lanes of a given word are contiguous. This is the single most important layout
decision for both backends:

- On CPU it makes a word's lane-vector a contiguous load → one NEON/AVX2 vector.
- On GPU it makes lane `l` = thread `l`; adjacent threads touch adjacent words →
  fully coalesced global-memory access.

## 3. Lane-vectorized CPU backend

`SimdCpuSimulator` executes the packed IR with one SIMD lane per stimulus instance
(NEON on Apple Silicon, AVX2 on x86_64). Because of WordMajor layout, each scalar op
in the IR becomes one vector op across `W` lanes. This is the throughput baseline the
GPU is measured against, and the per-group engine inside the multi-core partitioner.

> Note: x86_64/AVX2 paths are feature-gated and cannot be compiled on the dev
> machine (no rustup toolchain); they require separate CI verification.

## 4. Design-as-data GPU interpreter (the central contribution)

### 4.1 Why not codegen a shader

The natural GPU path — generate a WGSL shader specialized to the design — **does not
scale**. For a large design the generated shader reaches ~115 MB of WGSL and the
Metal shader compiler hangs. Shader size grows with design size, so the codegen
approach has the same compile-wall pathology as Verilator's C++, just relocated to
the GPU driver.

### 4.2 The fix: one fixed kernel that interprets data

RRTL instead ships a **single fixed-size WGSL kernel** (`INTERP_WGSL`) that reads the
design from storage buffers and interprets it. Shader size is **O(1)** in design size;
the design lives in `code`/`aux` buffers. There is no per-design shader compile — the
same compiled kernel runs every design. This is the GPU analogue of an interpreter vs
a JIT, chosen precisely because the "JIT" (codegen) blows the driver up.

Buffers (8 storage bindings; requires `adapter.limits()`, since `downlevel_defaults`
caps storage buffers at 4):

| binding | buffer | contents |
|---|---|---|
| 0 | `sig` | current signal/state words (WordMajor) |
| 1 | `reg_next` | next-state scratch |
| 2 | `values` | per-instruction SSA value workspace |
| 3 | `code` | encoded instruction stream |
| 4 | `captured` | `[offset, limbs]` pairs for register commit |
| 5 | `params` | dispatch ranges, lane count, ml, voff base, … |
| 6 | `mem` | memory arrays |
| 7 | `aux` | immediates, operand lists, reset entries, voff table |

One GPU thread = one lane. The kernel walks the four stream ranges (`params`-driven),
evaluating each packet by opcode dispatch (opcodes 0–24: bitwise, add/sub, mul, mux,
slice/zext/sext, eq/ne, concat, load/store, mem read/write, async/sync capture).

### 4.3 Multi-limb arithmetic without u64/u128

WGSL has no 64-bit integers. RRTL stores values as **32-bit limbs** and implements
multi-precision arithmetic on `array<u32,4>` (≤128-bit):

- **add/sub**: limb-wise with explicit carry/borrow.
- **multiply**: the hard case. Split each 32-bit limb into two 16-bit halves; a
  16×16→32 partial product fits in `u32`, and the accumulation
  `ai*bj + acc[k] + carry` is provably ≤ 2³²−1, so it never overflows `u32`.
- **compare** (eq/ne/lt): MSB-limb-first.
- **slice/sext/concat**: cross-limb bit plumbing.

WGSL gotcha worth a paper footnote: a dynamically-indexed local array must be declared
`var`, not `let`, or the shader fails to compile ("expression may only be indexed by a
constant"). This bit every multi-limb helper.

### 4.4 Per-op width dispatch

A uniform `max_limbs` workspace (every value zero-extended to the widest value's limb
count) is simple but taxes mixed-width designs: real designs are *mostly* 1-bit control
+ 32-bit data with a *few* 64-bit addresses, yet one 64-bit signal forces
`max_limbs = 2` everywhere.

**Measured tax:** a mostly-32-bit datapath plus one barely-used 64-bit register dropped
GPU throughput from 3.3 → 1.4 Mlc/s — **2.4× slower** despite ~99% of the work being
32-bit.

Fix: each op carries its own result width `nl = limbs_of(width)` and operand width
`ol` (encoded in field `c` for eq/ne/zext/slice; sext already carried it). An op reads
and writes only its own limbs, so stale high limbs are never observed.
**Result:** the mixed design recovered 1.4 → 2.1 Mlc/s (tax 2.4× → ~1.5×). A dedicated
straight-line 1-limb kernel (`run_block1`) handles `max_limbs == 1` designs with zero
multi-limb overhead, so pure-32-bit designs are unaffected.

### 4.5 Per-value layout packing

Even with per-op dispatch, the value *workspace* was still strided at `max_limbs`
(`values[(v·max_limbs + l)·lanes + lane]`), so a 32-bit value occupied 2 words under
`max_limbs = 2`. Packing replaces the uniform stride with a per-value offset table:
`value_limbs[v]` = max limbs ever written to value-id `v`; a cumulative prefix sum
`voff[v]` (stored in `aux`) gives each value a tight base; the workspace shrinks from
`num_values·max_limbs` to `Σ value_limbs`. The kernel indexes every value by
`voff[id] = aux[value_offsets_base + id]`. The 1-limb path is unaffected because
`max_limbs = 1 ⇒ voff[v] = v`.

**Honest result (a paper should report this as a negative/nuanced result).** Packing is
bit-exact and shrinks the workspace, but on the mixed benchmark it did **not** move
throughput (2.1 → 2.2 Mlc/s, within noise). Controlled measurement isolates the real
cost:

| design (lanes=16384, width=64, depth=8) | kernel path | Mlc/s |
|---|---|---|
| pure 32-bit | `run_block1` (straight-line scalar) | 3.2 |
| mixed (one 64-bit reg, ops mostly 1-limb) | `run_block_ml`, ml=2 | 2.2 |
| pure 64-bit (ops genuinely 2-limb) | `run_block_ml`, ml=2 | 0.8 |

The mixed case sits *between* pure-32 and pure-64, confirming per-op width dispatch
already does its job. The residual 3.2 → 2.2 (~1.45×) is therefore **not** value-buffer
stride. We decomposed it into two costs by adding a **per-op 1-limb fast path** to
`run_block_ml` (§4.5.1).

### 4.5.1 Per-op 1-limb fast path, and what it reveals

When an op's result *and* operands fit in one limb, the multi-limb body's
`array<u32,4>` machinery is pure overhead. We added a scalar fast path that the kernel
takes per-instruction when the op is "narrow." The narrowness test is derived purely at
runtime — **no extra encoder metadata**: because each SSA value has a single width,
`nl = limbs_of(result) = value_limbs[dst]`; for same-width ops `nl ≤ 1` suffices, and
only the width-changing ops (eq/ne/slt/sge/slice/zext/sext, which carry operand width in
field `c`) additionally require `ol ≤ 1` so a scalar compare/slice never reads a second
source limb.

Two results worth a paper paragraph each:

1. **Inlining matters on GPUs.** Implemented first as a *called* helper (one call per
   narrow instruction), the fast path *regressed* the mixed design 2.2 → 1.9 Mlc/s:
   Metal does not inline the large opcode switch, so we paid a function call per
   instruction — whereas the multi-limb `run_block_ml` and the 1-limb `run_block1` each
   loop internally (one call per block). **Inlining** the same scalar switch directly
   into the loop body instead recovered the mixed design to ~2.4 Mlc/s (stable 2.3–2.5),
   with pure-32 unchanged at 3.1 and all outputs bit-exact. Lesson: per-instruction
   function calls in shader code are not free; hot per-op interpreter code must be
   inlined.

2. **The remaining gap is pointer indirection, not arrays.** Even with the scalar fast
   path, the mixed design (~2.4) does not reach pure-32 (3.2). The difference is the
   per-value packing indirection: `run_block1` reads `values[id·lanes + lane]` directly,
   while the packed path reads `values[aux[voffBase + id]·lanes + lane]` — an extra `aux`
   load per operand. That indirection is the floor the packed path cannot beat. So the
   ~1.45× decomposes into (a) the `array<u32,4>` machinery — recoverable, ~half — and
   (b) the `voff` indirection load — inherent to packing. Removing (b) would mean
   abandoning packing for designs that don't need the footprint win; we judged the
   remaining headroom not worth the layout-mode complexity.

So packing's net effect is a **memory-footprint/bandwidth** optimization (helps
genuinely-wide designs and high lane counts where the workspace overflows cache), with
its indirection cost partly offset by the per-op fast path on mixed-width designs.

### 4.6 Headline throughput

Across widths, the GPU interpreter sustains ~20–25× CPU-SIMD lane throughput while
remaining bit-exact (verified op-by-op against `SimdCpuSimulator`):

| width | CPU-SIMD | GPU-interp | speedup |
|---|---|---|---|
| 32-bit (1 limb) | 0.1–0.2 | 3.2–3.3 | ~20–25× |
| 64-bit (2 limb) | ~0.1 | 0.8–1.2 | ~20–23× |

Genuine 64-bit arithmetic costs ~2.7–4× vs 32-bit (3.3 → 0.8–1.2) — that is *real*
work (a 64-bit multiply is four 16-bit half-products), not overhead.

### 4.7 Engineering notes for reproducibility

- Dispatch is **async**: `tick_many` enqueues without polling; `synchronize` polls.
  This lets host work and successive kernels overlap.
- Workgroup size 256 (`INTERP_DEFAULT_WORKGROUP`).
- Register commit copies *all* limbs of a captured value (an early bug copied only
  limb 0, silently corrupting multi-limb registers); `captured` is `[offset, limbs]`
  pairs precisely so the kernel knows how many limbs to commit.
- Metal can hang the process; all GPU example runs use a background launch + kill-after-
  timeout guard.

## 4b. AOT design specialization (a static optimizing pass over the IR)

The central design-specific thesis: **static, value-oblivious specialization of the
packed IR composes perfectly with the data-parallel lane model** (it produces another
oblivious program — no runtime control-flow divergence), whereas dynamic activity-based
skipping fights SIMT/SIMD (lanes diverge; a warp must do a cone's work if *any* lane
needs it). So RRTL recovers much of Verilator's design-specific constant-factor wins
*without* surrendering batch throughput and *without* the giant per-design compile (the
specialized IR is still interpreted as data).

`rrtl-sim-ir`'s `specialize` module is the front of this pipeline. Pass 1 is constant
folding + algebraic identities (`x&1s→x`, `x|0→x`, `x*0→0`, `x*1→x`, `x-x→0`,
`a==a→1`, const-cond mux, …) + copy propagation + dead-code elimination + value-id
compaction, run per block (SSA value ids are block-local and ordered defs-before-uses;
within-block CSE already happens during lowering). It is verified bit-exact against the
unspecialized program — including 64-bit multi-limb folds — and idempotent (one forward
pass reaches the constant-propagation fixpoint because defs precede uses).

### 4b.1 Result: the payoff scales with the *cost* of what's eliminated

On a datapath whose per-stage logic is partly constant-foldable (a stand-in for the
tie-off/config logic pervasive in real SoCs), specialization removed **37.7%** of
instructions. But throughput moved very differently by width:

| datapath | instrs removed | GPU orig | GPU specialized | speedup |
|---|---|---|---|---|
| 32-bit (1 limb) | 37.7% | 2.3 Mlc/s | 2.4 Mlc/s | **1.05×** |
| 64-bit (2 limb) | 37.7% | 0.8 Mlc/s | 1.3 Mlc/s | **1.68×** |

The *same structural reduction* yields a negligible 32-bit win but a large 64-bit win.
This is a direct corroboration of the bottleneck decomposition in §4.5: the 32-bit
kernel is **bandwidth-bound**, so cutting (cheap) instructions barely helps; the 64-bit
kernel is **compute-bound on multi-limb multiply**, so eliminating expensive ops pays.
The lesson for the specializer is that its leverage is greatest exactly where ops are
expensive (multi-limb arithmetic) — which is also where the limb tax of §4.4–4.5 hurts
most. (Honesty note: the 64-bit win here is dominated by eliminating an identity
multiply-by-one, a stand-in for genuine constant-multiply strength reduction, which is
the next pass in the pipeline.)

### 4b.2 A negative result: strength reduction loses in a bandwidth-bound interpreter

Pass 2 was constant-multiply strength reduction — the classic CPU/ASIC win of lowering
`x·C` to a sum of shifted copies, `Σ ± (x≪k_i)`, via a non-adjacent-form (canonical
signed-digit) decomposition of `C`. RRTL expresses a left shift with no new opcode as
`Trunc_w(Concat([x, 0_k]))` (Concat is MSB-first), and the pass is gated to multi-limb
multiplies by sparse constants — the regime where a multiply is supposedly expensive.

It is bit-exact (verified across power-of-two, odd, dense, and ×(2^64−1) multipliers) and
it **consistently makes the GPU slower**: on a 64-bit datapath, ×4 (one shift) ran at
0.83×, ×5 and ×9 (two terms) at 0.71×, ×640 at 0.49×; the dense multiplier (not reduced)
was unchanged. The cause is the throughline of §4.5–§4b: the interpreter is
**value-traffic bound**. A multiply, however expensive *internally* (a 16-iteration
half-product loop), is a *single* value-buffer write; the shift/concat/truncate/add
expansion is several ops, each round-tripping the value buffer. Trading internal compute
for instruction count is a *loss* when bandwidth, not compute, is the binding constraint.

This negative result is itself the signpost: the productive levers must **reduce value
traffic**, not trade compute for op count. Strength reduction is therefore kept available
but off by the default pipeline.

### 4b.3 Superoperator fusion: removing value traffic wins

Guided by 4b.2 — the productive lever *removes* value traffic — pass 3 is operator
fusion. The interpreter has a fixed opcode menu (it is data, not codegen), so fusion adds
a *curated* fused opcode rather than a design-specific one: `OP_MULADD`, computing
`(a·b + c) mod 2^w` in one step. The encoder folds `Add(Mul(a,b), c)` into a single
`MulAdd` whenever the multiply's result is used exactly once (by that add), so the
product **never round-trips the value buffer** — it stays in a register inside the fused
op. This is bit-exact because `(a·b + c) mod 2^w = ((a·b mod 2^w) + c) mod 2^w`, so the
skipped intermediate truncation is immaterial. It is always-on: it strictly removes one
instruction and its value-buffer write *and* read, so it cannot regress.

Multiply-accumulate is the canonical datapath pattern (FIR taps, MACs, address
generation). On the `prev·C + din` chain benchmark:

| datapath | before fusion | after fusion | speedup |
|---|---|---|---|
| 32-bit (1 limb) | 3.1 Mlc/s | ~4.1 Mlc/s | **~1.3×** |
| 64-bit (2 limb) | 0.8 Mlc/s | 1.5 Mlc/s | **~1.9×** |

This is the **first** optimization in the saga to materially help the 32-bit,
bandwidth-bound case — precisely *because* it removes value traffic. It is the exact
mirror image of the strength-reduction negative result (4b.2): same bottleneck, opposite
lever. Adding ops to trade compute for instruction count *lost* (0.5–0.8×); removing ops
to cut value traffic *won* (1.3–1.9×). Together these two passes make the empirical case
that, for a data-parallel RTL interpreter on unified memory, **value-buffer traffic — not
arithmetic — is the currency to optimize.**

### 4b.4 Slot allocation: footprint is not the currency — access count is

Pass 4 is liveness-based value-slot allocation: a per-block linear scan that reuses value
*slots* across non-overlapping lifetimes (a slot is freed one packet after a value's last
use, so a same-packet effect — which runs after the packet's instructions — still reads
the old occupant safely). It needs no kernel change: the interpreter already indexes the
value buffer through the per-id offset table, so fewer distinct ids simply yields a
smaller table and buffer.

It shrinks the workspace dramatically — **88%** on the mul-add-chain benchmark (1552 → 193
live slots) — and yet, measured in isolation (fusion disabled so the rewrite is valid),
throughput is essentially flat: **1.07× at 32-bit, 0.98× at 64-bit**.

This is the sharpest statement of the thesis. The binding constraint is the **number of
value accesses (reads + writes), not the buffer's footprint**. Slot reuse cuts footprint
but leaves the access count untouched — every instruction still reads its operands and
writes its result — and at 16 384 coalesced lanes the buffer is streamed regardless of
its size, so reuse yields only a sliver of temporal locality. Across the four passes the
pattern is now unambiguous:

| pass | effect on access count | throughput |
|---|---|---|
| constant folding | removes ops (helps where ops are *expensive*) | win on 64-bit only |
| strength reduction | *adds* ops/accesses | loss (0.5–0.8×) |
| **multiply-add fusion** | **removes** an intermediate write+read | **win (1.3–1.9×)** |
| slot allocation | unchanged (footprint only) | flat (~1.0×) |

Slot allocation also does **not compose** with fusion: fusion assumes SSA value ids, while
slot reuse breaks that (the same slot id denotes different values over time, and a fused
multiply's operand live-ranges extend to the consuming add). Wiring it in would cost the
~37% fusion win to gain ~7%, so it stays an opt-in pass.

Its real value is **capacity, not speed**: an 88%-smaller value buffer means ~8× more
lanes — or a ~8× larger design — fit in the same GPU memory at a fixed per-run rate. For
the "huge designs" thesis that is a meaningful enabler even though it does not move the
throughput needle.

### 4b.5 A curated fused-opcode menu

Because the interpreter dispatches a fixed opcode set, fusion grows a *curated* menu. The
3-operand record budget admits the two highest-value additions beyond multiply-add:
`Add3` (`a+b+c`, from `Add(Add(a,b),c)` — adder trees, accumulators, base+index+offset
address calc) and `AndOr` (`(a&b)|c`, from `Or(And(a,b),c)` — AOI gates, the backbone of
control/mux logic). The encoder recognizes them with a small pattern table
(outer `Add` → prefer `MulAdd` else `Add3`; outer `Or` → `AndOr`), gated on the inner op
being used exactly once.

A subtlety surfaced here that is worth a footnote: with `Add3` an addition can be *both* a
fused outer (a `MulAdd`) and a candidate inner of a later `Add3`. Allowing both
double-claims its operands. The invariant — each instruction takes part in at most one
fusion — is restored by never skipping an instruction already chosen as a fused outer.

On an adder/AOI-heavy datapath, the menu fused away **39%** of records and was bit-exact
against the independent CPU engine. Throughput again tracks the *weight* of what is
removed:

| datapath | plain | fused | speedup |
|---|---|---|---|
| 32-bit (1 limb) | 2.58 | 2.66 | 1.03× |
| 64-bit (2 limb) | 0.98 | 1.42 | **1.45×** |

The multi-limb win is solid; the 32-bit win is marginal here because `add`/`and` are cheap
single-limb ops and this design is lighter (more dispatch-bound than bandwidth-bound) than
the mul-add benchmark, where fusion hit 32-bit at 1.3×.

### 4b.6 The gate-level sweet spot, and two-level standard cells

The menu was grown further — `MulSub` (`a*b−c`), `Xor3`, `OrAnd` (OAI), and crucially the
gate-level primitives `NAND`/`NOR`/`XNOR` (`~(a op b)`, from `Not(And/Or/Xor)`). The last
group exposes a sweet spot. A gate-level netlist is built *entirely* from these cells, and
each lowers today to two ops (a bit-op plus a `Not`) with a purely transient intermediate.
Fusing them removes ~half of *all* value traffic, so the win shows up even at 32-bit:

| NAND/NOR/XNOR tree | plain | fused | speedup |
|---|---|---|---|
| 32-bit | 4.36 | 6.59 | **1.51×** |
| 64-bit | 1.71 | 3.15 | **1.84×** |

Gate-level simulation is exactly the regime where RRTL must beat Verilator on huge designs,
so this is a load-bearing result, not an incidental one.

The dominant standard cells, though, are two-level: AOI21 = `~((a&b)|c)` and OAI21 =
`~((a|b)&c)`. These still fit the three-operand record, but require matching a `Not` over an
`Or`-over-`And` (or `And`-over-`Or`). Two subtleties made them work:

1. **Ordering / priority.** Run in one pass, the single-level `Or(And(a,b),c)→ANDOR` fires
   first (program order reaches the inner `Or` before the outer `Not`), claims the `Or`, and
   blocks the deeper cell — leaving `ANDOR + Not` (a 3→2 fold). Splitting fusion into two
   ordered passes — **Not-rooted cells first**, then the one-level arithmetic/logic fusions —
   lets AOI21/OAI21 claim their inners first, restoring the full 3→1 fold.
2. **Single-fusion invariant.** Each instruction may take part in at most one fusion (as
   outer or inner, never both); a two-level cell additionally claims *two* inner ops, both of
   which must be single-use and unclaimed.

The payoff of getting this right is large:

| AOI21/OAI21 tree | one pass (3→2) | two passes (3→1) | speedup (two-pass) |
|---|---|---|---|
| records removed | 29.4% | **58.8%** | — |
| 32-bit throughput | — | — | **1.83×** |
| 64-bit throughput | — | — | **2.05×** |

### 4b.7 Four-operand cells without widening the record

The most common synthesized cells are the 2-2 forms: AOI22 = `~((a&b)|(c&d))` and
OAI22 = `~((a|b)&(c|d))`, four operands each. Rather than widen every instruction record
(which would churn all the stride arithmetic and slightly inflate code fetch for *every*
design), the encoder stores the 3rd and 4th operands in the side `aux` buffer — exactly the
mechanism `Concat` already uses — and keeps the 6-word record. Only these rare ops pay the
extra `aux` indirection. Detection extends the Not-rooted matcher to claim *three* inner ops
(the inner `Or`/`And` and *both* of its `And`/`Or` operands), folding four IR ops to one.

This is the biggest fusion win in the suite:

| AOI22/OAI22 tree | plain | fused | speedup |
|---|---|---|---|
| 32-bit | 4.52 | 9.64 | **2.13×** |
| 64-bit | 1.35 | 3.58 | **2.65×** |

57% of records removed, bit-exact. Because real synthesized netlists are *dominated* by
2-2 cells, this is the load-bearing result for gate-level simulation — the regime where
RRTL most needs to beat Verilator on scale.

The general lesson holds throughout: a fused opcode's payoff is proportional to the value
traffic it removes, so the menu pays most on multi-limb datapaths, on bandwidth-bound
designs with a high fusion fraction, and — most strongly — on gate-level netlists, where
nearly every operation is a fusable cell. The curated menu now spans arithmetic
multiply-accumulate (`MulAdd`/`MulSub`), reduction trees (`Add3`/`Xor3`), and the full
standard gate/cell library (`NAND`/`NOR`/`XNOR`/`AndOr`/`OrAnd`/`AOI21`/`OAI21`/`AOI22`/
`OAI22`) — thirteen fused superoperators, each removing at least one value round-trip, with
up to **2.65×** on gate-level netlists.

### 4b.8 Automatic superoptimization, and why a generic macro-op doesn't beat the curated menu

The curated menu is hand-written; the principled generalization is an **RTL superoptimizer**
(à la Souper for LLVM) — and RTL fits it *better* than LLVM, because combinational logic inside
a cone is pure bit-vector logic with no memory or poison, so equivalence is decidable (exhaustive
truth tables for small cones, SMT-BV in general) and the cost model is concrete (value traffic,
gate count, depth). Two probes sized the opportunity, with opposite verdicts.

*Boolean minimization is subsumed.* Building the ground-truth optimal AND/OR/XOR/NOT form of every
≤3-variable function and harvesting real cones shows 52% of small boolean cones are sub-optimal in
the *raw* IR — but after the existing specializer (§4b) only **1.7%** remain. The const-fold /
algebraic / CSE / DCE passes already capture small-cone superoptimization; a peephole superoptimizer
would find essentially nothing new.

*Value-traffic fusion looked huge — but the generic mechanism doesn't pay.* The other probe counted,
on picorv32, how much the curated menu leaves on the table: the menu fuses **0.7%** of records, while
the theoretical minimum (every single-use intermediate fused) is **−75%**. The reason the menu misses
it is revealing: the fusible intermediates feed **Mux (1582)**, Concat, Slice, Zext — and the menu is
arithmetic/gate-tuned, with no mux or width-op fusion. A *control* core is mux/width-dominated.

So we built the general mechanism: a **generic register-based macro-op**. The encoder greedily groups
maximal single-use simple-op cones into one `OP_MACRO` (the sub-DAG in `aux`); the interpreter
evaluates each in a local register file, writing only the root. It is bit-exact and removes **52% of
picorv32's records** — but it is **throughput-neutral**. On the CPU interpreter (0.8–0.9×) the value
buffer is RAM and so is the register file, so fusion does not reduce work, it regroups it and adds
per-sub-op framing. On the GPU it is worse than the record count predicts (crc32 1.28×, cpu 0.99×) for
two structural reasons: a dynamically-indexed local array **spills to uncoalesced thread-private
memory** (it does *not* stay in registers), and the value buffer it replaces — `values[id*lanes+lane]`
— is **coalesced** global memory, already cheap on the GPU. The record-count metric overcounted its
cost.

The lesson sharpens the whole fusion story: the curated menu wins because each fixed superop is
compiled into the kernel with **static scalar registers**; a generic macro-*interpreter* cannot match
that, because runtime-indexed registers spill. And the genuinely general answer needs no new mechanism
at all — on the **compiled** backends (the Cranelift JIT, the AOT, and the per-design GPU kernel of
§4i), single-use intermediates are register-allocated by the compiler *for free*, exactly as the
single-instance arc found clang subsumes value-level rewrites. So fusion is a property to *delegate to
the codegen backend*, not to re-implement in a data-oblivious interpreter. (Drivers:
`examples/superopt_probe.rs`, `fused_superop_probe.rs`, `toposched_probe.rs`; the macro-op in
`interp.rs::fuse_macros` + `fused_macro_check.rs`.)

## 4c. Instance-level data parallelism

A design with N copies of a module flattens, today, to N× the code. RRTL's data-parallel
model suggests a different treatment: fold N *independent* instances into the **lane
axis** — simulate one module at N·L lanes instead of N modules at L lanes. Identical total
work, but the instances now occupy the GPU grid in parallel rather than as N serial copies
of the instruction stream per thread.

Measured on a 64-instance datapath (fair metric: datapath-megacycles/sec, identical total
work both ways):

| base lanes | naive (N flattened, L lanes) | folded (1 module, N·L lanes) | speedup |
|---|---|---|---|
| 64 | 5.3 | 95.4 | **17.95×** |
| 256 | 20.7 | 186.9 | 9.0× |
| 1024 | 79.0 | 272.5 | 3.45× |
| 4096 | 249.7 | 268.5 | 1.08× |
| 16384 | 248.7 | 208.8 | 0.84× |

Two effects, both significant:

1. **Throughput, regime-dependent.** Below GPU saturation, folding converts idle
   parallelism into useful work — up to **~18×** at 64 lanes. At and above saturation it
   is neutral-to-slightly-negative: the access-count thesis once more — flattened-serial
   and folded-parallel execute the *same* total accesses, so there is nothing to gain when
   the hardware is already full, and 10⁶ effective lanes adds a little dispatch overhead.
   The sweet spot is precisely **a large repetitive design driven by few stimuli** — the
   regime where batch-lane parallelism alone underfills the GPU, which is common for
   "huge design, limited test vectors."
2. **Code size, always.** Folding 64 instances to one shrank the encoded program
   **41.6×** (11 232 → 270 words). Independent of regime, this is a compile/upload and
   capacity win — the same lever as slot allocation (§4b.4), here applied to the
   instruction stream rather than the value workspace.

### 4c.1 The automatic fold

This is now implemented end-to-end. `analyze_instance_fold` detects the foldable case
structurally: the top is a flat wrapper of N≥2 instances of one non-external child module,
with no logic of its own, and the instances are independent — no signal is driven by two
instance outputs, and no instance output feeds another instance's input (shared inputs such
as a common clock are allowed). `InstanceFoldSimulator` then lowers the child *standalone*
(`lower_to_packed_program` accepts any module as root), runs it at `N × base_lanes`, and
exposes per-instance I/O by lane block (`lane = instance·base_lanes + local`). The fold is
verified **bit-exact** against the naively flattened design on the deterministic CPU
reference.

Auto-detected, on a 64-instance datapath:

| base lanes | naive (flattened Top) | folded (auto) | speedup |
|---|---|---|---|
| 64 | 3.6 | 96.9 | **27.0×** |
| 256 | 12.7 | 159.7 | 12.6× |
| 1024 | 50.7 | 252.2 | 5.0× |
| 4096 | 145.9 | 258.0 | 1.77× |

plus a **61.4×** code-size reduction (16 566 → 270 records). The win is even larger than the
hand-folded measurement above, because the naive path now carries the *full* flattened
program. The regime is exactly as predicted: a large win below GPU saturation that tapers as
the base lane count fills the device. The detection is deliberately conservative (clean,
glue-free wrappers only); generalizing to designs with top-level logic between independent
sub-clusters is future work.

## 4d. Range-analysis limb reduction: a measured non-starter

The multi-limb tax (§4.4–4.6) suggests a tempting idea: a value declared 64-bit but
*provably* narrow (high limbs always zero) could be computed in fewer limbs. RRTL has the
analysis — `block_maxbits` is a sound known-zero-high-bit forward pass (Lit exact, `AND`
takes the min of operand bounds, `Mul` the sum, `Zext`/`Trunc` keep the source bound,
`Sext` narrows only when the source sign bit is provably zero, and inputs/registers/`Sub`/
`Not` are full width). It correctly narrows zero-extended values and never narrows genuine
64-bit arithmetic.

The opportunity estimate, however, kills the kernel-level optimization before it is built:

| design | values narrowable | value-words saved |
|---|---|---|
| genuine 64-bit datapath | 0.0% | 0.0% |
| deliberately zext-heavy | 5.5% | 4.7% |

Even a design *built* to be favorable yields under 5% word savings, for three structural
reasons: most values are either genuinely wide or already a single limb (narrow-declared);
**register feedback resets the analysis** (a value read back from a register is full-width
again, so only within-cycle combinational chains narrow); and — per §4b.4 — reducing
*footprint* barely moves throughput, so only the still-smaller subset of reducible
*compute* limbs would help. Against that, the kernel change is invasive (a per-value
effective-limb table consulted on the hot path, with per-operand limb lookups). The
measure-first verdict is a clean **do-not-build**: the sound analysis is kept as a reusable
component (it could feed a CPU-side width minimizer or a cross-cycle register fixpoint),
but the GPU narrowing is not justified. This joins strength reduction (§4b.2, negative) and
slot allocation (§4b.4, neutral) as cases where the access-count thesis correctly predicted,
ahead of implementation, that the optimization would not pay.

## 4e. Capstone: the pipeline stacked, and a measurement-methodology lesson

Stacking the passes on a mixed datapath (AOI gate logic + multiply-accumulate arithmetic +
a constant-foldable config term), measured with **interleaved best-of-N** (all variants
timed back-to-back each rep, so they share one thermal profile):

| variant | records | 32-bit | 64-bit |
|---|---|---|---|
| baseline (oblivious) | 3020 | 1.00× | 1.00× |
| + constant folding | 3014 | 1.01× | 1.02× |
| + fusion | 1804 | 1.63× | 1.44× |
| + full pipeline | 1798 | **1.68×** | **1.46×** |

All variants are bit-identical. Fusion is the robust, dominant lever; constant folding is
neutral on this design (it folds only the one config term) but composes cleanly, so the
full pipeline is best. On constant-heavy designs folding contributes more (§4b.1), and the
two never conflict.

**Methodology lesson (worth a paper footnote).** An earlier *sequential* run of the same
experiment showed constant folding at 0.66× — an apparent 34% regression. It was a pure
artifact: comparing GPU programs one after another lets the device heat up, so a variant
that does the *same* work but runs *second* looks slower. Interleaving the timed runs (and
taking best-of-N per variant) cancels the drift and reveals the true ~1.0×. Large effects
(fusion's 1.6×) survive the confound — at worst they are slightly understated because the
fused variant runs while the device is already warm — but small effects have their *sign*
determined by run order. All throughput ratios in this document for closely-matched
variants should be read as interleaved best-of-N; the fusion and instance-fold wins, being
large, are robust to the methodology either way.

## 4f. Activity-based skipping is anti-correlated with lane count

Activity skipping (ESSENT-style: skip a register's combinational cone on cycles where none of
its fan-in leaves changed) is the biggest *algorithmic* single-instance win in the literature.
RRTL has the substrate — a register-cone (RepCut) partitioner whose groups are combinationally
independent — so a group can be skipped wholesale when its cone leaves (its `Input`/`Reg`
signals) are unchanged. `PartitionedSimulator::tick_activity` implements this (two-phase:
capture-gated on leaves-changed-since-last-capture, then an observability re-settle of groups
that committed or consume a committed group), and is **bit-exact** with the unskipped oracle.

Measured first with a profiler, the skip *potential* is large and regime-split:

| stimulus | per-lane skip | tile (all-lanes-idle) skip |
|---|---|---|
| correlated | 67–78% | 67–78% |
| decorrelated | 67–78% | **0%** |

But realizing it is where the thesis bites. On 64 independent gated cones (93.8% of cones idle
each cycle, bit-exact), the *speedup* depends entirely on lane count:

| lanes | skip rate | speedup |
|---|---|---|
| 1 | 93.8% | **1.15×** (win) |
| 4 | 93.8% | 1.06× |
| 16 | 93.8% | 0.83× (loss) |
| 64 | 93.8% | 0.75× (loss) |

The crossover is ~4–8 lanes. The reason is structural: RRTL's per-group engine is
**SIMD-vectorized**, so the eval is amortized across lanes and *cheap*; the change detection
(reading each cone leaf, per lane, and diffing) is **not** amortized and costs more than the eval
it skips once the lane count is more than a handful. ESSENT wins because it competes against
*scalar* eval; RRTL's eval is far cheaper, so skipping is a net loss exactly where RRTL is
strongest — high lane counts.

This is the sharpest statement of the project's through-line: **for a data-parallel simulator,
optimizations that add per-element bookkeeping lose as parallelism grows.** Activity skipping is a
single-instance/scalar technique fundamentally at odds with the batch model; it belongs on the
single-instance path (e.g. behind a JIT), not the batch GPU/SIMD engine. The mechanism is kept as
a validated opt-in (it wins at very low lane counts; cheaper dirty-flag change detection would
raise the low-lane win but not change the anti-lane scaling).

## 4g. The Cranelift JIT: native single-instance latency

The optimizations so far target *batch throughput* (the GPU/SIMD moat). The complementary
axis is *single-instance latency* — one design, one stimulus, billions of cycles — which is
Verilator's home turf, won by compiling to native code. RRTL's CPU path was interpreted, so
it could not compete there; the `JitCpuSimulator` was a stub.

It is now a real Cranelift JIT (`rrtl-sim-ir::jit`, behind the `jit` feature so the default
build pulls in no codegen dependency). `JitSimulator::compile` lowers the SSA machine IR — the
same `PackedInstr`/`PackedEffect` form the GPU encoder consumes — into a native
`extern "C" fn tick(*mut i64)`: one `i64` of state per signal, SSA values mapped directly to
Cranelift SSA values (no interpreter env array), a `tick` that emits comb-settle → register
capture → commit → comb-settle as straight-line machine code. It covers ≤128-bit signals (a
uniform 16-byte state slot per signal; `I64` for narrow, `I128` for wide, with `imul`'s
`__multi3` libcall resolved by the host), synchronous *and* asynchronous resets, memories
(register files / RAMs, branch-free clamp+select addressing), and the full combinational op
set; it errors (for interpreter fallback) only on >128-bit values.

On a 16-wide, depth-8 32-bit mul-add datapath, single-instance, bit-exact against the
interpreter:

| backend | M-cycles/s | vs JIT |
|---|---|---|
| **JIT (native)** | 1.01 | 1.0× |
| scalar interpreter | 0.05 | **19.6× slower** |
| SIMD engine @ 1 lane | 0.03 | 35.7× slower |

So **~20×** over the best single-instance interpreter (the SIMD-engine-at-1-lane number is a
strawman — that engine is built to amortize across lanes). This rounds out the story: RRTL is
now fast on *both* axes — the GPU/SIMD engine for batch throughput, and the JIT for
single-instance latency. It is also the natural home for activity skipping (§4f), which only
paid at low lane counts — pursued next.

### 4g.1 Activity skipping inside the JIT: the win is gated on cone coarseness

§4f showed activity skipping is anti-correlated with lane count and "belongs on a
single-instance path (e.g. behind a JIT)." We built exactly that and measured it. The activity
JIT (`compile_activity`) emits from the *tree* IR rather than the SSA stream: each combinational
signal and each register becomes a **guarded cone** whose defining expression is recomputed only
when one of its direct fan-in signals changed value since the cone last ran (compared against a
per-cone snapshot). Cones are emitted in topological stream order, so a dirty input is refreshed
before its consumers read it; *lagging* snapshots — updated to the pre-evaluation value only when
a cone runs — make a self-feeding register (a counter) re-evaluate every cycle while genuinely
stable cones skip. It is bit-exact with the oblivious JIT.

The result is more interesting than a flat win or loss. On 64 gated 32-bit accumulators,
single-instance, at **99.9 % skip when idle**:

| cone granularity | IDLE (idle stimulus) | BUSY (all active) |
|---|---|---|
| **fine** (one mul-add per wire) | 0.96× (loss *even at ~100 % skip*) | 0.48× |
| **coarse** (deep inline expr per register) | **1.45× (win)** | 0.68× |

So native-branch event skipping **can** win at single-instance latency — unlike the SIMD path,
which lost as lanes grew — **but only when cones are coarse**: an expensive eval behind a guard
that watches few leaves, so one guard amortizes a lot of skipped work. At the IR's natural
per-wire granularity the guard (load each fan-in leaf and its snapshot, compare, branch) costs as
much as the single mul-add it skips, so it is a wash-to-loss even when nearly everything is
skipped — Cranelift's oblivious straight-line code is simply too tight to beat per-signal. Busy
or decorrelated stimulus always loses (the guard is pure overhead on top of full work). **Cone
coarseness is to the JIT what lane count is to the SIMD engine: the per-element-overhead lever.**
The actionable consequence is that realizing the win on real (fine-grained) IR needs an
ESSENT-style cone-*merging* pass — fuse chains of single-fan-out wires into one guarded cone —
before the guard pays. The mechanism is built, validated, and kept opt-in.

### 4g.2 The vector JIT: native code *and* lane parallelism on one backend

The scalar JIT (§4g) and the SIMD/GPU engines (§4) sit on opposite axes — native
single-instance latency vs. interpreted batch throughput. The vector JIT unions them:
compile the design *once*, but make every value a 4-lane `I32X4` vector, so one native
instruction stream advances 4 independent instances (distinct stimulus) per pass. State is
structure-of-arrays per signal — each signal's 4 lanes are one 16-byte-aligned slot, so a
whole signal is a single aligned vector load — and the codegen is the tree-IR lowering with
vector ops throughout (`imul`/`band`/… on `I32X4`, `icmp`→mask, `bitselect` for mux and
reset). It covers ≤32-bit signals with synchronous reset and **memories** (register files /
RAMs): each memory entry is its own 4-lane slot, and because per-lane addresses differ — and
Cranelift has no portable vector gather — reads and writes drop to a scalar 4-iteration lane
loop (`extractlane` → clamped scalar load/store → `insertlane`), the one part that does *not*
vectorize. Bit-exact against the SIMD interpreter at 4 lanes (datapath and a 4-lane register
file with independent per-lane gather/scatter).

On the depth-8 32-bit mul-add datapath, measured in **lane-cycles/s** (throughput):

| backend | M-lane-cycles/s | vs vector JIT |
|---|---|---|
| **vector JIT** (×4 lanes) | 13.4 | 1.0× |
| scalar JIT (×1 lane) | 2.7 | **5.0× less** |
| SIMD interpreter (×4 lanes) | 0.19 | **72× less** |

So the vector JIT delivers **5.0× the scalar JIT's throughput** — super-linear in lanes
because a 4-lane vector pass is itself slightly *faster* than one scalar tick (vector ops
cost about the same as scalar but do four lanes at once), and **72× the SIMD interpreter**
(it pays no per-op dispatch). It wins on *both* axes simultaneously. The contrast with
activity skipping (§4g.1) is the thesis in miniature: vectorization adds **zero per-element
control overhead** — it is pure width, the data-parallel strength RRTL is built around — so
it wins decisively, whereas activity skipping added a per-cone guard and mostly lost. Lanes
are capped at 4 by 128-bit NEON (`I32X4`); AVX2 (`I32X8`) would double it — but a 128-bit
vector holds *more* than four lanes when the signals are narrow, which §4g.2a exploits.

### 4g.2a Mixed-width lane packing

A 128-bit vector holds sixteen 8-bit lanes, eight 16-bit, or four 32-bit. So the lane count —
and thus the batch throughput — need not be fixed at four: pick the packing per design by its
widest signal (`I8X16` / `I16X8` / `I32X4`). Per-*signal* packing can't work (the instance
count must be uniform across all signals), but per-*design* by max width does, and narrow
designs — the 8/16-bit control logic that pervades real RTL — get 2–4× the lanes for free. The
whole vector backend is generalized from a hardcoded `I32X4` to a `VecCfg{element, lanes, …}`
threaded through codegen; state stays one 16-byte slot per signal. (64-bit lanes are excluded:
aarch64 NEON has no 64-bit vector multiply.) Bit-exact across all three packings — the
register-file test runs on `I8X16`, the 16-bit batch test on `I16X8`, the 32-bit on `I32X4`.

Throughput scales with the lane count (= 128 / element-bits). The cleanest measurement, a
single thermally-consistent run, gives the 16-bit design (8 lanes) at **2.03× the 32-bit**
design (4 lanes) — exactly the lane ratio — and the 8-bit design (16 lanes) at ~4× (3.7–8×,
thermal-noisy on the laptop but consistently far above 32-bit). It composes with the vector
mask-elimination (§4g.5, also narrow-only) and the rayon multicore split.

### 4g.3 Scaling the vector JIT into a CPU batch backend

Four lanes is one vector group; a *batch* of `N` instances is `ceil(N/4)` independent
groups. State is group-major (stride = one group's signals+memory), and the native `tick`
is a nested loop — outer over cycles, inner over groups — where each group runs the same
4-lane kernel rooted at its own base pointer. So one native call advances `N` instances of
distinct stimulus, and the vector JIT becomes a drop-in **CPU batch backend**, the role the
interpreted `SimdCpuSimulator` fills today.

The comparison that matters is native-SIMD vs *interpreted*-SIMD: both vectorize across
lanes with NEON, so any gap is purely the cost of per-op interpreter dispatch. On the
depth-8 mul-add design at **256 lanes** (64 groups):

| backend | M-lane-cycles/s | |
|---|---|---|
| **vector JIT batch** | 20.6 | 1.0× |
| SIMD interpreter | 1.3 | **15.4× less** |

The vector JIT is **15× the interpreter** at full batch width (it was 72× at 4 lanes, where
the interpreter amortizes its dispatch over too few lanes; the gap narrows but never closes,
because straight-line native code pays no dispatch at all). RRTL now has *both* a native
single-instance path (the scalar JIT) and a native batch path (the vector JIT) — the two
axes Verilator and the SIMD interpreter respectively own, now both on compiled code.

### 4g.4 Multicore: the batch stack compounds

The group loop is embarrassingly parallel — groups are independent instances with no shared
state — so the native `tick` takes the group count as a runtime argument and a worker thread
drives any contiguous sub-range of the group-major state. `rayon`'s `par_chunks_mut` splits
the groups (~4 tasks per core); each task calls the same native kernel on its chunk. No
synchronization, and chunk boundaries stay 16-byte aligned so the vector loads do too.

On the depth-8 design at **1024 lanes** (256 groups), 8-core Apple Silicon:

| backend | M-lane-cycles/s | |
|---|---|---|
| **vector JIT (rayon)** | 111.3 | 1.0× |
| vector JIT (serial) | 18.5 | 6.0× less (= multicore gain) |
| SIMD interpreter | 1.2 | **96× less** |

So the three CPU-batch levers **compound**: native codegen (no per-op dispatch) × SIMD lanes
(`I32X4`) × multicore (rayon over independent groups) take the vector JIT to **96× the SIMD
interpreter** at 1024 lanes. The 6.0× from 8 cores is the expected aggregate on a
perf+efficiency-core mix. RRTL's CPU batch path is now native, vectorized, and parallel —
the same shape as the GPU moat, on the CPU.

### 4g.4a Comb-fusion: one combinational pass per cycle

A `tick` is `comb → capture-next-state → commit → comb`: the combinational stream runs **twice**
— once to settle the current state before sampling the registers, and once after the commit so
the new state is observable. But those two passes are on the *same* register state (the trailing
comb of cycle *N* and the leading comb of cycle *N+1* both settle the post-commit registers of
*N*), so across a `tick_many` — where the inputs are fixed for the whole call — the second of each
adjacent pair is redundant. The fused `tick_many` evaluates comb **once up front**, then per cycle
does `capture; commit; comb`, where that single comb serves as both the next cycle's leading settle
and, on the last iteration, the final observable settle. It is bit-identical to looping `tick()`
(validated by an explicit naive-`tick()`-loop oracle and the `n = 0,1,2,3,17,64,257` boundary
cases), at roughly **half the comb work** — and comb is the bulk of an RTL tick.

The payoff scales with the comb fraction of the design (SIMD CPU batch engine, 256 lanes,
interleaved best-of-3, bit-exact per lane):

| design | naive `tick()` loop | fused `tick_many` | speedup |
|---|---|---|---|
| comb-heavy (depth-48 pipeline) | 1.2 | 2.3 M-lane-cycles/s | **1.99×** |
| shallower (depth-8) | 5.4 | 9.4 M-lane-cycles/s | **1.74×** |

It approaches 2× on comb-dominated designs — exactly the gate-level/large-design regime that is
the batch moat's target — and **multiplies every batch-throughput number** above, for free. The
same fusion is applied to both CPU batch engines (`SimdCpuSimulator`, `PackedSimulator`); the GPU
interpreter already uses it. (Driver `examples/comb_fusion_check.rs`.)

### 4g.4b Bit-parallel gate-level batch: one word = 64 lanes

The vector backends pack lanes into SIMD *elements* — but a **gate-level** design (every signal
1 bit, every op a boolean gate) wastes 7/8 of an 8-bit element on a 1-bit signal. The classic
logic-simulation answer is **bit-parallelism**: store one signal's state across `L` lanes as
`ceil(L/64)` 64-bit words — *one bit per lane* — and evaluate each gate (`&`/`|`/`^`/`~`, and a
mux as `(c&t)|(~c&e)`) as a plain bitwise op, so **one word op advances 64 lanes** at 100 %
density (vs the I8X16 path's 16 lanes at 12.5 %). A sizing probe confirms the regime: a pure
gate-level netlist is **100 % 1-bit signals and 100 % bitwise ops** (no arithmetic, no width ops).

We built three engines, and the measure-first sequence is the lesson. An **interpreter**
(`BitParallelSimulator`) validates the representation and is bit-exact, hitting ~60 M-lane-cyc/s
= **~36× the SIMD-CPU interpreter** — but it *loses ~10×* to the compiled vector JIT: interpreter
dispatch outweighs the density gain. A **Cranelift JIT** (`BitParallelJitSimulator`, 64 lanes per
scalar `i64` op) recovers that, but only *ties* the vector JIT — 64 scalar-64-bit lanes/op trade
evenly against 16 NEON-128-bit lanes/op. The win comes from the **AOT**: emit `uint64_t` bitwise C
with an inner group loop and let **clang -O3 auto-vectorize** it (NEON `i64x2` = 128 lanes/op,
which the hand-written Cranelift did not do). The bit-parallel AOT runs **~1600 M-lane-cyc/s —
≈3.4× the vector JIT** (the prior best RRTL batch engine), bit-exact, and ~3× the scalar
bit-parallel JIT. So gate-level netlists — the Verilator-competition scale regime — get a new
batch engine ~3.4× faster than anything else here, and the right backend for it is the
auto-vectorizing C compiler, not hand-scalar codegen. Hand-vectorizing the Cranelift JIT to
`I64X2` (128 lanes/op) doubles it to ~1.7× the vector JIT, but still trails the AOT ~1.7× —
Cranelift vectorizes but does not match clang's loop-unrolling/scheduling.

**Against Verilator.** On the *identical* gate netlist, Verilator 5.030 runs **19.7 M-cycles/s**
single-instance. Measured in the same unit (instance-cycles/s), the bit-parallel AOT does **~1900
M — ≈96× a single Verilator instance, on one core**. Verilator has no batch axis (N instances = N
processes), so even against its 8-process batch (~158 M) the single-core bit-parallel AOT is ~12×
ahead. And the groups are fully independent (no shared state, no per-cycle sync), so a rayon
**multicore** `tick_many_parallel` splits the group-major state across cores for near-linear
scaling: **5.0× on an 8-core M3** (bit-exact), i.e. **~186× a single Verilator instance** (and ~23×
its 8-process batch) on gate-level batch. The honest boundary is the same as everywhere in this
work: this is the *batch* regime (independent instances — fuzzing, Monte-Carlo, fault simulation);
for *single-instance latency* Verilator wins, because using a 128-lane engine for one instance
wastes 127/128 of it. (Drivers: `examples/bitparallel_probe.rs` sizing, `bitparallel_bench.rs` the
5-way throughput + multicore + bit-exactness comparison; `bench/sv/gate/` the Verilator harness.)

### 4g.5 Lazy width-masking: clean/dirty tracking

The single largest instruction class in an RTL simulator is the width-mask — the `& ((1<<w)-1)`
that discards a value's upper bits. The naive JIT masks after *every* operation. But most of
those masks are dead: for `add`/`sub`/`and`/`or`/`xor`/`not` the low `w` result bits are correct
regardless of garbage above `w`, so a mask is only needed before a consumer that actually reads
the upper bits. We track a `clean` bit per SSA value (upper bits known zero) and emit a mask only
at **strict consumers** — equality/unsigned compare, zero-extend, and memory *addresses* — and at
**stores** (which keep state, hence `Signal` loads, clean). The correctness crux is width-relative:
a dirty operand only corrupts an op whose result is *wider* than the operand (its garbage then
lands inside the result range), so same-width arithmetic — the common case — needs no operand mask;
and signed-compare / sign-extend / slice read only the low `w` bits (the shift discards the
garbage), so they tolerate dirty inputs too. This is Verilator's clean/dirty idea, applied in the
Cranelift backend, and it is bit-exact (the full JIT test suite and the Verilator cross-check pass).

A controlled A/B (a compile-time toggle, both modes built and run back-to-back):

| design | mask after every op | lazy masking | speedup |
|---|---|---|---|
| W=16, D=8 | 4.1 | 11.2 | ~2.8× |
| W=64, D=16 | 0.30 | 0.98 | ~3.3× |

(M-cycles/s; absolute values are thermally noisy on the test laptop, but lazy masking's *worst*
run beat eager masking's *best* with no overlap.) The win exceeds the raw instruction-count
reduction because the per-op masks sat **on the critical path** of the serial multiply chains
(`mul → mask → add → mask → …`): removing them both deletes instructions and shortens the
dependency chain so Cranelift can pipeline the independent channels. ~2–3× on multiply-heavy
datapaths, for an analysis that is cheap and entirely at compile time.

The same pass applies to the **vector** backend, with one twist: every vector value lives in a
32-bit `I32X4` lane, so any signal ≥ 32 bits is *automatically* clean (there are no bits above
the lane) and 32-bit designs are already mask-free. The optimization therefore helps only the
**sub-32-bit** signals that pervade real control logic. On an 8-bit batch design it is ~1.5× (a
clean serial A/B: lazy 16.6–20.5 vs eager 11.9–13.9 M-lane-cycles/s, no overlap), smaller than
the scalar win because a vector `band(splat)` mask is cheap and amortized across four lanes; a
32-bit design is unchanged, as expected. Bit-exact against the SIMD interpreter.

### 4g.5a Priority mux-chain rebalancing: a JIT-only lever, and what it proves

The lazy-masking win above came partly from *shortening dependency chains* so Cranelift could
fill its issue width. That raises a question: are there other serial chains worth shortening? A
critical-path analysis of picorv32 says yes — a **48-deep combinational mux chain** that is 96 % of
the comb critical path. It is a Verilog priority `if/else`/`case` lowered to a right-nested
`c₀?t₀:(c₁?t₁:(…:base))`: every condition is a 1-bit signal read (ready immediately) and every value
is computed off to the side, yet the muxes resolve **serially** because each one's `else` is the
next. That is the latency-bound shape a wide out-of-order core cannot hide.

Priority-select is **associative** — `combine((h₁,v₁),(h₂,v₂)) = (h₁∨h₂, h₁?v₁:v₂)` with the left
operand higher priority — so the chain can be rebuilt as a **balanced tree**: the same mux count,
plus one OR per node for the subtree "hit" bit, depth `N → ⌈log₂ N⌉` (48 → 7 on picorv32). It is
bit-identical (exact priority preserved, no one-hot assumption), and the C compiler cannot do it
itself — the ternary chain encodes a serial dependency it has no licence to reorder. The pass
(`specialize::rebalance_mux_chains_program`) is validated bit-exact: a unit test over 400 random
cycles, and the full picorv32 mem-bus transaction trace.

The result is the sharpest illustration of this paper's recurring lesson:

| backend | plain → rebalanced (picorv32) | |
|---|---|---|
| **Cranelift JIT** | 2.39 → 2.70 M-cycles/s | **1.13× (win)** |
| clang `-O3` AOT | 3.15 → 2.83 M-cycles/s | **0.90× (loss)** |

The *same* transformation helps one backend and hurts the other. **clang already schedules for
ILP** — it extracts the parallelism from the deep chain on its own, so the explicit rebalance is
redundant and the +584 OR instructions are pure overhead (0.90×). **Cranelift, the fast-compile
JIT, has weaker scheduling** — so doing the depth reduction in the IR genuinely raises its IPC
(1.13×). Mux-rebalancing is therefore a *JIT-backend* lever (the low-latency, no-compile-wait
path), not an AOT one, and a clean confirmation that the residual single-instance levers live in
the *weaker* compiler, not in clang (cf. §4h.1). It is a standalone, opt-in pass for exactly that
reason — applied to the JIT, skipped for the AOT. (Driver `examples/rebalance_check.rs`.)

### 4g.5b Control-flow recovery: eval-taken, the last structural control-heavy lever

The deepest reason a control-heavy core (picorv32) trails Verilator single-instance is the
**mux-eval-all tax**: the divergence-free dataflow IR computes *every* arm of every mux each cycle
(45% of picorv32's ops are muxes), while Verilator's control-flow `if` computes only the taken
branch. clang cannot fix this — a flattened ternary `c ? f(x) : g(x)` over pure sub-expressions has
no branch to preserve, so both sides are computed. The fix is to *re-introduce* the control flow in
codegen: sink each single-use mux-arm cone into a real `if/else` so the untaken arm's work is
skipped. The AOT (emitting C) is the natural home — `if (cond) { <then-cone>; dst = then; } else
{ ... }`, recursively for nested priority chains. It is **latency-only** (it trades the IR's
divergence-freedom, the property that lets the batch engines run lane-parallel), so it is opt-in
(`AOT_CFLOW`), restricted to the mux-heavy `comb`/`tick_next` blocks, and applied only to arms whose
root is real compute (branching a trivial arm only adds a mispredictable test).

It is bit-exact (`aot_check` and the picorv32 mem-bus trace) and a measured **~1.08–1.14×** on
picorv32 (AOT, 4.99→5.40 and 6.25→7.12 in two A/Bs). That is below the ~1.2× the static op-count
reduction predicts — the gap is **branch mispredicts** on picorv32's data-dependent control muxes,
which eat part of the saved work, exactly the risk control-flow recovery carries. It is the last
clean structural lever for control-heavy latency: real, clang-can't-do-it, but bounded. Stacking the
control-heavy levers (testbench fusion 1.7× → rebalance/control-flow ~1.1× each) reaches roughly the
honest floor — ~1.4–1.5× behind Verilator — with the batch IR intact. (Driver `aot_settle_obs.rs`
under `AOT_CFLOW=1`.)

### 4g.5c Decoder tabulation: a ROM for the decode cone, the "table-izing" lever

§4h.1 attributes part of Verilator's residual edge to RTL-level passes clang cannot reconstruct from
flattened C, naming **table-izing decoders**. We built it. picorv32's per-instruction decode-flag
registers lower to a one-level *hold-mux*, `flag <= trigger ? decode(instr_bits) : flag`, where the
`decode` branch of a group of flags is a pure function of a single instruction register
(`mem_rdata_latched`). `tabulate_decode_program` finds that group, evaluates the decode cone over all
`2^key` keys at build time, and packs one field per decoded value into a ROM row — replacing the deep
decode cone with one `MemRead` plus a `Slice` per value (the hold-mux `cond`/`else` are untouched, so
it is correct by construction), then DCEs the dead cone. clang can't do this: *which* function the
decode computes, and that it depends on only those bits, is a property of the RTL graph, not the
flattened branches. It is **latency-only** (a ROM read is a data-dependent gather, which loses on the
lane-parallel batch/GPU backends) and targets the JIT, whose weaker scheduler benefits most from
removing work.

A subtlety the build surfaced: decode values are *not* all one bit — `decoded_rd` is the 5-bit `rd`
field — so the ROM row packs each value at its own bit offset/width (an early one-bit-per-value
version decoded correctly *except* for the multi-bit fields, diverging at the first branch). On
picorv32 it tabulates 12 decode values (keyed on a 23-bit instruction slice), removing **1448 of 3215
machine instructions (45 %)**, and is a **~1.12–1.13×** bit-exact win on the Cranelift JIT (validated
against the plain JIT over a 2 M-cycle mem-bus trace). Two honest caveats. First, the 45 % instruction
cut yields only ~1.12× because the tabulated flags live in the `tick_next` stream, not the comb
critical path — the win is pure op-count reduction, and (as throughout) the JIT is not purely
instruction-count-bound. Second, picorv32's decode key is **irreducibly 23 bits** (every key bit is
shared by ≥2 flags, so no flag can be dropped to shrink it) → a 2^23-entry / 128 MB ROM. That is only
viable because the ROM *address is the instruction encoding*: a real program executes only tens of
*distinct* instruction words, so the runtime read working-set is a handful of cache lines and stays
L1-hot despite the 128 MB nominal size (the same sparse-access argument as §4k). Build cost is a
one-time ~3.7 s ROM construction. Like mux-rebalancing (§4g.5a), it is a JIT lever; the AOT, whose
clang backend already schedules the decode well, is expected to be neutral. (Pass `tabulate.rs` with
unit test `tabulated_decode_matches_plain_jit`; driver `picorv32_tabulate.rs`, `key_max` arg.)

### 4g.6 Dynamic (profile-guided) specialization: re-JIT on a runtime quasi-constant

The static specializer (§4b) can only fold what it can *prove* constant. But real SoCs are full
of **configuration / mode registers** — set once at boot from an input and then held for billions
of cycles — that static analysis must treat as variable because, in general, they *are*. A long
run is exactly where a profiler can do better: observe that such a register never changes, freeze
it to its value, and **re-JIT** a program in which const-folding collapses the control logic and
DCEs the datapaths that configuration used to gate. This is a tracing-JIT's *speculate-guard-
deoptimize* loop, applied to RTL.

The bridge is `freeze_signals_program` (in `specialize.rs`): rewrite every read of a profiled-
stable signal to its literal, then run the ordinary static specializer. Crucially the **output is
still data-oblivious** — the dynamism is only in *which* values we treat as constant; there is no
runtime branch on data, so a frozen program composes with the lane engines exactly like any other
specializer output (no SIMT divergence, unlike activity-skipping, §4f). The driver
(`dyn_specialize.rs`) runs four phases on the Cranelift JIT:

1. **Profile.** Run the generic JIT for a warm-up window, snapshotting every register; the ones
   that never move are freeze candidates (the accumulator moves every cycle and is correctly left
   alone).
2. **Specialize + re-JIT.** Freeze the stable registers, specialize, compile the result, and
   transfer live state across (signal slots are stable under specialization, so it's a copy).
3. **Guarded execution.** Run the specialized JIT. Each tick a guard checks the frozen registers
   still hold — and the protocol is *correct by construction*: a frozen register can only change at
   its own commit, so the tick that changes it still read the right (old) value during settle; the
   guard catches the change and switches **before** the next tick.
4. **Deoptimize.** On a guard violation, transfer state back to the generic JIT and re-profile —
   which re-specializes to the *new* configuration if it restabilizes.

On a 64-bit configuration-gated DSP — a 2-bit `mode` selecting one of four multiply-heavy arms into
an accumulator — freezing `mode` folds the select and DCEs three arms (48 → 21 machine instructions,
the bulk of them multiplies). Measured on the single-instance JIT, bit-exact against a generic
reference (best-of-5, interleaved):

| | generic | specialized | speedup |
|---|---|---|---|
| raw tick (`tick_many`) | 202 M-cyc/s | 401 M-cyc/s | **1.99×** |
| steady state + per-tick guard | 70 M-cyc/s | 115 M-cyc/s | **1.64×** |

The raw row is the ceiling (pure compute); the steady-state row is the realistic number with the
guard's per-tick register read included — the guard costs ~18% of the win, the honest price of
deopt-safety. A reconfiguration mid-run (`mode 1 → 2`) is caught by the guard, deoptimizes, and
re-specializes to the new mode, staying bit-exact across the transition. As the static-specializer
results predict (§4b.1), the payoff scales with the *cost* of what's eliminated: on a small
cheap-op design the guard + dispatch overhead can erase the win (an early 32-bit version measured
net-neutral) — dynamic specialization pays precisely when configuration gates *expensive* logic,
which in real designs it routinely does.

### 4g.7 A tiered JIT: lightweight hot-spot detection and speculation beyond constants

§4g.6 freezes a register that is *provably* stable over a window. A tiered JIT generalizes this in
the shape of a tracing JIT (V8/JVM): a cheap Tier 0 that profiles, and a specialized Tier 1
promoted on a *hot* spot — but with two RTL-specific extensions.

**Lightweight sampled profiling.** Snapshotting every register every cycle (§4g.6) is not cheap.
Instead the profiler (i) watches only the *narrow* (≤16-bit) registers — the control/state bits
whose value gates logic, never the wide datapaths — which bounds its histogram cost and memory, and
(ii) *samples* every `S` cycles rather than every cycle. The sample period must be **coprime to the
design's activity periods**: an early version used `S = 16` against a control register that excurses
every 128 cycles, and 16 | 128 meant every sample landed on an excursion, over-counting the rare
value 8× (12.5% vs the true 0.8%) and defeating the profiler. A prime `S` (13) fixes it. (Classic
sampling aliasing — worth stating because it silently corrupts the profile rather than erroring.)

**Speculation beyond exact constants.** Tier 1 speculates a register's *dominant* value when its
sampled frequency exceeds a bias threshold — not necessarily 100%. This is strictly more than
freezing: a mode register that *blips* off its dominant value occasionally is **never** 100%-stable
in any window, so the freeze policy can never find a clean discovery window and never promotes, yet
it is still ~97–99% dominant, so bias speculation promotes it and runs the fast tier the rest of the
time. The guard and deopt are identical to §4g.6 (a register only moves at its own commit, so the
tick that breaks the speculation was still correct); the only change is the policy that *decides*
what to speculate. Specialized engines are **cached keyed by the speculated values**, so when the
dominant value recurs after a blip, re-promotion is a state-transfer engine swap, not a recompile —
in a 4M-cycle run with 62 k excursions, **one** compile served all 62 k promotions. The hot tier
runs as a **tight burst loop** (tick + guard, nothing else) until the guard trips; a per-cycle
phase-`match`/closure dispatch costs as much as the cheap tick it guards, so amortizing it is what
makes a net gain possible at all.

On `cfgdsp` (the §4g.6 mode-gated DSP) with a 2-cycle excursion every `period` cycles, single-instance
JIT, best-of-3 interleaved, all bit-exact. The **fast-tier fraction** is the stable, reproducible metric
(it measures whether the policy *promotes*); throughput hovers near 1× and is noisy at this tick size:

| excursion period | bias-0.90 fast % | freeze-1.0 fast % |
|---|---|---|
| every 4096 cyc | **99.9%** | 99.9% |
| every 64 cyc | **93.7%** | **0.0%** (never promotes) |

The frequent-excursion row is the point: bias keeps a 97%-dominant register in the fast tier 94% of the
time where the exact-constant policy **gives up entirely** — its discovery window never sees a clean
100%-stable stretch — so speculation captures a fast-tier residency that freezing structurally cannot
reach. Throughput itself is ≈1× and noisy here (per-cycle input/output marshalling, ~15 ns, dwarfs the
~5 ns JIT tick; and Cranelift pipelines independent multiplies, so even "heavy" arms are nearly free —
the eliminable work has to be *serial* to matter). Two honest consequences: in the frequent-blip case
bias's deopt/transfer traffic makes it slightly *slower* than the pure-generic freeze run; and the wall-
clock benefit only emerges, as in §4g.6, when per-tick compute dominates the fixed per-cycle cost. The
*mechanism* — lightweight sampled profiling, dominant-value speculation, cached re-promotion, guarded
deopt — is the contribution; it is what scales to the compute-dominated regime.

### 4g.8 The payoff on a real core: instruction-subset specialization of picorv32

§4g.6–4g.7 are bounded by the cfgdsp microbenchmark's cheap tick. A real CPU core is the compute-
dominated regime they point to, and it carries a perfect runtime quasi-constant: **a workload uses only
a subset of the ISA**. The YosysHQ **picorv32** decodes each instruction into ~65 latched decode-flag
registers (`instr_sll`, `instr_mul`, `instr_slt`, `instr_lw`, the `is_*` class helpers, …); for the
instruction classes a given program never executes, those flags are **0 for the entire run**. Static
analysis cannot know which — it depends on the bytes in RAM, not the RTL — but a profiler can simply
watch them.

We JIT picorv32 (driving its native memory bus with a 1-cycle-latency host RAM), run a workload — a
nested add/branch loop that touches only `addi/add/blt/sw/j` — and profile the decode flags. 50 of the
65 are never set. Freezing those 50 to 0 and re-JITting lets const-folding + DCE delete the logic they
gate: the barrel shifter, the comparators, the load/store byte-lane datapaths, the logic-op ALU arms,
the CSR/counter and IRQ blocks. Measured (300 k cycles, best-of-3, single-instance JIT):

| | machine instrs | throughput | |
|---|---|---|---|
| generic picorv32 | 3221 | 2.3 M-cyc/s | |
| specialized (50 flags frozen) | **1479 (−54%)** | **3.7–4.1 M-cyc/s** | **1.6–1.7×** |

The specialized core is **bit-exact** — the full memory-bus transaction trace (every address, write
datum, and strobe over 300 k cycles) is identical to the generic core's, and it computes the same
result. Two things are worth noting. First, **over half the design is dead for this workload** and the
specializer removes it — the eliminated logic is real per-tick work on a core whose tick is ~270 ns, so
unlike cfgdsp it converts directly to wall-clock (the specialized JIT reaches the same throughput as the
*AOT* backend, §4b, by simply having half as much to run). Second, this is honest only because the
workload is narrow: a program exercising the full ISA would freeze nothing. But narrow is the common
case — embedded firmware, a regression test pinned to one extension, a kernel that never uses
floating-point — and the simulator discovers the subset automatically, per run, with a profile and a
guard. This is the dynamic-specialization thesis (§4g.6–4g.7) realized on a real design: *find the
runtime invariant a workload exposes, specialize the oblivious program to it, verify.*

### 4g.9 Partitioned compilation: parallel and incremental, defeating the compile-wall

The JIT inherits Verilator's worst structural property if used naïvely: a huge design lowers to **one
enormous function**, and Cranelift's per-function cost (register allocation especially) is *super-linear*
in function size — the C++ compile-wall, relocated. The fix is the partitioner from §5: slice the design
into independent register-cone partitions and compile each as a **separate** function. Three wins fall
out, none requiring any change to the codegen itself, only to its granularity.

We lower a large memory-free design — `K` independent pipelined PEs (deep multiply chains), partitioned
one-cone-per-PE — to the JIT both monolithically and partitioned, and compile the partitions in parallel
across cores (rayon). Measured:

| design | machine instrs | monolithic compile | partitioned (parallel) | incremental (1 partition) |
|---|---|---|---|---|
| K=32 | 1129 | 25 ms | 17 ms (1.5×) | 1.0 ms (25×) |
| K=64 | 2249 | 54 ms | 13 ms (**4.0×**) | 0.7 ms (80×) |
| K=128 | 4489 | 205 ms | 17 ms (**12×**) | 0.7 ms (**297×**) |

Three things to read off this. **(1) The monolithic column is super-linear** — doubling the design
(2249 → 4489 instrs) nearly *quadruples* compile time (54 → 205 ms). That is the wall, reproduced.
**(2) The partitioned-parallel column is roughly flat** (~13–17 ms) — bounded per-function cost times
a fixed core count — so the speedup *grows* with design size (1.5× → 4× → 12×) and would keep growing.
Even ignoring parallelism, the serial *sum* of the small compiles beats the one big compile, purely from
the super-linearity (e.g. K=32: 18 ms summed vs 25 ms monolithic). **(3) Incremental recompile is
~constant** (~0.7 ms — one partition, ~1/K of the work) and so gets *cheaper relative to the whole* as
the design grows: 297× at K=128. Edit one module, recompile one cone, not the world — the iteration
story (§7) that an AOT-codegen simulator structurally cannot match.

The partitions then run with **register-stable boundary exchange** — because the slicer replicates shared
combinational logic into each cone (RepCut-style), every cross-partition signal is a register or a stable
input, so each partition is self-contained: snapshot-exchange the boundary registers, tick all partitions
(independently — this is also the multi-core *execution* path, §5), done. The partitioned JIT is
**bit-exact** against the monolithic JIT across the run. The one design constraint surfaced while building
it: a combinational signal at a partition boundary (an instance port driven by `a ^ b`, or a
combinational top-level output) is *not* register-stable and needs the topological boundary schedule the
interpreter's `PartitionedSimulator` already implements; keeping boundaries registered (the common case
after cone replication) keeps the exchange a single snapshot.

### 4g.10 Activity-driven skipping at partition granularity

The defining empirical fact of huge-RTL simulation is that **most of the design is idle every cycle**; an
oblivious simulator wastes the majority of its work re-evaluating unchanged logic (ESSENT/Khronos exploit
exactly this). §4g.1 found activity skipping inside the JIT "gated on cone coarseness" — fine-grained
guards don't pay. The partitions of §4g.9 are the *right* coarse granularity, so we skip a whole partition's
tick when it has reached a fixpoint and its drivers are stable.

The mechanism took three tries; the first two are the instructive part. **(a)** Skipping by inspecting the
partition's own register state each cycle (compare to last) is *wrong* — a counter looks "unchanged" right
before it increments — and also *slow*: reading a partition's registers via `get_signal` (~20 ns each) costs
far more than re-evaluating its ~20-multiply pipeline on native code (~10 ns). **(b)** The JIT's in-engine
`compile_activity` cone-guards (§4g.1) don't fire on these fine-grained pipeline cones (it reports ~0%
skipped). The lesson from both: the activity test must be *cheaper than the tick it guards*, and must reason
about fixpoints, not raw state.

**(c)** What works: skip a partition once its **external drivers** — captured *for free from the boundary
exchange we already do*, no extra reads — have been stable longer than its **settling bound** (its register
count, a safe upper bound on how long a feed-forward cone takes to reach a fixpoint). The one subtlety is
that a *self-driving* partition (a counter, whose next state reads its own register) never reaches a
fixpoint however stable its inputs; so skip-eligibility is verified **once** per partition with a single
pre/post register check the first time it becomes a candidate, then cached. The whole steady-state test is
then a per-partition counter plus an in-hand vector compare — no `get_signal` of state at all.

**Redefining the boundary access mechanics.** This only became measurable once the *exchange itself* was
fixed. The partitioned runner first moved boundary signals between engines with `get_signal`/`set_signal`
(~20 ns each: bounds checks, width masking, a settle check, method dispatch) — which is absurd for what is a
16-byte memory copy of a register slot whose value is already correctly masked. Exposing the JIT's raw `i64`
state buffer (`state_words`/`state_words_mut`; signal `i` at words `[2i, 2i+2)`) and doing the exchange as a
two-phase raw word copy (gather then scatter, so producer reads never alias consumer writes) cut the per-
boundary cost ~10× and made the runner **5–9× faster overall** — the no-skip baseline went from ~0.18–0.48
to ~0.4–3.5 M-cyc/s. The exchange was the floor; it no longer is.

On a design with a geometric activity gradient (a counter driving per-PE inputs that change every 2^i
cycles, so low-index PEs are hot and high-index PEs almost always idle), single-instance JIT, raw exchange,
bit-exact against the monolithic core:

| partitions | % skipped | speedup from skipping |
|---|---|---|
| 9 | 20% | 0.8× (loses) |
| 17 | 57% | 1.6× |
| 25 | 70% | 2.1–2.5× |
| 33 | 78% | 2.9× |

The speedup tracks the skip rate, which grows with design size as more of the design falls idle — the
huge-design regime. Note the crossover: with the now-fast exchange, skipping only **20%** of a cheap-exchange
baseline *loses* (the per-cycle activity bookkeeping exceeds the saved ticks), while skipping 57%+ wins
clearly. This is the honest shape — activity skipping pays above a skip-rate threshold, and fixing the
exchange *raised* that threshold by making the thing it competes against cheaper. It works at all for the two
reasons the failed attempts isolate: the check is cheap (no state reads in steady state) and the cones are
coarse and heavy (a skipped partition's deep multiply pipeline is real saved work). On the *batch* lane
engine activity skipping is anti-correlated with lane count (§4f); on the single-instance JIT at partition
granularity it pays, and composes with the partitioned compile (§4g.9). The remaining structural cost is that
the exchange still *copies* (producer slot → consumer slot); a true shared-memory layout where a producer
register and its consumer boundary are the *same* slot would be zero-copy, but needs the partition functions
compiled against one global state buffer — which §4g.11 now does.

### 4g.11 Zero-copy shared-state partitioned JIT

The previous section's exchange, even at raw-memory speed, still *copies* every boundary register slot each
cycle. The structural fix is to stop having separate per-partition state at all: compile every partition over
**one global state layout** and let them share a single pair of buffers, so a partition reads every other
partition's registers *directly from shared memory* — no copy, no exchange.

The obstacle is the synchronous-update ordering: with one shared buffer, if partition A commits its registers
before B reads them, B sees this-cycle values instead of last-cycle (an off-by-one the copy sidestepped by
snapshotting). The fix is **double-buffered registers** in the JIT itself: a new `tick_db(cur, nxt)` cycle body
where every *register* read comes from `cur` and comb/inputs and all writes go to `nxt`; the runner swaps
`cur`/`nxt` each cycle. Now every partition reads all registers (its own and its boundaries) from the read-only
`cur` and commits only to `nxt`, so the order partitions run in does not matter — the result is identical and
the partitions are **embarrassingly parallel** (each writes a disjoint set of register slots). The codegen
change is one new pointer threaded to the Signal-load sites plus a register mask (`read_ptr` picks `cur` for
registers, `nxt` otherwise); with `cur == nxt` it is exactly the existing single-buffer tick, so all 110 JIT
tests pass unchanged, and `tick_db` is separately verified bit-exact against the single-buffer tick.

The partitioned runner needs no IR remapping: the lowered program is *already* global-indexed, so each
partition's program is the full program with only the ops it owns retained (filtered by the RepCut register-cone
mask), compiled with `compile_db(machine, global_is_reg)` — the *global* register mask so that boundary
registers a partition reads (owned by others) also route to `cur`. Per cycle: write inputs into `nxt`, call each
partition's `tick_db(&cur, &mut nxt)`, swap. **No boundary exchange exists.** On a flat 64-stage pipeline split
8 ways it is bit-exact against the monolithic JIT and runs at **~16 M-cyc/s — matching the single-engine
monolithic throughput** (the raw-exchange runner on a comparable design was several× slower; the copy is simply
gone).

Two correctness lessons cost real debugging time and are worth recording. **(1)** The boundary must be a *true*
register. The slicer happily makes an instance's `output reg` port the boundary, but that port is a
*combinational alias* of the register (`is_reg` is false for it), so a double-buffered read of it is one cycle
stale — and you observe a uniform +1-cycle lag. Partitioning a *flat* design by register cones keeps every
boundary a genuine flip-flop. **(2)** For the same reason, *observe* internal registers, not `output reg`
ports: the port alias needs a post-commit settle to be current, which `tick_db` deliberately omits (the values
that matter are registers, committed into `nxt`). With both heeded, the shared-state partitioned JIT is the
zero-copy, parallel-ready execution substrate the partitioned compile (§4g.9) was built toward.

### 4g.12 Multi-core partition execution, and the per-cycle barrier wall

The zero-copy substrate (§4g.11) makes every partition's tick **order-independent** — all reads come from
the read-only `cur`, all writes go to disjoint register slots of `nxt`. That is exactly the property a
multi-core *execution* engine needs: a whole cycle's partitions can run on separate cores, syncing only at
the per-cycle buffer swap. We build it with one persistent thread per partition (= #cores, so the run
forks/joins **once**, not per cycle), a barrier each cycle, and thread 0 doubling as the coordinator
(swap + output + next-cycle inputs between the two barriers while the others wait). It is **bit-exact**
against the monolithic JIT — the parallel execution is correct.

It is also, on this hardware, **slower** — and the reason is the honest, important part. Per-cycle
synchronization is a hard floor: two barriers per simulated cycle, across the M3's heterogeneous P+E cores
(the slowest efficiency core gates every barrier), cost **~30 µs/cycle**, so the parallel engine is pinned
near **~30 kcycle/s regardless of how much work each partition does**. Serial, by contrast, runs as fast as
the work allows (multi-Mcycle/s on small designs). So the measured speedup *grows* with design size as serial
slows toward the barrier rate — 0.01× → 0.18× across 193→3073 registers here, and ~0.67× at ~4600 registers
in a longer run — and only **crosses 1× on a design large enough that serial itself drops below ~30 kcycle/s
(~16 k+ registers)**. Three barrier implementations bracket the cost: rayon `par_iter` re-forks every cycle
(worst, ~hundreds of µs); a `std::sync::Barrier` (futex) is the ~30 µs floor above; a sense-reversing **spin**
barrier would be ~100 ns but *deadlocks under oversubscription* — with 8 busy spinners on 8 P+E cores plus
system threads, a descheduled spinner stalls everyone, so it never completes.

This is the canonical **per-cycle RTL-parallelism wall**: a cycle is too little work to amortize a cross-core
barrier unless the design is enormous. Verilator's `--threads` hits the same wall and helps only large
designs; the research that gets past it (Manticore, ASPLOS'23; Parendi, 2024) does so with **bulk-synchronous**
scheduling — many cycles or a whole region of logic between syncs — or specialized hardware, not per-cycle
fork/join. RRTL's own answer is consistent: its scaling moat is the **batch/data-parallel** axis (§4–4.7,
§4g.2–4g.4), where the work between syncs is thousands of independent lanes, not the per-cycle multi-core axis.
The contribution here is the substrate — the zero-copy order-independence that makes correct multi-core
execution *possible* — and the honest map of where it pays (it needs sub-µs sync or huge per-cycle work).

### 4g.13 Bulk-synchronous batch on the scalar JIT — past the wall

The §4g.12 wall is fundamental for *one* design across cores. The way past it is the bulk-synchronous
principle (Manticore/Parendi): make the work between syncs huge. RRTL's huge-work-between-syncs is the **batch**
axis — run MANY independent instances (different stimulus / seeds / parameter points) and let each advance its
*entire* run on a core, syncing exactly **once, at the join**, never per cycle. The barrier cost (§4g.12) is
then amortized over N cycles × M instances and disappears.

It is also trivially safe and simple: one compiled design (the `tick_many` code is reentrant over its state
pointer), M per-instance state buffers, and `par_iter_mut` hands each thread *disjoint* `&mut` buffers — no
locks, no atomics, no unsafe, no shared state. Crucially this is the **scalar** JIT, so unlike the SIMD/GPU
batch (§4–4.7, §4g.2–4g.4, which packs lanes into one vector op and needs them in lockstep) it **tolerates
control-flow divergence** between instances — one instance can take a branch another doesn't with zero penalty.
That is exactly the regression / fuzzing / design-space-exploration regime, where instances diverge by design.

On `cfgdsp`, 512 instances × 200 k cycles, single compiled design:

| | throughput | |
|---|---|---|
| 1 core (serial over instances) | ~30–73 M inst-cyc/s | |
| 8 cores (`par_iter_mut`) | **~740 M inst-cyc/s** | **~10× over 1 core** |

bit-exact (each instance's result is identical serial vs parallel). The speedup is at-or-above linear; the
*above*-linear part is an honest M3 artifact (the 1-core baseline is not pinned to a fast performance core, so
it is sometimes scheduled on a slower efficiency core) — the takeaway is **near-linear core scaling**, which is
what bulk-synchronous batch buys and what §4g.12's per-cycle engine could not. So RRTL has *two* multi-core
throughput engines now — the SIMD/vector batch (fast per core, lockstep lanes) and this scalar instance batch
(divergence-tolerant, dead simple) — alongside the GPU batch, all on the same packed IR; and the per-cycle
single-design multi-core path is documented for what it is: a wall that wants bulk-synchronous scheduling, which
on the throughput axis RRTL already has.

### 4g.14 Work-stealing for divergent instance batches

§4g.13 assumed every instance costs the same. Real batches don't: a fuzzed input triggers a slow path, a CPU
program halts early, a parameter point iterates more. With **static** partitioning (a fixed contiguous chunk
of instances per core — what a naïve `par_iter` split or a hand-rolled chunking does), whichever core draws
the heavy instances **gates the whole batch** while the others sit idle. The fix is dynamic per-instance
dispatch: a shared atomic "next instance" counter that each core `fetch_add`s when it's free — the simplest
form of **work-stealing**, at per-instance granularity (finer than range-splitting schedulers, which is what
skew needs). Same one-compiled-design / disjoint-per-instance-state setup as §4g.13; only the assignment of
instances to cores changes.

On a deliberately adversarial workload — `cfgdsp`, 512 instances, the first 1/8 (clustered, so they land on
the first static chunk) run **32× longer** — 8 cores, bit-exact between the two schedulers:

| schedule | throughput | |
|---|---|---|
| static (fixed chunks) | 40 M cyc/s | core 0 grinds the ~6 M-cycle heavy chunk; 7 cores idle |
| dynamic (work-stealing) | **413 M cyc/s** | **~10× over static** |

Static collapses to roughly *single-core* throughput because seven of eight cores finish their light chunks
and stop; dynamic keeps every core fed until the work is gone. The result is identical either way (work-
stealing only changes *who* runs each instance, not *what* it computes). This is the standard answer for
load-balancing irregular parallel work, and it's what makes the bulk-synchronous batch robust to the
divergence it was built to tolerate — exactly the fuzzing / regression / design-space regime, where instance
costs are unequal *by design*.

**Real divergence, not assigned cost.** The example above sets each instance's length by fiat. The same
schedulers run on *genuine data-driven* divergence (`examples/divergent_batch.rs`): a `workload` module loads a
per-instance `seed` into a counter and does a per-cycle multiply-accumulate until the counter hits 0, so each
instance **halts at a data-dependent cycle**. Execution stays fast — the baked `tick_many` advances in chunks
of K cycles, and the host checks the *counter register* between chunks to early-exit (no per-cycle round-trip;
the counter is a register so the raw read needs no settle). With the front 1/8 of instances seeded 25× longer
(clustered onto the first static chunk), work-stealing is again bit-exact and ~2× over static here. The
multiplier is smaller than the synthetic case for an honest hardware reason: the static run's single busy
thread keeps the core at *turbo* frequency, while the 8-core dynamic run throttles to base frequency across the
M3's P+E cores — so the per-core rate drops under full load. The *mechanism* (dynamic dispatch eliminating the
straggler) is what's being shown; the absolute multiplier is bounded by frequency scaling and core
heterogeneity, as every multi-core number on this laptop is.

### 4g.15 Fault simulation: the batch moat's killer application

Everything from §4g.13 on argues that the batch axis — *many independent instances in the lanes of one engine*
— is where a data-parallel simulator dominates. **Fault simulation** is the canonical industrial workload with
exactly that shape, and it makes the argument concrete. Given a gate netlist and a stimulus, a *stuck-at* fault
sim asks: for each of the ~2·(#nets) faults (each net pinned to 0 or to 1), does the fault ever change an
observable output versus the fault-free "golden" circuit? The ratio detected/total is **fault coverage**, the
number an ATPG flow optimizes. It is embarrassingly the batch regime: one design, one stimulus, N faults — put
the golden circuit in lane 0 and one fault per remaining lane, and a P-lane engine grades P faults for the cost
of a single fault-free simulation.

The one primitive the engine needs beyond §4g.13 is **fault injection**: a stuck-at is a net *clamped* to a
constant every cycle, and the clamp must *propagate* through the downstream cone — not merely overwrite the
stored value, which the next settle would recompute away. We add `set_force(signal, lane, value)` to the
bit-parallel engine (`bitparallel.rs`): after each `commit`, forced nets are re-clamped over the freshly-
committed register values, and *then* the settle runs, so the stuck value reaches the outputs and the next
cycle's logic. It is per-lane, so distinct faults coexist in one word; lane 0 carries no force and stays golden.

On the real synthesized `crc32` netlist (§4g.4b's import path — 195 cells, 32 flops) `examples/fault_sim.rs`
injects both stuck-at polarities on every flop (64 faults), one per lane, alongside the golden lane, and runs a
single 65-lane batch:

| check | result |
|---|---|
| golden lane vs fault-free scalar sim | **bit-exact** (96 cycles) |
| fault coverage | **64 / 64 = 100%** in one 65-lane sweep |
| detection latency (cycles to first divergence) | min 1, max 6, mean 2.1 |

100% is the *correct* answer here — every flop is a scrambled, fully-observed CRC output, and the LFSR excites
every bit — but the number is not the point. The **latency spread (1–6 cycles)** is: it shows faults
propagating through combinational cones of *different depths* before surfacing at an output, i.e. this is a
genuine time-domain fault simulation, not a one-cycle output mask. The mechanism is validated structurally
(a unit test forces a register stuck-at-0/1 on separate lanes and asserts the clamp holds at the output, an
un-forced lane stays bit-exact vs a control sim, and the fault eventually diverges — `bitparallel.rs` tests),
so the deliverable is correctness, not a throttle-sensitive throughput claim. The throughput *argument* is
§4g.4b's unchanged: 64 lanes ride in one u64, so the P-fault sweep costs one fault-free simulation's work —
which is exactly why serial fault simulators are slow and why the batch moat is the right tool for the job.

**At scale, on a real core.** The same campaign runs on the synthesized **picorv32** netlist — 11,735 cells,
**1,597 flops → 3,194 stuck-at faults graded in one 3,195-lane batch** (`examples/fault_sim_picorv32.rs`).
Here coverage is genuinely partial, and *that is the interesting result*: it depends on the stimulus's ability
to excite a fault, propagate it, and let it reach an observed output (the 71-bit memory interface). Feeding the
core three different fixed instruction streams for 400 cycles each, golden bit-exact throughout:

| stimulus (400 cyc, 71 observed bus bits) | coverage | detection latency (min/max/mean cyc) |
|---|---|---|
| random 32-bit words | 5.3% (169/3194) | 1 / 10 / 5.2 |
| `addi`-only (compute, no memory traffic) | 4.7% (151/3194) | 1 / 390 / 14.4 |
| compute-**and-store** loop (`sw` exposes the accumulator) | **20.3% (648/3194)** | 1 / 224 / 37.0 |

The jump from ~5% to 20% when the program *stores* its computed values is the mechanism validating itself:
random words trap the decoder into a near-quiescent state; `addi`-only keeps the core running but nothing
computed ever reaches the bus, so only fetch/PC/control faults (~150) are observable; adding a `sw` drives the
accumulator onto `mem_wdata`, and register-file/ALU faults *deep in the datapath* become detectable — the
latency climbs to a 224-cycle tail as those faults propagate across many cycles before surfacing. Coverage
that **responds to observability this way is the signature of a real fault simulator**, not an output mask, and
it connects directly to the observability-slicing story (§4j): what you observe determines what you can grade.
None of these are ATPG-grade test sets — they are three-line toy stimuli — but they grade *all* 3,194 faults in
a single batch sweep, which is the point: the moat turns "3,194 fault simulations" into one.

## 4h. Measured against Verilator and PyRTL

The JIT claims are only meaningful against the reference tools. We benchmark RRTL vs
**Verilator 5.030** (the compiled-C++ gold standard) and **PyRTL 0.12 FastSimulation** on an
identical design — `W` channels of a depth-`D` 32-bit mul-add accumulator, XOR-reduced to one
output — with identical stimulus, and **cross-check that all three produce the same output
word** as an independent reference model before trusting any timing. (Harness in
`bench/extsim/`.)

Single-instance, native `-O3`, bit-exact across all three:

| design | Verilator | PyRTL Fast | RRTL scalar JIT | RRTL / Verilator |
|---|---|---|---|---|
| W=1, D=1 | 19.3 | — | 35.3 | 1.8× |
| W=16, D=8 | 2.11 | 0.006 | 6.75 | **3.2×** |
| W=64, D=16 | 0.35 | — | 0.61 | 1.7× |

(M-cycles/s.) RRTL's JIT is **1.7–3.2× Verilator** on these designs and ~1000× PyRTL
FastSimulation (Python interpretation). Part of the margin over Verilator is structural:
RRTL evaluates the register-input (next-state) cone **once** per cycle — it lives in the
`tick_next` stream — whereas Verilator's standard two-`eval()`-per-cycle clocking (clk-low
then clk-high, both mandatory for edge detection) recomputes the full combinational cone on
*both*, and Cranelift schedules the independent channels into pipelined multiplies. This is a
handful of synthetic datapaths, not a full SoC, and Verilator is the more general tool — but
the comparison is fair and cross-checked.

And RRTL has an axis the others lack: the **batch** vector JIT (§4g.2–4g.4) sustains ~170
M-lane-cycles/s at 1024 lanes — roughly **80× a single Verilator instance's throughput**;
matching it would take ~80 Verilator processes. Plus there is no multi-hour C++ build:
`compile()` is milliseconds (§7).

**Methodology honesty.** Four gotchas each silently corrupted the result until caught — the
real lesson of the exercise:
1. *UNOPTFLAT.* Coding the chain as an unpacked wire array `t[j+1]=f(t[j])` makes Verilator
   treat the array as combinationally circular and fall back to iterative settling —
   re-evaluating the comb cone many times per `eval()`, **9× too slow**. Suppressing the
   warning keeps the penalty; the fix is a procedural `always @(*)` with blocking assignments.
   Always lint for UNOPTFLAT before quoting a Verilator number.
2. *OPT_FAST.* Verilator's default model optimization is `-Os` (size), not speed; force
   `OPT_FAST=-O3`.
3. *PyRTL output phase.* PyRTL's registered output lags one step (it samples pre-commit); a
   flush `step()` outside the timer aligns it.
4. *Trivial checksum.* XOR-reducing identical channels is always 0; a per-channel salt makes
   the cross-check meaningful.

Without fixing (1) alone, RRTL would have appeared to "win" by 9× — an artifact, not a result.

### 4h.1 The other side: a control-heavy core, and where the gap actually is

The datapaths above flatter RRTL. The honest counter-case is a real CPU — the YosysHQ
**picorv32** — driven through a host memory bus, same core/program/cycle-count for every tool.
Here Verilator leads: ~8.0 M-cycles/s vs RRTL's AOT (clang `-O3`) ~3.3–4.6 and the Cranelift
JIT ~1.6. The interesting question is *why*, because the answer rules out the obvious fixes.

We isolated each candidate by measurement rather than assumption:

- **It is not the optimizer.** The AOT lowers the same IR through clang `-O3` — Verilator's
  own toolchain. Running the IR-level specializer (§4b: const-fold, algebraic identities,
  copy-propagation, DCE) *first* removes **45.6 % of picorv32's machine instructions** and
  shrinks the generated C by 40 %, yet AOT throughput is **0.99× — neutral**, bit-exact. clang
  already performs exactly these passes on the flattened C; doing them in our IR is redundant
  for single-CPU codegen. (The same 45.6 % cut is *not* redundant for the GPU interpreter,
  which is op-dispatch-bound and unoptimized — §4b — nor for compile time, where the smaller C
  compiles ~1.1× faster. The pass pays on the throughput and iteration axes, not single-CPU
  latency.)
- **…but it *is* the optimizer for information clang cannot derive.** The neutrality above is
  specifically about *value-level* redundancy — const-folding, copy-propagation, DCE of provably
  dead code — which clang reconstructs from the flattened C. A categorically different kind of
  specialization does *not* fall to clang: freezing a **profiled runtime quasi-constant**. The
  instruction-subset freeze of §4g.8 — profile picorv32's ~65 per-instruction decode-flag
  registers, observe that the 50 for the workload's unused ISA classes stay 0 the whole run, and
  freeze them — encodes a fact clang fundamentally lacks, because *which* flags are 0 depends on
  the program bytes in RAM, not on the RTL. Const-folding then DCEs whole functional units (the
  barrel shifter, the comparators, the load/store byte lanes, the CSR/IRQ logic) **before clang
  ever sees the C**, so clang compiles half the design (3215 → 1479 machine instructions, 54 %
  removed). Applied to the AOT — not just the JIT of §4g.8 — this is a genuine wall-clock win:
  **1.19×** (bit-exact bus trace, same-machine A/B), rising to **1.25×** stacked with the
  control-flow recovery of §4g.5b (which itself gains under freeze — removing shared logic leaves
  proportionally more single-use mux cones to sink), for **1.40×** over the plain AOT. It is
  smaller than the JIT's 1.6–1.7× (§4g.8) for the expected reason: clang `-O3` already pipelines
  the full design well, so a 54 % structural cut buys less on the stronger backend — the recurring
  pattern that structural wins shrink as the codegen improves. The honest scope is unchanged from
  §4g.8: a full-ISA program freezes nothing, but a narrow workload (embedded firmware, no-FP
  kernels, pinned regressions) is the common case, and the simulator discovers the live subset
  per-run. (Driver: `examples/picorv32_aot_subset.rs`, with `AOT_CFLOW=1` for the stacked number.)
- **It is not state layout or localization.** The AOT already packs each signal into its
  natural 1/2/4/8/16-byte slot (not a uniform 16-byte slot) and keeps combinational
  intermediates in C locals (registers), writing only registers and observed outputs to the
  state buffer — the two structural tricks Verilator is known for. Both were already in place.
- **Part of it is the observation model, and that part is recoverable.** A CPU's memory bus is
  read *every cycle*, so the simulation cannot batch: each cycle pays the tick's leading
  combinational pass *plus* a full `settle()` (a second comb pass) triggered by reading the
  outputs. That redundant settle costs **1.51×** on picorv32. It cannot be deleted (the outputs
  reflect the post-commit state, which the tick's pre-commit comb did not compute), but it can
  be *sliced*: the testbench reads output **ports**, while `settle()` was writing all ~838
  internal comb signals back to the state buffer. Since internal values already flow through C
  locals, a settle that stores **only the output-port cone leaves** is bit-exact and far
  cheaper. This **observability-sliced settle** (the §4j idea applied to the settle pass)
  lifts picorv32 AOT **1.21×** (4.57 → 5.53 M-cycles/s, bit-exact vs the JIT oracle), closing
  the Verilator gap from ~1.74× to ~1.44× on the same machine. A `get_signal` of a genuine
  internal signal transparently falls back to the full settle, so generality is preserved.
- **The residual is Verilator's maturity.** With the settle sliced and codegen equalized, what
  remains is a ~1.4× edge from a decade of eval-loop tuning and RTL-level passes clang cannot
  reconstruct from flattened C. The headline example of those — *table-izing decoders* — we
  since built (§4g.5c): a ROM keyed on the instruction bits replaces picorv32's decode cone for a
  bit-exact ~1.12× on the JIT, confirming the residual is recoverable RTL-level structure, not a
  codegen deficiency. But the rest of that residual resisted three further structural probes, each
  refuted measure-first before building: a *case-to-jump-table* dispatch is marginal (picorv32's
  one-hot FSM is only ~8 of ~2443 sunk branches — the mux-eval-all tax is in the *datapath*, not
  the FSM); and a *topological comb scheduler* is clang-subsumed (a value-numbering probe shows the
  19% cross-stream redundancy is already removed by the settle→capture fusion + clang's GVN — the
  irreducible remainder is the dataflow IR's deliberate mux-eval-all, not a schedulable inefficiency).
  It is a real gap, but a narrow one,
  and on a *control-heavy serial core* — the workload least suited to RRTL's data-parallel
  thesis. The constructive reading: the levers that are neutral here (the specializer, batch,
  the vanished-settle of `tick_many`) are exactly the ones that pay on the throughput and
  compile axes where RRTL already leads. (Drivers: `examples/aot_settle_obs.rs`,
  `aot_specialize_probe.rs`, `aot_settle_probe.rs`, `toposched_probe.rs`.)

### 4h.2 The batch question, tested on a real core: N picorv32 vs N Verilator

The central thesis is that RRTL wins the *batch* regime (regression/fuzzing/DSE) because it runs
many instances in lane-parallel lockstep while Verilator has no batch model at all — N full
processes. We had shown this on a synthetic datapath (~80× one instance); the honest test is a real
core. We wrapped picorv32 in a self-contained SoC — core + an internal synchronous RAM + handshake
(`bench/sv/picorv32_soc.v`) — so each instance runs **autonomously**, no host bus, which is the
precondition for lane-batching a CPU: every lane carries its own RAM (its own program). It runs
**bit-exact** in RRTL's batch engines (independent CPUs in lockstep, each computing its program's
result).

A useful byproduct first: running picorv32+RAM as one design lifts the **single-instance** JIT from
~2.2 to **3.8 M-cycles/s**, because the autonomous `tick_many` removes the per-cycle host boundary
(the get/set-signal marshalling Verilator's inlined C++ testbench never pays) — narrowing the
single-instance Verilator gap from ~3.5× to ~2.1×.

The batch result, all measured on the same 8-core machine (256 instances, independent programs):

| engine | M-instance-cycles/s | vs Verilator |
|---|---|---|
| Verilator (16 parallel processes) | **69.4** | 1.0× |
| RRTL scalar JIT batch (rayon over instances) | **20.2** | ~3.4× behind |
| RRTL SIMD lane batch | 0.4 | ~170× behind |

This is an **honest negative for the control-heavy case**, and it is instructive. The SIMD lane
engine — the moat for datapaths — is the *wrong* tool here: picorv32 is control-heavy (it falls to
the SIMD fallback path) and, fatally, each lane's RAM is independent, so every memory access is a
divergent per-lane **gather** that cannot vectorize. The right batch engine is the **scalar JIT
batch** (§4g.13): one independent native-code instance per core, normal array memory, divergence-
tolerant. But it is bounded by the single-instance gap — and because *both* RRTL and Verilator
parallelize embarrassingly across cores, the batch ratio (~3.4×) is essentially the single-instance
ratio (~2.1×), slightly widened by Verilator's native host-array memory versus RRTL's in-RTL RAM.

The conclusion sharpens the thesis rather than denting it: **the batch moat creates a new advantage
only where RRTL's per-lane cost is competitive** — i.e. SIMD-friendly *datapath* designs (lanes
vectorize 4-wide and Verilator still needs N processes), or massively-parallel GPU hardware
(thousands of lanes ≫ N processes). A *control-heavy* core on a CPU is exactly where it does not
win: the same divergence-free-dataflow tax that costs single-instance latency (§4h.1) also denies a
batch advantage, because the SIMD engine can't amortize a control-heavy design across lanes. Honest
boundary of the moat: datapaths and GPUs, not control cores on a laptop CPU. (Drivers:
`examples/picorv32_soc_check.rs`, `picorv32_jit_batch.rs`, `picorv32_batch_bench.rs`;
`bench/sv/picorv32_soc.v`.)

We also closed the one remaining cell — *the control core on the GPU*. The design-as-data GPU
interpreter runs the picorv32 SoC **bit-exact across 32 768 lanes** (each lane an autonomous core with
its own RAM; we added `set_memory_replicated` to load the program image per lane, and chunk the
multi-cycle dispatch to stay under the Metal GPU watchdog). But throughput **saturates at ~1.7
M-lane-cyc/s** (0.3 → 1.2 → 1.5 → 1.7 at 1 K → 4 K → 16 K → 32 K lanes), which is **~12× behind the
CPU scalar-JIT batch (~20 M)** and ~40× behind 16 Verilator processes (~69 M). The reason is the same
two costs, now on the GPU: the *interpreter* dispatches all ~21 k packed records per lane per cycle
(it is not compiled native code), and each lane's instruction RAM is an uncoalesced **gather**. So GPU
scale does *not* flip the control-core result — for a control core the GPU interpreter is the *worst*
batch engine, and beating the CPU would require a *compiled* per-design GPU kernel (§4i), which still
inherits the per-lane gather. The moat is for datapaths and compiled kernels, not interpreted control
cores. (Driver `crates/rrtl-gpu-sim/examples/picorv32_gpu.rs [lanes] [cycles]`.)

### 4h.3 The other side of the boundary: a datapath, where the moat wins ~14×

The picorv32 result is only half the story; the honest test is to measure the *positive* case on the
same machine. On a **datapath** — the `bench/extsim/dut.v` design, `W=16` channels of a depth-8
32-bit mul-add accumulator, the same source run through Verilator — RRTL's vector-JIT batch engine
runs N independent instances in lane-parallel lockstep, while Verilator runs N OS processes:

| engine | M-instance-cycles/s | |
|---|---|---|
| **RRTL vector-JIT batch** (1024 lanes, one process) | **474** | 1.0× |
| Verilator (8 parallel processes, one per core) | 33 | **~14× behind** |
| Verilator (single instance) | 5.2 | — |

So the moat boundary is measured on **both sides** on the same hardware:

| workload | RRTL batch vs N Verilator processes |
|---|---|
| datapath (W×D mul-add) | **~14× ahead** |
| control core (picorv32) | ~3.4× behind |

Same architecture, opposite outcomes — which is exactly the thesis, not a hedge. The datapath
vectorizes cleanly (four lanes per SIMD op, regular structure, no divergent control, no per-lane
memory gather), so RRTL's per-lane cost is competitive *and* Verilator has no batch model at all —
N lane-cycles beat N processes by an order of magnitude. The control core can neither vectorize (it
falls to the SIMD fallback) nor avoid the per-lane memory gather, so its per-lane cost collapses and
the single-instance gap carries through. The moat is real and large where RRTL's lane model fits;
where it doesn't, RRTL trails by the single-instance ratio. (Driver `examples/extsim_compare.rs`
vs `bench/extsim/` Verilator.)

## 4i. Compiled per-design GPU kernels, and a tensor-core linear-cone offload

§4.1 rejected *codegen a shader* because a WGSL shader specialized to a large design
reaches ~115 MB and the Metal compiler hangs — the same compile-wall pathology as
Verilator's C++, relocated to the GPU driver. That argument is specific to the
WGSL/Metal pipeline. Emitting a per-design kernel in a language with a **mature C
compiler** — OpenCL C (Apple's framework is clang-based) or CUDA, the same family as
the AOT C backend (§4b) — does *not* hit that wall: the emitted kernels are compact
(crc32 → 29 lines, a RISC-V execute core with a register file → 369 lines) and compile
in **seconds, not hours**. So alongside the interpreter we ship a **compiled per-design
GPU backend** — the GPU analogue of the AOT C backend, and a deliberate latency/
throughput-max complement that *re-accepts* a per-design compile in exchange for two
things the interpreter cannot offer: zero opcode-dispatch overhead, and a clean mapping
of linear logic onto **tensor cores**. (The interpreter, §4–4.7, remains the default for
its O(1) iteration story, §7; this is the other end of the tradeoff.)

`gpu_codegen::emit_kernel(&machine, Flavor::Cuda|OpenCl)` walks the packed machine IR
and emits one kernel where **one GPU thread = one lane**, state laid out lane-major
(`st[slot*nl + lane]`, so consecutive threads touch consecutive addresses = coalesced),
performing an eager tick (async-reset → settle → capture-to-next-temps → commit →
post-settle) per cycle. Value-ids are emitted as plain locals prefixed per stream
(`a/c/n/m/A/C`) so all streams share one C scope without collision. The OpenCL kernel
**compiles and runs bit-exact on a real Apple M3 GPU** (1024 lanes, crc32 → `0xf7c3094b`,
identical to the in-process gate reference); the CUDA kernel emits valid source for an
NVIDIA CI path (no NVIDIA on the dev machine).

**Memory without a new buffer.** A naïve port would add a second GPU buffer for register
files / RAMs. But in the lane-major layout a memory *entry* is just another *slot at a
computed index*: lay memories after the signals, and entry `e` of memory `m` lives at
slot `base_m + e`. Then `MemRead` is `(addr < depth) ? st[(base_m+addr)*nl+lane] : 0`
and `MemoryWrite` is the mirror — no extra binding, no extra kernel argument, just a
larger state buffer (sized by `state_slots = signals + Σ depth`). A RISC-V execute core
with a 32×32 register file runs 256 lanes × 500 cycles on the M3 **bit-exact against the
Cranelift JIT** (`pc=0x7d0, x10=5`).

### 4i.1 GF(2)-linear cones as a 1-bit matrix product (the BMMA target)

A combinational cone built only from XOR / NOT / constant-AND-OR / bit-routing is
**affine-linear over GF(2)**: its transfer function is `out = M·in ⊕ c` for a binary
matrix `M` and constant `c`. This is exactly the structure of CRC, FEC/scramblers, LFSRs
and much symmetric-crypto round logic. RRTL detects such cones and extracts `M` by
probing the cone's own reference evaluator with the zero vector (→ `c`) and each basis
input bit (→ that column of `M`) — `O(in_bits)` evaluations, no symbolic algebra.

The payoff on a GPU is that `out = M·in ⊕ c` is a **1-bit matrix product**, and the
1-bit AND-popcount that realizes each output bit — `out[i] = parity(row_i ∧ in) ⊕ c[i]`
— is *precisely the primitive a tensor-core BMMA unit executes in bulk*
(`wmma::bmma_sync`, Ampere+). `emit_kernel_linear` emits a register update as exactly
this: pack the cone's leaves into an input word, then one `popcount(row_i & in) & 1` per
output bit. The crc32 register cone extracts as a `M[32×32] ⊕ const`; the emitted BMMA
kernel runs on the M3 **bit-exact** against the gate-tree kernel (both `0xf7c3094b`).

### 4i.2 Selective per-cone offload in a mixed design

Real designs are not wholly linear. For a *mixed* module — a linear CRC block beside a
non-linear multiply-accumulate and a compare-gated counter — `emit_kernel_hybrid` routes
**each register cone to the backend that fits**: linear cones to the BMMA matrix form,
everything else to the gate tree, in one kernel. The offload is only real if the linear
cone's gate logic actually *leaves* the SIMT path, so we run a **liveness pass** over the
`tick_next` stream: a value-id is live iff some non-linear capture / store / memory-write
needs it; instructions feeding *only* an offloaded register are pruned. On the mixed
design this drops the CRC's 14-instruction XOR tree (25 → 11 tick instructions kept), and
the hybrid kernel matches the full gate-tree kernel **bit-exact** across all three outputs
on the M3. A mixed SoC can thus run its linear blocks on tensor cores and the rest on SIMT
lanes within a single compiled kernel.

### 4i.3 Throughput, and an honest crossover

Batch throughput on the M3 (crc32, 2000 cycles/launch, NDRange→finish timed after warm-up),
in aggregate **lane·cycles/s**, against one fast CPU core (a hand-tuned native byte-parallel
crc32 at **118.8 M-cycles/s** — a strong baseline, not a slow interpreter):

| lanes | gate-tree | BMMA/linear | vs 1 CPU core |
|---|---|---|---|
| 16,384 | 4,182 M | 2,869 M | 24× |
| 65,536 | 4,719 M | 7,277 M | 61× |
| 262,144 | 19,315 M | 11,035 M | **93×** |
| 1,048,576 | 17,129 M | 11,067 M | **93×** |

The gate kernel saturates ~262k lanes at ~17–19 G-lane-cycles/s — **~93× a single fast CPU
core**, the concrete batch moat for regression / fuzzing / design-space fleets (matching it
would take ~90 such cores).

The second column carries the more useful lesson. **The BMMA form is *slower* than the gate
tree here** (11 G vs 17–19 G). On the M3, OpenCL exposes no tensor cores, so `M·in` runs as
32 `popcount`s per register per cycle — *more* ALU work than crc32's *sparse* XOR tree, which
is a handful of cheap gates. The matrix form therefore wins only where one of two things
holds: (a) a real tensor core collapses the per-bit popcount loop into a single
`wmma::bmma_sync` (the NVIDIA CI path), or (b) the matrix is *dense* enough that the
equivalent gate tree explodes — `M·in` is `O(out×in)` regardless of density, whereas a dense
GF(2) cone is a quadratic XOR mess. For a sparse linear cone on a popcount-only GPU, the right
emission is *gates*, and the hybrid emitter can make that choice per cone. The value of the
linear analysis is not that BMMA is unconditionally faster — it is that the cone is now in a
form the hardware accelerator can take directly, with the crossover characterized.

### 4i.4 Engineering notes

- The same value-width-tracking bug class as the AOT backend recurs: a stubbed operand-width
  helper made `Concat`/replication shift by multiples of 64 instead of real part widths
  (`<<64` on a `u64` is undefined) — caught only by *running* on the GPU (it compiled fine and
  produced a wrong CRC). Per-block width maps fix it; "compiles on the GPU" is not "correct on
  the GPU."
- Slot-lookup gotcha: a compiled module's signals are unprefixed, but the *packed* program's
  signals are top-prefixed and slot-indexed — address the kernel by packed slot index.
- All GPU example runs use a background launch + kill-after-timeout guard (Metal/driver hangs).

### 4i.5 The matrix form as a native CPU engine, and a multi-engine partitioned simulator

§4i.3 showed the BMMA form *losing* to the gate tree on a popcount-only GPU. On the **CPU** the same
matrix `M` has a different, winning realization. Transpose `M` (column-major) into per-output-bit rows and
each output bit becomes a flat **XOR of input-bit planes** — no AND-popcount, no per-bit loop. Emitted as
bit-sliced `uint64_t` C (one word = 64 lanes) with an inner word loop, clang `-O3` auto-vectorizes it; and
because it is *pure XOR* — no ripple-carry, no width dispatch, no mux — it vectorizes better than the
gate-level bit-slice AOT (§4g.4b), which must evaluate the full unrolled-CRC netlist. crc32 collapses to
73 planes / 91 word-XORs per cycle. Measured (4096 lanes, bit-exact against the SIMD-CPU oracle;
same-process back-to-back ratios, since this laptop throttles erratically — see the methodology note below):
the **matrix XOR-AOT runs ~2.0–3.5× the gate-level bit-slice AOT and ~1.0–1.7× the vector JIT** on crc32.
So the linear analysis pays *on the CPU too*: the compact matrix beats evaluating the gate netlist, and it is
the portable XOR-GEMM that a real tensor core would only accelerate further.

That makes the matrix a *backend* like any other, and the natural next step is a simulator that uses it for
the linear cones and a general engine for the rest — **automatically**. `HybridSimulator::new(program, lanes)`
(1) detects the GF(2)-linear register cones and compiles them to the matrix XOR-AOT, (2) derives a
backward cone-of-influence over the non-linear registers and observability-slices the design (§4j) so the
linear cones are *pruned* from the general engine, and (3) auto-picks the best general backend for the
remainder — the vector JIT when the slice fits it (≤32-bit, no async reset), else the SIMD-CPU interpreter.
It presents one handle-keyed `set`/`get`/`tick` interface and routes internally: a linear register, or an
output port aliasing one, is read from the matrix AOT; everything else from the general engine; a shared
input is driven into both. On the mixed design it auto-selects the vector JIT for the multiply-accumulate
slice, pairs it with the matrix AOT for the CRC, and is **bit-exact** against the full SIMD CPU on every
output. This is genuinely *multi-engine* simulation: heterogeneous backends, each chosen for its partition,
composed behind one interface.

The two partitions hold disjoint register state, so for **independent** partitions `tick_many` runs them
**concurrently** — the matrix AOT (which is `Send`) on a worker thread, the general engine on the main
thread (the JIT's code module is not `Send`, so keeping it on the main thread side-steps that cleanly). When
the partitions are **coupled** — each reads the other's registers across the cut — correctness needs a
per-cycle **boundary exchange**: the slicer's `signal_origin`/`boundary_inputs` maps already expose which
signals cross, and we copy them (register-stable, using the previous cycle's committed values, before both
partitions tick). Boundary signals that are internal registers carry no top-level handle, so they are
addressed by *local index*; that is why the coupled path requires the index-addressed JIT general backend.
`tick_many` runs coupled partitions sequentially per cycle behind the exchange barrier and keeps the
concurrent batch for independent ones; both are bit-exact (validated on a mutually-coupled CRC/accumulator
where `crc <= linearCRC(crc ⊕ acc)` and `acc <= acc + a·b + crc`).

**The honest accounting.** The offload's value depends entirely on what it replaces. Against the SIMD-CPU
*interpreter*, removing the unrolled CRC and running it as 91 XORs is ~3.6×. Against a *compiled* general
engine the win is far smaller — the vector JIT already compiles the CRC efficiently — measured ~1.15× on a
single thread for the 62%-linear mixed design, rising to ~1.47× when the independent partitions run on two
threads. The dramatic gains are reserved for highly-linear designs (CRC/FEC/crypto-dominated) and for real
1-bit tensor cores (the BMMA path, which this Apple-Silicon machine cannot run). The durable, machine-
independent results are structural: bit-exactness, the linear work genuinely leaving the general engine, and
the partitions being independent (concurrency-friendly) or coupled (correct under boundary exchange).

*Measurement methodology (transferable).* On a thermally-throttling laptop, throughput ratios are treacherous:
naïve timing reported a 2-thread "speedup" of 2.2× (above the 2× ceiling — physically impossible), because the
concurrent run was always timed *last* each trial and caught the warmest clock state. Rotating the measurement
order across trials and taking best-of-N stabilized it at a believable ~1.47×. A measured ratio above its
theoretical bound is the tell-tale of ordering/thermal bias; rotate and take the best sample.

### 4i.6 Temporal leap: closed-form cycle jumps for linear blocks

Every optimization so far reduces the *constant factor* per cycle. The linear extraction unlocks something
stronger for GF(2)-linear blocks — a reduction in the *number of cycles simulated*. A linear register cone
evolves `s(t+1) = M·s(t) ⊕ c` over GF(2), so `s(t+N) = M^N·s(t) ⊕ (Σ_{k<N} M^k)·c`, and **`M^N` by repeated
squaring is O(log N) matrix mults** rather than N cycle-steps. This is the same structure that *defeated* the
two ideas of §4f and the C-slow probe — register feedback / self-recurrence — turned into an asset: those
techniques need the absence of feedback or its stability, whereas the leap *is* the recurrence, solved in closed
form. `LinearLeap::build` assembles the combined transition matrix over a design's linear register cones
(crc32's is two coupled registers, `crc` plus the blocking temp `c`, a 64-bit combined state) and the per-cycle
constant; `leap_idle(s0, N)` exponentiates an augmented matrix. Validated bit-exact against step-by-step: N=100
000 in 22 matmuls, **N=2⁴⁰ (≈10¹²) in 41 matmuls** — a state a stepping loop could never reach — all matching the
stepped reference. Because the win is *op-count*, it is provable on a throttled machine where wall-clock is noise.

Three generalizations make it a usable engine primitive. **(1) Constant input.** With the input held at `u`
(not just idle), the per-cycle offset is `c_u = B·u ⊕ c` (the extracted input matrix `B`), and
`s(N) = M^N·s0 ⊕ G_N·c_u` with `G_N = Σ_{k<N} M^k`. Obtaining `G_N` *as a matrix* — so each lane's distinct `c_u`
can multiply it — uses the block matrix `[[M, I],[0, I]]^N = [[M^N, G_N],[0, I]]` (fits a 128-bit row when the
state is ≤64 bits; crc32's 64 fits exactly). **(2) Streaming.** A *varying* input stream of length N splits into
P segments, each processed from zero state independently (`V_i`) and stitched as
`total = M^N·s0 ⊕ Σ_i M^{(P−1−i)L}·V_i` — the generalized zlib `crc32_combine`, for any linear block. Because the
`V_i` are independent, one sequential linear stream becomes **P-way parallel** (`fold_stream` runs the segments
on rayon), bit-exact. **(3) Integration.** `HybridSimulator::leap(N)` advances the *independent* linear partition
by `M^N` (built once, applied per lane to the matrix-AOT's bit-sliced state) while the non-linear partition steps
normally — bit-identical to `tick_many(N)` when the linear inputs are held constant; on `mixed.sv`,
`leap(100 000) == tick_many(100 000)` exactly while the CRC partition jumps in ~17 matmuls.

The regime is precise: the leap delivers the *final* state after N constant-input (or P-segmented varying-input)
cycles — fast-forwarding a free-running LFSR/scrambler/counter to a checkpoint, aligning a descrambler, computing
a CRC over a buffer, or skipping an idle linear submodule inside a hybrid — not per-cycle intermediates, and (for
the constant-input form) a ≤64-bit linear state. Within that regime it is the rare optimization that changes the
asymptotics: O(N) cycles → O(log N), or → N/P wall-clock for a single stream.

## 4j. Observability slicing: compute only what you observe

The most RTL-sim-specific lever, and one of the largest, is almost embarrassingly simple: a testbench
*observes* a tiny fraction of a design's signals — a checksum, a trap line, one bus trace — yet an oblivious
simulator evaluates the whole design every cycle. The fix is a backward **cone-of-influence** from the
observed set: a signal (or memory) is live iff it transitively feeds something observed, through
combinational definitions *and* register next-state cones. Everything outside the cone is dead and is pruned.

`cone_of_influence(program, observed_signals, observed_memories)` computes the live masks by fixpoint (an op's
reads enter the cone once its destination is in the cone — reusing the slicer's `collect_op_reads`/`op_present`),
and `slice_present` (already in the partitioner) emits the pruned program. It is **static** — the observed set
is fixed, so the result is fully data-oblivious and composes with *every* backend, multiplying the JIT, the
SIMD/scalar batch, and the GPU throughput rather than trading against them.

On 16 independent depth-16 multiply pipelines, observing only pipeline 0's output:

| | full | sliced (observe 1 of 16) |
|---|---|---|
| signals | 274 | 17 (**94% pruned**) |
| machine instrs | 2254 | 154 (**93% pruned**) |
| JIT throughput | ~4–5 M-cyc/s | ~40–120 M-cyc/s (**~9–28×**) |

bit-exact on the observed register. The structural prune (94% of signals, 93% of instructions) is the robust,
deterministic result; the *throughput* multiplier varies run-to-run (cache/thermal) but is reliably well above
the naïve 16× — because the full program's larger instruction and state working set also thrashes cache, so
pruning to the cone wins on footprint as well as op count. A nice correctness detail falls out for free: the
**clock** is *not* in the cone (it drives no data
path; RRTL is cycle-based, so each `tick` already *is* an edge), so it is pruned with everything else, and the
sliced design ticks correctly without it. For the common case — a regression run that checks one output, a
fuzzing campaign that watches a coverage signal, a bring-up test that reads one register — this turns "simulate
the SoC" into "simulate the cone that reaches the thing you're looking at," statically, on top of every other
optimization here.

## 4k. Sparse / lazy huge memories

The canonical huge-memory case in RTL sim is a CPU's address space: 32 bits is **4 GB**, which a dense array
can't hold — yet any real program touches a handful of pages (code low, stack high, a little MMIO). The model
that fits is a **demand-paged sparse memory**: a page is allocated on first write, unwritten addresses read as
0, so the *full* address space lives in a few KB. Two things worth separating. RRTL's *internal* memories (in
the JIT state buffer) are already physically lazy for free — a zeroed `Vec` is `calloc`/demand-paged, so
untouched pages cost no physical RAM even single-instance. The case that genuinely *needs* sparsity is the
**host-driven address space**: a flat host array can't span 4 GB per instance, and a bulk-synchronous batch
(§4g.13) would multiply that by the instance count.

A 64-line page-table memory (`HashMap<page, Box<[u32; 1024]>>`) demonstrates it: picorv32 runs with code at
address 0 and its stack initialized near 1 GB, pushing/popping there and storing a result low. The program is
bit-exact (sum-via-stack = 20), the highest address touched is at the very top of the space, and:

| | dense flat array | sparse page table |
|---|---|---|
| RAM to model the run | ~4 GiB (span to the stack) | **8 KiB (2 pages)** |

So the whole 32-bit space is modeled in 8 KiB, and only touched pages cost RAM. This is exactly what makes the
batch composable: N CPUs exploring a 4 GiB space cost N × (touched pages), not N × 4 GiB — a stack page, a code
page, some heap per instance. (The same structure extends to designs with large on-chip RAMs whose access is
sparse; the host-bus path is just the most common place it's *required* rather than merely nice.)

## 4l. Symbolic execution: the simulator as a formal engine

Every engine so far advances *concrete* values. Swap each 1-bit signal's concrete value for a **reduced ordered
binary decision diagram** (ROBDD) — a canonical boolean function of symbolic input variables — and the very
same gate schedule becomes a *symbolic* execution: run the packed streams with the gate ops (And/Or/Xor/Not/Mux)
mapped 1:1 onto BDD `apply`, and each output signal ends up as *the boolean function it computes over the
symbolic inputs*. `symbolic.rs` is the bit-parallel engine (§4g.4b) with one BDD per signal instead of
`ceil(L/64)` u64 lanes, plus a small hash-consed BDD manager (ite / and / or / xor / not, `sat_one`, `size`) —
no external SAT/SMT/BDD dependency. It is the same ~250-line engine shape, reused for a different semiring.

This turns the simulator into a lightweight formal tool. Validation first: symbolic-executing the synthesized
**crc32 netlist** for one cycle with the 8 `din` bits as variables yields 32 output BDDs totalling **196 nodes**,
and they agree with a concrete run on **all 256** input assignments — an *exhaustive* equivalence proof, not a
sampled one (`examples/symbolic_exec.rs`, Act 1). The BDDs stay small because crc is GF(2)-linear (parity
functions have linear BDD size), the same structure §4i–§4i.6 exploited for the tensor-core and temporal-leap
paths.

The payoff is **ATPG** (automatic test-pattern generation), which closes the loop with the fault-sim work
(§4g.15). A stuck-at fault's *detection function* is `OR_o(golden_o XOR faulty_o)`, built by snapshotting the
golden output BDDs, injecting the fault (`set_force_const`, the symbolic sibling of the batch engine's stuck-at
clamp), and re-settling — both live in one BDD manager, so the XOR is valid. Then:

* a **satisfying assignment** (`sat_one`) *is* a test vector; and
* the **false BDD** is a *proof the fault is untestable* — a redundancy, a negative that simulation can only ever
  fail to disprove, never establish.

On a reconvergent-fanout cone `y = (a&b) | (a&~b)` (which is just `a`, so the `~b` path is redundant), the engine
reports the `nb` wire stuck-at-1 as **provably untestable** (detection BDD = false) while generating concrete
test vectors for the testable faults (`nb` s-a-0 → `a=1,b=0`; `w` s-a-0 → `a=1,b=1`; `v` s-a-1 → `a=0`) — Act 2.
And Act 3 unifies the two halves: symbolically generate a test for a real crc32 register fault (`crc_5`
stuck-at-0 → `din=0x01`, initial state `0`), then *confirm it on the concrete bit-parallel engine* — golden
`0x690ce0ee` vs faulty `0x690ce0ce`, detected. Symbolic **generates** the test; the batch moat **grades** it (and
thousands more) — the two engines share the same packed IR and the same stuck-at clamp semantics, which is what
lets a test cross from one to the other.

Reconciling those semantics surfaced a real refinement: a forced *output that aliases a register* is a
combinational-wire fault, so the batch engine's clamp — previously applied only after register commit — now also
re-applies after a comb store in `settle`, matching the symbolic engine. Register-only fault campaigns
(crc32 100%, picorv32) are unchanged (registers aren't comb-stored), but comb-wire faults now propagate
identically in both engines. Forcing a *primary input* is deliberately not modeled as a fault (it is just
constraining that input). The method is exponential in the worst case — BDDs are — so it targets bounded cones
(a few cycles, a partition, a linear block), the complement to the concrete engines' unbounded-but-per-point
reach; the two compose (symbolically find the input that excites a corner, concretely run the fleet from there).

## 5. Single-design latency: register-cone partitioning

For the *other* axis (one massive design, billions of cycles), RRTL has a RepCut-style
(ASPLOS'23) **register-cone partitioner**: slice the packed program into cones feeding
disjoint register groups, replicate shared combinational logic across cones to cut the
cross-partition critical path, and run groups on a rayon thread pool
(`PartitionedSimulator`, with `SimdCpuSimulator` as the per-group engine). This is the
single-instance-latency complement to the GPU's batch-throughput story.

## 6. Negative result: heterogeneous CPU‖GPU lane-splitting

The intuition "split the lane batch across CPU and GPU and run both" is **net-negative
on unified-memory Apple Silicon**. Once the GPU saturates memory bandwidth, the CPU's
share of the same unified memory contends with it; the disjoint-lane `LaneSplitSimulator`
and the work-stealing balancer (`WorkStealingBatch`, shared atomic lane cursor, large
GPU tile + small CPU tile) both lose to GPU-only. The honest takeaway: on shared-memory
SoCs, *add* parallelism only where it adds bandwidth, not just compute. (On discrete-GPU
/ separate-memory systems the balance may differ — untested here.)

## 7. Iteration scalability

Because the GPU path is a **fixed kernel interpreting data**, adding/altering a design
is a buffer upload, not a recompile — no per-design shader build, no Verilator-style
hours-long C++ compile, no multi-GB objects. This is the compile/iteration axis where
an interpreted packed-IR simulator structurally beats an AOT-codegen one on huge designs.

## 8. Limitations and future work

- **2-state only.** No X/Z; combinational settle assumed acyclic. 4-state and loop
  handling are unaddressed.
- **Single implicit clock, multi-clock seeded.** Every engine treats one `tick` as a
  simultaneous edge of all clocks (the clock→register association exists in the IR but is
  dropped at lowering — `CaptureReg` carries no clock — so every register captures each
  tick). A multi-rate design therefore cannot be simulated correctly yet. The first slice
  landed: the gold simulator gained `tick_clocked(active)`, which captures only registers
  (and memory writes) whose clock has a rising edge this step (harness-scheduled, exact for
  a flat top module; equivalent to `tick()` when all clocks are listed). The clock→register
  map is then carried into the lowered IR as a side-table (`PackedProgram.reg_clocks`,
  populated at lowering — no change to the `CaptureReg` variant, so the ~65 match sites are
  untouched), and the first batch engine is gated: `SimdCpuSimulator::tick_clocked` commits
  only captures whose clock is active, validated **bit-exact against the oracle** on an
  independent multi-rate design. The compiled JIT — the harder case, since one compiled tick
  serves a fixed register set — was then handled: a `tick_clocked(state, mask)` function takes
  the active-clock set as a runtime bitmask and commits each clocked register via a branch-free
  `select((mask>>bit)&1, next, current)` (the same idiom as its sync-reset), also bit-exact
  against the oracle. The AOT followed as the direct C mirror — `tick_clocked(st, mask)` emits
  `if ((mask>>bit)&1) reg = next;` per clocked register, sharing a deterministic clock→bit
  assignment with its wrapper — likewise bit-exact. So both compiled backends and the SIMD
  batch engine now support multi-clock against a common oracle. **Memory writes** gate the same
  way via a second side-table (`mem_clocks`): the SIMD engine skips inactive-clock memories, and
  both compiled backends AND the clock-active bit into the write's enable condition — validated
  bit-exact on a RAM written by a slow clock and read continuously (one subtlety the test caught:
  the clock→bit map must union *register and memory* clocks, since a clock can drive only a RAM).
  Remaining: the interpreter/GPU paths (thread the side-tables through encoding) and mapping
  clocks through instance ports for hierarchy. It composes with the batch moat — each domain
  stays data-parallel.
- **Oblivious GPU/SIMD path.** Activity-based skipping is implemented and pays on the
  single-instance JIT (§4g.1) but only for coarse cones; it remains anti-correlated with
  lane count on the batch engine (§4f), and a cone-merging pass to realize the JIT win on
  fine-grained IR is future work.
- **≤128-bit values** (4 limbs) in the GPU interpreter.
- **Compiled GPU kernel (§4i) is ≤64-bit** (OpenCL C has no `__int128`); wider buses need a
  CUDA-only `__int128` / two-slot path. Its CUDA flavor and the real `wmma::bmma_sync` swap-in
  for the BMMA popcount loop (§4i.3) are emit-only here — untested without NVIDIA hardware.
- **Mixed-width residual.** The per-op 1-limb fast path (§4.5.1) recovers ~half the
  mixed tax; the rest is the packing `voff` indirection load, left in place for the
  footprint win. Closing it fully would require a non-packed layout mode.
- **AVX2 unverified** on the dev machine (toolchain-gated); needs CI.

## 9. Reproducing the numbers

All figures come from examples in `crates/rrtl-gpu-sim/examples/`:

- `gpu_throughput <bits> <width> <depth> <lanes> <steps> <mixed>` — §4.4–4.6 throughput
  (the `mixed` flag adds one 64-bit register to force `max_limbs = 2`).
- `gpu_multilimb_check`, `gpu_feature_check`, `gpu_mem_check` — bit-exactness vs
  `SimdCpuSimulator` (multi-limb mul/add/xor/slice/concat; reset+concat; memories).

The compiled per-design GPU kernels (§4i) come from examples in `crates/rrtl-sim-ir/examples/`:

- `gpu_emit <design> <top>` — emit the OpenCL + CUDA kernels; `bench/sv/check_opencl.c` build-
  validates the OpenCL one on the M3.
- `opencl_run` / `opencl_run_cpu` — run crc32 / a register-file CPU core on the M3, bit-exact vs
  an in-process reference / the Cranelift JIT (§4i, memory).
- `linear_gpu_run` / `hybrid_gpu_run` — the BMMA linear kernel and the mixed-design hybrid
  selective offload, each bit-exact vs the gate-tree kernel on the M3 (§4i.1–4i.2).
- `gpu_throughput` — the lane-sweep batch throughput and CPU-core comparison of §4i.3.

The Verilator-gap dissection (§4h.1) uses `crates/rrtl-sim-ir/examples/` (`--features "aot jit"`):
`aot_specialize_probe` (45.6 % IR cut, runtime-neutral), `aot_settle_probe` (the 1.51× per-cycle
settle cost), and `aot_settle_obs <N>` — picorv32 on the AOT with the observability-sliced settle,
bit-exact vs the JIT oracle, A/B'd against the full settle via `AOT_NOOBS=1` (1.21×). The
profile-guided instruction-subset freeze on the AOT (§4h.1) is `picorv32_aot_subset` (`--features
aot`): profiles the decode flags, freezes the always-0 ones, AOT-compiles plain vs specialized, and
reports the 1.19× (1.25× with `AOT_CFLOW=1`) with a bit-exact bus-trace assertion. The harness
`bench/sv/run_picorv32.sh [CYCLES]` runs RRTL JIT/AOT vs Verilator on the same core/program/cycles
(it caps the Verilator build at `-j 2` — many parallel `-O3` compiles of the generated picorv32 C++
otherwise exhaust memory on constrained machines).

Dynamic specialization (§4g.6) is `crates/rrtl-sim-ir/examples/dyn_specialize.rs`
(`--features jit`): profiles a stable `mode` register on `bench/sv/cfgdsp.sv`, freezes + re-JITs,
runs guarded, and forces a deopt via a mid-run reconfiguration — reporting raw and
steady-state-with-guard speedups, all bit-exact against a generic reference. The pass itself
(`specialize::freeze_signals_program`) is unit-tested for prune + (under `jit`) bit-exactness.

The tiered JIT (§4g.7) is `crates/rrtl-sim-ir/examples/tiered_jit.rs` (`--features jit`): a Tier-0
generic JIT with a lightweight sampled profiler over the narrow control registers, promoting a
dominant-value-speculated Tier 1 (cached, guarded, deopt-safe) on `bench/sv/cfgdsp.sv` under a
blip stimulus — comparing bias-0.90 vs exact-freeze policies across excursion frequencies,
best-of-3 interleaved, all bit-exact.

Instruction-subset specialization of picorv32 (§4g.8) is
`crates/rrtl-sim-ir/examples/picorv32_specialize.rs` (`--features jit`, arg = path to `picorv32.v`):
profiles the decode-flag registers over a workload run, freezes the always-0 ones, re-JITs, and
reports the machine-instruction reduction, generic-vs-specialized throughput, and bus-trace
bit-exactness.

Decoder tabulation (§4g.5c) is `crates/rrtl-sim-ir/src/tabulate.rs` (unit test
`tabulated_decode_matches_plain_jit`) with driver `crates/rrtl-sim-ir/examples/picorv32_tabulate.rs`
(`--features jit`, args = cycles and `key_max`): builds the decode ROM, runs picorv32 through the host
mem bus on the plain and tabulated JITs, and asserts a bit-exact bus trace while reporting the
instruction reduction and throughput. `TAB_NODCE=1` keeps the dead decode cone (isolating the rewrite
from the DCE pass). The opportunity sizing is `examples/tabulate_probe.rs` (bit-level cone-support
analysis: per-flag and joint decode keys, and the cones that are *not* tabulatable).

Partitioned compilation (§4g.9) is `crates/rrtl-sim-ir/examples/partitioned_jit.rs` (`--features jit`,
arg = number of PEs `K`): generates a large memory-free design, slices it one-cone-per-PE, compiles
monolithic vs partitioned-parallel (rayon) vs incremental, and runs the partitioned JIT with
register-stable boundary exchange bit-exact against the monolithic JIT.

Activity-driven skipping (§4g.10) is `crates/rrtl-sim-ir/examples/activity_jit.rs` (`--features jit`,
arg = number of PEs): a geometric-activity-gradient design, partitioned, run with and without
per-partition fixpoint skipping (settling-bound + once-verified eligibility), reporting skip rate and
speedup, bit-exact against the monolithic JIT.

The zero-copy shared-state partitioned JIT (§4g.11) is `crates/rrtl-sim-ir/examples/zerocopy_jit.rs`
(`--features jit`, args = stages, groups): a flat pipeline partitioned by register cones, each cone compiled
over the global layout with `JitSimulator::compile_db` and run via `tick_db(cur, nxt)` on shared buffers with
no boundary exchange, bit-exact against the monolithic JIT. `examples/tick_db_check.rs` validates the
double-buffered tick itself against the single-buffer tick. `examples/parallel_jit.rs` (§4g.12) runs the
partitions across cores (one persistent thread each, barrier per cycle, `tick_db_raw`/`tick_db_fn_ptr`),
bit-exact, and reports the per-cycle barrier wall. `examples/batch_jit.rs` (§4g.13) is the bulk-synchronous
answer — one compiled design, M per-instance state buffers run to completion via `par_iter_mut`
(`tick_many_fn_ptr`), reporting near-linear core scaling, bit-exact. `examples/worksteal_jit.rs` (§4g.14)
adds an adversarial skewed workload and compares static chunking vs an atomic-counter work-stealing schedule;
`examples/divergent_batch.rs` does the same on genuine data-driven divergence (`bench/sv/workload.sv`, a
per-instance countdown with chunked early-exit on the counter register).

Observability slicing (§4j) is `cone_of_influence` + `slice_present` (crate root), exercised by
`examples/observe_slice.rs` (`--features jit`): 16 pipelines, observe one, reporting the prune fraction,
instruction reduction, and JIT speedup, bit-exact on the observed register; unit-tested by
`cone_of_influence_prunes_unobserved_logic`.

The optimization policy layer is `crates/rrtl-sim-ir/src/policy.rs`, exercised by
`examples/optimization_policy.rs`: it maps a goal plus optional observations to a conservative
backend-aware pass stack, applies packed-level rewrites, lowers to machine IR, applies machine-level
rewrites, and returns pass reports plus checked memory initializers for generated ROMs such as
decode-tabulation tables. Unit coverage is under `policy::tests`.

Sparse memory (§4k) is `examples/sparse_mem.rs` (`--features jit`, arg = `picorv32.v`): a demand-paged
host memory runs picorv32 with code at 0 and stack near 1 GB, reporting pages allocated vs the dense span,
bit-exact result.

Methodology: warm-up ticks, then best-of-3 timed runs, with cooldown between
configurations to avoid thermal/measurement noise (a lesson learned the hard way — an
early 4.35× claim turned out to be noise).
