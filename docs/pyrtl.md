# PyRTL Bridge

Import elaborated PyRTL `Block` netlists into native RRTL IR, migrate large
source trees, and benchmark against PyRTL `FastSimulation`.

## PyRTL Import

Existing PyRTL source does not need to be rewritten. Elaborate the design as a
normal PyRTL `Block`, export it with the Python helper, then import or emit it
with the Rust CLI:

```sh
PYTHONPATH=python python -m rrtl_pyrtl my_design:build --top-name MyTop --out /tmp/mytop.pyrtl.json
cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- check /tmp/mytop.pyrtl.json
cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- sv /tmp/mytop.pyrtl.json
cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- compare /tmp/mytop.pyrtl.json /tmp/mytop.trace.json
```

The bridge targets PyRTL's normalized `LogicNet` block representation rather
than Python source. PyRTL's implicit sequential domain is imported as an
explicit 1-bit `clk` input by default, with register reset values and ROM
contents preserved as simulation/SystemVerilog initial values.
The Python helper also exposes `simulate_trace(input_vectors)` for creating
trace JSON that the `compare` command replays against the imported RRTL design.

## PyRTL Corpus Migration

For larger migrations, run a corpus manifest and keep the per-design artifacts:

```json
{
  "targets": [
    {
      "name": "counter",
      "target": "my_designs.counter:build",
      "top_name": "Counter",
      "inputs": [{ "en": 1 }, { "en": 0 }]
    }
  ]
}
```

Use `python/rrtl_pyrtl/corpus_template.json` as a starting point for private
manifests. Keep the filled manifest next to the private PyRTL sources, add those
sources to `PYTHONPATH`, and validate the manifest before running the full
bridge:

```sh
PYTHONPATH=python:/path/to/private/sources \
  python -m rrtl_pyrtl.corpus --validate-only /path/to/corpus.json
```

For a large existing source tree, generate a draft manifest with the static
discovery scanner first. Discovery parses Python source without importing or
executing private modules, then writes both a draft JSON manifest and a Markdown
candidate report:

```sh
PYTHONPATH=python python -m rrtl_pyrtl.discover /path/to/private/sources \
  --package-root /path/to/private/sources \
  --out target/rrtl-pyrtl-corpus/discovered.json \
  --report target/rrtl-pyrtl-corpus/discovered.md
```

Review the report, edit the draft manifest as needed, then validate it with
`rrtl_pyrtl.corpus --validate-only`.

To generate discovery artifacts and run manifest validation in one safe step,
use the prepare wrapper. It still does not execute PyRTL builders:

```sh
PYTHONPATH=python:/path/to/private/sources \
  python -m rrtl_pyrtl.prepare_corpus /path/to/private/sources \
  --package-root /path/to/private/sources \
  --out-dir target/rrtl-pyrtl-corpus/private
```

This writes `discovered.json`, `discovered.md`, `validation.json`, and
`validation.md` under the output directory. It also writes `next_commands.md`
with exact commands for re-running prepare, validating the generated manifest,
and running full corpus triage.

For CI or repeated private-tree baselining, use the report-only gate. It runs
discovery, validation, full corpus triage, and high-level gate reporting in one
command. Target import or compare failures are reported in the artifacts but do
not make the gate exit nonzero; manifest/tooling failures still do.

```sh
PYTHONPATH=python \
  python -m rrtl_pyrtl.gate /path/to/private/sources \
  --package-root /path/to/private/sources \
  --out-dir target/rrtl-pyrtl-corpus/private-gate
```

The gate writes `gate.json`, `gate.md`, `summary.json`, `summary.md`,
`validation.json`, and `validation.md`. Pass `--manifest /path/to/corpus.json`
to skip discovery and run against a reviewed manifest.

```sh
PYTHONPATH=python python -m rrtl_pyrtl.corpus corpus.json \
  --emit-sv \
  --emit-json \
  --summary-json target/rrtl-pyrtl-corpus/summary.json \
  --summary-md target/rrtl-pyrtl-corpus/summary.md
```

Each target is exported, checked, optionally emitted as SV/compiled JSON, and
compared against a PyRTL trace when `inputs` are provided. Failures include the
target name and the importer reports the PyRTL net index/op/args/dests for
unsupported constructs. Corpus runs continue after failures by default so the
summary can show the full compatibility surface; pass `--fail-fast` when a local
debugging run should stop at the first failing target. `--validate-only` checks
manifest shape, duplicate target names, target importability, and input-vector
shape without invoking design builders.

A checked-in smoke corpus is available for validating the toolchain itself:

```sh
PYTHONPATH=python python -m rrtl_pyrtl.corpus \
  python/rrtl_pyrtl/corpus_smoke.json \
  --emit-sv \
  --emit-json \
  --summary-json target/rrtl-pyrtl-corpus/summary.json \
  --summary-md target/rrtl-pyrtl-corpus/summary.md
```

The summary JSON uses stable fields for triage automation: `name`, `target`,
`phase`, `ok`, `error`, `bucket`, and `outputs`. It also includes aggregate
totals, phase counts, and failure bucket counts. The Markdown summary is the
human-readable triage report for reviewing large private manifests.

## PyRTL Benchmarking

The bridge includes a synthetic systolic-style PyRTL benchmark for establishing
a first throughput baseline against PyRTL `FastSimulation`. It builds a
repo-local signed MAC array, exports it through the PyRTL bridge, checks RRTL
scalar, packed, single-lane machine, and backend-selected trace replay for
bit-exactness, and writes timing artifacts:

```sh
PYTHONPATH=python python -m rrtl_pyrtl.bench \
  --out-dir target/rrtl-pyrtl-bench/smoke \
  --rows 2 \
  --cols 2 \
  --steps 32 \
  --repeat 2 \
  --packed-lanes 1
```

For an XNN-like single-lane baseline, use a larger array and longer stream:

```sh
PYTHONPATH=python python -m rrtl_pyrtl.bench \
  --out-dir target/rrtl-pyrtl-bench/xnn-r1 \
  --rows 8 \
  --cols 8 \
  --steps 512 \
  --repeat 5 \
  --warmup 1
```

The benchmark writes `bench.json`, `bench.md`, the exported PyRTL JSON, and the
trace JSON. Timing is report-only for now; correctness mismatches fail. The
single-lane machine row is a scalar CPU baseline and correctness oracle. The
backend-selected report exposes the shared scalar, packed CPU, SIMD CPU, and JIT
CPU backend surface; SIMD currently shares the packed CPU execution path while
the dedicated SIMD kernels are filled in, and JIT reports unavailable until
Cranelift codegen is implemented behind the `jit` feature. This is the runtime
architecture hook for the intended Apple Silicon/NVGPU path, not the final
performance endpoint. The
underlying Rust replay timers are also available directly:

```sh
cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- bench-trace \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.pyrtl.json \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.trace.json \
  --repeat 5 \
  --warmup 1

cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- bench-packed-trace \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.pyrtl.json \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.trace.json \
  --repeat 5 \
  --warmup 1 \
  --lanes 1

cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- bench-single-trace \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.pyrtl.json \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.trace.json \
  --repeat 5 \
  --warmup 1

cargo run -p rrtl-pyrtl --bin pyrtl2rrtl -- bench-backends \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.pyrtl.json \
  target/rrtl-pyrtl-bench/smoke/systolic_mac.trace.json \
  --backend scalar,packed-cpu,simd-cpu,jit-cpu \
  --repeat 5 \
  --warmup 1 \
  --lanes 4
```
