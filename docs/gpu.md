# Experimental GPU Batch Simulation

See [`simulation.md`](simulation.md) for where the GPU path fits among the
simulation backends.


`rrtl-gpu-sim` is a research crate for running many independent lanes of the
same design on a GPU through `wgpu`. It consumes the `rrtl-sim-ir` packed
simulation IR, which flattens hierarchy, lays out scalar state as `u32` limbs,
and schedules combinational and tick work into packet streams before shader
generation:

```rust
use rrtl_gpu_sim::GpuBatchSimulator;

let mut gpu = GpuBatchSimulator::new(&design, "Counter", 1024).unwrap();
gpu.set_input(en, &vec![1; 1024]).unwrap();
gpu.tick().unwrap();
let counts = gpu.get_signal(count).unwrap();
```

Signals wider than 32 bits use explicit limb APIs:

```rust
gpu.set_input_limbs(wide_in, &[vec![0xffff_fffe, 0xff]; 1024]).unwrap();
let wide_values = gpu.get_signal_limbs(wide_out).unwrap();
```

The current experimental GPU path supports flattened non-external hierarchy,
signed and unsigned scalar values represented as little-endian `u32` limbs,
concat/slice/cast/extension expressions, registers with sync or async resets,
and per-lane memories with combinational reads and synchronous writes. It still
rejects external modules, inout ports, assertions, and cover points for GPU
execution with explicit diagnostics. Bundle and interface signals work through
their flattened scalar leaves.

The packed IR also has a CPU-side `PackedSimulator` for differential testing
and backend development. It lowers expressions into a lower-level
`PackedMachineProgram` with reusable SSA-like values and packeted effects
before interpretation or GPU shader emission. A lightweight benchmark example
compares scalar CPU lanes, packed interpretation, and GPU batch execution:

```sh
cargo run -p rrtl-gpu-sim --example batch_bench --release
```

The same benchmark can sweep scheduler and shader layout knobs, rank each
configuration, and export the best recommendation for each benchmark
`(case, lanes, steps)` tuple:

```sh
cargo run -p rrtl-gpu-sim --example batch_bench --release -- \
  --quick \
  --autotune \
  --autotune-metric gpu_tick_many \
  --format human

cargo run -p rrtl-gpu-sim --example batch_bench --release -- \
  --quick \
  --autotune \
  --recommend-config \
  --output /tmp/rrtl-gpu-recommend.json
```

On machines without a usable GPU adapter, GPU timing fields are unavailable and
autotune ranking falls back to packed CPU timing. The recommendation JSON still
captures the selected scheduling cap, memory read cap, liveness priority,
temporary reuse, memory layout, workgroup size, and timing fields.

Runtime code can load the recommendation file and construct a simulator with
the matching options:

```rust
use std::fs::File;

use rrtl_gpu_sim::{
    load_gpu_autotune_recommendations_report, GpuBatchSimulator,
};

let recommendations =
    load_gpu_autotune_recommendations_report(File::open("/tmp/rrtl-gpu-recommend.json").unwrap())
        .unwrap();
let mut gpu = GpuBatchSimulator::new_with_autotune_recommendations(
    &design,
    "CounterBench",
    "counter",
    64,
    16,
    &recommendations,
)
.unwrap();
```

Recommendation lookup is exact on `(case, lanes, steps)`. JSON parse failures
use `E_GPU_AUTOTUNE_JSON`, missing exact matches use
`E_GPU_AUTOTUNE_RECOMMENDATION`, and invalid memory layout values use
`E_GPU_AUTOTUNE_LAYOUT`.
