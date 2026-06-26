# RTL Authoring

Typed RTL construction, the macro DSL, VCD tracing, and assertions. See
[`../README.md`](../README.md) for the overview.

## Current capabilities

- Strict explicit bit types for signals, literals, expressions, assignments,
  register next values, memory ports, and instance connections.
- Signed and unsigned types through `uint(width)` and `sint(width)`, with
  explicit `lit_u`, `lit_s`, `zext`, `sext`, `trunc`, `as_uint`, and `as_sint`.
- Enum-like state encodings through `state_type`, `state_reg`, `state_next`,
  `state_next_hold`, and `StateReg::is`.
- Structured bundles for ports and wires through `bundle_type`, `field`,
  `nested`, `input_bundle`, `output_bundle`, `wire_bundle`, field/path lookup,
  aggregate bundle assignment, and exact-match bundle instance connections. The
  `rrtl::bundle!` macro can generate the same `BundleType` values from a
  compact Rust DSL.
- Typed interfaces for reusable multi-port contracts through `interface_type`,
  `iface_input`, `iface_output`, `ModuleBuilder::interface`, scalar/bundle port
  lookup, and exact-match interface instance connections. The
  `rrtl::interface!` macro can generate the same `InterfaceType` values from a
  compact Rust DSL.
- Ready/valid channel contract helpers through `rv_source`, `rv_sink`,
  `rv_scalar`, `rv_bundle`, builder channel constructors, `ReadyValidRef`, and
  forward register-slice, one-entry skid-buffer, and register-backed FIFO
  helpers, plus memory-backed FIFO generation.
- Continuous assignments, registers with synchronous or asynchronous,
  active-high or active-low reset behavior, simple memories with combinational
  reads and synchronous writes, and module instances. The macro DSL covers
  common declarations, logic statements, state transitions, memory access, and
  instance, ready/valid, and external-module boilerplate.
- External module declarations for vendor primitives and blackbox IP
  instantiation, including scalar bidirectional `inout` ports for pad-level
  connections.
- Deterministic cycle simulation for a validated design.
- Experimental packed simulation IR and `wgpu` batched GPU simulation for
  accelerated multi-lane simulation research.
- Distributed heterogeneous runtime topology for routing huge independent lane
  batches across packed CPU and GPU workers while preserving global lane order.
- Hierarchy-aware runtime partition planning, launch/deployment payloads,
  structural worker actions, scripted partition-session runs, and JSON
  telemetry for huge single-design split orchestration.
- VCD waveform tracing for simulator sessions through explicit `VcdTrace`
  samples.
- Design assertions and cover points for simulation verification and
  SystemVerilog assertion/cover emission.
- A compiler pipeline via `rrtl_core::compile` / `Design::compile` that
  validates and normalizes authoring IR before backend emission.
- SystemVerilog emission for the supported RTL subset, including
  typed enum emission for state helpers and `rrtl_sv::emit_compiled` for
  already-compiled designs.
- Pretty JSON serialization for compiled designs through
  `CompiledDesign::to_json_pretty()`.
- PyRTL migration bridge for importing elaborated PyRTL `Block` netlists into
  native RRTL IR, including co-simulation-friendly register and memory initial
  values.

## VCD Waveform Tracing

Simulator sessions can be sampled into deterministic VCD output for waveform
viewers:

```rust
use rrtl_core::{Simulator, VcdTrace};

let mut sim = Simulator::new(&design, "Counter").unwrap();
let mut trace = VcdTrace::new(&design, "Counter").unwrap();

trace.sample(&sim, 0).unwrap();
sim.tick();
trace.sample(&sim, 1).unwrap();

let vcd = trace.finish();
println!("{vcd}");
```

Tracing records scalar signals across the simulated instance hierarchy,
including flattened bundle/interface fields. Memory words can be included
explicitly when debugging register files and FIFOs:

```rust
use rrtl_core::{VcdMemoryTrace, VcdTraceOptions};

let mut trace = VcdTrace::with_options(
    &design,
    "Counter",
    VcdTraceOptions {
        memories: VcdMemoryTrace::All,
    },
)
.unwrap();
```

## Bundle Macro

Manual bundle construction remains the canonical runtime API:

```rust
use rrtl::{bundle_type, field, nested, uint};

let req_ty = bundle_type(
    "Req",
    [
        field("valid", uint(1)),
        field("addr", uint(8)),
        nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
    ],
);
```

The procedural macro removes the repetitive constructor calls while producing
the same value:

```rust
use rrtl::bundle;

let req_ty = bundle! {
    Req {
        valid: uint(1),
        addr: uint(8),
        meta: ReqMeta {
            tag: uint(4),
        },
    }
};
```

The first macro slice supports nested bundle blocks and `uint(width)` /
`sint(width)` leaf fields.

Bundles can be assigned leaf-for-leaf when their paths and types match:

```rust
let req = m.input_bundle("req", req_ty.clone());
let pipe = m.wire_bundle("pipe", req_ty.clone());
let resp = m.output_bundle("resp", req_ty);

m.assign_bundle(&pipe, &req);
m.assign_bundle_when(&resp, accepted, &pipe);
```

## Interface Macro

Manual interface construction remains available:

```rust
use rrtl::{iface_input, iface_output, interface_type, uint};

let bus_ty = interface_type(
    "Bus",
    [
        iface_input("req", req_ty),
        iface_output("resp", resp_ty),
        iface_input("ready", uint(1)),
    ],
);
```

The companion macro supports scalar ports and inline bundle ports:

```rust
use rrtl::interface;

let bus_ty = interface! {
    Bus {
        input req: Req {
            valid: uint(1),
            addr: uint(8),
            meta: ReqMeta {
                tag: uint(4),
            },
        },
        output resp: Resp {
            valid: uint(1),
            data: uint(8),
        },
        input ready: uint(1),
    }
};
```

Interface handles expose scalar and bundle leaves directly:

```rust
let bus = m.interface("bus", bus_ty);
let req_valid = bus.field("req", "valid").unwrap();
let req_tag = bus.path("req", ["meta", "tag"]).unwrap();
let ready = bus.signal("ready").unwrap();
```

## Signals Macro

Manual module signal declarations remain available:

```rust
let clk = m.input("clk", uint(1));
let rst = m.input("rst", uint(1));
let en = m.input("en", uint(1));
let out = m.output("out", uint(8));
let count = m.reg("count", uint(8));
```

The `signals!` macro keeps binding explicit while removing repeated builder
calls:

```rust
use rrtl::signals;

let (clk, rst, en, out, count) = signals! {
    m {
        input clk: uint(1),
        input rst: uint(1),
        input en: uint(1),
        output out: uint(8),
        reg count: uint(8),
    }
};
```

It supports scalar `input`, `output`, `wire`, and `reg` declarations, plus
`input_bundle`, `output_bundle`, `wire_bundle`, inline `interface`, `mem`,
`rv_sink`, and `rv_source` declarations.

## Logic Macro

Manual module body statements remain available:

```rust
m.clock(count, clk);
m.reset(count, rst, 0);
m.next(count, mux(en, count + lit_u(1, 8), count));
m.assign(out, count);
```

The `logic!` macro groups common builder statements while keeping expressions
as normal Rust:

```rust
use rrtl::logic;

logic! {
    m {
        clock count: clk;
        reset count: rst = 0;
        next count = mux(en, count + lit_u(1, 8), count);
        assign out = count;
    }
}
```

It supports `assign`, `clock`, `reset`, `reset_low`, `async_reset`,
`async_reset_low`, `next`, `state_next_hold`, `state_next`, `mem_write`,
`assign_bundle`, `assign_bundle_when`, `assert`, `assert_when`,
`assert_clocked`, `assert_msg`, `cover`, `cover_when`, `cover_clocked`, and
`cover_msg`.

## Design Assertions And Covers

Assertions encode simulation and generated-HDL invariants. Cover points count
interesting states or events without failing simulation:

```rust
m.assert("count_small", count.value().lt_expr(lit_u(10, 8)));
m.assert_when("accepted_small", accepted, count.value().lt_expr(lit_u(10, 8)));
m.assert_clocked("count_checked", clk, count.value().lt_expr(lit_u(10, 8)));
m.cover("done_seen", done);
m.cover_when("accepted_done", accepted, done);
m.cover_clocked("done_clocked", clk, done);

let mut sim = Simulator::new(&design, "Counter").unwrap();
sim.check_assertions().unwrap();
sim.tick_checked().unwrap();
assert_eq!(sim.cover_hits("Counter.done_seen"), 1);
```

The `logic!` macro has matching assertion and cover forms:

```rust
logic! {
    m {
        assert count_small: count.value().lt_expr(lit_u(10, 8));
        assert_when accepted_small: accepted, count.value().lt_expr(lit_u(10, 8));
        assert_clocked count_checked: clk, count.value().lt_expr(lit_u(10, 8));
        assert_msg count_message: count.value().lt_expr(lit_u(10, 8)), "count too large";
        cover done_seen: done;
        cover_when accepted_done: accepted, done;
        cover_clocked done_clocked: clk, done;
        cover_msg done_message: done, "done state reached";
    }
}
```

## Ready/Valid Macro

Manual ready/valid channels and helpers remain available:

```rust
let input = m.rv_sink("in", rv_scalar(uint(8)));
let output = m.rv_source("out", rv_scalar(uint(8)));

// Choose one passthrough/buffering helper for a pair of endpoints.
m.rv_connect(&input, &output);
// m.rv_register_slice("slice", &input, &output, clk, rst);
```

The macro form keeps channel handles explicit while removing repeated payload
and helper boilerplate:

```rust
use rrtl::{ready_valid, signals};

let (clk, rst, input, output) = signals! {
    m {
        input clk: uint(1),
        input rst: uint(1),
        rv_sink input: scalar uint(8),
        rv_source output: scalar uint(8),
    }
};

ready_valid! {
    m {
        // Choose one passthrough/buffering helper for a pair of endpoints.
        connect input => output;
        // register_slice slice: input => output, clk, rst;
        // skid_buffer skid: input => output, clk, rst;
        // fifo fifo: input => output, clk, rst, depth 3;
        // mem_fifo mem_fifo: input => output, clk, rst, depth 4;
    }
}
```

Ready/valid declarations also support bundle payloads:

```rust
rv_sink input: bundle Payload {
    data: uint(8),
    last: uint(1),
},
```

## Memory Macro

Manual memory construction remains available:

```rust
let regs = m.mem("regs", 2, uint(8), 4);
m.mem_write(regs, clk, we, waddr, wdata);
m.assign(rdata, m.mem_read(regs, raddr));
```

The macro form keeps reads usable as normal expressions:

```rust
use rrtl::{logic, signals};

let (clk, we, waddr, wdata, raddr, rdata, regs) = signals! {
    m {
        input clk: uint(1),
        input we: uint(1),
        input waddr: uint(2),
        input wdata: uint(8),
        input raddr: uint(2),
        output rdata: uint(8),
        mem regs: addr(2), data uint(8), depth 4,
    }
};

logic! {
    m {
        mem_write regs: clk, we, waddr, wdata;
        assign rdata = rrtl::mem_read!(m, regs, raddr);
    }
}
```

## Instances Macro

Manual instance construction remains available:

```rust
m.instance("u_child", "Child", [("a", a), ("y", y)]);
m.instance_bundles("u_responder", "Responder", [("req", req), ("resp", resp)]);
m.instance_interfaces("u_bus", "BusResponder", [("bus", bus)]);
```

The `instances!` macro removes repeated string tuples for scalar, bundle, and
interface connections:

```rust
use rrtl::instances;

instances! {
    m {
        instance u_child: Child {
            a: a,
            y: y,
        },
        instance_bundles u_responder: Responder {
            req: req,
            resp: resp,
        },
        instance_interfaces u_bus: BusResponder {
            bus: bus,
        },
    }
}
```

## External Module Macro

Manual external module declarations remain available:

```rust
let mut ext = design.extern_module("SB_PLL40_CORE");
ext.input("REFERENCECLK", uint(1));
ext.input("RESETB", uint(1));
ext.output("PLLOUTCORE", uint(1));
```

The `extern_module!` macro declares vendor and blackbox module ports without
repeated string calls:

```rust
use rrtl::extern_module;

extern_module! {
    design SB_PLL40_CORE {
        input REFERENCECLK: uint(1),
        input RESETB: uint(1),
        output PLLOUTCORE: uint(1),
    }
}

extern_module! {
    design IOBUF {
        inout PAD: uint(1),
        input I: uint(1),
        input T: uint(1),
        output O: uint(1),
    }
}
```

## State Macro

Manual state type construction remains available:

```rust
let states = state_type(
    "ControllerState",
    uint(2),
    [("Idle", 0), ("Run", 1), ("Done", 2)],
);
```

The `state!` macro removes the string tuple boilerplate:

```rust
use rrtl::state;

let states = state! {
    ControllerState: uint(2) {
        Idle = 0,
        Run = 1,
        Done = 2,
    }
};
```

State transitions can be grouped in `logic!`:

```rust
logic! {
    m {
        state_next_hold flow {
            start.value() => Run,
            done.value() => Done,
        };
        assign busy = flow.is("Run");
    }
}
```

## Current limitations

- Memory reads outside declared depth simulate as zero, and writes outside
  declared depth are ignored.
- Bidirectional electrical behavior and internal tri-state simulation are not
  modeled yet.
