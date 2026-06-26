    use super::*;
    use rrtl_core::{
        bundle_type, compile, field, iface_input, iface_output, interface_type, lit_s,
        lit_u as lit, nested, rv_bundle, rv_scalar, sint, uint, zext, Design, Signal, SignalKind,
        Simulator,
    };

    fn signal(imported: &SvImport, module: &str, name: &str) -> Signal {
        let compiled = compile(&imported.design).unwrap();
        compiled
            .find_module(module)
            .unwrap()
            .signals
            .iter()
            .find(|signal| signal.name == name)
            .unwrap()
            .handle
    }

    fn reimport_emitted(design: &Design, top: &str) -> SvImport {
        let sv = rrtl_sv::emit(design).unwrap();
        import_sv(&sv, Some(top)).unwrap_or_else(|err| panic!("failed to reimport:\n{sv}\n{err}"))
    }

    #[test]
    fn imports_combinational_alu() {
        let imported = import_sv(
            r#"
            module Alu(a, b, sum, eq);
              input logic [7:0] a, b;
              output logic [7:0] sum;
              output logic eq;
              assign sum = a + b;
              assign eq = a == b;
            endmodule
            "#,
            None,
        )
        .unwrap();
        assert_eq!(imported.top_name, "Alu");
        let mut sim = Simulator::new(&imported.design, "Alu").unwrap();
        let a = signal(&imported, "Alu", "a");
        let b = signal(&imported, "Alu", "b");
        let sum = signal(&imported, "Alu", "sum");
        let eq = signal(&imported, "Alu", "eq");
        sim.set(a, 3);
        sim.set(b, 4);
        assert_eq!(sim.get(sum), 7);
        assert_eq!(sim.get(eq), 0);
    }

    #[test]
    fn imports_signed_arithmetic() {
        // Signed compare, arithmetic shift, sign-extension, signed multiply, and
        // signed→unsigned assignment all follow SV signedness rules.
        let imported = import_sv(
            r#"
            module S(s4, a, b, lt, sra, sext, smul, s4u8);
              input  signed [3:0] s4;
              input  signed [7:0] a, b;
              output logic        lt;
              output signed [7:0] sra;
              output signed [15:0] sext, smul;
              output [7:0]        s4u8;
              assign lt   = (a < b);
              assign sra  = a >>> 2;
              assign sext = a;
              assign smul = a * b;
              assign s4u8 = s4;     // sign-extend (RHS signed) into unsigned target
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "S").unwrap();
        let g = |n: &str| signal(&imported, "S", n);
        let (s4, a, b) = (g("s4"), g("a"), g("b"));
        let (lt, sra, sext, smul, s4u8) = (g("lt"), g("sra"), g("sext"), g("smul"), g("s4u8"));
        sim.set(a, 0xFF); // -1
        sim.set(b, 0x01); // 1
        sim.set(s4, 0xF); // -1
        assert_eq!(sim.get(lt), 1); // -1 < 1 (both signed)
        assert_eq!(sim.get(sra) & 0xff, 0xff); // -1 >>> 2 = -1
        assert_eq!(sim.get(sext) & 0xffff, 0xffff); // sign-extend -1
        assert_eq!(sim.get(s4u8) & 0xff, 0xff); // sign-extend -1 into u8
        sim.set(a, 0xFE); // -2
        sim.set(b, 0x03); // 3
        assert_eq!(sim.get(smul) & 0xffff, 0xfffa); // -2 * 3 = -6
    }

    #[test]
    fn imports_2d_memory_read() {
        // A 4x4 unpacked memory loaded row-major (flat[k]=k); mem[i][j] must read
        // flat[i*4 + j].
        let path = std::env::temp_dir().join("rrtl_2d_test.hex");
        let hex: String = (0..16).map(|k| format!("{k:02x}\n")).collect();
        std::fs::write(&path, hex).unwrap();
        let src = format!(
            r#"
            module M(i, j, o);
              input  logic [3:0] i, j;
              output logic [7:0] o;
              logic [7:0] mem [0:3][0:3];
              initial $readmemh("{}", mem);
              assign o = mem[i][j];
            endmodule
            "#,
            path.display()
        );
        let imported = import_sv(&src, Some("M")).unwrap();
        let mut sim = Simulator::new(&imported.design, "M").unwrap();
        let (ii, jj, o) = (signal(&imported, "M", "i"), signal(&imported, "M", "j"), signal(&imported, "M", "o"));
        for (i, j) in [(0u128, 0u128), (1, 2), (3, 3), (2, 1)] {
            sim.set(ii, i);
            sim.set(jj, j);
            assert_eq!(sim.get(o), i * 4 + j, "mem[{i}][{j}]");
        }
    }

    #[test]
    fn imports_interface_bundle() {
        // A producer and consumer wired by an interface bundle with modports.
        // The interface flattens to per-member signals (bus.data/valid/ready),
        // crossing the module boundary via member-wise connection expansion.
        let imported = import_sv(
            r#"
            interface VX_bus_if #(parameter W = 8) ();
              logic [W-1:0] data;
              logic         valid;
              logic         ready;
              modport tx (output data, output valid, input ready);
              modport rx (input data, input valid, output ready);
            endinterface

            module Producer(input logic clk, input logic [7:0] din, input logic dv, VX_bus_if.tx bus);
              always_ff @(posedge clk) begin
                bus.data  <= din;
                bus.valid <= dv;
              end
            endmodule

            module Consumer(input logic clk, VX_bus_if.rx bus, output logic [7:0] dout, output logic got);
              assign bus.ready = 1'b1;
              always_ff @(posedge clk) begin
                dout <= bus.data;
                got  <= bus.valid;
              end
            endmodule

            module Top(input logic clk, input logic [7:0] din, input logic dv,
                       output logic [7:0] dout, output logic got);
              VX_bus_if bus();
              Producer p(.clk(clk), .din(din), .dv(dv), .bus(bus));
              Consumer c(.clk(clk), .bus(bus), .dout(dout), .got(got));
            endmodule
            "#,
            Some("Top"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Top").unwrap();
        let s = |n: &str| signal(&imported, "Top", n);
        let (din, dv, dout, got) = (s("din"), s("dv"), s("dout"), s("got"));
        sim.set(din, 0x42);
        sim.set(dv, 1);
        sim.tick(); // Producer latches din → bus.data
        sim.tick(); // Consumer latches bus.data → dout
        assert_eq!(sim.get(dout), 0x42, "data crossed the interface");
        assert_eq!(sim.get(got), 1, "valid crossed the interface");
        // Drop valid; two cycles later `got` should clear.
        sim.set(dv, 0);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(got), 0, "valid cleared");
    }

    #[test]
    fn imports_package() {
        // A package supplies a localparam (used scoped in an expression and in a
        // type range), an enum (imported, used bare), a scoped type, and a struct
        // type (imported). All must resolve and survive to lowering.
        let imported = import_sv(
            r#"
            package VX_pkg;
              localparam STEP  = 4;
              localparam WIDTH = 8;
              typedef enum logic [1:0] { S_IDLE, S_RUN, S_DONE } state_t;
              typedef struct packed { logic [7:0] tag; logic [7:0] data; } entry_t;
            endpackage

            module Top(input logic clk, input logic go,
                       output logic [7:0] count_out, output logic [1:0] st_out, output logic [7:0] tag_out);
              import VX_pkg::*;
              logic [VX_pkg::WIDTH-1:0] counter;   // scoped param in a range
              VX_pkg::state_t state;               // scoped type
              entry_t e;                           // imported (bare) type
              assign count_out = counter;
              assign st_out    = state;
              assign tag_out   = e.tag;
              always_ff @(posedge clk) begin
                if (go) counter <= counter + VX_pkg::STEP;  // scoped param in expr
                else    counter <= 8'd0;
                state <= S_RUN;                              // imported enum constant
                e.tag <= 8'hAB;
              end
            endmodule
            "#,
            Some("Top"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Top").unwrap();
        let s = |n: &str| signal(&imported, "Top", n);
        let (go, count_out, st_out, tag_out) = (s("go"), s("count_out"), s("st_out"), s("tag_out"));
        sim.set(go, 1);
        sim.tick();
        assert_eq!(sim.get(count_out), 4, "counter += STEP(4)");
        assert_eq!(sim.get(st_out), 1, "state == S_RUN(1)");
        assert_eq!(sim.get(tag_out), 0xAB, "struct member from package type");
        sim.tick();
        assert_eq!(sim.get(count_out), 8, "counter += STEP again");
    }

    #[test]
    fn imports_interface_array() {
        // An array of 3 interface bundles, each driven by its own generated module
        // instance wired with `.lane(lane[i])`; the genvar index folds during
        // generate unrolling. Element member reads `lane[k].data` round-trip.
        let imported = import_sv(
            r#"
            interface VX_lane_if #(parameter W = 8) ();
              logic [W-1:0] data;
              logic         valid;
              modport tx (output data, output valid);
              modport rx (input data, input valid);
            endinterface

            module Gen(input logic clk, input logic [7:0] base, VX_lane_if.tx lane);
              always_ff @(posedge clk) begin
                lane.data  <= base;
                lane.valid <= 1'b1;
              end
            endmodule

            module Top(input logic clk, input logic [7:0] base,
                       output logic [7:0] o0, output logic [7:0] o1, output logic [7:0] o2);
              VX_lane_if lane[3]();
              genvar i;
              generate
                for (i = 0; i < 3; i = i + 1) begin : g
                  Gen u(.clk(clk), .base(base + i), .lane(lane[i]));
                end
              endgenerate
              assign o0 = lane[0].data;
              assign o1 = lane[1].data;
              assign o2 = lane[2].data;
            endmodule
            "#,
            Some("Top"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Top").unwrap();
        let s = |n: &str| signal(&imported, "Top", n);
        let base = s("base");
        let (o0, o1, o2) = (s("o0"), s("o1"), s("o2"));
        sim.set(base, 0x10);
        sim.tick();
        // Each lane registered base + its index.
        assert_eq!(sim.get(o0), 0x10, "lane[0]");
        assert_eq!(sim.get(o1), 0x11, "lane[1]");
        assert_eq!(sim.get(o2), 0x12, "lane[2]");
    }

    #[test]
    fn imports_interface_param_override() {
        // The interface defaults to W=8 but is instantiated with #(.W(16)). The
        // override must reach BOTH the instance's member signals AND each child's
        // interface-port widths (via specialization) — otherwise a 16-bit value
        // would truncate to the 8-bit default as it crosses the bundle.
        let imported = import_sv(
            r#"
            interface VX_bus_if #(parameter W = 8) ();
              logic [W-1:0] data;
              logic         valid;
              logic         ready;
              modport tx (output data, output valid, input ready);
              modport rx (input data, input valid, output ready);
            endinterface

            module Producer(input logic clk, input logic [15:0] din, input logic dv, VX_bus_if.tx bus);
              always_ff @(posedge clk) begin
                bus.data  <= din;
                bus.valid <= dv;
              end
            endmodule

            module Consumer(input logic clk, VX_bus_if.rx bus, output logic [15:0] dout, output logic got);
              assign bus.ready = 1'b1;
              always_ff @(posedge clk) begin
                dout <= bus.data;
                got  <= bus.valid;
              end
            endmodule

            module Top(input logic clk, input logic [15:0] din, input logic dv,
                       output logic [15:0] dout, output logic got);
              VX_bus_if #(.W(16)) bus();
              Producer p(.clk(clk), .din(din), .dv(dv), .bus(bus));
              Consumer c(.clk(clk), .bus(bus), .dout(dout), .got(got));
            endmodule
            "#,
            Some("Top"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Top").unwrap();
        let s = |n: &str| signal(&imported, "Top", n);
        let (din, dv, dout) = (s("din"), s("dv"), s("dout"));
        sim.set(din, 0xBEEF);
        sim.set(dv, 1);
        sim.tick();
        sim.tick();
        // The full 16-bit value survived — the #(.W(16)) override propagated.
        assert_eq!(sim.get(dout), 0xBEEF, "16-bit value crossed the parameterized bundle");
    }

    #[test]
    fn imports_packed_struct() {
        // A packed struct is a flat bit-vector with the first field in the most-
        // significant bits: {hdr[7:0], flags[3:0], v} packs to bits [12:5],[4:1],[0].
        // Drive the whole struct from a concat, then read each field back out —
        // verifying the field→slice offsets (MSB-first) round-trip.
        let imported = import_sv(
            r#"
            module S(a, b, c, packed_out, fa, fb, fc);
              input  logic [7:0] a;
              input  logic [3:0] b;
              input  logic c;
              output logic [12:0] packed_out;
              output logic [7:0] fa;
              output logic [3:0] fb;
              output logic fc;
              typedef struct packed { logic [7:0] hdr; logic [3:0] flags; logic v; } pkt_t;
              pkt_t p;
              assign p = {a, b, c};
              assign packed_out = p;
              assign fa = p.hdr;
              assign fb = p.flags;
              assign fc = p.v;
            endmodule
            "#,
            Some("S"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "S").unwrap();
        let s = |n: &str| signal(&imported, "S", n);
        let (a, b, c) = (s("a"), s("b"), s("c"));
        let (packed_out, fa, fb, fc) = (s("packed_out"), s("fa"), s("fb"), s("fc"));
        for (av, bv, cv) in [(0xA5u128, 0xCu128, 1u128), (0x00, 0x0, 0), (0xFF, 0xF, 1), (0x12, 0x3, 0)] {
            sim.set(a, av);
            sim.set(b, bv);
            sim.set(c, cv);
            // MSB-first packing: hdr<<5 | flags<<1 | v.
            assert_eq!(sim.get(packed_out), (av << 5) | (bv << 1) | cv, "packed {av:#x},{bv:#x},{cv}");
            assert_eq!(sim.get(fa), av, "fa");
            assert_eq!(sim.get(fb), bv, "fb");
            assert_eq!(sim.get(fc), cv, "fc");
        }
    }

    #[test]
    fn imports_packed_struct_register() {
        // A clocked struct register: nonblocking field writes update the slices,
        // a field read observes the registered value one cycle later.
        let imported = import_sv(
            r#"
            module R(clk, din, tag_in, tag_out, data_out);
              input  logic clk;
              input  logic [7:0] din;
              input  logic [3:0] tag_in;
              output logic [3:0] tag_out;
              output logic [7:0] data_out;
              typedef struct packed { logic [3:0] tag; logic [7:0] data; } entry_t;
              entry_t e;
              always_ff @(posedge clk) begin
                e.tag  <= tag_in;
                e.data <= din;
              end
              assign tag_out  = e.tag;
              assign data_out = e.data;
            endmodule
            "#,
            Some("R"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "R").unwrap();
        let s = |n: &str| signal(&imported, "R", n);
        let (din, tag_in, tag_out, data_out) = (s("din"), s("tag_in"), s("tag_out"), s("data_out"));
        sim.set(tag_in, 0x9);
        sim.set(din, 0xC3);
        sim.tick();
        assert_eq!(sim.get(tag_out), 0x9);
        assert_eq!(sim.get(data_out), 0xC3);
        sim.set(tag_in, 0x5);
        sim.set(din, 0x7E);
        sim.tick();
        assert_eq!(sim.get(tag_out), 0x5);
        assert_eq!(sim.get(data_out), 0x7E);
    }

    #[test]
    fn imports_enum_fsm() {
        // An enum typedef supplies the state names (auto-incrementing IDLE=0,
        // RUN=1, DONE=2); the names must resolve like localparams in the case
        // labels, the comparison, and the next-state assignments.
        let imported = import_sv(
            r#"
            module Fsm(clk, go, busy, state_out);
              input  logic clk, go;
              output logic busy;
              output logic [1:0] state_out;
              typedef enum logic [1:0] { IDLE, RUN, DONE } state_t;
              state_t state;
              assign state_out = state;
              assign busy = (state == RUN);
              always_ff @(posedge clk) begin
                case (state)
                  IDLE: if (go) state <= RUN;
                  RUN:  state <= DONE;
                  DONE: state <= IDLE;
                  default: state <= IDLE;
                endcase
              end
            endmodule
            "#,
            Some("Fsm"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Fsm").unwrap();
        let go = signal(&imported, "Fsm", "go");
        let busy = signal(&imported, "Fsm", "busy");
        let state_out = signal(&imported, "Fsm", "state_out");
        // Stays in IDLE(0) while go=0.
        sim.set(go, 0);
        sim.tick();
        assert_eq!(sim.get(state_out), 0, "IDLE held");
        assert_eq!(sim.get(busy), 0);
        // go=1 → IDLE→RUN(1) → DONE(2) → IDLE(0).
        sim.set(go, 1);
        sim.tick();
        assert_eq!(sim.get(state_out), 1, "RUN");
        assert_eq!(sim.get(busy), 1, "busy in RUN");
        sim.tick();
        assert_eq!(sim.get(state_out), 2, "DONE");
        assert_eq!(sim.get(busy), 0);
        sim.tick();
        assert_eq!(sim.get(state_out), 0, "back to IDLE");
    }

    #[test]
    fn imports_enum_explicit_values() {
        // Explicit + omitted values mix: auto-increment resumes from the last
        // explicit value (A=0, B=5, C=6).
        let imported = import_sv(
            r#"
            module E(sel, o);
              input  logic [1:0] sel;
              output logic [7:0] o;
              typedef enum logic [7:0] { A, B = 5, C } code_t;
              assign o = (sel == 0) ? A : (sel == 1) ? B : C;
            endmodule
            "#,
            Some("E"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "E").unwrap();
        let sel = signal(&imported, "E", "sel");
        let o = signal(&imported, "E", "o");
        for (s, want) in [(0u128, 0u128), (1, 5), (2, 6)] {
            sim.set(sel, s);
            assert_eq!(sim.get(o), want, "sel={s}");
        }
    }

    #[test]
    fn imports_2d_memory_write() {
        // Clocked write to mem[i][j] in an always_ff, read back via mem[i][j]:
        // the write must hit the same row-major flat[i*4 + j] the read targets.
        let imported = import_sv(
            r#"
            module M(clk, we, wi, wj, ri, rj, o);
              input  logic clk, we;
              input  logic [3:0] wi, wj, ri, rj;
              output logic [7:0] o;
              logic [7:0] mem [0:3][0:3];
              always_ff @(posedge clk) begin
                if (we) mem[wi][wj] <= {wi, wj};
              end
              assign o = mem[ri][rj];
            endmodule
            "#,
            Some("M"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "M").unwrap();
        let s = |n: &str| signal(&imported, "M", n);
        let (we, wi, wj, ri, rj, o) = (s("we"), s("wi"), s("wj"), s("ri"), s("rj"), s("o"));
        // Write a distinct value {i,j} into each of the 16 cells.
        sim.set(we, 1);
        for i in 0u128..4 {
            for j in 0u128..4 {
                sim.set(wi, i);
                sim.set(wj, j);
                sim.tick();
            }
        }
        sim.set(we, 0);
        // Read every cell back: mem[i][j] == (i<<4)|j.
        for i in 0u128..4 {
            for j in 0u128..4 {
                sim.set(ri, i);
                sim.set(rj, j);
                assert_eq!(sim.get(o), (i << 4) | j, "mem[{i}][{j}]");
            }
        }
    }

    #[test]
    fn imports_readmemh() {
        // $readmemh loads a memory from a hex file at lowering time, honoring
        // `@address` directives, `//` comments, and `_` digit separators.
        let path = std::env::temp_dir().join("rrtl_readmemh_test.hex");
        std::fs::write(&path, "0a 14 // first two\nFF\n@10\n9_9\n").unwrap();
        let src = format!(
            r#"
            module M(addr, o);
              input  logic [7:0] addr;
              output logic [7:0] o;
              logic [7:0] mem [0:255];
              initial $readmemh("{}", mem);
              assign o = mem[addr];
            endmodule
            "#,
            path.display()
        );
        let imported = import_sv(&src, Some("M")).unwrap();
        let mut sim = Simulator::new(&imported.design, "M").unwrap();
        let addr = signal(&imported, "M", "addr");
        let o = signal(&imported, "M", "o");
        for (a, want) in [(0u128, 0x0a), (1, 0x14), (2, 0xff), (16, 0x99), (3, 0x00)] {
            sim.set(addr, a);
            assert_eq!(sim.get(o), want, "mem[{a}]");
        }
    }

    #[test]
    fn imports_function_calls() {
        // Two functions — one with a ternary return, one with a local variable —
        // each inline-expanded at the continuous-assign call sites.
        let imported = import_sv(
            r#"
            module FnTest(a, b, mx, sumw);
              input  logic [7:0] a, b;
              output logic [7:0] mx;
              output logic [8:0] sumw;
              function [7:0] maxv(input [7:0] x, input [7:0] y);
                maxv = (x > y) ? x : y;
              endfunction
              function [8:0] add_local(input [7:0] x, input [7:0] y);
                logic [8:0] s;
                s = x + y;
                return s;
              endfunction
              assign mx   = maxv(a, b);
              assign sumw = add_local(a, b);
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FnTest").unwrap();
        let a = signal(&imported, "FnTest", "a");
        let b = signal(&imported, "FnTest", "b");
        let mx = signal(&imported, "FnTest", "mx");
        let sumw = signal(&imported, "FnTest", "sumw");
        sim.set(a, 5);
        sim.set(b, 9);
        assert_eq!(sim.get(mx), 9);
        assert_eq!(sim.get(sumw), 14);
        sim.set(a, 200);
        sim.set(b, 100);
        assert_eq!(sim.get(mx), 200);
        assert_eq!(sim.get(sumw), 300); // 9-bit, no overflow
    }

    #[test]
    fn imports_counter_with_async_active_low_reset() {
        let imported = import_sv(
            r#"
            module Counter(clk, rst_n, en, out);
              input logic clk, rst_n, en;
              output logic [3:0] out;
              logic [3:0] count;
              assign out = count;
              always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) count <= 4'd0;
                else if (en) count <= count + 4'd1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Counter").unwrap();
        let rst_n = signal(&imported, "Counter", "rst_n");
        let en = signal(&imported, "Counter", "en");
        let out = signal(&imported, "Counter", "out");
        sim.set(rst_n, 0);
        sim.tick();
        sim.set(rst_n, 1);
        sim.set(en, 1);
        sim.tick();
        assert_eq!(sim.get(out), 1);
    }

    #[test]
    fn imports_simple_memory_write_read() {
        let imported = import_sv(
            r#"
            module Mem(clk, we, waddr, wdata, raddr, rdata);
              input logic clk, we;
              input logic [1:0] waddr, raddr;
              input logic [7:0] wdata;
              output logic [7:0] rdata;
              logic [7:0] regs [0:3];
              assign rdata = regs[raddr];
              always_ff @(posedge clk) begin
                if (we) regs[waddr] <= wdata;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Mem").unwrap();
        let we = signal(&imported, "Mem", "we");
        let waddr = signal(&imported, "Mem", "waddr");
        let wdata = signal(&imported, "Mem", "wdata");
        let raddr = signal(&imported, "Mem", "raddr");
        let rdata = signal(&imported, "Mem", "rdata");
        sim.set(we, 1);
        sim.set(waddr, 2);
        sim.set(wdata, 0xab);
        sim.tick();
        sim.set(we, 0);
        sim.set(raddr, 2);
        assert_eq!(sim.get(rdata), 0xab);
    }

    #[test]
    fn imports_ansi_port_alu() {
        let imported = import_sv(
            r#"
            module AnsiAlu(
              input logic [7:0] a, b,
              output logic [7:0] sum,
              output logic eq
            );
              assign sum = a + b;
              assign eq = a == b;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "AnsiAlu").unwrap();
        let a = signal(&imported, "AnsiAlu", "a");
        let b = signal(&imported, "AnsiAlu", "b");
        let sum = signal(&imported, "AnsiAlu", "sum");
        let eq = signal(&imported, "AnsiAlu", "eq");
        sim.set(a, 9);
        sim.set(b, 9);
        assert_eq!(sim.get(sum), 18);
        assert_eq!(sim.get(eq), 1);
    }

    #[test]
    fn imports_parameterized_width_counter() {
        let imported = import_sv(
            r#"
            module ParamCounter #(parameter int WIDTH = 4) (
              input logic clk,
              input logic en,
              output logic [WIDTH-1:0] out
            );
              logic [WIDTH-1:0] count;
              assign out = count;
              always_ff @(posedge clk) begin
                if (en) count <= count + 4'd1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ParamCounter").unwrap();
        let en = signal(&imported, "ParamCounter", "en");
        let out = signal(&imported, "ParamCounter", "out");
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(out), 2);
    }

    #[test]
    fn imports_parameterized_instance_overrides_with_multiple_specializations() {
        let imported = import_sv(
            r#"
            module WidthChild #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module WidthTop(
              input logic [3:0] a4,
              input logic [7:0] a8,
              output logic [3:0] y4,
              output logic [7:0] y8
            );
              WidthChild #(.WIDTH(4)) u4(.a(a4), .y(y4));
              WidthChild #(.WIDTH(8)) u8(.a(a8), .y(y8));
            endmodule
            "#,
            Some("WidthTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("WidthChild__WIDTH_4").is_some());
        assert!(compiled.find_module("WidthChild__WIDTH_8").is_some());

        let mut sim = Simulator::new(&imported.design, "WidthTop").unwrap();
        let a4 = signal(&imported, "WidthTop", "a4");
        let a8 = signal(&imported, "WidthTop", "a8");
        let y4 = signal(&imported, "WidthTop", "y4");
        let y8 = signal(&imported, "WidthTop", "y8");
        sim.set(a4, 3);
        sim.set(a8, 3);
        assert_eq!(sim.get(y4), 12);
        assert_eq!(sim.get(y8), 252);
    }

    #[test]
    fn imports_nested_parameterized_instance_overrides() {
        let imported = import_sv(
            r#"
            module Leaf #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module Mid #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              Leaf #(.WIDTH(WIDTH)) u_leaf(.a(a), .y(y));
            endmodule

            module NestedTop(input logic [5:0] a, output logic [5:0] y);
              Mid #(.WIDTH(6)) u_mid(.a(a), .y(y));
            endmodule
            "#,
            Some("NestedTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Mid__WIDTH_6").is_some());
        assert!(compiled.find_module("Leaf__WIDTH_6").is_some());

        let mut sim = Simulator::new(&imported.design, "NestedTop").unwrap();
        let a = signal(&imported, "NestedTop", "a");
        let y = signal(&imported, "NestedTop", "y");
        sim.set(a, 10);
        assert_eq!(sim.get(y), 53);
    }

    #[test]
    fn imports_parameterized_instance_override_from_parent_localparam() {
        let imported = import_sv(
            r#"
            module LocalWidth #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module LocalTop(input logic [7:0] a, output logic [7:0] y);
              localparam int BASE = 4;
              LocalWidth #(.WIDTH(BASE * 2)) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("LocalTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("LocalWidth__WIDTH_8").is_some());

        let mut sim = Simulator::new(&imported.design, "LocalTop").unwrap();
        let a = signal(&imported, "LocalTop", "a");
        let y = signal(&imported, "LocalTop", "y");
        sim.set(a, 2);
        assert_eq!(sim.get(y), 253);
    }

    #[test]
    fn imports_external_instance_parameter_overrides_as_metadata() {
        let imported = import_sv(
            r#"
            module ExtParamTop(input logic [7:0] a, output logic [7:0] y);
              Vendor #(.WIDTH(8)) u_vendor(.a(a), .y(y));
            endmodule
            "#,
            Some("ExtParamTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Vendor").is_some());
        assert!(compiled.find_module("Vendor__WIDTH_8").is_none());
    }

    #[test]
    fn imports_positional_instance_parameter_override() {
        let imported = import_sv(
            r#"
            module Child #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module PosTop(input logic [7:0] a, output logic [7:0] y);
              Child #(8) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("PosTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Child__WIDTH_8").is_some());
        let mut sim = Simulator::new(&imported.design, "PosTop").unwrap();
        let a = signal(&imported, "PosTop", "a");
        let y = signal(&imported, "PosTop", "y");
        sim.set(a, 0xab);
        assert_eq!(sim.get(y), 0xab);
    }

    #[test]
    fn imports_multiple_positional_instance_parameter_overrides() {
        let imported = import_sv(
            r#"
            module SliceChild #(parameter int WIDTH = 4, parameter int MASK = 0) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ MASK[WIDTH-1:0];
            endmodule

            module PosMultiTop(input logic [7:0] a, output logic [7:0] y);
              SliceChild #(8, 8'hff) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("PosMultiTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled
            .find_module("SliceChild__MASK_255__WIDTH_8")
            .is_some());
        let mut sim = Simulator::new(&imported.design, "PosMultiTop").unwrap();
        let a = signal(&imported, "PosMultiTop", "a");
        let y = signal(&imported, "PosMultiTop", "y");
        sim.set(a, 0x03);
        assert_eq!(sim.get(y), 0xfc);
    }

    #[test]
    fn imports_nested_positional_instance_parameter_overrides() {
        let imported = import_sv(
            r#"
            module Leaf #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module Mid #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              Leaf #(WIDTH) u_leaf(.a(a), .y(y));
            endmodule

            module NestedPosTop(input logic [5:0] a, output logic [5:0] y);
              Mid #(6) u_mid(.a(a), .y(y));
            endmodule
            "#,
            Some("NestedPosTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Mid__WIDTH_6").is_some());
        assert!(compiled.find_module("Leaf__WIDTH_6").is_some());
        let mut sim = Simulator::new(&imported.design, "NestedPosTop").unwrap();
        let a = signal(&imported, "NestedPosTop", "a");
        let y = signal(&imported, "NestedPosTop", "y");
        sim.set(a, 10);
        assert_eq!(sim.get(y), 53);
    }

    #[test]
    fn imports_positional_instance_override_from_parent_localparam() {
        let imported = import_sv(
            r#"
            module LocalWidth #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a ^ {WIDTH{1'b1}};
            endmodule

            module LocalPosTop(input logic [7:0] a, output logic [7:0] y);
              localparam int BASE = 4;
              LocalWidth #(BASE * 2) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("LocalPosTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("LocalWidth__WIDTH_8").is_some());
        let mut sim = Simulator::new(&imported.design, "LocalPosTop").unwrap();
        let a = signal(&imported, "LocalPosTop", "a");
        let y = signal(&imported, "LocalPosTop", "y");
        sim.set(a, 2);
        assert_eq!(sim.get(y), 253);
    }

    #[test]
    fn rejects_mixed_instance_parameter_overrides() {
        let err = parse_sv(
            r#"
            module Child #(parameter int WIDTH = 4, parameter int MASK = 0) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module Bad(input logic [7:0] a, output logic [7:0] y);
              Child #(8, .MASK(1)) u_child(.a(a), .y(y));
            endmodule
            "#,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn rejects_too_many_positional_instance_parameter_overrides() {
        let err = import_sv(
            r#"
            module Child #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module Bad(input logic [7:0] a, output logic [7:0] y);
              Child #(8, 2) u_child(.a(a), .y(y));
            endmodule
            "#,
            Some("Bad"),
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn rejects_external_positional_instance_parameter_overrides() {
        let err = import_sv(
            r#"
            module Bad(input logic [7:0] a, output logic [7:0] y);
              Vendor #(8) u_vendor(.a(a), .y(y));
            endmodule
            "#,
            Some("Bad"),
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn rejects_import_source_defined_parameter_overrides() {
        let source = parse_sv(
            r#"
            module Child #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module Bad(input logic [7:0] a, output logic [7:0] y);
              Child #(.WIDTH(8)) u_child(.a(a), .y(y));
            endmodule
            "#,
        )
        .unwrap();
        let err = import_source(source, Some("Bad")).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn rejects_import_source_defined_positional_parameter_overrides() {
        let source = parse_sv(
            r#"
            module Child #(parameter int WIDTH = 4) (
              input logic [WIDTH-1:0] a,
              output logic [WIDTH-1:0] y
            );
              assign y = a;
            endmodule

            module Bad(input logic [7:0] a, output logic [7:0] y);
              Child #(8) u_child(.a(a), .y(y));
            endmodule
            "#,
        )
        .unwrap();
        let err = import_source(source, Some("Bad")).unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_PARAM_OVERRIDE");
    }

    #[test]
    fn imports_localparam_memory_depth() {
        let imported = import_sv(
            r#"
            module ParamMem(clk, we, waddr, wdata, raddr, rdata);
              localparam int DEPTH = 4;
              input logic clk, we;
              input logic [1:0] waddr, raddr;
              input logic [7:0] wdata;
              output logic [7:0] rdata;
              logic [7:0] regs [0:DEPTH-1];
              assign rdata = regs[raddr];
              always_ff @(posedge clk) begin
                if (we) regs[waddr] <= wdata;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ParamMem").unwrap();
        let we = signal(&imported, "ParamMem", "we");
        let waddr = signal(&imported, "ParamMem", "waddr");
        let wdata = signal(&imported, "ParamMem", "wdata");
        let raddr = signal(&imported, "ParamMem", "raddr");
        let rdata = signal(&imported, "ParamMem", "rdata");
        sim.set(we, 1);
        sim.set(waddr, 3);
        sim.set(wdata, 0xcd);
        sim.tick();
        sim.set(we, 0);
        sim.set(raddr, 3);
        assert_eq!(sim.get(rdata), 0xcd);
    }

    #[test]
    fn imports_packed_bit_and_part_selects() {
        let imported = import_sv(
            r#"
            module Selects(
              input logic [7:0] a,
              output logic bit2,
              output logic [3:0] upper
            );
              assign bit2 = a[2];
              assign upper = a[7:4];
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "Selects").unwrap();
        let a = signal(&imported, "Selects", "a");
        let bit2 = signal(&imported, "Selects", "bit2");
        let upper = signal(&imported, "Selects", "upper");
        sim.set(a, 0b1010_0100);
        assert_eq!(sim.get(bit2), 1);
        assert_eq!(sim.get(upper), 0b1010);
    }

    #[test]
    fn imports_localparam_bit_and_part_select_constants() {
        let imported = import_sv(
            r#"
            module ConstSelects(output logic bit2, output logic [2:0] mid);
              localparam int P = 8'b1010_1100;
              localparam int B = P[2];
              localparam int M = P[5:3];
              assign bit2 = B;
              assign mid = M;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "ConstSelects").unwrap();
        let bit2 = signal(&imported, "ConstSelects", "bit2");
        let mid = signal(&imported, "ConstSelects", "mid");
        assert_eq!(sim.get(bit2), 1);
        assert_eq!(sim.get(mid), 0b101);
    }

    #[test]
    fn imports_assignment_truncates_localparam_to_destination_width() {
        let imported = import_sv(
            r#"
            module AssignTrunc(output logic [2:0] y);
              localparam int P = 8'b1010_1100;
              assign y = P;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "AssignTrunc").unwrap();
        let y = signal(&imported, "AssignTrunc", "y");
        assert_eq!(sim.get(y), 0b100);
    }

    #[test]
    fn imports_assignment_zero_extends_unsigned_narrow_expr() {
        let imported = import_sv(
            r#"
            module AssignZext(output logic [7:0] y);
              assign y = 4'b1111;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "AssignZext").unwrap();
        let y = signal(&imported, "AssignZext", "y");
        assert_eq!(sim.get(y), 0x0f);
    }

    #[test]
    fn imports_assignment_sign_extends_signed_narrow_expr() {
        let imported = import_sv(
            r#"
            module AssignSext(output logic signed [7:0] y);
              assign y = $signed(4'b1111);
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "AssignSext").unwrap();
        let y = signal(&imported, "AssignSext", "y");
        assert_eq!(sim.get(y), 0xff);
    }

    #[test]
    fn imports_always_comb_assignment_sizing() {
        let imported = import_sv(
            r#"
            module CombAssignSizing(
              input logic a,
              output logic [2:0] y
            );
              localparam int P = 8'b1010_1100;
              always_comb begin
                if (a) y = P;
                else y = 4'b0010;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombAssignSizing").unwrap();
        let a = signal(&imported, "CombAssignSizing", "a");
        let y = signal(&imported, "CombAssignSizing", "y");
        sim.set(a, 0);
        assert_eq!(sim.get(y), 0b010);
        sim.set(a, 1);
        assert_eq!(sim.get(y), 0b100);
    }

    #[test]
    fn imports_always_ff_assignment_sizing() {
        let imported = import_sv(
            r#"
            module FfAssignSizing(
              input logic clk,
              input logic sel,
              output logic [2:0] y
            );
              localparam int P = 8'b1010_1100;
              reg [2:0] q;
              always_ff @(posedge clk) begin
                q <= 4'b0010;
                if (sel) q <= P;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfAssignSizing").unwrap();
        let sel = signal(&imported, "FfAssignSizing", "sel");
        let y = signal(&imported, "FfAssignSizing", "y");
        sim.set(sel, 0);
        sim.tick();
        assert_eq!(sim.get(y), 0b010);
        sim.set(sel, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0b100);
    }

    #[test]
    fn imports_localparam_concat_and_repeat_constants() {
        let imported = import_sv(
            r#"
            module ConstPack(output logic [7:0] y, output logic [3:0] ones);
              localparam int C = {2'b10, 2'b01};
              localparam int R = {4{1'b1}};
              assign y = {C[3:0], R[3:0]};
              assign ones = R[3:0];
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "ConstPack").unwrap();
        let y = signal(&imported, "ConstPack", "y");
        let ones = signal(&imported, "ConstPack", "ones");
        assert_eq!(sim.get(y), 0b1001_1111);
        assert_eq!(sim.get(ones), 0b1111);
    }

    #[test]
    fn imports_generate_for_constant_bit_select_bound() {
        let imported = import_sv(
            r#"
            module GenConstSelect(output logic [1:0] y);
              generate
                for (genvar i = 0; i < 2'b10[1:0]; i++) begin : g
                  assign y[i] = i[0];
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "GenConstSelect").unwrap();
        let y = signal(&imported, "GenConstSelect", "y");
        assert_eq!(sim.get(y), 0b10);
    }

    #[test]
    fn rejects_dynamic_packed_bit_select() {
        let err = import_sv(
            r#"
            module DynamicSelect(
              input logic [7:0] a,
              input logic [2:0] sel,
              output logic y
            );
              assign y = a[sel];
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_CONST");
    }

    #[test]
    fn rejects_constant_bit_select_beyond_expression_width() {
        let err = import_sv(
            r#"
            module BadConstSelect(output logic y);
              localparam int P = 8'b1010_1100[8];
              assign y = P[0];
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_CONST");
    }

    #[test]
    fn rejects_constant_repeat_wider_than_128_bits() {
        let err = import_sv(
            r#"
            module BadConstRepeat(output logic y);
              localparam int P = {129{1'b1}};
              assign y = P[0];
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_CONST");
    }

    #[test]
    fn imports_comb_for_loop_with_localparam_bound() {
        let imported = import_sv(
            r#"
            module CombFor(
              input logic [3:0] mask,
              output logic any
            );
              localparam int N = 4;
              always_comb begin
                any = 1'b0;
                for (int i = 0; i < N; i = i + 1) begin
                  if (mask[i]) any = 1'b1;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombFor").unwrap();
        let mask = signal(&imported, "CombFor", "mask");
        let any = signal(&imported, "CombFor", "any");
        sim.set(mask, 0);
        assert_eq!(sim.get(any), 0);
        sim.set(mask, 0b0100);
        assert_eq!(sim.get(any), 1);
    }

    #[test]
    fn imports_comb_for_loop_decrement() {
        let imported = import_sv(
            r#"
            module CombForDec(
              input logic [3:0] mask,
              output logic any
            );
              always_comb begin
                any = 1'b0;
                for (integer i = 3; i >= 0; i--) begin
                  if (mask[i]) any = 1'b1;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombForDec").unwrap();
        let mask = signal(&imported, "CombForDec", "mask");
        let any = signal(&imported, "CombForDec", "any");
        sim.set(mask, 0);
        assert_eq!(sim.get(any), 0);
        sim.set(mask, 0b1000);
        assert_eq!(sim.get(any), 1);
    }

    #[test]
    fn imports_clocked_for_loop_register_updates() {
        let imported = import_sv(
            r#"
            module FfFor(
              input logic clk,
              output logic [31:0] q
            );
              always_ff @(posedge clk) begin
                for (int i = 0; i < 4; i++) begin
                  q <= i;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfFor").unwrap();
        let q = signal(&imported, "FfFor", "q");
        sim.tick();
        assert_eq!(sim.get(q), 3);
    }

    #[test]
    fn imports_reset_branch_for_loop_assignments() {
        let imported = import_sv(
            r#"
            module ResetFor(
              input logic clk,
              input logic rst_n,
              output logic [3:0] q
            );
              always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                  for (int i = 0; i < 1; i++) begin
                    q <= i;
                  end
                end else begin
                  q <= q + 4'd1;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ResetFor").unwrap();
        let rst_n = signal(&imported, "ResetFor", "rst_n");
        let q = signal(&imported, "ResetFor", "q");
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0);
        sim.set(rst_n, 1);
        sim.tick();
        assert_eq!(sim.get(q), 1);
    }

    #[test]
    fn rejects_dynamic_for_loop_bound() {
        let err = import_sv(
            r#"
            module DynamicFor(
              input logic [3:0] n,
              output logic y
            );
              always_comb begin
                y = 1'b0;
                for (int i = 0; i < n; i++) y = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_FOR");
    }

    #[test]
    fn rejects_zero_for_loop_step() {
        let err = import_sv(
            r#"
            module ZeroStepFor(output logic y);
              always_comb begin
                y = 1'b0;
                for (int i = 0; i < 4; i = i + 0) y = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_FOR");
    }

    #[test]
    fn rejects_excessive_for_loop_unroll() {
        let err = import_sv(
            r#"
            module BigFor(output logic y);
              always_comb begin
                y = 1'b0;
                for (int i = 0; i <= 4096; i++) y = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_FOR");
    }

    #[test]
    fn imports_generate_for_continuous_bit_assigns() {
        let imported = import_sv(
            r#"
            module GenAssign(
              input logic [3:0] a,
              input logic [3:0] b,
              output logic [3:0] y
            );
              genvar i;
              generate
                for (i = 0; i < 4; i++) begin : g
                  assign y[i] = a[i] ^ b[i];
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenAssign").unwrap();
        let a = signal(&imported, "GenAssign", "a");
        let b = signal(&imported, "GenAssign", "b");
        let y = signal(&imported, "GenAssign", "y");
        sim.set(a, 0b1010);
        sim.set(b, 0b0011);
        assert_eq!(sim.get(y), 0b1001);
    }

    #[test]
    fn imports_generate_for_repeated_instance_inputs() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a);
            endmodule

            module GenInst(input logic [1:0] a);
              generate
                for (genvar i = 0; i < 2; i++) begin : lane
                  Leaf u(.a(a[i]));
                end
              endgenerate
            endmodule
            "#,
            Some("GenInst"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("GenInst").unwrap();
        assert!(top
            .instances
            .iter()
            .any(|instance| instance.name == "lane__0__u"));
        assert!(top
            .instances
            .iter()
            .any(|instance| instance.name == "lane__1__u"));
    }

    #[test]
    fn imports_comb_selected_lvalue_assignments() {
        let imported = import_sv(
            r#"
            module SelectedComb(
              input logic a,
              input logic [2:0] mid,
              output logic [3:0] y
            );
              always_comb begin
                y = 4'b0000;
                y[0] = a;
                y[3:1] = mid;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "SelectedComb").unwrap();
        let a = signal(&imported, "SelectedComb", "a");
        let mid = signal(&imported, "SelectedComb", "mid");
        let y = signal(&imported, "SelectedComb", "y");
        sim.set(a, 1);
        sim.set(mid, 0b101);
        assert_eq!(sim.get(y), 0b1011);
    }

    #[test]
    fn imports_clocked_selected_lvalue_assignments() {
        let imported = import_sv(
            r#"
            module SelectedFf(
              input logic clk,
              input logic a,
              input logic [2:0] mid,
              output logic [3:0] q
            );
              always_ff @(posedge clk) begin
                q <= 4'b0000;
                q[0] <= a;
                q[3:1] <= mid;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "SelectedFf").unwrap();
        let a = signal(&imported, "SelectedFf", "a");
        let mid = signal(&imported, "SelectedFf", "mid");
        let q = signal(&imported, "SelectedFf", "q");
        sim.set(a, 1);
        sim.set(mid, 0b110);
        sim.tick();
        assert_eq!(sim.get(q), 0b1101);
    }

    #[test]
    fn imports_reset_selected_lvalue_assignments() {
        let imported = import_sv(
            r#"
            module SelectedReset(
              input logic clk,
              input logic rst,
              output logic [3:0] q
            );
              always_ff @(posedge clk or posedge rst) begin
                if (rst) begin
                  q[0] <= 1'b1;
                  q[3:1] <= 3'b101;
                end else begin
                  q <= 4'b0010;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "SelectedReset").unwrap();
        let rst = signal(&imported, "SelectedReset", "rst");
        let q = signal(&imported, "SelectedReset", "q");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(q), 0b1011);
        sim.set(rst, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0b0010);
    }

    #[test]
    fn imports_sync_reset_assignment_truncation() {
        let imported = import_sv(
            r#"
            module ResetTrunc(
              input logic clk,
              input logic rst,
              output logic [2:0] y
            );
              reg [2:0] q;
              always_ff @(posedge clk) begin
                if (rst) q <= 8'hac;
                else q <= 3'b001;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ResetTrunc").unwrap();
        let rst = signal(&imported, "ResetTrunc", "rst");
        let y = signal(&imported, "ResetTrunc", "y");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0b100);
        sim.set(rst, 0);
        sim.tick();
        assert_eq!(sim.get(y), 0b001);
    }

    #[test]
    fn imports_async_active_low_reset_assignment_truncation() {
        let imported = import_sv(
            r#"
            module AsyncResetTrunc(
              input logic clk,
              input logic rst_n,
              output logic [2:0] y
            );
              reg [2:0] q;
              always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) q <= 8'hac;
                else q <= 3'b001;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "AsyncResetTrunc").unwrap();
        let rst_n = signal(&imported, "AsyncResetTrunc", "rst_n");
        let y = signal(&imported, "AsyncResetTrunc", "y");
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(y), 0b100);
        sim.set(rst_n, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0b001);
    }

    #[test]
    fn imports_reset_assignment_sign_extension() {
        let imported = import_sv(
            r#"
            module ResetSext(
              input logic clk,
              input logic rst,
              output logic signed [7:0] y
            );
              reg signed [7:0] q;
              always_ff @(posedge clk) begin
                if (rst) q <= $signed(4'b1111);
                else q <= 8'sd0;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ResetSext").unwrap();
        let rst = signal(&imported, "ResetSext", "rst");
        let y = signal(&imported, "ResetSext", "y");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0xff);
    }

    #[test]
    fn imports_reset_assignment_zero_extension() {
        let imported = import_sv(
            r#"
            module ResetZext(
              input logic clk,
              input logic rst,
              output logic [7:0] y
            );
              reg [7:0] q;
              always_ff @(posedge clk) begin
                if (rst) q <= 4'b1111;
                else q <= 8'h00;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "ResetZext").unwrap();
        let rst = signal(&imported, "ResetZext", "rst");
        let y = signal(&imported, "ResetZext", "y");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(y), 0x0f);
    }

    #[test]
    fn nonconstant_reset_branch_uses_normal_next_path() {
        let imported = import_sv(
            r#"
            module DynamicResetBranch(
              input logic clk,
              input logic rst,
              input logic [2:0] d,
              output logic [2:0] y
            );
              reg [2:0] q;
              always_ff @(posedge clk) begin
                if (rst) q <= d;
                else q <= 3'b001;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "DynamicResetBranch").unwrap();
        let rst = signal(&imported, "DynamicResetBranch", "rst");
        let d = signal(&imported, "DynamicResetBranch", "d");
        let y = signal(&imported, "DynamicResetBranch", "y");
        sim.set(rst, 1);
        sim.set(d, 0b101);
        sim.tick();
        assert_eq!(sim.get(y), 0b101);
    }

    #[test]
    fn imports_initial_bit_select_assignment() {
        let imported = import_sv(
            r#"
            module InitialBit(
              input logic clk,
              output logic [3:0] y
            );
              reg [3:0] q;
              initial begin
                q[0] = 1'b1;
              end
              always_ff @(posedge clk) begin
                q <= q;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialBit").unwrap();
        let y = signal(&imported, "InitialBit", "y");
        assert_eq!(sim.get(y), 0b0001);
    }

    #[test]
    fn imports_initial_slice_assignment() {
        let imported = import_sv(
            r#"
            module InitialSlice(
              input logic clk,
              output logic [3:0] y
            );
              reg [3:0] q;
              initial begin
                q[3:1] = 3'b101;
              end
              always_ff @(posedge clk) begin
                q <= q;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialSlice").unwrap();
        let y = signal(&imported, "InitialSlice", "y");
        assert_eq!(sim.get(y), 0b1010);
    }

    #[test]
    fn imports_initial_whole_and_selected_assignments_in_order() {
        let imported = import_sv(
            r#"
            module InitialMixed(
              input logic clk,
              output logic [3:0] y
            );
              reg [3:0] q;
              initial begin
                q = 4'b1111;
                q[2:1] = 2'b00;
                q[0] = 1'b0;
              end
              always_ff @(posedge clk) begin
                q <= q;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialMixed").unwrap();
        let y = signal(&imported, "InitialMixed", "y");
        assert_eq!(sim.get(y), 0b1000);
    }

    #[test]
    fn imports_multiple_initial_blocks_for_same_register() {
        let imported = import_sv(
            r#"
            module InitialMulti(
              input logic clk,
              output logic [3:0] y
            );
              reg [3:0] q;
              initial begin
                q[1:0] = 2'b01;
              end
              initial begin
                q[3:2] = 2'b10;
              end
              always_ff @(posedge clk) begin
                q <= q;
              end
              assign y = q;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialMulti").unwrap();
        let y = signal(&imported, "InitialMulti", "y");
        assert_eq!(sim.get(y), 0b1001);
    }

    #[test]
    fn imports_generated_initial_selected_registers() {
        let imported = import_sv(
            r#"
            module InitialGen(
              input logic clk,
              output logic [1:0] y
            );
              generate
                for (genvar i = 0; i < 2; i++) begin : lane
                  reg tmp;
                  initial begin
                    tmp[0] = i[0];
                  end
                  always_ff @(posedge clk) begin
                    tmp <= tmp;
                  end
                  assign y[i] = tmp;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "InitialGen").unwrap();
        let y = signal(&imported, "InitialGen", "y");
        assert_eq!(sim.get(y), 0b10);
    }

    #[test]
    fn initial_memory_bit_syntax_still_initializes_memory() {
        let imported = import_sv(
            r#"
            module InitialMemoryCompat(
              input logic [1:0] addr,
              output logic [7:0] y
            );
              logic [7:0] mem [0:3];
              initial begin
                mem[2] = 8'hab;
              end
              assign y = mem[addr];
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "InitialMemoryCompat").unwrap();
        let addr = signal(&imported, "InitialMemoryCompat", "addr");
        let y = signal(&imported, "InitialMemoryCompat", "y");
        sim.set(addr, 2);
        assert_eq!(sim.get(y), 0xab);
    }

    #[test]
    fn rejects_initial_selected_assignment_to_wire() {
        let err = import_sv(
            r#"
            module BadInitialWire(output logic [1:0] y);
              initial begin
                y[0] = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_INITIAL_REG_TARGET");
    }

    #[test]
    fn rejects_dynamic_generate_for_bound() {
        let err = import_sv(
            r#"
            module BadGen(input logic [3:0] n, output logic [3:0] y);
              generate
                for (genvar i = 0; i < n; i++) begin : g
                  assign y[i] = 1'b1;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_FOR");
    }

    #[test]
    fn imports_generate_for_local_wire() {
        let imported = import_sv(
            r#"
            module GenLocalWire(
              input logic [1:0] a,
              output logic [1:0] y
            );
              generate
                for (genvar i = 0; i < 2; i++) begin : g
                  logic tmp;
                  assign tmp = a[i];
                  assign y[i] = tmp;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalWire").unwrap();
        let a = signal(&imported, "GenLocalWire", "a");
        let y = signal(&imported, "GenLocalWire", "y");
        sim.set(a, 0b10);
        assert_eq!(sim.get(y), 0b10);
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("GenLocalWire").unwrap();
        assert!(top.signals.iter().any(|signal| signal.name == "g__0__tmp"));
        assert!(top.signals.iter().any(|signal| signal.name == "g__1__tmp"));
    }

    #[test]
    fn imports_generate_local_wire_used_in_always_comb() {
        let imported = import_sv(
            r#"
            module GenLocalComb(
              input logic [1:0] a,
              output logic [1:0] y
            );
              generate
                for (genvar i = 0; i < 2; i++) begin : g
                  logic tmp;
                  always_comb begin
                    tmp = ~a[i];
                  end
                  assign y[i] = tmp;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalComb").unwrap();
        let a = signal(&imported, "GenLocalComb", "a");
        let y = signal(&imported, "GenLocalComb", "y");
        sim.set(a, 0b01);
        assert_eq!(sim.get(y), 0b10);
    }

    #[test]
    fn imports_generate_local_reg_used_in_always_ff() {
        let imported = import_sv(
            r#"
            module GenLocalReg(
              input logic clk,
              input logic [1:0] a,
              output logic [1:0] y
            );
              generate
                for (genvar i = 0; i < 2; i++) begin : g
                  reg tmp;
                  always_ff @(posedge clk) begin
                    tmp <= a[i];
                  end
                  assign y[i] = tmp;
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalReg").unwrap();
        let a = signal(&imported, "GenLocalReg", "a");
        let y = signal(&imported, "GenLocalReg", "y");
        sim.set(a, 0b11);
        sim.tick();
        assert_eq!(sim.get(y), 0b11);
    }

    #[test]
    fn imports_nested_generate_local_names() {
        let imported = import_sv(
            r#"
            module NestedGenLocal(output logic [1:0] y);
              generate
                for (genvar i = 0; i < 2; i++) begin : row
                  for (genvar j = 0; j < 1; j++) begin : col
                    logic tmp;
                    assign tmp = i[0];
                    assign y[i] = tmp;
                  end
                end
              endgenerate
            endmodule
            "#,
            None,
        )
        .unwrap();
        let sim = Simulator::new(&imported.design, "NestedGenLocal").unwrap();
        let y = signal(&imported, "NestedGenLocal", "y");
        assert_eq!(sim.get(y), 0b10);
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("NestedGenLocal").unwrap();
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "row__0__col__0__tmp"));
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "row__1__col__0__tmp"));
    }

    #[test]
    fn imports_generated_instance_through_local_wire() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a);
            endmodule

            module GenLocalInst(input logic [1:0] a, output logic [1:0] y);
              generate
                for (genvar i = 0; i < 2; i++) begin : lane
                  logic tmp;
                  assign tmp = a[i];
                  assign y[i] = tmp;
                  Leaf u(.a(tmp));
                end
              endgenerate
            endmodule
            "#,
            Some("GenLocalInst"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalInst").unwrap();
        let a = signal(&imported, "GenLocalInst", "a");
        let y = signal(&imported, "GenLocalInst", "y");
        sim.set(a, 0b01);
        assert_eq!(sim.get(y), 0b01);
    }

    #[test]
    fn imports_instance_output_bit_select_connection() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a, output logic y);
              assign y = a;
            endmodule

            module InstOutBit(input logic a, output logic [1:0] y);
              assign y[1] = 1'b0;
              Leaf u(.a(a), .y(y[0]));
            endmodule
            "#,
            Some("InstOutBit"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "InstOutBit").unwrap();
        let a = signal(&imported, "InstOutBit", "a");
        let y = signal(&imported, "InstOutBit", "y");
        sim.set(a, 1);
        assert_eq!(sim.get(y), 0b01);
    }

    #[test]
    fn imports_generate_for_instance_output_bit_selects() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a, output logic y);
              assign y = a;
            endmodule

            module GenInstOut(input logic [3:0] a, output logic [3:0] y);
              generate
                for (genvar i = 0; i < 4; i++) begin : lane
                  Leaf u(.a(a[i]), .y(y[i]));
                end
              endgenerate
            endmodule
            "#,
            Some("GenInstOut"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenInstOut").unwrap();
        let a = signal(&imported, "GenInstOut", "a");
        let y = signal(&imported, "GenInstOut", "y");
        sim.set(a, 0b1011);
        assert_eq!(sim.get(y), 0b1011);
    }

    #[test]
    fn imports_instance_output_slice_connection() {
        let imported = import_sv(
            r#"
            module Leaf(input logic [1:0] a, output logic [1:0] y);
              assign y = a;
            endmodule

            module InstOutSlice(input logic [1:0] a, output logic [3:0] y);
              assign y[1:0] = 2'b01;
              Leaf u(.a(a), .y(y[3:2]));
            endmodule
            "#,
            Some("InstOutSlice"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "InstOutSlice").unwrap();
        let a = signal(&imported, "InstOutSlice", "a");
        let y = signal(&imported, "InstOutSlice", "y");
        sim.set(a, 0b10);
        assert_eq!(sim.get(y), 0b1001);
    }

    #[test]
    fn imports_generated_local_wire_to_selected_instance_output() {
        let imported = import_sv(
            r#"
            module Leaf(input logic a, output logic y);
              assign y = a;
            endmodule

            module GenLocalInstOut(input logic [1:0] a, output logic [1:0] y);
              generate
                for (genvar i = 0; i < 2; i++) begin : lane
                  logic tmp;
                  assign tmp = a[i];
                  Leaf u(.a(tmp), .y(y[i]));
                end
              endgenerate
            endmodule
            "#,
            Some("GenLocalInstOut"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenLocalInstOut").unwrap();
        let a = signal(&imported, "GenLocalInstOut", "a");
        let y = signal(&imported, "GenLocalInstOut", "y");
        sim.set(a, 0b10);
        assert_eq!(sim.get(y), 0b10);
    }

    #[test]
    fn rejects_selected_inout_instance_connection() {
        let err = import_sv(
            r#"
            module Pad(inout logic p);
            endmodule

            module BadInout(inout logic [1:0] bus);
              Pad u(.p(bus[0]));
            endmodule
            "#,
            Some("BadInout"),
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_INSTANCE_CONN");
    }

    #[test]
    fn rejects_generated_memory_declarations() {
        let err = import_sv(
            r#"
            module BadGenMem(output logic y);
              generate
                for (genvar i = 0; i < 1; i++) begin : g
                  logic tmp [0:1];
                end
              endgenerate
              assign y = 1'b0;
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_GENERATE");
    }

    #[test]
    fn imports_generate_scope_localparams() {
        // A generate-scope localparam that depends on the genvar — folded per
        // iteration during unrolling (here a single iteration with i=2 → 21).
        let imported = import_sv(
            r#"
            module GenParam(output logic [7:0] o);
              for (genvar i = 2; i < 3; i++) begin : g
                localparam P = 10 * i + 1;   // 21
                assign o = P;
              end
            endmodule
            "#,
            Some("GenParam"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "GenParam").unwrap();
        let o = signal(&imported, "GenParam", "o");
        assert_eq!(sim.get(o), 21);
    }

    #[test]
    fn rejects_generated_typedefs() {
        let err = import_sv(
            r#"
            module BadGenTypedef(output logic y);
              generate
                for (genvar i = 0; i < 1; i++) begin : g
                  typedef enum logic [0:0] { A = 1'b0 } t;
                  t tmp;
                end
              endgenerate
              assign y = 1'b0;
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_GENERATE");
    }

    #[test]
    fn rejects_selected_lvalue_out_of_range() {
        let err = import_sv(
            r#"
            module BadSelect(output logic [1:0] y);
              always_comb begin
                y = 2'b00;
                y[2] = 1'b1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_LVALUE");
    }

    #[test]
    fn parses_case_statement_variants() {
        let source = parse_sv(
            r#"
            module Decoder(input logic [1:0] op, output logic [3:0] y);
              always_comb begin
                priority case (op)
                  2'd0, 2'd1: y = 4'd1;
                  default: y = 4'd0;
                endcase
              end
            endmodule
            "#,
        )
        .unwrap();
        let SvItem::AlwaysComb(stmts) = source.modules[0]
            .items
            .iter()
            .find(|item| matches!(item, SvItem::AlwaysComb(_)))
            .unwrap()
        else {
            panic!("expected always_comb item");
        };
        let SvStmt::Case { kind, expr, items } = &stmts[0] else {
            panic!("expected case statement");
        };
        assert_eq!(*kind, SvCaseKind::Normal);
        assert!(matches!(expr, SvExpr::Ident(name) if name == "op"));
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].labels.len(), 2);
        assert!(matches!(items[0].labels[0], SvCaseLabel::Expr(_)));
        assert!(!items[0].is_default);
        assert!(items[1].is_default);
    }

    #[test]
    fn imports_casez_wildcard_decoder() {
        let imported = import_sv(
            r#"
            module WildCaseZ(input logic [3:0] op, output logic [1:0] y);
              always_comb begin
                y = 2'd0;
                unique casez (op)
                  4'b10??: y = 2'd1;
                  4'b0z01, 4'b0?10: y = 2'd2;
                  default: y = 2'd3;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "WildCaseZ").unwrap();
        let op = signal(&imported, "WildCaseZ", "op");
        let y = signal(&imported, "WildCaseZ", "y");
        sim.set(op, 0b1000);
        assert_eq!(sim.get(y), 1);
        sim.set(op, 0b0101);
        assert_eq!(sim.get(y), 2);
        sim.set(op, 0b0010);
        assert_eq!(sim.get(y), 2);
        sim.set(op, 0b1111);
        assert_eq!(sim.get(y), 3);
    }

    #[test]
    fn imports_casex_wildcard_decoder() {
        let imported = import_sv(
            r#"
            module WildCaseX(input logic [3:0] op, output logic [1:0] y);
              always_comb begin
                y = 2'd0;
                priority casex (op)
                  4'b1x0?: y = 2'd1;
                  4'hz: y = 2'd2;
                  default: y = 2'd3;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "WildCaseX").unwrap();
        let op = signal(&imported, "WildCaseX", "op");
        let y = signal(&imported, "WildCaseX", "y");
        sim.set(op, 0b1001);
        assert_eq!(sim.get(y), 1);
        sim.set(op, 0b1111);
        assert_eq!(sim.get(y), 2);
    }

    #[test]
    fn imports_clocked_casez_wildcard_updates() {
        let imported = import_sv(
            r#"
            module WildFf(
              input logic clk,
              input logic [2:0] op,
              output logic [3:0] q
            );
              always_ff @(posedge clk) begin
                casez (op)
                  3'b1??: q <= 4'd8;
                  3'b0?1: q <= 4'd3;
                  default: q <= q;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "WildFf").unwrap();
        let op = signal(&imported, "WildFf", "op");
        let q = signal(&imported, "WildFf", "q");
        sim.set(op, 0b101);
        sim.tick();
        assert_eq!(sim.get(q), 8);
        sim.set(op, 0b001);
        sim.tick();
        assert_eq!(sim.get(q), 3);
        sim.set(op, 0b010);
        sim.tick();
        assert_eq!(sim.get(q), 3);
    }

    #[test]
    fn rejects_casez_x_wildcard_labels() {
        let err = import_sv(
            r#"
            module BadCaseZ(input logic [1:0] op, output logic y);
              always_comb begin
                y = 1'd0;
                casez (op)
                  2'b1x: y = 1'd1;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_CASE");
    }

    #[test]
    fn imports_comb_case_with_defaults_and_comma_labels() {
        let imported = import_sv(
            include_str!("../tests/fixtures/comb_case_decoder.sv"),
            Some("CombCaseDecoder"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombCaseDecoder").unwrap();
        let op = signal(&imported, "CombCaseDecoder", "op");
        let y = signal(&imported, "CombCaseDecoder", "y");

        sim.set(op, 0);
        assert_eq!(sim.get(y), 1);
        sim.set(op, 1);
        assert_eq!(sim.get(y), 7);
        sim.set(op, 2);
        assert_eq!(sim.get(y), 7);
        sim.set(op, 3);
        assert_eq!(sim.get(y), 15);
    }

    #[test]
    fn imports_comb_if_defaults_and_nested_updates() {
        let imported = import_sv(
            include_str!("../tests/fixtures/comb_if_defaults.sv"),
            Some("CombIfDefaults"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CombIfDefaults").unwrap();
        let en = signal(&imported, "CombIfDefaults", "en");
        let sel = signal(&imported, "CombIfDefaults", "sel");
        let a = signal(&imported, "CombIfDefaults", "a");
        let b = signal(&imported, "CombIfDefaults", "b");
        let c = signal(&imported, "CombIfDefaults", "c");
        let y = signal(&imported, "CombIfDefaults", "y");
        let z = signal(&imported, "CombIfDefaults", "z");

        sim.set(a, 0x11);
        sim.set(b, 0x22);
        sim.set(c, 0x33);
        assert_eq!(sim.get(y), 0x11);
        assert_eq!(sim.get(z), 0);

        sim.set(en, 1);
        assert_eq!(sim.get(y), 0x22);
        assert_eq!(sim.get(z), 0);

        sim.set(sel, 1);
        assert_eq!(sim.get(y), 0x22);
        assert_eq!(sim.get(z), 0x33);
    }

    #[test]
    fn imports_legacy_always_star_comb_logic() {
        let imported = import_sv(
            r#"
            module LegacyCombStar(
              input logic en,
              input logic sel,
              input logic [7:0] a,
              input logic [7:0] b,
              input logic [7:0] c,
              output logic [7:0] y
            );
              always @(*) begin
                y = a;
                if (en) begin
                  y = b;
                  if (sel) y = c;
                end
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "LegacyCombStar").unwrap();
        let en = signal(&imported, "LegacyCombStar", "en");
        let sel = signal(&imported, "LegacyCombStar", "sel");
        let a = signal(&imported, "LegacyCombStar", "a");
        let b = signal(&imported, "LegacyCombStar", "b");
        let c = signal(&imported, "LegacyCombStar", "c");
        let y = signal(&imported, "LegacyCombStar", "y");
        sim.set(a, 0x11);
        sim.set(b, 0x22);
        sim.set(c, 0x33);
        assert_eq!(sim.get(y), 0x11);
        sim.set(en, 1);
        assert_eq!(sim.get(y), 0x22);
        sim.set(sel, 1);
        assert_eq!(sim.get(y), 0x33);
    }

    #[test]
    fn imports_legacy_always_at_star_comb_logic() {
        let imported = import_sv(
            r#"
            module LegacyCombAtStar(input logic a, input logic b, output logic y);
              always @* y = a & b;
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "LegacyCombAtStar").unwrap();
        let a = signal(&imported, "LegacyCombAtStar", "a");
        let b = signal(&imported, "LegacyCombAtStar", "b");
        let y = signal(&imported, "LegacyCombAtStar", "y");
        sim.set(a, 1);
        sim.set(b, 0);
        assert_eq!(sim.get(y), 0);
        sim.set(b, 1);
        assert_eq!(sim.get(y), 1);
    }

    #[test]
    fn imports_handwritten_ready_valid_comb_control() {
        let imported = import_sv(
            include_str!("../tests/fixtures/rv_handwritten_slice.sv"),
            Some("RvHandwrittenSlice"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "RvHandwrittenSlice").unwrap();
        let in_valid = signal(&imported, "RvHandwrittenSlice", "in_valid");
        let in_ready = signal(&imported, "RvHandwrittenSlice", "in_ready");
        let in_bits = signal(&imported, "RvHandwrittenSlice", "in_bits");
        let out_valid = signal(&imported, "RvHandwrittenSlice", "out_valid");
        let out_ready = signal(&imported, "RvHandwrittenSlice", "out_ready");
        let out_bits = signal(&imported, "RvHandwrittenSlice", "out_bits");
        let stall = signal(&imported, "RvHandwrittenSlice", "stall");

        sim.set(in_valid, 1);
        sim.set(in_bits, 0xa5);
        sim.set(out_ready, 0);
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0xa5);
        assert_eq!(sim.get(stall), 1);

        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(stall), 0);
    }

    #[test]
    fn imports_instance_input_expressions_for_defined_child() {
        let imported = import_sv(
            r#"
            module ExprChild(
              input logic [7:0] a,
              input logic en,
              output logic [7:0] y
            );
              assign y = en ? a : 8'd0;
            endmodule

            module ExprTop(
              input logic [7:0] x,
              input logic valid,
              output logic [7:0] y
            );
              ExprChild u_child(
                .a(x + 8'd1),
                .en(valid & 1'b1),
                .y(y)
              );
            endmodule
            "#,
            Some("ExprTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("ExprTop").unwrap();
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "__sv_inst_u_child_a"));
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "__sv_inst_u_child_en"));

        let mut sim = Simulator::new(&imported.design, "ExprTop").unwrap();
        let x = signal(&imported, "ExprTop", "x");
        let valid = signal(&imported, "ExprTop", "valid");
        let y = signal(&imported, "ExprTop", "y");
        sim.set(x, 0x12);
        sim.set(valid, 1);
        assert_eq!(sim.get(y), 0x13);
        sim.set(valid, 0);
        assert_eq!(sim.get(y), 0);
    }

    #[test]
    fn imports_instance_constant_tieoff_and_avoids_temp_name_collision() {
        let imported = import_sv(
            r#"
            module ConstChild(input logic [7:0] a, output logic [7:0] y);
              assign y = a;
            endmodule

            module ConstTop(output logic [7:0] y);
              logic [7:0] __sv_inst_u_child_a;
              assign __sv_inst_u_child_a = 8'd0;
              ConstChild u_child(.a(1'b1), .y(y));
            endmodule
            "#,
            Some("ConstTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let top = compiled.find_module("ConstTop").unwrap();
        assert!(top
            .signals
            .iter()
            .any(|signal| signal.name == "__sv_inst_u_child_a_1"));

        let sim = Simulator::new(&imported.design, "ConstTop").unwrap();
        let y = signal(&imported, "ConstTop", "y");
        assert_eq!(sim.get(y), 1);
    }

    #[test]
    fn imports_external_instance_input_expressions() {
        let imported = import_sv(
            r#"
            module ExtTop(input logic sel, output logic [3:0] y);
              Vendor u_vendor(
                .a(4'hf),
                .en(sel | 1'b0),
                .y(y)
              );
            endmodule
            "#,
            Some("ExtTop"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let vendor = compiled.find_module("Vendor").unwrap();
        let a = vendor
            .signals
            .iter()
            .find(|signal| signal.name == "a")
            .unwrap();
        let en = vendor
            .signals
            .iter()
            .find(|signal| signal.name == "en")
            .unwrap();
        let y = vendor
            .signals
            .iter()
            .find(|signal| signal.name == "y")
            .unwrap();
        assert_eq!(a.ty, uint(4));
        assert_eq!(en.ty, uint(1));
        assert_eq!(y.ty, uint(4));
        assert!(matches!(a.kind, SignalKind::Input));
        assert!(matches!(en.kind, SignalKind::Input));
        assert!(matches!(y.kind, SignalKind::Output));
    }

    #[test]
    fn rejects_defined_instance_output_expressions() {
        for expr in ["a & b", "1'b0"] {
            let err = import_sv(
                &format!(
                    r#"
                    module OutChild(output logic y);
                      assign y = 1'b1;
                    endmodule

                    module BadTop(input logic a, input logic b);
                      OutChild u_child(.y({expr}));
                    endmodule
                    "#
                ),
                Some("BadTop"),
            )
            .unwrap_err();
            assert_eq!(err.diagnostics[0].code, "E_SV_INSTANCE_CONN");
        }
    }

    #[test]
    fn imports_legacy_posedge_always_register_update() {
        let imported = import_sv(
            r#"
            module LegacyFf(input logic clk, input logic en, output logic [3:0] q);
              always @(posedge clk) begin
                if (en) q <= q + 4'd1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "LegacyFf").unwrap();
        let en = signal(&imported, "LegacyFf", "en");
        let q = signal(&imported, "LegacyFf", "q");
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(q), 2);
    }

    #[test]
    fn imports_legacy_posedge_always_async_active_low_reset() {
        let imported = import_sv(
            r#"
            module LegacyAsyncReset(
              input logic clk,
              input logic rst_n,
              input logic en,
              output logic [3:0] q
            );
              always @(posedge clk or negedge rst_n) begin
                if (!rst_n) q <= 4'd0;
                else if (en) q <= q + 4'd1;
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "LegacyAsyncReset").unwrap();
        let rst_n = signal(&imported, "LegacyAsyncReset", "rst_n");
        let en = signal(&imported, "LegacyAsyncReset", "en");
        let q = signal(&imported, "LegacyAsyncReset", "q");
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0);
        sim.set(rst_n, 1);
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(q), 2);
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0);
    }

    #[test]
    fn rejects_unsupported_legacy_always_event_controls() {
        for source in [
            r#"
            module BadLevel(input logic a, input logic b, output logic y);
              always @(a or b) y = a & b;
            endmodule
            "#,
            r#"
            module BadNegedge(input logic clk, output logic q);
              always @(negedge clk) q <= 1'd1;
            endmodule
            "#,
            r#"
            module BadLatch(input logic a, output logic y);
              always_latch y = a;
            endmodule
            "#,
        ] {
            let err = import_sv(source, None).unwrap_err();
            assert_eq!(err.diagnostics[0].code, "E_SV_ALWAYS");
        }
    }

    #[test]
    fn imports_case_without_default_using_prior_assignment_fallback() {
        let imported = import_sv(
            r#"
            module CaseFallback(
              input logic sel,
              input logic [7:0] a,
              input logic [7:0] b,
              output logic [7:0] y
            );
              always_comb begin
                y = a;
                case (sel)
                  1'd1: y = b;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CaseFallback").unwrap();
        let sel = signal(&imported, "CaseFallback", "sel");
        let a = signal(&imported, "CaseFallback", "a");
        let b = signal(&imported, "CaseFallback", "b");
        let y = signal(&imported, "CaseFallback", "y");
        sim.set(a, 0x12);
        sim.set(b, 0x34);
        assert_eq!(sim.get(y), 0x12);
        sim.set(sel, 1);
        assert_eq!(sim.get(y), 0x34);
    }

    #[test]
    fn imports_case_labels_from_localparams() {
        let imported = import_sv(
            r#"
            module CaseParams(input logic [1:0] op, output logic [3:0] y);
              localparam int A = 1;
              localparam int B = 2;
              always_comb begin
                y = 4'd0;
                case (op)
                  A: y = 4'd5;
                  B: y = 4'd6;
                endcase
              end
            endmodule
            "#,
            None,
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "CaseParams").unwrap();
        let op = signal(&imported, "CaseParams", "op");
        let y = signal(&imported, "CaseParams", "y");
        sim.set(op, 1);
        assert_eq!(sim.get(y), 5);
        sim.set(op, 2);
        assert_eq!(sim.get(y), 6);
        sim.set(op, 3);
        assert_eq!(sim.get(y), 0);
    }

    #[test]
    fn imports_clocked_if_else_register_updates() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_if_else_counter.sv"),
            Some("FfIfElseCounter"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfIfElseCounter").unwrap();
        let en = signal(&imported, "FfIfElseCounter", "en");
        let load = signal(&imported, "FfIfElseCounter", "load");
        let din = signal(&imported, "FfIfElseCounter", "din");
        let q = signal(&imported, "FfIfElseCounter", "q");

        sim.set(load, 1);
        sim.set(din, 9);
        sim.tick();
        assert_eq!(sim.get(q), 9);

        sim.set(load, 0);
        sim.set(en, 1);
        sim.tick();
        assert_eq!(sim.get(q), 10);

        sim.set(en, 0);
        sim.tick();
        assert_eq!(sim.get(q), 10);
    }

    #[test]
    fn imports_clocked_case_fsm_updates() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_case_fsm.sv"),
            Some("FfCaseFsm"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfCaseFsm").unwrap();
        let rst = signal(&imported, "FfCaseFsm", "rst");
        let start = signal(&imported, "FfCaseFsm", "start");
        let done = signal(&imported, "FfCaseFsm", "done");
        let state = signal(&imported, "FfCaseFsm", "state");

        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(state), 0);

        sim.set(rst, 0);
        sim.set(start, 1);
        sim.tick();
        assert_eq!(sim.get(state), 1);

        sim.set(start, 0);
        sim.tick();
        assert_eq!(sim.get(state), 1);

        sim.set(done, 1);
        sim.tick();
        assert_eq!(sim.get(state), 2);

        sim.set(done, 0);
        sim.tick();
        assert_eq!(sim.get(state), 0);
    }

    #[test]
    fn imports_clocked_case_with_prior_default_next() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_case_default_next.sv"),
            Some("FfCaseDefaultNext"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfCaseDefaultNext").unwrap();
        let op = signal(&imported, "FfCaseDefaultNext", "op");
        let a = signal(&imported, "FfCaseDefaultNext", "a");
        let b = signal(&imported, "FfCaseDefaultNext", "b");
        let q = signal(&imported, "FfCaseDefaultNext", "q");

        sim.set(a, 0x11);
        sim.set(b, 0x22);
        sim.set(op, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0x11);

        sim.set(op, 1);
        sim.tick();
        assert_eq!(sim.get(q), 0x22);

        sim.set(op, 2);
        sim.tick();
        assert_eq!(sim.get(q), 0x23);
    }

    #[test]
    fn imports_clocked_output_logic_as_internal_register() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_output_logic.sv"),
            Some("FfOutputLogic"),
        )
        .unwrap();
        let compiled = compile(&imported.design).unwrap();
        let module = compiled.find_module("FfOutputLogic").unwrap();
        assert!(module.signals.iter().any(|signal| signal.name == "q"));
        assert!(module
            .signals
            .iter()
            .any(|signal| signal.name == "q__sv_reg"));

        let mut sim = Simulator::new(&imported.design, "FfOutputLogic").unwrap();
        let d = signal(&imported, "FfOutputLogic", "d");
        let q = signal(&imported, "FfOutputLogic", "q");
        sim.set(d, 1);
        assert_eq!(sim.get(q), 0);
        sim.tick();
        assert_eq!(sim.get(q), 1);
        sim.set(d, 0);
        sim.tick();
        assert_eq!(sim.get(q), 0);
    }

    #[test]
    fn imports_output_reg_self_reference_as_internal_register() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_output_reg_counter.sv"),
            Some("FfOutputRegCounter"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfOutputRegCounter").unwrap();
        let en = signal(&imported, "FfOutputRegCounter", "en");
        let q = signal(&imported, "FfOutputRegCounter", "q");
        sim.set(en, 1);
        sim.tick();
        sim.tick();
        assert_eq!(sim.get(q), 2);
        sim.set(en, 0);
        sim.tick();
        assert_eq!(sim.get(q), 2);
    }

    #[test]
    fn imports_clocked_output_case_reset_without_explicit_kind() {
        let imported = import_sv(
            include_str!("../tests/fixtures/ff_output_case_reset.sv"),
            Some("FfOutputCaseReset"),
        )
        .unwrap();
        let mut sim = Simulator::new(&imported.design, "FfOutputCaseReset").unwrap();
        let rst = signal(&imported, "FfOutputCaseReset", "rst");
        let op = signal(&imported, "FfOutputCaseReset", "op");
        let d = signal(&imported, "FfOutputCaseReset", "d");
        let q = signal(&imported, "FfOutputCaseReset", "q");

        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(q), 0);

        sim.set(rst, 0);
        sim.set(op, 1);
        sim.set(d, 9);
        sim.tick();
        assert_eq!(sim.get(q), 9);

        sim.set(op, 2);
        sim.tick();
        assert_eq!(sim.get(q), 10);

        sim.set(op, 0);
        sim.tick();
        assert_eq!(sim.get(q), 10);
    }

    #[test]
    fn reimports_emitted_state_enum_register() {
        let mut design = Design::new();
        {
            let mut m = design.module("Controller");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let start = m.input("start", uint(1));
            let busy = m.output("busy", uint(1));
            let states = rrtl_core::state_type(
                "ControllerState",
                uint(2),
                [("Idle", 0), ("Run", 1), ("Done", 2)],
            );
            let state = m.state_reg("state", states, clk, rst, "Idle");
            m.state_next_hold(&state, [(start.value(), "Run")]);
            m.assign(busy, state.is("Run"));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("Controller")).unwrap();
        let mut sim = Simulator::new(&imported.design, "Controller").unwrap();
        let rst = signal(&imported, "Controller", "rst");
        let start = signal(&imported, "Controller", "start");
        let busy = signal(&imported, "Controller", "busy");
        sim.set(rst, 1);
        sim.tick();
        assert_eq!(sim.get(busy), 0);
        sim.set(rst, 0);
        sim.set(start, 1);
        sim.tick();
        assert_eq!(sim.get(busy), 1);
    }

    #[test]
    fn reimports_emitted_async_active_low_state_enum_register() {
        let mut design = Design::new();
        {
            let mut m = design.module("ActiveLowState");
            let clk = m.input("clk", uint(1));
            let rst_n = m.input("rst_n", uint(1));
            let start = m.input("start", uint(1));
            let states =
                rrtl_core::state_type("ControllerState", uint(1), [("Idle", 0), ("Run", 1)]);
            let state = m.state_reg_async_reset_low("state", states, clk, rst_n, "Idle");
            m.state_next_hold(&state, [(start.value(), "Run")]);
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("ActiveLowState")).unwrap();
        let mut sim = Simulator::new(&imported.design, "ActiveLowState").unwrap();
        let rst_n = signal(&imported, "ActiveLowState", "rst_n");
        let start = signal(&imported, "ActiveLowState", "start");
        let state = signal(&imported, "ActiveLowState", "state");
        sim.set(rst_n, 0);
        sim.tick();
        assert_eq!(sim.get(state), 0);
        sim.set(rst_n, 1);
        sim.set(start, 1);
        sim.tick();
        assert_eq!(sim.get(state), 1);
    }

    #[test]
    fn reimports_emitted_assertions_and_covers() {
        let mut design = Design::new();
        {
            let mut m = design.module("Props");
            let clk = m.input("clk", uint(1));
            let en = m.input("en", uint(1));
            let count = m.reg("count", uint(4));
            m.clock(count, clk);
            m.next(count, count.value() + lit(1, 4));
            m.assert_msg(
                "comb_ok",
                count.value().lt_expr(lit(10, 4)),
                "count too high",
            );
            m.assert_when_msg(
                "enabled_ok",
                en.value(),
                count.value().lt_expr(lit(12, 4)),
                "enabled_ok",
            );
            m.assert_clocked_msg(
                "clocked_ok",
                clk,
                count.value().lt_expr(lit(14, 4)),
                "clocked_ok",
            );
            m.cover("comb_hit", count.value().eq_expr(lit(3, 4)));
            m.cover_when("enabled_hit", en.value(), count.value().eq_expr(lit(5, 4)));
            m.cover_clocked("clocked_hit", clk, count.value().eq_expr(lit(7, 4)));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("Props")).unwrap();
        let mut sim = Simulator::new(&imported.design, "Props").unwrap();
        let en = signal(&imported, "Props", "en");
        sim.set(en, 1);
        for _ in 0..8 {
            sim.tick_checked().unwrap();
            sim.check_assertions().unwrap();
        }
        assert!(sim.cover_hits("Props.COMB_HIT") > 0);
        assert!(sim.cover_hits("Props.ENABLED_HIT") > 0);
        assert!(sim.cover_hits("Props.CLOCKED_HIT") > 0);
    }

    #[test]
    fn reimports_emitted_initial_register_values() {
        let mut design = Design::new();
        {
            let mut m = design.module("InitialReg");
            let clk = m.input("clk", uint(1));
            let q = m.reg("q", uint(8));
            let out = m.output("out", uint(8));
            m.clock(q, clk);
            m.next(q, q.value());
            m.initial(q, 0x5a);
            m.assign(out, q.value());
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("InitialReg")).unwrap();
        let sim = Simulator::new(&imported.design, "InitialReg").unwrap();
        let out = signal(&imported, "InitialReg", "out");
        assert_eq!(sim.get(out), 0x5a);
    }

    #[test]
    fn reimports_emitted_initial_memory_values() {
        let mut design = Design::new();
        {
            let mut m = design.module("InitialMem");
            let addr = m.input("addr", uint(2));
            let rdata = m.output("rdata", uint(8));
            let mem = m.mem("regs", 2, uint(8), 4);
            m.mem_init(mem, 2, 0xab);
            m.assign(rdata, rrtl_core::mem_read(mem, addr.value()));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("InitialMem")).unwrap();
        let mut sim = Simulator::new(&imported.design, "InitialMem").unwrap();
        let addr = signal(&imported, "InitialMem", "addr");
        let rdata = signal(&imported, "InitialMem", "rdata");
        sim.set(addr, 2);
        assert_eq!(sim.get(rdata), 0xab);
    }

    #[test]
    fn reimports_emitted_signed_casts() {
        let mut design = Design::new();
        {
            let mut m = design.module("Casts");
            let a = m.input("a", uint(8));
            let signed_y = m.output("signed_y", sint(8));
            let unsigned_y = m.output("unsigned_y", uint(8));
            m.assign(signed_y, rrtl_core::as_sint(a.value()));
            m.assign(unsigned_y, rrtl_core::as_uint(signed_y.value()));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("Casts")).unwrap();
        let mut sim = Simulator::new(&imported.design, "Casts").unwrap();
        let a = signal(&imported, "Casts", "a");
        let signed_y = signal(&imported, "Casts", "signed_y");
        let unsigned_y = signal(&imported, "Casts", "unsigned_y");
        sim.set(a, 0xf0);
        assert_eq!(sim.get(signed_y), 0xf0);
        assert_eq!(sim.get(unsigned_y), 0xf0);
    }

    #[test]
    fn reimports_emitted_sign_extension_replication() {
        let mut design = Design::new();
        {
            let mut m = design.module("SignExtend");
            let a = m.input("a", sint(8));
            let wide = m.output("wide", sint(16));
            m.assign(wide, rrtl_core::sext(a.value() + lit_s(-1, 8), 16));
        }

        let sv = rrtl_sv::emit(&design).unwrap();
        let imported = import_sv(&sv, Some("SignExtend")).unwrap();
        let mut sim = Simulator::new(&imported.design, "SignExtend").unwrap();
        let a = signal(&imported, "SignExtend", "a");
        let wide = signal(&imported, "SignExtend", "wide");
        sim.set(a, 0);
        assert_eq!(sim.get(wide), 0xffff);
        sim.set(a, 2);
        assert_eq!(sim.get(wide), 1);
    }

    #[test]
    fn reimports_emitted_bundle_and_interface_flattened_ports() {
        let req_ty = bundle_type(
            "Req",
            [
                field("valid", uint(1)),
                field("addr", uint(8)),
                nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
            ],
        );
        let mut bundle_design = Design::new();
        {
            let mut m = bundle_design.module("BundlePorts");
            let req = m.input_bundle("req", req_ty);
            let y = m.output("y", uint(8));
            m.assign(y, zext(req.path(["meta", "tag"]).unwrap(), 8));
        }

        let imported = reimport_emitted(&bundle_design, "BundlePorts");
        let mut sim = Simulator::new(&imported.design, "BundlePorts").unwrap();
        let req_tag = signal(&imported, "BundlePorts", "req_meta_tag");
        let y = signal(&imported, "BundlePorts", "y");
        sim.set(req_tag, 0xa);
        assert_eq!(sim.get(y), 0x0a);

        let req_ty = bundle_type(
            "Req",
            [
                field("valid", uint(1)),
                nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
            ],
        );
        let resp_ty = bundle_type("Resp", [field("data", uint(8))]);
        let bus_ty = interface_type(
            "Bus",
            [
                iface_input("req", req_ty),
                iface_output("resp", resp_ty),
                iface_input("ready", uint(1)),
            ],
        );
        let mut interface_design = Design::new();
        {
            let mut m = interface_design.module("InterfacePorts");
            let bus = m.interface("bus", bus_ty);
            let req = bus.port("req").unwrap().bundle().unwrap().clone();
            let resp = bus.port("resp").unwrap().bundle().unwrap().clone();
            m.assign(
                resp.field("data").unwrap(),
                zext(req.path(["meta", "tag"]).unwrap(), 8),
            );
        }

        let imported = reimport_emitted(&interface_design, "InterfacePorts");
        let mut sim = Simulator::new(&imported.design, "InterfacePorts").unwrap();
        let req_tag = signal(&imported, "InterfacePorts", "bus_req_meta_tag");
        let resp_data = signal(&imported, "InterfacePorts", "bus_resp_data");
        sim.set(req_tag, 0xb);
        assert_eq!(sim.get(resp_data), 0x0b);
    }

    #[test]
    fn reimports_emitted_bundle_assignment() {
        let req_ty = bundle_type(
            "Req",
            [
                field("valid", uint(1)),
                field("addr", uint(8)),
                nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
            ],
        );
        let mut design = Design::new();
        {
            let mut m = design.module("BundleAssign");
            let req = m.input_bundle("req", req_ty.clone());
            let wire = m.wire_bundle("pipe", req_ty.clone());
            let resp = m.output_bundle("resp", req_ty);
            m.assign_bundle(&wire, &req);
            m.assign_bundle(&resp, &wire);
        }

        let imported = reimport_emitted(&design, "BundleAssign");
        let mut sim = Simulator::new(&imported.design, "BundleAssign").unwrap();
        let req_addr = signal(&imported, "BundleAssign", "req_addr");
        let req_tag = signal(&imported, "BundleAssign", "req_meta_tag");
        let resp_addr = signal(&imported, "BundleAssign", "resp_addr");
        let resp_tag = signal(&imported, "BundleAssign", "resp_meta_tag");
        sim.set(req_addr, 0x5a);
        sim.set(req_tag, 0xc);
        assert_eq!(sim.get(resp_addr), 0x5a);
        assert_eq!(sim.get(resp_tag), 0xc);
    }

    #[test]
    fn reimports_emitted_external_and_inout_instances() {
        let mut extern_design = Design::new();
        {
            let mut ext = extern_design.extern_module("VendorAddOne");
            ext.input("a", uint(8));
            ext.output("y", uint(8));
        }
        {
            let mut m = extern_design.module("Top");
            let a = m.input("a", uint(8));
            let y = m.output("y", uint(8));
            m.instance("u_vendor", "VendorAddOne", [("a", a), ("y", y)]);
        }
        let imported = reimport_emitted(&extern_design, "Top");
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Top").is_some());

        let mut inout_design = Design::new();
        {
            let mut ext = inout_design.extern_module("IOBUF");
            ext.inout("PAD", uint(1));
            ext.input("I", uint(1));
            ext.input("T", uint(1));
            ext.output("O", uint(1));
        }
        {
            let mut m = inout_design.module("Top");
            let pad = m.inout("pad", uint(1));
            let i = m.input("i", uint(1));
            let t = m.input("t", uint(1));
            let o = m.output("o", uint(1));
            m.instance(
                "u_iobuf",
                "IOBUF",
                [("PAD", pad), ("I", i), ("T", t), ("O", o)],
            );
        }
        let imported = reimport_emitted(&inout_design, "Top");
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("Top").is_some());
    }

    #[test]
    fn reimports_emitted_ready_valid_connect_and_ports() {
        let payload_ty = bundle_type("Payload", [field("data", uint(8)), field("last", uint(1))]);
        let mut ports_design = Design::new();
        {
            let mut m = ports_design.module("ReadyValidPorts");
            let stream = m.rv_source("stream", rv_bundle(payload_ty));
            let bits = stream.bits_bundle().unwrap();
            let fired = m.output("fired", uint(1));
            m.assign(fired, stream.fire());
            m.assign(bits.field("data").unwrap(), lit(0, 8));
        }
        let imported = reimport_emitted(&ports_design, "ReadyValidPorts");
        let compiled = compile(&imported.design).unwrap();
        assert!(compiled.find_module("ReadyValidPorts").is_some());

        let mut connect_design = Design::new();
        {
            let mut m = connect_design.module("ReadyValidConnect");
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_connect(&input, &output);
        }

        let imported = reimport_emitted(&connect_design, "ReadyValidConnect");
        let mut sim = Simulator::new(&imported.design, "ReadyValidConnect").unwrap();
        let in_valid = signal(&imported, "ReadyValidConnect", "in_valid");
        let in_ready = signal(&imported, "ReadyValidConnect", "in_ready");
        let in_bits = signal(&imported, "ReadyValidConnect", "in_bits");
        let out_valid = signal(&imported, "ReadyValidConnect", "out_valid");
        let out_ready = signal(&imported, "ReadyValidConnect", "out_ready");
        let out_bits = signal(&imported, "ReadyValidConnect", "out_bits");
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x5a);
        sim.set(out_ready, 0);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x5a);
        assert_eq!(sim.get(in_ready), 0);
        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 1);
    }

    #[test]
    fn reimports_emitted_ready_valid_register_slice_and_skid() {
        let mut slice_design = Design::new();
        {
            let mut m = slice_design.module("ReadyValidSlice");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_register_slice("slice", &input, &output, clk, rst);
        }
        let imported = reimport_emitted(&slice_design, "ReadyValidSlice");
        let mut sim = Simulator::new(&imported.design, "ReadyValidSlice").unwrap();
        let in_valid = signal(&imported, "ReadyValidSlice", "in_valid");
        let in_ready = signal(&imported, "ReadyValidSlice", "in_ready");
        let in_bits = signal(&imported, "ReadyValidSlice", "in_bits");
        let out_valid = signal(&imported, "ReadyValidSlice", "out_valid");
        let out_ready = signal(&imported, "ReadyValidSlice", "out_ready");
        let out_bits = signal(&imported, "ReadyValidSlice", "out_bits");
        sim.set(out_ready, 1);
        assert_eq!(sim.get(in_ready), 1);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0xaa);
        sim.tick();
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0xaa);

        let mut skid_design = Design::new();
        {
            let mut m = skid_design.module("ReadyValidSkid");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_skid_buffer("skid", &input, &output, clk, rst);
        }
        let imported = reimport_emitted(&skid_design, "ReadyValidSkid");
        let mut sim = Simulator::new(&imported.design, "ReadyValidSkid").unwrap();
        let in_valid = signal(&imported, "ReadyValidSkid", "in_valid");
        let in_ready = signal(&imported, "ReadyValidSkid", "in_ready");
        let in_bits = signal(&imported, "ReadyValidSkid", "in_bits");
        let out_valid = signal(&imported, "ReadyValidSkid", "out_valid");
        let out_ready = signal(&imported, "ReadyValidSkid", "out_ready");
        let out_bits = signal(&imported, "ReadyValidSkid", "out_bits");
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x22);
        assert_eq!(sim.get(in_ready), 1);
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x22);
        sim.tick();
        sim.set(in_bits, 0x33);
        assert_eq!(sim.get(in_ready), 0);
        assert_eq!(sim.get(out_bits), 0x22);
    }

    #[test]
    fn reimports_emitted_ready_valid_fifo_and_mem_fifo() {
        let mut fifo_design = Design::new();
        {
            let mut m = fifo_design.module("ReadyValidFifo");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_scalar(uint(8)));
            let output = m.rv_source("out", rv_scalar(uint(8)));
            m.rv_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let imported = reimport_emitted(&fifo_design, "ReadyValidFifo");
        let mut sim = Simulator::new(&imported.design, "ReadyValidFifo").unwrap();
        let in_valid = signal(&imported, "ReadyValidFifo", "in_valid");
        let in_ready = signal(&imported, "ReadyValidFifo", "in_ready");
        let in_bits = signal(&imported, "ReadyValidFifo", "in_bits");
        let out_valid = signal(&imported, "ReadyValidFifo", "out_valid");
        let out_ready = signal(&imported, "ReadyValidFifo", "out_ready");
        let out_bits = signal(&imported, "ReadyValidFifo", "out_bits");
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_bits, 0x10);
        assert_eq!(sim.get(in_ready), 1);
        sim.tick();
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_bits), 0x10);

        let payload_ty = bundle_type("Payload", [field("data", uint(8)), field("last", uint(1))]);
        let mut mem_fifo_design = Design::new();
        {
            let mut m = mem_fifo_design.module("ReadyValidMemFifo");
            let clk = m.input("clk", uint(1));
            let rst = m.input("rst", uint(1));
            let input = m.rv_sink("in", rv_bundle(payload_ty.clone()));
            let output = m.rv_source("out", rv_bundle(payload_ty));
            m.rv_mem_fifo("fifo", &input, &output, clk, rst, 2);
        }
        let imported = reimport_emitted(&mem_fifo_design, "ReadyValidMemFifo");
        let mut sim = Simulator::new(&imported.design, "ReadyValidMemFifo").unwrap();
        let in_valid = signal(&imported, "ReadyValidMemFifo", "in_valid");
        let in_ready = signal(&imported, "ReadyValidMemFifo", "in_ready");
        let in_data = signal(&imported, "ReadyValidMemFifo", "in_bits_data");
        let in_last = signal(&imported, "ReadyValidMemFifo", "in_bits_last");
        let out_valid = signal(&imported, "ReadyValidMemFifo", "out_valid");
        let out_ready = signal(&imported, "ReadyValidMemFifo", "out_ready");
        let out_data = signal(&imported, "ReadyValidMemFifo", "out_bits_data");
        let out_last = signal(&imported, "ReadyValidMemFifo", "out_bits_last");
        sim.set(out_ready, 0);
        sim.set(in_valid, 1);
        sim.set(in_data, 0xa5);
        sim.set(in_last, 1);
        assert_eq!(sim.get(in_ready), 1);
        sim.tick();
        assert_eq!(sim.get(out_valid), 1);
        assert_eq!(sim.get(out_data), 0xa5);
        assert_eq!(sim.get(out_last), 1);
    }

    #[test]
    fn reimports_emitted_register_file_and_width_helpers() {
        let mut design = Design::new();
        {
            let mut m = design.module("RegFile");
            let clk = m.input("clk", 1);
            let we = m.input("we", 1);
            let waddr = m.input("waddr", 2);
            let raddr = m.input("raddr", 2);
            let wdata = m.input("wdata", 8);
            let rdata = m.output("rdata", 8);
            let low = m.output("low", 4);
            let mem = m.mem("regs", 2, 8, 4);

            m.mem_write(mem, clk, we, waddr, zext(rrtl_core::trunc(wdata, 4), 8));
            let read = m.mem_read(mem, raddr);
            m.assign(rdata, read);
            m.assign(low, rrtl_core::trunc(wdata, 4));
        }

        let imported = reimport_emitted(&design, "RegFile");
        let mut sim = Simulator::new(&imported.design, "RegFile").unwrap();
        let we = signal(&imported, "RegFile", "we");
        let waddr = signal(&imported, "RegFile", "waddr");
        let wdata = signal(&imported, "RegFile", "wdata");
        let raddr = signal(&imported, "RegFile", "raddr");
        let rdata = signal(&imported, "RegFile", "rdata");
        let low = signal(&imported, "RegFile", "low");
        sim.set(wdata, 0xab);
        assert_eq!(sim.get(low), 0xb);
        sim.set(we, 1);
        sim.set(waddr, 2);
        sim.tick();
        sim.set(we, 0);
        sim.set(raddr, 2);
        assert_eq!(sim.get(rdata), 0x0b);
    }

    #[test]
    fn emitted_sv_can_be_reimported_for_supported_subset() {
        let imported = import_sv(
            r#"
            module Alu(a, b, y);
              input logic [3:0] a, b;
              output logic [3:0] y;
              assign y = (a == b) ? a : (a + b);
            endmodule
            "#,
            None,
        )
        .unwrap();
        let emitted = rrtl_sv::emit(&imported.design).unwrap();
        let reimported = import_sv(&emitted, Some("Alu")).unwrap();
        let compiled = compile(&reimported.design).unwrap();
        assert!(compiled.find_module("Alu").is_some());
    }

    #[test]
    fn rejects_multiple_modules_without_top() {
        let err = import_sv(
            "module A(a); input logic a; endmodule module B(b); input logic b; endmodule",
            None,
        )
        .unwrap_err();
        assert_eq!(err.diagnostics[0].code, "E_SV_TOP");
    }
