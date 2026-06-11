use rrtl::{ready_valid, signals, Design, Simulator};

#[test]
fn ready_valid_macro_builds_scalar_connect() {
    let mut design = Design::new();
    let (in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);

    {
        let mut m = design.module("ReadyValidMacroConnect");
        let (input, output) = signals! {
            m {
                rv_sink input: scalar uint(8),
                rv_source output: scalar uint(8),
            }
        };

        in_valid = input.valid();
        in_ready = input.ready();
        in_bits = input.bits_signal().unwrap();
        out_valid = output.valid();
        out_ready = output.ready();
        out_bits = output.bits_signal().unwrap();

        ready_valid! {
            m {
                connect input => output;
            }
        }
    }

    design.validate().unwrap();
    let mut sim = Simulator::new(&design, "ReadyValidMacroConnect").unwrap();
    sim.set(in_valid, 1);
    sim.set(in_bits, 0x44);
    sim.set(out_ready, 1);
    assert_eq!(sim.get(out_valid), 1);
    assert_eq!(sim.get(out_bits), 0x44);
    assert_eq!(sim.get(in_ready), 1);
}

#[test]
fn ready_valid_macro_builds_scalar_register_slice() {
    let mut design = Design::new();
    let (rst, in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);

    {
        let mut m = design.module("ReadyValidMacroSlice");
        let (clk, rst_local, input, output) = signals! {
            m {
                input clk: uint(1),
                input rst: uint(1),
                rv_sink input: scalar uint(8),
                rv_source output: scalar uint(8),
            }
        };

        rst = rst_local;
        in_valid = input.valid();
        in_ready = input.ready();
        in_bits = input.bits_signal().unwrap();
        out_valid = output.valid();
        out_ready = output.ready();
        out_bits = output.bits_signal().unwrap();

        ready_valid! {
            m {
                register_slice slice: input => output, clk, rst;
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "ReadyValidMacroSlice").unwrap();
    sim.set(out_ready, 1);
    assert_eq!(sim.get(in_ready), 1);

    sim.set(in_valid, 1);
    sim.set(in_bits, 0x33);
    assert_eq!(sim.get(out_valid), 0);

    sim.tick();
    assert_eq!(sim.get(out_valid), 1);
    assert_eq!(sim.get(out_bits), 0x33);

    sim.set(rst, 1);
    sim.tick();
    assert_eq!(sim.get(out_valid), 0);
}

#[test]
fn ready_valid_macro_builds_scalar_skid_buffer() {
    let mut design = Design::new();
    let (in_valid, in_ready, in_bits, out_ready, out_bits);

    {
        let mut m = design.module("ReadyValidMacroSkid");
        let (clk, rst, input, output) = signals! {
            m {
                input clk: uint(1),
                input rst: uint(1),
                rv_sink input: scalar uint(8),
                rv_source output: scalar uint(8),
            }
        };

        in_valid = input.valid();
        in_ready = input.ready();
        in_bits = input.bits_signal().unwrap();
        out_ready = output.ready();
        out_bits = output.bits_signal().unwrap();

        ready_valid! {
            m {
                skid_buffer skid: input => output, clk, rst;
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "ReadyValidMacroSkid").unwrap();
    sim.set(out_ready, 1);
    sim.set(in_valid, 1);
    sim.set(in_bits, 0x11);
    assert_eq!(sim.get(in_ready), 1);
    assert_eq!(sim.get(out_bits), 0x11);

    sim.set(out_ready, 0);
    sim.set(in_bits, 0x22);
    assert_eq!(sim.get(in_ready), 1);
    sim.tick();
    assert_eq!(sim.get(in_ready), 0);
    assert_eq!(sim.get(out_bits), 0x22);
}

#[test]
fn ready_valid_macro_builds_scalar_fifo() {
    let mut design = Design::new();
    let (in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);

    {
        let mut m = design.module("ReadyValidMacroFifo");
        let (clk, rst, input, output) = signals! {
            m {
                input clk: uint(1),
                input rst: uint(1),
                rv_sink input: scalar uint(8),
                rv_source output: scalar uint(8),
            }
        };

        in_valid = input.valid();
        in_ready = input.ready();
        in_bits = input.bits_signal().unwrap();
        out_valid = output.valid();
        out_ready = output.ready();
        out_bits = output.bits_signal().unwrap();

        ready_valid! {
            m {
                fifo fifo: input => output, clk, rst, depth 3;
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "ReadyValidMacroFifo").unwrap();
    sim.set(out_ready, 0);
    sim.set(in_valid, 1);
    sim.set(in_bits, 0x10);
    sim.tick();
    assert_eq!(sim.get(out_valid), 1);
    assert_eq!(sim.get(out_bits), 0x10);

    sim.set(in_bits, 0x20);
    sim.tick();
    sim.set(in_bits, 0x30);
    sim.tick();
    assert_eq!(sim.get(in_ready), 0);
    assert_eq!(sim.get(out_bits), 0x10);

    sim.set(out_ready, 1);
    sim.tick();
    assert_eq!(sim.get(in_ready), 1);
    assert_eq!(sim.get(out_bits), 0x20);
}

#[test]
fn ready_valid_macro_builds_bundle_mem_fifo() {
    let mut design = Design::new();
    let (in_valid, in_ready, out_ready, out_valid, in_data, in_last, out_data, out_last);

    {
        let mut m = design.module("ReadyValidMacroBundleMemFifo");
        let (clk, rst, input, output) = signals! {
            m {
                input clk: uint(1),
                input rst: uint(1),
                rv_sink input: bundle Payload {
                    data: uint(8),
                    last: uint(1),
                },
                rv_source output: bundle Payload {
                    data: uint(8),
                    last: uint(1),
                },
            }
        };

        let input_bits = input.bits_bundle().unwrap();
        let output_bits = output.bits_bundle().unwrap();
        in_valid = input.valid();
        in_ready = input.ready();
        out_ready = output.ready();
        out_valid = output.valid();
        in_data = input_bits.field("data").unwrap();
        in_last = input_bits.field("last").unwrap();
        out_data = output_bits.field("data").unwrap();
        out_last = output_bits.field("last").unwrap();

        ready_valid! {
            m {
                mem_fifo fifo: input => output, clk, rst, depth 2;
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "ReadyValidMacroBundleMemFifo").unwrap();
    sim.set(out_ready, 0);
    sim.set(in_valid, 1);
    sim.set(in_data, 0xaa);
    sim.set(in_last, 0);
    sim.tick();
    assert_eq!(sim.get(out_valid), 1);
    assert_eq!(sim.get(out_data), 0xaa);
    assert_eq!(sim.get(out_last), 0);

    sim.set(in_data, 0xbb);
    sim.set(in_last, 1);
    sim.tick();
    assert_eq!(sim.get(in_ready), 0);
    assert_eq!(sim.get(out_data), 0xaa);
    assert_eq!(sim.get(out_last), 0);

    sim.set(out_ready, 1);
    sim.tick();
    assert_eq!(sim.get(in_ready), 1);
    assert_eq!(sim.get(out_data), 0xbb);
    assert_eq!(sim.get(out_last), 1);
}
