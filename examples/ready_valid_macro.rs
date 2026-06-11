use rrtl::{ready_valid, signals, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (rst, in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);

    {
        let mut m = design.module("ReadyValidMacro");
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

    let mut sim = Simulator::new(&design, "ReadyValidMacro").unwrap();
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

    println!("{}", rrtl::sv::emit(&design).unwrap());
}
