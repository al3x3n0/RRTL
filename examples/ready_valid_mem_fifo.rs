use rrtl_core::{rv_scalar, uint, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);
    {
        let mut m = design.module("ReadyValidMemFifo");
        let clk = m.input("clk", uint(1));
        let rst = m.input("rst", uint(1));
        let input = m.rv_sink("in", rv_scalar(uint(8)));
        let output = m.rv_source("out", rv_scalar(uint(8)));
        in_valid = input.valid();
        in_ready = input.ready();
        in_bits = input.bits_signal().unwrap();
        out_valid = output.valid();
        out_ready = output.ready();
        out_bits = output.bits_signal().unwrap();

        m.rv_mem_fifo("fifo", &input, &output, clk, rst, 4);
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "ReadyValidMemFifo").unwrap();
    assert_eq!(sim.get(in_ready), 1);
    assert_eq!(sim.get(out_valid), 0);

    sim.set(out_ready, 0);
    sim.set(in_valid, 1);
    for value in [0x10, 0x20, 0x30, 0x40] {
        sim.set(in_bits, value);
        sim.tick();
    }
    assert_eq!(sim.get(in_ready), 0);
    assert_eq!(sim.get(out_valid), 1);
    assert_eq!(sim.get(out_bits), 0x10);

    sim.set(in_valid, 0);
    sim.set(out_ready, 1);
    for value in [0x20, 0x30, 0x40] {
        sim.tick();
        assert_eq!(sim.get(out_bits), value);
    }

    println!("{}", rrtl_sv::emit(&design).unwrap());
}
