use rrtl_core::{rv_scalar, uint, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (rst, in_valid, in_ready, in_bits, out_valid, out_ready, out_bits);
    {
        let mut m = design.module("ReadyValidSlice");
        let clk = m.input("clk", uint(1));
        rst = m.input("rst", uint(1));
        let input = m.rv_sink("in", rv_scalar(uint(8)));
        let output = m.rv_source("out", rv_scalar(uint(8)));
        in_valid = input.valid();
        in_ready = input.ready();
        in_bits = input.bits_signal().unwrap();
        out_valid = output.valid();
        out_ready = output.ready();
        out_bits = output.bits_signal().unwrap();

        m.rv_register_slice("slice", &input, &output, clk, rst);
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "ReadyValidSlice").unwrap();
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

    println!("{}", rrtl_sv::emit(&design).unwrap());
}
