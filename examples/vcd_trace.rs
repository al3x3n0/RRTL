use rrtl_core::{lit_u, mux, uint, Design, Simulator, VcdTrace};

fn main() {
    let mut design = Design::new();
    let (rst, en, out);

    {
        let mut m = design.module("Counter");
        let clk = m.input("clk", uint(1));
        rst = m.input("rst", uint(1));
        en = m.input("en", uint(1));
        out = m.output("out", uint(8));
        let count = m.reg("count", uint(8));

        m.clock(count, clk);
        m.reset(count, rst, 0);
        m.next(count, mux(en, count + lit_u(1, 8), count));
        m.assign(out, count);
    }

    let mut sim = Simulator::new(&design, "Counter").unwrap();
    let mut trace = VcdTrace::new(&design, "Counter").unwrap();

    trace.sample(&sim, 0).unwrap();
    sim.set(rst, 1);
    sim.tick();
    trace.sample(&sim, 1).unwrap();
    sim.set(rst, 0);
    sim.set(en, 1);
    sim.tick();
    trace.sample(&sim, 2).unwrap();
    sim.tick();
    trace.sample(&sim, 3).unwrap();

    assert_eq!(sim.get(out), 2);
    println!("{}", trace.finish());
}
