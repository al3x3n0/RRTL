use rrtl_core::{lit_u, mux, uint, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (rst, start, done, busy);

    {
        let mut m = design.module("TinyFsm");
        let clk = m.input("clk", uint(1));
        rst = m.input("rst", uint(1));
        start = m.input("start", uint(1));
        done = m.input("done", uint(1));
        busy = m.output("busy", uint(1));
        let state = m.reg("state", uint(2));

        m.clock(state, clk);
        m.reset(state, rst, 0);
        m.next(
            state,
            mux(done, lit_u(0, 2), mux(start, lit_u(1, 2), state)),
        );
        m.assign(busy, state.value().eq_expr(lit_u(1, 2)));
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "TinyFsm").unwrap();
    sim.set(rst, 1);
    sim.tick();
    sim.set(rst, 0);
    sim.set(start, 1);
    sim.tick();
    assert_eq!(sim.get(busy), 1);
    sim.set(start, 0);
    sim.set(done, 1);
    sim.tick();
    assert_eq!(sim.get(busy), 0);

    println!("{}", rrtl_sv::emit(&design).unwrap());
}
