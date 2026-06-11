use rrtl_core::{lit_u, mux, uint, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (sync_rst_n, async_rst_n, en, sync_out, async_out);
    {
        let mut m = design.module("ActiveLowReset");
        let clk = m.input("clk", uint(1));
        sync_rst_n = m.input("sync_rst_n", uint(1));
        async_rst_n = m.input("async_rst_n", uint(1));
        en = m.input("en", uint(1));
        sync_out = m.output("sync_out", uint(4));
        async_out = m.output("async_out", uint(4));
        let sync_count = m.reg("sync_count", uint(4));
        let async_count = m.reg("async_count", uint(4));

        m.clock(sync_count, clk);
        m.reset_low(sync_count, sync_rst_n, 0);
        m.next(sync_count, mux(en, sync_count + lit_u(1, 4), sync_count));
        m.assign(sync_out, sync_count);

        m.clock(async_count, clk);
        m.async_reset_low(async_count, async_rst_n, 0);
        m.next(async_count, mux(en, async_count + lit_u(1, 4), async_count));
        m.assign(async_out, async_count);
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "ActiveLowReset").unwrap();
    sim.set(sync_rst_n, 1);
    sim.set(async_rst_n, 1);
    sim.set(en, 1);
    sim.tick();
    assert_eq!(sim.get(sync_out), 1);
    assert_eq!(sim.get(async_out), 1);

    sim.set(sync_rst_n, 0);
    assert_eq!(sim.get(sync_out), 1);
    sim.tick();
    assert_eq!(sim.get(sync_out), 0);

    sim.set(async_rst_n, 0);
    assert_eq!(sim.get(async_out), 0);

    println!("{}", rrtl_sv::emit(&design).unwrap());
}
