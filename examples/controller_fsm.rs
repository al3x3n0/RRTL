use rrtl_core::{state_type, uint, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (start, done, busy, finished);

    {
        let mut m = design.module("Controller");
        let clk = m.input("clk", uint(1));
        let rst = m.input("rst", uint(1));
        start = m.input("start", uint(1));
        done = m.input("done", uint(1));
        busy = m.output("busy", uint(1));
        finished = m.output("finished", uint(1));

        let states = state_type(
            "ControllerState",
            uint(2),
            [("Idle", 0), ("Run", 1), ("Done", 2)],
        );
        let state = m.state_reg("state", states, clk, rst, "Idle");
        m.state_next_hold(
            &state,
            [
                (start.value(), "Run".to_string()),
                (done.value(), "Done".to_string()),
            ],
        );
        m.assign(busy, state.is("Run"));
        m.assign(finished, state.is("Done"));
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "Controller").unwrap();
    sim.set(start, 1);
    sim.tick();
    assert_eq!(sim.get(busy), 1);

    sim.set(start, 0);
    sim.set(done, 1);
    sim.tick();
    assert_eq!(sim.get(finished), 1);

    println!("{}", rrtl_sv::emit(&design).unwrap());
}
