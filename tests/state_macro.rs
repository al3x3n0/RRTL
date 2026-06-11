use rrtl::{logic, signals, state, Design, Simulator};

#[test]
fn state_macro_matches_manual_state_type() {
    let from_macro = state! {
        ControllerState: uint(2) {
            Idle = 0,
            Run = 1,
            Done = 2,
        }
    };

    let manual = rrtl::state_type(
        "ControllerState",
        rrtl::uint(2),
        [("Idle", 0), ("Run", 1), ("Done", 2)],
    );

    assert_eq!(from_macro, manual);
}

#[test]
fn state_macro_builds_controller_fsm() {
    let mut design = Design::new();
    let (rst, start, done, busy, finished);

    {
        let mut m = design.module("StateMacroController");
        let (clk, rst_local, start_local, done_local, busy_local, finished_local) = signals! {
            m {
                input clk: uint(1),
                input rst: uint(1),
                input start: uint(1),
                input done: uint(1),
                output busy: uint(1),
                output finished: uint(1),
            }
        };

        rst = rst_local;
        start = start_local;
        done = done_local;
        busy = busy_local;
        finished = finished_local;

        let states = state! {
            ControllerState: uint(2) {
                Idle = 0,
                Run = 1,
                Done = 2,
            }
        };
        let flow = m.state_reg("flow", states, clk, rst, "Idle");

        logic! {
            m {
                state_next_hold flow {
                    start.value() => Run,
                    done.value() => Done,
                };
                assign busy = flow.is("Run");
                assign finished = flow.is("Done");
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "StateMacroController").unwrap();
    sim.set(rst, 1);
    sim.tick();
    sim.set(rst, 0);

    sim.set(start, 1);
    sim.tick();
    assert_eq!(sim.get(busy), 1);

    sim.set(start, 0);
    sim.set(done, 1);
    sim.tick();
    assert_eq!(sim.get(finished), 1);

    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("typedef enum logic [1:0]"));
    assert!(sv.contains("CONTROLLERSTATE_RUN"));
}

#[test]
fn logic_macro_supports_state_next_default() {
    let mut design = Design::new();
    let (rst, start, busy);

    {
        let mut m = design.module("StateMacroDefault");
        let (clk, rst_local, start_local, busy_local) = signals! {
            m {
                input clk: uint(1),
                input rst: uint(1),
                input start: uint(1),
                output busy: uint(1),
            }
        };

        rst = rst_local;
        start = start_local;
        busy = busy_local;

        let states = state! {
            ControllerState: uint(1) {
                Idle = 0,
                Run = 1,
            }
        };
        let flow = m.state_reg("flow", states, clk, rst, "Idle");

        logic! {
            m {
                state_next flow default Idle {
                    start.value() => Run,
                };
                assign busy = flow.is("Run");
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "StateMacroDefault").unwrap();
    sim.set(rst, 1);
    sim.tick();
    sim.set(rst, 0);
    sim.set(start, 1);
    sim.tick();
    assert_eq!(sim.get(busy), 1);
    sim.set(start, 0);
    sim.tick();
    assert_eq!(sim.get(busy), 0);
}
