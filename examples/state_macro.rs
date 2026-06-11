use rrtl::{logic, signals, state, Design, Simulator};

fn main() {
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

    println!("{}", rrtl::sv::emit(&design).unwrap());
}
