use rrtl::{lit_u, logic, mux, signals, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (rst, en, out);

    {
        let mut m = design.module("LogicCounter");
        let (clk, rst_local, en_local, out_local, count) = signals! {
            m {
                input clk: uint(1),
                input rst: uint(1),
                input en: uint(1),
                output out: uint(8),
                reg count: uint(8),
            }
        };

        rst = rst_local;
        en = en_local;
        out = out_local;

        logic! {
            m {
                clock count: clk;
                reset count: rst = 0;
                next count = mux(en, count + lit_u(1, 8), count);
                assign out = count;
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "LogicCounter").unwrap();
    sim.set(rst, 1);
    sim.tick();
    sim.set(rst, 0);
    sim.set(en, 1);
    sim.tick();
    sim.tick();
    assert_eq!(sim.get(out), 2);

    println!("{}", rrtl::sv::emit(&design).unwrap());
}
