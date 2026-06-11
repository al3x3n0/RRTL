use rrtl::{instances, signals, Design, Simulator};

fn main() {
    let mut design = Design::new();

    {
        let mut m = design.module("AddOne");
        let (a, y) = signals! {
            m {
                input a: uint(8),
                output y: uint(8),
            }
        };

        m.assign(y, a + rrtl::lit_u(1, 8));
    }

    let (a, y);
    {
        let mut m = design.module("Top");
        let (a_local, y_local) = signals! {
            m {
                input a: uint(8),
                output y: uint(8),
            }
        };
        a = a_local;
        y = y_local;

        instances! {
            m {
                instance u_add_one: AddOne {
                    a: a,
                    y: y,
                },
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "Top").unwrap();
    sim.set(a, 0x2a);
    assert_eq!(sim.get(y), 0x2b);

    println!("{}", rrtl::sv::emit(&design).unwrap());
}
