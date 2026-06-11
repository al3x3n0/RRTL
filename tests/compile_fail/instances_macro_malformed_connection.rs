use rrtl::{instances, Design};

fn main() {
    let mut design = Design::new();
    let mut m = design.module("BadInstanceConnection");

    instances! {
        m {
            instance u_child: Child {
                a = a,
            },
        }
    }
}
