use rrtl::{instances, Design};

fn main() {
    let mut design = Design::new();
    let mut m = design.module("BadInstanceKind");

    instances! {
        m {
            connect u_child: Child {
                a: a,
            },
        }
    }
}
