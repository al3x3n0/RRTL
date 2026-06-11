use rrtl::{signals, Design};

fn main() {
    let mut design = Design::new();
    let mut m = design.module("BadMemoryDecl");
    let _ = signals! {
        m {
            mem regs: data uint(8), depth 4,
        }
    };
}
