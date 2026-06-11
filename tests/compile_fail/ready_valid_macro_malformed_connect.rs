use rrtl::{ready_valid, signals, Design};

fn main() {
    let mut design = Design::new();
    let mut m = design.module("BadReadyValidConnect");
    let (_input, _output) = signals! {
        m {
            rv_sink input: scalar uint(8),
            rv_source output: scalar uint(8),
        }
    };
    ready_valid! {
        m {
            connect input: output;
        }
    }
}
