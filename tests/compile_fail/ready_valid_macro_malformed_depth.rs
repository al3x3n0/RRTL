use rrtl::{ready_valid, signals, Design};

fn main() {
    let mut design = Design::new();
    let mut m = design.module("BadReadyValidDepth");
    let (clk, rst, input, output) = signals! {
        m {
            input clk: uint(1),
            input rst: uint(1),
            rv_sink input: scalar uint(8),
            rv_source output: scalar uint(8),
        }
    };

    ready_valid! {
        m {
            fifo fifo: input => output, clk, rst, size 3;
        }
    }

    let _ = (clk, rst, input, output);
}
