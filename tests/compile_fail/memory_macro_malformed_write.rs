use rrtl::{logic, signals, Design};

fn main() {
    let mut design = Design::new();
    let mut m = design.module("BadMemoryWrite");
    let (clk, we, waddr, wdata, regs) = signals! {
        m {
            input clk: uint(1),
            input we: uint(1),
            input waddr: uint(2),
            input wdata: uint(8),
            mem regs: addr(2), data uint(8), depth 4,
        }
    };

    logic! {
        m {
            mem_write regs = clk, we, waddr, wdata;
        }
    }

    let _ = (clk, we, waddr, wdata, regs);
}
