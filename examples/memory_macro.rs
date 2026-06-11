use rrtl::{logic, signals, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (we, waddr, wdata, raddr, rdata);

    {
        let mut m = design.module("RegFile");
        let (clk, we_local, waddr_local, wdata_local, raddr_local, rdata_local, regs) = signals! {
            m {
                input clk: uint(1),
                input we: uint(1),
                input waddr: uint(2),
                input wdata: uint(8),
                input raddr: uint(2),
                output rdata: uint(8),
                mem regs: addr(2), data uint(8), depth 4,
            }
        };

        we = we_local;
        waddr = waddr_local;
        wdata = wdata_local;
        raddr = raddr_local;
        rdata = rdata_local;

        logic! {
            m {
                mem_write regs: clk, we, waddr, wdata;
                assign rdata = rrtl::mem_read!(m, regs, raddr);
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "RegFile").unwrap();
    sim.set(we, 1);
    sim.set(waddr, 2);
    sim.set(wdata, 0x5a);
    sim.tick();
    sim.set(raddr, 2);
    assert_eq!(sim.get(rdata), 0x5a);

    println!("{}", rrtl::sv::emit(&design).unwrap());
}
