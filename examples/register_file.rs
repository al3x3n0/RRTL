use rrtl_core::{uint, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (we, waddr, wdata, raddr, rdata);

    {
        let mut m = design.module("RegFile");
        let clk = m.input("clk", uint(1));
        we = m.input("we", uint(1));
        waddr = m.input("waddr", uint(2));
        wdata = m.input("wdata", uint(8));
        raddr = m.input("raddr", uint(2));
        rdata = m.output("rdata", uint(8));
        let regs = m.mem("regs", 2, uint(8), 4);

        m.mem_write(regs, clk, we, waddr, wdata);
        let read_data = m.mem_read(regs, raddr);
        m.assign(rdata, read_data);
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "RegFile").unwrap();
    sim.set(we, 1);
    sim.set(waddr, 2);
    sim.set(wdata, 0x5a);
    sim.tick();
    sim.set(raddr, 2);
    assert_eq!(sim.get(rdata), 0x5a);

    println!("{}", rrtl_sv::emit(&design).unwrap());
}
