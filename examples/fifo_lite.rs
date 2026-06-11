use rrtl_core::{lit_u, mux, uint, Design};

fn main() {
    let mut design = Design::new();

    {
        let mut m = design.module("FifoLite");
        let clk = m.input("clk", uint(1));
        let rst = m.input("rst", uint(1));
        let write_en = m.input("write_en", uint(1));
        let read_en = m.input("read_en", uint(1));
        let din = m.input("din", uint(8));
        let dout = m.output("dout", uint(8));
        let empty = m.output("empty", uint(1));

        let mem = m.mem("storage", 2, uint(8), 4);
        let wptr = m.reg("wptr", uint(2));
        let rptr = m.reg("rptr", uint(2));

        m.clock(wptr, clk);
        m.reset(wptr, rst, 0);
        m.next(wptr, mux(write_en, wptr + lit_u(1, 2), wptr));

        m.clock(rptr, clk);
        m.reset(rptr, rst, 0);
        m.next(rptr, mux(read_en, rptr + lit_u(1, 2), rptr));

        m.mem_write(mem, clk, write_en, wptr, din);
        let read_data = m.mem_read(mem, rptr);
        m.assign(dout, read_data);
        m.assign(empty, wptr.value().eq_expr(rptr));
    }

    design.validate().unwrap();
    println!("{}", rrtl_sv::emit(&design).unwrap());
}
