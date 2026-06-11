use rrtl_core::{lit_s, sext, sint, uint, Design, Simulator};

fn main() {
    let mut design = Design::new();
    let (a, b, sum, diff, lt, wide);

    {
        let mut m = design.module("SignedAlu");
        a = m.input("a", sint(8));
        b = m.input("b", sint(8));
        sum = m.output("sum", sint(8));
        diff = m.output("diff", sint(8));
        lt = m.output("lt", uint(1));
        wide = m.output("wide", sint(16));

        m.assign(sum, a + b);
        m.assign(diff, a - b);
        m.assign(lt, a.value().lt_expr(b));
        m.assign(wide, sext(a + lit_s(-1, 8), 16));
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "SignedAlu").unwrap();
    sim.set(a, 0xfe);
    sim.set(b, 0x01);
    assert_eq!(sim.get(sum), 0xff);
    assert_eq!(sim.get(diff), 0xfd);
    assert_eq!(sim.get(lt), 1);
    assert_eq!(sim.get(wide), 0xfffd);

    println!("{}", rrtl_sv::emit(&design).unwrap());
}
