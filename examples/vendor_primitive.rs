use rrtl_core::{uint, Design};

fn main() {
    let mut design = Design::new();
    {
        let mut ext = design.extern_module("SB_PLL40_CORE");
        ext.input("REFERENCECLK", uint(1));
        ext.input("RESETB", uint(1));
        ext.output("PLLOUTCORE", uint(1));
    }
    {
        let mut m = design.module("ClockTop");
        let refclk = m.input("refclk", uint(1));
        let reset_n = m.input("reset_n", uint(1));
        let pll_clk = m.output("pll_clk", uint(1));
        m.instance(
            "u_pll",
            "SB_PLL40_CORE",
            [
                ("REFERENCECLK", refclk),
                ("RESETB", reset_n),
                ("PLLOUTCORE", pll_clk),
            ],
        );
    }

    design.validate().unwrap();
    println!("{}", rrtl_sv::emit(&design).unwrap());
}
