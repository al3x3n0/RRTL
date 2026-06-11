use rrtl::{extern_module, instances, signals, Design};

fn main() {
    let mut design = Design::new();

    extern_module! {
        design SB_PLL40_CORE {
            input REFERENCECLK: uint(1),
            input RESETB: uint(1),
            output PLLOUTCORE: uint(1),
        }
    }

    {
        let mut m = design.module("ClockTop");
        let (refclk, reset_n, pll_clk) = signals! {
            m {
                input refclk: uint(1),
                input reset_n: uint(1),
                output pll_clk: uint(1),
            }
        };

        instances! {
            m {
                instance u_pll: SB_PLL40_CORE {
                    REFERENCECLK: refclk,
                    RESETB: reset_n,
                    PLLOUTCORE: pll_clk,
                },
            }
        }
    }

    design.validate().unwrap();
    println!("{}", rrtl::sv::emit(&design).unwrap());
}
