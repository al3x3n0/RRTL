use rrtl::{extern_module, instances, signals, Design};

#[test]
fn extern_module_macro_declares_vendor_primitive() {
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
    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(!sv.contains("module SB_PLL40_CORE"));
    assert!(sv.contains("module ClockTop(refclk, reset_n, pll_clk);"));
    assert!(sv.contains(
        "SB_PLL40_CORE u_pll (.REFERENCECLK(refclk), .RESETB(reset_n), .PLLOUTCORE(pll_clk));"
    ));
}

#[test]
fn extern_module_macro_declares_inout_vendor_primitive() {
    let mut design = Design::new();

    extern_module! {
        design IOBUF {
            inout PAD: uint(1),
            input I: uint(1),
            input T: uint(1),
            output O: uint(1),
        }
    }

    {
        let mut m = design.module("PadTop");
        let pad = m.inout("pad", rrtl::uint(1));
        let (drive_value, output_enable_n, sampled_value) = signals! {
            m {
                input drive_value: uint(1),
                input output_enable_n: uint(1),
                output sampled_value: uint(1),
            }
        };

        instances! {
            m {
                instance u_pad: IOBUF {
                    PAD: pad,
                    I: drive_value,
                    T: output_enable_n,
                    O: sampled_value,
                },
            }
        }
    }

    design.validate().unwrap();
    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(!sv.contains("module IOBUF"));
    assert!(sv.contains("module PadTop(pad, drive_value, output_enable_n, sampled_value);"));
    assert!(sv.contains("inout wire pad;"));
    assert!(sv.contains(
        "IOBUF u_pad (.PAD(pad), .I(drive_value), .T(output_enable_n), .O(sampled_value));"
    ));
}
