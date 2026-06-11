fn main() {
    let mut design = rrtl::Design::new();
    let mut m = design.module("Bad");
    let _ = rrtl::signals! {
        m {
            sink clk: uint(1),
        }
    };
}
