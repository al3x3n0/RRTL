fn main() {
    let mut design = rrtl::Design::new();
    let mut m = design.module("Bad");
    let _ = rrtl::logic! {
        m {
            reset count = rst = 0;
        }
    };
}
