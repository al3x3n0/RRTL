use rrtl::{extern_module, Design};

fn main() {
    let mut design = Design::new();

    extern_module! {
        design Vendor {
            clock CLK: uint(1),
        }
    }
}
