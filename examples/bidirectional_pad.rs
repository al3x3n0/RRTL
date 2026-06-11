use rrtl_core::{uint, Design};

fn main() {
    let mut design = Design::new();
    {
        let mut ext = design.extern_module("IOBUF");
        ext.inout("PAD", uint(1));
        ext.input("I", uint(1));
        ext.input("T", uint(1));
        ext.output("O", uint(1));
    }
    {
        let mut m = design.module("PadTop");
        let pad = m.inout("pad", uint(1));
        let drive_value = m.input("drive_value", uint(1));
        let output_enable_n = m.input("output_enable_n", uint(1));
        let sampled_value = m.output("sampled_value", uint(1));
        m.instance(
            "u_pad",
            "IOBUF",
            [
                ("PAD", pad),
                ("I", drive_value),
                ("T", output_enable_n),
                ("O", sampled_value),
            ],
        );
    }

    design.validate().unwrap();
    println!("{}", rrtl_sv::emit(&design).unwrap());
}
