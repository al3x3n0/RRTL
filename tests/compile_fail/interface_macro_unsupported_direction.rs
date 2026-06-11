fn main() {
    let _ = rrtl::interface! {
        Bad {
            sink req: uint(1),
        }
    };
}
