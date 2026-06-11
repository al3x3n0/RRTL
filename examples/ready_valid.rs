use rrtl_core::{rv_scalar, uint, Design, Simulator};

fn main() {
    let mut design = Design::new();
    {
        let mut m = design.module("Consumer");
        let stream = m.rv_sink("stream", rv_scalar(uint(8)));
        m.assign(stream.ready(), stream.valid());
    }

    let (valid, ready, bits, accepted);
    {
        let mut m = design.module("Top");
        let stream = m.rv_sink("stream", rv_scalar(uint(8)));
        valid = stream.valid();
        ready = stream.ready();
        bits = stream.bits_signal().unwrap();
        accepted = m.output("accepted", uint(1));

        m.assign(accepted, stream.fire());
        m.instance_interfaces(
            "u_consumer",
            "Consumer",
            [("stream", stream.into_interface())],
        );
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "Top").unwrap();
    sim.set(valid, 1);
    sim.set(bits, 0x5a);
    assert_eq!(sim.get(ready), 1);
    assert_eq!(sim.get(accepted), 1);

    println!("{}", rrtl_sv::emit(&design).unwrap());
}
