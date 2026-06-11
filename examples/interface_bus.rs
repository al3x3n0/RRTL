use rrtl_core::{
    bundle_type, field, iface_input, iface_output, interface_type, nested, uint, zext, Design,
    Simulator,
};

fn main() {
    let req_ty = bundle_type(
        "Req",
        [
            field("valid", uint(1)),
            field("addr", uint(8)),
            nested("meta", bundle_type("ReqMeta", [field("tag", uint(4))])),
        ],
    );
    let resp_ty = bundle_type("Resp", [field("valid", uint(1)), field("data", uint(8))]);
    let bus_ty = interface_type(
        "Bus",
        [
            iface_input("req", req_ty),
            iface_output("resp", resp_ty),
            iface_input("ready", uint(1)),
        ],
    );

    let mut design = Design::new();
    {
        let mut m = design.module("Responder");
        let bus = m.interface("bus", bus_ty.clone());

        m.assign(
            bus.field("resp", "valid").unwrap(),
            bus.field("req", "valid").unwrap(),
        );
        m.assign(
            bus.field("resp", "data").unwrap(),
            zext(bus.path("req", ["meta", "tag"]).unwrap(), 8),
        );
    }

    let (req_valid, req_tag, resp_valid, resp_data);
    {
        let mut m = design.module("Top");
        let bus = m.interface("bus", bus_ty);

        req_valid = bus.field("req", "valid").unwrap();
        req_tag = bus.path("req", ["meta", "tag"]).unwrap();
        resp_valid = bus.field("resp", "valid").unwrap();
        resp_data = bus.field("resp", "data").unwrap();

        m.instance_interfaces("u_responder", "Responder", [("bus", bus)]);
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "Top").unwrap();
    sim.set(req_valid, 1);
    sim.set(req_tag, 0xd);
    assert_eq!(sim.get(resp_valid), 1);
    assert_eq!(sim.get(resp_data), 0xd);

    println!("{}", rrtl_sv::emit(&design).unwrap());
}
