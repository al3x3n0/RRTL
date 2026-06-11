use rrtl::{interface, zext, Design, Simulator};

#[test]
fn interface_macro_builds_nested_bundle_interface_and_emits_sv() {
    let bus_ty = interface! {
        Bus {
            input req: Req {
                valid: uint(1),
                addr: uint(8),
                meta: ReqMeta {
                    tag: uint(4),
                },
            },
            output resp: Resp {
                valid: uint(1),
                data: uint(8),
            },
            input ready: uint(1),
        }
    };

    let mut design = Design::new();
    {
        let mut m = design.module("Responder");
        let bus = m.interface("bus", bus_ty.clone());
        let req = bus.port("req").unwrap().bundle().unwrap().clone();
        let resp = bus.port("resp").unwrap().bundle().unwrap().clone();

        m.assign(resp.field("valid").unwrap(), req.field("valid").unwrap());
        m.assign(
            resp.field("data").unwrap(),
            zext(req.path(["meta", "tag"]).unwrap(), 8),
        );
    }

    let (req_valid, req_tag, resp_valid, resp_data);
    {
        let mut m = design.module("Top");
        let bus = m.interface("bus", bus_ty);
        let req = bus.port("req").unwrap().bundle().unwrap().clone();
        let resp = bus.port("resp").unwrap().bundle().unwrap().clone();

        req_valid = req.field("valid").unwrap();
        req_tag = req.path(["meta", "tag"]).unwrap();
        resp_valid = resp.field("valid").unwrap();
        resp_data = resp.field("data").unwrap();

        m.instance_interfaces("u_responder", "Responder", [("bus", bus)]);
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "Top").unwrap();
    sim.set(req_valid, 1);
    sim.set(req_tag, 0xc);
    assert_eq!(sim.get(resp_valid), 1);
    assert_eq!(sim.get(resp_data), 0xc);

    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("module Top"));
    assert!(sv.contains("input logic bus_req_valid"));
    assert!(sv.contains("output logic [7:0] bus_resp_data"));
}

#[test]
fn interface_macro_matches_manual_interface_type() {
    let from_macro = interface! {
        Bus {
            input req: Req {
                valid: uint(1),
                meta: ReqMeta {
                    tag: uint(4),
                },
            },
            output resp: Resp {
                data: uint(8),
            },
            input ready: sint(1),
        }
    };

    let manual = rrtl::interface_type(
        "Bus",
        [
            rrtl::iface_input(
                "req",
                rrtl::bundle_type(
                    "Req",
                    [
                        rrtl::field("valid", rrtl::uint(1)),
                        rrtl::nested(
                            "meta",
                            rrtl::bundle_type("ReqMeta", [rrtl::field("tag", rrtl::uint(4))]),
                        ),
                    ],
                ),
            ),
            rrtl::iface_output(
                "resp",
                rrtl::bundle_type("Resp", [rrtl::field("data", rrtl::uint(8))]),
            ),
            rrtl::iface_input("ready", rrtl::sint(1)),
        ],
    );

    assert_eq!(from_macro, manual);
}
