use rrtl::{bundle, instances, interface, signals, zext, Design, Simulator};

#[test]
fn instances_macro_instantiates_scalar_child() {
    let mut design = Design::new();

    {
        let mut m = design.module("ScalarChild");
        let (a, y) = signals! {
            m {
                input a: uint(8),
                output y: uint(8),
            }
        };

        m.assign(y, a + rrtl::lit_u(1, 8));
    }

    let (a, y);
    {
        let mut m = design.module("ScalarTop");
        let (a_local, y_local) = signals! {
            m {
                input a: uint(8),
                output y: uint(8),
            }
        };
        a = a_local;
        y = y_local;

        instances! {
            m {
                instance u_child: ScalarChild {
                    a: a,
                    y: y,
                },
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "ScalarTop").unwrap();
    sim.set(a, 0x2a);
    assert_eq!(sim.get(y), 0x2b);

    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("ScalarChild u_child (.a(a), .y(y));"));
}

#[test]
fn instances_macro_instantiates_bundle_child() {
    let req_ty = bundle! {
        Req {
            valid: uint(1),
            meta: ReqMeta {
                tag: uint(4),
            },
        }
    };
    let resp_ty = bundle! {
        Resp {
            valid: uint(1),
            data: uint(8),
        }
    };

    let mut design = Design::new();
    {
        let mut m = design.module("BundleChild");
        let req = m.input_bundle("req", req_ty.clone());
        let resp = m.output_bundle("resp", resp_ty.clone());

        m.assign(resp.field("valid").unwrap(), req.field("valid").unwrap());
        m.assign(
            resp.field("data").unwrap(),
            zext(req.path(["meta", "tag"]).unwrap(), 8),
        );
    }

    let (req_valid, req_tag, resp_valid, resp_data);
    {
        let mut m = design.module("BundleTop");
        let req = m.input_bundle("req", req_ty);
        let resp = m.output_bundle("resp", resp_ty);

        req_valid = req.field("valid").unwrap();
        req_tag = req.path(["meta", "tag"]).unwrap();
        resp_valid = resp.field("valid").unwrap();
        resp_data = resp.field("data").unwrap();

        instances! {
            m {
                instance_bundles u_child: BundleChild {
                    req: req,
                    resp: resp,
                },
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "BundleTop").unwrap();
    sim.set(req_valid, 1);
    sim.set(req_tag, 0xb);
    assert_eq!(sim.get(resp_valid), 1);
    assert_eq!(sim.get(resp_data), 0xb);

    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("BundleChild u_child"));
    assert!(sv.contains(".req_valid(req_valid)"));
    assert!(sv.contains(".resp_data(resp_data)"));
}

#[test]
fn instances_macro_instantiates_interface_child() {
    let bus_ty = interface! {
        Bus {
            input req: Req {
                valid: uint(1),
                meta: ReqMeta {
                    tag: uint(4),
                },
            },
            output resp: Resp {
                valid: uint(1),
                data: uint(8),
            },
        }
    };

    let mut design = Design::new();
    {
        let mut m = design.module("InterfaceChild");
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
        let mut m = design.module("InterfaceTop");
        let bus = m.interface("bus", bus_ty);
        let req = bus.port("req").unwrap().bundle().unwrap().clone();
        let resp = bus.port("resp").unwrap().bundle().unwrap().clone();

        req_valid = req.field("valid").unwrap();
        req_tag = req.path(["meta", "tag"]).unwrap();
        resp_valid = resp.field("valid").unwrap();
        resp_data = resp.field("data").unwrap();

        instances! {
            m {
                instance_interfaces u_child: InterfaceChild {
                    bus: bus,
                },
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "InterfaceTop").unwrap();
    sim.set(req_valid, 1);
    sim.set(req_tag, 0xd);
    assert_eq!(sim.get(resp_valid), 1);
    assert_eq!(sim.get(resp_data), 0xd);

    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("InterfaceChild u_child"));
    assert!(sv.contains(".bus_req_valid(bus_req_valid)"));
    assert!(sv.contains(".bus_resp_data(bus_resp_data)"));
}
