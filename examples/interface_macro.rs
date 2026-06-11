use rrtl::{interface, zext, Design, Simulator};

fn main() {
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
    sim.set(req_tag, 0xd);
    assert_eq!(sim.get(resp_valid), 1);
    assert_eq!(sim.get(resp_data), 0xd);

    println!("{}", rrtl::sv::emit(&design).unwrap());
}
