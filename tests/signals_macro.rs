use rrtl::{interface, signals, zext, Design, Simulator};

#[test]
fn signals_macro_declares_counter_handles() {
    let mut design = Design::new();
    let (rst, en, out);
    {
        let mut m = design.module("SignalsCounter");
        let (clk, rst_local, en_local, out_local, count) = signals! {
            m {
                input clk: uint(1),
                input rst: uint(1),
                input en: uint(1),
                output out: uint(8),
                reg count: uint(8),
            }
        };

        rst = rst_local;
        en = en_local;
        out = out_local;

        m.clock(count, clk);
        m.reset(count, rst, 0);
        m.next(count, rrtl::mux(en, count + rrtl::lit_u(1, 8), count));
        m.assign(out, count);
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "SignalsCounter").unwrap();
    sim.set(rst, 1);
    sim.tick();
    sim.set(rst, 0);
    sim.set(en, 1);
    sim.tick();
    sim.tick();
    assert_eq!(sim.get(out), 2);

    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("module SignalsCounter"));
    assert!(sv.contains("output logic [7:0] out"));
}

#[test]
fn signals_macro_declares_structured_handles() {
    let mut design = Design::new();
    {
        let mut m = design.module("SignalsStructured");
        let (req, resp, tmp, bus) = signals! {
            m {
                input_bundle req: Req {
                    valid: uint(1),
                    meta: ReqMeta {
                        tag: uint(4),
                    },
                },
                output_bundle resp: Resp {
                    valid: uint(1),
                    data: uint(8),
                },
                wire_bundle tmp: Tmp {
                    data: uint(8),
                },
                interface bus: Bus {
                    input ready: uint(1),
                    output data: uint(8),
                },
            }
        };

        m.assign(resp.field("valid").unwrap(), req.field("valid").unwrap());
        m.assign(
            resp.field("data").unwrap(),
            zext(req.path(["meta", "tag"]).unwrap(), 8),
        );
        m.assign(tmp.field("data").unwrap(), resp.field("data").unwrap());
        m.assign(
            bus.port("data").unwrap().signal().unwrap(),
            tmp.field("data").unwrap(),
        );
    }

    design.validate().unwrap();
    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("input logic req_valid"));
    assert!(sv.contains("output logic [7:0] bus_data"));
}

#[test]
fn signals_macro_accepts_interface_macro_types_separately() {
    let bus_ty = interface! {
        Bus {
            input ready: uint(1),
            output data: uint(8),
        }
    };

    let mut design = Design::new();
    {
        let mut m = design.module("SignalsInterfaceManual");
        let bus = m.interface("bus", bus_ty);
        let (_, data) = signals! {
            m {
                input ready_alias: uint(1),
                output data_alias: uint(8),
            }
        };

        m.assign(data, bus.port("data").unwrap().signal().unwrap());
    }

    design.validate().unwrap();
}
