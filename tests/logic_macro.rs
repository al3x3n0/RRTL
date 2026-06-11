use rrtl::{lit_u, logic, mux, signals, Design, Simulator};

#[test]
fn logic_macro_builds_counter_body() {
    let mut design = Design::new();
    let (rst, en, out);

    {
        let mut m = design.module("LogicCounter");
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

        logic! {
            m {
                clock count: clk;
                reset count: rst = 0;
                next count = mux(en, count + lit_u(1, 8), count);
                assign out = count;
            }
        }
    }

    design.validate().unwrap();

    let mut sim = Simulator::new(&design, "LogicCounter").unwrap();
    sim.set(rst, 1);
    sim.tick();
    sim.set(rst, 0);
    sim.set(en, 1);
    sim.tick();
    sim.tick();
    assert_eq!(sim.get(out), 2);

    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("module LogicCounter"));
    assert!(sv.contains("assign out = count"));
}

#[test]
fn logic_macro_expands_reset_variants() {
    let mut design = Design::new();

    {
        let mut m = design.module("LogicResets");
        let (clk, rst, rst_n, arst, arst_n, sync_low, async_high, async_low) = signals! {
            m {
                input clk: uint(1),
                input rst: uint(1),
                input rst_n: uint(1),
                input arst: uint(1),
                input arst_n: uint(1),
                reg sync_low: uint(4),
                reg async_high: uint(4),
                reg async_low: uint(4),
            }
        };

        logic! {
            m {
                clock sync_low: clk;
                reset_low sync_low: rst_n = 0;
                next sync_low = sync_low + lit_u(1, 4);
                clock async_high: clk;
                async_reset async_high: arst = 0;
                next async_high = async_high + lit_u(1, 4);
                clock async_low: clk;
                async_reset_low async_low: arst_n = 0;
                next async_low = async_low + lit_u(1, 4);
            }
        }

        let _ = rst;
    }

    design.validate().unwrap();
    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("if (!rst_n) sync_low <= 4'd0"));
    assert!(sv.contains("posedge arst"));
    assert!(sv.contains("negedge arst_n"));
}

#[test]
fn logic_macro_expands_assertions() {
    let mut design = Design::new();
    let value;

    {
        let mut m = design.module("LogicAssertions");
        let (clk, en, value_local) = signals! {
            m {
                input clk: uint(1),
                input en: uint(1),
                input value: uint(4),
            }
        };
        value = value_local;

        logic! {
            m {
                assert value_small: value.value().lt_expr(lit_u(10, 4));
                assert_when enabled_value_small: en, value.value().lt_expr(lit_u(10, 4));
                assert_clocked clocked_value_small: clk, value.value().lt_expr(lit_u(10, 4));
                assert_msg value_message: value.value().lt_expr(lit_u(10, 4)), "value too large";
            }
        }
    }

    design.validate().unwrap();
    let mut sim = Simulator::new(&design, "LogicAssertions").unwrap();
    sim.set(value, 9);
    sim.check_assertions().unwrap();
    sim.set(value, 10);
    assert!(sim.check_assertions().is_err());

    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("VALUE_SMALL: assert"));
    assert!(sv.contains("CLOCKED_VALUE_SMALL: assert"));
    assert!(sv.contains("value too large"));
}

#[test]
fn logic_macro_expands_cover_points() {
    let mut design = Design::new();
    let value;

    {
        let mut m = design.module("LogicCovers");
        let (clk, en, value_local) = signals! {
            m {
                input clk: uint(1),
                input en: uint(1),
                input value: uint(4),
            }
        };
        value = value_local;

        logic! {
            m {
                cover value_three: value.value().eq_expr(lit_u(3, 4));
                cover_when enabled_value_three: en, value.value().eq_expr(lit_u(3, 4));
                cover_clocked clocked_value_three: clk, value.value().eq_expr(lit_u(3, 4));
                cover_msg value_message: value.value().eq_expr(lit_u(3, 4)), "value reached three";
            }
        }
    }

    design.validate().unwrap();
    let mut sim = Simulator::new(&design, "LogicCovers").unwrap();
    sim.set(value, 3);
    sim.check_assertions().unwrap();
    assert_eq!(sim.cover_hits("LogicCovers.value_three"), 1);
    assert_eq!(sim.cover_hits("LogicCovers.value_message"), 1);

    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("VALUE_THREE: cover"));
    assert!(sv.contains("CLOCKED_VALUE_THREE: cover"));
}

#[test]
fn logic_macro_expands_bundle_assignments() {
    let mut design = Design::new();
    let (en, req_valid, req_tag, resp_valid, resp_tag, gated_valid, gated_tag);

    {
        let mut m = design.module("LogicBundleAssign");
        let (en_local, req, resp, gated) = signals! {
            m {
                input en: uint(1),
                input_bundle req: Req {
                    valid: uint(1),
                    meta: ReqMeta {
                        tag: uint(4),
                    },
                },
                output_bundle resp: Req {
                    valid: uint(1),
                    meta: ReqMeta {
                        tag: uint(4),
                    },
                },
                output_bundle gated: Req {
                    valid: uint(1),
                    meta: ReqMeta {
                        tag: uint(4),
                    },
                },
            }
        };
        en = en_local;
        req_valid = req.field("valid").unwrap();
        req_tag = req.path(["meta", "tag"]).unwrap();
        resp_valid = resp.field("valid").unwrap();
        resp_tag = resp.path(["meta", "tag"]).unwrap();
        gated_valid = gated.field("valid").unwrap();
        gated_tag = gated.path(["meta", "tag"]).unwrap();

        logic! {
            m {
                assign_bundle resp = req;
                assign_bundle_when gated: en, req;
            }
        }
    }

    design.validate().unwrap();
    let mut sim = Simulator::new(&design, "LogicBundleAssign").unwrap();
    sim.set(en, 1);
    sim.set(req_valid, 1);
    sim.set(req_tag, 0xa);
    assert_eq!(sim.get(resp_valid), 1);
    assert_eq!(sim.get(resp_tag), 0xa);
    assert_eq!(sim.get(gated_valid), 1);
    assert_eq!(sim.get(gated_tag), 0xa);

    sim.set(en, 0);
    sim.set(req_valid, 0);
    sim.set(req_tag, 0x3);
    assert_eq!(sim.get(resp_valid), 0);
    assert_eq!(sim.get(resp_tag), 0x3);
    assert_eq!(sim.get(gated_valid), 1);
    assert_eq!(sim.get(gated_tag), 0xa);
}
