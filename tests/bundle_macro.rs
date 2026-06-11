use rrtl::{bundle, zext, Design};

#[test]
fn bundle_macro_builds_nested_bundle_ports_and_emits_sv() {
    let req_ty = bundle! {
        Req {
            valid: uint(1),
            addr: uint(8),
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
        let mut m = design.module("BundleMacro");
        let req = m.input_bundle("req", req_ty);
        let resp = m.output_bundle("resp", resp_ty);

        m.assign(resp.field("valid").unwrap(), req.field("valid").unwrap());
        m.assign(
            resp.field("data").unwrap(),
            zext(req.path(["meta", "tag"]).unwrap(), 8),
        );
    }

    design.validate().unwrap();
    let sv = rrtl::sv::emit(&design).unwrap();
    assert!(sv.contains("module BundleMacro"));
    assert!(sv.contains("input logic req_valid"));
    assert!(sv.contains("output logic [7:0] resp_data"));
}

#[test]
fn bundle_macro_matches_manual_bundle_type() {
    let from_macro = bundle! {
        Req {
            valid: uint(1),
            data: sint(8),
            meta: ReqMeta {
                tag: uint(4),
            },
        }
    };

    let manual = rrtl::bundle_type(
        "Req",
        [
            rrtl::field("valid", rrtl::uint(1)),
            rrtl::field("data", rrtl::sint(8)),
            rrtl::nested(
                "meta",
                rrtl::bundle_type("ReqMeta", [rrtl::field("tag", rrtl::uint(4))]),
            ),
        ],
    );

    assert_eq!(from_macro, manual);
}
