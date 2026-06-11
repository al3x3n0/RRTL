use rrtl::{bundle, zext, Design};

fn main() {
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
        let mut m = design.module("BundleMacroResponder");
        let req = m.input_bundle("req", req_ty);
        let resp = m.output_bundle("resp", resp_ty);

        m.assign(resp.field("valid").unwrap(), req.field("valid").unwrap());
        m.assign(
            resp.field("data").unwrap(),
            zext(req.path(["meta", "tag"]).unwrap(), 8),
        );
    }

    design.validate().unwrap();
    println!("{}", rrtl::sv::emit(&design).unwrap());
}
