#[test]
fn ready_valid_macro_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/ready_valid_macro_unsupported_kind.rs");
    t.compile_fail("tests/compile_fail/ready_valid_macro_malformed_depth.rs");
    t.compile_fail("tests/compile_fail/ready_valid_macro_malformed_connect.rs");
}
