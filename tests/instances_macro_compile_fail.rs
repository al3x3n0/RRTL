#[test]
fn instances_macro_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/instances_macro_unsupported_kind.rs");
    t.compile_fail("tests/compile_fail/instances_macro_malformed_connection.rs");
}
