#[test]
fn interface_macro_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/interface_macro_unsupported_direction.rs");
    t.compile_fail("tests/compile_fail/interface_macro_unsupported_constructor.rs");
}
