#[test]
fn signals_macro_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/signals_macro_unsupported_kind.rs");
    t.compile_fail("tests/compile_fail/signals_macro_scalar_with_bundle_type.rs");
    t.compile_fail("tests/compile_fail/signals_macro_bundle_with_scalar_type.rs");
}
