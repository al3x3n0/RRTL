#[test]
fn bundle_macro_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/bundle_macro_unsupported_constructor.rs");
    t.compile_fail("tests/compile_fail/bundle_macro_malformed_nested.rs");
}
