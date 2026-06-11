#[test]
fn state_macro_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/state_macro_empty_variants.rs");
    t.compile_fail("tests/compile_fail/state_macro_malformed_transition.rs");
}
