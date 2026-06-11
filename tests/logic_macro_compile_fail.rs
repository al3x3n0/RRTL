#[test]
fn logic_macro_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/logic_macro_unsupported_kind.rs");
    t.compile_fail("tests/compile_fail/logic_macro_malformed_reset.rs");
    t.compile_fail("tests/compile_fail/logic_macro_malformed_assert.rs");
    t.compile_fail("tests/compile_fail/logic_macro_malformed_cover.rs");
    t.compile_fail("tests/compile_fail/logic_macro_malformed_assign_bundle.rs");
}
