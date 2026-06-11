#[test]
fn memory_macro_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/memory_macro_malformed_decl.rs");
    t.compile_fail("tests/compile_fail/memory_macro_malformed_write.rs");
}
