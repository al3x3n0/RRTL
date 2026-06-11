#[test]
fn extern_module_macro_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/extern_module_macro_unsupported_direction.rs");
    t.compile_fail("tests/compile_fail/extern_module_macro_empty.rs");
}
