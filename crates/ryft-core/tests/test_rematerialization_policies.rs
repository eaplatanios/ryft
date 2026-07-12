//! Compile-time coverage for rematerialization policy capability bounds: policies state their operation
//! requirements as derive-generated variant-projection bounds, so configuring a policy in a domain whose operation
//! family cannot satisfy it must fail to compile at the configuration site.

#[test]
fn test_errors() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/compile_fail/*.rs");
}
