use ryft::mermaid;

/// Graph with a single edge.
///
/// ```mermaid
/// flowchart LR
///   source --> target
/// ```
#[mermaid]
#[derive(Debug)]
struct Graph;

#[test]
fn test_mermaid() {
    // Rendering the diagrams of an item keeps the item and its other attributes intact.
    assert_eq!(format!("{Graph:?}"), "Graph");
}

#[test]
fn test_errors() {
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/diagrams/error_diagrams.rs");
}
