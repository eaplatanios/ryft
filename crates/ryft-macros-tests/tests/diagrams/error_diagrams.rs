use ryft::mermaid;

#[mermaid(theme = "dark")]
struct Arguments;

/// ```mermaid
/// flowchart LR
#[mermaid]
struct Unterminated;

/// ```mermaid
/// ```text
/// ```
#[mermaid]
struct Nested;

/// ```mermaid
#[allow(dead_code)]
/// ```
#[mermaid]
struct NonLiteral;

fn main() {
    // The annotated items are still emitted, so using them reports no additional errors.
    let _ = (Arguments, Unterminated, Nested, NonLiteral);
}
