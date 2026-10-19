use ryft::{Array, ArrayType, GuardOperation, TagOperation};

/// Stores no native guard but hand-writes the guard conversion, which enables the blanket supporting provider next to
/// the rejecting provider that the derive generates.
#[derive(Clone, Debug, ryft::Operation)]
#[ryft(type(ArrayType), constant(Array))]
enum ConvertedGuardOperation {
    Tag(TagOperation<ArrayType>),
}

impl From<GuardOperation<ArrayType>> for ConvertedGuardOperation {
    fn from(_operation: GuardOperation<ArrayType>) -> Self {
        Self::Tag(TagOperation::new("guard"))
    }
}

/// Alias that hides the `GuardOperation` path segment by which the derive recognizes native guard payloads.
type CheckedInput = GuardOperation<ArrayType>;

/// Stores its guard through the alias, so the derive generates the rejecting provider next to the blanket supporting
/// provider that the generated conversion enables.
#[derive(Clone, Debug, ryft::Operation)]
#[ryft(type(ArrayType), constant(Array))]
enum AliasedGuardOperation {
    Checked(CheckedInput),
}

fn main() {}
