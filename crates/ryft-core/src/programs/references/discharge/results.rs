use std::fmt::Display;

use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::types::Type;
use crate::programs::values::Value;

/// [`Reference`](crate::Reference)-free [`Program`] and external-reference bindings produced by the reference
/// discharge transform. A full result is a [`PartialReferenceDischargeResult`] whose complete attached region closure
/// has been proven to contain neither reference-typed atoms nor operations with reference semantics. The [`TryFrom`]
/// implementation performs that proof and otherwise wraps the partial result unchanged. The proof examines every
/// attached region, including dormant rule regions. It rejects the conversion with [`ProgramError::MalformedProgram`]
/// if any reference-typed atom or operation with reference semantics remains; unrelated ordered-state operations do not
/// prevent conversion.
#[derive(Debug)]
pub struct ReferenceDischargeResult<V: Value, O: Operation<Type = V::Type>> {
    /// Underlying [`PartialReferenceDischargeResult`].
    partial: PartialReferenceDischargeResult<V, O>,
}

impl<V: Value, O: Operation<Type = V::Type>> ReferenceDischargeResult<V, O> {
    /// Returns the underlying [`Reference`](crate::Reference)-free [`Program`].
    #[inline]
    pub const fn program(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        self.partial.program()
    }

    /// Returns the number of leading [`Program`] inputs lifted from the source program's capture table. The discharged
    /// program has one flat input boundary in `[captures..., inputs...]` order. This count is the split point between
    /// those two groups; it counts every lifted capture, not only captures that contain references or appear in
    /// [`Self::external_reference_bindings`]. [`ReferenceSource::Capture`] indices are relative to the first group,
    /// while [`ReferenceSource::Input`] indices are relative to the second.
    ///
    /// For example, a count of `2` gives the following boundary:
    ///
    /// ```text
    /// [capture 0, capture 1 | input 0, input 1, ...]
    ///                       ^ capture_count
    /// ```
    #[inline]
    pub const fn capture_count(&self) -> usize {
        self.partial.capture_count()
    }

    /// Returns the number of public outputs at the front of the [`Program`]'s complete output boundary. Public outputs
    /// occupy output indices `[0, output_count)`. Any remaining outputs form a hidden suffix containing the final
    /// values of mutated external references, in [`Self::external_reference_bindings`] order after read-only bindings
    /// are omitted. A read-only binding has no hidden output.
    ///
    /// For example, an `output_count` of `2` with one mutated external reference gives:
    ///
    /// ```text
    /// [output 0, output 1 | final external state]
    ///                     ^ output_count
    /// ```
    #[inline]
    pub const fn output_count(&self) -> usize {
        self.partial.output_count()
    }

    /// Returns the bindings between caller-owned references and the discharged [`Program`] boundary. Bindings appear
    /// in canonical entry-boundary order: captures first, then public inputs, with each group ordered by its logical
    /// index. Each binding's [`ExternalReferenceBinding::source`] identifies the program input that receives the
    /// reference's initial value. A mutated binding also identifies the hidden output containing its final value.
    /// A read-only binding has no output index. Local allocations never appear because no caller owns their state.
    ///
    /// For example, the following metadata describes a read-only captured reference and a mutated reference supplied
    /// as public input 1:
    ///
    /// ```text
    /// capture_count = 1
    /// inputs         = [capture 0 | input 0, input 1]
    /// output_count   = 2
    /// outputs        = [output 0, output 1 | final input 1]
    /// bindings       = [capture 0 -> None, input 1 -> Some(2)]
    /// ```
    ///
    /// An empty slice means that executing the program requires no caller-owned reference state.
    #[inline]
    pub fn external_reference_bindings(&self) -> &[ExternalReferenceBinding] {
        self.partial.external_reference_bindings()
    }

    /// Consumes this [`ReferenceDischargeResult`] and returns its [`Program`] when execution requires no caller-owned
    /// reference state. A full discharge is reference-free, but that alone does not make its metadata discardable. Even
    /// a read-only external reference needs a binding so the caller can provide its initial value. A mutated external
    /// reference additionally needs a binding for its hidden final-state output. This conversion therefore accepts only
    /// an empty [`Self::external_reference_bindings`] slice.
    ///
    /// An empty binding slice also guarantees that there are no hidden external final-state outputs, so
    /// [`Self::output_count`] equals the returned program's complete output count. [`Self::capture_count`] may still be
    /// nonzero because ordinary, non-reference captures require no binding. The returned program is both reference-free
    /// and independent of caller-owned references.
    ///
    /// Conceptually, a program with only ordinary captures or local reference allocations can be returned directly,
    /// while either of the following external dependencies is rejected:
    ///
    /// ```text
    /// read-only external: input initial state
    /// mutated external:   input initial state -> hidden final state
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::UnsupportedOperation`] identifying the first caller-owned reference when this result
    /// contains any external binding.
    pub fn into_program_without_external_references(self) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        if let Some(binding) = self.external_reference_bindings().first() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("reference discharge cannot discard the binding for external `{}`", binding.source()),
            });
        }
        let (program, _, _, _) = self.into_parts();
        Ok(program)
    }

    /// Consumes this [`ReferenceDischargeResult`] and returns its underlying [`Program`], capture count, public output
    /// count, and external reference bindings, in that order.
    #[inline]
    pub fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, usize, usize, Vec<ExternalReferenceBinding>) {
        self.partial.into_parts()
    }
}

impl<V: Value, O: Operation<Type = V::Type>> TryFrom<PartialReferenceDischargeResult<V, O>>
    for ReferenceDischargeResult<V, O>
{
    type Error = ProgramError;

    fn try_from(partial: PartialReferenceDischargeResult<V, O>) -> Result<Self, Self::Error> {
        let entry = partial.program.entry_region_ref();
        if entry.contains_atom_type_in_closure(Type::is_reference) {
            return Err(ProgramError::MalformedProgram(
                "reference discharge program still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ));
        }

        // The closure traversal visits regions in an unspecified order, so the reported occurrence is the smallest
        // coordinate rather than the first one encountered, keeping the diagnostic reproducible.
        if let Some((instruction_id, instruction)) = entry
            .instructions_in_closure()
            .filter(|(_, instruction)| !instruction.operation().reference_semantics().is_empty())
            .min_by_key(|(instruction_id, _)| *instruction_id)
        {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge program retains reference operation `{}` at `{}` and cannot form a full discharge",
                instruction.operation().name(),
                instruction_id,
            )));
        }

        Ok(Self { partial })
    }
}

/// [`Program`] produced by _partial_ reference discharge, in which only the caller-selected reference targets became
/// explicit immutable state and every unselected allocation survives as a well-typed reference value. The discharged
/// part of the boundary obeys exactly the invariants of [`ReferenceDischargeResult`]: discharged external allocations
/// are reported as [`ExternalReferenceBinding`]s in canonical entry-boundary order, and the mutated subset of those
/// bindings tiles the hidden output suffix that follows the public outputs. Discharged local allocations leave no
/// binding, because no caller owns their state. Preserved references contribute neither bindings nor hidden outputs;
/// they simply remain reference-typed values inside the program, and their accesses replay verbatim.
#[derive(Debug)]
pub struct PartialReferenceDischargeResult<V: Value, O: Operation<Type = V::Type>> {
    /// Refer to [`Self::program`].
    program: Program<V, O, Vec<V>, Vec<V>>,

    /// Refer to [`Self::capture_count`].
    capture_count: usize,

    /// Refer to [`Self::output_count`].
    output_count: usize,

    /// Refer to [`Self::external_reference_bindings`].
    external_reference_bindings: Vec<ExternalReferenceBinding>,
}

impl<V: Value, O: Operation<Type = V::Type>> PartialReferenceDischargeResult<V, O> {
    /// Creates a new [`PartialReferenceDischargeResult`]. The provided external reference bindings
    /// describe the discharged allocations only and must satisfy the same canonical boundary invariants as
    /// [`ReferenceDischargeResult`] (i.e., they must name valid discharged inputs in canonical source order, and their
    /// final state output indices, omitting read-only bindings, must exactly cover the hidden output suffix in binding
    /// order).
    ///
    /// # Parameters
    ///
    ///   - `program`: Partially discharged [`Program`].
    ///   - `capture_count`: Number of leading inputs originating in the source program's capture table.
    ///   - `output_count`: Number of public outputs preceding hidden final-state outputs.
    ///   - `external_reference_bindings`: Logical bindings for the discharged external references, in canonical
    ///     entry-boundary order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the counts and bindings do not describe one canonical
    /// discharged boundary.
    pub fn new(
        program: Program<V, O, Vec<V>, Vec<V>>,
        capture_count: usize,
        output_count: usize,
        external_reference_bindings: Vec<ExternalReferenceBinding>,
    ) -> Result<Self, ProgramError> {
        let total_input_count = program.input_count();
        let total_output_count = program.output_count();
        if capture_count > total_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge reports {capture_count} captures but discharged input count is \
                 {total_input_count}",
            )));
        }
        if output_count > total_output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge reports {output_count} public outputs but discharged output count is \
                 {total_output_count}",
            )));
        }
        for binding in &external_reference_bindings {
            let input_index = binding.source().flat_input_index(capture_count)?;
            if input_index >= total_input_count {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge state for `{}` names input {} but discharged input count is {}",
                    binding.source(),
                    input_index,
                    total_input_count,
                )));
            }
        }
        for adjacent_bindings in external_reference_bindings.windows(2) {
            let previous_source = adjacent_bindings[0].source();
            let source = adjacent_bindings[1].source();
            if source <= previous_source {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge state source `{source}` does not follow source `{previous_source}` in \
                     canonical boundary order",
                )));
            }
        }
        let mut expected_output_index = output_count;
        for binding in external_reference_bindings.iter().filter(|binding| binding.is_mutated()) {
            let output_index = binding.output_index().unwrap();
            if output_index != expected_output_index {
                return Err(ProgramError::MalformedProgram(format!(
                    "reference discharge final-state output {} for `{}` does not match expected hidden output {}",
                    output_index,
                    binding.source(),
                    expected_output_index,
                )));
            }
            expected_output_index = expected_output_index.checked_add(1).ok_or_else(|| {
                ProgramError::MalformedProgram("reference discharge hidden output index overflows `usize`".to_string())
            })?;
        }
        if expected_output_index != total_output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge final states end at output {expected_output_index} but discharged output count is \
                 {total_output_count}",
            )));
        }
        Ok(Self { program, capture_count, output_count, external_reference_bindings })
    }

    /// Returns the underlying partially discharged [`Program`]. Unlike [`ReferenceDischargeResult::program`], this
    /// program may still contain reference-typed values and operations for allocations that the caller did not select
    /// for discharge.
    #[inline]
    pub const fn program(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.program
    }

    /// Returns the number of leading [`Program`] inputs lifted from the source program's capture table. The boundary
    /// uses `[captures..., inputs...]` order, and this count is the split point between the two groups. It counts all
    /// lifted captures, including ordinary captures and preserved reference captures that do not appear in
    /// [`Self::external_reference_bindings`]. For example, a count of `1` gives `[capture 0 | input 0, input 1, ...]`.
    #[inline]
    pub const fn capture_count(&self) -> usize {
        self.capture_count
    }

    /// Returns the number of public outputs at the front of the [`Program`]'s complete output boundary. Public outputs
    /// occupy `[0, output_count)`. Hidden final state outputs for mutated, discharged external references follow in
    /// [`Self::external_reference_bindings`] order after read-only bindings are omitted. Preserved references remain
    /// reference-typed values and add no hidden output. For example, an `output_count` of `1` gives
    /// `[output 0 | hidden final states...]`.
    #[inline]
    pub const fn output_count(&self) -> usize {
        self.output_count
    }

    /// Returns the bindings for external references selected and successfully discharged by the partial reference
    /// discharge transform. Bindings use the same canonical source ordering and input/output interpretation as
    /// [`ReferenceDischargeResult::external_reference_bindings`]. The difference is completeness: an external reference
    /// omitted from this slice may still survive as a reference-typed value because it was not selected. Local
    /// allocations never appear, and preserved external references contribute neither a binding nor a hidden output.
    /// For example, with inputs `[capture 0 | input 0]`, selecting only `input 0` produces a binding for
    /// [`ReferenceSource::Input`] index `0`; an unselected reference in `capture 0` remains in the program
    /// and does not produce a binding.
    #[inline]
    pub fn external_reference_bindings(&self) -> &[ExternalReferenceBinding] {
        self.external_reference_bindings.as_slice()
    }

    /// Consumes this [`PartialReferenceDischargeResult`] and returns its underlying [`Program`], capture count,
    /// public output count, and external reference bindings, in that order.
    #[inline]
    pub fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, usize, usize, Vec<ExternalReferenceBinding>) {
        let Self { program, capture_count, output_count, external_reference_bindings } = self;
        (program, capture_count, output_count, external_reference_bindings)
    }
}

/// Metadata connecting one caller-owned [`Reference`] to its explicit inputs and outputs after reference discharge.
/// Reference discharge turns implicit access to a reference into ordinary value flow that a reference-free backend can
/// execute. For example, consider a source function that takes parameter state by reference, updates it, and returns a
/// public result:
///
/// ```text
/// train_step(parameters: Reference<Array>, batch: Array) -> Array
/// ```
///
/// Its discharged boundary has the following conceptual shape:
///
/// ```text
/// train_step(parameters: Array, batch: Array) -> (result: Array, updated_parameters: Array)
/// ```
///
/// [`Self::source`] identifies where the caller supplied the original reference. Before execution, the stateful
/// invocation layer reads the reference's current value into that discharged input. The `updated_parameters` value
/// is a synthetic writeback output (i.e., it is part of the complete discharged boundary, but the invocation layer
/// consumes it instead of returning it as part of the source function's public result). It "installs" that value back
/// into the caller's reference after execution.
///
/// In this example there is one public output, so [`Self::output_index`] is `Some(1)`: the absolute index of
/// `updated_parameters` in the complete output list `[result, updated_parameters]`. A reference that the function only
/// reads needs no writeback output and therefore has an output index of [`None`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ExternalReferenceBinding {
    /// Capture or public input through which the caller supplies the reference.
    source: ReferenceSource,

    /// Absolute complete-output index containing the final reference value, or [`None`] for a read-only reference.
    output_index: Option<usize>,
}

impl ExternalReferenceBinding {
    /// Creates a new [`ExternalReferenceBinding`].
    ///
    /// # Parameters
    ///
    ///   - `source`: Capture or public input through which the caller supplies the reference.
    ///   - `output_index`: Absolute complete-output index containing the final reference value,
    ///     or [`None`] when the program only reads the reference.
    pub const fn new(source: ReferenceSource, output_index: Option<usize>) -> Self {
        Self { source, output_index }
    }

    /// Returns the capture or public input through which the caller supplies the reference.
    pub const fn source(&self) -> ReferenceSource {
        self.source
    }

    /// Returns whether the corresponding [`Program`] may mutate this external reference.
    pub const fn is_mutated(&self) -> bool {
        self.output_index.is_some()
    }

    /// Returns the absolute complete-output index containing the final reference value, if one must be written back.
    pub const fn output_index(&self) -> Option<usize> {
        self.output_index
    }
}

/// Source of a [`Reference`] that is external to a [`Program`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceSource {
    /// Reference to a capture lifted into the entry boundary before input arguments.
    Capture {
        /// Zero-based capture position in the lifted capture prefix.
        index: usize,
    },

    /// Reference input argument after the lifted capture prefix.
    Input {
        /// Zero-based input position, excluding lifted captures.
        index: usize,
    },
}

// TODO(eaplatanios): Review from here onwards.

impl ReferenceSource {
    /// Returns the logical source occupying one position in a program's flat entry input boundary.
    ///
    /// Capture lifting forms one flat input list in canonical `[captures..., inputs...]` order: the first
    /// `capture_count` positions correspond to the source program's capture table, and every remaining position
    /// corresponds to a public input. This function classifies `flat_input_index` relative to that split and expresses
    /// public input positions without the leading capture prefix.
    ///
    /// This function cannot validate that `flat_input_index` is within the complete input boundary because it receives
    /// only the capture-prefix length; callers enumerating a program boundary must supply one of that program's valid
    /// input positions.
    ///
    /// # Parameters
    ///
    ///   - `flat_input_index`: Zero-based position in the complete flat `[captures..., inputs...]` entry boundary.
    ///   - `capture_count`: Number of leading boundary positions originating in the source program's capture table.
    #[inline]
    pub const fn from_flat_input_index(flat_input_index: usize, capture_count: usize) -> Self {
        if flat_input_index < capture_count {
            Self::Capture { index: flat_input_index }
        } else {
            Self::Input { index: flat_input_index - capture_count }
        }
    }

    /// Returns this logical source's position in the program's flat entry input boundary.
    ///
    /// Capture lifting forms one flat input list in canonical `[captures..., inputs...]` order. A capture's logical
    /// index is therefore already its flat position, while a public input's logical index is offset by
    /// `capture_count`, the length of the leading capture prefix.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading boundary positions originating in the source program's capture table.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when a capture index lies outside the leading capture prefix or when
    /// offsetting a public input index by that prefix overflows `usize`.
    pub fn flat_input_index(self, capture_count: usize) -> Result<usize, ProgramError> {
        match self {
            Self::Capture { index } if index < capture_count => Ok(index),
            Self::Capture { index } => Err(ProgramError::MalformedProgram(format!(
                "reference source capture {index} lies outside the capture prefix of length {capture_count}",
            ))),
            Self::Input { index } => capture_count.checked_add(index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "reference source input {index} overflows the flat boundary after {capture_count} captures",
                ))
            }),
        }
    }
}

impl Display for ReferenceSource {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capture { index } => write!(formatter, "capture {index}"),
            Self::Input { index } => write!(formatter, "input {index}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{Effect, Effects};
    use crate::programs::operations::Operation;
    use crate::programs::references::discharge::tests::{
        TestOperation, TestType, TestValue, boundary_program, reference_type,
    };
    use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceInput, ReferenceOperationSemantics};
    use crate::programs::regions::RegionInterface;
    use crate::programs::types::TypeError;

    use super::*;

    #[test]
    fn test_reference_discharge_result_accessors_and_into_parts() {
        let bindings = vec![
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, Some(1)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None),
        ];
        let result = ReferenceDischargeResult::try_from(
            PartialReferenceDischargeResult::new(boundary_program(2, 2), 1, 1, bindings.clone()).unwrap(),
        )
        .unwrap();
        assert_eq!(result.program().input_count(), 2);
        assert_eq!(result.program().output_count(), 2);
        assert_eq!(result.capture_count(), 1);
        assert_eq!(result.output_count(), 1);
        assert_eq!(result.external_reference_bindings(), bindings);

        let (program, capture_count, output_count, external_reference_bindings) = result.into_parts();
        assert_eq!(program.input_count(), 2);
        assert_eq!(program.output_count(), 2);
        assert_eq!(capture_count, 1);
        assert_eq!(output_count, 1);
        assert_eq!(external_reference_bindings, bindings);
    }

    #[test]
    fn test_reference_discharge_result_converts_only_without_external_references() {
        let local = ReferenceDischargeResult::try_from(
            PartialReferenceDischargeResult::new(boundary_program(1, 1), 0, 1, Vec::new()).unwrap(),
        )
        .unwrap();
        let program = local.into_program_without_external_references().unwrap();
        assert_eq!(program.input_count(), 1);
        assert_eq!(program.output_count(), 1);

        let external = ReferenceDischargeResult::try_from(
            PartialReferenceDischargeResult::new(
                boundary_program(1, 1),
                0,
                1,
                vec![ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None)],
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            external.into_program_without_external_references().unwrap_err(),
            ProgramError::UnsupportedOperation {
                message: "reference discharge cannot discard the binding for external `input 0`".to_string(),
            },
        );
    }

    #[test]
    fn test_partial_reference_discharge_result_accessors_and_into_parts() {
        let bindings = vec![
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, Some(1)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None),
        ];
        let result = PartialReferenceDischargeResult::new(boundary_program(2, 2), 1, 1, bindings.clone()).unwrap();
        assert_eq!(result.program().input_count(), 2);
        assert_eq!(result.program().output_count(), 2);
        assert_eq!(result.capture_count(), 1);
        assert_eq!(result.output_count(), 1);
        assert_eq!(result.external_reference_bindings(), bindings);

        let (program, capture_count, output_count, external_reference_bindings) = result.into_parts();
        assert_eq!(program.input_count(), 2);
        assert_eq!(program.output_count(), 2);
        assert_eq!(capture_count, 1);
        assert_eq!(output_count, 1);
        assert_eq!(external_reference_bindings, bindings);
    }

    #[test]
    fn test_partial_reference_discharge_result_new_rejects_invalid_boundary_counts() {
        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(0, 0), 1, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 1 captures but discharged input count is 0".to_string(),
            ),
        );

        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(0, 1), 0, 2, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 2 public outputs but discharged output count is 1".to_string(),
            ),
        );
    }

    #[test]
    fn test_partial_reference_discharge_result_new_rejects_invalid_external_sources() {
        assert_eq!(
            PartialReferenceDischargeResult::new(
                boundary_program(2, 0),
                1,
                0,
                vec![ExternalReferenceBinding::new(ReferenceSource::Capture { index: 1 }, None)],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference source capture 1 lies outside the capture prefix of length 1".to_string(),
            ),
        );
        assert_eq!(
            PartialReferenceDischargeResult::new(
                boundary_program(0, 0),
                0,
                0,
                vec![ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None)],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge state for `input 0` names input 0 but discharged input count is 0".to_string(),
            ),
        );

        let duplicate = vec![
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None),
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None),
        ];
        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(2, 0), 1, 0, duplicate).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge state source `capture 0` does not follow source `capture 0` in canonical boundary \
                 order"
                    .to_string(),
            ),
        );

        let decreasing = vec![
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None),
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None),
        ];
        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(2, 0), 1, 0, decreasing).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge state source `capture 0` does not follow source `input 0` in canonical boundary \
                 order"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_partial_reference_discharge_result_new_rejects_invalid_hidden_outputs() {
        // A program output not covered by the public prefix or a mutated binding is an unaccounted hidden output.
        assert_eq!(
            PartialReferenceDischargeResult::new(boundary_program(1, 1), 0, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final states end at output 0 but discharged output count is 1".to_string(),
            ),
        );

        // A mutated binding cannot append a hidden output after the program's complete output boundary.
        assert_eq!(
            PartialReferenceDischargeResult::new(
                boundary_program(1, 1),
                0,
                1,
                vec![ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final states end at output 2 but discharged output count is 1".to_string(),
            ),
        );

        // Mutated bindings tile the hidden suffix exactly in binding order; they cannot name a public output.
        assert_eq!(
            PartialReferenceDischargeResult::new(
                boundary_program(1, 2),
                0,
                1,
                vec![ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(0))],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final-state output 0 for `input 0` does not match expected hidden output 1"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_discharge_result_try_from_enforces_reference_freedom() {
        // Operation family that separates the two facts the reference-freedom proof must distinguish: an unrelated
        // ordered-state operation that discharge never touches, and a retained reference operation that it must
        // reject even though its boundary types are ordinary.
        #[derive(Copy, Clone, Debug)]
        enum ProofOperation {
            OrderedIo,
            RetainedReference,
        }

        impl Display for ProofOperation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.name())
            }
        }

        impl Operation for ProofOperation {
            type Type = TestType;

            fn name(&self) -> &'static str {
                match self {
                    Self::OrderedIo => "test.ordered_io",
                    Self::RetainedReference => "test.retained_reference",
                }
            }

            fn infer_output_types(
                &self,
                input_types: &[TestType],
                _region_interfaces: &[RegionInterface<TestType>],
            ) -> Result<Vec<TestType>, TypeError> {
                Ok(input_types.to_vec())
            }

            fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
                match self {
                    Self::OrderedIo => Cow::Borrowed(ReferenceOperationSemantics::empty()),
                    Self::RetainedReference => Cow::Owned(ReferenceOperationSemantics::new(
                        vec![ReferenceInput::new(0, ReferenceAccessMode::Read)],
                        Vec::new(),
                    )),
                }
            }

            fn effects(&self) -> Effects {
                Effects::single(Effect::OrderedIo)
            }
        }

        let program = |operations: &[ProofOperation], input_type: TestType| {
            let mut builder = ProgramBuilder::<TestValue, ProofOperation>::new();
            let mut value = builder.add_input(input_type);
            for operation in operations {
                value = builder.add_instruction(*operation, Vec::new(), vec![value], None).unwrap()[0];
            }
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let partial = |program| PartialReferenceDischargeResult::new(program, 0, 1, Vec::new()).unwrap();

        // Discharge normalizes references and nothing else, so an unrelated ordered-state operation is proof-neutral
        // and its program converts into the reference-free envelope unchanged.
        let discharged =
            ReferenceDischargeResult::try_from(partial(program(&[ProofOperation::OrderedIo], TestType::Value(0))))
                .unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert!(discharged.program().effects().contains(Effect::OrderedIo));

        // A surviving reference-typed value is disqualifying wherever it appears, including on the boundary.
        assert_eq!(
            ReferenceDischargeResult::try_from(partial(program(&[ProofOperation::OrderedIo], reference_type(0),)))
                .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge program still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );

        // A retained reference operation is disqualifying even when every value in the program is ordinary.
        assert_eq!(
            ReferenceDischargeResult::try_from(partial(program(
                &[ProofOperation::OrderedIo, ProofOperation::RetainedReference],
                TestType::Value(0),
            )))
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge program retains reference operation `test.retained_reference` at `^0[1]` and \
                 cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_discharge_result_try_from_checks_the_attached_region_closure() {
        // The entry boundary is reference-free, but its attached callee allocates and consumes a local reference.
        // Inspecting only the entry region would therefore accept this program incorrectly.
        let mut callee_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = callee_builder.add_input(TestType::Value(0));
        let allocation = callee_builder
            .add_instruction(TestOperation::NewAllocation, Vec::new(), vec![initial], None)
            .unwrap()[0];
        let value =
            callee_builder.add_instruction(TestOperation::Consume, Vec::new(), vec![allocation], None).unwrap()[0];
        let callee = callee_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(TestType::Value(0));
        let callee = builder.import_program(callee);
        let value = builder.add_instruction(TestOperation::Call, vec![callee], vec![initial], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let partial = PartialReferenceDischargeResult::new(program, 0, 1, Vec::new()).unwrap();

        assert_eq!(
            ReferenceDischargeResult::try_from(partial).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge program still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );
    }

    #[test]
    fn test_external_reference_binding_accessors_and_serialization() {
        let read_only = ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, None);
        let mutated = ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, Some(3));

        assert_eq!(read_only.source(), ReferenceSource::Capture { index: 0 });
        assert!(!read_only.is_mutated());
        assert_eq!(read_only.output_index(), None);
        assert_eq!(mutated.source(), ReferenceSource::Input { index: 2 });
        assert!(mutated.is_mutated());
        assert_eq!(mutated.output_index(), Some(3));
        assert_eq!(read_only, read_only);
        assert_ne!(read_only, mutated);
        let bindings = HashMap::from([(read_only, "read-only"), (mutated, "mutated")]);
        assert_eq!(bindings.get(&read_only), Some(&"read-only"));
        assert_eq!(bindings.get(&mutated), Some(&"mutated"));
        assert_eq!(
            format!("{mutated:?}"),
            "ExternalReferenceBinding { source: Input { index: 2 }, output_index: Some(3) }",
        );
        assert_eq!(
            serde_json::to_string(&[read_only, mutated]).unwrap(),
            r#"[{"source":{"capture":{"index":0}},"output_index":null},{"source":{"input":{"index":2}},"output_index":3}]"#,
        );
    }

    #[test]
    fn test_reference_source_flat_input_index_round_trips() {
        for (flat_input_index, capture_count, source) in [
            (0, 0, ReferenceSource::Input { index: 0 }),
            (0, 2, ReferenceSource::Capture { index: 0 }),
            (1, 2, ReferenceSource::Capture { index: 1 }),
            (2, 2, ReferenceSource::Input { index: 0 }),
            (4, 2, ReferenceSource::Input { index: 2 }),
        ] {
            assert_eq!(ReferenceSource::from_flat_input_index(flat_input_index, capture_count), source);
            assert_eq!(source.flat_input_index(capture_count), Ok(flat_input_index));
        }
    }

    #[test]
    fn test_reference_source_flat_input_index_rejects_invalid_sources() {
        assert_eq!(
            ReferenceSource::Capture { index: 1 }.flat_input_index(1),
            Err(ProgramError::MalformedProgram(
                "reference source capture 1 lies outside the capture prefix of length 1".to_string(),
            )),
        );
        assert_eq!(
            ReferenceSource::Input { index: usize::MAX }.flat_input_index(1),
            Err(ProgramError::MalformedProgram(format!(
                "reference source input {} overflows the flat boundary after 1 captures",
                usize::MAX,
            ))),
        );
    }

    #[test]
    fn test_reference_source_ordering_rendering_and_serialization() {
        let first_capture = ReferenceSource::Capture { index: 0 };
        let second_capture = ReferenceSource::Capture { index: 1 };
        let first_input = ReferenceSource::Input { index: 0 };
        let second_input = ReferenceSource::Input { index: 1 };

        assert!(first_capture < second_capture);
        assert!(second_capture < first_input);
        assert!(first_input < second_input);
        assert_eq!(first_capture.to_string(), "capture 0");
        assert_eq!(second_input.to_string(), "input 1");
        assert_eq!(format!("{first_capture:?}"), "Capture { index: 0 }");
        assert_eq!(format!("{second_input:?}"), "Input { index: 1 }");
        assert_eq!(
            serde_json::to_string(&[first_capture, second_input]).unwrap(),
            r#"[{"capture":{"index":0}},{"input":{"index":1}}]"#,
        );
    }
}
