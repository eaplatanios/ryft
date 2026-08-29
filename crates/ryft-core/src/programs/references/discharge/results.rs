use std::fmt::Display;

use crate::parameters::Parameterized;
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::types::Type;
use crate::programs::values::Value;

// TODO(eaplatanios): Review from here onwards.

/// Reference-free program payload and logical external-state metadata produced by reference discharge.
///
/// A full result is a [`PartialReferenceDischargeResult`] plus the reference-freedom proof, which is exactly how it
/// is represented: [`PartialReferenceDischargeResult::try_into_full`] carries out the proof and wraps the partial
/// result unchanged.
#[derive(Debug)]
pub struct ReferenceDischargeResult<P: ReferenceDischargePayload> {
    /// Proven reference-free partial result.
    partial: PartialReferenceDischargeResult<P>,
}

impl<P: ReferenceDischargePayload> ReferenceDischargeResult<P> {
    /// Creates a full discharge result after invoking the payload provider's reference-freedom proof.
    ///
    /// [`ReferenceDischargePayload::validate_reference_free`] proves the payload property, and this constructor then
    /// validates the discharged boundary layout. For [`Program`] payloads the implementation performs the same
    /// closure-wide scan as [`PartialReferenceDischargeResult::try_into_full`], so this entry point cannot bypass it.
    ///
    /// # Parameters
    ///
    ///   - `program`: Discharged program payload validated through its provider implementation.
    ///   - `capture_count`: Number of leading inputs originating in the source program's capture table.
    ///   - `public_output_count`: Number of public outputs preceding hidden final-state outputs.
    ///   - `external_states`: Logical external-state bindings in canonical entry-boundary order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the payload retains references or when the counts and bindings
    /// are not arithmetically consistent with one canonical discharged boundary: strict canonical source ordering,
    /// in-range flat positions, and exact hidden-suffix coverage. The constructor cannot prove *semantic* identity —
    /// that each named source is the state the payload actually threads at that position remains the provider's
    /// obligation, exactly as with any positional ABI.
    pub fn from_provider_payload(
        program: P,
        capture_count: usize,
        public_output_count: usize,
        external_states: Vec<ExternalReferenceBinding>,
    ) -> Result<Self, ProgramError> {
        program.validate_reference_free()?;
        Ok(Self {
            partial: PartialReferenceDischargeResult::new(
                program,
                capture_count,
                public_output_count,
                external_states,
            )?,
        })
    }

    /// Returns the reference-free program payload.
    #[inline]
    pub const fn program(&self) -> &P {
        self.partial.program()
    }

    /// Returns the number of leading inputs originating in the source program's capture table.
    #[inline]
    pub const fn capture_count(&self) -> usize {
        self.partial.capture_count()
    }

    /// Returns the number of public outputs at the front of the program payload's output boundary.
    #[inline]
    pub const fn public_output_count(&self) -> usize {
        self.partial.public_output_count()
    }

    /// Returns external reference binding recipes in canonical entry-boundary order.
    #[inline]
    pub fn external_states(&self) -> &[ExternalReferenceBinding] {
        self.partial.external_states()
    }

    /// Consumes this result and returns its payload, capture count, public-output prefix, and external-state bindings.
    #[inline]
    pub fn into_parts(self) -> (P, usize, usize, Vec<ExternalReferenceBinding>) {
        self.partial.into_parts()
    }
}

/// Program payload produced by *partial* reference discharge, in which only the caller-selected reference sites became
/// explicit immutable state and every unselected allocation survives as a well-typed reference value.
///
/// The discharged part of the boundary obeys exactly the invariants of [`ReferenceDischargeResult`]: discharged
/// external allocations are reported as [`ExternalReferenceBinding`]s in canonical entry-boundary order, and the
/// mutated subset of those bindings tiles the hidden output suffix that follows the public outputs. Discharged local
/// allocations leave no binding, because no caller owns their state. Preserved references contribute neither bindings
/// nor hidden outputs; they simply remain reference-typed values inside the payload, and their accesses replay
/// verbatim.
///
/// There is deliberately no blanket conversion into [`ReferenceDischargeResult`]: "every site was selected" is a
/// statement about the request, not a proof about the produced payload, and a malformed provider could satisfy it
/// while still emitting references. [`try_into_full`](Self::try_into_full) therefore exists only for
/// [`Program`] payloads, where the reference-freedom proof can actually be carried out. Providers of other payload
/// families encode their equivalent proof through [`ReferenceDischargePayload`].
#[derive(Debug)]
pub struct PartialReferenceDischargeResult<P: ReferenceDischargePayload> {
    /// Program payload whose public outputs form a prefix of its complete outputs.
    program: P,

    /// Number of leading program inputs originating in the source program's capture table.
    capture_count: usize,

    /// Number of public output leaves before hidden final-state outputs.
    public_output_count: usize,

    /// Discharged external reference binding recipes in canonical entry-boundary order.
    external_states: Vec<ExternalReferenceBinding>,
}

impl<P: ReferenceDischargePayload> PartialReferenceDischargeResult<P> {
    /// Creates a checked partial reference discharge result.
    ///
    /// The external-state bindings describe the *discharged* allocations only and must satisfy the same canonical
    /// boundary invariants as [`ReferenceDischargeResult::from_provider_payload`]: they must name valid discharged
    /// inputs in canonical source order, and their final-state output indices, omitting read-only bindings, must
    /// exactly cover the hidden output suffix in binding order.
    ///
    /// # Parameters
    ///
    ///   - `program`: Mixed discharged program payload.
    ///   - `capture_count`: Number of leading inputs originating in the source program's capture table.
    ///   - `public_output_count`: Number of public outputs preceding hidden final-state outputs.
    ///   - `external_states`: Logical external-reference bindings for the discharged references, in canonical
    ///     entry-boundary order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the counts and bindings do not describe one canonical
    /// discharged boundary.
    pub fn new(
        program: P,
        capture_count: usize,
        public_output_count: usize,
        external_states: Vec<ExternalReferenceBinding>,
    ) -> Result<Self, ProgramError> {
        validate_discharged_boundary(
            capture_count,
            program.input_count(),
            program.output_count(),
            public_output_count,
            external_states.as_slice(),
        )?;
        Ok(Self { program, capture_count, public_output_count, external_states })
    }

    /// Returns the mixed program payload.
    #[inline]
    pub const fn program(&self) -> &P {
        &self.program
    }

    /// Returns the number of leading inputs originating in the source program's capture table.
    #[inline]
    pub const fn capture_count(&self) -> usize {
        self.capture_count
    }

    /// Returns the number of public outputs at the front of the program payload's output boundary.
    #[inline]
    pub const fn public_output_count(&self) -> usize {
        self.public_output_count
    }

    /// Returns the binding recipes of the discharged external reference allocations, in canonical entry-boundary order.
    /// Preserved references are deliberately absent: they were never turned into state and so have nothing to bind.
    #[inline]
    pub fn external_states(&self) -> &[ExternalReferenceBinding] {
        self.external_states.as_slice()
    }

    /// Consumes this result and returns its payload, capture count, public-output prefix, and external-state bindings.
    #[inline]
    pub fn into_parts(self) -> (P, usize, usize, Vec<ExternalReferenceBinding>) {
        let Self { program, capture_count, public_output_count, external_states } = self;
        (program, capture_count, public_output_count, external_states)
    }
}

impl<V, O, Input, Output> PartialReferenceDischargeResult<Program<V, O, Input, Output>>
where
    V: Value,
    O: Operation<Type = V::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Proves that this partial result is in fact reference-free and converts it into a
    /// [`ReferenceDischargeResult`].
    ///
    /// The proof inspects the complete attached region closure of the payload, dormant transformation rule regions
    /// included, and requires that no atom carries a reference type and that no operation declares nonempty
    /// [`ReferenceOperationSemantics`](crate::ReferenceOperationSemantics). Because every boundary position and every
    /// stored constant is itself an atom, the first check covers input types, output types, and constants alike.
    ///
    /// The proof is deliberately reference-specific rather than a general state-purification check. Reference
    /// discharge normalizes references and nothing else, so an unrelated ordered-state operation contributed by a
    /// third-party backend passes through untouched, and the consumers that care about ordered state keep their own
    /// gates.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the payload still contains a reference-typed atom or an
    /// operation with nonempty reference semantics.
    pub fn try_into_full(self) -> Result<ReferenceDischargeResult<Program<V, O, Input, Output>>, ProgramError> {
        self.program.validate_reference_free()?;
        Ok(ReferenceDischargeResult { partial: self })
    }
}

/// Boundary-inspection and reference-freedom-validation capability for reference discharge payloads.
///
/// Both partial and full discharge results use the reported entry-boundary arities to validate their external reference
/// bindings. A payload may implement this trait while still containing references: the implementation of
/// [`Self::validate_reference_free`] must inspect the complete payload and return an error until no reference-typed
/// value or nonempty reference semantics remains. Full discharge invokes that validation before constructing its
/// result.
///
/// These are provider contracts for downstream payload families. Rust's coherence rules prevent a downstream crate
/// from replacing the checked implementation for [`Program`].
pub trait ReferenceDischargePayload {
    /// Returns the number of inputs in this payload's entry boundary.
    fn input_count(&self) -> usize;

    /// Returns the number of outputs in this payload's entry boundary.
    fn output_count(&self) -> usize;

    /// Validates that this complete payload is reference-free.
    fn validate_reference_free(&self) -> Result<(), ProgramError>;
}

impl<V, O, Input, Output> ReferenceDischargePayload for Program<V, O, Input, Output>
where
    V: Value,
    O: Operation<Type = V::Type>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    #[inline]
    fn input_count(&self) -> usize {
        self.entry_region_ref().input_ids().len()
    }

    #[inline]
    fn output_count(&self) -> usize {
        self.entry_region_ref().output_ids().len()
    }

    fn validate_reference_free(&self) -> Result<(), ProgramError> {
        let entry = self.entry_region_ref();
        if entry.contains_atom_type_in_closure(Type::is_reference) {
            return Err(ProgramError::MalformedProgram(
                "reference discharge payload still contains a reference-typed value and cannot form a full discharge"
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
                "reference discharge payload retains reference operation `{}` at `{instruction_id}` and cannot form \
                 a full discharge",
                instruction.operation().name(),
            )));
        }
        Ok(())
    }
}

/// Metadata connecting one caller-owned [`Reference`] to its explicit inputs and outputs after discharge.
///
/// Reference discharge turns implicit access to a reference into ordinary value flow that a reference-free backend
/// can execute. For example, consider a source function that takes parameter state by reference, updates it, and
/// returns a public result:
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
/// invocation layer reads the reference's current value into that discharged input. The `updated_parameters` value is
/// a synthetic writeback output: it is part of the complete discharged boundary, but the invocation layer consumes it
/// instead of returning it as part of the source function's public result. It installs that value back into the
/// caller's reference after execution.
///
/// In this example there is one public output, so [`Self::output_index`] is `Some(1)`: the absolute index of
/// `updated_parameters` in the complete output list `[result, updated_parameters]`. A reference that the function only
/// reads needs no writeback output and therefore has an output index of [`None`].
///
/// The [`serde::Serialize`] implementation exposes the canonical in-memory shape for diagnostics and snapshots and is
/// deliberately distinct from the stable XLA persistence schema, which keeps its own versioned representation
/// (including a redundant validated flat-input coordinate) independent of this type's evolution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct ExternalReferenceBinding {
    /// Capture or public input through which the caller supplies the reference.
    source: ReferenceSource,

    /// Absolute complete-output index containing the final reference value, or [`None`] for a read-only reference.
    output_index: Option<usize>,
}

impl ExternalReferenceBinding {
    /// Creates a discharged-program boundary binding for one external reference.
    ///
    /// # Parameters
    ///
    ///   - `source`: Capture or public input through which the caller supplies the reference.
    ///   - `output_index`: Absolute complete-output index containing the final reference value, or [`None`] when the
    ///     program only reads the reference.
    pub const fn new(source: ReferenceSource, output_index: Option<usize>) -> Self {
        Self { source, output_index }
    }

    /// Returns the capture or public input through which the caller supplies the reference.
    pub const fn source(&self) -> ReferenceSource {
        self.source
    }

    /// Returns whether the program may mutate this external reference.
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
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Capture { index } => write!(formatter, "capture {index}"),
            Self::Input { index } => write!(formatter, "input {index}"),
        }
    }
}

/// Validates that a discharged program boundary and its external-state bindings describe one canonical discharged
/// shape, shared by the full and partial result envelopes.
///
/// # Parameters
///
///   - `capture_count`: Number of leading inputs originating in the source program's capture table.
///   - `total_input_count`: Number of inputs in the discharged payload.
///   - `total_output_count`: Number of outputs in the discharged payload.
///   - `public_output_count`: Number of public outputs preceding hidden final-state outputs.
///   - `external_states`: External-state bindings in canonical entry-boundary order.
fn validate_discharged_boundary(
    capture_count: usize,
    total_input_count: usize,
    total_output_count: usize,
    public_output_count: usize,
    external_states: &[ExternalReferenceBinding],
) -> Result<(), ProgramError> {
    if capture_count > total_input_count {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge reports {capture_count} captures but discharged input count is {total_input_count}",
        )));
    }
    if public_output_count > total_output_count {
        return Err(ProgramError::MalformedProgram(format!(
            "reference discharge reports {public_output_count} public outputs but discharged output count is \
             {total_output_count}",
        )));
    }
    for state in external_states {
        let input_index = state.source().flat_input_index(capture_count)?;
        if input_index >= total_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge state for `{}` names input {input_index} but discharged input count is \
                 {total_input_count}",
                state.source(),
            )));
        }
    }
    for adjacent_states in external_states.windows(2) {
        let previous_source = adjacent_states[0].source();
        let source = adjacent_states[1].source();
        if source <= previous_source {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge state source `{source}` does not follow source `{previous_source}` in canonical \
                 boundary order",
            )));
        }
    }
    let mut expected_output_index = public_output_count;
    for state in external_states.iter().filter(|state| state.is_mutated()) {
        let output_index = state.output_index().unwrap();
        if output_index != expected_output_index {
            return Err(ProgramError::MalformedProgram(format!(
                "reference discharge final-state output {output_index} for `{}` does not match expected hidden output \
                 {expected_output_index}",
                state.source(),
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;

    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{Effect, Effects};

    use crate::programs::operations::Operation;

    use crate::programs::references::discharge::tests::*;
    use crate::programs::references::semantics::{ReferenceAccessMode, ReferenceInput, ReferenceOperationSemantics};

    use crate::programs::regions::RegionInterface;
    use crate::programs::types::TypeError;

    use super::*;

    #[test]
    fn test_reference_discharge_result_validates_boundaries() {
        let bindings = vec![
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, Some(1)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(2)),
        ];
        let result =
            ReferenceDischargeResult::from_provider_payload(TestPayload::new("program", 2, 3), 1, 1, bindings.clone())
                .unwrap();
        assert_eq!(result.program().value, "program");
        assert_eq!(result.capture_count(), 1);
        assert_eq!(result.public_output_count(), 1);
        assert_eq!(result.external_states(), bindings);

        assert_eq!(ReferenceSource::Capture { index: 0 }.flat_input_index(1), Ok(0));
        assert_eq!(ReferenceSource::Input { index: 2 }.flat_input_index(1), Ok(3));
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

        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 0, 0), 1, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 1 captures but discharged input count is 0".to_string(),
            ),
        );

        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 0, 1), 0, 2, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 2 public outputs but discharged output count is 1".to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(
                TestPayload::new((), 0, 0),
                0,
                0,
                vec![ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None)],
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge state for `input 0` names input 0 but discharged input count is 0".to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 1, 1), 0, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final states end at output 0 but discharged output count is 1".to_string(),
            ),
        );
        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 0, usize::MAX), 0, 0, Vec::new(),)
                .unwrap_err(),
            ProgramError::MalformedProgram(format!(
                "reference discharge final states end at output 0 but discharged output count is {}",
                usize::MAX,
            )),
        );
        for sources in [
            [ReferenceSource::Capture { index: 0 }, ReferenceSource::Capture { index: 0 }],
            [ReferenceSource::Input { index: 0 }, ReferenceSource::Capture { index: 0 }],
        ] {
            let bindings = sources
                .into_iter()
                .enumerate()
                .map(|(_, source)| ExternalReferenceBinding::new(source, None))
                .collect();
            assert_eq!(
                ReferenceDischargeResult::from_provider_payload(TestPayload::new((), 2, 0), 1, 0, bindings)
                    .unwrap_err(),
                ProgramError::MalformedProgram(format!(
                    "reference discharge state source `{}` does not follow source `{}` in canonical boundary order",
                    sources[1], sources[0],
                )),
            );
        }
    }

    #[test]
    fn test_partial_reference_discharge_result_reports_only_discharged_bindings() {
        // The partial envelope keeps the canonical discharged boundary of the full envelope, so its accessors report
        // the discharged bindings and the public-output prefix that precedes their hidden final-state suffix.
        let bindings = vec![
            ExternalReferenceBinding::new(ReferenceSource::Capture { index: 0 }, Some(1)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None),
        ];
        let result =
            PartialReferenceDischargeResult::new(TestPayload::new("program", 2, 2), 1, 1, bindings.clone()).unwrap();
        assert_eq!(result.program().value, "program");
        assert_eq!(result.capture_count(), 1);
        assert_eq!(result.public_output_count(), 1);
        assert_eq!(result.external_states(), bindings);
        assert_eq!(result.into_parts(), (TestPayload::new("program", 2, 2), 1, 1, bindings));

        // The shared boundary validation applies to the partial envelope exactly as it does to the full one.
        assert_eq!(
            PartialReferenceDischargeResult::new(TestPayload::new((), 0, 1), 0, 2, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge reports 2 public outputs but discharged output count is 1".to_string(),
            ),
        );
        assert_eq!(
            PartialReferenceDischargeResult::new(TestPayload::new((), 1, 1), 0, 0, Vec::new()).unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge final states end at output 0 but discharged output count is 1".to_string(),
            ),
        );
    }

    #[test]
    fn test_partial_reference_discharge_result_try_into_full_proves_reference_freedom() {
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
        let discharged = partial(program(&[ProofOperation::OrderedIo], TestType::Value(0))).try_into_full().unwrap();
        assert_eq!(discharged.public_output_count(), 1);
        assert!(discharged.program().effects().contains(Effect::OrderedIo));

        // A surviving reference-typed value is disqualifying wherever it appears, including on the boundary.
        assert_eq!(
            ReferenceDischargeResult::from_provider_payload(
                program(&[ProofOperation::OrderedIo], reference_type(0)),
                0,
                1,
                Vec::new(),
            )
            .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge payload still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );
        assert_eq!(
            partial(program(&[ProofOperation::OrderedIo], reference_type(0))).try_into_full().unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge payload still contains a reference-typed value and cannot form a full discharge"
                    .to_string(),
            ),
        );

        // A retained reference operation is disqualifying even when every value in the program is ordinary.
        assert_eq!(
            partial(program(&[ProofOperation::OrderedIo, ProofOperation::RetainedReference], TestType::Value(0)))
                .try_into_full()
                .unwrap_err(),
            ProgramError::MalformedProgram(
                "reference discharge payload retains reference operation `test.retained_reference` at `^0[1]` and \
                 cannot form a full discharge"
                    .to_string(),
            ),
        );
    }
}
