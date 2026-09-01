use std::fmt::Display;

use crate::programs::ProgramError;

/// Represents the type of [`Reference`](crate::Reference) access performed by an [`Operation`](crate::Operation).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceAccessMode {
    /// Reads the referenced value without replacing it.
    Read,

    /// Replaces the selected referenced value without observing its previous contents.
    Write,

    /// Observes the selected referenced value and replaces it with a successor in program order.
    /// [`ReferenceSwapOperation`](crate::ReferenceSwapOperation) remains [`ReferenceAccessMode::ReadWrite`] even
    /// when a caller leaves its old-value result dead: liveness is a use-site fact, not operation semantics.
    ReadWrite,

    /// Combines an update with the current state as an _ordered_ additive accumulation. Accumulation stays distinct
    /// from [`ReferenceAccessMode::Write`] because it is linear in the update operand and therefore transposable,
    /// unlike a replacement. It carries no commutativity or atomicity promise: same-allocation accumulations execute
    /// in program order (floating-point addition cannot generally be reordered while preserving results), and
    /// atomic/commutative accumulation is not supported by this mode.
    Accumulate,

    /// Consumes the allocation. After such an access, the allocation and its entire alias family are invalid.
    /// Consumption is a lifetime event, not a memory-access flavor:
    /// [`ReferenceFreezeOperation`](crate::ReferenceFreezeOperation) is the consuming access that also returns
    /// the final value.
    Consume,
}

impl ReferenceAccessMode {
    /// Returns whether this [`ReferenceAccessMode`] consumes the complete reference allocation.
    pub const fn is_consuming(self) -> bool {
        match self {
            Self::Read | Self::Write | Self::ReadWrite | Self::Accumulate => false,
            Self::Consume => true,
        }
    }
}

impl Display for ReferenceAccessMode {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(formatter, "read"),
            Self::Write => write!(formatter, "write"),
            Self::ReadWrite => write!(formatter, "read/write"),
            Self::Accumulate => write!(formatter, "accumulate"),
            Self::Consume => write!(formatter, "consume"),
        }
    }
}

/// Kind of an allocation-preserving [`ReferenceOutput::Alias`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceAliasKind {
    /// The alias preserves the input handle's exact referent type and mapping.
    Identity,

    /// The alias selects a view of the input handle's allocation, and the aliasing operation itself carries the
    /// metadata that maps the new handle's coordinates onto that allocation. The generic program layer records only
    /// that the output aliases the input's allocation; interpreting and validating the operation-owned metadata is the
    /// job of the value family's reference discharge policy, which obtains it through the operation family's reference
    /// view contract.
    ///
    /// For example, [`ReferenceSliceOperation`](crate::ReferenceSliceOperation) declares `Alias { output_index: 0,
    /// input_index: 0, kind: View }` specifying that its result is a handle onto the same allocation whose referent
    /// is the sliced window, the slice axes live on the operation, and the array discharge policy reads those axes to
    /// materialize or reconstruct the selected coordinates during discharge.
    View,
}

/// Information about a [`Reference`](crate::Reference)-valued [`Operation`](crate::Operation) output.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceInput {
    /// Index of the [`Operation`](crate::Operation) input being accessed.
    input_index: usize,

    /// [`ReferenceAccessMode`] of this [`ReferenceInput`].
    mode: ReferenceAccessMode,
}

impl ReferenceInput {
    /// Creates a new [`ReferenceInput`].
    pub const fn new(input_index: usize, mode: ReferenceAccessMode) -> Self {
        Self { input_index, mode }
    }

    /// Returns the index of the [`Operation`](crate::Operation) input being accessed.
    pub const fn input_index(self) -> usize {
        self.input_index
    }

    /// Returns the [`ReferenceAccessMode`] of this [`ReferenceInput`].
    pub const fn mode(self) -> ReferenceAccessMode {
        self.mode
    }
}

/// Reference classification of a [`Reference`](crate::Reference)-valued [`Operation`](crate::Operation) output that is
/// either a fresh canonical allocation or an alias of one input's allocation. The two cases are mutually exclusive by
/// construction, so an output can never be declared as both a new allocation and an alias. [`Self::Alias`] carries
/// exactly one `input_index` because every reference operand must resolve to exactly one canonical allocation, so
/// multi-source aliases (e.g., a hypothetical `select_reference(a, b)`) are structurally unrepresentable rather than
/// merely rejected. [`ReferenceAliasKind`] distinguishes an identity-preserving edge from an operation-owned view edge.
/// Generic allocation analysis needs only that marker; the value family's discharge policy obtains and validates its
/// exact metadata through the operation family's view-operation contract.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceOutput {
    /// The output allocates a fresh resource and defines a new canonical allocation.
    Allocation {
        /// Index of the [`Operation`](crate::Operation) output defining the new allocation.
        output_index: usize,
    },

    /// The output aliases the canonical allocation of a [`Reference`](crate::Reference)-valued input.
    Alias {
        /// Index of the [`Operation`](crate::Operation) output producing the alias.
        output_index: usize,

        /// [`Operation`](crate::Operation) input index whose canonical allocation is preserved.
        input_index: usize,

        /// [`ReferenceAliasKind`] specifying whether this alias preserves the exact handle or adds an
        /// [`Operation`](crate::Operation)-owned view mapping.
        kind: ReferenceAliasKind,
    },
}

impl ReferenceOutput {
    /// Returns the index of the [`Operation`](crate::Operation) output this [`ReferenceOutput`] corresponds to.
    pub const fn output_index(self) -> usize {
        match self {
            Self::Allocation { output_index } | Self::Alias { output_index, .. } => output_index,
        }
    }
}

// Shared empty descriptor returned by `ReferenceOperationSemantics::empty` so that the `Operation` trait default can
// hand out a borrow without allocating (`Vec::new` is `const`, so this static needs no lazy initialization).
static EMPTY_REFERENCE_OPERATION_SEMANTICS: ReferenceOperationSemantics =
    ReferenceOperationSemantics { inputs: Vec::new(), outputs: Vec::new() };

/// [`Operation`](crate::Operation)-local reference semantics that describes the input [`Reference`](crate::Reference)
/// accesses and the output allocation/alias [`Reference`](crate::Reference) classifications, expressed in input/operand
/// and output/result index space. All indices are operation-local input/operand and output/result positions, never
/// resource identifiers. [`Program`](crate::Program)-level analysis later resolves them to canonical allocations (i.e.,
/// an entry input, a capture, or an allocation instruction), so the descriptor intentionally contains no runtime
/// resource IDs. The empty default describes operations that neither create, alias, nor access references.
///
/// # Examples
///
/// Array reference operations declare the following semantics:
///
/// ```text
/// reference_new(x) -> r
///     inputs  = []
///     outputs = [Allocation { output_index: 0 }]
///
/// reference_read(r) -> x
///     inputs  = [ReferenceInput { input_index: 0, mode: Read }]
///     outputs = []
///
/// reference_write(r, x) -> ()
///     inputs  = [ReferenceInput { input_index: 0, mode: Write }]
///     outputs = []
///
/// reference_swap(r, x) -> old
///     inputs  = [ReferenceInput { input_index: 0, mode: ReadWrite }]
///     outputs = []
///
/// reference_add_update(r, x) -> ()
///     inputs  = [ReferenceInput { input_index: 0, mode: Accumulate }]
///     outputs = []
///
/// reference_freeze(r) -> x
///     inputs  = [ReferenceInput { input_index: 0, mode: Consume }]
///     outputs = []
///
/// reference_index(r, axis, index) -> view
///     inputs  = []
///     outputs = [Alias { output_index: 0, input_index: 0, kind: View }]
///
/// reference_slice(r, axes) -> view
///     inputs  = []
///     outputs = [Alias { output_index: 0, input_index: 0, kind: View }]
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReferenceOperationSemantics {
    /// [`ReferenceInput`]s of the underlying [`Operation`](crate::Operation). Single Static Assignment (SSA) value
    /// (i.e., non-reference) inputs are omitted from this list.
    inputs: Vec<ReferenceInput>,

    /// [`ReferenceOutput`]s of the underlying [`Operation`](crate::Operation). Single Static Assignment (SSA) value
    /// (i.e., non-reference) outputs are omitted from this list.
    outputs: Vec<ReferenceOutput>,
}

impl ReferenceOperationSemantics {
    /// Creates a reference semantics descriptor from its ordered components.
    ///
    /// # Panics
    ///
    /// Panics when one input index receives two accesses or one output index receives two classifications. These are
    /// operation-author contract violations: the documented mutual exclusivity of [`ReferenceOutput`] holds
    /// only if each operand/result position appears at most once, and checked program construction trusts that
    /// invariant. Index ranges cannot be checked here because the descriptor carries no arity information; the
    /// builder validates them against each instruction's actual operand/result arity.
    pub fn new(inputs: Vec<ReferenceInput>, outputs: Vec<ReferenceOutput>) -> Self {
        for (index, access) in inputs.iter().enumerate() {
            let input_index = access.input_index();
            assert!(
                inputs[..index].iter().all(|previous| previous.input_index() != input_index),
                "input {input_index} received two reference accesses",
            );
        }
        for (index, output) in outputs.iter().enumerate() {
            let output_index = output.output_index();
            assert!(
                outputs[..index].iter().all(|previous| previous.output_index() != output_index),
                "output {output_index} received two reference classifications",
            );
        }
        Self { inputs, outputs }
    }

    /// Returns the shared empty [`ReferenceOperationSemantics`] used by [`Operation`](crate::Operation)s that neither
    /// create, alias, nor access [`Reference`](crate::Reference)s.
    #[inline]
    pub fn empty() -> &'static Self {
        &EMPTY_REFERENCE_OPERATION_SEMANTICS
    }

    /// Returns `true` if this [`ReferenceOperationSemantics`] declares no reference accesses and no reference outputs,
    /// and `false` otherwise.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inputs.is_empty() && self.outputs.is_empty()
    }

    /// Returns the [`ReferenceInput`]s of the underlying [`Operation`](crate::Operation) in operation-defined order.
    /// Single Static Assignment (SSA) value (i.e., non-reference) inputs are omitted from this list.
    #[inline]
    pub fn inputs(&self) -> &[ReferenceInput] {
        self.inputs.as_slice()
    }

    /// Returns the [`ReferenceOutput`]s of the underlying [`Operation`](crate::Operation) in operation-defined order.
    /// Single Static Assignment (SSA) value (i.e., non-reference) outputs are omitted from this list.
    #[inline]
    pub fn outputs(&self) -> &[ReferenceOutput] {
        self.outputs.as_slice()
    }

    /// Returns the output positions at which the corresponding [`Operation`](crate::Operation) allocates a fresh
    /// [`ReferenceOutput::Allocation`], in deterministic operation-defined order.
    #[inline]
    pub fn allocation_output_indices(&self) -> impl Iterator<Item = usize> {
        self.outputs.iter().filter_map(|output| match output {
            ReferenceOutput::Allocation { output_index } => Some(*output_index),
            ReferenceOutput::Alias { .. } => None,
        })
    }

    /// Validates that every position named by this [`ReferenceOperationSemantics`] exists in an
    /// [`Operation`](crate::Operation) application (i.e., an [`Instruction`](crate::Instruction)) with the provided
    /// input and output arity.
    ///
    /// # Parameters
    ///
    ///   - `operation_name`: Name of the operation whose descriptor is being validated, used for diagnostic purposes.
    ///   - `input_count`: Number of inputs/operands in the application.
    ///   - `output_count`: Number of outputs/results inferred for the application.
    pub(crate) fn validate_arity(
        &self,
        operation_name: &str,
        input_count: usize,
        output_count: usize,
    ) -> Result<(), ProgramError> {
        let validate_input = |input_index: usize, role: &str| {
            if input_index < input_count {
                return Ok(());
            }
            Err(ProgramError::MalformedProgram(format!(
                "operation `{operation_name}` names {role} input {input_index} but the application input count is \
                 {input_count}",
            )))
        };

        let validate_output = |output_index: usize| {
            if output_index < output_count {
                return Ok(());
            }
            Err(ProgramError::MalformedProgram(format!(
                "operation `{operation_name}` classifies output {output_index} but the application output count is \
                 {output_count}",
            )))
        };

        for access in &self.inputs {
            validate_input(access.input_index(), "an accessed")?;
        }

        for output in &self.outputs {
            validate_output(output.output_index())?;
            if let ReferenceOutput::Alias { input_index, .. } = output {
                validate_input(*input_index, "an aliased")?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::arrays::ArrayType;
    use crate::programs::operations::Operation;
    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::RegionInterface;
    use crate::programs::types::TypeError;

    use super::*;

    #[test]
    fn test_reference_access_mode() {
        let cases = [
            (ReferenceAccessMode::Read, "read", false),
            (ReferenceAccessMode::Write, "write", false),
            (ReferenceAccessMode::ReadWrite, "read/write", false),
            (ReferenceAccessMode::Accumulate, "accumulate", false),
            (ReferenceAccessMode::Consume, "consume", true),
        ];
        for (mode, display, is_consuming) in cases {
            assert_eq!(mode.to_string(), display);
            assert_eq!(mode.is_consuming(), is_consuming);
        }
    }

    #[test]
    fn test_reference_semantics_describes_an_alias_without_resource_identity() {
        #[derive(Clone)]
        struct TestAliasingOperation;

        impl Operation for TestAliasingOperation {
            type Type = ReferenceType<ArrayType>;

            fn name(&self) -> &'static str {
                "test_alias"
            }

            fn infer_output_types(
                &self,
                input_types: &[ReferenceType<ArrayType>],
                _region_interfaces: &[RegionInterface<ReferenceType<ArrayType>>],
            ) -> Result<Vec<ReferenceType<ArrayType>>, TypeError> {
                Ok(input_types.to_vec())
            }

            fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
                Cow::Owned(ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceOutput::Alias {
                        output_index: 0,
                        input_index: 0,
                        kind: ReferenceAliasKind::Identity,
                    }],
                ))
            }
        }

        let semantics = TestAliasingOperation.reference_semantics();
        assert!(semantics.inputs().is_empty());
        assert!(!semantics.is_empty());
        assert_eq!(
            semantics.outputs(),
            &[ReferenceOutput::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::Identity }],
        );
        assert_eq!(ReferenceOperationSemantics::empty(), &ReferenceOperationSemantics::default());
        assert!(ReferenceOperationSemantics::empty().is_empty());

        let boxed: Box<TestAliasingOperation> = Box::new(TestAliasingOperation);
        assert_eq!(boxed.reference_semantics(), TestAliasingOperation.reference_semantics());
    }

    #[test]
    fn test_reference_semantics_validates_application_arity() {
        let semantics = ReferenceOperationSemantics::new(
            vec![ReferenceInput::new(0, ReferenceAccessMode::Read)],
            vec![ReferenceOutput::Alias { output_index: 0, input_index: 1, kind: ReferenceAliasKind::Identity }],
        );
        assert_eq!(semantics.validate_arity("test.alias", 2, 1), Ok(()));

        let invalid_access =
            ReferenceOperationSemantics::new(vec![ReferenceInput::new(2, ReferenceAccessMode::Read)], Vec::new());
        assert_eq!(
            invalid_access.validate_arity("test.read", 2, 0),
            Err(ProgramError::MalformedProgram(
                "operation `test.read` names an accessed input 2 but the application input count is 2".to_string(),
            )),
        );

        let invalid_alias_input = ReferenceOperationSemantics::new(
            Vec::new(),
            vec![ReferenceOutput::Alias { output_index: 0, input_index: 1, kind: ReferenceAliasKind::Identity }],
        );
        assert_eq!(
            invalid_alias_input.validate_arity("test.alias", 1, 1),
            Err(ProgramError::MalformedProgram(
                "operation `test.alias` names an aliased input 1 but the application input count is 1".to_string(),
            )),
        );

        let invalid_alias_output = ReferenceOperationSemantics::new(
            Vec::new(),
            vec![ReferenceOutput::Alias { output_index: 1, input_index: 0, kind: ReferenceAliasKind::Identity }],
        );
        assert_eq!(
            invalid_alias_output.validate_arity("test.alias", 1, 1),
            Err(ProgramError::MalformedProgram(
                "operation `test.alias` classifies output 1 but the application output count is 1".to_string(),
            )),
        );

        let invalid_allocation =
            ReferenceOperationSemantics::new(Vec::new(), vec![ReferenceOutput::Allocation { output_index: 0 }]);
        assert_eq!(
            invalid_allocation.validate_arity("test.reference_new", 1, 0),
            Err(ProgramError::MalformedProgram(
                "operation `test.reference_new` classifies output 0 but the application output count is 0".to_string(),
            )),
        );
    }

    #[test]
    #[should_panic(expected = "output 0 received two reference classifications")]
    fn test_reference_semantics_rejects_two_classifications_for_one_output() {
        ReferenceOperationSemantics::new(
            Vec::new(),
            vec![
                ReferenceOutput::Allocation { output_index: 0 },
                ReferenceOutput::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::Identity },
            ],
        );
    }

    #[test]
    #[should_panic(expected = "input 0 received two reference accesses")]
    fn test_reference_semantics_rejects_two_accesses_for_one_input() {
        ReferenceOperationSemantics::new(
            vec![ReferenceInput::new(0, ReferenceAccessMode::Read), ReferenceInput::new(0, ReferenceAccessMode::Write)],
            Vec::new(),
        );
    }
}
