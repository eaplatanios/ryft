//! Operations that select values and control the execution of attached branch or loop programs. Each operation is
//! represented by an [`Operation`] type; value capabilities such as [`Condition`] and [`Select`]
//! apply the corresponding operation to eager arrays and traced values through the same API.
//!
//! The operations provide four forms of control flow:
//!
//!   - **Conditional Execution:** [`ConditionOperation`] evaluates the `true` or `false` branch according to a scalar
//!     Boolean predicate. [`Condition::condition`] traces two functions over structured inputs and returns the selected
//!     branch's outputs. Both functions are traced when the condition is built, while only the selected branch executes
//!     at runtime. [`ConditionType`] defines predicate and manual-variation constraints for each type universe.
//!   - **Elementwise Selection:** [`Select`] chooses between two already-computed values according to a Boolean mask.
//!     Input shapes broadcast and branch element types promote, so a scalar mask can select a whole array or a shaped
//!     mask can select individual elements. This does not defer either input's computation.
//!   - **Predicate-Controlled Loops:** [`WhileOperation`] repeatedly tests its condition region and applies its body to
//!     loop-carried state while the predicate is true. [`WhilePredicate`] also supports batched predicates through
//!     per-item masking. An explicit [`iteration_bound`](WhileOperation::with_iteration_bound) additionally permits
//!     reverse-mode differentiation through a masked tangent scan.
//!   - **Shape-Determined Loops:** [`ScanOperation`] threads loop-carried state through a body while consuming slices
//!     of stacked array inputs and producing stacked outputs. [`Scan::scan`] traces a body over structured state in an
//!     explicit context; [`scan()`] recovers the context from input values. Its body also receives the slice index;
//!     reference inputs supply complete roots. [`reverse`](ScanOperation::with_reverse) changes visit order without
//!     changing the pairing of input and output slices, while [`unroll`](ScanOperation::with_unroll) controls lowering.
//!
//! Branch and loop computations are [`Region`](crate::Region)s attached to an instruction, rather than fields of its
//! operation payload. [`WhileType`] and [`ScanType`] define their type-family contracts. The operation rules provide
//! interpretation, reference discharge, partial evaluation, batching, differentiation, and transposition.
//! Mapped-predicate batching evaluates both pure branches and selects their outputs per item, so branch computation
//! that is only valid when selected needs care under batching. A concrete predicate known during partial evaluation
//! instead specializes the condition to its selected branch.
//!
//! XLA lowers conditional execution to [StableHLO `if`](https://openxla.org/stablehlo/spec#if), and both loop forms to
//! [StableHLO `while`](https://openxla.org/stablehlo/spec#while). Individual operation types document their boundary,
//! effect and transformation constraints.
//!
//! # Examples
//!
//! Conditional execution traces branch functions over the input structure and selects their result at runtime:
//!
//! ```rust
//! # use ryft_core::{Array, Condition, ProgramError};
//! # fn main() -> Result<(), ProgramError> {
//! let predicate = Array::scalar(true)?;
//! let output = predicate.condition(
//!     |(first, second)| Ok(first + second),
//!     |(first, second)| Ok(first * second),
//!     (Array::scalar(2f32)?, Array::scalar(3f32)?),
//! )?;
//! assert_eq!(output, Array::scalar(5f32)?);
//! # Ok(())
//! # }
//! ```
//!
//! Elementwise selection chooses between values that have already been computed:
//!
//! ```rust
//! # use ryft_core::{Array, ProgramError, Select};
//! # fn main() -> Result<(), ProgramError> {
//! let mask = Array::vector(vec![true, false, true])?;
//! let values = Array::vector(vec![1f32, 2.0, 3.0])?;
//! let output = mask.select(&values, &Array::scalar(0f32)?)?;
//! assert_eq!(output, Array::vector(vec![1f32, 0.0, 3.0])?);
//! # Ok(())
//! # }
//! ```

use std::fmt::Display;
use std::ops::Range;

use crate::arrays::{ArrayIrType, ArrayType};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchedProgram, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy,
    ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::Context;
use crate::macros::check_count;
use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::dimensions::dimension_to_scalar::DimensionToScalarOperation;
use crate::programs::{
    Operation, ProgramError, ReferenceId, ReferenceRoot, RegionRef, Type, TypeError, TypeIdentityPosition,
    TypeRefinements, Value,
};

// TODO(eaplatanios): Review this module since it is mostly vibe coded and also add a module docstring that follows
//  our established conventions.

pub mod condition;
pub mod scan;
pub mod select;
pub mod r#while;

pub use condition::{CONDITION_OPERATION_NAME, Condition, ConditionOperation, ConditionType};
pub use scan::{SCAN_OPERATION_NAME, Scan, ScanInterpretationDomain, ScanOperation, ScanOptions, ScanType, scan};
pub use select::{SELECT_OPERATION_NAME, Select, SelectOperation};
pub use r#while::{WHILE_OPERATION_NAME, WhileOperation, WhilePredicate, WhileType};

/// Operations for storing a loop's intermediate values in stacks and recovering them when needed. When partial
/// evaluation or differentiation splits a loop body, the later computation may need values produced by the earlier
/// computation. Those saved values are _residuals_. [`ScanOperation`] partial evaluation and bounded [`WhileOperation`]
/// differentiation store per-iteration residuals in array stacks. Arrays already have a stackable representation;
/// first-class dimensions are converted to scalar arrays first:
///
/// ```text
///   dimension<n> --save--> i64[] --stack--> i64[iterations] --slice--> i64[] --restore--> dimension<n>
/// ```
///
/// Save and restore functions supply region-free unary operations with one output, or `None` when no conversion is
/// needed. [`Self::residual_storage_type`] infers the type of one stored slice from the save operation. References
/// cannot be stored as residual-stack entries; they must be threaded as loop carries.
///
/// If a consumer's dimension or ordinary array input already carries a saved dimension's identity, the consumer reuses
/// that input or calls [`Self::residual_from_carrier`] to recover the dimension from it. This avoids creating a second
/// definition of the same identity.
///
/// # Backend Integration
///
/// Operation families over [`ArrayType`] receive a blanket implementation that stores arrays unchanged.
/// Families over [`ArrayIrType`] receive one when they implement [`From`] for [`DimensionToScalarOperation`],
/// [`DimensionFromScalarOperation`], and [`DimensionSizeOperation`]. Those conversions give ordinary custom backends
/// the same array passthrough and first-class dimension encoding as the built-in array IR family; they do not need a
/// separate implementation of this trait and cannot supply one that overlaps the blanket implementation.
///
/// A backend with its own type universe can implement this trait on its operation family to provide the residual
/// representation used by scan partial evaluation and separate-context differentiation. The storage representation must
/// be an ordinary array in that universe, so callers can prepend the loop axis and recover individual slices. Save and
/// restore must preserve the logical value and its type, including nominal dimension identities. Array passthrough
/// preserves all array metadata. References must be rejected rather than encoded as array data. Carrier recovery is
/// optional; it must preserve the identity already carried by its input rather than introduce an independent identity.
/// This protocol supplies conversions only, and does not imply that the family supports every loop or differentiation
/// rule that consumes them.
pub trait ResidualStackOperation<T: Type>: Operation<Type = T> {
    /// Returns the type of one stored residual slice, inferred from [`Self::residual_to_storage`] when conversion is
    /// needed; otherwise returns `residual_type` unchanged. This is the type before the caller prepends the loop axis,
    /// not the type of the complete residual stack.
    ///
    /// # Errors
    ///
    /// Returns conversion selection or type inference errors, or a [`TypeError`] when the save operation does not
    /// return exactly one output.
    fn residual_storage_type(residual_type: &T) -> Result<T, TypeError> {
        let Some(operation) = Self::residual_to_storage(residual_type)? else {
            return Ok(residual_type.clone());
        };
        let mut output_types = operation.infer_output_types(std::slice::from_ref(residual_type), &[])?;
        check_count!("output", output_types, 1, TypeError);
        Ok(output_types.remove(0))
    }

    /// Returns the region-free unary operation that converts a value of type `residual_type` to its array storage
    /// representation, or [`None`] when that type already is its storage representation. The operation must return
    /// exactly one output, whose type determines [`Self::residual_storage_type`].
    ///
    /// # Errors
    ///
    /// Returns an error when this residual cannot be stored in an array stack, including for reference types.
    fn residual_to_storage(residual_type: &T) -> Result<Option<Self>, TypeError>;

    /// Returns the region-free unary operation that restores a value of logical type `residual_type` from
    /// one stored array slice, or [`None`] when no conversion is needed. Its input type must be the result of
    /// [`Self::residual_storage_type`] for `residual_type`, and its single output must have `residual_type`,
    /// preserving that type's nominal identities.
    ///
    /// # Errors
    ///
    /// Returns an error when this residual has no supported storage representation or restoration, including for
    /// reference types.
    fn residual_from_storage(residual_type: &T) -> Result<Option<Self>, TypeError>;

    /// Returns the operation that recovers a residual of type `residual_type` from a value of type `carrier_type`
    /// that already carries the residual's identity (e.g., the size of the array axis that a first-class dimension
    /// describes), or [`None`] when no supported recovery exists. The operation must be region-free, consume one value
    /// of `carrier_type`, and return one value of `residual_type` with the existing identity. Callers reuse a carrier
    /// of exactly `residual_type` directly. A rebuilt loop body uses recovery instead of restoring another definition
    /// of that identity from storage. The array IR implementation recovers dimensions from ordinary array axes and
    /// never reads a reference referent for this purpose.
    ///
    /// # Errors
    ///
    /// Returns an error when constructing a supported recovery operation fails.
    fn residual_from_carrier(residual_type: &T, carrier_type: &T) -> Result<Option<Self>, TypeError>;
}

impl<O: Operation<Type = ArrayType>> ResidualStackOperation<ArrayType> for O {
    #[inline]
    fn residual_to_storage(_residual_type: &ArrayType) -> Result<Option<Self>, TypeError> {
        Ok(None)
    }

    #[inline]
    fn residual_from_storage(_residual_type: &ArrayType) -> Result<Option<Self>, TypeError> {
        Ok(None)
    }

    #[inline]
    fn residual_from_carrier(_residual_type: &ArrayType, _carrier_type: &ArrayType) -> Result<Option<Self>, TypeError> {
        Ok(None)
    }
}

impl<
    O: Operation<Type = ArrayIrType>
        + From<DimensionFromScalarOperation>
        + From<DimensionSizeOperation>
        + From<DimensionToScalarOperation>,
> ResidualStackOperation<ArrayIrType> for O
{
    // Array IR residuals store arrays directly and first-class dimensions as scalar arrays.
    // References are threaded through loop carries instead of stored in residual stacks.

    fn residual_to_storage(residual_type: &ArrayIrType) -> Result<Option<Self>, TypeError> {
        Ok(match residual_type {
            ArrayIrType::Array(_) => None,
            ArrayIrType::Dimension(_) => Some(Self::from(DimensionToScalarOperation)),
            ArrayIrType::Reference(_) => {
                return Err(TypeError::invalid(
                    "a reference cannot be stored in a residual stack; references are threaded as carries",
                ));
            }
        })
    }

    fn residual_from_storage(residual_type: &ArrayIrType) -> Result<Option<Self>, TypeError> {
        Ok(match residual_type {
            ArrayIrType::Array(_) => None,
            ArrayIrType::Dimension(r#type) => Some(Self::from(DimensionFromScalarOperation::new(r#type.clone()))),
            ArrayIrType::Reference(_) => {
                return Err(TypeError::invalid(
                    "a reference cannot be stored in a residual stack; references are threaded as carries",
                ));
            }
        })
    }

    fn residual_from_carrier(
        residual_type: &ArrayIrType,
        carrier_type: &ArrayIrType,
    ) -> Result<Option<Self>, TypeError> {
        // Only array axes are recovered. A first-class dimension of the same type is the residual itself, which callers
        // reuse directly, and reading the extent of a reference's referent would need a reference access.
        let (ArrayIrType::Dimension(residual_type), ArrayIrType::Array(carrier_type)) = (residual_type, carrier_type)
        else {
            return Ok(None);
        };
        carrier_type
            .shape()
            .dimensions()
            .iter()
            .position(|dimension| {
                residual_type.variable().is_some_and(|variable| dimension.variable() == Some(variable))
            })
            .map(|axis| Ok(Self::from(DimensionSizeOperation::new(carrier_type, axis)?)))
            .transpose()
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Validates that instruction input `index` of the control-flow operation named `operation_name`, of type
/// `input_type`, may feed a region input of type `region_input_type`, i.e., that it refines that type (refer to
/// [`Type::is_refined_by`]). A reference input refines its region input type by its referent (e.g., a `ref<f32[5]>`
/// input feeds a `ref<f32[extent]>` region input), so the facts that its referent establishes (here, `extent = 5`)
/// take part in refining the operation's outputs exactly like the facts of a value input, and conflicting facts are
/// rejected (refer to [`refine_output_types`]). Its allocation keeps its own type: a reference output that aliases it
/// takes that type, and the region only sees the refined facts.
///
/// # Errors
///
/// Returns a [`TypeError`] that describes the region input type as `region_input_description` (e.g., `"state type"`)
/// when the input does not refine it.
pub(crate) fn validate_region_input<T: Type>(
    operation_name: &str,
    index: usize,
    input_type: &T,
    region_input_type: &T,
    region_input_description: &str,
) -> Result<(), TypeError> {
    if region_input_type.is_refined_by(input_type) {
        return Ok(());
    }
    Err(TypeError::invalid(format!(
        "`{operation_name}` input {index} has type `{input_type}`, which does not refine its \
         {region_input_description} `{region_input_type}`",
    )))
}

/// Returns the output types of a control-flow operation whose `input_types` refine the `declared_input_types` of its
/// regions, by applying the refinement facts that those inputs establish (e.g., `rows = 3` for an `f32[3]` input that
/// a region declares as `f32[rows]`) to the `declared_output_types`. The facts are established from the complete input
/// signature, so conflicting facts are rejected, but identities that an output type defines stay symbolic: a loop's
/// first-class dimension carry may change its extent across iterations, and a branch produces the identities that its
/// outputs define. A reference output whose aliased input is known (i.e., for which `aliased_input` returns that
/// input's index) takes exactly that input's type, so the output denotes the input's allocation at the allocation's own
/// type. Any other reference output (e.g., a `condition` output that either branch may forward from a different input)
/// is refined by the same facts as a value output: every reference input refines its region input type by its referent
/// (refer to [`validate_region_input`]), so a `ref<f32[extent]>` output of a region entered with a `ref<f32[5]>` input
/// becomes `ref<f32[5]>`. Metadata that is not a fact about an identity (e.g., a sharding or layout that only the
/// inputs carry) is never propagated to the outputs.
pub(crate) fn refine_output_types<T: Type>(
    declared_input_types: &[T],
    input_types: &[T],
    declared_output_types: &[T],
    aliased_input: impl Fn(usize) -> Option<usize>,
) -> Result<Vec<T>, TypeError> {
    let refinements = T::Refinements::new(declared_input_types, input_types)?;
    let symbolic_identities = defined_identities(declared_output_types);
    declared_output_types
        .iter()
        .enumerate()
        .map(|(index, declared_output_type)| match aliased_input(index) {
            Some(input_index) if declared_output_type.is_reference() => Ok(input_types[input_index].clone()),
            _ => refinements.refine_declared(declared_output_type, symbolic_identities.as_slice()),
        })
        .collect()
}

/// Returns the type identities that the `types` define (i.e., that occur at a [`TypeIdentityPosition::Definition`]
/// position), in order of occurrence and with repetitions. These are the identities that a region produces through
/// its outputs rather than receiving them through its inputs, so they stay symbolic when the facts that the inputs
/// establish refine those outputs.
pub(crate) fn defined_identities<T: Type>(types: &[T]) -> Vec<T::Identity> {
    types
        .iter()
        .flat_map(Type::identities)
        .filter_map(|(position, identity)| (position == TypeIdentityPosition::Definition).then(|| identity.clone()))
        .collect()
}

/// Validates that every type identity that one of the `output_types` of a control-flow operation refers to is either
/// carried by one of its `input_types` or defined by one of its `output_types`, so that the instruction only produces
/// types whose identities it consumes or defines. [`refine_output_types`] substitutes only the exact facts that the
/// inputs establish, so a refined output type can still refer to an identity that a region defines internally (e.g.,
/// the extent of an array computed from a `dimension_from_scalar` result that the region does not return) or to a
/// declared region identity that the inputs do not carry. This function rejects such an output as the final step of
/// `while`, `scan`, and `condition` type inference.
pub(crate) fn validate_output_identities<T: Type>(
    operation_name: &str,
    input_types: &[T],
    output_types: &[T],
) -> Result<(), TypeError> {
    let defined_identities = defined_identities(output_types);
    let input_identities =
        input_types.iter().flat_map(Type::identities).map(|(_, identity)| identity).collect::<Vec<_>>();
    for (index, output_type) in output_types.iter().enumerate() {
        if let Some((_, identity)) = output_type
            .identities()
            .find(|(_, identity)| !defined_identities.contains(*identity) && !input_identities.contains(identity))
        {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` output {index} has type `{output_type}`, which refers to the identity `{identity}` \
                 that no input carries and no output defines",
            )));
        }
    }
    Ok(())
}

/// Reference that a root of the reference analysis of a control-flow region (refer to
/// [`RegionRef::reference_analysis_with_configuration`]) denotes at the boundary of that region. Control-flow rules
/// compare these sources to validate which reference a region returns (e.g., that a loop body returns every reference
/// carry as the reference it entered with; refer to [`validate_reference_carry_roots`]).
///
/// [`Display`] describes sources relative to the region: input positions count region inputs independently of the
/// enclosing instruction's input layout.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum RegionReferenceSource {
    /// Input of the region at this position.
    Input(usize),

    /// Captured reference that names this capture position.
    Capture(usize),

    /// Concrete reference allocation embedded as a constant.
    Constant(ReferenceId),

    /// Reference allocated inside the region.
    Allocation,
}

impl RegionReferenceSource {
    /// Returns the [`RegionReferenceSource`] of every output of `region`, with [`None`] for each output that is not a
    /// reference, as the reference analysis of `region` with the capture prefix `capture_input_count` reports it
    /// (refer to [`Operation::region_capture_input_count`]).
    ///
    /// # Errors
    ///
    /// Returns an error when the reference analysis of `region` fails, or when it reports a root that is an input of a
    /// nested region or a constant that names neither a capture nor an allocation.
    pub(crate) fn of_outputs<V: Value, O: Operation<Type = V::Type>>(
        operation_name: &str,
        region: RegionRef<'_, V, O>,
        capture_input_count: Option<usize>,
    ) -> Result<Vec<Option<Self>>, ProgramError> {
        let analysis = region.reference_analysis_with_configuration(capture_input_count, true, &[])?;
        analysis
            .output_roots()
            .iter()
            .map(|root| root.map(|root| Self::from_root(operation_name, region, root)).transpose())
            .collect()
    }

    /// Returns the [`RegionReferenceSource`] that `root`, a root of the reference analysis of `region`, denotes.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when `root` is an input of a region other than `region`, or a
    /// constant root whose atom is not a reference constant that names a capture or an allocation.
    pub(crate) fn from_root<V: Value, O: Operation<Type = V::Type>>(
        operation_name: &str,
        region: RegionRef<'_, V, O>,
        root: ReferenceRoot,
    ) -> Result<Self, ProgramError> {
        match root {
            ReferenceRoot::RegionInput { region: owner, input_index } if owner == region.id() => {
                Ok(Self::Input(input_index))
            }
            ReferenceRoot::RegionInput { .. } => Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` region reference analysis reports a root that is an input of a nested region at \
                 the region boundary",
            ))),
            ReferenceRoot::Allocation { .. } => Ok(Self::Allocation),
            ReferenceRoot::Constant { value } => {
                let owner = region.with_id(value.region())?;
                match owner.atoms()[value.atom().index()].as_constant() {
                    Some(constant) => match (constant.capture_index(), constant.reference_id()) {
                        (Some(index), _) => Ok(Self::Capture(index)),
                        (None, Some(reference)) => Ok(Self::Constant(reference)),
                        (None, None) => Err(ProgramError::MalformedProgram(format!(
                            "`{operation_name}` region reference analysis reports a constant root that names neither \
                             a capture nor an allocation",
                        ))),
                    },
                    None => Err(ProgramError::MalformedProgram(format!(
                        "`{operation_name}` region reference analysis reports a constant root whose atom is not a \
                         constant",
                    ))),
                }
            }
        }
    }
}

impl Display for RegionReferenceSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(index) => write!(formatter, "region input {index}"),
            Self::Capture(index) => write!(formatter, "captured reference {index}"),
            Self::Constant(_) => formatter.write_str("a reference constant"),
            Self::Allocation => formatter.write_str("a reference allocated in the region"),
        }
    }
}

/// Validates that the `body` of the loop operation named `operation_name` returns every reference carry as the
/// reference that it entered with, i.e., as its own body input: carry `i` enters `body` at input `carry_inputs.start
/// + i` and leaves it at output `i`. A loop threads the allocation of a reference carry through every iteration and its
/// output denotes that same allocation, so a body that returns another reference (e.g., another reference carry, a
/// reference that it allocates, or a captured reference) gives the loop state no fixed point, and the outputs of a loop
/// that runs no iteration would disagree with those of every other trip count. The rule is structural: it compares body
/// positions rather than the allocations that the inputs of one application denote, so every rule (eager
/// interpretation, reference discharge, and the staged transforms alike) accepts the same bodies. Aliased handles
/// therefore satisfy it unless the body swaps them.
///
/// # Errors
///
/// Returns a [`TypeError`] naming the first carry that leaves the body as a different reference, and the errors of the
/// reference analysis of `body` otherwise (refer to [`RegionReferenceSource::of_outputs`]).
pub(crate) fn validate_reference_carry_roots<V: Value, O: Operation<Type = V::Type>>(
    operation_name: &str,
    body: RegionRef<'_, V, O>,
    carry_inputs: Range<usize>,
) -> Result<(), ProgramError> {
    let body_input_types = body.input_types();
    if !body_input_types[carry_inputs.clone()].iter().any(Type::is_reference) {
        return Ok(());
    }
    let sources = RegionReferenceSource::of_outputs(operation_name, body, None)?;
    for (position, input) in carry_inputs.enumerate() {
        let entering = body_input_types[input].is_reference().then_some(RegionReferenceSource::Input(input));
        if sources.get(position).copied().flatten() != entering {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` body does not return carry {position} as the reference it entered with, so the \
                 loop state has no fixed point",
            ))
            .into());
        }
    }
    Ok(())
}

/// Returns whether the body of a loop passes each of its carries through unchanged (i.e., returns the input atom of the
/// carry as its next value), where carry `i` enters the body at the input position `carry_inputs.start + i` and
/// leaves it at output position `i`. Such a carry holds its initial value on every iteration.
pub(crate) fn passed_through_carries<V: Value, O: Operation<Type = V::Type>>(
    body: RegionRef<'_, V, O>,
    carry_inputs: Range<usize>,
) -> Vec<bool> {
    carry_inputs
        .enumerate()
        .map(|(output, input)| body.output_ids()[output] == body.input_ids()[input])
        .collect()
}

/// Aligns every mapped carry among the `carries` of a batched loop to batch axis `0`, so that the fixed point over the
/// carry batch axes (refer to [`stabilize_carry_axes`]) only distinguishes replicated carries from carries batched at
/// axis `0`. A replicated carry stays replicated, and a reference carry keeps the batch axis that its referent fixes,
/// because shared storage cannot be moved.
pub(crate) fn canonicalize_carry_axes<C: Context, P: BatchingPolicy<C>, D: BatchingDriver<C, P>>(
    context: &BatchingContext<C, P>,
    driver: &D,
    carries: &[P::Batch],
) -> Result<Vec<P::Batch>, BatchingError> {
    carries
        .iter()
        .map(|carry| {
            if P::batch_axis(carry).is_replicated() || P::unbatched_type(carry).is_reference() {
                Ok(carry.clone())
            } else {
                driver.align_batch_axis(context, carry.clone(), Axis::from(0))
            }
        })
        .collect()
}

/// Batches the `body` of the loop operation named `operation_name` at `input_axes` while iterating the batch axes of
/// its carries, the body inputs at the positions `carry_inputs`, to a fixed point, and returns the batched body of the
/// pass that widens no carry. The leading body outputs are the next values of the carries. A loop's carry types are
/// loop-invariant, so a replicated carry whose next value is batched becomes batched at axis `0`, which this function
/// records in `input_axes`. Every pass instantiates the next-carry outputs at the current carry axes (refer to
/// [`ProgramBatchingOutputAxesPolicy::AlignEachTo`], whose replicated entries are lower bounds that keep the natural
/// axes of the other outputs), so the stabilizing pass is already aligned to the loop-invariant carry layout. A
/// reference carry is never widened: its referent fixes its axis, so its body must return it at that axis (refer to
/// [`validate_reference_carry_axis`]).
///
/// Every pass that does not converge widens at least one carry, so the fixed point converges within one more pass than
/// there are carries. The returned [`None`] reports a fixed point that did not converge within those passes, which only
/// a batching rule that reports widening non-monotonically can cause.
pub(crate) fn stabilize_carry_axes<C: Context, P: BatchingPolicy<C>, D: BatchingDriver<C, P>>(
    operation_name: &str,
    context: &BatchingContext<C, P>,
    driver: &D,
    body: RegionRef<'_, C::Constant, C::Operation>,
    input_axes: &mut [BatchAxis],
    carry_inputs: Range<usize>,
) -> Result<Option<P::BatchedProgram>, BatchingError> {
    let body_input_types = body.input_types();
    let carry_count = carry_inputs.len();
    for _ in 0..=carry_count {
        let output_axes = carry_inputs
            .clone()
            .map(|input| match body_input_types[input].is_reference() {
                true => BatchAxis::replicated(),
                false => input_axes[input],
            })
            .chain(std::iter::repeat_n(BatchAxis::replicated(), body.output_ids().len() - carry_count))
            .collect();
        let candidate = driver.batch_program(
            context,
            body,
            input_axes,
            ProgramBatchingOutputAxesPolicy::AlignEachTo(output_axes),
        )?;
        let mut widened = false;
        for (index, (input, output_axis)) in carry_inputs.clone().zip(candidate.output_axes()).enumerate() {
            if body_input_types[input].is_reference() {
                validate_reference_carry_axis(operation_name, index, input_axes[input], *output_axis)?;
            } else if input_axes[input].is_replicated() && !output_axis.is_replicated() {
                input_axes[input] = BatchAxis::new(0);
                widened = true;
            }
        }
        if !widened {
            return Ok(Some(candidate));
        }
    }
    Ok(None)
}

/// Requires a reference carry of a batched loop to leave its body carrying the batch axis it entered with. A
/// reference's batch axis is fixed by its referent, so the fixed-point iteration over carry axes can neither widen nor
/// move it, and a body that returns the carry at any other axis is rejected.
///
/// # Parameters
///
///   - `operation_name`: Name of the looping operation, used in the rejection diagnostic.
///   - `index`: Position of the reference carry among the loop's carries.
///   - `entering`: Batch axis the carry enters the body with.
///   - `returned`: Batch axis the batched body returns the carry with.
///
/// # Errors
///
/// Returns [`BatchingError::UnsupportedOperation`] when `returned` differs from `entering`.
pub(crate) fn validate_reference_carry_axis(
    operation_name: &str,
    index: usize,
    entering: BatchAxis,
    returned: BatchAxis,
) -> Result<(), BatchingError> {
    if entering == returned {
        return Ok(());
    }
    Err(BatchingError::UnsupportedOperation {
        message: format!(
            "`{operation_name}` reference carry {index} enters carrying `{entering}` but its body returns it carrying \
             `{returned}`; a reference carry cannot change its batch axis, so pass the reference as a batched input at \
             the axis the body produces",
        ),
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionBounds,
        DimensionType, DimensionValue, Memory,
    };
    use crate::axes::Axis;
    use crate::batching::{
        BatchAxis, BatchingContext, BatchingDriver, BatchingError, ProgramBatchingOutputAxesPolicy,
        RecursiveBatchingDriver, RecursiveBatchingPolicy,
    };
    use crate::captures::CaptureReference;
    use crate::contexts::Context;
    use crate::parameters::Placeholder;
    use crate::programs::{
        Atom, Operation, Program, ProgramError, ReferenceType, Region, RegionDriver, RegionLiveness, RegionRef,
        TypeError, Value,
    };

    use super::{ResidualStackOperation, SCAN_OPERATION_NAME, validate_reference_carry_axis};

    /// Array IR value, which admits references and first-class dimensions.
    pub(super) type TestIrValue = ArrayIrValue<Array>;

    /// Operation family of the array IR programs under test.
    pub(super) type TestIrOperation = ArrayIrOperation<Array>;

    /// Captured array IR value of the reference discharge fixtures.
    pub(super) type DischargeCapture = CaptureReference<ArrayIrType>;

    /// Operation family of the captured array IR programs in the reference discharge fixtures.
    pub(super) type DischargeCaptureOperation = ArrayIrOperation<CaptureReference<ArrayType>>;

    /// [`RegionLiveness`] of an operation with a single attached region, whose live inputs `used_inputs` derives from
    /// the used outputs of each query. The fixture records every query, so that region data flow tests can pin the
    /// fixed-point passes that a rule performs over its region.
    pub(super) struct TestRegionLiveness<F: FnMut(&[bool]) -> Vec<bool>> {
        /// Derives the live region inputs from the used region outputs.
        used_inputs: F,

        /// Used outputs of every query so far, in query order.
        queries: Vec<Vec<bool>>,
    }

    impl<F: FnMut(&[bool]) -> Vec<bool>> TestRegionLiveness<F> {
        /// Creates a new [`TestRegionLiveness`] whose live region inputs `used_inputs` derives.
        pub(super) fn new(used_inputs: F) -> Self {
            Self { used_inputs, queries: Vec::new() }
        }

        /// Returns the used outputs of every query so far, in query order.
        pub(super) fn queries(&self) -> &[Vec<bool>] {
            &self.queries
        }
    }

    impl<F: FnMut(&[bool]) -> Vec<bool>> RegionLiveness for TestRegionLiveness<F> {
        fn used_region_inputs(
            &mut self,
            region_index: usize,
            used_outputs: &[bool],
        ) -> Result<Vec<bool>, ProgramError> {
            assert_eq!(region_index, 0);
            self.queries.push(used_outputs.to_vec());
            Ok((self.used_inputs)(used_outputs))
        }
    }

    /// Wraps an [`Array`] as an array [`ArrayIrValue`].
    pub(super) fn array(value: Array) -> ArrayIrValue<Array> {
        ArrayIrValue::Array(value)
    }

    /// Returns a first-class dimension [`ArrayIrValue`] of type `r#type` whose runtime extent is `extent`.
    pub(super) fn dimension(r#type: &DimensionType, extent: usize) -> ArrayIrValue<Array> {
        ArrayIrValue::Dimension(DimensionValue::new(r#type.clone(), extent).unwrap())
    }

    /// Replaces every capture constant of a capture-lifted `program` with the concrete capture value it names, so that
    /// control-flow tests can interpret the discharged form of a captured program eagerly. Lifting turns the entry
    /// region's capture constants into leading inputs but keeps the capture constants of attached regions, which a
    /// backend resolves against the same leading capture arguments while lowering, and eager interpretation has no
    /// capture table to resolve them against.
    pub(super) fn resolve_captures<O: Operation<Type = ArrayIrType>>(
        program: &Program<
            CaptureReference<ArrayIrType>,
            O,
            Vec<CaptureReference<ArrayIrType>>,
            Vec<CaptureReference<ArrayIrType>>,
        >,
        captures: &[ArrayIrValue<Array>],
    ) -> Program<ArrayIrValue<Array>, O, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> {
        let regions = program
            .regions()
            .iter()
            .map(|region| {
                let atoms = region
                    .atoms()
                    .iter()
                    .map(|atom| match atom {
                        Atom::Constant(capture) => Atom::Constant(captures[capture.index()].clone()),
                        Atom::Variable(r#type) => Atom::Variable(r#type.clone()),
                    })
                    .collect();
                Region::new(
                    atoms,
                    region.input_ids().to_vec(),
                    region.output_ids().to_vec(),
                    region.instructions().to_vec(),
                )
            })
            .collect();
        Program::new(
            vec![Placeholder; program.input_count()],
            vec![Placeholder; program.output_count()],
            regions,
            program.entry(),
        )
        .unwrap()
    }

    /// [`BatchingDriver`] that counts the structural [`batch_program`](BatchingDriver::batch_program) requests a
    /// region-carrying batching rule makes, delegating every request to the ordinary [`RecursiveBatchingDriver`] over
    /// the same regions. Region-carrying rules discover their nested programs' natural output axes before instantiating
    /// them at reconciled targets, and reuse a discovery program when its axes already match. This fixture lets rule
    /// tests pin how many structural passes such a rule actually performs, which a program rendering alone cannot
    /// observe.
    pub(super) struct CountingBatchingDriver<'r, V: Value, O: Operation<Type = V::Type>> {
        /// [`Region`](crate::programs::Region)s attached to the operation application under test,
        /// in operation-defined order.
        regions: &'r Vec<Program<V, O, Vec<V>, Vec<V>>>,

        /// Number of structural program-batching requests observed so far.
        batch_program_calls: Cell<usize>,
    }

    impl<'r, V: Value, O: Operation<Type = V::Type>> CountingBatchingDriver<'r, V, O> {
        /// Creates a new [`CountingBatchingDriver`] over the provided attached regions.
        pub(super) fn new(regions: &'r Vec<Program<V, O, Vec<V>, Vec<V>>>) -> Self {
            Self { regions, batch_program_calls: Cell::new(0) }
        }

        /// Returns the number of structural program-batching requests observed so far.
        pub(super) fn batch_program_calls(&self) -> usize {
            self.batch_program_calls.get()
        }
    }

    impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for CountingBatchingDriver<'_, V, O> {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
        where
            V: 'r,
            O: 'r,
        {
            self.regions.regions()
        }
    }

    impl<C: Context, P: RecursiveBatchingPolicy<C>> BatchingDriver<C, P>
        for CountingBatchingDriver<'_, C::Constant, C::Operation>
    {
        fn batch_program(
            &self,
            context: &BatchingContext<C, P>,
            region: RegionRef<'_, C::Constant, C::Operation>,
            input_axes: &[BatchAxis],
            output_axes_policy: ProgramBatchingOutputAxesPolicy,
        ) -> Result<P::BatchedProgram, BatchingError> {
            self.batch_program_calls.set(self.batch_program_calls.get() + 1);
            RecursiveBatchingDriver::new(self.regions).batch_program(context, region, input_axes, output_axes_policy)
        }

        fn batch_region(
            &self,
            context: &BatchingContext<C, P>,
            index: usize,
            inputs: Vec<P::Batch>,
        ) -> Result<Vec<P::Batch>, BatchingError> {
            RecursiveBatchingDriver::new(self.regions).batch_region(context, index, inputs)
        }

        fn align_batch_axis(
            &self,
            context: &BatchingContext<C, P>,
            batch: P::Batch,
            axis: Axis,
        ) -> Result<P::Batch, BatchingError> {
            P::align_batch_axis(context, batch, axis)
        }
    }

    #[test]
    fn test_residual_stack_operation_residual_storage_type() {
        // Arrays keep their metadata in both families; dimensions use scalar array storage, and references are rejected.
        let array_type = ArrayType::new_static(DataType::F32, [2]).with_memory(Memory::Host { pinned: true });
        assert_eq!(ArrayOperation::<Array>::residual_storage_type(&array_type), Ok(array_type.clone()));
        assert_eq!(TestIrOperation::residual_storage_type(&array_type.clone().into()), Ok(array_type.clone().into()));

        let dimension_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        assert_eq!(
            TestIrOperation::residual_storage_type(&dimension_type.into()),
            Ok(ArrayType::scalar(DataType::I64).into()),
        );
        assert_eq!(
            TestIrOperation::residual_storage_type(&ReferenceType::new(array_type).into()),
            Err(TypeError::invalid(
                "a reference cannot be stored in a residual stack; references are threaded as carries",
            )),
        );
    }

    #[test]
    fn test_validate_reference_carry_axis() {
        // A reference carry that leaves the body at the axis it entered with is accepted whether mapped or replicated,
        // while any change of axis is rejected because the referent fixes the axis.
        assert_eq!(validate_reference_carry_axis(SCAN_OPERATION_NAME, 1, BatchAxis::new(0), BatchAxis::new(0)), Ok(()));
        assert_eq!(
            validate_reference_carry_axis(SCAN_OPERATION_NAME, 1, BatchAxis::replicated(), BatchAxis::replicated()),
            Ok(()),
        );
        assert_eq!(
            validate_reference_carry_axis(SCAN_OPERATION_NAME, 1, BatchAxis::new(0), BatchAxis::new(1)),
            Err(BatchingError::UnsupportedOperation {
                message:
                    "`scan` reference carry 1 enters carrying `axis 0` but its body returns it carrying `axis 1`; a \
                          reference carry cannot change its batch axis, so pass the reference as a batched input at \
                          the axis the body produces"
                        .to_string(),
            }),
        );
        assert_eq!(
            validate_reference_carry_axis(SCAN_OPERATION_NAME, 0, BatchAxis::replicated(), BatchAxis::new(0)),
            Err(BatchingError::UnsupportedOperation {
                message:
                    "`scan` reference carry 0 enters carrying `replicated` but its body returns it carrying `axis \
                          0`; a reference carry cannot change its batch axis, so pass the reference as a batched input \
                          at the axis the body produces"
                        .to_string(),
            }),
        );
    }
}
