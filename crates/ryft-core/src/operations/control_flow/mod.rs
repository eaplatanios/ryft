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
//!   - **Predicate-Controlled Loops:** [`WhileOperation`] repeatedly tests its condition region and applies its body
//!     to loop-carried state while the predicate is `true`. The `r#while` function of [`While`] traces a condition and
//!     a body over structured state in an explicit context; [`r#while()`](fn@while) recovers the context from state
//!     values. A batched predicate masks the state updates per item, which predicate values implement through
//!     [`WhilePredicate`]. An explicit [`iteration_bound`](WhileOperation::with_iteration_bound) truncates the
//!     loop by definition and additionally permits reverse-mode differentiation through a masked tangent scan.
//!   - **Shape-Determined Loops:** [`ScanOperation`] threads loop-carried state through a body while consuming slices
//!     of stacked array inputs and producing stacked outputs. [`Scan::scan`] traces a body over structured state in an
//!     explicit context; [`scan()`] recovers the context from input values. Its body also receives the slice index;
//!     reference inputs supply complete roots. [`reverse`](ScanOperation::with_reverse) changes visit order without
//!     changing the pairing of input and output slices, while [`unroll`](ScanOperation::with_unroll) controls lowering.
//!
//! Branch and loop computations are [`Region`](crate::Region)s attached to an instruction, rather than fields of its
//! operation payload. [`ConditionType`], [`WhileType`], and [`ScanType`] state the type-family contracts of conditions,
//! while loops, and scans once per type universe, and every rule of these operations validates its complete contract
//! before it executes, rewrites, or stages the attached regions. The operation rules provide interpretation, reference
//! discharge, partial evaluation, batching, differentiation, and transposition. Mapped-predicate batching evaluates
//! both pure branches and selects their outputs per item, so branch computation that is only valid when selected needs
//! care under batching. A concrete predicate known during partial evaluation instead specializes the condition to its
//! selected branch.
//!
//! The XLA backend lowers conditional execution to [StableHLO `if`](https://openxla.org/stablehlo/spec#if), and both
//! loop forms to [StableHLO `while`](https://openxla.org/stablehlo/spec#while). Individual operation types document
//! their boundary, effect and transformation constraints.
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

pub mod condition;
pub mod scan;
pub mod select;
pub mod r#while;

pub use condition::{CONDITION_OPERATION_NAME, Condition, ConditionOperation, ConditionType};
pub use scan::{
    SCAN_OPERATION_NAME, Scan, ScanInterpretationDomain, ScanIterationCount, ScanOperation, ScanOptions, ScanType, scan,
};
pub use select::{SELECT_OPERATION_NAME, Select, SelectOperation};
pub use r#while::{WHILE_OPERATION_NAME, While, WhileOperation, WhileOptions, WhilePredicate, WhileType, r#while};

/// Operations that convert loop residuals to and from their array storage representation. When partial evaluation
/// or differentiation splits a loop body, the later computation may need values produced by the earlier computation.
/// Those saved values are _residuals_. [`ScanOperation`] partial evaluation and bounded [`WhileOperation`]
/// differentiation store per-iteration residuals in array stacks. Arrays already have a stackable
/// representation; first-class dimensions are converted to scalar arrays first:
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
pub trait ResidualStorageOperation<T: Type>: Operation<Type = T> {
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

impl<O: Operation<Type = ArrayType>> ResidualStorageOperation<ArrayType> for O {
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
> ResidualStorageOperation<ArrayIrType> for O
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

/// Reference that a root of the reference analysis of a control flow region denotes at the boundary of that region.
/// Control flow rules compare these sources to validate which reference a region returns (e.g., that a loop body
/// returns every reference carry as the reference it entered with).
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
    /// Returns the [`RegionReferenceSource`] of every output of `region`, with [`None`] for each output that is not
    /// a reference, as the reference analysis of `region` with the capture prefix `capture_input_count` reports it
    /// (refer to [`Operation::region_capture_input_count`]).
    ///
    /// # Errors
    ///
    /// Returns an error when the reference analysis of `region` fails, or when it reports a root that is an input
    /// of a nested region or a constant that names neither a capture nor an allocation.
    pub(crate) fn from_outputs<V: Value, O: Operation<Type = V::Type>>(
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
    /// Returns [`ProgramError::MalformedProgram`] when `root` is an input of a region other than `region`,
    /// or a constant root whose atom is not a reference constant that names a capture or an allocation.
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

/// Validates that instruction input `index` of the control-flow operation named `operation_name`, of type
/// `input_type`, may feed a region input of type `region_input_type`, i.e., that it refines that type (refer to
/// [`Type::is_refined_by`]). A reference input refines its region input type by its referent (e.g., a `ref<f32[5]>`
/// input feeds a `ref<f32[extent]>` region input), so the facts that its referent establishes (here, `extent = 5`) take
/// part in refining the operation's outputs exactly like the facts of a value input, and conflicting facts are rejected
/// (refer to [`refine_control_flow_output_types`]). Its allocation keeps its own type: a reference output that aliases
/// it takes that type, and the region only sees the refined facts.
///
/// # Errors
///
/// Returns a [`TypeError`] that describes the region input type as `region_input_description` (e.g., `"state type"`)
/// when the input does not refine it.
pub(crate) fn validate_control_flow_region_input_refinement<T: Type>(
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

/// Returns the output types of a control flow operation whose `input_types` refine the `declared_input_types` of its
/// regions, by applying the refinement facts that those inputs establish (e.g., `rows = 3` for an `f32[3]` input that a
/// region declares as `f32[rows]`) to the `declared_output_types`. The facts are established from the complete input
/// signature, so conflicting facts are rejected, but identities that an output type defines stay symbolic: a loop's
/// first-class dimension carry may change its extent across iterations, and a branch produces the identities that its
/// outputs define. A reference output whose aliased input is known (i.e., for which `aliased_input` returns that
/// input's index) takes exactly that input's type, so the output denotes the input's allocation at the allocation's own
/// type. Any other reference output (e.g., a `condition` output that either branch may forward from a different input)
/// is refined by the same facts as a value output: every reference input refines its region input type by its referent
/// (refer to [`validate_control_flow_region_input_refinement`]), so a `ref<f32[extent]>` output of a region entered
/// with a `ref<f32[5]>` input becomes `ref<f32[5]>`. Metadata that is not a fact about an identity (e.g., a sharding or
/// layout that only the inputs carry) is never propagated to the outputs.
pub(crate) fn refine_control_flow_output_types<T: Type>(
    declared_input_types: &[T],
    input_types: &[T],
    declared_output_types: &[T],
    aliased_input: impl Fn(usize) -> Option<usize>,
) -> Result<Vec<T>, TypeError> {
    let refinements = T::Refinements::new(declared_input_types, input_types)?;
    let symbolic_identities = declared_output_types
        .iter()
        .flat_map(Type::identities)
        .filter(|&(position, _)| position == TypeIdentityPosition::Definition)
        .map(|(_, identity)| identity.clone())
        .collect::<Vec<_>>();
    declared_output_types
        .iter()
        .enumerate()
        .map(|(index, declared_output_type)| match aliased_input(index) {
            Some(input_index) if declared_output_type.is_reference() => Ok(input_types[input_index].clone()),
            _ => refinements.refine_declared(declared_output_type, symbolic_identities.as_slice()),
        })
        .collect()
}

/// Validates that every type identity that one of the `output_types` of a control-flow operation refers to is either
/// carried by one of its `input_types` or defined by one of its `output_types`, so that the instruction only produces
/// types whose identities it consumes or defines. [`refine_control_flow_output_types`] substitutes only the exact facts
/// that the inputs establish, so a refined output type can still refer to an identity that a region defines internally
/// (e.g., the extent of an array computed from a `dimension_from_scalar` result that the region does not return) or to
/// a declared region identity that the inputs do not carry. This function rejects such an output as the final step of
/// `while`, `scan`, and `condition` type inference.
pub(crate) fn validate_control_flow_output_identity_scope<T: Type>(
    operation_name: &str,
    input_types: &[T],
    output_types: &[T],
) -> Result<(), TypeError> {
    let defined_identities = output_types
        .iter()
        .flat_map(Type::identities)
        .filter(|&(position, _)| position == TypeIdentityPosition::Definition)
        .map(|(_, identity)| identity.clone())
        .collect::<Vec<_>>();
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
/// reference analysis of `body` otherwise (refer to [`RegionReferenceSource::from_outputs`]).
pub(crate) fn validate_loop_reference_carry_roots<V: Value, O: Operation<Type = V::Type>>(
    operation_name: &str,
    body: RegionRef<'_, V, O>,
    carry_inputs: Range<usize>,
) -> Result<(), ProgramError> {
    let body_input_types = body.input_types();
    if !body_input_types[carry_inputs.clone()].iter().any(Type::is_reference) {
        return Ok(());
    }
    let sources = RegionReferenceSource::from_outputs(operation_name, body, None)?;
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

/// Requires a reference carry of a batched loop to leave its body carrying the batch axis it entered with.
/// A reference's batch axis is fixed by its referent, so the fixed-point iteration over carry axes can neither
/// widen nor move it, and a body that returns the carry at any other axis is rejected.
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
pub(crate) fn validate_loop_reference_carry_batch_axis(
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

/// Returns whether the body of a loop passes each of its carries through unchanged (i.e., returns the input atom of
/// the carry as its next value), where carry `i` enters the body at the input position `carry_inputs.start + i` and
/// leaves it at output position `i`. Such a carry holds its initial value on every iteration.
pub(crate) fn loop_carry_passthrough_mask<V: Value, O: Operation<Type = V::Type>>(
    body: RegionRef<'_, V, O>,
    carry_inputs: Range<usize>,
) -> Vec<bool> {
    carry_inputs
        .enumerate()
        .map(|(output, input)| body.output_ids()[output] == body.input_ids()[input])
        .collect()
}

/// Aligns every mapped carry among the `carries` of a batched loop to batch axis `0`, so that the fixed point over the
/// carry batch axes (refer to [`stabilize_loop_carry_batch_axes`]) only distinguishes replicated carries from carries
/// batched at axis `0`. A replicated carry stays replicated, and a reference carry keeps the batch axis that its
/// referent fixes, because shared storage cannot be moved.
pub(crate) fn canonicalize_loop_carry_batch_axes<C: Context, P: BatchingPolicy<C>, D: BatchingDriver<C, P>>(
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
/// [`validate_loop_reference_carry_batch_axis`]).
///
/// Every pass that does not converge widens at least one carry, so the fixed point converges within one more pass than
/// there are carries. The returned [`None`] reports a fixed point that did not converge within those passes, which only
/// a batching rule that reports widening non-monotonically can cause.
pub(crate) fn stabilize_loop_carry_batch_axes<C: Context, P: BatchingPolicy<C>, D: BatchingDriver<C, P>>(
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
                validate_loop_reference_carry_batch_axis(operation_name, index, input_axes[input], *output_axis)?;
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::Cell;
    use std::fmt::Display;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation,
        ArrayReference, ArrayType, DataType, Dimension, DimensionBounds, DimensionError, DimensionType, DimensionValue,
        DimensionVariable, LogicalMesh, Memory, MeshAxis, MeshAxisType, Shape, Sharding,
    };
    use crate::axes::Axis;
    use crate::batching::{
        BatchAxis, BatchedProgram, BatchingContext, BatchingDriver, BatchingError, ProgramBatchingOutputAxesPolicy,
        RecursiveBatchingDriver, RecursiveBatchingPolicy,
    };
    use crate::captures::CaptureReference;
    use crate::contexts::{Context, EagerContext, StagingContext};
    use crate::operations::arithmetic::NegOperation;
    use crate::operations::dimensions::dimension_from_scalar::DimensionFromScalarOperation;
    use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
    use crate::operations::dimensions::dimension_to_scalar::DimensionToScalarOperation;
    use crate::operations::references::ReferenceNewOperation;
    use crate::parameters::{Parameter, Placeholder};
    use crate::programs::{
        Atom, AttachedRegions, InstructionId, Operation, Program, ProgramBuilder, ProgramError, ReferenceAnalysisError,
        ReferenceId, ReferenceRoot, ReferenceType, Region, RegionDriver, RegionId, RegionLiveness, RegionRef,
        TypeError, Typed, Value, ValueDirectDispatch, ValueId,
    };
    use crate::tracing::TracingContext;

    use super::{
        CONDITION_OPERATION_NAME, RegionReferenceSource, ResidualStorageOperation, SCAN_OPERATION_NAME,
        WHILE_OPERATION_NAME, canonicalize_loop_carry_batch_axes, loop_carry_passthrough_mask,
        refine_control_flow_output_types, stabilize_loop_carry_batch_axes, validate_control_flow_output_identity_scope,
        validate_control_flow_region_input_refinement, validate_loop_reference_carry_batch_axis,
        validate_loop_reference_carry_roots,
    };

    /// Array IR value, which admits references and first-class dimensions.
    pub(super) type TestIrValue = ArrayIrValue<Array>;

    /// Operation family of the array IR programs under test.
    pub(super) type TestIrOperation = ArrayIrOperation<Array>;

    /// Captured array IR value of the reference discharge fixtures.
    pub(super) type DischargeCapture = CaptureReference<ArrayIrType>;

    /// Operation family of the captured array IR programs in the reference discharge fixtures.
    pub(super) type DischargeCaptureOperation = ArrayIrOperation<CaptureReference<ArrayType>>;

    /// [`RegionLiveness`] of an operation with a single attached region, which derives the live region inputs from the
    /// used region outputs of each query through `used_inputs`. The fixture records every query, so that region data
    /// flow tests can pin the fixed-point passes that a rule performs over its region.
    pub(super) struct TestRegionLiveness<F: FnMut(&[bool]) -> Vec<bool>> {
        /// Derives the live region inputs from the used region outputs.
        used_inputs: F,

        /// Used outputs of every query so far, in query order.
        queries: Vec<Vec<bool>>,
    }

    impl<F: FnMut(&[bool]) -> Vec<bool>> TestRegionLiveness<F> {
        /// Creates a new [`TestRegionLiveness`] that derives the live region inputs through `used_inputs`.
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
        /// [`Region`]s attached to the operation application under test, in operation-defined order.
        regions: &'r AttachedRegions<'r, V, O>,

        /// Number of structural program-batching requests observed so far.
        batch_program_calls: Cell<usize>,
    }

    impl<'r, V: Value, O: Operation<Type = V::Type>> CountingBatchingDriver<'r, V, O> {
        /// Creates a new [`CountingBatchingDriver`] over the provided attached regions.
        pub(super) fn new(regions: &'r AttachedRegions<'r, V, O>) -> Self {
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

        fn region_count(&self) -> usize {
            self.regions.region_count()
        }

        fn region(&self, index: usize) -> Result<RegionRef<'_, V, O>, ProgramError> {
            self.regions.region(index)
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
    fn test_residual_storage_operation_residual_storage_type() {
        // Arrays keep their metadata in both families, dimensions use scalar array storage, and references are
        // rejected.
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
    fn test_residual_storage_operation_residual_to_storage() {
        // Arrays are stored unchanged in both families, dimensions are saved as scalar arrays, and references are
        // rejected.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        assert_eq!(ArrayOperation::<Array>::residual_to_storage(&array_type), Ok(None));
        assert_eq!(TestIrOperation::residual_to_storage(&array_type.clone().into()), Ok(None));
        assert_eq!(
            TestIrOperation::residual_to_storage(
                &DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap()).into(),
            ),
            Ok(Some(TestIrOperation::from(DimensionToScalarOperation))),
        );
        assert_eq!(
            TestIrOperation::residual_to_storage(&ReferenceType::new(array_type).into()),
            Err(TypeError::invalid(
                "a reference cannot be stored in a residual stack; references are threaded as carries",
            )),
        );
    }

    #[test]
    fn test_residual_storage_operation_residual_from_storage() {
        // Arrays are restored unchanged in both families, dimensions are restored from scalar arrays at their own
        // type, which preserves their identity, and references are rejected.
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        let dimension_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        assert_eq!(ArrayOperation::<Array>::residual_from_storage(&array_type), Ok(None));
        assert_eq!(TestIrOperation::residual_from_storage(&array_type.clone().into()), Ok(None));
        assert_eq!(
            TestIrOperation::residual_from_storage(&dimension_type.clone().into()),
            Ok(Some(TestIrOperation::from(DimensionFromScalarOperation::new(dimension_type)))),
        );
        assert_eq!(
            TestIrOperation::residual_from_storage(&ReferenceType::new(array_type).into()),
            Err(TypeError::invalid(
                "a reference cannot be stored in a residual stack; references are threaded as carries",
            )),
        );
    }

    #[test]
    fn test_residual_storage_operation_residual_from_carrier() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let other = DimensionVariable::new("other", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::from(DimensionType::from(extent.clone()));
        let carrier_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(3), Dimension::Dynamic(extent.clone())]));

        // Array IR recovers a dimension from the array axis that carries its identity (here, axis 1).
        assert_eq!(
            TestIrOperation::residual_from_carrier(&dimension_type, &carrier_type.clone().into()),
            Ok(Some(TestIrOperation::from(DimensionSizeOperation::new(&carrier_type, 1).unwrap()))),
        );

        // There is nothing to recover from an array without an axis of that identity, or for a static dimension,
        // which has no identity.
        assert_eq!(
            TestIrOperation::residual_from_carrier(
                &dimension_type,
                &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(other)])).into(),
            ),
            Ok(None),
        );
        assert_eq!(
            TestIrOperation::residual_from_carrier(
                &DimensionType::from(Dimension::Static(3)).into(),
                &ArrayType::new_static(DataType::F32, [3]).into(),
            ),
            Ok(None),
        );

        // Only array axes are carriers: a dimension of the residual type is reused directly by callers, and the
        // referent of a reference is never read for its extents. Array residuals need no recovery in either family.
        assert_eq!(TestIrOperation::residual_from_carrier(&dimension_type, &dimension_type), Ok(None));
        assert_eq!(
            TestIrOperation::residual_from_carrier(&dimension_type, &ReferenceType::new(carrier_type.clone()).into()),
            Ok(None),
        );
        assert_eq!(
            TestIrOperation::residual_from_carrier(&carrier_type.clone().into(), &carrier_type.clone().into()),
            Ok(None),
        );
        assert_eq!(ArrayOperation::<Array>::residual_from_carrier(&carrier_type, &carrier_type), Ok(None));
    }

    #[test]
    fn test_region_reference_source_from_outputs() {
        // A reference output reports the boundary source of its root, and every other output reports `None`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type.clone()).into());
        let value = builder.add_input(scalar_type.into());
        let allocation =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![value, reference, allocation],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        assert_eq!(
            RegionReferenceSource::from_outputs(SCAN_OPERATION_NAME, program.entry_region_ref(), None),
            Ok(vec![None, Some(RegionReferenceSource::Input(0)), Some(RegionReferenceSource::Allocation)]),
        );
    }

    #[test]
    fn test_region_reference_source_from_outputs_captures() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let capture = builder.add_constant(DischargeCapture::new(1, reference_type.into()));
        let program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![capture], Vec::new(), vec![Placeholder])
            .unwrap();

        // An inherited capture reports the capture position that it names.
        assert_eq!(
            RegionReferenceSource::from_outputs(SCAN_OPERATION_NAME, program.entry_region_ref(), None),
            Ok(vec![Some(RegionReferenceSource::Capture(1))]),
        );

        // An explicitly empty capture prefix leaves the capture unresolved, which the reference analysis rejects.
        assert_eq!(
            RegionReferenceSource::from_outputs(SCAN_OPERATION_NAME, program.entry_region_ref(), Some(0)),
            Err(ProgramError::from(ReferenceAnalysisError::InvalidReferenceCapture {
                region: program.entry(),
                atom: capture,
                capture_index: 1,
                capture_count: 0,
            })),
        );
    }

    #[test]
    fn test_region_reference_source_from_root() {
        /// Value family whose reference-typed program constants name a concrete allocation, or no allocation for
        /// [`None`]. Array IR values reject reference constants when a region is sealed, so this family stands in for
        /// one that admits them.
        #[derive(Clone, Debug)]
        struct ReferenceConstant(Option<ArrayReference<Array>>);

        impl Display for ReferenceConstant {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("reference_constant")
            }
        }

        impl Typed for ReferenceConstant {
            type Type = ArrayIrType;

            fn r#type(&self) -> Cow<'_, ArrayIrType> {
                Cow::Owned(ReferenceType::new(ArrayType::scalar(DataType::F32)).into())
            }
        }

        impl Parameter for ReferenceConstant {}

        impl Value for ReferenceConstant {
            type Dispatch = ValueDirectDispatch;
            type Domain = EagerContext<Self>;

            fn domain(&self) -> EagerContext<Self> {
                EagerContext::new()
            }

            fn reference_id(&self) -> Option<ReferenceId> {
                self.0.as_ref().map(ArrayReference::id)
            }
        }

        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let allocation = ArrayReference::new(Array::scalar(1f32).unwrap());
        let mut builder = ProgramBuilder::<ReferenceConstant, TestIrOperation>::new();
        let input = builder.add_input(reference_type.clone().into());
        let allocated = builder.add_constant(ReferenceConstant(Some(allocation.clone())));
        let unnamed = builder.add_constant(ReferenceConstant(None));
        let program = builder
            .build::<Vec<ReferenceConstant>, Vec<ReferenceConstant>>(
                vec![input, allocated, unnamed],
                vec![Placeholder],
                vec![Placeholder; 3],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let mut builder = ProgramBuilder::<DischargeCapture, DischargeCaptureOperation>::new();
        let capture = builder.add_constant(DischargeCapture::new(1, reference_type.into()));
        let captured_program = builder
            .build::<Vec<DischargeCapture>, Vec<DischargeCapture>>(vec![capture], Vec::new(), vec![Placeholder])
            .unwrap();
        let captured_region = captured_program.entry_region_ref();

        // Inputs of the region, allocations, and constants that name a capture or an allocation are boundary sources.
        assert_eq!(
            RegionReferenceSource::from_root(
                SCAN_OPERATION_NAME,
                region,
                ReferenceRoot::RegionInput { region: region.id(), input_index: 0 },
            ),
            Ok(RegionReferenceSource::Input(0)),
        );
        assert_eq!(
            RegionReferenceSource::from_root(
                SCAN_OPERATION_NAME,
                region,
                ReferenceRoot::Allocation { instruction: InstructionId::new(region.id(), 0), output_index: 0 },
            ),
            Ok(RegionReferenceSource::Allocation),
        );
        assert_eq!(
            RegionReferenceSource::from_root(
                SCAN_OPERATION_NAME,
                captured_region,
                ReferenceRoot::Constant { value: ValueId::new(captured_region.id(), capture) },
            ),
            Ok(RegionReferenceSource::Capture(1)),
        );
        assert_eq!(
            RegionReferenceSource::from_root(
                SCAN_OPERATION_NAME,
                region,
                ReferenceRoot::Constant { value: ValueId::new(region.id(), allocated) },
            ),
            Ok(RegionReferenceSource::Constant(allocation.id())),
        );

        // Roots that cannot denote a reference at the region boundary are malformed.
        assert_eq!(
            RegionReferenceSource::from_root(
                SCAN_OPERATION_NAME,
                region,
                ReferenceRoot::RegionInput { region: RegionId::new(1), input_index: 0 },
            ),
            Err(ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` region reference analysis reports a root that is an input of a nested region \
                 at the region boundary",
            ))),
        );
        assert_eq!(
            RegionReferenceSource::from_root(
                SCAN_OPERATION_NAME,
                region,
                ReferenceRoot::Constant { value: ValueId::new(RegionId::new(1), allocated) },
            ),
            Err(ProgramError::MalformedProgram(format!("region {} is out of range", RegionId::new(1)))),
        );
        assert_eq!(
            RegionReferenceSource::from_root(
                SCAN_OPERATION_NAME,
                region,
                ReferenceRoot::Constant { value: ValueId::new(region.id(), input) },
            ),
            Err(ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` region reference analysis reports a constant root whose atom is not a \
                 constant",
            ))),
        );
        assert_eq!(
            RegionReferenceSource::from_root(
                SCAN_OPERATION_NAME,
                region,
                ReferenceRoot::Constant { value: ValueId::new(region.id(), unnamed) },
            ),
            Err(ProgramError::MalformedProgram(format!(
                "`{SCAN_OPERATION_NAME}` region reference analysis reports a constant root that names neither a \
                 capture nor an allocation",
            ))),
        );
    }

    #[test]
    fn test_region_reference_source_display() {
        let allocation = ArrayReference::new(Array::scalar(1f32).unwrap());
        assert_eq!(RegionReferenceSource::Input(2).to_string(), "region input 2");
        assert_eq!(RegionReferenceSource::Capture(1).to_string(), "captured reference 1");
        assert_eq!(RegionReferenceSource::Constant(allocation.id()).to_string(), "a reference constant");
        assert_eq!(RegionReferenceSource::Allocation.to_string(), "a reference allocated in the region");
    }

    #[test]
    fn test_validate_control_flow_region_input_refinement() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let region_input_type =
            ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())])));
        let reference_region_input_type = ArrayIrType::from(ReferenceType::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent)]),
        )));

        // An input that refines its region input type is accepted, and a reference input refines a reference region
        // input type by its referent.
        assert_eq!(
            validate_control_flow_region_input_refinement(
                WHILE_OPERATION_NAME,
                0,
                &ArrayType::new_static(DataType::F32, [5]).into(),
                &region_input_type,
                "state type",
            ),
            Ok(()),
        );
        assert_eq!(
            validate_control_flow_region_input_refinement(
                WHILE_OPERATION_NAME,
                0,
                &ReferenceType::new(ArrayType::new_static(DataType::F32, [5])).into(),
                &reference_region_input_type,
                "state type",
            ),
            Ok(()),
        );

        // An input that does not refine its region input type is rejected, describing the region input as requested.
        assert_eq!(
            validate_control_flow_region_input_refinement(
                WHILE_OPERATION_NAME,
                1,
                &ArrayType::new_static(DataType::F64, [5]).into(),
                &region_input_type,
                "state type",
            ),
            Err(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` input 1 has type `f64[5]`, which does not refine its state type \
                 `f32[extent]`",
            ))),
        );
        assert_eq!(
            validate_control_flow_region_input_refinement(
                CONDITION_OPERATION_NAME,
                2,
                &ArrayType::new_static(DataType::F32, [5]).into(),
                &reference_region_input_type,
                "branch input type",
            ),
            Err(TypeError::invalid(format!(
                "`{CONDITION_OPERATION_NAME}` input 2 has type `f32[5]`, which does not refine its branch input type \
                 `ref<f32[extent]>`",
            ))),
        );
    }

    #[test]
    fn test_refine_control_flow_output_types() {
        // Output types take the facts that the inputs establish (here, `rows = 3`) but not the input metadata (here,
        // a sharding), except for a reference output that aliases a known input, which takes exactly that input's type.
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let declared_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let sharded_type =
            ArrayType::new_static(DataType::F32, [3]).with_sharding(Sharding::replicated(mesh, 1)).unwrap();
        assert_eq!(
            refine_control_flow_output_types(
                &[declared_type.clone().into(), ReferenceType::new(declared_type.clone()).into()],
                &[sharded_type.clone().into(), ReferenceType::new(sharded_type.clone()).into()],
                &[
                    declared_type.clone().into(),
                    ReferenceType::new(declared_type.clone()).into(),
                    ReferenceType::new(declared_type).into(),
                ],
                |index| (index == 1).then_some(1),
            ),
            Ok(vec![
                ArrayIrType::from(ArrayType::new_static(DataType::F32, [3])),
                ReferenceType::new(sharded_type).into(),
                ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into(),
            ]),
        );
    }

    #[test]
    fn test_refine_control_flow_output_types_rejects_conflicting_facts() {
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let declared_type =
            ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows)])));
        assert_eq!(
            refine_control_flow_output_types(
                &[declared_type.clone(), declared_type.clone()],
                &[ArrayType::new_static(DataType::F32, [3]).into(), ArrayType::new_static(DataType::F32, [4]).into()],
                &[declared_type],
                |_| None,
            ),
            Err(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "rows".to_string(),
                expected: 3,
                actual: 4,
            })),
        );
    }

    #[test]
    fn test_refine_control_flow_output_types_keeps_output_definitions_symbolic() {
        // An identity that an output type defines stays symbolic in every output, even though the inputs establish a
        // fact about it, while the other identities of the same output types are still refined.
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let rows = DimensionVariable::new("rows", DimensionBounds::positive(Some(8)).unwrap());
        let dimension_type = ArrayIrType::from(DimensionType::from(extent.clone()));
        let declared_input_types = [
            dimension_type.clone(),
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone())])).into(),
        ];
        let input_types =
            [DimensionType::from(Dimension::Static(5)).into(), ArrayType::new_static(DataType::F32, [3]).into()];
        let output_type = ArrayIrType::from(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Dynamic(rows)]),
        ));
        assert_eq!(
            refine_control_flow_output_types(
                &declared_input_types,
                &input_types,
                &[dimension_type.clone(), output_type],
                |_| None,
            ),
            Ok(vec![
                dimension_type,
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Dynamic(extent.clone()), Dimension::Static(3)]),
                )
                .into(),
            ]),
        );

        // Without an output that defines it, the same identity is refined to the extent that the inputs establish.
        assert_eq!(
            refine_control_flow_output_types(
                &declared_input_types,
                &input_types,
                &[ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])).into()],
                |_| None,
            ),
            Ok(vec![ArrayType::new_static(DataType::F32, [5]).into()]),
        );
    }

    #[test]
    fn test_validate_control_flow_output_identity_scope() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let other = DimensionVariable::new("other", DimensionBounds::positive(Some(8)).unwrap());
        let extent_type =
            ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent.clone())])));
        let other_type =
            ArrayIrType::from(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(other.clone())])));

        // Outputs may refer to identities that an input carries, including through the referent of a reference, and
        // to identities that an output defines.
        assert_eq!(
            validate_control_flow_output_identity_scope(
                CONDITION_OPERATION_NAME,
                &[ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])))
                    .into()],
                &[extent_type.clone(), DimensionType::from(other).into(), other_type.clone()],
            ),
            Ok(()),
        );

        // An output that refers to an identity that neither the inputs carry nor an output defines is rejected.
        assert_eq!(
            validate_control_flow_output_identity_scope(
                CONDITION_OPERATION_NAME,
                std::slice::from_ref(&extent_type),
                &[extent_type.clone(), other_type],
            ),
            Err(TypeError::invalid(format!(
                "`{CONDITION_OPERATION_NAME}` output 1 has type `f32[other]`, which refers to the identity `other` \
                 that no input carries and no output defines",
            ))),
        );
    }

    #[test]
    fn test_validate_loop_reference_carry_roots() {
        // A body that returns every reference carry as its own body input is accepted. Carries enter the body behind
        // its leading index input.
        let reference_type = ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let first = builder.add_input(reference_type.clone());
        let value = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let second = builder.add_input(reference_type);
        let next_value = builder
            .add_instruction(
                TestIrOperation::Array(ArrayOperation::Neg(NegOperation::new())),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first, next_value, second],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();
        assert_eq!(validate_loop_reference_carry_roots(SCAN_OPERATION_NAME, body.entry_region_ref(), 1..4), Ok(()));

        // Bodies without reference carries are not constrained, so their array carries may be returned in any order.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let first = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let second = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![second, first],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();
        assert_eq!(validate_loop_reference_carry_roots(SCAN_OPERATION_NAME, body.entry_region_ref(), 1..3), Ok(()));
    }

    #[test]
    fn test_validate_loop_reference_carry_roots_rejects_other_references() {
        // A body that returns a reference carry as another reference carry or as a reference that it allocates gives
        // the loop state no fixed point, and the first such carry is reported.
        let reference_type = ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = builder.add_input(reference_type.clone());
        let value = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let second = builder.add_input(reference_type.clone());
        let swapped_body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![second, value, first],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        assert_eq!(
            validate_loop_reference_carry_roots(WHILE_OPERATION_NAME, swapped_body.entry_region_ref(), 0..3),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` body does not return carry 0 as the reference it entered with, so the loop \
                 state has no fixed point",
            )))),
        );

        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = builder.add_input(reference_type.clone());
        let value = builder.add_input(ArrayType::scalar(DataType::F32).into());
        builder.add_input(reference_type);
        let allocation =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
        let allocating_body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first, value, allocation],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        assert_eq!(
            validate_loop_reference_carry_roots(WHILE_OPERATION_NAME, allocating_body.entry_region_ref(), 0..3),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{WHILE_OPERATION_NAME}` body does not return carry 2 as the reference it entered with, so the loop \
                 state has no fixed point",
            )))),
        );
    }

    #[test]
    fn test_validate_loop_reference_carry_batch_axis() {
        // A reference carry that leaves the body at the axis it entered with is accepted whether mapped or replicated,
        // while any change of axis is rejected because the referent fixes the axis.
        assert_eq!(
            validate_loop_reference_carry_batch_axis(SCAN_OPERATION_NAME, 1, BatchAxis::new(0), BatchAxis::new(0)),
            Ok(()),
        );
        assert_eq!(
            validate_loop_reference_carry_batch_axis(
                SCAN_OPERATION_NAME,
                1,
                BatchAxis::replicated(),
                BatchAxis::replicated(),
            ),
            Ok(()),
        );
        assert_eq!(
            validate_loop_reference_carry_batch_axis(SCAN_OPERATION_NAME, 1, BatchAxis::new(0), BatchAxis::new(1)),
            Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{SCAN_OPERATION_NAME}` reference carry 1 enters carrying `axis 0` but its body returns it \
                     carrying `axis 1`; a reference carry cannot change its batch axis, so pass the reference as a \
                     batched input at the axis the body produces",
                ),
            }),
        );
        assert_eq!(
            validate_loop_reference_carry_batch_axis(
                SCAN_OPERATION_NAME,
                0,
                BatchAxis::replicated(),
                BatchAxis::new(0),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{SCAN_OPERATION_NAME}` reference carry 0 enters carrying `replicated` but its body returns it \
                     carrying `axis 0`; a reference carry cannot change its batch axis, so pass the reference as a \
                     batched input at the axis the body produces",
                ),
            }),
        );
    }

    #[test]
    fn test_loop_carry_passthrough_mask() {
        // Only a carry whose next value is its own body input passes through. Returning another carry's input or a
        // computed value does not.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(ArrayType::scalar(DataType::I64));
        let first = builder.add_input(scalar_type.clone());
        builder.add_input(scalar_type.clone());
        let third = builder.add_input(scalar_type);
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![first, third, negated, first],
                vec![Placeholder; 4],
                vec![Placeholder; 4],
            )
            .unwrap();
        assert_eq!(loop_carry_passthrough_mask(body.entry_region_ref(), 1..4), vec![true, false, false]);
    }

    #[test]
    fn test_canonicalize_loop_carry_batch_axes() {
        // A replicated carry stays replicated and a mapped array carry moves to axis `0`, while a reference carry
        // keeps the axis that its referent fixes.
        let context = TracingContext::<TestIrValue, TestIrOperation>::new();
        let extent = context.input(DimensionType::from(Dimension::Static(2)).into());
        let replicated = context.input(ArrayType::new_static(DataType::F32, [3]).into());
        let mapped = context.input(ArrayType::new_static(DataType::F32, [3, 2]).into());
        let reference = context.input(ReferenceType::new(ArrayType::new_static(DataType::F32, [3, 2])).into());
        let batching_context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(context.clone(), extent.clone());
        let regions = AttachedRegions::from(Vec::<Program<TestIrValue, TestIrOperation, _, _>>::new());
        let carries = canonicalize_loop_carry_batch_axes(
            &batching_context,
            &RecursiveBatchingDriver::new(&regions),
            &[
                ArrayIrBatch::replicated(replicated),
                ArrayIrBatch::new(mapped, BatchAxis::new(1)).unwrap(),
                ArrayIrBatch::new(reference, BatchAxis::new(1)).unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(
            carries.iter().map(ArrayIrBatch::batch_axis).collect::<Vec<_>>(),
            vec![BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::new(1)],
        );
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                carries.iter().map(|carry| carry.value().atom_id().unwrap()).collect(),
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:f32[3], %2:f32[3, 2], %3:ref<f32[3, 2]> .
                let %4:f32[2, 3] = transpose [permutation=[1, 0]] %2
                in (%1, %4, %3)"},
        );
    }

    #[test]
    fn test_stabilize_loop_carry_batch_axes() {
        // The second carry's next value is the mapped input, so the first pass widens it, which in turn makes the next
        // value of the first carry mapped and widens it in the second pass. The third pass widens nothing and its
        // batched body is returned. The third carry passes through and stays replicated.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(scalar_type.clone());
        let second = builder.add_input(scalar_type.clone());
        let third = builder.add_input(scalar_type.clone());
        let input = builder.add_input(scalar_type);
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![second, negated, third, first],
                vec![Placeholder; 4],
                vec![Placeholder; 4],
            )
            .unwrap();
        let regions = AttachedRegions::from(vec![body]);
        let driver = CountingBatchingDriver::new(&regions);
        let context = BatchingContext::new(TracingContext::<Array, ArrayOperation<Array>>::new(), 2);
        let mut input_axes =
            vec![BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::new(0)];
        let batched_body = stabilize_loop_carry_batch_axes(
            SCAN_OPERATION_NAME,
            &context,
            &driver,
            driver.region(0).unwrap(),
            &mut input_axes,
            0..3,
        )
        .unwrap()
        .unwrap();
        assert_eq!(driver.batch_program_calls(), 3);
        assert_eq!(input_axes, vec![BatchAxis::new(0), BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)]);
        assert_eq!(
            batched_body.output_axes(),
            &[BatchAxis::new(0), BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::new(0)],
        );
        assert_eq!(
            batched_body.into_parts().0.to_string(),
            indoc! {"
                lambda %0:f32[2], %1:f32[2], %2:f32[], %3:f32[2] .
                let %4:f32[2] = neg %3
                in (%1, %4, %2, %0)"},
        );
    }

    #[test]
    fn test_stabilize_loop_carry_batch_axes_reference_carries() {
        let reference_type = ArrayIrType::from(ReferenceType::new(ArrayType::new_static(DataType::F32, [3])));
        let context = TracingContext::<TestIrValue, TestIrOperation>::new();
        let extent = context.input(DimensionType::from(Dimension::Static(2)).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(context, extent);

        // A reference carry is never widened, so a body that returns each reference carry at the axis it entered with
        // converges in one pass and leaves the carry axes unchanged.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = builder.add_input(reference_type.clone());
        let second = builder.add_input(reference_type.clone());
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![first, second],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let regions = AttachedRegions::from(vec![body]);
        let driver = CountingBatchingDriver::new(&regions);
        let mut input_axes = vec![BatchAxis::new(0), BatchAxis::replicated()];
        let batched_body = stabilize_loop_carry_batch_axes(
            WHILE_OPERATION_NAME,
            &context,
            &driver,
            driver.region(0).unwrap(),
            &mut input_axes,
            0..2,
        )
        .unwrap()
        .unwrap();
        assert_eq!(driver.batch_program_calls(), 1);
        assert_eq!(input_axes, vec![BatchAxis::new(0), BatchAxis::replicated()]);
        assert_eq!(batched_body.output_axes(), &[BatchAxis::new(0), BatchAxis::replicated()]);
        assert_eq!(
            batched_body.into_parts().0.to_string(),
            indoc! {"
                lambda %0:dimension<2>, %1:ref<f32[2, 3]>, %2:ref<f32[3]> .
                in (%0, %1, %2)"},
        );

        // A body that returns a reference carry at another axis is rejected.
        let mut builder = ProgramBuilder::<TestIrValue, TestIrOperation>::new();
        let first = builder.add_input(reference_type.clone());
        let second = builder.add_input(reference_type);
        let body = builder
            .build::<Vec<TestIrValue>, Vec<TestIrValue>>(
                vec![second, first],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let regions = AttachedRegions::from(vec![body]);
        let driver = RecursiveBatchingDriver::new(&regions);
        let mut input_axes = vec![BatchAxis::new(0), BatchAxis::replicated()];
        assert!(matches!(
            stabilize_loop_carry_batch_axes(
                WHILE_OPERATION_NAME,
                &context,
                &driver,
                driver.region(0).unwrap(),
                &mut input_axes,
                0..2,
            ),
            Err(BatchingError::UnsupportedOperation { message }) if message == format!(
                "`{WHILE_OPERATION_NAME}` reference carry 0 enters carrying `axis 0` but its body returns it carrying \
                 `replicated`; a reference carry cannot change its batch axis, so pass the reference as a batched \
                 input at the axis the body produces",
            ),
        ));
    }
}
