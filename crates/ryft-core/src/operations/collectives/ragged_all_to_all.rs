//! Contains the named-axis [`RaggedAllToAllOperation`], which exchanges variable-length leading-axis segments between
//! participants, together with its type contract, eager reference interpretation, batching gate, and staging
//! capability.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;

use crate::arrays::batching::DynamicArrayBatchingPolicy;
use crate::arrays::{
    Array, ArrayBatch, ArrayBatching, ArrayIrBatch, ArrayIrBatching, ArrayIrType, ArrayType, DataType, Dimension,
    DimensionVariable, Shape,
};
use crate::axes::NamedAxes;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    MemberBatchableOperation, batch_projected_operation,
};
use crate::contexts::{Context, Domain, ProjectedContext, ValueResolution};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    MemberDifferentiableOperation, TransposableOperation, TranspositionContext, TranspositionDriver,
    jvp_projected_operation,
};
use crate::interpretation::{
    InterpretableOperation, InterpretationDriver, MemberInterpretableOperation, interpret_projected_operation,
};
use crate::macros::check_count;
use crate::operations::compare::{Compare, CompareOperation};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::iota::IotaOperation;
use crate::operations::constants::one::{One, OneOperation};
use crate::operations::constants::zero::{Zero, ZeroOperation};
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::select::{Select, SelectOperation};
use crate::operations::cumulative::cumulative_sum::{CumulativeSum, CumulativeSumOperation};
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::operations::manipulation::concatenation::{Concatenate, ConcatenateOperation};
use crate::operations::manipulation::conversion::{ConvertElementType, ConvertElementTypeOperation};
use crate::operations::manipulation::memory::{TransferToMemory, TransferToMemoryOperation};
use crate::operations::manipulation::reshaping::{Reshape, ReshapeOperation};
use crate::operations::manipulation::scattering::{
    Scatter, ScatterDimensionNumbers, ScatterOperation, ScatterReductionKind,
};
use crate::operations::manipulation::slicing::{Slice, SliceOperation};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::add::{Add, AddOperation};
use crate::operations::math::mul::{Mul, MulOperation};
use crate::operations::math::neg::{Neg, NegOperation};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EmptyRegionDriver, MaybeZero, MemberOperation, Operation, OperationFormatter, OperationProjection, ProgramError,
    ProvenanceScope, RegionInterface, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
    infer_projected_operation_output_types, infer_projected_operation_region_input_types,
};
use crate::tracing::{Tracer, TracingContext};

use super::all_to_all::AllToAllOperation;
use super::{
    CollectiveBatchingPolicy, CollectiveOptions, effective_collective_axis_size, reject_ragged_collective_inputs,
    resolve_named_axis_size,
};

/// Canonical name of the [`RaggedAllToAllOperation`].
pub const RAGGED_ALL_TO_ALL_OPERATION_NAME: &str = "ragged_all_to_all";

/// Operand representation carried by [`RaggedAllToAllOperation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
enum RaggedAllToAllRepresentation {
    /// Public per-participant representation with rank-one metadata operands.
    Logical,

    /// Batching-internal representation with one leading participant axis on every operand.
    Physical,
}

/// Update semantics used by the public forward exchange and its internal operand adjoint.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum RaggedAllToAllUpdateKind {
    /// Received segments replace the corresponding output seed regions.
    Overwrite,

    /// Received segments are added into the corresponding output seed regions.
    Add,
}

/// Reference-value capability used by eager interpretation of [`RaggedAllToAllOperation`].
///
/// The public [`RaggedAllToAll`] trait stages the operation through a named-axis context. This narrower crate-owned
/// capability instead executes already-materialized values in either the public per-participant representation or
/// the explicitly marked internal batching representation.
pub(crate) trait RaggedAllToAllEvaluation: Sized {
    /// Executes `operation` over the six operands in their canonical order.
    fn evaluate_ragged_all_to_all(
        operation: &RaggedAllToAllOperation,
        operand: &Self,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
    ) -> Result<Self, ProgramError>;
}

/// Primitive that exchanges variable-length leading-axis segments between participants of a named axis.
///
/// The six operands, in order, are `operand (N, A, ...)`, `output (M, A, ...)`, and rank-one integer arrays
/// `input_offsets`, `send_sizes`, `output_offsets`, and `receive_sizes`, each of length `K`. The result has exactly
/// `output`'s type and starts with `output`'s value, so elements outside received regions pass through unchanged.
/// `K` must be positive and divisible by the effective participant-group size.
/// The batching rule uses one internal physical form that prefixes every operand with the participant axis, making
/// the data operands `(P, N, A, ...)` and `(P, M, A, ...)` and the metadata operands `(P, K)`; this representation is
/// normalized back to the public contract during type inference and eager interpretation.
///
/// `output_offsets` are supplied by each sender but are expressed in the corresponding receiver's
/// coordinate frame. Runtime metadata must satisfy
/// `send_sizes == all_to_all(receive_sizes)`, every source and destination region must be in bounds, and received
/// regions within one output must be disjoint. Send regions may overlap, which intentionally permits resending a
/// source slice. Concrete eager execution validates these conditions with overflow-safe host arithmetic. Staged XLA
/// execution treats them as preconditions, matching
/// [JAX's `ragged_all_to_all`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.ragged_all_to_all.html).
///
/// [`RaggedAxis`](crate::arrays::RaggedAxis) is batching-time metadata and does not participate in this explicitly
/// packed operation contract. Batching rejects operands that carry it because one per-item logical extent does not
/// determine the per-source/per-destination sizes and two coordinate-frame offset vectors required by this operation.
/// A future adapter must therefore accept an explicit routing descriptor; it cannot infer routing from
/// [`RaggedAxis`](crate::arrays::RaggedAxis) alone. The two representations also describe different frames:
/// `RaggedAllToAllOperation` metadata describes participant chunks, whereas a `RaggedAxis` describes packed batch
/// items and has no carrier outside a batching transform.
/// A batch transform whose named axis matches this operation executes concrete array metadata eagerly. Unresolved
/// non-constant metadata are deliberately gated: dynamic-length copies cannot use
/// [`DynamicSliceOperation`](crate::DynamicSliceOperation), whose slice sizes are static payload fields. A future
/// staged implementation can express the copies with iota-based index arithmetic, gather, and `select` masking at
/// `O(group_size × M)` staged work.
///
/// The transpose stages additional collectives rather than performing a local rewrite, and its metadata operands are
/// primal residuals: ordinary runtime values retained by linearization, not compile-time constants. Participant groups
/// are forwarded through both dense metadata exchanges and the adjoint ragged exchange. This deliberately corrects
/// JAX's grouped transpose, which accepts groups on the ragged primitive but omits them from its offset exchanges.
/// The output leading dimension must currently be static so the `M + 1` marker and its final slice can be represented
/// by the existing static scatter and slice operations.
///
/// Batching over an unrelated mapped axis currently requires a static mapped extent and statically shaped data
/// operands. Offset rebasing stages `N` and `M` as scalar constants, and the shared reshape interface cannot yet
/// recover dynamic trailing extents from both homogeneous and composite carriers. Grouped operation batching is
/// rejected in this case because merging an unrelated mapped axis would change the meaning of each fixed participant
/// group.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedAllToAllOperation {
    /// Axis name referenced by this collective.
    axis_name: String,

    /// Full number of participants along the named axis, resolved when the operation is staged.
    axis_size: usize,

    /// Optional ordered partition of logical participant indices.
    axis_index_groups: Option<Vec<Vec<usize>>>,

    /// Public logical or batching-internal physical operand representation.
    representation: RaggedAllToAllRepresentation,

    /// Public overwrite or transpose-internal additive update semantics.
    update_kind: RaggedAllToAllUpdateKind,
}

impl RaggedAllToAllOperation {
    /// Creates an ungrouped operation over the named axis with the provided resolved axis size.
    #[inline]
    pub fn new(axis_name: String, axis_size: usize) -> Self {
        Self {
            axis_name,
            axis_size,
            axis_index_groups: None,
            representation: RaggedAllToAllRepresentation::Logical,
            update_kind: RaggedAllToAllUpdateKind::Overwrite,
        }
    }

    /// Creates a grouped operation after validating that `axis_index_groups` is an equal-sized exact partition of
    /// `0..axis_size`.
    pub fn grouped(axis_name: String, axis_size: usize, axis_index_groups: Vec<Vec<usize>>) -> Result<Self, TypeError> {
        effective_collective_axis_size(
            RAGGED_ALL_TO_ALL_OPERATION_NAME,
            axis_size,
            Some(axis_index_groups.as_slice()),
        )?;
        Ok(Self {
            axis_name,
            axis_size,
            axis_index_groups: Some(axis_index_groups),
            representation: RaggedAllToAllRepresentation::Logical,
            update_kind: RaggedAllToAllUpdateKind::Overwrite,
        })
    }

    /// Returns the axis name referenced by this collective.
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }

    /// Returns the full number of participants along the named axis.
    #[inline]
    pub fn axis_size(&self) -> usize {
        self.axis_size
    }

    /// Returns the ordered participant groups, if any.
    #[inline]
    pub fn axis_index_groups(&self) -> Option<&[Vec<usize>]> {
        self.axis_index_groups.as_deref()
    }

    /// Validates the participant partition and returns its common group size.
    #[inline]
    pub fn effective_axis_size(&self) -> Result<usize, TypeError> {
        effective_collective_axis_size(RAGGED_ALL_TO_ALL_OPERATION_NAME, self.axis_size, self.axis_index_groups())
    }

    /// Returns whether this operation carries the batching-internal physical operand representation.
    ///
    /// This predicate is exposed for backend lowerings, which must reject the host-batching representation before
    /// emitting a custom call whose operands are local to one device participant.
    #[doc(hidden)]
    #[inline]
    pub fn is_physical(&self) -> bool {
        self.representation == RaggedAllToAllRepresentation::Physical
    }

    /// Returns a clone marked with the batching-internal physical operand representation.
    #[inline]
    fn with_physical_representation(&self) -> Self {
        Self { representation: RaggedAllToAllRepresentation::Physical, ..self.clone() }
    }

    /// Returns a clone whose received segments add into the output seed.
    #[inline]
    fn with_additive_updates(&self) -> Self {
        Self { update_kind: RaggedAllToAllUpdateKind::Add, ..self.clone() }
    }

    /// Returns the update semantics carried by this operation.
    #[inline]
    pub(crate) fn update_kind(&self) -> RaggedAllToAllUpdateKind {
        self.update_kind
    }

    /// Returns whether received segments add into the output seed instead of overwriting it.
    ///
    /// This mode is produced only by the transpose rule. It is exposed so backend lowerings can preserve the
    /// accumulation semantics without exposing the internal update-kind representation itself.
    #[doc(hidden)]
    #[inline]
    pub fn accumulates_updates(&self) -> bool {
        self.update_kind == RaggedAllToAllUpdateKind::Add
    }
}

impl Display for RaggedAllToAllOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for RaggedAllToAllOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        RAGGED_ALL_TO_ALL_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        check_count!("input", input_types, 6, TypeError);
        let effective_axis_size = self.effective_axis_size()?;
        let result_type = input_types[1].clone();
        let batched = self.is_physical();
        let normalized_input_types = if batched {
            Some(
                input_types
                    .iter()
                    .enumerate()
                    .map(|(index, input_type)| {
                        let Some(participant_extent) =
                            input_type.shape().dimensions().first().and_then(|extent| extent.value())
                        else {
                            return Err(TypeError::invalid(format!(
                                "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` physical operand {index} must have a static \
                                 leading participant dimension",
                            )));
                        };
                        if participant_extent != self.axis_size {
                            return Err(TypeError::invalid(format!(
                                "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` physical operand {index} leading participant \
                                 dimension {participant_extent} must equal axis size {}",
                                self.axis_size,
                            )));
                        }
                        Ok(input_type.without_dimension(0)?.0)
                    })
                    .collect::<Result<Vec<_>, TypeError>>()?,
            )
        } else {
            None
        };
        let input_types = normalized_input_types.as_deref().unwrap_or(input_types);
        let [operand, output, input_offsets, send_sizes, output_offsets, receive_sizes] = input_types else {
            unreachable!();
        };

        if operand.rank() == 0 || output.rank() == 0 {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` data operands must have rank at least 1 but got `{operand}` and \
                 `{output}`",
            )));
        }
        if operand.data_type() != output.data_type() {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` operand and output data types must match but got `{}` and `{}`",
                operand.data_type(),
                output.data_type(),
            )));
        }
        if operand.shape().dimensions()[1..] != output.shape().dimensions()[1..] {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` operand and output trailing dimensions must match but got `{}` \
                 and `{}`",
                operand.shape(),
                output.shape(),
            )));
        }

        let metadata = [
            ("input_offsets", input_offsets),
            ("send_sizes", send_sizes),
            ("output_offsets", output_offsets),
            ("receive_sizes", receive_sizes),
        ];
        let metadata_data_type = input_offsets.data_type();
        let mut metadata_length = None;
        for (name, r#type) in metadata {
            if r#type.rank() != 1 {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` `{name}` must be rank 1 but got `{type}`",
                    r#type = r#type,
                )));
            }
            if !r#type.data_type().is_integer() {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` `{name}` must have an integer data type but got `{}`",
                    r#type.data_type(),
                )));
            }
            if r#type.data_type() != metadata_data_type {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` metadata operands must share one integer data type but \
                     `input_offsets` has `{metadata_data_type}` and `{name}` has `{}`",
                    r#type.data_type(),
                )));
            }
            let Some(length) = r#type.shape().dimensions()[0].value() else {
                return Err(TypeError::invalid(format!(
                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` `{name}` must have a static length but got `{type}`",
                    r#type = r#type,
                )));
            };
            match metadata_length {
                Some(expected) if length != expected => {
                    return Err(TypeError::invalid(format!(
                        "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` metadata operands must have equal lengths but \
                         `input_offsets` has length {expected} and `{name}` has length {length}",
                    )));
                }
                None => metadata_length = Some(length),
                _ => {}
            }
        }
        let metadata_length = metadata_length.unwrap();
        if metadata_length == 0 {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` metadata length must be greater than zero",
            )));
        }
        if metadata_length % effective_axis_size != 0 {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` metadata length {metadata_length} is not divisible by group \
                 size {effective_axis_size}",
            )));
        }
        Ok(vec![result_type])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
            operation.field("axis_size", self.axis_size)?;
            if let Some(axis_index_groups) = &self.axis_index_groups {
                operation.field("axis_index_groups", format_args!("{axis_index_groups:?}"))?;
            }
            if self.is_physical() {
                operation.field("representation", "Physical")?;
            }
            if self.update_kind == RaggedAllToAllUpdateKind::Add {
                operation.field("update_kind", "Add")?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: RaggedAllToAllEvaluation>> InterpretableOperation<C>
    for RaggedAllToAllOperation
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 6, ProgramError);
        let [operand, output, input_offsets, send_sizes, output_offsets, receive_sizes] = inputs else {
            unreachable!();
        };

        let batched = self.is_physical();
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        self.infer_output_types(input_types.as_slice(), &[])?;
        if !batched && self.effective_axis_size()? != 1 {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot interpret `{RAGGED_ALL_TO_ALL_OPERATION_NAME}` over axis `{}` of size {} without an \
                     enclosing binder",
                    self.axis_name, self.axis_size,
                ),
            });
        }
        Ok(vec![C::Value::evaluate_ragged_all_to_all(
            self,
            operand,
            output,
            input_offsets,
            send_sizes,
            output_offsets,
            receive_sizes,
        )?])
    }
}

// This direct composite carrier has an array-only boundary, but it cannot be a second projected `ArrayType` member in
// `ArrayIrOperation`. Keep the projection explicit so its contract remains identical to the homogeneous operation.
impl MemberOperation<ArrayIrType> for RaggedAllToAllOperation {
    fn infer_parent_region_input_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
        infer_projected_operation_region_input_types(self, input_types, region_interfaces)
    }

    fn infer_parent_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        infer_projected_operation_output_types(self, input_types, region_interfaces)
    }

    fn rename_parent_type_identities(
        &self,
        _renaming: &TypeIdentityRenaming<DimensionVariable>,
    ) -> Result<Self, TypeError> {
        Ok(self.clone())
    }
}

impl<C> MemberInterpretableOperation<C> for RaggedAllToAllOperation
where
    C: Domain<
            Type = ArrayIrType,
            Value: ValueProjection<ArrayType, Projected: RaggedAllToAllEvaluation + Value<Type = ArrayType>>,
        >,
{
    fn interpret_in_parent<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        interpret_projected_operation(context, self, driver, inputs)
    }
}

// Partial evaluation uses the default fold-or-residualize behavior. Known metadata remain ordinary runtime values;
// the rule never assumes that a known primal operand is a compile-time literal.
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for RaggedAllToAllOperation where
    C::Operation: From<RaggedAllToAllOperation>
{
}

// The two data operands are jointly linear. Metadata remain primal values and therefore become ordinary residuals
// whenever the tangent exchange survives partial evaluation.
impl<C: Context<Type = ArrayType, Value: ZeroLike>> DifferentiableOperation<C> for RaggedAllToAllOperation
where
    C::Operation: From<RaggedAllToAllOperation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 6, ProgramError);
        let [operand, output, input_offsets, send_sizes, output_offsets, receive_sizes] = inputs else {
            unreachable!();
        };
        let primal_inputs = [
            operand.primal().clone(),
            output.primal().clone(),
            input_offsets.primal().clone(),
            send_sizes.primal().clone(),
            output_offsets.primal().clone(),
            receive_sizes.primal().clone(),
        ];
        let mut primal_outputs = context.bind(self.clone(), Vec::new(), &primal_inputs)?;
        check_count!("output", primal_outputs, 1, ProgramError);
        let primal = primal_outputs.remove(0);
        let tangent = if operand.tangent().is_zero() && output.tangent().is_zero() {
            MaybeZero::Zero(primal.r#type().tangent()?)
        } else {
            let operand_tangent = match operand.tangent() {
                MaybeZero::Zero(_) => operand.primal().zero_like()?,
                MaybeZero::Value(tangent) => tangent.clone(),
            };
            let output_tangent = match output.tangent() {
                MaybeZero::Zero(_) => output.primal().zero_like()?,
                MaybeZero::Value(tangent) => tangent.clone(),
            };
            let tangent_inputs = [
                operand_tangent,
                output_tangent,
                input_offsets.primal().clone(),
                send_sizes.primal().clone(),
                output_offsets.primal().clone(),
                receive_sizes.primal().clone(),
            ];
            let mut tangent_outputs = context.bind(self.clone(), Vec::new(), &tangent_inputs)?;
            check_count!("output", tangent_outputs, 1, ProgramError);
            MaybeZero::Value(tangent_outputs.remove(0))
        };
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

impl<C> MemberDifferentiableOperation<C> for RaggedAllToAllOperation
where
    C: Context<
            Type = ArrayIrType,
            Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: OperationProjection<ArrayType>,
        >,
    <C::Operation as OperationProjection<ArrayType>>::Projected:
        DifferentiableOperation<ProjectedContext<C, ArrayType>> + From<RaggedAllToAllOperation>,
{
    fn jvp_in_parent<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let operation = <C::Operation as OperationProjection<ArrayType>>::Projected::from(self.clone());
        jvp_projected_operation(context, &operation, inputs)
    }
}

/// Returns `input` as the known primal residual required by the transpose rule.
fn known_transpose_input<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType>>(
    input: &PartialValue<Tracer<TracingContext<V, O>>>,
    name: &str,
) -> Result<Tracer<TracingContext<V, O>>, DifferentiationError> {
    input.as_known().cloned().ok_or_else(|| {
        ProgramError::UnsupportedOperation {
            message: format!(
                "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` transpose requires `{name}` to be a known primal residual",
            ),
        }
        .into()
    })
}

/// Stages the logical named-axis exchange that transposes sender-owned offset metadata.
fn transpose_logical_offsets<V, O>(
    operation: &RaggedAllToAllOperation,
    context: &mut TracingContext<V, O>,
    offsets: &Tracer<TracingContext<V, O>>,
) -> Result<Tracer<TracingContext<V, O>>, DifferentiationError>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<AllToAllOperation>,
{
    let options = operation.axis_index_groups().map_or_else(CollectiveOptions::tiled, |groups| {
        CollectiveOptions::tiled().with_axis_index_groups(groups.to_vec())
    });
    let exchange = AllToAllOperation::new(operation.axis_name().to_string(), operation.axis_size(), 0, 0, options);
    let mut outputs = context.bind(exchange, Vec::new(), std::slice::from_ref(offsets))?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// Transposes physical sender/receiver offset blocks within each participant group using only static slices and
/// concatenations. Physical batching has already materialized every participant, so no named-axis binder remains in
/// which a dense collective could run.
fn transpose_physical_offsets<V, O>(
    operation: &RaggedAllToAllOperation,
    offsets: &Tracer<TracingContext<V, O>>,
) -> Result<Tracer<TracingContext<V, O>>, DifferentiationError>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<ConcatenateOperation<ArrayType>> + From<SliceOperation>,
{
    let offset_type = offsets.r#type();
    let metadata_length = offset_type.shape().dimensions()[1].value().unwrap();
    let group_size = operation.effective_axis_size()?;
    let slices_per_peer = metadata_length / group_size;
    let groups = operation
        .axis_index_groups()
        .map_or_else(|| vec![(0..operation.axis_size()).collect()], |groups| groups.to_vec());
    let mut rows = Vec::with_capacity(operation.axis_size());
    for participant in 0..operation.axis_size() {
        let (group, participant_position) = groups
            .iter()
            .find_map(|group| {
                group.iter().position(|candidate| *candidate == participant).map(|position| (group, position))
            })
            .unwrap();
        let start = participant_position * slices_per_peer;
        let mut blocks = Vec::with_capacity(group_size);
        for &sender in group {
            blocks.push(offsets.slice(&[sender, start], &[sender + 1, start + slices_per_peer], &[1, 1])?);
        }
        rows.push(Tracer::concatenate(blocks.iter(), 1)?);
    }
    Ok(Tracer::concatenate(rows.iter(), 0)?)
}

/// Transposes sender-owned offset metadata in the operation's current logical or physical representation.
fn transpose_offsets<V, O>(
    operation: &RaggedAllToAllOperation,
    context: &mut TracingContext<V, O>,
    offsets: &Tracer<TracingContext<V, O>>,
) -> Result<Tracer<TracingContext<V, O>>, DifferentiationError>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + From<AllToAllOperation>
        + From<ConcatenateOperation<ArrayType>>
        + From<SliceOperation>,
{
    if operation.is_physical() {
        transpose_physical_offsets(operation, offsets)
    } else {
        transpose_logical_offsets(operation, context, offsets)
    }
}

/// Stages the interval mask that preserves the output seed's cotangent outside received regions.
fn mask_output_cotangent<V, O>(
    context: &mut TracingContext<V, O>,
    cotangent: &Tracer<TracingContext<V, O>>,
    output_offsets: &Tracer<TracingContext<V, O>>,
    receive_sizes: &Tracer<TracingContext<V, O>>,
    physical: bool,
) -> Result<Tracer<TracingContext<V, O>>, DifferentiationError>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType>
        + From<AddOperation<ArrayType>>
        + From<BroadcastOperation>
        + From<ConvertElementTypeOperation<ArrayType>>
        + From<CompareOperation<ArrayType>>
        + From<CumulativeSumOperation>
        + From<NegOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + From<ReshapeOperation>
        + From<ScatterOperation>
        + From<SelectOperation<ArrayType>>
        + From<SliceOperation>
        + From<TransferToMemoryOperation>
        + From<ZeroOperation<ArrayType>>,
{
    let output_type = cotangent.r#type().into_owned();
    let leading_axis = usize::from(physical);
    let output_extent =
        output_type.shape().dimensions()[leading_axis]
            .value()
            .ok_or_else(|| ProgramError::UnsupportedOperation {
                message: format!(
                    "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` transpose requires a static output leading dimension",
                ),
            })?;
    let marker_extent = output_extent.checked_add(1).ok_or_else(|| ProgramError::InvalidArgument {
        message: format!("`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` transpose marker extent does not fit in `usize`"),
    })?;
    let mut marker_dimensions = output_type.shape().dimensions()[..=leading_axis].to_vec();
    marker_dimensions[leading_axis] = Dimension::Static(marker_extent);
    let marker_type = ArrayType::new(DataType::I64, Shape::new(marker_dimensions)).with_memory(output_type.memory());

    // Metadata may use any integer width and memory placement. Widen index arithmetic to `u64` before adding and
    // move it beside the cotangent so scatter's three operands share one memory space.
    let normalize_metadata = |value: &Tracer<TracingContext<V, O>>| -> Result<_, ProgramError> {
        let value = if value.r#type().memory() == output_type.memory() {
            value.clone()
        } else {
            value.transfer_to_memory(output_type.memory())
        };
        if value.r#type().data_type() == DataType::U64 {
            Ok(value)
        } else {
            Ok(value.convert_element_type(DataType::U64)?)
        }
    };
    let output_offsets = normalize_metadata(output_offsets)?;
    let receive_sizes = normalize_metadata(receive_sizes)?;
    let update_type = output_offsets.r#type().into_owned().with_data_type(DataType::I64);
    let marker = context.zero(&marker_type)?;
    let ones = context.one(&update_type)?;
    let negative_ones = ones.neg()?;
    let end_offsets = output_offsets.add(&receive_sizes)?;
    let mut index_dimensions = output_offsets.r#type().shape().dimensions().to_vec();
    index_dimensions.push(Dimension::Static(1));
    let start_indices = output_offsets.reshape(Shape::new(index_dimensions.clone()))?;
    let end_indices = end_offsets.reshape(Shape::new(index_dimensions))?;
    let scatter_dimensions = if physical {
        ScatterDimensionNumbers::new(Vec::new(), vec![1], vec![1]).with_batching_dimensions(vec![0], vec![0])
    } else {
        ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0])
    };
    // Both boundaries are additive. A zero-length region contributes `+1` and `-1` at the same position, while
    // adjacent regions combine deterministically at their shared boundary.
    let scatter = ScatterOperation::new(scatter_dimensions, ScatterReductionKind::Add);
    let markers = marker.scatter(&start_indices, &ones, &scatter)?.scatter(&end_indices, &negative_ones, &scatter)?;
    let markers = markers.cumulative_sum(leading_axis)?;
    let start_indices = vec![0; markers.r#type().rank()];
    let mut limit_indices = markers
        .r#type()
        .shape()
        .dimensions()
        .iter()
        .map(|dimension| dimension.value().unwrap())
        .collect::<Vec<_>>();
    limit_indices[leading_axis] = output_extent;
    let strides = vec![1; markers.r#type().rank()];
    let markers = markers.slice(start_indices.as_slice(), limit_indices.as_slice(), strides.as_slice())?;
    let marker_zero = context.zero(markers.r#type().as_ref())?;
    let received = markers.not_equal(&marker_zero)?;
    let condition_type =
        ArrayType::new(DataType::Boolean, output_type.shape().clone()).with_memory(output_type.memory());
    let received = received.broadcast(condition_type, &(0..=leading_axis).collect::<Vec<_>>())?;
    let zero = context.zero(&output_type)?;
    Ok(Tracer::select(&received, &zero, cotangent)?)
}

impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType>> TransposableOperation<V, O> for RaggedAllToAllOperation
where
    O: From<AddOperation<ArrayType>>
        + From<AllToAllOperation>
        + From<BroadcastOperation>
        + From<ConcatenateOperation<ArrayType>>
        + From<CompareOperation<ArrayType>>
        + From<ConvertElementTypeOperation<ArrayType>>
        + From<CumulativeSumOperation>
        + From<NegOperation<ArrayType>>
        + From<OneOperation<ArrayType>>
        + From<RaggedAllToAllOperation>
        + From<ReshapeOperation>
        + From<ScatterOperation>
        + From<SelectOperation<ArrayType>>
        + From<SliceOperation>
        + From<TransferToMemoryOperation>
        + From<ZeroOperation<ArrayType>>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<'_, V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        let provenance_context = context.clone();
        provenance_context.invoke_with_provenance_scope(ProvenanceScope::new("ryft"), || {
            provenance_context.invoke_with_provenance_scope(ProvenanceScope::new("differentiation"), || {
                provenance_context.invoke_with_provenance_scope(
                    ProvenanceScope::new("ragged_all_to_all_transpose"),
                    || {
                        check_count!("input", inputs, 6, ProgramError);
                        check_count!("output", outputs, 1, ProgramError);
                        let [operand, output, input_offsets, send_sizes, output_offsets, receive_sizes] = inputs else {
                            unreachable!()
                        };
                        let zero_inputs = || {
                            inputs
                                .iter()
                                .map(|input| Ok(MaybeZero::Zero(input.r#type().cotangent()?)))
                                .collect::<Result<Vec<_>, DifferentiationError>>()
                        };
                        let MaybeZero::Value(cotangent) = &outputs[0] else {
                            return zero_inputs();
                        };
                        if operand.is_known() && output.is_known() {
                            return zero_inputs();
                        }

                        let input_offsets = known_transpose_input(input_offsets, "input_offsets")?;
                        let send_sizes = known_transpose_input(send_sizes, "send_sizes")?;
                        let output_offsets = known_transpose_input(output_offsets, "output_offsets")?;
                        let receive_sizes = known_transpose_input(receive_sizes, "receive_sizes")?;
                        let (operand_cotangent, permuted_output_offsets) = if operand.is_known() {
                            (MaybeZero::Zero(operand.r#type().cotangent()?), None)
                        } else {
                            let permuted_output_offsets = transpose_offsets(self, context, &output_offsets)?;
                            let permuted_input_offsets = transpose_offsets(self, context, &input_offsets)?;
                            let zero = context.zero(&operand.r#type().cotangent()?)?;
                            let adjoint_inputs = [
                                cotangent.clone(),
                                zero,
                                permuted_output_offsets.clone(),
                                receive_sizes.clone(),
                                permuted_input_offsets,
                                send_sizes.clone(),
                            ];
                            let mut contributions =
                                context.bind(self.with_additive_updates(), Vec::new(), &adjoint_inputs)?;
                            check_count!("output", contributions, 1, ProgramError);
                            (MaybeZero::Value(contributions.remove(0)), Some(permuted_output_offsets))
                        };
                        let output_cotangent = if output.is_known() {
                            MaybeZero::Zero(output.r#type().cotangent()?)
                        } else if self.update_kind == RaggedAllToAllUpdateKind::Add {
                            MaybeZero::Value(cotangent.clone())
                        } else {
                            let permuted_output_offsets = match permuted_output_offsets {
                                Some(permuted_output_offsets) => permuted_output_offsets,
                                None => transpose_offsets(self, context, &output_offsets)?,
                            };
                            MaybeZero::Value(mask_output_cotangent(
                                context,
                                cotangent,
                                &permuted_output_offsets,
                                &receive_sizes,
                                self.is_physical(),
                            )?)
                        };
                        Ok(vec![
                            operand_cotangent,
                            output_cotangent,
                            MaybeZero::Zero(input_offsets.r#type().cotangent()?),
                            MaybeZero::Zero(send_sizes.r#type().cotangent()?),
                            MaybeZero::Zero(output_offsets.r#type().cotangent()?),
                            MaybeZero::Zero(receive_sizes.r#type().cotangent()?),
                        ])
                    },
                )
            })
        })
    }
}

/// Constructs a `u64` scalar array containing `extent` in the memory space of `metadata_type`.
fn metadata_extent_scalar(metadata_type: &ArrayType, extent: usize) -> Result<Array, ProgramError> {
    let extent = u64::try_from(extent).map_err(|_| ProgramError::InvalidArgument {
        message: format!("`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` extent {extent} does not fit in `u64`"),
    })?;
    Array::from_elements(ArrayType::scalar(DataType::U64).with_memory(metadata_type.memory()), &[extent])
}

// A matching named batch axis is the eager reference implementation's participant axis. All operands are aligned to
// physical axis zero before one parent bind executes the complete exchange. Unresolved non-constant metadata are gated
// because no existing slicing primitive carries a dynamic segment length. A non-matching mapped axis merges its batch
// into the packed leading data and metadata axes with sender/receiver offsets rebased by the mapped item index. An
// all-replicated application can be forwarded unchanged.
impl<C, P: CollectiveBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>> for RaggedAllToAllOperation
where
    C: Context<Type = ArrayType>,
    C::Operation: From<ConstantOperation<Array>>
        + From<ConvertElementTypeOperation<ArrayType>>
        + From<IotaOperation<ArrayType>>
        + From<RaggedAllToAllOperation>,
    C::Value: Broadcast + Transpose,
    AddOperation<ArrayType>: BatchableOperation<C, ArrayBatching<P>>,
    MulOperation<ArrayType>: BatchableOperation<C, ArrayBatching<P>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        reject_ragged_collective_inputs(self.name(), inputs)?;
        check_count!("input", inputs, 6, ProgramError);

        if context.axis_name() != Some(self.axis_name()) {
            if inputs.iter().all(|input| input.batch_axis().is_replicated()) {
                let parent_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
                let mut outputs = context.parent().bind(self.clone(), Vec::new(), parent_inputs.as_slice())?;
                check_count!("output", outputs, 1, ProgramError);
                return Ok(vec![ArrayBatch::replicated(outputs.remove(0))].into());
            }
            if self.axis_index_groups().is_some() {
                return Err(BatchingError::UnsupportedOperation {
                    message: "`ragged_all_to_all` axis index groups are not supported when merging an unrelated \
                              mapped axis"
                        .to_string(),
                });
            }

            let provenance_context = context.parent().clone();
            return provenance_context.invoke_with_provenance_scope(ProvenanceScope::new("ryft"), || {
                provenance_context.invoke_with_provenance_scope(ProvenanceScope::new("batching"), || {
                    provenance_context.invoke_with_provenance_scope(ProvenanceScope::new("ragged_all_to_all"), || {
                        let participant_axis_count = usize::from(self.is_physical());
                        let input_leading_axis = participant_axis_count;
                        let metadata_batch_axis = participant_axis_count + 1;
                        let output_batch_axis = participant_axis_count;
                        let logical_input_types = inputs.iter().map(ArrayBatch::unbatched_type).collect::<Vec<_>>();
                        self.infer_output_types(logical_input_types.as_slice(), &[])?;
                        let batch_size =
                            P::axis_dimension(context)?.value().ok_or_else(|| BatchingError::UnsupportedOperation {
                                message: "`ragged_all_to_all` merged batching requires a statically known \
                                          mapped-axis extent"
                                    .to_string(),
                            })?;
                        if batch_size == 0 {
                            let output = P::match_axis(context, &inputs[1], output_batch_axis.into())?;
                            return Ok(vec![output].into());
                        }
                        let static_extent = |r#type: &ArrayType, axis: usize, name: &str| {
                            r#type.shape().dimensions()[axis].value().ok_or_else(|| {
                                BatchingError::UnsupportedOperation {
                                    message: format!(
                                        "`{RAGGED_ALL_TO_ALL_OPERATION_NAME}` merged batching requires `{name}` \
                                         axis {axis} to have a static extent",
                                    ),
                                }
                            })
                        };
                        let input_extent = static_extent(&logical_input_types[0], input_leading_axis, "operand")?;
                        let output_extent = static_extent(&logical_input_types[1], input_leading_axis, "output")?;
                        let metadata_length =
                            logical_input_types[2].shape().dimensions()[participant_axis_count].value().unwrap();
                        let batch_extent = P::collective_extent_constant(context, batch_size)?;
                        let input_extent_value = P::collective_extent_constant(context, input_extent)?;
                        let output_extent_value = P::collective_extent_constant(context, output_extent)?;
                        let metadata_length_value = P::collective_extent_constant(context, metadata_length)?;
                        let participant_extent = self
                            .is_physical()
                            .then(|| P::collective_extent_constant(context, self.axis_size()))
                            .transpose()?;
                        let trailing_extents = (input_leading_axis + 1..logical_input_types[0].rank())
                            .map(|axis| {
                                static_extent(&logical_input_types[0], axis, "operand")
                                    .and_then(|extent| P::collective_extent_constant(context, extent))
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        let aligned_operand = P::match_axis(context, &inputs[0], output_batch_axis.into())?;
                        let aligned_output = P::match_axis(context, &inputs[1], output_batch_axis.into())?;
                        let restored_output_sharding = aligned_output.r#type().sharding().cloned();
                        let mut operand_extents = Vec::new();
                        let mut output_extents = Vec::new();
                        if let Some(participant_extent) = &participant_extent {
                            operand_extents.push(participant_extent.clone());
                            output_extents.push(participant_extent.clone());
                        }
                        operand_extents.push(batch_extent.mul(&input_extent_value)?);
                        output_extents.push(batch_extent.mul(&output_extent_value)?);
                        operand_extents.extend(trailing_extents.iter().cloned());
                        output_extents.extend(trailing_extents.iter().cloned());
                        let operand = P::reshape_collective(
                            context,
                            aligned_operand.into_value(),
                            operand_extents.as_slice(),
                            None,
                        )?;
                        let output = P::reshape_collective(
                            context,
                            aligned_output.into_value(),
                            output_extents.as_slice(),
                            None,
                        )?;

                        let mut metadata = inputs[2..]
                            .iter()
                            .map(|input| {
                                let metadata = P::match_axis(context, input, metadata_batch_axis.into())?;
                                if metadata.r#type().data_type() == DataType::U64 {
                                    return Ok(metadata);
                                }
                                let batch_axis = metadata.batch_axis();
                                let value = context
                                    .parent()
                                    .bind(
                                        ConvertElementTypeOperation::new(DataType::U64),
                                        Vec::new(),
                                        std::slice::from_ref(metadata.value()),
                                    )?
                                    .remove(0);
                                ArrayBatch::new(value, batch_axis)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        for (metadata_index, leading_extent) in [(0, input_extent), (2, output_extent)] {
                            let iota_type = metadata[metadata_index].r#type().into_owned();
                            let iota = context
                                .parent()
                                .bind(IotaOperation::new(iota_type.clone(), metadata_batch_axis)?, Vec::new(), &[])?
                                .remove(0);
                            let iota = ArrayBatch::new(iota, BatchAxis::from_position(metadata_batch_axis))?;
                            let mut scale = context.parent().bind(
                                ConstantOperation::new(metadata_extent_scalar(&iota_type, leading_extent)?),
                                Vec::new(),
                                &[],
                            )?;
                            check_count!("output", scale, 1, ProgramError);
                            let scale = scale.remove(0);
                            let scale = ArrayBatch::replicated(scale);
                            let (mut rebasing, _) =
                                MulOperation::new().batch(context, &EmptyRegionDriver, &[iota, scale])?.into_parts();
                            check_count!("output", rebasing, 1, ProgramError);
                            let (mut rebased, _) = AddOperation::new()
                                .batch(
                                    context,
                                    &EmptyRegionDriver,
                                    &[metadata[metadata_index].clone(), rebasing.remove(0)],
                                )?
                                .into_parts();
                            check_count!("output", rebased, 1, ProgramError);
                            metadata[metadata_index] = rebased.remove(0);
                        }
                        let mut metadata_extents = Vec::new();
                        if let Some(participant_extent) = &participant_extent {
                            metadata_extents.push(participant_extent.clone());
                        }
                        metadata_extents.push(metadata_length_value.mul(&batch_extent)?);
                        let metadata = metadata
                            .into_iter()
                            .map(|metadata| {
                                P::reshape_collective(context, metadata.into_value(), metadata_extents.as_slice(), None)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let merged_inputs =
                            std::iter::once(operand).chain(std::iter::once(output)).chain(metadata).collect::<Vec<_>>();
                        let mut outputs = context.parent().bind(self.clone(), Vec::new(), merged_inputs.as_slice())?;
                        check_count!("output", outputs, 1, ProgramError);
                        let mut restored_extents = Vec::new();
                        if let Some(participant_extent) = participant_extent {
                            restored_extents.push(participant_extent);
                        }
                        restored_extents.push(batch_extent);
                        restored_extents.push(output_extent_value);
                        restored_extents.extend(trailing_extents);
                        let output = P::reshape_collective(
                            context,
                            outputs.remove(0),
                            restored_extents.as_slice(),
                            restored_output_sharding,
                        )?;
                        Ok(vec![ArrayBatch::new(output, BatchAxis::from_position(output_batch_axis))?].into())
                    })
                })
            });
        }

        P::collective_axis_extent(context, self.name(), self.axis_name(), self.axis_size)?;
        let inputs =
            inputs.iter().map(|input| P::match_axis(context, input, 0.into())).collect::<Result<Vec<_>, _>>()?;
        if inputs[2..]
            .iter()
            .any(|input| !matches!(context.parent().resolve(input.value()), ValueResolution::Constant(_)))
        {
            return Err(BatchingError::UnsupportedOperation {
                message: "`ragged_all_to_all` cannot materialize a batch-bound collective with staged metadata"
                    .to_string(),
            });
        }
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let mut outputs =
            context.parent().bind(self.with_physical_representation(), Vec::new(), physical_inputs.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(vec![ArrayBatch::new(outputs.remove(0), BatchAxis::from_position(0))?].into())
    }
}

impl<C> MemberBatchableOperation<C, ArrayIrBatching> for RaggedAllToAllOperation
where
    C: Context<
            Type = ArrayIrType,
            Value: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: OperationProjection<ArrayType>,
        >,
    ProjectedContext<C, ArrayType>: Context<
            Type = ArrayType,
            Value = <C::Value as ValueProjection<ArrayType>>::Projected,
            Constant = <C::Constant as ValueProjection<ArrayType>>::Projected,
            Operation = <C::Operation as OperationProjection<ArrayType>>::Projected,
        >,
    RaggedAllToAllOperation:
        BatchableOperation<ProjectedContext<C, ArrayType>, ArrayBatching<DynamicArrayBatchingPolicy>>,
{
    fn batch_in_parent<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatching>, BatchingError> {
        batch_projected_operation(context, self, inputs)
    }
}

/// Stages an explicitly packed ragged all-to-all in any named-axis array operation domain that carries
/// [`RaggedAllToAllOperation`].
pub trait RaggedAllToAll: Sized {
    /// Exchanges segments over the full named axis.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of the mapped or mesh axis whose participants exchange segments.
    ///   - `output`: Seed value updated at the received regions and returned with the same type.
    ///   - `input_offsets`: Sender-local leading-axis offsets of the segments in `self`.
    ///   - `send_sizes`: Sender-local leading-axis lengths of the segments in `self`.
    ///   - `output_offsets`: Sender-owned offsets expressed in each corresponding receiver's output coordinate frame.
    ///   - `receive_sizes`: Receiver-local leading-axis lengths, indexed by sending participant.
    fn ragged_all_to_all(
        &self,
        axis_name: &str,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
    ) -> Result<Self, ProgramError>;

    /// Exchanges segments within the provided ordered participant groups.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of the mapped or mesh axis whose participants exchange segments.
    ///   - `output`: Seed value updated at the received regions and returned with the same type.
    ///   - `input_offsets`: Sender-local leading-axis offsets of the segments in `self`.
    ///   - `send_sizes`: Sender-local leading-axis lengths of the segments in `self`.
    ///   - `output_offsets`: Sender-owned offsets expressed in each corresponding receiver's output coordinate frame.
    ///   - `receive_sizes`: Receiver-local leading-axis lengths, indexed by sending participant.
    ///   - `axis_index_groups`: Ordered equal-sized partition of the full axis indices; exchange stays within groups.
    fn ragged_all_to_all_with_axis_index_groups(
        &self,
        axis_name: &str,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError>;
}

impl<V> RaggedAllToAll for V
where
    V: Value,
    V::DispatchDomain: Context + NamedAxes,
    <V::DispatchDomain as Domain>::Operation: From<RaggedAllToAllOperation>,
{
    fn ragged_all_to_all(
        &self,
        axis_name: &str,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let inputs = [self, output, input_offsets, send_sizes, output_offsets, receive_sizes].map(Clone::clone);
        let mut outputs =
            context.bind(RaggedAllToAllOperation::new(axis_name.to_string(), axis_size), Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    fn ragged_all_to_all_with_axis_index_groups(
        &self,
        axis_name: &str,
        output: &Self,
        input_offsets: &Self,
        send_sizes: &Self,
        output_offsets: &Self,
        receive_sizes: &Self,
        axis_index_groups: Vec<Vec<usize>>,
    ) -> Result<Self, ProgramError> {
        let context = self.dispatch_domain();
        let axis_size = resolve_named_axis_size(&context, axis_name)?;
        let operation = RaggedAllToAllOperation::grouped(axis_name.to_string(), axis_size, axis_index_groups)?;
        let inputs = [self, output, input_offsets, send_sizes, output_offsets, receive_sizes].map(Clone::clone);
        let mut outputs = context.bind(operation, Vec::new(), &inputs)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionVariable, LogicalMesh, MeshAxis, MeshAxisType, RaggedAxis, Shape,
        Sharding, ShardingDimension,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingTracer, batch};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::{check_gradient, check_operation_transposition, check_operation_type_inference};
    use crate::operations::math::reduce::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationValue, PartialTracer};
    use crate::programs::{ProgramBuilder, Provenance};

    use super::*;

    // Returns a static array type with the provided element type and dimensions.
    fn array_type(data_type: DataType, dimensions: impl IntoIterator<Item = usize>) -> ArrayType {
        ArrayType::new(data_type, Shape::new(dimensions.into_iter().map(Dimension::Static).collect()))
    }

    #[test]
    fn test_ragged_all_to_all_type_inference() {
        let data = || array_type(DataType::F32, [3, 2]);
        let output = || array_type(DataType::F32, [4, 2]);
        let metadata = || array_type(DataType::I32, [2]);
        let metadata_length = DimensionVariable::new("metadata_length", DimensionBounds::positive(Some(4)).unwrap());
        let dynamic_metadata = ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(metadata_length)]));
        check_operation_type_inference!(
            operation = RaggedAllToAllOperation::new("x".to_string(), 2),
            cases = [
                {
                    input_types = [data(), output(), metadata(), metadata(), metadata(), metadata()],
                    output_types = [output()],
                },
                {
                    input_types = [
                        ArrayType::scalar(DataType::F32),
                        output(),
                        metadata(),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`ragged_all_to_all` data operands must have rank at least 1 but got `f32[]` and \
                             `f32[4, 2]`",
                },
                {
                    input_types = [
                        data(),
                        array_type(DataType::F64, [4, 2]),
                        metadata(),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`ragged_all_to_all` operand and output data types must match but got `f32` and `f64`",
                },
                {
                    input_types = [
                        data(),
                        array_type(DataType::F32, [4, 3]),
                        metadata(),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`ragged_all_to_all` operand and output trailing dimensions must match but got `[3, 2]` \
                             and `[4, 3]`",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        array_type(DataType::I32, [2, 1]),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`ragged_all_to_all` `input_offsets` must be rank 1 but got `i32[2, 1]`",
                },
                {
                    input_types = [
                        array_type(DataType::F32, [2, 3]),
                        array_type(DataType::F32, [4, 3]),
                        array_type(DataType::I32, [2, 2]),
                        array_type(DataType::I32, [2, 2]),
                        array_type(DataType::I32, [2, 2]),
                        array_type(DataType::I32, [2, 2]),
                    ],
                    error = "`ragged_all_to_all` `input_offsets` must be rank 1 but got `i32[2, 2]`",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        array_type(DataType::F32, [2]),
                        metadata(),
                        metadata(),
                        metadata(),
                    ],
                    error = "`ragged_all_to_all` `input_offsets` must have an integer data type but got `f32`",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        metadata(),
                        array_type(DataType::I64, [2]),
                        metadata(),
                        metadata(),
                    ],
                    error = "`ragged_all_to_all` metadata operands must share one integer data type but \
                             `input_offsets` has `i32` and `send_sizes` has `i64`",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        metadata(),
                        array_type(DataType::I32, [4]),
                        metadata(),
                        metadata(),
                    ],
                    error = "`ragged_all_to_all` metadata operands must have equal lengths but `input_offsets` has \
                             length 2 and `send_sizes` has length 4",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        dynamic_metadata.clone(),
                        dynamic_metadata.clone(),
                        dynamic_metadata.clone(),
                        dynamic_metadata.clone(),
                    ],
                    error = format!(
                        "`ragged_all_to_all` `input_offsets` must have a static length but got `{dynamic_metadata}`",
                    ),
                },
                {
                    input_types = [
                        data(),
                        output(),
                        array_type(DataType::I32, [0]),
                        array_type(DataType::I32, [0]),
                        array_type(DataType::I32, [0]),
                        array_type(DataType::I32, [0]),
                    ],
                    error = "`ragged_all_to_all` metadata length must be greater than zero",
                },
                {
                    input_types = [
                        data(),
                        output(),
                        array_type(DataType::I32, [3]),
                        array_type(DataType::I32, [3]),
                        array_type(DataType::I32, [3]),
                        array_type(DataType::I32, [3]),
                    ],
                    error = "`ragged_all_to_all` metadata length 3 is not divisible by group size 2",
                },
            ],
        );
        assert_eq!(
            RaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1], vec![2, 2]])
                .unwrap_err()
                .to_string(),
            "`ragged_all_to_all` axis index groups contain participant 2 more than once",
        );
        assert_eq!(
            RaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1], vec![2]])
                .unwrap_err()
                .to_string(),
            "`ragged_all_to_all` axis index group 1 has size 1 but every group must have size 2",
        );
        assert_eq!(
            RaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1], vec![2, 4]])
                .unwrap_err()
                .to_string(),
            "`ragged_all_to_all` axis index 4 is out of bounds for axis size 4",
        );
        assert_eq!(
            RaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 1]]).unwrap_err().to_string(),
            "`ragged_all_to_all` axis index groups do not contain participant 2",
        );
    }

    #[test]
    fn test_ragged_all_to_all_staging_contracts() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let input_types = vec![
            ArrayIrType::Array(array_type(DataType::F32, [3])),
            ArrayIrType::Array(array_type(DataType::F32, [4])),
            ArrayIrType::Array(array_type(DataType::I32, [2])),
            ArrayIrType::Array(array_type(DataType::I32, [2])),
            ArrayIrType::Array(array_type(DataType::I32, [2])),
            ArrayIrType::Array(array_type(DataType::I32, [2])),
        ];
        let unbound = TestContext::trace(
            |inputs: Vec<_>| {
                inputs[0].ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
            },
            input_types.clone(),
        )
        .unwrap_err();
        assert_eq!(unbound.to_string(), "axis name `x` is not bound by any enclosing transform");

        let (_, program) = TestContext::trace_with_named_axes(
            |inputs: Vec<_>| {
                inputs[0].ragged_all_to_all("x", &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5])
            },
            input_types,
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2 })],
        )
        .unwrap();
        assert_eq!(program.instructions().len(), 1);
        let ArrayIrOperation::RaggedAllToAll(operation) = program.instructions()[0].operation() else {
            panic!("ragged all-to-all must use its direct composite carrier");
        };
        assert_eq!(operation.to_string(), "ragged_all_to_all [axis_name=\"x\", axis_size=2]");
        assert_eq!(program.instructions()[0].inputs().len(), 6);

        type HomogeneousTestContext = TracingContext<Array, ArrayOperation<Array>>;
        let physical_input_types = vec![
            array_type(DataType::F32, [2, 3]),
            array_type(DataType::F32, [2, 4]),
            array_type(DataType::I32, [2, 2]),
            array_type(DataType::I32, [2, 2]),
            array_type(DataType::I32, [2, 2]),
            array_type(DataType::I32, [2, 2]),
        ];
        let error = HomogeneousTestContext::trace(
            |inputs: Vec<_>| {
                let context = BatchingContext::new(inputs[0].context().clone(), 2).with_axis_name("x".to_string());
                let inputs = inputs
                    .into_iter()
                    .map(|input| ArrayBatch::new(input, BatchAxis::new(0)))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = RaggedAllToAllOperation::new("x".to_string(), 2)
                    .batch(&context, &EmptyRegionDriver, inputs.as_slice())?
                    .into_parts()
                    .0;
                Ok(outputs[0].value().clone())
            },
            physical_input_types,
        )
        .unwrap_err();
        let error = error.downcast_custom::<BatchingError>().unwrap();
        assert_eq!(
            error,
            &BatchingError::UnsupportedOperation {
                message: "`ragged_all_to_all` cannot materialize a batch-bound collective with staged metadata"
                    .to_string(),
            },
        );

        let partial_context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let partial_inputs = [
            array_type(DataType::F32, [2, 3]),
            array_type(DataType::F32, [2, 4]),
            array_type(DataType::I32, [2, 2]),
            array_type(DataType::I32, [2, 2]),
            array_type(DataType::I32, [2, 2]),
            array_type(DataType::I32, [2, 2]),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, r#type)| {
            let value = partial_context.unknown_input(r#type, index);
            ArrayBatch::new(PartialTracer::new(partial_context.clone(), value), BatchAxis::new(0)).unwrap()
        })
        .collect::<Vec<_>>();
        let batching_context = BatchingContext::new(partial_context, 2).with_axis_name("x".to_string());
        assert_eq!(
            RaggedAllToAllOperation::new("x".to_string(), 2).batch(
                &batching_context,
                &EmptyRegionDriver,
                partial_inputs.as_slice(),
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`ragged_all_to_all` cannot materialize a batch-bound collective with staged metadata"
                    .to_string(),
            }),
        );

        let partial_context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let operand = PartialTracer::new(
            partial_context.clone(),
            partial_context.unknown_input(array_type(DataType::F32, [2, 3]), 0),
        );
        let output = PartialTracer::new(
            partial_context.clone(),
            partial_context.unknown_input(array_type(DataType::F32, [2, 4]), 1),
        );
        let metadata = || {
            PartialTracer::new(
                partial_context.clone(),
                PartialEvaluationValue::known_constant(Array::matrix(2, 2, vec![0_i32; 4])),
            )
        };
        let inputs = [operand, output, metadata(), metadata(), metadata(), metadata()]
            .into_iter()
            .map(|input| ArrayBatch::new(input, BatchAxis::new(0)).unwrap())
            .collect::<Vec<_>>();
        let batching_context = BatchingContext::new(partial_context, 2).with_axis_name("x".to_string());
        let outputs = RaggedAllToAllOperation::new("x".to_string(), 2)
            .batch(&batching_context, &EmptyRegionDriver, inputs.as_slice())
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
    }

    #[test]
    fn test_ragged_all_to_all_batching_rejects_ragged_operands() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let ragged_operand = ArrayBatch::new(Array::matrix(2, 3, vec![1.0_f32; 6]), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]), variable, vec![0])])
            .unwrap();
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("x".to_string());
        let metadata = || ArrayBatch::new(Array::matrix(2, 2, vec![0_i32; 4]), BatchAxis::new(0)).unwrap();
        assert_eq!(
            RaggedAllToAllOperation::new("x".to_string(), 2).batch(
                &context,
                &EmptyRegionDriver,
                &[
                    ragged_operand,
                    ArrayBatch::new(Array::matrix(2, 4, vec![0.0_f32; 8]), BatchAxis::new(0)).unwrap(),
                    metadata(),
                    metadata(),
                    metadata(),
                    metadata(),
                ],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "`ragged_all_to_all` does not support bounded ragged dimension `length` on operand 0"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_ragged_all_to_all_jvp_handles_joint_and_structural_zero_tangents() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let operation = RaggedAllToAllOperation::new("x".to_string(), 1);
        let operand = Array::vector(vec![10.0_f64, 11.0, 12.0]);
        let output = Array::vector(vec![100.0_f64, 101.0, 102.0, 103.0]);
        let metadata = [
            Array::vector(vec![1_i32]),
            Array::vector(vec![2_i32]),
            Array::vector(vec![0_i32]),
            Array::vector(vec![2_i32]),
        ];
        let operand_tangent = Array::vector(vec![1.0_f64, 2.0, 3.0]);
        let output_tangent = Array::vector(vec![10.0_f64, 20.0, 30.0, 40.0]);
        let structural_zero = |primal: &Array| MaybeZero::Zero(primal.r#type().tangent().unwrap());
        let duals = |operand_tangent: MaybeZero<Array>, output_tangent: MaybeZero<Array>| {
            let mut inputs = vec![
                DifferentiationDual::new(operand.clone(), operand_tangent).unwrap(),
                DifferentiationDual::new(output.clone(), output_tangent).unwrap(),
            ];
            inputs.extend(metadata.iter().cloned().map(|primal| {
                let tangent = structural_zero(&primal);
                DifferentiationDual::new(primal, tangent).unwrap()
            }));
            inputs
        };

        let evaluate = |inputs: Vec<DifferentiationDual<Array>>| {
            operation.jvp(&context, &EmptyRegionDriver, inputs.as_slice()).unwrap().remove(0)
        };
        let zero = evaluate(duals(structural_zero(&operand), structural_zero(&output)));
        assert_eq!(zero.primal().to_f64s(), vec![11.0, 12.0, 102.0, 103.0]);
        assert!(zero.tangent().is_zero());

        let operand_only = evaluate(duals(MaybeZero::Value(operand_tangent.clone()), structural_zero(&output)));
        assert_eq!(operand_only.tangent().as_value().unwrap().to_f64s(), vec![2.0, 3.0, 0.0, 0.0]);

        let output_only = evaluate(duals(structural_zero(&operand), MaybeZero::Value(output_tangent.clone())));
        assert_eq!(output_only.tangent().as_value().unwrap().to_f64s(), vec![0.0, 0.0, 30.0, 40.0]);

        let joint = evaluate(duals(MaybeZero::Value(operand_tangent), MaybeZero::Value(output_tangent)));
        assert_eq!(joint.tangent().as_value().unwrap().to_f64s(), vec![2.0, 3.0, 30.0, 40.0]);
    }

    #[test]
    fn test_ragged_all_to_all_value_and_gradient_through_named_batch_axis() {
        let operand = Array::matrix(2, 3, vec![10.0_f64, 11.0, 12.0, 20.0, 21.0, 22.0]);
        let output = Array::matrix(2, 4, vec![100.0_f64, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0]);
        let input_offsets = Array::matrix(2, 2, vec![0_i32, 0, 0, 2]);
        let send_sizes = Array::matrix(2, 2, vec![0_i32, 1, 1, 0]);
        let output_offsets = Array::matrix(2, 2, vec![1_i32, 1, 2, 3]);
        let receive_sizes = Array::matrix(2, 2, vec![0_i32, 1, 1, 0]);

        let (value, gradient) = differentiate_at((operand, output))
            .value_and_gradient(move |(operand, output)| {
                let context = operand.context().clone();
                let input_offsets = context.lift(input_offsets.clone())?;
                let send_sizes = context.lift(send_sizes.clone())?;
                let output_offsets = context.lift(output_offsets.clone())?;
                let receive_sizes = context.lift(receive_sizes.clone())?;
                let exchanged = batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.ragged_all_to_all(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                        )
                    },
                    (operand, output, input_offsets, send_sizes, output_offsets, receive_sizes),
                    (
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                    ),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                Ok(exchanged.reduce(&[0, 1], ReductionKind::Sum))
            })
            .unwrap();

        assert_eq!(value.to_f64s(), vec![939.0]);
        assert_eq!(gradient.0.to_f64s(), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gradient.1.to_f64s(), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);

        check_gradient!(
            |operand, output| {
                let context = operand.dispatch_domain();
                let input_offsets = context.lift(Array::matrix(2, 2, vec![0_i32, 0, 0, 2]))?;
                let send_sizes = context.lift(Array::matrix(2, 2, vec![0_i32, 1, 1, 0]))?;
                let output_offsets = context.lift(Array::matrix(2, 2, vec![1_i32, 1, 2, 3]))?;
                let receive_sizes = context.lift(Array::matrix(2, 2, vec![0_i32, 1, 1, 0]))?;
                let exchanged = batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.ragged_all_to_all(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                        )
                    },
                    (operand, output, input_offsets, send_sizes, output_offsets, receive_sizes),
                    (
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                    ),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                Ok(exchanged.reduce(&[0, 1], ReductionKind::Sum))
            },
            at = Array::matrix(2, 3, vec![10.0_f64, 11.0, 12.0, 20.0, 21.0, 22.0]),
            with = Array::matrix(2, 4, vec![100.0_f64, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0]),
            step = 1e-6,
            tolerance = 1e-6,
        );
        check_gradient!(
            |output, operand| {
                let context = output.dispatch_domain();
                let input_offsets = context.lift(Array::matrix(2, 2, vec![0_i32, 0, 0, 2]))?;
                let send_sizes = context.lift(Array::matrix(2, 2, vec![0_i32, 1, 1, 0]))?;
                let output_offsets = context.lift(Array::matrix(2, 2, vec![1_i32, 1, 2, 3]))?;
                let receive_sizes = context.lift(Array::matrix(2, 2, vec![0_i32, 1, 1, 0]))?;
                let exchanged = batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.ragged_all_to_all(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                        )
                    },
                    (operand, output, input_offsets, send_sizes, output_offsets, receive_sizes),
                    (
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                    ),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?;
                Ok(exchanged.reduce(&[0, 1], ReductionKind::Sum))
            },
            at = Array::matrix(2, 4, vec![100.0_f64, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0],),
            with = Array::matrix(2, 3, vec![10.0_f64, 11.0, 12.0, 20.0, 21.0, 22.0]),
            step = 1e-6,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_named_batch_axis_composes_outside_ragged_all_to_all_differentiation() {
        let (values, gradients) = batch(
            |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                differentiate_at((operand, output))
                    .with_captures((input_offsets, send_sizes, output_offsets, receive_sizes))
                    .value_and_gradient(
                        |(operand, output), (input_offsets, send_sizes, output_offsets, receive_sizes)| {
                            Ok(operand
                                .ragged_all_to_all(
                                    "x",
                                    &output,
                                    &input_offsets,
                                    &send_sizes,
                                    &output_offsets,
                                    &receive_sizes,
                                )?
                                .reduce(&[0], ReductionKind::Sum))
                        },
                    )
                    .map_err(ProgramError::from)
            },
            (
                Array::matrix(2, 3, vec![10.0_f64, 11.0, 12.0, 20.0, 21.0, 22.0]),
                Array::matrix(2, 4, vec![100.0_f64, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0]),
                Array::matrix(2, 2, vec![0_i32, 0, 0, 2]),
                Array::matrix(2, 2, vec![0_i32, 1, 1, 0]),
                Array::matrix(2, 2, vec![1_i32, 1, 2, 3]),
                Array::matrix(2, 2, vec![0_i32, 1, 1, 0]),
            ),
            (
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxis::new(0),
            ),
            (BatchAxis::new(0), (BatchAxis::new(0), BatchAxis::new(0))),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();

        assert_eq!(values.to_f64s(), vec![324.0, 615.0]);
        assert_eq!(gradients.0.to_f64s(), vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(gradients.1.to_f64s(), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_grouped_physical_ragged_all_to_all_transpose_routes_cotangents() {
        let operation = RaggedAllToAllOperation::grouped("x".to_string(), 4, vec![vec![0, 2], vec![3, 1]])
            .unwrap()
            .with_physical_representation();
        let data_type = ArrayType::new_static(DataType::F64, [4, 2]);
        let output_type = ArrayType::new_static(DataType::F64, [4, 2]);
        check_operation_transposition!(
            @exact,
            operation = operation,
            cases = [{
                inputs = [
                    (@linear(type = data_type)),
                    (@linear(type = output_type)),
                    (@known, Array::matrix(4, 2, vec![0_i32, 1, 0, 1, 0, 1, 0, 1])),
                    (@known, Array::matrix(4, 2, vec![1_i32; 8])),
                    (@known, Array::matrix(4, 2, vec![0_i32, 0, 1, 1, 1, 1, 0, 0])),
                    (@known, Array::matrix(4, 2, vec![1_i32; 8])),
                ],
                output_cotangents = [Array::matrix(4, 2, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])],
                input_cotangents = [
                    Array::matrix(4, 2, vec![1.0_f64, 5.0, 8.0, 4.0, 2.0, 6.0, 7.0, 3.0]),
                    Array::matrix(4, 2, vec![0.0_f64; 8]),
                ],
            }],
        );
    }

    #[test]
    fn test_ragged_all_to_all_transpose_accumulates_resent_source_cotangents() {
        check_operation_transposition!(
            @exact,
            operation = RaggedAllToAllOperation::new("x".to_string(), 1),
            cases = [
                {
                    inputs = [
                        (@linear(type = ArrayType::new_static(DataType::F64, [2]))),
                        (@linear(type = ArrayType::new_static(DataType::F64, [2]))),
                        (@known, Array::vector(vec![0_i32, 0])),
                        (@known, Array::vector(vec![1_i32, 1])),
                        (@known, Array::vector(vec![0_i32, 1])),
                        (@known, Array::vector(vec![1_i32, 1])),
                    ],
                    output_cotangents = [Array::vector(vec![3.0_f64, 5.0])],
                    input_cotangents = [
                        Array::vector(vec![8.0_f64, 0.0]),
                        Array::vector(vec![0.0_f64, 0.0]),
                    ],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new_static(DataType::F64, [3]))),
                        (@linear(type = ArrayType::new_static(DataType::F64, [3]))),
                        (@known, Array::vector(vec![0_i32, 1, 1])),
                        (@known, Array::vector(vec![1_i32, 0, 1])),
                        (@known, Array::vector(vec![0_i32, 1, 1])),
                        (@known, Array::vector(vec![1_i32, 0, 1])),
                    ],
                    output_cotangents = [Array::vector(vec![3.0_f64, 5.0, 7.0])],
                    input_cotangents = [
                        Array::vector(vec![3.0_f64, 5.0, 0.0]),
                        Array::vector(vec![0.0_f64, 0.0, 7.0]),
                    ],
                },
            ],
        );
    }

    #[test]
    fn test_ragged_all_to_all_merges_an_unrelated_mapped_axis() {
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2)
            .with_axis_name("y".to_string());
        let operation = RaggedAllToAllOperation::new("x".to_string(), 2).with_physical_representation();
        let operand = Array::from_f64s(
            ArrayType::new_static(DataType::F64, [2, 2, 3]),
            vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0, 42.0],
        );
        let output = Array::from_f64s(
            ArrayType::new_static(DataType::F64, [2, 2, 4]),
            vec![
                100.0, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 113.0, 200.0, 201.0, 202.0, 203.0, 210.0, 211.0,
                212.0, 213.0,
            ],
        );
        let metadata_type = ArrayType::new_static(DataType::I8, [2, 2, 2]);
        let metadata = |elements: &[i8]| Array::from_elements(metadata_type.clone(), elements).unwrap();
        let inputs = vec![
            ArrayBatch::new(operand, BatchAxis::new(1)).unwrap(),
            ArrayBatch::new(output, BatchAxis::new(1)).unwrap(),
            ArrayBatch::new(metadata(&[0, 2, 1, 0, 0, 1, 2, 2]), BatchAxis::new(2)).unwrap(),
            ArrayBatch::new(metadata(&[1; 8]), BatchAxis::new(2)).unwrap(),
            ArrayBatch::new(metadata(&[0, 1, 1, 0, 2, 3, 3, 2]), BatchAxis::new(2)).unwrap(),
            ArrayBatch::new(metadata(&[1; 8]), BatchAxis::new(2)).unwrap(),
        ];
        let outputs = operation.batch(&context, &EmptyRegionDriver, inputs.as_slice()).unwrap().into_parts().0;

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(1));
        assert_eq!(
            outputs[0].value().to_f64s(),
            vec![
                10.0, 101.0, 30.0, 103.0, 110.0, 22.0, 112.0, 41.0, 200.0, 11.0, 202.0, 32.0, 20.0, 211.0, 42.0, 213.0,
            ],
        );
    }

    #[test]
    fn test_ragged_all_to_all_composes_named_batch_axes_in_both_orders() {
        let metadata_type = ArrayType::new_static(DataType::I8, [2, 2, 2]);
        let inputs = (
            Array::from_f64s(
                ArrayType::new_static(DataType::F64, [2, 2, 3]),
                vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0, 42.0],
            ),
            Array::from_f64s(
                ArrayType::new_static(DataType::F64, [2, 2, 4]),
                vec![
                    100.0, 101.0, 102.0, 103.0, 110.0, 111.0, 112.0, 113.0, 200.0, 201.0, 202.0, 203.0, 210.0, 211.0,
                    212.0, 213.0,
                ],
            ),
            Array::from_elements(metadata_type.clone(), &[0_i8, 1, 2, 0, 0, 2, 1, 2]).unwrap(),
            Array::from_elements(metadata_type.clone(), &[1_i8; 8]).unwrap(),
            Array::from_elements(metadata_type.clone(), &[0_i8, 1, 1, 0, 2, 3, 3, 2]).unwrap(),
            Array::from_elements(metadata_type, &[1_i8; 8]).unwrap(),
        );
        let transpose = |input: &Array| input.transpose([1, 0, 2]).unwrap();
        let transposed_inputs = (
            transpose(&inputs.0),
            transpose(&inputs.1),
            transpose(&inputs.2),
            transpose(&inputs.3),
            transpose(&inputs.4),
            transpose(&inputs.5),
        );
        let x_then_y: Array = batch(
            |inputs| {
                Ok(batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.ragged_all_to_all(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                        )
                    },
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("y"),
                )?)
            },
            inputs,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("x"),
        )
        .unwrap();
        assert_eq!(x_then_y.r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2, 2, 4]));
        assert_eq!(
            x_then_y.to_f64s(),
            vec![
                10.0, 101.0, 30.0, 103.0, 110.0, 22.0, 112.0, 41.0, 200.0, 11.0, 202.0, 32.0, 20.0, 211.0, 42.0, 213.0,
            ],
        );

        let y_then_x: Array = batch(
            |inputs| {
                Ok(batch(
                    |(operand, output, input_offsets, send_sizes, output_offsets, receive_sizes)| {
                        operand.ragged_all_to_all(
                            "x",
                            &output,
                            &input_offsets,
                            &send_sizes,
                            &output_offsets,
                            &receive_sizes,
                        )
                    },
                    inputs,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    BatchAxisSpecification::named("x"),
                )?)
            },
            transposed_inputs,
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("y"),
        )
        .unwrap();
        assert_eq!(y_then_x.r#type().as_ref(), &ArrayType::new_static(DataType::F64, [2, 2, 4]));
        assert_eq!(
            y_then_x.to_f64s(),
            vec![
                10.0, 101.0, 30.0, 103.0, 200.0, 11.0, 202.0, 32.0, 110.0, 22.0, 112.0, 41.0, 20.0, 211.0, 42.0, 213.0,
            ],
        );
    }

    #[test]
    fn test_logical_ragged_all_to_all_stages_an_unrelated_axis_merge() {
        type TestContext = TracingContext<Array, ArrayOperation<Array>>;

        let input_types = vec![
            ArrayType::new_static(DataType::F32, [2, 3]),
            ArrayType::new_static(DataType::F32, [2, 4]),
            ArrayType::new_static(DataType::I8, [2, 2]),
            ArrayType::new_static(DataType::I8, [2, 2]),
            ArrayType::new_static(DataType::I8, [2, 2]),
            ArrayType::new_static(DataType::I8, [2, 2]),
        ];
        let (output, program) = TestContext::trace(
            |inputs: Vec<_>| {
                let context = BatchingContext::new(inputs[0].context().clone(), 2).with_axis_name("y".to_string());
                let inputs = inputs
                    .into_iter()
                    .enumerate()
                    .map(|(index, input)| ArrayBatch::new(input, BatchAxis::new(usize::from(index >= 2))))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut outputs = RaggedAllToAllOperation::new("x".to_string(), 2)
                    .batch(&context, &EmptyRegionDriver, inputs.as_slice())?
                    .into_parts()
                    .0;
                Ok(outputs.remove(0).into_value())
            },
            input_types,
        )
        .unwrap();

        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::F32, [2, 4]));
        let (operation, instruction) = program
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                ArrayOperation::RaggedAllToAll(operation) => Some((operation, instruction)),
                _ => None,
            })
            .unwrap();
        assert!(!operation.is_physical());
        let merged_input_types = instruction
            .inputs()
            .iter()
            .map(|input| program.atoms()[input.index()].r#type().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(merged_input_types[0], ArrayType::new_static(DataType::F32, [6]));
        assert_eq!(merged_input_types[1], ArrayType::new_static(DataType::F32, [8]));
        for metadata_type in &merged_input_types[2..] {
            assert_eq!(metadata_type, &ArrayType::new_static(DataType::U64, [4]));
        }
        assert_eq!(
            program.instructions().iter().filter(|instruction| instruction.operation().name() == "iota").count(),
            2
        );
        let expected_provenance = Provenance::scope(
            ProvenanceScope::new("ryft"),
            Provenance::scope(
                ProvenanceScope::new("batching"),
                Provenance::scope(ProvenanceScope::new("ragged_all_to_all"), Provenance::unknown()),
            ),
        );
        assert!(
            program.instructions().iter().all(|instruction| instruction.provenance() == &expected_provenance),
            "every merged batching instruction is attributed",
        );
    }

    #[test]
    fn test_unrelated_ragged_all_to_all_batching_rejects_a_dynamic_packed_extent() {
        type TestContext = TracingContext<Array, ArrayOperation<Array>>;

        let input_extent = DimensionVariable::new("input_extent", DimensionBounds::new(0, Some(8)).unwrap());
        let error = TestContext::trace(
            |inputs: Vec<_>| {
                let context = BatchingContext::new(inputs[0].context().clone(), 2).with_axis_name("y".to_string());
                let inputs = inputs
                    .into_iter()
                    .enumerate()
                    .map(|(index, input)| ArrayBatch::new(input, BatchAxis::new(usize::from(index >= 2))))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut outputs = RaggedAllToAllOperation::new("x".to_string(), 2)
                    .batch(&context, &EmptyRegionDriver, inputs.as_slice())?
                    .into_parts()
                    .0;
                Ok(outputs.remove(0).into_value())
            },
            vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(input_extent)])),
                ArrayType::new_static(DataType::F32, [2, 4]),
                ArrayType::new_static(DataType::I32, [2, 2]),
                ArrayType::new_static(DataType::I32, [2, 2]),
                ArrayType::new_static(DataType::I32, [2, 2]),
                ArrayType::new_static(DataType::I32, [2, 2]),
            ],
        )
        .unwrap_err();
        let error = error.downcast_custom::<BatchingError>().unwrap();
        assert_eq!(
            error,
            &BatchingError::UnsupportedOperation {
                message: "`ragged_all_to_all` merged batching requires `operand` axis 0 to have a static extent"
                    .to_string(),
            },
        );

        let trailing_extent = DimensionVariable::new("trailing_extent", DimensionBounds::new(0, Some(8)).unwrap());
        let error = TestContext::trace(
            |inputs: Vec<_>| {
                let context = BatchingContext::new(inputs[0].context().clone(), 2).with_axis_name("y".to_string());
                let inputs = inputs
                    .into_iter()
                    .enumerate()
                    .map(|(index, input)| ArrayBatch::new(input, BatchAxis::new(usize::from(index >= 2))))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut outputs = RaggedAllToAllOperation::new("x".to_string(), 2)
                    .batch(&context, &EmptyRegionDriver, inputs.as_slice())?
                    .into_parts()
                    .0;
                Ok(outputs.remove(0).into_value())
            },
            vec![
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![
                        Dimension::Static(2),
                        Dimension::Static(3),
                        Dimension::Dynamic(trailing_extent.clone()),
                    ]),
                ),
                ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![Dimension::Static(2), Dimension::Static(4), Dimension::Dynamic(trailing_extent)]),
                ),
                ArrayType::new_static(DataType::I32, [2, 2]),
                ArrayType::new_static(DataType::I32, [2, 2]),
                ArrayType::new_static(DataType::I32, [2, 2]),
                ArrayType::new_static(DataType::I32, [2, 2]),
            ],
        )
        .unwrap_err();
        let error = error.downcast_custom::<BatchingError>().unwrap();
        assert_eq!(
            error,
            &BatchingError::UnsupportedOperation {
                message: "`ragged_all_to_all` merged batching requires `operand` axis 1 to have a static extent"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_unrelated_ragged_all_to_all_batching_rejects_a_dynamic_mapped_extent() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let context = TestContext::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let batch_extent = context.input(DimensionType::new(batch.clone()).into());
        let batching_context =
            BatchingContext::<_, ArrayIrBatching>::new(context.clone(), batch_extent).with_axis_name("y".to_string());
        let packed_type = |data_type, dimensions: &[usize]| {
            ArrayType::new(
                data_type,
                Shape::new(
                    std::iter::once(Dimension::Dynamic(batch.clone()))
                        .chain(dimensions.iter().copied().map(Dimension::Static))
                        .collect(),
                ),
            )
        };
        let metadata_type = packed_type(DataType::I32, &[2, 2]);
        let input_types = [
            packed_type(DataType::F32, &[2, 3]),
            packed_type(DataType::F32, &[2, 4]),
            metadata_type.clone(),
            metadata_type.clone(),
            metadata_type.clone(),
            metadata_type,
        ];
        let inputs = input_types.map(|r#type| {
            BatchingTracer::new(
                batching_context.clone(),
                ArrayIrBatch::new(context.input(r#type.into()), BatchAxis::new(0)).unwrap(),
            )
        });
        let error = batching_context
            .bind(
                ArrayIrOperation::RaggedAllToAll(
                    RaggedAllToAllOperation::new("x".to_string(), 2).with_physical_representation(),
                ),
                Vec::new(),
                &inputs,
            )
            .unwrap_err();
        let error = error.downcast_custom::<BatchingError>().unwrap();
        assert_eq!(
            error,
            &BatchingError::UnsupportedOperation {
                message: "`ragged_all_to_all` merged batching requires a statically known mapped-axis extent"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_unrelated_ragged_all_to_all_batching_handles_empty_batches_and_rejects_groups() {
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 0)
            .with_axis_name("y".to_string());
        let empty = |r#type: ArrayType| Array::from_elements::<f32>(r#type, &[]).unwrap();
        let empty_metadata = || Array::from_elements::<i32>(ArrayType::new_static(DataType::I32, [2, 0]), &[]).unwrap();
        let inputs = vec![
            ArrayBatch::new(empty(ArrayType::new_static(DataType::F32, [0, 3])), BatchAxis::new(0)).unwrap(),
            ArrayBatch::new(empty(ArrayType::new_static(DataType::F32, [0, 4])), BatchAxis::new(0)).unwrap(),
            ArrayBatch::new(empty_metadata(), BatchAxis::new(1)).unwrap(),
            ArrayBatch::new(empty_metadata(), BatchAxis::new(1)).unwrap(),
            ArrayBatch::new(empty_metadata(), BatchAxis::new(1)).unwrap(),
            ArrayBatch::new(empty_metadata(), BatchAxis::new(1)).unwrap(),
        ];
        let outputs = RaggedAllToAllOperation::new("x".to_string(), 2)
            .batch(&context, &EmptyRegionDriver, inputs.as_slice())
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().r#type().as_ref(), &ArrayType::new_static(DataType::F32, [0, 4]));

        let grouped = RaggedAllToAllOperation::grouped("x".to_string(), 2, vec![vec![0, 1]]).unwrap();
        assert_eq!(
            grouped.batch(&context, &EmptyRegionDriver, inputs.as_slice()),
            Err(BatchingError::UnsupportedOperation {
                message:
                    "`ragged_all_to_all` axis index groups are not supported when merging an unrelated mapped axis"
                        .to_string(),
            }),
        );

        let mut invalid_inputs = inputs;
        invalid_inputs[1] = ArrayBatch::new(
            Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0, 4]), &[]).unwrap(),
            BatchAxis::new(0),
        )
        .unwrap();
        assert_eq!(
            RaggedAllToAllOperation::new("x".to_string(), 2)
                .batch(&context, &EmptyRegionDriver, invalid_inputs.as_slice())
                .unwrap_err()
                .to_string(),
            "`ragged_all_to_all` operand and output data types must match but got `f32` and `f64`",
        );
    }

    #[test]
    fn test_physical_ragged_all_to_all_skips_zero_byte_transfers_without_offset_overflow() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let data_type = ArrayType::new_static(DataType::F32, [3, usize::MAX, 0]);
        let data = || Array::from_elements::<f32>(data_type.clone(), &[]).unwrap();
        let metadata_type = ArrayType::new_static(DataType::U64, [3, 3]);
        let offsets = Array::from_elements(metadata_type.clone(), &[u64::MAX; 9]).unwrap();
        let sizes = Array::from_elements(metadata_type, &[0_u64; 9]).unwrap();
        let mut outputs = context
            .bind(
                RaggedAllToAllOperation::new("x".to_string(), 3).with_physical_representation(),
                Vec::new(),
                &[data(), data(), offsets.clone(), sizes.clone(), offsets, sizes],
            )
            .unwrap();
        assert_eq!(outputs.remove(0).r#type().as_ref(), &data_type);
    }

    #[test]
    fn test_grouped_ragged_all_to_all_transpose_stages_the_adjoint_sequence() {
        let groups = vec![vec![0, 2], vec![3, 1]];
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let operand = builder.add_input(array_type(DataType::F32, [3]));
        let output = builder.add_input(array_type(DataType::F32, [4]));
        let input_offsets = builder.add_constant(Array::vector(vec![0_i32, 1, 0, 2]));
        let send_sizes = builder.add_constant(Array::vector(vec![1_i32, 1, 0, 1]));
        let output_offsets = builder.add_constant(Array::vector(vec![0_i32, 2, 1, 3]));
        let receive_sizes = builder.add_constant(Array::vector(vec![1_i32, 0, 1, 1]));
        let result = builder
            .add_instruction(
                RaggedAllToAllOperation::grouped("x".to_string(), 4, groups.clone()).unwrap(),
                Vec::new(),
                vec![operand, output, input_offsets, send_sizes, output_offsets, receive_sizes],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![result], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0, 1]).unwrap();

        let all_to_all_groups = pullback
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                ArrayOperation::AllToAll(operation) => operation.options().axis_index_groups(),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(all_to_all_groups, vec![groups.as_slice(), groups.as_slice()]);
        let adjoint = pullback
            .instructions()
            .iter()
            .find_map(|instruction| match instruction.operation() {
                ArrayOperation::RaggedAllToAll(operation) => Some(operation),
                _ => None,
            })
            .unwrap();
        assert_eq!(adjoint.axis_index_groups(), Some(groups.as_slice()));
        let expected_provenance = Provenance::scope(
            ProvenanceScope::new("ryft"),
            Provenance::scope(
                ProvenanceScope::new("differentiation"),
                Provenance::scope(ProvenanceScope::new("ragged_all_to_all_transpose"), Provenance::unknown()),
            ),
        );
        assert!(
            pullback.instructions().iter().all(|instruction| instruction.provenance() == &expected_provenance),
            "every transpose instruction is attributed",
        );
        assert_eq!(
            pullback.to_string(),
            indoc! {r#"
                lambda %0:f32[4] .
                let %1:i32[4] = const [0, 1, 0, 2]
                    %2:i32[4] = const [1, 1, 0, 1]
                    %3:i32[4] = const [0, 2, 1, 3]
                    %4:i32[4] = const [1, 0, 1, 1]
                    %5:i32[4] = all_to_all [
                        axis_name="x",
                        axis_size=4,
                        split_axis=0,
                        concat_axis=0,
                        options=CollectiveOptions { mode: Tiled, axis_index_groups: [[0, 2], [3, 1]] },
                    ] %3
                    %6:i32[4] = all_to_all [
                        axis_name="x",
                        axis_size=4,
                        split_axis=0,
                        concat_axis=0,
                        options=CollectiveOptions { mode: Tiled, axis_index_groups: [[0, 2], [3, 1]] },
                    ] %1
                    %7:f32[3] = zero [type=f32[3]]
                    __ADJOINT__
                    %9:u64[4] = convert_element_type [data_type=u64] %5
                    %10:u64[4] = convert_element_type [data_type=u64] %4
                    %11:i64[5] = zero [type=i64[5]]
                    %12:i64[4] = one [type=i64[4]]
                    %13:i64[4] = neg %12
                    %14:u64[4] = add %9 %10
                    %15:u64[4, 1] = reshape [shape=[4, 1]] %9
                    %16:u64[4, 1] = reshape [shape=[4, 1]] %14
                    %17:i64[5] = scatter [
                        kind=add,
                        __SCATTER_DIMENSIONS__
                    ] %11 %15 %12
                    %18:i64[5] = scatter [
                        kind=add,
                        __SCATTER_DIMENSIONS__
                    ] %17 %16 %13
                    %19:i64[5] = cumulative_sum [axis=0] %18
                    %20:i64[4] = slice [start_indices=[0], limit_indices=[4]] %19
                    %21:i64[4] = zero [type=i64[4]]
                    %22:bool[4] = compare [direction=NotEqual] %20 %21
                    %23:f32[4] = zero [type=f32[4]]
                    %24:f32[4] = select %22 %23 %0
                in (%8, %24)
            "#}
            .replace(
                "__ADJOINT__",
                concat!(
                    "%8:f32[3] = ragged_all_to_all [axis_name=\"x\", axis_size=4, ",
                    "axis_index_groups=[[0, 2], [3, 1]], update_kind=Add] %0 %7 %5 %4 %6 %2",
                ),
            )
            .replace(
                "__SCATTER_DIMENSIONS__",
                concat!(
                    "dimensions=(update_window=[], inserted_window=[0], scatter_to_operand=[0], ",
                    "operand_batching=[], scatter_indices_batching=[]),",
                ),
            )
            .trim_end(),
        );

        let transposed_twice = pullback.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(transposed_twice.input_types(), program.input_types());
        assert_eq!(transposed_twice.output_types(), program.output_types());
    }

    #[test]
    fn test_additive_ragged_all_to_all_output_transpose_does_not_exchange_offsets() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let operand = builder.add_input(array_type(DataType::F32, [3]));
        let output = builder.add_input(array_type(DataType::F32, [4]));
        let input_offsets = builder.add_constant(Array::vector(vec![0_i32]));
        let send_sizes = builder.add_constant(Array::vector(vec![1_i32]));
        let output_offsets = builder.add_constant(Array::vector(vec![0_i32]));
        let receive_sizes = builder.add_constant(Array::vector(vec![1_i32]));
        let result = builder
            .add_instruction(
                RaggedAllToAllOperation::new("x".to_string(), 1).with_additive_updates(),
                Vec::new(),
                vec![operand, output, input_offsets, send_sizes, output_offsets, receive_sizes],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![result], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();

        let pullback = program.transpose_with_respect_to(&[1]).unwrap();

        assert_eq!(
            pullback
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), ArrayOperation::AllToAll(_)))
                .count(),
            0,
        );
    }

    #[test]
    fn test_ragged_all_to_all_transpose_rejects_unavailable_or_unrepresentable_metadata() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = [
            builder.add_input(ArrayType::new_static(DataType::F32, [3])),
            builder.add_input(ArrayType::new_static(DataType::F32, [4])),
            builder.add_input(ArrayType::new_static(DataType::I32, [1])),
            builder.add_input(ArrayType::new_static(DataType::I32, [1])),
            builder.add_input(ArrayType::new_static(DataType::I32, [1])),
            builder.add_input(ArrayType::new_static(DataType::I32, [1])),
        ];
        let result = builder
            .add_instruction(RaggedAllToAllOperation::new("x".to_string(), 1), Vec::new(), inputs.to_vec(), None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(
                vec![result],
                vec![Placeholder, Placeholder, Placeholder, Placeholder, Placeholder, Placeholder],
                Placeholder,
            )
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1, 2]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`ragged_all_to_all` transpose requires `input_offsets` to be a known primal residual"
        ));

        let output_extent = DimensionVariable::new("output_extent", DimensionBounds::new(0, Some(8)).unwrap());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let operand = builder.add_input(ArrayType::new_static(DataType::F32, [3]));
        let output =
            builder.add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(output_extent)])));
        let input_offsets = builder.add_constant(Array::vector(vec![0_i32]));
        let send_sizes = builder.add_constant(Array::vector(vec![1_i32]));
        let output_offsets = builder.add_constant(Array::vector(vec![0_i32]));
        let receive_sizes = builder.add_constant(Array::vector(vec![1_i32]));
        let result = builder
            .add_instruction(
                RaggedAllToAllOperation::new("x".to_string(), 1),
                Vec::new(),
                vec![operand, output, input_offsets, send_sizes, output_offsets, receive_sizes],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![result], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`ragged_all_to_all` transpose requires a static output leading dimension"
        ));

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let operand = builder.add_input(ArrayType::new_static(DataType::F32, [1]));
        let output = builder.add_input(ArrayType::new_static(DataType::F32, [usize::MAX]));
        let input_offsets = builder.add_constant(Array::vector(vec![0_i32]));
        let send_sizes = builder.add_constant(Array::vector(vec![0_i32]));
        let output_offsets = builder.add_constant(Array::vector(vec![0_i32]));
        let receive_sizes = builder.add_constant(Array::vector(vec![0_i32]));
        let result = builder
            .add_instruction(
                RaggedAllToAllOperation::new("x".to_string(), 1),
                Vec::new(),
                vec![operand, output, input_offsets, send_sizes, output_offsets, receive_sizes],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![result], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "`ragged_all_to_all` transpose marker extent does not fit in `usize`"
        ));
    }

    #[test]
    fn test_ragged_all_to_all_transpose_supports_leading_and_trailing_sharding() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("data", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        for dimensions in [
            vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()],
            vec![ShardingDimension::replicated(), ShardingDimension::sharded(["data"])],
        ] {
            let sharding = Sharding::new(mesh.clone(), dimensions).unwrap();
            let operand_type = ArrayType::new_static(DataType::F32, [3, 2]).with_sharding(sharding.clone()).unwrap();
            let output_type = ArrayType::new_static(DataType::F32, [4, 2]).with_sharding(sharding).unwrap();
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let operand = builder.add_input(operand_type.clone());
            let output = builder.add_input(output_type.clone());
            let input_offsets = builder.add_constant(Array::vector(vec![0_i32]));
            let send_sizes = builder.add_constant(Array::vector(vec![1_i32]));
            let output_offsets = builder.add_constant(Array::vector(vec![0_i32]));
            let receive_sizes = builder.add_constant(Array::vector(vec![1_i32]));
            let result = builder
                .add_instruction(
                    RaggedAllToAllOperation::new("x".to_string(), 1),
                    Vec::new(),
                    vec![operand, output, input_offsets, send_sizes, output_offsets, receive_sizes],
                    None,
                )
                .unwrap()[0];
            let program = builder
                .build::<Vec<Array>, Array>(vec![result], vec![Placeholder, Placeholder], Placeholder)
                .unwrap();
            let pullback = program.transpose_with_respect_to(&[0, 1]).unwrap();
            assert_eq!(pullback.input_types(), &[output_type]);
            assert_eq!(pullback.output_types(), &[operand_type, program.input_types()[1].clone()]);
        }
    }
}
