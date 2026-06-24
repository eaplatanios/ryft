//! Reusable staged operation enums for built-in primitives.
//!
//! [`ArrayOperation`] and [`LinearArrayOperation`] contain the core operations implemented by `ryft-core`. Backends
//! that need additional operations should define their own operation enum that wraps these core enums together with
//! backend-specific variants, so transform, interpretation, and lowering rules remain statically typed and owned by
//! the backend that understands each operation.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Sub};

use ryft_macros::{Operation, TransposableOperation};

use crate::batching::BatchingError;
use crate::contexts::{Context, EagerContext, StagingContext};
use crate::differentiation::Cotangent;
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::arithmetic::{
    AddOperation, DivOperation, MulOperation, NegOperation, Scale, ScaleOperation, SubOperation,
};
use crate::operations::compare::{Compare, CompareOperation};
use crate::operations::constants::{
    ConstantOperation, Fill, FillOperation, MaybeZeroOperation, One, OneLike, OneLikeOperation, OneOperation, Zero,
    ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::scan::{interpret_scan_lanes, read_scan_lane};
use crate::operations::control_flow::{
    ConditionOperation, ScanOperation, Select, SelectCondition, SelectOperation, WhileOperation,
};
use crate::operations::differentiation::StopGradientOperation;
use crate::operations::logical::{AndOperation, NotOperation, OrOperation, XorOperation};
use crate::operations::manipulation::{
    Broadcast, BroadcastOperation, ConcatenateOperation, DynamicSliceOperation, DynamicUpdateSliceOperation,
    GatherOperation, LinearDynamicSliceOperation, LinearDynamicUpdateSliceOperation, LinearGatherOperation,
    LinearScatterAddOperation, PadOperation, ReshapeOperation, ScatterDimensionNumbers, ScatterOperation,
    ScatterReductionKind, Slice, SliceOperation, Transpose, TransposeOperation, UpdateSliceOperation,
};
use crate::operations::sharding::{ConstrainSharding, Reshard, ReshardOperation, ShardingConstraintOperation};
use crate::operations::trigonometric::{CosOperation, SinOperation};
use crate::operations::{BooleanLike, InterpretableOperation, InterpretableProgramOperation, Operation};
use crate::parameters::{Parameter, Parameterized, Placeholder};
use crate::payloads::{Captured, Input};
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{AbstractTracingContext, Tracer};
use crate::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchableProgramOperation, BatchingContext, ProgramBatchingContext,
    ProgramBatchingOutputAxes,
};
use crate::tracing_v2::differentiation::{
    CaptureParameterizedOperation, DifferentiationContext, JvpTracer, LinearOperationOf, LinearizableProgramOperation,
    LinearizationContextOf, NestedLinearization, TangentContext,
};
use crate::tracing_v2::operations::collective::CollectiveOperation;
use crate::tracing_v2::operations::custom_derivatives::{
    CustomJvpOperation, CustomVjpCallOperation, CustomVjpOperation, CustomVjpResidual,
};
use crate::tracing_v2::operations::dot::{LeftDot, LeftDotOperation, MaybeDot, RightDot, RightDotOperation};
use crate::tracing_v2::operations::memory::{TransferToMemory, TransferToMemoryOperation};
use crate::tracing_v2::operations::recompute::RecomputeOperation;
use crate::tracing_v2::operations::reduce::ReduceOperation;
use crate::tracing_v2::operations::select::LinearSelectOperation;
use crate::tracing_v2::operations::{DotDimensionNumbers, DotOperation};
use crate::tracing_v2::rematerialization::{MaybeRematerializationName, RematerializationNameOperation};
use crate::tracing_v2::{DifferentiableOperation, RematerializationName, ResidualizedOperation, ValueOrCapture};
use crate::types::{ArrayType, Type, TypeError, Typed};

use super::bounds::{
    SupportsArithmeticOperations, SupportsComparisonOperations, SupportsConstantOperations,
    SupportsLinearAlgebraOperations, SupportsLinearArithmeticOperations, SupportsLinearArrayOperation,
    SupportsManipulationOperations, SupportsTrigonometricOperations,
};
use super::captures::MaterializeCaptureOperation;
use super::control_flow::{
    DefactorizedOperation, LinearOperandConditionOperation, SupportsLinearWhile, batch_condition_with_interpreter,
    batch_while_with_interpreter,
};
use super::dot::DotOps;
use super::scan::{LinearScanInterpretation, LinearScanOperation};
use crate::operations::manipulation::Reshape;

/// Reusable operation enum for ordinary staged programs.
///
/// [`ArrayOperation`] is the ordinary operation enum for core tests and backend crates. Most variants are thin tags
/// around one semantic primitive defined elsewhere in [`super`].
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation): for example [`Zero`](Self::Zero) wraps a [`ZeroOperation`],
/// [`Scale`](Self::Scale) a [`ScaleOperation`], and [`Dot`](Self::Dot) a [`DotOperation`].
#[derive(Clone, Debug, Operation)]
pub enum ArrayOperation<V: Value<ArrayType>> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<ArrayType, V>),
    Fill(FillOperation<ArrayType, f64>),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Scale(ScaleOperation<ArrayType, V>),
    Mul(MulOperation),
    Div(DivOperation),
    Sin(SinOperation),
    Cos(CosOperation),
    StopGradient(StopGradientOperation),
    RematerializationName(RematerializationNameOperation),
    TransferToMemory(TransferToMemoryOperation),
    Dot(DotOperation),
    Transpose(TransposeOperation),
    Reshape(ReshapeOperation),
    Reshard(ReshardOperation),
    ShardingConstraint(ShardingConstraintOperation),
    Broadcast(BroadcastOperation),
    Slice(SliceOperation),
    UpdateSlice(UpdateSliceOperation),
    DynamicSlice(DynamicSliceOperation),
    DynamicUpdateSlice(DynamicUpdateSliceOperation),
    Pad(PadOperation),
    Concatenate(ConcatenateOperation),
    Gather(GatherOperation),
    Scatter(ScatterOperation),
    Reduce(ReduceOperation),
    Compare(CompareOperation),
    Not(NotOperation),
    And(AndOperation),
    Or(OrOperation),
    Xor(XorOperation),
    Collective(CollectiveOperation),
    Select(SelectOperation),
    Condition(Box<ConditionOperation<ArrayType, V, Self>>),
    While(Box<WhileOperation<ArrayType, V, Self>>),
    Scan(Box<ScanOperation<ArrayType, V, Self>>),
    CustomJvp(Box<CustomJvpOperation<ArrayType, V, Self>>),
    CustomVjp(Box<CustomVjpOperation<ArrayType, V, Self>>),
}

// Differentiation (JVP) for the `ArrayOperation` sum type: each variant delegates to its backing operation's own
// `DifferentiableOperation` rule. The per-variant `<Payload>: DifferentiableOperation<D>` bounds cover the
// non-self-referential variants; the self-referential higher-order
// `Condition`/`While`/`Scan` and `CustomJvp`/`CustomVjp` arms resolve against this impl's assumed
// `Self: DifferentiableOperation<D>`. The remaining where-clause spells the leaf closure of value and
// linear-operation capabilities those per-variant rules require.
impl<V: Value<ArrayType>, D> DifferentiableOperation<D> for ArrayOperation<V>
where
    ZeroOperation<ArrayType>: DifferentiableOperation<D>,
    ZeroLikeOperation: DifferentiableOperation<D>,
    OneOperation<ArrayType>: DifferentiableOperation<D>,
    OneLikeOperation: DifferentiableOperation<D>,
    ConstantOperation<ArrayType, V>: DifferentiableOperation<D>,
    FillOperation<ArrayType, f64>: DifferentiableOperation<D>,
    NegOperation: DifferentiableOperation<D>,
    AddOperation: DifferentiableOperation<D>,
    SubOperation: DifferentiableOperation<D>,
    ScaleOperation<ArrayType, V>: DifferentiableOperation<D>,
    MulOperation: DifferentiableOperation<D>,
    DivOperation: DifferentiableOperation<D>,
    SinOperation: DifferentiableOperation<D>,
    CosOperation: DifferentiableOperation<D>,
    StopGradientOperation: DifferentiableOperation<D>,
    RematerializationNameOperation: DifferentiableOperation<D>,
    TransferToMemoryOperation: DifferentiableOperation<D>,
    DotOperation: DifferentiableOperation<D>,
    TransposeOperation: DifferentiableOperation<D>,
    ReshapeOperation: DifferentiableOperation<D>,
    ReshardOperation: DifferentiableOperation<D>,
    ShardingConstraintOperation: DifferentiableOperation<D>,
    BroadcastOperation: DifferentiableOperation<D>,
    SliceOperation: DifferentiableOperation<D>,
    UpdateSliceOperation: DifferentiableOperation<D>,
    DynamicSliceOperation: DifferentiableOperation<D>,
    DynamicUpdateSliceOperation: DifferentiableOperation<D>,
    PadOperation: DifferentiableOperation<D>,
    ConcatenateOperation: DifferentiableOperation<D>,
    GatherOperation: DifferentiableOperation<D>,
    ScatterOperation: DifferentiableOperation<D>,
    ReduceOperation: DifferentiableOperation<D>,
    CompareOperation: DifferentiableOperation<D>,
    NotOperation: DifferentiableOperation<D>,
    AndOperation: DifferentiableOperation<D>,
    OrOperation: DifferentiableOperation<D>,
    XorOperation: DifferentiableOperation<D>,
    CollectiveOperation: DifferentiableOperation<D>,
    SelectOperation: DifferentiableOperation<D>,
    D: DifferentiationContext<Type = ArrayType, Constant = V> + Domain<Operation = ArrayOperation<V>>,
    D::Operation: From<ZeroOperation<ArrayType>> + From<OneOperation<ArrayType>> + From<FillOperation<ArrayType, f64>>,
    D::Value: RematerializationName + TransferToMemory,
    D::Value: Add<Output = D::Value>
        + Sub<Output = D::Value>
        + Mul<Output = D::Value>
        + Div<Output = D::Value>
        + Neg<Output = D::Value>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + Compare<Output = D::Value>
        + BitAnd<Output = D::Value>
        + BitOr<Output = D::Value>
        + BitXor<Output = D::Value>
        + Not<Output = D::Value>
        + Select<Condition = D::Value>
        + SelectCondition<Condition = D::Value>
        + BooleanLike,
    D::Tangent: Transpose + Broadcast + super::reduce::Reduce + Slice + Reshard + ConstrainSharding,
    <D::Value as Parameterized<D::Value>>::ParameterStructure: std::fmt::Debug + PartialEq,
    LinearOperationOf<D>: SupportsLinearArrayOperation<ValueOrCapture<ArrayType, D::Value>>
        + ResidualizedOperation<D>
        + From<CustomVjpCallOperation<ArrayType, V, ArrayOperation<V>, ValueOrCapture<ArrayType, D::Value>>>
        + From<TransferToMemoryOperation>
        + From<ConcatenateOperation>
        + From<LinearSelectOperation<ValueOrCapture<ArrayType, D::Value>>>
        + From<LinearDynamicSliceOperation<ValueOrCapture<ArrayType, D::Value>>>
        + From<LinearDynamicUpdateSliceOperation<ValueOrCapture<ArrayType, D::Value>>>
        + From<LinearGatherOperation<ValueOrCapture<ArrayType, D::Value>>>
        + From<LinearScatterAddOperation<ValueOrCapture<ArrayType, D::Value>>>
        + From<
            ConditionOperation<
                ArrayType,
                D::Tangent,
                LinearOperationOf<D>,
                ValueOrCapture<ArrayType, D::Value>,
                Captured,
            >,
        > + SupportsLinearWhile<ArrayType, D::Tangent, ValueOrCapture<ArrayType, D::Value>, ArrayOperation<V>>
        + LinearScanOperation<ArrayType, D::Tangent, D::Value>,
    LinearOperationOf<D>: MaybeZeroOperation<ArrayType>,
    ArrayOperation<V>: Clone + LinearizableProgramOperation<D>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        match self {
            Self::Zero(operation) => operation.jvp(context, inputs),
            Self::ZeroLike(operation) => operation.jvp(context, inputs),
            Self::One(operation) => operation.jvp(context, inputs),
            Self::OneLike(operation) => operation.jvp(context, inputs),
            Self::Constant(operation) => operation.jvp(context, inputs),
            Self::Fill(operation) => operation.jvp(context, inputs),
            Self::Neg(operation) => operation.jvp(context, inputs),
            Self::Add(operation) => operation.jvp(context, inputs),
            Self::Sub(operation) => operation.jvp(context, inputs),
            Self::Scale(operation) => operation.jvp(context, inputs),
            Self::Mul(operation) => operation.jvp(context, inputs),
            Self::Div(operation) => operation.jvp(context, inputs),
            Self::Sin(operation) => operation.jvp(context, inputs),
            Self::Cos(operation) => operation.jvp(context, inputs),
            Self::StopGradient(operation) => operation.jvp(context, inputs),
            Self::RematerializationName(operation) => operation.jvp(context, inputs),
            Self::TransferToMemory(operation) => operation.jvp(context, inputs),
            Self::Dot(operation) => operation.jvp(context, inputs),
            Self::Transpose(operation) => operation.jvp(context, inputs),
            Self::Reshape(operation) => operation.jvp(context, inputs),
            Self::Reshard(operation) => operation.jvp(context, inputs),
            Self::ShardingConstraint(operation) => operation.jvp(context, inputs),
            Self::Broadcast(operation) => operation.jvp(context, inputs),
            Self::Slice(operation) => operation.jvp(context, inputs),
            Self::UpdateSlice(operation) => operation.jvp(context, inputs),
            Self::DynamicSlice(operation) => operation.jvp(context, inputs),
            Self::DynamicUpdateSlice(operation) => operation.jvp(context, inputs),
            Self::Pad(operation) => operation.jvp(context, inputs),
            Self::Concatenate(operation) => operation.jvp(context, inputs),
            Self::Gather(operation) => operation.jvp(context, inputs),
            Self::Scatter(operation) => operation.jvp(context, inputs),
            Self::Reduce(operation) => operation.jvp(context, inputs),
            Self::Compare(operation) => operation.jvp(context, inputs),
            Self::Not(operation) => operation.jvp(context, inputs),
            Self::And(operation) => operation.jvp(context, inputs),
            Self::Or(operation) => operation.jvp(context, inputs),
            Self::Xor(operation) => operation.jvp(context, inputs),
            Self::Collective(operation) => operation.jvp(context, inputs),
            Self::Select(operation) => operation.jvp(context, inputs),
            Self::Condition(operation) => operation.jvp(context, inputs),
            Self::While(operation) => operation.jvp(context, inputs),
            Self::Scan(operation) => operation.jvp(context, inputs),
            Self::CustomJvp(operation) => operation.jvp(context, inputs),
            Self::CustomVjp(operation) => operation.jvp(context, inputs),
        }
    }
}

/// Reusable operation enum for staged linear programs.
///
/// [`LinearArrayOperation`] is the linear-program sibling of [`ArrayOperation`]. It contains operations that can
/// appear in tangent and cotangent programs, including captured-factor linear maps such as [`LeftDot`](Self::LeftDot)
/// and [`RightDot`](Self::RightDot), and the linearized higher-order operations needed by rematerialization and
/// control flow.
///
/// Each variant wraps exactly the backing operation struct that owns the variant's semantics (type inference,
/// rendering, and interpretation). The [`Operation`]/[`Display`] and per-variant [`From`]/[`TryFrom`] impls are
/// defined for [`ArrayType`].
///
/// The `V` parameter is the linear program's value and constant-table type. It instantiates to concrete tangent
/// values for eager linear execution and to tracers when one transform stages another. The `C` parameter is the
/// constant type of captured primal programs such as [`CustomVjpCall`](Self::CustomVjpCall), which are written over
/// context constants rather than over the linear program's tangent constants.
#[derive(Clone, Debug, Operation, TransposableOperation)]
pub enum LinearArrayOperation<
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    F: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
> {
    Zero(ZeroOperation<ArrayType>),
    ZeroLike(ZeroLikeOperation),
    One(OneOperation<ArrayType>),
    OneLike(OneLikeOperation),
    Constant(ConstantOperation<ArrayType, V, Input>),
    Fill(FillOperation<ArrayType, f64>),
    Neg(NegOperation),
    Add(AddOperation),
    Sub(SubOperation),
    Scale(ScaleOperation<ArrayType, F, Input>),
    Mul(MulOperation),
    TransferToMemory(TransferToMemoryOperation),
    Transpose(TransposeOperation),
    LeftDot(LeftDotOperation<F, Input>),
    RightDot(RightDotOperation<F, Input>),
    Reshape(ReshapeOperation),
    Reshard(ReshardOperation),
    ShardingConstraint(ShardingConstraintOperation),
    Broadcast(BroadcastOperation),
    Slice(SliceOperation),
    UpdateSlice(UpdateSliceOperation),
    DynamicSlice(LinearDynamicSliceOperation<F>),
    DynamicUpdateSlice(LinearDynamicUpdateSliceOperation<F>),
    Gather(LinearGatherOperation<F>),
    ScatterAdd(LinearScatterAddOperation<F>),
    Pad(PadOperation),
    Concatenate(ConcatenateOperation),
    Reduce(ReduceOperation),
    Select(LinearSelectOperation<F>),
    Residual(MaterializeCaptureOperation<F>),
    Recompute(RecomputeOperation<P>),
    Condition(ConditionOperation<ArrayType, V, Self, F, Captured>),
    OperandCondition(LinearOperandConditionOperation<V, Self>),
    While(Box<WhileOperation<ArrayType, V, Self>>),
    Scan(Box<ScanOperation<ArrayType, V, LinearArrayOperation<V, C, ValueOrCapture<ArrayType, V>, P>, F>>),
    CustomVjpCall(Box<CustomVjpCallOperation<ArrayType, C, P, F>>),
}

impl<V: Value<ArrayType>> MaybeRematerializationName for ArrayOperation<V> {
    #[inline]
    fn rematerialization_name(&self) -> Option<&str> {
        match self {
            Self::RematerializationName(operation) => Some(operation.tag()),
            _ => None,
        }
    }
}

impl<V> MaybeDot for ArrayOperation<V>
where
    V: Value<ArrayType>,
{
    #[inline]
    fn dot_dimensions(&self) -> Option<&DotDimensionNumbers> {
        match self {
            Self::Dot(operation) => Some(operation.dimensions()),
            _ => None,
        }
    }
}

/// Disposition of one residual-reference index while defactorizing a nested linear program (see
/// [`defactorize_nested_linear_program`]).
#[derive(Copy, Clone)]
enum NestedResidualDisposition {
    /// The referenced residual enters the rewritten program as the trailing input at this position, and references
    /// to it are rewritten into operand form against that input.
    Operand(usize),

    /// The referenced residual stays a factor payload, re-indexed to this position.
    Factor(usize),
}

/// Rewrites a nested linear `program`'s residual references into operand form against new trailing inputs.
///
/// This is the whole-program counterpart of [`SupportsLinearWhile::defactorize`], used by the higher-order
/// defactorization arms: operand-form condition branches receive their forwarded while-body residuals as trailing
/// inputs, and operand-form scan bodies receive the lane slices of their moved residual stacks as trailing scanned
/// inputs. The returned program consumes `[original_inputs..., forwarded_inputs...]` with one trailing input per
/// entry of `forwarded_input_types`, and each instruction is rewritten according to `dispositions`, indexed by the
/// program's residual-reference namespace:
///
///   - Instructions whose references all map to [`NestedResidualDisposition::Factor`] keep their factor form with
///     the references re-indexed to the compacted factor positions.
///   - Instructions whose references all map to [`NestedResidualDisposition::Operand`] are rewritten into operand
///     form against the trailing input atoms through [`SupportsLinearWhile::defactorize`] (a nested residual
///     injection collapses to forwarding the trailing input).
///   - Instructions referencing both kinds are rejected, mirroring the mixed constant/reference index rejection of
///     the dynamic-slicing defactorization arms (defactorization stages exactly one instruction per source
///     instruction).
fn defactorize_nested_linear_program<V, C, R, P>(
    program: &Program<ArrayType, V, LinearArrayOperation<V, C, ValueOrCapture<ArrayType, R>, P>, Vec<V>, Vec<V>>,
    dispositions: &[Option<NestedResidualDisposition>],
    forwarded_input_types: &[ArrayType],
) -> Result<
    Program<ArrayType, V, LinearArrayOperation<V, C, ValueOrCapture<ArrayType, R>, P>, Vec<V>, Vec<V>>,
    ProgramError,
>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    R: Value<ArrayType>,
    P: Clone
        + Operation<ArrayType>
        + From<MulOperation>
        + From<DotOperation>
        + From<SelectOperation>
        + From<DynamicSliceOperation>
        + From<DynamicUpdateSliceOperation>
        + From<ConcatenateOperation>,
{
    let mut builder =
        ProgramBuilder::<ArrayType, V, LinearArrayOperation<V, C, ValueOrCapture<ArrayType, R>, P>>::new();
    let mut atom_map: Vec<Option<AtomId>> = vec![None; program.atoms().len()];
    for (program_atom, input_type) in program.input_ids().iter().zip(program.input_types().into_iter()) {
        atom_map[program_atom.index()] = Some(builder.add_input(input_type));
    }
    let forwarded_atoms = forwarded_input_types
        .iter()
        .map(|forwarded_type| builder.add_input(forwarded_type.clone()))
        .collect::<Vec<_>>();
    for (atom_index, atom) in program.atoms().iter().enumerate() {
        if let Atom::Constant(constant) = atom {
            atom_map[atom_index] = Some(builder.add_constant(constant.clone()));
        }
    }
    let map_atom = |atom_map: &[Option<AtomId>], atom: AtomId| {
        atom_map.get(atom.index()).copied().flatten().ok_or(ProgramError::UnboundAtomId { id: atom })
    };
    let resolve_disposition = |index: usize| {
        dispositions.get(index).copied().flatten().ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "nested linear program references residual {index} but only {} residuals were dispositioned",
                dispositions.len(),
            ))
        })
    };
    for instruction in program.instructions() {
        let inputs = instruction
            .inputs()
            .iter()
            .map(|input| map_atom(atom_map.as_slice(), *input))
            .collect::<Result<Vec<_>, _>>()?;
        let mut references_operand_residual = false;
        let mut references_factor_residual = false;
        instruction.operation().try_map_captures(&mut |factor: &ValueOrCapture<ArrayType, R>| {
            if let ValueOrCapture::Capture { index, .. } = factor {
                match resolve_disposition(*index)? {
                    NestedResidualDisposition::Operand(_) => references_operand_residual = true,
                    NestedResidualDisposition::Factor(_) => references_factor_residual = true,
                }
            }
            Ok(factor.clone())
        })?;
        if references_operand_residual && references_factor_residual {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "jvp of a while loop whose body pushforward stages {} over a mix of loop-varying and \
                     constant-stack residual references is not supported",
                    instruction.operation().name(),
                ),
            });
        }
        let remapped = instruction.operation().try_map_captures(&mut |factor| match factor {
            ValueOrCapture::Capture { index, r#type } => {
                let position = match resolve_disposition(*index)? {
                    NestedResidualDisposition::Operand(position) => position,
                    NestedResidualDisposition::Factor(position) => position,
                };
                Ok(ValueOrCapture::Capture { index: position, r#type: r#type.clone() })
            }
            ValueOrCapture::Value(value) => Ok(ValueOrCapture::Value(value.clone())),
        })?;
        if !references_operand_residual {
            let outputs = builder.add_instruction(remapped, inputs)?.to_vec();
            check_count!("output", outputs, instruction.outputs().len(), ProgramError);
            for (program_atom, builder_atom) in instruction.outputs().iter().zip(outputs.into_iter()) {
                atom_map[program_atom.index()] = Some(builder_atom);
            }
            continue;
        }
        match remapped.defactorize(forwarded_atoms.as_slice(), inputs)? {
            DefactorizedOperation::Operation { operation, inputs } => {
                let outputs = builder.add_instruction(operation, inputs)?.to_vec();
                check_count!("output", outputs, instruction.outputs().len(), ProgramError);
                for (program_atom, builder_atom) in instruction.outputs().iter().zip(outputs.into_iter()) {
                    atom_map[program_atom.index()] = Some(builder_atom);
                }
            }
            DefactorizedOperation::Forward { atom } => {
                check_count!("output", instruction.outputs(), 1, ProgramError);
                atom_map[instruction.outputs()[0].index()] = Some(atom);
            }
        }
    }
    let outputs = program
        .output_ids()
        .iter()
        .map(|output| map_atom(atom_map.as_slice(), *output))
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let input_count = program.input_ids().len() + forwarded_input_types.len();
    let output_count = outputs.len();
    builder.build(outputs, vec![Placeholder; input_count], vec![Placeholder; output_count])
}

// TODO(eaplatanios): Can we get rid of this similar to what we did for some of the scan-related functionality?
impl<V, C, R, P> SupportsLinearWhile<ArrayType, V, ValueOrCapture<ArrayType, R>, P>
    for LinearArrayOperation<V, C, ValueOrCapture<ArrayType, R>, P>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    R: Value<ArrayType>,
    P: Clone
        + Operation<ArrayType>
        + From<MulOperation>
        + From<DotOperation>
        + From<SelectOperation>
        + From<DynamicSliceOperation>
        + From<DynamicUpdateSliceOperation>
        + From<ConcatenateOperation>,
{
    #[inline]
    fn recompute_operation(operation: P) -> Self {
        LinearArrayOperation::Recompute(RecomputeOperation::new(operation))
    }

    #[inline]
    fn residual_operation(factor: ValueOrCapture<ArrayType, R>) -> Self {
        LinearArrayOperation::Residual(MaterializeCaptureOperation::new(factor))
    }

    fn defactorize(
        &self,
        residual_atoms: &[AtomId],
        mut inputs: Vec<AtomId>,
    ) -> Result<DefactorizedOperation<Self>, ProgramError> {
        let resolve_residual_atom = |index: usize| {
            residual_atoms.get(index).copied().ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "while body pushforward references residual {index} but only {} residuals were captured",
                    residual_atoms.len(),
                ))
            })
        };
        match self {
            // `Scale` by a loop-varying residual becomes a recomputed elementwise product against the recomputed
            // residual atom; `LeftDot` / `RightDot` become the recomputed operand-form dot with the residual spliced
            // in on the side the captured factor occupied. All three target `Recompute` so that every
            // recomputed-primal instruction in a fused while body carries the same provenance.
            Self::Scale(operation) if matches!(operation.factor(), ValueOrCapture::Capture { .. }) => {
                let ValueOrCapture::Capture { index, .. } = operation.factor() else { unreachable!() };
                inputs.insert(0, resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(RecomputeOperation::new(P::from(MulOperation))),
                    inputs,
                })
            }
            Self::LeftDot(operation) if matches!(operation.factor(), ValueOrCapture::Capture { .. }) => {
                let ValueOrCapture::Capture { index, .. } = operation.factor() else { unreachable!() };
                inputs.insert(0, resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(RecomputeOperation::new(P::from(
                        DotOperation::new(operation.dimensions().clone())
                            .with_output_sharding(operation.output_sharding().cloned()),
                    ))),
                    inputs,
                })
            }
            Self::RightDot(operation) if matches!(operation.factor(), ValueOrCapture::Capture { .. }) => {
                let ValueOrCapture::Capture { index, .. } = operation.factor() else { unreachable!() };
                inputs.push(resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(RecomputeOperation::new(P::from(
                        DotOperation::new(operation.dimensions().clone())
                            .with_output_sharding(operation.output_sharding().cloned()),
                    ))),
                    inputs,
                })
            }
            // `DynamicSlice` / `DynamicUpdateSlice` over loop-varying residual start indices become the recomputed
            // operand-form primal operations with the residual atoms spliced in as index operands. Mixed
            // constant/reference index lists are rejected because defactorization stages exactly one instruction,
            // while constant indices would need their own materializing instructions.
            Self::DynamicSlice(operation)
                if operation.start_indices().iter().any(|index| matches!(index, ValueOrCapture::Capture { .. })) =>
            {
                for start_index in operation.start_indices() {
                    let ValueOrCapture::Capture { index, .. } = start_index else {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "jvp of a while loop whose body captures a mix of loop-varying and constant \
                                      dynamic_slice start indices is not supported"
                                .to_string(),
                        });
                    };
                    inputs.push(resolve_residual_atom(*index)?);
                }
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(RecomputeOperation::new(P::from(
                        DynamicSliceOperation::new(operation.sizes().to_vec()),
                    ))),
                    inputs,
                })
            }
            Self::DynamicUpdateSlice(operation)
                if operation.start_indices().iter().any(|index| matches!(index, ValueOrCapture::Capture { .. })) =>
            {
                for start_index in operation.start_indices() {
                    let ValueOrCapture::Capture { index, .. } = start_index else {
                        return Err(ProgramError::UnsupportedOperation {
                            message: "jvp of a while loop whose body captures a mix of loop-varying and constant \
                                      dynamic_update_slice start indices is not supported"
                                .to_string(),
                        });
                    };
                    inputs.push(resolve_residual_atom(*index)?);
                }
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(RecomputeOperation::new(P::from(
                        DynamicUpdateSliceOperation,
                    ))),
                    inputs,
                })
            }
            // A nested loop's residual injection materializes a value the fused body already recomputes, so the
            // instruction collapses to forwarding the residual atom.
            Self::Residual(operation) if matches!(operation.capture(), ValueOrCapture::Capture { .. }) => {
                let ValueOrCapture::Capture { index, .. } = operation.capture() else { unreachable!() };
                Ok(DefactorizedOperation::Forward { atom: resolve_residual_atom(*index)? })
            }
            // `Select` over a loop-varying residual condition becomes the recomputed operand-form primal select
            // with the residual atom spliced in as the condition operand.
            Self::Select(operation) if matches!(operation.condition(), ValueOrCapture::Capture { .. }) => {
                let ValueOrCapture::Capture { index, .. } = operation.condition() else { unreachable!() };
                inputs.insert(0, resolve_residual_atom(*index)?);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::Recompute(RecomputeOperation::new(P::from(SelectOperation))),
                    inputs,
                })
            }
            // A loop-varying condition predicate becomes operand 0 of an operand-form condition
            // (`OperandCondition`). The branch programs may carry their own references into the same while-body
            // residual table (the condition JVP rule remapped them onto the enclosing linearization environment), so
            // the union of the residual indices referenced by both branches is forwarded as additional trailing
            // operands — both branches receive the full union because their signatures must agree — and each branch
            // is recursively defactorized against the new trailing branch inputs.
            Self::Condition(operation) if matches!(operation.predicate(), ValueOrCapture::Capture { .. }) => {
                let ValueOrCapture::Capture { index, .. } = operation.predicate() else { unreachable!() };
                let (true_branch, false_branch) = (operation.true_branch(), operation.false_branch());
                let predicate_atom = resolve_residual_atom(*index)?;
                let mut forwarded_residuals = BTreeMap::new();
                for branch in [true_branch, false_branch] {
                    for instruction in branch.instructions() {
                        instruction.operation().try_map_captures(&mut |factor: &ValueOrCapture<ArrayType, R>| {
                            if let ValueOrCapture::Capture { index, r#type } = factor {
                                forwarded_residuals.entry(*index).or_insert_with(|| r#type.clone());
                            }
                            Ok(factor.clone())
                        })?;
                    }
                }
                let mut dispositions = vec![None; residual_atoms.len()];
                let mut forwarded_types = Vec::with_capacity(forwarded_residuals.len());
                let mut forwarded_atoms = Vec::with_capacity(forwarded_residuals.len());
                for (position, (residual_index, residual_type)) in forwarded_residuals.into_iter().enumerate() {
                    forwarded_atoms.push(resolve_residual_atom(residual_index)?);
                    dispositions[residual_index] = Some(NestedResidualDisposition::Operand(position));
                    forwarded_types.push(residual_type);
                }
                let true_branch = defactorize_nested_linear_program(
                    true_branch,
                    dispositions.as_slice(),
                    forwarded_types.as_slice(),
                )?;
                let false_branch = defactorize_nested_linear_program(
                    false_branch,
                    dispositions.as_slice(),
                    forwarded_types.as_slice(),
                )?;
                let mut condition_inputs = Vec::with_capacity(1 + inputs.len() + forwarded_atoms.len());
                condition_inputs.push(predicate_atom);
                condition_inputs.extend(inputs);
                condition_inputs.extend(forwarded_atoms);
                Ok(DefactorizedOperation::Operation {
                    operation: LinearArrayOperation::OperandCondition(LinearOperandConditionOperation::new(
                        Box::new(true_branch),
                        Box::new(false_branch),
                    )),
                    inputs: condition_inputs,
                })
            }
            // A linear scan whose residual stacks reference loop-varying residuals moves those stacks into operand
            // position: each referenced stack becomes one extra scanned input, the body gains one trailing lane
            // input per moved stack (the stack type minus its leading length axis), and the body's scan-local
            // references to moved stacks are rewritten into operand form against those inputs. Constant stacks stay
            // factor payloads, with the surviving body references re-indexed against the compacted constant-only
            // stack list.
            Self::Scan(operation)
                if operation.captures().iter().any(|stack| matches!(stack, ValueOrCapture::Capture { .. })) =>
            {
                let residual_stacks = operation.captures();
                let mut dispositions = Vec::with_capacity(residual_stacks.len());
                let mut lane_types = Vec::new();
                let mut moved_stack_atoms = Vec::new();
                let mut surviving_stacks = Vec::new();
                for stack in residual_stacks {
                    match stack {
                        ValueOrCapture::Capture { index, r#type } => {
                            dispositions.push(Some(NestedResidualDisposition::Operand(lane_types.len())));
                            lane_types.push(r#type.without_dimension(0)?.0);
                            moved_stack_atoms.push(resolve_residual_atom(*index)?);
                        }
                        constant_stack => {
                            dispositions.push(Some(NestedResidualDisposition::Factor(surviving_stacks.len())));
                            surviving_stacks.push(constant_stack.clone());
                        }
                    }
                }
                let body = defactorize_nested_linear_program(
                    operation.body(),
                    dispositions.as_slice(),
                    lane_types.as_slice(),
                )?;
                inputs.extend(moved_stack_atoms);
                let scan = ScanOperation::<ArrayType, _, _>::new(body, operation.carry_count(), operation.length())?
                    .with_reverse(operation.reverse())
                    .with_unroll(operation.unroll())?
                    .with_captures(surviving_stacks);
                Ok(DefactorizedOperation::Operation { operation: LinearArrayOperation::Scan(Box::new(scan)), inputs })
            }
            operation => {
                // Closed constant factors and factor-free operations pass through unchanged. Residual references
                // hidden in payloads this rule cannot splice operands into — custom VJP call residuals, factor-form
                // while payloads, and condition branches whose predicate factor is a closed constant (defactorization
                // stages exactly one instruction, so a constant predicate cannot be materialized as the operand the
                // rewritten branches would require) — are rejected with the offending operation's name.
                let mut references_residual = false;
                operation.try_map_captures(&mut |factor: &ValueOrCapture<ArrayType, R>| {
                    if matches!(factor, ValueOrCapture::Capture { .. }) {
                        references_residual = true;
                    }
                    Ok(factor.clone())
                })?;
                if references_residual {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "jvp of a while loop whose body pushforward stages {} over a loop-varying residual \
                             reference is not supported",
                            operation.name(),
                        ),
                    });
                }
                Ok(DefactorizedOperation::Operation { operation: operation.clone(), inputs })
            }
        }
    }

    fn linear_while_operation(
        condition: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        body: Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
    ) -> Result<Self, TypeError> {
        Ok(LinearArrayOperation::While(Box::new(WhileOperation::new(condition, body)?)))
    }
}

// TODO(eaplatanios): Can we get rid of this similar to what we did for some of the scan-related functionality?
impl<V, C, P> InterpretableProgramOperation<ArrayType, V> for LinearArrayOperation<V, C, V, P>
where
    V: Value<ArrayType>
        + Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + SupportsConstantOperations<ArrayType>
        + Transpose
        + SupportsManipulationOperations
        + Select<Condition = V>
        + BooleanLike
        + TransferToMemory,
    C: Value<ArrayType>,
    V::InterpretationContext: Context<Type = ArrayType, Constant = C, Value = V>
        + Zero<ArrayType, V>
        + One<ArrayType, V>
        + Fill<ArrayType, f64, V>,
    ScaleOperation<ArrayType, V, Input>: InterpretableOperation<ArrayType, V>,
    ConstantOperation<ArrayType, V, Input>: InterpretableOperation<ArrayType, V>,
    LeftDotOperation<V, Input>: InterpretableOperation<ArrayType, V>,
    RightDotOperation<V, Input>: InterpretableOperation<ArrayType, V>,
    V: CustomVjpResidual<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
    P: Clone + InterpretableOperation<ArrayType, V>,
{
    fn interpret_program(
        context: &V::InterpretationContext,
        program: &Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        input: Vec<V>,
    ) -> Result<Vec<V>, ProgramError> {
        program.interpret_with(
            input,
            |_, constant| Ok(constant.clone()),
            |instruction, instruction_inputs| match instruction.operation() {
                Self::CustomVjpCall(operation) => operation.interpret(context, instruction_inputs),
                Self::TransferToMemory(operation) => operation.interpret(context, instruction_inputs),
                Self::Zero(operation) => operation.interpret(context, instruction_inputs),
                Self::One(operation) => operation.interpret(context, instruction_inputs),
                Self::Constant(operation) => operation.interpret(context, instruction_inputs),
                Self::Fill(operation) => operation.interpret(context, instruction_inputs),
                Self::ZeroLike(operation) => operation.interpret(context, instruction_inputs),
                Self::OneLike(operation) => operation.interpret(context, instruction_inputs),
                Self::Add(operation) => operation.interpret(context, instruction_inputs),
                Self::Sub(operation) => operation.interpret(context, instruction_inputs),
                Self::Mul(operation) => operation.interpret(context, instruction_inputs),
                Self::Neg(operation) => operation.interpret(context, instruction_inputs),
                Self::Transpose(operation) => operation.interpret(context, instruction_inputs),
                Self::Scale(operation) => operation.interpret(context, instruction_inputs),
                Self::LeftDot(operation) => operation.interpret(context, instruction_inputs),
                Self::RightDot(operation) => operation.interpret(context, instruction_inputs),
                Self::Reshape(operation) => operation.interpret(context, instruction_inputs),
                Self::Reshard(operation) => operation.interpret(context, instruction_inputs),
                Self::ShardingConstraint(operation) => operation.interpret(context, instruction_inputs),
                Self::Broadcast(operation) => operation.interpret(context, instruction_inputs),
                Self::Slice(operation) => operation.interpret(context, instruction_inputs),
                Self::UpdateSlice(operation) => operation.interpret(context, instruction_inputs),
                Self::DynamicSlice(operation) => operation.interpret(context, instruction_inputs),
                Self::DynamicUpdateSlice(operation) => operation.interpret(context, instruction_inputs),
                Self::Gather(operation) => operation.interpret(context, instruction_inputs),
                Self::ScatterAdd(operation) => operation.interpret(context, instruction_inputs),
                Self::Pad(operation) => operation.interpret(context, instruction_inputs),
                Self::Concatenate(operation) => operation.interpret(context, instruction_inputs),
                Self::Reduce(operation) => operation.interpret(context, instruction_inputs),
                Self::Select(operation) => operation.interpret(context, instruction_inputs),
                Self::Residual(operation) => operation.interpret(context, instruction_inputs),
                Self::Recompute(operation) => operation.interpret(context, instruction_inputs),
                Self::Condition(operation) => {
                    let input_types =
                        instruction_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                    operation.infer_output_types(input_types.as_slice())?;
                    let branch = if operation.predicate().residual_value()?.boolean()? {
                        operation.true_branch()
                    } else {
                        operation.false_branch()
                    };
                    Self::interpret_program(context, branch, instruction_inputs.to_vec())
                }
                Self::OperandCondition(operation) => {
                    let input_types =
                        instruction_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                    operation.infer_output_types(input_types.as_slice())?;
                    let branch = if instruction_inputs[0].boolean()? {
                        operation.true_branch()
                    } else {
                        operation.false_branch()
                    };
                    Self::interpret_program(context, branch, instruction_inputs[1..].to_vec())
                }
                Self::While(operation) => {
                    let input_types =
                        instruction_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                    operation.infer_output_types(input_types.as_slice())?;
                    let mut state = instruction_inputs.to_vec();
                    let mut completed_iterations = 0;
                    loop {
                        if operation.iteration_bound().is_some_and(|bound| completed_iterations >= bound) {
                            break Ok(state);
                        }
                        let condition_outputs = Self::interpret_program(context, operation.condition(), state.clone())?;
                        check_count!("output", condition_outputs, 1, ProgramError);
                        if !condition_outputs[0].boolean()? {
                            break Ok(state);
                        }
                        state = Self::interpret_program(context, operation.body(), state)?;
                        check_count!("output", state, operation.state_types().len(), ProgramError);
                        completed_iterations += 1;
                    }
                }
                Self::Scan(operation) => {
                    let input_types =
                        instruction_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
                    operation.infer_output_types(input_types.as_slice())?;
                    let stack_values = operation
                        .captures()
                        .iter()
                        .map(CustomVjpResidual::residual_value)
                        .collect::<Result<Vec<_>, _>>()?;
                    let body = operation.body();
                    let carry_count = operation.carry_count();
                    let y_slice_types = body.output_types().split_off(carry_count);
                    interpret_scan_lanes(
                        carry_count,
                        operation.length(),
                        operation.reverse(),
                        y_slice_types.as_slice(),
                        instruction_inputs,
                        |stacked_type| context.zero(stacked_type),
                        |lane, lane_inputs| {
                            let lane_residuals = stack_values
                                .iter()
                                .map(|stack| read_scan_lane(stack, lane))
                                .collect::<Result<Vec<_>, _>>()?;
                            let lane_body = body.map_operations(|operation| {
                                operation.try_map_captures(&mut |capture: &ValueOrCapture<ArrayType, V>| {
                                    capture.instantiate(lane_residuals.as_slice())
                                })
                            })?;
                            Self::interpret_program(context, &lane_body, lane_inputs)
                        },
                    )
                }
            },
        )
    }
}

/// Transposes a captured-condition select (the `Select` variant of [`LinearArrayOperation`] and the scalar
/// [`LinearSelectOperation`](crate::tracing_v2::operations::select::LinearSelectOperation)).
///
/// The forward linear map is `(t, f) ↦ select(condition, t, f)`. Its transpose routes the output cotangent into the
/// branch that the condition selected: the `on_true` cotangent is `select(condition, cotangent, 0)` and the
/// `on_false` cotangent is `select(condition, 0, cotangent)`. The zero operand is staged as a typed `Zero` operation
/// via [`stage_cotangent`](crate::tracing_v2::operations::control_flow::stage_cotangent), and `make_operation`
/// rebuilds the captured-condition select for staging into the transpose builder.
pub(crate) fn transpose_captured_condition_select<'transpose, T, V, P, MakeOperationFn>(
    make_operation: MakeOperationFn,
    context: &mut AbstractTracingContext<'transpose, T, V, P>,
    input_types: &[&T],
    output_cotangents: &[Cotangent<'transpose, T, V, P>],
) -> Result<Vec<Cotangent<'transpose, T, V, P>>, ProgramError>
where
    T: Type,
    V: Value<T>,
    P: Operation<T> + From<ZeroOperation<T>>,
    MakeOperationFn: Fn() -> P,
{
    check_count!("input", input_types, 2, ProgramError);
    check_count!("output", output_cotangents, 1, ProgramError);
    match &output_cotangents[0] {
        Cotangent::Zero => Ok(vec![Cotangent::Zero, Cotangent::Zero]),
        Cotangent::Staged(cotangent) => {
            let zero =
                crate::tracing_v2::operations::control_flow::stage_cotangent(context, &Cotangent::Zero, input_types[0]);
            let on_true = context.stage_operation(make_operation(), &[cotangent.clone(), zero.clone()])?;
            check_count!("output", on_true, 1, ProgramError);
            let on_false = context.stage_operation(make_operation(), &[zero, cotangent.clone()])?;
            check_count!("output", on_false, 1, ProgramError);
            Ok(vec![
                Cotangent::Staged(on_true.into_iter().next().unwrap()),
                Cotangent::Staged(on_false.into_iter().next().unwrap()),
            ])
        }
    }
}

/// Builds the scatter-add operation that is the transpose dual of a captured-index gather. The forward gather
/// `t ↦ gather(t, indices, ...)` has adjoint `cotangent ↦ scatter_add(zeros, indices, cotangent, ...)`, so the scatter
/// dimension numbers mirror the gather's axis-for-axis (offset↔update-window, collapsed↔inserted-window,
/// `start_index_map`↔`scatter_dimensions_to_operand_dimensions`, the batching pairs carried over), the combiner is
/// [`ScatterReductionKind::Add`], and the mode/flags carry through unchanged. The scatter writes into a zero operand of
/// the gather operand's type, so no `output_sharding` is requested (that zero operand already carries it).
pub(crate) fn gather_to_scatter_operation(operation: &GatherOperation) -> ScatterOperation {
    let dimensions = operation.dimensions();
    let scatter_dimensions = ScatterDimensionNumbers::new(
        dimensions.offset_dimensions().to_vec(),
        dimensions.collapsed_slice_dimensions().to_vec(),
        dimensions.start_index_map().to_vec(),
    )
    .with_batching_dimensions(
        dimensions.operand_batching_dimensions().to_vec(),
        dimensions.start_indices_batching_dimensions().to_vec(),
    );
    ScatterOperation::new(scatter_dimensions, ScatterReductionKind::Add)
        .with_mode(operation.mode())
        .with_indices_are_sorted(operation.indices_are_sorted())
        .with_unique_indices(operation.unique_indices())
}

/// Clones one factor payload unchanged; used as a stable `fn`-pointer identity mapping by the scan-body
/// traversal of [`map_linear_array_operation_factors`].
fn clone_factor<F: Clone>(factor: &F) -> Result<F, ProgramError> {
    Ok(factor.clone())
}

/// Shared payload-mapping core behind [`CaptureParameterizedOperation::try_map_captures`] for
/// [`LinearArrayOperation`].
fn map_linear_array_operation_factors<V, C, F, MappedFactor, P, MapFactorFn>(
    operation: &LinearArrayOperation<V, C, F, P>,
    map_factor: &mut MapFactorFn,
) -> Result<LinearArrayOperation<V, C, MappedFactor, P>, ProgramError>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    F: Value<ArrayType>,
    MappedFactor: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
    MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
{
    {
        match operation {
            LinearArrayOperation::CustomVjpCall(call) => {
                Ok(LinearArrayOperation::CustomVjpCall(Box::new(call.map_captures(map_factor)?)))
            }
            LinearArrayOperation::Zero(zero) => Ok(LinearArrayOperation::Zero(zero.clone())),
            LinearArrayOperation::One(one) => Ok(LinearArrayOperation::One(one.clone())),
            LinearArrayOperation::Constant(constant) => Ok(LinearArrayOperation::Constant(constant.clone())),
            LinearArrayOperation::Fill(fill) => Ok(LinearArrayOperation::Fill(fill.clone())),
            LinearArrayOperation::ZeroLike(operation) => Ok(LinearArrayOperation::ZeroLike(operation.clone())),
            LinearArrayOperation::OneLike(operation) => Ok(LinearArrayOperation::OneLike(operation.clone())),
            LinearArrayOperation::Add(operation) => Ok(LinearArrayOperation::Add(operation.clone())),
            LinearArrayOperation::Sub(operation) => Ok(LinearArrayOperation::Sub(operation.clone())),
            LinearArrayOperation::Neg(operation) => Ok(LinearArrayOperation::Neg(operation.clone())),
            LinearArrayOperation::Mul(operation) => Ok(LinearArrayOperation::Mul(operation.clone())),
            LinearArrayOperation::TransferToMemory(operation) => {
                Ok(LinearArrayOperation::TransferToMemory(operation.clone()))
            }
            LinearArrayOperation::Transpose(operation) => Ok(LinearArrayOperation::Transpose(operation.clone())),
            LinearArrayOperation::Scale(operation) => {
                Ok(LinearArrayOperation::Scale(ScaleOperation::new(map_factor(operation.factor())?)))
            }
            LinearArrayOperation::LeftDot(operation) => Ok(LinearArrayOperation::LeftDot(
                LeftDotOperation::new(map_factor(operation.factor())?, operation.dimensions().clone())
                    .with_output_sharding(operation.output_sharding().cloned()),
            )),
            LinearArrayOperation::RightDot(operation) => Ok(LinearArrayOperation::RightDot(
                RightDotOperation::new(map_factor(operation.factor())?, operation.dimensions().clone())
                    .with_output_sharding(operation.output_sharding().cloned()),
            )),
            LinearArrayOperation::Reshape(operation) => Ok(LinearArrayOperation::Reshape(operation.clone())),
            LinearArrayOperation::Reshard(operation) => Ok(LinearArrayOperation::Reshard(operation.clone())),
            LinearArrayOperation::ShardingConstraint(operation) => {
                Ok(LinearArrayOperation::ShardingConstraint(operation.clone()))
            }
            LinearArrayOperation::Broadcast(operation) => Ok(LinearArrayOperation::Broadcast(operation.clone())),
            LinearArrayOperation::Slice(operation) => Ok(LinearArrayOperation::Slice(operation.clone())),
            LinearArrayOperation::UpdateSlice(operation) => Ok(LinearArrayOperation::UpdateSlice(operation.clone())),
            LinearArrayOperation::Pad(operation) => Ok(LinearArrayOperation::Pad(operation.clone())),
            LinearArrayOperation::Concatenate(operation) => Ok(LinearArrayOperation::Concatenate(operation.clone())),
            LinearArrayOperation::DynamicSlice(operation) => {
                Ok(LinearArrayOperation::DynamicSlice(LinearDynamicSliceOperation::new(
                    operation.start_indices().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                    operation.sizes().to_vec(),
                )))
            }
            LinearArrayOperation::DynamicUpdateSlice(operation) => {
                Ok(LinearArrayOperation::DynamicUpdateSlice(LinearDynamicUpdateSliceOperation::new(
                    operation.start_indices().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?,
                )))
            }
            LinearArrayOperation::Gather(operation) => Ok(LinearArrayOperation::Gather(LinearGatherOperation::new(
                operation.operation().clone(),
                map_factor(operation.indices())?,
            ))),
            LinearArrayOperation::ScatterAdd(operation) => Ok(LinearArrayOperation::ScatterAdd(
                LinearScatterAddOperation::new(operation.operation().clone(), map_factor(operation.indices())?),
            )),
            LinearArrayOperation::Reduce(operation) => Ok(LinearArrayOperation::Reduce(operation.clone())),
            LinearArrayOperation::Select(operation) => {
                Ok(LinearArrayOperation::Select(LinearSelectOperation::new(map_factor(operation.condition())?)))
            }
            LinearArrayOperation::Residual(operation) => {
                Ok(LinearArrayOperation::Residual(MaterializeCaptureOperation::new(map_factor(operation.capture())?)))
            }
            LinearArrayOperation::Recompute(operation) => Ok(LinearArrayOperation::Recompute(operation.clone())),
            LinearArrayOperation::Condition(operation) => {
                Ok(LinearArrayOperation::Condition(ConditionOperation::new_captured(
                    map_factor(operation.predicate())?,
                    operation.true_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                    })?,
                    operation.false_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                    })?,
                )?))
            }
            // Operand-form condition branches carry only closed constant factors after defactorization, but the
            // traversal stays total over them like the factor-form variant's.
            LinearArrayOperation::OperandCondition(operation) => {
                Ok(LinearArrayOperation::OperandCondition(LinearOperandConditionOperation::new(
                    Box::new(operation.true_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                    })?),
                    Box::new(operation.false_branch().map_operations(|operation| {
                        map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                    })?),
                )))
            }
            LinearArrayOperation::While(while_operation) => {
                let condition = while_operation.condition().map_operations(|operation| {
                    map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                })?;
                let body = while_operation.body().map_operations(|operation| {
                    map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, map_factor)
                })?;
                Ok(LinearArrayOperation::While(Box::new(
                    WhileOperation::new(condition, body)?.with_iteration_bound(while_operation.iteration_bound())?,
                )))
            }
            // The scan body's factor space is scan-local (references index `residual_stacks` per lane), so enclosing
            // factor passes map only the stack payloads and clone the body-internal factors unchanged.
            LinearArrayOperation::Scan(operation) => {
                // The factor-cloning function is passed as a `fn` pointer (not a closure) so the recursive
                // monomorphization below reaches a fixed point: nested scans reuse the exact same mapper
                // instantiation instead of minting a fresh closure type per level.
                let mut clone_scan_local_factor = clone_factor::<ValueOrCapture<ArrayType, V>>
                    as fn(&ValueOrCapture<ArrayType, V>) -> Result<ValueOrCapture<ArrayType, V>, ProgramError>;
                let body = operation.body().map_operations(|operation| {
                    map_linear_array_operation_factors::<_, _, _, _, _, _>(operation, &mut clone_scan_local_factor)
                })?;
                let scan = ScanOperation::<ArrayType, _, _>::new(body, operation.carry_count(), operation.length())?
                    .with_reverse(operation.reverse())
                    .with_unroll(operation.unroll())?
                    .with_captures(operation.captures().iter().map(&mut *map_factor).collect::<Result<Vec<_>, _>>()?);
                Ok(LinearArrayOperation::Scan(Box::new(scan)))
            }
        }
    }
}

// TODO(eaplatanios): Can we get rid of this similar to what we did for some of the scan-related functionality?
impl<V, C, F, P> CaptureParameterizedOperation<ArrayType, F> for LinearArrayOperation<V, C, F, P>
where
    V: Value<ArrayType>,
    C: Value<ArrayType>,
    F: Value<ArrayType>,
    P: Clone + Operation<ArrayType>,
{
    type WithCapture<MappedFactor: Value<ArrayType>> = LinearArrayOperation<V, C, MappedFactor, P>;

    fn try_map_captures<MappedFactor: Value<ArrayType>, MapFactorFn>(
        &self,
        map_factor: &mut MapFactorFn,
    ) -> Result<Self::WithCapture<MappedFactor>, ProgramError>
    where
        MapFactorFn: FnMut(&F) -> Result<MappedFactor, ProgramError>,
    {
        map_linear_array_operation_factors::<_, _, _, _, _, _>(self, map_factor)
    }
}

/// [`InterpretableOperation`] for [`ArrayOperation`] requires the full union of value capabilities exercised by the
/// closed default ordinary operation enum.
///
/// The value-side bound list is expressed via the orthogonal capability bundles defined in [`super::bounds`] (one
/// per operation category — arithmetic, trigonometric, constants, manipulation, comparison) plus the few singleton
/// traits ([`DotOps`], [`Select`], [`BooleanLike`]) that the dispatcher requires directly. Context-side nullary
/// capabilities ([`Zero`], [`One`], [`Fill`]) are listed on `V::InterpretationContext`. Each impl site composes only
/// the categories it actually exercises, so downstream consumers never depend on a single monolithic value-bundle
/// trait.
impl<V> InterpretableOperation<ArrayType, V> for ArrayOperation<V>
where
    V: Value<ArrayType, InterpretationContext = EagerContext<ArrayType, V>>
        + Parameter
        + SupportsArithmeticOperations
        + SupportsTrigonometricOperations
        + SupportsConstantOperations<ArrayType>
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = V>
        + BooleanLike
        + TransferToMemory,
    V::InterpretationContext: Context<Type = ArrayType, Constant = V, Value = V>
        + Zero<ArrayType, V>
        + One<ArrayType, V>
        + Fill<ArrayType, f64, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
{
    fn interpret(
        &self,
        context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomJvp(operation) => operation.interpret(context, inputs),
            Self::CustomVjp(operation) => operation.interpret(context, inputs),
            Self::Zero(operation) => operation.interpret(context, inputs),
            Self::One(operation) => operation.interpret(context, inputs),
            Self::Constant(operation) => operation.interpret(context, inputs),
            Self::Fill(operation) => operation.interpret(context, inputs),
            Self::ZeroLike(operation) => operation.interpret(context, inputs),
            Self::OneLike(operation) => operation.interpret(context, inputs),
            Self::Add(operation) => operation.interpret(context, inputs),
            Self::Sub(operation) => operation.interpret(context, inputs),
            Self::Mul(operation) => operation.interpret(context, inputs),
            Self::Div(operation) => operation.interpret(context, inputs),
            Self::Neg(operation) => operation.interpret(context, inputs),
            Self::Sin(operation) => operation.interpret(context, inputs),
            Self::Cos(operation) => operation.interpret(context, inputs),
            Self::StopGradient(operation) => operation.interpret(context, inputs),
            Self::RematerializationName(operation) => operation.interpret(context, inputs),
            Self::TransferToMemory(operation) => operation.interpret(context, inputs),
            Self::Dot(operation) => operation.interpret(context, inputs),
            Self::Transpose(operation) => operation.interpret(context, inputs),
            Self::Scale(operation) => operation.interpret(context, inputs),
            Self::Reshape(operation) => operation.interpret(context, inputs),
            Self::Reshard(operation) => operation.interpret(context, inputs),
            Self::ShardingConstraint(operation) => operation.interpret(context, inputs),
            Self::Broadcast(operation) => operation.interpret(context, inputs),
            Self::Slice(operation) => operation.interpret(context, inputs),
            Self::UpdateSlice(operation) => operation.interpret(context, inputs),
            Self::DynamicSlice(operation) => operation.interpret(context, inputs),
            Self::DynamicUpdateSlice(operation) => operation.interpret(context, inputs),
            Self::Pad(operation) => operation.interpret(context, inputs),
            Self::Concatenate(operation) => operation.interpret(context, inputs),
            Self::Gather(operation) => operation.interpret(context, inputs),
            Self::Scatter(operation) => operation.interpret(context, inputs),
            Self::Reduce(operation) => operation.interpret(context, inputs),
            Self::Compare(operation) => operation.interpret(context, inputs),
            Self::Not(operation) => operation.interpret(context, inputs),
            Self::And(operation) => operation.interpret(context, inputs),
            Self::Or(operation) => operation.interpret(context, inputs),
            Self::Xor(operation) => operation.interpret(context, inputs),
            Self::Collective(operation) => operation.interpret(context, inputs),
            Self::Select(operation) => operation.interpret(context, inputs),
            Self::Condition(condition) => condition.interpret(context, inputs),
            Self::While(while_operation) => while_operation.interpret(context, inputs),
            Self::Scan(scan) => scan.interpret(context, inputs),
        }
    }
}

impl<S, C> InterpretableOperation<ArrayType, Tracer<S>> for ArrayOperation<C>
where
    S: StagingContext<Type = ArrayType>,
    C: Value<ArrayType>,
    Self: Clone + Into<S::Operation>,
{
    fn interpret(&self, context: &S, inputs: &[Tracer<S>]) -> Result<Vec<Tracer<S>>, ProgramError> {
        context.stage_operation(self.clone(), inputs)
    }
}

impl<V, C, F: Value<ArrayType>, P> InterpretableOperation<ArrayType, V> for LinearArrayOperation<V, C, F, P>
where
    V: Value<ArrayType>
        + Parameter
        + Add<Output = V>
        + Sub<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + SupportsConstantOperations<ArrayType>
        + Transpose
        + SupportsManipulationOperations
        + Select<Condition = V>
        + BooleanLike
        + TransferToMemory,
    C: Value<ArrayType>,
    V::InterpretationContext: Context<Type = ArrayType, Constant = C, Value = V>
        + Zero<ArrayType, V>
        + One<ArrayType, V>
        + Fill<ArrayType, f64, V>
        + Scale<ArrayType, V, V, Input>
        + LeftDot<V, V, Input>
        + RightDot<V, V, Input>
        + LinearScanInterpretation<V, F, LinearArrayOperation<V, C, ValueOrCapture<ArrayType, V>, P>>,
    ScaleOperation<ArrayType, F, Input>: InterpretableOperation<ArrayType, V>,
    ConstantOperation<ArrayType, V, Input>: InterpretableOperation<ArrayType, V>,
    LeftDotOperation<F, Input>: InterpretableOperation<ArrayType, V>,
    RightDotOperation<F, Input>: InterpretableOperation<ArrayType, V>,
    F: CustomVjpResidual<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: std::fmt::Debug + PartialEq>,
    P: Clone + InterpretableOperation<ArrayType, V>,
{
    fn interpret(
        &self,
        context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        match self {
            Self::CustomVjpCall(operation) => operation.interpret(context, inputs),
            Self::TransferToMemory(operation) => operation.interpret(context, inputs),
            Self::Zero(operation) => operation.interpret(context, inputs),
            Self::One(operation) => operation.interpret(context, inputs),
            Self::Constant(operation) => operation.interpret(context, inputs),
            Self::Fill(operation) => operation.interpret(context, inputs),
            Self::ZeroLike(operation) => operation.interpret(context, inputs),
            Self::OneLike(operation) => operation.interpret(context, inputs),
            Self::Add(operation) => operation.interpret(context, inputs),
            Self::Sub(operation) => operation.interpret(context, inputs),
            Self::Mul(operation) => operation.interpret(context, inputs),
            Self::Neg(operation) => operation.interpret(context, inputs),
            Self::Transpose(operation) => operation.interpret(context, inputs),
            Self::Scale(operation) => operation.interpret(context, inputs),
            Self::LeftDot(operation) => operation.interpret(context, inputs),
            Self::RightDot(operation) => operation.interpret(context, inputs),
            Self::Reshape(operation) => operation.interpret(context, inputs),
            Self::Reshard(operation) => operation.interpret(context, inputs),
            Self::ShardingConstraint(operation) => operation.interpret(context, inputs),
            Self::Broadcast(operation) => operation.interpret(context, inputs),
            Self::Slice(operation) => operation.interpret(context, inputs),
            Self::UpdateSlice(operation) => operation.interpret(context, inputs),
            Self::DynamicSlice(operation) => operation.interpret(context, inputs),
            Self::DynamicUpdateSlice(operation) => operation.interpret(context, inputs),
            Self::Gather(operation) => operation.interpret(context, inputs),
            Self::ScatterAdd(operation) => operation.interpret(context, inputs),
            Self::Pad(operation) => operation.interpret(context, inputs),
            Self::Concatenate(operation) => operation.interpret(context, inputs),
            Self::Reduce(operation) => operation.interpret(context, inputs),
            Self::Select(operation) => operation.interpret(context, inputs),
            Self::Residual(operation) => operation.interpret(context, inputs),
            Self::Recompute(operation) => operation.interpret(context, inputs),
            Self::Condition(operation) => operation.interpret(context, inputs),
            Self::OperandCondition(operation) => operation.interpret(context, inputs),
            Self::While(operation) => operation.interpret(context, inputs),
            Self::Scan(operation) => context.interpret_linear_scan(operation, inputs),
        }
    }
}

/// Builds the common error for zero-input operation enum variants that must be handled by the staging path.
fn missing_zero_input_batch_rule(operation_enum: &str, kind: &str) -> ProgramError {
    BatchingError::UnsupportedOperation {
        message: format!(
            "{operation_enum}::{kind}: zero-input operations are lane-uniform by construction — stage them through the \
             active context, which handles the lane-uniform short-circuit, instead of invoking `batch` directly",
        ),
    }
    .into()
}

/// Dispatches non-control-flow [`ArrayOperation`] variants to their primitive batching rules.
///
/// Higher-order variants are intentionally returned as `None` so concrete impls can handle them with their specialized
/// recursive bounds instead of forcing the trait solver through one fully generic recursive operation impl.
fn batch_array_non_control_operation<F, V>(
    operation: &ArrayOperation<F>,
    context: &V::InterpretationContext,
    inputs: &[ArrayBatch<V>],
) -> Result<Option<Vec<ArrayBatch<V>>>, ProgramError>
where
    F: Value<ArrayType>,
    V: Value<ArrayType>
        + SupportsArithmeticOperations<F>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = V>,
    V::InterpretationContext: Scale<ArrayType, V, F>,
{
    let outputs = match operation {
        ArrayOperation::Add(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Sub(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Mul(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Div(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Neg(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Sin(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Cos(operation) => operation.batch(context, inputs)?,
        ArrayOperation::StopGradient(operation) => operation.batch(context, inputs)?,
        ArrayOperation::RematerializationName(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Select(operation) => operation.batch(context, inputs)?,
        ArrayOperation::ZeroLike(operation) => operation.batch(context, inputs)?,
        ArrayOperation::OneLike(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Scale(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Dot(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Transpose(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Reshape(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Reshard(operation) => operation.batch(context, inputs)?,
        ArrayOperation::ShardingConstraint(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Broadcast(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Slice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::UpdateSlice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::DynamicSlice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::DynamicUpdateSlice(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Pad(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Concatenate(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Gather(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Scatter(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Reduce(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Compare(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Not(operation) => operation.batch(context, inputs)?,
        ArrayOperation::And(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Or(operation) => operation.batch(context, inputs)?,
        ArrayOperation::Xor(operation) => operation.batch(context, inputs)?,
        ArrayOperation::TransferToMemory(_)
        | ArrayOperation::Collective(_)
        | ArrayOperation::Condition(_)
        | ArrayOperation::While(_)
        | ArrayOperation::Scan(_)
        | ArrayOperation::CustomJvp(_)
        | ArrayOperation::CustomVjp(_) => return Ok(None),
        ArrayOperation::Zero(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Zero")),
        ArrayOperation::One(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "One")),
        ArrayOperation::Constant(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Constant")),
        ArrayOperation::Fill(_) => return Err(missing_zero_input_batch_rule("ArrayOperation", "Fill")),
    };
    Ok(Some(outputs))
}

/// Blanket active batching impl for the [`ArrayOperation`] sum type over a staged tracer context: each non-control
/// variant delegates to its backing operation's batching rule (shared with the eager impl through
/// [`batch_array_non_control_operation`]), while the lane-uniform memory transfer, the named-axis collective, and the
/// higher-order control-flow variants are handled by their specialized recursive rules.
impl<C, V> BatchableOperation<Tracer<C>, BatchingContext<C>> for ArrayOperation<V>
where
    C: StagingContext<Type = ArrayType, Constant = V, Operation = ArrayOperation<V>>,
    V: Value<ArrayType> + BooleanLike,
    C::Operation: From<CollectiveOperation> + From<FillOperation<ArrayType, f64>>,
    Tracer<C>: SupportsArithmeticOperations<V>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = Tracer<C>>
        + BooleanLike
        + Broadcast
        + Transpose,
    Vec<Tracer<C>>: Parameterized<Tracer<C>, To<Tracer<C>> = Vec<Tracer<C>>, ParameterStructure: Debug + PartialEq>,
    Self: BatchableProgramOperation<V>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        if let Some(outputs) = batch_array_non_control_operation(self, context.parent_context(), inputs)? {
            return Ok(outputs);
        }
        match self {
            // Memory placement is lane-uniform: the same transfer applies to every lane, so it is staged unchanged on
            // the physical batched value in the parent context and the lane axis is preserved.
            Self::TransferToMemory(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let tracer = inputs[0].value().transfer_to_memory(operation.destination());
                let physical_type = tracer.r#type().into_owned();
                Ok(vec![ArrayBatch::new(physical_type, tracer, inputs[0].batch_axis())?])
            }
            // The staged collective rule owns named-axis resolution: it consumes the lane axis when this context's
            // axis name matches and forwards the collective to the parent context otherwise.
            Self::Collective(operation) => operation.batch(context, inputs),
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::Scan(scan) => scan.batch(context, inputs),
            Self::CustomJvp(operation) => operation.batch(context, inputs),
            Self::CustomVjp(operation) => operation.batch(context, inputs),
            _ => unreachable!("non-control-flow ArrayOperation variants are handled above"),
        }
    }
}

/// Blanket value-level batching impl for the [`ArrayOperation`] sum type.
impl<V> BatchableOperation<V, EagerContext<ArrayType, V, ArrayOperation<V>>> for ArrayOperation<V>
where
    V::InterpretationContext: Default,
    V::InterpretationContext: Scale<ArrayType, V, V>,
    V: Value<ArrayType>
        + SupportsArithmeticOperations
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = V>
        + BooleanLike,
    EagerContext<ArrayType, V, ArrayOperation<V>>: Zero<ArrayType, V> + Fill<ArrayType, f64, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, ArrayOperation<V>>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let interpretation_context = V::InterpretationContext::default();
        if let Some(outputs) = batch_array_non_control_operation(self, &interpretation_context, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::TransferToMemory(_) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(inputs.to_vec())
            }
            Self::Collective(operation) => operation.batch(context, inputs),
            Self::Condition(condition) => condition.batch(context, inputs),
            Self::While(while_operation) => while_operation.batch(context, inputs),
            Self::Scan(scan) => scan.batch(context, inputs),
            Self::CustomJvp(operation) => operation.batch(context, inputs),
            Self::CustomVjp(operation) => operation.batch(context, inputs),
            _ => unreachable!("non-control-flow ArrayOperation variants are handled above"),
        }
    }
}

/// Blanket active batching impl for the [`ArrayOperation`] sum type.
///
/// The `Operation = Self` projection equality and the
/// [`BatchableProgramOperation`](crate::tracing_v2::batching::BatchableProgramOperation) / lane-alignment bounds exist
/// for the custom-derivative arms: their re-wrapping batch rules batch the captured programs and stage a new
/// custom-derivative call into the parent context, which is only expressible when the staged operation type is this
/// enum itself. Both extra bounds are leaf obligations (a structural type equality and a closed-enum capability
/// whose impl carries no batching-context obligations of its own), so instantiating this impl never recurses into
/// another batching-context obligation.
/// Program-level batching for the [`ArrayOperation`] sum type, backing the re-wrapping `batch` rules of
/// [`CustomJvpOperation`] and [`CustomVjpOperation`]; see
/// [`BatchableProgramOperation`](crate::tracing_v2::batching::BatchableProgramOperation).
///
/// The where clauses here are deliberately the *leaf* closure of what `batch_program::<V, Self>` needs — the
/// blanket traced batching impl's bounds instantiated at [`ProgramBatchingContext`] — rather than the
/// `Self: BatchableOperation<..>` bound itself. Spelling out the leaves keeps instantiating this impl free of
/// batching-context obligations, which is what lets the traced batching impl require
/// `Self: BatchableProgramOperation<..>` without sending the trait solver into an unbounded
/// batching-context recursion.
impl<V> BatchableProgramOperation<V> for ArrayOperation<V>
where
    V: Value<ArrayType> + BooleanLike + 'static,
    Tracer<ProgramBatchingContext<V, Self>>: SupportsArithmeticOperations<V>
        + SupportsTrigonometricOperations
        + ZeroLike
        + OneLike
        + DotOps
        + SupportsManipulationOperations
        + SupportsComparisonOperations
        + Select<Condition = Tracer<ProgramBatchingContext<V, Self>>>
        + BooleanLike
        + Broadcast
        + Transpose,
    Vec<Tracer<ProgramBatchingContext<V, Self>>>: Parameterized<
            Tracer<ProgramBatchingContext<V, Self>>,
            To<Tracer<ProgramBatchingContext<V, Self>>> = Vec<Tracer<ProgramBatchingContext<V, Self>>>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    fn batch_program(
        program: &crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        axis_size: usize,
        input_batch_axes: &[Option<usize>],
        output_batch_axes: ProgramBatchingOutputAxes,
    ) -> Result<(crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>, Vec<Option<usize>>), ProgramError> {
        crate::tracing_v2::batching::batch_program::<V, Self>(program, axis_size, input_batch_axes, output_batch_axes)
    }
}

/// Nested symbolic linearization for the [`ArrayOperation`] sum type, backing the staged-condition JVP rule of
/// [`ConditionOperation`]; see [`LinearizableProgramOperation`](crate::tracing_v2::LinearizableProgramOperation).
///
/// The where clauses here are deliberately the *leaf* closure of what
/// [`linearize_program`](crate::tracing_v2::linearize_program)`::<E, Self>` needs — the generic JVP
/// dispatch impl's bounds instantiated at [`LinearizationContextOf`] — rather than the
/// `Self: DifferentiableOperation<LinearizationContextOf<E, Self>>` bound itself. Spelling out the leaves keeps
/// instantiating this impl free of derived-context differentiation obligations (the recursive obligation is
/// discharged once, as a definition-time body check), which is what lets the JVP dispatch impl require
/// `Self: LinearizableProgramOperation<E>` without sending the trait solver into an unbounded nested-context
/// recursion. The `WithCapture<V> = ..` equality pins the canonical linear operation as a fixed point of factor
/// reparameterization, which is what collapses `LinearizationContextOf<LinearizationContextOf<E, ..>, ..>`
/// to `LinearizationContextOf<E, ..>` and keeps the obligations finite for nested conditions.
impl<V, E> LinearizableProgramOperation<E> for ArrayOperation<V>
where
    V: Value<ArrayType>,
    E: DifferentiationContext<Type = ArrayType, Constant = V>,
    E::Tangent: Transpose + Broadcast + super::reduce::Reduce + Slice + Reshard + ConstrainSharding,
    E::LinearOperation<E::Tangent, V>:
        CaptureParameterizedOperation<ArrayType, V, WithCapture<V> = E::LinearOperation<E::Tangent, V>>,
    LinearOperationOf<LinearizationContextOf<E, Self>>: SupportsLinearArrayOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>
        + crate::tracing_v2::ResidualizedOperation<LinearizationContextOf<E, Self>>
        + From<ZeroOperation<ArrayType>>
        + From<
            CustomVjpCallOperation<
                ArrayType,
                V,
                Self,
                ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
            >,
        > + From<TransferToMemoryOperation>
        + From<ConcatenateOperation>
        + From<LinearSelectOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<LinearDynamicSliceOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<LinearDynamicUpdateSliceOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<LinearGatherOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<LinearScatterAddOperation<ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>>>
        + From<
            ConditionOperation<
                ArrayType,
                E::Tangent,
                LinearOperationOf<LinearizationContextOf<E, Self>>,
                ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
                Captured,
            >,
        > + SupportsLinearWhile<
            ArrayType,
            E::Tangent,
            ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
            Self,
        > + LinearScanOperation<ArrayType, E::Tangent, Tracer<LinearizationContextOf<E, Self>>>,
    LinearOperationOf<LinearizationContextOf<E, Self>>: CaptureParameterizedOperation<
            ArrayType,
            ValueOrCapture<ArrayType, Tracer<LinearizationContextOf<E, Self>>>,
            WithCapture<ValueOrCapture<ArrayType, E::Value>> = LinearOperationOf<E>,
        > + MaybeZeroOperation<ArrayType>,
{
    fn linearize_program(
        differentiable: &E,
        program: &crate::programs::Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
    ) -> Result<NestedLinearization<E, Self>, ProgramError> {
        crate::tracing_v2::differentiation::linearize_program(differentiable, program)
    }
}

/// Dispatches non-control-flow [`LinearArrayOperation`] variants to their primitive batching rules.
fn batch_linear_non_control_operation<F, C, V>(
    operation: &LinearArrayOperation<F, C, F, ArrayOperation<C>>,
    context: &V::InterpretationContext,
    inputs: &[ArrayBatch<V>],
) -> Result<Option<Vec<ArrayBatch<V>>>, ProgramError>
where
    F: Value<ArrayType>,
    C: Value<ArrayType>,
    V: Value<ArrayType>
        + SupportsLinearArithmeticOperations<F>
        + ZeroLike
        + OneLike
        + SupportsLinearAlgebraOperations<F>
        + SupportsManipulationOperations
        + BitAnd<Output = V>
        + Select<Condition = V>,
    V::InterpretationContext: Scale<ArrayType, V, F> + LeftDot<V, F, Captured> + RightDot<V, F, Captured>,
{
    let outputs = match operation {
        LinearArrayOperation::Add(_) => AddOperation.batch(context, inputs)?,
        LinearArrayOperation::Sub(_) => SubOperation.batch(context, inputs)?,
        LinearArrayOperation::Mul(_) => MulOperation.batch(context, inputs)?,
        LinearArrayOperation::Neg(_) => NegOperation.batch(context, inputs)?,
        LinearArrayOperation::ZeroLike(_) => ZeroLikeOperation.batch(context, inputs)?,
        LinearArrayOperation::OneLike(_) => OneLikeOperation.batch(context, inputs)?,
        LinearArrayOperation::Scale(operation) => {
            ScaleOperation::new(operation.factor().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::Transpose(operation) => {
            TransposeOperation::new(operation.permutation().to_vec()).batch(context, inputs)?
        }
        LinearArrayOperation::LeftDot(operation) => {
            LeftDotOperation::new(operation.factor().clone(), operation.dimensions().clone())
                .with_output_sharding(operation.output_sharding().cloned())
                .batch(context, inputs)?
        }
        LinearArrayOperation::RightDot(operation) => {
            RightDotOperation::new(operation.factor().clone(), operation.dimensions().clone())
                .with_output_sharding(operation.output_sharding().cloned())
                .batch(context, inputs)?
        }
        LinearArrayOperation::Reshape(operation) => {
            ReshapeOperation::new(operation.output_shape().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::Reshard(operation) => {
            ReshardOperation::new(operation.sharding().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::ShardingConstraint(operation) => {
            ShardingConstraintOperation::new(operation.sharding().clone()).batch(context, inputs)?
        }
        LinearArrayOperation::Broadcast(operation) => {
            BroadcastOperation::new(operation.output_type().clone(), operation.output_axes().to_vec())
                .batch(context, inputs)?
        }
        LinearArrayOperation::Reduce(operation) => ReduceOperation::new(operation.axes().to_vec(), operation.kind())
            .with_output_sharding(operation.output_sharding().cloned())
            .batch(context, inputs)?,
        LinearArrayOperation::Slice(operation) => {
            SliceOperation::new(operation.start_indices().to_vec(), operation.limit_indices().to_vec())
                .with_strides(operation.strides().to_vec())?
                .batch(context, inputs)?
        }
        LinearArrayOperation::UpdateSlice(operation) => {
            UpdateSliceOperation::new(operation.start_indices().to_vec()).batch(context, inputs)?
        }
        LinearArrayOperation::Pad(operation) => PadOperation::new(
            operation.edge_padding_low().to_vec(),
            operation.edge_padding_high().to_vec(),
            operation.interior_padding().to_vec(),
        )?
        .batch(context, inputs)?,
        LinearArrayOperation::Concatenate(operation) => {
            ConcatenateOperation::new(operation.axis()).batch(context, inputs)?
        }
        LinearArrayOperation::TransferToMemory(_)
        | LinearArrayOperation::DynamicSlice(_)
        | LinearArrayOperation::DynamicUpdateSlice(_)
        | LinearArrayOperation::Gather(_)
        | LinearArrayOperation::ScatterAdd(_)
        | LinearArrayOperation::Select(_)
        | LinearArrayOperation::Residual(_)
        | LinearArrayOperation::Recompute(_)
        | LinearArrayOperation::Condition(_)
        | LinearArrayOperation::OperandCondition(_)
        | LinearArrayOperation::While(_)
        | LinearArrayOperation::Scan(_)
        | LinearArrayOperation::CustomVjpCall(_) => {
            return Ok(None);
        }
        LinearArrayOperation::Zero(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Zero")),
        LinearArrayOperation::One(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "One")),
        LinearArrayOperation::Constant(_) => {
            return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Constant"));
        }
        LinearArrayOperation::Fill(_) => return Err(missing_zero_input_batch_rule("LinearArrayOperation", "Fill")),
    };
    Ok(Some(outputs))
}

/// Blanket value-level batching impl for the [`LinearArrayOperation`] sum type.
impl<V> BatchableOperation<V, EagerContext<ArrayType, V, LinearArrayOperation<V, V, V, ArrayOperation<V>>>>
    for LinearArrayOperation<V, V, V, ArrayOperation<V>>
where
    ArrayOperation<V>: BatchableOperation<V, EagerContext<ArrayType, V, ArrayOperation<V>>>,
    V::InterpretationContext: Default,
    V::InterpretationContext: Scale<ArrayType, V, V> + LeftDot<V, V, Captured> + RightDot<V, V, Captured>,
    V: Value<ArrayType>
        + SupportsLinearArithmeticOperations
        + ZeroLike
        + OneLike
        + SupportsLinearAlgebraOperations
        + SupportsManipulationOperations
        + BitAnd<Output = V>
        + Select<Condition = V>
        + BooleanLike,
    EagerContext<ArrayType, V, LinearArrayOperation<V, V, V, ArrayOperation<V>>>: Zero<ArrayType, V>,
    Vec<V>: Parameterized<V, To<V> = Vec<V>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &EagerContext<ArrayType, V, LinearArrayOperation<V, V, V, ArrayOperation<V>>>,
        inputs: &[ArrayBatch<V>],
    ) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        let interpretation_context = V::InterpretationContext::default();
        if let Some(outputs) = batch_linear_non_control_operation(self, &interpretation_context, inputs)? {
            return Ok(outputs);
        }
        match self {
            Self::TransferToMemory(_) => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(inputs.to_vec())
            }
            // The captured condition is lane-uniform: prepending it as an unbatched operand lets the elementwise
            // select batching rule broadcast it to the batched physical shape before selecting per lane.
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                SelectOperation.batch(
                    &interpretation_context,
                    &[ArrayBatch::unbatched(operation.condition().clone()), inputs[0].clone(), inputs[1].clone()],
                )
            }
            // The captured start indices are lane-uniform by construction: appending them as unbatched operands
            // lets the primal dynamic-slice batching rule lift the lane axis.
            Self::DynamicSlice(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs
                    .extend(operation.start_indices().iter().map(|index| ArrayBatch::unbatched(index.clone())));
                DynamicSliceOperation::new(operation.sizes().to_vec())
                    .batch(&interpretation_context, lifted_inputs.as_slice())
            }
            // The captured start indices are lane-uniform by construction: appending them as unbatched operands
            // lets the primal dynamic-update-slice batching rule lift the lane axis.
            Self::DynamicUpdateSlice(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs
                    .extend(operation.start_indices().iter().map(|index| ArrayBatch::unbatched(index.clone())));
                DynamicUpdateSliceOperation.batch(&interpretation_context, lifted_inputs.as_slice())
            }
            // The captured index operand is lane-uniform by construction: inserting it as the second (unbatched)
            // operand lets the primal gather batching rule lift the lane axis.
            Self::Gather(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                operation.operation().batch(
                    &interpretation_context,
                    &[inputs[0].clone(), ArrayBatch::unbatched(operation.indices().clone())],
                )
            }
            // The captured index operand is lane-uniform by construction: inserting it between the operand and update
            // tangents (unbatched) lets the primal scatter batching rule lift the lane axis.
            Self::ScatterAdd(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                operation.operation().batch(
                    &interpretation_context,
                    &[inputs[0].clone(), ArrayBatch::unbatched(operation.indices().clone()), inputs[1].clone()],
                )
            }
            // The captured factor is lane-uniform by construction: the same residual value applies to every lane.
            Self::Residual(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![ArrayBatch::unbatched(operation.capture().clone())])
            }
            // Recomputed primal operations batch through the wrapped operation's own primal batching rule.
            Self::Recompute(operation) => {
                let primal_context = EagerContext::<ArrayType, V, ArrayOperation<V>>::new();
                operation.batch(&primal_context, inputs)
            }
            // The captured predicate is lane-uniform: prepending it as an unbatched input lets the condition
            // batching helper read the branch choice from input 0, exactly like an ordinary runtime predicate.
            Self::Condition(operation) => {
                let mut condition_inputs = Vec::with_capacity(inputs.len() + 1);
                condition_inputs.push(ArrayBatch::unbatched(operation.predicate().clone()));
                condition_inputs.extend(inputs.iter().cloned());
                batch_condition_with_interpreter(
                    operation.true_branch(),
                    operation.false_branch(),
                    condition_inputs.as_slice(),
                    |program, program_inputs| {
                        program.interpret_with(
                            program_inputs,
                            |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                            |instruction, instruction_inputs| {
                                instruction.operation().batch(context, instruction_inputs)
                            },
                        )
                    },
                )
            }
            // The operand-form condition already reads its predicate from input 0, which is exactly the layout the
            // condition batching helper expects for an ordinary runtime predicate.
            Self::OperandCondition(operation) => batch_condition_with_interpreter(
                operation.true_branch(),
                operation.false_branch(),
                inputs,
                |program, program_inputs| {
                    program.interpret_with(
                        program_inputs,
                        |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                        |instruction, instruction_inputs| instruction.operation().batch(context, instruction_inputs),
                    )
                },
            ),
            Self::While(operation) => operation.batch(context, inputs),
            // Each lane's body pushforward is bound against that lane's residual slices and batched through the
            // shared scan loop; the residual stacks are concrete values in the direct linear form.
            Self::Scan(operation) => {
                let body = operation.body();
                let carry_count = operation.carry_count();
                let residual_stacks = operation.captures();
                let y_slice_types = body.output_types().split_off(carry_count);
                crate::tracing_v2::operations::scan::batch_scan_with_interpreter(
                    carry_count,
                    operation.length(),
                    operation.reverse(),
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| context.zero(stacked_type),
                    |lane, lane_inputs| {
                        let lane_residuals = residual_stacks
                            .iter()
                            .map(|stack| read_scan_lane(stack, lane))
                            .collect::<Result<Vec<_>, _>>()?;
                        let lane_body = body.map_operations(|operation| {
                            operation.try_map_captures(&mut |factor| factor.instantiate(lane_residuals.as_slice()))
                        })?;
                        lane_body.interpret_with(
                            lane_inputs,
                            |_, constant| Ok(ArrayBatch::unbatched(constant.clone())),
                            |instruction, instruction_inputs| {
                                instruction.operation().batch(context, instruction_inputs)
                            },
                        )
                    },
                )
            }
            Self::CustomVjpCall(call) => {
                let primal_context = EagerContext::<ArrayType, V, ArrayOperation<V>>::new();
                call.batch(&primal_context, inputs)
            }
            _ => unreachable!("non-control-flow LinearArrayOperation variants are handled above"),
        }
    }
}

/// Blanket active batching impl for the [`LinearArrayOperation`] sum type.
impl<C> BatchableOperation<Tracer<C>, BatchingContext<C>>
    for LinearArrayOperation<C::Constant, C::Constant, C::Constant, ArrayOperation<C::Constant>>
where
    ArrayOperation<C::Constant>: BatchableOperation<Tracer<C>, BatchingContext<C>>,
    C: StagingContext<Type = ArrayType>
        + Scale<ArrayType, Tracer<C>, C::Constant>
        + LeftDot<Tracer<C>, C::Constant, Captured>
        + RightDot<Tracer<C>, C::Constant, Captured>,
    C::Constant: Value<ArrayType> + BooleanLike + Slice + Reshape,
    C::Operation: From<ZeroOperation<ArrayType>>,
    Tracer<C>: SupportsLinearArithmeticOperations<C::Constant>
        + ZeroLike
        + OneLike
        + SupportsLinearAlgebraOperations<C::Constant>
        + SupportsManipulationOperations
        + BitAnd<Output = Tracer<C>>
        + Select<Condition = Tracer<C>>
        + BooleanLike
        + TransferToMemory,
    Vec<Tracer<C>>: Parameterized<Tracer<C>, To<Tracer<C>> = Vec<Tracer<C>>, ParameterStructure: Debug + PartialEq>,
{
    fn batch(
        &self,
        context: &BatchingContext<C>,
        inputs: &[ArrayBatch<Tracer<C>>],
    ) -> Result<Vec<ArrayBatch<Tracer<C>>>, ProgramError> {
        if let Some(outputs) = batch_linear_non_control_operation(self, context.parent_context(), inputs)? {
            return Ok(outputs);
        }
        match self {
            // Memory placement is lane-uniform: the same transfer applies to every lane, so the transfer is
            // staged unchanged on the physical batched value (in its own parent context) and the lane axis is
            // preserved. The parent operation type is generic here, so the value-level capability stages it.
            Self::TransferToMemory(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let tracer = inputs[0].value().transfer_to_memory(operation.destination());
                let physical_type = tracer.r#type().into_owned();
                Ok(vec![ArrayBatch::new(physical_type, tracer, inputs[0].batch_axis())?])
            }
            // The captured condition is a lane-uniform parent-context constant: lift it into the parent trace and
            // let the elementwise select batching rule broadcast it to the batched physical shape.
            Self::Select(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let condition = context.parent_context().constant(operation.condition().clone());
                SelectOperation.batch(
                    context.parent_context(),
                    &[ArrayBatch::unbatched(condition), inputs[0].clone(), inputs[1].clone()],
                )
            }
            // The captured start indices are lane-uniform parent-context constants: lift them into the parent
            // trace and let the primal dynamic-slice batching rule lift the lane axis.
            Self::DynamicSlice(operation) => {
                check_count!("input", inputs, 1, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(
                    operation
                        .start_indices()
                        .iter()
                        .map(|index| ArrayBatch::unbatched(context.parent_context().constant(index.clone()))),
                );
                DynamicSliceOperation::new(operation.sizes().to_vec())
                    .batch(context.parent_context(), lifted_inputs.as_slice())
            }
            // The captured start indices are lane-uniform parent-context constants: lift them into the parent
            // trace and let the primal dynamic-update-slice batching rule lift the lane axis.
            Self::DynamicUpdateSlice(operation) => {
                check_count!("input", inputs, 2, ProgramError);
                let mut lifted_inputs = inputs.to_vec();
                lifted_inputs.extend(
                    operation
                        .start_indices()
                        .iter()
                        .map(|index| ArrayBatch::unbatched(context.parent_context().constant(index.clone()))),
                );
                DynamicUpdateSliceOperation.batch(context.parent_context(), lifted_inputs.as_slice())
            }
            // The captured factor is a lane-uniform parent-context constant: lift it into the parent trace.
            Self::Residual(operation) => {
                check_count!("input", inputs, 0, ProgramError);
                Ok(vec![ArrayBatch::unbatched(context.parent_context().constant(operation.capture().clone()))])
            }
            // Recomputed primal operations batch through the wrapped operation's own primal batching rule.
            Self::Recompute(operation) => operation.batch(context, inputs),
            // The captured predicate is a lane-uniform parent-context constant, so the branch choice is concrete:
            // extract it from the factor and batch only the selected branch. Prepending a lifted predicate tracer
            // would defeat the lane-uniform extraction because tracers cannot be concretized.
            Self::Condition(operation) => {
                let branch =
                    if operation.predicate().boolean()? { operation.true_branch() } else { operation.false_branch() };
                context.interpret_program(branch, inputs.to_vec())
            }
            // The operand-form condition already reads its predicate from input 0, which is exactly the layout the
            // condition batching helper expects for an ordinary runtime predicate (lane-uniform predicates extract
            // concretely, lane-varying ones run both branches and select per lane).
            Self::OperandCondition(operation) => batch_condition_with_interpreter::<C::Constant, Tracer<C>, _, _>(
                operation.true_branch(),
                operation.false_branch(),
                inputs,
                |program, program_inputs| context.interpret_program(program, program_inputs),
            ),
            // The fused doubled-state linear while keeps the operational masked-unrolling rule even under tracing:
            // its condition recomputes the loop predicate from captured residual injections (parent-context
            // constants), so the per-iteration predicate extraction stays concrete and the loop unrolls through the
            // batched tracers. The staged batching rule on the primal `WhileOperation` does not apply here because
            // the loop's nested operation type is this linear enum, not the staged program's operation type.
            Self::While(operation) => {
                batch_while_with_interpreter(operation.as_ref(), inputs, |program, program_inputs| {
                    context.interpret_program(program, program_inputs)
                })
            }
            // Each lane's body pushforward is bound against that lane's residual slices at the constant level
            // (the stacks are lane-uniform parent-context constants) and batched over the traced lanes through
            // the shared scan loop; stacked output accumulators are staged as typed zeros in the parent trace.
            Self::Scan(operation) => {
                let body = operation.body();
                let carry_count = operation.carry_count();
                let residual_stacks = operation.captures();
                let y_slice_types = body.output_types().split_off(carry_count);
                crate::tracing_v2::operations::scan::batch_scan_with_interpreter(
                    carry_count,
                    operation.length(),
                    operation.reverse(),
                    y_slice_types.as_slice(),
                    inputs,
                    |stacked_type| {
                        let mut outputs = context
                            .parent_context()
                            .stage_nullary_operation(C::Operation::from(ZeroOperation::new(stacked_type.clone())))?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs.remove(0))
                    },
                    |lane, lane_inputs| {
                        let lane_residuals = residual_stacks
                            .iter()
                            .map(|stack| read_scan_lane(stack, lane))
                            .collect::<Result<Vec<_>, _>>()?;
                        let lane_body = body.map_operations(|operation| {
                            operation.try_map_captures(&mut |factor| factor.instantiate(lane_residuals.as_slice()))
                        })?;
                        context.interpret_program(&lane_body, lane_inputs)
                    },
                )
            }
            Self::CustomVjpCall(call) => {
                if !call.transposed() {
                    return Err(crate::types::TypeError {
                        message: "custom_vjp does not support forward-mode differentiation; use reverse mode (vjp, \
                            value_and_grad, or jacrev) instead"
                            .to_string(),
                    }
                    .into());
                }
                let mut values = call
                    .residuals()
                    .iter()
                    .map(|residual| ArrayBatch::unbatched(context.parent_context().constant(residual.clone())))
                    .collect::<Vec<_>>();
                values.extend(inputs.iter().cloned());
                context.interpret_program(call.backward(), values)
            }
            _ => unreachable!("non-control-flow LinearArrayOperation variants are handled above"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use pretty_assertions::assert_eq;

    use crate::differentiation::TransposableOperation;
    use crate::domains::AbstractDomain;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::tests::TestArray;
    use crate::tracing::AbstractTracingContext;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_linear_condition_transpose_supports_runtime_predicates() {
        // Linear-condition transposition is total: the captured predicate factor is a residual of the primal
        // computation rather than a linear operand, so it is carried verbatim into one staged transposed condition
        // over the transposed branch programs. Runtime (factor) predicates used to be rejected with an
        // `UnsupportedOperation` error.
        type TestLinearOperation = LinearArrayOperation<TestArray, TestArray, TestArray, ArrayOperation<TestArray>>;
        let scale_branch = |factor: f64| {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, TestLinearOperation>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let output = builder
                .add_instruction(
                    TestLinearOperation::Scale(ScaleOperation::new(TestArray::scalar(factor))),
                    vec![input],
                )
                .unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = TestLinearOperation::Condition(
            ConditionOperation::new_captured(
                TestArray::new(ArrayType::scalar(DataType::Boolean), vec![1.0]),
                scale_branch(2.0),
                scale_branch(3.0),
            )
            .unwrap(),
        );

        let domain = AbstractDomain::new();
        let builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, TestArray, TestLinearOperation>::new()));
        let cotangent_input = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let mut context = AbstractTracingContext::new(&domain, builder.clone());
        let cotangent = context.tracer(cotangent_input, None);
        let cotangents = operation
            .transpose(&mut context, &[&ArrayType::scalar(DataType::F64)], &[Cotangent::Staged(cotangent)])
            .unwrap();
        assert_eq!(cotangents.len(), 1);
        assert!(!cotangents[0].is_zero());
        let pullback_output = cotangents[0].as_staged().unwrap().atom_id().unwrap();
        assert!(matches!(builder.borrow().instructions()[0].operation(), TestLinearOperation::Condition { .. }));

        // Interpreting the pullback applies the transposed branch selected by the carried predicate (scale by 2).
        drop(cotangents);
        drop(context);
        let builder = Rc::try_unwrap(builder).unwrap().into_inner();
        let pullback = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![pullback_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let outputs = pullback.interpret(vec![TestArray::scalar(5.0)]).unwrap();
        assert_eq!(outputs[0].values, vec![10.0]);
    }
}
