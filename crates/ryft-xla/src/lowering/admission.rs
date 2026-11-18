//! Admission of data-dependent bounded-dynamic programs to XLA lowering.
//!
//! A dimension produced by `dimension_from_scalar` derives its extent from runtime data, so every operation that
//! consumes such a dimension lowers to bounded-dynamic StableHLO and must keep its physical padding unobservable.
//! [`validate_data_dependent_compilation`] admits only the operation configurations with a known padding discipline and
//! rejects every other consumer before any MLIR is constructed.

use std::collections::HashSet;

use DataDependentPaddingDiscipline::{Propagated, RyftMasked, Unsupported, XlaMasked, XlaZeroPadded};
use ryft_core::{
    ArrayIrType, ArrayOperation, DimensionVariable, Operation as CoreOperation, ProgramError, ReductionKind,
    ScatterMode, ScatterReductionKind, SortDirection, Type, Typed,
};

use crate::{Error, FlatProgram, Operation};

// TODO(eaplatanios): Review this module.

/// Validates the bounded-dynamic compilation contract for dimensions produced by `dimension_from_scalar`.
///
/// [`XlaMasked`] and [`XlaZeroPadded`] are explicit delegation decisions, because the bounded dynamic XLA legalizer
/// owns those transformations. [`RyftMasked`] operations instead materialize their physical kernel inputs and restore
/// logical extents in the lowering of Ryft.
/// Only configurations with execution evidence are admitted, and unbounded gateways and unsupported consumers are
/// rejected before MLIR construction.
///
/// # Errors
///
/// Returns an [`Error`] wrapping [`ProgramError::InvalidArgument`] when a data-derived dimension has no finite upper
/// bound, or wrapping [`ProgramError::UnsupportedOperation`] when an operation without a supported padding discipline
/// consumes a data-derived dimension.
pub(crate) fn validate_data_dependent_compilation(program: &FlatProgram) -> Result<(), Error> {
    let mut variables = HashSet::new();
    for region in program.regions() {
        for instruction in region.instructions() {
            if let Operation::DimensionFromScalar(operation) = instruction.operation() {
                // A static output defines no identity and is checked against its extent.
                let Some(variable) = operation.output_type().variable() else {
                    continue;
                };
                if variable.bounds().upper().is_none() {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "data-derived dimension `{variable}` with bounds {} needs a finite upper bound for XLA \
                             compilation",
                            variable.bounds(),
                        ),
                    }
                    .into());
                }
                variables.insert(variable.clone());
            }
        }
    }
    if variables.is_empty() {
        return Ok(());
    }

    // Dimension arithmetic produces fresh identities, so close the data-derived set transitively over dimension-typed
    // instruction outputs before classifying the array operations that consume those identities.
    loop {
        let count = variables.len();
        for region in program.regions() {
            for instruction in region.instructions() {
                if !instruction.inputs().iter().any(|input| {
                    references_data_derived_dimension(region.atoms()[input.index()].r#type().as_ref(), &variables)
                }) {
                    continue;
                }
                for output in instruction.outputs() {
                    // A static output defines no identity, like the static output of `dimension_from_scalar` above.
                    if let ArrayIrType::Dimension(r#type) = region.atoms()[output.index()].r#type().as_ref()
                        && let Some(variable) = r#type.variable()
                    {
                        variables.insert(variable.clone());
                    }
                }
            }
        }
        if variables.len() == count {
            break;
        }
    }

    for region in program.regions() {
        for instruction in region.instructions() {
            let consumes_data_derived_extent = instruction.inputs().iter().chain(instruction.outputs()).any(|atom| {
                references_data_derived_dimension(region.atoms()[atom.index()].r#type().as_ref(), &variables)
            });
            if consumes_data_derived_extent
                && let DataDependentPaddingDiscipline::Unsupported { reason } =
                    data_dependent_padding_discipline(instruction.operation())
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "operation `{}` cannot consume data-derived dimensions during XLA lowering because \
                             {reason}",
                        instruction.operation().name(),
                    ),
                }
                .into());
            }
        }
    }
    Ok(())
}

/// Padding contract used when admitting one staged operation to data-dependent bounded-dynamic lowering.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DataDependentPaddingDiscipline {
    /// StableHLO propagates the logical dynamic dimensions of the operation without inspecting physical padding
    /// positions.
    Propagated,

    /// Ryft delegates masking the padding positions with identities or sentinels to the bounded dynamic legalizer of
    /// XLA.
    XlaMasked,

    /// Ryft delegates padding the contraction positions with zeros to the bounded dynamic legalizer of XLA.
    XlaZeroPadded,

    /// Ryft explicitly materializes bounded physical operands with operation-specific padding values or masks and
    /// restores dynamic result metadata after the physical computation.
    RyftMasked,

    /// Ryft lacks sufficient semantic or execution evidence to admit this operation safely.
    Unsupported { reason: &'static str },
}

/// Returns the XLA masking contract for one reduction kind. The exhaustive match makes adding a reduction kind a
/// compile-time update to the data-dependent inventory instead of silently inheriting another kind's padding rule.
fn reduction_data_dependent_padding_discipline(kind: ReductionKind) -> DataDependentPaddingDiscipline {
    match kind {
        ReductionKind::Sum
        | ReductionKind::Product
        | ReductionKind::Max
        | ReductionKind::Min
        | ReductionKind::Any
        | ReductionKind::All => XlaMasked,
        // A log-sum-exp `reduce` expands into maximum and summation reductions over the same axes, so it inherits
        // the masking contract of exactly those two reduction kinds. XLA masks the operand of each reduction with the
        // identity of that reduction, which wipes the padding positions of the shifted exponentials before they are
        // summed. The rank-2 arm of `data_derived_padding_fixture` executes that expansion with the
        // data-derived extent on the kept axis, which is the admission's execution evidence.
        ReductionKind::LogSumExp => XlaMasked,
        ReductionKind::Mean => {
            Unsupported { reason: "mean over a dynamically sized reduction axis is not supported by the XLA lowering" }
        }
    }
}

/// Returns the XLA masking contract for one sort direction. Keeping the match exhaustive means that adding a sort
/// direction fails to compile until the inventory covers it, because ascending and descending sorts require opposite
/// sentinels.
fn sort_data_dependent_padding_discipline(direction: SortDirection) -> DataDependentPaddingDiscipline {
    match direction {
        SortDirection::Ascending | SortDirection::Descending => XlaMasked,
    }
}

/// Rejects every scatter combiner until its padding behavior has kind-specific execution evidence. The exhaustive
/// match deliberately forces new combiners through the same review rather than admitting them as ordinary operations
/// that preserve shapes.
fn scatter_data_dependent_padding_discipline(kind: ScatterReductionKind) -> DataDependentPaddingDiscipline {
    match kind {
        ScatterReductionKind::Overwrite => Unsupported {
            reason: "overwrite-scatter padding semantics have not been verified for data-derived dimensions",
        },
        ScatterReductionKind::Add => {
            Unsupported { reason: "add-scatter padding semantics have not been verified for data-derived dimensions" }
        }
        ScatterReductionKind::Mul => Unsupported {
            reason: "multiply-scatter padding semantics have not been verified for data-derived dimensions",
        },
        ScatterReductionKind::Min => Unsupported {
            reason: "minimum-scatter padding semantics have not been verified for data-derived dimensions",
        },
        ScatterReductionKind::Max => Unsupported {
            reason: "maximum-scatter padding semantics have not been verified for data-derived dimensions",
        },
    }
}

/// Classifies every homogeneous array operation by the padding discipline required for data-derived dimensions.
/// Keeping this match exhaustive makes a newly added operation a compile-time inventory update rather than silently
/// assuming that arbitrary padding is safe.
fn array_data_dependent_padding_discipline(
    operation: &ArrayOperation<crate::values::ArrayConstant>,
) -> DataDependentPaddingDiscipline {
    match operation {
        ArrayOperation::Reduce(operation) => reduction_data_dependent_padding_discipline(operation.kind()),
        ArrayOperation::Sort(operation) => sort_data_dependent_padding_discipline(operation.direction()),
        // XLA pads every input of the variadic reduction that these lower to with that input's own initial value along
        // a dynamic reduced axis (i.e., the extremum identity paired with index zero), which never beats a live element
        // and ties only when every live element equals the identity, where index zero is correct. Dynamic axes that
        // are not reduced pass through. `data_derived_padding_fixture` executes both configurations.
        ArrayOperation::ArgMax(_) | ArrayOperation::ArgMin(_) => XlaMasked,
        // Runtime-sized slice pullbacks use unique, in-bounds point updates. XLA masks inactive update positions, and
        // the compiled slice pullback fixture checks non-maximum and empty logical extents on CPU and CUDA.
        // Other window shapes, modes, and combiners retain their independent admission requirements.
        ArrayOperation::Scatter(operation)
            if operation.kind() == ScatterReductionKind::Add
                && operation.mode() == ScatterMode::PromiseInBounds
                && operation.unique_indices()
                && operation.dimensions().update_window_dimensions().is_empty() =>
        {
            XlaMasked
        }
        ArrayOperation::Scatter(operation) => scatter_data_dependent_padding_discipline(operation.kind()),
        // A prefix scan mixes positions only along its scanned axis, and type inference already requires that axis to
        // be static, so a data-derived extent can only sit on an axis the scan treats elementwise. That reasoning is
        // the same for every combining operator, and the prefix sum in `data_derived_padding_fixture` executes this arm
        // with the data-derived extent on the unscanned axis.
        ArrayOperation::Cumulative(_) => Propagated,
        ArrayOperation::Dot(_) => XlaZeroPadded,
        ArrayOperation::RaggedDot(_) => {
            Unsupported { reason: "group-size metadata may describe positions in bounded physical padding" }
        }
        ArrayOperation::ScaledDot(_) => RyftMasked,
        ArrayOperation::RngBitGenerator(_) => Unsupported {
            reason: "random generation advances state according to physical rather than logical element count",
        },
        ArrayOperation::Dropout(_) => {
            Unsupported { reason: "dropout masks are drawn according to physical rather than logical extents" }
        }
        ArrayOperation::ParallelRaggedAllToAll(_) => {
            Unsupported { reason: "ragged collective metadata may address physical padding positions" }
        }
        ArrayOperation::CustomCall(_) => Unsupported { reason: "custom-call physical-padding semantics are opaque" },
        ArrayOperation::Print(_) => Unsupported { reason: "print exposes the physical representation of its operand" },
        // Entry guards and `cast`s narrow or widen bounded dynamic values explicitly. The physical buffer is sliced or
        // padded, and every axis that stays dynamic gets its logical size back.
        ArrayOperation::Guard(_) | ArrayOperation::Cast(_) => RyftMasked,
        ArrayOperation::Assert(_)
        | ArrayOperation::Zero(_)
        | ArrayOperation::ZeroLike(_)
        | ArrayOperation::One(_)
        | ArrayOperation::OneLike(_)
        | ArrayOperation::Constant(_)
        | ArrayOperation::FillLike(_)
        | ArrayOperation::Iota(_)
        | ArrayOperation::IotaLike(_)
        | ArrayOperation::Abs(_)
        | ArrayOperation::Neg(_)
        | ArrayOperation::Add(_)
        | ArrayOperation::Sub(_)
        | ArrayOperation::SaturatingSub(_)
        | ArrayOperation::Mul(_)
        | ArrayOperation::Div(_)
        | ArrayOperation::Sin(_)
        | ArrayOperation::Cos(_)
        | ArrayOperation::Tan(_)
        | ArrayOperation::Atan2(_)
        | ArrayOperation::Exp(_)
        | ArrayOperation::ExpM1(_)
        | ArrayOperation::Log(_)
        | ArrayOperation::Ln1p(_)
        | ArrayOperation::LogAddExp(_)
        | ArrayOperation::Sqrt(_)
        | ArrayOperation::Rsqrt(_)
        | ArrayOperation::Tanh(_)
        | ArrayOperation::Logistic(_)
        | ArrayOperation::Erf(_)
        | ArrayOperation::Pow(_)
        | ArrayOperation::Sign(_)
        | ArrayOperation::Floor(_)
        | ArrayOperation::Ceil(_)
        | ArrayOperation::Round(_)
        | ArrayOperation::Max(_)
        | ArrayOperation::Min(_)
        | ArrayOperation::Clamp(_)
        | ArrayOperation::Rem(_)
        | ArrayOperation::Not(_)
        | ArrayOperation::And(_)
        | ArrayOperation::Or(_)
        | ArrayOperation::Xor(_)
        | ArrayOperation::Complex(_)
        | ArrayOperation::Conjugate(_)
        | ArrayOperation::Real(_)
        | ArrayOperation::Imaginary(_)
        | ArrayOperation::ParallelReduce(_)
        | ArrayOperation::ParallelVary(_)
        | ArrayOperation::ParallelCast(_)
        | ArrayOperation::ParallelAllGather(_)
        | ArrayOperation::ParallelSumScatter(_)
        | ArrayOperation::ParallelPermute(_)
        | ArrayOperation::ParallelBroadcast(_)
        | ArrayOperation::ParallelAllToAll(_)
        | ArrayOperation::AxisIndex(_)
        | ArrayOperation::Reverse(_)
        | ArrayOperation::Transpose(_)
        | ArrayOperation::Reshape(_)
        | ArrayOperation::Broadcast(_)
        | ArrayOperation::BroadcastLike(_)
        | ArrayOperation::Pad(_)
        | ArrayOperation::Concatenate(_)
        | ArrayOperation::Gather(_)
        | ArrayOperation::Slice(_)
        | ArrayOperation::UpdateSlice(_)
        | ArrayOperation::DynamicSlice(_)
        | ArrayOperation::DynamicUpdateSlice(_)
        | ArrayOperation::Compare(_)
        | ArrayOperation::Select(_)
        | ArrayOperation::Condition(_)
        | ArrayOperation::While(_)
        | ArrayOperation::Scan(_)
        | ArrayOperation::ConvertElementType(_)
        | ArrayOperation::ReducePrecision(_)
        | ArrayOperation::TransferToMemory(_)
        | ArrayOperation::Reshard(_)
        | ArrayOperation::ConstrainSharding(_)
        | ArrayOperation::StopGradient(_)
        | ArrayOperation::Tag(_)
        | ArrayOperation::Rematerialize(_)
        | ArrayOperation::Composite(_)
        | ArrayOperation::CompositeTranspose(_)
        | ArrayOperation::LinearCall(_) => Propagated,
    }
}

/// Classifies every XLA operation by the bounded-dynamic padding discipline it requires.
fn data_dependent_padding_discipline(operation: &Operation) -> DataDependentPaddingDiscipline {
    match operation {
        Operation::Array(operation) => array_data_dependent_padding_discipline(operation),
        Operation::RngBitGenerator(_) => Unsupported {
            reason: "random generation advances state according to physical rather than logical element count",
        },
        Operation::ParallelRaggedAllToAll(_) => {
            Unsupported { reason: "ragged collective metadata may address physical padding positions" }
        }
        Operation::CustomCall(_) => Unsupported { reason: "custom-call physical-padding semantics are opaque" },
        Operation::ReferenceNew(_)
        | Operation::ReferenceRead(_)
        | Operation::ReferenceWrite(_)
        | Operation::ReferenceSwap(_)
        | Operation::ReferenceAddUpdate(_)
        | Operation::ReferenceAtomicAddUpdate(_)
        | Operation::ReferenceFreeze(_) => {
            Unsupported { reason: "references must be discharged before bounded-dynamic XLA validation" }
        }
        Operation::Kernel(_) => {
            Unsupported { reason: "kernel calls must be selected before bounded-dynamic XLA validation" }
        }
        // Entry guards and `cast`s narrow or widen bounded-dynamic values explicitly (refer to the array arms).
        Operation::Guard(_) | Operation::Cast(_) => RyftMasked,
        Operation::ShardMap(_) => {
            Unsupported { reason: "shard-map lowering requires Shardy, which does not support bounded-dynamic tensors" }
        }
        Operation::Zero(_)
        | Operation::DynamicOne(_)
        | Operation::DynamicIota(_)
        | Operation::Dimension(_)
        | Operation::Compare(_)
        | Operation::Assert(_)
        | Operation::DimensionSize(_)
        | Operation::DimensionFromScalar(_)
        | Operation::DimensionToScalar(_)
        | Operation::Reshape(_)
        | Operation::Broadcast(_)
        | Operation::Concatenate(_)
        | Operation::Pad(_)
        | Operation::DynamicSlice(_)
        | Operation::ParallelAllGather(_)
        | Operation::ParallelSumScatter(_)
        | Operation::ParallelAllToAll(_)
        | Operation::Condition(_)
        | Operation::While(_)
        | Operation::Scan(_)
        | Operation::Composite(_)
        | Operation::LinearCall(_)
        | Operation::Rematerialize(_)
        | Operation::CompositeTranspose(_)
        | Operation::LiftedComposite(_)
        | Operation::LiftedCompositeTranspose(_)
        | Operation::JitCall(_) => Propagated,
    }
}

/// Returns whether `r#type` carries an identity derived from `dimension_from_scalar`.
fn references_data_derived_dimension(r#type: &ArrayIrType, variables: &HashSet<DimensionVariable>) -> bool {
    r#type.identities().any(|(_, identity)| variables.contains(identity))
}

#[cfg(test)]
mod tests {
    use DataDependentPaddingDiscipline::{Propagated, RyftMasked, Unsupported, XlaMasked};
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArgMaxOperation, ArgMinOperation, Array as CoreArray, ArrayIrType, ArrayOperation, ArrayType,
        AttentionConfiguration, AttentionImplementation, AttentionInputs, CompareOperation, ComparisonDirection,
        ConstantOperation, Context, ConvertElementTypeOperation, CumulativeKind, CumulativeOperation,
        CustomCallOperation, DataType, Differentiate, Dimension as CoreDimension, DimensionAddOperation,
        DimensionBounds, DimensionFromScalarOperation, DimensionToScalarOperation, DimensionType, DimensionVariable,
        DomainTracingContext, DotDimensionNumbers, DotOperation, DynamicBroadcastOperation, IotaOperation,
        NegOperation, OneOperation, Placeholder, PrintOperation, ProgramError, RandomAlgorithm, ReduceOperation,
        ReductionKind, ReferenceNewOperation, RngBitGeneratorOperation, ScaledDotOperation, ScatterDimensionNumbers,
        ScatterOperation, ScatterReductionKind, Shape, SortDirection, SortOperation, StaticShape, SubOperation,
        Value as CoreValue, differentiable_dot_product_attention,
    };
    use ryft_pjrt::Client;
    #[cfg(feature = "cuda-13")]
    use ryft_pjrt::{ClientOptions, GpuClientOptions, GpuMemoryAllocator, GpuPlatform};

    #[cfg(feature = "cuda-13")]
    use crate::Platform;
    use crate::sessions::tests::{
        domain_mesh, read_booleans, read_f32s, read_u64s, replicated_scalar_type, session_with_client,
    };
    use crate::tests::{cpu_client, execution_client, values_from_bytes};
    use crate::{Array, Constant, Error, FlatProgram, Operation, Options, ProgramBuilder, Session, Tracer, Value};

    use super::*;

    /// Reads the first addressable shard as `bfloat16` values converted to `f32`.
    fn read_bf16s(client: &Client<'_>, array: &Array) -> Vec<f32> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<u16>(bytes.as_slice())
            .into_iter()
            .map(|bits| half::bf16::from_bits(bits).to_f32())
            .collect()
    }

    /// Reads the first addressable shard as native-endian signed 32-bit integers.
    fn read_i32s(client: &Client<'_>, array: &Array) -> Vec<i32> {
        let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
        let bytes = array
            .device_shard(device_id)
            .unwrap()
            .buffer()
            .unwrap()
            .copy_to_host(None)
            .unwrap()
            .r#await()
            .unwrap();
        values_from_bytes::<i32>(bytes.as_slice())
    }

    /// Builds the kind-sensitive bounded-dynamic padding fixture shared by CPU and CUDA execution tests.
    fn data_derived_padding_fixture(size_type: ArrayType, extent: DimensionVariable) -> FlatProgram {
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let index_type = dynamic_type.clone().with_data_type(DataType::U64);
        let matrix_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![CoreDimension::Dynamic(extent.clone()), CoreDimension::Static(4)]),
        );

        // The iota and its negation make physical suffix positions numerically distinguishable from every admitted live
        // prefix. This catches both reduction identities and the opposite sentinels needed by ascending and descending
        // sort. The boolean reductions cover their own false and true identities, and dot covers a contraction padded
        // with zeros.
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let iota = builder
            .add_instruction(IotaOperation::new(dynamic_type.clone(), 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let indices = builder
            .add_instruction(IotaOperation::new(index_type, 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let negative = builder.add_instruction(NegOperation::new(), Vec::new(), vec![iota], None).unwrap()[0];
        let sorted_ascending = builder
            .add_instruction(SortOperation::new(0, SortDirection::Ascending), Vec::new(), vec![negative], None)
            .unwrap()[0];
        let sorted_descending = builder
            .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![iota], None)
            .unwrap()[0];
        let argmin_indices = builder
            .add_instruction(SortOperation::new(0, SortDirection::Ascending), Vec::new(), vec![negative, indices], None)
            .unwrap()[1];
        let argmax_indices = builder
            .add_instruction(SortOperation::new(0, SortDirection::Descending), Vec::new(), vec![iota, indices], None)
            .unwrap()[1];
        let sum = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Sum), Vec::new(), vec![iota], None)
            .unwrap()[0];
        // Shift the iota to nonzero factors so including physical padding changes the product.
        let ones =
            builder.add_instruction(OneOperation::new(dynamic_type), Vec::new(), vec![dimension], None).unwrap()[0];
        let factors = builder.add_instruction(AddOperation::new(), Vec::new(), vec![iota, ones], None).unwrap()[0];
        let product = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Product), Vec::new(), vec![factors], None)
            .unwrap()[0];
        let maximum = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Max), Vec::new(), vec![iota], None)
            .unwrap()[0];
        let minimum = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Min), Vec::new(), vec![negative], None)
            .unwrap()[0];
        let any_input = builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan)),
                Vec::new(),
                vec![negative, iota],
                None,
            )
            .unwrap()[0];
        let any = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Any), Vec::new(), vec![any_input], None)
            .unwrap()[0];
        let all_input = builder
            .add_instruction(
                ArrayOperation::Compare(CompareOperation::<ArrayType>::new(ComparisonDirection::GreaterThanOrEqual)),
                Vec::new(),
                vec![iota, negative],
                None,
            )
            .unwrap()[0];
        let all = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::All), Vec::new(), vec![all_input], None)
            .unwrap()[0];
        let dot = builder
            .add_instruction(
                DotOperation::new(DotDimensionNumbers::inner_product()),
                Vec::new(),
                vec![iota, iota],
                None,
            )
            .unwrap()[0];

        // The rank-2 `i + j` matrix carries the data-derived extent on an axis that neither the reduction nor the scan
        // touches, which is the configuration those two disciplines actually admit. A log-sum-exp `reduce` consumes the
        // static axis and still produces a dynamically shaped output, and the prefix scan consumes the static axis
        // while every row it emits keeps the dynamic extent.
        let rows = builder
            .add_instruction(IotaOperation::new(matrix_type.clone(), 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let columns = builder
            .add_instruction(IotaOperation::new(matrix_type, 1).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let matrix = builder.add_instruction(AddOperation::new(), Vec::new(), vec![rows, columns], None).unwrap()[0];
        let log_sum_exp = builder
            .add_instruction(ReduceOperation::new(vec![1], ReductionKind::LogSumExp), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        let cumulative_sum = builder
            .add_instruction(CumulativeOperation::new(1, CumulativeKind::Sum), Vec::new(), vec![matrix], None)
            .unwrap()[0];

        // Index reductions over the data-derived axis would select a physical suffix position if its padding were not
        // masked, because the suffix positions of the increasing iota are larger than every live position and those of
        // its negation are smaller. The matrix covers both a reduced data-derived axis and a data-derived axis that the
        // reduction keeps.
        let argmax = builder
            .add_instruction(ArgMaxOperation::new(0, DataType::I32), Vec::new(), vec![iota], None)
            .unwrap()[0];
        let argmin = builder
            .add_instruction(ArgMinOperation::new(0, DataType::I32), Vec::new(), vec![negative], None)
            .unwrap()[0];
        let row_argmax = builder
            .add_instruction(ArgMaxOperation::new(1, DataType::I32), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        let column_argmax = builder
            .add_instruction(ArgMaxOperation::new(0, DataType::I32), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![
                    sorted_ascending,
                    sorted_descending,
                    argmin_indices,
                    argmax_indices,
                    sum,
                    maximum,
                    minimum,
                    any,
                    all,
                    dot,
                    log_sum_exp,
                    cumulative_sum,
                    product,
                    argmax,
                    argmin,
                    row_argmax,
                    column_argmax,
                ],
                vec![Placeholder],
                vec![Placeholder; 17],
            )
            .unwrap()
    }

    /// Builds the bounded dynamic scaled dot and attention proof fixture. The attention values vary along the key and
    /// value sequence axis, so including any physical suffix position changes every valid output row. Attention is
    /// staged by the training function with the provided implementation, whose reverse mode differentiation calls the
    /// forward attention definition with its residual and the backward attention definition.
    fn data_derived_scaled_dot_attention_fixture(
        size_type: ArrayType,
        extent: DimensionVariable,
        scaled_element_type: DataType,
        scaled_scale_type: DataType,
        attention_implementation: AttentionImplementation,
    ) -> FlatProgram {
        /// Binds `operation` to `inputs` in the context of `value` and returns its only output.
        fn bind<O: Into<Operation>>(value: &Tracer, operation: O, inputs: &[Tracer]) -> Tracer {
            value.domain().bind(operation, Vec::<FlatProgram>::new(), inputs).unwrap().remove(0)
        }

        let scaled_block_size = match scaled_element_type {
            DataType::F8E4M3FN | DataType::F8E5M2 => 32,
            _ => 16,
        };
        let attention_batch = DimensionVariable::new("attention_batch", extent.bounds());
        let scaled_lhs_type = ArrayType::new(
            scaled_element_type,
            Shape::new(vec![CoreDimension::Dynamic(extent.clone()), CoreDimension::Static(scaled_block_size)]),
        );
        let scaled_lhs_scale_type = ArrayType::new(
            scaled_scale_type,
            Shape::new(vec![CoreDimension::Dynamic(extent.clone()), CoreDimension::Static(1)]),
        );
        let scaled_rhs_type = ArrayType::new(
            scaled_element_type,
            Shape::new(vec![CoreDimension::Static(1), CoreDimension::Static(scaled_block_size)]),
        );
        let scaled_rhs_scale_type =
            ArrayType::new(scaled_scale_type, Shape::new(vec![CoreDimension::Static(1), CoreDimension::Static(1)]));
        let attention_type = ArrayType::new(
            DataType::BF16,
            Shape::new(vec![
                CoreDimension::Dynamic(attention_batch.clone()),
                CoreDimension::Dynamic(extent.clone()),
                CoreDimension::Static(1),
                CoreDimension::Static(8),
            ]),
        );
        let attention = differentiable_dot_product_attention::<Session>(
            AttentionConfiguration::new().with_scale(1.0).with_implementation(attention_implementation),
        );
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| {
                let size = &inputs[0];
                let dimension =
                    bind(size, DimensionFromScalarOperation::new(extent.clone().into()), std::slice::from_ref(size));
                let batch_dimension = bind(
                    size,
                    DimensionFromScalarOperation::new(attention_batch.clone().into()),
                    std::slice::from_ref(size),
                );

                // The runtime row extent of `scaled_dot` is not contracted. Both dynamic element and scale operands
                // share it, while the static right-hand side input fixes one output column and the static block shape
                // fixes `k` to the block size of the element type (i.e., 32 for FP8 elements and 16 otherwise).
                let scaled_lhs =
                    bind(size, OneOperation::new(scaled_lhs_type.clone()), std::slice::from_ref(&dimension));
                let scaled_lhs_scales =
                    bind(size, OneOperation::new(scaled_lhs_scale_type.clone()), std::slice::from_ref(&dimension));
                let scaled_rhs = bind(size, OneOperation::new(scaled_rhs_type.clone()), &[]);
                let scaled_rhs_scales = bind(size, OneOperation::new(scaled_rhs_scale_type.clone()), &[]);
                let scaled_dot = bind(
                    size,
                    ScaledDotOperation::new(
                        DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                        DataType::F32,
                        true,
                        true,
                    ),
                    &[scaled_lhs, scaled_rhs, scaled_lhs_scales, scaled_rhs_scales],
                );

                // Equal query and key scores make the live attention softmax uniform, while values equal their
                // sequence index. Explicit lengths of `extent - 1` exercise the supported contract of restricting the
                // extent further. Visible query rows return the mean `(extent - 2) / 2`, while the final logical row
                // and all physical suffix positions stay zero. The backward pass has exact `dQ = 0`,
                // `dK[j] = 8 * (j - mean)`, and `dV = 1` on visible rows and zeros elsewhere.
                let extents = [batch_dimension.clone(), dimension.clone()];
                let query = bind(size, OneOperation::new(attention_type.clone()), &extents);
                let key = bind(size, OneOperation::new(attention_type.clone()), &extents);
                let value = bind(size, IotaOperation::new(attention_type.clone(), 1).unwrap(), &extents);
                let output_cotangent = bind(size, OneOperation::new(attention_type.clone()), &extents);
                let logical_length = bind(size, DimensionToScalarOperation, std::slice::from_ref(&dimension));
                let logical_length =
                    bind(size, ConvertElementTypeOperation::<ArrayType>::new(DataType::I32, false), &[logical_length]);
                let one = CoreArray::from_elements(ArrayType::scalar(DataType::I32), &[1i32]).unwrap();
                let one = bind(size, ConstantOperation::new(one), &[]);
                let length = bind(size, SubOperation::new(), &[logical_length, one]);
                let lengths =
                    bind(size, DynamicBroadcastOperation::new(Vec::new()), &[length, batch_dimension.clone()]);
                let inputs = AttentionInputs {
                    query_sequence_lengths: Some(lengths.clone()),
                    key_value_sequence_lengths: Some(lengths),
                    ..AttentionInputs::new(query, key, value)
                };
                let (output, pullback) = size.domain().differentiate_at(inputs).vjp(|inputs| attention.call(inputs))?;
                let cotangents = pullback.apply_materialized(output_cotangent)?;
                Ok(vec![scaled_dot, output, cotangents.query, cotangents.key, cotangents.value])
            },
            vec![ArrayIrType::from(size_type)],
        )
        .unwrap();
        program.into_flat_program()
    }

    #[test]
    fn test_validate_data_dependent_compilation() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client).with_device_mesh(mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);

        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let extent = DimensionVariable::new("unbounded", DimensionBounds::unbounded());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![dimension], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            session.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::InvalidArgument { message })
                    if *message == "data-derived dimension `unbounded` with bounds [0, ∞) needs a finite upper bound \
                        for XLA compilation",
            ),
        ));

        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let input = builder
            .add_instruction(IotaOperation::new(dynamic_type.clone(), 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                CustomCallOperation::new("ryft.test.opaque_dynamic", vec![dynamic_type]),
                Vec::new(),
                vec![input, dimension],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            session.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "operation `custom_call` cannot consume data-derived dimensions during XLA lowering \
                        because custom-call physical-padding semantics are opaque",
            ),
        ));

        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let input = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(ReduceOperation::new(vec![0], ReductionKind::Mean), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            session.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "operation `reduce` cannot consume data-derived dimensions during XLA lowering \
                        because mean over a dynamically sized reduction axis is not supported by the XLA lowering",
            ),
        ));

        // `print` renders whatever the physical buffer holds, so the padded suffix of a bounded-dynamic operand would
        // become observable.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let input = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("value"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            session.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message
                        == "operation `print` cannot consume data-derived \
                        dimensions during XLA lowering because \
                        print exposes the physical representation of its operand",
            ),
        ));

        // Bit generation over a bounded-dynamic result would advance the functional generator state by the physical
        // upper-bound element count rather than the logical one.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let bits_type = ArrayType::new(DataType::U32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.clone().into());
        let state = builder.add_input(RandomAlgorithm::ThreeFry.state_type().into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::ThreeFry, bits_type),
                Vec::new(),
                vec![state, dimension],
                None,
            )
            .unwrap()[1];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();
        assert!(matches!(
            session.lower_program(&program, 0, &Options::new(mesh.clone())),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "operation `rng_bit_generator` cannot consume data-derived dimensions during XLA \
                        lowering because random generation advances state according to physical rather than logical \
                        element count",
            ),
        ));

        // Every scatter combiner is unverified against physical padding, so a data-derived operand extent is rejected
        // per kind rather than per operation name.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Dynamic(extent.clone())]));
        let indices_type =
            ArrayType::new(DataType::I32, Shape::new(vec![CoreDimension::Static(2), CoreDimension::Static(1)]));
        let updates_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(2)]));
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(size_type.into());
        let indices = builder.add_input(indices_type.into());
        let updates = builder.add_input(updates_type.into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let operand = builder
            .add_instruction(IotaOperation::new(dynamic_type, 0).unwrap(), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(
                ScatterOperation::new(
                    ScatterDimensionNumbers::new(Vec::new(), vec![0], vec![0]),
                    ScatterReductionKind::Add,
                ),
                Vec::new(),
                vec![operand, indices, updates],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(
                vec![output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            session.lower_program(&program, 0, &Options::new(mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::UnsupportedOperation { message })
                    if *message
                        == "operation `scatter` cannot consume data-derived \
                        dimensions during XLA lowering because \
                        add-scatter padding semantics have not been verified for data-derived dimensions",
            ),
        ));
    }

    #[test]
    fn test_validate_data_dependent_compilation_accepts_static_dimension_outputs() {
        // The sum of a data-derived dimension with itself has singleton bounds here, so its type is static and
        // defines no identity to track.
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(4, Some(5)).unwrap());
        let dimension_type = DimensionType::from(extent.clone());
        let mut builder = ProgramBuilder::new();
        let size = builder.add_input(replicated_scalar_type(&mesh, DataType::I64).into());
        let dimension = builder
            .add_instruction(DimensionFromScalarOperation::new(extent.into()), Vec::new(), vec![size], None)
            .unwrap()[0];
        let sum = builder
            .add_instruction(
                DimensionAddOperation::new(&dimension_type, &dimension_type).unwrap(),
                Vec::new(),
                vec![dimension, dimension],
                None,
            )
            .unwrap()[0];
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![sum], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.regions()[0].atoms()[sum.index()].r#type().as_ref(),
            &ArrayIrType::Dimension(DimensionType::from(CoreDimension::Static(8))),
        );
        assert_eq!(validate_data_dependent_compilation(&program).map_err(|error| error.to_string()), Ok(()));
    }

    #[test]
    fn test_data_dependent_padding_discipline() {
        // Every reference operation shares one collapsed match arm, so one representative pins that arm.
        assert_eq!(
            data_dependent_padding_discipline(&Operation::ReferenceNew(ReferenceNewOperation::new())),
            Unsupported { reason: "references must be discharged before bounded-dynamic XLA validation" },
        );

        for kind in [
            ReductionKind::Sum,
            ReductionKind::Product,
            ReductionKind::LogSumExp,
            ReductionKind::Max,
            ReductionKind::Min,
            ReductionKind::Any,
            ReductionKind::All,
        ] {
            assert_eq!(reduction_data_dependent_padding_discipline(kind), XlaMasked);
        }
        assert_eq!(
            reduction_data_dependent_padding_discipline(ReductionKind::Mean),
            Unsupported { reason: "mean over a dynamically sized reduction axis is not supported by the XLA lowering" },
        );
        for direction in [SortDirection::Ascending, SortDirection::Descending] {
            assert_eq!(sort_data_dependent_padding_discipline(direction), XlaMasked);
        }
        for operation in [
            ArrayOperation::ArgMax(ArgMaxOperation::new(0, DataType::I32)),
            ArrayOperation::ArgMin(ArgMinOperation::new(0, DataType::I32)),
        ] {
            assert_eq!(array_data_dependent_padding_discipline(&operation), XlaMasked);
        }
        for (kind, combiner) in [
            (ScatterReductionKind::Overwrite, "overwrite"),
            (ScatterReductionKind::Add, "add"),
            (ScatterReductionKind::Mul, "multiply"),
            (ScatterReductionKind::Min, "minimum"),
            (ScatterReductionKind::Max, "maximum"),
        ] {
            let Unsupported { reason } = scatter_data_dependent_padding_discipline(kind) else {
                panic!("scatter combiner `{kind:?}` must be unsupported");
            };
            assert_eq!(
                reason,
                format!("{combiner}-scatter padding semantics have not been verified for data-derived dimensions"),
            );
        }

        assert_eq!(
            array_data_dependent_padding_discipline(&ArrayOperation::ScaledDot(ScaledDotOperation::new(
                DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
                DataType::F32,
                true,
                true,
            ))),
            RyftMasked,
        );

        // A log-sum-exp `reduce` expands into maximum and summation reductions, so it carries the same masking
        // contract as those two reduction kinds.
        assert_eq!(
            array_data_dependent_padding_discipline(&ArrayOperation::Reduce(ReduceOperation::new(
                vec![0],
                ReductionKind::LogSumExp,
            ))),
            XlaMasked,
        );

        // The prefix scans own their own arm rather than sharing the collapsed shape-preserving one, because the
        // classification rests on type inference keeping the scanned axis static.
        for kind in [
            CumulativeKind::Sum,
            CumulativeKind::Product,
            CumulativeKind::Max,
            CumulativeKind::Min,
            CumulativeKind::LogSumExp,
        ] {
            assert_eq!(
                array_data_dependent_padding_discipline(&ArrayOperation::Cumulative(CumulativeOperation::new(0, kind))),
                Propagated,
            );
        }
    }

    #[test]
    fn test_data_derived_padding_disciplines_execute_without_observable_padding() {
        let client = execution_client();
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client).with_device_mesh(mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program = data_derived_padding_fixture(size_type.clone(), extent);

        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&session, size_type.clone(), mesh.clone(), None, size.to_ne_bytes().as_slice())
                    .unwrap();
            let eager = program
                .interpret_in_context(&session, vec![Value::Array(input.clone())])
                .unwrap()
                .into_iter()
                .map(|value| match value {
                    Value::Array(array) => array,
                    Value::Dimension(_) => panic!("padding fixture returned a first-class dimension"),
                    Value::Reference(_) => panic!("padding fixture returned a reference"),
                })
                .collect::<Vec<_>>();
            let compiled = session.execute_program(&compiled, vec![input]).unwrap();
            assert_eq!(compiled.len(), eager.len());
            for (compiled, eager) in compiled.iter().zip(&eager) {
                assert_eq!(compiled.shape(), eager.shape());
                assert_eq!(compiled.data_type(), eager.data_type());
                let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
                let compiled_bytes = compiled
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                let eager_bytes = eager
                    .device_shard(device_id)
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                assert_eq!(compiled_bytes, eager_bytes);
            }
            compiled
        };

        let small = execute(2);
        assert_eq!(small[0].shape(), StaticShape::new(vec![2]));
        assert_eq!(read_f32s(client, &small[0]), vec![-1.0, 0.0]);
        assert_eq!(read_f32s(client, &small[1]), vec![1.0, 0.0]);
        assert_eq!(read_u64s(client, &small[2]), vec![1, 0]);
        assert_eq!(read_u64s(client, &small[3]), vec![1, 0]);
        assert_eq!(read_f32s(client, &small[4]), vec![1.0]);
        assert_eq!(read_f32s(client, &small[5]), vec![1.0]);
        assert_eq!(read_f32s(client, &small[6]), vec![-1.0]);
        assert_eq!(read_booleans(client, &small[7]), vec![true]);
        assert_eq!(read_booleans(client, &small[8]), vec![true]);
        assert_eq!(read_f32s(client, &small[9]), vec![1.0]);
        assert_eq!(read_f32s(client, &small[12]), vec![2.0]);
        assert_eq!(read_i32s(client, &small[13]), vec![1]);
        assert_eq!(read_i32s(client, &small[14]), vec![1]);
        assert_eq!(read_i32s(client, &small[15]), vec![3, 3]);
        assert_eq!(read_i32s(client, &small[16]), vec![1, 1, 1, 1]);

        // Row `i` of the `i + j` matrix reduces to `i + ln(1 + e + e² + e³)`, so exactly the live rows appear and a
        // physical suffix row would show up as an extra element. The tolerance is here because the expansion runs
        // `exp` and `log`, not because the discipline is approximate, and the byte comparison against the reference
        // kernel inside `execute` is exact.
        assert_eq!(small[10].shape(), StaticShape::new(vec![2]));
        let log_sum_exp = read_f32s(client, &small[10]);
        assert!((log_sum_exp[0] - 3.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
        assert!((log_sum_exp[1] - 4.440_19).abs() < 1e-4, "{log_sum_exp:?}");

        // The prefix scan runs along the static axis, so row `i` accumulates `[i, 2i + 1, 3i + 3, 4i + 6]` while the
        // data-derived extent rides through on the rows it never mixes.
        assert_eq!(small[11].shape(), StaticShape::new(vec![2, 4]));
        assert_eq!(read_f32s(client, &small[11]), vec![0.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 10.0]);

        // With a single live position, every physical suffix position of the data-derived axis is padding, which index
        // reductions must still never select.
        let single = execute(1);
        assert_eq!(read_i32s(client, &single[13]), vec![0]);
        assert_eq!(read_i32s(client, &single[14]), vec![0]);
        assert_eq!(read_i32s(client, &single[15]), vec![3]);
        assert_eq!(read_i32s(client, &single[16]), vec![0, 0, 0, 0]);

        let at_bound = execute(4);
        assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4]));
        assert_eq!(read_f32s(client, &at_bound[0]), vec![-3.0, -2.0, -1.0, 0.0]);
        assert_eq!(read_f32s(client, &at_bound[1]), vec![3.0, 2.0, 1.0, 0.0]);
        assert_eq!(read_u64s(client, &at_bound[2]), vec![3, 2, 1, 0]);
        assert_eq!(read_u64s(client, &at_bound[3]), vec![3, 2, 1, 0]);
        assert_eq!(read_f32s(client, &at_bound[4]), vec![6.0]);
        assert_eq!(read_f32s(client, &at_bound[5]), vec![3.0]);
        assert_eq!(read_f32s(client, &at_bound[6]), vec![-3.0]);
        assert_eq!(read_booleans(client, &at_bound[7]), vec![true]);
        assert_eq!(read_booleans(client, &at_bound[8]), vec![true]);
        assert_eq!(read_f32s(client, &at_bound[9]), vec![14.0]);
        assert_eq!(read_f32s(client, &at_bound[12]), vec![24.0]);
        assert_eq!(read_i32s(client, &at_bound[13]), vec![3]);
        assert_eq!(read_i32s(client, &at_bound[14]), vec![3]);
        assert_eq!(read_i32s(client, &at_bound[15]), vec![3, 3, 3, 3]);
        assert_eq!(read_i32s(client, &at_bound[16]), vec![3, 3, 3, 3]);

        // At the bound the reduction and the scan grow by exactly the two extra live rows, which is what rules out a
        // padded row silently participating at the smaller size.
        assert_eq!(at_bound[10].shape(), StaticShape::new(vec![4]));
        let log_sum_exp = read_f32s(client, &at_bound[10]);
        assert!((log_sum_exp[0] - 3.440_189_7).abs() < 1e-4, "{log_sum_exp:?}");
        assert!((log_sum_exp[1] - 4.440_19).abs() < 1e-4, "{log_sum_exp:?}");
        assert!((log_sum_exp[2] - 5.440_19).abs() < 1e-4, "{log_sum_exp:?}");
        assert!((log_sum_exp[3] - 6.440_19).abs() < 1e-4, "{log_sum_exp:?}");
        assert_eq!(at_bound[11].shape(), StaticShape::new(vec![4, 4]));
        assert_eq!(
            read_f32s(client, &at_bound[11]),
            vec![0.0, 1.0, 3.0, 6.0, 1.0, 3.0, 6.0, 10.0, 2.0, 5.0, 9.0, 14.0, 3.0, 7.0, 12.0, 18.0],
        );
    }

    #[test]
    fn test_data_derived_scaled_dot_and_attention_execute_without_observable_padding() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = session_with_client(client).with_device_mesh(mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program = data_derived_scaled_dot_attention_fixture(
            size_type.clone(),
            extent,
            DataType::F32,
            DataType::F32,
            AttentionImplementation::Automatic,
        );
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        assert!(lowered.stable_hlo().contains("stablehlo.composite \"xla.scaled_dot\""));
        assert!(!lowered.stable_hlo().contains("__cudnn$fmha"));
        let compiled = session.compile_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&session, size_type.clone(), mesh.clone(), None, size.to_ne_bytes().as_slice())
                    .unwrap();
            let eager = program
                .interpret_in_context(&session, vec![Value::Array(input.clone())])
                .unwrap()
                .into_iter()
                .map(|value| match value {
                    Value::Array(array) => array,
                    Value::Dimension(_) => panic!("attention fixture returned a first-class dimension"),
                    Value::Reference(_) => panic!("attention fixture returned a reference"),
                })
                .collect::<Vec<_>>();
            let compiled = session.execute_program(&compiled, vec![input]).unwrap();
            assert_eq!(compiled.len(), eager.len());
            for (compiled, eager) in compiled.iter().zip(&eager) {
                assert_eq!(compiled.shape(), eager.shape());
                assert_eq!(compiled.data_type(), eager.data_type());
                let device_id = client.addressable_devices().unwrap()[0].id().unwrap();
                assert_eq!(
                    compiled
                        .device_shard(device_id)
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .copy_to_host(None)
                        .unwrap()
                        .r#await()
                        .unwrap(),
                    eager
                        .device_shard(device_id)
                        .unwrap()
                        .buffer()
                        .unwrap()
                        .copy_to_host(None)
                        .unwrap()
                        .r#await()
                        .unwrap(),
                );
            }
            compiled
        };

        let small = execute(2);
        assert_eq!(small[0].shape(), StaticShape::new(vec![2, 1]));
        assert_eq!(read_f32s(client, &small[0]), vec![16.0; 2]);
        assert_eq!(small[1].shape(), StaticShape::new(vec![2, 2, 1, 8]));
        assert_eq!(read_bf16s(client, &small[1]), vec![0.0; 32]);
        assert_eq!(read_bf16s(client, &small[2]), vec![0.0; 32]);
        assert_eq!(read_bf16s(client, &small[3]), vec![0.0; 32]);
        assert_eq!(
            read_bf16s(client, &small[4]),
            [1.0, 0.0].into_iter().cycle().take(4).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );

        let at_bound = execute(4);
        assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4, 1]));
        assert_eq!(read_f32s(client, &at_bound[0]), vec![16.0; 4]);
        assert_eq!(at_bound[1].shape(), StaticShape::new(vec![4, 4, 1, 8]));
        assert_eq!(
            read_bf16s(client, &at_bound[1]),
            [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );
        assert_eq!(read_bf16s(client, &at_bound[2]), vec![0.0; 128]);
        // Three visible keys have equal scores and values 0, 1, 2. Each query contributes
        // `dS[j] = (1 / 3) * 8 * (j - 1)`, so summing three visible queries gives `dK[j] = 8 * (j - 1)`
        // and `dV[j] = 3 * (1 / 3) = 1`. The rounded BF16 residual must not rescale these probabilities.
        assert_eq!(
            read_bf16s(client, &at_bound[3]),
            [-8.0, 0.0, 8.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );
        assert_eq!(
            read_bf16s(client, &at_bound[4]),
            [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
        );

        let zero =
            Array::from_host_buffer(&session, size_type.clone(), mesh.clone(), None, 0i64.to_ne_bytes().as_slice())
                .unwrap();
        let error = session.execute_program(&compiled, vec![zero]).unwrap_err();
        assert!(
            error.to_string().contains(
                "`dimension_from_scalar` failed: input dimension `extent` = 0 is outside its declared bounds [1, 5)"
            ),
            "{error}",
        );

        // Eager execution replays the attention calls of the bounded-dynamic program at the exact types of their
        // inputs, so a program that explicitly requests the fused kernels reports that it cannot use them instead of
        // silently computing the portable result.
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let fused = data_derived_scaled_dot_attention_fixture(
            size_type.clone(),
            extent,
            DataType::F32,
            DataType::F32,
            AttentionImplementation::Fused,
        );
        let input = Array::from_host_buffer(&session, size_type, mesh, None, 2i64.to_ne_bytes().as_slice()).unwrap();
        let error = fused.interpret_in_context(&session, vec![Value::Array(input)]).unwrap_err();
        assert!(matches!(
            error,
            ProgramError::UnsupportedOperation { message }
                if message == "cannot lower `ryft.dot_product_attention`: the fused attention implementation is \
                    unavailable because the target platform is not CUDA",
        ));
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_cuda_data_derived_padding_disciplines_execute_without_observable_padding() {
        let client = Platform::Cuda
            .client(Some(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            })))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let session = session_with_client(&client).with_device_mesh(mesh.clone());
        let size_type = replicated_scalar_type(&mesh, DataType::I64);
        let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
        let program = data_derived_padding_fixture(size_type.clone(), extent);
        let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
        let compiled = session.compile_program(&lowered).unwrap();
        let execute = |size: i64| {
            let input =
                Array::from_host_buffer(&session, size_type.clone(), mesh.clone(), None, size.to_ne_bytes().as_slice())
                    .unwrap();
            session.execute_program(&compiled, vec![input]).unwrap()
        };

        let small = execute(2);
        assert_eq!(read_f32s(&client, &small[0]), vec![-1.0, 0.0]);
        assert_eq!(read_f32s(&client, &small[1]), vec![1.0, 0.0]);
        assert_eq!(read_u64s(&client, &small[2]), vec![1, 0]);
        assert_eq!(read_u64s(&client, &small[3]), vec![1, 0]);
        assert_eq!(read_f32s(&client, &small[4]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[5]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[6]), vec![-1.0]);
        assert_eq!(read_booleans(&client, &small[7]), vec![true]);
        assert_eq!(read_booleans(&client, &small[8]), vec![true]);
        assert_eq!(read_f32s(&client, &small[9]), vec![1.0]);
        assert_eq!(read_f32s(&client, &small[12]), vec![2.0]);
        assert_eq!(read_i32s(&client, &small[13]), vec![1]);
        assert_eq!(read_i32s(&client, &small[14]), vec![1]);
        assert_eq!(read_i32s(&client, &small[15]), vec![3, 3]);
        assert_eq!(read_i32s(&client, &small[16]), vec![1, 1, 1, 1]);

        let at_bound = execute(4);
        assert_eq!(read_f32s(&client, &at_bound[0]), vec![-3.0, -2.0, -1.0, 0.0]);
        assert_eq!(read_f32s(&client, &at_bound[1]), vec![3.0, 2.0, 1.0, 0.0]);
        assert_eq!(read_u64s(&client, &at_bound[2]), vec![3, 2, 1, 0]);
        assert_eq!(read_u64s(&client, &at_bound[3]), vec![3, 2, 1, 0]);
        assert_eq!(read_f32s(&client, &at_bound[4]), vec![6.0]);
        assert_eq!(read_f32s(&client, &at_bound[5]), vec![3.0]);
        assert_eq!(read_f32s(&client, &at_bound[6]), vec![-3.0]);
        assert_eq!(read_booleans(&client, &at_bound[7]), vec![true]);
        assert_eq!(read_booleans(&client, &at_bound[8]), vec![true]);
        assert_eq!(read_f32s(&client, &at_bound[9]), vec![14.0]);
        assert_eq!(read_f32s(&client, &at_bound[12]), vec![24.0]);
        assert_eq!(read_i32s(&client, &at_bound[13]), vec![3]);
        assert_eq!(read_i32s(&client, &at_bound[14]), vec![3]);
        assert_eq!(read_i32s(&client, &at_bound[15]), vec![3, 3, 3, 3]);
        assert_eq!(read_i32s(&client, &at_bound[16]), vec![3, 3, 3, 3]);
    }

    #[cfg(feature = "cuda-13")]
    #[test]
    fn test_cuda_data_derived_scaled_dot_and_attention_execute_fused_without_observable_padding() {
        let client = Platform::Cuda
            .client(Some(ClientOptions::GPU(GpuClientOptions {
                platform: Some(GpuPlatform::CUDA),
                allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                ..Default::default()
            })))
            .unwrap();
        let mesh = domain_mesh(&client, "x", 1);
        let session = session_with_client(&client).with_device_mesh(mesh.clone());
        for (scaled_element_type, scaled_scale_type) in [
            (DataType::F4E2M1FN, DataType::F8E4M3FN),
            (DataType::F8E4M3FN, DataType::F8E8M0FNU),
            (DataType::F8E5M2, DataType::F8E8M0FNU),
        ] {
            let scaled_dot_value = match scaled_element_type {
                DataType::F8E4M3FN | DataType::F8E5M2 => 32.0,
                _ => 16.0,
            };
            let size_type = replicated_scalar_type(&mesh, DataType::I64);
            let extent = DimensionVariable::new("extent", DimensionBounds::new(1, Some(5)).unwrap());
            let program = data_derived_scaled_dot_attention_fixture(
                size_type.clone(),
                extent,
                scaled_element_type,
                scaled_scale_type,
                AttentionImplementation::Fused,
            );
            let lowered = session.lower_program(&program, 0, &Options::new(mesh.clone())).unwrap();
            // cuDNN accepts the NVFP4 and `f8e4m3fn` MXFP8 pairs, which lower to the block-scaled-dot custom call, but
            // not `f8e5m2` on both sides, which keeps the composite.
            let uses_custom_call = scaled_element_type != DataType::F8E5M2;
            assert_eq!(lowered.stable_hlo().contains("stablehlo.custom_call @__op$block_scaled_dot"), uses_custom_call);
            assert_eq!(lowered.stable_hlo().contains("stablehlo.composite \"xla.scaled_dot\""), !uses_custom_call);
            assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmax"));
            assert!(lowered.stable_hlo().contains("__cudnn$fmhaSoftmaxBackward"));
            let compiled = session.compile_program(&lowered).unwrap();
            let execute = |size: i64| {
                let input = Array::from_host_buffer(
                    &session,
                    size_type.clone(),
                    mesh.clone(),
                    None,
                    size.to_ne_bytes().as_slice(),
                )
                .unwrap();
                session.execute_program(&compiled, vec![input]).unwrap()
            };

            let small = execute(2);
            assert_eq!(small[0].shape(), StaticShape::new(vec![2, 1]));
            assert_eq!(read_f32s(&client, &small[0]), vec![scaled_dot_value; 2]);
            assert_eq!(small[1].shape(), StaticShape::new(vec![2, 2, 1, 8]));
            assert_eq!(read_bf16s(&client, &small[1]), vec![0.0; 32]);
            assert_eq!(read_bf16s(&client, &small[2]), vec![0.0; 32]);
            assert_eq!(read_bf16s(&client, &small[3]), vec![0.0; 32]);
            assert_eq!(
                read_bf16s(&client, &small[4]),
                [1.0, 0.0].into_iter().cycle().take(4).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
            );

            let at_bound = execute(4);
            assert_eq!(at_bound[0].shape(), StaticShape::new(vec![4, 1]));
            assert_eq!(read_f32s(&client, &at_bound[0]), vec![scaled_dot_value; 4]);
            assert_eq!(at_bound[1].shape(), StaticShape::new(vec![4, 4, 1, 8]));
            assert_eq!(
                read_bf16s(&client, &at_bound[1]),
                [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>(),
            );
            assert_eq!(read_bf16s(&client, &at_bound[2]), vec![0.0; 128]);
            let actual_query_gradient = read_bf16s(&client, &at_bound[3]);
            let expected_query_gradient =
                [-8.0, 0.0, 8.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>();
            // cuDNN's fused BF16 backward-attention kernel is not bit-identical to the unfused reference
            // composition, so compare its gradients with an explicit reduced-precision tolerance.
            actual_query_gradient.iter().zip(&expected_query_gradient).for_each(|(actual, expected)| {
                assert!((actual - expected).abs() <= 0.25, "expected {expected} within 0.25, but found {actual}");
            });
            let actual_value_gradient = read_bf16s(&client, &at_bound[4]);
            let expected_value_gradient =
                [1.0, 1.0, 1.0, 0.0].into_iter().cycle().take(16).flat_map(|value| [value; 8]).collect::<Vec<_>>();
            actual_value_gradient.iter().zip(&expected_value_gradient).for_each(|(actual, expected)| {
                assert!((actual - expected).abs() <= 0.25, "expected {expected} within 0.25, but found {actual}");
            });

            let zero = Array::from_host_buffer(&session, size_type, mesh.clone(), None, 0i64.to_ne_bytes().as_slice())
                .unwrap();
            let error = session.execute_program(&compiled, vec![zero]).unwrap_err();
            assert!(
                error.to_string().contains(
                    "`dimension_from_scalar` failed: input dimension `extent` = 0 is outside its declared bounds [1, 5)"
                ),
                "{error}",
            );
        }
    }
}
