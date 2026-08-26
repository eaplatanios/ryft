//! Contains the [`CumulativeProductOperation`], the prefix-product member of the cumulative family, together with
//! the [`CumulativeProduct`] staging capability and every rule the member does not choose for itself, all generated
//! by `define_cumulative_operation!`.

// TODO(eaplatanios): Review this module.

use std::fmt::Display;

use crate::arrays::{ArrayBatch, ArrayBatching, ArrayType, DataType, RaggedArrayBatchingPolicy, RaggedMaskIdentity};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::constants::zero::ZeroOperationProvider;
use crate::operations::cumulative::{
    cumulative_abstract, define_cumulative_operation, jvp_through_associative_scan, lift_cumulative_axis,
};
use crate::operations::manipulation::concatenation::ConcatenateOperation;
use crate::operations::manipulation::padding::PadOperation;
use crate::operations::manipulation::slicing::SliceOperation;
use crate::operations::math::add::AddOperation;
use crate::operations::math::mul::{Mul, MulOperation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};

define_cumulative_operation! {
    /// Primitive representing one inclusive prefix product along a single array axis.
    ///
    /// Output element `i` along [`axis`](Self::axis) holds the product of input elements `0..=i`, or the product of
    /// elements `i..` when [`reverse`](Self::reverse) is set. The output type is the input type, so a cumulative
    /// product is a shape-preserving unary primitive.
    ///
    /// The identity of the combining operator is one, which is what a batching rule writes over the padding of a
    /// bounded ragged scanned axis so that padded positions cannot change a live prefix.
    ///
    /// The scanned dimension must be static and unsharded, as documented on [`cumulative_abstract`]. The element data
    /// type must be real or complex numeric, or the payload-free structural zero, every prefix product of which is
    /// again zero.
    operation = CumulativeProductOperation,
    name = CUMULATIVE_PRODUCT_OPERATION_NAME = "cumulative_product",
    abstract_rule = cumulative_product_abstract,
    element_domain = |data_type| data_type.is_numeric() || data_type == DataType::Zero,
    element_domain_error = format!(
        "`{CUMULATIVE_PRODUCT_OPERATION_NAME}` requires numeric inputs but got {data_type}"
    ),
    ragged_identity = RaggedMaskIdentity::One,
    combine_operation = MulOperation<ArrayType>,
    combine = |left, right| left.mul(right),
    /// Value-level cumulative multiplication capability.
    ///
    /// [`CumulativeProduct`] is the receiver-style entry point for staging or executing a
    /// [`CumulativeProductOperation`]: it scans the receiver along `axis`, returning a value of the receiver's own
    /// type whose element `i` along that axis holds the product of the receiver's elements `0..=i` (or `i..` for the
    /// reverse direction).
    capability = CumulativeProduct::{cumulative_product, reverse_cumulative_product},
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::batching::DynamicArrayBatchingPolicy;
    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, DataType, Dimension, DimensionBounds, DimensionType, DimensionVariable,
        LogicalMesh, MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding, ShardingDimension,
    };
    use crate::contexts::{EagerContext, ProjectedContext, StagingContext};
    use crate::macros::{check_operation_batching, check_operation_differentiation};
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ValueProjection};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_cumulative_product_operation_type_inference() {
        // The output type follows the staged input type exactly, and complex payloads are accepted alongside the
        // real numeric ones.
        let operation = CumulativeProductOperation::new(1);
        let input = ArrayType::new_static(DataType::F64, [3, 2]);
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input), &[]), Ok(vec![input]));
        let complex = ArrayType::new_static(DataType::C64, [3, 2]);
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&complex), &[]), Ok(vec![complex]));

        // Non-numeric payloads have no multiplication.
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new_static(DataType::Boolean, [3, 2])], &[]),
            Err(TypeError::invalid("`cumulative_product` requires numeric inputs but got bool".to_string())),
        );

        // The geometry rules of the family are enforced against the staged input type.
        assert_eq!(
            operation.infer_output_types(&[ArrayType::new_static(DataType::F64, [3])], &[]),
            Err(TypeError::invalid("`cumulative_product` axis 1 is out of bounds for rank 1".to_string())),
        );
        let dynamic = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Static(3),
                Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap())),
            ]),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&dynamic), &[]),
            Err(TypeError::invalid(format!(
                "`cumulative_product` requires a static scanned dimension but axis 1 of {dynamic} is dynamic"
            ))),
        );
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = ArrayType::new_static(DataType::F64, [3, 2])
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&sharded), &[]),
            Err(TypeError::invalid(format!(
                "`cumulative_product` requires an unsharded scanned dimension but axis 1 of {sharded} is sharded"
            ))),
        );
    }

    #[test]
    fn test_cumulative_product_operation_rendering() {
        // The scan direction renders only when it is set, keeping the common forward scan compact.
        assert_eq!(CumulativeProductOperation::new(1).to_string(), "cumulative_product [axis=1]");
        assert_eq!(
            CumulativeProductOperation::new(0).with_reverse(true).to_string(),
            "cumulative_product [axis=0, reverse=true]",
        );
        assert_eq!(CumulativeProductOperation::new(0).with_reverse(false), CumulativeProductOperation::new(0));
    }

    #[test]
    fn test_cumulative_product_operation_interpretation() {
        let context = EagerContext::<Array>::new();
        let interpret = |operation: &CumulativeProductOperation, input: &Array| {
            operation.interpret(&context, &EmptyRegionDriver, std::slice::from_ref(input)).unwrap().remove(0)
        };

        // Forward scans accumulate prefixes and reverse scans accumulate suffixes, along the selected axis only.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            interpret(&CumulativeProductOperation::new(1), &input),
            Array::matrix(2, 3, vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]),
        );
        assert_eq!(
            interpret(&CumulativeProductOperation::new(1).with_reverse(true), &input),
            Array::matrix(2, 3, vec![6.0, 6.0, 3.0, 120.0, 30.0, 6.0]),
        );
        assert_eq!(
            interpret(&CumulativeProductOperation::new(0), &input),
            Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 10.0, 18.0]),
        );

        // A zero absorbs every later prefix going forward and every earlier suffix going backward.
        let with_zero = Array::vector(vec![2.0, 0.0, 3.0, 4.0]);
        let zeroed = Array::vector(vec![2.0, 0.0, 0.0, 0.0]);
        assert_eq!(interpret(&CumulativeProductOperation::new(0), &with_zero), zeroed);
        assert_eq!(
            interpret(&CumulativeProductOperation::new(0).with_reverse(true), &with_zero),
            Array::vector(vec![0.0, 0.0, 12.0, 4.0]),
        );

        // A zero-length scanned axis has nothing to accumulate and keeps the operand's exact type.
        let empty = Array::new(ArrayType::new_static(DataType::F32, [0, 2]), Vec::new()).unwrap();
        assert_eq!(interpret(&CumulativeProductOperation::new(0), &empty), empty);

        // Complex payloads multiply as complex numbers.
        let complex = Array::vector(vec![ComplexNumber::new(0.0_f64, 1.0), ComplexNumber::new(0.0, 1.0)]);
        assert_eq!(
            interpret(&CumulativeProductOperation::new(0), &complex),
            Array::vector(vec![ComplexNumber::new(0.0_f64, 1.0), ComplexNumber::new(-1.0, 0.0)]),
        );
    }

    #[test]
    fn test_cumulative_product_capability_over_eager_arrays() {
        // The capability is the receiver-style entry point of the same kernel, in both scan directions, and it
        // reports the operation's own validation errors instead of panicking.
        let input = Array::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(input.cumulative_product(0), Ok(Array::vector(vec![1.0, 2.0, 6.0])));
        assert_eq!(input.reverse_cumulative_product(0), Ok(Array::vector(vec![6.0, 6.0, 3.0])));
        assert_eq!(
            input.cumulative_product(1),
            Err(ProgramError::Type(TypeError::invalid(
                "`cumulative_product` axis 1 is out of bounds for rank 1".to_string(),
            ))),
        );
    }

    #[test]
    fn test_cumulative_product_operation_batches_along_the_shifted_axis() {
        // A replicated operand carries no inserted batch dimension, so the scanned axis needs no shift.
        check_operation_batching!(
            @exact,
            operation = CumulativeProductOperation::new(0),
            axis_size = 2,
            cases = [{
                inputs = [(@replicated, Array::vector(vec![1.0, 2.0, 3.0]))],
                outputs = [(@replicated, Array::vector(vec![1.0, 2.0, 6.0]))],
            }],
        );

        // Physical input is [2 batch items, 3 columns] mapped at axis 0, so the per-item axis 0 scans physical axis
        // 1 and each batch item accumulates independently.
        check_operation_batching!(
            @exact,
            operation = CumulativeProductOperation::new(0),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))],
                outputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![1.0, 2.0, 6.0, 4.0, 20.0, 120.0]))],
            }],
        );

        // With the batch dimension inserted after the scanned axis, the scanned axis keeps its position.
        check_operation_batching!(
            @exact,
            operation = CumulativeProductOperation::new(0),
            axis_size = 3,
            cases = [{
                inputs = [(@mapped(axis = 1), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))],
                outputs = [(@mapped(axis = 1), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 10.0, 18.0]))],
            }],
        );
    }

    #[test]
    fn test_cumulative_product_operation_masks_ragged_padding_with_one() {
        // Scanning a ragged axis would multiply its padding into every later live prefix, so the rule asks the
        // policy to neutralize that padding with the multiplicative identity first. Static array batching cannot,
        // and says so rather than silently scanning padding.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input = ArrayBatch::new(Array::matrix(2, 3, vec![1.0_f32; 6]), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![1_i32, 3]), variable, vec![0])])
            .unwrap();
        assert_eq!(
            CumulativeProductOperation::new(0).batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                &[input],
            ),
            Err(BatchingError::UnsupportedOperation {
                message: "static array batching cannot identity-mask bounded ragged dimension `length` on axis 1 \
                          with `One`"
                    .to_string(),
            }),
        );

        // The composite dynamic policy can, and stages the mask ahead of the scan: the padded positions of the
        // scanned axis are selected away in favor of a one, and the ragged axis survives on the result because a
        // scan consumes no axis.
        type TraceContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        let trace = TraceContext::new();
        let items = DimensionVariable::new("items", DimensionBounds::new(1, Some(9)).unwrap());
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let batch_extent = trace.input(DimensionType::new(items.clone()).into());
        let packed = trace.input(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(items.clone()), Dimension::Static(3)]))
                .into(),
        );
        let extents = trace.input(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Dynamic(items)])).into());
        let context = BatchingContext::<_, ArrayBatching<DynamicArrayBatchingPolicy>>::with_policy(
            ProjectedContext::new(trace.clone()),
            batch_extent,
        );
        let input = ArrayBatch::new(packed.into_projected().unwrap(), BatchAxis::new(0))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.into_projected().unwrap(), length.clone(), vec![0])])
            .unwrap();
        // The per-item scan of axis 0 is the packed axis 1 that carries the ragged extents.
        let (outputs, evidence) = CumulativeProductOperation::new(0)
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].ragged_axes().len(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].axis(), 1);
        assert_eq!(outputs[0].ragged_axes()[0].dimension(), &length);
        assert!(evidence.is_empty());

        let output_id = outputs.into_iter().next().unwrap().into_value().into_value().atom_id().unwrap();
        drop(context);
        let program = trace
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output_id],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:dimension<items ∈ [1, 9)>, %1:f32[items, 3], %2:i32[items] .
                let %3:dimension<items ∈ [1, 9)> = dimension_size [axis=0] %1
                    %4:dimension<3> = constant [value=3]
                    %5:i32[3] = iota [type=i32[3], dimension=0]
                    %6:i32[items, 3] = broadcast [output_axes=[1]] %5 %3 %4
                    %7:i32[items, 3] = broadcast [output_axes=[0]] %2 %3 %4
                    %8:bool[items, 3] = compare [direction=LessThan] %6 %7
                    %9:f32[] = constant [value=1.0]
                    %10:f32[items, 3] = broadcast [output_axes=[]] %9 %3 %4
                    %11:f32[items, 3] = select %8 %1 %10
                    %12:f32[items, 3] = cumulative_product [axis=1] %11
                in (%12)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_cumulative_product_operation_differentiation() {
        // The primitive is nonlinear, so the rule differentiates through the associative-scan decomposition. The
        // expected tangents below are the exact product rule applied to each prefix (and each suffix).
        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeProductOperation::new(0),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0, 4.0])],
                tangents = [Array::vector(vec![1.0, 1.0, 1.0, 1.0])],
                primal_outputs = [Array::vector(vec![1.0, 2.0, 6.0, 24.0])],
                tangent_outputs = [Array::vector(vec![1.0, 3.0, 11.0, 50.0])],
            }],
        );
        check_operation_differentiation!(
            @approx(step = 1e-4, epsilon = 1e-6),
            operation = CumulativeProductOperation::new(0).with_reverse(true),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0, 4.0])],
                tangents = [Array::vector(vec![1.0, 1.0, 1.0, 1.0])],
                primal_outputs = [Array::vector(vec![24.0, 24.0, 12.0, 4.0])],
                tangent_outputs = [Array::vector(vec![50.0, 26.0, 7.0, 1.0])],
            }],
        );
    }
}
