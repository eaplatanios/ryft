use std::marker::PhantomData;

use ryft_macros::Parameterized;

use crate::contexts::Context;
use crate::differentiation::DifferentiationError;
use crate::differentiation::forward::{DifferentiableOperation, DifferentiationContext, LinearizationTracer};
use crate::differentiation::jacobian::{jacobian_forward_in_context, jacobian_reverse_in_context};
use crate::differentiation::linear::ResidualZeroProvider;
use crate::differentiation::reverse::TransposableOperation;
use crate::differentiation::types::DenseDifferentiableType;
use crate::operations::AddOperation;
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::{ProgramError, Type, Typed, Value};
use crate::tracing::TracingContext;

/// Hessian of a function, represented as the Cartesian product of its output, first input, and second input
/// [`Parameter`] leaves. `I` and `O` retain the input and output [`Type`] trees. Derivative values are stored in
/// deterministic output-major / first-input-major / second-input-minor order and remain [`Parameter`]s so that the
/// complete Hessian can cross tracing and compilation boundaries as well as participate in higher-order transforms.
/// The physical representation of a block is defined by [`DenseDifferentiableType`].
/// For [`ArrayType`](crate::ArrayType), the block for an output leaf with shape `O` and
/// input leaves with shapes `I1` and `I2` has shape `O` concatenated with `I1` and `I2`.
#[derive(Clone, Debug, Parameterized)]
pub struct Hessian<T: Type, V: Parameter, I: Parameterized<T>, O: Parameterized<T>> {
    /// [`Type`] of the differentiated inputs.
    input_types: I,

    /// [`Type`] of the differentiated outputs.
    output_types: O,

    /// Derivative values in output-major/first-input-major/second-input-minor order.
    values: Vec<V>,

    /// [`PhantomData`] marker for `T`, needed because the input and output fields use `T` only through their bounds.
    _type: PhantomData<fn() -> T>,
}

impl<T: Type, V: Parameter, I: Parameterized<T>, O: Parameterized<T>> Hessian<T, V, I, O> {
    /// Creates a new [`Hessian`].
    pub fn new(input_types: I, output_types: O, values: Vec<V>) -> Result<Self, ProgramError> {
        let input_count = input_types.parameter_count();
        let expected_count = output_types
            .parameter_count()
            .checked_mul(input_count)
            .and_then(|count| count.checked_mul(input_count))
            .ok_or_else(|| ProgramError::InvalidArgument {
                message: "Hessian block count overflows usize".to_string(),
            })?;
        if values.len() != expected_count {
            return Err(ProgramError::InvalidArgument {
                message: format!("Hessian requires {} derivative values but got {}", expected_count, values.len()),
            });
        }
        Ok(Self { input_types, output_types, values, _type: PhantomData })
    }

    /// Returns the [`Type`] of the differentiated inputs.
    #[inline]
    pub fn input_types(&self) -> &I {
        &self.input_types
    }

    /// Returns the [`Type`] of the differentiated outputs.
    #[inline]
    pub fn output_types(&self) -> &O {
        &self.output_types
    }

    /// Returns derivative values in output-major/first-input-major/second-input-minor order.
    #[inline]
    pub fn values(&self) -> &[V] {
        self.values.as_slice()
    }

    /// Consumes this [`Hessian`] and returns its derivative values in output-major/first-input-major/second-input-minor
    /// order.
    #[inline]
    pub fn into_values(self) -> Vec<V> {
        self.values
    }

    /// Returns the [`HessianBlock`] of this [`Hessian`] for the specified output, first input, and second input
    /// [`ParameterPath`]s, or `None` if any of the provided paths is absent.
    pub fn block(
        &self,
        output_path: &ParameterPath,
        first_input_path: &ParameterPath,
        second_input_path: &ParameterPath,
    ) -> Option<HessianBlock<'_, T, V>> {
        let (output_index, (_, output_type)) =
            self.output_types.named_parameters().enumerate().find(|(_, (path, _))| path == output_path)?;
        let input_count = self.input_types.parameter_count();
        let (first_input_index, (_, first_input_type)) =
            self.input_types.named_parameters().enumerate().find(|(_, (path, _))| path == first_input_path)?;
        let (second_input_index, (_, second_input_type)) =
            self.input_types.named_parameters().enumerate().find(|(_, (path, _))| path == second_input_path)?;
        Some(HessianBlock {
            output_path: output_path.clone(),
            output_type,
            first_input_path: first_input_path.clone(),
            first_input_type,
            second_input_path: second_input_path.clone(),
            second_input_type,
            value: &self.values
                [output_index * input_count * input_count + first_input_index * input_count + second_input_index],
        })
    }

    /// Returns borrowed views of all [`HessianBlock`]s of this [`Hessian`] in
    /// output-major/first-input-major/second-input-minor order.
    pub fn iter_blocks(&self) -> impl Iterator<Item = HessianBlock<'_, T, V>> {
        let input_count = self.input_types.parameter_count();
        self.output_types
            .named_parameters()
            .enumerate()
            .flat_map(move |(output_index, (output_path, output_type))| {
                self.input_types.named_parameters().enumerate().flat_map(
                    move |(first_input_index, (first_input_path, first_input_type))| {
                        let output_path = output_path.clone();
                        self.input_types.named_parameters().enumerate().map(
                            move |(second_input_index, (second_input_path, second_input_type))| HessianBlock {
                                output_path: output_path.clone(),
                                output_type,
                                first_input_path: first_input_path.clone(),
                                first_input_type,
                                second_input_path,
                                second_input_type,
                                value: &self.values[output_index * input_count * input_count
                                    + first_input_index * input_count
                                    + second_input_index],
                            },
                        )
                    },
                )
            })
    }
}

/// Borrowed view of one output/input/input block in a [`Hessian`].
#[derive(Debug)]
pub struct HessianBlock<'o, T: Type, V> {
    /// [`ParameterPath`] of the differentiated output [`Parameter`] that this [`HessianBlock`] corresponds to.
    output_path: ParameterPath,

    /// [`Type`] of the differentiated output [`Parameter`] that this [`HessianBlock`] corresponds to.
    output_type: &'o T,

    /// [`ParameterPath`] of the first differentiated input [`Parameter`] that this [`HessianBlock`] corresponds to.
    first_input_path: ParameterPath,

    /// [`Type`] of the first differentiated input [`Parameter`] that this [`HessianBlock`] corresponds to.
    first_input_type: &'o T,

    /// [`ParameterPath`] of the second differentiated input [`Parameter`] that this [`HessianBlock`] corresponds to.
    second_input_path: ParameterPath,

    /// [`Type`] of the second differentiated input [`Parameter`] that this [`HessianBlock`] corresponds to.
    second_input_type: &'o T,

    /// Derivative value for this [`HessianBlock`].
    value: &'o V,
}

impl<'o, T: Type, V> HessianBlock<'o, T, V> {
    /// Returns the [`ParameterPath`] of the differentiated output [`Parameter`] that this [`HessianBlock`]
    /// corresponds to.
    #[inline]
    pub fn output_path(&self) -> &ParameterPath {
        &self.output_path
    }

    /// Returns the [`Type`] of the differentiated output [`Parameter`] that this [`HessianBlock`] corresponds to.
    #[inline]
    pub fn output_type(&self) -> &'o T {
        self.output_type
    }

    /// Returns the [`ParameterPath`] of the first differentiated input [`Parameter`] that this [`HessianBlock`]
    /// corresponds to.
    #[inline]
    pub fn first_input_path(&self) -> &ParameterPath {
        &self.first_input_path
    }

    /// Returns the [`Type`] of the first differentiated input [`Parameter`] that this [`HessianBlock`] corresponds to.
    #[inline]
    pub fn first_input_type(&self) -> &'o T {
        self.first_input_type
    }

    /// Returns the [`ParameterPath`] of the second differentiated input [`Parameter`] that this [`HessianBlock`]
    /// corresponds to.
    #[inline]
    pub fn second_input_path(&self) -> &ParameterPath {
        &self.second_input_path
    }

    /// Returns the [`Type`] of the second differentiated input [`Parameter`] that this [`HessianBlock`] corresponds to.
    #[inline]
    pub fn second_input_type(&self) -> &'o T {
        self.second_input_type
    }

    /// Returns the derivative value for this [`HessianBlock`].
    #[inline]
    pub fn value(&self) -> &'o V {
        self.value
    }
}

impl<'o, T: Type, V> Clone for HessianBlock<'o, T, V> {
    fn clone(&self) -> Self {
        Self {
            output_path: self.output_path.clone(),
            output_type: self.output_type,
            first_input_path: self.first_input_path.clone(),
            first_input_type: self.first_input_type,
            second_input_path: self.second_input_path.clone(),
            second_input_type: self.second_input_type,
            value: self.value,
        }
    }
}

/// Implements the forward-over-reverse differentiation transform used by
/// [`hessian`](crate::DifferentiationBuilder::hessian) in an explicitly provided [`Context`].
/// Refer to the documentation of that function for information on the mathematical interpretation,
/// block representation, cost model, complex-type contract, runtime-capture behavior, and auxiliary-output
/// semantics.
///
/// # Parameters
///
///   - `context`: Context in which to trace and replay the transform.
///   - `function`: Function returning the differentiated output and auxiliary output.
///   - `primal`: Structured input values specifying the differentiation point.
///   - `capture`: Structured runtime values held fixed through both derivative levels.
///   - `holomorphic`: Whether to validate all differentiated leaves under a holomorphy promise.
pub(crate) fn hessian_in_context<C, Input, Capture, Output, Aux, F>(
    context: &C,
    function: F,
    primal: Input,
    capture: Capture,
    holomorphic: bool,
) -> Result<
    (Hessian<C::Type, C::Value, Input::To<C::Type>, Output::To<C::Type>>, Aux::To<C::Value>),
    DifferentiationError,
>
where
    C: Context<
            Type: DenseDifferentiableType<C>
                      + DenseDifferentiableType<DifferentiationContext<PartialEvaluationContext<C>>>,
            Operation: PartiallyEvaluatableOperation<C>
                           + PartiallyEvaluatableOperation<DifferentiationContext<PartialEvaluationContext<C>>>
                           + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
                           + DifferentiableOperation<PartialEvaluationContext<C>>
                           + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
                           + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
                           + DifferentiableOperation<
                PartialEvaluationContext<DifferentiationContext<PartialEvaluationContext<C>>>,
            > + TransposableOperation<C::Constant, C::Operation>
                           + ResidualZeroProvider<C::Type>
                           + From<AddOperation<C::Type>>,
        >,
    Input: Parameterized<
            C::Value,
            To<C::Value> = Input,
            To<C::Type>: Clone,
            To<LinearizationTracer<C>>: Parameterized<
                LinearizationTracer<C>,
                To<LinearizationTracer<C>> = Input::To<LinearizationTracer<C>>,
                To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = Input::To<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
                To<C::Type> = Input::To<C::Type>,
            >,
            Family: ParameterizedFamily<C::Type>
                        + ParameterizedFamily<LinearizationTracer<C>>
                        + ParameterizedFamily<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
        >,
    Capture: Parameterized<
            C::Value,
            To<C::Value> = Capture,
            To<LinearizationTracer<C>>: Parameterized<
                LinearizationTracer<C>,
                To<LinearizationTracer<C>> = Capture::To<LinearizationTracer<C>>,
                To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = Capture::To<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
            >,
            Family: ParameterizedFamily<LinearizationTracer<C>>
                        + ParameterizedFamily<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
        >,
    Output: Parameterized<
            LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
            To<C::Type>: Clone,
            Family: ParameterizedFamily<C::Type> + ParameterizedFamily<LinearizationTracer<C>>,
        >,
    Aux: Parameterized<
            LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
            To<LinearizationTracer<C>>: Parameterized<LinearizationTracer<C>, To<C::Value> = Aux::To<C::Value>>,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Value>,
        >,
    F: FnOnce(
        Input::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
        Capture::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
    ) -> Result<(Output, Aux), ProgramError>,
{
    let (outer, auxiliary) = jacobian_forward_in_context(
        context,
        |outer_primals, outer_capture| {
            let nested_context = outer_primals
                .parameters()
                .next()
                .map(Value::execution_domain)
                .ok_or(DifferentiationError::EmptyInput)?;
            jacobian_reverse_in_context(&nested_context, function, outer_primals, outer_capture, holomorphic)
                .map_err(ProgramError::from)
        },
        primal,
        capture,
        holomorphic,
    )?;
    let input_types = outer.input_type().clone();
    let output_types = outer.output_type().output_type().clone();
    let values = outer.into_values();
    let mut value_index = 0;
    for output_type in output_types.parameters() {
        for first_input_type in input_types.parameters() {
            for second_input_type in input_types.parameters() {
                <C::Type as DenseDifferentiableType<C>>::validate_hessian_block_type(
                    values[value_index].r#type().as_ref(),
                    output_type,
                    first_input_type,
                    second_input_type,
                )?;
                value_index += 1;
            }
        }
    }
    Ok((Hessian::new(input_types, output_types, values)?, auxiliary))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use approx::assert_abs_diff_eq;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::DataType::{F32, F64};
    use crate::arrays::{Array, ArrayType, DataType, Dimension, Shape};
    use crate::differentiation::{Differentiate, differentiate_at};
    use crate::operations::Sin;
    use crate::parameters::{ParameterPath, Parameterized};
    use crate::programs::Typed;

    use super::*;

    #[test]
    fn test_hessian() {
        // Parameterization preserves the complete output/first-input/second-input Cartesian-product order.
        let parameterized_hessian = Hessian::new((F32, F64), F64, vec![1_i32, 2, 3, 4]).unwrap();
        assert_eq!(parameterized_hessian.parameter_count(), 4);
        let blocks = parameterized_hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.iter().map(|block| *block.value()).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(blocks[0].output_path(), &ParameterPath::root());
        assert_eq!(blocks[0].first_input_path().to_string(), "$.0");
        assert_eq!(blocks[0].second_input_path().to_string(), "$.0");
        assert_eq!(blocks[3].first_input_path().to_string(), "$.1");
        assert_eq!(blocks[3].second_input_path().to_string(), "$.1");

        let output_path = blocks[0].output_path().clone();
        let first_input_path = blocks[2].first_input_path().clone();
        let second_input_path = blocks[2].second_input_path().clone();
        assert_eq!(
            *parameterized_hessian.block(&output_path, &first_input_path, &second_input_path).unwrap().value(),
            3,
        );
        assert!(
            parameterized_hessian
                .block(&ParameterPath::root().field("missing"), &first_input_path, &second_input_path)
                .is_none()
        );

        let reparameterized = <Hessian<DataType, f32, _, _>>::from_parameters(
            parameterized_hessian.parameter_structure(),
            [5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        assert_eq!(reparameterized.input_types(), &(F32, F64));
        assert_eq!(reparameterized.output_types(), &F64);
        assert_eq!(reparameterized.values(), &[5.0, 6.0, 7.0, 8.0]);

        // A scalar-valued function of two scalar inputs produces the expected dense 2-by-2 Hessian blocks.
        let scalar_hessian = differentiate_at((Array::scalar(2.0), Array::scalar(3.0)))
            .hessian(|(x, y)| Ok(x.clone() * y + x.sin()?))
            .unwrap();
        let blocks = scalar_hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 4);
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], -2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[2].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[3].value().to_f64s()[0], 0.0, epsilon = 1e-9);

        // Narrow primal element types use their widened differential representation for dense Hessian blocks.
        let input = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let widened_hessian = differentiate_at(input).hessian(|value| value.sin()).unwrap();
        let block = widened_hessian.iter_blocks().next().unwrap();
        assert_eq!(block.value().r#type().as_ref(), &ArrayType::scalar(F32));
        assert_abs_diff_eq!(block.value().to_f64s()[0], -2.0f64.sin(), epsilon = 1e-6);

        // Zero-sized inputs and outputs remain concrete, honestly typed dense blocks.
        let r#type = ArrayType::new(F64, Shape::new(vec![Dimension::Static(0)]));
        let zero_sized_hessian = differentiate_at(Array::from_f64s(r#type, Vec::new()))
            .hessian(|input| Ok(input.clone() * input))
            .unwrap();
        let block = zero_sized_hessian.iter_blocks().next().unwrap();
        assert_eq!(block.value().r#type().static_shape().unwrap().as_slice(), &[0, 0, 0]);
        assert!(block.value().storage_bytes().is_empty());

        // Structured outputs retain a distinct Hessian block for each output leaf.
        let structured_hessian = differentiate_at(Array::scalar(2.0))
            .hessian(|x| Ok((x.clone() * x.clone(), x.clone() * x.clone() * x)))
            .unwrap();
        let blocks = structured_hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].first_input_path().to_string(), "$");
        assert_eq!(blocks[0].second_input_path().to_string(), "$");
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_eq!(blocks[1].output_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 12.0, epsilon = 1e-9);

        // Mixed-rank structured inputs materialize the entire block Cartesian product with output axes leading both
        // input-axis groups.
        let mixed_rank_hessian = differentiate_at((Array::vector(vec![1.0, 2.0]), Array::scalar(3.0)))
            .hessian(|(vector, scalar)| Ok((vector.clone() * vector, scalar.clone() * scalar)))
            .unwrap();
        let blocks = mixed_rank_hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 8);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].first_input_path().to_string(), "$.0");
        assert_eq!(blocks[0].second_input_path().to_string(), "$.0");
        assert_eq!(blocks[0].value().r#type().static_shape().unwrap().as_slice(), &[2, 2, 2]);
        assert_eq!(blocks[0].value().to_f64s(), vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
        assert_eq!(blocks[1].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[1].value().to_f64s(), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(blocks[2].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[2].value().to_f64s(), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(blocks[3].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[3].value().to_f64s(), vec![0.0, 0.0]);
        assert_eq!(blocks[4].output_path().to_string(), "$.1");
        assert_eq!(blocks[4].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[4].value().to_f64s(), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(blocks[5].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[5].value().to_f64s(), vec![0.0, 0.0]);
        assert_eq!(blocks[6].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[6].value().to_f64s(), vec![0.0, 0.0]);
        assert_eq!(blocks[7].first_input_path().to_string(), "$.1");
        assert_eq!(blocks[7].second_input_path().to_string(), "$.1");
        assert!(blocks[7].value().r#type().static_shape().unwrap().as_slice().is_empty());
        assert_eq!(blocks[7].value().to_f64s(), vec![2.0]);

        // Holomorphic Hessians remain complex linear at both derivative levels.
        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0));
        let holomorphic_hessian =
            differentiate_at(input).holomorphic().hessian(|x| Ok(x.clone() * x.clone() * x)).unwrap();
        assert_eq!(
            holomorphic_hessian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(12.0, 6.0)]),
        );
    }

    #[test]
    fn test_hessian_with_auxiliary_outputs() {
        let evaluations = Cell::new(0);
        let (ordinary_hessian, auxiliary) = differentiate_at(Array::scalar(2.0))
            .with_auxiliary()
            .hessian(|x| {
                evaluations.set(evaluations.get() + 1);
                Ok((x.clone() * x.clone(), x))
            })
            .unwrap();
        assert_eq!(evaluations.get(), 1);
        assert_abs_diff_eq!(ordinary_hessian.iter_blocks().next().unwrap().value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_eq!(auxiliary.to_f64s(), vec![2.0]);

        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0));
        let (holomorphic_hessian, auxiliary) = differentiate_at(input.clone())
            .with_auxiliary()
            .holomorphic()
            .hessian(|x| Ok((x.clone() * x.clone() * x.clone(), x)))
            .unwrap();
        assert_eq!(
            holomorphic_hessian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(12.0, 6.0)]),
        );
        assert_eq!(auxiliary, input);
    }

    #[test]
    fn test_hessian_nested_in_jacobian_forward() {
        // For f(x) = x³, the Hessian is f″(x) = 6x. Differentiating that materialized Hessian with a forward
        // Jacobian computes the third derivative f‴(x) = 6.
        let derivative = differentiate_at(Array::scalar(2.0))
            .jacobian_forward(|input| {
                let context = input.context().clone();
                let hessian = context
                    .differentiate_at(input)
                    .hessian(|value| Ok(value.clone() * value.clone() * value))
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))?;
                Ok(hessian.into_values().remove(0))
            })
            .unwrap();
        assert_abs_diff_eq!(derivative.iter_blocks().next().unwrap().value().to_f64s()[0], 6.0, epsilon = 1e-9);
    }
}
