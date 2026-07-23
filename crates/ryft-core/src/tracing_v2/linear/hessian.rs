use std::marker::PhantomData;

use ryft_macros::Parameterized;

use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationError;
use crate::differentiation::forward::{DifferentiableOperation, DifferentiationContext, LinearizationTracer};
use crate::differentiation::jacobian::{Jacobian, jacobian_forward_in_context, jacobian_reverse_in_context};
use crate::differentiation::reverse::TransposableOperation;
use crate::differentiation::types::DenseDifferentiableType;
use crate::operations::constants::ZeroOperation;
use crate::operations::math::AddOperation;
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::types::{Type, Typed};
use crate::programs::values::Value;
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

/// Defines one non-auxiliary [`HessianDifferentiate`] method. It keeps the nested differentiation bounds shared while
/// adapting its corresponding auxiliary method with a unit auxiliary value.
macro_rules! define_hessian_function_in_trait {
    (
        $(#[doc = $documentation:literal])*
        $method:ident,
        delegate = $delegate:ident,
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        fn $method<
            F: FnOnce(I::To<LinearizationTracer<NestedDenseContext<Self>>>) -> Result<O, ProgramError>,
            I: Parameterized<
                    Self::Value,
                    To<Self::Value> = I,
                    To<Self::Type>: Clone,
                    To<LinearizationTracer<Self>>: Parameterized<
                        LinearizationTracer<Self>,
                        To<LinearizationTracer<Self>> = I::To<LinearizationTracer<Self>>,
                        To<LinearizationTracer<NestedDenseContext<Self>>> = I::To<
                            LinearizationTracer<NestedDenseContext<Self>>,
                        >,
                        To<Self::Type> = I::To<Self::Type>,
                    >,
                    Family: ParameterizedFamily<Self::Type>
                                + ParameterizedFamily<LinearizationTracer<Self>>
                                + ParameterizedFamily<LinearizationTracer<NestedDenseContext<Self>>>,
                >,
            O: Parameterized<
                    LinearizationTracer<NestedDenseContext<Self>>,
                    To<Self::Type>: Clone,
                    Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<LinearizationTracer<Self>>,
                >,
        >(
            &self,
            function: F,
            primals: I,
        ) -> Result<Hessian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, DifferentiationError> {
            let (hessian, ()) = self.$delegate(|input| Ok((function(input)?, ())), primals)?;
            Ok(hessian)
        }
    };
}

/// Defines one auxiliary-output [`HessianDifferentiate`] method. It centralizes the nested auxiliary parameter bounds
/// so the ordinary and holomorphic variants cannot drift apart.
macro_rules! define_hessian_auxiliary_function_in_trait {
    (
        $(#[doc = $documentation:literal])*
        $method:ident,
        holomorphic = $holomorphic:literal,
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        fn $method<
            F: FnOnce(I::To<LinearizationTracer<NestedDenseContext<Self>>>) -> Result<(O, A), ProgramError>,
            I: Parameterized<
                    Self::Value,
                    To<Self::Value> = I,
                    To<Self::Type>: Clone,
                    To<LinearizationTracer<Self>>: Parameterized<
                        LinearizationTracer<Self>,
                        To<LinearizationTracer<Self>> = I::To<LinearizationTracer<Self>>,
                        To<LinearizationTracer<NestedDenseContext<Self>>> = I::To<
                            LinearizationTracer<NestedDenseContext<Self>>,
                        >,
                        To<Self::Type> = I::To<Self::Type>,
                    >,
                    Family: ParameterizedFamily<Self::Type>
                                + ParameterizedFamily<LinearizationTracer<Self>>
                                + ParameterizedFamily<LinearizationTracer<NestedDenseContext<Self>>>,
                >,
            O: Parameterized<
                    LinearizationTracer<NestedDenseContext<Self>>,
                    To<Self::Type>: Clone,
                    Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<LinearizationTracer<Self>>,
                >,
            A: Parameterized<
                    LinearizationTracer<NestedDenseContext<Self>>,
                    To<LinearizationTracer<Self>>: Parameterized<
                        LinearizationTracer<Self>,
                        To<Self::Value> = A::To<Self::Value>,
                    >,
                    Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Value>,
                >,
        >(
            &self,
            function: F,
            primals: I,
        ) -> Result<
            (Hessian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, A::To<Self::Value>),
            DifferentiationError,
        > {
            hessian_in_context(self, function, primals, $holomorphic)
        }
    };
}

// TODO(eaplatanios): Review from here onwards.

/// Extension trait for materializing dense Hessians in an execution or staging [`Context`].
///
/// For a structured function `y = f(x)`, these methods materialize every block of the derivative of the Jacobian,
/// `H_f(x) = ∂J_f/∂x`, using forward-over-reverse differentiation. The inner reverse transform constructs `J_f(x)`;
/// the outer forward transform differentiates every one of its entries. Ordinary variants require real input and
/// output leaves, while holomorphic variants require every differentiated leaf to be complex.
pub trait HessianDifferentiate:
    Context<
        Type: DenseDifferentiableType<Self> + DenseDifferentiableType<NestedDenseContext<Self>>,
        Operation: PartiallyEvaluatableOperation<Self>
                       + PartiallyEvaluatableOperation<NestedDenseContext<Self>>
                       + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                       + DifferentiableOperation<PartialEvaluationContext<Self>>
                       + DifferentiableOperation<TracingContext<Self::Constant, Self::Operation>>
                       + DifferentiableOperation<
            PartialEvaluationContext<TracingContext<Self::Constant, Self::Operation>>,
        > + DifferentiableOperation<PartialEvaluationContext<NestedDenseContext<Self>>>
                       + TransposableOperation<Self::Constant, Self::Operation>
                       + From<ZeroOperation<Self::Type>>
                       + From<AddOperation>,
    >
{
    define_hessian_function_in_trait!(
        /// Materializes the complete output/input/input Hessian using forward-over-reverse differentiation. Refer to
        /// [`hessian`] for the mathematical interpretation and representation.
        hessian,
        delegate = hessian_with_aux,
    );

    define_hessian_function_in_trait!(
        /// Materializes the complete holomorphic Hessian using forward-over-reverse differentiation. Refer to
        /// [`hessian_holomorphic`] for the holomorphy contract.
        hessian_holomorphic,
        delegate = hessian_holomorphic_with_aux,
    );

    define_hessian_auxiliary_function_in_trait!(
        /// Materializes the complete Hessian and returns nondifferentiated auxiliary outputs. Refer to
        /// [`hessian_with_aux`] for details.
        hessian_with_aux,
        holomorphic = false,
    );

    define_hessian_auxiliary_function_in_trait!(
        /// Materializes the complete holomorphic Hessian and returns nondifferentiated auxiliary outputs. Refer to
        /// [`hessian_holomorphic_with_aux`] for details.
        hessian_holomorphic_with_aux,
        holomorphic = true,
    );
}

impl<C> HessianDifferentiate for C
where
    C: Context,
    C::Type: DenseDifferentiableType<C> + DenseDifferentiableType<NestedDenseContext<C>>,
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<NestedDenseContext<C>>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<C>>
        + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
        + DifferentiableOperation<PartialEvaluationContext<NestedDenseContext<C>>>
        + TransposableOperation<C::Constant, C::Operation>
        + From<ZeroOperation<C::Type>>
        + From<AddOperation>,
{
}

/// Nested differentiation context used by the inner reverse-mode Jacobian of a forward-over-reverse Hessian.
pub(super) type NestedDenseContext<C> = DifferentiationContext<PartialEvaluationContext<C>>;

/// Converts the Jacobian of an inner Jacobian into the canonical [`Hessian`] representation and validates every
/// output/first-input/second-input block type.
///
/// # Parameters
///
///   - `outer`: Forward Jacobian whose output is the reverse Jacobian of the original function.
fn hessian_from_outer<
    C: Context<Type: DenseDifferentiableType<C>>,
    I: Clone + Parameterized<C::Type>,
    O: Clone + Parameterized<C::Type>,
>(
    outer: Jacobian<C::Type, C::Value, I, Jacobian<C::Type, C::Type, I, O>>,
) -> Result<Hessian<C::Type, C::Value, I, O>, DifferentiationError> {
    let input_types = outer.input_type().clone();
    let output_types = outer.output_type().output_type().clone();
    let values = outer.into_values();
    let mut value_index = 0;
    for output_type in output_types.parameters() {
        for first_input_type in input_types.parameters() {
            for second_input_type in input_types.parameters() {
                C::Type::validate_hessian_block_type(
                    values[value_index].r#type().as_ref(),
                    output_type,
                    first_input_type,
                    second_input_type,
                )?;
                value_index += 1;
            }
        }
    }
    Ok(Hessian::new(input_types, output_types, values)?)
}

/// Implements forward-over-reverse Hessian materialization in an explicitly provided [`Context`]. The inner reverse
/// transform materializes the original Jacobian, and the outer forward transform differentiates that structured
/// Jacobian with respect to every input coordinate. Only the first component returned by `function` is differentiated;
/// the auxiliary component is materialized from its primal trace and returned unchanged.
///
/// # Parameters
///
///   - `context`: Context in which to trace and replay both derivative transforms.
///   - `function`: Function returning the differentiated output and auxiliary output.
///   - `primals`: Structured input values specifying the linearization point.
///   - `holomorphic`: Whether to validate all differentiated leaves under a holomorphy promise.
pub(super) fn hessian_in_context<
    C: Context<
            Type: DenseDifferentiableType<C> + DenseDifferentiableType<NestedDenseContext<C>>,
            Operation: PartiallyEvaluatableOperation<C>
                           + PartiallyEvaluatableOperation<NestedDenseContext<C>>
                           + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
                           + DifferentiableOperation<PartialEvaluationContext<C>>
                           + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
                           + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
                           + DifferentiableOperation<PartialEvaluationContext<NestedDenseContext<C>>>
                           + TransposableOperation<C::Constant, C::Operation>
                           + From<ZeroOperation<C::Type>>
                           + From<AddOperation>,
        >,
    F: FnOnce(I::To<LinearizationTracer<NestedDenseContext<C>>>) -> Result<(O, A), ProgramError>,
    I: Parameterized<
            C::Value,
            To<C::Value> = I,
            To<C::Type>: Clone,
            To<LinearizationTracer<C>>: Parameterized<
                LinearizationTracer<C>,
                To<LinearizationTracer<C>> = I::To<LinearizationTracer<C>>,
                To<LinearizationTracer<NestedDenseContext<C>>> = I::To<LinearizationTracer<NestedDenseContext<C>>>,
                To<C::Type> = I::To<C::Type>,
            >,
            Family: ParameterizedFamily<C::Type>
                        + ParameterizedFamily<LinearizationTracer<C>>
                        + ParameterizedFamily<LinearizationTracer<NestedDenseContext<C>>>,
        >,
    O: Parameterized<
            LinearizationTracer<NestedDenseContext<C>>,
            To<C::Type>: Clone,
            Family: ParameterizedFamily<C::Type> + ParameterizedFamily<LinearizationTracer<C>>,
        >,
    A: Parameterized<
            LinearizationTracer<NestedDenseContext<C>>,
            To<LinearizationTracer<C>>: Parameterized<LinearizationTracer<C>, To<C::Value> = A::To<C::Value>>,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Value>,
        >,
>(
    context: &C,
    function: F,
    primals: I,
    holomorphic: bool,
) -> Result<(Hessian<C::Type, C::Value, I::To<C::Type>, O::To<C::Type>>, A::To<C::Value>), DifferentiationError> {
    let (outer, auxiliary): (
        Jacobian<C::Type, C::Value, I::To<C::Type>, Jacobian<C::Type, C::Type, I::To<C::Type>, O::To<C::Type>>>,
        A::To<C::Value>,
    ) = jacobian_forward_in_context(
        context,
        |outer_primals| {
            let nested_context = outer_primals
                .parameters()
                .next()
                .map(Value::execution_domain)
                .ok_or(DifferentiationError::EmptyInput)?;
            jacobian_reverse_in_context(&nested_context, function, outer_primals, holomorphic)
                .map_err(ProgramError::from)
        },
        primals,
        holomorphic,
    )?;

    Ok((hessian_from_outer::<C, _, _>(outer)?, auxiliary))
}

/// Defines one context-recovering Hessian function without auxiliary outputs. It centralizes the nested structured
/// generic signature, operation requirements, empty-input handling, and same-named context-method delegation.
macro_rules! define_hessian_function {
    // Generates one non-auxiliary Hessian function and delegates to its same-named context method.
    (
        $(#[doc = $documentation:literal])*
        $function_name:ident,
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        pub fn $function_name<
            V: Value<
                    Type: DenseDifferentiableType<V::ExecutionDomain>
                              + DenseDifferentiableType<NestedDenseContext<V::ExecutionDomain>>,
                    ExecutionDomain: Context<
                        Operation: PartiallyEvaluatableOperation<V::ExecutionDomain>
                                       + PartiallyEvaluatableOperation<NestedDenseContext<V::ExecutionDomain>>
                                       + PartiallyEvaluatableOperation<
                            TracingContext<
                                <V::ExecutionDomain as Domain>::Constant,
                                <V::ExecutionDomain as Domain>::Operation,
                            >,
                        > + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                                       + DifferentiableOperation<
                            TracingContext<
                                <V::ExecutionDomain as Domain>::Constant,
                                <V::ExecutionDomain as Domain>::Operation,
                            >,
                        > + DifferentiableOperation<
                            PartialEvaluationContext<
                                TracingContext<
                                    <V::ExecutionDomain as Domain>::Constant,
                                    <V::ExecutionDomain as Domain>::Operation,
                                >,
                            >,
                        > + DifferentiableOperation<PartialEvaluationContext<NestedDenseContext<V::ExecutionDomain>>>
                                       + TransposableOperation<
                            <V::ExecutionDomain as Domain>::Constant,
                            <V::ExecutionDomain as Domain>::Operation,
                        > + From<ZeroOperation<V::Type>>
                                       + From<AddOperation>,
                    >,
                >,
            F: FnOnce(
                I::To<LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>>,
            ) -> Result<O, ProgramError>,
            I: Parameterized<
                    V,
                    To<V> = I,
                    To<V::Type>: Clone,
                    To<LinearizationTracer<V::ExecutionDomain>>: Parameterized<
                        LinearizationTracer<V::ExecutionDomain>,
                        To<LinearizationTracer<V::ExecutionDomain>> = I::To<
                            LinearizationTracer<V::ExecutionDomain>,
                        >,
                        To<LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>> = I::To<
                            LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>,
                        >,
                        To<V::Type> = I::To<V::Type>,
                    >,
                    Family: ParameterizedFamily<V::Type>
                                + ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>
                                + ParameterizedFamily<LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>>,
                >,
            O: Parameterized<
                    LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>,
                    To<V::Type>: Clone,
                    Family: ParameterizedFamily<V::Type> + ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>,
                >,
        >(
            function: F,
            primals: I,
        ) -> Result<Hessian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, DifferentiationError> {
            let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
                return Err(DifferentiationError::EmptyInput);
            };
            context.$function_name(function, primals)
        }
    };
}

define_hessian_function!(
    /// Materializes the complete Hessian of `function` at `primals` using forward-over-reverse differentiation.
    ///
    /// For `y = f(x)`, each Hessian entry is `H[k, i, j] = ∂²y[k]/(∂x[i] ∂x[j])`. Ryft first uses
    /// [`jacobian_reverse`](crate::tracing_v2::jacobian_reverse) to materialize the inner output/first-input Jacobian,
    /// then uses [`jacobian_forward`](crate::tracing_v2::jacobian_forward) to differentiate it with respect to the
    /// second input. The resulting [`Hessian`] stores blocks in output-major/first-input-major/second-input-minor order.
    /// For arrays, a block places the output axes first, followed by the first-input axes and then the second-input axes.
    ///
    /// The active context is recovered from the first value in `primals`, so the same entry point works for eager
    /// values and staged tracers. Complete materialization requires finite, statically enumerable coordinate spaces and
    /// ordinary Hessians require real input and output leaves. Use [`hessian_holomorphic`] for a complex holomorphic
    /// function.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose Hessian is materialized.
    ///   - `primals`: Structured input values specifying the evaluation point.
    hessian,
);

define_hessian_function!(
    /// Materializes the Hessian of a complex holomorphic `function` at `primals` using forward-over-reverse
    /// differentiation.
    ///
    /// This has the algorithm and representation described by [`hessian`], but treats both derivative transforms as
    /// complex linear and requires every differentiated input and output leaf to be complex. Passing `function` is a
    /// promise of holomorphy; Ryft validates the leaf types but cannot prove that the function satisfies the
    /// Cauchy-Riemann equations.
    ///
    /// # Parameters
    ///
    ///   - `function`: Holomorphic function whose complex Hessian is materialized.
    ///   - `primals`: Structured complex input values specifying the evaluation point.
    hessian_holomorphic,
);

/// Defines one context-recovering Hessian function with auxiliary outputs. It centralizes the nested structured generic
/// signature, operation requirements, empty-input handling, and same-named context-method delegation.
macro_rules! define_hessian_auxiliary_function {
    // Generates one auxiliary-output Hessian function and delegates to its same-named context method.
    (
        $(#[doc = $documentation:literal])*
        $function_name:ident,
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        pub fn $function_name<
            V: Value<
                    Type: DenseDifferentiableType<V::ExecutionDomain>
                              + DenseDifferentiableType<NestedDenseContext<V::ExecutionDomain>>,
                    ExecutionDomain: Context<
                        Operation: PartiallyEvaluatableOperation<V::ExecutionDomain>
                                       + PartiallyEvaluatableOperation<NestedDenseContext<V::ExecutionDomain>>
                                       + PartiallyEvaluatableOperation<
                            TracingContext<
                                <V::ExecutionDomain as Domain>::Constant,
                                <V::ExecutionDomain as Domain>::Operation,
                            >,
                        > + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                                       + DifferentiableOperation<
                            TracingContext<
                                <V::ExecutionDomain as Domain>::Constant,
                                <V::ExecutionDomain as Domain>::Operation,
                            >,
                        > + DifferentiableOperation<
                            PartialEvaluationContext<
                                TracingContext<
                                    <V::ExecutionDomain as Domain>::Constant,
                                    <V::ExecutionDomain as Domain>::Operation,
                                >,
                            >,
                        > + DifferentiableOperation<PartialEvaluationContext<NestedDenseContext<V::ExecutionDomain>>>
                                       + TransposableOperation<
                            <V::ExecutionDomain as Domain>::Constant,
                            <V::ExecutionDomain as Domain>::Operation,
                        > + From<ZeroOperation<V::Type>>
                                       + From<AddOperation>,
                    >,
                >,
            F: FnOnce(
                I::To<LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>>,
            ) -> Result<(O, A), ProgramError>,
            I: Parameterized<
                    V,
                    To<V> = I,
                    To<V::Type>: Clone,
                    To<LinearizationTracer<V::ExecutionDomain>>: Parameterized<
                        LinearizationTracer<V::ExecutionDomain>,
                        To<LinearizationTracer<V::ExecutionDomain>> = I::To<
                            LinearizationTracer<V::ExecutionDomain>,
                        >,
                        To<LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>> = I::To<
                            LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>,
                        >,
                        To<V::Type> = I::To<V::Type>,
                    >,
                    Family: ParameterizedFamily<V::Type>
                                + ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>
                                + ParameterizedFamily<LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>>,
                >,
            O: Parameterized<
                    LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>,
                    To<V::Type>: Clone,
                    Family: ParameterizedFamily<V::Type> + ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>,
                >,
            A: Parameterized<
                    LinearizationTracer<NestedDenseContext<V::ExecutionDomain>>,
                    To<LinearizationTracer<V::ExecutionDomain>>: Parameterized<
                        LinearizationTracer<V::ExecutionDomain>,
                        To<V> = A::To<V>,
                    >,
                    Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>> + ParameterizedFamily<V>,
                >,
        >(
            function: F,
            primals: I,
        ) -> Result<(Hessian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, A::To<V>), DifferentiationError> {
            let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
                return Err(DifferentiationError::EmptyInput);
            };
            context.$function_name(function, primals)
        }
    };
}

define_hessian_auxiliary_function!(
    /// Materializes a Hessian and returns nondifferentiated auxiliary outputs.
    ///
    /// The closure returns `(output, auxiliary)`. Only `output` contributes to the Hessian; `auxiliary` is materialized
    /// from its primal trace and returned with it. Refer to [`hessian`] for the mathematical interpretation, block
    /// layout, context recovery, and ordinary complex-type rules.
    hessian_with_aux,
);

define_hessian_auxiliary_function!(
    /// Materializes a holomorphic Hessian and returns nondifferentiated auxiliary outputs.
    ///
    /// The closure and auxiliary-output behavior are described by [`hessian_with_aux`]. The holomorphy promise and
    /// complex-type requirements are the same as for [`hessian_holomorphic`].
    hessian_holomorphic_with_aux,
);

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use approx::assert_abs_diff_eq;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::backends::scalars::Scalar;
    use crate::operations::math::Sin;
    use crate::parameters::{ParameterPath, Parameterized};
    use crate::programs::types::Typed;
    use crate::types::DataType;
    use crate::types::DataType::{F32, F64};

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
        let scalar_hessian =
            hessian(|(x, y)| Ok(x.clone() * y + x.sin()?), (Array::scalar(2.0), Array::scalar(3.0))).unwrap();
        let blocks = scalar_hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 4);
        assert_abs_diff_eq!(blocks[0].value().values()[0], -2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().values()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[2].value().values()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[3].value().values()[0], 0.0, epsilon = 1e-9);

        // Structured outputs retain a distinct Hessian block for each output leaf.
        let structured_hessian =
            hessian(|x| Ok((x.clone() * x.clone(), x.clone() * x.clone() * x)), Array::scalar(2.0)).unwrap();
        let blocks = structured_hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].first_input_path().to_string(), "$");
        assert_eq!(blocks[0].second_input_path().to_string(), "$");
        assert_abs_diff_eq!(blocks[0].value().values()[0], 2.0, epsilon = 1e-9);
        assert_eq!(blocks[1].output_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[1].value().values()[0], 12.0, epsilon = 1e-9);

        // Mixed-rank structured inputs materialize the entire block Cartesian product with output axes leading both
        // input-axis groups.
        let mixed_rank_hessian = hessian(
            |(vector, scalar)| Ok((vector.clone() * vector, scalar.clone() * scalar)),
            (Array::vector(vec![1.0, 2.0]), Array::scalar(3.0)),
        )
        .unwrap();
        let blocks = mixed_rank_hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 8);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].first_input_path().to_string(), "$.0");
        assert_eq!(blocks[0].second_input_path().to_string(), "$.0");
        assert_eq!(blocks[0].value().r#type().static_shape().unwrap().as_slice(), &[2, 2, 2]);
        assert_eq!(blocks[0].value().values(), &[2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
        assert_eq!(blocks[1].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[1].value().values(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(blocks[2].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[2].value().values(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(blocks[3].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[3].value().values(), &[0.0, 0.0]);
        assert_eq!(blocks[4].output_path().to_string(), "$.1");
        assert_eq!(blocks[4].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[4].value().values(), &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(blocks[5].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[5].value().values(), &[0.0, 0.0]);
        assert_eq!(blocks[6].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[6].value().values(), &[0.0, 0.0]);
        assert_eq!(blocks[7].first_input_path().to_string(), "$.1");
        assert_eq!(blocks[7].second_input_path().to_string(), "$.1");
        assert!(blocks[7].value().r#type().static_shape().unwrap().as_slice().is_empty());
        assert_eq!(blocks[7].value().values(), &[2.0]);

        // Holomorphic Hessians remain complex linear at both derivative levels.
        let input = Array::scalar(Scalar::C64(ComplexNumber::new(2.0, 1.0)));
        let holomorphic_hessian = hessian_holomorphic(|x| Ok(x.clone() * x.clone() * x), input).unwrap();
        assert_eq!(
            holomorphic_hessian.iter_blocks().next().unwrap().value().values(),
            &[Scalar::C64(ComplexNumber::new(12.0, 6.0))],
        );
    }

    #[test]
    fn test_hessian_with_auxiliary_outputs() {
        let evaluations = Cell::new(0);
        let (ordinary_hessian, auxiliary) = hessian_with_aux(
            |x| {
                evaluations.set(evaluations.get() + 1);
                Ok((x.clone() * x.clone(), x))
            },
            Array::scalar(2.0),
        )
        .unwrap();
        assert_eq!(evaluations.get(), 1);
        assert_abs_diff_eq!(ordinary_hessian.iter_blocks().next().unwrap().value().values()[0], 2.0, epsilon = 1e-9);
        assert_eq!(auxiliary.values(), &[2.0]);

        let input = Array::scalar(Scalar::C64(ComplexNumber::new(2.0, 1.0)));
        let (holomorphic_hessian, auxiliary) =
            hessian_holomorphic_with_aux(|x| Ok((x.clone() * x.clone() * x.clone(), x)), input.clone()).unwrap();
        assert_eq!(
            holomorphic_hessian.iter_blocks().next().unwrap().value().values(),
            &[Scalar::C64(ComplexNumber::new(12.0, 6.0))],
        );
        assert_eq!(auxiliary.values(), input.values());
    }
}
