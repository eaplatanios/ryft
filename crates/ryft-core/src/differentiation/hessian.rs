use std::marker::PhantomData;

use ryft_macros::Parameterized;

use crate::contexts::Context;
use crate::differentiation::DifferentiationError;
use crate::differentiation::forward::{DifferentiableOperation, DifferentiationContext, LinearizationTracer};
use crate::differentiation::jacobian::{jacobian_forward_in_context, jacobian_reverse_in_context};
use crate::differentiation::linear::ResidualZeroProvider;
use crate::differentiation::reverse::TransposableOperation;
use crate::differentiation::types::DenseDifferentiableType;
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
            F: FnOnce(
                I::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>>,
            ) -> Result<O, ProgramError>,
            I: Parameterized<
                    Self::Value,
                    To<Self::Value> = I,
                    To<Self::Type>: Clone,
                    To<LinearizationTracer<Self>>: Parameterized<
                        LinearizationTracer<Self>,
                        To<LinearizationTracer<Self>> = I::To<LinearizationTracer<Self>>,
                        To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>> = I::To<
                            LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>,
                        >,
                        To<Self::Type> = I::To<Self::Type>,
                    >,
                    Family: ParameterizedFamily<Self::Type>
                                + ParameterizedFamily<LinearizationTracer<Self>>
                                + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>,
                >,
                >,
            O: Parameterized<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>,
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
        fn $method<
            F: FnOnce(
                I::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>>,
            ) -> Result<(O, A), ProgramError>,
            I: Parameterized<
                    Self::Value,
                    To<Self::Value> = I,
                    To<Self::Type>: Clone,
                    To<LinearizationTracer<Self>>: Parameterized<
                        LinearizationTracer<Self>,
                        To<LinearizationTracer<Self>> = I::To<LinearizationTracer<Self>>,
                        To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>> = I::To<
                            LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>,
                        >,
                        To<Self::Type> = I::To<Self::Type>,
                    >,
                    Family: ParameterizedFamily<Self::Type>
                                + ParameterizedFamily<LinearizationTracer<Self>>
                                + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>,
                >,
                >,
            O: Parameterized<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>,
                    To<Self::Type>: Clone,
                    Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<LinearizationTracer<Self>>,
                >,
            A: Parameterized<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<Self>>>,
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
            let (outer, auxiliary) = jacobian_forward_in_context(
                self,
                |outer_primals| {
                    let nested_context = outer_primals
                        .parameters()
                        .next()
                        .map(Value::execution_domain)
                        .ok_or(DifferentiationError::EmptyInput)?;
                    jacobian_reverse_in_context(&nested_context, function, outer_primals, $holomorphic)
                        .map_err(ProgramError::from)
                },
                primals,
                $holomorphic,
            )?;
            let input_types = outer.input_type().clone();
            let output_types = outer.output_type().output_type().clone();
            let values = outer.into_values();
            let mut value_index = 0;
            for output_type in output_types.parameters() {
                for first_input_type in input_types.parameters() {
                    for second_input_type in input_types.parameters() {
                        <Self::Type as DenseDifferentiableType<Self>>::validate_hessian_block_type(
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
    };
}

/// Extension trait carrying the value-level *Hessian* differentiation transforms on every [`Context`], mirroring
/// how [`JacobianDifferentiate`](crate::JacobianDifferentiate) carries *Jacobian* differentiation transforms.
/// Implementations enumerate every finite input and output coordinate and return structured second-derivative blocks.
/// For a structured function `y = f(x)`, these methods materialize every block of `H_f(x) = ∂J_f(x)/∂x`, where
/// `H_f(x)[k, i, j] = ∂²y[k]/(∂x[i] ∂x[j])`. The inner reverse transform constructs `J_f(x)` by applying the
/// [`Pullback`](crate::Pullback) to packed output-coordinate basis cotangents, and the outer forward transform
/// differentiates every Jacobian entry by applying its [`Pushforward`](crate::Pushforward) to packed input-coordinate
/// basis tangents. The result uses the output-major/first-input-major/second-input-minor [`Hessian`] representation.
///
/// Below we provide a cost model for computing a [`Hessian`] using this forward-over-reverse decomposition.
/// The cost model uses the following notation:
///
///   - `n`: Total input coordinate-space dimension size.
///   - `m`: Total output coordinate-space dimension size.
///   - `T_inner_vjp`: One-time computation needed to evaluate the primal function and construct the inner pullback.
///   - `T_inner_pullback`: Computation needed to propagate one output basis cotangent through the inner pullback.
///   - `T_outer_linearize`: Additional one-time computation needed to linearize the inner reverse-Jacobian
///     materialization and construct the outer pushforward.
///   - `T_outer_pushforward`: Computation needed to propagate one input basis tangent through the outer pushforward,
///     including its differentiated inner reverse-mode work.
///   - `R_hessian`: Memory occupied by nested linearization residuals shared by all packed outer tangent directions.
///   - `M_inner_pullback`: Additional peak intermediate memory needed by one inner pullback direction.
///   - `M_outer_pushforward`: Additional peak intermediate memory needed by one outer pushforward direction.
///
/// The inner reverse transform evaluates an `m`-way packed pullback, and the outer forward transform evaluates
/// an `n`-way packed pushforward through that reverse-Jacobian computation. The derivative work is approximately
/// `O(T_inner_vjp + m · T_inner_pullback + T_outer_linearize + n · T_outer_pushforward)`, and the working memory
/// excluding the result is approximately `O(R_hessian + m · M_inner_pullback + n · M_outer_pushforward)`. The
/// materialized Hessian itself requires the unavoidable `Θ(mn²)` memory. This decomposition is particularly well
/// suited to scalar-output functions where `m = 1`. In that case, reverse mode constructs the gradient with one
/// output cotangent direction, and forward mode differentiates that square input-to-gradient map. Packing may
/// execute directions in parallel but does not change these total-work or storage scalings.
///
/// Ordinary variants require every differentiated input and output parameter to be real. Holomorphic variants require
/// every differentiated parameter to be complex and treat both nested transforms as complex linear.
pub trait HessianDifferentiate:
    Context<
        Type: DenseDifferentiableType<Self>
                  + DenseDifferentiableType<DifferentiationContext<PartialEvaluationContext<Self>>>,
        Operation: PartiallyEvaluatableOperation<Self>
                       + PartiallyEvaluatableOperation<DifferentiationContext<PartialEvaluationContext<Self>>>
                       + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                       + DifferentiableOperation<PartialEvaluationContext<Self>>
                       + DifferentiableOperation<TracingContext<Self::Constant, Self::Operation>>
                       + DifferentiableOperation<
            PartialEvaluationContext<TracingContext<Self::Constant, Self::Operation>>,
        > + DifferentiableOperation<
            PartialEvaluationContext<DifferentiationContext<PartialEvaluationContext<Self>>>,
        > + TransposableOperation<Self::Constant, Self::Operation>
                       + ResidualZeroProvider<Self::Type>
                       + From<AddOperation<Self::Type>>,
    >
{
    define_hessian_function_in_trait!(
        /// Materializes the complete [`Hessian`] using forward-over-reverse differentiation.
        /// Refer to [`hessian`] for the mathematical interpretation and representation.
        hessian,
        delegate = hessian_with_aux,
    );

    define_hessian_function_in_trait!(
        /// Materializes the complete holomorphic [`Hessian`] using forward-over-reverse differentiation.
        /// Refer to [`hessian_holomorphic`] for the holomorphy contract.
        hessian_holomorphic,
        delegate = hessian_holomorphic_with_aux,
    );

    define_hessian_auxiliary_function_in_trait!(
        /// Materializes the complete [`Hessian`] and returns nondifferentiated auxiliary outputs.
        /// Refer to [`hessian_with_aux`] for details.
        hessian_with_aux,
        holomorphic = false,
    );

    define_hessian_auxiliary_function_in_trait!(
        /// Materializes the complete holomorphic [`Hessian`] and returns nondifferentiated auxiliary outputs.
        /// Refer to [`hessian_holomorphic_with_aux`] for details.
        hessian_holomorphic_with_aux,
        holomorphic = true,
    );
}

impl<C> HessianDifferentiate for C
where
    C: Context,
    C::Type: DenseDifferentiableType<C> + DenseDifferentiableType<DifferentiationContext<PartialEvaluationContext<C>>>,
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<DifferentiationContext<PartialEvaluationContext<C>>>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<C>>
        + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
        + DifferentiableOperation<PartialEvaluationContext<DifferentiationContext<PartialEvaluationContext<C>>>>
        + TransposableOperation<C::Constant, C::Operation>
        + ResidualZeroProvider<C::Type>
        + From<AddOperation<C::Type>>,
{
}

/// Defines one context-recovering Hessian function without auxiliary outputs. It centralizes the nested structured
/// generic signature, operation requirements, empty-input handling, and same-named context-method delegation.
macro_rules! define_hessian_function {
    (
        $(#[doc = $documentation:literal])*
        $function_name:ident,
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        pub fn $function_name<C, V, F, I, O>(
            function: F,
            primals: I,
        ) -> Result<Hessian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, DifferentiationError>
        where
            C: HessianDifferentiate<Type = V::Type, Value = V, Operation: ResidualZeroProvider<V::Type>>,
            V: Value<
                Type: DenseDifferentiableType<DifferentiationContext<PartialEvaluationContext<C>>>,
                ExecutionDomain = C,
            >,
            F: FnOnce(I::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>)
                -> Result<O, ProgramError>,
            I: Parameterized<V, To<V> = I>,
            I::Family: ParameterizedFamily<V::Type>
                + ParameterizedFamily<LinearizationTracer<C>>
                + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
            I::To<V::Type>: Clone,
            I::To<LinearizationTracer<C>>: Parameterized<
                LinearizationTracer<C>,
                To<LinearizationTracer<C>> = I::To<LinearizationTracer<C>>,
                To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = I::To<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
                To<V::Type> = I::To<V::Type>,
            >,
            O: Parameterized<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
            O::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<LinearizationTracer<C>>,
            O::To<V::Type>: Clone,
        {
            let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
                return Err(DifferentiationError::EmptyInput);
            };
            context.$function_name(function, primals)
        }
    };
}

/// Defines one context-recovering Hessian function with auxiliary outputs. It centralizes the nested structured
/// generic signature, operation requirements, empty-input handling, and same-named context-method delegation.
macro_rules! define_hessian_auxiliary_function {
    (
        $(#[doc = $documentation:literal])*
        $function_name:ident,
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        pub fn $function_name<C, V, F, I, O, A>(
            function: F,
            primals: I,
        ) -> Result<(Hessian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, A::To<V>), DifferentiationError>
        where
            C: HessianDifferentiate<Type = V::Type, Value = V, Operation: ResidualZeroProvider<V::Type>>,
            V: Value<
                Type: DenseDifferentiableType<DifferentiationContext<PartialEvaluationContext<C>>>,
                ExecutionDomain = C,
            >,
            F: FnOnce(I::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>)
                -> Result<(O, A), ProgramError>,
            I: Parameterized<V, To<V> = I>,
            I::Family: ParameterizedFamily<V::Type>
                + ParameterizedFamily<LinearizationTracer<C>>
                + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
            I::To<V::Type>: Clone,
            I::To<LinearizationTracer<C>>: Parameterized<
                LinearizationTracer<C>,
                To<LinearizationTracer<C>> = I::To<LinearizationTracer<C>>,
                To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = I::To<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
                To<V::Type> = I::To<V::Type>,
            >,
            O: Parameterized<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
            O::Family: ParameterizedFamily<V::Type> + ParameterizedFamily<LinearizationTracer<C>>,
            O::To<V::Type>: Clone,
            A: Parameterized<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
            A::Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<V>,
            A::To<LinearizationTracer<C>>: Parameterized<LinearizationTracer<C>, To<V> = A::To<V>>,
        {
            let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
                return Err(DifferentiationError::EmptyInput);
            };
            context.$function_name(function, primals)
        }
    };
}

define_hessian_function!(
    /// Computes the [`Hessian`] of `function` at `primals` using forward-over-reverse differentiation.
    /// For `y = f(x)`, each Hessian entry is `H[k, i, j] = ∂²y[k] / (∂x[i] ∂x[j])`. This function first uses
    /// [`jacobian_reverse`](crate::jacobian_reverse) to materialize the inner output/first-input Jacobian, and then
    /// uses [`jacobian_forward`](crate::jacobian_forward) to differentiate it with respect to the second input. The
    /// resulting [`Hessian`] stores blocks in output-major/first-input-major/second-input-minor order. For arrays,
    /// a block places the output axes first, followed by the first-input axes and then the second-input axes.
    ///
    /// The active context is recovered from the first value in `primals`, so the same entry point works for eager
    /// values and staged tracers. Complete materialization requires finite, statically enumerable coordinate spaces and
    /// ordinary Hessians require real input and output leaves. Use [`hessian_holomorphic`] for a complex holomorphic
    /// function.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose Hessian is materialized.
    ///   - `primals`: Structured input values specifying the differentiation point.
    hessian,
);

define_hessian_function!(
    /// Computes the [`Hessian`] of a complex holomorphic `function` at `primals` using forward-over-reverse
    /// differentiation. This has the algorithm and representation described by [`hessian`], but treats both derivative
    /// transforms as complex linear and requires every differentiated input and output parameter to be complex. Passing
    /// `function` is a promise of holomorphy. It validates the parameter types but cannot prove that the function
    /// satisfies the [Cauchy-Riemann equations](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations).
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose Hessian is materialized.
    ///   - `primals`: Structured input values specifying the differentiation point.
    hessian_holomorphic,
);

define_hessian_auxiliary_function!(
    /// Computes a [`Hessian`] and returns nondifferentiated auxiliary outputs. The closure returns
    /// `(output, auxiliary)`. Only `output` contributes to the Hessian. `auxiliary` is materialized from its primal
    /// trace and returned with it. Refer to [`hessian`] for the mathematical interpretation, block layout, context
    /// recovery, and ordinary complex-type rules.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose Hessian is materialized.
    ///   - `primals`: Structured input values specifying the differentiation point.
    hessian_with_aux,
);

define_hessian_auxiliary_function!(
    /// Computes a holomorphic [`Hessian`] and returns nondifferentiated auxiliary outputs. The closure and auxiliary
    /// output behavior are described by [`hessian_with_aux`]. The holomorphy promise and complex-type requirements are
    /// the same as for [`hessian_holomorphic`].
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose Hessian is materialized.
    ///   - `primals`: Structured input values specifying the differentiation point.
    hessian_holomorphic_with_aux,
);

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use approx::assert_abs_diff_eq;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::differentiation::jacobian::jacobian_forward;
    use crate::operations::math::Sin;
    use crate::parameters::{ParameterPath, Parameterized};
    use crate::programs::types::Typed;
    use crate::types::DataType;
    use crate::types::DataType::{F32, F64};
    use crate::types::{ArrayType, Dimension, Shape};

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
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], -2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[2].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[3].value().to_f64s()[0], 0.0, epsilon = 1e-9);

        // Narrow primal element types use their widened differential representation for dense Hessian blocks.
        let input = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let widened_hessian = hessian(|value| value.sin(), input).unwrap();
        let block = widened_hessian.iter_blocks().next().unwrap();
        assert_eq!(block.value().r#type().as_ref(), &ArrayType::scalar(F32));
        assert_abs_diff_eq!(block.value().to_f64s()[0], -2.0f64.sin(), epsilon = 1e-6);

        // Zero-sized inputs and outputs remain concrete, honestly typed dense blocks.
        let r#type = ArrayType::new(F64, Shape::new(vec![Dimension::Static(0)]));
        let zero_sized_hessian =
            hessian(|input| Ok(input.clone() * input), Array::from_f64s(r#type, Vec::new())).unwrap();
        let block = zero_sized_hessian.iter_blocks().next().unwrap();
        assert_eq!(block.value().r#type().static_shape().unwrap().as_slice(), &[0, 0, 0]);
        assert!(block.value().storage_bytes().is_empty());

        // Structured outputs retain a distinct Hessian block for each output leaf.
        let structured_hessian =
            hessian(|x| Ok((x.clone() * x.clone(), x.clone() * x.clone() * x)), Array::scalar(2.0)).unwrap();
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
        let holomorphic_hessian = hessian_holomorphic(|x| Ok(x.clone() * x.clone() * x), input).unwrap();
        assert_eq!(
            holomorphic_hessian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(12.0, 6.0)]),
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
        assert_abs_diff_eq!(ordinary_hessian.iter_blocks().next().unwrap().value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_eq!(auxiliary.to_f64s(), vec![2.0]);

        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0));
        let (holomorphic_hessian, auxiliary) =
            hessian_holomorphic_with_aux(|x| Ok((x.clone() * x.clone() * x.clone(), x)), input.clone()).unwrap();
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
        let derivative = jacobian_forward(
            |input| {
                let hessian = input
                    .context()
                    .clone()
                    .hessian(|value| Ok(value.clone() * value.clone() * value), input)
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))?;
                Ok(hessian.into_values().remove(0))
            },
            Array::scalar(2.0),
        )
        .unwrap();
        assert_abs_diff_eq!(derivative.iter_blocks().next().unwrap().value().to_f64s()[0], 6.0, epsilon = 1e-9);
    }
}
