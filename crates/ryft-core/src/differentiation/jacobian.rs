use std::marker::PhantomData;

use ryft_macros::Parameterized;

use crate::contexts::{Context, Domain};
use crate::differentiation::forward::{DifferentiableOperation, ForwardModeDifferentiate, LinearizationTracer};
use crate::differentiation::linear::ResidualZeroProvider;
use crate::differentiation::reverse::{ReverseModeDifferentiate, TransposableOperation};
use crate::differentiation::types::{DenseDifferentiableType, DifferentiableType};
use crate::differentiation::{DerivativeTransform, DifferentiationError, DifferentiationParameterRole};
use crate::macros::check_count;
use crate::operations::AddOperation;
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{ProgramError, Type, Typed, Value};
use crate::tracing::TracingContext;

/// Jacobian of a function, represented as the Cartesian product of its output and input [`Parameter`] leaves. `I` and
/// `O` retain the input and output [`Type`] trees. Derivative values are stored in deterministic output-major /
/// input-minor order and remain [`Parameter`]s so that the complete Jacobian can cross tracing and compilation
/// boundaries as well as participate in higher-order transforms. The physical representation of a block is defined by
/// [`DenseDifferentiableType`]. For [`ArrayType`](crate::ArrayType), the block for an output leaf with shape `O` and
/// an input leaf with shape `I` has shape `O` concatenated with `I`.
#[derive(Clone, Debug, Parameterized)]
pub struct Jacobian<T: Type, V: Parameter, I: Parameterized<T>, O: Parameterized<T>> {
    /// [`Type`] of the differentiated inputs.
    input_type: I,

    /// [`Type`] of the differentiated outputs.
    output_type: O,

    /// Derivative values in output-major/input-minor order.
    values: Vec<V>,

    /// [`PhantomData`] marker for `T`, needed because the input and output fields use `T` only through their bounds.
    _type: PhantomData<fn() -> T>,
}

impl<T: Type, V: Parameter, I: Parameterized<T>, O: Parameterized<T>> Jacobian<T, V, I, O> {
    /// Creates a new [`Jacobian`].
    pub fn new(input_type: I, output_type: O, values: Vec<V>) -> Result<Self, ProgramError> {
        let input_count = input_type.parameter_count();
        let output_count = output_type.parameter_count();
        let expected_count = input_count.checked_mul(output_count).ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("Jacobian block count ({input_count} x {output_count}) overflows usize"),
        })?;
        if values.len() != expected_count {
            return Err(ProgramError::InvalidArgument {
                message: format!("Jacobian requires {} derivative values but got {}", expected_count, values.len()),
            });
        }
        Ok(Self { input_type, output_type, values, _type: PhantomData })
    }

    /// Returns the [`Type`] of the differentiated inputs.
    #[inline]
    pub fn input_type(&self) -> &I {
        &self.input_type
    }

    /// Returns the [`Type`] of the differentiated outputs.
    #[inline]
    pub fn output_type(&self) -> &O {
        &self.output_type
    }

    /// Returns the derivative values in output-major/input-minor order.
    #[inline]
    pub fn values(&self) -> &[V] {
        self.values.as_slice()
    }

    /// Consumes this [`Jacobian`] and returns its derivative values in output-major/input-minor order.
    #[inline]
    pub fn into_values(self) -> Vec<V> {
        self.values
    }

    /// Returns the [`JacobianBlock`] of this [`Jacobian`] for the specified output and input [`ParameterPath`]s,
    /// or `None` if either path is absent.
    pub fn block(&self, output_path: &ParameterPath, input_path: &ParameterPath) -> Option<JacobianBlock<'_, T, V>> {
        let input_count = self.input_type.parameter_count();
        let (output_index, (_, output_type)) =
            self.output_type.named_parameters().enumerate().find(|(_, (path, _))| path == output_path)?;
        let (input_index, (_, input_type)) =
            self.input_type.named_parameters().enumerate().find(|(_, (path, _))| path == input_path)?;
        Some(JacobianBlock {
            output_path: output_path.clone(),
            output_type,
            input_path: input_path.clone(),
            input_type,
            value: &self.values[output_index * input_count + input_index],
        })
    }

    /// Returns borrowed views of all [`JacobianBlock`]s of this [`Jacobian`] in output-major/input-minor order.
    pub fn iter_blocks(&self) -> impl Iterator<Item = JacobianBlock<'_, T, V>> {
        let input_count = self.input_type.parameter_count();
        self.output_type
            .named_parameters()
            .enumerate()
            .flat_map(move |(output_index, (output_path, output_type))| {
                self.input_type.named_parameters().enumerate().map(move |(input_index, (input_path, input_type))| {
                    JacobianBlock {
                        output_path: output_path.clone(),
                        output_type,
                        input_path,
                        input_type,
                        value: &self.values[output_index * input_count + input_index],
                    }
                })
            })
    }
}

/// Borrowed view of one output/input block in a [`Jacobian`].
#[derive(Debug)]
pub struct JacobianBlock<'o, T: Type, V> {
    /// [`ParameterPath`] of the differentiated output [`Parameter`] that this [`JacobianBlock`] corresponds to.
    output_path: ParameterPath,

    /// [`Type`] of the differentiated output [`Parameter`] that this [`JacobianBlock`] corresponds to.
    output_type: &'o T,

    /// [`ParameterPath`] of the differentiated input [`Parameter`] that this [`JacobianBlock`] corresponds to.
    input_path: ParameterPath,

    /// [`Type`] of the differentiated input [`Parameter`] that this [`JacobianBlock`] corresponds to.
    input_type: &'o T,

    /// Derivative value for this [`JacobianBlock`].
    value: &'o V,
}

impl<'o, T: Type, V> JacobianBlock<'o, T, V> {
    /// Returns the [`ParameterPath`] of the differentiated output [`Parameter`] that this [`JacobianBlock`]
    /// corresponds to.
    #[inline]
    pub fn output_path(&self) -> &ParameterPath {
        &self.output_path
    }

    /// Returns the [`Type`] of the differentiated output [`Parameter`] that this [`JacobianBlock`] corresponds to.
    #[inline]
    pub fn output_type(&self) -> &'o T {
        self.output_type
    }

    /// Returns the [`ParameterPath`] of the differentiated input [`Parameter`] that this [`JacobianBlock`]
    /// corresponds to.
    #[inline]
    pub fn input_path(&self) -> &ParameterPath {
        &self.input_path
    }

    /// Returns the [`Type`] of the differentiated input [`Parameter`] that this [`JacobianBlock`] corresponds to.
    #[inline]
    pub fn input_type(&self) -> &'o T {
        self.input_type
    }

    /// Returns the derivative value for this [`JacobianBlock`].
    #[inline]
    pub fn value(&self) -> &'o V {
        self.value
    }
}

impl<'o, T: Type, V> Clone for JacobianBlock<'o, T, V> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            output_path: self.output_path.clone(),
            output_type: self.output_type,
            input_path: self.input_path.clone(),
            input_type: self.input_type,
            value: self.value,
        }
    }
}

/// Defines one non-auxiliary [`JacobianDifferentiate`] method. It centralizes the shared structured-input and output
/// bounds and adapts its corresponding auxiliary method with a unit auxiliary value.
macro_rules! define_jacobian_function_in_trait {
    (
        $(#[doc = $documentation:literal])*
        $method:ident,
        delegate = $delegate:ident,
        operation_bounds = [$($operation_bounds:tt)+],
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        fn $method<
            F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<O, ProgramError>,
            I: Parameterized<
                    Self::Value,
                    To<Self::Value> = I,
                    Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                >,
            O: Parameterized<
                    LinearizationTracer<Self>,
                    Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
                >,
        >(
            &self,
            function: F,
            primals: I,
        ) -> Result<
            Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>,
            DifferentiationError,
        >
        where
            Self::Operation: $($operation_bounds)+,
        {
            let (jacobian, ()) = self.$delegate(|input| Ok((function(input)?, ())), primals)?;
            Ok(jacobian)
        }
    };
}

/// Defines one auxiliary-output [`JacobianDifferentiate`] method. It keeps the additional auxiliary parameter bounds
/// consistent across differentiation directions and holomorphy modes.
macro_rules! define_jacobian_auxiliary_function_in_trait {
    (
        $(#[doc = $documentation:literal])*
        $method:ident,
        delegate = $delegate:ident,
        holomorphic = $holomorphic:literal,
        operation_bounds = [$($operation_bounds:tt)+],
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        fn $method<
            F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<(O, A), ProgramError>,
            I: Parameterized<
                    Self::Value,
                    To<Self::Value> = I,
                    Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                >,
            O: Parameterized<
                    LinearizationTracer<Self>,
                    Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
                >,
            A: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
        >(
            &self,
            function: F,
            primals: I,
        ) -> Result<
            (
                Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>,
                A::To<Self::Value>,
            ),
            DifferentiationError,
        >
        where
            Self::Operation: $($operation_bounds)+,
        {
            $delegate(self, function, primals, $holomorphic)
        }
    };
}

/// Extension trait carrying the value-level *Jacobian* differentiation transforms on every [`Context`], mirroring
/// how [`ForwardModeDifferentiate`] carries forward-mode differentiation transforms.
/// [`HessianDifferentiate`](crate::HessianDifferentiate) is its sibling for *Hessian* differentiation transforms.
/// Implementations enumerate every finite input or output coordinate and return structured derivative blocks. For a
/// structured function `y = f(x)`, these methods materialize every block of `J_f(x) = ∂f/∂x`. Forward variants apply
/// the [`Pushforward`](crate::Pushforward) to packed input-coordinate basis tangents, while reverse variants apply the
/// [`Pullback`](crate::Pullback) to packed output-coordinate basis cotangents. Both produce the same
/// output-major/input-minor [`Jacobian`] representation.
///
/// Below we provide a cost model that drives the decision of when to use forward-mode or reverse-mode differentiation
/// for computing a [`Jacobian`]. The cost model uses the following notation:
///
///   - `n`: Total input coordinate-space dimension size.
///   - `m`: Total output coordinate-space dimension size.
///   - `T_linearize`: One-time computation needed to evaluate the primal function and construct its pushforward.
///   - `T_pushforward`: Computation needed to propagate one input basis tangent through the constructed pushforward.
///   - `T_vjp`: One-time computation needed to evaluate the primal function and construct its pullback.
///   - `T_pullback`: Computation needed to propagate one output basis cotangent through the constructed pullback.
///   - `R_forward`: Memory occupied by forward-linearization residuals shared by all packed tangent directions.
///   - `R_reverse`: Memory occupied by reverse-linearization residuals shared by all packed cotangent directions.
///   - `M_pushforward`: Additional peak intermediate memory needed by one pushforward direction.
///   - `M_pullback`: Additional peak intermediate memory needed by one pullback direction.
///
/// Forward mode evaluates an `n`-way packed pushforward, so use it when `n <= m` (i.e., for functions with few inputs
/// and many outputs). Its derivative work is approximately `O(T_linearize + n · T_pushforward)`, and its working memory
/// excluding the result is approximately `O(R_forward + n · M_pushforward)`. Reverse mode evaluates an `m`-way packed
/// pullback, so use it when `m < n` (i.e., for functions with many inputs and few outputs). Its derivative work is
/// approximately `O(T_vjp + m · T_pullback)`, and its working memory excluding the result is approximately
/// `O(R_reverse + m · M_pullback)`. Both approaches additionally require the unavoidable `Θ(mn)` memory for the
/// materialized Jacobian itself. Packing may execute the directions in parallel, but does not change these total-work
/// or storage scalings.
///
/// Ordinary forward mode requires real differentiated inputs but permits complex outputs. Ordinary reverse mode
/// requires real differentiated outputs but permits complex inputs. Holomorphic variants require every differentiated
/// input and output leaf to be complex. [`HessianDifferentiate`](crate::HessianDifferentiate) provides the
/// corresponding second-order transform.
pub trait JacobianDifferentiate: Context<Type: DenseDifferentiableType<Self>> {
    define_jacobian_function_in_trait!(
        /// Materializes the complete [`Jacobian`] using forward-mode differentiation.
        /// Refer to [`jacobian_forward`] for the mathematical interpretation and representation.
        jacobian_forward,
        delegate = jacobian_forward_with_aux,
        operation_bounds = [
            PartiallyEvaluatableOperation<Self>
                + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                + ResidualZeroProvider<Self::Type>
        ],
    );

    define_jacobian_function_in_trait!(
        /// Materializes the complete forward-mode [`Jacobian`] under the promise that `function` is holomorphic.
        /// Refer to [`jacobian_forward_holomorphic`] for the holomorphy contract.
        jacobian_forward_holomorphic,
        delegate = jacobian_forward_holomorphic_with_aux,
        operation_bounds = [
            PartiallyEvaluatableOperation<Self>
                + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                + ResidualZeroProvider<Self::Type>
        ],
    );

    define_jacobian_auxiliary_function_in_trait!(
        /// Materializes a forward-mode [`Jacobian`] and returns nondifferentiated auxiliary outputs.
        /// Refer to [`jacobian_forward_with_aux`] for details.
        jacobian_forward_with_aux,
        delegate = jacobian_forward_in_context,
        holomorphic = false,
        operation_bounds = [
            PartiallyEvaluatableOperation<Self>
                + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                + ResidualZeroProvider<Self::Type>
        ],
    );

    define_jacobian_auxiliary_function_in_trait!(
        /// Materializes a holomorphic forward-mode [`Jacobian`] and returns nondifferentiated auxiliary outputs.
        /// Refer to [`jacobian_forward_holomorphic_with_aux`] for details.
        jacobian_forward_holomorphic_with_aux,
        delegate = jacobian_forward_in_context,
        holomorphic = true,
        operation_bounds = [
            PartiallyEvaluatableOperation<Self>
                + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                + ResidualZeroProvider<Self::Type>
        ],
    );

    define_jacobian_function_in_trait!(
        /// Materializes the complete [`Jacobian`] using reverse-mode differentiation.
        /// Refer to [`jacobian_reverse`] for the mathematical interpretation and representation.
        jacobian_reverse,
        delegate = jacobian_reverse_with_aux,
        operation_bounds = [
            PartiallyEvaluatableOperation<Self>
                + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                + DifferentiableOperation<PartialEvaluationContext<Self>>
                + TransposableOperation<Self::Constant, Self::Operation>
                + ResidualZeroProvider<Self::Type>
                + From<AddOperation<Self::Type>>
        ],
    );

    define_jacobian_function_in_trait!(
        /// Materializes the complete reverse-mode [`Jacobian`] under the promise that `function` is holomorphic.
        /// Refer to [`jacobian_reverse_holomorphic`] for the holomorphy contract.
        jacobian_reverse_holomorphic,
        delegate = jacobian_reverse_holomorphic_with_aux,
        operation_bounds = [
            PartiallyEvaluatableOperation<Self>
                + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                + DifferentiableOperation<PartialEvaluationContext<Self>>
                + TransposableOperation<Self::Constant, Self::Operation>
                + ResidualZeroProvider<Self::Type>
                + From<AddOperation<Self::Type>>
        ],
    );

    define_jacobian_auxiliary_function_in_trait!(
        /// Materializes a reverse-mode [`Jacobian`] and returns nondifferentiated auxiliary outputs.
        /// Refer to [`jacobian_reverse_with_aux`] for details.
        jacobian_reverse_with_aux,
        delegate = jacobian_reverse_in_context,
        holomorphic = false,
        operation_bounds = [
            PartiallyEvaluatableOperation<Self>
                + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                + DifferentiableOperation<PartialEvaluationContext<Self>>
                + TransposableOperation<Self::Constant, Self::Operation>
                + ResidualZeroProvider<Self::Type>
                + From<AddOperation<Self::Type>>
        ],
    );

    define_jacobian_auxiliary_function_in_trait!(
        /// Materializes a holomorphic reverse-mode [`Jacobian`] and returns nondifferentiated auxiliary outputs.
        /// Refer to [`jacobian_reverse_holomorphic_with_aux`] for details.
        jacobian_reverse_holomorphic_with_aux,
        delegate = jacobian_reverse_in_context,
        holomorphic = true,
        operation_bounds = [
            PartiallyEvaluatableOperation<Self>
                + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                + DifferentiableOperation<PartialEvaluationContext<Self>>
                + TransposableOperation<Self::Constant, Self::Operation>
                + ResidualZeroProvider<Self::Type>
                + From<AddOperation<Self::Type>>
        ],
    );
}

impl<C: Context<Type: DenseDifferentiableType<C>>> JacobianDifferentiate for C {}

/// Defines one context-recovering Jacobian function without auxiliary outputs. It centralizes the structured generic
/// signature and empty-input handling while keeping each variant's operation requirements explicit at its invocation.
macro_rules! define_jacobian_function {
    (
        $(#[doc = $documentation:literal])*
        $function_name:ident,
        operation_bounds = [$($operation_bounds:tt)+],
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        pub fn $function_name<
            V: Value<
                    Type: DenseDifferentiableType<V::ExecutionDomain>,
                    ExecutionDomain: Context<Operation: $($operation_bounds)+>,
                >,
            F: FnOnce(I::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<O, ProgramError>,
            I: Parameterized<
                    V,
                    To<V> = I,
                    Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>> + ParameterizedFamily<V::Type>,
                >,
            O: Parameterized<
                    LinearizationTracer<V::ExecutionDomain>,
                    Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
                >,
        >(
            function: F,
            primals: I,
        ) -> Result<Jacobian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, DifferentiationError> {
            let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
                return Err(DifferentiationError::EmptyInput);
            };
            context.$function_name(function, primals)
        }
    };
}

/// Defines one context-recovering Jacobian function with auxiliary outputs. It centralizes the structured generic
/// signature and empty-input handling while keeping each variant's operation requirements explicit at its invocation.
macro_rules! define_jacobian_auxiliary_function {
    (
        $(#[doc = $documentation:literal])*
        $function_name:ident,
        operation_bounds = [$($operation_bounds:tt)+],
    ) => {
        $(#[doc = $documentation])*
        #[inline]
        pub fn $function_name<
            V: Value<
                    Type: DenseDifferentiableType<V::ExecutionDomain>,
                    ExecutionDomain: Context<Operation: $($operation_bounds)+>,
                >,
            F: FnOnce(I::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<(O, A), ProgramError>,
            I: Parameterized<
                    V,
                    To<V> = I,
                    Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>> + ParameterizedFamily<V::Type>,
                >,
            O: Parameterized<
                    LinearizationTracer<V::ExecutionDomain>,
                    Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
                >,
            A: Parameterized<LinearizationTracer<V::ExecutionDomain>, Family: ParameterizedFamily<V>>,
        >(
            function: F,
            primals: I,
        ) -> Result<(Jacobian<V::Type, V, I::To<V::Type>, O::To<V::Type>>, A::To<V>), DifferentiationError> {
            let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
                return Err(DifferentiationError::EmptyInput);
            };
            context.$function_name(function, primals)
        }
    };
}

define_jacobian_function!(
    /// Computes the complete [`Jacobian`] of `function` at `primals` using forward-mode differentiation.
    /// For `y = f(x)`, the Jacobian is the linear map `J_f(x) = ∂f/∂x` satisfying `ẏ = J_f(x) · ẋ`. This function
    /// linearizes `function` once, applies the resulting [`Pushforward`](crate::Pushforward) to a packed basis of the
    /// finite input coordinate space, and assembles the resulting columns into a [`Jacobian`]. Each block corresponds
    /// to one output parameter and one input parameter; array blocks place the output axes before the input axes.
    /// [`jacobian_reverse`] produces the same representation by applying the transposed map and is generally
    /// preferable when the output coordinate space is smaller.
    ///
    /// The active context is recovered from the first value in `primals`, so the same entry point works for eager
    /// values and staged tracers. Ordinary forward materialization requires real input leaves, although output leaves
    /// may be complex. Use [`jacobian_forward_holomorphic`] for a complex holomorphic function.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose Jacobian is materialized.
    ///   - `primals`: Structured input values specifying the linearization point.
    jacobian_forward,
    operation_bounds = [
        PartiallyEvaluatableOperation<V::ExecutionDomain>
            + PartiallyEvaluatableOperation<
                TracingContext<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                >,
            >
            + ResidualZeroProvider<V::Type>
    ],
);

define_jacobian_function!(
    /// Computes the forward-mode [`Jacobian`] of a complex holomorphic `function` at `primals`. This function uses
    /// the algorithm and representation described by [`jacobian_forward`], but treats the derivative as complex linear
    /// and requires every differentiated input and output parameter to be complex. Passing `function` is a promise of
    /// holomorphy; this function validates the parameter types but cannot prove that the function satisfies the
    /// [Cauchy-Riemann equations](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations).
    ///
    /// # Parameters
    ///
    ///   - `function`: Holomorphic function whose complex Jacobian is materialized.
    ///   - `primals`: Structured complex input values specifying the linearization point.
    jacobian_forward_holomorphic,
    operation_bounds = [
        PartiallyEvaluatableOperation<V::ExecutionDomain>
            + PartiallyEvaluatableOperation<
                TracingContext<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                >,
            >
            + ResidualZeroProvider<V::Type>
    ],
);

define_jacobian_auxiliary_function!(
    /// Computes a forward-mode [`Jacobian`] and returns nondifferentiated auxiliary outputs. The closure returns
    /// `(output, auxiliary)`. Only `output` contributes to the Jacobian. `auxiliary` is materialized from its primal
    /// trace and returned with it. Refer to [`jacobian_forward`] for the mathematical interpretation, block layout,
    /// context recovery, and ordinary complex-type rules.
    ///
    /// # Parameters
    ///
    ///   - `function`: Holomorphic function whose complex Jacobian is materialized.
    ///   - `primals`: Structured complex input values specifying the linearization point.
    jacobian_forward_with_aux,
    operation_bounds = [
        PartiallyEvaluatableOperation<V::ExecutionDomain>
            + PartiallyEvaluatableOperation<
                TracingContext<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                >,
            >
            + ResidualZeroProvider<V::Type>
    ],
);

define_jacobian_auxiliary_function!(
    /// Computes a holomorphic forward-mode [`Jacobian`] and returns nondifferentiated auxiliary outputs. The
    /// closure and auxiliary-output behavior are described by [`jacobian_forward_with_aux`]. The holomorphy promise
    /// and complex-type requirements are the same as for [`jacobian_forward_holomorphic`].
    ///
    /// # Parameters
    ///
    ///   - `function`: Holomorphic function whose complex Jacobian is materialized.
    ///   - `primals`: Structured complex input values specifying the linearization point.
    jacobian_forward_holomorphic_with_aux,
    operation_bounds = [
        PartiallyEvaluatableOperation<V::ExecutionDomain>
            + PartiallyEvaluatableOperation<
                TracingContext<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                >,
            >
            + ResidualZeroProvider<V::Type>
    ],
);

define_jacobian_function!(
    /// Computes the complete [`Jacobian`] of `function` at `primals` using reverse-mode differentiation.
    /// For `y = f(x)`, the pullback maps an output cotangent `ȳ` to `x̄ = J_f(x)ᵀ · ȳ`, where `J_f(x) = ∂f/∂x`. This
    /// function constructs that [`Pullback`](crate::Pullback) once, applies it to a packed basis of the finite output
    /// coordinate space, and reorients the resulting rows into the same output-major/input-minor [`Jacobian`]
    /// representation returned by [`jacobian_forward`]. Reverse materialization is generally preferable when the
    /// output coordinate space is smaller than the input coordinate space.
    ///
    /// The active context is recovered from the first value in `primals`, so the same entry point works for eager
    /// values and staged tracers. Ordinary reverse materialization requires real output leaves, although input leaves
    /// may be complex. Use [`jacobian_reverse_holomorphic`] for a complex holomorphic function.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose Jacobian is materialized.
    ///   - `primals`: Structured input values specifying the linearization point.
    jacobian_reverse,
    operation_bounds = [
        PartiallyEvaluatableOperation<V::ExecutionDomain>
            + PartiallyEvaluatableOperation<
                TracingContext<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                >,
            >
            + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
            + TransposableOperation<
                <V::ExecutionDomain as Domain>::Constant,
                <V::ExecutionDomain as Domain>::Operation,
            >
            + ResidualZeroProvider<V::Type>
            + From<AddOperation<V::Type>>
    ],
);

define_jacobian_function!(
    /// Computes the reverse-mode [`Jacobian`] of a complex holomorphic `function` at `primals`. This function uses
    /// the algorithm and representation described by [`jacobian_reverse`], but treats the derivative as complex linear
    /// and requires every differentiated input and output parameter to be complex. Passing `function` is a promise of
    /// holomorphy; this function validates the parameter types but cannot prove that the function satisfies the
    /// [Cauchy-Riemann equations](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations).
    ///
    /// # Parameters
    ///
    ///   - `function`: Holomorphic function whose complex Jacobian is materialized.
    ///   - `primals`: Structured complex input values specifying the linearization point.
    jacobian_reverse_holomorphic,
    operation_bounds = [
        PartiallyEvaluatableOperation<V::ExecutionDomain>
            + PartiallyEvaluatableOperation<
                TracingContext<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                >,
            >
            + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
            + TransposableOperation<
                <V::ExecutionDomain as Domain>::Constant,
                <V::ExecutionDomain as Domain>::Operation,
            >
            + ResidualZeroProvider<V::Type>
            + From<AddOperation<V::Type>>
    ],
);

define_jacobian_auxiliary_function!(
    /// Computes a reverse-mode [`Jacobian`] and returns nondifferentiated auxiliary outputs. The closure returns
    /// `(output, auxiliary)`. Only `output` contributes to the Jacobian; `auxiliary` is materialized from its primal
    /// trace and returned with it. Refer to [`jacobian_reverse`] for the mathematical interpretation, block layout,
    /// context recovery, and ordinary complex-type rules.
    ///
    /// # Parameters
    ///
    ///   - `function`: Holomorphic function whose complex Jacobian is materialized.
    ///   - `primals`: Structured complex input values specifying the linearization point.
    jacobian_reverse_with_aux,
    operation_bounds = [
        PartiallyEvaluatableOperation<V::ExecutionDomain>
            + PartiallyEvaluatableOperation<
                TracingContext<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                >,
            >
            + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
            + TransposableOperation<
                <V::ExecutionDomain as Domain>::Constant,
                <V::ExecutionDomain as Domain>::Operation,
            >
            + ResidualZeroProvider<V::Type>
            + From<AddOperation<V::Type>>
    ],
);

define_jacobian_auxiliary_function!(
    /// Computes a holomorphic reverse-mode [`Jacobian`] and returns nondifferentiated auxiliary outputs. The
    /// closure and auxiliary-output behavior are described by [`jacobian_reverse_with_aux`]. The holomorphy promise
    /// and complex-type requirements are the same as for [`jacobian_reverse_holomorphic`].
    ///
    /// # Parameters
    ///
    ///   - `function`: Holomorphic function whose complex Jacobian is materialized.
    ///   - `primals`: Structured complex input values specifying the linearization point.
    jacobian_reverse_holomorphic_with_aux,
    operation_bounds = [
        PartiallyEvaluatableOperation<V::ExecutionDomain>
            + PartiallyEvaluatableOperation<
                TracingContext<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                >,
            >
            + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
            + TransposableOperation<
                <V::ExecutionDomain as Domain>::Constant,
                <V::ExecutionDomain as Domain>::Operation,
            >
            + ResidualZeroProvider<V::Type>
            + From<AddOperation<V::Type>>
    ],
);

/// Implements forward-mode [`Jacobian`] materialization in an explicitly provided [`Context`] for a function that also
/// returns auxiliary outputs. It linearizes the differentiated output once, packs one tangent basis direction for every
/// input coordinate, replays the resulting pushforward over the packed batch, and returns the materialized primal
/// auxiliary output unchanged.
///
/// # Parameters
///
///   - `context`: Context in which to trace and replay the transform.
///   - `function`: Function returning the differentiated output and auxiliary output.
///   - `primals`: Structured input values specifying the linearization point.
///   - `holomorphic`: Whether to validate all differentiated leaves under a holomorphy promise.
pub(crate) fn jacobian_forward_in_context<
    C: Context<
            Type: DenseDifferentiableType<C>,
            Operation: PartiallyEvaluatableOperation<C>
                           + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
                           + ResidualZeroProvider<C::Type>,
        >,
    I: Parameterized<
            C::Value,
            To<C::Value> = I,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Type>,
        >,
    O: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value> + ParameterizedFamily<C::Type>>,
    Aux: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value>>,
    F: FnOnce(I::To<LinearizationTracer<C>>) -> Result<(O, Aux), ProgramError>,
>(
    context: &C,
    function: F,
    primals: I,
    holomorphic: bool,
) -> Result<(Jacobian<C::Type, C::Value, I::To<C::Type>, O::To<C::Type>>, Aux::To<C::Value>), DifferentiationError> {
    // Preserve the input tree while deriving an isomorphic tree of input types. Validate differentiability and the
    // ordinary-versus-holomorphic complex-type contract before tracing the derivative program.
    let input_structure = primals.parameter_structure();
    let input_values = primals.into_parameters().collect::<Vec<_>>();
    let input_types = I::To::<C::Type>::from_parameters(
        input_structure.clone(),
        input_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Forward.validate_types(&input_types, holomorphic, DifferentiationParameterRole::Input)?;
    let primals = I::from_parameters(input_structure, input_values)?;

    // Assign each input parameter a contiguous range in the flattened input coordinate space. The final prefix offset
    // is the total number of basis directions packed into the replay batch. Validating this before linearization also
    // reports a non-finite input coordinate space with its input role before the closure observes the outputs.
    let input_offsets = coordinate_prefix_offsets::<C, _>(
        &input_types,
        DerivativeTransform::JacobianForward,
        DifferentiationParameterRole::Input,
    )?;
    let batch_size = input_offsets.last().copied().unwrap();

    // Evaluate and linearize the differentiated output once. `linearize` returns only the differentiated output,
    // so capture the auxiliary primals while its closure runs and move them back out after constructing the Jacobian.
    // Output types are validated inside the closure, before linearization materializes a structural zero in a nonzero
    // dynamic array differential space, so a non-finite coordinate space retains the precise Jacobian diagnostic.
    // Validate here because a dynamic structural zero still cannot be materialized by a nullary type-only constructor.
    // A future operand-relative zero operation may make this early validation redundant.
    let mut auxiliary = None;
    let (output, pushforward) = context.linearize(
        |input| {
            let (output, value) = function(input)?;
            auxiliary = Some(extract_auxiliary_primals(value).map_err(ProgramError::from)?);
            let output_structure = output.parameter_structure();
            let output_values = output.into_parameters().collect::<Vec<_>>();
            let output_types = O::To::<C::Type>::from_parameters(
                output_structure.clone(),
                output_values.iter().map(|value| value.r#type().into_owned()),
            )?;
            JacobianMode::Forward
                .validate_types(&output_types, holomorphic, DifferentiationParameterRole::Output)
                .map_err(ProgramError::from)?;
            coordinate_prefix_offsets::<C, _>(
                &output_types,
                DerivativeTransform::JacobianForward,
                DifferentiationParameterRole::Output,
            )
            .map_err(ProgramError::from)?;
            Ok(O::from_parameters(output_structure, output_values)?)
        },
        primals,
    )?;

    // Recover and validate the output type tree from the primal output returned by linearization. These types later
    // determine how each packed pushforward output is partitioned into output/input Jacobian blocks.
    let output_structure = output.parameter_structure();
    let output_values = output.into_parameters().collect::<Vec<_>>();
    let output_types = O::To::<C::Type>::from_parameters(
        output_structure,
        output_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Forward.validate_types(&output_types, holomorphic, DifferentiationParameterRole::Output)?;

    // A pushforward program consumes one tangent per input parameter followed by the closed-over primal residuals.
    // Verify that the derived program still satisfies this contract before constructing its packed inputs.
    let (program, residuals) = pushforward.into_parts();
    let program_input_types = program.input_types();
    let tangent_input_count = program_input_types.len().checked_sub(residuals.len()).ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "pushforward program consumes {} inputs which is fewer than its {} residuals",
            program_input_types.len(),
            residuals.len(),
        ))
    })?;
    if tangent_input_count != input_types.parameter_count() {
        return Err(ProgramError::MalformedProgram(format!(
            "pushforward program consumes {} tangent inputs but the differentiated input has {} leaves",
            tangent_input_count,
            input_types.parameter_count(),
        ))
        .into());
    }
    for (index, (program_input_type, (path, input_type))) in
        program_input_types[..tangent_input_count].iter().zip(input_types.named_parameters()).enumerate()
    {
        let tangent_type = input_type.tangent();
        if tangent_type.is_zero_space() {
            return Err(DifferentiationError::NonDifferentiableParameter {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: path.to_string(),
                r#type: input_type.to_string(),
            });
        }
        if program_input_type != &tangent_type {
            return Err(ProgramError::MalformedProgram(format!(
                "pushforward tangent input {index} has type {program_input_type} but the differentiated input \
                parameter has tangent type {tangent_type}",
            ))
            .into());
        }
    }

    // For every input parameter, construct a packed standard basis occupying that parameter's coordinate range in the
    // shared batch. Replicate residuals because the same linearization point is reused for every tangent direction.
    let mut packed_inputs = input_types
        .named_parameters()
        .enumerate()
        .map(|(index, (path, r#type))| {
            let tangent_type = r#type.tangent();
            if tangent_type.is_zero_space() {
                return Err(DifferentiationError::NonDifferentiableParameter {
                    transform: DerivativeTransform::JacobianForward,
                    role: DifferentiationParameterRole::Input,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
            C::Type::coordinate_basis(context, r#type, &tangent_type, input_offsets[index], batch_size)
        })
        .collect::<Result<Vec<_>, _>>()?;
    packed_inputs.extend(residuals.into_iter().map(C::Type::replicated));

    // Replay all basis tangents together. Each packed output now contains the derivative of one output parameter
    // with respect to every flattened input coordinate.
    let packed_outputs =
        C::Type::replay_derivative_region(context, program.entry_region_ref(), batch_size, packed_inputs)?;
    check_count!("output", packed_outputs, output_types.parameter_count(), ProgramError);

    // Slice each packed output by the coordinate range of each input parameter. Iterating outputs outside inputs
    // produces the output-major/input-minor block order required by `Jacobian`.
    let mut values = Vec::new();
    for (output_index, output_type) in output_types.parameters().enumerate() {
        for (input_index, input_type) in input_types.parameters().enumerate() {
            let value = C::Type::extract_forward_jacobian_block(
                &packed_outputs[output_index],
                batch_size,
                input_offsets[input_index],
                input_type,
                output_type,
            )?;
            values.push(value);
        }
    }

    // Reattach the type trees to the extracted blocks and return the auxiliary primals captured during linearization.
    let jacobian = Jacobian::new(input_types, output_types, values)?;
    let auxiliary = auxiliary.ok_or_else(|| {
        ProgramError::MalformedProgram(
            "the forward-mode Jacobian computation did not evaluate its function".to_string(),
        )
    })?;
    Ok((jacobian, auxiliary))
}

/// Implements reverse-mode [`Jacobian`] materialization in an explicitly provided [`Context`] for a function that also
/// returns auxiliary outputs. It constructs the pullback of the differentiated output once, packs one cotangent basis
/// direction for every output coordinate, replays that pullback over the packed batch, and returns the materialized
/// primal auxiliary output unchanged.
///
/// # Parameters
///
///   - `context`: Context in which to trace and replay the transform.
///   - `function`: Function returning the differentiated output and auxiliary output.
///   - `primals`: Structured input values specifying the linearization point.
///   - `holomorphic`: Whether to validate all differentiated leaves under a holomorphy promise.
pub(crate) fn jacobian_reverse_in_context<
    C: Context<
            Type: DenseDifferentiableType<C>,
            Operation: PartiallyEvaluatableOperation<C>
                           + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
                           + DifferentiableOperation<PartialEvaluationContext<C>>
                           + TransposableOperation<C::Constant, C::Operation>
                           + ResidualZeroProvider<C::Type>
                           + From<AddOperation<C::Type>>,
        >,
    I: Parameterized<
            C::Value,
            To<C::Value> = I,
            Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<C::Type>,
        >,
    O: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value> + ParameterizedFamily<C::Type>>,
    Aux: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value>>,
    F: FnOnce(I::To<LinearizationTracer<C>>) -> Result<(O, Aux), ProgramError>,
>(
    context: &C,
    function: F,
    primals: I,
    holomorphic: bool,
) -> Result<(Jacobian<C::Type, C::Value, I::To<C::Type>, O::To<C::Type>>, Aux::To<C::Value>), DifferentiationError> {
    // Preserve the input tree while deriving and validating its isomorphic type tree. Reverse mode permits complex
    // inputs in the ordinary case, but still requires each input parameter to have a nonzero cotangent space.
    let input_structure = primals.parameter_structure();
    let input_values = primals.into_parameters().collect::<Vec<_>>();
    let input_types = I::To::<C::Type>::from_parameters(
        input_structure.clone(),
        input_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Reverse.validate_types(&input_types, holomorphic, DifferentiationParameterRole::Input)?;
    let primals = I::from_parameters(input_structure, input_values)?;

    // Validate the input coordinate spaces before deriving the pullback, so a non-finite input coordinate space
    // reports its input role instead of the identity-bearing nullary-constructor error that materializing a zero
    // input cotangent during the pullback trace would otherwise raise first. This pre-validation remains necessary
    // while pullback tracing can materialize a dynamic nullary cotangent before reporting that the corresponding
    // coordinate space is non-finite. A future operand-relative zero operation may make it redundant.
    coordinate_prefix_offsets::<C, _>(
        &input_types,
        DerivativeTransform::JacobianReverse,
        DifferentiationParameterRole::Input,
    )?;

    // Evaluate the differentiated output once and construct its reusable pullback. `vjp` returns only the
    // differentiated output, so capture the auxiliary primals while its closure runs and move them back out after
    // constructing the Jacobian.
    let mut auxiliary = None;
    let (output, pullback) = context.vjp(
        |input| {
            let (output, value) = function(input)?;
            auxiliary = Some(extract_auxiliary_primals(value).map_err(ProgramError::from)?);
            Ok(output)
        },
        primals,
    )?;

    // Recover and validate the output type tree from the primal output. In reverse mode these types determine the
    // cotangent basis used to enumerate the rows of the Jacobian.
    let output_structure = output.parameter_structure();
    let output_values = output.into_parameters().collect::<Vec<_>>();
    let output_types = O::To::<C::Type>::from_parameters(
        output_structure,
        output_values.iter().map(|value| value.r#type().into_owned()),
    )?;
    JacobianMode::Reverse.validate_types(&output_types, holomorphic, DifferentiationParameterRole::Output)?;

    // Assign each output parameter a contiguous range in the flattened output coordinate space. The final output
    // offset is the number of cotangent basis directions in the replay batch.
    let output_offsets = coordinate_prefix_offsets::<C, _>(
        &output_types,
        DerivativeTransform::JacobianReverse,
        DifferentiationParameterRole::Output,
    )?;
    let batch_size = output_offsets.last().copied().unwrap();

    // A pullback program consumes one cotangent per output parameter followed by its closed-over primal residuals.
    // Verify that the derived program exposes that input contract before constructing the packed cotangents.
    let (program, residuals) = pullback.into_parts();
    let program_input_types = program.input_types();
    let cotangent_input_count = program_input_types.len().checked_sub(residuals.len()).ok_or_else(|| {
        ProgramError::MalformedProgram(format!(
            "pullback program consumes {} inputs which is fewer than its {} residuals",
            program_input_types.len(),
            residuals.len(),
        ))
    })?;
    if cotangent_input_count != output_types.parameter_count() {
        return Err(ProgramError::MalformedProgram(format!(
            "pullback program consumes {} cotangent inputs but the differentiated output has {} leaves",
            cotangent_input_count,
            output_types.parameter_count(),
        ))
        .into());
    }

    // Construct one packed standard cotangent basis for each output parameter in its assigned coordinate range.
    // Replicate residuals so every cotangent direction reuses the same primal linearization point.
    let mut packed_inputs = output_types
        .named_parameters()
        .enumerate()
        .map(|(index, (path, r#type))| {
            let cotangent_type = r#type.cotangent();
            if cotangent_type.is_zero_space() {
                return Err(DifferentiationError::NonDifferentiableParameter {
                    transform: DerivativeTransform::JacobianReverse,
                    role: DifferentiationParameterRole::Output,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
            if program_input_types[index] != cotangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "pullback cotangent input {} has type {} but output parameter {} has cotangent type {}",
                    index, program_input_types[index], path, cotangent_type,
                ))
                .into());
            }
            C::Type::coordinate_basis(context, r#type, &cotangent_type, output_offsets[index], batch_size)
        })
        .collect::<Result<Vec<_>, _>>()?;
    packed_inputs.extend(residuals.into_iter().map(C::Type::replicated));

    // Replay all output cotangent directions together. Each packed result corresponds to one input parameter and
    // contains that input parameter's sensitivities to every flattened output coordinate.
    let packed_outputs =
        C::Type::replay_derivative_region(context, program.entry_region_ref(), batch_size, packed_inputs)?;
    check_count!("output", packed_outputs, input_types.parameter_count(), ProgramError);

    // Slice each packed input sensitivity by output-coordinate range and orient it as an output/input block.
    // The loop order restores the same output-major/input-minor representation produced by forward mode.
    let mut values = Vec::new();
    for (output_index, output_type) in output_types.parameters().enumerate() {
        for (input_index, input_type) in input_types.parameters().enumerate() {
            let value = C::Type::extract_reverse_jacobian_block(
                &packed_outputs[input_index],
                batch_size,
                output_offsets[output_index],
                output_type,
                input_type,
            )?;
            values.push(value);
        }
    }

    // Reattach the type trees to the extracted blocks and return the auxiliary primals captured
    // while building the pullback.
    let jacobian = Jacobian::new(input_types, output_types, values)?;
    let auxiliary = auxiliary.ok_or_else(|| {
        ProgramError::MalformedProgram(
            "the reverse-mode Jacobian computation did not evaluate its function".to_string(),
        )
    })?;
    Ok((jacobian, auxiliary))
}

/// Direction of a dense Jacobian materialization.
#[derive(Copy, Clone)]
enum JacobianMode {
    Forward,
    Reverse,
}

impl JacobianMode {
    /// Validates the differential representations and complex-type requirements of the provided parameter types.
    ///
    /// # Parameters
    ///
    ///   - `types`: Parameter types to validate.
    ///   - `holomorphic`: Whether the Jacobian is being materialized under a holomorphy promise.
    ///   - `role`: Role of `types` in the Jacobian transform.
    fn validate_types<T: DifferentiableType, Types: Parameterized<T>>(
        self,
        types: &Types,
        holomorphic: bool,
        role: DifferentiationParameterRole,
    ) -> Result<(), DifferentiationError> {
        let transform = match self {
            Self::Forward => DerivativeTransform::JacobianForward,
            Self::Reverse => DerivativeTransform::JacobianReverse,
        };
        for (path, r#type) in types.named_parameters() {
            let differential_type = match self {
                Self::Forward => r#type.tangent(),
                Self::Reverse => r#type.cotangent(),
            };
            if differential_type.is_zero_space() {
                return Err(DifferentiationError::NonDifferentiableParameter {
                    transform,
                    role,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
            if holomorphic && !r#type.is_complex() {
                return Err(DifferentiationError::NonComplexParameter {
                    transform,
                    role,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
            if !holomorphic
                && r#type.is_complex()
                && !matches!(
                    (self, role),
                    (Self::Forward, DifferentiationParameterRole::Output)
                        | (Self::Reverse, DifferentiationParameterRole::Input)
                )
            {
                return Err(DifferentiationError::ComplexParameter {
                    transform,
                    role,
                    path: path.to_string(),
                    r#type: r#type.to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Returns the exclusive prefix offsets of the flattened coordinate spaces represented by `types`. If the parameter
/// leaves have coordinate-space dimensions `[d0, d1, ..., dn]`, the returned vector is `[0, d0, d0 + d1, ...,
/// d0 + d1 + ... + dn]`. Consequently, element `i` is the first packed coordinate direction belonging to leaf `i`,
/// and the final element is the total number of packed directions. Each leaf dimension is obtained from
/// [`DenseDifferentiableType::coordinate_space_dimension`]. Returns a [`DifferentiationError`] if any
/// leaf does not have a finite coordinate space or if a cumulative offset does not fit in [`usize`].
///
/// # Parameters
///
///   - `types`: Structured parameter types whose leaf coordinate spaces are flattened in parameter order.
///   - `transform`: Derivative transform requesting the offsets, used in diagnostics.
///   - `role`: Role of `types` in the derivative transform, used in diagnostics.
fn coordinate_prefix_offsets<C: Context<Type: DenseDifferentiableType<C>>, Types: Parameterized<C::Type>>(
    types: &Types,
    transform: DerivativeTransform,
    role: DifferentiationParameterRole,
) -> Result<Vec<usize>, DifferentiationError> {
    let mut offsets = Vec::new();
    offsets.push(0usize);
    for (path, r#type) in types.named_parameters() {
        let dimension = C::Type::coordinate_space_dimension(r#type, transform, role, &path)?;
        offsets.push(offsets.last().copied().unwrap().checked_add(dimension).ok_or_else(|| {
            DifferentiationError::CoordinateCountOverflow {
                transform,
                role,
                path: path.to_string(),
                r#type: r#type.to_string(),
            }
        })?);
    }
    Ok(offsets)
}

/// Extracts the known underlying primal value from each auxiliary [`LinearizationTracer`] and reconstructs the
/// auxiliary parameter structure.  This function does not evaluate or otherwise materialize the auxiliary outputs.
/// Every tracer must already carry a known primal because auxiliary outputs are not differentiated.
fn extract_auxiliary_primals<
    C: Context<
        Operation: PartiallyEvaluatableOperation<C>
                       + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    >,
    A: Parameterized<LinearizationTracer<C>, Family: ParameterizedFamily<C::Value>>,
>(
    auxiliary: A,
) -> Result<A::To<C::Value>, DifferentiationError> {
    let structure = auxiliary.parameter_structure();
    let values = auxiliary
        .into_parameters()
        .map(|tracer| {
            let (primal, _) = tracer.into_dual().into_parts();
            match primal.into_value()?.value().clone() {
                PartialValue::Known(value) => Ok(value),
                PartialValue::Unknown(r#type) => Err(ProgramError::MalformedProgram(format!(
                    "auxiliary output has unknown primal type {type} but depends only on known primal inputs",
                ))),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(A::To::<C::Value>::from_parameters(structure, values)?)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use approx::assert_abs_diff_eq;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::DataType::{F32, F64};
    use crate::arrays::{ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::backends::Array;
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::{DerivativeTransform, DifferentiationError, DifferentiationParameterRole};
    use crate::operations::{Add, Compare, ComparisonDirection, Select, Sin, ZeroLike};
    use crate::parameters::{ParameterPath, Parameterized};
    use crate::programs::{Typed, Value};

    use super::*;

    /// Returns `2x` for positive `x` and `3x` otherwise, expressed generically so both dense Jacobian modes exercise
    /// comparison, selection, and arithmetic while constructing their coordinate-basis replays.
    fn piecewise_select<V: Value<Type = ArrayType>>(x: V) -> V
    where
        V::DispatchDomain: Context<Type = ArrayType, Constant = Array, Operation = ArrayOperation<Array>>,
    {
        let condition = x.compare(&x.zero_like(), ComparisonDirection::GreaterThan).unwrap();
        let doubled = x.add(&x).unwrap();
        let tripled = doubled.add(&x).unwrap();
        Select::select(&condition, &doubled, &tripled).unwrap()
    }

    #[test]
    fn test_jacobian() {
        // Parameterization preserves input/output metadata and the flattened block values.
        let jacobian = Jacobian::new((F32, vec![F64, F32]), F64, vec![1.0_f32, 2.0, 3.0]).unwrap();
        assert_eq!(jacobian.parameter_count(), 3);
        assert_eq!(jacobian.values(), &[1.0, 2.0, 3.0]);
        let reparameterized =
            <Jacobian<DataType, f64, _, _>>::from_parameters(jacobian.parameter_structure(), [4.0, 5.0, 6.0]).unwrap();
        assert_eq!(reparameterized.input_type(), &(F32, vec![F64, F32]));
        assert_eq!(reparameterized.output_type(), &F64);
        assert_eq!(reparameterized.values(), &[4.0, 5.0, 6.0]);

        // Block iteration uses output-major/input-minor order and block lookup addresses the same structure by path.
        let input_types = (ArrayType::scalar(F32), ArrayType::new(F32, Shape::new(vec![2.into()])));
        let output_types = ArrayType::new(F32, Shape::new(vec![3.into()]));
        let jacobian = Jacobian::new(input_types.clone(), output_types.clone(), vec![10_i32, 20]).unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].output_path(), &ParameterPath::root());
        assert_eq!(blocks[0].input_path().to_string(), "$.0");
        assert_eq!(blocks[0].output_type(), &output_types);
        assert_eq!(blocks[0].input_type(), &input_types.0);
        assert_eq!(*blocks[0].value(), 10);
        assert_eq!(blocks[1].input_path().to_string(), "$.1");
        assert_eq!(*blocks[1].value(), 20);
        let second_input_path = blocks[1].input_path().clone();
        assert_eq!(*jacobian.block(&ParameterPath::root(), &second_input_path).unwrap().value(), 20);
        assert!(jacobian.block(&ParameterPath::root(), &ParameterPath::root().field("missing")).is_none());
    }

    #[test]
    fn test_jacobian_forward() {
        // A vector identity function packs every input coordinate direction into one replay and reconstructs the
        // complete identity matrix as a single output/input block.
        let jacobian = jacobian_forward(|input| Ok(input), Array::vector(vec![1.0, 2.0, 3.0])).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(
            block.value().r#type().into_owned(),
            ArrayType::new(F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(3)])),
        );
        assert_eq!(block.value().to_f64s(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        // Structured inputs and outputs produce blocks in output-major/input-minor order.
        let jacobian = jacobian_forward(
            |(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)),
            (Array::scalar(2.0), Array::scalar(3.0)),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].input_path().to_string(), "$.0");
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        assert_eq!(blocks[1].output_path().to_string(), "$.0");
        assert_eq!(blocks[1].input_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_eq!(blocks[2].output_path().to_string(), "$.1");
        assert_eq!(blocks[2].input_path().to_string(), "$.0");
        assert_abs_diff_eq!(blocks[2].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_eq!(blocks[3].output_path().to_string(), "$.1");
        assert_eq!(blocks[3].input_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[3].value().to_f64s()[0], 1.0, epsilon = 1e-9);

        // An output independent of one input retains an explicit zero block in the Cartesian product.
        let jacobian = jacobian_forward(
            |(x, y)| Ok((x.clone() * y.clone() + x.sin()?, y.clone(), x + y)),
            (Array::scalar(2.0), Array::scalar(3.0)),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 6);
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[2].value().to_f64s()[0], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[3].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[4].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[5].value().to_f64s()[0], 1.0, epsilon = 1e-9);

        // Forward replay follows primal control flow and selects the tangent of the branch taken at the primal point.
        let jacobian = jacobian_forward(|x| Ok(piecewise_select(x)), Array::scalar(2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 2.0, epsilon = 1e-9);

        let jacobian = jacobian_forward(|x| Ok(piecewise_select(x)), Array::scalar(-2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 3.0, epsilon = 1e-9);

        // Narrow primal element types use their widened differential representation for dense Jacobian blocks.
        let input = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let jacobian = jacobian_forward(|value| value.sin(), input).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().r#type().as_ref(), &ArrayType::scalar(F32));
        assert_abs_diff_eq!(block.value().to_f64s()[0], 2.0f64.cos(), epsilon = 1e-6);

        // Scalar inputs broadcast into vector outputs are unbroadcast when their dense block is reconstructed.
        let jacobian = jacobian_forward(
            |(scalar, vector)| Ok(scalar.clone() * vector + scalar),
            (Array::scalar(2.0), Array::vector(vec![3.0, 4.0])),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[0].value().to_f64s(), vec![4.0, 5.0]);
        assert_eq!(blocks[1].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[1].value().to_f64s(), vec![2.0, 0.0, 0.0, 2.0]);

        // Zero-sized inputs and outputs remain concrete, honestly typed dense blocks.
        let r#type = ArrayType::new(F64, Shape::new(vec![Dimension::Static(0)]));
        let jacobian =
            jacobian_forward(|input| Ok(input.clone() + input), Array::from_f64s(r#type.clone(), Vec::new())).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(block.value().r#type().as_ref(), &r#type.with_inserted_dimension(0, Dimension::Static(0)).unwrap(),);
        assert!(block.value().storage_bytes().is_empty());

        // The holomorphic entry point treats a complex derivative as complex linear.
        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0));
        let jacobian = jacobian_forward_holomorphic(|x| Ok(x.clone() * x), input).unwrap();
        assert_eq!(
            jacobian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(4.0, 2.0)]),
        );
    }

    #[test]
    fn test_jacobian_reverse() {
        // Structured inputs and outputs produce the same output-major/input-minor blocks as forward mode.
        let jacobian = jacobian_reverse(
            |(x, y)| Ok((x.clone() * y.clone() + x.sin()?, x + y)),
            (Array::scalar(2.0), Array::scalar(3.0)),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0].output_path().to_string(), "$.0");
        assert_eq!(blocks[0].input_path().to_string(), "$.0");
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 3.0 + 2.0f64.cos(), epsilon = 1e-9);
        assert_eq!(blocks[1].output_path().to_string(), "$.0");
        assert_eq!(blocks[1].input_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_eq!(blocks[2].output_path().to_string(), "$.1");
        assert_eq!(blocks[2].input_path().to_string(), "$.0");
        assert_abs_diff_eq!(blocks[2].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_eq!(blocks[3].output_path().to_string(), "$.1");
        assert_eq!(blocks[3].input_path().to_string(), "$.1");
        assert_abs_diff_eq!(blocks[3].value().to_f64s()[0], 1.0, epsilon = 1e-9);

        // Reverse replay routes output cotangents through the branch selected at the primal point.
        let jacobian = jacobian_reverse(|x| Ok(piecewise_select(x)), Array::scalar(2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 2.0, epsilon = 1e-9);

        let jacobian = jacobian_reverse(|x| Ok(piecewise_select(x)), Array::scalar(-2.0)).unwrap();
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 3.0, epsilon = 1e-9);

        // Per-element masking over a vector input makes the Jacobian diagonal, with entries 2 for positive inputs and
        // 3 otherwise.
        let jacobian = jacobian_reverse(|x| Ok(piecewise_select(x)), Array::vector(vec![1.0, -1.0])).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[2]);
        assert_abs_diff_eq!(block.value().to_f64s()[0], 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[2], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(block.value().to_f64s()[3], 3.0, epsilon = 1e-9);

        // Pullback replay unbroadcasts a selected scalar branch and preserves each input's differential data type.
        let scalar = Array::from_f64s(ArrayType::scalar(F32), vec![5.0]);
        let f32_vector_type = ArrayType::new(F32, Shape::new(vec![Dimension::Static(2)]));
        let vector = Array::from_f64s(ArrayType::new(F64, Shape::new(vec![Dimension::Static(2)])), vec![2.0, -3.0]);

        let jacobian = jacobian_reverse(
            |(scalar, vector)| {
                let condition = vector.compare(&vector.zero_like(), ComparisonDirection::GreaterThan)?;
                Select::select(&condition, &scalar, &vector)
            },
            (scalar.clone(), vector.clone()),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().into_owned(), f32_vector_type);
        assert_eq!(blocks[0].value().to_f64s(), vec![1.0, 0.0]);
        assert_eq!(
            blocks[1].value().r#type().into_owned(),
            ArrayType::new(F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(blocks[1].value().to_f64s(), vec![0.0, 0.0, 0.0, 1.0]);

        let jacobian = jacobian_reverse(
            |(scalar, vector)| {
                let condition = vector.compare(&vector.zero_like(), ComparisonDirection::GreaterThan)?;
                Select::select(&condition, &vector, &scalar)
            },
            (scalar, vector),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().into_owned(), f32_vector_type);
        assert_eq!(blocks[0].value().to_f64s(), vec![0.0, 1.0]);
        assert_eq!(
            blocks[1].value().r#type().into_owned(),
            ArrayType::new(F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(blocks[1].value().to_f64s(), vec![1.0, 0.0, 0.0, 0.0]);

        // Promoted elementwise cotangents are converted back to the differential type of each input leaf.
        let f32 = Array::from_f64s(ArrayType::scalar(F32), vec![2.0]);
        let f64 = Array::from_f64s(ArrayType::scalar(F64), vec![3.0]);
        let jacobian = jacobian_reverse(|(left, right)| Ok(left + right), (f32.clone(), f64.clone())).unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().data_type(), F32);
        assert_eq!(blocks[1].value().r#type().data_type(), F64);
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 1.0, epsilon = 1e-9);

        let jacobian = jacobian_reverse(|(left, right)| Ok(left - right), (f32.clone(), f64.clone())).unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], -1.0, epsilon = 1e-9);

        let jacobian = jacobian_reverse(|(left, right)| Ok(left * right), (f32, f64)).unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_abs_diff_eq!(blocks[0].value().to_f64s()[0], 3.0, epsilon = 1e-9);
        assert_abs_diff_eq!(blocks[1].value().to_f64s()[0], 2.0, epsilon = 1e-9);

        // Narrow primal element types use their widened differential representation for dense Jacobian blocks.
        let input = Array::from_f64s(ArrayType::scalar(DataType::F8E8M0FNU), vec![2.0]);
        let jacobian = jacobian_reverse(|value| value.sin(), input).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.value().r#type().as_ref(), &ArrayType::scalar(F32));
        assert_abs_diff_eq!(block.value().to_f64s()[0], 2.0f64.cos(), epsilon = 1e-6);

        // Scalar inputs broadcast into vector outputs are unbroadcast when their dense block is reconstructed.
        let jacobian = jacobian_reverse(
            |(scalar, vector)| Ok(scalar.clone() * vector + scalar),
            (Array::scalar(2.0), Array::vector(vec![3.0, 4.0])),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].value().r#type().static_shape().unwrap().as_slice(), &[2]);
        assert_eq!(blocks[0].value().to_f64s(), vec![4.0, 5.0]);
        assert_eq!(blocks[1].value().r#type().static_shape().unwrap().as_slice(), &[2, 2]);
        assert_eq!(blocks[1].value().to_f64s(), vec![2.0, 0.0, 0.0, 2.0]);

        // Zero-sized inputs and outputs remain concrete, honestly typed dense blocks.
        let r#type = ArrayType::new(F64, Shape::new(vec![Dimension::Static(0)]));
        let jacobian =
            jacobian_reverse(|input| Ok(input.clone() + input), Array::from_f64s(r#type.clone(), Vec::new())).unwrap();
        let block = jacobian.iter_blocks().next().unwrap();
        assert_eq!(block.output_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(block.input_type().static_shape().unwrap().as_slice(), &[0]);
        assert_eq!(block.value().r#type().as_ref(), &r#type.with_inserted_dimension(0, Dimension::Static(0)).unwrap(),);
        assert!(block.value().storage_bytes().is_empty());

        // The holomorphic entry point transposes a complex-linear pushforward without conjugating it.
        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0));
        let jacobian = jacobian_reverse_holomorphic(|x| Ok(x.clone() * x), input).unwrap();
        assert_eq!(
            jacobian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(4.0, 2.0)]),
        );
    }

    #[test]
    fn test_jacobian_with_auxiliary_outputs() {
        let forward_evaluations = Cell::new(0);
        let (jacobian, auxiliary) = jacobian_forward_with_aux(
            |x| {
                forward_evaluations.set(forward_evaluations.get() + 1);
                Ok((x.clone() * x.clone(), x))
            },
            Array::scalar(2.0),
        )
        .unwrap();
        assert_eq!(forward_evaluations.get(), 1);
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 4.0, epsilon = 1e-9);
        assert_eq!(auxiliary.to_f64s(), vec![2.0]);

        let reverse_evaluations = Cell::new(0);
        let (jacobian, auxiliary) = jacobian_reverse_with_aux(
            |x| {
                reverse_evaluations.set(reverse_evaluations.get() + 1);
                Ok((x.clone() * x.clone(), x))
            },
            Array::scalar(2.0),
        )
        .unwrap();
        assert_eq!(reverse_evaluations.get(), 1);
        assert_abs_diff_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s()[0], 4.0, epsilon = 1e-9);
        assert_eq!(auxiliary.to_f64s(), vec![2.0]);

        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0));
        let (jacobian, auxiliary) =
            jacobian_forward_holomorphic_with_aux(|x| Ok((x.clone() * x.clone(), x)), input.clone()).unwrap();
        assert_eq!(
            jacobian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(4.0, 2.0)]),
        );
        assert_eq!(auxiliary, input);

        let (jacobian, auxiliary) =
            jacobian_reverse_holomorphic_with_aux(|x| Ok((x.clone() * x.clone(), x)), input.clone()).unwrap();
        assert_eq!(
            jacobian.iter_blocks().next().unwrap().value().elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(4.0, 2.0)]),
        );
        assert_eq!(auxiliary, input);
    }

    #[test]
    fn test_jacobian_validation() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();

        assert_eq!(
            jacobian_forward(|inputs| Ok(inputs), Vec::<Array>::new()).unwrap_err(),
            DifferentiationError::EmptyInput,
        );

        let integer = Array::from_f64s(ArrayType::scalar(DataType::I32), vec![2.0]);
        assert_eq!(
            context.jacobian_forward(|x| Ok(x), integer).unwrap_err(),
            DifferentiationError::NonDifferentiableParameter {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "i32[]".to_string(),
            },
        );

        let complex = Array::scalar(ComplexNumber::new(2.0f32, 0.0));
        assert_eq!(
            context.jacobian_forward(|x| Ok(x), complex).unwrap_err(),
            DifferentiationError::ComplexParameter {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "c64[]".to_string(),
            },
        );
        assert_eq!(
            context.jacobian_reverse_holomorphic(|x| Ok(x), Array::scalar(2.0)).unwrap_err(),
            DifferentiationError::NonComplexParameter {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "f64[]".to_string(),
            },
        );

        let complex_output_error = context
            .jacobian_reverse(
                |input| Ok(input.context().lift(Array::scalar(ComplexNumber::new(1.0f32, 0.0)))?),
                Array::scalar(2.0),
            )
            .unwrap_err();
        assert_eq!(
            complex_output_error,
            DifferentiationError::ComplexParameter {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Output,
                path: "$".to_string(),
                r#type: "c64[]".to_string(),
            },
        );

        let dynamic_type = ArrayType::new(
            F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::unbounded()))]),
        );
        let dynamic = Array::with_unchecked_type(dynamic_type.clone(), 1.0f64.to_le_bytes().to_vec());
        assert_eq!(
            context.jacobian_forward(|x| Ok(x), dynamic).unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "f64[dynamic]".to_string(),
            },
        );
        assert_eq!(
            context
                .jacobian_forward(
                    |input| Ok(input
                        .context()
                        .lift(Array::with_unchecked_type(dynamic_type.clone(), 1.0f64.to_le_bytes().to_vec()))?),
                    Array::scalar(1.0),
                )
                .unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Output,
                path: "$".to_string(),
                r#type: "f64[dynamic]".to_string(),
            },
        );

        let dynamic = Array::with_unchecked_type(dynamic_type, 1.0f64.to_le_bytes().to_vec());
        assert_eq!(
            context
                .jacobian_reverse(|input| Ok(input.context().lift(Array::scalar(1.0))?), dynamic)
                .unwrap_err(),
            DifferentiationError::NonFiniteCoordinateSpace {
                transform: DerivativeTransform::JacobianReverse,
                role: DifferentiationParameterRole::Input,
                path: "$".to_string(),
                r#type: "f64[dynamic]".to_string(),
            },
        );
    }

    #[test]
    fn test_jacobian_forward_nested_in_batch() {
        let jacobian = batch(
            |input| {
                input
                    .context()
                    .clone()
                    .jacobian_forward(|value| Ok(value.clone() * value), input)
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))
            },
            Array::vector(vec![1.0, 2.0, 3.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(jacobian.iter_blocks().next().unwrap().value().to_f64s(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_jacobian_forward_nested_in_jacobian_reverse() {
        // For f(x) = x², the forward Jacobian is f′(x) = 2x. Differentiating that materialized Jacobian with a
        // reverse Jacobian computes the second derivative f″(x) = 2.
        let derivative = jacobian_reverse(
            |input| {
                let nested_context = input.context().clone();
                let jacobian = nested_context
                    .jacobian_forward(|value| Ok(value.clone() * value), input)
                    .map_err(|error| ProgramError::MalformedProgram(error.to_string()))?;
                Ok(jacobian.into_values().remove(0))
            },
            Array::scalar(3.0),
        )
        .unwrap();
        assert_abs_diff_eq!(derivative.iter_blocks().next().unwrap().value().to_f64s()[0], 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_coordinate_prefix_offsets() {
        let input_types =
            (ArrayType::new(F32, Shape::new(vec![Dimension::Static(usize::MAX)])), ArrayType::scalar(F32));
        assert_eq!(
            coordinate_prefix_offsets::<EagerContext<Array, ArrayOperation<Array>>, _>(
                &input_types,
                DerivativeTransform::JacobianForward,
                DifferentiationParameterRole::Input,
            )
            .unwrap_err(),
            DifferentiationError::CoordinateCountOverflow {
                transform: DerivativeTransform::JacobianForward,
                role: DifferentiationParameterRole::Input,
                path: "$.1".to_string(),
                r#type: "f32[]".to_string(),
            },
        );
        let empty_input_types = ArrayType::new(
            F32,
            Shape::new(vec![Dimension::Static(usize::MAX), Dimension::Static(usize::MAX), Dimension::Static(0)]),
        );
        assert_eq!(
            coordinate_prefix_offsets::<EagerContext<Array, ArrayOperation<Array>>, _>(
                &empty_input_types,
                DerivativeTransform::JacobianForward,
                DifferentiationParameterRole::Input,
            )
            .unwrap(),
            vec![0, 0],
        );
    }
}
