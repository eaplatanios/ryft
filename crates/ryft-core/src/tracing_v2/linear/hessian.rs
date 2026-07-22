use std::marker::PhantomData;

use ryft_macros::Parameterized;

use crate::contexts::{Context, Domain};
use crate::differentiation::{
    DenseDifferentiableType, DifferentiableOperation, DifferentiationContext, DifferentiationError,
    LinearizationTracer, TransposableOperation,
};
use crate::operations::constants::ZeroOperation;
use crate::operations::math::AddOperation;
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::types::{Type, Typed};
use crate::programs::{ProgramError, Value};
use crate::tracing::TracingContext;

// TODO(eaplatanios): Review this module.

use super::jacobian::{Jacobian, jacobian_forward_in_context, jacobian_reverse_in_context};

/// Defines one non-auxiliary [`HessianDifferentiate`] method. It keeps the nested differentiation bounds shared while
/// adapting its corresponding auxiliary method with a unit auxiliary value.
macro_rules! define_hessian_function_in_trait {
    // Generates one non-auxiliary Hessian context method by adapting its auxiliary counterpart.
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
    // Generates one auxiliary Hessian context method with a statically selected holomorphy mode.
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

/// Dense Hessian of a structured function, represented as its complete output/input/input Cartesian product.
///
/// `I` and `O` retain the input and output type trees. Derivative values are stored in deterministic
/// output-major/first-input-major/second-input-minor order and remain parameters so that the Hessian can cross tracing
/// and compilation boundaries or participate in higher-order transforms.
///
/// The physical representation of a block is defined by [`DenseDifferentiableType`]. For
/// [`ArrayType`](crate::types::ArrayType), the block for shapes `O`, `I1`, and `I2` has shape `O ++ I1 ++ I2`.
#[derive(Clone, Debug, Parameterized)]
pub struct Hessian<T: Type, V: Parameter, I: Parameterized<T>, O: Parameterized<T>> {
    /// Type tree of the differentiated inputs.
    input_types: I,

    /// Type tree of the differentiated outputs.
    output_types: O,

    /// Derivative values in output-major/first-input-major/second-input-minor order.
    values: Vec<V>,

    /// Descriptor-family marker. The input and output fields use `T` only through their bounds.
    _type: PhantomData<fn() -> T>,
}

impl<T: Type, V: Parameter, I: Parameterized<T>, O: Parameterized<T>> Hessian<T, V, I, O> {
    /// Constructs a [`Hessian`] from its input/output type trees and derivative values.
    ///
    /// # Parameters
    ///
    ///   - `input_types`: Type tree of the differentiated inputs.
    ///   - `output_types`: Type tree of the differentiated outputs.
    ///   - `values`: Derivative values in output-major/first-input-major/second-input-minor order. Its length must
    ///     equal the output leaf count multiplied by the square of the input leaf count.
    pub(super) fn new(input_types: I, output_types: O, values: Vec<V>) -> Result<Self, ProgramError> {
        let input_count = input_types.parameter_count();
        let expected_count = output_types
            .parameter_count()
            .checked_mul(input_count)
            .and_then(|count| count.checked_mul(input_count))
            .ok_or_else(|| ProgramError::InvalidArgument {
                message: "hessian block count overflows usize".to_string(),
            })?;
        if values.len() != expected_count {
            return Err(ProgramError::InvalidArgument {
                message: format!("hessian requires {expected_count} derivative values but got {}", values.len()),
            });
        }
        Ok(Self { input_types, output_types, values, _type: PhantomData })
    }

    /// Returns the type tree of the differentiated inputs.
    #[inline]
    pub fn input_types(&self) -> &I {
        &self.input_types
    }

    /// Returns the type tree of the differentiated outputs.
    #[inline]
    pub fn output_types(&self) -> &O {
        &self.output_types
    }

    /// Returns derivative values in output-major/first-input-major/second-input-minor order.
    #[inline]
    pub fn values(&self) -> &[V] {
        self.values.as_slice()
    }

    /// Consumes this [`Hessian`] and returns its derivative values.
    #[inline]
    pub fn into_values(self) -> Vec<V> {
        self.values
    }

    /// Returns borrowed views of all derivative blocks in deterministic Cartesian-product order.
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

    /// Returns the derivative block for the specified output and two input paths.
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
}

/// Borrowed view of one output/input/input block in a [`Hessian`].
#[derive(Debug)]
pub struct HessianBlock<'a, T: Type, V> {
    /// Path of the differentiated output leaf.
    output_path: ParameterPath,

    /// Type of the differentiated output leaf.
    output_type: &'a T,

    /// Path of the first differentiated input leaf.
    first_input_path: ParameterPath,

    /// Type of the first differentiated input leaf.
    first_input_type: &'a T,

    /// Path of the second differentiated input leaf.
    second_input_path: ParameterPath,

    /// Type of the second differentiated input leaf.
    second_input_type: &'a T,

    /// Derivative value for this output/input/input triple.
    value: &'a V,
}

impl<'a, T: Type, V> Clone for HessianBlock<'a, T, V> {
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

impl<'a, T: Type, V> HessianBlock<'a, T, V> {
    /// Returns the path of the differentiated output leaf.
    #[inline]
    pub fn output_path(&self) -> &ParameterPath {
        &self.output_path
    }

    /// Returns the type of the differentiated output leaf.
    #[inline]
    pub fn output_type(&self) -> &'a T {
        self.output_type
    }

    /// Returns the path of the first differentiated input leaf.
    #[inline]
    pub fn first_input_path(&self) -> &ParameterPath {
        &self.first_input_path
    }

    /// Returns the type of the first differentiated input leaf.
    #[inline]
    pub fn first_input_type(&self) -> &'a T {
        self.first_input_type
    }

    /// Returns the path of the second differentiated input leaf.
    #[inline]
    pub fn second_input_path(&self) -> &ParameterPath {
        &self.second_input_path
    }

    /// Returns the type of the second differentiated input leaf.
    #[inline]
    pub fn second_input_type(&self) -> &'a T {
        self.second_input_type
    }

    /// Returns the derivative value for this output/input/input triple.
    #[inline]
    pub fn value(&self) -> &'a V {
        self.value
    }
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
    use crate::parameters::{ParameterPath, Parameterized};
    use crate::types::DataType;
    use crate::types::DataType::{F32, F64};

    use super::Hessian;

    #[test]
    fn test_hessian_parameterization_and_block_order() {
        let hessian = Hessian::new((F32, F64), F64, vec![1_i32, 2, 3, 4]).unwrap();
        assert_eq!(hessian.parameter_count(), 4);
        let blocks = hessian.iter_blocks().collect::<Vec<_>>();
        assert_eq!(blocks.iter().map(|block| *block.value()).collect::<Vec<_>>(), vec![1, 2, 3, 4]);
        assert_eq!(blocks[0].output_path(), &ParameterPath::root());
        assert_eq!(blocks[0].first_input_path().to_string(), "$.0");
        assert_eq!(blocks[0].second_input_path().to_string(), "$.0");
        assert_eq!(blocks[3].first_input_path().to_string(), "$.1");
        assert_eq!(blocks[3].second_input_path().to_string(), "$.1");

        let output_path = blocks[0].output_path().clone();
        let first_input_path = blocks[2].first_input_path().clone();
        let second_input_path = blocks[2].second_input_path().clone();
        assert_eq!(*hessian.block(&output_path, &first_input_path, &second_input_path).unwrap().value(), 3,);
        assert!(
            hessian
                .block(&ParameterPath::root().field("missing"), &first_input_path, &second_input_path)
                .is_none()
        );

        let reparameterized =
            <Hessian<DataType, f32, _, _>>::from_parameters(hessian.parameter_structure(), [5.0, 6.0, 7.0, 8.0])
                .unwrap();
        assert_eq!(reparameterized.input_types(), &(F32, F64));
        assert_eq!(reparameterized.output_types(), &F64);
        assert_eq!(reparameterized.values(), &[5.0, 6.0, 7.0, 8.0]);
    }
}
