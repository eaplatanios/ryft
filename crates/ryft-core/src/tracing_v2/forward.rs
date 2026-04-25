use std::{
    borrow::Cow,
    fmt::Debug,
    fmt::Display,
    ops::{Add, Mul, Neg},
};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder},
    tracing::{InterpretableOperation, Operation, Program, Traceable, TracingError, Value},
    tracing_v2::{
        LinearPrimitiveOperation,
        engines::{DifferentiableEngine, Engine},
        jit::{DifferentiableTracer, Tracer},
        linear::{Linearized, jvp_program, jvp_traced},
        operations::{
            CoreLinearReplayOperation, LinearAddOperation, LinearNegOperation, LinearScaleOperation,
            constants::{OneLike, ZeroLike},
        },
    },
    types::{ArrayType, Type, Typed},
};

/// Tangent representation for a traced primal value.
///
/// [`TangentSpace`] tells the forward-mode machinery what kind of object should travel alongside a
/// given primal leaf while computing a Jacobian-vector product. The simplest case uses the primal
/// type itself, but staged linearization replaces tangents with symbolic values such as
/// [`crate::tracing_v2::LinearTerm`] so that the same primitive JVP rules can build reusable linear
/// programs instead of concrete numbers.
///
/// [`jvp_program`]: crate::tracing_v2::jvp_program
pub trait TangentSpace<T: Type, V: Typed<T>>: Clone + Parameter {
    /// Adds two tangent values.
    fn add(lhs: Self, rhs: Self) -> Self;

    /// Negates a tangent value.
    fn neg(value: Self) -> Self;

    /// Scales a tangent by a primal value.
    fn scale(factor: V, tangent: Self) -> Self;

    /// Produces a zero tangent matching the primal shape.
    fn zero_like(primal: &V, tangent: &Self) -> Self;
}

impl<T: Type, V: Traceable<T> + Add<Output = V> + Mul<Output = V> + Neg<Output = V> + ZeroLike> TangentSpace<T, V>
    for V
{
    #[inline]
    fn add(lhs: Self, rhs: Self) -> Self {
        lhs + rhs
    }

    #[inline]
    fn neg(value: Self) -> Self {
        -value
    }

    #[inline]
    fn scale(factor: V, tangent: Self) -> Self {
        factor * tangent
    }

    #[inline]
    fn zero_like(primal: &V, _tangent: &Self) -> Self {
        primal.zero_like()
    }
}

/// Value-level differentiation metadata for one abstract type family.
///
/// This trait chooses how a leaf value participates in forward-mode differentiation for abstract
/// descriptor `T`. The staged linear carrier stays a generic associated type parameter so engine
/// choices remain in the tracing layer instead of becoming part of the value trait itself.
pub trait Differentiable<T: Type>: Traceable<T> {
    /// Tangent payload carried alongside `Self` during primitive-level JVP staging.
    type Tangent<LinearOperation>: TangentSpace<T, Self>
    where
        T: Display,
        LinearOperation: Clone
            + Operation<T>
            + LinearAddOperation<T, Self>
            + LinearNegOperation<T, Self>
            + LinearScaleOperation<T, Self>;
}

/// Convenience alias for the primitive-level tangent representation associated with engine `E`.
pub type EngineTangent<E> = <<E as Engine>::Value as Differentiable<<E as Engine>::Type>>::Tangent<
    <E as DifferentiableEngine>::LinearOperation,
>;

impl<T, V> Differentiable<T> for V
where
    T: Type + Display,
    V: Traceable<T> + ZeroLike,
{
    type Tangent<LinearOperation>
        = crate::tracing_v2::LinearTerm<T, V, LinearOperation>
    where
        LinearOperation: Clone
            + Operation<T>
            + LinearAddOperation<T, Self>
            + LinearNegOperation<T, Self>
            + LinearScaleOperation<T, Self>;
}

/// Forward-mode tracer carrying both a primal and a tangent.
///
/// [`JvpTracer`] is to forward-mode AD what [`Tracer`](crate::tracing_v2::Tracer) is to ordinary
/// staging: it is the leaf wrapper that primitive operations see when a function is being evaluated
/// in JVP mode. The `primal` field carries the usual runtime value, while the `tangent` field
/// carries the directional derivative information flowing alongside it.
///
/// The type parameters have no bounds on the struct itself so that `JvpTracer` can appear in
/// signatures without eagerly propagating all tangent-space requirements. The required
/// relationship is enforced only on the impl blocks that actually manipulate the values.
#[derive(Clone, Debug, Parameter)]
pub struct JvpTracer<V, T> {
    /// The primal value.
    pub primal: V,

    /// The tangent value associated with the primal.
    pub tangent: T,
}

impl<Ty: Type, V: Traceable<Ty>, T: TangentSpace<Ty, V>> Typed<Ty> for JvpTracer<V, T> {
    #[inline]
    fn r#type(&self) -> Cow<'_, Ty> {
        <V as Typed<Ty>>::r#type(&self.primal)
    }
}

impl<Ty: Type, V: Traceable<Ty>, T: TangentSpace<Ty, V>> Traceable<Ty> for JvpTracer<V, T> {}

impl<V: Traceable<ArrayType> + ZeroLike, T: TangentSpace<ArrayType, V>> ZeroLike for JvpTracer<V, T> {
    #[inline]
    fn zero_like(&self) -> Self {
        Self { primal: self.primal.zero_like(), tangent: T::zero_like(&self.primal, &self.tangent) }
    }
}

impl<V: Traceable<ArrayType> + OneLike, T: TangentSpace<ArrayType, V>> OneLike for JvpTracer<V, T> {
    #[inline]
    fn one_like(&self) -> Self {
        Self { primal: self.primal.one_like(), tangent: T::zero_like(&self.primal, &self.tangent) }
    }
}

/// Standard dual number representation used for first-order forward-mode evaluation.
///
/// [`Dual`] is the common case where primal and tangent live in the same space, which is exactly
/// the setup used by ordinary first-order JVPs over concrete leaves.
pub type Dual<V> = JvpTracer<V, V>;

/// Dispatch trait used by [`jvp`] so it can operate both on concrete values and on already traced values.
///
/// The public transform is intentionally small; this trait is where the concrete, traced, and
/// batched execution strategies branch apart.
#[doc(hidden)]
pub trait JvpInvocationLeaf<E, Input, Output>: Parameter + Sized
where
    E: Engine<Type = ArrayType>,
    Input: Parameterized<Self, ParameterStructure: Clone + Debug + PartialEq>,
    Output: Parameterized<Self, ParameterStructure: Clone>,
{
    /// Input type expected by the user-provided function.
    type FunctionInput<'engine>
    where
        E: 'engine;

    /// Output type produced by the user-provided function.
    type FunctionOutput<'engine>
    where
        E: 'engine;

    /// Invokes [`jvp`] for one leaf regime.
    fn invoke<'engine, F>(
        engine: &'engine E,
        function: F,
        primals: Input,
        tangents: Input,
    ) -> Result<(Output, Output), TracingError>
    where
        F: FnOnce(Self::FunctionInput<'engine>) -> Self::FunctionOutput<'engine>;
}

impl<V: Traceable<ArrayType> + Add<Output = V> + ZeroLike, T: TangentSpace<ArrayType, V>> Add for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self { primal: self.primal + rhs.primal, tangent: T::add(self.tangent, rhs.tangent) }
    }
}

impl<V: Traceable<ArrayType> + Mul<Output = V> + ZeroLike, T: TangentSpace<ArrayType, V>> Mul for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            primal: self.primal.clone() * rhs.primal.clone(),
            tangent: T::add(T::scale(rhs.primal, self.tangent), T::scale(self.primal, rhs.tangent)),
        }
    }
}

impl<V: Traceable<ArrayType> + Neg<Output = V> + ZeroLike, T: TangentSpace<ArrayType, V>> Neg for JvpTracer<V, T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self { primal: -self.primal, tangent: T::neg(self.tangent) }
    }
}

/// Concrete-value dispatch for [`jvp`]: traces the user function with [`Tracer`] to build a staged
/// pushforward via [`jvp_program`] and evaluates it at the supplied tangents.
impl<
    E,
    V: Value<ArrayType> + ZeroLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> JvpInvocationLeaf<E, Input, Output> for V
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    Input::Family: for<'engine> ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    Output::Family: for<'engine> ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    E::DifferentiableOperation: InterpretableOperation<ArrayType, V>,
    V: Differentiable<ArrayType>,
    E::LinearOperation: InterpretableOperation<ArrayType, V>
        + LinearAddOperation<ArrayType, V>
        + LinearNegOperation<ArrayType, V>
        + LinearScaleOperation<ArrayType, V>,
{
    type FunctionInput<'engine>
        = Input::To<DifferentiableTracer<'engine, E>>
    where
        E: 'engine;
    type FunctionOutput<'engine>
        = Output::To<DifferentiableTracer<'engine, E>>
    where
        E: 'engine;

    fn invoke<'engine, F>(
        engine: &'engine E,
        function: F,
        primals: Input,
        tangents: Input,
    ) -> Result<(Output, Output), TracingError>
    where
        F: FnOnce(Self::FunctionInput<'engine>) -> Self::FunctionOutput<'engine>,
    {
        let primal_structure = primals.parameter_structure();
        let tangent_structure = tangents.parameter_structure();
        if primal_structure != tangent_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        let (primal_output, tangent_program): (Output, Program<ArrayType, V, E::LinearOperation, Input, Output>) =
            jvp_program(engine, |input| Ok(function(input)), primals)?;
        let tangent_output = tangent_program.interpret(tangents)?;
        Ok((primal_output, tangent_output))
    }
}

/// Already-traced dispatch for [`jvp`]: delegates to [`jvp_traced`] to replay the user function
/// symbolically inside an enclosing [`Tracer`] scope, staging both the primal output and the
/// tangent propagation as part of the outer compiled program.
impl<
    'engine,
    E,
    V: Traceable<ArrayType> + ZeroLike + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone + Debug + PartialEq, To<Tracer<'engine, E>> = Input>,
    Output: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone, To<Tracer<'engine, E>> = Output>,
> JvpInvocationLeaf<E, Input, Output> for Tracer<'engine, E>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    Input::Family: ParameterizedFamily<Tracer<'engine, E>> + ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<Tracer<'engine, E>> + ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Output>,
    E::TracingOperation:
        InterpretableOperation<ArrayType, Linearized<Tracer<'engine, E>, LinearPrimitiveOperation<Tracer<'engine, E>>>>,
    LinearPrimitiveOperation<Tracer<'engine, E>>: CoreLinearReplayOperation<Tracer<'engine, E>>,
{
    type FunctionInput<'call>
        = Input
    where
        E: 'call;
    type FunctionOutput<'call>
        = Output
    where
        E: 'call;

    fn invoke<'call, F>(
        _engine: &'call E,
        function: F,
        primals: Input,
        tangents: Input,
    ) -> Result<(Output, Output), TracingError>
    where
        F: FnOnce(Self::FunctionInput<'call>) -> Self::FunctionOutput<'call>,
    {
        jvp_traced::<_, _, _, V, E>(|input| Ok(function(input)), primals, tangents)
    }
}

/// Evaluates `function` on `primals` and propagates the supplied tangent values forward.
///
/// The returned pair is `(primal_output, tangent_output)`. Architecturally, [`jvp`] is the most
/// direct forward-mode transform in the crate: it either traces the body once to build a staged
/// pushforward or stages the whole JVP into an outer trace if the inputs are already symbolic.
/// Primitive-specific local JVP rules live in [`crate::tracing_v2::operations`]; [`jvp`] is the
/// orchestration layer that selects the concrete or traced execution path.
#[allow(private_bounds, private_interfaces)]
pub fn jvp<'engine, E, F, Input, Output, Leaf>(
    engine: &'engine E,
    function: F,
    primals: Input,
    tangents: Input,
) -> Result<(Output, Output), TracingError>
where
    E: Engine<Type = ArrayType>,
    Leaf: JvpInvocationLeaf<E, Input, Output>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + Debug + PartialEq>,
    Output: Parameterized<Leaf, ParameterStructure: Clone>,
    F: FnOnce(
        <Leaf as JvpInvocationLeaf<E, Input, Output>>::FunctionInput<'engine>,
    ) -> <Leaf as JvpInvocationLeaf<E, Input, Output>>::FunctionOutput<'engine>,
{
    Leaf::invoke(engine, function, primals, tangents)
}

#[cfg(test)]
mod tests {
    use std::{
        borrow::Cow,
        fmt,
        ops::{Add, Mul, Neg},
    };

    use ryft_macros::Parameter;

    use crate::parameters::{ParameterError, Parameterized};
    use crate::tracing_v2::{engines::ArrayScalarEngine, operations::constants::OneLike, test_support};
    use crate::types::{Type, Typed};

    use super::*;

    #[test]
    fn dual_zero_like_zeros_the_tangent_component() {
        let dual = JvpTracer { primal: 3.0f64, tangent: 4.0f64 };
        let zero = dual.zero_like();
        assert_eq!(zero.primal, 0.0);
        assert_eq!(zero.tangent, 0.0);

        let ones = dual.one_like();
        assert_eq!(ones.primal, 1.0);
        assert_eq!(ones.tangent, 0.0);
        test_support::assert_quadratic_pushforward_rendering();
    }

    #[test]
    fn jvp_rejects_mismatched_parameter_structures() {
        let engine = ArrayScalarEngine::<f64>::new();
        let result: Result<(f64, f64), TracingError> =
            jvp(&engine, |xs| xs[0].clone(), vec![2.0f64], vec![1.0f64, 2.0f64]);
        assert!(matches!(
            result,
            Err(TracingError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure,
                right_structure,
            })) if left_structure == format!("{:?}", vec![2.0f64].parameter_structure())
                && right_structure == format!("{:?}", vec![1.0f64, 2.0f64].parameter_structure())
        ));
        test_support::assert_quadratic_pushforward_rendering();
    }

    #[test]
    fn jvp_tracer_exposes_generic_type_metadata_for_non_array_types() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct TestType(&'static str);

        impl Type for TestType {
            fn is_compatible_with(&self, other: &Self) -> bool {
                self == other
            }
        }

        impl fmt::Display for TestType {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(self.0)
            }
        }

        #[derive(Clone, Debug, PartialEq, Eq, Parameter)]
        struct TestValue {
            r#type: TestType,
            value: i32,
        }

        impl TestValue {
            fn new(r#type: TestType, value: i32) -> Self {
                Self { r#type, value }
            }
        }

        impl Typed<TestType> for TestValue {
            fn r#type(&self) -> Cow<'_, TestType> {
                Cow::Borrowed(&self.r#type)
            }
        }

        impl Traceable<TestType> for TestValue {}

        impl ZeroLike for TestValue {
            fn zero_like(&self) -> Self {
                Self::new(self.r#type.clone(), 0)
            }
        }

        impl Add for TestValue {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                assert_eq!(self.r#type, rhs.r#type);
                Self::new(self.r#type, self.value + rhs.value)
            }
        }

        impl Mul for TestValue {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                assert_eq!(self.r#type, rhs.r#type);
                Self::new(self.r#type, self.value * rhs.value)
            }
        }

        impl Neg for TestValue {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self::new(self.r#type, -self.value)
            }
        }

        let scalar_type = TestType("test_scalar");
        let left = JvpTracer {
            primal: TestValue::new(scalar_type.clone(), 3),
            tangent: TestValue::new(scalar_type.clone(), 4),
        };
        assert_eq!(left.r#type().into_owned(), scalar_type.clone());

        fn assert_traceable<T: Type, V: Traceable<T>>(_value: &V) {}

        assert_traceable::<TestType, _>(&left);

        assert_eq!(
            <TestValue as TangentSpace<TestType, TestValue>>::zero_like(&left.primal, &left.tangent),
            TestValue::new(scalar_type.clone(), 0),
        );
        assert_eq!(
            <TestValue as TangentSpace<TestType, TestValue>>::add(
                TestValue::new(scalar_type.clone(), 4),
                TestValue::new(scalar_type.clone(), 5),
            ),
            TestValue::new(scalar_type.clone(), 9),
        );
        assert_eq!(
            <TestValue as TangentSpace<TestType, TestValue>>::scale(
                TestValue::new(scalar_type.clone(), 3),
                TestValue::new(scalar_type.clone(), 5),
            ),
            TestValue::new(scalar_type.clone(), 15),
        );
        assert_eq!(
            <TestValue as TangentSpace<TestType, TestValue>>::neg(TestValue::new(scalar_type.clone(), 4)),
            TestValue::new(scalar_type, -4),
        );
    }
}
