use std::{borrow::Cow, cell::RefCell, fmt::Debug, fmt::Display, rc::Rc};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder},
    tracing::{
        AtomId, Instruction, InterpretableOperation, Operation, Program, ProgramBuilder, Traceable, TracingError, Value,
    },
    tracing_v2::{
        engines::{
            DifferentiableEngine, DifferentiableOperationStagingEngine, DifferentiableStagingEngine, Engine,
            StagingEngine,
        },
        jit::Tracer,
        linear::{jvp_program, jvp_traced},
        operations::{
            DifferentiableOperation, SupportsAdd, SupportsNeg, SupportsScale,
            constants::{One, Zero},
        },
    },
    types::{ArrayType, Type, Typed},
};

/// Concrete state threaded through forward-mode JVP rules.
///
/// [`JvpContext`] owns the active linear-program builder where tangent ops are staged. It is the
/// forward-mode counterpart of
/// [`TranspositionContext`](crate::tracing_v2::operations::TranspositionContext): JVP rules call
/// [`apply_operation`](Self::apply_operation) to stage tangent ops on the active builder.
#[doc(hidden)]
pub struct JvpContext<'a, V, LinearCarrier, T = ArrayType>
where
    T: Type,
    V: Traceable<T>,
    LinearCarrier: Clone + Operation<T>,
{
    /// Builder for the currently active linear program.
    builder: Rc<RefCell<ProgramBuilder<T, V, LinearCarrier>>>,

    /// Phantom marker reserving a context lifetime for future per-pass borrows without forcing an
    /// API change when one is added.
    marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, T, V, LinearCarrier> JvpContext<'a, V, LinearCarrier, T>
where
    T: Type,
    V: Traceable<T>,
    LinearCarrier: Clone + Operation<T>,
{
    /// Creates a JVP context that stages into `builder`.
    #[doc(hidden)]
    pub fn new(builder: Rc<RefCell<ProgramBuilder<T, V, LinearCarrier>>>) -> Self {
        Self { builder, marker: std::marker::PhantomData }
    }

    /// Returns the builder for the currently active linear program.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<T, V, LinearCarrier>>> {
        &self.builder
    }

    /// Stages one operation in the currently active linear program.
    pub fn apply_operation(
        &self,
        inputs: &[AtomId],
        operation: LinearCarrier,
        output_count: usize,
    ) -> Result<Vec<AtomId>, TracingError> {
        let mut builder_borrow = self.builder.borrow_mut();
        let input_types =
            inputs.iter().map(|atom| builder_borrow.atoms[atom.index].r#type().into_owned()).collect::<Vec<_>>();
        let output_types = operation.infer_output_types(&input_types)?;
        if output_types.len() != output_count {
            return Err(TracingError::InvalidOutputCount { expected: output_count, got: output_types.len() });
        }
        let outputs = output_types.into_iter().map(|r#type| builder_borrow.add_variable(r#type)).collect::<Vec<_>>();
        builder_borrow
            .instructions
            .push(Instruction { operation, inputs: inputs.to_vec(), outputs: outputs.clone() });
        Ok(outputs)
    }

    /// Stages a constant tangent on the active linear builder.
    pub fn add_constant(&self, value: V) -> AtomId {
        self.builder.borrow_mut().add_constant(value)
    }
}

/// Value-level contract for leaves that participate in automatic differentiation over `T`.
///
/// The associated [`Tangent`](Self::Tangent) type makes the tangent representation explicit even
/// though today's staged linear-program IR still requires `Tangent = Self` at the transform
/// boundary. The default methods synthesize zero tangents and unit gradient seeds from the
/// value's abstract type metadata through [`Zero`] and [`One`].
pub trait Differentiable<T: Type>: Traceable<T> {
    /// Tangent and cotangent leaf type associated with this primal leaf.
    type Tangent: Traceable<T> + Zero<T> + One<T>;
}

impl<'engine, E> Differentiable<E::Type> for Tracer<'engine, E>
where
    E: StagingEngine + ?Sized,
    E::Value: Differentiable<E::Type>,
{
    type Tangent = Self;
}

/// Forward-mode tracer carrying both a primal and a tangent.
///
/// [`JvpTracer`] is to forward-mode AD what [`Tracer`](crate::tracing_v2::Tracer) is to ordinary
/// staging: it is the leaf wrapper that primitive operations see when a function is being evaluated
/// in JVP mode. The `primal` field carries the usual runtime value, while the `tangent` field
/// carries the directional derivative information flowing alongside it.
///
/// The type parameters have no bounds on the struct itself so that `JvpTracer` can appear in
/// signatures without eagerly propagating all tangent requirements. `tracing_v2` uses `T = AtomId`
/// for the rule-based JVP path threaded through [`JvpContext`], where rules manipulate symbolic
/// tangent atoms directly.
#[derive(Clone, Debug, Parameter)]
pub struct JvpTracer<V, T> {
    /// The primal value.
    pub primal: V,

    /// The tangent value associated with the primal.
    pub tangent: T,
}

impl<Ty: Type, V: Typed<Ty>, T> Typed<Ty> for JvpTracer<V, T> {
    #[inline]
    fn r#type(&self) -> Cow<'_, Ty> {
        <V as Typed<Ty>>::r#type(&self.primal)
    }
}

impl<V: Display, T> Display for JvpTracer<V, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.primal, formatter)
    }
}

impl<Ty: Type, V: Traceable<Ty>, T: Clone + Parameter> Traceable<Ty> for JvpTracer<V, T> {}

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

/// Concrete-value dispatch for [`jvp`]: traces the user function with [`Tracer`] to build a staged
/// pushforward via [`jvp_program`] and evaluates it at the supplied tangents.
impl<
    E,
    V: Value<ArrayType>
        + Differentiable<ArrayType, Tangent = V>
        + Zero<ArrayType>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<V, ParameterStructure: Clone + Debug + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> JvpInvocationLeaf<E, Input, Output> for V
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    Input::Family: for<'engine> ParameterizedFamily<Tracer<'engine, DifferentiableOperationStagingEngine<E>>>,
    Output::Family: for<'engine> ParameterizedFamily<Tracer<'engine, DifferentiableOperationStagingEngine<E>>>,
    E::DifferentiableOperation: DifferentiableOperation<E>,
    E::LinearOperation: InterpretableOperation<ArrayType, V>
        + SupportsAdd<ArrayType, V>
        + SupportsNeg<ArrayType, V>
        + SupportsScale<ArrayType, V>,
{
    type FunctionInput<'engine>
        = Input::To<Tracer<'engine, DifferentiableOperationStagingEngine<E>>>
    where
        E: 'engine;
    type FunctionOutput<'engine>
        = Output::To<Tracer<'engine, DifferentiableOperationStagingEngine<E>>>
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
    V: Traceable<ArrayType> + Differentiable<ArrayType, Tangent = V> + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone + Debug + PartialEq, To<Tracer<'engine, E>> = Input>,
    Output: Parameterized<Tracer<'engine, E>, ParameterStructure: Clone, To<Tracer<'engine, E>> = Output>,
> JvpInvocationLeaf<E, Input, Output> for Tracer<'engine, E>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V>
        + DifferentiableStagingEngine<Type = ArrayType, Value = V>
        + StagingEngine
        + 'static,
    Input::Family: ParameterizedFamily<Tracer<'engine, E>> + ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<Tracer<'engine, E>> + ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Input>,
    Output::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'engine, E>> = Output>,
    E::Operation: crate::tracing_v2::linear::TracedLinearizableOperation<'engine, E>,
    <E as DifferentiableStagingEngine>::LinearOperation<'engine>: InterpretableOperation<ArrayType, Tracer<'engine, E>>,
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
    use crate::parameters::{ParameterError, Parameterized};
    use crate::tracing::ProgramBuilder;
    use crate::tracing_v2::{
        LinearPrimitiveOperation, PrimitiveOperation,
        engines::ArrayScalarEngine,
        jit::TracingEngine,
        operations::{AddOperation, DifferentiableOperation},
        test_support,
    };

    use super::*;

    /// Validates that [`TracingEngine`] can host a JVP rule like [`AddOperation`] when its
    /// `Value` is `Tracer<E>`: the rule stages its primal effect through the underlying engine and
    /// its tangent effect through the context's `LinearOperation` carrier.
    #[test]
    fn tracing_engine_dispatches_add_jvp_with_traced_primals() {
        let engine = ArrayScalarEngine::<f64>::new();
        let outer_builder = Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, PrimitiveOperation<f64>>::new()));
        let outer_input_a = outer_builder.borrow_mut().add_input(ArrayType::scalar(crate::types::DataType::F64));
        let outer_input_b = outer_builder.borrow_mut().add_input(ArrayType::scalar(crate::types::DataType::F64));
        let outer_tracing_engine = TracingEngine::new(&engine, outer_builder.clone());
        let primal_a = outer_tracing_engine.tracer_from_atom(outer_input_a);
        let primal_b = outer_tracing_engine.tracer_from_atom(outer_input_b);

        let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<
            ArrayType,
            Tracer<'_, ArrayScalarEngine<f64>>,
            LinearPrimitiveOperation<Tracer<'_, ArrayScalarEngine<f64>>>,
        >::new()));
        let tangent_a = linear_builder.borrow_mut().add_input(ArrayType::scalar(crate::types::DataType::F64));
        let tangent_b = linear_builder.borrow_mut().add_input(ArrayType::scalar(crate::types::DataType::F64));
        let mut context = JvpContext::new(linear_builder.clone());

        let outputs = AddOperation
            .jvp(
                &outer_tracing_engine,
                &mut context,
                &[
                    JvpTracer { primal: primal_a, tangent: tangent_a },
                    JvpTracer { primal: primal_b, tangent: tangent_b },
                ],
            )
            .expect("AddOperation::jvp should run on a TracingEngine");

        assert_eq!(outputs.len(), 1);
        assert_eq!(linear_builder.borrow().instructions.len(), 1);
        assert_eq!(outer_builder.borrow().instructions.len(), 1);
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
}
