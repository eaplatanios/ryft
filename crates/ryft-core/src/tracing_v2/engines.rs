use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing::{InterpretableOperation, Operation, Program, ProgramBuilder, Traceable, TracingError},
    tracing_v2::{
        LinearPrimitiveOperation, PrimitiveOperation,
        differentiation::{DifferentiableEngine, DifferentiableStagingEngine},
        jit::{Tracer, TracingEngine},
    },
    types::{ArrayType, Type, Typed},
};

/// Synthesizes concrete leaf values from abstract type metadata.
///
/// [`Engine`] is the backend token threaded through the public `tracing_v2` transforms. It has
/// one narrow job: synthesize representative zero and one values from abstract metadata when a
/// transform needs an exemplar but only knows a leaf's type. That responsibility is what lets
/// higher-order transforms stay generic. Linearization,
/// reverse-mode transposition, and rematerialization all occasionally need to rebuild a value from
/// shape/type information alone; [`Engine`] is the narrow seam where backend-specific knowledge
/// enters the otherwise backend-agnostic transform code.
///
/// Per-instruction evaluation stays outside this trait: replay and abstract-eval continue to go
/// straight through [`crate::tracing::InterpretableOperation`] and [`crate::tracing::Operation`]
/// so the common fast path never needs an extra dispatch layer.
///
/// Engines are passed by shared reference to user-facing transforms. Implementations should be
/// cheap to clone (the common case is a [`Copy`] zero-sized type) and must return values whose type
/// metadata agrees with the input descriptor.
pub trait Engine {
    /// Abstract type metadata interpreted by this engine.
    ///
    /// This is the descriptor carried by staged atoms and used during abstract evaluation. For the
    /// default core pipeline it is usually [`ArrayType`](crate::types::ArrayType), but the trait is
    /// generic so backends can substitute a richer metadata type if needed.
    type Type: Type + Parameter;

    /// Concrete leaf value produced by this engine.
    ///
    /// The value is what program replay and eager transforms actually operate on. In other words,
    /// [`Engine::Type`] is the abstract description used while staging, while [`Engine::Value`] is
    /// the runtime leaf that inhabits traced programs once they are executed.
    type Value: Traceable<Self::Type>;

    /// Returns the additive-identity value corresponding to the provided type metadata.
    ///
    /// Transforms use this when they need a representative value for a leaf without having a
    /// concrete witness available, for example when replaying a staged program from retained input
    /// types or constructing zero cotangents in a transposed linear program.
    fn zero(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError>;

    /// Returns the multiplicative-identity value corresponding to the provided type metadata.
    ///
    /// This is used less frequently than [`Engine::zero`] but plays the same architectural role:
    /// it lets traced code materialize identity seeds without depending on an existing exemplar.
    fn one(&self, r#type: &Self::Type) -> Result<Self::Value, TracingError>;
}

/// Engine capability for selecting a staged operation carrier.
///
/// [`StagingEngine`] extends [`Engine`] with the closed operation carrier that the paired
/// [`ProgramBuilder`](crate::tracing::ProgramBuilder) stores. This keeps carrier selection on the
/// engine value that is actually threaded through tracers instead of splitting it across a separate
/// generic parameter.
pub trait StagingEngine: Engine {
    /// Staged operation type selected by this staging engine.
    type Operation: Clone + Operation<Self::Type>;

    /// Stages `function` directly from type metadata using this engine's ordinary staged op set.
    ///
    /// This is the most symbolic tracing entry point: it never needs concrete runtime inputs, only
    /// the parameterized input metadata. The closure is executed once on [`Tracer`] leaves that
    /// stand in for those abstract inputs, and the resulting builder state is finalized into a
    /// [`Program`].
    ///
    /// The returned pair contains both the structured output metadata inferred during tracing and
    /// the staged program itself.
    fn trace<
        'engine,
        F: FnOnce(Input::To<Tracer<'engine, Self>>) -> Result<Output::To<Tracer<'engine, Self>>, TracingError>,
        Input: Parameterized<
                Self::Type,
                ParameterStructure: Clone,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Tracer<'engine, Self>>,
            >,
        Output: Parameterized<
                Self::Type,
                ParameterStructure: Clone,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Tracer<'engine, Self>>,
            >,
    >(
        &'engine self,
        function: F,
        input_types: Input,
    ) -> Result<
        (Output, Program<Self::Type, Self::Value, Self::Operation, Input::To<Self::Value>, Output::To<Self::Value>>),
        TracingError,
    > {
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        TracingEngine::new(self, builder).trace(function, input_types)
    }

    /// Stages `function` with this engine's ordinary op carrier, interprets it, and returns both results.
    fn interpret_and_trace<
        'engine,
        F: FnOnce(Input::To<Tracer<'engine, Self>>) -> Result<Output::To<Tracer<'engine, Self>>, TracingError>,
        Input: Parameterized<
                Self::Value,
                ParameterStructure: Clone + std::fmt::Debug + PartialEq,
                Family: ParameterizedFamily<Tracer<'engine, Self>>,
            >,
        Output: Parameterized<Self::Value, ParameterStructure: Clone, Family: ParameterizedFamily<Tracer<'engine, Self>>>,
    >(
        &'engine self,
        function: F,
        input: Input,
    ) -> Result<(Output, Program<Self::Type, Self::Value, Self::Operation, Input, Output>), TracingError>
    where
        Self::Operation: InterpretableOperation<Self::Type, Self::Value>,
    {
        let input_structure = input.parameter_structure();
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let mut output_structure = None;
        let (_, flat_program): (
            Vec<Self::Type>,
            Program<Self::Type, Self::Value, Self::Operation, Vec<Self::Value>, Vec<Self::Value>>,
        ) = self.trace(
            |flat_traced_input| {
                let traced_input =
                    Input::To::<Tracer<'engine, Self>>::from_parameters(input_structure.clone(), flat_traced_input)?;
                let traced_output = function(traced_input)?;
                output_structure = Some(traced_output.parameter_structure());
                Ok(traced_output.into_parameters().collect::<Vec<_>>())
            },
            input_types,
        )?;
        let output_structure = output_structure
            .expect("interpret_and_trace should record the staged output structure before returning successfully");
        let Program { atoms, input_ids, output_ids, instructions, .. } = flat_program;
        let mut builder = ProgramBuilder::new();
        builder.atoms = atoms;
        builder.input_ids = input_ids;
        builder.instructions = instructions;
        let program = builder.build::<Input, Output>(output_ids, input_structure, output_structure)?;
        let program = program.simplified()?;
        let concrete_input = Input::from_parameters(program.input_structure.clone(), input_values)?;
        Ok((program.interpret(concrete_input)?, program))
    }
}

/// Stateless engine that synthesizes scalar-compatible values from [`ArrayType`] metadata.
///
/// [`ScalarEngine<V>`] is the "minimal backend" used throughout tests and scalar-only
/// examples. It demonstrates the intended role of an [`Engine`] in the smallest possible form:
/// there is no device handle, no mesh state, and no backend registry, just the choice of the
/// built-in primitive carriers plus metadata-driven construction of scalar zeros and ones.
///
/// The engine ignores most of the supplied [`ArrayType`] metadata because scalar leaves have a
/// single canonical runtime representation. That makes it a good teaching example for the rest of
/// the tracing stack: if a transform works against [`ScalarEngine`], the same code path can be
/// reused by richer engines that need sharding, device, or runtime context.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarEngine<V> {
    /// Phantom marker that ties the zero-sized engine to its scalar leaf type.
    marker: PhantomData<fn() -> V>,
}

impl<V> ScalarEngine<V> {
    /// Returns a new [`ScalarEngine<V>`].
    ///
    /// This is a no-op at runtime because the engine is zero-sized; the method mainly exists to
    /// give examples and tests an explicit, readable backend token.
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

macro_rules! impl_engine_for_scalar_engine {
    ($ty:ty, $zero:expr, $one:expr) => {
        impl Engine for ScalarEngine<$ty> {
            type Type = ArrayType;
            type Value = $ty;

            #[inline]
            fn zero(&self, _type: &ArrayType) -> Result<$ty, TracingError> {
                Ok($zero)
            }

            #[inline]
            fn one(&self, _type: &ArrayType) -> Result<$ty, TracingError> {
                Ok($one)
            }
        }

        impl StagingEngine for ScalarEngine<$ty> {
            type Operation = PrimitiveOperation<$ty>;
        }

        impl DifferentiableEngine for ScalarEngine<$ty> {
            type DifferentiableOperation = PrimitiveOperation<$ty>;
            type LinearOperation = LinearPrimitiveOperation<$ty>;
        }

        impl DifferentiableStagingEngine for ScalarEngine<$ty> {
            type LinearOperation<'engine>
                = LinearPrimitiveOperation<Tracer<'engine, Self>>
            where
                Self: 'engine;
        }
    };
}

impl_engine_for_scalar_engine!(f32, 0.0, 1.0);
impl_engine_for_scalar_engine!(f64, 0.0, 1.0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DataType;

    #[test]
    fn test_array_scalar_engine_is_zero_sized() {
        assert_eq!(size_of::<ScalarEngine<f64>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f32>>(), 0);
    }

    #[test]
    fn test_array_scalar_engine_produces_canonical_zero_and_one() {
        let engine = ScalarEngine::<f64>::new();
        let r#type = ArrayType::scalar(DataType::F64);
        assert_eq!(Engine::zero(&engine, &r#type), Ok(0.0));
        assert_eq!(Engine::one(&engine, &r#type), Ok(1.0));
    }
}
