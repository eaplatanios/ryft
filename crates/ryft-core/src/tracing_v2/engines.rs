use std::{cell::RefCell, marker::PhantomData, rc::Rc};

use half::{bf16, f16};

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

macro_rules! impl_scalar_engine_for_scalar {
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
    };
}

macro_rules! impl_differentiable_engine_for_scalar {
    ($ty:ty) => {
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

impl_scalar_engine_for_scalar!(bool, false, true);
impl_scalar_engine_for_scalar!(i8, 0i8, 1i8);
impl_scalar_engine_for_scalar!(i16, 0i16, 1i16);
impl_scalar_engine_for_scalar!(i32, 0i32, 1i32);
impl_scalar_engine_for_scalar!(i64, 0i64, 1i64);
impl_scalar_engine_for_scalar!(u8, 0u8, 1u8);
impl_scalar_engine_for_scalar!(u16, 0u16, 1u16);
impl_scalar_engine_for_scalar!(u32, 0u32, 1u32);
impl_scalar_engine_for_scalar!(u64, 0u64, 1u64);
impl_scalar_engine_for_scalar!(bf16, bf16::ZERO, bf16::ONE);
impl_scalar_engine_for_scalar!(f16, f16::ZERO, f16::ONE);
impl_scalar_engine_for_scalar!(f32, 0.0, 1.0);
impl_scalar_engine_for_scalar!(f64, 0.0, 1.0);

impl_differentiable_engine_for_scalar!(bf16);
impl_differentiable_engine_for_scalar!(f16);
impl_differentiable_engine_for_scalar!(f32);
impl_differentiable_engine_for_scalar!(f64);

#[cfg(test)]
mod tests {
    use crate::{tracing_v2::jvp, types::DataType};

    use super::*;
    
    fn assert_differentiable_engine<E: DifferentiableEngine>() {}

    #[test]
    fn test_array_scalar_engine_is_zero_sized() {
        assert_eq!(size_of::<ScalarEngine<bool>>(), 0);
        assert_eq!(size_of::<ScalarEngine<i8>>(), 0);
        assert_eq!(size_of::<ScalarEngine<u64>>(), 0);
        assert_eq!(size_of::<ScalarEngine<bf16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f16>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f64>>(), 0);
        assert_eq!(size_of::<ScalarEngine<f32>>(), 0);
    }

    #[test]
    fn test_array_scalar_engine_produces_canonical_zero_and_one() {
        let bool_type = ArrayType::scalar(DataType::Boolean);
        let bool_engine = ScalarEngine::<bool>::new();
        assert_eq!(Engine::zero(&bool_engine, &bool_type), Ok(false));
        assert_eq!(Engine::one(&bool_engine, &bool_type), Ok(true));

        let i32_type = ArrayType::scalar(DataType::I32);
        let i32_engine = ScalarEngine::<i32>::new();
        assert_eq!(Engine::zero(&i32_engine, &i32_type), Ok(0i32));
        assert_eq!(Engine::one(&i32_engine, &i32_type), Ok(1i32));

        let u64_type = ArrayType::scalar(DataType::U64);
        let u64_engine = ScalarEngine::<u64>::new();
        assert_eq!(Engine::zero(&u64_engine, &u64_type), Ok(0u64));
        assert_eq!(Engine::one(&u64_engine, &u64_type), Ok(1u64));

        let bf16_type = ArrayType::scalar(DataType::BF16);
        let bf16_engine = ScalarEngine::<bf16>::new();
        assert_eq!(Engine::zero(&bf16_engine, &bf16_type), Ok(bf16::ZERO));
        assert_eq!(Engine::one(&bf16_engine, &bf16_type), Ok(bf16::ONE));

        let f16_type = ArrayType::scalar(DataType::F16);
        let f16_engine = ScalarEngine::<f16>::new();
        assert_eq!(Engine::zero(&f16_engine, &f16_type), Ok(f16::ZERO));
        assert_eq!(Engine::one(&f16_engine, &f16_type), Ok(f16::ONE));

        let f32_type = ArrayType::scalar(DataType::F32);
        let f32_engine = ScalarEngine::<f32>::new();
        assert_eq!(Engine::zero(&f32_engine, &f32_type), Ok(0.0f32));
        assert_eq!(Engine::one(&f32_engine, &f32_type), Ok(1.0f32));

        let f64_type = ArrayType::scalar(DataType::F64);
        let f64_engine = ScalarEngine::<f64>::new();
        assert_eq!(Engine::zero(&f64_engine, &f64_type), Ok(0.0f64));
        assert_eq!(Engine::one(&f64_engine, &f64_type), Ok(1.0f64));
    }

    #[test]
    fn test_half_and_float_scalar_engines_are_differentiable() {
        assert_differentiable_engine::<ScalarEngine<bf16>>();
        assert_differentiable_engine::<ScalarEngine<f16>>();
        assert_differentiable_engine::<ScalarEngine<f32>>();
        assert_differentiable_engine::<ScalarEngine<f64>>();
    }

    #[test]
    fn test_half_scalar_engines_run_jvp() {
        let bf16_engine = ScalarEngine::<bf16>::new();
        assert_eq!(
            jvp(&bf16_engine, |x| x.clone() + x, bf16::from_f32(3.0), bf16::ONE),
            Ok((bf16::from_f32(6.0), bf16::from_f32(2.0)))
        );

        let f16_engine = ScalarEngine::<f16>::new();
        assert_eq!(
            jvp(&f16_engine, |x| x.clone() + x, f16::from_f32(3.0), f16::ONE),
            Ok((f16::from_f32(6.0), f16::from_f32(2.0)))
        );
    }
}
