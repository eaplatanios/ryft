use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{
    ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionOperation, DimensionType,
    DimensionValue, DimensionVariable, Shape, ShardingDimension,
};
use crate::axes::Axis;
use crate::batching::array_ir::align_array_batch;
use crate::batching::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatching, BatchAxis, BatchableOperation,
    BatchingContext, BatchingDriver, BatchingError,
};
use crate::contexts::{Context, Domain, EagerContext};
use crate::differentiation::{DifferentiationError, TransposableOperation, TranspositionDriver};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation};
use crate::operations::constants::fill::Fill;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::scan::ScanOperation;
use crate::operations::dimensions::dimension_size::{DimensionSize, DimensionSizeOperation};
use crate::operations::manipulation::broadcasting::BroadcastOperation;
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::conversion::ConvertElementType;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::operations::math::add::Add;
use crate::operations::math::cos::Cos;
use crate::operations::math::div::Div;
use crate::operations::math::log::Log;
use crate::operations::math::mul::Mul;
use crate::operations::math::neg::Neg;
use crate::operations::math::sqrt::Sqrt;
use crate::operations::math::sub::Sub;
use crate::operations::sort::ArgMax;
use crate::parameters::Placeholder;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramBuilder, ProgramError, RegionInterface, Type,
    TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

// TODO(eaplatanios): Review this module.

/// Canonical operation name for [`RngBitGeneratorOperation`].
pub const RNG_BIT_GENERATOR_OPERATION_NAME: &str = "rng_bit_generator";

/// Deterministic counter-based pseudorandom bit-generation algorithm used by an [`RngBitGeneratorOperation`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RandomAlgorithm {
    /// The ThreeFry-2x32 counter-based generator (Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"),
    /// with a `ui64[2]` state holding `[key, counter]`.
    ThreeFry,

    /// The Philox-4x32 counter-based generator from the same paper, with a `ui64[3]` state holding `[key, counter]`
    /// where the 128-bit counter is split into its low and high `u64` halves.
    Philox,
}

impl RandomAlgorithm {
    /// Returns the state type consumed and produced by this algorithm: `ui64[2]` (holding `[key, counter]`) for
    /// [`ThreeFry`](Self::ThreeFry) and `ui64[3]` (holding `[key, counter]` with the 128-bit counter split into its
    /// low and high `u64` halves) for [`Philox`](Self::Philox).
    pub fn state_type(self) -> ArrayType {
        let size = match self {
            Self::ThreeFry => 2,
            Self::Philox => 3,
        };
        ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(size)]))
    }
}

impl Display for RandomAlgorithm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ThreeFry => formatter.write_str("three_fry"),
            Self::Philox => formatter.write_str("philox"),
        }
    }
}

/// [`Operation`] that deterministically generates uniformly distributed random bits from a counter-based generator
/// state in the `T` type universe — the analogue of
/// [StableHLO's `rng_bit_generator`](https://openxla.org/stablehlo/spec#rng_bit_generator) and the primitive
/// underneath [JAX's `jax.random.bits`](https://docs.jax.dev/en/latest/_autosummary/jax.random.bits.html). The
/// single input is the generator state (see [`RandomAlgorithm::state_type`]), and the two outputs are the advanced
/// state followed by the generated bits at the declared output type. Randomness is functional: the same state
/// always produces the same bits, and drawing again requires threading the advanced state (or deriving fresh states
/// with [`split_key`]) — there is no hidden generator state.
///
/// The output element type must be an unsigned-integer type (`ui8`, `ui16`, `ui32`, or `ui64`); distributions over
/// floating-point values are compositions on top of the raw bits (see [`uniform`](RngBitGenerator::uniform)).
/// The declared output must not be sharded (each shard would otherwise see the same bits; derive per-shard states
/// inside `shard_map` instead). The homogeneous array contract requires a static output shape. In an
/// [`ArrayIrType`] graph, a bounded dynamic bits axis instead has one trailing first-class dimension operand;
/// eager execution resolves those operands before generating bits. XLA lowering rejects dynamic bits outputs:
/// generating the physical upper-bound buffer would advance the functional generator state by the physical rather
/// than logical element count, which is observably incorrect.
///
/// Both outputs are discrete, so differentiation assigns structural-zero tangents and transposition is rejected.
/// Homogeneous array batching of a *mapped* state (one state per batch item, e.g. derived with [`split_key`]) stages
/// one carry-free [`ScanOperation`] over the per-item states, so each batch item draws its own bits from its own state.
/// Composite array IR batching remains unsupported because the scan must retain first-class extent operands across
/// its region boundary. Batching a *replicated* state is rejected in either contract because every batch item would see
/// the same state and draw identical bits. The reference array backend implements both
/// [`ThreeFry`](RandomAlgorithm::ThreeFry) and [`Philox`](RandomAlgorithm::Philox)
/// bit-exactly with XLA's implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct RngBitGeneratorOperation<T: Type> {
    /// Algorithm generating the bits.
    algorithm: RandomAlgorithm,

    /// Declared type of the generated-bits output.
    output_type: ArrayType,

    /// Type universe whose bit-generator contract this payload represents.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> RngBitGeneratorOperation<T> {
    /// Creates a new [`RngBitGeneratorOperation`] with the provided algorithm and declared bits output type.
    #[inline]
    pub fn new(algorithm: RandomAlgorithm, output_type: ArrayType) -> Self {
        Self { algorithm, output_type, marker: PhantomData }
    }

    /// Returns the algorithm generating the bits.
    #[inline]
    pub fn algorithm(&self) -> RandomAlgorithm {
        self.algorithm
    }

    /// Returns the declared type of the generated-bits output.
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Returns this payload with every declared output identity renamed according to `renaming`.
    fn renamed(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self::new(self.algorithm, self.output_type.rename_identities(renaming)?))
    }

    /// Renders this payload independently of its homogeneous or composite operation contract.
    fn render_operation(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, RNG_BIT_GENERATOR_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("algorithm", self.algorithm)?;
            operation.field("output_type", &self.output_type)
        })
    }
}

impl<T: Type> Display for RngBitGeneratorOperation<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render_operation(formatter, 0)
    }
}

/// Validates state, bits element type, and sharding for both RNG operation contracts.
fn validate_rng_bit_generator_types(
    algorithm: RandomAlgorithm,
    state_type: &ArrayType,
    output_type: &ArrayType,
) -> Result<(), TypeError> {
    let expected_state_type = algorithm.state_type();
    if state_type.data_type() != expected_state_type.data_type() || state_type.shape() != expected_state_type.shape() {
        return Err(TypeError::invalid(format!(
            "'rng_bit_generator' with the {algorithm} algorithm needs a {expected_state_type} state but got \
             {state_type}",
        )));
    }
    if !matches!(output_type.data_type(), DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64) {
        return Err(TypeError::invalid(format!(
            "'rng_bit_generator' does not support output data type {}",
            output_type.data_type(),
        )));
    }
    let has_sharded_dimension = |array_type: &ArrayType| {
        array_type.sharding().is_some_and(|sharding| {
            sharding.dimensions().iter().any(|dimension| matches!(dimension, ShardingDimension::Sharded(_)))
        })
    };
    if has_sharded_dimension(state_type) || has_sharded_dimension(output_type) {
        return Err(TypeError::invalid(
            "'rng_bit_generator' does not support sharded states or outputs; derive per-shard states inside shard_map \
             instead"
                .to_string(),
        ));
    }
    Ok(())
}

/// Homogeneous bit-generation contract: the single input is the generator state and the declared bits output must be
/// statically shaped.
impl Operation for RngBitGeneratorOperation<ArrayType> {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        RNG_BIT_GENERATOR_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        validate_rng_bit_generator_types(self.algorithm, &input_types[0], &self.output_type)?;
        if self.output_type.static_shape().is_none() {
            return Err(TypeError::invalid(
                "'rng_bit_generator' does not support dynamically shaped outputs".to_string(),
            ));
        }
        Ok(vec![input_types[0].clone(), self.output_type.clone()])
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        self.renamed(renaming)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

/// Composite bit-generation contract: the generator state is followed by one explicit first-class extent operand per
/// dynamic bits axis, each of which must define the dimension variable that the declared bits axis refers to.
impl Operation for RngBitGeneratorOperation<ArrayIrType> {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        RNG_BIT_GENERATOR_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let dynamic_output_dimensions =
            self.output_type.shape().dimensions().iter().filter_map(Dimension::variable).collect::<Vec<_>>();
        let expected_input_count = dynamic_output_dimensions.len() + 1;
        check_count!("input", input_types, expected_input_count, TypeError);
        let state_type = <&ArrayType>::try_from(&input_types[0])?;
        validate_rng_bit_generator_types(self.algorithm, state_type, &self.output_type)?;
        for (input_type, expected_variable) in input_types[1..].iter().zip(dynamic_output_dimensions) {
            let actual_variable = <&DimensionType>::try_from(input_type)?.variable();
            if actual_variable != expected_variable {
                return Err(TypeError::invalid(format!(
                    "'{RNG_BIT_GENERATOR_OPERATION_NAME}' output-extent operand defines dimension variable \
                     '{actual_variable}', but the corresponding declared bits axis refers to '{expected_variable}'",
                )));
            }
        }
        Ok(vec![state_type.clone().into(), self.output_type.clone().into()])
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        self.renamed(renaming)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }
}

impl<C: Domain<Type = ArrayType, Value: RngBitGenerator>> InterpretableOperation<C>
    for RngBitGeneratorOperation<ArrayType>
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (state, bits) = inputs[0].rng_bit_generator(self.algorithm, &self.output_type)?;
        Ok(vec![state, bits])
    }
}

impl<A: DimensionSize<usize> + RngBitGenerator + Value<Type = ArrayType>>
    InterpretableOperation<EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>>
    for RngBitGeneratorOperation<ArrayIrType>
{
    fn interpret<D: InterpretationDriver<EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>>>(
        &self,
        _context: &EagerContext<ArrayIrValue<A>, ArrayIrOperation<A>>,
        driver: &D,
        inputs: &[ArrayIrValue<A>],
    ) -> Result<Vec<ArrayIrValue<A>>, ProgramError> {
        if driver.region_count() != 0 {
            return Err(TypeError::invalid(format!("expected 0 regions but got {}", driver.region_count())).into());
        }
        self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(), &[])?;
        let state = <ArrayIrValue<A> as ValueProjection<ArrayType>>::projected(&inputs[0])?;
        let mut output_extents = inputs[1..].iter();
        let concrete_output_dimensions = self
            .output_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| match dimension {
                Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
                Dimension::Dynamic(_) => {
                    let extent = output_extents.next().unwrap();
                    Ok(Dimension::Static(
                        <ArrayIrValue<A> as ValueProjection<DimensionType>>::projected(extent)?.extent(),
                    ))
                }
            })
            .collect::<Result<Vec<_>, TypeError>>()?;
        let concrete_output_type = self.output_type.clone().with_shape(Shape::new(concrete_output_dimensions));
        let (advanced_state, bits) = state.rng_bit_generator(self.algorithm, &concrete_output_type)?;
        for (axis, dimension) in self.output_type.shape().dimensions().iter().enumerate() {
            if matches!(dimension, Dimension::Dynamic(_)) {
                let expected_extent =
                    concrete_output_type.shape().dimensions()[axis].value().expect("the concrete output is static");
                let actual_extent = bits.dimension_size(axis)?;
                if actual_extent != expected_extent {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "'{RNG_BIT_GENERATOR_OPERATION_NAME}' bits output axis {axis} has extent {actual_extent}, \
                             but its explicit extent operand is {expected_extent}",
                        ),
                    });
                }
            }
        }
        Ok(vec![ArrayIrValue::Array(advanced_state), ArrayIrValue::Array(bits)])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<RngBitGeneratorOperation<T>>>> PartiallyEvaluatableOperation<C>
    for RngBitGeneratorOperation<T>
where
    RngBitGeneratorOperation<T>: Operation<Type = T>,
{
}

impl_non_differentiable_operation!(<T> RngBitGeneratorOperation<T> where T: Type);

/// Random bits are discrete and therefore never form a linear map that can be transposed.
impl<T: Type, V: Value<Type = T>, O: Operation<Type = T>> TransposableOperation<V, O> for RngBitGeneratorOperation<T>
where
    RngBitGeneratorOperation<T>: Operation<Type = T>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        _context: &mut TracingContext<V, O>,
        _driver: &D,
        _inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        _outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: "'rng_bit_generator' cannot be transposed because random bits are discrete".to_string(),
        }
        .into())
    }
}

/// Batching rule for [`RngBitGeneratorOperation`]. A state mapped at some batch axis is realigned to batch axis 0
/// (a `[b, state_width]` stack of per-item states for both [`ThreeFry`](RandomAlgorithm::ThreeFry) and
/// [`Philox`](RandomAlgorithm::Philox)) and one carry-free [`ScanOperation`] is staged over it, whose body binds this
/// same operation on a single per-item state: iteration `i` consumes state row `i` and yields that item's advanced
/// state and bits, and the scan stacks them into the mapped `[b, state_width]` advanced states and `[b, ...]` bits,
/// both at batch axis 0. Each batch item therefore draws exactly the bits its own state would produce unbatched, the
/// staged program's size stays independent of the batch size, and the rule composes with nested batching because the
/// scan is bound through the parent context (an enclosing batching context batches the staged scan structurally).
///
/// A *replicated* state is rejected: every batch item would see the same state and silently draw identical,
/// correlated bits, so callers derive one state per batch item with [`split_key`] and map over the states instead.
impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for RngBitGeneratorOperation<ArrayType>
where
    C::Value: Transpose,
    C::Operation: From<RngBitGeneratorOperation<ArrayType>> + From<ScanOperation<C::Constant>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        if inputs[0].batch_axis().is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: "'rng_bit_generator' cannot batch a replicated state because every batch item would see \
                          the same state; derive one state per batch item with `split_key` and map over the states \
                          explicitly"
                    .to_string(),
            });
        }

        // Realign the mapped states to batch axis 0, so the scan consumes one per-item state row per iteration.
        let states = inputs[0].move_axis(0)?;

        // Build the scan body: one application of this same operation mapping a single per-item state to that
        // item's advanced state and bits.
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let state_input = builder.add_input(self.algorithm.state_type());
        let outputs = builder.add_instruction(self.clone(), Vec::new(), vec![state_input])?.to_vec();
        let body = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            outputs,
            vec![Placeholder],
            vec![Placeholder, Placeholder],
        )?;

        // Stage one carry-free scan over the per-item states; its stacked outputs are the mapped advanced states
        // and the mapped bits, both at batch axis 0.
        let scan = ScanOperation::<C::Constant>::new(0, P::axis_size(context)?);
        let mut outputs = context.parent().bind(scan, vec![body], std::slice::from_ref(states.value()))?;
        check_count!("output", outputs, 2, ProgramError);
        let bits = outputs.remove(1);
        let advanced_states = outputs.remove(0);
        Ok(vec![
            ArrayBatch::new(advanced_states.r#type().into_owned(), advanced_states, Some(0))?,
            ArrayBatch::new(bits.r#type().into_owned(), bits, Some(0))?,
        ])
    }
}

/// Composite batching rule for [`RngBitGeneratorOperation`]. Replicated first-class output extents become invariant
/// scan carries, while one mapped state row is consumed per iteration. This preserves one independently advanced state
/// and one dynamically shaped bits value per batch item without duplicating the generator state.
impl<C: Context<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatching> for RngBitGeneratorOperation<ArrayIrType>
where
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    C::Operation: From<BroadcastOperation>
        + From<DimensionOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + From<RngBitGeneratorOperation<ArrayIrType>>
        + From<ScanOperation<C::Constant>>
        + OperationProjection<ArrayType>,
    <C::Operation as OperationProjection<ArrayType>>::Projected: From<TransposeOperation>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatching>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatching>,
        _driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
        let Some((state, output_extents)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };
        for extent in output_extents {
            extent.validate_replicated_dimension()?;
        }
        if state.batch_axis().is_replicated() {
            return Err(BatchingError::UnsupportedOperation {
                message: "'rng_bit_generator' cannot batch a replicated state because every batch item would see \
                          the same state; derive one state per batch item with `split_key` and map over the states \
                          explicitly"
                    .to_string(),
            });
        }

        let state = align_array_batch(context, state.clone(), Axis::from(0))?;
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        let extent_inputs = output_extents
            .iter()
            .map(|extent| builder.add_input(extent.unbatched_type().clone()))
            .collect::<Vec<_>>();
        let state_input = builder.add_input(ArrayIrType::Array(self.algorithm().state_type()));
        let operation_inputs = std::iter::once(state_input).chain(extent_inputs.iter().copied()).collect::<Vec<_>>();
        let random_outputs = builder.add_instruction(self.clone(), Vec::new(), operation_inputs)?.to_vec();
        let body_outputs = extent_inputs.iter().copied().chain(random_outputs).collect::<Vec<_>>();
        let body = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            body_outputs,
            vec![Placeholder; output_extents.len() + 1],
            vec![Placeholder; output_extents.len() + 2],
        )?;

        let extent_type = context.axis_extent().r#type();
        let length = <&DimensionType>::try_from(extent_type.as_ref())?.to_dimension();
        let scan = ScanOperation::<C::Constant>::new(output_extents.len(), length.clone());
        let mut packed_inputs = output_extents.iter().map(|extent| extent.value().clone()).collect::<Vec<_>>();
        packed_inputs.push(state.into_value());
        if length.variable().is_some() {
            packed_inputs.push(context.axis_extent().clone());
        }
        let mut outputs = context.parent().bind(scan, vec![body], packed_inputs.as_slice())?;
        check_count!("output", outputs, output_extents.len() + 2, ProgramError);
        outputs.drain(..output_extents.len());
        let bits = outputs.remove(1);
        let advanced_states = outputs.remove(0);
        Ok(vec![ArrayIrBatch::new(advanced_states, BatchAxis::new(0))?, ArrayIrBatch::new(bits, BatchAxis::new(0))?])
    }
}

/// Represents the ability to generate deterministic random bits from a counter-based generator state.
/// [`RngBitGenerator`] stages or executes an [`RngBitGeneratorOperation`]; refer to its documentation for the
/// state contract and the transform rules.
pub trait RngBitGenerator: Sized {
    /// Generates random bits of the provided output type from this generator state, returning the advanced state
    /// together with the bits, and a [`ProgramError`] if something goes wrong.
    fn rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
    ) -> Result<(Self, Self), ProgramError>;
}

/// Any context-carrying value generates bits by binding an [`RngBitGeneratorOperation`] through its own context.
/// The `From<RngBitGeneratorOperation<ArrayType>>` bound makes this disjoint from the eager reference value types (whose
/// context operation is [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers the
/// transform tracers and backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> RngBitGenerator for V
where
    V::DispatchDomain: Context<Operation: From<RngBitGeneratorOperation<ArrayType>>>,
{
    fn rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
    ) -> Result<(Self, Self), ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            RngBitGeneratorOperation::new(algorithm, output_type.clone()),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        let bits = outputs.remove(1);
        let state = outputs.remove(0);
        Ok((state, bits))
    }
}

/// Value-level random-sampling capability composed on top of [`RngBitGenerator`]: key splitting and the uniform,
/// normal, and categorical distributions. Every method is a pure composition of existing operations (bit
/// generation, integer division, element-type conversion, elementwise arithmetic, and ranking sorts), so the
/// distributions inherit their transform rules from the primitives, and every method threads the generator state
/// functionally by returning the advanced state alongside the sample.
///
/// The sampling recipes are:
///
///   - [`uniform`](Self::uniform) draws `u32` bits and keeps their top 24 bits (an exact integer in every binary
///     floating-point type involved), scaling by `2⁻²⁴` for `f32` samples in `[0, 1 - 2⁻²⁴]` with no rounding
///     anywhere, and converting the full 32 bits by `2⁻³²` for `f64` samples in `[0, 1)` (with 32 bits of
///     entropy).
///   - [`normal`](Self::normal) applies the Box–Muller transform `√(-2 ln(1 - u₁)) · cos(2π u₂)` to two uniform
///     draws (`1 - u₁ > 0` keeps the logarithm finite).
///   - [`categorical`](Self::categorical) applies the Gumbel-max trick `argmax(logits - ln(-ln(1 - u)))` along the
///     category axis, with ties and the (probability `2⁻²⁴`) infinite-Gumbel tail resolving to the lowest index.
pub trait Random: Sized {
    /// Splits this generator state into `count` fresh, statistically independent states (each seeded by one
    /// generated `u64` key with a zero counter), returning the advanced state followed by the fresh states, and a
    /// [`ProgramError`] if something goes wrong.
    fn split_key(&self, count: usize) -> Result<(Self, Vec<Self>), ProgramError>;

    /// Draws uniformly distributed samples in `[0, 1)` of the provided shape and floating-point data type (`f32`
    /// or `f64`), returning the advanced state together with the samples, and a [`ProgramError`] if something goes
    /// wrong.
    fn uniform(&self, shape: Shape, data_type: DataType) -> Result<(Self, Self), ProgramError>;

    /// Draws standard-normal samples of the provided shape and floating-point data type (`f32` or `f64`), returning
    /// the advanced state together with the samples, and a [`ProgramError`] if something goes wrong.
    fn normal(&self, shape: Shape, data_type: DataType) -> Result<(Self, Self), ProgramError>;

    /// Draws one categorical sample per lane of `logits` along `axis` (the unnormalized log-probability axis),
    /// returning the advanced state together with the sampled `i32` indices (with `axis` dropped from the shape),
    /// and a [`ProgramError`] if something goes wrong.
    fn categorical(&self, logits: &Self, axis: usize) -> Result<(Self, Self), ProgramError>;
}

/// Fills a constant of the provided floating-point array type, narrowing the `f64` payload to the array's element
/// type first so the filled scalar matches the array data type exactly.
fn fill_float_constant<V: Value<Type = ArrayType>>(
    domain: &V::DispatchDomain,
    array_type: &ArrayType,
    value: f64,
) -> Result<V, ProgramError>
where
    V::DispatchDomain: Fill<f64, V>,
{
    domain.fill(array_type, value)
}

impl<V> Random for V
where
    V: Value<Type = ArrayType>
        + RngBitGenerator
        + Add
        + ArgMax
        + Concatenate
        + ConvertElementType
        + Cos
        + Div
        + Log
        + Mul
        + Neg
        + Slice
        + Sqrt
        + Sub
        + ZeroLike,
    V::DispatchDomain: Fill<f64, V>,
{
    fn split_key(&self, count: usize) -> Result<(Self, Vec<Self>), ProgramError> {
        let keys_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(count)]));
        let (state, keys) = self.rng_bit_generator(RandomAlgorithm::ThreeFry, &keys_type)?;
        let fresh_states = (0..count)
            .map(|index| {
                let key = keys.slice(&[index], &[index + 1], &[1])?;
                Concatenate::concatenate([&key, &key.zero_like()], 0)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((state, fresh_states))
    }

    fn uniform(&self, shape: Shape, data_type: DataType) -> Result<(Self, Self), ProgramError> {
        let bits_type = ArrayType::new(DataType::U32, shape.clone());
        let (state, bits) = self.rng_bit_generator(RandomAlgorithm::ThreeFry, &bits_type)?;
        let sample_type = ArrayType::new(data_type, shape);
        let domain = self.dispatch_domain();
        let samples = match data_type {
            DataType::F32 => {
                // The top 24 bits are exactly representable in `f32`, so the division, conversion, and scaling are
                // all exact and the samples cover `[0, 1 - 2⁻²⁴]` without ever rounding up to `1.0`.
                let shifted = bits.div(&domain.fill(&bits_type, f64::from(1u32 << 8))?)?;
                let scale = domain.fill(&sample_type, f64::from(2.0f32.powi(-24)))?;
                shifted.convert_element_type(DataType::F32)?.mul(&scale)?
            }
            DataType::F64 => {
                // All 32 bits are exactly representable in `f64` and `u32::MAX · 2⁻³² < 1` exactly.
                let scale = domain.fill(&sample_type, 2.0f64.powi(-32))?;
                bits.convert_element_type(DataType::F64)?.mul(&scale)?
            }
            data_type => {
                return Err(
                    TypeError::invalid(format!("'uniform' does not support output data type {data_type}")).into()
                );
            }
        };
        Ok((state, samples))
    }

    fn normal(&self, shape: Shape, data_type: DataType) -> Result<(Self, Self), ProgramError> {
        let (state, first) = self.uniform(shape.clone(), data_type)?;
        let (state, second) = state.uniform(shape.clone(), data_type)?;
        let sample_type = ArrayType::new(data_type, shape);
        let domain = self.dispatch_domain();
        let one: Self = fill_float_constant(&domain, &sample_type, 1.0)?;
        let minus_two: Self = fill_float_constant(&domain, &sample_type, -2.0)?;
        let two_pi: Self = fill_float_constant(&domain, &sample_type, std::f64::consts::TAU)?;
        let radius = one.sub(&first)?.log()?.mul(&minus_two)?.sqrt()?;
        let angle = second.mul(&two_pi)?.cos()?;
        Ok((state, radius.mul(&angle)?))
    }

    fn categorical(&self, logits: &Self, axis: usize) -> Result<(Self, Self), ProgramError> {
        let logits_type = logits.r#type().into_owned();
        let data_type = logits_type.data_type();
        if !matches!(data_type, DataType::F32 | DataType::F64) {
            return Err(
                TypeError::invalid(format!("'categorical' does not support logits data type {data_type}")).into()
            );
        }
        let shape = Shape::new(logits_type.shape().dimensions().to_vec());
        let (state, uniform) = self.uniform(shape.clone(), data_type)?;
        let sample_type = ArrayType::new(data_type, shape);
        let one: Self = fill_float_constant(&self.dispatch_domain(), &sample_type, 1.0)?;
        // Gumbel noise `-ln(-ln(1 - u))` stays finite because `1 - u > 0`; the `u = 0` tail maps to `+∞`, which
        // `argmax` resolves like any other maximal value.
        let gumbel = one.sub(&uniform)?.log()?.neg()?.log()?.neg()?;
        Ok((state, logits.add(&gumbel)?.argmax(axis)?))
    }
}

/// Applies the 20-round ThreeFry-2x32 block cipher to one counter pair under the provided key pair, following
/// Salmon et al., ["Parallel Random Numbers: As Easy as 1, 2, 3"](https://doi.org/10.1145/2063384.2063405) and
/// matching [XLA's implementation](https://github.com/openxla/xla/blob/main/xla/hlo/builder/lib/prng.cc) bit for
/// bit (four-round blocks with rotation groups `[13, 15, 26, 6]` and `[17, 29, 16, 24]`, and key injections
/// derived from `key[0]`, `key[1]`, and `key[0] ^ key[1] ^ 0x1BD11BDA`).
pub fn threefry2x32(key: [u32; 2], counter: [u32; 2]) -> [u32; 2] {
    const ROTATIONS: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];
    let key_schedule = [key[0], key[1], key[0] ^ key[1] ^ 0x1BD11BDA];
    let mut x = [counter[0].wrapping_add(key_schedule[0]), counter[1].wrapping_add(key_schedule[1])];
    for block in 0..5usize {
        for round in 0..4 {
            let rotation = ROTATIONS[(block % 2) * 4 + round];
            x[0] = x[0].wrapping_add(x[1]);
            x[1] = x[1].rotate_left(rotation);
            x[1] ^= x[0];
        }
        x[0] = x[0].wrapping_add(key_schedule[(block + 1) % 3]);
        x[1] = x[1].wrapping_add(key_schedule[(block + 2) % 3].wrapping_add(block as u32 + 1));
    }
    x
}

/// Generates `count` uniformly distributed `u32` words from a ThreeFry `[key, counter]` state, returning the words
/// together with the advanced counter. The word layout matches XLA's `rng_bit_generator` expansion for 32-bit
/// outputs: `ceil(count / 2)` counters `counter + i` are split into their low and high halves, each counter is
/// encrypted with [`threefry2x32`], and each counter's two cipher words land in adjacent output positions
/// (truncated to `count` for odd sizes). The counter advances by the `ceil(count / 2)` cipher invocations that
/// actually ran, matching XLA.
pub fn threefry_u32_words(key: u64, counter: u64, count: usize) -> (Vec<u32>, u64) {
    let key = [key as u32, (key >> 32) as u32];
    let pair_count = count.div_ceil(2);
    let mut words = Vec::with_capacity(count + 1);
    for index in 0..pair_count {
        let pair_counter = counter.wrapping_add(index as u64);
        let output = threefry2x32(key, [pair_counter as u32, (pair_counter >> 32) as u32]);
        words.push(output[0]);
        words.push(output[1]);
    }
    words.truncate(count);
    (words, counter.wrapping_add(pair_count as u64))
}

/// Generates `count` uniformly distributed `u64` words from a ThreeFry `[key, counter]` state, returning the words
/// together with the advanced counter. The word layout matches XLA's `rng_bit_generator` expansion for 64-bit
/// outputs: one counter per word, with the two cipher outputs combined as `first | (second << 32)`.
pub fn threefry_u64_words(key: u64, counter: u64, count: usize) -> (Vec<u64>, u64) {
    let key = [key as u32, (key >> 32) as u32];
    let mut words = Vec::with_capacity(count);
    for index in 0..count {
        let word_counter = counter.wrapping_add(index as u64);
        let output = threefry2x32(key, [word_counter as u32, (word_counter >> 32) as u32]);
        words.push(u64::from(output[0]) | (u64::from(output[1]) << 32));
    }
    (words, counter.wrapping_add(count as u64))
}

/// Applies the 10-round Philox-4x32 block cipher to one counter quad under the provided key pair, following
/// Salmon et al., ["Parallel Random Numbers: As Easy as 1, 2, 3"](https://doi.org/10.1145/2063384.2063405) and
/// matching [XLA's implementation](https://github.com/openxla/xla/blob/main/xla/hlo/builder/lib/prng.cc) bit for
/// bit (per-round `u32` multiplier constants `0xD2511F53` and `0xCD9E8D57` producing 64-bit products, and per-round
/// key increments `0x9E3779B9` and `0xBB67AE85`).
pub fn philox4x32(key: [u32; 2], counter: [u32; 4]) -> [u32; 4] {
    const MULTIPLIERS: [u32; 2] = [0xD2511F53, 0xCD9E8D57];
    const KEY_INCREMENTS: [u32; 2] = [0x9E3779B9, 0xBB67AE85];
    let mut key = key;
    let mut x = counter;
    for _ in 0..10 {
        let product0 = u64::from(x[0]) * u64::from(MULTIPLIERS[0]);
        let product1 = u64::from(x[2]) * u64::from(MULTIPLIERS[1]);
        x = [
            (product1 >> 32) as u32 ^ x[1] ^ key[0],
            product1 as u32,
            (product0 >> 32) as u32 ^ x[3] ^ key[1],
            product0 as u32,
        ];
        key = [key[0].wrapping_add(KEY_INCREMENTS[0]), key[1].wrapping_add(KEY_INCREMENTS[1])];
    }
    x
}

/// Generates `count` uniformly distributed `u32` words from a Philox `[key, counter]` state, returning the words
/// together with the advanced 128-bit counter. The word layout matches XLA's `rng_bit_generator` expansion for
/// 32-bit outputs: `ceil(count / 4)` counters `counter + i` are split into their four `u32` limbs (least
/// significant first), each counter is encrypted with [`philox4x32`] under the key's low and high `u32` halves,
/// and each counter's four cipher words land in adjacent output positions (truncated to `count` for sizes that are
/// not multiples of four). The counter advances by the `ceil(count / 4)` cipher invocations that actually ran,
/// matching XLA.
pub fn philox_u32_words(key: u64, counter: u128, count: usize) -> (Vec<u32>, u128) {
    let key = [key as u32, (key >> 32) as u32];
    let quad_count = count.div_ceil(4);
    let mut words = Vec::with_capacity(quad_count * 4);
    for index in 0..quad_count {
        let quad_counter = counter.wrapping_add(index as u128);
        let output = philox4x32(
            key,
            [
                quad_counter as u32,
                (quad_counter >> 32) as u32,
                (quad_counter >> 64) as u32,
                (quad_counter >> 96) as u32,
            ],
        );
        words.extend_from_slice(&output);
    }
    words.truncate(count);
    (words, counter.wrapping_add(quad_count as u128))
}

/// Generates `count` uniformly distributed `u64` words from a Philox `[key, counter]` state, returning the words
/// together with the advanced 128-bit counter. The word layout matches XLA's `rng_bit_generator` expansion for
/// 64-bit outputs: each of the `ceil(count / 2)` cipher invocations combines its four cipher words into the two
/// adjacent output words `first | (second << 32)` and `third | (fourth << 32)` (truncated to `count` for odd
/// sizes), and the counter advances by the `ceil(count / 2)` cipher invocations that actually ran, matching XLA.
pub fn philox_u64_words(key: u64, counter: u128, count: usize) -> (Vec<u64>, u128) {
    let key = [key as u32, (key >> 32) as u32];
    let quad_count = count.div_ceil(2);
    let mut words = Vec::with_capacity(quad_count * 2);
    for index in 0..quad_count {
        let quad_counter = counter.wrapping_add(index as u128);
        let output = philox4x32(
            key,
            [
                quad_counter as u32,
                (quad_counter >> 32) as u32,
                (quad_counter >> 64) as u32,
                (quad_counter >> 96) as u32,
            ],
        );
        words.push(u64::from(output[0]) | (u64::from(output[1]) << 32));
        words.push(u64::from(output[2]) | (u64::from(output[3]) << 32));
    }
    words.truncate(count);
    (words, counter.wrapping_add(quad_count as u128))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayIrValue, DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
        ShardingDimension,
    };
    use crate::backends::{Array, ArrayOperation};
    use crate::batching::{
        BatchAxis, BatchedProgram, BatchingTracer, ProgramBatchingOutputAxesPolicy, RecursiveBatchingPolicy,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::macros::check_operation_type_inference;
    use crate::parameters::Placeholder;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, Typed};

    use super::*;

    /// Returns a ThreeFry `[key, counter]` state array for the reference backend.
    fn state(key: u64, counter: u64) -> Array {
        Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[key, counter]).unwrap()
    }

    /// Returns the `u32[count]` bits output type used throughout these tests.
    fn bits_type(count: usize) -> ArrayType {
        ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Static(count)]))
    }

    #[test]
    fn test_threefry2x32_known_answers() {
        // The Random123 known-answer vectors for ThreeFry-2x32 with 20 rounds.
        assert_eq!(threefry2x32([0, 0], [0, 0]), [0x6b200159, 0x99ba4efe]);
        assert_eq!(threefry2x32([0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff]), [0x1cb996fc, 0xbb002be7]);
        assert_eq!(threefry2x32([0x13198a2e, 0x03707344], [0x243f6a88, 0x85a308d3]), [0xc4923a9c, 0x483df7a0]);
    }

    #[test]
    fn test_threefry_words() {
        // A key with a nonzero high half exercises the `u64 -> [u32; 2]` key split, low half first.
        let key = (1u64 << 32) | 42;
        let key_pair = [42u32, 1u32];

        // 32-bit words: each counter's two cipher words land in adjacent positions, odd counts truncate the final
        // pair, and the counter advances by the `ceil(count / 2)` cipher invocations that actually ran.
        let (words, counter) = threefry_u32_words(key, 7, 5);
        assert_eq!(words.len(), 5);
        assert_eq!(counter, 10);
        assert_eq!(words[0..2], threefry2x32(key_pair, [7, 0]));
        assert_eq!(words[2..4], threefry2x32(key_pair, [8, 0]));
        assert_eq!(words[4], threefry2x32(key_pair, [9, 0])[0]);
        let (even_words, even_counter) = threefry_u32_words(key, 7, 4);
        assert_eq!(even_words, words[0..4]);
        assert_eq!(even_counter, 9);

        // 64-bit words: one counter per word, with the two cipher outputs combined as `first | (second << 32)`, and
        // the counter advances by `count`.
        let (words, counter) = threefry_u64_words(key, 7, 3);
        assert_eq!(counter, 10);
        let word = |pair_counter: u32| {
            let output = threefry2x32(key_pair, [pair_counter, 0]);
            u64::from(output[0]) | (u64::from(output[1]) << 32)
        };
        assert_eq!(words, vec![word(7), word(8), word(9)]);
    }

    #[test]
    fn test_philox4x32_known_answers() {
        // The Random123 known-answer vectors for Philox-4x32 with 10 rounds.
        assert_eq!(philox4x32([0, 0], [0, 0, 0, 0]), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
        assert_eq!(
            philox4x32([0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]),
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
        );
        assert_eq!(
            philox4x32([0xa4093822, 0x299f31d0], [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]),
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
        );
    }

    #[test]
    fn test_philox_words() {
        // A key with a nonzero high half exercises the `u64 -> [u32; 2]` key split, low half first, and a counter
        // just below the 64-bit boundary exercises the carry into the high `u64` half of the 128-bit counter.
        let key = (1u64 << 32) | 42;
        let key_pair = [42u32, 1u32];
        let counter = u128::from(u64::MAX - 1);
        let quad = |quad_counter: u128| {
            philox4x32(
                key_pair,
                [
                    quad_counter as u32,
                    (quad_counter >> 32) as u32,
                    (quad_counter >> 64) as u32,
                    (quad_counter >> 96) as u32,
                ],
            )
        };

        // 32-bit words: each counter's four cipher words land in adjacent positions, non-multiple-of-four counts
        // truncate the final quad, and the counter advances by the `ceil(count / 4)` cipher invocations that
        // actually ran (carrying into the high half here).
        let (words, advanced) = philox_u32_words(key, counter, 9);
        assert_eq!(words.len(), 9);
        assert_eq!(advanced, counter + 3);
        assert_eq!(advanced >> 64, 1);
        assert_eq!(words[0..4], quad(counter));
        assert_eq!(words[4..8], quad(counter + 1));
        assert_eq!(words[8], quad(counter + 2)[0]);

        // 64-bit words: each cipher invocation yields the two adjacent words `first | (second << 32)` and
        // `third | (fourth << 32)`, odd counts truncate the final pair, and the counter advances by the
        // `ceil(count / 2)` cipher invocations.
        let (words, advanced) = philox_u64_words(key, counter, 3);
        assert_eq!(advanced, counter + 2);
        let first_quad = quad(counter);
        let second_quad = quad(counter + 1);
        assert_eq!(
            words,
            vec![
                u64::from(first_quad[0]) | (u64::from(first_quad[1]) << 32),
                u64::from(first_quad[2]) | (u64::from(first_quad[3]) << 32),
                u64::from(second_quad[0]) | (u64::from(second_quad[1]) << 32),
            ],
        );
    }

    #[test]
    fn test_rng_bit_generator() {
        let operation = RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type(5));
        assert_eq!(operation.name(), RNG_BIT_GENERATOR_OPERATION_NAME);
        assert_eq!(operation.algorithm(), RandomAlgorithm::ThreeFry);
        assert_eq!(operation.output_type(), &bits_type(5));
        assert_eq!(operation.to_string(), "rng_bit_generator [algorithm=three_fry, output_type=u32[5]]");
        assert_eq!(
            operation.infer_output_types(&[RandomAlgorithm::ThreeFry.state_type()], &[]),
            Ok(vec![RandomAlgorithm::ThreeFry.state_type(), bits_type(5)]),
        );

        // Eager interpretation through the reference backend matches the reference word expansion, advances the
        // counter by the number of cipher invocations, and is deterministic in the state.
        let state = state(42, 7);
        let outputs = InterpretableOperation::<EagerContext<Array>>::interpret(
            &operation,
            &EagerContext::new(),
            &EmptyRegionDriver,
            std::slice::from_ref(&state),
        )
        .unwrap();
        let (expected_words, expected_counter) = threefry_u32_words(42, 7, 5);
        assert_eq!(outputs[0].elements::<u64>(), Ok(vec![42, expected_counter]));
        assert_eq!(expected_counter, 10);
        assert_eq!(outputs[1].elements::<u32>(), Ok(expected_words));
        let replayed = InterpretableOperation::<EagerContext<Array>>::interpret(
            &operation,
            &EagerContext::new(),
            &EmptyRegionDriver,
            std::slice::from_ref(&state),
        )
        .unwrap();
        assert_eq!(replayed, outputs);

        let extent_variable = DimensionVariable::new("count", DimensionBounds::new(1, Some(9)).unwrap());
        let dynamic_bits_type =
            ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(extent_variable.clone())]));
        let dynamic_operation = RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, dynamic_bits_type.clone());
        assert_eq!(
            dynamic_operation.infer_output_types(
                &[RandomAlgorithm::ThreeFry.state_type().into(), DimensionType::new(extent_variable.clone()).into(),],
                &[],
            ),
            Ok(vec![RandomAlgorithm::ThreeFry.state_type().into(), dynamic_bits_type.into()]),
        );
        let dynamic_outputs =
            InterpretableOperation::<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::interpret(
                &dynamic_operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[
                    ArrayIrValue::Array(state),
                    ArrayIrValue::Dimension(DimensionValue::new(DimensionType::new(extent_variable), 5).unwrap()),
                ],
            )
            .unwrap();
        assert_eq!(dynamic_outputs[0], ArrayIrValue::Array(outputs[0].clone()));
        assert_eq!(dynamic_outputs[1], ArrayIrValue::Array(outputs[1].clone()));

        // Staging through a program builder produces one instruction with both outputs.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(RandomAlgorithm::ThreeFry.state_type());
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder, Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:u64[2] .
                let %1:u64[2], %2:u32[5] = rng_bit_generator [algorithm=three_fry, output_type=u32[5]] %0
                in (%1, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_rng_bit_generator_type_inference() {
        // Wrong state shape and wrong state data type are rejected against the algorithm's state contract.
        let operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, bits_type(4));
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3)]))],
                    error = "'rng_bit_generator' with the three_fry algorithm needs a u64[2] state but got u64[3]",
                },
                {
                    input_types = [ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))],
                    error = "'rng_bit_generator' with the three_fry algorithm needs a u64[2] state but got f32[2]",
                },
            ],
        );

        // Floating-point bits outputs are rejected; every unsigned-integer width is accepted.
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type),
            cases = [{
                input_types = [RandomAlgorithm::ThreeFry.state_type()],
                error = "'rng_bit_generator' does not support output data type f32",
            }],
        );
        for data_type in [DataType::U8, DataType::U16, DataType::U64] {
            let output_type = ArrayType::new(data_type, Shape::new(vec![Dimension::Static(4)]));
            check_operation_type_inference!(
                operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type.clone()),
                cases = [{
                    input_types = [RandomAlgorithm::ThreeFry.state_type()],
                    output_types = [RandomAlgorithm::ThreeFry.state_type(), output_type],
                }],
            );
        }
    }

    #[test]
    fn test_rng_bit_generator_rejects_replicated_batching_and_differentiation() {
        let operation = RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type(4));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(RandomAlgorithm::ThreeFry.state_type());
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder, Placeholder])
            .unwrap();

        // Batching a replicated state is rejected because every batch item would see the same state and draw
        // identical bits.
        let batched = program.batched(
            2,
            ShardingDimension::Replicated,
            &[BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        );
        assert!(matches!(
            batched,
            Err(error) if error.to_string().contains("derive one state per batch item with `split_key`"),
        ));

        // Differentiation succeeds with structural-zero tangents: the integer inputs and outputs live in zero
        // tangent spaces and the tangent side stages no bit generation.
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:u64[2] .
                let %1:u64[2], %2:u32[4] = rng_bit_generator [algorithm=three_fry, output_type=u32[4]] %0
                in (%1, %2)
            "}
            .trim_end(),
        );
    }

    /// Returns an active batching frame over an eager reference-backend parent for direct batching-rule tests.
    fn batching_context(
        axis_size: usize,
    ) -> BatchingContext<EagerContext<Array, ArrayOperation<Array>>, ArrayBatching> {
        BatchingContext::new(EagerContext::new(), axis_size)
    }

    #[test]
    fn test_rng_bit_generator_batching_threefry() {
        // Two distinct ThreeFry states stacked at batch axis 1 (`u64[2, b]` holding `[[k0, k1], [c0, c1]]`) exercise
        // the realignment to batch axis 0 before the scan consumes one state row per iteration.
        let states = Array::from_elements(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            &[42u64, 3, 7, 11],
        )
        .unwrap();
        let input = ArrayBatch::new(states.r#type().into_owned(), states, Some(1)).unwrap();
        let operation = RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type(5));
        let outputs = operation.batch(&batching_context(2), &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(0));

        // Each batch item's advanced state and bits equal the unbatched per-state results exactly.
        let (first_state, first_bits) =
            state(42, 7).rng_bit_generator(RandomAlgorithm::ThreeFry, &bits_type(5)).unwrap();
        let (second_state, second_bits) =
            state(3, 11).rng_bit_generator(RandomAlgorithm::ThreeFry, &bits_type(5)).unwrap();
        let expected_states =
            [first_state.elements::<u64>().unwrap(), second_state.elements::<u64>().unwrap()].concat();
        let expected_bits = [first_bits.elements::<u32>().unwrap(), second_bits.elements::<u32>().unwrap()].concat();
        assert_eq!(outputs[0].value().elements::<u64>(), Ok(expected_states));
        assert_eq!(outputs[1].value().elements::<u32>(), Ok(expected_bits));
    }

    #[test]
    fn test_rng_bit_generator_batching_philox() {
        // Two distinct Philox `u64[3]` states stacked at batch axis 0 (`u64[2, 3]`).
        let states = Array::from_elements(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
            &[42u64, 7, 9, 3, 11, 0],
        )
        .unwrap();
        let input = ArrayBatch::new(states.r#type().into_owned(), states, Some(0)).unwrap();
        let operation = RngBitGeneratorOperation::new(RandomAlgorithm::Philox, bits_type(5));
        let outputs = operation.batch(&batching_context(2), &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(0));

        // Each batch item's advanced state and bits equal the unbatched per-state results exactly.
        let philox_state = |key: u64, counter_low: u64, counter_high: u64| {
            Array::from_elements(RandomAlgorithm::Philox.state_type(), &[key, counter_low, counter_high]).unwrap()
        };
        let (first_state, first_bits) =
            philox_state(42, 7, 9).rng_bit_generator(RandomAlgorithm::Philox, &bits_type(5)).unwrap();
        let (second_state, second_bits) =
            philox_state(3, 11, 0).rng_bit_generator(RandomAlgorithm::Philox, &bits_type(5)).unwrap();
        let expected_states =
            [first_state.elements::<u64>().unwrap(), second_state.elements::<u64>().unwrap()].concat();
        let expected_bits = [first_bits.elements::<u32>().unwrap(), second_bits.elements::<u32>().unwrap()].concat();
        assert_eq!(outputs[0].value().elements::<u64>(), Ok(expected_states));
        assert_eq!(outputs[1].value().elements::<u32>(), Ok(expected_bits));
    }

    #[test]
    fn test_rng_bit_generator_batching_rejects_replicated_state() {
        let operation = RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type(5));
        let input = ArrayBatch::replicated(state(42, 7));
        let result = operation.batch(&batching_context(2), &EmptyRegionDriver, &[input]);
        assert!(matches!(
            result,
            Err(BatchingError::UnsupportedOperation { message })
                if message.contains("derive one state per batch item with `split_key`"),
        ));
    }

    #[test]
    fn test_rng_bit_generator_batching_stages_a_scan() {
        // Under a staging parent, batching a mapped state stages one carry-free scan over the per-item states, so
        // the batched program's size stays independent of the batch size.
        let operation = RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type(4));
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(RandomAlgorithm::ThreeFry.state_type());
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input]).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder, Placeholder])
            .unwrap();
        let (batched, output_axes) = program
            .batched(3, ShardingDimension::Replicated, &[BatchAxis::new(0)], ProgramBatchingOutputAxesPolicy::Natural)
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:u64[3, 2] .
                let %1:u64[3, 2], %2:u32[3, 4] = scan [carry_count=0, length=3, reverse=false] %0 [
                    body={
                        lambda %0:u64[2] .
                        let %1:u64[2], %2:u32[4] = rng_bit_generator [algorithm=three_fry, output_type=u32[4]] %0
                        in (%1, %2)
                    },
                ]
                in (%1, %2)
            "}
            .trim_end(),
        );

        // The batched program computes each batch item's unbatched result exactly.
        let states = Array::from_elements(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
            &[42u64, 7, 3, 11, 5, 0],
        )
        .unwrap();
        let outputs = batched.interpret(vec![states]).unwrap();
        let mut expected_states = Vec::new();
        let mut expected_bits = Vec::new();
        for (key, counter) in [(42, 7), (3, 11), (5, 0)] {
            let (advanced, bits) =
                state(key, counter).rng_bit_generator(RandomAlgorithm::ThreeFry, &bits_type(4)).unwrap();
            expected_states.extend(advanced.elements::<u64>().unwrap());
            expected_bits.extend(bits.elements::<u32>().unwrap());
        }
        assert_eq!(outputs[0].elements::<u64>(), Ok(expected_states));
        assert_eq!(outputs[1].elements::<u32>(), Ok(expected_bits));
    }

    #[test]
    fn test_mapped_rng_batching_stages_one_dynamic_composite_scan() -> Result<(), ProgramError> {
        type Context = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(7))?);
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(11))?);
        let trace = Context::new();
        let batch_extent = trace.input(DimensionType::new(batch.clone()).into());
        let states = trace.input(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]))
                .into(),
        );
        let row_extent = trace.input(DimensionType::new(rows.clone()).into());
        let column_extent = trace.input(DimensionType::new(columns.clone()).into());
        let input_ids = [batch_extent.clone(), states.clone(), row_extent.clone(), column_extent.clone()]
            .map(|input| input.atom_id().unwrap());
        let context = BatchingContext::<_, ArrayIrBatching>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayIrOperation::RngBitGenerator(RngBitGeneratorOperation::new(
                RandomAlgorithm::ThreeFry,
                ArrayType::new(
                    DataType::U32,
                    Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())]),
                ),
            )),
            Vec::new(),
            &[
                BatchingTracer::new(context.clone(), ArrayIrBatch::new(states, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(row_extent)),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(column_extent)),
            ],
        )?;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[1].batch().unbatched_type(),
            &ArrayIrType::Array(ArrayType::new(
                DataType::U32,
                Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())]),
            )),
        );

        let output_ids = outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            output_ids,
            vec![Placeholder; 4],
            vec![Placeholder; 2],
        )?;
        let [scan] = program.entry_region().instructions() else {
            panic!("mapped RNG batching should stage exactly one scan instruction");
        };
        let ArrayIrOperation::Scan(scan_operation) = scan.operation() else {
            panic!("mapped RNG batching should stage the direct composite scan carrier");
        };
        assert_eq!(scan_operation.carry_count(), 2);
        assert_eq!(scan_operation.length(), &Dimension::Dynamic(batch.clone()));
        assert_eq!(scan.inputs(), &[input_ids[2], input_ids[3], input_ids[1], input_ids[0]]);
        assert_eq!(scan.regions().len(), 1);
        assert!(matches!(
            program.region(scan.regions()[0])?.instructions()[0].operation(),
            ArrayIrOperation::RngBitGenerator(_),
        ));
        let rendered = program.to_string();
        let mut imported_builder = ProgramBuilder::new();
        let imported = imported_builder.import_region(program.entry_region_ref());
        assert_eq!(imported_builder.region_ref(imported)?.to_program().to_string(), rendered);

        // A second vmap structurally replays the already scan-decomposed RNG program. The inner runtime scan length
        // remains an explicit replicated dimension operand while the new mapped extent becomes its leading carry.
        let nested_trace = Context::new();
        let outer = DimensionVariable::new("outer", DimensionBounds::new(1, Some(5))?);
        let outer_extent = nested_trace.input(DimensionType::new(outer.clone()).into());
        let nested_context = BatchingContext::<_, ArrayIrBatching>::new(nested_trace, outer_extent);
        let nested = <ArrayIrBatching as RecursiveBatchingPolicy<Context>>::batch_program(
            &nested_context,
            program.entry_region_ref(),
            &[BatchAxis::replicated(), BatchAxis::new(0), BatchAxis::replicated(), BatchAxis::replicated()],
            ProgramBatchingOutputAxesPolicy::Natural,
        )?;
        assert_eq!(nested.output_axes(), &[BatchAxis::new(1), BatchAxis::new(1)]);
        let (nested, _) = nested.into_parts();
        assert_eq!(
            nested
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), ArrayIrOperation::Scan(_)))
                .count(),
            1,
        );
        assert_eq!(nested.input_types()[0], ArrayIrType::Dimension(DimensionType::new(outer)));

        Ok(())
    }

    #[test]
    fn test_philox_rng_bit_generator() {
        // The reference backend maps the `ui64[3]` state to `[key, counter]` with the 128-bit counter split into
        // its low and high `u64` halves.
        let state = Array::from_elements(RandomAlgorithm::Philox.state_type(), &[42u64, 7, 9]).unwrap();
        let counter = 7u128 | (9u128 << 64);

        // Five `u32` words run two cipher invocations, and the counter advances by that invocation count.
        let (advanced, bits) = state.rng_bit_generator(RandomAlgorithm::Philox, &bits_type(5)).unwrap();
        let (expected_words, expected_counter) = philox_u32_words(42, counter, 5);
        assert_eq!(expected_counter, counter + 2);
        assert_eq!(advanced.elements::<u64>(), Ok(vec![42, expected_counter as u64, (expected_counter >> 64) as u64]),);
        assert_eq!(bits.elements::<u32>(), Ok(expected_words));

        // Three `u64` words also run two cipher invocations (two words per invocation, truncated).
        let u64_bits_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3)]));
        let (advanced, bits) = state.rng_bit_generator(RandomAlgorithm::Philox, &u64_bits_type).unwrap();
        let (expected_words, expected_counter) = philox_u64_words(42, counter, 3);
        assert_eq!(expected_counter, counter + 2);
        assert_eq!(advanced.elements::<u64>(), Ok(vec![42, expected_counter as u64, (expected_counter >> 64) as u64]),);
        assert_eq!(bits.elements::<u64>(), Ok(expected_words));
    }

    #[test]
    fn test_split_key() {
        let parent = state(42, 7);
        let (advanced, fresh_states) = parent.split_key(3).unwrap();

        // Splitting draws three `u64` keys, so the parent state advances by three cipher invocations.
        let (expected_keys, expected_counter) = threefry_u64_words(42, 7, 3);
        assert_eq!(advanced.elements::<u64>(), Ok(vec![42, expected_counter]));
        assert_ne!(advanced, parent);

        // Each fresh state is a `u64[2]` seeded by one generated key with a zero counter, and the keys are distinct.
        assert_eq!(fresh_states.len(), 3);
        for (fresh_state, expected_key) in fresh_states.iter().zip(expected_keys) {
            assert_eq!(fresh_state.r#type().into_owned(), RandomAlgorithm::ThreeFry.state_type());
            assert_eq!(fresh_state.elements::<u64>(), Ok(vec![expected_key, 0]));
        }
        assert_ne!(fresh_states[0], fresh_states[1]);
        assert_ne!(fresh_states[0], fresh_states[2]);
        assert_ne!(fresh_states[1], fresh_states[2]);
    }

    #[test]
    fn test_uniform() {
        let shape = Shape::new(vec![Dimension::Static(4096)]);
        let parent = state(42, 7);
        let (_, samples) = parent.uniform(shape.clone(), DataType::F32).unwrap();
        assert_eq!(samples.r#type().into_owned(), ArrayType::new(DataType::F32, shape.clone()));
        let values = samples.to_f64s();
        assert!(values.iter().all(|value| (0.0..1.0).contains(value)));
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / values.len() as f64;
        assert!((mean - 0.5).abs() < 0.02);
        assert!((variance - 1.0 / 12.0).abs() < 0.01);

        // Randomness is functional: the same state produces the same samples.
        let (_, replayed) = parent.uniform(shape.clone(), DataType::F32).unwrap();
        assert_eq!(replayed, samples);

        // The `f64` recipe converts all 32 bits, and stays in `[0, 1)`.
        let (_, samples) = parent.uniform(shape.clone(), DataType::F64).unwrap();
        assert_eq!(samples.r#type().into_owned(), ArrayType::new(DataType::F64, shape));
        assert!(samples.to_f64s().iter().all(|value| (0.0..1.0).contains(value)));
    }

    #[test]
    fn test_normal() {
        let shape = Shape::new(vec![Dimension::Static(4096)]);
        let (_, samples) = state(42, 7).normal(shape.clone(), DataType::F64).unwrap();
        assert_eq!(samples.r#type().into_owned(), ArrayType::new(DataType::F64, shape));
        let values = samples.to_f64s();
        assert!(values.iter().all(|value| value.is_finite()));
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / values.len() as f64;
        assert!(mean.abs() < 0.05);
        assert!((variance - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_categorical() {
        // With a logit gap of 10, a non-peak draw has probability ~9e-5 per sample, so all 64 samples must pick the
        // peak category.
        let logits = Array::vector(vec![0.0, 10.0, 0.0]);
        let mut state = state(42, 0);
        for _ in 0..64 {
            let (advanced, sample) = state.categorical(&logits, 0).unwrap();
            assert_eq!(sample.r#type().into_owned(), ArrayType::scalar(DataType::I32));
            assert_eq!(sample.elements::<i32>(), Ok(vec![1]));
            state = advanced;
        }
    }
}
