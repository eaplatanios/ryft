//! Counter-based pseudorandom number generation. Randomness in Ryft is _functional_. Specifically, a generator state
//! is an ordinary `u64` array, the same state always produces the same bits, and every random draw returns an advanced
//! state that the caller threads into the next draw. There is no hidden generator state, so random programs remain
//! deterministic and every transform treats them like any other pure computation. The module is organized in three
//! layers:
//!
//!   - [`RandomAlgorithm`] selects a counter-based bit generator (e.g., ThreeFry-2x32 or Philox-4x32) and defines the
//!     type of its state.
//!   - [`RngBitGenerator`] generates raw unsigned-integer bits from a state by applying an [`RngBitGeneratorOperation`]
//!     which is analogous to StableHLO's [`rng_bit_generator`](https://openxla.org/stablehlo/spec#rng_bit_generator).
//!     [`DynamicRngBitGenerator`] does the same for dynamically shaped outputs in the mixed [`ArrayIrValue`] family.
//!   - [`Random`] composes those bits with ordinary array operations into key splitting and the uniform, normal, and
//!     categorical distributions, which therefore inherit their transform rules from the operations they use.
//!
//! # Example
//!
//! The following example splits a state into two independent states and draws uniform samples from one of them:
//!
//! ```rust
//! # use ryft_core::{Array, ArrayType, DataType, ProgramError, Random, RandomAlgorithm};
//! # fn main() -> Result<(), ProgramError> {
//! let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[42u64, 0])?;
//! let (_, keys) = state.split_rng_key(2)?;
//! let sample_type = ArrayType::new_static(DataType::F32, [3]);
//! let (_, samples) = keys[0].random_uniform(&sample_type)?;
//! assert!(samples.to_f64s().iter().all(|sample| (0.0..1.0).contains(sample)));
//!
//! // Drawing from the same state again reproduces the same samples, while a different state draws different ones.
//! assert_eq!(keys[0].random_uniform(&sample_type)?.1, samples);
//! assert_ne!(keys[1].random_uniform(&sample_type)?.1, samples);
//! # Ok(())
//! # }
//! ```

use std::fmt::Display;
use std::marker::PhantomData;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy,
    ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension, DimensionType, DimensionValue, DimensionVariable, Shape,
    ShardingDimension,
};
use crate::axes::Axis;
use crate::batching::{BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, impl_non_differentiable_operation, impl_non_transposable_operation,
    impl_reference_dischargeable_operation,
};
use crate::operations::arithmetic::{Add, Div, Mul, Neg, Sqrt, Sub};
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::scan::ScanOperation;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::exponential::Log;
use crate::operations::manipulation::broadcasting::DynamicBroadcastOperation;
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::operations::reductions::ArgMax;
use crate::operations::trigonometric::Cos;
use crate::parameters::Placeholder;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    Operation, OperationFormatter, OperationProjection, ProgramBuilder, ProgramError, RegionInterface, Type, TypeError,
    TypeIdentityRenaming, Typed, Value, ValueProjection,
};

/// Deterministic counter-based pseudorandom bit-generation algorithm used by an [`RngBitGeneratorOperation`].
/// Both algorithms come from [Salmon et al. paper](https://doi.org/10.1145/2063384.2063405), and their states hold
/// a `u64` key followed by a counter that every draw advances by the number of cipher invocations it performs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RandomAlgorithm {
    /// The 20-round ThreeFry-2x32 generator, whose `u64[2]` state holds `[key, counter]`.
    ThreeFry,

    /// The 10-round Philox-4x32 generator, whose `u64[3]` state holds the key followed by the low and high `u64` halves
    /// of its 128-bit counter. StableHLO also accepts a `u64[2]` Philox state that reuses the key as the high counter
    /// half, but Ryft requires the explicit three-element form so that every algorithm has exactly one state type.
    Philox,
}

impl RandomAlgorithm {
    /// Returns the algorithm whose [`state_type`](Self::state_type) has the data type and shape of `state_type`,
    /// ignoring layout, memory, and sharding metadata. This is how [`Random`] selects the algorithm of a state.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if `state_type` is neither `u64[2]` nor `u64[3]`.
    pub fn from_state_type(state_type: &ArrayType) -> Result<Self, TypeError> {
        match (state_type.data_type(), state_type.shape().dimensions()) {
            (DataType::U64, [Dimension::Static(2)]) => Ok(Self::ThreeFry),
            (DataType::U64, [Dimension::Static(3)]) => Ok(Self::Philox),
            _ => Err(TypeError::invalid(format!(
                "random generator states must have type `u64[2]` (i.e., for `three_fry`) or \
                 `u64[3]` (i.e., for `philox`) but got `{state_type}`",
            ))),
        }
    }

    /// Returns the type of the states consumed and produced by this algorithm, which is `u64[2]`
    /// for [`ThreeFry`](Self::ThreeFry) and `u64[3]` for [`Philox`](Self::Philox).
    #[inline]
    pub fn state_type(self) -> ArrayType {
        ArrayType::new_static(
            DataType::U64,
            [match self {
                Self::ThreeFry => 2,
                Self::Philox => 3,
            }],
        )
    }
}

impl Display for RandomAlgorithm {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ThreeFry => formatter.write_str("three_fry"),
            Self::Philox => formatter.write_str("philox"),
        }
    }
}

/// Canonical operation name for [`RngBitGeneratorOperation`].
pub const RNG_BIT_GENERATOR_OPERATION_NAME: &str = "rng_bit_generator";

/// [`Operation`] that deterministically generates uniformly distributed random bits from a counter-based generator
/// state. Refer to the documentation of [`RngBitGenerator`] for the generation semantics. The first input is the
/// generator state, and the two outputs are the advanced state followed by the generated bits at the declared
/// [`output_type`](Self::output_type).
///
/// The type parameter selects the input contract without introducing a separate bit-generation operation:
///
///   - `RngBitGeneratorOperation<ArrayType>` accepts only the state and requires a statically shaped output.
///   - `RngBitGeneratorOperation<ArrayIrType>` additionally accepts one trailing first-class dimension input per
///     dynamic axis of the declared output, in axis order, and each input must define the dimension variable that its
///     axis refers to. Note that, lowering when using the XLA backend rejects dynamic outputs, because generating the
///     physical upper-bound buffer would advance the state by the physical rather than the logical element count.
///
/// Both outputs are discrete, so differentiation assigns them structural-zero tangents and transposition is rejected.
/// Batching a replicated state binds this operation once and replicates both outputs, since every batch item computes
/// the same function of the same state. Batching a mapped state (e.g., one state per batch item derived with
/// [`Random::split_rng_key`]) stages one carry-free [`ScanOperation`] over the per-item states, so each batch item
/// draws exactly the bits that its own state produces unbatched and the staged program size is independent of the
/// batch size. The composite form threads its output extents through that scan as invariant carries, which requires
/// them to be replicated.
#[derive(Clone, Debug, PartialEq)]
pub struct RngBitGeneratorOperation<T: Type> {
    /// Algorithm generating the random bits.
    algorithm: RandomAlgorithm,

    /// Declared type of the generated random bits.
    output_type: ArrayType,

    /// Type universe whose input contract this payload represents.
    marker: PhantomData<fn() -> T>,
}

impl<T: Type> RngBitGeneratorOperation<T> {
    /// Creates a new [`RngBitGeneratorOperation`] with the provided algorithm and declared bits output type.
    #[inline]
    pub fn new(algorithm: RandomAlgorithm, output_type: ArrayType) -> Self {
        Self { algorithm, output_type, marker: PhantomData }
    }

    /// Returns the algorithm generating the bits for this [`RngBitGeneratorOperation`].
    #[inline]
    pub fn algorithm(&self) -> RandomAlgorithm {
        self.algorithm
    }

    /// Returns the declared type of the generated bits for this [`RngBitGeneratorOperation`].
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Validates the provided state type, the output element type, and the absence of sharded axes, which are shared
    /// by both input contracts, for this [`RngBitGeneratorOperation`].
    fn validate_types(&self, state_type: &ArrayType) -> Result<(), TypeError> {
        let algorithm = self.algorithm;
        let expected_state_type = algorithm.state_type();
        if state_type.data_type() != expected_state_type.data_type()
            || state_type.shape() != expected_state_type.shape()
        {
            return Err(TypeError::invalid(format!(
                "`{RNG_BIT_GENERATOR_OPERATION_NAME}` with the `{algorithm}` algorithm requires a \
                 `{expected_state_type}` state but got `{state_type}`",
            )));
        }

        let data_type = self.output_type.data_type();
        if !matches!(data_type, DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64) {
            return Err(TypeError::invalid(format!(
                "`{RNG_BIT_GENERATOR_OPERATION_NAME}` does not support output data type `{data_type}`",
            )));
        }

        let is_sharded = |array_type: &ArrayType| {
            array_type.sharding().is_some_and(|sharding| {
                sharding.dimensions().iter().any(|dimension| matches!(dimension, ShardingDimension::Sharded(_)))
            })
        };
        if is_sharded(state_type) || is_sharded(&self.output_type) {
            return Err(TypeError::invalid(format!(
                "`{RNG_BIT_GENERATOR_OPERATION_NAME}` does not support sharded states or outputs; derive per-shard \
                 states inside `shard_map` instead",
            )));
        }

        Ok(())
    }

    /// Renders this payload independently of its homogeneous or composite input contract. This is a separate function
    /// rather than the body of [`Operation::render`] because the [`ArrayType`] and [`ArrayIrType`] payloads have
    /// separate [`Operation`] implementations that must render identically, and both must forward their `indentation`
    /// so that [`OperationFormatter`] can lay out continuation lines. Inherent functions take precedence over trait
    /// functions during method resolution, so the `self.render(...)` calls in those implementations and in the
    /// [`Display`] implementation resolve to this function rather than recursing into [`Operation::render`].
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, RNG_BIT_GENERATOR_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("algorithm", self.algorithm)?;
            operation.field("output_type", &self.output_type)
        })
    }
}

impl<T: Type> Display for RngBitGeneratorOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

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
        self.validate_types(&input_types[0])?;
        if self.output_type.static_shape().is_none() {
            return Err(TypeError::invalid(format!(
                "`{RNG_BIT_GENERATOR_OPERATION_NAME}` does not support dynamically shaped outputs",
            )));
        }
        Ok(vec![input_types[0].clone(), self.output_type.clone()])
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self::new(self.algorithm, self.output_type.rename_identities(renaming)?))
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render(formatter, indentation)
    }
}

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
        // The state is followed by one first-class extent input per dynamic output axis, and each extent
        // must define the dimension variable that the corresponding declared output axis refers to.
        check_count!("region", region_interfaces, 0, TypeError);
        let dynamic_output_dimensions =
            self.output_type.shape().dimensions().iter().filter_map(Dimension::variable).collect::<Vec<_>>();
        check_count!("input", input_types, dynamic_output_dimensions.len() + 1, TypeError);
        let state_type = <&ArrayType>::try_from(&input_types[0])?;
        self.validate_types(state_type)?;
        for (input_type, expected_variable) in input_types[1..].iter().zip(dynamic_output_dimensions) {
            let actual_variable = <&DimensionType>::try_from(input_type)?.variable();
            if actual_variable != expected_variable {
                return Err(TypeError::invalid(format!(
                    "`{RNG_BIT_GENERATOR_OPERATION_NAME}` output-extent input defines dimension variable \
                     `{actual_variable}`, but the corresponding declared output axis refers to `{expected_variable}`",
                )));
            }
        }
        Ok(vec![state_type.clone().into(), self.output_type.clone().into()])
    }

    #[inline]
    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<DimensionVariable>) -> Result<Self, TypeError> {
        Ok(Self::new(self.algorithm, self.output_type.rename_identities(renaming)?))
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render(formatter, indentation)
    }
}

impl_reference_dischargeable_operation!(@reference_free <T> RngBitGeneratorOperation<T> where T: Type);

impl<C: Domain<Type = ArrayType, Value: RngBitGenerator>> InterpretableOperation<C>
    for RngBitGeneratorOperation<ArrayType>
{
    #[inline]
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

impl<C: Domain<Type = ArrayIrType, Value: DynamicRngBitGenerator>> InterpretableOperation<C>
    for RngBitGeneratorOperation<ArrayIrType>
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let Some((state, output_dimensions)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
        };
        let (state, bits) = state.dynamic_rng_bit_generator(self.algorithm, &self.output_type, output_dimensions)?;
        Ok(vec![state, bits])
    }
}

impl<T: Type, C: Context<Type = T, Operation: From<RngBitGeneratorOperation<T>>>> PartiallyEvaluatableOperation<C>
    for RngBitGeneratorOperation<T>
where
    Self: Operation<Type = T>,
{
}

impl<
    C: Context<
            Type = ArrayType,
            Value: Transpose,
            Operation: From<RngBitGeneratorOperation<ArrayType>> + From<ScanOperation<C::Constant>>,
        >,
    P: ArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for RngBitGeneratorOperation<ArrayType>
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);

        // Every batch item of a replicated state computes the same function of the same state, so generating the
        // bits once and replicating both outputs is exact.
        if inputs[0].batch_axis().is_replicated() {
            let mut outputs =
                context.parent().bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].value()))?;
            check_count!("output", outputs, 2, ProgramError);
            let bits = outputs.remove(1);
            let state = outputs.remove(0);
            return Ok(vec![ArrayBatch::replicated(state), ArrayBatch::replicated(bits)].into());
        }

        // A mapped state is realigned to batch axis 0 and one carry-free scan is staged over it, whose body applies
        // this same operation to a single per-item state. Iteration `i` consumes state row `i` and yields that item's
        // advanced state and bits, which the scan stacks at batch axis 0. Binding the scan through the parent context
        // lets an enclosing batching context batch it structurally.
        let state_type = inputs[0].unbatched_type();
        let states = inputs[0].move_axis(0)?;
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64));
        let state = builder.add_input(state_type);
        let outputs = builder.add_instruction(self.clone(), Vec::new(), vec![state], None)?.to_vec();
        let body =
            builder.build::<Vec<C::Constant>, Vec<C::Constant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])?;
        let scan = ScanOperation::<C::Constant>::new(0, P::axis_size(context)?);
        let mut outputs = context.parent().bind(scan, vec![body], std::slice::from_ref(states.value()))?;
        check_count!("output", outputs, 2, ProgramError);
        let bits = outputs.remove(1);
        let states = outputs.remove(0);
        Ok(vec![ArrayBatch::new(states, BatchAxis::new(0))?, ArrayBatch::new(bits, BatchAxis::new(0))?].into())
    }
}

impl<C: Context<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatchingPolicy>
    for RngBitGeneratorOperation<ArrayIrType>
where
    C::Value: ValueProjection<ArrayType, Projected: Transpose + Value<Type = ArrayType>>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + From<RngBitGeneratorOperation<ArrayIrType>>
        + From<ScanOperation<C::Constant>>
        + OperationProjection<ArrayType, Projected: From<TransposeOperation>>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayIrBatchingPolicy>, BatchingError> {
        let Some((state, output_dimensions)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };

        for output_dimension in output_dimensions {
            output_dimension.validate_replicated_dimension()?;
        }

        // Every batch item of a replicated state computes the same function of the same state, so generating the
        // bits once and replicating both outputs is exact.
        if state.batch_axis().is_replicated() {
            let inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
            let mut outputs = context.parent().bind(self.clone(), Vec::new(), &inputs)?;
            check_count!("output", outputs, 2, ProgramError);
            let bits = outputs.remove(1);
            let state = outputs.remove(0);
            return Ok(vec![ArrayIrBatch::replicated(state), ArrayIrBatch::replicated(bits)].into());
        }

        // A mapped state is realigned to batch axis 0 and consumed one row per iteration of a scan whose replicated
        // output extents are invariant carries. This keeps one independently advanced state and one dynamically shaped
        // bits value per batch item without duplicating the generator state.
        let state_type = state.unbatched_type();
        let state = driver.align_batch_axis(context, state.clone(), Axis::from(0))?;
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64).into());
        let extent_inputs = output_dimensions
            .iter()
            .map(|output_dimension| builder.add_input(output_dimension.unbatched_type()))
            .collect::<Vec<_>>();
        let state_input = builder.add_input(state_type);
        let operation_inputs = std::iter::once(state_input).chain(extent_inputs.iter().copied()).collect::<Vec<_>>();
        let random_outputs = builder.add_instruction(self.clone(), Vec::new(), operation_inputs, None)?.to_vec();
        let body_outputs = extent_inputs.iter().copied().chain(random_outputs).collect::<Vec<_>>();
        let body = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            body_outputs,
            vec![Placeholder; output_dimensions.len() + 2],
            vec![Placeholder; output_dimensions.len() + 2],
        )?;

        // A dynamic batch extent becomes the scan length, which the scan consumes as a trailing runtime input.
        let extent_type = context.axis_extent().r#type();
        let length = <&DimensionType>::try_from(extent_type.as_ref())?.to_dimension();
        let scan = ScanOperation::<C::Constant>::new(output_dimensions.len(), length.clone());
        let mut scan_inputs = output_dimensions
            .iter()
            .map(|output_dimension| output_dimension.value().clone())
            .collect::<Vec<_>>();
        scan_inputs.push(state.into_value());
        if length.variable().is_some() {
            scan_inputs.push(context.axis_extent().clone());
        }
        let mut outputs = context.parent().bind(scan, vec![body], scan_inputs.as_slice())?;
        check_count!("output", outputs, output_dimensions.len() + 2, ProgramError);
        let bits = outputs.remove(output_dimensions.len() + 1);
        let states = outputs.remove(output_dimensions.len());
        Ok(vec![ArrayIrBatch::new(states, BatchAxis::new(0))?, ArrayIrBatch::new(bits, BatchAxis::new(0))?].into())
    }
}

impl_non_differentiable_operation!(<T> RngBitGeneratorOperation<T> where T: Type);

// Random bits are discrete and therefore never form a linear map that can be transposed.
impl_non_transposable_operation!(<T> RngBitGeneratorOperation<T> where T: Type);

/// Represents the ability to generate deterministic, uniformly distributed random bits from a counter-based generator
/// state, which is the primitive underlying [`Random`]. Randomness is _functional_. Specifically, the same state always
/// produces the same bits and the same advanced state, and drawing again requires threading the advanced state or
/// deriving fresh states with [`Random::split_rng_key`].
///
/// The state must have the [`RandomAlgorithm::state_type`] of the requested algorithm, and the output element type must
/// be `u8`, `u16`, `u32`, or `u64`. Narrower outputs keep the low bits of one 32-bit word per element. The output must
/// be statically shaped (refer to [`DynamicRngBitGenerator`] for dynamic shape support), and neither the state nor the
/// output may be sharded, since every shard would otherwise draw the same bits. For XLA's `shard_map` operation, for
/// example, you must derive per-shard states inside that operation instead. Concrete [`Array`]s generate the bits
/// immediately, bit-identical with XLA's
/// [`rng_bit_generator`](https://github.com/openxla/xla/blob/main/xla/hlo/builder/lib/prng.cc) expansion, while
/// context-carrying values bind an [`RngBitGeneratorOperation`] through their own context. The bits are integers,
/// and so their derivative is a structural zero.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ArrayType, DataType, ProgramError, RandomAlgorithm, RngBitGenerator};
/// # fn main() -> Result<(), ProgramError> {
/// let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[42u64, 0])?;
/// let output_type = ArrayType::new_static(DataType::U32, [4]);
/// let (advanced_state, bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type)?;
///
/// // Four 32-bit words take two ThreeFry invocations, so the counter advances by two.
/// assert_eq!(advanced_state.elements::<u64>()?, vec![42, 2]);
///
/// // The same state always reproduces the same bits.
/// assert_eq!(state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type)?.1, bits);
/// # Ok(())
/// # }
/// ```
pub trait RngBitGenerator: Sized {
    /// Generates random bits of `output_type` from this generator state using `algorithm`, returning the advanced state
    /// together with the bits.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if this state does not have the state type of `algorithm`, if `output_type` is not
    /// a statically shaped and unsharded unsigned-integer type, or if the context of the value fails to bind the
    /// operation.
    fn rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
    ) -> Result<(Self, Self), ProgramError>;
}

impl RngBitGenerator for Array {
    fn rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
    ) -> Result<(Self, Self), ProgramError> {
        RngBitGeneratorOperation::<ArrayType>::new(algorithm, output_type.clone())
            .infer_output_types(&[self.r#type().into_owned()], &[])?;

        // Type inference guarantees a static output shape, an unsigned-integer output data type, and exactly as many
        // decoded `u64` state elements as the algorithm requires. Narrower-than-32-bit outputs keep the low bits of
        // each generated `u32` word. The generated values are encoded in logical order, so the declared physical
        // layouts of both the state and the bits are preserved.
        let dimensions = output_type.static_shape().unwrap().dimensions().to_vec();
        let count = dimensions.iter().product::<usize>();
        let data_type = output_type.data_type();
        let bits_from_u32_words = |words: Vec<u32>| match data_type {
            DataType::U8 => Array::from_fn_elements(output_type.clone(), |index| Ok(words[index] as u8)),
            DataType::U16 => Array::from_fn_elements(output_type.clone(), |index| Ok(words[index] as u16)),
            _ => Array::from_elements(output_type.clone(), &words),
        };

        let state = self.elements::<u64>()?;
        let key = state[0];
        let (advanced_state, bits) = match algorithm {
            RandomAlgorithm::ThreeFry => {
                let counter = state[1];
                let (counter, bits) = if data_type == DataType::U64 {
                    let (words, counter) = threefry_u64_words(key, counter, count);
                    (counter, Array::from_elements(output_type.clone(), &words)?)
                } else {
                    let (words, counter) = threefry_u32_words(key, counter, &dimensions);
                    (counter, bits_from_u32_words(words)?)
                };
                (vec![key, counter], bits)
            }
            RandomAlgorithm::Philox => {
                let counter = u128::from(state[1]) | (u128::from(state[2]) << 64);
                let (counter, bits) = if data_type == DataType::U64 {
                    let (words, counter) = philox_u64_words(key, counter, count);
                    (counter, Array::from_elements(output_type.clone(), &words)?)
                } else {
                    let (words, counter) = philox_u32_words(key, counter, count);
                    (counter, bits_from_u32_words(words)?)
                };
                (vec![key, counter as u64, (counter >> 64) as u64], bits)
            }
        };

        Ok((Array::from_elements(self.r#type().into_owned(), &advanced_state)?, bits))
    }
}

impl<
    V: Value<
            Type = ArrayType,
            DispatchDomain: Context<Type = ArrayType, Operation: From<RngBitGeneratorOperation<ArrayType>>>,
        >,
> RngBitGenerator for V
{
    fn rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
    ) -> Result<(Self, Self), ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            RngBitGeneratorOperation::<ArrayType>::new(algorithm, output_type.clone()),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 2, ProgramError);
        let bits = outputs.remove(1);
        let state = outputs.remove(0);
        Ok((state, bits))
    }
}

/// Represents the ability to generate random bits whose output shape may be dynamic, in the mixed [`ArrayIrValue`]
/// family. [`Self::dynamic_rng_bit_generator`] follows the semantics of [`RngBitGenerator`], except that each
/// dynamic axis of the declared output type takes its runtime extent from one first-class dimension value. Concrete
/// [`ArrayIrValue`]s generate exactly the bits that [`RngBitGenerator`] generates for the resolved static shape, while
/// context-carrying values bind an [`RngBitGeneratorOperation<ArrayIrType>`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayIrValue, ArrayType, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue,
/// #     DimensionVariable, DynamicRngBitGenerator, ProgramError, RandomAlgorithm, RngBitGenerator, Shape,
/// # };
/// # fn main() -> Result<(), ProgramError> {
/// let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[42u64, 0])?;
/// let count = DimensionVariable::new("count", DimensionBounds::unbounded());
/// let output_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(count.clone())]));
/// let extent = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::from(count), 3)?);
/// let (_, bits) = ArrayIrValue::Array(state.clone()).dynamic_rng_bit_generator(
///     RandomAlgorithm::ThreeFry,
///     &output_type,
///     &[extent],
/// )?;
/// let static_output_type = ArrayType::new_static(DataType::U32, [3]);
/// assert_eq!(bits, ArrayIrValue::Array(state.rng_bit_generator(RandomAlgorithm::ThreeFry, &static_output_type)?.1));
/// # Ok(())
/// # }
/// ```
pub trait DynamicRngBitGenerator: Value<Type = ArrayIrType> + Sized {
    /// Generates random bits of `output_type` from this generator state using `algorithm`, returning the advanced state
    /// together with the bits.
    ///
    /// # Parameters
    ///
    ///   - `algorithm`: [`RandomAlgorithm`] generating the bits, whose state type this state must have.
    ///   - `output_type`: Declared type of the generated bits, which may contain dynamic dimensions.
    ///   - `output_dimensions`: One dimension value per dynamic axis of `output_type`, in axis order. Each value must
    ///     define the dimension variable that its axis refers to and supplies that axis's runtime extent.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the state, the output type, or the output dimensions violate the contract of
    /// [`RngBitGeneratorOperation<ArrayIrType>`], or if the context of the value fails to bind the operation.
    fn dynamic_rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
        output_dimensions: &[Self],
    ) -> Result<(Self, Self), ProgramError>;
}

impl<A: Value<Type = ArrayType> + RngBitGenerator> DynamicRngBitGenerator for ArrayIrValue<A> {
    fn dynamic_rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
        output_dimensions: &[Self],
    ) -> Result<(Self, Self), ProgramError> {
        let input_types = std::iter::once(self)
            .chain(output_dimensions)
            .map(|input| input.r#type().into_owned())
            .collect::<Vec<_>>();
        RngBitGeneratorOperation::<ArrayIrType>::new(algorithm, output_type.clone())
            .infer_output_types(&input_types, &[])?;

        // Type inference guarantees one dimension input per dynamic output axis, in axis order.
        let mut output_dimensions = output_dimensions.iter();
        let dimensions = output_type
            .shape()
            .dimensions()
            .iter()
            .map(|dimension| match dimension {
                Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
                Dimension::Dynamic(_) => {
                    let output_dimension = output_dimensions.next().unwrap();
                    Ok(Dimension::Static(
                        <Self as ValueProjection<DimensionType>>::projected(output_dimension)?.extent(),
                    ))
                }
            })
            .collect::<Result<Vec<_>, TypeError>>()?;
        let output_type = output_type.clone().with_shape(Shape::new(dimensions));
        let state = <Self as ValueProjection<ArrayType>>::projected(self)?;
        let (state, bits) = state.rng_bit_generator(algorithm, &output_type)?;
        Ok((Self::Array(state), Self::Array(bits)))
    }
}

impl<
    V: Value<
            Type = ArrayIrType,
            DispatchDomain: Context<Type = ArrayIrType, Operation: From<RngBitGeneratorOperation<ArrayIrType>>>,
        >,
> DynamicRngBitGenerator for V
{
    fn dynamic_rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
        output_dimensions: &[Self],
    ) -> Result<(Self, Self), ProgramError> {
        let inputs = std::iter::once(self).chain(output_dimensions).cloned().collect::<Vec<_>>();
        let mut outputs = self.dispatch_domain().bind(
            RngBitGeneratorOperation::<ArrayIrType>::new(algorithm, output_type.clone()),
            Vec::new(),
            &inputs,
        )?;
        check_count!("output", outputs, 2, ProgramError);
        let bits = outputs.remove(1);
        let state = outputs.remove(0);
        Ok((state, bits))
    }
}

/// Represents the ability to draw random samples from a counter-based generator state. Every function selects the
/// [`RandomAlgorithm`] from the state type (see [`RandomAlgorithm::from_state_type`]), threads the state functionally
/// by returning the advanced state alongside its result, and is a pure composition of [`RngBitGenerator`] and ordinary
/// array operations, so the distributions inherit their transform rules from those operations. The recipes are:
///
///   - [`split_rng_key`](Self::split_rng_key) draws one `u64` key per fresh state and pairs it with a zero counter.
///   - [`random_uniform`](Self::random_uniform) draws 32-bit words for `f32` samples and 64-bit words for `f64`
///     samples, keeps their top 24 or 53 bits (i.e., the precision of the sample data type), and scales them by `2⁻²⁴`
///     or `2⁻⁵³`. Every step is exact, so the samples are multiples of that spacing in `[0, 1 - 2⁻²⁴]` or
///     `[0, 1 - 2⁻⁵³]` and never round up to `1`.
///   - [`random_normal`](Self::random_normal) applies the Box-Muller transform `√(-2 ln(1 - u₁)) · cos(2π u₂)` to two
///     uniform draws, where `1 - u₁ > 0` keeps the logarithm finite.
///   - [`random_categorical`](Self::random_categorical) applies the Gumbel-max trick
///     `argmax(logits - ln(-ln(u + tiny)))` along the category axis, where `tiny` is the smallest positive normal value
///     of the logits data type. The Gumbel noise is therefore always finite, so `-∞` logits are never sampled unless
///     every logit along the axis is `-∞`, and ties resolve to the lowest index.
///
/// These are the distributions of JAX's
/// [`jax.random.uniform`](https://docs.jax.dev/en/latest/_autosummary/jax.random.uniform.html),
/// [`jax.random.normal`](https://docs.jax.dev/en/latest/_autosummary/jax.random.normal.html), and
/// [`jax.random.categorical`](https://docs.jax.dev/en/latest/_autosummary/jax.random.categorical.html), although the
/// samples differ bitwise because JAX derives its bits differently and draws normal samples through the inverse error
/// function.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ArrayType, DataType, ProgramError, Random, RandomAlgorithm};
/// # fn main() -> Result<(), ProgramError> {
/// let state = Array::from_elements(RandomAlgorithm::Philox.state_type(), &[7u64, 0, 0])?;
/// let (state, normal) = state.random_normal(&ArrayType::new_static(DataType::F64, [2, 3]))?;
/// assert!(normal.to_f64s().iter().all(|sample| sample.is_finite()));
///
/// // The masked category has probability zero and the second category dominates the remaining two.
/// let logits = Array::matrix(2, 3, vec![f64::NEG_INFINITY, 20.0, 0.0, f64::NEG_INFINITY, 0.0, 20.0])?;
/// let (_, samples) = state.random_categorical(&logits, -1)?;
/// assert_eq!(samples, Array::vector(vec![1i32, 2])?);
/// # Ok(())
/// # }
/// ```
pub trait Random: Sized {
    /// Splits this generator state into `count` fresh, statistically independent states of the same algorithm,
    /// returning the advanced state followed by the fresh states.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if this value is not a generator state or if the context of the value fails to bind
    /// an operation.
    fn split_rng_key(&self, count: usize) -> Result<(Self, Vec<Self>), ProgramError>;

    /// Draws uniformly distributed samples in `[0, 1)` of type `r#type`, returning the advanced state together with
    /// the samples. The samples have the shape, element data type, and memory space of `r#type`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if this value is not a generator state, if `r#type` is not statically shaped, if its
    /// element data type is neither `f32` nor `f64`, if it is sharded, or if the context of the value fails to bind an
    /// operation.
    fn random_uniform(&self, r#type: &ArrayType) -> Result<(Self, Self), ProgramError>;

    /// Draws standard-normal samples of type `r#type`, returning the advanced state together with the samples. The
    /// samples have the shape, element data type, and memory space of `r#type`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::random_uniform`].
    fn random_normal(&self, r#type: &ArrayType) -> Result<(Self, Self), ProgramError>;

    /// Draws one categorical sample for every position of `logits` outside `axis`, where the values along `axis` are
    /// unnormalized log-probabilities, returning the advanced state together with the sampled `i32` indices. The
    /// sample shape is the shape of `logits` with `axis` removed.
    ///
    /// # Parameters
    ///
    ///   - `logits`: `f32` or `f64` unnormalized log-probabilities. `-∞` entries are never sampled unless every entry
    ///     along `axis` is `-∞`.
    ///   - `axis`: [`Axis`] holding the categories. Negative axes count from the end.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if this value is not a generator state, if `logits` is neither an `f32` nor an
    /// `f64` statically shaped array, if `axis` is out of bounds or empty, or if the context of the value fails to
    /// bind an operation.
    fn random_categorical<A: Into<Axis>>(&self, logits: &Self, axis: A) -> Result<(Self, Self), ProgramError>;
}

impl<
    V: Value<Type = ArrayType, DispatchDomain: Fill<f64, V>>
        + ZeroLike
        + Neg
        + Add
        + Sub
        + Mul
        + Div
        + Sqrt
        + Log
        + Cos
        + ArgMax
        + Concatenate
        + Slice
        + ConvertElementType
        + RngBitGenerator,
> Random for V
{
    fn split_rng_key(&self, count: usize) -> Result<(Self, Vec<Self>), ProgramError> {
        let state_type = self.r#type();
        let algorithm = RandomAlgorithm::from_state_type(&state_type)?;
        let (state, keys) = self.rng_bit_generator(algorithm, &ArrayType::new_static(DataType::U64, [count]))?;

        // Every fresh state pairs one generated key with a zero counter of the parent state's algorithm.
        let state_width = state_type.dimension(0).value().unwrap();
        let zero_counter = self.slice(&[1], &[state_width], &[1])?.zero_like()?;
        let fresh_states = (0..count)
            .map(|index| Concatenate::concatenate([&keys.slice(&[index], &[index + 1], &[1])?, &zero_counter], 0))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((state, fresh_states))
    }

    fn random_uniform(&self, r#type: &ArrayType) -> Result<(Self, Self), ProgramError> {
        let data_type = r#type.data_type();
        let (bits_data_type, bit_count, precision) = match data_type {
            DataType::F32 => (DataType::U32, 32, 24),
            DataType::F64 => (DataType::U64, 64, 53),
            data_type => {
                return Err(TypeError::invalid(format!(
                    "`random_uniform` does not support output data type `{data_type}`"
                ))
                .into());
            }
        };
        let algorithm = RandomAlgorithm::from_state_type(&self.r#type())?;

        // The bits share the shape, memory space, and sharding of the samples, but not their physical layout, whose
        // strides depend on the element size.
        let bits_type = r#type.clone().with_data_type(bits_data_type).with_layout(None);
        let (state, bits) = self.rng_bit_generator(algorithm, &bits_type)?;

        // Dividing by `2^(bit_count - precision)` keeps the top `precision` bits as an integer below `2^precision`,
        // which `data_type` represents exactly, so the conversion and the scaling by `2^-precision` are exact as well.
        let domain = self.dispatch_domain();
        let divisor: Self = domain.fill(&bits_type, 2.0f64.powi(bit_count - precision))?;
        let scale: Self = domain.fill(r#type, 2.0f64.powi(-precision))?;
        Ok((state, bits.div(&divisor)?.convert_element_type(data_type)?.mul(&scale)?))
    }

    fn random_normal(&self, r#type: &ArrayType) -> Result<(Self, Self), ProgramError> {
        let (state, first) = self.random_uniform(r#type)?;
        let (state, second) = state.random_uniform(r#type)?;
        let domain = self.dispatch_domain();
        let one: Self = domain.fill(r#type, 1.0)?;
        let minus_two: Self = domain.fill(r#type, -2.0)?;
        let two_pi: Self = domain.fill(r#type, std::f64::consts::TAU)?;
        let radius = one.sub(&first)?.log()?.mul(&minus_two)?.sqrt()?;
        let angle = second.mul(&two_pi)?.cos()?;
        Ok((state, radius.mul(&angle)?))
    }

    fn random_categorical<A: Into<Axis>>(&self, logits: &Self, axis: A) -> Result<(Self, Self), ProgramError> {
        let logits_type = logits.r#type();
        let tiny = match logits_type.data_type() {
            DataType::F32 => f64::from(f32::MIN_POSITIVE),
            DataType::F64 => f64::MIN_POSITIVE,
            data_type => {
                return Err(TypeError::invalid(format!(
                    "`random_categorical` does not support logits data type `{data_type}`"
                ))
                .into());
            }
        };
        let (state, uniform) = self.random_uniform(&logits_type)?;

        // Shifting the samples from `[0, 1)` to `[tiny, 1)` keeps the Gumbel noise `-ln(-ln(u + tiny))` finite.
        // The shift only changes `u = 0`, because `tiny` is far below the spacing between non-zero samples.
        let tiny: Self = self.dispatch_domain().fill(uniform.r#type().as_ref(), tiny)?;
        let gumbel = uniform.add(&tiny)?.log()?.neg()?.log()?.neg()?;
        Ok((state, logits.add(&gumbel)?.argmax(axis)?))
    }
}

/// Applies the 20-round ThreeFry-2x32 block cipher to one counter pair under the provided key pair, bit-identical with
/// XLA's [implementation](https://github.com/openxla/xla/blob/main/xla/hlo/builder/lib/prng.cc). The rounds are grouped
/// in blocks of four with alternating rotation groups `[13, 15, 26, 6]` and `[17, 29, 16, 24]`, and the key injections
/// are derived from `key[0]`, `key[1]`, and `key[0] ^ key[1] ^ 0x1BD11BDA`.
fn threefry_2x32(key: [u32; 2], counter: [u32; 2]) -> [u32; 2] {
    const ROTATIONS: [u32; 8] = [13, 15, 26, 6, 17, 29, 16, 24];
    let key_schedule = [key[0], key[1], key[0] ^ key[1] ^ 0x1BD11BDA];
    let mut words = [counter[0].wrapping_add(key_schedule[0]), counter[1].wrapping_add(key_schedule[1])];
    for block in 0..5usize {
        for round in 0..4 {
            words[0] = words[0].wrapping_add(words[1]);
            words[1] = words[1].rotate_left(ROTATIONS[(block % 2) * 4 + round]);
            words[1] ^= words[0];
        }
        words[0] = words[0].wrapping_add(key_schedule[(block + 1) % 3]);
        words[1] = words[1].wrapping_add(key_schedule[(block + 2) % 3].wrapping_add(block as u32 + 1));
    }
    words
}

/// Generates the `u32` words of an output with the provided `dimensions` from a ThreeFry `[key, counter]` state,
/// returning the words in row-major order together with the advanced counter. The layout is XLA's: the split axis is
/// the first even-sized axis, or the first largest axis if none is even, and the cipher runs once per element of the
/// half shape, which halves the split axis (rounding up). The invocation at half index `(…, h, …)` uses counter
/// `counter + i`, where `i` is the row-major index of `(…, h, …)` in the half shape, and its two words land at
/// `(…, 2h, …)` and `(…, 2h + 1, …)`, dropping the last word of an odd split axis. A scalar uses the first word
/// of one invocation. The counter advances by the number of invocations that ran.
fn threefry_u32_words(key: u64, counter: u64, dimensions: &[usize]) -> (Vec<u32>, u64) {
    let key = [key as u32, (key >> 32) as u32];
    let dimensions = if dimensions.is_empty() { &[1][..] } else { dimensions };
    let split_axis = dimensions.iter().position(|dimension| dimension % 2 == 0).unwrap_or_else(|| {
        let largest = dimensions.iter().max().unwrap();
        dimensions.iter().position(|dimension| dimension == largest).unwrap()
    });
    let split_size = dimensions[split_axis];
    let half_split_size = split_size.div_ceil(2);
    let outer_size = dimensions[..split_axis].iter().product::<usize>();
    let inner_size = dimensions[split_axis + 1..].iter().product::<usize>();
    let mut words = vec![0; outer_size * split_size * inner_size];
    for outer_index in 0..outer_size {
        for half_index in 0..half_split_size {
            for inner_index in 0..inner_size {
                let invocation_index = (outer_index * half_split_size + half_index) * inner_size + inner_index;
                let invocation_counter = counter.wrapping_add(invocation_index as u64);
                let output = threefry_2x32(key, [invocation_counter as u32, (invocation_counter >> 32) as u32]);
                for (word_index, word) in output.into_iter().enumerate() {
                    let split_index = 2 * half_index + word_index;
                    if split_index < split_size {
                        words[(outer_index * split_size + split_index) * inner_size + inner_index] = word;
                    }
                }
            }
        }
    }
    (words, counter.wrapping_add((outer_size * half_split_size * inner_size) as u64))
}

/// Generates `count` `u64` words from a ThreeFry `[key, counter]` state, returning the words together with the advanced
/// counter. Following XLA, word `i` combines the two cipher words of counter `counter + i` as `first | (second << 32)`,
/// and the counter advances by `count`.
fn threefry_u64_words(key: u64, counter: u64, count: usize) -> (Vec<u64>, u64) {
    let key = [key as u32, (key >> 32) as u32];
    let words = (0..count)
        .map(|index| {
            let word_counter = counter.wrapping_add(index as u64);
            let output = threefry_2x32(key, [word_counter as u32, (word_counter >> 32) as u32]);
            u64::from(output[0]) | (u64::from(output[1]) << 32)
        })
        .collect();
    (words, counter.wrapping_add(count as u64))
}

/// Applies the 10-round Philox-4x32 block cipher to one counter quad under the provided key pair, bit-identical with
/// XLA's [implementation](https://github.com/openxla/xla/blob/main/xla/hlo/builder/lib/prng.cc). Each round multiplies
/// two counter words by the `u32` constants `0xD2511F53` and `0xCD9E8D57` into 64-bit products, and the key words are
/// incremented by `0x9E3779B9` and `0xBB67AE85` after every round.
fn philox_4x32(key: [u32; 2], counter: [u32; 4]) -> [u32; 4] {
    const MULTIPLIERS: [u32; 2] = [0xD2511F53, 0xCD9E8D57];
    const KEY_INCREMENTS: [u32; 2] = [0x9E3779B9, 0xBB67AE85];
    let mut key = key;
    let mut words = counter;
    for _ in 0..10 {
        let first_product = u64::from(words[0]) * u64::from(MULTIPLIERS[0]);
        let second_product = u64::from(words[2]) * u64::from(MULTIPLIERS[1]);
        words = [
            (second_product >> 32) as u32 ^ words[1] ^ key[0],
            second_product as u32,
            (first_product >> 32) as u32 ^ words[3] ^ key[1],
            first_product as u32,
        ];
        key = [key[0].wrapping_add(KEY_INCREMENTS[0]), key[1].wrapping_add(KEY_INCREMENTS[1])];
    }
    words
}

/// Generates the cipher words of `invocation_count` consecutive Philox counters starting at `counter`, with each
/// 128-bit counter split into four `u32` words (least significant first) and the key split into its low and high
/// `u32` halves.
fn philox_invocations(key: u64, counter: u128, invocation_count: usize) -> impl Iterator<Item = [u32; 4]> {
    let key = [key as u32, (key >> 32) as u32];
    (0..invocation_count).map(move |index| {
        let invocation_counter = counter.wrapping_add(index as u128);
        philox_4x32(
            key,
            [
                invocation_counter as u32,
                (invocation_counter >> 32) as u32,
                (invocation_counter >> 64) as u32,
                (invocation_counter >> 96) as u32,
            ],
        )
    })
}

/// Generates `count` `u32` words from a Philox `[key, counter]` state, returning the words together with the advanced
/// 128-bit counter. Following XLA, the four cipher words of counter `counter + i` land at positions `4i` through
/// `4i + 3` in row-major order regardless of the output shape, the final invocation is truncated to `count` words,
/// and the counter advances by the `ceil(count / 4)` invocations that ran.
fn philox_u32_words(key: u64, counter: u128, count: usize) -> (Vec<u32>, u128) {
    let invocation_count = count.div_ceil(4);
    let mut words = philox_invocations(key, counter, invocation_count).flatten().collect::<Vec<_>>();
    words.truncate(count);
    (words, counter.wrapping_add(invocation_count as u128))
}

/// Generates `count` `u64` words from a Philox `[key, counter]` state, returning the words together with the advanced
/// 128-bit counter. Following XLA, the cipher words `[first, second, third, fourth]` of counter `counter + i` form the
/// words `first | (second << 32)` and `third | (fourth << 32)` at positions `2i` and `2i + 1`, the final invocation is
/// truncated to `count` words, and the counter advances by the `ceil(count / 2)` invocations that ran.
fn philox_u64_words(key: u64, counter: u128, count: usize) -> (Vec<u64>, u128) {
    let invocation_count = count.div_ceil(2);
    let mut words = philox_invocations(key, counter, invocation_count)
        .flat_map(|output| {
            [u64::from(output[0]) | (u64::from(output[1]) << 32), u64::from(output[2]) | (u64::from(output[3]) << 32)]
        })
        .collect::<Vec<_>>();
    words.truncate(count);
    (words, counter.wrapping_add(invocation_count as u128))
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, DimensionBounds, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType,
        Sharding, StridedLayout,
    };
    use crate::batching::{BatchedProgram, BatchingTracer, ProgramBatchingOutputAxesPolicy, RecursiveBatchingPolicy};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, DifferentiationError,
        TransposableOperation, TranspositionContext,
    };
    use crate::macros::{check_operation_batching, check_operation_partial_evaluation, check_operation_type_inference};
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, MaybeZero};
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns a ThreeFry `[key, counter]` state.
    fn threefry_state(key: u64, counter: u64) -> Array {
        Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[key, counter]).unwrap()
    }

    /// Returns a Philox state holding `key` followed by the low and high halves of `counter`.
    fn philox_state(key: u64, counter: u128) -> Array {
        Array::from_elements(RandomAlgorithm::Philox.state_type(), &[key, counter as u64, (counter >> 64) as u64])
            .unwrap()
    }

    /// Returns the ThreeFry cipher words of `counter` under `key`, splitting both into their low and high `u32` halves.
    fn threefry_block(key: u64, counter: u64) -> [u32; 2] {
        threefry_2x32([key as u32, (key >> 32) as u32], [counter as u32, (counter >> 32) as u32])
    }

    /// Returns the Philox cipher words of `counter` under `key`, splitting both into their `u32` words.
    fn philox_block(key: u64, counter: u128) -> [u32; 4] {
        philox_4x32(
            [key as u32, (key >> 32) as u32],
            [counter as u32, (counter >> 32) as u32, (counter >> 64) as u32, (counter >> 96) as u32],
        )
    }

    #[test]
    fn test_random_algorithm_from_state_type() {
        assert_eq!(
            RandomAlgorithm::from_state_type(&ArrayType::new_static(DataType::U64, [2])),
            Ok(RandomAlgorithm::ThreeFry),
        );
        assert_eq!(
            RandomAlgorithm::from_state_type(&ArrayType::new_static(DataType::U64, [3])),
            Ok(RandomAlgorithm::Philox),
        );

        // Physical layouts do not change the algorithm, while other data types and shapes are rejected.
        assert_eq!(
            RandomAlgorithm::from_state_type(
                &ArrayType::new_static(DataType::U64, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-8]))),
            ),
            Ok(RandomAlgorithm::ThreeFry),
        );
        assert_eq!(
            RandomAlgorithm::from_state_type(&ArrayType::new_static(DataType::U32, [2])),
            Err(TypeError::invalid(
                "random generator states must have type `u64[2]` (i.e., for `three_fry`) or `u64[3]` (i.e., for \
                 `philox`) but got `u32[2]`",
            )),
        );
        assert_eq!(
            RandomAlgorithm::from_state_type(&ArrayType::new_static(DataType::U64, [2, 2])),
            Err(TypeError::invalid(
                "random generator states must have type `u64[2]` (i.e., for `three_fry`) or `u64[3]` (i.e., for \
                 `philox`) but got `u64[2, 2]`",
            )),
        );
    }

    #[test]
    fn test_random_algorithm_state_type() {
        assert_eq!(RandomAlgorithm::ThreeFry.state_type(), ArrayType::new_static(DataType::U64, [2]));
        assert_eq!(RandomAlgorithm::Philox.state_type(), ArrayType::new_static(DataType::U64, [3]));
    }

    #[test]
    fn test_random_algorithm_display() {
        assert_eq!(RandomAlgorithm::ThreeFry.to_string(), "three_fry");
        assert_eq!(RandomAlgorithm::Philox.to_string(), "philox");
    }

    #[test]
    fn test_rng_bit_generator() {
        let output_type = ArrayType::new_static(DataType::U32, [5]);
        let operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type.clone());
        assert_eq!(operation.name(), RNG_BIT_GENERATOR_OPERATION_NAME);
        assert_eq!(operation.algorithm(), RandomAlgorithm::ThreeFry);
        assert_eq!(operation.output_type(), &output_type);
        assert_eq!(operation.to_string(), "rng_bit_generator [algorithm=three_fry, output_type=u32[5]]");

        let length = DimensionVariable::new("length", DimensionBounds::unbounded());
        let output_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(length)]));
        let operation = RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::Philox, output_type.clone());
        assert_eq!(operation.name(), RNG_BIT_GENERATOR_OPERATION_NAME);
        assert_eq!(operation.algorithm(), RandomAlgorithm::Philox);
        assert_eq!(operation.output_type(), &output_type);
        assert_eq!(operation.to_string(), "rng_bit_generator [algorithm=philox, output_type=u64[length]]");
    }

    #[test]
    fn test_rng_bit_generator_type_inference() {
        // The state must have the state type of the algorithm.
        let output_type = ArrayType::new_static(DataType::U32, [4]);
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type.clone()),
            cases = [
                {
                    input_types = [RandomAlgorithm::ThreeFry.state_type()],
                    output_types = [RandomAlgorithm::ThreeFry.state_type(), output_type.clone()],
                },
                {
                    input_types = [ArrayType::new_static(DataType::U64, [3])],
                    error = "`rng_bit_generator` with the `three_fry` algorithm requires a `u64[2]` state but got \
                             `u64[3]`",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2])],
                    error = "`rng_bit_generator` with the `three_fry` algorithm requires a `u64[2]` state but got \
                             `f32[2]`",
                },
            ],
        );
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::Philox, output_type.clone()),
            cases = [
                {
                    input_types = [RandomAlgorithm::Philox.state_type()],
                    output_types = [RandomAlgorithm::Philox.state_type(), output_type],
                },
                {
                    input_types = [RandomAlgorithm::ThreeFry.state_type()],
                    error = "`rng_bit_generator` with the `philox` algorithm requires a `u64[3]` state but got \
                             `u64[2]`",
                },
            ],
        );

        // Every unsigned-integer output width is supported, while other output data types are rejected.
        for data_type in [DataType::U8, DataType::U16, DataType::U64] {
            let output_type = ArrayType::new_static(data_type, [4]);
            check_operation_type_inference!(
                operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type.clone()),
                cases = [{
                    input_types = [RandomAlgorithm::ThreeFry.state_type()],
                    output_types = [RandomAlgorithm::ThreeFry.state_type(), output_type],
                }],
            );
        }
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayType>::new(
                RandomAlgorithm::ThreeFry,
                ArrayType::new_static(DataType::I32, [4]),
            ),
            cases = [{
                input_types = [RandomAlgorithm::ThreeFry.state_type()],
                error = "`rng_bit_generator` does not support output data type `i32`",
            }],
        );

        // The homogeneous contract has no extent inputs, so its output must be statically shaped.
        let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(8)).unwrap());
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayType>::new(
                RandomAlgorithm::ThreeFry,
                ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(length)])),
            ),
            cases = [{
                input_types = [RandomAlgorithm::ThreeFry.state_type()],
                error = "`rng_bit_generator` does not support dynamically shaped outputs",
            }],
        );
    }

    #[test]
    fn test_rng_bit_generator_type_inference_sharding() {
        // Replicated states and outputs are supported, while sharded ones are rejected because every shard would draw
        // the same bits.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let output_type = ArrayType::new_static(DataType::U32, [4]);
        let replicated_output_type = output_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let sharded_state_type = state_type
            .clone()
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let sharded_output_type = output_type
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayType>::new(
                RandomAlgorithm::ThreeFry,
                replicated_output_type.clone(),
            ),
            cases = [
                {
                    input_types = [state_type.clone()],
                    output_types = [state_type.clone(), replicated_output_type],
                },
                {
                    input_types = [sharded_state_type],
                    error = "`rng_bit_generator` does not support sharded states or outputs; derive per-shard states \
                             inside `shard_map` instead",
                },
            ],
        );
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, sharded_output_type),
            cases = [{
                input_types = [state_type],
                error = "`rng_bit_generator` does not support sharded states or outputs; derive per-shard states \
                         inside `shard_map` instead",
            }],
        );
    }

    #[test]
    fn test_rng_bit_generator_type_inference_dynamic() {
        // The composite contract takes one extent input per dynamic output axis, in axis order, and each extent must
        // define the dimension variable of its axis.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(4)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(4)).unwrap());
        let output_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(rows.clone()), 3.into()]));
        let state_type = ArrayIrType::from(RandomAlgorithm::ThreeFry.state_type());
        let rows_type = ArrayIrType::from(DimensionType::from(rows));
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::ThreeFry, output_type.clone()),
            cases = [
                {
                    input_types = [state_type.clone(), rows_type.clone()],
                    output_types = [state_type.clone(), output_type.into()],
                },
                {
                    input_types = [state_type.clone()],
                    error = "expected 2 inputs but got 1",
                },
                {
                    input_types = [state_type.clone(), DimensionType::from(columns).into()],
                    error = "`rng_bit_generator` output-extent input defines dimension variable `columns`, but the \
                             corresponding declared output axis refers to `rows`",
                },
                {
                    input_types = [rows_type.clone(), rows_type],
                    error = "expected array type but got dimension type",
                },
            ],
        );

        // Static outputs take no extent inputs.
        let output_type = ArrayType::new_static(DataType::U32, [2, 3]);
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::ThreeFry, output_type.clone()),
            cases = [{ input_types = [state_type.clone()], output_types = [state_type, output_type.into()] }],
        );
    }

    #[test]
    fn test_rng_bit_generator_interpretation() {
        // Interpretation advances the counter by the number of cipher invocations and is deterministic in the state.
        let state = threefry_state(42, 7);
        let operation = RngBitGeneratorOperation::<ArrayType>::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new_static(DataType::U32, [5]),
        );
        let outputs = InterpretableOperation::<EagerContext<Array>>::interpret(
            &operation,
            &EagerContext::new(),
            &EmptyRegionDriver,
            std::slice::from_ref(&state),
        )
        .unwrap();
        let [first, second, third] = [7, 8, 9].map(|counter| threefry_block(42, counter));
        assert_eq!(outputs[0].elements::<u64>(), Ok(vec![42, 10]));
        assert_eq!(outputs[1].elements::<u32>(), Ok(vec![first[0], first[1], second[0], second[1], third[0]]));
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                std::slice::from_ref(&state),
            ),
            Ok(outputs.clone()),
        );

        // Narrower outputs keep the low bits of the same 32-bit words, and 64-bit outputs run one invocation per word.
        let (advanced_state, bits) = state
            .rng_bit_generator(RandomAlgorithm::ThreeFry, &ArrayType::new_static(DataType::U16, [5]))
            .unwrap();
        assert_eq!(advanced_state, outputs[0]);
        assert_eq!(
            bits.elements::<u16>(),
            Ok(outputs[1].elements::<u32>().unwrap().into_iter().map(|word| word as u16).collect()),
        );
        let (_, bits) = state
            .rng_bit_generator(RandomAlgorithm::ThreeFry, &ArrayType::new_static(DataType::U8, [5]))
            .unwrap();
        assert_eq!(
            bits.elements::<u8>(),
            Ok(outputs[1].elements::<u32>().unwrap().into_iter().map(|word| word as u8).collect()),
        );
        let (advanced_state, bits) = state
            .rng_bit_generator(RandomAlgorithm::ThreeFry, &ArrayType::new_static(DataType::U64, [3]))
            .unwrap();
        assert_eq!(advanced_state.elements::<u64>(), Ok(vec![42, 10]));
        assert_eq!(
            bits.elements::<u64>(),
            Ok([first, second, third].map(|words| u64::from(words[0]) | (u64::from(words[1]) << 32)).to_vec()),
        );

        // Philox states carry a 128-bit counter, and every invocation produces four 32-bit or two 64-bit words.
        let counter = 7u128 | (9u128 << 64);
        let state = philox_state(42, counter);
        let [first, second] = [counter, counter + 1].map(|counter| philox_block(42, counter));
        let (advanced_state, bits) = state
            .rng_bit_generator(RandomAlgorithm::Philox, &ArrayType::new_static(DataType::U32, [5]))
            .unwrap();
        assert_eq!(advanced_state, philox_state(42, counter + 2));
        assert_eq!(bits.elements::<u32>(), Ok(vec![first[0], first[1], first[2], first[3], second[0]]));
        let (advanced_state, bits) = state
            .rng_bit_generator(RandomAlgorithm::Philox, &ArrayType::new_static(DataType::U64, [3]))
            .unwrap();
        assert_eq!(advanced_state, philox_state(42, counter + 2));
        assert_eq!(
            bits.elements::<u64>(),
            Ok(vec![
                u64::from(first[0]) | (u64::from(first[1]) << 32),
                u64::from(first[2]) | (u64::from(first[3]) << 32),
                u64::from(second[0]) | (u64::from(second[1]) << 32),
            ]),
        );

        // Direct calls enforce the same contract as type inference.
        assert!(matches!(
            Array::vector(vec![42u64, 7, 9])
                .unwrap()
                .rng_bit_generator(RandomAlgorithm::ThreeFry, &ArrayType::new_static(DataType::U32, [5])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`rng_bit_generator` with the `three_fry` algorithm requires a `u64[2]` state but got \
                               `u64[3]`",
        ));
    }

    #[test]
    fn test_rng_bit_generator_interpretation_layouts() {
        // ThreeFry 32-bit words follow the shape-dependent layout of XLA. For `u32[2, 3]`, the split axis is axis 0,
        // so the first row holds the first cipher word of each invocation and the second row holds the second.
        let state = threefry_state(42, 7);
        let (advanced_state, bits) = state
            .rng_bit_generator(RandomAlgorithm::ThreeFry, &ArrayType::new_static(DataType::U32, [2, 3]))
            .unwrap();
        let [first, second, third] = [7, 8, 9].map(|counter| threefry_block(42, counter));
        assert_eq!(advanced_state.elements::<u64>(), Ok(vec![42, 10]));
        assert_eq!(bits.elements::<u32>(), Ok(vec![first[0], second[0], third[0], first[1], second[1], third[1]]));

        // Physical layouts of the state and the bits are storage contracts, while generation follows logical order.
        let state_type =
            RandomAlgorithm::ThreeFry.state_type().with_layout(Layout::Strided(StridedLayout::new(vec![-8])));
        let state = Array::from_elements(state_type.clone(), &[42u64, 7]).unwrap();
        let output_type =
            ArrayType::new_static(DataType::U16, [5]).with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        let (advanced_state, bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        let (expected_words, _) = threefry_u32_words(42, 7, &[5]);
        let expected_words = expected_words.into_iter().map(|word| word as u16).collect::<Vec<_>>();
        assert_eq!(advanced_state.r#type().as_ref(), &state_type);
        assert_eq!(advanced_state.elements::<u64>(), Ok(vec![42, 10]));
        assert_eq!(bits.r#type().as_ref(), &output_type);
        assert_eq!(bits.elements::<u16>(), Ok(expected_words.clone()));
        let mut expected_storage = vec![0; 18];
        for (index, word) in expected_words.into_iter().enumerate() {
            expected_storage[index * 4..index * 4 + 2].copy_from_slice(&word.to_le_bytes());
        }
        assert_eq!(bits.storage_bytes(), expected_storage);
    }

    #[test]
    fn test_rng_bit_generator_interpretation_dynamic() {
        // Concrete mixed values resolve each dynamic output axis from its extent input and then generate exactly the
        // bits of the resolved static shape.
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(4)).unwrap());
        let output_type = ArrayType::new(DataType::U32, Shape::new(vec![Dimension::Dynamic(rows.clone()), 3.into()]));
        let state = threefry_state(42, 7);
        let rows_value = ArrayIrValue::Dimension(DimensionValue::new(DimensionType::from(rows.clone()), 2).unwrap());
        let (expected_state, expected_bits) = state
            .rng_bit_generator(RandomAlgorithm::ThreeFry, &ArrayType::new_static(DataType::U32, [2, 3]))
            .unwrap();
        let operation = RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::ThreeFry, output_type.clone());
        assert_eq!(
            InterpretableOperation::<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>::interpret(
                &operation,
                &EagerContext::new(),
                &EmptyRegionDriver,
                &[ArrayIrValue::Array(state.clone()), rows_value.clone()],
            ),
            Ok(vec![ArrayIrValue::Array(expected_state), ArrayIrValue::Array(expected_bits)]),
        );

        // Context-carrying mixed values stage one composite bit-generation instruction.
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let state = context.input(RandomAlgorithm::ThreeFry.state_type().into());
        let rows_value = context.input(DimensionType::from(rows).into());
        let (advanced_state, bits) =
            state.dynamic_rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type, &[rows_value]).unwrap();
        assert_eq!(advanced_state.r#type().as_ref(), &ArrayIrType::from(RandomAlgorithm::ThreeFry.state_type()));
        assert_eq!(bits.r#type().as_ref(), &ArrayIrType::from(output_type));
        let builder = context.builder();
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), ArrayIrOperation::RngBitGenerator(_)));
    }

    #[test]
    fn test_rng_bit_generator_partial_evaluation() {
        let state = threefry_state(42, 7);
        let output_type = ArrayType::new_static(DataType::U32, [4]);
        let (advanced_state, bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type),
            cases = [
                {
                    inputs = [(@known, state.clone())],
                    outputs = [(@known, advanced_state.clone()), (@known, bits.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = RandomAlgorithm::ThreeFry.state_type(), replay = state))],
                    outputs = [(@residual, advanced_state), (@residual, bits)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_rng_bit_generator_batching() {
        // Each batch item of a mapped state draws exactly the bits that its own state draws unbatched, whichever axis
        // the states are stacked along, while a replicated state draws once and replicates both outputs.
        let output_type = ArrayType::new_static(DataType::U32, [5]);
        let (first_state, first_bits) =
            threefry_state(42, 7).rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        let (second_state, second_bits) =
            threefry_state(3, 11).rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        check_operation_batching!(
            @exact,
            context = EagerContext::<Array, ArrayOperation<Array>>::new(),
            driver = &EmptyRegionDriver,
            operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type.clone()),
            axis_size = 2,
            axis_sharding = ShardingDimension::Replicated,
            cases = [
                {
                    inputs = [(@mapped(axis = 1), Array::matrix(2, 2, vec![42u64, 3, 7, 11]).unwrap())],
                    outputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![42u64, 10, 3, 14]).unwrap()),
                        (@mapped(axis = 0), Array::matrix(
                            2,
                            5,
                            [first_bits.elements::<u32>().unwrap(), second_bits.elements::<u32>().unwrap()].concat(),
                        ).unwrap()),
                    ],
                },
                {
                    inputs = [(@replicated, threefry_state(42, 7))],
                    outputs = [(@replicated, first_state), (@replicated, first_bits)],
                },
            ],
        );
        assert_eq!(second_state, threefry_state(3, 14));

        // Philox states batch the same way.
        let (first_state, first_bits) =
            philox_state(42, 7).rng_bit_generator(RandomAlgorithm::Philox, &output_type).unwrap();
        let (second_state, second_bits) =
            philox_state(3, 11).rng_bit_generator(RandomAlgorithm::Philox, &output_type).unwrap();
        check_operation_batching!(
            @exact,
            context = EagerContext::<Array, ArrayOperation<Array>>::new(),
            driver = &EmptyRegionDriver,
            operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::Philox, output_type),
            axis_size = 2,
            axis_sharding = ShardingDimension::Replicated,
            cases = [{
                inputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![42u64, 7, 0, 3, 11, 0]).unwrap())],
                outputs = [
                    (@mapped(axis = 0), Array::matrix(
                        2,
                        3,
                        [first_state.elements::<u64>().unwrap(), second_state.elements::<u64>().unwrap()].concat(),
                    ).unwrap()),
                    (@mapped(axis = 0), Array::matrix(
                        2,
                        5,
                        [first_bits.elements::<u32>().unwrap(), second_bits.elements::<u32>().unwrap()].concat(),
                    ).unwrap()),
                ],
            }],
        );
    }

    #[test]
    fn test_rng_bit_generator_batching_staging() {
        let operation = RngBitGeneratorOperation::<ArrayType>::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new_static(DataType::U32, [4]),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(RandomAlgorithm::ThreeFry.state_type());
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder, Placeholder])
            .unwrap();

        // A mapped state stages one carry-free scan over the per-item states, so the batched program size is
        // independent of the batch size.
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
                        lambda %0:i64[], %1:u64[2] .
                        let %2:u64[2], %3:u32[4] = rng_bit_generator [algorithm=three_fry, output_type=u32[4]] %1
                        in (%2, %3)
                    },
                ]
                in (%1, %2)
            "}
            .trim_end(),
        );
        let outputs = batched.interpret(vec![Array::matrix(3, 2, vec![42u64, 7, 3, 11, 5, 0]).unwrap()]).unwrap();
        let mut expected_states = Vec::new();
        let mut expected_bits = Vec::new();
        for (key, counter) in [(42, 7), (3, 11), (5, 0)] {
            let (state, bits) = threefry_state(key, counter)
                .rng_bit_generator(RandomAlgorithm::ThreeFry, &ArrayType::new_static(DataType::U32, [4]))
                .unwrap();
            expected_states.extend(state.elements::<u64>().unwrap());
            expected_bits.extend(bits.elements::<u32>().unwrap());
        }
        assert_eq!(outputs[0].elements::<u64>(), Ok(expected_states));
        assert_eq!(outputs[1].elements::<u32>(), Ok(expected_bits));

        // A replicated state keeps the unbatched program.
        let (batched, output_axes) = program
            .batched(
                3,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::replicated(), BatchAxis::replicated()]);
        assert_eq!(batched.to_string(), program.to_string());
    }

    #[test]
    fn test_rng_bit_generator_batching_dynamic() -> Result<(), ProgramError> {
        type Context = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

        // A mapped state over a dynamic batch extent stages one composite scan whose replicated output extents are
        // invariant carries and whose dynamic length is a trailing runtime input.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(9))?);
        let rows = DimensionVariable::new("rows", DimensionBounds::new(1, Some(7))?);
        let columns = DimensionVariable::new("columns", DimensionBounds::new(1, Some(11))?);
        let output_type = ArrayType::new(
            DataType::U32,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns.clone())]),
        );
        let trace = Context::new();
        let batch_extent = trace.input(DimensionType::from(batch.clone()).into());
        let states = trace
            .input(ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(batch.clone()), 2.into()])).into());
        let row_extent = trace.input(DimensionType::from(rows.clone()).into());
        let column_extent = trace.input(DimensionType::from(columns.clone()).into());
        let input_ids = [batch_extent.clone(), states.clone(), row_extent.clone(), column_extent.clone()]
            .map(|input| input.atom_id().unwrap());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayIrOperation::RngBitGenerator(RngBitGeneratorOperation::new(
                RandomAlgorithm::ThreeFry,
                output_type.clone(),
            )),
            Vec::new(),
            &[
                BatchingTracer::new(context.clone(), ArrayIrBatch::new(states, BatchAxis::new(0))?),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(row_extent.clone())),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(column_extent.clone())),
            ],
        )?;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch().unbatched_type(), ArrayIrType::Array(output_type.clone()));
        let output_ids = outputs.iter().map(|output| output.batch().value().atom_id().unwrap()).collect::<Vec<_>>();
        let program = trace.builder().borrow().clone().build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
            output_ids,
            vec![Placeholder; 4],
            vec![Placeholder; 2],
        )?;
        let [scan] = program.entry_region().instructions() else {
            panic!("mapped random bit generation should stage exactly one scan instruction");
        };
        let ArrayIrOperation::Scan(scan_operation) = scan.operation() else {
            panic!("mapped random bit generation should stage the direct composite scan carrier");
        };
        assert_eq!(scan_operation.carry_count(), 2);
        assert_eq!(scan_operation.length(), &Dimension::Dynamic(batch));
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

        // A second batching transform structurally replays the already scan-decomposed program. The inner runtime scan
        // length remains an explicit replicated dimension input while the new mapped extent becomes its leading carry.
        let nested_trace = Context::new();
        let outer = DimensionVariable::new("outer", DimensionBounds::new(1, Some(5))?);
        let outer_extent = nested_trace.input(DimensionType::from(outer.clone()).into());
        let nested_context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(nested_trace, outer_extent);
        let nested = <ArrayIrBatchingPolicy as RecursiveBatchingPolicy<Context>>::batch_program(
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
        assert_eq!(nested.input_types()[0], ArrayIrType::Dimension(DimensionType::from(outer)));

        // A replicated state stages the composite operation once and replicates both outputs.
        let trace = Context::new();
        let batch_extent = trace.input(DimensionType::new("batch", DimensionBounds::new(1, Some(9))?).into());
        let state = trace.input(RandomAlgorithm::ThreeFry.state_type().into());
        let row_extent = trace.input(DimensionType::from(rows).into());
        let column_extent = trace.input(DimensionType::from(columns).into());
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(trace.clone(), batch_extent);
        let outputs = context.bind(
            ArrayIrOperation::RngBitGenerator(RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, output_type)),
            Vec::new(),
            &[
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(state)),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(row_extent)),
                BatchingTracer::new(context.clone(), ArrayIrBatch::replicated(column_extent)),
            ],
        )?;
        assert_eq!(outputs[0].batch().batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[1].batch().batch_axis(), BatchAxis::replicated());
        let builder = trace.builder();
        let builder = builder.borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), ArrayIrOperation::RngBitGenerator(_)));

        Ok(())
    }

    #[test]
    fn test_rng_bit_generator_differentiation() {
        // The integer state and bits have structural-zero tangents, so the tangent side stages no bit generation.
        let state = threefry_state(42, 7);
        let output_type = ArrayType::new_static(DataType::U32, [4]);
        let operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type.clone());
        let outputs = operation
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(state.clone()).unwrap()],
            )
            .unwrap();
        let (advanced_state, bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].primal(), &advanced_state);
        assert_eq!(outputs[1].primal(), &bits);
        assert!(outputs.iter().all(|output| output.tangent().is_zero()));

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(RandomAlgorithm::ThreeFry.state_type());
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder, Placeholder])
            .unwrap();
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

    #[test]
    fn test_rng_bit_generator_transposition() {
        // Program transposition elides the zero-space cotangents of integer outputs, so check the rejection of the
        // operation directly.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            RngBitGeneratorOperation::<ArrayType>::new(
                RandomAlgorithm::ThreeFry,
                ArrayType::new_static(DataType::U32, [4]),
            )
            .transpose(
                &mut TranspositionContext::new(context),
                &EmptyRegionDriver,
                &[PartialValue::Unknown(RandomAlgorithm::ThreeFry.state_type())],
                &[
                    MaybeZero::Zero(ArrayType::new_static(DataType::Zero, [2])),
                    MaybeZero::Zero(ArrayType::new_static(DataType::Zero, [4])),
                ],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "operation `rng_bit_generator` is not transposable",
        ));
    }

    #[test]
    fn test_split_rng_key() {
        // Splitting draws one `u64` key per fresh state, and each fresh state pairs its key with a zero counter.
        let (advanced_state, fresh_states) = threefry_state(42, 7).split_rng_key(3).unwrap();
        let (keys, counter) = threefry_u64_words(42, 7, 3);
        assert_eq!(advanced_state, threefry_state(42, counter));
        assert_eq!(fresh_states, keys.iter().map(|key| threefry_state(*key, 0)).collect::<Vec<_>>());
        assert_ne!(fresh_states[0], fresh_states[1]);
        assert_ne!(fresh_states[1], fresh_states[2]);

        // Philox states split into Philox states.
        let counter = u128::from(u64::MAX);
        let (advanced_state, fresh_states) = philox_state(5, counter).split_rng_key(2).unwrap();
        let (keys, counter) = philox_u64_words(5, counter, 2);
        assert_eq!(advanced_state, philox_state(5, counter));
        assert_eq!(fresh_states, keys.iter().map(|key| philox_state(*key, 0)).collect::<Vec<_>>());

        // Zero splits leave the state unchanged, and values that are not states are rejected.
        assert_eq!(threefry_state(42, 7).split_rng_key(0), Ok((threefry_state(42, 7), Vec::new())));
        assert!(matches!(
            Array::vector(vec![42u32, 7]).unwrap().split_rng_key(2),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "random generator states must have type `u64[2]` (i.e., for `three_fry`) or \
                               `u64[3]` (i.e., for `philox`) but got `u32[2]`",
        ));
    }

    #[test]
    fn test_random_uniform() {
        // `f32` samples keep the top 24 bits of each 32-bit word, and `f64` samples keep the top 53 bits of each 64-bit
        // word, so both are exact multiples of their spacing.
        let state = threefry_state(42, 7);
        let (advanced_state, samples) = state.random_uniform(&ArrayType::new_static(DataType::F32, [2, 3])).unwrap();
        let (words, counter) = threefry_u32_words(42, 7, &[2, 3]);
        assert_eq!(advanced_state, threefry_state(42, counter));
        assert_eq!(samples.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [2, 3]));
        assert_eq!(
            samples.elements::<f32>(),
            Ok(words.into_iter().map(|word| (word >> 8) as f32 * 2.0f32.powi(-24)).collect()),
        );
        let (advanced_state, samples) = state.random_uniform(&ArrayType::new_static(DataType::F64, [2, 3])).unwrap();
        let (words, counter) = threefry_u64_words(42, 7, 6);
        assert_eq!(advanced_state, threefry_state(42, counter));
        assert_eq!(
            samples.elements::<f64>(),
            Ok(words.into_iter().map(|word| (word >> 11) as f64 * 2.0f64.powi(-53)).collect()),
        );

        // Philox states draw with Philox, and the same state always draws the same samples.
        let (_, samples) = philox_state(42, 7).random_uniform(&ArrayType::new_static(DataType::F32, [5])).unwrap();
        let (words, _) = philox_u32_words(42, 7, 5);
        assert_eq!(
            samples.elements::<f32>(),
            Ok(words.into_iter().map(|word| (word >> 8) as f32 * 2.0f32.powi(-24)).collect()),
        );
        assert_eq!(philox_state(42, 7).random_uniform(&ArrayType::new_static(DataType::F32, [5])).unwrap().1, samples);

        // Many samples stay in `[0, 1)` with the moments of the uniform distribution.
        let (_, samples) = state.random_uniform(&ArrayType::new_static(DataType::F32, [4096])).unwrap();
        let values = samples.to_f64s();
        assert!(values.iter().all(|value| (0.0..1.0).contains(value)));
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / values.len() as f64;
        assert!((mean - 0.5).abs() < 0.02);
        assert!((variance - 1.0 / 12.0).abs() < 0.01);

        // The samples keep the memory space of the requested type, while its physical layout is a storage request that
        // the elementwise sampling arithmetic does not preserve.
        let sample_type = ArrayType::new_static(DataType::F32, [2, 3]).with_memory(Memory::Host { pinned: true });
        let (_, samples) = state.random_uniform(&sample_type).unwrap();
        assert_eq!(samples.r#type().as_ref(), &sample_type);
        let sample_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let (_, samples) = state
            .random_uniform(&sample_type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![4, 8]))))
            .unwrap();
        assert_eq!(samples.r#type().as_ref(), &sample_type);

        assert!(matches!(
            state.random_uniform(&ArrayType::new_static(DataType::F16, [2])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`random_uniform` does not support output data type `f16`",
        ));
    }

    #[test]
    fn test_random_normal() {
        // Samples follow the Box–Muller transform of two consecutive uniform draws.
        let state = threefry_state(42, 7);
        let (advanced_state, samples) = state.random_normal(&ArrayType::new_static(DataType::F64, [3])).unwrap();
        let (state_after_first, first) = state.random_uniform(&ArrayType::new_static(DataType::F64, [3])).unwrap();
        let (expected_state, second) =
            state_after_first.random_uniform(&ArrayType::new_static(DataType::F64, [3])).unwrap();
        assert_eq!(advanced_state, expected_state);
        for ((sample, first), second) in samples.to_f64s().into_iter().zip(first.to_f64s()).zip(second.to_f64s()) {
            let expected = (-2.0 * (1.0 - first).ln()).sqrt() * (std::f64::consts::TAU * second).cos();
            assert!((sample - expected).abs() < 1e-12);
        }

        // Many samples are finite with the moments of the standard normal distribution.
        let (_, samples) = state.random_normal(&ArrayType::new_static(DataType::F32, [4096])).unwrap();
        assert_eq!(samples.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [4096]));
        let values = samples.to_f64s();
        assert!(values.iter().all(|value| value.is_finite()));
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>() / values.len() as f64;
        assert!(mean.abs() < 0.05);
        assert!((variance - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_random_categorical() {
        // With a logit gap of 10, a non-peak draw has probability about 9e-5 per sample, so all 64 samples must pick
        // the peak category, and a `-∞` logit is never sampled.
        let logits = Array::vector(vec![0.0, 10.0, f64::NEG_INFINITY]).unwrap();
        let mut state = threefry_state(42, 0);
        for _ in 0..64 {
            let (advanced_state, sample) = state.random_categorical(&logits, 0).unwrap();
            assert_eq!(sample, Array::scalar(1i32).unwrap());
            state = advanced_state;
        }

        // The category axis may be negative and is removed from the sample shape.
        let logits = Array::matrix(2, 3, vec![20.0f32, 0.0, 0.0, 0.0, 0.0, 20.0]).unwrap();
        let (_, samples) = state.random_categorical(&logits, -1).unwrap();
        assert_eq!(samples, Array::vector(vec![0i32, 2]).unwrap());
        let (_, samples) = state.random_categorical(&logits, 0).unwrap();
        assert_eq!(samples, Array::vector(vec![0i32, 0, 1]).unwrap());

        assert!(matches!(
            state.random_categorical(&Array::vector(vec![0i32, 1]).unwrap(), 0),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`random_categorical` does not support logits data type `i32`",
        ));
    }

    #[test]
    fn test_threefry2x32() {
        // These are the Random123 known-answer vectors for ThreeFry-2x32 with 20 rounds.
        assert_eq!(threefry_2x32([0, 0], [0, 0]), [0x6b200159, 0x99ba4efe]);
        assert_eq!(threefry_2x32([0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff]), [0x1cb996fc, 0xbb002be7]);
        assert_eq!(threefry_2x32([0x13198a2e, 0x03707344], [0x243f6a88, 0x85a308d3]), [0xc4923a9c, 0x483df7a0]);
    }

    #[test]
    fn test_threefry_u32_words() {
        // A key with a non-zero high half and a counter that crosses the 32-bit boundary check that both are split
        // into their low and high halves, low half first.
        let key = (1u64 << 32) | 42;
        let counter = u64::from(u32::MAX);
        let block = |offset: u64| threefry_block(key, counter + offset);

        // A scalar keeps the first word of one invocation, and vectors pair adjacent words.
        assert_eq!(threefry_u32_words(key, counter, &[]), (vec![block(0)[0]], counter + 1));
        assert_eq!(
            threefry_u32_words(key, counter, &[5]),
            (vec![block(0)[0], block(0)[1], block(1)[0], block(1)[1], block(2)[0]], counter + 3),
        );

        // Higher-rank outputs split their first even-sized axis. Splitting the last axis pairs adjacent words in
        // row-major order, while splitting an earlier axis places the two words of an invocation in different rows.
        assert_eq!(
            threefry_u32_words(key, counter, &[3, 4]),
            ((0..12).map(|index| block(index as u64 / 2)[index % 2]).collect::<Vec<_>>(), counter + 6,),
        );
        assert_eq!(
            threefry_u32_words(key, counter, &[2, 3]),
            (vec![block(0)[0], block(1)[0], block(2)[0], block(0)[1], block(1)[1], block(2)[1]], counter + 3),
        );

        // Without an even-sized axis, the first largest axis is split and its last invocation word is dropped.
        assert_eq!(
            threefry_u32_words(key, counter, &[3, 5]),
            (
                (0..3)
                    .flat_map(|row| (0..5).map(move |column| block(3 * row + column / 2)[column as usize % 2]))
                    .collect::<Vec<_>>(),
                counter + 9,
            ),
        );
        assert_eq!(
            threefry_u32_words(key, counter, &[3, 3]),
            (
                (0..3)
                    .flat_map(|row| (0..3).map(move |column| block(3 * (row / 2) + column)[row as usize % 2]))
                    .collect::<Vec<_>>(),
                counter + 6,
            ),
        );

        // Empty outputs run no invocations.
        assert_eq!(threefry_u32_words(key, counter, &[0, 3]), (Vec::new(), counter));
    }

    #[test]
    fn test_threefry_u64_words() {
        // Each word combines the two cipher words of one counter, low word first.
        let key = (1u64 << 32) | 42;
        let word = |counter: u64| {
            let output = threefry_block(key, counter);
            u64::from(output[0]) | (u64::from(output[1]) << 32)
        };
        assert_eq!(threefry_u64_words(key, 7, 3), (vec![word(7), word(8), word(9)], 10));
        assert_eq!(threefry_u64_words(key, u64::MAX, 2), (vec![word(u64::MAX), word(0)], 1));
        assert_eq!(threefry_u64_words(key, 7, 0), (Vec::new(), 7));
    }

    #[test]
    fn test_philox4x32() {
        // These are the Random123 known-answer vectors for Philox-4x32 with 10 rounds.
        assert_eq!(philox_4x32([0, 0], [0, 0, 0, 0]), [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
        assert_eq!(
            philox_4x32([0xffffffff, 0xffffffff], [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]),
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd],
        );
        assert_eq!(
            philox_4x32([0xa4093822, 0x299f31d0], [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344]),
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1],
        );
    }

    #[test]
    fn test_philox_u32_words() {
        // Each invocation contributes four adjacent words, the last invocation is truncated, and a counter just below
        // the 64-bit boundary carries into the high half of the 128-bit counter.
        let key = (1u64 << 32) | 42;
        let counter = u128::from(u64::MAX - 1);
        let [first, second, third] = [0, 1, 2].map(|offset| philox_block(key, counter + offset));
        let (words, advanced_counter) = philox_u32_words(key, counter, 9);
        assert_eq!(words, [first.to_vec(), second.to_vec(), vec![third[0]]].concat());
        assert_eq!(advanced_counter, counter + 3);
        assert_eq!(advanced_counter >> 64, 1);
        assert_eq!(philox_u32_words(key, counter, 0), (Vec::new(), counter));
    }

    #[test]
    fn test_philox_u64_words() {
        // Each invocation contributes two adjacent words that combine its cipher words pairwise, low word first, and
        // the last invocation is truncated.
        let key = (1u64 << 32) | 42;
        let counter = u128::from(u64::MAX);
        let [first, second] = [0, 1].map(|offset| philox_block(key, counter + offset));
        assert_eq!(
            philox_u64_words(key, counter, 3),
            (
                vec![
                    u64::from(first[0]) | (u64::from(first[1]) << 32),
                    u64::from(first[2]) | (u64::from(first[3]) << 32),
                    u64::from(second[0]) | (u64::from(second[1]) << 32),
                ],
                counter + 2,
            ),
        );
        assert_eq!(philox_u64_words(key, counter, 0), (Vec::new(), counter));
    }
}
