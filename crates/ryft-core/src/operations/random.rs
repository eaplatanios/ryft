use std::fmt::Display;

use crate::backends::scalars::Scalar;
use crate::batching::{ArrayBatch, BatchableOperation, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_non_transposable_operation};
use crate::operations::constants::{Fill, ZeroLike};
use crate::operations::control_flow::ScanOperation;
use crate::operations::manipulation::{Concatenate, ConvertElementType, Slice, Transpose};
use crate::operations::math::{Add, Cos, Div, Log, Mul, Neg, Sqrt, Sub};
use crate::operations::sort::ArgMax;
use crate::parameters::Placeholder;
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::builders::ProgramBuilder;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::Value;
use crate::programs::{ProgramError, TypeIdentityRenaming};
use crate::types::{ArrayType, DataType, Dimension, Shape};

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
/// state — the analogue of
/// [StableHLO's `rng_bit_generator`](https://openxla.org/stablehlo/spec#rng_bit_generator) and the primitive
/// underneath [JAX's `jax.random.bits`](https://docs.jax.dev/en/latest/_autosummary/jax.random.bits.html). The
/// single input is the generator state (see [`RandomAlgorithm::state_type`]), and the two outputs are the advanced
/// state followed by the generated bits at the declared output type. Randomness is functional: the same state
/// always produces the same bits, and drawing again requires threading the advanced state (or deriving fresh states
/// with [`split_key`]) — there is no hidden generator state.
///
/// The output element type must be an unsigned-integer type (`ui8`, `ui16`, `ui32`, or `ui64`); distributions over
/// floating-point values are compositions on top of the raw bits (see [`uniform`](RngBitGenerator::uniform)).
/// The declared output shape must be static and must not be sharded (each shard would otherwise see the same bits;
/// derive per-shard states inside `shard_map` instead).
///
/// Both outputs are discrete, so differentiation assigns structural-zero tangents and transposition is rejected.
/// Batching a *mapped* state (one state per batch item, e.g. derived with [`split_key`]) stages one carry-free
/// [`ScanOperation`] over the per-item states, so each batch item draws its own bits from its own state; batching a
/// *replicated* state is rejected because every batch item would see the same state and draw identical bits. The
/// reference array backend implements both [`ThreeFry`](RandomAlgorithm::ThreeFry) and
/// [`Philox`](RandomAlgorithm::Philox) bit-exactly with XLA's implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct RngBitGeneratorOperation {
    /// Algorithm generating the bits.
    algorithm: RandomAlgorithm,

    /// Declared type of the generated-bits output.
    output_type: ArrayType,
}

impl RngBitGeneratorOperation {
    /// Creates a new [`RngBitGeneratorOperation`] with the provided algorithm and declared bits output type.
    #[inline]
    pub fn new(algorithm: RandomAlgorithm, output_type: ArrayType) -> Self {
        Self { algorithm, output_type }
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
}

impl Display for RngBitGeneratorOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl Operation<ArrayType> for RngBitGeneratorOperation {
    #[inline]
    fn name(&self) -> &'static str {
        RNG_BIT_GENERATOR_OPERATION_NAME
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, RNG_BIT_GENERATOR_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("algorithm", &self.algorithm)?;
            operation.field("output_type", &self.output_type)
        })
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        let state_type = self.algorithm.state_type();
        if input_types[0].data_type() != state_type.data_type() || input_types[0].shape() != state_type.shape() {
            return Err(TypeError::invalid(format!(
                "'rng_bit_generator' with the {} algorithm needs a {} state but got {}",
                self.algorithm, state_type, input_types[0],
            )));
        }
        if !matches!(self.output_type.data_type(), DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64,) {
            return Err(TypeError::invalid(format!(
                "'rng_bit_generator' does not support output data type {}",
                self.output_type.data_type(),
            )));
        }
        if self.output_type.static_shape().is_none() {
            return Err(TypeError::invalid(
                "'rng_bit_generator' does not support dynamically shaped outputs".to_string(),
            ));
        }
        let has_sharded_dimension = |array_type: &ArrayType| {
            array_type.sharding().is_some_and(|sharding| {
                sharding
                    .dimensions()
                    .iter()
                    .any(|dimension| matches!(dimension, crate::sharding::ShardingDimension::Sharded(_)))
            })
        };
        if has_sharded_dimension(&input_types[0]) || has_sharded_dimension(&self.output_type) {
            return Err(TypeError::invalid(
                "'rng_bit_generator' does not support sharded states or outputs; derive per-shard states \
                          inside shard_map instead"
                    .to_string(),
            ));
        }
        Ok(vec![input_types[0].clone(), self.output_type.clone()])
    }

    fn rename_identities(
        &self,
        renaming: &TypeIdentityRenaming<<ArrayType as crate::Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Ok(Self { algorithm: self.algorithm, output_type: self.output_type.rename_identities(renaming)? })
    }
}

impl<C: Domain<Type = ArrayType, Value: RngBitGenerator>> InterpretableOperation<C> for RngBitGeneratorOperation {
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

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for RngBitGeneratorOperation where
    C::Operation: From<RngBitGeneratorOperation>
{
}

impl_non_differentiable_operation!(RngBitGeneratorOperation);
impl_non_transposable_operation!(RngBitGeneratorOperation);

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
impl<C: Context<Type = ArrayType>> BatchableOperation<C> for RngBitGeneratorOperation
where
    C::Value: Transpose,
    C::Operation: From<RngBitGeneratorOperation> + From<ScanOperation<C::Constant>>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
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
        let scan = ScanOperation::<C::Constant>::new(0, context.axis_size());
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
/// The `From<RngBitGeneratorOperation>` bound makes this disjoint from the eager reference value types (whose
/// context operation is [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers the
/// transform tracers and backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> RngBitGenerator for V
where
    V::DispatchDomain: Context<Operation: From<RngBitGeneratorOperation>>,
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
    V::DispatchDomain: Fill<Scalar, V>,
{
    let scalar = match array_type.data_type() {
        DataType::F32 => Scalar::from(value as f32),
        _ => Scalar::from(value),
    };
    domain.fill(array_type, scalar)
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
    V::DispatchDomain: Fill<Scalar, V>,
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
                let shifted = bits.div(&domain.fill(&bits_type, Scalar::from(1u32 << 8))?)?;
                let scale = domain.fill(&sample_type, Scalar::from(2.0f32.powi(-24)))?;
                shifted.convert_element_type(DataType::F32)?.mul(&scale)?
            }
            DataType::F64 => {
                // All 32 bits are exactly representable in `f64` and `u32::MAX · 2⁻³² < 1` exactly.
                let scale = domain.fill(&sample_type, Scalar::from(2.0f64.powi(-32)))?;
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

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::EagerContext;
    use crate::macros::check_operation_type_inference;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;
    use crate::sharding::ShardingDimension;

    use super::*;

    /// Returns a ThreeFry `[key, counter]` state array for the reference backend.
    fn state(key: u64, counter: u64) -> Array {
        Array::new(RandomAlgorithm::ThreeFry.state_type(), vec![Scalar::U64(key), Scalar::U64(counter)]).unwrap()
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
        assert_eq!(outputs[0].values(), &[Scalar::U64(42), Scalar::U64(expected_counter)]);
        assert_eq!(expected_counter, 10);
        assert_eq!(outputs[1].values(), expected_words.into_iter().map(Scalar::U32).collect::<Vec<_>>());
        let replayed = InterpretableOperation::<EagerContext<Array>>::interpret(
            &operation,
            &EagerContext::new(),
            &EmptyRegionDriver,
            std::slice::from_ref(&state),
        )
        .unwrap();
        assert_eq!(replayed, outputs);

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
        let operation = RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, bits_type(4));
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
            operation = RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, output_type),
            cases = [{
                input_types = [RandomAlgorithm::ThreeFry.state_type()],
                error = "'rng_bit_generator' does not support output data type f32",
            }],
        );
        for data_type in [DataType::U8, DataType::U16, DataType::U64] {
            let output_type = ArrayType::new(data_type, Shape::new(vec![Dimension::Static(4)]));
            check_operation_type_inference!(
                operation = RngBitGeneratorOperation::new(RandomAlgorithm::ThreeFry, output_type.clone()),
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
                lambda %0:u64[2], %1:zero[2] .
                let %2:u64[2], %3:u32[4] = rng_bit_generator [algorithm=three_fry, output_type=u32[4]] %0
                    %4:zero[2] = zero [type=zero[2]]
                    %5:zero[4] = zero [type=zero[4]]
                in (%2, %3, %4, %5)
            "}
            .trim_end(),
        );
    }

    /// Returns an active batching frame over an eager reference-backend parent for direct batching-rule tests.
    fn batching_context(axis_size: usize) -> BatchingContext<EagerContext<Array, ArrayOperation<Array>>> {
        BatchingContext::new(EagerContext::new(), axis_size)
    }

    #[test]
    fn test_rng_bit_generator_batching_threefry() {
        // Two distinct ThreeFry states stacked at batch axis 1 (`u64[2, b]` holding `[[k0, k1], [c0, c1]]`) exercise
        // the realignment to batch axis 0 before the scan consumes one state row per iteration.
        let states = Array::new(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            vec![Scalar::U64(42), Scalar::U64(3), Scalar::U64(7), Scalar::U64(11)],
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
        let expected_states = [first_state.values(), second_state.values()].concat();
        let expected_bits = [first_bits.values(), second_bits.values()].concat();
        assert_eq!(outputs[0].value().values(), expected_states);
        assert_eq!(outputs[1].value().values(), expected_bits);
    }

    #[test]
    fn test_rng_bit_generator_batching_philox() {
        // Two distinct Philox `u64[3]` states stacked at batch axis 0 (`u64[2, 3]`).
        let states = Array::new(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
            vec![Scalar::U64(42), Scalar::U64(7), Scalar::U64(9), Scalar::U64(3), Scalar::U64(11), Scalar::U64(0)],
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
            Array::new(
                RandomAlgorithm::Philox.state_type(),
                vec![Scalar::U64(key), Scalar::U64(counter_low), Scalar::U64(counter_high)],
            )
            .unwrap()
        };
        let (first_state, first_bits) =
            philox_state(42, 7, 9).rng_bit_generator(RandomAlgorithm::Philox, &bits_type(5)).unwrap();
        let (second_state, second_bits) =
            philox_state(3, 11, 0).rng_bit_generator(RandomAlgorithm::Philox, &bits_type(5)).unwrap();
        let expected_states = [first_state.values(), second_state.values()].concat();
        let expected_bits = [first_bits.values(), second_bits.values()].concat();
        assert_eq!(outputs[0].value().values(), expected_states);
        assert_eq!(outputs[1].value().values(), expected_bits);
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
            .unwrap();
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
        let states = Array::new(
            ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
            vec![Scalar::U64(42), Scalar::U64(7), Scalar::U64(3), Scalar::U64(11), Scalar::U64(5), Scalar::U64(0)],
        )
        .unwrap();
        let outputs = batched.interpret(vec![states]).unwrap();
        let mut expected_states = Vec::new();
        let mut expected_bits = Vec::new();
        for (key, counter) in [(42, 7), (3, 11), (5, 0)] {
            let (advanced, bits) =
                state(key, counter).rng_bit_generator(RandomAlgorithm::ThreeFry, &bits_type(4)).unwrap();
            expected_states.extend_from_slice(advanced.values());
            expected_bits.extend_from_slice(bits.values());
        }
        assert_eq!(outputs[0].values(), expected_states);
        assert_eq!(outputs[1].values(), expected_bits);
    }

    #[test]
    fn test_philox_rng_bit_generator() {
        // The reference backend maps the `ui64[3]` state to `[key, counter]` with the 128-bit counter split into
        // its low and high `u64` halves.
        let state =
            Array::new(RandomAlgorithm::Philox.state_type(), vec![Scalar::U64(42), Scalar::U64(7), Scalar::U64(9)])
                .unwrap();
        let counter = 7u128 | (9u128 << 64);

        // Five `u32` words run two cipher invocations, and the counter advances by that invocation count.
        let (advanced, bits) = state.rng_bit_generator(RandomAlgorithm::Philox, &bits_type(5)).unwrap();
        let (expected_words, expected_counter) = philox_u32_words(42, counter, 5);
        assert_eq!(expected_counter, counter + 2);
        assert_eq!(
            advanced.values(),
            &[Scalar::U64(42), Scalar::U64(expected_counter as u64), Scalar::U64((expected_counter >> 64) as u64)],
        );
        assert_eq!(bits.values(), expected_words.into_iter().map(Scalar::U32).collect::<Vec<_>>());

        // Three `u64` words also run two cipher invocations (two words per invocation, truncated).
        let u64_bits_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3)]));
        let (advanced, bits) = state.rng_bit_generator(RandomAlgorithm::Philox, &u64_bits_type).unwrap();
        let (expected_words, expected_counter) = philox_u64_words(42, counter, 3);
        assert_eq!(expected_counter, counter + 2);
        assert_eq!(
            advanced.values(),
            &[Scalar::U64(42), Scalar::U64(expected_counter as u64), Scalar::U64((expected_counter >> 64) as u64)],
        );
        assert_eq!(bits.values(), expected_words.into_iter().map(Scalar::U64).collect::<Vec<_>>());
    }

    #[test]
    fn test_split_key() {
        let parent = state(42, 7);
        let (advanced, fresh_states) = parent.split_key(3).unwrap();

        // Splitting draws three `u64` keys, so the parent state advances by three cipher invocations.
        let (expected_keys, expected_counter) = threefry_u64_words(42, 7, 3);
        assert_eq!(advanced.values(), &[Scalar::U64(42), Scalar::U64(expected_counter)]);
        assert_ne!(advanced.values(), parent.values());

        // Each fresh state is a `u64[2]` seeded by one generated key with a zero counter, and the keys are distinct.
        assert_eq!(fresh_states.len(), 3);
        for (fresh_state, expected_key) in fresh_states.iter().zip(expected_keys) {
            assert_eq!(fresh_state.r#type().into_owned(), RandomAlgorithm::ThreeFry.state_type());
            assert_eq!(fresh_state.values(), &[Scalar::U64(expected_key), Scalar::U64(0)]);
        }
        assert_ne!(fresh_states[0].values(), fresh_states[1].values());
        assert_ne!(fresh_states[0].values(), fresh_states[2].values());
        assert_ne!(fresh_states[1].values(), fresh_states[2].values());
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
            assert_eq!(sample.values(), &[Scalar::I32(1)]);
            state = advanced;
        }
    }
}
