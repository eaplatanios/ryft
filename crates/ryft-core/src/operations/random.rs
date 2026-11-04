//! Counter-based pseudorandom number generation and dropout. Randomness in Ryft is _functional_. Specifically, a
//! generator state is an ordinary `u64` array, the same state always produces the same bits, and every random draw
//! returns an advanced state that the caller threads into the next draw. There is no hidden generator state, so random
//! programs remain deterministic and every transform treats them like any other pure computation. This module provides:
//!
//!   - [`RandomAlgorithm`] selects a counter-based bit generator (e.g., ThreeFry-2x32 or Philox-4x32) and defines the
//!     type of its state.
//!   - [`RngBitGenerator`] generates raw unsigned-integer bits from a state by applying an [`RngBitGeneratorOperation`]
//!     which is analogous to StableHLO's [`rng_bit_generator`](https://openxla.org/stablehlo/spec#rng_bit_generator).
//!     [`DynamicRngBitGenerator`] does the same for dynamically shaped outputs in the mixed [`ArrayIrValue`] family.
//!   - [`Random`] composes those bits with ordinary array operations into key splitting and the uniform, normal, and
//!     categorical distributions, which therefore inherit their transform rules from the operations they use.
//!   - [`Dropout`] randomly zeroes input elements and rescales the remaining elements to preserve their expected
//!     values. Its mask can be shared along selected input axes.
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
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use ryft_macros::capability;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch,
    ArrayIrBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, ArrayTypeRefinements, DataType, Dimension,
    DimensionRefinements, DimensionType, DimensionValue, DimensionVariable, Shape, Sharding, ShardingDimension,
};
use crate::axes::{Axes, Axis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{
    check_count, check_types, impl_differentiable_operation, impl_non_differentiable_operation,
    impl_non_transposable_operation, impl_reference_dischargeable_operation,
};
use crate::operations::Capability;
use crate::operations::arithmetic::{Add, Div, Mul, Neg, Sqrt, Sub};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::comparisons::Compare;
use crate::operations::complex::Complex;
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::zero_like::ZeroLike;
use crate::operations::control_flow::scan::ScanOperation;
use crate::operations::control_flow::select::Select;
use crate::operations::dimensions::dimension_size::DimensionSizeOperation;
use crate::operations::exponential::{Ln1p, Log};
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcastOperation};
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::slicing::Slice;
use crate::operations::manipulation::transposition::{Transpose, TransposeOperation};
use crate::operations::reductions::ArgMax;
use crate::operations::trigonometric::Cos;
use crate::parameters::Placeholder;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, ProgramBuilder, ProgramError, RegionInterface, Type,
    TypeError, TypeIdentityRenaming, TypeRefinements, Typed, Value, ValueDomainDispatch, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

/// Deterministic counter-based pseudorandom bit-generation algorithm used by an [`RngBitGeneratorOperation`].
/// Both algorithms come from [Salmon et al. paper](https://doi.org/10.1145/2063384.2063405), and their states hold
/// a `u64` key followed by a counter that every draw advances by the number of cipher invocations it performs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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
/// generator state, and the two outputs are the advanced state (at the state type) followed by the generated bits at
/// the declared [`output_type`](Self::output_type). Inside a manual region (e.g., `shard_map`), a state that varies
/// over manual mesh axes draws different bits on different devices, so the bits additionally vary over the varying
/// manual axes of the state. A declared output type that cannot vary over those axes (e.g., because it is reduced
/// over one of them) is rejected.
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
/// batch size. The array IR form threads its output extents through that scan as invariant carries, which requires
/// them to be replicated.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

    /// Returns the declared type of the generated bits for this [`RngBitGeneratorOperation`]. The inferred bits type
    /// additionally varies over the varying manual axes of the state.
    #[inline]
    pub fn output_type(&self) -> &ArrayType {
        &self.output_type
    }

    /// Validates the state type, output element type, and sharding constraints shared by both input contracts, and
    /// returns the type of the generated bits. States and outputs must be unsharded and carry no unreduced mesh axes
    /// because generation is nonlinear. Any statically known output element count must fit in [`usize`]. The bits
    /// vary over the union of the declared varying manual axes and those of the state, because states that differ
    /// across devices draw different bits.
    fn infer_bits_type(&self, state_type: &ArrayType) -> Result<ArrayType, TypeError> {
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

        check_types!(@no_unreduced, RNG_BIT_GENERATOR_OPERATION_NAME, &[state_type]);
        if !self.output_type.unreduced_axes().is_empty() {
            return Err(TypeError::invalid(format!(
                "`{RNG_BIT_GENERATOR_OPERATION_NAME}` does not support unreduced outputs",
            )));
        }
        self.output_type.element_count()?;

        // Declaring an invariant bits type for a varying state would let a replicated `shard_map` output accept bits
        // that actually differ across devices, so the state's variation is union-ed into the declared type. A declared
        // type that cannot vary over those axes (e.g., because it is reduced over one of them) is rejected.
        let Some(state_sharding) = state_type.sharding().filter(|sharding| !sharding.varying_manual_axes().is_empty())
        else {
            return Ok(self.output_type.clone());
        };
        let mut output_sharding = match self.output_type.sharding() {
            Some(sharding) => sharding.clone(),
            None => Sharding::replicated(state_sharding.mesh().clone(), self.output_type.rank()),
        };
        output_sharding
            .extend_varying_manual_axes(state_sharding.varying_manual_axes().iter().cloned())
            .map_err(|error| {
                TypeError::invalid(format!(
                    "`{RNG_BIT_GENERATOR_OPERATION_NAME}` output type `{}` cannot vary over the varying manual axes \
                     of its state: {error}",
                    self.output_type,
                ))
            })?;
        self.output_type
            .clone()
            .with_sharding(output_sharding)
            .map_err(|error| TypeError::invalid(error.to_string()))
    }

    /// Renders this payload independently of its homogeneous or array IR input contract. This is a separate function
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
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let bits_type = self.infer_bits_type(&input_types[0])?;
        if self.output_type.static_shape().is_none() {
            return Err(TypeError::invalid(format!(
                "`{RNG_BIT_GENERATOR_OPERATION_NAME}` does not support dynamically shaped outputs",
            )));
        }
        Ok(vec![input_types[0].clone(), bits_type])
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
        // The state is followed by one first-class extent input per dynamic output axis. Each extent must define the
        // dimension variable that the corresponding declared output axis refers to, or have an exact type that refines
        // that axis to its extent (refer to `DimensionRefinements::record_input_refinement`).
        check_count!("region", region_interfaces, 0, TypeError);
        let dynamic_output_dimensions =
            self.output_type.shape().dimensions().iter().filter_map(Dimension::variable).collect::<Vec<_>>();
        check_count!("input", input_types, dynamic_output_dimensions.len() + 1, TypeError);
        let state_type = <&ArrayType>::try_from(&input_types[0])?;
        let bits_type = self.infer_bits_type(state_type)?;
        let mut refinements = DimensionRefinements::default();
        for (input_type, expected_variable) in input_types[1..].iter().zip(dynamic_output_dimensions) {
            let input_type = <&DimensionType>::try_from(input_type)?;
            if !refinements.record_input_refinement(expected_variable, input_type)? {
                return Err(TypeError::invalid(format!(
                    "`{}` output-extent input defines dimension variable `{}`, but the corresponding \
                     declared output axis refers to `{}`",
                    RNG_BIT_GENERATOR_OPERATION_NAME,
                    input_type.dimension(),
                    expected_variable,
                )));
            }
        }
        Ok(vec![
            state_type.clone().into(),
            ArrayTypeRefinements::from(refinements).refine_declared(&bits_type, &[])?.into(),
        ])
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

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for RngBitGeneratorOperation<ArrayType>
where
    C::Value: RngBitGenerator,
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

impl<C: Domain<Type = ArrayIrType>> InterpretableOperation<C> for RngBitGeneratorOperation<ArrayIrType>
where
    C::Value: DynamicRngBitGenerator,
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
    C: Context<Type = ArrayType, Operation: From<RngBitGeneratorOperation<ArrayType>> + From<ScanOperation<C::Type>>>,
    P: ArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for RngBitGeneratorOperation<ArrayType>
where
    C::Value: Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);

        // Every batch item of a replicated state computes the same function of the same state, so generating the
        // bits once and replicating both outputs is exact.
        if inputs[0].batch_axis().is_replicated() {
            let mut outputs =
                context.parent().bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].value()))?;
            check_count!("output", outputs, 2, ProgramError);
            let bits = outputs.remove(1);
            let state = outputs.remove(0);
            return Ok(vec![ArrayBatch::replicated(state), ArrayBatch::replicated(bits)]);
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
        let scan = ScanOperation::<C::Type>::new(0, P::axis_size(context)?);
        let mut outputs = context.parent().bind(scan, vec![body], std::slice::from_ref(states.value()))?;
        check_count!("output", outputs, 2, ProgramError);
        let bits = outputs.remove(1);
        let states = outputs.remove(0);
        Ok(vec![ArrayBatch::new(states, BatchAxis::new(0))?, ArrayBatch::new(bits, BatchAxis::new(0))?])
    }
}

impl<C: Context<Type = ArrayIrType>> BatchableOperation<C, ArrayIrBatchingPolicy>
    for RngBitGeneratorOperation<ArrayIrType>
where
    C::Value: ValueProjection<ArrayType>,
    <C::Value as ValueProjection<ArrayType>>::Projected: Transpose + Value<Type = ArrayType>,
    C::Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    C::Operation: From<DynamicBroadcastOperation>
        + From<ConstantOperation<DimensionValue>>
        + From<DimensionSizeOperation>
        + From<RngBitGeneratorOperation<ArrayIrType>>
        + From<ScanOperation<C::Type>>
        + OperationProjection<ArrayType, Projected: From<TransposeOperation>>,
{
    fn batch<D: BatchingDriver<C, ArrayIrBatchingPolicy>>(
        &self,
        context: &BatchingContext<C, ArrayIrBatchingPolicy>,
        driver: &D,
        inputs: &[ArrayIrBatch<C::Value>],
    ) -> Result<Vec<ArrayIrBatch<C::Value>>, BatchingError> {
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
            return Ok(vec![ArrayIrBatch::replicated(state), ArrayIrBatch::replicated(bits)]);
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
        let length = <&DimensionType>::try_from(extent_type.as_ref())?.simplified_dimension();
        let scan = ScanOperation::<C::Type>::new(output_dimensions.len(), length.clone());
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
        Ok(vec![ArrayIrBatch::new(states, BatchAxis::new(0))?, ArrayIrBatch::new(bits, BatchAxis::new(0))?])
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
/// output may be sharded, since every shard would otherwise draw the same bits, or carry unreduced mesh axes. To draw
/// distinct bits per shard, derive per-shard states inside the body of a
/// [`ShardMapOperation`](crate::ShardMapOperation) instead. The bits then vary over the varying manual axes of their
/// state in addition to those of `output_type`. Concrete [`Array`]s generate the bits immediately, bit-identical with
/// XLA's [`rng_bit_generator`](https://github.com/openxla/xla/blob/main/xla/hlo/builder/lib/prng.cc) expansion, while
/// context-carrying values bind an [`RngBitGeneratorOperation`] through their own context. The bits are integers,
/// and so their derivative is a structural zero.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// Note that array IR values also generate random bits through their array views, which stages the static array
/// generator; the native array IR generator with dynamic output dimensions is [`DynamicRngBitGenerator`].
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
#[capability(projection(ArrayIrType => ArrayType))]
pub trait RngBitGenerator<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Generates random bits of `output_type` from this generator state using `algorithm`, returning the advanced state
    /// together with the bits.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if this state does not have the state type of `algorithm`, if `output_type` is not
    /// a statically shaped and unsharded unsigned-integer type, if the state or output has unreduced mesh axes, if
    /// the output cannot vary over the varying manual axes of the state, if the output element count overflows
    /// [`usize`], or if the context of the value fails to bind the operation.
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
        let mut output_types = RngBitGeneratorOperation::<ArrayType>::new(algorithm, output_type.clone())
            .infer_output_types(&[self.r#type().into_owned()], &[])?;
        let output_type = output_types.remove(1);

        // Type inference guarantees a static output shape, an unsigned-integer output data type, and exactly as many
        // decoded `u64` state elements as the algorithm requires. Narrower-than-32-bit outputs keep the low bits of
        // each generated `u32` word. The generated values are encoded in logical order, so the declared physical
        // layouts of both the state and the bits are preserved. The bits take the inferred output type, which also
        // varies over the varying manual axes of the state.

        // Validate storage size and layout before allocating any cipher-word buffers.
        let addressing = ArrayAddressing::new(output_type.clone())?;
        let dimensions = output_type.static_shape().unwrap().dimensions().to_vec();
        let count = addressing.element_count();
        if count == 0 {
            return Ok((self.clone(), Array::from_logical_bytes(output_type.clone(), &[])?));
        }

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
            Dispatch = ValueDomainDispatch,
            Type = ArrayType,
            Domain: Context<Type = ArrayType, Operation: From<RngBitGeneratorOperation<ArrayType>>>,
        >,
> RngBitGenerator<ArrayType> for V
{
    fn rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
    ) -> Result<(Self, Self), ProgramError> {
        let mut outputs = self.domain().bind(
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
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that array IR values
/// implement this capability for [`ArrayIrType`], whose first-class dimension members supply the dynamic extents.
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
#[capability]
pub trait DynamicRngBitGenerator<T = <Self as Capability>::Universe>: Capability + Sized {
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

impl<A: Value<Type = ArrayType> + RngBitGenerator> DynamicRngBitGenerator<ArrayIrType> for ArrayIrValue<A> {
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
            Dispatch = ValueDomainDispatch,
            Type = ArrayIrType,
            Domain: Context<Type = ArrayIrType, Operation: From<RngBitGeneratorOperation<ArrayIrType>>>,
        >,
> DynamicRngBitGenerator<ArrayIrType> for V
{
    fn dynamic_rng_bit_generator(
        &self,
        algorithm: RandomAlgorithm,
        output_type: &ArrayType,
        output_dimensions: &[Self],
    ) -> Result<(Self, Self), ProgramError> {
        let inputs = std::iter::once(self).chain(output_dimensions).cloned().collect::<Vec<_>>();
        let mut outputs = self.domain().bind(
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

/// Precision of the Gumbel noise used by [`Random::random_categorical_with_mode`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CategoricalSamplingMode {
    /// Draws one uniform per logit. Rare events below the uniform spacing may be biased (approximately `2⁻²⁴`
    /// for `f32` computation or `2⁻⁵³` for `f64` computation).
    LowPrecision,

    /// Draws two uniforms per logit and uses [`Ln1p`] to preserve the upper Gumbel tail. This approximately doubles
    /// sampling work and extends the rare-event range to approximately `2⁻⁴⁸` for `f32` or `2⁻¹⁰⁶` for `f64`.
    HighPrecision,
}

/// Represents the ability to draw random samples from a counter-based generator state. Every function selects the
/// [`RandomAlgorithm`] from the state type (see [`RandomAlgorithm::from_state_type`]), threads the state functionally
/// by returning the advanced state alongside its result, and is a pure composition of [`RngBitGenerator`] and ordinary
/// array operations, so the distributions inherit their transform rules from those operations. The recipes are:
///
///   - [`split_rng_key`](Self::split_rng_key) draws one `u64` key per fresh state and pairs it with a zero counter.
///   - [`random_uniform`](Self::random_uniform) draws 32-bit words for `f32` samples and 64-bit words for `f64`
///     samples, keeps their top 24 or 53 bits, and scales them by `2⁻²⁴` or `2⁻⁵³`. Narrower floating-point samples
///     use 32-bit words and an exactly representable grid computed in `f32` before conversion. Every grid excludes
///     `1`. `f4e2m1fn` and `f6e2m3fn` use fewer precision bits because their exponent ranges cannot represent the
///     otherwise smallest grid points.
///   - [`random_normal`](Self::random_normal) applies the Box-Muller transform `√(-2 ln(1 - u₁)) · cos(2π u₂)` to two
///     uniform draws, where `1 - u₁ > 0` keeps the logarithm finite. Narrower types round an `f32` computation once.
///     Complex samples combine two independent real normals scaled by `1/√2`, giving unit expected squared modulus.
///   - [`random_categorical`](Self::random_categorical) applies the Gumbel-max trick
///     `argmax(logits - ln(-ln(u + tiny)))` along the category axis. Narrower logits use `f32` arithmetic, and `tiny`
///     is the smallest positive normal value of the computation data type. The Gumbel noise is therefore always
///     finite, so `-∞` logits are never sampled unless every logit along the axis is `-∞`, and ties resolve to the
///     lowest index. [`random_categorical_with_mode`](Self::random_categorical_with_mode) can instead use two uniform
///     draws and accurate `ln(1 - x)` to retain more of the upper Gumbel tail.
///
/// These are the distributions of JAX's
/// [`jax.random.uniform`](https://docs.jax.dev/en/latest/_autosummary/jax.random.uniform.html),
/// [`jax.random.normal`](https://docs.jax.dev/en/latest/_autosummary/jax.random.normal.html), and
/// [`jax.random.categorical`](https://docs.jax.dev/en/latest/_autosummary/jax.random.categorical.html), although the
/// samples differ bitwise because JAX derives its bits differently and draws normal samples through the inverse error
/// function.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// Note that array IR values sample through their array views, so that the whole composition runs in the array
/// universe.
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
#[capability(projection(ArrayIrType => ArrayType))]
pub trait Random<T = <Self as Capability>::Universe>: Capability + Sized {
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
    /// element data type is not a floating-point type with zero (in particular, `f8e8m0fnu` is unsupported), if it
    /// has sharded dimensions or unreduced axes, or if the context of the value fails to bind an operation.
    fn random_uniform(&self, r#type: &ArrayType) -> Result<(Self, Self), ProgramError>;

    /// Draws standard-normal samples of type `r#type`, returning the advanced state together with the samples. The
    /// samples have the shape, element data type, and memory space of `r#type`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] under the same conditions as [`Self::random_uniform`], except that `c64` and
    /// `c128` outputs are also supported.
    fn random_normal(&self, r#type: &ArrayType) -> Result<(Self, Self), ProgramError>;

    /// Draws one categorical sample for every position of `logits` outside `axis`, using
    /// [`CategoricalSamplingMode::LowPrecision`]. Refer to [`Self::random_categorical_with_mode`]
    /// for semantics and errors. Existing `f32` and `f64` draws use one uniform per logit.
    #[inline]
    fn random_categorical<A: Into<Axis>>(&self, logits: &Self, axis: A) -> Result<(Self, Self), ProgramError> {
        self.random_categorical_with_mode(logits, axis, CategoricalSamplingMode::LowPrecision)
    }

    /// Draws one categorical sample for every position of `logits` outside `axis`, returning the advanced state
    /// together with `i32` indices. The sample shape is the shape of `logits` with `axis` removed. Ties select the
    /// lowest index and NaNs follow [`ArgMax`] semantics. Infinite negative logits remain masked unless every category
    /// is masked, in which case the lowest index is returned.
    ///
    /// # Parameters
    ///
    ///   - `logits`: Floating-point unnormalized log-probabilities. Narrower types use `f32` arithmetic, while `f64`
    ///     retains its precision. The zero-free `f8e8m0fnu` type is unsupported.
    ///   - `axis`: [`Axis`] holding the categories. Negative axes count from the end.
    ///   - `mode`: Precision of the Gumbel noise. High precision consumes two uniform draws instead of one.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if this value is not a generator state, if `logits` has unsupported elements,
    /// a dynamic shape, sharded dimensions, or unreduced axes, if `axis` is out of bounds or empty, if its indices
    /// cannot be represented by `i32`, or if the context of the value fails to bind an operation.
    fn random_categorical_with_mode<A: Into<Axis>>(
        &self,
        logits: &Self,
        axis: A,
        mode: CategoricalSamplingMode,
    ) -> Result<(Self, Self), ProgramError>;
}

impl<
    V: Value<Type = ArrayType, Domain: Fill<f64, V>>
        + ZeroLike
        + Neg
        + Add
        + Sub
        + Mul
        + Div
        + Sqrt
        + Log
        + Ln1p
        + Cos
        + Complex
        + Compare
        + Select
        + ArgMax
        + Concatenate
        + Slice
        + ConvertElementType
        + RngBitGenerator,
> Random<ArrayType> for V
{
    fn split_rng_key(&self, count: usize) -> Result<(Self, Vec<Self>), ProgramError> {
        let state_type = self.r#type();
        let algorithm = RandomAlgorithm::from_state_type(&state_type)?;
        let keys_type = state_type.as_ref().clone().with_shape([count]).with_layout(None);
        let (state, keys) = self.rng_bit_generator(algorithm, &keys_type)?;

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
        let (bits_data_type, bit_count, precision, compute_data_type) = match data_type {
            DataType::F4E2M1FN => (DataType::U32, 32, 1, DataType::F32),
            DataType::F6E2M3FN | DataType::F6E3M2FN | DataType::F8E5M2 | DataType::F8E5M2FNUZ => {
                (DataType::U32, 32, 3, DataType::F32)
            }
            DataType::F8E3M4 => (DataType::U32, 32, 5, DataType::F32),
            DataType::F8E4M3 | DataType::F8E4M3FN | DataType::F8E4M3FNUZ | DataType::F8E4M3B11FNUZ => {
                (DataType::U32, 32, 4, DataType::F32)
            }
            DataType::BF16 => (DataType::U32, 32, 8, DataType::F32),
            DataType::F16 => (DataType::U32, 32, 11, DataType::F32),
            DataType::F32 => (DataType::U32, 32, 24, DataType::F32),
            DataType::F64 => (DataType::U64, 64, 53, DataType::F64),
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

        // Keep the top precision bits and scale in a type that represents both the integer and the grid exactly.
        // Scaling before narrowing avoids overflowing the narrow type or rounding the largest sample up to one.
        let compute_type = if data_type == compute_data_type {
            r#type.clone()
        } else {
            r#type.clone().with_layout(None).with_data_type(compute_data_type)
        };
        let domain = self.domain();
        let divisor: Self = domain.fill(&bits_type, 2.0f64.powi(bit_count - precision))?;
        let scale: Self = domain.fill(&compute_type, 2.0f64.powi(-precision))?;
        let samples = bits.div(&divisor)?.convert_element_type(compute_data_type)?.mul(&scale)?;
        let samples = if data_type == compute_data_type { samples } else { samples.convert_element_type(data_type)? };
        Ok((state, samples))
    }

    fn random_normal(&self, r#type: &ArrayType) -> Result<(Self, Self), ProgramError> {
        let data_type = r#type.data_type();
        if data_type.is_complex() {
            let part_type = r#type.clone().with_layout(None).with_data_type(if data_type == DataType::C64 {
                DataType::F32
            } else {
                DataType::F64
            });
            let (state, real) = self.random_normal(&part_type)?;
            let (state, imaginary) = state.random_normal(&part_type)?;
            let scale: Self = self.domain().fill(&part_type, std::f64::consts::FRAC_1_SQRT_2)?;
            return Ok((state, real.mul(&scale)?.complex(&imaginary.mul(&scale)?)?));
        }

        if data_type.is_floating_point() && !matches!(data_type, DataType::F32 | DataType::F64 | DataType::F8E8M0FNU) {
            let compute_type = r#type.clone().with_layout(None).with_data_type(DataType::F32);
            let (state, samples) = self.random_normal(&compute_type)?;
            return Ok((state, samples.convert_element_type(data_type)?));
        }

        let (state, first) = self.random_uniform(r#type)?;
        let (state, second) = state.random_uniform(r#type)?;
        let domain = self.domain();
        let one: Self = domain.fill(r#type, 1.0)?;
        let minus_two: Self = domain.fill(r#type, -2.0)?;
        let two_pi: Self = domain.fill(r#type, std::f64::consts::TAU)?;
        let radius = one.sub(&first)?.log()?.mul(&minus_two)?.sqrt()?;
        let angle = second.mul(&two_pi)?.cos()?;
        Ok((state, radius.mul(&angle)?))
    }

    fn random_categorical_with_mode<A: Into<Axis>>(
        &self,
        logits: &Self,
        axis: A,
        mode: CategoricalSamplingMode,
    ) -> Result<(Self, Self), ProgramError> {
        let logits_type = logits.r#type();
        let data_type = logits_type.data_type();
        if !data_type.is_floating_point() || data_type == DataType::F8E8M0FNU {
            return Err(TypeError::invalid(format!(
                "`random_categorical` does not support logits data type `{data_type}`",
            ))
            .into());
        }

        // Form noise and compare in at least single precision, rather than rounding it at every narrow operation.
        let logits = if matches!(data_type, DataType::F32 | DataType::F64) {
            logits.clone()
        } else {
            logits.convert_element_type(DataType::F32)?
        };

        let compute_type = logits.r#type().into_owned();
        let (state, uniform) = self.random_uniform(&compute_type)?;
        let domain = self.domain();
        let (state, gumbel) = match mode {
            CategoricalSamplingMode::LowPrecision => {
                // The shift changes only zero samples and keeps the noise finite, including for masked logits.
                let tiny = if compute_type.data_type() == DataType::F64 {
                    f64::MIN_POSITIVE
                } else {
                    f64::from(f32::MIN_POSITIVE)
                };
                let tiny: Self = domain.fill(&compute_type, tiny)?;
                (state, uniform.add(&tiny)?.log()?.neg()?.log()?.neg()?)
            }
            CategoricalSamplingMode::HighPrecision => {
                let (state, low) = state.random_uniform(&compute_type)?;
                let precision = if compute_type.data_type() == DataType::F64 { 53 } else { 24 };
                let half: Self = domain.fill(&compute_type, 0.5)?;
                let scale: Self = domain.fill(&compute_type, 2.0f64.powi(-precision))?;
                let offset: Self = domain.fill(&compute_type, 2.0f64.powi(-(2 * precision + 1)))?;

                // Extend the lower uniform tail with a second draw. Keeping the upper half unchanged prevents
                // rounding to one, and ln_1p retains the small differences that ordinary `log(1 - x)` would lose.
                let extended = uniform.add(&low.mul(&scale)?)?.add(&offset)?;
                let uniform = uniform.greater_than_or_equal(&half)?.select(&uniform, &extended)?;
                (state, uniform.neg()?.ln_1p()?.neg()?.log()?.neg()?)
            }
        };

        Ok((state, logits.add(&gumbel)?.argmax(axis)?))
    }
}

/// Canonical operation name for [`DropoutOperation`].
pub const DROPOUT_OPERATION_NAME: &str = "dropout";

/// [`Operation`] that randomly zeroes elements of its input and scales the remaining elements so that the expected
/// value of every element is preserved. Refer to the documentation of [`Dropout`] for the dropout semantics. The first
/// input is the array to drop elements from, and an optional second input is a random generator state. The first
/// output is the result, and the advanced random state follows it when a state is provided.
///
/// Only the distribution of the dropout mask is specified, while the exact mask is implementation-defined, which lets
/// backends use their own random streams (e.g., the fused dropout of cuDNN attention kernels):
///
///   - **With a random state**, the mask is a deterministic function of the state, the mask shape (i.e.,
///     [`mask_type`](Self::mask_type)), and the rate, and a backend never replaces this operation with a fused kernel
///     that draws its own mask.
///   - **Without a random state**, a backend may replace this operation, possibly together with the computation around
///     it, by a fused kernel that draws its own mask. Ryft's own implementations use a fixed state instead, and so
///     every call with the same mask shape and rate applies the same mask (e.g., at every training step). Provide
///     a state when fresh masks are needed.
///
/// Either way, every implementation draws its mask deterministically, so applying this operation to two arrays with the
/// same mask shape applies the same mask to both. Differentiation relies on this, since the tangent of the output is
/// this same operation applied to the input tangent. Transposition therefore applies this same operation to the
/// output cotangent and requires a known random state, and the random states have structural zero tangents. A backend
/// that fuses a stateless dropout must therefore also apply its mask to the derivatives of that dropout (e.g., by
/// fusing the corresponding backward computation with the same random stream), because a derivative dropout that runs
/// in Ryft's own implementation would otherwise apply a different mask. When only the input is unknown during partial
/// evaluation, the advanced state is computed from the known state alone, so that later draws remain known.
///
/// Batching a replicated or absent state adds the batch axis to the [`broadcast_axes`](Self::broadcast_axes), because
/// every batch item applies the same mask when the operation is unbatched. Batching a mapped state (e.g., one state per
/// batch item derived with [`Random::split_rng_key`]) stages one carry-free [`ScanOperation`] over the per-item states
/// and inputs, so that each batch item applies exactly the mask that its own state draws unbatched. This requires a
/// statically known batch size, as for [`RngBitGeneratorOperation`]. Note also that the XLA backend lowers this
/// operation only for statically shaped inputs, even though dynamic extents are allowed along the broadcast axes.
#[derive(Clone, Debug)]
pub struct DropoutOperation {
    /// Refer to the documentation of [`rate`](Self::rate) for more information.
    rate: f64,

    /// Refer to the documentation of [`broadcast_axes`](Self::broadcast_axes) for more information.
    broadcast_axes: Axes,
}

impl DropoutOperation {
    /// Creates a new [`DropoutOperation`] with the provided rate and broadcast axes. Type inference validates both.
    #[inline]
    pub fn new<A: Into<Axes>>(rate: f64, broadcast_axes: A) -> Self {
        Self { rate, broadcast_axes: broadcast_axes.into() }
    }

    /// Returns the probability with which this operation drops each element of its mask, which must be finite and in
    /// `[0, 1)`. Equality and hashing compare it bitwise, so that `-0.0` and `0.0` rates are distinct operations.
    #[inline]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Returns the input axes along which this operation shares its mask, so that all elements that differ only in
    /// their coordinates along these axes are either all kept or all dropped. Negative axes count from the end.
    #[inline]
    pub fn broadcast_axes(&self) -> &Axes {
        &self.broadcast_axes
    }

    /// Returns the factor by which Ryft's own implementations scale the kept elements, before rounding it to the input
    /// data type. It is the reciprocal of the probability with which those implementations keep an element (i.e.,
    /// `2³² / (2³² - drop_threshold())`, which is `1 / (1 - rate)` up to the rounding of the rate to a multiple of
    /// `2⁻³²`), so that the expected value of every element is preserved exactly for every rate below one. A rate of
    /// one drops every element, which needs no scale, and returns zero.
    #[inline]
    pub fn scale(&self) -> f64 {
        if self.rate >= 1.0 { 0.0 } else { 2.0f64.powi(32) / (2.0f64.powi(32) - f64::from(self.drop_threshold())) }
    }

    /// Returns the `u32` threshold `round(rate · 2³²)`, clamped to [`u32::MAX`], with which Ryft's own implementations
    /// select the kept elements. Each implementation draws one uniformly distributed `u32` word per mask element with
    /// [`RngBitGenerator`] and keeps the element if and only if that word is greater than or equal to this threshold.
    /// The probability with which an element is kept is therefore `1 - rate` rounded to a multiple of `2⁻³²` (and at
    /// least `2⁻³²`), and [`scale`](Self::scale) is its reciprocal. A rate of one instead drops every element, although
    /// the words are still drawn, so that the advanced state does not depend on the rate. Backends that lower this
    /// operation without a fused kernel use the same rule, so that a given state draws the same mask in every one of
    /// Ryft's implementations.
    #[inline]
    pub fn drop_threshold(&self) -> u32 {
        (self.rate * 2.0f64.powi(32)).round().min(f64::from(u32::MAX)) as u32
    }

    /// Returns `true` if this operation drops every element (i.e., its rate is one).
    #[inline]
    pub fn drops_everything(&self) -> bool {
        self.rate >= 1.0
    }

    /// Returns the type of the `u32` words from which Ryft's own implementations draw the mask of an input of type
    /// `input_type` (refer to [`drop_threshold`](Self::drop_threshold)). Its shape is the input shape with every
    /// broadcast axis replaced by a static extent of one, and its memory space and sharding metadata are those of the
    /// input, except that the broadcast axes are replicated and that it has no unreduced or reduced axes. It has no
    /// physical layout. The mask is broadcast along the broadcast axes when it is applied.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] if a broadcast axis is out of bounds or repeated, or if an axis that is not a broadcast
    /// axis has a dynamic extent.
    pub fn mask_type(&self, input_type: &ArrayType) -> Result<ArrayType, TypeError> {
        let broadcast_axes = self
            .broadcast_axes
            .normalize(input_type.rank())
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let dimensions = input_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .map(|(axis, dimension)| match dimension {
                _ if broadcast_axes.contains(&axis) => Ok(Dimension::Static(1)),
                Dimension::Static(extent) => Ok(Dimension::Static(*extent)),
                Dimension::Dynamic(_) => Err(TypeError::invalid(format!(
                    "`{DROPOUT_OPERATION_NAME}` requires static extents along the axes that are not broadcast axes, \
                     but axis {axis} of `{input_type}` has extent `{dimension}`",
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_type = input_type.without_reduction_axes();
        let sharding = input_type
            .sharding()
            .map(|sharding| {
                sharding.with_dimensions(
                    sharding
                        .dimensions()
                        .iter()
                        .enumerate()
                        .map(|(axis, dimension)| {
                            if broadcast_axes.contains(&axis) {
                                ShardingDimension::replicated()
                            } else {
                                dimension.clone()
                            }
                        })
                        .collect(),
                )
            })
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        input_type
            .clone()
            .with_data_type(DataType::U32)
            .with_shape(Shape::new(dimensions))
            .with_layout(None)
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()))
    }
}

impl PartialEq for DropoutOperation {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.rate.to_bits() == other.rate.to_bits() && self.broadcast_axes == other.broadcast_axes
    }
}

impl Eq for DropoutOperation {}

impl Hash for DropoutOperation {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.rate.to_bits().hash(state);
        self.broadcast_axes.hash(state);
    }
}

impl Display for DropoutOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DropoutOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        DROPOUT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("region", region_interfaces, 0, TypeError);
        let (input_type, state_type) = match input_types {
            [input_type] => (input_type, None),
            [input_type, state_type] => (input_type, Some(state_type)),
            _ => {
                return Err(TypeError::invalid(format!(
                    "`{DROPOUT_OPERATION_NAME}` expects an input followed by an optional random state, \
                     but got {} inputs",
                    input_types.len(),
                )));
            }
        };

        // `contains` rejects NaN rates. A rate of one drops every element.
        if !(0.0..=1.0).contains(&self.rate) {
            return Err(TypeError::invalid(format!(
                "`{}` requires a rate in `[0, 1]` but got `{:?}`",
                DROPOUT_OPERATION_NAME, self.rate,
            )));
        }

        let data_type = input_type.data_type();
        if !(data_type.is_floating_point() || data_type.is_complex()) || data_type == DataType::F8E8M0FNU {
            return Err(TypeError::invalid(format!(
                "`{DROPOUT_OPERATION_NAME}` does not support input data type `{data_type}`",
            )));
        }

        // A sharded mask would draw the same words on every shard, as for `rng_bit_generator`, while a sharded
        // broadcast axis only replicates the mask. For a fixed state, dropout is linear and its mask is the same on
        // every device of a reduction group, so it commutes with the pending reduction of an unreduced input, whose
        // unreduced axes the output keeps. An unreduced state has no meaning.
        let is_sharded = |array_type: &ArrayType| {
            array_type.sharding().is_some_and(|sharding| {
                sharding.dimensions().iter().any(|dimension| matches!(dimension, ShardingDimension::Sharded(_)))
            })
        };

        if is_sharded(&self.mask_type(input_type)?) || state_type.is_some_and(is_sharded) {
            return Err(TypeError::invalid(format!(
                "`{DROPOUT_OPERATION_NAME}` supports sharding only along the broadcast axes of its input \
                 and never for its random state, so apply it inside `shard_map` instead",
            )));
        }

        if let Some(state_type) = state_type {
            check_types!(@no_unreduced, DROPOUT_OPERATION_NAME, &[state_type]);
            RandomAlgorithm::from_state_type(state_type)?;
        }

        // A state that varies over manual mesh axes draws different masks on different devices, so the input must
        // already vary over those axes, as the inputs of other operations must agree on their manual variation. The
        // capabilities align the input (e.g., with `parallel_vary`), whose adjoint then reduces the cotangent over the
        // axes that only the state varies over.
        let Some(state_type) = state_type else {
            return Ok(vec![input_type.clone()]);
        };
        if let Some(state_sharding) = state_type.sharding() {
            let input_axes = input_type.sharding().map(Sharding::varying_manual_axes);
            let missing = state_sharding
                .varying_manual_axes()
                .iter()
                .filter(|axis| !input_axes.is_some_and(|input_axes| input_axes.contains(*axis)))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(TypeError::invalid(format!(
                    "`{DROPOUT_OPERATION_NAME}` input type `{input_type}` must vary over the varying manual axes of \
                     its random state, but it does not vary over {missing:?}",
                )));
            }
        }
        Ok(vec![input_type.clone(), state_type.clone()])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("rate", format_args!("{:?}", self.rate))?;
            operation.field(
                "broadcast_axes",
                format_args!("{:?}", self.broadcast_axes.iter().map(|axis| axis.value()).collect::<Vec<_>>()),
            )
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free DropoutOperation);

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for DropoutOperation
where
    C::Value: Dropout,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let Some((input, state)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
        };
        if state.len() > 1 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() });
        }
        let (output, state) = input.dropout_with_broadcast_axes(self.rate, &self.broadcast_axes, state.first())?;
        Ok(std::iter::once(output).chain(state).collect())
    }
}

impl<C: Context<Type = ArrayType, Operation: From<DropoutOperation> + From<RngBitGeneratorOperation<ArrayType>>>>
    PartiallyEvaluatableOperation<C> for DropoutOperation
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        _driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // The advanced state depends only on the state and on the number of words drawn at the mask type, so when only
        // the input is unknown, it is computed on the known side by the same `rng_bit_generator` draw, which keeps
        // later draws from that state known. The residual dropout recomputes the same state, which is then unused.
        let mut outputs = context.fold_or_residualize(self.clone(), Vec::new(), inputs)?;
        if let [input, state] = inputs
            && !input.is_known()
            && state.is_known()
        {
            check_count!("output", outputs, 2, ProgramError);
            let algorithm = RandomAlgorithm::from_state_type(&state.r#type())?;
            let generator = RngBitGeneratorOperation::<ArrayType>::new(algorithm, self.mask_type(&input.r#type())?);
            let mut generated = context.fold_or_residualize(generator, Vec::new(), std::slice::from_ref(state))?;
            check_count!("output", generated, 2, ProgramError);
            outputs[1] = generated.remove(0);
        }
        Ok(outputs)
    }
}

impl<
    C: Context<Type = ArrayType, Operation: From<DropoutOperation> + From<ScanOperation<C::Type>>>,
    P: ArrayExtentBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatchingPolicy<P>> for DropoutOperation
where
    C::Value: Dropout + Transpose,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        let Some((input, state)) = inputs.split_first() else {
            return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
        };

        if state.len() > 1 {
            return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
        }

        // Every batch item of a replicated or absent state draws the same mask from the same state, so a mapped input
        // shares one mask along its batch axis, which becomes an additional broadcast axis. The advanced state remains
        // replicated, because the mask shape, and therefore the number of words drawn, is unchanged.
        let state = state.first();
        if state.is_none_or(|state| state.batch_axis().is_replicated()) {
            let output_axes = std::iter::once(input.batch_axis())
                .chain(state.map(|_| BatchAxis::replicated()))
                .collect::<Vec<_>>();
            let Some(batch_axis) = input.batch_axis_position() else {
                return self.interpret_with_batch_axes(context, inputs, &output_axes);
            };
            let broadcast_axes = self
                .broadcast_axes
                .normalize(input.unbatched_type().rank())
                .map_err(|error| TypeError::invalid(error.to_string()))?
                .into_iter()
                .map(|axis| if axis >= batch_axis { axis + 1 } else { axis })
                .chain(std::iter::once(batch_axis))
                .collect::<Vec<_>>();
            return DropoutOperation::new(self.rate, broadcast_axes).interpret_with_batch_axes(
                context,
                inputs,
                &output_axes,
            );
        }

        // A mapped state is realigned to batch axis `0`, the input is moved or broadcast to batch axis `0`, and one
        // carry-free scan is staged over both, whose body applies this same operation to one per-item input and state.
        // Iteration `i` therefore applies exactly the mask that state row `i` draws unbatched, and the scan stacks the
        // per-item outputs and advanced states at batch axis `0`. Binding the scan through the parent context lets an
        // enclosing batching context batch it structurally.
        let state = state.unwrap();
        let input_type = input.unbatched_type();
        let state_type = state.unbatched_type();
        let input = P::match_axis(context, input, Axis::from(0))?;
        let states = state.move_axis(0)?;
        let mut builder = ProgramBuilder::<C::Constant, C::Operation>::new();
        builder.add_input(ArrayType::scalar(DataType::I64));
        let input_value = builder.add_input(input_type);
        let state_value = builder.add_input(state_type);
        let outputs = builder.add_instruction(self.clone(), Vec::new(), vec![input_value, state_value], None)?.to_vec();
        let body =
            builder.build::<Vec<C::Constant>, Vec<C::Constant>>(outputs, vec![Placeholder; 3], vec![Placeholder; 2])?;
        let scan = ScanOperation::<C::Type>::new(0, P::axis_size(context)?);
        let mut outputs = context.parent().bind(scan, vec![body], &[input.into_value(), states.into_value()])?;
        check_count!("output", outputs, 2, ProgramError);
        let states = outputs.remove(1);
        let output = outputs.remove(0);
        Ok(vec![ArrayBatch::new(output, BatchAxis::new(0))?, ArrayBatch::new(states, BatchAxis::new(0))?])
    }
}

impl_differentiable_operation! {
    DropoutOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: Dropout,
    {
        |operation, context, _driver, inputs| {
            // For a fixed state, dropout is a linear map of its input whose mask depends only on the state and the
            // mask shape. Applying this same operation with the same primal state to the input tangent, whose mask
            // shape equals that of the primal input, therefore applies the same mask. The integer random states have
            // structural zero tangents, and a zero input tangent yields a typed zero output tangent.
            let Some((input, state)) = inputs.split_first() else {
                return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }.into());
            };
            if state.len() > 1 {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            }
            let (rate, broadcast_axes) = (operation.rate(), operation.broadcast_axes());
            let state = state.first().map(|state| state.primal().clone());
            let (primal, advanced_state) =
                input.primal().dropout_with_broadcast_axes(rate, broadcast_axes, state.as_ref())?;
            let tangent = match input.tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => {
                    let state = state.map(|state| context.primal_to_tangent(state)).transpose()?;
                    let (tangent, _) = tangent.dropout_with_broadcast_axes(rate, broadcast_axes, state.as_ref())?;
                    MaybeZero::Value(tangent)
                }
            };
            let mut outputs = vec![DifferentiationDual::new(primal, tangent)?];
            if let Some(advanced_state) = advanced_state {
                outputs.push(DifferentiationDual::new_with_zero_tangent(advanced_state)?);
            }
            Ok(outputs)
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<DropoutOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType> + Dropout,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // The mask is a diagonal of zeros and real scales, so the transpose of the linear map of the input applies
            // the same mask to the output cotangent, which requires the same known state. The integer state has no
            // cotangent space and the cotangent of the advanced state is always a structural zero.
            if inputs.is_empty() || inputs.len() > 2 {
                return Err(ProgramError::InvalidInputCount { expected: 2, actual: inputs.len() }.into());
            }
            check_count!("output", outputs, inputs.len(), ProgramError);
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            match &outputs[0] {
                MaybeZero::Zero(_) => Ok(()),
                MaybeZero::Value(_) if !accumulators[0].is_needed() => Ok(()),
                MaybeZero::Value(cotangent) => {
                    let state = inputs
                        .get(1)
                        .map(|state| {
                            state.as_known().cloned().ok_or_else(|| ProgramError::InvalidArgument {
                                message: format!("`{DROPOUT_OPERATION_NAME}` transpose requires a known random state"),
                            })
                        })
                        .transpose()?;
                    let (contribution, _) = cotangent.dropout_with_broadcast_axes(
                        operation.rate(),
                        operation.broadcast_axes(),
                        state.as_ref(),
                    )?;
                    let contribution = contribution.unalign_cotangent(&inputs[0].r#type().cotangent()?)?;
                    accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;
                    Ok(())
                }
            }
        }
    },
}

/// Represents the ability to apply dropout, which randomly zeroes elements of an array and scales the remaining
/// elements so that the expected value of every element is preserved. Each element of a random mask is kept with
/// probability `1 - rate`, and the result multiplies every kept element by the reciprocal of the probability with which
/// it is kept (i.e., `1 / (1 - rate)`, refer to [`DropoutOperation::scale`] for its exact value), rounded to the input
/// data type, and sets every dropped element to zero (including non-finite ones). A rate of one drops every element.
/// The elements along the broadcast axes share their mask element. This is the dropout of
/// [Srivastava et al.](https://jmlr.org/papers/v15/srivastava14a.html) and the broadcast axes are `broadcast_dims` of
/// [`flax.linen.Dropout`](https://flax.readthedocs.io/en/latest/api_reference/flax.linen/layers.html#flax.linen.Dropout).
///
/// Only the distribution of the mask is specified, while the exact mask is implementation-defined. With a random state,
/// the mask is a deterministic function of the state, the shape of the input with every broadcast axis replaced by one,
/// and the rate, and the advanced state is returned for drawing further random values. Without a random state, backends
/// may replace the dropout, possibly together with the computation around it, by a fused kernel that draws its own mask
/// (e.g., the fused dropout of cuDNN attention kernels), while Ryft's own implementations draw the mask from a fixed
/// state, so that every call with the same mask shape and rate applies the same mask. Refer to [`DropoutOperation`]
/// for how Ryft's implementations draw the mask and how dropout is batched and differentiated. Concrete [`Array`]s
/// apply the dropout immediately, while context-carrying values bind a [`DropoutOperation`] through their own context.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// Note that array IR values apply dropout through their array views.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, Dropout, ProgramError, RandomAlgorithm};
/// # fn main() -> Result<(), ProgramError> {
/// let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[42u64, 0])?;
/// let input = Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// let (output, state) = input.dropout(0.5, Some(&state))?;
///
/// // Every element is either dropped or doubled, and the same state always applies the same mask.
/// let elements = output.elements::<f32>()?;
/// assert!(elements.iter().zip(1..).all(|(element, index)| [0.0, 2.0 * index as f32].contains(element)));
/// assert!(state.is_some());
///
/// // A broadcast axis shares the mask along it, so each column is either kept or dropped as a whole.
/// let state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[7u64, 0])?;
/// let (output, _) = Array::matrix(2, 3, vec![1.0f32; 6])?.dropout_with_broadcast_axes(0.5, [0], Some(&state))?;
/// let elements = output.elements::<f32>()?;
/// assert_eq!(elements[..3], elements[3..]);
/// # Ok(())
/// # }
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait Dropout<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Applies dropout with probability `rate` and no broadcast axes, returning the result together with the advanced
    /// random state when `state` is provided. Refer to [`Self::dropout_with_broadcast_axes`] for more information.
    #[inline]
    fn dropout(&self, rate: f64, state: Option<&Self>) -> Result<(Self, Option<Self>), ProgramError> {
        self.dropout_with_broadcast_axes(rate, Axes::default(), state)
    }

    /// Applies dropout, returning the result together with the advanced random state when `state` is provided.
    ///
    /// # Parameters
    ///
    ///   - `rate`: Probability with which each mask element is dropped, which must be in `[0, 1]`. A rate of one drops
    ///     every element and returns zeros.
    ///   - `broadcast_axes`: Axes along which the mask is shared. Negative axes count from the end.
    ///   - `state`: Optional random generator state with the [`RandomAlgorithm::state_type`] of its algorithm. When it
    ///     is [`None`], backends may replace the dropout with a fused kernel, and Ryft's own implementations use a
    ///     fixed state (refer to [`DropoutOperation`]).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `rate` is not in `[0, 1]`, if this value is not a floating-point or complex array
    /// (in particular, `f8e8m0fnu` is unsupported), if a broadcast axis is out of bounds or repeated, if an axis that
    /// is not a broadcast axis has a dynamic extent or is sharded, if `state` is not a generator state, is sharded, or
    /// has unreduced axes, if this value cannot vary over the varying manual axes of `state` (e.g., because it is
    /// reduced over one of them), or if the context of the value fails to bind the operation. Unreduced inputs are
    /// supported, and the result keeps their unreduced axes. This value is first made to vary over the varying manual
    /// axes of `state`.
    fn dropout_with_broadcast_axes<A: Into<Axes>>(
        &self,
        rate: f64,
        broadcast_axes: A,
        state: Option<&Self>,
    ) -> Result<(Self, Option<Self>), ProgramError>;
}

impl Dropout for Array {
    fn dropout_with_broadcast_axes<A: Into<Axes>>(
        &self,
        rate: f64,
        broadcast_axes: A,
        state: Option<&Self>,
    ) -> Result<(Self, Option<Self>), ProgramError> {
        let operation = DropoutOperation::new(rate, broadcast_axes);

        // Concrete arrays hold their manual variation as type metadata only, so aligning the input with the variation
        // of the state retypes it.
        let mut input_type = self.r#type().into_owned();
        if let Some(state_sharding) = state.map(|state| state.r#type().into_owned()).and_then(|state_type| {
            state_type.sharding().filter(|sharding| !sharding.varying_manual_axes().is_empty()).cloned()
        }) {
            let mut sharding = match input_type.sharding() {
                Some(sharding) => sharding.clone(),
                None => Sharding::replicated(state_sharding.mesh().clone(), input_type.rank()),
            };
            sharding.extend_varying_manual_axes(state_sharding.varying_manual_axes().iter().cloned()).map_err(
                |error| {
                    TypeError::invalid(format!(
                        "`{DROPOUT_OPERATION_NAME}` input type `{input_type}` cannot vary over the varying manual axes \
                         of its random state: {error}",
                    ))
                },
            )?;
            input_type = input_type.with_sharding(sharding).map_err(|error| TypeError::invalid(error.to_string()))?;
        }
        let input_types = std::iter::once(input_type.clone())
            .chain(state.map(|state| state.r#type().into_owned()))
            .collect::<Vec<_>>();
        let mut output_types = operation.infer_output_types(&input_types, &[])?;

        // Draw one word per mask element from the provided state, or from the fixed `[0, 0]` ThreeFry state, keep the
        // elements whose words reach the drop threshold, and broadcast the mask along the broadcast axes. Selecting
        // rather than multiplying by the mask keeps dropped non-finite elements at zero.
        let fixed_state;
        let (algorithm, generator_state) = match state {
            Some(state) => (RandomAlgorithm::from_state_type(&state.r#type())?, state),
            None => {
                fixed_state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[0u64, 0])?;
                (RandomAlgorithm::ThreeFry, &fixed_state)
            }
        };

        // The mask is computed on views whose types carry neither the varying manual axes of the state, which the words
        // inherit from it, nor the pending reductions of the input, which dropout commutes with. The output then takes
        // the inferred type, which restores both.
        let mask_type = operation.mask_type(&input_type)?;
        let (advanced_state, words) = generator_state.rng_bit_generator(algorithm, &mask_type)?;
        let words = if words.r#type().as_ref() == &mask_type {
            words
        } else {
            Self::from_logical_bytes(mask_type.clone(), words.logical_bytes().as_slice())?
        };
        let compute_type = input_type.without_reduction_axes();
        let input = if &compute_type == self.r#type().as_ref() {
            self.clone()
        } else {
            Self::from_logical_bytes(compute_type.clone(), self.logical_bytes().as_slice())?
        };
        let domain = self.domain();
        let threshold: Self = domain.fill(&mask_type, operation.drop_threshold())?;
        let keep = words.greater_than_or_equal(&threshold)?.broadcast(
            compute_type.clone().with_data_type(DataType::Boolean).with_layout(None),
            &(0..compute_type.rank()).collect::<Vec<_>>(),
        )?;
        let scale: Self = domain.fill(&compute_type, operation.scale())?;
        let output = if operation.drops_everything() {
            input.zero_like()?
        } else {
            keep.select(&input.mul(&scale)?, &input.zero_like()?)?
        };
        let output_type = output_types.remove(0);
        let output = if output.r#type().as_ref() == &output_type {
            output
        } else {
            Self::from_logical_bytes(output_type, output.logical_bytes().as_slice())?
        };
        Ok((output, state.map(|_| advanced_state)))
    }
}

impl<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayType,
            Domain: Context<Type = ArrayType, Operation: From<DropoutOperation>>,
        > + ManualVariationAlignment,
> Dropout<ArrayType> for V
{
    fn dropout_with_broadcast_axes<A: Into<Axes>>(
        &self,
        rate: f64,
        broadcast_axes: A,
        state: Option<&Self>,
    ) -> Result<(Self, Option<Self>), ProgramError> {
        // The input must vary over the varying manual axes of the state.
        let input = match state.map(|state| state.r#type().into_owned()) {
            Some(state_type)
                if state_type.sharding().is_some_and(|sharding| !sharding.varying_manual_axes().is_empty()) =>
            {
                Self::align_manual_variation_to(std::slice::from_ref(self), state_type.sharding().unwrap())?.remove(0)
            }
            _ => self.clone(),
        };
        let inputs = std::iter::once(input).chain(state.cloned()).collect::<Vec<_>>();
        let mut outputs = self.domain().bind(DropoutOperation::new(rate, broadcast_axes), Vec::new(), &inputs)?;
        check_count!("output", outputs, inputs.len(), ProgramError);
        let state = (outputs.len() == 2).then(|| outputs.remove(1));
        Ok((outputs.remove(0), state))
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
    if dimensions.contains(&0) {
        return (Vec::new(), counter);
    }
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
    use std::collections::HashSet;

    use half::f16;
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayIrValue, ArrayOperation, DimensionBounds, Layout, LogicalMesh, Memory, MeshAxis,
        MeshAxisType, Sharding, StridedLayout,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchedProgram, BatchingTracer, ProgramBatchingOutputAxesPolicy, RecursiveBatchingPolicy};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationDual, DifferentiationError,
        TransposableOperation, TranspositionContext, differentiate_at,
    };
    use crate::macros::{
        check_operation_batching, check_operation_partial_evaluation, check_operation_transposition,
        check_operation_type_inference,
    };
    use crate::operations::collectives::parallel_vary::ParallelVary;
    use crate::partial::PartialValue;
    use crate::programs::{EffectClasses, EmptyRegionDriver, MaybeZero, TypedProgram};
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
    fn test_rng_bit_generator_type_inference_output_count_and_regions() {
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let operation = RngBitGeneratorOperation::<ArrayType>::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new_static(DataType::U32, [usize::MAX, 2]),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&state_type), &[]),
            Err(TypeError::invalid(format!("shape [{}, 2] element count does not fit in `usize`", usize::MAX))),
        );
        let operation = RngBitGeneratorOperation::<ArrayType>::new(
            RandomAlgorithm::ThreeFry,
            ArrayType::new_static(DataType::U32, [1]),
        );
        assert_eq!(
            operation.infer_output_types(
                &[state_type],
                &[RegionInterface::new(Vec::new(), Vec::new(), EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
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
    fn test_rng_bit_generator_type_inference_unreduced_axes() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap();
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let output_type = ArrayType::new_static(DataType::U32, [4]);
        let unreduced_state = state_type.clone().with_sharding(unreduced.clone()).unwrap();
        let unreduced_output = output_type.clone().with_sharding(unreduced).unwrap();

        // Both type universes enforce the same nonlinear state and output contract.
        let operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type.clone());
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&unreduced_state), &[]),
            Err(TypeError::invalid("`rng_bit_generator` does not support unreduced inputs")),
        );
        assert_eq!(
            RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::ThreeFry, output_type)
                .infer_output_types(&[unreduced_state.into()], &[]),
            Err(TypeError::invalid("`rng_bit_generator` does not support unreduced inputs")),
        );
        assert_eq!(
            RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, unreduced_output.clone())
                .infer_output_types(std::slice::from_ref(&state_type), &[]),
            Err(TypeError::invalid("`rng_bit_generator` does not support unreduced outputs")),
        );
        assert_eq!(
            RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::ThreeFry, unreduced_output)
                .infer_output_types(&[state_type.into()], &[]),
            Err(TypeError::invalid("`rng_bit_generator` does not support unreduced outputs")),
        );
    }

    #[test]
    fn test_rng_bit_generator_type_inference_manual_variation() {
        // Bits drawn from a state that varies over a manual mesh axis differ across devices, so they vary over that
        // axis even when the declared output type is invariant or unsharded. An output type that is reduced over that
        // axis cannot hold them and is rejected.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let varying_state_type = state_type
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let output_type = ArrayType::new_static(DataType::U32, [4]);
        let invariant_output_type = output_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let varying_output_type = output_type
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let reduced_output_type = output_type
            .clone()
            .with_sharding(Sharding::replicated(mesh, 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, output_type.clone()),
            cases = [
                {
                    input_types = [varying_state_type.clone()],
                    output_types = [varying_state_type.clone(), varying_output_type.clone()],
                },
                {
                    input_types = [state_type.clone()],
                    output_types = [state_type, output_type.clone()],
                },
            ],
        );
        check_operation_type_inference!(
            operation = RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, invariant_output_type),
            cases = [{
                input_types = [varying_state_type.clone()],
                output_types = [varying_state_type.clone(), varying_output_type.clone()],
            }],
        );
        check_operation_type_inference!(
            operation =
                RngBitGeneratorOperation::<ArrayType>::new(RandomAlgorithm::ThreeFry, reduced_output_type.clone()),
            cases = [{
                input_types = [varying_state_type.clone()],
                error = format!(
                    "`rng_bit_generator` output type `{reduced_output_type}` cannot vary over the varying manual axes \
                     of its state: manual axis `x` cannot be both varying and reduced",
                ),
            }],
        );

        // The array IR type universe applies the same rule.
        assert_eq!(
            RngBitGeneratorOperation::<ArrayIrType>::new(RandomAlgorithm::ThreeFry, output_type)
                .infer_output_types(&[varying_state_type.clone().into()], &[]),
            Ok(vec![varying_state_type.into(), varying_output_type.into()]),
        );
    }

    #[test]
    fn test_rng_bit_generator_type_inference_dynamic() {
        // The array IR contract takes one extent input per dynamic output axis, in axis order, and each extent must
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
    fn test_rng_bit_generator_interpretation_empty_output() {
        for (algorithm, state) in [
            (RandomAlgorithm::ThreeFry, threefry_state(42, u64::MAX)),
            (RandomAlgorithm::Philox, philox_state(42, u128::MAX)),
        ] {
            let output_type = ArrayType::new_static(DataType::U32, [usize::MAX, 2, 0]);
            assert_eq!(
                state.rng_bit_generator(algorithm, &output_type),
                Ok((state.clone(), Array::from_logical_bytes(output_type, &[]).unwrap())),
            );
        }
        assert_eq!(threefry_u32_words(42, u64::MAX, &[usize::MAX, 2, 0]), (Vec::new(), u64::MAX));
    }

    #[test]
    fn test_rng_bit_generator_interpretation_invalid_storage() {
        let state = threefry_state(42, 7);
        let output_type = ArrayType::new_static(DataType::U64, [usize::MAX]);
        assert_eq!(
            state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type),
            Err(TypeError::invalid(format!("array type `{output_type}` requires more bytes than can be represented"))
                .into()),
        );
        let output_type =
            ArrayType::new_static(DataType::U32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, 8])));
        assert_eq!(
            state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type),
            Err(TypeError::invalid("strided layout rank 2 does not match array rank 1").into()),
        );
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

        // Context-carrying mixed values stage one array IR bit-generation instruction.
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
                let %1:u64[3, 2], %2:u32[3, 4] = scan [carry_count=0, iteration_count=3, reverse=false] %0 [
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

        // A mapped state over a dynamic batch extent stages one array IR scan whose replicated output extents are
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
            panic!("mapped random bit generation should stage the direct array IR scan carrier");
        };
        assert_eq!(scan_operation.carry_count(), 2);
        assert_eq!(scan_operation.iteration_count(), &Dimension::Dynamic(batch));
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

        // A replicated state stages the array IR operation once and replicates both outputs.
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

        // Split keys and counters retain the parent's memory space; byte strides are recomputed for each child.
        let state_type = RandomAlgorithm::ThreeFry
            .state_type()
            .with_memory(Memory::Host { pinned: true })
            .with_layout(Layout::Strided(StridedLayout::new(vec![-8])));
        let parent = Array::from_elements(state_type.clone(), &[42u64, 7]).unwrap();
        let (advanced, children) = parent.split_rng_key(2).unwrap();
        assert_eq!(advanced.r#type().as_ref(), &state_type);
        for child in children {
            assert_eq!(child.r#type().memory(), state_type.memory());
            assert_eq!(child.elements::<u64>().unwrap()[1], 0);
        }

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
    fn test_random_array_ir() {
        // Concrete array IR values generate bits and sample through their array members.
        let state = threefry_state(42, 7);
        let array_ir = ArrayIrValue::Array(state.clone());
        let output_type = ArrayType::new_static(DataType::U32, [2]);
        let (next_state, bits) = state.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type).unwrap();
        assert_eq!(
            array_ir.rng_bit_generator(RandomAlgorithm::ThreeFry, &output_type),
            Ok((ArrayIrValue::Array(next_state), ArrayIrValue::Array(bits))),
        );
        let (next_state, keys) = state.split_rng_key(2).unwrap();
        assert_eq!(
            array_ir.split_rng_key(2),
            Ok((ArrayIrValue::Array(next_state), keys.into_iter().map(ArrayIrValue::Array).collect())),
        );
        let sample_type = ArrayType::new_static(DataType::F32, [3]);
        let (next_state, samples) = state.random_uniform(&sample_type).unwrap();
        assert_eq!(
            array_ir.random_uniform(&sample_type),
            Ok((ArrayIrValue::Array(next_state), ArrayIrValue::Array(samples))),
        );
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
            state.random_uniform(&ArrayType::new_static(DataType::F8E8M0FNU, [2])),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`random_uniform` does not support output data type `f8e8m0fnu`",
        ));
    }

    #[test]
    fn test_random_uniform_endpoints_and_narrow_types() {
        // These counters invert the ThreeFry cipher under key zero to blocks [0, 0] and [MAX, MAX].
        let zero = threefry_state(0, 0xdcf9249d7f72bced);
        let maximum = threefry_state(0, 0x864a12b6bcf893de);
        for (data_type, precision) in [
            (DataType::F4E2M1FN, 1),
            (DataType::F6E2M3FN, 3),
            (DataType::F6E3M2FN, 3),
            (DataType::F8E3M4, 5),
            (DataType::F8E4M3, 4),
            (DataType::F8E4M3FN, 4),
            (DataType::F8E4M3FNUZ, 4),
            (DataType::F8E4M3B11FNUZ, 4),
            (DataType::F8E5M2, 3),
            (DataType::F8E5M2FNUZ, 3),
            (DataType::BF16, 8),
            (DataType::F16, 11),
            (DataType::F32, 24),
            (DataType::F64, 53),
        ] {
            let output_type = ArrayType::scalar(data_type);
            assert_eq!(zero.random_uniform(&output_type).unwrap().1.to_f64s(), vec![0.0]);
            assert_eq!(maximum.random_uniform(&output_type).unwrap().1.to_f64s(), vec![1.0 - 2.0f64.powi(-precision)]);
            let output_type = ArrayType::new_static(data_type, [128]);
            let (_, samples) = threefry_state(42, 7).random_uniform(&output_type).unwrap();
            assert_eq!(samples.r#type().as_ref(), &output_type);
            for sample in samples.to_f64s() {
                assert!((0.0..1.0).contains(&sample));
                assert_eq!(sample * 2.0f64.powi(precision), (sample * 2.0f64.powi(precision)).floor());
            }
        }

        // Empty draws do not advance the counter, and staging preserves the narrow grid and state advancement.
        let output_type = ArrayType::new_static(DataType::F16, [0]);
        assert_eq!(zero.random_uniform(&output_type).unwrap().0, zero);
        let output_type = ArrayType::new_static(DataType::F6E2M3FN, [3]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |state| state.random_uniform(&output_type),
            RandomAlgorithm::ThreeFry.state_type(),
        )
        .unwrap();
        assert_eq!(program.interpret(maximum.clone()), maximum.random_uniform(&output_type));
    }

    #[test]
    fn test_random_uniform_manual_variation() {
        // Inside a manual region, a state that varies over a manual mesh axis draws different samples on every device,
        // so the samples vary over that axis even though the requested sample type is invariant.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let state_type = RandomAlgorithm::ThreeFry
            .state_type()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let sample_type = ArrayType::new_static(DataType::F32, [3]);
        let (output_types, _) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |state| state.random_uniform(&sample_type),
            state_type.clone(),
            vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
        )
        .unwrap();
        let varying_sample_type = sample_type
            .with_sharding(Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        assert_eq!(output_types, (state_type, varying_sample_type));
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
    fn test_random_normal_narrow_and_complex_types() {
        let state = threefry_state(42, 7);
        let compute_type = ArrayType::new_static(DataType::F32, [128]);
        let (expected_state, expected) = state.random_normal(&compute_type).unwrap();
        for data_type in [
            DataType::F4E2M1FN,
            DataType::F6E2M3FN,
            DataType::F6E3M2FN,
            DataType::F8E3M4,
            DataType::F8E4M3,
            DataType::F8E4M3FN,
            DataType::F8E4M3FNUZ,
            DataType::F8E4M3B11FNUZ,
            DataType::F8E5M2,
            DataType::F8E5M2FNUZ,
            DataType::BF16,
            DataType::F16,
        ] {
            let output_type = compute_type.clone().with_data_type(data_type);
            assert_eq!(
                state.random_normal(&output_type),
                Ok((expected_state.clone(), expected.convert_element_type(data_type).unwrap())),
            );
        }

        // A zero first uniform has zero radius rather than an infinite logarithm.
        for data_type in [DataType::F32, DataType::F64] {
            let (_, sample) =
                threefry_state(0, 0xdcf9249d7f72bced).random_normal(&ArrayType::scalar(data_type)).unwrap();
            assert_eq!(sample.to_f64s()[0], 0.0);
        }

        for (data_type, part_data_type) in [(DataType::C64, DataType::F32), (DataType::C128, DataType::F64)] {
            let output_type = ArrayType::new_static(data_type, [16]).with_memory(Memory::Host { pinned: true });
            let part_type = output_type.clone().with_data_type(part_data_type);
            let (after_real, real) = state.random_normal(&part_type).unwrap();
            let (expected_state, imaginary) = after_real.random_normal(&part_type).unwrap();
            let scale = EagerContext::<Array, ArrayOperation<Array>>::new()
                .fill(&part_type, std::f64::consts::FRAC_1_SQRT_2)
                .unwrap();
            let expected = real.mul(&scale).unwrap().complex(&imaginary.mul(&scale).unwrap()).unwrap();
            assert_eq!(state.random_normal(&output_type), Ok((expected_state, expected)));
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |state| state.random_normal(&output_type),
                RandomAlgorithm::ThreeFry.state_type(),
            )
            .unwrap();
            assert_eq!(program.interpret(state.clone()), state.random_normal(&output_type));
        }
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
    fn test_random_categorical_with_mode() {
        // This key-zero counter yields the first uniform pair [0, 0.5]. High precision retains the rare upper
        // Gumbel tail at zero; computing log(1 - x) would round to log(1) and make the noise infinite instead.
        let tail_state = threefry_state(0, 0x1badd8a64e5859ae);
        let tail_logits = Array::vector(vec![-16f32, 0.0]).unwrap();
        assert_eq!(
            tail_state
                .random_categorical_with_mode(&tail_logits, 0, CategoricalSamplingMode::LowPrecision)
                .unwrap()
                .1,
            Array::scalar(1i32).unwrap(),
        );
        assert_eq!(
            tail_state
                .random_categorical_with_mode(&tail_logits, 0, CategoricalSamplingMode::HighPrecision)
                .unwrap()
                .1,
            Array::scalar(0i32).unwrap(),
        );

        let state = threefry_state(42, 7);
        for data_type in [DataType::F16, DataType::BF16, DataType::F4E2M1FN, DataType::F8E4M3FNUZ] {
            let logits = Array::vector(vec![0f32, 4.0, 0.0]).unwrap().convert_element_type(data_type).unwrap();
            let wide_logits = logits.convert_element_type(DataType::F32).unwrap();
            for mode in [CategoricalSamplingMode::LowPrecision, CategoricalSamplingMode::HighPrecision] {
                assert_eq!(
                    state.random_categorical_with_mode(&logits, 0, mode),
                    state.random_categorical_with_mode(&wide_logits, 0, mode),
                );
            }
        }

        for data_type in [DataType::F32, DataType::F64] {
            let logits = Array::vector(vec![f64::NEG_INFINITY, 0.0]).unwrap().convert_element_type(data_type).unwrap();
            let masked = Array::vector(vec![f64::NEG_INFINITY, f64::NEG_INFINITY])
                .unwrap()
                .convert_element_type(data_type)
                .unwrap();
            for mode in [CategoricalSamplingMode::LowPrecision, CategoricalSamplingMode::HighPrecision] {
                for counter in [0xdcf9249d7f72bced, 0x864a12b6bcf893de] {
                    let state = threefry_state(0, counter);
                    assert_eq!(
                        state.random_categorical_with_mode(&logits, 0, mode).unwrap().1,
                        Array::scalar(1i32).unwrap(),
                    );
                    assert_eq!(
                        state.random_categorical_with_mode(&masked, 0, mode).unwrap().1,
                        Array::scalar(0i32).unwrap(),
                    );
                }
                let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                    |(state, logits)| state.random_categorical_with_mode(&logits, -1, mode),
                    (RandomAlgorithm::ThreeFry.state_type(), logits.r#type().into_owned()),
                )
                .unwrap();
                assert_eq!(
                    program.interpret((state.clone(), logits.clone())),
                    state.random_categorical_with_mode(&logits, -1, mode),
                );
            }
        }

        // High precision's extra draw is visible in the advanced state, even for a single category.
        let logits = Array::vector(vec![0f32]).unwrap();
        assert_eq!(
            state.random_categorical_with_mode(&logits, 0, CategoricalSamplingMode::LowPrecision).unwrap().0,
            threefry_state(42, 8),
        );
        assert_eq!(
            state.random_categorical_with_mode(&logits, 0, CategoricalSamplingMode::HighPrecision).unwrap().0,
            threefry_state(42, 9),
        );
    }

    #[test]
    fn test_dropout() {
        let operation = DropoutOperation::new(0.25, [-1]);
        assert_eq!(operation.name(), DROPOUT_OPERATION_NAME);
        assert_eq!(operation.rate(), 0.25);
        assert_eq!(operation.broadcast_axes(), &Axes::from([-1]));
        assert_eq!(operation.scale(), 4.0 / 3.0);
        assert_eq!(operation.drop_threshold(), 1 << 30);
        assert_eq!(operation.to_string(), "dropout [rate=0.25, broadcast_axes=[-1]]");

        // Thresholds round to the nearest multiple of `2⁻³²` and saturate just below a rate of one.
        assert_eq!(DropoutOperation::new(0.0, Axes::default()).drop_threshold(), 0);
        assert_eq!(DropoutOperation::new(2.0f64.powi(-34), Axes::default()).drop_threshold(), 0);
        assert_eq!(DropoutOperation::new(1.0 - 2.0f64.powi(-40), Axes::default()).drop_threshold(), u32::MAX);

        // The scale is the reciprocal of the keep probability that the threshold yields, so the expectation of every
        // element is preserved even where the rate rounds (i.e., `2³²` rather than `2⁴⁰` just below a rate of one),
        // and a rate of one drops everything without a scale.
        assert_eq!(DropoutOperation::new(1.0 - 2.0f64.powi(-40), Axes::default()).scale(), 2.0f64.powi(32));
        assert_eq!(DropoutOperation::new(0.0, Axes::default()).scale(), 1.0);
        assert!(!DropoutOperation::new(0.5, Axes::default()).drops_everything());
        assert!(DropoutOperation::new(1.0, Axes::default()).drops_everything());
        assert_eq!(DropoutOperation::new(1.0, Axes::default()).scale(), 0.0);

        // Rates compare and hash bitwise, so that operations are faithful keys of their attributes.
        let zero = DropoutOperation::new(0.0, Axes::default());
        let negative_zero = DropoutOperation::new(-0.0, Axes::default());
        assert_ne!(zero, negative_zero);
        assert_eq!(DropoutOperation::new(f64::NAN, [0]), DropoutOperation::new(f64::NAN, [0]));
        assert_ne!(DropoutOperation::new(0.5, [0]), DropoutOperation::new(0.5, [1]));
        let operations = HashSet::from([zero.clone(), negative_zero.clone()]);
        assert_eq!(operations.len(), 2);
        assert!(operations.contains(&zero) && operations.contains(&negative_zero));
    }

    #[test]
    fn test_dropout_mask_type() {
        // Broadcast axes have a single mask element, and the mask keeps the memory space but not the layout.
        let operation = DropoutOperation::new(0.5, [0]);
        let input_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![4, 8])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            operation.mask_type(&input_type),
            Ok(ArrayType::new_static(DataType::U32, [1, 3]).with_memory(Memory::Host { pinned: true })),
        );

        // Only axes that are not broadcast axes need static extents.
        let length = Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()));
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![length, Dimension::Static(3)]));
        assert_eq!(operation.mask_type(&input_type), Ok(ArrayType::new_static(DataType::U32, [1, 3])));
        assert_eq!(
            DropoutOperation::new(0.5, [1]).mask_type(&input_type),
            Err(TypeError::invalid(format!(
                "`dropout` requires static extents along the axes that are not broadcast axes, \
                 but axis 0 of `{}` has extent `{}`",
                input_type,
                input_type.dimension(0),
            ))),
        );
        assert_eq!(
            DropoutOperation::new(0.5, [2]).mask_type(&input_type),
            Err(TypeError::invalid("axis 2 is out of bounds for rank 2")),
        );
        assert_eq!(
            DropoutOperation::new(0.5, [0, -2]).mask_type(&input_type),
            Err(TypeError::invalid("axes contain duplicate axis 0")),
        );
    }

    #[test]
    fn test_dropout_type_inference() {
        let input_type = ArrayType::new_static(DataType::BF16, [2, 3]);
        let state_type = RandomAlgorithm::Philox.state_type();
        check_operation_type_inference!(
            operation = DropoutOperation::new(0.5, [1]),
            cases = [
                { input_types = [input_type.clone()], output_types = [input_type.clone()] },
                {
                    input_types = [input_type.clone(), state_type.clone()],
                    output_types = [input_type.clone(), state_type.clone()],
                },
                {
                    input_types = [ArrayType::new_static(DataType::I32, [2])],
                    error = "`dropout` does not support input data type `i32`",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F8E8M0FNU, [2])],
                    error = "`dropout` does not support input data type `f8e8m0fnu`",
                },
                {
                    input_types = [input_type.clone(), ArrayType::new_static(DataType::U32, [2])],
                    error = "random generator states must have type `u64[2]` (i.e., for `three_fry`) or `u64[3]` \
                             (i.e., for `philox`) but got `u32[2]`",
                },
                {
                    input_types = [input_type.clone(), state_type.clone(), state_type.clone()],
                    error = "`dropout` expects an input followed by an optional random state, but got 3 inputs",
                },
                {
                    input_types = [ArrayType::new_static(DataType::F32, [2])],
                    error = "axis 1 is out of bounds for rank 1",
                },
            ],
        );
        check_operation_type_inference!(
            operation = DropoutOperation::new(0.5, Axes::default()),
            cases = [{
                input_types = [ArrayType::new_static(DataType::C64, [2])],
                output_types = [ArrayType::new_static(DataType::C64, [2])],
            }],
        );
        for rate in [1.5, -0.5, f64::NAN, f64::INFINITY] {
            check_operation_type_inference!(
                operation = DropoutOperation::new(rate, Axes::default()),
                cases = [{
                    input_types = [input_type.clone()],
                    error = format!("`dropout` requires a rate in `[0, 1]` but got `{rate:?}`"),
                }],
            );
        }
    }

    #[test]
    fn test_dropout_type_inference_sharding() {
        // A sharded mask would draw the same words on every shard, while sharded broadcast axes
        // only replicate the mask.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded_type = ArrayType::new_static(DataType::F32, [4, 2])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        let sharded_state_type = RandomAlgorithm::ThreeFry
            .state_type()
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let error = "`dropout` supports sharding only along the broadcast axes of its input and never for its random \
                     state, so apply it inside `shard_map` instead";
        check_operation_type_inference!(
            operation = DropoutOperation::new(0.5, [0]),
            cases = [
                { input_types = [sharded_type.clone()], output_types = [sharded_type.clone()] },
                { input_types = [sharded_type.clone(), sharded_state_type], error = error },
            ],
        );
        check_operation_type_inference!(
            operation = DropoutOperation::new(0.5, [1]),
            cases = [{ input_types = [sharded_type], error = error }],
        );
    }

    #[test]
    fn test_dropout_type_inference_unreduced_axes() {
        // For a fixed state, dropout commutes with the pending reduction of an unreduced input, which it keeps,
        // while unreduced states are rejected.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let unreduced_state_type = state_type
            .clone()
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = DropoutOperation::new(0.5, Axes::default()),
            cases = [
                { input_types = [unreduced_type.clone()], output_types = [unreduced_type.clone()] },
                {
                    input_types = [unreduced_type.clone(), state_type.clone()],
                    output_types = [unreduced_type.clone(), state_type],
                },
                {
                    input_types = [unreduced_type, unreduced_state_type],
                    error = "`dropout` does not support unreduced inputs",
                },
            ],
        );
    }

    #[test]
    fn test_dropout_type_inference_manual_variation() {
        // A state that varies over a manual mesh axis draws different masks on different devices, so the input must
        // already vary over that axis, which the capabilities ensure (refer to `test_dropout_manual_variation`).
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let state_type = RandomAlgorithm::ThreeFry.state_type();
        let varying_state_type = state_type
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let invariant_input_type = input_type.clone().with_sharding(Sharding::replicated(mesh.clone(), 1)).unwrap();
        let varying_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let reduced_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh, 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = DropoutOperation::new(0.5, Axes::default()),
            cases = [
                {
                    input_types = [varying_input_type.clone(), varying_state_type.clone()],
                    output_types = [varying_input_type.clone(), varying_state_type.clone()],
                },
                {
                    input_types = [varying_input_type.clone(), state_type.clone()],
                    output_types = [varying_input_type.clone(), state_type],
                },
                {
                    input_types = [input_type.clone(), varying_state_type.clone()],
                    error = format!(
                        "`dropout` input type `{input_type}` must vary over the varying manual axes of its random \
                         state, but it does not vary over [\"x\"]",
                    ),
                },
                {
                    input_types = [invariant_input_type.clone(), varying_state_type.clone()],
                    error = format!(
                        "`dropout` input type `{invariant_input_type}` must vary over the varying manual axes of its \
                         random state, but it does not vary over [\"x\"]",
                    ),
                },
                {
                    input_types = [reduced_input_type.clone(), varying_state_type],
                    error = format!(
                        "`dropout` input type `{reduced_input_type}` must vary over the varying manual axes of its \
                         random state, but it does not vary over [\"x\"]",
                    ),
                },
            ],
        );
    }

    #[test]
    fn test_dropout_manual_variation() {
        // Inside a manual region, an invariant input is made to vary over the axes of its state before dropout, so
        // reverse mode reduces its cotangent back over those axes, through the capability and through direct binding
        // of explicitly aligned inputs alike.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let state_type = RandomAlgorithm::ThreeFry
            .state_type()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let varying_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        for direct in [false, true] {
            let (output_types, _) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |(input, state)| {
                    let (output, pullback) = differentiate_at((input, state)).vjp(|(input, state)| {
                        if direct {
                            let input = input.parallel_vary("x")?;
                            let operation = DropoutOperation::new(0.5, Axes::default());
                            Ok(input.domain().bind(operation, Vec::new(), &[input.clone(), state])?.remove(0))
                        } else {
                            Ok(input.dropout(0.5, Some(&state))?.0)
                        }
                    })?;
                    let (MaybeZero::Value(input_cotangent), _) = pullback.apply(output.clone())? else {
                        panic!("the input cotangent of dropout is never a structural zero");
                    };
                    Ok((output, input_cotangent))
                },
                (input_type.clone(), state_type.clone()),
                vec![("x".to_string(), NamedAxis::Mesh { axis: 0, size: 2, mesh: mesh.clone() })],
            )
            .unwrap();
            assert_eq!(output_types, (varying_input_type.clone(), input_type.clone()));
        }
    }

    #[test]
    fn test_dropout_interpretation() {
        // Each element is kept if and only if its mask word reaches the drop threshold, and kept elements are scaled.
        let state = threefry_state(42, 7);
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4, 8]),
            &(0..32).map(|value| value as f32 + 1.0).collect::<Vec<_>>(),
        )
        .unwrap();
        let operation = DropoutOperation::new(0.25, Axes::default());
        let (advanced_state, words) = state
            .rng_bit_generator(RandomAlgorithm::ThreeFry, &ArrayType::new_static(DataType::U32, [4, 8]))
            .unwrap();
        let expected = input
            .elements::<f32>()
            .unwrap()
            .into_iter()
            .zip(words.elements::<u32>().unwrap())
            .map(|(element, word)| if word >= operation.drop_threshold() { element * (4.0 / 3.0) } else { 0.0 })
            .collect::<Vec<_>>();
        let (output, output_state) = input.dropout(0.25, Some(&state)).unwrap();
        assert_eq!(output.elements::<f32>(), Ok(expected.clone()));
        assert_eq!(output_state, Some(advanced_state));
        assert!(expected.contains(&0.0) && expected.iter().any(|element| *element != 0.0));

        // The same state always applies the same mask, while different states and algorithms apply different ones.
        assert_eq!(input.dropout(0.25, Some(&state)).unwrap().0, output);
        assert_ne!(input.dropout(0.25, Some(&threefry_state(43, 7))).unwrap().0, output);
        assert_ne!(input.dropout(0.25, Some(&philox_state(42, 7))).unwrap().0, output);

        // Without a state, the mask is drawn from the fixed `[0, 0]` ThreeFry state and no state is returned.
        assert_eq!(input.dropout(0.25, None), Ok((input.dropout(0.25, Some(&threefry_state(0, 0))).unwrap().0, None)));

        // A zero rate keeps every element unchanged, and the drawn fraction approximates the keep probability.
        assert_eq!(input.dropout(0.0, Some(&state)).unwrap().0, input);
        let ones = Array::from_elements(ArrayType::new_static(DataType::F64, [10_000]), &[1.0; 10_000]).unwrap();
        let kept = ones
            .dropout(0.3, Some(&state))
            .unwrap()
            .0
            .to_f64s()
            .iter()
            .filter(|element| **element != 0.0)
            .count();
        assert!((6_800..7_200).contains(&kept), "kept {kept} of 10000 elements");

        // Broadcast axes share the mask along them, and dropped non-finite elements become zero.
        let input = Array::matrix(3, 4, vec![f32::INFINITY; 12]).unwrap();
        let (output, _) = input.dropout_with_broadcast_axes(0.5, [0], Some(&state)).unwrap();
        let elements = output.elements::<f32>().unwrap();
        assert!(elements.iter().all(|element| *element == 0.0 || *element == f32::INFINITY));
        assert!(elements.contains(&0.0) && elements.contains(&f32::INFINITY));
        assert_eq!(elements[..4], elements[4..8]);
        assert_eq!(elements[..4], elements[8..]);

        // Complex elements are scaled by a real factor, and layouts and narrow types are preserved.
        let input = Array::vector(vec![ComplexNumber::new(1.0f32, -2.0); 8]).unwrap();
        let (output, _) = input.dropout(0.5, Some(&state)).unwrap();
        let elements = output.elements::<ComplexNumber<f32>>().unwrap();
        assert!(
            elements
                .iter()
                .all(|element| [ComplexNumber::new(0.0, 0.0), ComplexNumber::new(2.0, -4.0)].contains(element))
        );
        let input_type =
            ArrayType::new_static(DataType::F16, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![2, 4])));
        let input = Array::from_elements(input_type.clone(), &[f16::from_f32(1.5); 4]).unwrap();
        assert_eq!(input.dropout(0.5, Some(&state)).unwrap().0.r#type().as_ref(), &input_type);

        // A state that varies over a manual axis makes an invariant input vary over that axis, and an unreduced input
        // keeps its pending reduction, while both apply the same mask as their plain counterparts.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = Sharding::replicated(mesh, 1).with_varying_manual_axes(["x"]).unwrap();
        let varying_state = Array::from_elements(
            RandomAlgorithm::ThreeFry.state_type().with_sharding(varying_sharding.clone()).unwrap(),
            &[42u64, 7],
        )
        .unwrap();
        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let expected = input.dropout(0.5, Some(&state)).unwrap().0.elements::<f32>().unwrap();
        let (output, advanced_state) = input.dropout(0.5, Some(&varying_state)).unwrap();
        assert_eq!(output.r#type().into_owned(), input.r#type().into_owned().with_sharding(varying_sharding).unwrap());
        assert_eq!(output.elements::<f32>(), Ok(expected.clone()));
        assert_eq!(advanced_state.unwrap().r#type(), varying_state.r#type());
        let mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh, 1).with_unreduced_axes(["y"]).unwrap())
            .unwrap();
        let unreduced = Array::from_elements(unreduced_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (output, _) = unreduced.dropout(0.5, Some(&state)).unwrap();
        assert_eq!(output.r#type().as_ref(), &unreduced_type);
        assert_eq!(output.elements::<f32>(), Ok(expected));

        // Just below a rate of one, the single kept word of `2³²` scales its element by `2³²`, so expectations hold.
        // A rate of one drops every element, including non-finite ones, and advances the state like any other rate.
        let input = Array::vector(vec![1.0f32, f32::INFINITY, f32::NAN]).unwrap();
        let (output, advanced_state) = input.dropout(1.0, Some(&state)).unwrap();
        assert_eq!(output, Array::vector(vec![0.0f32; 3]).unwrap());
        assert_eq!(advanced_state, input.dropout(0.5, Some(&state)).unwrap().1);

        // Invalid rates are rejected before drawing any words.
        assert!(matches!(
            input.dropout(1.5, None),
            Err(ProgramError::Type(error))
                if error.to_string() == "`dropout` requires a rate in `[0, 1]` but got `1.5`",
        ));
    }

    #[test]
    fn test_dropout_partial_evaluation() {
        let state = threefry_state(42, 7);
        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (output, advanced_state) = input.dropout(0.5, Some(&state)).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = DropoutOperation::new(0.5, Axes::default()),
            cases = [
                {
                    inputs = [(@known, input.clone()), (@known, state.clone())],
                    outputs = [(@known, output.clone()), (@known, advanced_state.clone().unwrap())],
                    residual_instructions = 0,
                },
                {
                    inputs = [
                        (@unknown(type = input.r#type().into_owned(), replay = input.clone())),
                        (@known, state.clone()),
                    ],
                    outputs = [(@residual, output.clone()), (@known, advanced_state.clone().unwrap())],
                    residual_instructions = 1,
                },
                {
                    inputs = [
                        (@known, input.clone()),
                        (@unknown(type = state.r#type().into_owned(), replay = state.clone())),
                    ],
                    outputs = [(@residual, output), (@residual, advanced_state.unwrap())],
                    residual_instructions = 1,
                },
            ],
        );

        // Without a state, an unknown input leaves the only output unknown.
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = DropoutOperation::new(0.5, Axes::default()),
            cases = [{
                inputs = [(@unknown(type = input.r#type().into_owned(), replay = input.clone()))],
                outputs = [(@residual, input.dropout(0.5, None).unwrap().0)],
                residual_instructions = 1,
            }],
        );
    }

    #[test]
    fn test_dropout_batching() {
        // A replicated or absent state applies the same mask to every batch item, as unbatched dropout does,
        // while a mapped state applies the mask of each item's own state.
        let first_state = threefry_state(42, 7);
        let second_state = threefry_state(3, 11);
        let first_input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let second_input = Array::vector(vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0]).unwrap();
        let unbatched = |input: &Array, state: Option<&Array>| input.dropout(0.5, state).unwrap();
        let (first_with_first, advanced_first) = unbatched(&first_input, Some(&first_state));
        let (second_with_first, _) = unbatched(&second_input, Some(&first_state));
        let (second_with_second, advanced_second) = unbatched(&second_input, Some(&second_state));
        let (first_without_state, _) = unbatched(&first_input, None);
        let (second_without_state, _) = unbatched(&second_input, None);
        let stack = |first: &Array, second: &Array| {
            Array::matrix(2, 6, [first.elements::<f32>().unwrap(), second.elements::<f32>().unwrap()].concat()).unwrap()
        };
        let inputs = stack(&first_input, &second_input);
        check_operation_batching!(
            @exact,
            operation = DropoutOperation::new(0.5, Axes::default()),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 0), inputs.clone()), (@replicated, first_state.clone())],
                    outputs = [
                        (@mapped(axis = 0), stack(&first_with_first, &second_with_first)),
                        (@replicated, advanced_first.clone().unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), inputs.clone()),
                        (@mapped(axis = 1), Array::matrix(2, 2, vec![42u64, 3, 7, 11]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), stack(&first_with_first, &second_with_second)),
                        (@mapped(axis = 0), Array::matrix(
                            2,
                            2,
                            [
                                advanced_first.unwrap().elements::<u64>().unwrap(),
                                advanced_second.unwrap().elements::<u64>().unwrap(),
                            ]
                            .concat(),
                        ).unwrap()),
                    ],
                },
                {
                    inputs = [
                        (@replicated, first_input.clone()),
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![42u64, 7, 3, 11]).unwrap()),
                    ],
                    outputs = [
                        (@mapped(axis = 0), stack(&first_with_first, &unbatched(&first_input, Some(&second_state)).0)),
                        (@mapped(axis = 0), Array::matrix(
                            2,
                            2,
                            [
                                unbatched(&first_input, Some(&first_state)).1.unwrap().elements::<u64>().unwrap(),
                                unbatched(&first_input, Some(&second_state)).1.unwrap().elements::<u64>().unwrap(),
                            ]
                            .concat(),
                        ).unwrap()),
                    ],
                },
            ],
        );
        check_operation_batching!(
            @exact,
            operation = DropoutOperation::new(0.5, Axes::default()),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 1), Array::matrix(6, 2, {
                        let first = first_input.elements::<f32>().unwrap();
                        let second = second_input.elements::<f32>().unwrap();
                        first.into_iter().zip(second).flat_map(|(first, second)| [first, second]).collect::<Vec<_>>()
                    }).unwrap())],
                    outputs = [(@mapped(axis = 1), Array::matrix(6, 2, {
                        let first = first_without_state.elements::<f32>().unwrap();
                        let second = second_without_state.elements::<f32>().unwrap();
                        first.into_iter().zip(second).flat_map(|(first, second)| [first, second]).collect::<Vec<_>>()
                    }).unwrap())],
                },
                {
                    inputs = [(@replicated, first_input.clone())],
                    outputs = [(@replicated, first_without_state.clone())],
                },
            ],
        );
    }

    #[test]
    fn test_dropout_batching_broadcast_axes() {
        // A batch axis inserted before a broadcast axis shifts it, and each item still shares its mask along it.
        let state = threefry_state(42, 7);
        let item_type = ArrayType::new_static(DataType::F32, [3, 4]);
        let first =
            Array::from_elements(item_type.clone(), &(1..=12).map(|value| value as f32).collect::<Vec<_>>()).unwrap();
        let second =
            Array::from_elements(item_type, &(1..=12).map(|value| -(value as f32)).collect::<Vec<_>>()).unwrap();
        let unbatched = |input: &Array| input.dropout_with_broadcast_axes(0.5, [0], Some(&state)).unwrap();
        let (first_output, advanced_state) = unbatched(&first);
        let (second_output, _) = unbatched(&second);
        let pack = |first: &Array, second: &Array| {
            let first = first.elements::<f32>().unwrap();
            let second = second.elements::<f32>().unwrap();
            let elements = (0..3)
                .flat_map(|row| [&first[4 * row..4 * row + 4], &second[4 * row..4 * row + 4]].concat())
                .collect::<Vec<_>>();
            Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2, 4]), &elements).unwrap()
        };
        check_operation_batching!(
            @exact,
            operation = DropoutOperation::new(0.5, [0]),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 1), pack(&first, &second)), (@replicated, state.clone())],
                outputs = [
                    (@mapped(axis = 1), pack(&first_output, &second_output)),
                    (@replicated, advanced_state.unwrap()),
                ],
            }],
        );
    }

    #[test]
    fn test_dropout_batching_staging() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [3]));
        let state = builder.add_input(RandomAlgorithm::ThreeFry.state_type());
        let outputs = builder
            .add_instruction(DropoutOperation::new(0.5, Axes::default()), Vec::new(), vec![input, state], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // A replicated state shares the mask along the batch axis, which becomes an additional broadcast axis.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(1), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(1), BatchAxis::replicated()]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3, 2], %1:u64[2] .
                let %2:f32[3, 2], %3:u64[2] = dropout [rate=0.5, broadcast_axes=[1]] %0 %1
                in (%2, %3)
            "}
            .trim_end(),
        );

        // A mapped state stages one carry-free scan over the per-item inputs and states.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0), BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f32[3], %1:u64[2, 2] .
                let %2:f32[2, 3] = broadcast [output_type=f32[2, 3], output_axes=[1]] %0
                    %3:f32[2, 3], %4:u64[2, 2] = scan [carry_count=0, iteration_count=2, reverse=false] %2 %1 [
                        body={
                            lambda %0:i64[], %1:f32[3], %2:u64[2] .
                            let %3:f32[3], %4:u64[2] = dropout [rate=0.5, broadcast_axes=[]] %1 %2
                            in (%3, %4)
                        },
                    ]
                in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dropout_differentiation() {
        // The tangent applies the mask of the primal state to the input tangent, and the states have zero tangents.
        let state = threefry_state(42, 7);
        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tangent = Array::vector(vec![6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
        let operation = DropoutOperation::new(0.5, Axes::default());
        let outputs = operation
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[
                    DifferentiationDual::new(input.clone(), MaybeZero::Value(tangent.clone())).unwrap(),
                    DifferentiationDual::new_with_zero_tangent(state.clone()).unwrap(),
                ],
            )
            .unwrap();
        let (output, advanced_state) = input.dropout(0.5, Some(&state)).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].primal(), &output);
        assert_eq!(outputs[0].tangent(), &MaybeZero::Value(tangent.dropout(0.5, Some(&state)).unwrap().0));
        assert_eq!(outputs[1].primal(), &advanced_state.unwrap());
        assert!(outputs[1].tangent().is_zero());

        // Without a state, the tangent applies the same fixed mask as the primal.
        let outputs = DropoutOperation::new(0.5, Axes::default())
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new(input.clone(), MaybeZero::Value(tangent.clone())).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &input.dropout(0.5, None).unwrap().0);
        assert_eq!(outputs[0].tangent(), &MaybeZero::Value(tangent.dropout(0.5, None).unwrap().0));

        // A rate of one drops every tangent element too.
        let outputs = DropoutOperation::new(1.0, Axes::default())
            .jvp(
                &DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new(input.clone(), MaybeZero::Value(tangent.clone())).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs[0].tangent(), &MaybeZero::Value(Array::vector(vec![0.0f32; 6]).unwrap()));

        // Reverse mode applies the same mask to the output cotangent.
        let (value, pullback) = differentiate_at((input.clone(), state.clone()))
            .vjp(|(input, state)| Ok(input.dropout(0.5, Some(&state))?.0))
            .unwrap();
        let cotangent = Array::vector(vec![1.0f32; 6]).unwrap();
        let (input_cotangent, state_cotangent) = pullback.apply(cotangent.clone()).unwrap();
        assert_eq!(value, output);
        assert_eq!(input_cotangent, MaybeZero::Value(cotangent.dropout(0.5, Some(&state)).unwrap().0));
        assert!(state_cotangent.is_zero());

        // Staged forward mode applies the same operation to the tangent with the primal state.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F32, [6]));
        let state = builder.add_input(RandomAlgorithm::ThreeFry.state_type());
        let outputs = builder.add_instruction(operation, Vec::new(), vec![input, state], None).unwrap().to_vec();
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.jvp().unwrap().to_string(),
            indoc! {"
                lambda %0:f32[6], %1:u64[2], %2:f32[6] .
                let %3:f32[6], %4:u64[2] = dropout [rate=0.5, broadcast_axes=[]] %0 %1
                    %5:f32[6], %6:u64[2] = dropout [rate=0.5, broadcast_axes=[]] %2 %1
                in (%3, %4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dropout_transposition() {
        let cotangent = Array::vector(vec![1.0f64, 2.0, 3.0, 4.0]).unwrap();
        check_operation_transposition!(
            @exact,
            operation = DropoutOperation::new(0.5, Axes::default()),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F64, [4])))],
                output_cotangents = [cotangent.clone()],
                input_cotangents = [cotangent.dropout(0.5, None).unwrap().0],
            }],
        );

        // A random state must be known, since it determines the mask of the transposed map.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input_type = ArrayType::new_static(DataType::F64, [4]);
        let inputs =
            [PartialValue::Unknown(input_type.clone()), PartialValue::Unknown(RandomAlgorithm::ThreeFry.state_type())];
        let mut rule_context = TranspositionContext::new(context.clone());
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(matches!(
            DropoutOperation::new(0.5, Axes::default()).transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[
                    MaybeZero::Value(context.input(input_type.cotangent().unwrap())),
                    MaybeZero::Zero(ArrayType::new_static(DataType::Zero, [2])),
                ],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "`dropout` transpose requires a known random state",
        ));
    }

    #[test]
    fn test_threefry_2x32() {
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
    fn test_philox_4x32() {
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
