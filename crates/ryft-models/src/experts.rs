//! Projections of routed experts, which are the contractions that hold almost all parameters of a mixture of experts
//! model and therefore decide its memory use and speed.
//!
//! A layer with routed experts stacks one projection matrix per expert. This module abstracts how activations are
//! contracted with such a stack behind the [`ExpertProjector`] trait, so that a model states which contraction it
//! needs and the caller chooses the numerics. A model calls [`ExpertProjector::project_shared`] when all experts
//! read the same input (e.g., the first projection of an expert, which reads the layer input) and
//! [`ExpertProjector::project_grouped`] when every expert reads its own input (e.g., the second projection of an
//! expert, which reads the activation of that expert).
//!
//! Two projectors are provided.
//!
//!   - [`FloatingExpertProjector`] contracts floating-point values at one data type. It is the numerics of floating
//!     reference implementations and has ordinary gradients.
//!   - [`BlockScaledExpertProjector`] contracts block-scaled values in the formats of the Inkling 2 releases and
//!     implements the mixed precision training recipe that those models were trained with. Refer to its
//!     documentation for the forward and backward computations.
//!
//! Both are dense. They evaluate every expert for every token and rely on the model to zero the contributions of the
//! experts that a token is not routed to. A projector that gathers the routed tokens of every expert is a further
//! implementation of the same trait.

use std::cell::RefCell;

use ryft_core::{
    Abs, Add, ArrayMemberType, ArrayOperations, ArrayType, BlockQuantize, BlockScaleFormat, BlockScaleRule,
    BlockScaledDot, BlockScaledDotConfiguration, BlockScaledDotResidualPolicy, BlockScaledDotResiduals, Broadcast,
    Clamp, Compare, CompositeCallContext, CompositeVjp, ConvertElementType, CustomRuleDefinition, DataType, Div,
    Domain, DomainTracer, DomainTracingContext, Dot, DotDimensionNumbers, Fill, Floor, Iota, ManualVariationAlignment,
    Mul, ProgramError, Random, Reduce, ReductionKind, Rem, Reshape, ScaledDotOperation, Select, StagingContext, Sub,
    SymbolicZeroVjp, Transpose, Value, block_scaled_dot,
};

use crate::layers::static_shape;

// TODO(eaplatanios): Review this module.

/// Contraction of activations with the stacked projection matrices of routed experts. Implementations choose the
/// numerical precision of the contraction and of its gradients. Refer to the [module documentation](self) for the
/// provided implementations.
pub trait ExpertProjector<V> {
    /// Contracts one input per token with every expert, computing `y[m, e, o] = ∑ₖ x[m, k] · w[e, o, k]`.
    ///
    /// # Parameters
    ///
    ///   - `layer`: Index of the physical layer that owns `weights`, which implementations may use to select a
    ///     format.
    ///   - `input`: Activations of shape `[tokens, inputs]`.
    ///   - `weights`: Stacked projections of shape `[experts, outputs, inputs]`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the shapes do not match or if the implementation cannot contract them.
    fn project_shared(&self, layer: usize, input: &V, weights: &V) -> Result<V, ProgramError>;

    /// Contracts one input per token and expert with that expert, computing
    /// `y[m, e, o] = ∑ₖ x[m, e, k] · w[e, o, k]`.
    ///
    /// # Parameters
    ///
    ///   - `layer`: Index of the physical layer that owns `weights`, which implementations may use to select a
    ///     format.
    ///   - `input`: Activations of shape `[tokens, experts, inputs]`.
    ///   - `weights`: Stacked projections of shape `[experts, outputs, inputs]`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the shapes do not match or if the implementation cannot contract them.
    fn project_grouped(&self, layer: usize, input: &V, weights: &V) -> Result<V, ProgramError>;
}

/// [`ExpertProjector`] that converts activations and weights to one floating-point data type and contracts them at
/// that data type, which is also the data type of its results.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FloatingExpertProjector {
    /// Data type at which the contractions are computed.
    compute_type: DataType,
}

impl FloatingExpertProjector {
    /// Creates a new [`FloatingExpertProjector`] that contracts at `compute_type`.
    #[inline]
    pub fn new(compute_type: DataType) -> Self {
        Self { compute_type }
    }

    /// Returns the data type at which the contractions are computed.
    #[inline]
    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }
}

impl<V: ArrayOperations> ExpertProjector<V> for FloatingExpertProjector {
    fn project_shared(&self, layer: usize, input: &V, weights: &V) -> Result<V, ProgramError> {
        let _ = layer;
        input.convert_element_type(self.compute_type)?.dot(
            &weights.convert_element_type(self.compute_type)?,
            &DotDimensionNumbers::new(vec![1], vec![2], Vec::new(), Vec::new()),
        )
    }

    fn project_grouped(&self, layer: usize, input: &V, weights: &V) -> Result<V, ProgramError> {
        // The contraction places its batch dimension (i.e., the experts) first, so tokens and experts are swapped
        // back afterwards.
        let _ = layer;
        input
            .convert_element_type(self.compute_type)?
            .dot(
                &weights.convert_element_type(self.compute_type)?,
                &DotDimensionNumbers::new(vec![2], vec![2], vec![1], vec![0]),
            )?
            .transpose([1, 0, 2])
    }
}

/// Block-scaled format of a projection of a [`BlockScaledExpertProjector`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlockScaledFormat {
    /// NVFP4 for both inputs of the forward contraction. Weights are quantized per expert, with the block scale
    /// that maps the largest magnitude of a block to the largest element magnitude, and activations per token, with
    /// the four-over-six rule.
    Nvfp4,

    /// MXFP8 for both inputs of the forward contraction, with block scales that are rounded up to a power of two.
    Mxfp8,
}

impl BlockScaledFormat {
    /// Returns the [`BlockScaledDotConfiguration`] of the mixed precision training recipe for projections in this
    /// format that contract with `dimensions`. Refer to [`BlockScaledExpertProjector`] for the recipe.
    fn configuration(self, dimensions: DotDimensionNumbers) -> BlockScaledDotConfiguration {
        let mxfp8 = BlockScaleFormat::mxfp8().with_scale_rule(BlockScaleRule::RoundUp);
        let configuration = match self {
            Self::Nvfp4 => BlockScaledDotConfiguration::new(dimensions, BlockScaleFormat::nvfp4())
                .with_lhs_format(BlockScaleFormat::nvfp4().with_scale_rule(BlockScaleRule::FourOverSix)),
            Self::Mxfp8 => BlockScaledDotConfiguration::new(dimensions, mxfp8),
        };
        configuration
            .with_gradient_format(mxfp8)
            .with_quantized_backward(true)
            .with_residual_policy(BlockScaledDotResidualPolicy::Quantized)
            .with_stochastic_gradient_rounding(true)
    }
}

/// Tracers of the inputs of a block-scaled projection, namely the activations, the weights, and the optional global
/// scales and random generator state of [`BlockScaledDot::call`].
type ProjectionInputs<D> = (DomainTracer<D>, DomainTracer<D>, [Vec<DomainTracer<D>>; 4]);

/// Retained block-scaled projection functions for a backend [`Domain`] `D`, one for every combination of
/// [`BlockScaledFormat`] and input layout. The functions own their differentiation rules, so they are created once
/// (e.g., before a training step is compiled) and shared by every [`BlockScaledExpertProjector`] of that domain.
pub struct BlockScaledProjections<D: Domain<Type = ArrayType>>
where
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Projections of shared inputs, for NVFP4 and for MXFP8.
    shared: [BlockScaledDot<D>; 2],

    /// Projections of grouped inputs, for NVFP4 and for MXFP8.
    grouped: [BlockScaledDot<D>; 2],
}

impl<D: Domain<Type = ArrayType>> BlockScaledProjections<D>
where
    D::Constant: 'static,
    D::Operation: 'static + From<ScaledDotOperation>,
    DomainTracer<D>: Abs
        + Add
        + Broadcast
        + BlockQuantize
        + Clamp
        + Compare
        + ConvertElementType
        + Div
        + Dot
        + Floor
        + Mul
        + ManualVariationAlignment<ArrayType>
        + Random
        + Reduce
        + Rem
        + Reshape
        + Select
        + Sub
        + Transpose,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>
        + Fill<f64, DomainTracer<D>>
        + Iota<DomainTracer<D>>,
{
    /// Creates the block-scaled projection functions of domain `D`.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if a projection function cannot be created.
    pub fn new() -> Result<Self, ProgramError> {
        // Shared inputs `[tokens, inputs]` contract with `[experts, outputs, inputs]` into
        // `[tokens, experts, outputs]`. Grouped inputs `[tokens, experts, inputs]` pair their experts with those of
        // the weights as a batching dimension, which the contraction places first.
        let shared = || DotDimensionNumbers::new(vec![1], vec![2], Vec::new(), Vec::new());
        let grouped = || DotDimensionNumbers::new(vec![2], vec![2], vec![1], vec![0]);
        Ok(Self {
            shared: [
                block_scaled_dot(BlockScaledFormat::Nvfp4.configuration(shared()))?,
                block_scaled_dot(BlockScaledFormat::Mxfp8.configuration(shared()))?,
            ],
            grouped: [
                block_scaled_dot(BlockScaledFormat::Nvfp4.configuration(grouped()))?,
                block_scaled_dot(BlockScaledFormat::Mxfp8.configuration(grouped()))?,
            ],
        })
    }
}

/// [`ExpertProjector`] that contracts block-scaled values and implements the mixed precision training recipe of the
/// Inkling 2 models, which is described in the Thinking Machines article
/// ["End-to-End 4-Bit Training"](https://thinkingmachines.ai/blog/end-to-end-4-bit-training/). Every projection is a
/// [`block_scaled_dot`] of `ryft-core`, so the quantizers, the contractions of narrow values, and the backward rule
/// are those of [`BlockQuantize`], [`ScaledDot`](ryft_core::ScaledDot), and [`BlockScaledDot`]. Backends lower the
/// contractions to native block-scaled matrix products where they have them.
///
/// # Forward
///
/// Layers before the provided number of NVFP4 layers use [`BlockScaledFormat::Nvfp4`] and the remaining layers use
/// [`BlockScaledFormat::Mxfp8`]. Both inputs of a projection are quantized along the contracted dimension, the weights
/// from their floating-point master values on every call and the activations from the values that the model
/// computed, and the projection is the contraction of the quantized inputs, `y = x_q · w_qᵀ`, accumulated in `f32`.
///
/// NVFP4 weights have one global scale per expert, `amax / 2688` for the largest magnitude `amax` of the expert, and
/// block scales that map the largest magnitude of every block to the largest element. NVFP4 activations have one global
/// scale per row, `amax / 1536`, and block scales that follow the four-over-six rule. MXFP8 block scales are rounded up
/// to a power of two. These are the rules of the weights of the Inkling 2 Mini release and of its native reference
/// kernels, which the host codecs of [`ryft_checkpoints::codecs`] reproduce byte for byte. The quantizers here compute
/// the same rules with the arithmetic of `ryft-core`, which divides where the reference kernels multiply by
/// reciprocals, so individual values can round differently.
///
/// # Backward
///
/// The backward pass only uses the quantized forward inputs, which are the residuals that the forward pass keeps in
/// their narrow formats (see [`BlockScaledDotResidualPolicy::Quantized`]). For an output cotangent `g`, the gradients
/// are `dx = g · w_q` and `dw = gᵀ · x_q`. Each of the two contractions runs over a different dimension than the
/// forward contraction, so its inputs are regrouped along that dimension (i.e., along the outputs for `dx` and along
/// the tokens for `dw`) and quantized again to MXFP8 with stochastic rounding, as is the cotangent. Stochastic
/// rounding makes every such input equal to the quantized forward input in expectation. The gradients pass straight
/// through the forward quantizers to the unquantized inputs, and the scales of the forward quantizers are treated as
/// constants.
///
/// Quantizing along the tokens needs a number of tokens that is a multiple of 32, and quantizing along the outputs
/// needs a number of outputs that is a multiple of 32.
///
/// # Stochastic Rounding
///
/// A projector is created with one random generator state and derives an independent state for every projection
/// that it computes, in the order of the calls. A model that calls its projector in a fixed order therefore draws
/// reproducible noise that differs between layers, between the applications of a shared layer, and between the two
/// projections of a layer. A projector must be created with a fresh state for every training step.
pub struct BlockScaledExpertProjector<'o, D: Domain<Type = ArrayType>, V>
where
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Retained projection functions.
    projections: &'o BlockScaledProjections<D>,

    /// Number of leading layers whose projections use NVFP4.
    nvfp4_layer_count: usize,

    /// Data type of the results of the projections.
    output_type: DataType,

    /// Random generator state from which the state of the next projection is derived.
    random_state: RefCell<V>,
}

impl<'o, D: Domain<Type = ArrayType>, V> BlockScaledExpertProjector<'o, D, V>
where
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
{
    /// Creates a new [`BlockScaledExpertProjector`].
    ///
    /// # Parameters
    ///
    ///   - `projections`: Retained projection functions of the domain of the values that are projected.
    ///   - `nvfp4_layer_count`: Number of leading layers whose projections use NVFP4.
    ///   - `output_type`: Data type that the results of the projections are converted to.
    ///   - `random_state`: Random generator state (refer to [`Random`]) that seeds stochastic rounding.
    #[inline]
    pub fn new(
        projections: &'o BlockScaledProjections<D>,
        nvfp4_layer_count: usize,
        output_type: DataType,
        random_state: V,
    ) -> Self {
        Self { projections, nvfp4_layer_count, output_type, random_state: RefCell::new(random_state) }
    }

    /// Returns the [`BlockScaledFormat`] of the projections of `layer`.
    #[inline]
    pub fn format(&self, layer: usize) -> BlockScaledFormat {
        if layer < self.nvfp4_layer_count { BlockScaledFormat::Nvfp4 } else { BlockScaledFormat::Mxfp8 }
    }
}

impl<D, V, C> ExpertProjector<V> for BlockScaledExpertProjector<'_, D, V>
where
    D: Domain<Type = ArrayType>,
    D::Constant: 'static,
    D::Operation: 'static,
    DomainTracer<D>: 'static,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
    V: ArrayOperations<Type = ArrayType> + Value<Domain = C> + ManualVariationAlignment<ArrayType> + Random,
    C: CompositeCallContext<Type = ArrayType, Value = V, Constant = D::Constant, Operation = D::Operation>,
    CustomRuleDefinition<C::Constant, C::Operation>: CompositeVjp<
            SymbolicZeroVjp<
                D::Constant,
                D::Operation,
                ProjectionInputs<D>,
                DomainTracer<D>,
                BlockScaledDotResiduals<DomainTracer<D>>,
            >,
            DomainTracer<D>,
            ProjectionInputs<D>,
            DomainTracer<D>,
        >,
{
    fn project_shared(&self, layer: usize, input: &V, weights: &V) -> Result<V, ProgramError> {
        let projection = &self.projections.shared[self.format(layer) as usize];
        self.project(projection, layer, input, weights)
    }

    fn project_grouped(&self, layer: usize, input: &V, weights: &V) -> Result<V, ProgramError> {
        // The contraction places its batching dimension (i.e., the experts) first, so tokens and experts are swapped
        // back afterwards.
        let projection = &self.projections.grouped[self.format(layer) as usize];
        self.project(projection, layer, input, weights)?.transpose([1, 0, 2])
    }
}

impl<D, V, C> BlockScaledExpertProjector<'_, D, V>
where
    D: Domain<Type = ArrayType>,
    D::Constant: 'static,
    D::Operation: 'static,
    DomainTracer<D>: 'static,
    DomainTracingContext<D>: StagingContext<Type = ArrayType, Constant = D::Constant, Operation = D::Operation>,
    V: ArrayOperations<Type = ArrayType> + Value<Domain = C> + ManualVariationAlignment<ArrayType> + Random,
    C: CompositeCallContext<Type = ArrayType, Value = V, Constant = D::Constant, Operation = D::Operation>,
    CustomRuleDefinition<C::Constant, C::Operation>: CompositeVjp<
            SymbolicZeroVjp<
                D::Constant,
                D::Operation,
                ProjectionInputs<D>,
                DomainTracer<D>,
                BlockScaledDotResiduals<DomainTracer<D>>,
            >,
            DomainTracer<D>,
            ProjectionInputs<D>,
            DomainTracer<D>,
        >,
{
    /// Calls `projection` for `layer` with the global scales of its format and a random generator state that is
    /// derived from the state of this projector. The result has the layout of the contraction and the output data
    /// type of this projector.
    fn project(&self, projection: &BlockScaledDot<D>, layer: usize, input: &V, weights: &V) -> Result<V, ProgramError> {
        let (random_state, mut call_states) = self.random_state.borrow().split_rng_key(1)?;
        *self.random_state.borrow_mut() = random_state;
        let input = input.convert_element_type(DataType::F32)?;
        let weights = weights.convert_element_type(DataType::F32)?;

        // NVFP4 activations have one global scale per row (i.e., per token, and per expert for grouped inputs), and
        // NVFP4 weights have one per expert. MXFP8 has no global scales.
        let (input_scales, weights_scales) = match self.format(layer) {
            BlockScaledFormat::Nvfp4 => {
                let input_rank = static_shape(&input)?.len();
                (
                    Some(global_scales(&input, &[input_rank - 1], 1536.0)?),
                    Some(global_scales(&weights, &[1, 2], 2688.0)?),
                )
            }
            BlockScaledFormat::Mxfp8 => (None, None),
        };
        projection
            .call(input, weights, input_scales, weights_scales, None, Some(call_states.remove(0)))?
            .convert_element_type(self.output_type)
    }
}

/// Returns the grouped NVFP4 global scales of `values`, which have the extent one along `axes` and hold, for every
/// group of values that share their indices along the other axes, the largest magnitude of the group divided by
/// `denominator`. A group of zeros gets the smallest positive normal scale rather than zero, so that its values
/// divide to zero.
fn global_scales<V: ArrayOperations>(values: &V, axes: &[usize], denominator: f64) -> Result<V, ProgramError>
where
    V::Type: ArrayMemberType,
{
    let mut shape = static_shape(values)?;
    axes.iter().for_each(|&axis| shape[axis] = 1);
    let amax = values.abs()?.reduce(axes, ReductionKind::Max)?.reshape(shape)?;
    Div::div(&amax, &amax.fill_like(denominator)?)?.max(&amax.fill_like(f32::MIN_POSITIVE)?)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_checkpoints::codecs::{
        Nvfp4OuterScale, Nvfp4ScaleSelection, decode_mxfp8, decode_nvfp4, encode_mxfp8, encode_nvfp4,
    };
    use ryft_core::{Array, ArrayOperation, EagerContext, RandomAlgorithm, Typed, differentiate_at};

    use super::*;

    /// Domain of the eager reference backend.
    type TestDomain = EagerContext<Array, ArrayOperation<Array>>;

    /// Returns an `f32` array of the provided shape with reproducible pseudo-random values in `[-1, 1)`.
    fn random_array(shape: &[usize], seed: u64) -> Array {
        let mut state = seed;
        let values = (0..shape.iter().product::<usize>())
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                2.0 * ((state >> 40) as f32 / (1u32 << 24) as f32) - 1.0
            })
            .collect::<Vec<_>>();
        Array::from_elements(ArrayType::new_static(DataType::F32, shape.to_vec()), &values).unwrap()
    }

    /// Returns a random generator state for `seed`.
    fn random_state(seed: u64) -> Array {
        Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[seed, 0u64]).unwrap()
    }

    /// Returns the values that the host codecs give for the forward quantization of `values` in `format`, with one
    /// outer scale per `group_size` values for NVFP4 weights and one per trailing axis row for NVFP4 activations.
    fn host_quantized(
        format: BlockScaledFormat,
        values: &Array,
        row_size: usize,
        weights_group_size: Option<usize>,
    ) -> Vec<f32> {
        let values = values.elements::<f32>().unwrap();
        match (format, weights_group_size) {
            (BlockScaledFormat::Mxfp8, _) => {
                let (elements, exponents) = encode_mxfp8(&values).unwrap();
                decode_mxfp8(&elements, &exponents).unwrap()
            }
            (BlockScaledFormat::Nvfp4, Some(group_size)) => values
                .chunks_exact(group_size)
                .flat_map(|group| {
                    let amax = group.iter().fold(0.0f32, |amax, value| amax.max(f32::abs(*value)));
                    let outer_scale = Nvfp4OuterScale::for_weights(amax, 2688.0).unwrap();
                    let (packed, scales) = encode_nvfp4(group, outer_scale, Nvfp4ScaleSelection::Six).unwrap();
                    decode_nvfp4(&packed, &scales, outer_scale.scale()).unwrap()
                })
                .collect(),
            (BlockScaledFormat::Nvfp4, None) => values
                .chunks_exact(row_size)
                .flat_map(|row| {
                    let outer_scale = Nvfp4OuterScale::for_activations(row);
                    let (packed, scales) = encode_nvfp4(row, outer_scale, Nvfp4ScaleSelection::FourOverSix).unwrap();
                    decode_nvfp4(&packed, &scales, outer_scale.scale()).unwrap()
                })
                .collect(),
        }
    }

    /// Asserts that two lists of `f32` values agree within a relative `tolerance` of the largest expected magnitude.
    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        let magnitude = expected.iter().fold(0.0f32, |magnitude, value| magnitude.max(f32::abs(*value)));
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance * magnitude,
                "element {index}: {actual} is not close to {expected}",
            );
        }
    }

    #[test]
    fn test_floating_expert_projector_new() {
        assert_eq!(FloatingExpertProjector::new(DataType::BF16).compute_type(), DataType::BF16);
    }

    #[test]
    fn test_floating_expert_projector_project_shared() {
        // Two tokens, two experts with two outputs each, and three inputs.
        let input = Array::matrix(2, 3, vec![1.0f32, 2.0, 3.0, -1.0, 0.0, 1.0]).unwrap();
        let weights = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2, 3]),
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 2.0],
        )
        .unwrap();
        let output = FloatingExpertProjector::new(DataType::F32).project_shared(0, &input, &weights).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [2, 2, 2]));
        assert_eq!(output.elements::<f32>().unwrap(), vec![1.0, 2.0, 6.0, 6.0, -1.0, 0.0, 0.0, 2.0]);

        // The result has the compute data type of the projector.
        let narrow = FloatingExpertProjector::new(DataType::BF16).project_shared(0, &input, &weights).unwrap();
        assert_eq!(narrow.r#type().data_type(), DataType::BF16);
        assert_eq!(narrow.to_f64s(), vec![1.0, 2.0, 6.0, 6.0, -1.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_floating_expert_projector_project_grouped() {
        // One token, two experts with one output each, and two inputs per expert.
        let input =
            Array::from_elements(ArrayType::new_static(DataType::F32, [1, 2, 2]), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let weights =
            Array::from_elements(ArrayType::new_static(DataType::F32, [2, 1, 2]), &[1.0f32, 1.0, 2.0, -1.0]).unwrap();
        let output = FloatingExpertProjector::new(DataType::F32).project_grouped(0, &input, &weights).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [1, 2, 1]));
        assert_eq!(output.elements::<f32>().unwrap(), vec![3.0, 2.0]);
    }

    #[test]
    fn test_block_scaled_expert_projector_new() {
        let projections = BlockScaledProjections::<TestDomain>::new().unwrap();
        let projector = BlockScaledExpertProjector::new(&projections, 2, DataType::F32, random_state(1));
        assert_eq!(projector.format(0), BlockScaledFormat::Nvfp4);
        assert_eq!(projector.format(1), BlockScaledFormat::Nvfp4);
        assert_eq!(projector.format(2), BlockScaledFormat::Mxfp8);
    }

    #[test]
    fn test_block_scaled_expert_projector_project_shared() {
        // The forward projection contracts exactly the values that the host codecs produce for both operands, for
        // an NVFP4 layer (layer 0) and for an MXFP8 layer (layer 1).
        let projections = BlockScaledProjections::<TestDomain>::new().unwrap();
        let input = random_array(&[32, 32], 1);
        let weights = random_array(&[2, 32, 32], 2);
        for (layer, format) in [(0, BlockScaledFormat::Nvfp4), (1, BlockScaledFormat::Mxfp8)] {
            let projector = BlockScaledExpertProjector::new(&projections, 1, DataType::F32, random_state(3));
            let output = projector.project_shared(layer, &input, &weights).unwrap();
            assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [32, 2, 32]));
            let quantized_input = host_quantized(format, &input, 32, None);
            let quantized_weights = host_quantized(format, &weights, 32, Some(32 * 32));
            let mut expected = vec![0.0f32; 32 * 2 * 32];
            for token in 0..32 {
                for expert in 0..2 {
                    for output_index in 0..32 {
                        expected[(token * 2 + expert) * 32 + output_index] = (0..32)
                            .map(|index| {
                                quantized_input[token * 32 + index]
                                    * quantized_weights[(expert * 32 + output_index) * 32 + index]
                            })
                            .sum();
                    }
                }
            }
            assert_close(&output.elements::<f32>().unwrap(), &expected, 1e-5);
        }
    }

    #[test]
    fn test_block_scaled_expert_projector_project_grouped() {
        let projections = BlockScaledProjections::<TestDomain>::new().unwrap();
        let input = random_array(&[32, 2, 32], 4);
        let weights = random_array(&[2, 32, 32], 5);
        let projector = BlockScaledExpertProjector::new(&projections, 0, DataType::BF16, random_state(6));
        let output = projector.project_grouped(0, &input, &weights).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::BF16, [32, 2, 32]));

        // Quantization changes the projection only slightly, and it does change it.
        let floating = FloatingExpertProjector::new(DataType::F32).project_grouped(0, &input, &weights).unwrap();
        let output = output.to_f64s().into_iter().map(|value| value as f32).collect::<Vec<_>>();
        let floating = floating.elements::<f32>().unwrap();
        assert_ne!(output, floating);
        assert_close(&output, &floating, 0.05);
    }

    #[test]
    fn test_block_scaled_expert_projector_differentiation() {
        // The gradients of the sum of a projection are close to the gradients through the quantized forward inputs
        // (i.e., the sums over the other input), because the backward pass quantizes them again to MXFP8 without
        // bias. Every gradient value sums up to 64 stochastically rounded values, whose rounding noise the tolerance
        // covers.
        let projections = BlockScaledProjections::<TestDomain>::new().unwrap();
        let input = random_array(&[32, 32], 7);
        let weights = random_array(&[2, 32, 32], 8);
        let gradient = |seed: u64| {
            differentiate_at((input.clone(), weights.clone()))
                .with_captures(random_state(seed))
                .gradient(|(input, weights), random_state| {
                    let projector = BlockScaledExpertProjector::new(&projections, 1, DataType::F32, random_state);
                    projector.project_shared(0, &input, &weights)?.reduce(&[0, 1, 2], ReductionKind::Sum)
                })
                .unwrap()
        };
        let (input_gradient, weights_gradient) = gradient(9);
        assert_eq!(input_gradient.r#type(), input.r#type());
        assert_eq!(weights_gradient.r#type(), weights.r#type());

        let quantized_input = host_quantized(BlockScaledFormat::Nvfp4, &input, 32, None);
        let quantized_weights = host_quantized(BlockScaledFormat::Nvfp4, &weights, 32, Some(32 * 32));
        let expected_input_gradient = (0..32 * 32)
            .map(|index| (0..2 * 32).map(|row| quantized_weights[row * 32 + index % 32]).sum::<f32>())
            .collect::<Vec<_>>();
        let expected_weights_gradient = (0..2 * 32 * 32)
            .map(|index| (0..32).map(|token| quantized_input[token * 32 + index % 32]).sum::<f32>())
            .collect::<Vec<_>>();
        assert_close(&input_gradient.elements::<f32>().unwrap(), &expected_input_gradient, 0.1);
        assert_close(&weights_gradient.elements::<f32>().unwrap(), &expected_weights_gradient, 0.1);

        // Stochastic rounding is reproducible for one generator state and differs between states.
        assert_eq!(gradient(9), (input_gradient.clone(), weights_gradient.clone()));
        assert_ne!(gradient(10), (input_gradient, weights_gradient));
    }

    #[test]
    fn test_block_scaled_expert_projector_differentiation_unaligned_tokens() {
        let projections = BlockScaledProjections::<TestDomain>::new().unwrap();
        let input = random_array(&[16, 32], 7);
        let weights = random_array(&[2, 32, 32], 8);
        let gradient = differentiate_at((input, weights)).with_captures(random_state(1)).gradient(
            |(input, weights), random_state| {
                let projector = BlockScaledExpertProjector::new(&projections, 1, DataType::F32, random_state);
                projector.project_shared(0, &input, &weights)?.reduce(&[0, 1, 2], ReductionKind::Sum)
            },
        );
        assert!(gradient.is_err_and(|error| {
            error.to_string().contains("left free extent 16 is not a positive multiple of block size 32")
        }));
    }
}
