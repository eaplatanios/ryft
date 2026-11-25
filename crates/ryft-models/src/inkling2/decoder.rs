//! Text decoder of the Inkling 2 model family.
//!
//! The decoder maps token identifiers to next token logits. It embeds the tokens, applies its layers according to
//! [`Inkling2Configuration::layer_schedule`], reads a learned mixture of the resulting block states, and projects that
//! onto the vocabulary. Three properties distinguish it from an ordinary transformer decoder.
//!
//!   - **Attention Residuals:** Layers are grouped into blocks of [`Inkling2Configuration::block_layer_count`]. The
//!     residual stream restarts at the first layer of every block, and every layer input is a learned mixture (refer
//!     to [`residual_read`]) of the states of the completed blocks and of the running state of the current block.
//!   - **Fixed Loop:** The core layers are applied [`Inkling2LoopConfiguration::pass_count`] times with shared
//!     parameters. Every pass restarts from the state before the core, and every pass after the first one also reads
//!     the block states that the previous pass produced. Those messages enter the inputs of a core layer through
//!     [`Inkling2LoopGate`]s. A shared parameter therefore receives the sum of the gradients of all of its
//!     applications, which reverse mode differentiation produces without any special handling because the same leaf
//!     is used in every application.
//!   - **Hybrid Attention With a Sink:** Local layers attend over a sliding window and global layers over the whole
//!     document, both with grouped queries, a query conditioned relative position term, and a learned sink that
//!     takes part in the normalization of the attention weights without contributing a value.
//!
//! The structures of this module are parameter structures (i.e., they implement
//! [`Parameterized`]), and their functions are written over the [`ArrayOperations`]
//! capability bundle, so one definition serves eager evaluation, differentiation, and compiled training steps.
//! [`Inkling2Decoder::specification`] is the single description of the parameters. It names the checkpoint array and
//! states the type of every leaf, and importers and initializers map over it.
//!
//! This is the dense reference implementation. Attention materializes scores for all pairs of positions, and every
//! token evaluates every expert with a routing weight of zero for the experts that it is not routed to, so memory
//! and time grow with the square of the sequence length and with the number of experts. It follows the pinned
//! floating reference implementation (i.e., `tests/unit_tests/models/inkling2/reference.py` of NVIDIA NeMo AutoModel
//! at commit `bdf1fff2e8e8ef3e4035308820495e593b7df089`) operation by operation, including the positions at which
//! values are rounded to [`Inkling2Configuration::compute_type`].
//!
//! [`Inkling2LoopConfiguration::pass_count`]: crate::inkling2::Inkling2LoopConfiguration::pass_count

use ryft_checkpoints::ArraySpecification;
use ryft_core::{
    Add, And, Array, ArrayMemberType, ArrayOperations, ArrayType, DataType, Div, GatherMode, Mul, Parameter,
    Parameterized, ProgramError, ReductionKind, Sub, TopK,
};
use ryft_macros::Parameterized as ParameterizedDerive;

use crate::experts::ExpertProjector;
use crate::inkling2::configuration::Inkling2Configuration;
use crate::layers::{clip_norm, delay, linear, residual_read, rms_norm, scale, softcap, static_shape};

// TODO(eaplatanios): Review this module.

/// Inputs of the Inkling 2 decoder for a batch of token sequences. A sequence may pack several documents one after
/// another, and it may end with padding.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ParameterizedDerive)]
#[ryft(crate = "ryft_core")]
pub struct Inkling2Inputs<P: Parameter> {
    /// Token identifiers, as an `i32` array of shape `[batch, sequence]`.
    tokens: P,

    /// Position of every token within its document, as an `i32` array of shape `[batch, sequence]`. Positions start
    /// at zero and increase by one within a document.
    positions: P,

    /// Identifier of the document of every token within its sequence, as an `i32` array of shape `[batch, sequence]`.
    /// Tokens attend and convolve only within their own document.
    segments: P,

    /// Whether every position holds a token rather than padding, as a Boolean array of shape `[batch, sequence]`.
    valid: P,
}

impl<P: Parameter> Inkling2Inputs<P> {
    /// Creates new [`Inkling2Inputs`]. Refer to the documentation of the fields of this structure, which share the
    /// names of the parameters, for the expected types.
    #[inline]
    pub fn new(tokens: P, positions: P, segments: P, valid: P) -> Self {
        Self { tokens, positions, segments, valid }
    }

    /// Returns the token identifiers.
    #[inline]
    pub fn tokens(&self) -> &P {
        &self.tokens
    }

    /// Returns the position of every token within its document.
    #[inline]
    pub fn positions(&self) -> &P {
        &self.positions
    }

    /// Returns the identifier of the document of every token within its sequence.
    #[inline]
    pub fn segments(&self) -> &P {
        &self.segments
    }

    /// Returns whether every position holds a token rather than padding.
    #[inline]
    pub fn valid(&self) -> &P {
        &self.valid
    }
}

/// Intermediate values of one evaluation of an [`Inkling2Decoder`], which tests compare against a reference
/// implementation to locate the first layer application at which two implementations diverge.
#[derive(Clone, Debug, PartialEq)]
pub struct Inkling2Diagnostics<V> {
    /// Running block state after every layer application, in the order of
    /// [`Inkling2Configuration::layer_schedule`]. Each state has shape `[batch, sequence, hidden]` and the compute
    /// data type.
    states: Vec<V>,

    /// Routing weight of every expert for every token after every layer application, in the same order. Each array
    /// has the `f32` data type and shape `[batch · sequence, experts]`, and the experts that a token is not routed to
    /// have weight zero.
    routing_weights: Vec<V>,
}

impl<V> Inkling2Diagnostics<V> {
    /// Returns the running block state after every layer application.
    #[inline]
    pub fn states(&self) -> &[V] {
        &self.states
    }

    /// Returns the routing weight of every expert for every token after every layer application.
    #[inline]
    pub fn routing_weights(&self) -> &[V] {
        &self.routing_weights
    }
}

/// Causal short convolution along the sequence axis, which adds a learned combination of the current and the
/// preceding positions of every channel to its input. With `w` taps, an input `x`, and tap coefficients `c`, position
/// `t` of the result is `(x[t] + ∑ₗ c[l] · x[t − l]) / z`, where the lag `l` ranges over `0..w` and positions before
/// the start of a document contribute zero.
///
/// A _static_ convolution has one learned coefficient per channel and lag and uses `z = sqrt(w + 1)`. A _generated_
/// convolution adds coefficients that a learned projection of another input generates per token, lag, and group of
/// channels (i.e., per attention head) to its static coefficients, and it uses `z = sqrt(2w + 1)`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ParameterizedDerive)]
#[ryft(crate = "ryft_core")]
pub struct Inkling2Convolution<P: Parameter> {
    /// Static tap coefficients. A static convolution stores them with shape `[channels, 1, width]` and with the taps
    /// in the order of decreasing lag. A generated convolution stores them with shape `[width, channels]` and with
    /// the taps in the order of increasing lag.
    taps: P,

    /// Projection of shape `[groups · width, hidden]` that generates the additional coefficients of a generated
    /// convolution, or [`None`] for a static convolution.
    generator: Option<P>,
}

impl Inkling2Convolution<ArraySpecification> {
    /// Returns the specification of a static convolution over `channels` channels whose taps are stored in the
    /// array named `name`.
    fn static_specification(configuration: &Inkling2Configuration, name: String, channels: usize) -> Self {
        let taps = ArrayType::new_static(DataType::F32, [channels, 1, configuration.convolution_width()]);
        Self { taps: ArraySpecification::new(name, taps), generator: None }
    }

    /// Returns the specification of a generated convolution over `channels` channels in `groups` groups, whose
    /// arrays are named with the prefix `prefix`.
    fn generated_specification(
        configuration: &Inkling2Configuration,
        prefix: &str,
        channels: usize,
        groups: usize,
    ) -> Self {
        let width = configuration.convolution_width();
        let taps = ArrayType::new_static(DataType::F32, [width, channels]);
        let generator = ArrayType::new_static(DataType::F32, [groups * width, configuration.hidden_size()]);
        Self {
            taps: ArraySpecification::new(format!("{prefix}.static_w"), taps),
            generator: Some(ArraySpecification::new(format!("{prefix}.weight_proj"), generator)),
        }
    }
}

impl<V: ArrayOperations> Inkling2Convolution<V>
where
    V::Type: ArrayMemberType,
{
    /// Applies this convolution to `input`, which has shape `[batch, sequence, channels]`, and returns an array of
    /// the same shape and data type. The sums are accumulated in `f32`.
    ///
    /// # Parameters
    ///
    ///   - `configuration`: Configuration of the decoder.
    ///   - `masks`: Masks of the inputs of the decoder.
    ///   - `input`: Array to convolve.
    ///   - `generator_input`: Array of shape `[batch, sequence, hidden]` from which a generated convolution generates
    ///     its additional coefficients, with one group of channels per `groups`. It is ignored by a static
    ///     convolution.
    ///   - `groups`: Number of channel groups of a generated convolution.
    fn forward(
        &self,
        configuration: &Inkling2Configuration,
        masks: &Inkling2Masks<V>,
        input: &V,
        generator_input: Option<&V>,
        groups: usize,
    ) -> Result<V, ProgramError> {
        let compute_type = configuration.compute_type();
        let width = configuration.convolution_width();
        let shape = static_shape(input)?;
        let (batch, length, channels) = (shape[0], shape[1], shape[2]);
        let input_type = input.r#type().as_array_type()?.data_type();
        let input = Mul::mul(&input.convert_element_type(DataType::F32)?, &masks.valid)?;
        let taps = self.taps.convert_element_type(compute_type)?.convert_element_type(DataType::F32)?;

        // The coefficients that a generated convolution adds have shape `[batch, sequence, groups, width]`, and every
        // group of channels shares the coefficients of its group.
        let generated = match (&self.generator, generator_input) {
            (Some(generator), Some(generator_input)) => Some(
                linear(generator_input, generator, compute_type, compute_type)?
                    .convert_element_type(DataType::F32)?
                    .reshape([batch, length, groups, width])?,
            ),
            _ => None,
        };

        let mut output = input.clone();
        for lag in 0..width {
            let coefficients = match &generated {
                None => taps.slice_axis(2, width - 1 - lag, width - lag, 1)?.reshape([1, 1, channels])?,
                Some(generated) => {
                    let generated = generated
                        .slice_axis(3, lag, lag + 1, 1)?
                        .broadcast_to([batch, length, groups, channels / groups])?
                        .reshape([batch, length, channels])?;
                    Add::add(&generated, &taps.slice_axis(0, lag, lag + 1, 1)?.reshape([1, 1, channels])?)?
                }
            };
            let delayed = match lag {
                0 => input.clone(),
                lag => Mul::mul(&delay(&input, lag)?, &masks.delays[lag - 1])?,
            };
            output = Add::add(&output, &Mul::mul(&delayed, &coefficients)?)?;
        }

        let normalizer = if generated.is_some() { 2 * width + 1 } else { width + 1 };
        let output = scale(&output, 1.0 / (normalizer as f64).sqrt())?.convert_element_type(input_type)?;
        Mul::mul(&output, &masks.valid.convert_element_type(input_type)?)
    }
}

/// Hybrid grouped query attention of one Inkling 2 layer, which attends from every position over the earlier positions
/// of its document.
///
/// Queries, relative position features, the output gate, and the sink logit are projected from the query input of the
/// layer, keys from its key input, and values from its value input. Keys and values pass through their short
/// convolutions. Query and key head vectors are clipped to an L2 length of
/// [`Inkling2Configuration::query_key_clip_threshold`], relative position features to
/// [`Inkling2Configuration::relative_clip_threshold`], and value head vectors are normalized by their root mean square.
/// A global layer additionally multiplies its queries and relative position features by
/// `1 + alpha · ln(max(1, (position + 1) / floor))`.
///
/// The score of query head `h` at position `t` for the key at position `s` is
/// `⟨q, k⟩ / sqrt(head_size) + ⟨r, table[:, d]⟩`, where `d` is the backward distance between the two positions and the
/// second term is present only for `d` below [`Inkling2Configuration::relative_extent`]. A local layer attends over
/// the keys with `d` below
/// [`Inkling2Configuration::local_window_size`] and a global layer over all earlier keys of the document. The attention
/// weights are the softmax over the visible keys together with the sink logit of the query head, so they sum to less
/// than one, and the weighted sum of the values is multiplied by the sigmoid of the output gate before the output
/// projection. Scores and sums are accumulated in `f32`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ParameterizedDerive)]
#[ryft(crate = "ryft_core")]
pub struct Inkling2Attention<P: Parameter> {
    /// Query projection of shape `[heads · head_size, hidden]`.
    query: P,

    /// Key projection of shape `[key_value_heads · head_size, hidden]`.
    key: P,

    /// Value projection of shape `[key_value_heads · head_size, hidden]`.
    value: P,

    /// Projection of shape `[heads · relative_size, hidden]` that produces the relative position features of every
    /// query head.
    relative: P,

    /// Projection of shape `[heads · head_size, hidden]` that produces the gate of the attention output.
    gate: P,

    /// Projection of shape `[heads, hidden]` that produces the sink logit of every query head.
    sink: P,

    /// Output projection of shape `[hidden, heads · head_size]`.
    output: P,

    /// Gain of shape `[heads, head_size]` that multiplies the query head vectors before their length is clipped.
    query_gain: P,

    /// Gain of shape `[heads, relative_size]` that multiplies the relative position features before their length is
    /// clipped.
    relative_gain: P,

    /// Relative position table of shape `[relative_size, relative_extent]`, whose column `d` holds the features of
    /// keys at backward distance `d`.
    relative_table: P,

    /// Short convolution of the projected keys.
    key_convolution: Inkling2Convolution<P>,

    /// Short convolution of the projected values.
    value_convolution: Inkling2Convolution<P>,
}

impl Inkling2Attention<ArraySpecification> {
    /// Returns the specification of the attention of `layer`.
    fn specification(configuration: &Inkling2Configuration, layer: usize) -> Self {
        let prefix = format!("model.llm.layers.{layer}.attn");
        let hidden = configuration.hidden_size();
        let heads = configuration.head_count();
        let key_value_heads = configuration.key_value_head_count();
        let head_size = configuration.head_size();
        let relative_size = configuration.relative_size();
        let array = |name: &str, shape: &[usize]| {
            ArraySpecification::new(format!("{prefix}.{name}"), ArrayType::new_static(DataType::F32, shape.to_vec()))
        };
        let convolution = |name: &str| {
            let channels = key_value_heads * head_size;
            if configuration.is_local_layer(layer) {
                Inkling2Convolution::static_specification(configuration, format!("{prefix}.{name}.weight"), channels)
            } else {
                Inkling2Convolution::generated_specification(
                    configuration,
                    &format!("{prefix}.{name}"),
                    channels,
                    key_value_heads,
                )
            }
        };
        Self {
            query: array("wq_du.weight", &[heads * head_size, hidden]),
            key: array("wk_dv.weight", &[key_value_heads * head_size, hidden]),
            value: array("wv_dv.weight", &[key_value_heads * head_size, hidden]),
            relative: array("wr_du.weight", &[heads * relative_size, hidden]),
            gate: array("gate_attn.weight", &[heads * head_size, hidden]),
            sink: array("attn_sink_proj.weight", &[heads, hidden]),
            output: array("wo_ud.weight", &[hidden, heads * head_size]),
            query_gain: array("q_l2_clip.scale", &[heads, head_size]),
            relative_gain: array("rel_l2_clip.scale", &[heads, relative_size]),
            relative_table: array("rel_logits_proj.proj", &[relative_size, configuration.relative_extent()]),
            key_convolution: convolution("k_sconv"),
            value_convolution: convolution("v_sconv"),
        }
    }
}

impl<V: ArrayOperations> Inkling2Attention<V>
where
    V::Type: ArrayMemberType,
{
    /// Attends from every position over the earlier positions of its document and returns the projected attention
    /// output with shape `[batch, sequence, hidden]` and the compute data type.
    fn forward(
        &self,
        configuration: &Inkling2Configuration,
        masks: &Inkling2Masks<V>,
        local: bool,
        query_input: &V,
        key_input: &V,
        value_input: &V,
    ) -> Result<V, ProgramError> {
        let compute_type = configuration.compute_type();
        let heads = configuration.head_count();
        let key_value_heads = configuration.key_value_head_count();
        let group_size = heads / key_value_heads;
        let head_size = configuration.head_size();
        let relative_size = configuration.relative_size();
        let shape = static_shape(query_input)?;
        let (batch, length) = (shape[0], shape[1]);
        let project = |input: &V, weight: &V| linear(input, weight, compute_type, compute_type);
        let rounded = |value: &V| value.convert_element_type(compute_type)?.convert_element_type(DataType::F32);

        let query = project(query_input, &self.query)?;
        let relative = project(query_input, &self.relative)?;
        let gate = project(query_input, &self.gate)?.convert_element_type(DataType::F32)?;
        let sink = project(query_input, &self.sink)?.convert_element_type(DataType::F32)?;
        let generator_input = |input| if local { None } else { Some(input) };
        let key = self.key_convolution.forward(
            configuration,
            masks,
            &project(key_input, &self.key)?,
            generator_input(key_input),
            key_value_heads,
        )?;
        let value = self.value_convolution.forward(
            configuration,
            masks,
            &project(value_input, &self.value)?,
            generator_input(value_input),
            key_value_heads,
        )?;

        let query = clip_norm(
            &query.reshape([batch, length, heads, head_size])?,
            Some(&self.query_gain.convert_element_type(compute_type)?),
            configuration.query_key_clip_threshold(),
        )?;
        let key = clip_norm(
            &key.reshape([batch, length, key_value_heads, head_size])?,
            None,
            configuration.query_key_clip_threshold(),
        )?;
        let relative = clip_norm(
            &relative.reshape([batch, length, heads, relative_size])?,
            Some(&self.relative_gain.convert_element_type(compute_type)?),
            configuration.relative_clip_threshold(),
        )?;
        let (query, relative) = if local {
            (rounded(&query)?, rounded(&relative)?)
        } else {
            (
                rounded(&Mul::mul(&rounded(&query)?, &masks.log_scale)?)?,
                rounded(&Mul::mul(&rounded(&relative)?, &masks.log_scale)?)?,
            )
        };
        let key = rounded(&key)?;
        let value = rounded(&rms_norm(
            &value.reshape([batch, length, key_value_heads, head_size])?,
            None,
            configuration.normalization_epsilon(),
        )?)?;

        // Scores have shape `[batch, key_value_heads, group, query, key]`, with the query heads of one key and value
        // head forming a group.
        let query = query.reshape([batch, length, key_value_heads, group_size, head_size])?;
        let scores = scale(&V::einsum("btkgd,bskd->bkgts", [&query, &key])?, 1.0 / (head_size as f64).sqrt())?;

        // The relative position term is rounded to the compute data type before it is added, and keys outside the
        // distance band of the table contribute no term.
        let table = rounded(&self.relative_table)?.slice_axis(1, 0, masks.table_extent, 1)?;
        let table = table.gather_axis(&masks.distances, 1, GatherMode::Clip)?;
        let relative_scores = rounded(&V::einsum("btnr,rbts->btns", [&relative, &table])?)?;
        let relative_scores = Mul::mul(&relative_scores, &masks.in_band)?.transpose([0, 2, 1, 3])?.reshape([
            batch,
            key_value_heads,
            group_size,
            length,
            length,
        ])?;
        let scores = Add::add(&scores, &relative_scores)?;

        // The sink is one additional key that is visible to every query and has a zero value.
        let sink = sink.reshape([batch, length, key_value_heads, group_size, 1])?.transpose([0, 2, 3, 1, 4])?;
        let visible = if local { &masks.local } else { &masks.global };
        let visible = visible.reshape([batch, 1, 1, length, length])?.broadcast_to([
            batch,
            key_value_heads,
            group_size,
            length,
            length,
        ])?;
        let sink_visible = visible.slice_axis(4, 0, 1, 1)?.fill_like(true)?;
        let weights = scores
            .concatenate_with([&sink], 4)?
            .masked_softmax(&visible.concatenate_with([&sink_visible], 4)?, 4)?
            .slice_axis(4, 0, length, 1)?;

        let context =
            V::einsum("bkgts,bskd->btkgd", [&weights, &value])?.reshape([batch, length, heads * head_size])?;
        let context = Mul::mul(&context, &gate.logistic()?)?;
        let output = project(&context, &self.output)?;
        Mul::mul(&output, &masks.valid.convert_element_type(compute_type)?)
    }
}

/// Routed experts of one Inkling 2 layer, which route every token to [`Inkling2Configuration::experts_per_token`]
/// experts and sum their outputs.
///
/// The routing score of an expert is the sigmoid of its router logit, computed in `f32`. The experts with the largest
/// sums of score and router bias are selected, with ties going to the expert with the smaller index, and the routing
/// weights of the selected experts are their scores (without the bias) normalized to sum to
/// [`Inkling2Configuration::route_scale`]. An expert computes its gate branch `g` and its up branch `u` with its first
/// projection and then applies its second projection to `weight · softcap(g) · sigmoid(g) · softcap(u)`, where the
/// product is computed in `f32` and rounded to the compute data type. The outputs of the selected experts are summed in
/// `f32`. Both projections are computed by the [`ExpertProjector`] of the caller, which determines their numerical
/// precision.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ParameterizedDerive)]
#[ryft(crate = "ryft_core")]
pub struct Inkling2Experts<P: Parameter> {
    /// Router projection of shape `[experts, hidden]`.
    router: P,

    /// Per expert bias of shape `[experts]` that is added to the routing scores only when selecting experts. It is
    /// selection metadata rather than a trained weight, because the selection has no gradient.
    router_bias: P,

    /// First projection of every expert, of shape `[experts, 2 · expert_size, hidden]`. Its even output rows produce
    /// the gate branch and its odd output rows the up branch of the expert activation.
    gate_up: P,

    /// Second projection of every expert, of shape `[experts, hidden, expert_size]`.
    down: P,
}

impl Inkling2Experts<ArraySpecification> {
    /// Returns the specification of the experts of `layer`.
    fn specification(configuration: &Inkling2Configuration, layer: usize) -> Self {
        let prefix = format!("model.llm.layers.{layer}.mlp");
        let hidden = configuration.hidden_size();
        let experts = configuration.expert_count();
        let expert_size = configuration.expert_size();
        let array = |name: &str, shape: &[usize]| {
            ArraySpecification::new(format!("{prefix}.{name}"), ArrayType::new_static(DataType::F32, shape.to_vec()))
        };
        Self {
            router: array("gate.weight", &[experts, hidden]),
            router_bias: array("gate.bias", &[experts]),
            gate_up: array("experts.w13_weight", &[experts, 2 * expert_size, hidden]),
            down: array("experts.w2_weight", &[experts, hidden, expert_size]),
        }
    }
}

impl<V: ArrayOperations + TopK> Inkling2Experts<V>
where
    V::Type: ArrayMemberType,
{
    /// Routes every token of `input`, which has shape `[batch, sequence, hidden]`, to
    /// [`Inkling2Configuration::experts_per_token`] experts and returns the sum of their outputs with the shape of
    /// `input` and the compute data type, together with the `f32` routing weights of shape
    /// `[batch · sequence, experts]`.
    fn forward<X: ExpertProjector<V>>(
        &self,
        configuration: &Inkling2Configuration,
        projector: &X,
        layer: usize,
        input: &V,
    ) -> Result<(V, V), ProgramError> {
        let compute_type = configuration.compute_type();
        let experts = configuration.expert_count();
        let selected = configuration.experts_per_token();
        let expert_size = configuration.expert_size();
        let shape = static_shape(input)?;
        let (batch, length, hidden) = (shape[0], shape[1], shape[2]);
        let tokens = batch * length;
        let input = input.reshape([tokens, hidden])?;

        // The selection is piecewise constant, so only the scores of the selected experts carry gradients.
        let router = self.router.convert_element_type(compute_type)?;
        let scores = linear(&input, &router, DataType::F32, DataType::F32)?.logistic()?;
        let (_, indices) = Add::add(&scores, &self.router_bias.reshape([1, experts])?)?.top_k(selected, 1)?;
        let expert_indices = scores.iota_like(1, DataType::I32)?.reshape([tokens, 1, experts])?;
        let is_selected = indices
            .reshape([tokens, selected, 1])?
            .equal(&expert_indices.broadcast_to([tokens, selected, experts])?)?
            .reduce_any(&[1])?;
        let selected_scores = Mul::mul(&scores, &is_selected.convert_element_type(DataType::F32)?)?;
        let total = selected_scores.reduce(&[1], ReductionKind::Sum)?.reshape([tokens, 1])?;
        let weights = scale(&Div::div(&selected_scores, &total)?, configuration.route_scale())?;

        // Every expert is evaluated for every token, and the experts that a token is not routed to contribute an
        // exact zero through their routing weight.
        let gate_up = projector.project_shared(layer, &input.convert_element_type(compute_type)?, &self.gate_up)?;
        let gate = gate_up.slice_axis(2, 0, 2 * expert_size, 2)?.convert_element_type(DataType::F32)?;
        let up = gate_up.slice_axis(2, 1, 2 * expert_size, 2)?.convert_element_type(DataType::F32)?;
        let activation = Mul::mul(
            &Mul::mul(&softcap(&gate, configuration.gate_softcap())?, &gate.logistic()?)?,
            &softcap(&up, configuration.up_softcap())?,
        )?;
        let activation = Mul::mul(&activation, &weights.reshape([tokens, experts, 1])?)?;
        let output = projector
            .project_grouped(layer, &activation.convert_element_type(compute_type)?, &self.down)?
            .convert_element_type(DataType::F32)?
            .reduce(&[1], ReductionKind::Sum)?
            .convert_element_type(compute_type)?
            .reshape([batch, length, hidden])?;
        Ok((output, weights))
    }
}

/// Low rank gate that merges a message of the previous pass of the loop into an input of a core layer. For an input
/// `x` and a message `m`, the gate normalizes both by their root mean square, projects their concatenation to
/// [`Inkling2LoopConfiguration::gate_rank`](crate::inkling2::Inkling2LoopConfiguration::gate_rank) dimensions, and
/// produces two bounded gates from that, a forget gate `f` and a gain `g`. The result is `(1 − f) · x + g · m`, so a
/// gate whose projections are zero passes its input through unchanged.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ParameterizedDerive)]
#[ryft(crate = "ryft_core")]
pub struct Inkling2LoopGate<P: Parameter> {
    /// Projection of shape `[gate_rank, 2 · hidden]` of the concatenated normalized input and message.
    down: P,

    /// Projection of shape `[hidden, gate_rank]` that produces the gain of the message.
    gain: P,

    /// Projection of shape `[hidden, gate_rank]` that produces the forget gate of the input.
    forget: P,
}

impl Inkling2LoopGate<ArraySpecification> {
    /// Returns the specification of the loop gate named `name` of `layer`.
    fn specification(configuration: &Inkling2Configuration, layer: usize, name: &str, gate_rank: usize) -> Self {
        let prefix = format!("model.llm.layers.{layer}.controllers.{name}");
        let hidden = configuration.hidden_size();
        let array = |name: &str, shape: [usize; 2]| {
            ArraySpecification::new(format!("{prefix}.{name}"), ArrayType::new_static(DataType::F32, shape))
        };
        Self {
            down: array("down_weight", [gate_rank, 2 * hidden]),
            gain: array("gain_weight", [hidden, gate_rank]),
            forget: array("forget_weight", [hidden, gate_rank]),
        }
    }
}

impl<V: ArrayOperations> Inkling2LoopGate<V>
where
    V::Type: ArrayMemberType,
{
    /// Merges `message` into `input`, both of shape `[batch, sequence, hidden]`, and returns an `f32` array of that
    /// shape. The projection to the gate rank is computed at the compute data type and the rest in `f32`.
    fn forward(&self, configuration: &Inkling2Configuration, input: &V, message: &V) -> Result<V, ProgramError> {
        let compute_type = configuration.compute_type();
        let epsilon = configuration.normalization_epsilon();
        let rounded = |value: &V| value.convert_element_type(compute_type)?.convert_element_type(DataType::F32);
        let joined = rms_norm(input, None, epsilon)?.concatenate_with([&rms_norm(message, None, epsilon)?], 2)?;
        let hidden = linear(&joined, &self.down, compute_type, compute_type)?.convert_element_type(DataType::F32)?;
        let gain = linear(&hidden, &rounded(&self.gain)?, DataType::F32, DataType::F32)?.tanh()?;
        let forget = linear(&hidden, &rounded(&self.forget)?, DataType::F32, DataType::F32)?.tanh()?;
        let input = input.convert_element_type(DataType::F32)?;
        let message = message.convert_element_type(DataType::F32)?;
        Add::add(&Sub::sub(&input, &Mul::mul(&forget, &input)?)?, &Mul::mul(&gain, &message)?)
    }
}

/// Loop gates of one core layer, one for each of its four inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ParameterizedDerive)]
#[ryft(crate = "ryft_core")]
pub struct Inkling2LoopGates<P: Parameter> {
    /// Gate of the input from which queries are projected.
    query: Inkling2LoopGate<P>,

    /// Gate of the input from which keys are projected.
    key: Inkling2LoopGate<P>,

    /// Gate of the input from which values are projected.
    value: Inkling2LoopGate<P>,

    /// Gate of the input of the experts.
    experts: Inkling2LoopGate<P>,
}

/// Messages that one core layer reads from the block states of the previous pass of the loop.
struct Inkling2LoopMessages<V> {
    /// Message for the input from which queries are projected.
    query: V,

    /// Message for the input from which keys are projected.
    key: V,

    /// Message for the input from which values are projected.
    value: V,

    /// Message for the input of the experts.
    experts: V,
}

/// One physical layer of the Inkling 2 decoder, which consists of attention followed by routed experts, each
/// followed by a short convolution, together with the queries and normalization gains of their inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ParameterizedDerive)]
#[ryft(crate = "ryft_core")]
pub struct Inkling2Layer<P: Parameter> {
    /// Queries of the residual reads that produce the attention inputs. A local layer has one query of shape
    /// `[hidden]` that is shared by its queries, keys, and values, and a global layer has three of shape
    /// `[3, hidden]`. The first layer of the decoder has none, because it reads the embeddings directly.
    attention_queries: Option<P>,

    /// Normalization gains of the attention inputs, each of shape `[hidden]`. A local layer has one that is shared
    /// by its queries, keys, and values, and a global layer has three.
    attention_norms: Vec<P>,

    /// Attention of this layer.
    attention: Inkling2Attention<P>,

    /// Short convolution of the attention output.
    attention_convolution: Inkling2Convolution<P>,

    /// Query of shape `[hidden]` of the residual read that produces the input of the experts.
    experts_query: P,

    /// Normalization gain of shape `[hidden]` of the input of the experts.
    experts_norm: P,

    /// Routed experts of this layer.
    experts: Inkling2Experts<P>,

    /// Short convolution of the output of the experts.
    experts_convolution: Inkling2Convolution<P>,

    /// Loop gates of this layer, which only core layers have.
    loop_gates: Option<Inkling2LoopGates<P>>,
}

impl Inkling2Layer<ArraySpecification> {
    /// Returns the specification of `layer`.
    fn specification(configuration: &Inkling2Configuration, layer: usize) -> Self {
        let hidden = configuration.hidden_size();
        let local = configuration.is_local_layer(layer);
        let vector = |name: String| ArraySpecification::new(name, ArrayType::new_static(DataType::F32, [hidden]));
        let prefix = format!("model.llm.layers.{layer}");
        let attention_queries = (layer > 0).then(|| {
            let shape = if local { vec![hidden] } else { vec![3, hidden] };
            ArraySpecification::new(
                format!("model.llm.attn_res.attn_queries.{}", layer - 1),
                ArrayType::new_static(DataType::F32, shape),
            )
        });
        let attention_norms = if local {
            vec![vector(format!("{prefix}.attn_norm.weight"))]
        } else {
            ["q_input_norm", "k_input_norm", "v_input_norm"]
                .map(|name| vector(format!("{prefix}.{name}.weight")))
                .into_iter()
                .collect()
        };
        let loop_gates = configuration.r#loop().filter(|_| configuration.is_core_layer(layer)).map(|r#loop| {
            let gate = |name: &str| Inkling2LoopGate::specification(configuration, layer, name, r#loop.gate_rank());
            Inkling2LoopGates { query: gate("attn"), key: gate("k"), value: gate("v"), experts: gate("mlp") }
        });
        Self {
            attention_queries,
            attention_norms,
            attention: Inkling2Attention::specification(configuration, layer),
            attention_convolution: Inkling2Convolution::static_specification(
                configuration,
                format!("{prefix}.attn_sconv.weight"),
                hidden,
            ),
            experts_query: vector(format!("model.llm.attn_res.mlp_queries.{layer}")),
            experts_norm: vector(format!("{prefix}.mlp_norm.weight")),
            experts: Inkling2Experts::specification(configuration, layer),
            experts_convolution: Inkling2Convolution::static_specification(
                configuration,
                format!("{prefix}.mlp_sconv.weight"),
                hidden,
            ),
            loop_gates,
        }
    }
}

impl<V: ArrayOperations + TopK> Inkling2Layer<V>
where
    V::Type: ArrayMemberType,
{
    /// Applies this layer to the block states of the decoder and returns the routing weights of its experts.
    ///
    /// # Parameters
    ///
    ///   - `configuration`: Configuration of the decoder.
    ///   - `projector`: Projector that computes the expert projections.
    ///   - `masks`: Masks of the inputs of the decoder.
    ///   - `layer`: Index of this layer.
    ///   - `state`: Block states of the decoder, which this function advances.
    ///   - `messages`: Messages from the previous pass of the loop, for the applications of a core layer after the
    ///     first pass.
    fn forward<X: ExpertProjector<V>>(
        &self,
        configuration: &Inkling2Configuration,
        projector: &X,
        masks: &Inkling2Masks<V>,
        layer: usize,
        state: &mut Inkling2State<V>,
        messages: Option<&Inkling2LoopMessages<V>>,
    ) -> Result<V, ProgramError> {
        let compute_type = configuration.compute_type();
        let epsilon = configuration.normalization_epsilon();
        let local = configuration.is_local_layer(layer);
        let rounded = |value: &V| value.convert_element_type(compute_type)?.convert_element_type(DataType::F32);

        // A local layer reads one input that its queries, keys, and values share, and a global layer reads three.
        let mut inputs = match &self.attention_queries {
            None => vec![state.current.clone()],
            Some(queries) if local => vec![state.read(configuration, queries)?],
            Some(queries) => (0..3)
                .map(|index| state.read(configuration, &queries.index_axis(0, index, false)?))
                .collect::<Result<Vec<_>, _>>()?,
        };
        if let (Some(messages), Some(gates)) = (messages, &self.loop_gates) {
            let input = |index: usize| &inputs[index.min(inputs.len() - 1)];
            inputs = vec![
                gates.query.forward(configuration, input(0), &messages.query)?,
                gates.key.forward(configuration, input(1), &messages.key)?,
                gates.value.forward(configuration, input(2), &messages.value)?,
            ];
        }
        let inputs = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let gain = rounded(&self.attention_norms[index.min(self.attention_norms.len() - 1)])?;
                rms_norm(input, Some(&gain), epsilon)?.convert_element_type(compute_type)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input = |index: usize| &inputs[index.min(inputs.len() - 1)];
        let attention = self.attention.forward(configuration, masks, local, input(0), input(1), input(2))?;
        let attention = self.attention_convolution.forward(configuration, masks, &attention, None, 1)?;

        // The first layer of a block completes the previous block and restarts the running state.
        let attention = attention.convert_element_type(DataType::F32)?;
        let current = if layer.is_multiple_of(configuration.block_layer_count()) {
            state.completed.push(state.current.clone());
            attention
        } else {
            Add::add(&state.current.convert_element_type(DataType::F32)?, &attention)?
        };
        state.current = current.clone();

        let experts_input = state.read(configuration, &self.experts_query)?;
        let experts_input = match (messages, &self.loop_gates) {
            (Some(messages), Some(gates)) => gates.experts.forward(configuration, &experts_input, &messages.experts)?,
            _ => experts_input,
        };
        let experts_input = rms_norm(&experts_input, Some(&rounded(&self.experts_norm)?), epsilon)?
            .convert_element_type(compute_type)?;
        let (experts, routing_weights) = self.experts.forward(configuration, projector, layer, &experts_input)?;
        let experts = self.experts_convolution.forward(configuration, masks, &experts, None, 1)?;
        state.current =
            Add::add(&current, &experts.convert_element_type(DataType::F32)?)?.convert_element_type(compute_type)?;
        Ok(routing_weights)
    }
}

/// Block states of the decoder while its layers are applied.
#[derive(Clone)]
struct Inkling2State<V> {
    /// States of the completed blocks, starting with the embeddings, each of shape `[batch, sequence, hidden]`.
    completed: Vec<V>,

    /// Running state of the current block.
    current: V,
}

impl<V: ArrayOperations> Inkling2State<V>
where
    V::Type: ArrayMemberType,
{
    /// Reads the mixture of the completed block states and the running state that `query` selects.
    fn read(&self, configuration: &Inkling2Configuration, query: &V) -> Result<V, ProgramError> {
        let sources = self.completed.iter().chain(std::iter::once(&self.current)).collect::<Vec<_>>();
        residual_read(&sources, query, true, configuration.normalization_epsilon())
    }
}

/// Masks and position dependent factors that every layer application shares, derived once from the
/// [`Inkling2Inputs`].
struct Inkling2Masks<V> {
    /// Whether every position holds a token, as an `f32` array of shape `[batch, sequence, 1]`.
    valid: V,

    /// For every lag from one up to the convolution width minus one, whether the position that many steps earlier
    /// belongs to the same document, as an `f32` array of shape `[batch, sequence, 1]`.
    delays: Vec<V>,

    /// Whether a query position (second axis) of a global layer attends to a key position (third axis), as a Boolean
    /// array of shape `[batch, sequence, sequence]`.
    global: V,

    /// The same for a local layer.
    local: V,

    /// Backward distance from every query position to every key position, clipped to the range of the columns of the
    /// relative position table that the sequence length can reach, as an `i32` array of shape
    /// `[batch, sequence, sequence]`.
    distances: V,

    /// Whether the backward distance from a query position to a key position has a relative position table entry, as
    /// an `f32` array of shape `[batch, sequence, 1, sequence]`.
    in_band: V,

    /// Logarithmic position scale of the queries of global layers, as an `f32` array of shape
    /// `[batch, sequence, 1, 1]`.
    log_scale: V,

    /// Number of leading columns of the relative position table that [`Self::distances`] index.
    table_extent: usize,
}

impl<V: ArrayOperations> Inkling2Masks<V>
where
    V::Type: ArrayMemberType,
{
    /// Derives the masks of `inputs`.
    fn new(configuration: &Inkling2Configuration, inputs: &Inkling2Inputs<V>) -> Result<Self, ProgramError> {
        let shape = static_shape(&inputs.tokens)?;
        if shape.len() != 2 {
            return Err(ProgramError::InvalidArgument {
                message: format!("Inkling 2 tokens must have shape `[batch, sequence]` but got shape {shape:?}"),
            });
        }
        let (batch, length) = (shape[0], shape[1]);
        let pairs = [batch, length, length];
        let valid = inputs.valid.convert_element_type(DataType::F32)?.reshape([batch, length, 1])?;
        let delays = (1..configuration.convolution_width())
            .map(|lag| {
                inputs
                    .segments
                    .equal(&delay(&inputs.segments.reshape([batch, length, 1])?, lag)?.reshape([batch, length])?)?
                    .convert_element_type(DataType::F32)?
                    .reshape([batch, length, 1])
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Positions are within documents, so the backward distance between two positions of one document is the
        // difference of their positions, and it is below the sequence length.
        let table_extent = configuration.relative_extent().min(length);
        let query_positions = inputs.positions.reshape([batch, length, 1])?.broadcast_to(pairs)?;
        let key_positions = inputs.positions.reshape([batch, 1, length])?.broadcast_to(pairs)?;
        let distances = Sub::sub(&query_positions, &key_positions)?;
        let at_least = |limit: i32| distances.greater_than_or_equal(&distances.fill_like(limit)?);
        let below = |limit: usize| distances.less_than(&distances.fill_like(limit as i32)?);
        let causal = distances
            .iota_like(1, DataType::I32)?
            .greater_than_or_equal(&distances.iota_like(2, DataType::I32)?)?;
        let same_document = inputs
            .segments
            .reshape([batch, length, 1])?
            .broadcast_to(pairs)?
            .equal(&inputs.segments.reshape([batch, 1, length])?.broadcast_to(pairs)?)?;
        let both_valid = And::and(
            &inputs.valid.reshape([batch, length, 1])?.broadcast_to(pairs)?,
            &inputs.valid.reshape([batch, 1, length])?.broadcast_to(pairs)?,
        )?;
        let global = And::and(&And::and(&causal, &same_document)?, &both_valid)?;
        let local = And::and(&global, &And::and(&at_least(0)?, &below(configuration.local_window_size())?)?)?;
        let in_band = And::and(&at_least(0)?, &below(configuration.relative_extent())?)?
            .convert_element_type(DataType::F32)?
            .reshape([batch, length, 1, length])?;

        let positions = inputs.positions.convert_element_type(DataType::F32)?;
        let scaled =
            scale(&Add::add(&positions, &positions.fill_like(1.0f32)?)?, 1.0 / configuration.log_scaling_floor())?;
        let log_scale = scale(&scaled.max(&scaled.fill_like(1.0f32)?)?.log()?, configuration.log_scaling_alpha())?;
        let log_scale = Add::add(&log_scale, &log_scale.fill_like(1.0f32)?)?.reshape([batch, length, 1, 1])?;
        let distances =
            distances.clamp(&distances.fill_like(0i32)?, &distances.fill_like((table_extent - 1) as i32)?)?;
        Ok(Self { valid, delays, global, local, distances, in_band, log_scale, table_extent })
    }
}

/// Text decoder of the Inkling 2 model family. Refer to the [module documentation](self) for its structure.
#[derive(Clone, Debug, PartialEq, Eq, Hash, ParameterizedDerive)]
#[ryft(crate = "ryft_core")]
pub struct Inkling2Decoder<P: Parameter> {
    /// Token embeddings of shape `[vocabulary, hidden]`.
    embedding: P,

    /// Normalization gain of shape `[hidden]` of the token embeddings.
    embedding_norm: P,

    /// Physical layers of the decoder.
    layers: Vec<Inkling2Layer<P>>,

    /// Queries of shape `[2 · core_layers, hidden]` of the residual reads that produce the loop messages for the
    /// query input (even rows) and for the experts input (odd rows) of every core layer. Only decoders with a loop
    /// have them.
    loop_queries: Option<P>,

    /// Queries of shape `[2 · core_layers, hidden]` of the residual reads that produce the loop messages for the key
    /// input (even rows) and for the value input (odd rows) of every core layer.
    loop_key_value_queries: Option<P>,

    /// Query of shape `[hidden]` of the residual read that produces the output state.
    output_query: P,

    /// Normalization gain of shape `[hidden]` of the output state.
    output_norm: P,

    /// Unembedding matrix of shape `[vocabulary, hidden]`.
    unembedding: P,
}

impl Inkling2Decoder<ArraySpecification> {
    /// Returns the specification of the parameters of the decoder that `configuration` describes. Every leaf names
    /// the array of an Inkling 2 release checkpoint that it is imported from and has the `f32` data type, which is
    /// the type of the trainable master parameters. The shapes follow the layout of the checkpoint arrays.
    pub fn specification(configuration: &Inkling2Configuration) -> Self {
        let hidden = configuration.hidden_size();
        let array = |name: &str, shape: &[usize]| {
            ArraySpecification::new(format!("model.llm.{name}"), ArrayType::new_static(DataType::F32, shape.to_vec()))
        };
        let loop_queries = |name: &str| {
            configuration
                .r#loop()
                .map(|r#loop| array(name, &[2 * (r#loop.core_end() - r#loop.core_start()), hidden]))
        };
        Self {
            embedding: array("embed.weight", &[configuration.vocabulary_size(), hidden]),
            embedding_norm: array("embed_norm.weight", &[hidden]),
            layers: (0..configuration.layer_count())
                .map(|layer| Inkling2Layer::specification(configuration, layer))
                .collect(),
            loop_queries: loop_queries("middle_cycle_cross_pass_queries"),
            loop_key_value_queries: loop_queries("middle_cycle_cross_pass_kv_queries"),
            output_query: array("attn_res.output_query", &[hidden]),
            output_norm: array("norm.weight", &[hidden]),
            unembedding: array("unembed.weight", &[configuration.vocabulary_size(), hidden]),
        }
    }
}

impl Inkling2Decoder<Array> {
    /// Returns a decoder for `configuration` whose parameters hold reproducible pseudo-random values, for tests and
    /// for experiments that do not need trained weights. This is not the initialization scheme that Inkling 2 models
    /// are trained from.
    ///
    /// Every leaf is filled from its own stream, which is seeded by `seed` and by the name of its checkpoint array,
    /// so the values of a leaf do not depend on the other leaves. Values are uniform in a range that depends on the
    /// role of the leaf. Normalization gains and clip gains are within `0.1` of one, residual read queries are within
    /// `1.5` of zero, convolution taps within `0.5`, router biases within `0.05`, relative position tables within
    /// `1`, and all other leaves within `sqrt(3 / n)` of zero, where `n` is their trailing extent, which gives the
    /// outputs of a projection roughly the variance of its inputs.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `configuration` describes a leaf that cannot be materialized.
    pub fn random(configuration: &Inkling2Configuration, seed: u64) -> Result<Self, ProgramError> {
        Inkling2Decoder::specification(configuration).try_map_parameters(|specification| {
            let name = specification.name();
            let shape = specification.r#type().static_shape().map(|shape| shape.dimensions().to_vec());
            let shape = shape.unwrap_or_default();
            let (center, radius) = if name.ends_with("norm.weight") || name.ends_with("clip.scale") {
                (1.0, 0.1)
            } else if name.contains("queries") || name.ends_with("output_query") {
                (0.0, 1.5)
            } else if name.ends_with("sconv.weight") || name.ends_with("sconv.static_w") {
                (0.0, 0.5)
            } else if name.ends_with("gate.bias") {
                (0.0, 0.05)
            } else if name.ends_with("rel_logits_proj.proj") {
                (0.0, 1.0)
            } else {
                (0.0, (3.0f32 / shape.last().copied().unwrap_or(1) as f32).sqrt())
            };

            // The stream of a leaf is a SplitMix64 generator whose state starts at the seed combined with the FNV-1a
            // hash of the array name, and every value uses the upper 24 bits of one output.
            let mut state = name.bytes().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            }) ^ seed;
            let values = (0..shape.iter().product::<usize>())
                .map(|_| {
                    state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut output = state;
                    output = (output ^ (output >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    output = (output ^ (output >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    output ^= output >> 31;
                    let uniform = (output >> 40) as f32 / (1u32 << 24) as f32;
                    center + radius * (2.0 * uniform - 1.0)
                })
                .collect::<Vec<_>>();
            Array::from_elements(specification.r#type().clone(), &values)
        })
    }
}

impl<V: ArrayOperations + TopK> Inkling2Decoder<V>
where
    V::Type: ArrayMemberType,
{
    /// Returns the next token logits of `inputs` over the usable vocabulary, as an `f32` array of shape
    /// `[batch, sequence, usable_vocabulary]`. The logits are soft capped to
    /// [`Inkling2Configuration::logit_softcap`], and the padding rows of the unembedding matrix take no part in them.
    ///
    /// # Parameters
    ///
    ///   - `configuration`: Configuration of the decoder.
    ///   - `projector`: Projector that computes the expert projections and thereby selects their precision (e.g., a
    ///     [`FloatingExpertProjector`](crate::experts::FloatingExpertProjector) at the compute data type for the
    ///     floating reference numerics).
    ///   - `inputs`: Inputs of the decoder.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the inputs or the parameters do not have the shapes that `configuration`
    /// describes, or if an operation fails.
    pub fn logits<X: ExpertProjector<V>>(
        &self,
        configuration: &Inkling2Configuration,
        projector: &X,
        inputs: &Inkling2Inputs<V>,
    ) -> Result<V, ProgramError> {
        Ok(self.logits_with_diagnostics(configuration, projector, inputs)?.0)
    }

    /// Returns the logits of [`Self::logits`] together with the [`Inkling2Diagnostics`] of the evaluation.
    pub fn logits_with_diagnostics<X: ExpertProjector<V>>(
        &self,
        configuration: &Inkling2Configuration,
        projector: &X,
        inputs: &Inkling2Inputs<V>,
    ) -> Result<(V, Inkling2Diagnostics<V>), ProgramError> {
        let compute_type = configuration.compute_type();
        let epsilon = configuration.normalization_epsilon();
        let rounded = |value: &V| value.convert_element_type(compute_type)?.convert_element_type(DataType::F32);
        if self.layers.len() != configuration.layer_count() {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "Inkling 2 decoder has {} layers but its configuration describes {}",
                    self.layers.len(),
                    configuration.layer_count(),
                ),
            });
        }

        let masks = Inkling2Masks::new(configuration, inputs)?;
        let embeddings = self.embedding.gather_axis(&inputs.tokens, 0, GatherMode::Clip)?;
        let embeddings = rms_norm(&rounded(&embeddings)?, Some(&rounded(&self.embedding_norm)?), epsilon)?;
        let embeddings = Mul::mul(&embeddings, &masks.valid)?.convert_element_type(compute_type)?;
        let mut state = Inkling2State { completed: Vec::new(), current: embeddings };
        let mut diagnostics = Inkling2Diagnostics { states: Vec::new(), routing_weights: Vec::new() };
        let mut apply = |layer: usize,
                         state: &mut Inkling2State<V>,
                         messages: Option<&Inkling2LoopMessages<V>>|
         -> Result<(), ProgramError> {
            let routing_weights =
                self.layers[layer].forward(configuration, projector, &masks, layer, state, messages)?;
            diagnostics.states.push(state.current.clone());
            diagnostics.routing_weights.push(routing_weights);
            Ok(())
        };

        match configuration.r#loop() {
            None => {
                for layer in 0..configuration.layer_count() {
                    apply(layer, &mut state, None)?;
                }
            }
            Some(r#loop) => {
                for layer in 0..r#loop.core_start() {
                    apply(layer, &mut state, None)?;
                }

                // Every pass restarts from the state before the core. A pass after the first one reads the states
                // of the blocks that the previous pass wrote (i.e., everything after the blocks that preceded the
                // core and after the running state at the start of the core, followed by its final running state).
                let start = state.clone();
                let mut bank: Option<Vec<V>> = None;
                for pass in 0..r#loop.pass_count() {
                    if pass > 0 {
                        let mut previous = state.completed[start.completed.len() + 1..].to_vec();
                        previous.push(state.current.clone());
                        bank = Some(previous);
                        state = start.clone();
                    }
                    for layer in r#loop.core_start()..r#loop.core_end() {
                        let messages = match &bank {
                            None => None,
                            Some(bank) => Some(self.loop_messages(configuration, bank, layer - r#loop.core_start())?),
                        };
                        apply(layer, &mut state, messages.as_ref())?;
                    }
                }
                for layer in r#loop.core_end()..configuration.layer_count() {
                    apply(layer, &mut state, None)?;
                }
            }
        }

        // The projection onto the vocabulary accumulates in `f32`, and only the usable rows of the unembedding
        // matrix take part in it.
        let output = state.read(configuration, &self.output_query)?;
        let output = rms_norm(&output, Some(&rounded(&self.output_norm)?), epsilon)?;
        let unembedding = self.unembedding.slice_axis(0, 0, configuration.usable_vocabulary_size(), 1)?;
        let logits = linear(&output, &unembedding, compute_type, DataType::F32)?;
        Ok((softcap(&logits, configuration.logit_softcap())?, diagnostics))
    }

    /// Returns the mean cross entropy of the next token predictions of `inputs`, as a rank-0 `f32` array. The loss
    /// of position `t` is the negative log-probability of `labels[t]` under the softmax of the logits of that
    /// position over the usable vocabulary, and the result is `∑ weights · losses / max(∑ weights, 1)`. Weights are
    /// zero for positions that are not supervised (e.g., prompt tokens, padding, and the last token of a document,
    /// which has no next token), and their labels are ignored but must still be valid token identifiers.
    ///
    /// # Parameters
    ///
    ///   - `configuration`: Configuration of the decoder.
    ///   - `projector`: Projector that computes the expert projections (refer to [`Self::logits`]).
    ///   - `inputs`: Inputs of the decoder.
    ///   - `labels`: Next token identifiers, as an `i32` array of shape `[batch, sequence]`.
    ///   - `weights`: Supervision weights, as an `f32` array of shape `[batch, sequence]`.
    pub fn loss<X: ExpertProjector<V>>(
        &self,
        configuration: &Inkling2Configuration,
        projector: &X,
        inputs: &Inkling2Inputs<V>,
        labels: &V,
        weights: &V,
    ) -> Result<V, ProgramError> {
        let logits = self.logits(configuration, projector, inputs)?;
        let losses = logits.softmax_cross_entropy_with_integer_labels(labels, 2)?;
        let total = Mul::mul(&losses, weights)?.reduce(&[0, 1], ReductionKind::Sum)?;
        let count = weights.reduce(&[0, 1], ReductionKind::Sum)?;
        Div::div(&total, &count.max(&count.fill_like(1.0f32)?)?)
    }

    /// Reads the messages for the core layer at `offset` from the start of the core out of the block states `bank`
    /// of the previous pass of the loop.
    fn loop_messages(
        &self,
        configuration: &Inkling2Configuration,
        bank: &[V],
        offset: usize,
    ) -> Result<Inkling2LoopMessages<V>, ProgramError> {
        let (Some(queries), Some(key_value_queries)) = (&self.loop_queries, &self.loop_key_value_queries) else {
            return Err(ProgramError::InvalidArgument {
                message: "Inkling 2 decoder with a loop has no loop queries".to_string(),
            });
        };
        let sources = bank.iter().collect::<Vec<_>>();
        let read = |queries: &V, row: usize| {
            residual_read(&sources, &queries.index_axis(0, row, false)?, true, configuration.normalization_epsilon())
        };
        Ok(Inkling2LoopMessages {
            query: read(queries, 2 * offset)?,
            experts: read(queries, 2 * offset + 1)?,
            key: read(key_value_queries, 2 * offset)?,
            value: read(key_value_queries, 2 * offset + 1)?,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;
    use ryft_core::{ArrayOperation, EagerContext, RandomAlgorithm, Typed, differentiate_at};

    use crate::experts::{
        BlockScaledExpertProjector, BlockScaledFormat, BlockScaledProjections, FloatingExpertProjector,
    };

    use super::*;

    /// Logits of the pinned floating reference implementation for the reference parity test, with shape `[1, 12, 60]`.
    #[rustfmt::skip]
    const REFERENCE_LOGITS: [f32; 720] = [
        -0.0041926624, -0.294529, 0.98872215, -1.8936015, 1.1010361, 0.70328635, 1.5931106, 1.155435, -0.0870258,
        1.784771, -0.3074603, -2.2054596, -1.0632741, -0.970659, 0.69281, 0.81480736, -0.027210128, 0.94196075,
        0.30156425, 0.040493432, 2.281586, 0.7989078, -2.515344, 1.2053494, -1.5420872, 0.0977053, -0.33173293,
        -0.1882148, 0.2673032, -0.1395614, 0.31231672, -0.6228177, 1.0102525, -1.3769597, -0.42995518, -0.6425873,
        -0.9286554, -1.6614392, 1.6279267, -0.8072321, -1.3300475, -1.5364203, -1.2169807, -0.9391319, -0.30235398,
        -1.2044344, 0.84517616, -0.2513546, -1.9962121, -1.53131, -0.56828547, -0.6285432, 1.6272936, 0.03282892,
        0.74944705, 0.14411029, 0.10921541, 0.29227135, 1.5280793, 2.1055105, -0.8835558, -0.53783363, 1.5592163,
        0.37031317, -1.5131531, 0.7270836, 1.0698204, 0.90139097, 0.39875394, -0.04969743, -1.0676708, -0.20452738,
        -1.7696468, 0.97422445, -1.1720115, 1.635237, 0.7688502, -0.051348675, -0.33485106, -1.2359003, -0.17691581,
        -0.6856081, 0.4418821, 0.5546247, -1.5507461, -1.4470935, -0.24452794, 1.6937572, -1.1501358, 1.257873,
        1.4683409, 0.14317581, 1.5382192, -1.607642, -2.9494803, -0.47512498, -0.6707014, -0.4149822, -1.0839734,
        -0.06042709, -2.2453516, -2.6066706, 0.17492463, -0.9271163, -0.27967253, 0.74374187, -0.47717142, 0.4761405,
        -2.7292345, 0.12801033, 0.26748836, 1.8302577, 0.39163077, -1.4345089, -0.25281546, 0.37606144, -1.229323,
        -0.21479002, 1.0194763, 2.1554668, 0.07204765, -0.6983464, -0.7242971, 0.39569157, -0.73837715, -0.4749973,
        1.172243, 0.36234912, -0.25225928, -0.5025293, -0.97356844, -0.5929824, -0.43762192, 1.3123701, -0.39325893,
        0.76093096, -0.17717926, -0.27630875, 2.1685266, 0.9548673, -1.7462672, -1.7100393, 1.2091922, -1.194226,
        -0.25874132, -0.6925724, -1.390141, 0.8840504, -0.110040456, 0.7339409, 1.0786015, -0.15856801, 0.3189681,
        -0.3125346, -0.97284895, 0.18202305, -0.9858257, -0.3890448, -0.8194032, 0.2646796, -0.18200532, -1.1004355,
        -0.14953302, -0.01764016, -1.0955839, -0.29392976, -0.55681074, -0.04804562, -0.5603325, -0.537028, 0.8131628,
        0.49571487, 0.3336598, 0.41286188, 0.7154961, -0.2949155, -0.87447816, -1.1451186, 0.48363435, 0.37542695,
        -1.1788464, -0.44831538, 1.2445691, 0.81100696, 1.1479332, 0.5416323, 1.0478635, 0.52946186, 2.435487,
        -0.52673805, -0.6940291, 0.49727637, -1.7647195, -1.3938142, 0.123091236, 0.9412077, -0.48462364, -0.014548745,
        0.2113118, 1.188616, -0.4828302, 0.35634217, 0.3952093, 0.87979305, 0.5911405, -0.35125685, 0.41337532,
        2.431325, -0.5259935, 1.7713441, 0.4873347, -1.5197073, 2.1732848, 0.5893899, -1.4574114, 0.0638198,
        -0.55925065, 1.1725471, -1.5773449, 0.36930946, -1.9161507, -1.3405577, -2.0239036, -1.3147697, -1.1017027,
        -1.1819344, 1.0524372, 0.15290917, 0.8742159, -0.043142747, -0.76576763, 0.6383624, 0.5427145, 0.60337824,
        -0.71316564, -0.33960664, -1.4491119, 1.4856809, 0.7005823, 1.1450182, 0.27486783, -1.1143246, 0.19002959,
        -2.381166, 0.90590173, -0.6882785, 0.49534974, -0.04705131, 1.4821324, 1.4397985, -0.13401264, -0.35567325,
        -1.8002017, -1.9789195, -1.4152926, -0.17139596, 0.3110849, 1.4879047, 0.18887484, 1.2461735, -0.661928,
        -0.7701708, -0.42986462, 0.6431805, -0.47035787, -0.14832501, -1.3170168, 1.2633313, 0.8046772, 0.44217616,
        0.119466946, -1.5883117, -0.19215178, -1.0195478, -1.0325931, 0.40879163, -1.8573297, -0.029873198, 0.10112955,
        1.4541388, 0.47578347, -0.10687942, -1.0593346, -0.5434081, -1.517642, -0.78336763, 0.88554454, 0.64176553,
        0.6841654, -0.84095365, -1.2348526, -1.741997, 1.3699471, 0.8721559, -0.5308418, -0.8629014, 0.64793766,
        -0.43441132, 1.9250474, 1.1458169, -0.18355133, 1.9211652, 1.51239, -1.9399904, 0.501073, -0.017839363,
        0.4397449, 0.4127786, 1.6895146, 0.4296849, -2.1242476, 1.7314237, 0.41012442, -0.28251255, 0.33306712,
        0.060337592, 1.3450876, 0.66263944, -0.5190798, 1.9369651, -2.5879793, -0.40360597, -0.025626859, 1.7037958,
        0.8651802, 1.0416822, 0.33870858, 2.6289601, -0.3185604, 1.5191661, 0.6933104, 0.62896305, 0.6658308, 1.0928093,
        0.10542467, -0.7167592, -0.7409781, 0.6584964, -1.2269881, 0.38300952, -0.87243325, -0.2556626, -1.6802837,
        -1.2653208, 0.71101326, -0.22407392, -0.95035976, 1.6984432, -0.49486277, 0.18679978, 1.2343832, 0.42097822,
        0.7015636, 0.33604681, -0.3582214, -0.3478567, -1.7459761, 0.10897221, 1.8996983, 2.0447297, -1.1942474,
        -0.5148887, 1.4890162, 0.63544154, -0.9174095, -0.11487336, -0.05592191, 1.2374318, -1.5188249, -0.2029359,
        -0.7712762, -0.27573073, -0.564533, 0.40927452, -0.53134716, -0.17412663, 0.19353361, -0.673013, 2.6158483,
        1.4212373, 0.16226843, -0.37239233, 2.248947, -0.9077576, -0.17264043, -0.94369334, -0.06398803, 1.0974592,
        -0.5432417, -0.14740752, -0.6350855, 0.66083014, 0.30403748, -0.75415814, -0.40274045, 0.91088307, -0.48337775,
        -1.8476996, -0.41008833, 1.140204, -0.29101038, -0.38595155, -0.65829724, -1.6422639, -1.6128393, -0.060422506,
        0.40761855, -0.37520227, 0.1689788, -1.1124252, 0.00029005343, 0.77334154, -0.6942811, 0.24063201, 1.9462435,
        1.343115, -1.0260987, -0.9900321, 0.7622077, 0.83768225, 0.0898374, -0.30570695, 0.41204, 0.47833917, 1.4383891,
        -0.58482605, 0.15922813, 1.2217689, -1.7712691, 1.8802696, -0.7857258, -1.1856533, -0.91422385, 0.5429873,
        0.44109154, 2.088131, -0.7494562, -0.8915295, 1.7781916, -0.20691937, 1.6022654, -0.6796504, -0.24000077,
        -0.5081963, 0.40064093, -0.901458, 0.9248056, -0.71315044, 0.89987296, 0.36940065, 0.0985779, -1.5883061,
        0.056186434, 0.84093505, -1.8270922, -1.0095837, -1.2416975, -0.6013996, 1.5209355, 1.4193337, -0.097896606,
        -1.2996886, 0.8561696, -0.3218562, -0.35844907, -1.6870325, 1.2175822, -0.41407552, -0.69997925, -0.7937236,
        0.79744285, 1.1018009, 0.11560528, 2.0064476, 1.4333411, -0.89882475, -0.517694, 1.3870058, -0.59971136,
        0.4548923, 1.6331686, -0.6775443, 0.9203158, -1.0697911, -0.2916495, -0.6458049, 0.98981345, 1.2430454,
        0.25783396, 0.39431214, -1.9514616, -0.7457841, 0.8764963, 0.69055545, 1.1500193, -1.0361223, 0.16686612,
        0.2797541, 1.6295233, 0.56005955, -0.2954748, -0.27569133, -0.39477277, 0.4726054, 0.08142551, 0.21259469,
        1.5932219, 0.8364515, -0.93217856, 0.5440914, 0.7037685, 0.46461728, -0.21734543, -0.57554394, -0.9947552,
        -1.1823275, 0.2550014, -1.5693815, 0.95808965, -1.2029433, -1.0356346, -0.35008526, -0.6956089, 0.26599416,
        1.3566719, -1.1363572, -0.5948013, 0.82893306, 0.8133523, -0.50819296, 0.6708992, 1.8552178, 1.0572358,
        0.84722644, 0.4185155, -0.5958121, -1.2422285, -0.58348817, 1.5211289, 0.04844419, 0.5510284, 0.7102635,
        2.0640807, -0.093107335, 0.45404756, -0.16027959, -0.6304519, 2.4477649, -1.5842794, 0.9419075, -0.543422,
        -1.0361311, -0.08294871, 1.224839, 0.4908966, -1.0428606, 0.3492467, 0.36802173, 2.1588192, -0.4344104,
        1.3363948, 2.0451066, -0.75892293, 1.1652062, -0.07863261, -1.0049708, 1.6110644, 0.10207116, 0.30334896,
        0.8311627, -2.0342894, 0.016057279, -0.3335888, -0.4365438, -0.01412096, -1.86574, 0.119183764, -1.1818742,
        1.5404344, 1.0050069, -2.2530136, -1.0919752, 0.3645607, -0.86540014, 0.46244746, 0.23615701, -0.46331048,
        -0.15706953, 0.023252754, -0.42511138, 1.4752926, 1.1421646, 0.26793766, 1.0568849, 1.0529753, 0.68054485,
        -2.086653, 0.2851783, -0.46077082, 0.87001336, 0.73186195, -1.0003775, 0.33080208, 1.3330973, -1.6699128,
        -0.3618437, 0.5103634, 0.18541409, -0.31241992, 0.24613799, 0.27683675, -1.35293, 0.19369031, 1.7391623,
        1.6225393, 1.3575857, -1.3761466, -1.033462, -1.0871029, -0.262454, 1.5882452, 0.07517068, -1.5760607,
        -0.6424593, -1.6428525, -2.8716521, 2.246107, -1.9023004, -2.1253748, -0.27240586, 2.0248039, -0.00024734667,
        -0.2647406, -1.89434, -1.0540795, -0.5383472, 0.9928095, 0.51275706, -0.89284515, -1.6504736, -0.8324866,
        -1.9920077, 1.7000419, 2.258132, 0.37737724, 0.49655652, 0.629353, -1.9466096, -1.099119, -0.5879613,
        0.36981863, 1.2722294, 1.2763387, 0.7240797, 0.5162945, -1.8486255, -0.7389117, 0.654755, -0.5378679,
        0.69786215, -0.98653054, 1.4849337, 0.6952537, 0.32574928, 1.2746998, -0.21250315, -0.63152844, 1.5747651,
        0.9003989, -0.5060788, 0.32628065, 0.0016523, -0.4557642, -0.15747164, 0.013957588, -0.023256602, -0.47930232,
        0.6237202, 0.4416293, -0.5300377, 0.9638729, 2.2344007, -0.82559425, 0.25695992, 0.7158951, 0.7306155,
        -0.582684, 1.6521835, 0.6718152, 0.9811989, -1.342855, 1.0050306, 1.6443766, 1.4164577, 1.0795205, 0.2946096,
        -0.52010494, 0.7765004, -1.0780042, 0.69762164, -1.953717, -0.5653967, -1.0129194, -0.3029557, -0.19084366,
        0.043605033, -0.010136271, -0.4341456, 0.9415489, -0.004748245, 1.1620948, -0.6934339, -0.90615827, -0.9847235,
        -1.8467591, 2.8167586, -1.3997514, 2.04938, -1.3154302, 1.0713198,
    ];

    /// Loss of the reference implementation for the reference parity test.
    const REFERENCE_LOSS: f32 = 3.8852878;

    /// Size of the block state of the reference implementation after every layer application.
    #[rustfmt::skip]
    const REFERENCE_STATE_NORMS: [f32; 12] = [
        36.44949, 47.396427, 29.179478, 38.73393, 34.33074, 48.281464, 34.456223, 47.1829, 37.5557, 49.12348, 30.223486,
        45.63131,
    ];

    /// Experts that the reference implementation routes the first token to in every layer application.
    #[rustfmt::skip]
    const REFERENCE_ROUTING: [[usize; 2]; 12] = [
        [0, 2], [0, 2], [1, 2], [2, 3], [0, 3], [0, 3], [0, 3], [2, 3], [1, 3], [1, 3], [1, 2], [2, 3],
    ];

    /// Gradient of the loss of the reference implementation with respect to every parameter, as the name of the
    /// checkpoint array of the parameter, the norm of its gradient, and the leading values of its gradient.
    #[rustfmt::skip]
    const REFERENCE_GRADIENTS: [(&str, f32, &[f32]); 246] = [
        ("model.llm.attn_res.attn_queries.0", 0.08515943, &[-0.0035764077, 6.693695e-05, 0.007943752]),
        ("model.llm.attn_res.attn_queries.1", 0.50832605, &[-0.040975057, -0.19545352, -0.04615037]),
        ("model.llm.attn_res.attn_queries.2", 0.08890091, &[-0.004415988, 0.0026350152, 0.0005616769]),
        ("model.llm.attn_res.attn_queries.3", 0.40798262, &[0.0071876273, -0.020845518, -0.06759992]),
        ("model.llm.attn_res.attn_queries.4", 0.11773505, &[0.011942254, -0.0074647586, 0.0028772089]),
        ("model.llm.attn_res.attn_queries.5", 0.05413186, &[0.011345601, -0.0058491714, 0.0077961027]),
        ("model.llm.attn_res.attn_queries.6", 0.01572435, &[0.00082922727, -0.0005691533, 0.0002516057]),
        ("model.llm.attn_res.mlp_queries.0", 0.7784, &[-0.2720319, -0.023025382, -0.07604931]),
        ("model.llm.attn_res.mlp_queries.1", 0.4053905, &[0.072768845, 0.010630403, -0.0647556]),
        ("model.llm.attn_res.mlp_queries.2", 0.9044588, &[0.18133365, -0.136228, -0.19208653]),
        ("model.llm.attn_res.mlp_queries.3", 0.68889856, &[-0.021996226, -0.1341967, -0.040402338]),
        ("model.llm.attn_res.mlp_queries.4", 0.5643214, &[0.058913153, 0.04026013, -0.14076775]),
        ("model.llm.attn_res.mlp_queries.5", 0.37076527, &[-0.00074138585, 0.04816774, 0.13738859]),
        ("model.llm.attn_res.mlp_queries.6", 0.1377818, &[-0.025074996, 0.014078194, 0.010545937]),
        ("model.llm.attn_res.mlp_queries.7", 0.090205275, &[0.0133667225, 0.0072826114, 0.001451853]),
        ("model.llm.attn_res.output_query", 0.21637551, &[0.007234564, 0.002616246, 0.0042723315]),
        ("model.llm.embed.weight", 16.546707, &[0.0, 0.0, 0.0]),
        ("model.llm.embed_norm.weight", 2.7362404, &[0.6763283, 0.73145384, 0.0008680485]),
        ("model.llm.layers.0.attn.attn_sink_proj.weight", 2.2036607, &[-0.40309095, 0.28011727, 0.047192607]),
        ("model.llm.layers.0.attn.gate_attn.weight", 2.5370374, &[0.12213459, -0.024626005, -0.14350161]),
        ("model.llm.layers.0.attn.k_sconv.weight", 0.39391387, &[-0.001859678, 0.01070589, -0.0736602]),
        ("model.llm.layers.0.attn.q_l2_clip.scale", 0.22490029, &[-0.031439863, -0.11175149, -0.07562183]),
        ("model.llm.layers.0.attn.rel_l2_clip.scale", 0.5090265, &[0.069999084, 0.007143693, -0.1732708]),
        ("model.llm.layers.0.attn.rel_logits_proj.proj", 1.1739243, &[0.88938856, -0.16845527, -0.07557472]),
        ("model.llm.layers.0.attn.v_sconv.weight", 2.0961723, &[-0.08791468, -0.026024936, -0.04756316]),
        ("model.llm.layers.0.attn.wk_dv.weight", 1.4607775, &[0.0064406293, 0.017124733, -0.03323162]),
        ("model.llm.layers.0.attn.wo_ud.weight", 8.677388, &[-0.47327337, -0.020970155, 0.22582583]),
        ("model.llm.layers.0.attn.wq_du.weight", 1.6723351, &[0.12486433, -0.009485894, 0.07473437]),
        ("model.llm.layers.0.attn.wr_du.weight", 3.6817143, &[0.27702373, 0.05160494, -0.19402681]),
        ("model.llm.layers.0.attn.wv_dv.weight", 7.5974092, &[0.07068639, 0.12718748, -0.0007718385]),
        ("model.llm.layers.0.attn_norm.weight", 1.8828577, &[0.69628733, 0.08663979, 0.13038854]),
        ("model.llm.layers.0.attn_sconv.weight", 1.8757039, &[0.10215906, -0.026335208, 0.08341355]),
        ("model.llm.layers.0.mlp.experts.w13_weight", 10.678317, &[0.09123503, 0.20473433, -0.033521987]),
        ("model.llm.layers.0.mlp.experts.w2_weight", 7.5285096, &[-0.06194307, -0.008790425, 0.0015226947]),
        ("model.llm.layers.0.mlp.gate.bias", 0.0, &[0.0, 0.0, 0.0]),
        ("model.llm.layers.0.mlp.gate.weight", 1.7083248, &[-0.092143096, -0.32611978, -0.21476719]),
        ("model.llm.layers.0.mlp_norm.weight", 1.4582336, &[-0.45760483, 0.3686662, 0.027405206]),
        ("model.llm.layers.0.mlp_sconv.weight", 2.0732625, &[-0.021248773, 0.23375647, 0.315309]),
        ("model.llm.layers.1.attn.attn_sink_proj.weight", 0.24028108, &[0.0011871038, 0.0015900475, 0.0035566476]),
        ("model.llm.layers.1.attn.gate_attn.weight", 0.24838552, &[-0.0010512654, -0.00017673445, -0.0011139237]),
        ("model.llm.layers.1.attn.k_sconv.static_w", 0.02442725, &[0.0060001225, 0.0009251833, -0.004867628]),
        ("model.llm.layers.1.attn.k_sconv.weight_proj", 0.10563087, &[-0.0041647675, 0.008272161, 0.00069983874]),
        ("model.llm.layers.1.attn.q_l2_clip.scale", 0.015559476, &[0.00073843246, 0.006839865, 0.0023685968]),
        ("model.llm.layers.1.attn.rel_l2_clip.scale", 0.039906904, &[-0.0009728556, 0.018001288, -0.012403884]),
        ("model.llm.layers.1.attn.rel_logits_proj.proj", 0.10272024, &[0.030313017, -0.018792612, -0.022816231]),
        ("model.llm.layers.1.attn.v_sconv.static_w", 0.23400201, &[-0.0066242525, -0.09271979, -0.008158584]),
        ("model.llm.layers.1.attn.v_sconv.weight_proj", 0.82354736, &[0.039314672, -0.011854365, -0.008971967]),
        ("model.llm.layers.1.attn.wk_dv.weight", 0.14866465, &[-0.00076907524, -0.002516189, -0.009112892]),
        ("model.llm.layers.1.attn.wo_ud.weight", 0.59829915, &[0.0054708724, 0.034043957, -0.011174587]),
        ("model.llm.layers.1.attn.wq_du.weight", 0.15942048, &[-0.00060835754, -0.00078040507, 0.004006235]),
        ("model.llm.layers.1.attn.wr_du.weight", 0.28694823, &[0.0013067332, 0.00047205112, 0.00096483255]),
        ("model.llm.layers.1.attn.wv_dv.weight", 1.0682712, &[0.00521582, -0.0026583553, 0.015465584]),
        ("model.llm.layers.1.attn_sconv.weight", 0.11873547, &[-0.0040153502, 0.0031649484, -0.0035833763]),
        ("model.llm.layers.1.k_input_norm.weight", 0.028830193, &[-0.0007708913, -0.0046134945, -0.010067954]),
        ("model.llm.layers.1.mlp.experts.w13_weight", 7.7757754, &[-0.017762836, 0.070327856, -0.030041948]),
        ("model.llm.layers.1.mlp.experts.w2_weight", 5.3080916, &[-0.01067508, 0.02882758, 0.010287251]),
        ("model.llm.layers.1.mlp.gate.bias", 0.0, &[0.0, 0.0, 0.0]),
        ("model.llm.layers.1.mlp.gate.weight", 1.1774404, &[0.016860658, -0.023987396, 0.009018998]),
        ("model.llm.layers.1.mlp_norm.weight", 1.3628932, &[0.13265835, 0.19826187, -0.18659821]),
        ("model.llm.layers.1.mlp_sconv.weight", 1.2775495, &[0.09841196, -0.042146657, 0.031971633]),
        ("model.llm.layers.1.q_input_norm.weight", 0.070949085, &[-0.00055370096, 0.008903256, 0.009260358]),
        ("model.llm.layers.1.v_input_norm.weight", 0.18839334, &[0.015605292, 0.030900497, -0.0074024987]),
        ("model.llm.layers.2.attn.attn_sink_proj.weight", 3.2170286, &[-0.03106091, 0.00262372, -0.035838217]),
        ("model.llm.layers.2.attn.gate_attn.weight", 2.4627514, &[0.06270659, 0.016167188, 0.13581023]),
        ("model.llm.layers.2.attn.k_sconv.weight", 0.34905168, &[-0.008101355, 0.0024938795, -0.013180884]),
        ("model.llm.layers.2.attn.q_l2_clip.scale", 0.36635026, &[-0.0054337466, 0.008124073, -0.029265797]),
        ("model.llm.layers.2.attn.rel_l2_clip.scale", 0.37813875, &[-0.007796035, 0.09813674, -0.014840961]),
        ("model.llm.layers.2.attn.rel_logits_proj.proj", 0.7289763, &[-0.29093832, -0.067095846, 0.112692654]),
        ("model.llm.layers.2.attn.v_sconv.weight", 1.0111357, &[0.23941644, 0.11370141, 0.16567297]),
        ("model.llm.layers.2.attn.wk_dv.weight", 1.7630254, &[0.047296204, 0.02109985, 0.08705659]),
        ("model.llm.layers.2.attn.wo_ud.weight", 7.4350386, &[0.025468862, -0.07648168, -0.00053242967]),
        ("model.llm.layers.2.attn.wq_du.weight", 2.2833247, &[0.001063338, -0.010161913, -0.0385773]),
        ("model.llm.layers.2.attn.wr_du.weight", 2.1338832, &[0.030557495, 0.027701957, 0.052853897]),
        ("model.llm.layers.2.attn.wv_dv.weight", 5.533789, &[0.009973617, 0.16052787, -0.0026846696]),
        ("model.llm.layers.2.attn_norm.weight", 1.228754, &[0.0042830426, 0.13700461, 0.11432632]),
        ("model.llm.layers.2.attn_sconv.weight", 1.6633987, &[-0.061288714, -0.03319451, 0.00030147657]),
        ("model.llm.layers.2.controllers.attn.down_weight", 1.2475069, &[0.037176196, 0.04396072, 0.043295402]),
        ("model.llm.layers.2.controllers.attn.forget_weight", 0.29975614, &[0.011004452, -0.0048503713, -0.0018222422]),
        ("model.llm.layers.2.controllers.attn.gain_weight", 0.36340547, &[0.004187278, -0.0010377866, 0.00033204397]),
        ("model.llm.layers.2.controllers.k.down_weight", 0.6216963, &[0.0066756443, -0.011289889, 0.0010935661]),
        ("model.llm.layers.2.controllers.k.forget_weight", 0.13059542, &[-0.0010215777, -9.506162e-05, 0.0017801917]),
        ("model.llm.layers.2.controllers.k.gain_weight", 0.12351635, &[0.0019109811, 0.0024243095, -0.0007729885]),
        ("model.llm.layers.2.controllers.mlp.down_weight", 2.6783967, &[0.06962422, 0.095068716, -0.050259992]),
        ("model.llm.layers.2.controllers.mlp.forget_weight", 0.6276377, &[0.047733616, 0.01659158, 0.062336463]),
        ("model.llm.layers.2.controllers.mlp.gain_weight", 0.556827, &[-0.006071783, 0.004000269, -0.008731817]),
        ("model.llm.layers.2.controllers.v.down_weight", 2.4703438, &[0.07563893, 0.13849452, 0.2168087]),
        ("model.llm.layers.2.controllers.v.forget_weight", 0.6049484, &[-0.008344516, 0.014962057, 0.0100846095]),
        ("model.llm.layers.2.controllers.v.gain_weight", 0.7742162, &[-0.006551415, 0.012243962, 0.009478755]),
        ("model.llm.layers.2.mlp.experts.w13_weight", 7.547047, &[-0.07839173, -0.0036326821, -0.015003697]),
        ("model.llm.layers.2.mlp.experts.w2_weight", 4.9732647, &[-0.09355792, 0.014760533, 0.065068275]),
        ("model.llm.layers.2.mlp.gate.bias", 0.0, &[0.0, 0.0, 0.0]),
        ("model.llm.layers.2.mlp.gate.weight", 2.2896895, &[-0.30384225, -0.08399693, 0.025776159]),
        ("model.llm.layers.2.mlp_norm.weight", 1.0658691, &[-0.05201308, 0.058431823, 0.09866143]),
        ("model.llm.layers.2.mlp_sconv.weight", 1.1406683, &[0.026084531, -0.15211897, -0.064117275]),
        ("model.llm.layers.3.attn.attn_sink_proj.weight", 0.10099738, &[-0.01105857, -0.0052388166, -0.007172522]),
        ("model.llm.layers.3.attn.gate_attn.weight", 0.15396674, &[0.0049215225, 0.0027514787, -0.0017385785]),
        ("model.llm.layers.3.attn.k_sconv.static_w", 0.01672208, &[-0.0020211134, 0.0007897014, 0.0018000179]),
        ("model.llm.layers.3.attn.k_sconv.weight_proj", 0.08948316, &[0.000801623, -0.0035835833, -0.0044008577]),
        ("model.llm.layers.3.attn.q_l2_clip.scale", 0.013999726, &[-0.0015699615, 0.0010930377, 0.0029575515]),
        ("model.llm.layers.3.attn.rel_l2_clip.scale", 0.021641705, &[-0.0011714692, -0.0012816521, 0.0020916732]),
        ("model.llm.layers.3.attn.rel_logits_proj.proj", 0.05450902, &[0.033586573, 0.0014207006, -0.002072148]),
        ("model.llm.layers.3.attn.v_sconv.static_w", 0.087333865, &[0.05431569, -0.037672926, 0.00069409236]),
        ("model.llm.layers.3.attn.v_sconv.weight_proj", 0.22987004, &[-0.02890871, 0.0130146425, -0.005647626]),
        ("model.llm.layers.3.attn.wk_dv.weight", 0.08809519, &[-0.0025984382, -0.0032723816, -0.0021608924]),
        ("model.llm.layers.3.attn.wo_ud.weight", 0.52230626, &[0.004200786, 0.0047872295, 0.0011554582]),
        ("model.llm.layers.3.attn.wq_du.weight", 0.11024279, &[0.0016946413, 0.0024328558, 0.000930737]),
        ("model.llm.layers.3.attn.wr_du.weight", 0.16391021, &[0.005121085, 0.010074297, 0.00016211692]),
        ("model.llm.layers.3.attn.wv_dv.weight", 0.52186394, &[-0.0141937705, 0.023235222, 0.037883222]),
        ("model.llm.layers.3.attn_sconv.weight", 0.11924866, &[-0.003519853, -0.0065806266, -0.010878163]),
        ("model.llm.layers.3.controllers.attn.down_weight", 0.12551056, &[0.0066313385, -0.0008666183, -0.0023167906]),
        (
            "model.llm.layers.3.controllers.attn.forget_weight",
            0.03722027,
            &[-0.00080104097, -0.0015226345, -0.0033291501],
        ),
        (
            "model.llm.layers.3.controllers.attn.gain_weight",
            0.031935647,
            &[-0.00016248437, -0.00075285096, -0.0031239055],
        ),
        ("model.llm.layers.3.controllers.k.down_weight", 0.054785717, &[0.0021434473, -0.0019902568, -0.00041483305]),
        ("model.llm.layers.3.controllers.k.forget_weight", 0.0135467835, &[-0.0025231964, -4.623975e-05, 0.0015054115]),
        ("model.llm.layers.3.controllers.k.gain_weight", 0.017606061, &[-0.0014268772, -0.000230345, 0.0012103306]),
        ("model.llm.layers.3.controllers.mlp.down_weight", 2.175412, &[0.023908364, -0.007246986, -0.23518538]),
        ("model.llm.layers.3.controllers.mlp.forget_weight", 0.44879714, &[0.017504027, -0.011912744, -0.014239606]),
        ("model.llm.layers.3.controllers.mlp.gain_weight", 0.52077657, &[-0.0061445595, 0.0051949923, 0.023490045]),
        ("model.llm.layers.3.controllers.v.down_weight", 0.47433192, &[-0.020979356, -0.00073526666, -0.0049874457]),
        ("model.llm.layers.3.controllers.v.forget_weight", 0.052328456, &[-0.0018869828, 0.0032796087, 0.0005143221]),
        ("model.llm.layers.3.controllers.v.gain_weight", 0.11210543, &[0.0013269461, -0.0048164474, -0.0018573981]),
        ("model.llm.layers.3.k_input_norm.weight", 0.021848336, &[-0.0056310683, -0.0012787944, -0.007123382]),
        ("model.llm.layers.3.mlp.experts.w13_weight", 6.168632, &[0.0072193854, -0.009142733, -0.023268297]),
        ("model.llm.layers.3.mlp.experts.w2_weight", 3.5192928, &[0.018794045, 0.023185642, 0.044407558]),
        ("model.llm.layers.3.mlp.gate.bias", 0.0, &[0.0, 0.0, 0.0]),
        ("model.llm.layers.3.mlp.gate.weight", 0.58373076, &[0.029239058, -0.0833215, 0.069849685]),
        ("model.llm.layers.3.mlp_norm.weight", 1.0919666, &[-0.22928937, 0.21599835, -0.020815823]),
        ("model.llm.layers.3.mlp_sconv.weight", 0.83600664, &[-0.06983175, 0.014750058, 0.07472941]),
        ("model.llm.layers.3.q_input_norm.weight", 0.042308766, &[-0.008171105, -0.0068456037, -0.004829835]),
        ("model.llm.layers.3.v_input_norm.weight", 0.0897834, &[-0.008819098, -0.00062486326, 0.003696383]),
        ("model.llm.layers.4.attn.attn_sink_proj.weight", 0.31013444, &[-0.006920678, -0.03049072, -0.012098183]),
        ("model.llm.layers.4.attn.gate_attn.weight", 0.810537, &[0.03100694, 0.02335243, 0.03873715]),
        ("model.llm.layers.4.attn.k_sconv.weight", 0.15396573, &[0.029842447, 0.014488355, -0.008285691]),
        ("model.llm.layers.4.attn.q_l2_clip.scale", 0.10532455, &[-0.0048956512, -0.0016594781, -0.006704305]),
        ("model.llm.layers.4.attn.rel_l2_clip.scale", 0.112504184, &[0.046973377, 0.045438394, 0.0020287507]),
        ("model.llm.layers.4.attn.rel_logits_proj.proj", 0.27889174, &[0.013772043, 0.03465129, -0.0054664216]),
        ("model.llm.layers.4.attn.v_sconv.weight", 0.65407896, &[-0.080550775, 0.009403851, -0.041056164]),
        ("model.llm.layers.4.attn.wk_dv.weight", 0.6727582, &[0.01210928, -0.013135321, -0.0071525676]),
        ("model.llm.layers.4.attn.wo_ud.weight", 2.468657, &[-0.022306835, -0.05717704, -0.037393622]),
        ("model.llm.layers.4.attn.wq_du.weight", 0.68271565, &[-0.0063184947, 0.0119943125, -0.005185266]),
        ("model.llm.layers.4.attn.wr_du.weight", 1.1474075, &[-0.040931694, -0.066304654, 0.0062451717]),
        ("model.llm.layers.4.attn.wv_dv.weight", 2.3981242, &[-0.13102295, -0.013057031, 0.010780644]),
        ("model.llm.layers.4.attn_norm.weight", 0.40527773, &[0.061232056, -0.03259891, 0.07013388]),
        ("model.llm.layers.4.attn_sconv.weight", 0.71268874, &[-0.026470935, 0.026243178, -0.022029772]),
        ("model.llm.layers.4.controllers.attn.down_weight", 0.9174777, &[0.0028742487, -0.019207317, -0.05420325]),
        (
            "model.llm.layers.4.controllers.attn.forget_weight",
            0.19583793,
            &[-0.008103073, -0.0028036726, -0.0024415555],
        ),
        ("model.llm.layers.4.controllers.attn.gain_weight", 0.20836447, &[0.0035477835, -0.003075338, 0.0054287584]),
        ("model.llm.layers.4.controllers.k.down_weight", 0.22670776, &[-0.008684624, -0.0016302596, -0.0005419991]),
        ("model.llm.layers.4.controllers.k.forget_weight", 0.059344295, &[-0.0025340924, 0.0004427296, -0.0047018733]),
        ("model.llm.layers.4.controllers.k.gain_weight", 0.086224124, &[-5.787409e-05, -0.0035310953, 0.00022497633]),
        ("model.llm.layers.4.controllers.mlp.down_weight", 1.6592015, &[0.11884456, 0.0594464, 0.034598548]),
        ("model.llm.layers.4.controllers.mlp.forget_weight", 0.3578633, &[0.0052141296, -0.010856869, -0.005705924]),
        ("model.llm.layers.4.controllers.mlp.gain_weight", 0.4904615, &[-0.014861714, -0.018841643, 0.009604045]),
        ("model.llm.layers.4.controllers.v.down_weight", 0.9669037, &[-0.017304096, 0.1057943, 0.074273765]),
        ("model.llm.layers.4.controllers.v.forget_weight", 0.23332869, &[-0.00079318736, 0.0011993519, 0.0023631975]),
        ("model.llm.layers.4.controllers.v.gain_weight", 0.3094418, &[-0.0035751688, 0.01209604, 0.001894105]),
        ("model.llm.layers.4.mlp.experts.w13_weight", 4.8275003, &[-0.0021983562, 0.053065717, 0.024034739]),
        ("model.llm.layers.4.mlp.experts.w2_weight", 3.0207684, &[0.00421547, 0.02783828, -0.0046330276]),
        ("model.llm.layers.4.mlp.gate.bias", 0.0, &[0.0, 0.0, 0.0]),
        ("model.llm.layers.4.mlp.gate.weight", 0.37553218, &[0.0062366175, 0.028259126, 0.011541501]),
        ("model.llm.layers.4.mlp_norm.weight", 0.86798507, &[0.01435596, -0.3517003, 0.035085104]),
        ("model.llm.layers.4.mlp_sconv.weight", 0.83447665, &[-0.05888654, -0.060200825, -0.05843196]),
        ("model.llm.layers.5.attn.attn_sink_proj.weight", 0.15757854, &[-0.009269776, 0.0017394192, -0.0013233339]),
        ("model.llm.layers.5.attn.gate_attn.weight", 0.099744454, &[-0.0022290852, 0.002044494, -0.0009220389]),
        ("model.llm.layers.5.attn.k_sconv.static_w", 0.013393088, &[-0.0033751463, -0.0012092069, 0.007121158]),
        ("model.llm.layers.5.attn.k_sconv.weight_proj", 0.062746614, &[-0.00704375, 0.0037796297, 0.0024627754]),
        ("model.llm.layers.5.attn.q_l2_clip.scale", 0.018580649, &[-0.00019288051, 0.0010520457, 0.0031361762]),
        ("model.llm.layers.5.attn.rel_l2_clip.scale", 0.037149176, &[0.016316192, -0.016314188, 0.0023517683]),
        ("model.llm.layers.5.attn.rel_logits_proj.proj", 0.049760588, &[-0.014430759, 0.0011269939, -0.0034607684]),
        ("model.llm.layers.5.attn.v_sconv.static_w", 0.12626965, &[0.06586299, -0.023121843, 0.022471141]),
        ("model.llm.layers.5.attn.v_sconv.weight_proj", 1.2602732, &[-0.22341974, 0.039322972, -0.18317765]),
        ("model.llm.layers.5.attn.wk_dv.weight", 0.14273742, &[-0.01392671, 0.01221724, 0.008199134]),
        ("model.llm.layers.5.attn.wo_ud.weight", 0.26446244, &[0.013523908, 1.3541838e-05, 0.008405017]),
        ("model.llm.layers.5.attn.wq_du.weight", 0.10754626, &[-0.0027624553, 0.0033179875, -0.0053514084]),
        ("model.llm.layers.5.attn.wr_du.weight", 0.17715026, &[0.0052140807, -0.013458843, -0.0070110867]),
        ("model.llm.layers.5.attn.wv_dv.weight", 0.48310733, &[-0.010451556, 0.013348952, 0.006074715]),
        ("model.llm.layers.5.attn_sconv.weight", 0.06396793, &[-0.00042312162, 0.00465798, 0.0035816855]),
        ("model.llm.layers.5.controllers.attn.down_weight", 0.07803781, &[0.0028440976, -0.00018766221, -0.002592856]),
        (
            "model.llm.layers.5.controllers.attn.forget_weight",
            0.014194227,
            &[-0.0005360297, 0.00028247604, -4.6533554e-05],
        ),
        (
            "model.llm.layers.5.controllers.attn.gain_weight",
            0.023326637,
            &[-0.00023690678, 0.00040992044, -0.00065416907],
        ),
        ("model.llm.layers.5.controllers.k.down_weight", 0.05545498, &[-0.0017221126, 0.002869028, 0.00087252236]),
        ("model.llm.layers.5.controllers.k.forget_weight", 0.010288171, &[-0.0011604519, 0.0004997094, 5.1864045e-05]),
        ("model.llm.layers.5.controllers.k.gain_weight", 0.012784118, &[-1.7046194e-05, 0.0003481115, 2.1149639e-05]),
        ("model.llm.layers.5.controllers.mlp.down_weight", 1.1000092, &[0.010018736, 0.024649559, 0.030241914]),
        (
            "model.llm.layers.5.controllers.mlp.forget_weight",
            0.28727987,
            &[-0.0042760377, -0.0045772996, -0.0019588291],
        ),
        ("model.llm.layers.5.controllers.mlp.gain_weight", 0.38956735, &[0.009012955, 0.013676417, -0.0043055853]),
        ("model.llm.layers.5.controllers.v.down_weight", 0.3131824, &[-0.009033636, 0.005550464, -0.022957802]),
        ("model.llm.layers.5.controllers.v.forget_weight", 0.07541791, &[-0.0021387844, -0.0028234927, -0.0013233878]),
        ("model.llm.layers.5.controllers.v.gain_weight", 0.12637186, &[-0.0002628186, 0.0003396713, -2.2319327e-05]),
        ("model.llm.layers.5.k_input_norm.weight", 0.026657635, &[0.0021034346, -0.0008199761, -0.000953026]),
        ("model.llm.layers.5.mlp.experts.w13_weight", 2.9747245, &[0.01695568, -0.010756346, -0.0142047815]),
        ("model.llm.layers.5.mlp.experts.w2_weight", 2.0027773, &[-0.012900945, -0.0060665198, 0.0533313]),
        ("model.llm.layers.5.mlp.gate.bias", 0.0, &[0.0, 0.0, 0.0]),
        ("model.llm.layers.5.mlp.gate.weight", 0.40273973, &[-0.055213697, 0.040368274, 0.009923889]),
        ("model.llm.layers.5.mlp_norm.weight", 0.5568921, &[-0.012083232, 0.04527709, -0.03709463]),
        ("model.llm.layers.5.mlp_sconv.weight", 0.5254076, &[0.035783004, 0.003697645, -0.089381754]),
        ("model.llm.layers.5.q_input_norm.weight", 0.04506682, &[-0.005743926, 0.00045356015, 0.0058436375]),
        ("model.llm.layers.5.v_input_norm.weight", 0.1789293, &[-0.044584848, -0.0064458125, -0.019557152]),
        ("model.llm.layers.6.attn.attn_sink_proj.weight", 0.0954722, &[-0.0115272645, -0.010911525, 0.0071459115]),
        ("model.llm.layers.6.attn.gate_attn.weight", 0.23280787, &[0.0065406924, -0.0016173286, -0.008630589]),
        ("model.llm.layers.6.attn.k_sconv.weight", 0.031961877, &[0.000370474, -0.0007613116, -0.0010145998]),
        ("model.llm.layers.6.attn.q_l2_clip.scale", 0.017342787, &[-0.00050663075, -0.00042252304, -0.0009710996]),
        ("model.llm.layers.6.attn.rel_l2_clip.scale", 0.065488994, &[0.00040914642, -0.0022143803, -0.009333216]),
        ("model.llm.layers.6.attn.rel_logits_proj.proj", 0.11660047, &[0.044119187, 0.0137486495, -0.032454435]),
        ("model.llm.layers.6.attn.v_sconv.weight", 0.17344199, &[-0.008784518, 0.016831672, 0.02157493]),
        ("model.llm.layers.6.attn.wk_dv.weight", 0.11916894, &[0.0018976836, -0.0017034615, 0.0023151094]),
        ("model.llm.layers.6.attn.wo_ud.weight", 0.59285915, &[0.0038352339, 0.0031708549, 0.002994557]),
        ("model.llm.layers.6.attn.wq_du.weight", 0.13326901, &[0.0009019896, 0.0002941199, 0.00033260614]),
        ("model.llm.layers.6.attn.wr_du.weight", 0.3375649, &[0.012152613, -0.0036688026, 0.0025857491]),
        ("model.llm.layers.6.attn.wv_dv.weight", 0.6906214, &[-0.029890426, -0.02273907, -0.029246971]),
        ("model.llm.layers.6.attn_norm.weight", 0.16862123, &[-0.015301992, -0.022125442, 0.047103703]),
        ("model.llm.layers.6.attn_sconv.weight", 0.19907184, &[-0.010827609, -0.023507847, -0.012030547]),
        ("model.llm.layers.6.mlp.experts.w13_weight", 1.2363255, &[-0.011031404, -0.011479268, -0.0043036286]),
        ("model.llm.layers.6.mlp.experts.w2_weight", 0.82264274, &[0.0066396953, -0.0028736265, 0.000102399674]),
        ("model.llm.layers.6.mlp.gate.bias", 0.0, &[0.0, 0.0, 0.0]),
        ("model.llm.layers.6.mlp.gate.weight", 0.13137001, &[2.7440517e-05, 0.006515532, 0.019622745]),
        ("model.llm.layers.6.mlp_norm.weight", 0.18145384, &[-0.04223981, -0.03909542, -0.010838505]),
        ("model.llm.layers.6.mlp_sconv.weight", 0.2748712, &[0.025416143, 0.017729975, 0.030823616]),
        ("model.llm.layers.7.attn.attn_sink_proj.weight", 0.015550975, &[0.00028827335, -0.0007159634, 0.0011456694]),
        ("model.llm.layers.7.attn.gate_attn.weight", 0.027140275, &[-0.00018718741, 0.00033332402, 0.00026193095]),
        ("model.llm.layers.7.attn.k_sconv.static_w", 0.0036153952, &[3.276694e-05, -0.00013464461, -1.2562377e-05]),
        ("model.llm.layers.7.attn.k_sconv.weight_proj", 0.020521142, &[-0.00070035446, -0.00058619806, -0.00092737214]),
        ("model.llm.layers.7.attn.q_l2_clip.scale", 0.0036883473, &[0.000405949, -2.8787217e-05, -0.00023380073]),
        ("model.llm.layers.7.attn.rel_l2_clip.scale", 0.0038670076, &[-0.00031645864, -0.00011503813, 0.0012259183]),
        ("model.llm.layers.7.attn.rel_logits_proj.proj", 0.010282493, &[0.0016469993, 0.0020700893, -0.000840513]),
        ("model.llm.layers.7.attn.v_sconv.static_w", 0.011021288, &[0.00025495305, 0.002664948, -0.0018229076]),
        ("model.llm.layers.7.attn.v_sconv.weight_proj", 0.071591906, &[-0.0003874014, -0.0028522105, 0.0034620839]),
        ("model.llm.layers.7.attn.wk_dv.weight", 0.023388023, &[-1.648389e-06, -5.8739092e-06, 0.00035060412]),
        ("model.llm.layers.7.attn.wo_ud.weight", 0.071833715, &[-0.0014511658, -0.0062768757, -0.003447688]),
        ("model.llm.layers.7.attn.wq_du.weight", 0.020101314, &[0.00013090062, -0.00032680994, 0.0004376706]),
        ("model.llm.layers.7.attn.wr_du.weight", 0.037180796, &[0.0004188111, -0.0017579866, 0.0005483515]),
        ("model.llm.layers.7.attn.wv_dv.weight", 0.07250131, &[0.0005958862, -0.0006608127, -0.0010208653]),
        ("model.llm.layers.7.attn_sconv.weight", 0.02558977, &[0.001397766, 0.0014938936, 0.0014337428]),
        ("model.llm.layers.7.k_input_norm.weight", 0.0045937514, &[0.00029920656, -0.0012878736, -2.375084e-05]),
        ("model.llm.layers.7.mlp.experts.w13_weight", 1.1127671, &[-0.0067474223, -0.033677083, 0.049458552]),
        ("model.llm.layers.7.mlp.experts.w2_weight", 0.8035065, &[0.004031355, 0.0044093467, -0.046813373]),
        ("model.llm.layers.7.mlp.gate.bias", 0.0, &[0.0, 0.0, 0.0]),
        ("model.llm.layers.7.mlp.gate.weight", 0.06498617, &[0.010296827, -0.0066361018, -0.0014621875]),
        ("model.llm.layers.7.mlp_norm.weight", 0.17605105, &[0.026479349, -0.0023329773, 0.04492807]),
        ("model.llm.layers.7.mlp_sconv.weight", 0.23523831, &[0.011227549, 0.020743996, 0.0029655322]),
        ("model.llm.layers.7.q_input_norm.weight", 0.008867762, &[0.0010680736, 0.0009932978, 0.00019823885]),
        ("model.llm.layers.7.v_input_norm.weight", 0.019778442, &[-0.00024010082, 0.005681996, 0.0032149435]),
        ("model.llm.middle_cycle_cross_pass_kv_queries", 0.11216471, &[0.0008312168, 0.019300157, 0.0065014767]),
        ("model.llm.middle_cycle_cross_pass_queries", 0.3348509, &[-0.0009644777, 0.011708567, 0.00589022]),
        ("model.llm.norm.weight", 0.30277503, &[0.009862576, -0.08493154, -0.098451026]),
        ("model.llm.unembed.weight", 1.7488251, &[0.0017246037, 0.002532447, -0.0019391123]),
    ];

    /// Returns a configuration that is small enough for the reference backend and still has every structural
    /// feature of the release, computing in `f32`. It has 8 layers in blocks of two (a local layer followed by a
    /// global one), a loop that applies layers `2..6` twice, two query heads per key and value head, a local window
    /// of 4 positions, relative position entries for 6 distances, and routing to 2 of 4 experts. The first 6 layers
    /// are the NVFP4 group.
    pub(crate) fn test_configuration() -> Inkling2Configuration {
        let json = r#"{"text_config": {
            "hidden_size": 32, "num_hidden_layers": 8, "vocab_size": 64, "unpadded_vocab_size": 60,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2, "swa_head_dim": 8, "d_rel": 4, "rel_extent": 6, "sliding_window_size": 4,
            "local_layer_ids": [0, 2, 4, 6], "log_scaling_n_floor": 4, "log_scaling_alpha": 0.1,
            "rms_norm_eps": 1e-06, "qk_l2_clip_threshold": 2.5, "rel_l2_clip_threshold": 1.5,
            "sconv_kernel_size": 4, "final_logit_softcapping": 30.0, "n_layers_per_attn_res_block": 2,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "intermediate_size": 32, "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0, "route_scale": 8.0,
            "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 2, "gate_rank": 8}
        }}"#;
        Inkling2Configuration::from_json(json)
            .unwrap()
            .with_nvfp4_layer_count(6)
            .with_compute_type(DataType::F32)
    }

    /// Returns a decoder with pseudo-random parameters for `configuration`.
    pub(crate) fn test_decoder(configuration: &Inkling2Configuration) -> Inkling2Decoder<Array> {
        Inkling2Decoder::random(configuration, 7).unwrap()
    }

    /// Returns inputs for one sequence that consists of the provided documents followed by `padding` padding
    /// positions.
    fn test_inputs(documents: &[&[i32]], padding: usize) -> Inkling2Inputs<Array> {
        let (mut tokens, mut positions, mut segments, mut valid) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (segment, document) in documents.iter().enumerate() {
            for (position, &token) in document.iter().enumerate() {
                tokens.push(token);
                positions.push(position as i32);
                segments.push(segment as i32);
                valid.push(true);
            }
        }
        for _ in 0..padding {
            tokens.push(0);
            positions.push(0);
            segments.push(documents.len() as i32);
            valid.push(false);
        }
        let length = tokens.len();
        let integers =
            |values: &[i32]| Array::from_elements(ArrayType::new_static(DataType::I32, [1, length]), values).unwrap();
        Inkling2Inputs::new(
            integers(&tokens),
            integers(&positions),
            integers(&segments),
            Array::from_elements(ArrayType::new_static(DataType::Boolean, [1, length]), &valid).unwrap(),
        )
    }

    /// Returns the logits of `position` of the single sequence of `logits`.
    fn position_logits(logits: &Array, position: usize) -> Vec<f32> {
        let vocabulary = logits.r#type().static_shape().unwrap().dimensions()[2];
        logits.elements::<f32>().unwrap()[position * vocabulary..(position + 1) * vocabulary].to_vec()
    }

    /// Asserts that two lists of `f32` values agree within `tolerance`.
    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!((actual - expected).abs() <= tolerance, "element {index}: {actual} is not close to {expected}");
        }
    }

    #[test]
    fn test_inkling2_inputs_new() {
        let inputs = test_inputs(&[&[3, 4], &[5]], 1);
        assert_eq!(inputs.tokens().elements::<i32>().unwrap(), vec![3, 4, 5, 0]);
        assert_eq!(inputs.positions().elements::<i32>().unwrap(), vec![0, 1, 0, 0]);
        assert_eq!(inputs.segments().elements::<i32>().unwrap(), vec![0, 0, 1, 2]);
        assert_eq!(inputs.valid().elements::<bool>().unwrap(), vec![true, true, true, false]);
    }

    #[test]
    fn test_inkling2_decoder_specification() {
        // The specification of the release configuration names one checkpoint array per leaf.
        let specification = Inkling2Decoder::specification(&Inkling2Configuration::inkling_2_mini());
        let leaves = specification.named_parameters().collect::<Vec<_>>();
        let leaf = |path: &str| {
            let (_, leaf) = leaves.iter().find(|(leaf_path, _)| leaf_path.to_string() == path).unwrap();
            (leaf.name().to_string(), leaf.r#type().static_shape().unwrap().dimensions().to_vec())
        };
        assert_eq!(leaf("$.embedding"), ("model.llm.embed.weight".to_string(), vec![201_024, 2304]));
        assert_eq!(
            leaf("$.layers[1].attention_queries.0"),
            ("model.llm.attn_res.attn_queries.0".to_string(), vec![2304]),
        );
        assert_eq!(
            leaf("$.layers[5].attention_queries.0"),
            ("model.llm.attn_res.attn_queries.4".to_string(), vec![3, 2304]),
        );
        assert_eq!(
            leaf("$.layers[5].attention_norms[2]"),
            ("model.llm.layers.5.v_input_norm.weight".to_string(), vec![2304]),
        );
        assert_eq!(
            leaf("$.layers[5].attention.key_convolution.generator.0"),
            ("model.llm.layers.5.attn.k_sconv.weight_proj".to_string(), vec![32, 2304]),
        );
        assert_eq!(
            leaf("$.layers[4].attention.key_convolution.taps"),
            ("model.llm.layers.4.attn.k_sconv.weight".to_string(), vec![1024, 1, 4]),
        );
        assert_eq!(
            leaf("$.layers[23].experts.gate_up"),
            ("model.llm.layers.23.mlp.experts.w13_weight".to_string(), vec![128, 2304, 2304]),
        );
        assert_eq!(
            leaf("$.layers[6].loop_gates.0.experts.down"),
            ("model.llm.layers.6.controllers.mlp.down_weight".to_string(), vec![64, 4608]),
        );
        assert_eq!(
            leaf("$.loop_key_value_queries.0"),
            ("model.llm.middle_cycle_cross_pass_kv_queries".to_string(), vec![24, 2304]),
        );

        // Only core layers have loop gates, and only the first layer has no attention queries.
        assert!(leaves.iter().all(|(path, _)| !path.to_string().starts_with("$.layers[5].loop_gates.")));
        assert!(leaves.iter().all(|(path, _)| !path.to_string().starts_with("$.layers[0].attention_queries.")));

        // The release stores 822 language model arrays, of which 128 are the scales of packed expert projections
        // (i.e., three per NVFP4 projection of 20 layers and one per MXFP8 projection of 4 layers).
        assert_eq!(leaves.len(), 822 - (20 * 2 * 3 + 4 * 2));
    }

    #[test]
    fn test_inkling2_decoder_random() {
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        assert_eq!(decoder, Inkling2Decoder::random(&configuration, 7).unwrap());
        assert_ne!(decoder, Inkling2Decoder::random(&configuration, 8).unwrap());
        let specification = Inkling2Decoder::specification(&configuration);
        for (leaf, array) in specification.parameters().zip(decoder.parameters()) {
            assert_eq!(array.r#type().as_ref(), leaf.r#type());
            let values = array.elements::<f32>().unwrap();
            let low = values.iter().fold(f32::MAX, |low, &value| low.min(value));
            let high = values.iter().fold(f32::MIN, |high, &value| high.max(value));
            if leaf.name().ends_with("norm.weight") || leaf.name().ends_with("clip.scale") {
                assert!((0.9..=1.1).contains(&low) && (0.9..=1.1).contains(&high), "{}", leaf.name());
            } else {
                assert!(low < 0.0 && high > 0.0 && low >= -1.5 && high <= 1.5, "{}", leaf.name());
            }
        }
    }

    #[test]
    fn test_inkling2_decoder_logits() {
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        let first = [11, 42, 7, 30, 5, 9, 13, 2, 57, 15];
        let second = [8, 59, 23];
        let logits = decoder
            .logits(&configuration, &FloatingExpertProjector::new(DataType::F32), &test_inputs(&[&first], 0))
            .unwrap();
        assert_eq!(logits.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [1, 10, 60]));
        let values = logits.elements::<f32>().unwrap();
        assert!(values.iter().all(|value| value.is_finite() && value.abs() < 30.0));
        assert!(values.iter().any(|value| value.abs() > 0.1));

        // The decoder is causal, so changing a token leaves the logits of the earlier positions unchanged.
        let mut changed = first;
        changed[6] = 14;
        let changed_logits = decoder
            .logits(&configuration, &FloatingExpertProjector::new(DataType::F32), &test_inputs(&[&changed], 0))
            .unwrap();
        for position in 0..6 {
            assert_eq!(position_logits(&changed_logits, position), position_logits(&logits, position));
        }
        assert_ne!(position_logits(&changed_logits, 6), position_logits(&logits, 6));

        // Padding and a second document that is packed into the same sequence do not change the logits of a
        // document, and the packed document gets the logits that it gets on its own.
        let packed = decoder
            .logits(&configuration, &FloatingExpertProjector::new(DataType::F32), &test_inputs(&[&first, &second], 2))
            .unwrap();
        let alone = decoder
            .logits(&configuration, &FloatingExpertProjector::new(DataType::F32), &test_inputs(&[&second], 0))
            .unwrap();
        for position in 0..10 {
            assert_close(&position_logits(&packed, position), &position_logits(&logits, position), 1e-4);
        }
        for position in 0..3 {
            assert_close(&position_logits(&packed, 10 + position), &position_logits(&alone, position), 1e-4);
        }
    }

    #[test]
    fn test_inkling2_decoder_logits_compute_type() {
        // Computing at `bf16` rounds parameters and activations at every layer, so the logits are still `f32` values
        // over the usable vocabulary but they differ from the ones that are computed at `f32`.
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        let inputs = test_inputs(&[&[11, 42, 7, 30]], 0);
        let reference = decoder.logits(&configuration, &FloatingExpertProjector::new(DataType::F32), &inputs).unwrap();
        let narrow = decoder
            .logits(
                &configuration.with_compute_type(DataType::BF16),
                &FloatingExpertProjector::new(DataType::BF16),
                &inputs,
            )
            .unwrap();
        assert_eq!(narrow.r#type(), reference.r#type());
        assert_ne!(narrow, reference);
        assert!(narrow.elements::<f32>().unwrap().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_inkling2_decoder_logits_block_scaled_experts() {
        // With block-scaled expert projections, the first 6 layers contract NVFP4 values and the last 2 contract
        // MXFP8 values. The forward pass rounds to nearest, so it does not depend on the random generator state, and
        // its logits differ from the floating-point ones.
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        let inputs = test_inputs(&[&[11, 42, 7, 30]], 0);
        let projections = BlockScaledProjections::<EagerContext<Array, ArrayOperation<Array>>>::new().unwrap();
        let projector = |seed: u64| {
            let random_state = Array::from_elements(RandomAlgorithm::ThreeFry.state_type(), &[seed, 0]).unwrap();
            BlockScaledExpertProjector::new(
                &projections,
                configuration.nvfp4_layer_count(),
                DataType::F32,
                random_state,
            )
        };
        assert_eq!(projector(3).format(5), BlockScaledFormat::Nvfp4);
        assert_eq!(projector(3).format(6), BlockScaledFormat::Mxfp8);
        let logits = decoder.logits(&configuration, &projector(3), &inputs).unwrap();
        let reference = decoder.logits(&configuration, &FloatingExpertProjector::new(DataType::F32), &inputs).unwrap();
        assert_eq!(logits.r#type(), reference.r#type());
        assert_ne!(logits, reference);
        assert!(logits.elements::<f32>().unwrap().iter().all(|value| value.is_finite()));
        assert_eq!(decoder.logits(&configuration, &projector(4), &inputs).unwrap(), logits);
    }

    #[test]
    fn test_inkling2_decoder_logits_with_diagnostics() {
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        let inputs = test_inputs(&[&[11, 42, 7]], 0);
        let (logits, diagnostics) = decoder
            .logits_with_diagnostics(&configuration, &FloatingExpertProjector::new(DataType::F32), &inputs)
            .unwrap();
        assert_eq!(
            logits,
            decoder.logits(&configuration, &FloatingExpertProjector::new(DataType::F32), &inputs).unwrap()
        );

        // The loop applies the 4 core layers twice, which gives 12 layer applications of the 8 layers.
        assert_eq!(configuration.layer_schedule(), vec![0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]);
        assert_eq!(diagnostics.states().len(), 12);
        assert_eq!(diagnostics.routing_weights().len(), 12);
        for state in diagnostics.states() {
            assert_eq!(state.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [1, 3, 32]));
        }

        // Every token is routed to exactly 2 experts whose weights sum to the route scale.
        for weights in diagnostics.routing_weights() {
            assert_eq!(weights.r#type().as_ref(), &ArrayType::new_static(DataType::F32, [3, 4]));
            for token in weights.elements::<f32>().unwrap().as_chunks::<4>().0 {
                assert_eq!(token.iter().filter(|&&weight| weight > 0.0).count(), 2);
                assert!((token.iter().sum::<f32>() - 8.0).abs() < 1e-4);
            }
        }

        // The second pass of the loop sees messages from the first pass, so it does not repeat the first pass.
        assert_ne!(diagnostics.states()[2], diagnostics.states()[6]);
    }

    #[test]
    fn test_inkling2_decoder_loss() {
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        let inputs = test_inputs(&[&[11, 42, 7, 30]], 1);
        let labels = Array::from_elements(ArrayType::new_static(DataType::I32, [1, 5]), &[42i32, 7, 30, 0, 0]).unwrap();
        let weights =
            |values: &[f32]| Array::from_elements(ArrayType::new_static(DataType::F32, [1, 5]), values).unwrap();
        let loss = decoder
            .loss(
                &configuration,
                &FloatingExpertProjector::new(DataType::F32),
                &inputs,
                &labels,
                &weights(&[0.0, 1.0, 1.0, 0.0, 0.0]),
            )
            .unwrap();
        assert_eq!(loss.r#type().as_ref(), &ArrayType::scalar(DataType::F32));

        // The loss is the mean over the supervised positions of the negative log-probabilities of their labels.
        let logits = decoder.logits(&configuration, &FloatingExpertProjector::new(DataType::F32), &inputs).unwrap();
        let negative_log_probability = |position: usize, label: usize| {
            let logits = position_logits(&logits, position);
            let maximum = logits.iter().fold(f32::MIN, |maximum, &logit| maximum.max(logit));
            let normalizer = logits.iter().map(|logit| (logit - maximum).exp()).sum::<f32>().ln() + maximum;
            normalizer - logits[label]
        };
        let expected = (negative_log_probability(1, 7) + negative_log_probability(2, 30)) / 2.0;
        assert_close(&loss.elements::<f32>().unwrap(), &[expected], 1e-4);

        // Without supervised positions the loss is zero rather than undefined.
        assert_eq!(
            decoder
                .loss(
                    &configuration,
                    &FloatingExpertProjector::new(DataType::F32),
                    &inputs,
                    &labels,
                    &weights(&[0.0; 5])
                )
                .unwrap()
                .elements::<f32>()
                .unwrap(),
            vec![0.0],
        );
    }

    #[test]
    fn test_inkling2_decoder_reference_parity() {
        // The reference constants hold the outputs of the pinned floating reference implementation (i.e., the oracle
        // of NVIDIA NeMo AutoModel at commit `bdf1fff2e8e8ef3e4035308820495e593b7df089`) for the test decoder computing
        // at `f32`, on one document of 12 tokens. They were produced by exporting the test decoder with
        // `export_inkling2_decoder` and evaluating the oracle on that checkpoint.
        let configuration = test_configuration();
        let decoder = test_decoder(&configuration);
        let projector = FloatingExpertProjector::new(DataType::F32);
        let inputs = test_inputs(&[&[11, 42, 7, 30, 5, 9, 13, 2, 57, 15, 21, 3]], 0);
        let (logits, diagnostics) = decoder.logits_with_diagnostics(&configuration, &projector, &inputs).unwrap();

        // The logits and the size of the block state after every layer application agree.
        assert_close(&logits.elements::<f32>().unwrap(), &REFERENCE_LOGITS, 2e-3);
        assert_eq!(REFERENCE_STATE_NORMS.len(), diagnostics.states().len());
        for (state, expected_norm) in diagnostics.states().iter().zip(&REFERENCE_STATE_NORMS) {
            let norm = state.elements::<f32>().unwrap().iter().map(|value| value * value).sum::<f32>().sqrt();
            assert!((norm - expected_norm).abs() <= 1e-3 * expected_norm, "{norm} is not close to {expected_norm}");
        }

        // The first token is routed to the same experts in every layer application.
        for (weights, expected) in diagnostics.routing_weights().iter().zip(&REFERENCE_ROUTING) {
            let weights = weights.elements::<f32>().unwrap();
            let selected = (0..4).filter(|&expert| weights[expert] > 0.0).collect::<Vec<_>>();
            assert_eq!(selected, expected.to_vec());
        }

        // The loss and the gradient of every parameter agree, including the gradients of the parameters of the
        // looped layers, which accumulate the contributions of both of their applications.
        let labels = Array::from_elements(
            ArrayType::new_static(DataType::I32, [1, 12]),
            &[42i32, 7, 30, 5, 9, 13, 2, 57, 15, 21, 3, 0],
        )
        .unwrap();
        let weights = Array::from_elements(
            ArrayType::new_static(DataType::F32, [1, 12]),
            &[1.0f32, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        )
        .unwrap();
        let (loss, gradients) = differentiate_at(decoder)
            .with_captures((inputs, labels, weights))
            .value_and_gradient(|decoder, (inputs, labels, weights)| {
                decoder.loss(&configuration, &projector, &inputs, &labels, &weights)
            })
            .unwrap();
        assert_close(&loss.elements::<f32>().unwrap(), &[REFERENCE_LOSS], 1e-4);
        let expected_gradients = REFERENCE_GRADIENTS
            .iter()
            .map(|&(name, norm, head)| (name, (norm, head)))
            .collect::<HashMap<_, _>>();
        let specification = Inkling2Decoder::specification(&configuration);
        let mut compared = 0;
        for (leaf, gradient) in specification.parameters().zip(gradients.parameters()) {
            let (expected_norm, head) = expected_gradients[leaf.name()];
            let gradient = gradient.elements::<f32>().unwrap();
            let norm = gradient.iter().map(|value| value * value).sum::<f32>().sqrt();
            let tolerance = 5e-3 * expected_norm + 1e-6;
            assert!((norm - expected_norm).abs() <= tolerance, "{}: norm {norm} vs {expected_norm}", leaf.name());
            for (index, (actual, expected)) in gradient.iter().zip(head).enumerate() {
                assert!((actual - expected).abs() <= tolerance, "{}[{index}]: {actual} vs {expected}", leaf.name());
            }
            compared += 1;
        }
        assert_eq!(compared, 246);
    }
}
