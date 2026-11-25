//! Configuration of the Inkling 2 text decoder.

use ryft_core::DataType;
use thiserror::Error;

use crate::json::JsonValue;

// TODO(eaplatanios): Review this module.

/// Error returned when an Inkling 2 configuration is invalid or cannot be parsed.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum Inkling2ConfigurationError {
    #[error("invalid Inkling 2 configuration JSON: {message}")]
    InvalidJson { message: String },

    #[error("invalid Inkling 2 configuration: {message}")]
    Invalid { message: String },
}

/// Configuration of the fixed loop of an Inkling 2 decoder, which applies a contiguous range of core layers several
/// times with shared weights.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Inkling2LoopConfiguration {
    /// Index of the first core layer.
    core_start: usize,

    /// Index after the last core layer.
    core_end: usize,

    /// Number of times that the core layers are applied.
    pass_count: usize,

    /// Rank of the low rank gates that merge the messages of the previous pass into the inputs of a core layer.
    gate_rank: usize,
}

impl Inkling2LoopConfiguration {
    /// Creates a new [`Inkling2LoopConfiguration`]. [`Inkling2Configuration::validate`] checks it against the layer
    /// layout of the decoder.
    #[inline]
    pub fn new(core_start: usize, core_end: usize, pass_count: usize, gate_rank: usize) -> Self {
        Self { core_start, core_end, pass_count, gate_rank }
    }

    /// Returns the index of the first core layer.
    #[inline]
    pub fn core_start(&self) -> usize {
        self.core_start
    }

    /// Returns the index after the last core layer.
    #[inline]
    pub fn core_end(&self) -> usize {
        self.core_end
    }

    /// Returns the number of times that the core layers are applied.
    #[inline]
    pub fn pass_count(&self) -> usize {
        self.pass_count
    }

    /// Returns the rank of the low rank gates that merge the messages of the previous pass into a core layer.
    #[inline]
    pub fn gate_rank(&self) -> usize {
        self.gate_rank
    }
}

/// Configuration of the Inkling 2 text decoder, which determines the shapes of its parameters and every constant of
/// its computation. [`Self::inkling_2_mini`] returns the configuration of the Inkling 2 Mini release, and
/// [`Self::reduced`] returns a small configuration that keeps the complete layer layout of the release (i.e., 24
/// physical layers, a three pass loop over the middle 12, both quantization groups, and every component) at widths
/// that are cheap enough for tests and for end to end experiments on one machine.
#[derive(Clone, Debug, PartialEq)]
pub struct Inkling2Configuration {
    /// Width of the residual stream.
    hidden_size: usize,

    /// Number of physical layers (i.e., distinct sets of layer weights).
    layer_count: usize,

    /// Number of rows of the embedding and unembedding matrices, including padding rows that never receive
    /// probability.
    vocabulary_size: usize,

    /// Number of leading vocabulary entries that the loss and the sampling distribution are normalized over.
    usable_vocabulary_size: usize,

    /// Number of query heads of every attention layer.
    head_count: usize,

    /// Number of key and value heads of every attention layer, which must divide [`Self::head_count`].
    key_value_head_count: usize,

    /// Width of every attention head.
    head_size: usize,

    /// Number of components of the query conditioned relative position features of every query head.
    relative_size: usize,

    /// Number of backward distances (starting at zero) that have a learned relative position table entry.
    relative_extent: usize,

    /// Number of positions (including the query position itself) that a local attention layer attends over.
    local_window_size: usize,

    /// Indices of the layers that use local attention. The remaining layers use global attention.
    local_layers: Vec<usize>,

    /// Position scale floor of the logarithmic scaling of global attention queries.
    log_scaling_floor: f64,

    /// Slope of the logarithmic scaling of global attention queries.
    log_scaling_alpha: f64,

    /// Stabilizer of every root mean square normalization.
    normalization_epsilon: f64,

    /// Largest L2 length of a query or key head vector.
    query_key_clip_threshold: f64,

    /// Largest L2 length of the relative position features of a query head.
    relative_clip_threshold: f64,

    /// Number of taps of every causal short convolution.
    convolution_width: usize,

    /// Soft cap of the output logits.
    logit_softcap: f64,

    /// Number of physical layers per attention residual block.
    block_layer_count: usize,

    /// Number of routed experts of every layer.
    expert_count: usize,

    /// Number of experts that every token is routed to.
    experts_per_token: usize,

    /// Width of the hidden activation of every expert.
    expert_size: usize,

    /// Soft cap of the gate branch of the expert activation.
    gate_softcap: f64,

    /// Soft cap of the up branch of the expert activation.
    up_softcap: f64,

    /// Sum of the routing weights of the experts selected for a token.
    route_scale: f64,

    /// Loop over the core layers, or [`None`] for a decoder that applies every layer once.
    r#loop: Option<Inkling2LoopConfiguration>,

    /// Number of leading layers whose expert projections are stored and trained in NVFP4. The remaining layers use
    /// MXFP8.
    nvfp4_layer_count: usize,

    /// Data type at which the decoder computes its projections and stores the residual stream between layers.
    /// Normalizations, residual reads, attention reductions, routing, and the loss always compute in `f32`.
    compute_type: DataType,
}

impl Inkling2Configuration {
    /// Returns the configuration of the Inkling 2 Mini release (i.e., the `text_config` of the `config.json` at
    /// revision `ed40e7da625f594fc9c4990f5f66be24873c22d7` of `thinkingmachines/Inkling-2-Mini`), computing at `bf16`.
    pub fn inkling_2_mini() -> Self {
        Self {
            hidden_size: 2304,
            layer_count: 24,
            vocabulary_size: 201_024,
            usable_vocabulary_size: 200_058,
            head_count: 24,
            key_value_head_count: 8,
            head_size: 128,
            relative_size: 16,
            relative_extent: 1024,
            local_window_size: 1024,
            local_layers: (0..24).filter(|layer| layer % 6 != 5).collect(),
            log_scaling_floor: 8192.0,
            log_scaling_alpha: 0.1,
            normalization_epsilon: 1e-6,
            query_key_clip_threshold: 512f64.sqrt(),
            relative_clip_threshold: 8.0,
            convolution_width: 4,
            logit_softcap: 30.0,
            block_layer_count: 6,
            expert_count: 128,
            experts_per_token: 8,
            expert_size: 1152,
            gate_softcap: 4.0,
            up_softcap: 25.0,
            route_scale: 8.0,
            r#loop: Some(Inkling2LoopConfiguration::new(6, 18, 3, 64)),
            nvfp4_layer_count: 20,
            compute_type: DataType::BF16,
        }
    }

    /// Returns a reduced configuration with the layer layout of the release and small widths. It keeps 24 physical
    /// layers in blocks of six with the last layer of every block global, the three pass loop over layers `6..18`
    /// with all four loop gates, grouped query attention with three query heads per key and value head, relative
    /// position features, routing to 8 experts, and 20 NVFP4 layers followed by 4 MXFP8 layers. The hidden size and
    /// the expert size are multiples of 32, so both block-scaled formats apply to every expert projection. The local
    /// window and the relative extent are small, so short sequences cross the window and extent boundaries.
    pub fn reduced() -> Self {
        Self {
            hidden_size: 128,
            vocabulary_size: 512,
            usable_vocabulary_size: 500,
            head_count: 6,
            key_value_head_count: 2,
            head_size: 32,
            relative_size: 8,
            relative_extent: 12,
            local_window_size: 8,
            expert_count: 16,
            expert_size: 64,
            r#loop: Some(Inkling2LoopConfiguration::new(6, 18, 3, 16)),
            ..Self::inkling_2_mini()
        }
    }

    /// Parses the configuration of an Inkling 2 release from the contents of its `config.json`, computing at
    /// `bf16`. The number of NVFP4 layers is not part of that file (it is recorded in the
    /// quantization manifest of a release), so it is set to the number of layers and
    /// [`Self::with_nvfp4_layer_count`] selects the layout of a particular release.
    ///
    /// # Errors
    ///
    /// Returns [`Inkling2ConfigurationError::InvalidJson`] if `json` is not a JSON object with a `text_config` that
    /// holds every required field, and [`Inkling2ConfigurationError::Invalid`] if the configuration it describes is
    /// not supported (refer to [`Self::validate`]) or uses different head layouts for local and global attention.
    pub fn from_json(json: &str) -> Result<Self, Inkling2ConfigurationError> {
        let invalid = |message: String| Inkling2ConfigurationError::InvalidJson { message };
        let root = JsonValue::parse(json).map_err(|error| invalid(error.to_string()))?;
        let text = root.get("text_config").ok_or_else(|| invalid("missing field `text_config`".to_string()))?;
        let field = |name: &str| text.get(name).ok_or_else(|| invalid(format!("missing field `text_config.{name}`")));
        let integer = |name: &str| match field(name)? {
            JsonValue::Integer(value) if *value >= 0 => Ok(*value as usize),
            _ => Err(invalid(format!("field `text_config.{name}` is not a non-negative integer"))),
        };
        let number = |name: &str| match field(name)? {
            JsonValue::Integer(value) => Ok(*value as f64),
            JsonValue::Float(value) => Ok(*value),
            _ => Err(invalid(format!("field `text_config.{name}` is not a number"))),
        };
        let boolean = |name: &str| match text.get(name) {
            Some(JsonValue::Boolean(value)) => Some(*value),
            _ => None,
        };

        let local_layers = match field("local_layer_ids")? {
            JsonValue::Array(layers) => layers
                .iter()
                .map(|layer| match layer {
                    JsonValue::Integer(layer) if *layer >= 0 => Some(*layer as usize),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>(),
            _ => None,
        }
        .ok_or_else(|| invalid("field `text_config.local_layer_ids` is not a list of layer indices".to_string()))?;
        let r#loop = match text.get("loop_config") {
            None | Some(JsonValue::Null) => None,
            Some(value) => {
                let loop_integer = |name: &str| match value.get(name) {
                    Some(JsonValue::Integer(value)) if *value >= 0 => Ok(*value as usize),
                    _ => Err(invalid(format!("field `text_config.loop_config.{name}` is not a non-negative integer"))),
                };
                if value.get("is_varloop") == Some(&JsonValue::Boolean(true)) {
                    return Err(Inkling2ConfigurationError::Invalid {
                        message: "variable loop checkpoints are not supported".to_string(),
                    });
                }
                Some(Inkling2LoopConfiguration::new(
                    loop_integer("core_start")?,
                    loop_integer("core_end")?,
                    loop_integer("num_passes")?,
                    loop_integer("gate_rank")?,
                ))
            }
        };

        for (local, global) in [
            ("swa_num_attention_heads", "num_attention_heads"),
            ("swa_num_key_value_heads", "num_key_value_heads"),
            ("swa_head_dim", "head_dim"),
        ] {
            if integer(local)? != integer(global)? {
                return Err(Inkling2ConfigurationError::Invalid {
                    message: format!("`{local}` must equal `{global}`"),
                });
            }
        }
        if matches!(text.get("n_shared_experts"), Some(JsonValue::Integer(count)) if *count != 0) {
            return Err(Inkling2ConfigurationError::Invalid { message: "`n_shared_experts` must be 0".to_string() });
        }
        for name in ["attn_res_normalize_values", "use_value_norm_no_gamma", "use_gate_bias"] {
            if boolean(name) == Some(false) {
                return Err(Inkling2ConfigurationError::Invalid { message: format!("`{name}` must be true") });
            }
        }
        for name in ["q_bias", "o_bias"] {
            if boolean(name) == Some(true) {
                return Err(Inkling2ConfigurationError::Invalid { message: format!("`{name}` must be false") });
            }
        }

        let layer_count = integer("num_hidden_layers")?;
        let configuration = Self {
            hidden_size: integer("hidden_size")?,
            layer_count,
            vocabulary_size: integer("vocab_size")?,
            usable_vocabulary_size: integer("unpadded_vocab_size")?,
            head_count: integer("num_attention_heads")?,
            key_value_head_count: integer("num_key_value_heads")?,
            head_size: integer("head_dim")?,
            relative_size: integer("d_rel")?,
            relative_extent: integer("rel_extent")?,
            local_window_size: integer("sliding_window_size")?,
            local_layers,
            log_scaling_floor: number("log_scaling_n_floor")?,
            log_scaling_alpha: number("log_scaling_alpha")?,
            normalization_epsilon: number("rms_norm_eps")?,
            query_key_clip_threshold: number("qk_l2_clip_threshold")?,
            relative_clip_threshold: number("rel_l2_clip_threshold")?,
            convolution_width: integer("sconv_kernel_size")?,
            logit_softcap: number("final_logit_softcapping")?,
            block_layer_count: integer("n_layers_per_attn_res_block")?,
            expert_count: integer("n_routed_experts")?,
            experts_per_token: integer("num_experts_per_tok")?,
            expert_size: integer("intermediate_size")?,
            gate_softcap: number("mlp_gate_softcap")?,
            up_softcap: number("mlp_up_softcap")?,
            route_scale: number("route_scale")?,
            r#loop,
            nvfp4_layer_count: layer_count,
            compute_type: DataType::BF16,
        };
        configuration.validate()?;
        Ok(configuration)
    }

    /// Returns this configuration with the provided number of leading layers whose expert projections use NVFP4.
    #[inline]
    pub fn with_nvfp4_layer_count(mut self, nvfp4_layer_count: usize) -> Self {
        self.nvfp4_layer_count = nvfp4_layer_count;
        self
    }

    /// Returns this configuration with the provided compute data type (refer to [`Self::compute_type`]).
    #[inline]
    pub fn with_compute_type(mut self, compute_type: DataType) -> Self {
        self.compute_type = compute_type;
        self
    }

    /// Returns the width of the residual stream.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of physical layers (i.e., distinct sets of layer weights).
    #[inline]
    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    /// Returns the number of rows of the embedding and unembedding matrices, including padding rows.
    #[inline]
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    /// Returns the number of leading vocabulary entries that the loss and the sampling distribution are normalized
    /// over. The remaining rows of the unembedding matrix are padding and never receive probability.
    #[inline]
    pub fn usable_vocabulary_size(&self) -> usize {
        self.usable_vocabulary_size
    }

    /// Returns the number of query heads of every attention layer.
    #[inline]
    pub fn head_count(&self) -> usize {
        self.head_count
    }

    /// Returns the number of key and value heads of every attention layer.
    #[inline]
    pub fn key_value_head_count(&self) -> usize {
        self.key_value_head_count
    }

    /// Returns the width of every attention head.
    #[inline]
    pub fn head_size(&self) -> usize {
        self.head_size
    }

    /// Returns the number of components of the query conditioned relative position features of every query head.
    #[inline]
    pub fn relative_size(&self) -> usize {
        self.relative_size
    }

    /// Returns the number of backward distances (starting at zero) that have a learned relative position table entry.
    /// Keys at larger distances still take part in global attention, with a zero relative position term.
    #[inline]
    pub fn relative_extent(&self) -> usize {
        self.relative_extent
    }

    /// Returns the number of positions, including the query position itself, that a local attention layer attends
    /// over.
    #[inline]
    pub fn local_window_size(&self) -> usize {
        self.local_window_size
    }

    /// Returns whether `layer` uses local attention.
    #[inline]
    pub fn is_local_layer(&self, layer: usize) -> bool {
        self.local_layers.contains(&layer)
    }

    /// Returns the position scale floor `n` of the logarithmic scaling `1 + alpha · ln(max(1, (position + 1) / n))` of
    /// global attention queries.
    #[inline]
    pub fn log_scaling_floor(&self) -> f64 {
        self.log_scaling_floor
    }

    /// Returns the slope `alpha` of the logarithmic scaling of global attention queries.
    #[inline]
    pub fn log_scaling_alpha(&self) -> f64 {
        self.log_scaling_alpha
    }

    /// Returns the stabilizer of every root mean square normalization.
    #[inline]
    pub fn normalization_epsilon(&self) -> f64 {
        self.normalization_epsilon
    }

    /// Returns the largest L2 length of a query or key head vector.
    #[inline]
    pub fn query_key_clip_threshold(&self) -> f64 {
        self.query_key_clip_threshold
    }

    /// Returns the largest L2 length of the relative position features of a query head.
    #[inline]
    pub fn relative_clip_threshold(&self) -> f64 {
        self.relative_clip_threshold
    }

    /// Returns the number of taps of every causal short convolution.
    #[inline]
    pub fn convolution_width(&self) -> usize {
        self.convolution_width
    }

    /// Returns the soft cap `c` of the output logits, which are computed as `c · tanh(logit / c)`.
    #[inline]
    pub fn logit_softcap(&self) -> f64 {
        self.logit_softcap
    }

    /// Returns the number of physical layers per attention residual block.
    #[inline]
    pub fn block_layer_count(&self) -> usize {
        self.block_layer_count
    }

    /// Returns the number of routed experts of every layer.
    #[inline]
    pub fn expert_count(&self) -> usize {
        self.expert_count
    }

    /// Returns the number of experts that every token is routed to.
    #[inline]
    pub fn experts_per_token(&self) -> usize {
        self.experts_per_token
    }

    /// Returns the width of the hidden activation of every expert.
    #[inline]
    pub fn expert_size(&self) -> usize {
        self.expert_size
    }

    /// Returns the soft cap of the gate branch of the expert activation.
    #[inline]
    pub fn gate_softcap(&self) -> f64 {
        self.gate_softcap
    }

    /// Returns the soft cap of the up branch of the expert activation.
    #[inline]
    pub fn up_softcap(&self) -> f64 {
        self.up_softcap
    }

    /// Returns the sum of the routing weights of the experts selected for a token.
    #[inline]
    pub fn route_scale(&self) -> f64 {
        self.route_scale
    }

    /// Returns the loop over the core layers, or [`None`] for a decoder that applies every layer once.
    #[inline]
    pub fn r#loop(&self) -> Option<&Inkling2LoopConfiguration> {
        self.r#loop.as_ref()
    }

    /// Returns the number of leading layers whose expert projections are stored and trained in NVFP4. The remaining
    /// layers use MXFP8. Block-scaled expert projectors (e.g.,
    /// [`BlockScaledExpertProjector`](crate::experts::BlockScaledExpertProjector)) read it to select their formats.
    #[inline]
    pub fn nvfp4_layer_count(&self) -> usize {
        self.nvfp4_layer_count
    }

    /// Returns the data type at which the decoder computes its projections and stores the residual stream between
    /// layers. Normalizations, residual reads, attention reductions, routing, and the loss always compute in `f32`.
    #[inline]
    pub fn compute_type(&self) -> DataType {
        self.compute_type
    }

    /// Returns whether `layer` is one of the core layers of the loop.
    #[inline]
    pub fn is_core_layer(&self, layer: usize) -> bool {
        self.r#loop.is_some_and(|r#loop| (r#loop.core_start..r#loop.core_end).contains(&layer))
    }

    /// Returns the indices of the physical layers in the order in which the decoder applies them, with one entry per
    /// layer application. Core layers appear once per pass.
    pub fn layer_schedule(&self) -> Vec<usize> {
        match self.r#loop {
            None => (0..self.layer_count).collect(),
            Some(r#loop) => (0..r#loop.core_start)
                .chain((0..r#loop.pass_count).flat_map(|_| r#loop.core_start..r#loop.core_end))
                .chain(r#loop.core_end..self.layer_count)
                .collect(),
        }
    }

    /// Checks that this configuration describes a decoder that this implementation supports.
    ///
    /// # Errors
    ///
    /// Returns [`Inkling2ConfigurationError::Invalid`] that names the first violated requirement.
    pub fn validate(&self) -> Result<(), Inkling2ConfigurationError> {
        let invalid = |message: String| Err(Inkling2ConfigurationError::Invalid { message });
        for (name, value) in [
            ("hidden size", self.hidden_size),
            ("layer count", self.layer_count),
            ("head count", self.head_count),
            ("key and value head count", self.key_value_head_count),
            ("head size", self.head_size),
            ("relative size", self.relative_size),
            ("relative extent", self.relative_extent),
            ("local window size", self.local_window_size),
            ("convolution width", self.convolution_width),
            ("block layer count", self.block_layer_count),
            ("expert count", self.expert_count),
            ("experts per token", self.experts_per_token),
            ("expert size", self.expert_size),
            ("usable vocabulary size", self.usable_vocabulary_size),
        ] {
            if value == 0 {
                return invalid(format!("{name} must be positive"));
            }
        }
        if self.usable_vocabulary_size > self.vocabulary_size {
            return invalid(format!(
                "usable vocabulary size {} exceeds vocabulary size {}",
                self.usable_vocabulary_size, self.vocabulary_size,
            ));
        }
        if !self.head_count.is_multiple_of(self.key_value_head_count) {
            return invalid(format!(
                "head count {} is not a multiple of key and value head count {}",
                self.head_count, self.key_value_head_count,
            ));
        }
        if self.experts_per_token > self.expert_count {
            return invalid(format!(
                "experts per token {} exceeds expert count {}",
                self.experts_per_token, self.expert_count,
            ));
        }
        if !self.layer_count.is_multiple_of(self.block_layer_count) {
            return invalid(format!(
                "layer count {} is not a multiple of block layer count {}",
                self.layer_count, self.block_layer_count,
            ));
        }
        if let Some(layer) = self.local_layers.iter().find(|&&layer| layer >= self.layer_count) {
            return invalid(format!("local layer {layer} is not one of the {} layers", self.layer_count));
        }
        if self.nvfp4_layer_count > self.layer_count {
            return invalid(format!(
                "NVFP4 layer count {} exceeds layer count {}",
                self.nvfp4_layer_count, self.layer_count,
            ));
        }
        if !matches!(self.compute_type, DataType::BF16 | DataType::F32) {
            return invalid(format!("compute data type must be `bf16` or `f32` but got `{}`", self.compute_type));
        }
        if let Some(r#loop) = self.r#loop {
            // The loop restarts every pass from the state after the last layer before the core, and it reads the
            // blocks that the previous pass completed, so the core must consist of whole blocks and must be preceded
            // by at least one block.
            if r#loop.core_start >= r#loop.core_end || r#loop.core_end > self.layer_count {
                return invalid(format!(
                    "core layers {}..{} are not a non-empty range of the {} layers",
                    r#loop.core_start, r#loop.core_end, self.layer_count,
                ));
            }
            if r#loop.core_start == 0
                || r#loop.core_start % self.block_layer_count != 0
                || r#loop.core_end % self.block_layer_count != 0
            {
                return invalid(format!(
                    "core layers {}..{} must start after the first block and consist of whole blocks of {} layers",
                    r#loop.core_start, r#loop.core_end, self.block_layer_count,
                ));
            }
            if r#loop.pass_count == 0 || r#loop.gate_rank == 0 {
                return invalid("loop pass count and gate rank must be positive".to_string());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    /// `text_config` of the `config.json` of the Inkling 2 Mini release, with the fields that this implementation
    /// reads.
    const INKLING_2_MINI_CONFIG: &str = r#"{
        "model_type": "inkling2_loop",
        "text_config": {
            "hidden_size": 2304, "num_hidden_layers": 24, "vocab_size": 201024, "num_attention_heads": 24,
            "num_key_value_heads": 8, "head_dim": 128, "d_rel": 16, "rel_extent": 1024, "q_bias": false,
            "o_bias": false, "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
            "use_value_norm_no_gamma": true, "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
            "sconv_kernel_size": 4, "unpadded_vocab_size": 200058, "final_logit_softcapping": 30.0,
            "swa_head_dim": 128, "swa_num_attention_heads": 24, "swa_num_key_value_heads": 8,
            "sliding_window_size": 1024, "n_layers_per_attn_res_block": 6, "attn_res_normalize_values": true,
            "n_routed_experts": 128, "num_experts_per_tok": 8, "n_shared_experts": 0, "intermediate_size": 1152,
            "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0, "route_scale": 8.0, "use_gate_bias": true,
            "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": false}
        }
    }"#;

    #[test]
    fn test_inkling2_loop_configuration() {
        let r#loop = Inkling2LoopConfiguration::new(6, 18, 3, 64);
        assert_eq!(r#loop.core_start(), 6);
        assert_eq!(r#loop.core_end(), 18);
        assert_eq!(r#loop.pass_count(), 3);
        assert_eq!(r#loop.gate_rank(), 64);
    }

    #[test]
    fn test_inkling2_configuration_inkling_2_mini() {
        let configuration = Inkling2Configuration::inkling_2_mini();
        assert_eq!(configuration.validate(), Ok(()));
        assert_eq!(configuration.hidden_size(), 2304);
        assert_eq!(configuration.layer_count(), 24);
        assert_eq!(configuration.vocabulary_size(), 201_024);
        assert_eq!(configuration.usable_vocabulary_size(), 200_058);
        assert_eq!(configuration.head_count(), 24);
        assert_eq!(configuration.key_value_head_count(), 8);
        assert_eq!(configuration.head_size(), 128);
        assert_eq!(configuration.relative_size(), 16);
        assert_eq!(configuration.relative_extent(), 1024);
        assert_eq!(configuration.local_window_size(), 1024);
        assert_eq!(configuration.log_scaling_floor(), 8192.0);
        assert_eq!(configuration.log_scaling_alpha(), 0.1);
        assert_eq!(configuration.normalization_epsilon(), 1e-6);
        assert_eq!(configuration.query_key_clip_threshold(), 22.627416997969522);
        assert_eq!(configuration.relative_clip_threshold(), 8.0);
        assert_eq!(configuration.convolution_width(), 4);
        assert_eq!(configuration.logit_softcap(), 30.0);
        assert_eq!(configuration.block_layer_count(), 6);
        assert_eq!(configuration.expert_count(), 128);
        assert_eq!(configuration.experts_per_token(), 8);
        assert_eq!(configuration.expert_size(), 1152);
        assert_eq!(configuration.gate_softcap(), 4.0);
        assert_eq!(configuration.up_softcap(), 25.0);
        assert_eq!(configuration.route_scale(), 8.0);
        assert_eq!(configuration.r#loop(), Some(&Inkling2LoopConfiguration::new(6, 18, 3, 64)));
        assert_eq!(configuration.nvfp4_layer_count(), 20);
        assert_eq!(configuration.compute_type(), DataType::BF16);

        // The last layer of every block of six is global, and the middle two blocks are the core of the loop.
        assert_eq!((0..24).filter(|&layer| !configuration.is_local_layer(layer)).collect::<Vec<_>>(), [5, 11, 17, 23]);
        assert_eq!(
            (0..24).filter(|&layer| configuration.is_core_layer(layer)).collect::<Vec<_>>(),
            (6..18).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inkling2_configuration_reduced() {
        let configuration = Inkling2Configuration::reduced();
        assert_eq!(configuration.validate(), Ok(()));
        assert_eq!(configuration.hidden_size(), 128);
        assert_eq!(configuration.layer_count(), 24);
        assert_eq!(configuration.expert_count(), 16);
        assert_eq!(configuration.experts_per_token(), 8);
        assert_eq!(configuration.r#loop(), Some(&Inkling2LoopConfiguration::new(6, 18, 3, 16)));
        assert_eq!(configuration.nvfp4_layer_count(), 20);
        assert_eq!(configuration.layer_schedule().len(), 48);
    }

    #[test]
    fn test_inkling2_configuration_from_json() {
        assert_eq!(
            Inkling2Configuration::from_json(INKLING_2_MINI_CONFIG)
                .map(|configuration| configuration.with_nvfp4_layer_count(20)),
            Ok(Inkling2Configuration::inkling_2_mini()),
        );

        assert!(matches!(
            Inkling2Configuration::from_json("[1"),
            Err(Inkling2ConfigurationError::InvalidJson { message })
                if message == "invalid JSON: unexpected end of input at byte 2",
        ));
        assert_eq!(
            Inkling2Configuration::from_json("{}"),
            Err(Inkling2ConfigurationError::InvalidJson { message: "missing field `text_config`".to_string() }),
        );
        assert_eq!(
            Inkling2Configuration::from_json(r#"{"text_config":{"local_layer_ids":[]}}"#),
            Err(Inkling2ConfigurationError::InvalidJson {
                message: "missing field `text_config.swa_num_attention_heads`".to_string(),
            }),
        );
        assert_eq!(
            Inkling2Configuration::from_json(
                &INKLING_2_MINI_CONFIG.replace("\"hidden_size\": 2304", "\"hidden_size\": 2.5")
            ),
            Err(Inkling2ConfigurationError::InvalidJson {
                message: "field `text_config.hidden_size` is not a non-negative integer".to_string(),
            }),
        );
        assert_eq!(
            Inkling2Configuration::from_json(
                &INKLING_2_MINI_CONFIG.replace("\"is_varloop\": false", "\"is_varloop\": true")
            ),
            Err(Inkling2ConfigurationError::Invalid {
                message: "variable loop checkpoints are not supported".to_string(),
            }),
        );
        assert_eq!(
            Inkling2Configuration::from_json(
                &INKLING_2_MINI_CONFIG.replace("\"swa_head_dim\": 128", "\"swa_head_dim\": 64")
            ),
            Err(Inkling2ConfigurationError::Invalid { message: "`swa_head_dim` must equal `head_dim`".to_string() }),
        );
        assert_eq!(
            Inkling2Configuration::from_json(
                &INKLING_2_MINI_CONFIG.replace("\"n_shared_experts\": 0", "\"n_shared_experts\": 1")
            ),
            Err(Inkling2ConfigurationError::Invalid { message: "`n_shared_experts` must be 0".to_string() }),
        );
        assert_eq!(
            Inkling2Configuration::from_json(&INKLING_2_MINI_CONFIG.replace("\"q_bias\": false", "\"q_bias\": true")),
            Err(Inkling2ConfigurationError::Invalid { message: "`q_bias` must be false".to_string() }),
        );
    }

    #[test]
    fn test_inkling2_configuration_with_nvfp4_layer_count() {
        assert_eq!(Inkling2Configuration::reduced().with_nvfp4_layer_count(3).nvfp4_layer_count(), 3);
    }

    #[test]
    fn test_inkling2_configuration_with_compute_type() {
        assert_eq!(Inkling2Configuration::reduced().with_compute_type(DataType::F32).compute_type(), DataType::F32);
    }

    #[test]
    fn test_inkling2_configuration_layer_schedule() {
        let schedule = Inkling2Configuration::inkling_2_mini().layer_schedule();
        let expected = (0..6).chain(6..18).chain(6..18).chain(6..18).chain(18..24).collect::<Vec<_>>();
        assert_eq!(schedule, expected);

        let mut configuration = Inkling2Configuration::reduced();
        configuration.r#loop = None;
        assert_eq!(configuration.layer_schedule(), (0..24).collect::<Vec<_>>());
    }

    #[test]
    fn test_inkling2_configuration_validate() {
        let invalid = |message: &str| Err(Inkling2ConfigurationError::Invalid { message: message.to_string() });
        let mut configuration = Inkling2Configuration::reduced();
        configuration.head_size = 0;
        assert_eq!(configuration.validate(), invalid("head size must be positive"));

        let mut configuration = Inkling2Configuration::reduced();
        configuration.usable_vocabulary_size = 513;
        assert_eq!(configuration.validate(), invalid("usable vocabulary size 513 exceeds vocabulary size 512"));

        let mut configuration = Inkling2Configuration::reduced();
        configuration.key_value_head_count = 4;
        assert_eq!(configuration.validate(), invalid("head count 6 is not a multiple of key and value head count 4"));

        let mut configuration = Inkling2Configuration::reduced();
        configuration.experts_per_token = 17;
        assert_eq!(configuration.validate(), invalid("experts per token 17 exceeds expert count 16"));

        let mut configuration = Inkling2Configuration::reduced();
        configuration.layer_count = 25;
        assert_eq!(configuration.validate(), invalid("layer count 25 is not a multiple of block layer count 6"));

        let mut configuration = Inkling2Configuration::reduced();
        configuration.local_layers.push(24);
        assert_eq!(configuration.validate(), invalid("local layer 24 is not one of the 24 layers"));

        assert_eq!(
            Inkling2Configuration::reduced().with_nvfp4_layer_count(25).validate(),
            invalid("NVFP4 layer count 25 exceeds layer count 24"),
        );
        assert_eq!(
            Inkling2Configuration::reduced().with_compute_type(DataType::F64).validate(),
            invalid("compute data type must be `bf16` or `f32` but got `f64`"),
        );

        let mut configuration = Inkling2Configuration::reduced();
        configuration.r#loop = Some(Inkling2LoopConfiguration::new(18, 30, 3, 16));
        assert_eq!(configuration.validate(), invalid("core layers 18..30 are not a non-empty range of the 24 layers"));
        configuration.r#loop = Some(Inkling2LoopConfiguration::new(0, 12, 3, 16));
        assert_eq!(
            configuration.validate(),
            invalid("core layers 0..12 must start after the first block and consist of whole blocks of 6 layers"),
        );
        configuration.r#loop = Some(Inkling2LoopConfiguration::new(6, 18, 0, 16));
        assert_eq!(configuration.validate(), invalid("loop pass count and gate rank must be positive"));
    }
}
