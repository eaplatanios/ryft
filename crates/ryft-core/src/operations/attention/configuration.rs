use super::*;

/// Backend implementation requested for an attention operation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum AttentionImplementation {
    /// Uses a fused implementation when the active backend supports the complete configuration and otherwise uses
    /// the portable semantic composition.
    #[default]
    Automatic,

    /// Always uses the portable semantic composition.
    Portable,

    /// Requires a fused backend implementation and reports an error when the configuration is not supported.
    Fused,
}

impl Display for AttentionImplementation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Automatic => formatter.write_str("automatic"),
            Self::Portable => formatter.write_str("portable"),
            Self::Fused => formatter.write_str("fused"),
        }
    }
}

/// Value-independent semantics of scaled dot-product attention.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct AttentionConfiguration {
    /// Explicit score scale, or [`None`] to use `1 / sqrt(head_dimension)`.
    scale: Option<f64>,

    /// Whether query position `i` may attend only to key/value positions `j <= i`.
    causal: bool,

    /// Optional inclusive `(left, right)` local-window radii.
    local_window: Option<(usize, usize)>,

    /// Requested backend implementation.
    implementation: AttentionImplementation,

    /// Whether the forward operation also returns its log-sum-exp residual.
    return_residual: bool,

    /// Optional fused-only `(rate, seed)` dropout extension.
    dropout: Option<(f64, u64)>,
}

impl AttentionConfiguration {
    /// Creates the default configuration: inferred scale, non-causal global attention, automatic implementation
    /// selection, no residual output, and no dropout.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets an explicit score scale. Passing [`None`] restores `1 / sqrt(head_dimension)`.
    #[inline]
    pub fn with_scale<S: Into<Option<f64>>>(mut self, scale: S) -> Self {
        self.scale = scale.into();
        self
    }

    /// Enables or disables causal masking.
    #[inline]
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Sets the inclusive `(left, right)` local-window radii. The local window is independent of causal masking.
    #[inline]
    pub fn with_local_window<W: Into<Option<(usize, usize)>>>(mut self, local_window: W) -> Self {
        self.local_window = local_window.into();
        self
    }

    /// Sets a symmetric local-window radius.
    #[inline]
    pub fn with_symmetric_local_window<W: Into<Option<usize>>>(mut self, local_window: W) -> Self {
        self.local_window = local_window.into().map(|window| (window, window));
        self
    }

    /// Sets the requested backend implementation.
    #[inline]
    pub fn with_implementation(mut self, implementation: AttentionImplementation) -> Self {
        self.implementation = implementation;
        self
    }

    /// Requests or suppresses the log-sum-exp residual output.
    #[inline]
    pub fn with_residual(mut self, return_residual: bool) -> Self {
        self.return_residual = return_residual;
        self
    }

    /// Sets the fused-only `(rate, seed)` dropout extension.
    #[inline]
    pub fn with_dropout<D: Into<Option<(f64, u64)>>>(mut self, dropout: D) -> Self {
        self.dropout = dropout.into();
        self
    }

    /// Returns the explicit score scale, if one was configured.
    #[inline]
    pub fn scale(&self) -> Option<f64> {
        self.scale
    }

    /// Returns whether causal masking is enabled.
    #[inline]
    pub fn causal(&self) -> bool {
        self.causal
    }

    /// Returns the optional inclusive `(left, right)` local-window radii.
    #[inline]
    pub fn local_window(&self) -> Option<(usize, usize)> {
        self.local_window
    }

    /// Returns the requested backend implementation.
    #[inline]
    pub fn implementation(&self) -> AttentionImplementation {
        self.implementation
    }

    /// Returns whether the forward operation produces its log-sum-exp residual.
    #[inline]
    pub fn return_residual(&self) -> bool {
        self.return_residual
    }

    /// Returns the optional fused-only `(rate, seed)` dropout extension.
    #[inline]
    pub fn dropout(&self) -> Option<(f64, u64)> {
        self.dropout
    }
}

/// Presence metadata for the optional attention operands in their canonical order.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttentionOperandSignature {
    /// Whether an additive bias follows query, key, and value.
    bias: bool,

    /// Whether an arbitrary Boolean visibility mask follows the bias.
    mask: bool,

    /// Whether per-batch query sequence lengths follow the mask.
    query_sequence_lengths: bool,

    /// Whether per-batch key/value sequence lengths follow query sequence lengths.
    key_value_sequence_lengths: bool,
}

impl AttentionOperandSignature {
    /// Creates an optional-operand signature.
    #[inline]
    pub fn new(bias: bool, mask: bool, query_sequence_lengths: bool, key_value_sequence_lengths: bool) -> Self {
        Self { bias, mask, query_sequence_lengths, key_value_sequence_lengths }
    }

    /// Returns whether an additive bias is present.
    #[inline]
    pub fn has_bias(&self) -> bool {
        self.bias
    }

    /// Returns whether an arbitrary Boolean visibility mask is present.
    #[inline]
    pub fn has_mask(&self) -> bool {
        self.mask
    }

    /// Returns whether per-batch query sequence lengths are present.
    #[inline]
    pub fn has_query_sequence_lengths(&self) -> bool {
        self.query_sequence_lengths
    }

    /// Returns whether per-batch key/value sequence lengths are present.
    #[inline]
    pub fn has_key_value_sequence_lengths(&self) -> bool {
        self.key_value_sequence_lengths
    }

    /// Returns the number of optional operands described by this signature.
    #[inline]
    pub fn count(&self) -> usize {
        usize::from(self.bias)
            + usize::from(self.mask)
            + usize::from(self.query_sequence_lengths)
            + usize::from(self.key_value_sequence_lengths)
    }
}

/// Query, key, value, and optional operands supplied to scaled dot-product attention.
#[derive(Clone, Debug, PartialEq, Parameterized)]
pub struct AttentionInputs<P: Parameter> {
    /// Query array in `TNH` or `BTNH` layout.
    pub query: P,

    /// Key array in `SKH` or `BSKH` layout.
    pub key: P,

    /// Value array with the same logical shape as `key`.
    pub value: P,

    /// Optional broadcastable additive bias.
    pub bias: Option<P>,

    /// Optional broadcastable Boolean visibility mask.
    pub mask: Option<P>,

    /// Optional per-batch query sequence lengths.
    pub query_sequence_lengths: Option<P>,

    /// Optional per-batch key/value sequence lengths.
    pub key_value_sequence_lengths: Option<P>,
}

impl<P: Parameter> AttentionInputs<P> {
    /// Creates attention inputs with no optional operands.
    #[inline]
    pub fn new(query: P, key: P, value: P) -> Self {
        Self {
            query,
            key,
            value,
            bias: None,
            mask: None,
            query_sequence_lengths: None,
            key_value_sequence_lengths: None,
        }
    }

    /// Returns the optional-operand signature of these inputs.
    #[inline]
    pub fn signature(&self) -> AttentionOperandSignature {
        AttentionOperandSignature::new(
            self.bias.is_some(),
            self.mask.is_some(),
            self.query_sequence_lengths.is_some(),
            self.key_value_sequence_lengths.is_some(),
        )
    }

    /// Parses values in canonical attention operand order according to `signature`.
    ///
    /// # Parameters
    ///
    ///   - `signature`: Presence metadata for the optional operands.
    ///   - `values`: Query, key, value, and the present optional operands in canonical order.
    pub fn from_values(signature: AttentionOperandSignature, values: &[P]) -> Result<Self, TypeError>
    where
        P: Clone,
    {
        let expected_count = 3 + signature.count();
        if values.len() != expected_count {
            return Err(TypeError::invalid(format!(
                "attention input signature expects {expected_count} values but got {}",
                values.len(),
            )));
        }
        let mut values = values.iter().cloned();
        Ok(Self {
            query: values.next().unwrap(),
            key: values.next().unwrap(),
            value: values.next().unwrap(),
            bias: signature.has_bias().then(|| values.next().unwrap()),
            mask: signature.has_mask().then(|| values.next().unwrap()),
            query_sequence_lengths: signature.has_query_sequence_lengths().then(|| values.next().unwrap()),
            key_value_sequence_lengths: signature.has_key_value_sequence_lengths().then(|| values.next().unwrap()),
        })
    }

    /// Returns the values in canonical attention operand order.
    pub fn into_values(self) -> Vec<P> {
        vec![self.query, self.key, self.value]
            .into_iter()
            .chain(self.bias)
            .chain(self.mask)
            .chain(self.query_sequence_lengths)
            .chain(self.key_value_sequence_lengths)
            .collect()
    }
}
