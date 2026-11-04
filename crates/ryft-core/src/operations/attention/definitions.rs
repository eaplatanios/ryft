//! Named attention definitions and the caller-owned inference function that stages them.
//!
//! Scaled dot-product attention is a named computation rather than a primitive operation. It has a portable body built
//! from ordinary array operations (refer to [`dot_product_attention_ir_composition`]), a stabilized analytical
//! derivative (refer to [`dot_product_attention_backward_ir_composition`]), and optional backend implementations (e.g.,
//! cuDNN's fused attention kernels). It is therefore staged as calls of named composites (refer to
//! [`Composite::with_named_composite`]), whose keys (refer to [`AttentionKey`]) identify the computation for backends,
//! kernel persistence, and shared lowering.
//!
//! Named composites are owned by their callers, so attention is staged through functions that are created once and
//! called many times, which are [`dot_product_attention`] for inference and
//! [`differentiable_dot_product_attention`](super::differentiable_dot_product_attention) for training. Each function
//! owns one cache of definitions with one named composite per definition kind, residual request, and optional-input
//! signature. Its calls therefore share the traced bodies, their lowering, and their rule specializations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use crate::batching::{BatchableType, BatchingLevelExtent};
use crate::contexts::StagingContext;
use crate::operations::composites::{
    BatchedComposite, BroadcastInvariantBatching, Composite, CompositeAttribute, CompositeBatching,
    CompositeCallContext, CompositeKey, CustomRuleDefinition, CustomRuleTracer, composite,
};
use crate::parameters::Parameterized as _;
use crate::tracing::{DomainTracer, DomainTracingContext, TracingContext};

use super::composition::{AttentionUniverse, extents};
use super::*;

/// Name of the forward named attention definition (refer to [`AttentionKey`]).
pub const DOT_PRODUCT_ATTENTION_COMPOSITE_NAME: &str = "ryft.dot_product_attention";

/// Name of the backward named attention definition (refer to [`AttentionKey`]).
pub const DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME: &str = "ryft.dot_product_attention_backward";

/// Name of the named composite that stages differentiable attention (refer to
/// [`differentiable_dot_product_attention`](super::differentiable_dot_product_attention)). Its key has the attributes
/// of the forward definition's key except for `residual` (refer to [`AttentionKey`]).
pub const DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME: &str = "ryft.dot_product_attention_training";

/// Version of the named attention definitions, which changes whenever their semantics or the encoding of their keys
/// changes.
pub const DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION: u32 = 1;

/// Named attention definition that an [`AttentionKey`] identifies.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AttentionDefinitionKind {
    /// Forward attention, whose inputs are the [`AttentionInputs`] and whose outputs are the attended output, followed
    /// by the log-sum-exp residual when [`AttentionConfiguration::return_residual`] is set.
    Forward,

    /// Analytical backward attention, whose inputs are the [`AttentionBackwardInputs`] and whose outputs are the
    /// cotangents of the query, key, and value, followed by that of a present bias whose cotangent space is not
    /// zero-dimensional.
    Backward,
}

/// Semantic identity of one named attention definition, consisting of its kind, its value-independent configuration,
/// and the signature of its optional inputs. It converts losslessly to and from the [`CompositeKey`] of the
/// definition's named composite, which is how backends recognize attention calls (e.g., to replace them with fused
/// kernels). The key's attributes are, in order:
///
///   - `scale`: Optional explicit score scale.
///   - `causal`: Whether causal masking is enabled.
///   - `local_window`: Optional inclusive `[left, right]` local-window radii.
///   - `implementation`: Requested [`AttentionImplementation`] (`0` for automatic, `1` for portable, and `2` for fused).
///   - `dropout_rate`: Optional dropout rate.
///   - `residual`: Whether the log-sum-exp residual is returned (forward definitions only).
///   - `bias`, `mask`, `query_sequence_lengths`, `key_value_sequence_lengths`, and `dropout_state`: Whether the
///     corresponding optional input is present.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AttentionKey {
    /// Kind of the definition.
    kind: AttentionDefinitionKind,

    /// Value-independent configuration of the definition.
    configuration: AttentionConfiguration,

    /// Signature of the optional inputs of the definition.
    signature: AttentionInputSignature,
}

impl AttentionKey {
    /// Creates a new [`AttentionKey`]. Backward definitions ignore [`AttentionConfiguration::return_residual`], which
    /// is therefore cleared for them.
    #[inline]
    pub fn new(
        kind: AttentionDefinitionKind,
        configuration: AttentionConfiguration,
        signature: AttentionInputSignature,
    ) -> Self {
        let configuration = match kind {
            AttentionDefinitionKind::Forward => configuration,
            AttentionDefinitionKind::Backward => configuration.with_residual(false),
        };
        Self { kind, configuration, signature }
    }

    /// Returns the kind of the definition.
    #[inline]
    pub fn kind(&self) -> AttentionDefinitionKind {
        self.kind
    }

    /// Returns the value-independent configuration of the definition.
    #[inline]
    pub fn configuration(&self) -> AttentionConfiguration {
        self.configuration
    }

    /// Returns the signature of the optional inputs of the definition.
    #[inline]
    pub fn signature(&self) -> AttentionInputSignature {
        self.signature
    }

    /// Returns the [`CompositeKey`] of the named composite of this definition.
    pub fn to_composite_key(&self) -> CompositeKey {
        let (name, residual) = match self.kind {
            AttentionDefinitionKind::Forward => {
                (DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, Some(self.configuration.return_residual()))
            }
            AttentionDefinitionKind::Backward => (DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, None),
        };
        attention_composite_key(name, self.configuration, self.signature, residual)
    }

    /// Returns the [`AttentionKey`] that `key` encodes, or [`None`] when `key` does not name an attention definition.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when `key` names an attention definition but has another version or attributes that
    /// differ from the encoding that [`Self::to_composite_key`] produces (e.g., missing, additional, reordered, or
    /// malformed attributes).
    pub fn from_composite_key(key: &CompositeKey) -> Result<Option<Self>, TypeError> {
        let kind = match key.name() {
            DOT_PRODUCT_ATTENTION_COMPOSITE_NAME => AttentionDefinitionKind::Forward,
            DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME => AttentionDefinitionKind::Backward,
            _ => return Ok(None),
        };
        if key.version() != DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION {
            return Err(TypeError::invalid(format!(
                "named composite `{key}` has attention definition version {}, but only version \
                 {DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION} is supported",
                key.version(),
            )));
        }
        let malformed = || TypeError::invalid(format!("named composite `{key}` is not a valid attention definition"));
        let attribute = |name: &str| {
            key.attributes()
                .iter()
                .find(|(attribute, _)| attribute == name)
                .map(|(_, value)| value)
                .ok_or_else(malformed)
        };
        let boolean = |name: &str| match attribute(name)? {
            CompositeAttribute::Boolean(value) => Ok(*value),
            _ => Err(malformed()),
        };
        let optional_float = |name: &str| match attribute(name)? {
            CompositeAttribute::Optional(None) => Ok(None),
            CompositeAttribute::Optional(Some(value)) => match value.as_ref() {
                CompositeAttribute::Float(value) => Ok(Some(*value)),
                _ => Err(malformed()),
            },
            _ => Err(malformed()),
        };
        let local_window = match attribute("local_window")? {
            CompositeAttribute::Optional(None) => None,
            CompositeAttribute::Optional(Some(value)) => match value.as_ref() {
                CompositeAttribute::Sequence(values) => match values.as_slice() {
                    [CompositeAttribute::Integer(left), CompositeAttribute::Integer(right)] => Some((
                        usize::try_from(*left).map_err(|_| malformed())?,
                        usize::try_from(*right).map_err(|_| malformed())?,
                    )),
                    _ => return Err(malformed()),
                },
                _ => return Err(malformed()),
            },
            _ => return Err(malformed()),
        };
        let implementation = match attribute("implementation")? {
            CompositeAttribute::Integer(0) => AttentionImplementation::Automatic,
            CompositeAttribute::Integer(1) => AttentionImplementation::Portable,
            CompositeAttribute::Integer(2) => AttentionImplementation::Fused,
            _ => return Err(malformed()),
        };
        let return_residual = match kind {
            AttentionDefinitionKind::Forward => boolean("residual")?,
            AttentionDefinitionKind::Backward => false,
        };
        let configuration = AttentionConfiguration::new()
            .with_scale(optional_float("scale")?)
            .with_causal(boolean("causal")?)
            .with_local_window(local_window)
            .with_implementation(implementation)
            .with_dropout(optional_float("dropout_rate")?)
            .with_residual(return_residual);
        let signature = AttentionInputSignature::new(
            boolean("bias")?,
            boolean("mask")?,
            boolean("query_sequence_lengths")?,
            boolean("key_value_sequence_lengths")?,
        )
        .with_dropout_state(boolean("dropout_state")?);
        let decoded = Self::new(kind, configuration, signature);
        if &decoded.to_composite_key() != key {
            return Err(malformed());
        }
        Ok(Some(decoded))
    }
}

/// Returns the [`CompositeKey`] named `name` with the attributes of `configuration` and `signature`, including the
/// `residual` attribute when `residual` is present (refer to [`AttentionKey`] for the encoding).
pub(super) fn attention_composite_key(
    name: &'static str,
    configuration: AttentionConfiguration,
    signature: AttentionInputSignature,
    residual: Option<bool>,
) -> CompositeKey {
    let implementation = match configuration.implementation() {
        AttentionImplementation::Automatic => 0u8,
        AttentionImplementation::Portable => 1,
        AttentionImplementation::Fused => 2,
    };
    let key = CompositeKey::new(name, DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION)
        .with_attribute("scale", configuration.scale())
        .with_attribute("causal", configuration.causal())
        .with_attribute("local_window", configuration.local_window())
        .with_attribute("implementation", implementation)
        .with_attribute("dropout_rate", configuration.dropout());
    let key = match residual {
        Some(residual) => key.with_attribute("residual", residual),
        None => key,
    };
    key.with_attribute("bias", signature.has_bias())
        .with_attribute("mask", signature.has_mask())
        .with_attribute("query_sequence_lengths", signature.has_query_sequence_lengths())
        .with_attribute("key_value_sequence_lengths", signature.has_key_value_sequence_lengths())
        .with_attribute("dropout_state", signature.has_dropout_state())
}

/// Inputs of the backward attention definition, consisting of the forward [`AttentionInputs`], the attended forward
/// output, its log-sum-exp residual, and the incoming cotangent of the output. Their canonical order is that of the
/// forward inputs followed by the output, the residual, and the output cotangent.
#[derive(Clone, Debug, PartialEq, Parameterized)]
pub struct AttentionBackwardInputs<P: Parameter> {
    /// Forward inputs, including the present optional inputs.
    pub inputs: AttentionInputs<P>,

    /// Attended forward output.
    pub output: P,

    /// Log-sum-exp residual of the forward output.
    pub residual: P,

    /// Incoming cotangent of the forward output.
    pub output_cotangent: P,
}

impl<P: Parameter> AttentionBackwardInputs<P> {
    /// Returns the values of these inputs in their canonical order.
    pub fn into_values(self) -> Vec<P> {
        let mut values = self.inputs.into_values();
        values.extend([self.output, self.residual, self.output_cotangent]);
        values
    }

    /// Returns the [`AttentionBackwardInputs`] whose canonical values are `values`, for forward inputs with the
    /// provided optional-input signature.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when the number of values does not match `signature`.
    pub fn from_values(signature: AttentionInputSignature, values: &[P]) -> Result<Self, TypeError>
    where
        P: Clone,
    {
        let input_count = 3 + signature.count();
        if values.len() != input_count + 3 {
            return Err(TypeError::invalid(format!(
                "attention backward inputs for signature {signature:?} have {} values but got {}",
                input_count + 3,
                values.len(),
            )));
        }
        Ok(Self {
            inputs: AttentionInputs::from_values(signature, &values[..input_count])?,
            output: values[input_count].clone(),
            residual: values[input_count + 1].clone(),
            output_cotangent: values[input_count + 2].clone(),
        })
    }
}

/// [`CustomRuleTracer`] of the operation family `(V, O)`.
type RuleTracer<V, O> = CustomRuleTracer<V, O>;

/// Named composite of a forward definition that batches its calls by folding.
type ForwardFolding<V, O> = BatchedComposite<V, O, AttentionInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>;

/// Named composite of a backward definition that batches its calls by folding.
type BackwardFolding<V, O> = BatchedComposite<V, O, AttentionBackwardInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>;

/// Rule configurations that the named attention definitions of the operation family `(V, O)` require, which collects
/// the bounds of their batching rules so that generic callers state one bound. It is implemented for every
/// [`CustomRuleDefinition`] that supports them, which holds for every operation family whose operations support
/// partial evaluation and differentiation (e.g., the [`ArrayOperation`](crate::arrays::ArrayOperation) family and
/// backend families).
pub trait AttentionRules<V: Value, O: Operation<Type = V::Type>>:
    CompositeBatching<
        BroadcastInvariantBatching<V, O, AttentionInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>,
        RuleTracer<V, O>,
        AttentionInputs<RuleTracer<V, O>>,
        Vec<RuleTracer<V, O>>,
    > + CompositeBatching<
        BroadcastInvariantBatching<V, O, AttentionBackwardInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>,
        RuleTracer<V, O>,
        AttentionBackwardInputs<RuleTracer<V, O>>,
        Vec<RuleTracer<V, O>>,
    >
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
}

impl<V: Value, O: Operation<Type = V::Type>, R> AttentionRules<V, O> for R
where
    R: CompositeBatching<
            BroadcastInvariantBatching<V, O, AttentionInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>,
            RuleTracer<V, O>,
            AttentionInputs<RuleTracer<V, O>>,
            Vec<RuleTracer<V, O>>,
        > + CompositeBatching<
            BroadcastInvariantBatching<V, O, AttentionBackwardInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>,
            RuleTracer<V, O>,
            AttentionBackwardInputs<RuleTracer<V, O>>,
            Vec<RuleTracer<V, O>>,
        >,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
}

/// Named composite of one attention definition. Definitions without dropout batch their calls by folding mapped axes
/// into attention's batch axis and calling the same definition at the folded boundary, which keeps the definition (and
/// therefore any backend implementation of it) for batched calls. Definitions with dropout are batched structurally
/// instead, because folding would change the shape of the dropout mask and therefore the mask of every batch item,
/// except that definitions with dropout and the fused implementation reject batching (refer to
/// [`fused_dropout_batching_error`]).
enum AttentionFunction<Folding, Structural> {
    /// Definition that folds mapped axes into attention's batch axis.
    Folding(Arc<Folding>),

    /// Definition that is batched structurally.
    Structural(Arc<Structural>),
}

impl<Folding, Structural> Clone for AttentionFunction<Folding, Structural> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Folding(function) => Self::Folding(function.clone()),
            Self::Structural(function) => Self::Structural(function.clone()),
        }
    }
}

/// Named composite of a forward definition.
type ForwardFunction<V, O> =
    AttentionFunction<ForwardFolding<V, O>, Composite<V, O, AttentionInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>>;

/// Named composite of a backward definition.
type BackwardFunction<V, O> = AttentionFunction<
    BackwardFolding<V, O>,
    Composite<V, O, AttentionBackwardInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>,
>;

/// Named attention definitions of one operation family `(V, O)` and one [`AttentionConfiguration`], with one named
/// composite per [`AttentionKey`] (i.e., per definition kind, residual request, and optional-input signature), each
/// created on first use. The functions that stage attention own this cache, so its definitions, their traced bodies,
/// and their rule specializations live exactly as long as those functions.
pub(super) struct AttentionDefinitions<V: Value, O: Operation<Type = V::Type>>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    /// Configuration of every definition in this cache. Forward definitions override its residual request.
    configuration: AttentionConfiguration,

    /// Forward definitions by residual request and optional-input signature.
    forward: Mutex<HashMap<(bool, AttentionInputSignature), ForwardFunction<V, O>>>,

    /// Backward definitions by optional-input signature.
    backward: Mutex<HashMap<AttentionInputSignature, BackwardFunction<V, O>>>,
}

impl<V: Value, O: Operation<Type = V::Type>> AttentionDefinitions<V, O>
where
    V: 'static,
    O: 'static,
    V::Type: AttentionUniverse<RuleTracer<V, O>> + BatchableType,
    CustomRuleDefinition<V, O>: AttentionRules<V, O>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>
        + CompositeCallContext<Type = V::Type, Value = RuleTracer<V, O>, Constant = V, Operation = O>,
{
    /// Creates an empty cache of the definitions with the provided configuration.
    pub(super) fn new(configuration: AttentionConfiguration) -> Self {
        Self { configuration, forward: Mutex::new(HashMap::new()), backward: Mutex::new(HashMap::new()) }
    }

    /// Returns the configuration of the definitions in this cache.
    #[inline]
    pub(super) fn configuration(&self) -> AttentionConfiguration {
        self.configuration
    }

    /// Stages one call of the forward definition with the provided residual request at `inputs`, returning the
    /// attended output followed by its log-sum-exp residual when `return_residual` is set.
    pub(super) fn forward<C, X>(
        &self,
        inputs: AttentionInputs<X>,
        return_residual: bool,
    ) -> Result<Vec<X>, ProgramError>
    where
        C: CompositeCallContext<Type = V::Type, Value = X, Constant = V, Operation = O>,
        X: Value<Type = V::Type, Domain = C>,
    {
        let signature = inputs.signature();
        let function = self
            .forward
            .lock()
            .expect("attention definitions mutex is poisoned")
            .entry((return_residual, signature))
            .or_insert_with(|| forward_function(self.configuration.with_residual(return_residual), signature))
            .clone();
        match function {
            AttentionFunction::Folding(function) => function.call(inputs),
            AttentionFunction::Structural(function) => function.call(inputs),
        }
    }

    /// Stages one call of the backward definition at `inputs`, returning the cotangents of the query, key, and value,
    /// followed by that of a present bias whose cotangent space is not zero-dimensional.
    pub(super) fn backward<C, X>(&self, inputs: AttentionBackwardInputs<X>) -> Result<Vec<X>, ProgramError>
    where
        C: CompositeCallContext<Type = V::Type, Value = X, Constant = V, Operation = O>,
        X: Value<Type = V::Type, Domain = C>,
    {
        let signature = inputs.inputs.signature();
        let function = self
            .backward
            .lock()
            .expect("attention definitions mutex is poisoned")
            .entry(signature)
            .or_insert_with(|| backward_function(self.configuration, signature))
            .clone();
        match function {
            AttentionFunction::Folding(function) => function.call(inputs),
            AttentionFunction::Structural(function) => function.call(inputs),
        }
    }
}

/// Returns the array types of `types`, which belong to the universe of the values `V`.
fn array_types<V: Value<Type: AttentionUniverse<V>>>(types: Vec<V::Type>) -> Result<Vec<ArrayType>, ProgramError> {
    Ok(types.iter().map(<V::Type as AttentionUniverse<V>>::to_array_type).collect::<Result<Vec<_>, _>>()?)
}

/// Returns an error for a call of a definition whose optional-input signature is `actual` rather than its own.
fn signature_mismatch(name: &str, expected: AttentionInputSignature, actual: AttentionInputSignature) -> ProgramError {
    TypeError::invalid(format!(
        "named composite `{name}` was defined for the optional-input signature {expected:?} but was called with \
         {actual:?}",
    ))
    .into()
}

/// Returns the error that rejects batching a call of the definition named `name` with dropout and the fused
/// implementation. A batched call loses its definition's identity (i.e., it lowers its structurally batched body), so
/// it would silently replace the fused kernels and their dropout mask with the portable body and its mask. That would
/// both change the requested numerics and break the agreement between the dropout masks of the forward and backward
/// definitions when only one of them is batched (e.g., when the pullback of an unbatched call is batched).
fn fused_dropout_batching_error(name: &str) -> ProgramError {
    ProgramError::UnsupportedOperation {
        message: format!(
            "named composite `{name}` with dropout cannot be batched with the fused implementation, because a batched \
             call cannot use the fused kernels or reproduce their dropout mask; use the portable implementation or a \
             dropout state",
        ),
    }
}

/// Returns the named composite of the forward definition with the provided configuration and optional-input signature.
fn forward_function<V: Value, O: Operation<Type = V::Type>>(
    configuration: AttentionConfiguration,
    signature: AttentionInputSignature,
) -> ForwardFunction<V, O>
where
    V: 'static,
    O: 'static,
    V::Type: AttentionUniverse<RuleTracer<V, O>> + BatchableType,
    CustomRuleDefinition<V, O>: AttentionRules<V, O>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>
        + CompositeCallContext<Type = V::Type, Value = RuleTracer<V, O>, Constant = V, Operation = O>,
{
    let key = AttentionKey::new(AttentionDefinitionKind::Forward, configuration, signature);
    let output_types = move |input_types: AttentionInputs<V::Type>| -> Result<Vec<V::Type>, ProgramError> {
        if input_types.signature() != signature {
            return Err(signature_mismatch(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, signature, input_types.signature()));
        }
        let input_types = array_types::<RuleTracer<V, O>>(input_types.into_values())?;
        let output_types = attention_forward_output_types(
            DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
            configuration,
            signature,
            input_types.as_slice(),
        )?;
        Ok(output_types
            .into_iter()
            .map(<V::Type as AttentionUniverse<RuleTracer<V, O>>>::from_array_type)
            .collect())
    };
    let body = move |inputs: AttentionInputs<RuleTracer<V, O>>| -> Result<Vec<RuleTracer<V, O>>, ProgramError> {
        let (output, residual) = dot_product_attention_ir_composition(&inputs, configuration)?;
        Ok(std::iter::once(output)
            .chain(residual)
            .map(<V::Type as AttentionUniverse<RuleTracer<V, O>>>::from_array)
            .collect())
    };
    let function = composite(body)
        .with_name(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME)
        .with_named_composite(key.to_composite_key(), output_types);
    if configuration.dropout().is_some() {
        return match configuration.implementation() {
            AttentionImplementation::Fused => AttentionFunction::Folding(Arc::new(
                function
                    .with_batching(|_, _, _| Err(fused_dropout_batching_error(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME))),
            )),
            AttentionImplementation::Automatic | AttentionImplementation::Portable => {
                AttentionFunction::Structural(Arc::new(function))
            }
        };
    }
    AttentionFunction::Folding(Arc::new_cyclic(|weak: &Weak<ForwardFolding<V, O>>| {
        let weak = weak.clone();
        let input_roles = attention_batch_input_roles(signature, AttentionDefinitionKind::Forward);
        function.with_batching(move |inputs, axes: AttentionInputs<BatchAxis>, extent| {
            // A staged call outlives the function that staged it when its program is batched after the function is
            // dropped, in which case the folded call stages an equivalent definition.
            let function = match weak.upgrade() {
                Some(function) => function,
                None => match forward_function::<V, O>(configuration, signature) {
                    AttentionFunction::Folding(function) => function,
                    AttentionFunction::Structural(_) => unreachable!("definitions without dropout batch by folding"),
                },
            };
            let output_roles =
                |output_count| attention_batch_output_roles(signature, AttentionDefinitionKind::Forward, output_count);
            batch_attention(
                inputs.into_values(),
                axes.into_values(),
                &extent,
                input_roles.as_slice(),
                output_roles,
                |inputs| function.call(AttentionInputs::from_values(signature, inputs.as_slice())?),
            )
        })
    }))
}

/// Returns the named composite of the backward definition with the provided configuration and optional-input
/// signature.
fn backward_function<V: Value, O: Operation<Type = V::Type>>(
    configuration: AttentionConfiguration,
    signature: AttentionInputSignature,
) -> BackwardFunction<V, O>
where
    V: 'static,
    O: 'static,
    V::Type: AttentionUniverse<RuleTracer<V, O>> + BatchableType,
    CustomRuleDefinition<V, O>: AttentionRules<V, O>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>
        + CompositeCallContext<Type = V::Type, Value = RuleTracer<V, O>, Constant = V, Operation = O>,
{
    let key = AttentionKey::new(AttentionDefinitionKind::Backward, configuration, signature);
    let configuration = key.configuration();
    let output_types = move |input_types: AttentionBackwardInputs<V::Type>| -> Result<Vec<V::Type>, ProgramError> {
        if input_types.inputs.signature() != signature {
            return Err(signature_mismatch(
                DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
                signature,
                input_types.inputs.signature(),
            ));
        }
        let input_types = array_types::<RuleTracer<V, O>>(input_types.into_values())?;
        let output_types = attention_backward_output_types(
            DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
            configuration,
            signature,
            input_types.as_slice(),
        )?;
        Ok(output_types
            .into_iter()
            .map(<V::Type as AttentionUniverse<RuleTracer<V, O>>>::from_array_type)
            .collect())
    };
    let body = move |inputs: AttentionBackwardInputs<RuleTracer<V, O>>| -> Result<Vec<RuleTracer<V, O>>, ProgramError> {
        let outputs = dot_product_attention_backward_ir_composition(
            &inputs.inputs,
            &inputs.output,
            &inputs.residual,
            &inputs.output_cotangent,
            configuration,
        )?;
        Ok(outputs.into_iter().map(<V::Type as AttentionUniverse<RuleTracer<V, O>>>::from_array).collect())
    };
    let function = composite(body)
        .with_name(DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME)
        .with_named_composite(key.to_composite_key(), output_types);
    if configuration.dropout().is_some() {
        return match configuration.implementation() {
            AttentionImplementation::Fused => {
                AttentionFunction::Folding(Arc::new(function.with_batching(|_, _, _| {
                    Err(fused_dropout_batching_error(DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME))
                })))
            }
            AttentionImplementation::Automatic | AttentionImplementation::Portable => {
                AttentionFunction::Structural(Arc::new(function))
            }
        };
    }
    AttentionFunction::Folding(Arc::new_cyclic(|weak: &Weak<BackwardFolding<V, O>>| {
        let weak = weak.clone();
        let input_roles = attention_batch_input_roles(signature, AttentionDefinitionKind::Backward);
        function.with_batching(move |inputs, axes: AttentionBackwardInputs<BatchAxis>, extent| {
            // Refer to the forward rule for when the function may have been dropped.
            let function = match weak.upgrade() {
                Some(function) => function,
                None => match backward_function::<V, O>(configuration, signature) {
                    AttentionFunction::Folding(function) => function,
                    AttentionFunction::Structural(_) => unreachable!("definitions without dropout batch by folding"),
                },
            };
            let output_roles =
                |output_count| attention_batch_output_roles(signature, AttentionDefinitionKind::Backward, output_count);
            batch_attention(
                inputs.into_values(),
                axes.into_values(),
                &extent,
                input_roles.as_slice(),
                output_roles,
                |inputs| function.call(AttentionBackwardInputs::from_values(signature, inputs.as_slice())?),
            )
        })
    }))
}

/// Semantic role of one attention batching input.
#[derive(Copy, Clone, Debug)]
enum AttentionBatchInput {
    /// Query, key, value, output, or output cotangent in `TNH` or `BTNH` form.
    Tensor,

    /// Broadcastable bias or Boolean mask, right-aligned to `BNTS`.
    Score,

    /// Per-batch sequence lengths.
    Length,

    /// Log-sum-exp statistic in `TN` or `BTN` form.
    Statistic,
}

/// Semantic role of one attention batching output.
#[derive(Copy, Clone, Debug)]
enum AttentionBatchOutput {
    /// Tensor whose logical rank follows the input at this index.
    Tensor(usize),

    /// Log-sum-exp statistic whose logical rank follows the query rank.
    Statistic,

    /// Bias cotangent restored to the logical bias shape of the input at this index.
    Score(usize),
}

/// Returns the batching roles of the canonical inputs of a definition of the provided kind and optional-input
/// signature. Dropout states never reach batching rules, because definitions with dropout are batched structurally.
fn attention_batch_input_roles(
    signature: AttentionInputSignature,
    kind: AttentionDefinitionKind,
) -> Vec<AttentionBatchInput> {
    let mut roles = vec![AttentionBatchInput::Tensor; 3];
    roles.extend(signature.has_bias().then_some(AttentionBatchInput::Score));
    roles.extend(signature.has_mask().then_some(AttentionBatchInput::Score));
    roles.extend(signature.has_query_sequence_lengths().then_some(AttentionBatchInput::Length));
    roles.extend(signature.has_key_value_sequence_lengths().then_some(AttentionBatchInput::Length));
    if kind == AttentionDefinitionKind::Backward {
        roles.extend([AttentionBatchInput::Tensor, AttentionBatchInput::Statistic, AttentionBatchInput::Tensor]);
    }
    roles
}

/// Returns the batching roles of the `output_count` outputs of a definition of the provided kind and optional-input
/// signature. A backward definition returns a bias cotangent, whose bias is always the fourth canonical input, only
/// when the bias is present and its cotangent space is not zero-dimensional.
fn attention_batch_output_roles(
    signature: AttentionInputSignature,
    kind: AttentionDefinitionKind,
    output_count: usize,
) -> Vec<AttentionBatchOutput> {
    match kind {
        AttentionDefinitionKind::Forward => [AttentionBatchOutput::Tensor(0), AttentionBatchOutput::Statistic]
            .into_iter()
            .take(output_count)
            .collect(),
        AttentionDefinitionKind::Backward => {
            let mut roles =
                vec![AttentionBatchOutput::Tensor(0), AttentionBatchOutput::Tensor(1), AttentionBatchOutput::Tensor(2)];
            if signature.has_bias() && output_count == 4 {
                roles.push(AttentionBatchOutput::Score(3));
            }
            roles
        }
    }
}

/// Batches one attention call by folding its mapped axis into attention's batch axis, staging the definition once at
/// the folded boundary through `call`, and restoring the outputs, which are all mapped at axis `0`.
///
/// Every input is first aligned to a leading mapped axis and then folded into the logical batch axis. Rank-three
/// attention has an implicit logical batch of one, so `[v, T, N, H]` already has the canonical rank-four form, while
/// rank-four attention folds `[v, B, T, N, H]` to `[v · B, T, N, H]`. Biases and masks are first materialized to
/// `[v, B, N, T, S]`, which handles every broadcastable rank and preserves mapped bias and mask inputs. Outputs reverse
/// the normalization, and a bias cotangent is reduced over precisely the axes that its logical bias input broadcasts.
/// Every extent comes from the values' [`AttentionUniverse`], so this supports static extents and, in array IR
/// families, first-class dynamic extents (i.e., a dynamic mapped extent, logical batch, or sequence length never
/// becomes host metadata). Calls whose inputs are all replicated stage the unbatched definition.
fn batch_attention<T: Value<Type: AttentionUniverse<T>>>(
    inputs: Vec<T>,
    axes: Vec<BatchAxis>,
    extent: &BatchingLevelExtent<T>,
    input_roles: &[AttentionBatchInput],
    output_roles: impl FnOnce(usize) -> Vec<AttentionBatchOutput>,
    call: impl FnOnce(Vec<T>) -> Result<Vec<T>, ProgramError>,
) -> Result<(Vec<T>, Vec<BatchAxis>), ProgramError> {
    check_count!("input", input_roles, inputs.len(), ProgramError);
    check_count!("input", axes, inputs.len(), ProgramError);
    if axes.iter().all(BatchAxis::is_replicated) {
        let outputs = call(inputs)?;
        let output_axes = vec![BatchAxis::replicated(); outputs.len()];
        return Ok((outputs, output_axes));
    }

    // Align every input to a leading mapped axis, moving mapped axes and broadcasting replicated inputs.
    let mapped_extent = T::Type::batch_extent(&inputs[0], extent)?;
    let aligned = inputs
        .iter()
        .zip(&axes)
        .map(|(input, axis)| {
            let rank = T::Type::array_type(input)?.rank();
            match axis.axis() {
                Some(axis) => {
                    let position = axis.normalize(rank).map_err(|error| TypeError::invalid(error.to_string()))?;
                    if position == 0 {
                        return Ok(input.clone());
                    }
                    let permutation =
                        std::iter::once(position).chain((0..rank).filter(|axis| *axis != position)).collect::<Vec<_>>();
                    Ok(T::Type::from_array(T::Type::into_array(input.clone())?.transpose(permutation)?))
                }
                None => {
                    let output_extents =
                        std::iter::once(mapped_extent.clone()).chain(extents(input)?).collect::<Vec<_>>();
                    T::Type::broadcast(input, output_extents.as_slice(), (1..=rank).collect::<Vec<_>>().as_slice())
                }
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let logical_rank = |value: &T| -> Result<usize, ProgramError> { Ok(T::Type::array_type(value)?.rank() - 1) };
    let query_rank = logical_rank(&aligned[0])?;
    let key_rank = logical_rank(&aligned[1])?;
    let batch_extent =
        if query_rank == 4 { T::Type::extent(&aligned[0], 1)? } else { T::Type::constant_extent(&aligned[0], 1)? };
    let folded_batch_extent = T::Type::multiply_extents(&aligned[0], &mapped_extent, &batch_extent)?;

    // Biases and masks are normalized to the canonical `[v, B, N, T, S]` score shape, whose extents are obtained only
    // when the call has such inputs (and therefore possibly a bias cotangent output).
    let score_extents = if input_roles.iter().any(|role| matches!(role, AttentionBatchInput::Score)) {
        let query_sequence_extent = T::Type::extent(&aligned[0], query_rank - 2)?;
        let head_extent = T::Type::extent(&aligned[0], query_rank - 1)?;
        let key_sequence_extent = T::Type::extent(&aligned[1], key_rank - 2)?;
        Some((
            [
                mapped_extent.clone(),
                batch_extent.clone(),
                head_extent.clone(),
                query_sequence_extent.clone(),
                key_sequence_extent.clone(),
            ],
            [folded_batch_extent.clone(), head_extent, query_sequence_extent, key_sequence_extent],
        ))
    } else {
        None
    };
    let score_extents = || {
        score_extents.as_ref().ok_or_else(|| {
            ProgramError::from(TypeError::invalid("attention batching produced a bias cotangent without a bias"))
        })
    };

    // Fold the mapped axis into attention's batch axis and stage the definition at the folded boundary.
    let folded = aligned
        .iter()
        .zip(input_roles)
        .map(|(value, role)| {
            let rank = T::Type::array_type(value)?.rank();
            let trailing_extents = |start: usize| {
                (start..rank).map(|axis| T::Type::extent(value, axis)).collect::<Result<Vec<_>, ProgramError>>()
            };
            match role {
                AttentionBatchInput::Tensor if rank == 5 => {
                    let batch_extent = T::Type::multiply_extents(value, &mapped_extent, &T::Type::extent(value, 1)?)?;
                    let folded_extents = std::iter::once(batch_extent).chain(trailing_extents(2)?).collect::<Vec<_>>();
                    T::Type::reshape(value, folded_extents.as_slice())
                }
                AttentionBatchInput::Tensor => Ok(value.clone()),
                AttentionBatchInput::Score => {
                    let (score_extents, folded_score_extents) = score_extents()?;
                    let output_axes =
                        std::iter::once(0).chain((1..rank).map(|axis| 5 - rank + axis)).collect::<Vec<_>>();
                    let normalized = T::Type::broadcast(value, score_extents, output_axes.as_slice())?;
                    T::Type::reshape(&normalized, folded_score_extents)
                }
                AttentionBatchInput::Length => T::Type::reshape(value, std::slice::from_ref(&folded_batch_extent)),
                AttentionBatchInput::Statistic if rank == 4 => {
                    let folded_extents =
                        std::iter::once(folded_batch_extent.clone()).chain(trailing_extents(2)?).collect::<Vec<_>>();
                    T::Type::reshape(value, folded_extents.as_slice())
                }
                AttentionBatchInput::Statistic => Ok(value.clone()),
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let outputs = call(folded)?;
    let output_roles = output_roles(outputs.len());
    check_count!("output", output_roles, outputs.len(), ProgramError);

    // Restore the mapped axis of every output and reduce bias cotangents over the axes that their biases broadcast.
    let outputs = outputs
        .into_iter()
        .zip(output_roles)
        .map(|(output, role)| {
            let trailing_extents = |output: &T| {
                let rank = T::Type::array_type(output)?.rank();
                (1..rank).map(|axis| T::Type::extent(output, axis)).collect::<Result<Vec<_>, ProgramError>>()
            };
            match role {
                AttentionBatchOutput::Tensor(input_index) if logical_rank(&aligned[input_index])? == 4 => {
                    let input_batch_extent = T::Type::extent(&aligned[input_index], 1)?;
                    let unfolded_extents = [mapped_extent.clone(), input_batch_extent]
                        .into_iter()
                        .chain(trailing_extents(&output)?)
                        .collect::<Vec<_>>();
                    T::Type::reshape(&output, unfolded_extents.as_slice())
                }
                AttentionBatchOutput::Tensor(_) => Ok(output),
                AttentionBatchOutput::Statistic if query_rank == 4 => {
                    let unfolded_extents = [mapped_extent.clone(), batch_extent.clone()]
                        .into_iter()
                        .chain(trailing_extents(&output)?)
                        .collect::<Vec<_>>();
                    T::Type::reshape(&output, unfolded_extents.as_slice())
                }
                AttentionBatchOutput::Statistic => Ok(output),
                AttentionBatchOutput::Score(input_index) => {
                    let bias_type = T::Type::array_type(&aligned[input_index])?;
                    let offset = 5 - bias_type.rank();
                    let reduction_axes = (0..4)
                        .filter(|&axis| {
                            axis < offset || matches!(bias_type.shape()[axis - offset + 1], Dimension::Static(1))
                        })
                        .map(|axis| axis + 1)
                        .collect::<Vec<_>>();
                    let normalized = T::Type::reshape(&output, &score_extents()?.0)?;
                    let reduced =
                        T::Type::into_array(normalized)?.reduce(reduction_axes.as_slice(), ReductionKind::Sum)?;
                    T::Type::reshape(&T::Type::from_array(reduced), extents(&aligned[input_index])?.as_slice())
                }
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let output_axes = vec![BatchAxis::from_position(0); outputs.len()];
    Ok((outputs, output_axes))
}

/// Returns `inputs` with their manual variation aligned (refer to [`ManualVariationAlignment::align_manual_variation`]),
/// which precedes every call of the functions that stage attention. Alignment is the identity when every input already
/// varies over the same manual axes (e.g., outside manual regions), and so those inputs are returned unchanged without
/// staging the alignment.
pub(super) fn align_attention_inputs<U, R, X>(inputs: AttentionInputs<X>) -> Result<AttentionInputs<X>, ProgramError>
where
    U: AttentionUniverse<R>,
    X: Value<Type = U> + ManualVariationAlignment<U>,
{
    let mut varying_manual_axes = None;
    let mut aligned = true;
    for input in inputs.parameters() {
        let array_type = U::to_array_type(&input.r#type())?;
        let axes = array_type.sharding().map(|sharding| sharding.varying_manual_axes().clone()).unwrap_or_default();
        match &varying_manual_axes {
            None => varying_manual_axes = Some(axes),
            Some(expected) => aligned &= *expected == axes,
        }
    }
    if aligned {
        return Ok(inputs);
    }
    let signature = inputs.signature();
    let inputs = ManualVariationAlignment::align_manual_variation(&inputs.into_values())?;
    Ok(AttentionInputs::from_values(signature, inputs.as_slice())?)
}

/// Scaled dot-product attention inference function, built by [`dot_product_attention`]. Each call stages one
/// call of the named forward attention definition (refer to [`AttentionKey`]) for the optional-input signature of its
/// inputs, which backends may replace with their own implementation of the definition (e.g., a fused kernel). The
/// function owns its definitions, so it should be created once and called many times, which traces each definition's
/// body once per input types and shares it, together with its lowering, across those calls.
///
/// # Semantics
///
/// For each batch item and query head, attention computes
///
/// ```text
/// softmax(scale * Q * transpose(K) + bias, visibility) * V.
/// ```
///
/// The query has `TNH` or `BTNH` form, and the key and value have `SKH` or `BSKH` form, where `T` and `S` are the query
/// and key/value sequence lengths, `N` is the number of query heads, `K` is the number of key/value heads, and `H` is
/// the head dimension. `K` must divide `N`, where `K == N` is multi-head attention, `K == 1` is multi-query attention,
/// and the remaining cases are grouped-query attention. The attended output has the query type. Reduced and unreduced
/// inputs are unsupported, because softmax and the log-sum-exp residual do not preserve partial-sum or zero-filled
/// replica representations. Calls align the manual variation of their inputs before staging the call.
///
/// [`AttentionInputs`] defines the complete input boundary, consisting of the query, key, and value, followed by an
/// optional broadcastable additive bias, Boolean visibility mask, per-batch query lengths, per-batch key/value lengths,
/// and dropout state. Dot products, scaling, bias addition, and masking use the query type promoted to at least `f32`,
/// and the masked logits are then converted to `f32` for the numerically stable softmax. Visibility is the conjunction
/// of the explicit mask, causal ordering, the inclusive `(left, right)` local window, and both sequence-length
/// predicates. Invisible scores use a finite large negative value in the logits type, and query rows beyond their
/// lengths are exact zeros. Without a configured scale, attention uses `1 / sqrt(H)`. Every batch item uses the same
/// rectangular storage shape, and sequence lengths control visibility within that storage.
///
/// With [`AttentionConfiguration::return_residual`], a second `TN` or `BTN` output contains the natural-log
/// log-sum-exp statistic at the query data type. With [`AttentionConfiguration::with_dropout`], the attention weights
/// are dropped and rescaled by [`Dropout`](crate::operations::random::Dropout) with the dropout state of the inputs,
/// which determines the mask (refer to [`AttentionInputs::dropout_state`]). Dropout supports sharding only along the
/// broadcast axes of its input, so attention with dropout over sharded inputs runs inside `shard_map`.
///
/// [`AttentionImplementation::Portable`] always evaluates the portable body (refer to
/// [`dot_product_attention_ir_composition`]). [`AttentionImplementation::Fused`] requires a backend implementation
/// that supports the complete call (e.g., cuDNN's fused kernels on the XLA backend). [`AttentionImplementation::Automatic`]
/// lets backends use such an implementation when it is eligible and evaluates the portable body otherwise.
///
/// # Transforms
///
/// Batching a call folds its mapped axis into the logical batch axis and calls the same definition at the folded
/// boundary, which keeps backend implementations for batched calls, while calls with dropout are batched structurally
/// and calls with dropout and the fused implementation cannot be batched. The named definitions have no derivative
/// rules, so differentiating a call fails. Use
/// [`differentiable_dot_product_attention`](super::differentiable_dot_product_attention) for the training path.
pub struct DotProductAttention<D: Domain>
where
    DomainTracingContext<D>: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>,
{
    /// Named attention definitions staged by this function.
    definitions: AttentionDefinitions<D::Constant, D::Operation>,
}

impl<D: Domain> DotProductAttention<D>
where
    D::Constant: 'static,
    D::Operation: 'static,
    D::Type: AttentionUniverse<DomainTracer<D>> + BatchableType,
    CustomRuleDefinition<D::Constant, D::Operation>: AttentionRules<D::Constant, D::Operation>,
    DomainTracingContext<D>: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>
        + CompositeCallContext<Type = D::Type, Value = DomainTracer<D>, Constant = D::Constant, Operation = D::Operation>,
{
    /// Returns the configuration of this function.
    #[inline]
    pub fn configuration(&self) -> AttentionConfiguration {
        self.definitions.configuration()
    }

    /// Aligns the manual variation of `inputs` and stages one attention call, returning the attended output together
    /// with its log-sum-exp residual when the configuration requests it (refer to
    /// [`AttentionConfiguration::with_residual`]).
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `inputs` are invalid for the configuration (e.g., because their shapes are
    /// incompatible) or when staging the call fails.
    pub fn call<C, X>(&self, inputs: AttentionInputs<X>) -> Result<(X, Option<X>), ProgramError>
    where
        C: CompositeCallContext<Type = D::Type, Value = X, Constant = D::Constant, Operation = D::Operation>,
        X: Value<Type = D::Type, Domain = C> + ManualVariationAlignment<D::Type>,
    {
        let inputs = align_attention_inputs::<D::Type, DomainTracer<D>, X>(inputs)?;
        let return_residual = self.definitions.configuration().return_residual();
        let mut outputs = self.definitions.forward(inputs, return_residual)?.into_iter();
        let output = outputs.next().ok_or(ProgramError::InvalidOutputCount { expected: 1, actual: 0 })?;
        Ok((output, outputs.next()))
    }
}

/// Creates the scaled dot-product attention inference function with the provided configuration for the values of the
/// domain `D`, whose calls stage the named forward attention definition (refer to [`DotProductAttention`]).
///
/// The definition's portable body computes attention with ordinary array operations, while backends may replace it
/// (e.g., the XLA backend replaces it with cuDNN's fused attention kernels when the configuration and the input types
/// are eligible). Homogeneous array domains support static extents only, while array IR domains also support dynamic
/// batch and sequence extents (e.g., in bounded-dynamic programs).
///
/// # Parameters
///
///   - `configuration`: Value-independent attention semantics and implementation selection.
pub fn dot_product_attention<D: Domain>(configuration: AttentionConfiguration) -> DotProductAttention<D>
where
    D::Constant: 'static,
    D::Operation: 'static,
    D::Type: AttentionUniverse<DomainTracer<D>> + BatchableType,
    CustomRuleDefinition<D::Constant, D::Operation>: AttentionRules<D::Constant, D::Operation>,
    DomainTracingContext<D>: StagingContext<Type = D::Type, Constant = D::Constant, Operation = D::Operation>
        + CompositeCallContext<Type = D::Type, Value = DomainTracer<D>, Constant = D::Constant, Operation = D::Operation>,
{
    DotProductAttention { definitions: AttentionDefinitions::new(configuration) }
}
