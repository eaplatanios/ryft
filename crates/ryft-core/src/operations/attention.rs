//! Scaled dot-product attention, staged as calls of named composites rather than as a primitive operation. Attention
//! has a portable body built from ordinary array operations, a stabilized analytical derivative, and optional backend
//! implementations (e.g., cuDNN's fused attention kernels). Staging it through named composites (i.e., using
//! [`Composite::with_named_composite`]) keeps one body that every program transform already supports, while
//! backends recognize the calls by their keys and may replace them with their own implementations.
//!
//! # Definitions
//!
//! A named attention definition is a named composite whose [`CompositeKey`] encodes an [`AttentionKey`], which
//! identifies the forward or the backward definition and records its value-independent [`AttentionConfiguration`] and
//! the structure of its [`AttentionInputs`] (i.e., which optional inputs are present). There are two definitions:
//!
//!   - **Forward:** Named [`DOT_PRODUCT_ATTENTION_COMPOSITE_NAME`]. It maps the [`AttentionInputs`] to the attended
//!     output, followed by the log-sum-exp residual of its softmax when its key requests it.
//!   - **Backward:** Named [`DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME`]. It maps the [`AttentionResiduals`] of
//!     a forward call and the cotangent of its output (i.e., the forward inputs, the forward output, its residual, and
//!     the output cotangent, in that order) to the cotangents of the query, key, and value, followed by that of a bias
//!     whose cotangent space is not zero-dimensional.
//!
//! The keys identify the computation for backends, kernel persistence, and shared lowering. A backend decodes them
//! with [`AttentionKey::from_composite_key`] and then either lowers the portable body of the definition or replaces
//! the call with its own implementation, as the [`AttentionImplementation`] of the configuration requests.
//!
//! The portable bodies are written once with ordinary value capabilities (refer to [`AttentionOperations`]).
//! Their constants, coordinates, and broadcasts take the runtime shapes of existing values (i.e., using [`FillLike`],
//! [`IotaLike`], and [`BroadcastLike`]), and their reshapes keep the batch and sequence dimensions, so they support
//! static and dynamic batch and sequence extents for values typed by [`ArrayType`] and for array IR values (i.e.,
//! values typed by [`ArrayIrType`](crate::ArrayIrType)) alike. Only the batching rule constructs shapes that no
//! existing value has, and it computes their extents with the extent capabilities of the values, which support dynamic
//! extents there only for array IR values, whose extents are first-class dimension values. Dropout is the exception in
//! both universes, because it draws the random bits of its mask for static extents.
//!
//! # Ownership and Lifecycle
//!
//! Named composites are owned by their callers, so attention is staged through functions that are created once and
//! called many times, which are the [`DotProductAttention`] that [`dot_product_attention`] creates for inference and
//! the [`DifferentiableDotProductAttention`] that [`differentiable_dot_product_attention`] creates for training. Each
//! function owns one named composite per definition and residual request, and the training function also owns the
//! named composite that carries its reverse mode rules. Each of them serves every input structure under the key of
//! that structure (refer to [`Composite::with_named_composite_per_structure`]). The calls of a function therefore
//! share the traced bodies, their lowering, and their rule specializations for as long as the function lives.
//! Programs that contain staged calls remain valid after their function is dropped, and batching such a program
//! stages an equivalent definition.
//!
//! # Transforms
//!
//!   - **Manual Variation:** Both functions align the manual variation of their inputs before the composite boundary
//!     (refer to [`ManualVariationAlignment`]), so that the transposes of those alignments accumulate the cotangent
//!     contributions of invariant inputs exactly once.
//!   - **Batching:** A batched call folds its mapped axis into attention's batch axis and calls the same definition
//!     at the folded boundary, which keeps backend implementations for batched calls (refer to the section below).
//!     Definitions with dropout are batched structurally instead, so that every batch item keeps the dropout mask
//!     of its unbatched call, and definitions with dropout and the fused implementation reject batched calls that
//!     have a mapped input.
//!   - **Differentiation:** The forward and backward definitions have no derivative rules, so differentiating
//!     a call of the inference function fails. The differentiable function stages a third named composite, named
//!     [`DOT_PRODUCT_ATTENTION_DIFFERENTIABLE_COMPOSITE_NAME`], whose reverse mode rule calls the forward definition
//!     with its residual and then the backward definition. Reverse mode over dynamic shapes with integer inputs
//!     (e.g., sequence lengths) needs an array IR family, because homogeneous array programs cannot materialize
//!     the structurally zero tangents of such inputs for a dynamic shape.
//!
//! ## Batching
//!
//! Batching an attention call gives its inputs a mapped axis (i.e., an additional axis that holds the batch items),
//! which is one axis more than a definition accepts. The batching rule does not batch the operations of the body one
//! by one. It folds the mapped axis into the batch axis that attention already has, calls the same definition once on
//! the folded inputs, and unfolds the outputs. A batched call therefore remains a call of the named definition, which
//! lets a backend keep using its own implementation of that definition (e.g., a fused kernel).
//!
//! The shapes below name one extent per letter. `v` is the mapped extent (i.e., the number of batch items), `B` is the
//! batch extent of attention, `T` and `S` are the query and key/value sequence lengths, `N` is the number of query
//! heads, `H` is the head dimension, and `v · B` is the product of two extents. For a call whose inputs have a batch
//! axis, the rule takes the steps in the following table, in which keys and values are treated like the query.
//!
//!   | Step      | Values                              | From               | To                 |
//!   |-----------|-------------------------------------|--------------------|--------------------|
//!   | Align     | Inputs that every batch item shares | `[B, T, N, H]`     | `[v, B, T, N, H]`  |
//!   | Normalize | Biases and masks                    | e.g., `[v, T, S]`  | `[v, B, N, T, S]`  |
//!   | Fold      | Queries, keys, and values           | `[v, B, T, N, H]`  | `[v · B, T, N, H]` |
//!   | Fold      | Biases and masks                    | `[v, B, N, T, S]`  | `[v · B, N, T, S]` |
//!   | Fold      | Sequence lengths                    | `[v, B]`           | `[v · B]`          |
//!   | Unfold    | Outputs                             | `[v · B, T, N, H]` | `[v, B, T, N, H]`  |
//!
//! A call whose inputs have no batch axis needs no product, because its mapped axis becomes the batch axis, and its
//! biases and masks are normalized to `[v, N, T, S]`.
//!
//! The rule constructs every shape in the last column, and in general no value has that shape beforehand. The
//! exemplar-relative capabilities (e.g., [`BroadcastLike`]) copy the shape of an existing value, so they cannot
//! produce it. [`Reshape`] can regroup static dimensions next to the dynamic dimensions that it keeps, but `v · B`
//! is a new extent whenever `v` or `B` is dynamic. The rule therefore reads extents with [`DimensionSize`], multiplies
//! two of them with [`Mul`], and broadcasts and reshapes values to the results with [`DynamicBroadcast`] and
//! [`DynamicReshape`]. Batching the body one operation at a time needs none of them, and definitions with
//! dropout are batched that way, but such a call is no longer a call of the named definition.
//!
//! The representation of an extent is [`DimensionSize::Extent`], which differs between the universes:
//!
//!   - Tracers of homogeneous array programs use static host numbers that are read from their types, and they
//!     broadcast and reshape to static shapes. A dynamic extent has no host number, so a homogeneous array family
//!     supports dynamically shaped attention except with batched calls whose batching rule needs a dynamic extent
//!     (e.g., a dynamic mapped extent, or a dynamic extent of a rank-4 input whose mapped axis is folded into its
//!     batch axis, of a bias or mask, or of a replicated input that is broadcast along the mapped axis). Such a call
//!     fails with an error that names the dynamic axis (e.g., "`dimension_size` cannot return the dynamic extent
//!     `length` of axis 2 of `f32[2, 1, length, 1, 1]` as a host extent; use array IR values for dynamic extents"),
//!     and such programs stage attention in an array IR family instead, which is how backends support bounded
//!     dynamic shapes (e.g., XLA tracers).
//!   - Array IR values use first-class dimension values, so the batching rule supports static and dynamic extents.
//!
//! # Example
//!
//! ```rust
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, AttentionConfiguration, AttentionInputs, DataType, EagerContext,
//! #     ProgramError, dot_product_attention,
//! # };
//! # fn main() -> Result<(), ProgramError> {
//! let attention = dot_product_attention::<EagerContext<Array, ArrayOperation<Array>>>(
//!     AttentionConfiguration::new().with_causal(true),
//! );
//!
//! // Two query positions attend over two key/value positions with one head and a head dimension of one.
//! let r#type = ArrayType::new_static(DataType::F32, [2, 1, 1]);
//! let query = Array::from_elements(r#type.clone(), &[1.0f32, 1.0])?;
//! let key = Array::from_elements(r#type.clone(), &[1.0f32, 1.0])?;
//! let value = Array::from_elements(r#type, &[3.0f32, 5.0])?;
//! let output = attention.call(AttentionInputs::new(query, key, value))?;
//!
//! // The first position sees only the first value, and the second position weighs both values equally.
//! assert_eq!(output.to_f64s(), vec![3.0, 4.0]);
//! # Ok(())
//! # }
//! ```

use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Weak};

use ryft_macros::Parameterized;

use crate::arrays::{
    ArrayMemberType, ArrayType, Broadcastable, DataType, Dimension, Shape, Sharding, ShardingDimension,
};
use crate::batching::{BatchAxis, BatchableType};
use crate::contexts::{Domain, StagingContext};
use crate::differentiation::DifferentiableType;
use crate::macros::check_count;
use crate::operations::Capability;
use crate::operations::arithmetic::{Add, Div, Mul, Sub};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::composites::builders::{BatchedComposite, Composite, CompositeCallContext, composite};
use crate::operations::composites::configurations::{
    BroadcastInvariantBatching, CompositeBatching, CompositeVjp, DefaultJvp, SymbolicZeroVjp,
};
use crate::operations::composites::rules::CustomRuleTracer;
use crate::operations::composites::rules::definitions::{CompositeAttribute, CompositeKey, CustomRuleDefinition};
use crate::operations::constants::fill_like::FillLike;
use crate::operations::constants::iota_like::IotaLike;
use crate::operations::control_flow::select::Select;
use crate::operations::differentiation::stop_gradient::StopGradient;
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::dot::{Dot, DotDimensionNumbers, infer_dot_output_type};
use crate::operations::exponential::{Exp, Log};
use crate::operations::logical::And;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastLike, DynamicBroadcast};
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape, infer_reshape_output_type};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::random::{Dropout, RandomAlgorithm};
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::operations::sharding::reshard::Reshard;
use crate::parameters::{Parameter, ParameterError, Parameterized, Placeholder};
use crate::programs::{MaybeZero, Operation, ProgramError, TypeError, Typed, Value};
use crate::tracing::{DomainTracer, DomainTracingContext, TracingContext};

/// Value capabilities that staging attention requires, collected so that the functions that stage it (e.g.,
/// [`dot_product_attention`]) state one bound. Those functions create the portable bodies and the batching rule of the
/// named definitions, so they need every capability that either of them uses, while each helper of the bodies bounds
/// only what it calls. The last three members are the extent capabilities with which the batching rule folds batched
/// calls (refer to the [module documentation](self#batching)). [`Reshard`] places the scores according to the bias,
/// the mask, and the sequence lengths, and returns each cotangent with the sharding of its input, and it is the member
/// that requires [`Parameter`]. This bundle is implemented automatically for every value that implements all of its
/// members, and so it must never be implemented manually.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor and is passed to every member,
/// so that tracers of homogeneous array programs implement this bundle for [`ArrayType`] and array IR values implement
/// it for [`ArrayIrType`](crate::ArrayIrType).
pub trait AttentionOperations<T = <Self as Capability>::Universe>:
    Capability
    + Parameter
    + FillLike<T>
    + IotaLike<T>
    + Add<T>
    + Sub<T>
    + Mul<T>
    + Div<T>
    + Exp<T>
    + Log<T>
    + And<T>
    + Reduce<T>
    + Dot<T>
    + Compare<Self, T>
    + Select<T>
    + Transpose<T>
    + Reshape<T>
    + Broadcast<T>
    + BroadcastLike<T>
    + ConvertElementType<T>
    + StopGradient<T>
    + Dropout<T>
    + Reshard<Self, T>
    + DimensionSize<T, Extent: Clone + Mul>
    + DynamicReshape<<Self as DimensionSize<T>>::Extent, T>
    + DynamicBroadcast<<Self as DimensionSize<T>>::Extent, T>
{
}

impl<
    T,
    V: Parameter
        + FillLike<T>
        + IotaLike<T>
        + Add<T>
        + Sub<T>
        + Mul<T>
        + Div<T>
        + Exp<T>
        + Log<T>
        + And<T>
        + Reduce<T>
        + Dot<T>
        + Compare<V, T>
        + Select<T>
        + Transpose<T>
        + Reshape<T>
        + Broadcast<T>
        + BroadcastLike<T>
        + ConvertElementType<T>
        + StopGradient<T>
        + Dropout<T>
        + Reshard<V, T>
        + DimensionSize<T, Extent: Clone + Mul>
        + DynamicReshape<<V as DimensionSize<T>>::Extent, T>
        + DynamicBroadcast<<V as DimensionSize<T>>::Extent, T>,
> AttentionOperations<T> for V
{
}

/// Rule configurations that staging attention requires of the [`CustomRuleDefinition`] of the operation family
/// `(V, O)`, collected so that the functions that stage it state one bound. They are the custom batching rules of the
/// forward and of the backward named definition, which fold batched calls (refer to [`AttentionOperations`]). Every
/// operation family whose operations support partial evaluation and differentiation satisfies them (e.g., the
/// [`ArrayOperation`](crate::ArrayOperation) family and backend families). This bundle is implemented automatically
/// for every type that satisfies its bounds, and so it must never be implemented manually.
pub trait AttentionRules<V: Value, O: Operation<Type = V::Type>>:
    CompositeBatching<
        BroadcastInvariantBatching<V, O, AttentionInputs<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>>,
        CustomRuleTracer<V, O>,
        AttentionInputs<CustomRuleTracer<V, O>>,
        Vec<CustomRuleTracer<V, O>>,
    > + CompositeBatching<
        BroadcastInvariantBatching<
            V,
            O,
            (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>),
            Vec<CustomRuleTracer<V, O>>,
        >,
        CustomRuleTracer<V, O>,
        (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>),
        Vec<CustomRuleTracer<V, O>>,
    >
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
}

impl<V: Value, O: Operation<Type = V::Type>, R> AttentionRules<V, O> for R
where
    R: CompositeBatching<
            BroadcastInvariantBatching<V, O, AttentionInputs<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>>,
            CustomRuleTracer<V, O>,
            AttentionInputs<CustomRuleTracer<V, O>>,
            Vec<CustomRuleTracer<V, O>>,
        > + CompositeBatching<
            BroadcastInvariantBatching<
                V,
                O,
                (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>),
                Vec<CustomRuleTracer<V, O>>,
            >,
            CustomRuleTracer<V, O>,
            (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>),
            Vec<CustomRuleTracer<V, O>>,
        >,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
}

/// Version of the attention definitions, which changes whenever their semantics or the encoding of their keys changes.
pub const DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION: u32 = 1;

/// Name of the forward named attention definition. Refer to the documentation of [`AttentionKey`]
/// for more information.
pub const DOT_PRODUCT_ATTENTION_COMPOSITE_NAME: &str = "ryft.dot_product_attention";

/// Name of the backward named attention definition. Refer to the documentation of [`AttentionKey`]
/// for more information.
pub const DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME: &str = "ryft.dot_product_attention_backward";

/// Name of the named composite that stages a differentiable attention operation. Its key has the attributes of the
/// forward definition's key except for `residual`. Refer to the documentation of [`AttentionKey::to_composite_key`]
/// for those attributes.
pub const DOT_PRODUCT_ATTENTION_DIFFERENTIABLE_COMPOSITE_NAME: &str = "ryft.dot_product_attention_differentiable";

/// Semantic identity of one named attention definition, which identifies the forward or the backward definition and
/// records its value-independent configuration and the structure of its inputs. It converts losslessly to and from
/// the [`CompositeKey`] of the definition's named composite, through [`AttentionKey::to_composite_key`] and
/// [`AttentionKey::from_composite_key`], which is how backends recognize attention calls (e.g., to
/// replace them with fused kernels).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AttentionKey {
    /// Forward attention, named [`DOT_PRODUCT_ATTENTION_COMPOSITE_NAME`], whose inputs are the [`AttentionInputs`] and
    /// whose outputs are the attended output, followed by the log-sum-exp residual when `return_residual` is set.
    Forward {
        /// Value-independent configuration of the definition.
        configuration: AttentionConfiguration,

        /// Structure of the [`AttentionInputs`] of the definition (i.e., which optional inputs are present).
        input_structure: AttentionInputs<Placeholder>,

        /// Whether the definition also returns the log-sum-exp residual of its softmax.
        return_residual: bool,
    },

    /// Analytical backward attention, named [`DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME`], whose inputs are the
    /// [`AttentionResiduals`] of a forward call followed by the cotangent of its output, and whose outputs are the
    /// cotangents of the query, key, and value, followed by that of a present bias whose cotangent space is not
    /// zero-dimensional.
    Backward {
        /// Value-independent configuration of the definition.
        configuration: AttentionConfiguration,

        /// Structure of the [`AttentionInputs`] of the definition (i.e., which optional inputs are present).
        input_structure: AttentionInputs<Placeholder>,
    },
}

impl AttentionKey {
    /// Returns the [`AttentionKey`] that `key` encodes, or [`None`] when `key` does not name an attention definition.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when `key` names an attention definition but has another version or attributes that
    /// differ from the encoding that [`Self::to_composite_key`] produces (e.g., missing, additional, reordered, or
    /// malformed attributes).
    pub fn from_composite_key(key: &CompositeKey) -> Result<Option<Self>, TypeError> {
        let name = key.name();
        if name != DOT_PRODUCT_ATTENTION_COMPOSITE_NAME && name != DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME {
            return Ok(None);
        }

        if key.version() != DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION {
            return Err(TypeError::invalid(format!(
                "named composite `{}` has attention definition version {}, but only version {} is supported",
                key,
                key.version(),
                DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION,
            )));
        }

        let attribute =
            |name: &str| {
                key.attributes().iter().find(|(attribute, _)| attribute == name).map(|(_, value)| value).ok_or_else(
                    || TypeError::invalid(format!("named composite `{key}` is not a valid attention definition")),
                )
            };

        let boolean = |name: &str| match attribute(name)? {
            CompositeAttribute::Boolean(value) => Ok(*value),
            _ => Err(TypeError::invalid(format!("named composite `{key}` is not a valid attention definition"))),
        };

        let optional_float = |name: &str| match attribute(name)? {
            CompositeAttribute::Optional(None) => Ok(None),
            CompositeAttribute::Optional(Some(value)) => match value.as_ref() {
                CompositeAttribute::Float(value) => Ok(Some(*value)),
                _ => Err(TypeError::invalid(format!("named composite `{key}` is not a valid attention definition"))),
            },
            _ => Err(TypeError::invalid(format!("named composite `{key}` is not a valid attention definition"))),
        };

        let implementation = match attribute("implementation")? {
            CompositeAttribute::Integer(0) => AttentionImplementation::Automatic,
            CompositeAttribute::Integer(1) => AttentionImplementation::Portable,
            CompositeAttribute::Integer(2) => AttentionImplementation::Fused,
            _ => {
                return Err(TypeError::invalid(format!("named composite `{key}` is not a valid attention definition")));
            }
        };

        let configuration = AttentionConfiguration::new()
            .with_scale(optional_float("scale")?)
            .with_causal(boolean("causal")?)
            .with_local_window(match attribute("local_window")? {
                CompositeAttribute::Optional(None) => None,
                CompositeAttribute::Optional(Some(value)) => {
                    if let CompositeAttribute::Sequence(values) = value.as_ref()
                        && let [CompositeAttribute::Integer(left), CompositeAttribute::Integer(right)] =
                            values.as_slice()
                        && let (Ok(left), Ok(right)) = (usize::try_from(*left), usize::try_from(*right))
                    {
                        Some((left, right))
                    } else {
                        return Err(TypeError::invalid(format!(
                            "named composite `{key}` is not a valid attention definition"
                        )));
                    }
                }
                _ => {
                    return Err(TypeError::invalid(format!(
                        "named composite `{key}` is not a valid attention definition"
                    )));
                }
            })
            .with_implementation(implementation)
            .with_dropout(optional_float("dropout_rate")?);

        let input_structure = AttentionInputs {
            bias: boolean("bias")?.then_some(Placeholder),
            mask: boolean("mask")?.then_some(Placeholder),
            query_sequence_lengths: boolean("query_sequence_lengths")?.then_some(Placeholder),
            key_value_sequence_lengths: boolean("key_value_sequence_lengths")?.then_some(Placeholder),
            dropout_state: boolean("dropout_state")?.then_some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };

        let decoded = if name == DOT_PRODUCT_ATTENTION_COMPOSITE_NAME {
            Self::Forward { configuration, input_structure, return_residual: boolean("residual")? }
        } else {
            Self::Backward { configuration, input_structure }
        };

        if &decoded.to_composite_key() != key {
            return Err(TypeError::invalid(format!("named composite `{key}` is not a valid attention definition")));
        }

        Ok(Some(decoded))
    }

    /// Returns the value-independent configuration of this [`AttentionKey`].
    #[inline]
    pub fn configuration(&self) -> AttentionConfiguration {
        match self {
            Self::Forward { configuration, .. } | Self::Backward { configuration, .. } => *configuration,
        }
    }

    /// Returns the structure of the [`AttentionInputs`] of this [`AttentionKey`].
    #[inline]
    pub fn input_structure(&self) -> AttentionInputs<Placeholder> {
        match self {
            Self::Forward { input_structure, .. } | Self::Backward { input_structure, .. } => *input_structure,
        }
    }

    /// Returns the [`CompositeKey`] of the named composite of this definition.
    /// Its name is [`DOT_PRODUCT_ATTENTION_COMPOSITE_NAME`] for a forward definition and
    /// [`DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME`] for a backward definition, and its version is
    /// [`DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION`]. Its attributes flatten the fields of this key, in
    /// the order of the following list:
    ///
    ///   - `scale`: Explicit score scale of the configuration, if any.
    ///   - `causal`: Whether the configuration enables causal masking.
    ///   - `local_window`: Inclusive `[left, right]` local window radii of the configuration, if any.
    ///   - `implementation`: [`AttentionImplementation`] that the configuration requests, which is `0` for automatic,
    ///     `1` for portable, and `2` for fused.
    ///   - `dropout_rate`: Dropout rate of the configuration, if any.
    ///   - `residual`: Whether the definition also returns the log-sum-exp residual of its softmax. Only the keys of
    ///     forward definitions have this attribute.
    ///   - `bias`, `mask`, `query_sequence_lengths`, `key_value_sequence_lengths`, and `dropout_state`: Whether the
    ///     input structure contains the corresponding optional input.
    #[inline]
    pub fn to_composite_key(&self) -> CompositeKey {
        match *self {
            Self::Forward { configuration, input_structure, return_residual } => attention_composite_key(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                configuration,
                input_structure,
                Some(return_residual),
            ),
            Self::Backward { configuration, input_structure } => attention_composite_key(
                DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
                configuration,
                input_structure,
                None,
            ),
        }
    }

    /// Returns the batching roles of the canonical inputs of this [`AttentionKey`]. Dropout states never reach batching
    /// rules, because definitions with dropout are batched structurally.
    fn batch_input_roles(&self) -> Vec<AttentionBatchInput> {
        let input_structure = self.input_structure();
        let mut roles = vec![AttentionBatchInput::Tensor; 3];
        roles.extend(input_structure.bias.is_some().then_some(AttentionBatchInput::Score));
        roles.extend(input_structure.mask.is_some().then_some(AttentionBatchInput::Score));
        roles.extend(input_structure.query_sequence_lengths.is_some().then_some(AttentionBatchInput::Length));
        roles.extend(input_structure.key_value_sequence_lengths.is_some().then_some(AttentionBatchInput::Length));
        if matches!(self, Self::Backward { .. }) {
            roles.extend([AttentionBatchInput::Tensor, AttentionBatchInput::Residual, AttentionBatchInput::Tensor]);
        }
        roles
    }

    /// Returns the batching roles of the `output_count` outputs of this definition. A backward definition returns
    /// a bias cotangent, whose bias is always the fourth canonical input, only when the bias is present and its
    /// cotangent space is not zero-dimensional.
    fn batch_output_roles(&self, output_count: usize) -> Vec<AttentionBatchOutput> {
        match self {
            Self::Forward { .. } => [AttentionBatchOutput::Tensor(0), AttentionBatchOutput::Residual]
                .into_iter()
                .take(output_count)
                .collect(),
            Self::Backward { input_structure, .. } => {
                let mut roles = vec![
                    AttentionBatchOutput::Cotangent(0),
                    AttentionBatchOutput::Cotangent(1),
                    AttentionBatchOutput::Cotangent(2),
                ];
                if input_structure.bias.is_some() && output_count == 4 {
                    roles.push(AttentionBatchOutput::Score(3));
                }
                roles
            }
        }
    }
}

/// Backend implementation requested for an attention operation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum AttentionImplementation {
    /// Uses a fused implementation when the active backend supports the complete configuration and otherwise uses
    /// the portable semantic composition. [`differentiable_dot_product_attention`] selects the portable composition
    /// for this choice so that training uses its stabilized backward rule. A fused implementation evaluates attention
    /// in the query data type, which bounds what it can represent. It rounds a bias of a wider data type to that
    /// data type (saturating finite biases at its largest finite magnitude), and it may apply a Boolean mask as a
    /// large finite penalty of that data type rather than by replacing the masked scores, so a masked score that
    /// exceeds the visible scores of its row by more than that penalty is not hidden. [`Self::Portable`] evaluates
    /// the scores in at least `f32` and hides every masked score.
    #[default]
    Automatic,

    /// Always uses the portable semantic composition.
    Portable,

    /// Requires a fused backend implementation and reports an error when the configuration is not supported.
    Fused,
}

impl Display for AttentionImplementation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Automatic => formatter.write_str("automatic"),
            Self::Portable => formatter.write_str("portable"),
            Self::Fused => formatter.write_str("fused"),
        }
    }
}

/// Value-independent semantics of a scaled dot-product attention operation. Equality and hashing compare the
/// floating-point scale and dropout rate bitwise, so that configurations are faithful keys of the attributes
/// that backends receive (e.g., `-0.0` and `+0.0` scales are distinct).
#[derive(Copy, Clone, Debug, Default)]
pub struct AttentionConfiguration {
    /// Explicit score scale, or [`None`] to use `1 / sqrt(head_dimension)` (or one for an empty head dimension).
    scale: Option<f64>,

    /// Whether query position `i` may attend only to key/value positions `j <= i`.
    causal: bool,

    /// Optional inclusive `(left, right)` local window radii.
    local_window: Option<(usize, usize)>,

    /// Requested backend implementation.
    implementation: AttentionImplementation,

    /// Optional probability with which dropout drops each attention probability.
    dropout: Option<f64>,
}

impl AttentionConfiguration {
    /// Creates the default configuration, which has an inferred scale, non-causal global attention, automatic
    /// implementation selection, and no dropout.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets an explicit score scale. Passing [`None`] restores `1 / sqrt(head_dimension)`, or one for an empty head
    /// dimension, whose scores are zero for every scale.
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

    /// Sets the inclusive `(left, right)` local window radii, with which query position `i` may attend only to
    /// key/value positions `j` with `i - left <= j <= i + right`. Causal masking already hides every position to
    /// the right, so the right radius has no effect with it and [`Self::local_window`] reports it as zero.
    #[inline]
    pub fn with_local_window<W: Into<Option<(usize, usize)>>>(mut self, local_window: W) -> Self {
        self.local_window = local_window.into();
        self
    }

    /// Sets a symmetric local window radius.
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

    /// Sets the probability with which dropout drops each attention probability, which must lie in the open interval
    /// `(0, 1)`. Passing [`None`] disables dropout. Dropout applies a [`DropoutOperation`](crate::DropoutOperation) to
    /// the normalized attention probabilities, whose random state is the optional [`AttentionInputs::dropout_state`]
    /// input. Refer to that input for how the state selects the dropout mask.
    #[inline]
    pub fn with_dropout<D: Into<Option<f64>>>(mut self, dropout: D) -> Self {
        self.dropout = dropout.into();
        self
    }

    /// Returns the explicit score scale, if one was configured.
    #[inline]
    pub fn scale(&self) -> Option<f64> {
        self.scale
    }

    /// Returns the score scale that this configuration selects for inputs with the provided head dimension, which
    /// is its explicit scale or, without one, `1 / sqrt(head_dimension)`. An empty head dimension has the scale
    /// one instead, because its scores are empty contractions, which are zero for every scale, and the infinite
    /// `1 / sqrt(0)` would turn them into NaNs. The portable bodies and backend implementations of the attention
    /// definitions (e.g., the fused kernels of the XLA backend) use this function, so that they agree on the scale.
    #[inline]
    pub fn scale_for(&self, head_dimension: usize) -> f64 {
        self.scale.unwrap_or(1.0 / (head_dimension.max(1) as f64).sqrt())
    }

    /// Returns whether causal masking is enabled.
    #[inline]
    pub fn causal(&self) -> bool {
        self.causal
    }

    /// Returns the optional inclusive `(left, right)` local window radii. The right radius is zero with causal
    /// masking, which hides every position to the right, so that configurations with the same semantics are equal
    /// and have the same keys.
    #[inline]
    pub fn local_window(&self) -> Option<(usize, usize)> {
        match self.local_window {
            Some((left, _)) if self.causal => Some((left, 0)),
            local_window => local_window,
        }
    }

    /// Returns the requested backend implementation.
    #[inline]
    pub fn implementation(&self) -> AttentionImplementation {
        self.implementation
    }

    /// Returns the optional probability with which dropout drops each attention probability.
    #[inline]
    pub fn dropout(&self) -> Option<f64> {
        self.dropout
    }
}

impl PartialEq for AttentionConfiguration {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.scale.map(f64::to_bits) == other.scale.map(f64::to_bits)
            && self.causal == other.causal
            && self.local_window() == other.local_window()
            && self.implementation == other.implementation
            && self.dropout.map(f64::to_bits) == other.dropout.map(f64::to_bits)
    }
}

impl Eq for AttentionConfiguration {}

impl Hash for AttentionConfiguration {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.scale.map(f64::to_bits).hash(state);
        self.causal.hash(state);
        self.local_window().hash(state);
        self.implementation.hash(state);
        self.dropout.map(f64::to_bits).hash(state);
    }
}

/// Query, key, value, and optional inputs supplied to a scaled dot-product attention operation. Their canonical order
/// is the order of the fields of this struct, which is the order of [`Parameterized::parameters`], and their structure
/// (i.e., [`Parameterized::parameter_structure`]) records which optional inputs are present.
///
/// # Shapes
///
/// The shapes of the inputs are written with one letter per axis, so that `BTNH` denotes a rank-4 array of shape
/// `[B, T, N, H]`. The letters stand for the following extents.
///
///   | Letter | Extent                                                      |
///   |--------|-------------------------------------------------------------|
///   | `B`    | Number of batch items.                                      |
///   | `T`    | Length of the query sequence (i.e., the target length).     |
///   | `S`    | Length of the key/value sequence (i.e., the source length). |
///   | `N`    | Number of query heads.                                      |
///   | `K`    | Number of key/value heads, which must divide `N`.           |
///   | `H`    | Size of each head (i.e., the head dimension).               |
///
/// The batch axis is optional. The query, key, and value have rank 3 for a single batch item and rank 4 for a batch.
/// The attention scores, which the bias and the mask apply to, have one entry per batch item, query head, query
/// position, and key/value position, so their shape is `[B, N, T, S]`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameterized)]
pub struct AttentionInputs<P: Parameter> {
    /// Query array of shape `[T, N, H]` or `[B, T, N, H]` (i.e., one vector of size `H` per query position and query
    /// head).
    pub query: P,

    /// Key array of shape `[S, K, H]` or `[B, S, K, H]` (i.e., one vector of size `H` per key/value position and
    /// key/value head).
    pub key: P,

    /// Value array with the same logical shape as `key`.
    pub value: P,

    /// Optional additive bias of the attention scores, of rank at most 4, which is broadcast to the score shape
    /// `[B, N, T, S]` by aligning its axes with the trailing score axes. Its data type is real numeric (i.e., an
    /// integer or floating-point type) or Boolean, and it is converted to the data type of the scores. Complex biases
    /// are rejected, because that conversion would silently discard their imaginary parts.
    pub bias: Option<P>,

    /// Optional Boolean visibility mask of the attention scores, of rank at most 4, which is broadcast to the score
    /// shape `[B, N, T, S]` like the bias. A key/value position is visible to a query position where it is `true`.
    pub mask: Option<P>,

    /// Optional rank-1 array with the query sequence length of every batch item (i.e., how many of the `T` query
    /// positions of that item are valid).
    pub query_sequence_lengths: Option<P>,

    /// Optional rank-1 array with the key/value sequence length of every batch item (i.e., how many of the `S`
    /// key/value positions of that item are valid).
    pub key_value_sequence_lengths: Option<P>,

    /// Optional random generator state of the dropout mask (refer to [`AttentionConfiguration::with_dropout`]), with
    /// the [`RandomAlgorithm::state_type`] of its algorithm. With a state, the mask is a deterministic function of
    /// that state and backends never replace the dropout with a fused kernel that draws its own mask. Without a state,
    /// backends may fuse the dropout (e.g., into the cuDNN attention kernel), while Ryft's own implementations draw
    /// the mask from a fixed state (refer to [`DropoutOperation`](crate::DropoutOperation)). It is only accepted
    /// together with a dropout rate.
    pub dropout_state: Option<P>,
}

impl<P: Parameter> AttentionInputs<P> {
    /// Creates attention inputs with no bias, mask, sequence lengths, or dropout state.
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
            dropout_state: None,
        }
    }

    /// Returns these inputs with their manual variation aligned through
    /// [`ManualVariationAlignment::align_manual_variation`], which precedes every call of the functions that stage
    /// attention. Alignment is the identity when every input already varies over the same manual axes (e.g., outside
    /// manual regions), and so those inputs are returned unchanged without staging the alignment.
    fn align_manual_variation(self) -> Result<Self, ProgramError>
    where
        P: Value<Type: ArrayMemberType> + ManualVariationAlignment<P::Type>,
    {
        let mut varying_manual_axes = None;
        let mut aligned = true;
        for input in self.parameters() {
            let r#type = input.r#type();
            let array_type = r#type.as_array_type()?;
            let axes = array_type.sharding().map(|sharding| sharding.varying_manual_axes().clone()).unwrap_or_default();
            match &varying_manual_axes {
                None => varying_manual_axes = Some(axes),
                Some(expected) => aligned &= *expected == axes,
            }
        }
        if aligned {
            return Ok(self);
        }
        let input_structure = self.parameter_structure();
        let inputs = ManualVariationAlignment::align_manual_variation(&self.into_parameters().collect::<Vec<_>>())?;
        Ok(Self::from_parameters(input_structure, inputs)?)
    }
}

/// Residuals of one forward attention call that its backward pass consumes, consisting of the forward
/// [`AttentionInputs`], the attended forward output, and its log-sum-exp residual. The reverse mode rules of
/// differentiable attention retain them between the forward and the backward call, and the backward attention
/// definition takes them paired with the incoming cotangent of the output (i.e., its inputs are
/// `(AttentionResiduals<P>, P)`). The canonical order of those inputs is that of the forward inputs followed by the
/// output, the residual, and the output cotangent, which is the order of [`Parameterized::parameters`]. The type is
/// public because it appears in the bounds of [`AttentionRules`] and of [`differentiable_dot_product_attention`], and
/// its contents are private to the attention rules.
#[derive(Clone, Debug, PartialEq, Parameterized)]
pub struct AttentionResiduals<P: Parameter> {
    /// Forward inputs, including the present optional inputs.
    inputs: AttentionInputs<P>,

    /// Attended forward output.
    output: P,

    /// Log-sum-exp residual of the forward output, which the backward definition consumes.
    residual: P,
}

/// Scaled dot-product attention inference function, built by [`dot_product_attention`]. Each call stages one call of
/// the named forward attention definition (refer to [`AttentionKey`]) for the structure of its inputs, which backends
/// may replace with their own implementation of the definition (e.g., a fused kernel). The function owns its
/// definitions, so it should be created once and called many times, which traces each definition's body once
/// per input types and shares it, together with its lowering, across those calls.
///
/// # Semantics
///
/// For each batch item and query head, attention computes:
///
/// ```text
/// softmax(scale * Q * transpose(K) + bias, visibility) * V.
/// ```
///
/// The query has `TNH` or `BTNH` form, and the key and value have `SKH` or `BSKH` form, where `T` and `S` are the query
/// and key/value sequence lengths, `N` is the number of query heads, `K` is the number of key/value heads, and `H` is
/// the head dimension. `K` must divide `N`, where `K == N` is multi-head attention, `K == 1` is multi-query attention,
/// and the remaining cases are grouped-query attention. The attended output has the data type, shape, and memory space
/// of the query, without its layout, and the sharding that the operations of attention infer. Its batch and head
/// placements come from the query, key, and value, its sequence placement from the query, and its head dimension
/// placement from the value, so the output has the sharding of the query whenever the key and the value are placed
/// like it. The bias, the mask, and the sequence lengths take part like the inputs of an elementwise operation on the
/// scores, so their batch, head, and query sequence placements reach the output as well. Dimensions that are contracted
/// and sharded on both sides (e.g., the head dimensions of the query and of the key) have no unambiguous placement and
/// are rejected, as are placements that cannot be combined (e.g., one mesh axis that shards two dimensions of the
/// scores). Reverse mode returns each gradient with the sharding of its input, however the inputs are placed.
/// Reduced and unreduced inputs are unsupported, because softmax and the log-sum-exp residual preserve neither
/// partial sums nor replicas that are filled with zeros. Calls align the manual variation of their inputs before
/// staging the call.
///
/// [`AttentionInputs`] defines the complete input boundary, consisting of the query, key, and value, followed by an
/// optional broadcastable additive bias, Boolean visibility mask, query and key/value sequence lengths of every batch
/// item, and dropout state. The scores (i.e., the contraction of the query with the key, its scaling, the bias
/// addition, and the masking) use the query data type promoted to at least `f32`, so that data types that are narrower
/// than `f32` neither round nor overflow them. The masked logits are then converted to `f32` for the numerically
/// stable softmax, and the attention weights return to the query data type before they are contracted with the value.
/// Visibility is the conjunction of the explicit mask, causal ordering, the inclusive `(left, right)` local window,
/// and both sequence length predicates. Invisible scores use a large negative value that is finite in `f32`, so a row
/// without any visible position attends uniformly to every key/value position and receives no score cotangents, and
/// query rows beyond their lengths are exact zeros. Without a configured scale, attention uses `1 / sqrt(H)`, or one
/// when `H` is zero, because the scores of an empty head dimension are zero for every scale. Every batch item uses the
/// same rectangular storage shape, and sequence lengths control visibility within that storage.
///
/// [`DotProductAttention::call_with_residual`] also returns the log-sum-exp residual (i.e., the natural
/// logarithm of the sum of the exponentiated logits), which has the `TN` or `BTN` form and the `f32` data type
/// of the softmax for every query data type, so that it is never rounded or overflowed by a narrower one. With
/// [`AttentionConfiguration::with_dropout`], the attention weights are dropped and rescaled by [`Dropout`] with the
/// dropout state of the inputs, which determines the mask (refer to [`AttentionInputs::dropout_state`]). Dropout
/// supports sharding only along the broadcast axes of its input, so attention with dropout over sharded inputs runs
/// inside `shard_map`.
///
/// [`AttentionImplementation::Portable`] always evaluates the portable body, which computes attention with ordinary
/// array operations. [`AttentionImplementation::Fused`] requires a backend implementation that supports the complete
/// call (e.g., cuDNN's fused kernels on the XLA backend). [`AttentionImplementation::Automatic`] lets backends use
/// such an implementation when it is eligible and evaluates the portable body otherwise.
///
/// # Transforms
///
/// Batching a call folds its mapped axis into the logical batch axis and calls the same definition at the folded
/// boundary, which keeps backend implementations for batched calls, while calls with dropout are batched structurally
/// and calls with dropout and the fused implementation cannot be batched, unless all of their inputs are replicated.
/// The named definitions have no derivative rules, so differentiating a call fails. Use
/// [`differentiable_dot_product_attention`] for the differentiable path.
pub struct DotProductAttention<C: Domain>
where
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    /// [`AttentionDefinitions`] staged by this [`DotProductAttention`] operation.
    definitions: AttentionDefinitions<C::Constant, C::Operation>,
}

impl<C: Domain> DotProductAttention<C>
where
    C::Type: ArrayMemberType + From<ArrayType> + BatchableType,
    C::Constant: 'static,
    C::Operation: 'static,
    DomainTracer<C>: AttentionOperations,
    CustomRuleDefinition<C::Constant, C::Operation>: AttentionRules<C::Constant, C::Operation>,
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>
        + CompositeCallContext<Type = C::Type, Value = DomainTracer<C>, Constant = C::Constant, Operation = C::Operation>,
{
    /// Returns the [`AttentionConfiguration`] of this [`DotProductAttention`].
    #[inline]
    pub fn configuration(&self) -> AttentionConfiguration {
        self.definitions.configuration()
    }

    /// Aligns the manual variation of `inputs` and stages one [`DotProductAttention`] call, returning
    /// the attended output.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `inputs` are invalid for the configuration (e.g., because their shapes are
    /// incompatible) or when staging the call fails.
    pub fn call<Caller, V>(&self, inputs: AttentionInputs<V>) -> Result<V, ProgramError>
    where
        Caller: CompositeCallContext<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>,
        V: Value<Type = C::Type, Domain = Caller> + ManualVariationAlignment<C::Type>,
    {
        let inputs = inputs.align_manual_variation()?;
        let mut outputs = self.definitions.forward(inputs, false)?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    /// Aligns the manual variation of `inputs` and stages one attention call, returning the attended output together
    /// with the log-sum-exp residual of its softmax.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `inputs` are invalid for the configuration (e.g., because their shapes are
    /// incompatible) or when staging the call fails.
    pub fn call_with_residual<Caller, V>(&self, inputs: AttentionInputs<V>) -> Result<(V, V), ProgramError>
    where
        Caller: CompositeCallContext<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>,
        V: Value<Type = C::Type, Domain = Caller> + ManualVariationAlignment<C::Type>,
    {
        let inputs = inputs.align_manual_variation()?;
        let mut outputs = self.definitions.forward(inputs, true)?;
        check_count!("output", outputs, 2, ProgramError);
        let residual = outputs.remove(1);
        Ok((outputs.remove(0), residual))
    }
}

/// Creates the scaled dot-product attention inference function with the provided configuration for the values of the
/// domain `C`, whose calls stage the named forward attention definition. Refer to [`DotProductAttention`] for more
/// information.
///
/// The definition's portable body computes attention with ordinary array operations, while backends may replace it
/// (e.g., the XLA backend replaces it with cuDNN's fused attention kernels when the configuration and the input types
/// are eligible). Array IR domains support dynamic batch and sequence extents (e.g., in bounded dynamic programs).
/// Homogeneous array domains support them as well, except in batching rules that need a dynamic extent.
///
/// # Parameters
///
///   - `configuration`: Value-independent attention semantics and implementation selection.
#[inline]
pub fn dot_product_attention<C: Domain>(configuration: AttentionConfiguration) -> DotProductAttention<C>
where
    C::Type: ArrayMemberType + From<ArrayType> + BatchableType,
    C::Constant: 'static,
    C::Operation: 'static,
    DomainTracer<C>: AttentionOperations,
    CustomRuleDefinition<C::Constant, C::Operation>: AttentionRules<C::Constant, C::Operation>,
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>
        + CompositeCallContext<Type = C::Type, Value = DomainTracer<C>, Constant = C::Constant, Operation = C::Operation>,
{
    DotProductAttention { definitions: AttentionDefinitions::new(configuration) }
}

/// Differentiable scaled dot-product attention function, built by [`differentiable_dot_product_attention`]. Each call
/// stages one call of a named composite with reverse mode rules, whose primal calls the forward attention definition
/// and whose rules call the forward definition with its residual and the backward definition. The manual variation
/// transitions of the inputs precede the composite boundary, which lets their transposes accumulate the cotangent
/// contributions of invariant inputs exactly once.
///
/// The function owns its named composite together with the definitions that its primal and its rules call, so it should
/// be created once and called many times. Its primal and its definitions' bodies are then traced once per input types,
/// and its rules once per specialization, for as long as the function lives.
#[allow(clippy::type_complexity)]
pub struct DifferentiableDotProductAttention<C: Domain>
where
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    /// [`AttentionConfiguration`] of this [`DifferentiableDotProductAttention`].
    configuration: AttentionConfiguration,

    /// Named composite that stages the calls of this function, whatever their input structure. Its primal and its
    /// rules own the named attention definitions that they call.
    function: Composite<
        C::Constant,
        C::Operation,
        AttentionInputs<DomainTracer<C>>,
        DomainTracer<C>,
        DefaultJvp,
        SymbolicZeroVjp<
            C::Constant,
            C::Operation,
            AttentionInputs<DomainTracer<C>>,
            DomainTracer<C>,
            AttentionResiduals<DomainTracer<C>>,
        >,
    >,
}

impl<C: Domain> DifferentiableDotProductAttention<C>
where
    C::Type: ArrayMemberType + From<ArrayType> + BatchableType + DifferentiableType,
    C::Constant: 'static,
    C::Operation: 'static,
    DomainTracer<C>: AttentionOperations,
    CustomRuleDefinition<C::Constant, C::Operation>: AttentionRules<C::Constant, C::Operation>
        + CompositeVjp<
            SymbolicZeroVjp<
                C::Constant,
                C::Operation,
                AttentionInputs<DomainTracer<C>>,
                DomainTracer<C>,
                AttentionResiduals<DomainTracer<C>>,
            >,
            DomainTracer<C>,
            AttentionInputs<DomainTracer<C>>,
            DomainTracer<C>,
        >,
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>
        + CompositeCallContext<Type = C::Type, Value = DomainTracer<C>, Constant = C::Constant, Operation = C::Operation>,
{
    /// Returns the configuration of this [`DifferentiableDotProductAttention`].
    #[inline]
    pub fn configuration(&self) -> AttentionConfiguration {
        self.configuration
    }

    /// Aligns the manual variation of `inputs` and stages one [`DifferentiableDotProductAttention`] call,
    /// returning the attended output.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `inputs` are invalid for the configuration (e.g., because their shapes are
    /// incompatible) or when staging the call fails.
    #[inline]
    pub fn call<Caller, V>(&self, inputs: AttentionInputs<V>) -> Result<V, ProgramError>
    where
        Caller: CompositeCallContext<Type = C::Type, Value = V, Constant = C::Constant, Operation = C::Operation>,
        V: Value<Type = C::Type, Domain = Caller> + ManualVariationAlignment<C::Type>,
    {
        self.function.call(inputs.align_manual_variation()?)
    }
}

/// Creates the differentiable scaled dot-product attention function with the provided configuration for the values
/// of the domain `C`, whose calls accept every structure of [`AttentionInputs`].
///
/// The primal path returns only the attended output. Reverse mode differentiation runs the forward definition with
/// its log-sum-exp residual and then applies the backward definition, whose portable body recomputes the normalization
/// with a stabilized composition. [`AttentionImplementation::Automatic`] selects [`AttentionImplementation::Portable`]
/// for this training entry point, which preserves the small gradients of dominant logits. Explicitly requesting
/// [`AttentionImplementation::Fused`] opts into the backend's native forward and backward kernels where the backend
/// provides them (e.g., cuDNN's fused attention on the XLA backend), including their numerical behavior. The portable
/// gradient guarantees do not apply to those kernels. A configured dropout applies the same mask in the forward and
/// backward definitions, which a present dropout state (refer to [`AttentionInputs::dropout_state`]) determines.
/// Inference through [`dot_product_attention`] retains automatic fusion.
///
/// Optional Boolean masks, integer sequence lengths, and dropout states remain ordinary structural leaves of `inputs`.
/// The backward rule returns structural zeros for their cotangents (refer to [`Composite::with_symbolic_zero_vjp`]),
/// which Ryft's zero space machinery materializes only where a consumer needs a value, while a present floating-point
/// bias receives the cotangent returned by the backward definition. A structurally zero output cotangent (e.g., when
/// the output is unused) skips the backward definition altogether.
///
/// The returned function owns its named composite and its rules (refer to [`DifferentiableDotProductAttention`]),
/// so it should be created once and called many times. To embed the resulting programs in a heterogeneous value
/// universe, use [`Program::into_unprojected`](crate::programs::Program::into_unprojected), which converts the
/// staged calls together with their rules.
///
/// # Parameters
///
///   - `configuration`: Value-independent attention semantics and implementation selection.
pub fn differentiable_dot_product_attention<C: Domain>(
    configuration: AttentionConfiguration,
) -> DifferentiableDotProductAttention<C>
where
    C::Type: ArrayMemberType + From<ArrayType> + BatchableType + DifferentiableType,
    C::Constant: 'static,
    C::Operation: 'static,
    DomainTracer<C>: AttentionOperations,
    CustomRuleDefinition<C::Constant, C::Operation>: AttentionRules<C::Constant, C::Operation>
        + CompositeVjp<
            SymbolicZeroVjp<
                C::Constant,
                C::Operation,
                AttentionInputs<DomainTracer<C>>,
                DomainTracer<C>,
                AttentionResiduals<DomainTracer<C>>,
            >,
            DomainTracer<C>,
            AttentionInputs<DomainTracer<C>>,
            DomainTracer<C>,
        >,
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>
        + CompositeCallContext<Type = C::Type, Value = DomainTracer<C>, Constant = C::Constant, Operation = C::Operation>,
{
    let configuration = match configuration.implementation() {
        AttentionImplementation::Automatic => configuration.with_implementation(AttentionImplementation::Portable),
        AttentionImplementation::Portable | AttentionImplementation::Fused => configuration,
    };
    let definitions = Arc::new(AttentionDefinitions::new(configuration));
    let function = composite({
        // The primal stages the forward definition without its residual.
        let definitions = definitions.clone();
        move |inputs: AttentionInputs<DomainTracer<C>>| -> Result<DomainTracer<C>, ProgramError> {
            let mut outputs = definitions.forward(inputs, false)?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.remove(0))
        }
    })
    .with_symbolic_zero_vjp(
        {
            // The forward rule stages the forward definition with its residual, and it retains the inputs,
            // the output, and the residual for the backward rule.
            let definitions = definitions.clone();
            move |inputs: AttentionInputs<DomainTracer<C>>| -> Result<_, ProgramError> {
                let mut outputs = definitions.forward(inputs.clone(), true)?;
                check_count!("output", outputs, 2, ProgramError);
                let residual = outputs.remove(1);
                let output = outputs.remove(0);
                Ok((output.clone(), AttentionResiduals { inputs, output, residual }))
            }
        },
        move |residuals: AttentionResiduals<DomainTracer<C>>,
              output_cotangent: MaybeZero<DomainTracer<C>>|
              -> Result<_, ProgramError> {
            // A structurally zero output cotangent (e.g., of an unused output) makes every input cotangent
            // a structural zero, without staging the backward definition.
            let MaybeZero::Value(output_cotangent) = output_cotangent else {
                return residuals.inputs.try_map_parameters(|input| Ok(MaybeZero::Zero(input.r#type().cotangent()?)));
            };

            // The backward definition returns the cotangents of the query, key, and value, and of a bias
            // whose cotangent space is not zero-dimensional, so the cotangents of the other inputs (i.e.,
            // of the masks, sequence lengths, and dropout states, and of a bias with a zero-dimensional
            // cotangent space) are structural zeros, which contribute nothing to their inputs' cotangents.
            let inputs = &residuals.inputs;
            let mask = inputs.mask.as_ref().map(|input| input.r#type().cotangent().map(MaybeZero::Zero)).transpose()?;
            let query_sequence_lengths = inputs
                .query_sequence_lengths
                .as_ref()
                .map(|input| input.r#type().cotangent().map(MaybeZero::Zero))
                .transpose()?;
            let key_value_sequence_lengths = inputs
                .key_value_sequence_lengths
                .as_ref()
                .map(|input| input.r#type().cotangent().map(MaybeZero::Zero))
                .transpose()?;
            let dropout_state = inputs
                .dropout_state
                .as_ref()
                .map(|input| input.r#type().cotangent().map(MaybeZero::Zero))
                .transpose()?;
            let bias_cotangent_type = inputs.bias.as_ref().map(|bias| bias.r#type().cotangent()).transpose()?;
            let mut outputs = definitions.backward((residuals, output_cotangent))?;
            let bias = match bias_cotangent_type {
                Some(_) if outputs.len() == 4 => Some(MaybeZero::Value(outputs.remove(3))),
                Some(r#type) => Some(MaybeZero::Zero(r#type)),
                None => None,
            };
            check_count!("output", outputs, 3, ProgramError);
            let value = MaybeZero::Value(outputs.remove(2));
            let key = MaybeZero::Value(outputs.remove(1));
            let query = MaybeZero::Value(outputs.remove(0));
            Ok(AttentionInputs {
                query,
                key,
                value,
                bias,
                mask,
                query_sequence_lengths,
                key_value_sequence_lengths,
                dropout_state,
            })
        },
    )
    .with_name(DOT_PRODUCT_ATTENTION_DIFFERENTIABLE_COMPOSITE_NAME)
    .with_named_composite_per_structure(
        move |input_structure: &AttentionInputs<Placeholder>| {
            attention_composite_key(
                DOT_PRODUCT_ATTENTION_DIFFERENTIABLE_COMPOSITE_NAME,
                configuration,
                *input_structure,
                None,
            )
        },
        move |input_types: AttentionInputs<C::Type>| -> Result<C::Type, ProgramError> {
            // The composite returns only the attended output, whose type the forward boundary determines.
            let input_structure = input_types.parameter_structure();
            let input_types = input_types
                .parameters()
                .map(|r#type| r#type.as_array_type().cloned())
                .collect::<Result<Vec<_>, _>>()?;
            let mut output_types = attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                configuration,
                input_structure,
                false,
                input_types.as_slice(),
            )?;
            check_count!("output", output_types, 1, ProgramError);
            Ok(output_types.remove(0).into())
        },
    );
    DifferentiableDotProductAttention { configuration, function }
}

/// Validated dimensions that the input contracts of the attention definitions share.
struct AttentionDimensions {
    /// Shared batch dimension of the query, key, and value.
    batch: Dimension,

    /// Query sequence length.
    query_sequence: Dimension,

    /// Number of query heads.
    query_heads: usize,

    /// Number of key/value heads, which divides `query_heads` and is strictly smaller for grouped-query attention.
    key_value_heads: usize,

    /// Head (i.e., feature) dimension of the query, key, and value.
    head_dimension: usize,

    /// Shared floating-point data type of the query, key, and value.
    data_type: DataType,
}

impl AttentionDimensions {
    /// Returns the data type of the attention scores, which is the query data type promoted to at least `f32` (i.e.,
    /// `f64` remains `f64` and every other data type becomes `f32`). The bodies evaluate everything other than the
    /// softmax operation in this data type, so that data types that are narrower than `f32` neither round nor
    /// overflow their scores and `f64` keeps its range.
    #[inline]
    fn score_data_type(&self) -> DataType {
        if self.data_type == DataType::F64 { DataType::F64 } else { DataType::F32 }
    }
}

/// Named attention definitions of one operation family `(V, O)` and one [`AttentionConfiguration`] (i.e., the forward
/// definition without and with its residual, and the backward definition). Each one is a named composite that serves
/// every input structure, under the [`AttentionKey`] of that structure. The functions that stage attention own these
/// definitions, so their traced bodies and their rule specializations live exactly as long as those functions.
#[allow(clippy::type_complexity)]
struct AttentionDefinitions<V: Value, O: Operation<Type = V::Type>>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    /// Configuration of every definition.
    configuration: AttentionConfiguration,

    /// Forward definition that returns only the attended output.
    forward: AttentionFunction<
        BatchedComposite<V, O, AttentionInputs<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>>,
        Composite<V, O, AttentionInputs<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>>,
    >,

    /// Forward definition that also returns the log-sum-exp residual of its softmax.
    forward_with_residual: AttentionFunction<
        BatchedComposite<V, O, AttentionInputs<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>>,
        Composite<V, O, AttentionInputs<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>>,
    >,

    /// Backward definition.
    backward: AttentionFunction<
        BatchedComposite<
            V,
            O,
            (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>),
            Vec<CustomRuleTracer<V, O>>,
        >,
        Composite<
            V,
            O,
            (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>),
            Vec<CustomRuleTracer<V, O>>,
        >,
    >,
}

impl<V: Value, O: Operation<Type = V::Type>> AttentionDefinitions<V, O>
where
    V: 'static,
    V::Type: ArrayMemberType + From<ArrayType> + BatchableType,
    O: 'static,
    CustomRuleTracer<V, O>: AttentionOperations,
    CustomRuleDefinition<V, O>: AttentionRules<V, O>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>
        + CompositeCallContext<Type = V::Type, Value = CustomRuleTracer<V, O>, Constant = V, Operation = O>,
{
    /// Creates an [`AttentionDefinitions`] instance based on the provided [`AttentionConfiguration`].
    #[inline]
    fn new(configuration: AttentionConfiguration) -> Self {
        Self {
            configuration,
            forward: forward_definition(configuration, false),
            forward_with_residual: forward_definition(configuration, true),
            backward: backward_definition(configuration),
        }
    }

    /// Returns the configuration of the definitions.
    #[inline]
    fn configuration(&self) -> AttentionConfiguration {
        self.configuration
    }

    /// Stages one call of the forward definition with the provided residual request at `inputs`, returning the
    /// attended output followed by its log-sum-exp residual when `return_residual` is set.
    #[inline]
    fn forward<C, InputValue>(
        &self,
        inputs: AttentionInputs<InputValue>,
        return_residual: bool,
    ) -> Result<Vec<InputValue>, ProgramError>
    where
        C: CompositeCallContext<Type = V::Type, Value = InputValue, Constant = V, Operation = O>,
        InputValue: Value<Type = V::Type, Domain = C>,
    {
        match if return_residual { &self.forward_with_residual } else { &self.forward } {
            AttentionFunction::Folding(function) => function.call(inputs),
            AttentionFunction::Structural(function) => function.call(inputs),
        }
    }

    /// Stages one call of the backward definition at `inputs`, returning the cotangents of the query, key, and value,
    /// followed by that of a present bias whose cotangent space is not zero-dimensional.
    #[inline]
    fn backward<C, InputValue>(
        &self,
        inputs: (AttentionResiduals<InputValue>, InputValue),
    ) -> Result<Vec<InputValue>, ProgramError>
    where
        C: CompositeCallContext<Type = V::Type, Value = InputValue, Constant = V, Operation = O>,
        InputValue: Value<Type = V::Type, Domain = C>,
    {
        match &self.backward {
            AttentionFunction::Folding(function) => function.call(inputs),
            AttentionFunction::Structural(function) => function.call(inputs),
        }
    }
}

/// Named composite of one attention definition. Definitions without dropout batch their calls by folding mapped axes
/// into attention's batch axis and calling the same definition at the folded boundary, which keeps the definition (and
/// therefore any backend implementation of it) for batched calls. Definitions with dropout are batched structurally
/// instead, because folding would change the shape of the dropout mask and therefore the mask of every batch item,
/// except that definitions with dropout and the fused implementation reject batched calls that have a mapped input.
/// In every definition with a batching rule, a call whose inputs are all replicated stages the unbatched definition.
enum AttentionFunction<Folding, Structural> {
    /// Definition that folds mapped axes into the attention's batch axis.
    Folding(Arc<Folding>),

    /// Definition that is batched structurally.
    Structural(Arc<Structural>),
}

/// Data type in which the attention bodies evaluate the softmax operation and its log-sum-exp residual.
const SOFTMAX_DATA_TYPE: DataType = DataType::F32;

/// Shared normalized inputs and masked logits consumed by the forward and backward attention bodies.
struct PreparedAttention<V> {
    /// Normalized query.
    query: V,

    /// Normalized key expanded from key/value heads to query heads.
    expanded_key: V,

    /// Normalized value expanded from key/value heads to query heads.
    expanded_value: V,

    /// Masked logits converted to [`SOFTMAX_DATA_TYPE`].
    logits: V,

    /// Combined Boolean visibility mask in the `BNTS` score form, or [`None`] when no masking is configured.
    /// The logits of masked scores are constants, so the backward body uses it to zero their cotangents.
    mask: Option<V>,
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

    /// Log-sum-exp residual in `TN` or `BTN` form.
    Residual,
}

/// Semantic role of one attention batching output.
#[derive(Copy, Clone, Debug)]
enum AttentionBatchOutput {
    /// Tensor whose logical rank follows the input at this index.
    Tensor(usize),

    /// Cotangent of the tensor input at this index, which has the cotangent type of that input.
    Cotangent(usize),

    /// Log-sum-exp residual whose logical rank follows the query rank.
    Residual,

    /// Bias cotangent restored to the logical bias shape and to the cotangent type of the input at this index.
    Score(usize),
}

/// Returns the [`CompositeKey`] named `name` with the attributes of `configuration` and `input_structure`, including
/// the `residual` attribute when `residual` is present (refer to [`AttentionKey::to_composite_key`] for the encoding).
fn attention_composite_key(
    name: &'static str,
    configuration: AttentionConfiguration,
    input_structure: AttentionInputs<Placeholder>,
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
    key.with_attribute("bias", input_structure.bias.is_some())
        .with_attribute("mask", input_structure.mask.is_some())
        .with_attribute("query_sequence_lengths", input_structure.query_sequence_lengths.is_some())
        .with_attribute("key_value_sequence_lengths", input_structure.key_value_sequence_lengths.is_some())
        .with_attribute("dropout_state", input_structure.dropout_state.is_some())
}

/// Validates one forward attention boundary (i.e., the [`AttentionInputs`] of `input_structure` in canonical order)
/// and returns the types of its outputs. The first output is the attended output, which has the query type without
/// its layout and with the sharding that the body infers, and it is followed by the log-sum-exp residual when
/// `return_residual` is set. Every input must have matching manual variation, and reduced and unreduced inputs are
/// rejected, because softmax and the log-sum-exp residual preserve neither partial sums nor replicas that are filled
/// with zeros.
fn attention_forward_output_types(
    operation_name: &str,
    configuration: AttentionConfiguration,
    input_structure: AttentionInputs<Placeholder>,
    return_residual: bool,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    if input_types.len() != input_structure.parameter_count() {
        return Err(TypeError::invalid(format!(
            "`{}` expects {} inputs for its input structure but got {}",
            operation_name,
            input_structure.parameter_count(),
            input_types.len(),
        )));
    }

    let types = AttentionInputs::from_parameters(input_structure, input_types.iter().cloned())
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    let dimensions = validate_attention_boundary(operation_name, configuration, &types, input_types)?;

    // The output takes its batch and head placements from the query, key, and value, its sequence placement from
    // the query, and its head dimension placement from the value. The residual takes the score placements, including
    // the bias, mask, and sequence lengths as `attention_score_sharding` describes. Inputs without sharding or fully
    // replicated over one mesh keep the query's placements without inferring the contractions.
    let placed_types = [
        Some(&types.query),
        Some(&types.key),
        Some(&types.value),
        types.bias.as_ref(),
        types.mask.as_ref(),
        types.query_sequence_lengths.as_ref(),
        types.key_value_sequence_lengths.as_ref(),
    ];
    let (output_sharding, residual_sharding) = match types.query.sharding() {
        None if placed_types.iter().flatten().all(|r#type| r#type.sharding().is_none()) => (None, None),
        Some(query_sharding)
            if placed_types.iter().flatten().all(|r#type| {
                r#type.sharding().is_some_and(|sharding| {
                    sharding.mesh() == query_sharding.mesh()
                        && sharding
                            .dimensions()
                            .iter()
                            .all(|dimension| matches!(dimension, ShardingDimension::Replicated))
                })
            }) =>
        {
            let residual_sharding = query_sharding
                .with_dimensions(query_sharding.dimensions()[..types.query.rank() - 1].to_vec())
                .map_err(|error| TypeError::invalid(error.to_string()))?;
            (Some(query_sharding.clone()), Some(residual_sharding))
        }
        _ => attention_inferred_output_shardings(operation_name, &dimensions, &types)?,
    };

    // The body computes its outputs with contractions, which return arrays without an explicit layout and with the
    // sharding that those contractions infer, so the attended output has every other property of the query type.
    let output_type = types
        .query
        .clone()
        .with_layout(None)
        .with_sharding(output_sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    let mut output_types = vec![output_type];

    if return_residual {
        output_types.push(attention_residual_type(&dimensions, &types.query, residual_sharding)?);
    }

    Ok(output_types)
}

/// Infers the shardings of the output types that [`attention_forward_output_types`] returns by applying the type
/// inference of the operations of the forward body that determine them (i.e., the reshapes that normalize rank-3
/// inputs and that expand grouped key/value heads, the two contractions, and the placement of the scores between
/// them) to the types of the inputs.
fn attention_inferred_output_shardings(
    operation_name: &str,
    dimensions: &AttentionDimensions,
    types: &AttentionInputs<ArrayType>,
) -> Result<(Option<Sharding>, Option<Sharding>), TypeError> {
    let normalize = |r#type: &ArrayType| -> Result<ArrayType, TypeError> {
        if r#type.rank() == 4 {
            return Ok(r#type.clone());
        }
        let normalized_dimensions =
            std::iter::once(Dimension::Static(1)).chain(r#type.shape().dimensions().iter().cloned()).collect();
        infer_reshape_output_type(r#type, Shape::new(normalized_dimensions), None)
    };
    let expand = |r#type: ArrayType| -> Result<ArrayType, TypeError> {
        match key_value_group_types(&r#type, dimensions)? {
            None => Ok(r#type),
            Some((grouped_type, expanded_shape)) => infer_reshape_output_type(&grouped_type, expanded_shape, None),
        }
    };
    let query = normalize(&types.query)?;
    let key = expand(normalize(&types.key)?)?;
    let value = expand(normalize(&types.value)?)?;
    let scores = infer_dot_output_type(
        operation_name,
        &query,
        &key,
        &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]),
        None,
        None,
    )?;
    let score_sharding = attention_score_sharding(
        operation_name,
        &scores,
        [types.bias.as_ref(), types.mask.as_ref()],
        [types.query_sequence_lengths.as_ref(), types.key_value_sequence_lengths.as_ref()],
    )?;
    let scores = scores.with_sharding(score_sharding).map_err(|error| TypeError::invalid(error.to_string()))?;
    let attended = infer_dot_output_type(
        operation_name,
        &scores,
        &value,
        &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]),
        None,
        None,
    )?;

    // The scores have the `BNTS` form and the attended values have the `BNTH` form, which the body transposes to the
    // `BTN` form of the residual and to the `BTNH` form of the output, without the batch dimension for rank-3 queries.
    let select = |sharding: Option<&Sharding>, axes: &[usize]| -> Result<Option<Sharding>, TypeError> {
        sharding
            .map(|sharding| {
                sharding.with_dimensions(axes.iter().map(|axis| sharding.dimensions()[*axis].clone()).collect())
            })
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))
    };
    let (output_axes, residual_axes): (&[usize], &[usize]) =
        if types.query.rank() == 4 { (&[0, 2, 1, 3], &[0, 2, 1]) } else { (&[2, 1, 3], &[2, 1]) };
    Ok((select(attended.sharding(), output_axes)?, select(scores.sharding(), residual_axes)?))
}

/// Returns the sharding of the `BNTS` scores of `scores_type` after the inputs that are applied to them elementwise
/// take part in their placement. `score_inputs` are the bias and the mask, which are aligned with the trailing score
/// dimensions, and `sequence_lengths` are the query and the key/value sequence lengths, which are aligned with the
/// batch dimension. The placements combine like those of the inputs of an elementwise operation (i.e., a replicated
/// dimension yields to a sharded one, and two dimensions that are sharded differently are rejected), so a bias that is
/// sharded along a dimension that the scores replicate keeps the scores sharded along it, rather than being gathered.
/// The type rule of the forward boundary and the bodies both use this function, so that they agree on the result.
fn attention_score_sharding(
    operation_name: &str,
    scores_type: &ArrayType,
    score_inputs: [Option<&ArrayType>; 2],
    sequence_lengths: [Option<&ArrayType>; 2],
) -> Result<Option<Sharding>, TypeError> {
    let rank = scores_type.rank();
    let template = scores_type.clone().with_layout(None);
    let aligned_inputs = score_inputs
        .into_iter()
        .flatten()
        .map(|r#type| (r#type, rank - r#type.rank()))
        .chain(sequence_lengths.into_iter().flatten().map(|r#type| (r#type, 0)));
    let mut placed = template.clone();
    for (r#type, offset) in aligned_inputs {
        let Some(sharding) = r#type.sharding() else {
            continue;
        };
        let mut dimensions = vec![ShardingDimension::Replicated; rank];
        dimensions[offset..offset + r#type.rank()].clone_from_slice(sharding.dimensions());
        let aligned = sharding
            .with_dimensions(dimensions)
            .and_then(|sharding| template.clone().with_sharding(sharding))
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        placed = Broadcastable::broadcast(&placed, &aligned).map_err(|_| {
            TypeError::invalid(format!(
                "`{operation_name}` input type `{type}` has a sharding that cannot be combined with that of the \
                 attention scores `{placed}`",
            ))
        })?;
    }
    Ok(placed.sharding().cloned())
}

/// Validates one backward attention boundary (i.e., the [`AttentionInputs`] of `input_structure` in canonical order,
/// followed by the forward output, the log-sum-exp residual, and the output cotangent) and returns the types of its
/// outputs, which are the cotangents of the query, key, and value, followed by that of a present bias whose cotangent
/// space is not zero-dimensional. The forward output, residual, and output cotangent are compared by data type and
/// shape only, so that per-input metadata such as sharding never fails the structural contract.
fn attention_backward_output_types(
    operation_name: &str,
    configuration: AttentionConfiguration,
    input_structure: AttentionInputs<Placeholder>,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    if input_types.len() != input_structure.parameter_count() + 3 {
        return Err(TypeError::invalid(format!(
            "`{}` expects {} inputs for its input structure but got {}",
            operation_name,
            input_structure.parameter_count() + 3,
            input_types.len(),
        )));
    }

    let structure =
        (AttentionResiduals { inputs: input_structure, output: Placeholder, residual: Placeholder }, Placeholder);
    let (AttentionResiduals { inputs: types, output, residual }, output_cotangent) =
        Parameterized::from_parameters(structure, input_types.iter().cloned())
            .map_err(|error: ParameterError| TypeError::invalid(error.to_string()))?;
    let dimensions = validate_attention_boundary(operation_name, configuration, &types, input_types)?;

    let matches_expected = |actual: &ArrayType, expected: &ArrayType| -> bool {
        actual.data_type() == expected.data_type() && actual.shape() == expected.shape()
    };

    if !matches_expected(&output, &types.query) {
        return Err(TypeError::invalid(format!(
            "`{}` output type {} does not match the expected forward output type {}",
            operation_name, output, types.query,
        )));
    }

    let expected_output_cotangent_type = types.query.cotangent()?;
    if !matches_expected(&output_cotangent, &expected_output_cotangent_type) {
        return Err(TypeError::invalid(format!(
            "`{}` output cotangent type {} does not match the expected cotangent type {}",
            operation_name, output_cotangent, expected_output_cotangent_type,
        )));
    }

    // The residual is compared by data type and shape only, so its expected type needs no sharding.
    let expected_residual_type = attention_residual_type(&dimensions, &types.query, None)?;
    if !matches_expected(&residual, &expected_residual_type) {
        return Err(TypeError::invalid(format!(
            "`{}` residual type {} does not match the expected residual type {}",
            operation_name, residual, expected_residual_type,
        )));
    }

    // Each cotangent has the cotangent type of its input, including its sharding, which the body restores after its
    // contractions (refer to `restore_cotangent_type`), because reverse mode accumulates every cotangent into the
    // cotangent of its input.
    let mut output_types = vec![types.query.cotangent()?, types.key.cotangent()?, types.value.cotangent()?];
    if let Some(bias) = &types.bias {
        let bias_cotangent = bias.cotangent()?;
        if !bias_cotangent.is_zero_space() {
            output_types.push(bias_cotangent);
        }
    }

    Ok(output_types)
}

/// Validates the contract shared by the forward and backward attention boundaries (i.e., matching manual variation,
/// the query, key, value, bias, and mask shapes and data types, the sequence lengths, the dropout rate and random
/// state, and the absence of reduced and unreduced inputs) and returns the validated [`AttentionDimensions`].
fn validate_attention_boundary(
    operation_name: &str,
    configuration: AttentionConfiguration,
    types: &AttentionInputs<ArrayType>,
    input_types: &[ArrayType],
) -> Result<AttentionDimensions, TypeError> {
    ArrayType::check_matching_manual_variation(operation_name, &input_types.iter().collect::<Vec<_>>())?;
    let dimensions = validate_attention_inputs(
        operation_name,
        &types.query,
        &types.key,
        &types.value,
        types.bias.as_ref(),
        types.mask.as_ref(),
    )?;

    // Each sequence lengths input holds one `i32` length per batch item, so it is a rank-1 array whose extent matches
    // the batch dimension that the query, key, and value share.
    for (descriptor, lengths_type) in [
        ("query sequence lengths", &types.query_sequence_lengths),
        ("key/value sequence lengths", &types.key_value_sequence_lengths),
    ] {
        let Some(lengths_type) = lengths_type else {
            continue;
        };

        if lengths_type.data_type() != DataType::I32 {
            return Err(TypeError::invalid(format!(
                "`{}` {} must have data type `i32` but got `{}`",
                operation_name,
                descriptor,
                lengths_type.data_type(),
            )));
        }

        match lengths_type.shape().dimensions() {
            [size] if size == &dimensions.batch => {}
            [size] => {
                return Err(TypeError::invalid(format!(
                    "`{}` {} size ({}) does not match the batch dimension ({})",
                    operation_name, descriptor, size, dimensions.batch,
                )));
            }
            lengths_dimensions => {
                return Err(TypeError::invalid(format!(
                    "`{}` {} must have rank 1 but got rank {}",
                    operation_name,
                    descriptor,
                    lengths_dimensions.len(),
                )));
            }
        }
    }

    // The dropout rate lies in the open interval `(0, 1)`, because a zero rate is expressed by omitting dropout
    // altogether. The comparison is written so that it also rejects a NaN rate. A random state is only accepted
    // together with a rate, and it must have the state type of a random algorithm.
    if let Some(rate) = configuration.dropout()
        && !(rate > 0.0 && rate < 1.0)
    {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` dropout rate must lie in the open interval (0, 1) but got {rate}",
        )));
    }
    if let Some(state_type) = &types.dropout_state {
        if configuration.dropout().is_none() {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` received a dropout random state without a dropout rate",
            )));
        }
        RandomAlgorithm::from_state_type(state_type)?;
    }

    // Softmax and the log-sum-exp residual preserve neither partial sums nor replicas that are filled with zeros,
    // so reduced and unreduced inputs have no meaningful attention.
    for input_type in input_types {
        if !input_type.reduced_axes().is_empty() {
            return Err(TypeError::invalid(format!("`{operation_name}` does not support reduced inputs")));
        }
        if !input_type.unreduced_axes().is_empty() {
            return Err(TypeError::invalid(format!("`{operation_name}` does not support unreduced inputs")));
        }
    }

    Ok(dimensions)
}

/// Validates the input contract that the attention definitions share (i.e., the `TNH` or `BTNH` query, key, and value
/// shapes and data types, the divisibility of the query heads by the key/value heads, and the optional broadcastable
/// bias and mask) and returns the validated [`AttentionDimensions`]. Refer to [`DotProductAttention`] for the contract
/// itself.
fn validate_attention_inputs(
    operation_name: &str,
    query_type: &ArrayType,
    key_type: &ArrayType,
    value_type: &ArrayType,
    bias_type: Option<&ArrayType>,
    mask_type: Option<&ArrayType>,
) -> Result<AttentionDimensions, TypeError> {
    let query = infer_attention_dimensions(operation_name, "query", query_type)?;
    let key = infer_attention_dimensions(operation_name, "key", key_type)?;
    let value = infer_attention_dimensions(operation_name, "value", value_type)?;

    let data_type = query_type.data_type();
    if !data_type.is_floating_point() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` requires floating-point inputs but got data type `{data_type}`",
        )));
    }

    // The attention weights and the attended output are converted to the query data type, and `f8e8m0fnu` represents
    // neither zero nor negative values, so it can hold neither the zero weight of a hidden position nor the weighted
    // sums of the output.
    if data_type == DataType::F8E8M0FNU {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` does not support `{data_type}` inputs, because that data type represents neither \
             zero nor negative values",
        )));
    }

    for (descriptor, dimensions, input_type) in [("key", &key, key_type), ("value", &value, value_type)] {
        if input_type.data_type() != data_type {
            return Err(TypeError::invalid(format!(
                "`{}` {} data type `{}` does not match the query data type `{}`",
                operation_name,
                descriptor,
                input_type.data_type(),
                data_type,
            )));
        }

        if dimensions[0] != query[0] {
            return Err(TypeError::invalid(format!(
                "`{}` {} batch dimension ({}) does not match the query batch dimension ({})",
                operation_name, descriptor, dimensions[0], query[0],
            )));
        }

        if dimensions[3] != query[3] {
            return Err(TypeError::invalid(format!(
                "`{}` {} head dimension ({}) does not match the query head dimension ({})",
                operation_name, descriptor, dimensions[3], query[3],
            )));
        }
    }

    if value[1] != key[1] {
        return Err(TypeError::invalid(format!(
            "`{}` value sequence dimension ({}) does not match the key sequence dimension ({})",
            operation_name, value[1], key[1],
        )));
    }

    if value[2] != key[2] {
        return Err(TypeError::invalid(format!(
            "`{}` value heads dimension ({}) does not match the key heads dimension ({})",
            operation_name, value[2], key[2],
        )));
    }

    let Dimension::Static(query_heads) = &query[2] else {
        return Err(TypeError::invalid(format!(
            "`{}` query heads dimension must be static but got {}",
            operation_name, query[2],
        )));
    };

    let Dimension::Static(key_value_heads) = &key[2] else {
        return Err(TypeError::invalid(format!(
            "`{}` key/value heads dimension must be static but got {}",
            operation_name, key[2],
        )));
    };

    let Dimension::Static(head_dimension) = &query[3] else {
        return Err(TypeError::invalid(format!(
            "`{}` head dimension must be static but got {}",
            operation_name, query[3],
        )));
    };

    if *key_value_heads == 0 || *query_heads % *key_value_heads != 0 {
        return Err(TypeError::invalid(format!(
            "`{}` key/value heads dimension ({}) must divide the query heads dimension ({})",
            operation_name, key_value_heads, query_heads,
        )));
    }

    for (descriptor, score_type, expected_data_type) in
        [("bias", bias_type, None), ("mask", mask_type, Some(DataType::Boolean))]
    {
        let Some(score_type) = score_type else {
            continue;
        };

        let score_dimensions = score_type.shape().dimensions();
        if score_dimensions.len() > 4 {
            return Err(TypeError::invalid(format!(
                "`{}` {} must have rank at most 4 but got rank {}",
                operation_name,
                descriptor,
                score_dimensions.len(),
            )));
        }

        let dimensions = std::iter::repeat_n(Dimension::Static(1), 4 - score_dimensions.len())
            .chain(score_dimensions.iter().cloned())
            .collect::<Vec<_>>();
        let [score_batch, score_heads, score_rows, score_columns] = dimensions.as_slice() else {
            unreachable!("attention bias and mask shapes are normalized to exactly four dimensions")
        };

        if let Some(expected_data_type) = expected_data_type {
            if score_type.data_type() != expected_data_type {
                return Err(TypeError::invalid(format!(
                    "`{}` {} must have data type `{}` but got `{}`",
                    operation_name,
                    descriptor,
                    expected_data_type,
                    score_type.data_type(),
                )));
            }
        } else if !score_type.data_type().is_real() && !score_type.data_type().is_boolean() {
            // The bias is converted to the real data type of the scores, which would discard an imaginary part.
            return Err(TypeError::invalid(format!(
                "`{}` bias must have a real numeric or Boolean data type but got `{}`",
                operation_name,
                score_type.data_type(),
            )));
        }

        if score_batch != &Dimension::Static(1) && score_batch != &query[0] {
            return Err(TypeError::invalid(format!(
                "`{}` {} batch dimension ({}) must be 1 or match the query batch dimension ({})",
                operation_name, descriptor, score_batch, query[0],
            )));
        }

        if score_heads != &Dimension::Static(1) && score_heads != &query[2] {
            return Err(TypeError::invalid(format!(
                "`{}` {} heads dimension ({}) must be 1 or match the query heads dimension ({})",
                operation_name, descriptor, score_heads, query[2],
            )));
        }

        if score_rows != &Dimension::Static(1) && score_rows != &query[1] {
            return Err(TypeError::invalid(format!(
                "`{}` {} query-sequence dimension ({}) must be 1 or match the query sequence dimension ({})",
                operation_name, descriptor, score_rows, query[1],
            )));
        }

        if score_columns != &Dimension::Static(1) && score_columns != &key[1] {
            return Err(TypeError::invalid(format!(
                "`{}` {} key/value-sequence dimension ({}) must be 1 or match the key sequence dimension ({})",
                operation_name, descriptor, score_columns, key[1],
            )));
        }
    }

    Ok(AttentionDimensions {
        batch: query[0].clone(),
        query_sequence: query[1].clone(),
        query_heads: *query_heads,
        key_value_heads: *key_value_heads,
        head_dimension: *head_dimension,
        data_type,
    })
}

/// Returns canonical `[batch, sequence, heads, head_dimension]` dimensions for a `TNH` or `BTNH` input.
fn infer_attention_dimensions(
    operation_name: &str,
    descriptor: &str,
    value_type: &ArrayType,
) -> Result<[Dimension; 4], TypeError> {
    match value_type.shape().dimensions() {
        [sequence, heads, head_dimension] => {
            Ok([Dimension::Static(1), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        [batch, sequence, heads, head_dimension] => {
            Ok([batch.clone(), sequence.clone(), heads.clone(), head_dimension.clone()])
        }
        dimensions => Err(TypeError::invalid(format!(
            "`{}` {} must have rank 3 or 4 but got rank {}",
            operation_name,
            descriptor,
            dimensions.len(),
        ))),
    }
}

/// Returns the `[batch, query_sequence, heads]` or `[query_sequence, heads]` type of the log-sum-exp residual,
/// which has the [`SOFTMAX_DATA_TYPE`], the memory space of the query, and the provided sharding (refer to
/// [`attention_forward_output_types`]). The residual keeps the data type in which the softmax computes it for
/// every query data type, because the backward definition of a backend may consume it (e.g., cuDNN's fused kernels,
/// which return and consume it in `f32`), and rounding it to a narrower query data type would lose the normalization
/// that it carries and could overflow it.
fn attention_residual_type(
    dimensions: &AttentionDimensions,
    query_type: &ArrayType,
    sharding: Option<Sharding>,
) -> Result<ArrayType, TypeError> {
    let dimensions = if query_type.rank() == 3 {
        vec![dimensions.query_sequence.clone(), Dimension::Static(dimensions.query_heads)]
    } else {
        vec![dimensions.batch.clone(), dimensions.query_sequence.clone(), Dimension::Static(dimensions.query_heads)]
    };
    ArrayType::new(SOFTMAX_DATA_TYPE, Shape::new(dimensions))
        .with_memory(query_type.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
}

/// Normalizes attention inputs and constructs the masked score logits shared by the forward and backward bodies.
fn prepare_attention<
    V: Value<Type: ArrayMemberType>
        + FillLike
        + IotaLike
        + Add
        + Sub
        + Mul
        + And
        + Dot
        + Compare
        + Select
        + Reshape
        + Broadcast
        + BroadcastLike
        + ConvertElementType
        + Reduce
        + Reshard,
>(
    operation_name: &str,
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
    dimensions: &AttentionDimensions,
) -> Result<PreparedAttention<V>, ProgramError> {
    /// Prepends unit axes until one bias or mask input has the normalized `BNTS` score rank, which preserves the
    /// dynamic extents of the input.
    fn normalize_attention_score_input<V: Value<Type: ArrayMemberType> + Reshape>(input: V) -> Result<V, ProgramError> {
        let rank = input.r#type().as_array_type()?.rank();
        if rank == 4 {
            return Ok(input);
        }
        let dimensions = std::iter::repeat_n(Dimension::Static(1), 4 - rank)
            .chain(input.r#type().as_array_type()?.shape().dimensions().iter().cloned())
            .collect::<Vec<_>>();
        input.reshape(Shape::new(dimensions))
    }

    let query = normalize_attention_input(inputs.query.clone())?;
    let key = normalize_attention_input(inputs.key.clone())?;
    let value_input = normalize_attention_input(inputs.value.clone())?;
    let expanded_key = expand_key_value_heads(key, dimensions)?;
    let expanded_value = expand_key_value_heads(value_input, dimensions)?;
    let data_type = dimensions.data_type;

    // The score contraction, scaling, bias, and masking promote low precision inputs to `f32`, while `f64` remains
    // `f64`. The stable softmax itself always runs in `f32`, matching the public attention contract.
    let logits_type = dimensions.score_data_type();

    // The scores contract the head dimension of the query and of the expanded key for every batch item and head,
    // which gives the `BNTS` score form. Low precision inputs accumulate the contraction in the promoted data type
    // and return it there, because a contraction that returns the narrow data type rounds every score and overflows
    // the scores that the narrow data type cannot represent before they could be promoted.
    let score_dimensions = DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]);
    let scores = if data_type == logits_type {
        query.dot(&expanded_key, &score_dimensions)?
    } else {
        query.dot_with_accumulation_type(&expanded_key, &score_dimensions, logits_type)?
    };
    let scale_value = scores.fill_like(configuration.scale_for(dimensions.head_dimension))?;
    let scores = scores.mul(&scale_value)?;

    // The bias, the mask, and the sequence lengths are broadcast to the scores below, and a broadcast takes the
    // placement of its exemplar. The scores are therefore first placed according to all of them, so that an input
    // that is sharded along a dimension which the scores replicate keeps its shards instead of being gathered.
    let scores_type = scores.r#type().as_array_type()?.clone();
    let array_type = |input: &Option<V>| -> Result<Option<ArrayType>, ProgramError> {
        input.as_ref().map(|input| Ok(input.r#type().as_array_type()?.clone())).transpose()
    };
    let (bias_type, mask_type) = (array_type(&inputs.bias)?, array_type(&inputs.mask)?);
    let (query_lengths_type, key_value_lengths_type) =
        (array_type(&inputs.query_sequence_lengths)?, array_type(&inputs.key_value_sequence_lengths)?);
    let score_sharding = attention_score_sharding(
        operation_name,
        &scores_type,
        [bias_type.as_ref(), mask_type.as_ref()],
        [query_lengths_type.as_ref(), key_value_lengths_type.as_ref()],
    )?;
    let mut scores = match &score_sharding {
        Some(sharding) if scores_type.sharding() != Some(sharding) => scores.reshard(sharding)?,
        _ => scores,
    };

    // A bias is right-aligned to `BNTS`, converted to the data type of the scores, and broadcast to their shape.
    // Every axis of the normalized bias either has the extent of the corresponding score axis or is a unit axis.
    if let Some(bias) = &inputs.bias {
        let bias = normalize_attention_score_input(bias.clone())?;
        let bias = if bias.r#type().as_array_type()?.data_type() == logits_type {
            bias
        } else {
            bias.convert_element_type(logits_type)?
        };
        let bias = bias.broadcast_like(&scores, &[0, 1, 2, 3])?;
        scores = scores.add(&bias)?;
    }

    // Visibility is the conjunction of the explicit mask, causal ordering, the local window, and both sequence
    // length predicates. Attention without any of them keeps its scores as they are.
    let (logits, mask) = if inputs.mask.is_none()
        && !configuration.causal()
        && configuration.local_window().is_none()
        && inputs.query_sequence_lengths.is_none()
        && inputs.key_value_sequence_lengths.is_none()
    {
        (scores, None)
    } else {
        // The row and column coordinates of the scores are integer constants that cannot carry tangents. They take the
        // runtime shape of the scores, and so do the broadcasts of the mask and of the sequence lengths below.
        let columns = scores.iota_like(3, DataType::I32)?;
        let mut mask = match &inputs.mask {
            None => None,
            Some(mask) => {
                let mask = normalize_attention_score_input(mask.clone())?;
                Some(mask.broadcast_like(&scores, &[0, 1, 2, 3])?)
            }
        };

        // Causal masking lets query position `i` attend only to key/value positions `j <= i`.
        if configuration.causal() {
            let rows = scores.iota_like(2, DataType::I32)?;
            let causal = columns.compare(&rows, ComparisonDirection::LessThanOrEqual)?;
            mask = Some(match mask {
                None => causal,
                Some(mask) => mask.and(&causal)?,
            });
        }

        // The local window keeps the key/value positions `j` with `i - left <= j <= i + right`, which is evaluated as
        // `-left <= j - i <= right`. The offsets `j - i` of two `i32` coordinates cannot overflow, unlike `i + right`
        // for a large radius, and a radius beyond the `i32` range admits every offset, so it is clamped to that range.
        if let Some((left, right)) = configuration.local_window() {
            let rows = scores.iota_like(2, DataType::I32)?;
            let offsets = columns.sub(&rows)?;
            let lower = offsets.fill_like(-(left.min(i32::MAX as usize) as f64))?;
            let upper = offsets.fill_like(right.min(i32::MAX as usize) as f64)?;
            let in_window = offsets
                .compare(&lower, ComparisonDirection::GreaterThanOrEqual)?
                .and(&offsets.compare(&upper, ComparisonDirection::LessThanOrEqual)?)?;
            mask = Some(match mask {
                None => in_window,
                Some(mask) => mask.and(&in_window)?,
            });
        }

        // Query rows at or beyond the query length of their batch item see no key/value position at all.
        if let Some(lengths) = &inputs.query_sequence_lengths {
            let rows = scores.iota_like(2, DataType::I32)?;
            let bounds = lengths.broadcast_like(&rows, &[0])?;
            let in_range = rows.compare(&bounds, ComparisonDirection::LessThan)?;
            mask = Some(match mask {
                None => in_range,
                Some(mask) => mask.and(&in_range)?,
            });
        }

        // Key/value columns at or beyond the key/value length of their batch item are invisible to every row.
        if let Some(lengths) = &inputs.key_value_sequence_lengths {
            let bounds = lengths.broadcast_like(&columns, &[0])?;
            let in_range = columns.compare(&bounds, ComparisonDirection::LessThan)?;
            mask = Some(match mask {
                None => in_range,
                Some(mask) => mask.and(&in_range)?,
            });
        }

        // Invisible scores use a finite large negative value rather than negative infinity, so that a row without
        // any visible position still has a finite maximum and a finite log-sum-exp residual. The value is finite
        // in the softmax data type for every data type of the scores, because a value that is only finite in `f64`
        // would become negative infinity when the logits are converted for the softmax. At least one of the masks
        // above is present on this path, so the visibility is known. That value only stands in for the scores of rows
        // without any visible position, which attend uniformly. In every other row the invisible scores are negative
        // infinity, so that they have exactly zero probability however small the visible scores of the row are (e.g.,
        // with a bias below the finite value).
        let mask = mask.unwrap();
        let has_visible = mask.reduce(&[3], ReductionKind::Any)?.broadcast_like(&mask, &[0, 1, 2])?;
        let masked =
            has_visible.select(&scores.fill_like(f64::NEG_INFINITY)?, &scores.fill_like(-0.7 * f32::MAX as f64)?)?;
        (mask.select(&scores, &masked)?, Some(mask))
    };

    let logits =
        if logits_type == SOFTMAX_DATA_TYPE { logits } else { logits.convert_element_type(SOFTMAX_DATA_TYPE)? };

    Ok(PreparedAttention { query, expanded_key, expanded_value, logits, mask })
}

/// Adds the implicit batch axis used to evaluate one unbatched `TNH` attention input (i.e., a query, key, value,
/// forward output, or output cotangent). Inserting a unit axis preserves the dynamic extents of the input.
fn normalize_attention_input<V: Value<Type: ArrayMemberType> + Reshape>(input: V) -> Result<V, ProgramError> {
    if input.r#type().as_array_type()?.rank() == 4 {
        return Ok(input);
    }
    input.expand_dimensions(0)
}

/// Axes of the `BSKGH` form to which [`expand_key_value_heads`] maps the axes of its `BSKH` input.
const KEY_VALUE_GROUP_AXES: [usize; 4] = [0, 1, 2, 4];

/// Returns the types of the two steps of [`expand_key_value_heads`] for a normalized `BSKH` key or value type (i.e.,
/// the `BSKGH` type that replicates the input along a new group axis, and the `BSNH` shape that merges the key/value
/// head and group axes into the query head axis), or [`None`] when the key/value heads are the query heads.
fn key_value_group_types(
    r#type: &ArrayType,
    dimensions: &AttentionDimensions,
) -> Result<Option<(ArrayType, Shape)>, TypeError> {
    if dimensions.key_value_heads == dimensions.query_heads {
        return Ok(None);
    }

    let group = dimensions.query_heads / dimensions.key_value_heads;
    let [batch, length, _, head_dimension] = r#type.shape().dimensions() else {
        return Err(TypeError::invalid(format!(
            "attention key or value type `{}` does not have the normalized rank 4",
            r#type,
        )));
    };

    // The grouped type keeps the memory space and the placement of the input along the mapped axes, and its group
    // axis is replicated, because a broadcast cannot change manual variation.
    let sharding = r#type
        .sharding()
        .map(|sharding| sharding.with_broadcasted_dimensions(5, &KEY_VALUE_GROUP_AXES))
        .transpose()
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    let grouped_dimensions =
        vec![batch.clone(), length.clone(), dimensions.key_value_heads.into(), group.into(), head_dimension.clone()];
    let grouped_type = ArrayType::new(r#type.data_type(), Shape::new(grouped_dimensions))
        .with_memory(r#type.memory())
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))?;
    let expanded_dimensions =
        vec![batch.clone(), length.clone(), dimensions.query_heads.into(), head_dimension.clone()];
    Ok(Some((grouped_type, Shape::new(expanded_dimensions))))
}

/// Expands grouped key/value heads to the query heads. The `BSKH` input is replicated along a new group axis, which
/// gives the `BSKGH` form, and its key/value head and group axes are then merged into the query head axis of the
/// `BSNH` form. Both steps keep the batch and sequence dimensions, so they support dynamic extents along them.
fn expand_key_value_heads<V: Value<Type: ArrayMemberType> + Reshape + Broadcast>(
    key_or_value: V,
    dimensions: &AttentionDimensions,
) -> Result<V, ProgramError> {
    let group_types = key_value_group_types(key_or_value.r#type().as_array_type()?, dimensions)?;
    match group_types {
        None => Ok(key_or_value),
        Some((grouped_type, expanded_shape)) => {
            key_or_value.broadcast(grouped_type, &KEY_VALUE_GROUP_AXES)?.reshape(expanded_shape)
        }
    }
}

/// Zeros query rows outside each batch item's logical sequence length.
fn zero_query_rows<V: Value + FillLike + IotaLike + Compare + Select + BroadcastLike>(
    rows_value: V,
    query_lengths: &V,
    row_axis: usize,
) -> Result<V, ProgramError> {
    let rows = rows_value.iota_like(row_axis, DataType::I32)?;
    let lengths = query_lengths.broadcast_like(&rows, &[0])?;
    let in_range = rows.compare(&lengths, ComparisonDirection::LessThan)?;
    let zero = rows_value.fill_like(0.0)?;
    in_range.select(&rows_value, &zero)
}

/// Applies the configured dropout to `probabilities`, whose shape is the normalized `BNTS` score shape, using the
/// dropout random state of `inputs` (refer to [`AttentionInputs::dropout_state`]). The forward and backward bodies
/// call this with arrays of the same shape and the same state, so they apply the same mask.
fn apply_dropout<V: Value + Dropout>(
    probabilities: V,
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
) -> Result<V, ProgramError> {
    let Some(rate) = configuration.dropout() else {
        return Ok(probabilities);
    };
    Ok(probabilities.dropout(rate, inputs.dropout_state.as_ref())?.0)
}

/// Returns `value` replicated along those of its dimensions `axes` that explicit mesh axes shard, and `value` itself
/// when no such axis shards any of them.
fn replicate_dimensions<V: Value<Type: ArrayMemberType> + Reshard>(
    value: V,
    axes: &[usize],
) -> Result<V, ProgramError> {
    let r#type = value.r#type().as_array_type()?.clone();
    let Some(sharding) = r#type.sharding() else {
        return Ok(value);
    };
    let mut dimensions = sharding.dimensions().to_vec();
    for axis in axes {
        if dimensions[*axis].has_explicit_axis(sharding.mesh()) {
            dimensions[*axis] = ShardingDimension::Replicated;
        }
    }
    if dimensions == sharding.dimensions() {
        return Ok(value);
    }
    let replicated = sharding.with_dimensions(dimensions).map_err(|error| TypeError::invalid(error.to_string()))?;
    value.reshard(&replicated)
}

/// Contracts `lhs` with `rhs` in the backward body, where one of them is a cotangent and the other one is a value of
/// the forward body, and returns the result with the provided sharding, or with the one that the contraction infers
/// when there is none. The placements that the forward body infers can give the two inputs placements that no
/// contraction of them has, even though every contraction of the forward body has one (e.g., an output cotangent and
/// a value whose contracted head dimensions are both sharded, or a cotangent whose query sequence and a value whose
/// key/value sequence use one mesh axis, which the result cannot use twice). The cotangent is then replicated before
/// the contraction, so that the values of the forward body keep their placements. The result of a contraction that
/// feeds elementwise operations of the backward body takes the sharding of the values that it meets there.
fn contract_cotangent<V: Value<Type: ArrayMemberType> + Dot + Reshard>(
    operation_name: &str,
    lhs: &V,
    rhs: &V,
    dimensions: &DotDimensionNumbers,
    cotangent_is_lhs: bool,
    sharding: Option<&Sharding>,
) -> Result<V, ProgramError> {
    let lhs_type = lhs.r#type().as_array_type()?.clone();
    let rhs_type = rhs.r#type().as_array_type()?.clone();
    let contraction = if infer_dot_output_type(operation_name, &lhs_type, &rhs_type, dimensions, None, None).is_ok() {
        lhs.dot(rhs, dimensions)?
    } else if cotangent_is_lhs {
        let axes = (0..lhs_type.rank()).collect::<Vec<_>>();
        replicate_dimensions(lhs.clone(), axes.as_slice())?.dot(rhs, dimensions)?
    } else {
        let axes = (0..rhs_type.rank()).collect::<Vec<_>>();
        lhs.dot(&replicate_dimensions(rhs.clone(), axes.as_slice())?, dimensions)?
    };
    match sharding {
        Some(sharding) if contraction.r#type().as_array_type()?.sharding() != Some(sharding) => {
            contraction.reshard(sharding)
        }
        _ => Ok(contraction),
    }
}

/// Returns `cotangent` with the sharding of `cotangent_type`, which is the cotangent type of the input that it belongs
/// to and has its data type and shape. The contractions of the backward body place each cotangent according to the
/// inputs that they contract (e.g., the head dimension of the query cotangent comes from the key), while reverse mode
/// accumulates each cotangent into the cotangent of its input and therefore requires that type. A cotangent that
/// already has it is returned unchanged, so inputs that are placed alike stage nothing here. A sharding that the input
/// has is restored with a reshard, while the absence of a sharding (i.e., an input without one that attends together
/// with placed inputs) is metadata that a broadcast to the shape of the cotangent records.
fn restore_cotangent_type<V: Value<Type: ArrayMemberType> + Broadcast + Reshard>(
    cotangent: V,
    cotangent_type: &ArrayType,
) -> Result<V, ProgramError> {
    let r#type = cotangent.r#type().as_array_type()?.clone();
    let cotangent = match cotangent_type.sharding() {
        Some(sharding) if r#type.sharding() != Some(sharding) => cotangent.reshard(sharding)?,
        _ => cotangent,
    };
    if cotangent_type.sharding().is_some() || r#type.sharding().is_none() {
        return Ok(cotangent);
    }

    // A broadcast to the shape that the cotangent already has only records the metadata of its output type.
    let axes = (0..cotangent_type.rank()).collect::<Vec<_>>();
    cotangent.broadcast(cotangent_type.clone(), axes.as_slice())
}

/// Returns the named composite of the forward definition with the provided configuration and residual request,
/// which serves every input structure under the [`AttentionKey`] of that structure. The body of the definition
/// evaluates portable scaled dot-product attention with ordinary array operations and returns the attended output,
/// followed by the log-sum-exp residual when `return_residual` is set. Its constants, coordinates, and broadcasts
/// take their shapes from existing values, and its reshapes keep the batch and sequence dimensions, so the same
/// body supports static and dynamic extents and backends need no attention mathematics of their own.
fn forward_definition<V: Value, O: Operation<Type = V::Type>>(
    configuration: AttentionConfiguration,
    return_residual: bool,
) -> AttentionFunction<
    BatchedComposite<V, O, AttentionInputs<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>>,
    Composite<V, O, AttentionInputs<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>>,
>
where
    V: 'static,
    V::Type: ArrayMemberType + From<ArrayType> + BatchableType,
    O: 'static,
    CustomRuleTracer<V, O>: AttentionOperations,
    CustomRuleDefinition<V, O>: AttentionRules<V, O>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>
        + CompositeCallContext<Type = V::Type, Value = CustomRuleTracer<V, O>, Constant = V, Operation = O>,
{
    let function = composite(
        move |inputs: AttentionInputs<CustomRuleTracer<V, O>>| -> Result<Vec<CustomRuleTracer<V, O>>, ProgramError> {
            // The output type rule has already validated this boundary, so the types are only needed for the
            // validated dimensions and the rank of the query.
            let input_types = inputs
                .parameters()
                .map(|input| -> Result<ArrayType, ProgramError> { Ok(input.r#type().as_array_type()?.clone()) })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let query_rank = input_types[0].rank();
            let dimensions = validate_attention_inputs(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                &input_types[0],
                &input_types[1],
                &input_types[2],
                inputs.bias.as_ref().map(|_| &input_types[3]),
                inputs.mask.as_ref().map(|_| &input_types[3 + usize::from(inputs.bias.is_some())]),
            )?;

            // The softmax over the key/value axis subtracts the maximum of every row before exponentiating,
            // so that the exponentials cannot overflow and the largest one is exactly one.
            let prepared =
                prepare_attention(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, &inputs, configuration, &dimensions)?;
            let maxima = prepared.logits.reduce(&[3], ReductionKind::Max)?;
            let maxima_broadcast = maxima.broadcast_like(&prepared.logits, &[0, 1, 2])?;
            let shifted = prepared.logits.sub(&maxima_broadcast)?;
            let exponentials = shifted.exp()?;
            let sums = exponentials.reduce(&[3], ReductionKind::Sum)?;
            let sums_broadcast = sums.broadcast_like(&exponentials, &[0, 1, 2])?;

            // Dropout applies to the normalized probabilities in the softmax data type, before they return to the data
            // type of the query.
            let weights = apply_dropout(exponentials.div(&sums_broadcast)?, &inputs, configuration)?;
            let weights = if dimensions.data_type == SOFTMAX_DATA_TYPE {
                weights
            } else {
                weights.convert_element_type(dimensions.data_type)?
            };

            // The weights have the `BNTS` score form, and the values have the `BSNH` form with their heads expanded to
            // the query heads. Contracting the key/value axis for every batch item and head gives `BNTH`, which is
            // transposed to the `BTNH` form of the query.
            let attended = weights
                .dot(&prepared.expanded_value, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]))?;
            let mut output = attended.transpose([0, 2, 1, 3])?;

            // Query rows at or beyond the query length of their batch item are exact zeros.
            if let Some(query_lengths) = &inputs.query_sequence_lengths {
                output = zero_query_rows(output, query_lengths, 1)?;
            }

            // Unbatched attention was evaluated with an implicit unit batch axis, which its outputs do not have.
            let output = if query_rank == 4 { output } else { output.squeeze(0)? };

            // The residual is the logarithm of the sum of the exponentiated logits of every row, which is the row
            // maximum plus the logarithm of the sum of the shifted exponentials. It is a statistic of the forward
            // pass rather than a differentiable output, so it stops gradients, and it keeps the softmax data type.
            let residual = if return_residual {
                let residual = maxima.add(&sums.log()?)?;
                let residual = residual.transpose([0, 2, 1])?;
                let residual = if query_rank == 4 { residual } else { residual.squeeze(0)? };
                Some(residual.stop_gradient()?)
            } else {
                None
            };

            Ok(std::iter::once(output).chain(residual).collect())
        },
    )
    .with_name(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME)
    .with_named_composite_per_structure(
        move |input_structure: &AttentionInputs<Placeholder>| {
            AttentionKey::Forward { configuration, input_structure: *input_structure, return_residual }
                .to_composite_key()
        },
        move |input_types: AttentionInputs<V::Type>| -> Result<Vec<V::Type>, ProgramError> {
            let input_structure = input_types.parameter_structure();
            let input_types = input_types
                .parameters()
                .map(|r#type| r#type.as_array_type().cloned())
                .collect::<Result<Vec<_>, _>>()?;
            let output_types = attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                configuration,
                input_structure,
                return_residual,
                input_types.as_slice(),
            )?;
            Ok(output_types.into_iter().map(Into::into).collect())
        },
    );

    if configuration.dropout().is_some() {
        match configuration.implementation() {
            AttentionImplementation::Fused => {
                // These definitions batch through the rule below, which rejects every call that has a mapped input.
            }
            AttentionImplementation::Automatic | AttentionImplementation::Portable => {
                // Folding would change the shape of the dropout mask and therefore the mask of every batch item,
                // so these definitions keep the structural batching of their bodies.
                return AttentionFunction::Structural(Arc::new(function));
            }
        }
    }

    AttentionFunction::Folding(Arc::new_cyclic(
        |weak: &Weak<BatchedComposite<V, O, AttentionInputs<CustomRuleTracer<V, O>>, Vec<CustomRuleTracer<V, O>>>>| {
            let weak = weak.clone();
            function.with_batching(move |inputs, axes: AttentionInputs<BatchAxis>, _| {
                // A staged call outlives the function that staged it when its program is batched after the function
                // is dropped, in which case the folded call stages an equivalent definition.
                let function = match weak.upgrade() {
                    Some(function) => function,
                    None => match forward_definition::<V, O>(configuration, return_residual) {
                        AttentionFunction::Folding(function) => function,
                        AttentionFunction::Structural(_) => {
                            unreachable!("structurally batched definitions have no batching rule")
                        }
                    },
                };

                if configuration.dropout().is_some() {
                    // Only definitions with dropout and the fused implementation reach this rule with dropout. A
                    // batched call loses the identity of its definition (i.e., it lowers its structurally batched
                    // body), so batching a fused call with dropout would silently replace the fused kernels and
                    // their dropout mask with the portable body and its mask. That would both change the requested
                    // numerics and break the agreement between the dropout masks of the forward and backward
                    // definitions when only one of them is batched (e.g., when the pullback of an unbatched call is
                    // batched). A call whose inputs are all replicated computes one result that every batch item
                    // shares, so it stages the unbatched definition and keeps the fused kernels and their mask.
                    if axes.parameters().any(|axis| axis.axis().is_some()) {
                        return Err(ProgramError::UnsupportedOperation {
                            message: format!(
                                "named composite `{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` with dropout cannot be \
                                 batched with the fused implementation, because a batched call cannot use the fused \
                                 kernels or reproduce their dropout mask; use the portable implementation or a \
                                 dropout state",
                            ),
                        });
                    }
                    let outputs = function.call(inputs)?;
                    let output_axes = vec![BatchAxis::replicated(); outputs.len()];
                    return Ok((outputs, output_axes));
                }

                let input_structure = axes.parameter_structure();
                let key = AttentionKey::Forward { configuration, input_structure, return_residual };
                batch_attention(
                    inputs.into_parameters().collect(),
                    axes.into_parameters().collect(),
                    key.batch_input_roles().as_slice(),
                    |output_count| key.batch_output_roles(output_count),
                    |inputs| function.call(AttentionInputs::from_parameters(input_structure, inputs)?),
                )
            })
        },
    ))
}

/// Returns the named composite of the backward definition with the provided configuration, which serves every input
/// structure under the [`AttentionKey`] of that structure. The body of the definition evaluates the portable attention
/// backward pass using the same extents and mask construction as the forward body (refer to [`forward_definition`]),
/// and returns the cotangents of the query, key, and value, followed by that of a present bias whose cotangent space
/// is not zero-dimensional. Its inputs are the forward inputs followed by the attended output, its `TN` or `BTN`
/// log-sum-exp residual, and the output cotangent. Boolean masks, integer sequence lengths, dropout states, and
/// a bias whose cotangent space is zero-dimensional have no cotangent outputs.
///
/// The body recomputes the normalized probabilities `P` from the masked logits in `f32`, which avoids the precision
/// that rounding the output to a narrower query data type loses, and so the output and the residual are validated
/// inputs for backend implementations whose numerical values this body does not use. Everything other than that
/// softmax runs in the query data type promoted to at least `f32`, like the scores of the forward body, so `f64`
/// inputs keep their range. It computes:
///
/// ```text
/// dV = transpose(round(P)) * dO
/// dW = dO * transpose(V)
/// C = dW - dW[pivot]
/// dS = mask * P * (C - sum(P * C))
/// dQ = scale * dS * K
/// dK = scale * transpose(dS) * Q,
/// ```
///
/// where the pivot is the first maximum logit of each row, whose `dS` entry is replaced by the negative sum of all
/// other entries, which preserves the dominant derivative when its probability rounds to one. `round` rounds the
/// probabilities to the query data type, which is how the forward body contracts them with the values, and `mask`
/// zeroes the cotangents of the invisible scores, whose logits are constants. That only changes rows without any
/// visible position, because the probabilities of invisible scores are zero in every other row. `C` is evaluated on
/// halved weight cotangents and multiplied by doubled probabilities, so that the difference of two finite cotangents
/// cannot overflow and a tiny probability cannot underflow its product before the factor of two returns. Grouped query
/// heads are reduced back to each key/value head, a broadcast bias cotangent is reduced to the shape of its bias, and
/// padded query rows are zeroed. With dropout, `P` is the dropped probabilities in `dV`, and the cotangent of the
/// dropped probabilities is mapped back through the same dropout mask, which the same random state and probability
/// shape reproduce.
fn backward_definition<V: Value, O: Operation<Type = V::Type>>(
    configuration: AttentionConfiguration,
) -> AttentionFunction<
    BatchedComposite<
        V,
        O,
        (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>),
        Vec<CustomRuleTracer<V, O>>,
    >,
    Composite<V, O, (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>), Vec<CustomRuleTracer<V, O>>>,
>
where
    V: 'static,
    V::Type: ArrayMemberType + From<ArrayType> + BatchableType,
    O: 'static,
    CustomRuleTracer<V, O>: AttentionOperations,
    CustomRuleDefinition<V, O>: AttentionRules<V, O>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>
        + CompositeCallContext<Type = V::Type, Value = CustomRuleTracer<V, O>, Constant = V, Operation = O>,
{
    /// Sums the `BSNH` key or value cotangent of grouped query heads back to the `BSKH` form of the key/value heads
    /// that those query heads share, by splitting its query head axis into the key/value head and group axes of the
    /// `BSKGH` form and summing over the group axis. Like [`expand_key_value_heads`], it keeps the batch and sequence
    /// dimensions. A query head axis that is sharded (i.e., like the query heads) is replicated first, because a
    /// mesh axis that shards the query heads cannot shard either of the two axes that they split into.
    fn sum_key_value_head_groups<V: Value<Type: ArrayMemberType> + Reduce + Reshape + Reshard>(
        cotangent: V,
        dimensions: &AttentionDimensions,
    ) -> Result<V, ProgramError> {
        let group = dimensions.query_heads / dimensions.key_value_heads;
        let cotangent = replicate_dimensions(cotangent, &[2])?;
        let r#type = cotangent.r#type();
        let r#type = r#type.as_array_type()?;
        let grouped_dimensions = vec![
            r#type.dimension(0),
            r#type.dimension(1),
            dimensions.key_value_heads.into(),
            group.into(),
            dimensions.head_dimension.into(),
        ];
        cotangent.reshape(Shape::new(grouped_dimensions))?.reduce(&[3], ReductionKind::Sum)
    }

    let function = composite(move |inputs: (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>)| {
        // The forward output and its residual are validated inputs for backend implementations of this definition.
        // The portable body recomputes the probabilities instead of using them, so it only needs the forward inputs
        // and the output cotangent.
        let (AttentionResiduals { inputs, .. }, output_cotangent) = &inputs;

        // The output type rule has already validated this boundary, so the types are only needed for the
        // validated dimensions and the ranks of the query, key, and value.
        let input_types = inputs
            .parameters()
            .map(|input| -> Result<ArrayType, ProgramError> { Ok(input.r#type().as_array_type()?.clone()) })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let input_ranks = [input_types[0].rank(), input_types[1].rank(), input_types[2].rank()];
        let dimensions = validate_attention_inputs(
            DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
            &input_types[0],
            &input_types[1],
            &input_types[2],
            inputs.bias.as_ref().map(|_| &input_types[3]),
            inputs.mask.as_ref().map(|_| &input_types[3 + usize::from(inputs.bias.is_some())]),
        )?;
        let prepared =
            prepare_attention(DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, inputs, configuration, &dimensions)?;
        let output_cotangent = normalize_attention_input(output_cotangent.clone())?;

        // The output has the query data type, which can lose the normalization correction in narrow formats.
        // Recompute probabilities in the softmax type instead of recovering them from the output and the residual.
        let maxima = prepared.logits.reduce(&[3], ReductionKind::Max)?;
        let maxima = maxima.broadcast_like(&prepared.logits, &[0, 1, 2])?;
        let exponentials = prepared.logits.sub(&maxima)?.exp()?;
        let sums = exponentials.reduce(&[3], ReductionKind::Sum)?;
        let sums = sums.broadcast_like(&prepared.logits, &[0, 1, 2])?;
        let weights = exponentials.div(&sums)?;

        // Pick exactly one maximum, including when logits tie. Keeping the pivot out of the final sum avoids
        // subtracting two nearly equal dominant contributions. The integer constants cannot carry tangents.
        let columns = prepared.logits.iota_like(3, DataType::I32)?;
        let sentinel = columns.fill_like(i32::MAX)?;
        let is_maximum = prepared.logits.compare(&maxima, ComparisonDirection::Equal)?;
        let pivot = is_maximum.select(&columns, &sentinel)?.reduce(&[3], ReductionKind::Min)?;

        // The column coordinates share the shape and the data type of the pivot, so they are its exemplar.
        let pivot = pivot.broadcast_like(&columns, &[0, 1, 2])?;
        let is_pivot = columns.compare(&pivot, ComparisonDirection::Equal)?;

        // Query rows at or beyond the query length of their batch item are zero in the forward output, so their
        // output cotangents contribute nothing.
        let output_cotangent = match &inputs.query_sequence_lengths {
            None => output_cotangent,
            Some(query_lengths) => zero_query_rows(output_cotangent, query_lengths, 1)?,
        };

        // The backward mathematics outside the softmax runs in the data type of the scores (i.e., the query data type
        // promoted to at least `f32`), like the forward body, so that it keeps the range and precision of `f64` inputs.
        let score_data_type = dimensions.score_data_type();
        let convert =
            |array: CustomRuleTracer<V, O>, data_type: DataType| -> Result<CustomRuleTracer<V, O>, ProgramError> {
                if array.r#type().as_array_type()?.data_type() == data_type {
                    Ok(array)
                } else {
                    Ok(array.convert_element_type(data_type)?)
                }
            };
        let query = convert(prepared.query, score_data_type)?;
        let key = convert(prepared.expanded_key, score_data_type)?;
        let value = convert(prepared.expanded_value, score_data_type)?;
        let output_cotangent = convert(output_cotangent, score_data_type)?;

        // With dropout, the output uses the dropped probabilities `D = dropout(P)`, so the value cotangent uses `D`
        // and the cotangent of `P` is the cotangent of `D` through the same mask. The forward body rounds `D` to the
        // query data type before it contracts it with the values, and the value cotangent uses the same rounded
        // weights, because the output is linear in the values through exactly those weights.
        let dropped_weights = apply_dropout(weights.clone(), inputs, configuration)?;
        let value_weights = convert(convert(dropped_weights, dimensions.data_type)?, score_data_type)?;
        let weights = convert(weights, score_data_type)?;

        // TODO(eaplatanios): Stabilize the weight cotangents before this contraction, if cotangents and values whose
        //  products leave the range of the score data type ever need to be supported. The issue is that the weight
        //  cotangents `dW = dO * transpose(V)` are contracted before any stabilization. A product of an output
        //  cotangent with a value that is beyond the range of the score data type makes them infinite, and the
        //  centering below then computes `inf - inf`, so the query, key, and bias cotangents of that row are NaN even
        //  when the exact ones are finite. For example, with `f32` inputs, a scale of one, the query `[0]`, the keys
        //  `[0, 0]`, the values `[1e20, 1e20]`, and the output cotangent `[1e20]`, the output does not depend on the
        //  query or on the keys, so their cotangents are exactly zero, but `dW` is `[inf, inf]` and they come out as
        //  NaN. The value cotangent does not use `dW` and is correct (i.e., `[5e19, 5e19]`). The halving below cannot
        //  help, because it only runs after the overflow. JAX's `dot_product_attention` has no range handling here
        //  either and returns the same NaNs for this example (checked with JAX 0.12). The change would rescale both
        //  inputs of the contraction and undo that after the centering, which is linear in `dW`. That is, it would
        //  divide every row of the output cotangent (i.e., along the head dimension, for each batch item, query
        //  position, and head) by its largest magnitude `a`, divide every row of the values (for each batch item,
        //  key/value position, and head) by its largest magnitude `b`, contract the two rescaled arrays, and multiply
        //  the logit cotangents by `a` and by `b`, each broadcast to the `BNTS` score shape, once they are centered.
        //  A row of zeros would divide by one, and a row with a non-finite magnitude would stay non-finite. To keep
        //  ordinary gradients bit-for-bit as they are, `a` and `b` would be replaced by one unless they exceed a
        //  threshold below which no product can overflow (which depends on the score data type and on the head
        //  dimension), because dividing and multiplying by one are exact. The trade-offs are the following. Every
        //  backward pass would stage two more reductions (the magnitudes of the output cotangent and of the values,
        //  which need a maximum and a minimum or an absolute value each) and two more multiplications of the size of
        //  the scores, together with the comparisons and selections of the threshold, whether or not anything is out
        //  of range. Without the threshold, every logit cotangent would be rounded two more times (by the division and
        //  by the multiplication), which gives up the single rounding that the halving below is arranged to keep. With
        //  the threshold, gradients near it would change in their last bits depending on whether a row is rescaled.
        //  The fused backward kernels do none of this, so the portable and the fused implementations would then differ
        //  in this range, and the staged backward program (and its snapshots and lowering) would change for every
        //  caller.
        let score_sharding = prepared.logits.r#type().as_array_type()?.sharding().cloned();
        let weight_cotangents = apply_dropout(
            contract_cotangent(
                DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
                &output_cotangent,
                &value,
                &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]),
                true,
                score_sharding.as_ref(),
            )?,
            inputs,
            configuration,
        )?;

        // The softmax derivative below subtracts weight cotangents from each other, and the difference of two finite
        // cotangents with opposite signs can overflow even when the logit cotangents are representable. It therefore
        // uses halved weight cotangents, whose differences are always finite, and the factor of two returns with the
        // probabilities that multiply those differences. Doubling a probability is exact and cannot overflow, so
        // each product is rounded once to the logit cotangent itself. Returning the factor only afterwards would let
        // the product of a tiny probability with a halved difference underflow first, and folding it into the score
        // scale would let a large scale overflow. The weight cotangents themselves have to be representable, though
        // (refer to the note above their contraction).
        let zero = weights.fill_like(0.0)?;
        let halved_weight_cotangents = weight_cotangents.mul(&weights.fill_like(0.5)?)?;
        let pivot_cotangent = is_pivot.select(&halved_weight_cotangents, &zero)?.reduce(&[3], ReductionKind::Sum)?;
        let pivot_cotangent = pivot_cotangent.broadcast_like(&weights, &[0, 1, 2])?;

        // Center the incoming weight cotangents before their weighted sum. This preserves a tiny dominant derivative
        // and gives exact zero when every weight cotangent is equal, even if the rounded probabilities do not sum to
        // one.
        let centered = halved_weight_cotangents.sub(&pivot_cotangent)?;
        let delta = weights.mul(&centered)?.reduce(&[3], ReductionKind::Sum)?;
        let delta = delta.broadcast_like(&weights, &[0, 1, 2])?;
        let logit_cotangents = weights.add(&weights)?.mul(&centered.sub(&delta)?)?;

        // The logit cotangents of every row sum to zero, so the pivot's entry is replaced by the negative sum of the
        // other entries. That keeps the dominant derivative when the probability of the pivot rounds to one.
        let other_cotangents = is_pivot.select(&zero, &logit_cotangents)?;
        let pivot_cotangent = other_cotangents.reduce(&[3], ReductionKind::Sum)?;
        let pivot_cotangent = pivot_cotangent.broadcast_like(&weights, &[0, 1, 2])?;
        let logit_cotangents = is_pivot.select(&zero.sub(&pivot_cotangent)?, &other_cotangents)?;

        // The logits of invisible scores are constants, so their cotangents contribute nothing to the scores. Rows
        // with a visible position already have zero probabilities there, while a row without any visible position
        // has uniform probabilities and would otherwise leak cotangents to its query, to the keys, and to the bias.
        let logit_cotangents = match &prepared.mask {
            Some(mask) => mask.select(&logit_cotangents, &zero)?,
            None => logit_cotangents,
        };

        // The scores are `scale * Q * transpose(K)`, so the query and key cotangents contract the scaled logit
        // cotangents with the keys and the queries, and the value cotangent contracts the dropped probabilities with
        // the output cotangent. Every result is transposed from `BNTH` or `BNSH` to the form of its input. The scale
        // is the coefficient of the forward body, so both bodies round it in the same way.
        let scale_value = logit_cotangents.fill_like(configuration.scale_for(dimensions.head_dimension))?;
        let scaled_logit_cotangents = logit_cotangents.mul(&scale_value)?;
        let mut query_cotangent = contract_cotangent(
            DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
            &scaled_logit_cotangents,
            &key,
            &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]),
            true,
            None,
        )?
        .transpose([0, 2, 1, 3])?;
        if let Some(query_lengths) = &inputs.query_sequence_lengths {
            query_cotangent = zero_query_rows(query_cotangent, query_lengths, 1)?;
        }
        let key_cotangent = contract_cotangent(
            DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
            &scaled_logit_cotangents,
            &query,
            &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]),
            true,
            None,
        )?
        .transpose([0, 2, 1, 3])?;
        let value_cotangent = contract_cotangent(
            DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
            &value_weights,
            &output_cotangent,
            &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]),
            false,
            None,
        )?
        .transpose([0, 2, 1, 3])?;

        // Grouped query heads share one key/value head, so the key and value cotangents of every group are summed
        // back to that head.
        let (key_cotangent, value_cotangent) = if dimensions.key_value_heads == dimensions.query_heads {
            (key_cotangent, value_cotangent)
        } else {
            (
                sum_key_value_head_groups(key_cotangent, &dimensions)?,
                sum_key_value_head_groups(value_cotangent, &dimensions)?,
            )
        };

        // Every cotangent returns to the cotangent data type, to the rank, and to the placement of its input.
        let mut cotangents = [query_cotangent, key_cotangent, value_cotangent]
            .into_iter()
            .zip(input_types[..3].iter().zip(input_ranks))
            .map(|(cotangent, (input_type, rank))| {
                let cotangent_type = input_type.cotangent()?;
                let cotangent = if cotangent.r#type().as_array_type()?.data_type() == cotangent_type.data_type() {
                    cotangent
                } else {
                    cotangent.convert_element_type(cotangent_type.data_type())?
                };
                let cotangent = if rank == 4 { cotangent } else { cotangent.squeeze(0)? };
                restore_cotangent_type(cotangent, &cotangent_type)
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        if let Some(bias) = &inputs.bias {
            let bias_type = bias.r#type().as_array_type()?.clone();
            let bias_cotangent_type = bias_type.cotangent()?;

            // A bias whose cotangent space is zero-dimensional (e.g., an integer bias) has no cotangent output.
            if bias_cotangent_type.is_zero_space() {
                return Ok(cotangents);
            }

            // The bias was right-aligned and broadcast to the `BNTS` score shape, so its cotangent sums the logit
            // cotangents over the axes along which it was broadcast and then returns to the shape of the bias.
            let bias_dimensions = std::iter::repeat_n(Dimension::Static(1), 4 - bias_type.rank())
                .chain(bias_type.shape().dimensions().iter().cloned())
                .collect::<Vec<_>>();
            let logit_type = logit_cotangents.r#type().as_array_type()?.clone();
            let reduce_axes = bias_dimensions
                .iter()
                .zip(logit_type.shape().dimensions())
                .enumerate()
                .filter_map(|(axis, (bias, logit))| (bias == &Dimension::Static(1) && logit != bias).then_some(axis))
                .collect::<Vec<_>>();
            let bias_cotangent = if reduce_axes.is_empty() {
                logit_cotangents
            } else {
                logit_cotangents.reduce(reduce_axes.as_slice(), ReductionKind::Sum)?
            };

            // The reduced cotangent keeps the axes along which the bias is not broadcast, which are the axes of the
            // bias other than its unit axes, so restoring the shape of the bias only inserts or removes unit axes.
            let bias_cotangent = bias_cotangent.reshape(bias_type.shape().clone())?;
            let bias_cotangent = convert(bias_cotangent, bias_cotangent_type.data_type())?;
            cotangents.push(restore_cotangent_type(bias_cotangent, &bias_cotangent_type)?);
        }

        Ok(cotangents)
    })
    .with_name(DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME)
    .with_named_composite_per_structure(
        move |input_structure: &(AttentionResiduals<Placeholder>, Placeholder)| {
            AttentionKey::Backward { configuration, input_structure: input_structure.0.inputs }.to_composite_key()
        },
        move |input_types: (AttentionResiduals<V::Type>, V::Type)| -> Result<Vec<V::Type>, ProgramError> {
            let input_structure = input_types.0.inputs.parameter_structure();
            let input_types = input_types
                .parameters()
                .map(|r#type| r#type.as_array_type().cloned())
                .collect::<Result<Vec<_>, _>>()?;
            let output_types = attention_backward_output_types(
                DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
                configuration,
                input_structure,
                input_types.as_slice(),
            )?;
            Ok(output_types.into_iter().map(Into::into).collect())
        },
    );
    if configuration.dropout().is_some() {
        match configuration.implementation() {
            AttentionImplementation::Fused => {
                // These definitions batch through the rule below, which rejects every call that has a mapped input.
            }
            AttentionImplementation::Automatic | AttentionImplementation::Portable => {
                // Folding would change the shape of the dropout mask and therefore the mask of every batch item,
                // so these definitions keep the structural batching of their bodies.
                return AttentionFunction::Structural(Arc::new(function));
            }
        }
    }

    AttentionFunction::Folding(Arc::new_cyclic(
        |weak: &Weak<
            BatchedComposite<
                V,
                O,
                (AttentionResiduals<CustomRuleTracer<V, O>>, CustomRuleTracer<V, O>),
                Vec<CustomRuleTracer<V, O>>,
            >,
        >| {
            let weak = weak.clone();
            function.with_batching(move |inputs, axes: (AttentionResiduals<BatchAxis>, BatchAxis), _| {
                // Refer to the forward definition for when the function may have been dropped.
                let function = match weak.upgrade() {
                    Some(function) => function,
                    None => match backward_definition::<V, O>(configuration) {
                        AttentionFunction::Folding(function) => function,
                        AttentionFunction::Structural(_) => {
                            unreachable!("structurally batched definitions have no batching rule")
                        }
                    },
                };

                if configuration.dropout().is_some() {
                    // Refer to the forward definition for why a fused call with dropout and a mapped input is
                    // rejected, and for why a call whose inputs are all replicated stages the unbatched definition.
                    if axes.parameters().any(|axis| axis.axis().is_some()) {
                        return Err(ProgramError::UnsupportedOperation {
                            message: format!(
                                "named composite `{DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME}` with dropout \
                                 cannot be batched with the fused implementation, because a batched call cannot use \
                                 the fused kernels or reproduce their dropout mask; use the portable implementation \
                                 or a dropout state",
                            ),
                        });
                    }
                    let outputs = function.call(inputs)?;
                    let output_axes = vec![BatchAxis::replicated(); outputs.len()];
                    return Ok((outputs, output_axes));
                }

                let input_structure = axes.0.inputs.parameter_structure();
                let key = AttentionKey::Backward { configuration, input_structure };
                batch_attention(
                    inputs.into_parameters().collect(),
                    axes.into_parameters().collect(),
                    key.batch_input_roles().as_slice(),
                    |output_count| key.batch_output_roles(output_count),
                    |inputs| {
                        let structure = (
                            AttentionResiduals { inputs: input_structure, output: Placeholder, residual: Placeholder },
                            Placeholder,
                        );
                        function.call(<(AttentionResiduals<_>, CustomRuleTracer<V, O>)>::from_parameters(
                            structure, inputs,
                        )?)
                    },
                )
            })
        },
    ))
}

/// Batches one attention call by folding its mapped axis into attention's batch axis, staging the definition
/// once at the folded boundary through `call`, and restoring the outputs, which are all mapped at axis `0`. The
/// [module documentation](self#batching) describes the steps and their shapes. Normalizing biases and masks to the
/// score shape handles every broadcastable rank and preserves mapped bias and mask inputs, and a bias cotangent is
/// reduced over precisely the axes that its logical bias input broadcasts. Every extent is read from an input with
/// [`DimensionSize`], so a dynamic mapped extent, logical batch, or sequence length never becomes host metadata.
/// Calls whose inputs are all replicated stage the unbatched definition.
fn batch_attention<
    V: Value<Type: ArrayMemberType>
        + Reduce
        + Transpose
        + Reshape
        + Broadcast
        + Reshard
        + DimensionSize<Extent: Clone + Mul>
        + DynamicReshape<<V as DimensionSize>::Extent>
        + DynamicBroadcast<<V as DimensionSize>::Extent>,
    OutputRolesFn: FnOnce(usize) -> Vec<AttentionBatchOutput>,
    CallFn: FnOnce(Vec<V>) -> Result<Vec<V>, ProgramError>,
>(
    inputs: Vec<V>,
    axes: Vec<BatchAxis>,
    input_roles: &[AttentionBatchInput],
    output_roles_fn: OutputRolesFn,
    call_fn: CallFn,
) -> Result<(Vec<V>, Vec<BatchAxis>), ProgramError> {
    check_count!("input", input_roles, inputs.len(), ProgramError);
    check_count!("input", axes, inputs.len(), ProgramError);

    let rank = |value: &V| -> Result<usize, ProgramError> { Ok(value.r#type().as_array_type()?.rank()) };
    let extents = |value: &V, start: usize| {
        (start..rank(value)?)
            .map(|axis| value.dimension_size(axis))
            .collect::<Result<Vec<_>, ProgramError>>()
    };

    let mapped_axes = inputs
        .iter()
        .zip(&axes)
        .map(|(input, axis)| match axis.axis() {
            Some(axis) => {
                Ok(Some(axis.normalize(rank(input)?).map_err(|error| TypeError::invalid(error.to_string()))?))
            }
            None => Ok(None),
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;

    // The mapped extent is the extent of every mapped input along its mapped axis.
    let Some((mapped_input, mapped_axis)) =
        inputs.iter().zip(&mapped_axes).find_map(|(input, axis)| axis.map(|axis| (input, axis)))
    else {
        let outputs = call_fn(inputs)?;
        let output_axes = vec![BatchAxis::replicated(); outputs.len()];
        return Ok((outputs, output_axes));
    };
    let mapped_extent = mapped_input.dimension_size(mapped_axis)?;

    // Align every input to a leading mapped axis, moving mapped axes and broadcasting replicated inputs.
    let aligned = inputs
        .iter()
        .zip(&mapped_axes)
        .map(|(input, mapped_axis)| {
            let rank = rank(input)?;
            match *mapped_axis {
                Some(0) => Ok(input.clone()),
                Some(position) => {
                    let permutation =
                        std::iter::once(position).chain((0..rank).filter(|axis| *axis != position)).collect::<Vec<_>>();
                    input.transpose(permutation)
                }
                None => {
                    let output_extents =
                        std::iter::once(mapped_extent.clone()).chain(extents(input, 0)?).collect::<Vec<_>>();
                    input.dynamic_broadcast(output_extents.as_slice(), (1..=rank).collect::<Vec<_>>().as_slice())
                }
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;

    let query_rank = rank(&aligned[0])? - 1;
    let key_rank = rank(&aligned[1])? - 1;

    // Rank-3 attention has no batch axis, so its mapped axis becomes the batch axis without a product.
    let batch_extent = if query_rank == 4 { Some(aligned[0].dimension_size(1)?) } else { None };
    let folded_batch_extent = match &batch_extent {
        Some(batch_extent) => mapped_extent.mul(batch_extent)?,
        None => mapped_extent.clone(),
    };

    // Biases and masks are normalized to the canonical `[v, B, N, T, S]` score shape (or to `[v, N, T, S]` for rank-3
    // attention), whose extents are obtained only when the call has such inputs (and therefore possibly a bias
    // cotangent output). The folded score shape differs from the normalized one only for rank-4 attention.
    let score_extents = if input_roles.iter().any(|role| matches!(role, AttentionBatchInput::Score)) {
        let query_sequence_extent = aligned[0].dimension_size(query_rank - 2)?;
        let head_extent = aligned[0].dimension_size(query_rank - 1)?;
        let key_sequence_extent = aligned[1].dimension_size(key_rank - 2)?;
        let score_extents = [head_extent, query_sequence_extent, key_sequence_extent];
        let normalized = std::iter::once(mapped_extent.clone())
            .chain(batch_extent.clone())
            .chain(score_extents.iter().cloned())
            .collect::<Vec<_>>();
        let folded = batch_extent
            .as_ref()
            .map(|_| std::iter::once(folded_batch_extent.clone()).chain(score_extents).collect::<Vec<_>>());
        Some((normalized, folded))
    } else {
        None
    };

    // A mesh axis that shards the logical batch axis cannot shard the folded batch axis, whose shards would interleave
    // the batch items of the mapped axis, so the inputs are replicated along the logical batch axis before they are
    // folded. The outputs return to that placement below. The logical batch axis follows the mapped axis in every
    // aligned input that has one (i.e., in a rank-4 query, key, value, or bias or mask, in a residual of rank-4
    // attention, and in sequence lengths), and the placements of those inputs along it agree wherever they are not
    // replicated, as they do for every batch item of the call. An aligned input of lower rank has another axis there
    // (e.g., the sequence axis of a rank-3 key).
    let batch_placement = aligned
        .iter()
        .zip(input_roles)
        .map(|(value, role)| -> Result<Option<ShardingDimension>, ProgramError> {
            let r#type = value.r#type().as_array_type()?.clone();
            let has_batch_axis = batch_extent.is_some()
                && match role {
                    AttentionBatchInput::Tensor | AttentionBatchInput::Score => r#type.rank() == 5,
                    AttentionBatchInput::Residual => r#type.rank() == 4,
                    AttentionBatchInput::Length => r#type.rank() == 2,
                };
            Ok(r#type
                .sharding()
                .filter(|_| has_batch_axis)
                .map(|sharding| (&sharding.dimensions()[1], sharding.mesh()))
                .filter(|(dimension, mesh)| dimension.has_explicit_axis(mesh))
                .map(|(dimension, _)| dimension.clone()))
        })
        .collect::<Result<Vec<_>, ProgramError>>()?
        .into_iter()
        .flatten()
        .next();
    let place_batch_axis = |output: V| -> Result<V, ProgramError> {
        let r#type = output.r#type().as_array_type()?.clone();
        match (&batch_placement, r#type.sharding()) {
            (Some(placement), Some(sharding)) if &sharding.dimensions()[1] != placement => {
                let mut dimensions = sharding.dimensions().to_vec();
                dimensions[1] = placement.clone();
                let placed =
                    sharding.with_dimensions(dimensions).map_err(|error| TypeError::invalid(error.to_string()))?;
                output.reshard(&placed)
            }
            _ => Ok(output),
        }
    };

    // A fold merges the mapped axis and the logical batch axis that follows it, and an unfold splits them again. Both
    // state the placement of their outputs, which is that of their inputs along the dimensions that they keep, because
    // a reshape that changes a dynamic extent cannot infer where those dimensions go. The folded batch axis takes the
    // placement of the mapped axis, since the logical batch axis is replicated first.
    let fold = |value: V, extents: &[<V as DimensionSize>::Extent]| -> Result<V, ProgramError> {
        let value = replicate_dimensions(value, &[1])?;
        let sharding = value
            .r#type()
            .as_array_type()?
            .sharding()
            .map(|sharding| {
                let mut dimensions = sharding.dimensions().to_vec();
                dimensions.remove(1);
                sharding.with_dimensions(dimensions)
            })
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        value.dynamic_reshape_with_output_sharding(extents, sharding)
    };

    let unfold = |value: V, extents: &[<V as DimensionSize>::Extent]| -> Result<V, ProgramError> {
        let sharding = value
            .r#type()
            .as_array_type()?
            .sharding()
            .map(|sharding| {
                let mut dimensions = sharding.dimensions().to_vec();
                dimensions.insert(1, ShardingDimension::Replicated);
                sharding.with_dimensions(dimensions)
            })
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        value.dynamic_reshape_with_output_sharding(extents, sharding)
    };

    // Fold the mapped axis into attention's batch axis and stage the definition at the folded boundary.
    let folded = aligned
        .iter()
        .zip(input_roles)
        .map(|(value, role)| {
            let rank = rank(value)?;
            match role {
                AttentionBatchInput::Tensor if rank == 5 => {
                    // Every folded input uses the one folded batch extent, because a product that is computed again
                    // is another dimension to the validation of the folded call, even when it has the same value.
                    let folded_extents =
                        std::iter::once(folded_batch_extent.clone()).chain(extents(value, 2)?).collect::<Vec<_>>();
                    fold(value.clone(), folded_extents.as_slice())
                }
                AttentionBatchInput::Tensor => Ok(value.clone()),
                AttentionBatchInput::Score => {
                    let (normalized_extents, folded_extents) = score_extents.as_ref().ok_or_else(|| {
                        ProgramError::from(TypeError::invalid(
                            "attention batching produced a bias cotangent without a bias",
                        ))
                    })?;

                    // A rank-4 bias or mask of rank-3 attention has a unit batch axis, which the normalized score
                    // shape of rank-3 attention does not have.
                    let value = if rank == normalized_extents.len() + 1 { value.squeeze(1)? } else { value.clone() };
                    let offset = normalized_extents.len() + 1 - rank.min(normalized_extents.len());
                    let output_axes = std::iter::once(0).chain(offset..normalized_extents.len()).collect::<Vec<_>>();
                    let normalized = value.dynamic_broadcast(normalized_extents.as_slice(), output_axes.as_slice())?;
                    match folded_extents {
                        Some(folded_extents) => fold(normalized, folded_extents.as_slice()),
                        None => Ok(normalized),
                    }
                }
                AttentionBatchInput::Length if rank == 2 => {
                    fold(value.clone(), std::slice::from_ref(&folded_batch_extent))
                }
                AttentionBatchInput::Length => value.dynamic_reshape(std::slice::from_ref(&folded_batch_extent)),
                AttentionBatchInput::Residual if rank == 4 => {
                    let folded_extents =
                        std::iter::once(folded_batch_extent.clone()).chain(extents(value, 2)?).collect::<Vec<_>>();
                    fold(value.clone(), folded_extents.as_slice())
                }
                AttentionBatchInput::Residual => Ok(value.clone()),
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;

    let outputs = call_fn(folded)?;
    let output_roles = output_roles_fn(outputs.len());
    check_count!("output", output_roles, outputs.len(), ProgramError);

    // Restore the mapped axis of every output and reduce bias cotangents over the axes that their biases broadcast.
    // The outputs of the forward definition return to the placement of the logical batch axis, and every cotangent
    // returns to the cotangent type of its aligned input (i.e., to its placement), which reverse mode requires in
    // order to accumulate it.
    let outputs = outputs
        .into_iter()
        .zip(output_roles)
        .map(|(output, role)| match role {
            AttentionBatchOutput::Tensor(input_index) | AttentionBatchOutput::Cotangent(input_index) => {
                let output = if rank(&aligned[input_index])? == 5 {
                    let unfolded_extents = [mapped_extent.clone(), aligned[input_index].dimension_size(1)?]
                        .into_iter()
                        .chain(extents(&output, 1)?)
                        .collect::<Vec<_>>();
                    unfold(output, unfolded_extents.as_slice())?
                } else {
                    output
                };
                match role {
                    AttentionBatchOutput::Cotangent(_) => {
                        let cotangent_type = aligned[input_index].r#type().as_array_type()?.cotangent()?;
                        restore_cotangent_type(output, &cotangent_type)
                    }
                    _ if rank(&aligned[input_index])? == 5 => place_batch_axis(output),
                    _ => Ok(output),
                }
            }
            AttentionBatchOutput::Residual => match &batch_extent {
                Some(batch_extent) => {
                    let unfolded_extents = [mapped_extent.clone(), batch_extent.clone()]
                        .into_iter()
                        .chain(extents(&output, 1)?)
                        .collect::<Vec<_>>();
                    place_batch_axis(unfold(output, unfolded_extents.as_slice())?)
                }
                None => Ok(output),
            },
            AttentionBatchOutput::Score(input_index) => {
                let (normalized_extents, folded_extents) = score_extents.as_ref().ok_or_else(|| {
                    ProgramError::from(TypeError::invalid(
                        "attention batching produced a bias cotangent without a bias",
                    ))
                })?;

                let normalized = match folded_extents {
                    Some(_) => unfold(output, normalized_extents.as_slice())?,
                    None => output,
                };

                // The logical bias is right-aligned to the score axes that follow the mapped axis, without the unit
                // batch axis that a rank-4 bias of rank-3 attention has.
                let bias = &aligned[input_index];
                let bias_type = bias.r#type().as_array_type()?.clone();
                let score_axis_count = normalized_extents.len() - 1;
                let bias_dimensions = &bias_type.shape().dimensions()[1..];
                let bias_dimensions = &bias_dimensions[bias_dimensions.len().saturating_sub(score_axis_count)..];
                let offset = score_axis_count - bias_dimensions.len();
                let reduction_axes = (0..score_axis_count)
                    .filter(|&axis| axis < offset || matches!(bias_dimensions[axis - offset], Dimension::Static(1)))
                    .map(|axis| axis + 1)
                    .collect::<Vec<_>>();
                let reduced = normalized.reduce(reduction_axes.as_slice(), ReductionKind::Sum)?;
                let cotangent = reduced.dynamic_reshape(extents(bias, 0)?.as_slice())?;
                restore_cotangent_type(cotangent, &bias_type.cotangent()?)
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let output_axes = vec![BatchAxis::from_position(0); outputs.len()];
    Ok((outputs, output_axes))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionVariable, Layout,
        LogicalMesh, Memory, MeshAxis, MeshAxisType, StridedLayout,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchingError, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationError, differentiate_at};
    use crate::macros::check_gradient;
    use crate::operations::collectives::parallel_vary::PARALLEL_VARY_OPERATION_NAME;
    use crate::operations::composites::operations::composite::COMPOSITE_OPERATION_NAME;
    use crate::operations::dimensions::dimension_size::DIMENSION_SIZE_OPERATION_NAME;
    use crate::operations::manipulation::slicing::Slice;
    use crate::operations::random::DROPOUT_OPERATION_NAME;
    use crate::programs::Type;
    use crate::tests::hash_of;
    use crate::tracing::Trace;

    use super::*;

    /// Eager array context in which the tests stage and evaluate attention.
    type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;

    /// Eager array IR context in which the tests stage and evaluate attention with first-class extents.
    type ArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Returns an `f32` array with the provided dimensions and elements.
    fn f32_array(dimensions: &[usize], elements: &[f32]) -> Array {
        Array::from_elements(ArrayType::new_static(DataType::F32, dimensions.to_vec()), elements).unwrap()
    }

    /// Returns an `f16` array with the provided dimensions and elements.
    fn f16_array(dimensions: &[usize], elements: &[f32]) -> Array {
        let elements = elements.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
        Array::from_elements(ArrayType::new_static(DataType::F16, dimensions.to_vec()), elements.as_slice()).unwrap()
    }

    /// Returns a `bf16` array with the provided dimensions and elements.
    fn bf16_array(dimensions: &[usize], elements: &[f32]) -> Array {
        let elements = elements.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
        Array::from_elements(ArrayType::new_static(DataType::BF16, dimensions.to_vec()), elements.as_slice()).unwrap()
    }

    /// Returns an `f64` array with the provided dimensions and elements.
    fn f64_array(dimensions: &[usize], elements: &[f64]) -> Array {
        Array::from_elements(ArrayType::new_static(DataType::F64, dimensions.to_vec()), elements).unwrap()
    }

    /// Returns the rendering of the sharding of an array with the provided rank that is replicated over a mesh with one
    /// manual axis `x` of size 2, over which the array optionally varies.
    fn rendered_sharding(rank: usize, varying: bool) -> String {
        let dimensions = vec!["{}"; rank].join(", ");
        let varying = if varying { ", varying_manual={'x'}" } else { "" };
        format!("[sharding={{mesh<['x'=2:manual]>, [{dimensions}]{varying}}}]")
    }

    /// Returns the attended output and the residual of one eager attention call with the provided configuration.
    fn attend(inputs: AttentionInputs<Array>, configuration: AttentionConfiguration) -> (Array, Array) {
        dot_product_attention::<ArrayContext>(configuration).call_with_residual(inputs).unwrap()
    }

    /// Returns the sum of one differentiable attention output at unit score scale, together with its gradient.
    fn attention_sum_gradient(inputs: AttentionInputs<Array>) -> (Array, AttentionInputs<Array>) {
        let function =
            differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_scale(1.0));
        differentiate_at(inputs)
            .value_and_gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap()
    }

    /// Returns the value of a cotangent that is not a structural zero.
    fn cotangent_value<V: Typed>(cotangent: MaybeZero<V>) -> V {
        match cotangent {
            MaybeZero::Value(cotangent) => cotangent,
            MaybeZero::Zero(_) => panic!("expected a cotangent value but got a structural zero"),
        }
    }

    #[test]
    fn test_attention_key_from_composite_key() {
        // Keys of every definition and implementation round trip, with and without their optional attributes.
        let configuration = AttentionConfiguration::new()
            .with_scale(0.125)
            .with_causal(true)
            .with_local_window((2, 0))
            .with_dropout(0.25);
        let input_structure = AttentionInputs {
            bias: Some(Placeholder),
            query_sequence_lengths: Some(Placeholder),
            dropout_state: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        for implementation in
            [AttentionImplementation::Automatic, AttentionImplementation::Portable, AttentionImplementation::Fused]
        {
            let default_configuration = AttentionConfiguration::new().with_implementation(implementation);
            let default_input_structure = AttentionInputs::new(Placeholder, Placeholder, Placeholder);
            let configuration = configuration.with_implementation(implementation);
            for key in [
                AttentionKey::Forward { configuration, input_structure, return_residual: true },
                AttentionKey::Forward {
                    configuration: default_configuration,
                    input_structure: default_input_structure,
                    return_residual: false,
                },
                AttentionKey::Backward { configuration, input_structure },
                AttentionKey::Backward {
                    configuration: default_configuration,
                    input_structure: default_input_structure,
                },
            ] {
                assert_eq!(AttentionKey::from_composite_key(&key.to_composite_key()), Ok(Some(key)));
            }
        }

        // Keys of other composites are not attention keys.
        assert_eq!(AttentionKey::from_composite_key(&CompositeKey::new("ryft.other", 1)), Ok(None));

        // Attention keys with another version or with attributes that differ from the encoding are rejected.
        let wrong_version = CompositeKey::new(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, 2);
        assert_eq!(
            AttentionKey::from_composite_key(&wrong_version),
            Err(TypeError::invalid(format!(
                "named composite `{wrong_version}` has attention definition version 2, but only version \
                 {DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION} is supported",
            ))),
        );
        let forward_key = AttentionKey::Forward {
            configuration: AttentionConfiguration::new(),
            input_structure: AttentionInputs::new(Placeholder, Placeholder, Placeholder),
            return_residual: false,
        }
        .to_composite_key();
        let extended = forward_key.clone().with_attribute("extra", true);
        assert_eq!(
            AttentionKey::from_composite_key(&extended),
            Err(TypeError::invalid(format!("named composite `{extended}` is not a valid attention definition"))),
        );

        // An unknown implementation, a local window with a negative radius, and a local window that is not a pair of
        // radii are all malformed.
        for (malformed_name, malformed_value) in [
            ("implementation", CompositeAttribute::Integer(3)),
            ("local_window", CompositeAttribute::from(Some((-1i64, 0i64)))),
            ("local_window", CompositeAttribute::from(Some(1i64))),
        ] {
            let malformed = forward_key.attributes().iter().fold(
                CompositeKey::new(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION),
                |key, (name, value)| {
                    let value = if name == malformed_name { malformed_value.clone() } else { value.clone() };
                    key.with_attribute(name.clone(), value)
                },
            );
            assert_eq!(
                AttentionKey::from_composite_key(&malformed),
                Err(TypeError::invalid(format!("named composite `{malformed}` is not a valid attention definition"))),
            );
        }
        let missing =
            CompositeKey::new(DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION)
                .with_attribute("causal", false);
        assert_eq!(
            AttentionKey::from_composite_key(&missing),
            Err(TypeError::invalid(format!("named composite `{missing}` is not a valid attention definition"))),
        );

        // Only forward keys have a residual request, so a forward key without one and a backward key with one are
        // rejected.
        let without_residual = forward_key.attributes().iter().filter(|(name, _)| name != "residual").fold(
            CompositeKey::new(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION),
            |key, (name, value)| key.with_attribute(name.clone(), value.clone()),
        );
        assert_eq!(
            AttentionKey::from_composite_key(&without_residual),
            Err(TypeError::invalid(format!(
                "named composite `{without_residual}` is not a valid attention definition",
            ))),
        );
        let with_residual = forward_key.attributes().iter().fold(
            CompositeKey::new(DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION),
            |key, (name, value)| key.with_attribute(name.clone(), value.clone()),
        );
        assert_eq!(
            AttentionKey::from_composite_key(&with_residual),
            Err(TypeError::invalid(format!("named composite `{with_residual}` is not a valid attention definition"))),
        );
    }

    #[test]
    fn test_attention_key_configuration() {
        let configuration = AttentionConfiguration::new().with_causal(true).with_dropout(0.25);
        let input_structure = AttentionInputs::new(Placeholder, Placeholder, Placeholder);
        assert_eq!(
            AttentionKey::Forward { configuration, input_structure, return_residual: true }.configuration(),
            configuration,
        );
        assert_eq!(AttentionKey::Backward { configuration, input_structure }.configuration(), configuration);
    }

    #[test]
    fn test_attention_key_input_structure() {
        let configuration = AttentionConfiguration::new();
        let input_structure = AttentionInputs {
            bias: Some(Placeholder),
            query_sequence_lengths: Some(Placeholder),
            dropout_state: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        assert_eq!(
            AttentionKey::Forward { configuration, input_structure, return_residual: true }.input_structure(),
            input_structure,
        );
        assert_eq!(AttentionKey::Backward { configuration, input_structure }.input_structure(), input_structure);
    }

    #[test]
    fn test_attention_key_to_composite_key() {
        let configuration = AttentionConfiguration::new()
            .with_scale(0.125)
            .with_causal(true)
            .with_local_window((2, 0))
            .with_implementation(AttentionImplementation::Fused)
            .with_dropout(0.25);
        let input_structure = AttentionInputs {
            bias: Some(Placeholder),
            query_sequence_lengths: Some(Placeholder),
            dropout_state: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };

        // Forward keys encode every configuration attribute and their residual request.
        let key = AttentionKey::Forward { configuration, input_structure, return_residual: true }.to_composite_key();
        assert_eq!(key.name(), DOT_PRODUCT_ATTENTION_COMPOSITE_NAME);
        assert_eq!(key.version(), DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION);
        assert_eq!(
            key.to_string(),
            "ryft.dot_product_attention.v1 {scale = some(0.125), causal = true, local_window = some([2, 0]), \
             implementation = 2, dropout_rate = some(0.25), residual = true, bias = true, mask = false, \
             query_sequence_lengths = true, key_value_sequence_lengths = false, dropout_state = true}",
        );

        // Forward keys with different residual requests identify different definitions.
        assert_ne!(
            key,
            AttentionKey::Forward { configuration, input_structure, return_residual: false }.to_composite_key(),
        );

        // Backward keys have no residual request, so they omit its attribute.
        let key = AttentionKey::Backward { configuration, input_structure }.to_composite_key();
        assert_eq!(key.name(), DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME);
        assert_eq!(key.version(), DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION);
        assert_eq!(
            key.to_string(),
            "ryft.dot_product_attention_backward.v1 {scale = some(0.125), causal = true, local_window = \
             some([2, 0]), implementation = 2, dropout_rate = some(0.25), bias = true, mask = false, \
             query_sequence_lengths = true, key_value_sequence_lengths = false, dropout_state = true}",
        );
    }

    #[test]
    fn test_attention_implementation_display() {
        assert_eq!(AttentionImplementation::Automatic.to_string(), "automatic");
        assert_eq!(AttentionImplementation::Portable.to_string(), "portable");
        assert_eq!(AttentionImplementation::Fused.to_string(), "fused");
    }

    #[test]
    fn test_attention_configuration() {
        let configuration = AttentionConfiguration::new();
        assert_eq!(configuration.scale(), None);
        assert!(!configuration.causal());
        assert_eq!(configuration.local_window(), None);
        assert_eq!(configuration.implementation(), AttentionImplementation::Automatic);
        assert_eq!(configuration.dropout(), None);

        let configuration = configuration
            .with_scale(0.25)
            .with_causal(true)
            .with_local_window((2, 1))
            .with_implementation(AttentionImplementation::Fused)
            .with_dropout(0.1);
        assert_eq!(configuration.scale(), Some(0.25));
        assert!(configuration.causal());
        assert_eq!(configuration.local_window(), Some((2, 0)));
        assert_eq!(configuration.implementation(), AttentionImplementation::Fused);
        assert_eq!(configuration.dropout(), Some(0.1));

        // Causal masking hides every position to the right, so the right radius is reported only without it.
        assert_eq!(configuration.with_causal(false).local_window(), Some((2, 1)));

        // Absent values restore the inferred scale and disable the local window and dropout, and a symmetric local
        // window uses its radius on both sides.
        assert_eq!(configuration.with_scale(None).scale(), None);
        assert_eq!(configuration.with_local_window(None).local_window(), None);
        assert_eq!(configuration.with_causal(false).with_symmetric_local_window(3).local_window(), Some((3, 3)));
        assert_eq!(configuration.with_symmetric_local_window(None).local_window(), None);
        assert_eq!(configuration.with_dropout(None).dropout(), None);
    }

    #[test]
    fn test_attention_configuration_scale_for() {
        // An explicit scale applies to every head dimension, and the inferred scale is `1 / sqrt(head_dimension)`.
        assert_eq!(AttentionConfiguration::new().with_scale(0.25).scale_for(16), 0.25);
        assert_eq!(AttentionConfiguration::new().with_scale(0.25).scale_for(0), 0.25);
        assert_eq!(AttentionConfiguration::new().scale_for(16), 0.25);
        assert_eq!(AttentionConfiguration::new().scale_for(1), 1.0);

        // The scores of an empty head dimension are zero for every scale, so its inferred scale is one instead of
        // the infinite `1 / sqrt(0)`.
        assert_eq!(AttentionConfiguration::new().scale_for(0), 1.0);
    }

    #[test]
    fn test_attention_configuration_equality_and_hashing() {
        // Equal configurations are equal and hash identically, so they work as map keys, and configurations that
        // differ in causal masking or in their dropout rate are distinct.
        let configuration = AttentionConfiguration::new().with_scale(0.25).with_causal(true).with_dropout(0.1);
        let same_configuration = AttentionConfiguration::new().with_scale(0.25).with_causal(true).with_dropout(0.1);
        assert_eq!(configuration, same_configuration);
        assert_eq!(hash_of(&configuration), hash_of(&same_configuration));
        let configurations = HashMap::from([(configuration, "configuration")]);
        assert_eq!(configurations.get(&same_configuration), Some(&"configuration"));
        assert_eq!(configurations.get(&configuration.with_dropout(0.2)), None);
        assert_ne!(configuration, configuration.with_causal(false));
        assert_ne!(configuration, configuration.with_dropout(0.2));
        assert_ne!(configuration, configuration.with_dropout(None));

        // The right radius of the local window has no effect with causal masking, so it does not distinguish
        // configurations or their keys there, while it does without causal masking.
        let left = configuration.with_local_window((2, 0));
        let both = configuration.with_local_window((2, 3));
        assert_eq!(left, both);
        assert_eq!(hash_of(&left), hash_of(&both));
        let input_structure = AttentionInputs::new(Placeholder, Placeholder, Placeholder);
        assert_eq!(
            AttentionKey::Backward { configuration: left, input_structure }.to_composite_key(),
            AttentionKey::Backward { configuration: both, input_structure }.to_composite_key(),
        );
        assert_ne!(left.with_causal(false), both.with_causal(false));

        // The scale and the dropout rate compare bitwise, so signed zeros are distinct even though `-0.0 == 0.0`.
        assert_ne!(AttentionConfiguration::new().with_scale(-0.0), AttentionConfiguration::new().with_scale(0.0));
        assert_ne!(AttentionConfiguration::new().with_scale(0.0), AttentionConfiguration::new());
        assert_ne!(AttentionConfiguration::new().with_dropout(-0.0), AttentionConfiguration::new().with_dropout(0.0));
    }

    #[test]
    fn test_attention_inputs_new() {
        let types = (0..3).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
        assert_eq!(
            AttentionInputs::new(types[0].clone(), types[1].clone(), types[2].clone()),
            AttentionInputs {
                query: types[0].clone(),
                key: types[1].clone(),
                value: types[2].clone(),
                bias: None,
                mask: None,
                query_sequence_lengths: None,
                key_value_sequence_lengths: None,
                dropout_state: None,
            },
        );
    }

    #[test]
    fn test_attention_inputs_parameters() {
        // The parameters of the inputs are their types in canonical order, in which the present optional inputs
        // follow the query, key, and value, and their structure records which optional inputs are present.
        let types = (0..8).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
        let inputs = AttentionInputs::new(types[0].clone(), types[1].clone(), types[2].clone());
        assert_eq!(inputs.parameter_structure(), AttentionInputs::new(Placeholder, Placeholder, Placeholder));
        assert_eq!(inputs.parameter_count(), 3);
        assert_eq!(inputs.clone().into_parameters().collect::<Vec<_>>(), types[..3].to_vec());
        assert_eq!(
            AttentionInputs::from_parameters(
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                types[..3].iter().cloned(),
            ),
            Ok(inputs.clone()),
        );

        let inputs = AttentionInputs {
            mask: Some(types[3].clone()),
            key_value_sequence_lengths: Some(types[4].clone()),
            ..inputs
        };
        let structure = AttentionInputs {
            mask: Some(Placeholder),
            key_value_sequence_lengths: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        assert_eq!(inputs.parameter_structure(), structure);
        assert_eq!(inputs.parameter_count(), 5);
        assert_eq!(inputs.clone().into_parameters().collect::<Vec<_>>(), types[..5].to_vec());
        assert_eq!(AttentionInputs::from_parameters(structure, types[..5].iter().cloned()), Ok(inputs));

        let inputs = AttentionInputs {
            query: types[0].clone(),
            key: types[1].clone(),
            value: types[2].clone(),
            bias: Some(types[3].clone()),
            mask: Some(types[4].clone()),
            query_sequence_lengths: Some(types[5].clone()),
            key_value_sequence_lengths: Some(types[6].clone()),
            dropout_state: Some(types[7].clone()),
        };
        let structure = AttentionInputs {
            query: Placeholder,
            key: Placeholder,
            value: Placeholder,
            bias: Some(Placeholder),
            mask: Some(Placeholder),
            query_sequence_lengths: Some(Placeholder),
            key_value_sequence_lengths: Some(Placeholder),
            dropout_state: Some(Placeholder),
        };
        assert_eq!(inputs.parameter_structure(), structure);
        assert_eq!(inputs.parameter_count(), 8);
        assert_eq!(inputs.clone().into_parameters().collect::<Vec<_>>(), types);
        assert_eq!(AttentionInputs::from_parameters(structure, types.iter().cloned()), Ok(inputs));
    }

    #[test]
    fn test_attention_residuals_parameters() {
        // The inputs of the backward definition pair the residuals with the output cotangent, which appends the
        // forward output, its residual, and the output cotangent to the forward inputs.
        let types = (0..7).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
        let inputs = (
            AttentionResiduals {
                inputs: AttentionInputs {
                    mask: Some(types[3].clone()),
                    ..AttentionInputs::new(types[0].clone(), types[1].clone(), types[2].clone())
                },
                output: types[4].clone(),
                residual: types[5].clone(),
            },
            types[6].clone(),
        );
        let structure = (
            AttentionResiduals {
                inputs: AttentionInputs {
                    mask: Some(Placeholder),
                    ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
                },
                output: Placeholder,
                residual: Placeholder,
            },
            Placeholder,
        );
        assert_eq!(inputs.parameter_structure(), structure);
        assert_eq!(inputs.parameter_count(), 7);
        assert_eq!(inputs.clone().into_parameters().collect::<Vec<_>>(), types);
        assert_eq!(<(AttentionResiduals<_>, _)>::from_parameters(structure, types.iter().cloned()), Ok(inputs));
    }

    #[test]
    fn test_dot_product_attention() {
        let r#type = ArrayType::new_static(DataType::F32, [1, 2, 1, 1]);
        let configuration = AttentionConfiguration::new().with_causal(true);
        let function = dot_product_attention::<ArrayContext>(configuration);
        assert_eq!(function.configuration(), configuration);

        // Calls with equal input types share one traced body, which the program imports once.
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| {
                let first = function.call(inputs.clone())?;
                function.call(AttentionInputs::new(first, inputs.key, inputs.value))
            },
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[1, 2, 1, 1], %1:f32[1, 2, 1, 1], %2:f32[1, 2, 1, 1] .
                let %3:f32[1, 2, 1, 1] = composite [
                    key=\"ryft.dot_product_attention.v1 {scale = none, causal = true, local_window = none, \
                        implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, \
                        query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                ] %0 %1 %2 [
                    primal=^0={
                        lambda %0:f32[1, 2, 1, 1], %1:f32[1, 2, 1, 1], %2:f32[1, 2, 1, 1] .
                        let %3:f32[1, 1, 2, 2] = dot [
                            dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                rhs_batching=[0, 2]),
                        ] %0 %1
                            %4:f32[] = constant [value=1.0]
                            %5:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[]] %4
                            %6:f32[1, 1, 2, 2] = mul %3 %5
                            %7:i32[1, 1, 2, 2] = iota [type=i32[1, 1, 2, 2], dimension=3]
                            %8:i32[1, 1, 2, 2] = iota [type=i32[1, 1, 2, 2], dimension=2]
                            %9:bool[1, 1, 2, 2] = compare [direction=LessThanOrEqual] %7 %8
                            %10:bool[1, 1, 2] = reduce [kind=any, axes=[3]] %9
                            %11:bool[1, 1, 2, 2] = broadcast [output_type=bool[1, 1, 2, 2], output_axes=[0, 1, 2]] %10
                            %12:f32[] = constant [value=-inf]
                            %13:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[]] %12
                            %14:f32[] = constant [value=-2.3819763e38]
                            %15:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[]] %14
                            %16:f32[1, 1, 2, 2] = select %11 %13 %15
                            %17:f32[1, 1, 2, 2] = select %9 %6 %16
                            %18:f32[1, 1, 2] = reduce [kind=max, axes=[3]] %17
                            %19:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[0, 1, 2]] %18
                            %20:f32[1, 1, 2, 2] = sub %17 %19
                            %21:f32[1, 1, 2, 2] = exp %20
                            %22:f32[1, 1, 2] = reduce [kind=sum, axes=[3]] %21
                            %23:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[0, 1, 2]] %22
                            %24:f32[1, 1, 2, 2] = div %21 %23
                            %25:f32[1, 1, 2, 1] = dot [
                                dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                    rhs_batching=[0, 2]),
                            ] %24 %2
                            %26:f32[1, 2, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %25
                        in (%26)
                    },
                ]
                    %4:f32[1, 2, 1, 1] = composite [
                        key=\"ryft.dot_product_attention.v1 {scale = none, causal = true, local_window = none, \
                            implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, \
                            query_sequence_lengths = false, key_value_sequence_lengths = false, \
                            dropout_state = false}\",
                    ] %3 %1 %2 [
                        primal=^0,
                    ]
                in (%4)
            "}
            .trim_end(),
        );

        // One function stages both forward definitions, which agree on the attended output.
        let ones = f32_array(&[1, 2, 1, 1], &[1.0, 1.0]);
        let inputs = AttentionInputs::new(ones.clone(), ones, f32_array(&[1, 2, 1, 1], &[3.0, 5.0]));
        let output = function.call(inputs.clone()).unwrap();
        assert_eq!(output, f32_array(&[1, 2, 1, 1], &[3.0, 4.0]));
        assert_eq!(
            function.call_with_residual(inputs),
            Ok((output, f32_array(&[1, 2, 1], &[1.0, 1.0 + std::f32::consts::LN_2]))),
        );

        // One function serves every input structure under the key of that structure, including structures whose
        // flattened input types are equal (i.e., the two kinds of sequence lengths).
        let r#type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]);
        let (_, program) = ArrayContext::trace(
            |(inputs, lengths): (AttentionInputs<_>, _)| {
                Ok((
                    function
                        .call(AttentionInputs { query_sequence_lengths: Some(lengths.clone()), ..inputs.clone() })?,
                    function.call(AttentionInputs { key_value_sequence_lengths: Some(lengths), ..inputs })?,
                ))
            },
            (AttentionInputs::new(r#type.clone(), r#type.clone(), r#type), ArrayType::new_static(DataType::I32, [1])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1], %3:i32[1] .
                let %4:f32[1, 1, 1, 1] = composite [
                    key=\"ryft.dot_product_attention.v1 {scale = none, causal = true, local_window = none, \
                        implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, \
                        query_sequence_lengths = true, key_value_sequence_lengths = false, dropout_state = false}\",
                ] %0 %1 %2 %3 [
                    primal={
                        lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1], %3:i32[1] .
                        let %4:f32[1, 1, 1, 1] = dot [
                            dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                rhs_batching=[0, 2]),
                        ] %0 %1
                            %5:f32[] = constant [value=1.0]
                            %6:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %5
                            %7:f32[1, 1, 1, 1] = mul %4 %6
                            %8:i32[1, 1, 1, 1] = iota [type=i32[1, 1, 1, 1], dimension=3]
                            %9:i32[1, 1, 1, 1] = iota [type=i32[1, 1, 1, 1], dimension=2]
                            %10:bool[1, 1, 1, 1] = compare [direction=LessThanOrEqual] %8 %9
                            %11:i32[1, 1, 1, 1] = iota [type=i32[1, 1, 1, 1], dimension=2]
                            %12:i32[1, 1, 1, 1] = broadcast [output_type=i32[1, 1, 1, 1], output_axes=[0]] %3
                            %13:bool[1, 1, 1, 1] = compare [direction=LessThan] %11 %12
                            %14:bool[1, 1, 1, 1] = and %10 %13
                            %15:bool[1, 1, 1] = reduce [kind=any, axes=[3]] %14
                            %16:bool[1, 1, 1, 1] = broadcast [output_type=bool[1, 1, 1, 1], output_axes=[0, 1, 2]] %15
                            %17:f32[] = constant [value=-inf]
                            %18:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %17
                            %19:f32[] = constant [value=-2.3819763e38]
                            %20:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %19
                            %21:f32[1, 1, 1, 1] = select %16 %18 %20
                            %22:f32[1, 1, 1, 1] = select %14 %7 %21
                            %23:f32[1, 1, 1] = reduce [kind=max, axes=[3]] %22
                            %24:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %23
                            %25:f32[1, 1, 1, 1] = sub %22 %24
                            %26:f32[1, 1, 1, 1] = exp %25
                            %27:f32[1, 1, 1] = reduce [kind=sum, axes=[3]] %26
                            %28:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %27
                            %29:f32[1, 1, 1, 1] = div %26 %28
                            %30:f32[1, 1, 1, 1] = dot [
                                dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                    rhs_batching=[0, 2]),
                            ] %29 %2
                            %31:f32[1, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %30
                            %32:i32[1, 1, 1, 1] = iota [type=i32[1, 1, 1, 1], dimension=1]
                            %33:i32[1, 1, 1, 1] = broadcast [output_type=i32[1, 1, 1, 1], output_axes=[0]] %3
                            %34:bool[1, 1, 1, 1] = compare [direction=LessThan] %32 %33
                            %35:f32[] = constant [value=0.0]
                            %36:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %35
                            %37:f32[1, 1, 1, 1] = select %34 %31 %36
                        in (%37)
                    },
                ]
                    %5:f32[1, 1, 1, 1] = composite [
                        key=\"ryft.dot_product_attention.v1 {scale = none, causal = true, local_window = none, \
                            implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, \
                            query_sequence_lengths = false, key_value_sequence_lengths = true, dropout_state = false}\",
                    ] %0 %1 %2 %3 [
                        primal={
                            lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1], %3:i32[1] .
                            let %4:f32[1, 1, 1, 1] = dot [
                                dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                    rhs_batching=[0, 2]),
                            ] %0 %1
                                %5:f32[] = constant [value=1.0]
                                %6:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %5
                                %7:f32[1, 1, 1, 1] = mul %4 %6
                                %8:i32[1, 1, 1, 1] = iota [type=i32[1, 1, 1, 1], dimension=3]
                                %9:i32[1, 1, 1, 1] = iota [type=i32[1, 1, 1, 1], dimension=2]
                                %10:bool[1, 1, 1, 1] = compare [direction=LessThanOrEqual] %8 %9
                                %11:i32[1, 1, 1, 1] = broadcast [output_type=i32[1, 1, 1, 1], output_axes=[0]] %3
                                %12:bool[1, 1, 1, 1] = compare [direction=LessThan] %8 %11
                                %13:bool[1, 1, 1, 1] = and %10 %12
                                %14:bool[1, 1, 1] = reduce [kind=any, axes=[3]] %13
                                %15:bool[1, 1, 1, 1] = \
                                    broadcast [output_type=bool[1, 1, 1, 1], output_axes=[0, 1, 2]] %14
                                %16:f32[] = constant [value=-inf]
                                %17:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %16
                                %18:f32[] = constant [value=-2.3819763e38]
                                %19:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %18
                                %20:f32[1, 1, 1, 1] = select %15 %17 %19
                                %21:f32[1, 1, 1, 1] = select %13 %7 %20
                                %22:f32[1, 1, 1] = reduce [kind=max, axes=[3]] %21
                                %23:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %22
                                %24:f32[1, 1, 1, 1] = sub %21 %23
                                %25:f32[1, 1, 1, 1] = exp %24
                                %26:f32[1, 1, 1] = reduce [kind=sum, axes=[3]] %25
                                %27:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %26
                                %28:f32[1, 1, 1, 1] = div %25 %27
                                %29:f32[1, 1, 1, 1] = dot [
                                    dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                        rhs_batching=[0, 2]),
                                ] %28 %2
                                %30:f32[1, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %29
                            in (%30)
                        },
                    ]
                in (%4, %5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_product_attention_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant = ArrayType::new_static(DataType::F32, [1, 1, 1, 1])
            .with_sharding(Sharding::replicated(mesh.clone(), 4))
            .unwrap();
        let varying = invariant
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 4).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let lengths = ArrayType::new_static(DataType::I32, [1])
            .with_sharding(Sharding::replicated(mesh.clone(), 1))
            .unwrap();
        let mask = ArrayType::new_static(DataType::Boolean, [1, 1, 1, 1])
            .with_sharding(Sharding::replicated(mesh.clone(), 4))
            .unwrap();
        let residual = ArrayType::new_static(DataType::F32, [1, 1, 1])
            .with_sharding(Sharding::replicated(mesh.clone(), 3).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());

        // The function aligns the manual variation of its inputs before the call, so the invariant inputs become
        // varying and the outputs vary like the query.
        let (output_types, program) = DomainTracingContext::<ArrayContext>::trace_with_named_axes(
            |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
            AttentionInputs {
                query: varying.clone(),
                key: invariant.clone(),
                value: invariant.clone(),
                bias: Some(invariant),
                mask: Some(mask),
                query_sequence_lengths: Some(lengths.clone()),
                key_value_sequence_lengths: Some(lengths),
                dropout_state: None,
            },
            vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: true })],
        )
        .unwrap();
        assert_eq!(output_types, (varying, residual));
        let varying_rank_4 = rendered_sharding(4, true);
        let varying_rank_3 = rendered_sharding(3, true);
        let varying_rank_1 = rendered_sharding(1, true);
        let varying_rank_0 = rendered_sharding(0, true);
        let invariant_rank_4 = rendered_sharding(4, false);
        let invariant_rank_1 = rendered_sharding(1, false);
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda \
                    %0:f32[1, 1, 1, 1]{varying_rank_4}, \
                    %1:f32[1, 1, 1, 1]{invariant_rank_4}, \
                    %2:f32[1, 1, 1, 1]{invariant_rank_4}, \
                    %3:f32[1, 1, 1, 1]{invariant_rank_4}, \
                    %4:bool[1, 1, 1, 1]{invariant_rank_4}, \
                    %5:i32[1]{invariant_rank_1}, \
                    %6:i32[1]{invariant_rank_1} .
                let %7:f32[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %1
                    %8:f32[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %2
                    %9:f32[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %3
                    %10:bool[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %4
                    %11:i32[1]{varying_rank_1} = parallel_vary [axis_name=\"x\"] %5
                    %12:i32[1]{varying_rank_1} = parallel_vary [axis_name=\"x\"] %6
                    %13:f32[1, 1, 1, 1]{varying_rank_4}, %14:f32[1, 1, 1]{varying_rank_3} = composite [
                        key=\"ryft.dot_product_attention.v1 {{scale = none, causal = false, local_window = none, \
                            implementation = 0, dropout_rate = none, residual = true, bias = true, mask = true, \
                            query_sequence_lengths = true, key_value_sequence_lengths = true, dropout_state = false}}\",
                    ] %0 %7 %8 %9 %10 %11 %12 [
                        primal={{
                            lambda \
                                %0:f32[1, 1, 1, 1]{varying_rank_4}, \
                                %1:f32[1, 1, 1, 1]{varying_rank_4}, \
                                %2:f32[1, 1, 1, 1]{varying_rank_4}, \
                                %3:f32[1, 1, 1, 1]{varying_rank_4}, \
                                %4:bool[1, 1, 1, 1]{varying_rank_4}, \
                                %5:i32[1]{varying_rank_1}, \
                                %6:i32[1]{varying_rank_1} .
                            let %7:f32[1, 1, 1, 1]{varying_rank_4} = dot [
                                dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                    rhs_batching=[0, 2]),
                            ] %0 %1
                                %8:f32[]{varying_rank_0} = constant [value=1.0]
                                %9:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[],
                                ] %8
                                %10:f32[1, 1, 1, 1]{varying_rank_4} = mul %7 %9
                                %11:f32[1, 1, 1, 1]{varying_rank_4} = add %10 %3
                                %12:i32[1, 1, 1, 1]{varying_rank_4} = iota [
                                    type=i32[1, 1, 1, 1]{varying_rank_4},
                                    dimension=3,
                                ]
                                %13:i32[1, 1, 1, 1]{varying_rank_4} = iota [
                                    type=i32[1, 1, 1, 1]{varying_rank_4},
                                    dimension=2,
                                ]
                                %14:i32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=i32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0],
                                ] %5
                                %15:bool[1, 1, 1, 1]{varying_rank_4} = compare [direction=LessThan] %13 %14
                                %16:bool[1, 1, 1, 1]{varying_rank_4} = and %4 %15
                                %17:i32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=i32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0],
                                ] %6
                                %18:bool[1, 1, 1, 1]{varying_rank_4} = compare [direction=LessThan] %12 %17
                                %19:bool[1, 1, 1, 1]{varying_rank_4} = and %16 %18
                                %20:bool[1, 1, 1]{varying_rank_3} = reduce [kind=any, axes=[3]] %19
                                %21:bool[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=bool[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0, 1, 2],
                                ] %20
                                %22:f32[]{varying_rank_0} = constant [value=-inf]
                                %23:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[],
                                ] %22
                                %24:f32[]{varying_rank_0} = constant [value=-2.3819763e38]
                                %25:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[],
                                ] %24
                                %26:f32[1, 1, 1, 1]{varying_rank_4} = select %21 %23 %25
                                %27:f32[1, 1, 1, 1]{varying_rank_4} = select %19 %11 %26
                                %28:f32[1, 1, 1]{varying_rank_3} = reduce [kind=max, axes=[3]] %27
                                %29:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0, 1, 2],
                                ] %28
                                %30:f32[1, 1, 1, 1]{varying_rank_4} = sub %27 %29
                                %31:f32[1, 1, 1, 1]{varying_rank_4} = exp %30
                                %32:f32[1, 1, 1]{varying_rank_3} = reduce [kind=sum, axes=[3]] %31
                                %33:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0, 1, 2],
                                ] %32
                                %34:f32[1, 1, 1, 1]{varying_rank_4} = div %31 %33
                                %35:f32[1, 1, 1, 1]{varying_rank_4} = dot [
                                    dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                        rhs_batching=[0, 2]),
                                ] %34 %2
                                %36:f32[1, 1, 1, 1]{varying_rank_4} = transpose [permutation=[0, 2, 1, 3]] %35
                                %37:i32[1, 1, 1, 1]{varying_rank_4} = iota [
                                    type=i32[1, 1, 1, 1]{varying_rank_4},
                                    dimension=1,
                                ]
                                %38:i32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=i32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0],
                                ] %5
                                %39:bool[1, 1, 1, 1]{varying_rank_4} = compare [direction=LessThan] %37 %38
                                %40:f32[]{varying_rank_0} = constant [value=0.0]
                                %41:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[],
                                ] %40
                                %42:f32[1, 1, 1, 1]{varying_rank_4} = select %39 %36 %41
                                %43:f32[1, 1, 1]{varying_rank_3} = log %32
                                %44:f32[1, 1, 1]{varying_rank_3} = add %28 %43
                                %45:f32[1, 1, 1]{varying_rank_3} = transpose [permutation=[0, 2, 1]] %44
                                %46:f32[1, 1, 1]{varying_rank_3} = stop_gradient %45
                            in (%42, %46)
                        }},
                    ]
                in (%13, %14)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_product_attention_type_inference() {
        let query = ArrayType::new_static(DataType::F32, [2, 2, 2]);
        let key_value = ArrayType::new_static(DataType::F32, [3, 1, 2]);
        let bias = ArrayType::new_static(DataType::F64, [3]);
        let mask = ArrayType::new_static(DataType::Boolean, [2, 3]);
        let lengths = ArrayType::new_static(DataType::I32, [1]);
        let input_structure = AttentionInputs {
            bias: Some(Placeholder),
            mask: Some(Placeholder),
            query_sequence_lengths: Some(Placeholder),
            key_value_sequence_lengths: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        let configuration = AttentionConfiguration::new().with_local_window((0, 0));
        let input_types = vec![
            query.clone(),
            key_value.clone(),
            key_value.clone(),
            bias.clone(),
            mask.clone(),
            lengths.clone(),
            lengths.clone(),
        ];

        // The attended output has the query type, and the residual drops the head dimension of the query.
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                configuration,
                input_structure,
                true,
                input_types.as_slice(),
            ),
            Ok(vec![query.clone(), ArrayType::new_static(DataType::F32, [2, 2])]),
        );
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                configuration,
                input_structure,
                false,
                input_types.as_slice(),
            ),
            Ok(vec![query.clone()]),
        );

        // The residual has the `f32` data type of the softmax for every query data type.
        for data_type in [DataType::F16, DataType::BF16, DataType::F64] {
            let query = ArrayType::new_static(data_type, [2, 2, 2]);
            let key_value = ArrayType::new_static(data_type, [3, 1, 2]);
            assert_eq!(
                attention_forward_output_types(
                    DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                    AttentionConfiguration::new(),
                    AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                    true,
                    &[query.clone(), key_value.clone(), key_value],
                ),
                Ok(vec![query, ArrayType::new_static(DataType::F32, [2, 2])]),
            );
        }

        // The number of input types must match the input structure.
        assert!(matches!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                configuration,
                input_structure,
                false,
                &input_types[..6],
            ),
            Err(TypeError::Invalid { message, .. })
                if message == format!(
                    "`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` expects 7 inputs for its input structure but got 6",
                ),
        ));

        // The query, key, and value must be rank-3 or rank-4 floating-point arrays that agree on their data type,
        // batch dimension, and head dimension, and the key and value must also agree on their sequence and heads
        // dimensions. The heads and head dimensions must be static, and the key/value heads must divide the query
        // heads. A bias and a mask must have rank at most 4 and broadcast to the scores, the bias must be numeric
        // or Boolean, the mask must be Boolean, and sequence lengths hold one `i32` length per batch item.
        let extent = Dimension::Dynamic(DimensionVariable::new("extent", DimensionBounds::positive(Some(4)).unwrap()));
        let plain = AttentionInputs::new(Placeholder, Placeholder, Placeholder);
        let scored = AttentionInputs { bias: Some(Placeholder), mask: Some(Placeholder), ..plain };
        let padded = AttentionInputs {
            query_sequence_lengths: Some(Placeholder),
            key_value_sequence_lengths: Some(Placeholder),
            ..plain
        };
        for (input_structure, input_types, message) in [
            (
                plain,
                vec![ArrayType::new_static(DataType::F32, [2, 2]), key_value.clone(), key_value.clone()],
                "query must have rank 3 or 4 but got rank 2",
            ),
            (
                plain,
                vec![ArrayType::new_static(DataType::I32, [2, 2, 2]), key_value.clone(), key_value.clone()],
                "requires floating-point inputs but got data type `i32`",
            ),
            (
                plain,
                vec![ArrayType::new_static(DataType::F8E8M0FNU, [2, 2, 2]); 3],
                "does not support `f8e8m0fnu` inputs, because that data type represents neither zero nor negative \
                 values",
            ),
            (
                plain,
                vec![query.clone(), ArrayType::new_static(DataType::F64, [3, 1, 2]), key_value.clone()],
                "key data type `f64` does not match the query data type `f32`",
            ),
            (
                plain,
                vec![query.clone(), ArrayType::new_static(DataType::F32, [3, 3, 1, 2]), key_value.clone()],
                "key batch dimension (3) does not match the query batch dimension (1)",
            ),
            (
                plain,
                vec![query.clone(), key_value.clone(), ArrayType::new_static(DataType::F32, [3, 1, 3])],
                "value head dimension (3) does not match the query head dimension (2)",
            ),
            (
                plain,
                vec![query.clone(), key_value.clone(), ArrayType::new_static(DataType::F32, [4, 1, 2])],
                "value sequence dimension (4) does not match the key sequence dimension (3)",
            ),
            (
                plain,
                vec![query.clone(), key_value.clone(), ArrayType::new_static(DataType::F32, [3, 2, 2])],
                "value heads dimension (2) does not match the key heads dimension (1)",
            ),
            (
                plain,
                vec![
                    ArrayType::new(DataType::F32, Shape::new(vec![2.into(), extent.clone(), 2.into()])),
                    key_value.clone(),
                    key_value.clone(),
                ],
                "query heads dimension must be static but got extent",
            ),
            (
                plain,
                vec![
                    query.clone(),
                    ArrayType::new(DataType::F32, Shape::new(vec![3.into(), extent.clone(), 2.into()])),
                    ArrayType::new(DataType::F32, Shape::new(vec![3.into(), extent.clone(), 2.into()])),
                ],
                "key/value heads dimension must be static but got extent",
            ),
            (
                plain,
                vec![
                    ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 2.into(), extent.clone()])),
                    ArrayType::new(DataType::F32, Shape::new(vec![3.into(), 1.into(), extent.clone()])),
                    ArrayType::new(DataType::F32, Shape::new(vec![3.into(), 1.into(), extent])),
                ],
                "head dimension must be static but got extent",
            ),
            (
                plain,
                vec![
                    ArrayType::new_static(DataType::F32, [2, 3, 2]),
                    ArrayType::new_static(DataType::F32, [3, 2, 2]),
                    ArrayType::new_static(DataType::F32, [3, 2, 2]),
                ],
                "key/value heads dimension (2) must divide the query heads dimension (3)",
            ),
            (
                scored,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::F64, [1, 1, 1, 1, 3]),
                    mask.clone(),
                ],
                "bias must have rank at most 4 but got rank 5",
            ),
            (
                scored,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    bias.clone(),
                    ArrayType::new_static(DataType::F32, [2, 3]),
                ],
                "mask must have data type `bool` but got `f32`",
            ),
            (
                scored,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::Zero, [3]),
                    mask.clone(),
                ],
                "bias must have a real numeric or Boolean data type but got `zero`",
            ),
            (
                scored,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::C64, [3]),
                    mask.clone(),
                ],
                "bias must have a real numeric or Boolean data type but got `c64`",
            ),
            (
                scored,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::F64, [3, 1, 1, 3]),
                    mask.clone(),
                ],
                "bias batch dimension (3) must be 1 or match the query batch dimension (1)",
            ),
            (
                scored,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::F64, [3, 1, 3]),
                    mask.clone(),
                ],
                "bias heads dimension (3) must be 1 or match the query heads dimension (2)",
            ),
            (
                scored,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::F64, [3, 3]),
                    mask.clone(),
                ],
                "bias query-sequence dimension (3) must be 1 or match the query sequence dimension (2)",
            ),
            (
                scored,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::F64, [2]),
                    mask,
                ],
                "bias key/value-sequence dimension (2) must be 1 or match the key sequence dimension (3)",
            ),
            (
                scored,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    bias,
                    ArrayType::new_static(DataType::Boolean, [2, 2]),
                ],
                "mask key/value-sequence dimension (2) must be 1 or match the key sequence dimension (3)",
            ),
            (
                padded,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::I64, [1]),
                    lengths.clone(),
                ],
                "query sequence lengths must have data type `i32` but got `i64`",
            ),
            (
                padded,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    lengths.clone(),
                    ArrayType::new_static(DataType::I32, [2]),
                ],
                "key/value sequence lengths size (2) does not match the batch dimension (1)",
            ),
            (
                padded,
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::I32, [1, 1]),
                    lengths,
                ],
                "query sequence lengths must have rank 1 but got rank 2",
            ),
        ] {
            assert_eq!(
                attention_forward_output_types(
                    DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                    AttentionConfiguration::new(),
                    input_structure,
                    false,
                    input_types.as_slice(),
                ),
                Err(TypeError::invalid(format!("`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` {message}"))),
            );
        }

        // Dropout requires a rate in the open unit interval and a valid random state when a state is present.
        let plain_types = vec![query.clone(), key_value.clone(), key_value];
        let stateful = AttentionInputs {
            dropout_state: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        let mut stateful_types = plain_types.clone();
        stateful_types.push(ArrayType::new_static(DataType::U64, [2]));
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new().with_dropout(0.5),
                stateful,
                false,
                stateful_types.as_slice(),
            ),
            Ok(vec![query.clone()]),
        );
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new().with_dropout(0.5),
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                false,
                plain_types.as_slice(),
            ),
            Ok(vec![query]),
        );
        for rate in [0.0, 1.0, -0.5, f64::NAN] {
            assert!(matches!(
                attention_forward_output_types(
                    DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                    AttentionConfiguration::new().with_dropout(rate),
                    stateful,
                    false,
                    stateful_types.as_slice(),
                ),
                Err(TypeError::Invalid { message, .. })
                    if message == format!(
                        "`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` dropout rate must lie in the open interval (0, 1) \
                         but got {rate}",
                    ),
            ));
        }
        assert!(matches!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new(),
                stateful,
                false,
                stateful_types.as_slice(),
            ),
            Err(TypeError::Invalid { message, .. })
                if message == format!(
                    "`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` received a dropout random state without a dropout rate",
                ),
        ));
        let mut invalid_state_types = plain_types;
        invalid_state_types.push(ArrayType::new_static(DataType::F32, [2]));
        assert!(matches!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new().with_dropout(0.5),
                stateful,
                false,
                invalid_state_types.as_slice(),
            ),
            Err(TypeError::Invalid { message, .. })
                if message == "random generator states must have type `u64[2]` (i.e., for `three_fry`) or `u64[3]` \
                               (i.e., for `philox`) but got `f32[2]`",
        ));

        // Inputs inside manual regions must have matching manual variation, and reduced and unreduced inputs are
        // rejected.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant = ArrayType::new_static(DataType::F32, [1, 1, 1, 1])
            .with_sharding(Sharding::replicated(mesh.clone(), 4))
            .unwrap();
        let varying = invariant
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 4).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let reduced = invariant
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 4).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        let unreduced = invariant
            .clone()
            .with_sharding(Sharding::replicated(mesh, 4).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        assert!(matches!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new(),
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                false,
                &[varying, invariant.clone(), invariant.clone()],
            ),
            Err(TypeError::Invalid { message, .. })
                if message == format!(
                    "`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` inputs must have matching varying manual axes; insert \
                     `{PARALLEL_VARY_OPERATION_NAME}` on the inputs that lack an axis, as `align_manual_variation` \
                     does",
                ),
        ));
        assert!(matches!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new(),
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                false,
                &[reduced, invariant.clone(), invariant.clone()],
            ),
            Err(TypeError::Invalid { message, .. })
                if message == format!("`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` does not support reduced inputs"),
        ));
        assert!(matches!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new(),
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                false,
                &[unreduced, invariant.clone(), invariant],
            ),
            Err(TypeError::Invalid { message, .. })
                if message == format!("`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` does not support unreduced inputs"),
        ));
    }

    #[test]
    fn test_dot_product_attention_type_inference_memory() {
        // The attended output and the residual reside in the memory space of the query.
        let r#type = ArrayType::new_static(DataType::F32, [2, 1, 2]).with_memory(Memory::Host { pinned: true });
        let residual_type = ArrayType::new_static(DataType::F32, [2, 1]).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new(),
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                true,
                &[r#type.clone(), r#type.clone(), r#type.clone()],
            ),
            Ok(vec![r#type.clone(), residual_type.clone()]),
        );

        // The bodies of the definitions agree with those boundaries, so staging attention over host memory succeeds
        // for the forward definition and for the backward definition that reverse mode stages.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let (output_types, _) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()),
        )
        .unwrap();
        assert_eq!(output_types, (r#type.clone(), residual_type));
        let function = differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let (gradient_types, _) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| {
                differentiate_at(inputs)
                    .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2], ReductionKind::Sum))
                    .map_err(ProgramError::from)
            },
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()),
        )
        .unwrap();
        assert_eq!(gradient_types, AttentionInputs::new(r#type.clone(), r#type.clone(), r#type));
    }

    #[test]
    fn test_dot_product_attention_type_inference_sharding() {
        // The outputs have the shardings that the contractions of the body infer. The output takes its batch
        // placement from all three inputs and its head dimension placement from the value, and the residual takes
        // the placements of the scores. Each staged call agrees with its boundary.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharded = |dimensions: [usize; 4], axis: Option<usize>| {
            let mut placements = vec![ShardingDimension::Replicated; 4];
            if let Some(axis) = axis {
                placements[axis] = ShardingDimension::sharded(["x"]);
            }
            ArrayType::new_static(DataType::F32, dimensions)
                .with_sharding(Sharding::new(mesh.clone(), placements).unwrap())
                .unwrap()
        };
        let residual = |axis: Option<usize>| {
            let mut placements = vec![ShardingDimension::Replicated; 3];
            if let Some(axis) = axis {
                placements[axis] = ShardingDimension::sharded(["x"]);
            }
            ArrayType::new_static(DataType::F32, [2, 2, 1])
                .with_sharding(Sharding::new(mesh.clone(), placements).unwrap())
                .unwrap()
        };
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        for (query_axis, key_axis, value_axis, output_axis, residual_axis) in [
            // A head dimension placement of the value reaches the output, and one of the query does not.
            (None, None, Some(3), Some(3), None),
            (Some(3), None, None, None, None),
            // A batch placement of any input reaches the output and the residual.
            (None, Some(0), None, Some(0), Some(0)),
            (None, None, Some(0), Some(0), None),
            (Some(0), Some(0), Some(0), Some(0), Some(0)),
            // A sequence placement of the query reaches both, and one of the key or of the value reaches neither,
            // because the contractions over the key/value sequence drop a placement that only one side has.
            (Some(1), None, None, Some(1), Some(1)),
            (None, Some(1), None, None, None),
            (None, None, Some(1), None, None),
        ] {
            let query = sharded([2, 2, 1, 2], query_axis);
            let key = sharded([2, 2, 1, 2], key_axis);
            let value = sharded([2, 2, 1, 2], value_axis);
            let expected = (sharded([2, 2, 1, 2], output_axis), residual(residual_axis));
            assert_eq!(
                attention_forward_output_types(
                    DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                    AttentionConfiguration::new(),
                    AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                    true,
                    &[query.clone(), key.clone(), value.clone()],
                ),
                Ok(vec![expected.0.clone(), expected.1.clone()]),
            );
            let (output_types, _) = ArrayContext::trace(
                |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
                AttentionInputs::new(query, key, value),
            )
            .unwrap();
            assert_eq!(output_types, expected);
        }

        // Rank-3 inputs and grouped key/value heads go through the reshapes that normalize and expand them, and the
        // sequence placement of the query still reaches both outputs.
        let mesh_sharding = |placements: Vec<ShardingDimension>| Sharding::new(mesh.clone(), placements).unwrap();
        let x = ShardingDimension::sharded(["x"]);
        let replicated = ShardingDimension::Replicated;
        let query = ArrayType::new_static(DataType::F32, [2, 2, 2])
            .with_sharding(mesh_sharding(vec![x.clone(), replicated.clone(), replicated.clone()]))
            .unwrap();
        let key_value = ArrayType::new_static(DataType::F32, [2, 1, 2])
            .with_sharding(mesh_sharding(vec![replicated.clone(), replicated.clone(), replicated.clone()]))
            .unwrap();
        let (output_types, _) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
            AttentionInputs::new(query.clone(), key_value.clone(), key_value),
        )
        .unwrap();
        assert_eq!(
            output_types,
            (
                query,
                ArrayType::new_static(DataType::F32, [2, 2])
                    .with_sharding(mesh_sharding(vec![x, replicated]))
                    .unwrap(),
            ),
        );

        // The bias, the mask, and the sequence lengths take part in the placement of the `BNTS` scores like the inputs
        // of an elementwise operation. Their batch and query sequence placements reach the output and the residual,
        // and their key/value sequence placement is contracted away with the value.
        let placed = |data_type: DataType, dimensions: &[usize], axis: Option<usize>| {
            let mut placements = vec![ShardingDimension::Replicated; dimensions.len()];
            if let Some(axis) = axis {
                placements[axis] = ShardingDimension::sharded(["x"]);
            }
            ArrayType::new_static(data_type, dimensions.to_vec())
                .with_sharding(Sharding::new(mesh.clone(), placements).unwrap())
                .unwrap()
        };
        let input_structure = AttentionInputs {
            bias: Some(Placeholder),
            mask: Some(Placeholder),
            query_sequence_lengths: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        for (bias_axis, mask_axis, lengths_axis, output_axis, residual_axis) in [
            (Some(0), None, None, Some(0), Some(0)),
            (Some(2), None, None, Some(1), Some(1)),
            (Some(3), None, None, None, None),
            (None, Some(0), None, Some(0), Some(0)),
            (None, Some(2), None, Some(1), Some(1)),
            (None, Some(3), None, None, None),
            (None, None, Some(0), Some(0), Some(0)),
            (Some(0), Some(0), Some(0), Some(0), Some(0)),
        ] {
            let types = AttentionInputs {
                bias: Some(placed(DataType::F32, &[2, 1, 2, 2], bias_axis)),
                mask: Some(placed(DataType::Boolean, &[2, 1, 2, 2], mask_axis)),
                query_sequence_lengths: Some(placed(DataType::I32, &[2], lengths_axis)),
                ..AttentionInputs::new(
                    sharded([2, 2, 1, 2], None),
                    sharded([2, 2, 1, 2], None),
                    sharded([2, 2, 1, 2], None),
                )
            };
            let expected = (sharded([2, 2, 1, 2], output_axis), residual(residual_axis));
            assert_eq!(
                attention_forward_output_types(
                    DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                    AttentionConfiguration::new(),
                    input_structure.clone(),
                    true,
                    types.parameters().cloned().collect::<Vec<_>>().as_slice(),
                ),
                Ok(vec![expected.0.clone(), expected.1.clone()]),
            );
            let (output_types, _) =
                ArrayContext::trace(|inputs: AttentionInputs<_>| function.call_with_residual(inputs), types).unwrap();
            assert_eq!(output_types, expected);
        }

        // A placement that cannot be combined with that of the scores is rejected (e.g., a bias that is sharded along
        // the query sequence over the mesh axis that already shards the batch dimension of the scores).
        let bias = placed(DataType::F32, &[2, 1, 2, 2], Some(2));
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new(),
                AttentionInputs {
                    bias: Some(Placeholder),
                    ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
                },
                false,
                &[
                    sharded([2, 2, 1, 2], Some(0)),
                    sharded([2, 2, 1, 2], None),
                    sharded([2, 2, 1, 2], None),
                    bias.clone()
                ],
            ),
            Err(TypeError::invalid(format!(
                "`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` input type `{bias}` has a sharding that cannot be combined \
                 with that of the attention scores `{}`",
                sharded([2, 1, 2, 2], Some(0)),
            ))),
        );

        // Contracted dimensions that are sharded on both sides make the placement of the result ambiguous (e.g., the
        // head dimensions of the query and of the key, or the sequence dimensions of the key and of the value), which
        // the boundary reports in the name of the definition.
        for (query_axis, key_axis, value_axis) in [(Some(3), Some(3), Some(3)), (None, Some(1), Some(1))] {
            assert_eq!(
                attention_forward_output_types(
                    DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                    AttentionConfiguration::new(),
                    AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                    false,
                    &[
                        sharded([2, 2, 1, 2], query_axis),
                        sharded([2, 2, 1, 2], key_axis),
                        sharded([2, 2, 1, 2], value_axis),
                    ],
                ),
                Err(TypeError::invalid(format!(
                    "`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` contracting dimensions are sharded, making the output \
                     sharding ambiguous; request an explicit output sharding (e.g., one with unreduced axes) to \
                     resolve it",
                ))),
            );
        }
    }

    #[test]
    fn test_dot_product_attention_type_inference_layout() {
        // The forward body returns arrays without an explicit layout, so the outputs of the forward definition do not
        // promise the layouts of their inputs, and staging it over inputs with an explicit layout succeeds.
        let plain = ArrayType::new_static(DataType::F32, [2, 1, 2]);
        let r#type = plain.clone().with_layout(Layout::Strided(StridedLayout::new(vec![8, 8, 4])));
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new(),
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                true,
                &[r#type.clone(), r#type.clone(), r#type.clone()],
            ),
            Ok(vec![plain.clone(), ArrayType::new_static(DataType::F32, [2, 1])]),
        );
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let (output_types, _) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()),
        )
        .unwrap();
        assert_eq!(output_types, (plain.clone(), ArrayType::new_static(DataType::F32, [2, 1])));

        // A layout describes how one array is stored, which its cotangent does not have to share, so the cotangents
        // of the backward definition and the gradients have no explicit layout either.
        let residual = ArrayType::new_static(DataType::F32, [2, 1]);
        assert_eq!(
            attention_backward_output_types(
                DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
                AttentionConfiguration::new(),
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                &[r#type.clone(), r#type.clone(), r#type.clone(), plain.clone(), residual, plain.clone()],
            ),
            Ok(vec![plain.clone(), plain.clone(), plain.clone()]),
        );
        let function = differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let (gradient_types, _) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| {
                differentiate_at(inputs)
                    .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2], ReductionKind::Sum))
                    .map_err(ProgramError::from)
            },
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
        )
        .unwrap();
        assert_eq!(gradient_types, AttentionInputs::new(plain.clone(), plain.clone(), plain));
    }

    #[test]
    fn test_dot_product_attention_backward_type_inference() {
        let query = ArrayType::new_static(DataType::F32, [2, 2, 2]);
        let key_value = ArrayType::new_static(DataType::F32, [3, 1, 2]);
        let bias = ArrayType::new_static(DataType::F64, [3]);
        let lengths = ArrayType::new_static(DataType::I32, [1]);
        let residual = ArrayType::new_static(DataType::F32, [2, 2]);
        let configuration = AttentionConfiguration::new().with_local_window((0, 0));

        // The cotangents of the query, key, and value are followed by that of a floating-point bias.
        let input_types = vec![
            query.clone(),
            key_value.clone(),
            key_value.clone(),
            bias.clone(),
            ArrayType::new_static(DataType::Boolean, [2, 3]),
            lengths.clone(),
            lengths,
            query.clone(),
            residual.clone(),
            query.clone(),
        ];
        assert_eq!(
            attention_backward_output_types(
                DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
                configuration,
                AttentionInputs {
                    bias: Some(Placeholder),
                    mask: Some(Placeholder),
                    query_sequence_lengths: Some(Placeholder),
                    key_value_sequence_lengths: Some(Placeholder),
                    ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
                },
                input_types.as_slice(),
            ),
            Ok(vec![query.clone(), key_value.clone(), key_value.clone(), bias]),
        );

        // The number of input types must match the input structure and the three inputs that follow it.
        assert!(matches!(
            attention_backward_output_types(
                DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
                configuration,
                AttentionInputs {
                    bias: Some(Placeholder),
                    mask: Some(Placeholder),
                    query_sequence_lengths: Some(Placeholder),
                    key_value_sequence_lengths: Some(Placeholder),
                    ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
                },
                &input_types[..9],
            ),
            Err(TypeError::Invalid { message, .. })
                if message == format!(
                    "`{DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME}` expects 10 inputs for its input structure but \
                     got 9",
                ),
        ));

        // The forward output, its residual, and the output cotangent must have the types that the forward inputs
        // determine.
        for (input_types, message) in [
            (
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    ArrayType::new_static(DataType::F32, [2, 2, 1]),
                    residual.clone(),
                    query.clone(),
                ],
                "output type f32[2, 2, 1] does not match the expected forward output type f32[2, 2, 2]",
            ),
            (
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    query.clone(),
                    ArrayType::new_static(DataType::F32, [2]),
                    query.clone(),
                ],
                "residual type f32[2] does not match the expected residual type f32[2, 2]",
            ),
            (
                vec![
                    query.clone(),
                    key_value.clone(),
                    key_value.clone(),
                    query.clone(),
                    residual.clone(),
                    ArrayType::new_static(DataType::F32, [2, 2, 1]),
                ],
                "output cotangent type f32[2, 2, 1] does not match the expected cotangent type f32[2, 2, 2]",
            ),
            // The residual keeps the `f32` data type of the softmax, so one in a narrower query data type is rejected.
            (
                vec![
                    ArrayType::new_static(DataType::BF16, [2, 2, 2]),
                    ArrayType::new_static(DataType::BF16, [3, 1, 2]),
                    ArrayType::new_static(DataType::BF16, [3, 1, 2]),
                    ArrayType::new_static(DataType::BF16, [2, 2, 2]),
                    ArrayType::new_static(DataType::BF16, [2, 2]),
                    ArrayType::new_static(DataType::BF16, [2, 2, 2]),
                ],
                "residual type bf16[2, 2] does not match the expected residual type f32[2, 2]",
            ),
        ] {
            assert_eq!(
                attention_backward_output_types(
                    DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
                    configuration,
                    AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                    input_types.as_slice(),
                ),
                Err(TypeError::invalid(format!("`{DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME}` {message}"))),
            );
        }

        // An integer bias remains a legal forward input, but its cotangent space is zero-dimensional and so it
        // contributes no backward output.
        let input_types = vec![
            query.clone(),
            key_value.clone(),
            key_value.clone(),
            ArrayType::new_static(DataType::I32, [3]),
            query.clone(),
            residual,
            query.clone(),
        ];
        assert_eq!(
            attention_backward_output_types(
                DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
                configuration,
                AttentionInputs {
                    bias: Some(Placeholder),
                    ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
                },
                input_types.as_slice(),
            ),
            Ok(vec![query, key_value.clone(), key_value]),
        );
    }

    #[test]
    fn test_dot_product_attention_interpretation() {
        // Rank-3 query, key, and value inputs normalize through an implicit batch. The arbitrary mask, asymmetric
        // local window, and query lengths compose independently, so only the first key is visible to the first query
        // position, while the second query position lies beyond the query length and is therefore zero. The omitted
        // scale defaults to `1 / sqrt(head_dimension)`.
        let inputs = AttentionInputs {
            bias: Some(Array::from_elements(ArrayType::new_static(DataType::I32, []), &[0i32]).unwrap()),
            mask: Some(
                Array::from_elements(
                    ArrayType::new_static(DataType::Boolean, [2, 3]),
                    &[true, false, true, true, true, false],
                )
                .unwrap(),
            ),
            query_sequence_lengths: Some(
                Array::from_elements(ArrayType::new_static(DataType::I32, [1]), &[1i32]).unwrap(),
            ),
            ..AttentionInputs::new(
                f32_array(&[2, 1, 2], &[1.0, 0.0, 0.0, 1.0]),
                f32_array(&[3, 1, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
                f32_array(&[3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            )
        };
        let configuration = AttentionConfiguration::new().with_local_window((1, 0));
        let (output, residual) = attend(inputs.clone(), configuration);
        assert_eq!(output, f32_array(&[2, 1, 2], &[1.0, 2.0, 0.0, 0.0]));
        let invisible = (-0.7 * f64::from(f32::MAX)) as f32;
        assert_eq!(residual, f32_array(&[2, 1], &[std::f32::consts::FRAC_1_SQRT_2, invisible]));
        let (explicit, _) = attend(inputs, configuration.with_scale(1.0 / 2f64.sqrt()));
        assert_eq!(output, explicit);

        // Dot products, scaling, and bias addition of `f64` inputs retain their precision until the logits reach the
        // `f32` softmax, so subtracting the large bias before that conversion preserves the unit gap of the logits.
        let query = Array::from_elements(ArrayType::new_static(DataType::F64, [1, 1, 1]), &[1.0e8f64]).unwrap();
        let key =
            Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[1.0f64, 1.0 + 1.0e-8]).unwrap();
        let value = Array::from_elements(ArrayType::new_static(DataType::F64, [2, 1, 1]), &[0.0f64, 1.0]).unwrap();
        let bias = Array::from_elements(ArrayType::new_static(DataType::F64, [2]), &[-1.0e8f64, -1.0e8]).unwrap();
        let (output, _) = attend(
            AttentionInputs { bias: Some(bias), ..AttentionInputs::new(query, key, value) },
            AttentionConfiguration::new().with_scale(1.0),
        );
        assert_abs_diff_eq!(output.to_f64s()[0], 0.7310586, epsilon = 1e-6);

        // Multi-query attention equals multi-head attention over explicitly repeated key and value heads.
        let query = f32_array(&[1, 1, 2, 1], &[1.0, 1.0]);
        let (grouped, _) = attend(
            AttentionInputs::new(
                query.clone(),
                f32_array(&[1, 2, 1, 1], &[1.0, 2.0]),
                f32_array(&[1, 2, 1, 1], &[10.0, 20.0]),
            ),
            AttentionConfiguration::new(),
        );
        let (repeated, _) = attend(
            AttentionInputs::new(
                query,
                f32_array(&[1, 2, 2, 1], &[1.0, 1.0, 2.0, 2.0]),
                f32_array(&[1, 2, 2, 1], &[10.0, 10.0, 20.0, 20.0]),
            ),
            AttentionConfiguration::new(),
        );
        assert_eq!(grouped, repeated);

        // Invalid inputs are rejected by the output type rule of the definition before its body is traced.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        assert!(matches!(
            function.call(AttentionInputs::new(
                f32_array(&[1, 1, 2], &[1.0, 1.0]),
                f32_array(&[1, 1, 1], &[1.0]),
                f32_array(&[1, 1, 1], &[1.0]),
            )),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == format!(
                    "`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` key head dimension (1) does not match the query head \
                     dimension (2)",
                ),
        ));
    }

    #[test]
    fn test_dot_product_attention_interpretation_array_ir() {
        // Array IR values compute the same attention as their array members.
        let query = f32_array(&[2, 1, 2], &[1.0, 0.0, 0.0, 1.0]);
        let key = f32_array(&[3, 1, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let value = f32_array(&[3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let configuration = AttentionConfiguration::new();
        let (output, residual) = attend(AttentionInputs::new(query.clone(), key.clone(), value.clone()), configuration);
        let function = dot_product_attention::<ArrayIrContext>(configuration);
        assert_eq!(
            function.call_with_residual(AttentionInputs::new(
                ArrayIrValue::Array(query),
                ArrayIrValue::Array(key),
                ArrayIrValue::Array(value),
            )),
            Ok((ArrayIrValue::Array(output), ArrayIrValue::Array(residual))),
        );
    }

    #[test]
    fn test_dot_product_attention_interpretation_low_precision() {
        // Data types that are narrower than `f32` contract their scores in `f32`. The first score is `65536`, which
        // `f16` cannot represent, so a contraction that returned `f16` scores would make it infinite and the softmax
        // would return NaN, while the promoted scores select the first value.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_scale(1.0));
        let inputs = AttentionInputs::new(
            f16_array(&[1, 1, 1, 1], &[256.0]),
            f16_array(&[1, 2, 1, 1], &[256.0, 255.0]),
            f16_array(&[1, 2, 1, 1], &[1.0, 2.0]),
        );
        assert_eq!(function.call(inputs), Ok(f16_array(&[1, 1, 1, 1], &[1.0])));

        // The residual keeps the `f32` data type of the softmax. Both `bf16` scores are `1000`, so the residual is
        // `1000 + log(2)`, which `bf16` would round to `1000` and thereby lose the normalization of the two keys.
        let inputs = AttentionInputs::new(
            bf16_array(&[1, 1, 1, 1], &[1.0]),
            bf16_array(&[1, 2, 1, 1], &[1000.0, 1000.0]),
            bf16_array(&[1, 2, 1, 1], &[1.0, 0.0]),
        );
        assert_eq!(
            function.call_with_residual(inputs),
            Ok((bf16_array(&[1, 1, 1, 1], &[0.5]), f32_array(&[1, 1, 1], &[1000.0 + std::f32::consts::LN_2]))),
        );

        // The residual of the `f16` call above is `65536`, which is finite in `f32` and would overflow `f16`.
        let inputs = AttentionInputs::new(
            f16_array(&[1, 1, 1, 1], &[256.0]),
            f16_array(&[1, 2, 1, 1], &[256.0, 255.0]),
            f16_array(&[1, 2, 1, 1], &[1.0, 2.0]),
        );
        assert_eq!(
            function.call_with_residual(inputs),
            Ok((f16_array(&[1, 1, 1, 1], &[1.0]), f32_array(&[1, 1, 1], &[65536.0]))),
        );
    }

    #[test]
    fn test_dot_product_attention_interpretation_invisible_rows() {
        // The mask hides every key/value position from the second query row, which therefore attends uniformly to
        // all four positions, including the ones with the largest scores, and whose residual is the finite logit of
        // the invisible scores. The first query row attends to its two visible positions with equal scores.
        let mask = Array::from_elements(
            ArrayType::new_static(DataType::Boolean, [2, 4]),
            &[true, true, false, false, false, false, false, false],
        )
        .unwrap();
        let invisible = (-0.7 * f64::from(f32::MAX)) as f32;
        let configuration = AttentionConfiguration::new().with_scale(1.0);
        let (output, residual) = attend(
            AttentionInputs {
                mask: Some(mask.clone()),
                ..AttentionInputs::new(
                    f32_array(&[2, 1, 1], &[1.0, 1.0]),
                    f32_array(&[4, 1, 1], &[0.0, 0.0, 5.0, 5.0]),
                    f32_array(&[4, 1, 1], &[1.0, 3.0, 8.0, 4.0]),
                )
            },
            configuration,
        );
        assert_eq!(output, f32_array(&[2, 1, 1], &[2.0, 4.0]));
        assert_eq!(residual, f32_array(&[2, 1], &[std::f32::consts::LN_2, invisible]));

        // An invisible position has no probability next to a visible one, however small the visible score is. Here
        // a bias gives the visible score the finite logit that stands in for the scores of rows without any visible
        // position, and the row still attends only to its visible position.
        let (output, residual) = attend(
            AttentionInputs {
                bias: Some(f32_array(&[1, 2], &[invisible, invisible])),
                mask: Some(
                    Array::from_elements(ArrayType::new_static(DataType::Boolean, [1, 2]), &[true, false]).unwrap(),
                ),
                ..AttentionInputs::new(
                    f32_array(&[1, 1, 1], &[0.0]),
                    f32_array(&[2, 1, 1], &[0.0, 0.0]),
                    f32_array(&[2, 1, 1], &[0.0, 1.0]),
                )
            },
            configuration,
        );
        assert_eq!(output, f32_array(&[1, 1, 1], &[0.0]));
        assert_eq!(residual, f32_array(&[1, 1], &[invisible]));

        // The logits of `f64` inputs are converted to the `f32` softmax data type, in which the logit of the
        // invisible scores remains finite, so such rows behave in the same way.
        let (output, residual) = attend(
            AttentionInputs {
                mask: Some(mask),
                ..AttentionInputs::new(
                    f64_array(&[2, 1, 1], &[1.0, 1.0]),
                    f64_array(&[4, 1, 1], &[0.0, 0.0, 5.0, 5.0]),
                    f64_array(&[4, 1, 1], &[1.0, 3.0, 8.0, 4.0]),
                )
            },
            configuration,
        );
        assert_eq!(output, f64_array(&[2, 1, 1], &[2.0, 4.0]));
        assert_eq!(residual, f32_array(&[2, 1], &[std::f32::consts::LN_2, invisible]));
    }

    #[test]
    fn test_dot_product_attention_interpretation_local_window() {
        // A radius beyond the `i32` range of the position coordinates admits every offset on its side. The scores
        // of all positions are equal, so every row averages the values that its window admits.
        let inputs = AttentionInputs::new(
            f32_array(&[2, 1, 1], &[1.0, 1.0]),
            f32_array(&[2, 1, 1], &[1.0, 1.0]),
            f32_array(&[2, 1, 1], &[1.0, 3.0]),
        );
        let (output, _) = attend(inputs.clone(), AttentionConfiguration::new().with_local_window((0, usize::MAX)));
        assert_eq!(output, f32_array(&[2, 1, 1], &[2.0, 3.0]));
        let (output, _) = attend(inputs.clone(), AttentionConfiguration::new().with_local_window((usize::MAX, 0)));
        assert_eq!(output, f32_array(&[2, 1, 1], &[1.0, 2.0]));
        let (output, _) = attend(inputs, AttentionConfiguration::new().with_symmetric_local_window(usize::MAX));
        assert_eq!(output, f32_array(&[2, 1, 1], &[2.0, 2.0]));
    }

    #[test]
    fn test_dot_product_attention_interpretation_empty_head_dimension() {
        // The scores of an empty head dimension are empty contractions, which are zero, so every query row attends
        // uniformly to the three key/value positions and its residual is `log(3)`. The default scale is one for such
        // inputs, because the infinite `1 / sqrt(0)` would turn the zero scores into NaNs.
        let (output, residual) = attend(
            AttentionInputs::new(
                f32_array(&[1, 2, 1, 0], &[]),
                f32_array(&[1, 3, 1, 0], &[]),
                f32_array(&[1, 3, 1, 0], &[]),
            ),
            AttentionConfiguration::new(),
        );
        assert_eq!(output, f32_array(&[1, 2, 1, 0], &[]));
        assert_eq!(residual, f32_array(&[1, 2, 1], &[3f32.ln(), 3f32.ln()]));
    }

    #[test]
    fn test_dot_product_attention_interpretation_dropout() {
        let inputs = AttentionInputs::new(
            f32_array(&[1, 4, 1, 1], &[0.0; 4]),
            f32_array(&[1, 4, 1, 1], &[0.0; 4]),
            f32_array(&[1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0]),
        );
        let with_state = |seed: u64| AttentionInputs {
            dropout_state: Some(Array::from_elements(ArrayType::new_static(DataType::U64, [2]), &[0, seed]).unwrap()),
            ..inputs.clone()
        };
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_dropout(0.5));

        // A state determines the dropout mask, and an absent state selects a fixed state.
        assert_eq!(function.call(with_state(1)), function.call(with_state(1)));
        assert_eq!(function.call(inputs.clone()), function.call(inputs.clone()));

        // Dropout scales the kept uniform attention weights by `1 / (1 - rate)`, so every output row is the sum of a
        // subset of the values scaled by `2 / 4`, and different states select different subsets. The subsets
        // themselves belong to the dropout operation, so this test does not pin them.
        let outputs = (0..8).map(|seed| function.call(with_state(seed)).unwrap().to_f64s()).collect::<Vec<_>>();
        assert!(outputs.iter().any(|output| output != &outputs[0]));
        for subset_sum in outputs.iter().flatten().map(|element| element / 0.5) {
            assert!((0.0..=10.0).contains(&subset_sum));
            assert_eq!(subset_sum.fract(), 0.0);
        }
    }

    #[test]
    fn test_dot_product_attention_dynamic_shapes() {
        // Homogeneous array programs stage attention over dynamic batch and sequence extents. The scale, the causal
        // mask, the sequence length masks, and the softmax broadcasts take the runtime shape of the scores, so the
        // staged program agrees with eager attention at every runtime shape.
        let batch = DimensionVariable::new("batch", DimensionBounds::positive(Some(4)).unwrap());
        let query_length = DimensionVariable::new("query_length", DimensionBounds::positive(Some(4)).unwrap());
        let key_length = DimensionVariable::new("key_length", DimensionBounds::positive(Some(4)).unwrap());
        let input_type = |length: &DimensionVariable| {
            let dimensions = vec![batch.clone().into(), length.clone().into(), 1.into(), 2.into()];
            ArrayType::new(DataType::F32, Shape::new(dimensions))
        };
        let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![batch.clone().into()]));
        let elements = |count: usize, offset: f32| {
            (0..count).map(|index| ((index * 7 + 3) % 11) as f32 * 0.25 - offset).collect::<Vec<_>>()
        };
        let attention_inputs = |batch: usize, query_length: usize, key_length: usize| {
            AttentionInputs::new(
                f32_array(&[batch, query_length, 1, 2], &elements(2 * batch * query_length, 1.0)),
                f32_array(&[batch, key_length, 1, 2], &elements(2 * batch * key_length, 0.5)),
                f32_array(&[batch, key_length, 1, 2], &elements(2 * batch * key_length, -2.0)),
            )
        };
        let lengths = |values: &[i32]| {
            Array::from_elements(ArrayType::new_static(DataType::I32, [values.len()]), values).unwrap()
        };

        let configuration = AttentionConfiguration::new().with_causal(true);
        let function = dot_product_attention::<ArrayContext>(configuration);
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
            AttentionInputs {
                query_sequence_lengths: Some(lengths_type.clone()),
                key_value_sequence_lengths: Some(lengths_type),
                ..AttentionInputs::new(input_type(&query_length), input_type(&key_length), input_type(&key_length))
            },
        )
        .unwrap();
        for inputs in [
            AttentionInputs {
                query_sequence_lengths: Some(lengths(&[2, 1])),
                key_value_sequence_lengths: Some(lengths(&[3, 2])),
                ..attention_inputs(2, 2, 3)
            },
            AttentionInputs {
                query_sequence_lengths: Some(lengths(&[2])),
                key_value_sequence_lengths: Some(lengths(&[1])),
                ..attention_inputs(1, 3, 2)
            },
        ] {
            assert_eq!(program.interpret(inputs.clone()), Ok(function.call_with_residual(inputs).unwrap()));
        }

        // Attention without masks broadcasts only the softmax statistics and the scale.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs::new(input_type(&query_length), input_type(&key_length), input_type(&key_length)),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda \
                    %0:f32[batch, query_length, 1, 2], \
                    %1:f32[batch, key_length, 1, 2], \
                    %2:f32[batch, key_length, 1, 2] .
                let %3:f32[batch, query_length, 1, 2] = composite [
                    key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, local_window = none, \
                        implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, \
                        query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                ] %0 %1 %2 [
                    primal={
                        lambda \
                            %0:f32[batch, query_length, 1, 2], \
                            %1:f32[batch, key_length, 1, 2], \
                            %2:f32[batch, key_length, 1, 2] .
                        let %3:f32[batch, 1, query_length, key_length] = dot [
                            dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                rhs_batching=[0, 2]),
                        ] %0 %1
                            %4:f32[batch, 1, query_length, key_length] = fill_like [value=0.70710677] %3
                            %5:f32[batch, 1, query_length, key_length] = mul %3 %4
                            %6:f32[batch, 1, query_length] = reduce [kind=max, axes=[3]] %5
                            %7:f32[batch, 1, query_length, key_length] = broadcast_like [output_axes=[0, 1, 2]] %6 %5
                            %8:f32[batch, 1, query_length, key_length] = sub %5 %7
                            %9:f32[batch, 1, query_length, key_length] = exp %8
                            %10:f32[batch, 1, query_length] = reduce [kind=sum, axes=[3]] %9
                            %11:f32[batch, 1, query_length, key_length] = broadcast_like [output_axes=[0, 1, 2]] %10 %9
                            %12:f32[batch, 1, query_length, key_length] = div %9 %11
                            %13:f32[batch, 1, query_length, 2] = dot [
                                dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                    rhs_batching=[0, 2]),
                            ] %12 %2
                            %14:f32[batch, query_length, 1, 2] = transpose [permutation=[0, 2, 1, 3]] %13
                        in (%14)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
        for inputs in [attention_inputs(2, 2, 3), attention_inputs(1, 3, 2)] {
            assert_eq!(program.interpret(inputs.clone()), Ok(function.call(inputs).unwrap()));
        }

        // A bias and a mask are right-aligned to the scores through unit axes and then take the runtime shape of the
        // scores, and the local window compares row and column coordinates of that shape.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_local_window((1, 0)));
        let bias_type =
            ArrayType::new(DataType::F32, Shape::new(vec![query_length.clone().into(), key_length.clone().into()]));
        let mask_type = ArrayType::new(
            DataType::Boolean,
            Shape::new(vec![batch.clone().into(), 1.into(), query_length.clone().into(), key_length.clone().into()]),
        );
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs {
                bias: Some(bias_type),
                mask: Some(mask_type),
                ..AttentionInputs::new(input_type(&query_length), input_type(&key_length), input_type(&key_length))
            },
        )
        .unwrap();
        for (items, rows, columns) in [(2, 2, 3), (1, 3, 2)] {
            let visible = (0..items * rows * columns).map(|index| index % 3 != 1).collect::<Vec<_>>();
            let mask_type = ArrayType::new_static(DataType::Boolean, [items, 1, rows, columns]);
            let inputs = AttentionInputs {
                bias: Some(f32_array(&[rows, columns], &elements(rows * columns, 0.75))),
                mask: Some(Array::from_elements(mask_type, &visible).unwrap()),
                ..attention_inputs(items, rows, columns)
            };
            assert_eq!(program.interpret(inputs.clone()), Ok(function.call(inputs).unwrap()));
        }

        // Dropout draws the random bits of its mask for a static shape, so definitions with dropout still require
        // static shapes.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_dropout(0.5));
        let result = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs::new(input_type(&query_length), input_type(&key_length), input_type(&key_length)),
        );
        assert!(matches!(
            result,
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == format!(
                    "`{DROPOUT_OPERATION_NAME}` requires static extents along the axes that are not broadcast axes, \
                     but axis 0 of `f32[batch, 1, query_length, key_length]` has extent `batch`",
                ),
        ));

        // Grouped key/value heads are replicated along a new group axis and merged into the query heads, which keeps
        // the dynamic batch and sequence dimensions.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let grouped_type = |length: &DimensionVariable, heads: usize| {
            let dimensions = vec![batch.clone().into(), length.clone().into(), heads.into(), 2.into()];
            ArrayType::new(DataType::F32, Shape::new(dimensions))
        };
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs::new(
                grouped_type(&query_length, 4),
                grouped_type(&key_length, 2),
                grouped_type(&key_length, 2),
            ),
        )
        .unwrap();
        for (items, rows, columns) in [(2, 2, 3), (1, 3, 1)] {
            let inputs = AttentionInputs::new(
                f32_array(&[items, rows, 4, 2], &elements(8 * items * rows, 1.0)),
                f32_array(&[items, columns, 2, 2], &elements(4 * items * columns, 0.5)),
                f32_array(&[items, columns, 2, 2], &elements(4 * items * columns, -2.0)),
            );
            assert_eq!(program.interpret(inputs.clone()), Ok(function.call(inputs).unwrap()));
        }
    }

    #[test]
    fn test_dot_product_attention_batching() {
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());

        // Each batch item carries one complete rank-4 attention problem. The batching rule folds the mapped axis into
        // the logical batch axis and restores it on the outputs.
        let ones = f32_array(&[2, 1, 1, 1, 1], &[1.0, 1.0]);
        let values = f32_array(&[2, 1, 1, 1, 1], &[5.0, 7.0]);
        let (output, residual) = batch(
            |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
            AttentionInputs::new(ones.clone(), ones, values.clone()),
            AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        assert_eq!(output, values);
        assert_eq!(residual, f32_array(&[2, 1, 1, 1], &[1.0, 1.0]));

        // Rank-3 attention has an implicit logical batch of one. A mapped rank-2 mask is normalized alongside the
        // query, key, and value rather than being mistaken for a batch prefix, and a replicated value is broadcast
        // along the mapped axis.
        let inputs = AttentionInputs {
            mask: Some(
                Array::from_elements(ArrayType::new_static(DataType::Boolean, [2, 1, 2]), &[true, false, false, true])
                    .unwrap(),
            ),
            ..AttentionInputs::new(
                f32_array(&[2, 1, 1, 1], &[1.0, 1.0]),
                f32_array(&[2, 2, 1, 1], &[1.0, 2.0, 1.0, 2.0]),
                f32_array(&[2, 1, 1], &[3.0, 10.0]),
            )
        };
        let axes = AttentionInputs {
            mask: Some(BatchAxis::new(0)),
            ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::replicated())
        };
        let (output, residual) = batch(
            |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
            inputs,
            axes,
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        assert_eq!(output, f32_array(&[2, 1, 1, 1], &[3.0, 10.0]));
        assert_eq!(residual, f32_array(&[2, 1, 1], &[1.0, 2.0]));

        // Mapped sequence lengths become the lengths of the folded batch, so every batch item keeps its own lengths.
        // Here the second key is padding for the first batch item, and the second query is padding for the second one.
        let zeros = f32_array(&[2, 1, 2, 1, 1], &[0.0; 4]);
        let inputs = AttentionInputs {
            query_sequence_lengths: Some(
                Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1]), &[2i32, 1]).unwrap(),
            ),
            key_value_sequence_lengths: Some(
                Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1]), &[1i32, 2]).unwrap(),
            ),
            ..AttentionInputs::new(zeros.clone(), zeros, f32_array(&[2, 1, 2, 1, 1], &[3.0, 5.0, 3.0, 5.0]))
        };
        let axes = AttentionInputs {
            query_sequence_lengths: Some(BatchAxis::new(0)),
            key_value_sequence_lengths: Some(BatchAxis::new(0)),
            ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0))
        };
        let output =
            batch(|inputs: AttentionInputs<_>| function.call(inputs), inputs, axes, BatchAxis::new(0), None).unwrap();
        assert_eq!(output, f32_array(&[2, 1, 2, 1, 1], &[3.0, 3.0, 4.0, 0.0]));

        // Batching inside a trace stages the folded call of the same definition inside the rule program of the
        // batched call.
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| {
                batch(
                    |inputs: AttentionInputs<_>| Ok(function.call_with_residual(inputs)?.0),
                    inputs,
                    AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(1), BatchAxis::new(1)),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            },
            AttentionInputs::new(
                ArrayType::new_static(DataType::F32, [3, 1, 2, 1, 1]),
                ArrayType::new_static(DataType::F32, [1, 3, 2, 1, 1]),
                ArrayType::new_static(DataType::F32, [1, 3, 2, 1, 1]),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[3, 1, 2, 1, 1], %1:f32[1, 3, 2, 1, 1], %2:f32[1, 3, 2, 1, 1] .
                let %3:f32[3, 1, 2, 1, 1], %4:f32[3, 1, 2, 1] = composite [
                    key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, local_window = none, \
                        implementation = 0, dropout_rate = none, residual = true, bias = false, mask = false, \
                        query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                    batching=[(extent=3, input_axes=[axis 0, axis 1, axis 1], output_axes=[axis 0, axis 0])],
                ] %0 %1 %2 [
                    primal={
                        lambda %0:f32[3, 1, 2, 1, 1], %1:f32[1, 3, 2, 1, 1], %2:f32[1, 3, 2, 1, 1] .
                        let %3:f32[3, 1, 2, 1, 1] = transpose [permutation=[1, 0, 2, 3, 4]] %1
                            %4:f32[3, 1, 2, 1, 1] = transpose [permutation=[1, 0, 2, 3, 4]] %2
                            %5:f32[3, 2, 1, 1] = reshape [shape=[3, 2, 1, 1]] %0
                            %6:f32[3, 2, 1, 1] = reshape [shape=[3, 2, 1, 1]] %3
                            %7:f32[3, 2, 1, 1] = reshape [shape=[3, 2, 1, 1]] %4
                            %8:f32[3, 2, 1, 1], %9:f32[3, 2, 1] = composite [
                                key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, \
                                    local_window = none, implementation = 0, dropout_rate = none, residual = true, \
                                    bias = false, mask = false, query_sequence_lengths = false, \
                                    key_value_sequence_lengths = false, dropout_state = false}\",
                            ] %5 %6 %7 [
                                primal={
                                    lambda %0:f32[3, 2, 1, 1], %1:f32[3, 2, 1, 1], %2:f32[3, 2, 1, 1] .
                                    let %3:f32[3, 1, 2, 2] = dot [
                                        dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                            rhs_batching=[0, 2]),
                                    ] %0 %1
                                        %4:f32[] = constant [value=1.0]
                                        %5:f32[3, 1, 2, 2] = broadcast [output_type=f32[3, 1, 2, 2], output_axes=[]] %4
                                        %6:f32[3, 1, 2, 2] = mul %3 %5
                                        %7:f32[3, 1, 2] = reduce [kind=max, axes=[3]] %6
                                        %8:f32[3, 1, 2, 2] = \
                                            broadcast [output_type=f32[3, 1, 2, 2], output_axes=[0, 1, 2]] %7
                                        %9:f32[3, 1, 2, 2] = sub %6 %8
                                        %10:f32[3, 1, 2, 2] = exp %9
                                        %11:f32[3, 1, 2] = reduce [kind=sum, axes=[3]] %10
                                        %12:f32[3, 1, 2, 2] = \
                                            broadcast [output_type=f32[3, 1, 2, 2], output_axes=[0, 1, 2]] %11
                                        %13:f32[3, 1, 2, 2] = div %10 %12
                                        %14:f32[3, 1, 2, 1] = dot [
                                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                                rhs_batching=[0, 2]),
                                        ] %13 %2
                                        %15:f32[3, 2, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %14
                                        %16:f32[3, 1, 2] = log %11
                                        %17:f32[3, 1, 2] = add %7 %16
                                        %18:f32[3, 2, 1] = transpose [permutation=[0, 2, 1]] %17
                                        %19:f32[3, 2, 1] = stop_gradient %18
                                    in (%15, %19)
                                },
                            ]
                            %10:f32[3, 1, 2, 1, 1] = reshape [shape=[3, 1, 2, 1, 1]] %8
                            %11:f32[3, 1, 2, 1] = reshape [shape=[3, 1, 2, 1]] %9
                        in (%10, %11)
                    },
                ]
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_product_attention_batching_manual_variation() {
        // Inside a manual region, a mapped bias is normalized to the canonical score shape with its own manual
        // variation, so that it still matches the varying query, key, and value of the folded call.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = |dimensions: Vec<usize>| {
            let rank = dimensions.len();
            ArrayType::new_static(DataType::F32, dimensions)
                .with_sharding(Sharding::replicated(mesh.clone(), rank).with_varying_manual_axes(["x"]).unwrap())
                .unwrap()
        };
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let (output_type, program) = DomainTracingContext::<ArrayContext>::trace_with_named_axes(
            |inputs: AttentionInputs<_>| {
                let axes = AttentionInputs {
                    bias: Some(BatchAxis::new(0)),
                    ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0))
                };
                batch(|inputs: AttentionInputs<_>| function.call(inputs), inputs, axes, BatchAxis::new(0), None)
                    .map_err(ProgramError::from)
            },
            AttentionInputs {
                bias: Some(varying(vec![2, 1, 2])),
                ..AttentionInputs::new(varying(vec![2, 1, 1, 1]), varying(vec![2, 2, 1, 1]), varying(vec![2, 2, 1, 1]))
            },
            vec![(
                "x".to_string(),
                NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true },
            )],
        )
        .unwrap();
        assert_eq!(output_type, varying(vec![2, 1, 1, 1]));
        let varying_rank_4 = rendered_sharding(4, true);
        let varying_rank_3 = rendered_sharding(3, true);
        let varying_rank_0 = rendered_sharding(0, true);
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda \
                    %0:f32[2, 1, 1, 1]{varying_rank_4}, \
                    %1:f32[2, 2, 1, 1]{varying_rank_4}, \
                    %2:f32[2, 2, 1, 1]{varying_rank_4}, \
                    %3:f32[2, 1, 2]{varying_rank_3} .
                let %4:f32[2, 1, 1, 1]{varying_rank_4} = composite [
                    key=\"ryft.dot_product_attention.v1 {{scale = none, causal = false, local_window = none, \
                        implementation = 0, dropout_rate = none, residual = false, bias = true, mask = false, \
                        query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}}\",
                    batching=[(extent=2, input_axes=[axis 0, axis 0, axis 0, axis 0], output_axes=[axis 0])],
                ] %0 %1 %2 %3 [
                    primal={{
                        lambda \
                            %0:f32[2, 1, 1, 1]{varying_rank_4}, \
                            %1:f32[2, 2, 1, 1]{varying_rank_4}, \
                            %2:f32[2, 2, 1, 1]{varying_rank_4}, \
                            %3:f32[2, 1, 2]{varying_rank_3} .
                        let %4:f32[2, 1, 1, 2]{varying_rank_4} = broadcast [
                            output_type=f32[2, 1, 1, 2]{varying_rank_4},
                            output_axes=[0, 2, 3],
                        ] %3
                            %5:f32[2, 1, 1, 1]{varying_rank_4} = composite [
                                key=\"ryft.dot_product_attention.v1 {{scale = none, causal = false, \
                                    local_window = none, implementation = 0, dropout_rate = none, residual = false, \
                                    bias = true, mask = false, query_sequence_lengths = false, \
                                    key_value_sequence_lengths = false, dropout_state = false}}\",
                            ] %0 %1 %2 %4 [
                                primal={{
                                    lambda \
                                        %0:f32[2, 1, 1, 1]{varying_rank_4}, \
                                        %1:f32[2, 2, 1, 1]{varying_rank_4}, \
                                        %2:f32[2, 2, 1, 1]{varying_rank_4}, \
                                        %3:f32[2, 1, 1, 2]{varying_rank_4} .
                                    let %4:f32[2, 1, 1, 2]{varying_rank_4} = dot [
                                        dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                            rhs_batching=[0, 2]),
                                    ] %0 %1
                                        %5:f32[]{varying_rank_0} = constant [value=1.0]
                                        %6:f32[2, 1, 1, 2]{varying_rank_4} = broadcast [
                                            output_type=f32[2, 1, 1, 2]{varying_rank_4},
                                            output_axes=[],
                                        ] %5
                                        %7:f32[2, 1, 1, 2]{varying_rank_4} = mul %4 %6
                                        %8:f32[2, 1, 1, 2]{varying_rank_4} = add %7 %3
                                        %9:f32[2, 1, 1]{varying_rank_3} = reduce [kind=max, axes=[3]] %8
                                        %10:f32[2, 1, 1, 2]{varying_rank_4} = broadcast [
                                            output_type=f32[2, 1, 1, 2]{varying_rank_4},
                                            output_axes=[0, 1, 2],
                                        ] %9
                                        %11:f32[2, 1, 1, 2]{varying_rank_4} = sub %8 %10
                                        %12:f32[2, 1, 1, 2]{varying_rank_4} = exp %11
                                        %13:f32[2, 1, 1]{varying_rank_3} = reduce [kind=sum, axes=[3]] %12
                                        %14:f32[2, 1, 1, 2]{varying_rank_4} = broadcast [
                                            output_type=f32[2, 1, 1, 2]{varying_rank_4},
                                            output_axes=[0, 1, 2],
                                        ] %13
                                        %15:f32[2, 1, 1, 2]{varying_rank_4} = div %12 %14
                                        %16:f32[2, 1, 1, 1]{varying_rank_4} = dot [
                                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                                rhs_batching=[0, 2]),
                                        ] %15 %2
                                        %17:f32[2, 1, 1, 1]{varying_rank_4} = transpose [permutation=[0, 2, 1, 3]] %16
                                    in (%17)
                                }},
                            ]
                        in (%5)
                    }},
                ]
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_product_attention_batching_dynamic_shapes() {
        // Homogeneous array families support dynamic sequence extents in batched calls whose mapped axis becomes the
        // implicit batch axis of rank-3 attention, because their batching rule reads no dynamic extent.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(3)).unwrap());
        let axes = AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0));
        let r#type =
            ArrayType::new(DataType::F32, Shape::new(vec![2.into(), length.clone().into(), 1.into(), 1.into()]));
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| {
                batch(|inputs: AttentionInputs<_>| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)
            },
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
        )
        .unwrap();
        let inputs = AttentionInputs::new(
            f32_array(&[2, 2, 1, 1], &[1.0, 0.5, -1.0, 2.0]),
            f32_array(&[2, 2, 1, 1], &[0.25, -0.5, 1.5, 1.0]),
            f32_array(&[2, 2, 1, 1], &[3.0, 5.0, 7.0, 11.0]),
        );
        let expected = batch(
            |inputs: AttentionInputs<_>| function.call(inputs),
            inputs.clone(),
            axes.clone(),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(program.interpret(inputs), Ok(expected));

        // Rank-4 attention folds its mapped axis into its batch axis with a reshape, whose output extents must be
        // static in homogeneous array families, so batching over a dynamic sequence extent fails.
        let r#type =
            ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 1.into(), length.into(), 1.into(), 1.into()]));
        let result = ArrayContext::trace(
            |inputs: AttentionInputs<_>| {
                batch(|inputs: AttentionInputs<_>| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)
            },
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
        );
        assert!(matches!(
            result,
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == format!(
                    "`{DIMENSION_SIZE_OPERATION_NAME}` cannot return the dynamic extent `length` of axis 2 of \
                     `f32[2, 1, length, 1, 1]` as a host extent; use array IR values for dynamic extents",
                ),
        ));
    }

    #[test]
    fn test_dot_product_attention_batching_dynamic_shapes_array_ir() {
        // Array IR families fold dynamic mapped extents with first-class dimension values, which the rule reads from
        // the mapped axis of a mapped input.
        let function = dot_product_attention::<ArrayIrContext>(AttentionConfiguration::new());
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let r#type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![
                Dimension::Dynamic(batch_variable),
                Dimension::Static(1),
                Dimension::Static(1),
                Dimension::Static(1),
                Dimension::Static(1),
            ]),
        );
        let query = context.input(r#type.clone().into());
        let key = context.input(r#type.clone().into());
        let value = context.input(r#type.clone().into());
        let output = batch(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs::new(query, key, value),
            AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, ArrayIrValue<Array>>(
                vec![output.atom_id().unwrap()],
                vec![Placeholder, Placeholder, Placeholder],
                Placeholder,
            )
            .unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(r#type));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[batch, 1, 1, 1, 1], %1:f32[batch, 1, 1, 1, 1], %2:f32[batch, 1, 1, 1, 1] .
                let %3:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %0
                    %4:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %1
                    %5:bool[] = const true
                    %6:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %2
                    %7:bool[] = const true
                    %8:f32[batch, 1, 1, 1, 1] = composite [
                        key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, local_window = none, \
                            implementation = 0, dropout_rate = none, residual = false, bias = false, mask = false, \
                            query_sequence_lengths = false, key_value_sequence_lengths = false, \
                            dropout_state = false}\",
                        non_differentiated_count=1,
                        batching=[(extent=dimension<batch ∈ [1, 5)>, input_axes=[axis 0, axis 0, axis 0], \
                            output_axes=[axis 0])],
                    ] %3 %0 %1 %2 [
                        primal={
                            lambda \
                                %0:dimension<batch ∈ [1, 5)>, \
                                %1:f32[batch, 1, 1, 1, 1], \
                                %2:f32[batch, 1, 1, 1, 1], \
                                %3:f32[batch, 1, 1, 1, 1] .
                            let %4:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %1
                                %5:dimension<1> = constant [value=1]
                                %6:dimension<1> = constant [value=1]
                                %7:dimension<1> = constant [value=1]
                                %8:dimension<1> = constant [value=1]
                                %9:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %1 %4 %6 %7 %8
                                %10:dimension<1> = constant [value=1]
                                %11:dimension<1> = constant [value=1]
                                %12:dimension<1> = constant [value=1]
                                %13:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %2 %4 %10 %11 %12
                                %14:dimension<1> = constant [value=1]
                                %15:dimension<1> = constant [value=1]
                                %16:dimension<1> = constant [value=1]
                                %17:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %3 %4 %14 %15 %16
                                %18:f32[batch, 1, 1, 1] = composite [
                                    key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, \
                                        local_window = none, implementation = 0, dropout_rate = none, \
                                        residual = false, bias = false, mask = false, query_sequence_lengths = false, \
                                        key_value_sequence_lengths = false, dropout_state = false}\",
                                ] %9 %13 %17 [
                                    primal={
                                        lambda %0:f32[batch, 1, 1, 1], %1:f32[batch, 1, 1, 1], %2:f32[batch, 1, 1, 1] .
                                        let %3:f32[batch, 1, 1, 1] = dot [
                                            dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                                rhs_batching=[0, 2]),
                                        ] %0 %1
                                            %4:f32[batch, 1, 1, 1] = fill_like [value=1.0] %3
                                            %5:f32[batch, 1, 1, 1] = mul %3 %4
                                            %6:f32[batch, 1, 1] = reduce [kind=max, axes=[3]] %5
                                            %7:f32[batch, 1, 1, 1] = \
                                                broadcast [output_type=f32[batch, 1, 1, 1], output_axes=[0, 1, 2]] %6
                                            %8:f32[batch, 1, 1, 1] = sub %5 %7
                                            %9:f32[batch, 1, 1, 1] = exp %8
                                            %10:f32[batch, 1, 1] = reduce [kind=sum, axes=[3]] %9
                                            %11:f32[batch, 1, 1, 1] = \
                                                broadcast [output_type=f32[batch, 1, 1, 1], output_axes=[0, 1, 2]] %10
                                            %12:f32[batch, 1, 1, 1] = div %9 %11
                                            %13:f32[batch, 1, 1, 1] = dot [
                                                dimensions=(lhs_contracting=[3], rhs_contracting=[1], \
                                                    lhs_batching=[0, 1], rhs_batching=[0, 2]),
                                            ] %12 %2
                                            %14:f32[batch, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %13
                                        in (%14)
                                    },
                                ]
                                %19:dimension<1> = constant [value=1]
                                %20:dimension<1> = constant [value=1]
                                %21:dimension<1> = constant [value=1]
                                %22:dimension<1> = constant [value=1]
                                %23:f32[batch, 1, 1, 1, 1] = \
                                    reshape [requires_runtime_assertion=false] %18 %4 %19 %20 %21 %22
                            in (%23)
                        },
                    ]
                in (%8)
            "}
            .trim_end(),
        );

        // A dynamic mapped extent and a dynamic batch extent fold into their product, which is one dimension that
        // every folded input shares, so the folded call validates and the staged program computes the unbatched
        // attention of every batch item.
        let mapped_variable = DimensionVariable::new("mapped", DimensionBounds::new(1, Some(4)).unwrap());
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(4)).unwrap());
        let r#type = |length: usize| -> ArrayIrType {
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![
                    Dimension::Dynamic(mapped_variable.clone()),
                    Dimension::Dynamic(batch_variable.clone()),
                    Dimension::Static(length),
                    Dimension::Static(1),
                    Dimension::Static(1),
                ]),
            )
            .into()
        };
        let axes = AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0));
        let (_, program) = ArrayIrContext::trace(
            |inputs: AttentionInputs<_>| {
                batch(|inputs: AttentionInputs<_>| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)
            },
            AttentionInputs::new(r#type(1), r#type(2), r#type(2)),
        )
        .unwrap();
        let query = [1.0, -1.0, 0.5, 2.0, -0.5, 1.5];
        let key = [0.25, -0.5, 1.5, 1.0, -1.0, 0.75, 2.0, 0.5, -0.25, 1.25, 0.0, -1.5];
        let value = [3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0, 31.0, 37.0, 41.0];
        let expected = (0..3)
            .flat_map(|item| {
                let (output, _) = attend(
                    AttentionInputs::new(
                        f32_array(&[2, 1, 1, 1], &query[2 * item..2 * item + 2]),
                        f32_array(&[2, 2, 1, 1], &key[4 * item..4 * item + 4]),
                        f32_array(&[2, 2, 1, 1], &value[4 * item..4 * item + 4]),
                    ),
                    AttentionConfiguration::new(),
                );
                output.elements::<f32>().unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            program.interpret(AttentionInputs::new(
                ArrayIrValue::Array(f32_array(&[3, 2, 1, 1, 1], &query)),
                ArrayIrValue::Array(f32_array(&[3, 2, 2, 1, 1], &key)),
                ArrayIrValue::Array(f32_array(&[3, 2, 2, 1, 1], &value)),
            )),
            Ok(ArrayIrValue::Array(f32_array(&[3, 2, 1, 1, 1], expected.as_slice()))),
        );
    }

    #[test]
    fn test_dot_product_attention_batching_dynamic_shapes_sharding() {
        // A reshape that changes a dynamic extent cannot infer where the dimensions that it keeps are placed, so the
        // folds and unfolds of the batching rule state the placement of their outputs. Here the logical batch extent
        // is dynamic and a mesh axis shards the heads, which neither the fold nor the unfold changes.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let batch_variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(4)).unwrap());
        let placed = |dimensions: Vec<Dimension>, head_axis: usize| {
            let mut placements = vec![ShardingDimension::Replicated; dimensions.len()];
            placements[head_axis] = ShardingDimension::sharded(["x"]);
            ArrayType::new(DataType::F32, Shape::new(dimensions))
                .with_sharding(Sharding::new(mesh.clone(), placements).unwrap())
                .unwrap()
        };
        let batch_dimension = Dimension::Dynamic(batch_variable);
        let packed_type = placed(
            vec![
                Dimension::Static(2),
                batch_dimension.clone(),
                Dimension::Static(1),
                Dimension::Static(2),
                Dimension::Static(1),
            ],
            3,
        );
        let residual_type =
            placed(vec![Dimension::Static(2), batch_dimension, Dimension::Static(1), Dimension::Static(2)], 3);
        let function = dot_product_attention::<ArrayIrContext>(AttentionConfiguration::new());
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = context.input(packed_type.clone().into());
        let (output, residual) = batch(
            |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
            AttentionInputs::new(input.clone(), input.clone(), input),
            AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayIrType::Array(packed_type));
        assert_eq!(residual.r#type().as_ref(), &ArrayIrType::Array(residual_type));
    }

    #[test]
    fn test_dot_product_attention_batching_dropout() {
        // Definitions with dropout batch structurally, so every batch item applies the mask of its unbatched call.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_dropout(0.5));
        let state = Array::from_elements(ArrayType::new_static(DataType::U64, [2]), &[0u64, 1]).unwrap();
        let zeros = f32_array(&[1, 4, 1, 1], &[0.0; 4]);
        let expected = function
            .call(AttentionInputs {
                dropout_state: Some(state.clone()),
                ..AttentionInputs::new(zeros.clone(), zeros, f32_array(&[1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0]))
            })
            .unwrap();
        let inputs = AttentionInputs {
            dropout_state: Some(state),
            ..AttentionInputs::new(
                f32_array(&[2, 1, 4, 1, 1], &[0.0; 8]),
                f32_array(&[2, 1, 4, 1, 1], &[0.0; 8]),
                f32_array(&[2, 1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
            )
        };
        let axes = AttentionInputs {
            dropout_state: Some(BatchAxis::replicated()),
            ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0))
        };
        let output =
            batch(|inputs: AttentionInputs<_>| function.call(inputs), inputs, axes, BatchAxis::new(0), None).unwrap();
        assert_eq!(output.to_f64s(), [expected.to_f64s(), expected.to_f64s()].concat());
    }

    #[test]
    fn test_dot_product_attention_batching_fused_dropout() {
        // Batching a call with dropout and the fused implementation would silently replace the fused kernels and
        // their dropout mask with the portable body, so a call with a mapped input is rejected.
        let configuration =
            AttentionConfiguration::new().with_dropout(0.5).with_implementation(AttentionImplementation::Fused);
        let function = dot_product_attention::<ArrayContext>(configuration);
        let ones = f32_array(&[2, 1, 1, 1, 1], &[1.0, 1.0]);
        let result = batch(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs::new(ones.clone(), ones.clone(), ones),
            AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert!(matches!(
            result,
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "named composite `{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` with dropout cannot be batched with \
                     the fused implementation, because a batched call cannot use the fused kernels or reproduce \
                     their dropout mask; use the portable implementation or a dropout state",
                ),
        ));

        // A call whose inputs are all replicated computes one result that every batch item shares, so it stages the
        // unbatched definition and its result equals that of the unbatched call.
        let ones = f32_array(&[1, 1, 1, 1], &[1.0]);
        let inputs = AttentionInputs::new(ones.clone(), ones, f32_array(&[1, 1, 1, 1], &[3.0]));
        let attended = function.call(inputs.clone()).unwrap().to_f64s()[0] as f32;
        let output = batch(
            |(inputs, offsets): (AttentionInputs<_>, _)| function.call(inputs)?.add(&offsets),
            (inputs, f32_array(&[2, 1, 1, 1, 1], &[10.0, 20.0])),
            (
                AttentionInputs::new(BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::replicated()),
                BatchAxis::new(0),
            ),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, f32_array(&[2, 1, 1, 1, 1], &[10.0 + attended, 20.0 + attended]));
    }

    #[test]
    fn test_dot_product_attention_batching_after_drop() {
        // A staged call keeps its definition after the function that staged it is dropped, and batching it later
        // stages an equivalent folded definition.
        let r#type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]);
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()),
        )
        .unwrap();
        drop(function);
        let program = program.into_flat_program();
        let ones = f32_array(&[2, 1, 1, 1, 1], &[1.0, 1.0]);
        let values = f32_array(&[2, 1, 1, 1, 1], &[5.0, 7.0]);
        let output = batch(
            |inputs: Vec<_>| {
                let context = inputs[0].domain();
                Ok(program.entry_region_ref().interpret_in_context(&context, inputs)?.remove(0))
            },
            vec![ones.clone(), ones.clone(), values.clone()],
            vec![BatchAxis::new(0); 3],
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, values);

        // A call that also returns its residual stages an equivalent folded definition that returns it as well.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
        )
        .unwrap();
        drop(function);
        let program = program.into_flat_program();
        let (output, residual) = batch(
            |inputs: Vec<_>| {
                let context = inputs[0].domain();
                let mut outputs = program.entry_region_ref().interpret_in_context(&context, inputs)?;
                let residual = outputs.remove(1);
                Ok((outputs.remove(0), residual))
            },
            vec![ones.clone(), ones, values.clone()],
            vec![BatchAxis::new(0); 3],
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        assert_eq!(output, values);
        assert_eq!(residual, f32_array(&[2, 1, 1, 1], &[1.0, 1.0]));
    }

    #[test]
    fn test_dot_product_attention_differentiation() {
        // The named definitions have no derivative rules, so the inference function rejects differentiation.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let ones = f32_array(&[1, 1, 1, 1], &[1.0]);
        assert!(matches!(
            differentiate_at(AttentionInputs::new(ones.clone(), ones.clone(), ones))
                .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum)),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "cannot differentiate a `{COMPOSITE_OPERATION_NAME}` call of \
                     `{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` that has no derivative rule",
                ),
        ));
    }

    #[test]
    fn test_differentiable_dot_product_attention() {
        let r#type = ArrayType::new_static(DataType::F64, [1, 1, 1, 1]);
        let lengths = Array::from_elements(ArrayType::new_static(DataType::I32, [1]), &[1i32]).unwrap();
        let function =
            differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_scale(0.5));

        // An attention distribution over one key is identically one, so the output is the value. The query, key, and
        // bias therefore have zero gradients, the value has a unit gradient, and the mask and the sequence lengths
        // have structural zero gradients.
        let inputs = AttentionInputs {
            query: Array::from_elements(r#type.clone(), &[2.0f64]).unwrap(),
            key: Array::from_elements(r#type.clone(), &[3.0f64]).unwrap(),
            value: Array::from_elements(r#type.clone(), &[5.0f64]).unwrap(),
            bias: Some(Array::from_elements(ArrayType::scalar(DataType::F64), &[1.0f64]).unwrap()),
            mask: Some(Array::from_elements(ArrayType::scalar(DataType::Boolean), &[true]).unwrap()),
            query_sequence_lengths: Some(lengths.clone()),
            key_value_sequence_lengths: Some(lengths),
            dropout_state: None,
        };
        let (output, gradients) = differentiate_at(inputs)
            .value_and_gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap();
        assert_eq!(output.to_f64s(), vec![5.0]);
        assert_eq!(gradients.query.to_f64s(), vec![0.0]);
        assert_eq!(gradients.key.to_f64s(), vec![0.0]);
        assert_eq!(gradients.value.to_f64s(), vec![1.0]);
        assert_eq!(gradients.bias.unwrap().to_f64s(), vec![0.0]);
        assert_eq!(gradients.mask.unwrap().r#type().data_type(), DataType::Zero);
        assert_eq!(gradients.query_sequence_lengths.unwrap().r#type().data_type(), DataType::Zero);
        assert_eq!(gradients.key_value_sequence_lengths.unwrap().r#type().data_type(), DataType::Zero);

        // An integer bias is a legal forward input whose cotangent space is zero-dimensional.
        let inputs = AttentionInputs {
            bias: Some(Array::from_elements(ArrayType::scalar(DataType::I32), &[0i32]).unwrap()),
            ..AttentionInputs::new(
                Array::from_elements(r#type.clone(), &[2.0f64]).unwrap(),
                Array::from_elements(r#type.clone(), &[3.0f64]).unwrap(),
                Array::from_elements(r#type, &[5.0f64]).unwrap(),
            )
        };
        let gradients = differentiate_at(inputs)
            .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap();
        assert_eq!(gradients.value.to_f64s(), vec![1.0]);
        assert_eq!(gradients.bias.unwrap().r#type().data_type(), DataType::Zero);

        // Reverse mode stages the forward definition with its residual, followed by the backward definition.
        let r#type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]);
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| {
                differentiate_at(inputs)
                    .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
                    .map_err(ProgramError::from)
            },
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1] .
                let %3:f32[1, 1, 1, 1], %4:f32[1, 1, 1] = composite [
                    key=\"ryft.dot_product_attention.v1 {scale = some(0.5), causal = false, local_window = none, \
                        implementation = 1, dropout_rate = none, residual = true, bias = false, mask = false, \
                        query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                ] %0 %1 %2 [
                    primal={
                        lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1] .
                        let %3:f32[1, 1, 1, 1] = dot [
                            dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                rhs_batching=[0, 2]),
                        ] %0 %1
                            %4:f32[] = constant [value=0.5]
                            %5:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %4
                            %6:f32[1, 1, 1, 1] = mul %3 %5
                            %7:f32[1, 1, 1] = reduce [kind=max, axes=[3]] %6
                            %8:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %7
                            %9:f32[1, 1, 1, 1] = sub %6 %8
                            %10:f32[1, 1, 1, 1] = exp %9
                            %11:f32[1, 1, 1] = reduce [kind=sum, axes=[3]] %10
                            %12:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %11
                            %13:f32[1, 1, 1, 1] = div %10 %12
                            %14:f32[1, 1, 1, 1] = dot [
                                dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                    rhs_batching=[0, 2]),
                            ] %13 %2
                            %15:f32[1, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %14
                            %16:f32[1, 1, 1] = log %11
                            %17:f32[1, 1, 1] = add %7 %16
                            %18:f32[1, 1, 1] = transpose [permutation=[0, 2, 1]] %17
                            %19:f32[1, 1, 1] = stop_gradient %18
                        in (%15, %19)
                    },
                ]
                    %5:f32[] = reduce [kind=sum, axes=[0, 1, 2, 3]] %3
                    %6:f32[] = one [type=f32[]]
                    %7:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %6
                    %8:f32[1, 1, 1, 1], %9:f32[1, 1, 1, 1], %10:f32[1, 1, 1, 1] = composite [
                        key=\"ryft.dot_product_attention_backward.v1 {scale = some(0.5), causal = false, \
                            local_window = none, implementation = 1, dropout_rate = none, bias = false, mask = false, \
                            query_sequence_lengths = false, key_value_sequence_lengths = false, \
                            dropout_state = false}\",
                    ] %0 %1 %2 %3 %4 %7 [
                        primal={
                            lambda \
                                %0:f32[1, 1, 1, 1], \
                                %1:f32[1, 1, 1, 1], \
                                %2:f32[1, 1, 1, 1], \
                                %3:f32[1, 1, 1, 1], \
                                %4:f32[1, 1, 1], \
                                %5:f32[1, 1, 1, 1] .
                            let %6:f32[1, 1, 1, 1] = dot [
                                dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                    rhs_batching=[0, 2]),
                            ] %0 %1
                                %7:f32[] = constant [value=0.5]
                                %8:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %7
                                %9:f32[1, 1, 1, 1] = mul %6 %8
                                %10:f32[1, 1, 1] = reduce [kind=max, axes=[3]] %9
                                %11:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %10
                                %12:f32[1, 1, 1, 1] = sub %9 %11
                                %13:f32[1, 1, 1, 1] = exp %12
                                %14:f32[1, 1, 1] = reduce [kind=sum, axes=[3]] %13
                                %15:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %14
                                %16:f32[1, 1, 1, 1] = div %13 %15
                                %17:i32[1, 1, 1, 1] = iota [type=i32[1, 1, 1, 1], dimension=3]
                                %18:i32[] = constant [value=2147483647]
                                %19:i32[1, 1, 1, 1] = broadcast [output_type=i32[1, 1, 1, 1], output_axes=[]] %18
                                %20:bool[1, 1, 1, 1] = compare [direction=Equal] %9 %11
                                %21:i32[1, 1, 1, 1] = select %20 %17 %19
                                %22:i32[1, 1, 1] = reduce [kind=min, axes=[3]] %21
                                %23:i32[1, 1, 1, 1] = broadcast [output_type=i32[1, 1, 1, 1], output_axes=[0, 1, 2]] %22
                                %24:bool[1, 1, 1, 1] = compare [direction=Equal] %17 %23
                                %25:f32[1, 1, 1, 1] = dot [
                                    dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                        rhs_batching=[0, 2]),
                                ] %5 %2
                                %26:f32[] = constant [value=0.0]
                                %27:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %26
                                %28:f32[] = constant [value=0.5]
                                %29:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %28
                                %30:f32[1, 1, 1, 1] = mul %25 %29
                                %31:f32[1, 1, 1, 1] = select %24 %30 %27
                                %32:f32[1, 1, 1] = reduce [kind=sum, axes=[3]] %31
                                %33:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %32
                                %34:f32[1, 1, 1, 1] = sub %30 %33
                                %35:f32[1, 1, 1, 1] = mul %16 %34
                                %36:f32[1, 1, 1] = reduce [kind=sum, axes=[3]] %35
                                %37:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %36
                                %38:f32[1, 1, 1, 1] = add %16 %16
                                %39:f32[1, 1, 1, 1] = sub %34 %37
                                %40:f32[1, 1, 1, 1] = mul %38 %39
                                %41:f32[1, 1, 1, 1] = select %24 %27 %40
                                %42:f32[1, 1, 1] = reduce [kind=sum, axes=[3]] %41
                                %43:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %42
                                %44:f32[1, 1, 1, 1] = sub %27 %43
                                %45:f32[1, 1, 1, 1] = select %24 %44 %41
                                %46:f32[] = constant [value=0.5]
                                %47:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %46
                                %48:f32[1, 1, 1, 1] = mul %45 %47
                                %49:f32[1, 1, 1, 1] = dot [
                                    dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                        rhs_batching=[0, 2]),
                                ] %48 %1
                                %50:f32[1, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %49
                                %51:f32[1, 1, 1, 1] = dot [
                                    dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0, 1], \
                                        rhs_batching=[0, 2]),
                                ] %48 %0
                                %52:f32[1, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %51
                                %53:f32[1, 1, 1, 1] = dot [
                                    dimensions=(lhs_contracting=[2], rhs_contracting=[1], lhs_batching=[0, 1], \
                                        rhs_batching=[0, 2]),
                                ] %16 %5
                                %54:f32[1, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %53
                            in (%50, %52, %54)
                        },
                    ]
                in (%8, %9, %10)
            "}
            .trim_end(),
        );

        // An unused output has a structurally zero cotangent, so reverse mode stages no backward definition. The
        // query and key cotangents are zeros, and the value cotangent comes only from the direct use of the value.
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| {
                differentiate_at(inputs)
                    .gradient(|inputs| {
                        function.call(inputs.clone())?;
                        inputs.value.reduce(&[0, 1, 2, 3], ReductionKind::Sum)
                    })
                    .map_err(ProgramError::from)
            },
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1] .
                let %3:f32[1, 1, 1, 1], %4:f32[1, 1, 1] = composite [
                    key=\"ryft.dot_product_attention.v1 {scale = some(0.5), causal = false, local_window = none, \
                        implementation = 1, dropout_rate = none, residual = true, bias = false, mask = false, \
                        query_sequence_lengths = false, key_value_sequence_lengths = false, dropout_state = false}\",
                ] %0 %1 %2 [
                    primal={
                        lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1] .
                        let %3:f32[1, 1, 1, 1] = dot [
                            dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                rhs_batching=[0, 2]),
                        ] %0 %1
                            %4:f32[] = constant [value=0.5]
                            %5:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %4
                            %6:f32[1, 1, 1, 1] = mul %3 %5
                            %7:f32[1, 1, 1] = reduce [kind=max, axes=[3]] %6
                            %8:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %7
                            %9:f32[1, 1, 1, 1] = sub %6 %8
                            %10:f32[1, 1, 1, 1] = exp %9
                            %11:f32[1, 1, 1] = reduce [kind=sum, axes=[3]] %10
                            %12:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %11
                            %13:f32[1, 1, 1, 1] = div %10 %12
                            %14:f32[1, 1, 1, 1] = dot [
                                dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                    rhs_batching=[0, 2]),
                            ] %13 %2
                            %15:f32[1, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %14
                            %16:f32[1, 1, 1] = log %11
                            %17:f32[1, 1, 1] = add %7 %16
                            %18:f32[1, 1, 1] = transpose [permutation=[0, 2, 1]] %17
                            %19:f32[1, 1, 1] = stop_gradient %18
                        in (%15, %19)
                    },
                ]
                    %5:f32[] = reduce [kind=sum, axes=[0, 1, 2, 3]] %2
                    %6:f32[] = one [type=f32[]]
                    %7:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %6
                    %8:f32[1, 1, 1, 1] = zero [type=f32[1, 1, 1, 1]]
                    %9:f32[1, 1, 1, 1] = zero [type=f32[1, 1, 1, 1]]
                in (%8, %9, %7)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_differentiable_dot_product_attention_implementation_selection() {
        // The differentiable function selects the portable implementation for automatic requests and keeps explicit
        // ones, which the keys of its staged composite and of the forward definition record.
        let r#type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]);
        for (requested, selected, code) in [
            (AttentionImplementation::Automatic, AttentionImplementation::Portable, 1),
            (AttentionImplementation::Portable, AttentionImplementation::Portable, 1),
            (AttentionImplementation::Fused, AttentionImplementation::Fused, 2),
        ] {
            let configuration = AttentionConfiguration::new().with_implementation(requested);
            let function = differentiable_dot_product_attention::<ArrayContext>(configuration);
            assert_eq!(function.configuration(), configuration.with_implementation(selected));
            let (_, program) = ArrayContext::trace(
                |inputs| function.call(inputs),
                AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()),
            )
            .unwrap();
            assert_eq!(
                program.to_string(),
                formatdoc! {"
                    lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1] .
                    let %3:f32[1, 1, 1, 1] = composite [
                        key=\"ryft.dot_product_attention_differentiable.v1 {{scale = none, causal = false, \
                            local_window = none, implementation = {code}, dropout_rate = none, bias = false, \
                            mask = false, query_sequence_lengths = false, key_value_sequence_lengths = false, \
                            dropout_state = false}}\",
                    ] %0 %1 %2 [
                        primal={{
                            lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1] .
                            let %3:f32[1, 1, 1, 1] = composite [
                                key=\"ryft.dot_product_attention.v1 {{scale = none, causal = false, \
                                    local_window = none, implementation = {code}, dropout_rate = none, \
                                    residual = false, bias = false, mask = false, query_sequence_lengths = false, \
                                    key_value_sequence_lengths = false, dropout_state = false}}\",
                            ] %0 %1 %2 [
                                primal={{
                                    lambda %0:f32[1, 1, 1, 1], %1:f32[1, 1, 1, 1], %2:f32[1, 1, 1, 1] .
                                    let %3:f32[1, 1, 1, 1] = dot [
                                        dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                            rhs_batching=[0, 2]),
                                    ] %0 %1
                                        %4:f32[] = constant [value=1.0]
                                        %5:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %4
                                        %6:f32[1, 1, 1, 1] = mul %3 %5
                                        %7:f32[1, 1, 1] = reduce [kind=max, axes=[3]] %6
                                        %8:f32[1, 1, 1, 1] = \
                                            broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %7
                                        %9:f32[1, 1, 1, 1] = sub %6 %8
                                        %10:f32[1, 1, 1, 1] = exp %9
                                        %11:f32[1, 1, 1] = reduce [kind=sum, axes=[3]] %10
                                        %12:f32[1, 1, 1, 1] = \
                                            broadcast [output_type=f32[1, 1, 1, 1], output_axes=[0, 1, 2]] %11
                                        %13:f32[1, 1, 1, 1] = div %10 %12
                                        %14:f32[1, 1, 1, 1] = dot [
                                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                                rhs_batching=[0, 2]),
                                        ] %13 %2
                                        %15:f32[1, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %14
                                    in (%15)
                                }},
                            ]
                            in (%3)
                        }},
                    ]
                    in (%3)
                "}
                .trim_end(),
            );
        }
    }

    #[test]
    fn test_differentiable_dot_product_attention_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let invariant = ArrayType::new_static(DataType::F32, [1, 1, 1, 1])
            .with_sharding(Sharding::replicated(mesh.clone(), 4))
            .unwrap();
        let varying = invariant
            .clone()
            .with_sharding(Sharding::replicated(mesh.clone(), 4).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let function = differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new());

        // The function aligns the manual variation of its inputs before the composite boundary, so the transposes of
        // those alignments accumulate the cotangent contributions of the invariant inputs.
        let (output_type, program) = DomainTracingContext::<ArrayContext>::trace_with_named_axes(
            |inputs| function.call(inputs),
            AttentionInputs::new(invariant.clone(), varying.clone(), invariant),
            vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: true })],
        )
        .unwrap();
        assert_eq!(output_type, varying);
        let varying_rank_4 = rendered_sharding(4, true);
        let varying_rank_3 = rendered_sharding(3, true);
        let varying_rank_0 = rendered_sharding(0, true);
        let invariant_rank_4 = rendered_sharding(4, false);
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda \
                    %0:f32[1, 1, 1, 1]{invariant_rank_4}, \
                    %1:f32[1, 1, 1, 1]{varying_rank_4}, \
                    %2:f32[1, 1, 1, 1]{invariant_rank_4} .
                let %3:f32[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %0
                    %4:f32[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %2
                    %5:f32[1, 1, 1, 1]{varying_rank_4} = composite [
                        key=\"ryft.dot_product_attention_differentiable.v1 {{scale = none, causal = false, \
                            local_window = none, implementation = 1, dropout_rate = none, bias = false, mask = false, \
                            query_sequence_lengths = false, key_value_sequence_lengths = false, \
                            dropout_state = false}}\",
                    ] %3 %1 %4 [
                        primal={{
                            lambda \
                                %0:f32[1, 1, 1, 1]{varying_rank_4}, \
                                %1:f32[1, 1, 1, 1]{varying_rank_4}, \
                                %2:f32[1, 1, 1, 1]{varying_rank_4} .
                            let %3:f32[1, 1, 1, 1]{varying_rank_4} = composite [
                                key=\"ryft.dot_product_attention.v1 {{scale = none, causal = false, \
                                    local_window = none, implementation = 1, dropout_rate = none, residual = false, \
                                    bias = false, mask = false, query_sequence_lengths = false, \
                                    key_value_sequence_lengths = false, dropout_state = false}}\",
                            ] %0 %1 %2 [
                                primal={{
                                    lambda \
                                        %0:f32[1, 1, 1, 1]{varying_rank_4}, \
                                        %1:f32[1, 1, 1, 1]{varying_rank_4}, \
                                        %2:f32[1, 1, 1, 1]{varying_rank_4} .
                                    let %3:f32[1, 1, 1, 1]{varying_rank_4} = dot [
                                        dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                            rhs_batching=[0, 2]),
                                    ] %0 %1
                                        %4:f32[]{varying_rank_0} = constant [value=1.0]
                                        %5:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                            output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                            output_axes=[],
                                        ] %4
                                        %6:f32[1, 1, 1, 1]{varying_rank_4} = mul %3 %5
                                        %7:f32[1, 1, 1]{varying_rank_3} = reduce [kind=max, axes=[3]] %6
                                        %8:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                            output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                            output_axes=[0, 1, 2],
                                        ] %7
                                        %9:f32[1, 1, 1, 1]{varying_rank_4} = sub %6 %8
                                        %10:f32[1, 1, 1, 1]{varying_rank_4} = exp %9
                                        %11:f32[1, 1, 1]{varying_rank_3} = reduce [kind=sum, axes=[3]] %10
                                        %12:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                            output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                            output_axes=[0, 1, 2],
                                        ] %11
                                        %13:f32[1, 1, 1, 1]{varying_rank_4} = div %10 %12
                                        %14:f32[1, 1, 1, 1]{varying_rank_4} = dot [
                                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                                rhs_batching=[0, 2]),
                                        ] %13 %2
                                        %15:f32[1, 1, 1, 1]{varying_rank_4} = transpose [permutation=[0, 2, 1, 3]] %14
                                    in (%15)
                                }},
                            ]
                            in (%3)
                        }},
                    ]
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_differentiable_dot_product_attention_batching() {
        // Batched differentiation folds the mapped axis into the forward and backward definitions, including the
        // reduction of the bias cotangent over the axes that its bias broadcasts, so the batched gradients equal the
        // gradients of every batch item.
        let query = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.2, 0.1];
        let key = [0.5, 0.2, -0.3, 0.7, 0.1, -0.6, 0.4, -0.1, 0.3, 0.2, -0.4, 0.5];
        let value = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.5, 2.5, -3.0, 1.5, 0.0];
        let bias = [0.25, -0.5, 1.0, -0.75, 0.5, 0.0];
        let array = |dimensions: &[usize], elements: &[f64]| {
            Array::from_elements(ArrayType::new_static(DataType::F64, dimensions.to_vec()), elements).unwrap()
        };
        let function = differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let inputs = AttentionInputs {
            bias: Some(array(&[2, 3], &bias)),
            ..AttentionInputs::new(
                array(&[2, 1, 2, 1, 2], &query),
                array(&[2, 1, 3, 1, 2], &key),
                array(&[2, 1, 3, 1, 2], &value),
            )
        };
        let axes = AttentionInputs {
            bias: Some(BatchAxis::new(0)),
            ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0))
        };
        let batched = differentiate_at(inputs)
            .gradient(|inputs| {
                let outputs = batch(|inputs| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)?;
                outputs.reduce(&[0, 1, 2, 3, 4], ReductionKind::Sum)
            })
            .unwrap();
        for item in 0..2 {
            let inputs = AttentionInputs {
                bias: Some(array(&[3], &bias[3 * item..3 * item + 3])),
                ..AttentionInputs::new(
                    array(&[1, 2, 1, 2], &query[4 * item..4 * item + 4]),
                    array(&[1, 3, 1, 2], &key[6 * item..6 * item + 6]),
                    array(&[1, 3, 1, 2], &value[6 * item..6 * item + 6]),
                )
            };
            let expected = differentiate_at(inputs)
                .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
                .unwrap();
            for (actual, expected, size) in [
                (&batched.query, &expected.query, 4),
                (&batched.key, &expected.key, 6),
                (&batched.value, &expected.value, 6),
                (batched.bias.as_ref().unwrap(), expected.bias.as_ref().unwrap(), 3),
            ] {
                actual.to_f64s()[size * item..size * (item + 1)]
                    .iter()
                    .zip(expected.to_f64s())
                    .for_each(|(actual, expected)| assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6));
            }
        }

        // Rank-3 attention has no batch axis, so its mapped axis becomes the batch axis without a product, and a
        // rank-4 bias with a unit batch axis is normalized without that axis.
        let bias = [0.25, -0.5, 1.0, -0.75, 0.5, 0.0, 0.1, 0.2, -0.3, 0.4, -0.6, 0.8];
        let inputs = AttentionInputs {
            bias: Some(array(&[2, 1, 1, 2, 3], &bias)),
            ..AttentionInputs::new(
                array(&[2, 2, 1, 2], &query),
                array(&[2, 3, 1, 2], &key),
                array(&[2, 3, 1, 2], &value),
            )
        };
        let batched = differentiate_at(inputs)
            .gradient(|inputs| {
                let outputs = batch(|inputs| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)?;
                outputs.reduce(&[0, 1, 2, 3], ReductionKind::Sum)
            })
            .unwrap();
        for item in 0..2 {
            let inputs = AttentionInputs {
                bias: Some(array(&[1, 1, 2, 3], &bias[6 * item..6 * item + 6])),
                ..AttentionInputs::new(
                    array(&[2, 1, 2], &query[4 * item..4 * item + 4]),
                    array(&[3, 1, 2], &key[6 * item..6 * item + 6]),
                    array(&[3, 1, 2], &value[6 * item..6 * item + 6]),
                )
            };
            let expected = differentiate_at(inputs)
                .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2], ReductionKind::Sum))
                .unwrap();
            for (actual, expected, size) in [
                (&batched.query, &expected.query, 4),
                (&batched.key, &expected.key, 6),
                (&batched.value, &expected.value, 6),
                (batched.bias.as_ref().unwrap(), expected.bias.as_ref().unwrap(), 6),
            ] {
                actual.to_f64s()[size * item..size * (item + 1)]
                    .iter()
                    .zip(expected.to_f64s())
                    .for_each(|(actual, expected)| assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6));
            }
        }

        // A replicated rank-2 bias of a rank-3 call with two heads is broadcast over the mapped axis and the heads,
        // so its batched gradient is the sum of its gradients over the batch items.
        let bias = &bias[..6];
        let inputs = AttentionInputs {
            bias: Some(array(&[2, 3], bias)),
            ..AttentionInputs::new(
                array(&[2, 2, 2, 1], &query),
                array(&[2, 3, 2, 1], &key),
                array(&[2, 3, 2, 1], &value),
            )
        };
        let axes = AttentionInputs {
            bias: Some(BatchAxis::replicated()),
            ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0))
        };
        let batched = differentiate_at(inputs)
            .gradient(|inputs| {
                let outputs = batch(|inputs| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)?;
                outputs.reduce(&[0, 1, 2, 3], ReductionKind::Sum)
            })
            .unwrap();
        let mut expected_bias = [0.0; 6];
        for item in 0..2 {
            let inputs = AttentionInputs {
                bias: Some(array(&[2, 3], bias)),
                ..AttentionInputs::new(
                    array(&[2, 2, 1], &query[4 * item..4 * item + 4]),
                    array(&[3, 2, 1], &key[6 * item..6 * item + 6]),
                    array(&[3, 2, 1], &value[6 * item..6 * item + 6]),
                )
            };
            let expected = differentiate_at(inputs)
                .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2], ReductionKind::Sum))
                .unwrap();
            for (actual, expected, size) in [
                (&batched.query, &expected.query, 4),
                (&batched.key, &expected.key, 6),
                (&batched.value, &expected.value, 6),
            ] {
                actual.to_f64s()[size * item..size * (item + 1)]
                    .iter()
                    .zip(expected.to_f64s())
                    .for_each(|(actual, expected)| assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6));
            }
            expected_bias
                .iter_mut()
                .zip(expected.bias.as_ref().unwrap().to_f64s())
                .for_each(|(sum, gradient)| *sum += gradient);
        }
        batched
            .bias
            .as_ref()
            .unwrap()
            .to_f64s()
            .iter()
            .zip(expected_bias)
            .for_each(|(actual, expected)| assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6));
    }

    #[test]
    fn test_differentiable_dot_product_attention_batching_sharding() {
        // A mesh axis that shards the logical batch axis cannot shard the batch axis that the mapped axis is folded
        // into, so the batching rule replicates the inputs along it for the folded call and returns the outputs and
        // the cotangents to its placement. The placement only changes where the elements live, so the batched outputs
        // and gradients have the elements of the ones over inputs without a sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(
            mesh,
            vec![
                ShardingDimension::Replicated,
                ShardingDimension::sharded(["x"]),
                ShardingDimension::Replicated,
                ShardingDimension::Replicated,
                ShardingDimension::Replicated,
            ],
        )
        .unwrap();
        let plain_type = ArrayType::new_static(DataType::F32, [2, 2, 2, 1, 2]);
        let placed_type = plain_type.clone().with_sharding(sharding).unwrap();
        let inputs = |r#type: &ArrayType| {
            let array = |start: f32| {
                let elements = (0..16).map(|index| start + 0.25 * index as f32).collect::<Vec<_>>();
                Array::from_elements(r#type.clone(), elements.as_slice()).unwrap()
            };
            AttentionInputs::new(array(-1.0), array(0.5), array(-2.0))
        };
        let axes = AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0));

        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let outputs = |r#type: &ArrayType| -> (Array, Array) {
            batch(
                |inputs: AttentionInputs<_>| function.call_with_residual(inputs),
                inputs(r#type),
                axes.clone(),
                (BatchAxis::new(0), BatchAxis::new(0)),
                None,
            )
            .unwrap()
        };
        let (output, residual) = outputs(&placed_type);
        let (plain_output, plain_residual) = outputs(&plain_type);
        assert_eq!(output.r#type().as_ref(), &placed_type);
        assert_eq!(output.elements::<f32>().unwrap(), plain_output.elements::<f32>().unwrap());
        assert_eq!(residual.r#type().sharding().unwrap().dimensions()[1], ShardingDimension::sharded(["x"]));
        assert_eq!(residual.elements::<f32>().unwrap(), plain_residual.elements::<f32>().unwrap());

        // Only a rank-4 input has its logical batch axis after the mapped axis. A rank-3 key or value has its sequence
        // axis there, whose placement is not that of the batch axis of the rank-4 query that it attends with.
        let mixed = |sequence_sharded: bool| -> Array {
            let mesh = placed_type.sharding().unwrap().mesh().clone();
            let query_type = ArrayType::new_static(DataType::F32, [2, 1, 1, 1, 1]);
            let key_value_type = ArrayType::new_static(DataType::F32, [2, 2, 1, 1]);
            let mut value_placements = vec![ShardingDimension::Replicated; 4];
            if sequence_sharded {
                value_placements[1] = ShardingDimension::sharded(["x"]);
            }
            let replicated = |r#type: ArrayType| {
                let rank = r#type.rank();
                r#type.with_sharding(Sharding::replicated(mesh.clone(), rank)).unwrap()
            };
            let value_type = key_value_type
                .clone()
                .with_sharding(Sharding::new(mesh.clone(), value_placements).unwrap())
                .unwrap();
            batch(
                |inputs: AttentionInputs<_>| function.call(inputs),
                AttentionInputs::new(
                    Array::from_elements(replicated(query_type), &[0.5f32, -1.0]).unwrap(),
                    Array::from_elements(replicated(key_value_type), &[1.0f32, -1.0, 0.5, 2.0]).unwrap(),
                    Array::from_elements(value_type, &[1.0f32, 3.0, 5.0, 9.0]).unwrap(),
                ),
                axes.clone(),
                BatchAxis::new(0),
                None,
            )
            .unwrap()
        };
        assert_eq!(mixed(true).elements::<f32>().unwrap(), mixed(false).elements::<f32>().unwrap());

        let function = differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let gradients = |r#type: &ArrayType| -> AttentionInputs<Array> {
            differentiate_at(inputs(r#type))
                .gradient(|inputs| {
                    let outputs = batch(|inputs| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                        .map_err(ProgramError::from)?;
                    outputs.mul(&outputs)?.reduce(&[0, 1, 2, 3, 4], ReductionKind::Sum)
                })
                .unwrap()
        };
        let (gradients, plain_gradients) = (gradients(&placed_type), gradients(&plain_type));
        for (gradient, plain_gradient) in gradients.parameters().zip(plain_gradients.parameters()) {
            assert_eq!(gradient.r#type().as_ref(), &placed_type);
            assert_eq!(gradient.elements::<f32>().unwrap(), plain_gradient.elements::<f32>().unwrap());
        }
    }

    #[test]
    fn test_differentiable_dot_product_attention_batching_layout() {
        // The rules of a call that is batched after it was staged are traced at the types of its batch items, which
        // cannot describe how the packed inputs are stored. Cotangent types have no explicit layout, so the gradients
        // of packed inputs with one are arrays of the same data type, shape, and placement, whichever operations
        // contribute to them (e.g., the batched call alone, or together with the square of the query below), and
        // they have the elements of the gradients over inputs without a layout.
        let plain_type = ArrayType::new_static(DataType::F32, [2, 2, 2, 1, 2]);
        let packed_type = plain_type.clone().with_layout(Layout::Strided(StridedLayout::new(vec![32, 16, 8, 8, 4])));
        let inputs = |r#type: &ArrayType| {
            let array = |start: f32| {
                let elements = (0..16).map(|index| start + 0.25 * index as f32).collect::<Vec<_>>();
                Array::from_elements(r#type.clone(), elements.as_slice()).unwrap()
            };
            AttentionInputs::new(array(-1.0), array(0.5), array(-2.0))
        };
        let axes = AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0));
        let function = differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let gradients = |r#type: &ArrayType, square_query: bool| -> AttentionInputs<Array> {
            differentiate_at(inputs(r#type))
                .gradient(|inputs| {
                    let query = inputs.query.clone();
                    let outputs = batch(|inputs| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                        .map_err(ProgramError::from)?;
                    let loss = outputs.mul(&outputs)?.reduce(&[0, 1, 2, 3, 4], ReductionKind::Sum)?;
                    match square_query {
                        true => loss.add(&query.mul(&query)?.reduce(&[0, 1, 2, 3, 4], ReductionKind::Sum)?),
                        false => Ok(loss),
                    }
                })
                .unwrap()
        };
        for square_query in [false, true] {
            let (gradients, plain_gradients) =
                (gradients(&packed_type, square_query), gradients(&plain_type, square_query));
            for (gradient, plain_gradient) in gradients.parameters().zip(plain_gradients.parameters()) {
                assert!(packed_type.is_interchangeable_with(gradient.r#type().as_ref()));
                assert_eq!(gradient.elements::<f32>().unwrap(), plain_gradient.elements::<f32>().unwrap());
            }
        }
    }

    #[test]
    fn test_differentiable_dot_product_attention_batching_dropout() {
        // Definitions with dropout batch structurally, so every batch item applies the mask of its unbatched call in
        // the forward and in the backward definition, and the batched gradients equal the gradients of every item.
        let function =
            differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_dropout(0.5));
        let state = Array::from_elements(ArrayType::new_static(DataType::U64, [2]), &[0u64, 3]).unwrap();
        let query = [0.5, -1.0, 0.25, 2.0];
        let key = [1.0, -0.5, 0.75, 0.25, -1.5, 0.5, 1.25, -0.25];
        let value = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0];
        let inputs = AttentionInputs {
            dropout_state: Some(state.clone()),
            ..AttentionInputs::new(
                f32_array(&[2, 1, 2, 1, 1], &query),
                f32_array(&[2, 1, 4, 1, 1], &key),
                f32_array(&[2, 1, 4, 1, 1], &value),
            )
        };
        let axes = AttentionInputs {
            dropout_state: Some(BatchAxis::replicated()),
            ..AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0))
        };
        let batched = differentiate_at(inputs)
            .gradient(|inputs| {
                let outputs = batch(|inputs| function.call(inputs), inputs, axes.clone(), BatchAxis::new(0), None)
                    .map_err(ProgramError::from)?;
                outputs.reduce(&[0, 1, 2, 3, 4], ReductionKind::Sum)
            })
            .unwrap();
        for item in 0..2 {
            let inputs = AttentionInputs {
                dropout_state: Some(state.clone()),
                ..AttentionInputs::new(
                    f32_array(&[1, 2, 1, 1], &query[2 * item..2 * item + 2]),
                    f32_array(&[1, 4, 1, 1], &key[4 * item..4 * item + 4]),
                    f32_array(&[1, 4, 1, 1], &value[4 * item..4 * item + 4]),
                )
            };
            let expected = differentiate_at(inputs)
                .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
                .unwrap();
            assert_eq!(batched.query.to_f64s()[2 * item..2 * item + 2], expected.query.to_f64s());
            assert_eq!(batched.key.to_f64s()[4 * item..4 * item + 4], expected.key.to_f64s());
            assert_eq!(batched.value.to_f64s()[4 * item..4 * item + 4], expected.value.to_f64s());
        }

        // Batching the pullback of an unbatched call with dropout and the fused implementation leaves the forward
        // definition unbatched, because its inputs are replicated, and batches only the backward definition, which
        // is rejected for the same reason as a batched forward definition.
        let configuration =
            AttentionConfiguration::new().with_dropout(0.5).with_implementation(AttentionImplementation::Fused);
        let function = differentiable_dot_product_attention::<ArrayContext>(configuration);
        let r#type = ArrayType::new_static(DataType::F32, [1, 1, 1, 1]);
        let (_, program) = ArrayContext::trace(
            |(inputs, cotangent): (AttentionInputs<_>, _)| {
                let (_, pullback) = differentiate_at(inputs).vjp(|inputs| function.call(inputs))?;
                Ok(cotangent_value(pullback.apply(cotangent)?.value))
            },
            (AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone()), r#type),
        )
        .unwrap();
        let program = program.into_flat_program();
        let ones = f32_array(&[1, 1, 1, 1], &[1.0]);
        let result = batch(
            |inputs: Vec<_>| {
                let context = inputs[0].domain();
                Ok(program.entry_region_ref().interpret_in_context(&context, inputs)?.remove(0))
            },
            vec![ones.clone(), ones.clone(), ones, f32_array(&[2, 1, 1, 1, 1], &[1.0, 1.0])],
            vec![BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::replicated(), BatchAxis::new(0)],
            BatchAxis::new(0),
            None,
        );
        assert!(matches!(
            result,
            Err(BatchingError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "named composite `{DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME}` with dropout cannot be batched \
                     with the fused implementation, because a batched call cannot use the fused kernels or reproduce \
                     their dropout mask; use the portable implementation or a dropout state",
                ),
        ));
    }

    #[test]
    fn test_differentiable_dot_product_attention_dropout() {
        // The backward definition applies the dropout mask of the forward definition, which the random state
        // determines, so the analytical gradient agrees with central finite differences of the forward definition.
        // The tolerance reflects the `f32` softmax of attention, which limits the accuracy of the differences.
        let function =
            differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_dropout(0.5));
        let state = Array::from_elements(ArrayType::new_static(DataType::U64, [2]), &[0u64, 3]).unwrap();
        check_gradient!(
            |input, state| {
                // The input packs the `[1, 2, 1, 2]` query, the `[1, 3, 1, 2]` key, and the `[1, 3, 1, 2]` value.
                let inputs = AttentionInputs {
                    dropout_state: Some(state),
                    ..AttentionInputs::new(
                        input.slice_axis(0, 0, 4, 1)?.reshape([1, 2, 1, 2])?,
                        input.slice_axis(0, 4, 10, 1)?.reshape([1, 3, 1, 2])?,
                        input.slice_axis(0, 10, 16, 1)?.reshape([1, 3, 1, 2])?,
                    )
                };
                function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum)
            },
            at = Array::vector(vec![
                0.1f64, -0.2, 0.3, 0.4, 0.5, 0.2, -0.3, 0.7, 0.1, -0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            ])
            .unwrap(),
            with = state,
            step = 5e-3,
            tolerance = 5e-3,
        );
    }

    #[test]
    fn test_differentiable_dot_product_attention_saturated_logits() {
        // The first probability rounds to one, and the stabilized backward body still recovers the small derivative
        // `p (1 - p)` of the dominant logit instead of an exact zero.
        let inputs = AttentionInputs::new(
            f32_array(&[1, 1, 1, 1], &[1.0]),
            f32_array(&[1, 2, 1, 1], &[0.0, -20.0]),
            f32_array(&[1, 2, 1, 1], &[1.0, 0.0]),
        );
        let (output, gradients) = attention_sum_gradient(inputs);
        assert_eq!(output.to_f64s(), vec![1.0]);
        let key_gradient = gradients.key.to_f64s();
        let tail = (-20.0f64).exp();
        let expected = tail / (1.0 + tail).powi(2);
        assert_abs_diff_eq!(key_gradient[0], expected, epsilon = 1e-15);
        assert_eq!(key_gradient[0], -key_gradient[1]);
        assert_abs_diff_eq!(gradients.query.to_f64s()[0], 20.0 * expected, epsilon = 1e-14);
    }

    #[test]
    fn test_differentiable_dot_product_attention_ties_and_mask() {
        // The two visible keys tie, so the backward body picks exactly one of them as its pivot, and the masked key
        // receives no gradient even though its score is the largest.
        let inputs = AttentionInputs {
            mask: Some(
                Array::from_elements(ArrayType::new_static(DataType::Boolean, [1, 1, 1, 3]), &[true, true, false])
                    .unwrap(),
            ),
            ..AttentionInputs::new(
                f32_array(&[1, 1, 1, 1], &[1.0]),
                f32_array(&[1, 3, 1, 1], &[0.0, 0.0, 100.0]),
                f32_array(&[1, 3, 1, 1], &[1.0, 0.0, 12.0]),
            )
        };
        let (output, gradients) = attention_sum_gradient(inputs);
        assert_eq!(output.to_f64s(), vec![0.5]);
        assert_eq!(gradients.query.to_f64s(), vec![0.0]);
        assert_eq!(gradients.key.to_f64s(), vec![0.25, -0.25, 0.0]);
        assert_eq!(gradients.value.to_f64s(), vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_differentiable_dot_product_attention_large_logits() {
        // Both `bf16` scores are `1000`, so the log-sum-exp residual is `1000 + log(2)`, which `bf16` would round to
        // `1000`. Probabilities that were reconstructed from such a rounded residual would double the gradient. The
        // residual keeps the `f32` data type of the softmax, and the backward body recomputes the normalization, so
        // the gradients are the correct quarters.
        let inputs = AttentionInputs::new(
            bf16_array(&[1, 1, 1, 1], &[1.0]),
            bf16_array(&[1, 2, 1, 1], &[1000.0, 1000.0]),
            bf16_array(&[1, 2, 1, 1], &[1.0, 0.0]),
        );
        let (output, gradients) = attention_sum_gradient(inputs);
        assert_eq!(output.to_f64s(), vec![0.5]);
        assert_eq!(gradients.query.to_f64s(), vec![0.0]);
        assert_eq!(gradients.key.to_f64s(), vec![0.25, -0.25]);
    }

    #[test]
    fn test_differentiable_dot_product_attention_low_precision() {
        // The first `f16` score is `65536`, which only the promoted score contraction represents, so the first
        // probability is one and the gradients are finite instead of NaN.
        let inputs = AttentionInputs::new(
            f16_array(&[1, 1, 1, 1], &[256.0]),
            f16_array(&[1, 2, 1, 1], &[256.0, 255.0]),
            f16_array(&[1, 2, 1, 1], &[1.0, 2.0]),
        );
        let (output, gradients) = attention_sum_gradient(inputs);
        assert_eq!(output, f16_array(&[], &[1.0]));
        assert_eq!(gradients.query, f16_array(&[1, 1, 1, 1], &[0.0]));
        assert_eq!(gradients.key, f16_array(&[1, 2, 1, 1], &[0.0, 0.0]));
        assert_eq!(gradients.value, f16_array(&[1, 2, 1, 1], &[1.0, 0.0]));
    }

    #[test]
    fn test_differentiable_dot_product_attention_rounded_weights() {
        // The probabilities of the two query rows differ in `f32` but round to the same `bf16` weights, with which
        // the forward body contracts the values. The two output rows are therefore equal for every value, so the
        // value gradient of their difference is an exact zero, which unrounded weights would not return.
        let function =
            differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_scale(1.0));
        let inputs = AttentionInputs::new(
            bf16_array(&[1, 2, 1, 1], &[1.0, 1.0078125]),
            bf16_array(&[1, 2, 1, 1], &[0.0, 0.25]),
            bf16_array(&[1, 2, 1, 1], &[1.0, 3.0]),
        );
        let (difference, gradients): (Array, AttentionInputs<Array>) = differentiate_at(inputs)
            .value_and_gradient(|inputs| {
                let output = function.call(inputs)?;
                let difference = output.slice_axis(1, 0, 1, 1)?.sub(&output.slice_axis(1, 1, 2, 1)?)?;
                difference.reduce(&[0, 1, 2, 3], ReductionKind::Sum)
            })
            .unwrap();
        assert_eq!(difference, bf16_array(&[], &[0.0]));
        assert_eq!(gradients.value, bf16_array(&[1, 2, 1, 1], &[0.0, 0.0]));
    }

    #[test]
    fn test_differentiable_dot_product_attention_sharding() {
        // Reverse mode accumulates each cotangent into the cotangent of its input, so every gradient has the sharding
        // of its input, however the query, key, value, bias, mask, and sequence lengths are placed. The contractions
        // of the backward definition place their results according to the inputs that they contract, and two of their
        // inputs can be sharded along a contracted dimension even though no forward contraction is (e.g., the output
        // cotangent and a value that is sharded along its head dimension), so the cases cover each placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let placed = |data_type: DataType, dimensions: &[usize], axis: Option<usize>| {
            let mut placements = vec![ShardingDimension::Replicated; dimensions.len()];
            if let Some(axis) = axis {
                placements[axis] = ShardingDimension::sharded(["x"]);
            }
            ArrayType::new_static(data_type, dimensions.to_vec())
                .with_sharding(Sharding::new(mesh.clone(), placements).unwrap())
                .unwrap()
        };
        let function = differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let gradient_types = |types: AttentionInputs<ArrayType>| {
            ArrayContext::trace(
                |inputs: AttentionInputs<_>| {
                    differentiate_at(inputs)
                        .gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
                        .map_err(ProgramError::from)
                },
                types,
            )
            .map(|(gradient_types, _)| gradient_types)
        };
        for (query_axis, key_axis, value_axis) in [
            (Some(0), None, None),
            (None, Some(0), None),
            (None, None, Some(0)),
            (Some(0), Some(0), Some(0)),
            (Some(1), None, None),
            (None, Some(1), None),
            (None, None, Some(1)),
            (Some(3), None, None),
            (None, Some(3), None),
            (None, None, Some(3)),
        ] {
            let types = AttentionInputs::new(
                placed(DataType::F32, &[2, 2, 1, 2], query_axis),
                placed(DataType::F32, &[2, 2, 1, 2], key_axis),
                placed(DataType::F32, &[2, 2, 1, 2], value_axis),
            );
            assert_eq!(gradient_types(types.clone()), Ok(types));
        }
        for (bias_axis, mask_axis, lengths_axis) in [
            (Some(0), None, None),
            (Some(2), None, None),
            (Some(3), None, None),
            (None, Some(0), None),
            (None, Some(2), None),
            (None, None, Some(0)),
        ] {
            let types = AttentionInputs {
                bias: Some(placed(DataType::F32, &[2, 1, 2, 2], bias_axis)),
                mask: Some(placed(DataType::Boolean, &[2, 1, 2, 2], mask_axis)),
                query_sequence_lengths: Some(placed(DataType::I32, &[2], lengths_axis)),
                ..AttentionInputs::new(
                    placed(DataType::F32, &[2, 2, 1, 2], None),
                    placed(DataType::F32, &[2, 2, 1, 2], None),
                    placed(DataType::F32, &[2, 2, 1, 2], None),
                )
            };
            let gradient_types = gradient_types(types.clone()).unwrap();
            assert_eq!(
                (gradient_types.query, gradient_types.key, gradient_types.value, gradient_types.bias),
                (types.query, types.key, types.value, types.bias),
            );
        }

        // The same holds for every placement of one mesh axis over the query, key, and value that the forward
        // definition accepts, including the ones whose backward contractions have no placement of their own (e.g., a
        // query that is sharded along its sequence and a value that is sharded along its own, which the weight
        // cotangent cannot both keep on one mesh axis).
        let forward = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let axes = [None, Some(0), Some(1), Some(2), Some(3)];
        let mut accepted = 0;
        for (query_axis, key_axis, value_axis) in axes
            .into_iter()
            .flat_map(|query| axes.into_iter().flat_map(move |key| axes.map(|value| (query, key, value))))
        {
            let types = AttentionInputs::new(
                placed(DataType::F32, &[2, 2, 2, 2], query_axis),
                placed(DataType::F32, &[2, 2, 2, 2], key_axis),
                placed(DataType::F32, &[2, 2, 2, 2], value_axis),
            );
            if ArrayContext::trace(|inputs: AttentionInputs<_>| forward.call(inputs), types.clone()).is_ok() {
                accepted += 1;
                assert_eq!(gradient_types(types.clone()), Ok(types), "{query_axis:?} {key_axis:?} {value_axis:?}");
            }
        }
        assert_eq!(accepted, 57);

        // Grouped query heads that a mesh axis shards return key and value cotangents that are summed over the groups
        // of every key/value head, which neither axis of that split can keep sharded.
        let wide_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let grouped = |dimensions: [usize; 4], head_axis: bool| {
            let mut placements = vec![ShardingDimension::Replicated; 4];
            if head_axis {
                placements[2] = ShardingDimension::sharded(["x"]);
            }
            ArrayType::new_static(DataType::F32, dimensions)
                .with_sharding(Sharding::new(wide_mesh.clone(), placements).unwrap())
                .unwrap()
        };
        let types = AttentionInputs::new(
            grouped([1, 1, 4, 1], true),
            grouped([1, 1, 2, 1], false),
            grouped([1, 1, 2, 1], false),
        );
        assert_eq!(gradient_types(types.clone()), Ok(types));

        // An input without any sharding that attends together with placed inputs has a gradient without one.
        let plain = ArrayType::new_static(DataType::F32, [2, 2, 1, 2]);
        let replicated = placed(DataType::F32, &[2, 2, 1, 2], None);
        for types in [
            AttentionInputs::new(plain.clone(), placed(DataType::F32, &[2, 2, 1, 2], Some(0)), replicated.clone()),
            AttentionInputs::new(placed(DataType::F32, &[2, 2, 1, 2], Some(0)), plain.clone(), replicated.clone()),
            AttentionInputs::new(placed(DataType::F32, &[2, 2, 1, 2], Some(1)), replicated.clone(), plain.clone()),
            AttentionInputs::new(plain.clone(), plain.clone(), placed(DataType::F32, &[2, 2, 1, 2], Some(3))),
            AttentionInputs {
                bias: Some(ArrayType::new_static(DataType::F32, [2, 2])),
                ..AttentionInputs::new(
                    placed(DataType::F32, &[2, 2, 1, 2], Some(0)),
                    replicated.clone(),
                    replicated.clone(),
                )
            },
            AttentionInputs {
                bias: Some(placed(DataType::F32, &[2, 1, 2, 2], Some(0))),
                ..AttentionInputs::new(plain.clone(), plain.clone(), plain.clone())
            },
        ] {
            assert_eq!(gradient_types(types.clone()), Ok(types));
        }

        // The placements only change where the elements live, so the gradients over placed inputs have the elements
        // of the gradients over inputs without a sharding.
        let elements =
            |start: f32, count: usize| (0..count).map(|index| start + 0.25 * index as f32).collect::<Vec<_>>();
        let gradients =
            |query_axis: Option<usize>, value_axis: Option<usize>, bias_axis: Option<usize>, plain: bool| {
                let array = |dimensions: &[usize], axis: Option<usize>, start: f32| {
                    let r#type = if plain {
                        ArrayType::new_static(DataType::F32, dimensions.to_vec())
                    } else {
                        placed(DataType::F32, dimensions, axis)
                    };
                    Array::from_elements(r#type, elements(start, dimensions.iter().product()).as_slice()).unwrap()
                };
                let inputs = AttentionInputs {
                    bias: Some(array(&[2, 1, 2, 2], bias_axis, -0.5)),
                    ..AttentionInputs::new(
                        array(&[2, 2, 1, 2], query_axis, -1.0),
                        array(&[2, 2, 1, 2], None, 0.5),
                        array(&[2, 2, 1, 2], value_axis, -2.0),
                    )
                };
                let gradients: AttentionInputs<Array> = differentiate_at(inputs)
                    .gradient(|inputs| {
                        let output = function.call(inputs)?;
                        output.mul(&output)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum)
                    })
                    .unwrap();
                [gradients.query, gradients.key, gradients.value, gradients.bias.unwrap()]
                    .map(|gradient| gradient.elements::<f32>().unwrap())
            };
        let expected = gradients(None, None, None, true);
        for (query_axis, value_axis, bias_axis) in [
            (None, None, None),
            (Some(1), None, None),
            (None, Some(3), None),
            (None, None, Some(0)),
            (Some(1), Some(1), None),
            (Some(3), Some(0), None),
        ] {
            assert_eq!(gradients(query_axis, value_axis, bias_axis, false), expected);
        }
    }

    #[test]
    fn test_differentiable_dot_product_attention_invisible_rows() {
        // The mask hides every key/value position from the second query row, whose output is the mean of the values
        // whatever its scores are. The backward definition zeroes the cotangents of the invisible scores, so the
        // query, key, and bias gradients agree with central finite differences of the forward definition. The `f64`
        // inputs also cover the logit of the invisible scores, which is finite in the `f32` softmax data type.
        let function = differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let mask = Array::from_elements(
            ArrayType::new_static(DataType::Boolean, [2, 3]),
            &[true, true, false, false, false, false],
        )
        .unwrap();
        check_gradient!(
            |input, mask| {
                // The input packs the `[1, 2, 1, 2]` query, the `[1, 3, 1, 2]` key, the `[1, 3, 1, 2]` value, and
                // the `[2, 3]` bias.
                let inputs = AttentionInputs {
                    bias: Some(input.slice_axis(0, 16, 22, 1)?.reshape([2, 3])?),
                    mask: Some(mask),
                    ..AttentionInputs::new(
                        input.slice_axis(0, 0, 4, 1)?.reshape([1, 2, 1, 2])?,
                        input.slice_axis(0, 4, 10, 1)?.reshape([1, 3, 1, 2])?,
                        input.slice_axis(0, 10, 16, 1)?.reshape([1, 3, 1, 2])?,
                    )
                };
                function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum)
            },
            at = Array::vector(vec![
                0.5f64, -1.0, 2.0, 0.25, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0, -1.0, 4.0, 0.1, -0.2, 0.3,
                0.4, 0.5, -0.6,
            ])
            .unwrap(),
            with = mask,
            step = 5e-3,
            tolerance = 5e-3,
        );
    }

    #[test]
    fn test_differentiable_dot_product_attention_wide_range() {
        // The backward body keeps `f64` inputs in `f64` outside the softmax. The query is beyond the `f32` range, so
        // narrowing it would make the exactly zero key gradient the NaN product of zero and infinity.
        let inputs = AttentionInputs::new(
            f64_array(&[1, 1, 1, 1], &[1.0e40]),
            f64_array(&[1, 1, 1, 1], &[0.0]),
            f64_array(&[1, 1, 1, 1], &[1.0]),
        );
        let (output, gradients) = attention_sum_gradient(inputs);
        assert_eq!(output, f64_array(&[], &[1.0]));
        assert_eq!(gradients.query, f64_array(&[1, 1, 1, 1], &[0.0]));
        assert_eq!(gradients.key, f64_array(&[1, 1, 1, 1], &[0.0]));
        assert_eq!(gradients.value, f64_array(&[1, 1, 1, 1], &[1.0]));

        // The weight cotangents of the two keys are the values, whose difference overflows `f32` although the logit
        // cotangents are representable, so the backward body centers halved weight cotangents.
        let inputs = AttentionInputs::new(
            f32_array(&[1, 1, 1, 1], &[1.0]),
            f32_array(&[1, 2, 1, 1], &[0.0, 0.0]),
            f32_array(&[1, 2, 1, 1], &[3.0e38, -3.0e38]),
        );
        let (output, gradients) = attention_sum_gradient(inputs);
        assert_eq!(output, f32_array(&[], &[0.0]));
        assert_eq!(gradients.query, f32_array(&[1, 1, 1, 1], &[0.0]));
        assert_eq!(gradients.key, f32_array(&[1, 2, 1, 1], &[1.5e38, -1.5e38]));
        assert_eq!(gradients.value, f32_array(&[1, 2, 1, 1], &[0.5, 0.5]));

        // The backward body multiplies by the score scale of the forward body rather than by its double, which a
        // scale in the upper half of the `f32` range would overflow into the NaN product of zero and infinity.
        let function =
            differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_scale(2.0e38));
        let inputs = AttentionInputs::new(
            f32_array(&[1, 1, 1, 1], &[0.0]),
            f32_array(&[1, 1, 1, 1], &[0.0]),
            f32_array(&[1, 1, 1, 1], &[1.0]),
        );
        let (output, gradients): (Array, AttentionInputs<Array>) = differentiate_at(inputs)
            .value_and_gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap();
        assert_eq!(output, f32_array(&[], &[1.0]));
        assert_eq!(gradients.query, f32_array(&[1, 1, 1, 1], &[0.0]));
        assert_eq!(gradients.key, f32_array(&[1, 1, 1, 1], &[0.0]));

        // The probability of the second key is the smallest positive `f32`. Its logit cotangent is that probability
        // times the difference of the weight cotangents, which is representable, while half of it is not, so the
        // factor of two returns with the probability instead of after the product.
        let function =
            differentiable_dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_scale(1.0e38));
        let inputs = AttentionInputs::new(
            f32_array(&[1, 1, 1, 1], &[1.0]),
            f32_array(&[1, 2, 1, 1], &[0.0, -103.0e-38]),
            f32_array(&[1, 2, 1, 1], &[1.0, 0.0]),
        );
        let (_, gradients): (Array, AttentionInputs<Array>) = differentiate_at(inputs)
            .value_and_gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap();
        let tail = f32::from_bits(1) * 1.0e38;
        assert_eq!(gradients.key, f32_array(&[1, 2, 1, 1], &[tail, -tail]));
    }

    #[test]
    fn test_differentiable_dot_product_attention_dynamic_shapes() {
        // Reverse mode stages the forward and backward definitions over dynamic batch and sequence extents, whose
        // causal masks, sequence length masks, pivot coordinates, and softmax broadcasts take the runtime shape of
        // the logits. The staged gradient then matches the eager gradient at every runtime shape.
        let batch = DimensionVariable::new("batch", DimensionBounds::positive(Some(4)).unwrap());
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(4)).unwrap());
        let r#type = ArrayIrType::Array(ArrayType::new(
            DataType::F32,
            Shape::new(vec![batch.clone().into(), length.into(), 1.into(), 2.into()]),
        ));
        let lengths_type = ArrayIrType::Array(ArrayType::new(DataType::I32, Shape::new(vec![batch.into()])));
        let configuration = AttentionConfiguration::new().with_causal(true);
        let function = differentiable_dot_product_attention::<ArrayIrContext>(configuration);
        let (_, program) = ArrayIrContext::trace(
            |(inputs, lengths): ((_, _, _), _)| {
                differentiate_at(inputs)
                    .with_captures(lengths)
                    .gradient(|(query, key, value), lengths| {
                        function
                            .call(AttentionInputs {
                                query_sequence_lengths: Some(lengths),
                                ..AttentionInputs::new(query, key, value)
                            })?
                            .reduce(&[0, 1, 2, 3], ReductionKind::Sum)
                    })
                    .map_err(ProgramError::from)
            },
            ((r#type.clone(), r#type.clone(), r#type), lengths_type),
        )
        .unwrap();
        let expected_function = differentiable_dot_product_attention::<ArrayContext>(configuration);
        let elements = |count: usize, offset: f32| {
            (0..count).map(|index| ((index * 5 + 2) % 7) as f32 * 0.25 - offset).collect::<Vec<_>>()
        };
        for (batch, length) in [(2, 3), (1, 2)] {
            let dimensions = [batch, length, 1, 2];
            let count = 2 * batch * length;
            let inputs = (
                f32_array(&dimensions, &elements(count, 0.5)),
                f32_array(&dimensions, &elements(count, 1.0)),
                f32_array(&dimensions, &elements(count, -1.0)),
            );
            let query_lengths = (0..batch).map(|item| (length - item) as i32).collect::<Vec<_>>();
            let lengths = Array::from_elements(ArrayType::new_static(DataType::I32, [batch]), &query_lengths).unwrap();
            let (query, key, value) = differentiate_at(inputs.clone())
                .with_captures(lengths.clone())
                .gradient(|(query, key, value), lengths| {
                    expected_function
                        .call(AttentionInputs {
                            query_sequence_lengths: Some(lengths),
                            ..AttentionInputs::new(query, key, value)
                        })?
                        .reduce(&[0, 1, 2, 3], ReductionKind::Sum)
                })
                .unwrap();
            let (query_input, key_input, value_input) = inputs;
            assert_eq!(
                program.interpret((
                    (
                        ArrayIrValue::Array(query_input),
                        ArrayIrValue::Array(key_input),
                        ArrayIrValue::Array(value_input),
                    ),
                    ArrayIrValue::Array(lengths),
                )),
                Ok((ArrayIrValue::Array(query), ArrayIrValue::Array(key), ArrayIrValue::Array(value))),
            );
        }

        // Homogeneous array programs stage the pullback of an explicit output cotangent over dynamic extents, which
        // needs no runtime extent because the backward definition takes its shapes from its inputs. A bias also runs
        // the reduction of its cotangent over the axes along which it was broadcast.
        let batch = DimensionVariable::new("batch", DimensionBounds::positive(Some(4)).unwrap());
        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(4)).unwrap());
        let input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![batch.clone().into(), length.clone().into(), 1.into(), 2.into()]),
        );
        let bias_type = ArrayType::new(DataType::F32, Shape::new(vec![1.into(), length.clone().into()]));
        let function = differentiable_dot_product_attention::<ArrayContext>(configuration);
        let (_, program) = ArrayContext::trace(
            |(inputs, cotangent): ((_, _, _, _), _)| {
                let (_, pullback) = differentiate_at(inputs).vjp(|(query, key, value, bias)| {
                    function.call(AttentionInputs { bias: Some(bias), ..AttentionInputs::new(query, key, value) })
                })?;
                let (query, key, value, bias) = pullback.apply(cotangent)?;
                Ok((cotangent_value(query), cotangent_value(key), cotangent_value(value), cotangent_value(bias)))
            },
            ((input_type.clone(), input_type.clone(), input_type.clone(), bias_type), input_type.clone()),
        )
        .unwrap();
        for (batch, length) in [(2, 3), (1, 2)] {
            let dimensions = [batch, length, 1, 2];
            let count = 2 * batch * length;
            let inputs = (
                f32_array(&dimensions, &elements(count, 0.5)),
                f32_array(&dimensions, &elements(count, 1.0)),
                f32_array(&dimensions, &elements(count, -1.0)),
                f32_array(&[1, length], &elements(length, 0.25)),
            );
            let cotangent = f32_array(&dimensions, &elements(count, 0.75));
            let (_, pullback) = differentiate_at(inputs.clone())
                .vjp(|(query, key, value, bias)| {
                    function.call(AttentionInputs { bias: Some(bias), ..AttentionInputs::new(query, key, value) })
                })
                .unwrap();
            let (query, key, value, bias) = pullback.apply(cotangent.clone()).unwrap();
            let expected =
                (cotangent_value(query), cotangent_value(key), cotangent_value(value), cotangent_value(bias));
            assert_eq!(program.interpret((inputs, cotangent)), Ok(expected));
        }

        // Grouped key/value heads sum the cotangents of the query heads that share them, by splitting the query head
        // axis next to the dynamic batch and sequence dimensions.
        let grouped_type = |heads: usize| {
            ArrayType::new(
                DataType::F32,
                Shape::new(vec![batch.clone().into(), length.clone().into(), heads.into(), 2.into()]),
            )
        };
        let (_, program) = ArrayContext::trace(
            |(inputs, cotangent): ((_, _, _), _)| {
                let (_, pullback) = differentiate_at(inputs)
                    .vjp(|(query, key, value)| function.call(AttentionInputs::new(query, key, value)))?;
                let (query, key, value) = pullback.apply(cotangent)?;
                Ok((cotangent_value(query), cotangent_value(key), cotangent_value(value)))
            },
            ((grouped_type(4), grouped_type(2), grouped_type(2)), grouped_type(4)),
        )
        .unwrap();
        for (batch, length) in [(2, 3), (1, 1)] {
            let inputs = (
                f32_array(&[batch, length, 4, 2], &elements(8 * batch * length, 0.5)),
                f32_array(&[batch, length, 2, 2], &elements(4 * batch * length, 1.0)),
                f32_array(&[batch, length, 2, 2], &elements(4 * batch * length, -1.0)),
            );
            let cotangent = f32_array(&[batch, length, 4, 2], &elements(8 * batch * length, 0.75));
            let (_, pullback) = differentiate_at(inputs.clone())
                .vjp(|(query, key, value)| function.call(AttentionInputs::new(query, key, value)))
                .unwrap();
            let (query, key, value) = pullback.apply(cotangent.clone()).unwrap();
            let expected = (cotangent_value(query), cotangent_value(key), cotangent_value(value));
            assert_eq!(program.interpret((inputs, cotangent)), Ok(expected));
        }

        // Integer inputs (e.g., sequence lengths) have structurally zero tangents, which homogeneous array programs
        // cannot materialize for a dynamic shape, so reverse mode with such inputs needs an array IR family.
        let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![batch.into()]));
        let result = ArrayContext::trace(
            |(inputs, lengths): ((_, _, _), _)| {
                let (output, _) =
                    differentiate_at(inputs).with_captures(lengths).vjp(|(query, key, value), lengths| {
                        function.call(AttentionInputs {
                            query_sequence_lengths: Some(lengths),
                            ..AttentionInputs::new(query, key, value)
                        })
                    })?;
                Ok(output)
            },
            ((input_type.clone(), input_type.clone(), input_type), lengths_type),
        );
        assert_eq!(
            result.map(|_| ()),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot materialize a zero of type zero[batch] because no source has exactly that type \
                          (source types: [i32[batch]])"
                    .to_string(),
            }),
        );
    }
}
