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
//!     of its unbatched call, and definitions with dropout and the fused implementation reject batching.
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

use crate::arrays::{ArrayMemberType, ArrayType, DataType, Dimension, Shape};
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
use crate::operations::dot::{Dot, DotDimensionNumbers};
use crate::operations::exponential::{Exp, Log};
use crate::operations::logical::And;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastLike, DynamicBroadcast};
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::random::{Dropout, RandomAlgorithm};
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::parameters::{Parameter, Parameterized, Placeholder};
use crate::programs::{MaybeZero, Operation, ProgramError, TypeError, Typed, Value};
use crate::tracing::{DomainTracer, DomainTracingContext, TracingContext};

/// Value capabilities that staging attention requires, collected so that the functions that stage it (e.g.,
/// [`dot_product_attention`]) state one bound. Those functions create the portable bodies and the batching rule of the
/// named definitions, so they need every capability that either of them uses, while each helper of the bodies bounds
/// only what it calls. The last three members are the extent capabilities with which the batching rule folds batched
/// calls (refer to the [module documentation](self#batching)). This bundle is implemented automatically for every
/// value that implements all of its members, and so it must never be implemented manually.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor and is passed to every member,
/// so that tracers of homogeneous array programs implement this bundle for [`ArrayType`] and array IR values implement
/// it for [`ArrayIrType`](crate::ArrayIrType).
pub trait AttentionOperations<T = <Self as Capability>::Universe>:
    Capability
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
    + DimensionSize<T, Extent: Clone + Mul>
    + DynamicReshape<<Self as DimensionSize<T>>::Extent, T>
    + DynamicBroadcast<<Self as DimensionSize<T>>::Extent, T>
{
}

impl<
    T,
    V: FillLike<T>
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
                    AttentionBatchOutput::Tensor(0),
                    AttentionBatchOutput::Tensor(1),
                    AttentionBatchOutput::Tensor(2),
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
    /// for this choice so that training uses its stabilized backward rule.
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
    /// Explicit score scale, or [`None`] to use `1 / sqrt(head_dimension)`.
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
    /// `[B, N, T, S]` by aligning its axes with the trailing score axes.
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
/// and the remaining cases are grouped-query attention. The attended output has the query type. Reduced and unreduced
/// inputs are unsupported, because softmax and the log-sum-exp residual preserve neither partial sums nor replicas
/// that are filled with zeros. Calls align the manual variation of their inputs before staging the call.
///
/// [`AttentionInputs`] defines the complete input boundary, consisting of the query, key, and value, followed by an
/// optional broadcastable additive bias, Boolean visibility mask, query and key/value sequence lengths of every batch
/// item, and dropout state. Dot products, scaling, bias addition, and masking use the query type promoted to at least
/// `f32`, and the masked logits are then converted to `f32` for the numerically stable softmax. Visibility is the
/// conjunction of the explicit mask, causal ordering, the inclusive `(left, right)` local window, and both sequence
/// length predicates. Invisible scores use a finite large negative value in the logits type, and query rows beyond
/// their lengths are exact zeros. Without a configured scale, attention uses `1 / sqrt(H)`. Every batch item uses
/// the same rectangular storage shape, and sequence lengths control visibility within that storage.
///
/// [`DotProductAttention::call_with_residual`] also returns the log-sum-exp residual (i.e., the natural
/// logarithm of the sum of the exponentiated logits), which has the `TN` or `BTN` form and the query data type. With
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
/// and calls with dropout and the fused implementation cannot be batched. The named definitions have no derivative
/// rules, so differentiating a call fails. Use [`differentiable_dot_product_attention`] for the differentiable path.
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

/// Borrowed canonical view of attention input types.
struct AttentionInputTypes<'o> {
    /// Query type.
    query: &'o ArrayType,

    /// Key type.
    key: &'o ArrayType,

    /// Value type.
    value: &'o ArrayType,

    /// Optional bias type.
    bias: Option<&'o ArrayType>,

    /// Optional arbitrary Boolean mask type.
    mask: Option<&'o ArrayType>,

    /// Optional query sequence lengths type.
    query_sequence_lengths: Option<&'o ArrayType>,

    /// Optional key/value sequence lengths type.
    key_value_sequence_lengths: Option<&'o ArrayType>,

    /// Optional dropout random state type.
    dropout_state: Option<&'o ArrayType>,

    /// Forward output type in a backward boundary.
    output: Option<&'o ArrayType>,

    /// Forward residual type in a backward boundary.
    residual: Option<&'o ArrayType>,

    /// Incoming output cotangent type in a backward boundary.
    output_cotangent: Option<&'o ArrayType>,
}

impl<'o> AttentionInputTypes<'o> {
    /// Parses one attention forward operation boundary into an [`AttentionInputTypes`] instance.
    fn forward(input_structure: AttentionInputs<Placeholder>, input_types: &'o [ArrayType]) -> Result<Self, TypeError> {
        let [query, key, value, optional @ ..] = input_types else {
            return Err(TypeError::invalid("attention requires query, key, and value inputs"));
        };
        let mut index = 0;
        let bias = input_structure.bias.is_some().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let mask = input_structure.mask.is_some().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let query_sequence_lengths = input_structure.query_sequence_lengths.is_some().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let key_value_sequence_lengths = input_structure.key_value_sequence_lengths.is_some().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let dropout_state = input_structure.dropout_state.is_some().then(|| &optional[index]);
        Ok(Self {
            query,
            key,
            value,
            bias,
            mask,
            query_sequence_lengths,
            key_value_sequence_lengths,
            dropout_state,
            output: None,
            residual: None,
            output_cotangent: None,
        })
    }

    /// Parses one attention backward operation boundary into an [`AttentionInputTypes`] instance.
    fn backward(
        input_structure: AttentionInputs<Placeholder>,
        input_types: &'o [ArrayType],
    ) -> Result<Self, TypeError> {
        let input_count = input_structure.parameter_count();
        let mut types = Self::forward(input_structure, &input_types[..input_count])?;
        types.output = Some(&input_types[input_count]);
        types.residual = Some(&input_types[input_count + 1]);
        types.output_cotangent = Some(&input_types[input_count + 2]);
        Ok(types)
    }
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
    /// Returns the score scale of `configuration` for these dimensions, which is its explicit scale or, without one,
    /// `1 / sqrt(head_dimension)`.
    #[inline]
    fn scale(&self, configuration: AttentionConfiguration) -> f64 {
        configuration.scale().unwrap_or(1.0 / (self.head_dimension as f64).sqrt())
    }
}

/// Named attention definitions of one operation family `(V, O)` and one [`AttentionConfiguration`] (i.e., the forward
/// definition without and with its residual, and the backward definition). Each one is a named composite that serves
/// every input structure, under the [`AttentionKey`] of that structure. The functions that stage attention own these
/// definitions, so their traced bodies and their rule specializations live exactly as long as those functions.
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
/// except that definitions with dropout and the fused implementation reject batching.
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

    /// Log-sum-exp residual whose logical rank follows the query rank.
    Residual,

    /// Bias cotangent restored to the logical bias shape of the input at this index.
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
/// and returns the types of its outputs. The first output is the attended output, which has the query type and
/// therefore propagates the query's metadata (e.g., its sharding), and it is followed by the log-sum-exp residual when
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

    let types = AttentionInputTypes::forward(input_structure, input_types)?;
    let dimensions = validate_attention_boundary(operation_name, configuration, &types, input_types)?;
    let mut output_types = vec![types.query.clone()];

    if return_residual {
        output_types.push(attention_residual_type(&dimensions, types.query)?);
    }

    Ok(output_types)
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

    let types = AttentionInputTypes::backward(input_structure, input_types)?;
    let dimensions = validate_attention_boundary(operation_name, configuration, &types, input_types)?;

    let matches_expected = |actual: &ArrayType, expected: &ArrayType| -> bool {
        actual.data_type() == expected.data_type() && actual.shape() == expected.shape()
    };

    if !matches_expected(types.output.unwrap(), types.query) {
        return Err(TypeError::invalid(format!(
            "`{}` output type {} does not match the expected forward output type {}",
            operation_name,
            types.output.unwrap(),
            types.query,
        )));
    }

    let expected_output_cotangent_type = types.query.cotangent()?;
    if !matches_expected(types.output_cotangent.unwrap(), &expected_output_cotangent_type) {
        return Err(TypeError::invalid(format!(
            "`{}` output cotangent type {} does not match the expected cotangent type {}",
            operation_name,
            types.output_cotangent.unwrap(),
            expected_output_cotangent_type,
        )));
    }

    let expected_residual_type = attention_residual_type(&dimensions, types.query)?;
    if !matches_expected(types.residual.unwrap(), &expected_residual_type) {
        return Err(TypeError::invalid(format!(
            "`{}` residual type {} does not match the expected residual type {}",
            operation_name,
            types.residual.unwrap(),
            expected_residual_type,
        )));
    }

    let mut output_types = vec![types.query.cotangent()?, types.key.cotangent()?, types.value.cotangent()?];
    if let Some(bias) = types.bias {
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
    types: &AttentionInputTypes<'_>,
    input_types: &[ArrayType],
) -> Result<AttentionDimensions, TypeError> {
    ArrayType::check_matching_manual_variation(operation_name, &input_types.iter().collect::<Vec<_>>())?;
    let dimensions =
        validate_attention_inputs(operation_name, types.query, types.key, types.value, types.bias, types.mask)?;

    // Each sequence lengths input holds one `i32` length per batch item, so it is a rank-1 array whose extent matches
    // the batch dimension that the query, key, and value share.
    for (descriptor, lengths_type) in [
        ("query sequence lengths", types.query_sequence_lengths),
        ("key/value sequence lengths", types.key_value_sequence_lengths),
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
    if let Some(state_type) = types.dropout_state {
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
        } else if !score_type.data_type().is_numeric() && !score_type.data_type().is_boolean() {
            return Err(TypeError::invalid(format!(
                "`{}` bias must have a numeric or Boolean data type but got `{}`",
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

/// Returns the `[batch, query_sequence, heads]` or `[query_sequence, heads]` type of the log-sum-exp residual, which
/// has the query data type and inherits the query's sharding with only the trailing head dimension removed.
fn attention_residual_type(dimensions: &AttentionDimensions, query_type: &ArrayType) -> Result<ArrayType, TypeError> {
    let dimensions = if query_type.rank() == 3 {
        vec![dimensions.query_sequence.clone(), Dimension::Static(dimensions.query_heads)]
    } else {
        vec![dimensions.batch.clone(), dimensions.query_sequence.clone(), Dimension::Static(dimensions.query_heads)]
    };

    let residual_type = ArrayType::new(query_type.data_type(), Shape::new(dimensions));
    let Some(query_sharding) = query_type.sharding() else {
        return Ok(residual_type);
    };

    query_sharding
        .with_dimensions(query_sharding.dimensions()[..query_type.rank() - 1].to_vec())
        .and_then(|sharding| residual_type.with_sharding(sharding))
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
        + ConvertElementType,
>(
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

    // Dot products, scaling, bias, and masking promote low precision inputs to `f32`, while `f64` remains `f64`.
    // The stable softmax itself always runs in `f32`, matching the public attention contract.
    let logits_type = if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 };

    // The scores contract the head dimension of the query and of the expanded key for every batch item and head,
    // which gives the `BNTS` score form.
    let scores = query.dot(&expanded_key, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]))?;
    let scores = if data_type == logits_type { scores } else { scores.convert_element_type(logits_type)? };
    let scale_value = scores.fill_like(dimensions.scale(configuration))?;
    let mut scores = scores.mul(&scale_value)?;

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
    let logits = if inputs.mask.is_none()
        && !configuration.causal()
        && configuration.local_window().is_none()
        && inputs.query_sequence_lengths.is_none()
        && inputs.key_value_sequence_lengths.is_none()
    {
        scores
    } else {
        // The row and column coordinates of the scores are integer constants that cannot carry tangents. They take the
        // runtime shape of the scores, and so do the broadcasts of the mask and of the sequence lengths below.
        let columns = scores.iota_like(3, DataType::I32)?;
        let mut visible = match &inputs.mask {
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
            visible = Some(match visible {
                None => causal,
                Some(visible) => visible.and(&causal)?,
            });
        }

        // The local window keeps the key/value positions `j` with `i - left <= j <= i + right`.
        if let Some((left, right)) = configuration.local_window() {
            let rows = scores.iota_like(2, DataType::I32)?;
            let left = rows.fill_like(left as f64)?;
            let right = rows.fill_like(right as f64)?;
            let lower = rows.sub(&left)?;
            let upper = rows.add(&right)?;
            let in_window = columns
                .compare(&lower, ComparisonDirection::GreaterThanOrEqual)?
                .and(&columns.compare(&upper, ComparisonDirection::LessThanOrEqual)?)?;
            visible = Some(match visible {
                None => in_window,
                Some(visible) => visible.and(&in_window)?,
            });
        }

        // Query rows at or beyond the query length of their batch item see no key/value position at all.
        if let Some(lengths) = &inputs.query_sequence_lengths {
            let rows = scores.iota_like(2, DataType::I32)?;
            let bounds = lengths.broadcast_like(&rows, &[0])?;
            let in_range = rows.compare(&bounds, ComparisonDirection::LessThan)?;
            visible = Some(match visible {
                None => in_range,
                Some(visible) => visible.and(&in_range)?,
            });
        }

        // Key/value columns at or beyond the key/value length of their batch item are invisible to every row.
        if let Some(lengths) = &inputs.key_value_sequence_lengths {
            let bounds = lengths.broadcast_like(&columns, &[0])?;
            let in_range = columns.compare(&bounds, ComparisonDirection::LessThan)?;
            visible = Some(match visible {
                None => in_range,
                Some(visible) => visible.and(&in_range)?,
            });
        }

        // Invisible scores use a finite large negative value rather than negative infinity, so that a row without
        // any visible position still has a finite maximum and a finite log-sum-exp residual. At least one of the
        // masks above is present on this path, so the visibility is known.
        let large_negative = if scores.r#type().as_array_type()?.data_type() == DataType::F64 {
            -0.7 * f64::MAX
        } else {
            -0.7 * f32::MAX as f64
        };

        let masked = scores.fill_like(large_negative)?;
        visible.unwrap().select(&scores, &masked)?
    };

    let logits =
        if logits_type == SOFTMAX_DATA_TYPE { logits } else { logits.convert_element_type(SOFTMAX_DATA_TYPE)? };

    Ok(PreparedAttention { query, expanded_key, expanded_value, logits })
}

/// Adds the implicit batch axis used to evaluate one unbatched `TNH` attention input (i.e., a query, key, value,
/// forward output, or output cotangent). Inserting a unit axis preserves the dynamic extents of the input.
fn normalize_attention_input<V: Value<Type: ArrayMemberType> + Reshape>(input: V) -> Result<V, ProgramError> {
    if input.r#type().as_array_type()?.rank() == 4 {
        return Ok(input);
    }
    input.expand_dimensions(0)
}

/// Expands grouped key/value heads to the query heads. The `BSKH` input is replicated along a new group axis, which
/// gives the `BSKGH` form, and its key/value head and group axes are then merged into the query head axis of the
/// `BSNH` form. Both steps keep the batch and sequence dimensions, so they support dynamic extents along them.
fn expand_key_value_heads<V: Value<Type: ArrayMemberType> + Reshape + Broadcast>(
    key_or_value: V,
    dimensions: &AttentionDimensions,
) -> Result<V, ProgramError> {
    if dimensions.key_value_heads == dimensions.query_heads {
        return Ok(key_or_value);
    }

    let group = dimensions.query_heads / dimensions.key_value_heads;
    let r#type = key_or_value.r#type();
    let r#type = r#type.as_array_type()?;
    let [batch, length, _, head_dimension] = r#type.shape().dimensions() else {
        return Err(TypeError::invalid(format!(
            "attention key or value type `{}` does not have the normalized rank 4",
            r#type,
        ))
        .into());
    };

    // The output keeps the memory space and the placement of the input along the mapped axes, and its group axis is
    // replicated, because a broadcast cannot change manual variation.
    let axes = [0, 1, 2, 4];
    let sharding = r#type
        .sharding()
        .map(|sharding| sharding.with_broadcasted_dimensions(5, &axes))
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
    key_or_value.broadcast(grouped_type, &axes)?.reshape(Shape::new(expanded_dimensions))
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
            let prepared = prepare_attention(&inputs, configuration, &dimensions)?;
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
            // pass rather than a differentiable output, so it stops gradients.
            let residual = if return_residual {
                let residual = maxima.add(&sums.log()?)?;
                let residual = if SOFTMAX_DATA_TYPE == dimensions.data_type {
                    residual
                } else {
                    residual.convert_element_type(dimensions.data_type)?
                };
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
        return match configuration.implementation() {
            AttentionImplementation::Fused => {
                // A batched call loses the identity of its definition (i.e., it lowers its structurally batched body),
                // so batching a fused call with dropout would silently replace the fused kernels and their dropout mask
                // with the portable body and its mask. That would both change the requested numerics and break the
                // agreement between the dropout masks of the forward and backward definitions when only one of them
                // is batched (e.g., when the pullback of an unbatched call is batched).
                AttentionFunction::Folding(Arc::new(function.with_batching(|_, _, _| {
                    Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "named composite `{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` with dropout cannot be batched \
                             with the fused implementation, because a batched call cannot use the fused kernels or \
                             reproduce their dropout mask; use the portable implementation or a dropout state",
                        ),
                    })
                })))
            }
            AttentionImplementation::Automatic | AttentionImplementation::Portable => {
                // Folding would change the shape of the dropout mask and therefore the mask of every batch item,
                // so these definitions keep the structural batching of their bodies.
                AttentionFunction::Structural(Arc::new(function))
            }
        };
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
                            unreachable!("definitions without dropout batch by folding")
                        }
                    },
                };
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
/// that rounding the output or the residual to a narrower query data type loses, and so the output and the residual
/// are validated inputs for backend implementations whose numerical values this body does not use. It computes:
///
/// ```text
/// dV = transpose(P) * dO
/// dW = dO * transpose(V)
/// C = dW - dW[pivot]
/// dS = P * (C - sum(P * C))
/// dQ = scale * dS * K
/// dK = scale * transpose(dS) * Q,
/// ```
///
/// where the pivot is the first maximum logit of each row, whose `dS` entry is replaced by the negative sum of all
/// other entries, which preserves the dominant derivative when its probability rounds to one. Grouped query heads are
/// reduced back to each key/value head, a broadcast bias cotangent is reduced to the shape of its bias, and padded
/// query rows are zeroed. With dropout, `P` is the dropped probabilities in `dV`, and the cotangent of the dropped
/// probabilities is mapped back through the same dropout mask, which the same random state and probability shape
/// reproduce.
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
    /// dimensions.
    fn sum_key_value_head_groups<V: Value<Type: ArrayMemberType> + Reduce + Reshape>(
        cotangent: V,
        dimensions: &AttentionDimensions,
    ) -> Result<V, ProgramError> {
        let group = dimensions.query_heads / dimensions.key_value_heads;
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
        let prepared = prepare_attention(inputs, configuration, &dimensions)?;
        let output_cotangent = normalize_attention_input(output_cotangent.clone())?;

        // The public residual and output have the query data type, which can lose the normalization correction in
        // narrow formats. Recompute probabilities in the softmax type instead of recovering them from those values.
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

        // The backward mathematics runs in the softmax data type, like the recomputed probabilities.
        let convert = |array: CustomRuleTracer<V, O>| -> Result<CustomRuleTracer<V, O>, ProgramError> {
            if dimensions.data_type == SOFTMAX_DATA_TYPE {
                Ok(array)
            } else {
                Ok(array.convert_element_type(SOFTMAX_DATA_TYPE)?)
            }
        };
        let softmax_query = convert(prepared.query)?;
        let softmax_key = convert(prepared.expanded_key)?;
        let softmax_value = convert(prepared.expanded_value)?;
        let softmax_output_cotangent = convert(output_cotangent)?;

        // With dropout, the output uses the dropped probabilities `D = dropout(P)`, so the value cotangent uses `D`
        // and the cotangent of `P` is the cotangent of `D` through the same mask.
        let dropped_weights = apply_dropout(weights.clone(), inputs, configuration)?;
        let weight_cotangents = apply_dropout(
            softmax_output_cotangent
                .dot(&softmax_value, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]))?,
            inputs,
            configuration,
        )?;
        let zero = prepared.logits.fill_like(0.0)?;
        let pivot_cotangent = is_pivot.select(&weight_cotangents, &zero)?.reduce(&[3], ReductionKind::Sum)?;
        let pivot_cotangent = pivot_cotangent.broadcast_like(&prepared.logits, &[0, 1, 2])?;

        // Center the incoming weight cotangents before their weighted sum. This preserves a tiny dominant derivative
        // and gives exact zero when every weight cotangent is equal, even if the rounded probabilities do not sum to
        // one.
        let centered = weight_cotangents.sub(&pivot_cotangent)?;
        let delta = weights.mul(&centered)?.reduce(&[3], ReductionKind::Sum)?;
        let delta = delta.broadcast_like(&prepared.logits, &[0, 1, 2])?;
        let logit_cotangents = weights.mul(&centered.sub(&delta)?)?;

        // The logit cotangents of every row sum to zero, so the pivot's entry is replaced by the negative sum of the
        // other entries. That keeps the dominant derivative when the probability of the pivot rounds to one.
        let other_cotangents = is_pivot.select(&zero, &logit_cotangents)?;
        let pivot_cotangent = other_cotangents.reduce(&[3], ReductionKind::Sum)?;
        let pivot_cotangent = pivot_cotangent.broadcast_like(&prepared.logits, &[0, 1, 2])?;
        let logit_cotangents = is_pivot.select(&zero.sub(&pivot_cotangent)?, &other_cotangents)?;

        // The scores are `scale * Q * transpose(K)`, so the query and key cotangents contract the scaled logit
        // cotangents with the keys and the queries, and the value cotangent contracts the dropped probabilities with
        // the output cotangent. Every result is transposed from `BNTH` or `BNSH` to the form of its input.
        let scale_value = logit_cotangents.fill_like(dimensions.scale(configuration))?;
        let scaled_logit_cotangents = logit_cotangents.mul(&scale_value)?;
        let mut query_cotangent = scaled_logit_cotangents
            .dot(&softmax_key, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]))?
            .transpose([0, 2, 1, 3])?;
        if let Some(query_lengths) = &inputs.query_sequence_lengths {
            query_cotangent = zero_query_rows(query_cotangent, query_lengths, 1)?;
        }
        let key_cotangent = scaled_logit_cotangents
            .dot(&softmax_query, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))?
            .transpose([0, 2, 1, 3])?;
        let value_cotangent = dropped_weights
            .dot(&softmax_output_cotangent, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))?
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

        // Every cotangent returns to the cotangent data type and to the rank of its input.
        let mut cotangents = [query_cotangent, key_cotangent, value_cotangent]
            .into_iter()
            .zip(input_types[..3].iter().zip(input_ranks))
            .map(|(cotangent, (input_type, rank))| {
                let cotangent_data_type = input_type.cotangent()?.data_type();
                let cotangent = if cotangent.r#type().as_array_type()?.data_type() == cotangent_data_type {
                    cotangent
                } else {
                    cotangent.convert_element_type(cotangent_data_type)?
                };
                if rank == 4 { Ok(cotangent) } else { cotangent.squeeze(0) }
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
            let bias_cotangent = if bias_cotangent_type.data_type() == SOFTMAX_DATA_TYPE {
                bias_cotangent
            } else {
                bias_cotangent.convert_element_type(bias_cotangent_type.data_type())?
            };
            cotangents.push(bias_cotangent);
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
        return match configuration.implementation() {
            AttentionImplementation::Fused => {
                // Refer to the forward definition for why a batched fused call with dropout is rejected.
                AttentionFunction::Folding(Arc::new(function.with_batching(|_, _, _| {
                    Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "named composite `{DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME}` with dropout cannot be \
                             batched with the fused implementation, because a batched call cannot use the fused \
                             kernels or reproduce their dropout mask; use the portable implementation or a dropout \
                             state",
                        ),
                    })
                })))
            }
            AttentionImplementation::Automatic | AttentionImplementation::Portable => {
                // Folding would change the shape of the dropout mask and therefore the mask of every batch item,
                // so these definitions keep the structural batching of their bodies.
                AttentionFunction::Structural(Arc::new(function))
            }
        };
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
                            unreachable!("definitions without dropout batch by folding")
                        }
                    },
                };
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

    // Fold the mapped axis into attention's batch axis and stage the definition at the folded boundary.
    let folded = aligned
        .iter()
        .zip(input_roles)
        .map(|(value, role)| {
            let rank = rank(value)?;
            match role {
                AttentionBatchInput::Tensor if rank == 5 => {
                    let batch_extent = mapped_extent.mul(&value.dimension_size(1)?)?;
                    let folded_extents = std::iter::once(batch_extent).chain(extents(value, 2)?).collect::<Vec<_>>();
                    value.dynamic_reshape(folded_extents.as_slice())
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
                        Some(folded_extents) => normalized.dynamic_reshape(folded_extents.as_slice()),
                        None => Ok(normalized),
                    }
                }
                AttentionBatchInput::Length => value.dynamic_reshape(std::slice::from_ref(&folded_batch_extent)),
                AttentionBatchInput::Residual if rank == 4 => {
                    let folded_extents =
                        std::iter::once(folded_batch_extent.clone()).chain(extents(value, 2)?).collect::<Vec<_>>();
                    value.dynamic_reshape(folded_extents.as_slice())
                }
                AttentionBatchInput::Residual => Ok(value.clone()),
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;

    let outputs = call_fn(folded)?;
    let output_roles = output_roles_fn(outputs.len());
    check_count!("output", output_roles, outputs.len(), ProgramError);

    // Restore the mapped axis of every output and reduce bias cotangents over the axes that their biases broadcast.
    let outputs = outputs
        .into_iter()
        .zip(output_roles)
        .map(|(output, role)| match role {
            AttentionBatchOutput::Tensor(input_index) if rank(&aligned[input_index])? == 5 => {
                let unfolded_extents = [mapped_extent.clone(), aligned[input_index].dimension_size(1)?]
                    .into_iter()
                    .chain(extents(&output, 1)?)
                    .collect::<Vec<_>>();
                output.dynamic_reshape(unfolded_extents.as_slice())
            }
            AttentionBatchOutput::Tensor(_) => Ok(output),
            AttentionBatchOutput::Residual => match &batch_extent {
                Some(batch_extent) => {
                    let unfolded_extents = [mapped_extent.clone(), batch_extent.clone()]
                        .into_iter()
                        .chain(extents(&output, 1)?)
                        .collect::<Vec<_>>();
                    output.dynamic_reshape(unfolded_extents.as_slice())
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
                    Some(_) => output.dynamic_reshape(normalized_extents.as_slice())?,
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
                reduced.dynamic_reshape(extents(bias, 0)?.as_slice())
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let output_axes = vec![BatchAxis::from_position(0); outputs.len()];
    Ok((outputs, output_axes))
}

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use approx::assert_abs_diff_eq;
    use half::bf16;
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionVariable,
        LogicalMesh, MeshAxis, MeshAxisType, Sharding,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchingError, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationError, differentiate_at};
    use crate::macros::check_gradient;
    use crate::operations::manipulation::slicing::Slice;
    use crate::parameters::{ParameterError, Placeholder};
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

    #[test]
    fn test_attention_key() {
        let configuration = AttentionConfiguration::new().with_causal(true).with_dropout(0.25);
        let input_structure = AttentionInputs {
            bias: Some(Placeholder),
            query_sequence_lengths: Some(Placeholder),
            dropout_state: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        let key = AttentionKey::Forward { configuration, input_structure, return_residual: true };
        assert_eq!(key.configuration(), configuration);
        assert_eq!(key.input_structure(), input_structure);

        // Forward keys with different residual requests identify different definitions.
        assert_ne!(key, AttentionKey::Forward { configuration, input_structure, return_residual: false });

        // Backward definitions have no residual request.
        let key = AttentionKey::Backward { configuration, input_structure };
        assert_eq!(key.configuration(), configuration);
        assert_eq!(key.input_structure(), input_structure);
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
    fn test_attention_configuration_equality_and_hashing() {
        // Equal configurations are equal and hash identically, so they work as map keys.
        let configuration = AttentionConfiguration::new().with_scale(0.25).with_causal(true).with_dropout(0.1);
        let same_configuration = AttentionConfiguration::new().with_scale(0.25).with_causal(true).with_dropout(0.1);
        assert_eq!(configuration, same_configuration);
        assert_eq!(hash_of(&configuration), hash_of(&same_configuration));

        // The right local-window radius has no effect with causal masking, so it does not distinguish configurations
        // or their keys there, while it does without causal masking.
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
        let configurations = HashMap::from([(configuration, "configuration")]);
        assert_eq!(configurations.get(&same_configuration), Some(&"configuration"));
        assert_eq!(configurations.get(&configuration.with_dropout(0.2)), None);
        assert_ne!(configuration, configuration.with_causal(false));
        assert_ne!(configuration, configuration.with_dropout(0.2));
        assert_ne!(configuration, configuration.with_dropout(None));

        // The scale and the dropout rate compare bitwise, so signed zeros are distinct even though `-0.0 == 0.0`.
        assert_ne!(AttentionConfiguration::new().with_scale(-0.0), AttentionConfiguration::new().with_scale(0.0));
        assert_ne!(AttentionConfiguration::new().with_scale(0.0), AttentionConfiguration::new());
        assert_ne!(AttentionConfiguration::new().with_dropout(-0.0), AttentionConfiguration::new().with_dropout(0.0));
    }

    #[test]
    fn test_attention_inputs() {
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
        // The parameters of the inputs are their values in canonical order, in which the present optional inputs
        // follow the query, key, and value, and their structure records which optional inputs are present.
        let values = (0..8).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
        let inputs = AttentionInputs::new(values[0].clone(), values[1].clone(), values[2].clone());
        assert_eq!(inputs.parameter_structure(), AttentionInputs::new(Placeholder, Placeholder, Placeholder));
        assert_eq!(inputs.parameter_count(), 3);
        assert_eq!(inputs.clone().into_parameters().collect::<Vec<_>>(), values[..3].to_vec());
        assert_eq!(
            AttentionInputs::from_parameters(
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                values[..3].iter().cloned()
            ),
            Ok(inputs.clone()),
        );

        let inputs = AttentionInputs {
            mask: Some(values[3].clone()),
            key_value_sequence_lengths: Some(values[4].clone()),
            ..inputs
        };
        let structure = AttentionInputs {
            mask: Some(Placeholder),
            key_value_sequence_lengths: Some(Placeholder),
            ..AttentionInputs::new(Placeholder, Placeholder, Placeholder)
        };
        assert_eq!(inputs.parameter_structure(), structure);
        assert_eq!(inputs.parameter_count(), 5);
        assert_eq!(inputs.clone().into_parameters().collect::<Vec<_>>(), values[..5].to_vec());
        assert_eq!(AttentionInputs::from_parameters(structure, values[..5].iter().cloned()), Ok(inputs));

        let inputs = AttentionInputs {
            query: values[0].clone(),
            key: values[1].clone(),
            value: values[2].clone(),
            bias: Some(values[3].clone()),
            mask: Some(values[4].clone()),
            query_sequence_lengths: Some(values[5].clone()),
            key_value_sequence_lengths: Some(values[6].clone()),
            dropout_state: Some(values[7].clone()),
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
        assert_eq!(inputs.clone().into_parameters().collect::<Vec<_>>(), values);
        assert_eq!(AttentionInputs::from_parameters(structure, values.iter().cloned()), Ok(inputs));

        // A structure only accepts as many values as it has parameters.
        assert_eq!(
            AttentionInputs::from_parameters(structure, values[..7].iter().cloned()),
            Err(ParameterError::MissingParameters { expected_count: 8, paths: None }),
        );
        assert_eq!(
            AttentionInputs::from_parameters(
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                values[..4].iter().cloned()
            ),
            Err(ParameterError::UnusedParameters { paths: None }),
        );
    }

    #[test]
    fn test_attention_residuals_parameters() {
        // The inputs of the backward definition pair the residuals with the output cotangent, which appends the
        // forward output, its residual, and the output cotangent to the forward inputs.
        let values = (0..7).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
        let inputs = (
            AttentionResiduals {
                inputs: AttentionInputs {
                    mask: Some(values[3].clone()),
                    ..AttentionInputs::new(values[0].clone(), values[1].clone(), values[2].clone())
                },
                output: values[4].clone(),
                residual: values[5].clone(),
            },
            values[6].clone(),
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
        assert_eq!(inputs.clone().into_parameters().collect::<Vec<_>>(), values);
        assert_eq!(<(AttentionResiduals<_>, _)>::from_parameters(structure, values.iter().cloned()), Ok(inputs));
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
                            %10:f32[] = constant [value=-2.3819763e38]
                            %11:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[]] %10
                            %12:f32[1, 1, 2, 2] = select %9 %6 %11
                            %13:f32[1, 1, 2] = reduce [kind=max, axes=[3]] %12
                            %14:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[0, 1, 2]] %13
                            %15:f32[1, 1, 2, 2] = sub %12 %14
                            %16:f32[1, 1, 2, 2] = exp %15
                            %17:f32[1, 1, 2] = reduce [kind=sum, axes=[3]] %16
                            %18:f32[1, 1, 2, 2] = broadcast [output_type=f32[1, 1, 2, 2], output_axes=[0, 1, 2]] %17
                            %19:f32[1, 1, 2, 2] = div %16 %18
                            %20:f32[1, 1, 2, 1] = dot [
                                dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                    rhs_batching=[0, 2]),
                            ] %19 %2
                            %21:f32[1, 2, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %20
                        in (%21)
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
        let r#type = ArrayType::new_static(DataType::F32, [1, 2, 1, 1]);
        let (_, program) = ArrayContext::trace(
            |(inputs, lengths): (AttentionInputs<_>, _)| {
                Ok((
                    function.call(inputs.clone())?,
                    function
                        .call(AttentionInputs { query_sequence_lengths: Some(lengths.clone()), ..inputs.clone() })?,
                    function.call(AttentionInputs { key_value_sequence_lengths: Some(lengths), ..inputs })?,
                ))
            },
            (AttentionInputs::new(r#type.clone(), r#type.clone(), r#type), ArrayType::new_static(DataType::I32, [1])),
        )
        .unwrap();
        let keys = program
            .entry_region_ref()
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                ArrayOperation::Composite(operation) => operation.key().cloned(),
                _ => None,
            })
            .collect::<Vec<_>>();
        let input_structure = AttentionInputs::new(Placeholder, Placeholder, Placeholder);
        assert_eq!(
            keys,
            [
                input_structure,
                AttentionInputs { query_sequence_lengths: Some(Placeholder), ..input_structure },
                AttentionInputs { key_value_sequence_lengths: Some(Placeholder), ..input_structure },
            ]
            .map(|input_structure| {
                AttentionKey::Forward { configuration, input_structure, return_residual: false }.to_composite_key()
            }),
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
                                %20:f32[]{varying_rank_0} = constant [value=-2.3819763e38]
                                %21:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[],
                                ] %20
                                %22:f32[1, 1, 1, 1]{varying_rank_4} = select %19 %11 %21
                                %23:f32[1, 1, 1]{varying_rank_3} = reduce [kind=max, axes=[3]] %22
                                %24:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0, 1, 2],
                                ] %23
                                %25:f32[1, 1, 1, 1]{varying_rank_4} = sub %22 %24
                                %26:f32[1, 1, 1, 1]{varying_rank_4} = exp %25
                                %27:f32[1, 1, 1]{varying_rank_3} = reduce [kind=sum, axes=[3]] %26
                                %28:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0, 1, 2],
                                ] %27
                                %29:f32[1, 1, 1, 1]{varying_rank_4} = div %26 %28
                                %30:f32[1, 1, 1, 1]{varying_rank_4} = dot [
                                    dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                        rhs_batching=[0, 2]),
                                ] %29 %2
                                %31:f32[1, 1, 1, 1]{varying_rank_4} = transpose [permutation=[0, 2, 1, 3]] %30
                                %32:i32[1, 1, 1, 1]{varying_rank_4} = iota [
                                    type=i32[1, 1, 1, 1]{varying_rank_4},
                                    dimension=1,
                                ]
                                %33:i32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=i32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0],
                                ] %5
                                %34:bool[1, 1, 1, 1]{varying_rank_4} = compare [direction=LessThan] %32 %33
                                %35:f32[]{varying_rank_0} = constant [value=0.0]
                                %36:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[],
                                ] %35
                                %37:f32[1, 1, 1, 1]{varying_rank_4} = select %34 %31 %36
                                %38:f32[1, 1, 1]{varying_rank_3} = log %27
                                %39:f32[1, 1, 1]{varying_rank_3} = add %23 %38
                                %40:f32[1, 1, 1]{varying_rank_3} = transpose [permutation=[0, 2, 1]] %39
                                %41:f32[1, 1, 1]{varying_rank_3} = stop_gradient %40
                            in (%37, %41)
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
            ArrayType::new_static(DataType::F64, [3]),
            ArrayType::new_static(DataType::Boolean, [2, 3]),
            lengths.clone(),
            lengths,
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

        // Inputs inside manual regions must have matching manual variation, and reduced inputs are rejected.
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
            .with_sharding(Sharding::replicated(mesh, 4).with_reduced_axes(["x"]).unwrap())
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
                     `parallel_vary` on the inputs that lack an axis, as `align_manual_variation` does",
                ),
        ));
        assert!(matches!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new(),
                AttentionInputs::new(Placeholder, Placeholder, Placeholder),
                false,
                &[reduced, invariant.clone(), invariant],
            ),
            Err(TypeError::Invalid { message, .. })
                if message == format!("`{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` does not support reduced inputs"),
        ));
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
        let operand_type = |length: &DimensionVariable| {
            let dimensions = vec![batch.clone().into(), length.clone().into(), 1.into(), 2.into()];
            ArrayType::new(DataType::F32, Shape::new(dimensions))
        };
        let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![batch.clone().into()]));
        let elements = |count: usize, offset: f32| {
            (0..count).map(|index| ((index * 7 + 3) % 11) as f32 * 0.25 - offset).collect::<Vec<_>>()
        };
        let operands = |batch: usize, query_length: usize, key_length: usize| {
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
                ..AttentionInputs::new(
                    operand_type(&query_length),
                    operand_type(&key_length),
                    operand_type(&key_length),
                )
            },
        )
        .unwrap();
        for inputs in [
            AttentionInputs {
                query_sequence_lengths: Some(lengths(&[2, 1])),
                key_value_sequence_lengths: Some(lengths(&[3, 2])),
                ..operands(2, 2, 3)
            },
            AttentionInputs {
                query_sequence_lengths: Some(lengths(&[2])),
                key_value_sequence_lengths: Some(lengths(&[1])),
                ..operands(1, 3, 2)
            },
        ] {
            assert_eq!(program.interpret(inputs.clone()), Ok(function.call_with_residual(inputs).unwrap()));
        }

        // Attention without masks broadcasts only the softmax statistics and the scale.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let (_, program) = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs::new(operand_type(&query_length), operand_type(&key_length), operand_type(&key_length)),
        )
        .unwrap();
        for inputs in [operands(2, 2, 3), operands(1, 3, 2)] {
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
                ..AttentionInputs::new(
                    operand_type(&query_length),
                    operand_type(&key_length),
                    operand_type(&key_length),
                )
            },
        )
        .unwrap();
        for (items, rows, columns) in [(2, 2, 3), (1, 3, 2)] {
            let visible = (0..items * rows * columns).map(|index| index % 3 != 1).collect::<Vec<_>>();
            let mask_type = ArrayType::new_static(DataType::Boolean, [items, 1, rows, columns]);
            let inputs = AttentionInputs {
                bias: Some(f32_array(&[rows, columns], &elements(rows * columns, 0.75))),
                mask: Some(Array::from_elements(mask_type, &visible).unwrap()),
                ..operands(items, rows, columns)
            };
            assert_eq!(program.interpret(inputs.clone()), Ok(function.call(inputs).unwrap()));
        }

        // Dropout draws the random bits of its mask for a static shape, so definitions with dropout still require
        // static shapes.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_dropout(0.5));
        let result = ArrayContext::trace(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs::new(operand_type(&query_length), operand_type(&key_length), operand_type(&key_length)),
        );
        assert!(matches!(
            result,
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "`dropout` requires static extents along the axes that are not broadcast axes, but axis \
                               0 of `f32[batch, 1, query_length, key_length]` has extent `batch`",
        ));
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());

        // Grouped key/value heads are replicated along a new group axis and merged into the query heads, which keeps
        // the dynamic batch and sequence dimensions.
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

        // Rank-4 attention folds its mapped axis into its batch axis, whose extents and trailing extents must be
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
                if message == "`dimension_size` cannot return the dynamic extent `length` of axis 2 of \
                               `f32[2, 1, length, 1, 1]` as a host extent; use array IR values for dynamic extents",
        ));
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
    fn test_dot_product_attention_batching_dynamic_extent() {
        // Array IR families fold dynamic mapped extents with first-class dimension values, which the rule reads from
        // the mapped axis of a mapped input.
        let function = dot_product_attention::<ArrayIrContext>(AttentionConfiguration::new());
        let trace = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
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
        let query = trace.input(r#type.clone().into());
        let key = trace.input(r#type.clone().into());
        let value = trace.input(r#type.clone().into());
        let output = batch(
            |inputs: AttentionInputs<_>| function.call(inputs),
            AttentionInputs::new(query, key, value),
            AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let program = trace
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
                                %9:dimension<1> = constant [value=1]
                                %10:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %1 %4 %7 %8 %9
                                %11:dimension<1> = constant [value=1]
                                %12:dimension<1> = constant [value=1]
                                %13:dimension<1> = constant [value=1]
                                %14:dimension<1> = constant [value=1]
                                %15:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %2 %4 %12 %13 %14
                                %16:dimension<1> = constant [value=1]
                                %17:dimension<1> = constant [value=1]
                                %18:dimension<1> = constant [value=1]
                                %19:dimension<1> = constant [value=1]
                                %20:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %3 %4 %17 %18 %19
                                %21:f32[batch, 1, 1, 1] = composite [
                                    key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, \
                                        local_window = none, implementation = 0, dropout_rate = none, \
                                        residual = false, bias = false, mask = false, query_sequence_lengths = false, \
                                        key_value_sequence_lengths = false, dropout_state = false}\",
                                ] %10 %15 %20 [
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
                                %22:dimension<1> = constant [value=1]
                                %23:dimension<1> = constant [value=1]
                                %24:dimension<1> = constant [value=1]
                                %25:dimension<1> = constant [value=1]
                                %26:f32[batch, 1, 1, 1, 1] = \
                                    reshape [requires_runtime_assertion=false] %21 %4 %22 %23 %24 %25
                            in (%26)
                        },
                    ]
                in (%8)
            "}
            .trim_end(),
        );
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
        // their dropout mask with the portable body, so it is rejected.
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
                    "cannot differentiate a `composite` call of `{DOT_PRODUCT_ATTENTION_COMPOSITE_NAME}` that has no \
                     derivative rule",
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
    fn test_differentiable_dot_product_attention_narrow_residual() {
        // The `bf16` log-sum-exp residual rounds `1000 + log(2)` to `1000`. Reconstructing the probabilities from
        // that residual would double the gradient, while recomputing the normalization retains the correct quarter
        // gradients.
        let bf16_array = |dimensions: &[usize], elements: &[f32]| {
            let elements = elements.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
            Array::from_elements(ArrayType::new_static(DataType::BF16, dimensions.to_vec()), elements.as_slice())
                .unwrap()
        };
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
            |(operands, lengths): ((_, _, _), _)| {
                differentiate_at(operands)
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
            let operands = (
                f32_array(&dimensions, &elements(count, 0.5)),
                f32_array(&dimensions, &elements(count, 1.0)),
                f32_array(&dimensions, &elements(count, -1.0)),
            );
            let query_lengths = (0..batch).map(|item| (length - item) as i32).collect::<Vec<_>>();
            let lengths = Array::from_elements(ArrayType::new_static(DataType::I32, [batch]), &query_lengths).unwrap();
            let (query, key, value) = differentiate_at(operands.clone())
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
            let (query_input, key_input, value_input) = operands;
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
        let operand_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![batch.clone().into(), length.clone().into(), 1.into(), 2.into()]),
        );
        let bias_type = ArrayType::new(DataType::F32, Shape::new(vec![1.into(), length.clone().into()]));
        let function = differentiable_dot_product_attention::<ArrayContext>(configuration);
        let (_, program) = ArrayContext::trace(
            |(operands, cotangent): ((_, _, _, _), _)| {
                let (_, pullback) = differentiate_at(operands).vjp(|(query, key, value, bias)| {
                    function.call(AttentionInputs { bias: Some(bias), ..AttentionInputs::new(query, key, value) })
                })?;
                let (query, key, value, bias) = pullback.apply(cotangent)?;
                let cotangents = [query, key, value, bias].map(|cotangent| match cotangent {
                    MaybeZero::Value(cotangent) => cotangent,
                    MaybeZero::Zero(_) => unreachable!("every floating-point input has a cotangent"),
                });
                let [query, key, value, bias] = cotangents;
                Ok((query, key, value, bias))
            },
            ((operand_type.clone(), operand_type.clone(), operand_type.clone(), bias_type), operand_type.clone()),
        )
        .unwrap();
        for (batch, length) in [(2, 3), (1, 2)] {
            let dimensions = [batch, length, 1, 2];
            let count = 2 * batch * length;
            let operands = (
                f32_array(&dimensions, &elements(count, 0.5)),
                f32_array(&dimensions, &elements(count, 1.0)),
                f32_array(&dimensions, &elements(count, -1.0)),
                f32_array(&[1, length], &elements(length, 0.25)),
            );
            let cotangent = f32_array(&dimensions, &elements(count, 0.75));
            let (_, pullback) = differentiate_at(operands.clone())
                .vjp(|(query, key, value, bias)| {
                    function.call(AttentionInputs { bias: Some(bias), ..AttentionInputs::new(query, key, value) })
                })
                .unwrap();
            let (query, key, value, bias) = pullback.apply(cotangent.clone()).unwrap();
            let expected = [query, key, value, bias].map(|cotangent| match cotangent {
                MaybeZero::Value(cotangent) => cotangent,
                MaybeZero::Zero(_) => unreachable!("every floating-point input has a cotangent"),
            });
            let [query, key, value, bias] = expected;
            assert_eq!(program.interpret((operands, cotangent)), Ok((query, key, value, bias)));
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
            |(operands, cotangent): ((_, _, _), _)| {
                let (_, pullback) = differentiate_at(operands)
                    .vjp(|(query, key, value)| function.call(AttentionInputs::new(query, key, value)))?;
                let (query, key, value) = pullback.apply(cotangent)?;
                let cotangents = [query, key, value].map(|cotangent| match cotangent {
                    MaybeZero::Value(cotangent) => cotangent,
                    MaybeZero::Zero(_) => unreachable!("every floating-point input has a cotangent"),
                });
                let [query, key, value] = cotangents;
                Ok((query, key, value))
            },
            ((grouped_type(4), grouped_type(2), grouped_type(2)), grouped_type(4)),
        )
        .unwrap();
        for (batch, length) in [(2, 3), (1, 1)] {
            let operands = (
                f32_array(&[batch, length, 4, 2], &elements(8 * batch * length, 0.5)),
                f32_array(&[batch, length, 2, 2], &elements(4 * batch * length, 1.0)),
                f32_array(&[batch, length, 2, 2], &elements(4 * batch * length, -1.0)),
            );
            let cotangent = f32_array(&[batch, length, 4, 2], &elements(8 * batch * length, 0.75));
            let (_, pullback) = differentiate_at(operands.clone())
                .vjp(|(query, key, value)| function.call(AttentionInputs::new(query, key, value)))
                .unwrap();
            let (query, key, value) = pullback.apply(cotangent.clone()).unwrap();
            let expected = [query, key, value].map(|cotangent| match cotangent {
                MaybeZero::Value(cotangent) => cotangent,
                MaybeZero::Zero(_) => unreachable!("every floating-point input has a cotangent"),
            });
            let [query, key, value] = expected;
            assert_eq!(program.interpret((operands, cotangent)), Ok((query, key, value)));
        }

        // Integer inputs (e.g., sequence lengths) have structurally zero tangents, which homogeneous array programs
        // cannot materialize for a dynamic shape, so reverse mode with such inputs needs an array IR family.
        let lengths_type = ArrayType::new(DataType::I32, Shape::new(vec![batch.into()]));
        let result = ArrayContext::trace(
            |(operands, lengths): ((_, _, _), _)| {
                let (output, _) =
                    differentiate_at(operands).with_captures(lengths).vjp(|(query, key, value), lengths| {
                        function.call(AttentionInputs {
                            query_sequence_lengths: Some(lengths),
                            ..AttentionInputs::new(query, key, value)
                        })
                    })?;
                Ok(output)
            },
            ((operand_type.clone(), operand_type.clone(), operand_type), lengths_type),
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
