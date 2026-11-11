//! Scaled dot-product attention, staged as calls of named composites rather than as a primitive operation. Attention
//! has a portable body built from ordinary array operations, a stabilized analytical derivative, and optional backend
//! implementations (e.g., cuDNN's fused attention kernels). Staging it through named composites (refer to
//! [`Composite::with_named_composite`]) keeps one body that every program transform already supports, while backends
//! recognize the calls by their keys and may replace them with their own implementations.
//!
//! # Definitions
//!
//! A named attention definition is a named composite whose [`CompositeKey`] encodes an [`AttentionKey`], which
//! consists of an [`AttentionDefinitionKind`], the value-independent [`AttentionConfiguration`], and the
//! [`AttentionInputSignature`] of the optional inputs that are present. There are two kinds of definitions:
//!
//!   - **Forward:** Named [`DOT_PRODUCT_ATTENTION_COMPOSITE_NAME`]. It maps the [`AttentionInputs`] to the attended
//!     output, followed by the log-sum-exp residual of its softmax when the configuration requests it.
//!   - **Backward:** Named [`DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME`]. It maps the [`AttentionBackwardInputs`]
//!     (i.e., the forward inputs, the forward output, its residual, and the output cotangent) to the cotangents of the
//!     query, key, and value, followed by that of a bias whose cotangent space is not zero-dimensional.
//!
//! The keys identify the computation for backends, kernel persistence, and shared lowering. A backend decodes them
//! with [`AttentionKey::from_composite_key`] and then either lowers the portable body of the definition or replaces
//! the call with its own implementation, as the [`AttentionImplementation`] of the configuration requests.
//!
//! The portable bodies are written once over an [`AttentionUniverse`], which abstracts how they obtain and construct
//! the extents of the arrays that they shape. Values typed by [`ArrayType`] use static host extents, while array IR
//! values (i.e., values typed by [`ArrayIrType`]) use first-class dimension values and therefore also support dynamic
//! batch and sequence extents.
//!
//! # Ownership and Lifecycle
//!
//! Named composites are owned by their callers, so attention is staged through functions that are created once and
//! called many times, which are the [`DotProductAttention`] that [`dot_product_attention`] creates for inference and
//! the [`DifferentiableDotProductAttention`] that [`differentiable_dot_product_attention`] creates for training. Each
//! function owns one cache of definitions with one named composite per definition kind, residual request, and
//! optional input signature, each created on first use. Its calls therefore share the traced bodies, their lowering,
//! and their rule specializations for as long as the function lives. Programs that contain staged calls remain valid
//! after their function is dropped, and batching such a program stages an equivalent definition.
//!
//! # Transforms
//!
//!   - **Manual Variation:** Both functions align the manual variation of their inputs before the composite boundary
//!     (refer to [`ManualVariationAlignment`]), so that the transposes of those alignments accumulate the cotangent
//!     contributions of invariant inputs exactly once.
//!   - **Batching:** A batched call folds its mapped axis into attention's batch axis and calls the same definition at
//!     the folded boundary, which keeps backend implementations for batched calls. Definitions with dropout are
//!     batched structurally instead, so that every batch item keeps the dropout mask of its unbatched call, and
//!     definitions with dropout and the fused implementation reject batching.
//!   - **Differentiation:** The forward and backward definitions have no derivative rules, so differentiating a call
//!     of the inference function fails. The training function stages a third named composite, named
//!     [`DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME`], whose reverse mode rule calls the forward definition with
//!     its residual and then the backward definition. Higher-order derivatives differentiate the backward body.
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
//! // Two query positions attend over two key/value positions with one head and a head dimension of one.
//! let r#type = ArrayType::new_static(DataType::F32, [2, 1, 1]);
//! let query = Array::from_elements(r#type.clone(), &[1.0f32, 1.0])?;
//! let key = Array::from_elements(r#type.clone(), &[1.0f32, 1.0])?;
//! let value = Array::from_elements(r#type, &[3.0f32, 5.0])?;
//! let (output, residual) = attention.call(AttentionInputs::new(query, key, value))?;
//! // The first position sees only the first value, and the second position weighs both values equally.
//! assert_eq!(output.to_f64s(), vec![3.0, 4.0]);
//! assert!(residual.is_none());
//! # Ok(())
//! # }
//! ```

// TODO(eaplatanios): Review this module.

use std::collections::HashMap;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};

use ryft_macros::Parameterized;

use crate::arrays::{ArrayIrType, ArrayType, DataType, Dimension, DimensionType, Shape};
use crate::batching::{BatchAxis, BatchableType, BatchingLevelExtent};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::DifferentiableType;
use crate::macros::check_count;
use crate::operations::arithmetic::{Add, Div, Mul, Sub};
use crate::operations::collectives::parallel_vary::ManualVariationAlignment;
use crate::operations::comparisons::{Compare, ComparisonDirection};
use crate::operations::composites::builders::{BatchedComposite, Composite, CompositeCallContext, composite};
use crate::operations::composites::configurations::{
    BroadcastInvariantBatching, CompositeBatching, CompositeVjp, DefaultJvp, SymbolicZeroVjp,
};
use crate::operations::composites::rules::CustomRuleTracer;
use crate::operations::composites::rules::definitions::{CompositeAttribute, CompositeKey, CustomRuleDefinition};
use crate::operations::constants::constant::DimensionConstant;
use crate::operations::constants::fill::Fill;
use crate::operations::constants::iota::{DynamicIota, Iota};
use crate::operations::control_flow::select::Select;
use crate::operations::differentiation::stop_gradient::StopGradient;
use crate::operations::dimensions::dimension_mul::DimensionMulOperation;
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::dot::{Dot, DotDimensionNumbers};
use crate::operations::exponential::{Exp, Log};
use crate::operations::logical::And;
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast};
use crate::operations::manipulation::conversions::ConvertElementType;
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::random::{Dropout, RandomAlgorithm};
use crate::operations::reductions::{Reduce, ReductionKind};
use crate::parameters::{Parameter, Parameterized};
use crate::programs::{MaybeZero, Operation, ProgramError, Type, TypeError, Typed, Value, ValueProjection};
use crate::tracing::{DomainTracer, DomainTracingContext, TracingContext};

/// Type universe in which the portable attention bodies run, which determines how they obtain and construct the extents
/// of the arrays that they shape (i.e., for broadcasts, reshapes, iotas, and fills). The bodies are written once over a
/// value `V` whose [`Type`] implements this trait, and they compute on its array view [`Self::ArrayValue`]. The
/// following universes implement it:
///
///   - [`ArrayType`] implements it for homogeneous array values, whose extents are static host numbers. The array
///     view is the value itself, and every extent must be static, so attention over dynamically shaped arrays in a
///     homogeneous array family fails with an exact error. Such programs stage attention in an array IR family
///     instead, which is how backends support bounded dynamic shapes (e.g., XLA tracers).
///   - [`ArrayIrType`] implements it for array IR values, whose extents are first-class dimension values. Every
///     constructor, broadcast, and reshape that depends on a shape takes its extents from ordinary dimension values,
///     so the same body supports static and dynamic batch and sequence extents.
///
/// The implementations are on the type universes rather than on the values, which keeps them coherent for every value
/// type. Both are blanket implementations over every value of their universe whose capabilities satisfy their bounds,
/// so a value of these universes (e.g., a backend tracer) participates by providing those capabilities rather than by
/// implementing this trait, which coherence permits only for values that the blanket implementations do not cover. A
/// new type universe participates by implementing this trait for its own type. The trait is public because it bounds
/// [`dot_product_attention`] and [`differentiable_dot_product_attention`].
pub trait AttentionUniverse<V>: Type {
    /// Array view of `V` on which the attention mathematics computes.
    type ArrayValue: Value<Type = ArrayType>
        + Add
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Div
        + Dot
        + Dropout
        + Exp
        + Log
        + Mul
        + Reduce
        + Reshape
        + Select
        + StopGradient
        + Sub
        + Transpose;

    /// Extent of one axis (e.g., `usize` for static extents or a dimension value for first-class extents).
    type Extent: Clone;

    /// Returns the array view of `value`.
    fn into_array(value: V) -> Result<Self::ArrayValue, ProgramError>;

    /// Returns the value whose array view is `array`.
    fn from_array(array: Self::ArrayValue) -> V;

    /// Returns the array type of `value`.
    fn array_type(value: &V) -> Result<ArrayType, ProgramError>;

    /// Returns the array type that `r#type` describes.
    fn to_array_type(r#type: &Self) -> Result<ArrayType, TypeError>;

    /// Returns the type of this universe that describes the array type `r#type`.
    fn from_array_type(r#type: ArrayType) -> Self;

    /// Returns the extent of the batching level whose extent is `extent`, constructed next to `value` (i.e., in its
    /// context).
    fn batch_extent(value: &V, extent: &BatchingLevelExtent<V>) -> Result<Self::Extent, ProgramError>;

    /// Returns the extent of `value` along `axis`.
    fn extent(value: &V, axis: usize) -> Result<Self::Extent, ProgramError>;

    /// Returns the constant extent `extent`, constructed next to `value` (i.e., in its context).
    fn constant_extent(value: &V, extent: usize) -> Result<Self::Extent, ProgramError>;

    /// Returns the product of two extents, constructed next to `value` (i.e., in its context).
    fn multiply_extents(value: &V, left: &Self::Extent, right: &Self::Extent) -> Result<Self::Extent, ProgramError>;

    /// Broadcasts `value` to the provided output extents, where `axes` names the output axis of every input axis.
    fn broadcast(value: &V, extents: &[Self::Extent], axes: &[usize]) -> Result<V, ProgramError>;

    /// Reshapes `value` to the provided output extents.
    fn reshape(value: &V, extents: &[Self::Extent]) -> Result<V, ProgramError>;

    /// Returns an `i32` iota with the shape of `target` that counts along `axis`.
    fn iota(target: &V, axis: usize) -> Result<V, ProgramError>;

    /// Returns `value` at the data type of `target`, broadcast to the shape of `target`.
    fn fill(target: &V, value: f64) -> Result<V, ProgramError>;
}

impl<A> AttentionUniverse<A> for ArrayType
where
    A: Value<Type = ArrayType>
        + Add
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Div
        + Dot
        + Dropout
        + Exp
        + Log
        + Mul
        + Reduce
        + Reshape
        + Select
        + StopGradient
        + Sub
        + Transpose,
    A::Domain: Fill<f64, A> + Iota<A>,
{
    type ArrayValue = A;
    type Extent = usize;

    #[inline]
    fn into_array(value: A) -> Result<A, ProgramError> {
        Ok(value)
    }

    #[inline]
    fn from_array(array: A) -> A {
        array
    }

    #[inline]
    fn array_type(value: &A) -> Result<ArrayType, ProgramError> {
        Ok(value.r#type().into_owned())
    }

    #[inline]
    fn to_array_type(r#type: &ArrayType) -> Result<ArrayType, TypeError> {
        Ok(r#type.clone())
    }

    #[inline]
    fn from_array_type(r#type: ArrayType) -> ArrayType {
        r#type
    }

    fn batch_extent(_value: &A, extent: &BatchingLevelExtent<A>) -> Result<usize, ProgramError> {
        match extent {
            BatchingLevelExtent::Static(extent) => Ok(*extent),
            BatchingLevelExtent::Dynamic(_) => Err(TypeError::invalid(
                "attention over homogeneous arrays cannot be batched with a dynamic extent; use array IR values for \
                 dynamic extents",
            )
            .into()),
        }
    }

    fn extent(value: &A, axis: usize) -> Result<usize, ProgramError> {
        let r#type = value.r#type();
        let dimension = r#type.dimension(axis);
        dimension.value().ok_or_else(|| {
            TypeError::invalid(format!(
                "attention over homogeneous arrays requires static extents, but axis {axis} of `{}` has extent \
                 `{dimension}`; use array IR values for dynamic extents",
                r#type.as_ref(),
            ))
            .into()
        })
    }

    #[inline]
    fn constant_extent(_value: &A, extent: usize) -> Result<usize, ProgramError> {
        Ok(extent)
    }

    fn multiply_extents(_value: &A, left: &usize, right: &usize) -> Result<usize, ProgramError> {
        left.checked_mul(*right).ok_or_else(|| {
            TypeError::invalid(format!("attention extent product {left} × {right} overflows `usize`")).into()
        })
    }

    fn broadcast(value: &A, extents: &[usize], axes: &[usize]) -> Result<A, ProgramError> {
        // The output keeps the memory space and the placement of the input along the mapped axes, and its new axes are
        // replicated, because a broadcast cannot change manual variation.
        let r#type = value.r#type();
        let sharding = r#type
            .sharding()
            .map(|sharding| sharding.with_broadcasted_dimensions(extents.len(), axes))
            .transpose()
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        let output_type = ArrayType::new(r#type.data_type(), static_shape(extents))
            .with_memory(r#type.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()))?;
        Broadcast::broadcast(value, output_type, axes)
    }

    #[inline]
    fn reshape(value: &A, extents: &[usize]) -> Result<A, ProgramError> {
        value.reshape(static_shape(extents))
    }

    fn iota(target: &A, axis: usize) -> Result<A, ProgramError> {
        let output_type = target.r#type().into_owned().with_data_type(DataType::I32).with_layout(None);
        target.domain().iota(&output_type, axis)
    }

    fn fill(target: &A, value: f64) -> Result<A, ProgramError> {
        let target_type = target.r#type();
        let scalar = target.domain().fill(&ArrayType::scalar(target_type.data_type()), value)?;
        let extents = (0..target_type.rank())
            .map(|axis| <Self as AttentionUniverse<A>>::extent(target, axis))
            .collect::<Result<Vec<_>, _>>()?;
        <Self as AttentionUniverse<A>>::broadcast(&scalar, extents.as_slice(), &[])
    }
}

impl<V> AttentionUniverse<V> for ArrayIrType
where
    V: Value<Type = ArrayIrType>
        + DimensionSize
        + DynamicBroadcast
        + DynamicReshape
        + ValueProjection<ArrayType>
        + ValueProjection<DimensionType>,
    <V as ValueProjection<ArrayType>>::Projected: Value<Type = ArrayType>
        + Add
        + And
        + Broadcast
        + Compare
        + ConvertElementType
        + Div
        + Dot
        + Dropout
        + Exp
        + Log
        + Mul
        + Reduce
        + Reshape
        + Select
        + StopGradient
        + Sub
        + Transpose,
    <<V as ValueProjection<ArrayType>>::Projected as Value>::Domain:
        Fill<f64, <V as ValueProjection<ArrayType>>::Projected>,
    V::Domain: Context<Type = ArrayIrType, Operation: From<DimensionMulOperation>> + DimensionConstant + DynamicIota<V>,
{
    type ArrayValue = <V as ValueProjection<ArrayType>>::Projected;
    type Extent = V;

    #[inline]
    fn into_array(value: V) -> Result<Self::ArrayValue, ProgramError> {
        ValueProjection::<ArrayType>::into_projected(value).map_err(Into::into)
    }

    #[inline]
    fn from_array(array: Self::ArrayValue) -> V {
        <V as ValueProjection<ArrayType>>::from_projected(array)
    }

    #[inline]
    fn array_type(value: &V) -> Result<ArrayType, ProgramError> {
        Ok(<&ArrayType>::try_from(value.r#type().as_ref())?.clone())
    }

    #[inline]
    fn to_array_type(r#type: &ArrayIrType) -> Result<ArrayType, TypeError> {
        Ok(<&ArrayType>::try_from(r#type)?.clone())
    }

    #[inline]
    fn from_array_type(r#type: ArrayType) -> ArrayIrType {
        r#type.into()
    }

    fn batch_extent(value: &V, extent: &BatchingLevelExtent<V>) -> Result<V, ProgramError> {
        match extent {
            BatchingLevelExtent::Static(extent) => value.domain().dimension_constant(*extent),
            BatchingLevelExtent::Dynamic(extent) => Ok(extent.clone()),
        }
    }

    #[inline]
    fn extent(value: &V, axis: usize) -> Result<V, ProgramError> {
        value.dimension_size(axis)
    }

    #[inline]
    fn constant_extent(value: &V, extent: usize) -> Result<V, ProgramError> {
        value.domain().dimension_constant(extent)
    }

    fn multiply_extents(value: &V, left: &V, right: &V) -> Result<V, ProgramError> {
        let left_type = left.r#type();
        let left_type = <&DimensionType>::try_from(left_type.as_ref())?;
        let right_type = right.r#type();
        let right_type = <&DimensionType>::try_from(right_type.as_ref())?;
        let mut outputs = value.domain().bind(
            DimensionMulOperation::new(left_type, right_type)?,
            Vec::new(),
            &[left.clone(), right.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }

    #[inline]
    fn broadcast(value: &V, extents: &[V], axes: &[usize]) -> Result<V, ProgramError> {
        value.dynamic_broadcast(extents, axes)
    }

    #[inline]
    fn reshape(value: &V, extents: &[V]) -> Result<V, ProgramError> {
        value.clone().dynamic_reshape(extents)
    }

    fn iota(target: &V, axis: usize) -> Result<V, ProgramError> {
        let target_type = target.r#type();
        let target_type = <&ArrayType>::try_from(target_type.as_ref())?;
        let output_type = target_type.clone().with_data_type(DataType::I32);
        // Only the dynamic axes of the output type take dimension inputs, in axis order.
        let dynamic_extents = output_type
            .shape()
            .dimensions()
            .iter()
            .enumerate()
            .filter(|(_, dimension)| matches!(dimension, Dimension::Dynamic(_)))
            .map(|(axis, _)| target.dimension_size(axis))
            .collect::<Result<Vec<_>, _>>()?;
        target.domain().dynamic_iota(&output_type, axis, dynamic_extents.as_slice())
    }

    fn fill(target: &V, value: f64) -> Result<V, ProgramError> {
        let target_type = target.r#type();
        let target_type = <&ArrayType>::try_from(target_type.as_ref())?;
        let scalar = <Self as AttentionUniverse<V>>::into_array(target.clone())?
            .domain()
            .fill(&ArrayType::scalar(target_type.data_type()), value)?;
        let extents = (0..target_type.rank()).map(|axis| target.dimension_size(axis)).collect::<Result<Vec<_>, _>>()?;
        <Self as AttentionUniverse<V>>::from_array(scalar).dynamic_broadcast(extents.as_slice(), &[])
    }
}

/// Rule configurations that the named attention definitions of the operation family `(V, O)` require, which collects
/// the bounds of their batching rules so that generic callers state one bound. It is implemented for every
/// [`CustomRuleDefinition`] that supports them, which holds for every operation family whose operations support
/// partial evaluation and differentiation (e.g., the [`ArrayOperation`](crate::arrays::ArrayOperation) family and
/// backend families). That blanket implementation covers every type that satisfies the collected bounds, so this trait
/// is never implemented directly.
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

/// Name of the forward named attention definition (refer to [`AttentionKey`]).
pub const DOT_PRODUCT_ATTENTION_COMPOSITE_NAME: &str = "ryft.dot_product_attention";

/// Name of the backward named attention definition (refer to [`AttentionKey`]).
pub const DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME: &str = "ryft.dot_product_attention_backward";

/// Name of the named composite that stages differentiable attention (refer to
/// [`differentiable_dot_product_attention`]). Its key has the attributes of the forward definition's key except for
/// `residual` (refer to [`AttentionKey`]).
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
///   - `local_window`: Optional inclusive `[left, right]` local window radii.
///   - `implementation`: Requested [`AttentionImplementation`], which is `0` for automatic, `1` for portable, and `2`
///     for fused.
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
}

/// Backend implementation requested for attention.
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
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Automatic => formatter.write_str("automatic"),
            Self::Portable => formatter.write_str("portable"),
            Self::Fused => formatter.write_str("fused"),
        }
    }
}

/// Value-independent semantics of scaled dot-product attention.
///
/// Equality and hashing compare the floating-point scale and dropout rate bitwise, so that configurations are faithful
/// keys of the attributes that backends receive (e.g., `-0.0` and `+0.0` scales are distinct).
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

    /// Whether the forward operation also returns its log-sum-exp residual.
    return_residual: bool,

    /// Optional probability with which dropout drops each attention probability.
    dropout: Option<f64>,
}

impl AttentionConfiguration {
    /// Creates the default configuration, which has an inferred scale, non-causal global attention, automatic
    /// implementation selection, no residual output, and no dropout.
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

    /// Sets the inclusive `(left, right)` local window radii. The local window is independent of causal masking.
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

    /// Requests or suppresses the log-sum-exp residual output.
    #[inline]
    pub fn with_residual(mut self, return_residual: bool) -> Self {
        self.return_residual = return_residual;
        self
    }

    /// Sets the probability with which dropout drops each attention probability, which must lie in the open interval
    /// `(0, 1)`. Passing [`None`] disables dropout. Dropout applies a [`DropoutOperation`](crate::DropoutOperation) to
    /// the normalized attention probabilities, whose random state is the optional
    /// [`AttentionInputs::dropout_state`] input. Refer to that input for how the state selects the dropout mask.
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

    /// Returns the optional inclusive `(left, right)` local window radii.
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

    /// Returns the optional probability with which dropout drops each attention probability.
    #[inline]
    pub fn dropout(&self) -> Option<f64> {
        self.dropout
    }
}

impl PartialEq for AttentionConfiguration {
    fn eq(&self, other: &Self) -> bool {
        self.scale.map(f64::to_bits) == other.scale.map(f64::to_bits)
            && self.causal == other.causal
            && self.local_window == other.local_window
            && self.implementation == other.implementation
            && self.return_residual == other.return_residual
            && self.dropout.map(f64::to_bits) == other.dropout.map(f64::to_bits)
    }
}

impl Eq for AttentionConfiguration {}

impl Hash for AttentionConfiguration {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.scale.map(f64::to_bits).hash(state);
        self.causal.hash(state);
        self.local_window.hash(state);
        self.implementation.hash(state);
        self.return_residual.hash(state);
        self.dropout.map(f64::to_bits).hash(state);
    }
}

/// Presence metadata for the optional attention inputs (i.e., the bias, mask, sequence lengths, and dropout state) in
/// their canonical order.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AttentionInputSignature {
    /// Whether an additive bias follows query, key, and value.
    bias: bool,

    /// Whether an arbitrary Boolean visibility mask follows the bias.
    mask: bool,

    /// Whether the query sequence lengths of every batch item follow the mask.
    query_sequence_lengths: bool,

    /// Whether the key/value sequence lengths of every batch item follow the query sequence lengths.
    key_value_sequence_lengths: bool,

    /// Whether a dropout random state follows key/value sequence lengths.
    dropout_state: bool,
}

impl AttentionInputSignature {
    /// Creates an optional input signature without a dropout random state.
    #[inline]
    pub fn new(bias: bool, mask: bool, query_sequence_lengths: bool, key_value_sequence_lengths: bool) -> Self {
        Self { bias, mask, query_sequence_lengths, key_value_sequence_lengths, dropout_state: false }
    }

    /// Returns this signature with the provided presence of a dropout random state.
    #[inline]
    pub fn with_dropout_state(mut self, dropout_state: bool) -> Self {
        self.dropout_state = dropout_state;
        self
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

    /// Returns whether the query sequence lengths of every batch item are present.
    #[inline]
    pub fn has_query_sequence_lengths(&self) -> bool {
        self.query_sequence_lengths
    }

    /// Returns whether the key/value sequence lengths of every batch item are present.
    #[inline]
    pub fn has_key_value_sequence_lengths(&self) -> bool {
        self.key_value_sequence_lengths
    }

    /// Returns whether a dropout random state is present.
    #[inline]
    pub fn has_dropout_state(&self) -> bool {
        self.dropout_state
    }

    /// Returns the number of optional inputs described by this signature.
    #[inline]
    pub fn count(&self) -> usize {
        usize::from(self.bias)
            + usize::from(self.mask)
            + usize::from(self.query_sequence_lengths)
            + usize::from(self.key_value_sequence_lengths)
            + usize::from(self.dropout_state)
    }
}

/// Query, key, value, and optional inputs supplied to scaled dot-product attention.
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

    /// Optional query sequence length of every batch item.
    pub query_sequence_lengths: Option<P>,

    /// Optional key/value sequence length of every batch item.
    pub key_value_sequence_lengths: Option<P>,

    /// Optional random generator state of the dropout mask (refer to [`AttentionConfiguration::with_dropout`]), with
    /// the [`RandomAlgorithm::state_type`](crate::RandomAlgorithm::state_type) of its algorithm. With a state, the
    /// mask is a deterministic function of that state and backends never replace the dropout with a fused kernel that
    /// draws its own mask. Without a state, backends may fuse the dropout (e.g., into the cuDNN attention kernel),
    /// while Ryft's own implementations draw the mask from a fixed state (refer to
    /// [`DropoutOperation`](crate::DropoutOperation)). It is only accepted together with a dropout rate.
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

    /// Creates the [`AttentionInputs`] whose values in canonical attention input order are `values`.
    ///
    /// # Parameters
    ///
    ///   - `signature`: Presence metadata for the optional inputs.
    ///   - `values`: Query, key, value, and the present optional inputs in canonical order.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] when the number of values does not match `signature`.
    pub fn from_values(signature: AttentionInputSignature, values: &[P]) -> Result<Self, TypeError>
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
            dropout_state: signature.has_dropout_state().then(|| values.next().unwrap()),
        })
    }

    /// Returns the optional input signature of these inputs.
    #[inline]
    pub fn signature(&self) -> AttentionInputSignature {
        AttentionInputSignature::new(
            self.bias.is_some(),
            self.mask.is_some(),
            self.query_sequence_lengths.is_some(),
            self.key_value_sequence_lengths.is_some(),
        )
        .with_dropout_state(self.dropout_state.is_some())
    }

    /// Returns the values in canonical attention input order.
    pub fn into_values(self) -> Vec<P> {
        vec![self.query, self.key, self.value]
            .into_iter()
            .chain(self.bias)
            .chain(self.mask)
            .chain(self.query_sequence_lengths)
            .chain(self.key_value_sequence_lengths)
            .chain(self.dropout_state)
            .collect()
    }
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
    /// Returns the [`AttentionBackwardInputs`] whose canonical values are `values`, for forward inputs with the
    /// provided optional input signature.
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

    /// Returns the values of these inputs in their canonical order.
    pub fn into_values(self) -> Vec<P> {
        let mut values = self.inputs.into_values();
        values.extend([self.output, self.residual, self.output_cotangent]);
        values
    }
}

/// Scaled dot-product attention inference function, built by [`dot_product_attention`]. Each call stages one call of
/// the named forward attention definition (refer to [`AttentionKey`]) for the optional input signature of its inputs,
/// which backends may replace with their own implementation of the definition (e.g., a fused kernel). The function
/// owns its definitions, so it should be created once and called many times, which traces each definition's body once
/// per input types and shares it, together with its lowering, across those calls.
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
/// inputs are unsupported, because softmax and the log-sum-exp residual preserve neither partial sums nor replicas
/// that are filled with zeros. Calls align the manual variation of their inputs before staging the call.
///
/// [`AttentionInputs`] defines the complete input boundary, consisting of the query, key, and value, followed by an
/// optional broadcastable additive bias, Boolean visibility mask, query and key/value sequence lengths of every batch
/// item, and dropout state. Dot products, scaling, bias addition, and masking use the query type promoted to at least
/// `f32`, and the masked logits are then converted to `f32` for the numerically stable softmax. Visibility is the
/// conjunction of the explicit mask, causal ordering, the inclusive `(left, right)` local window, and both sequence
/// length predicates. Invisible scores use a finite large negative value in the logits type, and query rows beyond
/// their lengths are exact zeros. Without a configured scale, attention uses `1 / sqrt(H)`. Every batch item uses the
/// same rectangular storage shape, and sequence lengths control visibility within that storage.
///
/// With [`AttentionConfiguration::return_residual`], a second `TN` or `BTN` output contains the log-sum-exp residual
/// (i.e., the natural logarithm of the sum of the exponentiated logits) at the query data type. With
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
/// rules, so differentiating a call fails. Use [`differentiable_dot_product_attention`] for the training path.
pub struct DotProductAttention<C: Domain>
where
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    /// Named attention definitions staged by this function.
    definitions: AttentionDefinitions<C::Constant, C::Operation>,
}

impl<C: Domain> DotProductAttention<C>
where
    C::Constant: 'static,
    C::Operation: 'static,
    C::Type: AttentionUniverse<DomainTracer<C>> + BatchableType,
    CustomRuleDefinition<C::Constant, C::Operation>: AttentionRules<C::Constant, C::Operation>,
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>
        + CompositeCallContext<Type = C::Type, Value = DomainTracer<C>, Constant = C::Constant, Operation = C::Operation>,
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
    pub fn call<Caller, X>(&self, inputs: AttentionInputs<X>) -> Result<(X, Option<X>), ProgramError>
    where
        Caller: CompositeCallContext<Type = C::Type, Value = X, Constant = C::Constant, Operation = C::Operation>,
        X: Value<Type = C::Type, Domain = Caller> + ManualVariationAlignment<C::Type>,
    {
        let inputs = align_attention_inputs::<C::Type, DomainTracer<C>, X>(inputs)?;
        let return_residual = self.definitions.configuration().return_residual();
        let mut outputs = self.definitions.forward(inputs, return_residual)?.into_iter();
        let output = outputs.next().ok_or(ProgramError::InvalidOutputCount { expected: 1, actual: 0 })?;
        Ok((output, outputs.next()))
    }
}

/// Creates the scaled dot-product attention inference function with the provided configuration for the values of the
/// domain `C`, whose calls stage the named forward attention definition (refer to [`DotProductAttention`]).
///
/// The definition's portable body computes attention with ordinary array operations, while backends may replace it
/// (e.g., the XLA backend replaces it with cuDNN's fused attention kernels when the configuration and the input types
/// are eligible). Homogeneous array domains support static extents only, while array IR domains also support dynamic
/// batch and sequence extents (e.g., in bounded dynamic programs).
///
/// # Parameters
///
///   - `configuration`: Value-independent attention semantics and implementation selection.
pub fn dot_product_attention<C: Domain>(configuration: AttentionConfiguration) -> DotProductAttention<C>
where
    C::Constant: 'static,
    C::Operation: 'static,
    C::Type: AttentionUniverse<DomainTracer<C>> + BatchableType,
    CustomRuleDefinition<C::Constant, C::Operation>: AttentionRules<C::Constant, C::Operation>,
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>
        + CompositeCallContext<Type = C::Type, Value = DomainTracer<C>, Constant = C::Constant, Operation = C::Operation>,
{
    DotProductAttention { definitions: AttentionDefinitions::new(configuration) }
}

/// Residuals retained by the reverse mode rules of differentiable attention. The type is public because it appears in
/// [`AttentionVjp`], and its contents are private to those rules.
#[derive(Clone, Debug, PartialEq, Parameterized)]
pub struct AttentionResiduals<P: Parameter> {
    /// Forward inputs, including the present optional inputs.
    inputs: AttentionInputs<P>,

    /// Attended forward output.
    output: P,

    /// Log-sum-exp residual of the forward output, which the backward definition consumes.
    residual: P,
}

/// Reverse mode configuration of the named composites that stage differentiable attention. It is public because the
/// bounds of [`differentiable_dot_product_attention`] name it.
pub type AttentionVjp<V, O> =
    SymbolicZeroVjp<V, O, AttentionInputs<RuleTracer<V, O>>, RuleTracer<V, O>, AttentionResiduals<RuleTracer<V, O>>>;

/// Differentiable scaled dot-product attention training function, built by [`differentiable_dot_product_attention`].
/// Each call stages one call of a named composite (refer to [`DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME`]) with
/// reverse mode rules, whose primal calls the forward attention definition and whose rules call the forward
/// definition with its residual and the backward definition (refer to [`AttentionKey`]). The manual variation
/// transitions of the inputs precede the composite boundary, which lets their transposes accumulate the cotangent
/// contributions of invariant inputs exactly once.
///
/// The function owns its named composites, one per optional input signature, together with their definitions, so it
/// should be created once and called many times. Its primal and its definitions' bodies are then traced once per input
/// types, and its rules once per specialization, for as long as the function lives.
pub struct DifferentiableDotProductAttention<C: Domain>
where
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
{
    /// Named attention definitions called by the rules of this function.
    definitions: Arc<AttentionDefinitions<C::Constant, C::Operation>>,

    /// Named composites that stage the calls of this function, by optional input signature.
    functions: Mutex<HashMap<AttentionInputSignature, Arc<TrainingFunction<C::Constant, C::Operation>>>>,
}

impl<C: Domain> DifferentiableDotProductAttention<C>
where
    C::Constant: 'static,
    C::Operation: 'static,
    C::Type: AttentionUniverse<DomainTracer<C>> + BatchableType + DifferentiableType,
    CustomRuleDefinition<C::Constant, C::Operation>: AttentionRules<C::Constant, C::Operation>
        + CompositeVjp<
            AttentionVjp<C::Constant, C::Operation>,
            DomainTracer<C>,
            AttentionInputs<DomainTracer<C>>,
            DomainTracer<C>,
        >,
    DomainTracingContext<C>: StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>
        + CompositeCallContext<Type = C::Type, Value = DomainTracer<C>, Constant = C::Constant, Operation = C::Operation>,
{
    /// Returns the configuration of this function.
    #[inline]
    pub fn configuration(&self) -> AttentionConfiguration {
        self.definitions.configuration()
    }

    /// Aligns the manual variation of `inputs` and stages one differentiable attention call, returning the attended
    /// output.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when `inputs` are invalid for the configuration (e.g., because their shapes are
    /// incompatible) or when staging the call fails.
    pub fn call<Caller, X>(&self, inputs: AttentionInputs<X>) -> Result<X, ProgramError>
    where
        Caller: CompositeCallContext<Type = C::Type, Value = X, Constant = C::Constant, Operation = C::Operation>,
        X: Value<Type = C::Type, Domain = Caller> + ManualVariationAlignment<C::Type>,
    {
        let signature = inputs.signature();
        let inputs = align_attention_inputs::<C::Type, DomainTracer<C>, X>(inputs)?;
        let function = self
            .functions
            .lock()
            .expect("attention functions mutex is poisoned")
            .entry(signature)
            .or_insert_with(|| Arc::new(training_function(self.definitions.clone(), signature)))
            .clone();
        function.call(inputs)
    }
}

/// Creates the differentiable scaled dot-product attention training function with the provided configuration for the
/// values of the domain `C`, for one canonical [`AttentionInputs`] tree.
///
/// The primal path returns only the attended output. Reverse mode differentiation runs the forward definition with its
/// log-sum-exp residual and then applies the backward definition, whose portable body recomputes the normalization
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
/// The returned function owns its named composites and their rules (refer to [`DifferentiableDotProductAttention`]),
/// so it should be created once and called many times. To embed the resulting programs in a heterogeneous value
/// universe, use [`Program::into_unprojected`](crate::programs::Program::into_unprojected), which converts the staged
/// calls together with their rules.
///
/// # Parameters
///
///   - `configuration`: Value-independent attention semantics and implementation selection.
pub fn differentiable_dot_product_attention<C: Domain>(
    configuration: AttentionConfiguration,
) -> DifferentiableDotProductAttention<C>
where
    C::Constant: 'static,
    C::Operation: 'static,
    C::Type: AttentionUniverse<DomainTracer<C>> + BatchableType + DifferentiableType,
    CustomRuleDefinition<C::Constant, C::Operation>: AttentionRules<C::Constant, C::Operation>
        + CompositeVjp<
            AttentionVjp<C::Constant, C::Operation>,
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
    DifferentiableDotProductAttention {
        definitions: Arc::new(AttentionDefinitions::new(configuration.with_residual(false))),
        functions: Mutex::new(HashMap::new()),
    }
}

/// [`CustomRuleTracer`] of the operation family `(V, O)`.
type RuleTracer<V, O> = CustomRuleTracer<V, O>;

/// Array view of a value of an [`AttentionUniverse`].
type ArrayOf<V> = <<V as Typed>::Type as AttentionUniverse<V>>::ArrayValue;

/// Extent of a value of an [`AttentionUniverse`].
type ExtentOf<V> = <<V as Typed>::Type as AttentionUniverse<V>>::Extent;

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
    /// Parses one forward operation boundary.
    fn forward(signature: AttentionInputSignature, input_types: &'o [ArrayType]) -> Result<Self, TypeError> {
        let [query, key, value, optional @ ..] = input_types else {
            return Err(TypeError::invalid("attention requires query, key, and value inputs"));
        };
        let mut index = 0;
        let bias = signature.has_bias().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let mask = signature.has_mask().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let query_sequence_lengths = signature.has_query_sequence_lengths().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let key_value_sequence_lengths = signature.has_key_value_sequence_lengths().then(|| {
            let value = &optional[index];
            index += 1;
            value
        });
        let dropout_state = signature.has_dropout_state().then(|| &optional[index]);
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

    /// Parses one backward operation boundary.
    fn backward(signature: AttentionInputSignature, input_types: &'o [ArrayType]) -> Result<Self, TypeError> {
        let optional_count = signature.count();
        let mut types = Self::forward(signature, &input_types[..3 + optional_count])?;
        types.output = Some(&input_types[3 + optional_count]);
        types.residual = Some(&input_types[4 + optional_count]);
        types.output_cotangent = Some(&input_types[5 + optional_count]);
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

/// Named attention definitions of one operation family `(V, O)` and one [`AttentionConfiguration`], with one named
/// composite per [`AttentionKey`] (i.e., per definition kind, residual request, and optional input signature), each
/// created on first use. The functions that stage attention own this cache, so its definitions, their traced bodies,
/// and their rule specializations live exactly as long as those functions.
struct AttentionDefinitions<V: Value, O: Operation<Type = V::Type>>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    /// Configuration of every definition in this cache. Forward definitions override its residual request.
    configuration: AttentionConfiguration,

    /// Forward definitions by residual request and optional input signature.
    forward: Mutex<HashMap<(bool, AttentionInputSignature), ForwardFunction<V, O>>>,

    /// Backward definitions by optional input signature.
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
    fn new(configuration: AttentionConfiguration) -> Self {
        Self { configuration, forward: Mutex::new(HashMap::new()), backward: Mutex::new(HashMap::new()) }
    }

    /// Returns the configuration of the definitions in this cache.
    #[inline]
    fn configuration(&self) -> AttentionConfiguration {
        self.configuration
    }

    /// Stages one call of the forward definition with the provided residual request at `inputs`, returning the
    /// attended output followed by its log-sum-exp residual when `return_residual` is set.
    fn forward<C, X>(&self, inputs: AttentionInputs<X>, return_residual: bool) -> Result<Vec<X>, ProgramError>
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
    fn backward<C, X>(&self, inputs: AttentionBackwardInputs<X>) -> Result<Vec<X>, ProgramError>
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

/// Named composite of a forward definition that batches its calls by folding.
type ForwardFolding<V, O> = BatchedComposite<V, O, AttentionInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>;

/// Named composite of a backward definition that batches its calls by folding.
type BackwardFolding<V, O> = BatchedComposite<V, O, AttentionBackwardInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>;

/// Named composite of a forward definition.
type ForwardFunction<V, O> =
    AttentionFunction<ForwardFolding<V, O>, Composite<V, O, AttentionInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>>;

/// Named composite of a backward definition.
type BackwardFunction<V, O> = AttentionFunction<
    BackwardFolding<V, O>,
    Composite<V, O, AttentionBackwardInputs<RuleTracer<V, O>>, Vec<RuleTracer<V, O>>>,
>;

/// Named composite that stages differentiable attention for one optional input signature.
type TrainingFunction<V, O> =
    Composite<V, O, AttentionInputs<RuleTracer<V, O>>, RuleTracer<V, O>, DefaultJvp, AttentionVjp<V, O>>;

/// Shared normalized inputs and masked logits consumed by the forward and backward attention bodies.
struct PreparedAttention<V: Value<Type: AttentionUniverse<V>>> {
    /// Normalized query.
    query: V,

    /// Normalized key.
    key: V,

    /// Normalized key expanded from key/value heads to query heads.
    expanded_key: V,

    /// Normalized value expanded from key/value heads to query heads.
    expanded_value: V,

    /// Masked logits converted to the data type used by softmax.
    logits: ArrayOf<V>,

    /// Shared query/key/value data type.
    data_type: DataType,

    /// Data type used by softmax and its log-sum-exp residual.
    softmax_type: DataType,

    /// Explicit or default score scale.
    scale: f64,
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

/// Returns the static [`Shape`] with the provided dimensions.
fn static_shape(dimensions: &[usize]) -> Shape {
    Shape::new(dimensions.iter().map(|&size| Dimension::Static(size)).collect())
}

/// Returns the [`CompositeKey`] named `name` with the attributes of `configuration` and `signature`, including the
/// `residual` attribute when `residual` is present (refer to [`AttentionKey`] for the encoding).
fn attention_composite_key(
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

/// Returns `inputs` with their manual variation aligned through [`ManualVariationAlignment::align_manual_variation`],
/// which precedes every call of the functions that stage attention. Alignment is the identity when every input already
/// varies over the same manual axes (e.g., outside manual regions), and so those inputs are returned unchanged without
/// staging the alignment.
fn align_attention_inputs<U, R, X>(inputs: AttentionInputs<X>) -> Result<AttentionInputs<X>, ProgramError>
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

/// Validates one forward attention boundary (i.e., the [`AttentionInputs`] of `signature` in canonical order) and
/// returns the types of its outputs. The first output is the attended output, which has the query type and therefore
/// propagates the query's metadata (e.g., its sharding), and it is followed by the log-sum-exp residual when
/// `configuration` requests it. Every input must have matching manual variation, and reduced and unreduced inputs are
/// rejected, because softmax and the log-sum-exp residual preserve neither partial sums nor replicas that are filled
/// with zeros.
fn attention_forward_output_types(
    operation_name: &str,
    configuration: AttentionConfiguration,
    signature: AttentionInputSignature,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    if input_types.len() != 3 + signature.count() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` expects {} inputs for its optional-input signature but got {}",
            3 + signature.count(),
            input_types.len(),
        )));
    }
    let types = AttentionInputTypes::forward(signature, input_types)?;
    let dimensions = validate_attention_boundary(operation_name, configuration, &types, input_types)?;
    let mut output_types = vec![types.query.clone()];
    if configuration.return_residual() {
        output_types.push(attention_residual_type(&dimensions, types.query)?);
    }
    Ok(output_types)
}

/// Validates one backward attention boundary (i.e., the [`AttentionInputs`] of `signature` in canonical order,
/// followed by the forward output, the log-sum-exp residual, and the output cotangent) and returns the types of its
/// outputs, which are the cotangents of the query, key, and value, followed by that of a present bias whose cotangent
/// space is not zero-dimensional. The forward output, residual, and output cotangent are compared by data type and
/// shape only, so that per-input metadata such as sharding never fails the structural contract.
fn attention_backward_output_types(
    operation_name: &str,
    configuration: AttentionConfiguration,
    signature: AttentionInputSignature,
    input_types: &[ArrayType],
) -> Result<Vec<ArrayType>, TypeError> {
    if input_types.len() != 6 + signature.count() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` expects {} inputs for its optional-input signature but got {}",
            6 + signature.count(),
            input_types.len(),
        )));
    }
    let types = AttentionInputTypes::backward(signature, input_types)?;
    let dimensions = validate_attention_boundary(operation_name, configuration, &types, input_types)?;
    let matches_expected = |actual: &ArrayType, expected: &ArrayType| -> bool {
        actual.data_type() == expected.data_type() && actual.shape() == expected.shape()
    };
    if !matches_expected(types.output.unwrap(), types.query) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` output type {} does not match the expected forward output type {}",
            types.output.unwrap(),
            types.query,
        )));
    }
    let expected_output_cotangent_type = types.query.cotangent()?;
    if !matches_expected(types.output_cotangent.unwrap(), &expected_output_cotangent_type) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` output cotangent type {} does not match the expected cotangent type \
             {expected_output_cotangent_type}",
            types.output_cotangent.unwrap(),
        )));
    }
    let expected_residual_type = attention_residual_type(&dimensions, types.query)?;
    if !matches_expected(types.residual.unwrap(), &expected_residual_type) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` residual type {} does not match the expected residual type \
             {expected_residual_type}",
            types.residual.unwrap(),
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
    validate_sequence_length_inputs(
        operation_name,
        types.query_sequence_lengths,
        types.key_value_sequence_lengths,
        &dimensions.batch,
    )?;
    validate_dropout(operation_name, configuration.dropout(), types.dropout_state)?;
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
    let query = attention_dimensions(operation_name, "query", query_type)?;
    let key = attention_dimensions(operation_name, "key", key_type)?;
    let value = attention_dimensions(operation_name, "value", value_type)?;
    let data_type = query_type.data_type();
    if !data_type.is_floating_point() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` requires floating-point inputs but got data type `{data_type}`"
        )));
    }
    for (descriptor, dimensions, input_type) in [("key", &key, key_type), ("value", &value, value_type)] {
        if input_type.data_type() != data_type {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} data type `{}` does not match the query data type `{data_type}`",
                input_type.data_type(),
            )));
        }
        if dimensions[0] != query[0] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} batch dimension ({}) does not match the query batch dimension \
                     ({})",
                dimensions[0], query[0],
            )));
        }
        if dimensions[3] != query[3] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} head dimension ({}) does not match the query head dimension \
                     ({})",
                dimensions[3], query[3],
            )));
        }
    }
    if value[1] != key[1] {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` value sequence dimension ({}) does not match the key sequence dimension ({})",
            value[1], key[1],
        )));
    }
    if value[2] != key[2] {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` value heads dimension ({}) does not match the key heads dimension ({})",
            value[2], key[2],
        )));
    }
    let Dimension::Static(query_heads) = &query[2] else {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` query heads dimension must be static but got {}",
            query[2],
        )));
    };
    let Dimension::Static(key_value_heads) = &key[2] else {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` key/value heads dimension must be static but got {}",
            key[2],
        )));
    };
    let Dimension::Static(head_dimension) = &query[3] else {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` head dimension must be static but got {}",
            query[3],
        )));
    };
    if *key_value_heads == 0 || *query_heads % *key_value_heads != 0 {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` key/value heads dimension ({}) must divide the query heads dimension ({})",
            key_value_heads, query_heads,
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
                "`{operation_name}` {descriptor} must have rank at most 4 but got rank {}",
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
                    "`{operation_name}` {descriptor} must have data type `{expected_data_type}` but got `{}`",
                    score_type.data_type(),
                )));
            }
        } else if !score_type.data_type().is_numeric() && !score_type.data_type().is_boolean() {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` bias must have a numeric or Boolean data type but got `{}`",
                score_type.data_type(),
            )));
        }
        if score_batch != &Dimension::Static(1) && score_batch != &query[0] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} batch dimension ({score_batch}) must be 1 or match the query \
                 batch dimension ({})",
                query[0],
            )));
        }
        if score_heads != &Dimension::Static(1) && score_heads != &query[2] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} heads dimension ({score_heads}) must be 1 or match the query \
                 heads dimension ({})",
                query[2],
            )));
        }
        if score_rows != &Dimension::Static(1) && score_rows != &query[1] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} query-sequence dimension ({score_rows}) must be 1 or match the \
                 query sequence dimension ({})",
                query[1],
            )));
        }
        if score_columns != &Dimension::Static(1) && score_columns != &key[1] {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} key/value-sequence dimension ({score_columns}) must be 1 or \
                 match the key sequence dimension ({})",
                key[1],
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
fn attention_dimensions(
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
            "`{operation_name}` {descriptor} must have rank 3 or 4 but got rank {}",
            dimensions.len(),
        ))),
    }
}

/// Validates the optional query and key/value sequence lengths that the attention definitions share. Each of them
/// must be a rank-1 `i32` array whose extent matches the shared batch dimension. Refer to [`DotProductAttention`] for
/// the padding semantics.
fn validate_sequence_length_inputs(
    operation_name: &str,
    query_lengths_type: Option<&ArrayType>,
    key_value_lengths_type: Option<&ArrayType>,
    batch: &Dimension,
) -> Result<(), TypeError> {
    for (descriptor, value_type) in
        [("query sequence lengths", query_lengths_type), ("key/value sequence lengths", key_value_lengths_type)]
    {
        let Some(value_type) = value_type else {
            continue;
        };
        if value_type.data_type() != DataType::I32 {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` {descriptor} must have data type `i32` but got `{}`",
                value_type.data_type(),
            )));
        }
        match value_type.shape().dimensions() {
            [size] if size == batch => {}
            [size] => {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` {descriptor} size ({size}) does not match the batch dimension ({batch})",
                )));
            }
            ref dimensions => {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` {descriptor} must have rank 1 but got rank {}",
                    dimensions.len(),
                )));
            }
        }
    }
    Ok(())
}

/// Validates the optional dropout rate and random state shared by the attention definitions. The rate must lie in the
/// open interval `(0, 1)` (a zero rate is expressed by omitting dropout altogether), a random state is only accepted
/// together with a rate, and it must have the state type of a [`RandomAlgorithm`].
fn validate_dropout(
    operation_name: &str,
    dropout: Option<f64>,
    dropout_state_type: Option<&ArrayType>,
) -> Result<(), TypeError> {
    if let Some(rate) = dropout
        && !(rate > 0.0 && rate < 1.0)
    {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` dropout rate must lie in the open interval (0, 1) but got {rate}",
        )));
    }
    if let Some(state_type) = dropout_state_type {
        if dropout.is_none() {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` received a dropout random state without a dropout rate",
            )));
        }
        RandomAlgorithm::from_state_type(state_type)?;
    }
    Ok(())
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

/// Returns the named composite of the forward definition with the provided configuration and optional input signature.
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
        let (output, residual) = dot_product_attention_forward_body(&inputs, configuration)?;
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
        let outputs = dot_product_attention_backward_body(
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

/// Returns the named composite that stages differentiable attention for the provided optional input signature, whose
/// primal and rules call the definitions in `definitions`.
fn training_function<V: Value, O: Operation<Type = V::Type>>(
    definitions: Arc<AttentionDefinitions<V, O>>,
    signature: AttentionInputSignature,
) -> TrainingFunction<V, O>
where
    V: 'static,
    O: 'static,
    V::Type: AttentionUniverse<RuleTracer<V, O>> + BatchableType + DifferentiableType,
    CustomRuleDefinition<V, O>: AttentionRules<V, O>,
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>
        + CompositeCallContext<Type = V::Type, Value = RuleTracer<V, O>, Constant = V, Operation = O>,
{
    /// Returns the structural zero cotangent of `value`.
    fn zero_cotangent<X: Value<Type: DifferentiableType>>(value: &X) -> Result<MaybeZero<X>, ProgramError> {
        Ok(MaybeZero::Zero(value.r#type().cotangent()?))
    }

    let configuration = definitions.configuration();
    let output_types = move |input_types: AttentionInputs<V::Type>| -> Result<V::Type, ProgramError> {
        let input_types = input_types
            .into_values()
            .iter()
            .map(<V::Type as AttentionUniverse<RuleTracer<V, O>>>::to_array_type)
            .collect::<Result<Vec<_>, _>>()?;
        let mut output_types = attention_forward_output_types(
            DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
            configuration,
            signature,
            input_types.as_slice(),
        )?;
        check_count!("output", output_types, 1, ProgramError);
        Ok(<V::Type as AttentionUniverse<RuleTracer<V, O>>>::from_array_type(output_types.remove(0)))
    };
    let primal = {
        let definitions = definitions.clone();
        move |inputs: AttentionInputs<RuleTracer<V, O>>| -> Result<RuleTracer<V, O>, ProgramError> {
            let mut outputs = definitions.forward(inputs, false)?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(outputs.remove(0))
        }
    };
    let forward = {
        let definitions = definitions.clone();
        move |inputs: AttentionInputs<RuleTracer<V, O>>| -> Result<_, ProgramError> {
            let mut outputs = definitions.forward(inputs.clone(), true)?;
            check_count!("output", outputs, 2, ProgramError);
            let residual = outputs.remove(1);
            let output = outputs.remove(0);
            Ok((output.clone(), AttentionResiduals { inputs, output, residual }))
        }
    };
    let backward = move |residuals: AttentionResiduals<RuleTracer<V, O>>,
                         output_cotangent: MaybeZero<RuleTracer<V, O>>|
          -> Result<_, ProgramError> {
        let AttentionResiduals { inputs, output, residual } = residuals;
        let MaybeZero::Value(output_cotangent) = output_cotangent else {
            return inputs.try_map_parameters(|input| zero_cotangent(&input));
        };

        // The backward definition returns the cotangents of the query, key, and value, and of a bias whose cotangent
        // space is not zero-dimensional, so the cotangents of the other inputs (i.e., of the masks, sequence lengths,
        // and dropout states, and of a bias with a zero-dimensional cotangent space) are structural zeros, which
        // contribute nothing to their inputs' cotangents.
        let zero = |input: &Option<RuleTracer<V, O>>| input.as_ref().map(zero_cotangent).transpose();
        let mask = zero(&inputs.mask)?;
        let query_sequence_lengths = zero(&inputs.query_sequence_lengths)?;
        let key_value_sequence_lengths = zero(&inputs.key_value_sequence_lengths)?;
        let dropout_state = zero(&inputs.dropout_state)?;
        let bias_cotangent_type = inputs.bias.as_ref().map(|bias| bias.r#type().cotangent()).transpose()?;
        let mut outputs =
            definitions.backward(AttentionBackwardInputs { inputs, output, residual, output_cotangent })?;
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
    };
    composite(primal)
        .with_symbolic_zero_vjp(forward, backward)
        .with_name(DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME)
        .with_named_composite(
            attention_composite_key(DOT_PRODUCT_ATTENTION_TRAINING_COMPOSITE_NAME, configuration, signature, None),
            output_types,
        )
}

/// Returns the array types of `types`, which belong to the universe of the values `V`.
fn array_types<V: Value<Type: AttentionUniverse<V>>>(types: Vec<V::Type>) -> Result<Vec<ArrayType>, ProgramError> {
    Ok(types.iter().map(<V::Type as AttentionUniverse<V>>::to_array_type).collect::<Result<Vec<_>, _>>()?)
}

/// Returns an error for a call of a definition whose optional input signature is `actual` rather than its own.
fn signature_mismatch(name: &str, expected: AttentionInputSignature, actual: AttentionInputSignature) -> ProgramError {
    TypeError::invalid(format!(
        "named composite `{name}` was defined for the optional-input signature {expected:?} but was called with \
         {actual:?}",
    ))
    .into()
}

/// Evaluates portable scaled dot-product attention using ordinary array operations, returning the attended output
/// together with the log-sum-exp residual when `configuration` requests it.
///
/// This is the body of the forward named attention definition (refer to [`AttentionKey`]). Every constructor,
/// broadcast, and reshape that depends on a shape takes its extents from the value's [`AttentionUniverse`], so the
/// same body supports static extents in homogeneous array families and static and dynamic extents in array IR
/// families, and backends need no attention mathematics of their own.
fn dot_product_attention_forward_body<V: Value<Type: AttentionUniverse<V>>>(
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
) -> Result<(ArrayOf<V>, Option<ArrayOf<V>>), ProgramError> {
    let input_types = attention_input_types(inputs)?;
    attention_forward_output_types(
        DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
        configuration,
        inputs.signature(),
        input_types.as_slice(),
    )?;
    let query_rank = input_types[0].rank();
    let dimensions = validate_attention_inputs(
        DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
        &input_types[0],
        &input_types[1],
        &input_types[2],
        inputs.bias.as_ref().map(|_| &input_types[3]),
        inputs.mask.as_ref().map(|_| &input_types[3 + usize::from(inputs.bias.is_some())]),
    )?;

    let prepared = prepare_attention(inputs, configuration, &dimensions)?;
    let logits = value::<V>(prepared.logits.clone());
    let maxima = prepared.logits.reduce(&[3], ReductionKind::Max)?;
    let maxima_broadcast = array(broadcast_like::<V>(maxima.clone(), &logits, &[0, 1, 2])?)?;
    let shifted = prepared.logits.sub(&maxima_broadcast)?;
    let exponentials = shifted.exp()?;
    let exponentials_value = value::<V>(exponentials.clone());
    let sums = exponentials.reduce(&[3], ReductionKind::Sum)?;
    let sums_broadcast = array(broadcast_like::<V>(sums.clone(), &exponentials_value, &[0, 1, 2])?)?;
    let weights = apply_dropout(exponentials.div(&sums_broadcast)?, inputs, configuration)?;
    let weights = if prepared.data_type == prepared.softmax_type {
        weights
    } else {
        weights.convert_element_type(prepared.data_type)?
    };
    let attended = weights
        .dot(&array(prepared.expanded_value)?, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]))?;
    let mut output = attended.transpose([0, 2, 1, 3])?;
    if let Some(query_lengths) = &inputs.query_sequence_lengths {
        output = zero_query_rows::<V>(output, query_lengths, 1)?;
    }
    let output = denormalize_attention_output::<V>(output, query_rank)?;
    let residual = if configuration.return_residual() {
        let residual = maxima.add(&sums.log()?)?;
        let residual = if prepared.softmax_type == prepared.data_type {
            residual
        } else {
            residual.convert_element_type(prepared.data_type)?
        };
        let residual = residual.transpose([0, 2, 1])?;
        Some(denormalize_attention_output::<V>(residual, query_rank)?.stop_gradient()?)
    } else {
        None
    };
    Ok((output, residual))
}

/// Evaluates the portable attention backward pass using the same extents and mask construction as
/// [`dot_product_attention_forward_body`], returning the cotangents of the query, key, and value, followed by that of
/// a present bias whose cotangent space is not zero-dimensional. This is the body of the backward named attention
/// definition (refer to [`AttentionKey`]), whose inputs are the forward inputs followed by the attended output, its
/// `TN` or `BTN` log-sum-exp residual, and the output cotangent. Boolean masks, integer sequence lengths, dropout
/// states, and a bias whose cotangent space is zero-dimensional have no cotangent outputs.
///
/// The body recomputes the normalized probabilities `P` from the masked logits in `f32`, which avoids the precision
/// that rounding the output or the residual to a narrower query data type loses, and so the output and the residual
/// are validated inputs for backend implementations whose numerical values this body does not use. It computes
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
/// reproduce. Higher-order derivatives differentiate this body directly.
fn dot_product_attention_backward_body<V: Value<Type: AttentionUniverse<V>>>(
    inputs: &AttentionInputs<V>,
    output: &V,
    residual: &V,
    output_cotangent: &V,
    configuration: AttentionConfiguration,
) -> Result<Vec<ArrayOf<V>>, ProgramError> {
    let mut input_types = attention_input_types(inputs)?;
    input_types.extend([
        V::Type::array_type(output)?,
        V::Type::array_type(residual)?,
        V::Type::array_type(output_cotangent)?,
    ]);
    attention_backward_output_types(
        DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME,
        configuration,
        inputs.signature(),
        input_types.as_slice(),
    )?;
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
    let output_cotangent = normalize_attention_input(output_cotangent)?;
    let logits = value::<V>(prepared.logits.clone());
    // The public residual and output have the query data type, which can lose the normalization correction in
    // narrow formats. Recompute probabilities in the softmax type instead of recovering them from those values.
    let maxima = prepared.logits.reduce(&[3], ReductionKind::Max)?;
    let maxima = array(broadcast_like::<V>(maxima, &logits, &[0, 1, 2])?)?;
    let exponentials = prepared.logits.sub(&maxima)?.exp()?;
    let sums = exponentials.reduce(&[3], ReductionKind::Sum)?;
    let sums = array(broadcast_like::<V>(sums, &logits, &[0, 1, 2])?)?;
    let weights = exponentials.div(&sums)?;

    // Pick exactly one maximum, including when logits tie. Keeping the pivot out of the final sum avoids
    // subtracting two nearly equal dominant contributions. The integer constants cannot carry tangents.
    let columns = V::Type::iota(&logits, 3)?;
    let column_array = array(columns.clone())?;
    let sentinel = array(V::Type::fill(&columns, i32::MAX as f64)?)?;
    let is_maximum = prepared.logits.compare(&maxima, ComparisonDirection::Equal)?;
    let pivot = is_maximum.select(&column_array, &sentinel)?.reduce(&[3], ReductionKind::Min)?;
    let pivot = array(broadcast_like::<V>(pivot, &logits, &[0, 1, 2])?)?;
    let is_pivot = column_array.compare(&pivot, ComparisonDirection::Equal)?;

    let output_cotangent = match &inputs.query_sequence_lengths {
        None => array(output_cotangent)?,
        Some(query_lengths) => zero_query_rows::<V>(array(output_cotangent)?, query_lengths, 1)?,
    };
    let convert = |array: ArrayOf<V>| -> Result<ArrayOf<V>, ProgramError> {
        if prepared.data_type == prepared.softmax_type {
            Ok(array)
        } else {
            Ok(array.convert_element_type(prepared.softmax_type)?)
        }
    };
    let softmax_query = convert(array(prepared.query.clone())?)?;
    let softmax_key = convert(array(prepared.expanded_key.clone())?)?;
    let softmax_value = convert(array(prepared.expanded_value)?)?;
    let softmax_output_cotangent = convert(output_cotangent)?;

    // With dropout, the output uses the dropped probabilities `D = dropout(P)`, so the value cotangent uses `D` and the
    // cotangent of `P` is the cotangent of `D` through the same mask.
    let dropped_weights = apply_dropout(weights.clone(), inputs, configuration)?;
    let weight_cotangents = apply_dropout(
        softmax_output_cotangent
            .dot(&softmax_value, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]))?,
        inputs,
        configuration,
    )?;
    let zero = array(V::Type::fill(&logits, 0.0)?)?;
    let pivot_cotangent = is_pivot.select(&weight_cotangents, &zero)?.reduce(&[3], ReductionKind::Sum)?;
    let pivot_cotangent = array(broadcast_like::<V>(pivot_cotangent, &logits, &[0, 1, 2])?)?;
    // Center the incoming weight cotangents before their weighted sum. This preserves a tiny dominant derivative
    // and gives exact zero when every weight cotangent is equal, even if the rounded probabilities do not sum to one.
    let centered = weight_cotangents.sub(&pivot_cotangent)?;
    let delta = weights.mul(&centered)?.reduce(&[3], ReductionKind::Sum)?;
    let delta = array(broadcast_like::<V>(delta, &logits, &[0, 1, 2])?)?;
    let logit_cotangents = weights.mul(&centered.sub(&delta)?)?;
    let other_cotangents = is_pivot.select(&zero, &logit_cotangents)?;
    let pivot_cotangent = other_cotangents.reduce(&[3], ReductionKind::Sum)?;
    let pivot_cotangent = array(broadcast_like::<V>(pivot_cotangent, &logits, &[0, 1, 2])?)?;
    let logit_cotangents = is_pivot.select(&zero.sub(&pivot_cotangent)?, &other_cotangents)?;
    let logit_cotangents_value = value::<V>(logit_cotangents.clone());
    let scale_value = array(V::Type::fill(&logit_cotangents_value, prepared.scale)?)?;
    let scaled_logit_cotangents = logit_cotangents.mul(&scale_value)?;
    let mut query_cotangent = scaled_logit_cotangents
        .dot(&softmax_key, &DotDimensionNumbers::new(vec![3], vec![1], vec![0, 1], vec![0, 2]))?
        .transpose([0, 2, 1, 3])?;
    if let Some(query_lengths) = &inputs.query_sequence_lengths {
        query_cotangent = zero_query_rows::<V>(query_cotangent, query_lengths, 1)?;
    }
    let key_cotangent = scaled_logit_cotangents
        .dot(&softmax_query, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))?
        .transpose([0, 2, 1, 3])?;
    let value_cotangent = dropped_weights
        .dot(&softmax_output_cotangent, &DotDimensionNumbers::new(vec![2], vec![1], vec![0, 1], vec![0, 2]))?
        .transpose([0, 2, 1, 3])?;

    let (key_cotangent, value_cotangent) = if dimensions.key_value_heads == dimensions.query_heads {
        (key_cotangent, value_cotangent)
    } else {
        let group = dimensions.query_heads / dimensions.key_value_heads;
        let key_extents = extents(&prepared.key)?;
        let key_value_heads = V::Type::constant_extent(&prepared.key, dimensions.key_value_heads)?;
        let group = V::Type::constant_extent(&prepared.key, group)?;
        let head_dimension = V::Type::constant_extent(&prepared.key, dimensions.head_dimension)?;
        let grouped_extents = [key_extents[0].clone(), key_extents[1].clone(), key_value_heads, group, head_dimension];
        let grouped_key = V::Type::reshape(&value::<V>(key_cotangent), &grouped_extents)?;
        let grouped_value = V::Type::reshape(&value::<V>(value_cotangent), &grouped_extents)?;
        (array(grouped_key)?.reduce(&[3], ReductionKind::Sum)?, array(grouped_value)?.reduce(&[3], ReductionKind::Sum)?)
    };
    let mut cotangents = [query_cotangent, key_cotangent, value_cotangent]
        .into_iter()
        .zip(input_types[..3].iter().zip(input_ranks))
        .map(|(cotangent, (input_type, rank))| {
            let cotangent_data_type = input_type.cotangent()?.data_type();
            let cotangent = if cotangent.r#type().data_type() == cotangent_data_type {
                cotangent
            } else {
                cotangent.convert_element_type(cotangent_data_type)?
            };
            denormalize_attention_output::<V>(cotangent, rank)
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    if let Some(bias) = &inputs.bias {
        let bias_type = V::Type::array_type(bias)?;
        let bias_cotangent_type = bias_type.cotangent()?;
        if bias_cotangent_type.is_zero_space() {
            return Ok(cotangents);
        }
        let bias_dimensions = std::iter::repeat_n(Dimension::Static(1), 4 - bias_type.rank())
            .chain(bias_type.shape().dimensions().iter().cloned())
            .collect::<Vec<_>>();
        let logit_type = logit_cotangents.r#type().into_owned();
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
        let bias_cotangent = array(V::Type::reshape(&value::<V>(bias_cotangent), extents(bias)?.as_slice())?)?;
        let bias_cotangent = if bias_cotangent_type.data_type() == prepared.softmax_type {
            bias_cotangent
        } else {
            bias_cotangent.convert_element_type(bias_cotangent_type.data_type())?
        };
        cotangents.push(bias_cotangent);
    }
    Ok(cotangents)
}

/// Returns the canonical input types of `inputs`.
fn attention_input_types<V: Value<Type: AttentionUniverse<V>>>(
    inputs: &AttentionInputs<V>,
) -> Result<Vec<ArrayType>, ProgramError> {
    inputs.clone().into_values().iter().map(V::Type::array_type).collect()
}

/// Normalizes attention inputs and constructs the masked score logits shared by the forward and backward bodies.
fn prepare_attention<V: Value<Type: AttentionUniverse<V>>>(
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
    dimensions: &AttentionDimensions,
) -> Result<PreparedAttention<V>, ProgramError> {
    let query = normalize_attention_input(&inputs.query)?;
    let key = normalize_attention_input(&inputs.key)?;
    let value_input = normalize_attention_input(&inputs.value)?;
    let expanded_key = expand_key_value_heads(&key, dimensions)?;
    let expanded_value = expand_key_value_heads(&value_input, dimensions)?;
    let data_type = dimensions.data_type;
    // Dot products, scaling, bias, and masking promote low precision inputs to `f32`, while `f64` remains `f64`.
    // The stable softmax itself always runs in `f32`, matching the public attention contract.
    let logits_type = if data_type == DataType::F64 { DataType::F64 } else { DataType::F32 };
    let softmax_type = DataType::F32;
    let scores = array(query.clone())?
        .dot(&array(expanded_key.clone())?, &DotDimensionNumbers::new(vec![3], vec![3], vec![0, 2], vec![0, 2]))?;
    let scores = if data_type == logits_type { scores } else { scores.convert_element_type(logits_type)? };
    let mut scores = value::<V>(scores);
    let scale = configuration.scale().unwrap_or(1.0 / (dimensions.head_dimension as f64).sqrt());
    let scale_value = array(V::Type::fill(&scores, scale)?)?;
    scores = value::<V>(array(scores)?.mul(&scale_value)?);
    if let Some(bias) = &inputs.bias {
        let bias = array(normalize_attention_score_input(bias)?)?;
        let bias =
            if bias.r#type().data_type() == logits_type { bias } else { bias.convert_element_type(logits_type)? };
        let bias = array(broadcast_like::<V>(bias, &scores, &[0, 1, 2, 3])?)?;
        scores = value::<V>(array(scores)?.add(&bias)?);
    }
    let logits = apply_masks(
        scores,
        inputs.mask.as_ref(),
        configuration,
        inputs.query_sequence_lengths.as_ref(),
        inputs.key_value_sequence_lengths.as_ref(),
    )?;
    let logits = array(logits)?;
    let logits = if logits_type == softmax_type { logits } else { logits.convert_element_type(softmax_type)? };
    Ok(PreparedAttention { query, key, expanded_key, expanded_value, logits, data_type, softmax_type, scale })
}

/// Adds the implicit batch axis used to evaluate one unbatched `TNH` attention input (i.e., a query, key, value,
/// forward output, or output cotangent).
fn normalize_attention_input<V: Value<Type: AttentionUniverse<V>>>(input: &V) -> Result<V, ProgramError> {
    if V::Type::array_type(input)?.rank() == 4 {
        return Ok(input.clone());
    }
    let mut input_extents = extents(input)?;
    input_extents.insert(0, V::Type::constant_extent(input, 1)?);
    V::Type::reshape(input, input_extents.as_slice())
}

/// Expands grouped key/value heads to the query heads through broadcast and reshape extents.
fn expand_key_value_heads<V: Value<Type: AttentionUniverse<V>>>(
    key_or_value: &V,
    dimensions: &AttentionDimensions,
) -> Result<V, ProgramError> {
    if dimensions.key_value_heads == dimensions.query_heads {
        return Ok(key_or_value.clone());
    }
    let group = dimensions.query_heads / dimensions.key_value_heads;
    let key_or_value_extents = extents(key_or_value)?;
    let key_value_heads = V::Type::constant_extent(key_or_value, dimensions.key_value_heads)?;
    let group = V::Type::constant_extent(key_or_value, group)?;
    let query_heads = V::Type::constant_extent(key_or_value, dimensions.query_heads)?;
    let expanded = V::Type::broadcast(
        key_or_value,
        &[
            key_or_value_extents[0].clone(),
            key_or_value_extents[1].clone(),
            key_value_heads,
            group,
            key_or_value_extents[3].clone(),
        ],
        &[0, 1, 2, 4],
    )?;
    V::Type::reshape(
        &expanded,
        &[
            key_or_value_extents[0].clone(),
            key_or_value_extents[1].clone(),
            query_heads,
            key_or_value_extents[3].clone(),
        ],
    )
}

/// Prepends singleton axes until one bias or mask input has the normalized `BNTS` score rank.
fn normalize_attention_score_input<V: Value<Type: AttentionUniverse<V>>>(input: &V) -> Result<V, ProgramError> {
    let rank = V::Type::array_type(input)?.rank();
    if rank == 4 {
        return Ok(input.clone());
    }
    let singleton = V::Type::constant_extent(input, 1)?;
    let mut input_extents = vec![singleton; 4 - rank];
    input_extents.extend(extents(input)?);
    V::Type::reshape(input, input_extents.as_slice())
}

/// Applies the explicit mask, the causal and local window masks, and the sequence length masks to the attention
/// logits.
fn apply_masks<V: Value<Type: AttentionUniverse<V>>>(
    scores: V,
    mask: Option<&V>,
    configuration: AttentionConfiguration,
    query_sequence_lengths: Option<&V>,
    key_value_sequence_lengths: Option<&V>,
) -> Result<V, ProgramError> {
    if mask.is_none()
        && !configuration.causal()
        && configuration.local_window().is_none()
        && query_sequence_lengths.is_none()
        && key_value_sequence_lengths.is_none()
    {
        return Ok(scores);
    }
    let columns = V::Type::iota(&scores, 3)?;
    let mut visible = match mask {
        None => None,
        Some(mask) => {
            let mask = array(normalize_attention_score_input(mask)?)?;
            Some(array(broadcast_like::<V>(mask, &scores, &[0, 1, 2, 3])?)?)
        }
    };
    if configuration.causal() {
        let rows = array(V::Type::iota(&scores, 2)?)?;
        let causal = array(columns.clone())?.compare(&rows, ComparisonDirection::LessThanOrEqual)?;
        visible = Some(match visible {
            None => causal,
            Some(visible) => visible.and(&causal)?,
        });
    }
    if let Some((left, right)) = configuration.local_window() {
        let rows = V::Type::iota(&scores, 2)?;
        let columns = array(columns.clone())?;
        let left = array(V::Type::fill(&rows, left as f64)?)?;
        let right = array(V::Type::fill(&rows, right as f64)?)?;
        let rows = array(rows)?;
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
    if let Some(lengths) = query_sequence_lengths {
        let rows = V::Type::iota(&scores, 2)?;
        let bounds = V::Type::broadcast(lengths, extents(&rows)?.as_slice(), &[0])?;
        let in_range = array(rows)?.compare(&array(bounds)?, ComparisonDirection::LessThan)?;
        visible = Some(match visible {
            None => in_range,
            Some(visible) => visible.and(&in_range)?,
        });
    }
    if let Some(lengths) = key_value_sequence_lengths {
        let bounds = V::Type::broadcast(lengths, extents(&columns)?.as_slice(), &[0])?;
        let in_range = array(columns)?.compare(&array(bounds)?, ComparisonDirection::LessThan)?;
        visible = Some(match visible {
            None => in_range,
            Some(visible) => visible.and(&in_range)?,
        });
    }
    let score_type = array(scores.clone())?.r#type().into_owned();
    let large_negative = if score_type.data_type() == DataType::F64 { -0.7 * f64::MAX } else { -0.7 * f32::MAX as f64 };
    let masked = array(V::Type::fill(&scores, large_negative)?)?;
    Ok(value::<V>(visible.unwrap().select(&array(scores)?, &masked)?))
}

/// Applies the configured dropout to `probabilities`, whose shape is the normalized `BNTS` score shape, using the
/// dropout random state of `inputs` (refer to [`AttentionInputs::dropout_state`]). The forward and backward bodies call
/// this with arrays of the same shape and the same state, so they apply the same mask.
fn apply_dropout<V: Value<Type: AttentionUniverse<V>>>(
    probabilities: ArrayOf<V>,
    inputs: &AttentionInputs<V>,
    configuration: AttentionConfiguration,
) -> Result<ArrayOf<V>, ProgramError> {
    let Some(rate) = configuration.dropout() else {
        return Ok(probabilities);
    };
    let state = inputs.dropout_state.clone().map(array).transpose()?;
    Ok(probabilities.dropout(rate, state.as_ref())?.0)
}

/// Zeros query rows outside each batch item's logical sequence length.
fn zero_query_rows<V: Value<Type: AttentionUniverse<V>>>(
    rows_value: ArrayOf<V>,
    query_lengths: &V,
    row_axis: usize,
) -> Result<ArrayOf<V>, ProgramError> {
    let rows_value = value::<V>(rows_value);
    let rows = V::Type::iota(&rows_value, row_axis)?;
    let lengths = V::Type::broadcast(query_lengths, extents(&rows_value)?.as_slice(), &[0])?;
    let in_range = array(rows)?.compare(&array(lengths)?, ComparisonDirection::LessThan)?;
    let zero = array(V::Type::fill(&rows_value, 0.0)?)?;
    in_range.select(&array(rows_value)?, &zero)
}

/// Removes the implicit batch axis after evaluating unbatched attention.
fn denormalize_attention_output<V: Value<Type: AttentionUniverse<V>>>(
    output: ArrayOf<V>,
    rank: usize,
) -> Result<ArrayOf<V>, ProgramError> {
    if rank == 4 {
        return Ok(output);
    }
    let output = value::<V>(output);
    let output_extents = extents(&output)?;
    array(V::Type::reshape(&output, &output_extents[1..])?)
}

/// Broadcasts one array to the exact shape of `target`, where `axes` names the output axis of every input axis.
fn broadcast_like<V: Value<Type: AttentionUniverse<V>>>(
    array: ArrayOf<V>,
    target: &V,
    axes: &[usize],
) -> Result<V, ProgramError> {
    V::Type::broadcast(&value::<V>(array), extents(target)?.as_slice(), axes)
}

/// Returns the extent of every axis of `value`.
fn extents<V: Value<Type: AttentionUniverse<V>>>(value: &V) -> Result<Vec<ExtentOf<V>>, ProgramError> {
    (0..V::Type::array_type(value)?.rank()).map(|axis| V::Type::extent(value, axis)).collect()
}

/// Returns the array view of `value`.
fn array<V: Value<Type: AttentionUniverse<V>>>(value: V) -> Result<ArrayOf<V>, ProgramError> {
    V::Type::into_array(value)
}

/// Returns the value whose array view is `array`.
fn value<V: Value<Type: AttentionUniverse<V>>>(array: ArrayOf<V>) -> V {
    V::Type::from_array(array)
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
        roles.extend([AttentionBatchInput::Tensor, AttentionBatchInput::Residual, AttentionBatchInput::Tensor]);
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
        AttentionDefinitionKind::Forward => [AttentionBatchOutput::Tensor(0), AttentionBatchOutput::Residual]
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
/// Every input is first aligned to a leading mapped axis and then folded into the logical batch axis. Rank-3
/// attention has an implicit logical batch of one, so `[v, T, N, H]` already has the canonical rank-4 form, while
/// rank-4 attention folds `[v, B, T, N, H]` to `[v · B, T, N, H]`. Biases and masks are first materialized to
/// `[v, B, N, T, S]`, which handles every broadcastable rank and preserves mapped bias and mask inputs. Outputs reverse
/// the normalization, and a bias cotangent is reduced over precisely the axes that its logical bias input broadcasts.
/// Every extent comes from the values' [`AttentionUniverse`], so this supports static extents and, in array IR
/// families, first-class dynamic extents (i.e., a dynamic mapped extent, logical batch, or sequence length never
/// becomes host metadata). Calls whose inputs are all replicated stage the unbatched definition.
fn batch_attention<V: Value<Type: AttentionUniverse<V>>>(
    inputs: Vec<V>,
    axes: Vec<BatchAxis>,
    extent: &BatchingLevelExtent<V>,
    input_roles: &[AttentionBatchInput],
    output_roles: impl FnOnce(usize) -> Vec<AttentionBatchOutput>,
    call: impl FnOnce(Vec<V>) -> Result<Vec<V>, ProgramError>,
) -> Result<(Vec<V>, Vec<BatchAxis>), ProgramError> {
    check_count!("input", input_roles, inputs.len(), ProgramError);
    check_count!("input", axes, inputs.len(), ProgramError);
    if axes.iter().all(BatchAxis::is_replicated) {
        let outputs = call(inputs)?;
        let output_axes = vec![BatchAxis::replicated(); outputs.len()];
        return Ok((outputs, output_axes));
    }

    // Align every input to a leading mapped axis, moving mapped axes and broadcasting replicated inputs.
    let mapped_extent = V::Type::batch_extent(&inputs[0], extent)?;
    let aligned = inputs
        .iter()
        .zip(&axes)
        .map(|(input, axis)| {
            let rank = V::Type::array_type(input)?.rank();
            match axis.axis() {
                Some(axis) => {
                    let position = axis.normalize(rank).map_err(|error| TypeError::invalid(error.to_string()))?;
                    if position == 0 {
                        return Ok(input.clone());
                    }
                    let permutation =
                        std::iter::once(position).chain((0..rank).filter(|axis| *axis != position)).collect::<Vec<_>>();
                    Ok(V::Type::from_array(V::Type::into_array(input.clone())?.transpose(permutation)?))
                }
                None => {
                    let output_extents =
                        std::iter::once(mapped_extent.clone()).chain(extents(input)?).collect::<Vec<_>>();
                    V::Type::broadcast(input, output_extents.as_slice(), (1..=rank).collect::<Vec<_>>().as_slice())
                }
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let logical_rank = |value: &V| -> Result<usize, ProgramError> { Ok(V::Type::array_type(value)?.rank() - 1) };
    let query_rank = logical_rank(&aligned[0])?;
    let key_rank = logical_rank(&aligned[1])?;
    let batch_extent =
        if query_rank == 4 { V::Type::extent(&aligned[0], 1)? } else { V::Type::constant_extent(&aligned[0], 1)? };
    let folded_batch_extent = V::Type::multiply_extents(&aligned[0], &mapped_extent, &batch_extent)?;

    // Biases and masks are normalized to the canonical `[v, B, N, T, S]` score shape, whose extents are obtained only
    // when the call has such inputs (and therefore possibly a bias cotangent output).
    let score_extents = if input_roles.iter().any(|role| matches!(role, AttentionBatchInput::Score)) {
        let query_sequence_extent = V::Type::extent(&aligned[0], query_rank - 2)?;
        let head_extent = V::Type::extent(&aligned[0], query_rank - 1)?;
        let key_sequence_extent = V::Type::extent(&aligned[1], key_rank - 2)?;
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
            let rank = V::Type::array_type(value)?.rank();
            let trailing_extents = |start: usize| {
                (start..rank).map(|axis| V::Type::extent(value, axis)).collect::<Result<Vec<_>, ProgramError>>()
            };
            match role {
                AttentionBatchInput::Tensor if rank == 5 => {
                    let batch_extent = V::Type::multiply_extents(value, &mapped_extent, &V::Type::extent(value, 1)?)?;
                    let folded_extents = std::iter::once(batch_extent).chain(trailing_extents(2)?).collect::<Vec<_>>();
                    V::Type::reshape(value, folded_extents.as_slice())
                }
                AttentionBatchInput::Tensor => Ok(value.clone()),
                AttentionBatchInput::Score => {
                    let (score_extents, folded_score_extents) = score_extents()?;
                    let output_axes =
                        std::iter::once(0).chain((1..rank).map(|axis| 5 - rank + axis)).collect::<Vec<_>>();
                    let normalized = V::Type::broadcast(value, score_extents, output_axes.as_slice())?;
                    V::Type::reshape(&normalized, folded_score_extents)
                }
                AttentionBatchInput::Length => V::Type::reshape(value, std::slice::from_ref(&folded_batch_extent)),
                AttentionBatchInput::Residual if rank == 4 => {
                    let folded_extents =
                        std::iter::once(folded_batch_extent.clone()).chain(trailing_extents(2)?).collect::<Vec<_>>();
                    V::Type::reshape(value, folded_extents.as_slice())
                }
                AttentionBatchInput::Residual => Ok(value.clone()),
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
            let trailing_extents = |output: &V| {
                let rank = V::Type::array_type(output)?.rank();
                (1..rank).map(|axis| V::Type::extent(output, axis)).collect::<Result<Vec<_>, ProgramError>>()
            };
            match role {
                AttentionBatchOutput::Tensor(input_index) if logical_rank(&aligned[input_index])? == 4 => {
                    let input_batch_extent = V::Type::extent(&aligned[input_index], 1)?;
                    let unfolded_extents = [mapped_extent.clone(), input_batch_extent]
                        .into_iter()
                        .chain(trailing_extents(&output)?)
                        .collect::<Vec<_>>();
                    V::Type::reshape(&output, unfolded_extents.as_slice())
                }
                AttentionBatchOutput::Tensor(_) => Ok(output),
                AttentionBatchOutput::Residual if query_rank == 4 => {
                    let unfolded_extents = [mapped_extent.clone(), batch_extent.clone()]
                        .into_iter()
                        .chain(trailing_extents(&output)?)
                        .collect::<Vec<_>>();
                    V::Type::reshape(&output, unfolded_extents.as_slice())
                }
                AttentionBatchOutput::Residual => Ok(output),
                AttentionBatchOutput::Score(input_index) => {
                    let bias_type = V::Type::array_type(&aligned[input_index])?;
                    let offset = 5 - bias_type.rank();
                    let reduction_axes = (0..4)
                        .filter(|&axis| {
                            axis < offset || matches!(bias_type.shape()[axis - offset + 1], Dimension::Static(1))
                        })
                        .map(|axis| axis + 1)
                        .collect::<Vec<_>>();
                    let normalized = V::Type::reshape(&output, &score_extents()?.0)?;
                    let reduced =
                        V::Type::into_array(normalized)?.reduce(reduction_axes.as_slice(), ReductionKind::Sum)?;
                    V::Type::reshape(&V::Type::from_array(reduced), extents(&aligned[input_index])?.as_slice())
                }
            }
        })
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let output_axes = vec![BatchAxis::from_position(0); outputs.len()];
    Ok((outputs, output_axes))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::bf16;
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, DimensionBounds, DimensionValue, DimensionVariable,
        LogicalMesh, MeshAxis, MeshAxisType, Sharding,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchingError, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{DifferentiationError, differentiate_at};
    use crate::parameters::Placeholder;
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

    /// Returns the outputs of one eager attention call with the provided configuration.
    fn attend(inputs: AttentionInputs<Array>, configuration: AttentionConfiguration) -> (Array, Option<Array>) {
        dot_product_attention::<ArrayContext>(configuration).call(inputs).unwrap()
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
    fn test_attention_universe() {
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let array = f32_array(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Homogeneous array values are their own array views, and their types are array types.
        assert_eq!(<ArrayType as AttentionUniverse<Array>>::into_array(array.clone()), Ok(array.clone()));
        assert_eq!(<ArrayType as AttentionUniverse<Array>>::from_array(array.clone()), array);
        assert_eq!(<ArrayType as AttentionUniverse<Array>>::array_type(&array), Ok(r#type.clone()));
        assert_eq!(<ArrayType as AttentionUniverse<Array>>::to_array_type(&r#type), Ok(r#type.clone()));
        assert_eq!(<ArrayType as AttentionUniverse<Array>>::from_array_type(r#type.clone()), r#type);

        // Extents are static host numbers, so dynamic batching extents and overflowing products are rejected.
        assert_eq!(
            <ArrayType as AttentionUniverse<Array>>::batch_extent(&array, &BatchingLevelExtent::Static(4)),
            Ok(4),
        );
        assert!(matches!(
            <ArrayType as AttentionUniverse<Array>>::batch_extent(&array, &BatchingLevelExtent::Dynamic(array.clone())),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "attention over homogeneous arrays cannot be batched with a dynamic extent; use array IR \
                               values for dynamic extents",
        ));
        assert_eq!(<ArrayType as AttentionUniverse<Array>>::extent(&array, 1), Ok(3));
        assert_eq!(<ArrayType as AttentionUniverse<Array>>::constant_extent(&array, 7), Ok(7));
        assert_eq!(<ArrayType as AttentionUniverse<Array>>::multiply_extents(&array, &2, &3), Ok(6));
        assert!(matches!(
            <ArrayType as AttentionUniverse<Array>>::multiply_extents(&array, &usize::MAX, &2),
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == format!("attention extent product {} × 2 overflows `usize`", usize::MAX),
        ));

        // Broadcasts, reshapes, iotas, and fills take static extents.
        assert_eq!(
            <ArrayType as AttentionUniverse<Array>>::broadcast(&f32_array(&[3], &[1.0, 2.0, 3.0]), &[2, 3], &[1]),
            Ok(f32_array(&[2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0])),
        );
        assert_eq!(
            <ArrayType as AttentionUniverse<Array>>::reshape(&array, &[3, 2]),
            Ok(f32_array(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        );
        assert_eq!(
            <ArrayType as AttentionUniverse<Array>>::iota(&array, 1),
            Ok(Array::from_elements(ArrayType::new_static(DataType::I32, [2, 3]), &[0i32, 1, 2, 0, 1, 2]).unwrap()),
        );
        assert_eq!(<ArrayType as AttentionUniverse<Array>>::fill(&array, 2.5), Ok(f32_array(&[2, 3], &[2.5; 6])));
    }

    #[test]
    fn test_attention_universe_array_ir() {
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let array = f32_array(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let value = ArrayIrValue::Array(array.clone());
        let dimension = |extent| ArrayIrValue::<Array>::Dimension(DimensionValue::constant(extent).unwrap());

        // Array IR values project to their array members, and only their array types describe arrays.
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::into_array(value.clone()),
            Ok(array.clone()),
        );
        assert_eq!(<ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::from_array(array), value);
        assert_eq!(<ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::array_type(&value), Ok(r#type.clone()));
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::to_array_type(&ArrayIrType::Array(r#type.clone())),
            Ok(r#type.clone()),
        );
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::from_array_type(r#type.clone()),
            ArrayIrType::Array(r#type),
        );

        // Extents are first-class dimension values, which include dynamic batching extents.
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::batch_extent(
                &value,
                &BatchingLevelExtent::Static(4),
            ),
            Ok(dimension(4)),
        );
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::batch_extent(
                &value,
                &BatchingLevelExtent::Dynamic(dimension(5)),
            ),
            Ok(dimension(5)),
        );
        assert_eq!(<ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::extent(&value, 1), Ok(dimension(3)));
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::constant_extent(&value, 7),
            Ok(dimension(7)),
        );
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::multiply_extents(
                &value,
                &dimension(2),
                &dimension(3),
            ),
            Ok(dimension(6)),
        );

        // Broadcasts, reshapes, iotas, and fills take their extents from dimension values.
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::broadcast(
                &ArrayIrValue::Array(f32_array(&[3], &[1.0, 2.0, 3.0])),
                &[dimension(2), dimension(3)],
                &[1],
            ),
            Ok(ArrayIrValue::Array(f32_array(&[2, 3], &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]))),
        );
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::reshape(&value, &[dimension(3), dimension(2)]),
            Ok(ArrayIrValue::Array(f32_array(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))),
        );
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::iota(&value, 1),
            Ok(ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::I32, [2, 3]), &[0i32, 1, 2, 0, 1, 2]).unwrap(),
            )),
        );
        assert_eq!(
            <ArrayIrType as AttentionUniverse<ArrayIrValue<Array>>>::fill(&value, 2.5),
            Ok(ArrayIrValue::Array(f32_array(&[2, 3], &[2.5; 6]))),
        );
    }

    #[test]
    fn test_attention_universe_array_ir_dynamic_extents() {
        // Staged array IR values construct iotas whose dimension inputs are the dynamic extents of their targets.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let shape = Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(3)]);
        let (output_type, program) = ArrayIrContext::trace(
            |value: DomainTracer<ArrayIrContext>| {
                <ArrayIrType as AttentionUniverse<DomainTracer<ArrayIrContext>>>::iota(&value, 1)
            },
            ArrayIrType::Array(ArrayType::new(DataType::F32, shape.clone())),
        )
        .unwrap();
        assert_eq!(output_type, ArrayIrType::Array(ArrayType::new(DataType::I32, shape)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[batch, 3] .
                let %1:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %0
                    %2:i32[batch, 3] = iota [type=i32[batch, 3], dimension=1] %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_attention_key() {
        let configuration = AttentionConfiguration::new().with_causal(true).with_dropout(0.25).with_residual(true);
        let signature = AttentionInputSignature::new(true, false, true, false).with_dropout_state(true);
        let key = AttentionKey::new(AttentionDefinitionKind::Forward, configuration, signature);
        assert_eq!(key.kind(), AttentionDefinitionKind::Forward);
        assert_eq!(key.configuration(), configuration);
        assert_eq!(key.signature(), signature);

        // Backward definitions ignore the residual request, so their keys clear it.
        let key = AttentionKey::new(AttentionDefinitionKind::Backward, configuration, signature);
        assert_eq!(key.kind(), AttentionDefinitionKind::Backward);
        assert_eq!(key.configuration(), configuration.with_residual(false));
        assert_eq!(key.signature(), signature);
    }

    #[test]
    fn test_attention_key_from_composite_key() {
        // Keys of every definition kind and implementation round trip, with and without their optional attributes.
        let configuration = AttentionConfiguration::new()
            .with_scale(0.125)
            .with_causal(true)
            .with_local_window((2, 0))
            .with_dropout(0.25)
            .with_residual(true);
        let signature = AttentionInputSignature::new(true, false, true, false).with_dropout_state(true);
        for kind in [AttentionDefinitionKind::Forward, AttentionDefinitionKind::Backward] {
            for implementation in
                [AttentionImplementation::Automatic, AttentionImplementation::Portable, AttentionImplementation::Fused]
            {
                let key = AttentionKey::new(kind, configuration.with_implementation(implementation), signature);
                assert_eq!(AttentionKey::from_composite_key(&key.to_composite_key()), Ok(Some(key)));
                let key = AttentionKey::new(
                    kind,
                    AttentionConfiguration::new().with_implementation(implementation),
                    AttentionInputSignature::default(),
                );
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
        let forward_key = AttentionKey::new(
            AttentionDefinitionKind::Forward,
            AttentionConfiguration::new(),
            AttentionInputSignature::default(),
        )
        .to_composite_key();
        let extended = forward_key.clone().with_attribute("extra", true);
        assert_eq!(
            AttentionKey::from_composite_key(&extended),
            Err(TypeError::invalid(format!("named composite `{extended}` is not a valid attention definition"))),
        );
        let malformed = forward_key.attributes().iter().fold(
            CompositeKey::new(DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION),
            |key, (name, value)| {
                let value = if name == "implementation" { CompositeAttribute::Integer(3) } else { value.clone() };
                key.with_attribute(name.clone(), value)
            },
        );
        assert_eq!(
            AttentionKey::from_composite_key(&malformed),
            Err(TypeError::invalid(format!("named composite `{malformed}` is not a valid attention definition"))),
        );
        let missing =
            CompositeKey::new(DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION)
                .with_attribute("causal", false);
        assert_eq!(
            AttentionKey::from_composite_key(&missing),
            Err(TypeError::invalid(format!("named composite `{missing}` is not a valid attention definition"))),
        );
    }

    #[test]
    fn test_attention_key_to_composite_key() {
        let configuration = AttentionConfiguration::new()
            .with_scale(0.125)
            .with_causal(true)
            .with_local_window((2, 0))
            .with_implementation(AttentionImplementation::Fused)
            .with_dropout(0.25)
            .with_residual(true);
        let signature = AttentionInputSignature::new(true, false, true, false).with_dropout_state(true);

        // Forward keys encode every configuration attribute, including the residual request.
        let key = AttentionKey::new(AttentionDefinitionKind::Forward, configuration, signature).to_composite_key();
        assert_eq!(key.name(), DOT_PRODUCT_ATTENTION_COMPOSITE_NAME);
        assert_eq!(key.version(), DOT_PRODUCT_ATTENTION_COMPOSITE_VERSION);
        assert_eq!(
            key.to_string(),
            "ryft.dot_product_attention.v1 {scale = some(0.125), causal = true, local_window = some([2, 0]), \
             implementation = 2, dropout_rate = some(0.25), residual = true, bias = true, mask = false, \
             query_sequence_lengths = true, key_value_sequence_lengths = false, dropout_state = true}",
        );

        // Backward keys ignore the residual request, so they omit its attribute.
        let key = AttentionKey::new(AttentionDefinitionKind::Backward, configuration, signature).to_composite_key();
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
        assert!(!configuration.return_residual());
        assert_eq!(configuration.dropout(), None);

        let configuration = configuration
            .with_scale(0.25)
            .with_causal(true)
            .with_local_window((2, 1))
            .with_implementation(AttentionImplementation::Fused)
            .with_residual(true)
            .with_dropout(0.1);
        assert_eq!(configuration.scale(), Some(0.25));
        assert!(configuration.causal());
        assert_eq!(configuration.local_window(), Some((2, 1)));
        assert_eq!(configuration.implementation(), AttentionImplementation::Fused);
        assert!(configuration.return_residual());
        assert_eq!(configuration.dropout(), Some(0.1));

        // Absent values restore the inferred scale and disable the local window and dropout, and a symmetric local
        // window uses its radius on both sides.
        assert_eq!(configuration.with_scale(None).scale(), None);
        assert_eq!(configuration.with_local_window(None).local_window(), None);
        assert_eq!(configuration.with_symmetric_local_window(3).local_window(), Some((3, 3)));
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
    fn test_attention_input_signature() {
        let signature = AttentionInputSignature::new(true, false, true, false);
        assert!(signature.has_bias());
        assert!(!signature.has_mask());
        assert!(signature.has_query_sequence_lengths());
        assert!(!signature.has_key_value_sequence_lengths());
        assert!(!signature.has_dropout_state());
        assert_eq!(signature.count(), 2);

        let signature = AttentionInputSignature::new(false, true, false, true).with_dropout_state(true);
        assert!(!signature.has_bias());
        assert!(signature.has_mask());
        assert!(!signature.has_query_sequence_lengths());
        assert!(signature.has_key_value_sequence_lengths());
        assert!(signature.has_dropout_state());
        assert_eq!(signature.count(), 3);
        assert_eq!(AttentionInputSignature::default().count(), 0);
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
    fn test_attention_inputs_from_values() {
        let values = (0..8).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
        assert_eq!(
            AttentionInputs::from_values(AttentionInputSignature::default(), &values[..3]),
            Ok(AttentionInputs::new(values[0].clone(), values[1].clone(), values[2].clone())),
        );

        // The present optional inputs follow the query, key, and value in canonical order.
        let signature = AttentionInputSignature::new(false, true, false, true);
        assert_eq!(
            AttentionInputs::from_values(signature, &values[..5]),
            Ok(AttentionInputs {
                mask: Some(values[3].clone()),
                key_value_sequence_lengths: Some(values[4].clone()),
                ..AttentionInputs::new(values[0].clone(), values[1].clone(), values[2].clone())
            }),
        );
        let signature = AttentionInputSignature::new(true, true, true, true).with_dropout_state(true);
        assert_eq!(
            AttentionInputs::from_values(signature, values.as_slice()),
            Ok(AttentionInputs {
                query: values[0].clone(),
                key: values[1].clone(),
                value: values[2].clone(),
                bias: Some(values[3].clone()),
                mask: Some(values[4].clone()),
                query_sequence_lengths: Some(values[5].clone()),
                key_value_sequence_lengths: Some(values[6].clone()),
                dropout_state: Some(values[7].clone()),
            }),
        );
        assert!(matches!(
            AttentionInputs::from_values(signature, &values[..7]),
            Err(TypeError::Invalid { message, .. })
                if message == "attention input signature expects 8 values but got 7",
        ));
    }

    #[test]
    fn test_attention_inputs_signature() {
        let r#type = ArrayType::new_static(DataType::F32, [1]);
        let inputs = AttentionInputs::new(r#type.clone(), r#type.clone(), r#type.clone());
        assert_eq!(inputs.signature(), AttentionInputSignature::default());
        let inputs = AttentionInputs { mask: Some(r#type.clone()), dropout_state: Some(r#type.clone()), ..inputs };
        assert_eq!(
            inputs.signature(),
            AttentionInputSignature::new(false, true, false, false).with_dropout_state(true),
        );
        let inputs = AttentionInputs {
            bias: Some(r#type.clone()),
            query_sequence_lengths: Some(r#type.clone()),
            key_value_sequence_lengths: Some(r#type),
            ..inputs
        };
        assert_eq!(inputs.signature(), AttentionInputSignature::new(true, true, true, true).with_dropout_state(true));
    }

    #[test]
    fn test_attention_inputs_into_values() {
        let values = (0..8).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
        let inputs = AttentionInputs::new(values[0].clone(), values[1].clone(), values[2].clone());
        assert_eq!(inputs.clone().into_values(), values[..3].to_vec());

        // The present optional inputs follow the query, key, and value in canonical order.
        let inputs = AttentionInputs {
            bias: Some(values[3].clone()),
            mask: Some(values[4].clone()),
            query_sequence_lengths: Some(values[5].clone()),
            key_value_sequence_lengths: Some(values[6].clone()),
            dropout_state: Some(values[7].clone()),
            ..inputs
        };
        assert_eq!(inputs.into_values(), values);
    }

    #[test]
    fn test_attention_backward_inputs_from_values() {
        // Backward inputs append the forward output, its residual, and the output cotangent to the forward inputs.
        let signature = AttentionInputSignature::new(false, true, false, false);
        let values = (0..7).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
        assert_eq!(
            AttentionBackwardInputs::from_values(signature, values.as_slice()),
            Ok(AttentionBackwardInputs {
                inputs: AttentionInputs {
                    mask: Some(values[3].clone()),
                    ..AttentionInputs::new(values[0].clone(), values[1].clone(), values[2].clone())
                },
                output: values[4].clone(),
                residual: values[5].clone(),
                output_cotangent: values[6].clone(),
            }),
        );
        assert!(matches!(
            AttentionBackwardInputs::from_values(signature, &values[..6]),
            Err(TypeError::Invalid { message, .. })
                if message == "attention backward inputs for signature AttentionInputSignature { bias: false, mask: \
                               true, query_sequence_lengths: false, key_value_sequence_lengths: false, dropout_state: \
                               false } have 7 values but got 6",
        ));
    }

    #[test]
    fn test_attention_backward_inputs_into_values() {
        let values = (0..7).map(|size| ArrayType::new_static(DataType::F32, [size])).collect::<Vec<_>>();
        let inputs = AttentionBackwardInputs {
            inputs: AttentionInputs {
                mask: Some(values[3].clone()),
                ..AttentionInputs::new(values[0].clone(), values[1].clone(), values[2].clone())
            },
            output: values[4].clone(),
            residual: values[5].clone(),
            output_cotangent: values[6].clone(),
        };
        assert_eq!(inputs.into_values(), values);
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
                let (first, _) = function.call(inputs.clone())?;
                let (second, _) = function.call(AttentionInputs::new(first, inputs.key, inputs.value))?;
                Ok(second)
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
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_residual(true));

        // The function aligns the manual variation of its inputs before the call, so the invariant inputs become
        // varying and the outputs vary like the query.
        let (output_types, program) = DomainTracingContext::<ArrayContext>::trace_with_named_axes(
            |inputs: AttentionInputs<_>| {
                let (output, residual) = function.call(inputs)?;
                Ok((output, residual.unwrap()))
            },
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
                                %8:f32[] = constant [value=1.0]
                                %9:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %8
                                %10:f32[1, 1, 1, 1]{invariant_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{invariant_rank_4},
                                    output_axes=[0, 1, 2, 3],
                                ] %9
                                %11:f32[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %10
                                %12:f32[1, 1, 1, 1]{varying_rank_4} = mul %7 %11
                                %13:f32[1, 1, 1, 1]{varying_rank_4} = add %12 %3
                                %14:i32[1, 1, 1, 1]{varying_rank_4} = iota [
                                    type=i32[1, 1, 1, 1]{varying_rank_4},
                                    dimension=3,
                                ]
                                %15:i32[1, 1, 1, 1]{varying_rank_4} = iota [
                                    type=i32[1, 1, 1, 1]{varying_rank_4},
                                    dimension=2,
                                ]
                                %16:i32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=i32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0],
                                ] %5
                                %17:bool[1, 1, 1, 1]{varying_rank_4} = compare [direction=LessThan] %15 %16
                                %18:bool[1, 1, 1, 1]{varying_rank_4} = and %4 %17
                                %19:i32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=i32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0],
                                ] %6
                                %20:bool[1, 1, 1, 1]{varying_rank_4} = compare [direction=LessThan] %14 %19
                                %21:bool[1, 1, 1, 1]{varying_rank_4} = and %18 %20
                                %22:f32[] = constant [value=-2.3819763e38]
                                %23:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %22
                                %24:f32[1, 1, 1, 1]{invariant_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{invariant_rank_4},
                                    output_axes=[0, 1, 2, 3],
                                ] %23
                                %25:f32[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %24
                                %26:f32[1, 1, 1, 1]{varying_rank_4} = select %21 %13 %25
                                %27:f32[1, 1, 1]{varying_rank_3} = reduce [kind=max, axes=[3]] %26
                                %28:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0, 1, 2],
                                ] %27
                                %29:f32[1, 1, 1, 1]{varying_rank_4} = sub %26 %28
                                %30:f32[1, 1, 1, 1]{varying_rank_4} = exp %29
                                %31:f32[1, 1, 1]{varying_rank_3} = reduce [kind=sum, axes=[3]] %30
                                %32:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0, 1, 2],
                                ] %31
                                %33:f32[1, 1, 1, 1]{varying_rank_4} = div %30 %32
                                %34:f32[1, 1, 1, 1]{varying_rank_4} = dot [
                                    dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                        rhs_batching=[0, 2]),
                                ] %33 %2
                                %35:f32[1, 1, 1, 1]{varying_rank_4} = transpose [permutation=[0, 2, 1, 3]] %34
                                %36:i32[1, 1, 1, 1]{varying_rank_4} = iota [
                                    type=i32[1, 1, 1, 1]{varying_rank_4},
                                    dimension=1,
                                ]
                                %37:i32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                    output_type=i32[1, 1, 1, 1]{varying_rank_4},
                                    output_axes=[0],
                                ] %5
                                %38:bool[1, 1, 1, 1]{varying_rank_4} = compare [direction=LessThan] %36 %37
                                %39:f32[] = constant [value=0.0]
                                %40:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %39
                                %41:f32[1, 1, 1, 1]{invariant_rank_4} = broadcast [
                                    output_type=f32[1, 1, 1, 1]{invariant_rank_4},
                                    output_axes=[0, 1, 2, 3],
                                ] %40
                                %42:f32[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %41
                                %43:f32[1, 1, 1, 1]{varying_rank_4} = select %38 %35 %42
                                %44:f32[1, 1, 1]{varying_rank_3} = log %31
                                %45:f32[1, 1, 1]{varying_rank_3} = add %27 %44
                                %46:f32[1, 1, 1]{varying_rank_3} = transpose [permutation=[0, 2, 1]] %45
                                %47:f32[1, 1, 1]{varying_rank_3} = stop_gradient %46
                            in (%43, %47)
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
        let signature = AttentionInputSignature::new(true, true, true, true);
        let configuration = AttentionConfiguration::new().with_local_window((0, 0)).with_residual(true);
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
                signature,
                input_types.as_slice(),
            ),
            Ok(vec![query.clone(), ArrayType::new_static(DataType::F32, [2, 2])]),
        );
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                configuration.with_residual(false),
                signature,
                input_types.as_slice(),
            ),
            Ok(vec![query.clone()]),
        );

        // Dropout requires a rate in the open unit interval and a valid random state when a state is present.
        let plain_types = vec![query.clone(), key_value.clone(), key_value];
        let stateful = AttentionInputSignature::default().with_dropout_state(true);
        let mut stateful_types = plain_types.clone();
        stateful_types.push(ArrayType::new_static(DataType::U64, [2]));
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new().with_dropout(0.5),
                stateful,
                stateful_types.as_slice(),
            ),
            Ok(vec![query.clone()]),
        );
        assert_eq!(
            attention_forward_output_types(
                DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
                AttentionConfiguration::new().with_dropout(0.5),
                AttentionInputSignature::default(),
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
                AttentionInputSignature::default(),
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
                AttentionInputSignature::default(),
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
                AttentionInputSignature::new(true, true, true, true),
                input_types.as_slice(),
            ),
            Ok(vec![query.clone(), key_value.clone(), key_value.clone(), bias]),
        );

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
                AttentionInputSignature::new(true, false, false, false),
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
        let configuration = AttentionConfiguration::new().with_local_window((1, 0)).with_residual(true);
        let (output, residual) = attend(inputs.clone(), configuration);
        assert_eq!(output, f32_array(&[2, 1, 2], &[1.0, 2.0, 0.0, 0.0]));
        let invisible = (-0.7 * f64::from(f32::MAX)) as f32;
        assert_eq!(residual, Some(f32_array(&[2, 1], &[std::f32::consts::FRAC_1_SQRT_2, invisible])));
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
        let configuration = AttentionConfiguration::new().with_residual(true);
        let (output, residual) = attend(AttentionInputs::new(query.clone(), key.clone(), value.clone()), configuration);
        let function = dot_product_attention::<ArrayIrContext>(configuration);
        assert_eq!(
            function.call(AttentionInputs::new(
                ArrayIrValue::Array(query),
                ArrayIrValue::Array(key),
                ArrayIrValue::Array(value),
            )),
            Ok((ArrayIrValue::Array(output), residual.map(ArrayIrValue::Array))),
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
        let outputs = (0..8).map(|seed| function.call(with_state(seed)).unwrap().0.to_f64s()).collect::<Vec<_>>();
        assert!(outputs.iter().any(|output| output != &outputs[0]));
        for subset_sum in outputs.iter().flatten().map(|element| element / 0.5) {
            assert!((0.0..=10.0).contains(&subset_sum));
            assert_eq!(subset_sum.fract(), 0.0);
        }
    }

    #[test]
    fn test_dot_product_attention_batching() {
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new().with_residual(true));

        // Each batch item carries one complete rank-4 attention problem. The batching rule folds the mapped axis into
        // the logical batch axis and restores it on the outputs.
        let ones = f32_array(&[2, 1, 1, 1, 1], &[1.0, 1.0]);
        let values = f32_array(&[2, 1, 1, 1, 1], &[5.0, 7.0]);
        let (output, residual) = batch(
            |inputs: AttentionInputs<_>| {
                let (output, residual) = function.call(inputs)?;
                Ok((output, residual.unwrap()))
            },
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
            |inputs: AttentionInputs<_>| {
                let (output, residual) = function.call(inputs)?;
                Ok((output, residual.unwrap()))
            },
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
                    |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
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
    fn test_dot_product_attention_batching_static_extents() {
        // Homogeneous array families support static extents only, so batching over dynamic extents fails.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let length = DimensionVariable::new("length", DimensionBounds::new(1, Some(3)).unwrap());
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), length.into(), 1.into(), 1.into()]));
        let result = ArrayContext::trace(
            |inputs: AttentionInputs<_>| {
                batch(
                    |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
                    inputs,
                    AttentionInputs::new(BatchAxis::new(0), BatchAxis::new(0), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )
                .map_err(ProgramError::from)
            },
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
        );
        assert!(matches!(
            result,
            Err(ProgramError::Type(TypeError::Invalid { message, .. }))
                if message == "attention over homogeneous arrays requires static extents, but axis 0 of \
                               `f32[length, 1, 1]` has extent `length`; use array IR values for dynamic extents",
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
                batch(|inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0), inputs, axes, BatchAxis::new(0), None)
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
        let varying_rank_5 = rendered_sharding(5, true);
        let varying_rank_4 = rendered_sharding(4, true);
        let varying_rank_3 = rendered_sharding(3, true);
        let invariant_rank_4 = rendered_sharding(4, false);
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
                        let %4:f32[2, 1, 1, 1, 2]{varying_rank_5} = broadcast [
                            output_type=f32[2, 1, 1, 1, 2]{varying_rank_5},
                            output_axes=[0, 3, 4],
                        ] %3
                            %5:f32[2, 1, 1, 2]{varying_rank_4} = reshape [shape=[2, 1, 1, 2]] %4
                            %6:f32[2, 1, 1, 1]{varying_rank_4} = composite [
                                key=\"ryft.dot_product_attention.v1 {{scale = none, causal = false, \
                                    local_window = none, implementation = 0, dropout_rate = none, residual = false, \
                                    bias = true, mask = false, query_sequence_lengths = false, \
                                    key_value_sequence_lengths = false, dropout_state = false}}\",
                            ] %0 %1 %2 %5 [
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
                                        %5:f32[] = constant [value=1.0]
                                        %6:f32[2, 1, 1, 2] = broadcast [output_type=f32[2, 1, 1, 2], output_axes=[]] %5
                                        %7:f32[2, 1, 1, 2]{invariant_rank_4} = broadcast [
                                            output_type=f32[2, 1, 1, 2]{invariant_rank_4},
                                            output_axes=[0, 1, 2, 3],
                                        ] %6
                                        %8:f32[2, 1, 1, 2]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %7
                                        %9:f32[2, 1, 1, 2]{varying_rank_4} = mul %4 %8
                                        %10:f32[2, 1, 1, 2]{varying_rank_4} = add %9 %3
                                        %11:f32[2, 1, 1]{varying_rank_3} = reduce [kind=max, axes=[3]] %10
                                        %12:f32[2, 1, 1, 2]{varying_rank_4} = broadcast [
                                            output_type=f32[2, 1, 1, 2]{varying_rank_4},
                                            output_axes=[0, 1, 2],
                                        ] %11
                                        %13:f32[2, 1, 1, 2]{varying_rank_4} = sub %10 %12
                                        %14:f32[2, 1, 1, 2]{varying_rank_4} = exp %13
                                        %15:f32[2, 1, 1]{varying_rank_3} = reduce [kind=sum, axes=[3]] %14
                                        %16:f32[2, 1, 1, 2]{varying_rank_4} = broadcast [
                                            output_type=f32[2, 1, 1, 2]{varying_rank_4},
                                            output_axes=[0, 1, 2],
                                        ] %15
                                        %17:f32[2, 1, 1, 2]{varying_rank_4} = div %14 %16
                                        %18:f32[2, 1, 1, 1]{varying_rank_4} = dot [
                                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                                rhs_batching=[0, 2]),
                                        ] %17 %2
                                        %19:f32[2, 1, 1, 1]{varying_rank_4} = transpose [permutation=[0, 2, 1, 3]] %18
                                    in (%19)
                                }},
                            ]
                        in (%6)
                    }},
                ]
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_dot_product_attention_batching_dynamic_extent() {
        // Array IR families fold dynamic mapped extents with first-class dimension values.
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
            |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
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
                            let %4:dimension<1> = constant [value=1]
                                %5:dimension<1> = constant [value=1]
                                %6:dimension<1> = constant [value=1]
                                %7:dimension<1> = constant [value=1]
                                %8:dimension<1> = constant [value=1]
                                %9:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %1 %0 %6 %7 %8
                                %10:dimension<1> = constant [value=1]
                                %11:dimension<1> = constant [value=1]
                                %12:dimension<1> = constant [value=1]
                                %13:dimension<1> = constant [value=1]
                                %14:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %2 %0 %11 %12 %13
                                %15:dimension<1> = constant [value=1]
                                %16:dimension<1> = constant [value=1]
                                %17:dimension<1> = constant [value=1]
                                %18:dimension<1> = constant [value=1]
                                %19:f32[batch, 1, 1, 1] = reshape [requires_runtime_assertion=false] %3 %0 %16 %17 %18
                                %20:f32[batch, 1, 1, 1] = composite [
                                    key=\"ryft.dot_product_attention.v1 {scale = none, causal = false, \
                                        local_window = none, implementation = 0, dropout_rate = none, \
                                        residual = false, bias = false, mask = false, query_sequence_lengths = false, \
                                        key_value_sequence_lengths = false, dropout_state = false}\",
                                ] %9 %14 %19 [
                                    primal={
                                        lambda %0:f32[batch, 1, 1, 1], %1:f32[batch, 1, 1, 1], %2:f32[batch, 1, 1, 1] .
                                        let %3:f32[batch, 1, 1, 1] = dot [
                                            dimensions=(lhs_contracting=[3], rhs_contracting=[3], lhs_batching=[0, 2], \
                                                rhs_batching=[0, 2]),
                                        ] %0 %1
                                            %4:f32[] = constant [value=1.0]
                                            %5:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %3
                                            %6:dimension<1> = constant [value=1]
                                            %7:dimension<1> = constant [value=1]
                                            %8:dimension<1> = constant [value=1]
                                            %9:f32[batch, 1, 1, 1] = broadcast [output_axes=[]] %4 %5 %6 %7 %8
                                            %10:f32[batch, 1, 1, 1] = mul %3 %9
                                            %11:f32[batch, 1, 1] = reduce [kind=max, axes=[3]] %10
                                            %12:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %10
                                            %13:dimension<1> = constant [value=1]
                                            %14:dimension<1> = constant [value=1]
                                            %15:dimension<1> = constant [value=1]
                                            %16:f32[batch, 1, 1, 1] = \
                                                broadcast [output_axes=[0, 1, 2]] %11 %12 %13 %14 %15
                                            %17:f32[batch, 1, 1, 1] = sub %10 %16
                                            %18:f32[batch, 1, 1, 1] = exp %17
                                            %19:f32[batch, 1, 1] = reduce [kind=sum, axes=[3]] %18
                                            %20:dimension<batch ∈ [1, 5)> = dimension_size [axis=0] %18
                                            %21:dimension<1> = constant [value=1]
                                            %22:dimension<1> = constant [value=1]
                                            %23:dimension<1> = constant [value=1]
                                            %24:f32[batch, 1, 1, 1] = \
                                                broadcast [output_axes=[0, 1, 2]] %19 %20 %21 %22 %23
                                            %25:f32[batch, 1, 1, 1] = div %18 %24
                                            %26:f32[batch, 1, 1, 1] = dot [
                                                dimensions=(lhs_contracting=[3], rhs_contracting=[1], \
                                                    lhs_batching=[0, 1], rhs_batching=[0, 2]),
                                            ] %25 %2
                                            %27:f32[batch, 1, 1, 1] = transpose [permutation=[0, 2, 1, 3]] %26
                                        in (%27)
                                    },
                                ]
                                %21:dimension<1> = constant [value=1]
                                %22:dimension<1> = constant [value=1]
                                %23:dimension<1> = constant [value=1]
                                %24:dimension<1> = constant [value=1]
                                %25:f32[batch, 1, 1, 1, 1] = \
                                    reshape [requires_runtime_assertion=false] %20 %0 %21 %22 %23 %24
                            in (%25)
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
        let (expected, _) = function
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
            batch(|inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0), inputs, axes, BatchAxis::new(0), None)
                .unwrap();
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
            |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
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
            |inputs: AttentionInputs<_>| Ok(function.call(inputs)?.0),
            AttentionInputs::new(r#type.clone(), r#type.clone(), r#type),
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
            vec![ones.clone(), ones, values.clone()],
            vec![BatchAxis::new(0); 3],
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output, values);
    }

    #[test]
    fn test_dot_product_attention_differentiation() {
        // The named definitions have no derivative rules, so the inference function rejects differentiation.
        let function = dot_product_attention::<ArrayContext>(AttentionConfiguration::new());
        let ones = f32_array(&[1, 1, 1, 1], &[1.0]);
        assert!(matches!(
            differentiate_at(AttentionInputs::new(ones.clone(), ones.clone(), ones))
                .gradient(|inputs| function.call(inputs)?.0.reduce(&[0, 1, 2, 3], ReductionKind::Sum)),
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
        // The training function selects the portable implementation for automatic requests and keeps explicit ones,
        // which the keys of its staged training composite and of the forward definition record.
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
                        key=\"ryft.dot_product_attention_training.v1 {{scale = none, causal = false, \
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
                        key=\"ryft.dot_product_attention_training.v1 {{scale = none, causal = false, \
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
                                        %4:f32[] = constant [value=1.0]
                                        %5:f32[1, 1, 1, 1] = broadcast [output_type=f32[1, 1, 1, 1], output_axes=[]] %4
                                        %6:f32[1, 1, 1, 1]{invariant_rank_4} = broadcast [
                                            output_type=f32[1, 1, 1, 1]{invariant_rank_4},
                                            output_axes=[0, 1, 2, 3],
                                        ] %5
                                        %7:f32[1, 1, 1, 1]{varying_rank_4} = parallel_vary [axis_name=\"x\"] %6
                                        %8:f32[1, 1, 1, 1]{varying_rank_4} = mul %3 %7
                                        %9:f32[1, 1, 1]{varying_rank_3} = reduce [kind=max, axes=[3]] %8
                                        %10:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                            output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                            output_axes=[0, 1, 2],
                                        ] %9
                                        %11:f32[1, 1, 1, 1]{varying_rank_4} = sub %8 %10
                                        %12:f32[1, 1, 1, 1]{varying_rank_4} = exp %11
                                        %13:f32[1, 1, 1]{varying_rank_3} = reduce [kind=sum, axes=[3]] %12
                                        %14:f32[1, 1, 1, 1]{varying_rank_4} = broadcast [
                                            output_type=f32[1, 1, 1, 1]{varying_rank_4},
                                            output_axes=[0, 1, 2],
                                        ] %13
                                        %15:f32[1, 1, 1, 1]{varying_rank_4} = div %12 %14
                                        %16:f32[1, 1, 1, 1]{varying_rank_4} = dot [
                                            dimensions=(lhs_contracting=[3], rhs_contracting=[1], lhs_batching=[0, 1], \
                                                rhs_batching=[0, 2]),
                                        ] %15 %2
                                        %17:f32[1, 1, 1, 1]{varying_rank_4} = transpose [permutation=[0, 2, 1, 3]] %16
                                    in (%17)
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
    }

    #[test]
    fn test_differentiable_dot_product_attention_dropout() {
        // The backward definition applies the dropout mask of the forward definition, so the analytical gradient
        // agrees with the gradient of the portable forward body, whose dropout is differentiated automatically, up to
        // the rounding of their `f32` softmax computations.
        let array = |dimensions: &[usize], elements: &[f64]| {
            Array::from_elements(ArrayType::new_static(DataType::F64, dimensions.to_vec()), elements).unwrap()
        };
        let inputs = AttentionInputs {
            dropout_state: Some(Array::from_elements(ArrayType::new_static(DataType::U64, [2]), &[0u64, 3]).unwrap()),
            ..AttentionInputs::new(
                array(&[1, 2, 1, 2], &[0.1, -0.2, 0.3, 0.4]),
                array(&[1, 3, 1, 2], &[0.5, 0.2, -0.3, 0.7, 0.1, -0.6]),
                array(&[1, 3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            )
        };
        let configuration = AttentionConfiguration::new().with_dropout(0.5);
        let function = differentiable_dot_product_attention::<ArrayContext>(configuration);
        let (output, gradients) = differentiate_at(inputs.clone())
            .value_and_gradient(|inputs| function.call(inputs)?.reduce(&[0, 1, 2, 3], ReductionKind::Sum))
            .unwrap();
        let (expected_output, expected_gradients) = differentiate_at(inputs)
            .value_and_gradient(|inputs| {
                let (output, _) = dot_product_attention_forward_body(&inputs, configuration)?;
                output.reduce(&[0, 1, 2, 3], ReductionKind::Sum)
            })
            .unwrap();
        assert_abs_diff_eq!(output.to_f64s()[0], expected_output.to_f64s()[0], epsilon = 1e-6);
        for (actual, expected) in [
            (gradients.query, expected_gradients.query),
            (gradients.key, expected_gradients.key),
            (gradients.value, expected_gradients.value),
        ] {
            actual
                .to_f64s()
                .iter()
                .zip(expected.to_f64s())
                .for_each(|(actual, expected)| assert_abs_diff_eq!(*actual, expected, epsilon = 1e-6));
        }
        assert_eq!(gradients.dropout_state.unwrap().r#type().data_type(), DataType::Zero);
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
    fn test_dot_product_attention_forward_body_differentiation() {
        let r#type = ArrayType::new_static(DataType::F64, [1, 1, 1, 1]);
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) = ArrayIrContext::trace(
            |inputs: AttentionInputs<_>| {
                let (output, residual) = dot_product_attention_forward_body(
                    &inputs,
                    AttentionConfiguration::new().with_scale(0.5).with_residual(true),
                )?;
                Ok((output.into_value(), residual.unwrap().into_value()))
            },
            AttentionInputs {
                bias: Some(ArrayIrType::Array(scalar_type.clone())),
                ..AttentionInputs::new(
                    ArrayIrType::Array(r#type.clone()),
                    ArrayIrType::Array(r#type.clone()),
                    ArrayIrType::Array(r#type.clone()),
                )
            },
        )
        .unwrap();
        let array = |r#type: &ArrayType, element: f64| {
            ArrayIrValue::Array(Array::from_elements(r#type.clone(), &[element]).unwrap())
        };

        let outputs = program
            .into_flat_program()
            .jvp()
            .unwrap()
            .interpret(vec![
                array(&r#type, 2.0),
                array(&r#type, 3.0),
                array(&r#type, 5.0),
                array(&scalar_type, 1.0),
                array(&r#type, 7.0),
                array(&r#type, 11.0),
                array(&r#type, 13.0),
                array(&scalar_type, 17.0),
            ])
            .unwrap();
        let [
            ArrayIrValue::Array(output),
            ArrayIrValue::Array(residual),
            ArrayIrValue::Array(tangent),
            ArrayIrValue::Array(residual_tangent),
        ] = outputs.as_slice()
        else {
            panic!("the attention JVP produces array primals and tangents")
        };

        // The softmax over one visible key is identically one, so the output is the value and the residual is the
        // only logit. The query, key, and bias tangents therefore vanish, the value tangent passes through unchanged,
        // and the residual has a zero tangent because it stops gradients.
        assert_eq!(output.to_f64s(), vec![5.0]);
        assert_eq!(residual.to_f64s(), vec![4.0]);
        assert_eq!(tangent.to_f64s(), vec![13.0]);
        assert_eq!(residual_tangent.to_f64s(), vec![0.0]);
    }
}
