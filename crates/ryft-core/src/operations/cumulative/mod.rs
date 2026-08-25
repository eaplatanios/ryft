//! Cumulative reduction (prefix scan) operation family.
//!
//! A cumulative reduction combines every prefix of one array axis with an associative binary operator, producing a
//! result of exactly the input's type: element `i` along the scanned axis holds the combination of elements
//! `0..=i` (or `i..` when the scan runs in reverse). The family mirrors JAX's cumulative operations (refer to
//! [`jax.lax.cumsum`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cumsum.html)), whose members share one
//! payload shape and differ only in what they combine.
//!
//! # Family design
//!
//! Every member carries the same payload — the scanned `axis` and a `reverse` flag — and shares the type-inference
//! contract implemented by [`cumulative_abstract`], the batching axis shift implemented by
//! [`lift_cumulative_axis`], and the eager sequential prefix scan implemented by [`cumulative_evaluate`]. Members
//! supply only their own combine function, identity element, accepted element data types, and differentiation rule.
//! For the nonlinear members, even that much is spelled once per member as an invocation of
//! `define_cumulative_operation!`, which generates everything else they share.
//!
//! The identity matters beyond the empty-prefix case: it is what a batching rule writes over the padding of a
//! bounded ragged scanned axis (refer to [`RaggedMaskIdentity`](crate::arrays::RaggedMaskIdentity)), so that padded
//! positions cannot contribute to any live prefix. Masking is where the family's batching discipline parts company
//! with a reduction's: a prefix scan keeps every axis it touches, so it *consumes* no bounded ragged axis. The
//! operand's ragged axes ride through onto the result unchanged and the rule reports no consumption evidence. A scan
//! whose axis is not ragged never reaches the masking hook at all, and so passes through exactly as it would with no
//! ragged metadata.
//!
//! # Differentiation
//!
//! [`CumulativeSum`] is the family's one *linear* member, so it differentiates and transposes as itself. The
//! nonlinear members ([`CumulativeProduct`], [`CumulativeMax`], [`CumulativeMin`], and [`CumulativeLogSumExp`])
//! have no closed-form primitive derivative and none is invented for them: their forward-mode rules differentiate
//! through the parallel-prefix [`associative_scan`] decomposition, so each derivative is assembled from the rules
//! of the primitives that construction stages. None of them is transposable, and reverse mode reaches them by
//! transposing those staged primitives instead. This is exactly how JAX defines the same operations
//! (`_cumulative_jvp_rule`).

// TODO(eaplatanios): Review this module.

use crate::arrays::{ArrayType, Dimension, ShardingDimension, StaticShape};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiationDriver, DifferentiationDual, DifferentiationError};
use crate::macros::check_count;
use crate::operations::constants::zero::{Zero, ZeroOperationProvider};
use crate::operations::manipulation::concatenation::{Concatenate, ConcatenateOperation};
use crate::operations::manipulation::padding::{Pad, PadOperation, dependency_scalar_type};
use crate::operations::manipulation::slicing::{Slice, SliceOperation};
use crate::operations::math::add::{Add, AddOperation};
use crate::programs::{MaybeZero, ProgramError, ProvenanceScope, TypeError, Typed, Value};
use crate::tracing::{Tracer, TracingContext};

pub mod cumulative_log_sum_exp;
pub mod cumulative_max;
pub mod cumulative_min;
pub mod cumulative_product;
pub mod cumulative_sum;

pub use cumulative_log_sum_exp::{
    CUMULATIVE_LOG_SUM_EXP_OPERATION_NAME, CumulativeLogSumExp, CumulativeLogSumExpOperation,
};
pub use cumulative_max::{CUMULATIVE_MAX_OPERATION_NAME, CumulativeMax, CumulativeMaxOperation};
pub use cumulative_min::{CUMULATIVE_MIN_OPERATION_NAME, CumulativeMin, CumulativeMinOperation};
pub use cumulative_product::{CUMULATIVE_PRODUCT_OPERATION_NAME, CumulativeProduct, CumulativeProductOperation};
pub use cumulative_sum::{CUMULATIVE_SUM_OPERATION_NAME, CumulativeSum, CumulativeSumOperation};

/// Returns the output [`ArrayType`] produced by scanning `input` along `axis`, validating the scan geometry shared
/// by every member of the cumulative family. The result *is* the input type: a prefix scan changes neither the
/// element data type nor the shape, layout, memory placement, or sharding.
///
/// Validates that:
///   - `axis` is within `0..rank(input)`;
///   - the scanned dimension is [`Dimension::Static`], because a prefix scan is defined by the exact number of
///     elements it accumulates over; and
///   - the scanned dimension is unsharded, mirroring JAX's cumulative sharding rule, because a prefix crosses shard
///     boundaries and this operation carries no cross-shard communication of its own.
///
/// Element data types are validated by each member rather than here, because they differ across the family (e.g.,
/// summation accepts complex inputs while extrema do not).
///
/// A dynamically sized scanned axis could be supported in the future by physicalizing the scan at the dimension's
/// declared upper bound and masking the elements past each runtime extent with the member's identity, which is the
/// same discipline the ragged batching rules already use. That extension is deliberately not implemented here: it
/// would silently change the operation's cost model, so it belongs to an explicit dynamic-scan surface.
///
/// # Parameters
///
///   - `input`: Type of the scanned operand.
///   - `axis`: Scanned axis, in the operand's own coordinate system.
///   - `operation_name`: Operation name used in diagnostics.
pub fn cumulative_abstract(
    input: &ArrayType,
    axis: usize,
    operation_name: &'static str,
) -> Result<ArrayType, TypeError> {
    let rank = input.rank();
    if axis >= rank {
        return Err(TypeError::invalid(format!("`{operation_name}` axis {axis} is out of bounds for rank {rank}")));
    }
    if !matches!(input.dimension(axis), Dimension::Static(_)) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` requires a static scanned dimension but axis {axis} of {input} is dynamic"
        )));
    }
    if let Some(sharding) = input.sharding()
        && matches!(sharding.dimensions()[axis], ShardingDimension::Sharded(_))
    {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` requires an unsharded scanned dimension but axis {axis} of {input} is sharded"
        )));
    }
    Ok(input.clone())
}

/// Lifts a cumulative reduction's `axis` through one batching level inserted at `batch_axis`.
///
/// Returns the rewritten scanned axis and the output batch axis position. The scanned axis is expressed in the
/// per-item coordinate system and therefore cannot name the inserted batch dimension, so a user axis at or above
/// `batch_axis` shifts up by one. A prefix scan preserves its operand's rank, and so the output batch axis is
/// `batch_axis` itself.
pub fn lift_cumulative_axis(axis: usize, batch_axis: usize) -> (usize, usize) {
    (if axis < batch_axis { axis } else { axis + 1 }, batch_axis)
}

/// Prefix-scan evaluation helper that operates on a flat row-major payload and shape.
///
/// Returns the scanned payload, which has the same length and shape as `values`. Output element `i` along `axis`
/// holds the accumulation of input elements `0..=i`, or of elements `i..` when `reverse` is set. A scanned axis
/// shorter than two elements (including a zero-length one) leaves the payload unchanged. The combiner is fallible
/// because the element-level arithmetic contracts of the reference backend are (e.g., a conversion into a
/// low-precision encoding can fail).
///
/// # Parameters
///
///   - `values`: Row-major input payload.
///   - `shape`: Input shape.
///   - `axis`: Scanned axis.
///   - `reverse`: Whether to accumulate from the end of the scanned axis toward its start.
///   - `combiner`: Binary associative operator, receiving the accumulated prefix and the next element.
pub fn cumulative_evaluate<T: Clone>(
    values: &[T],
    shape: &StaticShape,
    axis: usize,
    reverse: bool,
    combiner: impl Fn(T, T) -> Result<T, ProgramError>,
) -> Result<Vec<T>, ProgramError> {
    let mut output = values.to_vec();
    let extent = shape[axis];
    if extent < 2 {
        return Ok(output);
    }

    // Row-major storage splits into `outer` independent blocks of `extent` slices, each holding `inner` elements, so
    // one scan step moves by `inner` elements and the scan visits every `(outer, inner)` pair once. Both bounds are
    // computed as direct dimension products rather than from the payload length and the scanned axis stride, because
    // a zero-extent axis anywhere to the right of `axis` makes that stride zero.
    let inner = shape.dimensions()[axis + 1..].iter().product::<usize>();
    let outer = shape.dimensions()[..axis].iter().product::<usize>();
    for block in 0..outer {
        let base = block * extent * inner;
        for offset in 0..inner {
            let index = |position: usize| base + position * inner + offset;
            if reverse {
                for position in (0..extent - 1).rev() {
                    output[index(position)] =
                        combiner(output[index(position + 1)].clone(), values[index(position)].clone())?;
                }
            } else {
                for position in 1..extent {
                    output[index(position)] =
                        combiner(output[index(position - 1)].clone(), values[index(position)].clone())?;
                }
            }
        }
    }
    Ok(output)
}

/// Returns the inclusive prefix scan of `value` along `axis` under the associative operator `combine`, built out of
/// ordinary manipulation primitives instead of out of one cumulative primitive.
///
/// This is Ryft's port of the log-depth Blelloch construction that JAX's
/// [`lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html) implements
/// (`jax/_src/lax/control_flow/loops.py`), and it exists here for the reason it is reached for there: a cumulative
/// reduction whose combining operator is nonlinear has no closed-form primitive derivative, so the family's
/// nonlinear members define their forward mode by differentiating *through* this decomposition rather than by
/// carrying a bespoke gradient formula (JAX's `_cumulative_jvp_rule`).
///
/// The recursion combines adjacent pairs along `axis`, scans the halved sequence recursively, combines the scanned
/// halves back against the elements the pairing skipped, and interleaves the two halves into the result. `combine`
/// always receives its operands in scan order (the accumulated prefix first), so the construction stays correct for
/// associative operators that are not commutative. A `reverse` scan mirrors the same recursion around the end of the
/// axis — the pairing simply starts one element in when the extent is odd — which is what lets it work without an
/// array-reversal primitive, of which Ryft has none.
///
/// The whole operand shape must be static, because the construction slices at staging-time positions. A scanned axis
/// shorter than two elements is returned unchanged.
///
/// # Parameters
///
///   - `value`: Scanned operand.
///   - `axis`: Scanned axis.
///   - `reverse`: Whether to accumulate from the end of the scanned axis toward its start.
///   - `combine`: Associative binary operator, receiving the accumulated prefix and the next elements in scan order.
pub fn associative_scan<V, F>(value: &V, axis: usize, reverse: bool, combine: &F) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Add + Concatenate + Pad + Slice,
    V::DispatchDomain: Context<Type = ArrayType> + Zero<V>,
    F: Fn(&V, &V) -> Result<V, ProgramError>,
{
    let value_type = value.r#type().into_owned();
    let rank = value_type.rank();
    if axis >= rank {
        return Err(
            TypeError::invalid(format!("`associative_scan` axis {axis} is out of bounds for rank {rank}")).into()
        );
    }
    let shape = value_type.static_shape().ok_or_else(|| {
        TypeError::invalid(format!("`associative_scan` requires a statically shaped operand but got {value_type}"))
    })?;

    // The scopes below are purely diagnostic: they attribute every instruction the decomposition stages, and they are
    // a no-op under an eager context, which records no instructions at all.
    let domain = value.dispatch_domain();
    domain.invoke_with_provenance_scope(ProvenanceScope::new("ryft"), || {
        domain.invoke_with_provenance_scope(ProvenanceScope::new("differentiation"), || {
            domain.invoke_with_provenance_scope(ProvenanceScope::new("associative_scan"), || {
                associative_scan_recursively(value, &shape, axis, reverse, combine)
            })
        })
    })
}

/// Recursive half of [`associative_scan`], operating on an operand whose shape is already known to be static.
fn associative_scan_recursively<V, F>(
    value: &V,
    shape: &StaticShape,
    axis: usize,
    reverse: bool,
    combine: &F,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Add + Concatenate + Pad + Slice,
    V::DispatchDomain: Context<Type = ArrayType> + Zero<V>,
    F: Fn(&V, &V) -> Result<V, ProgramError>,
{
    let extent = shape[axis];
    if extent < 2 {
        return Ok(value.clone());
    }
    let half = extent / 2;

    // Pair adjacent elements. A forward scan pairs from the start of the axis and a reverse scan pairs from its end,
    // which is the one place the two directions differ: an odd extent leaves the first element unpaired going
    // forward and the *last* one unpaired going backward, so the pairing starts one element in.
    let pair_offset = match reverse {
        true => extent % 2,
        false => 0,
    };
    let earlier = scan_slice(value, shape, axis, pair_offset, extent - 1, 2)?;
    let later = scan_slice(value, shape, axis, pair_offset + 1, extent, 2)?;
    let reduced = match reverse {
        true => combine(&later, &earlier)?,
        false => combine(&earlier, &later)?,
    };

    // Scanning the pairwise reductions yields every other output element: the odd positions of a forward scan, and
    // the positions congruent to `pair_offset` of a reverse one.
    let halved_shape = StaticShape::new({
        let mut dimensions = shape.dimensions().to_vec();
        dimensions[axis] = half;
        dimensions
    });
    let aligned = associative_scan_recursively(&reduced, &halved_shape, axis, reverse, combine)?;

    // Each complementary position extends the aligned result before it by the one element that separates them,
    // except for the position at the scan's own start, which is just the operand element there. An even extent has
    // one fewer complementary combination than there are aligned results, so the aligned side is trimmed; an extent
    // of exactly two has none at all, and its complementary half is that lone start element.
    let complement_count = match extent % 2 {
        0 => half - 1,
        _ => half,
    };
    let (complement, aligned_leads) = match reverse {
        true => {
            let last = scan_slice(value, shape, axis, extent - 1, extent, 1)?;
            let complement = match complement_count {
                0 => last,
                _ => {
                    let trimmed = match extent % 2 {
                        0 => scan_slice(&aligned, &halved_shape, axis, 1, half, 1)?,
                        _ => aligned.clone(),
                    };
                    let start = (pair_offset + 1) % 2;
                    let operands = scan_slice(value, shape, axis, start, start + 2 * complement_count, 2)?;
                    V::concatenate([&combine(&trimmed, &operands)?, &last], axis)?
                }
            };
            (complement, extent % 2 == 0)
        }
        false => {
            let first = scan_slice(value, shape, axis, 0, 1, 1)?;
            let complement = match complement_count {
                0 => first,
                _ => {
                    let trimmed = match extent % 2 {
                        0 => scan_slice(&aligned, &halved_shape, axis, 0, half - 1, 1)?,
                        _ => aligned.clone(),
                    };
                    let operands = scan_slice(value, shape, axis, 2, (2 + 2 * complement_count).min(extent), 2)?;
                    V::concatenate([&first, &combine(&trimmed, &operands)?], axis)?
                }
            };
            (complement, false)
        }
    };

    match aligned_leads {
        true => scan_interleave(&aligned, &complement, shape, axis, half, extent - half),
        false => scan_interleave(&complement, &aligned, shape, axis, extent - half, half),
    }
}

/// Value that the nested trace staging a member's associative-scan decomposition flows.
type DecompositionTracer<C> = Tracer<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>;

/// Applies the forward-mode rule of a nonlinear cumulative member by differentiating through the associative-scan
/// decomposition of [`associative_scan`], and returns the resulting primal/tangent pair.
///
/// The decomposition is traced once into its own program over the caller's operation family, that program is
/// differentiated through the instruction-scoped `driver` — which re-enters the active differentiation machinery, so
/// every primitive the construction stages contributes its *own* forward-mode rule — and the resulting fused program
/// is replayed in `context` over the operand's primal and tangent. This mirrors JAX's `_cumulative_jvp_rule`, which
/// is literally `api.jvp(partial(associative_scan, combine_fn, ...), primals, tangents)`: the primal output comes
/// back from the decomposition too, rather than from the cumulative primitive, because the two are the same value
/// and the fused program computes it on the way to the tangent.
///
/// The caller is responsible for the structural-zero tangent shortcut; this helper requires a live tangent because
/// the decomposition is pure overhead when there is nothing to propagate.
///
/// # Parameters
///
///   - `context`: [`Context`] the fused forward-mode program is replayed in.
///   - `driver`: Instruction-scoped [`DifferentiationDriver`] serving the nested differentiation request.
///   - `primal`: Operand primal.
///   - `tangent`: Operand tangent.
///   - `axis`: Scanned axis.
///   - `reverse`: Whether the scan accumulates from the end of the scanned axis toward its start.
///   - `combine`: Member's associative operator, staged over the nested trace's values.
pub(crate) fn jvp_through_associative_scan<C, D, F>(
    context: &C,
    driver: &D,
    primal: &C::Value,
    tangent: &C::Value,
    axis: usize,
    reverse: bool,
    combine: F,
) -> Result<DifferentiationDual<C::Value>, DifferentiationError>
where
    C: Context<Type = ArrayType>,
    D: DifferentiationDriver<C>,
    C::Operation: From<AddOperation<ArrayType>>
        + From<ConcatenateOperation<ArrayType>>
        + From<PadOperation<ArrayType>>
        + From<SliceOperation>
        + ZeroOperationProvider<ArrayType>,
    F: Fn(&DecompositionTracer<C>, &DecompositionTracer<C>) -> Result<DecompositionTracer<C>, ProgramError>,
{
    let (_, decomposition) = TracingContext::<C::Constant, C::Operation>::trace::<_, ArrayType, _>(
        |value: DecompositionTracer<C>| associative_scan(&value, axis, reverse, &combine),
        primal.r#type().into_owned(),
    )?;
    let fused = driver.jvp_program(decomposition.entry_region_ref())?;
    let mut outputs = fused.interpret_in_context(context, vec![primal.clone(), tangent.clone()])?;
    check_count!("output", outputs, 2, ProgramError);
    let output_tangent = outputs.remove(1);
    Ok(DifferentiationDual::new(outputs.remove(0), MaybeZero::Value(output_tangent))?)
}

/// Returns the elements of `value` at positions `start`, `start + stride`, ... below `limit` along `axis`, keeping
/// every other axis whole.
fn scan_slice<V: Slice>(
    value: &V,
    shape: &StaticShape,
    axis: usize,
    start: usize,
    limit: usize,
    stride: usize,
) -> Result<V, ProgramError> {
    let mut start_indices = vec![0; shape.rank()];
    let mut limit_indices = shape.dimensions().to_vec();
    let mut strides = vec![1; shape.rank()];
    start_indices[axis] = start;
    limit_indices[axis] = limit;
    strides[axis] = stride;
    value.slice(start_indices.as_slice(), limit_indices.as_slice(), strides.as_slice())
}

/// Returns `left` and `right` interleaved along `axis`, starting with `left`. `left` must hold either as many
/// elements along `axis` as `right` or exactly one more.
///
/// Both operands are dilated into the output extent with interior padding — writing the padding identity into the
/// positions the other operand occupies — and added, which is JAX's `_interleave`. The addition is exact because the
/// two dilated operands have disjoint support and the padding is the additive identity.
fn scan_interleave<V>(
    left: &V,
    right: &V,
    shape: &StaticShape,
    axis: usize,
    left_count: usize,
    right_count: usize,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Add + Pad,
    V::DispatchDomain: Zero<V>,
{
    if left_count != right_count && left_count != right_count + 1 {
        return Err(TypeError::invalid(format!(
            "`associative_scan` cannot interleave {left_count} elements with {right_count} elements"
        ))
        .into());
    }
    let padding_value = left.dispatch_domain().zero(&dependency_scalar_type(left.r#type().as_ref())?)?;
    let mut edge_padding_low = vec![0; shape.rank()];
    let mut edge_padding_high = vec![0; shape.rank()];
    let mut interior_padding = vec![0; shape.rank()];
    interior_padding[axis] = 1;
    edge_padding_high[axis] = i64::from(left_count == right_count);
    let dilated_left = left.pad(&padding_value, &edge_padding_low, &edge_padding_high, &interior_padding)?;
    edge_padding_low[axis] = 1;
    edge_padding_high[axis] = i64::from(left_count != right_count);
    let dilated_right = right.pad(&padding_value, &edge_padding_low, &edge_padding_high, &interior_padding)?;
    dilated_left.add(&dilated_right)
}

/// Defines one *nonlinear* member of the cumulative family, which is everything the member does not choose for
/// itself: the canonical name constant, the `{axis, reverse}` payload with its accessors, the `Display` and
/// [`Operation`](crate::programs::Operation) implementations, the element-data-type gate layered onto
/// [`cumulative_abstract`], eager interpretation, default partial evaluation, the ragged-aware batching rule, the
/// forward-mode rule that differentiates through [`associative_scan`], the non-transposable declaration, and the
/// value-level staging capability with its blanket implementation over context-carrying values.
///
/// [`CumulativeSumOperation`] is deliberately not generated here. It is the family's one linear member, so it owns a
/// real transposition rule and a forward mode that simply rides the tangent through the same primitive instead of
/// paying for the associative-scan decomposition — two of the very pieces this macro fixes.
///
/// The expansion refers to its dependencies by bare name, as the collective family macros do, so each invoking
/// module imports exactly what the generated code mentions and the rustdoc links in the generated documentation
/// resolve in that module's own scope.
macro_rules! define_cumulative_operation {
    // Public and only form. `element_domain` and `element_domain_error` state the member's accepted element data
    // types and the exact diagnostic for everything else; `ragged_identity` names the value that neutralizes the
    // padding of a bounded ragged scanned axis; and `combine_operation` with `combine` supply the associative
    // operator that the forward-mode decomposition stages and differentiates through.
    (
        $(#[$operation_documentation:meta])*
        operation = $operation:ident,
        name = $operation_name:ident = $name_literal:literal,
        abstract_rule = $abstract_rule:ident,
        element_domain = |$data_type:ident| $element_domain:expr,
        element_domain_error = $element_domain_error:expr,
        ragged_identity = $ragged_identity:expr,
        combine_operation = $combine_operation:ty,
        combine = |$left:ident, $right:ident| $combine:expr,
        $(#[$capability_documentation:meta])*
        capability = $capability:ident::{$forward:ident, $reverse:ident} $(,)?
    ) => {
        #[doc = concat!("Canonical name of the [`", stringify!($operation), "`].")]
        pub const $operation_name: &str = $name_literal;

        $(#[$operation_documentation])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $operation {
            /// Scanned axis, in the operand's own coordinate system.
            axis: usize,

            /// Whether the scan accumulates from the end of the scanned axis toward its start.
            reverse: bool,
        }

        impl $operation {
            #[doc = concat!(
                "Creates a new forward [`", stringify!($operation), "`] scanning along `axis`. The scanned extent \
                 is not part of the operation payload: it is recoverable from the staged input type wherever a rule \
                 needs it.",
            )]
            #[inline]
            pub fn new(axis: usize) -> Self {
                Self { axis, reverse: false }
            }

            /// Returns this operation with its scan direction set to `reverse`, accumulating from the end of the
            /// scanned axis toward its start.
            #[inline]
            pub fn with_reverse(mut self, reverse: bool) -> Self {
                self.reverse = reverse;
                self
            }

            /// Returns the scanned axis.
            #[inline]
            pub fn axis(&self) -> usize {
                self.axis
            }

            /// Returns whether the scan accumulates from the end of the scanned axis toward its start.
            #[inline]
            pub fn reverse(&self) -> bool {
                self.reverse
            }
        }

        impl Display for $operation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.render(formatter, 0)
            }
        }

        impl Operation for $operation {
            type Type = ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                $operation_name
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![$abstract_rule(&input_types[0], self.axis)?])
            }

            // The scan direction renders only when it is set, keeping the common forward scan compact.
            fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
                OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
                    operation.field("axis", self.axis)?;
                    if self.reverse {
                        operation.field("reverse", self.reverse)?;
                    }
                    Ok(())
                })
            }
        }

        #[doc = concat!(
            "Returns the output [`ArrayType`] of a `", $name_literal, "` of `input` along `axis`, layering this \
             member's element data-type domain onto the geometry validated by [`cumulative_abstract`]. The eager \
             kernel and the operation's type inference share this rule, so a directly invoked [`",
            stringify!($capability),
            "`] capability rejects exactly what a staged program rejects.",
        )]
        pub(crate) fn $abstract_rule(input: &ArrayType, axis: usize) -> Result<ArrayType, TypeError> {
            let $data_type = input.data_type();
            if !($element_domain) {
                return Err(TypeError::invalid($element_domain_error));
            }
            cumulative_abstract(input, axis, $operation_name)
        }

        impl<C: Domain<Type = ArrayType, Value: $capability>> InterpretableOperation<C> for $operation {
            fn interpret<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![match self.reverse {
                    true => inputs[0].$reverse(self.axis)?,
                    false => inputs[0].$forward(self.axis)?,
                }])
            }
        }

        // Partial evaluation defers to the default fold-or-residualize behavior of
        // `Program::partially_evaluate`.
        impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for $operation where
            C::Operation: From<$operation>
        {
        }

        // The scanned axis is expressed in the per-item coordinate system, so the rule lifts it past the inserted
        // batch dimension with `lift_cumulative_axis` and re-interprets the lifted scan over the physical batched
        // value. Padding along a *scanned* ragged axis is neutralized with the member's identity first, and the
        // operand's ragged axes ride through onto the result because a scan consumes none of them.
        impl<C: Context<Type = ArrayType>, P: RaggedArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
            for $operation
        where
            $operation: InterpretableOperation<C>,
        {
            fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
                &self,
                context: &BatchingContext<C, ArrayBatching<P>>,
                _driver: &D,
                inputs: &[ArrayBatch<C::Value>],
            ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
                check_count!("input", inputs, 1, ProgramError);
                // A replicated operand carries no inserted batch dimension, so its scanned axis needs no shift.
                // Ragged axes are packed positions in both cases, and so the masking and rewrapping below use the
                // lifted axis.
                let (lifted_axis, output_batch_axis) = match inputs[0].batch_axis_position() {
                    Some(batch_axis) => {
                        let (lifted_axis, output_axis) = lift_cumulative_axis(self.axis, batch_axis);
                        (lifted_axis, BatchAxis::from_position(output_axis))
                    }
                    None => (self.axis, BatchAxis::replicated()),
                };
                let input =
                    match inputs[0].ragged_axes().iter().any(|ragged_axis| ragged_axis.axis() == lifted_axis) {
                        true => P::mask_identity_input(context, &inputs[0], &[lifted_axis], $ragged_identity)?,
                        false => inputs[0].clone(),
                    };
                let lifted = Self { axis: lifted_axis, reverse: self.reverse };
                let mut outputs = lifted.interpret_with_batch_axes(
                    context,
                    std::slice::from_ref(&input),
                    std::slice::from_ref(&output_batch_axis),
                )?;
                check_count!("output", outputs, 1, ProgramError);
                // Interpretation carries values rather than batch metadata, so the operand's ragged axes are
                // restored here.
                let output = ArrayBatch::new(outputs.remove(0).into_value(), output_batch_axis)?
                    .with_ragged_axes(input.ragged_axes().to_vec())?;
                Ok(BatchedOutputs::new(vec![output], Vec::new()))
            }
        }

        // The member is not linear in its operand, so the forward-mode rule differentiates through the
        // associative-scan decomposition with the member's own combining operator, and every primitive that
        // construction stages contributes its own forward-mode rule. The composite array universe reaches this rule
        // through the default projected fall-through of `MemberDifferentiableOperation`, because the operation is
        // shape-preserving and its operand never needs the replication a broadcasting elementwise member does.
        impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for $operation
        where
            C::Operation: From<AddOperation<ArrayType>>
                + From<ConcatenateOperation<ArrayType>>
                + From<$combine_operation>
                + From<$operation>
                + From<PadOperation<ArrayType>>
                + From<SliceOperation>
                + ZeroOperationProvider<ArrayType>,
            C::Value: $capability,
        {
            fn jvp<D: DifferentiationDriver<C>>(
                &self,
                context: &C,
                driver: &D,
                inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                check_count!("input", inputs, 1, ProgramError);
                let primal_input = inputs[0].primal();
                let MaybeZero::Value(tangent_input) = inputs[0].tangent() else {
                    // Every JVP is linear in its tangent, so a structural zero tangent stays a structural zero one,
                    // and the primal is cheaper to obtain from the primitive itself than from the decomposition.
                    let primal = match self.reverse {
                        true => primal_input.$reverse(self.axis)?,
                        false => primal_input.$forward(self.axis)?,
                    };
                    let tangent = MaybeZero::Zero(primal.r#type().tangent()?);
                    return Ok(vec![DifferentiationDual::new(primal, tangent)?]);
                };
                Ok(vec![jvp_through_associative_scan(
                    context,
                    driver,
                    primal_input,
                    tangent_input,
                    self.axis,
                    self.reverse,
                    |$left, $right| $combine,
                )?])
            }
        }

        // The member is not linear in its operand, so it has no primitive transposition rule. Reverse-mode
        // differentiation remains available by transposing the linear operations that the forward-mode rule stages.
        impl_non_transposable_operation!($operation);

        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            #[doc = concat!(
                "Returns the inclusive prefix scan of `self` along `axis`, as defined by [`",
                stringify!($operation),
                "`].",
            )]
            fn $forward(&self, axis: usize) -> Result<Self, ProgramError>;

            #[doc = concat!(
                "Returns the inclusive suffix scan of `self` along `axis`, as defined by [`",
                stringify!($operation),
                "`] in its reverse direction (i.e., the prefix scan accumulated from the end of the axis toward \
                 its start).",
            )]
            fn $reverse(&self, axis: usize) -> Result<Self, ProgramError>;
        }

        // Any context-carrying value scans by binding the operation through its own context. The
        // `From<..Operation>` bound makes this disjoint from the eager value types (whose context operation is
        // `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete
        // implementations.
        impl<V: Value<Type = ArrayType>> $capability for V
        where
            V::DispatchDomain: Context<Type = ArrayType>,
            <V::DispatchDomain as Domain>::Operation: From<$operation>,
        {
            #[inline]
            fn $forward(&self, axis: usize) -> Result<Self, ProgramError> {
                Ok(self.dispatch_domain().bind($operation::new(axis), Vec::new(), &[self.clone()])?.remove(0))
            }

            #[inline]
            fn $reverse(&self, axis: usize) -> Result<Self, ProgramError> {
                Ok(self
                    .dispatch_domain()
                    .bind($operation::new(axis).with_reverse(true), Vec::new(), &[self.clone()])?
                    .remove(0))
            }
        }
    };
}

pub(super) use define_cumulative_operation;

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension, StaticShape,
    };
    use crate::macros::{check_operation_partial_evaluation, check_operation_transposition};
    use crate::operations::cumulative::cumulative_product::CumulativeProductOperation;
    use crate::parameters::Placeholder;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::{ProgramRenderingMode, Provenance, TypeError};

    use super::*;

    #[test]
    fn test_cumulative_abstract_returns_the_input_type() {
        // A prefix scan preserves the complete operand type, including its memory placement and its sharding of the
        // unscanned axes.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input = ArrayType::new_static(DataType::F64, [2, 3])
            .with_memory(Memory::Host { pinned: true })
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            )
            .unwrap();
        assert_eq!(cumulative_abstract(&input, 1, "cumulative_sum"), Ok(input.clone()));

        // Scanning the sharded axis would need cross-shard communication that this operation does not carry.
        assert_eq!(
            cumulative_abstract(&input, 0, "cumulative_sum"),
            Err(TypeError::invalid(format!(
                "`cumulative_sum` requires an unsharded scanned dimension but axis 0 of {input} is sharded"
            ))),
        );
    }

    #[test]
    fn test_cumulative_abstract_rejects_out_of_bounds_and_dynamic_scanned_axes() {
        let input = ArrayType::new_static(DataType::F64, [2, 3]);
        assert_eq!(
            cumulative_abstract(&input, 2, "cumulative_sum"),
            Err(TypeError::invalid("`cumulative_sum` axis 2 is out of bounds for rank 2".to_string())),
        );

        // The scan is defined by the exact number of accumulated elements, so both bounded and unbounded dynamic
        // scanned dimensions are rejected while an unscanned dynamic axis passes through untouched.
        let bounded = DimensionVariable::new("length", DimensionBounds::new(0, Some(4)).unwrap());
        let unbounded = DimensionVariable::new("batch", DimensionBounds::unbounded());
        for variable in [bounded, unbounded] {
            let input = ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(variable.clone()), Dimension::Static(3)]),
            );
            assert_eq!(
                cumulative_abstract(&input, 0, "cumulative_sum"),
                Err(TypeError::invalid(format!(
                    "`cumulative_sum` requires a static scanned dimension but axis 0 of {input} is dynamic"
                ))),
            );
            assert_eq!(cumulative_abstract(&input, 1, "cumulative_sum"), Ok(input));
        }
    }

    #[test]
    fn test_lift_cumulative_axis_shifts_axes_at_or_above_the_batch_axis() {
        // A scanned axis below the inserted batch dimension keeps its position, and one at or above it shifts up.
        // The output batch axis never moves, because a prefix scan preserves the operand rank.
        assert_eq!(lift_cumulative_axis(0, 1), (0, 1));
        assert_eq!(lift_cumulative_axis(1, 1), (2, 1));
        assert_eq!(lift_cumulative_axis(0, 0), (1, 0));
        assert_eq!(lift_cumulative_axis(2, 0), (3, 0));
    }

    #[test]
    fn test_associative_scan_matches_the_sequential_scan() {
        // The decomposition is checked against the sequential scan of the same combiner, over both parities of the
        // scanned extent and in both directions. Summation pins the positions each output accumulates over, and the
        // left projection — associative but not commutative — additionally pins the operand order the construction
        // passes to the combiner: its forward scan is the first element repeated and its reverse scan the last.
        let add = |left: &Array, right: &Array| left.add(right);
        let first = |left: &Array, _right: &Array| Ok(left.clone());
        for extent in 0..=9_usize {
            let values = (1..=extent).map(|value| value as f64).collect::<Vec<_>>();
            let input = Array::vector(values.clone());
            let shape = StaticShape::new(vec![extent]);
            for reverse in [false, true] {
                assert_eq!(
                    associative_scan(&input, 0, reverse, &add).map(|output| output.to_f64s()),
                    cumulative_evaluate(values.as_slice(), &shape, 0, reverse, |left, right| Ok(left + right)),
                    "summation over extent {extent}, reverse {reverse}",
                );
                assert_eq!(
                    associative_scan(&input, 0, reverse, &first).map(|output| output.to_f64s()),
                    cumulative_evaluate(values.as_slice(), &shape, 0, reverse, |left, _right| Ok(left)),
                    "left projection over extent {extent}, reverse {reverse}",
                );
            }
        }

        // The construction scans one axis of a higher-rank operand independently per row.
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            associative_scan(&matrix, 1, false, &add),
            Ok(Array::matrix(2, 3, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0])),
        );
        assert_eq!(
            associative_scan(&matrix, 1, true, &add),
            Ok(Array::matrix(2, 3, vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0])),
        );
        assert_eq!(
            associative_scan(&matrix, 0, false, &add),
            Ok(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0])),
        );

        // The construction slices at staging-time positions, so it needs a fully static operand shape and an
        // in-bounds axis.
        assert_eq!(
            associative_scan(&matrix, 2, false, &add),
            Err(ProgramError::Type(TypeError::invalid(
                "`associative_scan` axis 2 is out of bounds for rank 2".to_string(),
            ))),
        );
    }

    #[test]
    fn test_cumulative_evaluate_scans_the_selected_axis() {
        let values = (1..=6).map(|value| value as f64).collect::<Vec<_>>();
        let shape = StaticShape::new(vec![2, 3]);

        let add = |left: f64, right: f64| Ok(left + right);

        // Forward scans accumulate prefixes and reverse scans accumulate suffixes, per row of the scanned axis.
        assert_eq!(
            cumulative_evaluate(values.as_slice(), &shape, 1, false, add),
            Ok(vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]),
        );
        assert_eq!(
            cumulative_evaluate(values.as_slice(), &shape, 1, true, add),
            Ok(vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0]),
        );
        // Scanning the outer axis accumulates across the row stride instead.
        assert_eq!(
            cumulative_evaluate(values.as_slice(), &shape, 0, false, add),
            Ok(vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]),
        );

        // A scanned axis with fewer than two elements has nothing to accumulate.
        assert_eq!(cumulative_evaluate(values.as_slice(), &StaticShape::new(vec![6, 1]), 1, false, add), Ok(values),);
        assert_eq!(cumulative_evaluate(&[], &StaticShape::new(vec![0, 3]), 0, false, add), Ok(Vec::<f64>::new()));

        // A zero-extent axis to the *right* of the scanned one leaves the payload empty while the scanned extent
        // itself is still at least two, so the block bounds are derived from dimension products rather than from the
        // payload length and the scanned axis stride, which is zero here.
        assert_eq!(cumulative_evaluate(&[], &StaticShape::new(vec![2, 0]), 0, false, add), Ok(Vec::<f64>::new()));
        assert_eq!(cumulative_evaluate(&[], &StaticShape::new(vec![3, 0, 2]), 0, true, add), Ok(Vec::<f64>::new()));
    }

    #[test]
    fn test_generated_cumulative_members_partially_evaluate_by_folding() {
        // The macro generates the default fold-or-residualize partial-evaluation behavior for every nonlinear
        // member, so a fully known operand folds to the scanned value. One member covers the generated
        // implementation for the whole family.
        check_operation_partial_evaluation!(
            operation = CumulativeProductOperation::new(0),
            inputs = [Array::vector(vec![1.0, 2.0, 3.0])],
            expected = Array::vector(vec![1.0, 2.0, 6.0]),
        );
    }

    #[test]
    fn test_generated_cumulative_members_reject_transposition() {
        // Every macro-generated member is nonlinear and therefore declares no transposition rule, reverse mode
        // reaching it instead through the primitives its forward-mode decomposition stages. One member covers the
        // generated declaration for the whole family.
        check_operation_transposition!(
            @rejected,
            operation = CumulativeProductOperation::new(0),
            input_types = [ArrayType::new_static(DataType::F64, [3])],
        );
    }

    #[test]
    fn test_cumulative_product_jvp_stages_the_associative_scan_decomposition() {
        // A nonlinear member's forward mode differentiates *through* the decomposition, so the fused program holds
        // no `cumulative_product` instruction at all: it is the parallel-prefix construction (four halving levels
        // over a length-four axis) with each of its primitives' own rules interleaved. The primal half is
        // recomputed there rather than taken from the primitive, exactly as in JAX's `_cumulative_jvp_rule`.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [4]));
        let outputs = builder
            .add_instruction(ArrayOperation::from(CumulativeProductOperation::new(0)), Vec::new(), vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder.build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder]).unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f64[4], %1:f64[4] .
                let %2:f64[2] = slice [start_indices=[0], limit_indices=[3], strides=[2]] %0
                    %3:f64[2] = slice [start_indices=[0], limit_indices=[3], strides=[2]] %1
                    %4:f64[2] = slice [start_indices=[1], limit_indices=[4], strides=[2]] %0
                    %5:f64[2] = slice [start_indices=[1], limit_indices=[4], strides=[2]] %1
                    %6:f64[2] = mul %2 %4
                    %7:f64[2] = mul %4 %3
                    %8:f64[2] = mul %2 %5
                    %9:f64[2] = add %7 %8
                    %10:f64[1] = slice [start_indices=[0], limit_indices=[1], strides=[2]] %6
                    %11:f64[1] = slice [start_indices=[0], limit_indices=[1], strides=[2]] %9
                    %12:f64[1] = slice [start_indices=[1], limit_indices=[2], strides=[2]] %6
                    %13:f64[1] = slice [start_indices=[1], limit_indices=[2], strides=[2]] %9
                    %14:f64[1] = mul %10 %12
                    %15:f64[1] = mul %12 %11
                    %16:f64[1] = mul %10 %13
                    %17:f64[1] = add %15 %16
                    %18:f64[1] = slice [start_indices=[0], limit_indices=[1]] %6
                    %19:f64[1] = slice [start_indices=[0], limit_indices=[1]] %9
                    %20:f64[] = zero [type=f64[]]
                    %21:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %18 %20
                    %22:f64[] = zero [type=f64[]]
                    %23:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %19 %22
                    %24:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %14 %20
                    %25:f64[] = zero [type=f64[]]
                    %26:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %17 %25
                    %27:f64[2] = add %21 %24
                    %28:f64[2] = add %23 %26
                    %29:f64[1] = slice [start_indices=[0], limit_indices=[1]] %0
                    %30:f64[1] = slice [start_indices=[0], limit_indices=[1]] %1
                    %31:f64[1] = slice [start_indices=[0], limit_indices=[1]] %27
                    %32:f64[1] = slice [start_indices=[0], limit_indices=[1]] %28
                    %33:f64[1] = slice [start_indices=[2], limit_indices=[4], strides=[2]] %0
                    %34:f64[1] = slice [start_indices=[2], limit_indices=[4], strides=[2]] %1
                    %35:f64[1] = mul %31 %33
                    %36:f64[1] = mul %33 %32
                    %37:f64[1] = mul %31 %34
                    %38:f64[1] = add %36 %37
                    %39:f64[2] = concatenate [axis=0] %29 %35
                    %40:f64[2] = concatenate [axis=0] %30 %38
                    %41:f64[] = zero [type=f64[]]
                    %42:f64[4] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %39 %41
                    %43:f64[] = zero [type=f64[]]
                    %44:f64[4] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %40 %43
                    %45:f64[4] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %27 %41
                    %46:f64[] = zero [type=f64[]]
                    %47:f64[4] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %28 %46
                    %48:f64[4] = add %42 %45
                    %49:f64[4] = add %44 %47
                in (%48, %49)
            "}
            .trim_end(),
        );

        // The staged program is the derivative it claims to be, at concrete values.
        let primals = Array::vector(vec![1.0, 2.0, 3.0, 4.0]);
        let tangents = Array::vector(vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            jvp.interpret(vec![primals, tangents]),
            Ok(vec![Array::vector(vec![1.0, 2.0, 6.0, 24.0]), Array::vector(vec![1.0, 3.0, 11.0, 50.0])]),
        );
    }

    #[test]
    fn test_associative_scan_attributes_its_staged_instructions() {
        // Every instruction the decomposition stages carries the nested framework scopes, which is purely
        // diagnostic: the canonical semantic rendering stays suffix-free and the values are unaffected.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::new_static(DataType::F64, [2]));
        let outputs = builder
            .add_instruction(ArrayOperation::from(CumulativeProductOperation::new(0)), Vec::new(), vec![input], None)
            .unwrap()
            .to_vec();
        let jvp = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder], vec![Placeholder])
            .unwrap()
            .jvp()
            .unwrap();
        let expected_provenance = Provenance::scope(
            ProvenanceScope::new("ryft"),
            Provenance::scope(
                ProvenanceScope::new("differentiation"),
                Provenance::scope(ProvenanceScope::new("associative_scan"), Provenance::unknown()),
            ),
        );
        assert!(
            jvp.instructions().iter().all(|instruction| instruction.provenance() == &expected_provenance),
            "every decomposition instruction is attributed",
        );
        assert_eq!(
            std::fmt::from_fn(|formatter| jvp.render(formatter, 0, ProgramRenderingMode::WithProvenance)).to_string(),
            indoc! {"
                lambda %0:f64[2], %1:f64[2] .
                let %2:f64[1] = slice [start_indices=[0], limit_indices=[1], strides=[2]] %0 ; ryft::differentiation::associative_scan
                    %3:f64[1] = slice [start_indices=[0], limit_indices=[1], strides=[2]] %1 ; ryft::differentiation::associative_scan
                    %4:f64[1] = slice [start_indices=[1], limit_indices=[2], strides=[2]] %0 ; ryft::differentiation::associative_scan
                    %5:f64[1] = slice [start_indices=[1], limit_indices=[2], strides=[2]] %1 ; ryft::differentiation::associative_scan
                    %6:f64[1] = mul %2 %4 ; ryft::differentiation::associative_scan
                    %7:f64[1] = mul %4 %3 ; ryft::differentiation::associative_scan
                    %8:f64[1] = mul %2 %5 ; ryft::differentiation::associative_scan
                    %9:f64[1] = add %7 %8 ; ryft::differentiation::associative_scan
                    %10:f64[1] = slice [start_indices=[0], limit_indices=[1]] %0 ; ryft::differentiation::associative_scan
                    %11:f64[1] = slice [start_indices=[0], limit_indices=[1]] %1 ; ryft::differentiation::associative_scan
                    %12:f64[] = zero [type=f64[]] ; ryft::differentiation::associative_scan
                    %13:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %10 %12 ; ryft::differentiation::associative_scan
                    %14:f64[] = zero [type=f64[]] ; ryft::differentiation::associative_scan
                    %15:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %11 %14 ; ryft::differentiation::associative_scan
                    %16:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %6 %12 ; ryft::differentiation::associative_scan
                    %17:f64[] = zero [type=f64[]] ; ryft::differentiation::associative_scan
                    %18:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %9 %17 ; ryft::differentiation::associative_scan
                    %19:f64[2] = add %13 %16 ; ryft::differentiation::associative_scan
                    %20:f64[2] = add %15 %18 ; ryft::differentiation::associative_scan
                in (%19, %20)
            "}
            .trim_end(),
        );
    }
}
