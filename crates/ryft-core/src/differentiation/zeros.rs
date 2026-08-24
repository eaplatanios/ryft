use std::fmt::Display;
use std::ops::Range;

use crate::contexts::Context;
use crate::differentiation::DifferentiationError;
use crate::differentiation::types::DifferentiableType;
use crate::macros::check_count;
use crate::operations::{Zero, ZeroOperation, ZeroOperationProvider};
use crate::programs::{AtomId, MaybeZero, Operation, ProgramBuilder, ProgramError, Type, Typed, Value};

/// Differentiation-owned protocol through which an operation family materializes zeros whose runtime geometry must
/// be supplied by explicitly captured _residual_ values, because it is not derivable from the zero's [`Type`] alone.
///
/// # Why this Protocol Exists
///
/// Differentiation is the one transform that must synthesize values with no data edge to derive them from. For example,
/// transposition is *defined* to return a cotangent for every differentiated input, including inputs that are
/// disconnected from every output, and the mathematically determined value for such an input is a zero of its cotangent
/// type. For a static type this is easy as [`ZeroOperationProvider::zero_operation`] constructs the zero from the type,
/// with no operands. For a type with dynamic axes it is impossible as a [`Type`] carries only dimension _identities_
/// and bounds, never defining values, and so the zero operation needs one explicit dimension operand per dynamic axis.
/// Also, the value that could supply those operands (i.e., the primal input the cotangent corresponds to) is _not an
/// input of the pullback program_ where the zero must be staged. The only moment both the need and the geometry are in
/// scope is during linearization, and so the required extents must be captured then and threaded to transposition as
/// ordinary residuals. This trait is that capture/spend contract, expressed once per operation *family*. It is a set of
/// associated functions with no receiver (i.e., `self` argument), because it is invoked precisely when no [`Operation`]
/// instance exists.
///
/// # How to Use It
///
/// The protocol has three steps, executed by the differentiation machinery rather than by implementors:
///
///   1. **Declare:** When linearization finds a differentiated input whose cotangent will be disconnected, it calls
///      [`Self::zero_residual_types`] with the zero's type to learn which residual values the eventual zero needs
///      (e.g., one first-class dimension per *distinct* dynamic identity for a composite array type, and nothing for
///      a static type).
///   2. **Capture:** While the primal value is still in scope, linearization records those residuals from it.
///      [`Self::capture_zero_residuals`] stages the reads into the program being built (i.e., the program-level
///      [`Program::linearize`](crate::Program::linearize) path), and [`Self::capture_zero_residual_values`] is its
///      value-level counterpart for
///      reusable derivative callables that close over concrete or tracer values. Program transposition appends captured
///      residuals to its ordinary trailing residual suffix. Reusable callables retain boundary-reconstruction residuals
///      beside that executable program.
///   3. **Spend:** [`Self::zero_operation_with_residuals`] assembles the zero operation and its operands from the
///      captured residuals. Callers then stage that operation inside a pullback program or bind it in the originating
///      [`Context`] of a reusable value-level derivative callable.
///
/// The three steps must agree on residual count and order. Every mismatch is a loud typed error (the capture sites
/// validate against the declared types, and the spend site validates the residual count), never a silently wrong-shaped
/// zero.
///
/// [`Self::materialize_zero_from_residual_sources`] runs the same three steps at a transform _boundary_, where the
/// primal that pinned a zero's extents is out of scope and the named quantities must instead be gathered one at a time
/// from the peers that are (i.e., live sibling cotangents, known operands, and first-class dimension operands). It
/// replaces the exemplar-matching materialization that structural zeros previously used, which could only construct a
/// zero whose type some live value reproduced exactly and therefore rejected every widened differential representation.
///
/// # Who Implements It
///
/// Almost nobody needs to implement this trait, by design. Every operation family with an input-free zero (i.e., every
/// family with a `From<ZeroOperation<T>>` conversion) receives the whole protocol through a blanket implementation
/// that declares nothing, captures nothing, and spends by constructing the type-only zero (i.e., the fail-loud default
/// rejects unexpected residuals rather than ignoring them, so a mismatched linearize/transpose pairing cannot be
/// silently accepted). Only families whose zero genuinely consumes runtime-geometry operands (e.g., the composite
/// program family and its XLA counterpart) override the declaration, capture, and operation-assembly functions. Every
/// spending path reuses that shared assembly.
///
/// [`LinearCallOperation`](crate::LinearCallOperation) is this protocol's sibling. It retains residual geometry for the
/// transpose of a *non-trivial* residual-parameterized linear map by attaching explicit forward/transpose regions to an
/// instruction, while this trait retains it for the degenerate zero map, which has no instruction to attach anything
/// to. Both exist for the same reason (i.e., reverse mode needs geometry at a moment when its defining values would
/// otherwise be out of scope) and both keep residual selection and threading owned by the differentiation transform
/// rather than leaking into primal operation payloads.
pub trait ResidualZeroProvider<T: Type>: ZeroOperationProvider<T> {
    /// Returns the types of the residual values that a zero of `r#type` needs, in the exact order in which
    /// [`Self::capture_zero_residuals`] captures them and [`Self::zero_operation_with_residuals`] consumes them.
    /// Input-free [`Operation`] families use the empty default. The array-dimension composite family returns one
    /// dimension type per _distinct_ dynamic identity of `r#type`, in first-occurrence order, so repeated axes share
    /// one residual.
    #[inline]
    fn zero_residual_types(_type: &T) -> Vec<T> {
        Vec::new()
    }

    /// Stages instructions into `builder` that read the residual values declared by [`Self::zero_residual_types`] from
    /// the primal value `source` (e.g., one `dimension_size` read per declared residual), returning the new atoms in
    /// declaration order. Linearization calls this while `source` is still in scope of the program being built. The
    /// returned atoms are then threaded to transposition as ordinary residuals. Input-free [`Operation`] families
    /// capture nothing.
    #[inline]
    fn capture_zero_residuals<V: Value<Type = T>>(
        _builder: &mut ProgramBuilder<V, Self>,
        _source: AtomId,
        _type: &T,
    ) -> Result<Vec<AtomId>, ProgramError> {
        Ok(Vec::new())
    }

    /// Captures the residual values declared by [`Self::zero_residual_types`] from the primal value `source`
    /// in a live `context`, returning them in declaration order. This is the value-level counterpart of
    /// [`Self::capture_zero_residuals`] used by reusable pullback callables, whose captured residuals are
    /// concrete values or tracers closed over by the callable rather than atoms of a program under construction. The
    /// default resolves each declared residual independently through [`Self::capture_zero_residual_value`], which lets
    /// operation families implement one identity-directed value-level capture primitive. Input-free families declare
    /// no residuals and therefore return an empty list without consulting `source`.
    fn capture_zero_residual_values<C: Context<Type = T, Operation = Self>>(
        context: &C,
        source: &C::Value,
        r#type: &T,
    ) -> Result<Vec<C::Value>, ProgramError> {
        Self::zero_residual_types(r#type)
            .into_iter()
            .enumerate()
            .map(|(index, residual_type)| {
                Self::capture_zero_residual_value(context, source, &residual_type)?.ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "zero residual {index} of type {residual_type} cannot be captured from source of type {}",
                        source.r#type().as_ref(),
                    ))
                })
            })
            .collect()
    }

    /// Captures the single residual value declared at `residual_type` from `source`, or returns [`None`] when `source`
    /// does not carry the runtime quantity that residual names. This is the _identity-directed_ capture step used by
    /// [`Self::materialize_zero_from_residual_sources`], where the geometry of one zero may have to be assembled from
    /// several unrelated values rather than read off one primal of exactly the zero's own type.
    ///
    /// Implementations must inspect `source`'s [`Type`] before staging anything and return [`None`] without side
    /// effects when it does not carry the named quantity, because the caller tries candidates in order and a
    /// speculative read would leave dead instructions behind. Input-free [`Operation`] families declare no residuals
    /// and therefore never reach this function, so the default answers [`None`].
    #[inline]
    fn capture_zero_residual_value<C: Context<Type = T, Operation = Self>>(
        _context: &C,
        _source: &C::Value,
        _residual_type: &T,
    ) -> Result<Option<C::Value>, ProgramError> {
        Ok(None)
    }

    /// Returns the canonical zero operation for `r#type` and expands `residuals` into its operand order. The default
    /// represents an input-free zero operation. Families whose zero consumes runtime geometry override this function
    /// so that value-level binding, residualization, and builder-level staging share one operation assembly.
    #[inline]
    fn zero_operation_with_residuals<R: Clone>(r#type: T, residuals: &[R]) -> Result<(Self, Vec<R>), ProgramError>
    where
        Self: Operation<Type = T>,
    {
        if !residuals.is_empty() {
            return Err(ProgramError::InvalidArgument {
                message: format!("input-free zero expected 0 residuals but got {}", residuals.len()),
            });
        }
        Ok((Self::zero_operation(r#type)?, Vec::new()))
    }

    /// Returns the value inside `zero`, materializing a structural [`MaybeZero::Zero`] whose [`Type`] cannot construct
    /// it alone by reading the runtime geometry it names from the values in `geometry_sources`. This is the boundary
    /// form of the residual protocol. [`Self::capture_zero_residuals`] and [`Self::capture_zero_residual_values`]
    /// capture from _the_ primal, which every linearization site has in hand. A transform boundary often does not.
    /// A transposed control-flow instruction needs a real operand for the cotangent of a dead output, and the primal
    /// that pinned that output's extents is long out of scope. What is in scope is a set of peers (i.e., live sibling
    /// cotangents, known operands, and first-class dimension operands), among which the named runtime quantities are
    /// collectively available even when no single peer has the zero's type. This function therefore works _per declared
    /// residual_ rather than per exemplar: it asks each candidate in turn for one named quantity through
    /// [`Self::capture_zero_residual_value`] and assembles the zero from the answers.
    ///
    /// Being identity-directed rather than exemplar-directed is what makes it type-general. A tangent or cotangent
    /// type is derived from its primal's by [`DifferentiableType`], which rewrites element representation, layout, and
    /// sharding while preserving geometry exactly, so requiring an exemplar of the zero's own type would reject every
    /// widened differential representation (e.g., an `f8e8m0fnu[n]` primal whose tangent is `f32[n]`). Naming the
    /// runtime quantity instead of matching the whole type accepts them all.
    ///
    /// A type that declares no residuals keeps the type-only nullary zero, so a statically shaped program stages
    /// no additional instruction and its zero-producing marker keeps higher-order partial evaluation structural. A
    /// declared residual that no candidate supplies is a loud [`ProgramError::UnsupportedOperation`] naming the type
    /// and the missing quantity, never a silently wrong-shaped zero.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] in which the zero and any geometry reads are staged or computed.
    ///   - `zero`: Structural zero or live value to materialize.
    ///   - `sources`: Candidate values in scope at the boundary, searched in order for each declared residual. Only
    ///     zero types that declare residuals consult them.
    fn materialize_zero_from_residual_sources<
        'v,
        C: Context<Type = T, Value: 'v, Operation = Self> + Zero<C::Value>,
        I: IntoIterator<Item = &'v C::Value>,
    >(
        context: &C,
        zero: MaybeZero<C::Value>,
        sources: I,
    ) -> Result<C::Value, ProgramError>
    where
        Self: Operation<Type = T>,
    {
        let r#type = match zero {
            MaybeZero::Value(value) => return Ok(value),
            MaybeZero::Zero(r#type) => r#type,
        };
        let residual_types = Self::zero_residual_types(&r#type);
        if residual_types.is_empty() {
            return context.zero(&r#type);
        }
        let sources = sources.into_iter().collect::<Vec<_>>();
        let mut residuals = Vec::with_capacity(residual_types.len());
        for (index, residual_type) in residual_types.iter().enumerate() {
            let mut captured = None;
            for source in &sources {
                if let Some(value) = Self::capture_zero_residual_value(context, source, residual_type)? {
                    captured = Some(value);
                    break;
                }
            }
            let residual = captured.ok_or_else(|| ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot materialize a zero of type {type} because no value in scope supplies the runtime \
                     geometry {residual_type} that its residual {index} names",
                ),
            })?;
            if residual.r#type().as_ref() != residual_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "zero residual {} has type {} but expected {}",
                    index,
                    residual.r#type().as_ref(),
                    residual_type,
                )));
            }
            residuals.push(residual);
        }
        let (operation, operands) = Self::zero_operation_with_residuals(r#type, residuals.as_slice())?;
        let mut outputs = context.bind(operation, Vec::new(), operands.as_slice())?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

// Every operation family that absorbs a type-only `ZeroOperation` has an input-free zero, and so the defaulted
// residual protocol applies verbatim. Composite families without that conversion implement the protocol directly.
impl<T: Type, O: Operation<Type = T> + From<ZeroOperation<T>>> ResidualZeroProvider<T> for O {}

/// Captures the runtime values needed to materialize a zero of `r#type` and validates the operation family's residual
/// protocol. [`ResidualZeroProvider::zero_residual_types`] declares the residual signature, while
/// [`ResidualZeroProvider::capture_zero_residual_values`] performs the operation-family-specific reads from `source`.
/// This helper calls both and verifies that capture returns exactly the declared number and types of values, in the
/// declared order. A disagreement is a malformed provider implementation and is reported as a
/// [`ProgramError::MalformedProgram`] before the residuals can construct a zero with incorrect runtime geometry.
///
/// For example, suppose `r#type` is `zero[n, n, m]` and `source` is the corresponding primal array. A composite array
/// provider declares one dimension residual per distinct dynamic identity, `[dimension(n), dimension(m)]`, and captures
/// the runtime extents `[n, m]` from the first source axis carrying each identity. This helper validates those two
/// captured values. Later, [`ResidualZeroProvider::zero_operation_with_residuals`] expands them into the per-axis
/// operand order `[n, n, m]`. A static zero declares and captures no residuals.
///
/// This function neither chooses which geometry to retain nor constructs the zero; those responsibilities belong to
/// the operation family. It only enforces the declaration/capture contract for concrete or tracer [`Value`]s. The
/// program-level [`AtomId`] capture path performs the corresponding validation where its atoms are staged.
///
/// # Parameters
///
///   - `context`: Context in which the provider reads residual values from `source`.
///   - `source`: Primal value whose runtime geometry determines the zero.
///   - `r#type`: Type of the zero that will eventually consume the captured residuals.
///   - `site`: Description of the capture site included in malformed-provider diagnostics.
pub(crate) fn capture_and_validate_zero_residual_values<C: Context<Operation: ResidualZeroProvider<C::Type>>>(
    context: &C,
    source: &C::Value,
    r#type: &C::Type,
    site: &str,
) -> Result<Vec<C::Value>, ProgramError> {
    let expected_types = C::Operation::zero_residual_types(r#type);
    let residuals = C::Operation::capture_zero_residual_values(context, source, r#type)?;
    if residuals.len() != expected_types.len() {
        return Err(ProgramError::MalformedProgram(format!(
            "{} captured {} zero residuals but declared {}",
            site,
            residuals.len(),
            expected_types.len(),
        )));
    }
    for (index, (residual, expected_type)) in residuals.iter().zip(expected_types).enumerate() {
        if residual.r#type().as_ref() != &expected_type {
            return Err(ProgramError::MalformedProgram(format!(
                "{} zero residual {} has type {} but expected {}",
                site,
                index,
                residual.r#type().as_ref(),
                expected_type,
            )));
        }
    }
    Ok(residuals)
}

/// Role of differential boundary whose [`ZeroSpaceBoundaryLeaf`]s are reconstructed
/// from [`ZeroSpaceBoundaryReconstruction`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ZeroSpaceBoundaryRole {
    /// Cotangents of the primal input boundary, returned by a _pullback_ function.
    InputCotangent,

    /// Tangents of the primal output boundary, returned by a _pushforward_ function.
    OutputTangent,
}

impl ZeroSpaceBoundaryRole {
    /// Returns the differential [`Type`] represented by this [`ZeroSpaceBoundaryRole`] for `primal_type`.
    #[inline]
    fn differential_type<T: DifferentiableType>(self, primal_type: &T) -> Result<T, DifferentiationError> {
        match self {
            Self::InputCotangent => primal_type.cotangent(),
            Self::OutputTangent => primal_type.tangent(),
        }
    }

    /// Returns the concise description of this [`ZeroSpaceBoundaryRole`] to be used in diagnostics.
    #[inline]
    const fn as_str(self) -> &'static str {
        match self {
            Self::InputCotangent => "input cotangent boundary",
            Self::OutputTangent => "output tangent boundary",
        }
    }
}

impl Display for ZeroSpaceBoundaryRole {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Reconstruction metadata for one zero-space leaf omitted from a compact derivative boundary.
/// This is only used as part of [`ZeroSpaceBoundaryReconstruction`].
pub(crate) struct ZeroSpaceBoundaryLeaf<T: Type> {
    /// Position of the omitted zero in the complete public differential boundary.
    index: usize,

    /// Tangent or cotangent type of the omitted zero.
    r#type: T,

    /// Range of captured residuals consumed to materialize the omitted zero.
    residual_range: Range<usize>,
}

/// Runtime-geometry residuals used to reconstruct the zero-space leaves omitted from one compact derivative boundary.
///
/// Let a flattened primal boundary have leaf types `T₁, …, Tₙ`, and let `D(Tᵢ)` denote the corresponding tangent or
/// cotangent type. When `D(Tᵢ)` is a _zero space_, it contains exactly one value, `0ᵢ`, so the executable derivative
/// [`Program`](crate::Program) omits that leaf entirely. That is because carrying a Single Static Assignment (SSA)
/// input or output for a value that cannot vary would add IR and ABI overhead without conveying information. A public
/// [`Pushforward`](crate::Pushforward) or [`Pullback`](crate::Pullback) must still reconstruct `0ᵢ` when it rebuilds
/// the complete user-facing boundary.
///
/// Static zeros can be constructed from their types alone. Dynamic zeros cannot. For example, the cotangent of a primal
/// `u64[n]` array has type `zero[n]` which records the identity and bounds of `n`, but not its runtime extent. While
/// the primal value is available, linearization therefore captures the minimal runtime geometry declared by
/// [`ResidualZeroProvider::zero_residual_types`] (e.g., the concrete value of `n`). This type stores the flattened
/// concatenation of those captured values in primal-leaf order together with a sparse reconstruction plan for the
/// zero-space leaves. [`Self::rebuild`] replays that plan to materialize each omitted `0ᵢ` and interleave it with the
/// live derivative values produced by the compact program.
///
/// These are **boundary reconstruction residuals**, not ordinary executable-program residuals. They are retained beside
/// the reusable derivative callable and are consumed only while restoring its public boundary. They never become
/// otherwise-unused inputs of the derivative program. The wrapper retains the boundary size, the position and type of
/// every omitted zero, and the range of residuals that reconstructs it. The tangent/cotangent mapping and primal
/// boundary are therefore consumed exactly once during capture and cannot be changed later during reconstruction.
pub struct ZeroSpaceBoundaryReconstruction<V: Value> {
    /// Semantic role of the differential boundary reconstructed by this instance.
    role: ZeroSpaceBoundaryRole,

    /// Flattened runtime-geometry residuals captured in zero-leaf and provider-declaration order.
    residuals: Vec<V>,

    /// Number of leaves in the complete public differential boundary.
    boundary_size: usize,

    /// Sparse reconstruction plan containing one entry for every zero-space boundary leaf, in boundary order.
    zero_leaves: Vec<ZeroSpaceBoundaryLeaf<V::Type>>,
}

impl<V: Value<Type: DifferentiableType>> ZeroSpaceBoundaryReconstruction<V> {
    /// Captures the runtime geometry and reconstruction plan for every zero-space leaf of one differential boundary.
    ///
    /// For each primal leaf type `Tᵢ`, `role` determines the boundary type `D(Tᵢ)`. A nonzero-space `D(Tᵢ)` remains a
    /// live input or output of the compact derivative program and needs no entry in the stored plan. For a zero-space
    /// `D(Tᵢ)`, this function captures and validates the runtime values declared by
    /// [`ResidualZeroProvider::zero_residual_types`] from the corresponding primal value, then records the leaf index,
    /// differential type, and captured residual range. Reconstruction therefore never needs the primal boundary or
    /// differential mapping again.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which the operation family reads runtime geometry from the primal values.
    ///   - `primal_values`: Complete flattened primal boundary values in leaf order.
    ///   - `primal_types`: Complete flattened primal boundary types in the same order as `primal_values`.
    ///   - `role`: Semantic boundary role that selects the tangent/cotangent mapping and identifies diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] if the primal value/type counts differ or if an operation family
    /// captures residual values whose count or types disagree with its declaration.
    pub fn capture<C: Context<Value = V, Type = V::Type, Operation: ResidualZeroProvider<C::Type>>>(
        context: &C,
        primal_values: &[C::Value],
        primal_types: &[C::Type],
        role: ZeroSpaceBoundaryRole,
    ) -> Result<Self, ProgramError> {
        if primal_values.len() != primal_types.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "{} has {} primal values but {} primal types",
                role,
                primal_values.len(),
                primal_types.len(),
            )));
        }
        let mut residuals = Vec::new();
        let mut zero_leaves = Vec::new();
        for (index, (value, primal_type)) in primal_values.iter().zip(primal_types).enumerate() {
            let differential_type = role.differential_type(primal_type)?;
            if differential_type.is_zero_space() {
                let residual_start = residuals.len();
                residuals.extend(capture_and_validate_zero_residual_values(
                    context,
                    value,
                    &differential_type,
                    role.as_str(),
                )?);
                zero_leaves.push(ZeroSpaceBoundaryLeaf {
                    index,
                    r#type: differential_type,
                    residual_range: residual_start..residuals.len(),
                });
            }
        }
        Ok(Self { role, residuals, boundary_size: primal_types.len(), zero_leaves })
    }

    /// Rebuilds the complete differential boundary described by this instance, interleaving materialized zero-space
    /// leaves with the compact derivative program's `live_values`. The boundary size, zero-leaf positions, differential
    /// types, and residual partitions were fixed and validated by [`Self::capture`]. Reconstruction consequently needs
    /// only the context in which to bind each zero and the live values produced by the compact derivative program.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which residual-backed zero operations are bound.
    ///   - `live_values`: Differential values for every nonzero-space boundary leaf, in boundary order.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if a zero cannot be materialized or if `live_values` does not contain exactly one
    /// value for every nonzero-space boundary leaf.
    pub fn rebuild<
        C: Context<Value = V, Type = V::Type, Operation: ResidualZeroProvider<C::Type>>,
        I: IntoIterator<Item = C::Value>,
    >(
        &self,
        context: &C,
        live_values: I,
    ) -> Result<Vec<C::Value>, ProgramError> {
        let mut live_values = live_values.into_iter();
        let mut zero_leaves = self.zero_leaves.iter().peekable();
        let mut values = Vec::with_capacity(self.boundary_size);
        for index in 0..self.boundary_size {
            if zero_leaves.peek().is_some_and(|leaf| leaf.index == index) {
                let zero_leaf = zero_leaves.next().unwrap();
                let residuals = self.residuals.get(zero_leaf.residual_range.clone()).unwrap();
                let (operation, operands) =
                    C::Operation::zero_operation_with_residuals(zero_leaf.r#type.clone(), residuals)?;
                let mut outputs = context.bind(operation, Vec::new(), operands.as_slice())?;
                check_count!("output", outputs, 1, ProgramError);
                values.push(outputs.remove(0));
            } else {
                values.push(live_values.next().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!("{} omitted a nonzero differential value", self.role))
                })?);
            }
        }
        if live_values.next().is_some() {
            return Err(ProgramError::MalformedProgram(format!(
                "{} produced too many nonzero differential values",
                self.role,
            )));
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionType, DimensionVariable, Shape,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::parameters::Placeholder;
    use crate::programs::{AtomId, MaybeZero, ProgramBuilder, ProgramError, Typed};
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_residual_zero_provider_input_free_defaults() {
        let r#type = ArrayType::scalar(DataType::F64);

        // An input-free operation family (reached through the blanket implementation) declares no residuals, and
        // both capture hooks record nothing: the builder-level hook stages no instructions and the value-level hook
        // returns no values.
        assert_eq!(ArrayOperation::<Array>::zero_residual_types(&r#type), Vec::<ArrayType>::new());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let source = builder.add_input(r#type.clone());
        assert_eq!(ArrayOperation::<Array>::capture_zero_residuals(&mut builder, source, &r#type), Ok(Vec::new()));
        assert!(builder.instructions().is_empty());
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            ArrayOperation::<Array>::capture_zero_residual_values(&context, &Array::scalar(3.0), &r#type),
            Ok(Vec::new()),
        );

        // Spending no residuals assembles the type-only zero, which the transposition path stages normally.
        let (operation, operands) =
            ArrayOperation::<Array>::zero_operation_with_residuals(r#type.clone(), &[] as &[AtomId]).unwrap();
        let zero = builder.add_instruction(operation, Vec::new(), operands, None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![zero], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(program.interpret(vec![Array::scalar(3.0)]), Ok(vec![Array::scalar(0.0)]));

        // The fail-loud default rejects unexpected residuals instead of ignoring them, so a mismatched
        // linearize/transpose pairing cannot be silently accepted.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let residual = builder.add_input(r#type.clone());
        assert_eq!(
            ArrayOperation::<Array>::zero_operation_with_residuals(r#type, &[residual]).map(|_| ()),
            Err(ProgramError::InvalidArgument {
                message: "input-free zero expected 0 residuals but got 1".to_string(),
            }),
        );
        assert!(builder.instructions().is_empty());
    }

    #[test]
    fn test_residual_zero_provider_captures_zero_residual_values() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type TestOperation = ArrayIrOperation<Array>;

        let first = DimensionVariable::new("first", DimensionBounds::positive(Some(8)).unwrap());
        let second = DimensionVariable::new("second", DimensionBounds::positive(Some(8)).unwrap());
        let primal_type = ArrayType::new(
            DataType::F8E8M0FNU,
            Shape::new(vec![
                Dimension::Dynamic(first.clone()),
                Dimension::Dynamic(first.clone()),
                Dimension::Dynamic(second.clone()),
            ]),
        );
        let tangent_type = ArrayIrType::Array(primal_type.tangent().unwrap());
        let context = TestContext::new();
        let primal = context.input(primal_type.into());

        // The default bulk value capture resolves each declared residual through the singular hook. Distinct
        // identities retain first-occurrence order, repeated identities share one residual, and the source may use a
        // different element representation than the zero being constructed.
        let residuals = TestOperation::capture_zero_residual_values(&context, &primal, &tangent_type).unwrap();
        assert_eq!(
            residuals.iter().map(|residual| residual.r#type().into_owned()).collect::<Vec<_>>(),
            vec![ArrayIrType::Dimension(DimensionType::new(first)), ArrayIrType::Dimension(DimensionType::new(second)),],
        );
        let builder = context.builder().borrow();
        let [first, second] = builder.instructions() else {
            panic!("expected two dimension-size instructions");
        };
        assert!(matches!(first.operation(), ArrayIrOperation::DimensionSize(operation) if operation.axis() == 0));
        assert!(matches!(second.operation(), ArrayIrOperation::DimensionSize(operation) if operation.axis() == 2));
        assert_eq!(residuals[0].atom_id(), Ok(first.outputs()[0]));
        assert_eq!(residuals[1].atom_id(), Ok(second.outputs()[0]));
    }

    #[test]
    fn test_residual_zero_provider_materializes_zero_from_residual_sources() {
        // The boundary form of the residual protocol assembles a zero's runtime geometry from the values in scope,
        // one named quantity at a time. This is what makes it type-general where exemplar matching was not: a widened
        // differential representation has no live value of its own type anywhere, and a scan's stacked cotangent
        // geometry is split across a first-class dimension operand and a per-iteration peer.

        type TestContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;
        type TestOperation = ArrayIrOperation<Array>;

        let length = DimensionVariable::new("length", DimensionBounds::positive(Some(8)).unwrap());
        let k = DimensionVariable::new("k", DimensionBounds::positive(Some(8)).unwrap());
        let context = TestContext::new();

        // A live value is returned unchanged and stages nothing.
        let live = context.input(ArrayType::scalar(DataType::F64).into());
        let materialized =
            TestOperation::materialize_zero_from_residual_sources(&context, MaybeZero::Value(live.clone()), &[])
                .unwrap();
        assert_eq!(materialized.atom_id().unwrap(), live.atom_id().unwrap());
        assert!(context.builder().borrow().instructions().is_empty());

        // An identity-free type declares no residuals and keeps the nullary zero, consulting no source.
        let static_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]));
        TestOperation::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(static_type.clone().into()),
            &[],
        )
        .unwrap();

        // A widened differential representation: the `f32` tangent of an `f8e8m0fnu[k]` primal has no live value of
        // its own type, yet the primal names the extent `k` and therefore supplies its geometry.
        let narrow_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Dynamic(k.clone())]));
        let narrow_primal = context.input(narrow_type.clone().into());
        let widened_tangent_type = narrow_type.tangent().unwrap();
        assert_eq!(widened_tangent_type.data_type(), DataType::F32);
        let widened = TestOperation::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(widened_tangent_type.clone().into()),
            std::slice::from_ref(&narrow_primal),
        )
        .unwrap();
        assert_eq!(widened.r#type().as_ref(), &ArrayIrType::Array(widened_tangent_type));

        // The scan stacked-output geometry: no peer has the `f64[length, k]` type, but the runtime length operand is a
        // first-class dimension that is reused directly and a per-iteration peer names `k` on its own axis `0`.
        let stacked_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(length.clone()), Dimension::Dynamic(k.clone())]),
        );
        let runtime_length = context.input(DimensionType::new(length.clone()).into());
        let peer = context.input(ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(k)])).into());
        let stacked = TestOperation::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(stacked_type.clone().into()),
            [&runtime_length, &peer],
        )
        .unwrap();
        assert_eq!(stacked.r#type().as_ref(), &ArrayIrType::Array(stacked_type.clone()));
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![stacked.atom_id().unwrap()],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %2:f8e8m0fnu[k], %5:dimension<length \u{2208} [1, 8)>, %6:f64[k] .
                let %1:f64[2] = zero [type=f64[2]]
                    %3:dimension<k \u{2208} [1, 8)> = dimension_size [axis=0] %2
                    %4:f32[k] = zero [type=f32[k]] %3
                    %7:dimension<k \u{2208} [1, 8)> = dimension_size [axis=0] %6
                    %8:f64[length, k] = zero [type=f64[length, k]] %5 %7
                in (%8)
            "}
            .trim_end(),
        );

        // A named quantity that no candidate carries is a loud diagnostic rather than a wrong-shaped zero.
        let error = TestOperation::materialize_zero_from_residual_sources(
            &context,
            MaybeZero::Zero(stacked_type.into()),
            std::slice::from_ref(&peer),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ProgramError::UnsupportedOperation {
                message: "cannot materialize a zero of type f64[length, k] because no value in scope supplies the \
                          runtime geometry dimension<length \u{2208} [1, 8)> that its residual 0 names"
                    .to_string(),
            },
        );
    }

    #[test]
    fn test_zero_space_boundary_reconstruction_reconstructs_dynamic_zero() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let key_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let accumulator_type = ArrayType::scalar(DataType::F64);
        let context = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let key = context.input(key_type.clone().into());
        let accumulator = context.input(accumulator_type.clone().into());
        let primal_values = vec![key, accumulator.clone()];
        let primal_types = vec![key_type.clone().into(), accumulator_type.into()];

        // Capture retains the key's dynamic extent and records that only the accumulator tangent remains live in the
        // compact output boundary. Rebuild must recover the omitted key tangent from that stored plan.
        let reconstruction = ZeroSpaceBoundaryReconstruction::capture(
            &context,
            primal_values.as_slice(),
            primal_types.as_slice(),
            ZeroSpaceBoundaryRole::OutputTangent,
        )
        .unwrap();
        let outputs = reconstruction.rebuild(&context, [accumulator.clone()]).unwrap();
        let key_tangent_type = ArrayIrType::Array(key_type.tangent().unwrap());
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].r#type().as_ref(), &key_tangent_type);
        assert_eq!(outputs[1].atom_id(), accumulator.atom_id());

        // The captured dimension-size result is the sole operand of the dynamic zero constructor, proving that the
        // stored residual range—not a type-only zero—is used during reconstruction.
        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 2);
        assert!(matches!(builder.instructions()[0].operation(), ArrayIrOperation::DimensionSize(_)));
        assert!(matches!(builder.instructions()[1].operation(), ArrayIrOperation::Zero(_)));
        assert_eq!(builder.instructions()[1].inputs(), builder.instructions()[0].outputs());
    }

    #[test]
    fn test_zero_space_boundary_reconstruction_reports_stored_boundary() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primal = Array::scalar(3.0);
        let primal_type = ArrayType::scalar(DataType::F64);

        // The output-tangent role retained during capture identifies a missing compact-program result without a
        // caller-supplied diagnostic context.
        let output_tangent = ZeroSpaceBoundaryReconstruction::capture(
            &context,
            std::slice::from_ref(&primal),
            std::slice::from_ref(&primal_type),
            ZeroSpaceBoundaryRole::OutputTangent,
        )
        .unwrap();
        assert_eq!(
            output_tangent.rebuild(&context, Vec::new()),
            Err(ProgramError::MalformedProgram(
                "output tangent boundary omitted a nonzero differential value".to_string(),
            )),
        );

        // The input-cotangent role independently identifies an excessive compact-program result.
        let input_cotangent = ZeroSpaceBoundaryReconstruction::capture(
            &context,
            std::slice::from_ref(&primal),
            std::slice::from_ref(&primal_type),
            ZeroSpaceBoundaryRole::InputCotangent,
        )
        .unwrap();
        assert_eq!(
            input_cotangent.rebuild(&context, [Array::scalar(1.0), Array::scalar(2.0)]),
            Err(ProgramError::MalformedProgram(
                "input cotangent boundary produced too many nonzero differential values".to_string(),
            )),
        );
    }
}
