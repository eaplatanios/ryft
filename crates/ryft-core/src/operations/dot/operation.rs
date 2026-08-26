use super::*;

/// Primitive representing a generalized dot (tensor contraction).
///
/// [`DotOperation`] is the unified primitive for matrix multiplication, batched matrix multiplication, vector inner
/// products, and arbitrary tensor contractions. It lowers to StableHLO's `dot_general` op in the XLA backend.
///
/// A dot is bilinear. Forward-mode differentiation applies
/// `d(dot(lhs, rhs)) = dot(dlhs, rhs) + dot(lhs, drhs)`. Transposition therefore accepts exactly one linear operand
/// and contracts the output cotangent with the other, known operand using the corresponding adjoint dimension
/// numbers. Accumulation-typed dots perform those contractions at the widened cotangent type before converting the
/// result back to the linear operand's cotangent representation.
///
/// Each forward-mode tangent term is staged as an ordinary dot that preserves the primal dimension numbers,
/// accumulation type, and requested output sharding without introducing captures. Transposition pins the adjoint
/// contraction's output sharding to the cotangent dual of the linear operand's sharding. A structural-zero output
/// cotangent remains structural zero.
///
/// Batching aligns the operands onto a common mapped axis and lifts the dimension numbers past it. Materializing an
/// axis on a replicated operand preserves a dynamic mapped extent as a first-class value. Two mapped operands must
/// describe the same mapped extent; for dynamic extents, they must share the same
/// [`DimensionVariable`](crate::arrays::DimensionVariable).
///
/// Every bounded ragged axis is either contracted or free. A contracted axis is zero-padded and consumed, and its
/// dimension variable is reported as [`BatchedOutputs`](crate::batching::BatchedOutputs) evidence so carrier validation
/// can distinguish deliberate consumption from a missing extent. Each operand is padded along only its own contracted
/// ragged axes because zeroing either factor removes the corresponding product. A free ragged axis propagates to the
/// result through the dot output layout: batching dimensions, then LHS free axes, then RHS free axes. A bounded ragged
/// axis on a batching dimension or a replicated operand is unsupported because no shared per-item extent identity
/// exists. Operands without bounded ragged axes use the dense path unchanged.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotOperation {
    /// Contracting and batching dimension specification.
    dimensions: DotDimensionNumbers,

    /// Optional accumulation data type. Refer to the documentation of [`Self::with_accumulation_type`].
    accumulation_type: Option<DataType>,

    /// Optional requested output [`Sharding`]. Refer to the documentation of [`Self::with_output_sharding`].
    output_sharding: Option<Sharding>,
}

impl DotOperation {
    /// Creates a new [`DotOperation`] with the supplied dimension numbers.
    #[inline]
    pub fn new(dimensions: DotDimensionNumbers) -> Self {
        Self { dimensions, accumulation_type: None, output_sharding: None }
    }

    /// Returns a [`DotOperation`] configured for standard rank-2 matrix multiplication.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(DotDimensionNumbers::matmul())
    }

    /// Attaches a requested output [`Sharding`] to this operation, mirroring the `out_sharding` parameter of JAX's
    /// `dot_general`. When set, type inference validates the requested sharding (rank, mesh, no auto axes, and the
    /// unreduced-output rule) and uses it for the output instead of the inferred sharding, bypassing the batch and
    /// contracting dimension consistency checks. This is the only way to produce an output with unreduced axes
    /// (i.e., per-device partial results whose cross-device reduction is delayed).
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns a copy of this [`DotOperation`] with the provided accumulation data type. The operand element types
    /// must still match each other and must promote to the accumulation type, which becomes the output element
    /// type: the backend upcasts the operands and accumulates the contraction at the wider type (XLA's
    /// `preferred_element_type` contract, which is what its low-precision matrix units implement natively — e.g.,
    /// `f8 × f8 → f32` and `bf16 × bf16 → f32`). Accumulation-typed dots differentiate like ordinary dots, with
    /// tangents and cotangents carried at the accumulation type (refer to the forward-mode and transpose rule
    /// documentation on this operation), and cannot yet be combined with a requested output sharding.
    #[inline]
    pub fn with_accumulation_type(mut self, accumulation_type: impl Into<Option<DataType>>) -> Self {
        self.accumulation_type = accumulation_type.into();
        self
    }

    /// Returns the optional accumulation data type. Refer to the documentation of
    /// [`Self::with_accumulation_type`].
    #[inline]
    pub fn accumulation_type(&self) -> Option<DataType> {
        self.accumulation_type
    }

    /// Returns the contracting and batching dimension specification.
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Display for DotOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for DotOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        DOT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        Ok(vec![dot_abstract(
            &input_types[0],
            &input_types[1],
            &self.dimensions,
            self.accumulation_type,
            self.output_sharding.as_ref(),
        )?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            if let Some(accumulation_type) = self.accumulation_type {
                operation.field("accumulation_type", &accumulation_type)?;
            }
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Dot>> InterpretableOperation<C> for DotOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        // The requested output sharding and accumulation type flow through the capability methods so that
        // interpretation over staging values (e.g., during program batching) preserves them; concrete values
        // ignore the sharding and upcast for the accumulation type. Type inference rejects combining the two.
        Ok(vec![match (&self.accumulation_type, &self.output_sharding) {
            (Some(accumulation_type), _) => {
                inputs[0].dot_with_accumulation_type(&inputs[1], &self.dimensions, *accumulation_type)
            }
            (None, Some(output_sharding)) => {
                inputs[0].dot_with_output_sharding(&inputs[1], &self.dimensions, output_sharding)
            }
            (None, None) => inputs[0].dot(&inputs[1], &self.dimensions),
        }])
    }
}

// Partial evaluation uses the default fold-or-residualize behavior.
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DotOperation where
    C::Operation: From<DotOperation>
{
}

/// Value-level generalized dot capability.
///
/// [`Dot`] is the receiver-style entry point for staging or executing [`DotOperation`]. It performs the contraction
/// described by `dimensions`, supporting standard matrix multiplication, batched matrix multiplication, vector inner
/// products, and arbitrary tensor contractions.
pub trait Dot<Rhs = Self>: Sized {
    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`.
    fn dot(&self, rhs: &Rhs, dimensions: &DotDimensionNumbers) -> Self;

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, requesting `output_sharding`
    /// for the result. The requested sharding overrides the inferred output sharding and is validated by the staged
    /// operation's type inference (refer to the documentation of [`DotOperation::with_output_sharding`]). The
    /// default implementation ignores the requested sharding and delegates to [`Self::dot`], which is correct for
    /// concrete (single-device) values, for which a sharding only describes distribution metadata; staging
    /// implementations override this method to attach the requested sharding to the staged operation.
    fn dot_with_output_sharding(
        &self,
        rhs: &Rhs,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        let _ = output_sharding;
        self.dot(rhs, dimensions)
    }

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, upcasting the operands to
    /// `accumulation_type` and accumulating the contraction there, so the result carries the accumulation type.
    /// Refer to the documentation of [`DotOperation::with_accumulation_type`] for the exact contract.
    fn dot_with_accumulation_type(
        &self,
        rhs: &Rhs,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Self;
}

// Context-carrying values stage a dot through their context. The `From<DotOperation>` bound keeps this implementation
// disjoint from eager values, whose context operation is `ConstantOperation`.
impl<V: Value<Type = ArrayType>> Dot for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<DotOperation>,
{
    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Self {
        self.dispatch_domain()
            .bind(DotOperation::new(dimensions.clone()), Vec::new(), &[self.clone(), rhs.clone()])
            .expect("`dot` operation failed")
            .remove(0)
    }

    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Self {
        self.dispatch_domain()
            .bind(
                DotOperation::new(dimensions.clone()).with_accumulation_type(accumulation_type),
                Vec::new(),
                &[self.clone(), rhs.clone()],
            )
            .expect("`dot` operation failed")
            .remove(0)
    }

    fn dot_with_output_sharding(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        self.dispatch_domain()
            .bind(
                DotOperation::new(dimensions.clone()).with_output_sharding(output_sharding.clone()),
                Vec::new(),
                &[self.clone(), rhs.clone()],
            )
            .expect("`dot` operation failed")
            .remove(0)
    }
}

/// Primitive representing a grouped generalized dot with explicit group sizes.
///
/// Exactly one LHS dimension is ragged. Its role selects one of three modes:
///
///   - A non-contracting ragged dimension is partitioned into consecutive groups. The corresponding RHS group
///     dimension selects one RHS slice per group, and the grouped products are written back along the LHS result
///     dimension. A zero-size group contributes nothing and any uncovered suffix of that dimension is zero.
///   - A contracting ragged dimension partitions the paired contracting dimensions into consecutive groups. The
///     output gains a leading group dimension, and a zero-size group produces a zero slice.
///   - A batching ragged dimension has ordinary batched-dot semantics. `group_sizes` participates in type inference
///     but its values do not affect the result.
///
/// `group_sizes` is either a rank-one `[group_count]` array shared by every prefix or an array whose trailing axis is
/// `group_count` and whose prefix matches the dimensions preceding the ragged position in the grouped-dot iteration
/// space.
///
/// In non-contracting and contracting modes every size must be nonnegative. The eager interpreter rejects negative
/// metadata in those modes. XLA's decomposition lowering additionally clamps signed negatives to zero before unsigned
/// accumulation so invalid runtime metadata cannot become a large interval. The instruction lowering passes metadata
/// unchanged to `chlo.ragged_dot` and therefore relies on the nonnegative-input contract.
///
/// The sizes define consecutive raw cumulative intervals. Each interval is intersected with the physical LHS ragged
/// extent, so an over-covering group is clipped and every later group is empty once its raw start reaches or exceeds
/// that extent.
///
/// Grouped expansion modes require an element type that can represent zero; in particular, they reject `f8e8m0fnu`.
/// Refer to [`RaggedDotDimensionNumbers`] for the dimension-number contract.
///
/// The operation is linear in either data operand separately, while `group_sizes` is nondifferentiable metadata.
/// Forward-mode differentiation applies the two-term product rule with the same group metadata. Transposition is
/// defined only in non-contracting mode and applies another grouped dot followed by the inverse adjoint-axis
/// permutation.
///
/// Batching accepts either three replicated operands or three operands mapped over leading axis zero. It rejects
/// [`RaggedAxis`](crate::arrays::RaggedAxis) metadata because `group_sizes` is the sole source of ragged extents for
/// this operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedDotOperation {
    /// Grouped-dot dimension-number specification.
    dimensions: RaggedDotDimensionNumbers,
}

impl RaggedDotOperation {
    /// Creates a grouped generalized dot.
    #[inline]
    pub fn new(dimensions: RaggedDotDimensionNumbers) -> Self {
        Self { dimensions }
    }

    /// Returns the grouped-dot dimension-number specification.
    #[inline]
    pub fn dimensions(&self) -> &RaggedDotDimensionNumbers {
        &self.dimensions
    }
}

impl Display for RaggedDotOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for RaggedDotOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        RAGGED_DOT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 3, TypeError);
        Ok(vec![ragged_dot_abstract(&input_types[0], &input_types[1], &input_types[2], &self.dimensions)?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: RaggedDot>> InterpretableOperation<C> for RaggedDotOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 3, ProgramError);
        Ok(vec![inputs[0].ragged_dot_general(&inputs[1], &inputs[2], &self.dimensions)?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for RaggedDotOperation where
    C::Operation: From<RaggedDotOperation>
{
}

/// Value-level grouped generalized dot capability.
pub trait RaggedDot: Sized {
    /// Computes a grouped generalized dot using explicit `group_sizes`. Refer to [`RaggedDotOperation`] for the three
    /// modes, metadata shapes, cumulative-interval clipping, and zero-group and uncovered-position semantics.
    fn ragged_dot_general(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError>;

    /// Computes the basic non-contracting form `[M, K] × [G, K, N] → [M, N]`. Refer to [`RaggedDotOperation`] for
    /// zero-size-group and uncovered-row behavior.
    #[inline]
    fn ragged_dot(&self, rhs: &Self, group_sizes: &Self) -> Result<Self, ProgramError> {
        self.ragged_dot_general(rhs, group_sizes, &RaggedDotDimensionNumbers::matmul())
    }
}

impl<V: Value<Type = ArrayType>> RaggedDot for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<RaggedDotOperation>,
{
    fn ragged_dot_general(
        &self,
        rhs: &Self,
        group_sizes: &Self,
        dimensions: &RaggedDotDimensionNumbers,
    ) -> Result<Self, ProgramError> {
        Ok(self
            .dispatch_domain()
            .bind(
                RaggedDotOperation::new(dimensions.clone()),
                Vec::new(),
                &[self.clone(), rhs.clone(), group_sizes.clone()],
            )?
            .remove(0))
    }
}

/// Combined generalized dot product and transposition capability.
///
/// This convenience trait groups the value-level [`Dot`] and [`Transpose`] operations used by the unified
/// [`DotOperation`] and [`TransposeOperation`](crate::operations::manipulation::TransposeOperation) primitives.
pub trait DotOps: Dot + Transpose {}

impl<T: Dot + Transpose> DotOps for T {}
