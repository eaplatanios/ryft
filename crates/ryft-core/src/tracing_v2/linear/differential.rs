use std::fmt::Debug;
use std::marker::PhantomData;

use ryft_macros::Parameter;

use crate::batching::ArrayBatch;
use crate::batching::BatchableOperation;
use crate::batching::BatchingContext;
use crate::batching::BatchingError;
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiationError, TransposableOperation};
use crate::interpretation::InterpretableOperation;
use crate::operations::BooleanLike;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::{
    FillOperation, OneLike, OneOperation, Zero, ZeroLike, ZeroLikeOperation, ZeroOperation,
};
use crate::operations::control_flow::MaybeWhile;
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{DomainTracingContext, NestedTracingContext, Tracer, TracingContext};
use crate::tracing_v2::differentiation::{DifferentiableOperation, Linearization, replay_via_bind};
use crate::tracing_v2::unroll::unroll_concretizable_whiles;
use crate::tracing_v2::{Differentiate, NestedTracer};
use crate::types::{ArrayType, Size, TypeError, Typed};

/// Leaf type that can be materialized into a dense finite-dimensional coordinate representation.
///
/// Structured differential materialization only makes sense for leaf types with a finite, explicit
/// basis. [`CoordinateValue`] is the bridge from the generic tracing world into that coordinate-based
/// view: it teaches the differential helpers how many coordinates a leaf contributes, what basis
/// vectors to probe with, and how to flatten outputs back into numeric entries.
pub trait CoordinateValue: Value<Type = ArrayType> + ZeroLike + OneLike {
    /// Scalar-like coordinate type used by [`DifferentialBlock`] entries.
    type Coordinate: Clone + Debug + PartialEq + 'static;

    /// Returns the number of coordinates contributed by this leaf.
    fn coordinate_count(&self) -> usize;

    /// Returns a standard basis for the coordinate space of this leaf.
    fn coordinate_basis(&self) -> Vec<Self>;

    /// Flattens the leaf into its coordinate values in a deterministic order.
    fn coordinates(&self) -> Vec<Self::Coordinate>;

    /// Stacks the provided values along a new leading batch axis. All values must share the same
    /// [`ArrayType`]; the resulting value carries that type prefixed with `Size::Static(values.len())`
    /// on axis `0`. Used by `Differential::from_linearization` and [`jacrev`] to pack `N`
    /// per-basis-tangent values into one batched input that flows through the value-level
    /// [`BatchableOperation::batch`] rule for tangent values.
    fn stack(values: Vec<Self>) -> Result<Self, ProgramError>;
}

/// Structured forward- or reverse-mode Jacobian of a function `Input -> Output` over leaf value
/// type `V`. Materialized by [`DifferentiableDomainExtension::jacfwd`] and [`jacrev`].
///
/// The outer [`Parameterized`] family mirrors the function's output; each output-leaf position
/// holds a [`DifferentialRow`] whose internal family mirrors the function's input and whose leaves
/// are [`DifferentialBlock`]s of partial derivatives. Block entries are stored as `V::Coordinate`
/// scalars.
pub type Jacobian<Input, Output, V> = Differential<
    <Output as Parameterized<V>>::To<
        DifferentialRow<
            <Input as Parameterized<V>>::To<DifferentialBlock<<V as CoordinateValue>::Coordinate>>,
            <V as CoordinateValue>::Coordinate,
        >,
    >,
    <Input as Parameterized<V>>::To<DifferentialBlock<<V as CoordinateValue>::Coordinate>>,
    <V as CoordinateValue>::Coordinate,
>;

/// Structured Hessian of a scalar-output function over a [`Parameterized`] input with leaf value
/// type `V`. Materialized by [`DifferentiableDomainExtension::hessian`].
///
/// Equivalent to a [`Jacobian<Input, Input, V>`] - both the outer and inner [`Parameterized`]
/// families mirror the input.
pub type Hessian<Input, V> = Jacobian<Input, Input, V>;

/// Concrete value type selected by a [`Domain`].
type DomainValue<D> = <D as Domain>::Value;

/// Scalar coordinate type used to materialize dense differential blocks for `V`.
type CoordinateScalar<V> = <V as CoordinateValue>::Coordinate;

/// Dense derivative materialization helpers for differentiable array domains.
///
/// This extension trait keeps the core [`Differentiate`] contract focused on primitive
/// linearization and AD transforms while providing structured Jacobian and Hessian materialization
/// for domains whose values expose finite coordinate bases.
pub trait DifferentiableDomainExtension: Context<Type = ArrayType> {
    /// Materializes a structured [`Jacobian`] using forward-mode differentiation.
    ///
    /// The returned [`Jacobian`] is a nested [`Parameterized`] value whose outer family mirrors
    /// the function's output and whose inner family mirrors its input. Each innermost leaf is a
    /// [`DifferentialBlock`] holding the partial derivatives of one output leaf with respect to
    /// one input leaf.
    #[allow(private_bounds)]
    fn jacfwd<'domain, F, Input, TracedOutput>(
        &'domain self,
        function: F,
        primal: Input,
    ) -> Result<Jacobian<Input, TracedOutput::To<DomainValue<Self>>, DomainValue<Self>>, ProgramError>
    where
        DomainValue<Self>: CoordinateValue + BooleanLike + 'domain,
        Input: Parameterized<
                DomainValue<Self>,
                To<DomainValue<Self>> = Input,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<Tracer<NestedTracingContext<Self>>, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<Tracer<NestedTracingContext<Self>>>
            + ParameterizedFamily<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>,
        TracedOutput::Family: ParameterizedFamily<DomainValue<Self>>
            + ParameterizedFamily<Tracer<NestedTracingContext<Self>>>
            + ParameterizedFamily<
                DifferentialRow<
                    Input::To<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>,
                    CoordinateScalar<DomainValue<Self>>,
                >,
            >,
        TracedOutput::To<DomainValue<Self>>: Parameterized<
                DomainValue<Self>,
                To<DomainValue<Self>> = TracedOutput::To<DomainValue<Self>>,
                To<
                    DifferentialRow<
                        Input::To<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>,
                        CoordinateScalar<DomainValue<Self>>,
                    >,
                > = TracedOutput::To<
                    DifferentialRow<
                        Input::To<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>,
                        CoordinateScalar<DomainValue<Self>>,
                    >,
                >,
        >,
        F: FnOnce(Input::To<Tracer<NestedTracingContext<Self>>>) -> Result<TracedOutput, ProgramError>,
        <Self as Domain>::Operation: Clone
            + InterpretableOperation<DomainValue<Self>, Self>
            + MaybeWhile<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + From<ZeroOperation<ArrayType>>
            + DifferentiableOperation<
                TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>,
            >
            + PartiallyEvaluatableOperation<
                TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>,
            >
            + BatchableOperation<DomainValue<Self>, BatchingContext<Self>>,
    {
        let input_structure = primal.parameter_structure();
        let input_parameters = primal.into_parameters().collect::<Vec<_>>();
        let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;

        // Forward-mode Jacobian over the capture-free front end: trace the function into a primal program and
        // (for eager domains) unroll any concretizable `while` loop at the primal point before fusing, mirroring
        // `Differentiate::jvp`. `Program::linearize` then splits the JVP program into the primal sub-program
        // (primal outputs followed by residuals) and the linear tangent sub-program. `from_linearization` replays
        // every input-coordinate basis tangent through the tangent sub-program in one batched pass — broadcasting the
        // primal-derived residuals as replicated values — preserving the exact Jacobian layout.
        let (program, _input_structure, output_structure, input_values) =
            self.trace_into_primal_program::<F, Input, TracedOutput>(function, primals)?;
        let program = unroll_concretizable_whiles(self, program, input_values.clone())?;
        let linearization = program.linearize()?;

        // Recover the structured primal output by replaying the primal sub-program at the primals and dropping its
        // trailing residuals, leaving the leading primal outputs which `from_linearization` consumes for output
        // shapes and structure. The replay goes through this context itself — lifting the sub-program's staged
        // constants into value space and binding each instruction — so an eager backend domain executes it with its
        // own value semantics rather than through a fresh constant-only `EagerContext`.
        let mut primal_side = replay_via_bind(self, &linearization.primal_program, input_values)?;
        let primal_output_count = primal_side.len().checked_sub(linearization.residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "primal program produced {} outputs which is fewer than its {} residuals",
                primal_side.len(),
                linearization.residual_count,
            ))
        })?;
        primal_side.truncate(primal_output_count);
        let output = TracedOutput::To::<DomainValue<Self>>::from_parameters(output_structure, primal_side)?;

        Differential::from_linearization::<Self, Input, TracedOutput::To<DomainValue<Self>>, DomainValue<Self>>(
            self,
            input_structure,
            input_parameters,
            output,
            linearization,
        )
    }

    /// Materializes a structured [`Hessian`] of a scalar-output function.
    ///
    /// Hessian evaluation is expressed internally as a forward-mode [`Jacobian`] over a
    /// reverse-mode gradient transform.
    #[allow(private_bounds)]
    fn hessian<'domain, F, Input>(
        &'domain self,
        function: F,
        primals: Input,
    ) -> Result<Hessian<Input, DomainValue<Self>>, DifferentiationError>
    where
        DomainValue<Self>: CoordinateValue + 'domain,
        Input: Parameterized<DomainValue<Self>, To<DomainValue<Self>> = Input, ParameterStructure: Debug + PartialEq>,
        Input::Family: ParameterizedFamily<Tracer<NestedTracingContext<DomainTracingContext<Self>>>>
            + ParameterizedFamily<Tracer<DomainTracingContext<Self>>>
            + ParameterizedFamily<<Self as Domain>::Constant>
            + ParameterizedFamily<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>
            + ParameterizedFamily<
                DifferentialRow<
                    Input::To<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>,
                    CoordinateScalar<DomainValue<Self>>,
                >,
            >,
        Input::To<Tracer<DomainTracingContext<Self>>>: Parameterized<
                Tracer<DomainTracingContext<Self>>,
                To<Tracer<DomainTracingContext<Self>>> = Input::To<Tracer<DomainTracingContext<Self>>>,
                To<DomainValue<Self>> = Input,
                To<<Self as Domain>::Constant> = Input::To<<Self as Domain>::Constant>,
                To<Tracer<NestedTracingContext<DomainTracingContext<Self>>>> = Input::To<
                    Tracer<NestedTracingContext<DomainTracingContext<Self>>>,
                >,
                ParameterStructure: Debug + PartialEq,
            >,
        F: FnOnce(
            Input::To<Tracer<NestedTracingContext<DomainTracingContext<Self>>>>,
        ) -> Tracer<NestedTracingContext<DomainTracingContext<Self>>>,
        <Self as Domain>::Operation: Clone
            + InterpretableOperation<DomainValue<Self>, Self>
            + InterpretableOperation<Tracer<DomainTracingContext<Self>>, DomainTracingContext<Self>>
            + TransposableOperation<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + MaybeWhile<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + DifferentiableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>
            + PartiallyEvaluatableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>
            + BatchableOperation<DomainValue<Self>, BatchingContext<Self>>
            + From<FillOperation<ArrayType, f64>>
            + From<ZeroOperation<ArrayType>>
            + From<OneOperation<ArrayType>>
            + From<ZeroLikeOperation>
            + From<AddOperation>,
        <Self as Domain>::Constant: Value<Type = ArrayType>,
        Tracer<DomainTracingContext<Self>>: BooleanLike,
    {
        let input_structure = primals.parameter_structure();
        let input_parameters = primals.into_parameters().collect::<Vec<_>>();
        let primals = Input::from_parameters(input_structure.clone(), input_parameters.iter().cloned())
            .map_err(ProgramError::from)?;
        let (gradient, gradient_program): (
            Input,
            Program<
                <Self as Domain>::Constant,
                <Self as Domain>::Operation,
                Input::To<<Self as Domain>::Constant>,
                Input::To<<Self as Domain>::Constant>,
            >,
        ) = self.interpret_and_trace(
            |input: Input::To<Tracer<DomainTracingContext<Self>>>| {
                let Some(context) = input.parameters().next().map(|tracer| tracer.context().clone()) else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 });
                };
                // `interpret_and_trace` fixes its closure error to `ProgramError`, so fold the inner gradient's
                // differentiation error into a program error to flow it through the trace. A non-scalar gradient
                // output cannot occur here for well-formed second-order use (the differentiated function is scalar).
                crate::tracing_v2::Differentiate::gradient(&context, function, input).map_err(|error| match error {
                    DifferentiationError::Program(error) => error,
                    error => ProgramError::MalformedProgram(error.to_string()),
                })
            },
            primals,
        )?;
        // Linearize the gradient program through the capture-free front end, then replay every input-coordinate
        // basis tangent through its linear tangent sub-program. The already-evaluated `gradient` supplies the output
        // shapes and structure `from_linearization` needs; its primal replay recovers the residuals internally.
        let linearization = gradient_program.linearize().map_err(DifferentiationError::from)?;
        Differential::from_linearization::<Self, Input, Input, DomainValue<Self>>(
            self,
            input_structure,
            input_parameters,
            gradient,
            linearization,
        )
        .map_err(DifferentiationError::from)
    }
}

impl<D> DifferentiableDomainExtension for D where D: Context<Type = ArrayType> {}

/// Partial derivatives of one output leaf with respect to one input leaf.
///
/// For an output leaf of shape `O` and an input leaf of shape `I`, a `DifferentialBlock` carries
/// the `O ++ I`-shaped partial-derivative tensor stored in row-major order, with the output
/// dimensions varying slowest. The two shapes are reported alongside the values so callers can
/// index into the tensor without consulting any external metadata.
#[derive(Parameter, Clone, Debug)]
pub struct DifferentialBlock<S> {
    /// Shape of the output leaf this block contributes to.
    output_shape: Vec<usize>,

    /// Shape of the input leaf this block differentiates with respect to.
    input_shape: Vec<usize>,

    /// Row-major partial-derivative values; length is `prod(output_shape) * prod(input_shape)`.
    values: Vec<S>,
}

impl<S> DifferentialBlock<S> {
    /// Constructs a [`DifferentialBlock`] from explicit shapes and row-major values.
    ///
    /// # Parameters
    ///
    ///   - `output_shape`: Shape of the output leaf this block contributes to.
    ///   - `input_shape`: Shape of the input leaf this block differentiates with respect to.
    ///   - `values`: Row-major partial-derivative values with the output dimensions varying slowest.
    ///     Length must equal `prod(output_shape) * prod(input_shape)`.
    pub fn new(output_shape: Vec<usize>, input_shape: Vec<usize>, values: Vec<S>) -> Self {
        debug_assert_eq!(
            values.len(),
            output_shape.iter().product::<usize>() * input_shape.iter().product::<usize>(),
            "differential block value count must equal the product of the output and input shape sizes",
        );
        Self { output_shape, input_shape, values }
    }

    /// Returns the shape of the output leaf this block contributes to.
    #[inline]
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Returns the shape of the input leaf this block differentiates with respect to.
    #[inline]
    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    /// Returns the row-major partial-derivative values stored in this block.
    #[inline]
    pub fn values(&self) -> &[S] {
        &self.values
    }

    /// Returns the partial derivative at `(output_index, input_index)` if both lie within the
    /// block's output and input shapes respectively.
    ///
    /// # Parameters
    ///
    ///   - `output_index`: Multidimensional index into the output leaf. Must have the same length
    ///     as [`Self::output_shape`].
    ///   - `input_index`: Multidimensional index into the input leaf. Must have the same length
    ///     as [`Self::input_shape`].
    pub fn get(&self, output_index: &[usize], input_index: &[usize]) -> Option<&S> {
        let output_offset = flat_offset(&self.output_shape, output_index)?;
        let input_offset = flat_offset(&self.input_shape, input_index)?;
        let input_size = self.input_shape.iter().product::<usize>();
        Some(&self.values[output_offset * input_size + input_offset])
    }
}

/// One row of a [`Differential`]: partial derivatives of one output leaf with respect to every
/// input leaf.
///
/// `Partials` is the input `Parameterized` value already reparameterized to [`DifferentialBlock`]
/// leaves (typically `Input::To<DifferentialBlock<S>>` at the call site). Carries [`Parameter`] —
/// **not** [`Parameterized`] — so it can appear as a leaf inside the outer `Parameterized` value
/// held by [`Differential`]. Internal structure is accessed via [`Self::partials`].
#[derive(Parameter, Clone, Debug)]
pub struct DifferentialRow<Partials, S>
where
    Partials: Parameterized<DifferentialBlock<S>>,
{
    /// Input-shaped [`Parameterized`] value whose leaves are the partial-derivative blocks for the
    /// output leaf this row corresponds to.
    partials: Partials,

    /// Marker keeping the scalar coordinate type fixed at the type level.
    _scalar: PhantomData<S>,
}

impl<Partials, S> DifferentialRow<Partials, S>
where
    Partials: Parameterized<DifferentialBlock<S>>,
{
    /// Constructs a [`DifferentialRow`] from an input-shaped [`Parameterized`] value of
    /// [`DifferentialBlock`]s.
    #[inline]
    pub fn new(partials: Partials) -> Self {
        Self { partials, _scalar: PhantomData }
    }

    /// Returns the input-shaped [`Parameterized`] value backing this row.
    #[inline]
    pub fn partials(&self) -> &Partials {
        &self.partials
    }

    /// Consumes this row and returns its underlying input-shaped [`Parameterized`] value.
    #[inline]
    pub fn into_partials(self) -> Partials {
        self.partials
    }

    /// Returns an iterator over the [`DifferentialBlock`]s in this row together with their
    /// [`ParameterPath`]s into the input structure.
    #[inline]
    pub fn iter_partials(
        &self,
    ) -> <Partials as Parameterized<DifferentialBlock<S>>>::NamedParameterIterator<'_, DifferentialBlock<S>> {
        self.partials.named_parameters()
    }
}

/// Structured differential of a function `Input -> Output`.
///
/// Returned by both forward- and reverse-mode Jacobian transforms and by the Hessian transform
/// (where `Output = Input` and `Rows = Partials`). The outer `Parameterized` family mirrors the
/// output (or the input, for a Hessian); each output-leaf position holds a [`DifferentialRow`]
/// whose internal `Parameterized` family mirrors the input and whose leaves are
/// [`DifferentialBlock`]s. Callers traverse with a two-level iteration:
///
/// ```ignore
/// for (output_path, row) in differential.iter_rows() {
///     for (input_path, block) in row.iter_partials() {
///         // (output_path, input_path) -> block
///     }
/// }
/// ```
#[derive(Clone, Debug)]
pub struct Differential<Rows, Partials, S>
where
    Rows: Parameterized<DifferentialRow<Partials, S>>,
    Partials: Parameterized<DifferentialBlock<S>>,
{
    /// Output-shaped [`Parameterized`] value whose leaves are the [`DifferentialRow`]s for each
    /// output leaf.
    rows: Rows,

    /// Marker keeping the inner partials and scalar types fixed at the type level.
    _phantom: PhantomData<(Partials, S)>,
}

impl<Rows, Partials, S> Differential<Rows, Partials, S>
where
    Rows: Parameterized<DifferentialRow<Partials, S>>,
    Partials: Parameterized<DifferentialBlock<S>>,
{
    /// Constructs a [`Differential`] from an output-shaped [`Parameterized`] value of
    /// [`DifferentialRow`]s.
    #[inline]
    pub fn new(rows: Rows) -> Self {
        Self { rows, _phantom: PhantomData }
    }

    /// Returns the output-shaped [`Parameterized`] value backing this differential.
    #[inline]
    pub fn rows(&self) -> &Rows {
        &self.rows
    }

    /// Consumes this differential and returns its underlying output-shaped [`Parameterized`] value.
    #[inline]
    pub fn into_rows(self) -> Rows {
        self.rows
    }

    /// Returns an iterator over the [`DifferentialRow`]s of this differential together with their
    /// [`ParameterPath`]s into the output structure.
    #[inline]
    pub fn iter_rows(
        &self,
    ) -> <Rows as Parameterized<DifferentialRow<Partials, S>>>::NamedParameterIterator<'_, DifferentialRow<Partials, S>>
    {
        self.rows.named_parameters()
    }

    /// Constructs a [`Differential`] from flat per-item output-coordinate columns.
    ///
    /// Each entry in `columns` is the full flattened output differential produced by one input-coordinate basis item.
    /// This helper only handles the structure rearrangement into output rows and input blocks; callers decide how those
    /// columns are produced.
    fn from_coordinate_columns<Input, Output, V>(
        input_structure: Input::ParameterStructure,
        input_parameters: &[V],
        output_structure: Output::ParameterStructure,
        output_parameters: &[V],
        columns: Vec<Vec<S>>,
    ) -> Result<Self, ProgramError>
    where
        S: Clone,
        V: CoordinateValue<Coordinate = S>,
        Input:
            Parameterized<V, To<V> = Input, To<DifferentialBlock<S>> = Partials, ParameterStructure: Debug + PartialEq>,
        Output:
            Parameterized<V, To<V> = Output, To<DifferentialRow<Partials, S>> = Rows, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<DifferentialBlock<S>>,
        Output::Family: ParameterizedFamily<DifferentialRow<Partials, S>>,
        Partials: Parameterized<DifferentialBlock<S>, ParameterStructure = Input::ParameterStructure>,
        Rows: Parameterized<DifferentialRow<Partials, S>, ParameterStructure = Output::ParameterStructure>,
    {
        let input_shapes = input_parameters
            .iter()
            .map(|parameter| static_shape(parameter.r#type().as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let input_coordinate_counts = coordinate_counts(input_parameters);
        let input_offsets = coordinate_offsets(&input_coordinate_counts);

        let output_shapes = output_parameters
            .iter()
            .map(|parameter| static_shape(parameter.r#type().as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let output_coordinate_counts = coordinate_counts(output_parameters);
        let output_offsets = coordinate_offsets(&output_coordinate_counts);

        let mut rows_list = Vec::with_capacity(output_coordinate_counts.len());
        for (output_leaf_index, &output_count) in output_coordinate_counts.iter().enumerate() {
            let output_offset = output_offsets[output_leaf_index];
            let output_shape = &output_shapes[output_leaf_index];

            let mut blocks = Vec::with_capacity(input_coordinate_counts.len());
            for (input_leaf_index, &input_count) in input_coordinate_counts.iter().enumerate() {
                let input_offset = input_offsets[input_leaf_index];
                let input_shape = &input_shapes[input_leaf_index];

                let mut values = Vec::with_capacity(output_count * input_count);
                for output_local in 0..output_count {
                    for input_local in 0..input_count {
                        let item = input_offset + input_local;
                        let coordinate = output_offset + output_local;
                        values.push(columns[item][coordinate].clone());
                    }
                }

                blocks.push(DifferentialBlock::new(output_shape.clone(), input_shape.clone(), values));
            }

            let partials = Partials::from_parameters(input_structure.clone(), blocks)?;
            rows_list.push(DifferentialRow::new(partials));
        }

        let rows = Rows::from_parameters(output_structure, rows_list)?;
        Ok(Self::new(rows))
    }

    /// Constructs a [`Differential`] from a capture-free [`Linearization`] by replaying its primal sub-program
    /// once at the primal point and batch-replaying its tangent sub-program across every input-coordinate basis
    /// tangent.
    ///
    /// This is the forward analogue of [`Self::from_pushforward`] over the front end. Where `from_pushforward`
    /// instantiates a residualized pushforward into one linear program over `C::LinearOperation` and batch-replays it,
    /// a [`Linearization`] carries two sub-programs over the *primal* operation family `C::Operation`: a known
    /// `primal_program` taking the primal inputs and producing `[primal_outputs..., residuals...]`, and an unknown
    /// `tangent_program` taking `[tangents..., residuals...]` and producing the tangent outputs. This helper:
    ///
    ///   1. Replays `primal_program` once at the concrete primal `input_parameters` through `context` itself —
    ///      lifting the sub-program's staged constants with [`Context::lift`](crate::Context::lift) and binding each
    ///      instruction — and splits its outputs at [`residual_count`](Linearization::residual_count) into the primal
    ///      outputs and the concrete residuals.
    ///   2. Batch-replays `tangent_program` across the stacked input-coordinate basis tangents (all basis items on
    ///      axis 0, exactly as `from_pushforward` stacks them), appending the residuals as replicated
    ///      [`ArrayBatch::replicated`] operands after the batched basis tangents — the same replicated mechanism
    ///      `from_pushforward` uses for closed constants and [`jacrev`] uses for its reverse-mode residuals. Staged
    ///      constants are lifted through `context` and broadcast as replicated operands the same way. Because the
    ///      tangent sub-program is expressed in the primal operation family, each instruction is lifted through its
    ///      primal-family [`BatchableOperation`] rule by [`batch_linear_program_instruction`], interpreting nullary
    ///      instructions through `context`.
    ///   3. Assembles the per-(output-leaf, input-leaf) [`DifferentialBlock`]s from the resulting per-output coordinate
    ///      columns exactly as [`Self::from_pushforward`] does.
    ///
    /// Tangents are ordinary [`Value`](DispatchDomain::Value)s of the same universe as the primals, so the concrete
    /// residuals recovered from the primal replay are tangent-typed and feed the tangent batch directly with no
    /// tangent-context bridging.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context whose [`lift`](crate::Context::lift) and [`bind`](crate::Context::bind) replay the
    ///     primal sub-program and supply the value semantics for the batched tangent replay.
    ///   - `input_structure`: Placeholder shape of the function's `Input` argument.
    ///   - `input_parameters`: Concrete leaf values of `Input` at the point of linearization, used both to derive each
    ///     input leaf's static shape and to replay the primal sub-program.
    ///   - `output`: Primal output of the linearized function, consumed to recover its placeholder shape and the static
    ///     shapes of its output leaves.
    ///   - `linearization`: Capture-free linearization whose primal and tangent sub-programs are replayed.
    pub(crate) fn from_linearization<C, Input, Output, V>(
        context: &C,
        input_structure: Input::ParameterStructure,
        input_parameters: Vec<V>,
        output: Output,
        linearization: Linearization<<C as Domain>::Constant, <C as Domain>::Operation>,
    ) -> Result<Self, ProgramError>
    where
        S: Clone,
        C: Context<Type = ArrayType, Value = V>,
        V: CoordinateValue<Coordinate = S>,
        Input:
            Parameterized<V, To<V> = Input, To<DifferentialBlock<S>> = Partials, ParameterStructure: Debug + PartialEq>,
        Output:
            Parameterized<V, To<V> = Output, To<DifferentialRow<Partials, S>> = Rows, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<DifferentialBlock<S>>,
        Output::Family: ParameterizedFamily<DifferentialRow<Partials, S>>,
        Partials: Parameterized<DifferentialBlock<S>, ParameterStructure = Input::ParameterStructure>,
        Rows: Parameterized<DifferentialRow<Partials, S>, ParameterStructure = Output::ParameterStructure>,
        <C as Domain>::Operation: Clone + InterpretableOperation<V, C> + BatchableOperation<V, BatchingContext<C>>,
    {
        let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
        let batch_size: usize = input_coordinate_counts.iter().sum();
        let batched_basis_parameters = batched_standard_basis::<V>(input_parameters.as_slice(), batch_size)?;

        let output_structure = output.parameter_structure();
        let output_parameters = output.into_parameters().collect::<Vec<_>>();

        // Replay the primal sub-program once at the concrete primals to recover `[primal_outputs..., residuals...]`,
        // then split off the residual tail. The residuals depend only on the primal point and so are identical across
        // every basis item.
        let mut primal_side = replay_via_bind(context, &linearization.primal_program, input_parameters.clone())?;
        let residuals =
            primal_side.split_off(primal_side.len().checked_sub(linearization.residual_count).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "primal program produced {} outputs which is fewer than its {} residuals",
                    primal_side.len() + linearization.residual_count,
                    linearization.residual_count,
                ))
            })?);

        let columns = if batch_size == 0 {
            Vec::new()
        } else {
            // Feed the tangent sub-program `[batched_basis_tangents..., unbatched_residuals...]`: the basis tangents
            // carry one item per input coordinate on axis 0, and the residuals are broadcast as replicated values.
            // Each instruction lifts through its batching dispatcher at an anonymous one-level `BatchingContext` over
            // this context, so primitive work executes through the context itself.
            let batching_context = BatchingContext::new(context.clone(), batch_size, None);
            let mut tangent_inputs = batched_basis_parameters;
            tangent_inputs.extend(residuals.into_iter().map(ArrayBatch::replicated));
            let batched_output = linearization.tangent_program.interpret_with(
                tangent_inputs,
                |_, constant| Ok::<_, BatchingError>(ArrayBatch::replicated(context.lift(constant.clone())?)),
                |instruction, inputs: &[ArrayBatch<V>]| {
                    batch_linear_program_instruction(&batching_context, instruction.operation(), inputs)
                },
            )?;
            unstack_batched_coordinates::<V>(batched_output, output_parameters.as_slice(), batch_size)?
        };

        Self::from_coordinate_columns::<Input, Output, V>(
            input_structure,
            input_parameters.as_slice(),
            output_structure,
            output_parameters.as_slice(),
            columns,
        )
    }
}

impl<Rows, Partials, S> Differential<Rows, Partials, S>
where
    Rows: Parameterized<DifferentialRow<Partials, S>>,
    Partials: Parameterized<DifferentialBlock<S>>,
    S: Clone,
{
    /// Returns an iterator over every (output path, input path, [`DifferentialBlock`]) triple in
    /// this differential. The output path is yielded by [`Self::iter_rows`] and the input path by
    /// [`DifferentialRow::iter_partials`].
    pub fn iter_blocks(&self) -> impl Iterator<Item = (ParameterPath, ParameterPath, &DifferentialBlock<S>)> + '_ {
        self.rows.named_parameters().flat_map(|(output_path, row)| {
            row.iter_partials().map(move |(input_path, block)| (output_path.clone(), input_path, block))
        })
    }
}

/// Computes the flat offset of `index` within a tensor of `shape`, returning `None` if `index` is
/// out of bounds or has the wrong rank.
fn flat_offset(shape: &[usize], index: &[usize]) -> Option<usize> {
    if shape.len() != index.len() {
        return None;
    }
    let mut offset = 0;
    for (dimension_size, dimension_index) in shape.iter().zip(index.iter()) {
        if *dimension_index >= *dimension_size {
            return None;
        }
        offset = offset * dimension_size + dimension_index;
    }
    Some(offset)
}

/// Extracts the static shape of `array_type` as a `Vec<usize>`. Returns an error if any dimension is
/// dynamic, since differential materialization only operates on concrete primal values.
fn static_shape(array_type: &ArrayType) -> Result<Vec<usize>, ProgramError> {
    array_type
        .shape()
        .dimensions()
        .iter()
        .map(|size| match size {
            Size::Static(value) => Ok(*value),
            Size::Dynamic(_) => Err(TypeError {
                message: format!(
                    "differential materialization requires a fully static array shape but got {array_type}"
                ),
            }
            .into()),
        })
        .collect()
}

/// Returns the per-leaf coordinate counts produced by [`CoordinateValue::coordinate_count`].
fn coordinate_counts<V: CoordinateValue>(parameters: &[V]) -> Vec<usize> {
    parameters.iter().map(CoordinateValue::coordinate_count).collect::<Vec<_>>()
}

/// Computes inclusive-prefix offsets given a slice of per-leaf coordinate counts. The returned
/// slice has length `counts.len() + 1`, with the first element being `0` and the last being the
/// total coordinate count.
fn coordinate_offsets(counts: &[usize]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0);
    for count in counts {
        offsets.push(offsets.last().copied().unwrap_or(0) + count);
    }
    offsets
}

/// Builds one leaf's per-item values for a stacked standard basis.
///
/// The returned vector has `batch_size` entries and contains the appropriate one-hot basis value for this leaf's
/// coordinate range and a zero-like value elsewhere.
fn standard_basis_leaf_item_values<V: CoordinateValue>(
    parameter: &V,
    leaf_start: usize,
    leaf_count: usize,
    batch_size: usize,
) -> Vec<V> {
    let leaf_basis = parameter.coordinate_basis();
    let leaf_zero = parameter.zero_like();

    (0..batch_size)
        .map(|item| {
            if item >= leaf_start && item < leaf_start + leaf_count {
                leaf_basis[item - leaf_start].clone()
            } else {
                leaf_zero.clone()
            }
        })
        .collect()
}

/// Builds the standard basis metadata shared by forward- and reverse-mode differential materialization.
fn standard_basis_metadata<V: CoordinateValue>(parameters: &[V], batch_size: usize) -> (Vec<usize>, Vec<usize>) {
    let counts = coordinate_counts(parameters);
    let offsets = coordinate_offsets(&counts);
    debug_assert_eq!(offsets.last().copied().unwrap_or(0), batch_size, "batch size must equal total coord count");
    (counts, offsets)
}

/// Builds the standard basis for the coordinate space of a [`Parameterized`] tangent value,
/// packed per-leaf into [`ArrayBatch`]es for batched program interpretation.
///
/// For each input leaf, the returned [`ArrayBatch`] carries a `batch_size`-item stacked tangent
/// whose item `k` is the one-hot basis vector at position `k - leaf_offset[i]` when `k` falls
/// within leaf `i`'s coordinate range, and `zero_like` otherwise.
fn batched_standard_basis<V: CoordinateValue>(
    parameters: &[V],
    batch_size: usize,
) -> Result<Vec<ArrayBatch<V>>, BatchingError> {
    let (counts, offsets) = standard_basis_metadata(parameters, batch_size);
    parameters
        .iter()
        .enumerate()
        .map(|(leaf_index, parameter)| -> Result<_, BatchingError> {
            let leaf_type = parameter.r#type().into_owned();
            let item_values =
                standard_basis_leaf_item_values(parameter, offsets[leaf_index], counts[leaf_index], batch_size);
            let stacked = V::stack(item_values)?;

            // Insert the batch axis through the type itself so the batched basis keeps the leaf's optional metadata:
            // in particular, a sharded leaf yields a batched basis carrying the same per-dimension sharding with a
            // replicated inserted batch axis, matching the sharded declared input types of the replayed tangent
            // program. Rank-changing insertion clears any explicit layout, per `with_inserted_dimension`'s contract.
            let batched_type = leaf_type.with_inserted_dimension(0, Size::Static(batch_size))?;

            ArrayBatch::new(batched_type, stacked, Some(0))
        })
        .collect()
}

/// Batches one direct linear-program instruction for dense forward/reverse Jacobian interpretation, special-casing
/// zero-input replicated operations.
///
/// The dense-Jacobian replay interprets a residual-free linear program over [`ArrayBatch`]es by lifting each
/// instruction through its [`BatchableOperation`] rule. Zero-input operations (for example
/// the nullary `zero`/`one`/`fill` operations a structurally-zero forward pushforward stages for a padding or fill
/// region) are replicated by construction: every batch item receives the same value and there is no input batch
/// axis to lift through. Their batching rules reject a direct [`BatchableOperation::batch`] call for exactly this
/// reason and expect the surrounding staging path to handle them, so this helper interprets such operations once over
/// the per-item value type through the parent context and surfaces each result as a replicated [`ArrayBatch`] —
/// mirroring how [`BatchingContext`] stages a zero-input operation with an empty input list. Operations with at least
/// one input dispatch through their batching rule at the [`BatchingContext`] — the one context every operation
/// family's batching dispatcher is implemented at — so backend domains whose nullary constructions need runtime state
/// (for example a PJRT client) participate in the dense-Jacobian replay through the batching context's parent instead
/// of a fresh constant-only [`EagerContext`](crate::contexts::EagerContext).
fn batch_linear_program_instruction<C, V>(
    context: &BatchingContext<C>,
    operation: &<C as Domain>::Operation,
    inputs: &[ArrayBatch<V>],
) -> Result<Vec<ArrayBatch<V>>, BatchingError>
where
    C: Context<Type = ArrayType, Value = V>,
    V: Value<Type = ArrayType>,
    <C as Domain>::Operation: BatchableOperation<V, BatchingContext<C>> + InterpretableOperation<V, C>,
{
    if inputs.is_empty() {
        // A zero-input operation has no operand batch axis to lift through and is replicated by construction, so
        // interpret it once over the per-item value type and surface the result as a replicated value.
        return operation
            .interpret(context.parent(), &[])?
            .into_iter()
            .map(|value| Ok(ArrayBatch::replicated(value)))
            .collect();
    }
    BatchableOperation::<V, BatchingContext<C>>::batch(operation, context, inputs)
}

/// Extracts per-item flat output coordinates from a structured batched program output.
///
/// For each output leaf, the batched leaf value contributes `leaf_coord_count` coordinates per batch item carved out
/// of the batched flat coordinate buffer.
fn unstack_batched_coordinates<V>(
    output_batches: Vec<ArrayBatch<V>>,
    output_parameters: &[V],
    batch_size: usize,
) -> Result<Vec<Vec<V::Coordinate>>, ProgramError>
where
    V: CoordinateValue,
{
    assert_eq!(output_batches.len(), output_parameters.len(), "output parameter count mismatch");
    let mut columns: Vec<Vec<V::Coordinate>> = (0..batch_size).map(|_| Vec::new()).collect();
    for (leaf_batch, leaf_exemplar) in output_batches.into_iter().zip(output_parameters.iter()) {
        let leaf_coord_count = leaf_exemplar.coordinate_count();
        let all_coords = leaf_batch.into_value().coordinates();
        assert_eq!(
            all_coords.len(),
            batch_size * leaf_coord_count,
            "expected {batch_size} batch items x {leaf_coord_count} coords per output leaf",
        );
        for (item, item_coords) in all_coords.chunks(leaf_coord_count).enumerate() {
            columns[item].extend_from_slice(item_coords);
        }
    }
    Ok(columns)
}

/// Materializes a structured [`Differential`] using reverse-mode differentiation.
///
/// [`jacrev`] replays all output-coordinate basis cotangents through the pullback and reassembles
/// the resulting input coordinates into per-(output-leaf, input-leaf) [`DifferentialBlock`]s.
#[allow(private_bounds)]
pub fn jacrev<C, F, Input, TracedOutput>(
    context: &C,
    function: F,
    primals: Input,
) -> Result<Jacobian<Input, TracedOutput::To<DomainValue<C>>, DomainValue<C>>, ProgramError>
where
    C: Context<Type = ArrayType>,
    DomainValue<C>: CoordinateValue + BooleanLike,
    <C as Domain>::Constant: Value<Type = ArrayType>,
    Input: Parameterized<DomainValue<C>, To<DomainValue<C>> = Input, ParameterStructure: Debug + PartialEq>,
    TracedOutput: Parameterized<NestedTracer<C>, ParameterStructure: Debug + PartialEq>,
    Input::Family: ParameterizedFamily<<C as Domain>::Value>
        + ParameterizedFamily<NestedTracer<C>>
        + ParameterizedFamily<DifferentialBlock<CoordinateScalar<DomainValue<C>>>>,
    TracedOutput::Family: ParameterizedFamily<DomainValue<C>>
        + ParameterizedFamily<<C as Domain>::Value>
        + ParameterizedFamily<NestedTracer<C>>
        + ParameterizedFamily<
            DifferentialRow<
                Input::To<DifferentialBlock<CoordinateScalar<DomainValue<C>>>>,
                CoordinateScalar<DomainValue<C>>,
            >,
        >,
    TracedOutput::To<DomainValue<C>>: Parameterized<
            DomainValue<C>,
            To<DomainValue<C>> = TracedOutput::To<DomainValue<C>>,
            To<<C as Domain>::Value> = TracedOutput::To<<C as Domain>::Value>,
            To<
                DifferentialRow<
                    Input::To<DifferentialBlock<CoordinateScalar<DomainValue<C>>>>,
                    CoordinateScalar<DomainValue<C>>,
                >,
            > = TracedOutput::To<
                DifferentialRow<
                    Input::To<DifferentialBlock<CoordinateScalar<DomainValue<C>>>>,
                    CoordinateScalar<DomainValue<C>>,
                >,
            >,
            ParameterStructure: Debug + PartialEq,
        >,
    F: FnOnce(Input::To<NestedTracer<C>>) -> Result<TracedOutput, ProgramError>,
    <C as Domain>::Operation: Clone
        + InterpretableOperation<DomainValue<C>, C>
        + BatchableOperation<DomainValue<C>, BatchingContext<C>>
        + TransposableOperation<<C as Domain>::Constant, <C as Domain>::Operation>
        + MaybeWhile<<C as Domain>::Constant, <C as Domain>::Operation>
        + From<ZeroOperation<ArrayType>>
        + From<AddOperation>
        + DifferentiableOperation<TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>>
        + PartiallyEvaluatableOperation<
            TracingContext<<C as Domain>::Constant, <C as Domain>::Operation>,
        >,
    C: Zero<<C as Domain>::Value>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_shapes = input_parameters
        .iter()
        .map(|parameter| static_shape(parameter.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let input_offsets = coordinate_offsets(&input_coordinate_counts);
    let tangent_input_parameters = input_parameters
        .iter()
        .map(|parameter| context.zero(parameter.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;

    let primals = Input::from_parameters(input_structure.clone(), input_parameters)?;
    let (output, pullback, residuals) = context.vjp(function, primals)?;
    let output_structure = output.parameter_structure();
    let output_parameters = output.into_parameters().collect::<Vec<_>>();
    let output_shapes = output_parameters
        .iter()
        .map(|parameter| static_shape(parameter.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let output_coordinate_counts = coordinate_counts(output_parameters.as_slice());
    let output_offsets = coordinate_offsets(&output_coordinate_counts);
    let batch_size: usize = output_coordinate_counts.iter().sum();
    let cotangent_parameters = output_parameters
        .iter()
        .map(|parameter| context.zero(parameter.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let mut batched_basis_parameters =
        batched_standard_basis::<<C as Domain>::Value>(cotangent_parameters.as_slice(), batch_size)?;
    // The direct-transpose pullback consumes `[output_cotangents ++ residuals]`. The residuals depend only on the
    // primal, so they are identical across all cotangent rows; feed them as replicated operands
    // appended after the per-item cotangent basis.
    batched_basis_parameters.extend(residuals.into_iter().map(ArrayBatch::replicated));

    let rows = if batch_size == 0 {
        Vec::new()
    } else {
        // Each pullback instruction lifts through its batching dispatcher at an anonymous one-level
        // `BatchingContext` over the caller's context, so primitive work executes through that context itself.
        let batching_context = BatchingContext::new(context.clone(), batch_size, None);
        let batched_input = pullback.interpret_with(
            batched_basis_parameters,
            |_, constant: &<C as Domain>::Value| Ok::<_, BatchingError>(ArrayBatch::replicated(constant.clone())),
            |instruction, inputs: &[ArrayBatch<<C as Domain>::Value>]| {
                batch_linear_program_instruction(&batching_context, instruction.operation(), inputs)
            },
        )?;
        unstack_batched_coordinates::<<C as Domain>::Value>(
            batched_input,
            tangent_input_parameters.as_slice(),
            batch_size,
        )?
    };

    let mut rows_list = Vec::with_capacity(output_coordinate_counts.len());
    for (output_leaf_index, &output_count) in output_coordinate_counts.iter().enumerate() {
        let output_offset = output_offsets[output_leaf_index];
        let output_shape = &output_shapes[output_leaf_index];

        let mut blocks = Vec::with_capacity(input_coordinate_counts.len());
        for (input_leaf_index, &input_count) in input_coordinate_counts.iter().enumerate() {
            let input_offset = input_offsets[input_leaf_index];
            let input_shape = &input_shapes[input_leaf_index];

            let mut values = Vec::with_capacity(output_count * input_count);
            for output_local in 0..output_count {
                for input_local in 0..input_count {
                    let item = output_offset + output_local;
                    let coordinate = input_offset + input_local;
                    values.push(rows[item][coordinate].clone());
                }
            }

            blocks.push(DifferentialBlock::new(output_shape.clone(), input_shape.clone(), values));
        }

        let partials = <Input::To<DifferentialBlock<CoordinateScalar<DomainValue<C>>>>>::from_parameters(
            input_structure.clone(),
            blocks,
        )?;
        rows_list.push(DifferentialRow::new(partials));
    }

    let outer_rows = <TracedOutput::To<
        DifferentialRow<
            Input::To<DifferentialBlock<CoordinateScalar<DomainValue<C>>>>,
            CoordinateScalar<DomainValue<C>>,
        >,
    >>::from_parameters(output_structure, rows_list)?;

    Ok(Differential::new(outer_rows))
}
