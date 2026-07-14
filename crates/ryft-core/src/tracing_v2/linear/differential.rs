use std::fmt::Debug;
use std::marker::PhantomData;

use ryft_macros::Parameter;

use crate::backends::scalars::Scalar;
use crate::batching::ArrayBatch;
use crate::batching::BatchableOperation;
use crate::batching::BatchingContext;
use crate::batching::BatchingError;
use crate::contexts::{Context, Domain};
use crate::differentiation::LinearizationTracer;
use crate::differentiation::{DifferentiableOperation, DifferentiationError, TransposableOperation};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::BooleanLike;
use crate::operations::RegionlessDriver;
use crate::operations::constants::{FillOperation, OneOperation, ZeroLikeOperation, ZeroOperation};
use crate::operations::manipulation::{Broadcast, Reshape, Slice, Transpose};
use crate::operations::math::AddOperation;
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{DomainTracingContext, Tracer, TracingContext};
use crate::tracing_v2::{ForwardModeDifferentiate, ReverseModeDifferentiate};
use crate::types::{ArrayType, Shape, Size, TypeError, Typed};

/// Context capability that synthesizes a packed standard-basis value for dense differential replay.
///
/// For a leaf type `T` with `n` row-major coordinates, `coordinate_basis(T, offset, size)` returns one value with
/// physical type `[size] ++ T.shape`. Row `k` is the leaf's one-hot vector at coordinate `k - offset` when
/// `offset <= k < offset + n`, and zero otherwise. The inserted leading basis axis is replicated; the leaf axes keep
/// `T`'s placement metadata. Owning this operation on the context lets eager device backends synthesize the whole
/// packed basis directly on device instead of uploading `n` host one-hot buffers.
pub trait CoordinateBasis<V: Value<Type = ArrayType>> {
    /// Synthesizes one packed standard-basis leaf with a leading axis of length `basis_size`.
    fn coordinate_basis(
        &self,
        leaf_type: &ArrayType,
        coordinate_offset: usize,
        basis_size: usize,
    ) -> Result<V, ProgramError>;
}

/// Structured forward- or reverse-mode Jacobian of a function `Input -> Output` over leaf value
/// type `V`. Materialized by [`DifferentiableDomainExtension::jacfwd`] and [`jacrev`].
///
/// The outer [`Parameterized`] family mirrors the function's output; each output-leaf position
/// holds a [`DifferentialRow`] whose internal family mirrors the function's input and whose leaves
/// are [`DifferentialBlock`]s whose partial-derivative tensors remain values in the same execution domain.
pub type Jacobian<Input, Output, V> = Differential<
    <Output as Parameterized<V>>::To<DifferentialRow<<Input as Parameterized<V>>::To<DifferentialBlock<V>>, V>>,
    <Input as Parameterized<V>>::To<DifferentialBlock<V>>,
    V,
>;

/// Structured Hessian of a scalar-output function over a [`Parameterized`] input with leaf value
/// type `V`. Materialized by [`DifferentiableDomainExtension::hessian`].
///
/// Equivalent to a [`Jacobian<Input, Input, V>`] - both the outer and inner [`Parameterized`]
/// families mirror the input.
pub type Hessian<Input, V> = Jacobian<Input, Input, V>;

/// Concrete value type selected by a [`Domain`].
type DomainValue<D> = <D as Domain>::Value;

/// Dense derivative materialization helpers for differentiable array domains.
///
/// This extension trait keeps the core [`ForwardModeDifferentiate`] and [`ReverseModeDifferentiate`]
/// contracts focused on primitive linearization and AD transforms while providing structured Jacobian
/// and Hessian materialization for domains that can synthesize finite coordinate bases.
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
        Self: CoordinateBasis<DomainValue<Self>>,
        DomainValue<Self>: Value<Type = ArrayType> + BooleanLike + Broadcast + Reshape + Slice + Transpose + 'domain,
        Input: Parameterized<
                DomainValue<Self>,
                To<DomainValue<Self>> = Input,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<LinearizationTracer<Self>, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<LinearizationTracer<Self>>
            + ParameterizedFamily<DifferentialBlock<DomainValue<Self>>>,
        TracedOutput::Family: ParameterizedFamily<DomainValue<Self>>
            + ParameterizedFamily<LinearizationTracer<Self>>
            + ParameterizedFamily<
                DifferentialRow<
                    Input::To<DifferentialBlock<DomainValue<Self>>>,
                    DomainValue<Self>,
                >,
            >,
        TracedOutput::To<DomainValue<Self>>: Parameterized<
                DomainValue<Self>,
                To<DomainValue<Self>> = TracedOutput::To<DomainValue<Self>>,
                To<
                    DifferentialRow<
                        Input::To<DifferentialBlock<DomainValue<Self>>>,
                        DomainValue<Self>,
                    >,
                > = TracedOutput::To<
                    DifferentialRow<
                        Input::To<DifferentialBlock<DomainValue<Self>>>,
                        DomainValue<Self>,
                    >,
                >,
        >,
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Result<TracedOutput, ProgramError>,
        <Self as Domain>::Operation: Clone
            + InterpretableOperation<DomainValue<Self>, Self>
            + PartiallyEvaluatableOperation<Self> + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + From<ZeroOperation<ArrayType>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + BatchableOperation<Self>,
    {
        let input_structure = primal.parameter_structure();
        let input_parameters = primal.into_parameters().collect::<Vec<_>>();
        let primals = Input::from_parameters(input_structure.clone(), input_parameters.iter().cloned())?;

        // Dual-driven forward-mode Jacobian: `ForwardModeDifferentiate::linearize` runs the closure once on differentiation
        // duals over a partial-evaluation context wrapping this context, which executes the primal work through this
        // context — recovering the structured primal output and the linearization-point residuals — while
        // accumulating the linear pushforward program. `from_pushforward_program` then replays every
        // input-coordinate basis tangent through that program in one batched pass — broadcasting the residuals as
        // replicated values — preserving the exact Jacobian layout.
        let (output, pushforward) = self.linearize(function, primals)?;
        let (program, residuals) = pushforward.into_parts();
        Differential::from_pushforward_program::<Self, Input, TracedOutput::To<DomainValue<Self>>>(
            self,
            input_structure,
            input_parameters,
            output,
            program,
            residuals,
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
        Self: CoordinateBasis<DomainValue<Self>>,
        DomainValue<Self>: Value<Type = ArrayType> + Broadcast + Reshape + Slice + Transpose + 'domain,
        Input: Parameterized<DomainValue<Self>, To<DomainValue<Self>> = Input, ParameterStructure: Debug + PartialEq>,
        Input::Family: ParameterizedFamily<LinearizationTracer<DomainTracingContext<Self>>>
            + ParameterizedFamily<Tracer<DomainTracingContext<Self>>>
            + ParameterizedFamily<<Self as Domain>::Constant>
            + ParameterizedFamily<DifferentialBlock<DomainValue<Self>>>
            + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<DomainValue<Self>>>, DomainValue<Self>>>,
        Input::To<Tracer<DomainTracingContext<Self>>>: Parameterized<
                Tracer<DomainTracingContext<Self>>,
                To<Tracer<DomainTracingContext<Self>>> = Input::To<Tracer<DomainTracingContext<Self>>>,
                To<DomainValue<Self>> = Input,
                To<<Self as Domain>::Constant> = Input::To<<Self as Domain>::Constant>,
                To<LinearizationTracer<DomainTracingContext<Self>>> = Input::To<
                    LinearizationTracer<DomainTracingContext<Self>>,
                >,
                ParameterStructure: Debug + PartialEq,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DomainTracingContext<Self>>>,
        ) -> LinearizationTracer<DomainTracingContext<Self>>,
        <Self as Domain>::Operation: Clone
            + InterpretableOperation<DomainValue<Self>, Self>
            + TransposableOperation<<Self as Domain>::Constant, <Self as Domain>::Operation>
            + DifferentiableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>
            + DifferentiableOperation<
                PartialEvaluationContext<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>,
            > + PartiallyEvaluatableOperation<TracingContext<<Self as Domain>::Constant, <Self as Domain>::Operation>>
            + BatchableOperation<Self>
            + From<FillOperation<ArrayType, Scalar>>
            + From<ZeroOperation<ArrayType>>
            + From<OneOperation<ArrayType>>
            + From<ZeroLikeOperation>
            + From<AddOperation>,
        <Self as Domain>::Constant: Value<Type = ArrayType>,
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
                context.gradient(function, input).map_err(|error| match error {
                    DifferentiationError::Program(error) => error,
                    error => ProgramError::MalformedProgram(error.to_string()),
                })
            },
            primals,
        )?;
        // Linearize the gradient program through the program-level split (defined on the canonical flat form, so the
        // structured trace is flattened first), replay its primal sub-program at the primals to recover the
        // linearization-point residuals — its trailing `residual_count` outputs, per the `Linearization` output
        // contract — and then replay every input-coordinate basis tangent through its pushforward program. The
        // already-evaluated `gradient` supplies the output shapes and structure `from_pushforward_program` needs.
        let linearization = gradient_program.into_flat_program().linearize()?;
        let (primal_program, pushforward_program, residual_count) = linearization.into_parts();
        let mut primal_outputs = primal_program
            .interpret_in_context(self, input_parameters.clone())
            .map_err(DifferentiationError::from)?;
        let residuals = primal_outputs.split_off(primal_outputs.len() - residual_count);
        Differential::from_pushforward_program::<Self, Input, Input>(
            self,
            input_structure,
            input_parameters,
            gradient,
            pushforward_program,
            residuals,
        )
        .map_err(DifferentiationError::from)
    }
}

impl<D> DifferentiableDomainExtension for D where D: Context<Type = ArrayType> {}

/// Partial derivatives of one output leaf with respect to one input leaf.
///
/// For an output leaf of shape `O` and an input leaf of shape `I`, a `DifferentialBlock` carries
/// one `O ++ I`-shaped partial-derivative tensor in the execution domain, with the output dimensions varying
/// slowest. Keeping the tensor as a domain value preserves device placement, sharding, and element type and avoids
/// an implicit device-to-host synchronization at the transform boundary.
#[derive(Parameter, Clone, Debug)]
pub struct DifferentialBlock<V> {
    /// Shape of the output leaf this block contributes to.
    output_shape: Vec<usize>,

    /// Shape of the input leaf this block differentiates with respect to.
    input_shape: Vec<usize>,

    /// Device/domain partial-derivative tensor with shape `output_shape ++ input_shape`.
    value: V,
}

impl<V: Value<Type = ArrayType>> DifferentialBlock<V> {
    /// Constructs a [`DifferentialBlock`] from explicit logical shapes and its domain tensor.
    ///
    /// # Parameters
    ///
    ///   - `output_shape`: Shape of the output leaf this block contributes to.
    ///   - `input_shape`: Shape of the input leaf this block differentiates with respect to.
    ///   - `value`: Partial-derivative tensor. Its shape must equal `output_shape ++ input_shape`.
    pub fn new(output_shape: Vec<usize>, input_shape: Vec<usize>, value: V) -> Result<Self, ProgramError> {
        let expected_shape =
            Shape::new(output_shape.iter().chain(input_shape.iter()).copied().map(Size::Static).collect());
        if value.r#type().shape() != &expected_shape {
            return Err(TypeError {
                message: format!(
                    "differential block value has shape {} but output shape {output_shape:?} and input shape \
                     {input_shape:?} require {expected_shape}",
                    value.r#type().shape(),
                ),
            }
            .into());
        }
        Ok(Self { output_shape, input_shape, value })
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

    /// Returns the domain value storing this block's partial-derivative tensor.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes this block and returns its partial-derivative tensor.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }
}

/// One row of a [`Differential`]: partial derivatives of one output leaf with respect to every
/// input leaf.
///
/// `Partials` is the input `Parameterized` value already reparameterized to [`DifferentialBlock`]
/// leaves (typically `Input::To<DifferentialBlock<V>>` at the call site). Carries [`Parameter`] —
/// **not** [`Parameterized`] — so it can appear as a leaf inside the outer `Parameterized` value
/// held by [`Differential`]. Internal structure is accessed via [`Self::partials`].
#[derive(Parameter, Clone, Debug)]
pub struct DifferentialRow<Partials, V>
where
    Partials: Parameterized<DifferentialBlock<V>>,
{
    /// Input-shaped [`Parameterized`] value whose leaves are the partial-derivative blocks for the
    /// output leaf this row corresponds to.
    partials: Partials,

    /// Marker keeping the domain value type fixed at the type level.
    _value: PhantomData<V>,
}

impl<Partials, V> DifferentialRow<Partials, V>
where
    Partials: Parameterized<DifferentialBlock<V>>,
{
    /// Constructs a [`DifferentialRow`] from an input-shaped [`Parameterized`] value of
    /// [`DifferentialBlock`]s.
    #[inline]
    pub fn new(partials: Partials) -> Self {
        Self { partials, _value: PhantomData }
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
    ) -> <Partials as Parameterized<DifferentialBlock<V>>>::NamedParameterIterator<'_, DifferentialBlock<V>> {
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
pub struct Differential<Rows, Partials, V>
where
    Rows: Parameterized<DifferentialRow<Partials, V>>,
    Partials: Parameterized<DifferentialBlock<V>>,
{
    /// Output-shaped [`Parameterized`] value whose leaves are the [`DifferentialRow`]s for each
    /// output leaf.
    rows: Rows,

    /// Marker keeping the inner partials and scalar types fixed at the type level.
    _phantom: PhantomData<(Partials, V)>,
}

impl<Rows, Partials, V> Differential<Rows, Partials, V>
where
    Rows: Parameterized<DifferentialRow<Partials, V>>,
    Partials: Parameterized<DifferentialBlock<V>>,
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
    ) -> <Rows as Parameterized<DifferentialRow<Partials, V>>>::NamedParameterIterator<'_, DifferentialRow<Partials, V>>
    {
        self.rows.named_parameters()
    }

    /// Constructs a [`Differential`] from a capture-free [`Linearization`] by replaying its primal sub-program
    /// once at the primal point and batch-replaying its tangent sub-program across every input-coordinate basis
    /// tangent.
    ///
    /// The pushforward `program` maps `[tangents..., residuals...]` to the tangent outputs and stays over the
    /// *primal* operation family `C::Operation`, with `residuals` carrying the linearization-point values its
    /// trailing inputs consume. This helper:
    ///
    ///   1. Batch-replays the pushforward program across the stacked input-coordinate basis tangents (all basis
    ///      items on axis 0), appending the residuals as replicated [`ArrayBatch::replicated`] operands after the
    ///      batched basis tangents — the same replicated mechanism [`jacrev`] uses for its reverse-mode residuals.
    ///      Staged constants are lifted through `context` and broadcast as replicated operands the same way. Because
    ///      the pushforward program is expressed in the primal operation family, each instruction is lifted through
    ///      its primal-family [`BatchableOperation`] rule by [`batch_linear_program_instruction`], interpreting
    ///      nullary instructions through `context`.
    ///   2. Slices, reshapes, and transposes the packed output values into per-(output-leaf, input-leaf)
    ///      [`DifferentialBlock`] domain tensors.
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
    ///   - `input_parameters`: Concrete leaf values of `Input` at the point of linearization, used to derive each input
    ///     leaf's type, static shape, and coordinate range.
    ///   - `output`: Primal output of the linearized function, consumed to recover its placeholder shape and the static
    ///     shapes of its output leaves.
    ///   - `linearization`: Capture-free linearization whose primal and tangent sub-programs are replayed.
    pub(crate) fn from_pushforward_program<C, Input, Output>(
        context: &C,
        input_structure: Input::ParameterStructure,
        input_parameters: Vec<V>,
        output: Output,
        program: Program<
            <C as Domain>::Constant,
            <C as Domain>::Operation,
            Vec<<C as Domain>::Constant>,
            Vec<<C as Domain>::Constant>,
        >,
        residuals: Vec<V>,
    ) -> Result<Self, ProgramError>
    where
        C: Context<Type = ArrayType, Value = V> + CoordinateBasis<V>,
        V: Value<Type = ArrayType> + Broadcast + Reshape + Slice + Transpose,
        Input:
            Parameterized<V, To<V> = Input, To<DifferentialBlock<V>> = Partials, ParameterStructure: Debug + PartialEq>,
        Output:
            Parameterized<V, To<V> = Output, To<DifferentialRow<Partials, V>> = Rows, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<DifferentialBlock<V>>,
        Output::Family: ParameterizedFamily<DifferentialRow<Partials, V>>,
        Partials: Parameterized<DifferentialBlock<V>, ParameterStructure = Input::ParameterStructure>,
        Rows: Parameterized<DifferentialRow<Partials, V>, ParameterStructure = Output::ParameterStructure>,
        <C as Domain>::Operation: Clone + InterpretableOperation<V, C> + BatchableOperation<C>,
    {
        let input_types = input_parameters.iter().map(|parameter| parameter.r#type().into_owned()).collect::<Vec<_>>();
        let input_shapes = input_types
            .iter()
            .map(|input_type| {
                input_type.static_shape().ok_or_else(|| TypeError {
                    message: format!(
                        "differential materialization requires a fully static array shape but got {input_type}"
                    ),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_coordinate_counts = coordinate_counts(input_types.as_slice())?;
        let input_offsets = coordinate_offsets(&input_coordinate_counts)?;
        let batch_size = input_offsets.last().copied().unwrap_or(0);
        let batched_basis_parameters = batched_standard_basis(context, input_types.as_slice(), batch_size)?;

        let output_structure = output.parameter_structure();
        let output_parameters = output.into_parameters().collect::<Vec<_>>();
        let output_shapes = output_parameters
            .iter()
            .map(|parameter| {
                let array_type = parameter.r#type();
                array_type.static_shape().ok_or_else(|| TypeError {
                    message: format!(
                        "differential materialization requires a fully static array shape but got {array_type}"
                    ),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Feed `[batched_basis_tangents..., unbatched_residuals...]` through one batched pushforward replay. The packed
        // result leaves stay in the execution domain; slicing and axis rearrangement below carve them directly into
        // per-(output, input) derivative tensors without coordinate readback.
        let batching_context = BatchingContext::new(context.clone(), batch_size, None);
        let mut tangent_inputs = batched_basis_parameters;
        tangent_inputs.extend(residuals.into_iter().map(ArrayBatch::replicated));
        let batched_outputs = program.interpret_with(
            tangent_inputs,
            |_, constant| Ok::<_, BatchingError>(ArrayBatch::replicated(context.lift(constant.clone())?)),
            |instruction, inputs: &[ArrayBatch<V>]| {
                batch_linear_program_instruction(&batching_context, instruction.operation(), inputs)
            },
        )?;
        check_count!("output", batched_outputs, output_parameters.len(), ProgramError);

        let mut rows_list = Vec::with_capacity(output_parameters.len());
        for (output_leaf_index, output_batch) in batched_outputs.iter().enumerate() {
            let output_shape = &output_shapes[output_leaf_index];
            let mut blocks = Vec::with_capacity(input_shapes.len());
            for (input_leaf_index, input_shape) in input_shapes.iter().enumerate() {
                // The packed replay lays each block out as `[input_coordinates] ++ output_shape`; move the input
                // basis axes behind the output axes to obtain the public `output_shape ++ input_shape` layout.
                let value = basis_range_value(
                    output_batch,
                    batch_size,
                    input_offsets[input_leaf_index],
                    input_shape.dimensions(),
                    output_shape.dimensions(),
                )?;
                let permutation = (input_shape.rank()..input_shape.rank() + output_shape.rank())
                    .chain(0..input_shape.rank())
                    .collect::<Vec<_>>();
                blocks.push(DifferentialBlock::new(
                    output_shape.dimensions().to_vec(),
                    input_shape.dimensions().to_vec(),
                    value.transpose(permutation)?,
                )?);
            }
            rows_list.push(DifferentialRow::new(Partials::from_parameters(input_structure.clone(), blocks)?));
        }

        Ok(Self::new(Rows::from_parameters(output_structure, rows_list)?))
    }
}

impl<Rows, Partials, V> Differential<Rows, Partials, V>
where
    Rows: Parameterized<DifferentialRow<Partials, V>>,
    Partials: Parameterized<DifferentialBlock<V>>,
{
    /// Returns an iterator over every (output path, input path, [`DifferentialBlock`]) triple in
    /// this differential. The output path is yielded by [`Self::iter_rows`] and the input path by
    /// [`DifferentialRow::iter_partials`].
    pub fn iter_blocks(&self) -> impl Iterator<Item = (ParameterPath, ParameterPath, &DifferentialBlock<V>)> + '_ {
        self.rows.named_parameters().flat_map(|(output_path, row)| {
            row.iter_partials().map(move |(input_path, block)| (output_path.clone(), input_path, block))
        })
    }
}

/// Returns the row-major coordinate count of each statically shaped leaf type.
fn coordinate_counts(types: &[ArrayType]) -> Result<Vec<usize>, ProgramError> {
    types
        .iter()
        .map(|r#type| {
            let shape = r#type.static_shape().ok_or_else(|| TypeError {
                message: format!("differential materialization requires a fully static array shape but got {type}"),
            })?;
            shape.dimensions().iter().copied().try_fold(1usize, |count, size| {
                count.checked_mul(size).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!("differential coordinate count overflows usize for array type {type}"),
                })
            })
        })
        .collect()
}

/// Computes inclusive-prefix offsets given a slice of per-leaf coordinate counts. The returned
/// slice has length `counts.len() + 1`, with the first element being `0` and the last being the
/// total coordinate count.
fn coordinate_offsets(counts: &[usize]) -> Result<Vec<usize>, ProgramError> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0);
    for count in counts {
        offsets.push(offsets.last().copied().unwrap_or(0usize).checked_add(*count).ok_or_else(|| {
            ProgramError::InvalidArgument { message: "total differential coordinate count overflows usize".into() }
        })?);
    }
    Ok(offsets)
}

/// Builds the standard basis for the coordinate space of a [`Parameterized`] tangent value,
/// packed per-leaf into [`ArrayBatch`]es for batched program interpretation.
///
/// For each leaf, the context produces one `[batch_size] ++ leaf_shape` value whose row `k` is the leaf-local one-hot
/// vector selected by the global coordinate offset, or zero when that global row belongs to another leaf.
fn batched_standard_basis<C, V>(
    context: &C,
    types: &[ArrayType],
    batch_size: usize,
) -> Result<Vec<ArrayBatch<V>>, ProgramError>
where
    C: CoordinateBasis<V>,
    V: Value<Type = ArrayType>,
{
    let counts = coordinate_counts(types)?;
    let offsets = coordinate_offsets(&counts)?;
    if offsets.last().copied().unwrap_or(0) != batch_size {
        return Err(ProgramError::InvalidArgument {
            message: "basis size must equal the total coordinate count".into(),
        });
    }
    types
        .iter()
        .enumerate()
        .map(|(leaf_index, leaf_type)| {
            let expected_type = leaf_type.with_inserted_dimension(0, Size::Static(batch_size))?;
            let value = context.coordinate_basis(leaf_type, offsets[leaf_index], batch_size)?;
            if value.r#type().as_ref() != &expected_type {
                return Err(TypeError {
                    message: format!(
                        "coordinate basis for leaf type {leaf_type} has type {} but expected {expected_type}",
                        value.r#type(),
                    ),
                }
                .into());
            }
            ArrayBatch::new(expected_type, value, Some(0)).map_err(ProgramError::from)
        })
        .collect()
}

/// Extracts a contiguous range of leading basis rows and reshapes that flattened basis range back to
/// `basis_shape`, leaving the per-item axes after it.
fn basis_range_value<V>(
    batch: &ArrayBatch<V>,
    batch_size: usize,
    basis_offset: usize,
    basis_shape: &[usize],
    item_shape: &[usize],
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Broadcast + Reshape + Slice + Transpose,
{
    let aligned = batch.match_axis(0, batch_size)?;
    let unbatched_type = aligned.unbatched_type()?;
    let actual_item_shape = unbatched_type.static_shape().ok_or_else(|| TypeError {
        message: format!("differential materialization requires a fully static array shape but got {unbatched_type}"),
    })?;
    if actual_item_shape.dimensions() != item_shape {
        return Err(TypeError {
            message: format!(
                "batched differential output has per-item shape {:?} but expected {item_shape:?}",
                actual_item_shape.dimensions(),
            ),
        }
        .into());
    }
    let basis_count = basis_shape.iter().try_fold(1usize, |count, size| {
        count.checked_mul(*size).ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("differential basis shape {basis_shape:?} overflows usize"),
        })
    })?;
    let physical_type = aligned.r#type();
    let physical_shape = physical_type.static_shape().ok_or_else(|| TypeError {
        message: format!("differential materialization requires a fully static array shape but got {physical_type}"),
    })?;
    let mut start_indices = vec![0; physical_shape.rank()];
    start_indices[0] = basis_offset;
    let mut limit_indices = physical_shape.dimensions().to_vec();
    limit_indices[0] = basis_offset + basis_count;
    let strides = vec![1; limit_indices.len()];
    let sliced = aligned.value().slice(&start_indices, &limit_indices, &strides)?;
    let reshaped_shape = Shape::new(basis_shape.iter().chain(item_shape.iter()).copied().map(Size::Static).collect());
    sliced.reshape(reshaped_shape)
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
    <C as Domain>::Operation: BatchableOperation<C> + InterpretableOperation<V, C>,
{
    if inputs.is_empty() {
        // A zero-input operation has no operand batch axis to lift through and is replicated by construction, so
        // interpret it once over the per-item value type and surface the result as a replicated value.
        return operation
            .interpret(&context.parent().clone(), &RegionlessDriver, &[])?
            .into_iter()
            .map(|value| Ok(ArrayBatch::replicated(value)))
            .collect();
    }
    operation.batch(context, &RegionlessDriver, inputs)
}

/// Materializes a structured [`Differential`] using reverse-mode differentiation.
///
/// [`jacrev`] replays all output-coordinate basis cotangents through the pullback and slices the packed input
/// cotangents into per-(output-leaf, input-leaf) device/domain [`DifferentialBlock`] tensors.
#[allow(private_bounds)]
pub fn jacrev<C, F, Input, TracedOutput>(
    context: &C,
    function: F,
    primals: Input,
) -> Result<Jacobian<Input, TracedOutput::To<DomainValue<C>>, DomainValue<C>>, ProgramError>
where
    C: Context<Type = ArrayType> + CoordinateBasis<DomainValue<C>>,
    DomainValue<C>: Value<Type = ArrayType> + BooleanLike + Broadcast + Reshape + Slice + Transpose,
    <C as Domain>::Constant: Value<Type = ArrayType>,
    Input: Parameterized<DomainValue<C>, To<DomainValue<C>> = Input, ParameterStructure: Debug + PartialEq>,
    TracedOutput: Parameterized<LinearizationTracer<C>, ParameterStructure: Debug + PartialEq>,
    Input::Family: ParameterizedFamily<<C as Domain>::Value>
        + ParameterizedFamily<LinearizationTracer<C>>
        + ParameterizedFamily<DifferentialBlock<DomainValue<C>>>,
    TracedOutput::Family: ParameterizedFamily<DomainValue<C>>
        + ParameterizedFamily<<C as Domain>::Value>
        + ParameterizedFamily<LinearizationTracer<C>>
        + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<DomainValue<C>>>, DomainValue<C>>>,
    TracedOutput::To<DomainValue<C>>: Parameterized<
            DomainValue<C>,
            To<DomainValue<C>> = TracedOutput::To<DomainValue<C>>,
            To<<C as Domain>::Value> = TracedOutput::To<<C as Domain>::Value>,
            To<DifferentialRow<Input::To<DifferentialBlock<DomainValue<C>>>, DomainValue<C>>> = TracedOutput::To<
                DifferentialRow<Input::To<DifferentialBlock<DomainValue<C>>>, DomainValue<C>>,
            >,
            ParameterStructure: Debug + PartialEq,
        >,
    F: FnOnce(Input::To<LinearizationTracer<C>>) -> Result<TracedOutput, ProgramError>,
    <C as Domain>::Operation: Clone
        + InterpretableOperation<DomainValue<C>, C>
        + BatchableOperation<C>
        + TransposableOperation<<C as Domain>::Constant, <C as Domain>::Operation>
        + PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + From<ZeroOperation<ArrayType>>
        + From<AddOperation>
        + DifferentiableOperation<PartialEvaluationContext<C>>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_types = input_parameters.iter().map(|parameter| parameter.r#type().into_owned()).collect::<Vec<_>>();
    let input_shapes = input_parameters
        .iter()
        .map(|parameter| {
            let array_type = parameter.r#type();
            array_type.static_shape().ok_or_else(|| TypeError {
                message: format!(
                    "differential materialization requires a fully static array shape but got {array_type}"
                ),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let primals = Input::from_parameters(input_structure.clone(), input_parameters)?;
    let (output, pullback) = context.vjp(function, primals)?;
    let (pullback, residuals) = pullback.into_parts();
    let output_structure = output.parameter_structure();
    let output_parameters = output.into_parameters().collect::<Vec<_>>();
    let output_types = output_parameters.iter().map(|parameter| parameter.r#type().into_owned()).collect::<Vec<_>>();
    let output_shapes = output_parameters
        .iter()
        .map(|parameter| {
            let array_type = parameter.r#type();
            array_type.static_shape().ok_or_else(|| TypeError {
                message: format!(
                    "differential materialization requires a fully static array shape but got {array_type}"
                ),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_coordinate_counts = coordinate_counts(output_types.as_slice())?;
    let output_offsets = coordinate_offsets(&output_coordinate_counts)?;
    let batch_size = output_offsets.last().copied().unwrap_or(0);
    let mut batched_basis_parameters = batched_standard_basis(context, output_types.as_slice(), batch_size)?;
    // The direct-transpose pullback consumes `[output_cotangents ++ residuals]`. The residuals depend only on the
    // primal, so they are identical across all cotangent rows; feed them as replicated operands
    // appended after the per-item cotangent basis.
    batched_basis_parameters.extend(residuals.into_iter().map(ArrayBatch::replicated));

    // Replay all output cotangent basis rows through the pullback in one batch. The returned input leaves stay as
    // packed domain values and are sliced directly into derivative blocks below.
    let batching_context = BatchingContext::new(context.clone(), batch_size, None);
    let batched_inputs = pullback.interpret_with(
        batched_basis_parameters,
        |_, constant: &<C as Domain>::Constant| {
            Ok::<_, BatchingError>(ArrayBatch::replicated(context.lift(constant.clone())?))
        },
        |instruction, inputs: &[ArrayBatch<<C as Domain>::Value>]| {
            batch_linear_program_instruction(&batching_context, instruction.operation(), inputs)
        },
    )?;
    check_count!("output", batched_inputs, input_types.len(), ProgramError);

    let mut rows_list = Vec::with_capacity(output_coordinate_counts.len());
    for output_leaf_index in 0..output_coordinate_counts.len() {
        let output_offset = output_offsets[output_leaf_index];
        let output_shape = &output_shapes[output_leaf_index];

        let mut blocks = Vec::with_capacity(input_shapes.len());
        for (input_leaf_index, input_batch) in batched_inputs.iter().enumerate() {
            // Pullback replay already packs the output basis rows before each input leaf's axes, so restoring the
            // flattened output range to `output_shape` yields the public layout directly.
            let input_shape = &input_shapes[input_leaf_index];
            let value = basis_range_value(
                input_batch,
                batch_size,
                output_offset,
                output_shape.dimensions(),
                input_shape.dimensions(),
            )?;
            blocks.push(DifferentialBlock::new(
                output_shape.dimensions().to_vec(),
                input_shape.dimensions().to_vec(),
                value,
            )?);
        }

        let partials =
            <Input::To<DifferentialBlock<DomainValue<C>>>>::from_parameters(input_structure.clone(), blocks)?;
        rows_list.push(DifferentialRow::new(partials));
    }

    let outer_rows = <TracedOutput::To<
        DifferentialRow<
            Input::To<DifferentialBlock<DomainValue<C>>>,
            DomainValue<C>,
        >,
    >>::from_parameters(output_structure, rows_list)?;

    Ok(Differential::new(outer_rows))
}
