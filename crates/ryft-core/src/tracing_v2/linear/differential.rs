use std::fmt::Debug;
use std::marker::PhantomData;

use ryft_macros::Parameter;

use crate::differentiation::{Tangent, TransposableOperation};
use crate::domains::Domain;
use crate::operations::InterpretableOperation;
use crate::operations::arithmetic::{AddOperation, SupportsAdd};
use crate::operations::constants::{
    One, OneLike, SupportsFill, SupportsOne, SupportsZero, SupportsZeroLike, Zero, ZeroLike,
};
use crate::parameters::{Parameter, ParameterPath, Parameterized, ParameterizedFamily};
use crate::programs::{Program, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};
use crate::tracing_v2::differentiation::direct_batched_jvp;
use crate::tracing_v2::{
    DifferentiableOperation, DifferentiationContext, DifferentiationError, DirectLinearOperationOf, LinearOperationOf,
    LinearizationContext, LinearizationTracer, Pushforward, ResidualizedOperation,
};
use crate::types::{ArrayType, Shape, Size, TypeError, Typed};

/// Leaf type that can be materialized into a dense finite-dimensional coordinate representation.
///
/// Structured differential materialization only makes sense for leaf types with a finite, explicit
/// basis. [`CoordinateValue`] is the bridge from the generic tracing world into that coordinate-based
/// view: it teaches the differential helpers how many coordinates a leaf contributes, what basis
/// vectors to probe with, and how to flatten outputs back into numeric entries.
pub trait CoordinateValue: Value<ArrayType> + ZeroLike + OneLike + Zero<ArrayType> + One<ArrayType> {
    /// Scalar-like coordinate type used by [`DifferentialBlock`] entries.
    type Coordinate: Clone + Debug + PartialEq + 'static;

    /// Returns the number of coordinates contributed by this leaf.
    fn coordinate_count(&self) -> usize;

    /// Returns a standard basis for the coordinate space of this leaf.
    fn coordinate_basis(&self) -> Vec<Self>;

    /// Flattens the leaf into its coordinate values in a deterministic order.
    fn coordinates(&self) -> Vec<Self::Coordinate>;

    /// Stacks the provided values along a new leading lane axis. All values must share the same
    /// [`ArrayType`]; the resulting value carries that type prefixed with `Size::Static(values.len())`
    /// on axis `0`. Used by [`Differential::from_pushforward`] and [`jacrev`] to pack `N`
    /// per-basis-tangent values into one batched input that flows through the value-level
    /// [`BatchableOperation::batch`] rule for `Tangent` values.
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
/// This extension trait keeps the core [`DifferentiationContext`] contract focused on primitive
/// linearization and AD transforms while providing structured Jacobian and Hessian materialization
/// for domains whose values expose finite coordinate bases.
pub trait DifferentiableDomainExtension: Domain<Type = ArrayType> + DifferentiationContext {
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
        DomainValue<Self>: CoordinateValue + 'domain,
        Self::Tangent: CoordinateValue<Coordinate = CoordinateScalar<DomainValue<Self>>>,
        Input: Parameterized<
                DomainValue<Self>,
                To<DomainValue<Self>> = Input,
                ParameterStructure: Debug + PartialEq,
            >,
        TracedOutput: Parameterized<LinearizationTracer<'domain, Self>, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<LinearizationTracer<'domain, Self>>
            + ParameterizedFamily<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>,
        TracedOutput::Family: ParameterizedFamily<DomainValue<Self>>
            + ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<LinearizationTracer<'domain, Self>>
            + ParameterizedFamily<
                DifferentialRow<
                    Input::To<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>,
                    CoordinateScalar<DomainValue<Self>>,
                >,
            >,
        TracedOutput::To<DomainValue<Self>>: Parameterized<
                DomainValue<Self>,
                To<DomainValue<Self>> = TracedOutput::To<DomainValue<Self>>,
                To<Self::Tangent> = TracedOutput::To<Self::Tangent>,
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
        F: FnOnce(Input::To<LinearizationTracer<'domain, Self>>) -> Result<TracedOutput, ProgramError>,
        <Self as Domain>::Operation: DifferentiableOperation<Self>,
        Self::Tangent: crate::tracing_v2::operations::broadcast::BroadcastInDim
            + crate::tracing_v2::operations::transpose::Transpose,
        DirectLinearOperationOf<Self>: BatchableOperation<Tangent<ArrayType, Self::Tangent>>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
    {
        let input_structure = primal.parameter_structure();
        let input_parameters = primal.into_parameters().collect::<Vec<_>>();
        let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
        let lane_count: usize = input_coordinate_counts.iter().sum();
        let tangent_parameters = input_parameters
            .iter()
            .map(|parameter| Self::Tangent::zero(parameter.r#type().as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let batched_basis_parameters =
            stacked_standard_basis::<Self::Tangent>(tangent_parameters.as_slice(), lane_count)?;
        let batched_tangents =
            Input::To::<Self::Tangent>::from_parameters(input_structure.clone(), batched_basis_parameters)?;
        let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
        let (output, batched_tangent_output) =
            direct_batched_jvp(self, function, primals, batched_tangents, lane_count)?;
        Differential::from_batched_jvp_output::<Self, Input, TracedOutput::To<DomainValue<Self>>, DomainValue<Self>>(
            input_structure,
            input_parameters,
            output,
            Some(batched_tangent_output),
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
        Self: Domain<Type = ArrayType> + Domain<Type = ArrayType> + 'static,
        DomainValue<Self>: CoordinateValue + 'domain,
        Self::Tangent: CoordinateValue<Coordinate = CoordinateScalar<DomainValue<Self>>>,
        Input: Parameterized<DomainValue<Self>, To<DomainValue<Self>> = Input, ParameterStructure: Debug + PartialEq>,
        Input::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<
                Tracer<
                    LinearizationContext<
                        'domain,
                        TracingContext<'domain, Self>,
                        TracingContext<'domain, Self>,
                    >,
                >,
            >
            + ParameterizedFamily<Tracer<TracingContext<'domain, Self>>>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<DomainValue<Self>>
            + ParameterizedFamily<<Self as Domain>::Constant>
            + ParameterizedFamily<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>
            + ParameterizedFamily<
                DifferentialRow<
                    Input::To<DifferentialBlock<CoordinateScalar<DomainValue<Self>>>>,
                    CoordinateScalar<DomainValue<Self>>,
                >,
            >,
        Input::To<ArrayType>: Parameterized<
                ArrayType,
                To<Tracer<TracingContext<'domain, Self>>> = Input::To<Tracer<TracingContext<'domain, Self>>>,
            >,
        Input::To<Tracer<TracingContext<'domain, Self>>>: Parameterized<
                Tracer<TracingContext<'domain, Self>>,
                To<Tracer<TracingContext<'domain, Self>>> = Input::To<Tracer<TracingContext<'domain, Self>>>,
                To<DomainValue<Self>> = Input,
                To<<Self as Domain>::Constant> = Input::To<<Self as Domain>::Constant>,
                To<
                    Tracer<
                        LinearizationContext<
                            'domain,
                            TracingContext<'domain, Self>,
                            TracingContext<'domain, Self>,
                        >,
                    >,
                > = Input::To<
                    Tracer<
                        LinearizationContext<
                            'domain,
                            TracingContext<'domain, Self>,
                            TracingContext<'domain, Self>,
                        >,
                    >,
                >,
                ParameterStructure: Debug + PartialEq,
            >,
        Input::To<<Self as Domain>::Constant>: Parameterized<
                <Self as Domain>::Constant,
                To<Self::Tangent> = Input::To<Self::Tangent>,
                ParameterStructure = Input::ParameterStructure,
            >,
        <Input::To<Tracer<TracingContext<'domain, Self>>> as Parameterized<
            Tracer<TracingContext<'domain, Self>>,
        >>::To<ArrayType>: Parameterized<
                ArrayType,
                To<Tracer<TracingContext<'domain, Self>>> = Input::To<Tracer<TracingContext<'domain, Self>>>,
            >,
        F: FnOnce(
            Input::To<
                Tracer<
                    LinearizationContext<
                        'domain,
                        TracingContext<'domain, Self>,
                        TracingContext<'domain, Self>,
                    >,
                >,
            >,
        ) -> Tracer<
            LinearizationContext<'domain, TracingContext<'domain, Self>, TracingContext<'domain, Self>>,
        >,
        <Self as Domain>::Operation: Clone
            + InterpretableOperation<ArrayType, DomainValue<Self>>
            + DifferentiableOperation<TracingContext<'domain, Self>>
            + DifferentiableOperation<Self>
            + SupportsFill<ArrayType, f64>
            + SupportsZero<ArrayType>
            + SupportsOne<ArrayType>
            + SupportsAdd<ArrayType>
            + SupportsZero<ArrayType>
            + SupportsOne<ArrayType>
            + SupportsZeroLike<ArrayType>
            + SupportsAdd<ArrayType>
            + 'static,
        AddOperation: InterpretableOperation<ArrayType, Tracer<TracingContext<'domain, Self>>>,
        DirectLinearOperationOf<Self>: BatchableOperation<Tangent<ArrayType, Self::Tangent>>,
        LinearOperationOf<Self>: ResidualizedOperation<Self>,
        <Self as DifferentiationContext>::LinearOperation<
            Tracer<TracingContext<'domain, Self>>,
            crate::tracing_v2::ResidualFactor<ArrayType, Tracer<TracingContext<'domain, Self>>>,
        >: ResidualizedOperation<TracingContext<'domain, Self>>,
        <Self as DifferentiationContext>::LinearOperation<
            Tracer<TracingContext<'domain, Self>>,
            Tracer<TracingContext<'domain, Self>>,
        >:
            InterpretableOperation<ArrayType, Tracer<TracingContext<'domain, Self>>>
            + TransposableOperation<
                ArrayType,
                Tracer<TracingContext<'domain, Self>>,
                <Self as DifferentiationContext>::LinearOperation<
                    Tracer<TracingContext<'domain, Self>>,
                    Tracer<TracingContext<'domain, Self>>,
                >,
            > + SupportsZero<ArrayType>
            + SupportsAdd<ArrayType>,
    {
        let input_structure = primals.parameter_structure();
        let input_parameters = primals.into_parameters().collect::<Vec<_>>();
        let primals = Input::from_parameters(input_structure.clone(), input_parameters.iter().cloned())
            .map_err(ProgramError::from)?;
        let (gradient, gradient_program): (
            Input,
            Program<
                ArrayType,
                <Self as Domain>::Constant,
                <Self as Domain>::Operation,
                Input::To<<Self as Domain>::Constant>,
                Input::To<<Self as Domain>::Constant>,
            >,
        ) = TracingContext::interpret_and_trace(
            self,
            |input: Input::To<Tracer<TracingContext<'domain, Self>>>| {
                let Some(context) = input.parameters().next().map(|tracer| tracer.context().clone()) else {
                    return Err(ProgramError::InvalidInputCount { expected: 1, got: 0 });
                };
                // `interpret_and_trace` fixes its closure error to `ProgramError`, so fold the inner gradient's
                // differentiation error into a program error to flow it through the trace. A non-scalar gradient
                // output cannot occur here for well-formed second-order use (the differentiated function is scalar).
                crate::tracing_v2::DifferentiationContext::value_and_gradient(&context, function, input).map_err(
                    |error| match error {
                        DifferentiationError::Program(error) => error,
                        error => ProgramError::MalformedProgram(error.to_string()),
                    },
                )
            },
            primals,
        )?;
        let (_, pushforward) = self.linearize_program(&gradient_program, input_parameters.clone())?;
        Differential::from_pushforward::<Self, Input, Input, DomainValue<Self>>(
            input_structure,
            input_parameters,
            gradient,
            pushforward,
        )
        .map_err(DifferentiationError::from)
    }
}

impl<D> DifferentiableDomainExtension for D where D: Domain<Type = ArrayType> + DifferentiationContext {}

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

    /// Constructs a [`Differential`] from flat per-lane output-coordinate columns.
    ///
    /// Each entry in `columns` is the full flattened output differential produced by one input-coordinate basis lane.
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
            .collect::<Vec<_>>();
        let input_coordinate_counts = coordinate_counts(input_parameters);
        let input_offsets = coordinate_offsets(&input_coordinate_counts);

        let output_shapes = output_parameters
            .iter()
            .map(|parameter| static_shape(parameter.r#type().as_ref()))
            .collect::<Vec<_>>();
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
                        let lane = input_offset + input_local;
                        let coordinate = output_offset + output_local;
                        values.push(columns[lane][coordinate].clone());
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

    /// Constructs a [`Differential`] from a forward-mode pushforward program by replaying every
    /// input-coordinate basis tangent through the program and rearranging the resulting output
    /// coordinates into per-(output-leaf, input-leaf) [`DifferentialBlock`]s.
    ///
    /// # Parameters
    ///
    ///   - `input_structure`: Placeholder shape of the function's `Input` argument.
    ///   - `input_parameters`: Concrete leaf values of `Input` at the point of linearization, used
    ///     both to derive each input leaf's static shape and to materialize zero-tangent exemplars.
    ///   - `output`: Primal output of the linearized function, consumed to recover its placeholder
    ///     shape and the static shapes of its output leaves.
    ///   - `pushforward`: Staged pushforward program whose inputs and outputs mirror `Input` and
    ///     `Output` reparameterized to the domain's tangent leaf type.
    pub(crate) fn from_pushforward<D, Input, Output, V>(
        input_structure: Input::ParameterStructure,
        input_parameters: Vec<V>,
        output: Output,
        pushforward: Pushforward<D, Input::To<D::Tangent>, Output::To<D::Tangent>>,
    ) -> Result<Self, ProgramError>
    where
        S: Clone,
        D: Domain<Type = ArrayType, Value = V> + DifferentiationContext,
        V: CoordinateValue<Coordinate = S>,
        D::Tangent: CoordinateValue<Coordinate = S>,
        Input:
            Parameterized<V, To<V> = Input, To<DifferentialBlock<S>> = Partials, ParameterStructure: Debug + PartialEq>,
        Output:
            Parameterized<V, To<V> = Output, To<DifferentialRow<Partials, S>> = Rows, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<DifferentialBlock<S>>,
        Output::Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<DifferentialRow<Partials, S>>,
        Partials: Parameterized<DifferentialBlock<S>, ParameterStructure = Input::ParameterStructure>,
        Rows: Parameterized<DifferentialRow<Partials, S>, ParameterStructure = Output::ParameterStructure>,
        DirectLinearOperationOf<D>: BatchableOperation<Tangent<ArrayType, D::Tangent>>,
        LinearOperationOf<D>: ResidualizedOperation<D>,
    {
        let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
        let lane_count: usize = input_coordinate_counts.iter().sum();

        let tangent_parameters = input_parameters
            .iter()
            .map(|parameter| D::Tangent::zero(parameter.r#type().as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let batched_basis_parameters = batched_standard_basis::<D::Tangent>(tangent_parameters.as_slice(), lane_count)?;

        let output_structure = output.parameter_structure();
        let output_parameters = output.into_parameters().collect::<Vec<_>>();
        let tangent_output_parameters = output_parameters
            .iter()
            .map(|parameter| D::Tangent::zero(parameter.r#type().as_ref()))
            .collect::<Result<Vec<_>, _>>()?;

        let columns = if lane_count == 0 {
            Vec::new()
        } else {
            let pushforward_program = pushforward.instantiate_program()?;
            let batched_output = pushforward_program.interpret_with(
                batched_basis_parameters,
                |_, constant: &D::Tangent| {
                    Ok::<_, ProgramError>(ArrayBatch::unbatched(Tangent::Value(constant.clone())))
                },
                |instruction, inputs: &[ArrayBatch<Tangent<ArrayType, D::Tangent>>]| {
                    BatchableOperation::<Tangent<ArrayType, D::Tangent>>::batch(instruction.operation(), &(), inputs)
                },
            )?;
            unstack_batched_tangent_coordinates::<D::Tangent>(
                batched_output,
                tangent_output_parameters.as_slice(),
                lane_count,
            )?
        };

        Self::from_coordinate_columns::<Input, Output, V>(
            input_structure,
            input_parameters.as_slice(),
            output_structure,
            output_parameters.as_slice(),
            columns,
        )
    }

    /// Constructs a [`Differential`] from one batched JVP tangent output.
    ///
    /// `batched_tangent_output` carries all input-coordinate basis responses stacked along a
    /// leading lane axis. A missing value is only valid when there are zero input-coordinate
    /// lanes, in which case every differential block is empty.
    pub(crate) fn from_batched_jvp_output<D, Input, Output, V>(
        input_structure: Input::ParameterStructure,
        input_parameters: Vec<V>,
        output: Output,
        batched_tangent_output: Option<Output::To<D::Tangent>>,
    ) -> Result<Self, ProgramError>
    where
        S: Clone,
        D: Domain<Type = ArrayType, Value = V> + DifferentiationContext,
        V: CoordinateValue<Coordinate = S>,
        D::Tangent: CoordinateValue<Coordinate = S>,
        Input:
            Parameterized<V, To<V> = Input, To<DifferentialBlock<S>> = Partials, ParameterStructure: Debug + PartialEq>,
        Output:
            Parameterized<V, To<V> = Output, To<DifferentialRow<Partials, S>> = Rows, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<DifferentialBlock<S>>,
        Output::Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<DifferentialRow<Partials, S>>,
        Partials: Parameterized<DifferentialBlock<S>, ParameterStructure = Input::ParameterStructure>,
        Rows: Parameterized<DifferentialRow<Partials, S>, ParameterStructure = Output::ParameterStructure>,
    {
        let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
        let lane_count: usize = input_coordinate_counts.iter().sum();

        let output_structure = output.parameter_structure();
        let output_parameters = output.into_parameters().collect::<Vec<_>>();

        let columns = if lane_count == 0 {
            Vec::new()
        } else {
            let batched_tangent_output =
                batched_tangent_output.ok_or(ProgramError::InvalidInputCount { expected: 1, got: 0 })?;
            unstack_tangent_coordinates::<D::Tangent, V>(
                batched_tangent_output.into_parameters().collect::<Vec<_>>(),
                output_parameters.as_slice(),
                lane_count,
            )?
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

/// Extracts the static shape of `array_type` as a `Vec<usize>`. Panics if any dimension is
/// dynamic, since differential materialization only operates on concrete primal values.
fn static_shape(array_type: &ArrayType) -> Vec<usize> {
    array_type
        .shape()
        .dimensions()
        .iter()
        .map(|size| match size {
            Size::Static(value) => *value,
            Size::Dynamic(_) => panic!("differential materialization requires a fully static array shape"),
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

/// Builds one leaf's lane values for a stacked standard basis.
///
/// The returned vector has `lane_count` entries and contains the appropriate one-hot basis value for this leaf's
/// coordinate range and a zero-like value elsewhere.
fn standard_basis_leaf_lane_values<V: CoordinateValue>(
    parameter: &V,
    leaf_start: usize,
    leaf_count: usize,
    lane_count: usize,
) -> Vec<V> {
    let leaf_basis = parameter.coordinate_basis();
    let leaf_zero = parameter.zero_like();

    (0..lane_count)
        .map(|lane| {
            if lane >= leaf_start && lane < leaf_start + leaf_count {
                leaf_basis[lane - leaf_start].clone()
            } else {
                leaf_zero.clone()
            }
        })
        .collect()
}

/// Builds the standard basis metadata shared by forward- and reverse-mode differential materialization.
fn standard_basis_metadata<V: CoordinateValue>(parameters: &[V], lane_count: usize) -> (Vec<usize>, Vec<usize>) {
    let counts = coordinate_counts(parameters);
    let offsets = coordinate_offsets(&counts);
    debug_assert_eq!(offsets.last().copied().unwrap_or(0), lane_count, "lane count must equal total coord count");
    (counts, offsets)
}

/// Builds the standard basis for the coordinate space of a [`Parameterized`] tangent value,
/// packed per-leaf into [`ArrayBatch`]es over [`Tangent`] runtime values for batched program
/// interpretation.
///
/// For each input leaf, the returned [`ArrayBatch`] carries a `lane_count`-lane stacked tangent
/// whose lane `k` is the one-hot basis vector at position `k - leaf_offset[i]` when `k` falls
/// within leaf `i`'s coordinate range, and `zero_like` otherwise. The batch is wrapped as
/// [`Tangent::Value`], so the per-operation symbolic-zero short-circuit applies only when an
/// upstream operation produces a structurally-zero batched intermediate.
fn batched_standard_basis<V: CoordinateValue>(
    parameters: &[V],
    lane_count: usize,
) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, ProgramError> {
    let (counts, offsets) = standard_basis_metadata(parameters, lane_count);
    parameters
        .iter()
        .enumerate()
        .map(|(leaf_index, parameter)| -> Result<_, ProgramError> {
            let leaf_type = parameter.r#type().into_owned();
            let lane_values =
                standard_basis_leaf_lane_values(parameter, offsets[leaf_index], counts[leaf_index], lane_count);
            let stacked = V::stack(lane_values)?;

            let mut batched_dimensions = Vec::with_capacity(leaf_type.shape().dimensions().len() + 1);
            batched_dimensions.push(Size::Static(lane_count));
            batched_dimensions.extend(leaf_type.shape().dimensions().iter().copied());
            let batched_type = ArrayType::new(
                leaf_type.data_type(),
                Shape::new(batched_dimensions),
                leaf_type.layout().cloned(),
                None,
            )
            .map_err(|error| TypeError { message: error.to_string() })?;

            ArrayBatch::new(batched_type, Tangent::Value(stacked), Some(0))
        })
        .collect()
}

/// Builds the standard basis for the coordinate space of a [`Parameterized`] tangent value,
/// packed per-leaf as ordinary stacked tangent values.
fn stacked_standard_basis<V: CoordinateValue>(parameters: &[V], lane_count: usize) -> Result<Vec<V>, ProgramError> {
    let (counts, offsets) = standard_basis_metadata(parameters, lane_count);
    parameters
        .iter()
        .enumerate()
        .map(|(leaf_index, parameter)| {
            V::stack(standard_basis_leaf_lane_values(parameter, offsets[leaf_index], counts[leaf_index], lane_count))
        })
        .collect()
}

/// Extracts per-lane flat output coordinates from a structured batched-Tangent program output.
///
/// For each output leaf, structurally-zero batches contribute a run of `leaf_coord_count` zero
/// coordinates per lane (sourced from `leaf_exemplar.zero_like().coordinates()`), and concrete
/// [`Tangent::Value`] batches contribute `leaf_coord_count` coordinates per lane carved out of
/// the batched leaf value's flat coordinate buffer.
fn unstack_batched_tangent_coordinates<V>(
    output_batches: Vec<ArrayBatch<Tangent<ArrayType, V>>>,
    output_parameters: &[V],
    lane_count: usize,
) -> Result<Vec<Vec<V::Coordinate>>, ProgramError>
where
    V: CoordinateValue,
{
    assert_eq!(output_batches.len(), output_parameters.len(), "output parameter count mismatch");
    let mut columns: Vec<Vec<V::Coordinate>> = (0..lane_count).map(|_| Vec::new()).collect();
    for (leaf_batch, leaf_exemplar) in output_batches.into_iter().zip(output_parameters.iter()) {
        let leaf_coord_count = leaf_exemplar.coordinate_count();
        match leaf_batch.into_value() {
            Tangent::Zero(_) => {
                let zero_coords = leaf_exemplar.zero_like().coordinates();
                assert_eq!(zero_coords.len(), leaf_coord_count);
                for column in columns.iter_mut() {
                    column.extend_from_slice(zero_coords.as_slice());
                }
            }
            Tangent::Value(value) => {
                let all_coords = value.coordinates();
                assert_eq!(
                    all_coords.len(),
                    lane_count * leaf_coord_count,
                    "expected {lane_count} lanes x {leaf_coord_count} coords per output leaf",
                );
                for (lane, lane_coords) in all_coords.chunks(leaf_coord_count).enumerate() {
                    columns[lane].extend_from_slice(lane_coords);
                }
            }
        }
    }
    Ok(columns)
}

/// Extracts per-lane flat output coordinates from stacked tangent output values.
fn unstack_tangent_coordinates<TangentValue, OutputValue>(
    output_values: Vec<TangentValue>,
    output_parameters: &[OutputValue],
    lane_count: usize,
) -> Result<Vec<Vec<TangentValue::Coordinate>>, ProgramError>
where
    TangentValue: CoordinateValue,
    OutputValue: CoordinateValue<Coordinate = TangentValue::Coordinate>,
{
    assert_eq!(output_values.len(), output_parameters.len(), "output parameter count mismatch");
    let mut columns: Vec<Vec<TangentValue::Coordinate>> = (0..lane_count).map(|_| Vec::new()).collect();
    for (value, leaf_exemplar) in output_values.into_iter().zip(output_parameters.iter()) {
        let leaf_coordinate_count = leaf_exemplar.coordinate_count();
        let all_coordinates = value.coordinates();
        assert_eq!(
            all_coordinates.len(),
            lane_count * leaf_coordinate_count,
            "expected {lane_count} lanes x {leaf_coordinate_count} coords per output leaf",
        );
        for (lane, lane_coordinates) in all_coordinates.chunks(leaf_coordinate_count).enumerate() {
            columns[lane].extend_from_slice(lane_coordinates);
        }
    }
    Ok(columns)
}

/// Materializes a structured [`Differential`] using reverse-mode differentiation.
///
/// [`jacrev`] replays all output-coordinate basis cotangents through the pullback and reassembles
/// the resulting input coordinates into per-(output-leaf, input-leaf) [`DifferentialBlock`]s.
#[allow(private_bounds)]
pub fn jacrev<'domain, D, F, Input, TracedOutput>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<Jacobian<Input, TracedOutput::To<DomainValue<D>>, DomainValue<D>>, ProgramError>
where
    D: Domain<Type = ArrayType> + DifferentiationContext,
    DomainValue<D>: CoordinateValue + 'domain,
    D::Tangent: CoordinateValue<Coordinate = CoordinateScalar<DomainValue<D>>>,
    Input: Parameterized<DomainValue<D>, To<DomainValue<D>> = Input, ParameterStructure: Debug + PartialEq>,
    TracedOutput: Parameterized<LinearizationTracer<'domain, D>, ParameterStructure: Debug + PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<LinearizationTracer<'domain, D>>
        + ParameterizedFamily<DifferentialBlock<CoordinateScalar<DomainValue<D>>>>,
    TracedOutput::Family: ParameterizedFamily<DomainValue<D>>
        + ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<LinearizationTracer<'domain, D>>
        + ParameterizedFamily<
            DifferentialRow<
                Input::To<DifferentialBlock<CoordinateScalar<DomainValue<D>>>>,
                CoordinateScalar<DomainValue<D>>,
            >,
        >,
    TracedOutput::To<DomainValue<D>>: Parameterized<
            DomainValue<D>,
            To<DomainValue<D>> = TracedOutput::To<DomainValue<D>>,
            To<D::Tangent> = TracedOutput::To<D::Tangent>,
            To<
                DifferentialRow<
                    Input::To<DifferentialBlock<CoordinateScalar<DomainValue<D>>>>,
                    CoordinateScalar<DomainValue<D>>,
                >,
            > = TracedOutput::To<
                DifferentialRow<
                    Input::To<DifferentialBlock<CoordinateScalar<DomainValue<D>>>>,
                    CoordinateScalar<DomainValue<D>>,
                >,
            >,
            ParameterStructure: Debug + PartialEq,
        >,
    F: FnOnce(Input::To<LinearizationTracer<'domain, D>>) -> Result<TracedOutput, ProgramError>,
    <D as Domain>::Operation: DifferentiableOperation<D>,
    DirectLinearOperationOf<D>: BatchableOperation<Tangent<ArrayType, D::Tangent>>
        + TransposableOperation<ArrayType, D::Tangent, DirectLinearOperationOf<D>>
        + SupportsZero<ArrayType>
        + SupportsAdd<ArrayType>,
    LinearOperationOf<D>: ResidualizedOperation<D>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_shapes = input_parameters
        .iter()
        .map(|parameter| static_shape(parameter.r#type().as_ref()))
        .collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let input_offsets = coordinate_offsets(&input_coordinate_counts);
    let tangent_input_parameters = input_parameters
        .iter()
        .map(|parameter| D::Tangent::zero(parameter.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;

    let primals = Input::from_parameters(input_structure.clone(), input_parameters)?;
    let (output, pullback) = domain.vjp(function, primals)?;
    let output_structure = output.parameter_structure();
    let output_parameters = output.into_parameters().collect::<Vec<_>>();
    let output_shapes = output_parameters
        .iter()
        .map(|parameter| static_shape(parameter.r#type().as_ref()))
        .collect::<Vec<_>>();
    let output_coordinate_counts = coordinate_counts(output_parameters.as_slice());
    let output_offsets = coordinate_offsets(&output_coordinate_counts);
    let lane_count: usize = output_coordinate_counts.iter().sum();
    let cotangent_parameters = output_parameters
        .iter()
        .map(|parameter| D::Tangent::zero(parameter.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let batched_basis_parameters = batched_standard_basis::<D::Tangent>(cotangent_parameters.as_slice(), lane_count)?;

    let rows = if lane_count == 0 {
        Vec::new()
    } else {
        let batched_input = pullback.interpret_with(
            batched_basis_parameters,
            |_, constant: &D::Tangent| Ok::<_, ProgramError>(ArrayBatch::unbatched(Tangent::Value(constant.clone()))),
            |instruction, inputs: &[ArrayBatch<Tangent<ArrayType, D::Tangent>>]| {
                BatchableOperation::<Tangent<ArrayType, D::Tangent>>::batch(instruction.operation(), &(), inputs)
            },
        )?;
        unstack_batched_tangent_coordinates::<D::Tangent>(
            batched_input,
            tangent_input_parameters.as_slice(),
            lane_count,
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
                    let lane = output_offset + output_local;
                    let coordinate = input_offset + input_local;
                    values.push(rows[lane][coordinate].clone());
                }
            }

            blocks.push(DifferentialBlock::new(output_shape.clone(), input_shape.clone(), values));
        }

        let partials = <Input::To<DifferentialBlock<CoordinateScalar<DomainValue<D>>>>>::from_parameters(
            input_structure.clone(),
            blocks,
        )?;
        rows_list.push(DifferentialRow::new(partials));
    }

    let outer_rows = <TracedOutput::To<
        DifferentialRow<
            Input::To<DifferentialBlock<CoordinateScalar<DomainValue<D>>>>,
            CoordinateScalar<DomainValue<D>>,
        >,
    >>::from_parameters(output_structure, rows_list)?;

    Ok(Differential::new(outer_rows))
}
