use std::fmt::Debug;
use std::marker::PhantomData;

use ryft_macros::Parameter;

use super::*;

use crate::differentiation::{LinearOperation, Tangent};
use crate::operations::constants::{SupportsOne, SupportsZero};
use crate::parameters::{Parameter, ParameterPath};
use crate::tracing::contexts::TracingContext;
use crate::tracing::domains::{DomainTracer, Tracer};
use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation};
use crate::tracing_v2::{LinearizationContext, LinearizationTracer};
use crate::types::{Shape, Size, TypeError};

/// Leaf type that can be materialized into a dense finite-dimensional coordinate representation.
///
/// Structured differential materialization only makes sense for leaf types with a finite, explicit
/// basis. [`CoordinateValue`] is the bridge from the generic tracing world into that coordinate-based
/// view: it teaches the differential helpers how many coordinates a leaf contributes, what basis
/// vectors to probe with, and how to flatten outputs back into numeric entries.
pub trait CoordinateValue: Traceable<ArrayType> + ZeroLike + OneLike + Zero<ArrayType> + One<ArrayType> {
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
    fn stack(values: Vec<Self>) -> Result<Self, TracingError>;
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

/// Dense derivative materialization helpers for differentiable array domains.
///
/// This extension trait keeps the core [`DifferentiableDomain`] contract focused on primitive
/// linearization and AD transforms while providing structured Jacobian and Hessian materialization
/// for domains whose values expose finite coordinate bases.
pub trait DifferentiableDomainExtension: DifferentiableDomain<Type = ArrayType> {
    /// Materializes a structured [`Jacobian`] using forward-mode differentiation.
    ///
    /// The returned [`Jacobian`] is a nested [`Parameterized`] value whose outer family mirrors
    /// the function's output and whose inner family mirrors its input. Each innermost leaf is a
    /// [`DifferentialBlock`] holding the partial derivatives of one output leaf with respect to
    /// one input leaf.
    #[allow(private_bounds)]
    fn jacfwd<'domain, F, Input, Output, V>(
        &'domain self,
        function: F,
        primal: Input,
    ) -> Result<Jacobian<Input, Output, V>, TracingError>
    where
        Self: DifferentiableDomain<Type = ArrayType, Value = V> + 'static,
        V: CoordinateValue + 'domain,
        Self::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
        Output: Parameterized<V, To<V> = Output, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<LinearizationTracer<'domain, Self>>
            + ParameterizedFamily<DifferentialBlock<V::Coordinate>>,
        Output::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<LinearizationTracer<'domain, Self>>
            + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
        Output::To<LinearizationTracer<'domain, Self>>:
            Parameterized<LinearizationTracer<'domain, Self>, To<V> = Output>,
        F: FnOnce(
            Input::To<LinearizationTracer<'domain, Self>>,
        ) -> Result<Output::To<LinearizationTracer<'domain, Self>>, TracingError>,
        Self::Operation: DifferentiableOperation<Self>,
        <Self as DifferentiableDomain>::LinearOperationCarrier: BatchableOperation<Tangent<ArrayType, Self::Tangent>>,
    {
        let input_structure = primal.parameter_structure();
        let input_parameters = primal.into_parameters().collect::<Vec<_>>();
        let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
        let (output, pushforward) = self.linearize(function, primals)?;
        Differential::from_pushforward::<Self, Input, Output, V>(input_structure, input_parameters, output, pushforward)
    }

    /// Materializes a structured [`Hessian`] of a scalar-output function.
    ///
    /// Hessian evaluation is expressed internally as a forward-mode [`Jacobian`] over a
    /// reverse-mode gradient transform.
    #[allow(private_bounds)]
    fn hessian<'domain, F, Input, V>(
        &'domain self,
        function: F,
        primals: Input,
    ) -> Result<Hessian<Input, V>, TracingError>
    where
        Self: DifferentiableDomain<Type = ArrayType, Value = V>
            + DifferentiableTracingDomain<Type = ArrayType, Value = V>
            + 'static,
        V: CoordinateValue + 'domain,
        Self::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
        Input::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<Tracer<LinearizationContext<'domain, TracingContext<'domain, Self>, TracingContext<'domain, Self>>>>
            + ParameterizedFamily<DomainTracer<'domain, Self>>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<V>
            + ParameterizedFamily<DifferentialBlock<V::Coordinate>>
            + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
        Input::To<ArrayType>:
            Parameterized<ArrayType, To<DomainTracer<'domain, Self>> = Input::To<DomainTracer<'domain, Self>>>,
        Input::To<DomainTracer<'domain, Self>>: Parameterized<
                DomainTracer<'domain, Self>,
                To<DomainTracer<'domain, Self>> = Input::To<DomainTracer<'domain, Self>>,
                To<V> = Input,
                To<Tracer<LinearizationContext<'domain, TracingContext<'domain, Self>, TracingContext<'domain, Self>>>> = Input::To<
                    Tracer<LinearizationContext<'domain, TracingContext<'domain, Self>, TracingContext<'domain, Self>>>,
                >,
                ParameterStructure: Debug + PartialEq,
            >,
        <Input::To<DomainTracer<'domain, Self>> as Parameterized<DomainTracer<'domain, Self>>>::To<ArrayType>:
            Parameterized<ArrayType, To<DomainTracer<'domain, Self>> = Input::To<DomainTracer<'domain, Self>>>,
        F: FnOnce(
            Input::To<Tracer<LinearizationContext<'domain, TracingContext<'domain, Self>, TracingContext<'domain, Self>>>>,
        ) -> Tracer<LinearizationContext<'domain, TracingContext<'domain, Self>, TracingContext<'domain, Self>>>,
        Self::Operation: Clone
            + InterpretableOperation<ArrayType, V>
            + DifferentiableOperation<TracingContext<'domain, Self>>
            + DifferentiableOperation<Self>
            + SupportsZero<ArrayType, V>
            + SupportsOne<ArrayType, V>
            + SupportsZeroLike<ArrayType, V>
            + SupportsAdd<ArrayType, V>
            + 'static,
        AddOperation: InterpretableOperation<ArrayType, DomainTracer<'domain, Self>>,
        <Self as DifferentiableDomain>::LinearOperationCarrier: BatchableOperation<Tangent<ArrayType, Self::Tangent>>,
        <Self as DifferentiableTracingDomain>::LinearOperationCarrier<'domain>: InterpretableOperation<
                ArrayType,
                DomainTracer<'domain, Self>,
            > + LinearOperation<
                ArrayType,
                DomainTracer<'domain, Self>,
                <Self as DifferentiableTracingDomain>::LinearOperationCarrier<'domain>,
            > + SupportsZero<ArrayType, DomainTracer<'domain, Self>>,
    {
        let input_structure = primals.parameter_structure();
        let input_parameters = primals.into_parameters().collect::<Vec<_>>();
        let primals = Input::from_parameters(input_structure.clone(), input_parameters.iter().cloned())?;
        let (gradient, gradient_program): (Input, Program<ArrayType, V, Self::Operation, Input, Input>) = self
            .interpret_and_trace(
                |input: Input::To<DomainTracer<'domain, Self>>| {
                    let Some(context) = input.parameters().next().map(|tracer| tracer.context().clone()) else {
                        return Err(TracingError::InvalidInputCount { expected: 1, got: 0 });
                    };
                    let gradient =
                        crate::tracing_v2::DifferentiableContext::value_and_gradient(&context, function, input)?;
                    Ok(gradient)
                },
                primals,
            )?;
        let (_, pushforward) = self.linearize_program(&gradient_program, input_parameters.clone())?;
        Differential::from_pushforward::<Self, Input, Input, V>(
            input_structure,
            input_parameters,
            gradient,
            pushforward,
        )
    }
}

impl<D: DifferentiableDomain<Type = ArrayType>> DifferentiableDomainExtension for D {}

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
        pushforward: Program<
            ArrayType,
            D::Tangent,
            D::LinearOperationCarrier,
            Input::To<D::Tangent>,
            Output::To<D::Tangent>,
        >,
    ) -> Result<Self, TracingError>
    where
        S: Clone,
        D: DifferentiableDomain<Type = ArrayType, Value = V>,
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
        D::LinearOperationCarrier: BatchableOperation<Tangent<ArrayType, D::Tangent>>,
    {
        let input_shapes = input_parameters
            .iter()
            .map(|parameter| static_shape(parameter.r#type().as_ref()))
            .collect::<Vec<_>>();
        let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
        let input_offsets = coordinate_offsets(&input_coordinate_counts);
        let lane_count: usize = input_coordinate_counts.iter().sum();

        let tangent_parameters = input_parameters
            .iter()
            .map(|parameter| D::Tangent::zero(parameter.r#type().as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        let batched_basis_parameters = batched_standard_basis::<D::Tangent>(tangent_parameters.as_slice(), lane_count)?;

        let output_structure = output.parameter_structure();
        let output_parameters = output.into_parameters().collect::<Vec<_>>();
        let output_shapes = output_parameters
            .iter()
            .map(|parameter| static_shape(parameter.r#type().as_ref()))
            .collect::<Vec<_>>();
        let output_coordinate_counts = coordinate_counts(output_parameters.as_slice());
        let output_offsets = coordinate_offsets(&output_coordinate_counts);
        let tangent_output_parameters = output_parameters
            .iter()
            .map(|parameter| D::Tangent::zero(parameter.r#type().as_ref()))
            .collect::<Result<Vec<_>, _>>()?;

        let columns = if lane_count == 0 {
            Vec::new()
        } else {
            let batched_output = pushforward.interpret_with(
                batched_basis_parameters,
                |_, constant: &D::Tangent| {
                    Ok::<_, TracingError>(ArrayBatch::unbatched(Tangent::Value(constant.clone())))
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
fn coordinate_counts<V>(parameters: &[V]) -> Vec<usize>
where
    V: CoordinateValue,
{
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

/// Builds the standard basis for the coordinate space of a [`Parameterized`] tangent value,
/// packed per-leaf into [`ArrayBatch`]es over [`Tangent`] runtime values for batched program
/// interpretation.
///
/// For each input leaf, the returned [`ArrayBatch`] carries a `lane_count`-lane stacked tangent
/// whose lane `k` is the one-hot basis vector at position `k - leaf_offset[i]` when `k` falls
/// within leaf `i`'s coordinate range, and `zero_like` otherwise. The batch is wrapped as
/// [`Tangent::Value`], so the per-operation symbolic-zero short-circuit applies only when an
/// upstream operation produces a structurally-zero batched intermediate.
fn batched_standard_basis<V>(
    parameters: &[V],
    lane_count: usize,
) -> Result<Vec<ArrayBatch<Tangent<ArrayType, V>>>, TracingError>
where
    V: CoordinateValue,
{
    let counts = coordinate_counts(parameters);
    let offsets = coordinate_offsets(&counts);
    debug_assert_eq!(offsets.last().copied().unwrap_or(0), lane_count, "lane count must equal total coord count");

    parameters
        .iter()
        .enumerate()
        .map(|(leaf_index, parameter)| -> Result<_, TracingError> {
            let leaf_type = parameter.r#type().into_owned();
            let leaf_start = offsets[leaf_index];
            let leaf_count = counts[leaf_index];

            let leaf_basis = parameter.coordinate_basis();
            let leaf_zero = parameter.zero_like();

            let lane_values: Vec<V> = (0..lane_count)
                .map(|lane| {
                    if lane >= leaf_start && lane < leaf_start + leaf_count {
                        leaf_basis[lane - leaf_start].clone()
                    } else {
                        leaf_zero.clone()
                    }
                })
                .collect();

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
) -> Result<Vec<Vec<V::Coordinate>>, TracingError>
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

/// Materializes a structured [`Differential`] using reverse-mode differentiation.
///
/// [`jacrev`] replays all output-coordinate basis cotangents through the pullback and reassembles
/// the resulting input coordinates into per-(output-leaf, input-leaf) [`DifferentialBlock`]s.
#[allow(private_bounds)]
pub fn jacrev<'domain, D, F, Input, Output, V>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<
    Differential<
        Output::To<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
        Input::To<DifferentialBlock<V::Coordinate>>,
        V::Coordinate,
    >,
    TracingError,
>
where
    D: DifferentiableDomain<Type = ArrayType, Value = V> + 'static,
    V: CoordinateValue + 'domain,
    D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
    Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<V, To<V> = Output, ParameterStructure: Debug + PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<LinearizationTracer<'domain, D>>
        + ParameterizedFamily<DifferentialBlock<V::Coordinate>>,
    Output::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<LinearizationTracer<'domain, D>>
        + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
    Output::To<LinearizationTracer<'domain, D>>: Parameterized<LinearizationTracer<'domain, D>, To<V> = Output>,
    F: FnOnce(
        Input::To<LinearizationTracer<'domain, D>>,
    ) -> Result<Output::To<LinearizationTracer<'domain, D>>, TracingError>,
    D::Operation: DifferentiableOperation<D>,
    D::LinearOperationCarrier: BatchableOperation<Tangent<ArrayType, D::Tangent>>
        + LinearOperation<ArrayType, D::Tangent, D::LinearOperationCarrier>,
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
    let (output, pullback) = domain.vjp::<F, Input, Output, V>(function, primals)?;
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
            |_, constant: &D::Tangent| Ok::<_, TracingError>(ArrayBatch::unbatched(Tangent::Value(constant.clone()))),
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

        let partials = <Input::To<DifferentialBlock<V::Coordinate>>>::from_parameters(input_structure.clone(), blocks)?;
        rows_list.push(DifferentialRow::new(partials));
    }

    let outer_rows =
        <Output::To<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>>::from_parameters(
            output_structure,
            rows_list,
        )?;

    Ok(Differential::new(outer_rows))
}
