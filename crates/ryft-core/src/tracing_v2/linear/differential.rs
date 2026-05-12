use std::fmt::Debug;
use std::marker::PhantomData;

use ryft_macros::Parameter;

use super::*;

use crate::parameters::{Parameter, ParameterPath};
use crate::tracing_v2::BatchingError;
use crate::tracing_v2::batching::{ReferenceBatch, interpret_reference_batched_program, reference_stack};
use crate::types::Size;

/// Forward-mode structured-differential transform.
pub(crate) struct JacFwd<F> {
    /// Function whose differential is materialized.
    function: F,
}

impl<F> JacFwd<F> {
    /// Creates a forward-mode structured-differential transform for `function`.
    #[inline]
    pub(crate) const fn new(function: F) -> Self {
        Self { function }
    }

    /// Evaluates this differential transform at `primals`.
    #[allow(private_bounds)]
    pub(crate) fn evaluate<'domain, D, Input, Output, V>(
        self,
        domain: &'domain D,
        primals: Input,
    ) -> Result<
        Differential<
            <Output as Parameterized<V>>::To<
                DifferentialRow<<Input as Parameterized<V>>::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>,
            >,
            <Input as Parameterized<V>>::To<DifferentialBlock<V::Coordinate>>,
            V::Coordinate,
        >,
        TracingError,
    >
    where
        D: DifferentiableDomain<Type = ArrayType, Value = V> + 'static,
        V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
        D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
        Output: Parameterized<V, To<V> = Output, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<D::Tangent>
            + ParameterizedFamily<ReferenceBatch<D::Tangent>>
            + ParameterizedFamily<Tracer<'domain, D>>
            + ParameterizedFamily<DifferentialBlock<V::Coordinate>>,
        Output::Family: ParameterizedFamily<D::Tangent>
            + ParameterizedFamily<ReferenceBatch<D::Tangent>>
            + ParameterizedFamily<Tracer<'domain, D>>
            + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
        Output::To<Tracer<'domain, D>>: Parameterized<Tracer<'domain, D>, To<V> = Output>,
        F: FnOnce(Input::To<Tracer<'domain, D>>) -> Result<Output::To<Tracer<'domain, D>>, TracingError>,
        D::OperationCarrier: DifferentiableOperation<D>,
    {
        jacfwd_at::<D, F, Input, Output, V>(domain, self.function, primals)
    }
}

impl<F> JacFwd<Grad<F>> {
    /// Evaluates a forward-mode differential of a reverse-mode gradient at `primals`.
    #[allow(private_bounds)]
    pub(crate) fn evaluate_gradient<'domain, D, Input, V>(
        self,
        domain: &'domain D,
        primals: Input,
    ) -> Result<
        Differential<
            <Input as Parameterized<V>>::To<
                DifferentialRow<<Input as Parameterized<V>>::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>,
            >,
            <Input as Parameterized<V>>::To<DifferentialBlock<V::Coordinate>>,
            V::Coordinate,
        >,
        TracingError,
    >
    where
        D: DifferentiableDomain<Type = ArrayType, Value = V>
            + DifferentiableTracingDomain<Type = ArrayType, Value = V>
            + 'static,
        V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
        D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
        Input::Family: ParameterizedFamily<D::Tangent>
            + ParameterizedFamily<ReferenceBatch<D::Tangent>>
            + ParameterizedFamily<Tracer<'domain, D>>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<V>
            + ParameterizedFamily<DifferentialBlock<V::Coordinate>>
            + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
        Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
        Input::To<Tracer<'domain, D>>:
            Parameterized<Tracer<'domain, D>, To<V> = Input, ParameterStructure: Debug + PartialEq>,
        <Input::To<Tracer<'domain, D>> as Parameterized<Tracer<'domain, D>>>::To<ArrayType>:
            Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
        F: FnOnce(Input::To<Tracer<'domain, D>>) -> Tracer<'domain, D>,
        D::OperationCarrier: Clone
            + InterpretableOperation<ArrayType, V>
            + DifferentiableOperation<TracingContext<'domain, D>>
            + DifferentiableOperation<D>
            + SupportsZeroLike<ArrayType, V>
            + SupportsAdd<ArrayType, V>
            + 'static,
        AddOperation: InterpretableOperation<ArrayType, Tracer<'domain, D>>,
    {
        hessian_at(domain, self.function.into_function(), primals)
    }
}

/// Scalar-output Hessian transform.
pub(crate) struct Hessian<F> {
    /// Function whose Hessian is materialized.
    function: F,
}

impl<F> Hessian<F> {
    /// Creates a Hessian transform for `function`.
    #[inline]
    pub(crate) const fn new(function: F) -> Self {
        Self { function }
    }

    /// Evaluates this Hessian transform at `primals`.
    #[allow(private_bounds)]
    pub(crate) fn evaluate<'domain, D, Input, V>(
        self,
        domain: &'domain D,
        primals: Input,
    ) -> Result<
        Differential<
            <Input as Parameterized<V>>::To<
                DifferentialRow<<Input as Parameterized<V>>::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>,
            >,
            <Input as Parameterized<V>>::To<DifferentialBlock<V::Coordinate>>,
            V::Coordinate,
        >,
        TracingError,
    >
    where
        D: DifferentiableDomain<Type = ArrayType, Value = V>
            + DifferentiableTracingDomain<Type = ArrayType, Value = V>
            + 'static,
        V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
        D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
        Input::Family: ParameterizedFamily<D::Tangent>
            + ParameterizedFamily<ReferenceBatch<D::Tangent>>
            + ParameterizedFamily<Tracer<'domain, D>>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<V>
            + ParameterizedFamily<DifferentialBlock<V::Coordinate>>
            + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
        Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
        Input::To<Tracer<'domain, D>>:
            Parameterized<Tracer<'domain, D>, To<V> = Input, ParameterStructure: Debug + PartialEq>,
        <Input::To<Tracer<'domain, D>> as Parameterized<Tracer<'domain, D>>>::To<ArrayType>:
            Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
        F: FnOnce(Input::To<Tracer<'domain, D>>) -> Tracer<'domain, D>,
        D::OperationCarrier: Clone
            + InterpretableOperation<ArrayType, V>
            + DifferentiableOperation<TracingContext<'domain, D>>
            + DifferentiableOperation<D>
            + SupportsZeroLike<ArrayType, V>
            + SupportsAdd<ArrayType, V>
            + 'static,
        AddOperation: InterpretableOperation<ArrayType, Tracer<'domain, D>>,
    {
        JacFwd::new(Grad::new(self.function)).evaluate_gradient(domain, primals)
    }
}

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
}

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
        .shape
        .dimensions
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

/// Unpacks a reference-batched program output into one coordinate vector per lane.
fn flatten_batched_coordinates<Value, V>(
    value: Value::To<ReferenceBatch<V>>,
    lane_count: usize,
) -> Result<Vec<Vec<V::Coordinate>>, TracingError>
where
    Value: Parameterized<V, Family: ParameterizedFamily<ReferenceBatch<V>>>,
    V: CoordinateValue,
{
    let mut lane_coordinates = (0..lane_count).map(|_| Vec::new()).collect::<Vec<_>>();
    for batch in value.into_parameters() {
        if batch.len() != lane_count {
            return Err(BatchingError::MismatchedBatchSize.into());
        }
        for (lane_index, parameter) in batch.into_lanes().into_iter().enumerate() {
            lane_coordinates[lane_index].extend(parameter.coordinates());
        }
    }
    Ok(lane_coordinates)
}

/// Builds the standard basis for the coordinate space of a [`Parameterized`] tangent value with
/// the provided per-leaf parameters.
fn standard_basis<Value, V>(structure: &Value::ParameterStructure, parameters: &[V]) -> Result<Vec<Value>, TracingError>
where
    Value: Parameterized<V>,
    V: CoordinateValue,
{
    let zero_parameters = parameters.iter().map(ZeroLike::zero_like).collect::<Vec<_>>();
    let mut basis = Vec::new();
    for (parameter_index, parameter) in parameters.iter().enumerate() {
        for basis_vector in parameter.coordinate_basis() {
            let mut tangent_parameters = zero_parameters.clone();
            tangent_parameters[parameter_index] = basis_vector;
            basis.push(Value::from_parameters(structure.clone(), tangent_parameters.into_iter())?);
        }
    }
    Ok(basis)
}

/// Materializes a structured [`Differential`] from a forward-mode pushforward program by batching
/// every input-coordinate basis tangent through one program replay and rearranging the resulting
/// output coordinates into per-(output-leaf, input-leaf) [`DifferentialBlock`]s.
fn materialize_differential_from_pushforward<'domain, D, Input, Output, V>(
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
) -> Result<
    Differential<
        Output::To<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
        Input::To<DifferentialBlock<V::Coordinate>>,
        V::Coordinate,
    >,
    TracingError,
>
where
    D: DifferentiableDomain<Type = ArrayType, Value = V>,
    V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent>,
    D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
    Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<V, To<V> = Output, ParameterStructure: PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<DifferentialBlock<V::Coordinate>>,
    Output::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
{
    let input_shapes = input_parameters
        .iter()
        .map(|parameter| static_shape(parameter.r#type().as_ref()))
        .collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let input_offsets = coordinate_offsets(&input_coordinate_counts);

    let tangent_parameters = input_parameters
        .iter()
        .map(|parameter| D::Tangent::zero(parameter.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let basis_inputs =
        standard_basis::<Input::To<D::Tangent>, D::Tangent>(&input_structure, tangent_parameters.as_slice())?;

    let output_structure = output.parameter_structure();
    let output_parameters = output.into_parameters().collect::<Vec<_>>();
    let output_shapes = output_parameters
        .iter()
        .map(|parameter| static_shape(parameter.r#type().as_ref()))
        .collect::<Vec<_>>();
    let output_coordinate_counts = coordinate_counts(output_parameters.as_slice());
    let output_offsets = coordinate_offsets(&output_coordinate_counts);

    let columns = if basis_inputs.is_empty() {
        Vec::new()
    } else {
        let lane_count = basis_inputs.len();
        let batched_tangents = reference_stack::<D::Tangent, Input::To<D::Tangent>>(basis_inputs)?;
        let batched_outputs = interpret_reference_batched_program(&pushforward, batched_tangents)?;
        flatten_batched_coordinates::<Output::To<D::Tangent>, D::Tangent>(batched_outputs, lane_count)?
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

        let partials = <Input::To<DifferentialBlock<V::Coordinate>>>::from_parameters(input_structure.clone(), blocks)?;
        rows_list.push(DifferentialRow::new(partials));
    }

    let rows =
        <Output::To<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>>::from_parameters(
            output_structure,
            rows_list,
        )?;

    Ok(Differential::new(rows))
}

/// Materializes a structured [`Differential`] using forward-mode differentiation.
///
/// [`DifferentiableDomain::jacfwd`] batches all input-coordinate basis tangents through one
/// pushforward replay and reassembles the resulting output coordinates into per-(output-leaf,
/// input-leaf) [`DifferentialBlock`]s.
#[allow(private_bounds)]
pub(crate) fn jacfwd_at<'domain, D, F, Input, Output, V>(
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
    V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
    D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
    Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<V, To<V> = Output, ParameterStructure: PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<Tracer<'domain, D>>
        + ParameterizedFamily<DifferentialBlock<V::Coordinate>>,
    Output::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<Tracer<'domain, D>>
        + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
    Output::To<Tracer<'domain, D>>: Parameterized<Tracer<'domain, D>, To<V> = Output>,
    F: FnOnce(Input::To<Tracer<'domain, D>>) -> Result<Output::To<Tracer<'domain, D>>, TracingError>,
    D::OperationCarrier: DifferentiableOperation<D>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
    let (output, pushforward) = linearize::<D, F, Input, Output, V>(domain, function, primals)?;
    materialize_differential_from_pushforward::<D, Input, Output, V>(
        input_structure,
        input_parameters,
        output,
        pushforward,
    )
}

/// Materializes a structured [`Differential`] using reverse-mode differentiation.
///
/// [`jacrev`] batches all output-coordinate basis cotangents through one pullback replay and
/// reassembles the resulting input coordinates into per-(output-leaf, input-leaf)
/// [`DifferentialBlock`]s.
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
    V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
    D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
    Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<V, To<V> = Output, ParameterStructure: Debug + PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<Tracer<'domain, D>>
        + ParameterizedFamily<DifferentialBlock<V::Coordinate>>,
    Output::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<Tracer<'domain, D>>
        + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
    Output::To<Tracer<'domain, D>>: Parameterized<Tracer<'domain, D>, To<V> = Output>,
    F: FnOnce(Input::To<Tracer<'domain, D>>) -> Result<Output::To<Tracer<'domain, D>>, TracingError>,
    D::OperationCarrier: DifferentiableOperation<D>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_shapes = input_parameters
        .iter()
        .map(|parameter| static_shape(parameter.r#type().as_ref()))
        .collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let input_offsets = coordinate_offsets(&input_coordinate_counts);

    let primals = Input::from_parameters(input_structure.clone(), input_parameters)?;
    let (output, pullback) = vjp::<D, F, Input, Output, V>(domain, function, primals)?;
    let output_structure = output.parameter_structure();
    let output_parameters = output.into_parameters().collect::<Vec<_>>();
    let output_shapes = output_parameters
        .iter()
        .map(|parameter| static_shape(parameter.r#type().as_ref()))
        .collect::<Vec<_>>();
    let output_coordinate_counts = coordinate_counts(output_parameters.as_slice());
    let output_offsets = coordinate_offsets(&output_coordinate_counts);
    let cotangent_parameters = output_parameters
        .iter()
        .map(|parameter| D::Tangent::zero(parameter.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let basis_outputs =
        standard_basis::<Output::To<D::Tangent>, D::Tangent>(&output_structure, cotangent_parameters.as_slice())?;

    let rows = if basis_outputs.is_empty() {
        Vec::new()
    } else {
        let lane_count = basis_outputs.len();
        let batched_cotangents = reference_stack::<D::Tangent, Output::To<D::Tangent>>(basis_outputs)?;
        let batched_inputs = interpret_reference_batched_program(&pullback, batched_cotangents)?;
        flatten_batched_coordinates::<Input::To<D::Tangent>, D::Tangent>(batched_inputs, lane_count)?
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

/// Materializes a structured Hessian of a scalar-output function.
///
/// [`DifferentiableDomain::hessian`] traces the reverse-mode gradient of `function`, linearizes
/// that gradient program, and then batches all input-coordinate basis tangents through the
/// resulting pushforward.
#[allow(private_bounds)]
pub(crate) fn hessian_at<'domain, D, F, Input, V>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<
    Differential<
        Input::To<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
        Input::To<DifferentialBlock<V::Coordinate>>,
        V::Coordinate,
    >,
    TracingError,
>
where
    D: DifferentiableDomain<Type = ArrayType, Value = V>
        + DifferentiableTracingDomain<Type = ArrayType, Value = V>
        + 'static,
    V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
    D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
    Input: Parameterized<V, To<V> = Input, ParameterStructure: Debug + PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<Tracer<'domain, D>>
        + ParameterizedFamily<ArrayType>
        + ParameterizedFamily<V>
        + ParameterizedFamily<DifferentialBlock<V::Coordinate>>
        + ParameterizedFamily<DifferentialRow<Input::To<DifferentialBlock<V::Coordinate>>, V::Coordinate>>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
    Input::To<Tracer<'domain, D>>:
        Parameterized<Tracer<'domain, D>, To<V> = Input, ParameterStructure: Debug + PartialEq>,
    <Input::To<Tracer<'domain, D>> as Parameterized<Tracer<'domain, D>>>::To<ArrayType>:
        Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
    F: FnOnce(Input::To<Tracer<'domain, D>>) -> Tracer<'domain, D>,
    D::OperationCarrier: Clone
        + InterpretableOperation<ArrayType, V>
        + DifferentiableOperation<TracingContext<'domain, D>>
        + DifferentiableOperation<D>
        + SupportsZeroLike<ArrayType, V>
        + SupportsAdd<ArrayType, V>
        + 'static,
    AddOperation: InterpretableOperation<ArrayType, Tracer<'domain, D>>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.iter().cloned())?;
    let (gradient, gradient_program): (Input, Program<ArrayType, V, D::OperationCarrier, Input, Input>) = domain
        .interpret_and_trace(
            |input: Input::To<Tracer<'domain, D>>| {
                let (_, gradient) = <Tracer<'domain, D> as ValueAndGradDispatch<
                    D,
                    Input::To<Tracer<'domain, D>>,
                    TracedValueAndGrad,
                >>::invoke(domain, function, input)?;
                Ok(gradient)
            },
            primals,
        )?;
    let pushforward = gradient_program.linearize(domain, input_parameters.clone())?;
    materialize_differential_from_pushforward::<D, Input, Input, V>(
        input_structure,
        input_parameters,
        gradient,
        pushforward,
    )
}
