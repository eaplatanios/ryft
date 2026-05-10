use super::*;

use crate::tracing_v2::batching::{ReferenceBatch, interpret_reference_batched_program, reference_stack};
use crate::tracing_v2::{BatchingError, DifferentiationError};

/// Forward-mode dense Jacobian transform.
pub(crate) struct JacFwd<F> {
    /// Function whose Jacobian is materialized.
    function: F,
}

impl<F> JacFwd<F> {
    /// Creates a forward-mode dense Jacobian transform for `function`.
    #[inline]
    pub(crate) const fn new(function: F) -> Self {
        Self { function }
    }

    /// Evaluates this Jacobian transform at `primals`.
    #[allow(private_bounds)]
    pub(crate) fn evaluate<'domain, D, Input, Output, V>(
        self,
        domain: &'domain D,
        primals: Input,
    ) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TracingError>
    where
        D: DifferentiableDomain<Type = ArrayType, Value = V> + 'static,
        V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
        D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        Output: Parameterized<V, To<V> = Output, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<D::Tangent>
            + ParameterizedFamily<ReferenceBatch<D::Tangent>>
            + ParameterizedFamily<DifferentiableTracer<'domain, D>>,
        Output::Family: ParameterizedFamily<D::Tangent>
            + ParameterizedFamily<ReferenceBatch<D::Tangent>>
            + ParameterizedFamily<DifferentiableTracer<'domain, D>>,
        Output::To<DifferentiableTracer<'domain, D>>: Parameterized<DifferentiableTracer<'domain, D>, To<V> = Output>,
        F: FnOnce(
            Input::To<DifferentiableTracer<'domain, D>>,
        ) -> Result<Output::To<DifferentiableTracer<'domain, D>>, TracingError>,
    {
        jacfwd_at::<D, F, Input, Output, V>(domain, self.function, primals)
    }
}

impl<F> JacFwd<Grad<F>> {
    /// Evaluates a forward-mode Jacobian of a reverse-mode gradient at `primals`.
    #[allow(private_bounds)]
    pub(crate) fn evaluate_gradient<'domain, D, Input, V>(
        self,
        domain: &'domain D,
        primals: Input,
    ) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Input::ParameterStructure>, TracingError>
    where
        D: DifferentiableDomain<Type = ArrayType, Value = V>
            + DifferentiableTracingDomain<Type = ArrayType, Value = V>
            + 'static,
        V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
        D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        Input::Family: ParameterizedFamily<D::Tangent>
            + ParameterizedFamily<ReferenceBatch<D::Tangent>>
            + ParameterizedFamily<Tracer<'domain, D>>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<V>
            + ParameterizedFamily<DifferentiableTracer<'domain, D>>,
        Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
        Input::To<Tracer<'domain, D>>:
            Parameterized<Tracer<'domain, D>, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        <Input::To<Tracer<'domain, D>> as Parameterized<Tracer<'domain, D>>>::To<ArrayType>:
            Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
        Input::To<DifferentiableTracer<'domain, D>>: Parameterized<DifferentiableTracer<'domain, D>, To<V> = Input>,
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

/// Scalar-output dense Hessian transform.
pub(crate) struct Hessian<F> {
    /// Function whose Hessian is materialized.
    function: F,
}

impl<F> Hessian<F> {
    /// Creates a dense Hessian transform for `function`.
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
    ) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Input::ParameterStructure>, TracingError>
    where
        D: DifferentiableDomain<Type = ArrayType, Value = V>
            + DifferentiableTracingDomain<Type = ArrayType, Value = V>
            + 'static,
        V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
        D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        Input::Family: ParameterizedFamily<D::Tangent>
            + ParameterizedFamily<ReferenceBatch<D::Tangent>>
            + ParameterizedFamily<Tracer<'domain, D>>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<V>
            + ParameterizedFamily<DifferentiableTracer<'domain, D>>,
        Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
        Input::To<Tracer<'domain, D>>:
            Parameterized<Tracer<'domain, D>, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        <Input::To<Tracer<'domain, D>> as Parameterized<Tracer<'domain, D>>>::To<ArrayType>:
            Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
        Input::To<DifferentiableTracer<'domain, D>>: Parameterized<DifferentiableTracer<'domain, D>, To<V> = Input>,
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
/// Dense Jacobian and Hessian materialization only makes sense for leaf types with a finite,
/// explicit basis. [`CoordinateValue`] is the bridge from the generic tracing world into that
/// coordinate-based view: it teaches the dense helpers how many coordinates a leaf contributes,
/// what basis vectors to probe with, and how to flatten outputs back into numeric entries.
pub trait CoordinateValue: Traceable<ArrayType> + ZeroLike + OneLike + Zero<ArrayType> + One<ArrayType> {
    /// Scalar-like coordinate type used by dense Jacobians and Hessians.
    type Coordinate: Clone + Debug + PartialEq + 'static;

    /// Returns the number of coordinates contributed by this leaf.
    fn coordinate_count(&self) -> usize;

    /// Returns a standard basis for the coordinate space of this leaf.
    fn coordinate_basis(&self) -> Vec<Self>;

    /// Flattens the leaf into its coordinate values in a deterministic order.
    fn coordinates(&self) -> Vec<Self::Coordinate>;
}
/// Dense matrix representation of a Jacobian- or Hessian-like linear map.
///
/// The stored matrix is accompanied by the input and output parameter structures plus per-leaf
/// coordinate counts so callers can relate rows and columns back to the original structured
/// function signature.
#[derive(Clone, Debug)]
pub struct DenseJacobian<S, InputStructure, OutputStructure> {
    /// Row-major matrix entries of the dense Jacobian.
    values: Vec<S>,

    /// Number of rows in the dense matrix.
    rows: usize,

    /// Number of columns in the dense matrix.
    cols: usize,

    /// Structured input parameter shape used by the differentiated function.
    input_structure: InputStructure,

    /// Structured output parameter shape used by the differentiated function.
    output_structure: OutputStructure,

    /// Coordinate counts contributed by each flattened input leaf.
    input_coordinate_counts: Vec<usize>,

    /// Coordinate counts contributed by each flattened output leaf.
    output_coordinate_counts: Vec<usize>,
}

impl<S: Clone, InputStructure, OutputStructure> DenseJacobian<S, InputStructure, OutputStructure> {
    fn from_rows(
        rows_data: Vec<Vec<S>>,
        input_structure: InputStructure,
        output_structure: OutputStructure,
        input_coordinate_counts: Vec<usize>,
        output_coordinate_counts: Vec<usize>,
    ) -> Result<Self, TracingError> {
        let rows = output_coordinate_counts.iter().sum::<usize>();
        let cols = input_coordinate_counts.iter().sum::<usize>();
        if rows_data.len() != rows {
            return Err(DifferentiationError::InvalidJacobianRowCount { expected: rows, got: rows_data.len() }.into());
        }
        let mut values = Vec::with_capacity(rows.saturating_mul(cols));
        for row in rows_data {
            if row.len() != cols {
                return Err(DifferentiationError::InvalidJacobianRowWidth { expected: cols, got: row.len() }.into());
            }
            values.extend(row);
        }
        Ok(Self {
            values,
            rows,
            cols,
            input_structure,
            output_structure,
            input_coordinate_counts,
            output_coordinate_counts,
        })
    }

    fn from_columns(
        columns: Vec<Vec<S>>,
        input_structure: InputStructure,
        output_structure: OutputStructure,
        input_coordinate_counts: Vec<usize>,
        output_coordinate_counts: Vec<usize>,
    ) -> Result<Self, TracingError> {
        let rows = output_coordinate_counts.iter().sum::<usize>();
        let cols = input_coordinate_counts.iter().sum::<usize>();
        if columns.len() != cols {
            return Err(DifferentiationError::InvalidJacobianColumnCount { expected: cols, got: columns.len() }.into());
        }
        let mut values = Vec::with_capacity(rows.saturating_mul(cols));
        for row in 0..rows {
            for column in columns.iter() {
                if column.len() != rows {
                    return Err(DifferentiationError::InvalidJacobianColumnHeight {
                        expected: rows,
                        got: column.len(),
                    }
                    .into());
                }
                values.push(column[row].clone());
            }
        }
        Ok(Self {
            values,
            rows,
            cols,
            input_structure,
            output_structure,
            input_coordinate_counts,
            output_coordinate_counts,
        })
    }

    /// Returns the total number of matrix rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the total number of matrix columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the dimension of the flattened input coordinate space.
    #[inline]
    pub fn input_dimension(&self) -> usize {
        self.cols
    }

    /// Returns the dimension of the flattened output coordinate space.
    #[inline]
    pub fn output_dimension(&self) -> usize {
        self.rows
    }

    /// Returns the matrix entries in row-major order.
    #[inline]
    pub fn values(&self) -> &[S] {
        self.values.as_slice()
    }

    /// Returns the structured input metadata the matrix columns correspond to.
    #[inline]
    pub fn input_structure(&self) -> &InputStructure {
        &self.input_structure
    }

    /// Returns the structured output metadata the matrix rows correspond to.
    #[inline]
    pub fn output_structure(&self) -> &OutputStructure {
        &self.output_structure
    }

    /// Returns how many flattened coordinates each input leaf contributes.
    #[inline]
    pub fn input_coordinate_counts(&self) -> &[usize] {
        self.input_coordinate_counts.as_slice()
    }

    /// Returns how many flattened coordinates each output leaf contributes.
    #[inline]
    pub fn output_coordinate_counts(&self) -> &[usize] {
        self.output_coordinate_counts.as_slice()
    }

    /// Returns one matrix element if the requested row and column are in bounds.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&S> {
        (row < self.rows && col < self.cols).then(|| &self.values[row * self.cols + col])
    }
}

fn coordinate_counts<V>(parameters: &[V]) -> Vec<usize>
where
    V: CoordinateValue,
{
    parameters.iter().map(CoordinateValue::coordinate_count).collect::<Vec<_>>()
}

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

fn materialize_dense_jacobian_from_pushforward<D, Input, Output, V>(
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
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TracingError>
where
    D: DifferentiableDomain<Type = ArrayType, Value = V>,
    V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent>,
    D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
    Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
    Output: Parameterized<V, To<V> = Output, ParameterStructure: PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<ReferenceBatch<D::Tangent>>,
    Output::Family: ParameterizedFamily<D::Tangent> + ParameterizedFamily<ReferenceBatch<D::Tangent>>,
{
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let tangent_parameters = input_parameters
        .iter()
        .map(|parameter| D::Tangent::zero(parameter.r#type().as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let basis_inputs =
        standard_basis::<Input::To<D::Tangent>, D::Tangent>(&input_structure, tangent_parameters.as_slice())?;
    let output_structure = output.parameter_structure();
    let output_parameters = output.into_parameters().collect::<Vec<_>>();
    let output_coordinate_counts = coordinate_counts(output_parameters.as_slice());

    let columns = if basis_inputs.is_empty() {
        Vec::new()
    } else {
        let lane_count = basis_inputs.len();
        let batched_tangents = reference_stack::<D::Tangent, Input::To<D::Tangent>>(basis_inputs)?;
        let batched_outputs = interpret_reference_batched_program(&pushforward, batched_tangents)?;
        flatten_batched_coordinates::<Output::To<D::Tangent>, D::Tangent>(batched_outputs, lane_count)?
    };

    DenseJacobian::from_columns(
        columns,
        input_structure,
        output_structure,
        input_coordinate_counts,
        output_coordinate_counts,
    )
}

/// Materializes a dense Jacobian using forward-mode differentiation.
///
/// [`DifferentiableDomain::jacfwd`] batches all input-coordinate basis tangents through one pushforward replay and
/// collects the resulting output coordinates as matrix columns.
#[allow(private_bounds)]
pub(crate) fn jacfwd_at<'domain, D, F, Input, Output, V>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TracingError>
where
    D: DifferentiableDomain<Type = ArrayType, Value = V> + 'static,
    V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
    D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
    Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
    Output: Parameterized<V, To<V> = Output, ParameterStructure: PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<DifferentiableTracer<'domain, D>>,
    Output::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<DifferentiableTracer<'domain, D>>,
    Output::To<DifferentiableTracer<'domain, D>>: Parameterized<DifferentiableTracer<'domain, D>, To<V> = Output>,
    F: FnOnce(
        Input::To<DifferentiableTracer<'domain, D>>,
    ) -> Result<Output::To<DifferentiableTracer<'domain, D>>, TracingError>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
    let (output, pushforward) = linearize::<D, F, Input, Output, V>(domain, function, primals)?;
    materialize_dense_jacobian_from_pushforward::<D, Input, Output, V>(
        input_structure,
        input_parameters,
        output,
        pushforward,
    )
}

/// Materializes a dense Jacobian using reverse-mode differentiation.
///
/// [`jacrev`] batches all output-coordinate basis cotangents through one pullback replay and collects
/// the resulting input coordinates as matrix rows.
#[allow(private_bounds)]
pub fn jacrev<'domain, D, F, Input, Output, V>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TracingError>
where
    D: DifferentiableDomain<Type = ArrayType, Value = V> + 'static,
    V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
    D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
    Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
    Output: Parameterized<V, To<V> = Output, ParameterStructure: std::fmt::Debug + PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<DifferentiableTracer<'domain, D>>,
    Output::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<DifferentiableTracer<'domain, D>>,
    Output::To<DifferentiableTracer<'domain, D>>: Parameterized<DifferentiableTracer<'domain, D>, To<V> = Output>,
    F: FnOnce(
        Input::To<DifferentiableTracer<'domain, D>>,
    ) -> Result<Output::To<DifferentiableTracer<'domain, D>>, TracingError>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
    let (output, pullback) = vjp::<D, F, Input, Output, V>(domain, function, primals)?;
    let output_structure = output.parameter_structure();
    let output_parameters = output.into_parameters().collect::<Vec<_>>();
    let output_coordinate_counts = coordinate_counts(output_parameters.as_slice());
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

    DenseJacobian::from_rows(rows, input_structure, output_structure, input_coordinate_counts, output_coordinate_counts)
}

/// Materializes a dense Hessian of a scalar-output function.
///
/// [`DifferentiableDomain::hessian`] traces the reverse-mode gradient of `function`, linearizes that gradient program,
/// and then batches all input-coordinate basis tangents through the resulting pushforward. This is the direct
/// scalar-output Hessian API: callers pass the original function, not a manually written gradient helper.
#[allow(private_bounds)]
pub(crate) fn hessian_at<'domain, D, F, Input, V>(
    domain: &'domain D,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Input::ParameterStructure>, TracingError>
where
    D: DifferentiableDomain<Type = ArrayType, Value = V>
        + DifferentiableTracingDomain<Type = ArrayType, Value = V>
        + 'static,
    V: CoordinateValue + Differentiable<ArrayType, Tangent = D::Tangent> + 'domain,
    D::Tangent: CoordinateValue<Coordinate = V::Coordinate>,
    Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
    Input::Family: ParameterizedFamily<D::Tangent>
        + ParameterizedFamily<ReferenceBatch<D::Tangent>>
        + ParameterizedFamily<Tracer<'domain, D>>
        + ParameterizedFamily<ArrayType>
        + ParameterizedFamily<V>
        + ParameterizedFamily<DifferentiableTracer<'domain, D>>,
    Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
    Input::To<Tracer<'domain, D>>:
        Parameterized<Tracer<'domain, D>, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
    <Input::To<Tracer<'domain, D>> as Parameterized<Tracer<'domain, D>>>::To<ArrayType>:
        Parameterized<ArrayType, To<Tracer<'domain, D>> = Input::To<Tracer<'domain, D>>>,
    Input::To<DifferentiableTracer<'domain, D>>: Parameterized<DifferentiableTracer<'domain, D>, To<V> = Input>,
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
    materialize_dense_jacobian_from_pushforward::<D, Input, Input, V>(
        input_structure,
        input_parameters,
        gradient,
        pushforward,
    )
}

#[cfg(test)]
mod tests {
    use crate::parameters::Placeholder;
    use crate::tracing::TracingError;
    use crate::tracing_v2::DifferentiationError;

    use super::DenseJacobian;

    #[test]
    fn test_dense_jacobian_from_rows_rejects_invalid_row_count() {
        let result = DenseJacobian::from_rows(vec![vec![1.0]], Placeholder, Placeholder, vec![1], vec![1, 1]);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::InvalidJacobianRowCount { expected: 2, got: 1 }))
        ));
    }

    #[test]
    fn test_dense_jacobian_from_rows_rejects_invalid_row_width() {
        let result = DenseJacobian::from_rows(vec![vec![1.0]], Placeholder, Placeholder, vec![2], vec![1]);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::InvalidJacobianRowWidth { expected: 2, got: 1 }))
        ));
    }

    #[test]
    fn test_dense_jacobian_from_columns_rejects_invalid_column_count() {
        let result = DenseJacobian::from_columns(vec![vec![1.0]], Placeholder, Placeholder, vec![1, 1], vec![1]);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::InvalidJacobianColumnCount {
                expected: 2,
                got: 1,
            }))
        ));
    }

    #[test]
    fn test_dense_jacobian_from_columns_rejects_invalid_column_height() {
        let result = DenseJacobian::from_columns(vec![vec![1.0]], Placeholder, Placeholder, vec![1], vec![1, 1]);

        assert!(matches!(
            result,
            Err(TracingError::Differentiation(DifferentiationError::InvalidJacobianColumnHeight {
                expected: 2,
                got: 1,
            }))
        ));
    }
}
