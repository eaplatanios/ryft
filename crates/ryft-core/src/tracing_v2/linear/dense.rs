use super::*;

use crate::tracing_v2::{
    BatchingError, DifferentiationError,
    batching::{ReferenceBatch, interpret_reference_batched_program, reference_stack},
};

/// Leaf type that can be materialized into a dense finite-dimensional coordinate representation.
///
/// Dense Jacobian and Hessian materialization only makes sense for leaf types with a finite,
/// explicit basis. [`CoordinateValue`] is the bridge from the generic tracing world into that
/// coordinate-based view: it teaches the dense helpers how many coordinates a leaf contributes,
/// what basis vectors to probe with, and how to flatten outputs back into numeric entries.
pub trait CoordinateValue:
    Traceable<ArrayType> + ZeroLike + OneLike + Zero<ArrayType> + One<ArrayType> + Differentiable<ArrayType, Tangent = Self>
{
    /// Scalar-like coordinate type used by dense Jacobians and Hessians.
    type Coordinate: Clone + Debug + PartialEq + 'static;

    /// Returns the number of coordinates contributed by this leaf.
    fn coordinate_count(&self) -> usize;

    /// Returns a standard basis for the coordinate space of this leaf.
    fn coordinate_basis(&self) -> Vec<Self>;

    /// Flattens the leaf into its coordinate values in a deterministic order.
    fn coordinates(&self) -> Vec<Self::Coordinate>;
}
impl CoordinateValue for f32 {
    type Coordinate = f32;

    #[inline]
    fn coordinate_count(&self) -> usize {
        1
    }

    #[inline]
    fn coordinate_basis(&self) -> Vec<Self> {
        vec![1.0]
    }

    #[inline]
    fn coordinates(&self) -> Vec<Self::Coordinate> {
        vec![*self]
    }
}

impl CoordinateValue for f64 {
    type Coordinate = f64;

    #[inline]
    fn coordinate_count(&self) -> usize {
        1
    }

    #[inline]
    fn coordinate_basis(&self) -> Vec<Self> {
        vec![1.0]
    }

    #[inline]
    fn coordinates(&self) -> Vec<Self::Coordinate> {
        vec![*self]
    }
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

    #[cfg(any(feature = "ndarray", test))]
    /// Converts the dense Jacobian into an `ndarray::Array2`.
    pub fn to_array2(&self) -> ndarray::Array2<S> {
        ndarray::Array2::from_shape_vec((self.rows, self.cols), self.values.clone())
            .expect("dense Jacobian dimensions should match the stored values")
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
    Value: Parameterized<V, ParameterStructure: Clone>,
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

/// Materializes a dense Jacobian using forward-mode differentiation.
///
/// [`jacfwd`] batches all input-coordinate basis tangents through one pushforward replay and collects
/// the resulting output coordinates as matrix columns.
#[allow(private_bounds)]
pub fn jacfwd<'engine, E, F, Input, Output, V>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<ReferenceBatch<V>> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    Output::Family: ParameterizedFamily<ReferenceBatch<V>> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    F: FnOnce(
        Input::To<DifferentiableTracer<'engine, E>>,
    ) -> Result<Output::To<DifferentiableTracer<'engine, E>>, TracingError>,
    E::Operation: DifferentiableOperation<E> + InterpretableOperation<ArrayType, V>,
    E::LinearOperation: InterpretableOperation<ArrayType, V>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let basis_inputs = standard_basis::<Input, V>(&input_structure, input_parameters.as_slice())?;
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
    let (output, pushforward) = jvp_program::<E, F, Input, Output, V>(engine, function, primals)?;
    let output_structure = output.parameter_structure();
    let output_parameters = output.into_parameters().collect::<Vec<_>>();
    let output_coordinate_counts = coordinate_counts(output_parameters.as_slice());

    let columns = if basis_inputs.is_empty() {
        Vec::new()
    } else {
        let lane_count = basis_inputs.len();
        let batched_tangents = reference_stack::<V, Input>(basis_inputs)?;
        let batched_outputs = interpret_reference_batched_program(&pushforward, batched_tangents)?;
        flatten_batched_coordinates::<Output, V>(batched_outputs, lane_count)?
    };

    DenseJacobian::from_columns(
        columns,
        input_structure,
        output_structure,
        input_coordinate_counts,
        output_coordinate_counts,
    )
}

/// Materializes a dense Jacobian using reverse-mode differentiation.
///
/// [`jacrev`] batches all output-coordinate basis cotangents through one pullback replay and collects
/// the resulting input coordinates as matrix rows.
#[allow(private_bounds)]
pub fn jacrev<'engine, E, F, Input, Output, V>(
    engine: &'engine E,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Input::Family: ParameterizedFamily<ReferenceBatch<V>> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    Output::Family: ParameterizedFamily<ReferenceBatch<V>> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    F: FnOnce(
        Input::To<DifferentiableTracer<'engine, E>>,
    ) -> Result<Output::To<DifferentiableTracer<'engine, E>>, TracingError>,
    E::Operation: DifferentiableOperation<E> + InterpretableOperation<ArrayType, V>,
    E::LinearOperation: Clone
        + InterpretableOperation<ArrayType, V>
        + LinearOperation<ArrayType, V, E::LinearOperation>
        + crate::tracing_v2::operations::SupportsZero<ArrayType, V>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
    let (output, pullback) = vjp::<E, F, Input, Output, V>(engine, function, primals)?;
    let output_structure = output.parameter_structure();
    let output_parameters = output.into_parameters().collect::<Vec<_>>();
    let output_coordinate_counts = coordinate_counts(output_parameters.as_slice());
    let basis_outputs = standard_basis::<Output, V>(&output_structure, output_parameters.as_slice())?;

    let rows = if basis_outputs.is_empty() {
        Vec::new()
    } else {
        let lane_count = basis_outputs.len();
        let batched_cotangents = reference_stack::<V, Output>(basis_outputs)?;
        let batched_inputs = interpret_reference_batched_program(&pullback, batched_cotangents)?;
        flatten_batched_coordinates::<Input, V>(batched_inputs, lane_count)?
    };

    DenseJacobian::from_rows(rows, input_structure, output_structure, input_coordinate_counts, output_coordinate_counts)
}

/// Materializes a dense Hessian by applying `jacfwd` to a gradient helper.
///
/// In the current prototype, callers pass a first-derivative function (for example `first_derivative`)
/// because Rust does not yet let this API re-instantiate an arbitrary closure at a deeper trace level.
#[allow(private_bounds)]
pub fn hessian<'engine, E, F, Input, V>(
    engine: &'engine E,
    gradient_function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Input::ParameterStructure>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = V> + 'static,
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + std::fmt::Debug + PartialEq>,
    Input::Family: ParameterizedFamily<ReferenceBatch<V>> + ParameterizedFamily<DifferentiableTracer<'engine, E>>,
    F: FnOnce(
        Input::To<DifferentiableTracer<'engine, E>>,
    ) -> Result<Input::To<DifferentiableTracer<'engine, E>>, TracingError>,
    E::Operation: DifferentiableOperation<E> + InterpretableOperation<ArrayType, V>,
    E::LinearOperation: InterpretableOperation<ArrayType, V>,
{
    jacfwd::<E, F, Input, Input, V>(engine, gradient_function, primals)
}

#[cfg(test)]
mod tests {
    use crate::{
        parameters::Placeholder,
        tracing::TracingError,
        tracing_v2::{DifferentiationError, Sin, engines::ArrayScalarEngine},
    };

    use super::{DenseJacobian, jacfwd, jacrev};

    fn assert_close(actual: f64, expected: f64) {
        let delta = (actual - expected).abs();
        assert!(delta <= 1e-12, "expected {actual} ~= {expected}; absolute error {delta} exceeded tolerance");
    }

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

    #[test]
    fn test_jacfwd_batches_basis_tangents() {
        let engine = ArrayScalarEngine::<f64>::new();
        let jacobian = jacfwd::<ArrayScalarEngine<f64>, _, (f64, f64), (f64, f64), f64>(
            &engine,
            |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), x + y)),
            (2.0f64, 3.0f64),
        )
        .unwrap();

        assert_eq!(jacobian.rows(), 2);
        assert_eq!(jacobian.cols(), 2);
        assert_close(jacobian.values()[0], 3.0 + 2.0f64.cos());
        assert_close(jacobian.values()[1], 2.0);
        assert_close(jacobian.values()[2], 1.0);
        assert_close(jacobian.values()[3], 1.0);
    }

    #[test]
    fn test_jacrev_batches_basis_cotangents() {
        let engine = ArrayScalarEngine::<f64>::new();
        let jacobian = jacrev::<ArrayScalarEngine<f64>, _, (f64, f64), (f64, f64), f64>(
            &engine,
            |(x, y)| Ok((x.clone() * y.clone() + x.clone().sin(), x + y)),
            (2.0f64, 3.0f64),
        )
        .unwrap();

        assert_eq!(jacobian.rows(), 2);
        assert_eq!(jacobian.cols(), 2);
        assert_close(jacobian.values()[0], 3.0 + 2.0f64.cos());
        assert_close(jacobian.values()[1], 2.0);
        assert_close(jacobian.values()[2], 1.0);
        assert_close(jacobian.values()[3], 1.0);
    }
}
