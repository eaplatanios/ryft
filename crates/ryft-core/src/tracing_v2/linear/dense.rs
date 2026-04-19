use super::*;

/// Leaf type that can be materialized into a dense finite-dimensional coordinate representation.
pub trait CoordinateValue: Traceable<ArrayType> + ZeroLike + OneLike {
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

#[derive(Clone, Debug)]
pub struct DenseJacobian<S, InputStructure, OutputStructure> {
    values: Vec<S>,
    rows: usize,
    cols: usize,
    input_structure: InputStructure,
    output_structure: OutputStructure,
    input_coordinate_counts: Vec<usize>,
    output_coordinate_counts: Vec<usize>,
}

impl<S: Clone, InputStructure, OutputStructure> DenseJacobian<S, InputStructure, OutputStructure> {
    fn from_rows(
        rows_data: Vec<Vec<S>>,
        input_structure: InputStructure,
        output_structure: OutputStructure,
        input_coordinate_counts: Vec<usize>,
        output_coordinate_counts: Vec<usize>,
    ) -> Result<Self, TraceError> {
        let rows = output_coordinate_counts.iter().sum::<usize>();
        let cols = input_coordinate_counts.iter().sum::<usize>();
        if rows_data.len() != rows {
            return Err(TraceError::InternalInvariantViolation(
                "row-major Jacobian materialization produced an unexpected number of rows",
            ));
        }
        let mut values = Vec::with_capacity(rows.saturating_mul(cols));
        for row in rows_data {
            if row.len() != cols {
                return Err(TraceError::InternalInvariantViolation(
                    "row-major Jacobian materialization produced an unexpected row width",
                ));
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
    ) -> Result<Self, TraceError> {
        let rows = output_coordinate_counts.iter().sum::<usize>();
        let cols = input_coordinate_counts.iter().sum::<usize>();
        if columns.len() != cols {
            return Err(TraceError::InternalInvariantViolation(
                "column-major Jacobian materialization produced an unexpected number of columns",
            ));
        }
        let mut values = Vec::with_capacity(rows.saturating_mul(cols));
        for row in 0..rows {
            for column in columns.iter() {
                if column.len() != rows {
                    return Err(TraceError::InternalInvariantViolation(
                        "column-major Jacobian materialization produced an unexpected column height",
                    ));
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

    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn input_dimension(&self) -> usize {
        self.cols
    }

    #[inline]
    pub fn output_dimension(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn values(&self) -> &[S] {
        self.values.as_slice()
    }

    #[inline]
    pub fn input_structure(&self) -> &InputStructure {
        &self.input_structure
    }

    #[inline]
    pub fn output_structure(&self) -> &OutputStructure {
        &self.output_structure
    }

    #[inline]
    pub fn input_coordinate_counts(&self) -> &[usize] {
        self.input_coordinate_counts.as_slice()
    }

    #[inline]
    pub fn output_coordinate_counts(&self) -> &[usize] {
        self.output_coordinate_counts.as_slice()
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<&S> {
        (row < self.rows && col < self.cols).then(|| &self.values[row * self.cols + col])
    }

    #[cfg(any(feature = "ndarray", test))]
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

fn flatten_coordinates<Value, V>(value: Value) -> Vec<V::Coordinate>
where
    Value: Parameterized<V>,
    V: CoordinateValue,
{
    value.into_parameters().flat_map(|parameter| parameter.coordinates()).collect::<Vec<_>>()
}

fn standard_basis<Value, V>(structure: &Value::ParameterStructure, parameters: &[V]) -> Result<Vec<Value>, TraceError>
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
#[allow(private_bounds)]
pub fn jacfwd<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Tracer<E>>,
    Output::Family: ParameterizedFamily<Tracer<E>>,
    F: FnOnce(Input::To<Tracer<E>>) -> Result<Output::To<Tracer<E>>, TraceError>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: CoreLinearReplayOp<V>,
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

    let mut columns = Vec::with_capacity(basis_inputs.len());
    for tangent in basis_inputs {
        columns.push(flatten_coordinates::<Output, V>(pushforward.call(tangent)?));
    }

    DenseJacobian::from_columns(
        columns,
        input_structure,
        output_structure,
        input_coordinate_counts,
        output_coordinate_counts,
    )
}

/// Materializes a dense Jacobian using reverse-mode differentiation.
#[allow(private_bounds)]
pub fn jacrev<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Tracer<E>>,
    Output::Family: ParameterizedFamily<Tracer<E>>,
    F: FnOnce(Input::To<Tracer<E>>) -> Result<Output::To<Tracer<E>>, TraceError>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V>,
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

    let mut rows = Vec::with_capacity(basis_outputs.len());
    for cotangent in basis_outputs {
        rows.push(flatten_coordinates::<Input, V>(pullback.call(cotangent)?));
    }

    DenseJacobian::from_rows(rows, input_structure, output_structure, input_coordinate_counts, output_coordinate_counts)
}

/// Materializes a dense Hessian by applying `jacfwd` to a gradient helper.
///
/// In the current prototype, callers pass a first-derivative function (for example `first_derivative`)
/// because Rust does not yet let this API re-instantiate an arbitrary closure at a deeper trace level.
#[allow(private_bounds)]
pub fn hessian<E, F, Input, V>(
    engine: &E,
    gradient_function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Input::ParameterStructure>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V> + 'static,
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Tracer<E>>,
    F: FnOnce(Input::To<Tracer<E>>) -> Result<Input::To<Tracer<E>>, TraceError>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: CoreLinearReplayOp<V>,
{
    jacfwd::<E, F, Input, Input, V>(engine, gradient_function, primals)
}
