//! Linearization, transposition, and higher-order differentiation utilities.
//!
//! This module turns forward-mode traces into staged linear programs, transposes those programs for reverse-mode
//! differentiation, and materializes dense Jacobians/Hessians for coordinate-based leaf types.

use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    marker::PhantomData,
    rc::Rc,
    sync::Arc,
};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    tracing_v2::{
        OneLike, TraceError, TraceValue, ZeroLike,
        context::{JvpContext, TransposeContext},
        forward::{JvpTracer, TangentSpace},
        graph::{AtomId, Graph, GraphBuilder},
        ops::{AddOp, LinearOpRef, NegOp, ScaleOp},
    },
};

/// Tangent representation backed by atoms in a staged linear graph.
#[derive(Clone, Debug, Parameter)]
pub struct LinearTerm<V>
where
    V: TraceValue,
{
    atom: AtomId,
    builder: Rc<RefCell<GraphBuilder<LinearOpRef<V>, V>>>,
}

impl<V> LinearTerm<V>
where
    V: TraceValue,
{
    #[inline]
    pub(crate) fn apply_linear_op(self, op: LinearOpRef<V>) -> Self {
        let atom = self
            .builder
            .borrow_mut()
            .add_equation(op, vec![self.atom])
            .expect("staging a linear op should succeed")[0];
        Self { atom, builder: self.builder }
    }

    #[inline]
    pub(crate) fn add(self, rhs: Self) -> Self {
        debug_assert!(Rc::ptr_eq(&self.builder, &rhs.builder));
        let atom = self
            .builder
            .borrow_mut()
            .add_equation(Arc::new(AddOp), vec![self.atom, rhs.atom])
            .expect("staging linear addition should succeed")[0];
        Self { atom, builder: self.builder }
    }

    #[inline]
    pub(crate) fn neg(self) -> Self {
        self.apply_linear_op(Arc::new(NegOp))
    }

    #[inline]
    pub(crate) fn scale(self, factor: V) -> Self {
        self.apply_linear_op(Arc::new(ScaleOp::new(factor)))
    }
}

impl<V> TangentSpace<V> for LinearTerm<V>
where
    V: TraceValue,
{
    #[inline]
    fn add(lhs: Self, rhs: Self) -> Self {
        lhs.add(rhs)
    }

    #[inline]
    fn neg(value: Self) -> Self {
        value.neg()
    }

    #[inline]
    fn scale(factor: V, tangent: Self) -> Self {
        tangent.scale(factor)
    }

    #[inline]
    fn zero_like(primal: &V, tangent: &Self) -> Self {
        let builder = tangent.builder.clone();
        let atom = builder.borrow_mut().add_constant(primal.zero_like());
        Self { atom, builder }
    }
}

/// Standard traced value used while building linear programs.
pub type Linearized<V> = JvpTracer<V, LinearTerm<V>>;

/// Staged linear map produced by `linearize`, `jvp_program`, or `vjp`.
pub struct LinearProgram<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    graph: Graph<LinearOpRef<V>, V, Input, Output>,
    zero: V,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<V, Input, Output> LinearProgram<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Returns the staged graph backing this linear program.
    #[inline]
    pub fn graph(&self) -> &Graph<LinearOpRef<V>, V, Input, Output> {
        &self.graph
    }

    /// Applies the linear program to a concrete input tangent or cotangent.
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.graph.call(input)
    }

    /// Transposes the linear program, turning a pushforward into a pullback.
    pub fn transpose(&self) -> Result<LinearProgram<V, Output, Input>, TraceError>
    where
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        fn accumulate<V>(
            builder: &mut GraphBuilder<LinearOpRef<V>, V>,
            adjoints: &mut [Option<AtomId>],
            atom: AtomId,
            contribution: AtomId,
        ) where
            V: TraceValue,
        {
            adjoints[atom] = Some(match adjoints[atom] {
                Some(existing) => builder
                    .add_equation(Arc::new(AddOp), vec![existing, contribution])
                    .expect("accumulating cotangents should succeed")[0],
                None => contribution,
            });
        }

        let mut builder = GraphBuilder::<LinearOpRef<V>, V>::new();
        let mut output_cotangent_inputs = Vec::with_capacity(self.graph.outputs().len());
        for output in self.graph.outputs() {
            let abstract_value =
                self.graph.atom(*output).ok_or(TraceError::UnboundAtomId { id: *output })?.abstract_value.clone();
            output_cotangent_inputs.push(builder.add_input_abstract(abstract_value));
        }

        let mut adjoints = vec![None; self.graph.atom_count()];
        for (cotangent, output) in output_cotangent_inputs.into_iter().zip(self.graph.outputs().iter().copied()) {
            accumulate(&mut builder, adjoints.as_mut_slice(), output, cotangent);
        }

        for equation in self.graph.equations().iter().rev() {
            let equation_output_cotangents =
                equation.outputs.iter().map(|output| adjoints[*output]).collect::<Option<Vec<_>>>();
            let Some(equation_output_cotangents) = equation_output_cotangents else {
                continue;
            };
            let input_cotangents = {
                let mut transpose_context = TransposeContext::new(&mut builder);
                equation.op.transpose(
                    &mut transpose_context,
                    equation.inputs.as_slice(),
                    equation.outputs.as_slice(),
                    equation_output_cotangents.as_slice(),
                )?
            };
            for (input, contribution) in equation.inputs.iter().copied().zip(input_cotangents) {
                if let Some(contribution) = contribution {
                    accumulate(&mut builder, adjoints.as_mut_slice(), input, contribution);
                }
            }
        }

        let zero_atom = builder.add_constant(self.zero.clone());
        let outputs = self
            .graph
            .input_atoms()
            .iter()
            .copied()
            .map(|input| adjoints[input].unwrap_or(zero_atom))
            .collect::<Vec<_>>();
        Ok(LinearProgram {
            graph: builder.build::<Output, Input>(
                outputs,
                self.graph.output_structure().clone(),
                self.graph.input_structure().clone(),
            ),
            zero: self.zero.clone(),
            marker: PhantomData,
        })
    }
}

impl<V, Input, Output> Display for LinearProgram<V, Input, Output>
where
    V: TraceValue,
    V::Abstract: Display,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.graph, f)
    }
}

fn try_jvp_program<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Input, Output>), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(
        &mut JvpContext<'context, Context, V>,
        Input::To<Linearized<V>>,
    ) -> Result<Output::To<Linearized<V>>, TraceError>,
{
    let input_structure = primals.parameter_structure();
    let primal_parameters = primals.into_parameters().collect::<Vec<_>>();
    let zero = primal_parameters
        .first()
        .map(|value| value.zero_like())
        .ok_or(TraceError::EmptyParameterizedValue)?;

    let mut jvp_context = JvpContext::new(context);
    let traced_input = Input::To::<Linearized<V>>::from_parameters(
        input_structure.clone(),
        primal_parameters.into_iter().map(|primal| {
            let builder = jvp_context.linear_builder();
            let atom = builder.borrow_mut().add_input(&primal);
            JvpTracer { primal, tangent: LinearTerm { atom, builder } }
        }),
    )?;

    let (output_structure, primal_outputs, tangent_outputs) = {
        let traced_output = function(&mut jvp_context, traced_input)?;
        let output_structure = traced_output.parameter_structure();
        let traced_outputs = traced_output.into_parameters().collect::<Vec<_>>();
        let primal_outputs = traced_outputs.iter().map(|output| output.primal.clone()).collect::<Vec<_>>();
        let tangent_outputs = traced_outputs.into_iter().map(|output| output.tangent.atom).collect::<Vec<_>>();
        (output_structure, primal_outputs, tangent_outputs)
    };

    let primal_output = Output::from_parameters(output_structure.clone(), primal_outputs)?;
    let (_context, builder) = jvp_context.finish();
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    Ok((
        primal_output,
        LinearProgram {
            graph: builder.build::<Input, Output>(tangent_outputs, input_structure, output_structure),
            zero,
            marker: PhantomData,
        },
    ))
}

/// Runs a forward trace and returns both the primal output and the staged pushforward.
pub fn jvp_program<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Input, Output>), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Output::To<Linearized<V>>,
{
    try_jvp_program(context, |context, input| Ok(function(context, input)), primals)
}

/// Alias for [`jvp_program`] that emphasizes the returned linear map.
pub fn linearize<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Input, Output>), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Output::To<Linearized<V>>,
{
    jvp_program(context, function, primals)
}

fn try_vjp<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Output, Input>), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(
        &mut JvpContext<'context, Context, V>,
        Input::To<Linearized<V>>,
    ) -> Result<Output::To<Linearized<V>>, TraceError>,
{
    let (output, pushforward) = try_jvp_program::<Context, F, Input, Output, V>(context, function, primals)?;
    Ok((output, pushforward.transpose()?))
}

/// Returns the primal output together with a pullback produced by transposing the staged pushforward.
pub fn vjp<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Output, Input>), TraceError>
where
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Output::To<Linearized<V>>,
{
    try_vjp(context, |context, input| Ok(function(context, input)), primals)
}

fn try_grad<'context, Context, F, Input, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<Input, TraceError>
where
    V: TraceValue + OneLike,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Result<Linearized<V>, TraceError>,
{
    let (output, pullback): (V, LinearProgram<V, V, Input>) = try_vjp(context, function, primals)?;
    pullback.call(output.one_like())
}

/// Computes the reverse-mode gradient of a scalar-output function.
pub fn grad<'context, Context, F, Input, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<Input, TraceError>
where
    V: TraceValue + OneLike,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Linearized<V>,
{
    try_grad(context, |context, input| Ok(function(context, input)), primals)
}

/// Computes both the primal scalar output and its reverse-mode gradient.
pub fn value_and_grad<'context, Context, F, Input, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<(V, Input), TraceError>
where
    V: TraceValue + OneLike,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Linearized<V>,
{
    let (output, pullback): (V, LinearProgram<V, V, Input>) = vjp(context, function, primals)?;
    let gradient = pullback.call(output.one_like())?;
    Ok((output, gradient))
}

/// Leaf type that can be materialized into a dense finite-dimensional coordinate representation.
pub trait CoordinateValue: TraceValue + OneLike {
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

impl<S, InputStructure, OutputStructure> DenseJacobian<S, InputStructure, OutputStructure>
where
    S: Clone,
{
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

fn try_jacfwd<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(
        &mut JvpContext<'context, Context, V>,
        Input::To<Linearized<V>>,
    ) -> Result<Output::To<Linearized<V>>, TraceError>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let basis_inputs = standard_basis::<Input, V>(&input_structure, input_parameters.as_slice())?;
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
    let (output, pushforward) = try_jvp_program::<Context, F, Input, Output, V>(context, function, primals)?;
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

/// Materializes a dense Jacobian using forward-mode differentiation.
pub fn jacfwd<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Output::To<Linearized<V>>,
{
    try_jacfwd::<Context, _, Input, Output, V>(context, |context, input| Ok(function(context, input)), primals)
}

fn try_jacrev<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(
        &mut JvpContext<'context, Context, V>,
        Input::To<Linearized<V>>,
    ) -> Result<Output::To<Linearized<V>>, TraceError>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
    let (output, pullback) = try_vjp::<Context, F, Input, Output, V>(context, function, primals)?;
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

/// Materializes a dense Jacobian using reverse-mode differentiation.
pub fn jacrev<'context, Context, F, Input, Output, V>(
    context: &'context mut Context,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Output::To<Linearized<V>>,
{
    try_jacrev::<Context, _, Input, Output, V>(context, |context, input| Ok(function(context, input)), primals)
}

/// Materializes a dense Hessian by applying `jacfwd` to a gradient helper.
///
/// In the current prototype, callers pass a first-derivative function (for example `first_derivative`)
/// because Rust does not yet let this API re-instantiate an arbitrary closure at a deeper trace level.
pub fn hessian<'context, Context, F, Input, V>(
    context: &'context mut Context,
    gradient_function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Input::ParameterStructure>, TraceError>
where
    V: CoordinateValue,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<Linearized<V>>,
    F: FnOnce(&mut JvpContext<'context, Context, V>, Input::To<Linearized<V>>) -> Input::To<Linearized<V>>,
{
    jacfwd::<Context, F, Input, Input, V>(context, gradient_function, primals)
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, Mul, Neg};

    use crate::tracing_v2::{FloatExt, PrototypeContext};

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    fn quadratic_plus_sin<Context, T>(_: &mut Context, x: T) -> T
    where
        T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        x.clone() * x.clone() + x.sin()
    }

    fn bilinear_sin<Context, T>(_: &mut Context, inputs: (T, T)) -> T
    where
        T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin()
    }

    #[test]
    fn linearize_returns_the_primal_output_and_pushforward() {
        let mut context = PrototypeContext::default();
        let (primal, pushforward) = linearize(&mut context, quadratic_plus_sin, 2.0f64).unwrap();

        approx_eq(primal, 2.0f64.powi(2) + 2.0f64.sin());
        approx_eq(pushforward.call(1.5f64).unwrap(), (4.0 + 2.0f64.cos()) * 1.5);
    }

    #[test]
    fn jvp_program_and_linearize_stage_the_same_pushforward() {
        let mut context = PrototypeContext::default();
        let (_, from_jvp_program) = jvp_program(&mut context, quadratic_plus_sin, 2.0f64).unwrap();
        let (_, from_linearize) = linearize(&mut context, quadratic_plus_sin, 2.0f64).unwrap();

        approx_eq(from_jvp_program.call(1.0f64).unwrap(), from_linearize.call(1.0f64).unwrap());
    }

    #[test]
    fn transposed_linear_program_matches_the_reverse_mode_pullback() {
        let mut context = PrototypeContext::default();
        let (primal, pushforward) = linearize(&mut context, bilinear_sin, (2.0f64, 3.0f64)).unwrap();
        let pullback = pushforward.transpose().unwrap();
        let cotangent = pullback.call(1.0f64).unwrap();

        approx_eq(primal, 2.0 * 3.0 + 2.0f64.sin());
        approx_eq(cotangent.0, 3.0 + 2.0f64.cos());
        approx_eq(cotangent.1, 2.0);
    }

    #[test]
    fn linear_program_display_delegates_to_the_underlying_graph() {
        let mut context = PrototypeContext::default();
        let (_, pushforward): (f64, LinearProgram<f64, f64, f64>) =
            linearize(&mut context, quadratic_plus_sin, 2.0f64).unwrap();

        let rendered = pushforward.to_string();
        assert_eq!(rendered, pushforward.graph().to_string());
        assert!(rendered.starts_with("lambda %0:f64[] .\n"));
        assert!(rendered.contains("scale %0"));
        assert!(rendered.ends_with(')'));
    }
}
