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
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        FloatExt, MatrixOps, OneLike, TraceError, TraceValue, TransformLeaf, ZeroLike,
        forward::{JvpTracer, TangentSpace},
        graph::{AtomId, Graph},
        jit::{JitTracer, try_trace_program},
        operations::{AddOp, NegOp, ScaleOp},
        ops::Op,
        program::{Program, ProgramBuilder, ProgramOpRef},
    },
};

/// Tangent representation backed by atoms in a staged linear graph.
#[derive(Clone, Debug, Parameter)]
pub(crate) struct LinearTerm<V>
where
    V: TraceValue,
{
    atom: AtomId,
    builder: Rc<RefCell<ProgramBuilder<V>>>,
}

impl<V> LinearTerm<V>
where
    V: TraceValue + FloatExt,
{
    #[inline]
    pub(crate) fn atom(&self) -> AtomId {
        self.atom
    }

    #[inline]
    pub(crate) fn builder_handle(&self) -> Rc<RefCell<ProgramBuilder<V>>> {
        self.builder.clone()
    }

    #[inline]
    pub(crate) fn from_staged_parts(atom: AtomId, builder: Rc<RefCell<ProgramBuilder<V>>>) -> Self {
        Self { atom, builder }
    }

    pub(crate) fn apply_staged_op(
        inputs: &[Self],
        op: ProgramOpRef<V>,
        output_count: usize,
    ) -> Result<Vec<Self>, TraceError> {
        if inputs.is_empty() {
            return Err(TraceError::EmptyParameterizedValue);
        }

        let builder = inputs[0].builder.clone();
        if inputs.iter().skip(1).any(|input| !Rc::ptr_eq(&builder, &input.builder)) {
            return Err(TraceError::InternalInvariantViolation(
                "linear tracer inputs for one staged op must share the same builder",
            ));
        }

        let input_atoms = inputs.iter().map(|input| input.atom).collect::<Vec<_>>();
        let output_atoms = builder.borrow_mut().add_equation(op, input_atoms)?;
        if output_atoms.len() != output_count {
            return Err(TraceError::InvalidOutputCount { expected: output_count, got: output_atoms.len() });
        }
        Ok(output_atoms.into_iter().map(|atom| Self { atom, builder: builder.clone() }).collect())
    }

    #[inline]
    pub(crate) fn apply_linear_op(self, op: ProgramOpRef<V>) -> Self {
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
    V: TraceValue + FloatExt + ZeroLike + MatrixOps,
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
pub(crate) type Linearized<V> = JvpTracer<V, LinearTerm<V>>;

/// Staged linear map produced by `linearize`, `jvp_program`, or `vjp`.
pub struct LinearProgram<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    program: Program<V, Input, Output>,
    zero: V,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<V, Input, Output> Clone for LinearProgram<V, Input, Output>
where
    V: TraceValue + Clone,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), zero: self.zero.clone(), marker: PhantomData }
    }
}

impl<V, Input, Output> LinearProgram<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    #[inline]
    pub(crate) fn from_program(program: Program<V, Input, Output>, zero: V) -> Self {
        Self { program, zero, marker: PhantomData }
    }

    /// Returns the staged graph backing this linear program.
    #[inline]
    pub(crate) fn program(&self) -> &Program<V, Input, Output> {
        &self.program
    }

    /// Applies the linear program to a concrete input tangent or cotangent.
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.program.call(input)
    }

    /// Transposes the linear program, turning a pushforward into a pullback.
    pub fn transpose(&self) -> Result<LinearProgram<V, Output, Input>, TraceError>
    where
        V: FloatExt + ZeroLike + MatrixOps,
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        transpose_linear_program(self)
    }
}

impl<V, Input, Output> Display for LinearProgram<V, Input, Output>
where
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.program, formatter)
    }
}

fn apply_program_jvp_rule<V>(
    op: &dyn Op<V>,
    inputs: &[JvpTracer<V, LinearTerm<V>>],
) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + MatrixOps,
{
    op.apply_program_jvp_rule(inputs)
}

fn transpose_program_op<V>(
    op: &dyn Op<V>,
    builder: &mut ProgramBuilder<V>,
    inputs: &[AtomId],
    outputs: &[AtomId],
    output_cotangents: &[AtomId],
) -> Result<Vec<Option<AtomId>>, TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + MatrixOps,
{
    op.transpose_program_op(builder, inputs, outputs, output_cotangents)
}

pub(crate) fn linearize_program<V, Input, Output>(
    program: &Program<V, Input, Output>,
) -> Result<LinearProgram<V, Input, Output>, TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + MatrixOps,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
{
    fn tangent_for_atom<V, Input, Output>(
        graph: &Graph<ProgramOpRef<V>, V, Input, Output>,
        builder: &Rc<RefCell<ProgramBuilder<V>>>,
        tangents: &mut [Option<LinearTerm<V>>],
        atom_id: AtomId,
    ) -> Result<LinearTerm<V>, TraceError>
    where
        V: TraceValue + FloatExt + ZeroLike,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
    {
        if let Some(term) = tangents[atom_id].clone() {
            return Ok(term);
        }
        let atom = graph.atom(atom_id).ok_or(TraceError::UnboundAtomId { id: atom_id })?;
        let tangent_atom = builder.borrow_mut().add_constant(atom.example_value.zero_like());
        let tangent = LinearTerm::from_staged_parts(tangent_atom, builder.clone());
        tangents[atom_id] = Some(tangent.clone());
        Ok(tangent)
    }

    let graph = program.graph();
    let zero = graph
        .input_atoms()
        .first()
        .and_then(|input_atom| graph.atom(*input_atom))
        .map(|atom| atom.example_value.zero_like())
        .ok_or(TraceError::EmptyParameterizedValue)?;
    let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
    let mut tangents = vec![None; graph.atom_count()];
    for input_atom in graph.input_atoms().iter().copied() {
        let input = graph.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
        let tangent_atom = builder
            .borrow_mut()
            .add_input_abstract(input.abstract_value.clone(), input.example_value.zero_like());
        tangents[input_atom] = Some(LinearTerm::from_staged_parts(tangent_atom, builder.clone()));
    }

    for equation in graph.equations() {
        let input_duals = equation
            .inputs
            .iter()
            .copied()
            .map(|input_atom| {
                let atom = graph.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
                Ok(JvpTracer {
                    primal: atom.example_value.clone(),
                    tangent: tangent_for_atom(graph, &builder, tangents.as_mut_slice(), input_atom)?,
                })
            })
            .collect::<Result<Vec<_>, TraceError>>()?;
        let output_duals = apply_program_jvp_rule(equation.op.as_ref(), input_duals.as_slice())?;
        if output_duals.len() != equation.outputs.len() {
            return Err(TraceError::InvalidOutputCount { expected: equation.outputs.len(), got: output_duals.len() });
        }
        for (output_atom, output_dual) in equation.outputs.iter().copied().zip(output_duals.into_iter()) {
            tangents[output_atom] = Some(output_dual.tangent);
        }
    }

    let output_tangents = graph
        .outputs()
        .iter()
        .copied()
        .map(|output_atom| {
            tangent_for_atom(graph, &builder, tangents.as_mut_slice(), output_atom).map(|term| term.atom)
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(tangents);
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    Ok(LinearProgram {
        program: Program::from_graph(builder.build::<Input, Output>(
            output_tangents,
            graph.input_structure().clone(),
            graph.output_structure().clone(),
        ))
        .simplify()?,
        zero,
        marker: PhantomData,
    })
}

pub(crate) fn transpose_linear_program<V, Input, Output>(
    program: &LinearProgram<V, Input, Output>,
) -> Result<LinearProgram<V, Output, Input>, TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + MatrixOps,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
{
    fn accumulate<V>(
        builder: &mut ProgramBuilder<V>,
        adjoints: &mut [Option<AtomId>],
        atom: AtomId,
        contribution: AtomId,
    ) -> Result<(), TraceError>
    where
        V: TraceValue + FloatExt,
    {
        adjoints[atom] = Some(match adjoints[atom] {
            Some(existing) => builder.add_equation(Arc::new(AddOp), vec![existing, contribution])?[0],
            None => contribution,
        });
        Ok(())
    }

    let graph = program.program.graph();
    let mut builder = ProgramBuilder::<V>::new();
    let mut output_cotangent_inputs = Vec::with_capacity(graph.outputs().len());
    for output in graph.outputs() {
        let output_atom = graph.atom(*output).ok_or(TraceError::UnboundAtomId { id: *output })?;
        output_cotangent_inputs.push(
            builder.add_input_abstract(output_atom.abstract_value.clone(), output_atom.example_value.zero_like()),
        );
    }

    let mut adjoints = vec![None; graph.atom_count()];
    for (cotangent, output) in output_cotangent_inputs.into_iter().zip(graph.outputs().iter().copied()) {
        accumulate(&mut builder, adjoints.as_mut_slice(), output, cotangent)?;
    }

    for equation in graph.equations().iter().rev() {
        let equation_output_cotangents =
            equation.outputs.iter().map(|output| adjoints[*output]).collect::<Option<Vec<_>>>();
        let Some(equation_output_cotangents) = equation_output_cotangents else {
            continue;
        };
        let input_cotangents = transpose_program_op(
            equation.op.as_ref(),
            &mut builder,
            equation.inputs.as_slice(),
            equation.outputs.as_slice(),
            equation_output_cotangents.as_slice(),
        )?;
        for (input, contribution) in equation.inputs.iter().copied().zip(input_cotangents) {
            if let Some(contribution) = contribution {
                accumulate(&mut builder, adjoints.as_mut_slice(), input, contribution)?;
            }
        }
    }

    let zero_atom = builder.add_constant(program.zero.clone());
    let outputs = graph
        .input_atoms()
        .iter()
        .copied()
        .map(|input| adjoints[input].unwrap_or(zero_atom))
        .collect::<Vec<_>>();
    Ok(LinearProgram {
        program: Program::from_graph(builder.build::<Output, Input>(
            outputs,
            graph.output_structure().clone(),
            graph.input_structure().clone(),
        ))
        .simplify()?,
        zero: program.zero.clone(),
        marker: PhantomData,
    })
}

fn lift_traced_constant<V>(constant: &V, inputs: &[JitTracer<V>]) -> Result<JitTracer<V>, TraceError>
where
    V: TraceValue,
{
    let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
    let atom = exemplar.builder_handle().borrow_mut().add_constant(constant.clone());
    Ok(JitTracer::from_staged_parts(constant.clone(), atom, exemplar.builder_handle(), exemplar.staging_error_handle()))
}

fn lift_linearized_traced_constant<V>(
    constant: &V,
    inputs: &[Linearized<JitTracer<V>>],
) -> Result<Linearized<JitTracer<V>>, TraceError>
where
    V: TransformLeaf,
{
    let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
    let primal = lift_traced_constant(constant, std::slice::from_ref(&exemplar.primal))?;
    let tangent_atom = exemplar.tangent.builder_handle().borrow_mut().add_constant(primal.zero_like());
    let tangent = LinearTerm::from_staged_parts(tangent_atom, exemplar.tangent.builder_handle());
    Ok(Linearized { primal, tangent })
}

fn replay_program_graph_with<GraphInput, GraphOutput, V, R, LiftConstant, ApplyOp>(
    graph: &Graph<ProgramOpRef<V>, V, GraphInput, GraphOutput>,
    inputs: Vec<R>,
    lift_constant: LiftConstant,
    apply_op: ApplyOp,
) -> Result<Vec<R>, TraceError>
where
    GraphInput: Parameterized<V>,
    GraphOutput: Parameterized<V>,
    V: TraceValue,
    R: Clone,
    LiftConstant: Fn(&V, &[R]) -> Result<R, TraceError>,
    ApplyOp: Fn(&dyn Op<V>, Vec<R>) -> Result<Vec<R>, TraceError>,
{
    let mut values = vec![None; graph.atom_count()];
    for (atom_id, value) in graph.input_atoms().iter().copied().zip(inputs.iter().cloned()) {
        values[atom_id] = Some(value);
    }

    let mut equation_by_first_output = vec![None; graph.atom_count()];
    for (equation_index, equation) in graph.equations().iter().enumerate() {
        if let Some(first_output) = equation.outputs.first() {
            equation_by_first_output[*first_output] = Some(equation_index);
        }
    }

    for atom_id in 0..graph.atom_count() {
        let atom = graph.atom(atom_id).expect("atom IDs should be dense");
        match atom.source {
            crate::tracing_v2::AtomSource::Input => {}
            crate::tracing_v2::AtomSource::Constant => {
                let seed_inputs = inputs.iter().cloned().chain(values.iter().flatten().cloned()).collect::<Vec<_>>();
                if seed_inputs.is_empty() {
                    return Err(TraceError::EmptyParameterizedValue);
                }
                values[atom_id] = Some(lift_constant(&atom.example_value, seed_inputs.as_slice())?);
            }
            crate::tracing_v2::AtomSource::Derived => {
                let Some(equation_index) = equation_by_first_output[atom_id] else {
                    continue;
                };
                let equation = &graph.equations()[equation_index];
                let input_values = equation
                    .inputs
                    .iter()
                    .map(|input| values[*input].clone().ok_or(TraceError::UnboundAtomId { id: *input }))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = apply_op(equation.op.as_ref(), input_values)?;
                for (output_atom, output_value) in equation.outputs.iter().copied().zip(outputs) {
                    values[output_atom] = Some(output_value);
                }
            }
        }
    }

    graph
        .outputs()
        .iter()
        .map(|output| values[*output].clone().ok_or(TraceError::UnboundAtomId { id: *output }))
        .collect()
}

pub(crate) fn replay_program_graph_linearized_jit<GraphInput, GraphOutput, V>(
    graph: &Graph<ProgramOpRef<V>, V, GraphInput, GraphOutput>,
    inputs: Vec<Linearized<JitTracer<V>>>,
) -> Result<Vec<Linearized<JitTracer<V>>>, TraceError>
where
    GraphInput: Parameterized<V>,
    GraphOutput: Parameterized<V>,
    V: TransformLeaf,
{
    replay_program_graph_with(graph, inputs, lift_linearized_traced_constant, |op, values| {
        op.replay_linearized_jit(values)
    })
}

fn try_linearize_traced_program<V>(
    program: &Program<V, Vec<V>, Vec<V>>,
    primals: Vec<JitTracer<V>>,
) -> Result<(Vec<JitTracer<V>>, LinearProgram<JitTracer<V>, Vec<JitTracer<V>>, Vec<JitTracer<V>>>), TraceError>
where
    V: TransformLeaf,
{
    let zero = primals.first().map(ZeroLike::zero_like).ok_or(TraceError::EmptyParameterizedValue)?;
    let input_count = primals.len();
    let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
    let traced_input = primals
        .into_iter()
        .map(|primal| {
            let atom = builder.borrow_mut().add_input(&primal);
            Linearized { primal, tangent: LinearTerm::from_staged_parts(atom, builder.clone()) }
        })
        .collect::<Vec<_>>();
    let traced_output = replay_program_graph_linearized_jit(program.graph(), traced_input)?;
    let primal_outputs = traced_output.iter().map(|output| output.primal.clone()).collect::<Vec<_>>();
    let tangent_outputs = traced_output.iter().map(|output| output.tangent.atom()).collect::<Vec<_>>();
    drop(traced_output);
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    let program = Program::from_graph(builder.build::<Vec<JitTracer<V>>, Vec<JitTracer<V>>>(
        tangent_outputs,
        vec![Placeholder; input_count],
        vec![Placeholder; primal_outputs.len()],
    ))
    .simplify()?;
    Ok((primal_outputs, LinearProgram::from_program(program, zero)))
}

pub(crate) fn try_jvp_program<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Input, Output>), TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + MatrixOps,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Result<Output::To<JitTracer<V>>, TraceError>,
{
    let (primal_output, program) = try_trace_program(function, primals)?;
    Ok((primal_output, linearize_program(&program)?))
}

/// Runs JVP for already traced inputs by staging the inner function once over base values and
/// replaying the resulting pushforward in the surrounding trace.
pub(crate) fn try_jvp_traced<F, Input, Output, V>(
    function: F,
    primals: Input,
    tangents: Input,
) -> Result<(Output, Output), TraceError>
where
    V: TransformLeaf,
    Input: Parameterized<JitTracer<V>, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<V>,
    Output: Parameterized<JitTracer<V>, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<V>,
    Input::To<V>: Parameterized<V, To<JitTracer<V>> = Input>,
    Output::To<V>: Parameterized<V, To<JitTracer<V>> = Output>,
    F: FnOnce(Input) -> Result<Output, TraceError>,
{
    if primals.parameter_structure() != tangents.parameter_structure() {
        return Err(TraceError::MismatchedParameterStructure);
    }

    let input_structure = primals.parameter_structure();
    let traced_primals = primals.into_parameters().collect::<Vec<_>>();
    let traced_tangents = tangents.into_parameters().collect::<Vec<_>>();
    let staged_primals = Input::To::<V>::from_parameters(
        input_structure.clone(),
        traced_primals.iter().map(|primal| primal.value.clone()).collect::<Vec<_>>(),
    )?;
    let (primal_output, traced_program): (Output::To<V>, Program<V, Input::To<V>, Output::To<V>>) =
        try_trace_program::<_, Input::To<V>, Output::To<V>, V>(
            move |staged_input| {
                let adapted_input =
                    Input::from_parameters(input_structure, staged_input.into_parameters().collect::<Vec<_>>())?;
                function(adapted_input)
            },
            staged_primals,
        )?;
    let output_structure = primal_output.parameter_structure();
    let traced_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
        vec![Placeholder; traced_primals.len()],
        vec![Placeholder; primal_output.parameter_count()],
    ))
    .simplify()?;
    let (traced_primal_output, pushforward) = try_linearize_traced_program(&traced_program, traced_primals)?;
    let traced_tangent_output = pushforward.call(traced_tangents)?;
    Ok((
        Output::from_parameters(output_structure.clone(), traced_primal_output)?,
        Output::from_parameters(output_structure, traced_tangent_output)?,
    ))
}

/// Runs a forward trace and returns both the primal output and the staged pushforward.
#[allow(private_bounds)]
pub fn jvp_program<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Input, Output>), TraceError>
where
    V: TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Output::To<JitTracer<V>>,
{
    try_jvp_program(|input| Ok(function(input)), primals)
}

/// Alias for [`jvp_program`] that emphasizes the returned linear map.
#[allow(private_bounds)]
pub fn linearize<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Input, Output>), TraceError>
where
    V: TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Output::To<JitTracer<V>>,
{
    jvp_program(function, primals)
}

pub(crate) fn try_vjp<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Output, Input>), TraceError>
where
    V: TraceValue + FloatExt + ZeroLike + MatrixOps,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Result<Output::To<JitTracer<V>>, TraceError>,
{
    let (output, pushforward) = try_jvp_program::<F, Input, Output, V>(function, primals)?;
    Ok((output, pushforward.transpose()?))
}

/// Returns the primal output together with a pullback produced by transposing the staged pushforward.
#[allow(private_bounds)]
pub fn vjp<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<V, Output, Input>), TraceError>
where
    V: TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Output::To<JitTracer<V>>,
{
    try_vjp(|input| Ok(function(input)), primals)
}

fn try_grad<F, Input, V>(function: F, primals: Input) -> Result<Input, TraceError>
where
    V: TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Result<JitTracer<V>, TraceError>,
{
    let (output, pullback): (V, LinearProgram<V, V, Input>) = try_vjp(function, primals)?;
    pullback.call(output.one_like())
}

/// Dispatch trait used by [`grad`] so it can operate both on concrete values and on already traced values.
#[doc(hidden)]
pub(crate) trait GradInvocationLeaf<Input>: Parameter + Sized
where
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
{
    /// Base leaf value used for the staged inner program.
    type Base: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps;

    /// Return type produced by [`grad`] for the corresponding input regime.
    type Return;

    /// Traced input type expected by the user-provided function.
    type FunctionInput;

    /// Invokes [`grad`] for one concrete leaf regime.
    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<Self::Base>;
}

impl<V, Input> GradInvocationLeaf<Input> for V
where
    V: TransformLeaf,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
{
    type Base = V;
    type Return = Input;
    type FunctionInput = Input::To<JitTracer<V>>;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<Self::Base>,
    {
        try_grad(|input| Ok(function(input)), primals)
    }
}

impl<V, Input> GradInvocationLeaf<Input> for JitTracer<V>
where
    V: TransformLeaf,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<V>,
{
    type Base = V;
    type Return = Input;
    type FunctionInput = Input;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<Self::Base>,
    {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let staged_primals = Input::To::<V>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|primal| primal.value.clone()).collect::<Vec<_>>(),
        )?;
        let (_, traced_program): (V, Program<V, Input::To<V>, V>) = try_trace_program(
            |staged_input| {
                let adapted_input = Input::from_parameters(
                    input_structure.clone(),
                    staged_input.into_parameters().collect::<Vec<_>>(),
                )?;
                Ok(function(adapted_input))
            },
            staged_primals,
        )?;
        let traced_program = Program::from_graph(
            traced_program
                .graph()
                .clone_with_structures::<Vec<V>, Vec<V>>(vec![Placeholder; traced_primals.len()], vec![Placeholder; 1]),
        )
        .simplify()?;
        let (outputs, pushforward) = try_linearize_traced_program(&traced_program, traced_primals)?;
        if outputs.len() != 1 {
            return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
        }
        let pullback = pushforward.transpose()?;
        let traced_gradient = pullback.call(vec![outputs[0].one_like()])?;
        Ok(Input::from_parameters(input_structure, traced_gradient)?)
    }
}

/// Computes the reverse-mode gradient of a scalar-output function.
#[allow(private_bounds, private_interfaces)]
pub fn grad<F, Input, Leaf>(
    function: F,
    primals: Input,
) -> Result<<Leaf as GradInvocationLeaf<Input>>::Return, TraceError>
where
    Leaf: GradInvocationLeaf<Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as GradInvocationLeaf<Input>>::FunctionInput,
    ) -> JitTracer<<Leaf as GradInvocationLeaf<Input>>::Base>,
{
    Leaf::invoke(function, primals)
}

/// Dispatch trait used by [`value_and_grad`] so it can operate both on concrete values and on already traced values.
#[doc(hidden)]
pub(crate) trait ValueAndGradInvocationLeaf<Input>: Parameter + Sized
where
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
{
    /// Base leaf value used for the staged inner program.
    type Base: TransformLeaf;

    /// Return type produced by [`value_and_grad`] for the corresponding input regime.
    type Return;

    /// Traced input type expected by the user-provided function.
    type FunctionInput;

    /// Invokes [`value_and_grad`] for one concrete leaf regime.
    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<Self::Base>;
}

impl<V, Input> ValueAndGradInvocationLeaf<Input> for V
where
    V: TransformLeaf,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
{
    type Base = V;
    type Return = (V, Input);
    type FunctionInput = Input::To<JitTracer<V>>;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<Self::Base>,
    {
        let (output, pullback): (V, LinearProgram<V, V, Input>) = vjp(function, primals)?;
        let gradient = pullback.call(output.one_like())?;
        Ok((output, gradient))
    }
}

impl<V, Input> ValueAndGradInvocationLeaf<Input> for JitTracer<V>
where
    V: TransformLeaf,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<V>,
    Input::To<V>: Parameterized<V, To<JitTracer<V>> = Input>,
{
    type Base = V;
    type Return = (JitTracer<V>, Input);
    type FunctionInput = Input;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<Self::Base>,
    {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let staged_primals = Input::To::<V>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|primal| primal.value.clone()).collect::<Vec<_>>(),
        )?;
        let (_, traced_program): (V, Program<V, Input::To<V>, V>) = try_trace_program(
            |staged_input| {
                let adapted_input = Input::from_parameters(
                    input_structure.clone(),
                    staged_input.into_parameters().collect::<Vec<_>>(),
                )?;
                Ok(function(adapted_input))
            },
            staged_primals,
        )?;
        let traced_program = Program::from_graph(
            traced_program
                .graph()
                .clone_with_structures::<Vec<V>, Vec<V>>(vec![Placeholder; traced_primals.len()], vec![Placeholder; 1]),
        )
        .simplify()?;
        let (outputs, pushforward) = try_linearize_traced_program(&traced_program, traced_primals)?;
        if outputs.len() != 1 {
            return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
        }
        let traced_output = outputs[0].clone();
        let pullback = pushforward.transpose()?;
        let traced_gradient = pullback.call(vec![traced_output.one_like()])?;
        Ok((traced_output, Input::from_parameters(input_structure, traced_gradient)?))
    }
}

/// Computes both the primal scalar output and its reverse-mode gradient.
#[allow(private_bounds, private_interfaces)]
pub fn value_and_grad<F, Input, Leaf>(
    function: F,
    primals: Input,
) -> Result<<Leaf as ValueAndGradInvocationLeaf<Input>>::Return, TraceError>
where
    Leaf: ValueAndGradInvocationLeaf<Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as ValueAndGradInvocationLeaf<Input>>::FunctionInput,
    ) -> JitTracer<<Leaf as ValueAndGradInvocationLeaf<Input>>::Base>,
{
    Leaf::invoke(function, primals)
}

/// Leaf type that can be materialized into a dense finite-dimensional coordinate representation.
pub trait CoordinateValue: TraceValue + ZeroLike + OneLike {
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

fn try_jacfwd<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue + TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Result<Output::To<JitTracer<V>>, TraceError>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let basis_inputs = standard_basis::<Input, V>(&input_structure, input_parameters.as_slice())?;
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
    let (output, pushforward) = try_jvp_program::<F, Input, Output, V>(function, primals)?;
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
#[allow(private_bounds)]
pub fn jacfwd<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue + TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Output::To<JitTracer<V>>,
{
    try_jacfwd::<_, Input, Output, V>(|input| Ok(function(input)), primals)
}

fn try_jacrev<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue + TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Result<Output::To<JitTracer<V>>, TraceError>,
{
    let input_structure = primals.parameter_structure();
    let input_parameters = primals.into_parameters().collect::<Vec<_>>();
    let input_coordinate_counts = coordinate_counts(input_parameters.as_slice());
    let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
    let (output, pullback) = try_vjp::<F, Input, Output, V>(function, primals)?;
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
#[allow(private_bounds)]
pub fn jacrev<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue + TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Output::To<JitTracer<V>>,
{
    try_jacrev::<_, Input, Output, V>(|input| Ok(function(input)), primals)
}

/// Materializes a dense Hessian by applying `jacfwd` to a gradient helper.
///
/// In the current prototype, callers pass a first-derivative function (for example `first_derivative`)
/// because Rust does not yet let this API re-instantiate an arbitrary closure at a deeper trace level.
#[allow(private_bounds)]
pub fn hessian<F, Input, V>(
    gradient_function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Input::ParameterStructure>, TraceError>
where
    V: CoordinateValue + TransformLeaf,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<V>>,
    F: FnOnce(Input::To<JitTracer<V>>) -> Input::To<JitTracer<V>>,
{
    jacfwd::<F, Input, Input, V>(gradient_function, primals)
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, Mul, Neg};

    use indoc::indoc;

    use crate::tracing_v2::{FloatExt, test_support};

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    fn quadratic_plus_sin<T>(x: T) -> T
    where
        T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        x.clone() * x.clone() + x.sin()
    }

    fn bilinear_sin<T>(inputs: (T, T)) -> T
    where
        T: Clone + FloatExt + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin()
    }

    #[test]
    fn linearize_returns_the_primal_output_and_pushforward() {
        let (primal, pushforward) = linearize(quadratic_plus_sin, 2.0f64).unwrap();

        approx_eq(primal, 2.0f64.powi(2) + 2.0f64.sin());
        approx_eq(pushforward.call(1.5f64).unwrap(), (4.0 + 2.0f64.cos()) * 1.5);
        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = scale %0
                    %2:f64[] = scale %0
                    %3:f64[] = add %1 %2
                    %4:f64[] = scale %0
                    %5:f64[] = add %3 %4
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn jvp_program_and_linearize_stage_the_same_pushforward() {
        let (_, from_jvp_program) = jvp_program(quadratic_plus_sin, 2.0f64).unwrap();
        let (_, from_linearize) = linearize(quadratic_plus_sin, 2.0f64).unwrap();

        approx_eq(from_jvp_program.call(1.0f64).unwrap(), from_linearize.call(1.0f64).unwrap());
        assert_eq!(
            from_jvp_program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = scale %0
                    %2:f64[] = scale %0
                    %3:f64[] = add %1 %2
                    %4:f64[] = scale %0
                    %5:f64[] = add %3 %4
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(from_jvp_program.to_string(), from_linearize.to_string());
    }

    #[test]
    fn transposed_linear_program_matches_the_reverse_mode_pullback() {
        let (primal, pushforward) = linearize(bilinear_sin, (2.0f64, 3.0f64)).unwrap();
        let pullback = pushforward.transpose().unwrap();
        let cotangent = pullback.call(1.0f64).unwrap();

        approx_eq(primal, 2.0 * 3.0 + 2.0f64.sin());
        approx_eq(cotangent.0, 3.0 + 2.0f64.cos());
        approx_eq(cotangent.1, 2.0);
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = scale %0
                    %2:f64[] = scale %0
                    %3:f64[] = add %1 %2
                    %4:f64[] = scale %0
                in (%3, %4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn linear_program_display_delegates_to_the_underlying_graph() {
        let (_, pushforward): (f64, LinearProgram<f64, f64, f64>) = linearize(quadratic_plus_sin, 2.0f64).unwrap();

        assert_eq!(
            pushforward.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = scale %0
                    %2:f64[] = scale %0
                    %3:f64[] = add %1 %2
                    %4:f64[] = scale %0
                    %5:f64[] = add %3 %4
                in (%5)
            "}
            .trim_end(),
        );
        assert_eq!(pushforward.to_string(), pushforward.program().graph().to_string());
        test_support::assert_quadratic_pushforward_rendering();
    }
}
