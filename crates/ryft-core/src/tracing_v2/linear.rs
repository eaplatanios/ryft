//! Linearization, transposition, and higher-order differentiation utilities.
//!
//! This module turns forward-mode traces into staged linear programs, transposes those programs for reverse-mode
//! differentiation, and materializes dense Jacobians/Hessians for coordinate-based leaf types.

use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    marker::PhantomData,
    rc::Rc,
};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        OneLike, TraceError, TraceInput, TraceOutput, Traceable, Value, ZeroLike,
        batch::{Batch, stack, unstack},
        engine::Engine,
        forward::{JvpTracer, TangentSpace},
        graph::{Atom, AtomId, Equation, Graph, GraphBuilder},
        jit::{
            CompiledFunction, JitTracer, try_jit, try_jit_for_operation, try_trace_program,
            try_trace_program_for_operation,
        },
        operations::{
            CoreLinearProgramOp, CoreLinearReplayOp, DifferentiableOp, InterpretableOp, LinearAddOperation,
            LinearNegOperation, LinearScaleOperation, Op, RematerializeTracingOperation,
            rematerialize::{FlatTracedRematerialize, RematerializeOp},
        },
        program::{LinearProgramBuilder, LinearProgramOpRef, Program, ProgramBuilder},
    },
    types::{ArrayType, Type, Typed},
};

/// Tangent representation backed by atoms in a staged linear graph.
#[derive(Clone, Parameter)]
pub struct LinearTerm<T: Type + Display, V: Traceable<T> + Parameter, O: Clone = LinearProgramOpRef<V>> {
    atom: AtomId,
    builder: Rc<RefCell<GraphBuilder<O, T, V>>>,
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> std::fmt::Debug for LinearTerm<T, V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("LinearTerm").field("atom", &self.atom).finish()
    }
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> LinearTerm<T, V, O> {
    #[inline]
    pub fn atom(&self) -> AtomId {
        self.atom
    }

    #[inline]
    pub fn builder_handle(&self) -> Rc<RefCell<GraphBuilder<O, T, V>>> {
        self.builder.clone()
    }

    #[inline]
    pub fn from_staged_parts(atom: AtomId, builder: Rc<RefCell<GraphBuilder<O, T, V>>>) -> Self {
        Self { atom, builder }
    }

    /// Stages a multi-input operation in the tangent program builder.
    ///
    /// Shape validation is performed via [`Op::abstract_eval`]. Concrete evaluation is intentionally
    /// skipped because tangent-program outputs remain abstract until the staged linear program is
    /// replayed on concrete tangents.
    pub fn apply_staged_op(inputs: &[Self], op: O, output_count: usize) -> Result<Vec<Self>, TraceError>
    where
        O: Op<T>,
    {
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
        let mut borrow = builder.borrow_mut();
        let output_abstracts = op.abstract_eval(
            &input_atoms
                .iter()
                .map(|id| borrow.atom(*id).expect("staged input should exist").tpe().into_owned())
                .collect::<Vec<_>>(),
        )?;
        let output_atoms = borrow.add_equation_prevalidated(op, input_atoms, output_abstracts);
        drop(borrow);
        if output_atoms.len() != output_count {
            return Err(TraceError::InvalidOutputCount { expected: output_count, got: output_atoms.len() });
        }
        Ok(output_atoms.into_iter().map(|atom| Self { atom, builder: builder.clone() }).collect())
    }

    /// Stages a unary linear op in the program builder.
    ///
    /// The output atom reuses the abstract type of the input atom, which is valid for shape-preserving
    /// linear operations in tangent programs.
    #[inline]
    pub fn apply_linear_op(self, op: O) -> Self {
        let mut borrow = self.builder.borrow_mut();
        let input_atom = borrow.atom(self.atom).expect("staged input should exist");
        let abstract_value = input_atom.tpe().into_owned();
        let atom = borrow.add_equation_prevalidated(op, vec![self.atom], vec![abstract_value])[0];
        drop(borrow);
        Self { atom, builder: self.builder }
    }

    /// Stages an addition of two tangent terms.
    #[inline]
    pub fn add(self, rhs: Self) -> Self
    where
        O: LinearAddOperation<T, V>,
    {
        debug_assert!(Rc::ptr_eq(&self.builder, &rhs.builder));
        let mut borrow = self.builder.borrow_mut();
        let input_atom = borrow.atom(self.atom).expect("staged input should exist");
        let abstract_value = input_atom.tpe().into_owned();
        let atom =
            borrow.add_equation_prevalidated(O::linear_add_op(), vec![self.atom, rhs.atom], vec![abstract_value])[0];
        drop(borrow);
        Self { atom, builder: self.builder }
    }

    /// Stages a negation of this tangent term.
    #[inline]
    pub fn neg(self) -> Self
    where
        O: LinearNegOperation<T, V>,
    {
        self.apply_linear_op(O::linear_neg_op())
    }

    /// Stages a scaling of this tangent term by a concrete factor.
    #[inline]
    pub fn scale(self, factor: V) -> Self
    where
        O: LinearScaleOperation<T, V>,
    {
        self.apply_linear_op(O::linear_scale_op(factor))
    }
}

impl<
    T: Type + Display,
    V: Traceable<T> + ZeroLike,
    O: LinearAddOperation<T, V> + LinearNegOperation<T, V> + LinearScaleOperation<T, V>,
> TangentSpace<T, V> for LinearTerm<T, V, O>
{
    #[inline]
    fn add(lhs: Self, rhs: Self) -> Self {
        debug_assert!(Rc::ptr_eq(&lhs.builder, &rhs.builder));
        let mut borrow = lhs.builder.borrow_mut();
        let input_atom = borrow.atom(lhs.atom).expect("staged input should exist");
        let abstract_value = input_atom.tpe().into_owned();
        let atom =
            borrow.add_equation_prevalidated(O::linear_add_op(), vec![lhs.atom, rhs.atom], vec![abstract_value])[0];
        drop(borrow);
        Self { atom, builder: lhs.builder }
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
pub type Linearized<V, O = LinearProgramOpRef<V>> = JvpTracer<V, LinearTerm<ArrayType, V, O>>;

type LinearizedTracedValue<V, O, L> =
    Linearized<JitTracer<ArrayType, V, O, L>, LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>>;

type TracedLinearProgram<V, O, L> = LinearProgram<
    ArrayType,
    JitTracer<ArrayType, V, O, L>,
    Vec<JitTracer<ArrayType, V, O, L>>,
    Vec<JitTracer<ArrayType, V, O, L>>,
    LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>,
>;

#[inline]
fn flat_leaf_parameter_structure(count: usize) -> Vec<Placeholder> {
    vec![Placeholder; count]
}

/// Staged linear map produced by `linearize`, `jvp_program`, or `vjp`.
pub struct LinearProgram<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O: Clone = LinearProgramOpRef<V>,
> {
    program: Program<T, V, Input, Output, O>,
    zero: V,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<
    T: Type + Display,
    V: Clone + Traceable<T>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: Clone,
> Clone for LinearProgram<T, V, Input, Output, O>
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), zero: self.zero.clone(), marker: PhantomData }
    }
}

impl<T: Type + Display, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone>
    LinearProgram<T, V, Input, Output, O>
{
    #[inline]
    pub fn from_program(program: Program<T, V, Input, Output, O>, zero: V) -> Self {
        Self { program, zero, marker: PhantomData }
    }

    /// Returns the staged graph backing this linear program.
    #[inline]
    pub fn program(&self) -> &Program<T, V, Input, Output, O> {
        &self.program
    }

    /// Applies the linear program to a concrete input tangent or cotangent.
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        O: InterpretableOp<T, V>,
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.program.call(input)
    }
}

impl<V: Traceable<ArrayType>, Input: Parameterized<V>, Output: Parameterized<V>, O: Clone>
    LinearProgram<ArrayType, V, Input, Output, O>
{
    /// Transposes the linear program, turning a pushforward into a pullback.
    #[allow(private_bounds)]
    pub fn transpose(&self) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TraceError>
    where
        V: ZeroLike,
        O: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V> + Clone,
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        transpose_linear_program(self)
    }
}

impl<V: Traceable<ArrayType>, Input: Parameterized<V>, Output: Parameterized<V>> Display
    for LinearProgram<ArrayType, V, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.program, formatter)
    }
}

/// Applies one primitive's semantic transpose rule while transposing a staged linear program.
///
/// The transpose builder's cotangent atoms are wrapped as [`LinearTerm<ArrayType, V>`] so primitive rules can
/// emit staged cotangent contributions directly.
///
/// The returned atom ids are staged cotangent contributions in the transpose builder, aligned with
/// the forward primitive inputs.
///
/// # Parameters
///   - `op`: primitive whose transpose rule should be applied.
///   - `builder`: transpose-program builder that owns the staged cotangent atoms created while
///     constructing the pullback program.
///   - `output_cotangents`: transpose-builder atom ids for the already-staged cotangents of
///     the primitive outputs.
fn transpose<V, O>(
    op: &O,
    builder: &Rc<RefCell<LinearProgramBuilder<V, O>>>,
    output_cotangents: &[AtomId],
) -> Result<Vec<Option<AtomId>>, TraceError>
where
    V: Traceable<ArrayType>,
    O: CoreLinearProgramOp<V> + Clone,
{
    let cotangent_terms = output_cotangents
        .iter()
        .map(|cotangent| LinearTerm::from_staged_parts(*cotangent, builder.clone()))
        .collect::<Vec<_>>();
    Ok(op
        .transpose(cotangent_terms.as_slice())?
        .into_iter()
        .map(|term| term.map(|term| term.atom()))
        .collect())
}

pub(crate) fn linearize_program<Input, Output, V, O, L>(
    engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
    program: &Program<ArrayType, V, Input, Output, O>,
    input_primals: Vec<V>,
) -> Result<LinearProgram<ArrayType, V, Input, Output, L>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    L: Clone + Op<ArrayType>,
    O: Clone + DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L>,
{
    fn tangent_for_atom<V, Input, Output, GraphOperation, LinearOperation>(
        _graph: &Graph<GraphOperation, ArrayType, V, Input, Output>,
        primal_values: &[Option<V>],
        builder: &Rc<RefCell<LinearProgramBuilder<V, LinearOperation>>>,
        tangents: &mut [Option<LinearTerm<ArrayType, V, LinearOperation>>],
        atom_id: AtomId,
    ) -> Result<LinearTerm<ArrayType, V, LinearOperation>, TraceError>
    where
        V: Traceable<ArrayType> + ZeroLike,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
        GraphOperation: Clone + Op<ArrayType>,
        LinearOperation: Clone + Op<ArrayType>,
    {
        if let Some(term) = tangents[atom_id].clone() {
            return Ok(term);
        }
        let primal = primal_values[atom_id].as_ref().ok_or(TraceError::UnboundAtomId { id: atom_id })?;
        let tangent_atom = builder.borrow_mut().add_constant(primal.zero_like());
        let tangent = LinearTerm::from_staged_parts(tangent_atom, builder.clone());
        tangents[atom_id] = Some(tangent.clone());
        Ok(tangent)
    }

    let graph = program.graph();
    if input_primals.len() != graph.input_atoms().len() {
        return Err(TraceError::InvalidInputCount { expected: graph.input_atoms().len(), got: input_primals.len() });
    }
    let zero = input_primals.first().map(ZeroLike::zero_like).ok_or(TraceError::EmptyParameterizedValue)?;
    let builder = Rc::new(RefCell::new(LinearProgramBuilder::<V, L>::new()));
    let mut primals: Vec<Option<V>> = vec![None; graph.atom_count()];
    let mut tangents: Vec<Option<LinearTerm<ArrayType, V, L>>> = vec![None; graph.atom_count()];
    for (input_atom, input_primal) in graph.input_atoms().iter().copied().zip(input_primals.into_iter()) {
        let tangent_atom = builder.borrow_mut().add_input(&input_primal.zero_like());
        tangents[input_atom] = Some(LinearTerm::from_staged_parts(tangent_atom, builder.clone()));
        primals[input_atom] = Some(input_primal);
    }
    for (atom_id, atom) in graph.atoms_iter() {
        if let Atom::Constant { value } = atom {
            primals[atom_id] = Some(value.clone());
        }
    }

    for equation in graph.equations() {
        let input_duals = equation
            .inputs
            .iter()
            .copied()
            .map(|input_atom| {
                Ok(JvpTracer {
                    primal: primals[input_atom].clone().ok_or(TraceError::UnboundAtomId { id: input_atom })?,
                    tangent: tangent_for_atom(
                        graph,
                        primals.as_slice(),
                        &builder,
                        tangents.as_mut_slice(),
                        input_atom,
                    )?,
                })
            })
            .collect::<Result<Vec<_>, TraceError>>()?;
        let output_duals = DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L>::jvp(
            &equation.op,
            engine,
            input_duals.as_slice(),
        )?;
        if output_duals.len() != equation.outputs.len() {
            return Err(TraceError::InvalidOutputCount { expected: equation.outputs.len(), got: output_duals.len() });
        }
        for (output_atom, output_dual) in equation.outputs.iter().copied().zip(output_duals.into_iter()) {
            primals[output_atom] = Some(output_dual.primal);
            tangents[output_atom] = Some(output_dual.tangent);
        }
    }

    let output_tangents = graph
        .outputs()
        .iter()
        .copied()
        .map(|output_atom| {
            tangent_for_atom(graph, primals.as_slice(), &builder, tangents.as_mut_slice(), output_atom)
                .map(|term| term.atom)
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

#[allow(private_bounds)]
pub fn transpose_linear_program<V, Input, Output, O>(
    program: &LinearProgram<ArrayType, V, Input, Output, O>,
) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    let zero = program.zero.zero_like();
    transpose_linear_program_with_output_inputs(program, |builder: &mut LinearProgramBuilder<V, O>, _, _| {
        Ok(builder.add_input(&zero))
    })
}

fn transpose_linear_program_with_output_inputs<V, Input, Output, O, F>(
    program: &LinearProgram<ArrayType, V, Input, Output, O>,
    mut make_output_cotangent_input: F,
) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    F: FnMut(&mut LinearProgramBuilder<V, O>, &ArrayType, usize) -> Result<AtomId, TraceError>,
    O: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    fn accumulate<V, O>(
        builder: &Rc<RefCell<LinearProgramBuilder<V, O>>>,
        adjoints: &mut [Option<AtomId>],
        atom: AtomId,
        contribution: AtomId,
    ) -> Result<(), TraceError>
    where
        V: Traceable<ArrayType>,
        O: LinearAddOperation<ArrayType, V> + Op<ArrayType> + Clone,
    {
        adjoints[atom] = Some(match adjoints[atom] {
            Some(existing) => {
                let mut builder_borrow = builder.borrow_mut();
                let abstract_value =
                    builder_borrow.atom(existing).expect("adjoint atom should exist").tpe().into_owned();
                builder_borrow.add_equation_prevalidated(
                    O::linear_add_op(),
                    vec![existing, contribution],
                    vec![abstract_value],
                )[0]
            }
            None => contribution,
        });
        Ok(())
    }

    let graph = program.program.graph();
    let builder = Rc::new(RefCell::new(LinearProgramBuilder::<V, O>::new()));
    let mut output_cotangent_inputs = Vec::with_capacity(graph.outputs().len());
    for (output_index, output) in graph.outputs().iter().enumerate() {
        let output_atom = graph.atom(*output).ok_or(TraceError::UnboundAtomId { id: *output })?;
        let cotangent_input = make_output_cotangent_input(&mut builder.borrow_mut(), &output_atom.tpe(), output_index)?;
        output_cotangent_inputs.push(cotangent_input);
    }

    let mut adjoints = vec![None; graph.atom_count()];
    for (cotangent, output) in output_cotangent_inputs.into_iter().zip(graph.outputs().iter().copied()) {
        accumulate(&builder, adjoints.as_mut_slice(), output, cotangent)?;
    }

    for equation in graph.equations().iter().rev() {
        let equation_output_cotangents =
            equation.outputs.iter().map(|output| adjoints[*output]).collect::<Option<Vec<_>>>();
        let Some(equation_output_cotangents) = equation_output_cotangents else {
            continue;
        };
        let input_cotangents = transpose(&equation.op, &builder, equation_output_cotangents.as_slice())?;
        for (input, contribution) in equation.inputs.iter().copied().zip(input_cotangents) {
            if let Some(contribution) = contribution {
                accumulate(&builder, adjoints.as_mut_slice(), input, contribution)?;
            }
        }
    }

    let zero_atom = builder.borrow_mut().add_constant(program.zero.clone());
    let outputs = graph
        .input_atoms()
        .iter()
        .copied()
        .map(|input| adjoints[input].unwrap_or(zero_atom))
        .collect::<Vec<_>>();
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation(
                "transpose builder should not have outstanding linear terms",
            ));
        }
    };
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

/// Transposes a linear program using concrete output examples to seed the cotangent inputs.
///
/// This variant is useful when the linear program's leaf type cannot be synthesized from bare
/// [`ArrayType`] metadata alone, but the caller still has representative output values available.
#[allow(private_bounds)]
pub fn transpose_linear_program_with_output_examples<V, Input, Output, O>(
    program: &LinearProgram<ArrayType, V, Input, Output, O>,
    output_examples: &[V],
) -> Result<LinearProgram<ArrayType, V, Output, Input, O>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    O: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V> + Clone,
{
    let expected_output_count = program.program().graph().outputs().len();
    if output_examples.len() != expected_output_count {
        return Err(TraceError::InvalidInputCount { expected: expected_output_count, got: output_examples.len() });
    }
    transpose_linear_program_with_output_inputs(program, |builder: &mut LinearProgramBuilder<V, O>, _, output_index| {
        Ok(builder.add_input(&output_examples[output_index].zero_like()))
    })
}

fn lift_traced_constant<V, O: Clone, L: Clone>(
    constant: &V,
    inputs: &[JitTracer<ArrayType, V, O, L>],
) -> Result<JitTracer<ArrayType, V, O, L>, TraceError>
where
    V: Traceable<ArrayType>,
{
    let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
    let atom = exemplar.builder_handle().borrow_mut().add_constant(constant.clone());
    Ok(JitTracer::from_staged_parts(
        atom,
        exemplar.builder_handle(),
        exemplar.staging_error_handle(),
        exemplar.engine(),
    ))
}

fn lift_linearized_traced_constant<V, O: Clone + 'static, L: Clone + 'static>(
    constant: &V,
    inputs: &[LinearizedTracedValue<V, O, L>],
) -> Result<LinearizedTracedValue<V, O, L>, TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
{
    let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
    let primal = lift_traced_constant(constant, std::slice::from_ref(&exemplar.primal))?;
    let tangent_atom = exemplar.tangent.builder_handle().borrow_mut().add_constant(primal.zero_like());
    let tangent = LinearTerm::from_staged_parts(tangent_atom, exemplar.tangent.builder_handle());
    Ok(Linearized { primal, tangent })
}

fn replay_program_graph_with<GraphInput, GraphOutput, V, O, R, LiftConstant, ApplyOp>(
    graph: &Graph<O, ArrayType, V, GraphInput, GraphOutput>,
    inputs: Vec<R>,
    lift_constant: LiftConstant,
    apply_op: ApplyOp,
) -> Result<Vec<R>, TraceError>
where
    GraphInput: Parameterized<V>,
    GraphOutput: Parameterized<V>,
    V: Traceable<ArrayType>,
    O: Clone,
    R: Clone,
    LiftConstant: Fn(&V, &[R]) -> Result<R, TraceError>,
    ApplyOp: Fn(&O, Vec<R>) -> Result<Vec<R>, TraceError>,
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
        match atom {
            Atom::Input { .. } => {}
            Atom::Constant { value } => {
                let seed_inputs = inputs.iter().cloned().chain(values.iter().flatten().cloned()).collect::<Vec<_>>();
                if seed_inputs.is_empty() {
                    return Err(TraceError::EmptyParameterizedValue);
                }
                values[atom_id] = Some(lift_constant(value, seed_inputs.as_slice())?);
            }
            Atom::Derived { .. } => {
                let Some(equation_index) = equation_by_first_output[atom_id] else {
                    continue;
                };
                let equation = &graph.equations()[equation_index];
                let input_values = equation
                    .inputs
                    .iter()
                    .map(|input| values[*input].clone().ok_or(TraceError::UnboundAtomId { id: *input }))
                    .collect::<Result<Vec<_>, _>>()?;
                let outputs = apply_op(&equation.op, input_values)?;
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

pub(crate) fn replay_program_graph_linearized_jit<GraphInput, GraphOutput, V, O, L>(
    graph: &Graph<O, ArrayType, V, GraphInput, GraphOutput>,
    inputs: Vec<LinearizedTracedValue<V, O, L>>,
) -> Result<Vec<LinearizedTracedValue<V, O, L>>, TraceError>
where
    GraphInput: Parameterized<V>,
    GraphOutput: Parameterized<V>,
    V: Traceable<ArrayType> + ZeroLike,
    L: Clone + 'static,
    O: InterpretableOp<ArrayType, LinearizedTracedValue<V, O, L>> + Clone,
{
    replay_program_graph_with(graph, inputs, lift_linearized_traced_constant::<V, O, L>, |op, values| {
        InterpretableOp::<ArrayType, LinearizedTracedValue<V, O, L>>::interpret(op, &values)
    })
}

pub(crate) fn try_linearize_traced_program<V, O, L>(
    program: &Program<ArrayType, V, Vec<V>, Vec<V>, O>,
    primals: Vec<JitTracer<ArrayType, V, O, L>>,
) -> Result<(Vec<JitTracer<ArrayType, V, O, L>>, TracedLinearProgram<V, O, L>), TraceError>
where
    V: Traceable<ArrayType> + ZeroLike,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + 'static,
    O: InterpretableOp<ArrayType, LinearizedTracedValue<V, O, L>> + Clone,
{
    let zero = primals.first().map(ZeroLike::zero_like).ok_or(TraceError::EmptyParameterizedValue)?;
    let input_count = primals.len();
    let builder = Rc::new(RefCell::new(LinearProgramBuilder::<
        JitTracer<ArrayType, V, O, L>,
        LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>,
    >::new()));
    let traced_input = primals
        .into_iter()
        .map(|primal| {
            let atom = builder.borrow_mut().add_input(&primal);
            Linearized { primal, tangent: LinearTerm::from_staged_parts(atom, builder.clone()) }
        })
        .collect::<Vec<_>>();
    let traced_output = replay_program_graph_linearized_jit::<_, _, _, O, L>(program.graph(), traced_input)?;
    let primal_outputs = traced_output.iter().map(|output| output.primal.clone()).collect::<Vec<_>>();
    let tangent_outputs = traced_output.iter().map(|output| output.tangent.atom()).collect::<Vec<_>>();
    drop(traced_output);
    let builder = match Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => {
            return Err(TraceError::InternalInvariantViolation("linearization builder escaped the tracing scope"));
        }
    };
    let program =
        Program::from_graph(builder.build::<Vec<JitTracer<ArrayType, V, O, L>>, Vec<JitTracer<ArrayType, V, O, L>>>(
            tangent_outputs,
            vec![Placeholder; input_count],
            vec![Placeholder; primal_outputs.len()],
        ))
        .simplify()?;
    Ok((primal_outputs.clone(), LinearProgram::from_program(program, zero)))
}

pub fn try_jvp_program<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Input, Output, E::LinearOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + ZeroLike,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::LinearOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
{
    let input_structure = primals.parameter_structure();
    let input_primals: Vec<V> = primals.into_parameters().collect();
    let reconstructed_primals = Input::from_parameters(input_structure, input_primals.iter().cloned())?;
    let (primal_output, program) =
        try_trace_program_for_operation::<_, Input, Output, V, E::TracingOperation, E::LinearOperation>(
            engine,
            function,
            reconstructed_primals,
        )?;
    Ok((
        primal_output,
        linearize_program::<Input, Output, V, E::TracingOperation, E::LinearOperation>(
            engine,
            &program,
            input_primals,
        )?,
    ))
}

#[allow(private_bounds)]
pub(crate) fn try_jvp_traced<F, Input, Output, V, O, L>(
    function: F,
    primals: Input,
    tangents: Input,
) -> Result<(Output, Output), TraceError>
where
    V: Traceable<ArrayType> + ZeroLike + Parameterized<V, ParameterStructure = Placeholder>,
    Input: Parameterized<JitTracer<ArrayType, V, O, L>, ParameterStructure: Clone + PartialEq>,
    Output: Parameterized<JitTracer<ArrayType, V, O, L>, ParameterStructure: Clone>,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + 'static,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Output::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<V>: TraceInput<V, O, L, Traced = Input>,
    Output::To<V>: TraceOutput<V, O, L, Traced = Output>,
    Input::To<ArrayType>: crate::tracing_v2::TypeTracing<ArrayType, V, O, L, Staged = Input::To<V>, Traced = Input>,
    Output::To<ArrayType>: crate::tracing_v2::TypeTracing<ArrayType, V, O, L, Staged = Output::To<V>, Traced = Output>,
    O: InterpretableOp<
            ArrayType,
            Linearized<JitTracer<ArrayType, V, O, L>, LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>>,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>: CoreLinearReplayOp<JitTracer<ArrayType, V, O, L>>,
    F: FnOnce(Input) -> Result<Output, TraceError>,
{
    if primals.parameter_structure() != tangents.parameter_structure() {
        return Err(TraceError::MismatchedParameterStructure);
    }

    let input_structure = primals.parameter_structure();
    let traced_primals = primals.into_parameters().collect::<Vec<_>>();
    let traced_tangents = tangents.into_parameters().collect::<Vec<_>>();
    let staged_input_types = Input::To::<ArrayType>::from_parameters(
        input_structure.clone(),
        traced_primals.iter().map(|primal| primal.tpe().into_owned()).collect::<Vec<_>>(),
    )?;
    let exemplar_engine = traced_primals.first().ok_or(TraceError::EmptyParameterizedValue)?.engine();
    let (primal_output_types, traced_program): (
        Output::To<ArrayType>,
        Program<ArrayType, V, Input::To<V>, Output::To<V>, O>,
    ) = crate::tracing_v2::jit::try_trace_program_from_types_for_operation::<
        _,
        Input::To<ArrayType>,
        Output::To<ArrayType>,
        ArrayType,
        V,
        O,
        L,
    >(
        exemplar_engine,
        move |staged_input| {
            let adapted_input =
                Input::from_parameters(input_structure, staged_input.into_parameters().collect::<Vec<_>>())?;
            function(adapted_input)
        },
        staged_input_types,
    )?;
    let output_structure = primal_output_types.parameter_structure();
    let traced_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
        flat_leaf_parameter_structure(traced_primals.len()),
        flat_leaf_parameter_structure(output_structure.parameter_count()),
    ))
    .simplify()?;
    let (traced_primal_output, pushforward) = try_linearize_traced_program::<V, O, L>(&traced_program, traced_primals)?;
    let traced_tangent_output = pushforward.call(traced_tangents)?;
    Ok((
        Output::from_parameters(output_structure.clone(), traced_primal_output)?,
        Output::from_parameters(output_structure, traced_tangent_output)?,
    ))
}

/// Runs a forward trace and returns both the primal output and the staged pushforward.
pub fn jvp_program<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Input, Output, E::LinearOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + ZeroLike,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Output::Traced,
    E::LinearOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
{
    try_jvp_program(engine, |input| Ok(function(input)), primals)
}

/// Alias for [`jvp_program`] that emphasizes the returned linear map.
pub fn linearize<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Input, Output, E::LinearOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + ZeroLike,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Output::Traced,
    E::LinearOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    Output::ParameterStructure: Clone,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
{
    jvp_program(engine, function, primals)
}

#[allow(private_bounds)]
pub fn try_vjp<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Output, Input, E::LinearOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + ZeroLike + OneLike,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
    E::LinearOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V>,
{
    let (output, pushforward) = try_jvp_program::<E, F, Input, Output, V>(engine, function, primals)?;
    let output_examples = output.parameters().cloned().collect::<Vec<_>>();
    let pullback = transpose_linear_program_with_output_examples(&pushforward, output_examples.as_slice())?;
    Ok((output, pullback))
}

/// Returns the primal output together with a pullback produced by transposing the staged pushforward.
#[allow(private_bounds)]
pub fn vjp<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Output, Input, E::LinearOperation>), TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType> + ZeroLike + OneLike,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>,
    F: FnOnce(Input::Traced) -> Output::Traced,
    E::LinearOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    Input::ParameterStructure: PartialEq,
    E::TracingOperation: DifferentiableOp<
            ArrayType,
            V,
            LinearTerm<ArrayType, V, E::LinearOperation>,
            E::TracingOperation,
            E::LinearOperation,
        >,
    E::LinearOperation: CoreLinearProgramOp<V> + LinearAddOperation<ArrayType, V>,
{
    try_vjp(engine, |input| Ok(function(input)), primals)
}

fn try_grad<E, F, Input, V>(engine: &E, function: F, primals: Input) -> Result<Input, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Value<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: PartialEq>,
    V: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(Input::Traced) -> Result<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>, TraceError>,
    E::LinearOperation: Clone + Op<ArrayType>,
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
    let (output, pullback): (V, LinearProgram<ArrayType, V, V, Input, E::LinearOperation>) =
        try_vjp(engine, function, primals)?;
    pullback.call(output.one_like())
}

/// Dispatch trait used by [`grad`] so it can operate both on concrete values and on already traced values.
#[doc(hidden)]
pub trait GradInvocationLeaf<E, Input>: Parameter + Sized
where
    E: Engine<Type = ArrayType>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
{
    /// Return type produced by [`grad`] for the corresponding input regime.
    type Return;

    /// Traced input type expected by the user-provided function.
    type FunctionInput;

    /// Traced scalar output type expected from the user-provided function.
    type FunctionOutput;

    /// Invokes [`grad`] for one concrete leaf regime.
    fn invoke<F>(engine: &E, function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput;
}

/// Concrete-value dispatch for [`grad`]: traces the user function with [`JitTracer`] to build a staged
/// reverse-mode gradient and evaluates it at the supplied primals.
impl<
    E,
    V: Value<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq> + TraceInput<V, E::TracingOperation, E::LinearOperation>,
> GradInvocationLeaf<E, Input> for V
where
    E: Engine<Type = ArrayType, Value = V>,
    V: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
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
    type Return = Input;
    type FunctionInput = Input::Traced;
    type FunctionOutput = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>;

    fn invoke<F>(engine: &E, function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        try_grad(engine, |input| Ok(function(input)), primals)
    }
}

/// Already-traced dispatch for [`grad`]: replays the user function symbolically inside an enclosing
/// [`JitTracer`] scope, linearizes the resulting [`Program`], transposes the pushforward into a pullback,
/// and stages the full backward pass so it becomes part of the outer compiled graph.
impl<
    E,
    V: Traceable<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + Op<ArrayType> + 'static,
> GradInvocationLeaf<E, Input> for JitTracer<ArrayType, V, O, L>
where
    E: Engine<Type = ArrayType, TracingOperation = O, LinearOperation = L>,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V: TraceOutput<V, O, L, Traced = JitTracer<ArrayType, V, O, L>>,
    V::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<V>: TraceInput<V, O, L, Traced = Input>,
    V::To<ArrayType>:
        crate::tracing_v2::TypeTracing<ArrayType, V, O, L, Staged = V, Traced = JitTracer<ArrayType, V, O, L>>,
    Input::To<ArrayType>: crate::tracing_v2::TypeTracing<ArrayType, V, O, L, Staged = Input::To<V>, Traced = Input>,
    O: InterpretableOp<
            ArrayType,
            Linearized<JitTracer<ArrayType, V, O, L>, LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>>,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>: CoreLinearProgramOp<JitTracer<ArrayType, V, O, L>>,
{
    type Return = Input;
    type FunctionInput = Input;
    type FunctionOutput = JitTracer<ArrayType, V, O, L>;

    fn invoke<F>(_engine: &E, function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let staged_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|primal| primal.tpe().into_owned()).collect::<Vec<_>>(),
        )?;
        let exemplar_engine = traced_primals.first().ok_or(TraceError::EmptyParameterizedValue)?.engine();
        let (_, traced_program) = crate::tracing_v2::jit::try_trace_program_from_types_for_operation::<
            _,
            Input::To<ArrayType>,
            V::To<ArrayType>,
            ArrayType,
            V,
            O,
            L,
        >(
            exemplar_engine,
            |staged_input| {
                let adapted_input = Input::from_parameters(
                    input_structure.clone(),
                    staged_input.into_parameters().collect::<Vec<_>>(),
                )?;
                Ok(function(adapted_input))
            },
            staged_input_types,
        )?;
        let traced_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
            flat_leaf_parameter_structure(traced_primals.len()),
            flat_leaf_parameter_structure(1),
        ))
        .simplify()?;
        let (outputs, pushforward) = try_linearize_traced_program::<V, O, L>(&traced_program, traced_primals)?;
        if outputs.len() != 1 {
            return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
        }
        let pullback = transpose_linear_program_with_output_examples::<JitTracer<ArrayType, V, O, L>, _, _, _>(
            &pushforward,
            outputs.as_slice(),
        )?;
        let traced_gradient = pullback.call(vec![outputs[0].one_like()])?;
        Ok(Input::from_parameters(input_structure, traced_gradient)?)
    }
}

/// Batched dispatch for [`grad`], enabling standalone `vmap(|x| grad(f, x), inputs)` -- computing
/// per-element gradients over a batch without requiring an outer [`jit`] wrapper.
///
/// Because the dispatch trait requires `F: FnOnce`, the user function can only be called once. This impl
/// traces the function once at the first lane's primals to obtain a [`Program`], then compiles a reusable
/// [`CompiledFunction`] via [`try_jit`] that embeds the full forward and backward passes symbolically.
/// The compiled gradient function is called independently for each lane.
impl<
    E,
    V: Traceable<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
> GradInvocationLeaf<E, Input> for Batch<V>
where
    E: Engine<Type = ArrayType, Value = V>,
    E::LinearOperation: Clone + Op<ArrayType>,
    V: Parameterized<
            V,
            ParameterStructure = Placeholder,
            To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>> = JitTracer<
                ArrayType,
                V,
                E::TracingOperation,
                E::LinearOperation,
            >,
        >,
    V: TraceInput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    V: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    Input::Family: ParameterizedFamily<V>,
    Input::To<V>: Clone
        + Parameterized<V, ParameterStructure: Clone + PartialEq, To<Batch<V>> = Input>
        + TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Vec<V>: TraceInput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = Vec<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
        >,
    Vec<V>: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = Vec<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
        >,
    E::TracingOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: InterpretableOp<
            ArrayType,
            Linearized<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
                LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
            >,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>:
        CoreLinearProgramOp<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
{
    type Return = Input;
    type FunctionInput = <Input::To<V> as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced;
    type FunctionOutput = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>;

    fn invoke<F>(engine: &E, function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        let lane_primals: Vec<Input::To<V>> = unstack(primals)?;
        if lane_primals.is_empty() {
            return Err(TraceError::EmptyBatch);
        }

        let lane0 = lane_primals[0].clone();
        let input_structure = lane0.parameter_structure();
        let parameter_count = input_structure.parameter_count();
        let lane0_flat: Vec<V> = lane0.into_parameters().collect();

        // Trace the user function once at lane 0 primals, consuming the FnOnce closure.
        let (_, traced_program): (V, Program<ArrayType, V, Input::To<V>, V, E::TracingOperation>) =
            try_trace_program(engine, |staged_input| Ok(function(staged_input)), lane_primals[0].clone())?;

        // Reshape the program to flat Vec<V> inputs and outputs for the JIT compilation step.
        let flat_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
            flat_leaf_parameter_structure(parameter_count),
            flat_leaf_parameter_structure(1),
        ))
        .simplify()?;

        // Compile the gradient into a reusable program by wrapping linearize + transpose + pullback
        // inside a JIT scope. This stages the full backward pass symbolically so it can be replayed
        // at arbitrary primal points.
        let (_, compiled_grad): (Vec<V>, CompiledFunction<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>) =
            try_jit(
                engine,
                |jit_primals: Vec<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>| {
                    let (outputs, pushforward) = try_linearize_traced_program::<
                        V,
                        E::TracingOperation,
                        E::LinearOperation,
                    >(&flat_program, jit_primals)?;
                    if outputs.len() != 1 {
                        return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
                    }
                    let pullback = transpose_linear_program_with_output_examples::<
                        JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
                        _,
                        _,
                        _,
                    >(&pushforward, outputs.as_slice())?;
                    pullback.call(vec![outputs[0].one_like()])
                },
                lane0_flat,
            )?;

        // Apply the compiled gradient per-lane and re-structure the flat results.
        let lane_grads = lane_primals
            .into_iter()
            .map(|lane| {
                let flat: Vec<V> = lane.into_parameters().collect();
                let flat_grad = compiled_grad.call(flat)?;
                Input::To::<V>::from_parameters(input_structure.clone(), flat_grad).map_err(TraceError::from)
            })
            .collect::<Result<Vec<_>, _>>()?;

        stack(lane_grads)
    }
}

/// Computes the reverse-mode gradient of a scalar-output function.
#[allow(private_bounds, private_interfaces)]
pub fn grad<E, F, Input, Leaf>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<<Leaf as GradInvocationLeaf<E, Input>>::Return, TraceError>
where
    E: Engine<Type = ArrayType>,
    Leaf: GradInvocationLeaf<E, Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as GradInvocationLeaf<E, Input>>::FunctionInput,
    ) -> <Leaf as GradInvocationLeaf<E, Input>>::FunctionOutput,
{
    Leaf::invoke(engine, function, primals)
}

/// Dispatch trait used by [`value_and_grad`] so it can operate both on concrete values and on already traced values.
#[doc(hidden)]
pub trait ValueAndGradInvocationLeaf<E, Input>: Parameter + Sized
where
    E: Engine<Type = ArrayType>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
{
    /// Return type produced by [`value_and_grad`] for the corresponding input regime.
    type Return;

    /// Traced input type expected by the user-provided function.
    type FunctionInput;

    /// Traced scalar output type expected from the user-provided function.
    type FunctionOutput;

    /// Invokes [`value_and_grad`] for one concrete leaf regime.
    fn invoke<F>(engine: &E, function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput;
}

/// Concrete-value dispatch for [`value_and_grad`]: evaluates the user function via [`vjp`] and
/// pulls back a unit seed to obtain both the primal scalar output and its gradient.
impl<
    E,
    V: Value<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq> + TraceInput<V, E::TracingOperation, E::LinearOperation>,
> ValueAndGradInvocationLeaf<E, Input> for V
where
    E: Engine<Type = ArrayType, Value = V>,
    V: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
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
    type Return = (V, Input);
    type FunctionInput = Input::Traced;
    type FunctionOutput = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>;

    fn invoke<F>(engine: &E, function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        let (output, pullback): (V, LinearProgram<ArrayType, V, V, Input, E::LinearOperation>) =
            vjp(engine, function, primals)?;
        let gradient = pullback.call(output.one_like())?;
        Ok((output, gradient))
    }
}

/// Already-traced dispatch for [`value_and_grad`]: replays the user function symbolically inside an
/// enclosing [`JitTracer`] scope, linearizes, transposes, and stages both the forward output and the
/// backward gradient so they become part of the outer compiled graph.
impl<
    E,
    V: Traceable<ArrayType> + ZeroLike + OneLike + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
    O: Clone + Op<ArrayType> + 'static,
    L: Clone + Op<ArrayType> + 'static,
> ValueAndGradInvocationLeaf<E, Input> for JitTracer<ArrayType, V, O, L>
where
    E: Engine<Type = ArrayType, TracingOperation = O, LinearOperation = L>,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V: TraceOutput<V, O, L, Traced = JitTracer<ArrayType, V, O, L>>,
    V::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<JitTracer<ArrayType, V, O, L>>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<ArrayType>,
    Input::To<V>: TraceInput<V, O, L, Traced = Input>,
    V::To<ArrayType>:
        crate::tracing_v2::TypeTracing<ArrayType, V, O, L, Staged = V, Traced = JitTracer<ArrayType, V, O, L>>,
    Input::To<ArrayType>: crate::tracing_v2::TypeTracing<ArrayType, V, O, L, Staged = Input::To<V>, Traced = Input>,
    O: InterpretableOp<
            ArrayType,
            Linearized<JitTracer<ArrayType, V, O, L>, LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>>,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, O, L>>: CoreLinearProgramOp<JitTracer<ArrayType, V, O, L>>,
{
    type Return = (JitTracer<ArrayType, V, O, L>, Input);
    type FunctionInput = Input;
    type FunctionOutput = JitTracer<ArrayType, V, O, L>;

    fn invoke<F>(_engine: &E, function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let staged_input_types = Input::To::<ArrayType>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|primal| primal.tpe().into_owned()).collect::<Vec<_>>(),
        )?;
        let exemplar_engine = traced_primals.first().ok_or(TraceError::EmptyParameterizedValue)?.engine();
        let (_, traced_program) = crate::tracing_v2::jit::try_trace_program_from_types_for_operation::<
            _,
            Input::To<ArrayType>,
            V::To<ArrayType>,
            ArrayType,
            V,
            O,
            L,
        >(
            exemplar_engine,
            |staged_input| {
                let adapted_input = Input::from_parameters(
                    input_structure.clone(),
                    staged_input.into_parameters().collect::<Vec<_>>(),
                )?;
                Ok(function(adapted_input))
            },
            staged_input_types,
        )?;
        let traced_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
            flat_leaf_parameter_structure(traced_primals.len()),
            flat_leaf_parameter_structure(1),
        ))
        .simplify()?;
        let (outputs, pushforward) = try_linearize_traced_program::<V, O, L>(&traced_program, traced_primals)?;
        if outputs.len() != 1 {
            return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
        }
        let traced_output = outputs[0].clone();
        let pullback = transpose_linear_program_with_output_examples::<JitTracer<ArrayType, V, O, L>, _, _, _>(
            &pushforward,
            outputs.as_slice(),
        )?;
        let traced_gradient = pullback.call(vec![traced_output.one_like()])?;
        Ok((traced_output, Input::from_parameters(input_structure, traced_gradient)?))
    }
}

/// Batched dispatch for [`value_and_grad`], enabling standalone
/// `vmap(|x| value_and_grad(f, x), inputs)` -- computing per-element function values and gradients
/// over a batch without requiring an outer [`jit`] wrapper.
///
/// Uses the same trace-once strategy as [`GradInvocationLeaf`] for [`Batch`]: the user function is
/// traced once to a [`Program`], and a [`CompiledFunction`] that produces `(V, Input::To<V>)` per lane
/// is compiled via [`try_jit`]. Values and gradients are collected per lane and stacked separately.
impl<
    E,
    V: Traceable<ArrayType> + ZeroLike + OneLike,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
> ValueAndGradInvocationLeaf<E, Input> for Batch<V>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Parameterized<
            V,
            ParameterStructure = Placeholder,
            To<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>> = JitTracer<
                ArrayType,
                V,
                E::TracingOperation,
                E::LinearOperation,
            >,
        >,
    V: TraceInput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    V: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    Input::Family: ParameterizedFamily<V>,
    Input::To<V>: Clone
        + Parameterized<V, ParameterStructure: Clone + PartialEq, To<Batch<V>> = Input>
        + TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Vec<V>: TraceInput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = Vec<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
        >,
    Vec<V>: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = Vec<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
        >,
    E::TracingOperation: Clone + Op<ArrayType>,
    E::TracingOperation: InterpretableOp<ArrayType, V>,
    E::TracingOperation: InterpretableOp<
            ArrayType,
            Linearized<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
                LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
            >,
        >,
    LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>:
        CoreLinearProgramOp<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
{
    type Return = (Batch<V>, Input);
    type FunctionInput = <Input::To<V> as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced;
    type FunctionOutput = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>;

    fn invoke<F>(engine: &E, function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> Self::FunctionOutput,
    {
        let lane_primals: Vec<Input::To<V>> = unstack(primals)?;
        if lane_primals.is_empty() {
            return Err(TraceError::EmptyBatch);
        }

        let lane0 = lane_primals[0].clone();
        let input_structure = lane0.parameter_structure();
        let parameter_count = input_structure.parameter_count();
        let lane0_flat: Vec<V> = lane0.into_parameters().collect();

        // Trace the user function once at lane 0 primals, consuming the FnOnce closure.
        let (_, traced_program): (V, Program<ArrayType, V, Input::To<V>, V, E::TracingOperation>) =
            try_trace_program(engine, |staged_input| Ok(function(staged_input)), lane_primals[0].clone())?;

        // Reshape the program to flat Vec<V> inputs and outputs for the JIT compilation step.
        let flat_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
            flat_leaf_parameter_structure(parameter_count),
            flat_leaf_parameter_structure(1),
        ))
        .simplify()?;

        // Compile both the forward evaluation and gradient into a reusable program.
        let (_, compiled_vg): (Vec<V>, CompiledFunction<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>) = try_jit(
            engine,
            |jit_primals: Vec<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>| {
                let (outputs, pushforward) = try_linearize_traced_program::<V, E::TracingOperation, E::LinearOperation>(
                    &flat_program,
                    jit_primals,
                )?;
                if outputs.len() != 1 {
                    return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
                }
                let pullback = transpose_linear_program_with_output_examples::<
                    JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
                    _,
                    _,
                    _,
                >(&pushforward, outputs.as_slice())?;
                let gradient = pullback.call(vec![outputs[0].one_like()])?;
                let mut result = Vec::with_capacity(1 + gradient.len());
                result.push(outputs[0].clone());
                result.extend(gradient);
                Ok(result)
            },
            lane0_flat,
        )?;

        // Apply per-lane and split into (value, gradient).
        let mut lane_values = Vec::with_capacity(lane_primals.len());
        let mut lane_grads = Vec::with_capacity(lane_primals.len());
        for lane in lane_primals {
            let flat: Vec<V> = lane.into_parameters().collect();
            let flat_result = compiled_vg.call(flat)?;
            let (value, grad_flat) = flat_result.split_first().ok_or(TraceError::EmptyParameterizedValue)?;
            lane_values.push(value.clone());
            lane_grads.push(
                Input::To::<V>::from_parameters(input_structure.clone(), grad_flat.to_vec())
                    .map_err(TraceError::from)?,
            );
        }

        let batched_values = Batch::new(lane_values);
        let batched_grads = stack(lane_grads)?;
        Ok((batched_values, batched_grads))
    }
}

/// Computes both the primal scalar output and its reverse-mode gradient.
#[allow(private_bounds, private_interfaces)]
pub fn value_and_grad<E, F, Input, Leaf>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<<Leaf as ValueAndGradInvocationLeaf<E, Input>>::Return, TraceError>
where
    E: Engine<Type = ArrayType>,
    Leaf: ValueAndGradInvocationLeaf<E, Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionInput,
    ) -> <Leaf as ValueAndGradInvocationLeaf<E, Input>>::FunctionOutput,
{
    Leaf::invoke(engine, function, primals)
}

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

fn try_jacfwd<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: CoordinateValue,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
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
    let (output, pushforward) = try_jvp_program::<E, F, Input, Output, V>(engine, function, primals)?;
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
pub fn jacfwd<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: CoordinateValue,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(Input::Traced) -> Output::Traced,
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
    try_jacfwd::<E, _, Input, Output, V>(engine, |input| Ok(function(input)), primals)
}

fn try_jacrev<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: CoordinateValue,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(Input::Traced) -> Result<Output::Traced, TraceError>,
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
    let (output, pullback) = try_vjp::<E, F, Input, Output, V>(engine, function, primals)?;
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
pub fn jacrev<E, F, Input, Output, V>(
    engine: &E,
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: CoordinateValue,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output: TraceOutput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(Input::Traced) -> Output::Traced,
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
    try_jacrev::<E, _, Input, Output, V>(engine, |input| Ok(function(input)), primals)
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
    E: Engine<Type = ArrayType, Value = V>,
    V: CoordinateValue,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>
        + TraceOutput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced,
    ) -> <Input as TraceOutput<V, E::TracingOperation, E::LinearOperation>>::Traced,
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

// ---------------------------------------------------------------------------
// Compiled transforms — reusable programs not specialized to a primal point
// ---------------------------------------------------------------------------

/// Compiles a reverse-mode gradient function into a reusable staged program.
///
/// Unlike [`grad`], which returns concrete gradient values at a single primal point, this function returns a
/// [`CompiledFunction`] that takes primal inputs and produces gradient outputs symbolically. The compiled
/// program embeds both the forward residual computation and the backward pass, so it can be replayed at
/// arbitrary primal points without re-tracing.
///
/// This is analogous to JAX's `jit(grad(f))`.
#[allow(private_bounds)]
pub fn compile_grad<E, F, Input, V>(
    _engine: &E,
    function: F,
    example_primals: Input,
) -> Result<CompiledFunction<ArrayType, V, Input, Input, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Value<ArrayType> + ZeroLike + OneLike,
    E::TracingOperation: InterpretableOp<ArrayType, V>
        + InterpretableOp<ArrayType, LinearizedTracedValue<V, E::TracingOperation, E::LinearOperation>>
        + Op<ArrayType>,
    LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>:
        CoreLinearProgramOp<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    V: TraceInput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    V: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>
        + TraceOutput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::To<V>: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Input::Family: ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: crate::tracing_v2::TypeTracing<
            ArrayType,
            V,
            E::TracingOperation,
            E::LinearOperation,
            Staged = Input::To<V>,
            Traced = <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced,
        >,
    V::To<ArrayType>: crate::tracing_v2::TypeTracing<
            ArrayType,
            V,
            E::TracingOperation,
            E::LinearOperation,
            Staged = V,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    F: Fn(
        <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced,
    ) -> JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
{
    let input_structure = example_primals.parameter_structure();
    let (_, compiled) = try_jit_for_operation::<_, Input, Input, V, E::TracingOperation, E::LinearOperation>(
        _engine,
        |primals: <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced| {
            let traced_primals = primals.into_parameters().collect::<Vec<_>>();
            let staged_input_types = Input::To::<ArrayType>::from_parameters(
                input_structure.clone(),
                traced_primals.iter().map(|primal| primal.tpe().into_owned()).collect::<Vec<_>>(),
            )?;
            let exemplar_engine = traced_primals.first().ok_or(TraceError::EmptyParameterizedValue)?.engine();
            let (_, traced_program) = crate::tracing_v2::jit::try_trace_program_from_types_for_operation::<
                _,
                Input::To<ArrayType>,
                V::To<ArrayType>,
                ArrayType,
                V,
                E::TracingOperation,
                E::LinearOperation,
            >(
                exemplar_engine,
                |staged_input: <Input::To<ArrayType> as crate::tracing_v2::TypeTracing<
                    ArrayType,
                    V,
                    E::TracingOperation,
                    E::LinearOperation,
                >>::Traced| {
                    let adapted_input =
                        <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced::from_parameters(
                            input_structure.clone(),
                            staged_input.into_parameters().collect::<Vec<_>>(),
                        )?;
                    Ok(function(adapted_input))
                },
                staged_input_types,
            )?;
            let traced_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
                flat_leaf_parameter_structure(traced_primals.len()),
                flat_leaf_parameter_structure(1),
            ))
            .simplify()?;
            let (outputs, pushforward) = try_linearize_traced_program(&traced_program, traced_primals)?;
            if outputs.len() != 1 {
                return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
            }
            let pullback = transpose_linear_program_with_output_examples::<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
                _,
                _,
                _,
            >(&pushforward, outputs.as_slice())?;
            let traced_gradient = pullback.call(vec![outputs[0].one_like()])?;
            Ok(<Input as TraceOutput<V, E::TracingOperation, E::LinearOperation>>::Traced::from_parameters(
                input_structure.clone(),
                traced_gradient,
            )?)
        },
        example_primals,
    )?;
    Ok(compiled)
}

/// Policy controlling how forward-pass intermediates are handled during reverse-mode differentiation.
///
/// This trades off memory usage against recomputation cost. [`SaveAll`](RematerializationPolicy::SaveAll) is the
/// default: all intermediates are saved, giving fast backward passes at the cost of high memory.
/// [`RecomputeAll`](RematerializationPolicy::RecomputeAll) saves nothing, recomputing everything from inputs.
/// [`Checkpoint`](RematerializationPolicy::Checkpoint) is the classic middle ground, saving intermediates at
/// regular intervals and recomputing within each segment.
#[derive(Clone, Debug)]
pub enum RematerializationPolicy {
    /// Save all forward-pass intermediates (maximum memory, no recomputation).
    SaveAll,

    /// Recompute all forward-pass intermediates from inputs (minimum memory, maximum recomputation).
    RecomputeAll,

    /// Save intermediates every `segment_size` equations, recomputing within each segment.
    ///
    /// With a program of N equations, setting `segment_size` to approximately the square root of N gives O(sqrt(N))
    /// memory usage. A `segment_size` of zero or one degenerates to [`SaveAll`](RematerializationPolicy::SaveAll)
    /// since each segment contains at most one equation.
    Checkpoint {
        /// Number of equations per rematerialization segment.
        segment_size: usize,
    },
}

/// Compiles a reverse-mode gradient function with an explicit rematerialization policy.
///
/// This generalizes [`compile_grad`] by letting the caller control how forward-pass intermediates are handled
/// during the backward pass:
///
///   - [`RematerializationPolicy::SaveAll`]: identical to [`compile_grad`] — no rematerialization boundaries are
///     inserted, so the XLA compiler decides which intermediates to save.
///   - [`RematerializationPolicy::RecomputeAll`]: the entire forward body is wrapped in a single
///     [`rematerialize`] boundary, forcing the backward pass to recompute all intermediates from inputs.
///   - [`RematerializationPolicy::Checkpoint`]: the forward body is partitioned into segments of at most
///     `segment_size` equations, each wrapped in its own [`rematerialize`] boundary. Intermediates at segment
///     boundaries are saved while within-segment intermediates are recomputed.
#[allow(private_bounds)]
pub fn compile_grad_with_policy<E, F, Input, V>(
    engine: &E,
    function: F,
    example_primals: Input,
    policy: RematerializationPolicy,
) -> Result<CompiledFunction<ArrayType, V, Input, Input, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Value<ArrayType> + ZeroLike + OneLike,
    E::TracingOperation: InterpretableOp<ArrayType, V>
        + InterpretableOp<ArrayType, LinearizedTracedValue<V, E::TracingOperation, E::LinearOperation>>
        + RematerializeTracingOperation<ArrayType, V, E::LinearOperation>
        + Op<ArrayType>,
    LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>:
        CoreLinearProgramOp<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    V: TraceInput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    V: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>
        + TraceOutput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::To<V>: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Input::Family: ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: crate::tracing_v2::TypeTracing<
            ArrayType,
            V,
            E::TracingOperation,
            E::LinearOperation,
            Staged = Input::To<V>,
            Traced = <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced,
        >,
    V::To<ArrayType>: crate::tracing_v2::TypeTracing<
            ArrayType,
            V,
            E::TracingOperation,
            E::LinearOperation,
            Staged = V,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    F: Fn(
        <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced,
    ) -> JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
{
    match policy {
        RematerializationPolicy::SaveAll => compile_grad(engine, &function, example_primals),
        RematerializationPolicy::RecomputeAll => compile_grad_segmented(engine, &function, example_primals, None),
        RematerializationPolicy::Checkpoint { segment_size } => {
            if segment_size <= 1 {
                return compile_grad(engine, &function, example_primals);
            }
            compile_grad_segmented(engine, &function, example_primals, Some(segment_size))
        }
    }
}

/// Compiles a gradient function with rematerialization boundaries inserted via program segmentation.
///
/// When `segment_size` is `None`, the entire program is wrapped in a single [`RematerializeOp`]
/// (equivalent to [`RematerializationPolicy::RecomputeAll`]). When `Some(s)`, the program is
/// partitioned into segments of at most `s` equations, each wrapped in its own [`RematerializeOp`].
///
/// Internally, this replicates the flow of `grad` for [`JitTracer`]-level inputs — trace, linearize,
/// transpose, stage pullback — but inserts a segmentation step between tracing and linearization so
/// that the differentiation transform sees and respects the rematerialization boundaries.
fn compile_grad_segmented<E, F, Input, V>(
    engine: &E,
    function: &F,
    example_primals: Input,
    segment_size: Option<usize>,
) -> Result<CompiledFunction<ArrayType, V, Input, Input, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Value<ArrayType> + ZeroLike + OneLike,
    E::TracingOperation: InterpretableOp<ArrayType, V>
        + InterpretableOp<ArrayType, LinearizedTracedValue<V, E::TracingOperation, E::LinearOperation>>
        + RematerializeTracingOperation<ArrayType, V, E::LinearOperation>
        + Op<ArrayType>,
    LinearProgramOpRef<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>:
        CoreLinearProgramOp<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    V: Parameterized<V, ParameterStructure = Placeholder>,
    V::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>>,
    Vec<V>: Parameterized<V, ParameterStructure = Vec<Placeholder>>,
    V: TraceInput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    V: TraceOutput<
            V,
            E::TracingOperation,
            E::LinearOperation,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    Input: TraceInput<V, E::TracingOperation, E::LinearOperation>
        + TraceOutput<V, E::TracingOperation, E::LinearOperation>
        + Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::To<V>: TraceInput<V, E::TracingOperation, E::LinearOperation>,
    Input::Family: ParameterizedFamily<ArrayType>,
    Input::To<ArrayType>: crate::tracing_v2::TypeTracing<
            ArrayType,
            V,
            E::TracingOperation,
            E::LinearOperation,
            Staged = Input::To<V>,
            Traced = <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced,
        >,
    V::To<ArrayType>: crate::tracing_v2::TypeTracing<
            ArrayType,
            V,
            E::TracingOperation,
            E::LinearOperation,
            Staged = V,
            Traced = JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
        >,
    F: Fn(
        <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced,
    ) -> JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
{
    let input_structure = example_primals.parameter_structure();
    let (_, compiled) = try_jit_for_operation::<_, Input, Input, V, E::TracingOperation, E::LinearOperation>(
        engine,
        |primals: <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced| {
            let traced_primals = primals.into_parameters().collect::<Vec<_>>();

            // Step 1: Trace the function at the base V level to get a program.
            let staged_input_types = Input::To::<ArrayType>::from_parameters(
                input_structure.clone(),
                traced_primals.iter().map(|primal| primal.tpe().into_owned()).collect::<Vec<_>>(),
            )?;
            let exemplar_engine = traced_primals.first().ok_or(TraceError::EmptyParameterizedValue)?.engine();
            let (_, traced_program) = crate::tracing_v2::jit::try_trace_program_from_types_for_operation::<
                _,
                Input::To<ArrayType>,
                V::To<ArrayType>,
                ArrayType,
                V,
                E::TracingOperation,
                E::LinearOperation,
            >(
                exemplar_engine,
                |staged_input: <Input::To<ArrayType> as crate::tracing_v2::TypeTracing<
                    ArrayType,
                    V,
                    E::TracingOperation,
                    E::LinearOperation,
                >>::Traced| {
                    let adapted_input =
                        <Input as TraceInput<V, E::TracingOperation, E::LinearOperation>>::Traced::from_parameters(
                            input_structure.clone(),
                            staged_input.into_parameters().collect::<Vec<_>>(),
                        )?;
                    Ok(function(adapted_input))
                },
                staged_input_types,
            )?;
            let traced_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
                flat_leaf_parameter_structure(traced_primals.len()),
                flat_leaf_parameter_structure(1),
            ))
            .simplify()?;

            // Step 2: Segment the traced program to insert rematerialization boundaries.
            let segmented_program = match segment_size {
                None => wrap_program_in_rematerialize(engine, &traced_program)?,
                Some(size) => segment_program(engine, &traced_program, size)?,
            };

            // Step 3: Linearize and transpose the segmented program to produce the pullback.
            // `try_linearize_traced_program` replays the graph at the JitTracer level (staging
            // both forward and backward equations in the outer JIT builder) and returns the
            // primal outputs alongside the linear pushforward map.
            let (outputs, pushforward) = try_linearize_traced_program(&segmented_program, traced_primals)?;
            if outputs.len() != 1 {
                return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
            }
            let pullback = transpose_linear_program_with_output_examples::<
                JitTracer<ArrayType, V, E::TracingOperation, E::LinearOperation>,
                _,
                _,
                _,
            >(&pushforward, outputs.as_slice())?;
            let traced_gradient = pullback.call(vec![outputs[0].one_like()])?;
            Ok(<Input as TraceOutput<V, E::TracingOperation, E::LinearOperation>>::Traced::from_parameters(
                input_structure.clone(),
                traced_gradient,
            )?)
        },
        example_primals,
    )?;
    Ok(compiled)
}

/// Partitions a program's equations into segments of at most `segment_size`, wrapping each segment in a
/// [`RematerializeOp`].
///
/// Given a program with N equations and a segment size S, this produces a new program with at most
/// `ceil(N / S)` equations. Each equation is a [`RematerializeOp`] whose body sub-program contains the
/// original equations from that segment. Atoms crossing segment boundaries become inputs/outputs of the
/// respective sub-programs.
///
/// The segmented program is semantically equivalent to the original: calling it on the same inputs produces
/// the same outputs. The difference is visible only during differentiation, where each [`RematerializeOp`]
/// boundary forces recomputation of within-segment intermediates rather than saving them.
fn segment_program<E, V>(
    engine: &E,
    program: &Program<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>,
    segment_size: usize,
) -> Result<Program<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    E::TracingOperation:
        InterpretableOp<ArrayType, V> + RematerializeTracingOperation<ArrayType, V, E::LinearOperation> + Op<ArrayType>,
{
    let graph = program.graph();
    let representative_values = graph.representative_atom_values(engine)?;
    let representative_inputs = graph.representative_input_values(engine)?;
    let equations = graph.equations();

    // If the program has fewer equations than a single segment, no segmentation is needed — wrap the
    // whole thing in a single RematerializeOp.
    if equations.len() <= segment_size {
        return wrap_program_in_rematerialize(engine, program);
    }

    // Divide equations into segments.
    let segments: Vec<&[Equation<E::TracingOperation>]> = equations.chunks(segment_size).collect();

    // Build a mapping from atom ID to which equation produces it (if any).
    let mut atom_producer: Vec<Option<usize>> = vec![None; graph.atom_count()];
    for (equation_index, equation) in equations.iter().enumerate() {
        for &output_atom in &equation.outputs {
            atom_producer[output_atom] = Some(equation_index);
        }
    }

    // Build a set tracking which atoms are consumed after a given equation index.
    // For each atom, track all equation indices that consume it.
    let mut atom_consumers: Vec<Vec<usize>> = vec![Vec::new(); graph.atom_count()];
    for (equation_index, equation) in equations.iter().enumerate() {
        for &input_atom in &equation.inputs {
            atom_consumers[input_atom].push(equation_index);
        }
    }
    // Also mark program outputs as "consumed" at equation_count (sentinel for "after all equations").
    let sentinel = equations.len();
    for &output_atom in graph.outputs() {
        atom_consumers[output_atom].push(sentinel);
    }

    // Build the outer program.
    let input_atoms = graph.input_atoms();
    let mut outer_builder: ProgramBuilder<V, E::TracingOperation> = ProgramBuilder::new();

    // Map from original atom IDs to outer-program atom IDs.
    let mut atom_mapping: Vec<Option<AtomId>> = vec![None; graph.atom_count()];

    // Register program inputs in the outer builder.
    for (&input_atom, representative_input) in input_atoms.iter().zip(representative_inputs.iter()) {
        let outer_atom = outer_builder.add_input(representative_input);
        atom_mapping[input_atom] = Some(outer_atom);
    }

    // Register constants that are used by equations (they might be referenced across segments).
    for (atom_id, atom) in graph.atoms_iter() {
        if let Atom::Constant { value } = atom {
            let outer_atom = outer_builder.add_constant(value.clone());
            atom_mapping[atom_id] = Some(outer_atom);
        }
    }

    // Process each segment.
    let mut equation_offset = 0;
    for segment in &segments {
        let segment_start = equation_offset;
        let segment_end = equation_offset + segment.len();

        // Identify boundary inputs: atoms consumed by this segment that are produced outside it
        // (by previous segments or program inputs/constants).
        let mut boundary_input_atoms: Vec<AtomId> = Vec::new();
        let mut boundary_input_set = std::collections::HashSet::new();
        for equation in *segment {
            for &input_atom in &equation.inputs {
                // If this atom is produced by an equation outside this segment (or is an input/constant).
                let produced_in_segment = atom_producer[input_atom]
                    .map_or(false, |producer_idx| producer_idx >= segment_start && producer_idx < segment_end);
                if !produced_in_segment && boundary_input_set.insert(input_atom) {
                    boundary_input_atoms.push(input_atom);
                }
            }
        }

        // Identify boundary outputs: atoms produced by this segment that are consumed outside it
        // (by later segments or as program outputs).
        let mut boundary_output_atoms: Vec<AtomId> = Vec::new();
        let mut boundary_output_set = std::collections::HashSet::new();
        for equation in *segment {
            for &output_atom in &equation.outputs {
                let consumed_outside = atom_consumers[output_atom]
                    .iter()
                    .any(|&consumer_idx| consumer_idx < segment_start || consumer_idx >= segment_end);
                if consumed_outside && boundary_output_set.insert(output_atom) {
                    boundary_output_atoms.push(output_atom);
                }
            }
        }

        // Build the sub-program for this segment.
        let sub_program = build_segment_sub_program(
            graph,
            representative_values.as_slice(),
            *segment,
            &boundary_input_atoms,
            &boundary_output_atoms,
        )?;

        // Build the RematerializeOp.
        let input_types: Vec<_> = boundary_input_atoms
            .iter()
            .map(|&atom_id| {
                graph
                    .atom(atom_id)
                    .ok_or(TraceError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.tpe().into_owned())
            })
            .collect::<Result<_, _>>()?;
        let output_types: Vec<_> = boundary_output_atoms
            .iter()
            .map(|&atom_id| {
                graph
                    .atom(atom_id)
                    .ok_or(TraceError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.tpe().into_owned())
            })
            .collect::<Result<_, _>>()?;

        let body = FlatTracedRematerialize::from_parts(
            input_types.clone(),
            output_types.clone(),
            CompiledFunction::from_program(sub_program),
        );
        let remat_op = RematerializeOp::new(body);

        // Add the RematerializeOp equation to the outer builder.
        let outer_inputs: Vec<AtomId> = boundary_input_atoms
            .iter()
            .map(|&orig_atom| atom_mapping[orig_atom].ok_or(TraceError::UnboundAtomId { id: orig_atom }))
            .collect::<Result<_, _>>()?;
        let outer_outputs = outer_builder.add_equation_prevalidated(
            E::TracingOperation::rematerialize_op(remat_op),
            outer_inputs,
            output_types,
        );

        // Map the boundary output atoms to their outer-program counterparts.
        for (orig_atom, outer_atom) in boundary_output_atoms.iter().zip(outer_outputs.iter()) {
            atom_mapping[*orig_atom] = Some(*outer_atom);
        }

        equation_offset = segment_end;
    }

    // Wire up the program outputs.
    let outer_outputs: Vec<AtomId> = graph
        .outputs()
        .iter()
        .map(|&orig_atom| atom_mapping[orig_atom].ok_or(TraceError::UnboundAtomId { id: orig_atom }))
        .collect::<Result<_, _>>()?;

    let outer_graph = outer_builder.build::<Vec<V>, Vec<V>>(
        outer_outputs,
        flat_leaf_parameter_structure(input_atoms.len()),
        flat_leaf_parameter_structure(graph.outputs().len()),
    );
    Ok(Program::from_graph(outer_graph))
}

/// Wraps an entire program in a single [`RematerializeOp`] boundary.
fn wrap_program_in_rematerialize<E, V>(
    engine: &E,
    program: &Program<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>,
) -> Result<Program<ArrayType, V, Vec<V>, Vec<V>, E::TracingOperation>, TraceError>
where
    E: Engine<Type = ArrayType, Value = V>,
    V: Traceable<ArrayType>,
    E::TracingOperation: RematerializeTracingOperation<ArrayType, V, E::LinearOperation>,
{
    let graph = program.graph();
    let representative_inputs = graph.representative_input_values(engine)?;
    let input_types: Vec<_> = graph
        .input_atoms()
        .iter()
        .map(|&atom_id| {
            graph
                .atom(atom_id)
                .ok_or(TraceError::UnboundAtomId { id: atom_id })
                .map(|atom| atom.tpe().into_owned())
        })
        .collect::<Result<_, _>>()?;
    let output_types: Vec<_> = graph
        .outputs()
        .iter()
        .map(|&atom_id| {
            graph
                .atom(atom_id)
                .ok_or(TraceError::UnboundAtomId { id: atom_id })
                .map(|atom| atom.tpe().into_owned())
        })
        .collect::<Result<_, _>>()?;

    let body = FlatTracedRematerialize::from_parts(
        input_types.clone(),
        output_types.clone(),
        CompiledFunction::from_program(program.clone()),
    );
    let remat_op = RematerializeOp::new(body);

    let mut outer_builder: ProgramBuilder<V, E::TracingOperation> = ProgramBuilder::new();
    let outer_inputs: Vec<AtomId> = representative_inputs
        .iter()
        .map(|representative_input| outer_builder.add_input(representative_input))
        .collect();

    let outer_outputs = outer_builder.add_equation_prevalidated(
        E::TracingOperation::rematerialize_op(remat_op),
        outer_inputs.clone(),
        output_types,
    );

    let outer_graph = outer_builder.build::<Vec<V>, Vec<V>>(
        outer_outputs,
        flat_leaf_parameter_structure(outer_inputs.len()),
        flat_leaf_parameter_structure(graph.outputs().len()),
    );
    Ok(Program::from_graph(outer_graph))
}

/// Builds a sub-program for a single segment of equations.
///
/// The sub-program takes the boundary input atoms as its inputs and produces the boundary output atoms as its
/// outputs. Internal atoms (produced and consumed entirely within the segment) are handled as internal constants
/// and equations within the sub-program.
fn build_segment_sub_program<V: Traceable<ArrayType>, O: Clone>(
    graph: &Graph<O, ArrayType, V, Vec<V>, Vec<V>>,
    representative_values: &[V],
    segment_equations: &[Equation<O>],
    boundary_input_atoms: &[AtomId],
    boundary_output_atoms: &[AtomId],
) -> Result<Program<ArrayType, V, Vec<V>, Vec<V>, O>, TraceError> {
    let mut sub_builder: ProgramBuilder<V, O> = ProgramBuilder::new();

    // Map from original atom IDs to sub-program atom IDs.
    let mut sub_atom_mapping: std::collections::HashMap<AtomId, AtomId> = std::collections::HashMap::new();

    // Register boundary inputs as sub-program inputs.
    for &input_atom in boundary_input_atoms {
        let sub_atom = sub_builder.add_input(&representative_values[input_atom]);
        sub_atom_mapping.insert(input_atom, sub_atom);
    }

    // Register constants used by equations in this segment.
    for equation in segment_equations {
        for &input_atom in &equation.inputs {
            if sub_atom_mapping.contains_key(&input_atom) {
                continue;
            }
            let atom = graph.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
            if let Atom::Constant { value } = atom {
                let sub_atom = sub_builder.add_constant(value.clone());
                sub_atom_mapping.insert(input_atom, sub_atom);
            }
        }
    }

    // Add equations to the sub-program.
    for equation in segment_equations {
        let sub_inputs: Vec<AtomId> = equation
            .inputs
            .iter()
            .map(|&orig_atom| {
                sub_atom_mapping.get(&orig_atom).copied().ok_or(TraceError::UnboundAtomId { id: orig_atom })
            })
            .collect::<Result<_, _>>()?;

        let output_abstracts: Vec<_> = equation
            .outputs
            .iter()
            .map(|&atom_id| {
                graph
                    .atom(atom_id)
                    .ok_or(TraceError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.tpe().into_owned())
            })
            .collect::<Result<_, _>>()?;
        let sub_outputs = sub_builder.add_equation_prevalidated(equation.op.clone(), sub_inputs, output_abstracts);

        for (orig_atom, sub_atom) in equation.outputs.iter().zip(sub_outputs.iter()) {
            sub_atom_mapping.insert(*orig_atom, *sub_atom);
        }
    }

    // Wire up boundary outputs.
    let sub_outputs: Vec<AtomId> = boundary_output_atoms
        .iter()
        .map(|&orig_atom| sub_atom_mapping.get(&orig_atom).copied().ok_or(TraceError::UnboundAtomId { id: orig_atom }))
        .collect::<Result<_, _>>()?;

    let sub_graph = sub_builder.build::<Vec<V>, Vec<V>>(
        sub_outputs,
        flat_leaf_parameter_structure(boundary_input_atoms.len()),
        flat_leaf_parameter_structure(boundary_output_atoms.len()),
    );
    Ok(Program::from_graph(sub_graph))
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, Mul, Neg};
    use std::{
        fmt::{Debug, Display},
        sync::Arc,
    };

    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{
            CustomPrimitive, DifferentiableOp, GraphBuilder, InterpretableOp, LinearOperation, LinearPrimitiveOp, Op,
            PrimitiveOp, ProgramOpRef, Sin, engine::ArrayScalarEngine, test_support,
        },
        types::{ArrayType, DataType},
    };

    use super::*;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta <= 1e-9, "expected {left} ~= {right}; absolute error {delta} exceeded tolerance");
    }

    fn quadratic_plus_sin<T>(x: T) -> T
    where
        T: Clone + Sin + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        x.clone() * x.clone() + x.sin()
    }

    fn bilinear_sin<T>(inputs: (T, T)) -> T
    where
        T: Clone + Sin + Add<Output = T> + Mul<Output = T> + Neg<Output = T>,
    {
        inputs.0.clone() * inputs.1 + inputs.0.sin()
    }

    #[derive(Clone, Default)]
    struct PanicReplayOp;

    impl Debug for PanicReplayOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "PanicReplay")
        }
    }

    impl Display for PanicReplayOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "panic_replay")
        }
    }

    impl Op for PanicReplayOp {
        fn name(&self) -> &'static str {
            "panic_replay"
        }

        fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0].clone()])
        }
    }

    impl InterpretableOp<ArrayType, f64> for PanicReplayOp {
        fn interpret(&self, _inputs: &[f64]) -> Result<Vec<f64>, TraceError> {
            panic!("panic_replay interpret should not run during this transform")
        }
    }

    impl LinearOperation<ArrayType, f64> for PanicReplayOp {
        fn transpose(
            &self,
            output_cotangents: &[LinearTerm<ArrayType, f64>],
        ) -> Result<Vec<Option<LinearTerm<ArrayType, f64>>>, TraceError> {
            if output_cotangents.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
            }
            Ok(vec![Some(output_cotangents[0].clone())])
        }
    }

    impl DifferentiableOp<ArrayType, f64, LinearTerm<ArrayType, f64>, ProgramOpRef<f64>, LinearProgramOpRef<f64>>
        for PanicReplayOp
    {
        fn jvp(
            &self,
            _engine: &dyn Engine<
                Type = ArrayType,
                Value = f64,
                TracingOperation = ProgramOpRef<f64>,
                LinearOperation = LinearProgramOpRef<f64>,
            >,
            inputs: &[JvpTracer<f64, LinearTerm<ArrayType, f64>>],
        ) -> Result<Vec<JvpTracer<f64, LinearTerm<ArrayType, f64>>>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0].clone()])
        }
    }

    #[test]
    fn linearize_returns_the_primal_output_and_pushforward() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (primal, pushforward) = linearize(&engine, quadratic_plus_sin, 2.0f64).unwrap();

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
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, from_jvp_program) = jvp_program(&engine, quadratic_plus_sin, 2.0f64).unwrap();
        let (_, from_linearize) = linearize(&engine, quadratic_plus_sin, 2.0f64).unwrap();

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
        let engine = ArrayScalarEngine::<f64>::new();
        let (primal, pushforward) = linearize(&engine, bilinear_sin, (2.0f64, 3.0f64)).unwrap();
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
    fn linearize_program_does_not_replay_the_forward_graph_to_recover_representatives() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_jvp_rule(PanicReplayOp);
        let mut builder = GraphBuilder::<ProgramOpRef<f64>, ArrayType, f64>::new();
        let input = builder.add_input(&3.0f64);
        let output = builder.add_equation_prevalidated(
            PrimitiveOp::Custom(Arc::new(primitive)),
            vec![input],
            vec![ArrayType::scalar(DataType::F64)],
        );
        let program = Program::from_graph(builder.build::<f64, f64>(output, Placeholder, Placeholder));

        let engine = ArrayScalarEngine::<f64>::new();
        let pushforward = linearize_program(&engine, &program, vec![3.0f64]).unwrap();
        approx_eq(pushforward.call(2.5f64).unwrap(), 2.5);
    }

    #[test]
    fn transpose_linear_program_does_not_replay_the_forward_linear_graph_to_recover_representatives() {
        let primitive = LinearPrimitiveOp::custom(
            CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_transpose_rule(PanicReplayOp),
        )
        .unwrap();
        let mut builder = GraphBuilder::<LinearProgramOpRef<f64>, ArrayType, f64>::new();
        let input = builder.add_input(&0.0f64);
        let output = builder.add_equation_prevalidated(primitive, vec![input], vec![ArrayType::scalar(DataType::F64)]);
        let program = Program::from_graph(builder.build::<f64, f64>(output, Placeholder, Placeholder));
        let pushforward = LinearProgram::from_program(program, 0.0f64);

        let pullback = transpose_linear_program(&pushforward).unwrap();
        approx_eq(pullback.call(4.0f64).unwrap(), 4.0);
    }

    #[test]
    fn linear_program_display_delegates_to_the_underlying_graph() {
        let engine = ArrayScalarEngine::<f64>::new();
        let (_, pushforward): (f64, LinearProgram<ArrayType, f64, f64, f64>) =
            linearize(&engine, quadratic_plus_sin, 2.0f64).unwrap();

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

    #[test]
    fn compile_grad_produces_reusable_gradient_program() {
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad(&engine, quadratic_plus_sin, 2.0f64).unwrap();

        // d/dx(x^2 + sin(x)) = 2x + cos(x)

        // Verify at the original primal point.
        let grad_at_2 = compiled.call(2.0f64).unwrap();
        approx_eq(grad_at_2, 2.0 * 2.0 + 2.0f64.cos());

        // Verify at a DIFFERENT primal point — this is the key test.
        let grad_at_half = compiled.call(0.5f64).unwrap();
        approx_eq(grad_at_half, 2.0 * 0.5 + 0.5f64.cos());

        let grad_at_pi = compiled.call(std::f64::consts::PI).unwrap();
        approx_eq(grad_at_pi, 2.0 * std::f64::consts::PI + std::f64::consts::PI.cos());

        // The program should contain cos (from sin's derivative), not baked constants.
        let ir = compiled.to_string();
        assert!(ir.contains("cos"), "compiled grad should compute cos symbolically, not bake constants");
    }

    #[test]
    fn compile_grad_bilinear_returns_both_partial_derivatives() {
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad(&engine, bilinear_sin, (2.0f64, 3.0f64)).unwrap();

        // df/dx = y + cos(x), df/dy = x
        let (grad_x, grad_y) = compiled.call((2.0f64, 3.0f64)).unwrap();
        approx_eq(grad_x, 3.0 + 2.0f64.cos());
        approx_eq(grad_y, 2.0);

        // At a different primal point:
        let (grad_x2, grad_y2) = compiled.call((1.0f64, 5.0f64)).unwrap();
        approx_eq(grad_x2, 5.0 + 1.0f64.cos());
        approx_eq(grad_y2, 1.0);
    }

    // -----------------------------------------------------------------------
    // RematerializationPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_grad_save_all_matches_compile_grad() {
        // SaveAll should produce the same gradient as the plain compile_grad.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled_plain = compile_grad(&engine, quadratic_plus_sin, 2.0f64).unwrap();
        let compiled_save_all =
            compile_grad_with_policy(&engine, quadratic_plus_sin, 2.0f64, RematerializationPolicy::SaveAll).unwrap();

        let grad_plain = compiled_plain.call(2.0f64).unwrap();
        let grad_save_all = compiled_save_all.call(2.0f64).unwrap();
        approx_eq(grad_plain, grad_save_all);

        // Also verify at a different primal point.
        let grad_plain_2 = compiled_plain.call(0.5f64).unwrap();
        let grad_save_all_2 = compiled_save_all.call(0.5f64).unwrap();
        approx_eq(grad_plain_2, grad_save_all_2);
    }

    #[test]
    fn test_compile_grad_recompute_all_gives_correct_gradient() {
        // RecomputeAll should give d/dx(x^2 + sin(x)) = 2x + cos(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled =
            compile_grad_with_policy(&engine, quadratic_plus_sin, 2.0f64, RematerializationPolicy::RecomputeAll)
                .unwrap();

        approx_eq(compiled.call(2.0f64).unwrap(), 2.0 * 2.0 + 2.0f64.cos());
        approx_eq(compiled.call(0.5f64).unwrap(), 2.0 * 0.5 + 0.5f64.cos());
        approx_eq(
            compiled.call(std::f64::consts::PI).unwrap(),
            2.0 * std::f64::consts::PI + std::f64::consts::PI.cos(),
        );
    }

    #[test]
    fn test_compile_grad_recompute_all_matches_compile_grad() {
        // RecomputeAll should give the same numerical gradient as compile_grad.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled_plain = compile_grad(&engine, quadratic_plus_sin, 2.0f64).unwrap();
        let compiled_recompute =
            compile_grad_with_policy(&engine, quadratic_plus_sin, 2.0f64, RematerializationPolicy::RecomputeAll)
                .unwrap();

        for x in [0.0, 0.5, 1.0, 2.0, 3.0, std::f64::consts::PI] {
            let grad_plain = compiled_plain.call(x).unwrap();
            let grad_recompute = compiled_recompute.call(x).unwrap();
            approx_eq(grad_plain, grad_recompute);
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_gives_correct_gradient() {
        // Checkpoint with segment_size=2 should give the correct gradient for a function with
        // ~4 equations: x*x, sin(x), x*x + sin(x).
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 2 },
        )
        .unwrap();

        approx_eq(compiled.call(2.0f64).unwrap(), 2.0 * 2.0 + 2.0f64.cos());
        approx_eq(compiled.call(0.5f64).unwrap(), 2.0 * 0.5 + 0.5f64.cos());
    }

    #[test]
    fn test_compile_grad_checkpoint_is_reusable_at_different_primals() {
        // The compiled gradient with Checkpoint can be called at multiple primal points.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            1.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 2 },
        )
        .unwrap();

        for x in [0.0, 0.5, 1.0, 2.0, 3.0, std::f64::consts::PI] {
            let expected = 2.0 * x + x.cos();
            approx_eq(compiled.call(x).unwrap(), expected);
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_matches_compile_grad() {
        // Checkpoint should give the same numerical gradient as compile_grad.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled_plain = compile_grad(&engine, quadratic_plus_sin, 2.0f64).unwrap();
        let compiled_checkpoint = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 2 },
        )
        .unwrap();

        for x in [0.0, 0.5, 1.0, 2.0, 3.0, std::f64::consts::PI] {
            let grad_plain = compiled_plain.call(x).unwrap();
            let grad_checkpoint = compiled_checkpoint.call(x).unwrap();
            approx_eq(grad_plain, grad_checkpoint);
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_segment_size_one_matches_save_all() {
        // Checkpoint with segment_size=1 should degenerate to SaveAll.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled_save_all =
            compile_grad_with_policy(&engine, quadratic_plus_sin, 2.0f64, RematerializationPolicy::SaveAll).unwrap();
        let compiled_checkpoint = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 1 },
        )
        .unwrap();

        for x in [0.0, 1.0, 2.0] {
            approx_eq(compiled_save_all.call(x).unwrap(), compiled_checkpoint.call(x).unwrap());
        }
    }

    #[test]
    fn test_compile_grad_checkpoint_large_segment_wraps_whole_program() {
        // Checkpoint with a segment_size larger than the number of equations should wrap
        // the entire program in a single RematerializeOp, equivalent to RecomputeAll.
        let engine = ArrayScalarEngine::<f64>::new();
        let compiled = compile_grad_with_policy(
            &engine,
            quadratic_plus_sin,
            2.0f64,
            RematerializationPolicy::Checkpoint { segment_size: 100 },
        )
        .unwrap();

        for x in [0.0, 1.0, 2.0, std::f64::consts::PI] {
            approx_eq(compiled.call(x).unwrap(), 2.0 * x + x.cos());
        }
    }
}
