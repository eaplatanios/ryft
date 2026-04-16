//! Linearization, transposition, and higher-order differentiation utilities.
//!
//! This module turns forward-mode traces into staged linear programs, transposes those programs for reverse-mode
//! differentiation, and materializes dense Jacobians/Hessians for coordinate-based leaf types.

use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, Mul, Neg},
    rc::Rc,
};

use ryft_macros::Parameter;

use crate::{
    parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder},
    tracing_v2::{
        FloatExt, MatrixOps, OneLike, TraceError, Traceable, Value, ZeroLike,
        batch::{Batch, stack, unstack},
        forward::{JvpTracer, TangentSpace},
        graph::{AtomId, AtomSource, Equation, Graph, GraphBuilder},
        jit::{CompiledFunction, JitTracer, try_jit, try_trace_program},
        operations::rematerialize::{FlatTracedRematerialize, RematerializeOp},
        operations::reshape::ReshapeOps,
        ops::{DifferentiableOp, InterpretableOp, LinearOp, LinearPrimitiveOp, Op, PrimitiveOp},
        program::{LinearProgramBuilder, LinearProgramOpRef, Program, ProgramBuilder, ProgramOpRef},
    },
    types::{ArrayType, Type, Typed},
};

/// Tangent representation backed by atoms in a staged linear graph.
#[derive(Clone, Parameter)]
pub struct LinearTerm<T: Type + Clone + Display, V: Typed<T> + Clone + Parameter> {
    atom: AtomId,
    builder: Rc<RefCell<GraphBuilder<LinearPrimitiveOp<T, V>, T, V>>>,
}

impl<T: Type + Clone + Display, V: Traceable<T>> std::fmt::Debug for LinearTerm<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("LinearTerm").field("atom", &self.atom).finish()
    }
}

impl<T: Type + Clone + Display, V: Traceable<T>> LinearTerm<T, V> {
    #[inline]
    pub fn atom(&self) -> AtomId {
        self.atom
    }

    #[inline]
    pub fn builder_handle(&self) -> Rc<RefCell<GraphBuilder<LinearPrimitiveOp<T, V>, T, V>>> {
        self.builder.clone()
    }

    #[inline]
    pub fn from_staged_parts(atom: AtomId, builder: Rc<RefCell<GraphBuilder<LinearPrimitiveOp<T, V>, T, V>>>) -> Self {
        Self { atom, builder }
    }

    /// Stages a multi-input operation in the tangent program builder.
    ///
    /// Shape validation is performed via [`Op::abstract_eval`]. Concrete evaluation is intentionally
    /// skipped because tangent-program outputs remain abstract until the staged linear program is
    /// replayed on concrete tangents.
    pub fn apply_staged_op(
        inputs: &[Self],
        op: LinearPrimitiveOp<T, V>,
        output_count: usize,
    ) -> Result<Vec<Self>, TraceError>
    where
        LinearPrimitiveOp<T, V>: Op<T>,
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
                .map(|id| borrow.atom(*id).expect("staged input should exist").abstract_value.clone())
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
    pub fn apply_linear_op(self, op: LinearPrimitiveOp<T, V>) -> Self {
        let mut borrow = self.builder.borrow_mut();
        let input_atom = borrow.atom(self.atom).expect("staged input should exist");
        let abstract_value = input_atom.abstract_value.clone();
        let atom = borrow.add_equation_prevalidated(op, vec![self.atom], vec![abstract_value])[0];
        drop(borrow);
        Self { atom, builder: self.builder }
    }

    /// Stages an addition of two tangent terms.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        debug_assert!(Rc::ptr_eq(&self.builder, &rhs.builder));
        let mut borrow = self.builder.borrow_mut();
        let input_atom = borrow.atom(self.atom).expect("staged input should exist");
        let abstract_value = input_atom.abstract_value.clone();
        let atom =
            borrow.add_equation_prevalidated(LinearPrimitiveOp::Add, vec![self.atom, rhs.atom], vec![abstract_value])
                [0];
        drop(borrow);
        Self { atom, builder: self.builder }
    }

    /// Stages a negation of this tangent term.
    #[inline]
    pub fn neg(self) -> Self {
        self.apply_linear_op(LinearPrimitiveOp::Neg)
    }

    /// Stages a scaling of this tangent term by a concrete factor.
    #[inline]
    pub fn scale(self, factor: V) -> Self {
        self.apply_linear_op(LinearPrimitiveOp::Scale { factor })
    }
}

impl<T: Type + Clone + Display, V: Traceable<T> + ZeroLike> TangentSpace<T, V> for LinearTerm<T, V>
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
pub type Linearized<V> = JvpTracer<V, LinearTerm<ArrayType, V>>;

#[inline]
fn flat_leaf_parameter_structure(count: usize) -> Vec<Placeholder> {
    vec![Placeholder; count]
}

/// Staged linear map produced by `linearize`, `jvp_program`, or `vjp`.
pub struct LinearProgram<
    T: Type + Clone + Display,
    V: Typed<T> + Clone + Parameter,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> {
    program: Program<T, V, Input, Output, LinearPrimitiveOp<T, V>>,
    zero: V,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<
    T: Type + Clone + Display,
    V: Traceable<T> + Clone,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> Clone for LinearProgram<T, V, Input, Output>
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), zero: self.zero.clone(), marker: PhantomData }
    }
}

impl<T: Type + Clone + Display, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>>
    LinearProgram<T, V, Input, Output>
{
    #[inline]
    pub fn from_program(program: Program<T, V, Input, Output, LinearPrimitiveOp<T, V>>, zero: V) -> Self {
        Self { program, zero, marker: PhantomData }
    }

    /// Returns the staged graph backing this linear program.
    #[inline]
    pub fn program(&self) -> &Program<T, V, Input, Output, LinearPrimitiveOp<T, V>> {
        &self.program
    }

    /// Applies the linear program to a concrete input tangent or cotangent.
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        LinearPrimitiveOp<T, V>: InterpretableOp<T, V>,
        V: Parameterized<V> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
        V::ParameterStructure: Clone + PartialEq,
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        self.program.call(input)
    }
}

impl<V: Traceable<ArrayType>, Input: Parameterized<V>, Output: Parameterized<V>> LinearProgram<ArrayType, V, Input, Output> {
    /// Transposes the linear program, turning a pushforward into a pullback.
    pub fn transpose(&self) -> Result<LinearProgram<ArrayType, V, Output, Input>, TraceError>
    where
        V: Parameterized<V, To<ArrayType> = ArrayType, ParameterStructure = Placeholder>
            + FloatExt
            + ZeroLike
            + OneLike
            + MatrixOps
            + ReshapeOps,
        V::Family: ParameterizedFamily<ArrayType>,
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
fn transpose<V>(
    op: &LinearProgramOpRef<V>,
    builder: &Rc<RefCell<LinearProgramBuilder<V>>>,
    output_cotangents: &[AtomId],
) -> Result<Vec<Option<AtomId>>, TraceError>
where
    V: Traceable<ArrayType>
        + Parameterized<V>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>,
    V::ParameterStructure: Clone + PartialEq,
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

pub fn linearize_program<V, Input, Output>(
    program: &Program<ArrayType, V, Input, Output>,
) -> Result<LinearProgram<ArrayType, V, Input, Output>, TraceError>
where
    V: Traceable<ArrayType>
        + Parameterized<V>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
{
    fn tangent_for_atom<V, Input, Output>(
        _graph: &Graph<ProgramOpRef<V>, ArrayType, V, Input, Output>,
        primal_values: &[Option<V>],
        builder: &Rc<RefCell<LinearProgramBuilder<V>>>,
        tangents: &mut [Option<LinearTerm<ArrayType, V>>],
        atom_id: AtomId,
    ) -> Result<LinearTerm<ArrayType, V>, TraceError>
    where
        V: Traceable<ArrayType> + FloatExt + ZeroLike,
        Input: Parameterized<V>,
        Output: Parameterized<V>,
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
    let representative_inputs = graph.representative_input_values()?;
    let zero = graph
        .input_atoms()
        .first()
        .and_then(|_| representative_inputs.first())
        .map(ZeroLike::zero_like)
        .ok_or(TraceError::EmptyParameterizedValue)?;
    let builder = Rc::new(RefCell::new(LinearProgramBuilder::new()));
    let mut primals = vec![None; graph.atom_count()];
    let mut tangents = vec![None; graph.atom_count()];
    for (input_atom, representative_input) in graph.input_atoms().iter().copied().zip(representative_inputs.iter()) {
        let input = graph.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
        primals[input_atom] = Some(representative_input.clone());
        let tangent_atom = builder
            .borrow_mut()
            .add_input_abstract(input.abstract_value.clone(), representative_input.zero_like());
        tangents[input_atom] = Some(LinearTerm::from_staged_parts(tangent_atom, builder.clone()));
    }
    for (atom_id, atom) in graph.atoms_iter() {
        if matches!(atom.source, AtomSource::Constant) {
            primals[atom_id] = Some(atom.constant_value().cloned().ok_or(TraceError::InternalInvariantViolation(
                "staged graph constant atom did not retain a literal value",
            ))?);
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
        let output_duals = DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&equation.op, input_duals.as_slice())?;
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

pub fn transpose_linear_program<V, Input, Output>(
    program: &LinearProgram<ArrayType, V, Input, Output>,
) -> Result<LinearProgram<ArrayType, V, Output, Input>, TraceError>
where
    V: Traceable<ArrayType>
        + Parameterized<V, To<ArrayType> = ArrayType, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
    V::Family: ParameterizedFamily<ArrayType>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
{
    let zero = program.zero.zero_like();
    transpose_linear_program_with_output_inputs(program, |builder: &mut LinearProgramBuilder<V>, abstract_value, _| {
        Ok(builder.add_input_abstract(abstract_value.clone(), zero.clone()))
    })
}

fn transpose_linear_program_with_output_inputs<V, Input, Output, F>(
    program: &LinearProgram<ArrayType, V, Input, Output>,
    mut make_output_cotangent_input: F,
) -> Result<LinearProgram<ArrayType, V, Output, Input>, TraceError>
where
    V: Traceable<ArrayType> + Parameterized<V> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    F: FnMut(&mut LinearProgramBuilder<V>, &ArrayType, usize) -> Result<AtomId, TraceError>,
{
    fn accumulate<V>(
        builder: &Rc<RefCell<LinearProgramBuilder<V>>>,
        adjoints: &mut [Option<AtomId>],
        atom: AtomId,
        contribution: AtomId,
    ) -> Result<(), TraceError>
    where
        V: Traceable<ArrayType> + FloatExt,
    {
        adjoints[atom] = Some(match adjoints[atom] {
            Some(existing) => {
                let mut builder_borrow = builder.borrow_mut();
                let abstract_value =
                    builder_borrow.atom(existing).expect("adjoint atom should exist").abstract_value.clone();
                builder_borrow.add_equation_prevalidated(
                    LinearPrimitiveOp::Add,
                    vec![existing, contribution],
                    vec![abstract_value],
                )[0]
            }
            None => contribution,
        });
        Ok(())
    }

    let graph = program.program.graph();
    let builder = Rc::new(RefCell::new(LinearProgramBuilder::<V>::new()));
    let mut output_cotangent_inputs = Vec::with_capacity(graph.outputs().len());
    for (output_index, output) in graph.outputs().iter().enumerate() {
        let output_atom = graph.atom(*output).ok_or(TraceError::UnboundAtomId { id: *output })?;
        let cotangent_input =
            make_output_cotangent_input(&mut builder.borrow_mut(), &output_atom.abstract_value, output_index)?;
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
pub fn transpose_linear_program_with_output_examples<V, Input, Output>(
    program: &LinearProgram<ArrayType, V, Input, Output>,
    output_examples: &[V],
) -> Result<LinearProgram<ArrayType, V, Output, Input>, TraceError>
where
    V: Traceable<ArrayType> + Parameterized<V> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
{
    let expected_output_count = program.program().graph().outputs().len();
    if output_examples.len() != expected_output_count {
        return Err(TraceError::InvalidInputCount { expected: expected_output_count, got: output_examples.len() });
    }
    transpose_linear_program_with_output_inputs(
        program,
        |builder: &mut LinearProgramBuilder<V>, abstract_value, output_index| {
            Ok(builder.add_input_abstract(abstract_value.clone(), output_examples[output_index].zero_like()))
        },
    )
}

fn lift_traced_constant<V>(constant: &V, inputs: &[JitTracer<ArrayType, V>]) -> Result<JitTracer<ArrayType, V>, TraceError>
where
    V: Traceable<ArrayType>,
{
    let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
    let atom = exemplar.builder_handle().borrow_mut().add_constant(constant.clone());
    Ok(JitTracer::from_staged_parts(constant.clone(), atom, exemplar.builder_handle(), exemplar.staging_error_handle()))
}

fn lift_linearized_traced_constant<V>(
    constant: &V,
    inputs: &[Linearized<JitTracer<ArrayType, V>>],
) -> Result<Linearized<JitTracer<ArrayType, V>>, TraceError>
where
    V: Traceable<ArrayType> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
{
    let exemplar = inputs.first().ok_or(TraceError::EmptyParameterizedValue)?;
    let primal = lift_traced_constant(constant, std::slice::from_ref(&exemplar.primal))?;
    let tangent_atom = exemplar.tangent.builder_handle().borrow_mut().add_constant(primal.zero_like());
    let tangent = LinearTerm::from_staged_parts(tangent_atom, exemplar.tangent.builder_handle());
    Ok(Linearized { primal, tangent })
}

fn replay_program_graph_with<GraphInput, GraphOutput, V, R, LiftConstant, ApplyOp>(
    graph: &Graph<ProgramOpRef<V>, ArrayType, V, GraphInput, GraphOutput>,
    inputs: Vec<R>,
    lift_constant: LiftConstant,
    apply_op: ApplyOp,
) -> Result<Vec<R>, TraceError>
where
    GraphInput: Parameterized<V>,
    GraphOutput: Parameterized<V>,
    V: Traceable<ArrayType>,
    R: Clone,
    LiftConstant: Fn(&V, &[R]) -> Result<R, TraceError>,
    ApplyOp: Fn(&PrimitiveOp<ArrayType, V>, Vec<R>) -> Result<Vec<R>, TraceError>,
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
                values[atom_id] = Some(lift_constant(
                    atom.constant_value().ok_or(TraceError::InternalInvariantViolation(
                        "staged graph constant atom did not retain a literal value",
                    ))?,
                    seed_inputs.as_slice(),
                )?);
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

pub fn replay_program_graph_linearized_jit<GraphInput, GraphOutput, V>(
    graph: &Graph<ProgramOpRef<V>, ArrayType, V, GraphInput, GraphOutput>,
    inputs: Vec<Linearized<JitTracer<ArrayType, V>>>,
) -> Result<Vec<Linearized<JitTracer<ArrayType, V>>>, TraceError>
where
    GraphInput: Parameterized<V>,
    GraphOutput: Parameterized<V>,
    V: Traceable<ArrayType>
        + Parameterized<V>
        + FloatExt
        + ZeroLike
        + OneLike
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + MatrixOps
        + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
{
    replay_program_graph_with(graph, inputs, lift_linearized_traced_constant, |op, values| {
        InterpretableOp::<ArrayType, Linearized<JitTracer<ArrayType, V>>>::interpret(op, &values)
    })
}

pub(crate) fn try_linearize_traced_program<V>(
    program: &Program<ArrayType, V, Vec<V>, Vec<V>>,
    primals: Vec<JitTracer<ArrayType, V>>,
) -> Result<
    (
        Vec<JitTracer<ArrayType, V>>,
        LinearProgram<ArrayType, JitTracer<ArrayType, V>, Vec<JitTracer<ArrayType, V>>, Vec<JitTracer<ArrayType, V>>>,
    ),
    TraceError,
>
where
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
{
    let zero = primals.first().map(ZeroLike::zero_like).ok_or(TraceError::EmptyParameterizedValue)?;
    let input_count = primals.len();
    let builder = Rc::new(RefCell::new(LinearProgramBuilder::new()));
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
    let program = Program::from_graph(builder.build::<Vec<JitTracer<ArrayType, V>>, Vec<JitTracer<ArrayType, V>>>(
        tangent_outputs,
        vec![Placeholder; input_count],
        vec![Placeholder; primal_outputs.len()],
    ))
    .simplify()?;
    Ok((primal_outputs.clone(), LinearProgram::from_program(program, zero)))
}

pub fn try_jvp_program<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Input, Output>), TraceError>
where
    V: Traceable<ArrayType> + Parameterized<V> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<ArrayType, V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Result<Output::To<JitTracer<ArrayType, V>>, TraceError>,
{
    let (primal_output, program) = try_trace_program(function, primals)?;
    Ok((primal_output, linearize_program(&program)?))
}

/// Runs JVP for already traced inputs by staging the inner function once over base values and
/// replaying the resulting pushforward in the surrounding trace.
pub fn try_jvp_traced<F, Input, Output, V>(
    function: F,
    primals: Input,
    tangents: Input,
) -> Result<(Output, Output), TraceError>
where
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
    Input: Parameterized<JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<V>,
    Output: Parameterized<JitTracer<ArrayType, V>, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<V>,
    Vec<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Vec<JitTracer<ArrayType, V>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Input::To<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Input>,
    Output::To<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Output>,
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
    let (primal_output, traced_program): (Output::To<V>, Program<ArrayType, V, Input::To<V>, Output::To<V>>) =
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
        flat_leaf_parameter_structure(traced_primals.len()),
        flat_leaf_parameter_structure(primal_output.parameter_count()),
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
) -> Result<(Output, LinearProgram<ArrayType, V, Input, Output>), TraceError>
where
    V: Traceable<ArrayType> + Parameterized<V> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<ArrayType, V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Output::To<JitTracer<ArrayType, V>>,
{
    try_jvp_program(|input| Ok(function(input)), primals)
}

/// Alias for [`jvp_program`] that emphasizes the returned linear map.
#[allow(private_bounds)]
pub fn linearize<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Input, Output>), TraceError>
where
    V: Traceable<ArrayType> + Parameterized<V> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<ArrayType, V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Output::To<JitTracer<ArrayType, V>>,
{
    jvp_program(function, primals)
}

pub fn try_vjp<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Output, Input>), TraceError>
where
    V: Traceable<ArrayType> + Parameterized<V> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Result<Output::To<JitTracer<ArrayType, V>>, TraceError>,
{
    let (output, pushforward) = try_jvp_program::<F, Input, Output, V>(function, primals)?;
    let output_examples = output.parameters().cloned().collect::<Vec<_>>();
    let pullback = transpose_linear_program_with_output_examples(&pushforward, output_examples.as_slice())?;
    Ok((output, pullback))
}

/// Returns the primal output together with a pullback produced by transposing the staged pushforward.
#[allow(private_bounds)]
pub fn vjp<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<(Output, LinearProgram<ArrayType, V, Output, Input>), TraceError>
where
    V: Traceable<ArrayType> + Parameterized<V> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Output::To<JitTracer<ArrayType, V>>,
{
    try_vjp(|input| Ok(function(input)), primals)
}

fn try_grad<F, Input, V>(function: F, primals: Input) -> Result<Input, TraceError>
where
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Result<JitTracer<ArrayType, V>, TraceError>,
{
    let (output, pullback): (V, LinearProgram<ArrayType, V, V, Input>) = try_vjp(function, primals)?;
    pullback.call(output.one_like())
}

/// Dispatch trait used by [`grad`] so it can operate both on concrete values and on already traced values.
#[doc(hidden)]
pub trait GradInvocationLeaf<Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>>:
    Parameter + Sized
{
    /// Base leaf value used for the staged inner program.
    type Base: Traceable<ArrayType> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps;

    /// Return type produced by [`grad`] for the corresponding input regime.
    type Return;

    /// Traced input type expected by the user-provided function.
    type FunctionInput;

    /// Invokes [`grad`] for one concrete leaf regime.
    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<ArrayType, Self::Base>;
}

/// Concrete-value dispatch for [`grad`]: traces the user function with [`JitTracer`] to build a staged
/// reverse-mode gradient and evaluates it at the supplied primals.
impl<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + Value<ArrayType>
        + MatrixOps
        + ReshapeOps,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
> GradInvocationLeaf<Input> for V
where
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    V: Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
{
    type Base = V;
    type Return = Input;
    type FunctionInput = Input::To<JitTracer<ArrayType, V>>;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<ArrayType, Self::Base>,
    {
        try_grad(|input| Ok(function(input)), primals)
    }
}

/// Already-traced dispatch for [`grad`]: replays the user function symbolically inside an enclosing
/// [`JitTracer`] scope, linearizes the resulting [`Program`], transposes the pushforward into a pullback,
/// and stages the full backward pass so it becomes part of the outer compiled graph.
impl<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
> GradInvocationLeaf<Input> for JitTracer<ArrayType, V>
where
    Input::Family: ParameterizedFamily<V>,
    V: Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Vec<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Vec<JitTracer<ArrayType, V>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
{
    type Base = V;
    type Return = Input;
    type FunctionInput = Input;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<ArrayType, Self::Base>,
    {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let staged_primals = Input::To::<V>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|primal| primal.value.clone()).collect::<Vec<_>>(),
        )?;
        let (_, traced_program): (V, Program<ArrayType, V, Input::To<V>, V>) = try_trace_program(
            |staged_input| {
                let adapted_input = Input::from_parameters(
                    input_structure.clone(),
                    staged_input.into_parameters().collect::<Vec<_>>(),
                )?;
                Ok(function(adapted_input))
            },
            staged_primals,
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
        let pullback = transpose_linear_program_with_output_examples(&pushforward, outputs.as_slice())?;
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
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
> GradInvocationLeaf<Input> for Batch<V>
where
    V: Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<ArrayType, V>>,
    Input::To<V>: Clone
        + Parameterized<
            V,
            ParameterStructure: Clone + PartialEq,
            To<Batch<V>> = Input,
            To<JitTracer<ArrayType, V>> = Input::To<JitTracer<ArrayType, V>>,
        >,
    <Input::To<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>> + ParameterizedFamily<Batch<V>>,
    Vec<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Vec<JitTracer<ArrayType, V>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
{
    type Base = V;
    type Return = Input;
    type FunctionInput = Input::To<JitTracer<ArrayType, V>>;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<ArrayType, Self::Base>,
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
        let (_, traced_program): (V, Program<ArrayType, V, Input::To<V>, V>) =
            try_trace_program(|staged_input| Ok(function(staged_input)), lane_primals[0].clone())?;

        // Reshape the program to flat Vec<V> inputs and outputs for the JIT compilation step.
        let flat_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
            flat_leaf_parameter_structure(parameter_count),
            flat_leaf_parameter_structure(1),
        ))
        .simplify()?;

        // Compile the gradient into a reusable program by wrapping linearize + transpose + pullback
        // inside a JIT scope. This stages the full backward pass symbolically so it can be replayed
        // at arbitrary primal points.
        let (_, compiled_grad): (Vec<V>, CompiledFunction<ArrayType, V, Vec<V>, Vec<V>>) = try_jit(
            |jit_primals: Vec<JitTracer<ArrayType, V>>| {
                let (outputs, pushforward) = try_linearize_traced_program(&flat_program, jit_primals)?;
                if outputs.len() != 1 {
                    return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
                }
                let pullback = transpose_linear_program_with_output_examples(&pushforward, outputs.as_slice())?;
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
pub fn grad<F, Input, Leaf>(
    function: F,
    primals: Input,
) -> Result<<Leaf as GradInvocationLeaf<Input>>::Return, TraceError>
where
    Leaf: GradInvocationLeaf<Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as GradInvocationLeaf<Input>>::FunctionInput,
    ) -> JitTracer<ArrayType, <Leaf as GradInvocationLeaf<Input>>::Base>,
{
    Leaf::invoke(function, primals)
}

/// Dispatch trait used by [`value_and_grad`] so it can operate both on concrete values and on already traced values.
#[doc(hidden)]
pub trait ValueAndGradInvocationLeaf<Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>>:
    Parameter + Sized
{
    /// Base leaf value used for the staged inner program.
    type Base: Traceable<ArrayType> + FloatExt + ZeroLike + OneLike + MatrixOps + ReshapeOps;

    /// Return type produced by [`value_and_grad`] for the corresponding input regime.
    type Return;

    /// Traced input type expected by the user-provided function.
    type FunctionInput;

    /// Invokes [`value_and_grad`] for one concrete leaf regime.
    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<ArrayType, Self::Base>;
}

/// Concrete-value dispatch for [`value_and_grad`]: evaluates the user function via [`vjp`] and
/// pulls back a unit seed to obtain both the primal scalar output and its gradient.
impl<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + Value<ArrayType>
        + MatrixOps
        + ReshapeOps,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
> ValueAndGradInvocationLeaf<Input> for V
where
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    V: Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
{
    type Base = V;
    type Return = (V, Input);
    type FunctionInput = Input::To<JitTracer<ArrayType, V>>;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<ArrayType, Self::Base>,
    {
        let (output, pullback): (V, LinearProgram<ArrayType, V, V, Input>) = vjp(function, primals)?;
        let gradient = pullback.call(output.one_like())?;
        Ok((output, gradient))
    }
}

/// Already-traced dispatch for [`value_and_grad`]: replays the user function symbolically inside an
/// enclosing [`JitTracer`] scope, linearizes, transposes, and stages both the forward output and the
/// backward gradient so they become part of the outer compiled graph.
impl<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
    Input: Parameterized<Self, ParameterStructure: Clone + PartialEq>,
> ValueAndGradInvocationLeaf<Input> for JitTracer<ArrayType, V>
where
    Input::Family: ParameterizedFamily<V>,
    V: Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Vec<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Vec<JitTracer<ArrayType, V>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Input::To<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Input>,
{
    type Base = V;
    type Return = (JitTracer<ArrayType, V>, Input);
    type FunctionInput = Input;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<ArrayType, Self::Base>,
    {
        let input_structure = primals.parameter_structure();
        let traced_primals = primals.into_parameters().collect::<Vec<_>>();
        let staged_primals = Input::To::<V>::from_parameters(
            input_structure.clone(),
            traced_primals.iter().map(|primal| primal.value.clone()).collect::<Vec<_>>(),
        )?;
        let (_, traced_program): (V, Program<ArrayType, V, Input::To<V>, V>) = try_trace_program(
            |staged_input| {
                let adapted_input = Input::from_parameters(
                    input_structure.clone(),
                    staged_input.into_parameters().collect::<Vec<_>>(),
                )?;
                Ok(function(adapted_input))
            },
            staged_primals,
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
        let traced_output = outputs[0].clone();
        let pullback = transpose_linear_program_with_output_examples(&pushforward, outputs.as_slice())?;
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
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
    Input: Parameterized<Batch<V>, ParameterStructure: Clone + PartialEq>,
> ValueAndGradInvocationLeaf<Input> for Batch<V>
where
    V: Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Input::Family: ParameterizedFamily<V> + ParameterizedFamily<JitTracer<ArrayType, V>>,
    Input::To<V>: Clone
        + Parameterized<
            V,
            ParameterStructure: Clone + PartialEq,
            To<Batch<V>> = Input,
            To<JitTracer<ArrayType, V>> = Input::To<JitTracer<ArrayType, V>>,
        >,
    <Input::To<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>> + ParameterizedFamily<Batch<V>>,
    Vec<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Vec<JitTracer<ArrayType, V>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
{
    type Base = V;
    type Return = (Batch<V>, Input);
    type FunctionInput = Input::To<JitTracer<ArrayType, V>>;

    fn invoke<F>(function: F, primals: Input) -> Result<Self::Return, TraceError>
    where
        F: FnOnce(Self::FunctionInput) -> JitTracer<ArrayType, Self::Base>,
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
        let (_, traced_program): (V, Program<ArrayType, V, Input::To<V>, V>) =
            try_trace_program(|staged_input| Ok(function(staged_input)), lane_primals[0].clone())?;

        // Reshape the program to flat Vec<V> inputs and outputs for the JIT compilation step.
        let flat_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
            flat_leaf_parameter_structure(parameter_count),
            flat_leaf_parameter_structure(1),
        ))
        .simplify()?;

        // Compile both the forward evaluation and gradient into a reusable program.
        let (_, compiled_vg): (Vec<V>, CompiledFunction<ArrayType, V, Vec<V>, Vec<V>>) = try_jit(
            |jit_primals: Vec<JitTracer<ArrayType, V>>| {
                let (outputs, pushforward) = try_linearize_traced_program(&flat_program, jit_primals)?;
                if outputs.len() != 1 {
                    return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
                }
                let pullback = transpose_linear_program_with_output_examples(&pushforward, outputs.as_slice())?;
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
pub fn value_and_grad<F, Input, Leaf>(
    function: F,
    primals: Input,
) -> Result<<Leaf as ValueAndGradInvocationLeaf<Input>>::Return, TraceError>
where
    Leaf: ValueAndGradInvocationLeaf<Input>,
    Input: Parameterized<Leaf, ParameterStructure: Clone + PartialEq>,
    F: FnOnce(
        <Leaf as ValueAndGradInvocationLeaf<Input>>::FunctionInput,
    ) -> JitTracer<ArrayType, <Leaf as ValueAndGradInvocationLeaf<Input>>::Base>,
{
    Leaf::invoke(function, primals)
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

fn try_jacfwd<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue + Parameterized<V> + FloatExt + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Result<Output::To<JitTracer<ArrayType, V>>, TraceError>,
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
    V: CoordinateValue + Parameterized<V> + FloatExt + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Output::To<JitTracer<ArrayType, V>>,
{
    try_jacfwd::<_, Input, Output, V>(|input| Ok(function(input)), primals)
}

fn try_jacrev<F, Input, Output, V>(
    function: F,
    primals: Input,
) -> Result<DenseJacobian<V::Coordinate, Input::ParameterStructure, Output::ParameterStructure>, TraceError>
where
    V: CoordinateValue + Parameterized<V, ParameterStructure = Placeholder> + FloatExt + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Result<Output::To<JitTracer<ArrayType, V>>, TraceError>,
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
    V: CoordinateValue + Parameterized<V, ParameterStructure = Placeholder> + FloatExt + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Output: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Output::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Output::To<JitTracer<ArrayType, V>>,
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
    V: CoordinateValue + Parameterized<V> + FloatExt + MatrixOps + ReshapeOps,
    V::ParameterStructure: Clone + PartialEq,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: FnOnce(Input::To<JitTracer<ArrayType, V>>) -> Input::To<JitTracer<ArrayType, V>>,
{
    jacfwd::<F, Input, Input, V>(gradient_function, primals)
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
pub fn compile_grad<F, Input, V>(
    function: F,
    example_primals: Input,
) -> Result<CompiledFunction<ArrayType, V, Input, Input>, TraceError>
where
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    V: Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Vec<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Vec<JitTracer<ArrayType, V>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: Fn(Input::To<JitTracer<ArrayType, V>>) -> JitTracer<ArrayType, V>,
{
    let input_structure = example_primals.parameter_structure();
    let (_, compiled) = try_jit(
        |primals: Input::To<JitTracer<ArrayType, V>>| {
            let traced_primals = primals.into_parameters().collect::<Vec<_>>();
            let staged_primals = Input::To::<V>::from_parameters(
                input_structure.clone(),
                traced_primals.iter().map(|primal| primal.value.clone()).collect::<Vec<_>>(),
            )?;
            let (_, traced_program): (V, Program<ArrayType, V, Input::To<V>, V>) = try_trace_program(
                |staged_input| {
                    let adapted_input = Input::To::<JitTracer<ArrayType, V>>::from_parameters(
                        input_structure.clone(),
                        staged_input.into_parameters().collect::<Vec<_>>(),
                    )?;
                    Ok(function(adapted_input))
                },
                staged_primals,
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
            let pullback = transpose_linear_program_with_output_examples(&pushforward, outputs.as_slice())?;
            let traced_gradient = pullback.call(vec![outputs[0].one_like()])?;
            Ok(Input::To::<JitTracer<ArrayType, V>>::from_parameters(input_structure.clone(), traced_gradient)?)
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
pub fn compile_grad_with_policy<F, Input, V>(
    function: F,
    example_primals: Input,
    policy: RematerializationPolicy,
) -> Result<CompiledFunction<ArrayType, V, Input, Input>, TraceError>
where
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps
        + Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Vec<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Vec<JitTracer<ArrayType, V>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: Fn(Input::To<JitTracer<ArrayType, V>>) -> JitTracer<ArrayType, V>,
{
    match policy {
        RematerializationPolicy::SaveAll => compile_grad(&function, example_primals),
        RematerializationPolicy::RecomputeAll => compile_grad_segmented(&function, example_primals, None),
        RematerializationPolicy::Checkpoint { segment_size } => {
            if segment_size <= 1 {
                return compile_grad(&function, example_primals);
            }
            compile_grad_segmented(&function, example_primals, Some(segment_size))
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
fn compile_grad_segmented<F, Input, V>(
    function: &F,
    example_primals: Input,
    segment_size: Option<usize>,
) -> Result<CompiledFunction<ArrayType, V, Input, Input>, TraceError>
where
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
    Input: Parameterized<V, ParameterStructure: Clone + PartialEq>,
    Input::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    V: Parameterized<V, To<JitTracer<ArrayType, V>> = JitTracer<ArrayType, V>, ParameterStructure: Clone + PartialEq>,
    V::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    Vec<V>: Parameterized<V, To<JitTracer<ArrayType, V>> = Vec<JitTracer<ArrayType, V>>, ParameterStructure = Vec<Placeholder>>,
    <Vec<V> as Parameterized<V>>::Family: ParameterizedFamily<JitTracer<ArrayType, V>>,
    F: Fn(Input::To<JitTracer<ArrayType, V>>) -> JitTracer<ArrayType, V>,
{
    let input_structure = example_primals.parameter_structure();
    let (_, compiled) = try_jit(
        |primals: Input::To<JitTracer<ArrayType, V>>| {
            let traced_primals = primals.into_parameters().collect::<Vec<_>>();

            // Step 1: Trace the function at the base V level to get a program.
            let staged_primals = Input::To::<V>::from_parameters(
                input_structure.clone(),
                traced_primals.iter().map(|primal| primal.value.clone()).collect::<Vec<_>>(),
            )?;
            let (_, traced_program): (V, Program<ArrayType, V, Input::To<V>, V>) = try_trace_program(
                |staged_input| {
                    let adapted_input = Input::To::<JitTracer<ArrayType, V>>::from_parameters(
                        input_structure.clone(),
                        staged_input.into_parameters().collect::<Vec<_>>(),
                    )?;
                    Ok(function(adapted_input))
                },
                staged_primals,
            )?;
            let traced_program = Program::from_graph(traced_program.graph().clone_with_structures::<Vec<V>, Vec<V>>(
                flat_leaf_parameter_structure(traced_primals.len()),
                flat_leaf_parameter_structure(1),
            ))
            .simplify()?;

            // Step 2: Segment the traced program to insert rematerialization boundaries.
            let segmented_program = match segment_size {
                None => wrap_program_in_rematerialize(&traced_program)?,
                Some(size) => segment_program(&traced_program, size)?,
            };

            // Step 3: Linearize and transpose the segmented program to produce the pullback.
            // `try_linearize_traced_program` replays the graph at the JitTracer level (staging
            // both forward and backward equations in the outer JIT builder) and returns the
            // primal outputs alongside the linear pushforward map.
            let (outputs, pushforward) = try_linearize_traced_program(&segmented_program, traced_primals)?;
            if outputs.len() != 1 {
                return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
            }
            let pullback = transpose_linear_program_with_output_examples(&pushforward, outputs.as_slice())?;
            let traced_gradient = pullback.call(vec![outputs[0].one_like()])?;
            Ok(Input::To::<JitTracer<ArrayType, V>>::from_parameters(input_structure.clone(), traced_gradient)?)
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
fn segment_program<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
>(
    program: &Program<ArrayType, V, Vec<V>, Vec<V>>,
    segment_size: usize,
) -> Result<Program<ArrayType, V, Vec<V>, Vec<V>>, TraceError>
where
    V::ParameterStructure: Clone + PartialEq,
{
    let graph = program.graph();
    let representative_values = graph.representative_atom_values()?;
    let representative_inputs = graph.representative_input_values()?;
    let equations = graph.equations();

    // If the program has fewer equations than a single segment, no segmentation is needed — wrap the
    // whole thing in a single RematerializeOp.
    if equations.len() <= segment_size {
        return wrap_program_in_rematerialize(program);
    }

    // Divide equations into segments.
    let segments: Vec<&[Equation<ProgramOpRef<V>>]> = equations.chunks(segment_size).collect();

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
    let mut outer_builder: ProgramBuilder<V> = ProgramBuilder::new();

    // Map from original atom IDs to outer-program atom IDs.
    let mut atom_mapping: Vec<Option<AtomId>> = vec![None; graph.atom_count()];

    // Register program inputs in the outer builder.
    for (&input_atom, representative_input) in input_atoms.iter().zip(representative_inputs.iter()) {
        let atom = graph.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
        let outer_atom = outer_builder.add_input_abstract(atom.abstract_value.clone(), representative_input.clone());
        atom_mapping[input_atom] = Some(outer_atom);
    }

    // Register constants that are used by equations (they might be referenced across segments).
    for (atom_id, atom) in graph.atoms_iter() {
        if matches!(atom.source, AtomSource::Constant) {
            let outer_atom = outer_builder.add_constant(atom.constant_value().cloned().ok_or(
                TraceError::InternalInvariantViolation("staged graph constant atom did not retain a literal value"),
            )?);
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
                    .map(|atom| atom.abstract_value.clone())
            })
            .collect::<Result<_, _>>()?;
        let output_types: Vec<_> = boundary_output_atoms
            .iter()
            .map(|&atom_id| {
                graph
                    .atom(atom_id)
                    .ok_or(TraceError::UnboundAtomId { id: atom_id })
                    .map(|atom| atom.abstract_value.clone())
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
            PrimitiveOp::Rematerialize(Box::new(remat_op)),
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
fn wrap_program_in_rematerialize<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
>(
    program: &Program<ArrayType, V, Vec<V>, Vec<V>>,
) -> Result<Program<ArrayType, V, Vec<V>, Vec<V>>, TraceError>
where
    V::ParameterStructure: Clone + PartialEq,
{
    let graph = program.graph();
    let representative_inputs = graph.representative_input_values()?;
    let input_types: Vec<_> = graph
        .input_atoms()
        .iter()
        .map(|&atom_id| {
            graph
                .atom(atom_id)
                .ok_or(TraceError::UnboundAtomId { id: atom_id })
                .map(|atom| atom.abstract_value.clone())
        })
        .collect::<Result<_, _>>()?;
    let output_types: Vec<_> = graph
        .outputs()
        .iter()
        .map(|&atom_id| {
            graph
                .atom(atom_id)
                .ok_or(TraceError::UnboundAtomId { id: atom_id })
                .map(|atom| atom.abstract_value.clone())
        })
        .collect::<Result<_, _>>()?;

    let body = FlatTracedRematerialize::from_parts(
        input_types.clone(),
        output_types.clone(),
        CompiledFunction::from_program(program.clone()),
    );
    let remat_op = RematerializeOp::new(body);

    let mut outer_builder: ProgramBuilder<V> = ProgramBuilder::new();
    let outer_inputs: Vec<AtomId> = graph
        .input_atoms()
        .iter()
        .zip(representative_inputs.iter())
        .map(|(&atom_id, representative_input)| {
            let atom = graph.atom(atom_id).expect("input atom should exist");
            outer_builder.add_input_abstract(atom.abstract_value.clone(), representative_input.clone())
        })
        .collect();

    let outer_outputs = outer_builder.add_equation_prevalidated(
        PrimitiveOp::Rematerialize(Box::new(remat_op)),
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
fn build_segment_sub_program<
    V: Traceable<ArrayType>
        + Parameterized<V, ParameterStructure = Placeholder>
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + ReshapeOps,
>(
    graph: &Graph<ProgramOpRef<V>, ArrayType, V, Vec<V>, Vec<V>>,
    representative_values: &[V],
    segment_equations: &[Equation<ProgramOpRef<V>>],
    boundary_input_atoms: &[AtomId],
    boundary_output_atoms: &[AtomId],
) -> Result<Program<ArrayType, V, Vec<V>, Vec<V>>, TraceError>
where
    V::ParameterStructure: Clone + PartialEq,
{
    let mut sub_builder: ProgramBuilder<V> = ProgramBuilder::new();

    // Map from original atom IDs to sub-program atom IDs.
    let mut sub_atom_mapping: std::collections::HashMap<AtomId, AtomId> = std::collections::HashMap::new();

    // Register boundary inputs as sub-program inputs.
    for &input_atom in boundary_input_atoms {
        let atom = graph.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
        let sub_atom =
            sub_builder.add_input_abstract(atom.abstract_value.clone(), representative_values[input_atom].clone());
        sub_atom_mapping.insert(input_atom, sub_atom);
    }

    // Register constants used by equations in this segment.
    for equation in segment_equations {
        for &input_atom in &equation.inputs {
            if sub_atom_mapping.contains_key(&input_atom) {
                continue;
            }
            let atom = graph.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
            if matches!(atom.source, AtomSource::Constant) {
                let sub_atom = sub_builder.add_constant(atom.constant_value().cloned().ok_or(
                    TraceError::InternalInvariantViolation("staged graph constant atom did not retain a literal value"),
                )?);
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
                    .map(|atom| atom.abstract_value.clone())
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
            CustomPrimitive, DifferentiableOp, FloatExt, GraphBuilder, InterpretableOp, LinearOp, LinearPrimitiveOp,
            Op, test_support,
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

    impl LinearOp<ArrayType, f64> for PanicReplayOp {
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

    impl DifferentiableOp<ArrayType, f64, LinearTerm<ArrayType, f64>> for PanicReplayOp {
        fn jvp(
            &self,
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

        let pushforward = linearize_program(&program).unwrap();
        approx_eq(pushforward.call(2.5f64).unwrap(), 2.5);
    }

    #[test]
    fn transpose_linear_program_does_not_replay_the_forward_linear_graph_to_recover_representatives() {
        let primitive =
            LinearPrimitiveOp::custom(CustomPrimitive::<ArrayType, f64>::new(PanicReplayOp).with_transpose_rule(PanicReplayOp))
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
        let (_, pushforward): (f64, LinearProgram<ArrayType, f64, f64, f64>) = linearize(quadratic_plus_sin, 2.0f64).unwrap();

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
        let compiled = compile_grad(quadratic_plus_sin, 2.0f64).unwrap();

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
        let compiled = compile_grad(bilinear_sin, (2.0f64, 3.0f64)).unwrap();

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
        let compiled_plain = compile_grad(quadratic_plus_sin, 2.0f64).unwrap();
        let compiled_save_all =
            compile_grad_with_policy(quadratic_plus_sin, 2.0f64, RematerializationPolicy::SaveAll).unwrap();

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
        let compiled =
            compile_grad_with_policy(quadratic_plus_sin, 2.0f64, RematerializationPolicy::RecomputeAll).unwrap();

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
        let compiled_plain = compile_grad(quadratic_plus_sin, 2.0f64).unwrap();
        let compiled_recompute =
            compile_grad_with_policy(quadratic_plus_sin, 2.0f64, RematerializationPolicy::RecomputeAll).unwrap();

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
        let compiled = compile_grad_with_policy(
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
        let compiled = compile_grad_with_policy(
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
        let compiled_plain = compile_grad(quadratic_plus_sin, 2.0f64).unwrap();
        let compiled_checkpoint = compile_grad_with_policy(
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
        let compiled_save_all =
            compile_grad_with_policy(quadratic_plus_sin, 2.0f64, RematerializationPolicy::SaveAll).unwrap();
        let compiled_checkpoint = compile_grad_with_policy(
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
        let compiled = compile_grad_with_policy(
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
