//! Shared graph representation used by staged transforms.
//!
//! `Graph<O, V, Input, Output>` stores a linear sequence of equations over an open set of operation objects `O`.
//! This common representation is reused for JIT graphs and for linear programs produced during differentiation.

use std::{collections::HashMap, fmt::Display, marker::PhantomData};

use crate::{
    parameters::Parameterized,
    tracing_v2::{InterpretableOp, Op, TraceError, TraceValue, Zero},
    types::{ArrayType, Typed},
};

/// Identifier for an atom within a staged graph.
pub type AtomId = usize;

/// Origin of a staged atom.
#[derive(Clone, Debug)]
pub enum AtomSource {
    /// Atom introduced as a graph input.
    Input,
    /// Atom introduced as a literal constant.
    Constant,
    /// Atom produced by evaluating an equation.
    Derived,
}

/// Staged atom carrying abstract metadata and provenance.
#[derive(Clone, Debug)]
pub struct Atom<V: TraceValue> {
    /// Array type used for validation and shape propagation.
    pub abstract_value: ArrayType,
    /// Stored concrete value when this atom semantically owns one.
    ///
    /// Inputs retain one exemplar so later transforms can recover representative primal values without
    /// requiring the caller to re-supply them. Constants retain their literal value because interpreter
    /// replay and MLIR lowering need it. Derived atoms do not retain concrete payloads in finalized
    /// staged graphs.
    stored_value: Option<V>,
    /// The way this atom entered the graph.
    pub source: AtomSource,
}

impl<V: TraceValue> Atom<V> {
    #[inline]
    fn input(abstract_value: ArrayType, exemplar_value: V) -> Self {
        Self { abstract_value, stored_value: Some(exemplar_value), source: AtomSource::Input }
    }

    #[inline]
    fn constant(abstract_value: ArrayType, value: V) -> Self {
        Self { abstract_value, stored_value: Some(value), source: AtomSource::Constant }
    }

    #[inline]
    fn derived(abstract_value: ArrayType, stored_value: Option<V>) -> Self {
        Self { abstract_value, stored_value, source: AtomSource::Derived }
    }

    #[inline]
    fn into_staged(mut self) -> Self {
        if matches!(self.source, AtomSource::Derived) {
            self.stored_value = None;
        }
        self
    }

    /// Returns the stored concrete value, if this atom retains one.
    #[inline]
    pub fn stored_value(&self) -> Option<&V> {
        self.stored_value.as_ref()
    }

    /// Returns the exemplar value retained for an input atom.
    #[inline]
    pub fn input_exemplar(&self) -> Option<&V> {
        if matches!(self.source, AtomSource::Input) { self.stored_value() } else { None }
    }

    /// Returns the literal value retained for a constant atom.
    #[inline]
    pub fn constant_value(&self) -> Option<&V> {
        if matches!(self.source, AtomSource::Constant) { self.stored_value() } else { None }
    }
}

/// Single equation in a staged graph.
#[derive(Clone, Debug)]
pub struct Equation<O> {
    /// Operation applied by this equation.
    pub op: O,
    /// Input atoms consumed by the equation.
    pub inputs: Vec<AtomId>,
    /// Output atoms produced by the equation.
    pub outputs: Vec<AtomId>,
}

/// Builder for staged graphs.
#[derive(Clone, Debug)]
pub struct GraphBuilder<O: Clone, V: TraceValue> {
    atoms: Vec<Atom<V>>,
    input_atoms: Vec<AtomId>,
    equations: Vec<Equation<O>>,
}

impl<O: Clone, V: TraceValue> GraphBuilder<O, V> {
    /// Creates an empty builder.
    #[inline]
    pub fn new() -> Self {
        Self { atoms: Vec::new(), input_atoms: Vec::new(), equations: Vec::new() }
    }

    /// Returns the number of atoms allocated so far.
    /// Returns the atom with the provided identifier.
    #[inline]
    pub fn atom(&self, id: AtomId) -> Option<&Atom<V>> {
        self.atoms.get(id)
    }

    /// Adds a new input atom with the supplied abstract value.
    #[inline]
    pub fn add_input_abstract(&mut self, abstract_value: ArrayType, example_value: V) -> AtomId {
        let id = self.atoms.len();
        self.atoms.push(Atom::input(abstract_value, example_value));
        self.input_atoms.push(id);
        id
    }

    /// Adds a new input atom from abstract metadata alone, synthesizing one zero witness internally.
    #[inline]
    pub fn add_input_abstract_zero(&mut self, abstract_value: ArrayType) -> AtomId
    where
        ArrayType: Zero<ArrayType, V>,
    {
        let example_value = abstract_value.zero();
        self.add_input_abstract(abstract_value, example_value)
    }

    /// Adds a new input atom using the abstract value of `example`.
    #[inline]
    pub fn add_input(&mut self, example: &V) -> AtomId {
        self.add_input_abstract(<V as Typed<ArrayType>>::tpe(example), example.clone())
    }

    /// Adds a constant atom to the graph.
    #[inline]
    pub fn add_constant(&mut self, value: V) -> AtomId {
        let id = self.atoms.len();
        self.atoms.push(Atom::constant(<V as Typed<ArrayType>>::tpe(&value), value));
        id
    }

    /// Adds a staged equation without running abstract or concrete evaluation.
    ///
    /// This is intended for linear program construction where the output types are already known.
    pub fn add_equation_prevalidated(
        &mut self,
        op: O,
        inputs: Vec<AtomId>,
        output_abstracts: Vec<ArrayType>,
    ) -> Vec<AtomId> {
        let outputs = output_abstracts
            .into_iter()
            .map(|abstract_value| {
                let id = self.atoms.len();
                self.atoms.push(Atom::derived(abstract_value, None));
                id
            })
            .collect::<Vec<_>>();
        self.equations.push(Equation { op, inputs, outputs: outputs.clone() });
        outputs
    }

    /// Returns the number of equations added so far.
    #[inline]
    pub fn equation_count(&self) -> usize {
        self.equations.len()
    }

    /// Adds a staged equation using pre-computed output values, performing abstract-eval validation,
    /// algebraic identity elimination, and constant folding.
    ///
    /// Unlike [`add_equation`](Self::add_equation) this method does **not** call [`InterpretableOp::eval`] —
    /// the caller supplies the concrete output values directly. Use this when the caller has already
    /// computed the outputs (e.g., inside [`JitTracer`](crate::tracing_v2::JitTracer) staging
    /// methods).
    pub fn add_equation_with_output_values(
        &mut self,
        op: O,
        inputs: Vec<AtomId>,
        output_values: Vec<V>,
    ) -> Result<Vec<AtomId>, TraceError>
    where
        O: Op,
    {
        let input_abstracts = inputs
            .iter()
            .map(|input| {
                self.atom(*input)
                    .map(|atom| atom.abstract_value.clone())
                    .ok_or(TraceError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_abstracts = op.abstract_eval(input_abstracts.as_slice())?;

        // Algebraic identity elimination: eliminate trivial ops like scale-by-1, add-by-0, mul-by-1.
        let is_zero =
            |id: usize| self.atom(id).map_or(false, |atom| atom.constant_value().is_some_and(is_identity_zero));
        let is_one = |id: usize| self.atom(id).map_or(false, |atom| atom.constant_value().is_some_and(is_identity_one));
        if let Some(simplified) = op.try_simplify(&inputs, &is_zero, &is_one) {
            return Ok(simplified);
        }

        let all_constant = inputs
            .iter()
            .all(|input| self.atom(*input).map_or(false, |atom| matches!(atom.source, AtomSource::Constant)));

        let outputs = output_abstracts
            .into_iter()
            .zip(output_values)
            .map(|(abstract_value, output_value)| {
                let id = self.atoms.len();
                self.atoms.push(if all_constant {
                    Atom::constant(abstract_value, output_value)
                } else {
                    Atom::derived(abstract_value, Some(output_value))
                });
                id
            })
            .collect::<Vec<_>>();

        if !all_constant {
            self.equations.push(Equation { op, inputs, outputs: outputs.clone() });
        }
        Ok(outputs)
    }

    /// Adds a staged equation, validating its inputs through abstract evaluation first.
    ///
    /// When every input atom has [`AtomSource::Constant`] as its source, the operation is folded at
    /// graph-construction time: `abstract_eval` and `eval` are still executed for validation, but the
    /// output atoms are recorded as constants and no equation is added to the graph.
    pub fn add_equation(&mut self, op: O, inputs: Vec<AtomId>) -> Result<Vec<AtomId>, TraceError>
    where
        O: InterpretableOp<V>,
    {
        let input_examples = inputs
            .iter()
            .map(|input| {
                self.atom(*input)
                    .and_then(|atom| atom.stored_value().cloned())
                    .ok_or(TraceError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_values = op.interpret(input_examples.as_slice())?;
        self.add_equation_with_output_values(op, inputs, output_values)
    }

    /// Finalizes the builder into a graph with the given input/output structures.
    pub fn build<Input, Output>(
        self,
        outputs: Vec<AtomId>,
        input_structure: Input::ParameterStructure,
        output_structure: Output::ParameterStructure,
    ) -> Graph<O, V, Input, Output>
    where
        Input: Parameterized<V>,
        Output: Parameterized<V>,
    {
        Graph {
            atoms: self.atoms.into_iter().map(Atom::into_staged).collect(),
            input_atoms: self.input_atoms,
            equations: self.equations,
            outputs,
            input_structure,
            output_structure,
            marker: PhantomData,
        }
    }
}

impl<O: Clone, V: TraceValue> Default for GraphBuilder<O, V> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Algebraic identity elimination helpers
// ---------------------------------------------------------------------------

/// Checks if a value is a constant zero through [`TraceValue::is_zero`].
pub(crate) fn is_identity_zero<V: TraceValue>(value: &V) -> bool {
    value.is_zero()
}

/// Checks if a value is a constant one through [`TraceValue::is_one`].
pub(crate) fn is_identity_one<V: TraceValue>(value: &V) -> bool {
    value.is_one()
}

/// Executable staged graph over an open operation set.
pub struct Graph<O: Clone, V: TraceValue, Input: Parameterized<V>, Output: Parameterized<V>> {
    atoms: Vec<Atom<V>>,
    input_atoms: Vec<AtomId>,
    equations: Vec<Equation<O>>,
    outputs: Vec<AtomId>,
    input_structure: Input::ParameterStructure,
    output_structure: Output::ParameterStructure,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<
    O: Clone,
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> Clone for Graph<O, V, Input, Output>
{
    fn clone(&self) -> Self {
        Self {
            atoms: self.atoms.clone(),
            input_atoms: self.input_atoms.clone(),
            equations: self.equations.clone(),
            outputs: self.outputs.clone(),
            input_structure: self.input_structure.clone(),
            output_structure: self.output_structure.clone(),
            marker: PhantomData,
        }
    }
}

impl<O: Clone, V: TraceValue, Input: Parameterized<V>, Output: Parameterized<V>> Graph<O, V, Input, Output> {
    /// Returns the number of atoms in the graph.
    #[inline]
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Returns the atom with the provided identifier.
    #[inline]
    pub fn atom(&self, id: AtomId) -> Option<&Atom<V>> {
        self.atoms.get(id)
    }

    /// Returns an iterator over all atoms in the graph, yielding `(atom_id, &Atom<V>)` pairs.
    #[inline]
    pub fn atoms_iter(&self) -> impl Iterator<Item = (AtomId, &Atom<V>)> {
        self.atoms.iter().enumerate()
    }

    /// Returns the graph input atoms in parameter order.
    #[inline]
    pub fn input_atoms(&self) -> &[AtomId] {
        self.input_atoms.as_slice()
    }

    /// Returns the equations in execution order.
    #[inline]
    pub fn equations(&self) -> &[Equation<O>] {
        self.equations.as_slice()
    }

    /// Returns the output atoms in parameter order.
    #[inline]
    pub fn outputs(&self) -> &[AtomId] {
        self.outputs.as_slice()
    }

    /// Returns the expected input parameter structure.
    #[inline]
    pub fn input_structure(&self) -> &Input::ParameterStructure {
        &self.input_structure
    }

    /// Returns the output parameter structure.
    #[inline]
    pub fn output_structure(&self) -> &Output::ParameterStructure {
        &self.output_structure
    }

    /// Returns the representative concrete inputs retained by this graph.
    pub fn representative_input_values(&self) -> Result<Vec<V>, TraceError> {
        self.input_atoms
            .iter()
            .copied()
            .map(|atom_id| {
                self.atom(atom_id).and_then(|atom| atom.input_exemplar().cloned()).ok_or(
                    TraceError::InternalInvariantViolation("staged graph input atom did not retain an exemplar value"),
                )
            })
            .collect()
    }

    /// Evaluates every atom in the graph on the supplied flat input values.
    pub fn evaluate_atom_values(&self, input_values: Vec<V>) -> Result<Vec<V>, TraceError>
    where
        O: InterpretableOp<V>,
    {
        if input_values.len() != self.input_atoms.len() {
            return Err(TraceError::InvalidInputCount { expected: self.input_atoms.len(), got: input_values.len() });
        }

        let mut values = vec![None; self.atoms.len()];
        for (atom, value) in self.input_atoms.iter().copied().zip(input_values) {
            values[atom] = Some(value);
        }

        for (atom_id, atom) in self.atoms.iter().enumerate() {
            if let Some(value) = atom.constant_value() {
                values[atom_id] = Some(value.clone());
            }
        }

        for equation in &self.equations {
            let inputs = equation
                .inputs
                .iter()
                .map(|input| values[*input].clone().ok_or(TraceError::UnboundAtomId { id: *input }))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = equation.op.interpret(inputs.as_slice())?;
            if outputs.len() != equation.outputs.len() {
                return Err(TraceError::InvalidOutputCount { expected: equation.outputs.len(), got: outputs.len() });
            }

            for (atom, value) in equation.outputs.iter().copied().zip(outputs) {
                values[atom] = Some(value);
            }
        }

        values
            .into_iter()
            .enumerate()
            .map(|(atom_id, value)| value.ok_or(TraceError::UnboundAtomId { id: atom_id }))
            .collect()
    }

    /// Evaluates every atom in the graph on its retained representative input exemplars.
    pub fn representative_atom_values(&self) -> Result<Vec<V>, TraceError>
    where
        O: InterpretableOp<V>,
    {
        self.evaluate_atom_values(self.representative_input_values()?)
    }

    /// Clones this graph while replacing only the typed input/output structures.
    pub fn clone_with_structures<NewInput, NewOutput>(
        &self,
        input_structure: NewInput::ParameterStructure,
        output_structure: NewOutput::ParameterStructure,
    ) -> Graph<O, V, NewInput, NewOutput>
    where
        NewInput: Parameterized<V>,
        NewOutput: Parameterized<V>,
    {
        Graph {
            atoms: self.atoms.clone(),
            input_atoms: self.input_atoms.clone(),
            equations: self.equations.clone(),
            outputs: self.outputs.clone(),
            input_structure,
            output_structure,
            marker: PhantomData,
        }
    }

    /// Interprets the staged graph on concrete input values.
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        O: InterpretableOp<V>,
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        if input.parameter_structure() != self.input_structure {
            return Err(TraceError::MismatchedParameterStructure);
        }

        let values = self.evaluate_atom_values(input.into_parameters().collect::<Vec<_>>())?;
        let outputs = self.outputs.iter().map(|output| values[*output].clone()).collect::<Vec<_>>();
        Ok(Output::from_parameters(self.output_structure.clone(), outputs)?)
    }

    /// Eliminates dead constants and equations that do not contribute to the graph outputs.
    pub fn simplify(&self) -> Result<Self, TraceError>
    where
        O: Op,
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        fn mark_live<O: Clone, V: TraceValue, Input: Parameterized<V>, Output: Parameterized<V>>(
            graph: &Graph<O, V, Input, Output>,
            atom_id: usize,
            live_atoms: &mut [bool],
            live_equations: &mut [bool],
            equation_by_output: &[Option<usize>],
        ) {
            if live_atoms[atom_id] {
                return;
            }
            live_atoms[atom_id] = true;
            if let Some(equation_index) = equation_by_output[atom_id] {
                if live_equations[equation_index] {
                    return;
                }
                live_equations[equation_index] = true;
                let equation = &graph.equations[equation_index];
                for input in equation.inputs.iter().copied() {
                    mark_live(graph, input, live_atoms, live_equations, equation_by_output);
                }
            }
        }

        fn remap_atom<O, V, Input, Output>(
            atom_id: usize,
            graph: &Graph<O, V, Input, Output>,
            builder: &mut GraphBuilder<O, V>,
            atom_mapping: &mut HashMap<usize, usize>,
            live_equations: &[bool],
            equation_by_output: &[Option<usize>],
        ) -> Result<usize, TraceError>
        where
            O: Clone + Op,
            V: TraceValue,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        {
            if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
                return Ok(*mapped_atom);
            }

            let atom = graph.atom(atom_id).ok_or(TraceError::UnboundAtomId { id: atom_id })?;
            let mapped_atom = match atom.source {
                AtomSource::Input => builder.add_input_abstract(
                    atom.abstract_value.clone(),
                    atom.input_exemplar().cloned().ok_or(TraceError::InternalInvariantViolation(
                        "staged graph input atom did not retain an exemplar value",
                    ))?,
                ),
                AtomSource::Constant => builder.add_constant(atom.constant_value().cloned().ok_or(
                    TraceError::InternalInvariantViolation("staged graph constant atom did not retain a literal value"),
                )?),
                AtomSource::Derived => {
                    let equation_index = equation_by_output[atom_id]
                        .ok_or(TraceError::InternalInvariantViolation("derived atom had no owning equation"))?;
                    if !live_equations[equation_index] {
                        return Err(TraceError::InternalInvariantViolation(
                            "attempted to remap a dead derived atom during graph simplification",
                        ));
                    }
                    let equation = &graph.equations[equation_index];
                    let remapped_inputs = equation
                        .inputs
                        .iter()
                        .copied()
                        .map(|input| {
                            remap_atom(input, graph, builder, atom_mapping, live_equations, equation_by_output)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let input_abstracts = equation
                        .outputs
                        .iter()
                        .map(|output| graph.atom(*output).expect("output atom should exist").abstract_value.clone())
                        .collect::<Vec<_>>();
                    let remapped_outputs =
                        builder.add_equation_prevalidated(equation.op.clone(), remapped_inputs, input_abstracts);
                    for (old_output, new_output) in
                        equation.outputs.iter().copied().zip(remapped_outputs.iter().copied())
                    {
                        atom_mapping.insert(old_output, new_output);
                    }
                    *atom_mapping
                        .get(&atom_id)
                        .ok_or(TraceError::InternalInvariantViolation("failed to record remapped graph outputs"))?
                }
            };
            atom_mapping.entry(atom_id).or_insert(mapped_atom);
            Ok(mapped_atom)
        }

        let mut equation_by_output = vec![None; self.atom_count()];
        for (equation_index, equation) in self.equations.iter().enumerate() {
            for output in equation.outputs.iter().copied() {
                equation_by_output[output] = Some(equation_index);
            }
        }

        let mut live_atoms = vec![false; self.atom_count()];
        let mut live_equations = vec![false; self.equations.len()];
        for output in self.outputs.iter().copied() {
            mark_live(
                self,
                output,
                live_atoms.as_mut_slice(),
                live_equations.as_mut_slice(),
                equation_by_output.as_slice(),
            );
        }

        let mut builder = GraphBuilder::<O, V>::new();
        let mut atom_mapping = HashMap::new();
        for input_atom in self.input_atoms.iter().copied() {
            let input = self.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
            let mapped = builder.add_input_abstract(
                input.abstract_value.clone(),
                input.input_exemplar().cloned().ok_or(TraceError::InternalInvariantViolation(
                    "staged graph input atom did not retain an exemplar value",
                ))?,
            );
            atom_mapping.insert(input_atom, mapped);
        }

        let outputs = self
            .outputs
            .iter()
            .copied()
            .map(|output| {
                remap_atom(
                    output,
                    self,
                    &mut builder,
                    &mut atom_mapping,
                    live_equations.as_slice(),
                    equation_by_output.as_slice(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(builder.build::<Input, Output>(outputs, self.input_structure.clone(), self.output_structure.clone()))
    }
}

impl<O: Clone + Display, V: TraceValue, Input: Parameterized<V>, Output: Parameterized<V>> Display
    for Graph<O, V, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let format_atom = |id: AtomId| format!("%{id}");
        let format_typed_atom = |id: AtomId| format!("%{id}:{}", self.atoms[id].abstract_value);

        let inputs = self.input_atoms.iter().map(|input| format_typed_atom(*input)).collect::<Vec<_>>().join(", ");
        writeln!(formatter, "lambda {inputs} .")?;

        let mut equation_by_first_output = vec![None; self.atoms.len()];
        for (index, equation) in self.equations.iter().enumerate() {
            if let Some(first_output) = equation.outputs.first() {
                equation_by_first_output[*first_output] = Some(index);
            }
        }

        let mut binding_count = 0usize;
        for (atom_id, atom) in self.atoms.iter().enumerate() {
            match &atom.source {
                AtomSource::Input => {}
                AtomSource::Constant => {
                    let prefix = if binding_count == 0 { "let" } else { "   " };
                    writeln!(formatter, "{prefix} {} = const", format_typed_atom(atom_id))?;
                    binding_count += 1;
                }
                AtomSource::Derived => {
                    let Some(equation_index) = equation_by_first_output[atom_id] else {
                        continue;
                    };
                    let equation = &self.equations[equation_index];
                    let outputs =
                        equation.outputs.iter().map(|output| format_typed_atom(*output)).collect::<Vec<_>>().join(", ");
                    let inputs = equation.inputs.iter().map(|input| format_atom(*input)).collect::<Vec<_>>().join(" ");
                    let prefix = if binding_count == 0 { "let" } else { "   " };
                    if inputs.is_empty() {
                        writeln!(formatter, "{prefix} {outputs} = {}", equation.op)?;
                    } else {
                        writeln!(formatter, "{prefix} {outputs} = {} {inputs}", equation.op)?;
                    }
                    binding_count += 1;
                }
            }
        }

        let outputs = self.outputs.iter().map(|output| format_atom(*output)).collect::<Vec<_>>().join(", ");
        write!(formatter, "in ({outputs})")
    }
}

#[cfg(test)]
mod tests {
    use std::ops::{Add, Mul, Neg};

    use indoc::indoc;
    use ryft_macros::Parameter;

    use crate::{
        parameters::{Parameter, Placeholder},
        tracing_v2::{
            ConcreteTraceValue, FloatExt, MatrixOps, One, OneLike, TraceError, Zero, ZeroLike, ops::PrimitiveOp,
            test_support,
        },
        types::{ArrayType, DataType, Shape, Typed},
    };

    use super::*;

    #[test]
    fn graph_builder_tracks_atom_sources_and_executes() {
        let mut builder = GraphBuilder::<PrimitiveOp<f64>, f64>::new();
        let x = builder.add_input(&2.0f64);
        let y = builder.add_input(&3.0f64);
        let two = builder.add_constant(2.0f64);
        let scaled_x = builder.add_equation(PrimitiveOp::Scale { factor: 2.0 }, vec![x]).unwrap()[0];
        let sum = builder.add_equation(PrimitiveOp::Add, vec![scaled_x, y]).unwrap()[0];
        let graph = builder.build::<(f64, f64), f64>(vec![sum], (Placeholder, Placeholder), Placeholder);

        assert!(matches!(graph.atom(x).unwrap().source, AtomSource::Input));
        assert!(matches!(graph.atom(two).unwrap().source, AtomSource::Constant));
        assert_eq!(graph.call((2.0, 3.0)).unwrap(), 7.0);
        assert_eq!(
            graph.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = const
                    %3:f64[] = scale %0
                    %4:f64[] = add %3 %1
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn graph_display_uses_typed_jaxpr_like_rendering() {
        let mut builder = GraphBuilder::<PrimitiveOp<f64>, f64>::new();
        let x = builder.add_input(&1.0f64);
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_equation(PrimitiveOp::Add, vec![x, three]).unwrap()[0];
        let graph = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);

        assert_eq!(
            graph.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const
                    %2:f64[] = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn graph_builder_rejects_unbound_inputs() {
        let mut builder = GraphBuilder::<PrimitiveOp<f64>, f64>::new();
        let result = builder.add_equation(PrimitiveOp::Add, vec![42, 99]);
        assert!(matches!(result, Err(TraceError::UnboundAtomId { id: 42 })));
        test_support::assert_reference_graph_rendering();
    }

    #[test]
    fn test_constant_folding_eliminates_equations() {
        let mut builder = GraphBuilder::<PrimitiveOp<f64>, f64>::new();
        let a = builder.add_constant(2.0f64);
        let b = builder.add_constant(3.0f64);

        // Adding two constants should fold: no equation, output is constant.
        let folded = builder.add_equation(PrimitiveOp::Add, vec![a, b]).unwrap();
        assert_eq!(folded.len(), 1);
        assert!(matches!(builder.atom(folded[0]).unwrap().source, AtomSource::Constant));
        assert_eq!(builder.equation_count(), 0);

        // Introduce a non-constant input and combine with the folded constant.
        let x = builder.add_input(&10.0f64);
        let result = builder.add_equation(PrimitiveOp::Mul, vec![folded[0], x]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(builder.atom(result[0]).unwrap().source, AtomSource::Derived));
        assert_eq!(builder.equation_count(), 1);

        // Build the graph and verify only the non-folded equation survived.
        let graph = builder.build::<f64, f64>(vec![result[0]], Placeholder, Placeholder);
        assert_eq!(graph.equations().len(), 1);
        assert_eq!(
            graph.to_string(),
            indoc! {"
                lambda %3:f64[] .
                let %0:f64[] = const
                    %1:f64[] = const
                    %2:f64[] = const
                    %4:f64[] = mul %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_constant_folding_graph_call_produces_correct_results() {
        let mut builder = GraphBuilder::<PrimitiveOp<f64>, f64>::new();
        let a = builder.add_constant(2.0f64);
        let b = builder.add_constant(3.0f64);
        let folded_sum = builder.add_equation(PrimitiveOp::Add, vec![a, b]).unwrap()[0];

        let x = builder.add_input(&10.0f64);
        let product = builder.add_equation(PrimitiveOp::Mul, vec![folded_sum, x]).unwrap()[0];
        let graph = builder.build::<f64, f64>(vec![product], Placeholder, Placeholder);

        // folded_sum = 2.0 + 3.0 = 5.0, product = 5.0 * input
        assert_eq!(graph.call(10.0).unwrap(), 50.0);
        assert_eq!(graph.call(0.5).unwrap(), 2.5);
        assert_eq!(graph.call(0.0).unwrap(), 0.0);
    }

    #[test]
    fn built_graph_drops_derived_stored_values_but_reconstructs_representatives() {
        let mut builder = GraphBuilder::<PrimitiveOp<f64>, f64>::new();
        let x = builder.add_input(&2.0f64);
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_equation(PrimitiveOp::Add, vec![x, three]).unwrap()[0];

        assert_eq!(builder.atom(x).unwrap().input_exemplar(), Some(&2.0));
        assert_eq!(builder.atom(three).unwrap().constant_value(), Some(&3.0));
        assert_eq!(builder.atom(sum).unwrap().stored_value(), Some(&5.0));

        let graph = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);
        assert_eq!(graph.atom(x).unwrap().input_exemplar(), Some(&2.0));
        assert_eq!(graph.atom(three).unwrap().constant_value(), Some(&3.0));
        assert_eq!(graph.atom(sum).unwrap().stored_value(), None);
        assert_eq!(graph.representative_atom_values().unwrap(), vec![2.0, 3.0, 5.0]);
        assert_eq!(graph.call(4.0).unwrap(), 7.0);
    }

    #[test]
    fn custom_identity_values_participate_in_algebraic_simplification() {
        #[derive(Clone, Debug, PartialEq, Parameter)]
        struct TestIdentityValue {
            r#type: ArrayType,
            value: f64,
        }

        impl TestIdentityValue {
            fn scalar(value: f64) -> Self {
                Self { r#type: ArrayType::scalar(DataType::F64), value }
            }
        }

        impl Typed<ArrayType> for TestIdentityValue {
            fn tpe(&self) -> ArrayType {
                self.r#type.clone()
            }
        }

        impl TraceValue for TestIdentityValue {
            fn is_zero(&self) -> bool {
                self.value == 0.0
            }

            fn is_one(&self) -> bool {
                self.value == 1.0
            }
        }

        impl ConcreteTraceValue for TestIdentityValue {}

        impl Add for TestIdentityValue {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                Self { r#type: self.r#type, value: self.value + rhs.value }
            }
        }

        impl Mul for TestIdentityValue {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                Self { r#type: self.r#type, value: self.value * rhs.value }
            }
        }

        impl Neg for TestIdentityValue {
            type Output = Self;

            fn neg(self) -> Self::Output {
                Self { r#type: self.r#type, value: -self.value }
            }
        }

        impl FloatExt for TestIdentityValue {
            fn sin(self) -> Self {
                self
            }

            fn cos(self) -> Self {
                self
            }
        }

        impl ZeroLike for TestIdentityValue {
            fn zero_like(&self) -> Self {
                Self::scalar(0.0)
            }
        }

        impl Zero<ArrayType, TestIdentityValue> for ArrayType {
            fn zero(&self) -> TestIdentityValue {
                TestIdentityValue { r#type: self.clone(), value: 0.0 }
            }
        }

        impl OneLike for TestIdentityValue {
            fn one_like(&self) -> Self {
                Self::scalar(1.0)
            }
        }

        impl One<ArrayType, TestIdentityValue> for ArrayType {
            fn one(&self) -> TestIdentityValue {
                TestIdentityValue { r#type: self.clone(), value: 1.0 }
            }
        }

        impl MatrixOps for TestIdentityValue {
            fn matmul(self, rhs: Self) -> Self {
                Self { r#type: self.r#type, value: self.value * rhs.value }
            }

            fn transpose_matrix(self) -> Self {
                self
            }
        }

        impl crate::tracing_v2::operations::reshape::ReshapeOps for TestIdentityValue {
            fn reshape(self, target_shape: Shape) -> Result<Self, TraceError> {
                Ok(Self { r#type: ArrayType::new(DataType::F64, target_shape, None, None).unwrap(), value: self.value })
            }
        }

        let mut builder = GraphBuilder::<PrimitiveOp<TestIdentityValue>, TestIdentityValue>::new();
        let x = builder.add_input(&TestIdentityValue::scalar(5.0));
        let zero = builder.add_constant(TestIdentityValue::scalar(0.0));

        let simplified_add = builder.add_equation(PrimitiveOp::Add, vec![x, zero]).unwrap();
        assert_eq!(simplified_add, vec![x]);
        assert_eq!(builder.equation_count(), 0);

        let simplified_scale = builder
            .add_equation(PrimitiveOp::Scale { factor: TestIdentityValue::scalar(1.0) }, vec![x])
            .unwrap();
        assert_eq!(simplified_scale, vec![x]);
        assert_eq!(builder.equation_count(), 0);
    }
}
