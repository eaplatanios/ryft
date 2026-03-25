//! Shared graph representation used by staged transforms.
//!
//! `Graph<O, V, Input, Output>` stores a linear sequence of equations over an open set of operation objects `O`.
//! This common representation is reused for JIT graphs and for linear programs produced during differentiation.

use std::{collections::HashMap, fmt::Display, marker::PhantomData};

use crate::{
    parameters::Parameterized,
    tracing_v2::{Op, TraceError, TraceValue},
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
pub struct Atom<V>
where
    V: TraceValue,
{
    /// Array type used for validation and shape propagation.
    pub abstract_value: ArrayType,
    /// Example value produced while staging this atom.
    pub example_value: V,
    /// The way this atom entered the graph.
    pub source: AtomSource,
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
pub struct GraphBuilder<O, V>
where
    O: Clone + Op<V>,
    V: TraceValue,
{
    atoms: Vec<Atom<V>>,
    input_atoms: Vec<AtomId>,
    equations: Vec<Equation<O>>,
}

impl<O, V> GraphBuilder<O, V>
where
    O: Clone + Op<V>,
    V: TraceValue,
{
    /// Creates an empty builder.
    #[inline]
    pub fn new() -> Self {
        Self { atoms: Vec::new(), input_atoms: Vec::new(), equations: Vec::new() }
    }

    /// Returns the number of atoms allocated so far.
    #[inline]
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Returns the atom with the provided identifier.
    #[inline]
    pub fn atom(&self, id: AtomId) -> Option<&Atom<V>> {
        self.atoms.get(id)
    }

    /// Adds a new input atom with the supplied abstract value.
    #[inline]
    pub fn add_input_abstract(&mut self, abstract_value: ArrayType, example_value: V) -> AtomId {
        let id = self.atoms.len();
        self.atoms.push(Atom { abstract_value, example_value, source: AtomSource::Input });
        self.input_atoms.push(id);
        id
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
        self.atoms.push(Atom {
            abstract_value: <V as Typed<ArrayType>>::tpe(&value),
            example_value: value,
            source: AtomSource::Constant,
        });
        id
    }

    /// Adds a staged equation, validating its inputs through abstract evaluation first.
    pub fn add_equation(&mut self, op: O, inputs: Vec<AtomId>) -> Result<Vec<AtomId>, TraceError> {
        let input_abstracts = inputs
            .iter()
            .map(|input| {
                self.atom(*input)
                    .map(|atom| atom.abstract_value.clone())
                    .ok_or(TraceError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_examples = inputs
            .iter()
            .map(|input| {
                self.atom(*input)
                    .map(|atom| atom.example_value.clone())
                    .ok_or(TraceError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_examples = op.eval(input_examples.as_slice())?;
        let outputs = op
            .abstract_eval(input_abstracts.as_slice())?
            .into_iter()
            .zip(output_examples)
            .map(|(abstract_value, example_value)| {
                let id = self.atoms.len();
                self.atoms.push(Atom { abstract_value, example_value, source: AtomSource::Derived });
                id
            })
            .collect::<Vec<_>>();
        self.equations.push(Equation { op, inputs, outputs: outputs.clone() });
        Ok(outputs)
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
            atoms: self.atoms,
            input_atoms: self.input_atoms,
            equations: self.equations,
            outputs,
            input_structure,
            output_structure,
            marker: PhantomData,
        }
    }
}

impl<O, V> Default for GraphBuilder<O, V>
where
    O: Clone + Op<V>,
    V: TraceValue,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Executable staged graph over an open operation set.
pub struct Graph<O, V, Input, Output>
where
    O: Clone + Op<V>,
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    atoms: Vec<Atom<V>>,
    input_atoms: Vec<AtomId>,
    equations: Vec<Equation<O>>,
    outputs: Vec<AtomId>,
    input_structure: Input::ParameterStructure,
    output_structure: Output::ParameterStructure,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<O, V, Input, Output> Clone for Graph<O, V, Input, Output>
where
    O: Clone + Op<V>,
    V: TraceValue,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
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

impl<O, V, Input, Output> Graph<O, V, Input, Output>
where
    O: Clone + Op<V>,
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
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
        Input::ParameterStructure: PartialEq,
        Output::ParameterStructure: Clone,
    {
        if input.parameter_structure() != self.input_structure {
            return Err(TraceError::MismatchedParameterStructure);
        }

        let input_values = input.into_parameters().collect::<Vec<_>>();
        if input_values.len() != self.input_atoms.len() {
            return Err(TraceError::InvalidInputCount { expected: self.input_atoms.len(), got: input_values.len() });
        }

        let mut values = vec![None; self.atoms.len()];
        for (atom, value) in self.input_atoms.iter().copied().zip(input_values) {
            values[atom] = Some(value);
        }

        for (atom_id, atom) in self.atoms.iter().enumerate() {
            if matches!(atom.source, AtomSource::Constant) {
                values[atom_id] = Some(atom.example_value.clone());
            }
        }

        for equation in &self.equations {
            let inputs = equation
                .inputs
                .iter()
                .map(|input| values[*input].clone().ok_or(TraceError::UnboundAtomId { id: *input }))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = equation.op.eval(inputs.as_slice())?;
            if outputs.len() != equation.outputs.len() {
                return Err(TraceError::InvalidOutputCount { expected: equation.outputs.len(), got: outputs.len() });
            }

            for (atom, value) in equation.outputs.iter().copied().zip(outputs) {
                values[atom] = Some(value);
            }
        }

        let outputs = self
            .outputs
            .iter()
            .map(|output| values[*output].clone().ok_or(TraceError::UnboundAtomId { id: *output }))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Output::from_parameters(self.output_structure.clone(), outputs)?)
    }

    /// Eliminates dead constants and equations that do not contribute to the graph outputs.
    pub fn simplify(&self) -> Result<Self, TraceError>
    where
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        fn mark_live<O, V, Input, Output>(
            graph: &Graph<O, V, Input, Output>,
            atom_id: usize,
            live_atoms: &mut [bool],
            live_equations: &mut [bool],
            equation_by_output: &[Option<usize>],
        ) where
            O: Clone + Op<V>,
            V: TraceValue,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        {
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
            O: Clone + Op<V>,
            V: TraceValue,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        {
            if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
                return Ok(*mapped_atom);
            }

            let atom = graph.atom(atom_id).ok_or(TraceError::UnboundAtomId { id: atom_id })?;
            let mapped_atom = match atom.source {
                AtomSource::Input => {
                    builder.add_input_abstract(atom.abstract_value.clone(), atom.example_value.clone())
                }
                AtomSource::Constant => builder.add_constant(atom.example_value.clone()),
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
                    let remapped_outputs = builder.add_equation(equation.op.clone(), remapped_inputs)?;
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
            let mapped = builder.add_input_abstract(input.abstract_value.clone(), input.example_value.clone());
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

impl<O, V, Input, Output> Display for Graph<O, V, Input, Output>
where
    O: Clone + Display + Op<V>,
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
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
    use indoc::indoc;

    use crate::{
        parameters::Placeholder,
        tracing_v2::{AddOp, ScaleOp, test_support},
    };

    use super::*;

    #[test]
    fn graph_builder_tracks_atom_sources_and_executes() {
        let mut builder = GraphBuilder::<std::sync::Arc<dyn crate::tracing_v2::Op<f64>>, f64>::new();
        let x = builder.add_input(&2.0f64);
        let y = builder.add_input(&3.0f64);
        let two = builder.add_constant(2.0f64);
        let scaled_x = builder.add_equation(std::sync::Arc::new(ScaleOp::new(2.0)), vec![x]).unwrap()[0];
        let sum = builder.add_equation(std::sync::Arc::new(AddOp), vec![scaled_x, y]).unwrap()[0];
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
        let mut builder = GraphBuilder::<std::sync::Arc<dyn crate::tracing_v2::Op<f64>>, f64>::new();
        let x = builder.add_input(&1.0f64);
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_equation(std::sync::Arc::new(AddOp), vec![x, three]).unwrap()[0];
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
        let mut builder = GraphBuilder::<std::sync::Arc<AddOp>, f64>::new();
        let result = builder.add_equation(std::sync::Arc::new(AddOp), vec![42, 99]);
        assert!(matches!(result, Err(TraceError::UnboundAtomId { id: 42 })));
        test_support::assert_reference_graph_rendering();
    }
}
