use std::{fmt::Display, marker::PhantomData};

use crate::{
    parameters::Parameterized,
    tracing_v2::{Op, TraceError, TraceValue},
};

pub type AtomId = usize;

#[derive(Clone, Debug)]
pub enum AtomSource<V> {
    Input,
    Constant(V),
    Derived,
}

#[derive(Clone, Debug)]
pub struct Atom<V>
where
    V: TraceValue,
{
    pub abstract_value: V::Abstract,
    pub source: AtomSource<V>,
}

#[derive(Clone, Debug)]
pub struct Equation<O> {
    pub op: O,
    pub inputs: Vec<AtomId>,
    pub outputs: Vec<AtomId>,
}

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
    #[inline]
    pub fn new() -> Self {
        Self { atoms: Vec::new(), input_atoms: Vec::new(), equations: Vec::new() }
    }

    #[inline]
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    #[inline]
    pub fn atom(&self, id: AtomId) -> Option<&Atom<V>> {
        self.atoms.get(id)
    }

    #[inline]
    pub fn add_input_abstract(&mut self, abstract_value: V::Abstract) -> AtomId {
        let id = self.atoms.len();
        self.atoms.push(Atom { abstract_value, source: AtomSource::Input });
        self.input_atoms.push(id);
        id
    }

    #[inline]
    pub fn add_input(&mut self, example: &V) -> AtomId {
        self.add_input_abstract(example.abstract_value())
    }

    #[inline]
    pub fn add_constant(&mut self, value: V) -> AtomId {
        let id = self.atoms.len();
        self.atoms
            .push(Atom { abstract_value: value.abstract_value(), source: AtomSource::Constant(value) });
        id
    }

    pub fn add_equation(&mut self, op: O, inputs: Vec<AtomId>) -> Result<Vec<AtomId>, TraceError> {
        let input_abstracts = inputs
            .iter()
            .map(|input| {
                self.atom(*input)
                    .map(|atom| atom.abstract_value.clone())
                    .ok_or(TraceError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = op
            .abstract_eval(input_abstracts.as_slice())?
            .into_iter()
            .map(|abstract_value| {
                let id = self.atoms.len();
                self.atoms.push(Atom { abstract_value, source: AtomSource::Derived });
                id
            })
            .collect::<Vec<_>>();
        self.equations.push(Equation { op, inputs, outputs: outputs.clone() });
        Ok(outputs)
    }

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

impl<O, V, Input, Output> Graph<O, V, Input, Output>
where
    O: Clone + Op<V>,
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    #[inline]
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    #[inline]
    pub fn atom(&self, id: AtomId) -> Option<&Atom<V>> {
        self.atoms.get(id)
    }

    #[inline]
    pub fn input_atoms(&self) -> &[AtomId] {
        self.input_atoms.as_slice()
    }

    #[inline]
    pub fn equations(&self) -> &[Equation<O>] {
        self.equations.as_slice()
    }

    #[inline]
    pub fn outputs(&self) -> &[AtomId] {
        self.outputs.as_slice()
    }

    #[inline]
    pub fn input_structure(&self) -> &Input::ParameterStructure {
        &self.input_structure
    }

    #[inline]
    pub fn output_structure(&self) -> &Output::ParameterStructure {
        &self.output_structure
    }

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
            if let AtomSource::Constant(value) = &atom.source {
                values[atom_id] = Some(value.clone());
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
}

impl<O, V, Input, Output> Display for Graph<O, V, Input, Output>
where
    O: Clone + Display + Op<V>,
    V: TraceValue,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inputs = self.input_atoms.iter().map(|input| format!("%{input}")).collect::<Vec<_>>().join(", ");
        writeln!(f, "lambda {inputs} .")?;
        for equation in &self.equations {
            let outputs = equation.outputs.iter().map(|output| format!("%{output}")).collect::<Vec<_>>().join(", ");
            let inputs = equation.inputs.iter().map(|input| format!("%{input}")).collect::<Vec<_>>().join(" ");
            writeln!(f, "  {outputs} = {} {inputs}", equation.op)?;
        }
        let outputs = self.outputs.iter().map(|output| format!("%{output}")).collect::<Vec<_>>().join(", ");
        write!(f, "in ({outputs})")
    }
}
