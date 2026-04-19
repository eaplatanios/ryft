//! Shared staged-program representation and default op-carrier aliases used by the tracing
//! transforms.
//!
//! `Program<T, V, Input, Output, O>` stores a linear sequence of equations over an open set of
//! operation objects `O`. This common representation is reused for JIT programs and for linear
//! programs produced during differentiation.

use std::{borrow::Cow, collections::HashMap, fmt::Display, marker::PhantomData};

use crate::{
    parameters::{Parameter, Parameterized},
    tracing_v2::{Engine, InterpretableOp, LinearPrimitiveOp, Op, PrimitiveOp, TraceError, Traceable},
    types::{ArrayType, Type, Typed},
};

/// Identifier for an atom within a staged program.
pub type AtomId = usize;

/// Staged atom carrying abstract metadata.
///
/// The variant encodes how the atom entered the program and determines which concrete state it
/// retains. See [`ProgramBuilder`] for how builder-time intermediate values for [`Atom::Derived`]
/// are kept separately during staging.
#[derive(Clone, Debug)]
pub enum Atom<T: Type, V: Typed<T>> {
    /// Literal constant folded or supplied at trace time. Constants retain their value so the
    /// interpreter and MLIR lowering can emit them.
    Constant {
        /// Literal value held by this atom.
        value: V,
    },

    /// Program input carrying only its abstract type. Any builder-time representative value is kept
    /// in the owning [`ProgramBuilder`]'s side table and is discarded when the program is finalized;
    /// later transforms recover representatives by synthesizing zeros from the retained type.
    Input {
        /// Abstract type retained for this input.
        r#type: T,
    },

    /// Atom produced by evaluating an equation. Carries only the abstract type; any eagerly
    /// evaluated intermediate value lives in the owning [`ProgramBuilder`]'s side table and is
    /// discarded when the program is finalized.
    Derived {
        /// Abstract type produced by the equation.
        r#type: T,
    },
}

impl<T: Type, V: Typed<T>> Typed<T> for Atom<T, V> {
    fn tpe(&self) -> Cow<'_, T> {
        match self {
            Self::Constant { value } => value.tpe(),
            Self::Input { r#type } | Self::Derived { r#type } => Cow::Borrowed(r#type),
        }
    }
}

/// Single equation in a staged program.
#[derive(Clone, Debug)]
pub struct Equation<O> {
    /// Operation applied by this equation.
    pub op: O,
    /// Input atoms consumed by the equation.
    pub inputs: Vec<AtomId>,
    /// Output atoms produced by the equation.
    pub outputs: Vec<AtomId>,
}

/// Builder for staged programs.
///
/// The builder keeps one entry in [`Self::intermediates`] for every atom: `Some` for [`Atom::Derived`]
/// atoms whose value has been eagerly computed during staging, `None` otherwise. These intermediate
/// values are used for on-the-fly interpretation and algebraic-identity checks but are discarded
/// when the program is finalized via [`Self::build`].
#[derive(Clone, Debug)]
pub struct ProgramBuilder<O, T: Type, V: Typed<T>> {
    atoms: Vec<Atom<T, V>>,
    intermediates: Vec<Option<V>>,
    input_atoms: Vec<AtomId>,
    equations: Vec<Equation<O>>,
}

impl<O: Clone, T: Type, V: Traceable<T>> ProgramBuilder<O, T, V> {
    /// Creates an empty builder.
    #[inline]
    pub fn new() -> Self {
        Self { atoms: Vec::new(), intermediates: Vec::new(), input_atoms: Vec::new(), equations: Vec::new() }
    }

    /// Returns the atom with the provided identifier.
    #[inline]
    pub fn atom(&self, id: AtomId) -> Option<&Atom<T, V>> {
        self.atoms.get(id)
    }

    /// Returns the concrete value associated with the provided atom, if one is available.
    ///
    /// For [`Atom::Constant`] this returns the retained value. For [`Atom::Input`] and
    /// [`Atom::Derived`] this returns the eagerly computed exemplar stored in the builder's
    /// side table, or `None` if none is available.
    #[inline]
    pub(crate) fn stored_value(&self, id: AtomId) -> Option<&V> {
        match self.atoms.get(id)? {
            Atom::Constant { value } => Some(value),
            Atom::Input { .. } | Atom::Derived { .. } => self.intermediates.get(id).and_then(Option::as_ref),
        }
    }

    /// Adds a new input atom retaining only its abstract type, without recording any exemplar in
    /// the builder's side table.
    ///
    /// Intended for program transforms that rebuild structure without needing intermediate values
    /// (for example [`Program::simplify`]). Callers that later need a representative value for this
    /// atom should obtain it from an [`Engine`](crate::tracing_v2::Engine) via
    /// [`Program::representative_input_values`].
    #[inline]
    pub fn add_input_abstract(&mut self, abstract_value: T) -> AtomId {
        let id = self.atoms.len();
        self.atoms.push(Atom::Input { r#type: abstract_value });
        self.intermediates.push(None);
        self.input_atoms.push(id);
        id
    }

    /// Adds a new input atom using the abstract type and value of `example`.
    #[inline]
    pub fn add_input(&mut self, example: &V) -> AtomId {
        let abstract_value = <V as Typed<T>>::tpe(example).into_owned();
        self.add_input_with_example(abstract_value, example.clone())
    }

    /// Adds a new input atom with the supplied abstract type and a caller-supplied exemplar value.
    #[inline]
    fn add_input_with_example(&mut self, abstract_value: T, example_value: V) -> AtomId {
        let id = self.atoms.len();
        self.atoms.push(Atom::Input { r#type: abstract_value });
        self.intermediates.push(Some(example_value));
        self.input_atoms.push(id);
        id
    }

    /// Adds a constant atom to the program.
    #[inline]
    pub fn add_constant(&mut self, value: V) -> AtomId {
        let id = self.atoms.len();
        self.atoms.push(Atom::Constant { value });
        self.intermediates.push(None);
        id
    }

    /// Adds a staged equation without running abstract or concrete evaluation.
    ///
    /// This is intended for linear program construction where the output types are already known.
    pub fn add_equation_prevalidated(&mut self, op: O, inputs: Vec<AtomId>, output_abstracts: Vec<T>) -> Vec<AtomId> {
        let outputs = output_abstracts
            .into_iter()
            .map(|r#type| {
                let id = self.atoms.len();
                self.atoms.push(Atom::Derived { r#type });
                self.intermediates.push(None);
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
        O: Op<T>,
    {
        let input_abstracts = inputs
            .iter()
            .map(|input| {
                self.atom(*input)
                    .map(|atom| atom.tpe().into_owned())
                    .ok_or(TraceError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_abstracts = op.abstract_eval(input_abstracts.as_slice())?;

        // Algebraic identity elimination: eliminate trivial ops like scale-by-1, add-by-0, mul-by-1.
        let is_zero = |id: usize| matches!(self.atom(id), Some(Atom::Constant { value }) if is_identity_zero(value));
        let is_one = |id: usize| matches!(self.atom(id), Some(Atom::Constant { value }) if is_identity_one(value));
        if let Some(simplified) = op.try_simplify(&inputs, &is_zero, &is_one) {
            return Ok(simplified);
        }

        let all_constant = inputs.iter().all(|input| matches!(self.atom(*input), Some(Atom::Constant { .. })));

        let outputs = output_abstracts
            .into_iter()
            .zip(output_values)
            .map(|(r#type, output_value)| {
                let id = self.atoms.len();
                if all_constant {
                    self.atoms.push(Atom::Constant { value: output_value });
                    self.intermediates.push(None);
                } else {
                    self.atoms.push(Atom::Derived { r#type });
                    self.intermediates.push(Some(output_value));
                }
                id
            })
            .collect::<Vec<_>>();

        if !all_constant {
            self.equations.push(Equation { op, inputs, outputs: outputs.clone() });
        }
        Ok(outputs)
    }

    /// Adds a staged equation using only abstract evaluation.
    ///
    /// This is the staging path used by type-directed tracing and any traced replay that does not
    /// have representative concrete values available for the participating atoms.
    pub fn add_equation_abstract(&mut self, op: O, inputs: Vec<AtomId>) -> Result<Vec<AtomId>, TraceError>
    where
        O: Op<T>,
    {
        let input_abstracts = inputs
            .iter()
            .map(|input| {
                self.atom(*input)
                    .map(|atom| atom.tpe().into_owned())
                    .ok_or(TraceError::UnboundAtomId { id: *input })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_abstracts = op.abstract_eval(input_abstracts.as_slice())?;

        let is_zero = |id: usize| matches!(self.atom(id), Some(Atom::Constant { value }) if is_identity_zero(value));
        let is_one = |id: usize| matches!(self.atom(id), Some(Atom::Constant { value }) if is_identity_one(value));
        if let Some(simplified) = op.try_simplify(&inputs, &is_zero, &is_one) {
            return Ok(simplified);
        }

        Ok(self.add_equation_prevalidated(op, inputs, output_abstracts))
    }

    /// Adds a staged equation, validating its inputs through abstract evaluation first.
    ///
    /// When every input atom is an [`Atom::Constant`], the operation is folded at program-construction
    /// time: `abstract_eval` and `eval` are still executed for validation, but the output atoms are
    /// recorded as constants and no equation is added to the program.
    pub fn add_equation(&mut self, op: O, inputs: Vec<AtomId>) -> Result<Vec<AtomId>, TraceError>
    where
        O: InterpretableOp<T, V>,
    {
        let input_examples = inputs
            .iter()
            .map(|input| self.stored_value(*input).cloned().ok_or(TraceError::UnboundAtomId { id: *input }))
            .collect::<Result<Vec<_>, _>>()?;
        let output_values = op.interpret(input_examples.as_slice())?;
        self.add_equation_with_output_values(op, inputs, output_values)
    }

    /// Finalizes the builder into a program with the given input/output structures. The builder's
    /// intermediate values are discarded; the resulting program retains only the atoms, equations,
    /// and input/output structure.
    pub fn build<Input, Output>(
        self,
        outputs: Vec<AtomId>,
        input_structure: Input::ParameterStructure,
        output_structure: Output::ParameterStructure,
    ) -> Program<T, V, Input, Output, O>
    where
        Input: Parameterized<V>,
        Output: Parameterized<V>,
    {
        Program {
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

impl<O: Clone, T: Type, V: Traceable<T>> Default for ProgramBuilder<O, T, V> {
    fn default() -> Self {
        Self::new()
    }
}

/// Canonical operation type used by the staged program IR.
pub type ProgramOpRef<V> = PrimitiveOp<ArrayType, V>;

/// Canonical operation type used by the staged linear-program IR.
pub type LinearProgramOpRef<V> = LinearPrimitiveOp<ArrayType, V>;

/// Shared builder used by the staged linear-program IR. The optional `O` parameter allows callers
/// to stage against an alternate linear operation carrier.
pub type LinearProgramBuilder<V, O = LinearProgramOpRef<V>> = ProgramBuilder<O, ArrayType, V>;

// ---------------------------------------------------------------------------
// Algebraic identity elimination helpers
// ---------------------------------------------------------------------------

/// Checks if one staged constant is an exact zero.
pub(crate) fn is_identity_zero<T: Type, V: Traceable<T>>(value: &V) -> bool {
    value.is_zero()
}

/// Checks if one staged constant is an exact one.
pub(crate) fn is_identity_one<T: Type, V: Traceable<T>>(value: &V) -> bool {
    value.is_one()
}

/// Executable staged program over an open operation set.
pub struct Program<
    T: Type,
    V: Typed<T> + Parameter,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
    O = ProgramOpRef<V>,
> {
    atoms: Vec<Atom<T, V>>,
    input_atoms: Vec<AtomId>,
    equations: Vec<Equation<O>>,
    outputs: Vec<AtomId>,
    input_structure: Input::ParameterStructure,
    output_structure: Output::ParameterStructure,
    marker: PhantomData<fn(Input) -> Output>,
}

impl<
    O: Clone,
    T: Type,
    V: Traceable<T>,
    Input: Parameterized<V, ParameterStructure: Clone>,
    Output: Parameterized<V, ParameterStructure: Clone>,
> Clone for Program<T, V, Input, Output, O>
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

impl<O: Clone, T: Type, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<T, V, Input, Output, O>
{
    /// Returns the number of atoms in the program.
    #[inline]
    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    /// Returns the atom with the provided identifier.
    #[inline]
    pub fn atom(&self, id: AtomId) -> Option<&Atom<T, V>> {
        self.atoms.get(id)
    }

    /// Returns an iterator over all atoms in the program, yielding `(atom_id, &Atom<T, V>)` pairs.
    #[inline]
    pub fn atoms_iter(&self) -> impl Iterator<Item = (AtomId, &Atom<T, V>)> {
        self.atoms.iter().enumerate()
    }

    /// Returns the program input atoms in parameter order.
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

    /// Returns representative concrete inputs for this program, synthesized as zero values from the
    /// retained input types using the provided [`Engine`].
    pub fn representative_input_values<E>(&self, engine: &E) -> Result<Vec<V>, TraceError>
    where
        E: Engine<Type = T, Value = V> + ?Sized,
    {
        self.input_atoms
            .iter()
            .copied()
            .map(|atom_id| match self.atom(atom_id) {
                Some(Atom::Input { r#type }) => Ok(engine.zero(r#type)),
                _ => Err(TraceError::InternalInvariantViolation("staged program input atom did not retain a type")),
            })
            .collect()
    }

    /// Evaluates every atom in the program on the supplied flat input values.
    pub fn evaluate_atom_values(&self, input_values: Vec<V>) -> Result<Vec<V>, TraceError>
    where
        O: InterpretableOp<T, V>,
    {
        if input_values.len() != self.input_atoms.len() {
            return Err(TraceError::InvalidInputCount { expected: self.input_atoms.len(), got: input_values.len() });
        }

        let mut values = vec![None; self.atoms.len()];
        for (atom, value) in self.input_atoms.iter().copied().zip(input_values) {
            values[atom] = Some(value);
        }

        for (atom_id, atom) in self.atoms.iter().enumerate() {
            if let Atom::Constant { value } = atom {
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

    /// Evaluates every atom in the program on its representative input exemplars, synthesized as
    /// zero values from the retained input types using the provided [`Engine`].
    pub fn representative_atom_values<E>(&self, engine: &E) -> Result<Vec<V>, TraceError>
    where
        O: InterpretableOp<T, V>,
        E: Engine<Type = T, Value = V> + ?Sized,
    {
        self.evaluate_atom_values(self.representative_input_values(engine)?)
    }

    /// Clones this program while replacing only the typed input/output structures.
    pub fn clone_with_structures<NewInput, NewOutput>(
        &self,
        input_structure: NewInput::ParameterStructure,
        output_structure: NewOutput::ParameterStructure,
    ) -> Program<T, V, NewInput, NewOutput, O>
    where
        NewInput: Parameterized<V>,
        NewOutput: Parameterized<V>,
    {
        Program {
            atoms: self.atoms.clone(),
            input_atoms: self.input_atoms.clone(),
            equations: self.equations.clone(),
            outputs: self.outputs.clone(),
            input_structure,
            output_structure,
            marker: PhantomData,
        }
    }

    /// Interprets the staged program on concrete input values.
    pub fn call(&self, input: Input) -> Result<Output, TraceError>
    where
        O: InterpretableOp<T, V>,
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

    /// Eliminates dead constants and equations that do not contribute to the program outputs.
    pub fn simplify(&self) -> Result<Self, TraceError>
    where
        O: Op<T>,
        Input::ParameterStructure: Clone,
        Output::ParameterStructure: Clone,
    {
        fn mark_live<O: Clone, T: Type, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>>(
            program: &Program<T, V, Input, Output, O>,
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
                let equation = &program.equations[equation_index];
                for input in equation.inputs.iter().copied() {
                    mark_live(program, input, live_atoms, live_equations, equation_by_output);
                }
            }
        }

        fn remap_atom<O, T, V, Input, Output>(
            atom_id: usize,
            program: &Program<T, V, Input, Output, O>,
            builder: &mut ProgramBuilder<O, T, V>,
            atom_mapping: &mut HashMap<usize, usize>,
            live_equations: &[bool],
            equation_by_output: &[Option<usize>],
        ) -> Result<usize, TraceError>
        where
            O: Clone + Op<T>,
            T: Type,
            V: Traceable<T>,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        {
            if let Some(mapped_atom) = atom_mapping.get(&atom_id) {
                return Ok(*mapped_atom);
            }

            let atom = program.atom(atom_id).ok_or(TraceError::UnboundAtomId { id: atom_id })?;
            let mapped_atom = match atom {
                Atom::Input { r#type } => builder.add_input_abstract(r#type.clone()),
                Atom::Constant { value } => builder.add_constant(value.clone()),
                Atom::Derived { .. } => {
                    let equation_index = equation_by_output[atom_id]
                        .ok_or(TraceError::InternalInvariantViolation("derived atom had no owning equation"))?;
                    if !live_equations[equation_index] {
                        return Err(TraceError::InternalInvariantViolation(
                            "attempted to remap a dead derived atom during program simplification",
                        ));
                    }
                    let equation = &program.equations[equation_index];
                    let remapped_inputs = equation
                        .inputs
                        .iter()
                        .copied()
                        .map(|input| {
                            remap_atom(input, program, builder, atom_mapping, live_equations, equation_by_output)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let output_abstracts = equation
                        .outputs
                        .iter()
                        .map(|output| program.atom(*output).expect("output atom should exist").tpe().into_owned())
                        .collect::<Vec<_>>();
                    let remapped_outputs =
                        builder.add_equation_prevalidated(equation.op.clone(), remapped_inputs, output_abstracts);
                    for (old_output, new_output) in
                        equation.outputs.iter().copied().zip(remapped_outputs.iter().copied())
                    {
                        atom_mapping.insert(old_output, new_output);
                    }
                    *atom_mapping
                        .get(&atom_id)
                        .ok_or(TraceError::InternalInvariantViolation("failed to record remapped program outputs"))?
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

        let mut builder = ProgramBuilder::<O, T, V>::new();
        let mut atom_mapping = HashMap::new();
        for input_atom in self.input_atoms.iter().copied() {
            let input = self.atom(input_atom).ok_or(TraceError::UnboundAtomId { id: input_atom })?;
            let Atom::Input { r#type } = input else {
                return Err(TraceError::InternalInvariantViolation("staged program input atom did not retain a type"));
            };
            let mapped = builder.add_input_abstract(r#type.clone());
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

impl<O: Clone + Display, T: Type + Display, V: Traceable<T>, Input: Parameterized<V>, Output: Parameterized<V>> Display
    for Program<T, V, Input, Output, O>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let format_atom = |id: AtomId| format!("%{id}");
        let format_typed_atom = |id: AtomId| format!("%{id}:{}", self.atoms[id].tpe());

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
            match atom {
                Atom::Input { .. } => {}
                Atom::Constant { .. } => {
                    let prefix = if binding_count == 0 { "let" } else { "   " };
                    writeln!(formatter, "{prefix} {} = const", format_typed_atom(atom_id))?;
                    binding_count += 1;
                }
                Atom::Derived { .. } => {
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
        tracing_v2::{Cos, MatrixOps, OneLike, PrimitiveOp, Sin, TraceError, Value, ZeroLike, test_support},
        types::{ArrayType, DataType, Shape, Typed},
    };

    use super::*;

    #[test]
    fn program_builder_tracks_atom_sources_and_executes() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let x = builder.add_input(&2.0f64);
        let y = builder.add_input(&3.0f64);
        let two = builder.add_constant(2.0f64);
        let scaled_x = builder.add_equation(PrimitiveOp::Scale { factor: 2.0 }, vec![x]).unwrap()[0];
        let sum = builder.add_equation(PrimitiveOp::Add, vec![scaled_x, y]).unwrap()[0];
        let program = builder.build::<(f64, f64), f64>(vec![sum], (Placeholder, Placeholder), Placeholder);

        assert!(matches!(program.atom(x).unwrap(), Atom::Input { .. }));
        assert!(matches!(program.atom(two).unwrap(), Atom::Constant { .. }));
        assert_eq!(program.call((2.0, 3.0)).unwrap(), 7.0);
        assert_eq!(
            program.to_string(),
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
    fn program_display_uses_typed_jaxpr_like_rendering() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let x = builder.add_input(&1.0f64);
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_equation(PrimitiveOp::Add, vec![x, three]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);

        assert_eq!(
            program.to_string(),
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
    fn program_builder_rejects_unbound_inputs() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let result = builder.add_equation(PrimitiveOp::Add, vec![42, 99]);
        assert!(matches!(result, Err(TraceError::UnboundAtomId { id: 42 })));
        test_support::assert_reference_program_rendering();
    }

    #[test]
    fn test_constant_folding_eliminates_equations() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let a = builder.add_constant(2.0f64);
        let b = builder.add_constant(3.0f64);

        // Adding two constants should fold: no equation, output is constant.
        let folded = builder.add_equation(PrimitiveOp::Add, vec![a, b]).unwrap();
        assert_eq!(folded.len(), 1);
        assert!(matches!(builder.atom(folded[0]).unwrap(), Atom::Constant { .. }));
        assert_eq!(builder.equation_count(), 0);

        // Introduce a non-constant input and combine with the folded constant.
        let x = builder.add_input(&10.0f64);
        let result = builder.add_equation(PrimitiveOp::Mul, vec![folded[0], x]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(matches!(builder.atom(result[0]).unwrap(), Atom::Derived { .. }));
        assert_eq!(builder.equation_count(), 1);

        // Build the program and verify only the non-folded equation survived.
        let program = builder.build::<f64, f64>(vec![result[0]], Placeholder, Placeholder);
        assert_eq!(program.equations().len(), 1);
        assert_eq!(
            program.to_string(),
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
    fn test_constant_folding_program_call_produces_correct_results() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let a = builder.add_constant(2.0f64);
        let b = builder.add_constant(3.0f64);
        let folded_sum = builder.add_equation(PrimitiveOp::Add, vec![a, b]).unwrap()[0];

        let x = builder.add_input(&10.0f64);
        let product = builder.add_equation(PrimitiveOp::Mul, vec![folded_sum, x]).unwrap()[0];
        let program = builder.build::<f64, f64>(vec![product], Placeholder, Placeholder);

        // folded_sum = 2.0 + 3.0 = 5.0, product = 5.0 * input
        assert_eq!(program.call(10.0).unwrap(), 50.0);
        assert_eq!(program.call(0.5).unwrap(), 2.5);
        assert_eq!(program.call(0.0).unwrap(), 0.0);
    }

    #[test]
    fn built_program_drops_derived_stored_values_but_reconstructs_representatives() {
        let mut builder = ProgramBuilder::<PrimitiveOp<ArrayType, f64>, ArrayType, f64>::new();
        let x = builder.add_input(&2.0f64);
        let three = builder.add_constant(3.0f64);
        let sum = builder.add_equation(PrimitiveOp::Add, vec![x, three]).unwrap()[0];

        assert!(
            matches!(builder.atom(x).unwrap(), Atom::Input { r#type } if *r#type == ArrayType::scalar(DataType::F64))
        );
        assert!(matches!(builder.atom(three).unwrap(), Atom::Constant { value } if *value == 3.0));
        assert!(matches!(builder.atom(sum).unwrap(), Atom::Derived { .. }));
        assert_eq!(builder.stored_value(x), Some(&2.0));
        assert_eq!(builder.stored_value(sum), Some(&5.0));

        let program = builder.build::<f64, f64>(vec![sum], Placeholder, Placeholder);
        let engine = crate::tracing_v2::engine::ArrayScalarEngine::<f64>::new();
        assert!(
            matches!(program.atom(x).unwrap(), Atom::Input { r#type } if *r#type == ArrayType::scalar(DataType::F64))
        );
        assert!(matches!(program.atom(three).unwrap(), Atom::Constant { value } if *value == 3.0));
        assert!(matches!(program.atom(sum).unwrap(), Atom::Derived { .. }));
        assert_eq!(program.representative_atom_values(&engine).unwrap(), vec![0.0, 3.0, 3.0]);
        assert_eq!(program.call(4.0).unwrap(), 7.0);
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
            fn tpe(&self) -> std::borrow::Cow<'_, ArrayType> {
                std::borrow::Cow::Borrowed(&self.r#type)
            }
        }

        impl Traceable<ArrayType> for TestIdentityValue {
            fn is_zero(&self) -> bool {
                self.value == 0.0
            }

            fn is_one(&self) -> bool {
                self.value == 1.0
            }
        }

        impl Value<ArrayType> for TestIdentityValue {}

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

        impl Sin for TestIdentityValue {
            fn sin(self) -> Self {
                self
            }
        }

        impl Cos for TestIdentityValue {
            fn cos(self) -> Self {
                self
            }
        }

        impl ZeroLike for TestIdentityValue {
            fn zero_like(&self) -> Self {
                Self::scalar(0.0)
            }
        }

        impl OneLike for TestIdentityValue {
            fn one_like(&self) -> Self {
                Self::scalar(1.0)
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

        let mut builder =
            ProgramBuilder::<PrimitiveOp<ArrayType, TestIdentityValue>, ArrayType, TestIdentityValue>::new();
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
