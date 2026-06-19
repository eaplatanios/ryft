use std::borrow::Cow;
use std::fmt::{Debug, Display};

use ryft_macros::Parameter;

use crate::macros::check_count;
use crate::operations::Operation;
use crate::parameters::{Parameter, Parameterized, Placeholder};
use crate::programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramError, Value};
use crate::types::{Type, Typed};

/// Reference to a value captured outside a staged [`Program`].
///
/// The program stores only this lifetime-free reference in its atom table. The corresponding
/// runtime value lives in the surrounding [`CapturedProgram`] capture table at [`Self::index`].
/// The IR remains abstract and reusable, while concrete runtime values stay in a side
/// environment owned by the compiled function.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct CapturedConstant<T: Type> {
    /// Index into the surrounding capture table.
    index: usize,

    /// Abstract type metadata for the captured value.
    r#type: T,
}

impl<T: Type> CapturedConstant<T> {
    /// Creates a captured-constant reference.
    #[inline]
    pub fn new(index: usize, r#type: T) -> Self {
        Self { index, r#type }
    }

    /// Returns the index into the surrounding capture table.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }
}

impl<T: Type> Display for CapturedConstant<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "capture#{}:{}", self.index, self.r#type)
    }
}

impl<T: Type> Typed<T> for CapturedConstant<T> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<T: Type> Value<T> for CapturedConstant<T> {
    type InterpretationContext = ();

    #[inline]
    fn interpretation_context(&self) -> Option<()> {
        Some(())
    }
}

/// A staged [`Program`] paired with the concrete runtime values referenced by its captured
/// constants.
///
/// `Program` remains lifetime-free except for its operation payloads. Concrete values of type
/// `V` live only in [`Self::captures`], and atom-table constants are
/// [`CapturedConstant<T>`] references into that side table.
pub struct CapturedProgram<
    T: Type,
    V: Value<T>,
    O,
    Input: Parameterized<CapturedConstant<T>>,
    Output: Parameterized<CapturedConstant<T>>,
> {
    /// Staged program whose constants are capture references.
    program: Program<T, CapturedConstant<T>, O, Input, Output>,

    /// Concrete captured values referenced by [`CapturedConstant`] indices in [`Self::program`].
    captures: Vec<V>,
}

impl<
    T: Type,
    V: Value<T>,
    O: Clone,
    Input: Parameterized<CapturedConstant<T>>,
    Output: Parameterized<CapturedConstant<T>>,
> Clone for CapturedProgram<T, V, O, Input, Output>
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), captures: self.captures.clone() }
    }
}

impl<T: Type, V: Value<T>, O, Input: Parameterized<CapturedConstant<T>>, Output: Parameterized<CapturedConstant<T>>>
    Debug for CapturedProgram<T, V, O, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CapturedProgram")
            .field("captures", &self.captures.len())
            .finish_non_exhaustive()
    }
}

impl<T: Type, V: Value<T>, O, Input: Parameterized<CapturedConstant<T>>, Output: Parameterized<CapturedConstant<T>>>
    CapturedProgram<T, V, O, Input, Output>
{
    /// Creates a captured program from an already capture-referenced program and capture table.
    #[inline]
    pub fn new(program: Program<T, CapturedConstant<T>, O, Input, Output>, captures: Vec<V>) -> Self {
        Self { program, captures }
    }

    /// Returns the staged program.
    #[inline]
    pub fn program(&self) -> &Program<T, CapturedConstant<T>, O, Input, Output> {
        &self.program
    }

    /// Returns the captured runtime values.
    #[inline]
    pub fn captures(&self) -> &[V] {
        self.captures.as_slice()
    }
}

impl<
    T: Type,
    V: Value<T>,
    O: Operation<T>,
    Input: Parameterized<CapturedConstant<T>>,
    Output: Parameterized<CapturedConstant<T>>,
> CapturedProgram<T, V, O, Input, Output>
{
    /// Validates that every captured-constant atom references an existing capture with the same type.
    pub fn validate_capture_references(&self) -> Result<(), ProgramError> {
        for (atom_index, atom) in self.program.atoms().iter().enumerate() {
            let Atom::Constant(captured) = atom else {
                continue;
            };
            let capture = self.captures.get(captured.index()).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "captured constant atom %{atom_index} references missing capture #{}",
                    captured.index(),
                ))
            })?;
            let expected_type = captured.r#type();
            let actual_type = capture.r#type();
            if expected_type.as_ref() != actual_type.as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "captured constant atom %{atom_index} references capture #{} with type {}, but the atom has type {}",
                    captured.index(),
                    actual_type,
                    expected_type,
                )));
            }
        }
        Ok(())
    }

    /// Validates capture references that will be supplied as the leading inputs of an opened captured program.
    ///
    /// Capture input indices belong to the caller's capture table and may differ from this program's local capture
    /// indices. This validates the positional arity and types that matter after captures are opened as leading flat
    /// inputs.
    pub fn validate_capture_inputs(&self, capture_inputs: &[CapturedConstant<T>]) -> Result<(), ProgramError> {
        check_count!("input", capture_inputs, self.captures.len(), ProgramError);
        for (index, (expected, actual)) in self.captures.iter().zip(capture_inputs).enumerate() {
            let expected_type = expected.r#type();
            let actual_type = actual.r#type();
            if expected_type.as_ref() != actual_type.as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "capture input #{index} has type {}, but capture #{index} has type {}",
                    actual_type, expected_type,
                )));
            }
        }
        Ok(())
    }

    /// Interprets this captured program by resolving captured constants through its capture table.
    pub fn interpret_with_captures<
        Value: Clone,
        Error: From<ProgramError>,
        LiftCapture: FnMut(usize, &V) -> Result<Value, Error>,
        InterpretInstruction: FnMut(&Instruction<O>, &[Value]) -> Result<Vec<Value>, Error>,
    >(
        &self,
        inputs: Vec<Value>,
        mut lift_capture: LiftCapture,
        interpret_instruction: InterpretInstruction,
    ) -> Result<Vec<Value>, Error> {
        self.validate_capture_references().map_err(Error::from)?;
        self.program.interpret_with(
            inputs,
            |_, captured| {
                let capture = &self.captures[captured.index()];
                lift_capture(captured.index(), capture)
            },
            interpret_instruction,
        )
    }
}

impl<
    T: Type,
    V: Value<T>,
    O: Clone + Operation<T>,
    Input: Parameterized<CapturedConstant<T>>,
    Output: Parameterized<CapturedConstant<T>>,
> CapturedProgram<T, V, O, Input, Output>
{
    /// Returns a flat program where captures are explicit leading inputs followed by the original program inputs.
    pub fn open_captures_as_inputs(
        &self,
    ) -> Result<Program<T, CapturedConstant<T>, O, Vec<CapturedConstant<T>>, Vec<CapturedConstant<T>>>, ProgramError>
    {
        fn map_atom<T: Type>(
            atoms: &[Atom<T, CapturedConstant<T>>],
            mapped_atoms: &[Option<AtomId>],
            capture_inputs: &[AtomId],
            atom_id: AtomId,
        ) -> Result<AtomId, ProgramError> {
            if let Some(mapped) = mapped_atoms.get(atom_id.index()).copied().flatten() {
                return Ok(mapped);
            }
            match atoms.get(atom_id.index()) {
                Some(Atom::Constant(captured)) => Ok(capture_inputs[captured.index()]),
                Some(Atom::Variable(_)) => Err(ProgramError::MalformedProgram(format!(
                    "variable atom {atom_id} has no mapped input or instruction output",
                ))),
                None => Err(ProgramError::UnboundAtomId { id: atom_id }),
            }
        }

        self.validate_capture_references()?;
        let mut builder = ProgramBuilder::new();
        let capture_inputs = self
            .captures
            .iter()
            .map(|capture| builder.add_input(capture.r#type().into_owned()))
            .collect::<Vec<_>>();
        let mut mapped_atoms = vec![None; self.program.atoms().len()];

        for input_id in self.program.input_ids().iter().copied() {
            let input =
                self.program.atoms().get(input_id.index()).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(input_type) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            mapped_atoms[input_id.index()] = Some(builder.add_input(input_type.clone()));
        }

        for instruction in self.program.instructions() {
            let inputs = instruction
                .inputs()
                .iter()
                .copied()
                .map(|input| map_atom(self.program.atoms(), mapped_atoms.as_slice(), capture_inputs.as_slice(), input))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = builder.add_instruction(instruction.operation().clone(), inputs)?.to_vec();
            check_count!("output", outputs, instruction.outputs().len(), ProgramError);
            for (old, new) in instruction.outputs().iter().copied().zip(outputs) {
                let mapped = mapped_atoms.get_mut(old.index()).ok_or(ProgramError::UnboundAtomId { id: old })?;
                *mapped = Some(new);
            }
        }

        let output_ids = self
            .program
            .output_ids()
            .iter()
            .copied()
            .map(|output| map_atom(self.program.atoms(), mapped_atoms.as_slice(), capture_inputs.as_slice(), output))
            .collect::<Result<Vec<_>, _>>()?;
        builder.build::<Vec<CapturedConstant<T>>, Vec<CapturedConstant<T>>>(
            output_ids,
            vec![Placeholder; self.captures.len() + self.program.input_ids().len()],
            vec![Placeholder; self.program.output_ids().len()],
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::macros::check_count;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError, Value};
    use crate::types::{DataType, TypeError};

    use super::{CapturedConstant, CapturedProgram};

    #[derive(Clone, Debug)]
    struct TestAddOperation;

    impl Operation<DataType> for TestAddOperation {
        fn name(&self) -> &'static str {
            "test_add"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 2, TypeError);
            Ok(vec![input_types[0].clone()])
        }
    }

    impl InterpretableOperation<DataType, f64> for TestAddOperation {
        fn interpret(
            &self,
            _context: &<f64 as Value<DataType>>::InterpretationContext,
            inputs: &[f64],
        ) -> Result<Vec<f64>, ProgramError> {
            check_count!("input", inputs, 2, ProgramError);
            Ok(vec![inputs[0] + inputs[1]])
        }
    }

    fn captured_add_program() -> CapturedProgram<
        DataType,
        f64,
        TestAddOperation,
        Vec<CapturedConstant<DataType>>,
        Vec<CapturedConstant<DataType>>,
    > {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(DataType::F64);
        let capture = builder.add_constant(CapturedConstant::new(0, DataType::F64));
        let output = builder.add_instruction(TestAddOperation, vec![input, capture]).unwrap()[0];
        let program = builder
            .build::<Vec<CapturedConstant<DataType>>, Vec<CapturedConstant<DataType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        CapturedProgram::new(program, vec![3.0])
    }

    #[test]
    fn test_captured_program_interprets_through_capture_table() {
        let program = captured_add_program();

        let output = program
            .interpret_with_captures(
                vec![2.0],
                |_, capture| Ok::<_, ProgramError>(*capture),
                |instruction, inputs| instruction.operation().interpret(&(), inputs),
            )
            .unwrap();

        assert_eq!(output, vec![5.0]);
    }

    #[test]
    fn test_captured_program_opens_captures_as_leading_inputs() {
        let program = captured_add_program();
        let opened = program.open_captures_as_inputs().unwrap();

        assert_eq!(opened.input_ids().len(), 2);
        assert!(opened.atoms().iter().all(|atom| !atom.is_constant()));
        let output = opened
            .interpret_with(
                vec![3.0, 2.0],
                |_, constant| Ok::<_, ProgramError>(constant.index() as f64),
                |instruction, inputs| instruction.operation().interpret(&(), inputs),
            )
            .unwrap();

        assert_eq!(output, vec![5.0]);
    }

    #[test]
    fn test_captured_program_rejects_missing_capture_index() {
        let mut builder = ProgramBuilder::<DataType, CapturedConstant<DataType>, TestAddOperation>::new();
        let capture = builder.add_constant(CapturedConstant::new(1, DataType::F64));
        let program = builder
            .build::<Vec<CapturedConstant<DataType>>, Vec<CapturedConstant<DataType>>>(
                vec![capture],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        let program = CapturedProgram::new(program, vec![3.0]);

        assert!(matches!(
            program.validate_capture_references(),
            Err(ProgramError::MalformedProgram(message))
                if message == "captured constant atom %0 references missing capture #1",
        ));
    }
}
