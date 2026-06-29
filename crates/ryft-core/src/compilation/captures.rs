use std::borrow::Cow;
use std::fmt::{Debug, Display};

use ryft_macros::Parameter;

use crate::contexts::EagerContext;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::parameters::{Parameter, Parameterized, Placeholder};
use crate::programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramError, Value};
use crate::types::{Type, Typed};

/// Reference to a value captured outside a staged [`Program`].
///
/// The program stores only this lifetime-free reference in its atom table. The corresponding
/// runtime value lives in the surrounding [`ClosedProgram`] capture table at [`Self::index`].
/// The IR remains abstract and reusable, while concrete runtime values stay in a side
/// environment owned by the compiled function.
///
/// # Why capture by reference instead of baking in a literal
///
/// Closed-over runtime values — for example, the arrays a just-in-time-compiled function closes over — are recorded
/// as captures and handed to the compiled program as runtime arguments, rather than embedded as literal constants in
/// its IR. The compiled program therefore depends only on the captured values' abstract types, never on their
/// concrete data, which buys three things:
///
///   - **Executable reuse.** Compiled executables are cached by operand type and shape, not by value, so a single
///     compilation serves any captured value of a given type. Baking a value in as a literal would make the
///     executable value-specific and force a recompile whenever the captured data changed.
///   - **On-device buffers.** Captured values are typically device buffers; passing them as arguments keeps them
///     resident on-device, whereas embedding them as literals would require reading them back to the host first.
///   - **Compact IR.** Large captured arrays never bloat the program IR or the serialized executable.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct CaptureReference<T: Type> {
    /// Index into the surrounding capture table.
    index: usize,

    /// Abstract type metadata for the captured value.
    r#type: T,
}

impl<T: Type> CaptureReference<T> {
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

impl<T: Type> Display for CaptureReference<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "capture#{}:{}", self.index, self.r#type)
    }
}

impl<T: Type> Typed<T> for CaptureReference<T> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<T: Type> Value<T> for CaptureReference<T> {
    type InterpretationContext = EagerContext<T, Self>;

    #[inline]
    fn interpretation_context(&self) -> Option<Self::InterpretationContext> {
        Some(EagerContext::new())
    }
}

/// A staged [`Program`] paired with the concrete runtime values referenced by its captured
/// constants.
///
/// `Program` remains lifetime-free except for its operation payloads. Concrete values of type
/// `V` live only in [`Self::captures`], and atom-table constants are
/// [`CaptureReference<T>`] references into that side table.
pub struct ClosedProgram<
    T: Type,
    V: Value<T>,
    O,
    Input: Parameterized<CaptureReference<T>>,
    Output: Parameterized<CaptureReference<T>>,
> {
    /// Staged program whose constants are capture references.
    program: Program<T, CaptureReference<T>, O, Input, Output>,

    /// Concrete captured values referenced by [`CaptureReference`] indices in [`Self::program`].
    captures: Vec<V>,
}

impl<
    T: Type,
    V: Value<T>,
    O: Clone,
    Input: Parameterized<CaptureReference<T>>,
    Output: Parameterized<CaptureReference<T>>,
> Clone for ClosedProgram<T, V, O, Input, Output>
{
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), captures: self.captures.clone() }
    }
}

impl<T: Type, V: Value<T>, O, Input: Parameterized<CaptureReference<T>>, Output: Parameterized<CaptureReference<T>>>
    Debug for ClosedProgram<T, V, O, Input, Output>
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClosedProgram")
            .field("captures", &self.captures.len())
            .finish_non_exhaustive()
    }
}

impl<T: Type, V: Value<T>, O, Input: Parameterized<CaptureReference<T>>, Output: Parameterized<CaptureReference<T>>>
    ClosedProgram<T, V, O, Input, Output>
{
    /// Creates a captured program from an already capture-referenced program and capture table.
    #[inline]
    pub fn new(program: Program<T, CaptureReference<T>, O, Input, Output>, captures: Vec<V>) -> Self {
        Self { program, captures }
    }

    /// Returns the staged program.
    #[inline]
    pub fn program(&self) -> &Program<T, CaptureReference<T>, O, Input, Output> {
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
    Input: Parameterized<CaptureReference<T>>,
    Output: Parameterized<CaptureReference<T>>,
> ClosedProgram<T, V, O, Input, Output>
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
    pub fn validate_capture_inputs(&self, capture_inputs: &[CaptureReference<T>]) -> Result<(), ProgramError> {
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
    Input: Parameterized<CaptureReference<T>>,
    Output: Parameterized<CaptureReference<T>>,
> ClosedProgram<T, V, O, Input, Output>
{
    /// Returns a flat program where captures are explicit leading inputs followed by the original program inputs.
    pub fn open_captures_as_inputs(
        &self,
    ) -> Result<Program<T, CaptureReference<T>, O, Vec<CaptureReference<T>>, Vec<CaptureReference<T>>>, ProgramError>
    {
        fn map_atom<T: Type>(
            atoms: &[Atom<T, CaptureReference<T>>],
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
        builder.build::<Vec<CaptureReference<T>>, Vec<CaptureReference<T>>>(
            output_ids,
            vec![Placeholder; self.captures.len() + self.program.input_ids().len()],
            vec![Placeholder; self.program.output_ids().len()],
        )
    }
}

impl<
    T: Type,
    V: Clone + Value<T>,
    O: Clone + Operation<T>,
    Input: Parameterized<CaptureReference<T>>,
    Output: Parameterized<CaptureReference<T>>,
> ClosedProgram<T, V, O, Input, Output>
{
    /// Rebuilds this program with its capture table replaced by `new_captures` and every
    /// [`CaptureReference`] reindexed through `capture_index_map`.
    ///
    /// The program structure (inputs, instructions, and outputs) is preserved; only constant atoms are rewritten. Each
    /// surviving [`CaptureReference`] at old index `old` becomes a reference to `capture_index_map[old]`, so callers
    /// must supply a map that assigns a new index to every capture still referenced by the program and order
    /// `new_captures` to match those new indices. Constant atoms whose old index maps to [`None`] must be unreachable
    /// (no atom may reference a dropped capture); reaching one is a malformed-program error.
    fn reindex_captures(
        &self,
        capture_index_map: &[Option<usize>],
        new_captures: Vec<V>,
    ) -> Result<Self, ProgramError> {
        let mut builder = ProgramBuilder::new();
        let mut mapped_atoms = vec![None; self.program.atoms().len()];

        // Materializes the rebuilt atom identifier for `atom_id`, lazily re-adding referenced capture constants with
        // their reindexed reference so atoms shared across instructions map to a single rebuilt constant.
        let map_atom = |builder: &mut ProgramBuilder<T, CaptureReference<T>, O>,
                        mapped_atoms: &mut [Option<AtomId>],
                        atom_id: AtomId|
         -> Result<AtomId, ProgramError> {
            if let Some(mapped) = mapped_atoms.get(atom_id.index()).copied().flatten() {
                return Ok(mapped);
            }
            match self.program.atoms().get(atom_id.index()) {
                Some(Atom::Constant(captured)) => {
                    let new_index = capture_index_map.get(captured.index()).copied().flatten().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "captured constant atom {atom_id} references capture #{} that has no reindexed slot",
                            captured.index(),
                        ))
                    })?;
                    let mapped = builder.add_constant(CaptureReference::new(new_index, captured.r#type().into_owned()));
                    mapped_atoms[atom_id.index()] = Some(mapped);
                    Ok(mapped)
                }
                Some(Atom::Variable(_)) => Err(ProgramError::MalformedProgram(format!(
                    "variable atom {atom_id} has no mapped input or instruction output",
                ))),
                None => Err(ProgramError::UnboundAtomId { id: atom_id }),
            }
        };

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
                .map(|input| map_atom(&mut builder, mapped_atoms.as_mut_slice(), input))
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
            .map(|output| map_atom(&mut builder, mapped_atoms.as_mut_slice(), output))
            .collect::<Result<Vec<_>, _>>()?;
        let program = builder.build::<Input, Output>(
            output_ids,
            self.program.input_structure().clone(),
            self.program.output_structure().clone(),
        )?;
        Ok(Self::new(program, new_captures))
    }

    /// Removes captures that no atom references and reindexes the survivors into a contiguous capture table.
    ///
    /// This is dead-capture elimination: any capture whose index never appears in an [`Atom::Constant`] is dropped, and
    /// the remaining captures are renumbered to occupy `0..captures().len()` in their original relative order. Every
    /// surviving [`CaptureReference`] is rewritten to its new index, so the returned program satisfies
    /// [`validate_capture_references`](Self::validate_capture_references) with a contiguous capture table. The pass is
    /// unconditional: it needs only `V: Clone` and `O: Clone`, never an equality comparison on capture values.
    pub fn prune_unused_captures(&self) -> Result<Self, ProgramError> {
        let mut capture_index_map = vec![None; self.captures.len()];
        let mut new_captures = Vec::new();
        for atom in self.program.atoms() {
            let Atom::Constant(captured) = atom else {
                continue;
            };
            let old_index = captured.index();
            if capture_index_map.get(old_index).copied().flatten().is_some() {
                continue;
            }
            let capture = self.captures.get(old_index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "captured constant atom references missing capture #{old_index}",
                ))
            })?;
            capture_index_map[old_index] = Some(new_captures.len());
            new_captures.push(capture.clone());
        }
        self.reindex_captures(capture_index_map.as_slice(), new_captures)
    }

    /// Merges captures that hold equal values into a single capture and reindexes the survivors contiguously.
    ///
    /// Captures comparing equal under [`PartialEq`] are collapsed to one entry: every [`CaptureReference`] to a
    /// duplicate is rewritten to the first (canonical) capture that holds that value, and the surviving captures are
    /// renumbered to `0..captures().len()`. The meaningful case is the *same runtime value captured more than once* —
    /// for example, a closed-over buffer referenced by several operations — which this pass deduplicates so the
    /// compiled function carries and transfers it only once. Captures that are never referenced are also dropped (a
    /// duplicate of an unused value still collapses to its first occurrence, which itself survives only if referenced),
    /// and the returned program satisfies [`validate_capture_references`](Self::validate_capture_references).
    ///
    /// # Capture-value equality
    ///
    /// Deduplication is only as correct and cheap as the value type's [`PartialEq`]. It is intended for value types
    /// whose equality is a cheap identity/handle comparison; types whose equality forces an expensive materialization
    /// (for example, reading a device buffer back to the host to compare element data) should not be deduplicated this
    /// way. The bound is therefore opt-in: value types that do not implement [`PartialEq`] simply cannot call this
    /// pass, whereas [`prune_unused_captures`](Self::prune_unused_captures) stays available unconditionally.
    pub fn deduplicate_captures(&self) -> Result<Self, ProgramError>
    where
        V: PartialEq,
    {
        let mut capture_index_map = vec![None; self.captures.len()];
        let mut new_captures: Vec<V> = Vec::new();
        for atom in self.program.atoms() {
            let Atom::Constant(captured) = atom else {
                continue;
            };
            let old_index = captured.index();
            if capture_index_map.get(old_index).copied().flatten().is_some() {
                continue;
            }
            let capture = self.captures.get(old_index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "captured constant atom references missing capture #{old_index}",
                ))
            })?;
            let new_index = match new_captures.iter().position(|existing| existing == capture) {
                Some(existing_index) => existing_index,
                None => {
                    new_captures.push(capture.clone());
                    new_captures.len() - 1
                }
            };
            capture_index_map[old_index] = Some(new_index);
        }
        self.reindex_captures(capture_index_map.as_slice(), new_captures)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::macros::check_count;
    use crate::operations::{InterpretableOperation, Operation};
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError, Value};
    use crate::scalars::Scalar;
    use crate::types::{DataType, TypeError};

    use super::{CaptureReference, ClosedProgram};

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

    impl InterpretableOperation<DataType, Scalar> for TestAddOperation {
        fn interpret(
            &self,
            _context: &<Scalar as Value<DataType>>::InterpretationContext,
            inputs: &[Scalar],
        ) -> Result<Vec<Scalar>, ProgramError> {
            check_count!("input", inputs, 2, ProgramError);
            Ok(vec![inputs[0] + inputs[1]])
        }
    }

    fn captured_add_program() -> ClosedProgram<
        DataType,
        Scalar,
        TestAddOperation,
        Vec<CaptureReference<DataType>>,
        Vec<CaptureReference<DataType>>,
    > {
        let mut builder = ProgramBuilder::new();
        let input = builder.add_input(DataType::F64);
        let capture = builder.add_constant(CaptureReference::new(0, DataType::F64));
        let output = builder.add_instruction(TestAddOperation, vec![input, capture]).unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        ClosedProgram::new(program, vec![Scalar::from(3.0)])
    }

    #[test]
    fn test_captured_program_interprets_through_capture_table() {
        let program = captured_add_program();

        let output = program
            .interpret_with_captures(
                vec![Scalar::from(2.0)],
                |_, capture| Ok::<_, ProgramError>(*capture),
                |instruction, inputs| instruction.operation().interpret(&EagerContext::new(), inputs),
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
                vec![Scalar::from(3.0), Scalar::from(2.0)],
                |_, constant| Ok::<_, ProgramError>(Scalar::from(constant.index() as f64)),
                |instruction, inputs| instruction.operation().interpret(&EagerContext::new(), inputs),
            )
            .unwrap();

        assert_eq!(output, vec![5.0]);
    }

    #[test]
    fn test_captured_program_rejects_missing_capture_index() {
        let mut builder = ProgramBuilder::<DataType, CaptureReference<DataType>, TestAddOperation>::new();
        let capture = builder.add_constant(CaptureReference::new(1, DataType::F64));
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                vec![capture],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        let program = ClosedProgram::new(program, vec![Scalar::from(3.0)]);

        assert!(matches!(
            program.validate_capture_references(),
            Err(ProgramError::MalformedProgram(message))
                if message == "captured constant atom %0 references missing capture #1",
        ));
    }

    #[test]
    fn test_prune_unused_captures_drops_dead_capture_and_reindexes() {
        // The program references only capture #1; capture #0 (value 3.0) is dead.
        let mut builder = ProgramBuilder::<DataType, CaptureReference<DataType>, TestAddOperation>::new();
        let input = builder.add_input(DataType::F64);
        let capture = builder.add_constant(CaptureReference::new(1, DataType::F64));
        let output = builder.add_instruction(TestAddOperation, vec![input, capture]).unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let program = ClosedProgram::new(program, vec![Scalar::from(3.0), Scalar::from(99.0)]);

        let pruned = program.prune_unused_captures().unwrap();

        // The dead capture is dropped and the survivor is reindexed to a contiguous table.
        assert_eq!(pruned.captures(), &[99.0]);
        let capture_indices = pruned
            .program()
            .atoms()
            .iter()
            .filter_map(|atom| atom.as_constant().map(|reference| reference.index()))
            .collect::<Vec<_>>();
        assert_eq!(capture_indices, vec![0]);
        pruned.validate_capture_references().unwrap();

        let output = pruned
            .interpret_with_captures(
                vec![Scalar::from(2.0)],
                |_, capture| Ok::<_, ProgramError>(*capture),
                |instruction, inputs| instruction.operation().interpret(&EagerContext::new(), inputs),
            )
            .unwrap();
        assert_eq!(output, vec![101.0]);
    }

    #[test]
    fn test_deduplicate_captures_merges_equal_captures() {
        // Both captures hold the same value (7) and are referenced by two distinct constant atoms.
        let mut builder = ProgramBuilder::<DataType, CaptureReference<DataType>, TestAddOperation>::new();
        let first = builder.add_constant(CaptureReference::new(0, DataType::I64));
        let second = builder.add_constant(CaptureReference::new(1, DataType::I64));
        let output = builder.add_instruction(TestAddOperation, vec![first, second]).unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                vec![output],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        let program = ClosedProgram::new(program, vec![Scalar::from(7_i64), Scalar::from(7_i64)]);

        let deduplicated = program.deduplicate_captures().unwrap();

        // The two equal captures collapse to one entry and both atoms reference the canonical index 0.
        assert_eq!(deduplicated.captures(), &[7_i64]);
        let capture_indices = deduplicated
            .program()
            .atoms()
            .iter()
            .filter_map(|atom| atom.as_constant().map(|reference| reference.index()))
            .collect::<Vec<_>>();
        assert_eq!(capture_indices, vec![0, 0]);
        deduplicated.validate_capture_references().unwrap();
    }
}
