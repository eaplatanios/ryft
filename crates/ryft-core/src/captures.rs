//! Contains machinery for representing runtime _captures_ for staged [`Program`]s.
//!
//! Runtime values closed over by a traced closure (e.g., the device buffers a just-in-time-compiled function reads) are
//! not embedded in staged programs as literal constants. Instead, a [`ClosedProgram`] keeps them in a side table while
//! the program's constant atoms store lifetime-free, typed [`CaptureReference`] indices into that table. This keeps the
//! IR compact, preserves device-resident buffers, and lets compilation depend only on capture *types* rather than
//! concrete data, so one compiled executable serves any captured value of a given type.
//!
//! The compilation lifecycle built on top of these types lives in [`crate::compilation`].

use std::borrow::Cow;
use std::fmt::{Debug, Display};

use ryft_macros::Parameter;

use crate::batching::{BatchableOperation, BatchingContext};
use crate::contexts::{Context, EagerContext};
use crate::differentiation::{DifferentiableOperation, DifferentiationContext};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::Zero;
use crate::parameters::{Parameter, Parameterized, Placeholder};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{NestedTracingContext, TracingContext};
use crate::types::{ArrayType, Type, Typed};

/// Reference to a value captured outside a staged [`Program`]. A program stores only this lifetime-free reference
/// in its [`Atom`] table. The corresponding runtime value lives at [`index`](Self::index) in the surrounding
/// [`ClosedProgram`]'s capture table. This keeps concrete runtime values in the closed program's side environment
/// instead of embedding them in reusable staged IR.
///
/// # Why capture by reference instead of baking in a literal?
///
/// Closed-over runtime values (e.g., the arrays a just-in-time-compiled function closes over) are recorded as captures
/// and handed to the compiled program as runtime arguments, rather than embedded as literal constants in its IR. The
/// compiled program therefore depends only on the captured values' abstract types and never on their concrete data,
/// which buys three things:
///
///   - **Executable Reuse:** Compiled executables are cached by operand type and shape, not by value, so a single
///     compilation serves any captured value of a given type. Baking a value in as a literal would make the executable
///     value-specific and force a recompilation whenever the captured data changes.
///   - **On-Device Buffers:** Captured values are typically device (e.g., GPU) buffers. Passing them as arguments keeps
///     them resident on-device, whereas embedding them as literals would require copying them back to the host first.
///   - **Compact IR:** Large captured arrays never bloat the program IR or the serialized executables.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct CaptureReference<T: Type> {
    /// Index of this [`CaptureReference`] into the surrounding capture table.
    index: usize,

    /// [`Type`] of the underlying captured value.
    r#type: T,
}

impl<T: Type> CaptureReference<T> {
    /// Creates a new [`CaptureReference`].
    #[inline]
    pub fn new(index: usize, r#type: T) -> Self {
        Self { index, r#type }
    }

    /// Returns the index of this [`CaptureReference`] into the surrounding capture table.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }
}

impl<T: Type> Display for CaptureReference<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "capture#{}:{}", self.index, self.r#type)
    }
}

impl<T: Type> Typed for CaptureReference<T> {
    type Type = T;

    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<T: Type> Value for CaptureReference<T> {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self>;

    #[inline]
    fn dispatch_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> EagerContext<Self> {
        EagerContext::new()
    }
}

/// [`Context`] that can register runtime values as captures (e.g., for a [`Program`] that is being built). The returned
/// value is the context's staged constant payload. For captured-program backends this is usually a lifetime-free
/// [`CaptureReference`] into a side table owned by the surrounding compiled function. Stackable transform contexts
/// delegate registration to their parent so captures follow the same nesting path as ordinary staged operations.
/// Capturing a closed-over value instead of staging it as a literal keeps the program independent of concrete data,
/// enabling executable reuse across captured values, retaining device buffers on-device, and keeping the IR compact.
pub trait CapturingContext: Context {
    /// Concrete runtime value type that this [`CapturingContext`] registers in its capture table. This is deliberately
    /// distinct from both [`Context::Value`] (i.e., the value type flowing through the context, which is typically
    /// a [`Tracer`](crate::Tracer) while capturing) and [`Context::Constant`] (i.e., the staged payload that
    /// [`capture`](Self::capture) returns).
    type Capture: Value<Type = Self::Type>;

    /// Appends `value` to the current captures table of this [`CapturingContext`] and returns the constant payload
    /// that refers to it.
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError>;
}

impl<T: Type, O: Operation<T>, C: Value<Type = T>> CapturingContext for TracingContext<CaptureReference<T>, O, C> {
    type Capture = C;

    #[inline]
    fn capture(&self, value: C) -> Result<Self::Constant, ProgramError> {
        let mut captures = self.captures().borrow_mut();
        let constant = CaptureReference::new(captures.len(), value.r#type().into_owned());
        captures.push(value);
        Ok(constant)
    }
}

impl<C: CapturingContext> CapturingContext for NestedTracingContext<C> {
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

impl<C: CapturingContext<Operation: PartiallyEvaluatableOperation<C>>> CapturingContext
    for PartialEvaluationContext<C>
{
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

impl<C: CapturingContext<Type = ArrayType, Operation: BatchableOperation<C::Value, BatchingContext<C>>>>
    CapturingContext for BatchingContext<C>
{
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

impl<C: CapturingContext<Operation: Clone + DifferentiableOperation<C>> + Zero<C::Value>> CapturingContext
    for DifferentiationContext<C>
{
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }
}

/// A [`Program`] paired with the concrete runtime values referenced by its captured constants. The [`Program`] remains
/// independent of the concrete capture data: values of type `V` live only in [`captures`](Self::captures), while its
/// constant atoms carry lifetime-free [`CaptureReference`]s into that table. [`new`](Self::new), the sole construction
/// path, validates that every reference names an existing capture with the same type, and so every [`ClosedProgram`]
/// upholds the capture-table invariant before it can be interpreted or transformed.
pub struct ClosedProgram<
    V: Value,
    O,
    Input: Parameterized<CaptureReference<V::Type>>,
    Output: Parameterized<CaptureReference<V::Type>>,
> {
    /// [`Program`] whose constants are [`CaptureReference`]s.
    program: Program<CaptureReference<V::Type>, O, Input, Output>,

    /// Captured values referenced by [`CaptureReference`] indices in [`Self::program`].
    captures: Vec<V>,
}

impl<
    V: Value,
    O: Operation<V::Type>,
    Input: Parameterized<CaptureReference<V::Type>>,
    Output: Parameterized<CaptureReference<V::Type>>,
> ClosedProgram<V, O, Input, Output>
{
    /// Creates a [`ClosedProgram`] from a capture-referenced `program` and its concrete `captures`, validating that
    /// every [`CaptureReference`] in the program references an existing capture whose type matches the type stored
    /// in the reference. This is the sole construction path and both the program and its capture table are immutable
    /// after construction, and so every [`ClosedProgram`] upholds this capture-table invariant for its entire lifetime
    /// and no separate re-validation is ever needed.
    pub fn new(
        program: Program<CaptureReference<V::Type>, O, Input, Output>,
        captures: Vec<V>,
    ) -> Result<Self, ProgramError> {
        for (atom_index, atom) in program.atoms().iter().enumerate() {
            let Atom::Constant(value) = atom else {
                continue;
            };
            let capture = captures.get(value.index()).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "captured constant atom %{atom_index} references missing capture #{}",
                    value.index(),
                ))
            })?;
            let expected_type = value.r#type();
            let actual_type = capture.r#type();
            if expected_type.as_ref() != actual_type.as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "captured constant atom %{atom_index} references capture #{} with type {}, \
                     but the atom has type {}",
                    value.index(),
                    actual_type,
                    expected_type,
                )));
            }
        }
        Ok(Self { program, captures })
    }

    /// Returns the underlying [`Program`].
    #[inline]
    pub fn program(&self) -> &Program<CaptureReference<V::Type>, O, Input, Output> {
        &self.program
    }

    /// Returns the underlying captures table.
    #[inline]
    pub fn captures(&self) -> &[V] {
        self.captures.as_slice()
    }

    /// Removes captures that no [`Atom`] of the underlying [`Program`] references and re-indexes the survivors into
    /// a contiguous capture table. This is _dead-capture elimination_: any capture whose index never appears in an
    /// [`Atom::Constant`] is dropped, and the remaining captures are renumbered to occupy `0..captures().len()` in
    /// their original relative order. The program structure (i.e., inputs, instructions, and outputs) is preserved
    /// and only constant atoms are rewritten to carry their new indices. This function consumes `self` so that the
    /// surviving captures, the instruction operations, and the input/output structures can all be moved into the
    /// rebuilt program instead of being cloned.
    pub fn without_unused_captures(self) -> Result<Self, ProgramError> {
        let Self { program, captures } = self;
        let Program { atoms, input_ids, output_ids, instructions, input_structure, output_structure, .. } = program;

        // Mark the captures that at least one constant atom references. Indexing into `is_referenced`
        // cannot fail because `new` validated every capture reference at construction time.
        let mut is_referenced = vec![false; captures.len()];
        for atom in &atoms {
            if let Atom::Constant(capture_reference) = atom {
                is_referenced[capture_reference.index()] = true;
            }
        }

        // Move the referenced captures into a contiguous table that preserves their original relative order,
        // recording each survivor's new index. Captures that are never referenced are dropped here.
        let mut capture_index_map = vec![None; captures.len()];
        let mut filtered_captures = Vec::new();
        for (source_index, capture) in captures.into_iter().enumerate() {
            if is_referenced[source_index] {
                capture_index_map[source_index] = Some(filtered_captures.len());
                filtered_captures.push(capture);
            }
        }

        /// Materializes the rebuilt atom identifiers for `atom_ids` in `builder`, memoizing the results in
        /// `mapped_atoms` and lazily re-adding referenced capture constants with their re-indexed references.
        /// The `capture_index_map` slot lookups cannot fail because the marking pass in
        /// [`ClosedProgram::without_unused_captures`] assigns a slot to every capture
        /// referenced by any constant atom.
        fn map_atoms<T: Type, O: Operation<T>>(
            atoms: &[Atom<CaptureReference<T>>],
            capture_index_map: &[Option<usize>],
            builder: &mut ProgramBuilder<CaptureReference<T>, O>,
            mapped_atoms: &mut [Option<AtomId>],
            atom_ids: Vec<AtomId>,
        ) -> Result<Vec<AtomId>, ProgramError> {
            atom_ids
                .into_iter()
                .map(|atom_id| {
                    if let Some(mapped) = mapped_atoms.get(atom_id.index()).copied().flatten() {
                        return Ok(mapped);
                    }
                    match atoms.get(atom_id.index()) {
                        Some(Atom::Constant(capture_reference)) => {
                            let index = capture_index_map[capture_reference.index()].unwrap();
                            let mapped = builder
                                .add_constant(CaptureReference::new(index, capture_reference.r#type().into_owned()));
                            mapped_atoms[atom_id.index()] = Some(mapped);
                            Ok(mapped)
                        }
                        Some(Atom::Variable(_)) => Err(ProgramError::MalformedProgram(format!(
                            "variable atom {atom_id} has no mapped input or instruction output",
                        ))),
                        None => Err(ProgramError::UnboundAtomId { id: atom_id }),
                    }
                })
                .collect()
        }

        // Rebuild the underlying program with the same structure, rewriting only the constant atoms. `mapped_atoms`
        // tracks the rebuilt `AtomId` of every source atom so that an atom shared across instructions maps to a single
        // rebuilt atom instead of being duplicated.
        let mut builder = ProgramBuilder::new();
        let mut mapped_atoms = vec![None; atoms.len()];

        // Program inputs must be re-added before any instruction is replayed so that instruction operands referencing
        // them resolve through `mapped_atoms` instead of being treated as unmapped atoms.
        for input_id in input_ids {
            let input_index = input_id.index();
            let input = atoms.get(input_index).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(input_type) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            mapped_atoms[input_index] = Some(builder.add_input(input_type.clone()));
        }

        // Replay the instructions in order, mapping their operands and recording their rebuilt outputs so that later
        // instructions (and the program outputs) can resolve them. The operations are moved into the rebuilt program.
        for instruction in instructions {
            let (operation, instruction_inputs, instruction_outputs) = instruction.into_parts();
            let inputs = map_atoms(
                atoms.as_slice(),
                capture_index_map.as_slice(),
                &mut builder,
                mapped_atoms.as_mut_slice(),
                instruction_inputs,
            )?;
            let outputs = builder.add_instruction(operation, inputs)?.to_vec();
            check_count!("output", outputs, instruction_outputs.len(), ProgramError);
            for (output, rebuilt_output) in instruction_outputs.into_iter().zip(outputs) {
                let mapped_atom =
                    mapped_atoms.get_mut(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?;
                *mapped_atom = Some(rebuilt_output);
            }
        }

        // Program outputs may be inputs, instruction outputs, or captured constants, and so they are resolved
        // through the same atom mapping as instruction operands.
        let output_ids = map_atoms(
            atoms.as_slice(),
            capture_index_map.as_slice(),
            &mut builder,
            mapped_atoms.as_mut_slice(),
            output_ids,
        )?;

        let program = builder.build(output_ids, input_structure, output_structure)?;

        // Constructing through `new` re-validates the rewritten references against the pruned capture table.
        Self::new(program, filtered_captures)
    }
}

impl<
    V: Value,
    O: Clone,
    Input: Parameterized<CaptureReference<V::Type>>,
    Output: Parameterized<CaptureReference<V::Type>>,
> Clone for ClosedProgram<V, O, Input, Output>
{
    #[inline]
    fn clone(&self) -> Self {
        Self { program: self.program.clone(), captures: self.captures.clone() }
    }
}

impl<V: Value, O, Input: Parameterized<CaptureReference<V::Type>>, Output: Parameterized<CaptureReference<V::Type>>>
    Debug for ClosedProgram<V, O, Input, Output>
{
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ClosedProgram")
            .field("captures", &self.captures.len())
            .finish_non_exhaustive()
    }
}

// TODO(eaplatanios): Review from here onwards.

impl<
    V: Value,
    O: Clone + Operation<V::Type>,
    Input: Parameterized<CaptureReference<V::Type>>,
    Output: Parameterized<CaptureReference<V::Type>>,
> ClosedProgram<V, O, Input, Output>
{
    /// Returns a [`Program`] where the captures have been lifted into explicit leading inputs that are followed by the
    /// original program inputs. The [`ClosedProgram`] itself is unchanged: the returned program is a derived view
    /// used by compilation, which supplies arguments in `[captures..., public inputs...]` order.
    pub fn to_program_with_lifted_captures(
        &self,
    ) -> Result<
        Program<CaptureReference<V::Type>, O, Vec<CaptureReference<V::Type>>, Vec<CaptureReference<V::Type>>>,
        ProgramError,
    > {
        fn map_atom<T: Type>(
            atoms: &[Atom<CaptureReference<T>>],
            mapped_atoms: &[Option<AtomId>],
            capture_inputs: &[AtomId],
            atom_id: AtomId,
        ) -> Result<AtomId, ProgramError> {
            if let Some(mapped) = mapped_atoms.get(atom_id.index()).copied().flatten() {
                return Ok(mapped);
            }
            match atoms.get(atom_id.index()) {
                Some(Atom::Constant(capture_reference)) => Ok(capture_inputs[capture_reference.index()]),
                Some(Atom::Variable(_)) => Err(ProgramError::MalformedProgram(format!(
                    "variable atom {atom_id} has no mapped input or instruction output",
                ))),
                None => Err(ProgramError::UnboundAtomId { id: atom_id }),
            }
        }

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
            for (source_output, rebuilt_output) in instruction.outputs().iter().copied().zip(outputs) {
                let mapped = mapped_atoms
                    .get_mut(source_output.index())
                    .ok_or(ProgramError::UnboundAtomId { id: source_output })?;
                *mapped = Some(rebuilt_output);
            }
        }

        let output_ids = self
            .program
            .output_ids()
            .iter()
            .copied()
            .map(|output| map_atom(self.program.atoms(), mapped_atoms.as_slice(), capture_inputs.as_slice(), output))
            .collect::<Result<Vec<_>, _>>()?;
        builder.build::<Vec<CaptureReference<V::Type>>, Vec<CaptureReference<V::Type>>>(
            output_ids,
            vec![Placeholder; self.captures.len() + self.program.input_ids().len()],
            vec![Placeholder; self.program.output_ids().len()],
        )
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_count;
    use crate::operations::Operation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::types::{DataType, TypeError};

    use super::*;

    // TODO(eaplatanios): `test_closed_program_without_unused_captures`,
    //  `test_closed_program_to_program_with_lifted_captures`.

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

    impl<C> InterpretableOperation<Scalar, C> for TestAddOperation {
        fn interpret(&self, _context: &C, inputs: &[Scalar]) -> Result<Vec<Scalar>, ProgramError> {
            check_count!("input", inputs, 2, ProgramError);
            Ok(vec![inputs[0] + inputs[1]])
        }
    }

    fn closed_add_program()
    -> ClosedProgram<Scalar, TestAddOperation, Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>> {
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
        ClosedProgram::new(program, vec![Scalar::from(3.0)]).unwrap()
    }

    #[test]
    fn test_closed_program_lifts_captures_as_leading_inputs() {
        let program = closed_add_program();
        let lifted = program.to_program_with_lifted_captures().unwrap();

        assert_eq!(lifted.input_ids().len(), 2);
        assert!(lifted.atoms().iter().all(|atom| !atom.is_constant()));
        let output = lifted
            .interpret_with(
                vec![Scalar::from(3.0), Scalar::from(2.0)],
                |_, constant| Ok::<_, ProgramError>(Scalar::from(constant.index() as f64)),
                |instruction, inputs| instruction.operation().interpret(&EagerContext::<Scalar>::new(), inputs),
            )
            .unwrap();

        assert_eq!(output, vec![5.0]);
    }

    #[test]
    fn test_closed_program_rejects_invalid_capture_references() {
        let mut builder = ProgramBuilder::<CaptureReference<DataType>, TestAddOperation>::new();
        let capture = builder.add_constant(CaptureReference::new(1, DataType::F64));
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                vec![capture],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            ClosedProgram::new(program, vec![Scalar::from(3.0)]),
            Err(ProgramError::MalformedProgram(message))
                if message == "captured constant atom %0 references missing capture #1",
        ));

        // A reference whose declared type differs from its capture's runtime type is rejected at the same boundary.
        let mut builder = ProgramBuilder::<CaptureReference<DataType>, TestAddOperation>::new();
        let capture = builder.add_constant(CaptureReference::new(0, DataType::I64));
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                vec![capture],
                Vec::<Placeholder>::new(),
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            ClosedProgram::new(program, vec![Scalar::from(3.0)]),
            Err(ProgramError::MalformedProgram(message))
                if message
                    == "captured constant atom %0 references capture #0 with type f64, but the atom has type i64",
        ));
    }

    #[test]
    fn test_without_unused_captures_drops_dead_capture_and_reindexes() {
        // The program references only capture #1; capture #0 (value 3.0) is dead.
        let mut builder = ProgramBuilder::<CaptureReference<DataType>, TestAddOperation>::new();
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
        let program = ClosedProgram::new(program, vec![Scalar::from(3.0), Scalar::from(99.0)]).unwrap();

        let pruned = program.without_unused_captures().unwrap();

        // The dead capture is dropped and the survivor is reindexed to a contiguous table.
        assert_eq!(pruned.captures(), &[99.0]);
        let capture_indices = pruned
            .program()
            .atoms()
            .iter()
            .filter_map(|atom| atom.as_constant().map(|reference| reference.index()))
            .collect::<Vec<_>>();
        assert_eq!(capture_indices, vec![0]);

        let output = pruned
            .program()
            .interpret_with(
                vec![Scalar::from(2.0)],
                |_, reference| Ok::<_, ProgramError>(pruned.captures()[reference.index()]),
                |instruction, inputs| instruction.operation().interpret(&EagerContext::<Scalar>::new(), inputs),
            )
            .unwrap();
        assert_eq!(output, vec![101.0]);
    }
}
