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
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Region, Value};
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

    // TODO(eaplatanios): Delete this method (together with its overrides, the `TracingContext` nested-capture flag
    //  and its `note_nested_capture`/`has_nested_captures` accessors,
    //  and the pruning skip in `compilation/function.rs`) once the operation-family migrations in phases 4-6 make
    //  capture discovery recursively region-aware, at which point nested-only capture references become visible to
    //  pruning and the conservative guard is no longer needed.
    /// Registers `value` like [`capture`](Self::capture) for a registration that originates inside a nested trace,
    /// whose builder is where the returned reference constant will be staged. The reference then lives only inside a
    /// nested payload program that top-level-only capture pruning cannot see, so the capture-reference root context
    /// overrides this to additionally record that pruning would be unsafe, while stackable transform contexts
    /// forward it to their parent like [`capture`](Self::capture). The default forwards to [`capture`](Self::capture)
    /// directly, which also makes nested-in-nested registration chain correctly through [`NestedTracingContext`]'s
    /// delegating [`capture`](Self::capture). This is part of the temporary conservative capture-pruning guard for the
    /// first-class-program-regions migration and is deleted once capture discovery is recursively region-aware for
    /// every operation family.
    #[inline]
    fn capture_in_nested_trace(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.capture(value)
    }
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

    #[inline]
    fn capture_in_nested_trace(&self, value: C) -> Result<Self::Constant, ProgramError> {
        self.note_nested_capture();
        self.capture(value)
    }
}

impl<C: CapturingContext> CapturingContext for NestedTracingContext<C> {
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        // The returned reference constant is staged into this nested trace's builder, so the registration takes the
        // nested-flavored path. Top-level-only capture pruning cannot see the nested reference.
        self.parent().capture_in_nested_trace(value)
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

    #[inline]
    fn capture_in_nested_trace(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture_in_nested_trace(value)
    }
}

impl<C: CapturingContext<Type = ArrayType, Operation: BatchableOperation<C>>> CapturingContext for BatchingContext<C> {
    type Capture = C::Capture;

    #[inline]
    fn capture(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture(value)
    }

    #[inline]
    fn capture_in_nested_trace(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture_in_nested_trace(value)
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

    #[inline]
    fn capture_in_nested_trace(&self, value: Self::Capture) -> Result<Self::Constant, ProgramError> {
        self.parent().capture_in_nested_trace(value)
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
        // Every region in the arena participates in the single capture scope, so validation walks all of them.
        for region in program.regions() {
            for (atom_index, atom) in region.atoms().iter().enumerate() {
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
        // TODO(eaplatanios): Fix this as part of later phase of the first-class program regions plan.
        if program.regions().len() != 1 {
            return Err(ProgramError::MalformedProgram(
                "multi-region programs are not yet supported by this transformation".to_string(),
            ));
        }
        let Program { regions, input_structure, output_structure, .. } = program;
        let Region { atoms, input_ids, output_ids, instructions } = regions.into_iter().next().unwrap();

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
            let (operation, instruction_inputs, instruction_outputs, regions) = instruction.into_parts();
            if !regions.is_empty() {
                // TODO(eaplatanios): Fix this as part of later phase of the first-class program regions plan.
                // This is unreachable by construction. The single-region check above already rejected multi-region
                // programs and a single-region program has no sealed regions for an instruction to attach.
                return Err(ProgramError::MalformedProgram(
                    "multi-region programs are not yet supported by this transformation".to_string(),
                ));
            }
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

    /// Returns a [`Program`] where the captures have been lifted into explicit leading inputs that are followed by
    /// the original program inputs. The [`ClosedProgram`] itself is unchanged. The returned program is a derived
    /// view used by compilation, which supplies arguments in `[captures..., public inputs...]` order.
    pub fn to_program_with_lifted_captures(
        &self,
    ) -> Result<
        Program<CaptureReference<V::Type>, O, Vec<CaptureReference<V::Type>>, Vec<CaptureReference<V::Type>>>,
        ProgramError,
    >
    where
        O: Clone,
    {
        /// Materializes the rebuilt atom identifiers for `atom_ids`, resolving each atom either through `mapped_atoms`
        /// (which memoizes the rebuilt identifiers of program inputs and instruction outputs) or, for a captured
        /// constant, to the leading capture input that `capture_inputs` records for the capture the constant
        /// references. The `capture_inputs` lookup cannot fail because [`ClosedProgram::new`] validated that
        /// every [`CaptureReference`] names an existing capture, and one leading input exists per capture.
        fn map_atoms<T: Type>(
            atoms: &[Atom<CaptureReference<T>>],
            mapped_atoms: &[Option<AtomId>],
            capture_inputs: &[AtomId],
            atom_ids: &[AtomId],
        ) -> Result<Vec<AtomId>, ProgramError> {
            atom_ids
                .iter()
                .copied()
                .map(|atom_id| {
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
                })
                .collect()
        }

        // Add one leading input per capture, in capture-table order and unconditionally (a capture that no atom
        // references still occupies its input slot), because execution supplies arguments positionally in
        // `[captures..., public inputs...]` order. A constant atom referencing capture `k` resolves to
        // `capture_inputs[k]` during the replay below.
        let mut builder = ProgramBuilder::new();
        let capture_inputs = self
            .captures
            .iter()
            .map(|capture| builder.add_input(capture.r#type().into_owned()))
            .collect::<Vec<_>>();

        // The original program inputs follow the capture inputs and must be re-added before any instruction is replayed
        // so that instruction operands referencing them resolve through `mapped_atoms` instead of being treated as
        // unmapped atoms. `mapped_atoms` tracks the rebuilt `AtomId` of every source atom so that an atom shared across
        // instructions maps to a single rebuilt atom instead of being duplicated.
        let mut mapped_atoms = vec![None; self.program.atoms().len()];
        for input_id in self.program.input_ids().iter().copied() {
            let input =
                self.program.atoms().get(input_id.index()).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
            let Atom::Variable(input_type) = input else {
                return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()));
            };
            mapped_atoms[input_id.index()] = Some(builder.add_input(input_type.clone()));
        }

        // Replay the instructions in order, mapping their operands and recording their rebuilt outputs so that later
        // instructions (and the program outputs) can resolve them. This borrows `self`, and so the operations are
        // cloned into the rebuilt program (cheaply, since operations share their nested programs).
        for instruction in self.program.instructions() {
            let inputs = map_atoms(
                self.program.atoms(),
                mapped_atoms.as_slice(),
                capture_inputs.as_slice(),
                instruction.inputs(),
            )?;
            let outputs = builder.add_instruction(instruction.operation().clone(), inputs)?.to_vec();
            check_count!("output", outputs, instruction.outputs().len(), ProgramError);
            for (source_output, rebuilt_output) in instruction.outputs().iter().copied().zip(outputs) {
                let mapped = mapped_atoms
                    .get_mut(source_output.index())
                    .ok_or(ProgramError::UnboundAtomId { id: source_output })?;
                *mapped = Some(rebuilt_output);
            }
        }

        // Program outputs may be original inputs, instruction outputs, or captured constants,
        // and so they are resolved through the same atom mapping as instruction operands.
        let output_ids = map_atoms(
            self.program.atoms(),
            mapped_atoms.as_slice(),
            capture_inputs.as_slice(),
            self.program.output_ids(),
        )?;

        // The lifted program is flat. Its input and output boundaries are placeholder vectors sized to the
        // `[captures..., public inputs...]` argument convention and the original outputs, respectively.
        builder.build::<Vec<CaptureReference<V::Type>>, Vec<CaptureReference<V::Type>>>(
            output_ids,
            vec![Placeholder; self.captures.len() + self.program.input_ids().len()],
            vec![Placeholder; self.program.output_ids().len()],
        )
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

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::contexts::StagingContext;
    use crate::interpretation::InterpretableOperation;
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::control_flow::WhileOperation;
    use crate::operations::math::AddOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::tracing::{NestedTracingContext, Tracer, TracingContext};
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_closed_program_without_unused_captures() {
        // Construction rejects references to missing captures.
        let mut builder = ProgramBuilder::<CaptureReference<DataType>, ScalarOperation<Scalar>>::new();
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

        // Construction rejects references whose declared type differs from their capture's runtime type.
        let mut builder = ProgramBuilder::<CaptureReference<DataType>, ScalarOperation<Scalar>>::new();
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

        // Pruning drops the dead capture #0 and re-indexes the surviving capture #1 into a contiguous table
        // while preserving the program structure.
        let mut builder = ProgramBuilder::<CaptureReference<DataType>, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let capture = builder.add_constant(CaptureReference::new(1, DataType::F64));
        let output = builder.add_instruction(AddOperation, vec![input, capture]).unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let program = ClosedProgram::new(program, vec![Scalar::from(3.0), Scalar::from(99.0)]).unwrap();
        let pruned = program.without_unused_captures().unwrap();
        assert_eq!(pruned.captures(), &[Scalar::from(99.0)]);
        let capture_indices = pruned
            .program()
            .atoms()
            .iter()
            .filter_map(|atom| atom.as_constant().map(CaptureReference::index))
            .collect::<Vec<_>>();
        assert_eq!(capture_indices, vec![0]);
        assert_eq!(
            pruned.program().to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Pruning a program whose captures are all referenced is an identity transformation.
        let repruned = pruned.clone().without_unused_captures().unwrap();
        assert_eq!(repruned.captures(), pruned.captures());
        assert_eq!(repruned.program().to_string(), pruned.program().to_string());

        // The pruned program interprets equivalently through the re-indexed capture table.
        let output = pruned
            .program()
            .interpret_with(
                vec![Scalar::from(2.0)],
                |_, reference| Ok::<_, ProgramError>(pruned.captures()[reference.index()]),
                |instruction, inputs| {
                    instruction.operation().interpret(&EagerContext::<Scalar, ScalarOperation<Scalar>>::new(), inputs)
                },
            )
            .unwrap();
        assert_eq!(output, vec![Scalar::from(101.0)]);
    }

    #[test]
    fn test_closed_program_to_program_with_lifted_captures() {
        // The program computes `(input + capture#0) + capture#0` through one shared constant atom, and capture #1 is
        // never referenced, so the lift covers shared references and dead captures at once.
        let mut builder = ProgramBuilder::<CaptureReference<DataType>, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let capture = builder.add_constant(CaptureReference::new(0, DataType::F64));
        let sum = builder.add_instruction(AddOperation, vec![input, capture]).unwrap()[0];
        let output = builder.add_instruction(AddOperation, vec![sum, capture]).unwrap()[0];
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let program = ClosedProgram::new(program, vec![Scalar::from(3.0), Scalar::from(7.0)]).unwrap();

        // Captures become leading inputs in capture-table order (one per capture, dead ones included), followed by
        // the original program input, and no captured-constant atoms remain. The shared constant atom maps to a
        // single capture input.
        let lifted = program.to_program_with_lifted_captures().unwrap();
        assert_eq!(lifted.input_ids().len(), 3);
        assert!(lifted.atoms().iter().all(|atom| !atom.is_constant()));
        assert_eq!(
            lifted.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64, %2:f64 .
                let %3:f64 = add %2 %0
                    %4:f64 = add %3 %0
                in (%4)
            "}
            .trim_end(),
        );

        // The closed program is unchanged: the lift is a derived view.
        assert_eq!(program.captures().len(), 2);
        assert_eq!(program.program().atoms().iter().filter(|atom| atom.is_constant()).count(), 1);

        // The lifted program interprets with arguments supplied in `[captures..., public inputs...]` order.
        let output = lifted
            .interpret_with::<Scalar, ProgramError, _, _>(
                vec![Scalar::from(3.0), Scalar::from(7.0), Scalar::from(2.0)],
                |_, _| unreachable!("the lifted program contains no captured-constant atoms"),
                |instruction, inputs| {
                    instruction.operation().interpret(&EagerContext::<Scalar, ScalarOperation<Scalar>>::new(), inputs)
                },
            )
            .unwrap();
        assert_eq!(output, vec![Scalar::from(8.0)]);
    }

    // TODO(eaplatanios): Review this test.
    #[test]
    fn test_capturing_context_capture_in_nested_trace_marks_the_root_context_and_preserves_nested_only_captures() {
        // Phase 1 guard pin for the first-class-program-regions plan: a capture registered through a nested trace
        // is referenced only by a constant inside a nested payload program, which top-level-only pruning cannot
        // see. The nested delegation marks the root context, and pruning is skipped for such traces (mirroring the
        // jit staging path), so the capture and its indices survive intact. The unguarded pruning behavior is also
        // pinned below to document why the guard exists until capture discovery becomes recursively region-aware.
        let root =
            TracingContext::<CaptureReference<DataType>, ScalarOperation<CaptureReference<DataType>>, Scalar>::new();
        let state = root.input(DataType::F64);

        // The condition is set to `state < state` (always false) and it is traced as a nested payload program.
        let (_, condition) = NestedTracingContext::trace(
            root.clone(),
            |inputs: Vec<Tracer<_>>| {
                inputs[0].context().bind(
                    CompareOperation::new(ComparisonDirection::LessThan),
                    &[],
                    &[],
                    &[inputs[0].clone(), inputs[0].clone()],
                )
            },
            vec![DataType::F64],
        )
        .unwrap();

        // The body is set to `state + capture#0`, with the capture registered through the *nested* trace,
        // and so its reference constant is staged only into the nested body program.
        let (_, body) = NestedTracingContext::trace(
            root.clone(),
            |inputs: Vec<Tracer<_>>| {
                let context = inputs[0].context().clone();
                let reference = context.capture(Scalar::from(3.0))?;
                let captured = StagingContext::constant(&context, reference);
                context.bind(AddOperation, &[], &[], &[inputs[0].clone(), captured])
            },
            vec![DataType::F64],
        )
        .unwrap();
        let operation = WhileOperation::new(condition, body).unwrap();
        let outputs = root
            .bind(ScalarOperation::While(Box::new(operation)), &[], &[], std::slice::from_ref(&state))
            .unwrap();

        // The nested registration marked the root context through the delegation chain.
        assert!(root.has_nested_captures());

        // The top-level program holds no capture-reference atoms. The only reference lives in the while body.
        let output_ids = outputs.iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>().unwrap();
        let builder = root.builder().borrow().clone();
        let captures = root.captures().borrow().clone();
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                output_ids,
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let closed = ClosedProgram::new(program, captures).unwrap();
        assert!(closed.program().atoms().iter().all(|atom| !atom.is_constant()));
        assert_eq!(closed.captures(), &[Scalar::from(3.0)]);

        // Unguarded pruning would silently drop the nested-only capture (the bug the guard prevents).
        // The guarded path keeps the closed program untouched instead.
        let pruned = closed.clone().without_unused_captures().unwrap();
        assert_eq!(pruned.captures(), &[] as &[Scalar]);
        let guarded = if root.has_nested_captures() { closed } else { closed.without_unused_captures().unwrap() };
        assert_eq!(guarded.captures(), &[Scalar::from(3.0)]);
    }
}
