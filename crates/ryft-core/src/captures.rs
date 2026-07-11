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
use crate::programs::{Atom, AtomId, Instruction, Program, ProgramBuilder, ProgramError, Value};
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
/// constant atoms carry lifetime-free [`CaptureReference`]s into that table. [`new`](Self::new) validates that every
/// reference names an existing capture with the same type, and so all public construction paths establish the capture
/// table invariant before a [`ClosedProgram`] can be interpreted or transformed.
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

// TODO(eaplatanios): Review from here onwards.

impl<V: Value, O: Operation<V::Type>, Input: Parameterized<CaptureReference<V::Type>>, Output: Parameterized<CaptureReference<V::Type>>>
    ClosedProgram<V, O, Input, Output>
{
    /// Creates a [`ClosedProgram`] from a capture-referenced `program` and its concrete `captures`, validating that
    /// every constant atom references an existing capture whose type matches the type stored in the reference.
    pub fn new(
        program: Program<CaptureReference<V::Type>, O, Input, Output>,
        captures: Vec<V>,
    ) -> Result<Self, ProgramError> {
        let closed_program = Self { program, captures };
        closed_program.validate_capture_references()?;
        Ok(closed_program)
    }

    /// Validates that every captured-constant atom references an existing capture with the same type.
    pub fn validate_capture_references(&self) -> Result<(), ProgramError> {
        for (atom_index, atom) in self.program.atoms().iter().enumerate() {
            let Atom::Constant(capture_reference) = atom else {
                continue;
            };
            let capture = self.captures.get(capture_reference.index()).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "captured constant atom %{atom_index} references missing capture #{}",
                    capture_reference.index(),
                ))
            })?;
            let expected_type = capture_reference.r#type();
            let actual_type = capture.r#type();
            if expected_type.as_ref() != actual_type.as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "captured constant atom %{atom_index} references capture #{} with type {}, \
                     but the atom has type {}",
                    capture_reference.index(),
                    actual_type,
                    expected_type,
                )));
            }
        }
        Ok(())
    }

    /// Returns the staged program.
    #[inline]
    pub fn program(&self) -> &Program<CaptureReference<V::Type>, O, Input, Output> {
        &self.program
    }

    /// Returns the captured runtime values.
    #[inline]
    pub fn captures(&self) -> &[V] {
        self.captures.as_slice()
    }

    /// Consumes this [`ClosedProgram`] and returns its staged program and capture table, in that order.
    #[inline]
    pub fn into_parts(self) -> (Program<CaptureReference<V::Type>, O, Input, Output>, Vec<V>) {
        (self.program, self.captures)
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

impl<
    V: Value,
    O: Operation<V::Type>,
    Input: Parameterized<CaptureReference<V::Type>>,
    Output: Parameterized<CaptureReference<V::Type>>,
> ClosedProgram<V, O, Input, Output>
{
    /// Validates capture references that will be supplied as the leading inputs of an opened captured program.
    ///
    /// Capture input indices belong to the caller's capture table and may differ from this program's local capture
    /// indices. This validates the positional arity and types that matter after captures are opened as leading flat
    /// inputs.
    pub fn validate_capture_inputs(&self, capture_inputs: &[CaptureReference<V::Type>]) -> Result<(), ProgramError> {
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
        RuntimeValue: Clone,
        Error: From<ProgramError>,
        LiftCapture: FnMut(usize, &V) -> Result<RuntimeValue, Error>,
        InterpretInstruction: FnMut(&Instruction<O>, &[RuntimeValue]) -> Result<Vec<RuntimeValue>, Error>,
    >(
        &self,
        inputs: Vec<RuntimeValue>,
        mut lift_capture: LiftCapture,
        interpret_instruction: InterpretInstruction,
    ) -> Result<Vec<RuntimeValue>, Error> {
        self.validate_capture_references().map_err(Error::from)?;
        self.program.interpret_with(
            inputs,
            |_, capture_reference| {
                let capture = &self.captures[capture_reference.index()];
                lift_capture(capture_reference.index(), capture)
            },
            interpret_instruction,
        )
    }
}

impl<
    V: Value,
    O: Clone + Operation<V::Type>,
    Input: Parameterized<CaptureReference<V::Type>>,
    Output: Parameterized<CaptureReference<V::Type>>,
> ClosedProgram<V, O, Input, Output>
{
    /// Returns a flat program where captures are explicit leading inputs followed by the original program inputs.
    pub fn open_captures_as_inputs(
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

impl<
    V: Clone + Value,
    O: Clone + Operation<V::Type>,
    Input: Parameterized<CaptureReference<V::Type>>,
    Output: Parameterized<CaptureReference<V::Type>>,
> ClosedProgram<V, O, Input, Output>
{
    /// Rebuilds this program with its capture table replaced by `rebuilt_captures` and every
    /// [`CaptureReference`] reindexed through `capture_index_map`.
    ///
    /// The program structure (inputs, instructions, and outputs) is preserved; only constant atoms are rewritten. A
    /// surviving [`CaptureReference`] at source index `source` becomes a reference to `capture_index_map[source]`.
    /// Callers must supply a map that assigns a rebuilt index to every capture still referenced by the program, and
    /// order `rebuilt_captures` to match those indices. Constant atoms whose source index maps to [`None`] must be
    /// unreachable (no atom may reference a dropped capture); reaching one is a malformed-program error.
    fn reindex_captures(
        &self,
        capture_index_map: &[Option<usize>],
        rebuilt_captures: Vec<V>,
    ) -> Result<Self, ProgramError> {
        self.validate_capture_references()?;
        let mut builder = ProgramBuilder::new();
        let mut mapped_atoms = vec![None; self.program.atoms().len()];

        // Materializes the rebuilt atom identifier for `atom_id`, lazily re-adding referenced capture constants with
        // their reindexed reference so atoms shared across instructions map to a single rebuilt constant.
        let map_atom = |builder: &mut ProgramBuilder<CaptureReference<V::Type>, O>,
                        mapped_atoms: &mut [Option<AtomId>],
                        atom_id: AtomId|
         -> Result<AtomId, ProgramError> {
            if let Some(mapped) = mapped_atoms.get(atom_id.index()).copied().flatten() {
                return Ok(mapped);
            }
            match self.program.atoms().get(atom_id.index()) {
                Some(Atom::Constant(capture_reference)) => {
                    let rebuilt_index =
                        capture_index_map.get(capture_reference.index()).copied().flatten().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "captured constant atom {atom_id} references capture #{} that has no reindexed slot",
                                capture_reference.index(),
                            ))
                        })?;
                    let rebuilt_capture = rebuilt_captures.get(rebuilt_index).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "captured constant atom {atom_id} maps to missing capture #{rebuilt_index}",
                        ))
                    })?;
                    if capture_reference.r#type().as_ref() != rebuilt_capture.r#type().as_ref() {
                        return Err(ProgramError::MalformedProgram(format!(
                            "captured constant atom {atom_id} has type {}, \
                             but its reindexed capture #{rebuilt_index} has type {}",
                            capture_reference.r#type(),
                            rebuilt_capture.r#type(),
                        )));
                    }
                    let mapped = builder
                        .add_constant(CaptureReference::new(rebuilt_index, capture_reference.r#type().into_owned()));
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
            .map(|output| map_atom(&mut builder, mapped_atoms.as_mut_slice(), output))
            .collect::<Result<Vec<_>, _>>()?;
        let program = builder.build::<Input, Output>(
            output_ids,
            self.program.input_structure().clone(),
            self.program.output_structure().clone(),
        )?;
        Self::new(program, rebuilt_captures)
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
        let mut rebuilt_captures = Vec::new();
        for atom in self.program.atoms() {
            let Atom::Constant(capture_reference) = atom else {
                continue;
            };
            let source_index = capture_reference.index();
            if capture_index_map.get(source_index).copied().flatten().is_some() {
                continue;
            }
            let capture = self.captures.get(source_index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "captured constant atom references missing capture #{source_index}",
                ))
            })?;
            capture_index_map[source_index] = Some(rebuilt_captures.len());
            rebuilt_captures.push(capture.clone());
        }
        self.reindex_captures(capture_index_map.as_slice(), rebuilt_captures)
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
        let mut rebuilt_captures: Vec<V> = Vec::new();
        for atom in self.program.atoms() {
            let Atom::Constant(capture_reference) = atom else {
                continue;
            };
            let source_index = capture_reference.index();
            if capture_index_map.get(source_index).copied().flatten().is_some() {
                continue;
            }
            let capture = self.captures.get(source_index).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "captured constant atom references missing capture #{source_index}",
                ))
            })?;
            // Value equality alone is insufficient: a value representation may deliberately compare equal across
            // distinct runtime types. Only captures with equal values *and* equal types are interchangeable.
            let capture_type = capture.r#type();
            let rebuilt_index = match rebuilt_captures
                .iter()
                .position(|existing| existing.r#type().as_ref() == capture_type.as_ref() && existing == capture)
            {
                Some(existing_index) => existing_index,
                None => {
                    rebuilt_captures.push(capture.clone());
                    rebuilt_captures.len() - 1
                }
            };
            capture_index_map[source_index] = Some(rebuilt_index);
        }
        self.reindex_captures(capture_index_map.as_slice(), rebuilt_captures)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fmt::{Display, Formatter};

    use pretty_assertions::assert_eq;

    use ryft_macros::Parameter;

    use crate::backends::scalars::Scalar;
    use crate::contexts::EagerContext;
    use crate::interpretation::InterpretableOperation;
    use crate::macros::check_count;
    use crate::operations::Operation;
    use crate::parameters::Placeholder;
    use crate::programs::{ProgramBuilder, ProgramError};
    use crate::types::{DataType, TypeError};

    use super::*;

    /// Test value whose equality deliberately ignores its runtime type. It verifies that capture deduplication treats
    /// type equality as a separate part of capture interchangeability instead of trusting value equality alone.
    #[derive(Clone, Debug, Parameter)]
    struct TypeBlindCapture {
        identity: usize,
        r#type: DataType,
    }

    impl PartialEq for TypeBlindCapture {
        #[inline]
        fn eq(&self, other: &Self) -> bool {
            self.identity == other.identity
        }
    }

    impl Display for TypeBlindCapture {
        #[inline]
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "capture({})", self.identity)
        }
    }

    impl Typed for TypeBlindCapture {
        type Type = DataType;

        #[inline]
        fn r#type(&self) -> Cow<'_, DataType> {
            Cow::Borrowed(&self.r#type)
        }
    }

    impl Value for TypeBlindCapture {
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
    fn test_closed_program_interprets_through_capture_table() {
        let program = closed_add_program();

        let output = program
            .interpret_with_captures(
                vec![Scalar::from(2.0)],
                |_, capture| Ok::<_, ProgramError>(*capture),
                |instruction, inputs| instruction.operation().interpret(&EagerContext::<Scalar>::new(), inputs),
            )
            .unwrap();

        assert_eq!(output, vec![5.0]);
    }

    #[test]
    fn test_closed_program_opens_captures_as_leading_inputs() {
        let program = closed_add_program();
        let opened = program.open_captures_as_inputs().unwrap();

        assert_eq!(opened.input_ids().len(), 2);
        assert!(opened.atoms().iter().all(|atom| !atom.is_constant()));
        let output = opened
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
    fn test_closed_program_into_parts() {
        let closed_program = closed_add_program();
        let (program, captures) = closed_program.into_parts();

        assert_eq!(captures, vec![Scalar::from(3.0)]);
        assert_eq!(program.input_ids().len(), 1);
        assert_eq!(program.output_ids().len(), 1);
        assert_eq!(program.instructions().len(), 1);
    }

    #[test]
    fn test_prune_unused_captures_drops_dead_capture_and_reindexes() {
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
                |instruction, inputs| instruction.operation().interpret(&EagerContext::<Scalar>::new(), inputs),
            )
            .unwrap();
        assert_eq!(output, vec![101.0]);
    }

    #[test]
    fn test_deduplicate_captures_merges_equal_captures() {
        // Both captures hold the same value (7) and are referenced by two distinct constant atoms.
        let mut builder = ProgramBuilder::<CaptureReference<DataType>, TestAddOperation>::new();
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
        let program = ClosedProgram::new(program, vec![Scalar::from(7_i64), Scalar::from(7_i64)]).unwrap();

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

        // Equal values of different types are not interchangeable and therefore remain separate captures.
        let mut builder = ProgramBuilder::<CaptureReference<DataType>, TestAddOperation>::new();
        let floating = builder.add_constant(CaptureReference::new(0, DataType::F64));
        let integer = builder.add_constant(CaptureReference::new(1, DataType::I64));
        let program = builder
            .build::<Vec<CaptureReference<DataType>>, Vec<CaptureReference<DataType>>>(
                vec![floating, integer],
                Vec::<Placeholder>::new(),
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let closed_program = ClosedProgram::new(
            program,
            vec![
                TypeBlindCapture { identity: 7, r#type: DataType::F64 },
                TypeBlindCapture { identity: 7, r#type: DataType::I64 },
            ],
        )
        .unwrap();
        let deduplicated = closed_program.deduplicate_captures().unwrap();
        assert_eq!(deduplicated.captures().len(), 2);
        let capture_indices = deduplicated
            .program()
            .atoms()
            .iter()
            .filter_map(|atom| atom.as_constant().map(CaptureReference::index))
            .collect::<Vec<_>>();
        assert_eq!(capture_indices, vec![0, 1]);
        deduplicated.validate_capture_references().unwrap();
    }
}
