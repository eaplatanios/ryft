use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;

use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::Placeholder;
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::types::{Type, Typed};

/// State of a [`Value`] during partial evaluation. A [`PartialValue`] is the value domain the partial evaluator
/// interprets a [`Program`] over. Every [`Atom`] and every intermediate result is either [`Known`](Self::Known)
/// (i.e., a concrete value available now) or [`Unknown`](Self::Unknown) (i.e., only its [`Type`] is available until
/// the residual program runs). For more information on partial evaluation, refer to the documentation of
/// [`Program::partially_evaluate`].
#[derive(Clone, Debug)]
pub enum PartialValue<T: Type, V: Value<T>> {
    /// [`Value`] that is fully known at partial-evaluation time and can be folded forward.
    Known(V),

    /// [`Value`] that is not known until the residual program runs and only its [`Type`] is known.
    Unknown(T),
}

impl<T: Type, V: Value<T>> PartialValue<T, V> {
    /// Returns `true` if this value is [`Known`](Self::Known).
    #[inline]
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Returns `true` if this value is [`Unknown`](Self::Unknown).
    #[inline]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }

    /// Returns the underlying concrete value when this is [`Known`](Self::Known) and [`None`] otherwise.
    #[inline]
    pub fn as_known(&self) -> Option<&V> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unknown(_) => None,
        }
    }
}

impl<T: Type, V: Value<T>> Typed<T> for PartialValue<T, V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Known(value) => value.r#type(),
            Self::Unknown(r#type) => Cow::Borrowed(r#type),
        }
    }
}

// TODO(eaplatanios): Review from here onwards.

/// Source feeding one input of the residual program.
///
/// The residual program's inputs are the original program's surviving unknown inputs followed by the known values
/// (residuals) that its unknown subcomputation consumes.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Clone, Debug)]
pub enum ResidualProgramInput<V> {
    /// Residual input fed by a value that partial evaluation folded to a concrete known residual.
    Known(V),

    /// Residual input fed by an unknown input of the original program, identified by that input's index in the
    /// original program's input list.
    Unknown(usize),
}

/// Source of one output of the partially evaluated program.
///
/// Partial evaluation splits the original outputs into those it could fold to a concrete value now and those that
/// remain computed by the residual program.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Clone, Debug)]
pub enum ResidualProgramOutput<V> {
    /// Output that was folded to a concrete value during partial evaluation.
    Known(V),

    /// Output produced by the residual program, identified by its index into the residual program's outputs.
    Unknown(usize),
}

/// Result of partially evaluating a flat [`Program`].
///
/// To reconstruct the original program's outputs: build the residual program's input vector by mapping each
/// [`inputs`](Self::inputs) entry to either a runtime unknown-input value or its carried known residual, interpret
/// [`program`](Self::program), then read each [`outputs`](Self::outputs) entry as either its folded value or the
/// indexed residual-program output.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Debug)]
pub struct ResidualProgram<T: Type, V: Value<T>, O> {
    /// Residual program over the surviving unknown inputs plus the known residuals, aligned with
    /// [`inputs`](Self::inputs) and producing the unknown outputs in original order.
    pub program: Program<T, V, O, Vec<V>, Vec<V>>,

    /// How to feed each residual-program input, in residual-program input order.
    pub inputs: Vec<ResidualProgramInput<V>>,

    /// Source of each original program output, in original output order.
    pub outputs: Vec<ResidualProgramOutput<V>>,
}

/// Decision for how an operation is handled during [`Program::partially_evaluate`].
///
/// Partial evaluation walks a program with each input classified as either [`Known`](PartialValue::Known) (a concrete
/// value available now) or [`Unknown`](PartialValue::Unknown) (a value that exists only at residual-program run time),
/// folds away everything computable now, and emits a residual program over the unknowns. For each operation it
/// consults that operation's [`PartiallyEvaluatableOperation::partially_evaluate`] rule, which returns one of the
/// variants below.
///
/// Most operations return [`Default`](Self::Default) and defer to the standard policy, which the driver applies per
/// operation from its operands' knowledge: when *all* operands are [`Known`](PartialValue::Known) it **folds** the
/// operation (interprets it now, so its outputs become known values and it leaves no trace in the residual program),
/// and otherwise it **residualizes** the operation unchanged (emits it into the residual program, materializing each
/// known operand as a residual input or an inlined constant). Control-flow operations may override that policy:
///
///   - [`Inline`](Self::Inline) splices a nested program in place of the operation, fed by the operation operands at
///     the listed indices. Partial evaluation then recurses into that program, folding and residualizing its instructions
///     independently. A known-predicate `condition` uses this to inline its taken branch, so the condition vanishes
///     and only the selected branch's work survives.
///   - [`Replace`](Self::Replace) emits a transformed operation in place of this one, each operand fed from a
///     [`ReplaceOperand`] (an original operand reused, or a value the rule folded). An unknown-predicate `condition`,
///     `while`, or `scan` uses this to residualize a rewritten operation over its partially-evaluated branches.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Clone, Debug)]
pub enum OperationPartialEvaluation<T: Type, V: Value<T>, O> {
    /// Defer to the default partial-evaluation policy for this operation: when all of its operands are
    /// [`Known`](PartialValue::Known) the operation is folded by interpretation, and otherwise it is residualized
    /// unchanged into the residual program. This is what most operations return.
    Default,

    /// Replace the operation by inlining this nested program, fed by the operation operands at the listed operand
    /// indices in order. A known-predicate `condition` uses this to inline its selected branch.
    Inline {
        /// Nested program to inline in place of the operation.
        program: Program<T, V, O, Vec<V>, Vec<V>>,

        /// Indices into the operation's operands that feed `program`'s inputs, in input order.
        operand_indices: Vec<usize>,
    },

    /// Residualize a transformed operation in place of this one, fed by the listed operand sources.
    /// An unknown-predicate `condition` uses this to residualize a new `condition` over partially-evaluated branches.
    Replace {
        /// Transformed operation to emit into the residual program.
        operation: O,

        /// What feeds each of `operation`'s operands, in operand order.
        operands: Vec<ReplaceOperand<V>>,
    },
}

/// Source feeding one operand of a [`OperationPartialEvaluation::Replace`] operation.
#[derive(Clone, Debug)]
pub enum ReplaceOperand<V> {
    /// Feed the original operation's operand at this index (its walk value is reused).
    Operand(usize),

    /// Feed this value, which the rule folded during partial evaluation.
    Known(V),
}

/// Operation that can override the default fold/residualize behavior of [`Program::partially_evaluate`].
///
/// Implemented per operation payload and forwarded by the owning operation enum's generated implementation, this
/// trait lets an individual operation decide how partial evaluation treats it. Most operations defer to the default;
/// control-flow operations may override. For example, a known-predicate `condition` inlines its selected branch.
///
/// # Default Behavior
///
/// The default [`partially_evaluate`](Self::partially_evaluate) returns
/// [`OperationPartialEvaluation::Default`], which tells [`Program::partially_evaluate`] to fall back to its default
/// handling of the operation:
///
///   - when *all* of the operation's operands are [`Known`](PartialValue::Known), it **folds** the operation by
///     interpreting it through [`InterpretableOperation::interpret`], so the operation's outputs become known values
///     and the operation contributes nothing to the residual program;
///   - otherwise it **residualizes** the operation unchanged: it emits the operation into the residual program over
///     its operands' residual-program atoms, materializing each known operand as a residual input for a known
///     variable or as an inlined residual-program constant for a literal, so the operation runs at
///     residual-program execution time.
///
/// # Overriding
///
/// Returning [`OperationPartialEvaluation::Inline`] or [`OperationPartialEvaluation::Replace`] overrides this default
/// with the requested rewrite. For example, a `condition` whose predicate is [`Known`](PartialValue::Known) returns
/// [`OperationPartialEvaluation::Inline`] for its selected branch, so the condition disappears from the residual
/// program and only the taken branch's work survives.
///
/// # Type Parameters
///
///   - `O`: Operation family of the residual program and of any inlined nested programs, namely the enum this
///     operation belongs to. It is the operation type of every [`Program`] in an
///     [`OperationPartialEvaluation`] this rule returns.
pub trait PartiallyEvaluatableOperation<T: Type, V: Value<T>, O>: Sized {
    /// Overrides partial evaluation of this operation given its operands' knowledge. Returning
    /// [`OperationPartialEvaluation::Default`] uses the default behavior described in the [trait documentation](Self):
    /// fold the operation when all operands are [`Known`](PartialValue::Known), otherwise residualize it unchanged.
    ///
    /// # Parameters
    ///
    ///   - `context`: Interpretation context available to a custom rule, such as for folding a known sub-result.
    ///   - `operands`: Knowledge state for each of this operation's operands, in operand order.
    fn partially_evaluate(
        &self,
        context: &V::InterpretationContext,
        operands: &[PartialValue<T, V>],
    ) -> Result<OperationPartialEvaluation<T, V, O>, ProgramError> {
        let _ = (context, operands);
        Ok(OperationPartialEvaluation::Default)
    }
}

/// Closed operation families that can recursively partially evaluate nested flat [`Program`]s of themselves.
///
/// This is the partial-evaluation analogue of
/// [`InterpretableProgramOperation`](crate::operations::InterpretableProgramOperation): it names the recursive fixed
/// point that nested-program operations, such as `scan`/`while`/`condition` bodies, use to partially evaluate their
/// bodies without restating the full [`partially_evaluate`](Program::partially_evaluate) bound at every recursive
/// payload boundary.
///
/// Unlike the linearization and transposition witnesses, whose context/operation type parameters grow with each
/// recursion level and must therefore name a fixed point to stop the trait solver from diverging, this witness's
/// parameters `T` and `V` are fixed across recursion. The blanket implementation grounds it in
/// [`InterpretableOperation`], so proving it for a self-containing operation enum (one whose higher-order variants hold
/// `Program`s of itself) reduces to that enum's existing [`InterpretableOperation`] proof and introduces no new
/// recursive obligation.
pub trait PartiallyEvaluatableProgramOperation<T: Type, V: Value<T>>: Operation<T> + Sized {
    /// Partially evaluates a nested flat [`Program`] of this operation family; see [`Program::partially_evaluate`].
    fn partially_evaluate_program(
        program: &Program<T, V, Self, Vec<V>, Vec<V>>,
        context: &V::InterpretationContext,
        inputs: &[PartialValue<T, V>],
    ) -> Result<ResidualProgram<T, V, Self>, ProgramError>;
}

impl<T, V, O> PartiallyEvaluatableProgramOperation<T, V> for O
where
    T: Type,
    V: Value<T>,
    O: Clone + InterpretableOperation<T, V> + PartiallyEvaluatableOperation<T, V, O>,
{
    #[inline]
    fn partially_evaluate_program(
        program: &Program<T, V, Self, Vec<V>, Vec<V>>,
        context: &V::InterpretationContext,
        inputs: &[PartialValue<T, V>],
    ) -> Result<ResidualProgram<T, V, Self>, ProgramError> {
        program.partially_evaluate(context, inputs)
    }
}

/// Walk-time classification of an [`Atom`] during [`Program::partially_evaluate`].
#[derive(Clone, Debug)]
enum Residualized<V> {
    /// Value known at partial-evaluation time.
    Known(V),

    /// Value produced by the residual program, identified by its atom in that program.
    Residual(AtomId),
}

/// Shared residual-build state threaded through the recursive partial-evaluation walk of
/// [`Program::partially_evaluate`].
///
/// It owns the [`ProgramBuilder`] that accumulates the residual program, the interpretation `context` used to fold
/// known subcomputations, and the running list of residual inputs. The recursive [`Self::inline`] walks a flat program
/// (the top-level program, or an inlined nested branch) and returns the walk value of each of its outputs.
struct ResidualBuilder<'context, T: Type, V: Value<T>, O: Operation<T>> {
    /// Builder accumulating the residual program's atoms and instructions.
    builder: ProgramBuilder<T, V, O>,

    /// Interpretation context used to fold instructions whose operands are all known.
    context: &'context V::InterpretationContext,

    /// How to feed each residual-program input, in residual-program input order.
    residual_inputs: Vec<ResidualProgramInput<V>>,
}

impl<'context, T, V, O> ResidualBuilder<'context, T, V, O>
where
    T: Type,
    V: Value<T>,
    O: Clone + InterpretableOperation<T, V> + PartiallyEvaluatableOperation<T, V, O>,
{
    /// Walks `program`'s instructions with `inputs` bound to its input atoms (in input order), folds every all-known
    /// instruction through its own [`InterpretableOperation::interpret`] rule, dispatches each instruction to its
    /// per-operation [`PartiallyEvaluatableOperation::partially_evaluate`] rule, and emits the residual work
    /// into [`self.builder`](Self::builder). Returns the walk value of each program output, in output order.
    ///
    /// A returned [`OperationPartialEvaluation::Inline`] is handled here by recursively walking the inlined program
    /// fed the selected operands, so an operation can rewrite itself into transformed work. For example, a
    /// known-predicate `condition` can inline its selected branch. Program constants are lifted to
    /// [`Residualized::Known`] on first use and rebuilt inline in the residual program when a residualized instruction
    /// consumes them.
    fn inline(
        &mut self,
        program: &Program<T, V, O, Vec<V>, Vec<V>>,
        inputs: Vec<Residualized<V>>,
    ) -> Result<Vec<Residualized<V>>, ProgramError> {
        // Walk-time value of each atom in `program`, populated as the forward pass reaches it.
        let mut values: Vec<Option<Residualized<V>>> = vec![None; program.atoms.len()];
        // Residual-program atom each known atom of `program` has been materialized into, so repeated uses are
        // deduplicated. Indexed by `program`'s atoms (not the residual program's), and local to this walk.
        let mut materialized: Vec<Option<AtomId>> = vec![None; program.atoms.len()];
        for (input_id, value) in program.input_ids.iter().copied().zip(inputs) {
            values[input_id.index()] = Some(value);
        }

        for instruction in program.instructions.iter() {
            // Resolve operands as (original atom, walk value), lifting a program constant to `Known` on first use.
            let mut operands: Vec<(AtomId, Residualized<V>)> = Vec::with_capacity(instruction.inputs().len());
            for operand in instruction.inputs().iter().copied() {
                let value = match values[operand.index()].clone() {
                    Some(value) => value,
                    None => match &program.atoms[operand.index()] {
                        Atom::Constant(constant) => {
                            let value = Residualized::Known(constant.clone());
                            values[operand.index()] = Some(value.clone());
                            value
                        }
                        Atom::Variable(_) => return Err(ProgramError::UnboundAtomId { id: operand }),
                    },
                };
                operands.push((operand, value));
            }

            // Build operand knowledge for the per-operation rule: a residual operand is `Unknown` of its type.
            let operand_knowledge: Vec<PartialValue<T, V>> = operands
                .iter()
                .map(|(atom, value)| match value {
                    Residualized::Known(value) => PartialValue::Known(value.clone()),
                    Residualized::Residual(_) => {
                        PartialValue::Unknown(program.atoms[atom.index()].r#type().into_owned())
                    }
                })
                .collect();

            let outputs = match instruction.operation().partially_evaluate(self.context, &operand_knowledge)? {
                OperationPartialEvaluation::Inline { program: inlined, operand_indices } => {
                    let mut inputs = Vec::with_capacity(operand_indices.len());
                    for operand_index in operand_indices {
                        let (_, value) = operands.get(operand_index).ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "partial-evaluation rule referenced operand {operand_index} but operation has {} operands",
                                operands.len(),
                            ))
                        })?;
                        inputs.push(value.clone());
                    }
                    self.inline(&inlined, inputs)?
                }
                OperationPartialEvaluation::Replace { operation, operands: sources } => {
                    // The rule rewrote the operation; resolve each source to a residual atom and emit the rewrite.
                    let mut operand_atoms = Vec::with_capacity(sources.len());
                    for source in sources {
                        let atom = match source {
                            ReplaceOperand::Operand(operand_index) => {
                                let (atom, value) = operands.get(operand_index).ok_or_else(|| {
                                    ProgramError::MalformedProgram(format!(
                                        "partial-evaluation rule referenced operand {operand_index} but operation has {} operands",
                                        operands.len(),
                                    ))
                                })?;
                                self.materialize(program, &mut materialized, *atom, value)?
                            }
                            ReplaceOperand::Known(value) => {
                                // A value the rule folded: feed it as a fresh residual input with no original atom.
                                let atom = self.builder.add_input(value.r#type().into_owned());
                                self.residual_inputs.push(ResidualProgramInput::Known(value));
                                atom
                            }
                        };
                        operand_atoms.push(atom);
                    }
                    self.emit(operation, operand_atoms)?
                }
                OperationPartialEvaluation::Default => {
                    if operands.iter().all(|(_, value)| matches!(value, Residualized::Known(_))) {
                        // All operands known: fold the instruction through its own interpretation rule.
                        let known = operands
                            .iter()
                            .map(|(_, value)| match value {
                                Residualized::Known(value) => value.clone(),
                                Residualized::Residual(_) => unreachable!("all operands are known"),
                            })
                            .collect::<Vec<_>>();
                        instruction
                            .operation()
                            .interpret(self.context, known.as_slice())?
                            .into_iter()
                            .map(Residualized::Known)
                            .collect()
                    } else {
                        // At least one operand unknown: materialize operands and emit the operation into the residual.
                        let mut operand_atoms = Vec::with_capacity(operands.len());
                        for (atom, value) in operands.iter() {
                            operand_atoms.push(self.materialize(program, &mut materialized, *atom, value)?);
                        }
                        self.emit(instruction.operation().clone(), operand_atoms)?
                    }
                }
            };
            for (output_id, output) in instruction.outputs().iter().copied().zip(outputs) {
                values[output_id.index()] = Some(output);
            }
        }

        program
            .output_ids
            .iter()
            .copied()
            .map(|output_id| match values[output_id.index()].clone() {
                Some(value) => Ok(value),
                None => match &program.atoms[output_id.index()] {
                    Atom::Constant(constant) => Ok(Residualized::Known(constant.clone())),
                    Atom::Variable(_) => Err(ProgramError::UnboundAtomId { id: output_id }),
                },
            })
            .collect()
    }

    /// Emits an operation into the residual program over the provided residual-program operand atoms and returns its
    /// outputs as [`Residualized::Residual`] walk values, in output order.
    fn emit<P: Into<O>>(
        &mut self,
        operation: P,
        operand_atoms: Vec<AtomId>,
    ) -> Result<Vec<Residualized<V>>, ProgramError> {
        let outputs = self.builder.add_instruction(operation, operand_atoms)?;
        Ok(outputs.iter().copied().map(Residualized::Residual).collect())
    }

    /// Maps a walk value to a residual-program atom, deduplicated by its originating atom in `program`. A residual
    /// value is already an atom in the residual program; a known *variable* (a program input or folded intermediate)
    /// becomes a residual input fed by its known value; and a program *constant* is rebuilt inline.
    fn materialize(
        &mut self,
        program: &Program<T, V, O, Vec<V>, Vec<V>>,
        materialized: &mut [Option<AtomId>],
        atom: AtomId,
        value: &Residualized<V>,
    ) -> Result<AtomId, ProgramError> {
        match value {
            Residualized::Residual(residual_atom) => Ok(*residual_atom),
            Residualized::Known(known) => {
                if let Some(residual_atom) = materialized[atom.index()] {
                    return Ok(residual_atom);
                }
                let residual_atom = if program.atoms[atom.index()].is_variable() {
                    // Known program input or folded intermediate: feed it as a residual input.
                    let residual_atom = self.builder.add_input(known.r#type().into_owned());
                    self.residual_inputs.push(ResidualProgramInput::Known(known.clone()));
                    residual_atom
                } else {
                    // Program constant: rebuild it inline in the residual program.
                    self.builder.add_constant(known.clone())
                };
                materialized[atom.index()] = Some(residual_atom);
                Ok(residual_atom)
            }
        }
    }
}

impl<T, V, O> Program<T, V, O, Vec<V>, Vec<V>>
where
    T: Type,
    V: Value<T>,
    O: Clone + InterpretableOperation<T, V>,
{
    /// Copies this already-residual program into `builder` over the caller-provided `inputs`, returning the builder
    /// atoms holding the spliced program's outputs in output order.
    ///
    /// This is a plain relocation, not a re-partial-evaluation: every instruction and every program constant is
    /// rebuilt verbatim into `builder`. It is the reconciliation primitive an unknown-predicate `condition` uses to
    /// graft each branch's residual program into the reconciled branch it emits.
    ///
    /// # Parameters
    ///
    ///   - `builder`: Builder accumulating the program the spliced instructions are appended to.
    ///   - `inputs`: Builder atoms feeding this program's inputs, aligned with its input atoms in input order.
    pub(crate) fn splice_into(
        &self,
        builder: &mut ProgramBuilder<T, V, O>,
        inputs: &[AtomId],
    ) -> Result<Vec<AtomId>, ProgramError> {
        // The builder is borrowed by both interpretation closures, which never run concurrently; a `RefCell` lets each
        // take a short-lived mutable borrow without the borrow checker conservatively rejecting the second closure.
        let builder = RefCell::new(builder);
        self.interpret_with::<AtomId, ProgramError, _, _>(
            inputs.to_vec(),
            |_, constant| Ok(builder.borrow_mut().add_constant(constant.clone())),
            |instruction, inputs| {
                Ok(builder.borrow_mut().add_instruction(instruction.operation().clone(), inputs.to_vec())?.to_vec())
            },
        )
    }
}

impl<T, V, O> Program<T, V, O, Vec<V>, Vec<V>>
where
    T: Type,
    V: Value<T>,
    O: Clone + InterpretableOperation<T, V> + PartiallyEvaluatableOperation<T, V, O>,
{
    /// Partially evaluates this flat program against the provided per-input knowledge.
    ///
    /// Partial evaluation classifies each [`Atom`] as *known* (computable now from the provided values) or *unknown*
    /// (dependent on a runtime input), folds the known subcomputation away, and carves the remaining unknown
    /// subcomputation into a residual [`Program`] that consumes only the unknown inputs plus the known values it
    /// actually needs.
    ///
    /// It is a forward pass that builds the residual program incrementally with a [`ProgramBuilder`], deliberately
    /// built on the existing operation semantics rather than reimplementing graph machinery: every instruction whose
    /// inputs are all [`Known`](PartialValue::Known) is folded eagerly through its own
    /// [`InterpretableOperation::interpret`] rule using `context` (so no operation semantics are duplicated) and
    /// contributes nothing to the residual program; any instruction with at least one
    /// [`Unknown`](PartialValue::Unknown) input is emitted into the residual program over its operands'
    /// residual-program atoms; and the residual program is finalized with [`Program::into_simplified`], which prunes
    /// instructions and constants that no longer feed an output.
    ///
    /// Each instruction is first offered to its own [`PartiallyEvaluatableOperation::partially_evaluate`] rule, which
    /// may override this default. For example, a known-predicate `condition` returns
    /// [`OperationPartialEvaluation::Inline`] to inline its selected branch in place of the operation, so the
    /// condition disappears from the residual program. Building the residual program with a builder (rather than
    /// projecting the original) is what lets these rules emit *transformed* work; flat instructions with no override
    /// are emitted unchanged. The walk is flat per program but recurses through
    /// [`OperationPartialEvaluation::Inline`] into inlined nested programs, such as a selected `condition` branch; an
    /// instruction carrying a nested program that is *not* inlined is folded only when all of its inputs are known and
    /// is otherwise emitted unchanged.
    ///
    /// Each known *variable* a residualized instruction consumes, whether a program input or a folded intermediate, becomes
    /// a residual input of the residual program; literal constants are rebuilt inline as residual-program constants,
    /// so they are never residual inputs. The resulting [`ResidualProgram`] carries everything a caller needs to
    /// reassemble the original outputs once the runtime (unknown) inputs are available.
    ///
    /// # Relationship to [`partition`](Self::partition)
    ///
    /// This is the **value-carrying** form of partial evaluation: it holds concrete known values, so it *evaluates*
    /// the known subcomputation away (folding it through [`InterpretableOperation::interpret`]) and applies
    /// per-operation rewrite rules, yielding a single residual [`Program`] plus the folded output and
    /// residual-input *values*. Reach for it to **specialize or constant-fold** a program against inputs that are
    /// known now.
    ///
    /// [`partition`](Self::partition) is the **structural** counterpart: from a stage id per input (no values, no
    /// context) it *partitions* the program into per-stage sub-programs joined by residuals, folding and rewriting
    /// nothing, which is the form linearization uses as a two-stage known/unknown split. The two are not reducible to
    /// each other: this method requires concrete values and *discards* the known computation (folds it to values),
    /// whereas a partition keeps each stage's computation as a runnable program.
    ///
    /// # Parameters
    ///
    ///   - `context`: Interpretation context used to fold instructions whose inputs are all known.
    ///   - `inputs`: Knowledge state for each program input, in input order.
    pub fn partially_evaluate(
        &self,
        context: &V::InterpretationContext,
        inputs: &[PartialValue<T, V>],
    ) -> Result<ResidualProgram<T, V, O>, ProgramError> {
        if inputs.len() != self.input_ids.len() {
            return Err(ProgramError::InvalidInputCount { expected: self.input_ids.len(), actual: inputs.len() });
        }

        let mut residual =
            ResidualBuilder { builder: ProgramBuilder::<T, V, O>::new(), context, residual_inputs: Vec::new() };

        // Seed top-level inputs: known inputs hold their value; unknown inputs lead the residual program's inputs.
        let mut seed = Vec::with_capacity(inputs.len());
        for (index, knowledge) in inputs.iter().enumerate() {
            match knowledge {
                PartialValue::Known(value) => seed.push(Residualized::Known(value.clone())),
                PartialValue::Unknown(r#type) => {
                    let atom = residual.builder.add_input(r#type.clone());
                    residual.residual_inputs.push(ResidualProgramInput::Unknown(index));
                    seed.push(Residualized::Residual(atom));
                }
            }
        }

        let output_values = residual.inline(self, seed)?;

        // Assemble outputs: folded values return directly; residual values index the residual program's outputs.
        let mut outputs = Vec::with_capacity(output_values.len());
        let mut residual_output_atoms: Vec<AtomId> = Vec::new();
        for value in output_values {
            match value {
                Residualized::Known(value) => outputs.push(ResidualProgramOutput::Known(value)),
                Residualized::Residual(atom) => {
                    outputs.push(ResidualProgramOutput::Unknown(residual_output_atoms.len()));
                    residual_output_atoms.push(atom);
                }
            }
        }

        let output_count = residual_output_atoms.len();
        let residual_inputs = residual.residual_inputs;
        let residual_program = residual
            .builder
            .build::<Vec<V>, Vec<V>>(
                residual_output_atoms,
                vec![Placeholder; residual_inputs.len()],
                vec![Placeholder; output_count],
            )?
            .into_simplified()?;

        Ok(ResidualProgram { program: residual_program, inputs: residual_inputs, outputs })
    }
}

/// Result of partitioning a flat [`Program`] into `stage_count` totally-ordered stages by a per-input stage
/// assignment; see [`Program::partition`].
///
/// Unlike [`ResidualProgram`], this is a purely *structural* partition: it carries no concrete values and folds
/// nothing. Each instruction is placed in the latest stage among its operands, and the forward cross-stage edges are
/// threaded as *residuals* between the per-stage sub-programs. Running the [`stages`](Self::stages) in stage order,
/// threading each stage's produced residuals forward to the stages that consume them, and reassembling the outputs via
/// [`output_stages`](Self::output_stages) reproduces the original program. Its two-stage known/unknown instance is the
/// partial-evaluation split.
///
/// To recombine the original outputs: interpret each [`PartitionStage::program`] in stage order over its own inputs
/// (the original inputs named by [`PartitionStage::input_indices`]) followed by its consumed residuals (each fetched
/// through its [`ResidualSource`] from an earlier stage's stored produced residuals); split each stage's outputs into
/// its leading [`PartitionStage::output_count`] own outputs and its trailing produced residuals; then, for each
/// original program output, take it from the own outputs of the stage named by [`output_stages`](Self::output_stages).
///
/// # Invariants
///
///   - `stages[i].program.input_ids().len() == stages[i].input_indices.len() + stages[i].residual_inputs.len()`.
///   - The producer's produced-residual count is `stages[i].program.output_ids().len() - stages[i].output_count`.
///   - Every [`ResidualSource`] `{ stage, index }` consumed by stage `i` has `stage < i` and `index` less than the
///     producer's produced-residual count.
#[derive(Debug)]
pub struct PartitionedProgram<T: Type, V: Value<T>, O> {
    /// One sub-program per stage, in stage order (the index of a stage is its stage id).
    pub stages: Vec<PartitionStage<T, V, O>>,

    /// For each original program output, in original output order, the stage that produces it.
    pub output_stages: Vec<usize>,
}

/// One stage of a [`PartitionedProgram`]. Its [`program`](Self::program) takes `[stage inputs..., consumed
/// residuals...]` and produces `[stage outputs..., produced residuals...]`.
#[derive(Debug)]
pub struct PartitionStage<T: Type, V: Value<T>, O> {
    /// Projected sub-program for this stage, over this stage's own inputs followed by its consumed residuals, producing
    /// this stage's own outputs followed by its produced residuals.
    pub program: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Original input indices feeding this stage's leading (own) inputs, in order; inputs that no surviving instruction
    /// consumes are dropped.
    pub input_indices: Vec<usize>,

    /// Source of each trailing consumed-residual input, in order: which earlier stage produced it and its index among
    /// that stage's produced residuals.
    pub residual_inputs: Vec<ResidualSource>,

    /// Count of this stage's leading (own) outputs; the remaining [`program`](Self::program) outputs are the produced
    /// residuals other stages consume.
    pub output_count: usize,
}

/// Where a consumed residual comes from: the producer `stage` and the `index` of the residual among that stage's
/// produced residuals (the trailing outputs of the producer's program, after its own outputs).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ResidualSource {
    /// Stage id of the producer, the earlier stage whose produced residuals include this value.
    pub stage: usize,

    /// Index of this value among the producer stage's produced residuals.
    pub index: usize,
}

impl<T, V, O> Program<T, V, O, Vec<V>, Vec<V>>
where
    T: Type,
    V: Value<T>,
    O: Clone + Operation<T>,
{
    /// Partitions this flat program structurally into `stage_count` totally-ordered stages by the per-input stage
    /// assignment `input_stages` (one stage id per program input, in input order).
    ///
    /// This structural partition carries no values, uses no interpretation context, and folds nothing. It only
    /// partitions the program's [`Atom`]s and
    /// [`Instruction`](crate::programs::Instruction)s across stages so that running the stages in stage order, fed the
    /// *residuals* (the cross-stage values each stage consumes from earlier stages), and reassembling their outputs
    /// reproduces the original program. The two-stage instance (stage 0 = known, stage 1 = unknown) is the
    /// partial-evaluation split that [`Program::linearize`](crate::Program::linearize) uses.
    ///
    /// Stage classification is a single forward pass: a program input takes its `input_stages` entry; a constant is
    /// stage `0` (literals are available from the start and rebuilt inline by [`Self::filtered`], never threaded); an
    /// instruction's outputs all share the maximum stage over the instruction's operands. Control-flow and other
    /// nested-program operations are treated as ordinary operations here; there are no operation-specific partition
    /// rules at this stage. A residual is any *variable* operand of an instruction at stage `j` whose own stage is some
    /// `i < j`; it is produced once (deduplicated globally on first encounter, so a value consumed by several later
    /// stages or by a non-adjacent stage is threaded once) and consumed by every stage that reads it (deduplicated per
    /// consuming stage). Known constants are rebuilt inline by [`Self::filtered`] and are never threaded as residuals.
    ///
    /// Each stage's sub-program is projected with [`Self::filtered`], which keeps only the instructions reachable from
    /// that stage's outputs and rebuilds constants inline. Because every residual is, by construction, both an output
    /// of its producer stage and an operand of each consuming stage, the residual wiring is consistent across stages.
    /// The reported [`input_indices`](PartitionStage::input_indices) and [`residual_inputs`](PartitionStage::residual_inputs)
    /// reflect exactly the inputs each projected program actually takes, so an own input or a residual that no surviving
    /// instruction consumes is dropped from the corresponding program and from its index lists.
    ///
    /// An empty stage (no own inputs, outputs, or instructions, such as a trailing stage that nothing reaches)
    /// yields an empty projected program. `stage_count` is taken explicitly rather than inferred from `input_stages`,
    /// so a trailing empty stage is still produced (this is what lets a two-stage known/unknown split always produce
    /// its unknown stage, even when every input is known).
    ///
    /// # Relationship to [`partially_evaluate`](Self::partially_evaluate)
    ///
    /// This is the **structural** form: at partition time there are no concrete values, the residuals are symbolic, and
    /// *every* stage survives as a reusable program. [`partially_evaluate`](Self::partially_evaluate) is the
    /// **value-carrying** counterpart: it folds the known half to concrete values (and applies per-operation rewrites)
    /// for constant-folding or specialization. The two are not reducible to each other: this method neither has values
    /// to fold nor produces them, and it preserves each stage's computation as a program rather than evaluating it away.
    ///
    /// # Parameters
    ///
    ///   - `input_stages`: One stage id per program input, in input order; each must be less than `stage_count`.
    ///   - `stage_count`: Number of stages to partition into; must be at least `1`.
    pub fn partition(
        &self,
        input_stages: &[usize],
        stage_count: usize,
    ) -> Result<PartitionedProgram<T, V, O>, ProgramError> {
        if input_stages.len() != self.input_ids.len() {
            return Err(ProgramError::InvalidInputCount { expected: self.input_ids.len(), actual: input_stages.len() });
        }
        if stage_count == 0 {
            return Err(ProgramError::MalformedProgram("partition requires at least one stage".into()));
        }
        if let Some(&stage) = input_stages.iter().find(|&&stage| stage >= stage_count) {
            return Err(ProgramError::MalformedProgram(format!(
                "input stage {stage} is out of range for {stage_count} stages",
            )));
        }

        // Classify every atom's stage by a forward pass. A program input takes its assignment; a constant is stage 0;
        // an instruction's outputs all share the maximum stage over its operands.
        let mut atom_stage = vec![0usize; self.atoms.len()];
        for (input_id, &stage) in self.input_ids.iter().copied().zip(input_stages) {
            atom_stage[input_id.index()] = stage;
        }
        for instruction in self.instructions.iter() {
            let stage = instruction.inputs().iter().map(|operand| atom_stage[operand.index()]).max().unwrap_or(0);
            for output_id in instruction.outputs().iter().copied() {
                atom_stage[output_id.index()] = stage;
            }
        }

        // Project each side with [`filtered`](Self::filtered), threading the forward cross-stage edges as residuals.
        //
        // Cross-stage residuals, discovered with two distinct dedup scopes:
        //   - producer side (`produced` + `source_of`): the identity set of each residual, fixed once on first
        //     encounter, so a value consumed by several later stages or by a non-adjacent stage is threaded once;
        //   - consumer side (`consumed`): a per-consuming-stage ordered, deduplicated candidate list.
        // Known constants are excluded by the `is_variable` guard; `filtered` rebuilds them inline.
        let mut produced: Vec<Vec<AtomId>> = vec![Vec::new(); stage_count];
        let mut source_of: HashMap<AtomId, ResidualSource> = HashMap::new();
        let mut consumed: Vec<Vec<AtomId>> = vec![Vec::new(); stage_count];
        let mut seen: Vec<Vec<bool>> = vec![vec![false; self.atoms.len()]; stage_count];
        for instruction in self.instructions.iter() {
            let consuming_stage = instruction.outputs().iter().map(|output_id| atom_stage[output_id.index()]).max();
            let Some(consuming_stage) = consuming_stage else {
                continue;
            };
            for operand in instruction.inputs().iter().copied() {
                let producing_stage = atom_stage[operand.index()];
                if producing_stage >= consuming_stage || !self.atoms[operand.index()].is_variable() {
                    continue;
                }
                // Producer side: assign this residual its identity once, shared by every consumer.
                source_of.entry(operand).or_insert_with(|| {
                    produced[producing_stage].push(operand);
                    ResidualSource { stage: producing_stage, index: produced[producing_stage].len() - 1 }
                });
                // Consumer side: record it in this consuming stage's candidate list, deduplicated per stage.
                if !seen[consuming_stage][operand.index()] {
                    seen[consuming_stage][operand.index()] = true;
                    consumed[consuming_stage].push(operand);
                }
            }
        }

        // Per original output: the stage that produces its atom.
        let output_stages = self.output_ids.iter().map(|output_id| atom_stage[output_id.index()]).collect::<Vec<_>>();

        let mut stages = Vec::with_capacity(stage_count);
        for stage in 0..stage_count {
            // Own inputs/outputs of this stage, in original order. Boundary inputs are the own inputs followed by this
            // stage's consumed residuals; boundary outputs are the own outputs followed by this stage's produced
            // residuals. A surviving `filtered` input position less than `own_inputs.len()` is an own input; otherwise
            // it indexes into `consumed[stage]`.
            let own_inputs = (0..self.input_ids.len())
                .filter(|&index| atom_stage[self.input_ids[index].index()] == stage)
                .collect::<Vec<_>>();
            let own_outputs = self
                .output_ids
                .iter()
                .copied()
                .zip(&output_stages)
                .filter_map(|(output_id, &output_stage)| (output_stage == stage).then_some(output_id))
                .collect::<Vec<_>>();
            let output_count = own_outputs.len();

            let mut boundary_inputs = own_inputs.iter().map(|&index| self.input_ids[index]).collect::<Vec<_>>();
            boundary_inputs.extend(consumed[stage].iter().copied());
            let mut boundary_outputs = own_outputs;
            boundary_outputs.extend(produced[stage].iter().copied());
            let (program, surviving_positions) = self.filtered(&boundary_inputs, &boundary_outputs)?;

            // Rebuild both input lists from the post-filter survivors, parallel to the filtered program's inputs: an
            // own-input survivor maps back to its original input index; a residual-tail survivor maps to its producer's
            // `ResidualSource`. Keeping (rather than dropping) surviving residual-tail positions preserves the invariant
            // `program.input_ids().len() == input_indices.len() + residual_inputs.len()` even when a dead residual is
            // pruned. `output_count` is exact because `filtered` never prunes outputs, so the produced-residual suffix
            // is exactly `produced[stage]`.
            let mut input_indices = Vec::new();
            let mut residual_inputs = Vec::new();
            for position in surviving_positions {
                if position < own_inputs.len() {
                    input_indices.push(own_inputs[position]);
                } else {
                    let residual = consumed[stage][position - own_inputs.len()];
                    let source = source_of.get(&residual).copied().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!("missing residual source for atom {residual}"))
                    })?;
                    residual_inputs.push(source);
                }
            }

            stages.push(PartitionStage { program, input_indices, residual_inputs, output_count });
        }

        Ok(PartitionedProgram { stages, output_stages })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::constants::ZeroOperation;
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::scalars::Scalar;
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{ArrayType, DataType};

    use super::*;

    type TestArrayProgram = Program<ArrayType, TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>;

    fn scalar_array_type() -> ArrayType {
        ArrayType::scalar(DataType::F64)
    }

    fn boolean_array(value: bool) -> TestArray {
        TestArray::new(ArrayType::scalar(DataType::Boolean), vec![f64::from(value as u8)])
    }

    fn scalar_branch(operation: ArrayOperation<TestArray>, factor: f64) -> TestArrayProgram {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(scalar_array_type());
        let factor = builder.add_constant(TestArray::scalar(factor));
        let output = builder.add_instruction(operation, vec![input, factor]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds `f(a, x) = (2*a*a, a*a*x, x + a)` over scalar `f64`, where `a*a` is a shared intermediate. With `a`
    /// known and `x` unknown: the first output folds to a constant, the second residualizes against the folded `a*a`
    /// (a known *intermediate*), and the third residualizes against `a` (a known *input*), exercising both kinds of
    /// residual boundary plus a fully folded output.
    #[test]
    fn test_partially_evaluate_folds_known_subcomputation_and_carves_residual() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let known_input = builder.add_input(DataType::F64);
        let runtime_input = builder.add_input(DataType::F64);
        let known_square = builder.add_instruction(MulOperation, vec![known_input, known_input]).unwrap()[0];
        let doubled_square = builder.add_instruction(AddOperation, vec![known_square, known_square]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![known_square, runtime_input]).unwrap()[0];
        let sum = builder.add_instruction(AddOperation, vec![runtime_input, known_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(
                vec![doubled_square, product, sum],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let context = EagerContext::<DataType, Scalar>::new();
        let knowledge = vec![PartialValue::Known(Scalar::from(3.0)), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        // The fully known output is folded; the other two are produced by the residual program.
        match &evaluation.outputs[0] {
            ResidualProgramOutput::Known(value) => assert_eq!(*value, 18.0),
            other => panic!("expected a folded known output but got {other:?}"),
        }
        assert!(matches!(&evaluation.outputs[1], ResidualProgramOutput::Unknown(0)));
        assert!(matches!(&evaluation.outputs[2], ResidualProgramOutput::Unknown(1)));

        // The residual program drops the two folded instructions (`a*a` and `2*a*a`), keeping only the two unknown
        // ones, and takes the unknown input plus the two known residuals (the folded `a*a` and the input `a`).
        assert_eq!(program.instructions().len(), 4);
        assert_eq!(evaluation.program.instructions().len(), 2);
        assert_eq!(evaluation.inputs.len(), 3);
        assert!(matches!(&evaluation.inputs[0], ResidualProgramInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs[1], ResidualProgramInput::Known(value) if *value == 9.0));
        assert!(matches!(&evaluation.inputs[2], ResidualProgramInput::Known(value) if *value == 3.0));

        // Reassembling the residual program's outputs with the folded outputs reproduces a full eager interpretation.
        let runtime_inputs = [Scalar::from(3.0), Scalar::from(5.0)];
        let residual_arguments = evaluation
            .inputs
            .iter()
            .map(|residual_input| match residual_input {
                ResidualProgramInput::Known(value) => *value,
                ResidualProgramInput::Unknown(original_input_index) => runtime_inputs[*original_input_index],
            })
            .collect::<Vec<_>>();
        let residual_outputs = evaluation.program.interpret(residual_arguments).unwrap();
        let reassembled = evaluation
            .outputs
            .iter()
            .map(|output| match output {
                ResidualProgramOutput::Known(value) => *value,
                ResidualProgramOutput::Unknown(index) => residual_outputs[*index],
            })
            .collect::<Vec<_>>();

        assert_eq!(reassembled, program.interpret(runtime_inputs.to_vec()).unwrap());
        assert_eq!(reassembled, vec![18.0, 45.0, 8.0]);
    }

    /// With every input unknown, nothing folds: the residual program equals the original computation and there are no
    /// known residuals.
    #[test]
    fn test_partially_evaluate_with_all_unknown_inputs_residualizes_everything() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let left_input = builder.add_input(DataType::F64);
        let right_input = builder.add_input(DataType::F64);
        let product = builder.add_instruction(MulOperation, vec![left_input, right_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![product], vec![Placeholder; 2], vec![Placeholder; 1])
            .unwrap();

        let context = EagerContext::<DataType, Scalar>::new();
        let knowledge = vec![PartialValue::Unknown(DataType::F64), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        assert!(matches!(&evaluation.outputs[0], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(evaluation.inputs.iter().all(|input| matches!(input, ResidualProgramInput::Unknown(_))));
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(3.0), Scalar::from(5.0)]).unwrap(), vec![15.0]);
    }

    /// A program constant consumed by an unknown instruction must not be carried as a residual input: `filtered`
    /// rebuilds constants inline and rejects constant atoms as filter inputs. The residual program keeps the constant
    /// inside it and takes only the unknown input.
    #[test]
    fn test_partially_evaluate_keeps_program_constants_inline_in_the_residual() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let five = builder.add_constant(Scalar::from(5.0));
        let sum = builder.add_instruction(AddOperation, vec![input, five]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![sum], vec![Placeholder; 1], vec![Placeholder; 1])
            .unwrap();

        let context = EagerContext::<DataType, Scalar>::new();
        let knowledge = vec![PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        // Only the unknown input feeds the residual program; the constant stays inside it.
        assert!(matches!(&evaluation.outputs[0], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.inputs.len(), 1);
        assert!(matches!(&evaluation.inputs[0], ResidualProgramInput::Unknown(0)));
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(2.0)]).unwrap(), vec![7.0]);
    }

    /// A nullary `zero` has no inputs, so it folds to a concrete known value during partial evaluation and is dropped
    /// from the residual program. The symbolic-zero fact falls out of folding with no special handling.
    #[test]
    fn test_partially_evaluate_folds_nullary_zero_to_a_known_value() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let zero = builder.add_instruction(ZeroOperation::new(DataType::F64), vec![]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![zero, input], vec![Placeholder; 1], vec![Placeholder; 2])
            .unwrap();

        let context = EagerContext::<DataType, Scalar>::new();
        let knowledge = vec![PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        match &evaluation.outputs[0] {
            ResidualProgramOutput::Known(value) => assert_eq!(*value, 0.0),
            other => panic!("expected the nullary zero to fold but got {other:?}"),
        }
        // The zero folded away; the residual program carries no instructions and just forwards the unknown input.
        assert!(matches!(&evaluation.outputs[1], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 0);
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(5.0)]).unwrap(), vec![5.0]);
    }

    /// The builder forward pass emits every unknown instruction, then `into_simplified` prunes those that do not feed
    /// an output, so a dead unknown computation does not survive into the residual program.
    #[test]
    fn test_partially_evaluate_prunes_dead_unknown_instructions() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let one = builder.add_constant(Scalar::from(1.0));
        let two = builder.add_constant(Scalar::from(2.0));
        let used = builder.add_instruction(AddOperation, vec![input, one]).unwrap()[0];
        let _dead = builder.add_instruction(MulOperation, vec![input, two]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![used], vec![Placeholder; 1], vec![Placeholder; 1])
            .unwrap();

        let context = EagerContext::<DataType, Scalar>::new();
        let knowledge = vec![PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        // Only the live `x + 1` survives; the dead `x * 2` (and its constant) are pruned.
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(4.0)]).unwrap(), vec![5.0]);
    }

    /// Stage 3 de-risking: the partial-evaluation witness must resolve for a *self-containing* operation enum.
    /// `ArrayOperation` holds `Scan`/`While`/`Condition` variants whose bodies are themselves
    /// `Program<..., ArrayOperation, ...>`, so satisfying the bound below is exactly the recursive case feared to overflow
    /// the trait solver. Because the witness's `T`/`V` are fixed and the blanket impl grounds it in
    /// `InterpretableOperation`, this reduces to the enum's existing interpretation proof, so it compiles with no
    /// recursive obligation and no overflow.
    #[test]
    fn array_operation_satisfies_the_partial_evaluation_witness() {
        fn assert_partially_evaluatable<T: Type, V: Value<T>, O: PartiallyEvaluatableProgramOperation<T, V>>() {}
        assert_partially_evaluatable::<ArrayType, TestArray, ArrayOperation<TestArray>>();
    }

    /// With a *known* predicate, a `condition` partially evaluates by inlining its selected branch: the condition
    /// disappears from the residual program, which then contains only the taken branch's work over the unknown
    /// operand. Exercises both branches by partially evaluating the same program with a `true` and a `false`
    /// predicate.
    #[test]
    fn test_partially_evaluate_selects_branch_of_a_known_predicate_condition() {
        // `condition(p, x) = if p { x * 2 } else { x + 100 }`, staged into a flat program over `[predicate, x]`.
        let build_program = || -> TestArrayProgram {
            let condition = ConditionOperation::new(
                scalar_branch(ArrayOperation::Mul(MulOperation), 2.0),
                scalar_branch(ArrayOperation::Add(AddOperation), 100.0),
            )
            .unwrap();
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
            let input = builder.add_input(scalar_array_type());
            let output = builder
                .add_instruction(ArrayOperation::Condition(Box::new(condition)), vec![predicate, input])
                .unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let context = EagerContext::<ArrayType, TestArray>::new();

        // Known `true` predicate, unknown `x`: only the `x * 2` branch survives; the condition is gone.
        let program = build_program();
        let knowledge = vec![PartialValue::Known(boolean_array(true)), PartialValue::Unknown(scalar_array_type())];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();
        assert!(matches!(&evaluation.outputs[0], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Mul(_)));
        assert!(matches!(&evaluation.inputs[0], ResidualProgramInput::Unknown(1)));
        assert_eq!(evaluation.program.interpret(vec![TestArray::scalar(4.0)]).unwrap()[0].values, vec![8.0]);

        // Known `false` predicate, unknown `x`: only the `x + 100` branch survives.
        let program = build_program();
        let knowledge = vec![PartialValue::Known(boolean_array(false)), PartialValue::Unknown(scalar_array_type())];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();
        assert!(matches!(&evaluation.outputs[0], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Add(_)));
        assert_eq!(evaluation.program.interpret(vec![TestArray::scalar(4.0)]).unwrap()[0].values, vec![104.0],);
    }

    /// With an *unknown* predicate, a `condition` cannot be inlined, so it survives and is shrunk: each branch
    /// is partially evaluated against the operand knowledge and the two residual branches are reconciled into one
    /// rewritten `condition`. The branches are `if p { x * 2 } else { a * a + x * x }` over `[x, a]` with `a` known
    /// and `x` and `p` unknown, so the false branch folds `a * a` to a constant and shrinks from three to two.
    /// The rewritten condition is the only instruction in the residual program, and interpreting it for both
    /// predicates reproduces the original program.
    #[test]
    fn test_partially_evaluate_unknown_predicate_condition_shrinks_branches() {
        // True branch over `[x, a]`: `x * 2` (the `a` operand is unused).
        let true_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let input = builder.add_input(scalar_array_type());
            let _known_input = builder.add_input(scalar_array_type());
            let two = builder.add_constant(TestArray::scalar(2.0));
            let output = builder.add_instruction(MulOperation, vec![input, two]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        // False branch over `[x, a]`: `a * a + x * x`. With `a` known, `a * a` folds away during partial evaluation.
        let false_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let input = builder.add_input(scalar_array_type());
            let known_input = builder.add_input(scalar_array_type());
            let known_square = builder.add_instruction(MulOperation, vec![known_input, known_input]).unwrap()[0];
            let input_square = builder.add_instruction(MulOperation, vec![input, input]).unwrap()[0];
            let output = builder.add_instruction(AddOperation, vec![known_square, input_square]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        // `condition(p, x, a)` staged into a flat program over `[predicate, x, a]`.
        let build_program = || -> TestArrayProgram {
            let condition = ConditionOperation::new(true_branch(), false_branch()).unwrap();
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
            let input = builder.add_input(scalar_array_type());
            let known_input = builder.add_input(scalar_array_type());
            let output = builder
                .add_instruction(ArrayOperation::Condition(Box::new(condition)), vec![predicate, input, known_input])
                .unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };

        let context = EagerContext::<ArrayType, TestArray>::new();

        // Predicate and `x` unknown, `a` known: the condition survives but is rewritten over shrunk branches.
        let program = build_program();
        let knowledge = vec![
            PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
            PartialValue::Unknown(scalar_array_type()),
            PartialValue::Known(TestArray::scalar(3.0)),
        ];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        // The output is produced by the residual program, whose only instruction is the rewritten condition.
        assert!(matches!(&evaluation.outputs[0], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        let ArrayOperation::Condition(rewritten) = evaluation.program.instructions()[0].operation() else {
            panic!("expected the residual program to contain a rewritten condition");
        };

        // The false branch folded `a * a` away: it shrinks from three instructions to two, while the true branch is
        // unchanged at one. Neither reconciled branch is larger than its original.
        assert_eq!(rewritten.true_branch().instructions().len(), 1);
        assert_eq!(rewritten.false_branch().instructions().len(), 2);
        assert!(rewritten.true_branch().instructions().len() <= true_branch().instructions().len());
        assert!(rewritten.false_branch().instructions().len() <= false_branch().instructions().len());

        // Interpreting the residual program for both predicates reproduces the original program over the same inputs.
        let runtime = |predicate: bool, input: f64| -> Vec<f64> {
            let arguments = evaluation
                .inputs
                .iter()
                .map(|residual_input| match residual_input {
                    ResidualProgramInput::Known(value) => value.clone(),
                    ResidualProgramInput::Unknown(0) => boolean_array(predicate),
                    ResidualProgramInput::Unknown(_) => TestArray::scalar(input),
                })
                .collect::<Vec<_>>();
            let residual_outputs = evaluation.program.interpret(arguments).unwrap();
            evaluation
                .outputs
                .iter()
                .map(|output| match output {
                    ResidualProgramOutput::Known(value) => value.values[0],
                    ResidualProgramOutput::Unknown(index) => residual_outputs[*index].values[0],
                })
                .collect()
        };
        let original = |predicate: bool, input: f64| {
            program
                .interpret(vec![boolean_array(predicate), TestArray::scalar(input), TestArray::scalar(3.0)])
                .unwrap()[0]
                .values
                .clone()
        };

        assert_eq!(runtime(true, 4.0), original(true, 4.0));
        assert_eq!(runtime(true, 4.0), vec![8.0]);
        assert_eq!(runtime(false, 4.0), original(false, 4.0));
        assert_eq!(runtime(false, 4.0), vec![25.0]);
    }

    /// The structural split must recombine to the original program. Builds `f(a, x) = (a*a, a*x, x + a)` over scalar
    /// `f64` with `a` known and `x` unknown, so the first output is known, the others unknown, and the unknown side
    /// consumes the known values `a` and `a*a` as residuals. Interpreting the known side, feeding its residual tail to
    /// the unknown side, and interleaving by `output_stages` must equal interpreting the original program.
    #[test]
    fn test_partition_two_stages_recombine_to_the_original() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let known_input = builder.add_input(DataType::F64);
        let runtime_input = builder.add_input(DataType::F64);
        let known_square = builder.add_instruction(MulOperation, vec![known_input, known_input]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![known_input, runtime_input]).unwrap()[0];
        let sum = builder.add_instruction(AddOperation, vec![runtime_input, known_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(
                vec![known_square, product, sum],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        // `a` known (stage 0, index 0), `x` unknown (stage 1, index 1).
        let split = program.partition(&[0, 1], 2).unwrap();
        let known = &split.stages[0];
        let unknown = &split.stages[1];
        let residual_count = unknown.residual_inputs.len();

        // `a*a` depends only on `a` (known, stage 0); `a*x` and `x + a` depend on `x` (unknown, stage 1).
        assert_eq!(split.output_stages, vec![0, 1, 1]);
        assert_eq!(known.input_indices, vec![0]);
        assert_eq!(unknown.input_indices, vec![1]);
        // The unknown side needs `a` (for `x + a`) and the folded `a*a`'s operand `a`; specifically it consumes `a`
        // directly, so at least one residual is threaded.
        assert!(residual_count > 0);
        // Known side produces the known outputs then the residuals; unknown side takes its inputs then the residuals.
        assert_eq!(known.program.outputs().count(), 1 + residual_count);
        assert_eq!(unknown.program.inputs().count(), unknown.input_indices.len() + residual_count);

        // Recombination: run the known side on the known inputs, peel off the residuals, run the unknown side on the
        // unknown inputs followed by those residuals, then interleave by `output_stages` (stage 1 is unknown).
        let recombine = |inputs: &[Scalar]| -> Vec<Scalar> {
            let known_inputs = known.input_indices.iter().map(|&index| inputs[index]).collect::<Vec<_>>();
            let mut known_outputs = known.program.interpret(known_inputs).unwrap();
            let residuals = known_outputs.split_off(known_outputs.len() - residual_count);

            let mut unknown_inputs = unknown.input_indices.iter().map(|&index| inputs[index]).collect::<Vec<_>>();
            unknown_inputs.extend(residuals);
            let unknown_outputs = unknown.program.interpret(unknown_inputs).unwrap();

            let mut known_outputs = known_outputs.into_iter();
            let mut unknown_outputs = unknown_outputs.into_iter();
            split
                .output_stages
                .iter()
                .map(|&stage| if stage == 1 { unknown_outputs.next().unwrap() } else { known_outputs.next().unwrap() })
                .collect()
        };

        let inputs = [Scalar::from(3.0), Scalar::from(5.0)];
        assert_eq!(recombine(&inputs), program.interpret(inputs.to_vec()).unwrap());
        assert_eq!(recombine(&inputs), vec![9.0, 15.0, 8.0]);
        // A second point confirms the split is value-independent (it carries no folded constants).
        let inputs = [Scalar::from(2.0), Scalar::from(7.0)];
        assert_eq!(recombine(&inputs), program.interpret(inputs.to_vec()).unwrap());
        assert_eq!(recombine(&inputs), vec![4.0, 14.0, 9.0]);
    }

    /// With every input unknown, the split puts the whole computation on the unknown side: nothing is known, there are
    /// no residuals, and the known program has no outputs. With every input known, the mirror image holds: everything
    /// lands on the known side and the unknown program is empty.
    #[test]
    fn test_partition_two_stages_handle_all_known_and_all_unknown() {
        let build = || {
            let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
            let left_input = builder.add_input(DataType::F64);
            let right_input = builder.add_input(DataType::F64);
            let product = builder.add_instruction(MulOperation, vec![left_input, right_input]).unwrap()[0];
            let sum = builder.add_instruction(AddOperation, vec![product, left_input]).unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![product, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };

        // All inputs unknown: the unknown side is the whole computation; the known side is empty with no residuals.
        let program = build();
        let split = program.partition(&[1, 1], 2).unwrap();
        let (known, unknown) = (&split.stages[0], &split.stages[1]);
        assert_eq!(split.output_stages, vec![1, 1]);
        assert_eq!(unknown.residual_inputs.len(), 0);
        assert_eq!(known.input_indices, Vec::<usize>::new());
        assert_eq!(unknown.input_indices, vec![0, 1]);
        assert_eq!(known.program.outputs().count(), 0);
        assert_eq!(known.program.instructions().len(), 0);
        assert_eq!(unknown.program.instructions().len(), 2);
        assert_eq!(unknown.program.interpret(vec![Scalar::from(3.0), Scalar::from(5.0)]).unwrap(), vec![15.0, 18.0]);

        // All inputs known: the known side is the whole computation; the unknown side is empty with no residuals.
        let program = build();
        let split = program.partition(&[0, 0], 2).unwrap();
        let (known, unknown) = (&split.stages[0], &split.stages[1]);
        assert_eq!(split.output_stages, vec![0, 0]);
        assert_eq!(unknown.residual_inputs.len(), 0);
        assert_eq!(known.input_indices, vec![0, 1]);
        assert_eq!(unknown.input_indices, Vec::<usize>::new());
        assert_eq!(unknown.program.outputs().count(), 0);
        assert_eq!(unknown.program.instructions().len(), 0);
        assert_eq!(known.program.instructions().len(), 2);
        assert_eq!(known.program.interpret(vec![Scalar::from(3.0), Scalar::from(5.0)]).unwrap(), vec![15.0, 18.0]);
    }

    /// A nested-program / control-flow operation is treated as an ordinary operation by the structural split: there is
    /// no special split rule. With an unknown predicate the whole `condition` lands on the unknown side, and the known
    /// side contributes only the residuals the condition consumes. Recombination still reproduces the original program.
    #[test]
    fn test_partition_two_stages_treat_control_flow_as_ordinary() {
        // Branches over `[x, a]`: true branch `x * 2`, false branch `x + a`.
        let true_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let input = builder.add_input(scalar_array_type());
            let _known_input = builder.add_input(scalar_array_type());
            let two = builder.add_constant(TestArray::scalar(2.0));
            let output = builder.add_instruction(MulOperation, vec![input, two]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let false_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let input = builder.add_input(scalar_array_type());
            let known_input = builder.add_input(scalar_array_type());
            let output = builder.add_instruction(AddOperation, vec![input, known_input]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        // `condition(p, x, a)` plus a known-only output `a * a`, over `[predicate, x, a]`.
        let condition = ConditionOperation::new(true_branch(), false_branch()).unwrap();
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(scalar_array_type());
        let known_input = builder.add_input(scalar_array_type());
        let conditional = builder
            .add_instruction(ArrayOperation::Condition(Box::new(condition)), vec![predicate, input, known_input])
            .unwrap()[0];
        let known_square = builder.add_instruction(MulOperation, vec![known_input, known_input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![conditional, known_square],
                vec![Placeholder; 3],
                vec![Placeholder; 2],
            )
            .unwrap();

        // `predicate` and `x` unknown (stage 1), `a` known (stage 0).
        let split = program.partition(&[1, 1, 0], 2).unwrap();
        let (known, unknown) = (&split.stages[0], &split.stages[1]);
        let residual_count = unknown.residual_inputs.len();

        // The conditional output is unknown (stage 1); the `a * a` output is known (stage 0).
        assert_eq!(split.output_stages, vec![1, 0]);
        assert_eq!(known.input_indices, vec![2]);
        assert_eq!(unknown.input_indices, vec![0, 1]);
        // The condition consumes `a` (known), so it is threaded as a residual; the whole condition survives on the
        // unknown side unchanged (no inlining).
        assert!(residual_count > 0);
        assert!(matches!(unknown.program.instructions()[0].operation(), ArrayOperation::Condition(_)));

        let recombine = |predicate: bool, input_value: f64, known_value: f64| -> Vec<Vec<f64>> {
            let inputs = [boolean_array(predicate), TestArray::scalar(input_value), TestArray::scalar(known_value)];
            let known_inputs = known.input_indices.iter().map(|&index| inputs[index].clone()).collect::<Vec<_>>();
            let mut known_outputs = known.program.interpret(known_inputs).unwrap();
            let residuals = known_outputs.split_off(known_outputs.len() - residual_count);
            let mut unknown_inputs =
                unknown.input_indices.iter().map(|&index| inputs[index].clone()).collect::<Vec<_>>();
            unknown_inputs.extend(residuals);
            let unknown_outputs = unknown.program.interpret(unknown_inputs).unwrap();
            let mut known_outputs = known_outputs.into_iter();
            let mut unknown_outputs = unknown_outputs.into_iter();
            split
                .output_stages
                .iter()
                .map(|&stage| if stage == 1 { unknown_outputs.next().unwrap() } else { known_outputs.next().unwrap() })
                .map(|array| array.values)
                .collect()
        };
        let original = |predicate: bool, input_value: f64, known_value: f64| -> Vec<Vec<f64>> {
            let inputs = vec![boolean_array(predicate), TestArray::scalar(input_value), TestArray::scalar(known_value)];
            program.interpret(inputs).unwrap().into_iter().map(|array| array.values).collect()
        };

        assert_eq!(recombine(true, 4.0, 3.0), original(true, 4.0, 3.0));
        assert_eq!(recombine(true, 4.0, 3.0), vec![vec![8.0], vec![9.0]]);
        assert_eq!(recombine(false, 4.0, 3.0), original(false, 4.0, 3.0));
        assert_eq!(recombine(false, 4.0, 3.0), vec![vec![7.0], vec![9.0]]);
    }

    /// A three-stage [`Program::partition`] must reassemble exactly. Builds
    /// `f(stage0, stage1, stage2) = (stage0*stage0, stage1*stage0, stage2*stage0, stage2 + stage0*stage0)` over scalar
    /// `f64` with inputs assigned to stages `[0, 1, 2]`. The data flow induces the two tricky residual shapes the
    /// random-access model must handle:
    ///
    ///   - `stage0*stage0` is produced at stage 0 and consumed only at stage 2 (the last output), so it *skips* stage 1, a
    ///     non-adjacent residual with no pass-through;
    ///   - `stage0` is consumed by both stage 1 (for `stage1*stage0`) and stage 2 (for `stage2*stage0`), so one
    ///     produced residual feeds *two* later stages.
    ///
    /// The round-trip runs the stages in order, stores each stage's produced residuals, feeds each stage its
    /// `input_indices` plus its `residual_inputs` (resolved through their [`ResidualSource`]s), and reassembles the
    /// outputs by `output_stages` (own outputs first) and must equal interpreting the original program.
    #[test]
    fn test_partition_three_stages_round_trips_with_skip_and_shared_residuals() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let stage_zero_input = builder.add_input(DataType::F64);
        let stage_one_input = builder.add_input(DataType::F64);
        let stage_two_input = builder.add_input(DataType::F64);
        let stage_zero_square =
            builder.add_instruction(MulOperation, vec![stage_zero_input, stage_zero_input]).unwrap()[0];
        let stage_one_product =
            builder.add_instruction(MulOperation, vec![stage_one_input, stage_zero_input]).unwrap()[0];
        let stage_two_product =
            builder.add_instruction(MulOperation, vec![stage_two_input, stage_zero_input]).unwrap()[0];
        let skip_sum = builder.add_instruction(AddOperation, vec![stage_two_input, stage_zero_square]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(
                vec![stage_zero_square, stage_one_product, stage_two_product, skip_sum],
                vec![Placeholder; 3],
                vec![Placeholder; 4],
            )
            .unwrap();

        let partition = program.partition(&[0, 1, 2], 3).unwrap();

        // Each output lands in the stage that produces its atom.
        assert_eq!(partition.output_stages, vec![0, 1, 2, 2]);
        assert_eq!(partition.stages.len(), 3);
        // Stage 0 produces two residuals (`stage_zero_input` and `stage_zero_square`); stage 1 produces none; stage 2
        // produces none.
        let produced_count = |stage: &PartitionStage<_, _, _>| stage.program.output_ids().len() - stage.output_count;
        assert_eq!(produced_count(&partition.stages[0]), 2);
        assert_eq!(produced_count(&partition.stages[1]), 0);
        assert_eq!(produced_count(&partition.stages[2]), 0);
        // `stage_zero_input` is shared: it is consumed by both stage 1 and stage 2.
        let consumes_stage_zero =
            |stage: &PartitionStage<_, _, _>| stage.residual_inputs.iter().any(|source| source.stage == 0);
        assert!(consumes_stage_zero(&partition.stages[1]));
        assert!(consumes_stage_zero(&partition.stages[2]));
        // The per-stage invariant: program inputs split exactly into own inputs and consumed residuals.
        for stage in partition.stages.iter() {
            assert_eq!(stage.program.input_ids().len(), stage.input_indices.len() + stage.residual_inputs.len());
        }

        // Round-trip: run the stages in order, threading residuals via `ResidualSource`, and reassemble by
        // `output_stages` (own outputs first within each stage).
        let recombine = |inputs: &[Scalar]| -> Vec<Scalar> {
            let mut own_outputs: Vec<Vec<Scalar>> = Vec::with_capacity(partition.stages.len());
            let mut produced: Vec<Vec<Scalar>> = Vec::with_capacity(partition.stages.len());
            for stage in partition.stages.iter() {
                let mut arguments = stage.input_indices.iter().map(|&index| inputs[index]).collect::<Vec<_>>();
                for source in stage.residual_inputs.iter() {
                    arguments.push(produced[source.stage][source.index]);
                }
                let mut outputs = stage.program.interpret(arguments).unwrap();
                let residuals = outputs.split_off(stage.output_count);
                own_outputs.push(outputs);
                produced.push(residuals);
            }

            // Read each original output from the own outputs of its producing stage, in original order.
            let mut cursor = vec![0usize; partition.stages.len()];
            partition
                .output_stages
                .iter()
                .map(|&stage| {
                    let index = cursor[stage];
                    cursor[stage] += 1;
                    own_outputs[stage][index]
                })
                .collect()
        };

        let inputs = [Scalar::from(3.0), Scalar::from(5.0), Scalar::from(7.0)];
        assert_eq!(recombine(&inputs), program.interpret(inputs.to_vec()).unwrap());
        assert_eq!(recombine(&inputs), vec![9.0, 15.0, 21.0, 16.0]);
        // A second point confirms the partition is value-independent.
        let inputs = [Scalar::from(2.0), Scalar::from(4.0), Scalar::from(6.0)];
        assert_eq!(recombine(&inputs), program.interpret(inputs.to_vec()).unwrap());
        assert_eq!(recombine(&inputs), vec![4.0, 8.0, 12.0, 10.0]);
    }

    /// [`Program::partition`] validates its arguments: a zero `stage_count`, an out-of-range input stage, and a wrong
    /// `input_stages` length each fail rather than producing a malformed partition.
    #[test]
    fn test_partition_rejects_invalid_arguments() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let left_input = builder.add_input(DataType::F64);
        let right_input = builder.add_input(DataType::F64);
        let product = builder.add_instruction(MulOperation, vec![left_input, right_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![product], vec![Placeholder; 2], vec![Placeholder; 1])
            .unwrap();

        // Zero stages is invalid.
        assert!(matches!(program.partition(&[0, 0], 0), Err(ProgramError::MalformedProgram(_))));
        // An input stage at or beyond `stage_count` is out of range.
        assert!(matches!(program.partition(&[0, 2], 2), Err(ProgramError::MalformedProgram(_))));
        // The stage assignment must have one entry per input.
        assert!(matches!(program.partition(&[0], 2), Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),));
    }
}
