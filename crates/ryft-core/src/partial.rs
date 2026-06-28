// TODO(eaplatanios): Review this whole module.

//! Flat-program partial evaluation — the foundation for JAX-style residual minimization in Ryft's transforms.
//!
//! Partial evaluation classifies each [`Atom`] of a [`Program`] as *known* (computable now from
//! stage-time-available values) or *unknown* (dependent on a runtime input), folds the known subcomputation away, and
//! carves the remaining unknown subcomputation into a residual [`Program`] that consumes only the unknown inputs plus
//! the known values it actually needs. This is the analogue of JAX's `partial_eval_jaxpr`.
//!
//! [`Program::partially_evaluate`] implements this with a forward walk that builds the residual program incrementally
//! with a [`ProgramBuilder`]. It is deliberately built on top of the existing operation semantics rather than
//! reimplementing graph machinery:
//!
//!   - known instructions are folded by their own [`InterpretableOperation::interpret`] rule, so no operation
//!     semantics are duplicated;
//!   - each instruction is first offered to its own [`PartiallyEvaluatableOperation::partially_evaluate`]
//!     rule, so an operation can override the default and rewrite itself into transformed work — for example a
//!     known-predicate `condition` inlines its selected branch in place of the operation;
//!   - the residual program is finalized with [`Program::into_simplified`], which prunes instructions and constants
//!     that no longer feed an output.
//!
//! The walk is flat per program but recurses through [`OperationPartialEvaluation::Inline`] into inlined nested
//! programs (e.g. a selected `condition` branch); instructions carrying nested programs that are *not* inlined are
//! folded only when all of their inputs are known and otherwise emitted unchanged.
//!
//! The result ([`ResidualProgram`]) carries everything a caller needs to reassemble the original outputs from the
//! residual program once the runtime (unknown) inputs are available.

use std::borrow::Cow;
use std::cell::RefCell;

use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::Placeholder;
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::types::{Type, Typed};

/// Knowledge state of a value during partial evaluation.
///
/// A [`PartialValue`] is the abstract value domain the partial evaluator interprets a [`Program`] over: every
/// [`Atom`] and every intermediate result is either [`Known`](Self::Known) (a concrete value available
/// now) or [`Unknown`](Self::Unknown) (only its [`Type`] is available until the residual program runs).
#[derive(Clone, Debug)]
pub enum PartialValue<T: Type, V: Value<T>> {
    /// Value that is fully known at partial-evaluation time and can be folded forward.
    Known(V),

    /// Value that is not known until the residual program runs; carries the value's [`Type`].
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
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Known(value) => value.r#type(),
            Self::Unknown(r#type) => Cow::Borrowed(r#type),
        }
    }
}

/// Source feeding one input of the residual program.
///
/// The residual program's inputs are the original program's surviving unknown inputs followed by the known values
/// (residuals) that its unknown subcomputation consumes.
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
#[derive(Clone, Debug)]
pub enum ResidualProgramOutput<V> {
    /// Output that was folded to a concrete value during partial evaluation.
    Known(V),

    /// Output produced by the residual program, identified by its index into the residual program's outputs.
    Unknown(usize),
}

/// Result of partially evaluating a flat [`Program`]; see [`Program::partially_evaluate`].
///
/// To reconstruct the original program's outputs: build the residual program's input vector by mapping each
/// [`inputs`](Self::inputs) entry to either a runtime unknown-input value or its carried known residual, interpret
/// [`program`](Self::program), then read each [`outputs`](Self::outputs) entry as either its folded value or the
/// indexed residual-program output.
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

/// Closed operation families that can recursively partially evaluate nested flat [`Program`]s of themselves.
///
/// This is the partial-evaluation analogue of
/// [`InterpretableProgramOperation`](crate::operations::InterpretableProgramOperation): it names the recursive fixed
/// point that nested-program operations (e.g. `scan`/`while`/`condition` bodies) use to partially evaluate their
/// bodies, without restating the full [`partially_evaluate`](Program::partially_evaluate) bound at every recursive payload
/// boundary.
///
/// Unlike the linearization and transposition witnesses — whose context/operation type parameters grow with each
/// recursion level and must therefore name a fixed point to stop the trait solver from diverging — this witness's
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

/// Outcome by which an operation overrides its own partial evaluation (see [`PartiallyEvaluatableOperation`]).
pub enum OperationPartialEvaluation<T: Type, V: Value<T>, O> {
    /// Replace the operation by inlining this nested program, fed the operation's operands at the listed operand
    /// indices (in order). Used e.g. by a known-predicate `condition`, which inlines its selected branch.
    Inline {
        /// Nested program to inline in place of the operation.
        program: Program<T, V, O, Vec<V>, Vec<V>>,

        /// Indices into the operation's operands that feed `program`'s inputs, in input order.
        operand_indices: Vec<usize>,
    },

    /// Residualize a (possibly transformed) operation in place of this one, fed the listed operand sources. Used when
    /// an operation cannot be folded or inlined but can be rewritten — e.g. an unknown-predicate `condition`
    /// residualizes a new `condition` over partially-evaluated branches.
    Replace {
        /// Transformed operation to emit into the residual program.
        operation: O,

        /// What feeds each of `operation`'s operands, in operand order.
        operands: Vec<ReplaceOperand<V>>,
    },
}

/// Source feeding one operand of a `Replace`d operation; see [`OperationPartialEvaluation::Replace`].
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
/// control-flow operations may override — for example a known-predicate `condition` inlines its selected branch.
///
/// # Default Behavior
///
/// The default [`partially_evaluate`](Self::partially_evaluate) returns [`None`], which tells
/// [`Program::partially_evaluate`] to fall back to its default handling of the operation:
///
///   - when *all* of the operation's operands are [`Known`](PartialValue::Known), it **folds** the operation by
///     interpreting it through [`InterpretableOperation::interpret`], so the operation's outputs become known values
///     and the operation contributes nothing to the residual program;
///   - otherwise it **residualizes** the operation unchanged: it emits the operation into the residual program over
///     its operands' residual-program atoms — materializing each known operand as a residual input (for a known
///     variable) or as an inlined residual-program constant (for a literal) — so the operation runs at
///     residual-program execution time.
///
/// # Overriding
///
/// Returning [`Some`]\([`OperationPartialEvaluation`]\) overrides this default with the requested rewrite. For
/// example, a `condition` whose predicate is [`Known`](PartialValue::Known) returns
/// [`OperationPartialEvaluation::Inline`] for its selected branch, so the condition disappears from the residual
/// program and only the taken branch's work survives.
///
/// # Type Parameters
///
///   - `O`: Operation family of the residual program and of any inlined nested programs — i.e. the enum this
///     operation belongs to. It is the operation type of every [`Program`] in an
///     [`OperationPartialEvaluation`] this rule returns.
pub trait PartiallyEvaluatableOperation<T: Type, V: Value<T>, O>: Sized {
    /// Optionally overrides partial evaluation of this operation given its operands' knowledge. Returning [`None`]
    /// uses the default behavior described in the [trait documentation](Self): fold the operation when all operands
    /// are [`Known`](PartialValue::Known), otherwise residualize it unchanged.
    ///
    /// # Parameters
    ///
    ///   - `context`: Interpretation context available to a custom rule, e.g. to fold a known sub-result.
    ///   - `operands`: Knowledge state for each of this operation's operands, in operand order.
    fn partially_evaluate(
        &self,
        context: &V::InterpretationContext,
        operands: &[PartialValue<T, V>],
    ) -> Result<Option<OperationPartialEvaluation<T, V, O>>, ProgramError> {
        let _ = (context, operands);
        Ok(None)
    }
}

/// Walk-time classification of an [`Atom`](crate::Atom) during [`Program::partially_evaluate`]: a value known now, or a
/// value that lives in the residual program at the given residual-program [`AtomId`].
#[derive(Clone)]
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
    /// fed the selected operands, so an operation can rewrite itself into transformed work (e.g. a known-predicate
    /// `condition` inlining its selected branch). Program constants are lifted to [`Residualized::Known`] on first use
    /// and rebuilt inline in the residual program when a residualized instruction consumes them.
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
                Some(OperationPartialEvaluation::Inline { program: inlined, operand_indices }) => {
                    let branch_inputs =
                        operand_indices.iter().map(|&index| operands[index].1.clone()).collect::<Vec<_>>();
                    self.inline(&inlined, branch_inputs)?
                }
                Some(OperationPartialEvaluation::Replace { operation, operands: sources }) => {
                    // The rule rewrote the operation; resolve each source to a residual atom and emit the rewrite.
                    let mut operand_atoms = Vec::with_capacity(sources.len());
                    for source in sources {
                        let atom = match source {
                            ReplaceOperand::Operand(index) => {
                                let (atom, value) = &operands[index];
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
                None => {
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
    fn emit(&mut self, operation: O, operand_atoms: Vec<AtomId>) -> Result<Vec<Residualized<V>>, ProgramError> {
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

/// Copies an already-residual `program` into `builder` over the caller-provided `input_atoms`, returning the builder
/// atoms holding the spliced program's outputs in output order.
///
/// This is a plain relocation, not a re-partial-evaluation: every instruction and every program constant is rebuilt
/// verbatim into `builder`. It is the reconciliation primitive an unknown-predicate `condition` uses to graft each
/// branch's residual program into the reconciled branch it emits.
///
/// # Parameters
///
///   - `builder`: Builder accumulating the program the spliced instructions are appended to.
///   - `program`: Already-residual program to copy in.
///   - `input_atoms`: Builder atoms feeding `program`'s inputs, aligned with `program`'s input atoms in input order.
pub(crate) fn splice_program_into<T, V, O>(
    builder: &mut ProgramBuilder<T, V, O>,
    program: &Program<T, V, O, Vec<V>, Vec<V>>,
    input_atoms: &[AtomId],
) -> Result<Vec<AtomId>, ProgramError>
where
    T: Type,
    V: Value<T>,
    O: Clone + InterpretableOperation<T, V>,
{
    // The builder is borrowed by both interpretation closures, which never run concurrently; a `RefCell` lets each
    // take a short-lived mutable borrow without the borrow checker conservatively rejecting the second closure.
    let builder = RefCell::new(builder);
    program.interpret_with::<AtomId, ProgramError, _, _>(
        input_atoms.to_vec(),
        |_, constant| Ok(builder.borrow_mut().add_constant(constant.clone())),
        |instruction, operand_atoms| {
            Ok(builder.borrow_mut().add_instruction(instruction.operation().clone(), operand_atoms.to_vec())?.to_vec())
        },
    )
}

impl<T, V, O> Program<T, V, O, Vec<V>, Vec<V>>
where
    T: Type,
    V: Value<T>,
    O: Clone + InterpretableOperation<T, V> + PartiallyEvaluatableOperation<T, V, O>,
{
    /// Partially evaluates this flat program against the provided per-input knowledge.
    ///
    /// This is a forward pass that builds the residual program incrementally with a [`ProgramBuilder`]: every
    /// instruction whose inputs are all [`Known`](PartialValue::Known) is folded eagerly through its
    /// [`InterpretableOperation::interpret`] rule using `context` and contributes nothing to the residual program; any
    /// instruction with at least one [`Unknown`](PartialValue::Unknown) input is emitted into the residual program
    /// over its operands' residual-program atoms.
    ///
    /// Each instruction is first offered to its own
    /// [`PartiallyEvaluatableOperation::partially_evaluate`] rule, which may override this default — for
    /// example a known-predicate `condition` returns [`OperationPartialEvaluation::Inline`] to inline its selected
    /// branch in place of the operation, so the condition disappears from the residual program. Building the residual
    /// program with a builder (rather than projecting the original) is what lets these rules emit *transformed* work;
    /// flat instructions with no override are emitted unchanged.
    ///
    /// Each known *variable* a residualized instruction consumes — a program input or a folded intermediate — becomes
    /// a residual input of the residual program; literal constants are rebuilt inline as residual-program constants,
    /// so they are never residual inputs.
    ///
    /// # Relationship to [`partial_eval_split`](Self::partial_eval_split)
    ///
    /// This is the **value-carrying** form of partial evaluation: it holds concrete known values, so it *evaluates*
    /// the known subcomputation away (folding it through [`InterpretableOperation::interpret`]) and applies
    /// per-operation rewrite rules, yielding a single residual [`Program`] plus the folded output and
    /// residual-input *values*. Reach for it to **specialize or constant-fold** a program against inputs that are
    /// known now. It is the analogue of JAX's value-carrying partial evaluation (tracing over `PartialVal`s);
    /// [`PartialValue`] mirrors JAX's `PartialVal`.
    ///
    /// [`partial_eval_split`](Self::partial_eval_split) is the **structural** counterpart: from a known/unknown
    /// *flag* per input (no values, no context) it *partitions* the program into two sub-programs joined by
    /// residuals, folding and rewriting nothing — the form linearization needs. The two are not reducible to each
    /// other: this method requires concrete values and *discards* the known computation (folds it to values),
    /// whereas the split keeps the known computation as a runnable program.
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

        let mut residual = ResidualBuilder {
            builder: ProgramBuilder::<T, V, O>::new(),
            context,
            residual_inputs: Vec::new(),
        };

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

/// Result of splitting a flat program by partial evaluation; see [`Program::partial_eval_split`].
///
/// Unlike [`ResidualProgram`], this is a purely *structural* split: it carries no concrete values and folds nothing.
/// It partitions the original program into two sub-programs by a per-input known/unknown classification — the
/// [`known_program`](Self::known_program) over the known inputs and the [`unknown_program`](Self::unknown_program) over
/// the unknown inputs plus the *residuals* (the known values the unknown side consumes) — so that interpreting the two
/// in sequence and interleaving their outputs reproduces the original program. It is the structural analogue of JAX's
/// `partial_eval_jaxpr` primitive.
///
/// To recombine the original outputs: interpret [`known_program`](Self::known_program) on the known inputs and split
/// its outputs into the known program-outputs followed by the [`residual_count`](Self::residual_count) residuals;
/// interpret [`unknown_program`](Self::unknown_program) on the unknown inputs followed by those residuals; then, for
/// each original output, take it from the unknown program's outputs where [`output_unknowns`](Self::output_unknowns) is
/// `true` and from the known program's outputs otherwise.
#[derive(Debug)]
pub struct PartitionedResidualProgram<T: Type, V: Value<T>, O> {
    /// Computes the program's known outputs followed by the residuals, from the known inputs.
    pub known_program: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Computes the unknown outputs from the unknown inputs followed by the residuals.
    pub unknown_program: Program<T, V, O, Vec<V>, Vec<V>>,

    /// For each original program output, `true` if it is produced by [`unknown_program`](Self::unknown_program),
    /// `false` if by [`known_program`](Self::known_program).
    pub output_unknowns: Vec<bool>,

    /// Original input indices (in order) that are known — the inputs of [`known_program`](Self::known_program).
    pub known_input_indices: Vec<usize>,

    /// Original input indices (in order) that are unknown — the leading inputs of
    /// [`unknown_program`](Self::unknown_program).
    pub unknown_input_indices: Vec<usize>,

    /// Number of residuals: the trailing outputs of [`known_program`](Self::known_program) and the trailing inputs of
    /// [`unknown_program`](Self::unknown_program).
    pub residual_count: usize,
}

impl<T, V, O> Program<T, V, O, Vec<V>, Vec<V>>
where
    T: Type,
    V: Value<T>,
    O: Clone + Operation<T>,
{
    /// Partially evaluates this flat program structurally, splitting it into a *known* sub-program and an *unknown*
    /// sub-program by the per-input classification `input_unknowns` (one bool per program input; `true` = unknown).
    ///
    /// This is the structural analogue of JAX's `partial_eval_jaxpr`: it carries no values, uses no interpretation
    /// context, and folds nothing. It only partitions the program's [`Atom`]s and
    /// [`Instruction`](crate::programs::Instruction)s into two sides so that running the known side, then the unknown
    /// side fed the *residuals* (the known values the unknown side consumes), and interleaving their outputs
    /// reproduces the original program.
    ///
    /// Classification is a single forward pass: a program input is unknown exactly when its `input_unknowns` flag is
    /// set; a constant is always known; an instruction's outputs are unknown when any of its input atoms is unknown,
    /// and known otherwise. Control-flow and other nested-program operations are treated as ordinary operations here —
    /// there are no operation-specific split rules at this stage. The residuals are the known *variable* atoms (program
    /// inputs or instruction outputs) consumed by at least one unknown instruction, in first-encountered order and
    /// deduplicated; known constants consumed by the unknown side are rebuilt inline in the unknown program rather than
    /// threaded as residuals.
    ///
    /// The two sub-programs are projected with [`Self::filtered`], which keeps only the instructions reachable from
    /// each side's outputs and rebuilds constants inline. Because every residual is, by construction, both an output of
    /// the known side and an operand of the unknown side, the residual count is consistent across both programs. The
    /// reported [`known_input_indices`](PartitionedResidualProgram::known_input_indices) and
    /// [`unknown_input_indices`](PartitionedResidualProgram::unknown_input_indices) reflect exactly the inputs each
    /// projected program actually takes, so a known or unknown input that no surviving instruction consumes is dropped
    /// from the corresponding program and from its index list.
    ///
    /// # Relationship to [`partially_evaluate`](Self::partially_evaluate)
    ///
    /// This is the **structural** form, and is what [`Program::linearize`](crate::Program::linearize) uses to split a
    /// jvp-traced program into its primal (known) and tangent (unknown) halves: at linearization time there are no
    /// concrete values, the residuals are symbolic, and *both* halves must survive as reusable programs. It is the
    /// analogue of JAX's `partial_eval_jaxpr`.
    ///
    /// [`partially_evaluate`](Self::partially_evaluate) is the **value-carrying** counterpart: it folds the known
    /// half to concrete values (and applies per-operation rewrites) for constant-folding or specialization. The two
    /// are not reducible to each other — this method neither has values to fold nor produces them, and it preserves
    /// the known computation as a program rather than evaluating it away.
    ///
    /// # Parameters
    ///
    ///   - `input_unknowns`: One flag per program input, in input order; `true` marks the input as unknown.
    pub fn partial_eval_split(
        &self,
        input_unknowns: &[bool],
    ) -> Result<PartitionedResidualProgram<T, V, O>, ProgramError> {
        if input_unknowns.len() != self.input_ids.len() {
            return Err(ProgramError::InvalidInputCount {
                expected: self.input_ids.len(),
                actual: input_unknowns.len(),
            });
        }

        // Classify every atom known/unknown by a forward pass. Constants are known; a program input takes its flag; an
        // instruction's outputs are unknown iff any operand is unknown.
        let mut atom_unknown = vec![false; self.atoms.len()];
        for (input_id, &unknown) in self.input_ids.iter().copied().zip(input_unknowns) {
            atom_unknown[input_id.index()] = unknown;
        }
        for instruction in self.instructions.iter() {
            let unknown = instruction.inputs().iter().any(|operand| atom_unknown[operand.index()]);
            for output_id in instruction.outputs().iter().copied() {
                atom_unknown[output_id.index()] = unknown;
            }
        }

        // Residuals: known *variable* atoms consumed by at least one unknown instruction, in first-encountered order,
        // deduplicated. Known constants are rebuilt inline by `filtered` and never threaded as residuals.
        let mut residual_atoms: Vec<AtomId> = Vec::new();
        let mut is_residual = vec![false; self.atoms.len()];
        for instruction in self.instructions.iter() {
            let unknown = instruction.outputs().iter().any(|output_id| atom_unknown[output_id.index()]);
            if !unknown {
                continue;
            }
            for operand in instruction.inputs().iter().copied() {
                if atom_unknown[operand.index()] || is_residual[operand.index()] {
                    continue;
                }
                if self.atoms[operand.index()].is_variable() {
                    is_residual[operand.index()] = true;
                    residual_atoms.push(operand);
                }
            }
        }

        // Per original output: unknown iff its atom is unknown.
        let output_unknowns =
            self.output_ids.iter().map(|output_id| atom_unknown[output_id.index()]).collect::<Vec<_>>();

        // Original input indices, partitioned by classification and kept in input order. The boundary atoms passed to
        // `filtered` are these inputs' atoms, so a surviving `filtered` position indexes straight back into them.
        let known_input_candidates = (0..self.input_ids.len()).filter(|&index| !input_unknowns[index]);
        let unknown_input_candidates = (0..self.input_ids.len()).filter(|&index| input_unknowns[index]);

        // Known program: inputs are the known program-input atoms (in order); outputs are the known program-outputs
        // (in original order) followed by the residuals.
        let known_input_candidates = known_input_candidates.collect::<Vec<_>>();
        let known_input_atoms = known_input_candidates.iter().map(|&index| self.input_ids[index]).collect::<Vec<_>>();
        let mut known_outputs = self
            .output_ids
            .iter()
            .copied()
            .zip(&output_unknowns)
            .filter_map(|(output_id, &unknown)| (!unknown).then_some(output_id))
            .collect::<Vec<_>>();
        known_outputs.extend(residual_atoms.iter().copied());
        let (known_program, known_live_inputs) = self.filtered(&known_input_atoms, &known_outputs)?;

        // Unknown program: inputs are the unknown program-input atoms (in order) followed by the residuals; outputs are
        // the unknown program-outputs (in original order).
        let unknown_input_candidates = unknown_input_candidates.collect::<Vec<_>>();
        let mut unknown_boundary_inputs =
            unknown_input_candidates.iter().map(|&index| self.input_ids[index]).collect::<Vec<_>>();
        let unknown_input_count = unknown_boundary_inputs.len();
        unknown_boundary_inputs.extend(residual_atoms.iter().copied());
        let unknown_outputs = self
            .output_ids
            .iter()
            .copied()
            .zip(&output_unknowns)
            .filter_map(|(output_id, &unknown)| unknown.then_some(output_id))
            .collect::<Vec<_>>();
        let (unknown_program, unknown_live_inputs) = self.filtered(&unknown_boundary_inputs, &unknown_outputs)?;

        // `filtered` returns the surviving boundary-input positions, in order. Drop positions in the residual tail (the
        // residuals are not program inputs) and map the rest back to original input indices, so the reported index
        // lists match each projected program's actual inputs.
        let known_input_indices = known_live_inputs
            .into_iter()
            .filter(|&position| position < known_input_candidates.len())
            .map(|position| known_input_candidates[position])
            .collect::<Vec<_>>();
        let unknown_input_indices = unknown_live_inputs
            .into_iter()
            .filter(|&position| position < unknown_input_count)
            .map(|position| unknown_input_candidates[position])
            .collect::<Vec<_>>();

        Ok(PartitionedResidualProgram {
            known_program,
            unknown_program,
            output_unknowns,
            known_input_indices,
            unknown_input_indices,
            residual_count: residual_atoms.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::contexts::EagerContext;
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::constants::ZeroOperation;
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::tests::TestArray;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{ArrayType, DataType};

    use super::*;

    /// Builds `f(a, x) = (2*a*a, a*a*x, x + a)` over scalar `f64`, where `a*a` is a shared intermediate. With `a`
    /// known and `x` unknown: the first output folds to a constant, the second residualizes against the folded `a*a`
    /// (a known *intermediate*), and the third residualizes against `a` (a known *input*) — exercising both kinds of
    /// residual boundary plus a fully folded output.
    #[test]
    fn test_partially_evaluate_folds_known_subcomputation_and_carves_residual() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let aa = builder.add_instruction(MulOperation, vec![a, a]).unwrap()[0];
        let out0 = builder.add_instruction(AddOperation, vec![aa, aa]).unwrap()[0];
        let out1 = builder.add_instruction(MulOperation, vec![aa, x]).unwrap()[0];
        let out2 = builder.add_instruction(AddOperation, vec![x, a]).unwrap()[0];
        let program = builder
            .build::<Vec<f64>, Vec<f64>>(vec![out0, out1, out2], vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let context = EagerContext::<DataType, f64>::new();
        let knowledge = vec![PartialValue::Known(3.0f64), PartialValue::Unknown(DataType::F64)];
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
        let runtime_inputs = [3.0f64, 5.0f64];
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
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let product = builder.add_instruction(MulOperation, vec![a, x]).unwrap()[0];
        let program =
            builder.build::<Vec<f64>, Vec<f64>>(vec![product], vec![Placeholder; 2], vec![Placeholder; 1]).unwrap();

        let context = EagerContext::<DataType, f64>::new();
        let knowledge = vec![PartialValue::Unknown(DataType::F64), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        assert!(matches!(&evaluation.outputs[0], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(evaluation.inputs.iter().all(|input| matches!(input, ResidualProgramInput::Unknown(_))));
        assert_eq!(evaluation.program.interpret(vec![3.0, 5.0]).unwrap(), vec![15.0]);
    }

    /// A program constant consumed by an unknown instruction must not be carried as a residual input: `filtered`
    /// rebuilds constants inline and rejects constant atoms as filter inputs. The residual program keeps the constant
    /// inside it and takes only the unknown input.
    #[test]
    fn test_partially_evaluate_keeps_program_constants_inline_in_the_residual() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let x = builder.add_input(DataType::F64);
        let five = builder.add_constant(5.0f64);
        let sum = builder.add_instruction(AddOperation, vec![x, five]).unwrap()[0];
        let program =
            builder.build::<Vec<f64>, Vec<f64>>(vec![sum], vec![Placeholder; 1], vec![Placeholder; 1]).unwrap();

        let context = EagerContext::<DataType, f64>::new();
        let knowledge = vec![PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        // Only the unknown input feeds the residual program; the constant stays inside it.
        assert!(matches!(&evaluation.outputs[0], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.inputs.len(), 1);
        assert!(matches!(&evaluation.inputs[0], ResidualProgramInput::Unknown(0)));
        assert_eq!(evaluation.program.interpret(vec![2.0]).unwrap(), vec![7.0]);
    }

    /// A nullary `zero` has no inputs, so it folds to a concrete known value during partial evaluation and is dropped
    /// from the residual program — the symbolic-zero fact falls out of folding with no special handling.
    #[test]
    fn test_partially_evaluate_folds_nullary_zero_to_a_known_value() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let x = builder.add_input(DataType::F64);
        let zero = builder.add_instruction(ZeroOperation::new(DataType::F64), vec![]).unwrap()[0];
        let program =
            builder.build::<Vec<f64>, Vec<f64>>(vec![zero, x], vec![Placeholder; 1], vec![Placeholder; 2]).unwrap();

        let context = EagerContext::<DataType, f64>::new();
        let knowledge = vec![PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        match &evaluation.outputs[0] {
            ResidualProgramOutput::Known(value) => assert_eq!(*value, 0.0),
            other => panic!("expected the nullary zero to fold but got {other:?}"),
        }
        // The zero folded away; the residual program carries no instructions and just forwards the unknown input.
        assert!(matches!(&evaluation.outputs[1], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 0);
        assert_eq!(evaluation.program.interpret(vec![5.0]).unwrap(), vec![5.0]);
    }

    /// The builder forward pass emits every unknown instruction, then `into_simplified` prunes those that do not feed
    /// an output — so a dead unknown computation does not survive into the residual program.
    #[test]
    fn test_partially_evaluate_prunes_dead_unknown_instructions() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let x = builder.add_input(DataType::F64);
        let one = builder.add_constant(1.0f64);
        let two = builder.add_constant(2.0f64);
        let used = builder.add_instruction(AddOperation, vec![x, one]).unwrap()[0];
        let _dead = builder.add_instruction(MulOperation, vec![x, two]).unwrap()[0];
        let program =
            builder.build::<Vec<f64>, Vec<f64>>(vec![used], vec![Placeholder; 1], vec![Placeholder; 1]).unwrap();

        let context = EagerContext::<DataType, f64>::new();
        let knowledge = vec![PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();

        // Only the live `x + 1` survives; the dead `x * 2` (and its constant) are pruned.
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert_eq!(evaluation.program.interpret(vec![4.0]).unwrap(), vec![5.0]);
    }

    /// Stage 3 de-risking: the partial-evaluation witness must resolve for a *self-containing* operation enum.
    /// `ArrayOperation` holds `Scan`/`While`/`Condition` variants whose bodies are themselves
    /// `Program<…, ArrayOperation, …>`, so satisfying the bound below is exactly the recursive case feared to overflow
    /// the trait solver. Because the witness's `T`/`V` are fixed and the blanket impl grounds it in
    /// `InterpretableOperation`, this reduces to the enum's existing interpretation proof — so it compiles, with no
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
        // Builds a single-input scalar `f64` branch program computing `f(x)` as `x <operation> factor`.
        fn branch(
            operation: ArrayOperation<TestArray>,
            factor: f64,
        ) -> Program<ArrayType, TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let constant = builder.add_constant(TestArray::scalar(factor));
            let output = builder.add_instruction(operation, vec![input, constant]).unwrap()[0];
            builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        }

        // `condition(p, x) = if p { x * 2 } else { x + 100 }`, staged into a flat program over `[predicate, x]`.
        let build_program =
            || -> Program<ArrayType, TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
                let condition = ConditionOperation::new(
                    branch(ArrayOperation::Mul(MulOperation), 2.0),
                    branch(ArrayOperation::Add(AddOperation), 100.0),
                )
                .unwrap();
                let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
                let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
                let x = builder.add_input(ArrayType::scalar(DataType::F64));
                let output = builder
                    .add_instruction(ArrayOperation::Condition(Box::new(condition)), vec![predicate, x])
                    .unwrap()[0];
                builder
                    .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                    .unwrap()
            };

        let context = EagerContext::<ArrayType, TestArray>::new();
        let boolean = |value: bool| TestArray::new(ArrayType::scalar(DataType::Boolean), vec![value as u8 as f64]);

        // Known `true` predicate, unknown `x`: only the `x * 2` branch survives; the condition is gone.
        let program = build_program();
        let knowledge =
            vec![PartialValue::Known(boolean(true)), PartialValue::Unknown(ArrayType::scalar(DataType::F64))];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();
        assert!(matches!(&evaluation.outputs[0], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Mul(_)));
        assert!(matches!(&evaluation.inputs[0], ResidualProgramInput::Unknown(1)));
        assert_eq!(evaluation.program.interpret(vec![TestArray::scalar(4.0)]).unwrap()[0].values, vec![8.0]);

        // Known `false` predicate, unknown `x`: only the `x + 100` branch survives.
        let program = build_program();
        let knowledge =
            vec![PartialValue::Known(boolean(false)), PartialValue::Unknown(ArrayType::scalar(DataType::F64))];
        let evaluation = program.partially_evaluate(&context, knowledge.as_slice()).unwrap();
        assert!(matches!(&evaluation.outputs[0], ResidualProgramOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Add(_)));
        assert_eq!(
            evaluation.program.interpret(vec![TestArray::scalar(4.0)]).unwrap()[0].values,
            vec![104.0],
        );
    }

    /// With an *unknown* predicate, a `condition` cannot be inlined, so it survives — but it is shrunk: each branch
    /// is partially evaluated against the operand knowledge and the two residual branches are reconciled into one
    /// rewritten `condition`. The branches are `if p { x * 2 } else { a * a + x * x }` over `[x, a]` with `a` known
    /// and `x` and `p` unknown, so the false branch folds `a * a` to a constant and shrinks from three to two.
    /// The rewritten condition is the only instruction in the residual program, and interpreting it for both
    /// predicates reproduces the original program.
    #[test]
    fn test_partially_evaluate_unknown_predicate_condition_shrinks_branches() {
        let scalar = || ArrayType::scalar(DataType::F64);

        // True branch over `[x, a]`: `x * 2` (the `a` operand is unused).
        let true_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let x = builder.add_input(scalar());
            let _a = builder.add_input(scalar());
            let two = builder.add_constant(TestArray::scalar(2.0));
            let output = builder.add_instruction(MulOperation, vec![x, two]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        // False branch over `[x, a]`: `a * a + x * x`. With `a` known, `a * a` folds away during partial evaluation.
        let false_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let x = builder.add_input(scalar());
            let a = builder.add_input(scalar());
            let aa = builder.add_instruction(MulOperation, vec![a, a]).unwrap()[0];
            let xx = builder.add_instruction(MulOperation, vec![x, x]).unwrap()[0];
            let output = builder.add_instruction(AddOperation, vec![aa, xx]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        // `condition(p, x, a)` staged into a flat program over `[predicate, x, a]`.
        let build_program =
            || -> Program<ArrayType, TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>> {
                let condition = ConditionOperation::new(true_branch(), false_branch()).unwrap();
                let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
                let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
                let x = builder.add_input(scalar());
                let a = builder.add_input(scalar());
                let output = builder
                    .add_instruction(ArrayOperation::Condition(Box::new(condition)), vec![predicate, x, a])
                    .unwrap()[0];
                builder
                    .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                    .unwrap()
            };

        let context = EagerContext::<ArrayType, TestArray>::new();
        let boolean = |value: bool| TestArray::new(ArrayType::scalar(DataType::Boolean), vec![value as u8 as f64]);

        // Predicate and `x` unknown, `a` known: the condition survives but is rewritten over shrunk branches.
        let program = build_program();
        let knowledge = vec![
            PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
            PartialValue::Unknown(scalar()),
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
        let runtime = |predicate: bool, x: f64| -> Vec<f64> {
            let arguments = evaluation
                .inputs
                .iter()
                .map(|residual_input| match residual_input {
                    ResidualProgramInput::Known(value) => value.clone(),
                    ResidualProgramInput::Unknown(0) => boolean(predicate),
                    ResidualProgramInput::Unknown(_) => TestArray::scalar(x),
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
        let original = |predicate: bool, x: f64| {
            program.interpret(vec![boolean(predicate), TestArray::scalar(x), TestArray::scalar(3.0)]).unwrap()[0]
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
    /// the unknown side, and interleaving by `output_unknowns` must equal interpreting the original program.
    #[test]
    fn test_partial_eval_split_recombines_to_the_original() {
        let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let aa = builder.add_instruction(MulOperation, vec![a, a]).unwrap()[0];
        let ax = builder.add_instruction(MulOperation, vec![a, x]).unwrap()[0];
        let xa = builder.add_instruction(AddOperation, vec![x, a]).unwrap()[0];
        let program = builder
            .build::<Vec<f64>, Vec<f64>>(vec![aa, ax, xa], vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        // `a` known (index 0), `x` unknown (index 1).
        let split = program.partial_eval_split(&[false, true]).unwrap();

        // `a*a` depends only on `a` (known); `a*x` and `x + a` depend on `x` (unknown).
        assert_eq!(split.output_unknowns, vec![false, true, true]);
        assert_eq!(split.known_input_indices, vec![0]);
        assert_eq!(split.unknown_input_indices, vec![1]);
        // The unknown side needs `a` (for `x + a`) and the folded `a*a`'s operand `a`; specifically it consumes `a`
        // directly, so at least one residual is threaded.
        assert!(split.residual_count > 0);
        // Known side produces the known outputs then the residuals; unknown side takes its inputs then the residuals.
        assert_eq!(split.known_program.outputs().count(), 1 + split.residual_count);
        assert_eq!(split.unknown_program.inputs().count(), split.unknown_input_indices.len() + split.residual_count);

        // Recombination: run the known side on the known inputs, peel off the residuals, run the unknown side on the
        // unknown inputs followed by those residuals, then interleave by `output_unknowns`.
        let recombine = |inputs: &[f64]| -> Vec<f64> {
            let known_inputs = split.known_input_indices.iter().map(|&index| inputs[index]).collect::<Vec<_>>();
            let mut known_outputs = split.known_program.interpret(known_inputs).unwrap();
            let residuals = known_outputs.split_off(known_outputs.len() - split.residual_count);

            let mut unknown_inputs =
                split.unknown_input_indices.iter().map(|&index| inputs[index]).collect::<Vec<_>>();
            unknown_inputs.extend(residuals);
            let unknown_outputs = split.unknown_program.interpret(unknown_inputs).unwrap();

            let mut known_outputs = known_outputs.into_iter();
            let mut unknown_outputs = unknown_outputs.into_iter();
            split
                .output_unknowns
                .iter()
                .map(|&unknown| if unknown { unknown_outputs.next().unwrap() } else { known_outputs.next().unwrap() })
                .collect()
        };

        let inputs = [3.0f64, 5.0f64];
        assert_eq!(recombine(&inputs), program.interpret(inputs.to_vec()).unwrap());
        assert_eq!(recombine(&inputs), vec![9.0, 15.0, 8.0]);
        // A second point confirms the split is value-independent (it carries no folded constants).
        let inputs = [2.0f64, 7.0f64];
        assert_eq!(recombine(&inputs), program.interpret(inputs.to_vec()).unwrap());
        assert_eq!(recombine(&inputs), vec![4.0, 14.0, 9.0]);
    }

    /// With every input unknown, the split puts the whole computation on the unknown side: nothing is known, there are
    /// no residuals, and the known program has no outputs. With every input known, the mirror image holds: everything
    /// lands on the known side and the unknown program is empty.
    #[test]
    fn test_partial_eval_split_handles_all_known_and_all_unknown() {
        let build = || {
            let mut builder = ProgramBuilder::<DataType, f64, ScalarOperation<f64>>::new();
            let a = builder.add_input(DataType::F64);
            let x = builder.add_input(DataType::F64);
            let product = builder.add_instruction(MulOperation, vec![a, x]).unwrap()[0];
            let sum = builder.add_instruction(AddOperation, vec![product, a]).unwrap()[0];
            builder
                .build::<Vec<f64>, Vec<f64>>(vec![product, sum], vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };

        // All inputs unknown: the unknown side is the whole computation; the known side is empty with no residuals.
        let program = build();
        let split = program.partial_eval_split(&[true, true]).unwrap();
        assert_eq!(split.output_unknowns, vec![true, true]);
        assert_eq!(split.residual_count, 0);
        assert_eq!(split.known_input_indices, Vec::<usize>::new());
        assert_eq!(split.unknown_input_indices, vec![0, 1]);
        assert_eq!(split.known_program.outputs().count(), 0);
        assert_eq!(split.known_program.instructions().len(), 0);
        assert_eq!(split.unknown_program.instructions().len(), 2);
        assert_eq!(split.unknown_program.interpret(vec![3.0, 5.0]).unwrap(), vec![15.0, 18.0]);

        // All inputs known: the known side is the whole computation; the unknown side is empty with no residuals.
        let program = build();
        let split = program.partial_eval_split(&[false, false]).unwrap();
        assert_eq!(split.output_unknowns, vec![false, false]);
        assert_eq!(split.residual_count, 0);
        assert_eq!(split.known_input_indices, vec![0, 1]);
        assert_eq!(split.unknown_input_indices, Vec::<usize>::new());
        assert_eq!(split.unknown_program.outputs().count(), 0);
        assert_eq!(split.unknown_program.instructions().len(), 0);
        assert_eq!(split.known_program.instructions().len(), 2);
        assert_eq!(split.known_program.interpret(vec![3.0, 5.0]).unwrap(), vec![15.0, 18.0]);
    }

    /// A nested-program / control-flow operation is treated as an ordinary operation by the structural split: there is
    /// no special split rule. With an unknown predicate the whole `condition` lands on the unknown side, and the known
    /// side contributes only the residuals the condition consumes. Recombination still reproduces the original program.
    #[test]
    fn test_partial_eval_split_treats_control_flow_as_ordinary() {
        let scalar = || ArrayType::scalar(DataType::F64);

        // Branches over `[x, a]`: true branch `x * 2`, false branch `x + a`.
        let true_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let x = builder.add_input(scalar());
            let _a = builder.add_input(scalar());
            let two = builder.add_constant(TestArray::scalar(2.0));
            let output = builder.add_instruction(MulOperation, vec![x, two]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let false_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let x = builder.add_input(scalar());
            let a = builder.add_input(scalar());
            let output = builder.add_instruction(AddOperation, vec![x, a]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        // `condition(p, x, a)` plus a known-only output `a * a`, over `[predicate, x, a]`.
        let condition = ConditionOperation::new(true_branch(), false_branch()).unwrap();
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let x = builder.add_input(scalar());
        let a = builder.add_input(scalar());
        let conditional = builder
            .add_instruction(ArrayOperation::Condition(Box::new(condition)), vec![predicate, x, a])
            .unwrap()[0];
        let aa = builder.add_instruction(MulOperation, vec![a, a]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![conditional, aa], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        // `predicate` and `x` unknown, `a` known.
        let split = program.partial_eval_split(&[true, true, false]).unwrap();

        // The conditional output is unknown; the `a * a` output is known.
        assert_eq!(split.output_unknowns, vec![true, false]);
        assert_eq!(split.known_input_indices, vec![2]);
        assert_eq!(split.unknown_input_indices, vec![0, 1]);
        // The condition consumes `a` (known), so it is threaded as a residual; the whole condition survives on the
        // unknown side unchanged (no inlining).
        assert!(split.residual_count > 0);
        assert!(matches!(split.unknown_program.instructions()[0].operation(), ArrayOperation::Condition(_)));

        let recombine = |predicate: bool, x_value: f64, a_value: f64| -> Vec<Vec<f64>> {
            let inputs = [
                TestArray::new(ArrayType::scalar(DataType::Boolean), vec![predicate as u8 as f64]),
                TestArray::scalar(x_value),
                TestArray::scalar(a_value),
            ];
            let known_inputs =
                split.known_input_indices.iter().map(|&index| inputs[index].clone()).collect::<Vec<_>>();
            let mut known_outputs = split.known_program.interpret(known_inputs).unwrap();
            let residuals = known_outputs.split_off(known_outputs.len() - split.residual_count);
            let mut unknown_inputs =
                split.unknown_input_indices.iter().map(|&index| inputs[index].clone()).collect::<Vec<_>>();
            unknown_inputs.extend(residuals);
            let unknown_outputs = split.unknown_program.interpret(unknown_inputs).unwrap();
            let mut known_outputs = known_outputs.into_iter();
            let mut unknown_outputs = unknown_outputs.into_iter();
            split
                .output_unknowns
                .iter()
                .map(|&unknown| if unknown { unknown_outputs.next().unwrap() } else { known_outputs.next().unwrap() })
                .map(|array| array.values)
                .collect()
        };
        let original = |predicate: bool, x_value: f64, a_value: f64| -> Vec<Vec<f64>> {
            let inputs = vec![
                TestArray::new(ArrayType::scalar(DataType::Boolean), vec![predicate as u8 as f64]),
                TestArray::scalar(x_value),
                TestArray::scalar(a_value),
            ];
            program.interpret(inputs).unwrap().into_iter().map(|array| array.values).collect()
        };

        assert_eq!(recombine(true, 4.0, 3.0), original(true, 4.0, 3.0));
        assert_eq!(recombine(true, 4.0, 3.0), vec![vec![8.0], vec![9.0]]);
        assert_eq!(recombine(false, 4.0, 3.0), original(false, 4.0, 3.0));
        assert_eq!(recombine(false, 4.0, 3.0), vec![vec![7.0], vec![9.0]]);
    }
}
