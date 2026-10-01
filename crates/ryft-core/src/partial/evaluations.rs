use std::collections::HashSet;
use std::fmt::Debug;

use crate::contexts::{Context, EagerContext};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::partial::contexts::{PartialEvaluationContext, ReferencePlacement};
use crate::partial::operations::PartiallyEvaluatableOperation;
use crate::partial::values::{PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue, PartialValue};
use crate::programs::{Operation, Program, ProgramError, RegionRef, Type, Typed, Value};
use crate::tracing::TracingContext;

#[cfg(doc)]
use crate::contexts::StagingContext;
#[cfg(doc)]
use crate::programs::ProgramBuilder;

/// Result of partially evaluating a [`Program`] against a known-side [`Context`]. The residual program operates in
/// the *staged constant* space `C::Constant`, while the feeders that connect it to the known side flow as `C::Value`s.
/// Under an eager known-side context the two coincide and every [`PartialEvaluationInput::Known`] carries a concrete
/// folded value, while under a staging known-side context the feeders are [`Tracer`](crate::Tracer)s naming atoms of
/// the *outer* program that partial evaluation folded the known work into. To reconstruct the original program's
/// outputs, one must build the residual program's input vector by mapping each input from [`inputs`](Self::inputs)
/// to either a runtime unknown-input value or its carried known residual, replay [`program`](Self::program) in the
/// known-side context, and then read each output from [`outputs`](Self::outputs) as either its folded value or the
/// indexed residual program output.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
pub struct PartialEvaluation<C: Context> {
    /// Refer to the documentation of [`program`](Self::program) for more information.
    pub(crate) program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,

    /// Refer to the documentation of [`inputs`](Self::inputs) for more information.
    pub(crate) inputs: Vec<PartialEvaluationInput<C::Value>>,

    /// Refer to the documentation of [`outputs`](Self::outputs) for more information.
    pub(crate) outputs: Vec<PartialEvaluationOutput<C::Value>>,
}

impl<C: Context> PartialEvaluation<C> {
    /// Returns the residual [`Program`] of this [`PartialEvaluation`], over the surviving unknown inputs plus the known
    /// residuals, aligned with [`inputs`](Self::inputs) and producing the unknown outputs in their original order.
    #[inline]
    pub fn program(&self) -> &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>> {
        &self.program
    }

    /// Returns the [`PartialEvaluationInput`]s of [`program`](Self::program), in residual program input order.
    #[inline]
    pub fn inputs(&self) -> &[PartialEvaluationInput<C::Value>] {
        &self.inputs
    }

    /// Returns the [`PartialEvaluationOutput`]s of [`program`](Self::program), in original output order.
    #[inline]
    pub fn outputs(&self) -> &[PartialEvaluationOutput<C::Value>] {
        &self.outputs
    }

    /// Returns the positions of reference-typed [`Known`](PartialEvaluationInput::Known) values among
    /// [`inputs`](Self::inputs). These positions index the residual program's inputs and not the original program's
    /// inputs. Unknown reference inputs are excluded.
    ///
    /// A known reference input carries a live handle and not a snapshot of its mutable contents. Under an eager known
    /// side it is the handle itself. Under a staging known side it is the tracer naming the outer program's reference
    /// atom. The residual program therefore observes the state when it runs. Such an input is needed when an access
    /// must stage (e.g., because of an unknown instruction input, an earlier deferred ordered effect, or
    /// [`Stage`](ReferencePlacement::Stage) placement) or when the residual program forwards the handle.
    ///
    /// Reference-typed program constants are excluded too: replay lifts them inline, so the residual program reaches
    /// them without an input carrying a known value.
    #[inline]
    pub fn known_reference_inputs(&self) -> impl '_ + Iterator<Item = usize> {
        self.inputs.iter().enumerate().filter_map(|(index, input)| match input {
            PartialEvaluationInput::Known(value) if value.r#type().is_reference() => Some(index),
            _ => None,
        })
    }
}

impl<C: Context<Operation: Debug>> Debug for PartialEvaluation<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PartialEvaluation")
            .field("program", &self.program)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish()
    }
}

impl<C: Context> PartialEvaluation<C> {
    /// Interprets the residual [`Program`] that this [`PartialEvaluation`] represents in the provided `context` and
    /// at the provided unknown input values, and reassembles the original program's outputs, in original output order.
    /// This is the single replay path for both known-side flavors: residual program constants are lifted through
    /// [`Context::lift`] and [`Instruction`](crate::Instruction)s are bound through [`Context::bind`], and so under
    /// an eager context the residual program is interpreted immediately, while under a [`StagingContext`] it is staged
    /// into the outer program that context is building. Each residual input is fed either by its carried known residual
    /// (i.e., a [`Known`](PartialEvaluationInput::Known) feeder) or by the next value of `inputs` (i.e., an
    /// [`Unknown`](PartialEvaluationInput::Unknown) feeder). Folded outputs are returned directly and the rest
    /// read the replayed residual program's outputs.
    ///
    /// # Parameters
    ///
    ///   - `context`: Known-side context to interpret the residual program in.
    ///   - `inputs`: Values for the original program's surviving *unknown* inputs only, in their original relative
    ///     order. The known inputs are fed from the carried residual feeders, and so the size of `inputs` must equal
    ///     the number of [`Unknown`](PartialEvaluationInput::Unknown) feeders exactly (and not the original program's
    ///     number of inputs).
    pub fn interpret(&self, context: &C, inputs: &[C::Value]) -> Result<Vec<C::Value>, ProgramError> {
        let unknown_count = self.inputs.iter().filter(|i| matches!(i, PartialEvaluationInput::Unknown(_))).count();
        if inputs.len() != unknown_count {
            return Err(ProgramError::InvalidInputCount { expected: unknown_count, actual: inputs.len() });
        }
        let mut remaining_inputs = inputs.iter();
        let residual_inputs = self
            .inputs
            .iter()
            .map(|feeder| match feeder {
                PartialEvaluationInput::Known(value) => value.clone(),
                PartialEvaluationInput::Unknown(_) => {
                    // The `.unwrap()` here is safe because of the earlier check for `inputs.len()`.
                    remaining_inputs.next().cloned().unwrap()
                }
            })
            .collect::<Vec<_>>();
        let residual_outputs = self.program.interpret_in_context(context, residual_inputs)?;
        self.outputs
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(value) => Ok(value.clone()),
                PartialEvaluationOutput::Unknown(index) => residual_outputs.get(*index).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "partial evaluation output references residual output {index} but the residual program \
                         produced {} output(s)",
                        residual_outputs.len(),
                    ))
                }),
            })
            .collect()
    }
}

impl<V: Value, O: Operation<Type = V::Type>> RegionRef<'_, V, O> {
    /// Partially evaluates this borrowed [`Region`](crate::Region) through the provided known-side context without
    /// materializing it. Refer to [`Program::partially_evaluate_in_context`] for the input and output conventions.
    ///
    /// Reference operations use [`ReferencePlacement::Stage`]. With an eager context they remain in the residual
    /// program, so specialization does not read or mutate live reference state. With a staging context, folding records
    /// work in the parent program instead of executing it, so reference placement adds no restriction. Disabling effect
    /// folding retains all effectful operations in the residual program in either case; pure known work can still fold.
    ///
    /// # Parameters
    ///
    ///   - `context`: Parent context through which known work is evaluated or staged.
    ///   - `inputs`: Known values or unknown input types, in region input order.
    ///   - `allow_effect_folding`: Specifies whether effects may fold into the parent when input knownness, reference
    ///     placement, and execution order constraints permit it. When `false`, every effectful operation remains
    ///     residual.
    pub fn partially_evaluate_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        self,
        context: &C,
        inputs: &[PartialValue<C::Value>],
        allow_effect_folding: bool,
    ) -> Result<PartialEvaluation<C>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        check_count!("input", inputs, self.input_ids().len(), ProgramError);
        let context = PartialEvaluationContext::new(context.clone())
            .with_reference_placement(ReferencePlacement::Stage)
            .with_allow_effect_folding(allow_effect_folding);
        let mut seed = Vec::with_capacity(inputs.len());
        for (index, knowledge) in inputs.iter().enumerate() {
            match knowledge {
                PartialValue::Known(value) => seed.push(PartialEvaluationValue::known_input(value.clone())),
                PartialValue::Unknown(r#type) => seed.push(context.unknown_input(r#type.clone(), index)),
            }
        }
        let outputs = context.inline_region(self, seed, &HashSet::new(), None, None)?;
        context.into_evaluation(outputs)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>> {
    /// Partially evaluates this [`Program`] against the provided [`PartialValue`] inputs, folding known work eagerly.
    /// This is the main partial evaluation entry point, instantiated at this program's own [`EagerContext`] so that
    /// known values are concrete values and folding interprets each all-known [`Instruction`](crate::Instruction)
    /// immediately. [`partially_evaluate_in_context`](Self::partially_evaluate_in_context) is the [`Context`]-taking
    /// core it delegates to and must be used instead with a [`StagingContext`] to fold known work into an enclosing
    /// trace.
    ///
    /// Partial evaluation classifies each [`Atom`](crate::Atom) as *known* (i.e., computable _now_ from the provided
    /// values) or *unknown* (i.e., dependent on a runtime input), folds the known subcomputation away, and carves the
    /// remaining unknown subcomputation into a residual [`Program`] that consumes only the unknown inputs plus the
    /// known values it actually needs. During partial evaluation, each instruction is first offered to its own
    /// [`PartiallyEvaluatableOperation::partially_evaluate`] implementation, which may override the default behavior.
    /// For example, a `condition` with a concretizable known predicate calls
    /// [`PartialEvaluationContext::inline_program`] to inline its selected branch in place of the operation, so that
    /// the condition disappears from the residual program. Building the residual program with a [`ProgramBuilder`]
    /// (rather than projecting the original) is what lets these rules emit *transformed* work; flat instructions with
    /// no override are emitted unchanged. The walk is flat per program but can recurse through operation rules into
    /// inlined nested programs, such as a selected `condition` branch; an instruction carrying a nested program that
    /// is *not* inlined is folded only when all of its inputs are known and is otherwise emitted unchanged.
    ///
    /// Each known *variable* a residualized instruction consumes, whether a program input or a folded intermediate,
    /// becomes a residual input of the residual program. Literal constants are rebuilt inline as residual-program
    /// constants (their staged payload is recovered through [`Context::resolve`]), so they are never residual inputs.
    /// The resulting [`PartialEvaluation`] carries everything a caller needs to reassemble the original outputs once
    /// the runtime (i.e., unknown) inputs are available.
    ///
    /// # Relationship to [`partially_evaluate_in_context`](Self::partially_evaluate_in_context)
    ///
    /// This function is the **eager** convenience form of partial evaluation: it evaluates known work under an
    /// [`EagerContext`], holding concrete known values and *folding the known subcomputation away* (through
    /// [`Context::bind`]) while applying per-operation rewrite rules, and yields a single residual [`Program`] with the
    /// folded output and residual-input *values*. Use it to **specialize or constant-fold** a program against inputs
    /// that are known. [`partially_evaluate_in_context`](Self::partially_evaluate_in_context) is the context-generic
    /// core behind it: passing a live [`StagingContext`] instead splits the program *online* against values known to an
    /// enclosing trace, staging the known work into the outer program rather than folding it to concrete values. The
    /// rewrite rules, residual construction, and output classification are identical across both; only the known-side
    /// [`Context`] differs.
    #[inline]
    pub fn partially_evaluate(
        &self,
        inputs: &[PartialValue<V>],
    ) -> Result<PartialEvaluation<EagerContext<V, O>>, ProgramError>
    where
        O: InterpretableOperation<EagerContext<V, O>>
            + PartiallyEvaluatableOperation<EagerContext<V, O>>
            + PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        self.partially_evaluate_in_context(&EagerContext::new(), inputs)
    }

    /// Partially evaluates this [`Program`] against the provided [`PartialValue`] inputs, folding known work through
    /// the provided known-side [`Context`]. This is the context-taking core behind
    /// [`partially_evaluate`](Self::partially_evaluate).
    #[inline]
    pub fn partially_evaluate_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        &self,
        context: &C,
        inputs: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        self.entry_region_ref().partially_evaluate_in_context(context, inputs, true)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceTransform, ArrayReferenceTransformIndex,
        ArrayType, DataType,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::{
        AddOperation, ConditionOperation, MulOperation, PrintOperation, ReferenceAddUpdateOperation,
        ReferenceReadOperation, ReferenceWriteOperation,
    };
    use crate::parameters::Placeholder;
    use crate::partial::tests::{TestOperation, TestValue};
    use crate::partial::values::{PartialEvaluationInput, PartialEvaluationOutput, PartialValue};
    use crate::programs::{
        AtomId, Operation, ProgramBuilder, ProgramError, Provenance, ProvenanceScope, ReferenceType, Typed,
    };
    use crate::tests::{
        TestArrayContext, TestArrayIrContext, TestArrayIrOperation, TestArrayOperation, TestArrayTracingContext,
    };
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_partial_evaluation_known_reference_inputs() {
        // Forward a mixture of known arrays, known references, and unknown references through a valid residual
        // boundary. Only the known reference positions are reported.
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        let inputs = vec![
            PartialEvaluationInput::Unknown(0),
            PartialEvaluationInput::Known(ArrayIrValue::Array(Array::scalar(2.0_f32).unwrap())),
            PartialEvaluationInput::Known(reference.clone()),
            PartialEvaluationInput::Unknown(1),
            PartialEvaluationInput::Known(reference.clone()),
        ];
        let mut builder = ProgramBuilder::<TestValue, TestArrayIrOperation>::new();
        let outputs = inputs
            .iter()
            .map(|input| {
                builder.add_input(match input {
                    PartialEvaluationInput::Known(value) => value.r#type().into_owned(),
                    PartialEvaluationInput::Unknown(_) => reference.r#type().into_owned(),
                })
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 5], vec![Placeholder; 5])
            .unwrap();
        let evaluation = PartialEvaluation::<TestArrayIrContext> {
            program,
            inputs,
            outputs: (0..5).map(PartialEvaluationOutput::Unknown).collect(),
        };
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![2, 4]);
    }

    #[test]
    fn test_partial_evaluation_interpret() {
        // Build a residual program `g(x, r) = x * r + 3`, with `x` standing for the original program's surviving
        // unknown input and `r` for a known residual feeder carrying the folded value `2`, and pair it with an
        // original output report whose first output folded to `5`.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let r = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(3.0).unwrap());
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, r], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![product, c], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let evaluation = PartialEvaluation::<TestArrayContext> {
            program: program.clone(),
            inputs: vec![
                PartialEvaluationInput::Unknown(1),
                PartialEvaluationInput::Known(Array::scalar(2.0).unwrap()),
            ],
            outputs: vec![
                PartialEvaluationOutput::Known(Array::scalar(5.0).unwrap()),
                PartialEvaluationOutput::Unknown(0),
            ],
        };

        // Interpretation takes exactly one value per `Unknown` feeder, feeds `Known` feeders from their carried
        // values, returns folded outputs directly, and reads the rest from the replayed residual program:
        // `(5, 4 * 2 + 3) = (5, 11)`.
        let context = TestArrayContext::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(4.0).unwrap()]),
            Ok(vec![Array::scalar(5.0).unwrap(), Array::scalar(11.0).unwrap()]),
        );
        assert!(matches!(
            evaluation.interpret(&context, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ),);
        assert!(matches!(
            evaluation.interpret(&context, &[Array::scalar(4.0).unwrap(), Array::scalar(5.0).unwrap()]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 2 }),
        ),);

        // An output that references a residual output the residual program does not produce is reported as a
        // malformed program.
        let evaluation = PartialEvaluation::<TestArrayContext> {
            program: program.clone(),
            inputs: evaluation.inputs,
            outputs: vec![PartialEvaluationOutput::Unknown(1)],
        };
        assert!(matches!(
            evaluation.interpret(&context, &[Array::scalar(4.0).unwrap()]),
            Err(ProgramError::MalformedProgram(message))
                if message == "partial evaluation output references residual output 1 but the residual program \
                    produced 1 output(s)",
        ),);

        // Under a staging known-side context, the same replay stages the residual program into the outer trace
        // instead of executing it. Its constant is lifted as a staged constant, its instructions are staged as outer
        // instructions, folded outputs return their tracers directly, and residual outputs are tracers naming the
        // staged atoms.
        let outer = TestArrayTracingContext::new();
        let folded = outer.input(ArrayType::scalar(DataType::F64));
        let unknown = outer.input(ArrayType::scalar(DataType::F64));
        let feeder = outer.constant(Array::scalar(2.0).unwrap());
        let evaluation = PartialEvaluation::<TestArrayTracingContext> {
            program,
            inputs: vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(feeder)],
            outputs: vec![PartialEvaluationOutput::Known(folded.clone()), PartialEvaluationOutput::Unknown(0)],
        };
        let outputs = evaluation.interpret(&outer, &[unknown]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].atom_id(), folded.atom_id());
        let staged = outputs[1].atom_id().unwrap();
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![staged], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = const 2.0
                    %3:f64[] = const 3.0
                    %4:f64[] = mul %1 %2
                    %5:f64[] = add %4 %3
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_partially_evaluate() {
        // `f(a, x) = (a * a, a * a * x + 1, a * a + x)` with `a` known and `x` unknown: the `a * a` subcomputation
        // folds to a known output, its two residual consumers share one residual feeder, and the literal is rebuilt
        // inline as a residual constant instead of becoming a feeder.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(1.0).unwrap());
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, x], None).unwrap()[0];
        let shifted = builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, c], None).unwrap()[0];
        let offset = builder.add_instruction(AddOperation::new(), Vec::new(), vec![squared, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared, shifted, offset], vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Array::scalar(9.0).unwrap()),],
        );
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Array::scalar(9.0).unwrap()),
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
            ],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = const 1.0
                    %4:f64[] = add %2 %3
                    %5:f64[] = add %1 %0
                in (%4, %5)
            "}
            .trim_end(),
        );

        // Replaying the partial evaluation at a concrete unknown input matches interpreting the original program.
        assert_eq!(
            evaluation.interpret(&EagerContext::<Array, ArrayOperation<Array>>::new(), &[Array::scalar(4.0).unwrap()]),
            Ok(vec![Array::scalar(9.0).unwrap(), Array::scalar(37.0).unwrap(), Array::scalar(13.0).unwrap()]),
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(3.0).unwrap(), Array::scalar(4.0).unwrap()]),
            Ok(vec![Array::scalar(9.0).unwrap(), Array::scalar(37.0).unwrap(), Array::scalar(13.0).unwrap()]),
        );

        // All-known inputs fold the whole program away: every output is known and the residual program is empty.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Known(Array::scalar(4.0).unwrap()),
            ])
            .unwrap();
        assert_eq!(evaluation.inputs, Vec::new());
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Array::scalar(9.0).unwrap()),
                PartialEvaluationOutput::Known(Array::scalar(37.0).unwrap()),
                PartialEvaluationOutput::Known(Array::scalar(13.0).unwrap()),
            ],
        );
        assert!(evaluation.program.instructions().is_empty());

        // All-unknown inputs residualize the whole program unchanged, with the literal rebuilt inline at its first
        // residual use rather than up front.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(evaluation.inputs, vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Unknown(1)]);
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
                PartialEvaluationOutput::Unknown(2),
            ],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %0
                    %3:f64[] = mul %2 %1
                    %4:f64[] = const 1.0
                    %5:f64[] = add %3 %4
                    %6:f64[] = add %2 %1
                in (%2, %5, %6)
            "}
            .trim_end(),
        );

        // Effectful operations place by input known-ness. An all-known `print` folds (firing its effect at partial
        // evaluation time), while a mixed-input `print` residualizes and is kept in the residual program even when
        // no output consumes it.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let printed = builder.add_instruction(PrintOperation::new("known"), Vec::new(), vec![a], None).unwrap()[0];
        builder.add_instruction(PrintOperation::new("dead"), Vec::new(), vec![x], None).unwrap();
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![printed, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(2.0).unwrap()),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Array::scalar(2.0).unwrap()),],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = print [label=dead] %0
                    %3:f64[] = mul %1 %0
                in (%3)
            "}
            .trim_end(),
        );

        // The number of provided inputs must match the number of program inputs.
        assert!(matches!(
            program.partially_evaluate(&[PartialValue::Known(Array::scalar(1.0).unwrap())]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        ),);
    }

    #[test]
    fn test_program_partially_evaluate_preserves_residual_provenance() {
        // `f(a, x) = (a * a * x + 1, print(x))` with `a` known and `x` unknown. Every residual instruction is a
        // deferred rewrite of one source instruction, so it must carry that instruction's provenance. The folded
        // known-side `a * a` contributes no residual instruction and so its scope must not appear.
        let scoped = |name: &str| Provenance::scope(ProvenanceScope::new(name), Provenance::unknown());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(Array::scalar(1.0).unwrap());
        let squared =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, a], Some(scoped("known"))).unwrap()[0];
        let scaled = builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![squared, x], Some(scoped("scaled")))
            .unwrap()[0];
        let shifted = builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![scaled, one], Some(scoped("shifted")))
            .unwrap()[0];
        let printed = builder
            .add_instruction(PrintOperation::new("x"), Vec::new(), vec![x], Some(scoped("printed")))
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![shifted, printed], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation
                .program
                .instructions()
                .iter()
                .map(|instruction| (instruction.operation().name(), instruction.provenance().clone()))
                .collect::<Vec<_>>(),
            // The effectful `print` residualizes ahead of the pure chain, which is a placement property of partial
            // evaluation; what matters here is that each residual instruction carries its own source provenance.
            vec![("print", scoped("printed")), ("mul", scoped("scaled")), ("add", scoped("shifted"))],
        );
    }

    #[test]
    fn test_program_partially_evaluate_with_residual_references() {
        // `f(r, x) = { add_update(r, 1); write(r, x); read(r) }` over a live reference `r` and an unknown `x`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type.clone().into());
        let x = builder.add_input(scalar_type.clone().into());
        let one = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32).unwrap()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, one], None)
            .unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, x], None)
            .unwrap();
        let read =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Specialization never touches live state: every reference operation residualizes even though the reference is
        // known, and the live handle becomes a residual reference threaded by identity.
        let live = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(TestValue::Reference(live.clone())),
                PartialValue::Unknown(scalar_type.clone().into()),
            ])
            .unwrap();
        assert_eq!(live.read(), Ok(Array::scalar(2.0_f32).unwrap()));
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(TestValue::Reference(live.clone()))],
        );
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]> .
                let %2:f32[] = const 1.0
                    () = reference_add_update %1 %2
                    () = reference_write %1 %0
                    %3:f32[] = reference_read %1
                in (%3)
            "}
            .trim_end(),
        );

        // Replaying the residual program binds the live reference by identity: the deferred accesses run against the
        // same allocation, so the read observes the write and the handle reflects it afterwards.
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<TestValue, TestOperation>::new(),
                &[TestValue::Array(Array::scalar(5.0_f32).unwrap())],
            ),
            Ok(vec![TestValue::Array(Array::scalar(5.0_f32).unwrap())]),
        );
        assert_eq!(live.read(), Ok(Array::scalar(5.0_f32).unwrap()));
    }

    #[test]
    fn test_program_partially_evaluate_in_context() {
        // `f(a, x) = (a * a) * x + 1` with `a` known as a live tracer of an enclosing trace and `x` unknown: the known
        // `a * a` folds by staging into the outer program, the residual program consumes its staged result through a
        // known feeder naming the outer atom, and the literal is rebuilt inline as a residual constant.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(1.0).unwrap());
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, x], None).unwrap()[0];
        let shifted = builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, c], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![shifted], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = TestArrayTracingContext::new();
        let known = outer.input(ArrayType::scalar(DataType::F64));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(ArrayType::scalar(DataType::F64))],
            )
            .unwrap();

        // The known feeder is a tracer naming the staged `a * a` atom of the outer program (atom 2, since the replay
        // lifts the live program constant into the outer trace up front, before replaying any instruction).
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(
            &evaluation.inputs[1],
            PartialEvaluationInput::Known(feeder) if feeder.atom_id() == Ok(AtomId::new(2)),
        ),);
        assert_eq!(evaluation.outputs.len(), 1);
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = const 1.0
                    %4:f64[] = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );

        // The outer trace accumulated the lifted literal followed by the folded known work. The literal stays dead
        // in the outer trace because the residual program rebuilds it inline.
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![AtomId::new(2)], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 1.0
                    %2:f64[] = mul %0 %0
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_partially_evaluate_with_known_root_and_unknown_transform_binding() {
        let root_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let index_type = ArrayType::scalar(DataType::I32);
        let transforms = vec![
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Static(1) },
            ArrayReferenceTransform::Index { axis: 0, index: ArrayReferenceTransformIndex::Dynamic },
        ];
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let root = builder.add_input(ReferenceType::new(root_type).into());
        let index = builder.add_input(index_type.clone().into());
        let output = builder
            .add_instruction(
                ReferenceReadOperation::new().with_transforms(transforms),
                Vec::new(),
                vec![root, index],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let live = ArrayReference::new(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(TestValue::Reference(live.clone())),
                PartialValue::Unknown(index_type.into()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.inputs(),
            &[PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(TestValue::Reference(live.clone())),],
        );
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:i32[], %1:ref<f32[2, 3]> .
                let %2:f32[] = reference_read [transforms=[index(axis=0, index=1), index(axis=0, index=dynamic)]] %1 %0
                in (%2)"},
        );
        assert_eq!(live.read(), Ok(Array::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap()));
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<TestValue, TestOperation>::new(),
                &[TestValue::Array(Array::scalar(2i32).unwrap())]
            ),
            Ok(vec![TestValue::Array(Array::scalar(6f32).unwrap())]),
        );
    }

    #[test]
    fn test_program_partially_evaluate_in_context_with_residual_references() {
        // `f(r, x) = { add_update(r, 1); write(r, x); read(r) }` over a live reference `r` and an unknown `x`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type.clone().into());
        let x = builder.add_input(scalar_type.clone().into());
        let one = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32).unwrap()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, one], None)
            .unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, x], None)
            .unwrap();
        let read =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Under a staging known side the known update folds into the outer program, the write of the unknown value
        // stages and prevents later ordered effects from folding, and the read stages over the root as a residual
        // reference. Replaying the residual program in the outer trace threads the outer reference atom by identity,
        // never a snapshot.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let reference = outer.input(reference_type.into());
        let x = outer.input(scalar_type.clone().into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(reference), PartialValue::Unknown(scalar_type.into())],
            )
            .unwrap();
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        let outputs = evaluation.interpret(&outer, &[x]).unwrap();
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![outputs[0].atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[] .
                let %2:f32[] = const 1.0
                    () = reference_add_update %0 %2
                    () = reference_write %0 %1
                    %3:f32[] = reference_read %0
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_partially_evaluate_in_context_preserves_order_in_selected_condition_branch() {
        // `f(p, a, b, x) = { if p { write(a, x) } else { write(b, x) }; (read(a), read(b)) }` over two reference roots.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let branch = |written: usize| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let references =
                [builder.add_input(reference_type.clone().into()), builder.add_input(reference_type.clone().into())];
            let x = builder.add_input(scalar_type.clone().into());
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![references[written], x], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(branch(0));
        let false_branch = builder.import_program(branch(1));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let a = builder.add_input(reference_type.clone().into());
        let b = builder.add_input(reference_type.clone().into());
        let x = builder.add_input(scalar_type.clone().into());
        builder
            .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], vec![predicate, a, b, x], None)
            .unwrap();
        let read_a = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let read_b = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![b], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read_a, read_b], vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();

        // A known predicate inlines only the selected branch. Its residual write prevents later ordered effects from
        // folding, so both reads stage after it; the unselected branch contributes no write.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let a = outer.input(reference_type.clone().into());
        let b = outer.input(reference_type.clone().into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.constant(TestValue::Array(Array::scalar(true).unwrap()))),
                    PartialValue::Known(a.clone()),
                    PartialValue::Known(b.clone()),
                    PartialValue::Unknown(scalar_type.clone().into()),
                ],
            )
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]>, %2:ref<f32[]> .
                let () = reference_write %1 %0
                    %3:f32[] = reference_read %1
                    %4:f32[] = reference_read %2
                in (%3, %4)
            "}
            .trim_end(),
        );
        let first = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let second = ArrayReference::new(Array::scalar(3.0_f32).unwrap());
        assert_eq!(
            evaluation.program.interpret(vec![
                TestValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestValue::Reference(first.clone()),
                TestValue::Reference(second.clone()),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestValue::Array(Array::scalar(3.0_f32).unwrap())
            ]),
        );
        assert_eq!(first.read(), Ok(Array::scalar(5.0_f32).unwrap()));
        assert_eq!(second.read(), Ok(Array::scalar(3.0_f32).unwrap()));

        // Specializing a false predicate selects the other root before the residual program is built.
        let false_evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.constant(TestValue::Array(Array::scalar(false).unwrap()))),
                    PartialValue::Known(a.clone()),
                    PartialValue::Known(b.clone()),
                    PartialValue::Unknown(scalar_type.clone().into()),
                ],
            )
            .unwrap();
        assert_eq!(
            false_evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]>, %2:ref<f32[]> .
                let () = reference_write %1 %0
                    %3:f32[] = reference_read %2
                    %4:f32[] = reference_read %1
                in (%3, %4)
            "}
            .trim_end(),
        );

        assert!(matches!(
            &false_evaluation.inputs[1],
            PartialEvaluationInput::Known(reference) if reference.atom_id() == b.atom_id(),
        ),);
        assert!(matches!(
            &false_evaluation.inputs[2],
            PartialEvaluationInput::Known(reference) if reference.atom_id() == a.atom_id(),
        ),);

        // An unknown predicate residualizes the whole conditional. Its ordered effects require both later reads
        // to remain residual, regardless of which branch eventually runs.
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean).into()),
                    PartialValue::Known(a),
                    PartialValue::Known(b),
                    PartialValue::Unknown(scalar_type.into()),
                ],
            )
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(
            evaluation
                .program
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["condition", "reference_read", "reference_read"],
        );
        // Taking the other branch at runtime must update the second root and preserve the first root.
        assert_eq!(
            evaluation.program.interpret(vec![
                TestValue::Array(Array::scalar(false).unwrap()),
                TestValue::Array(Array::scalar(7.0_f32).unwrap()),
                TestValue::Reference(first.clone()),
                TestValue::Reference(second.clone()),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestValue::Array(Array::scalar(7.0_f32).unwrap())
            ]),
        );
        assert_eq!(first.read(), Ok(Array::scalar(5.0_f32).unwrap()));
        assert_eq!(second.read(), Ok(Array::scalar(7.0_f32).unwrap()));
    }
}
