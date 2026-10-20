use std::collections::HashSet;
use std::fmt::Debug;

use crate::contexts::{Context, EagerContext, StagingContext};
use crate::interpretation::InterpretableOperation;
use crate::partial::contexts::{PartialEvaluationContext, ReferencePlacement};
use crate::partial::operations::PartiallyEvaluatableOperation;
use crate::partial::values::{PartialEvaluationInput, PartialEvaluationOutput, PartialValue};
use crate::programs::{Operation, Program, ProgramError, RegionRef, Type, Typed, Value};
use crate::tracing::TracingContext;

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
    /// [`Stage`](ReferencePlacement::Stage) placement), when the residual program forwards the handle,
    /// or when its referent type shares an identity with an unknown input, which it witnesses.
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

impl<V: Value, O: Operation<Type = V::Type>> RegionRef<'_, V, O>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
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
    ///
    /// # Errors
    ///
    /// Returns the errors documented for [`Program::partially_evaluate_in_context`].
    pub fn partially_evaluate_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        self,
        context: &C,
        inputs: &[PartialValue<C::Value>],
        allow_effect_folding: bool,
    ) -> Result<PartialEvaluation<C>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        let context = PartialEvaluationContext::new(context.clone())
            .with_reference_placement(ReferencePlacement::Stage)
            .with_allow_effect_folding(allow_effect_folding);
        let seed = context.prepare_root_inputs(self, inputs)?;
        let outputs = context.inline_region(self, seed, &HashSet::new(), None, None)?;
        context.into_evaluation(outputs)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    /// Partially evaluates this [`Program`] against the provided [`PartialValue`] inputs, folding known work eagerly.
    /// This is the main partial evaluation entry point, instantiated at this program's own [`EagerContext`] so that
    /// known values are concrete values and folding interprets each all-known [`Instruction`](crate::Instruction)
    /// immediately. [`partially_evaluate_in_context`](Self::partially_evaluate_in_context) is the [`Context`]-taking
    /// core it delegates to and must be used instead with a [`StagingContext`] to fold known work into an enclosing
    /// trace.
    ///
    /// Partial evaluation classifies each [`Atom`](crate::Atom) as *known* (i.e., computable _now_ from the provided
    /// values) or *unknown* (i.e., dependent on a runtime input), folds the known subcomputation away, and carves the
    /// remaining unknown subcomputation into a residual [`Program`] that consumes the unknown inputs plus the known
    /// values it needs. During partial evaluation, each instruction is first offered to its own
    /// [`PartiallyEvaluatableOperation::partially_evaluate`] implementation, which may override the default behavior.
    /// For example, a `condition` with a concretizable known predicate calls
    /// [`partially_evaluate_region`](crate::PartialEvaluationDriver::partially_evaluate_region) to inline its selected
    /// branch through the branch's guarded entry in place of the operation, so that the condition disappears from the
    /// residual program. Building the residual program with a [`ProgramBuilder`] (rather than projecting the original)
    /// is what lets these rules emit *transformed* work; flat instructions with no override are emitted unchanged. The
    /// walk is flat per program but can recurse through operation rules into inlined nested programs, such as a
    /// selected `condition` branch; an instruction carrying a nested program that is *not* inlined is folded only when
    /// all of its inputs are known and is otherwise emitted unchanged.
    ///
    /// Each known *variable* a residualized instruction consumes, whether a program input or a folded intermediate,
    /// becomes a residual input of the residual program. Literal constants are rebuilt inline as residual-program
    /// constants (their staged payload is recovered through [`Context::resolve`]), so they are never residual inputs.
    /// The resulting [`PartialEvaluation`] carries everything a caller needs to reassemble the original outputs once
    /// the runtime (i.e., unknown) inputs are available.
    ///
    /// Partial evaluation replays the program's body in place of its entry. The residual program's entry keeps the
    /// supplied types of the unknown inputs, so it accepts every value that the original program accepts, while the
    /// body is specialized at the facts that the entry proves. Identity relationships that the input types leave
    /// unproven are checked by an entry [`GuardOperation`](crate::GuardOperation) staged before anything that the body
    /// stages. Known values that share a type identity with an unknown input are also residual inputs, even when no
    /// residualized instruction consumes them, because they witness the extents of that input's symbolic dimensions.
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
    ///
    /// # Errors
    ///
    /// Returns the errors documented for [`partially_evaluate_in_context`](Self::partially_evaluate_in_context).
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
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] when the number of `inputs` differs from the number of
    /// program inputs, a type error when an input type does not refine the corresponding declared input type,
    /// [`ProgramError::UnsupportedOperation`] when the input types leave an entry relationship unproven but the
    /// operation family cannot provide the [`GuardOperation`](crate::GuardOperation) that checks it, and any error
    /// raised while folding or replaying an instruction.
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
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayReferenceTransform,
        ArrayReferenceTransformIndex, ArrayType, DataType, Dimension, DimensionBounds, DimensionError, DimensionType,
        DimensionValue, Shape,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::{
        AddOperation, ConditionOperation, ConstantOperation, DimensionAddOperation, DimensionSizeOperation,
        DotDimensionNumbers, DotOperation, GuardOperation, MulOperation, PrintOperation, ReferenceAddUpdateOperation,
        ReferenceReadOperation, ReferenceWriteOperation, ScanOperation, ZeroOperation,
    };
    use crate::parameters::Placeholder;
    use crate::partial::tests::{TestOperation, TestValue};
    use crate::partial::values::{PartialEvaluationInput, PartialEvaluationOutput, PartialValue};
    use crate::programs::{
        AtomId, Operation, ProgramBuilder, ProgramError, Provenance, ProvenanceScope, ReferenceType, TypeError, Typed,
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
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1f32).unwrap()));
        let inputs = vec![
            PartialEvaluationInput::Unknown(0),
            PartialEvaluationInput::Known(ArrayIrValue::Array(Array::scalar(2f32).unwrap())),
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
    fn test_region_ref_partially_evaluate_in_context() {
        // Two known symbolic dimensions `a` and `b` enter `[dimension<n>, dimension<n>]`, and the body returns the
        // first one and a zero built from it. With effect folding disabled, the entry check of their relationship
        // remains residual, but its unchanged first input and the pure zero computed from it stay known.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, a, b] = ["n", "a", "b"].map(|name| DimensionType::new(name, bounds));
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = builder.add_input(n.clone().into());
        builder.add_input(n.clone().into());
        let zero = builder
            .add_instruction(TestOperation::Zero(ZeroOperation::new(vector(&n))), Vec::new(), vec![first], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![first, zero], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let first = outer.input(a.clone().into());
        let second = outer.input(b.into());
        let evaluation = program
            .entry_region_ref()
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(first.clone()), PartialValue::Known(second)],
                false,
            )
            .unwrap();
        let [PartialEvaluationOutput::Known(dimension), PartialEvaluationOutput::Known(zero)] = evaluation.outputs()
        else {
            panic!("expected unchanged known input and pure known output");
        };
        assert_eq!(dimension, &first);
        assert_eq!(zero.r#type().as_ref(), &ArrayIrType::Array(vector(&a)));
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
            lambda %0:dimension<a ∈ [1, 8)>, %1:dimension<b ∈ [1, 8)> .
            let %2:dimension<a ∈ [1, 8)>, %3:dimension<a ∈ [1, 8)> = guard [
                boundary="partial evaluation",
                types=[dimension<n ∈ [1, 8)>, dimension<n ∈ [1, 8)>],
                retained_identities=[a],
            ] %0 %1
            in ()"#},
        );
        let outputs = evaluation.interpret(&outer, &[]).unwrap();
        let atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(atoms, vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let dimension = |extent| TestValue::Dimension(DimensionValue::constant(extent).unwrap());
        assert_eq!(
            replayed.interpret(vec![dimension(3), dimension(3)]),
            program.interpret(vec![dimension(3), dimension(3)]),
        );
        assert_eq!(
            replayed.interpret(vec![dimension(3), dimension(4)]),
            program.interpret(vec![dimension(3), dimension(4)]),
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
    fn test_program_partially_evaluate_preserves_known_entry_dimensions() {
        // A known `dimension<extent>` of extent 2 next to an unknown `f32[extent]` specializes the body at `extent =
        // 2`. The residual entry keeps the supplied `f32[extent]`, and its entry guard checks the extent and narrows
        // the array.
        let dimension = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![dimension.dimension().clone()]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let extent = builder.add_input(dimension.clone().into());
        let array = builder.add_input(array_type.clone().into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![extent, array], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let extent = TestValue::Dimension(DimensionValue::new(dimension, 2).unwrap());
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(extent.clone()), PartialValue::Unknown(array_type.into())])
            .unwrap();
        let context = EagerContext::<TestValue, TestOperation>::new();
        let matching = TestValue::Array(Array::vector(vec![4f32; 2]).unwrap());
        assert_eq!(
            evaluation.interpret(&context, &[matching.clone()]),
            program.interpret(vec![extent.clone(), matching]),
        );
        let mismatching = TestValue::Array(Array::vector(vec![4f32; 3]).unwrap());
        assert_eq!(
            program.interpret(vec![extent.clone(), mismatching.clone()]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "extent".to_string(),
                expected: 2,
                actual: 3,
            }))),
        );
        // The residual entry keeps the supplied `f32[extent]`, and the entry guard rejects the mismatch exactly like
        // the source program.
        assert_eq!(
            evaluation.interpret(&context, std::slice::from_ref(&mismatching)),
            program.interpret(vec![extent, mismatching]),
        );
        let guard = r#"guard [boundary="partial evaluation", types=[dimension<extent ∈ [1, 8)>, f32[extent]]]"#;
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:f32[extent], %1:dimension<2> .
                let %2:dimension<2>, %3:f32[2] = {guard} %1 %0
                in (%3)"},
        );

        // First-class dimensions admit an independently named actual type with narrower bounds. Facts attached to the
        // formal identity still constrain it even though looking up its actual identity would find no fact.
        let renamed = DimensionType::new("renamed", DimensionBounds::positive(Some(4)).unwrap());
        let known = TestValue::Array(Array::vector(vec![3f32; 2]).unwrap());
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(renamed.clone().into()), PartialValue::Known(known.clone())])
            .unwrap();
        let matching = TestValue::Dimension(DimensionValue::new(renamed.clone(), 2).unwrap());
        assert_eq!(
            evaluation.interpret(&context, std::slice::from_ref(&matching)),
            program.interpret(vec![matching, known.clone()]),
        );
        let mismatching = TestValue::Dimension(DimensionValue::new(renamed, 3).unwrap());
        assert_eq!(
            program.interpret(vec![mismatching.clone(), known.clone()]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "extent".to_string(),
                expected: 3,
                actual: 2,
            }))),
        );
        assert_eq!(
            evaluation.interpret(&context, std::slice::from_ref(&mismatching)),
            program.interpret(vec![mismatching, known]),
        );
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:dimension<renamed ∈ [1, 4)>, %1:f32[2] .
                let %2:dimension<2>, %3:f32[2] = {guard} %0 %1
                in (%2)"},
        );
    }

    #[test]
    fn test_program_partially_evaluate_refines_unknown_inputs_in_the_actual_scope() {
        // The unknown inputs swap the formal identities: `dimension<m>` enters `dimension<n>` and `dimension<n>` enters
        // `dimension<m>`, while the known `f32[2]` fixes the formal `m`. The residual entry keeps both supplied types,
        // and only the actual `n` is narrowed (by the entry guard), so the residual program accepts every extent of the
        // first input that the source program accepts.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m, p] = ["n", "m", "p"].map(|name| DimensionType::new(name, bounds));
        let array =
            |dimension: &DimensionType| ArrayType::new(DataType::F32, Shape::new(vec![dimension.dimension().clone()]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = builder.add_input(n.clone().into());
        let second = builder.add_input(m.clone().into());
        builder.add_input(array(&m).into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![first, second], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let known = TestValue::Array(Array::vector(vec![1f32; 2]).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(m.clone().into()),
                PartialValue::Unknown(n.clone().into()),
                PartialValue::Known(known.clone()),
            ])
            .unwrap();
        let guard = r#"guard [boundary="partial evaluation", types=[dimension<m ∈ [1, 8)>, f32[m]]]"#;
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:dimension<m ∈ [1, 8)>, %1:dimension<n ∈ [1, 8)>, %2:f32[2] .
                let %3:dimension<2>, %4:f32[2] = {guard} %1 %2
                in (%0, %3)"},
        );
        let context = EagerContext::<TestValue, TestOperation>::new();
        let dimension =
            |r#type: &DimensionType, extent| TestValue::Dimension(DimensionValue::new(r#type.clone(), extent).unwrap());
        let exact = |extent| TestValue::Dimension(DimensionValue::constant(extent).unwrap());
        let inputs = [dimension(&m, 3), exact(2)];
        assert_eq!(
            evaluation.interpret(&context, &inputs),
            program.interpret(vec![inputs[0].clone(), inputs[1].clone(), known.clone()]),
        );

        // A repeated actual identity carries the formal fact `n = 2` to the formal `p` that it also enters, so the
        // entry guard narrows both unknown inputs.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = builder.add_input(n.clone().into());
        let second = builder.add_input(p.into());
        builder.add_input(array(&n).into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![first, second], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(m.clone().into()),
                PartialValue::Unknown(m.into()),
                PartialValue::Known(known),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:dimension<m ∈ [1, 8)>, %1:dimension<m ∈ [1, 8)>, %2:f32[2] .
                let %3:dimension<2>, %4:dimension<2>, %5:f32[2] = guard [
                    boundary="partial evaluation",
                    types=[dimension<n ∈ [1, 8)>, dimension<p ∈ [1, 8)>, f32[n]],
                ] %0 %1 %2
                in (%3, %4)"#},
        );
    }

    #[test]
    fn test_program_partially_evaluate_narrows_identities_next_to_reference_identities() {
        // `[dimension<m>, ref<f32[n]>, f32[n, m], f32[m]]` with a known `dimension<2>` adds the vector to `zero[m]`.
        // The known dimension fixes `m = 2`, so the body's zero is `f32[2]`, and the vector must be narrowed to
        // `f32[2]` for the sum even though the matrix also shares `n` with the reference: only `n` stays symbolic.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m] = ["n", "m"].map(|name| DimensionType::new(name, bounds));
        let array = |dimensions: Vec<Dimension>| ArrayType::new(DataType::F32, Shape::new(dimensions));
        let vector_type = array(vec![m.simplified_dimension()]);
        let matrix_type = array(vec![n.simplified_dimension(), m.simplified_dimension()]);
        let reference_type: ArrayIrType = ReferenceType::new(array(vec![n.simplified_dimension()])).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let dimension = builder.add_input(m.clone().into());
        builder.add_input(reference_type.clone());
        builder.add_input(matrix_type.clone().into());
        let vector = builder.add_input(vector_type.clone().into());
        let zero = builder
            .add_instruction(
                TestOperation::Zero(ZeroOperation::new(vector_type.clone())),
                Vec::new(),
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![vector, zero], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let known = TestValue::Dimension(DimensionValue::constant(2).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(known.clone()),
                PartialValue::Unknown(reference_type),
                PartialValue::Unknown(matrix_type.into()),
                PartialValue::Unknown(vector_type.into()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:ref<f32[n]>, %1:f32[n, m], %2:f32[m], %3:dimension<2>, %4:f32[2] .
                let %5:dimension<2>, %6:f32[n, 2], %7:f32[2] = guard [
                    boundary="partial evaluation",
                    types=[dimension<m ∈ [1, 8)>, f32[n, m], f32[m]],
                ] %3 %1 %2
                    %8:f32[2] = add %7 %4
                in (%8)"#},
        );
        let reference = TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; 3]).unwrap()));
        let matrix = TestValue::Array(
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [3, 2]), &[0.0; 6]).unwrap(),
        );
        let context = EagerContext::<TestValue, TestOperation>::new();
        let vector = |extent| TestValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        assert_eq!(
            program.interpret(vec![known.clone(), reference.clone(), matrix.clone(), vector(2)]),
            Ok(vec![vector(2)]),
        );
        assert_eq!(
            evaluation.interpret(&context, &[reference.clone(), matrix.clone(), vector(2)]),
            Ok(vec![vector(2)]),
        );
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "m".to_string(),
            expected: 2,
            actual: 3,
        })));
        assert_eq!(program.interpret(vec![known, reference.clone(), matrix.clone(), vector(3)]), mismatch);
        assert_eq!(evaluation.interpret(&context, &[reference, matrix, vector(3)]), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_retains_reference_identities_while_narrowing_others() {
        // `[ref<f32[n]>, f32[n, m], f32[m, 1], f32[n], f32[m]]` with known exact witnesses that fix `n = 2` and `m =
        // 3`. The reference keeps `n`, so the entry guard retains it: the matrix becomes `f32[n, 3]` and the column
        // `f32[3, 1]`, and the body's product sees one type per identity.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m] = ["n", "m"].map(|name| DimensionType::new(name, bounds));
        let array = |dimensions: Vec<Dimension>| ArrayType::new(DataType::F32, Shape::new(dimensions));
        let matrix_type = array(vec![n.simplified_dimension(), m.simplified_dimension()]);
        let column_type = array(vec![m.simplified_dimension(), Dimension::Static(1)]);
        let reference_type: ArrayIrType = ReferenceType::new(array(vec![n.simplified_dimension()])).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(reference_type.clone());
        let matrix = builder.add_input(matrix_type.clone().into());
        let column = builder.add_input(column_type.clone().into());
        builder.add_input(array(vec![n.simplified_dimension()]).into());
        builder.add_input(array(vec![m.simplified_dimension()]).into());
        let product = builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(DotOperation::new(DotDimensionNumbers::new(
                    vec![1],
                    vec![0],
                    Vec::new(),
                    Vec::new(),
                )))),
                Vec::new(),
                vec![matrix, column],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![product], vec![Placeholder; 5], vec![Placeholder])
            .unwrap();
        let witness = |extent: usize| TestValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(reference_type),
                PartialValue::Unknown(matrix_type.into()),
                PartialValue::Unknown(column_type.into()),
                PartialValue::Known(witness(2)),
                PartialValue::Known(witness(3)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:ref<f32[n]>, %1:f32[n, m], %2:f32[m, 1], %3:f32[2], %4:f32[3] .
                let %5:f32[n, 3], %6:f32[3, 1], %7:f32[n], %8:f32[3] = guard [
                    boundary="partial evaluation",
                    types=[ref<f32[n]>, f32[n, m], f32[m, 1], f32[n], f32[m]],
                    retained_identities=[n],
                ] %0 %1 %2 %3 %4
                    %9:f32[n, 1] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %5 %6
                in (%9)"#},
        );
        let reference = TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap()));
        let matrix = |rows: usize| {
            TestValue::Array(
                Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [rows, 3]), &vec![1f32; rows * 3])
                    .unwrap(),
            )
        };
        let column = TestValue::Array(
            Array::from_elements::<f32>(ArrayType::new_static(DataType::F32, [3, 1]), &[1.0, 2.0, 3.0]).unwrap(),
        );
        let context = EagerContext::<TestValue, TestOperation>::new();
        let product = TestValue::Array(Array::matrix(2, 1, vec![6f32; 2]).unwrap());
        assert_eq!(
            program.interpret(vec![reference.clone(), matrix(2), column.clone(), witness(2), witness(3)]),
            Ok(vec![product.clone()]),
        );
        assert_eq!(evaluation.interpret(&context, &[reference.clone(), matrix(2), column.clone()]), Ok(vec![product]));
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 2,
            actual: 4,
        })));
        assert_eq!(
            program.interpret(vec![reference.clone(), matrix(4), column.clone(), witness(2), witness(3)]),
            mismatch,
        );
        assert_eq!(evaluation.interpret(&context, &[reference, matrix(4), column]), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_preserves_forwarded_definition_bounds() {
        // A proven fold forwards the unbounded input `value` as the output in place of the source's capped
        // `value + offset` result, keeping the input's own identity, while a shape payload that requires the capped
        // result is still rejected: forwarding must not silently widen a declared shape.
        let value = DimensionType::new("value", DimensionBounds::unbounded());
        let offset = DimensionType::new("offset", DimensionBounds::new(0, Some(2)).unwrap());
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(value.clone().into());
        let offset_input = builder.add_input(offset.clone().into());
        let result = builder
            .add_instruction(
                DimensionAddOperation::new(&value, &offset).unwrap(),
                Vec::new(),
                vec![input, offset_input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![result],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let partial_inputs = [
            PartialValue::Unknown(value.clone().into()),
            PartialValue::Unknown(DimensionValue::constant(0).unwrap().r#type().into_owned().into()),
        ];
        let evaluation = program.partially_evaluate(&partial_inputs).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:dimension<value ∈ [0, ∞)>, %1:dimension<0> .
                in (%0)"},
        );

        // A subsequent shape payload still has to admit the forwarded value's bounds. This application was already
        // rejected before shared identity bookkeeping, so forwarding must not silently widen its declared shape.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(value.clone().into());
        let offset_input = builder.add_input(offset.clone().into());
        let result = builder
            .add_instruction(
                DimensionAddOperation::new(&value, &offset).unwrap(),
                Vec::new(),
                vec![input, offset_input],
                None,
            )
            .unwrap()[0];
        let ArrayIrType::Dimension(result_type) = builder.atoms()[result.index()].r#type().into_owned() else {
            panic!("expected a dimension result");
        };
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![result_type.dimension().clone()]));
        let output =
            builder.add_instruction(ZeroOperation::new(vector_type), Vec::new(), vec![result], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            program.partially_evaluate(&partial_inputs).unwrap_err(),
            ProgramError::Type(TypeError::invalid(format!(
                "`zero` input 0 has type {value} but the output shape requires {result_type}",
            ))),
        );
    }

    #[test]
    fn test_program_partially_evaluate_preserves_unfixed_bounds_with_retained_references() {
        // One supplied dimension links two formal classes without fixing an extent. A reference keeps the wider
        // identity, while the narrower constructor and vector must agree on a compatible name within their bounds.
        let narrow = DimensionType::new("narrow", DimensionBounds::new(1, Some(4)).unwrap());
        let wide = DimensionType::new("wide", DimensionBounds::new(1, Some(8)).unwrap());
        let supplied = DimensionType::new("supplied", narrow.bounds());
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![narrow.dimension().clone()]));
        let reference_type: ArrayIrType =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![wide.dimension().clone()]))).into();
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        builder.add_input(reference_type.clone());
        let dimension = builder.add_input(narrow.into());
        builder.add_input(wide.into());
        let input = builder.add_input(vector.clone().into());
        let zero = builder
            .add_instruction(ZeroOperation::new(vector.clone()), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, zero], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 4],
                vec![Placeholder],
            )
            .unwrap();
        let reference =
            |extent| ArrayIrValue::Reference(ArrayReference::new(Array::vector(vec![1f32; extent]).unwrap()));
        let dimension = ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap());
        let input = ArrayIrValue::Array(Array::vector(vec![2f32; 3]).unwrap());
        let arguments = vec![reference(3), dimension.clone(), dimension.clone(), input.clone()];
        let expected = program.interpret(arguments.clone()).unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(reference_type),
                PartialValue::Unknown(supplied.clone().into()),
                PartialValue::Unknown(supplied.into()),
                PartialValue::Unknown(vector.into()),
            ])
            .unwrap();
        let supplied = "dimension<supplied ∈ [1, 4)>";
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {r#"
                lambda %0:ref<f32[wide]>, %1:{supplied}, %2:{supplied}, %3:f32[narrow] .
                let %4:dimension<narrow ∈ [1, 4)>, %5:dimension<wide ∈ [1, 8)>, %6:f32[narrow] = guard [
                    boundary="partial evaluation",
                    types=[ref<f32[wide]>, dimension<narrow ∈ [1, 4)>, dimension<wide ∈ [1, 8)>, f32[narrow]],
                    retained_identities=[wide],
                ] %0 %1 %2 %3
                    %7:f32[narrow] = zero [type=f32[narrow]] %4
                    %8:f32[narrow] = add %6 %7
                in (%8)"#},
        );
        assert_eq!(evaluation.interpret(&EagerContext::new(), &arguments), Ok(expected));
        let invalid = vec![reference(4), dimension.clone(), dimension, input];
        let direct_error = program.interpret(invalid.clone());
        assert_eq!(
            direct_error,
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "wide".to_string(),
                expected: 4,
                actual: 3,
            }))),
        );
        assert_eq!(evaluation.interpret(&EagerContext::new(), &invalid), direct_error);
    }

    #[test]
    fn test_program_partially_evaluate_represents_identity_classes_without_exact_extents() {
        // An unknown `dimension<q>` and an unknown `f32[n]` enter `[dimension<n>, f32[n]]`, and the body adds the
        // vector to `zero[n]` built from the dimension. Nothing fixes the class of `n`, so the entry guard represents
        // it by `n`, the identity with which the body's operations are declared: the dimension reaches the body as
        // `dimension<n>` behind the check that `q` equals the vector's extent, and the zero has the vector's type.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, q] = ["n", "q"].map(|name| DimensionType::new(name, bounds));
        let vector_type: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()])).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let dimension = builder.add_input(n.clone().into());
        let vector = builder.add_input(vector_type.clone());
        let ArrayIrType::Array(array_type) = &vector_type else { unreachable!() };
        let zero = builder
            .add_instruction(
                TestOperation::Zero(ZeroOperation::new(array_type.clone())),
                Vec::new(),
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![vector, zero], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(q.clone().into()), PartialValue::Unknown(vector_type)])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:dimension<q ∈ [1, 8)>, %1:f32[n] .
                let %2:dimension<n ∈ [1, 8)>, %3:f32[n] = guard [
                    boundary="partial evaluation",
                    types=[dimension<n ∈ [1, 8)>, f32[n]],
                    retained_identities=[n],
                ] %0 %1
                    %4:f32[n] = zero [type=f32[n]] %2
                    %5:f32[n] = add %3 %4
                in (%5)"#},
        );
        let context = EagerContext::<TestValue, TestOperation>::new();
        let dimension = |extent| TestValue::Dimension(DimensionValue::constant(extent).unwrap());
        let vector = |extent: usize| TestValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        assert_eq!(program.interpret(vec![dimension(2), vector(2)]), Ok(vec![vector(2)]));
        assert_eq!(evaluation.interpret(&context, &[dimension(2), vector(2)]), Ok(vec![vector(2)]));
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 3,
            actual: 2,
        })));
        assert_eq!(program.interpret(vec![dimension(3), vector(2)]), mismatch);
        assert_eq!(evaluation.interpret(&context, &[dimension(3), vector(2)]), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_consumes_known_values_at_reference_representations() {
        // The unknown reference fixes the representation that its read returns. The known exact array must consume
        // the guard's residual representation too, whatever its position in the source entry is.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let reference_type: ArrayIrType = ReferenceType::new(array_type.clone()).into();
        let known = TestValue::Array(Array::vector(vec![2f32; 3]).unwrap());
        for reference_first in [true, false] {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let (reference, array, seeds) = if reference_first {
                (
                    builder.add_input(reference_type.clone()),
                    builder.add_input(array_type.clone().into()),
                    vec![PartialValue::Unknown(reference_type.clone()), PartialValue::Known(known.clone())],
                )
            } else {
                let array = builder.add_input(array_type.clone().into());
                let reference = builder.add_input(reference_type.clone());
                (
                    reference,
                    array,
                    vec![PartialValue::Known(known.clone()), PartialValue::Unknown(reference_type.clone())],
                )
            };
            let read =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![read, array], None).unwrap()[0];
            let program = builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let evaluation = program.partially_evaluate(&seeds).unwrap();
            let guard = if reference_first {
                indoc! {r#"
                    guard [
                        boundary="partial evaluation",
                        types=[ref<f32[n]>, f32[n]],
                        retained_identities=[n],
                    ] %0 %1"#}
                .trim_end()
            } else {
                indoc! {r#"
                    guard [
                        boundary="partial evaluation",
                        types=[f32[n], ref<f32[n]>],
                        retained_identities=[n],
                    ] %1 %0"#}
                .trim_end()
            };
            assert_eq!(
                evaluation.program().to_string(),
                formatdoc! {"
                lambda %0:ref<f32[n]>, %1:f32[3] .
                let %2:f32[n] = {guard}
                    %3:f32[n] = reference_read %0
                    %4:f32[n] = add %3 %2
                in (%4)"},
            );
            assert!(matches!(evaluation.outputs(), [PartialEvaluationOutput::Unknown(0)]));
            let context = EagerContext::<TestValue, TestOperation>::new();
            let matching = TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; 3]).unwrap()));
            assert_eq!(
                evaluation.interpret(&context, &[matching]),
                Ok(vec![TestValue::Array(Array::vector(vec![3f32; 3]).unwrap())]),
            );
            let mismatching = TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap()));
            let source_inputs = if reference_first {
                vec![mismatching.clone(), known.clone()]
            } else {
                vec![known.clone(), mismatching.clone()]
            };
            let (expected_extent, actual_extent) = if reference_first { (2, 3) } else { (3, 2) };
            let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: expected_extent,
                actual: actual_extent,
            })));
            assert_eq!(program.interpret(source_inputs), mismatch);
            assert_eq!(evaluation.interpret(&context, &[mismatching]), mismatch);
        }
    }

    #[test]
    fn test_program_partially_evaluate_refines_disconnected_singleton_inputs() {
        // The first component needs a runtime check. That must not prevent the unrelated second component from applying
        // its intrinsic singleton facts, so that its constructor builds the static `f32[3]`.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m, a] = ["n", "m", "a"].map(|name| DimensionType::new(name, bounds));
        let exact = DimensionType::new("exact", DimensionBounds::new(3, Some(4)).unwrap());
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(n.clone().into());
        builder.add_input(vector(&n).into());
        let dimension = builder.add_input(m.clone().into());
        let input = builder.add_input(vector(&m).into());
        let zero =
            builder.add_instruction(ZeroOperation::new(vector(&m)), Vec::new(), vec![dimension], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, zero], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(a.into()),
                PartialValue::Unknown(vector(&n).into()),
                PartialValue::Unknown(exact.clone().into()),
                PartialValue::Unknown(vector(&exact).into()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:dimension<a ∈ [1, 8)>, %1:f32[n], %2:dimension<3>, %3:f32[3] .
                let %4:dimension<n ∈ [1, 8)>, %5:f32[n], %6:dimension<3> = guard [
                    boundary="partial evaluation",
                    types=[dimension<n ∈ [1, 8)>, f32[n], dimension<m ∈ [1, 8)>],
                    retained_identities=[n],
                ] %0 %1 %2
                    %7:f32[3] = zero [type=f32[m]] %6
                    %8:f32[3] = add %3 %7
                in (%8)"#},
        );
        let dimension = |extent| TestValue::Dimension(DimensionValue::constant(extent).unwrap());
        let vector = |extent| TestValue::Array(Array::vector(vec![2f32; extent]).unwrap());
        let arguments = vec![dimension(2), vector(2), dimension(3), vector(3)];
        let expected = program.interpret(arguments.clone()).unwrap();
        assert_eq!(expected, vec![vector(3)]);
        let context = EagerContext::<TestValue, TestOperation>::new();
        assert_eq!(evaluation.interpret(&context, &arguments), Ok(expected));
        let mismatching = vec![dimension(3), vector(2), dimension(3), vector(3)];
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 3,
            actual: 2,
        })));
        assert_eq!(program.interpret(mismatching.clone()), mismatch);
        assert_eq!(evaluation.interpret(&context, &mismatching), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_prepares_mixed_reference_reads() {
        // Two reads, from a symbolic `ref<f32[n]>` and from a reference whose supplied type is the exact `ref<f32[3]>`,
        // are added in either order. The entry guard checks the references against each other without outputs, so each
        // read keeps its own representation until the guard of the addition relates them.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let reference: ArrayIrType = ReferenceType::new(vector).into();
        let exact: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into();
        for reversed_reads in [false, true] {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let first = builder.add_input(reference.clone());
            let second = builder.add_input(reference.clone());
            let references = if reversed_reads { [second, first] } else { [first, second] };
            let reads = references
                .into_iter()
                .map(|reference| {
                    builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()
                        [0]
                })
                .collect();
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), reads, None).unwrap()[0];
            let program = builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let evaluation = program
                .partially_evaluate(&[PartialValue::Unknown(reference.clone()), PartialValue::Unknown(exact.clone())])
                .unwrap();
            let (first_type, first_input, second_type, second_input) =
                if reversed_reads { ("f32[3]", 1, "f32[n]", 0) } else { ("f32[n]", 0, "f32[3]", 1) };
            assert_eq!(
                evaluation.program().to_string(),
                formatdoc! {r#"
                lambda %0:ref<f32[n]>, %1:ref<f32[3]> .
                let () = guard [
                    boundary="partial evaluation",
                    types=[ref<f32[n]>, ref<f32[n]>],
                    retained_identities=[n],
                ] %0 %1
                    %2:{first_type} = reference_read %{first_input}
                    %3:{second_type} = reference_read %{second_input}
                    %4:f32[n], %5:f32[n] = guard [boundary="add", types=[f32[n], f32[n]], retained_identities=[n]] %2 %3
                    %6:f32[n] = add %4 %5
                in (%6)"#},
            );
            let eager = EagerContext::<TestValue, TestOperation>::new();
            let reference =
                |extent| TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; extent]).unwrap()));
            let matching = vec![reference(3), reference(3)];
            assert_eq!(evaluation.interpret(&eager, &matching), program.interpret(matching));
            let mismatching = vec![reference(4), reference(3)];
            let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 4,
                actual: 3,
            })));
            assert_eq!(program.interpret(mismatching.clone()), mismatch);
            assert_eq!(evaluation.interpret(&eager, &mismatching), mismatch);
        }
    }

    #[test]
    fn test_program_partially_evaluate_prepares_reference_reads_around_dimension_forwarding() {
        // Two reads, from a symbolic `ref<f32[n]>` and from a reference whose supplied type is the exact `ref<f32[3]>`,
        // are added, and a zero-trip scan forwards a `dimension<n>` input before or after them. The scan forwards its
        // definition in either source order, the live addition operands still require their checked common
        // representation, and both orders produce the same residual program.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let reference: ArrayIrType = ReferenceType::new(vector).into();
        let exact: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into();
        let mut body = ProgramBuilder::<TestValue, TestOperation>::new();
        body.add_input(ArrayType::scalar(DataType::I64).into());
        let carry = body.add_input(n.clone().into());
        let body = body
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![carry], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        for scan_first in [false, true] {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let first = builder.add_input(reference.clone());
            let second = builder.add_input(reference.clone());
            let dimension = builder.add_input(n.clone().into());
            let body = builder.import_program(body.clone());
            let earlier_forwarded = if scan_first {
                Some(
                    builder
                        .add_instruction(ScanOperation::<ArrayIrType>::new(1, 0), vec![body], vec![dimension], None)
                        .unwrap()[0],
                )
            } else {
                None
            };
            let first =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
            let second =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None).unwrap()[0];
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, second], None).unwrap()[0];
            let forwarded = earlier_forwarded.unwrap_or_else(|| {
                builder
                    .add_instruction(ScanOperation::<ArrayIrType>::new(1, 0), vec![body], vec![dimension], None)
                    .unwrap()[0]
            });
            let program = builder
                .build::<Vec<TestValue>, Vec<TestValue>>(
                    vec![sum, forwarded],
                    vec![Placeholder; 3],
                    vec![Placeholder; 2],
                )
                .unwrap();
            let evaluation = program
                .partially_evaluate(&[
                    PartialValue::Unknown(reference.clone()),
                    PartialValue::Unknown(exact.clone()),
                    PartialValue::Unknown(n.clone().into()),
                ])
                .unwrap();
            assert_eq!(
                evaluation.program().to_string(),
                indoc! {r#"
            lambda %0:ref<f32[n]>, %1:ref<f32[3]>, %2:dimension<n ∈ [1, 8)> .
            let %3:dimension<n ∈ [1, 8)> = guard [
                boundary="partial evaluation",
                types=[ref<f32[n]>, ref<f32[n]>, dimension<n ∈ [1, 8)>],
                retained_identities=[n],
            ] %0 %1 %2
                %4:f32[n] = reference_read %0
                %5:f32[3] = reference_read %1
                %6:f32[n], %7:f32[n] = guard [boundary="add", types=[f32[n], f32[n]], retained_identities=[n]] %4 %5
                %8:f32[n] = add %6 %7
            in (%8, %3)"#},
            );
            let eager = EagerContext::<TestValue, TestOperation>::new();
            let reference =
                |extent| TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; extent]).unwrap()));
            let dimension = |extent| TestValue::Dimension(DimensionValue::constant(extent).unwrap());
            let mismatch = |expected, actual| -> Result<Vec<TestValue>, ProgramError> {
                Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                    dimension: "n".to_string(),
                    expected,
                    actual,
                })))
            };
            let sum = TestValue::Array(Array::vector(vec![2f32; 3]).unwrap());
            for (extent, size, expected) in
                [(3, 3, Ok(vec![sum, dimension(3)])), (4, 4, mismatch(4, 3)), (3, 2, mismatch(3, 2))]
            {
                let inputs = vec![reference(extent), reference(3), dimension(size)];
                assert_eq!(program.interpret(inputs.clone()), expected);
                assert_eq!(evaluation.interpret(&eager, &inputs), expected);
            }
        }
    }

    #[test]
    fn test_program_partially_evaluate_prepares_reference_reads_with_reified_constructors() {
        // Two reads, from a symbolic `ref<f32[n]>` and from a reference whose supplied type is the exact `ref<f32[3]>`,
        // feed a `dimension_size` whose result builds a zero that is added to the second read. A size reified from the
        // exact read is `dimension<3>`, so the addition needs no guard, while a size reified from the symbolic read
        // keeps `n`, so the guard of the addition relates the two representations.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let reference: ArrayIrType = ReferenceType::new(vector.clone()).into();
        let exact: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into();
        for exact_size in [false, true] {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let first = builder.add_input(reference.clone());
            let second = builder.add_input(reference.clone());
            let first =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
            let second =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None).unwrap()[0];
            let size_input = if exact_size { second } else { first };
            let size = builder
                .add_instruction(DimensionSizeOperation::new(&vector, 0).unwrap(), Vec::new(), vec![size_input], None)
                .unwrap()[0];
            let zero = builder
                .add_instruction(TestOperation::Zero(ZeroOperation::new(vector.clone())), Vec::new(), vec![size], None)
                .unwrap()[0];
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![second, zero], None).unwrap()[0];
            let program = builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
                .unwrap();
            let evaluation = program
                .partially_evaluate(&[PartialValue::Unknown(reference.clone()), PartialValue::Unknown(exact.clone())])
                .unwrap();
            let expected = if exact_size {
                indoc! {r#"
                lambda %0:ref<f32[n]>, %1:ref<f32[3]> .
                let () = guard [
                    boundary="partial evaluation",
                    types=[ref<f32[n]>, ref<f32[n]>],
                    retained_identities=[n],
                ] %0 %1
                    %2:f32[n] = reference_read %0
                    %3:f32[3] = reference_read %1
                    %4:dimension<3> = dimension_size [axis=0] %3
                    %5:f32[3] = zero [type=f32[n]] %4
                    %6:f32[3] = add %3 %5
                in (%6)"#}
            } else {
                indoc! {r#"
                lambda %0:ref<f32[n]>, %1:ref<f32[3]> .
                let () = guard [
                    boundary="partial evaluation",
                    types=[ref<f32[n]>, ref<f32[n]>],
                    retained_identities=[n],
                ] %0 %1
                    %2:f32[n] = reference_read %0
                    %3:f32[3] = reference_read %1
                    %4:dimension<n ∈ [1, 8)> = dimension_size [axis=0] %2
                    %5:f32[n] = zero [type=f32[n]] %4
                    %6:f32[n], %7:f32[n] = guard [boundary="add", types=[f32[n], f32[n]], retained_identities=[n]] %3 %5
                    %8:f32[n] = add %6 %7
                in (%8)"#}
            };
            assert_eq!(evaluation.program().to_string(), expected);
            let eager = EagerContext::<TestValue, TestOperation>::new();
            let reference =
                |extent| TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; extent]).unwrap()));
            let matching = vec![reference(3), reference(3)];
            assert_eq!(evaluation.interpret(&eager, &matching), program.interpret(matching));
            let mismatching = vec![reference(4), reference(3)];
            let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 4,
                actual: 3,
            })));
            assert_eq!(program.interpret(mismatching.clone()), mismatch);
            assert_eq!(evaluation.interpret(&eager, &mismatching), mismatch);
        }
    }

    #[test]
    fn test_program_partially_evaluate_prepares_mixed_reference_reads_without_retype() {
        // Existing projected array operations plus a read and an entry guard suffice; no Cast capability is required.
        /// Minimal native array family used by the read fixture's computational projection.
        #[derive(Clone, Debug, ryft_macros::Operation)]
        #[ryft(type(ArrayType), constant(Array))]
        enum ArrayAdditionOperation {
            Constant(ConstantOperation<Array>),
            Add(AddOperation<ArrayType>),
        }

        /// Minimal projected family that supports entry checks without proof casts.
        #[derive(Clone, Debug, ryft_macros::Operation)]
        #[ryft(type(ArrayIrType), constant(TestValue), members(ArrayType))]
        enum ReadOperation {
            Constant(ConstantOperation<TestValue>),
            #[ryft(projected(ArrayType))]
            Array(ArrayAdditionOperation),
            ReferenceRead(ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>),
            Guard(GuardOperation<ArrayIrType>),
        }
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let reference: ArrayIrType =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]))).into();
        let exact: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into();
        let mut builder = ProgramBuilder::<TestValue, ReadOperation>::new();
        let first = builder.add_input(reference.clone());
        let second = builder.add_input(reference.clone());
        let first = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        let second = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None).unwrap()[0];
        let sum = builder
            .add_instruction(
                ReadOperation::Array(ArrayAdditionOperation::Add(AddOperation::new())),
                Vec::new(),
                vec![first, second],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(reference), PartialValue::Unknown(exact)])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
            lambda %0:ref<f32[n]>, %1:ref<f32[3]> .
            let () = guard [
                boundary="partial evaluation",
                types=[ref<f32[n]>, ref<f32[n]>],
                retained_identities=[n],
            ] %0 %1
                %2:f32[n] = reference_read %0
                %3:f32[3] = reference_read %1
                %4:f32[n], %5:f32[n] = guard [boundary="add", types=[f32[n], f32[n]], retained_identities=[n]] %2 %3
                %6:f32[n] = add %4 %5
            in (%6)"#},
        );
        let reference = || TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; 3]).unwrap()));
        assert_eq!(
            evaluation.interpret(&EagerContext::<TestValue, ReadOperation>::new(), &[reference(), reference()]),
            Ok(vec![TestValue::Array(Array::vector(vec![2f32; 3]).unwrap())]),
        );
    }

    #[test]
    fn test_program_partially_evaluate_replays_lone_renamed_dimensions() {
        // Both eager and staged parents replay the payload under `q`; residual execution must not retain the source
        // `n`.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, q] = ["n", "q"].map(|name| DimensionType::new(name, bounds));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let dimension = builder.add_input(n.clone().into());
        let zero = builder
            .add_instruction(
                TestOperation::Zero(ZeroOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![n.simplified_dimension()]),
                ))),
                Vec::new(),
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![zero], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let expected = indoc! {"
            lambda %0:dimension<q ∈ [1, 8)> .
            let %1:f32[q] = zero [type=f32[q]] %0
            in (%1)"};
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(q.clone().into())]).unwrap();
        assert_eq!(evaluation.program().to_string(), expected);
        let dimension = TestValue::Dimension(DimensionValue::constant(3).unwrap());
        let output = TestValue::Array(Array::vector(vec![0f32; 3]).unwrap());
        let context = EagerContext::<TestValue, TestOperation>::new();
        assert_eq!(evaluation.interpret(&context, &[dimension.clone()]), Ok(vec![output.clone()]));
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let staged = program.partially_evaluate_in_context(&outer, &[PartialValue::Unknown(q.clone().into())]).unwrap();
        assert_eq!(staged.program().to_string(), expected);
        let inputs = [outer.input(q.into())];
        let outputs = staged.interpret(&outer, &inputs).unwrap();
        let atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(atoms, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(replayed.to_string(), expected);
        assert_eq!(replayed.interpret(vec![dimension]), Ok(vec![output]));
    }

    #[test]
    fn test_program_partially_evaluate_preserves_multiple_reference_representations_in_one_class() {
        // Unknown dimension peers connect `a` and `b`, but neither reference handle can change its referent type.
        // Reading each handle and building its zero must therefore keep that handle's own representation.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [a, b, c] = ["a", "b", "c"].map(|name| DimensionType::new(name, bounds));
        let first_array = ArrayType::new(DataType::F32, Shape::new(vec![a.simplified_dimension()]));
        let second_array = ArrayType::new(DataType::F32, Shape::new(vec![b.simplified_dimension()]));
        let first_reference: ArrayIrType = ReferenceType::new(first_array.clone()).into();
        let second_reference: ArrayIrType = ReferenceType::new(second_array.clone()).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = builder.add_input(first_reference.clone());
        let second = builder.add_input(second_reference.clone());
        let first_extent = builder.add_input(a.into());
        let second_extent = builder.add_input(b.into());
        let first = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        let first_zero = builder
            .add_instruction(TestOperation::Zero(ZeroOperation::new(first_array)), Vec::new(), vec![first_extent], None)
            .unwrap()[0];
        let first = builder.add_instruction(AddOperation::new(), Vec::new(), vec![first, first_zero], None).unwrap()[0];
        let second = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None).unwrap()[0];
        let second_zero = builder
            .add_instruction(
                TestOperation::Zero(ZeroOperation::new(second_array)),
                Vec::new(),
                vec![second_extent],
                None,
            )
            .unwrap()[0];
        let second =
            builder.add_instruction(AddOperation::new(), Vec::new(), vec![second, second_zero], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![first, second], vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(first_reference),
                PartialValue::Unknown(second_reference),
                PartialValue::Unknown(c.clone().into()),
                PartialValue::Unknown(c.into()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
            lambda %0:ref<f32[a]>, %1:ref<f32[b]>, %2:dimension<c ∈ [1, 8)>, %3:dimension<c ∈ [1, 8)> .
            let %4:dimension<a ∈ [1, 8)>, %5:dimension<b ∈ [1, 8)> = guard [
                boundary="partial evaluation",
                types=[ref<f32[a]>, ref<f32[b]>, dimension<a ∈ [1, 8)>, dimension<b ∈ [1, 8)>],
                retained_identities=[a, b],
            ] %0 %1 %2 %3
                %6:f32[a] = reference_read %0
                %7:f32[b] = reference_read %1
                %8:f32[a] = zero [type=f32[a]] %4
                %9:f32[a] = add %6 %8
                %10:f32[b] = zero [type=f32[b]] %5
                %11:f32[b] = add %7 %10
            in (%9, %11)"#},
        );
        let context = EagerContext::<TestValue, TestOperation>::new();
        let first = TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; 2]).unwrap()));
        let second = TestValue::Reference(ArrayReference::new(Array::vector(vec![2f32; 2]).unwrap()));
        let dimension = TestValue::Dimension(DimensionValue::constant(2).unwrap());
        let matching = vec![first.clone(), second, dimension.clone(), dimension.clone()];
        assert_eq!(evaluation.interpret(&context, &matching), program.interpret(matching));
        let second = TestValue::Reference(ArrayReference::new(Array::vector(vec![2f32; 3]).unwrap()));
        let mismatching = vec![first, second, dimension.clone(), dimension];
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "b".to_string(),
            expected: 3,
            actual: 2,
        })));
        assert_eq!(program.interpret(mismatching.clone()), mismatch);
        assert_eq!(evaluation.interpret(&context, &mismatching), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_replays_renamed_dimensions_in_attached_regions() {
        // Both condition slots share the same source region. The selected body must see `q` in its zero payload too.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, q] = ["n", "q"].map(|name| DimensionType::new(name, bounds));
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let dimension = branch.add_input(n.clone().into());
        let zero = branch
            .add_instruction(
                TestOperation::Zero(ZeroOperation::new(ArrayType::new(
                    DataType::F32,
                    Shape::new(vec![n.simplified_dimension()]),
                ))),
                Vec::new(),
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let branch = branch
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![zero], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let region = builder.import_program(branch);
        let dimension = builder.add_input(n.into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let zero = builder
            .add_instruction(ConditionOperation::new(), vec![region, region], vec![predicate, dimension], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![zero], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(q.into()),
                PartialValue::Known(TestValue::Array(Array::scalar(true).unwrap())),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
            lambda %0:dimension<q ∈ [1, 8)> .
            let %1:f32[q] = zero [type=f32[q]] %0
            in (%1)"},
        );
        let dimension = TestValue::Dimension(DimensionValue::constant(3).unwrap());
        let context = EagerContext::<TestValue, TestOperation>::new();
        assert_eq!(
            evaluation.interpret(&context, &[dimension]),
            Ok(vec![TestValue::Array(Array::vector(vec![0f32; 3]).unwrap())]),
        );
    }

    #[test]
    fn test_program_partially_evaluate_orders_reference_guards_before_known_effects() {
        // A known exact `f32[2]` fixes `n`, which an unknown `ref<f32[n]>` must agree with. The entry guard checks the
        // reference before anything else, so its ordered effect also defers the known print behind it instead of
        // letting it run during partial evaluation, before a mismatched reference would be rejected.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]));
        let reference_type: ArrayIrType = ReferenceType::new(array_type.clone()).into();
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(reference_type.clone());
        builder.add_input(array_type.into());
        let scalar = builder.add_input(scalar_type.clone());
        let printed = builder
            .add_instruction(
                TestOperation::Array(ArrayOperation::from(PrintOperation::<ArrayType>::new("known"))),
                Vec::new(),
                vec![scalar],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![printed], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(reference_type.clone()),
                PartialValue::Known(TestValue::Array(Array::vector(vec![1f32; 2]).unwrap())),
                PartialValue::Known(TestValue::Array(Array::scalar(4f32).unwrap())),
            ])
            .unwrap();
        let residual = indoc! {r#"
            lambda %0:ref<f32[n]>, %1:f32[2], %2:f32[] .
            let %3:f32[n] = guard [
                boundary="partial evaluation",
                types=[ref<f32[n]>, f32[n]],
                retained_identities=[n],
            ] %0 %1
                %4:f32[] = print [label=known] %2
            in (%4)"#};
        assert_eq!(evaluation.program().to_string(), residual);
        let context = EagerContext::<TestValue, TestOperation>::new();
        let reference =
            |extent: usize| TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; extent]).unwrap()));
        let scalar = TestValue::Array(Array::scalar(4f32).unwrap());
        assert_eq!(evaluation.interpret(&context, &[reference(2)]), Ok(vec![scalar.clone()]));
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 3,
            actual: 2,
        })));
        assert_eq!(
            program.interpret(vec![reference(3), TestValue::Array(Array::vector(vec![1f32; 2]).unwrap()), scalar]),
            mismatch,
        );
        assert_eq!(evaluation.interpret(&context, &[reference(3)]), mismatch);

        // A staging known side produces the same residual program: the guard defers the known print, so nothing of the
        // body folds into the enclosing trace, which only receives the lifted known array.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let scalar = outer.input(scalar_type);
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Unknown(reference_type),
                    PartialValue::Known(outer.constant(TestValue::Array(Array::vector(vec![1f32; 2]).unwrap()))),
                    PartialValue::Known(scalar),
                ],
            )
            .unwrap();
        assert_eq!(evaluation.program().to_string(), residual);
        let parent = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            parent.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[2] = const [1.0, 1.0]
                in ()"},
        );
    }

    #[test]
    fn test_program_partially_evaluate_preserves_symbolic_entry_witnesses() {
        // A known symbolic `f32[extent]` from the enclosing trace next to an unknown `f32[extent]` proves their
        // relationship through the shared identity, so the residual program needs no instruction. The known value is
        // still a residual input because it witnesses the extent of the unknown input.
        let dimension = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![dimension.dimension().clone()]));
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        builder.add_input(array_type.clone());
        let unknown = builder.add_input(array_type.clone());
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![unknown], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let outer = TestArrayTracingContext::new();
        let known = outer.input(array_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known.clone()), PartialValue::Unknown(array_type)],
            )
            .unwrap();
        assert_eq!(evaluation.inputs(), &[PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(known)]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[extent], %1:f32[extent] .
                in (%0)"},
        );
        let known = Array::vector(vec![3f32; 3]).unwrap();
        let matching = Array::vector(vec![4f32; 3]).unwrap();
        assert_eq!(evaluation.program().interpret(vec![matching.clone(), known.clone()]), Ok(vec![matching]));
        let mismatching = Array::vector(vec![4f32; 4]).unwrap();
        assert_eq!(
            program.interpret(vec![known.clone(), mismatching.clone()]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "extent".to_string(),
                expected: 3,
                actual: 4,
            }))),
        );
        assert_eq!(
            evaluation.program().interpret(vec![mismatching, known]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "extent".to_string(),
                expected: 4,
                actual: 3,
            }))),
        );

        // A known symbolic `ref<f32[n]>` witnesses `n` for an unknown `f32[n]` in the same way, as a known reference
        // input that the residual program receives without consuming it.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let reference_type: ArrayIrType = ReferenceType::new(array_type.clone()).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(reference_type.clone());
        let array = builder.add_input(array_type.clone().into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![array], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(outer.input(reference_type)), PartialValue::Unknown(array_type.into())],
            )
            .unwrap();
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[n], %1:ref<f32[n]> .
                in (%0)"},
        );
    }

    #[test]
    fn test_program_partially_evaluate_rejects_unguardable_entries() {
        // An operation family without an entry guard cannot preserve an unproven entry relationship, so partially
        // evaluating a program with a mixed signature is rejected instead of dropping its entry check.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]));
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let first = builder.add_input(vector_type.clone());
        builder.add_input(vector_type.clone());
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![first], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program
                .partially_evaluate(&[
                    PartialValue::Known(Array::vector(vec![1f32; 2]).unwrap()),
                    PartialValue::Unknown(vector_type),
                ])
                .map(|_| ()),
            Err(ProgramError::UnsupportedOperation {
                message: "the input types of `partial evaluation` leave identity relationships unproven, but the \
                          operation family cannot represent the guard required to preserve these input checks"
                    .to_string(),
            }),
        );
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

    #[test]
    fn test_program_partially_evaluate_in_context_guards_unproven_entry_relationships() {
        // A known symbolic `dimension<m>` from the enclosing trace enters `dimension<n>` next to an unknown `f32[2]`
        // entering `f32[n]`. No residual type can express `m = 2`, so partial evaluation stages an entry guard, which
        // keeps the dimension as a residual witness, and the replayed program rejects `m = 5` like the source. The
        // known dimension keeps `m` in the body, so the guard retains it and gives the array the type `f32[m]`.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(n.clone().into());
        let array = builder.add_input(array_type.into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![array], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, vec![2]).into();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(outer.input(m.into())), PartialValue::Unknown(exact.clone())],
            )
            .unwrap();
        let guard = indoc! {r#"
            guard [
                boundary="partial evaluation",
                types=[dimension<n ∈ [1, 8)>, f32[n]],
                retained_identities=[m],
            ]"#}
        .trim();
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:f32[2], %1:dimension<m ∈ [1, 8)> .
                let %2:dimension<m ∈ [1, 8)>, %3:f32[m] = {guard} %1 %0
                in (%3)"},
        );
        let outputs = evaluation.interpret(&outer, &[outer.input(exact)]).unwrap();
        let output_atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(output_atoms, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            replayed.to_string(),
            formatdoc! {"
                lambda %0:dimension<m ∈ [1, 8)>, %1:f32[2] .
                let %2:dimension<m ∈ [1, 8)>, %3:f32[m] = {guard} %0 %1
                in (%3)"},
        );
        let dimension = |extent| TestValue::Dimension(DimensionValue::constant(extent).unwrap());
        let array = TestValue::Array(Array::vector(vec![1f32; 2]).unwrap());
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 5,
            actual: 2,
        })));
        assert_eq!(program.interpret(vec![dimension(5), array.clone()]), mismatch);
        assert_eq!(replayed.interpret(vec![dimension(2), array.clone()]), Ok(vec![array.clone()]));
        assert_eq!(replayed.interpret(vec![dimension(5), array]), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_in_context_guards_known_symbolic_relationships() {
        // A known `dimension<m>`, an unknown `dimension<m>`, and a known exact `f32[2]` enter `[n, n, f32[n]]`. The
        // known values alone prove that the known dimension is `dimension<2>`, so a guard that folds into the enclosing
        // trace checks and narrows it first. The residual entry guard then relates the unknown dimension to the
        // narrowed known one and to the exact array, and narrows it to `dimension<2>` as well, so the body sees one
        // type for `m`.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(n.clone().into());
        let dimension = builder.add_input(n.clone().into());
        builder.add_input(array_type.into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![dimension], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let known_dimension = outer.input(m.clone().into());
        let known_array = outer.lift(TestValue::Array(Array::vector(vec![1f32; 2]).unwrap())).unwrap();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(known_dimension.clone()),
                    PartialValue::Unknown(m.clone().into()),
                    PartialValue::Known(known_array),
                ],
            )
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:dimension<m ∈ [1, 8)>, %1:dimension<2>, %2:f32[2] .
                let %3:dimension<2>, %4:dimension<2>, %5:f32[2] = guard [
                    boundary="partial evaluation",
                    types=[dimension<n ∈ [1, 8)>, dimension<n ∈ [1, 8)>, f32[n]],
                ] %1 %0 %2
                in (%4)"#},
        );
        let dimension = |extent| TestValue::Dimension(DimensionValue::constant(extent).unwrap());
        let array = TestValue::Array(Array::vector(vec![1f32; 2]).unwrap());
        assert_eq!(
            evaluation.program().interpret(vec![dimension(2), dimension(2), array.clone()]),
            Ok(vec![dimension(2)]),
        );
        // The residual guard relates the unknown dimension to the narrowed known one and to the exact array, while
        // the residual entry itself only accepts the narrowed known dimension that the known side checked.
        assert_eq!(
            evaluation.program().interpret(vec![dimension(5), dimension(2), array.clone()]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 2,
                actual: 5,
            }))),
        );
        assert_eq!(
            program.interpret(vec![dimension(2), dimension(5), array.clone()]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 2,
                actual: 5,
            }))),
        );
        assert_eq!(
            evaluation.program().interpret(vec![dimension(2), dimension(5), array.clone()]),
            Err(ProgramError::Type(TypeError::invalid(
                "encountered input type dimension<5> which is incompatible with the region's declared type \
                 dimension<2>",
            ))),
        );

        // Replayed with a value of the supplied symbolic type (here, the known dimension itself), the guard checks the
        // relationship at runtime exactly like the source program.
        let outputs = evaluation.interpret(&outer, &[known_dimension]).unwrap();
        let output_atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(output_atoms, vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(replayed.interpret(vec![dimension(2)]), Ok(vec![dimension(2)]));
        assert_eq!(replayed.interpret(vec![dimension(5)]), program.interpret(vec![dimension(5), dimension(5), array]));
        assert_eq!(
            replayed.interpret(vec![dimension(5)]),
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "n".to_string(),
                expected: 5,
                actual: 2,
            }))),
        );
    }

    #[test]
    fn test_program_partially_evaluate_in_context_orders_guards_before_effects() {
        // The body prints a known scalar and an unknown scalar. The entry guard is staged first, and its ordered effect
        // also defers the known print, so both prints follow it in the residual program.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]));
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let print = |label: &str| TestOperation::Array(ArrayOperation::from(PrintOperation::<ArrayType>::new(label)));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(n.clone().into());
        let array = builder.add_input(array_type.into());
        let known = builder.add_input(scalar_type.clone());
        let unknown = builder.add_input(scalar_type.clone());
        let known = builder.add_instruction(print("known"), Vec::new(), vec![known], None).unwrap()[0];
        let unknown = builder.add_instruction(print("unknown"), Vec::new(), vec![unknown], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![array, known, unknown],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();
        let m = DimensionType::new("m", DimensionBounds::positive(Some(8)).unwrap());
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, vec![2]).into();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let known_scalar = outer.input(scalar_type.clone());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(m.into())),
                    PartialValue::Unknown(exact.clone()),
                    PartialValue::Known(known_scalar),
                    PartialValue::Unknown(scalar_type.clone()),
                ],
            )
            .unwrap();
        let guard = indoc! {r#"
            guard [
                boundary="partial evaluation",
                types=[dimension<n ∈ [1, 8)>, f32[n]],
                retained_identities=[m],
            ]"#}
        .trim();
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {"
                lambda %0:f32[2], %1:f32[], %2:dimension<m ∈ [1, 8)>, %3:f32[] .
                let %4:dimension<m ∈ [1, 8)>, %5:f32[m] = {guard} %2 %0
                    %6:f32[] = print [label=known] %3
                    %7:f32[] = print [label=unknown] %1
                in (%5, %6, %7)"},
        );
        let outputs = evaluation.interpret(&outer, &[outer.input(exact), outer.input(scalar_type)]).unwrap();
        let output_atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(output_atoms, vec![Placeholder; 4], vec![Placeholder; 3])
            .unwrap();
        assert_eq!(
            replayed.to_string(),
            formatdoc! {"
                lambda %0:f32[], %1:dimension<m ∈ [1, 8)>, %2:f32[2], %3:f32[] .
                let %4:dimension<m ∈ [1, 8)>, %5:f32[m] = {guard} %1 %2
                    %6:f32[] = print [label=known] %0
                    %7:f32[] = print [label=unknown] %3
                in (%5, %6, %7)"},
        );
    }

    #[test]
    fn test_program_partially_evaluate_in_context_guards_reference_relationships() {
        // A known symbolic `ref<f32[n]>` and an unknown exact `f32[2]` enter `[ref<f32[n]>, f32[n]]`. The reference's
        // allocation fixes its extent but not its equality with the array, so the entry guard checks it, and the
        // replayed program rejects a reference of extent 3 exactly like the source.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()]));
        let reference_type: ArrayIrType = ReferenceType::new(array_type.clone()).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(reference_type.clone());
        let array = builder.add_input(array_type.into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![array], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let exact: ArrayIrType = ArrayType::new_static(DataType::F32, [2]).into();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(outer.input(reference_type)), PartialValue::Unknown(exact.clone())],
            )
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:f32[2], %1:ref<f32[n]> .
                let %2:f32[n] = guard [
                    boundary="partial evaluation",
                    types=[ref<f32[n]>, f32[n]],
                    retained_identities=[n],
                ] %1 %0
                in (%2)"#},
        );
        let outputs = evaluation.interpret(&outer, &[outer.input(exact)]).unwrap();
        let output_atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(output_atoms, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let reference =
            |extent: usize| TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; extent]).unwrap()));
        let array = TestValue::Array(Array::vector(vec![1f32; 2]).unwrap());
        assert_eq!(replayed.interpret(vec![reference(2), array.clone()]), Ok(vec![array.clone()]));
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 3,
            actual: 2,
        })));
        assert_eq!(program.interpret(vec![reference(3), array.clone()]), mismatch);
        assert_eq!(replayed.interpret(vec![reference(3), array]), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_in_context_narrows_known_symbolic_inputs_by_known_facts() {
        // A known `dimension<2>`, a known symbolic `f32[n, m]` matrix, and an unknown `f32[m]` vector enter
        // `[dimension<m>, f32[n, m], f32[m]]`, and the body adds the vector to `zero[m]` built from the dimension. The
        // known dimension fixes `m = 2` for the known matrix too, so a guard that folds into the enclosing trace
        // narrows the matrix to `f32[n, 2]`, which leaves `m` free to be narrowed in the vector, whose sum with the
        // computed `f32[2]` zero is then well typed.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [n, m] = ["n", "m"].map(|name| DimensionType::new(name, bounds));
        let array = |dimensions: Vec<Dimension>| ArrayType::new(DataType::F32, Shape::new(dimensions));
        let matrix_type: ArrayIrType = array(vec![n.simplified_dimension(), m.simplified_dimension()]).into();
        let vector_type = array(vec![m.simplified_dimension()]);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let dimension = builder.add_input(m.clone().into());
        builder.add_input(matrix_type.clone());
        let vector = builder.add_input(vector_type.clone().into());
        let zero = builder
            .add_instruction(
                TestOperation::Zero(ZeroOperation::new(vector_type.clone())),
                Vec::new(),
                vec![dimension],
                None,
            )
            .unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![vector, zero], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let known_dimension = outer.lift(TestValue::Dimension(DimensionValue::constant(2).unwrap())).unwrap();
        let known_matrix = outer.input(matrix_type);
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(known_dimension),
                    PartialValue::Known(known_matrix),
                    PartialValue::Unknown(vector_type.clone().into()),
                ],
            )
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:f32[m], %1:dimension<2>, %2:f32[n, 2], %3:f32[2] .
                let %4:dimension<2>, %5:f32[n, 2], %6:f32[2] = guard [
                    boundary="partial evaluation",
                    types=[dimension<m ∈ [1, 8)>, f32[n, m], f32[m]],
                ] %1 %2 %0
                    %7:f32[2] = add %6 %3
                in (%7)"#},
        );
        let outputs = evaluation.interpret(&outer, &[outer.input(vector_type.into())]).unwrap();
        let output_atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(output_atoms, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let guard = r#"guard [boundary="partial evaluation", types=[dimension<m ∈ [1, 8)>, f32[n, m]]]"#;
        assert_eq!(
            replayed.to_string(),
            formatdoc! {r#"
                lambda %1:f32[n, m], %5:f32[m] .
                let %0:dimension<2> = const 2
                    %2:dimension<2>, %3:f32[n, 2] = {guard} %0 %1
                    %4:f32[2] = zero [type=f32[m]] %0
                    %6:dimension<2>, %7:f32[n, 2], %8:f32[2] = guard [
                        boundary="partial evaluation",
                        types=[dimension<m ∈ [1, 8)>, f32[n, m], f32[m]],
                    ] %0 %3 %5
                    %9:f32[2] = add %8 %4
                in (%9)"#},
        );
        let dimension = TestValue::Dimension(DimensionValue::constant(2).unwrap());
        let matrix = |columns: usize| {
            TestValue::Array(
                Array::from_elements::<f32>(
                    ArrayType::new_static(DataType::F32, [3, columns]),
                    &vec![1f32; 3 * columns],
                )
                .unwrap(),
            )
        };
        let vector = |extent: usize| TestValue::Array(Array::vector(vec![1f32; extent]).unwrap());
        let mismatch = |expected, actual| -> Result<Vec<TestValue>, ProgramError> {
            Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
                dimension: "m".to_string(),
                expected,
                actual,
            })))
        };
        assert_eq!(program.interpret(vec![dimension.clone(), matrix(2), vector(2)]), Ok(vec![vector(2)]));
        assert_eq!(replayed.interpret(vec![matrix(2), vector(2)]), Ok(vec![vector(2)]));
        assert_eq!(program.interpret(vec![dimension.clone(), matrix(3), vector(2)]), mismatch(2, 3));
        assert_eq!(replayed.interpret(vec![matrix(3), vector(2)]), mismatch(3, 2));
        assert_eq!(program.interpret(vec![dimension, matrix(2), vector(3)]), mismatch(2, 3));
        assert_eq!(replayed.interpret(vec![matrix(2), vector(3)]), mismatch(2, 3));
    }

    #[test]
    fn test_program_partially_evaluate_in_context_narrows_known_values_to_exact_reference_shape() {
        // An unknown exact reference fixes `n` at runtime, even though its type carries no nominal identity to retain.
        // The symbolic known array therefore consumes the residual guard's `f32[3]` output before meeting the reference
        // read.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![n.simplified_dimension()]));
        let reference_type: ArrayIrType = ReferenceType::new(array_type.clone()).into();
        let exact_reference: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [3])).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type);
        let array = builder.add_input(array_type.clone().into());
        let read =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![read, array], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let known = outer.input(array_type.into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Unknown(exact_reference.clone()), PartialValue::Known(known)],
            )
            .unwrap();
        assert!(matches!(evaluation.outputs(), [PartialEvaluationOutput::Unknown(0)]));
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
            lambda %0:ref<f32[3]>, %1:f32[n] .
            let %2:f32[3] = guard [boundary="partial evaluation", types=[ref<f32[n]>, f32[n]]] %0 %1
                %3:f32[3] = reference_read %0
                %4:f32[3] = add %3 %2
            in (%4)"#},
        );
        let inputs = [outer.input(exact_reference)];
        let outputs = evaluation.interpret(&outer, &inputs).unwrap();
        let atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(atoms, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let reference = TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; 3]).unwrap()));
        let matching = TestValue::Array(Array::vector(vec![2f32; 3]).unwrap());
        assert_eq!(
            replayed.interpret(vec![matching, reference.clone()]),
            Ok(vec![TestValue::Array(Array::vector(vec![3f32; 3]).unwrap())]),
        );
        let mismatching = TestValue::Array(Array::vector(vec![2f32; 2]).unwrap());
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "n".to_string(),
            expected: 3,
            actual: 2,
        })));
        assert_eq!(program.interpret(vec![reference.clone(), mismatching.clone()]), mismatch);
        assert_eq!(replayed.interpret(vec![mismatching, reference]), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_in_context_retains_guarded_unknown_boundary_types() {
        // The reference retains the wide `q` while the known witness fixes the narrow `n` to 3. A guard may represent
        // its second dimension by `q`, but the residual boundary must still accept the supplied symbolic `c` at both
        // positions.
        let q = DimensionType::new("q", DimensionBounds::new(1, Some(8)).unwrap());
        let n = DimensionType::new("n", DimensionBounds::new(1, Some(4)).unwrap());
        let c = DimensionType::new("c", DimensionBounds::new(1, Some(4)).unwrap());
        let vector = |dimension: &DimensionType| {
            ArrayType::new(DataType::F32, Shape::new(vec![dimension.simplified_dimension()]))
        };
        let reference_type: ArrayIrType = ReferenceType::new(vector(&q)).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(reference_type.clone());
        let dimension = builder.add_input(n.clone().into());
        builder.add_input(q.into());
        builder.add_input(vector(&n).into());
        let zero = builder
            .add_instruction(TestOperation::Zero(ZeroOperation::new(vector(&n))), Vec::new(), vec![dimension], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![zero], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let known = TestValue::Array(Array::vector(vec![1f32; 3]).unwrap());

        let outer = TracingContext::<TestValue, TestOperation>::new();
        let witness = outer.constant(known.clone());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Unknown(reference_type.clone()),
                    PartialValue::Unknown(c.clone().into()),
                    PartialValue::Unknown(c.clone().into()),
                    PartialValue::Known(witness),
                ],
            )
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
            lambda %0:ref<f32[q]>, %1:dimension<c ∈ [1, 4)>, %2:dimension<c ∈ [1, 4)>, %3:f32[3] .
            let %4:dimension<3>, %5:dimension<q ∈ [1, 8)>, %6:f32[3] = guard [
                boundary="partial evaluation",
                types=[ref<f32[q]>, dimension<n ∈ [1, 4)>, dimension<q ∈ [1, 8)>, f32[n]],
                retained_identities=[q],
            ] %0 %1 %2 %3
                %7:f32[3] = zero [type=f32[n]] %4
            in (%7)"#},
        );
        let inputs = [outer.input(reference_type), outer.input(c.clone().into()), outer.input(c.into())];
        let outputs = evaluation.interpret(&outer, &inputs).unwrap();
        let atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(atoms, vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let reference = |extent| TestValue::Reference(ArrayReference::new(Array::vector(vec![1f32; extent]).unwrap()));
        let dimension = TestValue::Dimension(DimensionValue::constant(3).unwrap());
        let zero = TestValue::Array(Array::vector(vec![0f32; 3]).unwrap());
        assert_eq!(
            program.interpret(vec![reference(3), dimension.clone(), dimension.clone(), known.clone()]),
            Ok(vec![zero.clone()]),
        );
        assert_eq!(replayed.interpret(vec![reference(3), dimension.clone(), dimension.clone()]), Ok(vec![zero]));
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "q".to_string(),
            expected: 4,
            actual: 3,
        })));
        assert_eq!(program.interpret(vec![reference(4), dimension.clone(), dimension.clone(), known]), mismatch);
        assert_eq!(replayed.interpret(vec![reference(4), dimension.clone(), dimension]), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_in_context_consumes_known_classes_connected_by_unknown_dimensions() {
        // The known-only entry leaves `a` and `b` unrelated. The unknown `c` joins their classes at the complete entry,
        // so representing the known `b` by `a` requires the residual check and makes that output unknown, without
        // references.
        let bounds = DimensionBounds::positive(Some(8)).unwrap();
        let [p, q, a, b, c] = ["p", "q", "a", "b", "c"].map(|name| DimensionType::new(name, bounds));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(p.clone().into());
        builder.add_input(p.into());
        let output = builder.add_input(q.clone().into());
        builder.add_input(q.into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let first = outer.input(a.into());
        let second = outer.input(b.into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(first),
                    PartialValue::Unknown(c.clone().into()),
                    PartialValue::Known(second),
                    PartialValue::Unknown(c.clone().into()),
                ],
            )
            .unwrap();
        assert!(matches!(evaluation.outputs(), [PartialEvaluationOutput::Unknown(0)]));
        let [dimension_a, dimension_b, dimension_c, dimension_p, dimension_q] =
            ["a", "b", "c", "p", "q"].map(|name| format!("dimension<{name} ∈ [1, 8)>"));
        assert_eq!(
            evaluation.program().to_string(),
            formatdoc! {r#"
            lambda %0:{dimension_c}, %1:{dimension_c}, %2:{dimension_a}, %3:{dimension_b} .
            let %4:{dimension_a}, %5:{dimension_a}, %6:{dimension_a}, %7:{dimension_a} = guard [
                boundary="partial evaluation",
                types=[{dimension_p}, {dimension_p}, {dimension_q}, {dimension_q}],
                retained_identities=[a],
            ] %2 %0 %3 %1
            in (%6)"#},
        );
        let inputs = [outer.input(c.clone().into()), outer.input(c.into())];
        let outputs = evaluation.interpret(&outer, &inputs).unwrap();
        let atoms = outputs.iter().map(|output| output.atom_id().unwrap()).collect();
        let replayed = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(atoms, vec![Placeholder; 4], vec![Placeholder])
            .unwrap();
        let dimension = |extent| TestValue::Dimension(DimensionValue::constant(extent).unwrap());
        assert_eq!(replayed.interpret(vec![dimension(2); 4]), Ok(vec![dimension(2)]));
        let values = vec![dimension(2), dimension(3), dimension(2), dimension(2)];
        let mismatch = Err(ProgramError::Type(TypeError::custom(DimensionError::InputDimensionMismatch {
            dimension: "q".to_string(),
            expected: 3,
            actual: 2,
        })));
        assert_eq!(
            program.interpret(vec![values[0].clone(), values[2].clone(), values[1].clone(), values[3].clone()]),
            mismatch,
        );
        assert_eq!(replayed.interpret(values), mismatch);
    }

    #[test]
    fn test_program_partially_evaluate_in_context_defers_known_effects_behind_shared_identities() {
        // A known `f32[n]` from the enclosing trace and an unknown `f32[n]` share `n`, which their types prove, but
        // only the residual invocation supplies the unknown value. A guard over both therefore precedes the body, so
        // that the known print follows the check instead of being staged in the enclosing trace before the residual
        // runs, exactly as the source program checks its entry before printing.
        let n = DimensionType::new("n", DimensionBounds::positive(Some(8)).unwrap());
        let vector_type: ArrayIrType = ArrayType::new(DataType::F32, Shape::new(vec![n.dimension().clone()])).into();
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let print = |label: &str| TestOperation::Array(ArrayOperation::from(PrintOperation::<ArrayType>::new(label)));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        builder.add_input(vector_type.clone());
        let unknown = builder.add_input(vector_type.clone());
        let known_scalar = builder.add_input(scalar_type.clone());
        let unknown_scalar = builder.add_input(scalar_type.clone());
        let known_print = builder.add_instruction(print("known"), Vec::new(), vec![known_scalar], None).unwrap()[0];
        let unknown_print =
            builder.add_instruction(print("unknown"), Vec::new(), vec![unknown_scalar], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![unknown, known_print, unknown_print],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.input(vector_type.clone())),
                    PartialValue::Unknown(vector_type.clone()),
                    PartialValue::Known(outer.input(scalar_type.clone())),
                    PartialValue::Unknown(scalar_type.clone()),
                ],
            )
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {r#"
                lambda %0:f32[n], %1:f32[], %2:f32[n], %3:f32[] .
                let %4:f32[n], %5:f32[n] = guard [boundary="partial evaluation", types=[f32[n], f32[n]]] %2 %0
                    %6:f32[] = print [label=known] %3
                    %7:f32[] = print [label=unknown] %1
                in (%5, %6, %7)"#},
        );
        // The enclosing trace stages nothing of the body before the residual runs.
        let staged = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        assert_eq!(
            staged.to_string(),
            indoc! {"
                lambda %0:f32[n], %1:f32[] .
                in ()"},
        );
    }
}
