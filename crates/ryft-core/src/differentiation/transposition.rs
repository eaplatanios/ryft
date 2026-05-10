use std::cell::RefCell;
use std::rc::Rc;

use crate::operations::Operation;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::SupportsZero;
use crate::parameters::{Parameter, Parameterized};
use crate::tracing::domains::{
    ProgramTracer, ProgramTracingContext, ProgramTracingDomain, RuntimeDomain, Tracer, TracingContext, TracingDomain,
};
use crate::tracing::{AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError};
use crate::types::{Type, Typed};

/// [`Cotangent`] produced when differentiating a [`Program`] and which is the main value type that
/// [_transposition_](Program::transpose) operates over.
///
/// In order to explain what a cotangent is more formally, let us introduce some notation:
///
///   - `f: X -> Y` is a _differentiable map_.
///   - `x` is a point in the input space `X`.
///   - `T_x X` is the _tangent_ space of `X` at `x`; its elements are input perturbations or directions.
///   - `T_x^* X` is the _dual_ of `T_x X`; its elements are _cotangents_ (i.e., linear functionals) `T_x X -> R`.
///   - `d f_x: T_x X -> T_{f(x)} Y` is the derivative of `f` at `x`, viewed as a _linear map_ that pushes input
///     tangents forward to output tangents.
///
/// Given an output cotangent `bar_y` in `T_{f(x)}^* Y`, reverse-mode differentiation computes the input cotangent
/// `bar_x` in `T_x^* X` by applying the dual, or pullback, of the derivative: `(d f_x)^*: T_{f(x)}^* Y -> T_x^* X`.
/// Formally, `bar_x = (d f_x)^*(bar_y)` is defined by `bar_x(dot_x) = bar_y(d f_x(dot_x))` for every input tangent
/// `dot_x` in `T_x X`. In finite-dimensional coordinates, if `d f_x` is represented by the Jacobian matrix `J_f(x)`,
///  this is the vector-Jacobian product `bar_x = J_f(x)^T bar_y`.
///
/// In the [`transposition`](crate::differentiation::transposition) module, the derivative has already been staged as a
/// linear tangent pushforward [`Program`]. Transposition builds the dual pullback program, and [`Cotangent`] is the
/// rule-boundary representation of one symbolic cotangent contribution during that construction. [`Cotangent::Zero`]
/// represents a structural zero: no atom is staged in the transpose builder because the current instruction contributes
/// nothing to that input cotangent. [`Cotangent::Staged`] carries an actual symbolic cotangent [`Tracer`] in the active
/// [`ProgramTracingContext`].
pub enum Cotangent<'domain, T: Type + Parameter, V: Traceable<T>, O: Operation<T>> {
    /// [`Cotangent`] value that is known to be zero, structurally, and thus has not corresponding staged atom.
    Zero,

    /// [`Cotangent`] value that is staged in a [`Program`] that is being traced.
    Staged(ProgramTracer<'domain, T, V, O>),
}

impl<'domain, T: Type + Parameter, V: Traceable<T>, O: Operation<T>> Cotangent<'domain, T, V, O> {
    /// Creates a new [`Cotangent::Zero`].
    #[inline]
    pub const fn zero() -> Self {
        Self::Zero
    }

    /// Creates a new [`Cotangent::Staged`].
    #[inline]
    pub const fn staged(cotangent: ProgramTracer<'domain, T, V, O>) -> Self {
        Self::Staged(cotangent)
    }

    /// Returns `true` if this is a [`Cotangent::Zero`].
    #[inline]
    pub const fn is_zero(&self) -> bool {
        matches!(self, Self::Zero)
    }

    /// Returns the [`ProgramTracer`] stored in this [`Cotangent`], if it is a [`Cotangent::Staged`],
    /// and `None` otherwise.
    #[inline]
    pub fn as_staged(&self) -> Option<&ProgramTracer<'domain, T, V, O>> {
        match self {
            Self::Zero => None,
            Self::Staged(cotangent) => Some(cotangent),
        }
    }
}

impl<'domain, T: Type + Parameter, V: Traceable<T>, O: Operation<T>> Clone for Cotangent<'domain, T, V, O> {
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Zero => Self::Zero,
            Self::Staged(cotangent) => Self::Staged(cotangent.clone()),
        }
    }
}

impl<'domain, T: Type + Parameter, V: Traceable<T>, O: Operation<T>> From<ProgramTracer<'domain, T, V, O>>
    for Cotangent<'domain, T, V, O>
{
    #[inline]
    fn from(cotangent: ProgramTracer<'domain, T, V, O>) -> Self {
        Self::staged(cotangent)
    }
}

/// Represents [`Operation`]s that are _linear_, meaning that they can be _transposed_. For a linear [`Instruction`]
/// `y = L(x)`, [`transpose`](Self::transpose) receives symbolic cotangent [`Tracer`]s for `y` and returns symbolic
/// cotangent contributions for `x`, representing the transposed cotangent. Rules may reuse existing cotangents,
/// return [`Cotangent::Zero`] for structural zeros, or stage additional linear operations in the active
/// [`ProgramTracingContext`]. The rule does not receive concrete primal values; any required metadata must be encoded
/// in the operation itself or in staged atom types.
///
/// Refer to the documentation of [`Program::transpose`] for more information what _transposition_ means here and how
/// it relates to the algebraic notion of transposition.
pub trait LinearOperation<T: Type + Parameter, V: Traceable<T>, O: Operation<T>>: Operation<T> {
    /// Applies this operation's transpose rule to the provided symbolic output cotangents. The returned vector must
    /// contain one entry per operation input. Each [`Cotangent::Staged`] value is a staged cotangent contribution in
    /// the active transpose builder, and each [`Cotangent::Zero`] means that the corresponding input receives a
    /// structural zero from this operation.
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Cotangent<'transpose, T, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, T, V, O>>, TracingError>;
}

impl<
    T: Type + Parameter,
    V: Traceable<T>,
    O: LinearOperation<T, V, O> + SupportsZero<T, V> + SupportsAdd<T, V>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> Program<T, V, O, Input, Output>
{
    /// Transposes this linear pushforward [`Program`] into its reverse-mode pullback. This is the main entrypoint for
    /// transposing linear [`Program`]s. In the algebraic sense, _transposing_ a linear map `L: X -> Y` gives a map on
    /// _dual_ spaces `L^T: Y* -> X*`. In finite dimensions this is the same operation represented by a matrix
    /// transpose. Here the linear map is not stored as a matrix. It is a staged [`Program`] that maps input tangents
    /// to output tangents, and transposition builds the dual program that maps output cotangents back to input
    /// cotangents. Operationally, transposition creates cotangent inputs for this program's outputs, walks the
    /// instructions in reverse order, and applies each primitive operation's [`LinearOperation::transpose`] rule
    /// to accumulate cotangent contributions for the original inputs. This is the same decomposition of reverse-mode
    /// automatic differentiation as in [this paper](https://arxiv.org/abs/2204.10923).
    ///
    /// Disconnected primal inputs are emitted as [`ZeroOperation`](crate::operations::ZeroOperation)s, which the value
    /// type's [`Zero`](crate::Zero) implementation evaluates at interpretation time. For linear programs whose values
    /// are [`Tracer`]s from an outer trace, use [`TracingContext::transpose`] instead so that those disconnected-input
    /// zeros can be materialized in the surrounding tracing context.
    #[inline]
    pub fn transpose(&self) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<T, V, O>::new()));
        let domain = ProgramTracingDomain::new();
        let mut context = ProgramTracingContext::new(&domain, builder);
        context.transpose(self)
    }
}

impl<'domain, D: RuntimeDomain + TracingDomain> TracingContext<'domain, D> {
    /// Transposes a traced linear [`Program`] and materializes standalone zero [`Cotangent`]s. This method performs the
    /// same program transposition as [`Program::transpose`], but is specialized for linear programs whose values are
    /// [`Tracer`]s belonging to this [`TracingContext`]. Use it when transposing a linear program inside an outer
    /// trace, such as when staging a traced reverse-mode pullback.
    ///
    /// The transposed program is first staged as an ordinary linear [`Program`]. When a primal input is disconnected
    /// from the outputs, transposition represents its cotangent as a [`ZeroOperation`](crate::ZeroOperation). Such an
    /// input-free operation cannot recover the surrounding [`TracingContext`] during later interpretation, and so this
    /// method replaces each standalone zero operation with a constant [`Tracer`] created in this [`TracingContext`].
    /// The concrete zero value stored in that tracer is synthesized through [`RuntimeDomain::zero`], while the final
    /// pullback still receives and returns traced [`Cotangent`]s.
    pub fn transpose<
        Input: Parameterized<Tracer<'domain, D>>,
        Output: Parameterized<Tracer<'domain, D>>,
        O: Clone
            + LinearOperation<D::Type, Tracer<'domain, D>, O>
            + SupportsZero<D::Type, Tracer<'domain, D>>
            + SupportsAdd<D::Type, Tracer<'domain, D>>,
    >(
        &self,
        program: &Program<D::Type, Tracer<'domain, D>, O, Input, Output>,
    ) -> Result<Program<D::Type, Tracer<'domain, D>, O, Output, Input>, TracingError> {
        // First build the ordinary transposed program. At this point disconnected inputs are still represented
        // as input-free zero operations in the transposed program.
        let builder = Rc::new(RefCell::new(ProgramBuilder::<D::Type, Tracer<'domain, D>, O>::new()));
        let domain = ProgramTracingDomain::new();
        let mut context = ProgramTracingContext::new(&domain, builder);
        let transposed_program = context.transpose(program)?;

        // Rewrite the transposed program into a sibling builder. We preserve the existing atom table and inputs,
        // then use `atom_remapping` only for atoms that need to point at replacement constants.
        let mut builder = ProgramBuilder::<D::Type, Tracer<'domain, D>, O>::new();
        builder.atoms = transposed_program.atoms.clone();
        builder.input_ids = transposed_program.input_ids.clone();
        let mut atom_remapping = vec![None; builder.atoms.len()];
        let mut rewritten_instructions = Vec::with_capacity(transposed_program.instructions.len());
        for instruction in &transposed_program.instructions {
            if let Some(zero_operation) = instruction.operation.as_zero_operation()
                && instruction.outputs.len() == 1
                && instruction.inputs.is_empty()
            {
                // Zero operations in traced pullbacks have no inputs from which interpretation can recover a tracing
                // context, and so we materialize each one as a constant in this tracing context and remap its uses.
                let zero = builder.add_constant(self.constant(self.domain.zero(&zero_operation.r#type)?));
                atom_remapping[instruction.outputs[0].index] = Some(zero);
            } else {
                // Preserve non-zero instructions, rewriting only the inputs that consumed a zero operation
                // we replaced with a traced constant above.
                let inputs = instruction
                    .inputs
                    .iter()
                    .map(|atom| atom_remapping[atom.index].unwrap_or(*atom))
                    .collect::<Vec<_>>();
                rewritten_instructions.push(Instruction {
                    operation: instruction.operation.clone(),
                    inputs,
                    outputs: instruction.outputs.clone(),
                });
            }
        }
        builder.instructions = rewritten_instructions;

        // Outputs can also refer directly to replaced zero-operation atoms, and so we apply the same remapping before
        // building. The subsequent simplification removes the skipped zero instructions and their old output atoms.
        let outputs = transposed_program
            .output_ids
            .iter()
            .map(|atom| atom_remapping[atom.index].unwrap_or(*atom))
            .collect::<Vec<_>>();
        builder
            .build(outputs, transposed_program.input_structure.clone(), transposed_program.output_structure.clone())?
            .into_simplified()
    }
}

impl<
    'domain,
    T: 'domain + Type + Parameter,
    V: 'domain + Traceable<T>,
    O: 'domain + LinearOperation<T, V, O> + SupportsZero<T, V> + SupportsAdd<T, V>,
> TracingContext<'domain, ProgramTracingDomain<T, V, O>>
{
    /// Transposes a complete linear [`Program`] using this context's current builder.
    ///
    /// This is the builder-level implementation behind [`Program::transpose`]. See
    /// [`Program::transpose`] for the conceptual relationship between program transposition,
    /// algebraic transposition, pushforwards, and pullbacks. This method is for callers that
    /// already own a [`ProgramTracingContext`] and need the transposed program to be staged through
    /// that context's active [`ProgramBuilder`].
    ///
    /// The method treats [`builder`](Self::builder) as the destination for the transposed program,
    /// records cotangent inputs for the primal outputs, walks `program` in reverse instruction
    /// order, and applies [`LinearOperation::transpose`] to each instruction. The active builder is
    /// consumed when the pullback is built. On success, this context is left with a fresh empty
    /// builder. If a transpose rule needs to transpose a nested program while preserving the
    /// surrounding builder, it should call [`transpose_nested`](Self::transpose_nested) instead.
    ///
    /// Most ordinary callers should use [`Program::transpose`] instead of constructing a
    /// [`ProgramTracingContext`] directly.
    pub fn transpose<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
    ) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        fn accumulate<T: Type, V: Traceable<T>, O: Operation<T>>(
            builder: &Rc<RefCell<ProgramBuilder<T, V, O>>>,
            adjoints: &mut [Option<AtomId>],
            atom: AtomId,
            contribution: AtomId,
        ) -> Result<(), TracingError>
        where
            O: SupportsAdd<T, V>,
        {
            adjoints[atom.index] = Some(match adjoints[atom.index] {
                Some(existing) => {
                    let mut builder_borrow = builder.borrow_mut();
                    let abstract_value = builder_borrow.atoms[existing.index].r#type().into_owned();
                    let output = builder_borrow.add_variable(abstract_value);
                    builder_borrow.instructions.push(Instruction {
                        operation: O::add_operation(),
                        inputs: vec![existing, contribution],
                        outputs: vec![output],
                    });
                    output
                }
                None => contribution,
            });
            Ok(())
        }

        fn stage_zero<T: Type, V: Traceable<T>, O: Operation<T>>(
            builder: &Rc<RefCell<ProgramBuilder<T, V, O>>>,
            r#type: T,
        ) -> AtomId
        where
            O: SupportsZero<T, V>,
        {
            let mut builder_borrow = builder.borrow_mut();
            let output = builder_borrow.add_variable(r#type.clone());
            builder_borrow.instructions.push(Instruction {
                operation: O::zero_operation(r#type),
                inputs: vec![],
                outputs: vec![output],
            });
            output
        }

        let builder = self.builder.clone();
        let mut output_cotangent_inputs = Vec::with_capacity(program.output_ids.len());
        for output in program.output_ids.iter() {
            let output_atom = program.atoms.get(output.index).ok_or(TracingError::UnboundAtomId { id: *output })?;
            let cotangent_input = builder.borrow_mut().add_input(output_atom.r#type().into_owned());
            output_cotangent_inputs.push(cotangent_input);
        }

        let mut adjoints = vec![None; program.atoms.len()];
        for (cotangent, output) in output_cotangent_inputs.into_iter().zip(program.output_ids.iter().copied()) {
            accumulate::<T, V, O>(&builder, adjoints.as_mut_slice(), output, cotangent)?;
        }

        for instruction in program.instructions.iter().rev() {
            if instruction.outputs.iter().all(|output| adjoints[output.index].is_none()) {
                continue;
            }
            let instruction_output_cotangents = instruction
                .outputs
                .iter()
                .map(|output| match adjoints[output.index] {
                    Some(atom) => Cotangent::staged(self.tracer(atom, None)),
                    None => Cotangent::Zero,
                })
                .collect::<Vec<_>>();
            let input_cotangents = instruction.operation.transpose(self, instruction_output_cotangents.as_slice())?;
            for (input, contribution) in instruction.inputs.iter().copied().zip(input_cotangents) {
                if let Some(contribution) = contribution.as_staged() {
                    accumulate::<T, V, O>(&builder, adjoints.as_mut_slice(), input, contribution.atom_id()?)?;
                }
            }
        }

        let outputs = program
            .input_ids
            .iter()
            .copied()
            .map(|input| match adjoints[input.index] {
                Some(adjoint) => adjoint,
                None => stage_zero::<T, V, O>(&builder, program.atoms[input.index].r#type().into_owned()),
            })
            .collect::<Vec<_>>();
        drop(builder);
        let builder = self.builder.clone();
        self.builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => return Err(TracingError::EscapedProgramBuilder),
        };
        builder.build(outputs, program.output_structure.clone(), program.input_structure.clone())
    }

    /// Transposes a nested linear [`Program`] without consuming this context's current builder.
    ///
    /// This is the nested-program variant of [`ProgramTracingContext::transpose`]. See
    /// [`Program::transpose`] for the conceptual meaning of transposition. This method is for
    /// transpose rules that carry linear subprograms as operation metadata, such as captured
    /// control-flow branches.
    ///
    /// It temporarily replaces [`builder`](Self::builder) with a fresh sibling builder, calls
    /// [`transpose`](Self::transpose) for the nested `program`, and then restores the original
    /// builder before returning the nested pullback result. This keeps nested transposition from
    /// appending instructions to the surrounding pullback or consuming the builder that the
    /// surrounding rule still needs. The original builder is restored whether the nested
    /// transposition succeeds or returns an error.
    #[inline]
    pub fn transpose_nested<Input: Parameterized<V>, Output: Parameterized<V>>(
        &mut self,
        program: &Program<T, V, O, Input, Output>,
    ) -> Result<Program<T, V, O, Output, Input>, TracingError> {
        let parent_builder = self.builder.clone();
        self.builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let result = self.transpose(program);
        self.builder = parent_builder;
        result
    }
}
