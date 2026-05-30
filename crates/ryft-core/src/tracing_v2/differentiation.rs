use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use thiserror::Error;

use crate::differentiation::{LinearOperation, Tangent};
use crate::macros::check_count;
use crate::operations::arithmetic::SupportsAdd;
use crate::operations::constants::{One, SupportsOne, SupportsZero};
use crate::operations::scalars::LinearScalarOperation;
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily};
use crate::tracing::contexts::{CaptureContext, Context, TracingContext};
use crate::tracing::domains::{
    Domain, LinearScalarDomain, ProgramTracingDomain, RuntimeDomain, ScalarDomain, Tracer, TracerState, TracingDomain,
};
use crate::tracing::{Atom, AtomId, Program, ProgramBuilder, Traceable, TracingError};
use crate::types::{ArrayType, DataType, Type, Typed};

/// Errors emitted by the differentiation helpers in [`crate::tracing_v2`].
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Reverse-mode gradient was requested for a non-scalar array output.
    #[error("gradient output must be a rank-0 scalar array but got {output_type}")]
    NonScalarGradientOutput { output_type: ArrayType },
}

/// Operation carrier selected by a [`Differentiable`] implementation for its active tangent program.
pub type LinearOperationCarrier<E> = <E as Differentiable>::LinearOperationCarrier<<E as Differentiable>::Tangent>;

/// Tracer leaf used while executing one active concrete-domain linearization pass.
pub type LinearizationTracer<'domain, D> = Tracer<LinearizationContext<'domain, TracingContext<'domain, D>, D>>;

/// Per-run trace context used by [`DifferentiableDomain::linearize`].
///
/// [`LinearizationContext`] is not a backend capability. It is a one-shot [`Context`] that
/// intercepts primitive staging while a function is being linearized, runs each primitive through
/// its JVP rule, stores primal values as they are computed, and records the tangent program in the
/// selected linear domain. No primal program is built for later interpretation.
///
/// `C` is the active operation context exposed to user-facing [`Tracer`] leaves. The `D` parameter is the
/// [`Differentiable`] implementation used to execute primitive rules. Concrete linearization uses an ordinary
/// [`TracingContext`] as `C` and the underlying backend domain as `D`. Nested linearization uses the enclosing context
/// as `D`, so primal values are outer-context [`Tracer`] leaves and tangent operations are staged into a linear program
/// whose values are those outer tracers.
pub struct LinearizationContext<'domain, C, D>
where
    C: Context + 'domain,
    D: Differentiable<Type = C::Type, CapturedValue = C::Value> + 'domain,
{
    /// [`Differentiable`] implementation used to run primitive JVP rules.
    differentiable: DifferentiableStorage<'domain, D>,

    /// Builder used as the primal-side atom arena for user-facing linearization tracers.
    ///
    /// The active linearization path does not build or interpret a primal [`Program`]. This builder
    /// exists because traced values still need stable primal-side [`AtomId`] handles, input and
    /// variable type metadata, constants created through [`Context::constant`], the shared
    /// construction error slot used by poisoned tracers, and builder-identity checks. The active
    /// [`Context::stage_operation`] implementation never appends primal instructions here; it
    /// evaluates primal values immediately and stores them in [`Self::primal_values`].
    primal_builder: Rc<RefCell<ProgramBuilder<<C as Context>::Type, <C as Context>::Value, C::Operation>>>,

    /// Builder that owns the staged pushforward program.
    linear_builder: Rc<RefCell<ProgramBuilder<C::Type, D::Tangent, LinearOperationCarrier<D>>>>,

    /// Primal values indexed by primal-side atom id in the differentiable implementation's value representation.
    primal_values: Rc<RefCell<Vec<Option<D::Value>>>>,

    /// Tangent atom identifiers indexed by primal-side atom id. Missing entries represent structural zeros.
    tangent_atoms: Rc<RefCell<Vec<Option<AtomId>>>>,

    /// Marker tying this linearization context to the active context type.
    marker: std::marker::PhantomData<fn() -> C>,
}

/// Borrowed-or-owned [`Differentiable`] storage used by [`LinearizationContext`].
enum DifferentiableStorage<'domain, D: Differentiable + 'domain> {
    /// Borrowed trace used by concrete linearization.
    Borrowed(&'domain D),

    /// Owned cloned trace used by traced linearization.
    Owned(Rc<D>),
}

impl<D: Differentiable> Clone for DifferentiableStorage<'_, D> {
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(trace) => Self::Borrowed(trace),
            Self::Owned(trace) => Self::Owned(trace.clone()),
        }
    }
}

impl<D: Differentiable> DifferentiableStorage<'_, D> {
    /// Returns the stored [`Differentiable`] implementation.
    #[inline]
    fn as_ref(&self) -> &D {
        match self {
            Self::Borrowed(trace) => trace,
            Self::Owned(trace) => trace.as_ref(),
        }
    }
}

/// Runs one active linearization pass for either a concrete domain or an already-active context.
///
/// `C` is the context whose tracers are exposed to the user closure. `D` is the [`Differentiable`] implementation that
/// owns primal semantics and the linear operation carrier. Concrete-domain linearization uses a borrowed domain as `D`
/// and an ordinary [`TracingContext`] as `C`; nested active-context linearization uses the same context for both roles.
fn linearize_with_context<'context, C, D, F, Input, Output, Validate>(
    differentiable: DifferentiableStorage<'context, D>,
    primals: Input,
    mut validate_primal: Validate,
    function: F,
) -> Result<
    (Output, Program<C::Type, D::Tangent, LinearOperationCarrier<D>, Input::To<D::Tangent>, Output::To<D::Tangent>>),
    TracingError,
>
where
    C: Context + 'context,
    D: Differentiable<Type = C::Type, CapturedValue = C::Value> + 'context,
    C::Operation: DifferentiableOperation<D>,
    F: FnOnce(
        Input::To<Tracer<LinearizationContext<'context, C, D>>>,
    ) -> Result<Output::To<Tracer<LinearizationContext<'context, C, D>>>, TracingError>,
    Input: Parameterized<
            D::Value,
            Family: ParameterizedFamily<Tracer<LinearizationContext<'context, C, D>>> + ParameterizedFamily<D::Tangent>,
            ParameterStructure: Debug + PartialEq,
        >,
    Output: Parameterized<
            D::Value,
            Family: ParameterizedFamily<Tracer<LinearizationContext<'context, C, D>>> + ParameterizedFamily<D::Tangent>,
        >,
    Output::To<Tracer<LinearizationContext<'context, C, D>>>:
        Parameterized<Tracer<LinearizationContext<'context, C, D>>, To<D::Value> = Output>,
    Validate: FnMut(&D::Value) -> Result<(), TracingError>,
{
    let input_structure = primals.parameter_structure();
    let input_primals = primals.into_parameters().collect::<Vec<_>>();
    let primal_builder = Rc::new(RefCell::new(ProgramBuilder::<C::Type, C::Value, C::Operation>::new()));
    let linear_builder = Rc::new(RefCell::new(ProgramBuilder::<C::Type, D::Tangent, LinearOperationCarrier<D>>::new()));
    let (output_structure, output_primals, output_tangent_atoms) = {
        let linearization_context = LinearizationContext::new_with_differentiable(
            differentiable,
            primal_builder.clone(),
            linear_builder.clone(),
        );
        let mut input_tracers = Vec::with_capacity(input_primals.len());
        for input_primal in input_primals {
            validate_primal(&input_primal)?;
            let input_type = input_primal.r#type().into_owned();
            let primal_atom = primal_builder.borrow_mut().add_input(input_type.clone());
            let tangent_atom = linear_builder.borrow_mut().add_input(input_type.clone());
            linearization_context.register_input(primal_atom, input_primal, tangent_atom);
            input_tracers.push(linearization_context.tracer(primal_atom, Some(input_type)));
        }

        let input = Input::To::<Tracer<LinearizationContext<'context, C, D>>>::from_parameters(
            input_structure.clone(),
            input_tracers,
        )?;
        let output = function(input).map_err(|error| primal_builder.borrow_mut().error.take().unwrap_or(error))?;
        primal_builder.borrow_mut().error.take().map_or(Ok(()), Err)?;

        let output_structure = output.parameter_structure();
        let output_atoms = output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
        drop(output);
        let (output_primals, output_tangent_atoms) = linearization_context.collect_outputs(output_atoms.as_slice())?;
        (output_structure, output_primals, output_tangent_atoms)
    };
    Rc::try_unwrap(primal_builder).map_err(|_| TracingError::EscapedProgramBuilder)?;
    let linear_builder = Rc::try_unwrap(linear_builder).map_err(|_| TracingError::EscapedProgramBuilder)?;
    let pushforward = linear_builder
        .into_inner()
        .build(output_tangent_atoms, input_structure, output_structure.clone())?
        .simplified()?;
    Ok((Output::from_parameters(output_structure, output_primals)?, pushforward))
}

/// Domain capability required for automatic-differentiation transforms that linearize staged programs.
///
/// This is the only backend-specific domain fact that `ryft-core` cannot infer from [`RuntimeDomain`] and
/// [`TracingDomain`]. Once a backend selects a linear domain, core derives the tangent leaf type from
/// [`Domain::Value`] on that linear domain, derives the tangent operation carrier from
/// [`TracingDomain::Operation`] on that linear domain, and uses the backend's ordinary tracing operation for
/// differentiable primal programs.
///
/// A linearizable domain chooses the tangent/cotangent carrier used while building linear programs. Interpretation and
/// transposition are additional capabilities of that carrier, so they are required only by the methods that replay or
/// transpose a completed linear program. This keeps pure linearization usable for staged domains whose values are
/// abstract metadata rather than runtime values.
pub trait LinearizableDomain: RuntimeDomain + TracingDomain<Operation: Clone> {
    /// Tangent and cotangent leaf type selected by this differentiable domain.
    type Tangent: Traceable<Self::Type>;

    /// Linear operation carrier specialized to the value representation used by an active transform frame.
    ///
    /// Concrete linear programs use `V = Self::Tangent`. Nested transform frames use the same backend carrier family
    /// with `V` set to the frame's tracer type. Keeping this as the domain's own GAT avoids a separate carrier-family
    /// registry and lets custom backend variants, such as XLA's linear call operations, reparameterize themselves at
    /// the same point where the backend chooses the rest of its linearization model.
    type LinearOperationCarrier<V>: Clone + Operation<Self::Type>
    where
        V: Traceable<Self::Type>;

    /// Tracing domain selected by this domain for tangent and cotangent programs.
    type LinearDomain: RuntimeDomain<Type = Self::Type, Value = Self::Tangent>
        + TracingDomain<Type = Self::Type, Value = Self::Tangent, Operation = Self::LinearOperationCarrier<Self::Tangent>>;

    /// Returns the linearizable domain used for tangent and cotangent programs.
    fn linear_domain(&self) -> &Self::LinearDomain;
}

impl<D> DifferentiableDomain for D
where
    D: LinearizableDomain,
    D::LinearOperationCarrier<<D as LinearizableDomain>::Tangent>: SupportsZero<D::Type, <D as LinearizableDomain>::Tangent>
        + SupportsAdd<D::Type, <D as LinearizableDomain>::Tangent>,
{
    type Tangent = <D as LinearizableDomain>::Tangent;
    type LinearDomain = <D as LinearizableDomain>::LinearDomain;
    type LinearOperationCarrier<V>
        = <D as LinearizableDomain>::LinearOperationCarrier<V>
    where
        V: Traceable<D::Type>;

    #[inline]
    fn linear_domain(&self) -> &Self::LinearDomain {
        LinearizableDomain::linear_domain(self)
    }
}

/// Extension of [`RuntimeDomain`] for backends that support automatic differentiation.
///
/// Backends that only need ordinary tracing implement [`TracingDomain`] without this extension. AD
/// transforms such as [`DifferentiableDomain::jvp`], [`DifferentiableDomain::value_and_gradient`], and
/// [`DifferentiableDomain::vjp`] require this trait so non-differentiable backends do not need to define fake tangent
/// carriers.
///
/// Backends usually do not implement this trait directly. Implement [`LinearizableDomain`] instead and let the
/// blanket implementation compose the full AD API in `ryft-core`.
///
/// Differentiated closures are traced with the domain's ordinary [`TracingDomain::Operation`]. Individual
/// transforms that linearize a staged primal program require that carrier to implement [`DifferentiableOperation`] for
/// the active domain, so backends do not need a second operation-carrier API just for AD.
pub trait DifferentiableDomain: RuntimeDomain + TracingDomain<Operation: Clone> {
    /// Tangent and cotangent leaf type selected by this differentiable domain.
    type Tangent: Traceable<Self::Type>;

    /// Tracing domain selected by this differentiable domain for tangent and cotangent programs.
    type LinearDomain: RuntimeDomain<Type = Self::Type, Value = Self::Tangent>
        + TracingDomain<Type = Self::Type, Value = Self::Tangent, Operation = Self::LinearOperationCarrier<Self::Tangent>>;

    /// Linear operation carrier specialized to the value representation used by an active transform frame.
    ///
    /// Concrete domain transforms use `V = Self::Tangent`; active nested transforms use `V = Tracer<C>` for the
    /// enclosing context. Backends define the family once, so core can stage the same linear operation language over
    /// concrete tangent values and over tracers without a separate carrier reparameterization trait.
    type LinearOperationCarrier<V>: Clone + Operation<Self::Type>
    where
        V: Traceable<Self::Type>;

    /// Returns the linearizable domain used for tangent and cotangent programs.
    fn linear_domain(&self) -> &Self::LinearDomain;

    /// Executes `function` once in an active linearization context and returns both its primal output
    /// and a reusable pushforward program.
    ///
    /// [`DifferentiableDomain::linearize`] is the staged counterpart to [`DifferentiableDomain::jvp`].
    /// Instead of immediately applying a tangent input, it captures the Jacobian-vector product as
    /// a staged [`Program`] over linear operations that can be replayed later on any tangent with
    /// the same parameter structure. Concrete primal values are interpreted as each primitive is
    /// staged through [`LinearizationContext`]; no top-level primal program is replayed.
    fn linearize<'domain, F, Input, Output>(
        &'domain self,
        function: F,
        primal: Input,
    ) -> Result<
        (
            Output,
            Program<
                Self::Type,
                Self::Tangent,
                Self::LinearOperationCarrier<Self::Tangent>,
                Input::To<Self::Tangent>,
                Output::To<Self::Tangent>,
            >,
        ),
        TracingError,
    >
    where
        Self: 'domain,
        Self::Operation: DifferentiableOperation<Self>,
        F: FnOnce(
            Input::To<LinearizationTracer<'domain, Self>>,
        ) -> Result<Output::To<LinearizationTracer<'domain, Self>>, TracingError>,
        Input: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<LinearizationTracer<'domain, Self>> + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<LinearizationTracer<'domain, Self>> + ParameterizedFamily<Self::Tangent>,
                To<LinearizationTracer<'domain, Self>>: Parameterized<
                    LinearizationTracer<'domain, Self>,
                    To<Self::Value> = Output,
                >,
            >,
    {
        linearize_with_context::<TracingContext<'domain, Self>, Self, F, Input, Output, _>(
            DifferentiableStorage::Borrowed(self),
            primal,
            |_| Ok(()),
            function,
        )
    }

    /// Evaluates `function` on `primals` and propagates the supplied tangent values forward.
    ///
    /// The returned pair is `(primal_output, tangent_output)`. This is the canonical user-facing forward-mode
    /// Jacobian-Vector Product (JVP) entry point for differentiable domains.
    fn jvp<'domain, F, Input, Output>(
        &'domain self,
        function: F,
        primal: Input,
        tangent: Input::To<Self::Tangent>,
    ) -> Result<(Output, Output::To<Self::Tangent>), TracingError>
    where
        Self: 'domain,
        Self::Operation: DifferentiableOperation<Self>,
        F: FnOnce(Input::To<LinearizationTracer<'domain, Self>>) -> Output::To<LinearizationTracer<'domain, Self>>,
        Input: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<LinearizationTracer<'domain, Self>> + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<LinearizationTracer<'domain, Self>> + ParameterizedFamily<Self::Tangent>,
                To<LinearizationTracer<'domain, Self>>: Parameterized<
                    LinearizationTracer<'domain, Self>,
                    To<Self::Value> = Output,
                >,
            >,
        Self::LinearOperationCarrier<Self::Tangent>: InterpretableOperation<Self::Type, Self::Tangent>,
    {
        let primal_structure = primal.parameter_structure();
        let tangent_structure = tangent.parameter_structure();
        if primal_structure != tangent_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        let (primal_output, tangent_program) = self.linearize(|input| Ok(function(input)), primal)?;
        let tangent_output = tangent_program.interpret(tangent)?;
        Ok((primal_output, tangent_output))
    }

    /// Returns the primal output together with a pullback produced by transposing the staged pushforward.
    ///
    /// [`DifferentiableDomain::vjp`] is the reusable reverse-mode primitive in the public API. It linearizes the
    /// primal function, builds the corresponding pushforward program, and then transposes that pushforward into a
    /// staged pullback that maps output cotangents back to input cotangents.
    fn vjp<'domain, F, Input, Output>(
        &'domain self,
        function: F,
        primals: Input,
    ) -> Result<
        (
            Output,
            Program<
                Self::Type,
                Self::Tangent,
                Self::LinearOperationCarrier<Self::Tangent>,
                Output::To<Self::Tangent>,
                Input::To<Self::Tangent>,
            >,
        ),
        TracingError,
    >
    where
        Self: 'domain,
        Self::Operation: DifferentiableOperation<Self>,
        F: FnOnce(
            Input::To<LinearizationTracer<'domain, Self>>,
        ) -> Result<Output::To<LinearizationTracer<'domain, Self>>, TracingError>,
        Input: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<LinearizationTracer<'domain, Self>> + ParameterizedFamily<Self::Tangent>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<LinearizationTracer<'domain, Self>> + ParameterizedFamily<Self::Tangent>,
                To<LinearizationTracer<'domain, Self>>: Parameterized<
                    LinearizationTracer<'domain, Self>,
                    To<Self::Value> = Output,
                >,
            >,
        Self::LinearOperationCarrier<Self::Tangent>: LinearOperation<Self::Type, Self::Tangent, Self::LinearOperationCarrier<Self::Tangent>>
            + SupportsZero<Self::Type, Self::Tangent>
            + SupportsAdd<Self::Type, Self::Tangent>,
    {
        let (output, pushforward) = self.linearize(function, primals)?;
        let pullback = pushforward.transpose()?;
        Ok((output, pullback))
    }

    /// Computes the reverse-mode gradient of a scalar-output function.
    ///
    /// This is the canonical user-facing reverse-mode entry point for differentiable domains. The function must return
    /// exactly one rank-0 scalar array leaf.
    #[allow(private_bounds)]
    fn value_and_gradient<'domain, F, Input>(
        &'domain self,
        function: F,
        primal: Input,
    ) -> Result<Input::To<Self::Tangent>, TracingError>
    where
        Self: 'domain,
        Self::Operation: DifferentiableOperation<Self>,
        F: FnOnce(Input::To<LinearizationTracer<'domain, Self>>) -> LinearizationTracer<'domain, Self>,
        Input: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<LinearizationTracer<'domain, Self>> + ParameterizedFamily<Self::Tangent>,
                To<Self::Value> = Input,
                ParameterStructure: Debug + PartialEq,
            >,
        Self::Value: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<LinearizationTracer<'domain, Self>> + ParameterizedFamily<Self::Tangent>,
                To<LinearizationTracer<'domain, Self>> = LinearizationTracer<'domain, Self>,
                To<Self::Value> = Self::Value,
                ParameterStructure: Debug + PartialEq,
            > + 'domain,
        Self::Tangent: One<Self::Type>,
        Self::LinearOperationCarrier<Self::Tangent>: InterpretableOperation<Self::Type, Self::Tangent>
            + LinearOperation<Self::Type, Self::Tangent, Self::LinearOperationCarrier<Self::Tangent>>
            + SupportsZero<Self::Type, Self::Tangent>
            + SupportsAdd<Self::Type, Self::Tangent>,
    {
        let (output, pullback): (
            Self::Value,
            Program<
                Self::Type,
                Self::Tangent,
                Self::LinearOperationCarrier<Self::Tangent>,
                <Self::Value as Parameterized<Self::Value>>::To<Self::Tangent>,
                Input::To<Self::Tangent>,
            >,
        ) = self.vjp(|input| Ok(function(input)), primal)?;
        let seed = <Self::Value as Parameterized<Self::Value>>::To::<Self::Tangent>::from_parameters(
            output.parameter_structure(),
            [<Self::Tangent as One<Self::Type>>::one(output.r#type().as_ref())?],
        )?;
        pullback.interpret(seed)
    }

    /// Converts a staged primal [`Program`] into a staged pushforward linear map.
    ///
    /// This is the reusable IR-level form of forward-mode differentiation. It replays the primal program through JVP
    /// rules once, returning both the primal program output at `input_primals` and a staged [`Program`] over linear
    /// operations that can be replayed later on arbitrary tangent inputs at the same primal point.
    ///
    /// # Parameters
    ///
    ///   - `program`: Staged primal program to linearize.
    ///   - `input_primals`: Concrete primal values aligned with the program's input atoms.
    fn linearize_program<O, Input, Output>(
        &self,
        program: &Program<Self::Type, Self::Constant, O, Input, Output>,
        input_primals: Vec<Self::Value>,
    ) -> Result<
        (
            Output::To<Self::Value>,
            Program<
                Self::Type,
                Self::Tangent,
                Self::LinearOperationCarrier<Self::Tangent>,
                Input::To<Self::Tangent>,
                Output::To<Self::Tangent>,
            >,
        ),
        TracingError,
    >
    where
        O: DifferentiableOperation<Self>,
        Input: Parameterized<Self::Constant, Family: ParameterizedFamily<Self::Tangent>>,
        Output: Parameterized<Self::Constant, Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Tangent>>,
    {
        fn tangent_for_atom<'jvp, D>(
            primal_values: &[Option<D::Value>],
            tangents: &[Option<Tangent<D::Type, Tracer<JvpContext<'jvp, D>>>>],
            atom_id: AtomId,
        ) -> Result<Tangent<D::Type, Tracer<JvpContext<'jvp, D>>>, TracingError>
        where
            D: DifferentiableDomain,
        {
            if let Some(tangent) = &tangents[atom_id.index()] {
                return Ok(tangent.clone());
            }
            // Atoms that are not connected to an input tangent are structurally zero. Carry that as a symbolic
            // `Tangent::Zero` so downstream JVP rules can short-circuit; the linearize loop materializes a concrete
            // zero atom only at the program output boundary.
            let primal = primal_values[atom_id.index()].as_ref().ok_or(TracingError::UnboundAtomId { id: atom_id })?;
            Ok(Tangent::Zero(primal.r#type().into_owned()))
        }

        check_count!("input", input_primals, program.input_ids().len(), TracingError);
        let builder = Rc::new(RefCell::new(ProgramBuilder::<
            Self::Type,
            Self::Tangent,
            Self::LinearOperationCarrier<Self::Tangent>,
        >::new()));
        // Keep every tracer and context that holds a clone of `builder` inside this scope. Only raw output atom IDs
        // escape, making `Rc::try_unwrap(builder)` below a real ownership check instead of depending on manual drops.
        let (output_primal_values, output_tangent_atoms) = {
            let mut primal_values: Vec<Option<Self::Value>> = vec![None; program.atoms().len()];
            let mut tangent_values: Vec<Option<Tangent<Self::Type, Tracer<JvpContext<'_, Self>>>>> =
                vec![None; program.atoms().len()];
            let mut context = JvpContext::new(self, builder.clone());

            // Program inputs become linear-program inputs. Their concrete primal values are kept in parallel so JVP
            // rules can evaluate primal semantics while staging tangent operations.
            for (input_atom, input_primal) in program.input_ids().iter().copied().zip(input_primals.into_iter()) {
                let tangent = context.input(input_primal.r#type().into_owned());
                tangent_values[input_atom.index()] = Some(Tangent::Value(tangent));
                primal_values[input_atom.index()] = Some(input_primal);
            }
            // Constants already have primal values in the original program. Their tangents are derived lazily by
            // `tangent_for_atom` as `Tangent::Zero(type)`, propagating through JVP rules until they meet a non-zero
            // tangent that forces materialization.
            for (atom_index, atom) in program.atoms().iter().enumerate() {
                if let Atom::Constant(value) = atom {
                    primal_values[atom_index] = Some(Differentiable::lift_captured_primal(self, value.clone())?);
                }
            }

            // Replay each primal instruction in JVP form. The rule returns both the concrete primal result and a
            // (possibly symbolic) `Tangent`, which becomes the state for the instruction's output atoms.
            for instruction in program.instructions() {
                let input_duals = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input_atom| {
                        Ok(JvpTracer::new(
                            primal_values[input_atom.index()]
                                .clone()
                                .ok_or(TracingError::UnboundAtomId { id: input_atom })?,
                            tangent_for_atom::<Self>(primal_values.as_slice(), tangent_values.as_slice(), input_atom)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, TracingError>>()?;
                let output_duals = instruction.operation().jvp(&mut context, input_duals.as_slice())?;
                check_count!("output", output_duals, instruction.outputs().len(), TracingError);
                for (output_atom, output_dual) in instruction.outputs().iter().copied().zip(output_duals.into_iter()) {
                    let (primal, tangent) = output_dual.into_parts();
                    primal_values[output_atom.index()] = Some(primal);
                    tangent_values[output_atom.index()] = Some(tangent);
                }
            }

            // Materialize tangents for the requested program outputs and retain the matching primal outputs. The
            // temporary tracers created here must not outlive this scope. A `Tangent::Zero` output is staged as a typed
            // zero constant on the linear builder so the resulting program has a concrete atom for every output.
            let mut output_remaining_uses = vec![0usize; program.atoms().len()];
            for output_atom in program.output_ids().iter().copied() {
                output_remaining_uses[output_atom.index()] += 1;
            }
            let mut output_primal_values = Vec::with_capacity(program.output_ids().len());
            let mut output_tangent_atoms = Vec::with_capacity(program.output_ids().len());
            for output_atom in program.output_ids().iter().copied() {
                let tangent =
                    tangent_for_atom::<Self>(primal_values.as_slice(), tangent_values.as_slice(), output_atom)?;
                let tangent_atom = context.materialize_tangent(tangent)?.atom_id()?;

                let remaining_uses = &mut output_remaining_uses[output_atom.index()];
                debug_assert!(*remaining_uses > 0);
                *remaining_uses -= 1;
                let primal = if *remaining_uses == 0 {
                    primal_values[output_atom.index()].take().ok_or(TracingError::UnboundAtomId { id: output_atom })?
                } else {
                    primal_values[output_atom.index()]
                        .as_ref()
                        .ok_or(TracingError::UnboundAtomId { id: output_atom })?
                        .clone()
                };
                output_primal_values.push(primal);
                output_tangent_atoms.push(tangent_atom);
            }
            (output_primal_values, output_tangent_atoms)
        };
        // At this point all tracing handles are out of scope, so the builder can be recovered and finalized.
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => {
                return Err(TracingError::EscapedProgramBuilder);
            }
        };
        let pushforward = builder
            .build(output_tangent_atoms, program.input_structure().clone(), program.output_structure().clone())?
            .simplified()?;
        Ok((
            Output::To::<Self::Value>::from_parameters(program.output_structure().clone(), output_primal_values)?,
            pushforward,
        ))
    }
}

/// Capability required to execute forward-mode JVP rules.
///
/// [`Differentiable`] is deliberately implemented by both concrete domains and active contexts. Concrete
/// [`DifferentiableDomain`]s use it to run JVP rules over concrete primal values and concrete tangent values. Active
/// [`Context`]s such as [`TracingContext`] and batching contexts use it to run the same JVP rules inside an enclosing
/// transform, where primals and tangents are [`Tracer`] leaves owned by that context.
///
/// This trait is separate from [`Context`] because concrete domains can execute JVP rules without being active
/// builders, and it is separate from [`JvpContext`] because [`JvpContext`] is only one per-pass tangent-program builder
/// frame. The [`Differentiable`] implementation supplies value semantics, constant materialization, captured-value
/// lifting, and the linear operation carrier; [`JvpContext`] borrows such an implementation while staging one tangent
/// program.
pub trait Differentiable {
    /// Type metadata used by primal and tangent values.
    type Type: Type + Parameter;

    /// Primal value type seen by JVP rules.
    type Value: Traceable<Self::Type>;

    /// Tangent value type staged in the active linear program.
    type Tangent: Traceable<Self::Type>;

    /// Captured operation value type that can be lifted into this implementation's primal value type.
    type CapturedValue: Traceable<Self::Type>;

    /// Linear operation carrier specialized to the value representation used by an active transform frame.
    type LinearOperationCarrier<V>: Clone + Operation<Self::Type>
    where
        V: Traceable<Self::Type>;

    /// Returns the canonical zero primal for `type_`.
    fn zero_primal(&self, type_: &Self::Type) -> Result<Self::Value, TracingError>;

    /// Returns the canonical one primal for `type_`.
    fn one_primal(&self, type_: &Self::Type) -> Result<Self::Value, TracingError>;

    /// Returns the canonical zero tangent for `type_`.
    fn zero_tangent(&self, type_: &Self::Type) -> Result<Self::Tangent, TracingError>;

    /// Lifts a captured operation value into the primal value representation used by this implementation.
    fn lift_captured_primal(&self, value: Self::CapturedValue) -> Result<Self::Value, TracingError>;
}

impl<D: DifferentiableDomain> Differentiable for D {
    type Type = <D as Domain>::Type;
    type Value = <D as Domain>::Value;
    type Tangent = <D as DifferentiableDomain>::Tangent;
    type CapturedValue = <D as TracingDomain>::Constant;
    type LinearOperationCarrier<V>
        = <D as DifferentiableDomain>::LinearOperationCarrier<V>
    where
        V: Traceable<D::Type>;

    #[inline]
    fn zero_primal(&self, type_: &Self::Type) -> Result<Self::Value, TracingError> {
        RuntimeDomain::zero(self, type_)
    }

    #[inline]
    fn one_primal(&self, type_: &Self::Type) -> Result<Self::Value, TracingError> {
        RuntimeDomain::one(self, type_)
    }

    #[inline]
    fn zero_tangent(&self, type_: &Self::Type) -> Result<Self::Tangent, TracingError> {
        DifferentiableDomain::linear_domain(self).zero(type_)
    }

    #[inline]
    fn lift_captured_primal(&self, value: Self::CapturedValue) -> Result<Self::Value, TracingError> {
        TracingDomain::lift_constant(self, value)
    }
}

/// Active context capability required for nesting automatic-differentiation transforms.
///
/// A [`DifferentiableContext`] is not a backend domain. It is an already-running transform stack that can create
/// traced primal constants, traced tangent constants, and choose the linear operation carrier used by pushforward and
/// pullback programs over [`Tracer<Self>`] leaves. Ordinary backend tracing implements this through
/// [`TracingContext`], while stackable transforms such as batching implement it by delegating constant materialization
/// to their parent context and reparameterizing the same linear carrier family over their own [`Tracer`] leaves.
pub trait DifferentiableContext:
    Context
    + Differentiable<
        Type = <Self as Context>::Type,
        Value = Tracer<Self>,
        Tangent = Tracer<Self>,
        CapturedValue = <Self as Context>::Value,
    >
{
    /// Executes `function` once through an active linearization context and returns the traced primal output plus a
    /// reusable pushforward program over tangent leaves from this same context.
    fn linearize<'context, F, Input, Output>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (Output, Program<<Self as Context>::Type, Tracer<Self>, LinearOperationCarrier<Self>, Input, Output>),
        TracingError,
    >
    where
        Self: 'context,
        Self::Operation: DifferentiableOperation<Self>,
        F: FnOnce(
            Input::To<Tracer<LinearizationContext<'context, Self, Self>>>,
        ) -> Result<Output::To<Tracer<LinearizationContext<'context, Self, Self>>>, TracingError>,
        Input: Parameterized<
                Tracer<Self>,
                To<Tracer<Self>> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Tracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<
                Tracer<Self>,
                To<Tracer<Self>> = Output,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Tracer<Self>>,
            >,
        Output::To<Tracer<LinearizationContext<'context, Self, Self>>>:
            Parameterized<Tracer<LinearizationContext<'context, Self, Self>>, To<Tracer<Self>> = Output>,
    {
        if primals.parameters().next().is_none() {
            return Err(TracingError::InvalidInputCount { expected: 1, got: 0 });
        }
        let context = self.clone();
        let input_builder = context.builder().clone();
        let validation_context = context.clone();
        linearize_with_context::<Self, Self, F, Input, Output, _>(
            DifferentiableStorage::Owned(Rc::new(context)),
            primals,
            move |input_primal| {
                if Rc::ptr_eq(&input_builder, input_primal.context().builder()) {
                    Ok(())
                } else {
                    Err(validation_context.error(TracingError::MismatchedProgramBuilders))
                }
            },
            function,
        )
    }

    /// Evaluates `function` on already-traced primal values and propagates traced tangent values forward.
    fn jvp<'context, F, Input, Output>(
        &self,
        function: F,
        primals: Input,
        tangents: Input,
    ) -> Result<(Output, Output), TracingError>
    where
        Self: 'context,
        Self::Operation: DifferentiableOperation<Self>,
        LinearOperationCarrier<Self>: InterpretableOperation<<Self as Context>::Type, Tracer<Self>>,
        F: FnOnce(
            Input::To<Tracer<LinearizationContext<'context, Self, Self>>>,
        ) -> Output::To<Tracer<LinearizationContext<'context, Self, Self>>>,
        Input: Parameterized<
                Tracer<Self>,
                To<Tracer<Self>> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Tracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<
                Tracer<Self>,
                To<Tracer<Self>> = Output,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Tracer<Self>>,
            >,
        Output::To<Tracer<LinearizationContext<'context, Self, Self>>>:
            Parameterized<Tracer<LinearizationContext<'context, Self, Self>>, To<Tracer<Self>> = Output>,
    {
        let primal_structure = primals.parameter_structure();
        let tangent_structure = tangents.parameter_structure();
        if primal_structure != tangent_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }
        if primals
            .parameters()
            .chain(tangents.parameters())
            .any(|tracer| !Rc::ptr_eq(self.builder(), tracer.context().builder()))
        {
            return Err(self.error(TracingError::MismatchedProgramBuilders));
        }

        let (primal_output, pushforward) = self.linearize(|input| Ok(function(input)), primals)?;
        let tangent_output = pushforward.interpret(tangents)?;
        Ok((primal_output, tangent_output))
    }

    /// Transposes a linear program whose values are leaves in this active context.
    fn transpose_linear_program<Input, Output, O>(
        &self,
        program: &Program<<Self as Context>::Type, Tracer<Self>, O, Input, Output>,
    ) -> Result<Program<<Self as Context>::Type, Tracer<Self>, O, Output, Input>, TracingError>
    where
        O: LinearOperation<<Self as Context>::Type, Tracer<Self>, O>
            + SupportsZero<<Self as Context>::Type, Tracer<Self>>
            + SupportsAdd<<Self as Context>::Type, Tracer<Self>>,
        Input: Parameterized<Tracer<Self>>,
        Output: Parameterized<Tracer<Self>>,
    {
        let builder = Rc::new(RefCell::new(ProgramBuilder::<<Self as Context>::Type, Tracer<Self>, O>::new()));
        let domain = ProgramTracingDomain::new();
        let mut context = TracingContext::new(&domain, builder);
        context.transpose_with_zero_fn(
            program,
            Some(
                |builder: &mut ProgramBuilder<<Self as Context>::Type, Tracer<Self>, O>,
                 r#type: &<Self as Context>::Type| {
                    Ok(builder.add_constant(Differentiable::zero_tangent(self, r#type)?))
                },
            ),
        )
    }

    /// Returns the traced primal output and a traced pullback program by transposing the active pushforward.
    fn vjp<'context, F, Input, Output>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<
        (Output, Program<<Self as Context>::Type, Tracer<Self>, LinearOperationCarrier<Self>, Output, Input>),
        TracingError,
    >
    where
        Self: 'context,
        Self::Operation: DifferentiableOperation<Self>,
        LinearOperationCarrier<Self>: InterpretableOperation<<Self as Context>::Type, Tracer<Self>>
            + LinearOperation<<Self as Context>::Type, Tracer<Self>, LinearOperationCarrier<Self>>
            + SupportsZero<<Self as Context>::Type, Tracer<Self>>
            + SupportsAdd<<Self as Context>::Type, Tracer<Self>>,
        F: FnOnce(
            Input::To<Tracer<LinearizationContext<'context, Self, Self>>>,
        ) -> Result<Output::To<Tracer<LinearizationContext<'context, Self, Self>>>, TracingError>,
        Input: Parameterized<
                Tracer<Self>,
                To<Tracer<Self>> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Tracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<
                Tracer<Self>,
                To<Tracer<Self>> = Output,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Tracer<Self>>,
            >,
        Output::To<Tracer<LinearizationContext<'context, Self, Self>>>:
            Parameterized<Tracer<LinearizationContext<'context, Self, Self>>, To<Tracer<Self>> = Output>,
    {
        let (output, pushforward) = self.linearize(function, primals)?;
        let pullback = self.transpose_linear_program(&pushforward)?;
        Ok((output, pullback))
    }

    /// Returns the traced scalar output and reverse-mode gradient for `function`.
    ///
    /// This is the active-context counterpart of [`crate::tracing_v2::value_and_grad`]. It uses
    /// [`DifferentiableContext::vjp`] directly, so nested reverse mode composes with any enclosing context that
    /// implements this trait instead of going through a separate tracer dispatch path.
    fn value_and_grad<'context, F, Input>(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Tracer<Self>, Input), TracingError>
    where
        Self: 'context,
        Self::Operation: DifferentiableOperation<Self>,
        LinearOperationCarrier<Self>: InterpretableOperation<<Self as Context>::Type, Tracer<Self>>
            + LinearOperation<<Self as Context>::Type, Tracer<Self>, LinearOperationCarrier<Self>>
            + SupportsZero<<Self as Context>::Type, Tracer<Self>>
            + SupportsAdd<<Self as Context>::Type, Tracer<Self>>,
        F: FnOnce(
            Input::To<Tracer<LinearizationContext<'context, Self, Self>>>,
        ) -> Tracer<LinearizationContext<'context, Self, Self>>,
        Input: Parameterized<
                Tracer<Self>,
                To<Tracer<Self>> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Tracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
    {
        let (output, pullback): (
            Tracer<Self>,
            Program<<Self as Context>::Type, Tracer<Self>, LinearOperationCarrier<Self>, Tracer<Self>, Input>,
        ) = self.vjp(|input| Ok(function(input)), primals)?;
        let seed = Differentiable::one_primal(self, output.r#type().as_ref())?;
        Ok((output, pullback.interpret(seed)?))
    }

    /// Returns the reverse-mode gradient of a traced scalar-output function.
    #[inline]
    fn value_and_gradient<'context, F, Input>(&self, function: F, primals: Input) -> Result<Input, TracingError>
    where
        Self: 'context,
        Self::Operation: DifferentiableOperation<Self>,
        LinearOperationCarrier<Self>: InterpretableOperation<<Self as Context>::Type, Tracer<Self>>
            + LinearOperation<<Self as Context>::Type, Tracer<Self>, LinearOperationCarrier<Self>>
            + SupportsZero<<Self as Context>::Type, Tracer<Self>>
            + SupportsAdd<<Self as Context>::Type, Tracer<Self>>,
        F: FnOnce(
            Input::To<Tracer<LinearizationContext<'context, Self, Self>>>,
        ) -> Tracer<LinearizationContext<'context, Self, Self>>,
        Input: Parameterized<
                Tracer<Self>,
                To<Tracer<Self>> = Input,
                Family: ParameterizedFamily<Tracer<LinearizationContext<'context, Self, Self>>>
                            + ParameterizedFamily<Tracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
    {
        self.value_and_grad(function, primals).map(|(_, gradient)| gradient)
    }
}

impl<C> DifferentiableContext for C where
    C: Context
        + Differentiable<
            Type = <C as Context>::Type,
            Value = Tracer<C>,
            Tangent = Tracer<C>,
            CapturedValue = <C as Context>::Value,
        >
{
}

impl<'domain, D, Capture> Differentiable for TracingContext<'domain, D, Capture>
where
    D: DifferentiableDomain + DifferentiableTracingDomain + RuntimeDomain + 'domain,
    D::Operation: SupportsZero<D::Type, D::Constant> + SupportsOne<D::Type, D::Constant>,
{
    type Type = D::Type;
    type Value = Tracer<TracingContext<'domain, D, Capture>>;
    type Tangent = Tracer<TracingContext<'domain, D, Capture>>;
    type CapturedValue = D::Constant;
    type LinearOperationCarrier<V>
        = D::LinearOperationCarrier<V>
    where
        V: Traceable<D::Type>;

    #[inline]
    fn zero_primal(&self, type_: &Self::Type) -> Result<Self::Value, TracingError> {
        let outputs = self.stage_operation(
            <D::Operation as SupportsZero<D::Type, D::Constant>>::zero_operation(type_.clone()),
            &[] as &[Tracer<TracingContext<'domain, D, Capture>>],
        )?;
        check_count!("output", outputs, 1, TracingError);
        Ok(outputs.into_iter().next().expect("checked above"))
    }

    #[inline]
    fn one_primal(&self, type_: &Self::Type) -> Result<Self::Value, TracingError> {
        let outputs = self.stage_operation(
            <D::Operation as SupportsOne<D::Type, D::Constant>>::one_operation(type_.clone()),
            &[] as &[Tracer<TracingContext<'domain, D, Capture>>],
        )?;
        check_count!("output", outputs, 1, TracingError);
        Ok(outputs.into_iter().next().expect("checked above"))
    }

    #[inline]
    fn zero_tangent(&self, type_: &Self::Type) -> Result<Self::Tangent, TracingError> {
        let outputs = self.stage_operation(
            <D::Operation as SupportsZero<D::Type, D::Constant>>::zero_operation(type_.clone()),
            &[] as &[Tracer<TracingContext<'domain, D, Capture>>],
        )?;
        check_count!("output", outputs, 1, TracingError);
        Ok(outputs.into_iter().next().expect("checked above"))
    }

    #[inline]
    fn lift_captured_primal(&self, value: Self::CapturedValue) -> Result<Self::Value, TracingError> {
        Ok(self.constant(value))
    }
}

/// Operation-level contract for forward-mode Jacobian-Vector Product (JVP) staging.
///
/// A [`DifferentiableOperation`] is keyed by the [`Differentiable`] implementation that supplies the value, type, and
/// linear-operation carrier used while differentiating. Implementors consume
/// [`JvpTracer`] inputs, each carrying a primal value and a tangent atom in the active linear
/// builder, and return traced primal/tangent outputs.
///
/// Primitive rules usually stage tangent operations through [`JvpContext::stage_operation`].
/// Higher-order rules use [`JvpContext::differentiable`] to recurse into nested programs with the same
/// [`Differentiable`] implementation.
pub trait DifferentiableOperation<E: Differentiable>: Operation<E::Type> {
    /// Applies this operation's forward-mode Jacobian-Vector Product (JVP) rule.
    ///
    /// The returned vector must be aligned with this operation's outputs and must carry both the
    /// primal output values and the staged tangent atoms for those outputs.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active JVP context used to stage tangent operations and access the
    ///     [`Differentiable`] implementation.
    ///   - `inputs`: Traced inputs aligned with this operation's inputs.
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
    ) -> Result<Vec<JvpTracer<'jvp, E>>, TracingError>
    where
        E: 'jvp;
}

/// Concrete state threaded through forward-mode JVP rules.
///
/// [`JvpContext`] owns the active linear-program builder where tangent ops are staged. It is itself a
/// [`Context`], so tangent tracers are ordinary [`Tracer`] leaves whose context is this JVP context. JVP rules
/// call [`stage_operation`](Self::stage_operation) to stage tangent ops and
/// [`differentiable`](Self::differentiable) to access primal constants or recursively linearize nested programs.
#[doc(hidden)]
pub struct JvpContext<'domain, E: Differentiable> {
    /// [`Differentiable`] implementation borrowed by this [`JvpContext`] for primal semantics.
    differentiable: &'domain E,

    /// [`ProgramBuilder`] that owns the staged linear [`Program`](crate::tracing::Program) that is currently being
    /// traced.
    builder: Rc<RefCell<ProgramBuilder<E::Type, E::Tangent, LinearOperationCarrier<E>>>>,
}

impl<'domain, E: Differentiable> JvpContext<'domain, E> {
    /// Creates a JVP context that stages into `builder`.
    #[doc(hidden)]
    pub fn new(
        differentiable: &'domain E,
        builder: Rc<RefCell<ProgramBuilder<E::Type, E::Tangent, LinearOperationCarrier<E>>>>,
    ) -> Self {
        Self { differentiable, builder }
    }

    /// Returns the [`Differentiable`] implementation borrowed by this JVP context.
    #[inline]
    pub fn differentiable(&self) -> &'domain E {
        self.differentiable
    }

    /// Materializes a [`Tangent`] into a tracer owned by this JVP context.
    ///
    /// Structural zeros carry only type metadata. When a nested linear program needs an actual
    /// input atom, this method stages the differentiable implementation's canonical zero tangent in the active linear
    /// builder. Non-zero tangents are returned unchanged.
    pub fn materialize_tangent(
        &self,
        tangent: Tangent<E::Type, Tracer<JvpContext<'domain, E>>>,
    ) -> Result<Tracer<JvpContext<'domain, E>>, TracingError> {
        match tangent {
            Tangent::Zero(r#type) => Ok(<Self as Context>::constant(self, self.differentiable.zero_tangent(&r#type)?)),
            Tangent::Value(tracer) => Ok(tracer),
        }
    }
}

impl<'domain, E: Differentiable> Clone for JvpContext<'domain, E> {
    fn clone(&self) -> Self {
        Self { differentiable: self.differentiable, builder: self.builder.clone() }
    }
}

impl<'domain, E: Differentiable> Debug for JvpContext<'domain, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("JvpContext").finish_non_exhaustive()
    }
}

impl<'domain, E: Differentiable> Context for JvpContext<'domain, E> {
    type Type = E::Type;
    type Value = E::Tangent;
    type Operation = LinearOperationCarrier<E>;

    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Value, Self::Operation>>> {
        &self.builder
    }
}

/// Forward-mode JVP tracer carrying both a primal and a [`Tangent`].
///
/// [`JvpTracer`] is the value wrapper primitive operations see while a function is evaluated in JVP mode. The `primal`
/// field carries the usual runtime value, while the `tangent` field carries the directional derivative information
/// flowing alongside it as a [`Tangent`]: either a structural [`Tangent::Zero`] with no atom staged on the linear
/// program, or a concrete [`Tangent::Value`] wrapping a tangent atom. Encoding the [`Tangent`] in the type makes the
/// symbolic-zero state part of the JVP rule contract. Rules pattern-match on the tangent variant, and the [`Tangent`]
/// arithmetic impls in [`crate::differentiation::tangent`] propagate `Zero` short-circuits through `+`, `-`, unary
/// negation, and `.scale(_)` without per-rule bookkeeping.
pub struct JvpTracer<'domain, E: Differentiable> {
    /// The primal value.
    primal: E::Value,

    /// The tangent associated with the primal, possibly structurally zero.
    tangent: Tangent<E::Type, Tracer<JvpContext<'domain, E>>>,
}

impl<'domain, E> Clone for JvpTracer<'domain, E>
where
    E: Differentiable,
{
    #[inline]
    fn clone(&self) -> Self {
        Self { primal: self.primal.clone(), tangent: self.tangent.clone() }
    }
}

impl<'domain, E> Debug for JvpTracer<'domain, E>
where
    E: Differentiable,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JvpTracer")
            .field("primal", &self.primal)
            .field("tangent", &self.tangent)
            .finish()
    }
}

impl<'domain, E: Differentiable> Parameter for JvpTracer<'domain, E> {}

impl<'domain, E> JvpTracer<'domain, E>
where
    E: Differentiable + 'domain,
{
    /// Constructs a [`JvpTracer`] from an explicit primal value and [`Tangent`].
    #[inline]
    pub fn new(primal: E::Value, tangent: Tangent<E::Type, Tracer<JvpContext<'domain, E>>>) -> Self {
        Self { primal, tangent }
    }

    /// Constructs a [`JvpTracer`] with a concrete [`Tangent::Value`] tangent.
    #[inline]
    pub fn from_value(primal: E::Value, tangent_value: Tracer<JvpContext<'domain, E>>) -> Self {
        Self { primal, tangent: Tangent::Value(tangent_value) }
    }

    /// Constructs a [`JvpTracer`] with a structurally-zero [`Tangent::Zero`] tangent carrying the
    /// provided tangent type.
    #[inline]
    pub fn from_zero_tangent(primal: E::Value, tangent_type: E::Type) -> Self {
        Self { primal, tangent: Tangent::Zero(tangent_type) }
    }

    /// Returns the primal value carried by this JVP tracer.
    #[inline]
    pub fn primal(&self) -> &E::Value {
        &self.primal
    }

    /// Returns the tangent carried by this JVP tracer.
    #[inline]
    pub fn tangent(&self) -> &Tangent<E::Type, Tracer<JvpContext<'domain, E>>> {
        &self.tangent
    }

    /// Consumes this JVP tracer and returns its primal and tangent components.
    #[inline]
    pub fn into_parts(self) -> (E::Value, Tangent<E::Type, Tracer<JvpContext<'domain, E>>>) {
        (self.primal, self.tangent)
    }
}

impl<'domain, E: Differentiable> Typed<E::Type> for JvpTracer<'domain, E> {
    #[inline]
    fn r#type(&self) -> Cow<'_, E::Type> {
        self.primal.r#type()
    }
}

impl<'domain, E: Differentiable> Display for JvpTracer<'domain, E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.primal, formatter)
    }
}

impl<'domain, E: Differentiable> Traceable<E::Type> for JvpTracer<'domain, E> {}

impl<'parent, C, D> LinearizationContext<'parent, C, D>
where
    C: Context + 'parent,
    D: Differentiable<Type = C::Type, CapturedValue = C::Value> + 'parent,
{
    /// Creates a new active linearization context from prepared differentiable storage.
    #[inline]
    fn new_with_differentiable(
        differentiable: DifferentiableStorage<'parent, D>,
        primal_builder: Rc<RefCell<ProgramBuilder<C::Type, C::Value, C::Operation>>>,
        linear_builder: Rc<RefCell<ProgramBuilder<C::Type, D::Tangent, LinearOperationCarrier<D>>>>,
    ) -> Self {
        Self {
            differentiable,
            primal_builder,
            linear_builder,
            primal_values: Rc::new(RefCell::new(Vec::new())),
            tangent_atoms: Rc::new(RefCell::new(Vec::new())),
            marker: std::marker::PhantomData,
        }
    }

    /// Registers an input atom together with its concrete primal and matching tangent input atom.
    fn register_input(&self, atom: AtomId, primal: D::Value, tangent: AtomId) {
        self.ensure_atom_capacity(atom);
        self.primal_values.borrow_mut()[atom.index()] = Some(primal);
        self.tangent_atoms.borrow_mut()[atom.index()] = Some(tangent);
    }

    /// Ensures that all per-atom state tables can address `atom`.
    fn ensure_atom_capacity(&self, atom: AtomId) {
        let capacity = atom.index() + 1;
        {
            let mut primals = self.primal_values.borrow_mut();
            if primals.len() < capacity {
                primals.resize_with(capacity, || None);
            }
        }
        let mut tangents = self.tangent_atoms.borrow_mut();
        if tangents.len() < capacity {
            tangents.resize_with(capacity, || None);
        }
    }

    /// Returns the stored primal for `atom`, lazily registering primal constants when needed.
    fn primal_for_atom(&self, atom: AtomId) -> Result<D::Value, TracingError> {
        self.ensure_atom_capacity(atom);
        if let Some(primal) = &self.primal_values.borrow()[atom.index()] {
            return Ok(primal.clone());
        }
        let constant = {
            let builder = self.primal_builder.borrow();
            match builder.atoms().get(atom.index()) {
                Some(Atom::Constant(value)) => Some(value.clone()),
                Some(Atom::Variable(_)) => None,
                None => return Err(TracingError::UnboundAtomId { id: atom }),
            }
        };
        let Some(constant) = constant else {
            return Err(TracingError::UnboundAtomId { id: atom });
        };
        let primal = self.differentiable.as_ref().lift_captured_primal(constant)?;
        self.primal_values.borrow_mut()[atom.index()] = Some(primal.clone());
        Ok(primal)
    }

    /// Returns the stored tangent for `atom`, materialized as a tracer in `context`.
    fn tangent_for_atom<'jvp>(
        &self,
        context: &JvpContext<'jvp, D>,
        atom: AtomId,
    ) -> Result<Tangent<C::Type, Tracer<JvpContext<'jvp, D>>>, TracingError>
    where
        D: 'jvp,
    {
        self.ensure_atom_capacity(atom);
        if let Some(tangent_atom) = self.tangent_atoms.borrow()[atom.index()] {
            return Ok(Tangent::Value(context.tracer(tangent_atom, None)));
        }
        Ok(Tangent::Zero(self.primal_for_atom(atom)?.r#type().into_owned()))
    }

    /// Collects concrete primal outputs and linear-program output atom ids.
    fn collect_outputs(&self, output_atoms: &[AtomId]) -> Result<(Vec<D::Value>, Vec<AtomId>), TracingError> {
        let context = JvpContext::new(self.differentiable.as_ref(), self.linear_builder.clone());
        let mut output_primals = Vec::with_capacity(output_atoms.len());
        let mut output_tangents = Vec::with_capacity(output_atoms.len());
        for output_atom in output_atoms.iter().copied() {
            let primal = self.primal_for_atom(output_atom)?;
            let tangent = self.tangent_for_atom(&context, output_atom)?;
            let tangent_atom = context.materialize_tangent(tangent)?.atom_id()?;
            output_primals.push(primal);
            output_tangents.push(tangent_atom);
        }
        Ok((output_primals, output_tangents))
    }
}

impl<'domain, C, D> Clone for LinearizationContext<'domain, C, D>
where
    C: Context + 'domain,
    D: Differentiable<Type = C::Type, CapturedValue = C::Value> + 'domain,
{
    fn clone(&self) -> Self {
        Self {
            differentiable: self.differentiable.clone(),
            primal_builder: self.primal_builder.clone(),
            linear_builder: self.linear_builder.clone(),
            primal_values: self.primal_values.clone(),
            tangent_atoms: self.tangent_atoms.clone(),
            marker: std::marker::PhantomData,
        }
    }
}

impl<'domain, C, D, Capture> CaptureContext<Capture> for LinearizationContext<'domain, C, D>
where
    C: Context + 'domain,
    D: CaptureContext<Capture, Type = C::Type, Value = C::Value>
        + Differentiable<Type = C::Type, CapturedValue = C::Value>
        + 'domain,
    LinearizationContext<'domain, C, D>: Context<Type = C::Type, Value = C::Value>,
    Capture: Traceable<C::Type>,
{
    #[inline]
    fn capture_value(&self, capture: Capture) -> Result<Self::Value, TracingError> {
        self.differentiable.as_ref().capture_value(capture)
    }
}

impl<'parent, C, D> Context for LinearizationContext<'parent, C, D>
where
    C: Context + 'parent,
    D: Differentiable<Type = C::Type, CapturedValue = C::Value> + 'parent,
    C::Operation: DifferentiableOperation<D>,
{
    type Type = C::Type;
    type Value = C::Value;
    type Operation = C::Operation;

    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Value, Self::Operation>>> {
        &self.primal_builder
    }

    fn stage_operation<I: std::borrow::Borrow<Tracer<Self>>>(
        &self,
        operation: Self::Operation,
        inputs: &[I],
    ) -> Result<Vec<Tracer<Self>>, TracingError> {
        if inputs.iter().any(|input| !Rc::ptr_eq(self.builder(), input.borrow().context().builder())) {
            return Err(self.error(TracingError::MismatchedProgramBuilders));
        }
        if self.builder().borrow().error().is_some() {
            let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice())?;
            return Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(TracerState::Poison, r#type, self.clone()))
                .collect());
        }

        let input_atoms = inputs
            .iter()
            .map(|input| input.borrow().atom_id())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| self.error(error))?;
        let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
        let output_types = operation.infer_output_types(input_types.as_slice())?;
        let mut jvp_context = JvpContext::new(self.differentiable.as_ref(), self.linear_builder.clone());
        let input_duals = input_atoms
            .iter()
            .copied()
            .map(|atom| Ok(JvpTracer::new(self.primal_for_atom(atom)?, self.tangent_for_atom(&jvp_context, atom)?)))
            .collect::<Result<Vec<_>, TracingError>>()?;
        let output_duals = operation.jvp(&mut jvp_context, input_duals.as_slice())?;
        check_count!("output", output_duals, output_types.len(), TracingError);

        let mut output_tracers = Vec::with_capacity(output_duals.len());
        let mut primal_builder = self.builder().borrow_mut();
        for (output_dual, output_type) in output_duals.into_iter().zip(output_types.into_iter()) {
            let (primal, tangent) = output_dual.into_parts();
            let atom = primal_builder.add_variable(output_type.clone());
            self.ensure_atom_capacity(atom);
            self.primal_values.borrow_mut()[atom.index()] = Some(primal);
            self.tangent_atoms.borrow_mut()[atom.index()] = match tangent {
                Tangent::Zero(_) => None,
                Tangent::Value(tracer) => Some(tracer.atom_id()?),
            };
            output_tracers.push(self.tracer(atom, Some(output_type)));
        }
        Ok(output_tracers)
    }
}

/// Optional extension for tracing domains that support differentiation inside an active trace.
///
/// Backends usually do not implement this trait directly. Implement [`LinearizableDomain`] instead. Core derives this
/// marker once the backend's primal carrier can synthesize primal zeros and ones and the backend has selected a linear
/// carrier family through [`LinearizableDomain::LinearOperationCarrier`].
pub trait DifferentiableTracingDomain:
    TracingDomain<
    Operation: SupportsAdd<Self::Type, Self::Constant>
                   + SupportsZero<Self::Type, Self::Constant>
                   + SupportsOne<Self::Type, Self::Constant>,
>
{
}

impl<D> DifferentiableTracingDomain for D where
    D: LinearizableDomain<
        Operation: SupportsAdd<D::Type, D::Constant>
                       + SupportsZero<D::Type, D::Constant>
                       + SupportsOne<D::Type, D::Constant>,
    >
{
}

impl<V> LinearizableDomain for ScalarDomain<V>
where
    V: Traceable<DataType>,
    ScalarDomain<V>: RuntimeDomain<Type = DataType> + TracingDomain<Type = DataType, Operation: Clone>,
    LinearScalarDomain<V>: RuntimeDomain<Type = DataType, Value = V>,
    LinearScalarDomain<V>: TracingDomain<Type = DataType, Value = V, Operation = LinearScalarOperation<V>>,
{
    type Tangent = V;
    type LinearOperationCarrier<W>
        = LinearScalarOperation<W>
    where
        W: Traceable<DataType>;
    type LinearDomain = LinearScalarDomain<V>;

    #[inline]
    fn linear_domain(&self) -> &Self::LinearDomain {
        ScalarDomain::linear_domain(self)
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::differentiation::Tangent;
    use crate::operations::constants::{One, Zero, ZeroLike};
    use crate::tracing::domains::ScalarDomain;
    use crate::types::{ArrayType, DataType, Shape, Size, Typed};

    use super::DifferentiableDomain;

    #[test]
    fn test_tangent_value_carries_symbolic_zero_or_value_tangent() {
        let zero = Tangent::<DataType, f64>::zero(DataType::F64);
        let value = Tangent::<DataType, f64>::value(2.5);

        assert!(zero.is_zero());
        assert_eq!(zero.as_value(), None);
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(zero.to_string(), "Zero[f64]");
        assert_eq!(<Tangent<DataType, f64> as Zero<DataType>>::zero(&DataType::F64), Ok(zero.clone()));
        assert_eq!(value.as_value(), Some(&2.5));
        assert_eq!(value.r#type().into_owned(), DataType::F64);
        assert_eq!(value.to_string(), "2.5");
        assert_eq!(<Tangent<DataType, f64> as One<DataType>>::one(&DataType::F64), Ok(Tangent::value(1.0)));
        assert_eq!(value.zero_like(), zero);

        let zero_only = Tangent::<DataType, Infallible>::zero(DataType::I32);
        assert_eq!(zero_only.r#type().into_owned(), DataType::I32);
        assert_eq!(zero_only.to_string(), "Zero[i32]");
        assert_eq!(<Tangent<DataType, Infallible> as Zero<DataType>>::zero(&DataType::I32), Ok(zero_only.clone()));
        assert_eq!(zero_only.zero_like(), zero_only);

        let array_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)]), None, None).unwrap();
        let array_tangent = Tangent::<ArrayType, Infallible>::zero(array_type.clone());
        assert_eq!(array_tangent.r#type().into_owned(), array_type);
    }

    #[test]
    fn test_scalar_domain_half_and_float_domains_are_differentiable() {
        let _: Option<<ScalarDomain<bf16> as DifferentiableDomain>::LinearOperationCarrier<bf16>> = None;
        let _: Option<<ScalarDomain<f16> as DifferentiableDomain>::LinearOperationCarrier<f16>> = None;
        let _: Option<<ScalarDomain<f32> as DifferentiableDomain>::LinearOperationCarrier<f32>> = None;
        let _: Option<<ScalarDomain<f64> as DifferentiableDomain>::LinearOperationCarrier<f64>> = None;
    }

    #[test]
    fn test_scalar_domain_half_domains_run_jvp() {
        let bf16_domain = ScalarDomain::<bf16>::new();
        assert_eq!(
            bf16_domain.jvp(|x| x.clone() + x, bf16::from_f32(3.0), bf16::ONE),
            Ok((bf16::from_f32(6.0), bf16::from_f32(2.0)))
        );

        let f16_domain = ScalarDomain::<f16>::new();
        assert_eq!(
            f16_domain.jvp(|x| x.clone() + x, f16::from_f32(3.0), f16::ONE),
            Ok((f16::from_f32(6.0), f16::from_f32(2.0)))
        );
    }
}
