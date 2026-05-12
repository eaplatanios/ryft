use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::Display;
use std::rc::Rc;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::differentiation::LinearOperation;
use crate::macros::check_count;
use crate::operations::arithmetic::{AddOperation, SupportsAdd, SupportsMul, SupportsNeg, SupportsScale, SupportsSub};
use crate::operations::constants::{SupportsOneLike, SupportsZero, SupportsZeroLike};
use crate::operations::scalars::LinearScalarOperation;
use crate::operations::{InterpretableOperation, Operation};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::tracing::domains::{
    Domain, LinearScalarDomain, RuntimeDomain, ScalarDomain, Tracer, TracingContext, TracingDomain,
};
use crate::tracing::{Atom, AtomId, Instruction, Program, ProgramBuilder, Traceable, TracingError};
use crate::tracing_v2::forward::JvpDispatch;
use crate::tracing_v2::operations::{
    LinearArrayOperation, NoOperationExtension, SupportsMatMul, SupportsMatrixTranspose, SupportsReshape,
};
use crate::types::{ArrayType, DataType, Type, Typed};

/// Errors emitted by the differentiation helpers in [`crate::tracing_v2`].
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Reverse-mode gradient was requested for a function with an invalid number of output leaves.
    #[error("gradient output must have exactly {expected} leaf but got {got}")]
    InvalidGradientOutputLeafCount { expected: usize, got: usize },

    /// Reverse-mode gradient was requested for a non-scalar array output.
    #[error("gradient output must be a rank-0 scalar array but got {output_type}")]
    NonScalarGradientOutput { output_type: ArrayType },

    /// Traced forward-mode differentiation was invoked without any staged input leaves.
    #[error("traced jvp requires at least one input leaf to recover the tracing context")]
    MissingTracedJvpInputLeaves,

    /// Traced reverse-mode differentiation was invoked without any staged input leaves.
    #[error("traced reverse-mode requires at least one input leaf to recover the tracing context")]
    MissingTracedReverseModeInputLeaves,
}

/// Value-level contract for leaves that participate in automatic differentiation over `T`.
///
/// The associated [`Tangent`](Self::Tangent) type makes the tangent representation explicit.
/// [`Differentiable::zero_tangent`] returns the canonical zero tangent/exemplar for this concrete primal value,
/// carrying whatever metadata the tangent representation needs. Transform code uses this hook when it must synthesize
/// disconnected tangents from primal values instead of guessing from abstract [`Type`] metadata alone.
pub trait Differentiable<T: Type>: Traceable<T> {
    /// Tangent and cotangent leaf type associated with this primal leaf.
    type Tangent: Traceable<T>;

    /// Returns the canonical zero tangent/exemplar associated with this primal value.
    fn zero_tangent(&self) -> Result<Self::Tangent, TracingError>;
}

impl<'domain, D: RuntimeDomain<Value: Differentiable<D::Type>> + TracingDomain> Differentiable<D::Type>
    for Tracer<'domain, D>
{
    type Tangent = Self;

    #[inline]
    fn zero_tangent(&self) -> Result<Self::Tangent, TracingError> {
        self.context.zero(self.r#type().as_ref())
    }
}

/// Ensures every tracer belongs to `context`.
pub(crate) fn ensure_tracers_belong_to_context<'domain, D: TracingDomain>(
    context: &TracingContext<'domain, D>,
    tracers: &[Tracer<'domain, D>],
) -> Result<(), TracingError> {
    if tracers.iter().any(|tracer| !Rc::ptr_eq(&context.builder, &tracer.context.builder)) {
        return Err(context.error(TracingError::MismatchedProgramBuilders));
    }
    Ok(())
}

/// Domain capability required for automatic-differentiation transforms that linearize staged programs.
///
/// This is the only backend-specific domain fact that `ryft-core` cannot infer from [`RuntimeDomain`] and
/// [`TracingDomain`]. Once a backend selects a linear domain, core derives the tangent leaf type from
/// [`Domain::Value`] on that linear domain, derives the tangent operation carrier from
/// [`TracingDomain::OperationCarrier`] on that linear domain, and uses the backend's ordinary tracing carrier for
/// differentiable primal programs.
///
/// A linearizable domain's selected linear carrier must itself be a [`LinearOperation`] carrier. That invariant lives
/// here, instead of only on the blanket [`DifferentiableDomain`] implementation, so implementing this trait is a
/// complete statement that programs over the domain can be linearized.
pub trait LinearizableDomain: RuntimeDomain + TracingDomain + Sized {
    /// Tracing domain selected by this domain for tangent and cotangent programs.
    type LinearDomain: TracingDomain<
            Type = Self::Type,
            OperationCarrier: Clone
                                  + InterpretableOperation<Self::Type, <Self::LinearDomain as Domain>::Value>
                                  + LinearOperation<
                Self::Type,
                <Self::LinearDomain as Domain>::Value,
                <Self::LinearDomain as TracingDomain>::OperationCarrier,
            > + SupportsZero<Self::Type, <Self::LinearDomain as Domain>::Value>
                                  + SupportsAdd<Self::Type, <Self::LinearDomain as Domain>::Value>,
        >;

    /// Returns the linearizable domain used for tangent and cotangent programs.
    fn linear_domain(&self) -> &Self::LinearDomain;
}

/// Type-level family for linear operation carriers that can be reparameterized over a new value type.
///
/// [`LinearizableDomain`] identifies the linear domain used for concrete tangent and cotangent programs. For nested
/// differentiation inside an active trace, `ryft-core` also needs the same linear operation family specialized to
/// [`Tracer`] leaves. Rust cannot derive that specialization from an arbitrary carrier type:
/// `LinearArrayOperation<Array<_>, ArrayType>` does not intrinsically tell the compiler that the traced carrier is
/// `LinearArrayOperation<Tracer<_>, ArrayType>`.
///
/// This trait records that relationship for reusable carrier families. It is implemented once for the core scalar and
/// array linear carriers, and backends that reuse those carriers inherit the traced-AD support automatically. There is
/// intentionally no paired `ForValue` associated type in this trait: the implementing type already is the carrier
/// specialized for the concrete value type `V`; [`ForTracer`](Self::ForTracer) names only the extra specialization that
/// cannot be recovered from `Self`.
pub trait LinearOperationCarrierFamily<D: TracingDomain, V: Traceable<D::Type>> {
    /// Same linear carrier family specialized to operate on traced leaves for `D`.
    type ForTracer<'domain>: Clone
        + InterpretableOperation<D::Type, Tracer<'domain, D>>
        + LinearOperation<D::Type, Tracer<'domain, D>, Self::ForTracer<'domain>>
        + SupportsZero<D::Type, Tracer<'domain, D>>
        + SupportsNeg<D::Type, Tracer<'domain, D>>
        + SupportsAdd<D::Type, Tracer<'domain, D>>
        + SupportsScale<D::Type, Tracer<'domain, D>>
    where
        D: 'domain;
}

/// Type-level family for the backend-owned extension portion of a linear operation carrier.
///
/// [`LinearOperationCarrierFamily`] can reparameterize the built-in carrier shell from concrete tangent values to
/// traced tangent values. This companion trait tells it how to reparameterize only the extension enum inside that
/// shell. For a backend extension such as `LinearBackendOperation<V>`, the implementation usually maps
/// `LinearBackendOperation<D::Tangent>` to `LinearBackendOperation<Tracer<'domain, D>>`.
///
/// The no-extension carrier uses [`NoOperationExtension`], whose reparameterization is itself. Backends that do not
/// add linear operations do not need to implement this trait.
pub trait LinearOperationExtensionFamily<D: TracingDomain<Type = ArrayType>, V: Traceable<ArrayType>>: Clone {
    /// Same extension family specialized to operate on traced leaves for `D`.
    type ForTracer<'domain>: Clone
        + InterpretableOperation<ArrayType, Tracer<'domain, D>>
        + LinearOperation<
            ArrayType,
            Tracer<'domain, D>,
            LinearArrayOperation<Tracer<'domain, D>, ArrayType, Self::ForTracer<'domain>>,
        >
    where
        D: 'domain;
}

impl<D, V> LinearOperationExtensionFamily<D, V> for NoOperationExtension
where
    D: TracingDomain<Type = ArrayType>,
    V: Traceable<ArrayType>,
{
    type ForTracer<'domain>
        = NoOperationExtension
    where
        D: 'domain;
}

impl<D, V> LinearOperationCarrierFamily<D, V> for LinearScalarOperation<V>
where
    D: TracingDomain<Type = DataType>,
    V: Traceable<DataType>,
    D::OperationCarrier: SupportsAdd<DataType, D::Value>
        + SupportsSub<DataType, D::Value>
        + SupportsNeg<DataType, D::Value>
        + SupportsMul<DataType, D::Value>
        + SupportsZeroLike<DataType, D::Value>
        + SupportsOneLike<DataType, D::Value>,
{
    type ForTracer<'domain>
        = LinearScalarOperation<Tracer<'domain, D>>
    where
        D: 'domain;
}

impl<D, V, Extension> LinearOperationCarrierFamily<D, V> for LinearArrayOperation<V, ArrayType, Extension>
where
    D: TracingDomain<Type = ArrayType>,
    V: Traceable<ArrayType>,
    Extension: LinearOperationExtensionFamily<D, V>,
    D::OperationCarrier: SupportsAdd<ArrayType, D::Value>
        + SupportsSub<ArrayType, D::Value>
        + SupportsNeg<ArrayType, D::Value>
        + SupportsMul<ArrayType, D::Value>
        + SupportsZeroLike<ArrayType, D::Value>
        + SupportsOneLike<ArrayType, D::Value>
        + SupportsMatMul<ArrayType, D::Value>
        + SupportsMatrixTranspose<ArrayType, D::Value>
        + SupportsReshape<ArrayType, D::Value>,
{
    type ForTracer<'domain>
        = LinearArrayOperation<Tracer<'domain, D>, ArrayType, Extension::ForTracer<'domain>>
    where
        D: 'domain;
}

/// Extension of [`RuntimeDomain`] for backends that support automatic differentiation.
///
/// Backends that only need ordinary tracing implement [`TracingDomain`] without this extension. AD
/// transforms such as [`DifferentiableDomain::jvp`], [`DifferentiableDomain::grad`], and
/// [`vjp`](crate::tracing_v2::vjp) require this trait so non-differentiable backends do not need to
/// define fake tangent carriers.
///
/// Backends usually do not implement this trait directly. Implement [`LinearizableDomain`] instead and let the
/// blanket implementation compose the full AD API in `ryft-core`.
///
/// Differentiated closures are traced with the domain's ordinary [`TracingDomain::OperationCarrier`]. Individual
/// transforms that linearize a staged primal program require that carrier to implement [`DifferentiableOperation`] for
/// the active domain, so backends do not need a second operation-carrier API just for AD.
pub trait DifferentiableDomain:
    RuntimeDomain + TracingDomain<OperationCarrier: Clone + InterpretableOperation<Self::Type, Self::Value>> + Sized
{
    /// Tangent and cotangent leaf type selected by this differentiable domain.
    type Tangent: Traceable<Self::Type>;

    /// Tracing domain selected by this differentiable domain for tangent and cotangent programs.
    type LinearDomain: TracingDomain<Type = Self::Type, Value = Self::Tangent, OperationCarrier = Self::LinearOperationCarrier>;

    /// Operation carrier selected by [`DifferentiableDomain::LinearDomain`] for tangent and cotangent programs.
    type LinearOperationCarrier: Clone
        + InterpretableOperation<Self::Type, Self::Tangent>
        + LinearOperation<Self::Type, Self::Tangent, Self::LinearOperationCarrier>
        + SupportsZero<Self::Type, Self::Tangent>
        + SupportsAdd<Self::Type, Self::Tangent>;

    /// Returns the linearizable domain used for tangent and cotangent programs.
    fn linear_domain(&self) -> &Self::LinearDomain;

    /// Evaluates `function` on `primals` and propagates the supplied tangent values forward.
    ///
    /// The returned pair is `(primal_output, tangent_output)`. This is the canonical user-facing forward-mode
    /// Jacobian-Vector Product (JVP) entry point for differentiable domains.
    #[allow(private_bounds)]
    fn jvp<
        'domain,
        F: FnOnce(Leaf::FunctionInput) -> Leaf::FunctionOutput,
        Input: Parameterized<Leaf, ParameterStructure: std::fmt::Debug + PartialEq>,
        Output: Parameterized<Leaf>,
        Leaf: JvpDispatch<'domain, Self, Input, Output, Marker>,
        Marker,
    >(
        &'domain self,
        function: F,
        primals: Input,
        tangents: Input::To<Leaf::Tangent>,
    ) -> Result<(Output, Output::To<Leaf::Tangent>), TracingError>
    where
        Input::Family: ParameterizedFamily<Leaf::Tangent>,
        Output::Family: ParameterizedFamily<Leaf::Tangent>,
    {
        crate::tracing_v2::forward::jvp_at(self, function, primals, tangents)
    }

    /// Computes the reverse-mode gradient of a scalar-output function.
    ///
    /// This is the canonical user-facing reverse-mode entry point for differentiable domains. The function must return
    /// exactly one rank-0 scalar array leaf.
    #[allow(private_bounds, private_interfaces)]
    fn grad<
        'domain,
        F,
        Input: Parameterized<Leaf, ParameterStructure: std::fmt::Debug + PartialEq>,
        Leaf: crate::tracing_v2::linear::ValueAndGradDispatch<Self, Input, Marker>,
        Marker,
    >(
        &'domain self,
        function: F,
        primals: Input,
    ) -> Result<<Leaf as crate::tracing_v2::linear::ValueAndGradDispatch<Self, Input, Marker>>::Gradient, TracingError>
    where
        F: FnOnce(
            <Leaf as crate::tracing_v2::linear::ValueAndGradDispatch<Self, Input, Marker>>::FunctionInput<'domain>,
        )
            -> <Leaf as crate::tracing_v2::linear::ValueAndGradDispatch<Self, Input, Marker>>::FunctionOutput<
            'domain,
        >,
    {
        Leaf::invoke(self, function, primals).map(|(_, gradient)| gradient)
    }

    /// Materializes a structured [`Jacobian`] using forward-mode differentiation.
    ///
    /// The returned [`Jacobian`] is a nested [`Parameterized`] value whose outer family mirrors
    /// the function's output and whose inner family mirrors its input. Each innermost leaf is a
    /// [`DifferentialBlock`](crate::tracing_v2::linear::DifferentialBlock) holding the partial
    /// derivatives of one output leaf with respect to one input leaf.
    #[allow(private_bounds)]
    fn jacfwd<'domain, F, Input, Output, V>(
        &'domain self,
        function: F,
        primals: Input,
    ) -> Result<Jacobian<Input, Output, V>, TracingError>
    where
        Self: DifferentiableDomain<Type = ArrayType, Value = V> + 'static,
        V: crate::tracing_v2::linear::CoordinateValue + Differentiable<ArrayType, Tangent = Self::Tangent> + 'domain,
        Self::Tangent: crate::tracing_v2::linear::CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        Output: Parameterized<V, To<V> = Output, ParameterStructure: PartialEq>,
        Input::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<crate::tracing_v2::batching::ReferenceBatch<Self::Tangent>>
            + ParameterizedFamily<Tracer<'domain, Self>>
            + ParameterizedFamily<crate::tracing_v2::linear::DifferentialBlock<V::Coordinate>>,
        Output::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<crate::tracing_v2::batching::ReferenceBatch<Self::Tangent>>
            + ParameterizedFamily<Tracer<'domain, Self>>
            + ParameterizedFamily<
                crate::tracing_v2::linear::DifferentialRow<
                    Input::To<crate::tracing_v2::linear::DifferentialBlock<V::Coordinate>>,
                    V::Coordinate,
                >,
            >,
        Output::To<Tracer<'domain, Self>>: Parameterized<Tracer<'domain, Self>, To<V> = Output>,
        F: FnOnce(Input::To<Tracer<'domain, Self>>) -> Result<Output::To<Tracer<'domain, Self>>, TracingError>,
        Self::OperationCarrier: DifferentiableOperation<Self>,
    {
        let input_structure = primals.parameter_structure();
        let input_parameters = primals.into_parameters().collect::<Vec<_>>();
        let primals = Input::from_parameters(input_structure.clone(), input_parameters.clone())?;
        let (output, pushforward) =
            crate::tracing_v2::linear::linearize::<Self, F, Input, Output, V>(self, function, primals)?;
        crate::tracing_v2::linear::materialize_differential_from_pushforward::<Self, Input, Output, V>(
            input_structure,
            input_parameters,
            output,
            pushforward,
        )
    }

    /// Materializes a structured [`Hessian`] of a scalar-output function.
    ///
    /// Hessian evaluation is expressed internally as a forward-mode [`Jacobian`] over a
    /// reverse-mode gradient transform.
    #[allow(private_bounds)]
    fn hessian<'domain, F, Input, V>(
        &'domain self,
        function: F,
        primals: Input,
    ) -> Result<Hessian<Input, V>, TracingError>
    where
        Self: DifferentiableDomain<Type = ArrayType, Value = V>
            + DifferentiableTracingDomain<Type = ArrayType, Value = V>
            + 'static,
        V: crate::tracing_v2::linear::CoordinateValue + Differentiable<ArrayType, Tangent = Self::Tangent> + 'domain,
        Self::Tangent: crate::tracing_v2::linear::CoordinateValue<Coordinate = V::Coordinate>,
        Input: Parameterized<V, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        Input::Family: ParameterizedFamily<Self::Tangent>
            + ParameterizedFamily<crate::tracing_v2::batching::ReferenceBatch<Self::Tangent>>
            + ParameterizedFamily<Tracer<'domain, Self>>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<V>
            + ParameterizedFamily<crate::tracing_v2::linear::DifferentialBlock<V::Coordinate>>
            + ParameterizedFamily<
                crate::tracing_v2::linear::DifferentialRow<
                    Input::To<crate::tracing_v2::linear::DifferentialBlock<V::Coordinate>>,
                    V::Coordinate,
                >,
            >,
        Input::To<ArrayType>: Parameterized<ArrayType, To<Tracer<'domain, Self>> = Input::To<Tracer<'domain, Self>>>,
        Input::To<Tracer<'domain, Self>>:
            Parameterized<Tracer<'domain, Self>, To<V> = Input, ParameterStructure: std::fmt::Debug + PartialEq>,
        <Input::To<Tracer<'domain, Self>> as Parameterized<Tracer<'domain, Self>>>::To<ArrayType>:
            Parameterized<ArrayType, To<Tracer<'domain, Self>> = Input::To<Tracer<'domain, Self>>>,
        F: FnOnce(Input::To<Tracer<'domain, Self>>) -> Tracer<'domain, Self>,
        Self::OperationCarrier: Clone
            + InterpretableOperation<ArrayType, V>
            + DifferentiableOperation<TracingContext<'domain, Self>>
            + DifferentiableOperation<Self>
            + SupportsZeroLike<ArrayType, V>
            + SupportsAdd<ArrayType, V>
            + 'static,
        AddOperation: InterpretableOperation<ArrayType, Tracer<'domain, Self>>,
    {
        let input_structure = primals.parameter_structure();
        let input_parameters = primals.into_parameters().collect::<Vec<_>>();
        let primals = Input::from_parameters(input_structure.clone(), input_parameters.iter().cloned())?;
        let (gradient, gradient_program): (Input, Program<ArrayType, V, Self::OperationCarrier, Input, Input>) = self
            .interpret_and_trace(
            |input: Input::To<Tracer<'domain, Self>>| {
                let (_, gradient) = <Tracer<'domain, Self> as crate::tracing_v2::linear::ValueAndGradDispatch<
                    Self,
                    Input::To<Tracer<'domain, Self>>,
                    crate::tracing_v2::linear::TracedValueAndGrad,
                >>::invoke(self, function, input)?;
                Ok(gradient)
            },
            primals,
        )?;
        let pushforward = gradient_program.linearize(self, input_parameters.clone())?;
        crate::tracing_v2::linear::materialize_differential_from_pushforward::<Self, Input, Input, V>(
            input_structure,
            input_parameters,
            gradient,
            pushforward,
        )
    }
}

/// Structured forward- or reverse-mode Jacobian of a function `Input -> Output` over leaf value
/// type `V`. Materialized by [`DifferentiableDomain::jacfwd`] and [`crate::tracing_v2::jacrev`].
///
/// The outer [`Parameterized`] family mirrors the function's output; each output-leaf position
/// holds a [`DifferentialRow`](crate::tracing_v2::linear::DifferentialRow) whose internal family
/// mirrors the function's input and whose leaves are
/// [`DifferentialBlock`](crate::tracing_v2::linear::DifferentialBlock)s of partial derivatives.
/// Block entries are stored as `V::Coordinate` scalars.
pub type Jacobian<Input, Output, V> = crate::tracing_v2::linear::Differential<
    <Output as Parameterized<V>>::To<
        crate::tracing_v2::linear::DifferentialRow<
            <Input as Parameterized<V>>::To<
                crate::tracing_v2::linear::DifferentialBlock<
                    <V as crate::tracing_v2::linear::CoordinateValue>::Coordinate,
                >,
            >,
            <V as crate::tracing_v2::linear::CoordinateValue>::Coordinate,
        >,
    >,
    <Input as Parameterized<V>>::To<
        crate::tracing_v2::linear::DifferentialBlock<<V as crate::tracing_v2::linear::CoordinateValue>::Coordinate>,
    >,
    <V as crate::tracing_v2::linear::CoordinateValue>::Coordinate,
>;

/// Structured Hessian of a scalar-output function over a [`Parameterized`] input with leaf value
/// type `V`. Materialized by [`DifferentiableDomain::hessian`].
///
/// Equivalent to a [`Jacobian<Input, Input, V>`] — both the outer and inner [`Parameterized`]
/// families mirror the input.
pub type Hessian<Input, V> = Jacobian<Input, Input, V>;

impl<D> DifferentiableDomain for D
where
    D: LinearizableDomain,
    D::Value: Differentiable<D::Type, Tangent = <<D as LinearizableDomain>::LinearDomain as Domain>::Value>,
    D::OperationCarrier: Clone + InterpretableOperation<D::Type, D::Value>,
{
    type Tangent = <<D as LinearizableDomain>::LinearDomain as Domain>::Value;
    type LinearDomain = <D as LinearizableDomain>::LinearDomain;
    type LinearOperationCarrier = <<D as LinearizableDomain>::LinearDomain as TracingDomain>::OperationCarrier;
    #[inline]
    fn linear_domain(&self) -> &Self::LinearDomain {
        LinearizableDomain::linear_domain(self)
    }
}

/// Operation-level contract for forward-mode Jacobian-Vector Product (JVP) staging.
///
/// A [`DifferentiableOperation`] is keyed by the [`DifferentiableDomain`] that supplies the value,
/// type, and linear-operation families used while differentiating. Implementors consume
/// [`JvpTracer`] inputs, each carrying a primal value and a tangent atom in the active linear
/// builder, and return traced primal/tangent outputs.
///
/// Primitive rules usually stage tangent operations through [`JvpContext::stage`].
/// Higher-order rules use [`JvpContext::domain`] to recurse into nested programs with the same
/// domain.
pub trait DifferentiableOperation<D: DifferentiableDomain>: Operation<D::Type> {
    /// Applies this operation's forward-mode Jacobian-Vector Product (JVP) rule.
    ///
    /// The returned vector must be aligned with this operation's outputs and must carry both the
    /// primal output values and the staged tangent atoms for those outputs.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active JVP context used to stage tangent operations and access the
    ///     differentiable domain.
    ///   - `inputs`: Traced inputs aligned with this operation's inputs.
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, D>,
        inputs: &[JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>],
    ) -> Result<Vec<JvpTracer<D::Value, Tracer<'jvp, D::LinearDomain>>>, TracingError>
    where
        D: 'jvp;
}

/// Concrete state threaded through forward-mode JVP rules.
///
/// [`JvpContext`] owns the active linear-program builder where tangent ops are staged. It is the
/// forward-mode counterpart of
/// [`ProgramTracingContext`](crate::tracing::ProgramTracingContext): JVP rules call
/// [`apply_operation`](Self::apply_operation) to stage tangent ops on the active builder.
#[doc(hidden)]
pub struct JvpContext<'domain, D: DifferentiableDomain> {
    /// Differentiable domain borrowed by this [`JvpContext`] for primal semantics and linear-domain selection.
    pub domain: &'domain D,

    /// [`TracingContext`] used to stage tangent operations into the active linear program.
    pub linear_context: TracingContext<'domain, D::LinearDomain>,

    /// [`ProgramBuilder`] that owns the staged linear [`Program`](crate::tracing::Program) that is currently being
    /// traced.
    pub builder: Rc<RefCell<ProgramBuilder<D::Type, D::Tangent, D::LinearOperationCarrier>>>,
}

impl<'domain, D: DifferentiableDomain> JvpContext<'domain, D> {
    /// Creates a JVP context that stages into `builder`.
    #[doc(hidden)]
    pub fn new(
        domain: &'domain D,
        builder: Rc<RefCell<ProgramBuilder<D::Type, D::Tangent, D::LinearOperationCarrier>>>,
    ) -> Self {
        Self { domain, linear_context: TracingContext::new(domain.linear_domain(), builder.clone()), builder }
    }

    /// Stages one operation in the currently active linear program.
    pub fn stage(
        &self,
        operation: D::LinearOperationCarrier,
        inputs: &[Tracer<'domain, D::LinearDomain>],
    ) -> Result<Vec<Tracer<'domain, D::LinearDomain>>, TracingError> {
        let input_refs = inputs.iter().collect::<Vec<_>>();
        self.linear_context.trace(operation, input_refs.as_slice())
    }

    /// Stages one operation from raw atom identifiers in the currently active linear program.
    pub(crate) fn stage_atom_ids(
        &self,
        operation: D::LinearOperationCarrier,
        inputs: &[AtomId],
    ) -> Result<Vec<AtomId>, TracingError> {
        let mut builder_borrow = self.builder.borrow_mut();
        let input_types = inputs
            .iter()
            .map(|atom| {
                builder_borrow
                    .atoms
                    .get(atom.index)
                    .map(|atom| atom.r#type().into_owned())
                    .ok_or(TracingError::UnboundAtomId { id: *atom })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_types = operation.infer_output_types(&input_types)?;
        let outputs = output_types.into_iter().map(|r#type| builder_borrow.add_variable(r#type)).collect::<Vec<_>>();
        builder_borrow
            .instructions
            .push(Instruction { operation, inputs: inputs.to_vec(), outputs: outputs.clone() });
        Ok(outputs)
    }

    /// Stages a constant tangent on the active linear builder.
    pub fn add_constant(&self, value: D::Tangent) -> Tracer<'domain, D::LinearDomain> {
        self.linear_context.constant(value)
    }
}

/// Forward-mode tracer carrying both a primal and a tangent.
///
/// [`JvpTracer`] is to forward-mode AD what [`Tracer`] is to ordinary
/// staging: it is the leaf wrapper that primitive operations see when a function is being evaluated
/// in JVP mode. The `primal` field carries the usual runtime value, while the `tangent` field
/// carries the directional derivative information flowing alongside it.
///
/// The type parameters have no bounds on the struct itself so that `JvpTracer` can appear in
/// signatures without eagerly propagating all tangent requirements. `tracing_v2` uses
/// `Tracer<'_, D::LinearDomain>` values for the rule-based JVP path threaded through
/// [`JvpContext`], so local primitive rules can stage tangent operations using the same
/// value-level traits as ordinary tracing.
#[derive(Clone, Debug, Parameter)]
pub struct JvpTracer<V, T> {
    /// The primal value.
    pub primal: V,

    /// The tangent value associated with the primal.
    pub tangent: T,
}

impl<Ty: Type, V: Typed<Ty>, T> Typed<Ty> for JvpTracer<V, T> {
    #[inline]
    fn r#type(&self) -> Cow<'_, Ty> {
        <V as Typed<Ty>>::r#type(&self.primal)
    }
}

impl<V: Display, T> Display for JvpTracer<V, T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.primal, formatter)
    }
}

impl<Ty: Type, V: Traceable<Ty>, T: Clone + std::fmt::Debug + Parameter> Traceable<Ty> for JvpTracer<V, T> {}

impl<T: Type + Parameter, V: Traceable<T>, O: Clone + Operation<T>, Input: Parameterized<V>, Output: Parameterized<V>>
    Program<T, V, O, Input, Output>
{
    /// Converts this staged primal [`Program`] into a staged pushforward linear map.
    ///
    /// This is the reusable IR-level form of forward-mode differentiation. Instead of evaluating
    /// the JVP immediately, it builds a staged [`Program`] over linear operations that can be
    /// replayed later on arbitrary tangent inputs at the same primal point.
    ///
    /// # Parameters
    ///
    ///   - `domain`: Linearizing domain that supplies the linear operation carrier and primitive
    ///     JVP rules.
    ///   - `input_primals`: Concrete primal values aligned with this program's input atoms.
    pub fn linearize<D: DifferentiableDomain<Type = T, Value = V>>(
        &self,
        domain: &D,
        input_primals: Vec<V>,
    ) -> Result<
        Program<T, D::Tangent, D::LinearOperationCarrier, Input::To<D::Tangent>, Output::To<D::Tangent>>,
        TracingError,
    >
    where
        V: Differentiable<T, Tangent = D::Tangent>,
        Input::Family: ParameterizedFamily<D::Tangent>,
        Output::Family: ParameterizedFamily<D::Tangent>,
        O: DifferentiableOperation<D>,
    {
        fn tangent_for_atom<'jvp, T, V, D>(
            primal_values: &[Option<V>],
            context: &JvpContext<'jvp, D>,
            tangents: &mut [Option<Tracer<'jvp, D::LinearDomain>>],
            atom_id: AtomId,
        ) -> Result<Tracer<'jvp, D::LinearDomain>, TracingError>
        where
            T: Type + Parameter,
            V: Differentiable<T, Tangent = D::Tangent>,
            D: DifferentiableDomain<Type = T>,
        {
            if let Some(tangent) = &tangents[atom_id.index] {
                return Ok(tangent.clone());
            }
            let primal = primal_values[atom_id.index].as_ref().ok_or(TracingError::UnboundAtomId { id: atom_id })?;
            let tangent = context.add_constant(primal.zero_tangent()?);
            tangents[atom_id.index] = Some(tangent.clone());
            Ok(tangent)
        }

        check_count!("input", input_primals, self.input_ids.len(), TracingError);
        let builder = Rc::new(RefCell::new(ProgramBuilder::<T, D::Tangent, D::LinearOperationCarrier>::new()));
        let mut primals: Vec<Option<V>> = vec![None; self.atoms.len()];
        let mut tangents: Vec<Option<Tracer<'_, D::LinearDomain>>> = vec![None; self.atoms.len()];
        let mut context = JvpContext::new(domain, builder.clone());
        for (input_atom, input_primal) in self.input_ids.iter().copied().zip(input_primals.into_iter()) {
            let tangent_atom = context.linear_context.input(input_primal.r#type().into_owned());
            tangents[input_atom.index] = Some(tangent_atom);
            primals[input_atom.index] = Some(input_primal);
        }
        for (atom_index, atom) in self.atoms.iter().enumerate() {
            let atom_id = AtomId { index: atom_index };
            if let Atom::Constant(value) = atom {
                primals[atom_id.index] = Some(value.clone());
            }
        }

        for instruction in &self.instructions {
            let input_duals = instruction
                .inputs
                .iter()
                .copied()
                .map(|input_atom| {
                    Ok(JvpTracer {
                        primal: primals[input_atom.index]
                            .clone()
                            .ok_or(TracingError::UnboundAtomId { id: input_atom })?,
                        tangent: tangent_for_atom::<T, V, D>(
                            primals.as_slice(),
                            &context,
                            tangents.as_mut_slice(),
                            input_atom,
                        )?,
                    })
                })
                .collect::<Result<Vec<_>, TracingError>>()?;
            let output_duals = instruction.operation.jvp(&mut context, input_duals.as_slice())?;
            check_count!("output", output_duals, instruction.outputs.len(), TracingError);
            for (output_atom, output_dual) in instruction.outputs.iter().copied().zip(output_duals.into_iter()) {
                primals[output_atom.index] = Some(output_dual.primal);
                tangents[output_atom.index] = Some(output_dual.tangent);
            }
        }

        let output_tangents = self
            .output_ids
            .iter()
            .copied()
            .map(|output_atom| {
                tangent_for_atom::<T, V, D>(primals.as_slice(), &context, tangents.as_mut_slice(), output_atom)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_tangent_atoms = output_tangents.iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
        drop(output_tangents);
        drop(context);
        drop(tangents);
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => {
                return Err(TracingError::EscapedProgramBuilder);
            }
        };
        builder
            .build(output_tangent_atoms, self.input_structure.clone(), self.output_structure.clone())?
            .simplified()
    }
}

/// Optional extension for tracing domains that support differentiation inside an active trace.
///
/// Backends usually do not implement this trait directly. Implement [`LinearizableDomain`] instead. If the selected
/// linear carrier implements [`LinearOperationCarrierFamily`], `ryft-core` derives the traced linear carrier by
/// reparameterizing that family over [`Tracer`] leaves.
pub trait DifferentiableTracingDomain: TracingDomain<OperationCarrier: SupportsAdd<Self::Type, Self::Value>> {
    /// Linear operation carrier selected for tangent and cotangent programs over traced values.
    type LinearOperationCarrier<'domain>: Clone
        + InterpretableOperation<Self::Type, Tracer<'domain, Self>>
        + LinearOperation<Self::Type, Tracer<'domain, Self>, Self::LinearOperationCarrier<'domain>>
        + SupportsZero<Self::Type, Tracer<'domain, Self>>
        + SupportsNeg<Self::Type, Tracer<'domain, Self>>
        + SupportsAdd<Self::Type, Tracer<'domain, Self>>
        + SupportsScale<Self::Type, Tracer<'domain, Self>>
    where
        Self: 'domain;
}

impl<D> DifferentiableTracingDomain for D
where
    D: LinearizableDomain<OperationCarrier: SupportsAdd<D::Type, D::Value>>,
    <<D as LinearizableDomain>::LinearDomain as TracingDomain>::OperationCarrier:
        LinearOperationCarrierFamily<D, <<D as LinearizableDomain>::LinearDomain as Domain>::Value>,
{
    type LinearOperationCarrier<'domain>
        =
        <<<D as LinearizableDomain>::LinearDomain as TracingDomain>::OperationCarrier as LinearOperationCarrierFamily<
            D,
            <<D as LinearizableDomain>::LinearDomain as Domain>::Value,
        >>::ForTracer<'domain>
    where
        Self: 'domain;
}

impl<'domain, D> TracingDomain for TracingContext<'domain, D>
where
    D: DifferentiableTracingDomain + 'domain,
{
    type OperationCarrier = D::LinearOperationCarrier<'domain>;
}

impl<'domain, D> DifferentiableDomain for TracingContext<'domain, D>
where
    D: DifferentiableTracingDomain + RuntimeDomain + 'domain,
    D::Value: Differentiable<D::Type>,
    D::OperationCarrier: SupportsAdd<D::Type, D::Value>,
    D::LinearOperationCarrier<'domain>: Clone
        + InterpretableOperation<D::Type, Tracer<'domain, D>>
        + LinearOperation<D::Type, Tracer<'domain, D>, D::LinearOperationCarrier<'domain>>
        + SupportsZero<D::Type, Tracer<'domain, D>>
        + SupportsAdd<D::Type, Tracer<'domain, D>>,
    AddOperation: InterpretableOperation<D::Type, Tracer<'domain, D>>,
{
    type Tangent = Tracer<'domain, D>;
    type LinearDomain = Self;
    type LinearOperationCarrier = D::LinearOperationCarrier<'domain>;

    #[inline]
    fn linear_domain(&self) -> &Self::LinearDomain {
        self
    }
}

impl<V> LinearizableDomain for ScalarDomain<V>
where
    V: Traceable<DataType>,
    ScalarDomain<V>: RuntimeDomain<Type = DataType> + TracingDomain<Type = DataType>,
    LinearScalarDomain<V>: TracingDomain<Type = DataType, Value = V, OperationCarrier = LinearScalarOperation<V>>,
    LinearScalarOperation<V>: Clone
        + InterpretableOperation<DataType, V>
        + LinearOperation<DataType, V, LinearScalarOperation<V>>
        + SupportsZero<DataType, V>
        + SupportsAdd<DataType, V>,
{
    type LinearDomain = LinearScalarDomain<V>;

    #[inline]
    fn linear_domain(&self) -> &Self::LinearDomain {
        &self.linear_domain
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
        assert_eq!(zero.to_string(), "zero_tangent[f64]");
        assert_eq!(<Tangent<DataType, f64> as Zero<DataType>>::zero(&DataType::F64), Ok(zero.clone()));
        assert_eq!(value.as_value(), Some(&2.5));
        assert_eq!(value.r#type().into_owned(), DataType::F64);
        assert_eq!(value.to_string(), "2.5");
        assert_eq!(<Tangent<DataType, f64> as One<DataType>>::one(&DataType::F64), Ok(Tangent::value(1.0)));
        assert_eq!(value.zero_like(), zero);

        let zero_only = Tangent::<DataType, Infallible>::zero(DataType::I32);
        assert_eq!(zero_only.r#type().into_owned(), DataType::I32);
        assert_eq!(zero_only.to_string(), "zero_tangent[i32]");
        assert_eq!(<Tangent<DataType, Infallible> as Zero<DataType>>::zero(&DataType::I32), Ok(zero_only.clone()));
        assert_eq!(zero_only.zero_like(), zero_only);

        let array_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Size::Static(2)]), None, None).unwrap();
        let array_tangent = Tangent::<ArrayType, Infallible>::zero(array_type.clone());
        assert_eq!(array_tangent.r#type().into_owned(), array_type);
    }

    #[test]
    fn test_scalar_domain_half_and_float_domains_are_differentiable() {
        let _: Option<<ScalarDomain<bf16> as DifferentiableDomain>::LinearOperationCarrier> = None;
        let _: Option<<ScalarDomain<f16> as DifferentiableDomain>::LinearOperationCarrier> = None;
        let _: Option<<ScalarDomain<f32> as DifferentiableDomain>::LinearOperationCarrier> = None;
        let _: Option<<ScalarDomain<f64> as DifferentiableDomain>::LinearOperationCarrier> = None;
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
