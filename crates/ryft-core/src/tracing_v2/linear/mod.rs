//! Dense Jacobian and Hessian materialization over structured values.

use std::fmt::Debug;

use crate::contexts::Context;
use crate::differentiation::{
    DenseDifferentiableType, DifferentiableOperation, DifferentiationError, LinearizationTracer, TransposableOperation,
};
use crate::operations::constants::ZeroOperation;
use crate::operations::math::AddOperation;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::tracing::TracingContext;

mod hessian;
mod jacobian;

pub use hessian::{
    Hessian, HessianBlock, hessian, hessian_holomorphic, hessian_holomorphic_with_aux, hessian_with_aux,
};
pub use jacobian::{
    Jacobian, JacobianBlock, jacfwd, jacfwd_holomorphic, jacfwd_holomorphic_with_aux, jacfwd_with_aux, jacrev,
    jacrev_holomorphic, jacrev_holomorphic_with_aux, jacrev_with_aux,
};

/// Jacobian and Hessian materialization operations supported by an execution or staging [`Context`].
///
/// Unlike the lower-level JVP and VJP transforms, these methods enumerate every finite input or output coordinate and
/// return structured derivative blocks. Availability is determined by [`DenseDifferentiableType`], so the result data
/// model remains generic over [`Type`](crate::Type) while execution is exposed only for type
/// families with an efficient packed replay representation.
///
/// Ordinary forward mode requires real differentiated inputs but permits complex outputs. Ordinary reverse mode
/// requires real differentiated outputs but permits complex inputs. Ordinary Hessians require real inputs and
/// outputs. The `*_holomorphic` variants instead require every differentiated input and output leaf to be complex and
/// treat the caller's holomorphy promise as part of the transform contract. Leaves without a tangent/cotangent space
/// and leaves whose coordinate count is not statically finite are rejected before packed replay.
pub trait DenseDifferentiate: Context {
    /// Materializes the complete forward-mode Jacobian of `function` at `primals`.
    fn jacfwd<F, I, O>(
        &self,
        function: F,
        primals: I,
    ) -> Result<Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, DifferentiationError>
    where
        Self::Type: DenseDifferentiableType<Self>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O: Parameterized<
                LinearizationTracer<Self>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
            >,
        O::To<Self::Value>: Parameterized<Self::Value, To<Self::Value> = O::To<Self::Value>>,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<O, ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + From<ZeroOperation<Self::Type>>,
    {
        jacobian::jacfwd_in(self, function, primals, false)
    }

    /// Materializes the complete forward-mode Jacobian under the promise that `function` is holomorphic.
    fn jacfwd_holomorphic<F, I, O>(
        &self,
        function: F,
        primals: I,
    ) -> Result<Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, DifferentiationError>
    where
        Self::Type: DenseDifferentiableType<Self>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O: Parameterized<
                LinearizationTracer<Self>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
            >,
        O::To<Self::Value>: Parameterized<Self::Value, To<Self::Value> = O::To<Self::Value>>,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<O, ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + From<ZeroOperation<Self::Type>>,
    {
        jacobian::jacfwd_in(self, function, primals, true)
    }

    /// Materializes the complete reverse-mode Jacobian of `function` at `primals`.
    fn jacrev<F, I, O>(
        &self,
        function: F,
        primals: I,
    ) -> Result<Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, DifferentiationError>
    where
        Self::Type: DenseDifferentiableType<Self>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O: Parameterized<
                LinearizationTracer<Self>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
            >,
        O::To<Self::Value>: Parameterized<Self::Value, To<Self::Value> = O::To<Self::Value>>,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<O, ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
    {
        jacobian::jacrev_in(self, function, primals, false)
    }

    /// Materializes the complete reverse-mode Jacobian under the promise that `function` is holomorphic.
    fn jacrev_holomorphic<F, I, O>(
        &self,
        function: F,
        primals: I,
    ) -> Result<Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, DifferentiationError>
    where
        Self::Type: DenseDifferentiableType<Self>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O: Parameterized<
                LinearizationTracer<Self>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
            >,
        O::To<Self::Value>: Parameterized<Self::Value, To<Self::Value> = O::To<Self::Value>>,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<O, ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
    {
        jacobian::jacrev_in(self, function, primals, true)
    }

    /// Materializes a forward-mode Jacobian while returning nondifferentiated auxiliary outputs.
    fn jacfwd_with_aux<F, I, O, A>(
        &self,
        function: F,
        primals: I,
    ) -> Result<
        (Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, A::To<Self::Value>),
        DifferentiationError,
    >
    where
        Self::Type: DenseDifferentiableType<Self>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O: Parameterized<
                LinearizationTracer<Self>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
            >,
        O::To<Self::Value>: Parameterized<Self::Value, To<Self::Value> = O::To<Self::Value>>,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        A: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
        F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<(O, A), ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + From<ZeroOperation<Self::Type>>,
    {
        jacobian::jacfwd_with_aux_in(self, function, primals, false)
    }

    /// Materializes a holomorphic forward-mode Jacobian while returning nondifferentiated auxiliary outputs.
    fn jacfwd_holomorphic_with_aux<F, I, O, A>(
        &self,
        function: F,
        primals: I,
    ) -> Result<
        (Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, A::To<Self::Value>),
        DifferentiationError,
    >
    where
        Self::Type: DenseDifferentiableType<Self>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O: Parameterized<
                LinearizationTracer<Self>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
            >,
        O::To<Self::Value>: Parameterized<Self::Value, To<Self::Value> = O::To<Self::Value>>,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        A: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
        F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<(O, A), ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + From<ZeroOperation<Self::Type>>,
    {
        jacobian::jacfwd_with_aux_in(self, function, primals, true)
    }

    /// Materializes a reverse-mode Jacobian while returning nondifferentiated auxiliary outputs.
    fn jacrev_with_aux<F, I, O, A>(
        &self,
        function: F,
        primals: I,
    ) -> Result<
        (Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, A::To<Self::Value>),
        DifferentiationError,
    >
    where
        Self::Type: DenseDifferentiableType<Self>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O: Parameterized<
                LinearizationTracer<Self>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
            >,
        O::To<Self::Value>: Parameterized<Self::Value, To<Self::Value> = O::To<Self::Value>>,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        A: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
        F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<(O, A), ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
    {
        jacobian::jacrev_with_aux_in(self, function, primals, false)
    }

    /// Materializes a holomorphic reverse-mode Jacobian while returning nondifferentiated auxiliary outputs.
    fn jacrev_holomorphic_with_aux<F, I, O, A>(
        &self,
        function: F,
        primals: I,
    ) -> Result<
        (Jacobian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, A::To<Self::Value>),
        DifferentiationError,
    >
    where
        Self::Type: DenseDifferentiableType<Self>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Type>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O: Parameterized<
                LinearizationTracer<Self>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Type>,
            >,
        O::To<Self::Value>: Parameterized<Self::Value, To<Self::Value> = O::To<Self::Value>>,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        A: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
        F: FnOnce(I::To<LinearizationTracer<Self>>) -> Result<(O, A), ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
    {
        jacobian::jacrev_with_aux_in(self, function, primals, true)
    }

    /// Materializes the complete output/input/input Hessian using forward-over-reverse differentiation.
    fn hessian<F, I, O>(
        &self,
        function: F,
        primals: I,
    ) -> Result<Hessian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, DifferentiationError>
    where
        Self::Type: DenseDifferentiableType<Self> + DenseDifferentiableType<hessian::NestedDenseContext<Self>>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<Self::Type>
                            + ParameterizedFamily<LinearizationTracer<Self>>
                            + ParameterizedFamily<LinearizationTracer<hessian::NestedDenseContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        I::To<LinearizationTracer<Self>>: Parameterized<
                LinearizationTracer<Self>,
                To<LinearizationTracer<Self>> = I::To<LinearizationTracer<Self>>,
                To<LinearizationTracer<hessian::NestedDenseContext<Self>>> = I::To<
                    LinearizationTracer<hessian::NestedDenseContext<Self>>,
                >,
                To<Self::Type> = I::To<Self::Type>,
            >,
        O: Parameterized<
                LinearizationTracer<hessian::NestedDenseContext<Self>>,
                Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<LinearizationTracer<Self>>,
            >,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O::To<LinearizationTracer<Self>>:
            Parameterized<LinearizationTracer<Self>, To<LinearizationTracer<Self>> = O::To<LinearizationTracer<Self>>>,
        F: FnOnce(I::To<LinearizationTracer<hessian::NestedDenseContext<Self>>>) -> Result<O, ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<hessian::NestedDenseContext<Self>>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + DifferentiableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<Self::Constant, Self::Operation>>>
            + DifferentiableOperation<PartialEvaluationContext<hessian::NestedDenseContext<Self>>>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
    {
        hessian::hessian_in(self, function, primals, false)
    }

    /// Materializes the complete holomorphic Hessian using forward-over-reverse differentiation.
    fn hessian_holomorphic<F, I, O>(
        &self,
        function: F,
        primals: I,
    ) -> Result<Hessian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, DifferentiationError>
    where
        Self::Type: DenseDifferentiableType<Self> + DenseDifferentiableType<hessian::NestedDenseContext<Self>>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<Self::Type>
                            + ParameterizedFamily<LinearizationTracer<Self>>
                            + ParameterizedFamily<LinearizationTracer<hessian::NestedDenseContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        I::To<LinearizationTracer<Self>>: Parameterized<
                LinearizationTracer<Self>,
                To<LinearizationTracer<Self>> = I::To<LinearizationTracer<Self>>,
                To<LinearizationTracer<hessian::NestedDenseContext<Self>>> = I::To<
                    LinearizationTracer<hessian::NestedDenseContext<Self>>,
                >,
                To<Self::Type> = I::To<Self::Type>,
            >,
        O: Parameterized<
                LinearizationTracer<hessian::NestedDenseContext<Self>>,
                Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<LinearizationTracer<Self>>,
            >,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O::To<LinearizationTracer<Self>>:
            Parameterized<LinearizationTracer<Self>, To<LinearizationTracer<Self>> = O::To<LinearizationTracer<Self>>>,
        F: FnOnce(I::To<LinearizationTracer<hessian::NestedDenseContext<Self>>>) -> Result<O, ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<hessian::NestedDenseContext<Self>>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + DifferentiableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<Self::Constant, Self::Operation>>>
            + DifferentiableOperation<PartialEvaluationContext<hessian::NestedDenseContext<Self>>>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
    {
        hessian::hessian_in(self, function, primals, true)
    }

    /// Materializes the complete Hessian while returning nondifferentiated auxiliary outputs.
    fn hessian_with_aux<F, I, O, A>(
        &self,
        function: F,
        primals: I,
    ) -> Result<
        (Hessian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, A::To<Self::Value>),
        DifferentiationError,
    >
    where
        Self::Type: DenseDifferentiableType<Self> + DenseDifferentiableType<hessian::NestedDenseContext<Self>>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<Self::Type>
                            + ParameterizedFamily<LinearizationTracer<Self>>
                            + ParameterizedFamily<LinearizationTracer<hessian::NestedDenseContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        I::To<LinearizationTracer<Self>>: Parameterized<
                LinearizationTracer<Self>,
                To<LinearizationTracer<Self>> = I::To<LinearizationTracer<Self>>,
                To<LinearizationTracer<hessian::NestedDenseContext<Self>>> = I::To<
                    LinearizationTracer<hessian::NestedDenseContext<Self>>,
                >,
                To<Self::Type> = I::To<Self::Type>,
            >,
        O: Parameterized<
                LinearizationTracer<hessian::NestedDenseContext<Self>>,
                Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<LinearizationTracer<Self>>,
            >,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O::To<LinearizationTracer<Self>>:
            Parameterized<LinearizationTracer<Self>, To<LinearizationTracer<Self>> = O::To<LinearizationTracer<Self>>>,
        A: Parameterized<
                LinearizationTracer<hessian::NestedDenseContext<Self>>,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Value>,
            >,
        A::To<LinearizationTracer<Self>>:
            Parameterized<LinearizationTracer<Self>, To<Self::Value> = A::To<Self::Value>>,
        F: FnOnce(I::To<LinearizationTracer<hessian::NestedDenseContext<Self>>>) -> Result<(O, A), ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<hessian::NestedDenseContext<Self>>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + DifferentiableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<Self::Constant, Self::Operation>>>
            + DifferentiableOperation<PartialEvaluationContext<hessian::NestedDenseContext<Self>>>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
    {
        hessian::hessian_with_aux_in(self, function, primals, false)
    }

    /// Materializes the complete holomorphic Hessian while returning nondifferentiated auxiliary outputs.
    fn hessian_holomorphic_with_aux<F, I, O, A>(
        &self,
        function: F,
        primals: I,
    ) -> Result<
        (Hessian<Self::Type, Self::Value, I::To<Self::Type>, O::To<Self::Type>>, A::To<Self::Value>),
        DifferentiationError,
    >
    where
        Self::Type: DenseDifferentiableType<Self> + DenseDifferentiableType<hessian::NestedDenseContext<Self>>,
        I: Parameterized<
                Self::Value,
                To<Self::Value> = I,
                Family: ParameterizedFamily<Self::Type>
                            + ParameterizedFamily<LinearizationTracer<Self>>
                            + ParameterizedFamily<LinearizationTracer<hessian::NestedDenseContext<Self>>>,
                ParameterStructure: Debug + PartialEq,
            >,
        I::To<Self::Type>: Clone + Parameterized<Self::Type>,
        I::To<LinearizationTracer<Self>>: Parameterized<
                LinearizationTracer<Self>,
                To<LinearizationTracer<Self>> = I::To<LinearizationTracer<Self>>,
                To<LinearizationTracer<hessian::NestedDenseContext<Self>>> = I::To<
                    LinearizationTracer<hessian::NestedDenseContext<Self>>,
                >,
                To<Self::Type> = I::To<Self::Type>,
            >,
        O: Parameterized<
                LinearizationTracer<hessian::NestedDenseContext<Self>>,
                Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<LinearizationTracer<Self>>,
            >,
        O::To<Self::Type>: Clone + Parameterized<Self::Type>,
        O::To<LinearizationTracer<Self>>:
            Parameterized<LinearizationTracer<Self>, To<LinearizationTracer<Self>> = O::To<LinearizationTracer<Self>>>,
        A: Parameterized<
                LinearizationTracer<hessian::NestedDenseContext<Self>>,
                Family: ParameterizedFamily<LinearizationTracer<Self>> + ParameterizedFamily<Self::Value>,
            >,
        A::To<LinearizationTracer<Self>>:
            Parameterized<LinearizationTracer<Self>, To<Self::Value> = A::To<Self::Value>>,
        F: FnOnce(I::To<LinearizationTracer<hessian::NestedDenseContext<Self>>>) -> Result<(O, A), ProgramError>,
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<hessian::NestedDenseContext<Self>>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + DifferentiableOperation<TracingContext<Self::Constant, Self::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<Self::Constant, Self::Operation>>>
            + DifferentiableOperation<PartialEvaluationContext<hessian::NestedDenseContext<Self>>>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
    {
        hessian::hessian_with_aux_in(self, function, primals, true)
    }
}

impl<C: Context> DenseDifferentiate for C {}
