use std::fmt::Display;

use crate::contexts::Context;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::{ElementwiseOperation, Operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{ProgramError, Value};
use crate::types::{ArrayType, DataType, TypeError};

// TODO(eaplatanios): Review this module (and split it into `exp`, `log`, and `sqrt`)? Also, put it under `math`.

/// Canonical operation name for [`ExponentialOperation`].
pub const EXPONENTIAL_OPERATION_NAME: &'static str = "exponential";

/// Canonical operation name for [`LogarithmOperation`].
pub const LOGARITHM_OPERATION_NAME: &'static str = "logarithm";

/// Canonical operation name for [`SquareRootOperation`].
pub const SQUARE_ROOT_OPERATION_NAME: &'static str = "square_root";

/// Declares one unary exponential-family [`Operation`] (a name constant is expected to already exist): the operation
/// struct with [`Display`], [`Operation`] impls over [`DataType`] and [`ArrayType`] that preserve the (floating-point
/// or complex) operand type, [`ElementwiseOperation`], interpretation through the paired value-level capability
/// trait, the [`PartiallyEvaluatableOperation`] default, and the capability trait itself with its staging blanket.
macro_rules! declare_exponential_operation {
    (
        $(#[$documentation:meta])*
        $operation:ident, $name:ident, $capability:ident, $method:ident,
        $(#[$capability_documentation:meta])*
    ) => {
        $(#[$documentation])*
        #[derive(Clone, Debug, Default)]
        pub struct $operation;

        impl Display for $operation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str($name)
            }
        }

        impl Operation<DataType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![input_types[0].clone()])
            }
        }

        impl Operation<ArrayType> for $operation {
            #[inline]
            fn name(&self) -> &'static str {
                $name
            }

            #[inline]
            fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
                ElementwiseOperation::infer_output_types(self, input_types)
            }
        }

        impl ElementwiseOperation for $operation {
            #[inline]
            fn input_count(&self) -> usize {
                1
            }
        }

        impl<V: Clone + Value + $capability, C> InterpretableOperation<V, C> for $operation
        where
            Self: Operation<V::Type>,
        {
            #[inline]
            fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![inputs[0].$method()?])
            }
        }

        impl<C: Context> PartiallyEvaluatableOperation<C> for $operation where C::Operation: From<$operation> {}

        $(#[$capability_documentation])*
        pub trait $capability: Sized {
            /// Computes this operation elementwise for this value, returning a [`ProgramError`] if something goes
            /// wrong (e.g., when the value is not floating-point or complex valued).
            fn $method(&self) -> Result<Self, ProgramError>;
        }

        impl<V: Value<DispatchDomain: Context<Operation: From<$operation>>>> $capability for V {
            #[inline]
            fn $method(&self) -> Result<Self, ProgramError> {
                Ok(self.dispatch_domain().bind($operation, &[self.clone()])?.remove(0))
            }
        }
    };
}

declare_exponential_operation!(
    /// [`Operation`] that computes the elementwise natural exponential of one value (i.e., `x ↦ eˣ`, the analytic
    /// continuation `e^z` on complex operands) while preserving its type metadata. Only floating-point and complex
    /// operands are supported.
    ExponentialOperation, EXPONENTIAL_OPERATION_NAME, Exponential, exponential,
    /// Value-level elementwise natural-exponential capability. [`Exponential`] fills the same role for
    /// [`ExponentialOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
);

declare_exponential_operation!(
    /// [`Operation`] that computes the elementwise natural logarithm of one value (i.e., `x ↦ ln(x)`, the principal
    /// branch `ln(z)` on complex operands) while preserving its type metadata. Only floating-point and complex
    /// operands are supported.
    LogarithmOperation, LOGARITHM_OPERATION_NAME, Logarithm, logarithm,
    /// Value-level elementwise natural-logarithm capability. [`Logarithm`] fills the same role for
    /// [`LogarithmOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
);

declare_exponential_operation!(
    /// [`Operation`] that computes the elementwise square root of one value (i.e., `x ↦ √x`, the principal branch
    /// `√z` on complex operands) while preserving its type metadata. Only floating-point and complex operands are
    /// supported.
    SquareRootOperation, SQUARE_ROOT_OPERATION_NAME, SquareRoot, square_root,
    /// Value-level elementwise square-root capability. [`SquareRoot`] fills the same role for
    /// [`SquareRootOperation`] that [`Sin`](crate::Sin) fills for [`SinOperation`](crate::SinOperation).
);
