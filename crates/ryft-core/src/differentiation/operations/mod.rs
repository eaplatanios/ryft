//! Differentiation-specific operation families.
//!
//! This module owns custom JVP and VJP calls, residual-parameterized linear calls, coordinate-basis construction, and
//! gradient barriers. Differentiation algorithms and transform contexts remain in the parent module, while the
//! transform-wide residual-zero protocol is owned separately by `differentiation::zeros`.

// TODO(eaplatanios): Review this module.

pub mod coordinate_basis;
pub mod custom_jvp;
pub mod custom_vjp;
pub mod linear_call;
pub mod stop_gradient;

pub use coordinate_basis::{COORDINATE_BASIS_OPERATION_NAME, CoordinateBasisOperation};
pub use custom_jvp::{CUSTOM_JVP_OPERATION_NAME, CustomJvp, CustomJvpOperation, custom_jvp};
pub use custom_vjp::{CUSTOM_VJP_OPERATION_NAME, CustomVjp, CustomVjpOperation, custom_vjp};
pub use linear_call::LinearCallOperation;
pub use stop_gradient::{STOP_GRADIENT_OPERATION_NAME, StopGradient, StopGradientOperation, stop_gradient};

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::differentiation::forward::ForwardModeDifferentiate;
    use crate::differentiation::reverse::ReverseModeDifferentiate;
    use crate::tracing::DomainTracer;

    use super::{custom_jvp, custom_vjp};

    // TODO(eaplatanios): Is this really where this belongs?
    #[test]
    fn test_custom_derivative_wrappers_use_zero_space_boundaries() {
        type ArrayContext = EagerContext<Array, ArrayOperation<Array>>;
        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();

        let function = custom_jvp(
            |token: DomainTracer<ArrayContext>| Ok(token),
            |token: DomainTracer<ArrayContext>, tangent| Ok((token, tangent)),
        );
        assert_eq!(
            ArrayContext::new().jvp(|token, ()| function.call(token), token.clone(), zero.clone(), ()),
            Ok((token.clone(), zero.clone())),
        );

        let function = custom_vjp(
            |token: DomainTracer<ArrayContext>| Ok(token),
            |token: DomainTracer<ArrayContext>| Ok((token.clone(), token)),
            |_residual: DomainTracer<ArrayContext>, cotangent| Ok(cotangent),
        );
        let (value, pullback) = ArrayContext::new().vjp(|token, ()| function.call(token), token.clone(), ()).unwrap();
        assert_eq!(value, token);
        assert_eq!(pullback.apply(zero.clone()), Ok(zero));
    }
}
