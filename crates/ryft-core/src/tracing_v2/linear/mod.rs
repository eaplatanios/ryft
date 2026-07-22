//! Dense Jacobian and Hessian materialization over structured values.

mod hessian;
mod jacobian;

pub use hessian::{
    Hessian, HessianBlock, HessianDifferentiate, hessian, hessian_holomorphic, hessian_holomorphic_with_aux,
    hessian_with_aux,
};
pub use jacobian::{
    Jacobian, JacobianBlock, JacobianDifferentiate, jacobian_forward, jacobian_forward_holomorphic,
    jacobian_forward_holomorphic_with_aux, jacobian_forward_with_aux, jacobian_reverse, jacobian_reverse_holomorphic,
    jacobian_reverse_holomorphic_with_aux, jacobian_reverse_with_aux,
};
