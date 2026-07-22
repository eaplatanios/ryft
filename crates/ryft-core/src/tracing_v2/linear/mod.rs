//! Dense Jacobian and Hessian materialization over structured values.

mod hessian;

pub use hessian::{
    Hessian, HessianBlock, HessianDifferentiate, hessian, hessian_holomorphic, hessian_holomorphic_with_aux,
    hessian_with_aux,
};
