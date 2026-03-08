use std::{
    fmt::Debug,
    ops::{Add, Mul, Neg},
};

use ryft_macros::Parameter;

use crate::parameters::Parameter;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Parameter)]
pub enum ScalarAbstract {
    F32,
    F64,
}

pub trait FloatExt: Clone + Add<Output = Self> + Mul<Output = Self> + Neg<Output = Self> {
    fn sin(self) -> Self;

    fn cos(self) -> Self;
}

pub trait TraceLeaf: Clone + Parameter {
    type Abstract: Clone + Debug + Eq + PartialEq;

    fn abstract_value(&self) -> Self::Abstract;
}

pub trait ZeroLike: Clone {
    fn zero_like(&self) -> Self;
}

pub trait OneLike: Clone {
    fn one_like(&self) -> Self;
}

pub trait TraceValue: FloatExt + TraceLeaf + ZeroLike + 'static {}

impl<T> TraceValue for T where T: FloatExt + TraceLeaf + ZeroLike + 'static {}

impl FloatExt for f32 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl TraceLeaf for f32 {
    type Abstract = ScalarAbstract;

    #[inline]
    fn abstract_value(&self) -> Self::Abstract {
        ScalarAbstract::F32
    }
}

impl ZeroLike for f32 {
    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl OneLike for f32 {
    #[inline]
    fn one_like(&self) -> Self {
        1.0
    }
}

impl FloatExt for f64 {
    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }
}

impl TraceLeaf for f64 {
    type Abstract = ScalarAbstract;

    #[inline]
    fn abstract_value(&self) -> Self::Abstract {
        ScalarAbstract::F64
    }
}

impl ZeroLike for f64 {
    #[inline]
    fn zero_like(&self) -> Self {
        0.0
    }
}

impl OneLike for f64 {
    #[inline]
    fn one_like(&self) -> Self {
        1.0
    }
}
