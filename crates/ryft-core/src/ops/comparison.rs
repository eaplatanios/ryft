// TODO(eaplatanios): Really need to figure this one out.

use half::{bf16, f16};

use crate::differentiation::JvpTracer;

// ======================================================= EQ =======================================================

// Our [PartialEq] implementation enables comparisons of the underlying value directly with some other value.
// It intentionally ignores the tangent value and it is meant to enable tracing through things like `if` statements.

// impl<V: PartialEq<RV>, VT: PartialEq<RT>, RV, RT> PartialEq<JvpTracer<RV, RT>> for JvpTracer<V, VT> {
//     fn eq(&self, other: &JvpTracer<RV, RT>) -> bool {
//         self.value.eq(&other.value) && self.tangent.eq(&other.tangent)
//     }
// }

// Due to Rust's orphan rule we have to special case some types that support equality with [JvpTracer]s
// appearing on the right-hand side. Since the implementation for all of them looks the same, we define
// the following macro for generating those implementations.
macro_rules! impl_partial_eq_with_jvp_tracer_rhs {
    ($($T:ty),* $(,)*) => {$(
        impl<V, VT> PartialEq<$T> for JvpTracer<V, VT>
        where
            V: PartialEq<$T>,
        {
            fn eq(&self, other: &$T) -> bool {
                self.value.eq(&other)
            }
        }

        impl<V, VT> PartialEq<JvpTracer<V, VT>> for $T
        where
            $T: PartialEq<V>,
        {
            fn eq(&self, other: &JvpTracer<V, VT>) -> bool {
                self.eq(&other.value)
            }
        }
    )*};
}

impl_partial_eq_with_jvp_tracer_rhs!(bool, i8, i16, i32, i64, u8, u16, u32, u64, bf16, f16, f32, f64);

// // ======================================================= CMP =======================================================

// // Our [PartialOrd] implementation enables comparisons of the underlying value directly with some other value.
// // It intentionally ignores the tangent value and it is meant to enable tracing through things like `if` statements.
// impl<V: PartialOrd<Rhs>, VT, Rhs: ?Sized> PartialOrd<Rhs> for JvpTracer<V, VT> {
//     fn partial_cmp(&self, other: &Rhs) -> Option<std::cmp::Ordering> {
//         self.value.partial_cmp(other)
//     }
// }

// Due to Rust's orphan rule we have to special case some types that support comparisons with [JvpTracer]s
// appearing on the right-hand side. We define a macro to generate implementations for those types, similar
// to what we did for [PartialEq].
macro_rules! impl_partial_ord_with_jvp_tracer_rhs {
    ($($T:ty),* $(,)*) => {$(
        impl<V, VT> PartialOrd<$T> for JvpTracer<V, VT>
        where
            V: PartialOrd<$T>,
        {
            fn partial_cmp(&self, other: &$T) -> Option<std::cmp::Ordering> {
                self.value.partial_cmp(&other)
            }
        }

        impl<V, VT> PartialOrd<JvpTracer<V, VT>> for $T
        where
            $T: PartialOrd<V>,
        {
            fn partial_cmp(&self, other: &JvpTracer<V, VT>) -> Option<std::cmp::Ordering> {
                self.partial_cmp(&other.value)
            }
        }
    )*};
}

impl_partial_ord_with_jvp_tracer_rhs!(bool, i8, i16, i32, i64, u8, u16, u32, u64, bf16, f16, f32, f64);

#[cfg(test)]
mod tests {
    use crate::{
        differentiation::{differential, linear, linearize},
        ops::{
            constants::Constant,
            trigonometric::{Cos, Sin},
        },
    };

    use std::ops::{Add, Mul};

    // ======================================================= FUNCTIONS =======================================================

    fn test<T: Clone + Constant<f32> + Sin + Cos + Add<Output = T> + Mul<Output = T> + PartialEq<f32>>(x: T) -> T {
        if x == 1f32 { x * T::constant(2f32) } else { x.clone().sin() + x.clone().cos() * x.sin() }
    }

    #[test]
    fn test_jvp() {
        let (y, d) = linearize(test, 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(test, 2f32).unwrap();
        let dy = d.interpret(2f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let differential_1 = differential(test)(3f32);
        println!("## @3f32 => {differential_1}");

        let differential_2 = differential(differential(test))(3f32);
        println!("## @3f32 => {differential_2}");

        let (y, d) = linearize(differential(test), 3f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let f = linear(differential(test));
        let y = f(1f32);
        println!("{y}");

        let (y, d) = linearize(linear(differential(test)), 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(linear(linear(differential(differential(differential(test))))), 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(linear(linear(differential(differential(differential(test))))), 2f32).unwrap();
        let dy = d.interpret(2f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);
    }
}
