use std::{
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg, Sub},
};

use half::{bf16, f16};

use crate::{
    assert_input_count_matches,
    differentiation::JvpTracer,
    programs::{InterpretableOp, LinearInterpretableOp, LinearOp, Op, ProgramError},
    tracing::{Traceable, TraceableOp, Tracer},
    types::{ArrayStructureType, Typed},
};

// ======================================================= NEG =======================================================

#[derive(Clone, Debug)]
pub struct NegOp;

impl Display for NegOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Neg")
    }
}

impl<T: Clone> Op<T> for NegOp {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        assert_input_count_matches!(input_types.len(), 1);
        Ok(vec![input_types[0].clone()])
    }
}

impl<T: Clone, V: Clone + Neg<Output = V>> InterpretableOp<T, V> for NegOp {
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        assert_input_count_matches!(inputs.len(), 1);
        Ok(vec![-inputs[0].clone()])
    }
}

impl<T: Clone, V: Clone> LinearOp<T, V> for NegOp
where
    Tracer<T, V, Box<dyn LinearOp<T, V>>>: Neg<Output = Tracer<T, V, Box<dyn LinearOp<T, V>>>>,
{
    fn transpose(
        &self,
        _: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
        output_tangents: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
    ) -> Vec<Tracer<T, V, Box<dyn LinearOp<T, V>>>> {
        output_tangents.iter().map(|output_tangent| (*output_tangent).clone().neg()).collect()
    }
}

impl<T: Clone, V: Clone + Neg<Output = V>> LinearInterpretableOp<T, V> for NegOp where
    Tracer<T, V, Box<dyn LinearOp<T, V>>>: Neg<Output = Tracer<T, V, Box<dyn LinearOp<T, V>>>>
{
}

impl<V: Neg<Output = VN>, VT: Neg<Output = VTN>, VN, VTN> Neg for JvpTracer<V, VT> {
    type Output = JvpTracer<VN, VTN>;

    fn neg(self) -> Self::Output {
        JvpTracer { value: -self.value, tangent: -self.tangent }
    }
}

// TODO(eaplatanios): Update the other macros below with similar logic.
macro_rules! impl_neg_for_tracer {
    ($ty:path) => { impl_neg_for_tracer!($ty, value_type_bounds = []); };
    ($ty:path, value_type_bounds = [$($value_type_bounds:path),*]) => {
        impl<T: Clone, V: Clone + Display + Typed<T> $(+$value_type_bounds)*> Neg for Tracer<T, V, Box<dyn $ty>> {
            type Output = Self;

            fn neg(self) -> Self::Output {
                let inputs = vec![&self];
                let outputs = (Box::new(NegOp) as Box<dyn $ty>).trace(inputs.as_slice()).unwrap();
                debug_assert_eq!(outputs.len(), 1);
                outputs[0].clone()
            }
        }
    };
}

// TODO(eaplatanios): We need to also add support for handling constants like `Tracer + Tensor`.
// TODO(eaplatanios): `Box<dyn Op<T>>` is way too generic. What can we do with this?
impl_neg_for_tracer!(Op<T>);
impl_neg_for_tracer!(InterpretableOp<T, V>, value_type_bounds = [Clone, Neg<Output = V>]);
impl_neg_for_tracer!(LinearOp<T, V>);
impl_neg_for_tracer!(LinearInterpretableOp<T, V>, value_type_bounds = [Clone, Neg<Output = V>]);

// ======================================================= ADD =======================================================

#[derive(Clone, Debug)]
pub struct AddOp;

impl Display for AddOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Add")
    }
}

impl<T: Clone + ArrayStructureType> Op<T> for AddOp {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        assert_input_count_matches!(input_types.len(), 2);
        Ok(T::broadcast(input_types).map(|tpe| vec![tpe])?)
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Add<Output = V>> InterpretableOp<T, V> for AddOp {
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        assert_input_count_matches!(inputs.len(), 2);
        Ok(vec![inputs[0].clone() + inputs[1].clone()])
    }
}

impl<T: Clone + ArrayStructureType, V: Clone> LinearOp<T, V> for AddOp {
    fn transpose(
        &self,
        inputs: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
        output_tangents: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
    ) -> Vec<Tracer<T, V, Box<dyn LinearOp<T, V>>>> {
        // TODO(eaplatanios): `transpose` should be returning a [Result].
        assert_eq!(inputs.len(), 2);
        assert_eq!(output_tangents.len(), 1);
        vec![output_tangents[0].clone(), output_tangents[0].clone()]
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Add<Output = V>> LinearInterpretableOp<T, V> for AddOp {}

impl<V: Add<VR, Output = V>, VT: Add<VTR, Output = VT>, VR, VTR> Add<JvpTracer<VR, VTR>> for JvpTracer<V, VT> {
    type Output = JvpTracer<V, VT>;

    fn add(self, rhs: JvpTracer<VR, VTR>) -> Self::Output {
        JvpTracer { value: self.value + rhs.value, tangent: self.tangent + rhs.tangent }
    }
}

// Tracer<V> + Tracer<V>
macro_rules! impl_add_for_tracer {
    ($ty:path, value_type_bounds = ($($value_type_bounds:path),*)) => {
        impl<T: Clone + ArrayStructureType, V: Clone + Display + Typed<T> $(+$value_type_bounds)*> Add for Tracer<T, V, Box<dyn $ty>> {
            type Output = Self;

            fn add(self, rhs: Self) -> Self::Output {
                let inputs = vec![&self, &rhs];
                let outputs = (Box::new(AddOp) as Box<dyn $ty>).trace(inputs.as_slice()).unwrap();
                debug_assert_eq!(outputs.len(), 1);
                outputs[0].clone()
            }
        }
    };
}

// TODO(eaplatanios): We need to also add support for handling constants like `Tracer + Tensor`.
impl_add_for_tracer!(Op<T>, value_type_bounds = ());
impl_add_for_tracer!(InterpretableOp<T, V>, value_type_bounds = (Add<Output = V>));
impl_add_for_tracer!(LinearOp<T, V>, value_type_bounds = ());
impl_add_for_tracer!(LinearInterpretableOp<T, V>, value_type_bounds = (Add<Output = V>));

// ======================================================= SUB =======================================================

#[derive(Clone, Debug)]
pub struct SubOp;

impl Display for SubOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sub")
    }
}

impl<T: Clone + ArrayStructureType> Op<T> for SubOp {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        assert_input_count_matches!(input_types.len(), 2);
        Ok(T::broadcast(input_types).map(|tpe| vec![tpe])?)
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Sub<Output = V>> InterpretableOp<T, V> for SubOp {
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        assert_input_count_matches!(inputs.len(), 2);
        Ok(vec![inputs[0].clone() - inputs[1].clone()])
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Display + Sub<Output = V> + Typed<T>> LinearOp<T, V> for SubOp {
    fn transpose(
        &self,
        inputs: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
        output_tangents: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
    ) -> Vec<Tracer<T, V, Box<dyn LinearOp<T, V>>>> {
        assert_eq!(inputs.len(), 2);
        assert_eq!(output_tangents.len(), 1);
        vec![output_tangents[0].clone(), -output_tangents[0].clone()]
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Display + Sub<Output = V> + Typed<T>> LinearInterpretableOp<T, V>
    for SubOp
{
}

impl<V: Sub<VR>, VT: Sub<VTR>, VR, VTR> Sub<JvpTracer<VR, VTR>> for JvpTracer<V, VT> {
    type Output = JvpTracer<V::Output, VT::Output>;

    fn sub(self, rhs: JvpTracer<VR, VTR>) -> Self::Output {
        JvpTracer { value: self.value - rhs.value, tangent: self.tangent - rhs.tangent }
    }
}

macro_rules! impl_sub_for_tracer {
    ($ty:path, value_type_bounds = ($($value_type_bounds:path),*)) => {
        impl<T: Clone + ArrayStructureType, V: Clone + Display + Typed<T> $(+$value_type_bounds)*> Sub for Tracer<T, V, Box<dyn $ty>> {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self::Output {
                let inputs = vec![&self, &rhs];
                let outputs = (Box::new(SubOp) as Box<dyn $ty>).trace(inputs.as_slice()).unwrap();
                debug_assert_eq!(outputs.len(), 1);
                outputs[0].clone()
            }
        }
    };
}

impl_sub_for_tracer!(Op<T>, value_type_bounds = ());
impl_sub_for_tracer!(InterpretableOp<T, V>, value_type_bounds = (Clone, Sub<Output = V>));
impl_sub_for_tracer!(LinearOp<T, V>, value_type_bounds = (Sub<Output = V>));
impl_sub_for_tracer!(LinearInterpretableOp<T, V>, value_type_bounds = (Clone, Sub<Output = V>));

// ======================================================= MUL =======================================================

#[derive(Clone, Debug)]
pub struct MulOp;

impl Display for MulOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mul")
    }
}

impl<T: Clone + ArrayStructureType> Op<T> for MulOp {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        assert_input_count_matches!(input_types.len(), 2);
        Ok(T::broadcast(input_types).map(|tpe| vec![tpe])?)
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Mul<Output = V>> InterpretableOp<T, V> for MulOp {
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        assert_input_count_matches!(inputs.len(), 2);
        Ok(vec![inputs[0].clone() * inputs[1].clone()])
    }
}

impl<V: Clone + Mul<VR> + Mul<VTR, Output: Add<<VT as Mul<VR>>::Output>>, VT: Mul<VR>, VR: Clone, VTR>
    Mul<JvpTracer<VR, VTR>> for JvpTracer<V, VT>
{
    type Output = JvpTracer<<V as Mul<VR>>::Output, <<V as Mul<VTR>>::Output as Add<<VT as Mul<VR>>::Output>>::Output>;

    fn mul(self, rhs: JvpTracer<VR, VTR>) -> Self::Output {
        JvpTracer {
            value: self.value.clone() * rhs.value.clone(),
            tangent: self.value * rhs.tangent + self.tangent * rhs.value,
        }
    }
}

macro_rules! impl_mul_for_tracer {
    ($ty:path, value_type_bounds = ($($value_type_bounds:path),*)) => {
        impl<T: Clone + ArrayStructureType, V: Clone + Display + Typed<T> $(+$value_type_bounds)*> Mul for Tracer<T, V, Box<dyn $ty>> {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self::Output {
                let inputs = vec![&self, &rhs];
                let outputs = (Box::new(MulOp) as Box<dyn $ty>).trace(inputs.as_slice()).unwrap();
                debug_assert_eq!(outputs.len(), 1);
                outputs[0].clone()
            }
        }
    };
}

impl_mul_for_tracer!(Op<T>, value_type_bounds = ());
impl_mul_for_tracer!(InterpretableOp<T, V>, value_type_bounds = (Clone, Mul<Output = V>));

// ======================================================= MUL-R =======================================================

#[derive(Clone, Debug)]
pub struct RightMulOp<V> {
    coefficient: V,
}

impl<V: Display> Display for RightMulOp<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RightMul[{}]", self.coefficient)
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Display + Typed<T>> Op<T> for RightMulOp<V> {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        assert_input_count_matches!(input_types.len(), 1);
        let coefficient_type = self.coefficient.tpe();
        let input_types = vec![input_types[0], &coefficient_type];
        Ok(T::broadcast(input_types.as_slice()).map(|tpe| vec![tpe])?)
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Mul<Output = V>, R: Clone + Display + Into<V> + Typed<T>>
    InterpretableOp<T, V> for RightMulOp<R>
{
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        assert_input_count_matches!(inputs.len(), 1);
        Ok(vec![inputs[0].clone() * self.coefficient.clone().into()])
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Display + Typed<T> + 'static, R: Clone + Display + Into<V> + Typed<T>>
    LinearOp<T, V> for RightMulOp<R>
{
    fn transpose(
        &self,
        inputs: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
        output_tangents: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
    ) -> Vec<Tracer<T, V, Box<dyn LinearOp<T, V>>>> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(output_tangents.len(), 1);
        let op = RightMulOp { coefficient: self.coefficient.clone().into() };
        let inputs = vec![inputs[0]];
        let outputs = (Box::new(op) as Box<dyn LinearOp<T, V>>).trace(inputs.as_slice()).unwrap();
        assert_eq!(outputs.len(), 1);
        outputs
    }
}

impl<
    T: Clone + ArrayStructureType,
    V: Clone + Display + Mul<Output = V> + Typed<T> + 'static,
    R: Clone + Display + Into<V> + Typed<T>,
> LinearInterpretableOp<T, V> for RightMulOp<R>
{
}

macro_rules! impl_right_mul_for_traceable {
    ($O:path) => {
        impl<
            T: Clone + ArrayStructureType,
            V: Clone + Display + Typed<T> + Mul<Output = V> + 'static,
            TR: Typed<T> + Into<V> + Traceable<Value = V>,
        > Mul<TR> for Tracer<T, V, Box<dyn $O>>
        {
            type Output = Self;

            fn mul(self, rhs: TR) -> Self::Output {
                let op = LeftMulOp { coefficient: rhs.into() };
                let inputs = vec![&self];
                let outputs = (Box::new(op) as Box<dyn $O>).trace(inputs.as_slice()).unwrap();
                debug_assert_eq!(outputs.len(), 1);
                outputs[0].clone()
            }
        }
    };
}

impl_right_mul_for_traceable!(Op<T>);
impl_right_mul_for_traceable!(InterpretableOp<T, V>);
impl_right_mul_for_traceable!(LinearOp<T, V>);
impl_right_mul_for_traceable!(LinearInterpretableOp<T, V>);

// ======================================================= MUL-L =======================================================

#[derive(Clone, Debug)]
pub struct LeftMulOp<V> {
    coefficient: V,
}

impl<V: Display> Display for LeftMulOp<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LeftMul[{}]", self.coefficient)
    }
}

impl<T: Clone + ArrayStructureType, V: Clone + Display + Typed<T>> Op<T> for LeftMulOp<V> {
    fn infer_output_types(&self, input_types: &[&T]) -> Result<Vec<T>, ProgramError> {
        assert_input_count_matches!(input_types.len(), 1);
        let coefficient_type = self.coefficient.tpe();
        let input_types = vec![&coefficient_type, input_types[0]];
        Ok(T::broadcast(input_types.as_slice()).map(|tpe| vec![tpe])?)
    }
}

impl<T: Clone + ArrayStructureType, L: Clone + Display + Into<V> + Typed<T>, V: Clone + Mul<Output = V>>
    InterpretableOp<T, V> for LeftMulOp<L>
{
    fn interpret(&self, inputs: &[&V]) -> Result<Vec<V>, ProgramError> {
        assert_input_count_matches!(inputs.len(), 1);
        Ok(vec![self.coefficient.clone().into() * inputs[0].clone()])
    }
}

impl<T: Clone + ArrayStructureType, L: Clone + Display + Into<V> + Typed<T>, V: Clone + Display + Typed<T> + 'static>
    LinearOp<T, V> for LeftMulOp<L>
{
    fn transpose(
        &self,
        inputs: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
        output_tangents: &[&Tracer<T, V, Box<dyn LinearOp<T, V>>>],
    ) -> Vec<Tracer<T, V, Box<dyn LinearOp<T, V>>>> {
        assert_eq!(inputs.len(), 1);
        assert_eq!(output_tangents.len(), 1);
        let op = LeftMulOp { coefficient: self.coefficient.clone().into() };
        let inputs = vec![inputs[0]];
        let outputs = (Box::new(op) as Box<dyn LinearOp<T, V>>).trace(inputs.as_slice()).unwrap();
        assert_eq!(outputs.len(), 1);
        outputs
    }
}

impl<
    T: Clone + ArrayStructureType,
    L: Clone + Display + Into<V> + Typed<T>,
    V: Clone + Display + Mul<Output = V> + Typed<T> + 'static,
> LinearInterpretableOp<T, V> for LeftMulOp<L>
{
}

// TODO(eaplatanios): It would be nice to have something like the following instead of the multiple subsequent
//  implementations but unfortunately the orphan rule prevents us from doing that.
// macro_rules! impl_left_mul_for_traceable {
//     ($O:path) => {
//         impl<
//                 T: Clone + ArrayStructureType,
//                 V: Clone + Display + Typed<T> + Mul<Output = V> + 'static,
//                 TR: Typed<T> + Into<V> + Traceable<Value = V>,
//             > Mul<Tracer<T, V, Box<dyn $O>>> for TR
//         {
//             type Output = Tracer<T, V, Box<dyn $O>>;

//             fn mul(self, rhs: Tracer<T, V, Box<dyn $O>>) -> Self::Output {
//                 let op = LeftMulOp { coefficient: self.into() };
//                 let inputs = vec![&rhs];
//                 let outputs = (Box::new(op) as Box<dyn $O>).trace(inputs.as_slice()).unwrap();
//                 debug_assert_eq!(outputs.len(), 1);
//                 outputs[0].clone()
//             }
//         }
//     };
// }

// // impl_left_mul_for_traceable!(Op<T>);
// // impl_left_mul_for_traceable!(InterpretableOp<T, $V>);
// impl_left_mul_for_traceable!(LinearOp<T, V>);
// impl_left_mul_for_traceable!(LinearInterpretableOp<T, V>);

macro_rules! impl_left_mul_for_scalar {
    ($V:path, $O:path) => {
        impl<T: Clone + ArrayStructureType, V: Clone + Display + Typed<T> + Mul<Output = V> + 'static>
            Mul<Tracer<T, V, Box<dyn $O>>> for $V
        where
            $V: Typed<T> + Into<V>,
        {
            type Output = Tracer<T, V, Box<dyn $O>>;

            fn mul(self, rhs: Tracer<T, V, Box<dyn $O>>) -> Self::Output {
                let op = LeftMulOp { coefficient: self.into() };
                let inputs = vec![&rhs];
                let outputs = (Box::new(op) as Box<dyn $O>).trace(inputs.as_slice()).unwrap();
                debug_assert_eq!(outputs.len(), 1);
                outputs[0].clone()
            }
        }
    };
}

macro_rules! impl_left_mul_linear_op_for_scalar {
    ($V:path) => {
        // impl_left_mul_for_scalar!($V, Op<T>);
        // impl_left_mul_for_scalar!($V, InterpretableOp<T, $V>);
        impl_left_mul_for_scalar!($V, LinearOp<T, V>);
        impl_left_mul_for_scalar!($V, LinearInterpretableOp<T, V>);
    };
}

macro_rules! impl_tracer_left_mul_for_scalars {
    ($($T:ty),* $(,)*) => {$(
        impl_left_mul_linear_op_for_scalar!($T);
    )*};
}

impl_tracer_left_mul_for_scalars!(i8, i16, i32, i64, u8, u16, u32, u64, bf16, f16, f32, f64);

// TODO(eaplatanios): Implement `RightMul` similarly.
// JvpTracer<V, T> * Tracer<JvpTracer<V, T>> => Tracer<JvpTracer<V, T>>
macro_rules! impl_left_mul_for_jvp_tracer {
    ($op:path) => {
        impl<
            T: Clone + ArrayStructureType,
            VL: Into<VR>,
            VTL: Into<VTR>,
            VR: Clone + Display + Typed<T> + Mul<Output = VR> + Mul<VTR, Output = VTR> + 'static,
            VTR: Clone + Display + Typed<T> + Add<Output = VTR> + Mul<VR, Output = VTR> + 'static,
        > Mul<Tracer<JvpTracer<T, T>, JvpTracer<VR, VTR>, Box<dyn $op>>> for JvpTracer<VL, VTL>
        {
            type Output = Tracer<JvpTracer<T, T>, JvpTracer<VR, VTR>, Box<dyn $op>>;

            fn mul(self, rhs: Tracer<JvpTracer<T, T>, JvpTracer<VR, VTR>, Box<dyn $op>>) -> Self::Output {
                let op =
                    LeftMulOp { coefficient: JvpTracer { value: self.value.into(), tangent: self.tangent.into() } };
                let inputs = vec![&rhs];
                let outputs = (Box::new(op) as Box<dyn $op>).trace(inputs.as_slice()).unwrap();
                debug_assert_eq!(outputs.len(), 1);
                outputs[0].clone()
            }
        }
    };
}

impl_left_mul_for_jvp_tracer!(Op<JvpTracer<T, T>>);
impl_left_mul_for_jvp_tracer!(InterpretableOp<JvpTracer<T, T>, JvpTracer<VR, VTR>>);
impl_left_mul_for_jvp_tracer!(LinearOp<JvpTracer<T, T>, JvpTracer<VR, VTR>>);
impl_left_mul_for_jvp_tracer!(LinearInterpretableOp<JvpTracer<T, T>, JvpTracer<VR, VTR>>);
