//! Legacy v0 automatic differentiation and related type utilities.

pub mod r#type;

use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    rc::Rc,
};

use half::{bf16, f16};

use ryft_macros::Parameter;

use crate::{
    broadcasting::{Broadcastable, BroadcastingError},
    ops::constants::{One, Zero},
    parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily},
    programs::{LinearInterpretableOp, LinearOp, ParameterizedProgram, ProgramBuilder},
    tracing_v0::{Tracer, VariableTracer},
    types::{ArrayType, Type, Typed},
};

// How do we handle things like `grad(lambda x: x**2 if x > 0 else 0.)`? In this case, we need to be able to keep the
// primals as known values and the tangents as staged computations that may depend on some known values.
// This would require [PartialOrd] and [PartialEq] implementations on [PartialValue] which I think is probably ok.
// https://en.wikipedia.org/wiki/Differentiable_manifold
//
// A type that can be used to represent derivatives with respect to a value whose type is `Self`. Mathematically,
// this is equivalent to the tangent bundle of the differentiable manifold represented by the differentiable type.
// https://en.wikipedia.org/wiki/Tangent_bundle
pub trait Differentiable<Tangent> {}

macro_rules! impl_differentiable_for_scalar {
    ($ty:ty) => {
        impl Differentiable<$ty> for $ty {}

        impl<O> Differentiable<Tracer<ArrayType, $ty, O>> for $ty {}
    };
}

impl_differentiable_for_scalar!(bf16);
impl_differentiable_for_scalar!(f16);
impl_differentiable_for_scalar!(f32);
impl_differentiable_for_scalar!(f64);

// TODO(eaplatanios): Do we need this to be `JvpTracer<Tangent, Tangent>` instead?
impl<Value, Tangent> Differentiable<JvpTracer<Value, Tangent>> for JvpTracer<Value, Tangent> {}

// TODO(eaplatanios): Do we need a `Typed<T>` bound here?
impl<T, Value, Tangent, O> Differentiable<Tracer<T, JvpTracer<Value, Tangent>, O>> for JvpTracer<Value, Tangent> {}

impl<T, V: Differentiable<Tangent>, O, Tangent> Differentiable<Tracer<T, Tangent, O>> for Tracer<T, V, O> {}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Parameter)]
pub struct JvpTracer<Value, Tangent> {
    pub value: Value,
    pub tangent: Tangent,
}

impl<Value: Broadcastable, Tangent: Broadcastable> Broadcastable for JvpTracer<Value, Tangent> {
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError> {
        Ok(Self { value: self.value.broadcast(&other.value)?, tangent: self.tangent.broadcast(&other.tangent)? })
    }

    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError> {
        Ok(Self { value: self.value.broadcast_to(&other.value)?, tangent: self.tangent.broadcast_to(&other.tangent)? })
    }

    fn is_broadcastable_to(&self, other: &Self) -> bool {
        self.value.is_broadcastable_to(&other.value) && self.tangent.is_broadcastable_to(&other.tangent)
    }
}

impl<Value: Display, Tangent: Display> Display for JvpTracer<Value, Tangent> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{{ value: {}, tangent: {} }}", self.value, self.tangent)
    }
}

impl<Value: Differentiable<Tangent> + Typed<Tangent::T>, Tangent: Zero> From<Value> for JvpTracer<Value, Tangent> {
    fn from(value: Value) -> Self {
        let tpe = value.tpe();
        Self { value, tangent: Tangent::zero(&tpe) }
    }
}

// TODO(eaplatanios): Can we avoid private helpers like this? Here the issue is inferring
//  `InputTangent` and `OutputTangent` when not provided.
#[inline]
fn _jvp<
    V: Parameter,
    VT: Parameter,
    Input: Parameterized<V, To<JvpTracer<V, VT>> = InputJvpTracer, Family: ParameterizedFamily<JvpTracer<V, VT>>>,
    InputTangent: Parameterized<VT, ParameterStructure = Input::ParameterStructure>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    OutputTangent: Parameterized<VT, ParameterStructure = Output::ParameterStructure>,
    InputJvpTracer: Parameterized<JvpTracer<V, VT>, ParameterStructure = Input::ParameterStructure>,
    OutputJvpTracer: Parameterized<
            JvpTracer<V, VT>,
            To<V> = Output,
            ParameterStructure = Output::ParameterStructure,
            Family: ParameterizedFamily<V>,
        >,
    F: FnOnce(InputJvpTracer) -> OutputJvpTracer,
>(
    function: F,
    value: Input,
    tangent: InputTangent,
) -> Result<(Output, OutputTangent), ParameterError> {
    let structure = value.parameter_structure();
    let input_values = value.into_parameters();
    let input_tangents = tangent.into_parameters();
    let tracers = input_values.zip(input_tangents).map(|(value, tangent)| JvpTracer { value, tangent });
    let input_tracer = InputJvpTracer::from_parameters(structure, tracers)?;
    let output_tracer = function(input_tracer);
    let output_structure = output_tracer.parameter_structure();
    let output_params = output_tracer.into_parameters();
    let (output_values, output_tangents): (Vec<V>, Vec<VT>) = output_params.map(|o| (o.value, o.tangent)).unzip();
    let output_value = Output::from_parameters(output_structure.clone(), output_values)?;
    let output_tangent = OutputTangent::from_parameters(output_structure, output_tangents)?;
    Ok((output_value, output_tangent))
}

#[inline]
pub fn jvp<
    V: Parameter,
    Input: Parameterized<V, To<JvpTracer<V, V>> = InputJvpTracer, Family: ParameterizedFamily<JvpTracer<V, V>>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    InputJvpTracer: Parameterized<JvpTracer<V, V>, ParameterStructure = Input::ParameterStructure>,
    OutputJvpTracer: Parameterized<
            JvpTracer<V, V>,
            To<V> = Output,
            ParameterStructure = Output::ParameterStructure,
            Family: ParameterizedFamily<V>,
        >,
    F: FnOnce(InputJvpTracer) -> OutputJvpTracer,
>(
    function: F,
    value: Input,
    tangent: Input,
) -> Result<(Output, Output), ParameterError> {
    _jvp(function, value, tangent)
}

#[inline]
pub fn _differential<
    V: Parameter,
    VT: Parameter,
    Input: Parameterized<V, To<JvpTracer<V, VT>> = InputJvpTracer, Family: ParameterizedFamily<JvpTracer<V, VT>>>
        + Typed<InputTangent::T>,
    InputTangent: Parameterized<VT, ParameterStructure = Input::ParameterStructure> + One,
    Output: Parameterized<V, ParameterStructure: Clone>,
    OutputTangent: Parameterized<VT, ParameterStructure = Output::ParameterStructure>,
    InputJvpTracer: Parameterized<JvpTracer<V, VT>, ParameterStructure = Input::ParameterStructure>,
    OutputJvpTracer: Parameterized<
            JvpTracer<V, VT>,
            To<V> = Output,
            ParameterStructure = Output::ParameterStructure,
            Family: ParameterizedFamily<V>,
        >,
    F: FnOnce(InputJvpTracer) -> OutputJvpTracer,
>(
    function: F,
) -> impl FnOnce(Input) -> OutputTangent {
    |value| {
        let tpe = value.tpe();
        _jvp(function, value, InputTangent::one(&tpe)).map(|(_, tangent)| tangent).unwrap()
    }
}

#[inline]
pub fn differential<
    V: Parameter,
    Input: Parameterized<V, To<JvpTracer<V, V>> = InputJvpTracer, Family: ParameterizedFamily<JvpTracer<V, V>>> + One,
    Output: Parameterized<V, ParameterStructure: Clone>,
    InputJvpTracer: Parameterized<JvpTracer<V, V>, ParameterStructure = Input::ParameterStructure>,
    OutputJvpTracer: Parameterized<
            JvpTracer<V, V>,
            To<V> = Output,
            ParameterStructure = Output::ParameterStructure,
            Family: ParameterizedFamily<V>,
        >,
    F: FnOnce(InputJvpTracer) -> OutputJvpTracer,
>(
    function: F,
) -> impl FnOnce(Input) -> Output {
    _differential::<V, V, Input, Input, Output, Output, _, _, _>(function)
}

// TODO(eaplatanios): Can we avoid private helpers like this? Here the issue is inferring `O` when not provided.
// TODO(eaplatanios): Can we make [ParameterizedProgram] callable like a function?
#[inline]
pub fn _linearize<
    T: Clone + Display + Debug + Type,
    O: Clone + LinearOp<T, V> + Debug,
    V: Parameter + Typed<T>,
    VT: Parameter + Clone + Typed<T>,
    Input: Parameterized<
            V,
            ParameterStructure: Clone,
            To<JvpTracer<V, Tracer<T, VT, O>>> = InputJvpTracer,
            Family: ParameterizedFamily<JvpTracer<V, Tracer<T, VT, O>>>,
        >,
    InputTangent: Parameterized<VT, ParameterStructure = Input::ParameterStructure, Family: ParameterizedFamily<Tracer<T, VT, O>>>,
    Output: Parameterized<V, ParameterStructure: Clone>,
    OutputTangent: Parameterized<VT, ParameterStructure = Output::ParameterStructure, Family: ParameterizedFamily<Tracer<T, VT, O>>>,
    InputJvpTracer: Parameterized<JvpTracer<V, Tracer<T, VT, O>>, ParameterStructure = Input::ParameterStructure>,
    OutputJvpTracer: Parameterized<
            JvpTracer<V, Tracer<T, VT, O>>,
            To<V> = Output,
            ParameterStructure = Output::ParameterStructure,
            Family: ParameterizedFamily<V>,
        >,
    F: FnOnce(InputJvpTracer) -> OutputJvpTracer,
>(
    function: F,
    input: Input,
) -> Result<(Output, ParameterizedProgram<T, VT, O, InputTangent, OutputTangent>), ParameterError> {
    let mut program_builder = ProgramBuilder::<T, VT, O>::new();
    let input_structure = input.parameter_structure();
    let input_tangent_ids = input.parameters().map(|v| program_builder.add_variable(v.tpe())).collect::<Vec<_>>();
    let program_builder = Rc::new(RefCell::new(program_builder));
    let (output, output_tangent_structure, output_tangent_ids) = {
        // The scoping here is to ensure that all references to `program_builder` are dropped before the subsequent
        // [Rc::try_unwrap]. Note that this only ensures this for references that are created in this function.
        // However, if `function` somehow ends up creating more references that remain alive after this block,
        // the subsequent [Rc::try_unwrap] could still fail resulting in an error.
        let input_tangent = InputTangent::To::<Tracer<T, VT, O>>::from_parameters(
            input_structure.clone(),
            input_tangent_ids
                .iter()
                .map(|id| Tracer::Variable(VariableTracer { id: *id, builder: program_builder.clone() })),
        )?;
        let (output, output_tangent): (Output, OutputTangent::To<Tracer<T, VT, O>>) =
            _jvp(function, input, input_tangent)?;
        let output_tangent_structure = output_tangent.parameter_structure();
        let mut mut_builder = program_builder.borrow_mut();
        let output_tangent_ids = output_tangent
            .parameters()
            .map(|tracer| match &tracer {
                Tracer::Constant(constant) => mut_builder.add_constant_expression(constant.clone()).unwrap(),
                Tracer::Variable(VariableTracer { id, .. }) => *id,
            })
            .collect::<Vec<_>>();
        (output, output_tangent_structure, output_tangent_ids)
    };
    // TODO(eaplatanios): Convert the unwraps to errors.
    let program_builder = Rc::try_unwrap(program_builder).ok().unwrap().into_inner();
    let program = program_builder.build(input_tangent_ids, output_tangent_ids).unwrap();
    let program = ParameterizedProgram::new(program, input_structure, output_tangent_structure);
    Ok((output, program))
}

#[inline]
pub fn linearize<
    T: Clone + Display + Debug + Type,
    V: Parameter + Clone + Typed<T>,
    Input: Parameterized<
            V,
            ParameterStructure: Clone,
            To<JvpTracer<V, Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>> = InputJvpTracer,
            Family: ParameterizedFamily<JvpTracer<V, Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>>
                        + ParameterizedFamily<Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>,
        >,
    Output: Parameterized<
            V,
            ParameterStructure: Clone,
            Family: ParameterizedFamily<Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>,
        >,
    InputJvpTracer: Parameterized<
            JvpTracer<V, Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>,
            ParameterStructure = Input::ParameterStructure,
        >,
    OutputJvpTracer: Parameterized<
            JvpTracer<V, Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>,
            To<V> = Output,
            ParameterStructure = Output::ParameterStructure,
            Family: ParameterizedFamily<V>,
        >,
    F: FnOnce(InputJvpTracer) -> OutputJvpTracer,
>(
    function: F,
    input: Input,
) -> Result<(Output, ParameterizedProgram<T, V, Box<dyn LinearInterpretableOp<T, V>>, Input, Output>), ParameterError> {
    _linearize(function, input)
}

#[inline]
pub fn linear<
    T: Clone + Display + Debug + Type,
    V: Parameter + Clone + ToOwned<Owned = V> + Typed<T>,
    Input: Parameterized<
            V,
            ParameterStructure: Clone,
            To<JvpTracer<V, Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>> = InputJvpTracer,
            Family: ParameterizedFamily<JvpTracer<V, Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>>
                        + ParameterizedFamily<Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>,
        > + One,
    Output: Parameterized<
            V,
            ParameterStructure: Clone,
            Family: ParameterizedFamily<Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>,
        >,
    InputJvpTracer: Parameterized<
            JvpTracer<V, Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>,
            ParameterStructure = Input::ParameterStructure,
        >,
    OutputJvpTracer: Parameterized<
            JvpTracer<V, Tracer<T, V, Box<dyn LinearInterpretableOp<T, V>>>>,
            To<V> = Output,
            ParameterStructure = Output::ParameterStructure,
            Family: ParameterizedFamily<V>,
        >,
    F: FnOnce(InputJvpTracer) -> OutputJvpTracer,
>(
    function: F,
) -> impl FnOnce(Input) -> Output {
    |value| {
        let tpe = value.tpe();
        _linearize(function, Input::one(&tpe)).unwrap().1.interpret(value).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::broadcasting::{Broadcastable, BroadcastingError};
    use crate::ops::{
        constants::Constant,
        trigonometric::{Cos, Sin},
    };
    use crate::types::data_types::DataType::*;
    use crate::types::{ArrayType, Shape, Size};

    use super::*;

    use std::ops::{Add, Mul, Neg};

    // ======================================================= FUNCTIONS =======================================================

    fn x_times_two<T: Constant<f32> + Mul<Output = T>>(x: T) -> T {
        x * T::constant(2f32)
    }

    fn two_times_x<T: Constant<f32> + Mul<Output = T>>(x: T) -> T {
        T::constant(2f32) * x
    }

    fn x_plus_x<T: Clone + Add<T, Output = T>>(x: T) -> T {
        x.clone() + x
    }

    fn x_times_x<T: Clone + Mul<T, Output = T>>(x: T) -> T {
        x.clone() * x
    }

    fn x_0_plus_x_1<T: Add<T, Output = T>>(inputs: (T, T)) -> T {
        inputs.0 + inputs.1
    }

    // fn x_plus_y<T: Add<T, Output = T>>(x: T, y: T) -> T {
    //     x + y
    // }

    fn x_plus_x_plus_y<T: Clone + Add<T, Output = T>>(inputs: (T, T)) -> T {
        inputs.0.clone() + inputs.0 + inputs.1
    }

    fn sin<T: Sin>(x: T) -> T {
        x.sin()
    }

    fn minus_sin<T: Neg<Output = T> + Sin>(x: T) -> T {
        -x.sin()
    }

    fn sin_plus_cos_times_sin<T: Clone + Sin + Cos + Add<Output = T> + Mul<Output = T>>(x: T) -> T {
        x.clone().sin() + x.clone().cos() * x.sin()
    }

    #[test]
    fn test_jvp_tracer_broadcastable_broadcast() {
        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, Shape::new(vec![10.into(), 5.into()]));
        let t2 = ArrayType::new(F32, Shape::new(vec![1.into(), 5.into()]));
        let t3 = ArrayType::new(F32, Shape::new(vec![10.into(), 1.into()]));
        let t4 = ArrayType::new(F32, Shape::new(vec![10.into(), 5.into()]));
        let t5 = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));
        let t6 = ArrayType::new(F32, Shape::new(vec![4.into(), 2.into()]));

        let j0 = JvpTracer { value: t1.clone(), tangent: t2.clone() };
        let j1 = JvpTracer { value: t0.clone(), tangent: t3.clone() };
        let j2 = JvpTracer { value: t1.clone(), tangent: t4.clone() };
        let j3 = JvpTracer { value: t5.clone(), tangent: t6.clone() };

        assert_eq!(JvpTracer::<ArrayType, ArrayType>::broadcasted(&[&j0]), Ok(j0.clone()));
        assert_eq!(JvpTracer::<ArrayType, ArrayType>::broadcasted(&[&j0, &j1]), Ok(j2));

        assert!(matches!(
            JvpTracer::<ArrayType, ArrayType>::broadcasted(&[]),
            Err(BroadcastingError::EmptyBroadcastingInput)
        ));
        assert!(matches!(
            JvpTracer::<ArrayType, ArrayType>::broadcasted(&[&j0, &j3]),
            Err(BroadcastingError::IncompatibleShapes { .. }),
        ));
    }

    #[test]
    fn test_nested_jvp_tracer_broadcastable_broadcast() {
        type NestedJvpTracer =
            JvpTracer<JvpTracer<ArrayType, ArrayType>, JvpTracer<ArrayType, JvpTracer<ArrayType, ArrayType>>>;

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into(), 2.into()]));
        let t2 = ArrayType::new(BF16, Shape::new(vec![4.into(), 1.into()]));
        let t3 = ArrayType::new(F16, Shape::new(vec![4.into(), Size::Dynamic(Some(1))]));
        let t4 = ArrayType::new(C64, Shape::new(vec![Size::Dynamic(None), 42.into(), Size::Dynamic(None)]));
        let t5 = ArrayType::new(BF16, Shape::new(vec![42.into(), Size::Dynamic(None)]));
        let t6 = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into(), 2.into()]));
        let t7 = ArrayType::new(BF16, Shape::new(vec![1.into(), 1.into()]));
        let t8 = ArrayType::new(F16, Shape::new(vec![4.into(), Size::Dynamic(Some(1))]));
        let t9 = ArrayType::new(C64, Shape::new(vec![1.into(), 42.into(), 1.into()]));
        let t11 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into(), 2.into()]));
        let t12 = ArrayType::new(BF16, Shape::new(vec![4.into(), 1.into()]));
        let t13 = ArrayType::new(F16, Shape::new(vec![4.into(), Size::Dynamic(Some(1))]));
        let t14 = ArrayType::new(C64, Shape::new(vec![Size::Dynamic(None), 42.into(), Size::Dynamic(None)]));
        let t15 = ArrayType::new(BF16, Shape::new(vec![42.into(), Size::Dynamic(None)]));
        let t17 = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));

        let j0 = NestedJvpTracer {
            value: JvpTracer { value: t0.clone(), tangent: t1.clone() },
            tangent: JvpTracer { value: t2.clone(), tangent: JvpTracer { value: t3.clone(), tangent: t4.clone() } },
        };
        let j1 = NestedJvpTracer {
            value: JvpTracer { value: t5.clone(), tangent: t6.clone() },
            tangent: JvpTracer { value: t7.clone(), tangent: JvpTracer { value: t8.clone(), tangent: t9.clone() } },
        };
        let j2 = NestedJvpTracer {
            value: JvpTracer { value: t15.clone(), tangent: t11.clone() },
            tangent: JvpTracer { value: t12.clone(), tangent: JvpTracer { value: t13.clone(), tangent: t14.clone() } },
        };
        let j3 = NestedJvpTracer {
            value: JvpTracer { value: t15.clone(), tangent: t17.clone() },
            tangent: JvpTracer { value: t12.clone(), tangent: JvpTracer { value: t13.clone(), tangent: t14.clone() } },
        };

        assert_eq!(NestedJvpTracer::broadcasted(&[&j0]), Ok(j0.clone()));
        assert_eq!(NestedJvpTracer::broadcasted(&[&j0, &j1]), Ok(j2));
        assert!(
            matches!(NestedJvpTracer::broadcasted(&[&j0, &j3]), Err(BroadcastingError::IncompatibleShapes { .. }),)
        );
    }

    #[test]
    fn test_jvp() {
        // minus_sin:
        // { lambda a:float64[] .
        //   let b:float64[] = sin a
        //       c:float64[] = neg b
        //   in ( c ) }
        //
        // JVP:
        // { lambda a:float64[] b:float64[] .
        //   let c:float64[] = sin a
        //       d:float64[] = cos a
        //       e:float64[] = mul d b
        //       f:float64[] = neg c
        //       g:float64[] = neg e
        //   in ( f, g ) }
        //
        // Primal:
        // { lambda a:float64[] .
        //   let c:float64[] = sin a
        //       d:float64[] = cos a
        //       f:float64[] = neg c
        //   in ( f, d ) }
        //
        // Tangent:
        // { lambda d:float64[] b:float64[] .
        //   let e:float64[] = mul d b
        //       g:float64[] = neg e
        //   in ( g ) }
        //
        // At a = 3, we have d = cos(a) = -0.9899925.
        //
        let (z, z_tangent) = jvp(x_plus_x, 2f32, 1f32).unwrap();
        println!("## {z} | {z_tangent}");

        let (z, z_tangent) = jvp(x_0_plus_x_1, (2f32, 4f32), (1f32, 1f32)).unwrap();
        println!("## {z} | {z_tangent}");

        // let (z, z_tangent) = jvp(x_plus_y, (2f32, 4f32), (1f32, 1f32)).unwrap();
        // println!("## {z} | {z_tangent}");

        let (z, z_tangent) = jvp(x_plus_x_plus_y, (2f32, 4f32), (1f32, 1f32)).unwrap();
        println!("## {z} | {z_tangent}");

        let differential_1 = differential(sin_plus_cos_times_sin)(3f32);
        println!("## @3f32 => {differential_1}");

        let differential_2 = differential(differential(sin_plus_cos_times_sin))(3f32);
        println!("## @3f32 => {differential_2}");

        let differential_3 = differential(differential(differential(sin_plus_cos_times_sin)))(3f32);
        println!("## @3f32 => {differential_3}");

        let differential_4 = differential(differential(differential(differential(sin_plus_cos_times_sin))))(3f32);
        println!("## @3f32 => {differential_4}");

        let differential_1 = differential(sin_plus_cos_times_sin)(3f64);
        println!("## @3f64 => {differential_1}");

        let differential_2 = differential(differential(sin_plus_cos_times_sin))(3f64);
        println!("## @3f64 => {differential_2}");

        let differential_3 = differential(differential(differential(sin_plus_cos_times_sin)))(3f64);
        println!("## @3f64 => {differential_3}");

        let differential_4 = differential(differential(differential(differential(sin_plus_cos_times_sin))))(3f64);
        println!("## @3f64 => {differential_4}");

        let (y, d) = linearize(x_times_two, 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(two_times_x, 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(sin, 3f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(minus_sin, 3f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(sin_plus_cos_times_sin, 3f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(differential(sin_plus_cos_times_sin), 3f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(differential(differential(differential(sin_plus_cos_times_sin))), 3f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(x_plus_x, 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(linear(differential(x_plus_x)), 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(x_times_x, 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let f = linear(differential(x_times_x));
        let y = f(1f32);
        println!("{y}");

        print!("{}", linearize(differential(x_times_x), 2f32).unwrap().1.program);
        // lambda %0:f32[] .
        // let %1 = LeftMul[2] %0
        //     %2 = RightMul[2] %0
        //     %3 = Add %1 %2
        //     %4 = RightMul[1] %0
        //     %6 = LeftMul[2] %5
        //     %7 = Add %6 %4
        //     %8 = LeftMul[1] %0
        //     %10 = RightMul[2] %9
        //     %11 = Add %8 %10
        //     %12 = Add %7 %11
        // in (%12)

        let (y, d) = linearize(linear(differential(x_times_x)), 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) = linearize(linear(differential(sin_plus_cos_times_sin)), 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        let (y, d) =
            linearize(linear(linear(differential(differential(differential(sin_plus_cos_times_sin))))), 1f32).unwrap();
        let dy = d.interpret(1f32).unwrap();
        println!("{y} || {dy} ||\n{}", d.program);

        // TODO(eaplatanios): How can we test with a function like `|x| 3f32 * x`?
    }
}
