use std::{
    cell::RefCell,
    fmt::{Debug, Display},
    rc::Rc,
};

use half::{bf16, f16};

use ryft_macros::Parameter;

use crate::{
    errors::Error,
    ops::constants::{One, Zero},
    parameters::{Parameter, Parameterized, ParameterizedFamily},
    programs::{LinearInterpretableOp, LinearOp, ParameterizedProgram, ProgramBuilder},
    tracing::{Tracer, VariableTracer},
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

impl<Value: Display, Tangent: Display> Display for JvpTracer<Value, Tangent> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{ value: {}, tangent: {} }}", self.value, self.tangent)
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
) -> Result<(Output, OutputTangent), Error> {
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
) -> Result<(Output, Output), Error> {
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
    T: Clone + Debug + Type,
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
) -> Result<(Output, ParameterizedProgram<T, VT, O, InputTangent, OutputTangent>), Error> {
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
    T: Clone + Debug + Type,
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
) -> Result<(Output, ParameterizedProgram<T, V, Box<dyn LinearInterpretableOp<T, V>>, Input, Output>), Error> {
    _linearize(function, input)
}

#[inline]
pub fn linear<
    T: Clone + Debug + Type,
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
    use crate::ops::{
        constants::Constant,
        trigonometric::{Cos, Sin},
    };

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
