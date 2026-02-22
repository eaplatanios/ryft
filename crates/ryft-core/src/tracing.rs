use std::{cell::RefCell, fmt::Display, rc::Rc};

use half::{bf16, f16};
use ryft_macros::Parameter;

use crate::{
    differentiation::JvpTracer,
    parameters::Parameter,
    programs::{AtomId, ConstantExpression, Op, ProgramBuilder, ProgramError},
    types::Typed,
};

// TODO(eaplatanios): Constant folding is unavoidable but can be expensive (e.g., `Tensor @ Tensor`). Can we trace it?
//  What if we introduced a way of handling nullary ops? E.g., how could we represent `ryft.zeros()`? There is no
//  builder yet, but maybe that can be supported by having a special tracer that holds just a nullary op.
//  But then, what do you do when you add the results of two nullary ops? That needs a builder. This is starting to
//  look more like a `ConstantExpressionTracer` or something like that. And then the actual tracer is an enum that is
//  is either a constant expression tracer or one that holds a program builder.
//  Maybe we just have a bunch of built-in nullary/constant ops which do not materialize values until combined
//  with another materialized value (or like also have a `.materialize()` function or something). Examples include:
//    - Zeros
//    - Ones
//    - Fill
//    - RandomUniform
//    - RandomNormal
//    - etc.
//  This will allow us to combine them with [VariableTracer]s (now just [Tracer]s?) without having materialized them.
//  Now say you have a [ryft::zeros()] function. What does this return and do we need to support a million operation
//  combinations for it? Maybe all this needs is some kind of "lazy tensor" abstraction that only supports some
//  operations before you have to make it concrete. And then, once you make it concrete, how does it interact with
//  other values and how does it integrate with our interpreter stack design/implementation.

#[derive(Clone)]
pub struct VariableTracer<T, V, O> {
    pub id: AtomId,
    pub builder: Rc<RefCell<ProgramBuilder<T, V, O>>>,
}

impl<T, V, O> Display for VariableTracer<T, V, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.id)
    }
}

#[derive(Clone, Parameter)]
pub enum Tracer<T, V, O> {
    Constant(ConstantExpression<T, V, O>),
    Variable(VariableTracer<T, V, O>),
}

impl<T, V, O> Tracer<T, V, O> {
    #[inline]
    pub fn constant(&self) -> Option<&ConstantExpression<T, V, O>> {
        match self {
            Tracer::Constant(constant) => Some(constant),
            Tracer::Variable(_) => None,
        }
    }

    #[inline]
    pub fn variable(&self) -> Option<&VariableTracer<T, V, O>> {
        match self {
            Tracer::Constant(_) => None,
            Tracer::Variable(variable) => Some(variable),
        }
    }
}

impl<T, V: Display, O: Display> Display for Tracer<T, V, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Tracer::Constant(constant) => write!(f, "{constant}"),
            Tracer::Variable(variable) => write!(f, "{variable}"),
        }
    }
}

impl<T: Clone, V: Typed<T>, O> From<V> for Tracer<T, V, O> {
    fn from(value: V) -> Self {
        Tracer::Constant(ConstantExpression::new_value(value))
    }
}

pub trait TraceableOp<T>: Sized {
    fn trace<V: Clone + Display + Typed<T>>(
        &self,
        inputs: &[&Tracer<T, V, Self>],
    ) -> Result<Vec<Tracer<T, V, Self>>, ProgramError>;
}

impl<T: Clone, O: Sized + Clone + Op<T>> TraceableOp<T> for O {
    fn trace<V: Clone + Display + Typed<T>>(
        &self,
        inputs: &[&Tracer<T, V, Self>],
    ) -> Result<Vec<Tracer<T, V, Self>>, ProgramError> {
        // TODO(eaplatanios): Document how/why we extract the builder first.
        let builder = inputs.iter().fold::<Result<Option<Rc<RefCell<ProgramBuilder<T, V, O>>>>, ProgramError>, _>(
            Ok(None),
            |builder, tracer| {
                match builder {
                    Ok(builder) => {
                        match tracer {
                            Tracer::Constant(_) => Ok(builder),
                            Tracer::Variable(VariableTracer { builder: tracer_builder, .. }) => {
                                match builder {
                                    None => Ok(Some(tracer_builder.clone())),
                                    Some(builder) => {
                                        // We are using a debug assertion here because performance is important and
                                        // users should generally make sure that they are using the built-in ryft
                                        // tracing functionality instead of trying to build out their own. They can
                                        // still do the latter if they want, but then they would need to worry about
                                        // error handling.
                                        debug_assert!(Rc::ptr_eq(&builder, tracer_builder));
                                        Ok(Some(builder))
                                    }
                                }
                            }
                        }
                    }
                    Err(_) => builder,
                }
            },
        )?;
        match builder {
            None => {
                // At this point, we know that all input [Tracer]s are [Tracer::Constant]s and so we want to extract
                // the corresponding expressions and construct a new [Tracer::Constant] for the output.
                Ok(vec![Tracer::Constant(ConstantExpression::new_expression(
                    self.clone(),
                    inputs
                        .into_iter()
                        .map(|input| match input {
                            Tracer::Constant(constant) => constant.clone(),
                            Tracer::Variable(_) => unsafe { std::hint::unreachable_unchecked() },
                        })
                        .collect(),
                ))])
            }
            Some(builder) => {
                // At this point, we know that there is at least one [Tracer::Variable] in the inputs and so want to
                // add any [Tracer::Constant]s to the builder, getting [AtomId]s for them, before adding an expression
                // for this [TraceableOp].
                let mut mut_builder = builder.try_borrow_mut().unwrap();
                let input_ids = inputs
                    .into_iter()
                    .map(|input| match input {
                        Tracer::Constant(constant) => mut_builder.add_constant_expression(constant.clone()),
                        Tracer::Variable(variable) => Ok(variable.id),
                    })
                    .collect::<Result<Vec<AtomId>, ProgramError>>()?;
                let output_ids = mut_builder.add_expression(self.clone(), input_ids)?;
                Ok(output_ids
                    .into_iter()
                    .map(|output_id| Tracer::Variable(VariableTracer { id: output_id, builder: builder.clone() }))
                    .collect())
            }
        }
    }
}

pub trait Traceable {
    type Value;
}

macro_rules! impl_traceable_for_primitive {
    ($T:ty) => {
        impl Traceable for $T {
            type Value = $T;
        }
    };
}

macro_rules! impl_traceable_for_primitives {
    ($($T:ty),* $(,)*) => {$(
        impl_traceable_for_primitive!($T);
    )*};
}

impl_traceable_for_primitives!(bool, i8, i16, i32, i64, u8, u16, u32, u64, bf16, f16, f32, f64);

impl<Value, Tangent> Traceable for JvpTracer<Value, Tangent> {
    type Value = JvpTracer<Value, Tangent>;
}

// #[macro_export]
// macro_rules! get_program_builder_v2 {
//     ([]) => { None };
//     ([$inputs_head:tt $(,$inputs_tail:tt)*]) => {
//         {
//             match &$inputs_head {
//                 Tracer::Constant(constant) => {
//                     crate::get_program_builder!([$($inputs_tail),*])
//                 },
//                 Tracer::Variable(crate::tracing::VariableTracer { id, builder }) => {
//                     crate::get_program_builder!(builder, [$($inputs_tail,)*])
//                 },
//             }
//         }
//     };
//     ($program_builder:ident, [$inputs_head:tt, $($inputs_tail:tt,)*]) => {
//         {
//             match &$inputs_head {
//                 Tracer::Constant(constant) => {
//                     crate::get_program_builder!($program_builder, [$($inputs_tail,)*])
//                 },
//                 Tracer::Variable(crate::tracing::VariableTracer { id, builder }) => {
//                     // We are using a debug assertion here because performance is important and users should generally
//                     // make sure that they are using the built-in ryft tracing functionality instead of trying to build
//                     // out their own. They can still do the latter if they want, but then they would need to worry
//                     // about error handling.
//                     debug_assert!(Rc::ptr_eq(&$program_builder, builder));
//                     crate::get_program_builder!($program_builder, [$($inputs_tail,)*])
//                 },
//             }
//         }
//     };
//     ($program_builder:ident, []) => {
//         Some($program_builder)
//     };
// }

// We need to handle:
//   - all constants
//   - all variables
//   - mixed constants and variables
// Carry forward a builder.
//   If missing, keep collecting all constant expressions so far.
//   Once found, convert all constant expressions.
//   Keep track of two parallel lists: constant_expressions and variable_ids, along with an optional program builder.

// TODO(eaplatanios): Returns vector of output tracers.
// #[macro_export]
// macro_rules! add_expression_to_builder_v2 {
//     ($op:expr, inputs = [$($inputs:tt),*] $(,)?) => { crate::add_expression_to_builder_v2!($op, inputs = [$($inputs),*], constants = [], variables = []) };
//     ($op:expr, inputs = [$inputs_head:tt $(,$inputs_tail:tt)*], constants = [$($constants:tt),*], variables = [] $(,)?) => {
//         match &$inputs_head {
//             Tracer::Constant(constant) => {
//                 crate::add_expression_to_builder_v2!(
//                     $op,
//                     inputs = [$($inputs_tail),*],
//                     constants = [$($constants,)* constant],
//                     variables = [],
//                 )
//             },
//             Tracer::Variable(crate::tracing::VariableTracer { id, builder }) => {
//                 // let mut mut_builder = builder.try_borrow_mut().unwrap();
//                 let variable = vec![*id];
//                 crate::add_expression_to_builder_v2!(
//                     $op,
//                     inputs = [$($inputs_tail),*],
//                     constants = [],
//                     variables = [$(mut_builder.add_constant_expression($constants)?,)* variable],
//                     program_builder = builder,
//                 )
//             },
//         }
//     };
//     ($op:expr, inputs = [$inputs_head:tt $(,$inputs_tail:tt)*], constants = [], variables = [$($variables:tt),*], program_builder = $program_builder:tt $(,)?) => {
//         match &$inputs_head {
//             Tracer::Constant(constant) => {
//                 // let mut builder = $program_builder.try_borrow_mut().unwrap();
//                 let constant = mut_builder.add_constant_expression(constant)?;
//                 crate::add_expression_to_builder_v2!(
//                     $op,
//                     inputs = [$($inputs_tail),*],
//                     constants = [],
//                     variables = [$($variables),* constant],
//                     program_builder = $program_builder,
//                 )
//             },
//             Tracer::Variable(crate::tracing::VariableTracer { id, builder }) => {
//                 // We are using a debug assertion here because performance is important and users should generally
//                 // make sure that they are using the built-in ryft tracing functionality instead of trying to build
//                 // out their own. They can still do the latter if they want, but then they would need to worry
//                 // about error handling.
//                 debug_assert!(Rc::ptr_eq(&$program_builder, builder));
//                 crate::add_expression_to_builder_v2!(
//                     $op,
//                     inputs = [$($inputs_tail,)*],
//                     constants = [],
//                     variables = [$($variables,)* vec![id]],
//                     program_builder = $program_builder,
//                 )
//             },
//         }
//     };
//     ($op:expr, inputs = [], constants = [$($constants:tt),*], variables = [] $(,)?) => {
//         {
//             todo!()
//         }
//     };
//     ($op:expr, inputs = [], constants = [], variables = [$($variables:tt),*], program_builder = $program_builder:tt $(,)?) => {
//         {
//             let op = $op;
//             let builder = $program_builder;
//             // TODO(eaplatanios): Avoid unwrap and return an error instead.
//             let mut mut_builder = builder.try_borrow_mut().unwrap();
//             let output_ids = mut_builder.add_expression(op, vec![$($variables),*].into_iter().flatten().collect()).unwrap();
//             output_ids.into_iter().map(|output_id| crate::tracing::Tracer::Variable(crate::tracing::VariableTracer { id: output_id, builder: builder.clone() })).collect::<Vec<_>>()
//         }
//     };
// }

// #[macro_export]
// macro_rules! add_single_output_expression_to_builder {
//     ($op:expr, inputs = [$($inputs:tt),*] $(,)?) => { add_single_output_expression_to_builder!($op, inputs = [$($inputs),*], constants = [], variables = []) };
//     ($op:expr, inputs = [$inputs_head:tt $(,$inputs_tail:tt)*], constants = [$($constants:expr),*], variables = [] $(,)?) => {
//         match &$inputs_head {
//             Tracer::Constant(constant) => {
//                 crate::add_single_output_expression_to_builder!(
//                     $op,
//                     inputs = [$($inputs_tail),*],
//                     constants = [$($constants,)* constant],
//                     variables = [],
//                 )
//             },
//             Tracer::Variable(crate::tracing::VariableTracer { id, builder }) => {
//                 crate::add_single_output_expression_to_builder!(
//                     $op,
//                     inputs = [$($inputs_tail),*],
//                     constants = [],
//                     variables = [$(builder.try_borrow_mut().unwrap().add_constant_expression($constants.clone()).unwrap(),)* vec![*id]],
//                     program_builder = builder,
//                 )
//             },
//         }
//     };
//     ($op:expr, inputs = [$inputs_head:tt $(,$inputs_tail:tt)*], constants = [], variables = [$($variables:expr),*], program_builder = $program_builder:tt $(,)?) => {
//         match &$inputs_head {
//             Tracer::Constant(constant) => {
//                 crate::add_single_output_expression_to_builder!(
//                     $op,
//                     inputs = [$($inputs_tail),*],
//                     constants = [],
//                     variables = [$($variables,)* $program_builder.try_borrow_mut().unwrap().add_constant_expression(constant.clone()).unwrap()],
//                     program_builder = $program_builder,
//                 )
//             },
//             Tracer::Variable(crate::tracing::VariableTracer { id, builder }) => {
//                 // We are using a debug assertion here because performance is important and users should generally
//                 // make sure that they are using the built-in ryft tracing functionality instead of trying to build
//                 // out their own. They can still do the latter if they want, but then they would need to worry
//                 // about error handling.
//                 debug_assert!(Rc::ptr_eq(&$program_builder, builder));
//                 crate::add_single_output_expression_to_builder!(
//                     $op,
//                     inputs = [$($inputs_tail,)*],
//                     constants = [],
//                     variables = [$($variables,)* vec![*id]],
//                     program_builder = $program_builder,
//                 )
//             },
//         }
//     };
//     ($op:expr, inputs = [], constants = [$($constants:expr),*], variables = [] $(,)?) => {
//         {
//             todo!()
//         }
//     };
//     ($op:expr, inputs = [], constants = [], variables = [$($variables:expr),*], program_builder = $program_builder:tt $(,)?) => {
//         {
//             let op = $op;
//             let builder = $program_builder;
//             // TODO(eaplatanios): Avoid unwrap and return an error instead.
//             let output_ids = builder.try_borrow_mut().unwrap().add_expression(op, vec![$($variables),*].into_iter().flatten().collect()).unwrap();
//             debug_assert_eq!(output_ids.len(), 1);
//             crate::tracing::Tracer::Variable(crate::tracing::VariableTracer { id: output_ids[0], builder: builder.clone() })
//         }
//     };
// }

// mod test {
//     use std::{cell::RefCell, rc::Rc};

//     use crate::{
//         programs::{Op, ProgramBuilder},
//         types::{Type, Typed},
//     };

//     use super::Tracer;

//     fn test() {
//         let tracer_0: Tracer<(), usize, Box<dyn Op<()>>> =
//             Tracer { id: 0, builder: Rc::new(RefCell::new(ProgramBuilder::default())) };
//         let tracer_1: Tracer<(), usize, Box<dyn Op<()>>> =
//             Tracer { id: 0, builder: Rc::new(RefCell::new(ProgramBuilder::default())) };
//         let tracer_2 = tracer_0 + tracer_1;
//     }
// }

// TODO(eaplatanios): For the right hand-side of operators, we can do something like `Add<Rhs = Into<Self>>` to support
//  adding constants to graphs that are being traced, etc. I wonder what happens if we try to do the same for the
//  receiver (i.e., the left-hand side of the operator).
//
// TODO(eaplatanios): For constants, we can:
//  1. Add a `ConstantOp` that has `0` inputs.
//  2. Add a `ConstantExpression` that is only used in tracers.
//  3. Have `ConstantTracer` hold a `ConstantExpression`.
//  4. Add a `ConstantExpression` that just holds a single constant `Value` and no op.

// TODO(eaplatanios): Move the op implementations somewhere else.
// TODO(eaplatanios): This design results in so much repetition. Figure out whether we can do something better here.
//  Macros are certainly an option though they may be a little bit of a nightmare to implement.
