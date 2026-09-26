//! Shared machinery of the single-operand linear collectives: [`ParallelPermuteOperation`],
//! [`AllGatherOperation`], [`ParallelSumScatterOperation`], and [`AllToAllOperation`]. Each of them carries the
//! referenced axis name together with the participant count resolved from the active [`NamedAxes`] environment when it
//! is staged, consumes exactly one statically shaped operand, has only degenerate (single-participant) per-item
//! semantics outside a binder, and is linear, so its tangent rides the same collective and its transpose is another
//! collective over the same axis. The [`linear_collective!`] macro generates that shared structure and the functions
//! in this module implement the pieces that the generated code and the hand-written rules have in common. The
//! collectives that also resize an array axis share additional machinery in the sibling `shape_changing` module.
//!
//! [`ParallelPermuteOperation`]: super::ParallelPermuteOperation
//! [`AllGatherOperation`]: super::AllGatherOperation
//! [`ParallelSumScatterOperation`]: super::ParallelSumScatterOperation
//! [`AllToAllOperation`]: super::AllToAllOperation
//! [`NamedAxes`]: crate::axes::NamedAxes

// TODO(eaplatanios): Review this module.

use crate::arrays::{ArrayType, Dimension, Shape};
use crate::contexts::Context;
use crate::differentiation::{DifferentiableType, DifferentiationError};
use crate::macros::check_count;
use crate::partial::PartialValue;
use crate::programs::{MaybeZero, Operation, ProgramError, TypeError, Typed, Value};
use crate::tracing::{Tracer, TracingContext};

/// Validates the shared operand contract of the linear collectives (exactly one statically shaped operand, which may
/// carry unreduced axes only when the collective accepts them) and returns the operand's static dimensions.
///
/// # Parameters
///
///   - `operation_name`: Name of the collective, used in diagnostics.
///   - `accepts_unreduced`: Whether the collective accepts operands with unreduced axes (e.g., a sum-scatter, which
///     completes the pending reduction as part of its exchange).
///   - `input_types`: Input types of the collective.
pub(super) fn linear_collective_dimensions(
    operation_name: &str,
    accepts_unreduced: bool,
    input_types: &[ArrayType],
) -> Result<Vec<usize>, TypeError> {
    check_count!("input", input_types, 1, TypeError);
    if !accepts_unreduced && !input_types[0].unreduced_axes().is_empty() {
        return Err(TypeError::invalid(format!("`{operation_name}` does not support unreduced operands")));
    }
    let Some(shape) = input_types[0].static_shape() else {
        return Err(TypeError::invalid(format!("`{operation_name}` does not support dynamically shaped operands")));
    };
    Ok(shape.dimensions().to_vec())
}

/// Builds a linear collective's output type from its operand and (possibly resized) dimensions, carrying the operand
/// sharding through with the same per-dimension placement (the dimension count never changes).
pub(super) fn linear_collective_output_type(
    operation_name: &'static str,
    input_type: &ArrayType,
    output_dimensions: Vec<usize>,
) -> Result<ArrayType, TypeError> {
    let output_sizes = output_dimensions.into_iter().map(Dimension::Static).collect::<Vec<_>>();
    let sharding = input_type.resized_sharding(output_sizes.as_slice(), operation_name)?;
    let mut output_type =
        ArrayType::new(input_type.data_type(), Shape::new(output_sizes)).with_memory(input_type.memory());
    output_type.sharding = sharding;
    Ok(output_type)
}

/// Interprets a linear collective outside any binder: only the degenerate single-participant axis
/// (`axis_size == 1`) has defined per-item semantics (the identity), and any larger axis reports an error because
/// the other participants do not exist per item.
pub(super) fn interpret_degenerate_collective<V: Clone>(
    operation_name: &str,
    axis_name: &str,
    axis_size: usize,
    inputs: &[V],
) -> Result<Vec<V>, ProgramError> {
    check_count!("input", inputs, 1, ProgramError);
    if axis_size != 1 {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "cannot interpret `{operation_name}` over axis `{axis_name}` of size {axis_size} without an \
                 enclosing binder",
            ),
        });
    }
    Ok(vec![inputs[0].clone()])
}

/// Implements the shared structure of the single-operand linear collectives: the operation constant and struct with
/// its accessors, the `Display`/`Operation` implementations (with payload-dependent output-shape inference provided as
/// a closure over the operand dimensions), degenerate interpretation, default partial evaluation, and the linear
/// forward-mode rule (the tangent rides the same collective). The batching and transposition rules and the
/// value-level staging capabilities are hand-written next to each macro invocation because each collective
/// materializes the mapped batch axis, and exposes its named axis to users, differently.
macro_rules! linear_collective {
    // Public form: generates the operation constant and struct with its accessors, the `Display`/`Operation`
    // implementations, degenerate interpretation, and default partial evaluation. `accepts_unreduced` states whether
    // type inference accepts operands with unreduced axes.
    (
        $(#[$operation_documentation:meta])*
        operation = $operation:ident,
        name = $operation_name:ident = $name_literal:literal,
        accepts_unreduced = $accepts_unreduced:literal,
        fields = { $($(#[$field_documentation:meta])* $field:ident: $field_type:ty),* $(,)? },
        infer = |$infer_self:ident, $input_type:ident, $dimensions:ident| $infer:block $(,)?
    ) => {
        /// Canonical operation name for the operation.
        pub const $operation_name: &str = $name_literal;

        $(#[$operation_documentation])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $operation {
            /// Axis name referenced by this collective.
            axis_name: String,

            /// Number of participants along the named axis, resolved from the active [`NamedAxes`] environment when
            /// the operation is staged.
            axis_size: usize,

            $($(#[$field_documentation])* $field: $field_type,)*
        }

        impl $operation {
            /// Creates a new operation over the named axis with the provided resolved axis size.
            #[inline]
            pub fn new(axis_name: String, axis_size: usize, $($field: $field_type),*) -> Self {
                Self { axis_name, axis_size, $($field),* }
            }

            /// Returns the axis name referenced by this collective.
            #[inline]
            pub fn axis_name(&self) -> &str {
                &self.axis_name
            }

            /// Returns the number of participants along the named axis.
            #[inline]
            pub fn axis_size(&self) -> usize {
                self.axis_size
            }
        }

        impl Display for $operation {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.render(formatter, 0)
            }
        }

        impl Operation for $operation {
            type Type = ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                $operation_name
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                check_count!("region", region_interfaces, 0, TypeError);
                validate_collective_axis_size($name_literal, self.axis_size)?;
                let $dimensions = linear_collective_dimensions($name_literal, $accepts_unreduced, input_types)?;
                let $infer_self = self;
                let $input_type = &input_types[0];
                Ok(vec![$infer?])
            }

            fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
                OperationFormatter::new(formatter, indentation, $operation_name)?.bracketed(|operation| {
                    operation.field("axis_name", format_args!("{:?}", self.axis_name))?;
                    operation.field("axis_size", &self.axis_size)?;
                    $(operation.field(stringify!($field), format_args!("{:?}", &self.$field))?;)*
                    Ok(())
                })
            }
        }

        impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for $operation {
            fn interpret<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, ProgramError> {
                interpret_degenerate_collective($name_literal, &self.axis_name, self.axis_size, inputs)
            }
        }

        // Partial evaluation defers to the default fold-or-residualize behavior of
        // `Program::partially_evaluate`.
        impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for $operation where
            C::Operation: From<$operation>
        {
        }
    };

    // Generates the linear forward-mode rule after the operation's batching implementation.
    (@differentiation $operation:ident) => {
        // Forward-mode rule: the collective is linear, so the tangent rides the same collective. Structural-zero
        // tangents stay symbolic, retyped to the output tangent type (the collective changes shapes).
        impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for $operation
        where
            C::Operation: From<$operation>,
        {
            fn jvp<D: DifferentiationDriver<C>, P: $crate::DifferentiationPolicy<C>>(
                &self,
                context: &$crate::DifferentiationContext<C, P>,
                _driver: &D,
                inputs: &[DifferentiationDual<C::Value>],
            ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
                check_count!("input", inputs, 1, ProgramError);
                let mut primal_outputs =
                    context.primal().bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].primal()))?;
                check_count!("output", primal_outputs, 1, ProgramError);
                let primal = primal_outputs.remove(0);
                let tangent = match inputs[0].tangent() {
                    MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                    MaybeZero::Value(tangent) => {
                        let mut tangent_outputs =
                            context.tangent().bind(self.clone(), Vec::new(), std::slice::from_ref(tangent))?;
                        check_count!("output", tangent_outputs, 1, ProgramError);
                        MaybeZero::Value(tangent_outputs.remove(0))
                    }
                };
                Ok(vec![DifferentiationDual::new(primal, tangent)?])
            }
        }
    };
}

pub(super) use linear_collective;

/// Stages the adjoint collective of a linear collective on the output cotangent: a known operand
/// receives a structural zero, a zero output cotangent stays symbolic, and a live cotangent rides the provided
/// adjoint operation.
pub(super) fn transpose_linear_collective<V, O, A>(
    context: &mut TracingContext<V, O>,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    adjoint: A,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError>
where
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<A>,
    A: Operation<Type = ArrayType>,
{
    check_count!("input", inputs, 1, ProgramError);
    check_count!("output", outputs, 1, ProgramError);
    if inputs[0].is_known() {
        return Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?)]);
    }
    match &outputs[0] {
        MaybeZero::Value(cotangent) => {
            let mut contributions = context.bind(O::from(adjoint), Vec::new(), std::slice::from_ref(cotangent))?;
            check_count!("output", contributions, 1, ProgramError);
            Ok(vec![MaybeZero::Value(contributions.remove(0))])
        }
        MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent()?)]),
    }
}
