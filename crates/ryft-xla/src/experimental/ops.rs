use std::fmt::Display;

use ryft_core::differentiation::{Cotangent, LinearOperation};
use ryft_core::macros::check_count;
use ryft_core::operations::{InterpretableOperation, Operation};
use ryft_core::tracing::domains::{Tracer, TracingContext, TracingDomain};
use ryft_core::tracing::{ProgramTracingContext, Traceable, TracingError};
use ryft_core::tracing_v2::{
    ArrayOperation, Differentiable, DifferentiableOperation, JvpContext, JvpTracer, LinearArrayOperation,
    LinearOperationExtensionFamily, TracerReplayValue,
};
use ryft_core::types::{ArrayType, TypeError};

use crate::experimental::domains::{LinearXlaDomain, XlaDomain};
use crate::experimental::operations::{LinearShardMapOperation, ShardMapOperation, WithShardingConstraintOperation};
use crate::experimental::shard_map::{ShardMapTensor, XlaTracer};

/// Backend-owned ordinary operations that extend the reusable core array operation set.
#[derive(Clone, Debug)]
pub enum XlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
{
    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<V>>),

    /// XLA-specific `linear_shard_map` staged in ordinary traced programs.
    LinearShardMap(Box<LinearShardMapOperation<V>>),

    /// XLA-specific sharding constraint.
    WithShardingConstraint(WithShardingConstraintOperation),
}

/// Ordinary staged-op universe owned by the XLA backend.
pub type XlaOperation = ArrayOperation<ShardMapTensor, ArrayType, XlaOperationExtension<ShardMapTensor>>;

/// Backend-owned linear operations that extend the reusable core linear array operation set.
#[derive(Clone, Debug)]
pub enum LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
{
    /// XLA-specific linear `shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<V>>),

    /// XLA-specific sharding constraint in tangent/cotangent programs.
    WithShardingConstraint(WithShardingConstraintOperation),
}

/// Linear staged-op universe owned by the XLA backend.
pub type LinearXlaOperation<V> = LinearArrayOperation<V, ArrayType, LinearXlaOperationExtension<V>>;

fn missing_traced_input() -> TracingError {
    TracingError::InvalidInputCount { expected: 1, got: 0 }
}

macro_rules! delegate_extension {
    ($self:expr, [$($variant:ident),+ $(,)?], |$operation:ident| $body:expr) => {
        match $self {
            $(Self::$variant($operation) => $body,)+
        }
    };
}

impl TracerReplayValue<ArrayType> for ShardMapTensor {}

impl Display for XlaOperationExtension<ShardMapTensor> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation<ArrayType> for XlaOperationExtension<ShardMapTensor> {
    #[inline]
    fn name(&self) -> &'static str {
        delegate_extension!(self, [ShardMap, LinearShardMap, WithShardingConstraint], |op| op.name())
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        delegate_extension!(self, [ShardMap, LinearShardMap, WithShardingConstraint], |op| {
            op.infer_output_types(input_types)
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        delegate_extension!(self, [ShardMap, LinearShardMap, WithShardingConstraint], |op| {
            op.render(formatter, indentation)
        })
    }
}

impl InterpretableOperation<ArrayType, ShardMapTensor> for XlaOperationExtension<ShardMapTensor> {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        delegate_extension!(self, [ShardMap, LinearShardMap, WithShardingConstraint], |op| { op.interpret(inputs) })
    }
}

impl InterpretableOperation<ArrayType, XlaTracer<'static, 'static>> for XlaOperationExtension<ShardMapTensor> {
    fn interpret(
        &self,
        inputs: &[XlaTracer<'static, 'static>],
    ) -> Result<Vec<XlaTracer<'static, 'static>>, TracingError> {
        match self {
            Self::ShardMap(op) => {
                let exemplar = inputs.first().ok_or_else(missing_traced_input)?;
                op.interpret_traced_with_context(exemplar.builder().clone(), inputs)
            }
            Self::LinearShardMap(op) => {
                let exemplar = inputs.first().ok_or_else(missing_traced_input)?;
                op.interpret_traced_with_context(exemplar.builder().clone(), inputs)
            }
            Self::WithShardingConstraint(op) => op.interpret(inputs),
        }
    }
}

impl<'c> DifferentiableOperation<XlaDomain<'c>> for XlaOperationExtension<ShardMapTensor> {
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, XlaDomain<'c>>,
        inputs: &[JvpTracer<ShardMapTensor, ArrayType, Tracer<'jvp, LinearXlaDomain>>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, ArrayType, Tracer<'jvp, LinearXlaDomain>>>, TracingError>
    where
        XlaDomain<'c>: 'jvp,
    {
        delegate_extension!(self, [ShardMap, LinearShardMap, WithShardingConstraint], |op| { op.jvp(context, inputs) })
    }
}

impl<'domain, 'context> DifferentiableOperation<TracingContext<'domain, XlaDomain<'context>>>
    for XlaOperationExtension<ShardMapTensor>
where
    XlaDomain<'context>: 'domain,
    'context: 'domain,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'domain, XlaDomain<'context>>>,
        inputs: &[JvpTracer<
            XlaTracer<'domain, 'context>,
            ArrayType,
            Tracer<'jvp, TracingContext<'domain, XlaDomain<'context>>>,
        >],
    ) -> Result<
        Vec<
            JvpTracer<
                XlaTracer<'domain, 'context>,
                ArrayType,
                Tracer<'jvp, TracingContext<'domain, XlaDomain<'context>>>,
            >,
        >,
        TracingError,
    >
    where
        TracingContext<'domain, XlaDomain<'context>>: 'jvp,
    {
        match self {
            Self::ShardMap(op) => {
                let primal_context = context.domain.clone();
                let traced_op = ShardMapOperation::<XlaTracer<'domain, 'context>>::new(op.body.clone());
                traced_op.jvp_with_context(&primal_context, context, inputs)
            }
            Self::LinearShardMap(op) => {
                let primal_context = context.domain.clone();
                op.jvp_traced_with_context(&primal_context, context, inputs)
            }
            Self::WithShardingConstraint(op) => {
                check_count!("input", inputs, 1, TracingError);
                let input = &inputs[0];
                let primal_outputs = input.primal.context.trace(
                    XlaOperation::Extension(XlaOperationExtension::WithShardingConstraint(op.clone())),
                    &[&input.primal],
                )?;
                check_count!("output", primal_outputs, 1, TracingError);
                let tangent_input = match input.tangent.clone() {
                    ryft_core::differentiation::Tangent::Zero(_) => context.add_constant(input.primal.zero_tangent()?),
                    ryft_core::differentiation::Tangent::Value(tracer) => tracer,
                };
                let mut tangent_outputs = context.stage(
                    LinearXlaOperation::Extension(LinearXlaOperationExtension::WithShardingConstraint(op.clone())),
                    &[tangent_input],
                )?;
                check_count!("output", tangent_outputs, 1, TracingError);
                Ok(vec![JvpTracer {
                    primal: primal_outputs[0].clone(),
                    tangent: ryft_core::differentiation::Tangent::Value(tangent_outputs.remove(0)),
                }])
            }
        }
    }
}

impl<V> Display for LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
    Self: Operation<ArrayType>,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V> Operation<ArrayType> for LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
{
    #[inline]
    fn name(&self) -> &'static str {
        delegate_extension!(self, [LinearShardMap, WithShardingConstraint], |op| op.name())
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        delegate_extension!(self, [LinearShardMap, WithShardingConstraint], |op| { op.infer_output_types(input_types) })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        delegate_extension!(self, [LinearShardMap, WithShardingConstraint], |op| { op.render(formatter, indentation) })
    }
}

impl<V> InterpretableOperation<ArrayType, V> for LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
    LinearShardMapOperation<V>: InterpretableOperation<ArrayType, V>,
    WithShardingConstraintOperation: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        delegate_extension!(self, [LinearShardMap, WithShardingConstraint], |op| op.interpret(inputs))
    }
}

impl<V> LinearOperation<ArrayType, V, LinearXlaOperation<V>> for LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
    LinearShardMapOperation<V>: LinearOperation<ArrayType, V, LinearXlaOperation<V>>,
    WithShardingConstraintOperation: LinearOperation<ArrayType, V, LinearXlaOperation<V>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V>>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>>, TracingError> {
        delegate_extension!(self, [LinearShardMap, WithShardingConstraint], |op| {
            op.transpose(context, output_cotangents)
        })
    }
}

impl<D, V> LinearOperationExtensionFamily<D, V> for LinearXlaOperationExtension<V>
where
    D: TracingDomain<Type = ArrayType, Value = ShardMapTensor, OperationCarrier = XlaOperation>,
    V: Traceable<ArrayType>,
{
    type ForTracer<'domain>
        = LinearXlaOperationExtension<Tracer<'domain, D>>
    where
        D: 'domain;
}
