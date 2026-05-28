use std::cell::RefCell;
use std::fmt::Display;
use std::rc::Rc;

use ryft_core::differentiation::{Cotangent, LinearOperation};
use ryft_core::macros::check_count;
use ryft_core::operations::constants::SupportsZero;
use ryft_core::operations::{InterpretableOperation, Operation, OperationFormatter};
use ryft_core::parameters::Placeholder;
use ryft_core::tracing::contexts::{Context, TracingContext};
use ryft_core::tracing::domains::{DomainTracer, Tracer, TracingDomain};
use ryft_core::tracing::{Program, ProgramBuilder, ProgramTracingContext, Traceable, TracingError};
use ryft_core::tracing_v2::batching::{
    ArrayBatch, BatchableOperation, BatchingContext, BatchingError, batch_input_metadata,
};
use ryft_core::tracing_v2::operations::{flat_program_input_types, flat_program_output_types};
use ryft_core::tracing_v2::{
    ArrayOperation, Differentiable, DifferentiableContext, DifferentiableOperation, JvpContext, JvpTracer,
    LinearArrayOperation, LinearOperationExtensionFamily,
};
use ryft_core::types::{ArrayType, Size, TypeError, Typed};

use crate::experimental::domains::XlaDomain;
use crate::experimental::operations::{LinearShardMapOperation, ShardMapOperation, WithShardingConstraintOperation};
use crate::experimental::shard_map::XlaTracer;

/// Backend-owned ordinary operations that extend the reusable core array operation set.
#[derive(Clone, Debug)]
pub enum XlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
{
    /// Call to a jitted XLA sub-program.
    JitCall(Box<JitCallOperation>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<V>>),

    /// XLA-specific `linear_shard_map` staged in ordinary traced programs.
    LinearShardMap(Box<LinearShardMapOperation<V>>),

    /// XLA-specific sharding constraint.
    WithShardingConstraint(WithShardingConstraintOperation),
}

/// Ordinary staged-op universe owned by the XLA backend.
pub type XlaOperation = ArrayOperation<ArrayType, ArrayType, XlaOperationExtension<ArrayType>>;

/// Flat XLA program payload used by staged call operations.
pub type FlatXlaProgram = Program<ArrayType, ArrayType, XlaOperation, Vec<ArrayType>, Vec<ArrayType>>;

/// Backend-owned linear operations that extend the reusable core linear array operation set.
#[derive(Clone, Debug)]
pub enum LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
{
    /// Linearized call to a jitted XLA sub-program.
    LinearJitCall(Box<LinearJitCallOperation<V>>),

    /// XLA-specific linear `shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<V>>),

    /// XLA-specific sharding constraint in tangent/cotangent programs.
    WithShardingConstraint(WithShardingConstraintOperation),
}

/// Linear staged-op universe owned by the XLA backend.
pub type LinearXlaOperation<V> = LinearArrayOperation<V, ArrayType, LinearXlaOperationExtension<V>>;

/// Staged call to a flat jitted XLA program.
#[derive(Clone, Debug)]
pub struct JitCallOperation {
    /// Flat callee program called by this operation.
    program: FlatXlaProgram,
}

impl JitCallOperation {
    /// Creates a staged jitted-call operation for `program`.
    #[inline]
    pub(crate) fn new(program: FlatXlaProgram) -> Self {
        Self { program }
    }

    /// Returns the flat callee program.
    #[inline]
    pub(crate) fn program(&self) -> &FlatXlaProgram {
        &self.program
    }
}

/// Linearized jitted call used inside tangent and cotangent programs.
#[derive(Clone, Debug)]
pub struct LinearJitCallOperation<V>
where
    V: Traceable<ArrayType>,
{
    /// Program applied by this linear call. Its inputs are `captured_inputs` followed by the operation inputs.
    program: FlatXlaProgram,

    /// Program for the transposed linear call with the same captured prefix inputs.
    transpose_program: FlatXlaProgram,

    /// Captured primal prefix inputs supplied to `program` before the linear operation inputs.
    captured_inputs: Vec<V>,

    /// Flat linear input types expected by this operation.
    input_types: Vec<ArrayType>,

    /// Flat output types produced by this operation.
    output_types: Vec<ArrayType>,
}

impl<V> LinearJitCallOperation<V>
where
    V: Traceable<ArrayType>,
{
    /// Creates a linear jitted-call operation.
    fn new(
        program: FlatXlaProgram,
        transpose_program: FlatXlaProgram,
        captured_inputs: Vec<V>,
        input_types: Vec<ArrayType>,
        output_types: Vec<ArrayType>,
    ) -> Self {
        Self { program, transpose_program, captured_inputs, input_types, output_types }
    }

    /// Returns the flat transformed callee program.
    #[inline]
    pub(crate) fn program(&self) -> &FlatXlaProgram {
        &self.program
    }

    /// Returns captured prefix inputs supplied before the operation's explicit inputs.
    #[inline]
    pub(crate) fn captured_inputs(&self) -> &[V] {
        self.captured_inputs.as_slice()
    }
}

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

fn ensure_call_input_types(
    operation_name: &'static str,
    expected_types: &[ArrayType],
    input_types: &[ArrayType],
) -> Result<(), TypeError> {
    if expected_types.len() != input_types.len() {
        return Err(TypeError {
            message: format!(
                "{operation_name} expected {} input(s) but got {}",
                expected_types.len(),
                input_types.len(),
            ),
        });
    }
    for (index, (expected, actual)) in expected_types.iter().zip(input_types).enumerate() {
        if expected != actual {
            return Err(TypeError {
                message: format!("{operation_name} input #{index} expected {expected} but got {actual}"),
            });
        }
    }
    Ok(())
}

fn build_jvp_call_program(program: &FlatXlaProgram) -> Result<FlatXlaProgram, TracingError> {
    let input_types = flat_program_input_types(program);
    let signature = input_types.iter().cloned().chain(input_types.iter().cloned()).collect::<Vec<_>>();
    let token = XlaDomain::token();
    let (_, traced): (Vec<ArrayType>, FlatXlaProgram) = token.trace(
        |inputs: Vec<DomainTracer<'static, XlaDomain<'static>>>| -> Result<
            Vec<DomainTracer<'static, XlaDomain<'static>>>,
            TracingError,
        > {
            let input_count = inputs.len() / 2;
            let primals = inputs[..input_count].to_vec();
            let tangents = inputs[input_count..].to_vec();
            let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
            let (_, pushforward) = context.linearize(
                |linearized_inputs| {
                    let linearization_context = linearized_inputs
                        .first()
                        .ok_or_else(missing_traced_input)?
                        .context()
                        .clone();
                    linearization_context.stage_program(program, linearized_inputs)
                },
                primals,
            )?;
            pushforward.interpret(tangents)
        },
        signature,
    )?;
    traced.into_simplified()
}

fn build_pullback_call_program(program: &FlatXlaProgram) -> Result<FlatXlaProgram, TracingError> {
    let input_types = flat_program_input_types(program);
    let output_types = flat_program_output_types(program);
    let signature = input_types.iter().cloned().chain(output_types.iter().cloned()).collect::<Vec<_>>();
    let token = XlaDomain::token();
    let (_, traced): (Vec<ArrayType>, FlatXlaProgram) = token.trace(
        |inputs: Vec<DomainTracer<'static, XlaDomain<'static>>>| -> Result<
            Vec<DomainTracer<'static, XlaDomain<'static>>>,
            TracingError,
        > {
            let input_count = input_types.len();
            let primals = inputs[..input_count].to_vec();
            let cotangents = inputs[input_count..].to_vec();
            let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
            let (_, pullback) = context.vjp(
                |linearized_inputs| {
                    let linearization_context = linearized_inputs
                        .first()
                        .ok_or_else(missing_traced_input)?
                        .context()
                        .clone();
                    linearization_context.stage_program(program, linearized_inputs)
                },
                primals,
            )?;
            pullback.interpret(cotangents)
        },
        signature,
    )?;
    traced.into_simplified()
}

fn build_batched_call_program(
    program: &FlatXlaProgram,
    input_axes: &[Option<usize>],
    axis_size: usize,
) -> Result<(FlatXlaProgram, Vec<Option<usize>>), TracingError> {
    let logical_input_types = flat_program_input_types(program);
    check_count!("input", input_axes, logical_input_types.len(), TracingError);
    let physical_input_types = logical_input_types
        .iter()
        .zip(input_axes)
        .map(|(logical_type, axis)| match axis {
            Some(axis) => logical_type.with_inserted_dimension(*axis, Size::Static(axis_size)),
            None => Ok(logical_type.clone()),
        })
        .collect::<Result<Vec<_>, _>>()?;

    let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
    let parent_context = TracingContext::new(XlaDomain::token(), builder.clone());
    let batching_context = BatchingContext::new(parent_context, axis_size);
    let mut input_tracers = Vec::with_capacity(physical_input_types.len());
    for ((physical_type, logical_type), axis) in physical_input_types.iter().zip(&logical_input_types).zip(input_axes) {
        let atom = builder.borrow_mut().add_input(physical_type.clone());
        batching_context.register_axis(atom, *axis);
        input_tracers.push(batching_context.tracer(atom, Some(logical_type.clone())));
    }
    let output_tracers = batching_context.stage_program(program, input_tracers)?;
    let output_atom_ids = output_tracers.iter().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
    let output_axes = output_atom_ids.iter().map(|atom| batching_context.axis_for(*atom)).collect::<Vec<_>>();
    drop(output_tracers);
    drop(batching_context);

    let builder = Rc::try_unwrap(builder).map_err(|_| TracingError::EscapedProgramBuilder)?.into_inner();
    let batched_program = builder
        .build::<Vec<ArrayType>, Vec<ArrayType>>(
            output_atom_ids,
            vec![Placeholder; physical_input_types.len()],
            vec![Placeholder; output_axes.len()],
        )?
        .into_simplified()?;
    Ok((batched_program, output_axes))
}

impl Operation<ArrayType> for JitCallOperation {
    #[inline]
    fn name(&self) -> &'static str {
        "jit_call"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ensure_call_input_types(self.name(), flat_program_input_types(&self.program).as_slice(), input_types)?;
        Ok(flat_program_output_types(&self.program))
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("inputs", self.program.input_ids().len())?;
            operation.field("outputs", self.program.output_ids().len())
        })
    }
}

impl JitCallOperation {
    /// Stages this call operation into the active XLA tracing context carried by `inputs`.
    fn interpret_traced_with_context(
        &self,
        inputs: &[XlaTracer<'static, 'static>],
    ) -> Result<Vec<XlaTracer<'static, 'static>>, TracingError> {
        let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
        context.stage_operation(XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(self.clone()))), inputs)
    }

    /// Creates the linear call operation corresponding to this ordinary call at `primals`.
    fn linear_call_operation<V>(&self, primals: Vec<V>) -> Result<LinearJitCallOperation<V>, TracingError>
    where
        V: Traceable<ArrayType>,
    {
        Ok(LinearJitCallOperation::new(
            build_jvp_call_program(&self.program)?,
            build_pullback_call_program(&self.program)?,
            primals,
            flat_program_input_types(&self.program),
            flat_program_output_types(&self.program),
        ))
    }

    /// Returns the call operation and output-axis metadata for batching this call.
    fn batched_call_operation<V>(&self, inputs: &[ArrayBatch<V>]) -> Result<(Self, Vec<Option<usize>>), TracingError> {
        let (_, input_axes, axis_size) = batch_input_metadata(inputs)?;
        let axis_size = input_axes.iter().any(Option::is_some).then_some(axis_size);
        match axis_size {
            Some(axis_size) => {
                let (batched_program, output_axes) =
                    build_batched_call_program(&self.program, input_axes.as_slice(), axis_size)?;
                Ok((JitCallOperation::new(batched_program), output_axes))
            }
            None => Ok((self.clone(), vec![None; flat_program_output_types(&self.program).len()])),
        }
    }

    /// Completes the JVP rule after the caller has produced primal outputs in its host representation.
    fn jvp_from_primal_outputs<'jvp, E, V>(
        &self,
        context: &mut JvpContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
        primals: Vec<V>,
        primal_outputs: Vec<V>,
    ) -> Result<Vec<JvpTracer<'jvp, E>>, TracingError>
    where
        E: Differentiable<Type = ArrayType, Value = V, Tangent = V, LinearOperationCarrier = LinearXlaOperation<V>>
            + 'jvp,
        V: Traceable<ArrayType>,
    {
        let tangent_inputs = inputs
            .iter()
            .map(|input| context.materialize_tangent(input.tangent().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_operation = self.linear_call_operation(primals)?;
        let tangent_outputs = context.stage_operation(
            LinearXlaOperation::Extension(LinearXlaOperationExtension::LinearJitCall(Box::new(linear_operation))),
            tangent_inputs.as_slice(),
        )?;
        check_count!("output", tangent_outputs, primal_outputs.len(), TracingError);
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer::new(primal, ryft_core::differentiation::Tangent::Value(tangent)))
            .collect())
    }
}

impl<'o, RuleContext> BatchableOperation<ArrayType, RuleContext> for JitCallOperation {
    fn batch(
        &self,
        _context: &RuleContext,
        inputs: &[ArrayBatch<ArrayType>],
    ) -> Result<Vec<ArrayBatch<ArrayType>>, TracingError> {
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let (operation, output_axes) = self.batched_call_operation(inputs)?;
        let outputs = operation.infer_output_types(physical_inputs.as_slice())?;
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output.r#type().into_owned(), output, axis))
            .collect()
    }
}

impl<RuleContext> BatchableOperation<DomainTracer<'static, XlaDomain<'static>>, RuleContext> for JitCallOperation {
    fn batch(
        &self,
        _context: &RuleContext,
        inputs: &[ArrayBatch<DomainTracer<'static, XlaDomain<'static>>>],
    ) -> Result<Vec<ArrayBatch<DomainTracer<'static, XlaDomain<'static>>>>, TracingError> {
        let context = inputs.first().ok_or_else(missing_traced_input)?.value().context().clone();
        let physical_inputs = inputs.iter().map(|input| input.value().clone()).collect::<Vec<_>>();
        let (operation, output_axes) = self.batched_call_operation(inputs)?;
        let outputs = context.stage_operation(
            XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(operation))),
            physical_inputs.as_slice(),
        )?;
        outputs
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| ArrayBatch::new(output.r#type().into_owned(), output, axis))
            .collect()
    }
}

impl<'c> DifferentiableOperation<XlaDomain<'c>> for JitCallOperation {
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, XlaDomain<'c>>,
        inputs: &[JvpTracer<'jvp, XlaDomain<'c>>],
    ) -> Result<Vec<JvpTracer<'jvp, XlaDomain<'c>>>, TracingError>
    where
        XlaDomain<'c>: 'jvp,
    {
        check_count!("input", inputs, flat_program_input_types(&self.program).len(), TracingError);
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = self.infer_output_types(primals.as_slice())?;
        self.jvp_from_primal_outputs(context, inputs, primals, primal_outputs)
    }
}

impl<'domain, 'context> DifferentiableOperation<TracingContext<'domain, XlaDomain<'context>>> for JitCallOperation
where
    XlaDomain<'context>: 'domain,
    'context: 'domain,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'domain, XlaDomain<'context>>>,
        inputs: &[JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>>>],
    ) -> Result<Vec<JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>>>>, TracingError>
    where
        TracingContext<'domain, XlaDomain<'context>>: 'jvp,
    {
        check_count!("input", inputs, flat_program_input_types(&self.program).len(), TracingError);
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = context.differentiable().stage_operation(
            XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(self.clone()))),
            primals.as_slice(),
        )?;
        self.jvp_from_primal_outputs(context, inputs, primals, primal_outputs)
    }
}

impl<V> Operation<ArrayType> for LinearJitCallOperation<V>
where
    V: Traceable<ArrayType>,
{
    #[inline]
    fn name(&self) -> &'static str {
        "linear_jit_call"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        ensure_call_input_types(self.name(), self.input_types.as_slice(), input_types)?;
        Ok(self.output_types.clone())
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("captured", self.captured_inputs.len())?;
            operation.field("inputs", self.input_types.len())?;
            operation.field("outputs", self.output_types.len())
        })
    }
}

impl<'o, C> InterpretableOperation<ArrayType, Tracer<C>> for LinearJitCallOperation<Tracer<C>>
where
    C: Context<Type = ArrayType, Value = ArrayType, Operation = XlaOperation>,
{
    fn interpret(&self, inputs: &[Tracer<C>]) -> Result<Vec<Tracer<C>>, TracingError> {
        let context = self
            .captured_inputs
            .first()
            .or_else(|| inputs.first())
            .ok_or_else(missing_traced_input)?
            .context()
            .clone();
        let full_inputs = self.captured_inputs.iter().cloned().chain(inputs.iter().cloned()).collect::<Vec<_>>();
        context.stage_operation(
            XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(JitCallOperation::new(
                self.program.clone(),
            )))),
            full_inputs.as_slice(),
        )
    }
}

impl<V> LinearOperation<ArrayType, V, LinearXlaOperation<V>> for LinearJitCallOperation<V>
where
    V: Traceable<ArrayType>,
    LinearXlaOperation<V>: SupportsZero<ArrayType, V>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V>>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>>, TracingError> {
        check_count!("output", output_cotangents, self.output_types.len(), TracingError);
        let mut cotangent_inputs = Vec::with_capacity(output_cotangents.len());
        for (cotangent, output_type) in output_cotangents.iter().zip(self.output_types.iter()) {
            match cotangent {
                Cotangent::Staged(cotangent) => cotangent_inputs.push(cotangent.clone()),
                Cotangent::Zero => {
                    let zero_outputs = context.stage_operation(
                        LinearXlaOperation::<V>::zero_operation(output_type.clone()),
                        &[] as &[ryft_core::tracing::domains::ProgramTracer<
                            'transpose,
                            ArrayType,
                            V,
                            LinearXlaOperation<V>,
                        >],
                    )?;
                    check_count!("output", zero_outputs, 1, TracingError);
                    cotangent_inputs.push(zero_outputs[0].clone());
                }
            }
        }
        let transposed = LinearJitCallOperation::new(
            self.transpose_program.clone(),
            self.program.clone(),
            self.captured_inputs.clone(),
            self.output_types.clone(),
            self.input_types.clone(),
        );
        let input_cotangents = context.stage_operation(
            LinearXlaOperation::Extension(LinearXlaOperationExtension::LinearJitCall(Box::new(transposed))),
            cotangent_inputs.as_slice(),
        )?;
        Ok(input_cotangents.into_iter().map(Cotangent::Staged).collect())
    }
}

impl Display for XlaOperationExtension<ArrayType> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation<ArrayType> for XlaOperationExtension<ArrayType> {
    #[inline]
    fn name(&self) -> &'static str {
        delegate_extension!(self, [JitCall, ShardMap, LinearShardMap, WithShardingConstraint], |op| op.name())
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        delegate_extension!(self, [JitCall, ShardMap, LinearShardMap, WithShardingConstraint], |op| {
            op.infer_output_types(input_types)
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        delegate_extension!(self, [JitCall, ShardMap, LinearShardMap, WithShardingConstraint], |op| {
            op.render(formatter, indentation)
        })
    }
}

impl InterpretableOperation<ArrayType, XlaTracer<'static, 'static>> for XlaOperationExtension<ArrayType> {
    fn interpret(
        &self,
        inputs: &[XlaTracer<'static, 'static>],
    ) -> Result<Vec<XlaTracer<'static, 'static>>, TracingError> {
        match self {
            Self::JitCall(op) => op.interpret_traced_with_context(inputs),
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

/// Batching rules for the XLA-specific extension variants.
///
/// `ShardMap` and `LinearShardMap` carry inner programs whose proper batching requires lifting the captured body through
/// context-aware batching rules per instruction. These variants return [`BatchingError::MissingBatchingRule`] so that
/// programs which use shard_map can't be silently mis-batched; programs that don't touch these ops batch correctly
/// through this trait impl.
///
/// `WithShardingConstraint` is a unary identity with a sharding annotation. Batching it
/// requires extending the sharding to cover the new lane axis. We don't yet have a generic
/// "insert a replicated mesh dim at position k" helper on [`Sharding`](ryft_core::sharding::Sharding),
/// so for now this variant also returns [`BatchingError::MissingBatchingRule`]. A future
/// follow-up can implement the extension once the sharding helper exists.
impl<VRule, RuleContext> BatchableOperation<VRule, RuleContext> for XlaOperationExtension<ArrayType>
where
    VRule: Traceable<ArrayType>,
    JitCallOperation: BatchableOperation<VRule, RuleContext>,
{
    fn batch(
        &self,
        context: &RuleContext,
        inputs: &[ArrayBatch<VRule>],
    ) -> Result<Vec<ArrayBatch<VRule>>, TracingError> {
        match self {
            Self::JitCall(op) => op.batch(context, inputs),
            _ => Err(BatchingError::MissingBatchingRule { operation: self.name().to_string() }.into()),
        }
    }
}

impl<'c> DifferentiableOperation<XlaDomain<'c>> for XlaOperationExtension<ArrayType> {
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, XlaDomain<'c>>,
        inputs: &[JvpTracer<'jvp, XlaDomain<'c>>],
    ) -> Result<Vec<JvpTracer<'jvp, XlaDomain<'c>>>, TracingError>
    where
        XlaDomain<'c>: 'jvp,
    {
        delegate_extension!(self, [JitCall, ShardMap, LinearShardMap, WithShardingConstraint], |op| {
            op.jvp(context, inputs)
        })
    }
}

impl<'domain, 'context> DifferentiableOperation<TracingContext<'domain, XlaDomain<'context>>>
    for XlaOperationExtension<ArrayType>
where
    XlaDomain<'context>: 'domain,
    'context: 'domain,
{
    fn jvp<'jvp>(
        &self,
        context: &mut JvpContext<'jvp, TracingContext<'domain, XlaDomain<'context>>>,
        inputs: &[JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>>>],
    ) -> Result<Vec<JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>>>>, TracingError>
    where
        TracingContext<'domain, XlaDomain<'context>>: 'jvp,
    {
        match self {
            Self::JitCall(op) => op.jvp(context, inputs),
            Self::ShardMap(op) => {
                let traced_op = ShardMapOperation::<XlaTracer<'domain, 'context>>::new(op.body().clone());
                traced_op.jvp_with_context(context.differentiable(), context, inputs)
            }
            Self::LinearShardMap(op) => op.jvp_traced_with_context(context.differentiable(), context, inputs),
            Self::WithShardingConstraint(op) => {
                check_count!("input", inputs, 1, TracingError);
                let input = &inputs[0];
                let primal_outputs = input.primal().context().stage_operation(
                    XlaOperation::Extension(XlaOperationExtension::WithShardingConstraint(op.clone())),
                    &[input.primal()],
                )?;
                check_count!("output", primal_outputs, 1, TracingError);
                let tangent_input = context.materialize_tangent(input.tangent().clone())?;
                let mut tangent_outputs = context.stage_operation(
                    LinearXlaOperation::Extension(LinearXlaOperationExtension::WithShardingConstraint(op.clone())),
                    &[tangent_input],
                )?;
                check_count!("output", tangent_outputs, 1, TracingError);
                Ok(vec![JvpTracer::from_value(primal_outputs[0].clone(), tangent_outputs.remove(0))])
            }
        }
    }
}

impl<V> Display for LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
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
        delegate_extension!(self, [LinearJitCall, LinearShardMap, WithShardingConstraint], |op| op.name())
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        delegate_extension!(self, [LinearJitCall, LinearShardMap, WithShardingConstraint], |op| {
            op.infer_output_types(input_types)
        })
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        delegate_extension!(self, [LinearJitCall, LinearShardMap, WithShardingConstraint], |op| {
            op.render(formatter, indentation)
        })
    }
}

impl<V> InterpretableOperation<ArrayType, V> for LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
    LinearShardMapOperation<V>: InterpretableOperation<ArrayType, V>,
    WithShardingConstraintOperation: InterpretableOperation<ArrayType, V>,
    LinearJitCallOperation<V>: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        delegate_extension!(self, [LinearJitCall, LinearShardMap, WithShardingConstraint], |op| op.interpret(inputs))
    }
}

impl<V> LinearOperation<ArrayType, V, LinearXlaOperation<V>> for LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
    LinearShardMapOperation<V>: LinearOperation<ArrayType, V, LinearXlaOperation<V>>,
    WithShardingConstraintOperation: LinearOperation<ArrayType, V, LinearXlaOperation<V>>,
    LinearJitCallOperation<V>: LinearOperation<ArrayType, V, LinearXlaOperation<V>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V>>,
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V>>>, TracingError> {
        delegate_extension!(self, [LinearJitCall, LinearShardMap, WithShardingConstraint], |op| {
            op.transpose(context, output_cotangents)
        })
    }
}

impl<V> LinearOperationExtensionFamily<ArrayType, V> for LinearXlaOperationExtension<V>
where
    V: Traceable<ArrayType>,
{
    type CarrierForContext<C>
        = LinearXlaOperation<Tracer<C>>
    where
        C: Context<Type = ArrayType>;
}
