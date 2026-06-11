use std::cell::RefCell;
use std::fmt::Display;
use std::rc::Rc;

use ryft_core::batching::BatchingError;
use ryft_core::compilation::CapturedConstant;
use ryft_core::contexts::StagingContext;
use ryft_core::differentiation::{Cotangent, TransposableOperation};
use ryft_core::domains::Domain;
use ryft_core::macros::check_count;
use ryft_core::operations::constants::SupportsZero;
use ryft_core::operations::control_flow::{flat_program_input_types, flat_program_output_types};
use ryft_core::operations::{InterpretableOperation, Operation, OperationFormatter};
use ryft_core::parameters::Placeholder;
use ryft_core::programs::{Program, ProgramBuilder, ProgramError, Value};
use ryft_core::tracing::{AbstractTracingContext, Tracer, TracingContext};
use ryft_core::tracing_v2::batching::{ArrayBatch, BatchableOperation, BatchingContext, batch_input_metadata};
use ryft_core::tracing_v2::{
    ArrayOperation, DifferentiableOperation, DifferentiationContext, JvpTracer, LinearArrayOperation, ResidualFactor,
    TangentContext,
};
use ryft_core::types::{ArrayType, Size, TypeError, Typed};

use crate::experimental::domains::{XlaDomain, XlaTracer};
use crate::experimental::operations::{LinearShardMapOperation, ShardMapOperation, WithShardingConstraintOperation};

/// Backend-owned ordinary operations that extend the reusable core array operation set.
#[derive(Clone, Debug)]
pub enum XlaOperationExtension<V: Value<ArrayType>> {
    /// Call to a jitted XLA sub-program.
    JitCall(Box<JitCallOperation>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<V>>),

    /// XLA-specific `linear_shard_map` staged in ordinary traced programs.
    LinearShardMap(Box<LinearShardMapOperation<V>>),

    /// XLA-specific sharding constraint.
    WithShardingConstraint(WithShardingConstraintOperation),
}

/// Lifetime-free reference to a concrete XLA value captured by a compiled program.
pub type XlaConstant = CapturedConstant<ArrayType>;

/// Ordinary staged-op universe owned by the XLA backend.
impl<V: Value<ArrayType>> ryft_core::tracing_v2::operations::MaybeDot for XlaOperationExtension<V> {
    #[inline]
    fn dot_dimensions(&self) -> Option<&ryft_core::tracing_v2::DotDimensionNumbers> {
        // Higher-order XLA calls may contain dots in their bodies but are not themselves dot primitives, mirroring
        // how JAX's `dots_saveable` rematerialization policy matches only dot primitives.
        None
    }
}

impl<V: Value<ArrayType>> ryft_core::tracing_v2::rematerialization::MaybeRematerializationName
    for XlaOperationExtension<V>
{
    #[inline]
    fn rematerialization_name(&self) -> Option<&str> {
        None
    }
}

pub type XlaOperation = ArrayOperation<XlaConstant, ArrayType, XlaOperationExtension<XlaConstant>>;

/// Staged XLA program specialized to the backend-owned XLA op universe.
pub type XlaProgram<Input, Output> = Program<ArrayType, XlaConstant, XlaOperation, Input, Output>;

/// Program builder specialized to the backend-owned XLA op universe.
pub type XlaProgramBuilder = ProgramBuilder<ArrayType, XlaConstant, XlaOperation>;

/// Flat XLA program payload used by staged call operations.
pub type FlatXlaProgram = XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>;

/// Backend-owned linear operations that extend the reusable core linear array operation set.
#[derive(Clone, Debug)]
pub enum LinearXlaOperationExtension<V: Value<ArrayType>> {
    /// Linearized call to a jitted XLA sub-program.
    LinearJitCall(Box<LinearJitCallOperation<V>>),

    /// XLA-specific linear `shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<V>>),

    /// XLA-specific sharding constraint in tangent/cotangent programs.
    WithShardingConstraint(WithShardingConstraintOperation),
}

/// Linear staged-op universe owned by the XLA backend.
pub type LinearXlaOperation<V, C = V, Factor = V> =
    LinearArrayOperation<V, C, ArrayType, LinearXlaOperationExtension<V>, Factor, XlaOperation>;

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
pub struct LinearJitCallOperation<V: Value<ArrayType>> {
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

impl<V: Value<ArrayType>> LinearJitCallOperation<V> {
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

fn missing_traced_input() -> ProgramError {
    ProgramError::InvalidInputCount { expected: 1, actual: 0 }
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

fn build_jvp_call_program(program: &FlatXlaProgram) -> Result<FlatXlaProgram, ProgramError> {
    let input_types = flat_program_input_types(program);
    let signature = input_types.iter().cloned().chain(input_types.iter().cloned()).collect::<Vec<_>>();
    let token = XlaDomain::token();
    let (_, traced): (Vec<ArrayType>, FlatXlaProgram) = TracingContext::trace(
        token,
        |inputs: Vec<XlaTracer<'static, 'static>>| -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
            let input_count = inputs.len() / 2;
            let primals = inputs[..input_count].to_vec();
            let tangents = inputs[input_count..].to_vec();
            let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
            let linearized = context.linearize(
                |linearized_inputs| {
                    let linearization_context =
                        linearized_inputs.first().ok_or_else(missing_traced_input)?.context().clone();
                    linearization_context.stage_program(program, linearized_inputs)
                },
                primals,
            )?;
            let (_, pushforward) = linearized.into_parts();
            pushforward.apply(tangents)
        },
        signature,
    )?;
    traced.into_simplified()
}

fn build_pullback_call_program(program: &FlatXlaProgram) -> Result<FlatXlaProgram, ProgramError> {
    let input_types = flat_program_input_types(program);
    let output_types = flat_program_output_types(program);
    let signature = input_types.iter().cloned().chain(output_types.iter().cloned()).collect::<Vec<_>>();
    let token = XlaDomain::token();
    let (_, traced): (Vec<ArrayType>, FlatXlaProgram) = TracingContext::trace(
        token,
        |inputs: Vec<XlaTracer<'static, 'static>>| -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
            let input_count = input_types.len();
            let primals = inputs[..input_count].to_vec();
            let cotangents = inputs[input_count..].to_vec();
            let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
            let (_, pullback) = context.vjp(
                |linearized_inputs| {
                    let linearization_context =
                        linearized_inputs.first().ok_or_else(missing_traced_input)?.context().clone();
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
) -> Result<(FlatXlaProgram, Vec<Option<usize>>), ProgramError> {
    let logical_input_types = flat_program_input_types(program);
    check_count!("input", input_axes, logical_input_types.len(), ProgramError);
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

    let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
    let batched_program = builder
        .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
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
    ) -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
        let context = inputs.first().ok_or_else(missing_traced_input)?.context().clone();
        context.stage_operation(XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(self.clone()))), inputs)
    }

    /// Creates the linear call operation corresponding to this ordinary call at `primals`.
    fn linear_call_operation<V: Value<ArrayType>>(
        &self,
        primals: Vec<V>,
    ) -> Result<LinearJitCallOperation<V>, ProgramError> {
        Ok(LinearJitCallOperation::new(
            build_jvp_call_program(&self.program)?,
            build_pullback_call_program(&self.program)?,
            primals,
            flat_program_input_types(&self.program),
            flat_program_output_types(&self.program),
        ))
    }

    /// Returns the call operation and output-axis metadata for batching this call.
    fn batched_call_operation<V: Typed<ArrayType>>(
        &self,
        inputs: &[ArrayBatch<V>],
    ) -> Result<(Self, Vec<Option<usize>>), ProgramError> {
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
    fn jvp_from_primal_outputs<'jvp, E, V: Value<ArrayType>>(
        &self,
        context: &mut TangentContext<'jvp, E>,
        inputs: &[JvpTracer<'jvp, E>],
        primals: Vec<V>,
        primal_outputs: Vec<V>,
    ) -> Result<Vec<JvpTracer<'jvp, E>>, ProgramError>
    where
        E: DifferentiationContext<
                Tangent = V,
                LinearOperation<V, ResidualFactor<ArrayType, V>> = LinearXlaOperation<
                    V,
                    XlaConstant,
                    ResidualFactor<ArrayType, V>,
                >,
            > + Domain<Type = ArrayType, Value = V>
            + 'jvp,
    {
        let tangent_inputs = inputs
            .iter()
            .map(|input| context.materialize_tangent(input.tangent().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let linear_operation = self.linear_call_operation(primals)?;
        let operation: LinearXlaOperation<V, XlaConstant, ResidualFactor<ArrayType, V>> =
            LinearXlaOperation::Extension(LinearXlaOperationExtension::LinearJitCall(Box::new(linear_operation)));
        let tangent_outputs = context.stage_operation(operation, tangent_inputs.as_slice())?;
        check_count!("output", tangent_outputs, primal_outputs.len(), ProgramError);
        Ok(primal_outputs
            .into_iter()
            .zip(tangent_outputs)
            .map(|(primal, tangent)| JvpTracer::from_value(primal, tangent))
            .collect())
    }
}

impl<C> BatchableOperation<ArrayType, C> for JitCallOperation {
    fn batch(
        &self,
        _context: &C,
        inputs: &[ArrayBatch<ArrayType>],
    ) -> Result<Vec<ArrayBatch<ArrayType>>, ProgramError> {
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

impl<S, C> BatchableOperation<Tracer<S>, C> for JitCallOperation
where
    S: StagingContext<Type = ArrayType, Operation = XlaOperation>,
{
    fn batch(
        &self,
        _context: &C,
        inputs: &[ArrayBatch<Tracer<S>>],
    ) -> Result<Vec<ArrayBatch<Tracer<S>>>, ProgramError> {
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

impl<'domain, 'context, Capture> DifferentiableOperation<TracingContext<'domain, XlaDomain<'context>, Capture>>
    for JitCallOperation
where
    XlaDomain<'context>: 'domain,
    'context: 'domain,
    Capture: Value<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>,
        inputs: &[JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>],
    ) -> Result<Vec<JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>>, ProgramError>
    where
        TracingContext<'domain, XlaDomain<'context>, Capture>: 'jvp,
    {
        check_count!("input", inputs, flat_program_input_types(&self.program).len(), ProgramError);
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_outputs = context.differentiable().stage_operation(
            XlaOperation::Extension(XlaOperationExtension::JitCall(Box::new(self.clone()))),
            primals.as_slice(),
        )?;
        self.jvp_from_primal_outputs(context, inputs, primals, primal_outputs)
    }
}

impl<V: Value<ArrayType>> Operation<ArrayType> for LinearJitCallOperation<V> {
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

impl<C> InterpretableOperation<ArrayType, Tracer<C>> for LinearJitCallOperation<Tracer<C>>
where
    C: StagingContext<Type = ArrayType, Operation = XlaOperation>,
{
    fn interpret(&self, inputs: &[Tracer<C>]) -> Result<Vec<Tracer<C>>, ProgramError> {
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

impl<V: Value<ArrayType>> TransposableOperation<ArrayType, V, LinearXlaOperation<V, XlaConstant>>
    for LinearJitCallOperation<V>
where
    LinearXlaOperation<V, XlaConstant>: SupportsZero<ArrayType>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant>>,
        _input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant>>>, ProgramError> {
        check_count!("output", output_cotangents, self.output_types.len(), ProgramError);
        let mut cotangent_inputs = Vec::with_capacity(output_cotangents.len());
        for (cotangent, output_type) in output_cotangents.iter().zip(self.output_types.iter()) {
            match cotangent {
                Cotangent::Staged(cotangent) => cotangent_inputs.push(cotangent.clone()),
                Cotangent::Zero => {
                    let zero_outputs = context.stage_operation(
                        LinearXlaOperation::<V, XlaConstant>::zero_operation(output_type.clone()),
                        &[] as &[ryft_core::tracing::AbstractTracer<
                            'transpose,
                            ArrayType,
                            V,
                            LinearXlaOperation<V, XlaConstant>,
                        >],
                    )?;
                    check_count!("output", zero_outputs, 1, ProgramError);
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

impl Display for XlaOperationExtension<XlaConstant> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation<ArrayType> for XlaOperationExtension<XlaConstant> {
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

impl InterpretableOperation<ArrayType, XlaTracer<'static, 'static>> for XlaOperationExtension<XlaConstant> {
    fn interpret(
        &self,
        inputs: &[XlaTracer<'static, 'static>],
    ) -> Result<Vec<XlaTracer<'static, 'static>>, ProgramError> {
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
/// `ShardMap` and `LinearShardMap` carry inner programs whose proper batching requires lifting the captured body
/// through context-aware batching rules per instruction. These variants return
/// [`BatchingError::UnsupportedOperation`] so that programs which use shard_map can't be silently mis-batched;
/// programs that don't touch these ops batch correctly through this trait impl.
///
/// `WithShardingConstraint` is a unary identity with a sharding annotation. Batching it
/// requires extending the sharding to cover the new lane axis. We don't yet have a generic
/// "insert a replicated mesh dim at position k" helper on [`Sharding`](ryft_core::sharding::Sharding),
/// so for now this variant also returns [`BatchingError::UnsupportedOperation`]. A future
/// follow-up can implement the extension once the sharding helper exists.
impl<V, C> BatchableOperation<V, C> for XlaOperationExtension<XlaConstant>
where
    V: Value<ArrayType>,
    JitCallOperation: BatchableOperation<V, C>,
{
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, ProgramError> {
        match self {
            Self::JitCall(op) => op.batch(context, inputs),
            _ => Err(BatchingError::UnsupportedOperation {
                message: format!("missing batching rule for operation '{}'", self.name()),
            }
            .into()),
        }
    }
}

impl<'domain, 'context, Capture> DifferentiableOperation<TracingContext<'domain, XlaDomain<'context>, Capture>>
    for XlaOperationExtension<XlaConstant>
where
    XlaDomain<'context>: 'domain,
    'context: 'domain,
    Capture: Value<ArrayType>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>,
        inputs: &[JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>],
    ) -> Result<Vec<JvpTracer<'jvp, TracingContext<'domain, XlaDomain<'context>, Capture>>>, ProgramError>
    where
        TracingContext<'domain, XlaDomain<'context>, Capture>: 'jvp,
    {
        match self {
            Self::JitCall(op) => op.jvp(context, inputs),
            Self::ShardMap(op) => {
                let traced_op = ShardMapOperation::<Tracer<TracingContext<'domain, XlaDomain<'context>, Capture>>>::new(
                    op.body().clone(),
                );
                traced_op.jvp_with_context(context.differentiable(), context, inputs)
            }
            Self::LinearShardMap(op) => op.jvp_traced_with_context(context.differentiable(), context, inputs),
            Self::WithShardingConstraint(op) => {
                check_count!("input", inputs, 1, ProgramError);
                let input = &inputs[0];
                let primal_outputs = input.primal().context().stage_operation(
                    XlaOperation::Extension(XlaOperationExtension::WithShardingConstraint(op.clone())),
                    &[input.primal()],
                )?;
                check_count!("output", primal_outputs, 1, ProgramError);
                let tangent_input = context.materialize_tangent(input.tangent().clone())?;
                let mut tangent_outputs = context.stage_operation(
                    LinearXlaOperation::Extension(LinearXlaOperationExtension::WithShardingConstraint(op.clone())),
                    &[tangent_input],
                )?;
                check_count!("output", tangent_outputs, 1, ProgramError);
                Ok(vec![JvpTracer::from_value(primal_outputs[0].clone(), tangent_outputs.remove(0))])
            }
        }
    }
}

impl<V: Value<ArrayType>> Display for LinearXlaOperationExtension<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl<V: Value<ArrayType>> Operation<ArrayType> for LinearXlaOperationExtension<V> {
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

impl<V: Value<ArrayType>> InterpretableOperation<ArrayType, V> for LinearXlaOperationExtension<V>
where
    LinearShardMapOperation<V>: InterpretableOperation<ArrayType, V>,
    WithShardingConstraintOperation: InterpretableOperation<ArrayType, V>,
    LinearJitCallOperation<V>: InterpretableOperation<ArrayType, V>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        delegate_extension!(self, [LinearJitCall, LinearShardMap, WithShardingConstraint], |op| op.interpret(inputs))
    }
}

impl<V: Value<ArrayType>> TransposableOperation<ArrayType, V, LinearXlaOperation<V, XlaConstant>>
    for LinearXlaOperationExtension<V>
where
    LinearShardMapOperation<V>: TransposableOperation<ArrayType, V, LinearXlaOperation<V, XlaConstant>>,
    WithShardingConstraintOperation: TransposableOperation<ArrayType, V, LinearXlaOperation<V, XlaConstant>>,
    LinearJitCallOperation<V>: TransposableOperation<ArrayType, V, LinearXlaOperation<V, XlaConstant>>,
{
    fn transpose<'transpose>(
        &self,
        context: &mut AbstractTracingContext<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant>>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant>>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, LinearXlaOperation<V, XlaConstant>>>, ProgramError> {
        delegate_extension!(self, [LinearJitCall, LinearShardMap, WithShardingConstraint], |op| {
            op.transpose(context, input_types, output_cotangents)
        })
    }
}
