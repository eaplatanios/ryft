use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use ryft_core::{
    parameters::Placeholder,
    tracing::{InterpretableOperation, Operation, Program, ProgramBuilder, TracingError},
    tracing_v2::{
        CustomPrimitive, DifferentiableOperation, DifferentiationError, LinearCustomPrimitive, LinearOperation,
        LinearPrimitiveOperation, LinearTerm, Tracer,
        engines::DifferentiableEngine,
        forward::{Differentiable, EngineTangent, JvpTracer},
        linear::{
            Linearized, linearize_program, linearize_traced_program, transpose_linear_program,
            transpose_linear_program_with_output_examples, transpose_traced_linear_program,
        },
        operations::{
            AddOperation, AddTracingOperation, ConditionOperation, ConditionPredicate, ControlFlowError, CosOperation,
            CosTracingOperation, CustomTracingOperation, FlatTracedRematerialize, LeadingAxisTracingOperation,
            LeftMatMulOperation, LeftMatMulTracingOperation, LinearRematerializeOperation, MatMulOperation,
            MatMulTracingOperation, MatrixTransposeOperation, MatrixTransposeTracingOperation, MulOperation,
            MulTracingOperation, NegOperation, NegTracingOperation, RematerializeOperation,
            RematerializeTracingOperation, ReshapeOperation, ReshapeTracingOperation, RightMatMulOperation,
            RightMatMulTracingOperation, ScaleOperation, ScaleTracingOperation, ScanError, ScanOperation, ScanOptions,
            ScanTracingOperation, ScatterLeadingAxisSliceOperation, SinOperation, SinTracingOperation,
            SliceLeadingAxisOperation, StackLeadingAxisOperation, WhileOperation,
            left_matmul::left_matmul_abstract_eval, lift_jit_constant, right_matmul::right_matmul_abstract_eval,
            scan_with_options, scan_without_xs_with_options,
        },
    },
    types::{ArrayType, TypeError, Typed},
};

use crate::experimental::{
    engines::XlaEngine,
    lowering::{
        LoweringError, ShardMapMlirLowerer, StableHloCustomLowering, StableHloCustomLoweringExtension,
        lower_scan_to_while,
    },
    operations::{
        LinearShardMapOperation, ShardMapCustomReplayExtension, ShardMapOperation, ShardMapReplayContext,
        WithShardingConstraintOperation,
    },
    shard_map::{ShardMapTensor, ShardMapTracer},
};

type XlaLinearOperation = LinearPrimitiveOperation<ShardMapTensor>;

fn make_linear_xla_rematerialize<E>(
    engine: &E,
    body: &FlatTracedRematerialize<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    input_primals: Vec<ShardMapTensor>,
) -> Result<LinearRematerializeOperation<ArrayType, ShardMapTensor>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = ShardMapTensor, LinearOperation = XlaLinearOperation>,
{
    let body_program = body.program();
    let output_primals = body_program.interpret(input_primals.clone())?;
    let pushforward = linearize_program(engine, body_program, input_primals)?;
    let pullback = transpose_linear_program_with_output_examples(engine, &pushforward, output_primals.as_slice())?;
    Ok(LinearRematerializeOperation::new(
        FlatTracedRematerialize::from_parts(body.input_types().to_vec(), body.output_types().to_vec(), pushforward),
        FlatTracedRematerialize::from_parts(body.output_types().to_vec(), body.input_types().to_vec(), pullback),
    ))
}

fn replay_xla_program_with_tracers(
    program: &ryft_core::tracing::Program<
        ArrayType,
        ShardMapTensor,
        XlaPrimitiveOperation,
        Vec<ShardMapTensor>,
        Vec<ShardMapTensor>,
    >,
    inputs: Vec<ShardMapTracer>,
) -> Result<Vec<ShardMapTracer>, TracingError> {
    let exemplar = inputs.first().cloned();
    let mut values = vec![None; program.atoms.len()];
    for (atom_id, value) in program.input_ids.iter().copied().zip(inputs) {
        values[atom_id.index] = Some(value);
    }
    for (atom_index, atom) in program.atoms.iter().enumerate() {
        if let ryft_core::tracing::Atom::Constant(value) = atom {
            let Some(exemplar) = exemplar.as_ref() else {
                return Err(ScanError::MissingTracedInvocationContext.into());
            };
            values[atom_index] = Some(lift_jit_constant(value, exemplar));
        }
    }
    for instruction in program.instructions.iter() {
        let inputs = instruction
            .inputs
            .iter()
            .map(|input| values[input.index].clone().ok_or(TracingError::UnboundAtomId { id: *input }))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = instruction.operation.interpret(inputs.as_slice())?;
        if outputs.len() != instruction.outputs.len() {
            return Err(TracingError::InvalidOutputCount { expected: instruction.outputs.len(), got: outputs.len() });
        }
        for (output, value) in instruction.outputs.iter().copied().zip(outputs) {
            values[output.index] = Some(value);
        }
    }
    program
        .output_ids
        .iter()
        .map(|output| values[output.index].clone().ok_or(TracingError::UnboundAtomId { id: *output }))
        .collect()
}

#[derive(Clone)]
struct LinearizedScanJvpOperation {
    scan: ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    primal_inputs: Vec<ShardMapTracer>,
}

#[derive(Clone)]
struct LinearizedScanTransposeOperation {
    scan: ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    primal_inputs: Vec<ShardMapTracer>,
}

#[derive(Clone)]
struct TensorLinearizedScanJvpOperation {
    scan: ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    primal_inputs: Vec<ShardMapTensor>,
}

#[derive(Clone)]
struct TensorLinearizedScanTransposeOperation {
    scan: ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    primal_inputs: Vec<ShardMapTensor>,
}

impl LinearizedScanJvpOperation {
    fn new(
        scan: ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
        primal_inputs: Vec<ShardMapTracer>,
    ) -> Self {
        Self { scan, primal_inputs }
    }

    fn input_types(&self) -> Vec<ArrayType> {
        self.scan.body().carry_types().iter().chain(self.scan.body().xs_types().iter()).cloned().collect()
    }

    fn output_types(&self) -> Vec<ArrayType> {
        self.scan.body().carry_types().iter().chain(self.scan.body().ys_types().iter()).cloned().collect()
    }

    fn scan_options(&self) -> ScanOptions {
        ScanOptions::default()
            .with_length(self.scan.body().length())
            .with_reverse(self.scan.reverse())
            .with_unroll(self.scan.unroll())
            .with_split_transpose(self.scan.split_transpose())
    }

    fn to_linear_custom_primitive(&self) -> LinearCustomPrimitive<ArrayType, ShardMapTracer> {
        CustomPrimitive::new(self.clone())
            .with_transpose_rule(self.clone())
            .into_linear()
            .expect("linearized scan JVP custom primitive always registers a transpose rule")
    }
}

impl TensorLinearizedScanJvpOperation {
    fn new(
        scan: ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
        primal_inputs: Vec<ShardMapTensor>,
    ) -> Self {
        Self { scan, primal_inputs }
    }

    fn input_types(&self) -> Vec<ArrayType> {
        self.scan.body().carry_types().iter().chain(self.scan.body().xs_types().iter()).cloned().collect()
    }

    fn output_types(&self) -> Vec<ArrayType> {
        self.scan.body().carry_types().iter().chain(self.scan.body().ys_types().iter()).cloned().collect()
    }

    fn scan_options(&self) -> ScanOptions {
        ScanOptions::default()
            .with_length(self.scan.body().length())
            .with_reverse(self.scan.reverse())
            .with_unroll(self.scan.unroll())
            .with_split_transpose(self.scan.split_transpose())
    }

    fn to_linear_custom_primitive(&self) -> LinearCustomPrimitive<ArrayType, ShardMapTensor> {
        CustomPrimitive::new(self.clone())
            .with_transpose_rule(self.clone())
            .with_extension(StableHloCustomLoweringExtension::new(Arc::new(self.clone())))
            .into_linear()
            .expect("tensor linearized scan JVP custom primitive always registers a transpose rule")
    }
}

impl TensorLinearizedScanTransposeOperation {
    fn new(
        scan: ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
        primal_inputs: Vec<ShardMapTensor>,
    ) -> Self {
        Self { scan, primal_inputs }
    }

    fn input_types(&self) -> Vec<ArrayType> {
        self.scan.body().carry_types().iter().chain(self.scan.body().ys_types().iter()).cloned().collect()
    }

    fn output_types(&self) -> Vec<ArrayType> {
        self.scan.body().carry_types().iter().chain(self.scan.body().xs_types().iter()).cloned().collect()
    }

    fn scan_options(&self) -> ScanOptions {
        ScanOptions::default()
            .with_length(self.scan.body().length())
            .with_reverse(self.scan.reverse())
            .with_unroll(self.scan.unroll())
            .with_split_transpose(self.scan.split_transpose())
    }

    fn reverse_scan_options(&self) -> ScanOptions {
        ScanOptions::default()
            .with_length(self.scan.body().length())
            .with_reverse(!self.scan.reverse())
            .with_unroll(self.scan.unroll())
            .with_split_transpose(self.scan.split_transpose())
    }

    fn to_linear_custom_primitive(&self) -> LinearCustomPrimitive<ArrayType, ShardMapTensor> {
        CustomPrimitive::new(self.clone())
            .with_transpose_rule(self.clone())
            .into_linear()
            .expect("tensor linearized scan transpose custom primitive always registers a transpose rule")
    }
}

impl LinearizedScanTransposeOperation {
    fn new(
        scan: ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
        primal_inputs: Vec<ShardMapTracer>,
    ) -> Self {
        Self { scan, primal_inputs }
    }

    fn input_types(&self) -> Vec<ArrayType> {
        self.scan.body().carry_types().iter().chain(self.scan.body().ys_types().iter()).cloned().collect()
    }

    fn output_types(&self) -> Vec<ArrayType> {
        self.scan.body().carry_types().iter().chain(self.scan.body().xs_types().iter()).cloned().collect()
    }

    fn scan_options(&self) -> ScanOptions {
        ScanOptions::default()
            .with_length(self.scan.body().length())
            .with_reverse(self.scan.reverse())
            .with_unroll(self.scan.unroll())
            .with_split_transpose(self.scan.split_transpose())
    }

    fn reverse_scan_options(&self) -> ScanOptions {
        ScanOptions::default()
            .with_length(self.scan.body().length())
            .with_reverse(!self.scan.reverse())
            .with_unroll(self.scan.unroll())
            .with_split_transpose(self.scan.split_transpose())
    }

    fn to_linear_custom_primitive(&self) -> LinearCustomPrimitive<ArrayType, ShardMapTracer> {
        CustomPrimitive::new(self.clone())
            .with_transpose_rule(self.clone())
            .into_linear()
            .expect("linearized scan transpose custom primitive always registers a transpose rule")
    }
}

impl Debug for LinearizedScanJvpOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LinearizedScanJvp")
    }
}

impl Debug for LinearizedScanTransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LinearizedScanTranspose")
    }
}

impl Debug for TensorLinearizedScanJvpOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("TensorLinearizedScanJvp")
    }
}

impl Debug for TensorLinearizedScanTransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("TensorLinearizedScanTranspose")
    }
}

impl Display for LinearizedScanJvpOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("linear_scan_jvp")
    }
}

impl Display for LinearizedScanTransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("linear_scan_transpose")
    }
}

impl Display for TensorLinearizedScanJvpOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("linear_scan_jvp")
    }
}

impl Display for TensorLinearizedScanTransposeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("linear_scan_transpose")
    }
}

impl Operation<ArrayType> for LinearizedScanJvpOperation {
    fn name(&self) -> &'static str {
        "linear_scan_jvp"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let expected_input_types = self.input_types();
        if input_types != expected_input_types.as_slice() {
            return Err(TypeError {
                message: "linear scan JVP input types do not match the captured scan signature".to_string(),
            });
        }
        Ok(self.output_types())
    }
}

impl Operation<ArrayType> for TensorLinearizedScanJvpOperation {
    fn name(&self) -> &'static str {
        "linear_scan_jvp"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let expected_input_types = self.input_types();
        if input_types != expected_input_types.as_slice() {
            return Err(TypeError {
                message: "linear scan JVP input types do not match the captured scan signature".to_string(),
            });
        }
        Ok(self.output_types())
    }
}

impl Operation<ArrayType> for TensorLinearizedScanTransposeOperation {
    fn name(&self) -> &'static str {
        "linear_scan_transpose"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let expected_input_types = self.input_types();
        if input_types != expected_input_types.as_slice() {
            return Err(TypeError {
                message: "linear scan transpose input types do not match the captured scan signature".to_string(),
            });
        }
        Ok(self.output_types())
    }
}

impl Operation<ArrayType> for LinearizedScanTransposeOperation {
    fn name(&self) -> &'static str {
        "linear_scan_transpose"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        let expected_input_types = self.input_types();
        if input_types != expected_input_types.as_slice() {
            return Err(TypeError {
                message: "linear scan transpose input types do not match the captured scan signature".to_string(),
            });
        }
        Ok(self.output_types())
    }
}

impl InterpretableOperation<ArrayType, ShardMapTensor> for TensorLinearizedScanJvpOperation {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        let _ = self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let carry_count = self.scan.body().carry_types().len();
        let x_count = self.scan.body().x_types().len();
        let y_count = self.scan.body().y_types().len();
        let primal_carry = self.primal_inputs[..carry_count].to_vec();
        let primal_xs = self.primal_inputs[carry_count..].to_vec();
        let tangent_carry = inputs[..carry_count].to_vec();
        let tangent_xs = inputs[carry_count..].to_vec();
        let mut initial_carry = Vec::with_capacity(carry_count * 2);
        initial_carry.extend(primal_carry);
        initial_carry.extend(tangent_carry);

        let body_program = self.scan.body().program().clone();
        let scan_body = move |(combined_carry, combined_x): (Vec<ShardMapTensor>, Vec<ShardMapTensor>)| {
            let primal_carry = combined_carry[..carry_count].to_vec();
            let tangent_carry = combined_carry[carry_count..].to_vec();
            let primal_x = combined_x[..x_count].to_vec();
            let tangent_x = combined_x[x_count..].to_vec();
            let mut body_primals = Vec::with_capacity(carry_count + x_count);
            body_primals.extend(primal_carry);
            body_primals.extend(primal_x);
            let mut body_tangents = Vec::with_capacity(carry_count + x_count);
            body_tangents.extend(tangent_carry);
            body_tangents.extend(tangent_x);
            let pushforward = linearize_program(XlaEngine::token(), &body_program, body_primals.clone())
                .expect("captured scan body should linearize while interpreting compact tensor scan JVP");
            let body_tangents = pushforward
                .interpret(body_tangents)
                .expect("captured scan body pushforward should replay while interpreting compact tensor scan JVP");
            let body_primals = body_program
                .interpret(body_primals)
                .expect("captured scan body should replay primals while interpreting compact tensor scan JVP");
            let mut next_carry = Vec::with_capacity(carry_count * 2);
            next_carry.extend(body_primals[..carry_count].iter().cloned());
            next_carry.extend(body_tangents[..carry_count].iter().cloned());
            let tangent_y = body_tangents[carry_count..carry_count + y_count].to_vec();
            (next_carry, tangent_y)
        };

        let (combined_carry, tangent_ys): (Vec<ShardMapTensor>, Vec<ShardMapTensor>) = if x_count == 0 {
            scan_without_xs_with_options(
                move |(combined_carry, ()): (Vec<ShardMapTensor>, ())| scan_body((combined_carry, Vec::new())),
                initial_carry,
                self.scan_options(),
            )?
        } else {
            let mut combined_xs = Vec::with_capacity(x_count * 2);
            combined_xs.extend(primal_xs);
            combined_xs.extend(tangent_xs);
            scan_with_options(scan_body, initial_carry, combined_xs, self.scan_options())?
        };

        let mut outputs = combined_carry[carry_count..].to_vec();
        outputs.extend(tangent_ys);
        Ok(outputs)
    }
}

impl InterpretableOperation<ArrayType, ShardMapTensor> for TensorLinearizedScanTransposeOperation {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        let _ = self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let carry_count = self.scan.body().carry_types().len();
        let x_count = self.scan.body().x_types().len();
        let y_count = self.scan.body().y_types().len();
        let primal_carry = self.primal_inputs[..carry_count].to_vec();
        let primal_xs = self.primal_inputs[carry_count..].to_vec();
        let output_carry_cotangents = inputs[..carry_count].to_vec();
        let ys_cotangents = inputs[carry_count..].to_vec();

        let body_program = self.scan.body().program().clone();
        let (_, carry_history): (Vec<ShardMapTensor>, Vec<ShardMapTensor>) = if carry_count == 0 {
            (Vec::new(), Vec::new())
        } else if x_count == 0 {
            let carry_history_body = {
                let body_program = body_program.clone();
                move |(carry, ()): (Vec<ShardMapTensor>, ())| {
                    let body_outputs = body_program
                        .interpret(carry.clone())
                        .expect("captured scan body should replay while interpreting compact tensor scan transpose");
                    (body_outputs[..carry_count].to_vec(), carry)
                }
            };
            scan_without_xs_with_options(carry_history_body, primal_carry.clone(), self.scan_options())?
        } else {
            let carry_history_body = {
                let body_program = body_program.clone();
                move |(carry, x): (Vec<ShardMapTensor>, Vec<ShardMapTensor>)| {
                    let mut body_inputs = Vec::with_capacity(carry_count + x_count);
                    body_inputs.extend(carry.clone());
                    body_inputs.extend(x);
                    let body_outputs = body_program
                        .interpret(body_inputs)
                        .expect("captured scan body should replay while interpreting compact tensor scan transpose");
                    (body_outputs[..carry_count].to_vec(), carry)
                }
            };
            scan_with_options(carry_history_body, primal_carry.clone(), primal_xs.clone(), self.scan_options())?
        };

        let mut transpose_xs = Vec::with_capacity(carry_count + x_count + y_count);
        transpose_xs.extend(carry_history);
        transpose_xs.extend(primal_xs);
        transpose_xs.extend(ys_cotangents);
        let transpose_body_program = self.scan.body().program().clone();
        let transpose_body = move |(carry_cotangents, xs): (Vec<ShardMapTensor>, Vec<ShardMapTensor>)| {
            let carry_primals = xs[..carry_count].to_vec();
            let x_primals = xs[carry_count..carry_count + x_count].to_vec();
            let y_cotangents = xs[carry_count + x_count..].to_vec();
            let mut body_primals = Vec::with_capacity(carry_count + x_count);
            body_primals.extend(carry_primals);
            body_primals.extend(x_primals);
            let body_outputs = transpose_body_program
                .interpret(body_primals.clone())
                .expect("captured scan body should replay primals while interpreting compact tensor scan transpose");
            let pushforward = linearize_program(XlaEngine::token(), &transpose_body_program, body_primals)
                .expect("captured scan body should linearize while interpreting compact tensor scan transpose");
            let pullback = transpose_linear_program(XlaEngine::token(), &pushforward).expect(
                "captured scan body pushforward should transpose while interpreting compact tensor scan transpose",
            );
            let mut body_output_cotangents = Vec::with_capacity(carry_count + y_count);
            body_output_cotangents.extend(carry_cotangents);
            body_output_cotangents.extend(y_cotangents);
            debug_assert_eq!(body_outputs.len(), body_output_cotangents.len());
            let body_input_cotangents = pullback
                .interpret(body_output_cotangents)
                .expect("captured scan body pullback should replay while interpreting compact tensor scan transpose");
            let next_carry_cotangents = body_input_cotangents[..carry_count].to_vec();
            let x_cotangents = body_input_cotangents[carry_count..].to_vec();
            (next_carry_cotangents, x_cotangents)
        };

        let (input_carry_cotangents, mut xs_cotangents): (Vec<ShardMapTensor>, Vec<ShardMapTensor>) = if transpose_xs
            .is_empty()
        {
            scan_without_xs_with_options(
                move |(carry_cotangents, ()): (Vec<ShardMapTensor>, ())| transpose_body((carry_cotangents, Vec::new())),
                output_carry_cotangents,
                self.reverse_scan_options(),
            )?
        } else {
            scan_with_options(transpose_body, output_carry_cotangents, transpose_xs, self.reverse_scan_options())?
        };

        if self.scan.split_transpose() && !xs_cotangents.is_empty() {
            let (_, split_xs_cotangents): ((), Vec<ShardMapTensor>) = scan_with_options(
                |((), xs_cotangent): ((), Vec<ShardMapTensor>)| ((), xs_cotangent),
                (),
                xs_cotangents,
                ScanOptions::default()
                    .with_length(self.scan.body().length())
                    .with_unroll(self.scan.unroll())
                    .with_split_transpose(true),
            )?;
            xs_cotangents = split_xs_cotangents;
        }

        let mut outputs = input_carry_cotangents;
        outputs.extend(xs_cotangents);
        Ok(outputs)
    }
}

impl InterpretableOperation<ArrayType, ShardMapTracer> for LinearizedScanJvpOperation {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TracingError> {
        let _ = self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let carry_count = self.scan.body().carry_types().len();
        let x_count = self.scan.body().x_types().len();
        let y_count = self.scan.body().y_types().len();
        let primal_carry = self.primal_inputs[..carry_count].to_vec();
        let primal_xs = self.primal_inputs[carry_count..].to_vec();
        let tangent_carry = inputs[..carry_count].to_vec();
        let tangent_xs = inputs[carry_count..].to_vec();
        let mut initial_carry = Vec::with_capacity(carry_count * 2);
        initial_carry.extend(primal_carry);
        initial_carry.extend(tangent_carry);

        let body_program = self.scan.body().program().clone();
        let scan_body = move |(combined_carry, combined_x): (Vec<ShardMapTracer>, Vec<ShardMapTracer>)| {
            let primal_carry = combined_carry[..carry_count].to_vec();
            let tangent_carry = combined_carry[carry_count..].to_vec();
            let primal_x = combined_x[..x_count].to_vec();
            let tangent_x = combined_x[x_count..].to_vec();
            let mut body_primals = Vec::with_capacity(carry_count + x_count);
            body_primals.extend(primal_carry);
            body_primals.extend(primal_x);
            let mut body_tangents = Vec::with_capacity(carry_count + x_count);
            body_tangents.extend(tangent_carry);
            body_tangents.extend(tangent_x);
            let exemplar_body_primal =
                body_primals.first().cloned().expect("captured scan body should have at least one primal input");
            let (body_primals, pushforward) = linearize_traced_program(
                XlaEngine::token(),
                exemplar_body_primal.builder.clone(),
                &body_program,
                body_primals,
            )
            .expect("captured scan body should linearize while tracing compact scan JVP");
            let body_tangents = pushforward
                .interpret(body_tangents)
                .expect("captured scan body pushforward should replay while tracing compact scan JVP");
            let mut next_carry = Vec::with_capacity(carry_count * 2);
            next_carry.extend(body_primals[..carry_count].iter().cloned());
            next_carry.extend(body_tangents[..carry_count].iter().cloned());
            let tangent_y = body_tangents[carry_count..carry_count + y_count].to_vec();
            (next_carry, tangent_y)
        };

        let (combined_carry, tangent_ys): (Vec<ShardMapTracer>, Vec<ShardMapTracer>) = if x_count == 0 {
            scan_without_xs_with_options(
                move |(combined_carry, ()): (Vec<ShardMapTracer>, ())| scan_body((combined_carry, Vec::new())),
                initial_carry,
                self.scan_options(),
            )?
        } else {
            let mut combined_xs = Vec::with_capacity(x_count * 2);
            combined_xs.extend(primal_xs);
            combined_xs.extend(tangent_xs);
            scan_with_options(scan_body, initial_carry, combined_xs, self.scan_options())?
        };

        let mut outputs = combined_carry[carry_count..].to_vec();
        outputs.extend(tangent_ys);
        Ok(outputs)
    }
}

impl InterpretableOperation<ArrayType, ShardMapTracer> for LinearizedScanTransposeOperation {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TracingError> {
        let _ = self.infer_output_types(&inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>())?;
        let carry_count = self.scan.body().carry_types().len();
        let x_count = self.scan.body().x_types().len();
        let y_count = self.scan.body().y_types().len();
        let primal_carry = self.primal_inputs[..carry_count].to_vec();
        let primal_xs = self.primal_inputs[carry_count..].to_vec();
        let output_carry_cotangents = inputs[..carry_count].to_vec();
        let ys_cotangents = inputs[carry_count..].to_vec();

        let body_program = self.scan.body().program().clone();
        let (_, carry_history): (Vec<ShardMapTracer>, Vec<ShardMapTracer>) = if carry_count == 0 {
            (Vec::new(), Vec::new())
        } else if x_count == 0 {
            let carry_history_body = {
                let body_program = body_program.clone();
                move |(carry, ()): (Vec<ShardMapTracer>, ())| {
                    let body_outputs = replay_xla_program_with_tracers(&body_program, carry.clone())
                        .expect("captured scan body should replay while tracing compact scan transpose carry history");
                    (body_outputs[..carry_count].to_vec(), carry)
                }
            };
            scan_without_xs_with_options(carry_history_body, primal_carry.clone(), self.scan_options())?
        } else {
            let carry_history_body = {
                let body_program = body_program.clone();
                move |(carry, x): (Vec<ShardMapTracer>, Vec<ShardMapTracer>)| {
                    let mut body_inputs = Vec::with_capacity(carry_count + x_count);
                    body_inputs.extend(carry.clone());
                    body_inputs.extend(x);
                    let body_outputs = replay_xla_program_with_tracers(&body_program, body_inputs)
                        .expect("captured scan body should replay while tracing compact scan transpose carry history");
                    (body_outputs[..carry_count].to_vec(), carry)
                }
            };
            scan_with_options(carry_history_body, primal_carry.clone(), primal_xs.clone(), self.scan_options())?
        };

        let mut transpose_xs = Vec::with_capacity(carry_count + x_count + y_count);
        transpose_xs.extend(carry_history);
        transpose_xs.extend(primal_xs);
        transpose_xs.extend(ys_cotangents);
        let transpose_body_program = self.scan.body().program().clone();
        let transpose_body = move |(carry_cotangents, xs): (Vec<ShardMapTracer>, Vec<ShardMapTracer>)| {
            let carry_primals = xs[..carry_count].to_vec();
            let x_primals = xs[carry_count..carry_count + x_count].to_vec();
            let y_cotangents = xs[carry_count + x_count..].to_vec();
            let mut body_primals = Vec::with_capacity(carry_count + x_count);
            body_primals.extend(carry_primals);
            body_primals.extend(x_primals);
            let exemplar_body_primal =
                body_primals.first().cloned().expect("captured scan body should have at least one primal input");
            let (body_outputs, pushforward) = linearize_traced_program(
                XlaEngine::token(),
                exemplar_body_primal.builder.clone(),
                &transpose_body_program,
                body_primals,
            )
            .expect("captured scan body should linearize while tracing compact scan transpose");
            let pullback =
                transpose_traced_linear_program(XlaEngine::token(), exemplar_body_primal.builder.clone(), &pushforward)
                    .expect("captured scan body pushforward should transpose while tracing compact scan transpose");
            let mut body_output_cotangents = Vec::with_capacity(carry_count + y_count);
            body_output_cotangents.extend(carry_cotangents);
            body_output_cotangents.extend(y_cotangents);
            debug_assert_eq!(body_outputs.len(), body_output_cotangents.len());
            let body_input_cotangents = pullback
                .interpret(body_output_cotangents)
                .expect("captured scan body pullback should replay while tracing compact scan transpose");
            let next_carry_cotangents = body_input_cotangents[..carry_count].to_vec();
            let x_cotangents = body_input_cotangents[carry_count..].to_vec();
            (next_carry_cotangents, x_cotangents)
        };

        let (input_carry_cotangents, mut xs_cotangents): (Vec<ShardMapTracer>, Vec<ShardMapTracer>) = if transpose_xs
            .is_empty()
        {
            scan_without_xs_with_options(
                move |(carry_cotangents, ()): (Vec<ShardMapTracer>, ())| transpose_body((carry_cotangents, Vec::new())),
                output_carry_cotangents,
                self.reverse_scan_options(),
            )?
        } else {
            scan_with_options(transpose_body, output_carry_cotangents, transpose_xs, self.reverse_scan_options())?
        };

        if self.scan.split_transpose() && !xs_cotangents.is_empty() {
            let (_, split_xs_cotangents): ((), Vec<ShardMapTracer>) = scan_with_options(
                |((), xs_cotangent): ((), Vec<ShardMapTracer>)| ((), xs_cotangent),
                (),
                xs_cotangents,
                ScanOptions::default()
                    .with_length(self.scan.body().length())
                    .with_unroll(self.scan.unroll())
                    .with_split_transpose(true),
            )?;
            xs_cotangents = split_xs_cotangents;
        }

        let mut outputs = input_carry_cotangents;
        outputs.extend(xs_cotangents);
        Ok(outputs)
    }
}

impl LinearOperation<ArrayType, ShardMapTensor> for TensorLinearizedScanJvpOperation {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, ShardMapTensor>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, ShardMapTensor>>>, TracingError> {
        let builder = output_cotangents.first().ok_or(ScanError::MissingTracedInvocationContext)?.builder.clone();
        let transpose = TensorLinearizedScanTransposeOperation::new(self.scan.clone(), self.primal_inputs.clone());
        Ok(LinearTerm::apply_staged_op(
            builder,
            output_cotangents,
            LinearPrimitiveOperation::Custom(Arc::new(transpose.to_linear_custom_primitive())),
            self.input_types().len(),
        )?
        .into_iter()
        .map(Some)
        .collect())
    }
}

impl LinearOperation<ArrayType, ShardMapTensor> for TensorLinearizedScanTransposeOperation {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, ShardMapTensor>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, ShardMapTensor>>>, TracingError> {
        let builder = output_cotangents.first().ok_or(ScanError::MissingTracedInvocationContext)?.builder.clone();
        let jvp = TensorLinearizedScanJvpOperation::new(self.scan.clone(), self.primal_inputs.clone());
        Ok(LinearTerm::apply_staged_op(
            builder,
            output_cotangents,
            LinearPrimitiveOperation::Custom(Arc::new(jvp.to_linear_custom_primitive())),
            self.input_types().len(),
        )?
        .into_iter()
        .map(Some)
        .collect())
    }
}

impl LinearOperation<ArrayType, ShardMapTracer> for LinearizedScanJvpOperation {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, ShardMapTracer>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, ShardMapTracer>>>, TracingError> {
        let builder = output_cotangents.first().ok_or(ScanError::MissingTracedInvocationContext)?.builder.clone();
        let transpose = LinearizedScanTransposeOperation::new(self.scan.clone(), self.primal_inputs.clone());
        Ok(LinearTerm::apply_staged_op(
            builder,
            output_cotangents,
            LinearPrimitiveOperation::Custom(Arc::new(transpose.to_linear_custom_primitive())),
            self.input_types().len(),
        )?
        .into_iter()
        .map(Some)
        .collect())
    }
}

impl LinearOperation<ArrayType, ShardMapTracer> for LinearizedScanTransposeOperation {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, ShardMapTracer>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, ShardMapTracer>>>, TracingError> {
        let builder = output_cotangents.first().ok_or(ScanError::MissingTracedInvocationContext)?.builder.clone();
        let jvp = LinearizedScanJvpOperation::new(self.scan.clone(), self.primal_inputs.clone());
        Ok(LinearTerm::apply_staged_op(
            builder,
            output_cotangents,
            LinearPrimitiveOperation::Custom(Arc::new(jvp.to_linear_custom_primitive())),
            self.input_types().len(),
        )?
        .into_iter()
        .map(Some)
        .collect())
    }
}

fn interpret_xla_scan_linearized_jit(
    scan: &ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    inputs: &[Linearized<ShardMapTracer>],
) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError> {
    let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
    let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
    let Some(exemplar_primal_input) = primal_inputs.first().cloned() else {
        return if scan.body().carry_types().is_empty() && scan.body().ys_types().is_empty() {
            Ok(Vec::new())
        } else {
            Err(ScanError::MissingTracedInvocationContext.into())
        };
    };
    let primal_outputs = ryft_core::tracing_v2::Tracer::apply_staged_op(
        exemplar_primal_input.engine,
        exemplar_primal_input.builder.clone(),
        primal_inputs.as_slice(),
        XlaPrimitiveOperation::Scan(Box::new(scan.clone())),
    )?;
    let linear_scan = LinearizedScanJvpOperation::new(scan.clone(), primal_inputs);
    let tangent_outputs = LinearTerm::apply_staged_op(
        inputs[0].tangent.builder.clone(),
        tangent_inputs.as_slice(),
        LinearPrimitiveOperation::Custom(Arc::new(linear_scan.to_linear_custom_primitive())),
        scan.body().carry_types().len() + scan.body().ys_types().len(),
    )?;
    Ok(primal_outputs
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| Linearized { primal, tangent })
        .collect())
}

fn interpret_xla_scan_jvp<E>(
    scan: &ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
    inputs: &[JvpTracer<ShardMapTensor, EngineTangent<E>>],
) -> Result<Vec<JvpTracer<ShardMapTensor, EngineTangent<E>>>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = ShardMapTensor, LinearOperation = XlaLinearOperation>,
{
    let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
    let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
    let Some(tangent_builder) = tangent_inputs.first().map(|input| input.builder.clone()) else {
        return if scan.body().carry_types().is_empty() && scan.body().ys_types().is_empty() {
            Ok(Vec::new())
        } else {
            Err(ScanError::MissingTracedInvocationContext.into())
        };
    };
    let primal_outputs = scan.interpret(primal_inputs.as_slice())?;
    let linear_scan = TensorLinearizedScanJvpOperation::new(scan.clone(), primal_inputs);
    let tangent_outputs = LinearTerm::apply_staged_op(
        tangent_builder,
        tangent_inputs.as_slice(),
        LinearPrimitiveOperation::Custom(Arc::new(linear_scan.to_linear_custom_primitive())),
        scan.body().carry_types().len() + scan.body().ys_types().len(),
    )?;
    Ok(primal_outputs
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| JvpTracer { primal, tangent })
        .collect())
}

fn interpret_xla_condition_jvp<E>(
    condition: &ConditionOperation<ShardMapTensor, XlaPrimitiveOperation>,
    inputs: &[JvpTracer<ShardMapTensor, EngineTangent<E>>],
    engine: &E,
) -> Result<Vec<JvpTracer<ShardMapTensor, EngineTangent<E>>>, TracingError>
where
    E: DifferentiableEngine<Type = ArrayType, Value = ShardMapTensor, LinearOperation = XlaLinearOperation>,
{
    let ConditionPredicate::Captured(predicate) = condition.predicate() else {
        return Err(ControlFlowError::MissingTransformRule { transform: "runtime-predicate condition jvp" }.into());
    };
    let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
    let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
    let Some(tangent_builder) = tangent_inputs.first().map(|input| input.builder.clone()) else {
        return if condition.output_types().is_empty() {
            Ok(Vec::new())
        } else {
            Err(ScanError::MissingTracedInvocationContext.into())
        };
    };

    let selected_branch = if *predicate { condition.true_branch() } else { condition.false_branch() };
    let primal_outputs = selected_branch.interpret(primal_inputs.clone())?;
    let true_pushforward = linearize_program(engine, condition.true_branch(), primal_inputs.clone())?;
    let false_pushforward = linearize_program(engine, condition.false_branch(), primal_inputs)?;
    let linear_condition = ConditionOperation::with_captured_predicate(*predicate, true_pushforward, false_pushforward)
        .map_err(TracingError::from)?;
    let tangent_outputs = LinearTerm::apply_staged_op(
        tangent_builder,
        tangent_inputs.as_slice(),
        LinearPrimitiveOperation::Condition(Box::new(linear_condition)),
        condition.output_types().len(),
    )?;
    Ok(primal_outputs
        .into_iter()
        .zip(tangent_outputs)
        .map(|(primal, tangent)| JvpTracer { primal, tangent })
        .collect())
}

fn tracing_error_to_lowering_error(error: TracingError) -> LoweringError {
    LoweringError::UnsupportedOp { op: error.to_string() }
}

fn finish_traced_xla_program(
    builder: std::rc::Rc<std::cell::RefCell<ProgramBuilder<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>>,
    output_atoms: Vec<ryft_core::tracing::AtomId>,
    input_count: usize,
    output_count: usize,
) -> Result<
    Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, Vec<ShardMapTensor>, Vec<ShardMapTensor>>,
    TracingError,
> {
    if let Some(error) = builder.borrow_mut().error.take() {
        return Err(error);
    }
    let builder = match std::rc::Rc::try_unwrap(builder) {
        Ok(builder) => builder.into_inner(),
        Err(_) => return Err(TracingError::EscapedProgramBuilder),
    };
    builder
        .into_typed::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(vec![Placeholder; input_count])
        .build(output_atoms, vec![Placeholder; output_count])
}

fn trace_linearized_scan_jvp_body(
    scan: &ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>,
) -> Result<
    Program<ArrayType, ShardMapTensor, XlaPrimitiveOperation, Vec<ShardMapTensor>, Vec<ShardMapTensor>>,
    TracingError,
> {
    let carry_count = scan.body().carry_types().len();
    let x_count = scan.body().x_types().len();
    let y_count = scan.body().y_types().len();
    let input_types = scan
        .body()
        .carry_types()
        .iter()
        .chain(scan.body().carry_types())
        .chain(scan.body().x_types())
        .chain(scan.body().x_types())
        .cloned()
        .collect::<Vec<_>>();
    let builder =
        std::rc::Rc::new(std::cell::RefCell::new(
            ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new(Vec::new()),
        ));
    let inputs = input_types
        .iter()
        .map(|input_type| {
            let atom = builder.borrow_mut().add_input(input_type.clone());
            Tracer::from_staged_parts(atom, input_type.clone(), builder.clone(), XlaEngine::token())
        })
        .collect::<Vec<_>>();
    let primal_carry = inputs[..carry_count].to_vec();
    let tangent_carry = inputs[carry_count..carry_count * 2].to_vec();
    let x_start = carry_count * 2;
    let tangent_x_start = x_start + x_count;
    let primal_x = inputs[x_start..tangent_x_start].to_vec();
    let tangent_x = inputs[tangent_x_start..].to_vec();
    let mut body_primals = Vec::with_capacity(carry_count + x_count);
    body_primals.extend(primal_carry);
    body_primals.extend(primal_x);
    let mut body_tangents = Vec::with_capacity(carry_count + x_count);
    body_tangents.extend(tangent_carry);
    body_tangents.extend(tangent_x);
    let (body_primals, pushforward) =
        linearize_traced_program(XlaEngine::token(), builder.clone(), scan.body().program(), body_primals)?;
    let body_tangents = pushforward.interpret(body_tangents)?;
    let mut outputs = Vec::with_capacity(carry_count * 2 + y_count);
    outputs.extend(body_primals[..carry_count].iter().cloned());
    outputs.extend(body_tangents[..carry_count].iter().cloned());
    outputs.extend(body_tangents[carry_count..carry_count + y_count].iter().cloned());
    let output_atoms = outputs.iter().map(|output| output.atom_id()).collect::<Result<Vec<_>, _>>()?;
    drop(outputs);
    drop(body_tangents);
    drop(body_primals);
    drop(pushforward);
    drop(inputs);
    finish_traced_xla_program(builder, output_atoms, input_types.len(), carry_count * 2 + y_count)
}

impl StableHloCustomLowering<ShardMapTensor> for TensorLinearizedScanJvpOperation {
    fn lower_to_mlir<'b, 'c: 'b, 't: 'c>(
        &self,
        _op: &CustomPrimitive<ArrayType, ShardMapTensor>,
        input_values: &[ryft_mlir::ValueRef<'b, 'c, 't>],
        _output_types: &[ArrayType],
        lowerer: &mut ShardMapMlirLowerer<'b, 'c, 't>,
    ) -> Result<Vec<ryft_mlir::ValueRef<'b, 'c, 't>>, LoweringError> {
        let carry_count = self.scan.body().carry_types().len();
        let body = trace_linearized_scan_jvp_body(&self.scan).map_err(tracing_error_to_lowering_error)?;
        let combined_carry_types = self
            .scan
            .body()
            .carry_types()
            .iter()
            .chain(self.scan.body().carry_types())
            .cloned()
            .collect::<Vec<_>>();
        let combined_x_types =
            self.scan.body().x_types().iter().chain(self.scan.body().x_types()).cloned().collect::<Vec<_>>();
        let combined_xs_types =
            self.scan.body().xs_types().iter().chain(self.scan.body().xs_types()).cloned().collect::<Vec<_>>();
        let combined_scan = ScanOperation::new(
            ryft_core::tracing_v2::operations::FlatTracedScan::from_parts(
                combined_carry_types,
                combined_x_types,
                self.scan.body().y_types().to_vec(),
                combined_xs_types,
                self.scan.body().ys_types().to_vec(),
                self.scan.body().length(),
                body,
            ),
            self.scan_options(),
        );
        let mut combined_inputs = Vec::with_capacity(self.primal_inputs.len() + input_values.len());
        for primal_input in &self.primal_inputs[..carry_count] {
            combined_inputs.push(lowerer.lower_literal_value(primal_input)?);
        }
        combined_inputs.extend(input_values[..carry_count].iter().copied());
        for primal_input in &self.primal_inputs[carry_count..] {
            combined_inputs.push(lowerer.lower_literal_value(primal_input)?);
        }
        combined_inputs.extend(input_values[carry_count..].iter().copied());
        let combined_outputs = lower_scan_to_while(
            &combined_scan,
            combined_inputs.as_slice(),
            &mut lowerer.block,
            lowerer.context,
            lowerer.location,
        )?;
        let mut outputs = combined_outputs[carry_count..carry_count * 2].to_vec();
        outputs.extend(combined_outputs[carry_count * 2..].iter().copied());
        Ok(outputs)
    }
}

/// Closed ordinary staged-op universe owned by the XLA backend.
#[allow(private_interfaces)]
#[derive(Clone)]
pub enum XlaPrimitiveOperation {
    /// Elementwise addition.
    Add,

    /// Elementwise multiplication.
    Mul,

    /// Elementwise negation.
    Neg,

    /// Elementwise sine.
    Sin,

    /// Elementwise cosine.
    Cos,

    /// Matrix multiplication.
    MatMul,

    /// Matrix transpose.
    MatrixTranspose,

    /// Scaling by one captured factor.
    Scale { factor: ShardMapTensor },

    /// Left matrix multiplication by one captured factor.
    LeftMatMul { factor: ShardMapTensor },

    /// Right matrix multiplication by one captured factor.
    RightMatMul { factor: ShardMapTensor },

    /// Reshape.
    Reshape { input_type: ArrayType, output_type: ArrayType },

    /// Static leading-axis slice.
    SliceLeadingAxis(SliceLeadingAxisOperation),

    /// Static leading-axis scatter into an otherwise-zero tensor.
    ScatterLeadingAxisSlice(ScatterLeadingAxisSliceOperation),

    /// Static leading-axis stack.
    StackLeadingAxis(StackLeadingAxisOperation),

    /// Higher-order rematerialization.
    Rematerialize(Box<RematerializeOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation, XlaLinearOperation>>),

    /// Higher-order static scan loop.
    Scan(Box<ScanOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation>>),

    /// Higher-order conditional.
    Condition(Box<ConditionOperation<ShardMapTensor, XlaPrimitiveOperation>>),

    /// Higher-order while loop.
    While(Box<WhileOperation<ShardMapTensor, XlaPrimitiveOperation>>),

    /// XLA-specific `shard_map`.
    ShardMap(Box<ShardMapOperation<ShardMapTensor>>),

    /// XLA-specific `linear_shard_map`.
    LinearShardMap(Box<LinearShardMapOperation<ShardMapTensor>>),

    /// XLA-specific sharding constraint.
    WithShardingConstraint(WithShardingConstraintOperation),

    /// Explicit escape hatch for custom XLA ops.
    Custom(Arc<CustomPrimitive<ArrayType, ShardMapTensor>>),
}

impl Debug for XlaPrimitiveOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(formatter, "Add"),
            Self::Mul => write!(formatter, "Mul"),
            Self::Neg => write!(formatter, "Neg"),
            Self::Sin => write!(formatter, "Sin"),
            Self::Cos => write!(formatter, "Cos"),
            Self::MatMul => write!(formatter, "MatMul"),
            Self::MatrixTranspose => write!(formatter, "MatrixTranspose"),
            Self::Scale { .. } => write!(formatter, "Scale"),
            Self::LeftMatMul { .. } => write!(formatter, "LeftMatMul"),
            Self::RightMatMul { .. } => write!(formatter, "RightMatMul"),
            Self::Reshape { input_type, output_type } => write!(formatter, "Reshape({input_type} -> {output_type})"),
            Self::SliceLeadingAxis(op) => Debug::fmt(op, formatter),
            Self::ScatterLeadingAxisSlice(op) => Debug::fmt(op, formatter),
            Self::StackLeadingAxis(op) => Debug::fmt(op, formatter),
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
            Self::Scan(scan) => Debug::fmt(scan, formatter),
            Self::Condition(condition) => Debug::fmt(condition, formatter),
            Self::While(while_operation) => Debug::fmt(while_operation, formatter),
            Self::ShardMap(op) => Debug::fmt(op, formatter),
            Self::LinearShardMap(op) => Debug::fmt(op, formatter),
            Self::WithShardingConstraint(op) => Debug::fmt(op, formatter),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl Display for XlaPrimitiveOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl Operation<ArrayType> for XlaPrimitiveOperation {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Neg => "neg",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::MatMul => "matmul",
            Self::MatrixTranspose => "matrix_transpose",
            Self::Scale { .. } => "scale",
            Self::LeftMatMul { .. } => "left_matmul",
            Self::RightMatMul { .. } => "right_matmul",
            Self::Reshape { .. } => "reshape",
            Self::SliceLeadingAxis(op) => op.name(),
            Self::ScatterLeadingAxisSlice(op) => op.name(),
            Self::StackLeadingAxis(op) => op.name(),
            Self::Rematerialize(remat) => remat.name(),
            Self::Scan(scan) => scan.name(),
            Self::Condition(condition) => condition.name(),
            Self::While(while_operation) => while_operation.name(),
            Self::ShardMap(op) => op.name(),
            Self::LinearShardMap(op) => op.name(),
            Self::WithShardingConstraint(op) => op.name(),
            Self::Custom(op) => op.name(),
        }
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        match self {
            Self::Add => AddOperation.infer_output_types(input_types),
            Self::Mul => MulOperation.infer_output_types(input_types),
            Self::Neg => NegOperation.infer_output_types(input_types),
            Self::Sin => SinOperation.infer_output_types(input_types),
            Self::Cos => CosOperation.infer_output_types(input_types),
            Self::MatMul => MatMulOperation.infer_output_types(input_types),
            Self::MatrixTranspose => MatrixTransposeOperation.infer_output_types(input_types),
            Self::Scale { .. } => ScaleOperation::<ArrayType, ShardMapTensor>::abstract_eval_static(input_types),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::r#type(factor), input_types),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::r#type(factor), input_types),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).infer_output_types(input_types)
            }
            Self::SliceLeadingAxis(op) => op.infer_output_types(input_types),
            Self::ScatterLeadingAxisSlice(op) => op.infer_output_types(input_types),
            Self::StackLeadingAxis(op) => op.infer_output_types(input_types),
            Self::Rematerialize(remat) => remat.infer_output_types(input_types),
            Self::Scan(scan) => scan.infer_output_types(input_types),
            Self::Condition(condition) => condition.infer_output_types(input_types),
            Self::While(while_operation) => while_operation.infer_output_types(input_types),
            Self::ShardMap(op) => op.infer_output_types(input_types),
            Self::LinearShardMap(op) => op.infer_output_types(input_types),
            Self::WithShardingConstraint(op) => op.infer_output_types(input_types),
            Self::Custom(op) => op.infer_output_types(input_types),
        }
    }
}

impl InterpretableOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
        match self {
            Self::Add => AddOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatMul => MatMulOperation.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::SliceLeadingAxis(op) => op.interpret(inputs),
            Self::ScatterLeadingAxisSlice(op) => op.interpret(inputs),
            Self::StackLeadingAxis(op) => op.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Scan(scan) => scan.interpret(inputs),
            Self::Condition(condition) => condition.interpret(inputs),
            Self::While(while_operation) => while_operation.interpret(inputs),
            Self::ShardMap(op) => op.interpret(inputs),
            Self::LinearShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl InterpretableOperation<ArrayType, ShardMapTracer> for XlaPrimitiveOperation {
    fn interpret(&self, inputs: &[ShardMapTracer]) -> Result<Vec<ShardMapTracer>, TracingError> {
        match self {
            Self::Add => AddOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatMul => MatMulOperation.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                ScaleOperation::new(lift_jit_constant(factor, exemplar)).interpret(inputs)
            }
            Self::LeftMatMul { factor } => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                LeftMatMulOperation::new(lift_jit_constant(factor, exemplar)).interpret(inputs)
            }
            Self::RightMatMul { factor } => {
                let exemplar = inputs.first().ok_or(TracingError::InvalidInputCount { expected: 1, got: 0 })?;
                RightMatMulOperation::new(lift_jit_constant(factor, exemplar)).interpret(inputs)
            }
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::SliceLeadingAxis(op) => op.interpret(inputs),
            Self::ScatterLeadingAxisSlice(op) => op.interpret(inputs),
            Self::StackLeadingAxis(op) => op.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Scan(scan) => {
                let exemplar = inputs.first().ok_or(ScanError::MissingTracedInvocationContext)?;
                ryft_core::tracing_v2::Tracer::apply_staged_op(
                    exemplar.engine,
                    exemplar.builder.clone(),
                    inputs,
                    XlaPrimitiveOperation::Scan(scan.clone()),
                )
            }
            Self::Condition(condition) => {
                let exemplar = inputs.first().ok_or(ScanError::MissingTracedInvocationContext)?;
                ryft_core::tracing_v2::Tracer::apply_staged_op(
                    exemplar.engine,
                    exemplar.builder.clone(),
                    inputs,
                    XlaPrimitiveOperation::Condition(condition.clone()),
                )
            }
            Self::While(while_operation) => {
                let exemplar = inputs.first().ok_or(ScanError::MissingTracedInvocationContext)?;
                ryft_core::tracing_v2::Tracer::apply_staged_op(
                    exemplar.engine,
                    exemplar.builder.clone(),
                    inputs,
                    XlaPrimitiveOperation::While(while_operation.clone()),
                )
            }
            Self::ShardMap(op) => {
                let exemplar = inputs.first().ok_or(ScanError::MissingTracedInvocationContext)?;
                op.interpret_traced_with_context(exemplar.builder.clone(), inputs)
            }
            Self::LinearShardMap(op) => {
                let exemplar = inputs.first().ok_or(ScanError::MissingTracedInvocationContext)?;
                op.interpret_traced_with_context(exemplar.builder.clone(), inputs)
            }
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => {
                let exemplar = inputs.first().ok_or(ScanError::MissingTracedInvocationContext)?;
                let replay_context = ShardMapReplayContext::new(exemplar.builder.clone());
                op.extensions()
                    .get::<ShardMapCustomReplayExtension<ShardMapTracer>>()
                    .ok_or_else(|| {
                        TracingError::CustomOperation(ryft_core::tracing_v2::CustomOperationError::MissingRule {
                            op: op.name(),
                            transform: "traced replay",
                        })
                    })?
                    .replay(&replay_context, inputs.to_vec())
                    .map_err(|error| TracingError::Type(TypeError { message: error.to_string() }))
            }
        }
    }
}

impl<E> DifferentiableOperation<E> for XlaPrimitiveOperation
where
    E: DifferentiableEngine<Type = ArrayType, Value = ShardMapTensor, LinearOperation = XlaLinearOperation>,
    ShardMapTensor: Differentiable<
            ArrayType,
            Tangent<XlaLinearOperation> = LinearTerm<ArrayType, ShardMapTensor, XlaLinearOperation>,
        >,
{
    fn jvp(
        &self,
        engine: &E,
        inputs: &[JvpTracer<ShardMapTensor, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<ShardMapTensor, EngineTangent<E>>>, TracingError> {
        match self {
            Self::Add => AddOperation.jvp(engine, inputs),
            Self::Mul => MulOperation.jvp(engine, inputs),
            Self::Neg => NegOperation.jvp(engine, inputs),
            Self::Sin => SinOperation.jvp(engine, inputs),
            Self::Cos => CosOperation.jvp(engine, inputs),
            Self::MatMul => MatMulOperation.jvp(engine, inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.jvp(engine, inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).jvp(engine, inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).jvp(engine, inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).jvp(engine, inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).jvp(engine, inputs)
            }
            Self::SliceLeadingAxis(op) => op.jvp(engine, inputs),
            Self::ScatterLeadingAxisSlice(_) => {
                Err(ScanError::MissingTransformRule { transform: "scatter jvp" }.into())
            }
            Self::StackLeadingAxis(op) => op.jvp(engine, inputs),
            Self::Rematerialize(remat) => {
                let primal_inputs = inputs.iter().map(|input| input.primal.clone()).collect::<Vec<_>>();
                let tangent_inputs = inputs.iter().map(|input| input.tangent.clone()).collect::<Vec<_>>();
                let primal_outputs = remat.interpret(primal_inputs.as_slice())?;
                let tangent_builder = if let Some(first_tangent) = tangent_inputs.first() {
                    first_tangent.builder.clone()
                } else if remat.body().output_types().is_empty() {
                    return Ok(Vec::new());
                } else {
                    return Err(DifferentiationError::MissingLinearRematerializeReplayTangentLeaves.into());
                };
                let tangent_outputs = LinearTerm::apply_staged_op(
                    tangent_builder,
                    tangent_inputs.as_slice(),
                    LinearPrimitiveOperation::Rematerialize(Box::new(make_linear_xla_rematerialize(
                        engine,
                        remat.body(),
                        primal_inputs,
                    )?)),
                    remat.body().output_types().len(),
                )?;
                Ok(primal_outputs
                    .into_iter()
                    .zip(tangent_outputs)
                    .map(|(primal, tangent)| JvpTracer { primal, tangent })
                    .collect::<Vec<_>>())
            }
            Self::Scan(scan) => interpret_xla_scan_jvp::<E>(scan, inputs),
            Self::Condition(condition) => interpret_xla_condition_jvp::<E>(condition, inputs, engine),
            Self::While(_) => Err(ControlFlowError::MissingTransformRule { transform: "while jvp" }.into()),
            Self::ShardMap(op) => op.jvp(engine, inputs),
            Self::LinearShardMap(op) => op.jvp(engine, inputs),
            Self::WithShardingConstraint(op) => op.jvp(engine, inputs),
            Self::Custom(op) => {
                Err(ryft_core::tracing_v2::CustomOperationError::MissingRule { op: op.name(), transform: "jvp" }.into())
            }
        }
    }
}

impl InterpretableOperation<ArrayType, Linearized<ShardMapTracer>> for XlaPrimitiveOperation {
    fn interpret(
        &self,
        inputs: &[Linearized<ShardMapTracer>],
    ) -> Result<Vec<Linearized<ShardMapTracer>>, TracingError> {
        match self {
            Self::Add => AddOperation.interpret(inputs),
            Self::Mul => MulOperation.interpret(inputs),
            Self::Neg => NegOperation.interpret(inputs),
            Self::Sin => SinOperation.interpret(inputs),
            Self::Cos => CosOperation.interpret(inputs),
            Self::MatMul => MatMulOperation.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOperation.interpret(inputs),
            Self::Scale { factor } => ScaleOperation::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOperation::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOperation::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::SliceLeadingAxis(op) => op.interpret(inputs),
            Self::ScatterLeadingAxisSlice(op) => op.interpret(inputs),
            Self::StackLeadingAxis(op) => op.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Scan(scan) => interpret_xla_scan_linearized_jit(scan, inputs),
            Self::Condition(_) | Self::While(_) => {
                Err(ControlFlowError::MissingTransformRule { transform: "linearized JIT replay" }.into())
            }
            Self::ShardMap(op) => op.interpret(inputs),
            Self::LinearShardMap(op) => op.interpret(inputs),
            Self::WithShardingConstraint(op) => op.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl AddTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn add_op() -> Self {
        XlaPrimitiveOperation::Add
    }
}

impl MulTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn mul_op() -> Self {
        XlaPrimitiveOperation::Mul
    }
}

impl NegTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn neg_op() -> Self {
        XlaPrimitiveOperation::Neg
    }
}

impl SinTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn sin_op() -> Self {
        XlaPrimitiveOperation::Sin
    }
}

impl CosTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn cos_op() -> Self {
        XlaPrimitiveOperation::Cos
    }
}

impl MatMulTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn matmul_op() -> Self {
        XlaPrimitiveOperation::MatMul
    }
}

impl MatrixTransposeTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn matrix_transpose_op() -> Self {
        XlaPrimitiveOperation::MatrixTranspose
    }
}

impl CustomTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn custom_op(primitive: Arc<CustomPrimitive<ArrayType, ShardMapTensor>>) -> Self {
        XlaPrimitiveOperation::Custom(primitive)
    }
}

impl RematerializeTracingOperation<ArrayType, ShardMapTensor, XlaLinearOperation> for XlaPrimitiveOperation {
    fn rematerialize_op(
        op: RematerializeOperation<ArrayType, ShardMapTensor, XlaPrimitiveOperation, XlaLinearOperation>,
    ) -> Self {
        XlaPrimitiveOperation::Rematerialize(Box::new(op))
    }
}

impl ScanTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn scan_op(op: ScanOperation<ArrayType, ShardMapTensor, Self>) -> Self {
        XlaPrimitiveOperation::Scan(Box::new(op))
    }
}

impl LeadingAxisTracingOperation<ShardMapTensor> for XlaPrimitiveOperation {
    fn slice_leading_axis_op(op: SliceLeadingAxisOperation) -> Self {
        XlaPrimitiveOperation::SliceLeadingAxis(op)
    }

    fn scatter_leading_axis_slice_op(op: ScatterLeadingAxisSliceOperation) -> Self {
        XlaPrimitiveOperation::ScatterLeadingAxisSlice(op)
    }

    fn stack_leading_axis_op(op: StackLeadingAxisOperation) -> Self {
        XlaPrimitiveOperation::StackLeadingAxis(op)
    }
}

impl ScaleTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn scale_op(factor: ShardMapTensor) -> Self {
        XlaPrimitiveOperation::Scale { factor }
    }
}

impl LeftMatMulTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn left_matmul_op(factor: ShardMapTensor) -> Self {
        XlaPrimitiveOperation::LeftMatMul { factor }
    }
}

impl RightMatMulTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn right_matmul_op(factor: ShardMapTensor) -> Self {
        XlaPrimitiveOperation::RightMatMul { factor }
    }
}

impl ReshapeTracingOperation<ArrayType, ShardMapTensor> for XlaPrimitiveOperation {
    fn reshape_op(input_type: ArrayType, output_type: ArrayType) -> Self {
        XlaPrimitiveOperation::Reshape { input_type, output_type }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, fmt::Display, rc::Rc, sync::Arc};

    use pretty_assertions::assert_eq;

    use ryft_core::parameters::Placeholder;
    use ryft_core::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use ryft_core::tracing::ProgramBuilder;
    use ryft_core::tracing_v2::operations::CustomOperationError;
    use ryft_core::types::{DataType, Shape, Size, Typed};

    use super::*;

    #[derive(Clone, Debug)]
    struct TestCustomXlaOp;

    impl Display for TestCustomXlaOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "test_custom_xla")
        }
    }

    impl Operation<ArrayType> for TestCustomXlaOp {
        fn name(&self) -> &'static str {
            "test_custom_xla"
        }

        fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
            Ok(input_types.to_vec())
        }
    }

    impl InterpretableOperation<ArrayType, ShardMapTensor> for TestCustomXlaOp {
        fn interpret(&self, inputs: &[ShardMapTensor]) -> Result<Vec<ShardMapTensor>, TracingError> {
            Ok(inputs.to_vec())
        }
    }

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn vector_type(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(length)]), None, None).unwrap()
    }

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    fn scan_program() -> ryft_core::tracing::Program<
        ArrayType,
        ShardMapTensor,
        XlaPrimitiveOperation,
        Vec<ShardMapTensor>,
        Vec<ShardMapTensor>,
    > {
        let scalar_type = scalar_type();
        let xs_type = vector_type(3);
        let mut body_builder = ProgramBuilder::<
            ArrayType,
            ShardMapTensor,
            XlaPrimitiveOperation,
            Vec<ShardMapTensor>,
            Vec<ShardMapTensor>,
        >::new(vec![Placeholder, Placeholder]);
        let carry = body_builder.add_input(scalar_type.clone());
        let x = body_builder.add_input(scalar_type.clone());
        let next_carry = body_builder
            .add_instruction(XlaPrimitiveOperation::Add, vec![carry, x])
            .expect("scan body add should stage")
            .into_iter()
            .next()
            .expect("add should produce one output");
        let body_program = body_builder.build(vec![next_carry, next_carry], vec![Placeholder, Placeholder]).unwrap();
        let body = ryft_core::tracing_v2::operations::FlatTracedScan::from_parts(
            vec![scalar_type.clone()],
            vec![scalar_type.clone()],
            vec![scalar_type.clone()],
            vec![xs_type.clone()],
            vec![xs_type],
            3,
            body_program,
        );
        let scan = ScanOperation::new(body, ScanOptions::default());
        let mut builder = ProgramBuilder::<
            ArrayType,
            ShardMapTensor,
            XlaPrimitiveOperation,
            Vec<ShardMapTensor>,
            Vec<ShardMapTensor>,
        >::new(vec![Placeholder, Placeholder]);
        let carry = builder.add_input(scalar_type);
        let xs = builder.add_input(vector_type(3));
        let outputs = builder.add_instruction(XlaPrimitiveOperation::Scan(Box::new(scan)), vec![carry, xs]).unwrap();
        builder.build(outputs, vec![Placeholder, Placeholder]).unwrap()
    }

    fn unary_rematerialize_body() -> FlatTracedRematerialize<ArrayType, ShardMapTensor, XlaPrimitiveOperation> {
        let mut builder = ProgramBuilder::<
            ArrayType,
            ShardMapTensor,
            XlaPrimitiveOperation,
            Vec<ShardMapTensor>,
            Vec<ShardMapTensor>,
        >::new(vec![Placeholder]);
        let input = builder.add_input(scalar_type());
        let output = builder
            .add_instruction(XlaPrimitiveOperation::Sin, vec![input])
            .expect("rematerialize body should stage one sine op")
            .into_iter()
            .next()
            .expect("sine should produce one output");
        let program = builder.build(vec![output], vec![Placeholder]).unwrap();
        FlatTracedRematerialize::from_parts(vec![scalar_type()], vec![scalar_type()], program)
    }

    #[test]
    fn test_custom_xla_op_missing_linearized_jit_rule_reports_missing_rule() {
        let operation = XlaPrimitiveOperation::Custom(Arc::new(CustomPrimitive::new(TestCustomXlaOp)));
        let inputs: Vec<Linearized<ShardMapTracer>> = vec![];

        assert!(matches!(
            operation.interpret(&inputs),
            Err(TracingError::CustomOperation(CustomOperationError::MissingRule {
                op: "test_custom_xla",
                transform: "linearized JIT replay",
            }))
        ));
    }

    #[test]
    fn test_xla_rematerialize_jvp_stages_a_linear_rematerialize() {
        let operation =
            XlaPrimitiveOperation::Rematerialize(Box::new(RematerializeOperation::new(unary_rematerialize_body())));
        let tangent_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaLinearOperation>::new(Vec::new())));
        let tangent_atom = tangent_builder.borrow_mut().add_input(scalar_type());
        let outputs = operation
            .jvp(
                crate::experimental::engines::XlaEngine::token(),
                &[JvpTracer {
                    primal: ShardMapTensor::new(scalar_type()),
                    tangent: LinearTerm::from_staged_parts(tangent_atom, tangent_builder.clone()),
                }],
            )
            .expect("xla rematerialize jvp should succeed");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal.r#type().into_owned(), scalar_type());

        let output_atoms = outputs.into_iter().map(|output| output.tangent.atom).collect::<Vec<_>>();
        let tangent_builder = Rc::try_unwrap(tangent_builder)
            .expect("rematerialize jvp builder should not have outstanding linear terms")
            .into_inner();
        let tangent_program = tangent_builder
            .into_typed::<Vec<ShardMapTensor>, Vec<ShardMapTensor>>(vec![Placeholder])
            .build(output_atoms, vec![Placeholder])
            .unwrap();
        assert!(
            tangent_program.to_string().contains("rematerialize"),
            "expected linearized xla rematerialize jvp to stage a linear rematerialize op: {}",
            tangent_program
        );
    }

    #[test]
    fn test_xla_scan_jvp_program_uses_compact_linear_scan() {
        let program = scan_program();
        let pushforward = linearize_program(
            XlaEngine::token(),
            &program,
            vec![ShardMapTensor::new(scalar_type()), ShardMapTensor::new(vector_type(3))],
        )
        .unwrap();
        let rendered = pushforward.to_string();

        assert!(rendered.contains("linear_scan_jvp"), "{rendered}");
        assert!(!rendered.contains("slice"), "{rendered}");
        assert!(!rendered.contains("stack"), "{rendered}");

        let pullback = transpose_linear_program(XlaEngine::token(), &pushforward).unwrap();
        let rendered_pullback = pullback.to_string();
        assert!(rendered_pullback.contains("linear_scan_transpose"), "{rendered_pullback}");
        assert!(!rendered_pullback.contains("slice"), "{rendered_pullback}");
        assert!(!rendered_pullback.contains("stack"), "{rendered_pullback}");

        let pushforward = linearize_program(
            XlaEngine::token(),
            &program,
            vec![ShardMapTensor::zero(scalar_type()), ShardMapTensor::zero(vector_type(3))],
        )
        .unwrap();
        let stablehlo = crate::experimental::lowering::to_mlir_module_for_plain_program(&pushforward, "main").unwrap();
        assert!(stablehlo.contains("stablehlo.while"), "{stablehlo}");
        assert_eq!(stablehlo.matches("stablehlo.while").count(), 1, "{stablehlo}");
    }

    #[test]
    fn test_replay_xla_program_with_tracers_uses_custom_replay_extension() {
        let sharding = Sharding::replicated(test_mesh(), 0);
        let custom = WithShardingConstraintOperation::new(sharding).to_tensor_custom_primitive();
        let mut program_builder = ProgramBuilder::<
            ArrayType,
            ShardMapTensor,
            XlaPrimitiveOperation,
            Vec<ShardMapTensor>,
            Vec<ShardMapTensor>,
        >::new(vec![Placeholder]);
        let input = program_builder.add_input(scalar_type());
        let output = program_builder
            .add_instruction(XlaPrimitiveOperation::Custom(Arc::new(custom)), vec![input])
            .expect("custom op should stage")
            .into_iter()
            .next()
            .expect("custom op should produce one output");
        let program = program_builder.build(vec![output], vec![Placeholder]).unwrap();
        let tracing_builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ShardMapTensor, XlaPrimitiveOperation>::new(Vec::new())));
        let traced_input_atom = tracing_builder.borrow_mut().add_input(scalar_type());
        let traced_input =
            Tracer::from_staged_parts(traced_input_atom, scalar_type(), tracing_builder, XlaEngine::token());

        let outputs = replay_xla_program_with_tracers(&program, vec![traced_input]).unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), scalar_type());
    }
}
