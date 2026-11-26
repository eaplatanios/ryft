//! General XLA program tracing. The [`trace`] entry point stages an arbitrary function over [`ArrayTracer`]s in a fresh
//! root tracing context of [`Session`]. Its result is a staged [`Program`] of `ryft-core` (refer to [`TracedProgram`]),
//! which [`ProgramExtension`](crate::ProgramExtension) lowers to a StableHLO/Shardy MLIR module. [`Error`] reports
//! the failures of both tracing and lowering.
//!
//! Traced functions may compose `shard_map` with other transforms (e.g., differentiation or batching). The logical
//! semantics of `shard_map` (i.e., boundary metadata, validation, closure tracing, and transform rules) are owned by
//! the [`shard_map`](mod@ryft_core::shard_map) module of `ryft-core`, and its manual computations lower to Shardy
//! `sdy.manual_computation` operations in the [`lowering`](crate::lowering) module. A `shard_map` traced on its own is
//! simply a traced function whose body binds one `shard_map` (e.g., through [`ShardMap`](ryft_core::ShardMap) on the
//! context that [`trace_with_context`] passes to it). The lowered function signature carries the sharding of every
//! global input and output type that has one (refer to
//! [`ProgramExtension::to_mlir_module`](crate::ProgramExtension::to_mlir_module)). Tracing such a function over input
//! types that carry the input shardings of its `shard_map` therefore yields function arguments and results placed
//! as the `shard_map` boundary specifies.
//!
//! Core failures use [`Error::Core`], and concrete lowering failures have their own [`Error`] variants. Logical
//! `shard_map` errors raised while tracing or lowering remain [`ProgramError::Custom`] payloads, recoverable through
//! [`ProgramError::downcast_custom`].

use std::cell::RefCell;

use ryft_core::{
    ArrayType, DomainTracingContext, Parameterized, ParameterizedFamily, Program, ProgramError, ProjectedValue,
    Tracer as CoreTracer, TracingContext, ValueProjection,
};

use crate::{Constant, Error, Operation, Session, Tracer, Type};

/// Array value that closures traced through [`trace`] receive and return. It is a
/// [`DomainTracer`](ryft_core::DomainTracer) of [`Session`] projected onto its array member. The `shard_map` bodies
/// traced in XLA programs (e.g., through [`ryft_core::shard_map()`] on values of type [`ArrayTracer`]) receive the
/// same values. This alias is public so that callers (e.g., tooling binaries) can annotate traced closure parameters.
pub type ArrayTracer = ProjectedValue<ArrayType, Tracer>;

/// [`ArrayTracer`] spelled without the type aliases over [`Session`] that it goes through, which contain projections.
/// The compiler does not match a bound whose trait argument is spelled through them, so the bounds of [`trace`] and
/// [`trace_with_context`] on the structure of the traced outputs name this type instead.
type NormalizedArrayTracer = ProjectedValue<ArrayType, CoreTracer<TracingContext<Constant, Operation>>>;

/// Traced XLA program over global array types structured as `Input` and `Output`, as returned by [`trace`]. It is
/// the staged [`Program`] of `ryft-core` itself, whose entry region carries the global input and output types, which
/// describe the whole logical arrays at the function boundary. For example, a global array of shape `[8]` split evenly
/// across two devices has shape `[4]` on each device. [`ProgramExtension`](crate::ProgramExtension) exports the program
/// as a StableHLO/Shardy MLIR module.
pub type TracedProgram<Input, Output> = Program<
    Constant,
    Operation,
    <Input as Parameterized<ArrayType>>::To<Constant>,
    <Output as Parameterized<ArrayType>>::To<Constant>,
>;

/// Stages an arbitrary traced XLA function over global array types. This is the general XLA tracing entry point used
/// when callers want to compose `shard_map` with differentiation transforms and then lower the resulting whole program
/// to StableHLO/Shardy MLIR. `function` is traced once, in a fresh root tracing context of [`Session`] that binds no
/// named axes, over [`ArrayTracer`]s with the global types in `input_type`, and the traced program is simplified.
/// Functions that need that context (e.g., to create values that derive from no input) or that can fail are traced
/// through [`trace_with_context`] instead.
///
/// # Parameters
///
///   - `function`: Function to trace over global XLA values.
///   - `input_type`: Global input types for the traced function.
///
/// # Errors
///
/// Returns the errors of [`trace_with_context`].
#[inline]
pub fn trace<F, Input, Output>(
    function: F,
    input_type: Input,
) -> Result<
    Program<Constant, Operation, Input::To<Constant>, <Output as Parameterized<NormalizedArrayTracer>>::To<Constant>>,
    Error,
>
where
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer>,
    Output: Parameterized<NormalizedArrayTracer>,
    <Output as Parameterized<NormalizedArrayTracer>>::Family: ParameterizedFamily<Constant>,
    F: FnOnce(Input::To<ArrayTracer>) -> Output,
{
    trace_with_context(|_, inputs| Ok(function(inputs)), input_type)
}

/// Stages a fallible XLA function over global array types like [`trace`], passing it the fresh root tracing context of
/// [`Session`] together with its inputs. The context lets the function create values that derive from no input (e.g.,
/// through [`Context::lift`](ryft_core::Context::lift)) and bind context capabilities directly (e.g., a `shard_map`
/// without inputs through [`ShardMap`](ryft_core::ShardMap)), and the function must not retain it or its tracers
/// after returning.
///
/// # Parameters
///
///   - `function`: Function to trace over the tracing context and global XLA values.
///   - `input_type`: Global input types for the traced function. Use `()` for no inputs.
///
/// # Errors
///
/// Returns [`Error::Core`] wrapping a [`ProgramError`] when `function` fails or when tracing or simplifying the program
/// fails (e.g., because `function` stages an invalid operation or registers a capture, which the fresh root context
/// discards), or a [`ParameterError`](ryft_core::ParameterError) when the traced inputs or outputs cannot be
/// restructured into `Input` or `Output`.
pub fn trace_with_context<F, Input, Output>(
    function: F,
    input_type: Input,
) -> Result<
    Program<Constant, Operation, Input::To<Constant>, <Output as Parameterized<NormalizedArrayTracer>>::To<Constant>>,
    Error,
>
where
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer>,
    Output: Parameterized<NormalizedArrayTracer>,
    <Output as Parameterized<NormalizedArrayTracer>>::Family: ParameterizedFamily<Constant>,
    F: FnOnce(&DomainTracingContext<Session>, Input::To<ArrayTracer>) -> Result<Output, ProgramError>,
{
    let input_structure = input_type.parameter_structure();
    let output_structure = RefCell::new(None);
    let (_, program) = DomainTracingContext::<Session>::trace_with_context_and_named_axes(
        |context, inputs: Vec<Tracer>| {
            let inputs = inputs
                .into_iter()
                .map(|input| ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs =
                function(context, Input::To::<ArrayTracer>::from_parameters(input_structure.clone(), inputs)?)?;
            output_structure.replace(Some(outputs.parameter_structure()));
            Ok(outputs.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>())
        },
        input_type.parameters().cloned().map(Type::Array).collect::<Vec<_>>(),
        Vec::new(),
    )?;
    let output_structure = output_structure.into_inner().ok_or_else(|| {
        ProgramError::MalformedProgram("XLA tracing completed without recording its output structure".to_string())
    })?;
    Ok(program
        .into_simplified()?
        .restructured::<Input::To<Constant>, <Output as Parameterized<NormalizedArrayTracer>>::To<Constant>>(
            input_structure,
            output_structure,
        )?)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::{
        DataType, Error as CoreError, LogicalMesh, MeshAxis, MeshAxisType, ShardMapOptions, Sharding,
        ShardingDimension, Sin, shard_map,
    };

    use super::*;

    #[test]
    fn test_trace() {
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(x, y): (ArrayTracer, ArrayTracer)| x.sin().unwrap() * y,
            (ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = sin %0
                    %3:f32[4] = mul %2 %1
                in (%3)
            "}
            .trim_end(),
        );

        // A `shard_map` is staged as one instruction whose body is a region of the simplified program.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            move |x: ArrayTracer| {
                shard_map(
                    |local_x: ArrayTracer| local_x.sin(),
                    x,
                    mesh,
                    sharding.clone(),
                    sharding,
                    ShardMapOptions::new(),
                )
                .unwrap()
            },
            ArrayType::new_static(DataType::F32, [8]),
        )
        .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[8] .
                let %1:f32[8][sharding={mesh<['x'=4:manual]>, [{'x'}]}] = shard_map [
                    mesh=['x'=4:manual],
                    input_shardings=[{mesh<['x'=4:manual]>, [{'x'}]}],
                    output_shardings=[{mesh<['x'=4:manual]>, [{'x'}]}],
                    manual_axes=['x'],
                    global_input_types=[f32[8][sharding={mesh<['x'=4:manual]>, [{'x'}]}]],
                    global_output_types=[f32[8][sharding={mesh<['x'=4:manual]>, [{'x'}]}]],
                ] %0 [
                    body={
                        lambda %0:f32[2][sharding={mesh<['x'=4:manual]>, [{}], varying_manual={'x'}}] .
                        let %1:f32[2][sharding={mesh<['x'=4:manual]>, [{}], varying_manual={'x'}}] = sin %0
                        in (%1)
                    },
                ]
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_trace_with_context() {
        // The traced function receives the tracing context together with its inputs and may fail.
        let traced =
            trace_with_context::<_, ArrayType, ArrayTracer>(|_, x| x.sin(), ArrayType::new_static(DataType::F32, [4]))
                .unwrap();
        assert_eq!(
            traced.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4] = sin %0
                in (%1)
            "}
            .trim_end(),
        );

        // Failures of the traced function propagate unchanged as core program errors.
        assert!(matches!(
            trace_with_context::<_, ArrayType, ArrayTracer>(
                |_, _| Err(ProgramError::MalformedProgram("traced function failure".to_string())),
                ArrayType::new_static(DataType::F32, [4]),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if message == "traced function failure",
            ),
        ));
    }
}
