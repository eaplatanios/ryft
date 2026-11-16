//! General XLA program tracing. The [`trace`] entry point stages an arbitrary function over [`ArrayTracer`]s in a fresh
//! root tracing context of [`Session`], and its [`TracedProgram`] result reports the statistics of the
//! staged program and lowers it to a StableHLO/Shardy MLIR module through the [`lowering`](crate::lowering) module.
//! [`Error`] reports the failures of both tracing and lowering.
//!
//! Traced functions may compose `shard_map` with other transforms (e.g., differentiation or batching). The logical
//! semantics of `shard_map` (i.e., boundary metadata, validation, closure tracing, and transform rules) are owned by
//! the [`shard_map`](mod@ryft_core::operations::sharding::shard_map) module of `ryft-core`, and its manual computations
//! lower to Shardy `sdy.manual_computation` operations in the [`lowering`](crate::lowering) module. A `shard_map`
//! traced on its own is simply a traced function whose body binds one `shard_map` (e.g., through
//! [`ShardMap`](ryft_core::ShardMap) on the context that [`trace_with_context`] passes to it). The lowered function
//! signature carries the sharding of every global input and output type that has one (refer to
//! [`TracedProgram::to_mlir_module`]). Tracing such a function over input types that carry the input shardings of its
//! `shard_map` therefore yields function arguments and results that are placed as the `shard_map` boundary specifies.
//!
//! Core failures use [`Error::Core`], and concrete lowering failures have their own
//! [`Error`] variants. Logical `shard_map` errors raised while tracing or lowering remain
//! [`ProgramError::Custom`] payloads, recoverable through
//! [`ProgramError::downcast_custom`](ryft_core::ProgramError::downcast_custom).

// TODO(eaplatanios): Review this module.

use std::cell::RefCell;

use ryft_core::{
    ArrayIrType, ArrayType, DomainTracingContext, Parameterized, ParameterizedFamily, ProgramError, ProgramStatistics,
    ProjectedValue, Sharding, ValueProjection,
};

use crate::{Constant, Error, Program, Session, Tracer};

/// Array value that closures traced through [`trace`] receive and return. It is a
/// [`DomainTracer`](ryft_core::DomainTracer) of [`Session`] projected onto its array member. The `shard_map` bodies
/// traced in XLA programs (e.g., through [`ryft_core::shard_map()`] on values of type [`ArrayTracer`]) receive the same
/// values. This alias is public so that callers (e.g., tooling binaries) can annotate traced closure parameters.
pub type ArrayTracer = ProjectedValue<ArrayType, Tracer<'static>>;

/// Stages an arbitrary traced XLA function over global array types.
///
/// This is the general XLA tracing entry point used when callers want to compose `shard_map` with differentiation
/// transforms and then lower the resulting whole program to StableHLO/Shardy MLIR. `function` is traced once, in a
/// fresh root tracing context of [`Session`] that binds no named axes, over [`ArrayTracer`]s of
/// `global_input_types`, and the traced program is simplified. Functions that need that context (e.g., to create values
/// that derive from no input) or that can fail are traced through [`trace_with_context`] instead.
///
/// # Parameters
///
///   - `function`: Function to trace over global XLA values.
///   - `global_input_types`: Global input array types passed to the traced function.
///
/// # Errors
///
/// Returns the errors of [`trace_with_context`].
pub fn trace<F, Input, Output>(function: F, global_input_types: Input) -> Result<TracedProgram<Input, Output>, Error>
where
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer>,
    Output: Parameterized<ArrayType>,
    Output::Family: ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer>,
    F: FnOnce(Input::To<ArrayTracer>) -> Output::To<ArrayTracer>,
{
    trace_with_context(|_, inputs| Ok(function(inputs)), global_input_types)
}

/// Stages a fallible XLA function over global array types like [`trace`], passing it the fresh root tracing context
/// of [`Session`] together with its inputs. The context lets the function create values that derive from no input
/// (e.g., through [`Context::lift`](ryft_core::Context::lift)) and bind context capabilities directly (e.g., a
/// `shard_map` without inputs through [`ShardMap`](ryft_core::ShardMap)), and the function must not retain it or its
/// tracers after returning.
///
/// # Parameters
///
///   - `function`: Function to trace over the tracing context and global XLA values.
///   - `global_input_types`: Global input array types passed to the traced function. Use `()` for no inputs.
///
/// # Errors
///
/// Returns [`Error::Core`] wrapping a [`ProgramError`] when `function` fails or when tracing or simplifying the program
/// fails (e.g., because `function` stages an invalid operation or registers a capture, which the fresh root context
/// discards), or a [`ParameterError`](ryft_core::ParameterError) when the traced inputs or outputs cannot be
/// restructured into `Input` or `Output`.
pub fn trace_with_context<F, Input, Output>(
    function: F,
    global_input_types: Input,
) -> Result<TracedProgram<Input, Output>, Error>
where
    Input: Parameterized<ArrayType>,
    Input::Family: ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer>,
    Output: Parameterized<ArrayType>,
    Output::Family: ParameterizedFamily<Constant> + ParameterizedFamily<ArrayTracer>,
    F: FnOnce(
        &DomainTracingContext<Session<'static>>,
        Input::To<ArrayTracer>,
    ) -> Result<Output::To<ArrayTracer>, ProgramError>,
{
    let input_structure = global_input_types.parameter_structure();
    let output_structure = RefCell::new(None);
    let (output_types, program) = DomainTracingContext::<Session<'static>>::trace_with_context_and_named_axes(
        |context, inputs: Vec<Tracer<'static>>| {
            let inputs = inputs
                .into_iter()
                .map(|input| ValueProjection::<ArrayType>::into_projected(input).map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs =
                function(context, Input::To::<ArrayTracer>::from_parameters(input_structure.clone(), inputs)?)?;
            output_structure.replace(Some(outputs.parameter_structure()));
            Ok(outputs.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>())
        },
        global_input_types.parameters().cloned().map(ArrayIrType::Array).collect::<Vec<_>>(),
        Vec::new(),
    )?;
    let output_structure = output_structure.into_inner().ok_or_else(|| {
        ProgramError::MalformedProgram("XLA tracing completed without recording its output structure".to_string())
    })?;
    let global_output_types = Output::from_parameters(
        output_structure.clone(),
        output_types
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?,
    )?;
    let program = program
        .into_simplified()?
        .restructured::<Input::To<Constant>, Output::To<Constant>>(input_structure, output_structure)?;
    Ok(TracedProgram { global_input_types, global_output_types, program })
}

/// Traced XLA program backed by a staged [`Program`](ryft_core::Program), as returned by [`trace`].
pub struct TracedProgram<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>
where
    Input::Family: ParameterizedFamily<Constant>,
    Output::Family: ParameterizedFamily<Constant>,
{
    /// Global input types supplied to the traced function.
    global_input_types: Input,

    /// Global output types inferred by tracing the function.
    global_output_types: Output,

    /// Staged traced XLA program specialized to abstract array leaves.
    program: Program<Input::To<Constant>, Output::To<Constant>>,
}

impl<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>> TracedProgram<Input, Output>
where
    Input::Family: ParameterizedFamily<Constant>,
    Output::Family: ParameterizedFamily<Constant>,
{
    /// Returns backend-neutral structural statistics for the staged traced XLA program backing this handle. The
    /// statistics describe the program as [`trace`] stores it, which it has already simplified (refer to
    /// [`Program::into_simplified`](ryft_core::Program::into_simplified)). Instructions that reach no program output
    /// are not counted unless they have observable effects, while the boundary pruning of operations with attached
    /// regions (refer to [`Program::into_pruned`](ryft_core::Program::into_pruned)), which lowering applies, is not
    /// reflected in the counts. Refer to the documentation of [`Program::statistics`](ryft_core::Program::statistics)
    /// for the precise semantics of the reported statistics.
    pub fn statistics(&self) -> ProgramStatistics {
        self.program.statistics()
    }

    /// Returns the traced global input types.
    pub fn global_input_types(&self) -> &Input {
        &self.global_input_types
    }

    /// Returns the traced global output types.
    pub fn global_output_types(&self) -> &Output {
        &self.global_output_types
    }

    /// Returns the staged program that backs this handle.
    #[cfg(test)]
    pub(crate) fn program(&self) -> &Program<Input::To<Constant>, Output::To<Constant>> {
        &self.program
    }

    /// Renders a full StableHLO/Shardy MLIR module for this traced XLA program, whose function signature carries the
    /// shardings of the global input and output types that have one (refer to
    /// [`Self::to_mlir_module_with_signature_shardings`]).
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Self::to_mlir_module_with_signature_shardings`].
    pub fn to_mlir_module<S: AsRef<str>>(&self, function_name: S) -> Result<String, Error> {
        self.to_mlir_module_with_signature_shardings(function_name, None, None)
    }

    /// Same as [`Self::to_mlir_module`] but with explicitly provided `sdy.sharding` attributes for the function's
    /// arguments and/or results. This is what the XLA SPMD partitioner reads to drive boundary slicing of per-device
    /// output buffers, including for shapes whose dimensions are not divisible by the partition count (e.g., shape
    /// `[5]` on 2 partitions producing `[3]` + `[2]`). A side without explicit shardings carries the sharding of each
    /// of its global types that has one, as the compilation path of [`Session`] does. Unlike that path, which places
    /// a value whose type has no sharding as replicated because it lowers for concrete placed arrays, such a value is
    /// left unannotated here, so that Shardy propagates its placement from the program (as JAX does for unspecified
    /// shardings).
    ///
    /// The traced program is pruned before lowering (refer to
    /// [`Program::into_pruned`](ryft_core::Program::into_pruned)), which drops dead outputs of instructions that carry
    /// regions together with the body computation that only they use (e.g., the dead primal outputs of a forward mode
    /// `shard_map` whose residuals feed a gradient), and is then simplified. A `shard_map` whose ordinary
    /// outputs are all dead but whose body has observable effects (e.g., prints with the
    /// [`DeviceOrderedIo`](ryft_core::EffectClass::DeviceOrderedIo) effect) is kept with no ordinary outputs and lowers
    /// to a manual computation that only threads its effect token.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func`.
    ///   - `argument_shardings`: Optional shardings to attach to each function argument instead of those of the global
    ///     input types. Must have the same length as the global input types, or be `None`.
    ///   - `result_shardings`: Optional shardings to attach to each function result instead of those of the global
    ///     output types. Must have the same length as the global output types, or be `None`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Core`] when pruning or simplifying the traced program fails. Lowering failures use the
    /// corresponding [`Error`] variants (e.g., for an invalid function name, mismatched signature shardings, unresolved
    /// references or state, or operations that the backend cannot lower).
    pub fn to_mlir_module_with_signature_shardings<S: AsRef<str>>(
        &self,
        function_name: S,
        argument_shardings: Option<&[Sharding]>,
        result_shardings: Option<&[Sharding]>,
    ) -> Result<String, Error> {
        /// Returns the explicit `shardings` as one optional sharding per value when they are provided, and otherwise
        /// the sharding of each of `types`, which is `None` for types that have no sharding.
        fn value_shardings<'o>(
            shardings: Option<&[Sharding]>,
            types: impl Iterator<Item = &'o ArrayType>,
        ) -> Vec<Option<Sharding>> {
            match shardings {
                Some(shardings) => shardings.iter().cloned().map(Some).collect(),
                None => types.map(|r#type| r#type.sharding().cloned()).collect(),
            }
        }

        let simplified_program = self.program.clone().into_pruned()?.into_simplified()?;
        Ok(super::lowering::lower_mlir_module_for_program_with_value_shardings(
            &simplified_program,
            &[],
            &self.global_input_types,
            &self.global_output_types,
            function_name,
            &value_shardings(argument_shardings, self.global_input_types.parameters()),
            &value_shardings(result_shardings, self.global_output_types.parameters()),
        )?
        .into_stable_hlo())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ConstrainSharding, DataType, Dimension as CoreDimension, EffectClass, LogicalMesh, MeshAxis, MeshAxisType,
        Print, RegionRole, Reshape, Reshard, Shape, ShardMapOptions, Sharding, ShardingDimension, Sin, shard_map,
    };
    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program};

    use crate::lowering::tests::{test_sharding, test_spmd_compilation_options};
    use crate::tests::{cpu_client, values_from_bytes, values_to_bytes};
    use crate::{ensure_print_handler_registered, with_captured_prints};

    use super::*;

    /// Traces a program over a sharded `f64[4]` input that invokes a `shard_map` over the manual mesh of two devices,
    /// whose body prints its local shard with the [`EffectClass::DeviceOrderedIo`] effect and whose only output is
    /// dead, and that returns its input unchanged.
    fn traced_effectful_dead_shard_map() -> (TracedProgram<ArrayType, ArrayType>, Sharding) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced = trace(
            {
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    let _printed = shard_map(
                        |local_x: ArrayTracer| local_x.print_with_effect_class("body", EffectClass::DeviceOrderedIo),
                        x.clone(),
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap();
                    x
                }
            },
            ArrayType::new_static(DataType::F64, [4]),
        )
        .unwrap();
        (traced, sharding)
    }

    #[test]
    fn test_trace() {
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(x, y): (ArrayTracer, ArrayTracer)| x.sin().unwrap() * y,
            (ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            traced.global_input_types(),
            &(ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [4])),
        );
        assert_eq!(traced.global_output_types(), &ArrayType::new_static(DataType::F32, [4]));
        assert_eq!(
            traced.program.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4] = sin %0
                    %3:f32[4] = mul %2 %1
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_trace_with_context() {
        // Failures of the traced function propagate unchanged as core program errors.
        assert!(matches!(
            trace_with_context::<_, ArrayType, ArrayType>(
                |_, _| Err(ProgramError::MalformedProgram("traced function failure".to_string())),
                ArrayType::new_static(DataType::F32, [4]),
            ),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                ryft_core::Error::Program(ProgramError::MalformedProgram(message))
                    if message == "traced function failure",
            ),
        ));
    }

    #[test]
    fn test_traced_program_statistics() {
        // `TracedProgram::statistics` delegates to the stored simplified staged program. The asserted numbers are
        // cross-checked against `traced.program.to_string()`, which renders the entry region as one `shard_map`
        // instruction that attaches a body region with a single `sin` instruction.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                move |x: ArrayTracer| {
                    shard_map(
                        |local_x: ArrayTracer| local_x.sin(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)])),
        )
        .unwrap();

        let statistics = traced.statistics();
        assert_eq!(statistics.region_count(), 2);

        let body = &statistics.regions()[0];
        assert_eq!(body.input_count(), 1);
        assert_eq!(body.output_count(), 1);
        assert_eq!(body.instruction_count(), 1);
        assert_eq!(body.constant_count(), 0);
        assert_eq!(body.operation_counts(), &BTreeMap::from([("sin", 1usize)]));
        assert_eq!(body.maximum_output_dependency_depth(), 1);
        assert_eq!(body.attached_regions(), &[]);

        let entry = statistics.entry_region_statistics();
        assert_eq!(entry.input_count(), 1);
        assert_eq!(entry.output_count(), 1);
        assert_eq!(entry.instruction_count(), 1);
        assert_eq!(entry.operation_counts(), &BTreeMap::from([("shard_map", 1usize)]));
        assert_eq!(entry.maximum_output_dependency_depth(), 1);
        assert_eq!(entry.attached_regions().len(), 1);
        let edge = &entry.attached_regions()[0];
        assert_eq!(edge.instruction_index(), 0);
        assert_eq!(edge.operation(), "shard_map");
        assert_eq!(edge.region_slot(), "body");
        assert_eq!(edge.region_role(), RegionRole::Computation);
        assert_eq!(edge.region_index(), 0);
        assert_eq!(edge.label(), "shard_map.body");
    }

    #[test]
    fn test_traced_program_to_mlir_module() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));

        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ArrayTracer| x.sin().unwrap().reshard(&sharding).unwrap()
            },
            global_input_type.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_traced_program_to_mlir_module_reshard_then_reshape() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]));
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    x.reshard(&sharding)
                        .unwrap()
                        .reshape(Shape::new(vec![
                            CoreDimension::Static(1),
                            CoreDimension::Static(8),
                            CoreDimension::Static(1),
                        ]))
                        .unwrap()
                }
            },
            input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<1x8x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x"}]> : tensor<8xf32>
                    %1 = stablehlo.reshape %0 : (tensor<8xf32>) -> tensor<1x8x1xf32>
                    return %1 : tensor<1x8x1xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_traced_program_to_mlir_module_merges_tracked_placement_into_sharding_constraint() {
        // The input is tracked as sharded over the explicit axis and the constraint places it over the auto axis, so
        // the emitted constraint must carry both placements rather than the constraint alone, which would contradict
        // the tracked type by marking the explicit axis replicated.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let tracked = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let constraint = test_sharding(&mesh, vec![ShardingDimension::sharded(["a"])], vec![]);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![CoreDimension::Static(8)]))
            .with_sharding(tracked.clone())
            .unwrap();

        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let constraint = constraint.clone();
                move |x: ArrayTracer| x.constrain_sharding(&constraint).unwrap()
            },
            global_input_type.clone(),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "a"=2]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{"x", "a"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_traced_program_to_mlir_module_with_signature_shardings_keeps_effectful_dead_shard_maps() {
        // Pruning keeps a `shard_map` whose ordinary results are all dead when its body has observable effects, with no
        // ordinary results, so the lowered manual computation only threads the effect token of its body.
        let (traced, sharding) = traced_effectful_dead_shard_map();
        assert_eq!(
            traced
                .to_mlir_module_with_signature_shardings(
                    "main",
                    Some(std::slice::from_ref(&sharding)),
                    Some(std::slice::from_ref(&sharding)),
                )
                .unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4xf64> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = stablehlo.after_all  : !stablehlo.token
                    %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{"x"}]>, <@mesh, []>] out_shardings=[<@mesh, []>] manual_axes={"x"} (%arg1: tensor<2xf64>, %arg2: !stablehlo.token) {
                      %2 = stablehlo.custom_call @ryft.print(%arg1, %arg2) {api_version = 4 : i32, backend_config = {label = "body"}, has_side_effect = true} : (tensor<2xf64>, !stablehlo.token) -> !stablehlo.token
                      sdy.return %2 : !stablehlo.token
                    } : (tensor<4xf64>, !stablehlo.token) -> !stablehlo.token
                    return %arg0 : tensor<4xf64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_traced_program_to_mlir_module_with_signature_shardings_executes_effectful_dead_shard_maps_on_cpu() {
        let (traced, sharding) = traced_effectful_dead_shard_map();
        let module = traced
            .to_mlir_module_with_signature_shardings(
                "main",
                Some(std::slice::from_ref(&sharding)),
                Some(std::slice::from_ref(&sharding)),
            )
            .unwrap();
        let client = cpu_client(2);
        ensure_print_handler_registered(&client).unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let shards = [vec![1.5f64, 2.5], vec![3.5f64, 4.5]];
        let inputs = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .zip(&shards)
            .map(|(device, shard)| {
                let bytes = values_to_bytes(shard.as_slice());
                vec![ExecutionInput {
                    buffer: std::sync::Arc::new(
                        client.buffer(bytes.as_slice(), BufferType::F64, [2u64], None, device.clone(), None).unwrap(),
                    ),
                    donatable: false,
                }]
            })
            .collect::<Vec<_>>();
        let (outputs, mut lines) = with_captured_prints(|| {
            executable
                .execute(
                    inputs.iter().map(|inputs| ExecutionDeviceInputs { inputs, ..Default::default() }).collect(),
                    Vec::new(),
                    0,
                    None,
                    Some(file!()),
                    None,
                    None,
                    &[],
                )
                .unwrap()
                .block_until_ready()
                .unwrap()
        });

        // Every device returns its own shard and prints it exactly once, although nothing uses the `shard_map` output.
        assert_eq!(outputs.len(), 2);
        for (output, shard) in outputs.iter().zip(&shards) {
            assert_eq!(output.outputs.len(), 1);
            let bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(&values_from_bytes::<f64>(&bytes), shard);
        }
        lines.sort();
        assert_eq!(lines, vec!["body: [1.5, 2.5]".to_string(), "body: [3.5, 4.5]".to_string()]);
    }
}
