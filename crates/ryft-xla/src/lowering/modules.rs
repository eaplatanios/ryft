//! Lowering of whole programs to StableHLO modules, including the entry function, its boundary attributes, the
//! private functions that calls share, and the passes that run over the emitted module.

use ryft_core::{
    ArrayType, Device, Dimension, EffectClass, MeshAxisType, Parameterized, Program, Sharding, ShardingDimension,
    Type as CoreType, Typed,
};
use ryft_mlir::{Module, Operation as MlirOperation, OperationPrintingFlags, WalkOrder, WalkResult};

use crate::lowering::emissions::emitted_regions;
use crate::lowering::states::ProgramLoweringContext;
use crate::lowering::{LowerableOperationFamily, LoweringError};
use crate::{Constant, Error};

// TODO(eaplatanios): Review this module.

/// Returns `true` if any operation in a region of `program` that lowering emits (i.e., its entry region and the
/// executed nested regions of that region) carries unresolved ordered state.
///
/// This scans those regions rather than consulting the effect summary of the program, because ordinary XLA cannot
/// preserve state in any lowered region, including one whose effects the summary of its owner does not expose.
/// Regions that lowering never reaches (e.g., the rule regions of composites, the transpose region of a `linear_call`,
/// or the body of a scan that admits no iteration, refer to [`emitted_regions`], here for no target platform) are not
/// part of the lowered module, so state in them is never lowered and cannot reject the program.
pub(crate) fn contains_unresolved_state<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
) -> bool
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    emitted_regions(program.entry_region_ref(), None).any(|region| {
        region
            .instructions()
            .iter()
            .any(|instruction| instruction.operation().effects().classes().contains(EffectClass::OrderedState))
    })
}

/// Returns `true` if `program` contains a reference-typed atom or a [`CoreOperation`](ryft_core::Operation) whose
/// [effects](ryft_core::Operation::effects) declare a [`ReferenceEffect`](ryft_core::ReferenceEffect) in a region that
/// lowering emits (i.e., its entry region and the executed nested regions of that region, refer to [`emitted_regions`],
/// here for no target platform). Regions that lowering never reaches (e.g., the rule regions of composites, the
/// transpose region of a `linear_call`, or the body of a scan that admits no iteration) are not part of the lowered
/// module, so references in them neither need discharge nor reject the program.
///
/// This check is independent from [`contains_unresolved_state`], because a pure reference pass-through or a forwarded
/// reference capture can carry reference values without executing a stateful instruction. Both predicates back the
/// state checks that precede compilation and the two direct module lowering entry points. Eager binding, dispatch, and
/// staging enforce their corresponding boundary invariants separately.
pub(crate) fn contains_unresolved_references<ProgramInput, ProgramOutput, O: LowerableOperationFamily>(
    program: &Program<Constant, O, ProgramInput, ProgramOutput>,
) -> bool
where
    ProgramInput: Parameterized<Constant>,
    ProgramOutput: Parameterized<Constant>,
{
    // The atom scan dominates for every well-formed program, because reference operations always touch an atom of a
    // reference type. The declaration scan is an independent check that still catches a core discharge bug which
    // retyped the boundary atoms but left a reference operation behind.
    emitted_regions(program.entry_region_ref(), None).any(|region| {
        region.atoms().iter().any(|atom| atom.r#type().is_reference())
            || region
                .instructions()
                .iter()
                .any(|instruction| instruction.operation().effects().has_reference_effects())
    })
}

/// Serializes one lowered module to StableHLO text, printing MLIR debug information (i.e., locations) exactly when
/// the lowering constructed provenance locations specific to instructions. Modules without provenance keep the plain
/// rendering, so their StableHLO text, snapshots, and compilation cache keys remain byte for byte what they were before
/// provenance existed, while modules that carry provenance embed their locations in the text and therefore in the
/// persistent compilation key derived from it. The pretty debug form is documented as unparsable and never used, and
/// the elision thresholds are disabled because the plain rendering performs no elision either.
///
/// Exact backend provenance can reduce the reuse of cached compiled artifacts, because two semantically equal modules
/// with different provenance serialize differently and therefore have different keys. If that cost ever matters, the
/// correct extension is an explicit compilation option that strips provenance before lowering, which makes this helper
/// fall back to the plain rendering. Silently returning an artifact labeled with the provenance of another program is
/// never acceptable.
pub(crate) fn serialize_lowered_module(
    module: &Module<'_, '_>,
    lowering_context: &ProgramLoweringContext,
) -> Result<String, Error> {
    if !lowering_context.has_provenance() {
        return Ok(module.to_string());
    }
    module
        .as_operation()?
        .to_string_with_flags(OperationPrintingFlags {
            elements_attribute_size_threshold: None,
            resource_string_size_threshold: None,
            enable_debug_information: true,
            pretty_print_debug_information: false,
            ..OperationPrintingFlags::default()
        })
        .map_err(|error| ryft_mlir::Error::internal(format!("serialized module is not valid UTF-8: {error}")).into())
}

/// Renders `devices` as a list in diagnostics (e.g., `[0, 1]`), naming the process of each device that is not on
/// process zero (e.g., `[0, 5@1]`, refer to the [`Display`](std::fmt::Display) implementation of [`Device`]), so that
/// devices of different processes with the same identifier are told apart.
pub(crate) fn render_devices(devices: &[Device]) -> String {
    format!("[{}]", devices.iter().map(Device::to_string).collect::<Vec<_>>().join(", "))
}

/// Returns whether a broadcast from `input_type` to `output_type`, which maps each input axis to the corresponding
/// entry of `output_axes`, changes the placement over explicit mesh axes and therefore requires an explicit output
/// placement constraint.
pub(crate) fn broadcast_changes_explicit_sharding(
    input_type: &ArrayType,
    output_type: &ArrayType,
    output_axes: &[usize],
) -> bool {
    let Some(output_sharding) = output_type.sharding() else {
        return false;
    };
    let input_sharding = input_type.sharding();
    let uses_explicit_axis = |dimension: &ShardingDimension, sharding: &Sharding| match dimension {
        ShardingDimension::Sharded(axis_names) => {
            axis_names.iter().any(|name| sharding.mesh().axis_type(name) == Some(MeshAxisType::Explicit))
        }
        ShardingDimension::Replicated | ShardingDimension::Unconstrained => false,
    };
    if input_sharding.is_none() {
        return output_sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Explicit);
    }
    if let Some(input_sharding) = input_sharding
        && input_sharding.mesh() != output_sharding.mesh()
    {
        return input_sharding
            .mesh()
            .axes()
            .iter()
            .chain(output_sharding.mesh().axes())
            .any(|axis| axis.r#type() == MeshAxisType::Explicit);
    }

    let projected_input_sharding = input_sharding
        .map(|sharding| sharding.with_broadcasted_dimensions(output_type.rank(), output_axes).unwrap())
        .unwrap_or_else(|| Sharding::replicated(output_sharding.mesh().clone(), output_type.rank()));

    // Even when the ranked placement labels are unchanged, expanding a unit dimension that is partitioned over an
    // explicit mesh axis changes which partitions own the replicated values and must remain visible to Shardy.
    let expands_explicit_dimension = output_axes.iter().copied().enumerate().any(|(input_axis, output_axis)| {
        input_type.dimension(input_axis as isize) == Dimension::Static(1)
            && output_type.dimension(output_axis as isize) != Dimension::Static(1)
            && (uses_explicit_axis(&projected_input_sharding.dimensions()[output_axis], &projected_input_sharding)
                || uses_explicit_axis(&output_sharding.dimensions()[output_axis], output_sharding))
    });

    expands_explicit_dimension || projected_input_sharding.conflicts_on_explicit_axes_with(output_sharding)
}

/// Returns whether a cast from `input_type` to `output_type` changes the placement that the type system tracks over
/// explicit mesh axes and therefore emits a sharding constraint carrying its complete output sharding, like
/// [`ReshardOperation`](ryft_core::ReshardOperation). Carrying an existing sharding forward requests no redistribution,
/// and an output without a sharding is untracked rather than replicated. Type inference guarantees that the placement
/// over auto and manual axes is unchanged, as documented on [`CastOperation`](ryft_core::CastOperation), so the emitted
/// constraint never contradicts the type the program was checked against.
pub(crate) fn cast_changes_explicit_placement(input_type: &ArrayType, output_type: &ArrayType) -> bool {
    let Some(output_sharding) = output_type.sharding() else {
        return false;
    };
    let has_explicit_axis =
        |sharding: &Sharding| sharding.mesh().axes().iter().any(|axis| axis.r#type() == MeshAxisType::Explicit);
    match input_type.sharding() {
        None => has_explicit_axis(output_sharding),
        Some(input_sharding) if input_sharding.mesh() != output_sharding.mesh() => {
            has_explicit_axis(input_sharding) || has_explicit_axis(output_sharding)
        }
        Some(input_sharding) => input_sharding.conflicts_on_explicit_axes_with(output_sharding),
    }
}

/// Removes every Shardy operation from `module` (refer to [`Module::remove_shardy_operations`]), for a plugin that
/// cannot compile with Shardy. Ryft runs programs on such targets on a single device. Every sharding then holds the
/// whole array, so removing its constraints preserves the result.
///
/// # Errors
///
/// Returns [`LoweringError::UnsupportedOperation`] for any other Shardy operation (e.g., the `sdy.manual_computation`
/// of a `shard_map`), which has no equivalent without the dialect.
pub(crate) fn remove_shardy_operations(module: &mut Module<'_, '_>) -> Result<(), Error> {
    let Err(error) = module.remove_shardy_operations() else {
        return Ok(());
    };
    // Name the operation that cannot be removed, which is the first one other than the two that can.
    let mut unsupported = None;
    module.as_operation()?.walk(WalkOrder::PreOrder, |operation| match operation.name().as_str() {
        Ok(name) if name.starts_with("sdy.") && name != "sdy.sharding_constraint" && name != "sdy.mesh" => {
            unsupported = Some(name.to_string());
            WalkResult::Interrupt
        }
        _ => WalkResult::Advance,
    });
    Err(match unsupported {
        Some(operation) => Error::Lowering(LoweringError::UnsupportedOperation {
            operation,
            reason: "the target plugin cannot compile with Shardy".to_string(),
        }),
        None => error.into(),
    })
}

/// Normalizes an MLIR symbol name provided by the user by trimming surrounding whitespace and one leading `@`.
///
/// # Errors
///
/// Returns [`LoweringError::InvalidFunctionName`] when the normalized name is empty or contains whitespace.
pub(crate) fn normalize_function_name(function_name: &str) -> Result<String, Error> {
    let function_name = function_name.trim();
    let function_name = function_name.strip_prefix('@').unwrap_or(function_name);
    if function_name.is_empty() || function_name.chars().any(char::is_whitespace) {
        return Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name: function_name.to_string() }));
    }
    Ok(function_name.to_string())
}

/// Expands optional shardings of every value of a signature with `count` values into one optional sharding per value.
/// Provided shardings are kept as they are, so that a count mismatch is still reported when the module is lowered.
pub(crate) fn value_shardings(shardings: Option<&[Sharding]>, count: usize) -> Vec<Option<Sharding>> {
    match shardings {
        Some(shardings) => shardings.iter().cloned().map(Some).collect(),
        None => vec![None; count],
    }
}

#[cfg(test)]
mod tests {

    use std::collections::HashMap;

    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, Array, ArrayOperation, ArrayType, CaptureReference, ConditionOperation, Context, Cos, DataType,
        Device, Differentiate, Dimension, DimensionBounds, DimensionToScalarOperation, DimensionType, DimensionValue,
        DimensionVariable, Dot, DotDimensionNumbers, EagerContext, EffectClass, EffectClasses,
        ExternalReferenceBinding, FlatProgram, Layout, LogicalMesh, MeshAxis, MeshAxisType, OneLike, PlaceOperation,
        Placeholder, PrintOperation, Program, ProgramBuilder as ArrayIrProgramBuilder, Provenance, ProvenanceScope,
        ReferenceFreezeOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceSource, ReferenceType,
        ReferenceWriteOperation, ReverseModeDifferentiate, Shape, ShardMapOptions, Sharding, ShardingDimension, Sin,
        TiledLayout, Transpose, TypedProgram, ZeroLike,
    };
    use ryft_kernels::{KernelBoundaryContract, KernelParameterAccess, validate_kernel_body};
    use ryft_mlir::dialects::{func, stable_hlo};
    use ryft_mlir::{
        Attribute, Block, Context as MlirContext, DialectHandle, Operation as MlirOperation, Region, Size as MlirSize,
        Type as MlirType, TypeAndAttributes,
    };

    use ryft_pjrt::{BufferType, ExecutionDeviceInputs, ExecutionInput, Program as PjrtProgram};

    use crate::{Constant, Error, ExecutionError, Operation, Session, Type};

    use crate::lowering::{LoweringError, token_threaded_effects};

    use crate::lowering::tests::{
        TEST_ARRAY_DOMAIN, add_reference_state, dynamic_dimension, linear_call_program,
        lower_replicated_shard_map_body, neg_branch, scalar_bilinear_sin, scalar_linear_region,
        scalar_quartic_plus_sin, shard_map_module, stateful_scan_program, test_manual_mesh, test_matrix_type,
        test_vector_type, to_rendered_mlir_for_plain_program, unproject_plain_program,
    };

    use crate::tests::{cpu_client, test_compilation_options, values_from_bytes, values_to_bytes};

    use super::*;
    use crate::lowering::programs::{MlirProgram, ProgramLoweringExtension, ProgramLoweringOptions};

    #[test]
    fn test_contains_unresolved_state_ignores_dormant_regions() {
        let stateful =
            |extended| scalar_linear_region(&ArrayType::scalar(DataType::F32), extended, add_reference_state);

        // State in the dormant transpose region of a `linear_call` is never lowered, unlike state in its forward
        // region, and neither is state in the body of a scan that admits no iteration, unlike state in one that
        // iterates.
        assert!(!contains_unresolved_state(&linear_call_program(&stateful(false), &stateful(true))));
        assert!(contains_unresolved_state(&linear_call_program(&stateful(true), &stateful(false))));
        assert!(!contains_unresolved_state(&stateful_scan_program(0)));
        assert!(contains_unresolved_state(&stateful_scan_program(1)));
    }

    #[test]
    fn test_contains_unresolved_references_ignores_dormant_regions() {
        let stateful =
            |extended| scalar_linear_region(&ArrayType::scalar(DataType::F32), extended, add_reference_state);

        // References in the dormant transpose region of a `linear_call` are never lowered, unlike references in its
        // forward region, and neither are references in the body of a scan that admits no iteration.
        assert!(!contains_unresolved_references(&linear_call_program(&stateful(false), &stateful(true))));
        assert!(contains_unresolved_references(&linear_call_program(&stateful(true), &stateful(false))));
        assert!(!contains_unresolved_references(&stateful_scan_program(0)));
        assert!(contains_unresolved_references(&stateful_scan_program(1)));
    }

    #[test]
    fn test_lowered_module_new() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| Ok(x.clone() + x),
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_renders_provenance_locations() {
        let global_input_type = test_vector_type(8);
        let mesh = test_manual_mesh("x", 4);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| {
                let context = x.value().context().clone();
                // One instruction staged under nested scopes, one under a fused origin, and one with unknown
                // provenance, covering every provenance shape in one lowered module.
                let scoped = context.invoke_with_provenance_scope(ProvenanceScope::new("outer"), || {
                    context.invoke_with_provenance_scope(ProvenanceScope::new("inner"), || x.clone() + x.clone())
                });
                let fused = Provenance::fused([
                    Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown()),
                    Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown()),
                ]);
                let product = context.invoke_with_provenance_origin(fused, || scoped.clone() * scoped);
                Ok(product + x)
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        // The scoped addition carries the nested chain of name locations, the fused multiplication carries a fused
        // location without metadata, the addition with unknown provenance keeps the unknown base location, and the
        // module and function scaffolding keeps the base location. Locations render because modules that carry
        // provenance serialize with debug information enabled, in the parsable form that is not pretty printed.
        assert_eq!(
            module,
            indoc! {r#"
                #loc = loc(unknown)
                module {
                  sdy.mesh @mesh = <["x"=4]> loc(#loc)
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>} loc(unknown)) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32> loc(unknown)) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32> loc(#loc4)
                      %2 = stablehlo.multiply %1, %1 : tensor<2xf32> loc(#loc5)
                      %3 = stablehlo.add %2, %arg1 : tensor<2xf32> loc(#loc)
                      sdy.return %3 : tensor<2xf32> loc(#loc)
                    } : (tensor<8xf32>) -> tensor<8xf32> loc(#loc)
                    return %0 : tensor<8xf32> loc(#loc)
                  } loc(#loc)
                } loc(#loc)
                #loc1 = loc("inner")
                #loc2 = loc("a")
                #loc3 = loc("b")
                #loc4 = loc("outer"(#loc1))
                #loc5 = loc(fused[#loc2, #loc3])
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_renders_constants_and_supported_ops() {
        let global_input_type = test_matrix_type(4, 4);
        let mesh = test_manual_mesh("x", 2);
        let module = shard_map_module::<_, ArrayType, ArrayType>(
            |_, x| {
                let product = x.transpose(vec![1, 0])?.dot(&x, &DotDimensionNumbers::matmul())?;
                let waveform = (-product).cos()?.sin()?;
                Ok((waveform.clone() * waveform.one_like()?) + waveform.zero_like()?)
            },
            global_input_type,
            mesh.clone(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap(),
            Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()]).unwrap(),
            ShardMapOptions::new(),
        )
        .unwrap();

        assert_eq!(
            module,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg1: tensor<2x4xf32>) {
                      %1 = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<2x4xf32>) -> tensor<4x2xf32>
                      %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x2xf32>, tensor<2x4xf32>) -> tensor<4x4xf32>
                      %3 = stablehlo.negate %2 : tensor<4x4xf32>
                      %4 = stablehlo.cosine %3 : tensor<4x4xf32>
                      %5 = stablehlo.sine %4 : tensor<4x4xf32>
                      %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
                      %6 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %7 = sdy.sharding_constraint %6 <@mesh, [{}, {}]> : tensor<4x4xf32>
                      %8 = stablehlo.multiply %5, %7 : tensor<4x4xf32>
                      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                      %9 = stablehlo.broadcast_in_dim %cst_0, dims = [] : (tensor<f32>) -> tensor<4x4xf32>
                      %10 = sdy.sharding_constraint %9 <@mesh, [{}, {}]> : tensor<4x4xf32>
                      %11 = stablehlo.add %8, %10 : tensor<4x4xf32>
                      sdy.return %11 : tensor<4x4xf32>
                    } : (tensor<4x4xf32>) -> tensor<8x4xf32>
                    return %0 : tensor<8x4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_preserves_boundary_layouts() {
        let array_type = ArrayType::new_static(DataType::F32, [2, 3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let zero_type = ArrayType::new_static(DataType::Zero, [2, 3]);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(array_type.clone().into());
        let zero = builder.add_input(zero_type.clone().into());
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![zero, input], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) \
                      -> (tensor<2x3xf32> {mhlo.layout_mode = \"{0,1}\"}) {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<2x3xi1>
                    return %arg0 : tensor<2x3xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_lowered_module_new_lowers_captures_as_hidden_inputs() {
        let array_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(array_type.clone().into());
        let capture = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, capture], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let capture_types = vec![array_type.clone()];
        let input_types = vec![array_type.clone()];
        let output_types = vec![array_type];
        let stablehlo = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_capture_types(capture_types.to_vec())
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings(),
            )
            .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg1, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_lowers_immediate_dimension_constants_in_place() {
        // An immediate dimension constant carries its own extent from the host, so unlike a capture reference, it
        // materializes as a scalar `i64` `stablehlo.constant` in the function body instead of consuming a hidden
        // capture input. That is exactly what makes first-class extents usable where no capture table is reachable,
        // such as inside a `shard_map` manual region.
        let extent_type = DimensionType::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let extent = builder.add_constant(Constant::Dimension(DimensionValue::new(extent_type, 4).unwrap()));
        let output = builder.add_instruction(DimensionToScalarOperation, Vec::new(), vec![extent], None).unwrap()[0];
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main() -> tensor<i64> {
                    %c = stablehlo.constant dense<4> : tensor<i64>
                    return %c : tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_lowers_capture_output_as_a_hidden_input() {
        let array_type = test_vector_type(4);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
        let program =
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
        let input_types: [ArrayType; 0] = [];
        let output_types = vec![array_type.clone()];
        let stablehlo = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_capture_types((std::slice::from_ref(&array_type)).to_vec())
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings(),
            )
            .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    return %arg0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_forwards_capture_into_nested_regions() {
        let array_type = test_vector_type(4);
        let branch = || {
            let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
            let output = builder.add_constant(Constant::Captured(CaptureReference::new(0, array_type.clone().into())));
            builder.build::<Vec<Constant>, Vec<Constant>>(vec![output], Vec::new(), vec![Placeholder]).unwrap()
        };
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let true_region = builder.import_program(branch());
        let false_region = builder.import_program(branch());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let output = builder
            .add_instruction(
                Operation::Condition(ConditionOperation::new()),
                vec![true_region, false_region],
                vec![predicate],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = vec![array_type.clone()];
        let stablehlo = program
            .to_rendered_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_capture_types((std::slice::from_ref(&array_type)).to_vec())
                    .with_input_types(&[ArrayType::scalar(DataType::Boolean)])
                    .with_output_types(&output_types)
                    .without_boundary_shardings(),
            )
            .unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<i1>) -> tensor<4xf32> {
                    %0 = "stablehlo.if"(%arg1) ({
                      stablehlo.return %arg0 : tensor<4xf32>
                    }, {
                      stablehlo.return %arg0 : tensor<4xf32>
                    }) : (tensor<i1>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_erases_static_zero_space_boundary() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatProgram<Session> = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main() {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    return
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_preserves_effects_that_produce_an_erased_output() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(zero_type.clone().into());
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("zero"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program: FlatProgram<Session> = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {"
                module {
                  func.func @main() {
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    %1 = stablehlo.after_all  : !stablehlo.token
                    %2 = stablehlo.custom_call @ryft.print(%0, %1) {api_version = 4 : i32, \
                        backend_config = {label = \"zero\", logical_type = \"zero\"}, has_side_effect = true, \
                        operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], \
                        result_layouts = [dense<> : tensor<0xindex>]} : (tensor<3xi1>, !stablehlo.token) \
                        -> !stablehlo.token
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_lowered_module_new_preserves_dynamic_zero_space_shape_carrier() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("zero", Some(3))]));
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatProgram<Session> = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<2xi1>, %arg1: tensor<i32>) -> (tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg1, dim = 0 : (tensor<2xi1>, tensor<i32>) -> tensor<?xi1, #stablehlo.bounds<2>>
                    %1 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xi1, #stablehlo.bounds<2>>) -> tensor<i32>
                    %2 = stablehlo.convert %1 : (tensor<i32>) -> tensor<i64>
                    return %0, %2 : tensor<?xi1, #stablehlo.bounds<2>>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_isolates_bounded_dynamic_outputs_behind_one_barrier() {
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("count", Some(4))]));
        let static_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let dynamic = builder.add_input(dynamic_type.clone().into());
        let r#static = builder.add_input(static_type.clone().into());
        let program: FlatProgram<Session> =
            builder.build(vec![dynamic, r#static, dynamic], vec![Placeholder; 2], vec![Placeholder; 3]).unwrap();

        let stablehlo = program.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap();

        // Both dynamic outputs pass through the barrier, while the static output and the hidden extents bypass it.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<i32>) -> (tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>) {
                    %0 = stablehlo.set_dimension_size %arg0, %arg2, dim = 0 : (tensor<3xf32>, tensor<i32>) -> tensor<?xf32, #stablehlo.bounds<3>>
                    %1:2 = stablehlo.optimization_barrier %0, %0 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<?xf32, #stablehlo.bounds<3>>
                    %2 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %3 = stablehlo.convert %2 : (tensor<i32>) -> tensor<i64>
                    %4 = stablehlo.get_dimension_size %0, dim = 0 : (tensor<?xf32, #stablehlo.bounds<3>>) -> tensor<i32>
                    %5 = stablehlo.convert %4 : (tensor<i32>) -> tensor<i64>
                    return %1#0, %arg1, %1#1, %3, %5 : tensor<?xf32, #stablehlo.bounds<3>>, tensor<2xf32>, tensor<?xf32, #stablehlo.bounds<3>>, tensor<i64>, tensor<i64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_declares_the_mesh_of_completed_device_reductions() {
        // A device target that completes a pending reduction over a mesh axis of size one constrains its output over
        // the mesh of its type, which neither the signature nor a payload names, so the module declares the mesh
        // because the lowered constraint refers to it.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let input_type = test_vector_type(4).with_sharding(unreduced).unwrap();
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(PlaceOperation::new(Device::new(0, 0), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = program
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            program
                .to_rendered_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .with_input_types(&vec![input_type])
                        .with_output_types(&output_types)
                        .without_boundary_shardings()
                )
                .unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_lowered_module_new_omits_shardy_for_targets_without_the_dialect() {
        // A device target that completes a pending reduction over a mesh axis of size one constrains its output, and
        // the signature shardings annotate the entry function. A target whose plugin lacks the Shardy dialect gets
        // neither, nor the mesh that both refer to.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let input_type = test_vector_type(4).with_sharding(unreduced.clone()).unwrap();
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(input_type.clone().into());
        let output = builder
            .add_instruction(PlaceOperation::new(Device::new(0, 0), None), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let output_types = program
            .output_types()
            .iter()
            .map(|r#type| <&ArrayType>::try_from(r#type).cloned().unwrap())
            .collect::<Vec<_>>();
        let lower = |supports_shardy| {
            program
                .to_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .without_pruning()
                        .with_input_types(&vec![input_type.clone()])
                        .with_output_types(&output_types)
                        .without_boundary_shardings()
                        .with_exact_input_shardings(vec![Some(unreduced.clone())])
                        .with_exact_output_shardings(vec![Some(replicated.clone())])
                        .with_target_platform(Some("cpu"))
                        .with_shardy_support(supports_shardy)
                        .with_reference_bindings(Vec::new(), Vec::new(), Vec::new()),
                )
                .unwrap()
        };
        let lowered = lower(true);
        assert_eq!(lowered.mesh(), Some(&mesh));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=1]>
                  func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x"}>}) -> (tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"x"}>}) {
                    %0 = sdy.sharding_constraint %arg0 <@mesh, [{}], replicated={"x"}> : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
        let lowered = lower(false);
        assert_eq!(lowered.mesh(), Some(&mesh));
        assert_eq!(
            lowered.into_stable_hlo(),
            indoc! {"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    return %arg0 : tensor<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_remove_shardy_operations() {
        let context = MlirContext::new();
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        context.load_dialect(DialectHandle::stable_hlo().unwrap()).unwrap();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();

        // A manual computation has no equivalent without the dialect.
        let body = unproject_plain_program(neg_branch(test_vector_type(4)));
        let mut module = context.parse_module(lower_replicated_shard_map_body(body, 1).unwrap()).unwrap();
        assert!(matches!(
            remove_shardy_operations(&mut module),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == "sdy.manual_computation"
                    && reason == "the target plugin cannot compile with Shardy",
        ));
    }

    #[test]
    fn test_lowered_module_new_rejects_unresolved_reference_state() {
        assert_eq!(token_threaded_effects(EffectClasses::single(EffectClass::OrderedState)).next(), None);

        // The reference scan over the whole artifact catches the reference atoms and intrinsic semantics before token
        // threading, so two representative reference operations are enough to pin the diagnostic.
        let array_type = ArrayType::scalar(DataType::F32);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(Type::Array(array_type.clone()));
        let reference = builder
            .add_instruction(Operation::ReferenceNew(ReferenceNewOperation::new()), Vec::new(), vec![input], None)
            .unwrap()[0];
        builder
            .add_instruction(
                Operation::ReferenceWrite(ReferenceWriteOperation::new()),
                Vec::new(),
                vec![reference, input],
                None,
            )
            .unwrap();
        let output = builder
            .add_instruction(
                Operation::ReferenceFreeze(ReferenceFreezeOperation::new()),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The reference scan runs before the generic state scan at the module entry, so a program whose state comes
        // from reference operations receives the dedicated reference diagnostic. The state error remains behind it for
        // future state that does not come from references, and `token_threaded_effects` above pins that `OrderedState`
        // has no token slot either way.
        assert!(matches!(
            program.to_mlir("main", &ProgramLoweringOptions::new()),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
        ));
    }

    #[test]
    fn test_lowered_module_new_rejects_preserved_reference_entries() {
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference = builder.add_input(Type::Reference(ReferenceType::new(scalar_type.clone())));
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let contract = KernelBoundaryContract::new(vec![Some(KernelParameterAccess::ReadOnly)]);
        validate_kernel_body(program.entry_region_ref(), &contract).unwrap();
        assert!(matches!(
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .with_input_types(&vec![scalar_type.clone()])
                    .with_output_types(&vec![scalar_type]),
            ),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct })) if construct == "program",
        ));
    }

    #[test]
    fn test_lowered_module_new_with_reference_states() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let state_type = ArrayType::new_static(DataType::F32, [4]);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let _zero = builder.add_input(Type::Array(zero_type.clone()));
        let mutated = builder.add_input(Type::Array(state_type.clone()));
        let _read_only = builder.add_input(Type::Array(state_type.clone()));
        let program: FlatProgram<Session> =
            builder.build(vec![mutated], vec![Placeholder; 3], vec![Placeholder]).unwrap();
        let input_types = vec![zero_type.clone(), state_type.clone(), state_type.clone()];
        let output_types = vec![state_type.clone()];
        let input_shardings = vec![
            Sharding::replicated(mesh.clone(), 1),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        ];
        let output_shardings = vec![Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()];
        let states = [
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 1 }, Some(0)),
            ExternalReferenceBinding::new(ReferenceSource::Input { index: 2 }, None),
        ];

        let lowered = program
            .to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings()
                    .with_exact_input_shardings(input_shardings.iter().cloned().map(Some).collect())
                    .with_exact_output_shardings(output_shardings.iter().cloned().map(Some).collect())
                    .with_reference_bindings(states.to_vec(), Vec::new(), Vec::new()),
            )
            .unwrap();
        assert_eq!(lowered.signature().input_mapping(), &[None, Some(0), Some(1)]);
        assert_eq!(lowered.signature().output_mapping(), &[Some(0)]);
        let expected_signature = concat!(
            "  func.func @main(",
            "%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>, ",
            "tf.aliasing_output = 0 : i64}, ",
            "%arg1: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) -> ",
            "(tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}) {",
        );
        assert_eq!(
            lowered.stable_hlo(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                @SIGNATURE@
                    %c = stablehlo.constant dense<false> : tensor<i1>
                    %0 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i1>) -> tensor<3xi1>
                    return %arg0 : tensor<4xf32>
                  }
                }
            "#}
            .replace("@SIGNATURE@", expected_signature),
        );

        let invalid_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, None);
        assert!(matches!(
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings()
                    .with_exact_input_shardings(input_shardings.iter().cloned().map(Some).collect())
                    .with_exact_output_shardings(output_shardings.iter().cloned().map(Some).collect())
                    .with_reference_bindings((std::slice::from_ref(&invalid_state)).to_vec(), Vec::new(), Vec::new()),
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "logical state input 0 is erased from the executable boundary",
        ));

        let asymmetric_sharding_message =
            "reference-state aliases require both input and output sharding metadata or neither";
        for (input_shardings, output_shardings) in
            [(Some(input_shardings.as_slice()), None), (None, Some(output_shardings.as_slice()))]
        {
            assert!(matches!(
                program.to_mlir(
                    "main",
                    &ProgramLoweringOptions::new()
                        .without_pruning()
                        .with_input_types(&input_types)
                        .with_output_types(&output_types)
                        .with_exact_input_shardings(value_shardings(input_shardings, input_types.len()))
                        .with_exact_output_shardings(value_shardings(output_shardings, output_types.len()))
                        .with_reference_bindings(states.to_vec(), Vec::new(), Vec::new()),
                ),
                Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                    if message == asymmetric_sharding_message,
            ));
        }

        let mismatched_output_shardings = vec![Sharding::replicated(mesh, 1)];
        assert!(matches!(
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings()
                    .with_exact_input_shardings(input_shardings.iter().cloned().map(Some).collect())
                    .with_exact_output_shardings(mismatched_output_shardings.iter().cloned().map(Some).collect())
                    .with_reference_bindings(states.to_vec(), Vec::new(), Vec::new()),
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "state input 1 and output 0 must use the same sharding",
        ));
    }

    #[test]
    fn test_lowered_module_new_with_reference_states_rejects_bounded_dynamic_aliases() {
        let extent = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let state_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh, 1);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let state = builder.add_input(Type::Array(state_type.clone()));
        let program: FlatProgram<Session> = builder.build(vec![state], vec![Placeholder], vec![Placeholder]).unwrap();
        let reference_state = ExternalReferenceBinding::new(ReferenceSource::Input { index: 0 }, Some(0));

        assert!(matches!(
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&vec![state_type.clone()])
                    .with_output_types(&vec![state_type])
                    .without_boundary_shardings()
                    .with_exact_input_shardings(vec![Some(sharding.clone())])
                    .with_exact_output_shardings(vec![Some(sharding.clone())])
                    .with_reference_bindings((std::slice::from_ref(&reference_state)).to_vec(), Vec::new(), Vec::new()),
            ),
            Err(Error::Lowering(LoweringError::InvalidReferenceStateAbi { message }))
                if message == "state input 0 and output 0 must be static because bounded-dynamic mutation alias \
                               compatibility is unsupported",
        ));
    }

    #[test]
    fn test_lowered_module_new_with_reference_states_aliases_donated_inputs() {
        let vector_type = ArrayType::new_static(DataType::F32, [4]);
        let pair_type = ArrayType::new_static(DataType::F32, [2]);
        let mut builder = ArrayIrProgramBuilder::<Constant, Operation>::new();
        let first = builder.add_input(Type::Array(vector_type.clone()));
        let second = builder.add_input(Type::Array(vector_type.clone()));
        let third = builder.add_input(Type::Array(pair_type.clone()));
        let program: FlatProgram<Session> =
            builder.build(vec![second, first, third], vec![Placeholder; 3], vec![Placeholder; 3]).unwrap();
        let input_types = vec![vector_type.clone(), vector_type.clone(), pair_type.clone()];
        let output_types = vec![vector_type.clone(), vector_type.clone(), pair_type.clone()];
        let lower_with = |input_shardings: Option<&[Sharding]>,
                          output_shardings: Option<&[Sharding]>,
                          donated_inputs: &[bool],
                          open_outputs: &[bool]| {
            program.to_mlir(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .with_exact_input_shardings(value_shardings(input_shardings, input_types.len()))
                    .with_exact_output_shardings(value_shardings(output_shardings, output_types.len()))
                    .with_reference_bindings(Vec::new(), (donated_inputs).to_vec(), Vec::new())
                    .with_open_outputs((open_outputs).to_vec()),
            )
        };
        let lower = |shardings: Option<&[Sharding]>, donated_inputs: &[bool], open_outputs: &[bool]| {
            lower_with(shardings, shardings, donated_inputs, open_outputs)
        };
        let signature = |lowered: MlirProgram| {
            let line = lowered.stable_hlo().lines().find(|line| line.contains("func.func @main")).unwrap();
            line.trim().to_string()
        };
        // Pairs of the index of each aliased input and the index of its output.
        let aliases = |lowered: MlirProgram| {
            signature(lowered)
                .split("%arg")
                .skip(1)
                .enumerate()
                .filter_map(|(index, argument)| {
                    let (_, alias) = argument.split_once("tf.aliasing_output = ")?;
                    Some((index, alias.split(' ').next().unwrap().parse::<usize>().unwrap()))
                })
                .collect::<Vec<_>>()
        };

        // Each donated input is paired with the first unpaired output of its type, and other inputs with none.
        assert_eq!(
            signature(lower(None, &[true, true, false], &[]).unwrap()),
            concat!(
                "func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i64}, ",
                "%arg1: tensor<4xf32> {tf.aliasing_output = 1 : i64}, %arg2: tensor<2xf32>) -> ",
                "(tensor<4xf32>, tensor<4xf32>, tensor<2xf32>) {",
            ),
        );
        assert_eq!(
            signature(lower(None, &[false, true, true], &[]).unwrap()),
            concat!(
                "func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32> {tf.aliasing_output = 0 : i64}, ",
                "%arg2: tensor<2xf32> {tf.aliasing_output = 2 : i64}) -> ",
                "(tensor<4xf32>, tensor<4xf32>, tensor<2xf32>) {",
            ),
        );
        assert!(!signature(lower(None, &[], &[]).unwrap()).contains("tf.aliasing_output"));

        // An open output is only paired on a single device, where the compiler cannot place it differently.
        let shardings = |device_count: usize| {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", device_count, MeshAxisType::Auto).unwrap()]).unwrap();
            vec![Sharding::replicated(mesh, 1); 3]
        };
        let open_outputs = [true, false, false];
        let single_device = signature(lower(Some(&shardings(1)), &[true, false, false], &open_outputs).unwrap());
        assert!(single_device.contains("tf.aliasing_output = 0 : i64"));
        let two_devices = signature(lower(Some(&shardings(2)), &[true, false, false], &open_outputs).unwrap());
        assert!(two_devices.contains("tf.aliasing_output = 1 : i64"));

        // An input that finds no output of its own sharding is paired with an output under which every device holds
        // shards of the same type, such as one that completes its pending sum or splits it along another axis of the
        // same size, and only after every donated input was paired with an output of its own sharding where one is
        // free. Before, only an output of the identical sharding was paired, so the first two cases had no alias.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let split = |axes: &[&str]| {
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(axes.iter().copied())]).unwrap()
        };
        let replicated = Sharding::replicated(mesh.clone(), 1);
        let pending = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let cases = [
            ("completed sum", [pending, replicated.clone()], [replicated.clone(), replicated.clone()], [true, false]),
            ("other axis", [split(&["x"]), replicated.clone()], [replicated.clone(), split(&["y"])], [true, false]),
            (
                "other extents",
                [split(&["x"]), replicated.clone()],
                [replicated.clone(), split(&["x", "y"])],
                [true, false],
            ),
            ("own sharding first", [split(&["x"]), split(&["x"])], [split(&["y"]), split(&["x"])], [true, true]),
        ];
        let outcomes = cases
            .into_iter()
            .map(|(name, [first_input, second_input], [first_output, second_output], [first, second])| {
                let input_shardings = [first_input, second_input, replicated.clone()];
                let output_shardings = [first_output, second_output, replicated.clone()];
                let donated_inputs = [first, second, false];
                let lowered = lower_with(Some(&input_shardings), Some(&output_shardings), &donated_inputs, &[]);
                (name, aliases(lowered.unwrap()))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            outcomes,
            [
                ("completed sum", vec![(0, 0)]),
                ("other axis", vec![(0, 1)]),
                ("other extents", vec![]),
                ("own sharding first", vec![(0, 1), (1, 0)]),
            ],
        );

        assert!(matches!(
            lower(None, &[true], &[]),
            Err(Error::Execution(ExecutionError::DonationFlagCountMismatch { expected_count: 3, actual_count: 1 })),
        ));
    }

    #[test]
    fn test_entry_alias_attribute_executes_in_place_on_cpu() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[MlirSize::Static(4)], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let function_block = context.block(&[(tensor_type, location)]);
                {
                    let input = function_block.argument(0).unwrap();
                    let mut function_block_ref = function_block.as_ref();
                    let doubled =
                        function_block_ref.append_operation(stable_hlo::add(input, input, location).unwrap()).unwrap();
                    function_block_ref
                        .append_operation(func::r#return(&[doubled.result(0).unwrap()], location).unwrap())
                        .unwrap();
                }
                let mut function_region = context.region();
                function_region.append_block(function_block).unwrap();
                func::func(
                    "main",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([
                                ("mhlo.sharding".into(), context.string_attribute("{replicated}").as_ref()),
                                (
                                    "tf.aliasing_output".into(),
                                    context.integer_attribute(context.signless_integer_type(64), 0).as_ref(),
                                ),
                            ])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    function_region,
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        let module = module.to_string();
        assert_eq!(
            module,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32> {mhlo.sharding = "{replicated}", tf.aliasing_output = 0 : i64}) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );

        let client = cpu_client(1);
        let executable = client
            .compile(&PjrtProgram::Mlir { bytecode: module.into_bytes() }, &test_compilation_options(client, None))
            .unwrap();
        let device = executable.addressable_devices().unwrap().remove(0);
        let input = client
            .buffer(values_to_bytes::<f32>(&[1.0, 2.0, 3.0, 4.0]).as_slice(), BufferType::F32, [4], None, device, None)
            .unwrap();
        input.ready().unwrap().r#await().unwrap();
        // The input is ready and remains alive in `inputs` until execution completes, so comparing its opaque address
        // with the synchronized output address does not dereference either device pointer.
        let input_pointer = unsafe { input.unsafe_pointer().unwrap() };
        let inputs = [ExecutionInput { buffer: Arc::new(input), donatable: true }];
        let execution = executable
            .execute(
                vec![ExecutionDeviceInputs { inputs: &inputs, ..Default::default() }],
                Vec::new(),
                0,
                None,
                Some(file!()),
                None,
                None,
                &[],
            )
            .unwrap();
        let mut device_outputs = execution.block_until_ready().unwrap().remove(0);
        let output = device_outputs.outputs.remove(0);
        let output_pointer = unsafe { output.unsafe_pointer().unwrap() };
        // `tf.aliasing_output` is a hint rather than a guarantee, because the copy insertion of XLA may materialize a
        // fresh output. This pointer equality is therefore a deliberate canary pinned to the CPU plugin version that
        // the repository pins. If a plugin upgrade breaks it while the numeric assertion below still passes,
        // re-evaluate whether the alias hint is still honored rather than assuming a correctness bug.
        assert_eq!(output_pointer, input_pointer);
        let output = output.copy_to_host(None).unwrap().r#await().unwrap();
        assert_eq!(values_from_bytes::<f32>(output.as_slice()), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_to_rendered_mlir_for_plain_program() {
        let (_, compiled): (Array, Program<Array, ArrayOperation<Array>, (Array, Array), Array>) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |inputs| Ok(scalar_bilinear_sin(inputs)),
                (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
            )
            .unwrap();

        let stablehlo = to_rendered_mlir_for_plain_program(&compiled, "main").unwrap();
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.multiply %arg0, %arg1 : tensor<f64>
                    %1 = stablehlo.sine %arg0 : tensor<f64>
                    %2 = stablehlo.add %0, %1 : tensor<f64>
                    return %2 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_rendered_mlir_for_plain_program_lowers_gradients() {
        let (_, compiled): (Array, Program<Array, ArrayOperation<Array>, Array, Array>) = TEST_ARRAY_DOMAIN
            .interpret_and_trace(
                |x| {
                    let context = x.context().clone();
                    Ok(context.differentiate_at(x).gradient(scalar_quartic_plus_sin).unwrap())
                },
                Array::scalar(2.0).unwrap(),
            )
            .unwrap();

        let stablehlo = to_rendered_mlir_for_plain_program(&compiled, "main").unwrap();

        // The gradient computes one cosine for the derivative of `sin(x)`, expands the `x^4` chain rule into
        // multiplications, accumulates the cotangent contributions with additions, and contains no sine.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>) -> tensor<f64> {
                    %0 = stablehlo.cosine %arg0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %1 = stablehlo.multiply %0, %cst : tensor<f64>
                    %2 = stablehlo.multiply %arg0, %arg0 : tensor<f64>
                    %3 = stablehlo.multiply %2, %arg0 : tensor<f64>
                    %4 = stablehlo.multiply %3, %cst : tensor<f64>
                    %5 = stablehlo.add %1, %4 : tensor<f64>
                    %6 = stablehlo.multiply %arg0, %cst : tensor<f64>
                    %7 = stablehlo.multiply %2, %6 : tensor<f64>
                    %8 = stablehlo.add %5, %7 : tensor<f64>
                    %9 = stablehlo.multiply %arg0, %6 : tensor<f64>
                    %10 = stablehlo.multiply %arg0, %9 : tensor<f64>
                    %11 = stablehlo.add %8, %10 : tensor<f64>
                    %12 = stablehlo.multiply %arg0, %9 : tensor<f64>
                    %13 = stablehlo.add %11, %12 : tensor<f64>
                    return %13 : tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_rendered_mlir_for_plain_program_lowers_standalone_pullbacks() {
        // Standalone pullback over the primal operation family, produced by the reverse path that is aware of
        // partitions. The reverse program of `f(x, y) = x * y + sin(x)` consumes `[output_cotangent ++ residuals]`,
        // where the residuals are the factors `cos(x)`, `x`, and `y` at the linearization point. It lowers the backward
        // pass, weighted by those residuals, as `stablehlo.multiply`s of the cotangent against the residual inputs
        // rather than baking the primal point in as constants. This is the analogue of the standalone `vjp_fn` of JAX,
        // with the residuals threaded as explicit inputs.
        let (_, pullback): (Array, _) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(
                |inputs, ()| Ok(scalar_bilinear_sin(inputs)),
                (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
                (),
            )
            .unwrap();
        let (pullback, _residuals) = pullback.into_transposed_parts().unwrap();
        let pullback = pullback.into_program();

        let stablehlo = to_rendered_mlir_for_plain_program(&pullback, "main").unwrap();
        // The pullback takes the output cotangent plus three residual inputs and returns one cotangent output for each
        // of `x` and `y`. It multiplies the cotangent by the residual inputs rather than baking the primal point in as
        // constants, so the lowering consists of multiplications and additions with no `stablehlo.constant`.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>, %arg2: tensor<f64>, %arg3: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                    %0 = stablehlo.multiply %arg3, %arg0 : tensor<f64>
                    %1 = stablehlo.multiply %arg2, %arg0 : tensor<f64>
                    %2 = stablehlo.multiply %arg1, %arg0 : tensor<f64>
                    %3 = stablehlo.add %0, %2 : tensor<f64>
                    return %3, %1 : tensor<f64>, tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_to_rendered_mlir_for_plain_program_lowers_staged_gradients() {
        // The gradient of `f` staged for JIT compilation stays symbolic, like `jit(grad(f))` in JAX. It uses the traced
        // gradient path, which traces through the VJP and its pullback.
        let (_, compiled): ((Array, Array), Program<Array, ArrayOperation<Array>, (Array, Array), (Array, Array)>) =
            TEST_ARRAY_DOMAIN
                .interpret_and_trace(
                    |inputs| {
                        let context = inputs.0.context().clone();
                        Ok(context.differentiate_at(inputs).gradient(scalar_bilinear_sin).unwrap())
                    },
                    (Array::scalar(2.0).unwrap(), Array::scalar(3.0).unwrap()),
                )
                .unwrap();

        let stablehlo = to_rendered_mlir_for_plain_program(&compiled, "main").unwrap();
        // `cos(x)` is computed symbolically from `%arg0` rather than baked in as a constant, both inputs are
        // referenced, and the gradient contains no sine, because the derivative of sine is cosine.
        assert_eq!(
            stablehlo,
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<f64>, %arg1: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
                    %0 = stablehlo.cosine %arg0 : tensor<f64>
                    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f64>
                    %1 = stablehlo.multiply %0, %cst : tensor<f64>
                    %2 = stablehlo.multiply %arg1, %cst : tensor<f64>
                    %3 = stablehlo.add %1, %2 : tensor<f64>
                    %4 = stablehlo.multiply %arg0, %cst : tensor<f64>
                    return %3, %4 : tensor<f64>, tensor<f64>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_broadcast_changes_explicit_sharding() {
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let replicated = test_vector_type(4).with_sharding(Sharding::replicated(explicit_mesh.clone(), 1)).unwrap();
        let explicitly_sharded = test_vector_type(4)
            .with_sharding(Sharding::new(explicit_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&test_vector_type(4), &replicated, &[0]));
        assert!(broadcast_changes_explicit_sharding(&replicated, &explicitly_sharded, &[0]));
        assert!(!broadcast_changes_explicit_sharding(&explicitly_sharded, &explicitly_sharded, &[0]));

        let first_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let second_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let first_mesh_replicated = test_vector_type(4).with_sharding(Sharding::replicated(first_mesh, 1)).unwrap();
        let second_mesh_replicated = test_vector_type(4).with_sharding(Sharding::replicated(second_mesh, 1)).unwrap();
        assert!(broadcast_changes_explicit_sharding(&first_mesh_replicated, &second_mesh_replicated, &[0]));

        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = test_vector_type(4).with_sharding(Sharding::replicated(manual_mesh.clone(), 1)).unwrap();
        let manually_sharded = test_vector_type(4)
            .with_sharding(
                Sharding::new(manual_mesh, vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert!(!broadcast_changes_explicit_sharding(&replicated, &manually_sharded, &[0]));

        // Expanding a partitioned singleton and changing explicit reduction state both require constraints even when
        // the ranked sharding dimensions themselves remain unchanged.
        let explicit_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let singleton = test_vector_type(1)
            .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let expanded = test_vector_type(4)
            .with_sharding(Sharding::new(explicit_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&singleton, &expanded, &[0]));

        let unreduced = test_vector_type(4)
            .with_sharding(Sharding::replicated(explicit_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let reduced = test_vector_type(4)
            .with_sharding(Sharding::replicated(explicit_mesh, 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert!(broadcast_changes_explicit_sharding(&unreduced, &reduced, &[0]));
    }

    #[test]
    fn test_cast_changes_explicit_placement() {
        let explicit_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let resized_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let extended_auto_mesh = LogicalMesh::new(vec![
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("b", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let unsharded = ArrayType::new_static(DataType::F64, [4, 4]);
        let placed = |mesh: &LogicalMesh, first: ShardingDimension| {
            unsharded
                .clone()
                .with_sharding(Sharding::new(mesh.clone(), vec![first, ShardingDimension::replicated()]).unwrap())
                .unwrap()
        };
        let sharded = |axes: &[&str]| ShardingDimension::sharded(axes.iter().copied());
        let replicated = ShardingDimension::replicated;

        // An output without a sharding stays untracked, whatever the input placement.
        assert!(!cast_changes_explicit_placement(&unsharded, &unsharded));
        assert!(!cast_changes_explicit_placement(&placed(&explicit_mesh, sharded(&["x"])), &unsharded));

        // An unsharded input acquires a tracked placement only when the output mesh has explicit axes. Types record no
        // placement over auto mesh axes, so the placements below are replicated over them.
        assert!(cast_changes_explicit_placement(&unsharded, &placed(&explicit_mesh, replicated())));
        assert!(!cast_changes_explicit_placement(&unsharded, &placed(&auto_mesh, replicated())));

        // Carrying a mesh forward requests a redistribution only when the placement over explicit axes conflicts.
        assert!(!cast_changes_explicit_placement(
            &placed(&explicit_mesh, sharded(&["x"])),
            &placed(&explicit_mesh, sharded(&["x"])),
        ));
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, sharded(&["x"])),
            &placed(&explicit_mesh, sharded(&["y"])),
        ));
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, replicated()),
            &placed(&explicit_mesh, sharded(&["x"])),
        ));

        // A mesh change requests one whenever either mesh has explicit axes, but never between auto-only meshes.
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, replicated()),
            &placed(&resized_mesh, replicated()),
        ));
        assert!(cast_changes_explicit_placement(
            &placed(&explicit_mesh, replicated()),
            &placed(&auto_mesh, replicated()),
        ));
        assert!(!cast_changes_explicit_placement(
            &placed(&auto_mesh, replicated()),
            &placed(&extended_auto_mesh, replicated()),
        ));
    }

    #[test]
    fn test_normalize_function_name() {
        assert_eq!(normalize_function_name(" main ").unwrap(), "main");
        assert_eq!(normalize_function_name(" @main ").unwrap(), "main");
        assert!(matches!(
            normalize_function_name(" @ "),
            Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name })) if function_name.is_empty(),
        ));
        assert!(matches!(
            normalize_function_name("main function"),
            Err(Error::Lowering(LoweringError::InvalidFunctionName { function_name })) if function_name == "main function",
        ));
    }
}
