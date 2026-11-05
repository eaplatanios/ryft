use std::collections::{HashMap, HashSet};

use ryft_core::{
    ArrayIrType, ArrayType, CompositeKey, Parameterized, ProgramError, RegionId, RegionRef,
    RematerializationOptimizationBarrier, ScaledDotOperation, Typed, TypedProgram,
};
use ryft_mlir::dialects::func;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Operation as MlirOperation, Region, SymbolVisibility, Type,
    TypeAndAttributes, Value as MlirValue, ValueAndAttributes, ValueRef,
};

use crate::Error;
use crate::experimental::lowering::regions::lower_nested_program_inline;
use crate::experimental::lowering::types::{lower_array_ir_type, lower_tensor_type};
use crate::experimental::lowering::{CollectiveLoweringState, EffectTokens, LowerableOperationFamily};
use crate::experimental::ops::{Constant, FlatProgram, Program};
use crate::operations::quantization::{
    scaled_dot_composition_key, scaled_dot_composition_types, trace_scaled_dot_composition,
};

/// Returns backend-local input types for a named decomposition after boundary validation and differentiation.
///
/// Private decomposition functions contain local array computations and have no manual-axis binders. Manual variation
/// has already determined the enclosing program's explicit transitions and adjoints, so it must not trigger new
/// transitions when expanding these functions. Retain mesh placement and reduction state, and keep the original types
/// in the named-composition key so the external operation contract remains unchanged.
pub(crate) fn named_composition_input_types(input_types: &[ArrayType]) -> Result<Vec<ArrayIrType>, Error> {
    input_types
        .iter()
        .map(|input| {
            let sharding = input
                .sharding()
                .map(|sharding| sharding.clone().with_varying_manual_axes(Vec::<String>::new()))
                .transpose()?;
            Ok(ArrayIrType::from(input.clone().with_sharding(sharding)?))
        })
        .collect()
}

/// Calls one emitted typed decomposition function at its registered array boundary.
pub(crate) fn lower_decomposition_call<'b, 'c: 'b, 't: 'c>(
    function: &NamedCompositionFunction,
    input_values: &[ValueRef<'b, 'c, 't>],
    output_types: &[ArrayType],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let result_types = output_types
        .iter()
        .map(|r#type| lower_tensor_type(r#type, context, location).map(|r#type| r#type.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;
    let call = block.append_operation(func::call(
        function.symbol.as_str(),
        func::CallProperties {
            arguments: input_values
                .iter()
                .map(|value| ValueAndAttributes { value: *value, attributes: None })
                .collect(),
            results: result_types
                .iter()
                .map(|r#type| TypeAndAttributes { r#type: *r#type, attributes: None })
                .collect(),
            no_inline: false,
        },
        location,
    )?)?;
    Ok((0..output_types.len())
        .map(|index| call.result(index).expect("func.call should return one result per output").as_ref())
        .collect())
}

/// Identity of one generated named-composition decomposition at a concrete typed boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NamedCompositionKey {
    /// Semantic identity of the composition, whose name also names the emitted private function.
    pub(crate) composite: CompositeKey,

    /// Logical operand types of the composition boundary.
    pub(crate) input_types: Vec<ArrayType>,

    /// Logical result types of the composition boundary.
    pub(crate) output_types: Vec<ArrayType>,
}

/// Private decomposition function emitted for one [`NamedCompositionKey`].
pub(crate) struct NamedCompositionFunction {
    /// Module-unique decomposition symbol.
    pub(crate) symbol: String,

    /// Canonical typed decomposition program.
    pub(crate) program: FlatProgram,
}

/// Module-owned registry of canonical named compositions and their private decomposition functions.
#[derive(Default)]
pub(crate) struct NamedCompositionFunctionMap {
    /// Functions indexed by semantic operation and typed boundary.
    pub(crate) functions: HashMap<NamedCompositionKey, NamedCompositionFunction>,

    /// Stable first-occurrence order used for deterministic function emission.
    pub(crate) order: Vec<NamedCompositionKey>,
}

impl NamedCompositionFunctionMap {
    /// Registers one typed decomposition lazily, sharing an existing private function when the complete semantic key
    /// is already present. The registry is independent of the operation family that produced `key` and `program`;
    /// closed backend operation enums only decide when to call this seam.
    fn register<F: FnOnce() -> Result<FlatProgram, Error>>(
        &mut self,
        key: NamedCompositionKey,
        program: F,
    ) -> Result<(), Error> {
        if self.functions.contains_key(&key) {
            return Ok(());
        }
        let name = key.composite.name();
        let symbol = match self.order.iter().filter(|existing| existing.composite.name() == name).count() {
            0 => name.to_string(),
            count => format!("{name}_{count}"),
        };
        self.functions.insert(key.clone(), NamedCompositionFunction { symbol, program: program()? });
        self.order.push(key);
        Ok(())
    }

    /// Returns the registered decomposition for `operation` at the provided typed boundary.
    pub(crate) fn get(
        &self,
        operation: &ScaledDotOperation,
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Option<&NamedCompositionFunction>, Error> {
        Ok(self.functions.get(&scaled_dot_composition_key(operation, input_types, output_types)?))
    }
}

/// Collects every named semantic composition required by `program` and traces each distinct typed decomposition once.
pub(crate) fn collect_named_composition_functions<Input, Output, O: LowerableOperationFamily>(
    program: &Program<Input, Output, O>,
    target_platform: Option<&str>,
) -> Result<NamedCompositionFunctionMap, Error>
where
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    fn walk<O: LowerableOperationFamily>(
        region: RegionRef<'_, Constant, O>,
        map: &mut NamedCompositionFunctionMap,
        visited: &mut HashSet<RegionId>,
        target_platform: Option<&str>,
    ) -> Result<(), Error> {
        if !visited.insert(region.id()) {
            return Ok(());
        }
        for instruction in region.instructions() {
            for nested in instruction.regions() {
                walk(RegionRef::new(region.arena(), *nested)?, map, visited, target_platform)?;
            }
            // Payloads are recognized through their projections, so any family that embeds them gets their
            // compositions.
            let operation = instruction.operation();
            let scaled_dot = operation.projected_payload::<ScaledDotOperation>();
            let Some(operation) = scaled_dot else {
                continue;
            };
            let input_types = instruction
                .inputs()
                .iter()
                .map(|input| {
                    let r#type = region.atoms()[input.index()].r#type();
                    <&ArrayType>::try_from(r#type.as_ref()).cloned().map_err(ProgramError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output_types = instruction
                .outputs()
                .iter()
                .map(|output| {
                    let r#type = region.atoms()[output.index()].r#type();
                    <&ArrayType>::try_from(r#type.as_ref()).cloned().map_err(ProgramError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            // Calls that lower to the block-scaled-dot custom call need no composition.
            if let Some((composition_input_types, composition_output_types)) = scaled_dot_composition_types(
                operation,
                input_types.as_slice(),
                output_types.as_slice(),
                target_platform,
            )? {
                map.register(
                    scaled_dot_composition_key(
                        operation,
                        composition_input_types.as_slice(),
                        composition_output_types.as_slice(),
                    )?,
                    || trace_scaled_dot_composition(operation, composition_input_types.as_slice()),
                )?;
            }
        }
        Ok(())
    }

    let mut map = NamedCompositionFunctionMap::default();
    walk(program.entry_region_ref(), &mut map, &mut HashSet::new(), target_platform)?;
    Ok(map)
}

/// Emits one private named-composition decomposition function.
pub(crate) fn emit_named_composition_function<'b, 'c: 'b, 't: 'c>(
    module_block: &mut BlockRef<'b, 'c, 't>,
    function: &NamedCompositionFunction,
    collective_state: &CollectiveLoweringState,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), Error> {
    let input_types = function.program.input_types();
    let output_types = function.program.output_types();
    let argument_tensor_types = input_types
        .iter()
        .map(|r#type| lower_array_ir_type(r#type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let result_tensor_types = output_types
        .iter()
        .map(|r#type| lower_array_ir_type(r#type, context, location))
        .collect::<Result<Vec<_>, _>>()?;
    let function_block = context.block(
        argument_tensor_types
            .iter()
            .map(|tensor_type| (*tensor_type, location))
            .collect::<Vec<_>>()
            .as_slice(),
    );
    {
        let mut function_block_ref = function_block.as_ref();
        let input_values = (0..input_types.len())
            .map(|index| function_block.argument(index).unwrap().as_ref())
            .collect::<Vec<_>>();
        let mut effect_tokens = EffectTokens::default();
        let outputs = lower_nested_program_inline(
            &function.program,
            input_values.as_slice(),
            &mut function_block_ref,
            context,
            location,
            &[],
            &RematerializationOptimizationBarrier::None,
            // Decompositions are traced independently of the lowered program and contain no `jit_call` sharing.
            None,
            collective_state,
            &mut effect_tokens,
        )?;
        function_block_ref.append_operation(func::r#return(outputs.as_slice(), location)?)?;
    }
    let mut function_region = context.region();
    function_region.append_block(function_block)?;
    module_block.append_operation(func::func(
        function.symbol.as_str(),
        func::FuncAttributes {
            arguments: argument_tensor_types
                .iter()
                .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                .collect(),
            results: result_tensor_types
                .iter()
                .map(|tensor_type| TypeAndAttributes { r#type: tensor_type.as_ref(), attributes: None })
                .collect(),
            visibility: SymbolVisibility::Private,
            ..Default::default()
        },
        function_region,
        location,
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use pretty_assertions::assert_eq;
    use ryft_core::{
        CompositeAttribute, DataType, DomainTracingContext, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
        ShardingDimension,
    };

    use crate::Domain;
    use crate::domains::Tracer;

    use super::*;

    #[test]
    fn test_named_composition_input_types() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("varying", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("placed", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("reduced", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("unreduced", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["placed"])])
            .unwrap()
            .with_reduced_axes(["reduced"])
            .unwrap()
            .with_unreduced_axes(["unreduced"])
            .unwrap();
        let local = ArrayType::new(DataType::F32, Shape::new(vec![4.into()])).with_sharding(sharding.clone()).unwrap();
        let boundary = local.clone().with_sharding(sharding.with_varying_manual_axes(["varying"]).unwrap()).unwrap();
        let unsharded = ArrayType::scalar(DataType::F32);

        assert_eq!(
            named_composition_input_types(&[boundary, unsharded.clone()]).unwrap(),
            vec![ArrayIrType::from(local), ArrayIrType::from(unsharded)],
        );
    }

    #[test]
    fn test_named_composition_registry_supports_downstream_semantics() {
        // Registration consumes a semantic key and an ordinary typed program, not a Ryft operation enum variant.
        // Consequently a downstream lowerer can use the same sharing seam after performing its own family conversion.
        let boundary_type = ArrayType::scalar(DataType::F32);
        let (_, program) = DomainTracingContext::<Domain<'static>>::trace(
            |inputs: Vec<Tracer<'static>>| Ok(inputs),
            vec![ArrayIrType::from(boundary_type.clone())],
        )
        .unwrap();
        let program = program.simplified().unwrap();
        let key = NamedCompositionKey {
            composite: CompositeKey::new("downstream.example", 1)
                .with_attribute("enabled", CompositeAttribute::Boolean(true)),
            input_types: vec![boundary_type.clone()],
            output_types: vec![boundary_type],
        };
        let trace_count = Cell::new(0);
        let mut registry = NamedCompositionFunctionMap::default();
        registry
            .register(key.clone(), || {
                trace_count.set(trace_count.get() + 1);
                Ok(program.clone())
            })
            .unwrap();
        registry
            .register(key, || {
                trace_count.set(trace_count.get() + 1);
                Ok(program)
            })
            .unwrap();

        assert_eq!(trace_count.get(), 1);
        assert_eq!(registry.functions.len(), 1);
        assert_eq!(registry.order.len(), 1);
        assert_eq!(registry.functions.values().next().unwrap().symbol, "downstream.example");
    }
}
