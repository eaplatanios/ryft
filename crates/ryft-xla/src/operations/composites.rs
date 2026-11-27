use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use ryft_core::{
    ArrayType, AttentionConfiguration, AttentionImplementation, AttentionInputs, COMPOSITE_OPERATION_NAME,
    COMPOSITE_TRANSPOSE_OPERATION_NAME, CompositeKey, CompositeOperation, CompositeTransposeOperation,
    CustomRuleSource, DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
    DifferentiableType, FlatProgram, Operation, Parameter, Parameterized, Placeholder, Program, ProgramError, RegionId,
    RegionRef, RegionRole, RematerializationOptimizationBarrier, ScaledDotOperation, Typed, TypedProgram,
};
#[cfg(test)]
use ryft_mlir::Operation as _;
use ryft_mlir::dialects::func;
use ryft_mlir::{
    Block, BlockRef, Context as MlirContext, LocationRef, Region, SymbolVisibility, Type as MlirType,
    TypeAndAttributes, Value as MlirValue, ValueRef,
};

use crate::lowering::EffectTokens;
use crate::lowering::emissions::omits_regions;
use crate::lowering::regions::NestedProgramLoweringExtension;
use crate::lowering::states::CollectiveLoweringState;
use crate::lowering::types::TypeLoweringExtension;
use crate::operations::attention::{AttentionConfigurationLoweringExtension, select_attention_implementation};
use crate::operations::call::lower_shared_function_call;
use crate::operations::quantization::{
    scaled_dot_composition_key, scaled_dot_composition_types, trace_scaled_dot_composition,
};
use crate::{Constant, Error, LowerableOperation, LowerableOperationFamily, Lowerer, LoweringError, Session, Type};

/// Backend implementation that replaces the calls of a named composite when the XLA lowering targets a
/// platform that supports it. Implementations are selected from an explicit table by the name and version of the
/// definition, together with the boundary of the call and the target platform, before anything is emitted, so that
/// a call either lowers to its implementation or never touches one (i.e., declining emits nothing). A call that no
/// implementation replaces lowers its body. Every implementation must validate the attributes of the key and the
/// boundary that it supports and decline any other call, because its key alone does not fix them.
pub(crate) enum NamedCompositeImplementation {
    /// cuDNN fused attention kernels for a call of the forward attention definition, which
    /// [`select_attention_implementation`] selects.
    Attention {
        /// Configuration of the definition.
        configuration: AttentionConfiguration,

        /// Structure of the inputs of the definition (i.e., which optional inputs are present).
        input_structure: AttentionInputs<Placeholder>,

        /// Whether the definition also returns the log-sum-exp residual of its softmax.
        return_residual: bool,
    },

    /// cuDNN fused attention backward kernels for a call of the backward attention definition, which
    /// [`select_attention_implementation`] selects.
    AttentionBackward {
        /// Configuration of the definition.
        configuration: AttentionConfiguration,

        /// Structure of the inputs of the definition (i.e., which optional inputs are present).
        input_structure: AttentionInputs<Placeholder>,
    },

    /// Implementation of an attention definition that requests the fused kernels for a call that they cannot
    /// implement, which reports why instead of silently lowering the portable body.
    FusedAttentionUnavailable {
        /// Name of the definition.
        name: &'static str,

        /// Reason why the fused kernels are unavailable.
        reason: String,
    },

    /// Squares the input of a named composite in lowering and eager dispatch tests.
    #[cfg(test)]
    TestSquare,
}

impl NamedCompositeImplementation {
    /// Returns the implementation that replaces the calls of the named composite `key` with the provided input
    /// and output types when lowering for `target_platform`, or [`None`] when the calls lower their body.
    pub(crate) fn select(
        key: &CompositeKey,
        input_types: &[Type],
        output_types: &[Type],
        target_platform: Option<&str>,
    ) -> Option<Self> {
        match (key.name(), key.version(), target_platform) {
            #[cfg(test)]
            ("ryft.tests.xla_named_square", 1, Some("cuda")) | ("ryft.tests.xla_cpu_named_square", 1, Some("cpu")) => {
                Some(Self::TestSquare)
            }
            _ => select_attention_implementation(key, input_types, output_types, target_platform),
        }
    }

    /// Returns the error that a call with this implementation reports when it cannot use the implementation and would
    /// otherwise lower its portable body instead (e.g., when eager execution replays the body of a call whose concrete
    /// inputs refine the types at which it was staged), or [`None`] when the body is an acceptable substitute. Calls
    /// that explicitly request a backend implementation (e.g., fused attention) never silently use their body.
    pub(crate) fn body_substitution_error(&self) -> Option<Error> {
        let (name, reason) = match self {
            Self::Attention { configuration, .. } | Self::AttentionBackward { configuration, .. }
                if configuration.implementation() != AttentionImplementation::Fused =>
            {
                return None;
            }
            Self::Attention { .. } => (DOT_PRODUCT_ATTENTION_COMPOSITE_NAME, None),
            Self::AttentionBackward { .. } => (DOT_PRODUCT_ATTENTION_BACKWARD_COMPOSITE_NAME, None),
            Self::FusedAttentionUnavailable { name, reason } => (*name, Some(reason.as_str())),
            #[cfg(test)]
            Self::TestSquare => return None,
        };
        let reason = reason.unwrap_or(
            "eager execution replays the body of a call whose inputs refine the types at which it was staged; \
             execute the compiled program instead",
        );
        Some(Error::Lowering(LoweringError::UnsupportedOperation {
            operation: name.to_string(),
            reason: format!("the fused attention implementation is unavailable because {reason}"),
        }))
    }

    /// Lowers one call with this implementation, whose boundary has the provided input and output types, returning
    /// one MLIR value per output.
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Type],
        output_types: &[Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let array_types = |types: &[Type]| {
            types
                .iter()
                .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()
                .map_err(ProgramError::from)
        };
        match self {
            Self::Attention { configuration, input_structure, return_residual } => (*configuration).lower_fused(
                *input_structure,
                *return_residual,
                input_values,
                array_types(input_types)?.as_slice(),
                array_types(output_types)?.as_slice(),
                &mut lowerer.block(),
                lowerer.context(),
                lowerer.location(),
            ),
            Self::AttentionBackward { configuration, input_structure } => (*configuration).lower_fused_backward(
                *input_structure,
                input_values,
                array_types(input_types)?.as_slice(),
                array_types(output_types)?.as_slice(),
                &mut lowerer.block(),
                lowerer.context(),
                lowerer.location(),
            ),
            Self::FusedAttentionUnavailable { name, reason } => {
                Err(Error::Lowering(LoweringError::UnsupportedOperation {
                    operation: name.to_string(),
                    reason: format!("the fused attention implementation is unavailable because {reason}"),
                }))
            }
            #[cfg(test)]
            Self::TestSquare => {
                let multiply = lowerer.block().append_operation(ryft_mlir::dialects::stable_hlo::multiply(
                    input_values[0],
                    input_values[0],
                    lowerer.location(),
                )?)?;
                Ok(vec![multiply.result(0).unwrap().as_ref()])
            }
        }
    }
}

impl<V: Typed + Parameter, RegionOperation, S: CustomRuleSource<V, RegionOperation>> LowerableOperation
    for CompositeOperation<V, RegionOperation, S>
where
    V::Type: Eq + Hash + DifferentiableType + Into<Type>,
    Self: Operation<Type = V::Type>,
{
    #[inline]
    fn composite_key(&self) -> Option<&CompositeKey> {
        self.key()
    }

    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Self::Type],
        output_types: &[Self::Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        // A named composite call lowers to the backend implementation that replaces it, if any, and otherwise to a call
        // of the shared function that implements its body, if it has one, before falling back to inlining its body.
        if let Some(key) = self.key() {
            let input_types = input_types.iter().cloned().map(Into::into).collect::<Vec<Type>>();
            let output_types = output_types.iter().cloned().map(Into::into).collect::<Vec<Type>>();
            let target_platform = lowerer.collective_state().target_platform();
            if let Some(backend) =
                NamedCompositeImplementation::select(key, &input_types, &output_types, target_platform)
            {
                return backend.lower(lowerer, input_values, &input_types, &output_types);
            }
            let function = lowerer
                .nested_functions()
                .as_ref()
                .and_then(|functions| functions.named(key, &input_types, &output_types));
            if let Some(function) = function {
                return lower_shared_function_call(
                    function.symbol(),
                    function.output_types(),
                    input_values,
                    &mut lowerer.block(),
                    lowerer.context(),
                    lowerer.location(),
                );
            }

            // Calls whose key and boundary are shared by calls with another body are not indexed, so they find their
            // shared function by their own body instead, as `CallFunctionMap::named` explains.
            let function = lowerer.nested_functions().zip(lowerer.regions().first()).and_then(|(functions, primal)| {
                functions.get(primal).filter(|function| function.input_types().len() == input_values.len())
            });
            if let Some(function) = function {
                return lower_shared_function_call(
                    function.symbol(),
                    function.output_types(),
                    input_values,
                    &mut lowerer.block(),
                    lowerer.context(),
                    lowerer.location(),
                );
            }
        }

        let [_, ..] = lowerer.regions() else {
            return Err(Error::from(ProgramError::MalformedProgram(format!(
                "`{COMPOSITE_OPERATION_NAME}` expected a primal region but got no regions"
            ))));
        };
        if lowerer.regions().len() != self.region_slots().len() {
            return Err(Error::from(ProgramError::MalformedProgram(format!(
                "`{}` expected {} attached regions but got {}",
                COMPOSITE_OPERATION_NAME,
                self.region_slots().len(),
                lowerer.regions().len(),
            ))));
        }
        // Composite call regions are traced through contexts with fresh roots whose local capture tables are
        // discarded, so they can never legally reference the enclosing function's captures, because the trace boundary
        // rejects bodies that register captures. Lowering the primal region with an empty capture namespace turns any
        // constant that references a capture and still sneaks in into a loud `MissingCapturedConstant` error instead of
        // silently aliasing whatever value occupies the referenced slot of the enclosing capture prefix. Runtime arrays
        // that such a body refers to are session constants instead, whose values are scoped to the module, so the
        // empty namespace keeps them reachable. Only the primal region is lowered, because attached rule regions are
        // dormant under lowering and retained rules are callbacks rather than regions. Regions traced as nested
        // programs (i.e., those of `while`, `scan`, `condition`, and `linear_call`) share the enclosing capture scope
        // and keep inheriting it.
        let (mut block, context, location) = (lowerer.block(), lowerer.context(), lowerer.location());
        let (effect_tokens, regions, captured_values, nested_functions, collective_state) =
            lowerer.split_effect_tokens();
        regions[0].lower_inline(
            input_values,
            &mut block,
            context,
            location,
            collective_state.session_values(captured_values),
            &RematerializationOptimizationBarrier::None,
            nested_functions,
            collective_state,
            effect_tokens,
        )
    }
}

impl<V: Typed + Parameter, RegionOperation, S> LowerableOperation for CompositeTransposeOperation<V, RegionOperation, S>
where
    Self: Operation,
{
    fn lower<'b, 'c: 'b, 't: 'c, O: LowerableOperationFamily>(
        &self,
        lowerer: &mut Lowerer<'b, 'c, 't, O>,
        input_values: &[ValueRef<'b, 'c, 't>],
        input_types: &[Self::Type],
        output_types: &[Self::Type],
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        let _ = (lowerer, input_values, input_types, output_types);
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "a `{COMPOSITE_TRANSPOSE_OPERATION_NAME}` has no forward program and cannot be lowered to \
                 StableHLO; transpose it before lowering",
            ),
        }
        .into())
    }
}

/// Returns the backend-local input types of a named decomposition after boundary validation and differentiation, which
/// are `input_types` with their varying manual axes cleared.
///
/// Private decomposition functions contain local array computations and bind no manual axes. Manual variation has
/// already determined the explicit transitions and adjoints of the enclosing program, so it must not trigger new
/// transitions when expanding these functions. Retain mesh placement and reduction state, and keep the original types
/// in the [`NamedCompositionKey`] so the external operation contract remains unchanged.
pub(crate) fn named_composition_input_types(input_types: &[ArrayType]) -> Result<Vec<Type>, Error> {
    input_types
        .iter()
        .map(|input| {
            let sharding = input
                .sharding()
                .map(|sharding| sharding.clone().with_varying_manual_axes(Vec::<String>::new()))
                .transpose()?;
            Ok(Type::from(input.clone().with_sharding(sharding)?))
        })
        .collect()
}

/// Identity of one generated decomposition of a named composition at a concrete typed boundary.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NamedCompositionKey {
    /// Semantic identity of the composition, whose name also names the emitted private function.
    composite: CompositeKey,

    /// Logical operand types of the composition boundary.
    input_types: Vec<ArrayType>,

    /// Logical result types of the composition boundary.
    output_types: Vec<ArrayType>,
}

impl NamedCompositionKey {
    /// Creates the key of the decomposition of `composite` at the boundary with the provided logical types.
    pub(crate) fn new(composite: CompositeKey, input_types: Vec<ArrayType>, output_types: Vec<ArrayType>) -> Self {
        Self { composite, input_types, output_types }
    }
}

/// Private decomposition function emitted for one [`NamedCompositionKey`].
pub(crate) struct NamedCompositionFunction {
    /// Module-unique decomposition symbol.
    symbol: String,

    /// Canonical typed decomposition program.
    program: FlatProgram<Session>,
}

impl NamedCompositionFunction {
    /// Returns the module-unique decomposition symbol.
    #[inline]
    pub(crate) fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// Module-owned registry of canonical named compositions and their private decomposition functions.
#[derive(Default)]
pub(crate) struct NamedCompositionFunctionMap {
    /// Functions indexed by semantic operation and typed boundary.
    functions: HashMap<NamedCompositionKey, NamedCompositionFunction>,

    /// Keys in the stable order of their first occurrence, used for deterministic function emission.
    order: Vec<NamedCompositionKey>,
}

impl NamedCompositionFunctionMap {
    /// Returns the functions indexed by semantic operation and typed boundary.
    #[inline]
    pub(crate) fn functions(&self) -> &HashMap<NamedCompositionKey, NamedCompositionFunction> {
        &self.functions
    }

    /// Returns the keys in the stable order of their first occurrence, used for deterministic function emission.
    #[inline]
    pub(crate) fn order(&self) -> &[NamedCompositionKey] {
        &self.order
    }

    /// Returns the registered decomposition for `operation` at the provided typed boundary, or `None` when no
    /// decomposition is registered for it.
    pub(crate) fn get(
        &self,
        operation: &ScaledDotOperation,
        input_types: &[ArrayType],
        output_types: &[ArrayType],
    ) -> Result<Option<&NamedCompositionFunction>, Error> {
        Ok(self.functions.get(&scaled_dot_composition_key(operation, input_types, output_types)?))
    }

    /// Registers one typed decomposition lazily, sharing an existing private function when the complete semantic key
    /// is already present. The registry is independent of the operation family that produced `key` and `program`, and
    /// closed backend operation enums only decide when to call this seam. A key whose composite name is already taken
    /// by another key receives a symbol suffixed with the number of earlier keys of that name.
    fn register<F: FnOnce() -> Result<FlatProgram<Session>, Error>>(
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
}

/// Collects every named semantic composition required by `program` and traces each distinct typed decomposition once.
/// Only the regions that lowering emits for `target_platform` contribute (refer to [`omits_regions`]), so neither a
/// composite call that a backend implementation replaces on that target nor the body of a scan that admits no
/// iteration contributes compositions.
pub(crate) fn collect_named_composition_functions<Input, Output, O: LowerableOperationFamily>(
    program: &Program<Constant, O, Input, Output>,
    target_platform: Option<&str>,
) -> Result<NamedCompositionFunctionMap, Error>
where
    Input: Parameterized<Constant>,
    Output: Parameterized<Constant>,
{
    /// Registers the compositions of `region` and of the computation regions that its instructions attach, visiting
    /// each region at most once.
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
            let operation = instruction.operation();
            if omits_regions(region, instruction, target_platform) {
                continue;
            }
            for (index, nested) in instruction.regions().iter().enumerate() {
                if operation.region_role(index) == Some(RegionRole::Computation) {
                    walk(RegionRef::new(region.arena(), *nested)?, map, visited, target_platform)?;
                }
            }
            // Payloads are recognized through their projections, so any family that embeds them gets their
            // compositions.
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

/// Emits the private decomposition function of one named composition into `module_block`.
pub(crate) fn emit_named_composition_function<'b, 'c: 'b, 't: 'c>(
    module_block: &mut BlockRef<'b, 'c, 't>,
    function: &NamedCompositionFunction,
    collective_state: &CollectiveLoweringState,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<(), Error> {
    let input_types = function.program.input_types();
    let output_types = function.program.output_types();
    let argument_tensor_types =
        input_types.iter().map(|r#type| r#type.lower(context, location)).collect::<Result<Vec<_>, _>>()?;
    let result_tensor_types =
        output_types.iter().map(|r#type| r#type.lower(context, location)).collect::<Result<Vec<_>, _>>()?;
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
        let outputs = function.program.lower_inline(
            input_values.as_slice(),
            &mut function_block_ref,
            context,
            location,
            &[],
            &RematerializationOptimizationBarrier::None, // Decompositions are traced independently of the lowered program and contain no `call` sharing.
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
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayOperation, BatchAxis, BatchingLevelExtent, Compare, ComparisonDirection, CompositeAttribute,
        CompositeJvpRule, CustomRuleDefinition, CustomRuleRegistration, DataType, DimensionValue, DomainTracer,
        DomainTracingContext, DotDimensionNumbers, Error as CoreError, LogicalMesh, MeshAxis, MeshAxisType, Mul,
        MulOperation, NamedComposite, PrintOperation, Program, ProgramBuilder, ProjectedContext, Reduce, ScanOperation,
        Shape, Sharding, ShardingDimension, Softmax, batch, composite, differentiate_at, softmax,
    };
    use ryft_mlir::{Block, Context as MlirContext, Location};

    use crate::lowering::emissions::lowers_without_regions;
    use crate::lowering::tests::test_vector_type;
    use crate::operations::Operation;
    use crate::sessions::tests::domain_mesh;
    use crate::tests::{execution_client, replicated_array, values_to_bytes};
    use crate::{Array, Constant, LowerableProgram, ProgramLoweringOptions, Session, Tracer};

    use super::*;

    // Value that composite rules see when a composite executes through the projection of the session onto arrays.
    type ArrayTracer = DomainTracer<ProjectedContext<Session, ArrayType>>;

    /// Returns a program that applies `calls` named composite calls of `rules` to the same input, each attaching the
    /// interned `body`, and returns their outputs.
    fn named_calls_program(
        rules: &CustomRuleRegistration<Constant, Operation>,
        body: &Arc<Program<Constant, Operation, Vec<Constant>, Vec<Constant>>>,
        calls: usize,
    ) -> FlatProgram<Session> {
        let vector_type = test_vector_type(4);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(vector_type.clone().into());
        let outputs = (0..calls)
            .map(|_| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; calls])
            .unwrap()
    }

    /// Returns a named composite definition `x ↦ x + x` with the provided key name and version.
    fn named_rules_with_version(key: &str, version: u32) -> CustomRuleRegistration<Constant, Operation> {
        let key = CompositeKey::new(key.to_string(), version);
        let output_types = |types: &[Type]| Ok(types.to_vec());
        let named = NamedComposite::new(key, output_types, |inputs| Ok(vec![inputs[0].clone() + inputs[0].clone()]));
        CustomRuleRegistration::new(CustomRuleDefinition::new("doubled").with_named_composite(named))
    }

    /// Returns the program `x ↦ x + x` over the test vector type, shared as a named composite body.
    fn doubled_body() -> Arc<Program<Constant, Operation, Vec<Constant>, Vec<Constant>>> {
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(test_vector_type(4).into());
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
        Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
    }

    /// Lowers `program`, whose outputs all have the test vector type, for `target_platform`.
    fn lower_named_calls(program: &FlatProgram<Session>, target_platform: Option<&str>) -> Result<String, Error> {
        let input_types = vec![test_vector_type(4)];
        let output_types = vec![test_vector_type(4); program.output_ids().len()];
        program
            .lower(
                "main",
                &ProgramLoweringOptions::new()
                    .without_pruning()
                    .with_input_types(&input_types)
                    .with_output_types(&output_types)
                    .without_boundary_shardings()
                    .with_target_platform(target_platform),
            )
            .map(|module| module.into_stable_hlo())
    }

    #[test]
    fn test_lowers_without_regions() {
        // Only a named composite call that a backend implementation replaces, or whose body lowers to a shared
        // function, lowers without reading its regions.
        let types = vec![Type::Array(test_vector_type(4))];
        let rules = named_rules_with_version("ryft.tests.xla_named_square", 1);
        let named = Operation::Composite(CompositeOperation::new(rules.reference()));
        assert!(lowers_without_regions(&named, &types, &types, None, Some("cuda")));
        assert!(!lowers_without_regions(&named, &types, &types, None, None));
        let unnamed = Operation::Array(ArrayOperation::Add(AddOperation::new()));
        assert!(!lowers_without_regions(&unnamed, &types, &types, None, Some("cuda")));
    }

    #[test]
    fn test_named_composite_implementation_body_substitution_error() {
        // An implementation that the call explicitly requests rejects substituting the portable body, while other
        // implementations accept it.
        let unavailable = NamedCompositeImplementation::FusedAttentionUnavailable {
            name: DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
            reason: "the inputs are not `bf16`".to_string(),
        };
        assert!(matches!(
            unavailable.body_substitution_error(),
            Some(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == DOT_PRODUCT_ATTENTION_COMPOSITE_NAME
                    && reason == "the fused attention implementation is unavailable because the inputs are not `bf16`",
        ));
        assert!(NamedCompositeImplementation::TestSquare.body_substitution_error().is_none());
    }

    #[test]
    fn test_named_composite_implementation_lower() {
        // An unavailable fused attention implementation reports why instead of emitting anything.
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block = context.block_with_no_arguments();
        let mut lowerer =
            Lowerer::new(block.as_ref(), &context, location).with_regions(Vec::<FlatProgram<Session>>::new());
        let unavailable = NamedCompositeImplementation::FusedAttentionUnavailable {
            name: DOT_PRODUCT_ATTENTION_COMPOSITE_NAME,
            reason: "the inputs are not `bf16`".to_string(),
        };
        assert!(matches!(
            unavailable.lower(&mut lowerer, &[], &[], &[]),
            Err(Error::Lowering(LoweringError::UnsupportedOperation { operation, reason }))
                if operation == DOT_PRODUCT_ATTENTION_COMPOSITE_NAME
                    && reason == "the fused attention implementation is unavailable because the inputs are not `bf16`",
        ));
        assert!(block.as_ref().is_empty());
    }

    #[test]
    fn test_composite_operation_lower_inlines_only_the_primal_program() {
        // A retained `composite` call lowers only its primal program and threads its effects onto the enclosing
        // ordered I/O chain, for every rule layout. Nothing from the rule programs that the user supplies, which the
        // multiplies on the tangent and cotangent sides mark here, reaches the emitted module.
        let vector_type = test_vector_type(4);
        let primal = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let printed = builder
                .add_instruction(PrintOperation::<ArrayType>::new("primal"), Vec::new(), vec![input], None)
                .unwrap()[0];
            let output =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![printed, printed], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let jvp = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let tangent = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            let output_tangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![tangent, tangent], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![output, output_tangent],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let forward = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![output, input],
                    vec![Placeholder],
                    vec![Placeholder, Placeholder],
                )
                .unwrap()
        };
        let backward = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let residual = builder.add_input(vector_type.clone().into());
            let cotangent = builder.add_input(vector_type.clone().into());
            let input_cotangent =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(
                    vec![input_cotangent],
                    vec![Placeholder, Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };
        for (operation, rules) in [
            (CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, false), vec![&jvp]),
            (CompositeOperation::from_rule_regions(CompositeJvpRule::Absent, true), vec![&forward, &backward]),
            (CompositeOperation::from_rule_regions(CompositeJvpRule::Explicit, true), vec![&jvp, &forward, &backward]),
        ] {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let mut regions = vec![builder.import_region(primal.entry_region_ref())];
            regions.extend(rules.into_iter().map(|rule| builder.import_region(rule.entry_region_ref())));
            let input = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(Operation::Composite(operation), regions, vec![input], None).unwrap()[0];
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap();
            let module = program.to_mlir_module("main").unwrap();

            assert_eq!(
                module,
                indoc! {r#"
                    module {
                      func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                        %0 = stablehlo.after_all  : !stablehlo.token
                        %1 = stablehlo.custom_call @ryft.print(%arg0, %0) {api_version = 4 : i32, backend_config = {label = "primal"}, has_side_effect = true, operand_layouts = [dense<0> : tensor<1xindex>, dense<> : tensor<0xindex>], result_layouts = [dense<> : tensor<0xindex>]} : (tensor<4xf32>, !stablehlo.token) -> !stablehlo.token
                        %2 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                        return %2 : tensor<4xf32>
                      }
                    }
                "#},
            );
        }
    }

    #[test]
    fn test_composite_operation_lower_rejects_malformed_instructions() {
        // A call without a key lowers its primal region, so it needs the primal region and one region per rule slot.
        let context = MlirContext::new();
        let location = context.unknown_location().as_ref();
        let block = context.block_with_no_arguments();
        let operation = CompositeOperation::<Constant, Operation>::from_rule_regions(CompositeJvpRule::Explicit, false);
        let mut lowerer =
            Lowerer::new(block.as_ref(), &context, location).with_regions(Vec::<FlatProgram<Session>>::new());
        assert!(matches!(
            operation.lower(&mut lowerer, &[], &[], &[]),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`composite` expected a primal region but got no regions",
            ),
        ));
        let mut lowerer =
            Lowerer::new(block.as_ref(), &context, location).with_regions(vec![doubled_body().as_ref().clone()]);
        assert!(matches!(
            operation.lower(&mut lowerer, &[], &[], &[]),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`composite` expected 2 attached regions but got 1",
            ),
        ));
    }

    #[test]
    fn test_composite_operation_lower_executes_batching_rule() {
        // A composite with a custom batching rule executes through the projection of the session onto arrays,
        // which lifts the regions of the calls that it binds. Its unbatched derivative differentiates the primal `x²`,
        // while batching it, before or after differentiating it, applies the rule, which computes `2x²` so that its use
        // is observable.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let input = replicated_array(&session, &mesh, DataType::F32, &[3], &values_to_bytes(&[0.5f32, 1.0, 1.5]));
        let tangent = replicated_array(&session, &mesh, DataType::F32, &[3], &values_to_bytes(&[1f32, 1.0, 1.0]));
        let function = composite(|x: ArrayTracer| Ok(x.clone() * x)).with_batching(
            |x: ArrayTracer, axis: BatchAxis, _: BatchingLevelExtent<ArrayTracer>| {
                Ok((x.clone() * x.clone() + x.clone() * x, axis))
            },
        );

        let (value, derivative) = differentiate_at(input.clone()).jvp(tangent.clone(), |x| function.call(x)).unwrap();
        assert_eq!(value.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![0.25f32, 1.0, 2.25]);
        assert_eq!(derivative.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1f32, 2.0, 3.0]);

        let batched: Array =
            batch(|x| function.call(x), input.clone(), BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        assert_eq!(batched.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![0.5f32, 2.0, 4.5]);

        let (value, derivative): (Array, Array) = batch(
            |(x, tangent)| Ok(differentiate_at(x).jvp(tangent, |x| function.call(x))?),
            (input, tangent),
            (BatchAxis::new(0), BatchAxis::new(0)),
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        assert_eq!(value.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![0.5f32, 2.0, 4.5]);
        assert_eq!(derivative.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2f32, 4.0, 6.0]);
    }

    #[test]
    fn test_composite_operation_lower_executes_broadcast_sensitive_batching_rule() {
        // A batching rule that is sensitive to broadcasting depends on the batch axes of its inputs (i.e., it computes
        // `2xy` when both inputs are mapped and `3xy` otherwise). Batching the derivative with a replicated `y` whose
        // tangent is mapped yields the exact tangents `3(ẋy + xẏ)` of each batch item, which requires taking a
        // diagonal on device.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let function = composite(|(x, y): (ArrayTracer, ArrayTracer)| Ok(x * y)).with_broadcast_sensitive_batching(
            |(x, y): (ArrayTracer, ArrayTracer),
             (x_axis, y_axis): (BatchAxis, BatchAxis),
             _: BatchingLevelExtent<ArrayTracer>| {
                let product = x * y;
                let scaled = match x_axis.is_replicated() || y_axis.is_replicated() {
                    true => product.clone() + product.clone() + product,
                    false => product.clone() + product,
                };
                Ok((scaled, if x_axis.is_replicated() { y_axis } else { x_axis }))
            },
        );
        let x = replicated_array(&session, &mesh, DataType::F32, &[3], &values_to_bytes(&[1f32, 2.0, 3.0]));
        let y = replicated_array(&session, &mesh, DataType::F32, &[], &values_to_bytes(&[4f32]));
        let x_tangent = replicated_array(&session, &mesh, DataType::F32, &[3], &values_to_bytes(&[1f32, 0.5, 0.25]));
        let y_tangent = replicated_array(&session, &mesh, DataType::F32, &[3], &values_to_bytes(&[2f32, 3.0, 4.0]));
        let (value, derivative): (Array, Array) = batch(
            |((x, y), (x_tangent, y_tangent))| {
                Ok(differentiate_at((x, y)).jvp((x_tangent, y_tangent), |inputs| function.call(inputs))?)
            },
            ((x, y), (x_tangent, y_tangent)),
            ((BatchAxis::new(0), BatchAxis::replicated()), (BatchAxis::new(0), BatchAxis::new(0))),
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        assert_eq!(value.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![12f32, 24.0, 36.0]);
        assert_eq!(derivative.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![18f32, 24.0, 39.0]);
    }

    #[test]
    fn test_composite_operation_lower_custom_rule_inlines_only_the_primal_program() {
        // A custom rule call lowers only its primal program, because its derivative rules are retained callbacks, not
        // regions.
        let vector_type = test_vector_type(4);
        let rules = CustomRuleRegistration::new(CustomRuleDefinition::<Constant, Operation>::new("doubled"));
        let primal = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(vector_type.clone().into());
            let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let primal_region = builder.import_region(primal.entry_region_ref());
        let input = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                Operation::Composite(CompositeOperation::new(rules.reference())),
                vec![primal_region],
                vec![input],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_composite_operation_lower_named_composite_shares_one_function_per_specialization() {
        // Repeated named composite calls with one body specialization lower to calls of one shared private function
        // named after the definition, while a single call inlines its body.
        let rules = named_rules_with_version("ryft.tests.xla_named_shared", 1);
        let body = doubled_body();
        assert_eq!(
            lower_named_calls(&named_calls_program(&rules, &body, 2), None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_shared.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
                    %0 = call @ryft.tests.xla_named_shared.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_shared.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1 : tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
        assert_eq!(
            lower_named_calls(&named_calls_program(&rules, &body, 1), None).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_composite_operation_lower_named_composite_selects_backend_implementations() {
        // A backend implementation replaces the calls on its target, which then need no shared function, and calls
        // lower their body elsewhere.
        let rules = named_rules_with_version("ryft.tests.xla_named_square", 1);
        let body = doubled_body();
        let program = named_calls_program(&rules, &body, 2);
        assert_eq!(
            lower_named_calls(&program, Some("cuda")).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    %1 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    return %0, %1 : tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
        assert_eq!(
            lower_named_calls(&program, None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_square.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
                    %0 = call @ryft.tests.xla_named_square.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_square.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1 : tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );

        // Calls of definitions that no backend implementation recognizes lower their body on every target.
        let rules = named_rules_with_version("ryft.tests.xla_named_unrecognized", 1);
        let program = named_calls_program(&rules, &body, 1);
        assert_eq!(
            lower_named_calls(&program, Some("cuda")).unwrap(),
            indoc! {r#"
                module {
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_composite_operation_lower_named_composite_separates_conflicting_bodies() {
        // Two separately registered definitions share one key and boundary but carry different bodies (e.g., because
        // their named axes differ). Each call must still lower to the shared function of its own body.
        let first = named_rules_with_version("ryft.tests.xla_named_conflict", 1);
        let second = named_rules_with_version("ryft.tests.xla_named_conflict", 1);
        let doubled = doubled_body();
        let squared = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(test_vector_type(4).into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(test_vector_type(4).into());
        let outputs = [(&first, &doubled), (&second, &squared), (&first, &doubled), (&second, &squared)]
            .into_iter()
            .map(|(rules, body)| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 4])
            .unwrap();
        assert_eq!(
            lower_named_calls(&program, None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_conflict.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func private @ryft.tests.xla_named_conflict.v1_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
                    %0 = call @ryft.tests.xla_named_conflict.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_conflict.v1_1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = call @ryft.tests.xla_named_conflict.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %3 = call @ryft.tests.xla_named_conflict.v1_1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1, %2, %3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_composite_operation_lower_named_composite_separates_bodies_calling_different_versions() {
        // Two definitions with one key have bodies that differ only in the version of the named composite that they
        // call. Their renderings include that version, so their bodies are not merged and each keeps its own call. The
        // nested calls of both versions have structurally equal bodies, which share one function named after the first.
        let inner_body = doubled_body();
        let outer_body = |version: u32| {
            let inner = named_rules_with_version("ryft.tests.xla_named_inner", version);
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(test_vector_type(4).into());
            let regions = vec![builder.intern_callee(&inner_body, None).unwrap()];
            let operation = Operation::Composite(CompositeOperation::new(inner.reference()));
            let output = builder.add_instruction(operation, regions, vec![input], None).unwrap()[0];
            (inner, Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()))
        };
        let (_first_inner, first_body) = outer_body(1);
        let (_second_inner, second_body) = outer_body(2);
        assert_ne!(first_body.to_string(), second_body.to_string());
        let first = named_rules_with_version("ryft.tests.xla_named_outer", 1);
        let second = named_rules_with_version("ryft.tests.xla_named_outer", 1);
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(test_vector_type(4).into());
        let outputs = [(&first, &first_body), (&second, &second_body), (&first, &first_body), (&second, &second_body)]
            .into_iter()
            .map(|(rules, body)| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 4])
            .unwrap();
        assert_eq!(
            lower_named_calls(&program, None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_inner.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func private @ryft.tests.xla_named_outer.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @ryft.tests.xla_named_inner.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func private @ryft.tests.xla_named_outer.v1_1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = call @ryft.tests.xla_named_inner.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
                    %0 = call @ryft.tests.xla_named_outer.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_outer.v1_1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = call @ryft.tests.xla_named_outer.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %3 = call @ryft.tests.xla_named_outer.v1_1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1, %2, %3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_composite_operation_lower_named_composite_selects_backends_per_key_of_a_shared_body() {
        // Calls of two versions share one interned body region, but only the first version has a backend implementation
        // on CUDA. The second version's calls must still share a function of their own instead of following the first
        // version's selection.
        let replaced = named_rules_with_version("ryft.tests.xla_named_square", 1);
        let lowered = named_rules_with_version("ryft.tests.xla_named_square", 2);
        let body = doubled_body();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(test_vector_type(4).into());
        let outputs = [&replaced, &lowered, &lowered]
            .into_iter()
            .map(|rules| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(&body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 3])
            .unwrap();
        assert_eq!(
            lower_named_calls(&program, Some("cuda")).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_square.v2(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_square.v2(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = call @ryft.tests.xla_named_square.v2(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1, %2 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_composite_operation_lower_named_composite_separates_bodies_that_cannot_be_shared() {
        // Two definitions share one key and boundary, and the body of the first holds a constant, which structural
        // deduplication cannot key. Its calls must inline their own body instead of calling the second body's function.
        let first = named_rules_with_version("ryft.tests.xla_named_unshareable", 1);
        let second = named_rules_with_version("ryft.tests.xla_named_unshareable", 1);
        let unshareable = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let input = builder.add_input(test_vector_type(4).into());
            builder.add_constant(Constant::Dimension(DimensionValue::constant(3).unwrap()));
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            Arc::new(builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap())
        };
        let doubled = doubled_body();
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let input = builder.add_input(test_vector_type(4).into());
        let outputs = [(&first, &unshareable), (&second, &doubled), (&first, &unshareable), (&second, &doubled)]
            .into_iter()
            .map(|(rules, body)| {
                let operation = Operation::Composite(CompositeOperation::new(rules.reference()));
                let regions = vec![builder.intern_callee(body, None).unwrap()];
                builder.add_instruction(operation, regions, vec![input], None).unwrap()[0]
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder], vec![Placeholder; 4])
            .unwrap();
        assert_eq!(
            lower_named_calls(&program, None).unwrap(),
            indoc! {r#"
                module {
                  func.func private @ryft.tests.xla_named_unshareable.v1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                  func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
                    %0 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    %1 = call @ryft.tests.xla_named_unshareable.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    %2 = stablehlo.multiply %arg0, %arg0 : tensor<4xf32>
                    %3 = call @ryft.tests.xla_named_unshareable.v1(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
                    return %0, %1, %2, %3 : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_composite_operation_lower_executes_softmax_as_one_eager_dispatch() {
        // A held softmax function executes each eager call as one dispatch of its whole body, where the capability
        // dispatches every operation of the same decomposition separately. Both compute the same probabilities bit for
        // bit, including the rows with NaN, positive infinity, only excluded classes, ties, and a saturated maximum.
        // NaN results are compared as NaN, because their payloads are not part of the contract.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let (nan, infinity) = (f32::NAN, f32::INFINITY);
        let values = [
            nan, 0.0, 1.0, infinity, 0.0, 1.0, -infinity, -infinity, -infinity, 0.0, 0.0, 0.0, -infinity, 0.0, 1.0,
            9.0, -9.0, -9.0,
        ];
        let logits = replicated_array(&session, &mesh, DataType::F32, &[6, 3], &values_to_bytes(&values));
        let function = softmax::<ProjectedContext<Session, ArrayType>>(-1);
        let dispatches = || {
            let statistics = session.eager_dispatch_statistics();
            (statistics.misses, statistics.hits)
        };

        let first = function.call(logits.clone()).unwrap();
        assert_eq!(dispatches(), (1, 0));
        let second = function.call(logits.clone()).unwrap();
        assert_eq!(dispatches(), (1, 1));
        // The capability dispatches its 21 operations one by one, and 8 of them reuse the programs that earlier ones of
        // equal types compiled.
        let primitive = logits.softmax(-1).unwrap();
        assert_eq!(dispatches(), (14, 9));
        let assert_same = |actual: &Array, expected: &[f32]| {
            let actual = actual.copy_to_host().unwrap().elements::<f32>().unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits(),
                    "{actual} differs from {expected}",
                );
            }
        };
        let probabilities = primitive.copy_to_host().unwrap().elements::<f32>().unwrap();
        assert_same(&first, &probabilities);
        assert_same(&second, &probabilities);
        assert!(probabilities[..9].iter().all(|probability| probability.is_nan()));
        // The saturated maximum rounds to the largest value below one on the CPU and to one on CUDA, whose exponential
        // is less accurate.
        assert!((0.99999994..=1.0).contains(&probabilities[15]), "{}", probabilities[15]);

        // The derivatives are those of the capability, because both differentiate the same decomposition.
        let weights = (0..18).map(|index| index as f32 - 8.0).collect::<Vec<_>>();
        let weights = replicated_array(&session, &mesh, DataType::F32, &[6, 3], &values_to_bytes(&weights));
        let gradient = differentiate_at(logits.clone())
            .with_captures(weights.clone())
            .gradient(|logits, weights| function.call(logits)?.mul(&weights)?.reduce_sum(&[0, 1], None))
            .unwrap();
        let expected = differentiate_at(logits)
            .with_captures(weights)
            .gradient(|logits, weights| logits.softmax(-1)?.mul(&weights)?.reduce_sum(&[0, 1], None))
            .unwrap();
        assert_same(&gradient, &expected.copy_to_host().unwrap().elements::<f32>().unwrap());
    }

    #[test]
    fn test_composite_operation_lower_executes_softmax_non_finite_rows_after_other_programs() {
        // XLA's CPU backend on aarch64 Linux returned garbage instead of NaN for the non-finite rows of a compiled
        // softmax once other programs had been compiled on the same client, when the body applied `exp` to a constant
        // NaN. The softmax now selects the NaN after its exponential. Compiling the one-bit integer comparisons of
        // `test_compare_operation_lower_one_bit_integers` first was one known trigger.
        let client = execution_client();
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        for data_type in [DataType::Boolean, DataType::I1, DataType::U1] {
            let left = replicated_array(&session, &mesh, data_type, &[4], &[0, 0, 1, 1]);
            let right = replicated_array(&session, &mesh, data_type, &[4], &[0, 1, 0, 1]);
            let signed = data_type == DataType::I1;
            for (direction, expected) in [
                (ComparisonDirection::Equal, [true, false, false, true]),
                (ComparisonDirection::NotEqual, [false, true, true, false]),
                (ComparisonDirection::LessThan, [false, !signed, signed, false]),
                (ComparisonDirection::LessThanOrEqual, [true, !signed, signed, true]),
                (ComparisonDirection::GreaterThan, [false, signed, !signed, false]),
                (ComparisonDirection::GreaterThanOrEqual, [true, signed, !signed, true]),
            ] {
                assert_eq!(
                    left.compare(&right, direction).unwrap().copy_to_host().unwrap().elements::<bool>().unwrap(),
                    expected,
                    "{data_type} {direction:?}",
                );
            }
        }

        // Rows with NaN, with positive infinity, and with only excluded classes are NaN, and a finite row is not.
        let (nan, infinity) = (f32::NAN, f32::INFINITY);
        let values = [nan, 0.0, 1.0, infinity, 0.0, 1.0, -infinity, -infinity, -infinity, 0.0, 0.0, 0.0];
        let logits = replicated_array(&session, &mesh, DataType::F32, &[4, 3], &values_to_bytes(&values));
        let function = softmax::<ProjectedContext<Session, ArrayType>>(-1);
        let probabilities = function.call(logits).unwrap().copy_to_host().unwrap().elements::<f32>().unwrap();
        assert!(probabilities[..9].iter().all(|probability| probability.is_nan()), "{probabilities:?}");
        assert!(
            probabilities[9..].iter().all(|probability| (probability - 1.0 / 3.0).abs() < 1e-7),
            "{probabilities:?}",
        );
    }

    #[test]
    fn test_composite_transpose_operation_lower() {
        // A composite transpose with an attached backward rule exists only in reverse mode and must be transposed away
        // before lowering, so lowering it is rejected.
        let vector_type = test_vector_type(4);
        let backward = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let residual = builder.add_input(vector_type.clone().into());
            let cotangent = builder.add_input(vector_type.clone().into());
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![residual, cotangent], None).unwrap()[0];
            builder
                .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap()
        };
        let operation = CompositeTransposeOperation::<Constant, Operation>::from_backward_region(
            1,
            vec![Type::Array(vector_type.clone())],
            vec![Type::Array(vector_type.clone())],
        );
        let mut builder = ProgramBuilder::<Constant, Operation>::new();
        let backward_region = builder.import_region(backward.entry_region_ref());
        let tangent = builder.add_input(vector_type.clone().into());
        let residual = builder.add_input(vector_type.clone().into());
        let output = builder
            .add_instruction(
                Operation::CompositeTranspose(operation),
                vec![backward_region],
                vec![tangent, residual],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Constant>, Vec<Constant>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            program.to_mlir_module("main"),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::UnsupportedOperation { message })
                    if *message == "a `composite_transpose` has no forward program and cannot be lowered to \
                        StableHLO; transpose it before lowering",
            ),
        ));
    }

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
            vec![Type::from(local), Type::from(unsharded)],
        );
    }

    #[test]
    fn test_named_composition_function_map_register() {
        // Registration consumes a semantic key and an ordinary typed program, not a Ryft operation enum variant.
        // Consequently a downstream lowerer can use the same sharing seam after performing its own family conversion.
        let boundary_type = ArrayType::scalar(DataType::F32);
        let (_, program) = DomainTracingContext::<Session>::trace(
            |inputs: Vec<Tracer>| Ok(inputs),
            vec![Type::from(boundary_type.clone())],
        )
        .unwrap();
        let program = program.into_simplified().unwrap();
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

    #[test]
    fn test_named_composition_function_map_register_suffixes_shared_names() {
        // Keys that share a composite name but differ in their boundary types receive distinct symbols, suffixed with
        // the number of earlier keys of that name.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![4.into()]));
        let composite = CompositeKey::new("downstream.example", 1);
        let mut registry = NamedCompositionFunctionMap::default();
        for boundary_type in [scalar_type.clone(), vector_type.clone(), scalar_type.clone()] {
            let (_, program) = DomainTracingContext::<Session>::trace(
                |inputs: Vec<Tracer>| Ok(inputs),
                vec![Type::from(boundary_type.clone())],
            )
            .unwrap();
            let key = NamedCompositionKey::new(composite.clone(), vec![boundary_type.clone()], vec![boundary_type]);
            registry.register(key, || Ok(program.into_simplified().unwrap())).unwrap();
        }

        assert_eq!(
            registry.order().iter().map(|key| registry.functions()[key].symbol()).collect::<Vec<_>>(),
            vec!["downstream.example", "downstream.example_1"],
        );
        assert_eq!(registry.order()[1].input_types, vec![vector_type]);
    }

    #[test]
    fn test_collect_named_composition_functions_skips_the_bodies_of_scans_without_iterations() {
        // A scaled dot in the body of a scan needs its composition when the length of the scan admits an iteration,
        // and not on any target when its length is zero, because lowering then emits nothing of its body (refer to
        // `omits_regions`).
        let element_type = ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![2.into(), 16.into()]));
        let scale_type = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![2.into(), 1.into()]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 2.into()]));
        let carry_types = [output_type, element_type.clone(), element_type, scale_type.clone(), scale_type];
        let body = {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            builder.add_input(ArrayType::scalar(DataType::I64).into());
            let carries = carry_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect::<Vec<_>>();
            let dimensions = DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new());
            let operation = ScaledDotOperation::new(dimensions, DataType::F32, true, true);
            let output = builder.add_instruction(operation, Vec::new(), carries[1..].to_vec(), None).unwrap()[0];
            let outputs = std::iter::once(output).chain(carries[1..].iter().copied()).collect();
            Arc::new(builder.build(outputs, vec![Placeholder; 6], vec![Placeholder; 5]).unwrap())
        };
        let symbols = |length: usize| {
            let mut builder = ProgramBuilder::<Constant, Operation>::new();
            let inputs = carry_types.iter().map(|r#type| builder.add_input(r#type.clone().into())).collect();
            let regions = vec![builder.intern_callee(&body, None).unwrap()];
            let scan = ScanOperation::<Type>::new(5, length);
            let outputs = builder.add_instruction(scan, regions, inputs, None).unwrap().to_vec();
            let program = builder
                .build::<Vec<Constant>, Vec<Constant>>(outputs, vec![Placeholder; 5], vec![Placeholder; 5])
                .unwrap();
            let map = collect_named_composition_functions(&program, None).unwrap();
            map.order().iter().map(|key| map.functions()[key].symbol().to_string()).collect::<Vec<_>>()
        };
        assert_eq!(symbols(0), Vec::<String>::new());
        assert_eq!(symbols(1), vec!["xla.scaled_dot".to_string()]);
    }
}
