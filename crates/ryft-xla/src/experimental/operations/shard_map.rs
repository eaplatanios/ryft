use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::Arc;

use ryft_core::macros::check_count;
use ryft_core::{
    Array as CpuArray, ArrayIrType, ArrayOperation, ArrayReferenceDischarge, ArrayType, BroadcastOperation,
    CalleeRegionDriver, CaptureConstant, Concretizable, ConstantOperation, Context, ConvertElementType,
    CotangentDestinationKind, DifferentiableOperation, DifferentiableType, DifferentiationContext,
    DifferentiationDriver, DifferentiationDual, DifferentiationError, DifferentiationPolicy, Dimension, DivOperation,
    InputRegionProvenance, LogicalMesh, MaybeZero, MeshAxisType, OperandCotangents, Operation, OperationFormatter,
    OutputRegionProvenance, ParallelReduceOperation, ParallelReductionKind, Parameterized, ParameterizedFamily,
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationInput, PartialEvaluationValue, PartialValue,
    PartiallyEvaluatableOperation, Placeholder, Program, ProgramBuilder, ProgramError, ProjectedValue,
    ReferenceAddUpdateOperation, ReferenceDischargeContext, ReferenceDischargeDriver, ReferenceDischargePolicy,
    ReferenceDischargeValue, ReferenceDischargeableOperation, ReferenceFreezeOperation, ReferenceNewOperation,
    ReferenceRoot, ReferenceSource, ReferenceType, RegionInterface, RegionRef, RegionSlot, ReshapeOperation,
    ReshapeParameters, Shape, Sharding, ShardingDimension, StagingContext, Tracer, TracingContext,
    TransposableOperation, TranspositionContext, TranspositionDriver, Type, TypeError, Typed, Value, ValueId,
    ValueProjection, Zero, ZeroOperation, discharge_reference_free_operation, operand_cotangents,
};

use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgram, materialize_transpose_cotangent};
use crate::experimental::shard_map::{
    FlatTracedShardMap, ShardMap, ShardMapInvocationLeaf, ShardMapLocalTraceInput, ShardMapLocalTraceOutput,
    ShardMapTraceError, ShardMapTracer, TracedShardMap, derive_global_output_types,
};

/// Canonical operation name for [`ShardMapOperation`].
pub(crate) const SHARD_MAP_OPERATION_NAME: &str = "shard_map";

/// Canonical higher-order shard-map op used for staged tracing, differentiation, and lowering. The local body
/// program is not part of this payload: it is the operation's one attached `body` region, so this payload carries
/// only the manual SPMD boundary metadata that the region program cannot represent.
///
/// A boundary position may be a reference under the contract in the `# References Under shard_map` section of the
/// [`shard_map`](mod@crate::experimental::shard_map) module documentation. A reference input `ref<T>` carries the
/// global referent `T` (sharded like an array input by its input sharding) and reaches the body as `ref<T_local>`, the
/// shard that the input sharding assigns to the executing device. A reference output must forward a reference input by
/// identity, which the operation states through its crate-private `output_forwarding` accessor; an output whose
/// forwarding is not declared is rejected by type inference, so a reference output is never accepted on provenance the
/// operation cannot name.
#[derive(Clone, Debug)]
pub struct ShardMapOperation<V> {
    /// Manual SPMD metadata (mesh, boundary shardings, and manual axes) governing the attached body region.
    shard_map: ShardMap,

    /// Global input types declared at the shard-map boundary. A reference position carries the global referent.
    input_types: Vec<ArrayIrType>,

    /// Global output types declared at the shard-map boundary. A reference position carries the global referent.
    output_types: Vec<ArrayIrType>,

    /// Refer to the documentation of [`Self::output_forwarding`].
    output_forwarding: Vec<Option<usize>>,

    /// Phantom marker tying the operation to the traced leaf type it will replay with.
    marker: PhantomData<fn() -> V>,
}

impl<V> ShardMapOperation<V> {
    /// Creates a shard-map operation for an already traced local `body`, including reference inputs and forwarded
    /// reference outputs. The body is attached when binding this operation. Global output shapes are derived from
    /// the output shardings, and reference forwarding is derived from the body's canonical reference analysis.
    ///
    /// # Parameters
    ///
    ///   - `body`: Local body whose input types describe the shards seen by each device.
    ///   - `global_input_types`: Global types corresponding positionally to the body's inputs.
    ///   - `mesh`: Logical mesh containing the manual axes.
    ///   - `in_specs`: Sharding of each global input or reference referent.
    ///   - `out_specs`: Sharding of each global output or forwarded reference referent.
    ///   - `manual_axes`: Active manual axes; an empty list selects all manual mesh axes.
    ///   - `check_vma`: Whether output specs must cover the axes along which local outputs vary.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid mesh/spec combinations, incompatible local body types, reference outputs that do
    /// not forward an input root, or writes to references replicated along an active manual axis.
    pub fn from_program<O: Operation<Type = ArrayIrType>>(
        body: &Program<V, O, Vec<V>, Vec<V>>,
        global_input_types: Vec<ArrayIrType>,
        mesh: LogicalMesh,
        in_specs: Vec<Sharding>,
        out_specs: Vec<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Result<Self, ShardMapTraceError>
    where
        V: Value<Type = ArrayIrType>,
    {
        let shard_map = ShardMap::new(mesh, in_specs, out_specs, manual_axes, check_vma)?;
        if global_input_types.len() != shard_map.in_shardings().len() {
            return Err(ShardMapTraceError::InputTypeCountMismatch {
                expected: shard_map.in_shardings().len(),
                actual: global_input_types.len(),
            });
        }
        let local_input_types = body.input_types();
        check_count!("input", local_input_types, global_input_types.len(), ProgramError);
        for (index, (global, local)) in global_input_types.iter().zip(&local_input_types).enumerate() {
            let expected =
                shard_map.local_input_type(index, boundary_array_type(global).map_err(ProgramError::from)?)?;
            if global.is_reference() != local.is_reference()
                || !shard_map_boundary_types_match(boundary_array_type(local).map_err(ProgramError::from)?, &expected)
            {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "`shard_map` body input {index} has type `{local}`, but its global input requires local {} \
                        `{expected}`",
                        if global.is_reference() { "reference referent" } else { "array type" },
                    ),
                }
                .into());
            }
        }
        let local_output_types = body.output_types();
        let referents = local_output_types
            .iter()
            .map(|r#type| boundary_array_type(r#type).cloned())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ProgramError::from)?;
        let global_outputs = derive_global_output_types(&shard_map, &referents)?;
        let output_types = local_output_types
            .iter()
            .zip(global_outputs)
            .map(|(local, global)| {
                if local.is_reference() {
                    ArrayIrType::Reference(ReferenceType::new(global))
                } else {
                    ArrayIrType::Array(global)
                }
            })
            .collect::<Vec<_>>();
        let analysis = body.entry_region_ref().reference_analysis(0).map_err(ProgramError::from)?;
        let output_forwarding = analysis
            .output_roots()
            .iter()
            .map(|root| match root {
                Some(ReferenceRoot::RegionInput { region, input_index }) if *region == body.entry_region_ref().id() => {
                    Some(*input_index)
                }
                _ => None,
            })
            .collect();
        let operation = Self::from_boundary(shard_map, global_input_types.clone(), output_types)
            .with_output_forwarding(output_forwarding)?;
        operation
            .infer_output_types(&global_input_types, &[body.entry_region_ref().interface()])
            .map_err(ProgramError::from)?;
        operation.validate_reference_body(body.entry_region_ref())?;
        Ok(operation)
    }

    /// Splits the provided erased shard-map body into a metadata-only operation and the local body program that the
    /// caller attaches as the operation's `body` region (or interns as a shared callee).
    #[inline]
    pub(crate) fn from_body(body: FlatTracedShardMap) -> (Self, XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>) {
        let (shard_map, input_types, output_types, program) = body.into_operation_parts();
        (Self::from_boundary(shard_map, input_types, output_types), program)
    }

    /// Returns the manual SPMD metadata governing the attached body region.
    #[inline]
    pub(crate) fn shard_map(&self) -> &ShardMap {
        &self.shard_map
    }

    /// Returns the global input types declared at the shard-map boundary.
    #[inline]
    pub(crate) fn global_input_types(&self) -> &[ArrayIrType] {
        &self.input_types
    }

    /// Returns the global output types declared at the shard-map boundary.
    #[inline]
    pub(crate) fn global_output_types(&self) -> &[ArrayIrType] {
        &self.output_types
    }

    /// Returns, for each global output, the input whose reference the output forwards by identity, or [`None`] for a
    /// value output. Every reference output must name its forwarded input here: the body's reference outputs can only
    /// forward the body's reference inputs (a reference allocated inside the body cannot escape and a derived view
    /// cannot be returned), and the forwarded input's sharding must equal the output's sharding.
    #[inline]
    pub(crate) fn output_forwarding(&self) -> &[Option<usize>] {
        &self.output_forwarding
    }

    /// Creates a metadata-only shard-map operation directly from its boundary parts, declaring every output as a
    /// value output. The local body program that realizes this boundary is authored separately and attached as the
    /// operation's `body` region; a boundary with reference outputs declares them through
    /// [`with_output_forwarding`](Self::with_output_forwarding).
    #[inline]
    pub(crate) fn from_boundary<I, O>(shard_map: ShardMap, input_types: I, output_types: O) -> Self
    where
        I: IntoIterator<Item: Into<ArrayIrType>>,
        O: IntoIterator<Item: Into<ArrayIrType>>,
    {
        let input_types = input_types.into_iter().map(Into::into).collect::<Vec<_>>();
        let output_types = output_types.into_iter().map(Into::into).collect::<Vec<_>>();
        let output_forwarding = vec![None; output_types.len()];
        Self { shard_map, input_types, output_types, output_forwarding, marker: PhantomData }
    }

    /// Returns a copy of this operation whose global output types are replaced by `global_output_types`, keeping the
    /// manual SPMD metadata, global input types, and output forwarding unchanged. Forward-mode differentiation uses
    /// this to align a tangent boundary's global output types with the tangent descriptors derived from the staged
    /// primal `shard_map`'s adapted output types (see `adapt_traced_shard_map_output_type`).
    pub(crate) fn with_global_output_types(
        mut self,
        global_output_types: Vec<ArrayIrType>,
    ) -> Result<Self, ShardMapTraceError> {
        if global_output_types.len() != self.output_types.len() {
            return Err(ShardMapTraceError::OutputTypeCountMismatch {
                expected: self.output_types.len(),
                actual: global_output_types.len(),
            });
        }
        self.output_types = global_output_types;
        Ok(self)
    }

    /// Returns a copy of this operation whose output forwarding is replaced by `output_forwarding` (refer to the
    /// documentation of [`output_forwarding`](Self::output_forwarding)), which must have one entry per global output.
    pub(crate) fn with_output_forwarding(
        mut self,
        output_forwarding: Vec<Option<usize>>,
    ) -> Result<Self, ShardMapTraceError> {
        if output_forwarding.len() != self.output_types.len() {
            return Err(ShardMapTraceError::OutputForwardingCountMismatch {
                expected: self.output_types.len(),
                actual: output_forwarding.len(),
            });
        }
        self.output_forwarding = output_forwarding;
        Ok(self)
    }

    /// Renders this operation's name followed by its complete manual SPMD boundary metadata, shared by [`Display`]
    /// and [`Operation::render`].
    ///
    /// Every field a consumer can observe is rendered, because [`Operation::render`] is the metadata fingerprint that
    /// the debug transform-cache diagnostic compares programs by: a field this rendering drops is a field whose
    /// corruption that diagnostic cannot see. All of `mesh`, `in_shardings`, `out_shardings`, `manual_axes`, and
    /// `check_vma` steer differentiation, transposition, and `sdy.manual_computation` lowering while being invisible
    /// to the instruction's rendered atom types, and so do the global boundary types: an operand type only has to
    /// match [`global_input_types`](Self::global_input_types) up to its dimension shardings, and a result type is the
    /// caller-ambient re-embedding of [`global_output_types`](Self::global_output_types) (see
    /// [`adapt_traced_shard_map_output_type`]) rather than those types themselves. The output forwarding is rendered
    /// exactly when some output forwards a reference input, so an all-value boundary renders as before and a boundary
    /// with a forwarded reference output never renders like one without. The only elided state is the [`PhantomData`]
    /// replay marker, which carries no semantics. Every field is sequence- or scalar-valued, so the rendering is
    /// deterministic without any ordering normalization, and every mesh axis name goes through [`render_axis_name`] so
    /// that distinct name lists always render distinctly.
    fn render_operation(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        let mesh_axes = self
            .shard_map
            .mesh()
            .axes()
            .iter()
            .map(|axis| format!("{}={}:{}", render_axis_name(axis.name()), axis.size(), axis.r#type()));
        let manual_axes = self.shard_map.manual_axes().iter().map(|axis| render_axis_name(axis.as_str()));
        OperationFormatter::new(formatter, indentation, SHARD_MAP_OPERATION_NAME)?.bracketed(|operation| {
            // The mesh is redundant with any rendered sharding, which embeds it, but a boundary may have no
            // shardings at all, so it is rendered unconditionally.
            operation.field("mesh", render_sequence(mesh_axes))?;
            operation.field("in_shardings", render_sequence(self.shard_map.in_shardings()))?;
            operation.field("out_shardings", render_sequence(self.shard_map.out_shardings()))?;
            operation.field("manual_axes", render_sequence(manual_axes))?;
            operation.field("check_vma", self.shard_map.check_vma())?;
            operation.field("global_input_types", render_sequence(self.input_types.as_slice()))?;
            operation.field("global_output_types", render_sequence(self.output_types.as_slice()))?;
            if self.output_forwarding.iter().any(Option::is_some) {
                let forwarding = self.output_forwarding.iter().map(|forwarded| match forwarded {
                    Some(input_index) => input_index.to_string(),
                    None => "_".to_string(),
                });
                operation.field("output_forwarding", render_sequence(forwarding))?;
            }
            Ok(())
        })
    }
}

/// Renders one mesh axis name as a single-quoted literal whose body uses Rust's canonical character escape codes
/// (see [`char::escape_debug`]), escaping single and double quotes, backslashes, and control characters.
///
/// Mesh axis names are arbitrary nonempty strings, so rendering them verbatim is not injective: the two-name list
/// `["a", "b"]` and the one-name list `["a', 'b"]` would both render as `['a', 'b']`. Because
/// [`ShardMapOperation::render_operation`] is the metadata fingerprint that the debug transform-cache diagnostic
/// compares programs by, such a collision would hide a genuinely different manual SPMD boundary. Escaped names
/// contain neither an unescaped quote nor an unescaped backslash, so no name can imitate the surrounding quoting or
/// the `, ` separator that [`render_sequence`] joins items with, and distinct name lists always render distinctly.
fn render_axis_name(name: &str) -> String {
    format!("'{}'", name.chars().flat_map(char::escape_debug).collect::<String>())
}

/// Renders one shard-map metadata sequence as a bracketed, comma-separated list. Items that embed mesh axis names
/// must render those names through [`render_axis_name`] for the result to remain injective.
fn render_sequence<I: IntoIterator<Item: Display>>(items: I) -> String {
    format!("[{}]", items.into_iter().map(|item| item.to_string()).collect::<Vec<_>>().join(", "))
}

impl<V> Display for ShardMapOperation<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render_operation(formatter, 0)
    }
}

/// Returns `true` when two shard-map boundary types agree apart from carried sharding metadata.
fn shard_map_boundary_types_match(actual: &ArrayType, expected: &ArrayType) -> bool {
    fn varying_manual_axes_match(actual: &Sharding, expected: &Sharding) -> bool {
        actual
            .varying_manual_axes()
            .iter()
            .filter(|axis_name| expected.mesh().axis_type(axis_name.as_str()) == Some(MeshAxisType::Manual))
            .eq(expected.varying_manual_axes().iter())
    }

    actual.data_type() == expected.data_type()
        && actual.shape() == expected.shape()
        && actual.layout() == expected.layout()
        && match (actual.sharding(), expected.sharding()) {
            (_, None) => true,
            (Some(actual), Some(expected)) => {
                actual.unreduced_axes() == expected.unreduced_axes()
                    && actual.reduced_axes() == expected.reduced_axes()
                    && varying_manual_axes_match(actual, expected)
            }
            (None, Some(expected)) => {
                expected.unreduced_axes().is_empty()
                    && expected.reduced_axes().is_empty()
                    && expected.varying_manual_axes().is_empty()
            }
        }
}

/// Re-embeds one traced shard-map output type into the caller's ambient sharding envelope.
///
/// Traced nested `shard_map` invocations stage as ordinary higher-order ops inside an already-local
/// caller context. When the caller's ambient sharding envelope differs from the captured shard-map
/// boundary, the staged result must use the ambient envelope again so downstream traced primitives
/// see a value in the surrounding local context rather than in the nested shard-map boundary space.
///
/// This mirrors the JAX intuition that a nested `shard_map` body returns to the enclosing
/// per-instance context after the inner manual region finishes; the inner boundary should not leak
/// out as the ambient type seen by surrounding primitives in the outer body.
fn adapt_traced_shard_map_output_type(
    actual_input_types: &[ArrayIrType],
    captured_input_types: &[ArrayIrType],
    captured_output_type: &ArrayType,
) -> ArrayType {
    if let ([ArrayIrType::Array(actual_input_type)], [ArrayIrType::Array(captured_input_type)]) =
        (actual_input_types, captured_input_types)
        && actual_input_type.sharding() != captured_input_type.sharding()
        && actual_input_type.shape().rank() == captured_output_type.shape().rank()
    {
        ArrayType::new(captured_output_type.data_type(), captured_output_type.shape().clone())
            .with_layout(captured_output_type.layout().cloned())
            .with_sharding(actual_input_type.sharding().cloned())
            .expect("adapted shard_map output type should preserve rank-compatible sharding")
    } else {
        captured_output_type.clone()
    }
}

/// Returns the global boundary type that a shard-map operand or result carries as an array: the type itself for an
/// array position and the global referent for a reference position. A dimension has no shard-map boundary type.
fn boundary_array_type(r#type: &ArrayIrType) -> Result<&ArrayType, TypeError> {
    match r#type {
        ArrayIrType::Reference(r#type) => Ok(r#type.referent()),
        r#type => <&ArrayType>::try_from(r#type),
    }
}

/// Validates the single attached shard-map body boundary and returns its interface.
fn shard_map_body_interface<T: Type>(
    region_interfaces: &[RegionInterface<T>],
    input_count: usize,
    output_count: usize,
) -> Result<&RegionInterface<T>, TypeError> {
    check_count!("region", region_interfaces, 1, TypeError);
    let interface = &region_interfaces[0];
    check_count!("body input", interface.input_types(), input_count, TypeError);
    check_count!("body output", interface.output_types(), output_count, TypeError);
    Ok(interface)
}

impl<V: Clone> Operation for ShardMapOperation<V> {
    type Type = ArrayIrType;

    #[inline]
    fn name(&self) -> &'static str {
        SHARD_MAP_OPERATION_NAME
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        const { &[RegionSlot::computation("body")] }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        self.render_operation(formatter, indentation)
    }

    // Every operand must agree with the declared global boundary type at its position up to carried sharding metadata,
    // an array operand exactly as before and a reference operand through its referent. A reference operand must reach
    // the body as a reference to its local shard, the global referent narrowed by the input sharding, and a reference
    // output must forward a declared reference input by identity under an equal output sharding. Array body types are
    // not checked against the local derivation because the tracing surface derives them itself; a reference body input
    // is checked because it types the state carry that discharge threads through the body.
    fn infer_output_types(
        &self,
        input_types: &[ArrayIrType],
        region_interfaces: &[RegionInterface<ArrayIrType>],
    ) -> Result<Vec<ArrayIrType>, TypeError> {
        let body_interface =
            shard_map_body_interface(region_interfaces, self.input_types.len(), self.output_types.len())?;
        check_count!("input", input_types, self.input_types.len(), TypeError);
        for (index, (actual, declared)) in input_types.iter().zip(&self.input_types).enumerate() {
            let body_input_type = &body_interface.input_types()[index];
            match declared {
                ArrayIrType::Reference(declared) => {
                    let actual = <&ReferenceType<ArrayType>>::try_from(actual)?;
                    if !shard_map_boundary_types_match(actual.referent(), declared.referent()) {
                        return Err(TypeError::invalid(format!(
                            "{} input types do not match the captured shard-map boundary",
                            self.name(),
                        )));
                    }
                    let local_referent = self
                        .shard_map
                        .local_input_type(index, declared.referent())
                        .map_err(|error| TypeError::invalid(error.to_string()))?;
                    let body_input_type = <&ReferenceType<ArrayType>>::try_from(body_input_type)?;
                    if !shard_map_boundary_types_match(body_input_type.referent(), &local_referent) {
                        return Err(TypeError::invalid(format!(
                            "{} body input {index} has type `{body_input_type}` but the local shard of reference \
                             input {index} is `{}`",
                            self.name(),
                            ReferenceType::new(local_referent),
                        )));
                    }
                }
                declared => {
                    let actual = <&ArrayType>::try_from(actual)?;
                    let declared = <&ArrayType>::try_from(declared)?;
                    if !shard_map_boundary_types_match(actual, declared) {
                        return Err(TypeError::invalid(format!(
                            "{} input types do not match the captured shard-map boundary",
                            self.name(),
                        )));
                    }
                    <&ArrayType>::try_from(body_input_type)?;
                }
            }
        }
        let mut output_types = Vec::with_capacity(self.output_types.len());
        for (index, declared) in self.output_types.iter().enumerate() {
            let body_output_type = &body_interface.output_types()[index];
            match declared {
                ArrayIrType::Reference(_) => {
                    <&ReferenceType<ArrayType>>::try_from(body_output_type)?;
                    let Some(forwarded) = self.output_forwarding[index] else {
                        return Err(TypeError::invalid(format!(
                            "{} output {index} is a reference whose forwarded input the operation does not declare",
                            self.name(),
                        )));
                    };
                    if forwarded >= input_types.len() || body_output_type != &body_interface.input_types()[forwarded] {
                        return Err(TypeError::invalid(format!(
                            "{} output {index} does not forward reference input {forwarded} by identity",
                            self.name(),
                        )));
                    }
                    if self.shard_map.out_shardings()[index] != self.shard_map.in_shardings()[forwarded] {
                        return Err(TypeError::invalid(format!(
                            "{} output {index} forwards reference input {forwarded} but its output sharding `{}` \
                             differs from the input sharding `{}`",
                            self.name(),
                            self.shard_map.out_shardings()[index],
                            self.shard_map.in_shardings()[forwarded],
                        )));
                    }
                    output_types.push(input_types[forwarded].clone());
                }
                declared => {
                    <&ArrayType>::try_from(body_output_type)?;
                    let declared = <&ArrayType>::try_from(declared)?;
                    output_types.push(ArrayIrType::Array(adapt_traced_shard_map_output_type(
                        input_types,
                        self.input_types.as_slice(),
                        declared,
                    )));
                }
            }
        }
        Ok(output_types)
    }

    // The body's inputs mirror the operands one for one and its outputs are the operation's outputs, exactly like a
    // jitted call, and the body is traced through a fresh-root context that discards captures, so it establishes an
    // empty capture namespace.
    #[inline]
    fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<InputRegionProvenance> {
        (region_index == 0).then_some(InputRegionProvenance::Forwarded { input_index })
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        vec![OutputRegionProvenance { region_index: 0, output_index }]
    }

    #[inline]
    fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
        (region_index == 0).then_some(0)
    }

    #[inline]
    fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
        self.output_forwarding().get(output_index).copied().flatten()
    }
}

impl<V: Clone> ShardMapOperation<V> {
    /// Validates the attached local `body` against the reference contract of the `# References Under shard_map`
    /// section of the [`shard_map`](crate::experimental::shard_map) module documentation, through the body's retained
    /// reference analysis. A body without references passes trivially.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the body stores a captured reference (a reference must be an
    /// explicit input with an input sharding), when a reference output is rooted in an allocation made inside the body
    /// or in a derived view, when a reference output forwards an input the operation does not declare, or when the
    /// forwarded input's sharding differs from the output's sharding, and [`ProgramError::UnsupportedOperation`] when
    /// the body mutates a reference input that is replicated along an active manual axis.
    pub(crate) fn validate_reference_body<O: Operation<Type = ArrayIrType>>(
        &self,
        body: RegionRef<'_, V, O>,
    ) -> Result<(), ProgramError>
    where
        V: Value<Type = ArrayIrType>,
    {
        let name = self.name();
        let analysis = body.reference_analysis(0)?;
        for (index, r#type) in body.input_types().iter().enumerate() {
            if !r#type.is_reference() {
                continue;
            }
            let root = ReferenceRoot::RegionInput { region: body.id(), input_index: index };
            if let Some(axis) = self.shard_map.input_replicated_manual_axes(index).first()
                && analysis.is_mutated(root)
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{name}` input {index} is a reference replicated along manual axis `{axis}`; mutating a \
                         replicated reference is not supported, shard it along that axis or read it only",
                    ),
                });
            }
        }
        for (index, root) in analysis.output_roots().iter().enumerate() {
            if analysis.is_view(ValueId::new(body.id(), body.output_ids()[index])) {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{name}` output {index} is a derived reference view; return its root reference instead",
                )));
            }
            let forwarded = match root {
                None => continue,
                Some(ReferenceRoot::RegionInput { region, input_index }) if *region == body.id() => *input_index,
                Some(_) => {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{name}` output {index} is a reference allocated inside the shard-map body, which cannot \
                         escape; a reference output must forward a reference input",
                    )));
                }
            };
            if self.output_forwarding().get(index).copied().flatten() != Some(forwarded) {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{name}` output {index} forwards reference input {forwarded} but the operation does not declare \
                     that forwarding",
                )));
            }
            if self.shard_map.out_shardings()[index] != self.shard_map.in_shardings()[forwarded] {
                return Err(ProgramError::MalformedProgram(format!(
                    "`{name}` output {index} forwards reference input {forwarded} but its output sharding `{}` \
                     differs from the input sharding `{}`",
                    self.shard_map.out_shardings()[index],
                    self.shard_map.in_shardings()[forwarded],
                )));
            }
        }
        Ok(())
    }
}

// A shard map with reference operands discharges its *local* body standalone: the body is an ordinary local program
// over local references, so `Program::discharge_references` types every state carry with the local shard type by
// construction, discharges allocations made inside the body within the body, and appends one hidden final-state output
// per mutated reference input. The rebuilt boundary keeps every operand position (a reference position becomes the
// global referent, an array sharded by the same input sharding), drops the forwarded reference outputs (the caller
// already holds those handles), and publishes each mutated input's final state as a trailing output whose output
// sharding is that input's input sharding, so the stateful ABI commits each device's shard into the global referent.
// The rule refuses a preserved reference, because a manual region threads state, not destination references, and it
// summarizes the body first so that the shared access-policy and consumption checks run exactly as for other
// structured rules. A body without references replays verbatim.
impl<V, C, P> ReferenceDischargeableOperation<C, P> for ShardMapOperation<V>
where
    V: PartialEq
        + Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + CaptureConstant
        + Concretizable<bool>,
    C: Context<Type = ArrayIrType, Constant = V, Operation = XlaOperation<V>>,
    P: ReferenceDischargePolicy<C, Referent = ArrayType>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, P>>(
        &self,
        context: &ReferenceDischargeContext<C, P>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, P>],
    ) -> Result<Vec<ReferenceDischargeValue<C, P>>, ProgramError> {
        let name = self.name();
        check_count!("input", inputs, self.input_types.len(), ProgramError);
        let body = driver.region(0)?;
        check_count!("input", body.input_ids(), inputs.len(), ProgramError);
        check_count!("output", body.output_ids(), self.output_types.len(), ProgramError);
        let allocations =
            inputs.iter().map(|input| context.operand_allocation(input, name)).collect::<Result<Vec<_>, _>>()?;
        if allocations.iter().all(Option::is_none) && !body.contains_references_in_closure() {
            return discharge_reference_free_operation(self, context, driver, inputs);
        }
        self.validate_reference_body(body)?;
        context.region_summary(self, 0, body, allocations.as_slice())?;
        for (index, allocation) in allocations.iter().enumerate() {
            if let Some(allocation) = allocation
                && !context.is_allocation_discharged(*allocation)?
            {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "`{name}` does not thread a preserved reference through its body, but input {index} denotes \
                         preserved {allocation}; discharge it or pass a value",
                    ),
                });
            }
        }

        // The forwarded reference outputs are dropped from the body before it is discharged standalone: a public
        // output that is a discharged reference has no state to publish, and the caller keeps the handle it passed.
        let value_output_indices = (0..self.output_types.len())
            .filter(|&index| self.output_forwarding[index].is_none())
            .collect::<Vec<_>>();
        let local_body = {
            let body = body.to_program();
            let mut builder = ProgramBuilder::<V, XlaOperation<V>>::new();
            let inputs = body.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let outputs = builder.splice_program(&body, inputs.as_slice())?;
            let outputs = value_output_indices.iter().map(|&index| outputs[index]).collect::<Vec<_>>();
            builder.build::<Vec<V>, Vec<V>>(
                outputs,
                vec![Placeholder; inputs.len()],
                vec![Placeholder; value_output_indices.len()],
            )?
        };
        let discharged = local_body.discharge_references::<ArrayReferenceDischarge>(0)?;
        let mutated_inputs = discharged
            .external_reference_bindings()
            .iter()
            .filter(|binding| binding.is_mutated())
            .map(|binding| match binding.source() {
                ReferenceSource::Input { index } => Ok(index),
                source => Err(ProgramError::MalformedProgram(format!(
                    "`{name}` body discharge bound {source} although the body has no capture prefix",
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Rebuild the boundary: every input keeps its position and input sharding, the value outputs keep theirs, and
        // one final-state output per mutated reference input follows them under that input's input sharding. The
        // final state is typed by the operand it replaces (the allocation's current state) rather than by the declared
        // global referent, because the declared boundary type may omit sharding metadata that the allocation carries
        // and the discharged state must match the allocation's referent type exactly.
        let operands = inputs
            .iter()
            .zip(&allocations)
            .map(|(input, allocation)| match allocation {
                Some(allocation) => context.allocation_value(*allocation),
                None => context.operand_value(input),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input_types = self
            .input_types
            .iter()
            .map(|r#type| boundary_array_type(r#type).cloned().map(ArrayIrType::Array))
            .collect::<Result<Vec<_>, _>>()?;
        let mut output_types =
            value_output_indices.iter().map(|&index| self.output_types[index].clone()).collect::<Vec<_>>();
        let mut out_shardings = value_output_indices
            .iter()
            .map(|&index| self.shard_map.out_shardings()[index].clone())
            .collect::<Vec<_>>();
        for &index in &mutated_inputs {
            output_types.push(operands[index].r#type().into_owned());
            out_shardings.push(self.shard_map.in_shardings()[index].clone());
        }
        let shard_map = ShardMap::from_shardings(
            self.shard_map.mesh().clone(),
            self.shard_map.in_shardings().to_vec(),
            out_shardings,
            self.shard_map.manual_axes().to_vec(),
            self.shard_map.check_vma(),
        );
        let operation = ShardMapOperation::<V>::from_boundary(shard_map, input_types, output_types);
        let program = discharged.program().clone();
        let mut outputs =
            context.parent().bind(XlaOperation::ShardMap(Box::new(operation)), vec![program], &operands)?;
        check_count!("output", outputs, value_output_indices.len() + mutated_inputs.len(), ProgramError);
        let final_states = outputs.split_off(value_output_indices.len());
        for (index, final_state) in mutated_inputs.iter().zip(final_states) {
            let allocation = allocations[*index].expect("a mutated body input is a reference operand");
            context.set_discharged_state(allocation, final_state, true)?;
        }

        // A forwarded reference output is reported as the handle the caller already holds at the forwarded position.
        let mut outputs = outputs.into_iter();
        Ok(self
            .output_forwarding
            .iter()
            .map(|forwarded| match forwarded {
                Some(input_index) => inputs[*input_index].clone(),
                None => ReferenceDischargeValue::Value(outputs.next().unwrap()),
            })
            .collect())
    }
}

// Online partial-evaluation rule for a staged `shard_map` — the map-boundary sibling of the
// [`JitCallOperation`](crate::experimental::ops::JitCallOperation) call rule: it splits the local body against the
// caller's known-ness while preserving the `shard_map` boundary, its mesh, and its shardings on both sides.
//
// The split fires only when some known input does *not* [`resolve`](Context::resolve) to a program constant in
// the known-side context — a genuine tracer into a live outer trace. All-known, all-unknown, and constant-resolved
// calls defer to the default fold-or-residualize behavior, which preserves the original boundary exactly.
//
// When the split fires, the *local* body program is split through the shared
// [`PartitionedProgram`](ryft_core::partial::PartitionedProgram) machinery. The known side is rewrapped as a
// `shard_map` whose global outputs
// are the fully known boundary outputs followed by the known→unknown residual edges (each edge's global type and
// sharding derived through `residual_boundary`, exactly as in the structural split), bound into the enclosing
// known-side context over the original known boundary inputs. The residual side is rewrapped as a `shard_map` over
// the surviving unknown boundary inputs plus those residual edges and emitted into the residual program.
impl<V, C> PartiallyEvaluatableOperation<C> for ShardMapOperation<V>
where
    V: PartialEq
        + Value<Type = ArrayIrType>
        + ryft_core::ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + CaptureConstant
        + Concretizable<bool>,
    C: Context<Type = ArrayIrType, Constant = V, Operation = XlaOperation<V>>,
{
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Split only a mixed boundary with at least one known-but-symbolic input; everything else keeps the default
        // fold-or-residualize behavior and therefore the original boundary.
        if !context.any_known_is_symbolic(inputs) || inputs.iter().all(PartialEvaluationValue::is_known) {
            return context.fold_or_residualize(
                XlaOperation::ShardMap(Box::new(self.clone())),
                driver.regions().map(|region| region.to_program()).collect(),
                inputs,
            );
        }
        // Split the local body through the shared online boundary machinery. The body's inputs are index-aligned
        // with the boundary inputs and carry the *local* types, so both split sides stay local programs.
        let body_program = driver.region(0)?;
        // A reference operand, a reference output, or a reference access anywhere in the body keeps the shard map
        // whole, for the same reason a `jit_call` stays whole: splitting would spread the accesses to one root across
        // two manual regions whose known side runs first, and the default rule preserves effect order by placing the
        // whole application on one side instead. The contract is validated first so a malformed body is reported as
        // such rather than residualized silently.
        if inputs.iter().any(|input| input.r#type().is_reference())
            || body_program.output_types().iter().any(Type::is_reference)
            || body_program.contains_reference_accesses_in_closure()
        {
            self.validate_reference_body(body_program)?;
            return context.fold_or_residualize(
                XlaOperation::ShardMap(Box::new(self.clone())),
                vec![body_program.to_program()],
                inputs,
            );
        }
        let input_known = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<bool>>();
        let partition = driver.partition_program(context, body_program, input_known.as_slice())?;
        // A trivial partition — one whose known program contains no instructions — hoists no work (its known side
        // can only forward known inputs as residual edges), so keep the original boundary and let the default
        // materialize those knowns directly as residual feeders.
        if partition.known_program().instructions().is_empty() {
            return context.fold_or_residualize(
                XlaOperation::ShardMap(Box::new(self.clone())),
                vec![body_program.to_program()],
                inputs,
            );
        }

        // Derive each residual edge's global boundary type and sharding from its local type.
        let mesh = self.shard_map.mesh();
        let residual_edge_boundaries = partition
            .residual_inputs()
            .iter()
            .zip(partition.residual_program().input_types())
            .filter_map(|(source, edge_type)| source.is_known().then_some(edge_type))
            .map(|edge_type| {
                let edge_type = <&ArrayType>::try_from(&edge_type).map_err(ProgramError::from)?;
                residual_boundary(edge_type, &self.shard_map).map_err(trace_error_from_shard_map)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Gather the known-side boundary metadata: shardings and global types per original index, with the residual
        // edges appended.
        let known_global_input_types = partition
            .known_input_indices()
            .iter()
            .map(|&index| self.input_types[index].clone())
            .collect::<Vec<_>>();
        let known_in_shardings = partition
            .known_input_indices()
            .iter()
            .map(|&index| self.shard_map.in_shardings()[index].clone())
            .collect::<Vec<_>>();
        let known_output_indices = partition
            .outputs()
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.is_known().then_some(index))
            .collect::<Vec<_>>();
        let mut known_global_output_types =
            known_output_indices.iter().map(|&index| self.output_types[index].clone()).collect::<Vec<_>>();
        let mut known_out_shardings = known_output_indices
            .iter()
            .map(|&index| self.shard_map.out_shardings()[index].clone())
            .collect::<Vec<_>>();
        for (global_type, sharding) in residual_edge_boundaries.iter() {
            known_global_output_types.push(global_type.clone().into());
            known_out_shardings.push(sharding.clone());
        }

        // Gather the staged-side boundary metadata: the surviving unknown boundary inputs plus the residual edges,
        // and the residual-owned outputs.
        let mut staged_global_input_types = Vec::with_capacity(partition.residual_inputs().len());
        let mut staged_in_shardings = Vec::with_capacity(partition.residual_inputs().len());
        for source in partition.residual_inputs().iter() {
            match source {
                PartialEvaluationInput::Unknown(index) => {
                    staged_global_input_types.push(self.input_types[*index].clone());
                    staged_in_shardings.push(self.shard_map.in_shardings()[*index].clone());
                }
                PartialEvaluationInput::Known(edge) => {
                    let (global_type, sharding) = &residual_edge_boundaries[*edge];
                    staged_global_input_types.push(global_type.clone().into());
                    staged_in_shardings.push(sharding.clone());
                }
            }
        }
        let mut staged_global_output_types = Vec::new();
        let mut staged_out_shardings = Vec::new();
        for (index, output) in partition.outputs().iter().enumerate() {
            if output.is_unknown() {
                staged_global_output_types.push(self.output_types[index].clone());
                staged_out_shardings.push(self.shard_map.out_shardings()[index].clone());
            }
        }

        let known_program = partition.known_program().clone();
        let mut known_output_types = known_program.output_types();
        for r#type in &mut known_output_types[known_output_indices.len()..] {
            *r#type = packed_residual_type(<&ArrayType>::try_from(&*r#type)?, &self.shard_map)?.into();
        }
        let known_input_types = known_program.input_types();
        let packed_known_program = reshape_program_boundary(known_program, known_input_types, known_output_types)?;
        let residual_program = partition.residual_program().clone();
        let mut residual_input_types = residual_program.input_types();
        for (source, r#type) in partition.residual_inputs().iter().zip(&mut residual_input_types) {
            if source.is_known() {
                *r#type = packed_residual_type(<&ArrayType>::try_from(&*r#type)?, &self.shard_map)?.into();
            }
        }
        let residual_output_types = residual_program.output_types();
        let packed_residual_program =
            reshape_program_boundary(residual_program, residual_input_types, residual_output_types)?;

        // Bind the known-side `shard_map` into the enclosing known-side context, emit the residual `shard_map`
        // over the surviving unknown boundary inputs plus the residual edges, and reassemble the original outputs.
        context.inline_partitioned_program(
            partition,
            inputs,
            |_known_program| {
                let known_shard_map = ShardMap::from_shardings(
                    mesh.clone(),
                    known_in_shardings,
                    known_out_shardings,
                    self.shard_map.manual_axes().to_vec(),
                    self.shard_map.check_vma(),
                );
                let known_operation = ShardMapOperation::from_boundary(
                    known_shard_map,
                    known_global_input_types,
                    known_global_output_types,
                );
                (XlaOperation::ShardMap(Box::new(known_operation)), vec![packed_known_program])
            },
            |_residual_program| {
                let staged_shard_map = ShardMap::from_shardings(
                    mesh.clone(),
                    staged_in_shardings,
                    staged_out_shardings,
                    self.shard_map.manual_axes().to_vec(),
                    self.shard_map.check_vma(),
                );
                let staged_operation = ShardMapOperation::from_boundary(
                    staged_shard_map,
                    staged_global_input_types,
                    staged_global_output_types,
                );
                (XlaOperation::ShardMap(Box::new(staged_operation)), vec![packed_residual_program])
            },
        )
    }
}

/// Packs a residual across the manual axes along which its local value can differ. Replicated residuals keep their
/// shape; varying residuals gain a leading dimension with one slot per distinct local value.
fn residual_boundary(
    local_type: &ArrayType,
    shard_map: &ShardMap,
) -> Result<(ArrayType, Sharding), ShardMapTraceError> {
    let axes = residual_manual_axes(local_type, shard_map);
    if axes.is_empty() {
        return Ok((local_type.clone(), Sharding::replicated(shard_map.mesh().clone(), local_type.rank())));
    }
    let extent = axes.iter().try_fold(1usize, |extent, axis| {
        extent
            .checked_mul(shard_map.mesh().axis_size(axis).unwrap())
            .ok_or_else(|| ProgramError::InvalidArgument {
                message: "shard-map residual extent overflows usize".to_string(),
            })
    })?;
    let mut dimensions = vec![ShardingDimension::sharded(axes)];
    dimensions.extend(vec![ShardingDimension::Replicated; local_type.rank()]);
    let sharding = Sharding::new(shard_map.mesh().clone(), dimensions)?;
    let shape = Shape::new(
        std::iter::once(Dimension::Static(extent))
            .chain(local_type.shape().dimensions().iter().cloned())
            .collect(),
    );
    Ok((ArrayType::new(local_type.data_type(), shape).with_memory(local_type.memory()), sharding))
}

/// Returns the active manual axes along which this residual's local value can vary.
fn residual_manual_axes(local_type: &ArrayType, shard_map: &ShardMap) -> Vec<String> {
    shard_map
        .manual_axes()
        .iter()
        .filter(|axis| local_type.sharding().is_some_and(|sharding| sharding.varying_manual_axes().contains(*axis)))
        .cloned()
        .collect()
}

/// Packs a varying local residual into its single-device slot without changing its elements or variation metadata.
fn packed_residual_type(local_type: &ArrayType, shard_map: &ShardMap) -> Result<ArrayType, TypeError> {
    if residual_manual_axes(local_type, shard_map).is_empty() {
        return Ok(local_type.clone());
    }
    local_type.with_inserted_dimension(0, Dimension::Static(1))
}

/// Rewraps a body's array boundary with element-preserving reshapes. Reference positions must remain unchanged.
fn reshape_program_boundary<V>(
    program: Program<V, XlaOperation<V>, Vec<V>, Vec<V>>,
    input_types: Vec<ArrayIrType>,
    output_types: Vec<ArrayIrType>,
) -> Result<Program<V, XlaOperation<V>, Vec<V>, Vec<V>>, ProgramError>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    let mut builder = ProgramBuilder::new();
    let inputs = input_types.iter().cloned().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
    let reshape =
        |builder: &mut ProgramBuilder<V, XlaOperation<V>>, value, source: &ArrayIrType, target: &ArrayIrType| {
            if source == target {
                return Ok(value);
            }
            let target = <&ArrayType>::try_from(target)?;
            let operation = ReshapeOperation::new(
                ReshapeParameters::new(target.shape().clone()).with_output_sharding(target.sharding().cloned()),
            );
            Ok::<_, ProgramError>(
                builder.add_instruction(
                    XlaOperation::Array(ArrayOperation::Reshape(operation)),
                    Vec::new(),
                    vec![value],
                    None,
                )?[0],
            )
        };
    let operands = inputs
        .iter()
        .copied()
        .zip(input_types.iter().zip(program.input_types()))
        .map(|(input, (source, target))| reshape(&mut builder, input, source, &target))
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = builder.splice_program(&program, &operands)?;
    let outputs = outputs
        .into_iter()
        .zip(program.output_types().iter().zip(&output_types))
        .map(|(output, (source, target))| reshape(&mut builder, output, source, target))
        .collect::<Result<Vec<_>, _>>()?;
    builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_types.len()])
}

/// Fuse-linearizes a shard-map body capture-free under the operands' activity mask into a primal body and a tangent
/// body that thread residuals as plain operand edges across the shard-map boundary. Returns both bodies with their
/// boundary operations and the output-activity mask used to reconstruct the caller's output duals.
///
/// The borrowed `body` region is linearized once through `driver`, yielding a primal sub-program
/// `local_inputs -> [local_outputs..., local_residuals...]` and a tangent sub-program
/// `[active(local_input_tangents)..., local_residuals...] -> [local_output_tangents...]` together with the residual
/// count. Each sub-program pairs with a fresh boundary [`ShardMapOperation`]: the primal boundary keeps every input and
/// gains the residual edges as trailing outputs, and the tangent boundary consumes the active inputs' tangents (for an
/// active reference input, a tangent reference `ref<tangent(T)>` under the primal's input sharding) followed by the
/// residual edges, with one residual slot per device from [`residual_boundary`]. A reference output that
/// forwards an active reference input forwards that input's tangent reference in the tangent boundary; one that
/// forwards an inactive reference input keeps its primal handle and has no tangent boundary slot. This is the
/// shard-map counterpart of the jitted-call rule, realizing `jvp(shard_map(f)) = shard_map(jvp f)` without introducing
/// symbolic captures.
///
/// # Parameters
///
///   - `operation`: Boundary metadata of the primal shard-map being linearized.
///   - `driver`: Call-scoped access to the active differentiation machinery.
///   - `body`: Borrowed `body` region of that shard-map.
///   - `activity`: One entry per operand recording whether the tangent body receives a tangent at that position.
#[allow(clippy::type_complexity)]
fn shard_map_bodies<C, D, V>(
    operation: &ShardMapOperation<V>,
    driver: &D,
    body: RegionRef<'_, V, XlaOperation<V>>,
    activity: &[bool],
) -> Result<
    (
        (ShardMapOperation<V>, Program<V, XlaOperation<V>, Vec<V>, Vec<V>>),
        (ShardMapOperation<V>, Program<V, XlaOperation<V>, Vec<V>, Vec<V>>),
        Vec<bool>,
    ),
    DifferentiationError,
>
where
    C: Context<Type = ArrayIrType, Constant = V, Operation = XlaOperation<V>>,
    D: DifferentiationDriver<C>,
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    let output_count = operation.output_types.len();
    check_count!("input", activity, operation.input_types.len(), ProgramError);
    let input_indices = activity
        .iter()
        .enumerate()
        .filter_map(|(index, &active)| active.then_some(index))
        .collect::<Vec<_>>();
    let output_activity = body.tangent_output_mask(&input_indices)?;
    let (primal_program, tangent_program, residual_count) =
        driver.linearize_program(body, &input_indices)?.into_parts();

    let shard_map = operation.shard_map();
    let mesh = shard_map.mesh();

    // The primal sub-program's trailing outputs beyond the original outputs are the residual edges; their local
    // types are authoritative and back the residual boundary on both bodies.
    let mut primal_output_types = primal_program.output_types();
    let mut residual_global_types = Vec::with_capacity(residual_count);
    let mut residual_shardings = Vec::with_capacity(residual_count);
    for residual_local_type in &primal_output_types[output_count..] {
        let residual_local_type = <&ArrayType>::try_from(residual_local_type).map_err(ProgramError::from)?;
        let (residual_global_type, residual_sharding) =
            residual_boundary(residual_local_type, shard_map).map_err(trace_error_from_shard_map)?;
        residual_global_types.push(ArrayIrType::Array(residual_global_type));
        residual_shardings.push(residual_sharding);
    }

    for r#type in &mut primal_output_types[output_count..] {
        *r#type = packed_residual_type(<&ArrayType>::try_from(&*r#type)?, shard_map)?.into();
    }
    let mut tangent_input_types = tangent_program.input_types();
    let tangent_residual_start = tangent_input_types.len() - residual_count;
    tangent_input_types[tangent_residual_start..].clone_from_slice(&primal_output_types[output_count..]);
    let primal_input_types = primal_program.input_types();
    let primal_program = reshape_program_boundary(primal_program, primal_input_types, primal_output_types)?;
    let tangent_output_types = tangent_program.output_types();
    let tangent_program = reshape_program_boundary(tangent_program, tangent_input_types, tangent_output_types)?;

    let primal_operation = ShardMapOperation::from_boundary(
        ShardMap::from_shardings(
            mesh.clone(),
            shard_map.in_shardings().to_vec(),
            shard_map.out_shardings().iter().cloned().chain(residual_shardings.iter().cloned()).collect(),
            shard_map.manual_axes().to_vec(),
            shard_map.check_vma(),
        ),
        operation.input_types.clone(),
        operation
            .output_types
            .iter()
            .cloned()
            .chain(residual_global_types.iter().cloned())
            .collect::<Vec<_>>(),
    )
    .with_output_forwarding(operation.output_forwarding.iter().copied().chain(vec![None; residual_count]).collect())
    .map_err(trace_error_from_shard_map)?;

    // The tangent boundary compacts the inputs by the activity mask, so a forwarded reference output must be remapped
    // onto the tangent position of the input it forwards.
    let mut tangent_positions = vec![None; activity.len()];
    let mut tangent_in_shardings = Vec::with_capacity(activity.len() + residual_count);
    let mut tangent_input_types = Vec::with_capacity(activity.len() + residual_count);
    for (index, _) in activity.iter().enumerate().filter(|(_, active)| **active) {
        tangent_positions[index] = Some(tangent_input_types.len());
        tangent_in_shardings.push(shard_map.in_shardings()[index].clone());
        tangent_input_types.push(operation.input_types[index].tangent()?);
    }
    tangent_in_shardings.extend(residual_shardings);
    tangent_input_types.extend(residual_global_types);
    let tangent_output_types = operation
        .output_types
        .iter()
        .zip(&output_activity)
        .filter(|(_, active)| **active)
        .map(|(r#type, _)| r#type.tangent())
        .collect::<Result<Vec<_>, _>>()?;
    let tangent_forwarding = operation
        .output_forwarding
        .iter()
        .zip(&output_activity)
        .filter(|(_, active)| **active)
        .map(|(forwarded, _)| forwarded.map(|index| tangent_positions[index].unwrap()))
        .collect();
    let tangent_out_shardings = shard_map
        .out_shardings()
        .iter()
        .zip(&output_activity)
        .filter(|(_, active)| **active)
        .map(|(sharding, _)| sharding.clone())
        .collect();
    let tangent_operation = ShardMapOperation::from_boundary(
        ShardMap::from_shardings(
            mesh.clone(),
            tangent_in_shardings,
            tangent_out_shardings,
            shard_map.manual_axes().to_vec(),
            shard_map.check_vma(),
        ),
        tangent_input_types,
        tangent_output_types,
    )
    .with_output_forwarding(tangent_forwarding)
    .map_err(trace_error_from_shard_map)?;

    Ok(((primal_operation, primal_program), (tangent_operation, tangent_program), output_activity))
}

// Capture-free forward-mode (JVP) rule for [`ShardMapOperation`], binding a primal `shard_map` and a tangent
// `shard_map` as ordinary [`XlaOperation`]s through the active context: a staging context stages both operations
// over its shared builder, while an eager context compiles and executes them immediately.
//
// This realizes the identity `jvp(shard_map(f)) = shard_map(jvp f)`: rather than capturing the global primals as
// residual factors and staging a linear `shard_map`, the rule keeps the
// manual region intact and threads every residual as a plain primal operand edge between two `shard_map`s, so no
// symbolic capture is ever introduced. The enclosing partial-evaluation split then discovers the residual operand
// edges structurally, exactly as for the jitted-call rule.
//
// The rule linearizes the body capture-free through `shard_map_bodies` under the operands' activity mask (a numeric
// operand is active, a plumbing reference operand that receives no tangent reference is not), giving a primal body
// `inputs -> [outputs..., residuals...]` and a tangent body
// `[active(input_tangents)..., residuals...] -> [output_tangents...]` together with the residual count. It then
// stages the primal `shard_map` over the operand primals (recovering the primal outputs followed by the residual
// values), stages the tangent `shard_map` over the active operand tangents followed by those residual values
// (recovering active output tangents, including forwarded active tangent references), and pairs each primal output
// with its tangent or a structural zero for an inactive output. The body program is keyed on
// [`XlaConstant`] regardless of the enclosing value type `V`, so the sub-programs are valid for every `V`.
//
// # Parameters
//
//   - `context`: Active evaluation or staging context used to bind the differentiated shard-map operations.
//   - `driver`: Call-scoped access to the attached shard-map body region.
//   - `inputs`: Primal and tangent values for the shard-map operands.
impl<C, V> DifferentiableOperation<C> for ShardMapOperation<V>
where
    C: Context<Type = ArrayIrType, Constant = V, Operation = XlaOperation<V>> + Zero<C::Value>,
    V: PartialEq
        + Value<Type = ArrayIrType>
        + ryft_core::ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>
        + CaptureConstant
        + Concretizable<bool>,
{
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let output_count = self.output_types.len();
        check_count!("input", inputs, self.input_types.len(), ProgramError);

        let body_program = driver.region(0)?;
        self.validate_reference_body(body_program)?;
        let activity = inputs.iter().map(DifferentiationDual::is_tangent_active).collect::<Vec<_>>();
        let ((primal_operation, primal_body_program), (tangent_operation, tangent_body_program), output_activity) =
            shard_map_bodies(self, driver, body_program, activity.as_slice())?;

        // Bind the primal `shard_map`, recovering the primal outputs followed by the residual values.
        let primal_operands = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let primal_operation = XlaOperation::ShardMap(Box::new(primal_operation));
        let mut primal_outputs =
            context.primal().bind(primal_operation, vec![primal_body_program], &primal_operands)?;
        if primal_outputs.len() < output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "shard_map primal body produced {} outputs which is fewer than its {output_count} primal \
                 output(s)",
                primal_outputs.len(),
            ))
            .into());
        }
        let residuals = primal_outputs.split_off(output_count);

        // The primal `shard_map` may re-embed its outputs into the caller's ambient sharding envelope. The tangent
        // `shard_map` carries residual operands and therefore cannot infer that envelope through the single-input
        // adaptation path, so derive its output descriptors from the adapted primal outputs while retaining the
        // tangent element representation.
        let tangent_output_types = primal_outputs
            .iter()
            .zip(&output_activity)
            .filter(|(_, active)| **active)
            .map(|(output, _)| output.r#type().tangent())
            .collect::<Result<Vec<_>, _>>()?;
        let tangent_operation = tangent_operation
            .with_global_output_types(tangent_output_types)
            .map_err(trace_error_from_shard_map)?;

        // Bind the tangent `shard_map` over the active operand tangents followed by the residual values, recovering
        // one output tangent per primal output. The tangent `shard_map` takes every active operand tangent as a real
        // program input, so materialize structural zeros at this sub-program boundary; an inactive operand (a
        // plumbing reference or a zero-space operand) has no tangent boundary slot.
        let mut tangent_operands = inputs
            .iter()
            .zip(&activity)
            .filter(|(_, active)| **active)
            .map(|(input, _)| input.tangent().clone().materialize(context.tangent()))
            .collect::<Result<Vec<_>, _>>()?;
        tangent_operands.extend(
            residuals.into_iter().map(|value| context.primal_to_tangent(value)).collect::<Result<Vec<_>, _>>()?,
        );
        let tangent_operation = XlaOperation::ShardMap(Box::new(tangent_operation));
        let tangent_outputs =
            context.tangent().bind(tangent_operation, vec![tangent_body_program], &tangent_operands)?;
        check_count!("output", tangent_outputs, output_activity.iter().filter(|active| **active).count(), ProgramError);
        let mut tangent_outputs = tangent_outputs.into_iter();
        primal_outputs
            .into_iter()
            .zip(output_activity)
            .map(|(primal, active)| {
                if active {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
                } else {
                    DifferentiationDual::new_with_zero_tangent(primal)
                }
            })
            .collect()
    }
}

/// Transposes a tangent [`ShardMapOperation`] while retaining its manual region boundary.
///
/// The [`TranspositionDriver`] transposes the attached body under the operands' linearity and cotangent-destination
/// masks. Known operands supply residual values and may appear anywhere in the input boundary. The reverse boundary
/// dualizes value shardings and forwards sharded cotangent references under their original input shardings.
/// Reference outputs forward input roots and therefore have no separate output-cotangent slot.
///
/// Replicated primal references are read-only. Their reverse contributions accumulate in fresh per-device references,
/// reduce across replicated axes, and update the caller's cotangent destination once outside the map. Replicated output
/// seeds are normalized across their copies before transposition, so the reductions produce one global cotangent.
/// Returned cotangents follow the original input order; reference, known, and ignored inputs receive structural zeros.
///
/// # Parameters
///
///   - `operation`: Primal tangent `shard_map` staged into the tangent program.
///   - `context`: Active transpose tracing context the pullback is staged into.
///   - `driver`: Instruction-scoped access to the attached body region and its recursive transposition machinery.
///   - `inputs`: Per-operand [`PartialValue`] knowledge, mirroring the body's global inputs one-to-one. The
///     [`Unknown`](PartialValue::Unknown) entries are the input tangents; the [`Known`](PartialValue::Known) entries
///     carry the residual tracers the pullback reads.
///   - `outputs`: Symbolic cotangents for the tangent `shard_map`'s outputs.
///   - `cotangents`: Cotangent destinations of the operands (refer to the documentation of
///     [`operand_cotangents`]). The body is transposed with their destination kinds, so a live
///     (`Reference`-kind) reference operand accumulates into its destination using the sharded or replicated policy
///     above. A dead (`Ignore`-kind) reference operand has no slot in the transposed body.
pub fn transpose_primal_shard_map<
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    D: TranspositionDriver<V, XlaOperation<V>>,
>(
    operation: &ShardMapOperation<V>,
    context: &mut TracingContext<V, XlaOperation<V>>,
    driver: &D,
    inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
    cotangents: &OperandCotangents<Tracer<TracingContext<V, XlaOperation<V>>>>,
) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, ProgramError> {
    let operand_linear = inputs.iter().map(PartialValue::is_unknown).collect::<Vec<_>>();
    check_count!("input", operand_linear, operation.input_types.len(), ProgramError);
    check_count!("input", cotangents.destination_kinds(), inputs.len(), ProgramError);

    // A shard_map with no live output cotangents and no live reference operand is a zero linear map, so every operand
    // cotangent is zero. A live reference operand keeps the shard map live, because its accumulated state cotangent
    // flows through the transposed body even when no ordinary output cotangent does.
    if outputs.iter().all(MaybeZero::is_zero)
        && !cotangents.is_live()
        && !driver.region(0)?.has_observable_transpose_effects()
    {
        return inputs
            .iter()
            .map(|input| input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from))
            .collect();
    }
    operation.validate_reference_body(driver.region(0)?)?;

    // Transpose the tangent body's flat program under the same per-operand linearity mask and destination kinds,
    // through the body region's retained transform cache so that a body shared by several programs is transposed once
    // per mask. The transposed program maps `[non_reference_output_cotangents..., cotangent_references...,
    // known_input_values...]` to `[linear_input_cotangents...]`, in body-input order on each side, where a live
    // reference input's cotangent is its cotangent reference itself and a dead reference input has no cotangent slot;
    // re-wrap it as a transposed shard-map boundary whose shardings are permuted to match.
    let (transposed_operation, transposed_body_program) =
        transpose_shard_map_body(operation, driver, operand_linear.as_slice(), cotangents.destination_kinds())?;

    // Stage the output cotangents, materializing a typed zero for each structurally zero cotangent, then stage a fresh
    // `shard_map` over the transposed body on `[outputs..., cotangent_references..., known_input_values...]`. Its
    // outputs are the linear-input cotangents.
    let output_types = &operation.output_types;
    check_count!("output", outputs, output_types.len(), ProgramError);
    let mut operands = Vec::with_capacity(output_types.len() + inputs.len());
    for (cotangent, output_type) in outputs.iter().zip(output_types.iter()) {
        // A reference output forwards a body input root whose state cotangent lives in that operand's cotangent
        // reference, so it owns no cotangent slot.
        if output_type.is_reference() {
            continue;
        }
        let output_type = output_type.cotangent()?;
        operands.push(materialize_transpose_cotangent(context, cotangent, &output_type, inputs)?);
    }
    let mut reference_destinations = cotangents.references().iter();
    for (index, kind) in cotangents.destination_kinds().iter().enumerate() {
        if *kind == CotangentDestinationKind::Reference {
            let reference = reference_destinations.next().unwrap();
            if operation.shard_map.input_replicated_manual_axes(index).is_empty() {
                operands.push(reference.clone());
            }
        }
    }
    operands.extend(inputs.iter().filter_map(PartialValue::as_known).cloned());
    let transposed_operation = XlaOperation::ShardMap(Box::new(transposed_operation));

    // The transposed body is attached as the shared handle the driver returned, so repeated binds of one transposed
    // body intern by `Arc` identity instead of copying the program into the trace again.
    let input_cotangents = context.stage_operation(
        transposed_operation,
        CalleeRegionDriver::new(&[transposed_body_program]),
        operands.as_slice(),
    )?;
    let output_count = (0..inputs.len())
        .filter(|&index| {
            operand_linear[index]
                && (cotangents.returns_cotangent(index)
                    || (cotangents.kind(index) == CotangentDestinationKind::Reference
                        && !operation.shard_map.input_replicated_manual_axes(index).is_empty()))
        })
        .count();
    check_count!("output", input_cotangents, output_count, ProgramError);

    // Reassemble one cotangent per operand: the known operands carry structural zeros, while the linear input tangents
    // receive the transposed `shard_map`'s outputs in body-input order. A live reference tangent's output is its
    // cotangent reference, whose contents were accumulated in place, and a dead one has no output, so every reference
    // operand receives a structural zero.
    let mut input_cotangents = input_cotangents.into_iter();
    let mut reference_destinations = cotangents.references().iter();
    operand_linear
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(index, (&linear, input))| match cotangents.kind(index) {
            CotangentDestinationKind::Return if linear => Ok(MaybeZero::Value(input_cotangents.next().unwrap())),
            CotangentDestinationKind::Reference => {
                let destination = reference_destinations.next().unwrap();
                if cotangents.is_reference(index) || !operation.shard_map.input_replicated_manual_axes(index).is_empty()
                {
                    let cotangent = input_cotangents.next().unwrap();
                    if !operation.shard_map.input_replicated_manual_axes(index).is_empty() {
                        context.bind(
                            ReferenceAddUpdateOperation::new(),
                            Vec::new(),
                            &[destination.clone(), cotangent],
                        )?;
                    }
                }
                input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from)
            }
            CotangentDestinationKind::Return | CotangentDestinationKind::Ignore => {
                input.r#type().cotangent().map(MaybeZero::Zero).map_err(ProgramError::from)
            }
        })
        .collect()
}

// Transpose rule for a primal tangent [`ShardMapOperation`], forwarding to [`transpose_primal_shard_map`] with the
// cotangent references of its linear reference-typed operands resolved (and allocated on first use) through the
// enclosing [`TranspositionContext`]. The body transposition happens on the body's flat [`XlaConstant`]-keyed
// program, so the recursion is resolved once at definition time and instantiating this implementation introduces no
// recursive [`TransposableOperation`] obligation on [`XlaOperation`].
impl<V> TransposableOperation<V, XlaOperation<V>> for ShardMapOperation<V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
{
    fn transpose<D: TranspositionDriver<V, XlaOperation<V>>>(
        &self,
        context: &mut TranspositionContext<'_, V, XlaOperation<V>>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, XlaOperation<V>>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>],
        accumulators: &[ryft_core::CotangentAccumulator],
    ) -> Result<(), DifferentiationError> {
        check_count!("output", outputs, self.output_types.len(), ProgramError);
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        let contributions =
            (|| -> Result<Vec<MaybeZero<Tracer<TracingContext<V, XlaOperation<V>>>>>, DifferentiationError> {
                let cotangents = operand_cotangents(context, inputs, accumulators)?;
                transpose_primal_shard_map(self, context, driver, inputs, outputs, &cotangents)
                    .map_err(DifferentiationError::from)
            })()?;
        check_count!("input", contributions, accumulators.len(), ProgramError);
        for (accumulator, contribution) in accumulators.iter().zip(contributions) {
            accumulator.accumulate(context, contribution)?;
        }
        Ok(())
    }
}

/// Transposes one tangent shard-map body into the reverse boundary and body consumed by
/// [`transpose_primal_shard_map`].
///
/// The tangent body's flat program is transposed with [`TranspositionDriver::transpose_program`] under
/// `input_linearity` and `destination_kinds`, producing a shared program mapping
/// `[non_reference_output_cotangents..., cotangent_references..., known_input_values...]` to
/// `[linear_input_cotangents...]`, where a `Reference`-kind input's cotangent is its cotangent reference forwarded by
/// identity and an `Ignore`-kind input has no cotangent slot. The transposed boundary permutes and dualizes the
/// original one to match: its global inputs are the cotangent descriptors of the original value outputs under the
/// cotangent duals of their output shardings, then the cotangent references `ref<cotangent(T)>` of the
/// `Reference`-kind inputs under those inputs' own input shardings, then the known operands' original global inputs;
/// its global outputs are the cotangent descriptors of the linear operands' original global inputs, dualizing the
/// input sharding of a value input and keeping the input sharding of a forwarded cotangent reference.
///
/// # Parameters
///
///   - `operation`: Boundary metadata of the tangent `shard_map` produced by [`shard_map_bodies`], whose global
///     inputs are `[active(input_tangents)..., residuals...]` and whose global outputs are `[output_tangents...]`.
///   - `driver`: Instruction-scoped access to the attached body region and its recursive transposition machinery.
///   - `input_linearity`: Per-input linearity flags over the tangent boundary's global inputs.
///   - `destination_kinds`: Per-input cotangent destination kinds over the tangent boundary's global inputs.
#[allow(clippy::type_complexity)]
fn transpose_shard_map_body<
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
    D: TranspositionDriver<V, XlaOperation<V>>,
>(
    operation: &ShardMapOperation<V>,
    driver: &D,
    input_linearity: &[bool],
    destination_kinds: &[CotangentDestinationKind],
) -> Result<(ShardMapOperation<V>, Arc<Program<V, XlaOperation<V>, Vec<V>, Vec<V>>>), ProgramError> {
    check_count!("input", input_linearity, operation.input_types.len(), ProgramError);
    check_count!("input", destination_kinds, operation.input_types.len(), ProgramError);
    let mut transposed_program = driver.transpose_program(driver.region(0)?, input_linearity, destination_kinds)?;
    let shard_map = operation.shard_map();
    let replicated_destinations = destination_kinds
        .iter()
        .enumerate()
        .map(|(index, kind)| {
            *kind == CotangentDestinationKind::Reference && !shard_map.input_replicated_manual_axes(index).is_empty()
        })
        .collect::<Vec<_>>();

    // Replicated primal references are read-only. Each device accumulates its contribution into fresh local state,
    // then reduces that value across the replicated axes. Only the enclosing map updates the caller's destination,
    // once, so existing destination contents are neither duplicated nor used as a per-device seed.
    let output_seed_axes = operation
        .output_types
        .iter()
        .enumerate()
        .filter(|(_, r#type)| !r#type.is_reference())
        .map(|(index, r#type)| {
            // Transpose keeps zero-space seeds in its flat boundary to preserve numbering, but they carry no
            // numerical contribution and must not enter replica normalization arithmetic.
            Ok(if r#type.cotangent()?.is_zero_space() {
                Vec::new()
            } else {
                shard_map.output_replicated_manual_axes(index)
            })
        })
        .collect::<Result<Vec<_>, TypeError>>()?;
    let needs_reduction = input_linearity.iter().enumerate().any(|(index, linear)| {
        *linear
            && destination_kinds[index] != CotangentDestinationKind::Ignore
            && !shard_map.input_replicated_manual_axes(index).is_empty()
    });
    if needs_reduction || output_seed_axes.iter().any(|axes| !axes.is_empty()) {
        let mut builder = ProgramBuilder::new();
        let local_destinations = destination_kinds
            .iter()
            .zip(&replicated_destinations)
            .filter_map(|(kind, replicated)| (*kind == CotangentDestinationKind::Reference).then_some(*replicated))
            .collect::<Vec<_>>();
        let mut input_count = 0;
        let mut operands = Vec::new();
        for (position, r#type) in transposed_program.input_types().into_iter().enumerate() {
            if position
                .checked_sub(output_seed_axes.len())
                .and_then(|index| local_destinations.get(index))
                .copied()
                .unwrap_or(false)
            {
                let referent = <&ReferenceType<ArrayType>>::try_from(&r#type)?.referent().clone();
                let zero = builder.add_instruction(
                    XlaOperation::Array(ArrayOperation::Zero(ZeroOperation::new(referent))),
                    Vec::new(),
                    Vec::new(),
                    None,
                )?[0];
                operands.push(builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None)?[0]);
            } else {
                let input = builder.add_input(r#type.clone());
                input_count += 1;
                let axes = output_seed_axes.get(position).map(Vec::as_slice).unwrap_or(&[]);
                if axes.is_empty() {
                    operands.push(input);
                } else {
                    // A replicated output denotes one global value, so distribute its seed across its copies before
                    // summing contributions at replicated input boundaries.
                    let count =
                        axes.iter().map(|axis| shard_map.mesh().axis_size(axis).unwrap() as f64).product::<f64>();
                    let r#type = boundary_array_type(&r#type)?;
                    let literal = CpuArray::scalar(count).convert_element_type(r#type.data_type())?;
                    let denominator =
                        builder.add_instruction(ConstantOperation::new(literal), Vec::new(), Vec::new(), None)?[0];
                    let denominator = builder.add_instruction(
                        XlaOperation::Array(ArrayOperation::Broadcast(BroadcastOperation::new(
                            r#type.clone(),
                            Vec::new(),
                        ))),
                        Vec::new(),
                        vec![denominator],
                        None,
                    )?[0];
                    operands.push(
                        builder.add_instruction(
                            XlaOperation::Array(ArrayOperation::Div(DivOperation::new())),
                            Vec::new(),
                            vec![input, denominator],
                            None,
                        )?[0],
                    );
                }
            }
        }
        let mut outputs = builder.splice_program(&transposed_program, &operands)?.into_iter();
        let mut rebuilt_outputs = Vec::new();
        let mut destination_index = 0;
        for (index, linear) in input_linearity.iter().enumerate() {
            let destination = if destination_kinds[index] == CotangentDestinationKind::Reference {
                let destination = operands[output_seed_axes.len() + destination_index];
                destination_index += 1;
                Some(destination)
            } else {
                None
            };
            if !linear
                || destination_kinds[index] == CotangentDestinationKind::Ignore
                || (destination_kinds[index] == CotangentDestinationKind::Reference
                    && !operation.input_types[index].is_reference()
                    && !replicated_destinations[index])
            {
                continue;
            }
            // Ordinary buffers produce no identity output. A replicated local buffer must nevertheless be frozen
            // and reduced after the body, so retrieve its local argument rather than consuming a body output.
            let mut output = if destination_kinds[index] == CotangentDestinationKind::Reference
                && !operation.input_types[index].is_reference()
            {
                destination.unwrap()
            } else {
                outputs.next().unwrap()
            };
            if replicated_destinations[index] {
                output = builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![output], None)?[0];
            }
            if replicated_destinations[index] || destination_kinds[index] == CotangentDestinationKind::Return {
                for axis in shard_map.input_replicated_manual_axes(index) {
                    output = builder.add_instruction(
                        XlaOperation::Array(ArrayOperation::ParallelReduce(ParallelReduceOperation::new(
                            axis,
                            ParallelReductionKind::Sum,
                        ))),
                        Vec::new(),
                        vec![output],
                        None,
                    )?[0];
                }
            }
            rebuilt_outputs.push(output);
        }
        let output_count = rebuilt_outputs.len();
        transposed_program = Arc::new(builder.build(
            rebuilt_outputs,
            vec![Placeholder; input_count],
            vec![Placeholder; output_count],
        )?);
    }

    // The transposed inputs: the cotangents of the value outputs, the cotangent references of the live reference
    // inputs, and the known inputs, in that order.
    let mut in_shardings = Vec::new();
    let mut global_input_types = Vec::new();
    for (output_type, sharding) in operation.output_types.iter().zip(shard_map.out_shardings()) {
        if output_type.is_reference() {
            continue;
        }
        in_shardings.push(sharding.cotangent());
        global_input_types.push(output_type.cotangent()?);
    }
    let mut destination_positions = vec![None; input_linearity.len()];
    for (index, kind) in destination_kinds.iter().enumerate() {
        if *kind != CotangentDestinationKind::Reference || replicated_destinations[index] {
            continue;
        }
        destination_positions[index] = Some(global_input_types.len());
        in_shardings.push(shard_map.in_shardings()[index].clone());
        let cotangent_type = operation.input_types[index].cotangent()?;
        global_input_types.push(if operation.input_types[index].is_reference() {
            cotangent_type
        } else {
            ReferenceType::new(boundary_array_type(&cotangent_type)?.clone()).into()
        });
    }
    for (index, _) in input_linearity.iter().enumerate().filter(|(_, linear)| !**linear) {
        in_shardings.push(shard_map.in_shardings()[index].clone());
        global_input_types.push(operation.input_types[index].clone());
    }

    // The transposed outputs: one cotangent per linear input in input order, a value for a `Return` destination and
    // the forwarded cotangent reference for a `Reference` destination; an `Ignore` destination has no output.
    let mut out_shardings = Vec::new();
    let mut global_output_types = Vec::new();
    let mut output_forwarding = Vec::new();
    for (index, _) in input_linearity.iter().enumerate().filter(|(_, linear)| **linear) {
        match destination_kinds[index] {
            CotangentDestinationKind::Return => {
                out_shardings.push(shard_map.in_shardings()[index].cotangent());
                global_output_types.push(operation.input_types[index].cotangent()?);
                output_forwarding.push(None);
            }
            CotangentDestinationKind::Reference
                if operation.input_types[index].is_reference() || replicated_destinations[index] =>
            {
                out_shardings.push(shard_map.in_shardings()[index].clone());
                let cotangent_type = operation.input_types[index].cotangent()?;
                if replicated_destinations[index] {
                    global_output_types.push(boundary_array_type(&cotangent_type)?.clone().into());
                    output_forwarding.push(None);
                } else {
                    global_output_types.push(cotangent_type);
                    output_forwarding.push(destination_positions[index]);
                }
            }
            CotangentDestinationKind::Reference | CotangentDestinationKind::Ignore => {}
        }
    }
    let shard_map = ShardMap::from_shardings(
        shard_map.mesh().clone(),
        in_shardings,
        out_shardings,
        shard_map.manual_axes().to_vec(),
        shard_map.check_vma(),
    );
    let operation = ShardMapOperation::from_boundary(shard_map, global_input_types, global_output_types)
        .with_output_forwarding(output_forwarding)
        .map_err(trace_error_from_shard_map)?;
    Ok((operation, transposed_program))
}

fn trace_error_from_shard_map(error: ShardMapTraceError) -> ProgramError {
    ProgramError::Type(TypeError::invalid(error.to_string()))
}

fn trace_flat_shard_map<
    F: FnOnce(ShardMapLocalTraceInput<Input>) -> ShardMapLocalTraceOutput<Output>,
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
>(
    function: F,
    global_input_types: Input,
    mesh: LogicalMesh,
    in_specs: Input::To<Sharding>,
    out_specs: Output::To<Sharding>,
    manual_axes: Vec<String>,
    check_vma: bool,
) -> Result<FlatTracedShardMap, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<Sharding>
        + ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<Sharding>
        + ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
{
    let shard_map = ShardMap::new(
        mesh,
        in_specs.into_parameters().collect::<Vec<_>>(),
        out_specs.into_parameters().collect::<Vec<_>>(),
        manual_axes,
        check_vma,
    )?;
    Ok(FlatTracedShardMap::from_traced(&shard_map.trace::<F, Input, Output>(function, global_input_types)?))
}

fn apply_traced_shard_map<C>(
    context: C,
    traced: FlatTracedShardMap,
    traced_inputs: Vec<C::Value>,
) -> Result<Vec<C::Value>, ShardMapTraceError>
where
    C: Context<Type = ArrayIrType, Constant = XlaConstant, Operation = XlaOperation>,
{
    let (operation, body_program) = ShardMapOperation::from_body(traced);
    Ok(context.bind(XlaOperation::ShardMap(Box::new(operation)), vec![body_program], traced_inputs.as_slice())?)
}

fn global_input_types_from_traced_inputs<V, Input>(
    traced_inputs: &Input,
) -> Result<Input::To<ArrayType>, ShardMapTraceError>
where
    V: Value<Type = ArrayType>,
    Input: Parameterized<V>,
    Input::Family: ParameterizedFamily<ArrayType>,
{
    Ok(Input::To::<ArrayType>::from_parameters(
        traced_inputs.parameter_structure(),
        traced_inputs.parameters().map(|input| input.r#type().into_owned()).collect::<Vec<_>>(),
    )?)
}

fn reparameterize_shardings<Source: Parameterized<Sharding>, Target: Parameterized<Sharding>>(
    specs: Source,
    target_structure: Target::ParameterStructure,
) -> Result<Target, ShardMapTraceError> {
    Ok(Target::from_parameters(target_structure, specs.into_parameters().collect::<Vec<_>>())?)
}

impl ShardMapInvocationLeaf for ArrayType {
    type Return<Input: Parameterized<Self>, Output: Parameterized<ArrayType>>
        = TracedShardMap<Input, Output>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>;

    fn invoke<F, Input, Output>(
        function: F,
        inputs: Input,
        mesh: LogicalMesh,
        in_specs: Input::To<Sharding>,
        out_specs: Output::To<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Result<Self::Return<Input, Output>, ShardMapTraceError>
    where
        Input: Parameterized<Self>,
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    {
        let shard_map = ShardMap::new(
            mesh,
            in_specs.into_parameters().collect::<Vec<_>>(),
            out_specs.into_parameters().collect::<Vec<_>>(),
            manual_axes,
            check_vma,
        )?;
        shard_map.trace(
            |local_inputs: ShardMapLocalTraceInput<Input>| {
                let adapted_inputs = ShardMapLocalTraceInput::<Input::To<ArrayType>>::from_parameters(
                    local_inputs.parameter_structure(),
                    local_inputs.into_parameters().collect::<Vec<_>>(),
                )
                .expect("array-typed shard_map inputs should preserve their canonical tracer structure");
                function(adapted_inputs)
            },
            inputs,
        )
    }
}

// Invokes a traced shard map through the composite value behind one public array projection.
impl<V> ShardMapInvocationLeaf for ProjectedValue<ArrayType, V>
where
    V: Value<Type = ArrayIrType> + ValueProjection<ArrayType, Projected = ProjectedValue<ArrayType, V>>,
    ProjectedValue<ArrayType, V>: Value<Type = ArrayType>,
    V::DispatchDomain: Context<Type = ArrayIrType, Constant = XlaConstant, Operation = XlaOperation>,
{
    type Return<Input: Parameterized<Self>, Output: Parameterized<ArrayType>>
        = Output::To<Self>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Self>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>;

    fn invoke<F, Input, Output>(
        function: F,
        inputs: Input,
        mesh: LogicalMesh,
        in_specs: Input::To<Sharding>,
        out_specs: Output::To<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Result<Self::Return<Input, Output>, ShardMapTraceError>
    where
        Input: Parameterized<Self>,
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output: Parameterized<ArrayType>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Self>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    {
        let output_structure = out_specs.parameter_structure();
        let global_input_types = global_input_types_from_traced_inputs::<Self, _>(&inputs)?;
        let global_in_specs = reparameterize_shardings::<
            Input::To<Sharding>,
            <Input::To<ArrayType> as Parameterized<ArrayType>>::To<Sharding>,
        >(in_specs, global_input_types.parameter_structure())?;
        let traced_inputs = inputs.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>();
        let context = match traced_inputs.first() {
            Some(input) => input.dispatch_domain(),
            None if output_structure.parameter_count() == 0 => {
                return Ok(Output::To::<Self>::from_parameters(output_structure, Vec::new())?);
            }
            None => return Err(ShardMapTraceError::MissingTracedInvocationDomain),
        };
        let traced = trace_flat_shard_map::<F, Input::To<ArrayType>, Output>(
            function,
            global_input_types,
            mesh,
            global_in_specs,
            out_specs,
            manual_axes,
            check_vma,
        )?;
        let outputs = apply_traced_shard_map(context, traced, traced_inputs)?
            .into_iter()
            .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Output::To::<Self>::from_parameters(output_structure, outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        AddOperation, ArrayIrType, ArrayOperation, ArrayType, CaptureReference, Context, CotangentDestinationKind,
        DataType, DifferentiableType, DifferentiationError, Dimension, DimensionBounds, DimensionType,
        DimensionVariable, DomainTracingContext, EffectClasses, LogicalMesh, MaybeZero, MeshAxis, MeshAxisType,
        MulOperation, OperandCotangents, Operation, PartialValue, Placeholder, Program, ProgramBuilder, ProgramError,
        ReferenceAddUpdateOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceSource, ReferenceType,
        RegionDriver, RegionInterface, RegionRef, Shape, Sharding, ShardingDimension, StagingContext, TracingContext,
        TransposableOperation, TranspositionContext, TranspositionDriver, TypeError, Typed, ZeroOperation,
    };

    use crate::experimental::domains::XlaDomain;
    use crate::experimental::ops::{XlaArrayConstant, XlaConstant, XlaOperation, XlaProgram, XlaProgramBuilder};
    use crate::experimental::shard_map::{FlatTracedShardMap, ShardMap, ShardMapTraceError};

    use super::{ShardMapOperation, transpose_primal_shard_map, transpose_shard_map_body};

    /// Test-only driver that returns a predetermined transpose for its one attached source region.
    struct TestTranspositionDriver {
        /// Source region exposed to the operation rule.
        source: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,

        /// Predetermined transposed program returned by the recursive request.
        transposed: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
    }

    impl RegionDriver<XlaConstant, XlaOperation> for TestTranspositionDriver {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, XlaConstant, XlaOperation>>
        where
            XlaConstant: 'r,
            XlaOperation: 'r,
        {
            std::iter::once(self.source.entry_region_ref())
        }
    }

    impl TranspositionDriver<XlaConstant, XlaOperation> for TestTranspositionDriver {
        fn transpose_program(
            &self,
            _region: RegionRef<'_, XlaConstant, XlaOperation>,
            _input_linearity: &[bool],
            _destination_kinds: &[CotangentDestinationKind],
        ) -> Result<Arc<Program<XlaConstant, XlaOperation, Vec<XlaConstant>, Vec<XlaConstant>>>, DifferentiationError>
        {
            Ok(Arc::new(self.transposed.clone()))
        }
    }

    fn test_array_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn single_input_test_shard_map() -> ShardMap {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 0)],
            vec![Sharding::replicated(mesh, 0)],
            vec!["x".to_string()],
            true,
        )
    }

    /// Builds a one-instruction program staging an identity `shard_map` over a two-manual-axis mesh, whose boundary
    /// differs from another such program only in the provided manual-axis selection and `check_vma` flag.
    fn metadata_fingerprint_program(
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>> {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let array_type = test_array_type();
        let shard_map = ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 0)],
            vec![Sharding::replicated(mesh, 0)],
            manual_axes,
            check_vma,
        );
        let operation = ShardMapOperation::<XlaConstant>::from_boundary(
            shard_map,
            vec![array_type.clone()],
            vec![array_type.clone()],
        );

        let body = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(array_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.into());
        let body_region = builder.import_program(body);
        let output = builder
            .add_instruction(XlaOperation::ShardMap(Box::new(operation)), vec![body_region], vec![input], None)
            .unwrap()[0];
        builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap()
    }

    /// Pins the metadata-fingerprint contract of [`Operation::render`] for `shard_map`. The manual SPMD boundary
    /// metadata steers differentiation, transposition, and `sdy.manual_computation` lowering, yet none of it is
    /// visible in the instruction's rendered atom types. Rendered *inequality* is exactly what the
    /// debug transform-cache diagnostic consumes: it compares a retained transform artifact against a freshly derived
    /// one purely by rendering, so two boundaries that differ semantically must never render alike.
    #[test]
    fn test_shard_map_render_fingerprints_boundary_metadata() {
        let baseline = metadata_fingerprint_program(vec!["x".to_string()], true);

        // The complete boundary metadata renders as deterministic operation fields beside the attached body region.
        assert_eq!(
            baseline.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = shard_map [
                    mesh=['x'=2:manual, 'y'=2:manual],
                    in_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, []}],
                    out_shardings=[{mesh<['x'=2:manual, 'y'=2:manual]>, []}],
                    manual_axes=['x'],
                    check_vma=true,
                    global_input_types=[f32[]],
                    global_output_types=[f32[]],
                ] %0 [
                    body={
                        lambda %0:f32[] .
                        in (%0)
                    },
                ]
                in (%1)"},
        );

        // Two boundaries differing only in `check_vma`, and two differing only in the active manual-axis subset,
        // must render differently even though their types and bodies are identical.
        assert_ne!(baseline.to_string(), metadata_fingerprint_program(vec!["x".to_string()], false).to_string());
        assert_ne!(
            baseline.to_string(),
            metadata_fingerprint_program(vec!["x".to_string(), "y".to_string()], true).to_string(),
        );

        // Equal metadata still renders equally, so the inequalities above isolate the metadata itself.
        assert_eq!(baseline.to_string(), metadata_fingerprint_program(vec!["x".to_string()], true).to_string());
    }

    /// Nothing restricts the characters of a mesh axis name: [`MeshAxis::new`] only rejects empty names. Rendering
    /// names verbatim would therefore break the fingerprint contract pinned above, because one axis named `a', 'b`
    /// renders exactly like the two axes `a` and `b`, letting the debug recheck accept a corrupted boundary.
    #[test]
    fn test_shard_map_render_escapes_axis_names() {
        // Boundaries over one fixed mesh that owns every adversarial name, so only `manual_axes` varies below.
        fn rendered_manual_axes(manual_axes: Vec<&str>) -> String {
            let mesh = LogicalMesh::new(
                ["a", "b", "a', 'b", "a\\", "a\\', 'b", "a\nb", "a\\nb"]
                    .into_iter()
                    .map(|name| MeshAxis::new(name, 2, MeshAxisType::Manual).unwrap())
                    .collect(),
            )
            .unwrap();
            let array_type = test_array_type();
            let shard_map = ShardMap::from_shardings(
                mesh.clone(),
                vec![Sharding::replicated(mesh.clone(), 0)],
                vec![Sharding::replicated(mesh, 0)],
                manual_axes.into_iter().map(str::to_string).collect(),
                true,
            );
            ShardMapOperation::<XlaConstant>::from_boundary(shard_map, vec![array_type.clone()], vec![array_type])
                .to_string()
        }

        // Boundaries with no shardings and no boundary types, so only the `mesh` field carries axis names.
        fn rendered_mesh(axis_names: Vec<&str>) -> String {
            let mesh = LogicalMesh::new(
                axis_names.into_iter().map(|name| MeshAxis::new(name, 2, MeshAxisType::Manual).unwrap()).collect(),
            )
            .unwrap();
            let shard_map = ShardMap::from_shardings(mesh, Vec::new(), Vec::new(), Vec::new(), true);
            let boundary = Vec::<ArrayIrType>::new();
            ShardMapOperation::<XlaConstant>::from_boundary(shard_map, boundary.clone(), boundary).to_string()
        }

        // A quote inside a name must not be able to imitate the separator between two rendered names, in either the
        // manual-axis list or the mesh, where the name is additionally followed by its size and type.
        assert_ne!(rendered_manual_axes(vec!["a", "b"]), rendered_manual_axes(vec!["a', 'b"]));
        assert_ne!(rendered_mesh(vec!["a", "b"]), rendered_mesh(vec!["a'=2:manual, 'b"]));

        // Backslashes are escaped as well, so a name ending in one cannot turn the quote that closes it into an
        // escaped quote, and control characters are escaped so that they cannot imitate their own escape codes.
        assert_ne!(rendered_manual_axes(vec!["a\\", "b"]), rendered_manual_axes(vec!["a\\', 'b"]));
        assert_ne!(rendered_manual_axes(vec!["a\nb"]), rendered_manual_axes(vec!["a\\nb"]));

        // The escaped form itself is Rust's canonical character escaping, inside the surrounding single quotes.
        assert_eq!(
            rendered_mesh(vec!["a'b", "c\\d", "e\nf"]),
            indoc! {r#"
                shard_map [
                    mesh=['a\'b'=2:manual, 'c\\d'=2:manual, 'e\nf'=2:manual],
                    in_shardings=[],
                    out_shardings=[],
                    manual_axes=[],
                    check_vma=true,
                    global_input_types=[],
                    global_output_types=[],
                ]"#},
        );
    }

    #[test]
    fn test_shard_map_composite_boundary_is_array_only() {
        let array_type = ArrayType::scalar(DataType::F32);
        let composite_array_type = ArrayIrType::Array(array_type.clone());
        let operation = ShardMapOperation::<XlaArrayConstant>::from_boundary(
            single_input_test_shard_map(),
            vec![array_type.clone()],
            vec![array_type],
        );
        let array_body = RegionInterface::new(
            vec![composite_array_type.clone()],
            vec![composite_array_type.clone()],
            EffectClasses::NONE,
        );

        assert_eq!(
            operation
                .infer_output_types(std::slice::from_ref(&composite_array_type), std::slice::from_ref(&array_body)),
            Ok(vec![composite_array_type.clone()]),
        );
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&composite_array_type), &[]),
            Err(TypeError::invalid("expected 1 region but got 0")),
        );

        let dimension_type = ArrayIrType::Dimension(DimensionType::new(DimensionVariable::new(
            "size",
            DimensionBounds::positive(Some(8)).unwrap(),
        )));
        assert_eq!(
            operation.infer_output_types(std::slice::from_ref(&dimension_type), std::slice::from_ref(&array_body)),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );

        let dimension_input_body =
            RegionInterface::new(vec![dimension_type.clone()], vec![composite_array_type.clone()], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&composite_array_type),
                std::slice::from_ref(&dimension_input_body),
            ),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );

        let dimension_output_body =
            RegionInterface::new(vec![composite_array_type.clone()], vec![dimension_type], EffectClasses::NONE);
        assert_eq!(
            operation.infer_output_types(
                std::slice::from_ref(&composite_array_type),
                std::slice::from_ref(&dimension_output_body),
            ),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_shard_map_jvp_uses_tangent_boundary_descriptors() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let boundary_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(4)]));
        let ambient_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let ambient_input_type = boundary_type.clone().with_sharding(ambient_sharding).unwrap();
        let ambient_tangent_type = ambient_input_type.tangent().unwrap();
        let boundary_tangent_type = boundary_type.tangent().unwrap();

        let body = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(boundary_type.clone().into());
            let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, input], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let shard_map = ShardMap::from_shardings(
            mesh.clone(),
            vec![Sharding::replicated(mesh.clone(), 1)],
            vec![Sharding::replicated(mesh, 1)],
            vec!["x".to_string()],
            true,
        );
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![boundary_type.clone()], vec![boundary_type.clone()]);
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(ambient_input_type.into());
        let body_region = builder.import_program(body);
        let output = builder
            .add_instruction(XlaOperation::ShardMap(Box::new(operation)), vec![body_region], vec![input], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let fused = program.jvp().unwrap();
        let shard_maps = fused
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                XlaOperation::ShardMap(operation) => Some((instruction, operation.as_ref())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(shard_maps.len(), 2);
        let (primal_instruction, primal_operation) = shard_maps[0];
        let (tangent_instruction, tangent_operation) = shard_maps[1];

        assert!(primal_operation.global_output_types().len() > 1);
        assert_eq!(tangent_operation.global_input_types()[0], ArrayIrType::Array(boundary_tangent_type.clone()));
        assert_eq!(&tangent_operation.global_input_types()[1..], &primal_operation.global_output_types()[1..],);
        assert_eq!(tangent_operation.global_output_types(), &[ArrayIrType::Array(ambient_tangent_type)]);

        let primal_body = fused.region_ref(primal_instruction.regions()[0]).unwrap().to_program();
        let tangent_body = fused.region_ref(tangent_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(primal_body.input_types(), vec![ArrayIrType::Array(boundary_type)]);
        assert_eq!(tangent_body.input_types()[0], ArrayIrType::Array(boundary_tangent_type));
        assert_eq!(&tangent_body.input_types()[1..], &primal_body.output_types()[1..]);
        assert_eq!(
            tangent_body.output_types(),
            vec![ArrayIrType::Array(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]),))]
        );
    }

    #[test]
    fn test_shard_map_transpose_dualizes_boundary_descriptors() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(tangent_sharding.clone())
            .unwrap();
        let cotangent_type = tangent_type.cotangent().unwrap();

        let source = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(tangent_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let transposed = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(cotangent_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let shard_map = ShardMap::from_shardings(
            mesh,
            vec![tangent_sharding.clone()],
            vec![tangent_sharding.clone()],
            Vec::new(),
            true,
        );
        let operation = ShardMapOperation::from_boundary(shard_map, vec![tangent_type.clone()], vec![tangent_type]);

        let (transposed, _) =
            transpose_shard_map_body(&operation, &driver, &[true], &[CotangentDestinationKind::Return]).unwrap();
        assert_eq!(transposed.global_input_types(), &[ArrayIrType::Array(cotangent_type.clone())]);
        assert_eq!(transposed.global_output_types(), &[ArrayIrType::Array(cotangent_type)]);
        assert_eq!(transposed.output_forwarding(), &[None]);
        assert_eq!(transposed.shard_map().in_shardings(), &[tangent_sharding.cotangent()]);
        assert_eq!(transposed.shard_map().out_shardings(), &[tangent_sharding.cotangent()]);
    }

    #[test]
    fn test_shard_map_transpose_preserves_zero_space_seed_positions() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 0);
        let value_type = ArrayType::scalar(DataType::F32).with_sharding(sharding.clone()).unwrap();
        let integer_type = ArrayType::scalar(DataType::I32).with_sharding(sharding.clone()).unwrap();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(value_type.clone()));
        let mut body = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let reference = body.add_input(reference_type.clone());
        let value = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let integer = body
            .add_instruction(
                XlaOperation::Array(ArrayOperation::Zero(ZeroOperation::new(integer_type.clone()))),
                Vec::new(),
                Vec::new(),
                None,
            )
            .unwrap()[0];
        let body = body
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                vec![integer, value, reference],
                vec![Placeholder],
                vec![Placeholder; 3],
            )
            .unwrap();
        let operation = ShardMapOperation::from_program(
            &body,
            vec![reference_type.clone()],
            mesh,
            vec![sharding.clone()],
            vec![sharding; 3],
            vec!["x".to_string()],
            true,
        )
        .unwrap();
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let reference = builder.add_input(reference_type.clone());
        let body = builder.import_program(body);
        let outputs = builder
            .add_instruction(XlaOperation::ShardMap(Box::new(operation)), vec![body], vec![reference], None)
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder], vec![Placeholder; 3])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        let seed_types = vec![ArrayIrType::Array(integer_type.cotangent().unwrap()), ArrayIrType::Array(value_type)];
        assert_eq!(transposed.input_types(), [seed_types.clone(), vec![reference_type.clone()]].concat());
        assert_eq!(transposed.output_types(), vec![reference_type]);
        let shard_maps = shard_map_instructions(&transposed);
        assert_eq!(shard_maps.len(), 1);
        assert_eq!(shard_maps[0].1.global_input_types(), seed_types.as_slice());
        let body = transposed.region_ref(shard_maps[0].0.regions()[0]).unwrap();
        let divisions = body
            .instructions()
            .iter()
            .filter(|instruction| instruction.operation().name() == "div")
            .collect::<Vec<_>>();
        assert_eq!(divisions.len(), 1);
        assert_eq!(body.atoms()[divisions[0].inputs()[0].index()].r#type().as_ref(), &seed_types[1]);
    }

    #[test]
    fn test_shard_map_zero_transpose_uses_cotangent_descriptors() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(tangent_sharding.clone())
            .unwrap();
        let cotangent_type = tangent_type.cotangent().unwrap();
        let shard_map =
            ShardMap::from_shardings(mesh, vec![tangent_sharding.clone()], vec![tangent_sharding], Vec::new(), true);
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![tangent_type.clone()], vec![tangent_type.clone()]);
        // Zero cotangents skip a pure callee only after its effects have been inspected. Supply the real
        // identity-region boundary rather than an empty driver that cannot answer that query.
        let mut source_builder = XlaProgramBuilder::new();
        let input = source_builder.add_input(ArrayIrType::Array(tangent_type.clone()));
        let source = source_builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut transpose_builder = XlaProgramBuilder::new();
        let input = transpose_builder.add_input(ArrayIrType::Array(cotangent_type.clone()));
        let transposed = transpose_builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let driver = TestTranspositionDriver { source, transposed };
        let mut context = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = context.input(ArrayIrType::Array(tangent_type.clone()));
        let cotangents = transpose_primal_shard_map(
            &operation,
            &mut context,
            &driver,
            &[PartialValue::Known(known)],
            &[MaybeZero::Zero(ArrayIrType::Array(tangent_type))],
            &OperandCotangents::without_references([true]),
        )
        .unwrap();
        assert!(matches!(&cotangents[..], [MaybeZero::Zero(actual)] if actual == &ArrayIrType::Array(cotangent_type)));
    }

    #[test]
    fn test_shard_map_mixed_output_transpose_materializes_zero_space_values() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::replicated(mesh.clone(), 1);
        let value_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let predicate_type = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(sharding.clone())
            .unwrap();
        let source = {
            let mut builder = XlaProgramBuilder::new();
            let value = builder.add_input(value_type.clone().into());
            let predicate = builder.add_constant(XlaConstant::Captured(CaptureReference::new(
                0,
                ArrayIrType::Array(predicate_type.clone()),
            )));
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![value, predicate],
                    vec![Placeholder],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let transposed = {
            let mut builder = XlaProgramBuilder::new();
            let value_cotangent = builder.add_input(value_type.clone().into());
            let _predicate_cotangent = builder.add_input(predicate_type.cotangent().unwrap().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![value_cotangent],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let shard_map =
            ShardMap::from_shardings(mesh, vec![sharding.clone()], vec![sharding.clone(), sharding], Vec::new(), true);
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![value_type.clone()],
            vec![value_type.clone(), predicate_type.clone()],
        );
        let mut context = TracingContext::<XlaConstant, XlaOperation>::new();
        let value_cotangent = context.input(ArrayIrType::Array(value_type.clone()));

        let contributions = transpose_primal_shard_map(
            &operation,
            &mut context,
            &driver,
            &[PartialValue::Unknown(ArrayIrType::Array(value_type.clone()))],
            &[
                MaybeZero::Value(value_cotangent),
                MaybeZero::Zero(ArrayIrType::Array(predicate_type.cotangent().unwrap())),
            ],
            &OperandCotangents::without_references([true]),
        )
        .unwrap();

        assert!(matches!(&contributions[..], [MaybeZero::Value(value)]
                if value.r#type().as_ref() == &ArrayIrType::Array(value_type)));
    }

    /// Reverse mode through a `shard_map` must reach [`transpose_primal_shard_map`] through the composite
    /// [`XlaOperation`] transposition dispatcher, not just through a direct call to the payload rule. The dispatcher is
    /// derived, so this pins that the `ShardMap` variant is dispatched as a composite-native payload and stages a
    /// transposed `shard_map` instead of reporting the operation as non-transposable.
    #[test]
    fn test_shard_map_transposes_through_the_composite_operation_dispatcher() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let tangent_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["x"])
            .unwrap();
        let tangent_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(tangent_sharding.clone())
            .unwrap();
        let cotangent_type = tangent_type.cotangent().unwrap();

        let source = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(tangent_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let transposed = {
            let mut builder = XlaProgramBuilder::new();
            let input = builder.add_input(cotangent_type.clone().into());
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let driver = TestTranspositionDriver { source, transposed };
        let shard_map = ShardMap::from_shardings(
            mesh,
            vec![tangent_sharding.clone()],
            vec![tangent_sharding],
            vec!["x".to_string()],
            true,
        );
        let operation =
            ShardMapOperation::from_boundary(shard_map, vec![tangent_type.clone()], vec![tangent_type.clone()]);
        let context = TracingContext::<XlaConstant, XlaOperation>::new();
        let output_cotangent = context.input(ArrayIrType::Array(cotangent_type.clone()));

        let mut transposition_context = TranspositionContext::new(context.clone());
        let inputs = [PartialValue::Unknown(ArrayIrType::Array(tangent_type))];
        let accumulators = transposition_context.input_accumulators(&inputs, &[]).unwrap();
        XlaOperation::<XlaConstant>::ShardMap(Box::new(operation))
            .transpose(
                &mut transposition_context,
                &driver,
                &inputs,
                &[MaybeZero::Value(output_cotangent)],
                &accumulators,
            )
            .expect("the composite dispatcher should reach the shard_map transpose rule");
        let cotangents = transposition_context.take_cotangents(&accumulators).unwrap();

        assert!(matches!(&cotangents[..], [MaybeZero::Value(cotangent)]
                if cotangent.r#type().as_ref() == &ArrayIrType::Array(cotangent_type)));

        // The staged pullback is a `shard_map` over the transposed body, so the manual region survives reverse mode.
        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), XlaOperation::ShardMap(_)));
        assert_eq!(builder.instructions()[0].regions().len(), 1);
    }

    /// Builds a one-input, one-output identity shard-map body whose global boundary carries no sharding, so its
    /// boundary types match any same-shape input fed to it through [`ShardMapOperation::interpret`].
    fn single_input_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mut builder = XlaProgramBuilder::new();
        let input = builder.add_input(array_type.clone().into());
        FlatTracedShardMap::from_parts(
            single_input_test_shard_map(),
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            vec![array_type.clone()],
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![input], vec![Placeholder], vec![Placeholder])
                .unwrap(),
        )
    }

    fn mixed_known_unknown_traced_shard_map_body() -> FlatTracedShardMap {
        let array_type = test_array_type();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let replicated = Sharding::replicated(mesh.clone(), 0);
        let shard_map = ShardMap::from_shardings(
            mesh,
            vec![replicated.clone(), replicated.clone()],
            vec![replicated.clone(), replicated.clone(), replicated],
            vec!["x".to_string()],
            true,
        );

        let mut builder = XlaProgramBuilder::new();
        let known_input = builder.add_input(array_type.clone().into());
        let runtime_input = builder.add_input(array_type.clone().into());
        let doubled = builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![known_input, known_input], None)
            .unwrap()[0];
        let product = builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![known_input, runtime_input], None)
            .unwrap()[0];
        let sum = builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![runtime_input, known_input], None)
            .unwrap()[0];
        FlatTracedShardMap::from_parts(
            shard_map,
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone(), array_type.clone()],
            vec![array_type.clone(), array_type.clone(), array_type.clone()],
            vec![array_type.clone(), array_type.clone(), array_type],
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![doubled, product, sum],
                    vec![Placeholder; 2],
                    vec![Placeholder; 3],
                )
                .unwrap(),
        )
    }

    #[test]
    fn test_traced_shard_map_binds_staging_onto_input_trace() {
        let body = single_input_traced_shard_map_body();

        // The input tracer already belongs to a trace; binding the shard_map through that trace's context composes
        // the staging onto the same builder, attaching the local body as the instruction's `body` region.
        let context = DomainTracingContext::<XlaDomain<'static>>::new();
        let input = context.input(ArrayIrType::Array(test_array_type()));

        let (operation, body_program) = ShardMapOperation::from_body(body);
        let outputs = context
            .bind(XlaOperation::ShardMap(Box::new(operation)), vec![body_program], std::slice::from_ref(&input))
            .expect("traced shard_map staging should compose onto the input tracer's trace");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), ArrayIrType::Array(test_array_type()));

        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert!(matches!(builder.instructions()[0].operation(), XlaOperation::ShardMap(_)));
        assert_eq!(builder.instructions()[0].inputs(), &[input.atom_id().unwrap()]);
        assert_eq!(builder.instructions()[0].regions().len(), 1);
    }

    /// Online partial evaluation of a mixed `shard_map` against a live outer trace: the known half of the local body
    /// is rewrapped as a known-side `shard_map` staged into the outer program over the symbolic known input, the
    /// unknown half stays behind a residual `shard_map`, the known→unknown residual edges flow between them, and the
    /// mesh and shardings are threaded onto both boundaries.
    #[test]
    fn test_shard_map_online_partial_evaluation_splits_body_against_a_live_outer_trace() {
        use ryft_core::{PartialEvaluationInput, PartialEvaluationOutput, TracingContext};

        let array_type = test_array_type();
        let (operation, body_program) =
            ShardMapOperation::<XlaConstant>::from_body(mixed_known_unknown_traced_shard_map_body());

        // Enclosing program staging one `shard_map` over `[a, x]`, with the local body attached as its region.
        let mut builder = XlaProgramBuilder::new();
        let known_input = builder.add_input(array_type.clone().into());
        let runtime_input = builder.add_input(array_type.clone().into());
        let body_region = builder.import_program(body_program);
        let outputs = builder
            .add_instruction(
                XlaOperation::ShardMap(Box::new(operation)),
                vec![body_region],
                vec![known_input, runtime_input],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(ArrayIrType::Array(array_type.clone()));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(ArrayIrType::Array(array_type.clone()))],
            )
            .unwrap();

        // The known half landed in the outer program as one known-side `shard_map` over the symbolic known input,
        // producing the fully known boundary output (`a + a`) plus the residual edge (`a` itself), with the residual
        // edge's boundary derived from its local type.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            let known_instruction = &outer_builder.instructions()[0];
            let XlaOperation::ShardMap(known_side) = known_instruction.operation() else {
                panic!("expected the outer program to contain the known-side shard_map");
            };
            let known_body = outer_builder.region_ref(known_instruction.regions()[0]).unwrap().to_program();
            assert_eq!(known_body.input_ids().len(), 1);
            assert_eq!(known_body.output_ids().len(), 2);
            assert_eq!(known_body.instructions().len(), 1);
            assert_eq!(known_side.global_output_types().len(), 2);
            assert_eq!(known_side.shard_map().in_shardings().len(), 1);
            assert_eq!(known_side.shard_map().out_shardings().len(), 2);
        }

        // The unknown half stayed behind one residual `shard_map` over the unknown boundary input plus the residual
        // edge, with the shardings threaded per input.
        assert_eq!(evaluation.program().instructions().len(), 1);
        let residual_instruction = &evaluation.program().instructions()[0];
        let XlaOperation::ShardMap(residual_side) = residual_instruction.operation() else {
            panic!("expected the residual program to contain the residual shard_map");
        };
        let residual_body = evaluation.program().region_ref(residual_instruction.regions()[0]).unwrap().to_program();
        assert_eq!(residual_body.input_ids().len(), 2);
        assert_eq!(residual_body.instructions().len(), 2);
        assert_eq!(residual_side.global_input_types().len(), 2);
        assert_eq!(residual_side.shard_map().in_shardings().len(), 2);
        assert_eq!(residual_side.shard_map().out_shardings().len(), 2);
        assert_eq!(residual_side.global_output_types().len(), 2);

        // The boundary descriptors: the unknown enclosing input feeds the residual side, the residual edge is a
        // known feeder naming the known-side call's staged output, and the outputs reassemble in original order.
        assert_eq!(evaluation.inputs().len(), 2);
        assert!(matches!(&evaluation.inputs()[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs()[1], PartialEvaluationInput::Known(value) if value.atom_id().is_ok()));
        assert_eq!(evaluation.outputs().len(), 3);
        assert!(matches!(&evaluation.outputs()[0], PartialEvaluationOutput::Known(value) if value.atom_id().is_ok()));
        assert!(matches!(&evaluation.outputs()[1], PartialEvaluationOutput::Unknown(0)));
        assert!(matches!(&evaluation.outputs()[2], PartialEvaluationOutput::Unknown(1)));
    }

    /// Two-device manual mesh over `x`.
    fn manual_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap()
    }

    /// Global `f32[size]` boundary type.
    fn f32_vector_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(size)]))
    }

    /// Reference-bearing shard-map fixture over the two-device manual mesh: the boundary is
    /// `[r: ref<reference_type>, x: f32[4]] -> f32[4]`, with the reference input sharded by `reference_sharding` and
    /// the value input and the output sharded along `x`, so every device owns a `f32[2]` shard of `x` and of the
    /// output. The body performs `add_update(r_local, x_local); read(r_local)` when `mutate` holds and
    /// `read(r_local) + x_local` otherwise, over the local shards, so the reference's local referent must be `f32[2]`
    /// (a `f32[4]` referent sharded along `x`, or a `f32[2]` referent replicated along `x`).
    fn reference_shard_map(
        reference_type: ArrayType,
        reference_sharding: Sharding,
        mutate: bool,
    ) -> (ShardMapOperation<XlaConstant>, XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>) {
        let mesh = manual_mesh();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shard_map = ShardMap::from_shardings(
            mesh,
            vec![reference_sharding, sharded.clone()],
            vec![sharded],
            vec!["x".to_string()],
            true,
        );
        let local_reference_type = ReferenceType::new(shard_map.local_input_type(0, &reference_type).unwrap());
        let local_value_type = shard_map.local_input_type(1, &f32_vector_type(4)).unwrap();
        let body = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let reference = builder.add_input(ArrayIrType::Reference(local_reference_type));
            let update = builder.add_input(ArrayIrType::Array(local_value_type));
            let output = if mutate {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
                    .unwrap();
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0]
            } else {
                let state = builder
                    .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
                    .unwrap()[0];
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![state, update], None).unwrap()[0]
            };
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let operation = ShardMapOperation::from_boundary(
            shard_map,
            vec![ArrayIrType::Reference(ReferenceType::new(reference_type)), ArrayIrType::Array(f32_vector_type(4))],
            vec![ArrayIrType::Array(f32_vector_type(4))],
        );
        (operation, body)
    }

    /// Stages `operation` over its attached `body` as the one instruction of a flat program whose inputs carry the
    /// operation's global input types and whose outputs are the instruction's outputs.
    fn shard_map_program(
        operation: ShardMapOperation<XlaConstant>,
        body: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
    ) -> Result<XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>, ProgramError> {
        let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
        let inputs = operation.global_input_types().iter().cloned().map(|r#type| builder.add_input(r#type));
        let inputs = inputs.collect::<Vec<_>>();
        let body_region = builder.import_program(body);
        let outputs = builder
            .add_instruction(XlaOperation::ShardMap(Box::new(operation)), vec![body_region], inputs.clone(), None)?
            .to_vec();
        let output_count = outputs.len();
        builder.build::<Vec<XlaConstant>, Vec<XlaConstant>>(
            outputs,
            vec![Placeholder; inputs.len()],
            vec![Placeholder; output_count],
        )
    }

    /// Returns the `shard_map` instructions of `program`'s entry region with their payloads, in program order.
    fn shard_map_instructions(
        program: &XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
    ) -> Vec<(&ryft_core::Instruction<XlaOperation>, &ShardMapOperation<XlaConstant>)> {
        program
            .instructions()
            .iter()
            .filter_map(|instruction| match instruction.operation() {
                XlaOperation::ShardMap(operation) => Some((instruction, operation.as_ref())),
                _ => None,
            })
            .collect()
    }

    /// Type inference over a reference-bearing boundary: a reference operand must reach the body as a reference to
    /// its local shard, and a reference output must name the input it forwards under an equal output sharding.
    /// Batching a reference input along a manual axis is unreachable here because the XLA operation family has no
    /// batching dispatch, so the boundary contract is only checked by type inference and the transform rules.
    #[test]
    fn test_shard_map_reference_boundary_type_inference() {
        let sharded = Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let body_interface = RegionInterface::new(body.input_types(), body.output_types(), body.effects().classes());
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let value_type = ArrayIrType::Array(f32_vector_type(4));
        assert_eq!(
            operation.infer_output_types(&[reference_type.clone(), value_type.clone()], &[body_interface.clone()]),
            Ok(vec![value_type.clone()]),
        );

        // Operand kinds must match the declared boundary positions.
        assert_eq!(
            operation.infer_output_types(&[value_type.clone(), value_type.clone()], &[body_interface.clone()]),
            Err(TypeError::invalid("expected reference type but got array type")),
        );
        assert_eq!(
            operation.infer_output_types(&[reference_type.clone(), reference_type.clone()], &[body_interface]),
            Err(TypeError::invalid("expected array type but got reference type")),
        );

        // The body must receive the local shard of the referent, not the global referent.
        let global_body = RegionInterface::new(
            vec![reference_type.clone(), body.input_types()[1].clone()],
            body.output_types(),
            body.effects().classes(),
        );
        assert_eq!(
            operation.infer_output_types(&[reference_type.clone(), value_type.clone()], &[global_body]),
            Err(TypeError::invalid(
                "shard_map body input 0 has type `ref<f32[4]>` but the local shard of reference input 0 is \
                 `ref<f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}], varying_manual={'x'}}]>`",
            )),
        );

        // A reference output must declare which input it forwards, forward it by identity, and share its sharding.
        let forwarding_body = RegionInterface::new(
            body.input_types(),
            vec![body.output_types()[0].clone(), body.input_types()[0].clone()],
            body.effects().classes(),
        );
        let forwarding_operation = ShardMapOperation::<XlaConstant>::from_boundary(
            ShardMap::from_shardings(
                manual_mesh(),
                vec![sharded.clone(), sharded.clone()],
                vec![sharded.clone(), sharded.clone()],
                vec!["x".to_string()],
                true,
            ),
            operation.global_input_types().to_vec(),
            vec![value_type.clone(), reference_type.clone()],
        );
        assert_eq!(
            forwarding_operation
                .infer_output_types(&[reference_type.clone(), value_type.clone()], &[forwarding_body.clone()]),
            Err(TypeError::invalid(
                "shard_map output 1 is a reference whose forwarded input the operation does not declare",
            )),
        );
        let forwarding_operation = forwarding_operation.with_output_forwarding(vec![None, Some(0)]).unwrap();
        assert_eq!(
            forwarding_operation
                .infer_output_types(&[reference_type.clone(), value_type.clone()], &[forwarding_body.clone()]),
            Ok(vec![value_type.clone(), reference_type.clone()]),
        );
        assert_eq!(
            forwarding_operation
                .clone()
                .with_output_forwarding(vec![None, Some(1)])
                .unwrap()
                .infer_output_types(&[reference_type.clone(), value_type.clone()], &[forwarding_body.clone()]),
            Err(TypeError::invalid("shard_map output 1 does not forward reference input 1 by identity")),
        );
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let mismatched_operation = ShardMapOperation::<XlaConstant>::from_boundary(
            ShardMap::from_shardings(
                manual_mesh(),
                vec![sharded.clone(), sharded.clone()],
                vec![sharded.clone(), replicated.clone()],
                vec!["x".to_string()],
                true,
            ),
            operation.global_input_types().to_vec(),
            vec![value_type.clone(), reference_type.clone()],
        )
        .with_output_forwarding(vec![None, Some(0)])
        .unwrap();
        assert_eq!(
            mismatched_operation.infer_output_types(&[reference_type, value_type], &[forwarding_body]),
            Err(TypeError::invalid(format!(
                "shard_map output 1 forwards reference input 0 but its output sharding `{replicated}` differs from \
                 the input sharding `{sharded}`",
            ))),
        );
        assert!(matches!(
            forwarding_operation.with_output_forwarding(vec![None]),
            Err(ShardMapTraceError::OutputForwardingCountMismatch { expected: 2, actual: 1 }),
        ));
    }

    /// The rendered fingerprint names the forwarded inputs exactly when some output forwards a reference.
    #[test]
    fn test_shard_map_render_includes_output_forwarding() {
        let sharded = Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (operation, _) = reference_shard_map(f32_vector_type(4), sharded, true);
        let without_forwarding = operation.to_string();
        assert!(!without_forwarding.contains("output_forwarding"));
        let with_forwarding = operation.with_output_forwarding(vec![Some(0)]).unwrap().to_string();
        assert_ne!(with_forwarding, without_forwarding);
        assert!(with_forwarding.ends_with("output_forwarding=[0],\n]"), "{with_forwarding}");
    }

    /// Discharging a shard map with a sharded reference input threads the state carry with the local shard type
    /// inside the body and with the global referent at the boundary: the reference position becomes the global
    /// referent under its input sharding, the mutated input publishes a hidden final-state output under that same
    /// sharding, and the body's state carry is the `f32[2]` shard the device owns.
    #[test]
    fn test_shard_map_reference_discharge_threads_local_state_and_publishes_global_final_state() {
        let sharded = Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let local_type = ArrayIrType::Array(operation.shard_map().local_input_type(1, &f32_vector_type(4)).unwrap());
        let program = shard_map_program(operation, body).unwrap();

        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 1);
        assert_eq!(discharged.program().output_count(), 2);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].source(), ReferenceSource::Input { index: 0 });
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(1));
        assert!(!discharged.program().entry_region_ref().contains_references_in_closure());

        // The boundary: the reference position carries the global referent, the value output keeps its position, and
        // the final state follows it under the reference input's sharding.
        let shard_maps = shard_map_instructions(discharged.program());
        assert_eq!(shard_maps.len(), 1);
        let (instruction, operation) = shard_maps[0];
        let global_type = ArrayIrType::Array(f32_vector_type(4));
        assert_eq!(operation.global_input_types(), &[global_type.clone(), global_type.clone()]);
        assert_eq!(operation.global_output_types(), &[global_type.clone(), global_type.clone()]);
        assert_eq!(operation.output_forwarding(), &[None, None]);
        assert_eq!(operation.shard_map().in_shardings(), &[sharded.clone(), sharded.clone()]);
        assert_eq!(operation.shard_map().out_shardings(), &[sharded.clone(), sharded]);
        assert_eq!(discharged.program().output_ids(), instruction.outputs());

        // The body: both inputs and both outputs are the local `f32[2]` shards.
        let body = discharged.program().region_ref(instruction.regions()[0]).unwrap();
        assert_eq!(body.input_types(), vec![local_type.clone(), local_type.clone()]);
        assert_eq!(body.output_types(), vec![local_type.clone(), local_type]);
        assert_eq!(
            body.to_program().to_string(),
            indoc! {"
                lambda \
                %0:f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}], varying_manual={'x'}}], \
                %1:f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}], varying_manual={'x'}}] .
                let \
                %2:f32[2][sharding={mesh<['x'=2:manual]>, [{'x'}], varying_manual={'x'}}] = add %0 %1
                in (%2, %2)
            "}
            .trim_end(),
        );
    }

    /// The hidden final-state output is typed by the operand that the mutated reference position replaces: when the
    /// staged operand carries sharding metadata that the declared global referent omits, the final state carries the
    /// operand's type, since the discharged state must match the allocation's referent exactly.
    #[test]
    fn test_shard_map_reference_discharge_types_final_state_by_the_operand_referent() {
        let sharded = Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let sharded_type = f32_vector_type(4).with_sharding(sharded).unwrap();
        let program = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let reference = builder.add_input(ArrayIrType::Reference(ReferenceType::new(sharded_type.clone())));
            let update = builder.add_input(ArrayIrType::Array(sharded_type.clone()));
            let body_region = builder.import_program(body);
            let operation = XlaOperation::ShardMap(Box::new(operation));
            let outputs = builder
                .add_instruction(operation, vec![body_region], vec![reference, update], None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        let discharged = program.discharge_references(0).unwrap();
        let (_, operation) = shard_map_instructions(discharged.program())[0];
        let global_type = ArrayIrType::Array(f32_vector_type(4));
        assert_eq!(operation.global_input_types(), &[global_type.clone(), global_type.clone()]);
        assert_eq!(operation.global_output_types(), &[global_type, ArrayIrType::Array(sharded_type.clone())]);
        assert_eq!(discharged.program().output_types()[1], ArrayIrType::Array(sharded_type));
    }

    /// A reference input replicated along the manual axis may be read, since every device sees the whole referent,
    /// but may not be mutated, since no device owns it.
    #[test]
    fn test_shard_map_reference_discharge_rejects_mutating_a_replicated_reference() {
        let replicated = Sharding::replicated(manual_mesh(), 1);
        let (operation, body) = reference_shard_map(f32_vector_type(2), replicated.clone(), false);
        let program = shard_map_program(operation, body).unwrap();
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.program().output_count(), 1);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert!(!discharged.external_reference_bindings()[0].is_mutated());
        let shard_maps = shard_map_instructions(discharged.program());
        assert_eq!(shard_maps.len(), 1);
        assert_eq!(shard_maps[0].1.global_input_types()[0], ArrayIrType::Array(f32_vector_type(2)));
        assert_eq!(shard_maps[0].1.shard_map().in_shardings()[0], replicated.clone());

        // Mutating the replicated reference with a value of its own (invariant) referent type is a well-formed local
        // body, since a replicated referent is not varying along `x`, so only the ownership check can reject it.
        let (operation, body) = reference_shard_map(f32_vector_type(2), replicated, false);
        let mutating_body = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let reference = builder.add_input(body.input_types()[0].clone());
            builder.add_input(body.input_types()[1].clone());
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, state], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let program = shard_map_program(operation, mutating_body).unwrap();
        assert!(matches!(
            program.discharge_references(0),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`shard_map` input 0 is a reference replicated along manual axis `x`; mutating a \
                               replicated reference is not supported, shard it along that axis or read it only",
        ));
    }

    /// A reference reaching the body as a captured constant has no input sharding and therefore no owner.
    #[test]
    fn test_shard_map_rejects_captured_reference_in_body() {
        let sharded = Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded, true);
        let local_reference_type = body.input_types()[0].clone();
        let capturing_body = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let reference = builder.add_input(local_reference_type.clone());
            let update = builder.add_input(body.input_types()[1].clone());
            let captured = builder.add_constant(XlaConstant::Captured(CaptureReference::new(0, local_reference_type)));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![captured, update], None)
                .unwrap();
            let output =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let program = shard_map_program(operation, capturing_body).unwrap();
        assert!(matches!(
            program.discharge_references(0),
            Err(ProgramError::MalformedProgram(message)) if message.contains("reference-typed constant"),
        ));
    }

    /// A reference output forwards a reference input by identity under an equal sharding; the caller keeps the handle
    /// it passed, a mismatched output sharding is rejected at construction, and an allocation made inside the body
    /// cannot escape.
    #[test]
    fn test_shard_map_forwarded_reference_output_keeps_the_forwarded_root() {
        let sharded = Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let forwarding_body = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let inputs = body.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let mut outputs = builder.splice_program(&body, inputs.as_slice()).unwrap();
            outputs.push(inputs[0]);
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        let forwarding_operation = |output_sharding: Sharding| {
            ShardMapOperation::<XlaConstant>::from_boundary(
                ShardMap::from_shardings(
                    manual_mesh(),
                    operation.shard_map().in_shardings().to_vec(),
                    vec![sharded.clone(), output_sharding],
                    vec!["x".to_string()],
                    true,
                ),
                operation.global_input_types().to_vec(),
                vec![ArrayIrType::Array(f32_vector_type(4)), reference_type.clone()],
            )
            .with_output_forwarding(vec![None, Some(0)])
            .unwrap()
        };

        // Equal shardings: the outer program reads through the forwarded output, which resolves to the same root as
        // the reference input, so discharge publishes exactly one final state and the read observes it.
        let program = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let reference = builder.add_input(reference_type.clone());
            let update = builder.add_input(ArrayIrType::Array(f32_vector_type(4)));
            let body_region = builder.import_program(forwarding_body.clone());
            let outputs = builder
                .add_instruction(
                    XlaOperation::ShardMap(Box::new(forwarding_operation(sharded.clone()))),
                    vec![body_region],
                    vec![reference, update],
                    None,
                )
                .unwrap()
                .to_vec();
            let state =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![outputs[1]], None).unwrap()[0];
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(
                    vec![outputs[0], state],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let discharged = program.discharge_references(0).unwrap();
        assert_eq!(discharged.output_count(), 2);
        assert_eq!(discharged.program().output_count(), 3);
        assert_eq!(discharged.external_reference_bindings().len(), 1);
        assert_eq!(discharged.external_reference_bindings()[0].output_index(), Some(2));
        let shard_maps = shard_map_instructions(discharged.program());
        assert_eq!(shard_maps.len(), 1);
        assert_eq!(shard_maps[0].1.global_output_types().len(), 2);
        assert_eq!(shard_maps[0].1.output_forwarding(), &[None, None]);
        // The read through the forwarded handle observes the final state published by the shard map.
        assert_eq!(discharged.program().output_ids()[1], shard_maps[0].0.outputs()[1]);
        assert_eq!(discharged.program().output_ids()[2], shard_maps[0].0.outputs()[1]);

        // Unequal shardings are rejected when the instruction is built.
        let replicated = Sharding::replicated(manual_mesh(), 1);
        assert!(matches!(
            shard_map_program(forwarding_operation(replicated.clone()), forwarding_body),
            Err(ProgramError::Type(error))
                if error == TypeError::invalid(format!(
                    "shard_map output 1 forwards reference input 0 but its output sharding `{replicated}` differs \
                     from the input sharding `{sharded}`",
                )),
        ));

        // An allocation made inside the body cannot escape through a reference output, even when the operation
        // declares it as forwarding an input of the same local type.
        let escaping_body = {
            let mut builder = ProgramBuilder::<XlaConstant, XlaOperation>::new();
            let inputs = body.input_types().into_iter().map(|r#type| builder.add_input(r#type)).collect::<Vec<_>>();
            let mut outputs = builder.splice_program(&body, inputs.as_slice()).unwrap();
            let allocation =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![inputs[1]], None).unwrap()[0];
            outputs.push(allocation);
            builder
                .build::<Vec<XlaConstant>, Vec<XlaConstant>>(outputs, vec![Placeholder; 2], vec![Placeholder; 2])
                .unwrap()
        };
        assert!(matches!(
            forwarding_operation(sharded.clone()).validate_reference_body(escaping_body.entry_region_ref()),
            Err(ProgramError::MalformedProgram(message))
                if message == "`shard_map` output 1 is a reference allocated inside the shard-map body, which cannot \
                               escape; a reference output must forward a reference input",
        ));
    }

    /// Forward mode threads a tangent reference beside an active reference input under the primal's input sharding,
    /// and a plumbing reference input (one without a tangent reference) has no tangent boundary slot.
    #[test]
    fn test_shard_map_jvp_threads_tangent_references() {
        let sharded = Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let local_reference_type = body.input_types()[0].clone();
        let local_value_type = body.input_types()[1].clone();
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let value_type = ArrayIrType::Array(f32_vector_type(4));
        let program = shard_map_program(operation, body).unwrap();

        // Every input active: the fused program consumes `[r, x, ṫr, ẋ]`, the primal shard map keeps its boundary
        // and gains the residual edges, and the tangent shard map consumes the tangent reference under the primal's
        // input sharding followed by the tangent value and the residuals.
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.input_types(),
            vec![reference_type.clone(), value_type.clone(), reference_type.clone(), value_type.clone()],
        );
        assert_eq!(jvp.output_types(), vec![value_type.clone(), value_type.clone()]);
        let shard_maps = shard_map_instructions(&jvp);
        assert_eq!(shard_maps.len(), 2);
        let (primal_instruction, primal_operation) = shard_maps[0];
        let (tangent_instruction, tangent_operation) = shard_maps[1];
        assert_eq!(primal_operation.global_input_types(), &[reference_type.clone(), value_type.clone()]);
        assert_eq!(primal_operation.global_output_types()[0], value_type);
        assert_eq!(&primal_instruction.inputs()[..2], &jvp.input_ids()[..2]);
        let residual_count = primal_operation.global_output_types().len() - 1;
        assert_eq!(&tangent_operation.global_input_types()[..2], &[reference_type.clone(), value_type.clone()]);
        assert_eq!(tangent_operation.global_input_types().len(), 2 + residual_count);
        assert_eq!(&tangent_operation.shard_map().in_shardings()[..2], &[sharded.clone(), sharded.clone()]);
        assert_eq!(tangent_operation.global_output_types(), &[value_type.clone()]);
        assert_eq!(&tangent_instruction.inputs()[..2], &jvp.input_ids()[2..]);
        let tangent_body = jvp.region_ref(tangent_instruction.regions()[0]).unwrap();
        assert_eq!(&tangent_body.input_types()[..2], &[local_reference_type, local_value_type.clone()]);

        // A plumbing reference (no tangent reference supplied) is inactive: the tangent shard map consumes only the
        // tangent value followed by the residuals. The body reads the reference without mutating it, since writing an
        // active tangent into a plumbing reference is rejected by the reference rules themselves.
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), false);
        let program = shard_map_program(operation, body).unwrap();
        let jvp = program.entry_region_ref().jvp(&[1]).unwrap();
        assert_eq!(jvp.input_types(), vec![reference_type.clone(), value_type.clone(), value_type.clone()]);
        let shard_maps = shard_map_instructions(&jvp);
        assert_eq!(shard_maps.len(), 2);
        let (tangent_instruction, tangent_operation) = shard_maps[1];
        assert_eq!(tangent_operation.global_input_types()[0], value_type);
        assert_eq!(tangent_operation.shard_map().in_shardings()[0], sharded);
        let tangent_body = jvp.region_ref(tangent_instruction.regions()[0]).unwrap();
        assert_eq!(tangent_body.input_types()[0], local_value_type);
    }

    /// Reverse mode threads a cotangent reference at the position of a live linear reference input under the
    /// primal's input sharding: the transposed shard map consumes `[ȳ, r̄]`, accumulates into `r̄` in place, returns it
    /// by identity ahead of `x̄`, and its body is exactly the transposition of the local body.
    #[test]
    fn test_shard_map_transposition_threads_cotangent_references() {
        let sharded = Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded.clone(), true);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let value_type = ArrayIrType::Array(f32_vector_type(4));
        let program = shard_map_program(operation, body.clone()).unwrap();

        let destination_kinds = [CotangentDestinationKind::Reference, CotangentDestinationKind::Return];
        let transposed = program.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed.input_types(), vec![value_type.clone(), reference_type.clone()]);
        assert_eq!(transposed.output_types(), vec![reference_type.clone(), value_type.clone()]);
        let shard_maps = shard_map_instructions(&transposed);
        assert_eq!(shard_maps.len(), 1);
        let (instruction, transposed_operation) = shard_maps[0];
        assert_eq!(instruction.inputs(), transposed.input_ids());
        assert_eq!(transposed.output_ids(), &[transposed.input_ids()[1], instruction.outputs()[1]]);
        assert_eq!(transposed_operation.global_input_types(), &[value_type.clone(), reference_type.clone()]);
        assert_eq!(transposed_operation.global_output_types(), &[reference_type, value_type]);
        assert_eq!(transposed_operation.output_forwarding(), &[Some(1), None]);
        assert_eq!(transposed_operation.shard_map().in_shardings(), &[sharded.cotangent(), sharded.clone()]);
        assert_eq!(transposed_operation.shard_map().out_shardings(), &[sharded.clone(), sharded.cotangent()]);
        let transposed_body = transposed.region_ref(instruction.regions()[0]).unwrap().to_program();
        let inlined = body.transpose_with_respect_to(&[0, 1], &destination_kinds).unwrap();
        assert_eq!(transposed_body.to_string(), inlined.to_string());
    }

    /// Partial evaluation keeps a reference-bearing shard map whole even when its known inputs are symbolic.
    #[test]
    fn test_shard_map_partial_evaluation_keeps_reference_bearing_bodies_whole() {
        let sharded = Sharding::new(manual_mesh(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let (operation, body) = reference_shard_map(f32_vector_type(4), sharded, true);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(f32_vector_type(4)));
        let value_type = ArrayIrType::Array(f32_vector_type(4));
        let program = shard_map_program(operation, body).unwrap();

        let outer = TracingContext::<XlaConstant, XlaOperation>::new();
        let known = outer.input(reference_type);
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(value_type)])
            .unwrap();
        let residual = evaluation.program();
        assert_eq!(residual.instructions().len(), 1);
        assert!(matches!(residual.instructions()[0].operation(), XlaOperation::ShardMap(_)));
        assert_eq!(residual.instructions()[0].inputs().len(), 2);
        let residual_names = residual
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(residual_names, vec!["shard_map", "reference_add_update", "reference_read"]);
        assert!(outer.builder().borrow().instructions().is_empty());
    }
}
