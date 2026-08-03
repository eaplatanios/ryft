use crate::{
    Attribute, BooleanAttributeRef, DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion, DialectHandle,
    DictionaryAttributeRef, Error, Location, Operation, OperationBuilder, RegionRef, StringRef, Type, TypeRef,
    ValueRef, mlir_op, mlir_op_trait,
};

use super::{FailurePropagationMode, FailurePropagationModeAttributeRef, MatchCmpIPredicate};

/// Name of the attribute used by Transform dialect operations with multiple variadic operand groups.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// Operation trait for `transform.alternatives`.
pub trait AlternativesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional transform scope operand.
    fn scope(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() == 0 { Ok(None) } else { self.operand_value(0).map(Some) }
    }

    /// Returns the handles yielded by the successful alternative region.
    fn yielded_results(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.results().map(|result| result.map(ValueRef::from))
    }

    /// Returns the alternative regions, attempted in operation order.
    fn alternatives(&self) -> impl Iterator<Item = Result<RegionRef<'o, 'c, 't>, Error>> {
        self.regions()
    }
}

mlir_op!(Alternatives);
mlir_op_trait!(Alternatives, ZeroSuccessors);

/// Constructs a new detached/owned [`AlternativesOperation`] at the specified [`Location`].
pub fn alternatives<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    scope: Option<ValueRef<'v, 'c, 't>>,
    result_types: &[TypeRef<'c, 't>],
    alternatives: Vec<DetachedRegion<'c, 't>>,
    location: L,
) -> Result<DetachedAlternativesOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.alternatives", location).add_results(result_types);
    if let Some(scope) = scope {
        builder = builder.add_operand(scope);
    }
    builder.add_regions(alternatives).build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::alternatives`"))
    })
}

/// Name of the `transform.annotate` attribute that stores the payload attribute name.
pub const NAME_ATTRIBUTE: &str = "name";

/// Operation trait for `transform.annotate`.
pub trait AnnotateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handle to annotate.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional parameter that provides annotation values.
    fn param(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() <= 1 { Ok(None) } else { self.operand_value(1).map(Some) }
    }

    /// Returns the name of the attribute to add to the targeted payload operations.
    fn attribute_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(NAME_ATTRIBUTE)?.string())
    }
}

mlir_op!(Annotate);
mlir_op_trait!(Annotate, ZeroSuccessors);

/// Constructs a new detached/owned [`AnnotateOperation`] at the specified [`Location`].
pub fn annotate<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    name: &str,
    param: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> Result<DetachedAnnotateOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.annotate", location)
        .add_operand(target)
        .add_attribute(NAME_ATTRIBUTE, context.string_attribute(name));
    if let Some(param) = param {
        builder = builder.add_operand(param);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::annotate`"))
    })
}

/// Operation trait for `transform.apply_cse`.
pub trait ApplyCommonSubexpressionEliminationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handle whose nested payload IR is rewritten.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(ApplyCommonSubexpressionElimination);
mlir_op_trait!(ApplyCommonSubexpressionElimination, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyCommonSubexpressionEliminationOperation`] at the specified [`Location`].
pub fn apply_cse<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedApplyCommonSubexpressionEliminationOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.apply_cse", location)
        .add_operand(target)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::apply_cse`"))
        })
}

/// Name of the `transform.apply_conversion_patterns` legal-operations attribute.
pub const LEGAL_OPS_ATTRIBUTE: &str = "legal_ops";

/// Name of the `transform.apply_conversion_patterns` illegal-operations attribute.
pub const ILLEGAL_OPS_ATTRIBUTE: &str = "illegal_ops";

/// Name of the `transform.apply_conversion_patterns` legal-dialects attribute.
pub const LEGAL_DIALECTS_ATTRIBUTE: &str = "legal_dialects";

/// Name of the `transform.apply_conversion_patterns` illegal-dialects attribute.
pub const ILLEGAL_DIALECTS_ATTRIBUTE: &str = "illegal_dialects";

/// Name of the `transform.apply_conversion_patterns` partial-conversion marker attribute.
pub const PARTIAL_CONVERSION_ATTRIBUTE: &str = "partial_conversion";

/// Name of the `transform.apply_conversion_patterns` preserve-handles marker attribute.
pub const PRESERVE_HANDLES_ATTRIBUTE: &str = "preserve_handles";

/// Operation trait for `transform.apply_conversion_patterns`.
pub trait ApplyConversionPatternsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handle whose nested payload IR is converted.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the graph region containing conversion pattern descriptors.
    fn patterns(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the optional graph region containing a default type converter.
    fn default_type_converter_region(&self) -> Result<Option<RegionRef<'o, 'c, 't>>, Error> {
        if self.region_count() <= 1 { Ok(None) } else { self.region(1).map(Some) }
    }

    /// Returns whether partial dialect conversion is requested.
    fn partial_conversion(&self) -> bool {
        self.has_attribute(PARTIAL_CONVERSION_ATTRIBUTE)
    }

    /// Returns whether this operation should preserve handles using tracking-listener updates.
    fn preserve_handles(&self) -> bool {
        self.has_attribute(PRESERVE_HANDLES_ATTRIBUTE)
    }
}

mlir_op!(ApplyConversionPatterns);
mlir_op_trait!(ApplyConversionPatterns, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyConversionPatternsOperation`] at the specified [`Location`].
pub fn apply_conversion_patterns<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    patterns: DetachedRegion<'c, 't>,
    default_type_converter_region: Option<DetachedRegion<'c, 't>>,
    partial_conversion: bool,
    preserve_handles: bool,
    location: L,
) -> Result<DetachedApplyConversionPatternsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.apply_conversion_patterns", location)
        .add_operand(target)
        .add_region(patterns);
    if let Some(region) = default_type_converter_region {
        builder = builder.add_region(region);
    }
    if partial_conversion {
        builder = builder.add_attribute(PARTIAL_CONVERSION_ATTRIBUTE, context.unit_attribute());
    }
    if preserve_handles {
        builder = builder.add_attribute(PRESERVE_HANDLES_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::apply_conversion_patterns`"))
    })
}

/// Name of the `transform.apply_conversion_patterns.dialect_to_llvm` dialect-name attribute.
pub const DIALECT_NAME_ATTRIBUTE: &str = "dialect_name";

/// Operation trait for `transform.apply_conversion_patterns.dialect_to_llvm`.
pub trait ApplyToLlvmConversionPatternsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source dialect whose operations are converted to LLVM dialect operations.
    fn dialect_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(DIALECT_NAME_ATTRIBUTE)?.string())
    }
}

mlir_op!(ApplyToLlvmConversionPatterns);
mlir_op_trait!(ApplyToLlvmConversionPatterns, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyToLlvmConversionPatternsOperation`] at the specified [`Location`].
pub fn apply_to_llvm_conversion_patterns<'c, 't: 'c, L: Location<'c, 't>>(
    dialect_name: &str,
    location: L,
) -> Result<DetachedApplyToLlvmConversionPatternsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.apply_conversion_patterns.dialect_to_llvm", location)
        .add_attribute(DIALECT_NAME_ATTRIBUTE, context.string_attribute(dialect_name))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| {
                Error::invalid_argument("invalid arguments to `transform::apply_to_llvm_conversion_patterns`")
            })
        })
}

/// Operation trait for `transform.apply_dce`.
pub trait ApplyDeadCodeEliminationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handle whose nested payload IR is rewritten.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(ApplyDeadCodeElimination);
mlir_op_trait!(ApplyDeadCodeElimination, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyDeadCodeEliminationOperation`] at the specified [`Location`].
pub fn apply_dce<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedApplyDeadCodeEliminationOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.apply_dce", location)
        .add_operand(target)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::apply_dce`"))
        })
}

/// Name of the `transform.apply_patterns` CSE marker attribute.
pub const APPLY_CSE_ATTRIBUTE: &str = "apply_cse";

/// Name of the `transform.apply_patterns` maximum-iteration attribute.
pub const MAX_ITERATIONS_ATTRIBUTE: &str = "max_iterations";

/// Name of the `transform.apply_patterns` maximum-rewrite attribute.
pub const MAX_NUM_REWRITES_ATTRIBUTE: &str = "max_num_rewrites";

/// Operation trait for `transform.apply_patterns`.
pub trait ApplyPatternsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handle whose nested payload IR is rewritten.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the graph region containing pattern descriptors.
    fn patterns(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns whether greedy pattern application should be interleaved with CSE.
    fn apply_cse(&self) -> bool {
        self.has_attribute(APPLY_CSE_ATTRIBUTE)
    }
}

mlir_op!(ApplyPatterns);
mlir_op_trait!(ApplyPatterns, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyPatternsOperation`] at the specified [`Location`].
pub fn apply_patterns<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    apply_cse: bool,
    max_iterations: Option<i64>,
    max_num_rewrites: Option<i64>,
    patterns: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedApplyPatternsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder =
        OperationBuilder::new("transform.apply_patterns", location).add_operand(target).add_region(patterns);
    if apply_cse {
        builder = builder.add_attribute(APPLY_CSE_ATTRIBUTE, context.unit_attribute());
    }
    if let Some(max_iterations) = max_iterations {
        builder = builder.add_attribute(
            MAX_ITERATIONS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), max_iterations),
        );
    }
    if let Some(max_num_rewrites) = max_num_rewrites {
        builder = builder.add_attribute(
            MAX_NUM_REWRITES_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), max_num_rewrites),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::apply_patterns`"))
    })
}

/// Operation trait for `transform.apply_patterns.canonicalization`.
pub trait ApplyCanonicalizationPatternsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(ApplyCanonicalizationPatterns);
mlir_op_trait!(ApplyCanonicalizationPatterns, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyCanonicalizationPatternsOperation`] at the specified [`Location`].
pub fn apply_canonicalization_patterns<'c, 't: 'c, L: Location<'c, 't>>(
    location: L,
) -> Result<DetachedApplyCanonicalizationPatternsOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.apply_patterns.canonicalization", location)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| {
                Error::invalid_argument("invalid arguments to `transform::apply_canonicalization_patterns`")
            })
        })
}

/// Operation trait for `transform.apply_licm`.
pub trait ApplyLoopInvariantCodeMotionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target loop-like payload handle.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(ApplyLoopInvariantCodeMotion);
mlir_op_trait!(ApplyLoopInvariantCodeMotion, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyLoopInvariantCodeMotionOperation`] at the specified [`Location`].
pub fn apply_licm<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedApplyLoopInvariantCodeMotionOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.apply_licm", location)
        .add_operand(target)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::apply_licm`"))
        })
}

/// Name of the `transform.apply_registered_pass` pass-name attribute.
pub const PASS_NAME_ATTRIBUTE: &str = "pass_name";

/// Name of the `transform.apply_registered_pass` static options attribute.
pub const OPTIONS_ATTRIBUTE: &str = "options";

/// Operation trait for `transform.apply_registered_pass`.
pub trait ApplyRegisteredPassOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handle passed to the pass or pass pipeline.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the dynamic option parameters.
    fn dynamic_options(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values().skip(1)
    }

    /// Returns the updated handle for the pass target.
    fn result(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.as_ref().as_ref().result(0)?.into())
    }
}

mlir_op!(ApplyRegisteredPass);
mlir_op_trait!(ApplyRegisteredPass, ZeroSuccessors);

/// Constructs a new detached/owned [`ApplyRegisteredPassOperation`] at the specified [`Location`].
pub fn apply_registered_pass<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    pass_name: &str,
    result_type: TypeRef<'c, 't>,
    dynamic_options: &[ValueRef<'v, 'c, 't>],
    options: Option<DictionaryAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedApplyRegisteredPassOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.apply_registered_pass", location)
        .add_operand(target)
        .add_operands(dynamic_options)
        .add_result(result_type)
        .add_attribute(PASS_NAME_ATTRIBUTE, context.string_attribute(pass_name));
    if let Some(options) = options {
        builder = builder.add_attribute(OPTIONS_ATTRIBUTE, options);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::apply_registered_pass`"))
    })
}

/// Operation trait for `transform.cast`.
pub trait CastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input handle being cast.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the cast output handle.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.into())
    }
}

mlir_op!(Cast);
mlir_op_trait!(Cast, ZeroSuccessors);

/// Constructs a new detached/owned [`CastOperation`] at the specified [`Location`].
pub fn cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedCastOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.cast", location)
        .add_operand(input)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::cast`"))
        })
}

/// Operation trait for `transform.num_associations`.
pub trait NumAssociationsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the handle or parameter whose payload associations are counted.
    fn handle(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the produced parameter containing the association count.
    fn num(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.into())
    }
}

mlir_op!(NumAssociations);
mlir_op_trait!(NumAssociations, ZeroSuccessors);

/// Constructs a new detached/owned [`NumAssociationsOperation`] at the specified [`Location`].
pub fn num_associations<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    handle: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedNumAssociationsOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.num_associations", location)
        .add_operand(handle)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::num_associations`"))
        })
}

/// Name of the symbolic matcher attribute used by `transform.collect_matching`.
pub const MATCHER_ATTRIBUTE: &str = "matcher";

/// Operation trait for `transform.collect_matching`.
pub trait CollectMatchingOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the root handle under which matching is performed.
    fn root(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the collected handles or parameters.
    fn collected_results(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.results().map(|result| result.map(ValueRef::from))
    }
}

mlir_op!(CollectMatching);
mlir_op_trait!(CollectMatching, ZeroSuccessors);

/// Constructs a new detached/owned [`CollectMatchingOperation`] at the specified [`Location`].
pub fn collect_matching<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    root: ValueRef<'v, 'c, 't>,
    matcher: &str,
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> Result<DetachedCollectMatchingOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.collect_matching", location)
        .add_operand(root)
        .add_results(result_types)
        .add_attribute(MATCHER_ATTRIBUTE, context.flat_symbol_ref_attribute(matcher))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::collect_matching`"))
        })
}

/// Name of the `transform.foreach_match` restrict-root marker attribute.
pub const RESTRICT_ROOT_ATTRIBUTE: &str = "restrict_root";

/// Name of the `transform.foreach_match` flatten-results marker attribute.
pub const FLATTEN_RESULTS_ATTRIBUTE: &str = "flatten_results";

/// Name of the `transform.foreach_match` matcher-symbols attribute.
pub const MATCHERS_ATTRIBUTE: &str = "matchers";

/// Name of the `transform.foreach_match` action-symbols attribute.
pub const ACTIONS_ATTRIBUTE: &str = "actions";

/// Operation trait for `transform.foreach_match`.
pub trait ForeachMatchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the root handle walked for matcher/action pairs.
    fn root(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the forwarded input handles and parameters.
    fn forwarded_inputs(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values().skip(1)
    }

    /// Returns the updated root handle.
    fn updated(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.into())
    }

    /// Returns the forwarded outputs accumulated from successful actions.
    fn forwarded_outputs(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.results().skip(1).map(|result| result.map(ValueRef::from))
    }
}

mlir_op!(ForeachMatch);
mlir_op_trait!(ForeachMatch, ZeroSuccessors);

/// Constructs a new detached/owned [`ForeachMatchOperation`] at the specified [`Location`].
pub fn foreach_match<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    root: ValueRef<'v, 'c, 't>,
    forwarded_inputs: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    matchers: &[&str],
    actions: &[&str],
    restrict_root: bool,
    flatten_results: bool,
    location: L,
) -> Result<DetachedForeachMatchOperation<'c, 't>, Error> {
    let context = location.context();
    let matcher_attributes =
        matchers.iter().map(|matcher| context.flat_symbol_ref_attribute(*matcher)).collect::<Vec<_>>();
    let action_attributes = actions.iter().map(|action| context.flat_symbol_ref_attribute(*action)).collect::<Vec<_>>();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.foreach_match", location)
        .add_operand(root)
        .add_operands(forwarded_inputs)
        .add_results(result_types)
        .add_attribute(MATCHERS_ATTRIBUTE, context.array_attribute(&matcher_attributes))
        .add_attribute(ACTIONS_ATTRIBUTE, context.array_attribute(&action_attributes));
    if restrict_root {
        builder = builder.add_attribute(RESTRICT_ROOT_ATTRIBUTE, context.unit_attribute());
    }
    if flatten_results {
        builder = builder.add_attribute(FLATTEN_RESULTS_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::foreach_match`"))
    })
}

/// Name of the `transform.foreach` zip-shortest marker attribute.
pub const WITH_ZIP_SHORTEST_ATTRIBUTE: &str = "with_zip_shortest";

/// Operation trait for `transform.foreach`.
pub trait ForeachOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handles and parameters iterated by this operation.
    fn targets(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }

    /// Returns whether iteration should stop at the shortest target payload list.
    fn with_zip_shortest(&self) -> bool {
        self.has_attribute(WITH_ZIP_SHORTEST_ATTRIBUTE)
    }

    /// Returns the body region executed for each payload element.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(Foreach);
mlir_op_trait!(Foreach, ZeroSuccessors);

/// Constructs a new detached/owned [`ForeachOperation`] at the specified [`Location`].
pub fn foreach<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    targets: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    with_zip_shortest: bool,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedForeachOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.foreach", location)
        .add_operands(targets)
        .add_results(result_types)
        .add_region(body);
    if with_zip_shortest {
        builder = builder.add_attribute(WITH_ZIP_SHORTEST_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::foreach`"))
    })
}

/// Name of the `transform.get_consumers_of_result` result-number attribute.
pub const RESULT_NUMBER_ATTRIBUTE: &str = "result_number";

/// Operation trait for `transform.get_consumers_of_result`.
pub trait GetConsumersOfResultOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the payload operation handle whose result consumers are queried.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the result number whose consumers are queried.
    fn result_number(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(RESULT_NUMBER_ATTRIBUTE)?.signless_value())
    }
}

mlir_op!(GetConsumersOfResult);
mlir_op_trait!(GetConsumersOfResult, ZeroSuccessors);

/// Constructs a new detached/owned [`GetConsumersOfResultOperation`] at the specified [`Location`].
pub fn get_consumers_of_result<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    result_number: i64,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGetConsumersOfResultOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.get_consumers_of_result", location)
        .add_operand(target)
        .add_result(result_type)
        .add_attribute(
            RESULT_NUMBER_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), result_number),
        )
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::get_consumers_of_result`"))
        })
}

/// Operation trait for `transform.get_defining_op`.
pub trait GetDefiningOpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value handle whose defining operation is queried.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(GetDefiningOp);
mlir_op_trait!(GetDefiningOp, ZeroSuccessors);

/// Constructs a new detached/owned [`GetDefiningOpOperation`] at the specified [`Location`].
pub fn get_defining_op<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGetDefiningOpOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.get_defining_op", location)
        .add_operand(target)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::get_defining_op`"))
        })
}

/// Name of the `transform.get_parent_op` isolated-from-above marker attribute.
pub const ISOLATED_FROM_ABOVE_ATTRIBUTE: &str = "isolated_from_above";

/// Name of the `transform.get_parent_op` allow-empty-results marker attribute.
pub const ALLOW_EMPTY_RESULTS_ATTRIBUTE: &str = "allow_empty_results";

/// Name of the `transform.get_parent_op` operation-name attribute.
pub const OP_NAME_ATTRIBUTE: &str = "op_name";

/// Name of the `transform.get_parent_op` deduplicate marker attribute.
pub const DEDUPLICATE_ATTRIBUTE: &str = "deduplicate";

/// Name of the `transform.get_parent_op` parent-depth attribute.
pub const NTH_PARENT_ATTRIBUTE: &str = "nth_parent";

/// Operation trait for `transform.get_parent_op`.
pub trait GetParentOpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handle whose parent operations are queried.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the required parent operation name, if one was specified.
    fn op_name(&self) -> Result<Option<StringRef<'c>>, Error> {
        if self.has_attribute(OP_NAME_ATTRIBUTE) {
            self.string_attribute(OP_NAME_ATTRIBUTE).map(|attribute| Some(attribute.string()))
        } else {
            Ok(None)
        }
    }
}

mlir_op!(GetParentOp);
mlir_op_trait!(GetParentOp, ZeroSuccessors);

/// Constructs a new detached/owned [`GetParentOpOperation`] at the specified [`Location`].
pub fn get_parent_op<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    isolated_from_above: bool,
    allow_empty_results: bool,
    op_name: Option<&str>,
    deduplicate: bool,
    nth_parent: Option<i64>,
    location: L,
) -> Result<DetachedGetParentOpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.get_parent_op", location)
        .add_operand(target)
        .add_result(result_type);
    if isolated_from_above {
        builder = builder.add_attribute(ISOLATED_FROM_ABOVE_ATTRIBUTE, context.unit_attribute());
    }
    if allow_empty_results {
        builder = builder.add_attribute(ALLOW_EMPTY_RESULTS_ATTRIBUTE, context.unit_attribute());
    }
    if let Some(op_name) = op_name {
        builder = builder.add_attribute(OP_NAME_ATTRIBUTE, context.string_attribute(op_name));
    }
    if deduplicate {
        builder = builder.add_attribute(DEDUPLICATE_ATTRIBUTE, context.unit_attribute());
    }
    if let Some(nth_parent) = nth_parent {
        builder = builder.add_attribute(
            NTH_PARENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), nth_parent),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::get_parent_op`"))
    })
}

/// Name of the `transform.get_producer_of_operand` operand-number attribute.
pub const OPERAND_NUMBER_ATTRIBUTE: &str = "operand_number";

/// Operation trait for `transform.get_producer_of_operand`.
pub trait GetProducerOfOperandOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the payload operation handle whose operand producer is queried.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the operand number whose producer is queried.
    fn operand_number(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(OPERAND_NUMBER_ATTRIBUTE)?.signless_value())
    }
}

mlir_op!(GetProducerOfOperand);
mlir_op_trait!(GetProducerOfOperand, ZeroSuccessors);

/// Constructs a new detached/owned [`GetProducerOfOperandOperation`] at the specified [`Location`].
pub fn get_producer_of_operand<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    operand_number: i64,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGetProducerOfOperandOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.get_producer_of_operand", location)
        .add_operand(target)
        .add_result(result_type)
        .add_attribute(
            OPERAND_NUMBER_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), operand_number),
        )
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::get_producer_of_operand`"))
        })
}

/// Name of the `transform.get_operand` / `transform.get_result` raw-position-list attribute.
pub const RAW_POSITION_LIST_ATTRIBUTE: &str = "raw_position_list";

/// Name of the `transform.get_operand` / `transform.get_result` inverted-position marker attribute.
pub const IS_INVERTED_ATTRIBUTE: &str = "is_inverted";

/// Name of the `transform.get_operand` / `transform.get_result` all-positions marker attribute.
pub const IS_ALL_ATTRIBUTE: &str = "is_all";

/// Operation trait for `transform.get_operand`.
pub trait GetOperandOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the payload operation handle whose operands are queried.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the raw operand positions attribute.
    fn raw_position_list(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(RAW_POSITION_LIST_ATTRIBUTE)
    }
}

mlir_op!(GetOperand);
mlir_op_trait!(GetOperand, ZeroSuccessors);

/// Constructs a new detached/owned [`GetOperandOperation`] at the specified [`Location`].
pub fn get_operand<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    positions: &[i64],
    is_inverted: bool,
    is_all: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGetOperandOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.get_operand", location)
        .add_operand(target)
        .add_result(result_type)
        .add_attribute(RAW_POSITION_LIST_ATTRIBUTE, context.dense_i64_array_attribute(positions)?);
    if is_inverted {
        builder = builder.add_attribute(IS_INVERTED_ATTRIBUTE, context.unit_attribute());
    }
    if is_all {
        builder = builder.add_attribute(IS_ALL_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::get_operand`"))
    })
}

/// Operation trait for `transform.get_result`.
pub trait GetResultOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the payload operation handle whose results are queried.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the raw result positions attribute.
    fn raw_position_list(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(RAW_POSITION_LIST_ATTRIBUTE)
    }
}

mlir_op!(GetResult);
mlir_op_trait!(GetResult, ZeroSuccessors);

/// Constructs a new detached/owned [`GetResultOperation`] at the specified [`Location`].
pub fn get_result<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    positions: &[i64],
    is_inverted: bool,
    is_all: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGetResultOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.get_result", location)
        .add_operand(target)
        .add_result(result_type)
        .add_attribute(RAW_POSITION_LIST_ATTRIBUTE, context.dense_i64_array_attribute(positions)?);
    if is_inverted {
        builder = builder.add_attribute(IS_INVERTED_ATTRIBUTE, context.unit_attribute());
    }
    if is_all {
        builder = builder.add_attribute(IS_ALL_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::get_result`"))
    })
}

/// Name of the `transform.get_type` elemental marker attribute.
pub const ELEMENTAL_ATTRIBUTE: &str = "elemental";

/// Operation trait for `transform.get_type`.
pub trait GetTypeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value handle whose associated payload types are extracted.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns whether tensor/vector element types are extracted.
    fn elemental(&self) -> bool {
        self.has_attribute(ELEMENTAL_ATTRIBUTE)
    }
}

mlir_op!(GetType);
mlir_op_trait!(GetType, ZeroSuccessors);

/// Constructs a new detached/owned [`GetTypeOperation`] at the specified [`Location`].
pub fn get_type<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'v, 'c, 't>,
    elemental: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGetTypeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.get_type", location).add_operand(value).add_result(result_type);
    if elemental {
        builder = builder.add_attribute(ELEMENTAL_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::get_type`"))
    })
}

/// Name of the `transform.include` target symbol attribute.
pub const TARGET_ATTRIBUTE: &str = "target";

/// Name of the failure-propagation-mode attribute used by transform container operations.
pub const FAILURE_PROPAGATION_MODE_ATTRIBUTE: &str = "failure_propagation_mode";

/// Operation trait for `transform.include`.
pub trait IncludeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operands forwarded to the included named sequence.
    fn operands(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }

    /// Returns the failure propagation mode.
    fn failure_propagation_mode(&self) -> Result<FailurePropagationMode, Error> {
        self.attribute(FAILURE_PROPAGATION_MODE_ATTRIBUTE)?
            .and_then(|attribute| attribute.cast::<FailurePropagationModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    FAILURE_PROPAGATION_MODE_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }
}

mlir_op!(Include);
mlir_op_trait!(Include, ZeroSuccessors);

/// Constructs a new detached/owned [`IncludeOperation`] at the specified [`Location`].
pub fn include<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: &str,
    failure_propagation_mode: FailurePropagationMode,
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> Result<DetachedIncludeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.include", location)
        .add_operands(operands)
        .add_results(result_types)
        .add_attribute(TARGET_ATTRIBUTE, context.flat_symbol_ref_attribute(target))
        .add_attribute(
            FAILURE_PROPAGATION_MODE_ATTRIBUTE,
            context.transform_failure_propagation_mode_attribute(failure_propagation_mode)?,
        )
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::include`"))
        })
}

/// Name of the `transform.match.operation_empty` operand-handle.
pub const OPERAND_HANDLE_ATTRIBUTE: &str = "operand_handle";

/// Operation trait for `transform.match.operation_empty`.
pub trait MatchOperationEmptyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the handle being matched.
    fn operand_handle(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(MatchOperationEmpty);
mlir_op_trait!(MatchOperationEmpty, ZeroSuccessors);

/// Constructs a new detached/owned [`MatchOperationEmptyOperation`] at the specified [`Location`].
pub fn match_operation_empty<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operand_handle: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedMatchOperationEmptyOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.match.operation_empty", location)
        .add_operand(operand_handle)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::match_operation_empty`"))
        })
}

/// Name of the `transform.match.operation_name` operation-names attribute.
pub const OP_NAMES_ATTRIBUTE: &str = "op_names";

/// Operation trait for `transform.match.operation_name`.
pub trait MatchOperationNameOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the handle being matched.
    fn operand_handle(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(MatchOperationName);
mlir_op_trait!(MatchOperationName, ZeroSuccessors);

/// Constructs a new detached/owned [`MatchOperationNameOperation`] at the specified [`Location`].
pub fn match_operation_name<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operand_handle: ValueRef<'v, 'c, 't>,
    op_names: &[&str],
    location: L,
) -> Result<DetachedMatchOperationNameOperation<'c, 't>, Error> {
    let context = location.context();
    let op_name_attributes = op_names.iter().map(|op_name| context.string_attribute(*op_name)).collect::<Vec<_>>();
    context.load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.match.operation_name", location)
        .add_operand(operand_handle)
        .add_attribute(OP_NAMES_ATTRIBUTE, context.array_attribute(&op_name_attributes))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::match_operation_name`"))
        })
}

/// Name of the `transform.match.param.cmpi` predicate attribute.
pub const PREDICATE_ATTRIBUTE: &str = "predicate";

/// Operation trait for `transform.match.param.cmpi`.
pub trait MatchParamCmpiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the parameter being matched.
    fn param(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the reference parameter being compared against.
    fn reference(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the signed integer comparison predicate.
    fn predicate(&self) -> Result<MatchCmpIPredicate, Error> {
        self.attribute(PREDICATE_ATTRIBUTE)?
            .and_then(|attribute| attribute.cast::<super::MatchCmpIPredicateAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    PREDICATE_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }
}

mlir_op!(MatchParamCmpi);
mlir_op_trait!(MatchParamCmpi, ZeroSuccessors);

/// Constructs a new detached/owned [`MatchParamCmpiOperation`] at the specified [`Location`].
pub fn match_param_cmpi<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    param: ValueRef<'v, 'c, 't>,
    reference: ValueRef<'v, 'c, 't>,
    predicate: MatchCmpIPredicate,
    location: L,
) -> Result<DetachedMatchParamCmpiOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.match.param.cmpi", location)
        .add_operand(param)
        .add_operand(reference)
        .add_attribute(PREDICATE_ATTRIBUTE, context.transform_match_cmp_i_predicate_attribute(predicate)?)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::match_param_cmpi`"))
        })
}

/// Operation trait for `transform.merge_handles`.
pub trait MergeHandlesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the handles or parameters being merged.
    fn handles(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }

    /// Returns whether duplicate payload associations should be removed.
    fn deduplicate(&self) -> bool {
        self.has_attribute(DEDUPLICATE_ATTRIBUTE)
    }
}

mlir_op!(MergeHandles);
mlir_op_trait!(MergeHandles, ZeroSuccessors);

/// Constructs a new detached/owned [`MergeHandlesOperation`] at the specified [`Location`].
pub fn merge_handles<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    handles: &[ValueRef<'v, 'c, 't>],
    deduplicate: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedMergeHandlesOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.merge_handles", location)
        .add_operands(handles)
        .add_result(result_type);
    if deduplicate {
        builder = builder.add_attribute(DEDUPLICATE_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::merge_handles`"))
    })
}

/// Name of the symbol-name attribute used by symbol operations.
pub const SYMBOL_NAME_ATTRIBUTE: &str = "sym_name";

/// Name of the function-type attribute used by `transform.named_sequence`.
pub const FUNCTION_TYPE_ATTRIBUTE: &str = "function_type";

/// Operation trait for `transform.named_sequence`.
pub trait NamedSequenceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the body region containing the named sequence.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(NamedSequence);
mlir_op_trait!(NamedSequence, ZeroSuccessors);

/// Constructs a new detached/owned [`NamedSequenceOperation`] at the specified [`Location`].
pub fn named_sequence<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    symbol_name: &str,
    function_type: T,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedNamedSequenceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.named_sequence", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, context.string_attribute(symbol_name))
        .add_attribute(FUNCTION_TYPE_ATTRIBUTE, context.type_attribute(function_type))
        .add_region(body)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::named_sequence`"))
        })
}

/// Name of the `transform.split_handle` empty-handle pass-through attribute.
pub const PASS_THROUGH_EMPTY_HANDLE_ATTRIBUTE: &str = "pass_through_empty_handle";

/// Name of the `transform.split_handle` too-small payload failure attribute.
pub const FAIL_ON_PAYLOAD_TOO_SMALL_ATTRIBUTE: &str = "fail_on_payload_too_small";

/// Name of the `transform.split_handle` overflow-result attribute.
pub const OVERFLOW_RESULT_ATTRIBUTE: &str = "overflow_result";

/// Operation trait for `transform.split_handle`.
pub trait SplitHandleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the handle or parameter being split.
    fn handle(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(SplitHandle);
mlir_op_trait!(SplitHandle, ZeroSuccessors);

/// Constructs a new detached/owned [`SplitHandleOperation`] at the specified [`Location`].
pub fn split_handle<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    handle: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    pass_through_empty_handle: Option<BooleanAttributeRef<'c, 't>>,
    fail_on_payload_too_small: Option<BooleanAttributeRef<'c, 't>>,
    overflow_result: Option<i64>,
    location: L,
) -> Result<DetachedSplitHandleOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.split_handle", location)
        .add_operand(handle)
        .add_results(result_types);
    if let Some(pass_through_empty_handle) = pass_through_empty_handle {
        builder = builder.add_attribute(PASS_THROUGH_EMPTY_HANDLE_ATTRIBUTE, pass_through_empty_handle);
    }
    if let Some(fail_on_payload_too_small) = fail_on_payload_too_small {
        builder = builder.add_attribute(FAIL_ON_PAYLOAD_TOO_SMALL_ATTRIBUTE, fail_on_payload_too_small);
    }
    if let Some(overflow_result) = overflow_result {
        builder = builder.add_attribute(
            OVERFLOW_RESULT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), overflow_result),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::split_handle`"))
    })
}

/// Name of the `transform.param.constant` value attribute.
pub const VALUE_ATTRIBUTE: &str = "value";

/// Operation trait for `transform.param.constant`.
pub trait ParamConstantOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the produced transform parameter.
    fn param(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.into())
    }
}

mlir_op!(ParamConstant);
mlir_op_trait!(ParamConstant, ZeroSuccessors);

/// Constructs a new detached/owned [`ParamConstantOperation`] at the specified [`Location`].
pub fn param_constant<'c, 't: 'c, A: Attribute<'c, 't>, L: Location<'c, 't>>(
    value: A,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedParamConstantOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.param.constant", location)
        .add_attribute(VALUE_ATTRIBUTE, value)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::param_constant`"))
        })
}

/// Name of the `transform.print` assume-verified marker attribute.
pub const ASSUME_VERIFIED_ATTRIBUTE: &str = "assume_verified";

/// Name of the `transform.print` use-local-scope marker attribute.
pub const USE_LOCAL_SCOPE_ATTRIBUTE: &str = "use_local_scope";

/// Name of the `transform.print` skip-regions marker attribute.
pub const SKIP_REGIONS_ATTRIBUTE: &str = "skip_regions";

/// Operation trait for `transform.print`.
pub trait PrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional target handle to print.
    fn target(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() == 0 { Ok(None) } else { self.operand_value(0).map(Some) }
    }
}

mlir_op!(Print);
mlir_op_trait!(Print, ZeroSuccessors);

/// Constructs a new detached/owned [`PrintOperation`] at the specified [`Location`].
pub fn print<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: Option<ValueRef<'v, 'c, 't>>,
    name: Option<&str>,
    assume_verified: bool,
    use_local_scope: bool,
    skip_regions: bool,
    location: L,
) -> Result<DetachedPrintOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.print", location);
    if let Some(target) = target {
        builder = builder.add_operand(target);
    }
    if let Some(name) = name {
        builder = builder.add_attribute(NAME_ATTRIBUTE, context.string_attribute(name));
    }
    if assume_verified {
        builder = builder.add_attribute(ASSUME_VERIFIED_ATTRIBUTE, context.unit_attribute());
    }
    if use_local_scope {
        builder = builder.add_attribute(USE_LOCAL_SCOPE_ATTRIBUTE, context.unit_attribute());
    }
    if skip_regions {
        builder = builder.add_attribute(SKIP_REGIONS_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::print`"))
    })
}

/// Operation trait for `transform.replicate`.
pub trait ReplicateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the pattern handle that determines the replication count.
    fn pattern(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the handles or parameters being replicated.
    fn handles(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values().skip(1)
    }
}

mlir_op!(Replicate);
mlir_op_trait!(Replicate, ZeroSuccessors);

/// Constructs a new detached/owned [`ReplicateOperation`] at the specified [`Location`].
pub fn replicate<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    pattern: ValueRef<'v, 'c, 't>,
    handles: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> Result<DetachedReplicateOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.replicate", location)
        .add_operand(pattern)
        .add_operands(handles)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::replicate`"))
        })
}

/// Operation trait for `transform.select`.
pub trait SelectOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handle whose payload operations are filtered.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the selected operation name.
    fn op_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(OP_NAME_ATTRIBUTE)?.string())
    }
}

mlir_op!(Select);
mlir_op_trait!(Select, ZeroSuccessors);

/// Constructs a new detached/owned [`SelectOperation`] at the specified [`Location`].
pub fn select<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    op_name: &str,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedSelectOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.select", location)
        .add_operand(target)
        .add_result(result_type)
        .add_attribute(OP_NAME_ATTRIBUTE, context.string_attribute(op_name))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::select`"))
        })
}

/// Operation trait for `transform.sequence`.
pub trait SequenceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional root handle operand.
    fn root(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() == 0 { Ok(None) } else { self.operand_value(0).map(Some) }
    }

    /// Returns the body region containing the ordered transform operations.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns the failure propagation mode.
    fn failure_propagation_mode(&self) -> Result<FailurePropagationMode, Error> {
        self.attribute(FAILURE_PROPAGATION_MODE_ATTRIBUTE)?
            .and_then(|attribute| attribute.cast::<FailurePropagationModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    FAILURE_PROPAGATION_MODE_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }
}

mlir_op!(Sequence);
mlir_op_trait!(Sequence, ZeroSuccessors);

/// Constructs a new detached/owned [`SequenceOperation`] at the specified [`Location`].
pub fn sequence<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    failure_propagation_mode: FailurePropagationMode,
    root: Option<ValueRef<'v, 'c, 't>>,
    extra_bindings: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedSequenceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::transform()?)?;
    let mut builder = OperationBuilder::new("transform.sequence", location)
        .add_results(result_types)
        .add_attribute(
            FAILURE_PROPAGATION_MODE_ATTRIBUTE,
            context.transform_failure_propagation_mode_attribute(failure_propagation_mode)?,
        )
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[if root.is_some() { 1 } else { 0 }, extra_bindings.len() as i32])?,
        )
        .add_region(body);
    if let Some(root) = root {
        builder = builder.add_operand(root);
    }
    builder = builder.add_operands(extra_bindings);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::sequence`"))
    })
}

/// Operation trait for `transform.verify`.
pub trait VerifyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the target handle being verified.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(Verify);
mlir_op_trait!(Verify, ZeroSuccessors);

/// Constructs a new detached/owned [`VerifyOperation`] at the specified [`Location`].
pub fn verify<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    target: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedVerifyOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.verify", location)
        .add_operand(target)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::verify`"))
        })
}

/// Operation trait for `transform.yield`.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the transform values yielded back to the parent operation.
    fn yielded_values(&self) -> impl Iterator<Item = Result<ValueRef<'o, 'c, 't>, Error>> {
        self.operand_values()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, SingleBlockRegions);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);
mlir_op_trait!(Yield, IsTerminator);

/// Constructs a new detached/owned [`YieldOperation`] at the specified [`Location`].
pub fn r#yield<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::transform()?)?;
    OperationBuilder::new("transform.yield", location)
        .add_operands(operands)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `transform::yield`"))
        })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Block, Context, Operation, Type};

    use super::*;

    fn assert_operation_name<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(operation: &O, expected: &str) {
        assert_eq!(operation.name().as_str(), Ok(expected));
    }

    #[test]
    fn test_operation_constructors() {
        let context = Context::new();
        let location = context.unknown_location();
        let handle_type = context.transform_any_op_type().unwrap().as_ref();
        let value_type = context.transform_any_value_type().unwrap().as_ref();
        let param_type = context.transform_param_type(context.signless_integer_type(32)).unwrap().as_ref();
        let type_param_type = context.transform_type_param_type().unwrap().as_ref();
        let block = context.block(&[(context.transform_any_op_type().unwrap(), location)]);
        let target = block.argument(0).unwrap().into();

        let operation = alternatives(Some(target), &[handle_type], vec![context.region()], location).unwrap();
        assert_operation_name(&operation, "transform.alternatives");
        assert_eq!(operation.scope().unwrap(), Some(target));
        assert_eq!(operation.alternatives().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);

        let operation = annotate(target, "ryft.annotation", Some(target), location).unwrap();
        assert_operation_name(&operation, "transform.annotate");
        assert_eq!(operation.target().unwrap(), target);
        assert_eq!(operation.param().unwrap(), Some(target));
        assert_eq!(operation.attribute_name().unwrap().as_str(), Ok("ryft.annotation"));

        let operation = apply_cse(target, location).unwrap();
        assert_operation_name(&operation, "transform.apply_cse");
        assert_eq!(operation.target().unwrap(), target);

        let operation =
            apply_conversion_patterns(target, context.region(), Some(context.region()), true, true, location).unwrap();
        assert_operation_name(&operation, "transform.apply_conversion_patterns");
        assert_eq!(operation.target().unwrap(), target);
        assert!(operation.partial_conversion());
        assert!(operation.preserve_handles());

        let operation = apply_to_llvm_conversion_patterns("func", location).unwrap();
        assert_operation_name(&operation, "transform.apply_conversion_patterns.dialect_to_llvm");
        assert_eq!(operation.dialect_name().unwrap().as_str(), Ok("func"));

        let operation = apply_dce(target, location).unwrap();
        assert_operation_name(&operation, "transform.apply_dce");
        assert_eq!(operation.target().unwrap(), target);

        let operation = apply_patterns(target, true, Some(2), Some(3), context.region(), location).unwrap();
        assert_operation_name(&operation, "transform.apply_patterns");
        assert_eq!(operation.target().unwrap(), target);
        assert!(operation.apply_cse());

        let operation = apply_canonicalization_patterns(location).unwrap();
        assert_operation_name(&operation, "transform.apply_patterns.canonicalization");

        let operation = apply_licm(target, location).unwrap();
        assert_operation_name(&operation, "transform.apply_licm");
        assert_eq!(operation.target().unwrap(), target);

        let operation = apply_registered_pass(target, "canonicalize", handle_type, &[target], None, location).unwrap();
        assert_operation_name(&operation, "transform.apply_registered_pass");
        assert_eq!(operation.target().unwrap(), target);
        assert_eq!(operation.dynamic_options().collect::<Result<Vec<_>, _>>().unwrap(), vec![target]);

        let operation = cast(target, handle_type, location).unwrap();
        assert_operation_name(&operation, "transform.cast");
        assert_eq!(operation.input().unwrap(), target);

        let operation = num_associations(target, param_type, location).unwrap();
        assert_operation_name(&operation, "transform.num_associations");
        assert_eq!(operation.handle().unwrap(), target);

        let operation = collect_matching(target, "matcher", &[handle_type], location).unwrap();
        assert_operation_name(&operation, "transform.collect_matching");
        assert_eq!(operation.root().unwrap(), target);

        let operation =
            foreach_match(target, &[target], &[handle_type], &["matcher"], &["action"], true, true, location).unwrap();
        assert_operation_name(&operation, "transform.foreach_match");
        assert_eq!(operation.root().unwrap(), target);
        assert_eq!(operation.forwarded_inputs().collect::<Result<Vec<_>, _>>().unwrap(), vec![target]);

        let operation = foreach(&[target], &[handle_type], true, context.region(), location).unwrap();
        assert_operation_name(&operation, "transform.foreach");
        assert_eq!(operation.targets().collect::<Result<Vec<_>, _>>().unwrap(), vec![target]);
        assert!(operation.with_zip_shortest());

        let operation = get_consumers_of_result(target, 0, handle_type, location).unwrap();
        assert_operation_name(&operation, "transform.get_consumers_of_result");
        assert_eq!(operation.target().unwrap(), target);
        assert_eq!(operation.result_number().unwrap(), 0);

        let operation = get_defining_op(target, handle_type, location).unwrap();
        assert_operation_name(&operation, "transform.get_defining_op");
        assert_eq!(operation.target().unwrap(), target);

        let operation =
            get_parent_op(target, handle_type, true, true, Some("func.func"), true, Some(1), location).unwrap();
        assert_operation_name(&operation, "transform.get_parent_op");
        assert_eq!(operation.target().unwrap(), target);
        assert_eq!(operation.op_name().unwrap().unwrap().as_str(), Ok("func.func"));

        let operation = get_producer_of_operand(target, 0, handle_type, location).unwrap();
        assert_operation_name(&operation, "transform.get_producer_of_operand");
        assert_eq!(operation.target().unwrap(), target);
        assert_eq!(operation.operand_number().unwrap(), 0);

        let operation = get_operand(target, &[0, 2], false, false, value_type, location).unwrap();
        assert_operation_name(&operation, "transform.get_operand");
        assert_eq!(operation.target().unwrap(), target);

        let operation = get_result(target, &[0], false, false, value_type, location).unwrap();
        assert_operation_name(&operation, "transform.get_result");
        assert_eq!(operation.target().unwrap(), target);

        let operation = get_type(target, true, type_param_type, location).unwrap();
        assert_operation_name(&operation, "transform.get_type");
        assert_eq!(operation.value().unwrap(), target);
        assert!(operation.elemental());

        let operation =
            include("target", FailurePropagationMode::Propagate, &[target], &[handle_type], location).unwrap();
        assert_operation_name(&operation, "transform.include");
        assert_eq!(operation.operand_values().collect::<Result<Vec<_>, _>>().unwrap(), vec![target]);
        assert_eq!(operation.failure_propagation_mode().unwrap(), FailurePropagationMode::Propagate);

        let operation = match_operation_empty(target, location).unwrap();
        assert_operation_name(&operation, "transform.match.operation_empty");
        assert_eq!(operation.operand_handle().unwrap(), target);

        let operation = match_operation_name(target, &["func.func"], location).unwrap();
        assert_operation_name(&operation, "transform.match.operation_name");
        assert_eq!(operation.operand_handle().unwrap(), target);

        let operation = match_param_cmpi(target, target, MatchCmpIPredicate::Equal, location).unwrap();
        assert_operation_name(&operation, "transform.match.param.cmpi");
        assert_eq!(operation.param().unwrap(), target);
        assert_eq!(operation.reference().unwrap(), target);
        assert_eq!(operation.predicate().unwrap(), MatchCmpIPredicate::Equal);

        let operation = merge_handles(&[target], true, handle_type, location).unwrap();
        assert_operation_name(&operation, "transform.merge_handles");
        assert_eq!(operation.handles().collect::<Result<Vec<_>, _>>().unwrap(), vec![target]);
        assert!(operation.deduplicate());

        let function_type = context.function_type::<TypeRef<'_, '_>, TypeRef<'_, '_>>(&[], &[]);
        let operation = named_sequence("sequence", function_type, context.region(), location).unwrap();
        assert_operation_name(&operation, "transform.named_sequence");

        let operation = split_handle(
            target,
            &[handle_type, handle_type],
            Some(context.boolean_attribute(true)),
            Some(context.boolean_attribute(false)),
            Some(1),
            location,
        )
        .unwrap();
        assert_operation_name(&operation, "transform.split_handle");
        assert_eq!(operation.handle().unwrap(), target);

        let value = context.integer_attribute(context.signless_integer_type(32), 42);
        let operation = param_constant(value, param_type, location).unwrap();
        assert_operation_name(&operation, "transform.param.constant");

        let operation = print(Some(target), Some("payload"), true, true, true, location).unwrap();
        assert_operation_name(&operation, "transform.print");
        assert_eq!(operation.target().unwrap(), Some(target));

        let operation = replicate(target, &[target], &[handle_type], location).unwrap();
        assert_operation_name(&operation, "transform.replicate");
        assert_eq!(operation.pattern().unwrap(), target);
        assert_eq!(operation.handles().collect::<Result<Vec<_>, _>>().unwrap(), vec![target]);

        let operation = select(target, "func.func", handle_type, location).unwrap();
        assert_operation_name(&operation, "transform.select");
        assert_eq!(operation.target().unwrap(), target);
        assert_eq!(operation.op_name().unwrap().as_str(), Ok("func.func"));

        let operation = sequence(
            FailurePropagationMode::Propagate,
            Some(target),
            &[target],
            &[handle_type],
            context.region(),
            location,
        )
        .unwrap();
        assert_operation_name(&operation, "transform.sequence");
        assert_eq!(operation.root().unwrap(), Some(target));
        assert_eq!(operation.failure_propagation_mode().unwrap(), FailurePropagationMode::Propagate);

        let operation = verify(target, location).unwrap();
        assert_operation_name(&operation, "transform.verify");
        assert_eq!(operation.target().unwrap(), target);

        let operation = r#yield(&[target], location).unwrap();
        assert_operation_name(&operation, "transform.yield");
        assert_eq!(operation.yielded_values().collect::<Result<Vec<_>, _>>().unwrap(), vec![target]);
    }
}
