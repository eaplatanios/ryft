use ryft_xla_sys::bindings::{
    MlirBlock, MlirNamedAttribute, MlirOperationState, MlirType, MlirValue, mlirOperationCreate,
    mlirOperationStateAddAttributes, mlirOperationStateAddOperands, mlirOperationStateAddOwnedRegions,
    mlirOperationStateAddResults, mlirOperationStateAddSuccessors, mlirOperationStateEnableResultTypeInference,
    mlirOperationStateGet,
};

use crate::operations::operation::{DetachedOperation, Operation};
use crate::{Attribute, Block, Context, DetachedRegion, Error, Location, Region, StringRef, Type, Value};

/// [`OperationBuilder`]s are used to build [`Operation`]s.
///
/// All components must belong to the builder's [`Context`]. Configuration retains the first context mismatch,
/// and subsequent additions are ignored. [`OperationBuilder::build`] returns the retained error, if there is one.
/// Batch additions validate every component before accepting any of them.
///
/// Pending components are stored in Rust-owned collections until construction. The builder owns added
/// [`DetachedRegion`]s and destroys them if it is dropped or returns a configuration error. Region-taking
/// functions also consume and destroy rejected regions, including regions supplied after an earlier error.
pub struct OperationBuilder<'c, 't: 'c> {
    /// Handle that represents this [`OperationBuilder`] in the MLIR C API. The underlying native builder's component
    /// arrays stay empty until [`OperationBuilder::build`], where MLIR allocates and consumes them within that call.
    handle: MlirOperationState,

    /// [`Context`] associated with this [`OperationBuilder`].
    context: &'c Context<'t>,

    /// Non-owning native [`MlirNamedAttribute`] handles to attach to the operation being built.
    attributes: Vec<MlirNamedAttribute>,

    /// Non-owning native [`MlirValue`] handles to attach to the operation being built as operands.
    operands: Vec<MlirValue>,

    /// Non-owning native result [`MlirType`] handles to attach to the operation being built.
    result_types: Vec<MlirType>,

    /// [`DetachedRegion`] owned by this builder until construction transfers them to MLIR.
    regions: Vec<DetachedRegion<'c, 't>>,

    /// Non-owning native successor [`MlirBlock`] handles to attach to the operation being built.
    successors: Vec<MlirBlock>,

    /// First error encountered while configuring this [`OperationBuilder`].
    error: Option<Error>,
}

impl<'c, 't: 'c> OperationBuilder<'c, 't> {
    /// Creates a new [`OperationBuilder`] using the provided [`Operation`] name and [`Location`].
    pub fn new<'b, 's: 'b, S: Into<StringRef<'s>>, L: Location<'c, 't>>(name: S, location: L) -> Self
    where
        Self: 'b,
    {
        OperationBuilder {
            handle: unsafe { mlirOperationStateGet(name.into().to_c_api(), location.to_c_api()) },
            context: location.context(),
            attributes: Vec::new(),
            operands: Vec::new(),
            result_types: Vec::new(),
            regions: Vec::new(),
            successors: Vec::new(),
            error: None,
        }
    }

    /// Returns a reference to the [`Context`] associated with this [`OperationBuilder`].
    pub fn context(&self) -> &'c Context<'t> {
        self.context
    }

    /// Adds the provided [`Attribute`] to the [`Operation`] that is being built under the provided name.
    pub fn add_attribute<'b, 's: 'b, N: Into<StringRef<'s>>, A: Attribute<'c, 't>>(
        mut self,
        name: N,
        attribute: A,
    ) -> Self
    where
        Self: 'b,
    {
        if !self.validate_context(attribute.context(), "attribute") {
            return self;
        }
        let named_attribute = self.context.named_attribute(self.context.identifier(name.into()), attribute);
        self.attributes.push(unsafe { named_attribute.to_c_api() });
        self
    }

    /// Adds the provided [`Value`] as an operand (i.e., input) to the [`Operation`] that is being built.
    pub fn add_operand<'v, V: Value<'v, 'c, 't>>(mut self, operand: V) -> Self
    where
        'c: 'v,
    {
        if !self.validate_context(operand.context(), "operand") {
            return self;
        }
        self.operands.push(unsafe { operand.to_c_api() });
        self
    }

    /// Adds the provided [`Value`]s as operands (i.e., inputs) to the [`Operation`] that is being built.
    pub fn add_operands<'v, V: Value<'v, 'c, 't>>(mut self, operands: &[V]) -> Self
    where
        'c: 'v,
    {
        if self.error.is_some() || !operands.iter().all(|operand| self.validate_context(operand.context(), "operand")) {
            return self;
        }
        self.operands.extend(operands.iter().map(|operand| unsafe { operand.to_c_api() }));
        self
    }

    /// Adds a result of the provided [`Type`] to the [`Operation`] that is being built.
    pub fn add_result<T: Type<'c, 't>>(mut self, result_type: T) -> Self {
        if !self.validate_context(result_type.context(), "result type") {
            return self;
        }
        self.result_types.push(unsafe { result_type.to_c_api() });
        self
    }

    /// Adds results of the provided [`Type`]s to the [`Operation`] that is being built.
    pub fn add_results<T: Type<'c, 't>>(mut self, result_types: &[T]) -> Self {
        if self.error.is_some()
            || !result_types.iter().all(|result_type| self.validate_context(result_type.context(), "result type"))
        {
            return self;
        }
        self.result_types.extend(result_types.iter().map(|result_type| unsafe { result_type.to_c_api() }));
        self
    }

    /// Adds the provided [`Region`] to the [`Operation`] that is being built (and takes ownership of it).
    /// A region from another context, or one supplied after an earlier configuration error, is destroyed immediately.
    pub fn add_region(mut self, region: DetachedRegion<'c, 't>) -> Self {
        if !self.validate_context(region.context(), "region") {
            return self;
        }
        self.regions.push(region);
        self
    }

    /// Adds the provided [`Region`]s to the [`Operation`] that is being built (and takes ownership of them).
    /// If any region belongs to another context, the entire batch is rejected and destroyed. Regions supplied after
    /// an earlier configuration error are also destroyed immediately.
    pub fn add_regions(mut self, regions: Vec<DetachedRegion<'c, 't>>) -> Self {
        if self.error.is_some() || !regions.iter().all(|region| self.validate_context(region.context(), "region")) {
            return self;
        }
        self.regions.extend(regions);
        self
    }

    /// Adds the provided [`Block`] as a successor to the [`Operation`] that is being built.
    /// Refer to [`Block::successors`] for information on how successors are defined.
    pub fn add_successor<'r, 'b: 'r, B: Block<'b, 'c, 't>>(mut self, block: &'r B) -> OperationBuilder<'c, 't>
    where
        'c: 'b,
    {
        if !self.validate_context(block.context(), "successor") {
            return self;
        }
        self.successors.push(unsafe { block.to_c_api() });
        self
    }

    /// Adds the provided [`Block`]s as successors to the [`Operation`] that is being built.
    /// Refer to [`Block::successors`] for information on how successors are defined.
    pub fn add_successors<'r, 'b: 'r, B: Block<'b, 'c, 't>>(mut self, blocks: &[&'r B]) -> OperationBuilder<'c, 't>
    where
        'c: 'b,
    {
        if self.error.is_some() || !blocks.iter().all(|block| self.validate_context(block.context(), "successor")) {
            return self;
        }
        self.successors.extend(blocks.iter().map(|block| unsafe { block.to_c_api() }));
        self
    }

    /// Enables result type inference for the [`Operation`] that is being built. If enabled, then the caller does
    /// not need to call [`OperationBuilder::add_result`] or [`OperationBuilder::add_results`] to declare the result
    /// [`Type`]s of the [`Operation`] that is being built. Instead, those types (and their number) will be inferred
    /// automatically from the operation's operands and [`Attribute`]s. If enabled, [`OperationBuilder::build`] will
    /// return an [`Error`] if type inference fails, while also emitting diagnostics.
    pub fn enable_result_type_inference(mut self) -> Self {
        if self.error.is_none() {
            unsafe { mlirOperationStateEnableResultTypeInference(&mut self.handle) };
        }
        self
    }

    /// Builds and returns an [`Operation`], consuming this [`OperationBuilder`] in the process. Returns the first error
    /// retained during configuration, if any, and destroys the pending regions. Otherwise, transfers the regions to
    /// MLIR. If enabled result type inference fails, MLIR destroys those regions and this function returns an [`Error`]
    /// after MLIR emits diagnostics.
    pub fn build(mut self) -> Result<DetachedOperation<'c, 't>, Error> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        let regions = self.regions.iter().map(|region| unsafe { region.to_c_api() }).collect::<Vec<_>>();
        let handle = {
            // The following context borrow ensures that access to the underlying MLIR data structures is done safely
            // from Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure
            // to MLIR internals that we have when working with the MLIR C API.
            let _guard = self.context.borrow_mut();

            // The C API allocates its own copies of these arrays. Keep native state construction and consumption
            // together (i.e., no fallible Rust operations or early returns may intervene after the first allocation).
            unsafe {
                if !self.attributes.is_empty() {
                    mlirOperationStateAddAttributes(
                        &mut self.handle,
                        self.attributes.len().cast_signed(),
                        self.attributes.as_ptr(),
                    );
                }

                if !self.operands.is_empty() {
                    mlirOperationStateAddOperands(
                        &mut self.handle,
                        self.operands.len().cast_signed(),
                        self.operands.as_ptr(),
                    );
                }

                if !self.result_types.is_empty() {
                    mlirOperationStateAddResults(
                        &mut self.handle,
                        self.result_types.len().cast_signed(),
                        self.result_types.as_ptr(),
                    );
                }

                if !self.successors.is_empty() {
                    mlirOperationStateAddSuccessors(
                        &mut self.handle,
                        self.successors.len().cast_signed(),
                        self.successors.as_ptr(),
                    );
                }

                if !regions.is_empty() {
                    mlirOperationStateAddOwnedRegions(&mut self.handle, regions.len().cast_signed(), regions.as_ptr());
                }

                // MLIR now owns each region. Drain the Rust wrappers rather than forgetting their vector allocation.
                for region in self.regions.drain(..) {
                    std::mem::forget(region);
                }

                // This consumes every native array and region even when type inference returns a null operation.
                // The remaining Rust fields own only their vector storage, not the consumed native state.
                mlirOperationCreate(&mut self.handle)
            }
        };

        unsafe { DetachedOperation::from_c_api(handle, self.context) }
            .map_err(|_| Error::invalid_argument("failed to build operation"))
    }

    /// Returns whether `context` matches this [`OperationBuilder`]'s context, retaining the first mismatch as an error.
    fn validate_context(&mut self, context: &Context<'t>, component: &str) -> bool {
        if self.error.is_some() {
            return false;
        }
        if self.context.eq(context) {
            true
        } else {
            let message = format!("{component} context does not match operation builder context");
            self.error = Some(Error::invalid_argument(message));
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_operation_builder() {
        let context = Context::new();
        context.allow_unregistered_dialects();

        let location = context.unknown_location();
        let index_type = context.index_type();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let u64_type = context.unsigned_integer_type(64);
        let block_0 = context.block(&[(index_type, location)]);
        let block_1 = context.block_with_no_arguments();
        let block_2 = context.block_with_no_arguments();
        let block_3 = context.block_with_no_arguments();
        let arg_0 = block_0.argument(0).unwrap();
        let arg_1 = block_0.argument(0).unwrap();
        let region_0 = context.region();
        let region_1 = context.region();
        let region_2 = context.region();

        let builder = OperationBuilder::new("test.op", location)
            .add_operand(arg_0)
            .add_operands(&[arg_1, arg_0])
            .add_attribute("attr_name", context.string_attribute("attr_value"))
            .add_result(i32_type)
            .add_results(&[i64_type, u64_type])
            .add_region(region_0)
            .add_regions(vec![region_1, region_2])
            .add_successor(&block_1)
            .add_successors(&[&block_2, &block_3]);
        assert_eq!(builder.context(), &context);
        assert_eq!(builder.handle.nAttributes, 0);
        assert_eq!(builder.handle.nOperands, 0);
        assert_eq!(builder.handle.nResults, 0);
        assert_eq!(builder.handle.nRegions, 0);
        assert_eq!(builder.handle.nSuccessors, 0);
        assert!(builder.handle.attributes.is_null());
        assert!(builder.handle.operands.is_null());
        assert!(builder.handle.results.is_null());
        assert!(builder.handle.regions.is_null());
        assert!(builder.handle.successors.is_null());

        let op = builder.build();
        assert!(op.is_ok());
        let op = op.unwrap();
        assert_eq!(op.name(), context.identifier("test.op"));
        assert_eq!(op.operand_count(), 3);
        assert_eq!(op.operand_value(0).unwrap(), arg_0);
        assert_eq!(op.operand_value(1).unwrap(), arg_1);
        assert_eq!(op.operand_value(2).unwrap(), arg_0);
        assert_eq!(op.result_count(), 3);
        assert_eq!(op.result_type(0).unwrap(), i32_type);
        assert_eq!(op.result_type(1).unwrap(), i64_type);
        assert_eq!(op.result_type(2).unwrap(), u64_type);
        assert_eq!(op.region_count(), 3);
        assert_eq!(op.successor_count(), 3);

        let attribute = op.attribute("attr_name").unwrap();
        assert!(attribute.is_some());
        assert_eq!(attribute.unwrap().to_string(), "\"attr_value\"");

        let error = OperationBuilder::new("test.op", location).enable_result_type_inference().build();
        assert!(matches!(
            error,
            Err(Error::InvalidArgument { message, .. }) if message == "failed to build operation",
        ));
    }

    #[test]
    fn test_operation_builder_add_attribute_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let builder = OperationBuilder::new("test.op", context.unknown_location())
            .add_attribute("name", other_context.string_attribute("value"));
        assert_eq!(builder.attributes.len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "attribute context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_operand_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let other_block = other_context.block(&[(other_context.index_type(), other_context.unknown_location())]);
        let builder =
            OperationBuilder::new("test.op", context.unknown_location()).add_operand(other_block.argument(0).unwrap());
        assert_eq!(builder.operands.len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "operand context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_operands_rejects_mixed_contexts_atomically() {
        let context = Context::new();
        let other_context = Context::new();
        let block = context.block(&[(context.index_type(), context.unknown_location())]);
        let other_block = other_context.block(&[(other_context.index_type(), other_context.unknown_location())]);
        let builder = OperationBuilder::new("test.op", context.unknown_location())
            .add_operands(&[block.argument(0).unwrap(), other_block.argument(0).unwrap()]);
        assert_eq!(builder.operands.len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "operand context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_result_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let builder =
            OperationBuilder::new("test.op", context.unknown_location()).add_result(other_context.index_type());
        assert_eq!(builder.result_types.len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "result type context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_results_rejects_mixed_contexts_atomically() {
        let context = Context::new();
        let other_context = Context::new();
        let builder = OperationBuilder::new("test.op", context.unknown_location())
            .add_results(&[context.index_type(), other_context.index_type()]);
        assert_eq!(builder.result_types.len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "result type context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_region_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let builder = OperationBuilder::new("test.op", context.unknown_location()).add_region(other_context.region());
        assert_eq!(builder.regions.len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "region context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_region_destroys_rejected_region() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let source_block = context.block(&[(context.index_type(), location)]);
        let source = source_block.argument(0).unwrap();
        let mut body = context.block_with_no_arguments();
        body.append_operation(OperationBuilder::new("test.use", location).add_operand(source).build().unwrap())
            .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        let other_context = Context::new();
        let builder = OperationBuilder::new("test.owner", other_context.unknown_location()).add_region(region);
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "region context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_regions_rejects_mixed_contexts_atomically() {
        let context = Context::new();
        let other_context = Context::new();
        let builder = OperationBuilder::new("test.op", context.unknown_location())
            .add_regions(vec![context.region(), other_context.region()]);
        assert_eq!(builder.regions.len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "region context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_successor_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let other_block = other_context.block_with_no_arguments();
        let builder = OperationBuilder::new("test.op", context.unknown_location()).add_successor(&other_block);
        assert_eq!(builder.successors.len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "successor context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_successors_rejects_mixed_contexts_atomically() {
        let context = Context::new();
        let other_context = Context::new();
        let block = context.block_with_no_arguments();
        let other_block = other_context.block_with_no_arguments();
        let builder =
            OperationBuilder::new("test.op", context.unknown_location()).add_successors(&[&block, &other_block]);
        assert_eq!(builder.successors.len(), 0);
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "successor context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_build() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let source_block = context.block(&[(context.index_type(), location)]);
        let source = source_block.argument(0).unwrap();
        let mut body = context.block_with_no_arguments();
        body.append_operation(OperationBuilder::new("test.use", location).add_operand(source).build().unwrap())
            .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        let operation = OperationBuilder::new("test.owner", location).add_region(region).build().unwrap();
        assert_eq!(operation.region_count(), 1);
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        drop(operation);
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 0);
    }

    #[test]
    fn test_operation_builder_build_retains_first_error() {
        let context = Context::new();
        let other_context = Context::new();
        let other_block = other_context.block(&[(other_context.index_type(), other_context.unknown_location())]);
        let builder = OperationBuilder::new("test.op", context.unknown_location())
            .add_region(context.region())
            .add_operand(other_block.argument(0).unwrap())
            .add_result(other_context.index_type())
            .add_attribute("ignored", context.unit_attribute())
            .add_operand(other_block.argument(0).unwrap())
            .add_results(&[context.index_type()])
            .add_results::<crate::TypeRef>(&[])
            .add_regions(vec![context.region()])
            .add_regions(vec![])
            .enable_result_type_inference();
        assert_eq!(builder.regions.len(), 1);
        assert_eq!(builder.operands.len(), 0);
        assert_eq!(builder.result_types.len(), 0);
        assert_eq!(builder.attributes.len(), 0);
        assert!(!builder.handle.enableResultTypeInference);

        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "operand context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_build_destroys_regions_on_configuration_error() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let source_block = context.block(&[(context.index_type(), location)]);
        let source = source_block.argument(0).unwrap();
        let mut body = context.block_with_no_arguments();
        body.append_operation(OperationBuilder::new("test.use", location).add_operand(source).build().unwrap())
            .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        let other_context = Context::new();
        let builder = OperationBuilder::new("test.owner", location)
            .add_region(region)
            .add_result(other_context.index_type());
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. })
                if message == "result type context does not match operation builder context",
        ));
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 0);
    }

    #[test]
    fn test_operation_builder_build_destroys_regions_on_inference_failure() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let source_block = context.block(&[(context.index_type(), location)]);
        let source = source_block.argument(0).unwrap();
        let mut body = context.block_with_no_arguments();
        body.append_operation(OperationBuilder::new("test.use", location).add_operand(source).build().unwrap())
            .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        // An unregistered operation cannot infer result types. MLIR must destroy the transferred region on failure.
        let builder = OperationBuilder::new("test.owner", location).add_region(region).enable_result_type_inference();
        assert!(matches!(
            builder.build(),
            Err(Error::InvalidArgument { message, .. }) if message == "failed to build operation",
        ));
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 0);
    }

    #[test]
    fn test_operation_builder_drop() {
        // Checks that an unused operation builder gets dropped properly without crashing.
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let u64_type = context.unsigned_integer_type(64);
        let block_0 = context.block(&[(index_type, location)]);
        let block_1 = context.block_with_no_arguments();
        let block_2 = context.block_with_no_arguments();
        let block_3 = context.block_with_no_arguments();
        let arg_0 = block_0.argument(0).unwrap();
        let arg_1 = block_0.argument(0).unwrap();
        let region_0 = context.region();
        let region_1 = context.region();
        let region_2 = context.region();
        let builder = OperationBuilder::new("test.op", location)
            .add_operand(arg_0)
            .add_operands(&[arg_1, arg_0])
            .add_attribute("attr_name", context.string_attribute("attr_value"))
            .add_result(i32_type)
            .add_results(&[i64_type, u64_type])
            .add_region(region_0)
            .add_regions(vec![region_1, region_2])
            .add_successor(&block_1)
            .add_successors(&[&block_2, &block_3]);
        assert_eq!(builder.context(), &context);
    }

    #[test]
    fn test_operation_builder_drop_destroys_regions() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let source_block = context.block(&[(context.index_type(), location)]);
        let source = source_block.argument(0).unwrap();
        let mut body = context.block_with_no_arguments();
        body.append_operation(OperationBuilder::new("test.use", location).add_operand(source).build().unwrap())
            .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        let builder = OperationBuilder::new("test.owner", location).add_region(region);
        drop(builder);
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 0);
    }
}
