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
/// All components must belong to the builder's [`Context`]. Each addition returns an [`Error`] immediately if a
/// component belongs to another context. Batch additions validate every component before accepting any of them.
/// Additions consume the builder, so a failed addition destroys it and its previously accepted regions.
///
/// Pending components are stored in Rust-owned collections until construction. The builder owns added
/// [`DetachedRegion`]s and destroys them if it is dropped or an addition fails. Region-taking functions also consume
/// and destroy rejected regions.
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
        }
    }

    /// Returns a reference to the [`Context`] associated with this [`OperationBuilder`].
    pub fn context(&self) -> &'c Context<'t> {
        self.context
    }

    /// Adds the provided [`Attribute`] to the [`Operation`] that is being built under the provided name.
    /// Returns an [`Error`] and destroys the builder if a component belongs to another [`Context`].
    pub fn add_attribute<'b, 's: 'b, N: Into<StringRef<'s>>, A: Attribute<'c, 't>>(
        mut self,
        name: N,
        attribute: A,
    ) -> Result<Self, Error>
    where
        Self: 'b,
    {
        self.validate_context(attribute.context(), "attribute")?;
        let named_attribute = self.context.named_attribute(self.context.identifier(name.into()), attribute);
        self.attributes.push(unsafe { named_attribute.to_c_api() });
        Ok(self)
    }

    /// Adds the provided [`Value`] as an operand (i.e., input) to the [`Operation`] that is being built.
    /// Returns an [`Error`] and destroys the builder if a component belongs to another [`Context`].
    pub fn add_operand<'v, V: Value<'v, 'c, 't>>(mut self, operand: V) -> Result<Self, Error>
    where
        'c: 'v,
    {
        self.validate_context(operand.context(), "operand")?;
        self.operands.push(unsafe { operand.to_c_api() });
        Ok(self)
    }

    /// Adds the provided [`Value`]s as operands (i.e., inputs) to the [`Operation`] that is being built.
    /// Returns an [`Error`] and destroys the builder if a component belongs to another [`Context`].
    pub fn add_operands<'v, V: Value<'v, 'c, 't>>(mut self, operands: &[V]) -> Result<Self, Error>
    where
        'c: 'v,
    {
        for operand in operands {
            self.validate_context(operand.context(), "operand")?;
        }
        self.operands.extend(operands.iter().map(|operand| unsafe { operand.to_c_api() }));
        Ok(self)
    }

    /// Adds a result of the provided [`Type`] to the [`Operation`] that is being built.
    /// Returns an [`Error`] and destroys the builder if a component belongs to another [`Context`].
    pub fn add_result<T: Type<'c, 't>>(mut self, result_type: T) -> Result<Self, Error> {
        self.validate_context(result_type.context(), "result type")?;
        self.result_types.push(unsafe { result_type.to_c_api() });
        Ok(self)
    }

    /// Adds results of the provided [`Type`]s to the [`Operation`] that is being built.
    /// Returns an [`Error`] and destroys the builder if a component belongs to another [`Context`].
    pub fn add_results<T: Type<'c, 't>>(mut self, result_types: &[T]) -> Result<Self, Error> {
        for result_type in result_types {
            self.validate_context(result_type.context(), "result type")?;
        }
        self.result_types.extend(result_types.iter().map(|result_type| unsafe { result_type.to_c_api() }));
        Ok(self)
    }

    /// Adds the provided [`Region`] to the [`Operation`] that is being built (and takes ownership of it).
    /// Returns an [`Error`] and destroys the builder and region if the region belongs to another context.
    pub fn add_region(mut self, region: DetachedRegion<'c, 't>) -> Result<Self, Error> {
        self.validate_context(region.context(), "region")?;
        self.regions.push(region);
        Ok(self)
    }

    /// Adds the provided [`Region`]s to the [`Operation`] that is being built (and takes ownership of them).
    /// If any region belongs to another context, this function returns an [`Error`] and destroys the builder
    /// and the entire batch.
    pub fn add_regions(mut self, regions: Vec<DetachedRegion<'c, 't>>) -> Result<Self, Error> {
        for region in &regions {
            self.validate_context(region.context(), "region")?;
        }
        self.regions.extend(regions);
        Ok(self)
    }

    /// Adds the provided [`Block`] as a successor to the [`Operation`] that is being built.
    /// Refer to [`Block::successors`] for information on how successors are defined.
    /// Returns an [`Error`] and destroys the builder if a component belongs to another [`Context`].
    pub fn add_successor<'r, 'b: 'r, B: Block<'b, 'c, 't>>(mut self, block: &'r B) -> Result<Self, Error>
    where
        'c: 'b,
    {
        self.validate_context(block.context(), "successor")?;
        self.successors.push(unsafe { block.to_c_api() });
        Ok(self)
    }

    /// Adds the provided [`Block`]s as successors to the [`Operation`] that is being built.
    /// Refer to [`Block::successors`] for information on how successors are defined.
    /// Returns an [`Error`] and destroys the builder if a component belongs to another [`Context`].
    pub fn add_successors<'r, 'b: 'r, B: Block<'b, 'c, 't>>(mut self, blocks: &[&'r B]) -> Result<Self, Error>
    where
        'c: 'b,
    {
        for block in blocks {
            self.validate_context(block.context(), "successor")?;
        }
        self.successors.extend(blocks.iter().map(|block| unsafe { block.to_c_api() }));
        Ok(self)
    }

    /// Enables result type inference for the [`Operation`] that is being built. If enabled, then the caller does
    /// not need to call [`OperationBuilder::add_result`] or [`OperationBuilder::add_results`] to declare the result
    /// [`Type`]s of the [`Operation`] that is being built. Instead, those types (and their number) will be inferred
    /// automatically from the operation's operands and [`Attribute`]s. If enabled, [`OperationBuilder::build`] will
    /// return an [`Error`] if type inference fails, while also emitting diagnostics.
    pub fn enable_result_type_inference(mut self) -> Self {
        unsafe { mlirOperationStateEnableResultTypeInference(&mut self.handle) };
        self
    }

    /// Builds and returns an [`Operation`], consuming this [`OperationBuilder`] and transferring its regions to MLIR.
    /// If enabled result type inference fails, MLIR destroys those regions and this function returns an [`Error`]
    /// after MLIR emits diagnostics.
    pub fn build(mut self) -> Result<DetachedOperation<'c, 't>, Error> {
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

    /// Checks that `context` matches this [`OperationBuilder`]'s context.
    fn validate_context(&self, context: &Context<'t>, component: &str) -> Result<(), Error> {
        if self.context.eq(context) {
            Ok(())
        } else {
            Err(Error::invalid_argument(format!("{component} context does not match operation builder context")))
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
            .unwrap()
            .add_operands(&[arg_1, arg_0])
            .unwrap()
            .add_attribute("attr_name", context.string_attribute("attr_value"))
            .unwrap()
            .add_result(i32_type)
            .unwrap()
            .add_results(&[i64_type, u64_type])
            .unwrap()
            .add_region(region_0)
            .unwrap()
            .add_regions(vec![region_1, region_2])
            .unwrap()
            .add_successor(&block_1)
            .unwrap()
            .add_successors(&[&block_2, &block_3])
            .unwrap();
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
    fn test_operation_builder_add_attribute() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let attribute = context.string_attribute("value");
        let operation = OperationBuilder::new("test.op", location)
            .add_attribute("name", attribute)
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(operation.attribute("name").unwrap().unwrap(), attribute);
    }

    #[test]
    fn test_operation_builder_add_attribute_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let result = OperationBuilder::new("test.op", context.unknown_location())
            .add_attribute("name", other_context.string_attribute("value"));
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "attribute context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_operand() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let block = context.block(&[(context.index_type(), location)]);
        let operand = block.argument(0).unwrap();
        let operation = OperationBuilder::new("test.op", location).add_operand(operand).unwrap().build().unwrap();
        assert_eq!(operation.operand_count(), 1);
        assert_eq!(operation.operand_value(0).unwrap(), operand);
    }

    #[test]
    fn test_operation_builder_add_operand_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let other_block = other_context.block(&[(other_context.index_type(), other_context.unknown_location())]);
        let result =
            OperationBuilder::new("test.op", context.unknown_location()).add_operand(other_block.argument(0).unwrap());
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "operand context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_operands() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let block = context.block(&[(context.index_type(), location)]);
        let operand = block.argument(0).unwrap();
        let operation = OperationBuilder::new("test.op", location)
            .add_operands(&[operand, operand])
            .unwrap()
            .add_operands::<crate::ValueRef>(&[])
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(operation.operand_count(), 2);
        assert_eq!(operation.operand_value(0).unwrap(), operand);
        assert_eq!(operation.operand_value(1).unwrap(), operand);
    }

    #[test]
    fn test_operation_builder_add_operands_rejects_mixed_contexts_atomically() {
        let context = Context::new();
        let other_context = Context::new();
        let block = context.block(&[(context.index_type(), context.unknown_location())]);
        let other_block = other_context.block(&[(other_context.index_type(), other_context.unknown_location())]);
        let result = OperationBuilder::new("test.op", context.unknown_location())
            .add_operands(&[block.argument(0).unwrap(), other_block.argument(0).unwrap()]);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "operand context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_result() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let result_type = context.index_type();
        let operation = OperationBuilder::new("test.op", location).add_result(result_type).unwrap().build().unwrap();
        assert_eq!(operation.result_count(), 1);
        assert_eq!(operation.result_type(0).unwrap(), result_type);
    }

    #[test]
    fn test_operation_builder_add_result_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let result =
            OperationBuilder::new("test.op", context.unknown_location()).add_result(other_context.index_type());
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "result type context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_result_destroys_accepted_regions_on_error() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let source_block = context.block(&[(context.index_type(), location)]);
        let source = source_block.argument(0).unwrap();
        let mut body = context.block_with_no_arguments();
        body.append_operation(
            OperationBuilder::new("test.use", location).add_operand(source).unwrap().build().unwrap(),
        )
        .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        let other_context = Context::new();
        let result = OperationBuilder::new("test.owner", location)
            .add_region(region)
            .unwrap()
            .add_result(other_context.index_type());
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "result type context does not match operation builder context",
        ));
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 0);
    }

    #[test]
    fn test_operation_builder_add_results() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let result_type = context.index_type();
        let operation = OperationBuilder::new("test.op", location)
            .add_results(&[result_type, result_type])
            .unwrap()
            .add_results::<crate::TypeRef>(&[])
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(operation.result_count(), 2);
        assert_eq!(operation.result_type(0).unwrap(), result_type);
        assert_eq!(operation.result_type(1).unwrap(), result_type);
    }

    #[test]
    fn test_operation_builder_add_results_rejects_mixed_contexts_atomically() {
        let context = Context::new();
        let other_context = Context::new();
        let result = OperationBuilder::new("test.op", context.unknown_location())
            .add_results(&[context.index_type(), other_context.index_type()]);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "result type context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_region() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        let operation =
            OperationBuilder::new("test.op", location).add_region(context.region()).unwrap().build().unwrap();
        assert_eq!(operation.region_count(), 1);
    }

    #[test]
    fn test_operation_builder_add_region_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let result = OperationBuilder::new("test.op", context.unknown_location()).add_region(other_context.region());
        assert!(matches!(
            result,
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
        body.append_operation(
            OperationBuilder::new("test.use", location).add_operand(source).unwrap().build().unwrap(),
        )
        .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        let other_context = Context::new();
        let result = OperationBuilder::new("test.owner", other_context.unknown_location()).add_region(region);
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 0);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "region context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_regions() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        let operation = OperationBuilder::new("test.op", location)
            .add_regions(vec![context.region(), context.region()])
            .unwrap()
            .add_regions(vec![])
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(operation.region_count(), 2);
    }

    #[test]
    fn test_operation_builder_add_regions_rejects_mixed_contexts_atomically() {
        let context = Context::new();
        let other_context = Context::new();
        let result = OperationBuilder::new("test.op", context.unknown_location())
            .add_regions(vec![context.region(), other_context.region()]);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "region context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_regions_destroys_accepted_and_rejected_regions() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let source_block = context.block(&[(context.index_type(), location)]);
        let source = source_block.argument(0).unwrap();
        let mut accepted_body = context.block_with_no_arguments();
        accepted_body
            .append_operation(OperationBuilder::new("test.use", location).add_operand(source).unwrap().build().unwrap())
            .unwrap();
        let mut rejected_body = context.block_with_no_arguments();
        rejected_body
            .append_operation(OperationBuilder::new("test.use", location).add_operand(source).unwrap().build().unwrap())
            .unwrap();
        let accepted_region: DetachedRegion<'_, '_> = accepted_body.try_into().unwrap();
        let rejected_region: DetachedRegion<'_, '_> = rejected_body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 2);
        let other_context = Context::new();
        let result = OperationBuilder::new("test.owner", location)
            .add_region(accepted_region)
            .unwrap()
            .add_regions(vec![rejected_region, other_context.region()]);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "region context does not match operation builder context",
        ));
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 0);
    }

    #[test]
    fn test_operation_builder_add_successor() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let block = context.block_with_no_arguments();
        let operation = OperationBuilder::new("test.op", location).add_successor(&block).unwrap().build().unwrap();
        assert_eq!(operation.successor_count(), 1);
        assert_eq!(operation.successor(0).unwrap(), block);
    }

    #[test]
    fn test_operation_builder_add_successor_rejects_different_context() {
        let context = Context::new();
        let other_context = Context::new();
        let other_block = other_context.block_with_no_arguments();
        let result = OperationBuilder::new("test.op", context.unknown_location()).add_successor(&other_block);
        assert!(matches!(
            result,
            Err(Error::InvalidArgument { message, .. })
                if message == "successor context does not match operation builder context",
        ));
    }

    #[test]
    fn test_operation_builder_add_successors() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let first_block = context.block_with_no_arguments();
        let second_block = context.block_with_no_arguments();
        let operation = OperationBuilder::new("test.op", location)
            .add_successors(&[&first_block, &second_block])
            .unwrap()
            .add_successors::<crate::DetachedBlock>(&[])
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(operation.successor_count(), 2);
        assert_eq!(operation.successor(0).unwrap(), first_block);
        assert_eq!(operation.successor(1).unwrap(), second_block);
    }

    #[test]
    fn test_operation_builder_add_successors_rejects_mixed_contexts_atomically() {
        let context = Context::new();
        let other_context = Context::new();
        let block = context.block_with_no_arguments();
        let other_block = other_context.block_with_no_arguments();
        let result =
            OperationBuilder::new("test.op", context.unknown_location()).add_successors(&[&block, &other_block]);
        assert!(matches!(
            result,
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
        body.append_operation(
            OperationBuilder::new("test.use", location).add_operand(source).unwrap().build().unwrap(),
        )
        .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        let operation = OperationBuilder::new("test.owner", location).add_region(region).unwrap().build().unwrap();
        assert_eq!(operation.region_count(), 1);
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        drop(operation);
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
        body.append_operation(
            OperationBuilder::new("test.use", location).add_operand(source).unwrap().build().unwrap(),
        )
        .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        // An unregistered operation cannot infer result types. MLIR must destroy the transferred region on failure.
        let builder = OperationBuilder::new("test.owner", location)
            .add_region(region)
            .unwrap()
            .enable_result_type_inference();
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
            .unwrap()
            .add_operands(&[arg_1, arg_0])
            .unwrap()
            .add_attribute("attr_name", context.string_attribute("attr_value"))
            .unwrap()
            .add_result(i32_type)
            .unwrap()
            .add_results(&[i64_type, u64_type])
            .unwrap()
            .add_region(region_0)
            .unwrap()
            .add_regions(vec![region_1, region_2])
            .unwrap()
            .add_successor(&block_1)
            .unwrap()
            .add_successors(&[&block_2, &block_3])
            .unwrap();
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
        body.append_operation(
            OperationBuilder::new("test.use", location).add_operand(source).unwrap().build().unwrap(),
        )
        .unwrap();
        let region: DetachedRegion<'_, '_> = body.try_into().unwrap();
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 1);
        let builder = OperationBuilder::new("test.owner", location).add_region(region).unwrap();
        drop(builder);
        assert_eq!(source.uses().unwrap().collect::<Result<Vec<_>, _>>().unwrap().len(), 0);
    }
}
