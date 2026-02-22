use ryft_xla_sys::bindings::{
    MlirOperationState, mlirOperationCreate, mlirOperationStateAddAttributes, mlirOperationStateAddOperands,
    mlirOperationStateAddOwnedRegions, mlirOperationStateAddResults, mlirOperationStateAddSuccessors,
    mlirOperationStateEnableResultTypeInference, mlirOperationStateGet, mlirRegionDestroy,
};

use crate::{
    Attribute, Block, Context, DetachedOperation, DetachedRegion, Location, Operation, Region, StringRef, Type, Value,
};

/// [`OperationBuilder`]s are used to build [`Operation`]s.
pub struct OperationBuilder<'c, 't: 'c> {
    /// Handle that represents this [`OperationBuilder`] in the MLIR C API.
    handle: MlirOperationState,

    /// [`Context`] associated with this [`OperationBuilder`].
    context: &'c Context<'t>,
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
        }
    }

    /// Returns a reference to the [`Context`] associated with this [`OperationBuilder`].
    pub fn context(&self) -> &'c Context<'t> {
        &self.context
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
        let named_attribute = self.context.named_attribute(self.context.identifier(name.into()), attribute);
        unsafe { mlirOperationStateAddAttributes(&mut self.handle, 1, &named_attribute.to_c_api()) };
        self
    }

    /// Adds the provided [`Value`] as an operand (i.e., input) to the [`Operation`] that is being built.
    pub fn add_operand<'v, V: Value<'v, 'c, 't>>(mut self, operand: V) -> Self
    where
        'c: 'v,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow_mut();
        unsafe { mlirOperationStateAddOperands(&mut self.handle, 1, &operand.to_c_api()) };
        self
    }

    /// Adds the provided [`Value`]s as operands (i.e., inputs) to the [`Operation`] that is being built.
    pub fn add_operands<'v, V: Value<'v, 'c, 't>>(mut self, operands: &[V]) -> Self
    where
        'c: 'v,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow_mut();
        unsafe {
            let operands = operands.iter().map(|operand| operand.to_c_api()).collect::<Vec<_>>();
            mlirOperationStateAddOperands(
                &mut self.handle,
                operands.len().cast_signed(),
                operands.as_ptr() as *const _,
            );
        }
        self
    }

    /// Adds a result of the provided [`Type`] to the [`Operation`] that is being built.
    pub fn add_result<T: Type<'c, 't>>(mut self, result_type: T) -> Self {
        unsafe { mlirOperationStateAddResults(&mut self.handle, 1, &result_type.to_c_api()) };
        self
    }

    /// Adds results of the provided [`Type`]s to the [`Operation`] that is being built.
    pub fn add_results<T: Type<'c, 't>>(mut self, result_types: &[T]) -> Self {
        unsafe {
            let result_types = result_types.iter().map(|r#type| r#type.to_c_api()).collect::<Vec<_>>();
            mlirOperationStateAddResults(
                &mut self.handle,
                result_types.len().cast_signed(),
                result_types.as_ptr() as *const _,
            );
        }
        self
    }

    /// Adds the provided [`Region`] to the [`Operation`] that is being built (and takes ownership of it).
    pub fn add_region(mut self, region: DetachedRegion<'c, 't>) -> Self {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow_mut();
        unsafe { mlirOperationStateAddOwnedRegions(&mut self.handle, 1, &region.to_c_api()) };
        std::mem::forget(region);
        self
    }

    /// Adds the provided [`Region`]s to the [`Operation`] that is being built (and takes ownership of them).
    pub fn add_regions(mut self, regions: Vec<DetachedRegion<'c, 't>>) -> Self {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow_mut();
        unsafe {
            let regions = regions.iter().map(|region| region.to_c_api()).collect::<Vec<_>>();
            mlirOperationStateAddOwnedRegions(
                &mut self.handle,
                regions.len().cast_signed(),
                regions.as_ptr() as *const _,
            );
        }
        std::mem::forget(regions);
        self
    }

    /// Adds the provided [`Block`] as a successor to the [`Operation`] that is being built.
    /// Refer to [`Block::successors`] for information on how successors are defined.
    pub fn add_successor<'r, 'b: 'r, B: Block<'b, 'c, 't>>(mut self, block: &'r B) -> OperationBuilder<'c, 't>
    where
        'c: 'b,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow_mut();
        unsafe { mlirOperationStateAddSuccessors(&mut self.handle, 1, &block.to_c_api()) };
        self
    }

    /// Adds the provided [`Block`]s as successors to the [`Operation`] that is being built.
    /// Refer to [`Block::successors`] for information on how successors are defined.
    pub fn add_successors<'r, 'b: 'r, B: Block<'b, 'c, 't>>(mut self, blocks: &[&'r B]) -> OperationBuilder<'c, 't>
    where
        'c: 'b,
    {
        // The following context borrow ensures that access to the underlying MLIR data structures is done safely from
        // Rust. It is maybe more conservative than would be ideal, but that is due to the limited exposure to MLIR
        // internals that we have when working with the MLIR C API.
        let _guard = self.context.borrow_mut();
        unsafe {
            let blocks = blocks.iter().map(|block| block.to_c_api()).collect::<Vec<_>>();
            mlirOperationStateAddSuccessors(&mut self.handle, blocks.len().cast_signed(), blocks.as_ptr() as *const _);
        }
        self
    }

    /// Enables result type inference for the [`Operation`] that is being built. If enabled, then the caller does
    /// not need to call [`OperationBuilder::add_result`] or [`OperationBuilder::add_results`] to declare the result
    /// [`Type`]s of the [`Operation`] that is being built. Instead, those types (and their number) will be inferred
    /// automatically from the operation's operands and [`Attribute`]s. If enabled, [`OperationBuilder::build`] will
    /// fail if type inference fails and return [`None`] while also emitting some diagnostics.
    pub fn enable_result_type_inference(mut self) -> Self {
        unsafe { mlirOperationStateEnableResultTypeInference(&mut self.handle) };
        self
    }

    /// Builds and returns an [`Operation`], consuming this [`OperationBuilder`] in the process. Note that, if type
    /// inference is enabled (via [`OperationBuilder::enable_result_type_inference`]) and fails, this function will
    /// return [`None`] and emit some diagnostics.
    pub fn build(mut self) -> Option<DetachedOperation<'c, 't>> {
        let operation = unsafe { DetachedOperation::from_c_api(mlirOperationCreate(&mut self.handle), self.context) };
        std::mem::forget(self);
        operation
    }
}

impl Drop for OperationBuilder<'_, '_> {
    fn drop(&mut self) {
        unsafe {
            // [`OperationBuilder`]s only own the [`Region`]s that they contain. So, we only drop any regions that
            // the current [`OperationBuilder`] owns and which are not `null` pointers (just in case something has
            // gone wrong; this should never really happen in practice).
            if self.handle.nRegions > 0 {
                for region in std::slice::from_raw_parts(self.handle.regions, self.handle.nRegions.cast_unsigned()) {
                    if !region.ptr.is_null() {
                        mlirRegionDestroy(*region);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Context;

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

        let op = builder.build();
        assert!(op.is_some());
        let op = op.unwrap();
        assert_eq!(op.name(), context.identifier("test.op"));
        assert_eq!(op.operand_count(), 3);
        assert_eq!(op.operand(0).unwrap(), arg_0);
        assert_eq!(op.operand(1).unwrap(), arg_1);
        assert_eq!(op.operand(2).unwrap(), arg_0);
        assert_eq!(op.result_count(), 3);
        assert_eq!(op.result_type(0).unwrap(), i32_type);
        assert_eq!(op.result_type(1).unwrap(), i64_type);
        assert_eq!(op.result_type(2).unwrap(), u64_type);
        assert_eq!(op.region_count(), 3);
        assert_eq!(op.successor_count(), 3);

        let attribute = op.attribute("attr_name");
        assert!(attribute.is_some());
        assert_eq!(attribute.unwrap().to_string(), "\"attr_value\"");
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
}
