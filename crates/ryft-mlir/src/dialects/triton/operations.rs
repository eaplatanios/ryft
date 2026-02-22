// TODO(eaplatanios): Clean this up and make sure it is correct.

use crate::{AttributeRef, BlockRef, DetachedOp, DetachedRegion, Location, OperationBuilder, TypeRef, ValueRef};

use super::TritonDialect;

/// Builder options shared by Triton operation constructors.
pub struct TritonOperationBuildOptions<'o, 'c, 't> {
    operands: Vec<ValueRef<'o, 'c, 't>>,
    attributes: Vec<(String, AttributeRef<'c, 't>)>,
    result_types: Vec<TypeRef<'c, 't>>,
    regions: Vec<DetachedRegion<'c, 't>>,
    successors: Vec<BlockRef<'o, 'c, 't>>,
    infer_result_types: bool,
}

impl<'o, 'c, 't> Default for TritonOperationBuildOptions<'o, 'c, 't> {
    fn default() -> Self {
        Self {
            operands: Vec::new(),
            attributes: Vec::new(),
            result_types: Vec::new(),
            regions: Vec::new(),
            successors: Vec::new(),
            infer_result_types: false,
        }
    }
}

impl<'o, 'c, 't> TritonOperationBuildOptions<'o, 'c, 't> {
    /// Creates a new empty [`TritonOperationBuildOptions`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an operation operand.
    pub fn add_operand(mut self, operand: ValueRef<'o, 'c, 't>) -> Self {
        self.operands.push(operand);
        self
    }

    /// Adds operation operands.
    pub fn add_operands(mut self, operands: &[ValueRef<'o, 'c, 't>]) -> Self {
        self.operands.extend_from_slice(operands);
        self
    }

    /// Adds an operation attribute.
    pub fn add_attribute<S: AsRef<str>>(mut self, name: S, attribute: AttributeRef<'c, 't>) -> Self {
        self.attributes.push((name.as_ref().to_owned(), attribute));
        self
    }

    /// Adds an operation result type.
    pub fn add_result_type(mut self, result_type: TypeRef<'c, 't>) -> Self {
        self.result_types.push(result_type);
        self
    }

    /// Adds operation result types.
    pub fn add_result_types(mut self, result_types: &[TypeRef<'c, 't>]) -> Self {
        self.result_types.extend_from_slice(result_types);
        self
    }

    /// Adds an operation region.
    pub fn add_region(mut self, region: DetachedRegion<'c, 't>) -> Self {
        self.regions.push(region);
        self
    }

    /// Adds operation regions.
    pub fn add_regions(mut self, regions: Vec<DetachedRegion<'c, 't>>) -> Self {
        self.regions.extend(regions);
        self
    }

    /// Adds an operation successor block.
    pub fn add_successor(mut self, successor: BlockRef<'o, 'c, 't>) -> Self {
        self.successors.push(successor);
        self
    }

    /// Adds operation successor blocks.
    pub fn add_successors(mut self, successors: &[BlockRef<'o, 'c, 't>]) -> Self {
        self.successors.extend_from_slice(successors);
        self
    }

    /// Enables MLIR result type inference for the operation being built.
    pub fn infer_result_types(mut self) -> Self {
        self.infer_result_types = true;
        self
    }
}

pub(crate) fn build_typed_triton_operation<'o, 'c: 'o, 't: 'c, O: DetachedOp<'o, 'c, 't>, L: Location<'c, 't>>(
    dialect: TritonDialect,
    mnemonic: &str,
    mut options: TritonOperationBuildOptions<'o, 'c, 't>,
    location: L,
    rust_path: &str,
) -> O {
    let context = location.context();
    let _dialect = context.load_triton_dialect(dialect);

    let operation_name = format!("{}.{}", dialect.namespace(), mnemonic);
    let mut builder = OperationBuilder::new(operation_name.as_str(), location);

    for operand in options.operands {
        builder = builder.add_operand(operand);
    }
    for (name, attribute) in options.attributes {
        builder = builder.add_attribute(name.as_str(), attribute);
    }
    for result_type in options.result_types {
        builder = builder.add_result(result_type);
    }
    for region in options.regions.drain(..) {
        builder = builder.add_region(region);
    }
    for successor in options.successors {
        builder = builder.add_successor(&successor);
    }
    if options.infer_result_types {
        builder = builder.enable_result_type_inference();
    }

    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect(format!("invalid arguments to `triton::{}`", rust_path).as_str())
}

pub mod tt {
    use crate::{mlir_op, Location, Operation};

    use super::super::TritonDialect;
    use super::{build_typed_triton_operation, TritonOperationBuildOptions};

    /// All known `tt` operation mnemonics.
    ///
    /// Refer to the upstream Triton ODS operation definitions in
    /// [`TritonOps.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonOps.td)
    /// for more information.
    pub const OPERATIONS: &[&str] = &[
        "addptr",
        "advance",
        "assert",
        "atomic_cas",
        "atomic_rmw",
        "bitcast",
        "broadcast",
        "call",
        "cat",
        "clampf",
        "descriptor_gather",
        "descriptor_load",
        "descriptor_reduce",
        "descriptor_scatter",
        "descriptor_store",
        "dot",
        "dot_scaled",
        "elementwise_inline_asm",
        "expand_dims",
        "extern_elementwise",
        "fp_to_fp",
        "func",
        "gather",
        "get_num_programs",
        "get_program_id",
        "histogram",
        "int_to_ptr",
        "join",
        "load",
        "make_range",
        "make_tensor_descriptor",
        "make_tensor_ptr",
        "map_elementwise",
        "map_elementwise.return",
        "mulhiui",
        "precise_divf",
        "precise_sqrt",
        "print",
        "ptr_to_int",
        "reduce",
        "reduce.return",
        "reshape",
        "return",
        "scan",
        "scan.return",
        "splat",
        "split",
        "store",
        "trans",
        "unsplat",
    ];

    /// Trait representing the `tt.addptr` operation.
    pub trait AddptrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Addptr, op_name = "tt.addptr");

    /// Constructs a new detached/owned [`AddptrOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn addptr<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedAddptrOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "addptr", options, location, "tt::addptr")
    }

    /// Trait representing the `tt.advance` operation.
    pub trait AdvanceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Advance, op_name = "tt.advance");

    /// Constructs a new detached/owned [`AdvanceOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn advance<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedAdvanceOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "advance", options, location, "tt::advance")
    }

    /// Trait representing the `tt.assert` operation.
    pub trait AssertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Assert, op_name = "tt.assert");

    /// Constructs a new detached/owned [`AssertOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn assert<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedAssertOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "assert", options, location, "tt::assert")
    }

    /// Trait representing the `tt.atomic_cas` operation.
    pub trait AtomicCasOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(AtomicCas, op_name = "tt.atomic_cas");

    /// Constructs a new detached/owned [`AtomicCasOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn atomic_cas<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedAtomicCasOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "atomic_cas", options, location, "tt::atomic_cas")
    }

    /// Trait representing the `tt.atomic_rmw` operation.
    pub trait AtomicRmwOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(AtomicRmw, op_name = "tt.atomic_rmw");

    /// Constructs a new detached/owned [`AtomicRmwOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn atomic_rmw<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedAtomicRmwOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "atomic_rmw", options, location, "tt::atomic_rmw")
    }

    /// Trait representing the `tt.bitcast` operation.
    pub trait BitcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Bitcast, op_name = "tt.bitcast");

    /// Constructs a new detached/owned [`BitcastOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn bitcast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedBitcastOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "bitcast", options, location, "tt::bitcast")
    }

    /// Trait representing the `tt.broadcast` operation.
    pub trait BroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Broadcast, op_name = "tt.broadcast");

    /// Constructs a new detached/owned [`BroadcastOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn broadcast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedBroadcastOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "broadcast", options, location, "tt::broadcast")
    }

    /// Trait representing the `tt.call` operation.
    pub trait CallOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Call, op_name = "tt.call");

    /// Constructs a new detached/owned [`CallOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn call<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedCallOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "call", options, location, "tt::call")
    }

    /// Trait representing the `tt.cat` operation.
    pub trait CatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Cat, op_name = "tt.cat");

    /// Constructs a new detached/owned [`CatOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn cat<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedCatOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "cat", options, location, "tt::cat")
    }

    /// Trait representing the `tt.clampf` operation.
    pub trait ClampfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Clampf, op_name = "tt.clampf");

    /// Constructs a new detached/owned [`ClampfOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn clampf<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedClampfOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "clampf", options, location, "tt::clampf")
    }

    /// Trait representing the `tt.descriptor_gather` operation.
    pub trait DescriptorGatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(DescriptorGather, op_name = "tt.descriptor_gather");

    /// Constructs a new detached/owned [`DescriptorGatherOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn descriptor_gather<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedDescriptorGatherOperation<'c, 't> {
        build_typed_triton_operation(
            TritonDialect::Triton,
            "descriptor_gather",
            options,
            location,
            "tt::descriptor_gather",
        )
    }

    /// Trait representing the `tt.descriptor_load` operation.
    pub trait DescriptorLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(DescriptorLoad, op_name = "tt.descriptor_load");

    /// Constructs a new detached/owned [`DescriptorLoadOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn descriptor_load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedDescriptorLoadOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "descriptor_load", options, location, "tt::descriptor_load")
    }

    /// Trait representing the `tt.descriptor_reduce` operation.
    pub trait DescriptorReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(DescriptorReduce, op_name = "tt.descriptor_reduce");

    /// Constructs a new detached/owned [`DescriptorReduceOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn descriptor_reduce<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedDescriptorReduceOperation<'c, 't> {
        build_typed_triton_operation(
            TritonDialect::Triton,
            "descriptor_reduce",
            options,
            location,
            "tt::descriptor_reduce",
        )
    }

    /// Trait representing the `tt.descriptor_scatter` operation.
    pub trait DescriptorScatterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(DescriptorScatter, op_name = "tt.descriptor_scatter");

    /// Constructs a new detached/owned [`DescriptorScatterOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn descriptor_scatter<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedDescriptorScatterOperation<'c, 't> {
        build_typed_triton_operation(
            TritonDialect::Triton,
            "descriptor_scatter",
            options,
            location,
            "tt::descriptor_scatter",
        )
    }

    /// Trait representing the `tt.descriptor_store` operation.
    pub trait DescriptorStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(DescriptorStore, op_name = "tt.descriptor_store");

    /// Constructs a new detached/owned [`DescriptorStoreOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn descriptor_store<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedDescriptorStoreOperation<'c, 't> {
        build_typed_triton_operation(
            TritonDialect::Triton,
            "descriptor_store",
            options,
            location,
            "tt::descriptor_store",
        )
    }

    /// Trait representing the `tt.dot` operation.
    pub trait DotOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Dot, op_name = "tt.dot");

    /// Constructs a new detached/owned [`DotOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn dot<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedDotOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "dot", options, location, "tt::dot")
    }

    /// Trait representing the `tt.dot_scaled` operation.
    pub trait DotScaledOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(DotScaled, op_name = "tt.dot_scaled");

    /// Constructs a new detached/owned [`DotScaledOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn dot_scaled<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedDotScaledOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "dot_scaled", options, location, "tt::dot_scaled")
    }

    /// Trait representing the `tt.elementwise_inline_asm` operation.
    pub trait ElementwiseInlineAsmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(ElementwiseInlineAsm, op_name = "tt.elementwise_inline_asm");

    /// Constructs a new detached/owned [`ElementwiseInlineAsmOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn elementwise_inline_asm<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedElementwiseInlineAsmOperation<'c, 't> {
        build_typed_triton_operation(
            TritonDialect::Triton,
            "elementwise_inline_asm",
            options,
            location,
            "tt::elementwise_inline_asm",
        )
    }

    /// Trait representing the `tt.expand_dims` operation.
    pub trait ExpandDimsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(ExpandDims, op_name = "tt.expand_dims");

    /// Constructs a new detached/owned [`ExpandDimsOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn expand_dims<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedExpandDimsOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "expand_dims", options, location, "tt::expand_dims")
    }

    /// Trait representing the `tt.extern_elementwise` operation.
    pub trait ExternElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(ExternElementwise, op_name = "tt.extern_elementwise");

    /// Constructs a new detached/owned [`ExternElementwiseOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn extern_elementwise<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedExternElementwiseOperation<'c, 't> {
        build_typed_triton_operation(
            TritonDialect::Triton,
            "extern_elementwise",
            options,
            location,
            "tt::extern_elementwise",
        )
    }

    /// Trait representing the `tt.fp_to_fp` operation.
    pub trait FpToFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(FpToFp, op_name = "tt.fp_to_fp");

    /// Constructs a new detached/owned [`FpToFpOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn fp_to_fp<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedFpToFpOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "fp_to_fp", options, location, "tt::fp_to_fp")
    }

    /// Trait representing the `tt.func` operation.
    pub trait FuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Func, op_name = "tt.func");

    /// Constructs a new detached/owned [`FuncOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn func<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedFuncOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "func", options, location, "tt::func")
    }

    /// Trait representing the `tt.gather` operation.
    pub trait GatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Gather, op_name = "tt.gather");

    /// Constructs a new detached/owned [`GatherOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn gather<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedGatherOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "gather", options, location, "tt::gather")
    }

    /// Trait representing the `tt.get_num_programs` operation.
    pub trait GetNumProgramsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(GetNumPrograms, op_name = "tt.get_num_programs");

    /// Constructs a new detached/owned [`GetNumProgramsOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn get_num_programs<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedGetNumProgramsOperation<'c, 't> {
        build_typed_triton_operation(
            TritonDialect::Triton,
            "get_num_programs",
            options,
            location,
            "tt::get_num_programs",
        )
    }

    /// Trait representing the `tt.get_program_id` operation.
    pub trait GetProgramIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(GetProgramId, op_name = "tt.get_program_id");

    /// Constructs a new detached/owned [`GetProgramIdOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn get_program_id<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedGetProgramIdOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "get_program_id", options, location, "tt::get_program_id")
    }

    /// Trait representing the `tt.histogram` operation.
    pub trait HistogramOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Histogram, op_name = "tt.histogram");

    /// Constructs a new detached/owned [`HistogramOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn histogram<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedHistogramOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "histogram", options, location, "tt::histogram")
    }

    /// Trait representing the `tt.int_to_ptr` operation.
    pub trait IntToPtrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(IntToPtr, op_name = "tt.int_to_ptr");

    /// Constructs a new detached/owned [`IntToPtrOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn int_to_ptr<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedIntToPtrOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "int_to_ptr", options, location, "tt::int_to_ptr")
    }

    /// Trait representing the `tt.join` operation.
    pub trait JoinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Join, op_name = "tt.join");

    /// Constructs a new detached/owned [`JoinOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn join<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedJoinOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "join", options, location, "tt::join")
    }

    /// Trait representing the `tt.load` operation.
    pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Load, op_name = "tt.load");

    /// Constructs a new detached/owned [`LoadOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedLoadOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "load", options, location, "tt::load")
    }

    /// Trait representing the `tt.make_range` operation.
    pub trait MakeRangeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(MakeRange, op_name = "tt.make_range");

    /// Constructs a new detached/owned [`MakeRangeOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn make_range<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedMakeRangeOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "make_range", options, location, "tt::make_range")
    }

    /// Trait representing the `tt.make_tensor_descriptor` operation.
    pub trait MakeTensorDescriptorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(MakeTensorDescriptor, op_name = "tt.make_tensor_descriptor");

    /// Constructs a new detached/owned [`MakeTensorDescriptorOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn make_tensor_descriptor<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedMakeTensorDescriptorOperation<'c, 't> {
        build_typed_triton_operation(
            TritonDialect::Triton,
            "make_tensor_descriptor",
            options,
            location,
            "tt::make_tensor_descriptor",
        )
    }

    /// Trait representing the `tt.make_tensor_ptr` operation.
    pub trait MakeTensorPtrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(MakeTensorPtr, op_name = "tt.make_tensor_ptr");

    /// Constructs a new detached/owned [`MakeTensorPtrOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn make_tensor_ptr<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedMakeTensorPtrOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "make_tensor_ptr", options, location, "tt::make_tensor_ptr")
    }

    /// Trait representing the `tt.map_elementwise` operation.
    pub trait MapElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(MapElementwise, op_name = "tt.map_elementwise");

    /// Constructs a new detached/owned [`MapElementwiseOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn map_elementwise<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedMapElementwiseOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "map_elementwise", options, location, "tt::map_elementwise")
    }

    /// Trait representing the `tt.map_elementwise.return` operation.
    pub trait MapElementwiseReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(MapElementwiseReturn, op_name = "tt.map_elementwise.return");

    /// Constructs a new detached/owned [`MapElementwiseReturnOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn map_elementwise_return<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedMapElementwiseReturnOperation<'c, 't> {
        build_typed_triton_operation(
            TritonDialect::Triton,
            "map_elementwise.return",
            options,
            location,
            "tt::map_elementwise_return",
        )
    }

    /// Trait representing the `tt.mulhiui` operation.
    pub trait MulhiuiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Mulhiui, op_name = "tt.mulhiui");

    /// Constructs a new detached/owned [`MulhiuiOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn mulhiui<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedMulhiuiOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "mulhiui", options, location, "tt::mulhiui")
    }

    /// Trait representing the `tt.precise_divf` operation.
    pub trait PreciseDivfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(PreciseDivf, op_name = "tt.precise_divf");

    /// Constructs a new detached/owned [`PreciseDivfOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn precise_divf<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedPreciseDivfOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "precise_divf", options, location, "tt::precise_divf")
    }

    /// Trait representing the `tt.precise_sqrt` operation.
    pub trait PreciseSqrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(PreciseSqrt, op_name = "tt.precise_sqrt");

    /// Constructs a new detached/owned [`PreciseSqrtOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn precise_sqrt<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedPreciseSqrtOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "precise_sqrt", options, location, "tt::precise_sqrt")
    }

    /// Trait representing the `tt.print` operation.
    pub trait PrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Print, op_name = "tt.print");

    /// Constructs a new detached/owned [`PrintOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn print<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedPrintOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "print", options, location, "tt::print")
    }

    /// Trait representing the `tt.ptr_to_int` operation.
    pub trait PtrToIntOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(PtrToInt, op_name = "tt.ptr_to_int");

    /// Constructs a new detached/owned [`PtrToIntOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn ptr_to_int<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedPtrToIntOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "ptr_to_int", options, location, "tt::ptr_to_int")
    }

    /// Trait representing the `tt.reduce` operation.
    pub trait ReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Reduce, op_name = "tt.reduce");

    /// Constructs a new detached/owned [`ReduceOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn reduce<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedReduceOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "reduce", options, location, "tt::reduce")
    }

    /// Trait representing the `tt.reduce.return` operation.
    pub trait ReduceReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(ReduceReturn, op_name = "tt.reduce.return");

    /// Constructs a new detached/owned [`ReduceReturnOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn reduce_return<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedReduceReturnOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "reduce.return", options, location, "tt::reduce_return")
    }

    /// Trait representing the `tt.reshape` operation.
    pub trait ReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Reshape, op_name = "tt.reshape");

    /// Constructs a new detached/owned [`ReshapeOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn reshape<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedReshapeOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "reshape", options, location, "tt::reshape")
    }

    /// Trait representing the `tt.return` operation.
    pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Return, op_name = "tt.return");

    /// Constructs a new detached/owned [`ReturnOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn r#return<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedReturnOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "return", options, location, "tt::return")
    }

    /// Trait representing the `tt.scan` operation.
    pub trait ScanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Scan, op_name = "tt.scan");

    /// Constructs a new detached/owned [`ScanOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn scan<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedScanOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "scan", options, location, "tt::scan")
    }

    /// Trait representing the `tt.scan.return` operation.
    pub trait ScanReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(ScanReturn, op_name = "tt.scan.return");

    /// Constructs a new detached/owned [`ScanReturnOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn scan_return<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedScanReturnOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "scan.return", options, location, "tt::scan_return")
    }

    /// Trait representing the `tt.splat` operation.
    pub trait SplatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Splat, op_name = "tt.splat");

    /// Constructs a new detached/owned [`SplatOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn splat<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedSplatOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "splat", options, location, "tt::splat")
    }

    /// Trait representing the `tt.split` operation.
    pub trait SplitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Split, op_name = "tt.split");

    /// Constructs a new detached/owned [`SplitOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn split<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedSplitOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "split", options, location, "tt::split")
    }

    /// Trait representing the `tt.store` operation.
    pub trait StoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Store, op_name = "tt.store");

    /// Constructs a new detached/owned [`StoreOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn store<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedStoreOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "store", options, location, "tt::store")
    }

    /// Trait representing the `tt.trans` operation.
    pub trait TransOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Trans, op_name = "tt.trans");

    /// Constructs a new detached/owned [`TransOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn trans<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedTransOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "trans", options, location, "tt::trans")
    }

    /// Trait representing the `tt.unsplat` operation.
    pub trait UnsplatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

    mlir_op!(Unsplat, op_name = "tt.unsplat");

    /// Constructs a new detached/owned [`UnsplatOperation`] at the specified [`Location`].
    ///
    /// This constructor accepts explicit operation state via [`TritonOperationBuildOptions`].
    pub fn unsplat<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
        options: TritonOperationBuildOptions<'o, 'c, 't>,
        location: L,
    ) -> DetachedUnsplatOperation<'c, 't> {
        build_typed_triton_operation(TritonDialect::Triton, "unsplat", options, location, "tt::unsplat")
    }
}
#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::{Attribute, Block, Context, DetachedOp, OpRef, Operation, Type};

    #[test]
    fn test_triton_operation_mnemonic_lists() {
        assert_eq!(tt::OPERATIONS.len(), 50);
    }

    #[test]
    fn test_triton_operation_constructors() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        let tt_operation = tt::addptr(TritonOperationBuildOptions::new(), location);
        assert_eq!(tt_operation.name().to_string(), "tt.addptr");
    }

    #[test]
    fn test_triton_operation_build_options() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let block = context.block(&[(index_type, location)]);
        let operand = block.argument(0).unwrap();
        let region = context.region();

        let operation = tt::map_elementwise(
            TritonOperationBuildOptions::new()
                .add_operand(operand.into())
                .add_attribute("test_attr", context.unit_attribute().as_ref())
                .add_result_type(index_type.as_ref())
                .add_region(region),
            location,
        );

        assert_eq!(operation.name().to_string(), "tt.map_elementwise");
    }

    #[test]
    fn test_triton_operation_casting_checks_operation_name() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        let location = context.unknown_location();

        let operation = tt::addptr(TritonOperationBuildOptions::new(), location);
        let operation_ref = operation.as_ref();

        let addptr = unsafe { operation_ref.cast::<tt::AddptrOperationRef>() };
        assert!(addptr.is_some());
        assert_eq!(addptr.unwrap().name().to_string(), "tt.addptr");

        let advance = unsafe { operation_ref.cast::<tt::AdvanceOperationRef>() };
        assert!(advance.is_none());

        let addptr_detached = unsafe { operation.clone().cast::<tt::DetachedAddptrOperation>() };
        assert!(addptr_detached.is_some());

        let advance_detached = unsafe { operation.cast::<tt::DetachedAdvanceOperation>() };
        assert!(advance_detached.is_none());
    }
}
