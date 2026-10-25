use std::sync::Arc;

use ryft_core::{ArrayType, DataType, Dimension, Shape, Sharding};

/// Logical-to-physical argument and result mapping of one XLA executable boundary.
///
/// Ryft retains every logical leaf in its staged and compiled function metadata. Statically shaped
/// [`DataType::Zero`] leaves carry no runtime payload and are omitted from the StableHLO/PJRT signature; each mapping
/// stores the corresponding physical ordinal for materialized leaves and `None` for erased leaves. Dynamically shaped
/// zero-space leaves deliberately retain private `i1` carriers because the current executable ABI has no independent
/// shape transport. Current XLA compilation may impose stricter limits on dynamic tensors than this projection layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct XlaExecutableSignature {
    /// Physical argument ordinal for each logical flattened input.
    input_mapping: Arc<[Option<usize>]>,

    /// Physical result ordinal for each logical flattened output.
    output_mapping: Arc<[Option<usize>]>,

    /// Hidden scalar arguments that restore bounded dynamic dimensions on physicalized input tensors.
    input_dimensions: Arc<[XlaInputDimension]>,

    /// Hidden scalar results that report bounded dynamic logical output extents.
    output_dimensions: Arc<[XlaOutputDimension]>,

    /// Whether the executable carries the ordered-I/O boundary token, i.e., whether the program contains
    /// `OrderedIo` or `DeviceOrderedIo` effects.
    pub(crate) ordered_io: bool,

    /// Whether every ordered-I/O occurrence is `DeviceOrderedIo`, so the boundary token only needs per-device
    /// ordering and the executable may run on several devices.
    device_ordered_io: bool,

    /// Whether the executable contains unordered I/O.
    unordered_io: bool,

    /// Whether lowering emitted a runtime assertion, including synthesized checks.
    pub(crate) requires_assertion_handler: bool,

    /// Whether native CUDA artifact calls require the session launcher and execution-context user data.
    pub(crate) requires_cuda_kernel_runtime: bool,

    /// Whether native ROCm artifact calls require the separate HIP session owner.
    pub(crate) requires_rocm_kernel_runtime: bool,
}

/// One bounded dynamic input axis transported as a hidden scalar executable argument.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct XlaInputDimension {
    /// Logical flattened input containing the dynamic axis.
    logical_input_index: usize,

    /// Axis within the logical input.
    axis: usize,

    /// Physical executable argument containing the runtime axis extent.
    physical_input_index: usize,
}

impl XlaInputDimension {
    /// Returns the logical flattened input containing this dynamic axis.
    #[inline]
    pub(crate) fn logical_input_index(self) -> usize {
        self.logical_input_index
    }

    /// Returns the dynamic axis within the logical input.
    #[inline]
    pub(crate) fn axis(self) -> usize {
        self.axis
    }

    /// Returns the physical executable argument containing the runtime extent.
    #[inline]
    pub(crate) fn physical_input_index(self) -> usize {
        self.physical_input_index
    }
}

/// One bounded dynamic output axis transported as a hidden scalar executable result.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct XlaOutputDimension {
    /// Logical flattened output containing the dynamic axis.
    logical_output_index: usize,

    /// Axis within the logical output.
    axis: usize,

    /// Physical executable result containing the runtime axis extent.
    physical_output_index: usize,
}

impl XlaOutputDimension {
    /// Returns the logical flattened output containing this dynamic axis.
    #[inline]
    pub(crate) fn logical_output_index(self) -> usize {
        self.logical_output_index
    }

    /// Returns the dynamic axis within the logical output.
    #[inline]
    pub(crate) fn axis(self) -> usize {
        self.axis
    }

    /// Returns the physical executable result containing the runtime extent.
    #[inline]
    pub(crate) fn physical_output_index(self) -> usize {
        self.physical_output_index
    }
}

impl XlaExecutableSignature {
    /// Derives the executable boundary mapping for the provided logical input and output types.
    pub(crate) fn new(input_types: &[ArrayType], output_types: &[ArrayType]) -> Self {
        fn mapping(types: &[ArrayType]) -> Arc<[Option<usize>]> {
            let mut physical_index = 0;
            types
                .iter()
                .map(|r#type| {
                    if r#type.data_type().is_zero() && r#type.static_shape().is_some() {
                        None
                    } else {
                        let index = physical_index;
                        physical_index += 1;
                        Some(index)
                    }
                })
                .collect()
        }
        let input_mapping = mapping(input_types);
        let mut physical_input_index = input_mapping.iter().flatten().count();
        let mut input_dimensions = Vec::new();
        for (logical_input_index, input_type) in input_types.iter().enumerate() {
            for (axis, dimension) in input_type.shape().dimensions().iter().enumerate() {
                if dimension.value().is_none() && dimension.upper_bound().is_some() {
                    input_dimensions.push(XlaInputDimension { logical_input_index, axis, physical_input_index });
                    physical_input_index += 1;
                }
            }
        }
        let output_mapping = mapping(output_types);
        let mut physical_output_index = output_mapping.iter().flatten().count();
        let mut output_dimensions = Vec::new();
        for (logical_output_index, output_type) in output_types.iter().enumerate() {
            for (axis, dimension) in output_type.shape().dimensions().iter().enumerate() {
                if dimension.value().is_none() && dimension.upper_bound().is_some() {
                    output_dimensions.push(XlaOutputDimension { logical_output_index, axis, physical_output_index });
                    physical_output_index += 1;
                }
            }
        }
        Self {
            input_mapping,
            output_mapping,
            input_dimensions: input_dimensions.into(),
            output_dimensions: output_dimensions.into(),
            ordered_io: false,
            device_ordered_io: false,
            unordered_io: false,
            requires_assertion_handler: false,
            requires_cuda_kernel_runtime: false,
            requires_rocm_kernel_runtime: false,
        }
    }

    /// Attaches the compile-time effect metadata, also used when restoring persistent executable metadata.
    /// `device_ordered_io` is only meaningful together with `ordered_io` and is cleared otherwise.
    pub(crate) fn with_effects(
        mut self,
        ordered_io: bool,
        unordered_io: bool,
        requires_assertion_handler: bool,
        device_ordered_io: bool,
    ) -> Self {
        self.ordered_io = ordered_io;
        self.device_ordered_io = ordered_io && device_ordered_io;
        self.unordered_io = unordered_io;
        self.requires_assertion_handler = requires_assertion_handler;
        self
    }

    /// Returns whether this executable carries an ordered-I/O boundary token.
    pub(crate) fn has_ordered_io(&self) -> bool {
        self.ordered_io
    }

    /// Returns whether every ordered-I/O occurrence is `DeviceOrderedIo`, permitting multi-device execution with
    /// per-device ordering.
    pub(crate) fn has_device_ordered_io(&self) -> bool {
        self.device_ordered_io
    }

    /// Returns whether this executable contains unordered I/O.
    pub(crate) fn has_unordered_io(&self) -> bool {
        self.unordered_io
    }

    /// Returns whether execution requires the runtime assertion handler.
    pub(crate) fn requires_assertion_handler(&self) -> bool {
        self.requires_assertion_handler
    }

    /// Returns whether execution needs session-owned CUDA artifact resources.
    pub(crate) fn requires_cuda_kernel_runtime(&self) -> bool {
        self.requires_cuda_kernel_runtime
    }

    /// Restores the persisted CUDA runtime requirement without changing the program's effect classes.
    pub(crate) fn with_cuda_kernel_runtime(mut self, required: bool) -> Self {
        self.requires_cuda_kernel_runtime = required;
        self
    }

    /// Returns whether execution needs session-owned HIP artifact resources.
    pub(crate) fn requires_rocm_kernel_runtime(&self) -> bool {
        self.requires_rocm_kernel_runtime
    }

    /// Restores the persisted HIP owner requirement without changing the effect classification.
    pub(crate) fn with_rocm_kernel_runtime(mut self, required: bool) -> Self {
        self.requires_rocm_kernel_runtime = required;
        self
    }

    /// Returns whether execution participates in the effect barrier.
    pub(crate) fn has_effects(&self) -> bool {
        self.ordered_io || self.unordered_io || self.requires_assertion_handler
    }

    /// Returns the hidden ordered-I/O token argument ordinal, after every array and extent argument.
    pub(crate) fn ordered_io_input_index(&self) -> Option<usize> {
        self.ordered_io.then(|| self.input_mapping.iter().flatten().count() + self.input_dimensions.len())
    }

    /// Returns the hidden ordered-I/O token result ordinal, after every array and extent result.
    pub(crate) fn ordered_io_output_index(&self) -> Option<usize> {
        self.ordered_io.then(|| self.output_mapping.iter().flatten().count() + self.output_dimensions.len())
    }

    /// Returns the per-logical-input physical argument mapping.
    #[inline]
    pub(crate) fn input_mapping(&self) -> &[Option<usize>] {
        &self.input_mapping
    }

    /// Returns the per-logical-output physical result mapping.
    #[inline]
    pub(crate) fn output_mapping(&self) -> &[Option<usize>] {
        &self.output_mapping
    }

    /// Returns hidden bounded-dimension scalar arguments in physical executable order.
    #[inline]
    pub(crate) fn input_dimensions(&self) -> &[XlaInputDimension] {
        &self.input_dimensions
    }

    /// Returns hidden bounded dynamic-output extent results in physical executable order.
    #[inline]
    pub(crate) fn output_dimensions(&self) -> &[XlaOutputDimension] {
        &self.output_dimensions
    }

    /// Returns the number of physical array arguments, including hidden extent scalars but excluding tokens.
    pub(crate) fn array_input_count(&self) -> usize {
        self.input_mapping.iter().flatten().count() + self.input_dimensions.len()
    }

    /// Returns the number of physical array results, including hidden extent scalars but excluding tokens.
    pub(crate) fn array_output_count(&self) -> usize {
        self.output_mapping.iter().flatten().count() + self.output_dimensions.len()
    }

    /// Returns the total number of physical executable arguments.
    #[inline]
    pub(crate) fn physical_input_count(&self) -> usize {
        self.input_mapping.iter().flatten().count() + self.input_dimensions.len() + usize::from(self.ordered_io)
    }

    /// Returns the total number of physical executable results, including the hidden ordered-I/O token.
    pub(crate) fn physical_output_count(&self) -> usize {
        self.array_output_count() + usize::from(self.ordered_io)
    }

    /// Projects logical inputs into physical executable order.
    pub(crate) fn project_inputs<T: Clone>(&self, inputs: &[T]) -> Vec<T> {
        Self::project(&self.input_mapping, inputs)
    }

    /// Projects logical outputs into physical executable order.
    pub(crate) fn project_outputs<T: Clone>(&self, outputs: &[T]) -> Vec<T> {
        Self::project(&self.output_mapping, outputs)
    }

    /// Projects logical data inputs to bounded physical tensor types and appends hidden extent scalars.
    /// Native effect tokens are separate from array types and are excluded.
    pub(crate) fn physical_input_types(&self, inputs: &[ArrayType]) -> Vec<ArrayType> {
        let mut physical = Self::project(&self.input_mapping, inputs);
        let mut converted = vec![false; physical.len()];
        for input_dimension in self.input_dimensions.iter() {
            // The conversion rewrites every axis of the owning tensor at once, so an input with several dynamic axes
            // is converted only for its first hidden extent entry.
            let physical_index = self.input_mapping[input_dimension.logical_input_index].unwrap();
            if std::mem::replace(&mut converted[physical_index], true) {
                continue;
            }
            let dimensions = physical[physical_index]
                .shape()
                .dimensions()
                .iter()
                .map(|dimension| {
                    // Dimension bounds validate an exclusive upper above a non-negative lower, so a bounded dynamic
                    // axis always yields a static bound here; only an unbounded axis stays dynamic, and bounded-input
                    // materialization rejects it as having no static upper shape.
                    dimension
                        .upper_bound()
                        .and_then(|upper| upper.checked_sub(1))
                        .map_or_else(|| dimension.clone(), Dimension::Static)
                })
                .collect::<Vec<_>>();
            physical[physical_index] = physical[physical_index].clone().with_shape(Shape::new(dimensions));
        }
        physical.extend(self.input_dimensions.iter().map(|_| ArrayType::scalar(DataType::I32)));
        physical
    }

    /// Projects logical outputs and appends hidden output-extent scalar types, excluding native effect tokens.
    pub(crate) fn physical_output_types(&self, outputs: &[ArrayType]) -> Vec<ArrayType> {
        let mut physical = Self::project(&self.output_mapping, outputs);
        physical.extend(self.output_dimensions.iter().map(|_| ArrayType::scalar(DataType::I64)));
        physical
    }

    /// Projects logical input shardings and appends a replicated scalar sharding per hidden extent argument, on the
    /// mesh of the argument that owns the extent.
    pub(crate) fn physical_input_shardings(&self, shardings: &[Sharding]) -> Vec<Sharding> {
        let mut physical = Self::project(&self.input_mapping, shardings);
        physical.extend(
            self.input_dimensions
                .iter()
                .map(|dimension| Sharding::replicated(shardings[dimension.logical_input_index].mesh().clone(), 0)),
        );
        physical
    }

    /// Projects logical output shardings and appends a replicated scalar sharding per hidden extent result, on the
    /// mesh of the output that owns the extent.
    pub(crate) fn physical_output_shardings(&self, shardings: &[Sharding]) -> Vec<Sharding> {
        let mut physical = Self::project(&self.output_mapping, shardings);
        physical.extend(
            self.output_dimensions
                .iter()
                .map(|dimension| Sharding::replicated(shardings[dimension.logical_output_index].mesh().clone(), 0)),
        );
        physical
    }

    /// Projects one logical sequence according to `mapping`.
    fn project<T: Clone>(mapping: &[Option<usize>], values: &[T]) -> Vec<T> {
        assert_eq!(mapping.len(), values.len());
        mapping
            .iter()
            .zip(values)
            .filter_map(|(physical_index, value)| physical_index.map(|_| value.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::Placeholder;

    use crate::experimental::lowering::tests::dynamic_dimension;
    use crate::experimental::lowering::to_mlir_module_for_program;
    use crate::experimental::ops::{FlatXlaProgram, XlaProgramBuilder as ArrayIrXlaProgramBuilder};

    use super::*;

    #[test]
    fn test_xla_executable_signature_erases_only_static_zero_space_leaves() {
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let dynamic_zero = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("dynamic_zero", Some(3))]));
        let boolean = ArrayType::new(DataType::Boolean, Shape::new(vec![Dimension::Static(3)]));
        let signature = XlaExecutableSignature::new(
            &[static_zero.clone(), boolean.clone(), dynamic_zero.clone()],
            &[dynamic_zero, static_zero],
        );

        assert_eq!(signature.input_mapping(), &[None, Some(0), Some(1)]);
        assert_eq!(signature.output_mapping(), &[Some(0), None]);
        assert_eq!(signature.project_inputs(&[0, 1, 2]), vec![1, 2]);
        assert_eq!(signature.project_outputs(&[0, 1]), vec![0]);
        assert_eq!(signature.physical_input_count(), 3);
        assert_eq!(signature.input_dimensions().len(), 1);
        assert_eq!(signature.input_dimensions()[0].logical_input_index(), 2);
        assert_eq!(signature.input_dimensions()[0].axis(), 0);
        assert_eq!(signature.input_dimensions()[0].physical_input_index(), 2);
        assert_eq!(signature.output_dimensions().len(), 1);
        assert_eq!(signature.output_dimensions()[0].logical_output_index(), 0);
        assert_eq!(signature.output_dimensions()[0].axis(), 0);
        assert_eq!(signature.output_dimensions()[0].physical_output_index(), 1);
    }

    #[test]
    fn test_to_mlir_module_for_program_erases_static_zero_space_boundary() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatXlaProgram = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

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
    fn test_to_mlir_module_for_program_preserves_effects_that_produce_an_erased_output() {
        use ryft_core::PrintOperation;

        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![Dimension::Static(3)]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(zero_type.clone().into());
        let output = builder
            .add_instruction(PrintOperation::<ArrayType>::new("zero"), Vec::new(), vec![input], None)
            .unwrap()[0];
        let program: FlatXlaProgram = builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

        assert!(stablehlo.contains("stablehlo.custom_call @ryft.print"), "{stablehlo}");
        assert!(stablehlo.contains("has_side_effect = true"), "{stablehlo}");
        assert!(stablehlo.contains("return\n"), "{stablehlo}");
    }

    #[test]
    fn test_to_mlir_module_for_program_preserves_dynamic_zero_space_shape_carrier() {
        let zero_type = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("zero", Some(3))]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let input = builder.add_input(zero_type.clone().into());
        let program: FlatXlaProgram = builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        let stablehlo = to_mlir_module_for_program(&program, &[], &zero_type, &zero_type, "main", None, None).unwrap();

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
    fn test_to_mlir_module_for_program_isolates_bounded_dynamic_results_behind_one_barrier() {
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("count", Some(4))]));
        let static_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]));
        let mut builder = ArrayIrXlaProgramBuilder::new();
        let dynamic = builder.add_input(dynamic_type.clone().into());
        let r#static = builder.add_input(static_type.clone().into());
        let program: FlatXlaProgram =
            builder.build(vec![dynamic, r#static, dynamic], vec![Placeholder; 2], vec![Placeholder; 3]).unwrap();

        let stablehlo = to_mlir_module_for_program(
            &program,
            &[],
            &vec![dynamic_type.clone(), static_type.clone()],
            &vec![dynamic_type.clone(), static_type, dynamic_type],
            "main",
            None,
            None,
        )
        .unwrap();

        // Both dynamic results pass through the barrier, while the static result and the hidden extents bypass it.
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
}
