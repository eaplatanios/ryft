use std::sync::Arc;

use ryft_core::{ArrayType, DataType, Dimension as CoreDimension, Shape, Sharding};

// TODO(eaplatanios): Review this module.

/// Mapping from the logical arguments and results of one XLA executable boundary to its physical arguments and
/// results.
///
/// Ryft retains every logical leaf in its staged and compiled function metadata. Statically shaped
/// [`DataType::Zero`] leaves carry no runtime payload and are omitted from the StableHLO/PJRT signature, and each
/// mapping stores the corresponding physical ordinal for materialized leaves and `None` for erased leaves. Dynamically
/// shaped [`DataType::Zero`] leaves deliberately retain private `i1` carriers because the current executable ABI has no
/// independent shape transport. Current XLA compilation may impose stricter limits on dynamic tensors than this
/// projection layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExecutableSignature {
    /// Physical argument ordinal for each logical flattened input.
    input_mapping: Arc<[Option<usize>]>,

    /// Physical result ordinal for each logical flattened output.
    output_mapping: Arc<[Option<usize>]>,

    /// Hidden scalar arguments that restore bounded dynamic dimensions on physicalized input tensors.
    input_dimensions: Arc<[InputDimension]>,

    /// Hidden scalar results that report bounded dynamic logical output extents.
    output_dimensions: Arc<[OutputDimension]>,

    /// Whether the executable carries the ordered I/O boundary token (i.e., whether the program contains
    /// [`OrderedIo`](ryft_core::EffectClass::OrderedIo) or [`DeviceOrderedIo`](ryft_core::EffectClass::DeviceOrderedIo)
    /// effects).
    ordered_io: bool,

    /// Whether every ordered I/O occurrence is `DeviceOrderedIo`, so the boundary token only needs per-device ordering
    /// and the executable may run on several devices.
    device_ordered_io: bool,

    /// Whether the executable contains unordered I/O.
    unordered_io: bool,

    /// Whether lowering emitted a runtime assertion, including synthesized checks.
    requires_assertion_handler: bool,
}

/// One bounded dynamic input axis transported as a hidden scalar executable argument.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct InputDimension {
    /// Logical flattened input containing the dynamic axis.
    logical_input_index: usize,

    /// Axis within the logical input.
    axis: usize,

    /// Physical executable argument containing the runtime axis extent.
    physical_input_index: usize,
}

impl InputDimension {
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
pub(crate) struct OutputDimension {
    /// Logical flattened output containing the dynamic axis.
    logical_output_index: usize,

    /// Axis within the logical output.
    axis: usize,

    /// Physical executable result containing the runtime axis extent.
    physical_output_index: usize,
}

impl OutputDimension {
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

impl ExecutableSignature {
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
                    input_dimensions.push(InputDimension { logical_input_index, axis, physical_input_index });
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
                    output_dimensions.push(OutputDimension { logical_output_index, axis, physical_output_index });
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
        }
    }

    /// Attaches the effect metadata known at compilation time, also used when restoring persistent executable metadata.
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

    /// Returns whether this executable carries an ordered I/O boundary token.
    pub(crate) fn has_ordered_io(&self) -> bool {
        self.ordered_io
    }

    /// Returns whether every ordered I/O occurrence is `DeviceOrderedIo`, permitting execution on several devices with
    /// per-device ordering.
    pub(crate) fn has_device_ordered_io(&self) -> bool {
        self.device_ordered_io
    }

    /// Returns whether this executable contains unordered I/O.
    pub(crate) fn has_unordered_io(&self) -> bool {
        self.unordered_io
    }

    /// Returns whether lowering emitted a runtime assertion, including synthesized checks.
    #[inline]
    pub(crate) fn requires_assertion_handler(&self) -> bool {
        self.requires_assertion_handler
    }

    /// Returns whether execution participates in the effect barrier, which is the case for executables with ordered or
    /// unordered I/O and for executables that require the runtime assertion handler.
    pub(crate) fn has_effects(&self) -> bool {
        self.ordered_io || self.unordered_io || self.requires_assertion_handler
    }

    /// Returns the ordinal of the hidden ordered I/O token argument, which follows every array and extent argument.
    pub(crate) fn ordered_io_input_index(&self) -> Option<usize> {
        self.ordered_io.then(|| self.input_mapping.iter().flatten().count() + self.input_dimensions.len())
    }

    /// Returns the ordinal of the hidden ordered I/O token result, which follows every array and extent result.
    pub(crate) fn ordered_io_output_index(&self) -> Option<usize> {
        self.ordered_io.then(|| self.output_mapping.iter().flatten().count() + self.output_dimensions.len())
    }

    /// Returns the physical argument ordinal of each logical input, or `None` for an erased input.
    #[inline]
    pub(crate) fn input_mapping(&self) -> &[Option<usize>] {
        &self.input_mapping
    }

    /// Returns the physical result ordinal of each logical output, or `None` for an erased output.
    #[inline]
    pub(crate) fn output_mapping(&self) -> &[Option<usize>] {
        &self.output_mapping
    }

    /// Returns the hidden scalar arguments of bounded dynamic input dimensions in physical executable order.
    #[inline]
    pub(crate) fn input_dimensions(&self) -> &[InputDimension] {
        &self.input_dimensions
    }

    /// Returns the hidden extent results of bounded dynamic output dimensions in physical executable order.
    #[inline]
    pub(crate) fn output_dimensions(&self) -> &[OutputDimension] {
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

    /// Returns the total number of physical executable arguments, including the hidden ordered I/O token.
    #[inline]
    pub(crate) fn physical_input_count(&self) -> usize {
        self.input_mapping.iter().flatten().count() + self.input_dimensions.len() + usize::from(self.ordered_io)
    }

    /// Returns the total number of physical executable results, including the hidden ordered I/O token.
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
                    // Dimension bounds validate an exclusive upper bound above a non-negative lower bound, so a bounded
                    // dynamic axis always yields a static bound here. Only an unbounded axis stays dynamic, and the
                    // materialization of bounded inputs rejects it as having no static upper shape.
                    dimension
                        .upper_bound()
                        .and_then(|upper| upper.checked_sub(1))
                        .map_or_else(|| dimension.clone(), CoreDimension::Static)
                })
                .collect::<Vec<_>>();
            physical[physical_index] = physical[physical_index].clone().with_shape(Shape::new(dimensions));
        }
        physical.extend(self.input_dimensions.iter().map(|_| ArrayType::scalar(DataType::I32)));
        physical
    }

    /// Projects logical outputs and appends the scalar types of hidden output extents, excluding native effect tokens.
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
                .map(|dimension| extent_sharding(&shardings[dimension.logical_input_index])),
        );
        physical
    }

    /// Projects optional logical input shardings like [`Self::physical_input_shardings`], leaving a hidden extent
    /// argument unannotated when the argument that owns it is unannotated.
    pub(crate) fn physical_optional_input_shardings(&self, shardings: &[Option<Sharding>]) -> Vec<Option<Sharding>> {
        let mut physical = Self::project(&self.input_mapping, shardings);
        physical.extend(
            self.input_dimensions
                .iter()
                .map(|dimension| shardings[dimension.logical_input_index].as_ref().map(extent_sharding)),
        );
        physical
    }

    /// Projects optional logical output shardings and appends the sharding of each hidden extent result, which is
    /// replicated on the mesh of the output that owns the extent, or absent when that output is unannotated.
    pub(crate) fn physical_optional_output_shardings(&self, shardings: &[Option<Sharding>]) -> Vec<Option<Sharding>> {
        let mut physical = Self::project(&self.output_mapping, shardings);
        physical.extend(
            self.output_dimensions
                .iter()
                .map(|dimension| shardings[dimension.logical_output_index].as_ref().map(extent_sharding)),
        );
        physical
    }

    /// Records that lowering emitted a runtime assertion, so that executions need the runtime assertion handler.
    #[inline]
    pub(crate) fn require_assertion_handler(&mut self) {
        self.requires_assertion_handler = true;
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

/// Returns the sharding of a hidden extent scalar, which is replicated on the mesh of the value that owns the extent.
fn extent_sharding(owner_sharding: &Sharding) -> Sharding {
    Sharding::replicated(owner_sharding.mesh().clone(), 0)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::lowering::tests::{dynamic_dimension, test_manual_mesh};

    use super::*;

    #[test]
    fn test_executable_signature_new() {
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));
        let dynamic_zero = ArrayType::new(DataType::Zero, Shape::new(vec![dynamic_dimension("dynamic_zero", Some(3))]));
        let boolean = ArrayType::new(DataType::Boolean, Shape::new(vec![CoreDimension::Static(3)]));
        let signature = ExecutableSignature::new(
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
    fn test_executable_signature_with_effects() {
        let scalar = ArrayType::scalar(DataType::F32);
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("count", Some(3))]));
        let signature = ExecutableSignature::new(&[scalar.clone(), dynamic.clone()], &[dynamic, scalar]);

        // Without effects, the boundary has no token and does not participate in the effect barrier.
        assert!(!signature.has_ordered_io());
        assert!(!signature.has_device_ordered_io());
        assert!(!signature.has_unordered_io());
        assert!(!signature.requires_assertion_handler());
        assert!(!signature.has_effects());
        assert_eq!(signature.ordered_io_input_index(), None);
        assert_eq!(signature.ordered_io_output_index(), None);
        assert_eq!(signature.array_input_count(), 3);
        assert_eq!(signature.array_output_count(), 3);
        assert_eq!(signature.physical_input_count(), 3);
        assert_eq!(signature.physical_output_count(), 3);

        // Ordered I/O appends one token after every array and hidden extent argument and result.
        let ordered = signature.clone().with_effects(true, false, false, true);
        assert!(ordered.has_ordered_io());
        assert!(ordered.has_device_ordered_io());
        assert!(!ordered.has_unordered_io());
        assert!(ordered.has_effects());
        assert_eq!(ordered.ordered_io_input_index(), Some(3));
        assert_eq!(ordered.ordered_io_output_index(), Some(3));
        assert_eq!(ordered.array_input_count(), 3);
        assert_eq!(ordered.array_output_count(), 3);
        assert_eq!(ordered.physical_input_count(), 4);
        assert_eq!(ordered.physical_output_count(), 4);

        // Device ordering only applies together with ordered I/O, while unordered I/O and runtime assertions still
        // participate in the effect barrier.
        let unordered = signature.clone().with_effects(false, true, false, true);
        assert!(!unordered.has_device_ordered_io());
        assert!(unordered.has_unordered_io());
        assert!(unordered.has_effects());
        assert_eq!(unordered.physical_input_count(), 3);
        let asserting = signature.with_effects(false, false, true, false);
        assert!(asserting.requires_assertion_handler());
        assert!(asserting.has_effects());
    }

    #[test]
    fn test_executable_signature_physical_input_types() {
        // Every bounded dynamic axis of a materialized input becomes its physical bound, and each one appends a hidden
        // `i32` extent argument, while static zero-sized inputs are erased.
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));
        let dynamic = ArrayType::new(
            DataType::F32,
            Shape::new(vec![dynamic_dimension("rows", Some(3)), dynamic_dimension("columns", Some(5))]),
        );
        let signature = ExecutableSignature::new(&[static_zero.clone(), dynamic.clone()], &[]);
        assert_eq!(
            signature.physical_input_types(&[static_zero, dynamic]),
            vec![
                ArrayType::new_static(DataType::F32, [2, 4]),
                ArrayType::scalar(DataType::I32),
                ArrayType::scalar(DataType::I32),
            ],
        );
    }

    #[test]
    fn test_executable_signature_physical_output_types() {
        // Dynamic outputs keep their logical types and each bounded dynamic axis appends a hidden `i64` extent result.
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(3))]));
        let signature = ExecutableSignature::new(&[], &[dynamic.clone(), static_zero.clone()]);
        assert_eq!(
            signature.physical_output_types(&[dynamic.clone(), static_zero]),
            vec![dynamic, ArrayType::scalar(DataType::I64)],
        );
    }

    #[test]
    fn test_executable_signature_physical_input_shardings() {
        // Each hidden extent argument is a replicated scalar on the mesh of the input that owns the extent, and it
        // stays unannotated when that input is unannotated.
        let mesh = test_manual_mesh("x", 2);
        let static_zero = ArrayType::new(DataType::Zero, Shape::new(vec![CoreDimension::Static(3)]));
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(3))]));
        let signature = ExecutableSignature::new(&[static_zero, dynamic.clone(), dynamic], &[]);
        let vector_sharding = Sharding::replicated(mesh.clone(), 1);
        let scalar_sharding = Sharding::replicated(mesh, 0);
        assert_eq!(
            signature.physical_input_shardings(&[
                vector_sharding.clone(),
                vector_sharding.clone(),
                vector_sharding.clone(),
            ]),
            vec![vector_sharding.clone(), vector_sharding.clone(), scalar_sharding.clone(), scalar_sharding.clone()],
        );
        assert_eq!(
            signature.physical_optional_input_shardings(&[None, Some(vector_sharding.clone()), None]),
            vec![Some(vector_sharding), None, Some(scalar_sharding), None],
        );
    }

    #[test]
    fn test_executable_signature_physical_optional_output_shardings() {
        let mesh = test_manual_mesh("x", 2);
        let dynamic = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(3))]));
        let signature = ExecutableSignature::new(&[], &[dynamic.clone(), dynamic]);
        let vector_sharding = Sharding::replicated(mesh.clone(), 1);
        assert_eq!(
            signature.physical_optional_output_shardings(&[Some(vector_sharding.clone()), None]),
            vec![Some(vector_sharding), None, Some(Sharding::replicated(mesh, 0)), None],
        );
    }

    #[test]
    fn test_executable_signature_require_assertion_handler() {
        let mut signature = ExecutableSignature::new(&[], &[]);
        assert!(!signature.requires_assertion_handler());
        signature.require_assertion_handler();
        assert!(signature.requires_assertion_handler());
        assert!(signature.has_effects());
    }
}
