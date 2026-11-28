use ryft_core::macros::check_count;
use ryft_core::{ArrayType, DataType, Dimension, DimensionError, Memory, ProgramError, Shape};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::dialects::stable_hlo::CustomCallApiVersion;
use ryft_mlir::{
    Attribute, Block, BlockRef, Context as MlirContext, DenseElementsAttributeRef, FloatTypeRef, IntegerTypeRef,
    Location, LocationRef, Operation as MlirOperation, Size as MlirSize, TensorTypeRef, Type as MlirType, TypeRef,
    Value as MlirValue, ValueRef,
};

use crate::devices::MemoryExtension;
use crate::lowering::LoweringError;
use crate::{ArrayError, ArrayTypeExtension, Error, Type};

/// Provides lowering support transfers to a [`Memory`] to StableHLO.
pub trait MemoryLoweringExtension {
    /// Lowers one staged [`TransferToMemoryOperation`](ryft_core::TransferToMemoryOperation)
    /// to the `stablehlo.custom_call @annotate_device_placement` annotation that the XLA
    /// `ConvertMemoryPlacementToInternalAnnotations` and
    /// [`HostOffloader`](https://openxla.org/xla/tools_and_passes/host_offloading) passes legalize into copies
    /// annotated with memory spaces. The annotation uses API version 1, `has_side_effect = true`, an empty
    /// `backend_config`, and the destination kind in `_xla_buffer_placement` inside `mhlo.frontend_attributes`.
    ///
    /// These constraints are retained on CPU as well, because reference arrays only retag their types, but native
    /// arrays must honor the requested physical placement on every platform. The current XLA offloading pass treats
    /// pinned and unpinned host annotations alike internally, whereas function boundary attributes and PJRT memory
    /// validation retain the distinction at runtime boundaries. A plugin must expose the requested memory kind for
    /// native allocation.
    ///
    /// Placement does not affect the MLIR tensor type, so the output type is the input's type unchanged. Identity
    /// transfers (i.e., transfers whose destination equals the input's current space) still lower to the annotation,
    /// because placement round trips are meaningful to `HostOffloader` and must not be optimized away here.
    fn lower_transfer<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        self,
        input_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>;
}

impl MemoryLoweringExtension for Memory {
    fn lower_transfer<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        self,
        input_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 1, ProgramError);
        let empty_backend_configuration = context.string_attribute("");
        let mut operation = stable_hlo::custom_call(
            input_values,
            "annotate_device_placement",
            true,
            Some(empty_backend_configuration.as_ref()),
            CustomCallApiVersion::Original,
            &[],
            None,
            &[],
            None,
            &[input_values[0].r#type()?],
            location,
        )?;
        operation.set_discardable_attribute(
            "mhlo.frontend_attributes",
            context.dictionary_attribute(&[context.named_attribute(
                context.identifier("_xla_buffer_placement"),
                context.string_attribute(self.placement_kind()),
            )]),
        );
        let operation = block.append_operation(operation)?;
        Ok(vec![operation.result(0)?.as_ref()])
    }
}

/// Provides lowering support of [`DataType`]s, and of values of that type, to StableHLO.
pub trait DataTypeLoweringExtension {
    /// Lowers one [`DataType`] to the corresponding MLIR element type.
    fn lower<'c, 't>(self, context: &'c MlirContext<'t>) -> Result<TypeRef<'c, 't>, Error>;

    /// Builds a dense elements attribute of `tensor_type` by splatting the `f64` input `value` in this data type.
    /// Boolean splats hold `value != 0.0`, integer splats truncate the fractional part, and floating-point splats
    /// round to the element format, including the supported `f8`, `f6`, and `f4` formats. [`DataType::Zero`] admits
    /// only zero, encoded in its `i1` carrier. Integer splats use the MLIR element type from [`Self::lower`],
    /// including signless `i1` for [`DataType::U1`].
    fn lower_f64_splat_attribute<'c, 't>(
        self,
        tensor_type: TensorTypeRef<'c, 't>,
        value: f64,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error>;

    /// Builds a dense elements attribute of `tensor_type` by splatting the `i64` input `value` in this data type.
    /// Integer splats encode `value` directly at the target width, preserving all 64 bits for 64-bit integers.
    /// They use the MLIR element type from [`Self::lower`], including signless `i1` for [`DataType::U1`]. Other
    /// data types convert `value` to `f64` and use [`Self::lower_f64_splat_attribute`].
    fn lower_i64_splat_attribute<'c, 't>(
        self,
        tensor_type: TensorTypeRef<'c, 't>,
        value: i64,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error>;
}

impl DataTypeLoweringExtension for DataType {
    fn lower<'c, 't>(self, context: &'c MlirContext<'t>) -> Result<TypeRef<'c, 't>, Error> {
        Ok(match self {
            DataType::Token => {
                return Err(Error::Lowering(LoweringError::UnsupportedDataType {
                    data_type: self,
                    context: "tensor element representation".to_string(),
                }));
            }
            DataType::Zero => {
                // StableHLO has no element type that carries zero information. Use an `i1` tensor as an unobservable
                // physical carrier while retaining the logical `DataType::Zero` in the lowered and compiled metadata
                // of Ryft.
                context.signless_integer_type(1).as_ref()
            }
            DataType::Boolean => context.signless_integer_type(1).as_ref(),
            DataType::I1 => context.signless_integer_type(1).as_ref(),
            DataType::I2 => context.signless_integer_type(2).as_ref(),
            DataType::I4 => context.signless_integer_type(4).as_ref(),
            DataType::I8 => context.signless_integer_type(8).as_ref(),
            DataType::I16 => context.signless_integer_type(16).as_ref(),
            DataType::I32 => context.signless_integer_type(32).as_ref(),
            DataType::I64 => context.signless_integer_type(64).as_ref(),
            DataType::U1 => {
                // StableHLO admits signless `i1` as its Boolean and one-bit carrier but rejects `ui1`. The unsigned
                // semantics of `U1` remain selected by Ryft operation metadata (e.g., comparison direction), while its
                // sole bit is unchanged.
                context.signless_integer_type(1).as_ref()
            }
            DataType::U2 => context.unsigned_integer_type(2).as_ref(),
            DataType::U4 => context.unsigned_integer_type(4).as_ref(),
            DataType::U8 => context.unsigned_integer_type(8).as_ref(),
            DataType::U16 => context.unsigned_integer_type(16).as_ref(),
            DataType::U32 => context.unsigned_integer_type(32).as_ref(),
            DataType::U64 => context.unsigned_integer_type(64).as_ref(),
            DataType::BF16 => context.bfloat16_type().as_ref(),
            DataType::F16 => context.float16_type().as_ref(),
            DataType::F32 => context.float32_type().as_ref(),
            DataType::F64 => context.float64_type().as_ref(),
            DataType::F4E2M1FN => context.float4e2m1fn_type().as_ref(),
            DataType::F6E2M3FN => context.float6e2m3fn_type().as_ref(),
            DataType::F6E3M2FN => context.float6e3m2fn_type().as_ref(),
            DataType::F8E3M4 => context.float8e3m4_type().as_ref(),
            DataType::F8E4M3 => context.float8e4m3_type().as_ref(),
            DataType::F8E4M3FN => context.float8e4m3fn_type().as_ref(),
            DataType::F8E4M3FNUZ => context.float8e4m3fnuz_type().as_ref(),
            DataType::F8E4M3B11FNUZ => context.float8e4m3b11fnuz_type().as_ref(),
            DataType::F8E5M2 => context.float8e5m2_type().as_ref(),
            DataType::F8E5M2FNUZ => context.float8e5m2fnuz_type().as_ref(),
            DataType::F8E8M0FNU => context.float8e8m0fnu_type().as_ref(),
            DataType::C64 => context.complex_type(context.float32_type()).as_ref(),
            DataType::C128 => context.complex_type(context.float64_type()).as_ref(),
        })
    }

    fn lower_f64_splat_attribute<'c, 't>(
        self,
        tensor_type: TensorTypeRef<'c, 't>,
        value: f64,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        let attribute = match self {
            DataType::Zero if value == 0.0 => context.boolean_attribute(false).as_ref(),
            DataType::Boolean => context.boolean_attribute(value != 0.0).as_ref(),
            data_type if data_type.is_integer() => {
                return self.lower_i64_splat_attribute(tensor_type, value as i64, context);
            }
            data_type if data_type.is_floating_point() => {
                let element_type = self.lower(context)?.cast::<FloatTypeRef>().unwrap();
                context.float_attribute(element_type, value).as_ref()
            }
            _ => {
                return Err(Error::Lowering(LoweringError::UnsupportedDataType {
                    data_type: self,
                    context: "floating-point splat constant".to_string(),
                }));
            }
        };
        context.splatted_dense_attribute_elements_attribute(tensor_type, attribute).map_err(Error::from)
    }

    fn lower_i64_splat_attribute<'c, 't>(
        self,
        tensor_type: TensorTypeRef<'c, 't>,
        value: i64,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        if !self.is_integer() {
            return self.lower_f64_splat_attribute(tensor_type, value as f64, context);
        }
        let element_type = self.lower(context)?.cast::<IntegerTypeRef>().unwrap();
        context
            .splatted_dense_attribute_elements_attribute(tensor_type, context.integer_attribute(element_type, value))
            .map_err(Error::from)
    }
}

/// Provides lowering support of [`ArrayType`]s, and of values of that type, to StableHLO.
pub trait ArrayTypeLoweringExtension {
    /// Lowers an [`ArrayType`] to a typed MLIR tensor type.
    fn lower<'c, 't, L: Location<'c, 't>>(
        &self,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<TensorTypeRef<'c, 't>, Error>;

    /// Exposes a bounded dynamic value as a statically shaped value of its [`ArrayTypeExtension::physical_bound_type`].
    /// Sets the size of each dynamic axis to its physical bound. Elements outside the original runtime shape have
    /// unspecified values. Callers must discard those elements or accept unspecified padding.
    fn lower_physical_storage<'b, 'c: 'b, 't: 'c>(
        &self,
        value: ValueRef<'b, 'c, 't>,
        block: &mut BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;

    /// Exposes the full buffer using [`Self::lower_physical_storage`] and fills every position outside the original
    /// runtime shape with `padding`. A statically shaped input passes through unchanged.
    fn lower_padded_physical_storage<'b, 'c: 'b, 't: 'c>(
        &self,
        value: ValueRef<'b, 'c, 't>,
        padding: PhysicalStoragePadding<'b, 'c, 't>,
        block: &mut BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;

    /// Restores the dynamic dimensions of this type on a value of its [`ArrayTypeExtension::physical_bound_type`].
    /// Each entry in `sources` identifies the value and axis that define the corresponding output dimension.
    fn restore_dynamic_dimensions<'b, 'c: 'b, 't: 'c>(
        &self,
        value: ValueRef<'b, 'c, 't>,
        sources: &[(ValueRef<'b, 'c, 't>, usize)],
        block: &mut BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;
}

impl ArrayTypeLoweringExtension for ArrayType {
    fn lower<'c, 't, L: Location<'c, 't>>(
        &self,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<TensorTypeRef<'c, 't>, Error> {
        let element_type = self.data_type().lower(context)?;
        let dimensions = self
            .shape()
            .dimensions()
            .iter()
            .map(|size| match size {
                Dimension::Static(value) => MlirSize::Static(*value),
                Dimension::Dynamic(_) => MlirSize::Dynamic,
            })
            .collect::<Vec<_>>();
        let bounds = self.shape().dimensions().iter().map(stable_hlo_dynamic_dimension_bound).collect::<Vec<_>>();
        let encoding = bounds
            .iter()
            .any(Option::is_some)
            .then(|| context.stable_hlo_tensor_type_extensions(bounds.as_slice()))
            .transpose()?
            .map(|attribute| attribute.as_ref());
        context.tensor_type(element_type, dimensions.as_slice(), encoding, location).map_err(Error::from)
    }

    fn lower_physical_storage<'b, 'c: 'b, 't: 'c>(
        &self,
        value: ValueRef<'b, 'c, 't>,
        block: &mut BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        let physical_type = self.physical_bound_type()?;
        let mut dimensions = self.shape().dimensions().to_vec();
        let mut value = value;
        for (axis, dimension) in self.shape().dimensions().iter().enumerate() {
            if matches!(dimension, Dimension::Static(_)) {
                continue;
            }
            let bound = physical_type.shape().dimensions()[axis].value().unwrap();
            let size = lower_unplaced_constant_output(
                &[ArrayType::scalar(DataType::I32)],
                i64::from(dimension_extent_i32(bound)?),
                block,
                context,
                location,
            )?
            .remove(0);
            dimensions[axis] = Dimension::Static(bound);
            let exposed_type = self.clone().with_shape(Shape::new(dimensions.clone()));
            let exposed = block.append_operation(stable_hlo::set_dimension_size(
                value,
                size,
                exposed_type.lower(context, location)?,
                axis,
                location,
            )?)?;
            value = exposed.result(0).unwrap().as_ref();
        }
        Ok(value)
    }

    fn lower_padded_physical_storage<'b, 'c: 'b, 't: 'c>(
        &self,
        value: ValueRef<'b, 'c, 't>,
        padding: PhysicalStoragePadding<'b, 'c, 't>,
        block: &mut BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        let mut runtime_sizes = Vec::new();
        for (axis, dimension) in self.shape().dimensions().iter().enumerate() {
            if matches!(dimension, Dimension::Dynamic(_)) {
                let size = block.append_operation(stable_hlo::get_dimension_size(value, axis, location)?)?;
                runtime_sizes.push((axis, size.result(0)?.as_ref()));
            }
        }

        if runtime_sizes.is_empty() {
            return Ok(value);
        }

        let data = self.lower_physical_storage(value, block, context, location)?;
        let physical_type = self.physical_bound_type()?;
        let index_type = ArrayType::new(DataType::I32, physical_type.shape().clone());
        let index_tensor_type = index_type.lower(context, location)?;
        let mut in_bounds = None;
        for (axis, size) in runtime_sizes {
            let indices = block.append_operation(stable_hlo::iota(index_tensor_type, axis, location)?)?;
            let indices = indices.result(0).unwrap().as_ref();
            let size = block.append_operation(stable_hlo::broadcast(size, index_tensor_type, &[], location)?)?;
            let size = size.result(0).unwrap().as_ref();
            let axis_in_bounds = block.append_operation(stable_hlo::compare(
                indices,
                size,
                stable_hlo::ComparisonDirection::LessThan,
                stable_hlo::ComparisonType::for_type(indices.r#type()?)?,
                location,
            )?)?;
            let axis_in_bounds = axis_in_bounds.result(0).unwrap().as_ref();
            in_bounds = Some(match in_bounds {
                None => axis_in_bounds,
                Some(in_bounds) => {
                    block.append_operation(stable_hlo::and(in_bounds, axis_in_bounds, location)?)?.result(0)?.as_ref()
                }
            });
        }

        let physical_tensor_type = physical_type.lower(context, location)?;
        let padding = match padding {
            PhysicalStoragePadding::Constant(value) if physical_type.data_type().is_complex() => {
                let part_type = physical_type.clone().with_data_type(if physical_type.data_type() == DataType::C64 {
                    DataType::F32
                } else {
                    DataType::F64
                });
                let part_tensor_type = part_type.lower(context, location)?;
                let real = lower_f64_constant_splat(value, &part_type, part_tensor_type, block, context, location)?;
                let imaginary = lower_f64_constant_splat(0.0, &part_type, part_tensor_type, block, context, location)?;
                block.append_operation(stable_hlo::complex(real, imaginary, location)?)?.result(0)?.as_ref()
            }
            PhysicalStoragePadding::Constant(value) => {
                lower_f64_constant_splat(value, &physical_type, physical_tensor_type, block, context, location)?
            }
            PhysicalStoragePadding::Scalar(scalar) => block
                .append_operation(stable_hlo::broadcast(scalar, physical_tensor_type, &[], location)?)?
                .result(0)?
                .as_ref(),
        };

        let masked = block.append_operation(stable_hlo::select(in_bounds.unwrap(), data, padding, location)?)?;
        Ok(masked.result(0)?.as_ref())
    }

    fn restore_dynamic_dimensions<'b, 'c: 'b, 't: 'c>(
        &self,
        mut value: ValueRef<'b, 'c, 't>,
        sources: &[(ValueRef<'b, 'c, 't>, usize)],
        block: &mut BlockRef<'b, 'c, 't>,
        context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        if sources.len() != self.rank() {
            return Err(ProgramError::InvalidInputCount { expected: self.rank(), actual: sources.len() }.into());
        }

        let mut dimensions = self.physical_bound_type()?.shape().dimensions().to_vec();
        for (axis, dimension) in self.shape().dimensions().iter().enumerate() {
            if matches!(dimension, Dimension::Static(_)) {
                continue;
            }

            let size =
                block.append_operation(stable_hlo::get_dimension_size(sources[axis].0, sources[axis].1, location)?)?;
            let size = size.result(0).unwrap().as_ref();
            dimensions[axis] = dimension.clone();
            let refined_type = self.clone().with_shape(Shape::new(dimensions.clone()));
            let refined = block.append_operation(stable_hlo::set_dimension_size(
                value,
                size,
                refined_type.lower(context, location)?,
                axis,
                location,
            )?)?;

            value = refined.result(0)?.as_ref();
        }

        Ok(value)
    }
}

/// Provides lowering support of [`Type`]s to StableHLO.
pub trait TypeLoweringExtension {
    /// Lowers a [`Type`] to its StableHLO tensor representation.
    fn lower<'c, 't, L: Location<'c, 't>>(
        &self,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<TensorTypeRef<'c, 't>, Error>;
}

impl TypeLoweringExtension for Type {
    #[inline]
    fn lower<'c, 't, L: Location<'c, 't>>(
        &self,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<TensorTypeRef<'c, 't>, Error> {
        match self {
            Type::Array(array_type) => array_type.lower(context, location),
            Type::Dimension(_) => ArrayType::scalar(DataType::I64).lower(context, location),
            Type::Reference(_) => {
                Err(Error::Lowering(LoweringError::UnresolvedReference { construct: self.to_string() }))
            }
        }
    }
}

/// Value used by [`ArrayTypeLoweringExtension::lower_padded_physical_storage`] for elements outside the runtime shape.
#[derive(Copy, Clone, Debug)]
pub enum PhysicalStoragePadding<'b, 'c: 'b, 't: 'c> {
    /// A real constant converted to the array's element type. Complex elements receive a zero imaginary component.
    Constant(f64),

    /// A rank-0 MLIR tensor of the array's element type, broadcast with its exact value.
    Scalar(ValueRef<'b, 'c, 't>),
}

// TODO(eaplatanios): Review from here onwards.

/// Converts one dimension extent to StableHLO's signed 64-bit shape representation.
pub fn dimension_extent_i64(value: usize) -> Result<i64, Error> {
    i64::try_from(value)
        .map_err(|_| DimensionError::ExtentExceedsBackendWidth { value, maximum: i64::MAX as usize }.into())
}

/// Converts one dimension extent to the signed 32-bit representation required by dynamic StableHLO shape operations.
pub fn dimension_extent_i32(value: usize) -> Result<i32, Error> {
    i32::try_from(value)
        .map_err(|_| DimensionError::ExtentExceedsBackendWidth { value, maximum: i32::MAX as usize }.into())
}

/// Lowers static start indices to scalar `i64` StableHLO constants, as consumed by the index operands of
/// `stablehlo.dynamic_update_slice` when lowering the statically indexed `update_slice` operation, because StableHLO
/// has no statically indexed update operation.
pub fn lower_static_index_constants<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    start_indices: &[usize],
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let element_type = DataType::I64.lower(context)?;
    let tensor_type = context.tensor_type(element_type, &[], None, location)?;
    start_indices
        .iter()
        .map(|index| {
            let elements = DataType::I64.lower_i64_splat_attribute(tensor_type, *index as i64, context)?;
            let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
            Ok(constant.result(0).unwrap().as_ref())
        })
        .collect()
}

/// Lowers `integer_value` as a constant of the single type in `output_types`, without applying its memory placement
/// (refer to [`lower_constant_output`] for the placed variant). Scalars lower to one `stablehlo.constant`, while
/// shaped outputs and complex values are built from scalar constants and broadcast to the output shape.
///
/// # Errors
///
/// Returns an [`Error`] when `output_types` does not contain exactly one type, or when the type cannot be lowered or
/// represent `integer_value`.
pub fn lower_unplaced_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    output_types: &[ArrayType],
    integer_value: i64,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    check_count!("output", output_types, 1, ProgramError);
    let output_type = &output_types[0];
    let tensor_type = output_type.lower(context, location)?;
    let data_type = output_type.data_type();

    // Complex constants are composed as `value + 0i` from two real part constants through `stablehlo.complex`, so that
    // the dense elements attribute helpers stay real valued, because MLIR has no complex scalar attribute to splat.
    if data_type.is_complex() {
        let part_data_type = if data_type == DataType::C64 { DataType::F32 } else { DataType::F64 };
        let part_tensor_type = context.tensor_type(part_data_type.lower(context)?, &[], None, location)?;
        let real_elements = part_data_type.lower_i64_splat_attribute(part_tensor_type, integer_value, context)?;
        let real = block.append_operation(stable_hlo::constant(real_elements, location)?)?;
        let imaginary_elements = part_data_type.lower_i64_splat_attribute(part_tensor_type, 0, context)?;
        let imaginary = block.append_operation(stable_hlo::constant(imaginary_elements, location)?)?;
        let complex = block.append_operation(stable_hlo::complex(
            real.result(0).unwrap().as_ref(),
            imaginary.result(0).unwrap().as_ref(),
            location,
        )?)?;
        let complex = complex.result(0).unwrap().as_ref();
        if output_type.shape().dimensions().is_empty() {
            return Ok(vec![complex]);
        }
        let broadcast = block.append_operation(stable_hlo::broadcast(complex, tensor_type, &[], location)?)?;
        return Ok(vec![broadcast.result(0).unwrap().as_ref()]);
    }

    if !output_type.shape().dimensions().is_empty() {
        let scalar_tensor_type = context.tensor_type(data_type.lower(context)?, &[], None, location)?;
        let scalar_elements = data_type.lower_i64_splat_attribute(scalar_tensor_type, integer_value, context)?;
        let scalar_constant = block.append_operation(stable_hlo::constant(scalar_elements, location)?)?;
        let broadcast = block.append_operation(stable_hlo::broadcast(
            scalar_constant.result(0).unwrap().as_ref(),
            tensor_type,
            &[],
            location,
        )?)?;
        return Ok(vec![broadcast.result(0).unwrap().as_ref()]);
    }
    let elements = data_type.lower_i64_splat_attribute(tensor_type, integer_value, context)?;
    let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    Ok(vec![constant.result(0).unwrap().as_ref()])
}

/// Lowers one constant with an integer value like [`lower_unplaced_constant_output`] and applies the declared memory
/// placement of its output type to the final value when that placement is not the device memory.
pub fn lower_constant_output<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
    output_types: &[ArrayType],
    integer_value: i64,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
    let values = lower_unplaced_constant_output(output_types, integer_value, block, context, location)?;
    check_count!("output", output_types, 1, ProgramError);
    match output_types[0].memory() {
        Memory::Device => Ok(values),
        memory => memory.lower_transfer(&values, block, context, location),
    }
}

/// Returns the static dimensions for one tensor type.
pub fn static_dimensions(array_type: &ArrayType) -> Result<Vec<usize>, Error> {
    array_type
        .shape()
        .dimensions()
        .iter()
        .map(|size| match size {
            Dimension::Static(value) => Ok(*value),
            Dimension::Dynamic(_) => Err(Error::from(ArrayError::DynamicShape { shape: array_type.shape().clone() })),
        })
        .collect()
}

/// Lowers a scalar `u64` constant. It is shared by the group interval arithmetic of ragged dots, the device shard and
/// batch item arithmetic of `parallel_ragged_all_to_all`, and the coordinate arithmetic of
/// [`AxisIndexOperation`](ryft_core::AxisIndexOperation).
pub fn lower_u64_scalar_constant<'b, 'c: 'b, 't: 'c>(
    value: u64,
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let tensor_type = ArrayType::scalar(DataType::U64).lower(context, location)?;
    let attribute = context.dense_u64_elements_attribute(tensor_type, &[value])?;
    let constant = block.append_operation(stable_hlo::constant(attribute, location)?)?;
    Ok(constant.result(0)?.as_ref())
}

/// Converts a Ryft exclusive dynamic dimension bound to StableHLO's inclusive maximum dimension size.
pub fn stable_hlo_dynamic_dimension_bound(size: &Dimension) -> Option<usize> {
    match size {
        Dimension::Static(_) => None,
        Dimension::Dynamic(variable) => variable.bounds().upper().and_then(|upper| upper.checked_sub(1)),
    }
}

/// Lowers an arbitrary `f64` factor into a splatted scalar StableHLO constant whose element type matches
/// `output_type`, then broadcasts that scalar to the full output shape. It is used when synthesizing zeros, ones, and
/// padding values and when lowering literal constants.
pub fn lower_f64_constant_splat<'b, 'c: 'b, 't: 'c, B, L>(
    factor: f64,
    output_type: &ArrayType,
    output_tensor_type: TensorTypeRef<'c, 't>,
    block: &mut B,
    context: &'c MlirContext<'t>,
    location: L,
) -> Result<ValueRef<'b, 'c, 't>, Error>
where
    B: Block<'b, 'c, 't>,
    L: Copy + Location<'c, 't>,
{
    let data_type = output_type.data_type();
    let scalar_tensor_type = context.tensor_type(data_type.lower(context)?, &[], None, location)?;
    let elements = data_type.lower_f64_splat_attribute(scalar_tensor_type, factor, context)?;
    let scalar_constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
    if output_type.shape().dimensions().is_empty() {
        return Ok(scalar_constant.result(0).unwrap().as_ref());
    }
    let broadcast = block.append_operation(stable_hlo::broadcast(
        scalar_constant.result(0).unwrap().as_ref(),
        output_tensor_type,
        &[],
        location,
    )?)?;
    Ok(broadcast.result(0).unwrap().as_ref())
}

/// Packs scalar first-class dimension operands into the rank-1 `i64` shape tensor consumed by dynamic StableHLO
/// shape operations.
pub fn lower_explicit_shape<'b, 'c: 'b, 't: 'c>(
    extents: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let shape_type =
        context.tensor_type(context.signless_integer_type(64), &[MlirSize::Static(extents.len())], None, location)?;
    if extents.is_empty() {
        let elements = context.dense_i64_elements_attribute(shape_type, &[])?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        return Ok(constant.result(0).unwrap().as_ref());
    }

    let dimensions = extents
        .iter()
        .map(|extent| {
            let reshape = block.append_operation(stable_hlo::reshape(*extent, &[1], location)?)?;
            Ok(reshape.result(0).unwrap().as_ref())
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if let [dimension] = dimensions.as_slice() {
        return Ok(*dimension);
    }
    let shape = block.append_operation(stable_hlo::concatenate(dimensions.as_slice(), 0, location)?)?;
    Ok(shape.result(0).unwrap().as_ref())
}

/// Packs scalar first-class dimension operands into the rank-1 `i32` shape tensor required by
/// `stablehlo.dynamic_reshape`.
pub(crate) fn lower_explicit_reshape_shape<'b, 'c: 'b, 't: 'c>(
    extents: &[ValueRef<'b, 'c, 't>],
    block: &mut BlockRef<'b, 'c, 't>,
    context: &'c MlirContext<'t>,
    location: LocationRef<'c, 't>,
) -> Result<ValueRef<'b, 'c, 't>, Error> {
    let scalar_type = context.tensor_type(context.signless_integer_type(32), &[], None, location)?;
    let shape_type =
        context.tensor_type(context.signless_integer_type(32), &[MlirSize::Static(extents.len())], None, location)?;
    if extents.is_empty() {
        let elements = context.dense_i32_elements_attribute(shape_type, &[])?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        return Ok(constant.result(0).unwrap().as_ref());
    }

    let dimensions = extents
        .iter()
        .map(|extent| {
            let extent = block.append_operation(stable_hlo::convert(*extent, scalar_type, location)?)?;
            let extent = extent.result(0).unwrap().as_ref();
            let reshape = block.append_operation(stable_hlo::reshape(extent, &[1], location)?)?;
            Ok(reshape.result(0).unwrap().as_ref())
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if let [dimension] = dimensions.as_slice() {
        return Ok(*dimension);
    }
    let shape = block.append_operation(stable_hlo::concatenate(dimensions.as_slice(), 0, location)?)?;
    Ok(shape.result(0)?.as_ref())
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        DimensionBounds, DimensionType, DimensionVariable, Error as CoreError, Placeholder, ReferenceType,
        ZeroOperation,
    };

    use crate::lowering::tests::{ProgramBuilder, dynamic_dimension, test_vector_type, unproject_plain_program};
    use crate::{ArrayConstant, ProgramLoweringExtension, ProgramLoweringOptions};

    use super::*;

    #[test]
    fn test_lower_constant_output_memory_placement() {
        // Device outputs need no annotation, while host outputs gain one `annotate_device_placement` custom call in
        // addition to their `mhlo.memory_kind` signature attribute.
        let zero_program = |memory| {
            let mut builder = ProgramBuilder::new();
            let output_type = test_vector_type(4).with_memory(memory);
            let output =
                builder.add_instruction(ZeroOperation::new(output_type), Vec::new(), Vec::new(), None).unwrap()[0];
            builder
                .build::<Vec<ArrayConstant>, Vec<ArrayConstant>>(vec![output], Vec::new(), vec![Placeholder])
                .unwrap()
        };
        assert_eq!(
            unproject_plain_program(zero_program(Memory::Device))
                .to_rendered_mlir("main", &ProgramLoweringOptions::new().without_pruning())
                .unwrap(),
            indoc! {r#"
            module {
              func.func @main() -> tensor<4xf32> {
                %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
                return %0 : tensor<4xf32>
              }
            }
        "#},
        );
        for (memory, kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let expected = indoc! {r#"
                module {
                  func.func @main() -> (tensor<4xf32> {mhlo.memory_kind = "KIND"}) {
                    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
                    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32>
                    %1 = stablehlo.custom_call @annotate_device_placement(%0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "KIND"}} : (tensor<4xf32>) -> tensor<4xf32>
                    return %1 : tensor<4xf32>
                  }
                }
            "#}
            .replace("KIND", kind);
            assert_eq!(
                unproject_plain_program(zero_program(memory))
                    .to_rendered_mlir("main", &ProgramLoweringOptions::new().without_pruning())
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn test_memory_lower_transfer() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block(&[(test_vector_type(4).lower(&context, location).unwrap(), location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut block_ref = block.as_ref();
        let location = location.as_ref();

        // Device memory is annotated too, because placement round trips are meaningful to XLA's host offloader.
        let host = Memory::Host { pinned: true }.lower_transfer(&[input], &mut block_ref, &context, location).unwrap();
        let device = Memory::Device.lower_transfer(&host, &mut block_ref, &context, location).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(host, vec![operations[0].result(0).unwrap().as_ref()]);
        assert_eq!(device, vec![operations[1].result(0).unwrap().as_ref()]);
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%0 = stablehlo.custom_call @annotate_device_placement(<<UNKNOWN SSA VALUE>>) {backend_config = \"\", \
                 has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = \"pinned_host\"}} : \
                 (tensor<4xf32>) -> tensor<4xf32>",
                "%0 = stablehlo.custom_call @annotate_device_placement(<<UNKNOWN SSA VALUE>>) {backend_config = \"\", \
                 has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = \"device\"}} : \
                 (tensor<4xf32>) -> tensor<4xf32>",
            ],
        );

        // A transfer annotates exactly one input.
        assert!(matches!(
            Memory::Device.lower_transfer(&[input, input], &mut block_ref, &context, location),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 2 }),
            ),
        ));
    }

    #[test]
    fn test_dimension_extent_i64() {
        if usize::MAX <= i64::MAX as usize {
            return;
        }

        let error = dimension_extent_i64(usize::MAX).unwrap_err();
        let Error::Core(core_error) = error else { unreachable!() };
        let CoreError::Type(error) = *core_error else { unreachable!() };
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::ExtentExceedsBackendWidth { value: usize::MAX, maximum: i64::MAX as usize }),
        );
    }

    #[test]
    fn test_dimension_extent_i32() {
        let value = i32::MAX as usize + 1;
        let error = dimension_extent_i32(value).unwrap_err();
        let Error::Core(core_error) = error else { unreachable!() };
        let CoreError::Type(error) = *core_error else { unreachable!() };
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::ExtentExceedsBackendWidth { value, maximum: i32::MAX as usize }),
        );
    }

    #[test]
    fn test_static_dimensions() {
        let static_type = ArrayType::new_static(DataType::F32, [2, 3]);
        assert_eq!(static_dimensions(&static_type).unwrap(), vec![2, 3]);
        let dimension = DimensionVariable::new("size", DimensionBounds::unbounded());
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![dimension.into()]));
        assert!(matches!(
            static_dimensions(&dynamic_type),
            Err(Error::Array(error)) if matches!(error.as_ref(),
                ArrayError::DynamicShape { shape } if shape == dynamic_type.shape(),
            ),
        ));
    }

    #[test]
    fn test_array_type_lower() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let lower = |r#type: ArrayType| r#type.lower(&context, location).unwrap().to_string();
        assert_eq!(lower(ArrayType::new_static(DataType::F32, [2, 3])), "tensor<2x3xf32>");

        // Bounded dynamic dimensions carry their inclusive StableHLO bound, while unbounded ones carry none.
        let bounded_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(6)), Dimension::Static(3)]));
        assert_eq!(lower(bounded_type), "tensor<?x3xf32, #stablehlo.bounds<5, ?>>");
        let unbounded_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", None), Dimension::Static(3)]));
        assert_eq!(lower(unbounded_type), "tensor<?x3xf32>");
    }

    #[test]
    fn test_array_type_lower_physical_storage() {
        // The runtime size of the dynamic dimension is set to its physical bound without masking.
        let bounded_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(4)), Dimension::Static(2)]));
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block(&[(bounded_type.lower(&context, location).unwrap(), location)]);
        let input = block.argument(0).unwrap().as_ref();
        let output = bounded_type
            .lower_physical_storage(input, &mut block.as_ref(), &context, location.as_ref())
            .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations.last().unwrap().result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%c = stablehlo.constant dense<3> : tensor<i32>",
                "%0 = stablehlo.set_dimension_size <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, dim = 0 : \
                 (tensor<?x2xf32, #stablehlo.bounds<3, ?>>, tensor<i32>) -> tensor<3x2xf32>",
            ],
        );
    }

    #[test]
    fn test_array_type_lower_padded_physical_storage() {
        // The positions beyond the runtime size of the dynamic dimension are replaced with the padding value.
        let bounded_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(4)), Dimension::Static(2)]));
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block(&[(bounded_type.lower(&context, location).unwrap(), location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut block_ref = block.as_ref();
        let location = location.as_ref();
        let output = bounded_type
            .lower_padded_physical_storage(
                input,
                PhysicalStoragePadding::Constant(-1.0),
                &mut block_ref,
                &context,
                location,
            )
            .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations.last().unwrap().result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%0 = stablehlo.get_dimension_size <<UNKNOWN SSA VALUE>>, dim = 0 : (tensor<?x2xf32, \
                 #stablehlo.bounds<3, ?>>) -> tensor<i32>",
                "%c = stablehlo.constant dense<3> : tensor<i32>",
                "%0 = stablehlo.set_dimension_size <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, dim = 0 : \
                 (tensor<?x2xf32, #stablehlo.bounds<3, ?>>, tensor<i32>) -> tensor<3x2xf32>",
                "%0 = stablehlo.iota dim = 0 : tensor<3x2xi32>",
                "%0 = stablehlo.broadcast_in_dim <<UNKNOWN SSA VALUE>>, dims = [] : (tensor<i32>) -> tensor<3x2xi32>",
                "%0 = stablehlo.compare LT, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, SIGNED : (tensor<3x2xi32>, \
                 tensor<3x2xi32>) -> tensor<3x2xi1>",
                "%cst = stablehlo.constant dense<-1.000000e+00> : tensor<f32>",
                "%0 = stablehlo.broadcast_in_dim <<UNKNOWN SSA VALUE>>, dims = [] : (tensor<f32>) -> tensor<3x2xf32>",
                "%0 = stablehlo.select <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>> : \
                 tensor<3x2xi1>, tensor<3x2xf32>",
            ],
        );

        // Static values are already physical.
        let static_type = test_vector_type(2);
        let block = context.block(&[(static_type.lower(&context, location).unwrap(), location)]);
        let input = block.argument(0).unwrap().as_ref();
        let output = static_type.lower_padded_physical_storage(
            input,
            PhysicalStoragePadding::Constant(-1.0),
            &mut block.as_ref(),
            &context,
            location,
        );
        assert_eq!(output.unwrap(), input);
        assert_eq!(block.operations().unwrap().count(), 0);
    }

    #[test]
    fn test_array_type_lower_padded_physical_storage_with_scalar() {
        // The exact padding scalar is broadcast across the physical buffer.
        let bounded_type = ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(4))]));
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block(&[
            (bounded_type.lower(&context, location).unwrap(), location),
            (ArrayType::scalar(DataType::F32).lower(&context, location).unwrap(), location),
        ]);
        let input = block.argument(0).unwrap().as_ref();
        let padding = block.argument(1).unwrap().as_ref();
        let mut block_ref = block.as_ref();
        let location = location.as_ref();
        let output = bounded_type
            .lower_padded_physical_storage(
                input,
                PhysicalStoragePadding::Scalar(padding),
                &mut block_ref,
                &context,
                location,
            )
            .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations.last().unwrap().result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%0 = stablehlo.get_dimension_size <<UNKNOWN SSA VALUE>>, dim = 0 : (tensor<?xf32, \
                 #stablehlo.bounds<3>>) -> tensor<i32>",
                "%c = stablehlo.constant dense<3> : tensor<i32>",
                "%0 = stablehlo.set_dimension_size <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, dim = 0 : \
                 (tensor<?xf32, #stablehlo.bounds<3>>, tensor<i32>) -> tensor<3xf32>",
                "%0 = stablehlo.iota dim = 0 : tensor<3xi32>",
                "%0 = stablehlo.broadcast_in_dim <<UNKNOWN SSA VALUE>>, dims = [] : (tensor<i32>) -> tensor<3xi32>",
                "%0 = stablehlo.compare LT, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, SIGNED : (tensor<3xi32>, \
                 tensor<3xi32>) -> tensor<3xi1>",
                "%0 = stablehlo.broadcast_in_dim <<UNKNOWN SSA VALUE>>, dims = [] : (tensor<f32>) -> tensor<3xf32>",
                "%0 = stablehlo.select <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>> : \
                 tensor<3xi1>, tensor<3xf32>",
            ],
        );
    }

    #[test]
    fn test_array_type_restore_dynamic_dimensions() {
        // The runtime size of the dynamic dimension is read from the source value and axis.
        let bounded_type =
            ArrayType::new(DataType::F32, Shape::new(vec![dynamic_dimension("rows", Some(4)), Dimension::Static(2)]));
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block(&[
            (bounded_type.physical_bound_type().unwrap().lower(&context, location).unwrap(), location),
            (bounded_type.lower(&context, location).unwrap(), location),
        ]);
        let input = block.argument(0).unwrap().as_ref();
        let source = block.argument(1).unwrap().as_ref();
        let mut block_ref = block.as_ref();
        let location = location.as_ref();
        let sources = [(source, 0), (source, 1)];
        let output = bounded_type
            .restore_dynamic_dimensions(input, &sources, &mut block_ref, &context, location)
            .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations.last().unwrap().result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%0 = stablehlo.get_dimension_size <<UNKNOWN SSA VALUE>>, dim = 0 : (tensor<?x2xf32, \
                 #stablehlo.bounds<3, ?>>) -> tensor<i32>",
                "%0 = stablehlo.set_dimension_size <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>>, dim = 0 : \
                 (tensor<3x2xf32>, tensor<i32>) -> tensor<?x2xf32, #stablehlo.bounds<3, ?>>",
            ],
        );

        // Every dimension needs one source.
        assert!(matches!(
            bounded_type.restore_dynamic_dimensions(input, &sources[..1], &mut block_ref, &context, location),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
            ),
        ));
    }

    #[test]
    fn test_stable_hlo_dynamic_dimension_bound() {
        assert_eq!(stable_hlo_dynamic_dimension_bound(&Dimension::Static(4)), None);
        assert_eq!(stable_hlo_dynamic_dimension_bound(&dynamic_dimension("unbounded", None)), None);
        assert_eq!(stable_hlo_dynamic_dimension_bound(&dynamic_dimension("bounded", Some(6))), Some(5));
    }

    #[test]
    fn test_data_type_lower() {
        // Zero, Boolean, and unsigned one-bit values share the signless `i1` carrier, and tokens have no element type.
        let context = MlirContext::new();
        for (data_type, expected) in [
            (DataType::Zero, "i1"),
            (DataType::Boolean, "i1"),
            (DataType::I4, "i4"),
            (DataType::U1, "i1"),
            (DataType::U8, "ui8"),
            (DataType::BF16, "bf16"),
            (DataType::F8E4M3FN, "f8E4M3FN"),
            (DataType::C64, "complex<f32>"),
        ] {
            assert_eq!(data_type.lower(&context).unwrap().to_string(), expected);
        }
        assert!(matches!(
            DataType::Token.lower(&context),
            Err(Error::Lowering(LoweringError::UnsupportedDataType { data_type: DataType::Token, context }))
                if context == "tensor element representation",
        ));
    }

    #[test]
    fn test_data_type_lower_f64_splat_attribute() {
        // Boolean splats hold whether the value is nonzero, integer splats truncate it, and float splats round it to
        // the nearest value of their element type.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let lower = |data_type: DataType, value: f64| {
            let tensor_type = ArrayType::scalar(data_type).lower(&context, location).unwrap();
            let attribute = data_type.lower_f64_splat_attribute(tensor_type, value, &context);
            attribute.map(|attribute| attribute.to_string())
        };
        assert_eq!(lower(DataType::Boolean, 0.5).unwrap(), "dense<true> : tensor<i1>");
        assert_eq!(lower(DataType::I32, -2.75).unwrap(), "dense<-2> : tensor<i32>");
        assert_eq!(lower(DataType::F16, 0.1).unwrap(), "dense<9.997550e-02> : tensor<f16>");
        assert_eq!(lower(DataType::Zero, 0.0).unwrap(), "dense<false> : tensor<i1>");

        // Zero admits only a zero value, and complex values have no real splat.
        for (data_type, value) in [(DataType::Zero, 1.0), (DataType::C64, 1.0)] {
            assert!(matches!(
                lower(data_type, value),
                Err(Error::Lowering(LoweringError::UnsupportedDataType { data_type: actual, context }))
                    if actual == data_type && context == "floating-point splat constant",
            ));
        }
    }

    #[test]
    fn test_data_type_lower_i64_splat_attribute() {
        // Integer splats hold the exact value, which a conversion to `f64` would round.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let lower = |data_type: DataType, value: i64| {
            let tensor_type = ArrayType::scalar(data_type).lower(&context, location).unwrap();
            data_type.lower_i64_splat_attribute(tensor_type, value, &context).unwrap().to_string()
        };
        assert_eq!(lower(DataType::I64, i64::MAX), "dense<9223372036854775807> : tensor<i64>");
        assert_eq!(lower(DataType::F32, 3), "dense<3.000000e+00> : tensor<f32>");
    }

    #[test]
    fn test_data_type_lower_i64_splat_attribute_with_unsigned_types() {
        // Unsigned splats use the physical carrier of their element type, so `U1` keeps only the lowest bit, and the
        // `f64` splat truncates its value to the same integer.
        let context = MlirContext::new();
        let location = context.unknown_location();
        for data_type in [DataType::U1, DataType::U8] {
            let tensor_type = ArrayType::scalar(data_type).lower(&context, location).unwrap();
            for value in [0, 1, 2, 3] {
                let expected = match data_type {
                    DataType::U1 => format!("dense<{}> : tensor<i1>", value & 1 != 0),
                    DataType::U8 => format!("dense<{value}> : tensor<ui8>"),
                    _ => unreachable!(),
                };
                assert_eq!(
                    data_type.lower_i64_splat_attribute(tensor_type, value, &context).unwrap().to_string(),
                    expected,
                );
                assert_eq!(
                    data_type
                        .lower_f64_splat_attribute(tensor_type, value as f64 + 0.75, &context)
                        .unwrap()
                        .to_string(),
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_type_lower() {
        // Dimensions lower to scalar `i64` tensors, and references must be resolved before lowering.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let array_type = Type::Array(test_vector_type(2));
        assert_eq!(array_type.lower(&context, location).unwrap().to_string(), "tensor<2xf32>");
        let dimension_type = Type::Dimension(DimensionType::new("n", DimensionBounds::unbounded()));
        assert_eq!(dimension_type.lower(&context, location).unwrap().to_string(), "tensor<i64>");
        let reference_type = Type::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        assert!(matches!(
            reference_type.lower(&context, location),
            Err(Error::Lowering(LoweringError::UnresolvedReference { construct }))
                if construct == reference_type.to_string(),
        ));
    }
}
