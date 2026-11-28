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
use crate::lowering::values::DimensionExtension;
use crate::{ArrayTypeExtension, Error, Type};

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
        mlir_context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error>;
}

impl MemoryLoweringExtension for Memory {
    fn lower_transfer<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        self,
        input_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        mlir_context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<Vec<ValueRef<'b, 'c, 't>>, Error> {
        check_count!("input", input_values, 1, ProgramError);
        let empty_backend_configuration = mlir_context.string_attribute("");
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
            mlir_context.dictionary_attribute(&[mlir_context.named_attribute(
                mlir_context.identifier("_xla_buffer_placement"),
                mlir_context.string_attribute(self.placement_kind()),
            )]),
        );
        let operation = block.append_operation(operation)?;
        Ok(vec![operation.result(0)?.as_ref()])
    }
}

/// Provides lowering support of [`DataType`]s, and of values of that type, to StableHLO.
pub trait DataTypeLoweringExtension {
    /// Lowers one [`DataType`] to the corresponding MLIR element type.
    fn lower<'c, 't>(self, mlir_context: &'c MlirContext<'t>) -> Result<TypeRef<'c, 't>, Error>;

    /// Builds a dense elements attribute of `tensor_type` by splatting the `f64` input `value` in this data type.
    /// Boolean splats hold `value != 0.0`, integer splats truncate the fractional part, and floating-point splats
    /// round to the element format, including the supported `f8`, `f6`, and `f4` formats. [`DataType::Zero`] admits
    /// only zero, encoded in its `i1` carrier. Integer splats use the MLIR element type from [`Self::lower`],
    /// including signless `i1` for [`DataType::U1`].
    fn lower_f64_splat_attribute<'c, 't>(
        self,
        tensor_type: TensorTypeRef<'c, 't>,
        value: f64,
        mlir_context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error>;

    /// Builds a dense elements attribute of `tensor_type` by splatting the `i64` input `value` in this data type.
    /// Integer splats encode `value` directly at the target width, preserving all 64 bits for 64-bit integers.
    /// They use the MLIR element type from [`Self::lower`], including signless `i1` for [`DataType::U1`]. Other
    /// data types convert `value` to `f64` and use [`Self::lower_f64_splat_attribute`].
    fn lower_i64_splat_attribute<'c, 't>(
        self,
        tensor_type: TensorTypeRef<'c, 't>,
        value: i64,
        mlir_context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error>;
}

impl DataTypeLoweringExtension for DataType {
    fn lower<'c, 't>(self, mlir_context: &'c MlirContext<'t>) -> Result<TypeRef<'c, 't>, Error> {
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
                mlir_context.signless_integer_type(1).as_ref()
            }
            DataType::Boolean => mlir_context.signless_integer_type(1).as_ref(),
            DataType::I1 => mlir_context.signless_integer_type(1).as_ref(),
            DataType::I2 => mlir_context.signless_integer_type(2).as_ref(),
            DataType::I4 => mlir_context.signless_integer_type(4).as_ref(),
            DataType::I8 => mlir_context.signless_integer_type(8).as_ref(),
            DataType::I16 => mlir_context.signless_integer_type(16).as_ref(),
            DataType::I32 => mlir_context.signless_integer_type(32).as_ref(),
            DataType::I64 => mlir_context.signless_integer_type(64).as_ref(),
            DataType::U1 => {
                // StableHLO admits signless `i1` as its Boolean and one-bit carrier but rejects `ui1`. The unsigned
                // semantics of `U1` remain selected by Ryft operation metadata (e.g., comparison direction), while its
                // sole bit is unchanged.
                mlir_context.signless_integer_type(1).as_ref()
            }
            DataType::U2 => mlir_context.unsigned_integer_type(2).as_ref(),
            DataType::U4 => mlir_context.unsigned_integer_type(4).as_ref(),
            DataType::U8 => mlir_context.unsigned_integer_type(8).as_ref(),
            DataType::U16 => mlir_context.unsigned_integer_type(16).as_ref(),
            DataType::U32 => mlir_context.unsigned_integer_type(32).as_ref(),
            DataType::U64 => mlir_context.unsigned_integer_type(64).as_ref(),
            DataType::BF16 => mlir_context.bfloat16_type().as_ref(),
            DataType::F16 => mlir_context.float16_type().as_ref(),
            DataType::F32 => mlir_context.float32_type().as_ref(),
            DataType::F64 => mlir_context.float64_type().as_ref(),
            DataType::F4E2M1FN => mlir_context.float4e2m1fn_type().as_ref(),
            DataType::F6E2M3FN => mlir_context.float6e2m3fn_type().as_ref(),
            DataType::F6E3M2FN => mlir_context.float6e3m2fn_type().as_ref(),
            DataType::F8E3M4 => mlir_context.float8e3m4_type().as_ref(),
            DataType::F8E4M3 => mlir_context.float8e4m3_type().as_ref(),
            DataType::F8E4M3FN => mlir_context.float8e4m3fn_type().as_ref(),
            DataType::F8E4M3FNUZ => mlir_context.float8e4m3fnuz_type().as_ref(),
            DataType::F8E4M3B11FNUZ => mlir_context.float8e4m3b11fnuz_type().as_ref(),
            DataType::F8E5M2 => mlir_context.float8e5m2_type().as_ref(),
            DataType::F8E5M2FNUZ => mlir_context.float8e5m2fnuz_type().as_ref(),
            DataType::F8E8M0FNU => mlir_context.float8e8m0fnu_type().as_ref(),
            DataType::C64 => mlir_context.complex_type(mlir_context.float32_type()).as_ref(),
            DataType::C128 => mlir_context.complex_type(mlir_context.float64_type()).as_ref(),
        })
    }

    fn lower_f64_splat_attribute<'c, 't>(
        self,
        tensor_type: TensorTypeRef<'c, 't>,
        value: f64,
        mlir_context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        let attribute = match self {
            DataType::Zero if value == 0.0 => mlir_context.boolean_attribute(false).as_ref(),
            DataType::Boolean => mlir_context.boolean_attribute(value != 0.0).as_ref(),
            data_type if data_type.is_integer() => {
                return self.lower_i64_splat_attribute(tensor_type, value as i64, mlir_context);
            }
            data_type if data_type.is_floating_point() => {
                let element_type = self.lower(mlir_context)?.cast::<FloatTypeRef>().unwrap();
                mlir_context.float_attribute(element_type, value).as_ref()
            }
            _ => {
                return Err(Error::Lowering(LoweringError::UnsupportedDataType {
                    data_type: self,
                    context: "floating-point splat constant".to_string(),
                }));
            }
        };
        mlir_context
            .splatted_dense_attribute_elements_attribute(tensor_type, attribute)
            .map_err(Error::from)
    }

    fn lower_i64_splat_attribute<'c, 't>(
        self,
        tensor_type: TensorTypeRef<'c, 't>,
        value: i64,
        mlir_context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        if !self.is_integer() {
            return self.lower_f64_splat_attribute(tensor_type, value as f64, mlir_context);
        }
        let element_type = self.lower(mlir_context)?.cast::<IntegerTypeRef>().unwrap();
        mlir_context
            .splatted_dense_attribute_elements_attribute(
                tensor_type,
                mlir_context.integer_attribute(element_type, value),
            )
            .map_err(Error::from)
    }
}

/// Provides lowering support of [`ArrayType`]s, and of values of that type, to StableHLO.
pub trait ArrayTypeLoweringExtension {
    /// Lowers an [`ArrayType`] to a typed MLIR tensor type.
    fn lower<'c, 't, L: Location<'c, 't>>(
        &self,
        mlir_context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<TensorTypeRef<'c, 't>, Error>;

    /// Emits a splat of `value` with this array's element type and shape. Integer elements retain the low bits of
    /// `value` at their declared width, including all 64 bits for 64-bit elements. Complex elements have a zero
    /// imaginary component. This emits a scalar constant followed by a broadcast for arrays of positive rank.
    /// Callers that need a runtime shape must supply its extents separately.
    ///
    /// # Errors
    ///
    /// Propagates errors from lowering the element type, encoding the constant, or constructing the MLIR operations.
    fn lower_i64_splat<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        &self,
        value: i64,
        block: &mut B,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;

    /// Emits a splat of `value` with this array's element type and shape. The real component uses the conversion
    /// rules of [`DataTypeLoweringExtension::lower_f64_splat_attribute`], and the imaginary component of a complex
    /// result is zero. Like [`Self::lower_i64_splat`], this uses a scalar constant and broadcasts it for arrays of
    /// positive rank.
    ///
    /// # Errors
    ///
    /// Propagates errors from lowering the element type, encoding the constant, or constructing the MLIR operations.
    fn lower_f64_splat<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        &self,
        value: f64,
        block: &mut B,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;

    /// Emits [`Self::lower_i64_splat`] and annotates its result with this array's memory placement when it resides
    /// outside device memory. Constructors use this function to place their result in its declared memory space.
    ///
    /// # Errors
    ///
    /// Propagates errors from the splat and from [`MemoryLoweringExtension::lower_transfer`].
    fn lower_i64_splat_in_declared_memory<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        &self,
        value: i64,
        block: &mut B,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;

    /// Exposes a bounded dynamic value as a statically shaped value of its [`ArrayTypeExtension::physical_bound_type`].
    /// Sets the size of each dynamic axis to its physical bound. Elements outside the original runtime shape have
    /// unspecified values. Callers must discard those elements or accept unspecified padding.
    fn lower_physical_storage<'b, 'c: 'b, 't: 'c>(
        &self,
        value: ValueRef<'b, 'c, 't>,
        block: &mut BlockRef<'b, 'c, 't>,
        mlir_context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;

    /// Exposes the full buffer using [`Self::lower_physical_storage`] and fills every position outside the original
    /// runtime shape with `padding`. A statically shaped input passes through unchanged.
    fn lower_padded_physical_storage<'b, 'c: 'b, 't: 'c>(
        &self,
        value: ValueRef<'b, 'c, 't>,
        padding: PhysicalStoragePadding<'b, 'c, 't>,
        block: &mut BlockRef<'b, 'c, 't>,
        mlir_context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;

    /// Restores the dynamic dimensions of this type on a value of its [`ArrayTypeExtension::physical_bound_type`].
    /// Each entry in `sources` identifies the value and axis that define the corresponding output dimension.
    fn restore_dynamic_dimensions<'b, 'c: 'b, 't: 'c>(
        &self,
        value: ValueRef<'b, 'c, 't>,
        sources: &[(ValueRef<'b, 'c, 't>, usize)],
        block: &mut BlockRef<'b, 'c, 't>,
        mlir_context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;
}

impl ArrayTypeLoweringExtension for ArrayType {
    fn lower<'c, 't, L: Location<'c, 't>>(
        &self,
        mlir_context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<TensorTypeRef<'c, 't>, Error> {
        let element_type = self.data_type().lower(mlir_context)?;
        let dimensions = self
            .shape()
            .dimensions()
            .iter()
            .map(|size| match size {
                Dimension::Static(value) => MlirSize::Static(*value),
                Dimension::Dynamic(_) => MlirSize::Dynamic,
            })
            .collect::<Vec<_>>();
        let bounds = self.shape().dimensions().iter().map(Dimension::stable_hlo_dynamic_bound).collect::<Vec<_>>();
        let encoding = bounds
            .iter()
            .any(Option::is_some)
            .then(|| mlir_context.stable_hlo_tensor_type_extensions(bounds.as_slice()))
            .transpose()?
            .map(|attribute| attribute.as_ref());
        mlir_context
            .tensor_type(element_type, dimensions.as_slice(), encoding, location)
            .map_err(Error::from)
    }

    #[inline]
    fn lower_i64_splat<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        &self,
        value: i64,
        block: &mut B,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        SplatValue::Integer(value).lower(self, block, location)
    }

    #[inline]
    fn lower_f64_splat<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        &self,
        value: f64,
        block: &mut B,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        SplatValue::Float(value).lower(self, block, location)
    }

    fn lower_i64_splat_in_declared_memory<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        &self,
        value: i64,
        block: &mut B,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        let splat = self.lower_i64_splat(value, block, location)?;
        match self.memory() {
            Memory::Device => Ok(splat),
            memory => Ok(memory.lower_transfer(&[splat], block, block.context(), location)?[0]),
        }
    }

    fn lower_physical_storage<'b, 'c: 'b, 't: 'c>(
        &self,
        value: ValueRef<'b, 'c, 't>,
        block: &mut BlockRef<'b, 'c, 't>,
        mlir_context: &'c MlirContext<'t>,
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
            let size = ArrayType::scalar(DataType::I32).lower_i64_splat(
                i64::from(dimension_extent_to_i32(bound)?),
                block,
                location,
            )?;
            dimensions[axis] = Dimension::Static(bound);
            let exposed_type = self.clone().with_shape(Shape::new(dimensions.clone()));
            let exposed = block.append_operation(stable_hlo::set_dimension_size(
                value,
                size,
                exposed_type.lower(mlir_context, location)?,
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
        mlir_context: &'c MlirContext<'t>,
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

        let data = self.lower_physical_storage(value, block, mlir_context, location)?;
        let physical_type = self.physical_bound_type()?;
        let index_type = ArrayType::new(DataType::I32, physical_type.shape().clone());
        let index_tensor_type = index_type.lower(mlir_context, location)?;
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

        let padding = match padding {
            PhysicalStoragePadding::Constant(value) => physical_type.lower_f64_splat(value, block, location)?,
            PhysicalStoragePadding::Scalar(scalar) => block
                .append_operation(stable_hlo::broadcast(
                    scalar,
                    physical_type.lower(mlir_context, location)?,
                    &[],
                    location,
                )?)?
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
        mlir_context: &'c MlirContext<'t>,
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
                refined_type.lower(mlir_context, location)?,
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
        mlir_context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<TensorTypeRef<'c, 't>, Error>;
}

impl TypeLoweringExtension for Type {
    #[inline]
    fn lower<'c, 't, L: Location<'c, 't>>(
        &self,
        mlir_context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<TensorTypeRef<'c, 't>, Error> {
        match self {
            Type::Array(array_type) => array_type.lower(mlir_context, location),
            Type::Dimension(_) => ArrayType::scalar(DataType::I64).lower(mlir_context, location),
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

/// Converts `extent` to StableHLO's signed 64-bit shape representation.
///
/// # Errors
///
/// Returns [`DimensionError::ExtentExceedsBackendWidth`] if `extent` exceeds [`i64::MAX`].
#[inline]
pub fn dimension_extent_to_i64(extent: usize) -> Result<i64, Error> {
    i64::try_from(extent)
        .map_err(|_| DimensionError::ExtentExceedsBackendWidth { value: extent, maximum: i64::MAX as usize }.into())
}

/// Converts `extent` to the signed 32-bit representation required by dynamic StableHLO shape operations.
///
/// # Errors
///
/// Returns [`DimensionError::ExtentExceedsBackendWidth`] if `extent` exceeds [`i32::MAX`].
#[inline]
pub fn dimension_extent_to_i32(extent: usize) -> Result<i32, Error> {
    i32::try_from(extent)
        .map_err(|_| DimensionError::ExtentExceedsBackendWidth { value: extent, maximum: i32::MAX as usize }.into())
}

/// Host value used to build a splat, retaining integer precision until the target element type is known.
enum SplatValue {
    /// An integer encoded at the target integer width or converted to the target floating-point format.
    Integer(i64),

    /// A real number converted to the target element type.
    Float(f64),
}

impl SplatValue {
    /// Emits a scalar constant and broadcasts it to the array shape when the result has positive rank.
    fn lower<'b, 'c: 'b, 't: 'c, B: Block<'b, 'c, 't>, L: Copy + Location<'c, 't>>(
        self,
        array_type: &ArrayType,
        block: &mut B,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        let context = block.context();
        let data_type = array_type.data_type();
        let real_type = match data_type {
            DataType::C64 => DataType::F32,
            DataType::C128 => DataType::F64,
            data_type => data_type,
        };

        let scalar_type = context.tensor_type(real_type.lower(context)?, &[], None, location)?;
        let elements = match self {
            Self::Integer(value) => real_type.lower_i64_splat_attribute(scalar_type, value, context)?,
            Self::Float(value) => real_type.lower_f64_splat_attribute(scalar_type, value, context)?,
        };

        let real = block.append_operation(stable_hlo::constant(elements, location)?)?.result(0)?.as_ref();

        // Build a complex scalar from real and imaginary constants before broadcasting it.
        let scalar = if data_type.is_complex() {
            let zero = real_type.lower_f64_splat_attribute(scalar_type, 0.0, context)?;
            let imaginary = block.append_operation(stable_hlo::constant(zero, location)?)?.result(0)?.as_ref();
            block.append_operation(stable_hlo::complex(real, imaginary, location)?)?.result(0)?.as_ref()
        } else {
            real
        };

        if array_type.rank() == 0 {
            return Ok(scalar);
        }

        let tensor_type = array_type.lower(context, location)?;
        let broadcast = block.append_operation(stable_hlo::broadcast(scalar, tensor_type, &[], location)?)?;

        Ok(broadcast.result(0)?.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{DimensionBounds, DimensionType, Error as CoreError, ReferenceType};

    use crate::lowering::tests::{dynamic_dimension, test_vector_type};

    use super::*;

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

        // Unsigned splats truncate too, and `U1` keeps only the lowest bit of the result in its `i1` carrier.
        assert_eq!(lower(DataType::U8, 2.75).unwrap(), "dense<2> : tensor<ui8>");
        assert_eq!(lower(DataType::U1, 3.75).unwrap(), "dense<true> : tensor<i1>");

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
        // Unsigned splats use the physical carrier of their element type, so `U1` keeps only the lowest bit.
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
            }
        }
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
    fn test_array_type_lower_i64_splat() {
        // An integer beyond the exact range of `f64` keeps all of its bits. The memory of the type adds no annotation,
        // which is what `lower_i64_splat_in_declared_memory` is for.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block_with_no_arguments();
        let r#type = ArrayType::new_static(DataType::I64, [2]).with_memory(Memory::Host { pinned: true });
        let output = r#type.lower_i64_splat((1i64 << 53) + 1, &mut block.as_ref(), location).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations.last().unwrap().result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%c = stablehlo.constant dense<9007199254740993> : tensor<i64>",
                "%0 = stablehlo.broadcast_in_dim <<UNKNOWN SSA VALUE>>, dims = [] : (tensor<i64>) -> tensor<2xi64>",
            ],
        );

        // A scalar needs no broadcast, and a floating-point element type receives the converted value.
        let block = context.block_with_no_arguments();
        let output = ArrayType::scalar(DataType::F32).lower_i64_splat(3, &mut block.as_ref(), location).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations[0].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec!["%cst = stablehlo.constant dense<3.000000e+00> : tensor<f32>"],
        );
    }

    #[test]
    fn test_array_type_lower_f64_splat() {
        // The value is converted to the element type and broadcast to the shape.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block_with_no_arguments();
        let output = test_vector_type(2).lower_f64_splat(1.5, &mut block.as_ref(), location).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations.last().unwrap().result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%cst = stablehlo.constant dense<1.500000e+00> : tensor<f32>",
                "%0 = stablehlo.broadcast_in_dim <<UNKNOWN SSA VALUE>>, dims = [] : (tensor<f32>) -> tensor<2xf32>",
            ],
        );

        // A complex element type receives the value as its real component and a zero imaginary component.
        let block = context.block_with_no_arguments();
        let output = ArrayType::new_static(DataType::C64, [2])
            .lower_f64_splat(1.5, &mut block.as_ref(), location)
            .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations.last().unwrap().result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%cst = stablehlo.constant dense<1.500000e+00> : tensor<f32>",
                "%cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>",
                "%0 = stablehlo.complex <<UNKNOWN SSA VALUE>>, <<UNKNOWN SSA VALUE>> : tensor<complex<f32>>",
                "%0 = stablehlo.broadcast_in_dim <<UNKNOWN SSA VALUE>>, dims = [] : (tensor<complex<f32>>) -> \
                 tensor<2xcomplex<f32>>",
            ],
        );
    }

    #[test]
    fn test_array_type_lower_i64_splat_in_declared_memory() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let splat = [
            "%cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>".to_string(),
            "%0 = stablehlo.broadcast_in_dim <<UNKNOWN SSA VALUE>>, dims = [] : (tensor<f32>) -> tensor<4xf32>"
                .to_string(),
        ];

        // A value in device memory is the splat itself.
        let block = context.block_with_no_arguments();
        let output = test_vector_type(4).lower_i64_splat_in_declared_memory(0, &mut block.as_ref(), location).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations.last().unwrap().result(0).unwrap().as_ref());
        assert_eq!(operations.iter().map(ToString::to_string).collect::<Vec<_>>(), splat);

        // A value in host memory is the splat annotated with its placement.
        for (memory, kind) in
            [(Memory::Host { pinned: true }, "pinned_host"), (Memory::Host { pinned: false }, "unpinned_host")]
        {
            let block = context.block_with_no_arguments();
            let output = test_vector_type(4)
                .with_memory(memory)
                .lower_i64_splat_in_declared_memory(0, &mut block.as_ref(), location)
                .unwrap();
            let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
            assert_eq!(output, operations.last().unwrap().result(0).unwrap().as_ref());
            let annotation = format!(
                "%0 = stablehlo.custom_call @annotate_device_placement(<<UNKNOWN SSA VALUE>>) {{backend_config = \"\", \
                 has_side_effect = true, mhlo.frontend_attributes = {{_xla_buffer_placement = \"{kind}\"}}}} : \
                 (tensor<4xf32>) -> tensor<4xf32>",
            );
            assert_eq!(
                operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
                vec![splat[0].clone(), splat[1].clone(), annotation],
            );
        }
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
        assert_eq!(output, Ok(input));
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

    #[test]
    fn test_dimension_extent_to_i64() {
        assert_eq!(dimension_extent_to_i64(0), Ok(0));
        let maximum = i64::MAX as usize;
        assert_eq!(dimension_extent_to_i64(maximum), Ok(maximum as i64));
        let Some(extent) = maximum.checked_add(1) else { return };
        let error = dimension_extent_to_i64(extent).unwrap_err();
        let Error::Core(core_error) = error else { unreachable!() };
        let CoreError::Type(error) = *core_error else { unreachable!() };
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::ExtentExceedsBackendWidth { value: extent, maximum }),
        );
    }

    #[test]
    fn test_dimension_extent_to_i32() {
        assert_eq!(dimension_extent_to_i32(0), Ok(0));
        let maximum = i32::MAX as usize;
        assert_eq!(dimension_extent_to_i32(maximum), Ok(i32::MAX));
        let extent = maximum + 1;
        let error = dimension_extent_to_i32(extent).unwrap_err();
        let Error::Core(core_error) = error else { unreachable!() };
        let CoreError::Type(error) = *core_error else { unreachable!() };
        assert_eq!(
            error.downcast_custom::<DimensionError>(),
            Some(&DimensionError::ExtentExceedsBackendWidth { value: extent, maximum }),
        );
    }
}
