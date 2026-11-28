use std::borrow::Cow;

use ryft_core::{
    Array, ArrayType, CaptureReference, DataType, DimensionValue, LogicalMesh, Memory, ProgramError, Type as CoreType,
    Typed, Value,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Attribute, Block, Context as MlirContext, DenseElementsAttributeRef, Location, Operation as MlirOperation,
    Value as MlirValue, ValueRef,
};

use crate::lowering::LoweringError;
use crate::lowering::contexts::ProgramLoweringContext;
use crate::lowering::types::{
    ArrayTypeLoweringExtension, DataTypeLoweringExtension, MemoryLoweringExtension, dimension_extent_i64,
};
use crate::{ArrayConstant, Constant, Error};

/// Provides lowering support for array literals to StableHLO. [`Self::lower_constant`] embeds the value's dense
/// elements in a StableHLO constant (e.g., for a [`ConstantOperation`](ryft_core::ConstantOperation) carrying
/// a concrete array).
pub(crate) trait ArrayLiteralLoweringExtension: 'static + Value<Type = ArrayType> {
    /// Builds a [`DenseElementsAttributeRef`] that contains this value.
    fn lower_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error>;

    /// Lowers this concrete value to a StableHLO constant operation and applies its declared memory placement.
    fn lower_constant<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        let value_type = self.r#type();
        let tensor_type = value_type.lower(context, location)?;
        let elements = self.lower_attribute(tensor_type, context)?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        let constant = constant.result(0)?.as_ref();
        match value_type.memory() {
            Memory::Device => Ok(constant),
            memory => Ok(memory.lower_transfer(&[constant], block, context, location)?[0]),
        }
    }
}

impl ArrayLiteralLoweringExtension for Array {
    fn lower_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        // This host literal lowering is used by `ConstantOperation` and by MLIR snapshot tooling.
        let data_type = self.r#type().data_type();
        if data_type == DataType::Zero {
            return data_type.lower_f64_scalar_elements_attribute(tensor_type, 0.0, context);
        }

        if data_type == DataType::Token {
            return Err(Error::Lowering(LoweringError::UnsupportedElementType {
                data_type,
                context: "dense literal materialization".to_string(),
            }));
        }

        // Layout-free arrays already store values in MLIR's logical row-major order. Explicit layouts instead require
        // traversal through `ArrayAddressing`, which `logical_bytes` performs while omitting holes and tile padding.
        let bytes = if self.r#type().layout().is_none() {
            Cow::Borrowed(self.storage_bytes())
        } else {
            Cow::Owned(self.logical_bytes())
        };

        // Ryft storage uses portable little-endian element encodings, whereas MLIR's raw-buffer API consumes native
        // storage representations. Normalize multi-byte components only on big-endian targets.
        #[cfg(target_endian = "big")]
        let bytes = {
            let mut bytes = bytes;
            let component_byte_count = match data_type {
                DataType::I16 | DataType::U16 | DataType::BF16 | DataType::F16 => 2,
                DataType::I32 | DataType::U32 | DataType::F32 | DataType::C64 => 4,
                DataType::I64 | DataType::U64 | DataType::F64 | DataType::C128 => 8,
                _ => 1,
            };
            if component_byte_count > 1 {
                bytes.to_mut().chunks_exact_mut(component_byte_count).for_each(<[u8]>::reverse);
            }
            bytes
        };

        // MLIR's typed constructors own the required packing for one-bit values. Ryft deliberately keeps these values
        // byte-padded in host arrays, so decode only their low bit at this representation boundary.
        if matches!(data_type, DataType::Boolean | DataType::I1 | DataType::U1) {
            return Ok(context
                .dense_bool_elements_attribute(
                    tensor_type,
                    bytes.iter().map(|value| value & 1 != 0).collect::<Vec<_>>().as_slice(),
                )?
                .cast::<DenseElementsAttributeRef>()
                .unwrap());
        }

        context.dense_elements_attribute_from_raw_buffer(tensor_type, bytes.as_ref()).map_err(Error::from)
    }
}

impl ArrayLiteralLoweringExtension for ArrayConstant {
    #[inline]
    fn lower_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        match self {
            Self::Captured(value) => Err(ProgramError::Concretization {
                message: format!(
                    "captured constant #{} cannot be materialized as a literal without capture bindings",
                    value.index()
                ),
            }
            .into()),
            Self::Session(value) => Err(ProgramError::Concretization {
                message: format!("session constant `{value}` cannot be materialized as a literal"),
            }
            .into()),
            Self::Boolean(value) => Array::scalar(*value)?.lower_attribute(tensor_type, context),
        }
    }
}

/// Provides lowering support for [`DimensionValue`]s to StableHLO.
pub(crate) trait DimensionValueLoweringExtension {
    /// Lowers one first-class dimension extent to the scalar `i64` [`stablehlo.constant`](stable_hlo::constant)
    /// that represents a dimension in StableHLO. Immediate [`Constant::Dimension`] atoms and staged
    /// [`DimensionOperation::Constant`](ryft_core::DimensionOperation::Constant) instructions must
    /// agree on this representation, and so both lowering paths route through this function.
    fn lower<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>;
}

impl DimensionValueLoweringExtension for DimensionValue {
    fn lower<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        let tensor_type = ArrayType::scalar(DataType::I64).lower(context, location)?;
        let extent = dimension_extent_i64(self.extent())?;
        let elements = DataType::I64.lower_constant_elements_attribute(tensor_type, extent, context)?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        Ok(constant.result(0)?.as_ref())
    }
}

/// Provides lowering support for [`CaptureReference`]s to StableHLO.
trait CaptureReferenceLoweringExtension {
    /// Lowers this captured constant reference by forwarding its runtime value among `captured_values`.
    fn lower<'b, 'c: 'b, 't: 'c>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;
}

impl<T: CoreType> CaptureReferenceLoweringExtension for CaptureReference<T> {
    #[inline]
    fn lower<'b, 'c: 'b, 't: 'c>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        captured_values
            .get(self.index())
            .copied()
            .ok_or(Error::Lowering(LoweringError::MissingCapturedConstant { index: self.index() }))
    }
}

impl Constant {
    /// Lowers this traced constant atom to a StableHLO value. A captured constant forwards its value from the capture
    /// table. A session constant forwards the hidden entry input or bufferless placeholder bound to its retained array
    /// in this scope. Immediate dimension extents are materialized in place, so they are available inside a
    /// `shard_map` manual region as well.
    #[inline]
    pub(crate) fn lower<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
        lowering_context: &ProgramLoweringContext,
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        match self {
            Constant::Captured(value) => value.lower(captured_values),
            Constant::Session(value) => lowering_context.lower_session_constant(value, captured_values),
            Constant::Dimension(value) => value.lower(block, context, location),
            Constant::Boolean(value) => ArrayConstant::Boolean(*value).lower_constant(block, context, location),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use ryft_core::{Complex, StridedLayout, i1, i2, i4, u1, u2, u4};
    use ryft_mlir::Size as MlirSize;
    use ryft_mlir::dialects::builtin::attributes::DenseElementsAttribute;

    use crate::lowering::tests::test_vector_type;
    use crate::sessions::tests::{domain_mesh, f32_scalar};
    use crate::tests::{cpu_client, values_to_bytes};
    use crate::{RetainedArray, Session, Type};

    use super::*;

    /// Copies the exact raw storage bytes from the MLIR dense attribute built for `literal`.
    fn literal_dense_bytes(literal: &Array, byte_count: usize) -> Vec<u8> {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = literal.r#type().as_ref().lower(&context, location).unwrap();
        let attribute = literal.lower_attribute(tensor_type, &context).unwrap();
        unsafe { std::slice::from_raw_parts(attribute.raw_data().cast::<u8>(), byte_count).to_vec() }
    }

    /// Renders the MLIR dense attribute built for `literal`, which shows the values that MLIR reads from it.
    fn literal_rendering(literal: &Array) -> String {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = literal.r#type().as_ref().lower(&context, location).unwrap();
        literal.lower_attribute(tensor_type, &context).unwrap().to_string()
    }

    #[test]
    fn test_array_lower_attribute() {
        // Every byte-aligned integer family preserves signedness, magnitude, and source order without floating-point
        // conversion. The `u64` case deliberately exceeds f64's exact-integer range.
        let integer_cases = vec![
            (Array::vector(vec![-127i8, 126]).unwrap(), values_to_bytes(&[-127i8, 126])),
            (Array::vector(vec![-0x1234i16, 0x2345]).unwrap(), values_to_bytes(&[-0x1234i16, 0x2345])),
            (
                Array::vector(vec![-0x0123_4567i32, 0x0234_5678]).unwrap(),
                values_to_bytes(&[-0x0123_4567i32, 0x0234_5678]),
            ),
            (
                Array::vector(vec![-0x0123_4567_89ab_cdefi64, 0x0234_5678_9abc_def0]).unwrap(),
                values_to_bytes(&[-0x0123_4567_89ab_cdefi64, 0x0234_5678_9abc_def0]),
            ),
            (Array::vector(vec![0x12u8, 0xfe]).unwrap(), values_to_bytes(&[0x12u8, 0xfe])),
            (Array::vector(vec![0x1234u16, 0xfedc]).unwrap(), values_to_bytes(&[0x1234u16, 0xfedc])),
            (
                Array::vector(vec![0x1234_5678u32, 0xfedc_ba98]).unwrap(),
                values_to_bytes(&[0x1234_5678u32, 0xfedc_ba98]),
            ),
            (
                Array::vector(vec![(1u64 << 53) + 1, u64::MAX - 1]).unwrap(),
                values_to_bytes(&[(1u64 << 53) + 1, u64::MAX - 1]),
            ),
        ];
        for (literal, expected) in integer_cases {
            assert_eq!(
                literal_dense_bytes(&literal, expected.len()),
                expected,
                "integer literal type {}",
                literal.r#type().as_ref(),
            );
        }

        // Standard floating-point families preserve signed zero, infinities, and NaN payload bits.
        let bf16_values = [half::bf16::from_bits(0x8000), half::bf16::from_bits(0x7fc1)];
        let f16_values = [half::f16::from_bits(0x8000), half::f16::from_bits(0x7e01)];
        let f32_values = [f32::from_bits(0x8000_0000), f32::INFINITY, f32::from_bits(0x7fc0_1234)];
        let f64_values =
            [f64::from_bits(0x8000_0000_0000_0000), f64::NEG_INFINITY, f64::from_bits(0x7ff8_0000_0000_1234)];
        for (literal, expected) in [
            (Array::vector(bf16_values.to_vec()).unwrap(), values_to_bytes(&bf16_values)),
            (Array::vector(f16_values.to_vec()).unwrap(), values_to_bytes(&f16_values)),
            (Array::vector(f32_values.to_vec()).unwrap(), values_to_bytes(&f32_values)),
            (Array::vector(f64_values.to_vec()).unwrap(), values_to_bytes(&f64_values)),
        ] {
            assert_eq!(
                literal_dense_bytes(&literal, expected.len()),
                expected,
                "floating-point literal type {}",
                literal.r#type().as_ref(),
            );
        }

        // Complex storage interleaves independently exact real and imaginary components in source order.
        let c64_components =
            [f32::from_bits(0x8000_0000), f32::from_bits(0x7fc0_1234), f32::INFINITY, f32::NEG_INFINITY];
        let c64 = Array::vector(vec![
            Complex::new(c64_components[0], c64_components[1]),
            Complex::new(c64_components[2], c64_components[3]),
        ])
        .unwrap();
        assert_eq!(literal_dense_bytes(&c64, size_of_val(&c64_components)), values_to_bytes(&c64_components));
        let c128_components = [
            f64::from_bits(0x8000_0000_0000_0000),
            f64::from_bits(0x7ff8_0000_0000_1234),
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let c128 = Array::vector(vec![
            Complex::new(c128_components[0], c128_components[1]),
            Complex::new(c128_components[2], c128_components[3]),
        ])
        .unwrap();
        assert_eq!(literal_dense_bytes(&c128, size_of_val(&c128_components)), values_to_bytes(&c128_components));

        // An empty array remains a valid dense constant without elements.
        assert_eq!(literal_rendering(&Array::vector(Vec::<i32>::new()).unwrap()), "dense<> : tensor<0xi32>");
    }

    #[test]
    fn test_array_lower_attribute_of_one_bit_values() {
        // Host arrays hold one byte for each one-bit value, of which only the low bit is its value, while MLIR packs
        // these values. Boolean values and one-bit integers of either signedness all lower to that packed `i1`
        // representation and keep their order.
        let boolean = Array::vector(vec![true, false, true]).unwrap();
        let signed = Array::vector(vec![i1::new(-1).unwrap(), i1::new(0).unwrap(), i1::new(-1).unwrap()]).unwrap();
        let unsigned = Array::vector(vec![u1::new(1).unwrap(), u1::new(0).unwrap(), u1::new(1).unwrap()]).unwrap();
        for literal in [boolean, signed, unsigned] {
            assert_eq!(
                literal_rendering(&literal),
                "dense<[true, false, true]> : tensor<3xi1>",
                "{}",
                literal.r#type().as_ref(),
            );
        }
    }

    #[test]
    fn test_array_lower_attribute_of_low_precision_values() {
        // Integers of two and four bits occupy one byte of the MLIR raw buffer for each element, which holds only
        // their declared bits.
        for (literal, expected) in [
            (Array::vector(vec![i2::new(-2).unwrap(), i2::new(1).unwrap()]).unwrap(), vec![0x02, 0x01]),
            (Array::vector(vec![i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap(), vec![0x08, 0x07]),
            (Array::vector(vec![u2::new(0).unwrap(), u2::new(3).unwrap()]).unwrap(), vec![0x00, 0x03]),
            (Array::vector(vec![u4::new(1).unwrap(), u4::new(15).unwrap()]).unwrap(), vec![0x01, 0x0f]),
        ] {
            assert_eq!(literal_dense_bytes(&literal, expected.len()), expected, "{}", literal.r#type().as_ref());
        }

        // Floating-point formats of up to eight bits keep their exact encodings, including NaN payloads, and MLIR
        // reads the values of those formats from them.
        for (data_type, bits, rendering) in [
            (DataType::F4E2M1FN, [0x01, 0x0f], "dense<[5.000000e-01, -6.000000e+00]> : tensor<2xf4E2M1FN>"),
            (DataType::F6E2M3FN, [0x01, 0x3f], "dense<[1.250000e-01, -7.500000e+00]> : tensor<2xf6E2M3FN>"),
            (DataType::F6E3M2FN, [0x02, 0x3e], "dense<[1.250000e-01, -2.400000e+01]> : tensor<2xf6E3M2FN>"),
            (DataType::F8E3M4, [0x01, 0xff], "dense<[1.562500e-02, 0xFF]> : tensor<2xf8E3M4>"),
            (DataType::F8E4M3, [0x02, 0xfe], "dense<[3.906250e-03, 0xFE]> : tensor<2xf8E4M3>"),
            (DataType::F8E4M3FN, [0x03, 0x7f], "dense<[5.859380e-03, 0x7F]> : tensor<2xf8E4M3FN>"),
            (DataType::F8E4M3FNUZ, [0x04, 0x80], "dense<[3.906250e-03, 0x80]> : tensor<2xf8E4M3FNUZ>"),
            (DataType::F8E4M3B11FNUZ, [0x05, 0x80], "dense<[6.103520e-04, 0x80]> : tensor<2xf8E4M3B11FNUZ>"),
            (DataType::F8E5M2, [0x06, 0x7f], "dense<[9.155270e-05, 0x7F]> : tensor<2xf8E5M2>"),
            (DataType::F8E5M2FNUZ, [0x07, 0x80], "dense<[5.340580e-05, 0x80]> : tensor<2xf8E5M2FNUZ>"),
            (DataType::F8E8M0FNU, [0x08, 0xff], "dense<[1.504630e-36, 0xFF]> : tensor<2xf8E8M0FNU>"),
        ] {
            let literal = Array::from_logical_bytes(ArrayType::new_static(data_type, [2]), &bits).unwrap();
            assert_eq!(literal_dense_bytes(&literal, bits.len()), bits, "{data_type}");
            assert_eq!(literal_rendering(&literal), rendering, "{data_type}");
        }
    }

    #[test]
    fn test_array_lower_attribute_follows_the_logical_order_of_layouts() {
        // An array with an explicit physical layout is traversed in logical row-major order, which its storage order
        // differs from.
        let layout_type =
            ArrayType::new_static(DataType::I16, [2, 2]).with_layout(Some(StridedLayout::new(vec![2, 4]).into()));
        let literal = Array::from_elements(layout_type, &[1i16, 2, 3, 4]).unwrap();
        assert_eq!(literal.storage_bytes(), values_to_bytes(&[1i16, 3, 2, 4]));
        assert_eq!(literal_dense_bytes(&literal, 8), values_to_bytes(&[1i16, 2, 3, 4]));
    }

    #[test]
    fn test_array_lower_attribute_of_zeros() {
        // An array of the zero data type has no payload bytes, and it lowers to its `false` carrier of one bit.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = context
            .tensor_type(context.signless_integer_type(1), &[MlirSize::Static(2)], None, location)
            .unwrap();
        let zero = Array::from_logical_bytes(ArrayType::new_static(DataType::Zero, [2]), &[]).unwrap();
        assert_eq!(zero.lower_attribute(tensor_type, &context).unwrap().to_string(), "dense<false> : tensor<2xi1>");
    }

    #[test]
    fn test_array_lower_attribute_rejects_tokens() {
        // Tokens are effect values, which have no dense literal representation.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = context
            .tensor_type(context.signless_integer_type(1), &[MlirSize::Static(2)], None, location)
            .unwrap();
        let token = Array::from_logical_bytes(ArrayType::new_static(DataType::Token, [2]), &[]).unwrap();
        assert_eq!(
            token.lower_attribute(tensor_type, &context).err(),
            Some(Error::Lowering(LoweringError::UnsupportedElementType {
                data_type: DataType::Token,
                context: "dense literal materialization".to_string(),
            })),
        );
    }

    #[test]
    fn test_array_constant_lower_attribute() {
        // Only a Boolean constant is a literal. A captured constant and a session constant are values of the function
        // that they are lowered in, so neither has elements to embed.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let boolean_type = ArrayType::scalar(DataType::Boolean).lower(&context, location).unwrap();
        let boolean = ArrayConstant::Boolean(true).lower_attribute(boolean_type, &context).unwrap();
        assert_eq!(boolean.to_string(), "dense<true> : tensor<i1>");

        let array_type = ArrayType::scalar(DataType::F32);
        let tensor_type = array_type.lower(&context, location).unwrap();
        let captured = ArrayConstant::Captured(CaptureReference::new(3, array_type));
        let message = "captured constant #3 cannot be materialized as a literal without capture bindings".to_string();
        assert_eq!(
            captured.lower_attribute(tensor_type, &context).err(),
            Some(Error::from(ProgramError::Concretization { message })),
        );

        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let retained = RetainedArray::new(&f32_scalar(&session, &domain_mesh(client, "x", 1), 2.0));
        let message = format!("session constant `{retained}` cannot be materialized as a literal");
        assert_eq!(
            ArrayConstant::Session(retained).lower_attribute(tensor_type, &context).err(),
            Some(Error::from(ProgramError::Concretization { message })),
        );
    }

    #[test]
    fn test_array_lower_constant() {
        // The constants are rendered outside any function. A literal that is placed in host memory is annotated with
        // that placement, while one in device memory is not.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block_with_no_arguments();
        let mut block_ref = block.as_ref();
        let device = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let host_type = ArrayType::new_static(DataType::F32, [2]).with_memory(Memory::Host { pinned: true });
        let host = Array::from_elements(host_type, &[3.0f32, 4.0]).unwrap();
        let device = device.lower_constant(&mut block_ref, &context, location).unwrap();
        let host = host.lower_constant(&mut block_ref, &context, location).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(device, operations[0].result(0).unwrap().as_ref());
        assert_eq!(host, operations[2].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%cst = stablehlo.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>",
                "%cst = stablehlo.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>",
                "%0 = stablehlo.custom_call @annotate_device_placement(<<UNKNOWN SSA VALUE>>) {backend_config = \"\", \
                 has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = \"pinned_host\"}} : \
                 (tensor<2xf32>) -> tensor<2xf32>",
            ],
        );
    }

    #[test]
    fn test_dimension_value_lower() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block_with_no_arguments();
        let output = DimensionValue::constant(3).unwrap().lower(&mut block.as_ref(), &context, location).unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations[0].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec!["%c = stablehlo.constant dense<3> : tensor<i64>"],
        );
    }

    #[test]
    fn test_capture_reference_lower() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = test_vector_type(2).lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location), (tensor_type, location)]);
        let captured_values = [block.argument(0).unwrap().as_ref(), block.argument(1).unwrap().as_ref()];
        let reference = CaptureReference::new(1, test_vector_type(2));
        assert_eq!(reference.lower(&captured_values), Ok(captured_values[1]));

        // A reference beyond the captures of the function that it is lowered in has no value there.
        let reference = CaptureReference::new(2, test_vector_type(2));
        assert_eq!(
            reference.lower(&captured_values),
            Err(Error::Lowering(LoweringError::MissingCapturedConstant { index: 2 })),
        );
    }

    #[test]
    fn test_constant_lower() {
        // A captured constant forwards its capture input, and a session constant forwards the hidden input that the
        // lowering context assigns to it, which follows the captures. A dimension extent and a Boolean predicate are
        // materialized in place.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let retained = RetainedArray::new(&f32_scalar(&session, &domain_mesh(client, "x", 1), 2.0));
        let context = MlirContext::new();
        let location = context.unknown_location();
        let vector_type = test_vector_type(2).lower(&context, location).unwrap();
        let scalar_type = ArrayType::scalar(DataType::F32).lower(&context, location).unwrap();
        let block = context.block(&[(vector_type, location), (scalar_type, location)]);
        let captured_values = [block.argument(0).unwrap().as_ref(), block.argument(1).unwrap().as_ref()];
        let lowering_context = ProgramLoweringContext::new().with_session_constants(std::slice::from_ref(&retained));
        let mut block_ref = block.as_ref();
        let mut lower = |constant: Constant| {
            constant.lower(&captured_values, &lowering_context, &mut block_ref, &context, location).unwrap()
        };
        let captured = lower(Constant::Captured(CaptureReference::new(0, Type::Array(test_vector_type(2)))));
        let retained = lower(Constant::Session(retained));
        let dimension = lower(Constant::Dimension(DimensionValue::constant(3).unwrap()));
        let boolean = lower(Constant::Boolean(true));
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(captured, captured_values[0]);
        assert_eq!(retained, captured_values[1]);
        assert_eq!(dimension, operations[0].result(0).unwrap().as_ref());
        assert_eq!(boolean, operations[1].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec!["%c = stablehlo.constant dense<3> : tensor<i64>", "%c = stablehlo.constant dense<true> : tensor<i1>"],
        );
    }
}
