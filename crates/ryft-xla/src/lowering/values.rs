//! Lowering of the values of programs (i.e., their constants) to StableHLO values. A value is either materialized
//! as a literal (refer to [`MlirLowerableValue`]) or forwarded from a hidden input of the function that it is lowered
//! in (refer to [`Constant::lower`]).

use std::borrow::Cow;

use ryft_core::{
    Array, ArrayType, CaptureReference, DataType, DimensionValue, ProgramError, Type as CoreType, Typed, Value,
};
use ryft_mlir::dialects::stable_hlo;
use ryft_mlir::{
    Attribute, Block, Context as MlirContext, DenseElementsAttributeRef, Location, Operation as MlirOperation,
    Value as MlirValue, ValueRef,
};

use crate::lowering::LoweringError;
use crate::lowering::states::ProgramLoweringContext;
use crate::lowering::types::{
    ArrayTypeLoweringExtension, DataTypeLoweringExtension, annotate_output_memory, dimension_extent_i64,
};
use crate::{ArrayConstant, Constant, Error};

// TODO(eaplatanios): Review this module.

/// Value type that lowering can materialize as a StableHLO dense constant or bind to a captured input (e.g., the
/// values of [`ArrayOperation`](ryft_core::ArrayOperation) constants).
pub(crate) trait MlirLowerableValue: 'static + Value<Type = ArrayType> {
    /// Builds a dense-elements attribute containing this value.
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error>;

    /// Lowers this concrete value to a StableHLO constant operation and applies its declared memory placement.
    fn lower_literal<'b, 'c: 'b, 't: 'c, B, L>(
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
        let elements = self.to_dense_elements_attribute(tensor_type, context)?;
        let constant = block.append_operation(stable_hlo::constant(elements, location)?)?;
        let constant = constant.result(0).unwrap().as_ref();
        annotate_output_memory(constant, &value_type, block, context, location)
    }

    /// Lowers this program constant inside a function body.
    #[inline]
    fn lower_constant_value<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        let _ = captured_values;
        self.lower_literal(block, context, location)
    }
}

impl MlirLowerableValue for ArrayConstant {
    fn to_dense_elements_attribute<'c, 't>(
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
            Self::Boolean(value) => Array::scalar(*value)?.to_dense_elements_attribute(tensor_type, context),
        }
    }

    fn lower_constant_value<'b, 'c: 'b, 't: 'c, B, L>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
        block: &mut B,
        context: &'c MlirContext<'t>,
        location: L,
    ) -> Result<ValueRef<'b, 'c, 't>, Error>
    where
        B: Block<'b, 'c, 't>,
        L: Copy + Location<'c, 't>,
    {
        match self {
            Self::Captured(value) => value.lower(captured_values),
            Self::Session(value) => {
                Err(Error::Lowering(LoweringError::UnreachableSessionConstant { constant: value.to_string() }))
            }
            Self::Boolean(_) => self.lower_literal(block, context, location),
        }
    }
}

// `ArrayType` is used as the value representation for abstract linear XLA programs. It can type program atoms, but it
// is not a concrete literal, so lowering paths that need a real value must supply it through captured inputs instead
// of materializing it from type metadata.
impl MlirLowerableValue for ArrayType {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
        let _ = (tensor_type, context);
        Err(ProgramError::Concretization {
            message: format!("abstract XLA value `{self}` cannot be materialized as a StableHLO literal"),
        }
        .into())
    }
}

// Concrete host literal lowering used by [`ConstantOperation`] and MLIR snapshot tooling.
impl MlirLowerableValue for Array {
    fn to_dense_elements_attribute<'c, 't>(
        &self,
        tensor_type: ryft_mlir::TensorTypeRef<'c, 't>,
        context: &'c MlirContext<'t>,
    ) -> Result<DenseElementsAttributeRef<'c, 't>, Error> {
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

/// Lowering of a [`CaptureReference`] to StableHLO.
trait CaptureReferenceLoweringExtension {
    /// Lowers this captured constant reference by forwarding its runtime value among `captured_values`.
    fn lower<'b, 'c: 'b, 't: 'c>(
        &self,
        captured_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<ValueRef<'b, 'c, 't>, Error>;
}

impl<T: CoreType> CaptureReferenceLoweringExtension for CaptureReference<T> {
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

/// Lowering of a [`DimensionValue`] to StableHLO.
pub(crate) trait DimensionValueLoweringExtension {
    /// Lowers one first-class dimension extent to the scalar `i64` [`stablehlo.constant`](stable_hlo::constant) that
    /// represents a dimension in StableHLO. Immediate [`Constant::Dimension`] atoms and staged
    /// [`DimensionOperation::Constant`](ryft_core::DimensionOperation::Constant) instructions must agree on this
    /// representation, and so both lowering paths route through this function.
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
        Ok(constant.result(0).unwrap().as_ref())
    }
}

impl Constant {
    /// Lowers this traced constant atom to a StableHLO value and returns it. A captured constant forwards the hidden
    /// capture input that carries its runtime value, and a session constant forwards the hidden entry input that
    /// `lowering_context` assigns to it, while an immediate dimension extent is materialized in place. The latter is
    /// exactly what makes extents usable inside a `shard_map` manual region, where no capture table is reachable.
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
            Constant::Boolean(value) => ArrayConstant::Boolean(*value).lower_literal(block, context, location),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayOperation, Complex, ConstantOperation, Dimension, Error as CoreError, Memory, Placeholder,
        ProgramBuilder as CoreProgramBuilder, Shape, StridedLayout, i1, i2, i4, u1, u2, u4,
    };
    use ryft_mlir::dialects::builtin::attributes::DenseElementsAttribute;
    use ryft_mlir::{ElementsAttribute, Size as MlirSize};

    use crate::Type;
    use crate::lowering::tests::{test_vector_type, to_rendered_mlir_for_plain_program};
    use crate::tests::values_to_bytes;

    use super::*;

    /// Copies the exact raw storage bytes from the MLIR dense attribute built for `literal`.
    fn literal_dense_bytes(literal: &Array, byte_count: usize) -> Vec<u8> {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = literal.r#type().as_ref().lower(&context, location).unwrap();
        let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
        unsafe { std::slice::from_raw_parts(attribute.raw_data().cast::<u8>(), byte_count).to_vec() }
    }

    #[test]
    fn test_mlir_lowerable_value_to_dense_elements_attribute() {
        // Boolean values retain their logical order through MLIR's bit-packed dense representation.
        let boolean = Array::vector(vec![true, false, true]).unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = boolean.r#type().as_ref().lower(&context, location).unwrap();
        let attribute = boolean.to_dense_elements_attribute(tensor_type, &context).unwrap();
        assert_eq!(
            unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() },
            vec![true, false, true],
        );

        // One-bit integers use the same packed MLIR representation as Boolean while retaining their Ryft data type.
        for (literal, expected) in [
            (
                Array::vector(vec![i1::new(-1).unwrap(), i1::new(0).unwrap(), i1::new(-1).unwrap()]).unwrap(),
                vec![true, false, true],
            ),
            (
                Array::vector(vec![u1::new(1).unwrap(), u1::new(0).unwrap(), u1::new(1).unwrap()]).unwrap(),
                vec![true, false, true],
            ),
        ] {
            let tensor_type = literal.r#type().as_ref().lower(&context, location).unwrap();
            let attribute = literal.to_dense_elements_attribute(tensor_type, &context).unwrap();
            assert_eq!(unsafe { attribute.bool_elements().collect::<Result<Vec<_>, _>>().unwrap() }, expected);
        }

        // Wider sub-byte integers occupy one MLIR raw-buffer byte per element and preserve only their declared bits.
        for (literal, expected) in [
            (Array::vector(vec![i2::new(-2).unwrap(), i2::new(1).unwrap()]).unwrap(), vec![0x02, 0x01]),
            (Array::vector(vec![i4::new(-8).unwrap(), i4::new(7).unwrap()]).unwrap(), vec![0x08, 0x07]),
            (Array::vector(vec![u2::new(0).unwrap(), u2::new(3).unwrap()]).unwrap(), vec![0x00, 0x03]),
            (Array::vector(vec![u4::new(1).unwrap(), u4::new(15).unwrap()]).unwrap(), vec![0x01, 0x0f]),
        ] {
            assert_eq!(literal_dense_bytes(&literal, expected.len()), expected);
        }

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

        // Sub-byte and eight-bit floating-point formats retain their exact encodings, including NaN payloads.
        for (data_type, bits, rendered_values) in [
            (DataType::F4E2M1FN, [0x01, 0x0f], "dense<[5.000000e-01, -6.000000e+00]>"),
            (DataType::F6E2M3FN, [0x01, 0x3f], "dense<[1.250000e-01, -7.500000e+00]>"),
            (DataType::F6E3M2FN, [0x02, 0x3e], "dense<[1.250000e-01, -2.400000e+01]>"),
            (DataType::F8E3M4, [0x01, 0xff], "dense<[1.562500e-02, 0xFF]>"),
            (DataType::F8E4M3, [0x02, 0xfe], "dense<[3.906250e-03, 0xFE]>"),
            (DataType::F8E4M3FN, [0x03, 0x7f], "dense<[5.859380e-03, 0x7F]>"),
            (DataType::F8E4M3FNUZ, [0x04, 0x80], "dense<[3.906250e-03, 0x80]>"),
            (DataType::F8E4M3B11FNUZ, [0x05, 0x80], "dense<[6.103520e-04, 0x80]>"),
            (DataType::F8E5M2, [0x06, 0x7f], "dense<[9.155270e-05, 0x7F]>"),
            (DataType::F8E5M2FNUZ, [0x07, 0x80], "dense<[5.340580e-05, 0x80]>"),
            (DataType::F8E8M0FNU, [0x08, 0xff], "dense<[1.504630e-36, 0xFF]>"),
        ] {
            let literal = Array::from_logical_bytes(
                ArrayType::new(data_type, Shape::new(vec![Dimension::Static(bits.len())])),
                &bits,
            )
            .unwrap();
            assert_eq!(literal_dense_bytes(&literal, bits.len()), bits, "low-precision literal type {data_type}");
            let mut builder = CoreProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let output = builder
                .add_instruction(
                    ArrayOperation::Constant(ConstantOperation::new(literal)),
                    Vec::new(),
                    Vec::new(),
                    None,
                )
                .unwrap()[0];
            let program = builder.build::<Vec<Array>, Vec<Array>>(vec![output], Vec::new(), vec![Placeholder]).unwrap();
            let stablehlo = to_rendered_mlir_for_plain_program(&program, "main").unwrap();
            assert!(stablehlo.contains(rendered_values), "low-precision literal {data_type}: {stablehlo}");
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

        // Explicit physical layouts are traversed in logical row-major order before constructing the literal.
        let layout_type = ArrayType::new(DataType::I16, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
            .with_layout(Some(StridedLayout::new(vec![2, 4]).into()));
        let layout_literal = Array::from_elements(layout_type, &[1i16, 2, 3, 4]).unwrap();
        assert_eq!(layout_literal.storage_bytes(), values_to_bytes(&[1i16, 3, 2, 4]));
        assert_eq!(literal_dense_bytes(&layout_literal, 8), values_to_bytes(&[1i16, 2, 3, 4]));

        // Empty tensors remain valid dense constants and carry no raw payload bytes.
        let empty = Array::vector(Vec::<i32>::new()).unwrap();
        let tensor_type = empty.r#type().as_ref().lower(&context, location).unwrap();
        let attribute = empty.to_dense_elements_attribute(tensor_type, &context).unwrap();
        assert_eq!(attribute.elements_count(), 0);

        // Zero literals use the established false `i1` carrier without introducing logical payload bytes.
        let tensor_type = context
            .tensor_type(context.signless_integer_type(1), &[MlirSize::Static(2)], None, location)
            .unwrap();
        let zero = Array::from_logical_bytes(ArrayType::new_static(DataType::Zero, [2]), &[]).unwrap();
        assert_eq!(
            zero.to_dense_elements_attribute(tensor_type, &context).unwrap().to_string(),
            "dense<false> : tensor<2xi1>",
        );

        // Tokens remain effect values and have no dense literal representation.
        let token = Array::from_logical_bytes(ArrayType::new_static(DataType::Token, [2]), &[]).unwrap();
        assert!(matches!(
            token.to_dense_elements_attribute(tensor_type, &context),
            Err(Error::Lowering(LoweringError::UnsupportedElementType { data_type: DataType::Token, .. })),
        ));
    }

    #[test]
    fn test_mlir_lowerable_value_to_dense_elements_attribute_rejects_symbolic_values() {
        let context = MlirContext::new();
        let array_type = ArrayType::scalar(DataType::F32);
        let tensor_type = array_type.lower(&context, context.unknown_location()).unwrap();
        assert!(matches!(
            array_type.to_dense_elements_attribute(tensor_type, &context),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::Concretization { message })
                    if *message == "abstract XLA value `f32[]` cannot be materialized as a StableHLO literal",
            ),
        ));
        let captured = ArrayConstant::Captured(CaptureReference::new(3, array_type));
        assert!(matches!(
            captured.to_dense_elements_attribute(tensor_type, &context),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::Concretization { message })
                    if *message == "captured constant #3 cannot be materialized as a literal without capture bindings",
            ),
        ));
    }

    #[test]
    fn test_mlir_lowerable_value_lower_literal() {
        // A literal that is placed in host memory is annotated with that placement.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let block = context.block_with_no_arguments();
        let mut block_ref = block.as_ref();
        let device = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let host_type = ArrayType::new_static(DataType::F32, [2]).with_memory(Memory::Host { pinned: true });
        let host = Array::from_elements(host_type, &[3.0f32, 4.0]).unwrap();
        let device = device.lower_literal(&mut block_ref, &context, location).unwrap();
        let host = host.lower_literal(&mut block_ref, &context, location).unwrap();
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
    fn test_capture_reference_lower() {
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = test_vector_type(2).lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location), (tensor_type, location)]);
        let captured_values = [block.argument(0).unwrap().as_ref(), block.argument(1).unwrap().as_ref()];
        let reference = CaptureReference::new(1, test_vector_type(2));
        assert_eq!(reference.lower(&captured_values).unwrap(), captured_values[1]);
        let reference = CaptureReference::new(2, test_vector_type(2));
        assert!(matches!(
            reference.lower(&captured_values),
            Err(Error::Lowering(LoweringError::MissingCapturedConstant { index: 2 })),
        ));
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
    fn test_constant_lower() {
        // A captured constant forwards its capture input, while dimension extents and Boolean predicates are
        // materialized in place.
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = test_vector_type(2).lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let captured_values = [block.argument(0).unwrap().as_ref()];
        let lowering_context = ProgramLoweringContext::new();
        let mut block_ref = block.as_ref();
        let mut lower = |constant: Constant| {
            constant.lower(&captured_values, &lowering_context, &mut block_ref, &context, location).unwrap()
        };
        let captured = lower(Constant::Captured(CaptureReference::new(0, Type::Array(test_vector_type(2)))));
        let dimension = lower(Constant::Dimension(DimensionValue::constant(3).unwrap()));
        let boolean = lower(Constant::Boolean(true));
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(captured, captured_values[0]);
        assert_eq!(dimension, operations[0].result(0).unwrap().as_ref());
        assert_eq!(boolean, operations[1].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec!["%c = stablehlo.constant dense<3> : tensor<i64>", "%c = stablehlo.constant dense<true> : tensor<i1>"],
        );
    }
}
