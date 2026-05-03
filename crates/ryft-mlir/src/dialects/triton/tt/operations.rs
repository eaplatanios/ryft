use crate::{
    ArrayAttributeRef, Attribute, BooleanAttributeRef, DenseInteger32ArrayAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, FlatSymbolRefAttributeRef, IntegerAttributeRef, Location, Operation, OperationBuilder,
    OperationResultRef, RegionRef, StringAttributeRef, TypeAttributeRef, TypeRef, ValueRef, mlir_op, mlir_op_trait,
};

use super::attributes::{
    CacheModifier, CacheModifierAttributeRef, DescriptorReduceKind, DescriptorReduceKindAttributeRef, EvictionPolicy,
    EvictionPolicyAttributeRef, InputPrecision, InputPrecisionAttributeRef, MemSemantic, MemSemanticAttributeRef,
    MemSyncScope, MemSyncScopeAttributeRef, PaddingOption, PaddingOptionAttributeRef, ProgramIdDim,
    ProgramIdDimAttributeRef, PropagateNan, PropagateNanAttributeRef, RmwOp, RmwOpAttributeRef, RoundingMode,
    RoundingModeAttributeRef, ScaleDotElemType, ScaleDotElemTypeAttributeRef,
};

/// Triton `tt` [`Operation`] that casts an integer value to a pointer value.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait IntToPtrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the integer source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the pointer result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(IntToPtr);
mlir_op_trait!(IntToPtr, OneOperand);
mlir_op_trait!(IntToPtr, OneResult);
mlir_op_trait!(IntToPtr, ZeroRegions);
mlir_op_trait!(IntToPtr, ZeroSuccessors);

/// Constructs a new detached/owned [`IntToPtrOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn int_to_ptr<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedIntToPtrOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.int_to_ptr", location)
        .add_operand(src)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::int_to_ptr`")
}

/// Triton `tt` [`Operation`] that casts a pointer value to an integer value.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait PtrToIntOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the pointer source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the integer result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(PtrToInt);
mlir_op_trait!(PtrToInt, OneOperand);
mlir_op_trait!(PtrToInt, OneResult);
mlir_op_trait!(PtrToInt, ZeroRegions);
mlir_op_trait!(PtrToInt, ZeroSuccessors);

/// Constructs a new detached/owned [`PtrToIntOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn ptr_to_int<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedPtrToIntOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.ptr_to_int", location)
        .add_operand(src)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::ptr_to_int`")
}

/// Triton `tt` [`Operation`] that casts between values with the same bit width.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait BitcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the bitcast result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Bitcast);
mlir_op_trait!(Bitcast, OneOperand);
mlir_op_trait!(Bitcast, OneResult);
mlir_op_trait!(Bitcast, ZeroRegions);
mlir_op_trait!(Bitcast, ZeroSuccessors);

/// Constructs a new detached/owned [`BitcastOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn bitcast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedBitcastOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.bitcast", location)
        .add_operand(src)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::bitcast`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` floating-point rounding mode.
pub const ROUNDING_ATTRIBUTE: &str = "rounding";

/// Triton `tt` [`Operation`] that casts between floating-point values using an optional rounding mode.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait FpToFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional rounding mode.
    fn rounding(&self) -> Option<RoundingModeAttributeRef<'c, 't>> {
        self.attribute(ROUNDING_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<RoundingModeAttributeRef>())
    }

    /// Returns the cast result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(FpToFp);
mlir_op_trait!(FpToFp, OneOperand);
mlir_op_trait!(FpToFp, OneResult);
mlir_op_trait!(FpToFp, ZeroRegions);
mlir_op_trait!(FpToFp, ZeroSuccessors);

/// Constructs a new detached/owned [`FpToFpOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn fp_to_fp<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    rounding: Option<RoundingMode>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedFpToFpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    let mut builder = OperationBuilder::new("tt.fp_to_fp", location).add_operand(src);
    if let Some(rounding) = rounding {
        builder = builder.add_attribute(ROUNDING_ATTRIBUTE, context.triton_tt_rounding_mode_attribute(rounding));
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::fp_to_fp`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` NaN propagation mode.
pub const PROPAGATE_NAN_ATTRIBUTE: &str = "propagateNan";

/// Triton `tt` [`Operation`] that clamps floating-point values to a minimum and maximum bound.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ClampFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to clamp.
    fn x(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the minimum bound.
    fn min(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the maximum bound.
    fn max(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the NaN propagation mode.
    fn propagate_nan(&self) -> PropagateNanAttributeRef<'c, 't> {
        self.attribute(PROPAGATE_NAN_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", PROPAGATE_NAN_ATTRIBUTE, "tt.clampf"))
            .cast::<PropagateNanAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", PROPAGATE_NAN_ATTRIBUTE, "tt.clampf"))
    }

    /// Returns the clamped result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(ClampF);
mlir_op_trait!(ClampF, OneResult);
mlir_op_trait!(ClampF, ZeroRegions);
mlir_op_trait!(ClampF, ZeroSuccessors);

/// Constructs a new detached/owned [`ClampFOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn clampf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    x: ValueRef<'v, 'c, 't>,
    min: ValueRef<'v, 'c, 't>,
    max: ValueRef<'v, 'c, 't>,
    propagate_nan: PropagateNan,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedClampFOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.clampf", location)
        .add_operand(x)
        .add_operand(min)
        .add_operand(max)
        .add_attribute(PROPAGATE_NAN_ATTRIBUTE, context.triton_tt_propagate_nan_attribute(propagate_nan))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::clampf`")
}

/// Triton `tt` [`Operation`] that computes a precise floating-point square root.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait PreciseSqrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input value.
    fn x(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the square-root result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(PreciseSqrt);
mlir_op_trait!(PreciseSqrt, OneOperand);
mlir_op_trait!(PreciseSqrt, OneResult);
mlir_op_trait!(PreciseSqrt, ZeroRegions);
mlir_op_trait!(PreciseSqrt, ZeroSuccessors);

/// Constructs a new detached/owned [`PreciseSqrtOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn precise_sqrt<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    x: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedPreciseSqrtOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.precise_sqrt", location)
        .add_operand(x)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::precise_sqrt`")
}

/// Triton `tt` [`Operation`] that computes a precise floating-point division.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait PreciseDivFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dividend.
    fn x(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the divisor.
    fn y(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the division result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(PreciseDivF);
mlir_op_trait!(PreciseDivF, OneResult);
mlir_op_trait!(PreciseDivF, ZeroRegions);
mlir_op_trait!(PreciseDivF, ZeroSuccessors);

/// Constructs a new detached/owned [`PreciseDivFOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn precise_divf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    x: ValueRef<'v, 'c, 't>,
    y: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedPreciseDivFOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.precise_divf", location)
        .add_operand(x)
        .add_operand(y)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::precise_divf`")
}

/// Triton `tt` [`Operation`] that returns the high bits of an unsigned integer product.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait MulhiUIOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side input.
    fn x(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right-hand-side input.
    fn y(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the high-product result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MulhiUI);
mlir_op_trait!(MulhiUI, OneResult);
mlir_op_trait!(MulhiUI, ZeroRegions);
mlir_op_trait!(MulhiUI, ZeroSuccessors);

/// Constructs a new detached/owned [`MulhiUIOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn mulhiui<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    x: ValueRef<'v, 'c, 't>,
    y: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMulhiUIOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.mulhiui", location)
        .add_operand(x)
        .add_operand(y)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::mulhiui`")
}

/// Triton `tt` [`Operation`] that adds an integer offset to a pointer.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait AddPtrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base pointer.
    fn ptr(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the integer offset.
    fn offset(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the adjusted pointer.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(AddPtr);
mlir_op_trait!(AddPtr, OneResult);
mlir_op_trait!(AddPtr, ZeroRegions);
mlir_op_trait!(AddPtr, ZeroSuccessors);

/// Constructs a new detached/owned [`AddPtrOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn addptr<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    ptr: ValueRef<'v, 'c, 't>,
    offset: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedAddPtrOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.addptr", location)
        .add_operand(ptr)
        .add_operand(offset)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::addptr`")
}

/// Name of the [`Attribute`] that stores Triton `tt` operand segment sizes.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// Name of the [`Attribute`] that stores a Triton `tt` cache modifier.
pub const CACHE_ATTRIBUTE: &str = "cache";

/// Name of the [`Attribute`] that stores a Triton `tt` eviction policy.
pub const EVICT_ATTRIBUTE: &str = "evict";

/// Name of the [`Attribute`] that marks a Triton `tt` load as volatile.
pub const IS_VOLATILE_ATTRIBUTE: &str = "isVolatile";

/// Triton `tt` [`Operation`] that loads from a pointer or tensor of pointers.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the pointer operand.
    fn ptr(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional mask operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes: Vec<i32> = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
            .unwrap_or_default();
        if sizes.get(1).copied().unwrap_or(0) > 0 { self.operand_value(1) } else { None }
    }

    /// Returns the optional fallback value operand.
    fn other(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes: Vec<i32> = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
            .unwrap_or_default();
        if sizes.get(2).copied().unwrap_or(0) > 0 {
            let index = 1 + sizes.get(1).copied().unwrap_or(0) as usize;
            self.operand_value(index)
        } else {
            None
        }
    }

    /// Returns the cache modifier.
    fn cache(&self) -> CacheModifierAttributeRef<'c, 't> {
        self.attribute(CACHE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", CACHE_ATTRIBUTE, "tt.load"))
            .cast::<CacheModifierAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", CACHE_ATTRIBUTE, "tt.load"))
    }

    /// Returns the eviction policy.
    fn evict(&self) -> EvictionPolicyAttributeRef<'c, 't> {
        self.attribute(EVICT_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", EVICT_ATTRIBUTE, "tt.load"))
            .cast::<EvictionPolicyAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", EVICT_ATTRIBUTE, "tt.load"))
    }

    /// Returns `true` if this load is volatile.
    fn is_volatile(&self) -> bool {
        self.attribute(IS_VOLATILE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", IS_VOLATILE_ATTRIBUTE, "tt.load"))
            .cast::<BooleanAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", IS_VOLATILE_ATTRIBUTE, "tt.load"))
            .value()
    }

    /// Returns the loaded value.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Load);
mlir_op_trait!(Load, OneResult);
mlir_op_trait!(Load, ZeroRegions);
mlir_op_trait!(Load, ZeroSuccessors);

/// Constructs a new detached/owned [`LoadOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    ptr: ValueRef<'v, 'c, 't>,
    mask: Option<ValueRef<'v, 'c, 't>>,
    other: Option<ValueRef<'v, 'c, 't>>,
    cache: CacheModifier,
    evict: EvictionPolicy,
    is_volatile: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    let mut builder = OperationBuilder::new("tt.load", location).add_operand(ptr);
    if let Some(mask) = mask {
        builder = builder.add_operand(mask);
    }
    if let Some(other) = other {
        builder = builder.add_operand(other);
    }
    builder
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[1, i32::from(mask.is_some()), i32::from(other.is_some())])
                .unwrap(),
        )
        .add_attribute(CACHE_ATTRIBUTE, context.triton_tt_cache_modifier_attribute(cache))
        .add_attribute(EVICT_ATTRIBUTE, context.triton_tt_eviction_policy_attribute(evict))
        .add_attribute(IS_VOLATILE_ATTRIBUTE, context.boolean_attribute(is_volatile))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::load`")
}

/// Name of the [`Attribute`] that marks a Triton `tt` store as ignoring CTA behavior.
pub const IGNORE_CTA_ATTRIBUTE: &str = "ignore_cta";

/// Triton `tt` [`Operation`] that stores through a pointer or tensor of pointers.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait StoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the pointer operand.
    fn ptr(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the value to store.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional mask operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes: Vec<i32> = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
            .unwrap_or_default();
        if sizes.get(2).copied().unwrap_or(0) > 0 { self.operand_value(2) } else { None }
    }

    /// Returns the cache modifier.
    fn cache(&self) -> CacheModifierAttributeRef<'c, 't> {
        self.attribute(CACHE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", CACHE_ATTRIBUTE, "tt.store"))
            .cast::<CacheModifierAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", CACHE_ATTRIBUTE, "tt.store"))
    }

    /// Returns the eviction policy.
    fn evict(&self) -> EvictionPolicyAttributeRef<'c, 't> {
        self.attribute(EVICT_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", EVICT_ATTRIBUTE, "tt.store"))
            .cast::<EvictionPolicyAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", EVICT_ATTRIBUTE, "tt.store"))
    }

    /// Returns `true` if this store is marked to ignore CTA behavior.
    fn ignore_cta(&self) -> bool {
        self.has_attribute(IGNORE_CTA_ATTRIBUTE)
    }
}

mlir_op!(Store);
mlir_op_trait!(Store, ZeroRegions);
mlir_op_trait!(Store, ZeroSuccessors);

/// Constructs a new detached/owned [`StoreOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    ptr: ValueRef<'v, 'c, 't>,
    value: ValueRef<'v, 'c, 't>,
    mask: Option<ValueRef<'v, 'c, 't>>,
    cache: CacheModifier,
    evict: EvictionPolicy,
    ignore_cta: bool,
    location: L,
) -> DetachedStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    let mut builder = OperationBuilder::new("tt.store", location).add_operand(ptr).add_operand(value);
    if let Some(mask) = mask {
        builder = builder.add_operand(mask);
    }
    builder = builder
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[1, 1, i32::from(mask.is_some())]).unwrap(),
        )
        .add_attribute(CACHE_ATTRIBUTE, context.triton_tt_cache_modifier_attribute(cache))
        .add_attribute(EVICT_ATTRIBUTE, context.triton_tt_eviction_policy_attribute(evict));
    if ignore_cta {
        builder = builder.add_attribute(IGNORE_CTA_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::store`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` atomic read-modify-write operation.
pub const ATOMIC_RMW_OP_ATTRIBUTE: &str = "atomic_rmw_op";

/// Name of the [`Attribute`] that stores a Triton `tt` memory semantic.
pub const SEM_ATTRIBUTE: &str = "sem";

/// Name of the [`Attribute`] that stores a Triton `tt` memory synchronization scope.
pub const SCOPE_ATTRIBUTE: &str = "scope";

/// Triton `tt` [`Operation`] that performs an atomic read-modify-write operation.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait AtomicRmwOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the atomic read-modify-write operation kind.
    fn atomic_rmw_op(&self) -> RmwOpAttributeRef<'c, 't> {
        self.attribute(ATOMIC_RMW_OP_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", ATOMIC_RMW_OP_ATTRIBUTE, "tt.atomic_rmw"))
            .cast::<RmwOpAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", ATOMIC_RMW_OP_ATTRIBUTE, "tt.atomic_rmw"))
    }

    /// Returns the pointer operand.
    fn ptr(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the value operand.
    fn val(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional mask operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes: Vec<i32> = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
            .unwrap_or_default();
        if sizes.get(2).copied().unwrap_or(0) > 0 { self.operand_value(2) } else { None }
    }

    /// Returns the memory semantic.
    fn sem(&self) -> MemSemanticAttributeRef<'c, 't> {
        self.attribute(SEM_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", SEM_ATTRIBUTE, "tt.atomic_rmw"))
            .cast::<MemSemanticAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", SEM_ATTRIBUTE, "tt.atomic_rmw"))
    }

    /// Returns the memory synchronization scope.
    fn scope(&self) -> MemSyncScopeAttributeRef<'c, 't> {
        self.attribute(SCOPE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", SCOPE_ATTRIBUTE, "tt.atomic_rmw"))
            .cast::<MemSyncScopeAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", SCOPE_ATTRIBUTE, "tt.atomic_rmw"))
    }

    /// Returns the old value read from memory.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(AtomicRmw);
mlir_op_trait!(AtomicRmw, OneResult);
mlir_op_trait!(AtomicRmw, ZeroRegions);
mlir_op_trait!(AtomicRmw, ZeroSuccessors);

/// Constructs a new detached/owned [`AtomicRmwOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn atomic_rmw<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    atomic_rmw_op: RmwOp,
    ptr: ValueRef<'v, 'c, 't>,
    val: ValueRef<'v, 'c, 't>,
    mask: Option<ValueRef<'v, 'c, 't>>,
    sem: MemSemantic,
    scope: MemSyncScope,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedAtomicRmwOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    let mut builder = OperationBuilder::new("tt.atomic_rmw", location).add_operand(ptr).add_operand(val);
    if let Some(mask) = mask {
        builder = builder.add_operand(mask);
    }
    builder
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[1, 1, i32::from(mask.is_some())]).unwrap(),
        )
        .add_attribute(ATOMIC_RMW_OP_ATTRIBUTE, context.triton_tt_rmw_op_attribute(atomic_rmw_op))
        .add_attribute(SEM_ATTRIBUTE, context.triton_tt_mem_semantic_attribute(sem))
        .add_attribute(SCOPE_ATTRIBUTE, context.triton_tt_mem_sync_scope_attribute(scope))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::atomic_rmw`")
}

/// Triton `tt` [`Operation`] that performs an atomic compare-and-swap operation.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait AtomicCasOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the pointer operand.
    fn ptr(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the comparison value.
    fn cmp(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the replacement value.
    fn val(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the memory semantic.
    fn sem(&self) -> MemSemanticAttributeRef<'c, 't> {
        self.attribute(SEM_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", SEM_ATTRIBUTE, "tt.atomic_cas"))
            .cast::<MemSemanticAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", SEM_ATTRIBUTE, "tt.atomic_cas"))
    }

    /// Returns the memory synchronization scope.
    fn scope(&self) -> MemSyncScopeAttributeRef<'c, 't> {
        self.attribute(SCOPE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", SCOPE_ATTRIBUTE, "tt.atomic_cas"))
            .cast::<MemSyncScopeAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", SCOPE_ATTRIBUTE, "tt.atomic_cas"))
    }

    /// Returns the old value read from memory.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(AtomicCas);
mlir_op_trait!(AtomicCas, OneResult);
mlir_op_trait!(AtomicCas, ZeroRegions);
mlir_op_trait!(AtomicCas, ZeroSuccessors);

/// Constructs a new detached/owned [`AtomicCasOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn atomic_cas<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    ptr: ValueRef<'v, 'c, 't>,
    cmp: ValueRef<'v, 'c, 't>,
    val: ValueRef<'v, 'c, 't>,
    sem: MemSemantic,
    scope: MemSyncScope,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedAtomicCasOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.atomic_cas", location)
        .add_operand(ptr)
        .add_operand(cmp)
        .add_operand(val)
        .add_attribute(SEM_ATTRIBUTE, context.triton_tt_mem_semantic_attribute(sem))
        .add_attribute(SCOPE_ATTRIBUTE, context.triton_tt_mem_sync_scope_attribute(scope))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::atomic_cas`")
}

/// Triton `tt` [`Operation`] that splats a scalar value into a tensor.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait SplatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the scalar source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Splat);
mlir_op_trait!(Splat, OneOperand);
mlir_op_trait!(Splat, OneResult);
mlir_op_trait!(Splat, ZeroRegions);
mlir_op_trait!(Splat, ZeroSuccessors);

/// Constructs a new detached/owned [`SplatOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn splat<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedSplatOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.splat", location)
        .add_operand(src)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::splat`")
}

/// Triton `tt` [`Operation`] that converts a single-element tensor to a scalar.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait UnsplatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the scalar result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Unsplat);
mlir_op_trait!(Unsplat, OneOperand);
mlir_op_trait!(Unsplat, OneResult);
mlir_op_trait!(Unsplat, ZeroRegions);
mlir_op_trait!(Unsplat, ZeroSuccessors);

/// Constructs a new detached/owned [`UnsplatOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn unsplat<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedUnsplatOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.unsplat", location)
        .add_operand(src)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::unsplat`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` dimension axis.
pub const AXIS_ATTRIBUTE: &str = "axis";

/// Triton `tt` [`Operation`] that inserts a dimension into a tensor.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ExpandDimsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the inserted axis.
    fn axis(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(AXIS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.expand_dims"))
            .cast::<IntegerAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.expand_dims"))
    }

    /// Returns the expanded tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(ExpandDims);
mlir_op_trait!(ExpandDims, OneOperand);
mlir_op_trait!(ExpandDims, OneResult);
mlir_op_trait!(ExpandDims, ZeroRegions);
mlir_op_trait!(ExpandDims, ZeroSuccessors);

/// Constructs a new detached/owned [`ExpandDimsOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn expand_dims<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    axis: i64,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedExpandDimsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.expand_dims", location)
        .add_operand(src)
        .add_attribute(AXIS_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), axis))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::expand_dims`")
}

/// Name of the [`Attribute`] that allows a Triton `tt` reshape to reorder elements.
pub const ALLOW_REORDER_ATTRIBUTE: &str = "allow_reorder";

/// Name of the [`Attribute`] that marks a Triton `tt` layout as efficient.
pub const EFFICIENT_LAYOUT_ATTRIBUTE: &str = "efficient_layout";

/// Triton `tt` [`Operation`] that reinterprets a tensor with a different shape.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns `true` if element reordering is allowed.
    fn allow_reorder(&self) -> bool {
        self.has_attribute(ALLOW_REORDER_ATTRIBUTE)
    }

    /// Returns `true` if the destination layout is marked efficient.
    fn efficient_layout(&self) -> bool {
        self.has_attribute(EFFICIENT_LAYOUT_ATTRIBUTE)
    }

    /// Returns the reshaped tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Reshape);
mlir_op_trait!(Reshape, OneOperand);
mlir_op_trait!(Reshape, OneResult);
mlir_op_trait!(Reshape, ZeroRegions);
mlir_op_trait!(Reshape, ZeroSuccessors);

/// Constructs a new detached/owned [`ReshapeOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn reshape<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    allow_reorder: bool,
    efficient_layout: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedReshapeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    let mut builder = OperationBuilder::new("tt.reshape", location).add_operand(src);
    if allow_reorder {
        builder = builder.add_attribute(ALLOW_REORDER_ATTRIBUTE, context.unit_attribute());
    }
    if efficient_layout {
        builder = builder.add_attribute(EFFICIENT_LAYOUT_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::reshape`")
}

/// Triton `tt` [`Operation`] that broadcasts a tensor to a larger shape.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait BroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the broadcast tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Broadcast);
mlir_op_trait!(Broadcast, OneOperand);
mlir_op_trait!(Broadcast, OneResult);
mlir_op_trait!(Broadcast, ZeroRegions);
mlir_op_trait!(Broadcast, ZeroSuccessors);

/// Constructs a new detached/owned [`BroadcastOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn broadcast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedBroadcastOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.broadcast", location)
        .add_operand(src)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::broadcast`")
}

/// Triton `tt` [`Operation`] that concatenates two tensors.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait CatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side tensor.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right-hand-side tensor.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the concatenated tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Cat);
mlir_op_trait!(Cat, OneResult);
mlir_op_trait!(Cat, ZeroRegions);
mlir_op_trait!(Cat, ZeroSuccessors);

/// Constructs a new detached/owned [`CatOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn cat<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    lhs: ValueRef<'v, 'c, 't>,
    rhs: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedCatOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.cat", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::cat`")
}

/// Triton `tt` [`Operation`] that joins two tensors along a new minor dimension.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait JoinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side tensor.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right-hand-side tensor.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the joined tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Join);
mlir_op_trait!(Join, OneResult);
mlir_op_trait!(Join, ZeroRegions);
mlir_op_trait!(Join, ZeroSuccessors);

/// Constructs a new detached/owned [`JoinOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn join<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    lhs: ValueRef<'v, 'c, 't>,
    rhs: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedJoinOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.join", location)
        .add_operand(lhs)
        .add_operand(rhs)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::join`")
}

/// Triton `tt` [`Operation`] that splits a tensor along its last dimension.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait SplitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the left split result.
    fn out_lhs(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }

    /// Returns the right split result.
    fn out_rhs(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 1).unwrap()
    }
}

mlir_op!(Split);
mlir_op_trait!(Split, ZeroRegions);
mlir_op_trait!(Split, ZeroSuccessors);

/// Constructs a new detached/owned [`SplitOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn split<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> DetachedSplitOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.split", location)
        .add_operand(src)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::split`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` transpose order.
pub const ORDER_ATTRIBUTE: &str = "order";

/// Triton `tt` [`Operation`] that permutes tensor dimensions.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait TransOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor source value.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the transpose order.
    fn order(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(ORDER_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", ORDER_ATTRIBUTE, "tt.trans"))
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", ORDER_ATTRIBUTE, "tt.trans"))
    }

    /// Returns the transposed tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Trans);
mlir_op_trait!(Trans, OneOperand);
mlir_op_trait!(Trans, OneResult);
mlir_op_trait!(Trans, ZeroRegions);
mlir_op_trait!(Trans, ZeroSuccessors);

/// Constructs a new detached/owned [`TransOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn trans<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    order: &[i32],
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedTransOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.trans", location)
        .add_operand(src)
        .add_attribute(ORDER_ATTRIBUTE, context.dense_i32_array_attribute(order).unwrap())
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::trans`")
}

/// Triton `tt` [`Operation`] that returns the current program identifier along an axis.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait GetProgramIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried program identifier axis.
    fn axis(&self) -> ProgramIdDimAttributeRef<'c, 't> {
        self.attribute(AXIS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.get_program_id"))
            .cast::<ProgramIdDimAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.get_program_id"))
    }

    /// Returns the program identifier.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(GetProgramId);
mlir_op_trait!(GetProgramId, ZeroOperands);
mlir_op_trait!(GetProgramId, OneResult);
mlir_op_trait!(GetProgramId, ZeroRegions);
mlir_op_trait!(GetProgramId, ZeroSuccessors);

/// Constructs a new detached/owned [`GetProgramIdOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn get_program_id<'c, 't: 'c, L: Location<'c, 't>>(
    axis: ProgramIdDim,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedGetProgramIdOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.get_program_id", location)
        .add_attribute(AXIS_ATTRIBUTE, context.triton_tt_program_id_dim_attribute(axis))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::get_program_id`")
}

/// Triton `tt` [`Operation`] that returns the number of programs along an axis.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait GetNumProgramsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried program-count axis.
    fn axis(&self) -> ProgramIdDimAttributeRef<'c, 't> {
        self.attribute(AXIS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.get_num_programs"))
            .cast::<ProgramIdDimAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.get_num_programs"))
    }

    /// Returns the number of programs.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(GetNumPrograms);
mlir_op_trait!(GetNumPrograms, ZeroOperands);
mlir_op_trait!(GetNumPrograms, OneResult);
mlir_op_trait!(GetNumPrograms, ZeroRegions);
mlir_op_trait!(GetNumPrograms, ZeroSuccessors);

/// Constructs a new detached/owned [`GetNumProgramsOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn get_num_programs<'c, 't: 'c, L: Location<'c, 't>>(
    axis: ProgramIdDim,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedGetNumProgramsOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.get_num_programs", location)
        .add_attribute(AXIS_ATTRIBUTE, context.triton_tt_program_id_dim_attribute(axis))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::get_num_programs`")
}

/// Name of the [`Attribute`] that stores Triton `tt` dot input precision.
pub const INPUT_PRECISION_ATTRIBUTE: &str = "inputPrecision";

/// Name of the [`Attribute`] that stores the maximum number of imprecise accumulations for a Triton `tt` dot.
pub const MAX_NUM_IMPRECISE_ACC_ATTRIBUTE: &str = "maxNumImpreciseAcc";

/// Triton `tt` [`Operation`] that computes matrix multiplication plus an accumulator.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait DotOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side matrix.
    fn a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right-hand-side matrix.
    fn b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the accumulator matrix.
    fn c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the input precision mode.
    fn input_precision(&self) -> InputPrecisionAttributeRef<'c, 't> {
        self.attribute(INPUT_PRECISION_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", INPUT_PRECISION_ATTRIBUTE, "tt.dot"))
            .cast::<InputPrecisionAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", INPUT_PRECISION_ATTRIBUTE, "tt.dot"))
    }

    /// Returns the maximum number of imprecise accumulations.
    fn max_num_imprecise_acc(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(MAX_NUM_IMPRECISE_ACC_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", MAX_NUM_IMPRECISE_ACC_ATTRIBUTE, "tt.dot"))
            .cast::<IntegerAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", MAX_NUM_IMPRECISE_ACC_ATTRIBUTE, "tt.dot"))
    }

    /// Returns the dot result.
    fn d(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Dot);
mlir_op_trait!(Dot, OneResult);
mlir_op_trait!(Dot, ZeroRegions);
mlir_op_trait!(Dot, ZeroSuccessors);

/// Constructs a new detached/owned [`DotOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dot<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    a: ValueRef<'v, 'c, 't>,
    b: ValueRef<'v, 'c, 't>,
    c: ValueRef<'v, 'c, 't>,
    input_precision: InputPrecision,
    max_num_imprecise_acc: i64,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedDotOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.dot", location)
        .add_operand(a)
        .add_operand(b)
        .add_operand(c)
        .add_attribute(INPUT_PRECISION_ATTRIBUTE, context.triton_tt_input_precision_attribute(input_precision))
        .add_attribute(
            MAX_NUM_IMPRECISE_ACC_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), max_num_imprecise_acc),
        )
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::dot`")
}

/// Name of the [`Attribute`] that stores the left scaled-dot element type.
pub const A_ELEM_TYPE_ATTRIBUTE: &str = "a_elem_type";

/// Name of the [`Attribute`] that stores the right scaled-dot element type.
pub const B_ELEM_TYPE_ATTRIBUTE: &str = "b_elem_type";

/// Name of the [`Attribute`] that marks a scaled dot as using fast math.
pub const FAST_MATH_ATTRIBUTE: &str = "fastMath";

/// Name of the [`Attribute`] that marks whether the left-hand-side K dimension is packed.
pub const LHS_K_PACK_ATTRIBUTE: &str = "lhs_k_pack";

/// Name of the [`Attribute`] that marks whether the right-hand-side K dimension is packed.
pub const RHS_K_PACK_ATTRIBUTE: &str = "rhs_k_pack";

/// Triton `tt` [`Operation`] that computes matrix multiplication with microscaling.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait DotScaledOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left-hand-side matrix.
    fn a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right-hand-side matrix.
    fn b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the accumulator matrix.
    fn c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the optional left-hand-side scale tensor.
    fn a_scale(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes: Vec<i32> = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
            .unwrap_or_default();
        if sizes.get(3).copied().unwrap_or(0) > 0 { self.operand_value(3) } else { None }
    }

    /// Returns the optional right-hand-side scale tensor.
    fn b_scale(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes: Vec<i32> = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
            .unwrap_or_default();
        if sizes.get(4).copied().unwrap_or(0) > 0 {
            let index = 3 + sizes.get(3).copied().unwrap_or(0) as usize;
            self.operand_value(index)
        } else {
            None
        }
    }

    /// Returns the left-hand-side element type.
    fn a_elem_type(&self) -> ScaleDotElemTypeAttributeRef<'c, 't> {
        self.attribute(A_ELEM_TYPE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", A_ELEM_TYPE_ATTRIBUTE, "tt.dot_scaled"))
            .cast::<ScaleDotElemTypeAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", A_ELEM_TYPE_ATTRIBUTE, "tt.dot_scaled"))
    }

    /// Returns the right-hand-side element type.
    fn b_elem_type(&self) -> ScaleDotElemTypeAttributeRef<'c, 't> {
        self.attribute(B_ELEM_TYPE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", B_ELEM_TYPE_ATTRIBUTE, "tt.dot_scaled"))
            .cast::<ScaleDotElemTypeAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", B_ELEM_TYPE_ATTRIBUTE, "tt.dot_scaled"))
    }

    /// Returns `true` if fast math is enabled.
    fn fast_math(&self) -> bool {
        self.attribute(FAST_MATH_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", FAST_MATH_ATTRIBUTE, "tt.dot_scaled"))
            .cast::<BooleanAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", FAST_MATH_ATTRIBUTE, "tt.dot_scaled"))
            .value()
    }

    /// Returns `true` if the left-hand-side K dimension is packed.
    fn lhs_k_pack(&self) -> bool {
        self.attribute(LHS_K_PACK_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", LHS_K_PACK_ATTRIBUTE, "tt.dot_scaled"))
            .cast::<BooleanAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", LHS_K_PACK_ATTRIBUTE, "tt.dot_scaled"))
            .value()
    }

    /// Returns `true` if the right-hand-side K dimension is packed.
    fn rhs_k_pack(&self) -> bool {
        self.attribute(RHS_K_PACK_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", RHS_K_PACK_ATTRIBUTE, "tt.dot_scaled"))
            .cast::<BooleanAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", RHS_K_PACK_ATTRIBUTE, "tt.dot_scaled"))
            .value()
    }

    /// Returns the scaled-dot result.
    fn d(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(DotScaled);
mlir_op_trait!(DotScaled, OneResult);
mlir_op_trait!(DotScaled, ZeroRegions);
mlir_op_trait!(DotScaled, ZeroSuccessors);

/// Constructs a new detached/owned [`DotScaledOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn dot_scaled<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    a: ValueRef<'v, 'c, 't>,
    b: ValueRef<'v, 'c, 't>,
    c: ValueRef<'v, 'c, 't>,
    a_scale: Option<ValueRef<'v, 'c, 't>>,
    b_scale: Option<ValueRef<'v, 'c, 't>>,
    a_elem_type: ScaleDotElemType,
    b_elem_type: ScaleDotElemType,
    fast_math: bool,
    lhs_k_pack: bool,
    rhs_k_pack: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedDotScaledOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    let mut builder = OperationBuilder::new("tt.dot_scaled", location).add_operand(a).add_operand(b).add_operand(c);
    if let Some(a_scale) = a_scale {
        builder = builder.add_operand(a_scale);
    }
    if let Some(b_scale) = b_scale {
        builder = builder.add_operand(b_scale);
    }
    builder
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[1, 1, 1, i32::from(a_scale.is_some()), i32::from(b_scale.is_some())])
                .unwrap(),
        )
        .add_attribute(A_ELEM_TYPE_ATTRIBUTE, context.triton_tt_scale_dot_elem_type_attribute(a_elem_type))
        .add_attribute(B_ELEM_TYPE_ATTRIBUTE, context.triton_tt_scale_dot_elem_type_attribute(b_elem_type))
        .add_attribute(FAST_MATH_ATTRIBUTE, context.boolean_attribute(fast_math))
        .add_attribute(LHS_K_PACK_ATTRIBUTE, context.boolean_attribute(lhs_k_pack))
        .add_attribute(RHS_K_PACK_ATTRIBUTE, context.boolean_attribute(rhs_k_pack))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::dot_scaled`")
}

/// Triton `tt` [`Operation`] that reduces one or more tensors using a combiner region.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source tensors.
    fn srcs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the reduction axis.
    fn axis(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(AXIS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.reduce"))
            .cast::<IntegerAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.reduce"))
    }

    /// Returns the combiner region.
    fn combine_op(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the reduction results.
    fn result_values(&self) -> Vec<OperationResultRef<'o, 'c, 't>> {
        self.results().collect()
    }
}

mlir_op!(Reduce);
mlir_op_trait!(Reduce, OneRegion);
mlir_op_trait!(Reduce, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduceOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn reduce<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    srcs: &[ValueRef<'v, 'c, 't>],
    axis: i64,
    result_types: &[TypeRef<'c, 't>],
    combine_op: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.reduce", location)
        .add_operands(srcs)
        .add_attribute(AXIS_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), axis))
        .add_region(combine_op)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::reduce`")
}

/// Triton `tt` [`Operation`] that terminates a `tt.reduce` combiner region.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ReduceReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded reduction values.
    fn result_values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(ReduceReturn);
mlir_op_trait!(ReduceReturn, ZeroRegions);
mlir_op_trait!(ReduceReturn, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduceReturnOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn reduce_return<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    result_values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedReduceReturnOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.reduce.return", location)
        .add_operands(result_values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::reduce_return`")
}

/// Name of the [`Attribute`] that marks a Triton `tt` scan as reverse.
pub const REVERSE_ATTRIBUTE: &str = "reverse";

/// Triton `tt` [`Operation`] that computes an associative scan using a combiner region.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ScanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source tensors.
    fn srcs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the scan axis.
    fn axis(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(AXIS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.scan"))
            .cast::<IntegerAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.scan"))
    }

    /// Returns `true` if the scan proceeds in reverse order.
    fn reverse(&self) -> bool {
        self.attribute(REVERSE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", REVERSE_ATTRIBUTE, "tt.scan"))
            .cast::<BooleanAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", REVERSE_ATTRIBUTE, "tt.scan"))
            .value()
    }

    /// Returns the combiner region.
    fn combine_op(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the scan results.
    fn result_values(&self) -> Vec<OperationResultRef<'o, 'c, 't>> {
        self.results().collect()
    }
}

mlir_op!(Scan);
mlir_op_trait!(Scan, OneRegion);
mlir_op_trait!(Scan, ZeroSuccessors);

/// Constructs a new detached/owned [`ScanOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn scan<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    srcs: &[ValueRef<'v, 'c, 't>],
    axis: i64,
    reverse: bool,
    result_types: &[TypeRef<'c, 't>],
    combine_op: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedScanOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.scan", location)
        .add_operands(srcs)
        .add_attribute(AXIS_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), axis))
        .add_attribute(REVERSE_ATTRIBUTE, context.boolean_attribute(reverse))
        .add_region(combine_op)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::scan`")
}

/// Triton `tt` [`Operation`] that terminates a `tt.scan` combiner region.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ScanReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded scan values.
    fn result_values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(ScanReturn);
mlir_op_trait!(ScanReturn, ZeroRegions);
mlir_op_trait!(ScanReturn, ZeroSuccessors);

/// Constructs a new detached/owned [`ScanReturnOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn scan_return<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    result_values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedScanReturnOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.scan.return", location)
        .add_operands(result_values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::scan_return`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` map-elementwise packing factor.
pub const PACK_ATTRIBUTE: &str = "pack";

/// Triton `tt` [`Operation`] that maps a scalar region over tensors elementwise.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait MapElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source tensors.
    fn srcs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the packing factor.
    fn pack(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(PACK_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", PACK_ATTRIBUTE, "tt.map_elementwise"))
            .cast::<IntegerAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", PACK_ATTRIBUTE, "tt.map_elementwise"))
    }

    /// Returns the scalar mapping region.
    fn scalar_op(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns the mapped tensors.
    fn result_values(&self) -> Vec<OperationResultRef<'o, 'c, 't>> {
        self.results().collect()
    }
}

mlir_op!(MapElementwise);
mlir_op_trait!(MapElementwise, OneRegion);
mlir_op_trait!(MapElementwise, ZeroSuccessors);

/// Constructs a new detached/owned [`MapElementwiseOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn map_elementwise<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    srcs: &[ValueRef<'v, 'c, 't>],
    pack: i64,
    result_types: &[TypeRef<'c, 't>],
    scalar_op: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedMapElementwiseOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.map_elementwise", location)
        .add_operands(srcs)
        .add_attribute(PACK_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), pack))
        .add_region(scalar_op)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::map_elementwise`")
}

/// Triton `tt` [`Operation`] that terminates a `tt.map_elementwise` region.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait MapElementwiseReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded mapped values.
    fn result_values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(MapElementwiseReturn);
mlir_op_trait!(MapElementwiseReturn, ZeroRegions);
mlir_op_trait!(MapElementwiseReturn, ZeroSuccessors);

/// Constructs a new detached/owned [`MapElementwiseReturnOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn map_elementwise_return<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    result_values: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedMapElementwiseReturnOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.map_elementwise.return", location)
        .add_operands(result_values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::map_elementwise_return`")
}

/// Name of the [`Attribute`] that stores an external elementwise library name.
pub const LIBNAME_ATTRIBUTE: &str = "libname";

/// Name of the [`Attribute`] that stores an external elementwise library path.
pub const LIBPATH_ATTRIBUTE: &str = "libpath";

/// Name of the [`Attribute`] that stores an external elementwise symbol name.
pub const SYMBOL_ATTRIBUTE: &str = "symbol";

/// Name of the [`Attribute`] that marks an external or inline operation as pure.
pub const PURE_ATTRIBUTE: &str = "pure";

/// Triton `tt` [`Operation`] that calls an external elementwise function.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ExternElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source values.
    fn srcs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the library name.
    fn libname(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(LIBNAME_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", LIBNAME_ATTRIBUTE, "tt.extern_elementwise"))
            .cast::<StringAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", LIBNAME_ATTRIBUTE, "tt.extern_elementwise"))
    }

    /// Returns the library path.
    fn libpath(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(LIBPATH_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", LIBPATH_ATTRIBUTE, "tt.extern_elementwise"))
            .cast::<StringAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", LIBPATH_ATTRIBUTE, "tt.extern_elementwise"))
    }

    /// Returns the external symbol name.
    fn symbol(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(SYMBOL_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", SYMBOL_ATTRIBUTE, "tt.extern_elementwise"))
            .cast::<StringAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", SYMBOL_ATTRIBUTE, "tt.extern_elementwise"))
    }

    /// Returns `true` if the external function is pure.
    fn pure(&self) -> bool {
        self.attribute(PURE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", PURE_ATTRIBUTE, "tt.extern_elementwise"))
            .cast::<BooleanAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", PURE_ATTRIBUTE, "tt.extern_elementwise"))
            .value()
    }

    /// Returns the external call result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(ExternElementwise);
mlir_op_trait!(ExternElementwise, OneResult);
mlir_op_trait!(ExternElementwise, ZeroRegions);
mlir_op_trait!(ExternElementwise, ZeroSuccessors);

/// Constructs a new detached/owned [`ExternElementwiseOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn extern_elementwise<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    srcs: &[ValueRef<'v, 'c, 't>],
    libname: &str,
    libpath: &str,
    symbol: &str,
    pure: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedExternElementwiseOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.extern_elementwise", location)
        .add_operands(srcs)
        .add_attribute(LIBNAME_ATTRIBUTE, context.string_attribute(libname))
        .add_attribute(LIBPATH_ATTRIBUTE, context.string_attribute(libpath))
        .add_attribute(SYMBOL_ATTRIBUTE, context.string_attribute(symbol))
        .add_attribute(PURE_ATTRIBUTE, context.boolean_attribute(pure))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::extern_elementwise`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` range start.
pub const START_ATTRIBUTE: &str = "start";

/// Name of the [`Attribute`] that stores a Triton `tt` range end.
pub const END_ATTRIBUTE: &str = "end";

/// Triton `tt` [`Operation`] that creates a one-dimensional integer range tensor.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait MakeRangeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the inclusive range start.
    fn start(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(START_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", START_ATTRIBUTE, "tt.make_range"))
            .cast::<IntegerAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", START_ATTRIBUTE, "tt.make_range"))
    }

    /// Returns the exclusive range end.
    fn end(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(END_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", END_ATTRIBUTE, "tt.make_range"))
            .cast::<IntegerAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", END_ATTRIBUTE, "tt.make_range"))
    }

    /// Returns the range tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MakeRange);
mlir_op_trait!(MakeRange, ZeroOperands);
mlir_op_trait!(MakeRange, OneResult);
mlir_op_trait!(MakeRange, ZeroRegions);
mlir_op_trait!(MakeRange, ZeroSuccessors);

/// Constructs a new detached/owned [`MakeRangeOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn make_range<'c, 't: 'c, L: Location<'c, 't>>(
    start: i64,
    end: i64,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMakeRangeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.make_range", location)
        .add_attribute(START_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), start))
        .add_attribute(END_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), end))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::make_range`")
}

/// Name of the [`Attribute`] that stores inline assembly text.
pub const ASM_STRING_ATTRIBUTE: &str = "asm_string";

/// Name of the [`Attribute`] that stores inline assembly constraints.
pub const CONSTRAINTS_ATTRIBUTE: &str = "constraints";

/// Name of the [`Attribute`] that stores the inline assembly packed element count.
pub const PACKED_ELEMENT_ATTRIBUTE: &str = "packed_element";

/// Triton `tt` [`Operation`] that applies inline assembly elementwise.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ElementwiseInlineAsmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the inline assembly text.
    fn asm_string(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(ASM_STRING_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{}` attribute in `{}`", ASM_STRING_ATTRIBUTE, "tt.elementwise_inline_asm")
            })
            .cast::<StringAttributeRef>()
            .unwrap_or_else(|| {
                panic!("invalid `{}` attribute in `{}`", ASM_STRING_ATTRIBUTE, "tt.elementwise_inline_asm")
            })
    }

    /// Returns the inline assembly constraints.
    fn constraints(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(CONSTRAINTS_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{}` attribute in `{}`", CONSTRAINTS_ATTRIBUTE, "tt.elementwise_inline_asm")
            })
            .cast::<StringAttributeRef>()
            .unwrap_or_else(|| {
                panic!("invalid `{}` attribute in `{}`", CONSTRAINTS_ATTRIBUTE, "tt.elementwise_inline_asm")
            })
    }

    /// Returns `true` if the inline assembly is pure.
    fn pure(&self) -> bool {
        self.attribute(PURE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", PURE_ATTRIBUTE, "tt.elementwise_inline_asm"))
            .cast::<BooleanAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", PURE_ATTRIBUTE, "tt.elementwise_inline_asm"))
            .value()
    }

    /// Returns the packed element count.
    fn packed_element(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(PACKED_ELEMENT_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{}` attribute in `{}`", PACKED_ELEMENT_ATTRIBUTE, "tt.elementwise_inline_asm")
            })
            .cast::<IntegerAttributeRef>()
            .unwrap_or_else(|| {
                panic!("invalid `{}` attribute in `{}`", PACKED_ELEMENT_ATTRIBUTE, "tt.elementwise_inline_asm")
            })
    }

    /// Returns the inline assembly operands.
    fn args(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the inline assembly results.
    fn result_values(&self) -> Vec<OperationResultRef<'o, 'c, 't>> {
        self.results().collect()
    }
}

mlir_op!(ElementwiseInlineAsm);
mlir_op_trait!(ElementwiseInlineAsm, ZeroRegions);
mlir_op_trait!(ElementwiseInlineAsm, ZeroSuccessors);

/// Constructs a new detached/owned [`ElementwiseInlineAsmOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn elementwise_inline_asm<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    args: &[ValueRef<'v, 'c, 't>],
    asm_string: &str,
    constraints: &str,
    pure: bool,
    packed_element: i64,
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> DetachedElementwiseInlineAsmOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.elementwise_inline_asm", location)
        .add_operands(args)
        .add_attribute(ASM_STRING_ATTRIBUTE, context.string_attribute(asm_string))
        .add_attribute(CONSTRAINTS_ATTRIBUTE, context.string_attribute(constraints))
        .add_attribute(PURE_ATTRIBUTE, context.boolean_attribute(pure))
        .add_attribute(
            PACKED_ELEMENT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), packed_element),
        )
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::elementwise_inline_asm`")
}

/// Triton `tt` [`Operation`] that computes a histogram tensor.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait HistogramOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input tensor.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional mask tensor.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(1)
    }

    /// Returns the histogram tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Histogram);
mlir_op_trait!(Histogram, OneResult);
mlir_op_trait!(Histogram, ZeroRegions);
mlir_op_trait!(Histogram, ZeroSuccessors);

/// Constructs a new detached/owned [`HistogramOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn histogram<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    mask: Option<ValueRef<'v, 'c, 't>>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedHistogramOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    let mut builder = OperationBuilder::new("tt.histogram", location).add_operand(src);
    if let Some(mask) = mask {
        builder = builder.add_operand(mask);
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::histogram`")
}

/// Triton `tt` [`Operation`] that gathers elements from a tensor along an axis.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait GatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source tensor.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the indices tensor.
    fn indices(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the gather axis.
    fn axis(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(AXIS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.gather"))
            .cast::<IntegerAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", AXIS_ATTRIBUTE, "tt.gather"))
    }

    /// Returns `true` if the gather is marked as having an efficient layout.
    fn efficient_layout(&self) -> bool {
        self.has_attribute(EFFICIENT_LAYOUT_ATTRIBUTE)
    }

    /// Returns the gathered tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Gather);
mlir_op_trait!(Gather, OneResult);
mlir_op_trait!(Gather, ZeroRegions);
mlir_op_trait!(Gather, ZeroSuccessors);

/// Constructs a new detached/owned [`GatherOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn gather<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    indices: ValueRef<'v, 'c, 't>,
    axis: i64,
    efficient_layout: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedGatherOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    let mut builder = OperationBuilder::new("tt.gather", location)
        .add_operand(src)
        .add_operand(indices)
        .add_attribute(AXIS_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), axis));
    if efficient_layout {
        builder = builder.add_attribute(EFFICIENT_LAYOUT_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::gather`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` print prefix.
pub const PREFIX_ATTRIBUTE: &str = "prefix";

/// Name of the [`Attribute`] that marks a Triton `tt` print as hexadecimal.
pub const HEX_ATTRIBUTE: &str = "hex";

/// Name of the [`Attribute`] that stores Triton `tt` print signedness flags.
pub const IS_SIGNED_ATTRIBUTE: &str = "isSigned";

/// Triton `tt` [`Operation`] that emits device-side debug output.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait PrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the print prefix.
    fn prefix(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(PREFIX_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", PREFIX_ATTRIBUTE, "tt.print"))
            .cast::<StringAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", PREFIX_ATTRIBUTE, "tt.print"))
    }

    /// Returns `true` if arguments are printed in hexadecimal.
    fn hex(&self) -> bool {
        self.attribute(HEX_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", HEX_ATTRIBUTE, "tt.print"))
            .cast::<BooleanAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", HEX_ATTRIBUTE, "tt.print"))
            .value()
    }

    /// Returns the values to print.
    fn args(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the signedness flags for printed values.
    fn is_signed(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(IS_SIGNED_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", IS_SIGNED_ATTRIBUTE, "tt.print"))
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", IS_SIGNED_ATTRIBUTE, "tt.print"))
    }
}

mlir_op!(Print);
mlir_op_trait!(Print, ZeroRegions);
mlir_op_trait!(Print, ZeroSuccessors);

/// Constructs a new detached/owned [`PrintOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn print<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    args: &[ValueRef<'v, 'c, 't>],
    prefix: &str,
    hex: bool,
    is_signed: &[i32],
    location: L,
) -> DetachedPrintOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.print", location)
        .add_operands(args)
        .add_attribute(PREFIX_ATTRIBUTE, context.string_attribute(prefix))
        .add_attribute(HEX_ATTRIBUTE, context.boolean_attribute(hex))
        .add_attribute(IS_SIGNED_ATTRIBUTE, context.dense_i32_array_attribute(is_signed).unwrap())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::print`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` assertion message.
pub const MESSAGE_ATTRIBUTE: &str = "message";

/// Triton `tt` [`Operation`] that performs device-side assertion checking.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait AssertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the assertion condition.
    fn condition(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the assertion message.
    fn message(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(MESSAGE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", MESSAGE_ATTRIBUTE, "tt.assert"))
            .cast::<StringAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", MESSAGE_ATTRIBUTE, "tt.assert"))
    }
}

mlir_op!(Assert);
mlir_op_trait!(Assert, ZeroRegions);
mlir_op_trait!(Assert, ZeroSuccessors);

/// Constructs a new detached/owned [`AssertOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn assert<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    condition: ValueRef<'v, 'c, 't>,
    message: &str,
    location: L,
) -> DetachedAssertOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.assert", location)
        .add_operand(condition)
        .add_attribute(MESSAGE_ATTRIBUTE, context.string_attribute(message))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::assert`")
}

/// Name of the [`Attribute`] that stores a tensor descriptor padding option.
pub const PADDING_ATTRIBUTE: &str = "padding";

/// Triton `tt` [`Operation`] that creates a tensor descriptor from pointer metadata.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait MakeTensorDescOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the base pointer.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the shape values.
    fn shape(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = (self.operand_count().saturating_sub(1)) / 2;
        self.operand_values().skip(1).take(count).collect()
    }

    /// Returns the stride values.
    fn strides(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = (self.operand_count().saturating_sub(1)) / 2;
        self.operand_values().skip(1 + count).collect()
    }

    /// Returns the padding option.
    fn padding(&self) -> PaddingOptionAttributeRef<'c, 't> {
        self.attribute(PADDING_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", PADDING_ATTRIBUTE, "tt.make_tensor_descriptor"))
            .cast::<PaddingOptionAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", PADDING_ATTRIBUTE, "tt.make_tensor_descriptor"))
    }

    /// Returns the tensor descriptor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MakeTensorDesc);
mlir_op_trait!(MakeTensorDesc, OneResult);
mlir_op_trait!(MakeTensorDesc, ZeroRegions);
mlir_op_trait!(MakeTensorDesc, ZeroSuccessors);

/// Constructs a new detached/owned [`MakeTensorDescOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn make_tensor_descriptor<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'v, 'c, 't>,
    shape: &[ValueRef<'v, 'c, 't>],
    strides: &[ValueRef<'v, 'c, 't>],
    padding: PaddingOption,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMakeTensorDescOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.make_tensor_descriptor", location)
        .add_operand(base)
        .add_operands(shape)
        .add_operands(strides)
        .add_attribute(PADDING_ATTRIBUTE, context.triton_tt_padding_option_attribute(padding))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::make_tensor_descriptor`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` call callee.
pub const CALLEE_ATTRIBUTE: &str = "callee";

/// Triton `tt` [`Operation`] that directly calls a Triton `tt` function symbol.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait CallOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the callee symbol reference.
    fn callee(&self) -> FlatSymbolRefAttributeRef<'c, 't> {
        self.attribute(CALLEE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", CALLEE_ATTRIBUTE, "tt.call"))
            .cast::<FlatSymbolRefAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", CALLEE_ATTRIBUTE, "tt.call"))
    }

    /// Returns the call arguments.
    fn operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the call results.
    fn result_values(&self) -> Vec<OperationResultRef<'o, 'c, 't>> {
        self.results().collect()
    }
}

mlir_op!(Call);
mlir_op_trait!(Call, ZeroRegions);
mlir_op_trait!(Call, ZeroSuccessors);

/// Constructs a new detached/owned [`CallOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn call<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    callee: &str,
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> DetachedCallOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.call", location)
        .add_operands(operands)
        .add_attribute(CALLEE_ATTRIBUTE, context.flat_symbol_ref_attribute(callee))
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::call`")
}

/// Name of the [`Attribute`] that stores a Triton `tt` function symbol name.
pub const SYMBOL_NAME_ATTRIBUTE: &str = "sym_name";

/// Name of the [`Attribute`] that stores a Triton `tt` function type.
pub const FUNCTION_TYPE_ATTRIBUTE: &str = "function_type";

/// Name of the [`Attribute`] that stores Triton `tt` symbol visibility.
pub const SYMBOL_VISIBILITY_ATTRIBUTE: &str = "sym_visibility";

/// Name of the [`Attribute`] that stores Triton `tt` function argument attributes.
pub const ARG_ATTRIBUTES_ATTRIBUTE: &str = "arg_attrs";

/// Name of the [`Attribute`] that stores Triton `tt` function result attributes.
pub const RESULT_ATTRIBUTES_ATTRIBUTE: &str = "res_attrs";

/// Triton `tt` [`Operation`] that defines a Triton `tt` function.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait FuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the function symbol name.
    fn sym_name(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(SYMBOL_NAME_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", SYMBOL_NAME_ATTRIBUTE, "tt.func"))
            .cast::<StringAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", SYMBOL_NAME_ATTRIBUTE, "tt.func"))
    }

    /// Returns the function type.
    fn function_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(FUNCTION_TYPE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", FUNCTION_TYPE_ATTRIBUTE, "tt.func"))
            .cast::<TypeAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", FUNCTION_TYPE_ATTRIBUTE, "tt.func"))
    }

    /// Returns the optional symbol visibility.
    fn sym_visibility(&self) -> Option<StringAttributeRef<'c, 't>> {
        self.attribute(SYMBOL_VISIBILITY_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
    }

    /// Returns the optional function argument attributes.
    fn arg_attrs(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(ARG_ATTRIBUTES_ATTRIBUTE).and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
    }

    /// Returns the optional function result attributes.
    fn res_attrs(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(RESULT_ATTRIBUTES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
    }

    /// Returns the function body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Func);
mlir_op_trait!(Func, OneRegion);
mlir_op_trait!(Func, ZeroSuccessors);

/// Constructs a new detached/owned [`FuncOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn func<'c, 't: 'c, L: Location<'c, 't>>(
    sym_name: &str,
    function_type: TypeRef<'c, 't>,
    sym_visibility: Option<&str>,
    arg_attrs: Option<ArrayAttributeRef<'c, 't>>,
    res_attrs: Option<ArrayAttributeRef<'c, 't>>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedFuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    let mut builder = OperationBuilder::new("tt.func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, context.string_attribute(sym_name))
        .add_attribute(FUNCTION_TYPE_ATTRIBUTE, context.type_attribute(function_type));
    if let Some(sym_visibility) = sym_visibility {
        builder = builder.add_attribute(SYMBOL_VISIBILITY_ATTRIBUTE, context.string_attribute(sym_visibility));
    }
    if let Some(arg_attrs) = arg_attrs {
        builder = builder.add_attribute(ARG_ATTRIBUTES_ATTRIBUTE, arg_attrs);
    }
    if let Some(res_attrs) = res_attrs {
        builder = builder.add_attribute(RESULT_ATTRIBUTES_ATTRIBUTE, res_attrs);
    }
    builder
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::func`")
}

/// Triton `tt` [`Operation`] that returns values from a Triton `tt` function.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the returned values.
    fn srcs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a new detached/owned [`ReturnOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn r#return<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    srcs: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedReturnOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.return", location)
        .add_operands(srcs)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::return`")
}

/// Triton `tt` [`Operation`] that loads from a tensor descriptor.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait DescriptorLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor descriptor.
    fn desc(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the descriptor indices.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the cache modifier.
    fn cache(&self) -> CacheModifierAttributeRef<'c, 't> {
        self.attribute(CACHE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", CACHE_ATTRIBUTE, "tt.descriptor_load"))
            .cast::<CacheModifierAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", CACHE_ATTRIBUTE, "tt.descriptor_load"))
    }

    /// Returns the eviction policy.
    fn evict(&self) -> EvictionPolicyAttributeRef<'c, 't> {
        self.attribute(EVICT_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", EVICT_ATTRIBUTE, "tt.descriptor_load"))
            .cast::<EvictionPolicyAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", EVICT_ATTRIBUTE, "tt.descriptor_load"))
    }

    /// Returns the loaded tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(DescriptorLoad);
mlir_op_trait!(DescriptorLoad, OneResult);
mlir_op_trait!(DescriptorLoad, ZeroRegions);
mlir_op_trait!(DescriptorLoad, ZeroSuccessors);

/// Constructs a new detached/owned [`DescriptorLoadOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn descriptor_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    desc: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    cache: CacheModifier,
    evict: EvictionPolicy,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedDescriptorLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.descriptor_load", location)
        .add_operand(desc)
        .add_operands(indices)
        .add_attribute(CACHE_ATTRIBUTE, context.triton_tt_cache_modifier_attribute(cache))
        .add_attribute(EVICT_ATTRIBUTE, context.triton_tt_eviction_policy_attribute(evict))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::descriptor_load`")
}

/// Triton `tt` [`Operation`] that stores a tensor through a tensor descriptor.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait DescriptorStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor descriptor.
    fn desc(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor to store.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the descriptor indices.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(2).collect()
    }
}

mlir_op!(DescriptorStore);
mlir_op_trait!(DescriptorStore, ZeroRegions);
mlir_op_trait!(DescriptorStore, ZeroSuccessors);

/// Constructs a new detached/owned [`DescriptorStoreOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn descriptor_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    desc: ValueRef<'v, 'c, 't>,
    src: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedDescriptorStoreOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.descriptor_store", location)
        .add_operand(desc)
        .add_operand(src)
        .add_operands(indices)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::descriptor_store`")
}

/// Name of the [`Attribute`] that stores a descriptor reduce kind.
pub const KIND_ATTRIBUTE: &str = "kind";

/// Triton `tt` [`Operation`] that performs a reducing store through a tensor descriptor.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait DescriptorReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the descriptor reduce kind.
    fn kind(&self) -> DescriptorReduceKindAttributeRef<'c, 't> {
        self.attribute(KIND_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{}` attribute in `{}`", KIND_ATTRIBUTE, "tt.descriptor_reduce"))
            .cast::<DescriptorReduceKindAttributeRef>()
            .unwrap_or_else(|| panic!("invalid `{}` attribute in `{}`", KIND_ATTRIBUTE, "tt.descriptor_reduce"))
    }

    /// Returns the tensor descriptor.
    fn desc(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor to reduce into the descriptor.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the descriptor indices.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(2).collect()
    }
}

mlir_op!(DescriptorReduce);
mlir_op_trait!(DescriptorReduce, ZeroRegions);
mlir_op_trait!(DescriptorReduce, ZeroSuccessors);

/// Constructs a new detached/owned [`DescriptorReduceOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn descriptor_reduce<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    kind: DescriptorReduceKind,
    desc: ValueRef<'v, 'c, 't>,
    src: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedDescriptorReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.descriptor_reduce", location)
        .add_operand(desc)
        .add_operand(src)
        .add_operands(indices)
        .add_attribute(KIND_ATTRIBUTE, context.triton_tt_descriptor_reduce_kind_attribute(kind))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::descriptor_reduce`")
}

/// Triton `tt` [`Operation`] that gathers rows from a tensor descriptor.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait DescriptorGatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor descriptor.
    fn desc(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the x-offset tensor.
    fn x_offsets(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the y-offset scalar.
    fn y_offset(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the gathered tensor.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(DescriptorGather);
mlir_op_trait!(DescriptorGather, OneResult);
mlir_op_trait!(DescriptorGather, ZeroRegions);
mlir_op_trait!(DescriptorGather, ZeroSuccessors);

/// Constructs a new detached/owned [`DescriptorGatherOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn descriptor_gather<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    desc: ValueRef<'v, 'c, 't>,
    x_offsets: ValueRef<'v, 'c, 't>,
    y_offset: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedDescriptorGatherOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.descriptor_gather", location)
        .add_operand(desc)
        .add_operand(x_offsets)
        .add_operand(y_offset)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::descriptor_gather`")
}

/// Triton `tt` [`Operation`] that scatters rows into a tensor descriptor.
///
/// Refer to the [official Triton operation documentation](https://triton-lang.org/main/dialects/TritonOps.html)
/// for more information.
pub trait DescriptorScatterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor descriptor.
    fn desc(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the x-offset tensor.
    fn x_offsets(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the y-offset scalar.
    fn y_offset(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the source tensor.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }
}

mlir_op!(DescriptorScatter);
mlir_op_trait!(DescriptorScatter, ZeroRegions);
mlir_op_trait!(DescriptorScatter, ZeroSuccessors);

/// Constructs a new detached/owned [`DescriptorScatterOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn descriptor_scatter<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    desc: ValueRef<'v, 'c, 't>,
    x_offsets: ValueRef<'v, 'c, 't>,
    y_offset: ValueRef<'v, 'c, 't>,
    src: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedDescriptorScatterOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::triton_tt());
    OperationBuilder::new("tt.descriptor_scatter", location)
        .add_operand(desc)
        .add_operand(x_offsets)
        .add_operand(y_offset)
        .add_operand(src)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `tt::descriptor_scatter`")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{
        AttributeRef, Block, Context, DialectHandle, Operation, Region, Size, Type, TypeRef, UnknownLocationRef, Value,
    };

    use super::super::attributes::{
        CacheModifier, DescriptorReduceKind, EvictionPolicy, InputPrecision, MemSemantic, MemSyncScope, PaddingOption,
        ProgramIdDim, PropagateNan, RmwOp, RoundingMode, ScaleDotElemType,
    };
    use super::*;

    /// Common scalar, tensor, pointer, descriptor, and function types used by Triton `tt` operation wrapper tests.
    #[derive(Copy, Clone)]
    struct TestTypes<'c, 't> {
        /// One-bit signless integer type.
        i1: TypeRef<'c, 't>,

        /// 32-bit signless integer type.
        i32: TypeRef<'c, 't>,

        /// 64-bit signless integer type.
        i64: TypeRef<'c, 't>,

        /// 32-bit floating-point type.
        f32: TypeRef<'c, 't>,

        /// 64-bit floating-point type.
        f64: TypeRef<'c, 't>,

        /// Boolean tensor type.
        tensor_i1: TypeRef<'c, 't>,

        /// Integer tensor type.
        tensor_i32: TypeRef<'c, 't>,

        /// Floating-point tensor type.
        tensor_f32: TypeRef<'c, 't>,

        /// Floating-point pointer type.
        pointer: TypeRef<'c, 't>,

        /// Tensor descriptor type.
        tensor_desc: TypeRef<'c, 't>,

        /// Function type.
        function: TypeRef<'c, 't>,
    }

    impl<'c, 't> TestTypes<'c, 't> {
        /// Builds the common test type set in `context`.
        fn new(context: &'c Context<'t>, location: UnknownLocationRef<'c, 't>) -> Self {
            let i1_type = context.signless_integer_type(1);
            let i32_type = context.signless_integer_type(32);
            let i64_type = context.signless_integer_type(64);
            let f32_type = context.float32_type();
            let f64_type = context.float64_type();
            let tensor_i1_type = context.tensor_type(i1_type, &[Size::Static(4)], None, location).unwrap();
            let tensor_i32_type = context.tensor_type(i32_type, &[Size::Static(4)], None, location).unwrap();
            let tensor_f32_type = context.tensor_type(f32_type, &[Size::Static(4)], None, location).unwrap();
            let pointer_type = context.triton_tt_pointer_type(f32_type, 1);
            let tensor_desc_type = context.triton_tt_tensor_desc_type(&[Size::Static(4)], f32_type, None);
            let function_type = context.function_type(&[i32_type], &[i32_type]);

            Self {
                i1: i1_type.as_ref(),
                i32: i32_type.as_ref(),
                i64: i64_type.as_ref(),
                f32: f32_type.as_ref(),
                f64: f64_type.as_ref(),
                tensor_i1: tensor_i1_type.as_ref(),
                tensor_i32: tensor_i32_type.as_ref(),
                tensor_f32: tensor_f32_type.as_ref(),
                pointer: pointer_type.as_ref(),
                tensor_desc: tensor_desc_type.as_ref(),
                function: function_type.as_ref(),
            }
        }
    }

    macro_rules! tt_operation_test {
        ($test_name:ident, |$context:ident, $location:ident, $values:ident, $types:ident| $body:block) => {
            #[test]
            fn $test_name() {
                let $context = Context::new();
                $context.load_dialect(DialectHandle::triton_tt());
                let $location = $context.unknown_location();
                let $types = TestTypes::new(&$context, $location);
                let block = $context.block(&[
                    ($types.i64, $location),
                    ($types.pointer, $location),
                    ($types.f32, $location),
                    ($types.f64, $location),
                    ($types.tensor_f32, $location),
                    ($types.tensor_i32, $location),
                    ($types.tensor_i1, $location),
                    ($types.i1, $location),
                    ($types.i32, $location),
                    ($types.tensor_desc, $location),
                    ($types.pointer, $location),
                    ($types.tensor_f32, $location),
                    ($types.i32, $location),
                    ($types.tensor_i32, $location),
                    ($types.f32, $location),
                    ($types.i64, $location),
                ]);
                let $values = (0..16).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();

                $body
            }
        };
    }

    tt_operation_test!(test_int_to_ptr_operation, |context, location, values, types| {
        let operation = int_to_ptr(values[0], types.pointer, location);

        assert_eq!(operation.name().as_str(), Ok("tt.int_to_ptr"));
        assert_eq!(operation.src(), values[0]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.pointer);
        assert_eq!(operation.context(), &context);
    });

    tt_operation_test!(test_ptr_to_int_operation, |context, location, values, types| {
        let operation = ptr_to_int(values[1], types.i64, location);

        assert_eq!(operation.name().as_str(), Ok("tt.ptr_to_int"));
        assert_eq!(operation.src(), values[1]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.i64);
        assert_eq!(operation.context(), &context);
    });

    tt_operation_test!(test_bitcast_operation, |_context, location, values, types| {
        let operation = bitcast(values[2], types.i32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.bitcast"));
        assert_eq!(operation.src(), values[2]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.i32);
    });

    tt_operation_test!(test_fp_to_fp_operation, |_context, location, values, types| {
        let operation = fp_to_fp(values[2], Some(RoundingMode::TowardsZero), types.f64, location);

        assert_eq!(operation.name().as_str(), Ok("tt.fp_to_fp"));
        assert_eq!(operation.src(), values[2]);
        assert_eq!(operation.rounding().map(|attribute| attribute.value()), Some(RoundingMode::TowardsZero));
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.f64);
    });

    tt_operation_test!(test_clampf_operation, |_context, location, values, types| {
        let operation = clampf(values[2], values[14], values[2], PropagateNan::All, types.f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.clampf"));
        assert_eq!(operation.x(), values[2]);
        assert_eq!(operation.min(), values[14]);
        assert_eq!(operation.max(), values[2]);
        assert_eq!(operation.propagate_nan().value(), PropagateNan::All);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.f32);
    });

    tt_operation_test!(test_precise_sqrt_operation, |_context, location, values, types| {
        let operation = precise_sqrt(values[2], types.f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.precise_sqrt"));
        assert_eq!(operation.x(), values[2]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.f32);
    });

    tt_operation_test!(test_precise_divf_operation, |_context, location, values, types| {
        let operation = precise_divf(values[2], values[14], types.f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.precise_divf"));
        assert_eq!(operation.x(), values[2]);
        assert_eq!(operation.y(), values[14]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.f32);
    });

    tt_operation_test!(test_mulhiui_operation, |_context, location, values, types| {
        let operation = mulhiui(values[8], values[12], types.i32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.mulhiui"));
        assert_eq!(operation.x(), values[8]);
        assert_eq!(operation.y(), values[12]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.i32);
    });

    tt_operation_test!(test_addptr_operation, |_context, location, values, types| {
        let operation = addptr(values[1], values[8], types.pointer, location);

        assert_eq!(operation.name().as_str(), Ok("tt.addptr"));
        assert_eq!(operation.ptr(), values[1]);
        assert_eq!(operation.offset(), values[8]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.pointer);
    });

    tt_operation_test!(test_load_operation, |_context, location, values, types| {
        let operation = load(
            values[1],
            Some(values[7]),
            Some(values[2]),
            CacheModifier::CacheAll,
            EvictionPolicy::EvictFirst,
            true,
            types.f32,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("tt.load"));
        assert_eq!(operation.ptr(), values[1]);
        assert_eq!(operation.mask(), Some(values[7]));
        assert_eq!(operation.other(), Some(values[2]));
        assert_eq!(operation.cache().value(), CacheModifier::CacheAll);
        assert_eq!(operation.evict().value(), EvictionPolicy::EvictFirst);
        assert!(operation.is_volatile());
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.f32);
    });

    tt_operation_test!(test_store_operation, |_context, location, values, _types| {
        let operation = store(
            values[1],
            values[2],
            Some(values[7]),
            CacheModifier::WriteBack,
            EvictionPolicy::EvictLast,
            true,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("tt.store"));
        assert_eq!(operation.ptr(), values[1]);
        assert_eq!(operation.value(), values[2]);
        assert_eq!(operation.mask(), Some(values[7]));
        assert_eq!(operation.cache().value(), CacheModifier::WriteBack);
        assert_eq!(operation.evict().value(), EvictionPolicy::EvictLast);
        assert!(operation.ignore_cta());
    });

    tt_operation_test!(test_atomic_rmw_operation, |_context, location, values, types| {
        let operation = atomic_rmw(
            RmwOp::FloatAdd,
            values[1],
            values[2],
            Some(values[7]),
            MemSemantic::AcquireRelease,
            MemSyncScope::Cta,
            types.f32,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("tt.atomic_rmw"));
        assert_eq!(operation.atomic_rmw_op().value(), RmwOp::FloatAdd);
        assert_eq!(operation.ptr(), values[1]);
        assert_eq!(operation.val(), values[2]);
        assert_eq!(operation.mask(), Some(values[7]));
        assert_eq!(operation.sem().value(), MemSemantic::AcquireRelease);
        assert_eq!(operation.scope().value(), MemSyncScope::Cta);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.f32);
    });

    tt_operation_test!(test_atomic_cas_operation, |_context, location, values, types| {
        let operation =
            atomic_cas(values[1], values[2], values[14], MemSemantic::Acquire, MemSyncScope::Gpu, types.f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.atomic_cas"));
        assert_eq!(operation.ptr(), values[1]);
        assert_eq!(operation.cmp(), values[2]);
        assert_eq!(operation.val(), values[14]);
        assert_eq!(operation.sem().value(), MemSemantic::Acquire);
        assert_eq!(operation.scope().value(), MemSyncScope::Gpu);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.f32);
    });

    tt_operation_test!(test_splat_operation, |_context, location, values, types| {
        let operation = splat(values[2], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.splat"));
        assert_eq!(operation.src(), values[2]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_unsplat_operation, |_context, location, values, types| {
        let operation = unsplat(values[4], types.f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.unsplat"));
        assert_eq!(operation.src(), values[4]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.f32);
    });

    tt_operation_test!(test_expand_dims_operation, |_context, location, values, types| {
        let operation = expand_dims(values[4], 0, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.expand_dims"));
        assert_eq!(operation.src(), values[4]);
        assert_eq!(operation.axis().signless_value(), 0);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_reshape_operation, |_context, location, values, types| {
        let operation = reshape(values[4], true, true, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.reshape"));
        assert_eq!(operation.src(), values[4]);
        assert!(operation.allow_reorder());
        assert!(operation.efficient_layout());
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_broadcast_operation, |_context, location, values, types| {
        let operation = broadcast(values[4], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.broadcast"));
        assert_eq!(operation.src(), values[4]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_cat_operation, |_context, location, values, types| {
        let operation = cat(values[4], values[11], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.cat"));
        assert_eq!(operation.lhs(), values[4]);
        assert_eq!(operation.rhs(), values[11]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_join_operation, |_context, location, values, types| {
        let operation = join(values[4], values[11], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.join"));
        assert_eq!(operation.lhs(), values[4]);
        assert_eq!(operation.rhs(), values[11]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_split_operation, |_context, location, values, types| {
        let operation = split(values[4], &[types.tensor_f32, types.tensor_f32], location);

        assert_eq!(operation.name().as_str(), Ok("tt.split"));
        assert_eq!(operation.src(), values[4]);
        assert_eq!(operation.out_lhs().r#type(), types.tensor_f32);
        assert_eq!(operation.out_rhs().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_trans_operation, |_context, location, values, types| {
        let operation = trans(values[4], &[0], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.trans"));
        assert_eq!(operation.src(), values[4]);
        assert_eq!(operation.order().values().collect::<Vec<_>>(), vec![0]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_get_program_id_operation, |_context, location, _values, types| {
        let operation = get_program_id(ProgramIdDim::X, types.i32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.get_program_id"));
        assert_eq!(operation.axis().value(), ProgramIdDim::X);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.i32);
    });

    tt_operation_test!(test_get_num_programs_operation, |_context, location, _values, types| {
        let operation = get_num_programs(ProgramIdDim::Y, types.i32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.get_num_programs"));
        assert_eq!(operation.axis().value(), ProgramIdDim::Y);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.i32);
    });

    tt_operation_test!(test_dot_operation, |_context, location, values, types| {
        let operation = dot(values[4], values[11], values[4], InputPrecision::Tf32, 2, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.dot"));
        assert_eq!(operation.a(), values[4]);
        assert_eq!(operation.b(), values[11]);
        assert_eq!(operation.c(), values[4]);
        assert_eq!(operation.input_precision().value(), InputPrecision::Tf32);
        assert_eq!(operation.max_num_imprecise_acc().signless_value(), 2);
        assert_eq!(operation.d().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_dot_scaled_operation, |_context, location, values, types| {
        let operation = dot_scaled(
            values[4],
            values[11],
            values[4],
            Some(values[11]),
            Some(values[4]),
            ScaleDotElemType::E4M3,
            ScaleDotElemType::E5M2,
            true,
            false,
            true,
            types.tensor_f32,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("tt.dot_scaled"));
        assert_eq!(operation.a(), values[4]);
        assert_eq!(operation.b(), values[11]);
        assert_eq!(operation.c(), values[4]);
        assert_eq!(operation.a_scale(), Some(values[11]));
        assert_eq!(operation.b_scale(), Some(values[4]));
        assert_eq!(operation.a_elem_type().value(), ScaleDotElemType::E4M3);
        assert_eq!(operation.b_elem_type().value(), ScaleDotElemType::E5M2);
        assert!(operation.fast_math());
        assert!(!operation.lhs_k_pack());
        assert!(operation.rhs_k_pack());
        assert_eq!(operation.d().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_reduce_operation, |context, location, values, types| {
        let operation = reduce(&[values[4]], 0, &[types.tensor_f32], context.region(), location);

        assert_eq!(operation.name().as_str(), Ok("tt.reduce"));
        assert_eq!(operation.srcs(), vec![values[4]]);
        assert_eq!(operation.axis().signless_value(), 0);
        assert_eq!(operation.combine_op().blocks().count(), 0);
        assert_eq!(operation.result_values()[0].r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_reduce_return_operation, |_context, location, values, _types| {
        let operation = reduce_return(&[values[2]], location);

        assert_eq!(operation.name().as_str(), Ok("tt.reduce.return"));
        assert_eq!(operation.result_values(), vec![values[2]]);
    });

    tt_operation_test!(test_scan_operation, |context, location, values, types| {
        let operation = scan(&[values[4]], 0, true, &[types.tensor_f32], context.region(), location);

        assert_eq!(operation.name().as_str(), Ok("tt.scan"));
        assert_eq!(operation.srcs(), vec![values[4]]);
        assert_eq!(operation.axis().signless_value(), 0);
        assert!(operation.reverse());
        assert_eq!(operation.combine_op().blocks().count(), 0);
        assert_eq!(operation.result_values()[0].r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_scan_return_operation, |_context, location, values, _types| {
        let operation = scan_return(&[values[2]], location);

        assert_eq!(operation.name().as_str(), Ok("tt.scan.return"));
        assert_eq!(operation.result_values(), vec![values[2]]);
    });

    tt_operation_test!(test_map_elementwise_operation, |context, location, values, types| {
        let operation = map_elementwise(&[values[4]], 1, &[types.tensor_f32], context.region(), location);

        assert_eq!(operation.name().as_str(), Ok("tt.map_elementwise"));
        assert_eq!(operation.srcs(), vec![values[4]]);
        assert_eq!(operation.pack().signless_value(), 1);
        assert_eq!(operation.scalar_op().blocks().count(), 0);
        assert_eq!(operation.result_values()[0].r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_map_elementwise_return_operation, |_context, location, values, _types| {
        let operation = map_elementwise_return(&[values[2]], location);

        assert_eq!(operation.name().as_str(), Ok("tt.map_elementwise.return"));
        assert_eq!(operation.result_values(), vec![values[2]]);
    });

    tt_operation_test!(test_extern_elementwise_operation, |_context, location, values, types| {
        let operation =
            extern_elementwise(&[values[2]], "libdevice", "/tmp/libdevice.so", "expf", true, types.f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.extern_elementwise"));
        assert_eq!(operation.srcs(), vec![values[2]]);
        assert_eq!(operation.libname().string().as_str(), Ok("libdevice"));
        assert_eq!(operation.libpath().string().as_str(), Ok("/tmp/libdevice.so"));
        assert_eq!(operation.symbol().string().as_str(), Ok("expf"));
        assert!(operation.pure());
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.f32);
    });

    tt_operation_test!(test_make_range_operation, |_context, location, _values, types| {
        let operation = make_range(0, 4, types.tensor_i32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.make_range"));
        assert_eq!(operation.start().signless_value(), 0);
        assert_eq!(operation.end().signless_value(), 4);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_i32);
    });

    tt_operation_test!(test_elementwise_inline_asm_operation, |_context, location, values, types| {
        let operation =
            elementwise_inline_asm(&[values[2]], "mov.u32 $0, $1;", "=f,f", true, 1, &[types.f32], location);

        assert_eq!(operation.name().as_str(), Ok("tt.elementwise_inline_asm"));
        assert_eq!(operation.asm_string().string().as_str(), Ok("mov.u32 $0, $1;"));
        assert_eq!(operation.constraints().string().as_str(), Ok("=f,f"));
        assert!(operation.pure());
        assert_eq!(operation.packed_element().signless_value(), 1);
        assert_eq!(operation.args(), vec![values[2]]);
        assert_eq!(operation.result_values()[0].r#type(), types.f32);
    });

    tt_operation_test!(test_histogram_operation, |_context, location, values, types| {
        let operation = histogram(values[5], Some(values[6]), types.tensor_i32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.histogram"));
        assert_eq!(operation.src(), values[5]);
        assert_eq!(operation.mask(), Some(values[6]));
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_i32);
    });

    tt_operation_test!(test_gather_operation, |_context, location, values, types| {
        let operation = gather(values[4], values[5], 0, true, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.gather"));
        assert_eq!(operation.src(), values[4]);
        assert_eq!(operation.indices(), values[5]);
        assert_eq!(operation.axis().signless_value(), 0);
        assert!(operation.efficient_layout());
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_print_operation, |_context, location, values, _types| {
        let operation = print(&[values[8], values[2]], "value", false, &[1, 0], location);

        assert_eq!(operation.name().as_str(), Ok("tt.print"));
        assert_eq!(operation.prefix().string().as_str(), Ok("value"));
        assert!(!operation.hex());
        assert_eq!(operation.args(), vec![values[8], values[2]]);
        assert_eq!(operation.is_signed().values().collect::<Vec<_>>(), vec![1, 0]);
    });

    tt_operation_test!(test_assert_operation, |_context, location, values, _types| {
        let operation = assert(values[7], "failed", location);

        assert_eq!(operation.name().as_str(), Ok("tt.assert"));
        assert_eq!(operation.condition(), values[7]);
        assert_eq!(operation.message().string().as_str(), Ok("failed"));
    });

    tt_operation_test!(test_make_tensor_descriptor_operation, |_context, location, values, types| {
        let operation = make_tensor_descriptor(
            values[1],
            &[values[0], values[15]],
            &[values[0], values[15]],
            PaddingOption::Zero,
            types.tensor_desc,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("tt.make_tensor_descriptor"));
        assert_eq!(operation.base(), values[1]);
        assert_eq!(operation.shape(), vec![values[0], values[15]]);
        assert_eq!(operation.strides(), vec![values[0], values[15]]);
        assert_eq!(operation.padding().value(), PaddingOption::Zero);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_desc);
    });

    tt_operation_test!(test_call_operation, |_context, location, values, types| {
        let operation = call("callee", &[values[2]], &[types.f32], location);

        assert_eq!(operation.name().as_str(), Ok("tt.call"));
        assert_eq!(operation.callee().reference().as_str(), Ok("callee"));
        assert_eq!(CallOperation::operands(&operation), vec![values[2]]);
        assert_eq!(operation.result_values()[0].r#type(), types.f32);
    });

    tt_operation_test!(test_func_operation, |context, location, _values, types| {
        let operation = func(
            "kernel",
            types.function,
            Some("private"),
            Some(context.array_attribute(&[] as &[AttributeRef])),
            Some(context.array_attribute(&[] as &[AttributeRef])),
            context.region(),
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("tt.func"));
        assert_eq!(operation.sym_name().string().as_str(), Ok("kernel"));
        assert_eq!(operation.function_type().r#type(), types.function);
        assert_eq!(operation.sym_visibility().and_then(|attribute| attribute.string().as_str().ok()), Some("private"));
        assert!(operation.arg_attrs().unwrap().is_empty());
        assert!(operation.res_attrs().unwrap().is_empty());
        assert_eq!(operation.body().blocks().count(), 0);
    });

    tt_operation_test!(test_return_operation, |_context, location, values, _types| {
        let operation = r#return(&[values[2]], location);

        assert_eq!(operation.name().as_str(), Ok("tt.return"));
        assert_eq!(operation.srcs(), vec![values[2]]);
    });

    tt_operation_test!(test_descriptor_load_operation, |_context, location, values, types| {
        let operation = descriptor_load(
            values[9],
            &[values[0], values[15]],
            CacheModifier::CacheGlobal,
            EvictionPolicy::Normal,
            types.tensor_f32,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("tt.descriptor_load"));
        assert_eq!(operation.desc(), values[9]);
        assert_eq!(operation.indices(), vec![values[0], values[15]]);
        assert_eq!(operation.cache().value(), CacheModifier::CacheGlobal);
        assert_eq!(operation.evict().value(), EvictionPolicy::Normal);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_descriptor_store_operation, |_context, location, values, _types| {
        let operation = descriptor_store(values[9], values[4], &[values[0], values[15]], location);

        assert_eq!(operation.name().as_str(), Ok("tt.descriptor_store"));
        assert_eq!(operation.desc(), values[9]);
        assert_eq!(operation.src(), values[4]);
        assert_eq!(operation.indices(), vec![values[0], values[15]]);
    });

    tt_operation_test!(test_descriptor_reduce_operation, |_context, location, values, _types| {
        let operation =
            descriptor_reduce(DescriptorReduceKind::Add, values[9], values[4], &[values[0], values[15]], location);

        assert_eq!(operation.name().as_str(), Ok("tt.descriptor_reduce"));
        assert_eq!(operation.kind().value(), DescriptorReduceKind::Add);
        assert_eq!(operation.desc(), values[9]);
        assert_eq!(operation.src(), values[4]);
        assert_eq!(operation.indices(), vec![values[0], values[15]]);
    });

    tt_operation_test!(test_descriptor_gather_operation, |_context, location, values, types| {
        let operation = descriptor_gather(values[9], values[5], values[0], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("tt.descriptor_gather"));
        assert_eq!(operation.desc(), values[9]);
        assert_eq!(operation.x_offsets(), values[5]);
        assert_eq!(operation.y_offset(), values[0]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    tt_operation_test!(test_descriptor_scatter_operation, |_context, location, values, _types| {
        let operation = descriptor_scatter(values[9], values[5], values[0], values[4], location);

        assert_eq!(operation.name().as_str(), Ok("tt.descriptor_scatter"));
        assert_eq!(operation.desc(), values[9]);
        assert_eq!(operation.x_offsets(), values[5]);
        assert_eq!(operation.y_offset(), values[0]);
        assert_eq!(operation.src(), values[4]);
    });
}
