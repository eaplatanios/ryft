#include "arith.h"

#include <cstdint>
#include <optional>

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"

namespace {

template <typename AttributeT>
bool isAttribute(MlirAttribute attribute) {
  return attribute.ptr != nullptr && llvm::isa<AttributeT>(unwrap(attribute));
}

}  // namespace

bool mlirAttributeIsAArithAtomicRmwKindAttr(MlirAttribute attribute) {
  return isAttribute<mlir::arith::AtomicRMWKindAttr>(attribute);
}

MlirAttribute mlirArithAtomicRmwKindAttrGet(MlirContext context, uint64_t value) {
  std::optional<mlir::arith::AtomicRMWKind> enumValue = mlir::arith::symbolizeAtomicRMWKind(value);
  if (!enumValue) {
    return {nullptr};
  }
  return wrap(mlir::arith::AtomicRMWKindAttr::get(unwrap(context), *enumValue));
}

uint64_t mlirArithAtomicRmwKindAttrGetValue(MlirAttribute attribute) {
  if (attribute.ptr == nullptr) {
    return 0;
  }
  auto typedAttribute = llvm::dyn_cast<mlir::arith::AtomicRMWKindAttr>(unwrap(attribute));
  return typedAttribute ? static_cast<uint64_t>(typedAttribute.getValue()) : 0;
}

bool mlirAttributeIsAArithFastMathFlagsAttr(MlirAttribute attribute) {
  return isAttribute<mlir::arith::FastMathFlagsAttr>(attribute);
}

MlirAttribute mlirArithFastMathFlagsAttrGet(MlirContext context, uint32_t value) {
  std::optional<mlir::arith::FastMathFlags> flags = mlir::arith::symbolizeFastMathFlags(value);
  if (!flags) {
    return {nullptr};
  }
  return wrap(mlir::arith::FastMathFlagsAttr::get(unwrap(context), *flags));
}

uint32_t mlirArithFastMathFlagsAttrGetValue(MlirAttribute attribute) {
  if (attribute.ptr == nullptr) {
    return 0;
  }
  auto typedAttribute = llvm::dyn_cast<mlir::arith::FastMathFlagsAttr>(unwrap(attribute));
  return typedAttribute ? static_cast<uint32_t>(typedAttribute.getValue()) : 0;
}

bool mlirAttributeIsAArithIntegerOverflowFlagsAttr(MlirAttribute attribute) {
  return isAttribute<mlir::arith::IntegerOverflowFlagsAttr>(attribute);
}

MlirAttribute mlirArithIntegerOverflowFlagsAttrGet(MlirContext context, uint32_t value) {
  std::optional<mlir::arith::IntegerOverflowFlags> flags = mlir::arith::symbolizeIntegerOverflowFlags(value);
  if (!flags) {
    return {nullptr};
  }
  return wrap(mlir::arith::IntegerOverflowFlagsAttr::get(unwrap(context), *flags));
}

uint32_t mlirArithIntegerOverflowFlagsAttrGetValue(MlirAttribute attribute) {
  if (attribute.ptr == nullptr) {
    return 0;
  }
  auto typedAttribute = llvm::dyn_cast<mlir::arith::IntegerOverflowFlagsAttr>(unwrap(attribute));
  return typedAttribute ? static_cast<uint32_t>(typedAttribute.getValue()) : 0;
}

bool mlirAttributeIsAArithRoundingModeAttr(MlirAttribute attribute) {
  return isAttribute<mlir::arith::RoundingModeAttr>(attribute);
}

MlirAttribute mlirArithRoundingModeAttrGet(MlirContext context, uint32_t value) {
  std::optional<mlir::arith::RoundingMode> enumValue = mlir::arith::symbolizeRoundingMode(value);
  if (!enumValue) {
    return {nullptr};
  }
  return wrap(mlir::arith::RoundingModeAttr::get(unwrap(context), *enumValue));
}

uint32_t mlirArithRoundingModeAttrGetValue(MlirAttribute attribute) {
  if (attribute.ptr == nullptr) {
    return 0;
  }
  auto typedAttribute = llvm::dyn_cast<mlir::arith::RoundingModeAttr>(unwrap(attribute));
  return typedAttribute ? static_cast<uint32_t>(typedAttribute.getValue()) : 0;
}
