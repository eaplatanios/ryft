#include "shape.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

namespace {

template <typename TypeT>
bool isType(MlirType type) {
  return type.ptr != nullptr && llvm::isa<TypeT>(unwrap(type));
}

}  // namespace

bool mlirTypeIsAShapeShapeType(MlirType type) {
  return isType<mlir::shape::ShapeType>(type);
}

MlirType mlirShapeShapeTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::shape::ShapeType::get(unwrap(context)));
}

bool mlirTypeIsAShapeSizeType(MlirType type) {
  return isType<mlir::shape::SizeType>(type);
}

MlirType mlirShapeSizeTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::shape::SizeType::get(unwrap(context)));
}

bool mlirTypeIsAShapeValueShapeType(MlirType type) {
  return isType<mlir::shape::ValueShapeType>(type);
}

MlirType mlirShapeValueShapeTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::shape::ValueShapeType::get(unwrap(context)));
}

bool mlirTypeIsAShapeWitnessType(MlirType type) {
  return isType<mlir::shape::WitnessType>(type);
}

MlirType mlirShapeWitnessTypeGet(MlirContext context) {
  if (context.ptr == nullptr) {
    return {nullptr};
  }
  return wrap(mlir::shape::WitnessType::get(unwrap(context)));
}
