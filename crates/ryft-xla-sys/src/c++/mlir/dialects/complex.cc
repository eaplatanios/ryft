#include "complex.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Complex/IR/Complex.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Complex, complex,
                                      mlir::complex::ComplexDialect)

bool mlirAttributeIsAComplex(MlirAttribute attribute) {
  return llvm::isa<mlir::complex::NumberAttr>(unwrap(attribute));
}

MlirAttribute mlirComplexAttrDoubleGet(MlirContext context, MlirType type,
                                       double real, double imaginary) {
  return wrap(mlir::complex::NumberAttr::get(
      llvm::cast<mlir::ComplexType>(unwrap(type)), real, imaginary));
}

MlirAttribute mlirComplexAttrDoubleGetChecked(MlirLocation location,
                                              MlirType type, double real,
                                              double imaginary) {
  return wrap(mlir::complex::NumberAttr::getChecked(
      unwrap(location), llvm::cast<mlir::ComplexType>(unwrap(type)), real,
      imaginary));
}

double mlirComplexAttrGetRealDouble(MlirAttribute attribute) {
  return llvm::cast<mlir::complex::NumberAttr>(unwrap(attribute))
      .getReal()
      .convertToDouble();
}

double mlirComplexAttrGetImagDouble(MlirAttribute attribute) {
  return llvm::cast<mlir::complex::NumberAttr>(unwrap(attribute))
      .getImag()
      .convertToDouble();
}

MlirTypeID mlirComplexAttrGetTypeID(void) {
  return wrap(mlir::complex::NumberAttr::getTypeID());
}
