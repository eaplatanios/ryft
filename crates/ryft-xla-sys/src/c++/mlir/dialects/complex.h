#pragma once

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Complex, complex);

/// Checks whether the given attribute is a complex attribute.
MLIR_CAPI_EXPORTED bool mlirAttributeIsAComplex(MlirAttribute attribute);

/// Creates a complex attribute with double-precision real and imaginary values.
MLIR_CAPI_EXPORTED MlirAttribute mlirComplexAttrDoubleGet(MlirContext context,
                                                          MlirType type,
                                                          double real,
                                                          double imaginary);

/// Creates a complex attribute, returning a null attribute when construction
/// fails.
MLIR_CAPI_EXPORTED MlirAttribute mlirComplexAttrDoubleGetChecked(
    MlirLocation location, MlirType type, double real, double imaginary);

/// Returns the real value stored in a complex attribute as a double.
MLIR_CAPI_EXPORTED double mlirComplexAttrGetRealDouble(MlirAttribute attribute);

/// Returns the imaginary value stored in a complex attribute as a double.
MLIR_CAPI_EXPORTED double mlirComplexAttrGetImagDouble(MlirAttribute attribute);

/// Returns the type ID of a complex attribute.
MLIR_CAPI_EXPORTED MlirTypeID mlirComplexAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif
