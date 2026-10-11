/// StableHLO lowering of broadcasting operations.
pub(crate) mod broadcasting;
/// StableHLO lowering of concatenation operations. Concatenation result refinements resize bounded storage, so
/// reconciling a different result extent or allocation bound requires finite input and output allocation bounds.
pub(crate) mod concatenation;
/// StableHLO lowering of element type conversion operations.
pub(crate) mod conversions;
/// StableHLO lowering of gathering operations.
pub(crate) mod gathering;
/// StableHLO lowering of memory placement operations.
pub(crate) mod memory;
/// StableHLO lowering of padding operations. Padding result refinements resize bounded storage, so reconciling a
/// different result extent or allocation bound requires finite input and output allocation bounds.
pub(crate) mod padding;
/// StableHLO lowering of reshaping operations.
pub(crate) mod reshaping;
/// StableHLO lowering of reversing operations.
pub(crate) mod reversing;
/// StableHLO lowering of scattering operations.
pub(crate) mod scattering;
/// StableHLO lowering of slicing operations.
pub(crate) mod slicing;
/// StableHLO lowering of transposition operations.
pub(crate) mod transposition;
