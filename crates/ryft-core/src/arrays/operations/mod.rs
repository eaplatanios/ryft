use ryft_macros::Operation;

pub use crate::arrays::{DimensionType, DimensionValue};
pub use crate::operations::constants::ConstantOperation;
pub use crate::operations::dimensions::{
    DimensionAddOperation, DimensionDivFloorOperation, DimensionMaxOperation, DimensionMinOperation,
    DimensionMulOperation, DimensionPowOperation, DimensionRemOperation, DimensionRequirementOperation,
    DimensionSaturatingSubOperation, DimensionSubOperation,
};
pub use crate::programs::values::Value;
pub use crate::tracing::TracingContext;

pub mod dimensions;

/// [`Operation`](crate::Operation) family used for staged [`DimensionValue`] [`Program`](crate::Program)s.
#[derive(Clone, Debug, Operation)]
pub enum DimensionOperation<V: Value<Type = DimensionType>> {
    Constant(ConstantOperation<V>),
    Add(DimensionAddOperation),
    Sub(DimensionSubOperation),
    SaturatingSub(DimensionSaturatingSubOperation),
    Mul(DimensionMulOperation),
    Pow(DimensionPowOperation),
    DivFloor(DimensionDivFloorOperation),
    Rem(DimensionRemOperation),
    Min(DimensionMinOperation),
    Max(DimensionMaxOperation),
    Requirement(DimensionRequirementOperation),
}

/// [`TracingContext`] over [`DimensionValue`]s and [`DimensionOperation`]s.
pub type DimensionTracingContext = TracingContext<DimensionValue, DimensionOperation<DimensionValue>>;
