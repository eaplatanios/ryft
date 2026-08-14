use std::collections::BTreeSet;
use std::fmt::{Debug, Display};

use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayType, DataType, Dimension, LogicalMesh, MeshAxisType, RaggedArrayBatchingPolicy,
    Shape, Sharding, ShardingDimension,
};
use crate::axes::Axis;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::manipulation::conversion::{ConvertElementType, ConvertElementTypeOperation};
use crate::operations::manipulation::transposition::Transpose;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};
use crate::tracing::{Tracer, TracingContext};

mod batching;
mod differentiation;
mod dimensions;
mod inference;
mod operation;

use dimensions::lift_output_sharding;

pub use dimensions::{
    DotDimensionNumbers, adjoint_dimensions_for_left_dot, adjoint_dimensions_for_right_dot, lhs_result_axes,
    lift_dot_dimensions, rhs_result_axes,
};
pub use inference::DOT_OPERATION_NAME;
pub(crate) use inference::dot_abstract;
pub use operation::{Dot, DotOperation, DotOps};

#[cfg(test)]
mod tests;
