use std::collections::BTreeSet;

use crate::batching::{ArrayBatch, BatchingTracer};
use crate::broadcasting::Broadcastable;
use crate::captures::CaptureReference;
use crate::contexts::Context;
use crate::differentiation::{DifferentiableType, DifferentiationDual, DifferentiationTracer};
use crate::macros::check_count;
use crate::partial::{PartialEvaluationValue, PartialTracer, PartialValue};
use crate::programs::ProgramError;
use crate::programs::operations::Operation;
use crate::programs::types::{TypeError, Typed};
use crate::tracing::Tracer;
use crate::types::{ArrayType, DataType};

// TODO(eaplatanios): Review this file.

/// Elementwise pairwise comparison operations and capability traits.
pub mod compare;

/// Elementwise complex-number construction, conjugation, and part-extraction operations and capability traits.
pub mod complex;

/// Type-driven constant operations and capability traits.
pub mod constants;

/// Higher-order control-flow operations and capability traits.
pub mod control_flow;

/// Debugging operations with observable effects (e.g., printing values from inside programs).
pub mod debugging;

/// Differentiation-control operations and capability traits.
pub mod differentiation;

/// Elementwise logical operations and capability traits.
pub mod logical;

/// Array shape and axis manipulation operations and capability traits.
pub mod manipulation;

/// Elementwise arithmetic and trigonometric math operations and capability traits.
pub mod math;

/// Shared marker types for operations with payload-dependent interpretation.
pub mod payloads;

/// Sharding-related operations (e.g., resharding and propagation hints) and capability traits.
pub mod sharding;

/// Value tagging — attaching a string key to a value in a program (consumed by, e.g., rematerialization policies).
pub mod tag;

// TODO(eaplatanios): We should be importing specific symbols here.
// The fallible `Add`/`Sub`/`Mul`/`Div`/`Neg` capability traits are intentionally not re-exported at this level so
// they do not shadow their `std::ops` counterparts; reach them through `crate::operations::math` instead.
pub use compare::*;
pub use constants::*;
pub use control_flow::*;
pub use debugging::{PRINT_OPERATION_NAME, Print, PrintOperation};
pub use differentiation::*;
pub use logical::*;
pub use manipulation::*;
pub use math::*;
pub use sharding::*;
pub use tag::{TAG_OPERATION_NAME, Tag, TagOperation};

/// Represents [`Operation`]s that operate elementwise on arrays and that support _broadcasting_ semantics.
/// [`ElementwiseOperation`] captures the shared type inference behavior of elementwise array operations:
/// implementations declare their fixed input count, while the default type inference implementation checks
/// the input count, broadcasts all input [`ArrayType`]s while tolerating shardings that differ only by
/// [`Sharding::varying_manual_axes`](crate::Sharding::varying_manual_axes).
pub trait ElementwiseOperation: Operation<ArrayType> {
    /// Returns the number of input arrays consumed by this elementwise [`Operation`].
    fn input_count(&self) -> usize;

    /// Infers the broadcasted output [`ArrayType`] for this elementwise [`Operation`]. Operations whose output sharding
    /// does not follow plain broadcasting semantics (e.g., [`MulOperation`], which is bilinear in its operands and
    /// combines their reduction state accordingly) must override this function, typically using
    /// [`broadcast_output_type`](Self::broadcast_output_type) for the data type, shapes, and placement, and layering
    /// their own sharding rule on top.
    #[inline]
    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, self.input_count(), TypeError);
        Ok(vec![self.broadcast_output_type(input_types)?])
    }

    /// Broadcasts the elementwise operands into a single output [`ArrayType`], tolerating shardings that differ only by
    /// their [`Sharding::varying_manual_axes`](crate::Sharding::varying_manual_axes). Ryft keeps generic [`ArrayType`]
    /// broadcasting conservative, and so this function retries inference after erasing only the varying-manual-axis
    /// (VMA) metadata and then restores the union of that metadata on the result, instead of weakening generic
    /// [`ArrayType`] broadcasting everywhere.
    ///
    /// This is effectively a shared helper function for the default [`infer_output_types`](Self::infer_output_types)
    /// implementation and for operations that override that default to layer extra sharding rules on top of the
    /// broadcasted placement (e.g., [`MulOperation`]'s bilinear reduction-state rule).
    fn broadcast_output_type(&self, input_types: &[ArrayType]) -> Result<ArrayType, TypeError> {
        match ArrayType::broadcasted(input_types) {
            Ok(output) => Ok(output),
            Err(_) => {
                let original_varying_manual_axes = input_types
                    .iter()
                    .filter_map(|input_type| input_type.sharding.as_ref())
                    .flat_map(|sharding| sharding.varying_manual_axes.iter().cloned())
                    .collect::<BTreeSet<_>>();
                let mut input_types = input_types.to_vec();
                for sharding in input_types.iter_mut().filter_map(|input_type| input_type.sharding.as_mut()) {
                    sharding.varying_manual_axes.clear();
                }
                let mut output = ArrayType::broadcasted(input_types.as_slice()).map_err(|_| TypeError {
                    message: format!("'{}' input types are not broadcast-compatible", self.name()),
                })?;
                if let Some(sharding) = &mut output.sharding {
                    sharding.varying_manual_axes = original_varying_manual_axes;
                }
                Ok(output)
            }
        }
    }
}

/// Represents [`Type`](crate::Type)s and [`Value`](crate::Value)s that have a Boolean counterpart and that may carry
/// a scalar Rust Boolean. [`BooleanLike`] is the shared contract between predicate-producing and predicate-consuming
/// operations:
///
/// - **Predicate-Producing Operations (e.g., [`CompareOperation`]):** Call [`as_boolean`](Self::as_boolean)
///   on *type metadata* to infer their output types from their broadcasted input types. For type metadata (e.g.,
///   [`DataType`] and [`ArrayType`]), the Boolean counterpart keeps the same structural metadata (e.g., shape, layout,
///   and sharding) but uses a Boolean element data type.
/// - **Predicate-Consuming Operations (e.g., [`ConditionOperation`] and [`WhileOperation`]):** Call
///   [`boolean`](Self::boolean) on *values* to extract the concrete scalar Rust Boolean that drives branching
///   or selection.
///
/// For values, [`as_boolean`](Self::as_boolean) reinterprets the carried payload as a Boolean value: zero maps to
/// `false` and any non-zero payload maps to `true`. Values that carry no concrete payload (e.g., staged tracers and
/// [`CaptureReference`]s) cannot reinterpret anything and return themselves unchanged. Similarly,
/// [`boolean`](Self::boolean) errors for type metadata and for staged values because they carry no
/// concrete payload to decode.
pub trait BooleanLike {
    /// Returns the Boolean counterpart of this instance. For type metadata this is the same structural metadata with
    /// a Boolean data type, and for values this is the value with its payload reinterpreted as Boolean (i.e., zero
    /// maps to `false` and any non-zero payload maps to `true`).
    fn as_boolean(&self) -> Self;

    /// Extracts the scalar Rust Boolean value represented by this instance when there is one. For scalar values zero
    /// gets interpreted as `false` while non-zero values get interpreted as `true`, while for array values this
    /// requires a rank-0 Boolean-typed payload. Type metadata and staged values (e.g., tracers) error because they
    /// carry no concrete payload.
    fn boolean(&self) -> Result<bool, ProgramError>;
}

impl BooleanLike for DataType {
    #[inline]
    fn as_boolean(&self) -> Self {
        DataType::Boolean
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        // `DataType` is type metadata and carries no concrete payload to decode.
        Err(ProgramError::Concretization {
            message: format!("cannot extract a concrete boolean from a data type instance ({self})"),
        })
    }
}

impl BooleanLike for ArrayType {
    #[inline]
    fn as_boolean(&self) -> Self {
        Self { data_type: DataType::Boolean, ..self.clone() }
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        // `ArrayType` is only abstract staged-program metadata. It satisfies generic operation-enum bounds for
        // transform composition, but it never contains the concrete boolean needed to choose a branch.
        Err(ProgramError::Concretization {
            message: format!("cannot extract a concrete boolean from an array type instance ({self})"),
        })
    }
}

impl<C: Context> BooleanLike for Tracer<C> {
    #[inline]
    fn as_boolean(&self) -> Self {
        // Returns this `Tracer` unchanged. Tracers carry no concrete payload to reinterpret, and a staged Boolean
        // reinterpretation must be expressed explicitly in the traced program (e.g., via a comparison against zero)
        // rather than implicitly through this trait.
        self.clone()
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        Err(ProgramError::Concretization { message: "cannot extract a concrete boolean from a tracer".to_string() })
    }
}

impl BooleanLike for CaptureReference<ArrayType> {
    #[inline]
    fn as_boolean(&self) -> Self {
        // Returns this `CaptureReference` unchanged. A captured constant is a reference into a side table,
        // not the concrete value itself, so there is no payload to reinterpret here.
        self.clone()
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        // A captured constant is a reference into a side table, not the concrete predicate value itself. Control-flow
        // staging must keep predicates in the IR or add a transform-specific rule instead of trying to branch here.
        Err(ProgramError::Concretization {
            message: "cannot extract a concrete boolean from a captured constant reference".to_string(),
        })
    }
}

// A partial-evaluation value's Boolean view uses its known payload's: a known value reinterprets (and decodes) the
// carried known-side value, so branching on a known value in a closure succeeds exactly when the known-side inner
// context is eager, while an unknown value names a residual program variable that carries no concrete payload and so
// returns itself unchanged from `as_boolean` and errors from `boolean`. This is what lets host control flow branch on
// known values while partial evaluation is in progress.
impl<C: Context<Value: BooleanLike, Type: BooleanLike>> BooleanLike for PartialTracer<C> {
    #[inline]
    fn as_boolean(&self) -> Self {
        // Unknown and poisoned values carry no concrete payload to reinterpret and return themselves unchanged.
        match self.value() {
            Ok(value) => match value.value() {
                PartialValue::Known(known) => {
                    PartialTracer::new(self.context().clone(), PartialEvaluationValue::known(known.as_boolean()))
                }
                PartialValue::Unknown(_) => self.clone(),
            },
            Err(_) => self.clone(),
        }
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        // A poisoned value surfaces its deferred error here, since branching on it cannot proceed anyway.
        match self.value()?.value() {
            PartialValue::Known(known) => known.boolean(),
            PartialValue::Unknown(_) => Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from an unknown partial-evaluation value".to_string(),
            }),
        }
    }
}

// A batch-carrying value's Boolean view uses its packed value's Boolean view. Branching on it via `boolean()` succeeds
// only for a *replicated* value whose packed value is concrete.  A batched value has one Boolean per item and cannot
// drive a single branch, and a staged value carries no concrete payload.
impl<C: Context<Type = ArrayType, Value: BooleanLike>> BooleanLike for BatchingTracer<C> {
    #[inline]
    fn as_boolean(&self) -> Self {
        let r#type = self.batch().r#type().as_boolean();
        let batch = ArrayBatch::new(r#type, self.batch().value().as_boolean(), self.batch().batch_axis()).unwrap();
        BatchingTracer::new(self.context().clone(), batch)
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        if !self.batch().batch_axis().is_replicated() {
            return Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a batched value".to_string(),
            });
        }
        self.batch().value().boolean()
    }
}

// TODO(eaplatanios): Review this implementation.
// A dual's Boolean view uses its primal's: `as_boolean` reinterprets the primal with a
// structural zero tangent, and `boolean` decodes the primal — so branching on a dual in a
// closure succeeds exactly when the primal is a concrete (eager) value and errors when it is a staged tracer.
impl<C: Context<Type: DifferentiableType, Value: BooleanLike>> BooleanLike for DifferentiationTracer<C> {
    #[inline]
    fn as_boolean(&self) -> Self {
        let primal = self.primal().as_boolean();
        Self::new(DifferentiationDual::new_with_zero_tangent(primal), self.context().clone())
    }

    #[inline]
    fn boolean(&self) -> Result<bool, ProgramError> {
        self.primal().boolean()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;

    use crate::programs::regions::RegionInterface;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    #[test]
    fn elementwise_array_operation() {
        #[derive(Clone, Debug)]
        struct TestElementwiseArrayOperation {
            input_count: usize,
        }

        impl Operation<ArrayType> for TestElementwiseArrayOperation {
            #[inline]
            fn name(&self) -> &'static str {
                "elementwise_test"
            }

            #[inline]
            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                ElementwiseOperation::infer_output_types(self, input_types)
            }
        }

        impl ElementwiseOperation for TestElementwiseArrayOperation {
            #[inline]
            fn input_count(&self) -> usize {
                self.input_count
            }
        }

        let operation = TestElementwiseArrayOperation { input_count: 1 };
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[input_type.clone()], &[]),
            Ok(vec![input_type])
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[], &[]),
            Err(TypeError { message: "expected 1 input but got 0".to_string() }),
        );

        let operation = TestElementwiseArrayOperation { input_count: 3 };
        let output = Operation::<ArrayType>::infer_output_types(
            &operation,
            &[
                ArrayType::scalar(DataType::F32),
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)])),
                ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(3)])),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(output, vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))],);

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let first = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            )
            .unwrap();
        let second = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["y"],
                )
                .unwrap(),
            )
            .unwrap();
        let third = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["z"],
                )
                .unwrap(),
            )
            .unwrap();
        let output = Operation::<ArrayType>::infer_output_types(&operation, &[first, second, third], &[]).unwrap();
        assert_eq!(
            output[0].sharding().unwrap().varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string(), "z".to_string()]),
        );

        // Dynamic dimensions flow through elementwise congruence when they match exactly, while static-vs-dynamic
        // mismatches are rejected.
        let operation = TestElementwiseArrayOperation { input_count: 2 };
        let dynamic_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Dynamic(None), Size::Static(3)]));
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(&operation, &[dynamic_type.clone(), dynamic_type.clone()], &[]),
            Ok(vec![dynamic_type.clone()]),
        );
        assert_eq!(
            Operation::<ArrayType>::infer_output_types(
                &operation,
                &[dynamic_type, ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(3)]))],
                &[],
            ),
            Err(TypeError { message: "'elementwise_test' input types are not broadcast-compatible".to_string() }),
        );
    }
}
