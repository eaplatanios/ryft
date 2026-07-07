/// Marker for [`Operation`](crate::Operation) payloads that are carried by the operation itself. Use [`Captured`]
/// when the payload is embedded in the operation object and should be treated as a closed-over part of the staged
/// instruction. An example is an ordinary [`ConstantOperation`](crate::ConstantOperation) payload, whose value is a
/// closed-over constant rather than a runtime input.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Captured;

/// Marker for [`Operation`](crate::Operation) payloads that are already inputs in the active interpretation domain.
/// Use [`Input`] when the payload is already a value in the same interpretation domain as the operation result.
/// In [`StagingContext`](crate::StagingContext) this usually means the payload is already represented by a
/// [`Tracer`](crate::Tracer), and so interpretation should validate that it belongs to the active builder and forward
/// or lower it through ordinary input-consuming operations.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Input;
