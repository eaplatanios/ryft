use std::borrow::Cow;
use std::cell::Cell;
use std::fmt::Debug;
use std::rc::Rc;

use crate::programs::{AtomId, Typed, Value};

#[cfg(doc)]
use crate::partial::contexts::PartialEvaluationContext;

#[cfg(doc)]
use crate::partial::evaluations::PartialEvaluation;

#[cfg(doc)]
use crate::partial::partitions::PartitionedProgram;

#[cfg(doc)]
use crate::programs::{Program, Type};

/// State of a [`Value`] during partial evaluation. A [`PartialValue`] is the value domain the partial context
/// interprets a [`Program`] over. Every [`Atom`](crate::Atom) and every intermediate result is either
/// [`Known`](Self::Known) (i.e., a concrete value available now) or [`Unknown`](Self::Unknown) (i.e., only its
/// [`Type`] is available until the residual program runs). For more information on partial evaluation, refer to
/// the documentation of [`Program::partially_evaluate`].
#[derive(Clone, Debug)]
pub enum PartialValue<V: Value> {
    /// [`Value`] that is fully known at partial-evaluation time and can be folded forward.
    Known(V),

    /// [`Value`] that is not known until the residual program runs and only its [`Type`] is known.
    Unknown(V::Type),
}

impl<V: Value> PartialValue<V> {
    /// Returns `true` if this value is [`Known`](Self::Known).
    #[inline]
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Returns `true` if this value is [`Unknown`](Self::Unknown).
    #[inline]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }

    /// Returns the underlying concrete value when this is [`Known`](Self::Known) and [`None`] otherwise.
    #[inline]
    pub fn as_known(&self) -> Option<&V> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unknown(_) => None,
        }
    }
}

impl<V: Value> Typed for PartialValue<V> {
    type Type = V::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, V::Type> {
        match self {
            Self::Known(value) => value.r#type(),
            Self::Unknown(r#type) => Cow::Borrowed(r#type),
        }
    }
}

/// Represents the way in which a [`PartialEvaluationValue`] is represented when _residual_ work depends on it.
/// A [`PartialValue`] only records whether a value is known now or unknown until a residual [`Program`] runs.
/// [`PartialValueMaterialization`] records how that value is represented at the residual boundary. Each materialization
/// lives in a slot shared by every clone of one logical [`PartialEvaluationValue`], and so the residual
/// [`Atom`](crate::Atom) assigned when a known value is first materialized (as a residual input or an inline residual
/// constant) is visible to every later consumer of the same value, which reuses that atom instead of materializing the
/// value again. By contrast, [`Variable`](Self::Variable) values were *created* in the residual program and so always
/// carry their residual atom.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PartialValueMaterialization {
    /// Known value with no residual materialization decision yet. If residual work depends on it, the corresponding
    /// [`PartialEvaluationContext`] will materialize it as a fresh residual input.
    Undecided,

    /// Known value that should be materialized as a residual program input.
    Input {
        /// Residual input atom assigned when this value was first materialized, if it was, so that later consumers
        /// of the same value reuse it. When absent, the value has not been materialized yet, and the first residualized
        /// consumer creates a fresh residual input and records it here.
        residual_atom: Option<AtomId>,
    },

    /// Known value that should be materialized as an inline residual program constant.
    Constant {
        /// Residual constant atom assigned when this value was first materialized, if it was, so that later consumers
        /// of the same value reuse it. When absent, the value has not been materialized yet, and the first residualized
        /// consumer creates a fresh residual constant and records it here.
        residual_atom: Option<AtomId>,
    },

    /// Unknown value already represented as a residual program variable.
    Variable {
        /// Atom in the residual program that carries this value. Residual operations consume it directly, and so it
        /// is not optional.
        residual_atom: AtomId,
    },
}

/// Represents the [`Value`] type used by [`PartialEvaluationContext`]s while partially evaluating [`Program`]s.
#[derive(Clone)]
pub struct PartialEvaluationValue<V: Value> {
    /// Underlying [`PartialValue`] that represents the abstract known/unknown classification of the value.
    pub(super) value: PartialValue<V>,

    /// [`PartialValueMaterialization`] that describes how the underlying value is represented at the residual program
    /// boundary. This is deliberately separate from the underlying [`PartialValue`] because it answers a different
    /// question. A [`Known`](PartialValue::Known) value can still be consumed by residual work, materializing as a
    /// residual input or an inline residual constant according to its [`PartialValueMaterialization`], while an
    /// [`Unknown`](PartialValue::Unknown) value is always represented by a residual program variable that already
    /// exists. The slot is shared via [`Rc`] across every clone of this value, so that the residual atom assigned by
    /// the first materialization is reused by every other residualized consumer, which is what deduplicates residual
    /// inputs and inline constants without keying on source-program atoms. Furthermore, the [`Cell`] supplies the
    /// interior mutability that this lazy assignment needs. The residual atom is recorded at _first residual use_,
    /// long after the value has been cloned and shared, and so the write must go through `&self`. Because
    /// [`PartialValueMaterialization`] is a small [`Copy`] value, [`Cell`] suffices without
    /// [`RefCell`](std::cell::RefCell)'s borrow tracking.
    pub(super) materialization: Rc<Cell<PartialValueMaterialization>>,
}

impl<V: Value> PartialEvaluationValue<V> {
    /// Creates a known [`PartialEvaluationValue`] with [`PartialValueMaterialization::Undecided`].
    #[inline]
    pub fn known(value: V) -> Self {
        Self {
            value: PartialValue::Known(value),
            materialization: Rc::new(Cell::new(PartialValueMaterialization::Undecided)),
        }
    }

    /// Creates a known [`PartialEvaluationValue`] with an unassigned [`PartialValueMaterialization::Input`].
    #[inline]
    pub fn known_input(value: V) -> Self {
        Self {
            value: PartialValue::Known(value),
            materialization: Rc::new(Cell::new(PartialValueMaterialization::Input { residual_atom: None })),
        }
    }

    /// Creates a known [`PartialEvaluationValue`] with an unassigned [`PartialValueMaterialization::Constant`].
    #[inline]
    pub fn known_constant(value: V) -> Self {
        Self {
            value: PartialValue::Known(value),
            materialization: Rc::new(Cell::new(PartialValueMaterialization::Constant { residual_atom: None })),
        }
    }

    /// Creates an unknown [`PartialEvaluationValue`] with [`PartialValueMaterialization::Variable`].
    #[inline]
    pub fn variable(r#type: V::Type, residual_atom: AtomId) -> Self {
        Self {
            value: PartialValue::Unknown(r#type),
            materialization: Rc::new(Cell::new(PartialValueMaterialization::Variable { residual_atom })),
        }
    }

    /// Returns the underlying [`PartialValue`].
    #[inline]
    pub fn value(&self) -> &PartialValue<V> {
        &self.value
    }

    /// Returns the [`PartialValueMaterialization`] of this [`PartialEvaluationValue`].
    #[inline]
    pub fn materialization(&self) -> PartialValueMaterialization {
        self.materialization.get()
    }

    /// Returns `true` if the underlying value is [`Known`](PartialValue::Known).
    #[inline]
    pub fn is_known(&self) -> bool {
        self.value.is_known()
    }

    /// Returns `true` if the underlying value is [`Unknown`](PartialValue::Unknown).
    #[inline]
    pub fn is_unknown(&self) -> bool {
        self.value.is_unknown()
    }

    /// Returns the underlying concrete value if this value is [`Known`](PartialValue::Known) and [`None`] otherwise.
    #[inline]
    pub fn as_known(&self) -> Option<&V> {
        self.value.as_known()
    }
}

impl<V: Value> Debug for PartialEvaluationValue<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PartialEvaluationValue")
            .field("value", &self.value)
            .field("materialization", &self.materialization.get())
            .finish()
    }
}

impl<V: Value> Typed for PartialEvaluationValue<V> {
    type Type = V::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, V::Type> {
        self.value.r#type()
    }
}

/// Input of a partially evaluated (i.e., a _residual_) [`Program`] (i.e., an input of a [`PartialEvaluation`]).
/// The residual program's inputs are the original program's surviving unknown inputs followed by the known values
/// (i.e., the residuals) that its unknown subcomputation consumes.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PartialEvaluationInput<V> {
    /// Residual input fed by a value that partial evaluation folded to a concrete known residual value. Note that a
    /// reference-typed known feeder is the live reference handle itself, threaded by identity and never as a snapshot
    /// of its contents, so that the residual program accesses the state as it is when the residual program runs. Refer
    /// to the documentation of [`PartialEvaluation::known_reference_inputs`] for more information.
    Known(V),

    /// Residual input fed by an unknown input of the original program, identified by that input's index in the
    /// original program's inputs.
    Unknown(usize),
}

impl<V> PartialEvaluationInput<V> {
    /// Returns `true` if this [`PartialEvaluationInput`] is [`Self::Known`].
    pub const fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Returns `true` if this [`PartialEvaluationInput`] is [`Self::Unknown`].
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }
}

/// Descriptor for one original output after partial evaluation. Partial evaluation splits the original outputs
/// into those it could fold to a known value and those that remain computed by the residual [`Program`]. A
/// [`PartialEvaluation`] stores these descriptors in original output order so that it can reconstruct the full
/// result after interpreting the residual program.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PartialEvaluationOutput<V> {
    /// Output that was folded to a concrete value during partial evaluation.
    Known(V),

    /// Output produced by the residual program, identified by its index into the residual program's outputs.
    Unknown(usize),
}

impl<V> PartialEvaluationOutput<V> {
    /// Returns `true` if this [`PartialEvaluationOutput`] is [`Self::Known`].
    pub const fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Returns `true` if this [`PartialEvaluationOutput`] is [`Self::Unknown`].
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }
}

/// Source of a residual program input of a [`PartitionedProgram`] obtained via [`PartitionedProgram::residual_inputs`].
/// [`Program::partition`] produces only [`UnknownInput`](Self::UnknownInput) and [`ResidualEdge`](Self::ResidualEdge)
/// sources, with edges numbered in residual input order. [`PartitionedProgram::forward_residuals`] additionally
/// produces [`KnownInput`](Self::KnownInput) and [`KnownOutput`](Self::KnownOutput) sources for residual inputs that
/// would otherwise repeat a value that the caller of the partition already has.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResidualInputSource {
    /// Original (i.e., pre-partitioning) unknown input `index`, which the residual program receives directly.
    /// For example, when the partition replays a fused Jacobian-Vector Product (JVP) program, this input is already
    /// a tangent value.
    UnknownInput(usize),

    /// Original known input `index`, which the residual program receives directly instead of through
    /// a residual edge that would return it unchanged. The index addresses the original boundary, even when
    /// [`PartitionedProgram::known_input_indices`] no longer contains it because the known program does not use it.
    KnownInput(usize),

    /// Fully known output `index` (i.e., known program output `index`), which the residual program receives instead
    /// of a residual edge that would return the same value a second time.
    KnownOutput(usize),

    /// Residual edge `index` (i.e., known program output `index` after the fully known outputs).
    ResidualEdge(usize),
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};

    use super::*;

    #[test]
    fn test_partial_value_is_known() {
        assert!(PartialValue::Known(Array::scalar(2.0f64).unwrap()).is_known());
        assert!(!PartialValue::<Array>::Unknown(ArrayType::scalar(DataType::F64)).is_known());
    }

    #[test]
    fn test_partial_value_is_unknown() {
        assert!(!PartialValue::Known(Array::scalar(2.0f64).unwrap()).is_unknown());
        assert!(PartialValue::<Array>::Unknown(ArrayType::scalar(DataType::F64)).is_unknown());
    }

    #[test]
    fn test_partial_value_as_known() {
        let value = Array::scalar(2.0f64).unwrap();
        assert_eq!(PartialValue::Known(value.clone()).as_known(), Some(&value));
        assert_eq!(PartialValue::<Array>::Unknown(ArrayType::scalar(DataType::F64)).as_known(), None);
    }

    #[test]
    fn test_partial_value_type() {
        // Known values report the type of their value and unknown values report the type they carry.
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let value = Array::new(r#type.clone(), vec![0u8; 24]).unwrap();
        assert_eq!(PartialValue::Known(value).r#type().as_ref(), &r#type);
        assert_eq!(PartialValue::<Array>::Unknown(r#type.clone()).r#type().as_ref(), &r#type);
    }

    #[test]
    fn test_partial_evaluation_value_known() {
        let value = PartialEvaluationValue::known(Array::scalar(2.0f64).unwrap());
        assert_eq!(value.as_known(), Some(&Array::scalar(2.0f64).unwrap()));
        assert_eq!(value.materialization(), PartialValueMaterialization::Undecided);
    }

    #[test]
    fn test_partial_evaluation_value_known_input() {
        let value = PartialEvaluationValue::known_input(Array::scalar(2.0f64).unwrap());
        assert_eq!(value.as_known(), Some(&Array::scalar(2.0f64).unwrap()));
        assert_eq!(value.materialization(), PartialValueMaterialization::Input { residual_atom: None });
    }

    #[test]
    fn test_partial_evaluation_value_known_constant() {
        let value = PartialEvaluationValue::known_constant(Array::scalar(2.0f64).unwrap());
        assert_eq!(value.as_known(), Some(&Array::scalar(2.0f64).unwrap()));
        assert_eq!(value.materialization(), PartialValueMaterialization::Constant { residual_atom: None });
    }

    #[test]
    fn test_partial_evaluation_value_variable() {
        let value = PartialEvaluationValue::<Array>::variable(ArrayType::scalar(DataType::F64), AtomId::new(3));
        assert_eq!(value.as_known(), None);
        assert_eq!(value.materialization(), PartialValueMaterialization::Variable { residual_atom: AtomId::new(3) });
    }

    #[test]
    fn test_partial_evaluation_value_value() {
        let value = PartialEvaluationValue::known(Array::scalar(2.0f64).unwrap());
        assert_eq!(value.value().as_known(), Some(&Array::scalar(2.0f64).unwrap()));
        let value = PartialEvaluationValue::<Array>::variable(ArrayType::scalar(DataType::F64), AtomId::new(3));
        assert!(value.value().is_unknown());
    }

    #[test]
    fn test_partial_evaluation_value_materialization() {
        // Clones share one materialization slot, so the residual atom recorded by the first materialization is visible
        // to every other clone of the same logical value.
        let value = PartialEvaluationValue::known_input(Array::scalar(2.0f64).unwrap());
        let clone = value.clone();
        let materialization = PartialValueMaterialization::Input { residual_atom: Some(AtomId::new(3)) };
        value.materialization.set(materialization);
        assert_eq!(value.materialization(), materialization);
        assert_eq!(clone.materialization(), materialization);

        // Separately constructed values have separate slots, even when they hold equal values.
        let other = PartialEvaluationValue::known_input(Array::scalar(2.0f64).unwrap());
        assert_eq!(other.materialization(), PartialValueMaterialization::Input { residual_atom: None });
    }

    #[test]
    fn test_partial_evaluation_value_is_known() {
        assert!(PartialEvaluationValue::known(Array::scalar(2.0f64).unwrap()).is_known());
        assert!(
            !PartialEvaluationValue::<Array>::variable(ArrayType::scalar(DataType::F64), AtomId::new(3)).is_known()
        );
    }

    #[test]
    fn test_partial_evaluation_value_is_unknown() {
        assert!(!PartialEvaluationValue::known(Array::scalar(2.0f64).unwrap()).is_unknown());
        assert!(
            PartialEvaluationValue::<Array>::variable(ArrayType::scalar(DataType::F64), AtomId::new(3)).is_unknown()
        );
    }

    #[test]
    fn test_partial_evaluation_value_as_known() {
        let value = Array::scalar(2.0f64).unwrap();
        assert_eq!(PartialEvaluationValue::known_constant(value.clone()).as_known(), Some(&value));
        let variable = PartialEvaluationValue::<Array>::variable(ArrayType::scalar(DataType::F64), AtomId::new(3));
        assert_eq!(variable.as_known(), None);
    }

    #[test]
    fn test_partial_evaluation_value_debug() {
        // The rendering shows the current contents of the shared materialization slot rather than the slot itself.
        let value = PartialEvaluationValue::<Array>::variable(ArrayType::scalar(DataType::F64), AtomId::new(3));
        assert_eq!(
            format!("{value:?}"),
            format!(
                "PartialEvaluationValue {{ value: {:?}, materialization: Variable {{ residual_atom: {:?} }} }}",
                PartialValue::<Array>::Unknown(ArrayType::scalar(DataType::F64)),
                AtomId::new(3),
            ),
        );
    }

    #[test]
    fn test_partial_evaluation_value_type() {
        let r#type = ArrayType::scalar(DataType::F64);
        assert_eq!(PartialEvaluationValue::known(Array::scalar(2.0f64).unwrap()).r#type().as_ref(), &r#type);
        let variable = PartialEvaluationValue::<Array>::variable(r#type.clone(), AtomId::new(3));
        assert_eq!(variable.r#type().as_ref(), &r#type);
    }

    #[test]
    fn test_partial_evaluation_input_is_known() {
        assert!(PartialEvaluationInput::Known(0).is_known());
        assert!(!PartialEvaluationInput::<usize>::Unknown(1).is_known());
    }

    #[test]
    fn test_partial_evaluation_input_is_unknown() {
        assert!(!PartialEvaluationInput::Known(0).is_unknown());
        assert!(PartialEvaluationInput::<usize>::Unknown(1).is_unknown());
    }

    #[test]
    fn test_partial_evaluation_output_is_known() {
        assert!(PartialEvaluationOutput::Known(0).is_known());
        assert!(!PartialEvaluationOutput::<usize>::Unknown(1).is_known());
    }

    #[test]
    fn test_partial_evaluation_output_is_unknown() {
        assert!(!PartialEvaluationOutput::Known(0).is_unknown());
        assert!(PartialEvaluationOutput::<usize>::Unknown(1).is_unknown());
    }
}
