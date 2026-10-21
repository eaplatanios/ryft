use std::borrow::Cow;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ryft_macros::Parameter;

use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::ragged::RaggedType;
use crate::parameters::Parameter;
use crate::programs::{LiteralIdentity, Type, TypeError, TypeIdentityRenaming, Typed, Value};

// TODO(eaplatanios): Review from here onwards.

/// Shared immutable payload of a [`RaggedValue`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RaggedValuePayload<A: Value<Type = ArrayType>> {
    /// Complete validated structural type of this value.
    r#type: RaggedType,

    /// Padded numerical storage, or the indexed extent data of a dimension value.
    storage: A,

    /// Full dense extent payloads in [`RaggedType::ragged_dimensions`] order.
    extents: Vec<A>,
}

/// One logical ragged value, consisting of padded array storage and one dense extent payload per
/// [`RaggedDimension`](crate::arrays::types::ragged::RaggedDimension) of its [`RaggedType`]. The payloads have the
/// complete shapes over their extent axes. A ragged extent coordinate's liveness comes from the sibling
/// [`RaggedDimension`](crate::arrays::types::ragged::RaggedDimension) of that storage axis, rather than another nested
/// ragged value.
///
/// `A` is the ordinary array representation, including an eager array or a context-carrying array value. Construction
/// checks representation types without executing operations or reading payloads back to the host. The producer must
/// establish that extents lie within their declared bounds at live coordinates and that related indexed identities
/// agree; attaching a type does not prove either property. [`RaggedNew`](crate::RaggedNew) supplies checked
/// construction for ordinary fields; explicit projections preserve the checked aggregate's value dependency.
///
/// Clones share immutable host metadata and the array handles. This indirection keeps the composite value's dense
/// members from growing to the size of an inline aggregate; it allocates no device buffers and copies no array data.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct RaggedValue<A: Value<Type = ArrayType>>(Arc<RaggedValuePayload<A>>);

impl<A: Value<Type = ArrayType>> RaggedValue<A> {
    /// Packages `storage` and `extents` after checking their exact types against `r#type`. This validates structure,
    /// not the integer contents of extent payloads; their producer must establish the live-coordinate bounds and
    /// identity relationships described by [`RaggedValue`]. Payloads are moved without copying their array data.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] for a mismatched storage type, payload count, or payload type. In particular, compact
    /// payloads cannot stand in for complete payloads over their extent axes.
    pub fn new(r#type: RaggedType, storage: A, extents: Vec<A>) -> Result<Self, TypeError> {
        if storage.r#type().as_ref() != r#type.storage_type() {
            return Err(TypeError::invalid(format!(
                "ragged storage has type `{}` but its declared storage type is `{}`",
                storage.r#type(),
                r#type.storage_type(),
            )));
        }
        let extent_types = r#type.extent_types();
        if extents.len() != extent_types.len() {
            return Err(TypeError::invalid(format!(
                "ragged value expected {} extent payloads but got {}",
                extent_types.len(),
                extents.len(),
            )));
        }
        for (index, (extent, expected)) in extents.iter().zip(extent_types).enumerate() {
            if extent.r#type().as_ref() != expected {
                return Err(TypeError::invalid(format!(
                    "ragged extent payload {index} has type `{}` but its declared type is `{expected}`",
                    extent.r#type(),
                )));
            }
        }
        Ok(Self(Arc::new(RaggedValuePayload { r#type, storage, extents })))
    }

    /// Returns the padded storage array without exposing it as an implicit ordinary-array projection.
    pub fn storage(&self) -> &A {
        &self.0.storage
    }

    /// Returns the dense extent payloads in the type's [`RaggedType::ragged_dimensions`] order.
    pub fn extents(&self) -> &[A] {
        &self.0.extents
    }

    /// Consumes this value and returns its structural type, storage, and extent payloads. Shared values clone their
    /// array handles; uniquely held values transfer the existing payload vector directly.
    pub fn into_parts(self) -> (RaggedType, A, Vec<A>) {
        let payload = Arc::unwrap_or_clone(self.0);
        (payload.r#type, payload.storage, payload.extents)
    }

    /// Renames every identity in the representation without changing any array data or identity relationship.
    pub(crate) fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<RaggedType as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        Self::new(
            self.0.r#type.rename_identities(renaming)?,
            self.storage().rename_type_identities(renaming)?,
            self.extents()
                .iter()
                .map(|value| value.rename_type_identities(renaming))
                .collect::<Result<Vec<_>, _>>()?,
        )
    }

    /// Compares complete literal payloads using the underlying array representation's literal contract.
    pub(crate) fn literal_eq(&self, other: &Self) -> bool
    where
        A: LiteralIdentity,
    {
        self.0.r#type == other.0.r#type
            && self.storage().literal_eq(other.storage())
            && self.extents().len() == other.extents().len()
            && self.extents().iter().zip(other.extents()).all(|(left, right)| left.literal_eq(right))
    }

    /// Hashes the complete structural type and every literal array payload in [`RaggedType::ragged_dimensions`] order.
    pub(crate) fn literal_hash<H: Hasher>(&self, state: &mut H)
    where
        A: LiteralIdentity,
    {
        self.0.r#type.hash(state);
        self.storage().literal_hash(state);
        self.extents().len().hash(state);
        for extent in self.extents() {
            extent.literal_hash(state);
        }
    }
}

impl<A: Value<Type = ArrayType>> Display for RaggedValue<A> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{} {{storage={}, extents=[", self.0.r#type, self.storage())?;
        for (index, extent) in self.extents().iter().enumerate() {
            if index > 0 {
                write!(formatter, ", ")?;
            }
            Display::fmt(extent, formatter)?;
        }
        write!(formatter, "]}}")
    }
}

impl<A: Value<Type = ArrayType>> Typed for RaggedValue<A> {
    type Type = RaggedType;

    fn r#type(&self) -> Cow<'_, RaggedType> {
        Cow::Borrowed(&self.0.r#type)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionType, DimensionVariable};
    use crate::arrays::types::ragged::{RaggedDimension, RaggedValueKind};

    use super::*;

    #[test]
    fn test_ragged_value_new() {
        let dimension = DimensionVariable::new("extent", DimensionBounds::new(0, Some(4)).unwrap());
        let storage = Array::matrix(2, 3, vec![2.0f32, 99.0, 99.0, 3.0, 4.0, 5.0]).unwrap();
        let extents = Array::vector(vec![1i64, 3]).unwrap();
        let r#type = RaggedType::new(
            storage.r#type().into_owned(),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, dimension, vec![0])],
        )
        .unwrap();
        let value = RaggedValue::new(r#type.clone(), storage.clone(), vec![extents.clone()]).unwrap();
        assert_eq!(value.r#type().as_ref(), &r#type);
        assert_eq!(value.storage(), &storage);
        assert_eq!(value.extents(), &[extents.clone()]);
        assert_eq!(
            value.to_string(),
            "ragged<f32[2, 3]; 1:extent over [0]> {storage=[[2.0, 99.0, 99.0], [3.0, 4.0, 5.0]], extents=[[1, 3]]}",
        );
        assert_eq!(value.clone().into_parts(), (r#type.clone(), storage.clone(), vec![extents]));
        assert!(matches!(
            RaggedValue::new(r#type.clone(), Array::vector(vec![0.0f32; 3]).unwrap(), Vec::new()),
            Err(TypeError::Invalid { message })
                if message == "ragged storage has type `f32[3]` but its declared storage type is `f32[2, 3]`",
        ));
        assert!(matches!(
            RaggedValue::new(r#type.clone(), storage.clone(), Vec::new()),
            Err(TypeError::Invalid { message }) if message == "ragged value expected 1 extent payloads but got 0",
        ));
        assert!(matches!(
            RaggedValue::new(r#type.clone(), storage.clone(), vec![Array::vector(vec![1i64]).unwrap()]),
            Err(TypeError::Invalid { message })
                if message == "ragged extent payload 0 has type `i64[1]` but its declared type is `i64[2]`",
        ));
        assert!(matches!(
            RaggedValue::new(r#type, storage, vec![Array::vector(vec![1i32, 3]).unwrap()]),
            Err(TypeError::Invalid { message })
                if message == "ragged extent payload 0 has type `i32[2]` but its declared type is `i64[2]`",
        ));
    }

    #[test]
    fn test_ragged_value_new_sibling_live_coordinates() {
        let rows = DimensionVariable::new("rows", DimensionBounds::new(0, Some(3)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::new(0, Some(4)).unwrap());
        let storage = Array::from_elements(ArrayType::new_static(DataType::F32, [2, 2, 3]), &[1.0f32; 12]).unwrap();
        let r#type = RaggedType::new(
            storage.r#type().into_owned(),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0]), RaggedDimension::new(2, columns, vec![0, 1])],
        )
        .unwrap();
        let extents = vec![Array::vector(vec![1i64, 2]).unwrap(), Array::matrix(2, 2, vec![2i64, 999, 1, 3]).unwrap()];
        // Construction checks structure without observing payloads. The later checked producer/live-coordinate mask
        // owns the distinction between the inactive entry `999` and the true live extents.
        let value = RaggedValue::new(r#type.clone(), storage, extents.clone()).unwrap();
        assert_eq!(value.extents(), extents);
        assert_eq!(value.r#type().extent_types()[1], ArrayType::new_static(DataType::I64, [2, 2]));
        assert_eq!(value.r#type().as_ref(), &r#type);
    }

    #[test]
    fn test_ragged_value_into_parts() {
        let storage = Array::vector(vec![3i64, 3]).unwrap();
        let r#type = RaggedType::new(
            storage.r#type().into_owned(),
            RaggedValueKind::Dimension(DimensionType::from(Dimension::Static(3))),
            Vec::new(),
        )
        .unwrap();
        let value = RaggedValue::new(r#type.clone(), storage.clone(), Vec::new()).unwrap();
        let shared = value.clone();
        assert_eq!(value.into_parts(), (r#type.clone(), storage.clone(), Vec::new()));
        assert_eq!(shared.into_parts(), (r#type, storage, Vec::new()));
    }
}
