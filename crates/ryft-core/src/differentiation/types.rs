use crate::sharding::Sharding;
use crate::types::{ArrayType, DataType, Type};

/// A [`Type`] whose reverse-mode cotangents carry a (possibly different) dual, or **cotangent**, [`Type`]. Reverse-mode
/// differentiation seeds an output cotangent, transposes the linear program, and emits structural zero cotangents for
/// disconnected inputs. Each of those cotangent values must be typed in the *dual* space rather than the primal one.
/// For types that carry no distribution metadata the dual type is the type itself (i.e., the identity). Types that
/// carry such metadata map it to its cotangent dual. For example, [`ArrayType`] swaps the unreduced and reduced axes of
/// its [`Sharding`] (refer to [`Sharding::cotangent`]). The swap is an **involution**, so that
/// `value.cotangent().cotangent()` recovers `value`.
pub trait DifferentiableType: Type {
    /// Returns the [`Type`] that reverse-mode cotangents of values of this type carry.
    fn cotangent(&self) -> Self;
}

impl DifferentiableType for DataType {
    #[inline]
    fn cotangent(&self) -> Self {
        // Scalar data types carry no distribution metadata, so a cotangent has the same type as its primal value.
        self.clone()
    }
}

impl DifferentiableType for ArrayType {
    #[inline]
    fn cotangent(&self) -> Self {
        // Swap the unreduced and reduced sharding axes (the cotangent dual); all other type metadata is unchanged.
        Self { sharding: self.sharding.as_ref().map(Sharding::cotangent), ..self.clone() }
    }
}

impl Sharding {
    /// Returns the [`Sharding`] that reverse-mode cotangents of values sharded like this one carry. It swaps
    /// [`Self::unreduced_axes`] with [`Self::reduced_axes`] and keeps all other state unchanged. The cotangent of a
    /// value that still carries per-device partial results along an axis is the same value on every device along that
    /// axis (i.e., marked reduced), while the cotangent of an already-reduced value carries per-device partial results
    /// that still need a reduction (i.e., marked unreduced). The swap is an **involution**, so that
    /// `sharding.cotangent().cotangent()` recovers `sharding`.
    pub fn cotangent(&self) -> Self {
        Self { unreduced_axes: self.reduced_axes.clone(), reduced_axes: self.unreduced_axes.clone(), ..self.clone() }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use pretty_assertions::assert_eq;

    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::DataType::F32;
    use crate::types::{ArrayType, Shape, Size};

    use super::*;

    #[test]
    fn test_data_type_cotangent() {
        // A scalar data type carries no distribution metadata, so its cotangent type is itself.
        assert_eq!(F32.cotangent(), F32);
    }

    #[test]
    fn test_array_type_cotangent() {
        // Without a sharding, the cotangent type is the type itself.
        let plain = ArrayType::new(F32, Shape::new(vec![Size::Static(4)]));
        assert_eq!(plain.cotangent(), plain);

        // With a sharding, the unreduced and reduced axes are swapped.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = plain
            .clone()
            .with_sharding(
                Sharding::with_unreduced_axes(mesh.clone(), vec![ShardingDimension::replicated()], ["x"]).unwrap(),
            )
            .unwrap();
        let reduced = plain
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::replicated()],
                    Vec::<&str>::new(),
                    ["x"],
                    Vec::<&str>::new(),
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(unreduced.cotangent(), reduced);
        assert_eq!(reduced.cotangent(), unreduced);
    }

    #[test]
    fn test_sharding_cotangent() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("model", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::with_manual_axes(
            mesh.clone(),
            vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()],
            ["model"],
            Vec::<&str>::new(),
            ["manual"],
        )
        .unwrap();

        // The cotangent swaps the unreduced and reduced sets and keeps all other state unchanged.
        let cotangent = sharding.cotangent();
        assert_eq!(cotangent.dimensions(), sharding.dimensions());
        assert_eq!(cotangent.unreduced_axes(), &BTreeSet::new());
        assert_eq!(cotangent.reduced_axes(), &BTreeSet::from(["model".to_string()]));
        assert_eq!(cotangent.varying_manual_axes(), &BTreeSet::from(["manual".to_string()]));

        // The swap is an involution.
        assert_eq!(cotangent.cotangent(), sharding);

        // Shardings without reduction state are their own cotangents.
        let replicated = Sharding::replicated(mesh, 2);
        assert_eq!(replicated.cotangent(), replicated);
    }
}
