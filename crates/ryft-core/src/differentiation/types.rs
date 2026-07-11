use crate::sharding::Sharding;
use crate::types::{ArrayType, DataType, Type};

/// A [`Type`] whose reverse-mode cotangents carry a (possibly different) dual, or **cotangent**, [`Type`]. Reverse-mode
/// differentiation seeds an output cotangent, transposes the linear program, and emits structural zero cotangents for
/// disconnected inputs. Each of those cotangent values must be typed in the *dual* space rather than the primal one.
pub trait DifferentiableType: Type {
    /// Returns the [`Type`] that reverse-mode cotangents of values of this type carry, or `None`
    /// when this type carries no cotangent space (i.e., analogous to the
    /// [JAX `float0` type](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.float0.html)).
    ///
    /// Differentiable types (e.g., floating-point and complex element types) carry a cotangent space, and so
    /// [`Self::cotangent`] returns `Some(dual)`. In most cases, the cotangent type is the type itself (i.e.,
    /// [`Self::cotangent`] is an identity function). However, that is not always the case. For example, [`ArrayType`]
    /// swaps the unreduced and reduced axes of its [`Sharding`] (refer to [`Sharding::cotangent`] for more
    /// information). For types that are actually differentiable, the swap is an **involution** so that if
    /// `value.cotangent()` is `Some(dual)`, then `dual.cotangent()` recovers `value`.
    ///
    /// Non-differentiable types (e.g., Boolean, integer, and token element types) carry no cotangent space and for
    /// those cases, [`Self::cotangent`] returns `None`. A `None` cotangent means that the type contributes no adjoint.
    /// Reverse mode differentiation will seed no cotangent for outputs corresponding to those types, and it will emit
    /// no cotangent for inputs corresponding to those types. The non-differentiable fixed point satisfies
    /// `cotangent().is_none()` reflexively (i.e., there is no dual to involute).
    fn cotangent(&self) -> Option<Self>;
}

impl DifferentiableType for DataType {
    #[inline]
    fn cotangent(&self) -> Option<Self> {
        match self {
            Self::Token
            | Self::Boolean
            | Self::I1
            | Self::I2
            | Self::I4
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::U1
            | Self::U2
            | Self::U4
            | Self::U8
            | Self::U16
            | Self::U32
            | Self::U64 => None,
            Self::F4E2M1FN
            | Self::F6E2M3FN
            | Self::F6E3M2FN
            | Self::F8E3M4
            | Self::F8E4M3
            | Self::F8E4M3FN
            | Self::F8E4M3FNUZ
            | Self::F8E4M3B11FNUZ
            | Self::F8E5M2
            | Self::F8E5M2FNUZ
            | Self::F8E8M0FNU
            | Self::BF16
            | Self::F16
            | Self::F32
            | Self::F64
            | Self::C64
            | Self::C128 => Some(self.clone()),
        }
    }
}

impl DifferentiableType for ArrayType {
    #[inline]
    fn cotangent(&self) -> Option<Self> {
        // An array is differentiable exactly when its element data type is. A non-differentiable element type carries
        // no cotangent space. For differentiable element data types, swap the unreduced and reduced sharding axes and
        // keep all other type metadata unchanged.
        self.data_type().cotangent()?;
        Some(Self { sharding: self.sharding.as_ref().map(Sharding::cotangent), ..self.clone() })
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
    use crate::types::DataType::{Boolean, F32, I32};
    use crate::types::{ArrayType, Shape, Size};

    use super::*;

    #[test]
    fn test_data_type_cotangent() {
        assert_eq!(F32.cotangent(), Some(F32));
        assert_eq!(Boolean.cotangent(), None);
        assert_eq!(I32.cotangent(), None);
    }

    #[test]
    fn test_array_type_cotangent() {
        // A non-differentiable element type carries no cotangent space regardless of shape.
        let boolean = ArrayType::new(Boolean, Shape::new(vec![Size::Static(4)]));
        assert_eq!(boolean.cotangent(), None);

        // Without a sharding, the cotangent type is the type itself.
        let plain = ArrayType::new(F32, Shape::new(vec![Size::Static(4)]));
        assert_eq!(plain.cotangent(), Some(plain.clone()));

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
        assert_eq!(unreduced.cotangent(), Some(reduced.clone()));
        assert_eq!(reduced.cotangent(), Some(unreduced));
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
