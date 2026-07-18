use crate::programs::types::Type;
use crate::sharding::Sharding;
use crate::types::{ArrayType, DataType};

/// A [`Type`] whose forward perturbations and reverse adjoints carry well-defined differential representations.
/// Differential values need not use the primal representation. A compact primal storage format may require a wider
/// signed type to support zero, addition, and negative linear contributions.
pub trait DifferentiableType: Type {
    /// Returns `true` if this [`Type`] represents the trivial differential space whose only possible value is zero
    /// (e.g., [`DataType::Zero`]). Generic transform code uses this property to distinguish a first-class zero-space
    /// type from a type that can carry live, potentially nonzero differential values.
    fn is_zero_space(&self) -> bool;

    /// Returns the [`Type`] that forward-mode tangents of values of this [`Type`] carry. The returned type is used
    /// for forward-mode inputs, outputs, structural zeros, and intermediate Jacobian-Vector Product (JVP) values. It
    /// preserves primal placement metadata because a tangent follows the same forward data flow as its primal. Most
    /// differentiable types use themselves, but specialized storage representations may use a wider differential
    /// representation. For example, [`DataType::F8E8M0FNU`] uses [`DataType::F32`] because its unsigned power-of-two
    /// representation cannot represent zero or negative linear contributions. Non-differentiable types return a
    /// first-class zero-space type, such as [`DataType::Zero`], preserving leaf-for-leaf transform boundaries without
    /// assigning an ordinary Boolean or numeric carrier type.
    fn tangent(&self) -> Self;

    /// Returns the [`Type`] that reverse-mode cotangents of values of this [`Type`] carry. The returned type is the
    /// representation used for reverse-mode seeds, accumulation, structural zeros, and outputs. In most cases it is the
    /// type itself, but it may instead be a distinct representation that supports the required linear operations. For
    /// example, [`DataType::F8E8M0FNU`] uses [`DataType::F32`] cotangents because its unsigned power-of-two storage
    /// format cannot represent zero or negative values, while [`ArrayType`] also swaps the unreduced and reduced axes
    /// of its [`Sharding`]. Refer to [`Sharding::cotangent`] for more information. This mapping is not required to be
    /// an _involution_ (i.e., a specialized primal representation may map to a general-purpose cotangent representation
    /// that is itself a fixed point). Non-differentiable types return a first-class zero-space type. Reverse mode
    /// accumulates no live adjoint for values of those types, while fixed-structure boundaries retain the corresponding
    /// zero-space leaf.
    fn cotangent(&self) -> Self;
}

impl DifferentiableType for DataType {
    #[inline]
    fn is_zero_space(&self) -> bool {
        *self == Self::Zero
    }

    #[inline]
    fn tangent(&self) -> Self {
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
            | Self::U64
            | Self::Zero => Self::Zero,
            Self::F4E2M1FN
            | Self::F6E2M3FN
            | Self::F6E3M2FN
            | Self::F8E3M4
            | Self::F8E4M3
            | Self::F8E4M3FN
            | Self::F8E4M3FNUZ
            | Self::F8E4M3B11FNUZ
            | Self::F8E5M2
            | Self::F8E5M2FNUZ => *self,
            Self::F8E8M0FNU => Self::F32,
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 | Self::C64 | Self::C128 => *self,
        }
    }

    #[inline]
    fn cotangent(&self) -> Self {
        self.tangent()
    }
}

impl DifferentiableType for ArrayType {
    #[inline]
    fn is_zero_space(&self) -> bool {
        self.data_type().is_zero_space()
    }

    #[inline]
    fn tangent(&self) -> Self {
        // Forward perturbations follow their primals' placement. An element-representation change clears explicit
        // layout because byte-level layout metadata cannot in general survive a change in element width.
        let data_type = self.data_type().tangent();
        let layout = if data_type == self.data_type() { self.layout.clone() } else { None };
        Self { data_type, layout, ..self.clone() }
    }

    #[inline]
    fn cotangent(&self) -> Self {
        // Use the element cotangent representation, clear explicit layout when that representation changes element
        // width, swap the unreduced and reduced sharding axes, and keep all other type metadata unchanged.
        let data_type = self.data_type().cotangent();
        let layout = if data_type == self.data_type() { self.layout.clone() } else { None };
        Self { data_type, layout, sharding: self.sharding.as_ref().map(Sharding::cotangent), ..self.clone() }
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
    use crate::types::DataType::*;
    use crate::types::{ArrayType, Layout, Memory, Shape, Size, StridedLayout};

    use super::*;

    #[test]
    fn test_data_type_differential_representations() {
        let non_differentiable = [Token, Boolean, I1, I2, I4, I8, I16, I32, I64, U1, U2, U4, U8, U16, U32, U64];
        for data_type in non_differentiable {
            assert_eq!(data_type.tangent(), Zero);
            assert_eq!(data_type.cotangent(), Zero);
        }
        assert!(Zero.is_zero_space());
        assert_eq!(Zero.tangent(), Zero);
        assert_eq!(Zero.cotangent(), Zero);

        let self_differentiable = [
            F4E2M1FN,
            F6E2M3FN,
            F6E3M2FN,
            F8E3M4,
            F8E4M3,
            F8E4M3FN,
            F8E4M3FNUZ,
            F8E4M3B11FNUZ,
            F8E5M2,
            F8E5M2FNUZ,
            BF16,
            F16,
            F32,
            F64,
            C64,
            C128,
        ];
        for data_type in self_differentiable {
            assert_eq!(data_type.tangent(), data_type);
            assert_eq!(data_type.cotangent(), data_type);
            assert!(!data_type.is_zero_space());
        }

        assert_eq!(F8E8M0FNU.tangent(), F32);
        assert_eq!(F8E8M0FNU.cotangent(), F32);
    }

    #[test]
    fn test_array_type_tangent() {
        let boolean = ArrayType::new(Boolean, Shape::new(vec![Size::Static(4)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])));
        assert_eq!(boolean.tangent(), boolean.clone().with_data_type(Zero).with_layout(None));

        let token = boolean.clone().with_data_type(Token).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            token.tangent(),
            boolean.with_data_type(Zero).with_layout(None).with_memory(Memory::Host { pinned: true }),
        );

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let primal = ArrayType::new(F8E8M0FNU, Shape::new(vec![Size::Static(4)]))
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap()
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])))
            .with_memory(Memory::Host { pinned: true });
        let tangent = primal.clone().with_data_type(F32).with_layout(None);
        assert_eq!(primal.tangent(), tangent);

        // An unchanged element representation retains its explicit physical layout.
        let laid_out = ArrayType::new(F32, Shape::new(vec![Size::Static(4)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out.tangent(), laid_out);
    }

    #[test]
    fn test_array_type_cotangent() {
        // A non-differentiable element type maps to a zero cotangent space with the same structural metadata.
        let boolean = ArrayType::new(Boolean, Shape::new(vec![Size::Static(4)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])));
        assert_eq!(boolean.cotangent(), boolean.clone().with_data_type(Zero).with_layout(None));

        // A different non-differentiable element representation also clears its element-dependent layout while
        // preserving shape and memory.
        let token = boolean.clone().with_data_type(Token).with_memory(Memory::Host { pinned: true });
        assert_eq!(
            token.cotangent(),
            boolean.clone().with_data_type(Zero).with_layout(None).with_memory(Memory::Host { pinned: true }),
        );

        // Without a sharding, the cotangent type is the type itself.
        let plain = ArrayType::new(F32, Shape::new(vec![Size::Static(4)]));
        assert_eq!(plain.cotangent(), plain.clone());

        // With a sharding, the unreduced and reduced axes are swapped.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let unreduced = plain
            .clone()
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        let reduced = plain
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["x"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(unreduced.cotangent(), reduced.clone());
        assert_eq!(reduced.cotangent(), unreduced.clone());

        // E8M0 arrays use F32 cotangent elements while also transforming sharding and preserving other metadata.
        let e8m0 = unreduced
            .with_data_type(F8E8M0FNU)
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])))
            .with_memory(Memory::Host { pinned: true });
        let e8m0_cotangent = reduced.with_data_type(F32).with_layout(None).with_memory(Memory::Host { pinned: true });
        assert_eq!(e8m0.cotangent(), e8m0_cotangent);

        // An unchanged element representation retains its explicit physical layout.
        let laid_out = ArrayType::new(F32, Shape::new(vec![Size::Static(4)]))
            .with_layout(Layout::Strided(StridedLayout::new(vec![4])));
        assert_eq!(laid_out.cotangent(), laid_out);
    }

    #[test]
    fn test_sharding_tangent() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("model", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()])
            .unwrap()
            .with_unreduced_axes(["model"])
            .unwrap()
            .with_varying_manual_axes(["manual"])
            .unwrap();
        let primal = ArrayType::new(F32, Shape::new(vec![Size::Static(4), Size::Static(2)]))
            .with_sharding(sharding.clone())
            .unwrap();

        // Forward tangents follow the primal data flow, so every sharding component remains unchanged.
        assert_eq!(primal.tangent().sharding(), Some(&sharding));
    }

    #[test]
    fn test_sharding_cotangent() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("data", 4, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("model", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("manual", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()])
                .unwrap()
                .with_unreduced_axes(["model"])
                .unwrap()
                .with_varying_manual_axes(["manual"])
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
