use std::borrow::Cow;
use std::fmt::{Debug, Display};

use ryft_macros::Parameter;

use crate::arrays::{ArrayBatch, ArrayType};
use crate::batching::{BatchingPolicy, BatchingTracer};
use crate::captures::CaptureReference;
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::DifferentiationTracer;
use crate::parameters::Parameter;
use crate::partial::{PartialTracer, PartialValue};
use crate::programs::ProgramError;
use crate::programs::atoms::AtomId;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::OperationProjection;
use crate::programs::regions::RegionId;
use crate::programs::types::{Type, TypeError, Typed};
use crate::tracing::Tracer;

/// Location of one Single Static Assignment (SSA) value in a multi-region [`Program`](crate::Program), identified by
/// its containing [`Region`](crate::Region) and its region-local [`AtomId`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId {
    /// [`Region`](crate::Region) containing the atom.
    region: RegionId,

    /// Region-local [`AtomId`] of the value.
    atom: AtomId,
}

impl ValueId {
    /// Creates a new [`ValueId`] from the provided containing region and region-local atom identifier.
    #[inline]
    pub fn new(region: RegionId, atom: AtomId) -> Self {
        Self { region, atom }
    }

    /// Returns the [`RegionId`] of the [`Region`](crate::Region) containing the atom.
    #[inline]
    pub fn region(self) -> RegionId {
        self.region
    }

    /// Returns the region-local [`AtomId`] of the value.
    #[inline]
    pub fn atom(self) -> AtomId {
        self.atom
    }
}

/// Represents leaf values that can participate in traced [`Program`](crate::Program)s. [`Value`] is implemented by
/// every type that can appear as a leaf in a staged [`Program`](crate::Program): both concrete data types such as
/// `f32`, `f64`, and backend arrays, and tracing wrappers such as [`Tracer`](crate::Tracer). It inherits its associated
/// [`Type`](crate::Type) from [`Typed`], so generic code recovers the type as `V::Type` and pinning sites write
/// `V: Value<Type = ArrayType>`. It additionally requires [`Debug`] and [`Display`] so that diagnostics, constants,
/// and [`Operation`](crate::Operation) metadata can render their carried values directly.
pub trait Value: Clone + Debug + Display + Parameter + Typed + Sized {
    /// [`Domain`] that operations involving this [`Value`] *dispatch* through. Every value names two domains:
    /// capability function calls dispatch through the [`DispatchDomain`](Self::DispatchDomain), while transform work
    /// executes in the [`ExecutionDomain`](Self::ExecutionDomain). The two domains coincide for every transform and
    /// staged value (e.g., a staged [`Tracer`](crate::Tracer)'s trace, a [`BatchingTracer`](crate::BatchingTracer)'s
    /// batching level, etc.): dispatch and execution both happen in the live context such a value flows through.
    /// However, they become separate for concrete backend values (e.g., concrete arrays). In those cases, the
    /// [`DispatchDomain`](Self::DispatchDomain) is the constant-only [`EagerContext`](crate::EagerContext) such that
    /// capability calls dispatch to direct implementations instead of a context, while the
    /// [`ExecutionDomain`](Self::ExecutionDomain) names the backend's *rich*, operation-executing eager domain. Backend
    /// values whose rich domain requires state or defaults that cannot be derived from a value (e.g., a client handle)
    /// keep the constant-only domain here too, which simply means free transform entry points do not serve them and an
    /// explicit context must be used instead.
    ///
    /// Blanket capability implementations (e.g., the value-level arithmetic sugar) bind through this domain and use its
    /// operation universe as their coherence discriminator: the sugar applies when `V::DispatchDomain::Operation` can
    /// accept the operation being bound. A staged [`Tracer`](crate::Tracer)'s dispatch domain is its live trace, so the
    /// sugar records instructions there. A concrete backend value's dispatch domain is the constant-only
    /// [`EagerContext`](crate::EagerContext), whose [`ConstantOperation`](crate::ConstantOperation) universe accepts
    /// nothing. This is precisely what keeps the blanket implementations coherent with (i.e., disjoint from) the direct
    /// capability implementations that concrete values provide instead.
    type DispatchDomain: Domain<Type = Self::Type, Value = Self>;

    /// [`Domain`] that transform work involving this [`Value`] *executes* in. Refer to the documentation of
    /// [`DispatchDomain`](Self::DispatchDomain) for information on the two types of [`Domain`]s that each value
    /// provides.
    type ExecutionDomain: Domain<Type = Self::Type, Value = Self>;

    /// Returns the [`Domain`] that operations involving this [`Value`] *dispatch* through. Refer to the
    /// documentation of [`DispatchDomain`](Self::DispatchDomain) for more information.
    fn dispatch_domain(&self) -> Self::DispatchDomain;

    /// Returns the [`Domain`] that transform work involving this [`Value`] *executes* in. Refer to the
    /// documentation of [`ExecutionDomain`](Self::ExecutionDomain) for more information.
    fn execution_domain(&self) -> Self::ExecutionDomain;

    /// Returns an equivalent value whose identity-bearing type metadata has been simultaneously renamed according to
    /// `renaming`. [`Value`] represents every kind of leaf that can participate in a [`Program`](crate::Program), not
    /// only concrete runtime payloads. Some values, such as metadata-only values and captured-value references, store
    /// their [`Type`](Self::Type) or other type metadata directly. When a program or region is instantiated under
    /// renamed [`TypeIdentity`](crate::TypeIdentity)s, that metadata must be renamed together with atom types and
    /// [`Operation`](crate::Operation) metadata so that [`Typed::r#type`] cannot continue to expose stale identities.
    ///
    /// This compiler-managed operation must preserve the represented runtime data, Single Static Assignment (SSA)
    /// identity, and execution semantics; it may only reconstruct metadata that depends on the value's type. The
    /// default implementation clones values whose type is unchanged by [`Type::rename_identities`] and rejects
    /// identity-bearing changes. Value types that can safely reconstruct their stored type metadata must override
    /// this method.
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        let current_type = self.r#type();
        let renamed_type = current_type.rename_identities(renaming)?;
        if current_type.as_ref() != &renamed_type {
            return Err(TypeError::invalid(format!(
                "cannot rename type identities in value of type {} without a value-specific \
                reconstruction implementation",
                current_type.as_ref(),
            )));
        }
        Ok(self.clone())
    }
}

/// Supports extracting this value as a concrete host-side value of type `V`. Concretization is distinct from a staged
/// value conversion as it makes data observable to Rust code and may therefore require synchronization, device-to-host
/// transfer, or another backend-specific readback. It fails with [`ProgramError::Concretization`] when the value is
/// symbolic, inaccessible, or incompatible with the requested representation. A value may implement this trait for
/// multiple concrete representations. Implementations should consequently be target-specific (e.g.,
/// [`Concretizable<bool>`]) rather than blanket implementations that claim support for every `V` and fail at runtime.
pub trait Concretizable<V> {
    /// Extracts this value as a concrete host-side value of type `V`. Returns a [`ProgramError::Concretization`] error
    /// if concretization fails.
    fn concretize(&self) -> Result<V, ProgramError>;
}

impl<C: Context> Concretizable<bool> for Tracer<C> {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        Err(ProgramError::Concretization { message: "cannot extract a concrete boolean from a tracer".to_string() })
    }
}

impl<T: Type> Concretizable<bool> for CaptureReference<T> {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        // A captured constant is a reference into a side table, not the concrete predicate value itself. Control-flow
        // staging must keep predicates in the IR or add a transform-specific rule instead of trying to branch here.
        Err(ProgramError::Concretization {
            message: "cannot extract a concrete boolean from a captured constant reference".to_string(),
        })
    }
}

impl<C: Context<Value: Concretizable<bool>>> Concretizable<bool> for PartialTracer<C> {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        // A partial evaluation value delegates concrete Boolean extraction to its known payload. This lets host control
        // flow branch on values known under an eager inner context, while unknown values report that they cannot be
        // concretized. Also, a poisoned value surfaces its deferred error here, since branching on it cannot proceed
        // anyway.
        match self.value()?.value() {
            PartialValue::Known(known) => known.concretize(),
            PartialValue::Unknown(_) => Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from an unknown partial-evaluation value".to_string(),
            }),
        }
    }
}

impl<C: Context, P: BatchingPolicy<C, Batch: Concretizable<bool>>> Concretizable<bool> for BatchingTracer<C, P> {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        // The policy-selected carrier owns whether this value is replicated and therefore concretizable. Keeping that
        // decision on the carrier lets every batching policy participate without exposing axis metadata generically.
        self.batch().concretize()
    }
}

impl<C: Context<Value: Concretizable<bool>>> Concretizable<bool> for DifferentiationTracer<C> {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        // A differentiation tracer delegates concrete Boolean extraction to its primal, so host branching succeeds
        // exactly when that primal is concrete.
        self.primal().concretize()
    }
}

impl<V: Value<Type = ArrayType> + Concretizable<bool>> Concretizable<bool> for ArrayBatch<V> {
    fn concretize(&self) -> Result<bool, ProgramError> {
        if let Some(axis) = self.batch_axis().axis() {
            return Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete boolean from a value batched along axis {axis}"),
            });
        }
        self.value().concretize()
    }
}

/// Provides checked access to one member kind of a composite [`Value`]. Some program families store several kinds of
/// values in one [`Value`] type so that all of them can flow through the same [`Program`](crate::Program). For example,
/// [`ArrayIrValue`](crate::ArrayIrValue) is either a backend array or a first-class runtime dimension, mirroring the
/// two member types of [`ArrayIrType`](crate::ArrayIrType). Most operations and transform rules, however, are written
/// against exactly one member kind (e.g., array-only rules consume values with `Value<Type = ArrayType>`), and
/// `ValueProjection<T>` is what lets them accept a composite value. [`Self::projected`] returns a read-only view
/// of the value as its `T`-typed member, [`Self::into_projected`] consumes the value and returns an owned member
/// representation, and [`Self::from_projected`] embeds a member representation back into the composite value type.
/// Both projection methods fail with a [`TypeError`] when the value holds a different member kind than the requested
/// one. The associated representations depend on how a value relates to its member:
///
///   - Values that contain their member directly, such as [`ArrayIrValue`](crate::ArrayIrValue), project to the member
///     itself (e.g., `&A` and `A` for the array member), so no payload is ever cloned or copied.
///   - Values that refer to a program instead of containing data (e.g, a [`Tracer`] naming an [`Atom`](crate::Atom),
///     or a [`CaptureReference`] naming a capture-table entry) have no member payload to extract. They project to
///     [`ProjectedValue`] views over the value itself or over a borrow of it (or to a retyped copy of themselves, as
///     [`CaptureReference`] does for its owned form), which keep the original value and the program identity it
///     carries intact and only narrow the type it reports.
pub trait ValueProjection<T: Type>: Value {
    /// Owned representation of this [`Value`]'s `T`-typed member.
    type Projected: Typed<Type = T>;

    /// Read-only representation of this [`Value`]'s `T`-typed member. `T: 'v` limits the view to a lifetime for which
    /// borrowed member-type metadata remains valid without requiring every projected type to be `'static`.
    type ProjectedRef<'v>: Typed<Type = T>
    where
        Self: 'v,
        T: 'v;

    /// Embeds an owned `T`-typed member representation back into this composite [`Value`] type.
    fn from_projected(value: Self::Projected) -> Self;

    /// Returns a read-only view of this [`Value`] as its `T`-typed member, without cloning any member payload.
    /// Returns a [`TypeError`] when this value holds a different member kind.
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        T: 'v;

    /// Consumes this [`Value`] and returns an owned representation of its `T`-typed member, transferring rather than
    /// copying any contained payload. Returns a [`TypeError`] when this value holds a different member kind.
    fn into_projected(self) -> Result<Self::Projected, TypeError>;
}

/// A [`Value`] whose reported [`Type`] has been narrowed to one member kind of its composite type, as returned by
/// [`ValueProjection::projected`] and [`ValueProjection::into_projected`]. This wrapper exists for values that refer
/// to a program rather than containing a member payload, such as a [`Tracer`] naming a Single Static Assignment (SSA)
/// [`Atom`](crate::Atom) or a [`PartialTracer`] carrying known/unknown state. Such a value cannot be split into its
/// member the way an enum of payloads can, so projecting it keeps the value intact (i.e., preserving the program
/// identity it carries) and pairs it with the member type it was validated against. [`Typed`] consequently reports
/// that member type rather than the value's original composite type. The wrapped value is owned for
/// [`ValueProjection::Projected`] and borrowed (i.e., `ProjectedValue<T, &V>`) for [`ValueProjection::ProjectedRef`],
/// so one type covers both the owned and the read-only projection.
///
/// The projection alone does not define how [`Operation`](crate::Operation)s on the wrapped value dispatch.
/// [`ProjectedContext`] provides that behavior through this type's blanket [`Value`] implementation whenever the
/// surrounding composite domains expose the corresponding [`OperationProjection`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ProjectedValue<T: Type, V> {
    /// Original value, kept intact so the program identity it carries is preserved.
    value: V,

    /// Member [`Type`] validated against `value` at construction.
    r#type: T,
}

impl<T: Type, V> ProjectedValue<T, V> {
    /// Creates a new [`ProjectedValue`] after the caller has validated `type` against `value`.
    #[inline]
    pub fn new(value: V, r#type: T) -> Self {
        Self { value, r#type }
    }

    /// Borrows the original value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes this [`ProjectedValue`] and returns the original value.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }
}

impl<T: Type, V: Display> Display for ProjectedValue<T, V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value.fmt(formatter)
    }
}

impl<T: Type, V> Typed for ProjectedValue<T, V> {
    type Type = T;

    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<T: Type, V: ValueProjection<T, Projected = Self>> Value for ProjectedValue<T, V>
where
    <V::DispatchDomain as Domain>::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    <V::DispatchDomain as Domain>::Operation: OperationProjection<T>,
    <V::ExecutionDomain as Domain>::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    <V::ExecutionDomain as Domain>::Operation: OperationProjection<T>,
{
    type DispatchDomain = ProjectedContext<V::DispatchDomain, T>;
    type ExecutionDomain = ProjectedContext<V::ExecutionDomain, T>;

    #[inline]
    fn dispatch_domain(&self) -> Self::DispatchDomain {
        ProjectedContext::new(self.value.dispatch_domain())
    }

    #[inline]
    fn execution_domain(&self) -> Self::ExecutionDomain {
        ProjectedContext::new(self.value.execution_domain())
    }
}

impl<T: Type, C, V: Concretizable<C>> Concretizable<C> for ProjectedValue<T, V> {
    #[inline]
    fn concretize(&self) -> Result<C, ProgramError> {
        self.value.concretize()
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::arrays::{DataType, Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::contexts::EagerContext;

    use super::*;

    #[test]
    fn test_value_rename_type_identities() {
        #[derive(Clone, Debug, PartialEq)]
        struct TestValue {
            r#type: ArrayType,
        }

        impl Display for TestValue {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                Display::fmt(&self.r#type, formatter)
            }
        }

        impl Parameter for TestValue {}

        impl Typed for TestValue {
            type Type = ArrayType;

            fn r#type(&self) -> Cow<'_, ArrayType> {
                Cow::Borrowed(&self.r#type)
            }
        }

        impl Value for TestValue {
            type DispatchDomain = EagerContext<Self>;
            type ExecutionDomain = EagerContext<Self>;

            fn dispatch_domain(&self) -> Self::DispatchDomain {
                EagerContext::new()
            }

            fn execution_domain(&self) -> Self::ExecutionDomain {
                EagerContext::new()
            }
        }

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let value =
            TestValue { r#type: ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())])) };
        assert_eq!(value.rename_type_identities(&TypeIdentityRenaming::new()), Ok(value.clone()));

        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target).unwrap();
        assert_eq!(
            value.rename_type_identities(&renaming),
            Err(TypeError::invalid(
                "cannot rename type identities in value of type f32[source] without a value-specific reconstruction \
                 implementation",
            )),
        );
    }

    #[test]
    fn test_projected_value() {
        let r#type = ArrayType::scalar(DataType::F32);
        let value = "value".to_string();

        // Owned form, as returned by `ValueProjection::into_projected`.
        let projected = ProjectedValue::new(value.clone(), r#type.clone());
        assert_eq!(projected.value(), &value);
        assert_eq!(projected.r#type(), Cow::Borrowed(&r#type));
        assert_eq!(projected.to_string(), value);
        assert_eq!(projected.into_value(), "value");

        // Borrowed form, as returned by `ValueProjection::projected`.
        let projected = ProjectedValue::new(&value, r#type.clone());
        assert_eq!(projected.value(), &&value);
        assert_eq!(projected.r#type(), Cow::Borrowed(&r#type));
        assert_eq!(projected.to_string(), value);
        assert_eq!(projected.into_value(), &value);
    }
}
