use std::borrow::Cow;
use std::fmt::{Debug, Display};

use crate::batching::{ArrayBatch, BatchingTracer};
use crate::captures::CaptureReference;
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationTracer;
use crate::parameters::Parameter;
use crate::partial::{PartialTracer, PartialValue};
use crate::programs::ProgramError;
use crate::programs::atoms::AtomId;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::regions::RegionId;
use crate::programs::types::{Type, TypeError, TypeProjection, Typed};
use crate::tracing::Tracer;
use crate::types::ArrayType;

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

/// Checked view of a value projected from a heterogeneous storage universe to one homogeneous member type.
///
/// The wrapper preserves `value` itself—including any Single Static Assignment (SSA), capture, partial-evaluation, or
/// transform identity—and stores only the member type obtained through [`TypeProjection`]. Concrete eager storage sums
/// project directly to borrowed or owned payloads instead and therefore do not need this wrapper. Projection alone does
/// not define how operations dispatch; a context adapter can add the appropriate [`Value`] implementation when binding
/// homogeneous operations through an outer heterogeneous graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ProjectedValue<T: Type, V> {
    /// Original heterogeneous value whose identity this projection preserves.
    value: V,

    /// Homogeneous member type validated against `value` at construction.
    r#type: T,
}

impl<T: Type, V> ProjectedValue<T, V> {
    /// Constructs a projected value after the caller has validated `type` against `value`.
    #[inline]
    pub(crate) fn new(value: V, r#type: T) -> Self {
        Self { value, r#type }
    }

    /// Borrows the original heterogeneous value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes this projection and returns the original heterogeneous value.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }
}

impl<T: Type, V: Display> Display for ProjectedValue<T, V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.value, formatter)
    }
}

impl<T: Type, V> Parameter for ProjectedValue<T, V> {}

impl<T: Type, V> Typed for ProjectedValue<T, V> {
    type Type = T;

    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        Cow::Borrowed(&self.r#type)
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

/// Projects one homogeneous member value from a heterogeneous [`Value`] storage universe.
///
/// Read-only consumers use [`Self::project_ref`] so concrete eager payloads remain borrowed and are never cloned.
/// Consumers that already own the storage value use [`Self::into_projected`] to transfer the member payload without
/// copying it. [`Self::lift`] embeds an owned projected member back into the heterogeneous universe.
///
/// Symbolic values whose storage representation cannot be changed without losing SSA or transform identity use
/// [`ProjectedValue`] for one or both associated representations. The wrapper keeps the original value intact while
/// exposing its checked homogeneous type.
pub trait ValueProjection<T: Type>: Value<Type: TypeProjection<T>> {
    /// Owned representation of the projected member.
    type Projected: Typed<Type = T>;

    /// Read-only representation of the projected member.
    type ProjectedRef<'a>: Typed<Type = T>
    where
        Self: 'a;

    /// Borrows this value as the requested homogeneous member without cloning an eager payload.
    ///
    /// Returns a [`TypeError`] when this value contains a different member kind.
    fn project_ref(&self) -> Result<Self::ProjectedRef<'_>, TypeError>;

    /// Consumes this value and transfers or preserves its requested homogeneous member representation.
    ///
    /// Returns a [`TypeError`] when this value contains a different member kind.
    fn into_projected(self) -> Result<Self::Projected, TypeError>;

    /// Lifts an owned projected member back into this heterogeneous value universe.
    fn lift(value: Self::Projected) -> Self;
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

impl<C: Context<Type = ArrayType, Value: Concretizable<bool>>> Concretizable<bool> for BatchingTracer<C> {
    #[inline]
    fn concretize(&self) -> Result<bool, ProgramError> {
        // A batch-carrying value delegates concrete Boolean extraction to its packed value only when it is replicated.
        // A mapped batch has one Boolean per item and cannot drive one host branch.
        if !self.batch().batch_axis().is_replicated() {
            return Err(ProgramError::Concretization {
                message: "cannot extract a concrete boolean from a batched value".to_string(),
            });
        }
        self.batch().value().concretize()
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::types::{DataType, Dimension, DimensionBounds, DimensionVariable, Shape};

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
}
