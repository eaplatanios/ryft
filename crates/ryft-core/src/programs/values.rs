use std::borrow::Cow;
use std::fmt::{Debug, Display};

use ryft_macros::Parameter;

use crate::arrays::{ArrayBatch, ArrayType};
use crate::batching::{BatchingPolicy, BatchingTracer};
use crate::captures::CaptureReference;
use crate::contexts::{Context, Domain, ProjectedContext};
use crate::differentiation::DifferentiationTracer;
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::partial::{PartialTracer, PartialValue};
use crate::programs::ProgramError;
use crate::programs::atoms::AtomId;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationProjection};
use crate::programs::references::ReferenceId;
use crate::programs::regions::{RegionId, RegionRef};
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
/// `f32`, `f64`, and backend arrays, and tracing wrappers such as [`Tracer`]. It inherits its associated [`Type`] from
/// [`Typed`], so generic code recovers the type as `V::Type` and pinning sites write `V: Value<Type = ArrayType>`. It
/// additionally requires [`Debug`] and [`Display`] so that diagnostics, constants, and [`Operation`] metadata can
/// render their carried values directly.
///
/// # Rendering Contract
///
/// [`Display`] must be deterministic and semantically complete for values that can be stored as
/// [`Atom::Constant`](crate::Atom::Constant)s (i.e., two constants with the same rendered [`Type`] but different
/// program semantics must render differently). [`Program`](crate::Program) combines this payload with the atom's
/// separately rendered type, so shape and other type-level semantics need not be duplicated in the payload. Program
/// constants are expected to be small and synchronously renderable. Large or device-resident values must instead be
/// represented by a compact semantic reference such as [`CaptureReference`], whose displayed table index and type
/// identify the program-level constant without reading back its runtime payload.
pub trait Value: Clone + Debug + Display + Parameter + Typed + Sized {
    /// Boolean value that represents whether this value family overrides [`Self::validate_eager_interpretation`],
    /// obligating every eager interpretation boundary to invoke that hook before executing anything. The default
    /// value (i.e., `false`) is correct for value universes that cannot carry concrete mutable resources. A family
    /// that overrides [`Self::validate_eager_interpretation`] must set this constant to `true`. That is because an
    /// override is not otherwise detectable statically, and so this constant is what makes the hook reachable.
    /// Specifically, it does two things:
    ///
    ///   - It lets eager interpretation boundaries skip validation entirely for ordinary value families. Because the
    ///     check is a monomorphization-time constant, the branch compiles away instead of invoking the no-op default
    ///     for every attached region of every bind.
    ///   - It gates the creation of [`EagerInterpretationValidation`](crate::EagerInterpretationValidation) evidence,
    ///     which lets nested eager interpretation skip revalidating regions that an enclosing root's boundary
    ///     validation already covered. Only a family whose boundary validation actually ran may issue that evidence;
    ///     staging and transform contexts enforce their own structural legality gates and never produce it.
    const VALIDATES_EAGER_INTERPRETATION: bool = false;

    /// [`Domain`] that operations involving this [`Value`] *dispatch* through. Every value names two domains:
    /// capability function calls dispatch through the [`DispatchDomain`](Self::DispatchDomain), while transform work
    /// executes in the [`ExecutionDomain`](Self::ExecutionDomain). The two domains coincide for every transform and
    /// staged value (e.g., a staged [`Tracer`]'s trace, a [`BatchingTracer`]'s batching level, etc.): dispatch and
    /// execution both happen in the live context such a value flows through. However, they become separate for concrete
    /// backend values (e.g., concrete arrays). In those cases, the [`DispatchDomain`](Self::DispatchDomain) is the
    /// constant-only [`EagerContext`](crate::EagerContext) such that capability calls dispatch to direct
    /// implementations instead of a context, while the [`ExecutionDomain`](Self::ExecutionDomain) names the backend's
    /// _rich_, operation-executing eager domain. Backend values whose rich domain requires state or defaults that
    /// cannot be derived from a value (e.g., a client handle) keep the constant-only domain here too, which simply
    /// means free transform entry points do not serve them and an explicit context must be used instead.
    ///
    /// Blanket capability implementations (e.g., the value-level arithmetic sugar) bind through this domain and use its
    /// operation universe as their coherence discriminator: the sugar applies when `V::DispatchDomain::Operation` can
    /// accept the operation being bound. A staged [`Tracer`]'s dispatch domain is its live trace, so the sugar records
    /// instructions there. A concrete backend value's dispatch domain is the constant-only
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

    /// Returns an equivalent value whose type-identity metadata has been simultaneously renamed according to
    /// `renaming`. [`Value`] represents every kind of leaf that can participate in a [`Program`](crate::Program), not
    /// only concrete runtime payloads. Some values, such as metadata-only values and captured-value references, store
    /// their [`Type`](Typed::Type) or other type metadata directly. When a program or region is instantiated under
    /// renamed [`TypeIdentity`](crate::TypeIdentity)s, that metadata must be renamed together with atom types and
    /// [`Operation`] metadata so that [`Typed::r#type`](Typed::type) cannot continue to expose stale identities.
    ///
    /// This compiler-managed operation must preserve the represented runtime data, Single Static Assignment (SSA)
    /// identity, and execution semantics; it may only reconstruct metadata that depends on the value's type. The
    /// default implementation clones values whose type is unchanged by [`Type::rename_identities`] and rejects
    /// changes to that metadata. Value types that can safely reconstruct their stored type metadata must override
    /// this method.
    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        rename_type_identities_by_rejection(self, renaming)
    }

    /// Validates that this value can be used as a program constant (i.e., [`Atom::Constant`](crate::Atom::Constant)).
    /// Such values must satisfy the following rendering contract: their [`Display`] output must be deterministic and
    /// semantically complete, because program renderings double as structural fingerprints. Values that cannot satisfy
    /// it (most notably mutable references, whose runtime identity is process-local and deliberately absent from
    /// their deterministic rendering) must reject constant storage here and enter programs through inputs or captures
    /// instead. [`Region`](crate::Region) sealing enforces this for every stored constant in every region, so all
    /// construction paths (i.e., program builders, [`Program::new`](crate::Program::new), and region imports alike)
    /// are covered. The default implementation accepts all values.
    #[inline]
    fn validate_as_constant(&self) -> Result<(), TypeError> {
        Ok(())
    }

    /// Validates a region closure at the eager interpretation boundary, before any of it executes. Eager interpretation
    /// of a program whose values carry concrete mutable resources performs irreversible mutations, so its legality must
    /// be established transactionally. This hook inspects the closure boundary up front and either accepts the closure
    /// or rejects it before any constant is lifted or instruction is executed. Eager program interpretation calls it
    /// with the program's entry region, and direct eager binding calls it with each attached region that carries no
    /// [`EagerInterpretationValidation`](crate::EagerInterpretationValidation) evidence.
    ///
    /// The default accepts every closure and is correct for value universes that cannot carry concrete mutable
    /// resources. A family that admits such resources (e.g., the core array IR with references) must override this
    /// function with its interpretation-boundary legality checks and set [`Self::VALIDATES_EAGER_INTERPRETATION`] to
    /// `true` so that eager interpretation boundaries actually invoke the override. Transform boundaries such as
    /// partial evaluation do not call this hook. Instead, they enforce their own type- and effect-level gates.
    ///
    /// This hook is intentionally distinct from [`Self::validate_as_constant`] because constant admissibility is
    /// a local storage property of a value, while this function sees the complete attached-region closure before
    /// observable execution begins.
    ///
    /// # Parameters
    ///
    ///   - `region`: Root of the complete region closure that is about to be replayed.
    #[inline]
    fn validate_eager_interpretation<V: Value<Type = Self::Type>, O: Operation<Type = Self::Type>>(
        region: RegionRef<'_, V, O>,
    ) -> Result<(), ProgramError> {
        let _ = region;
        Ok(())
    }

    /// Returns the capture-table position that this value names when it is a compact reference to a runtime value in
    /// the surrounding capture table (e.g., a [`CaptureReference`]), or [`None`] for every other value, including
    /// immediate constants that carry their own data. This is the one canonical capture query: capture validation,
    /// dead-capture elimination, capture lifting, reference discharge, and reference analysis all resolve captures
    /// through it, so a constant family that can name captures must override it, and families that cannot leave the
    /// default, which returns [`None`].
    #[inline]
    fn capture_index(&self) -> Option<usize> {
        None
    }

    /// Returns the process-local [`ReferenceId`] of the mutable reference allocation that this value is a live handle
    /// to, or [`None`] when it is not a live reference handle. Runtime alias validation at public transform boundaries
    /// uses this identity to detect one allocation arriving at two positions of a flattened signature, which
    /// [`Type::is_reference`] cannot see because it identifies references structurally only. Exactly the concrete
    /// runtime handles report an identity: a staged tracer reports [`None`] even when its type is a reference, and a
    /// reference-typed value whose family does not override this function is rejected by that validation instead of
    /// being silently accepted. The default returns [`None`].
    #[inline]
    fn reference_id(&self) -> Option<ReferenceId> {
        None
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
/// [`ArrayIrValue`](crate::ArrayIrValue) may be a backend array, a first-class runtime dimension, or an array
/// reference, mirroring the three member types of [`ArrayIrType`](crate::ArrayIrType). Most operations and transform
/// rules, however, are written against exactly one member kind (e.g., array-only rules consume values with `Value<Type
/// = ArrayType>`), and `ValueProjection<T>` is what lets them accept a composite value. [`Self::projected`] returns a
/// read-only view of the value as its `T`-typed member, [`Self::into_projected`] consumes the value and returns an
/// owned member representation, and [`Self::from_projected`] embeds a member representation back into the composite
/// value type. Both projection methods fail with a [`TypeError`] when the value holds a different member kind than
/// the requested one. The associated representations depend on how a value relates to its member:
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

/// Extension trait applying [`ValueProjection`] to a whole [`Parameterized`] tree at a boundary, rather than value by
/// value. Code that receives a tree of composite values (e.g., a model, a batch of inputs, a tuple of intermediates)
/// projects it once with [`project_parameters`](Self::project_parameters), computes with ordinary member capabilities,
/// and lifts the result back with [`lift_parameters`](Self::lift_parameters). Both directions preserve the tree's
/// structure exactly, and projection fails with the member-kind [`TypeError`] as soon as any leaf holds a different
/// member kind.
///
/// This trait is blanket-implemented for every [`Parameterized`] tree and has no items of its own to implement. Its
/// parameter `P` is the tree's leaf type, which plays a different role in each direction: it is the composite [`Value`]
/// being projected for [`project_parameters`](Self::project_parameters), and the member representation being lifted for
/// [`lift_parameters`](Self::lift_parameters). Each method's `where` clause carries the remaining requirements, so
/// whether a particular direction is available for a particular tree is decided per method at the call site. The
/// member [`Type`] is a method-level parameter, which keeps each call site down to a single turbofish: the projected
/// member type for one direction and the composite value type for the other.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ArrayIrValue, ArrayType, Mul, ParameterProjection, ProgramError};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let model = vec![ArrayIrValue::Array(Array::vector(vec![1.0, 2.0]))];
/// let arrays = model.project_parameters::<ArrayType>()?;
/// let squared = arrays.iter().map(|array| array.mul(array)).collect::<Result<Vec<_>, _>>()?;
/// let squared = squared.lift_parameters::<ArrayIrValue<Array>>()?;
/// assert_eq!(squared, vec![ArrayIrValue::Array(Array::vector(vec![1.0, 4.0]))]);
/// # Ok(())
/// # }
/// ```
pub trait ParameterProjection<P: Parameter>: Parameterized<P> {
    /// Projects every composite value [`Parameter`] of this [`Parameterized`] instance onto its `T`-typed member,
    /// preserving the [`Parameterized`] structure. This corresponds to [`ValueProjection::into_projected`] applied to
    /// every parameter, and it fails with that method's member-kind [`TypeError`] as soon as any parameter holds a
    /// different member kind.
    fn project_parameters<T: Type>(self) -> Result<Self::To<P::Projected>, ProgramError>
    where
        P: ValueProjection<T, Projected: Parameter>,
        Self::Family: ParameterizedFamily<P::Projected>,
    {
        let structure = self.parameter_structure();
        let parameters =
            self.into_parameters().map(ValueProjection::<T>::into_projected).collect::<Result<Vec<_>, _>>()?;
        Ok(Self::To::<P::Projected>::from_parameters(structure, parameters)?)
    }

    /// Lifts every member representation [`Parameter`] of this [`Parameterized`] instance back into the
    /// composite value type `V`, preserving the [`Parameterized`] structure. This function is the inverse
    /// of [`project_parameters`](Self::project_parameters) and is infallible per parameter, since
    /// [`ValueProjection::from_projected`] always accepts a member representation. The member [`Type`] is recovered
    /// from the parameters themselves (i.e., `P::Type`), so only the composite value type has to be named.
    fn lift_parameters<V: ValueProjection<P::Type, Projected = P>>(self) -> Result<Self::To<V>, ProgramError>
    where
        P: Typed,
        Self::Family: ParameterizedFamily<V>,
    {
        let structure = self.parameter_structure();
        let parameters = self.into_parameters().map(V::from_projected);
        Ok(Self::To::<V>::from_parameters(structure, parameters)?)
    }
}

impl<P: Parameter, Values: Parameterized<P>> ParameterProjection<P> for Values {}

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
/// The projection alone does not define how [`Operation`]s on the wrapped value dispatch. [`ProjectedContext`] provides
/// that behavior through this type's blanket [`Value`] implementation whenever the surrounding composite domains expose
/// the corresponding [`OperationProjection`].
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

    #[inline]
    fn capture_index(&self) -> Option<usize> {
        self.value.capture_index()
    }

    #[inline]
    fn reference_id(&self) -> Option<ReferenceId> {
        self.value.reference_id()
    }
}

impl<T: Type, C, V: Concretizable<C>> Concretizable<C> for ProjectedValue<T, V> {
    #[inline]
    fn concretize(&self) -> Result<C, ProgramError> {
        self.value.concretize()
    }
}

/// Renames a value's type-identity metadata by _rejection_ meaning that identity renamings and renamings that leave
/// the value's [`Type`] unchanged clone the value, while identity-changing renamings fail because the value has no
/// value-specific reconstruction. This one helper backs the [`Value::rename_type_identities`] default implementation
/// and composite member arms that deliberately keep the same semantics (e.g., eager references, whose shared
/// state must not be renamed through one alias), so the rejection policy and diagnostic have exactly one home.
pub(crate) fn rename_type_identities_by_rejection<V: Typed + Clone>(
    value: &V,
    renaming: &TypeIdentityRenaming<<V::Type as Type>::Identity>,
) -> Result<V, TypeError> {
    if renaming.is_identity() {
        return Ok(value.clone());
    }
    let current_type = value.r#type();
    let renamed_type = current_type.rename_identities(renaming)?;
    if current_type.as_ref() != &renamed_type {
        return Err(TypeError::invalid(format!(
            "cannot rename type identities in value of type {} without a value-specific \
            reconstruction implementation",
            current_type.as_ref(),
        )));
    }
    Ok(value.clone())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrValue, DataType, Dimension, DimensionBounds, DimensionType, DimensionValue, DimensionVariable,
        Shape,
    };
    use crate::contexts::EagerContext;
    use crate::operations::{Add, Mul};

    use super::*;

    #[test]
    fn test_value() {
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
        assert_eq!(value.validate_as_constant(), Ok(()));
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
    fn test_parameter_projection() {
        // A handwritten composite boundary projects its whole parameter tree once, computes with ordinary array
        // capabilities, and lifts the result back. The following example uses the Multi-Layer Perceptron (MLP) shape
        // where a model tree of weights and biases enters as composite values, and the layer arithmetic is homogeneous
        // array math.
        let model = (
            ArrayIrValue::<Array>::Array(Array::matrix(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0])),
            vec![ArrayIrValue::<Array>::Array(Array::vector(vec![5.0_f64, 6.0]))],
        );
        let (weights, biases) = model.project_parameters::<ArrayType>().unwrap();
        let weights = weights.mul(&weights).unwrap();
        let biases = biases.iter().map(|bias| bias.add(bias)).collect::<Result<Vec<_>, _>>().unwrap();
        let scaled = (weights, biases).lift_parameters::<ArrayIrValue<Array>>().unwrap();
        assert_eq!(
            scaled,
            (
                ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0_f64, 4.0, 9.0, 16.0])),
                vec![ArrayIrValue::Array(Array::vector(vec![10.0_f64, 12.0]))],
            ),
        );

        // A leaf holding a different member kind fails the projection with the canonical member diagnostic,
        // and the tree structure itself is preserved exactly across a projection round trip.
        let mixed = vec![
            ArrayIrValue::<Array>::Array(Array::scalar(1.0_f64)),
            ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()),
        ];
        assert_eq!(
            mixed.clone().project_parameters::<ArrayType>(),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
        let dimensions = vec![mixed[1].clone()].project_parameters::<DimensionType>().unwrap();
        let lifted = dimensions.lift_parameters::<ArrayIrValue<Array>>().unwrap();
        assert_eq!(lifted, vec![mixed[1].clone()]);
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
