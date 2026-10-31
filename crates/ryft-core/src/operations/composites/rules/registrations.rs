use std::any::{Any, TypeId};
use std::cell::Cell;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, Weak};
use std::thread::ThreadId;

use lru::LruCache;

use crate::differentiation::DifferentiationError;
use crate::operations::composites::operations::COMPOSITE_OPERATION_NAME;
use crate::operations::composites::rules::definitions::CustomRuleDefinition;
use crate::operations::composites::rules::specializations::{CustomRuleSource, specialization_error};
use crate::operations::composites::rules::{
    CustomRuleBackwardSpecializationKey, CustomRuleBatchingSpecializationKey, CustomRuleDerivedDefinitionKey,
    CustomRuleForwardSpecializationKey, CustomRuleSpecialization, CustomRuleSpecializationRequest,
};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Type, Typed};
use crate::specialization::{ReentrantSpecializationError, SpecializationCache, SpecializationCacheError};

/// Maximum number of specializations that each rule of a [`CustomRuleDefinition`] retains.
const CUSTOM_RULE_SPECIALIZATION_CAPACITY: usize = 16;

/// [`SpecializationCache`]es of one registered [`CustomRuleDefinition`], keyed by static information only and
/// owned by its [`CustomRuleRegistration`] handle.
struct CustomRuleCaches<V: Typed + Parameter, O> {
    /// Specializations of the forward mode differentiation rule.
    jvp_specializations:
        SpecializationCache<CustomRuleForwardSpecializationKey<V::Type>, Arc<CustomRuleSpecialization<V, O>>>,

    /// Specializations of the reverse mode differentiation forward rule.
    forward_specializations:
        SpecializationCache<CustomRuleForwardSpecializationKey<V::Type>, Arc<CustomRuleSpecialization<V, O>>>,

    /// Destination-specialized transposes of the reverse mode differentiation backward rule.
    backward_specializations:
        SpecializationCache<CustomRuleBackwardSpecializationKey<V::Type>, Arc<CustomRuleSpecialization<V, O>>>,

    /// Batched primal programs produced by the custom batching rule.
    batching_rule_specializations:
        SpecializationCache<CustomRuleBatchingSpecializationKey<V::Type>, Arc<CustomRuleSpecialization<V, O>>>,

    /// Registrations of the definitions derived from this definition, which these caches own so that they live at most
    /// as long as the specializations of this definition. Like the specializations, they are bounded, and an evicted
    /// registration frees the specializations of its definition.
    derived_registrations: Mutex<LruCache<CustomRuleDerivedDefinitionKey, CustomRuleRegistration<V, O>>>,

    /// Specializations of this definition converted into the families that contain its family, keyed by the [`TypeId`]
    /// of the containing family and by the request of the specialization from which each one was converted. Each is a
    /// shared [`CustomRuleSpecialization`] of that family.
    converted_specializations:
        SpecializationCache<(TypeId, CustomRuleSpecializationRequest<V::Type>), Arc<dyn Any + Send + Sync>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleCaches<V, O> {
    /// Creates empty [`CustomRuleCaches`] that retain up to [`CUSTOM_RULE_SPECIALIZATION_CAPACITY`] specializations
    /// of each rule and as many derived registrations.
    fn new() -> Self {
        Self {
            jvp_specializations: SpecializationCache::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY),
            forward_specializations: SpecializationCache::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY),
            backward_specializations: SpecializationCache::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY),
            batching_rule_specializations: SpecializationCache::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY),
            derived_registrations: Mutex::new(LruCache::new(
                NonZeroUsize::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY).unwrap(),
            )),
            converted_specializations: SpecializationCache::new(CUSTOM_RULE_SPECIALIZATION_CAPACITY),
        }
    }
}

/// Handle to a registered [`CustomRuleDefinition`], which owns the definition's specialization caches. Calls are staged
/// with a [`CustomRuleReference`] obtained from [`Self::reference`], and the specializations that their derivative
/// requests produce are cached for as long as some handle to the registration is alive. Once every handle is dropped,
/// the caches are freed, and later derivative requests of the remaining calls trace their rules again on every request
/// (enclosing region transform caches still apply).
///
/// # Referring to Definitions from Rules
///
/// A rule that calls a definition must capture a [`CustomRuleReference`] (or, for its own definition, the
/// [`WeakCustomRuleReference`] passed to [`Self::new_cyclic`]), never a handle, because a rule that owned the caches
/// of a definition whose programs reach the rule would form a reference cycle. Handles are deliberately not [`Sync`],
/// so the compiler rejects a rule, which must be [`Send`] and [`Sync`], that captures a handle directly. This guard is
/// not complete, because a handle wrapped in a [`Mutex`] is [`Sync`] whenever the handle is [`Send`], and so a rule
/// that captures a handle in any form is a misuse that the compiler does not always reject. References to other
/// definitions cannot form a cycle either, because a definition can only capture definitions that were registered
/// before it, except for itself through [`Self::new_cyclic`]. For example, two mutually recursive definitions are
/// registered by building the second one inside the [`Self::new_cyclic`] closure of the first one, capturing the
/// weak handle of the first one in the rules of the second one.
pub struct CustomRuleRegistration<V: Typed + Parameter, O> {
    /// Registered definition.
    definition: Arc<CustomRuleDefinition<V, O>>,

    /// Specialization caches of the definition, which only handles hold strongly.
    caches: Arc<CustomRuleCaches<V, O>>,

    /// Marker that keeps this handle from being [`Sync`].
    marker: PhantomData<Cell<()>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleRegistration<V, O> {
    /// Registers the provided definition with empty specialization caches, returning a new [`CustomRuleRegistration`].
    #[inline]
    pub fn new(definition: CustomRuleDefinition<V, O>) -> Self {
        Self::from_definition(Arc::new(definition))
    }

    /// Registers the definition returned by `build_fn`, which receives a weak reference to that same definition so that
    /// its rules can call it (i.e., recursively) without keeping it alive.
    pub fn new_cyclic<F: FnOnce(&WeakCustomRuleReference<V, O>) -> CustomRuleDefinition<V, O>>(build_fn: F) -> Self {
        let caches = Arc::new(CustomRuleCaches::new());
        let definition = Arc::new_cyclic(|definition| {
            build_fn(&WeakCustomRuleReference { definition: definition.clone(), caches: Arc::downgrade(&caches) })
        });
        Self { definition, caches, marker: PhantomData }
    }

    /// Registers the provided shared definition with empty specialization caches, which is how a derived definition
    /// that outlived its previous registration is registered again without changing its identity.
    fn from_definition(definition: Arc<CustomRuleDefinition<V, O>>) -> Self {
        Self { definition, caches: Arc::new(CustomRuleCaches::new()), marker: PhantomData }
    }

    /// Returns the [`CustomRuleDefinition`] owned by this registration.
    #[inline]
    pub fn definition(&self) -> &CustomRuleDefinition<V, O> {
        &self.definition
    }

    /// Returns a [`CustomRuleReference`] to this registration, with which calls are staged and rules refer to it.
    #[inline]
    pub fn reference(&self) -> CustomRuleReference<V, O> {
        CustomRuleReference { definition: self.definition.clone(), caches: Arc::downgrade(&self.caches) }
    }
}

// TODO(eaplatanios): Review from here onwards.

#[cfg(test)]
impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleRegistration<V, O> {
    /// Returns the specialization caches, which tests use to observe lazy tracing and cache ownership.
    fn caches(&self) -> &Arc<CustomRuleCaches<V, O>> {
        &self.caches
    }

    /// Returns the JVP specialization cache for tests that observe lazy tracing, eviction, and invalidation.
    pub(super) fn jvp_specializations(
        &self,
    ) -> &SpecializationCache<CustomRuleForwardSpecializationKey<V::Type>, Arc<CustomRuleSpecialization<V, O>>> {
        &self.caches.jvp_specializations
    }

    /// Returns the number of derived registrations retained by this registration's caches.
    pub(super) fn derived_registration_count(&self) -> usize {
        self.caches.derived_registrations.lock().expect("custom rule derivation mutex is poisoned").len()
    }

    /// Returns the keys of the cached forward mode, reverse mode forward, and reverse mode backward rule
    /// specializations, in that order, which tests of the operations use to observe lazy tracing.
    #[allow(clippy::type_complexity)]
    pub(in crate::operations::composites) fn cached_specialization_keys(
        &self,
    ) -> (
        Vec<CustomRuleForwardSpecializationKey<V::Type>>,
        Vec<CustomRuleForwardSpecializationKey<V::Type>>,
        Vec<CustomRuleBackwardSpecializationKey<V::Type>>,
    ) {
        (
            self.caches.jvp_specializations.keys(),
            self.caches.forward_specializations.keys(),
            self.caches.backward_specializations.keys(),
        )
    }

    /// Returns a probe that reports whether the registered definition and its specialization caches are still alive,
    /// in that order, which tests of the operations use to observe ownership after this handle is dropped.
    pub(in crate::operations::composites) fn ownership_probe(&self) -> impl Fn() -> (bool, bool) + use<V, O> {
        let (definition, caches) = (Arc::downgrade(&self.definition), Arc::downgrade(&self.caches));
        move || (definition.strong_count() > 0, caches.strong_count() > 0)
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for CustomRuleRegistration<V, O> {
    #[inline]
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone(), marker: PhantomData }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for CustomRuleRegistration<V, O> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("CustomRuleRegistration").field(&self.definition).finish()
    }
}

/// Reference to a registered [`CustomRuleDefinition`], with which calls are staged and rules refer to definitions. It
/// keeps the definition alive and uses the registration's specialization caches while some
/// [`CustomRuleRegistration`] handle keeps them alive, producing uncached specializations otherwise. Equality and
/// hashing use the identity of the definition.
pub struct CustomRuleReference<V: Typed + Parameter, O> {
    /// Referenced definition.
    pub(super) definition: Arc<CustomRuleDefinition<V, O>>,

    /// Specialization caches of the definition, which are alive exactly while some handle to it is alive.
    caches: Weak<CustomRuleCaches<V, O>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleReference<V, O> {
    /// Returns the referenced definition.
    #[inline]
    pub fn definition(&self) -> &CustomRuleDefinition<V, O> {
        &self.definition
    }

    /// Returns a reference to the derived `definition`, retaining its registration in this source's caches while they
    /// are alive. An existing registration for `key` is reused. Without the source's caches, the returned reference
    /// has no cache owner and produces its specializations without caching.
    pub(super) fn derived_reference(
        &self,
        key: CustomRuleDerivedDefinitionKey,
        definition: Arc<CustomRuleDefinition<V, O>>,
    ) -> Self {
        let Some(caches) = self.caches.upgrade() else {
            return Self { definition, caches: Weak::new() };
        };
        let mut registrations = caches.derived_registrations.lock().expect("custom rule derivation mutex is poisoned");
        if let Some(registration) = registrations.get(&key) {
            return registration.reference();
        }
        let registration = CustomRuleRegistration::from_definition(definition);
        let reference = registration.reference();

        // An evicted registration is dropped after the lock is released, because dropping its caches may drop
        // programs whose destructors reach other definitions.
        let evicted = registrations.push(key, registration);
        drop(registrations);
        drop(evicted);
        reference
    }

    /// Returns the specialization for `request`, produced by `produce_fn` on a miss. The specialization is retained in
    /// the registration's caches while they are alive, and produced without caching otherwise. A recursive request for
    /// the same specialization on the same thread is rejected with the [`ReentrantSpecializationError`] that a cache
    /// reports, including when the caches were dropped.
    pub(super) fn specialize<F>(
        &self,
        request: CustomRuleSpecializationRequest<V::Type>,
        produce_fn: F,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>
    where
        F: FnOnce() -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>,
    {
        /// Removes an in-flight marker when production finishes, fails, or unwinds.
        struct InFlightMarker<'o, T: Type + Eq + Hash> {
            /// Set from which the marker is removed.
            in_flight: &'o Mutex<HashSet<(ThreadId, CustomRuleSpecializationRequest<T>)>>,

            /// Marker to remove.
            marker: (ThreadId, CustomRuleSpecializationRequest<T>),
        }

        impl<T: Type + Eq + Hash> Drop for InFlightMarker<'_, T> {
            fn drop(&mut self) {
                self.in_flight.lock().expect("custom rule in-flight mutex is poisoned").remove(&self.marker);
            }
        }

        let Some(caches) = self.caches.upgrade() else {
            let marker = (std::thread::current().id(), request);
            if !self
                .definition
                .uncached_specializations()
                .lock()
                .expect("custom rule in-flight mutex is poisoned")
                .insert(marker.clone())
            {
                return Err(specialization_error(
                    self.name(),
                    SpecializationCacheError::<DifferentiationError>::Reentrant(ReentrantSpecializationError),
                ));
            }
            let _marker = InFlightMarker { in_flight: self.definition.uncached_specializations(), marker };
            return produce_fn();
        };
        match request {
            CustomRuleSpecializationRequest::Jvp(key) => {
                caches.jvp_specializations.get_or_try_insert_with(key, produce_fn)
            }
            CustomRuleSpecializationRequest::Forward(key) => {
                caches.forward_specializations.get_or_try_insert_with(key, produce_fn)
            }
            CustomRuleSpecializationRequest::Backward(key) => {
                caches.backward_specializations.get_or_try_insert_with(key, produce_fn)
            }
            CustomRuleSpecializationRequest::BatchingRule(key) => {
                caches.batching_rule_specializations.get_or_try_insert_with(key, produce_fn)
            }
        }
        .map_err(|error| specialization_error(self.name(), error))
    }

    /// Returns the specialization for `request` converted into the family `(Vc, Oc)`, which contains this reference's
    /// family, produced by `produce_fn` on a miss. Like the specializations themselves, it is retained in the
    /// registration's caches while they are alive, and produced without caching otherwise.
    pub(super) fn converted_specialization<Vc, Oc, F>(
        &self,
        request: CustomRuleSpecializationRequest<V::Type>,
        produce_fn: F,
    ) -> Result<Arc<CustomRuleSpecialization<Vc, Oc>>, DifferentiationError>
    where
        Vc: 'static + Typed + Parameter,
        Oc: 'static,
        CustomRuleSpecialization<Vc, Oc>: Send + Sync,
        F: FnOnce() -> Result<Arc<CustomRuleSpecialization<Vc, Oc>>, DifferentiationError>,
    {
        let Some(caches) = self.caches.upgrade() else {
            return produce_fn();
        };
        let specialization = caches
            .converted_specializations
            .get_or_try_insert_with((TypeId::of::<(Vc, Oc)>(), request), || {
                Ok::<_, DifferentiationError>(produce_fn()? as Arc<dyn Any + Send + Sync>)
            })
            .map_err(|error| specialization_error(self.name(), error))?;

        // The key names the family, so the retained specialization always belongs to it.
        Ok(specialization.downcast::<CustomRuleSpecialization<Vc, Oc>>().unwrap())
    }
}

#[cfg(test)]
impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleReference<V, O> {
    /// Returns whether this reference's specialization caches still have an owner.
    pub(super) fn has_caches(&self) -> bool {
        self.caches.upgrade().is_some()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for CustomRuleReference<V, O> {
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone() }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for CustomRuleReference<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("CustomRuleReference").field(&self.definition).finish()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> PartialEq for CustomRuleReference<V, O> {
    fn eq(&self, other: &Self) -> bool {
        self.definition.id() == other.definition.id()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Eq for CustomRuleReference<V, O> {}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Hash for CustomRuleReference<V, O> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.definition.id().hash(state);
    }
}

/// Weak reference to a [`CustomRuleDefinition`], which keeps neither the definition nor its caches alive. It is
/// how a definition's rules refer to that same definition (refer to [`CustomRuleRegistration::new_cyclic`]).
pub struct WeakCustomRuleReference<V: Typed + Parameter, O> {
    /// Registered definition.
    definition: Weak<CustomRuleDefinition<V, O>>,

    /// Specialization caches of the definition.
    caches: Weak<CustomRuleCaches<V, O>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> WeakCustomRuleReference<V, O> {
    /// Upgrades this weak reference to a [`CustomRuleReference`] with which rules stage calls of their own definition.
    /// It succeeds while the definition is alive, even if the registration's specialization caches were dropped.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when the definition is unavailable.
    pub fn upgrade(&self) -> Result<CustomRuleReference<V, O>, ProgramError> {
        let definition = self.definition.upgrade().ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("the `{COMPOSITE_OPERATION_NAME}` definition was dropped"),
        })?;
        Ok(CustomRuleReference { definition, caches: self.caches.clone() })
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for WeakCustomRuleReference<V, O> {
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone() }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for WeakCustomRuleReference<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("WeakCustomRuleReference").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::contexts::Context;
    use crate::operations::composites::operations::CompositeOperation;
    use crate::operations::composites::rules::CustomRuleDerivationKind;
    use crate::operations::composites::rules::specializations::CustomRuleSpecializer;
    use crate::operations::composites::tests::{
        MemberDefinition, RuleCounters, TestDefinition, call_jvp, cube_definition, cube_program, custom_rule_program,
    };
    use crate::tests::hash_of;

    use super::*;

    #[test]
    fn test_custom_rule_registration_new() {
        // A registration owns empty caches, which its clones share, and its references share its definition.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        assert!(registration.caches().jvp_specializations.is_empty());
        assert!(registration.caches().forward_specializations.is_empty());
        assert!(registration.caches().backward_specializations.is_empty());
        assert!(registration.caches().batching_rule_specializations.is_empty());
        assert_eq!(registration.caches().derived_registrations.lock().unwrap().len(), 0);
        assert_eq!(registration.reference(), registration.reference());
        let clone = registration.clone();
        assert!(std::ptr::eq(clone.definition(), registration.definition()));
        assert!(Arc::ptr_eq(clone.caches(), registration.caches()));
        assert_eq!(
            format!("{registration:?}"),
            "CustomRuleRegistration(CustomRuleDefinition { name: \"cube\", has_jvp: true, derives_jvp_from_primal: \
             false, has_vjp: false, .. })",
        );
    }

    #[test]
    fn test_custom_rule_registration_ownership() {
        // The outer JVP rule calls the inner definition through a reference, so the outer rule's cached specialization
        // retains the inner definition but not its caches, which only the inner handle owns.
        let counters = Arc::new(RuleCounters::default());
        let inner = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let (retained_inner, retained_inner_caches) =
            (Arc::downgrade(&inner.definition), Arc::downgrade(&inner.caches()));
        let inner_reference = inner.reference();
        let outer = CustomRuleRegistration::new(TestDefinition::new("outer").with_jvp(move |primals, tangents| {
            let r#type = primals[0].r#type().into_owned();
            let operation = CompositeOperation::new(inner_reference.clone());
            let outputs = primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
            Ok((outputs, vec![tangents[0].clone()]))
        }));
        let (retained_outer, retained_outer_caches) =
            (Arc::downgrade(&outer.definition), Arc::downgrade(&outer.caches()));
        let program = custom_rule_program(&outer, ArrayType::scalar(DataType::F64));
        let linearization = program.linearize().unwrap();
        assert_eq!(counters.counts(), (0, 0, 0));
        assert_eq!(outer.caches().jvp_specializations.len(), 1);

        // Dropping the handles frees their caches, including the outer specialization that called the inner definition,
        // while the staged programs keep both definitions alive until they are dropped as well.
        drop((inner, outer));
        assert!(retained_outer_caches.upgrade().is_none());
        assert!(retained_inner_caches.upgrade().is_none());
        assert!(retained_outer.upgrade().is_some());
        drop((program, linearization));
        assert!(retained_outer.upgrade().is_none());
        assert!(retained_inner.upgrade().is_none());
    }

    #[test]
    fn test_custom_rule_registration_drop() {
        // Once the handle is dropped, the remaining calls trace their rules again on every derivative request, with
        // results identical to the cached ones.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let cached = program.linearize().unwrap().tangent().to_string();
        drop(definition);
        for expected_invocations in [2, 3] {
            assert_eq!(program.linearize().unwrap().tangent().to_string(), cached);
            assert_eq!(counters.counts(), (expected_invocations, 0, 0));
        }

        // A rule that requests its own specialization while it is being traced is rejected, with and without a live
        // handle, and a rejected request leaves nothing in flight.
        let definition = CustomRuleRegistration::new_cyclic(|this| {
            let this = this.clone();
            TestDefinition::new("reentrant").with_jvp(move |primals, tangents| {
                let r#type = primals[0].r#type().into_owned();
                let key = CustomRuleForwardSpecializationKey {
                    input_types: vec![r#type.clone()],
                    output_types: vec![r#type],
                    kept_outputs: None,
                    non_differentiated_count: 0,
                    tangent_activity: vec![true],
                    levels: Vec::new(),
                    discharged: false,
                };
                this.upgrade()?
                    .jvp_specialization(key)
                    .map_err(|error| ProgramError::InvalidArgument { message: error.to_string() })?;
                Ok((primals.to_vec(), tangents.to_vec()))
            })
        });
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let reentrant = "`composite` `reentrant` makes a recursive request for a specialization that is already being \
                         produced on this thread";
        assert!(matches!(
            program.jvp(),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message })) if message == reentrant,
        ));
        let retained = Arc::downgrade(&definition.definition);
        drop(definition);
        for _ in 0..2 {
            assert!(matches!(
                program.jvp(),
                Err(DifferentiationError::Program(ProgramError::InvalidArgument { message })) if message == reentrant,
            ));
            assert!(!retained.upgrade().unwrap().has_uncached_specializations());
        }
    }

    #[test]
    fn test_custom_rule_registration_new_cyclic() {
        // The build function receives a weak reference to the definition that it builds, which refers to that same
        // definition and its caches once it is built, but keeps neither of them alive.
        let mut weak = None;
        let registration = CustomRuleRegistration::new_cyclic(|this| {
            weak = Some(this.clone());
            TestDefinition::new("cyclic")
        });
        let weak = weak.unwrap();
        let reference = weak.upgrade().unwrap();
        assert_eq!(reference, registration.reference());
        assert!(Arc::ptr_eq(&reference.caches.upgrade().unwrap(), registration.caches()));
        drop(reference);
        let (definition, caches) = (Arc::downgrade(&registration.definition), Arc::downgrade(registration.caches()));
        drop(registration);
        assert!(definition.upgrade().is_none());
        assert!(caches.upgrade().is_none());
    }

    #[test]
    fn test_custom_rule_registration_ownership_self_recursive() {
        // The JVP rule calls its own definition through the weak handle of `new_cyclic`, so its cached specialization
        // retains the definition. Only the handle owns the caches, so dropping it frees that specialization, and the
        // definition itself lives exactly as long as the staged programs that call it.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new_cyclic(|this| {
            let (this, counters) = (this.clone(), counters.clone());
            TestDefinition::new("recursive").with_jvp(move |primals, tangents| {
                counters.jvp.fetch_add(1, Ordering::SeqCst);
                let r#type = primals[0].r#type().into_owned();
                let operation = CompositeOperation::new(this.upgrade()?);
                let outputs =
                    primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
                Ok((outputs, vec![primals[0].clone() * primals[0].clone() * tangents[0].clone()]))
            })
        });
        let (retained_definition, retained_caches) =
            (Arc::downgrade(&definition.definition), Arc::downgrade(&definition.caches()));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let linearization = program.linearize().unwrap();
        assert_eq!(counters.counts(), (1, 0, 0));
        assert_eq!(definition.caches().jvp_specializations.len(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = composite [name=\"recursive\"] %0 [
                    primal={
                        lambda %0:f64[] .
                        let %1:f64[] = mul %0 %0
                            %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                    %2:f64[] = mul %0 %0
                in (%1, %2)
            "}
            .trim_end(),
        );
        drop(definition);
        assert!(retained_caches.upgrade().is_none());
        assert!(retained_definition.upgrade().is_some());
        drop((program, linearization));
        assert!(retained_definition.upgrade().is_none());
    }

    #[test]
    fn test_custom_rule_registration_ownership_mutually_recursive() {
        // The rules of `a` call `b` through a reference and the rules of `b` call `a` through its weak handle, which is
        // how mutually recursive definitions are registered without a cycle: `b` is registered inside the `new_cyclic`
        // closure of `a`. Filling both caches (the second-order linearization traces the rule of `b`) and then
        // dropping both handles and every program must free every definition and cache.
        let mut b = None;
        let a = CustomRuleRegistration::new_cyclic(|a| {
            let a = a.clone();
            let registration =
                CustomRuleRegistration::new(TestDefinition::new("b").with_jvp(move |primals, tangents| {
                    let r#type = primals[0].r#type().into_owned();
                    let operation = CompositeOperation::new(a.upgrade()?);
                    let outputs =
                        primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
                    Ok((outputs, vec![tangents[0].clone()]))
                }));
            let reference = registration.reference();
            b = Some(registration);
            TestDefinition::new("a").with_jvp(move |primals, tangents| {
                let r#type = primals[0].r#type().into_owned();
                let operation = CompositeOperation::new(reference.clone());
                let outputs =
                    primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
                Ok((outputs, vec![tangents[0].clone()]))
            })
        });
        let b = b.unwrap();
        let probes = [
            (Arc::downgrade(&a.definition), Arc::downgrade(&a.caches())),
            (Arc::downgrade(&b.definition), Arc::downgrade(&b.caches())),
        ];
        let program = custom_rule_program(&a, ArrayType::scalar(DataType::F64));
        let first = program.linearize().unwrap();
        let second = first.primal().linearize().unwrap();
        assert_eq!(a.caches().jvp_specializations.len(), 1);
        assert_eq!(b.caches().jvp_specializations.len(), 1);
        drop((a, b));
        assert!(probes.iter().all(|(_, caches)| caches.upgrade().is_none()));
        drop((program, first, second));
        assert!(probes.iter().all(|(definition, _)| definition.upgrade().is_none()));
    }

    #[test]
    fn test_custom_rule_registration_reference() {
        // References compare and hash by the identity of the definition, and they keep the definition alive but not the
        // caches, which only the registration owns.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let reference = registration.reference();
        assert_eq!(reference, registration.reference());
        assert_eq!(reference.clone(), reference);
        assert_eq!(hash_of(&reference), hash_of(&registration.reference()));
        assert_ne!(reference, CustomRuleRegistration::new(TestDefinition::new("cube")).reference());
        assert!(Arc::ptr_eq(&reference.caches.upgrade().unwrap(), registration.caches()));
        assert_eq!(
            format!("{reference:?}"),
            "CustomRuleReference(CustomRuleDefinition { name: \"cube\", has_jvp: false, derives_jvp_from_primal: \
             false, has_vjp: false, .. })",
        );
        drop(registration);
        assert!(reference.caches.upgrade().is_none());
        assert_eq!(reference.definition().name(), "cube");
    }

    #[test]
    fn test_custom_rule_reference_derived_reference_capacity() {
        // The source registration retains as many derived registrations as specializations of each rule. A derived
        // definition that is still alive is reused for its key, whether or not its registration was evicted or the
        // source registration was dropped, so that it keeps its identity.
        let registration = CustomRuleRegistration::new(
            MemberDefinition::new("sine")
                .with_jvp_from_primal()
                .with_batching_support()
                .with_batching_rule(|inputs, input_axes, _, _| Ok((inputs.to_vec(), input_axes.to_vec()))),
        );
        let key = |input_count| CustomRuleDerivedDefinitionKey {
            kind: CustomRuleDerivationKind::Jvp,
            input_count,
            active_input_indices: vec![0],
            output_tangent_mask: vec![true],
        };
        let derived = (1..=CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1)
            .map(|input_count| registration.reference().derived(key(input_count)).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            registration.caches().derived_registrations.lock().unwrap().len(),
            CUSTOM_RULE_SPECIALIZATION_CAPACITY,
        );
        assert_eq!(registration.reference().derived(key(1)).unwrap(), derived[0]);
        let source = registration.reference();
        drop(registration);
        assert_eq!(source.derived(key(2)).unwrap(), derived[1]);
    }

    #[test]
    fn test_custom_rule_reference_specialize_eviction() {
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let jvp = |extent: usize| {
            let x = Array::vector(vec![2f64; extent]).unwrap();
            let tangent = Array::vector(vec![1f64; extent]).unwrap();
            assert_eq!(
                call_jvp(&definition, x, tangent),
                Ok((
                    vec![Array::vector(vec![8f64; extent]).unwrap()],
                    vec![Array::vector(vec![4f64; extent]).unwrap()]
                )),
            );
        };
        let specialization = |extent: usize| {
            let r#type = ArrayType::new_static(DataType::F64, [extent]);
            let key = CustomRuleForwardSpecializationKey {
                input_types: vec![r#type.clone()],
                output_types: vec![r#type],
                kept_outputs: None,
                non_differentiated_count: 0,
                tangent_activity: vec![true],
                levels: Vec::new(),
                discharged: false,
            };
            Arc::downgrade(&definition.reference().jvp_specialization(key).unwrap())
        };
        let retained_definition = Arc::downgrade(&definition.reference().definition);
        let baseline = retained_definition.strong_count();
        let mut probes = Vec::new();
        for extent in 1..=CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1 {
            jvp(extent);
            probes.push(specialization(extent));
        }
        assert_eq!(definition.caches().jvp_specializations.len(), CUSTOM_RULE_SPECIALIZATION_CAPACITY);
        assert_eq!(counters.counts(), (CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1, 0, 0));

        // Bounded caches bound the retained memory: the evicted specialization is freed, and no cached program retains
        // the definition.
        assert!(probes[0].upgrade().is_none());
        assert!(probes[1..].iter().all(|probe| probe.upgrade().is_some()));
        assert_eq!(retained_definition.strong_count(), baseline);

        // The evicted oldest specialization is traced again, while a resident one is reused.
        jvp(1);
        jvp(CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1);
        assert_eq!(counters.counts(), (CUSTOM_RULE_SPECIALIZATION_CAPACITY + 2, 0, 0));
    }

    #[test]
    fn test_weak_custom_rule_reference_upgrade() {
        // A weak reference upgrades while the definition is alive (e.g., while a reference keeps it alive after the
        // registration is dropped), and fails once the definition was dropped.
        let mut weak = None;
        let registration = CustomRuleRegistration::new_cyclic(|this| {
            weak = Some(this.clone());
            TestDefinition::new("cyclic")
        });
        let weak = weak.unwrap();
        assert_eq!(format!("{weak:?}"), "WeakCustomRuleReference { .. }");
        assert_eq!(weak.upgrade().unwrap(), registration.reference());
        let reference = registration.reference();
        drop(registration);
        let upgraded = weak.upgrade().unwrap();
        assert_eq!(upgraded, reference);
        assert!(upgraded.caches.upgrade().is_none());
        drop((reference, upgraded));
        assert!(matches!(
            weak.upgrade(),
            Err(ProgramError::InvalidArgument { message }) if message == "the `composite` definition was dropped",
        ));
    }
}
