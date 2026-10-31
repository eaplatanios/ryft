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

    /// Returns a [`WeakCustomRuleReference`] to this registration, which keeps neither the definition nor its
    /// specialization caches alive (e.g., to observe a registration without extending the lifetime of either).
    #[inline]
    pub fn downgrade(&self) -> WeakCustomRuleReference<V, O> {
        WeakCustomRuleReference { definition: Arc::downgrade(&self.definition), caches: Arc::downgrade(&self.caches) }
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
/// keeps the definition alive and uses the registration's specialization caches while some [`CustomRuleRegistration`]
/// handle keeps them alive, producing uncached specializations otherwise. Equality and hashing use the identity of the
/// definition.
pub struct CustomRuleReference<V: Typed + Parameter, O> {
    /// Referenced definition.
    definition: Arc<CustomRuleDefinition<V, O>>,

    /// Specialization caches of the definition, which are alive exactly while some handle to it is alive.
    caches: Weak<CustomRuleCaches<V, O>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleReference<V, O> {
    /// Returns the referenced definition.
    #[inline]
    pub fn definition(&self) -> &CustomRuleDefinition<V, O> {
        &self.definition
    }

    /// Returns `true` if the specializations requested through this reference are cached, which they are while some
    /// [`CustomRuleRegistration`] handle to its definition is alive. Otherwise, every request traces its rule again.
    #[inline]
    pub fn is_cached(&self) -> bool {
        self.caches.strong_count() > 0
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

        let _evicted = registrations.push(key, registration);

        // Locals drop in reverse declaration order, so release the mutex before `_evicted` drops at function exit.
        // Its cached programs may run destructors that reenter the cache, which would otherwise deadlock.
        drop(registrations);

        reference
    }

    /// Returns the specialization for `request`, produced by `produce_fn` on a miss. The specialization is retained in
    /// the registration's caches while they are alive, and produced without caching otherwise. A recursive request for
    /// the same specialization on the same thread is rejected with the [`ReentrantSpecializationError`] that a cache
    /// reports, including when the caches were dropped.
    pub(super) fn specialization<F: FnOnce() -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError>>(
        &self,
        request: CustomRuleSpecializationRequest<V::Type>,
        produce_fn: F,
    ) -> Result<Arc<CustomRuleSpecialization<V, O>>, DifferentiationError> {
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
    pub(super) fn converted_specialization<
        VC: 'static + Typed + Parameter,
        OC: 'static,
        F: FnOnce() -> Result<Arc<CustomRuleSpecialization<VC, OC>>, DifferentiationError>,
    >(
        &self,
        request: CustomRuleSpecializationRequest<V::Type>,
        produce_fn: F,
    ) -> Result<Arc<CustomRuleSpecialization<VC, OC>>, DifferentiationError>
    where
        CustomRuleSpecialization<VC, OC>: Send + Sync,
    {
        let Some(caches) = self.caches.upgrade() else {
            return produce_fn();
        };
        let specialization = caches
            .converted_specializations
            .get_or_try_insert_with((TypeId::of::<(VC, OC)>(), request), || {
                Ok::<_, DifferentiationError>(produce_fn()? as Arc<dyn Any + Send + Sync>)
            })
            .map_err(|error| specialization_error(self.name(), error))?;

        // The key names the family, so the retained specialization always belongs to it.
        Ok(specialization.downcast::<CustomRuleSpecialization<VC, OC>>().unwrap())
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for CustomRuleReference<V, O> {
    #[inline]
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone() }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for CustomRuleReference<V, O> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("CustomRuleReference").field(&self.definition).finish()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> PartialEq for CustomRuleReference<V, O> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.definition.id() == other.definition.id()
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Eq for CustomRuleReference<V, O> {}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Hash for CustomRuleReference<V, O> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.definition.id().hash(state);
    }
}

/// Weak reference to a [`CustomRuleDefinition`], which keeps neither the definition nor its caches alive. It is
/// how a definition's rules refer to that same definition. Refer to [`CustomRuleRegistration::new_cyclic`] for more
/// information.
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
    #[inline]
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone() }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for WeakCustomRuleReference<V, O> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("WeakCustomRuleReference").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, ShardingDimension};
    use crate::batching::{BatchAxis, BatchingLevel, BatchingLevelExtent};
    use crate::contexts::Context;
    use crate::operations::composites::operations::CompositeOperation;
    use crate::operations::composites::rules::specializations::CustomRuleSpecializer;
    use crate::operations::composites::rules::{CustomRuleBatchingLevel, CustomRuleDerivationKind, CustomRuleTracer};
    use crate::operations::composites::tests::{
        RuleCounters, TestDefinition, call_jvp, cube_definition, cube_program, custom_rule_program,
    };
    use crate::tests::hash_of;
    use crate::tracing::TracingContext;

    use super::*;

    /// Returns the specialization key of an unbatched and undischarged call with one active input of type `r#type`
    /// and one output of that same type.
    fn unbatched_key(r#type: ArrayType) -> CustomRuleForwardSpecializationKey<ArrayType> {
        CustomRuleForwardSpecializationKey {
            input_types: vec![r#type.clone()],
            output_types: vec![r#type],
            kept_outputs: None,
            non_differentiated_count: 0,
            tangent_activity: vec![true],
            levels: Vec::new(),
            discharged: false,
        }
    }

    /// Returns the key of the definition derived for the Jacobian-Vector Product (JVP) of a call with `input_count`
    /// inputs, of which only the first one is active, and one output.
    fn jvp_derivation_key(input_count: usize) -> CustomRuleDerivedDefinitionKey {
        CustomRuleDerivedDefinitionKey {
            kind: CustomRuleDerivationKind::Jvp,
            input_count,
            active_input_indices: vec![0],
            output_tangent_mask: vec![true],
        }
    }

    #[test]
    fn test_custom_rule_registration_new() {
        // A registration owns the provided definition together with empty caches.
        let counters = Arc::new(RuleCounters::default());
        let registration = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        assert_eq!(registration.definition().name(), "cube");
        assert!(registration.caches.jvp_specializations.is_empty());
        assert!(registration.caches.forward_specializations.is_empty());
        assert!(registration.caches.backward_specializations.is_empty());
        assert!(registration.caches.batching_rule_specializations.is_empty());
        assert_eq!(registration.caches.derived_registrations.lock().unwrap().len(), 0);
        assert!(registration.caches.converted_specializations.is_empty());
        assert_eq!(counters.counts(), (0, 0, 0));
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
        assert!(Arc::ptr_eq(&reference.caches.upgrade().unwrap(), &registration.caches));
        drop(reference);
        let (definition, caches) = (Arc::downgrade(&registration.definition), Arc::downgrade(&registration.caches));
        drop(registration);
        assert!(definition.upgrade().is_none());
        assert!(caches.upgrade().is_none());
    }

    #[test]
    fn test_custom_rule_registration_new_cyclic_self_recursive() {
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
            (Arc::downgrade(&definition.definition), Arc::downgrade(&definition.caches));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let linearization = program.linearize().unwrap();
        assert_eq!(counters.counts(), (1, 0, 0));
        assert_eq!(definition.caches.jvp_specializations.len(), 1);
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
    fn test_custom_rule_registration_new_cyclic_mutually_recursive() {
        // The rules of `a` call `b` through a reference and the rules of `b` call `a` through its weak handle, which
        // is how mutually recursive definitions are registered without a cycle (i.e., `b` is registered inside the
        // `new_cyclic` closure of `a`). Filling both caches (the second-order linearization traces the rule of `b`)
        // and then dropping both handles and every program must free every definition and cache.
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
            (Arc::downgrade(&a.definition), Arc::downgrade(&a.caches)),
            (Arc::downgrade(&b.definition), Arc::downgrade(&b.caches)),
        ];
        let program = custom_rule_program(&a, ArrayType::scalar(DataType::F64));
        let first = program.linearize().unwrap();
        let second = first.primal().linearize().unwrap();
        assert_eq!(a.caches.jvp_specializations.len(), 1);
        assert_eq!(b.caches.jvp_specializations.len(), 1);
        drop((a, b));
        assert!(probes.iter().all(|(_, caches)| caches.upgrade().is_none()));
        drop((program, first, second));
        assert!(probes.iter().all(|(definition, _)| definition.upgrade().is_none()));
    }

    #[test]
    fn test_custom_rule_registration_definition() {
        // The registered definition is the one that the references of the registration refer to.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        assert_eq!(registration.definition().name(), "cube");
        assert!(std::ptr::eq(registration.definition(), registration.reference().definition()));
    }

    #[test]
    fn test_custom_rule_registration_reference() {
        // A reference refers to the registered definition and caches, and it keeps the definition alive but not the
        // caches, which only the registration owns.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let reference = registration.reference();
        assert!(Arc::ptr_eq(&reference.definition, &registration.definition));
        assert!(Arc::ptr_eq(&reference.caches.upgrade().unwrap(), &registration.caches));
        drop(registration);
        assert!(!reference.is_cached());
        assert_eq!(reference.definition().name(), "cube");
    }

    #[test]
    fn test_custom_rule_registration_reference_captured_by_rule() {
        // The outer JVP rule calls the inner definition through a reference, so the outer rule's cached specialization
        // retains the inner definition but not its caches, which only the inner handle owns.
        let counters = Arc::new(RuleCounters::default());
        let inner = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let (retained_inner, retained_inner_caches) =
            (Arc::downgrade(&inner.definition), Arc::downgrade(&inner.caches));
        let inner_reference = inner.reference();
        let outer = CustomRuleRegistration::new(TestDefinition::new("outer").with_jvp(move |primals, tangents| {
            let r#type = primals[0].r#type().into_owned();
            let operation = CompositeOperation::new(inner_reference.clone());
            let outputs = primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
            Ok((outputs, vec![tangents[0].clone()]))
        }));
        let (retained_outer, retained_outer_caches) =
            (Arc::downgrade(&outer.definition), Arc::downgrade(&outer.caches));
        let program = custom_rule_program(&outer, ArrayType::scalar(DataType::F64));
        let linearization = program.linearize().unwrap();
        assert_eq!(counters.counts(), (0, 0, 0));
        assert_eq!(outer.caches.jvp_specializations.len(), 1);

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
    fn test_custom_rule_registration_downgrade() {
        // The weak reference refers to the registration while it is alive, keeps neither its definition nor its caches
        // alive, and no longer upgrades once both were dropped.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let weak = registration.downgrade();
        let reference = weak.upgrade().unwrap();
        assert_eq!(reference, registration.reference());
        assert!(reference.is_cached());
        drop((reference, registration));
        assert!(matches!(
            weak.upgrade(),
            Err(ProgramError::InvalidArgument { message }) if message == "the `composite` definition was dropped",
        ));
    }

    #[test]
    fn test_custom_rule_registration_clone() {
        // Clones share the definition and the caches, which stay alive while any of the handles is alive.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let clone = registration.clone();
        assert!(Arc::ptr_eq(&clone.definition, &registration.definition));
        assert!(Arc::ptr_eq(&clone.caches, &registration.caches));
        let reference = registration.reference();
        drop(registration);
        assert!(reference.is_cached());
        drop(clone);
        assert!(!reference.is_cached());
    }

    #[test]
    fn test_custom_rule_registration_debug() {
        let counters = Arc::new(RuleCounters::default());
        assert_eq!(
            format!("{:?}", CustomRuleRegistration::new(cube_definition(&counters, true, false))),
            "CustomRuleRegistration(CustomRuleDefinition { name: \"cube\", has_jvp: true, derives_jvp_from_primal: \
             false, has_vjp: false, .. })",
        );
    }

    #[test]
    fn test_custom_rule_reference_definition() {
        // The referenced definition stays available after the registration was dropped.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let reference = registration.reference();
        assert!(std::ptr::eq(reference.definition(), registration.definition()));
        drop(registration);
        assert_eq!(reference.definition().name(), "cube");
    }

    #[test]
    fn test_custom_rule_reference_is_cached() {
        // A reference is cached while some handle to its registration is alive, including a clone of the original one.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let reference = registration.reference();
        assert!(reference.is_cached());
        let clone = registration.clone();
        drop(registration);
        assert!(reference.is_cached());
        drop(clone);
        assert!(!reference.is_cached());
    }

    #[test]
    fn test_custom_rule_reference_derived_reference() {
        // The source's caches retain the registration of a derived definition, which later requests for the same key
        // reuse instead of registering the provided definition, so the derived reference is cached while the source
        // registration is alive. Without the source's caches, the provided definition is not registered at all.
        let registration = CustomRuleRegistration::new(TestDefinition::new("source"));
        let derived = registration
            .reference()
            .derived_reference(jvp_derivation_key(1), Arc::new(TestDefinition::new("derived")));
        assert_eq!(derived.definition().name(), "derived");
        assert!(derived.is_cached());
        assert_eq!(registration.caches.derived_registrations.lock().unwrap().len(), 1);
        let reused = registration
            .reference()
            .derived_reference(jvp_derivation_key(1), Arc::new(TestDefinition::new("other")));
        assert_eq!(reused, derived);
        assert_eq!(reused.definition().name(), "derived");
        let source = registration.reference();
        drop(registration);
        assert!(!derived.is_cached());
        let uncached = source.derived_reference(jvp_derivation_key(1), Arc::new(TestDefinition::new("uncached")));
        assert_eq!(uncached.definition().name(), "uncached");
        assert!(!uncached.is_cached());
    }

    #[test]
    fn test_custom_rule_reference_derived_reference_capacity() {
        // The source registration retains as many derived registrations as specializations of each rule, evicting the
        // least recently used one, whose caches are freed with it, so that its key registers the provided definition.
        let registration = CustomRuleRegistration::new(TestDefinition::new("source"));
        let derived = (1..=CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1)
            .map(|input_count| {
                registration
                    .reference()
                    .derived_reference(jvp_derivation_key(input_count), Arc::new(TestDefinition::new("derived")))
            })
            .collect::<Vec<_>>();
        assert_eq!(
            registration.caches.derived_registrations.lock().unwrap().len(),
            CUSTOM_RULE_SPECIALIZATION_CAPACITY,
        );
        assert!(!derived[0].is_cached());
        assert!(derived[1..].iter().all(CustomRuleReference::is_cached));
        let replacement = Arc::new(TestDefinition::new("replacement"));
        let reregistered = registration.reference().derived_reference(jvp_derivation_key(1), replacement);
        assert_eq!(reregistered.definition().name(), "replacement");
        assert!(reregistered.is_cached());
    }

    #[test]
    fn test_custom_rule_reference_specialization() {
        // A miss produces the specialization and retains it, so that a hit returns the retained specialization without
        // producing it again. Each kind of request has its own cache.
        let registration = CustomRuleRegistration::new(TestDefinition::new("identity"));
        let reference = registration.reference();
        let scalar_type = ArrayType::scalar(DataType::F64);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<CustomRuleTracer<Array, ArrayOperation<Array>>>| Ok(inputs),
            vec![scalar_type.clone()],
        )
        .unwrap();
        let productions = AtomicUsize::new(0);
        let produce = || {
            productions.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(CustomRuleSpecialization::new(program.clone(), Vec::new())))
        };
        let key = unbatched_key(scalar_type);
        let first = reference.specialization(CustomRuleSpecializationRequest::Jvp(key.clone()), produce).unwrap();
        let second = reference.specialization(CustomRuleSpecializationRequest::Jvp(key.clone()), produce).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(productions.load(Ordering::SeqCst), 1);
        let forward = reference.specialization(CustomRuleSpecializationRequest::Forward(key), produce).unwrap();
        assert!(!Arc::ptr_eq(&first, &forward));
        assert_eq!(productions.load(Ordering::SeqCst), 2);
        assert_eq!(registration.caches.jvp_specializations.len(), 1);
        assert_eq!(registration.caches.forward_specializations.len(), 1);
    }

    #[test]
    fn test_custom_rule_reference_specialization_without_caches() {
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
    }

    #[test]
    fn test_custom_rule_reference_specialization_reentrant() {
        // A rule that requests its own specialization while it is being traced is rejected, with and without a live
        // handle, and a rejected request leaves nothing in flight.
        let definition = CustomRuleRegistration::new_cyclic(|this| {
            let this = this.clone();
            TestDefinition::new("reentrant").with_jvp(move |primals, tangents| {
                this.upgrade()?
                    .jvp_specialization(unbatched_key(primals[0].r#type().into_owned()))
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
    fn test_custom_rule_reference_specialization_eviction() {
        // The caches retain a bounded number of specializations of each rule, evicting the least recently used one.
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
            let key = unbatched_key(ArrayType::new_static(DataType::F64, [extent]));
            Arc::downgrade(&definition.reference().jvp_specialization(key).unwrap())
        };
        let retained_definition = Arc::downgrade(&definition.reference().definition);
        let baseline = retained_definition.strong_count();
        let mut probes = Vec::new();
        for extent in 1..=CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1 {
            jvp(extent);
            probes.push(specialization(extent));
        }
        assert_eq!(definition.caches.jvp_specializations.len(), CUSTOM_RULE_SPECIALIZATION_CAPACITY);
        assert_eq!(counters.counts(), (CUSTOM_RULE_SPECIALIZATION_CAPACITY + 1, 0, 0));

        // Bounded caches bound the retained memory. The evicted specialization is freed, and no cached program retains
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
    fn test_custom_rule_reference_specialization_outer_cache_hit() {
        // A batched specialization that is still cached is served without requesting the unbatched specialization from
        // which it was batched, so evicting the latter neither traces the rule again nor caches it again.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, false));
        let key = CustomRuleForwardSpecializationKey {
            levels: vec![CustomRuleBatchingLevel {
                level: BatchingLevel::new(BatchingLevelExtent::Static(3), None, ShardingDimension::Replicated),
                boundary_input_count: 0,
                input_axes: vec![BatchAxis::new(0)],
                output_axes: vec![BatchAxis::new(0)],
            }],
            ..unbatched_key(ArrayType::scalar(DataType::F64))
        };
        let first = definition.reference().jvp_specialization(key.clone()).unwrap();
        assert_eq!(definition.caches.jvp_specializations.len(), 2);
        assert_eq!(definition.caches.jvp_specializations.invalidate_entries_if(|key| key.levels.is_empty()), 1);
        let second = definition.reference().jvp_specialization(key).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(definition.caches.jvp_specializations.len(), 1);
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_custom_rule_reference_specialization_retry_after_failure() {
        // A failed specialization is not retained, so the next request invokes the rule again.
        let attempts = Arc::new(AtomicUsize::new(0));
        let definition = CustomRuleRegistration::new(TestDefinition::new("flaky").with_jvp({
            let attempts = attempts.clone();
            move |primals, tangents| {
                if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err(ProgramError::InvalidArgument { message: "transient failure".to_string() });
                }
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            }
        }));
        let x = Array::scalar(2f64).unwrap();
        let one = Array::scalar(1f64).unwrap();
        assert!(matches!(
            call_jvp(&definition, x.clone(), one.clone()),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transient failure",
        ));
        assert!(definition.caches.jvp_specializations.is_empty());
        for _ in 0..2 {
            assert_eq!(
                call_jvp(&definition, x.clone(), one.clone()),
                Ok((vec![Array::scalar(8f64).unwrap()], vec![Array::scalar(4f64).unwrap()])),
            );
        }
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_custom_rule_reference_specialization_concurrent_cold_requests() {
        // Both threads enter the rule before either finishes, so both derive the same cold specialization.
        // The duplicate derivations produce one resident result, which later requests reuse.
        let barrier = Arc::new(Barrier::new(2));
        let invocations = Arc::new(AtomicUsize::new(0));
        let definition = CustomRuleRegistration::new(TestDefinition::new("concurrent").with_jvp({
            let barrier = barrier.clone();
            let invocations = invocations.clone();
            move |primals, tangents| {
                if invocations.fetch_add(1, Ordering::SeqCst) < 2 {
                    barrier.wait();
                }
                let square = primals[0].clone() * primals[0].clone();
                Ok((vec![square.clone() * primals[0].clone()], vec![square * tangents[0].clone()]))
            }
        }));
        let x = Array::scalar(2f64).unwrap();
        let one = Array::scalar(1f64).unwrap();
        let expected = (vec![Array::scalar(8f64).unwrap()], vec![Array::scalar(4f64).unwrap()]);
        std::thread::scope(|scope| {
            let threads = (0..2)
                .map(|_| {
                    let (definition, x, one) = (definition.clone(), x.clone(), one.clone());
                    scope.spawn(move || call_jvp(&definition, x, one))
                })
                .collect::<Vec<_>>();
            for thread in threads {
                assert_eq!(thread.join().unwrap(), Ok(expected.clone()));
            }
        });
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
        assert_eq!(definition.caches.jvp_specializations.len(), 1);
        assert_eq!(call_jvp(&definition, x, one), Ok(expected));
        assert_eq!(invocations.load(Ordering::SeqCst), 2);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_custom_rule_reference_specialization_debug_recheck() {
        // A region-transform cache hit re-derives the linearization in debug builds and compares renderings. The
        // re-derivation reuses the retained specialization, so it neither invokes the rule nor changes the rendering.
        let counters = Arc::new(RuleCounters::default());
        let definition = CustomRuleRegistration::new(cube_definition(&counters, true, true));
        let program = custom_rule_program(&definition, ArrayType::scalar(DataType::F64));
        let first = program.linearize().unwrap();
        let second = program.linearize().unwrap();
        assert_eq!(second.tangent().to_string(), first.tangent().to_string());
        assert_eq!(counters.counts(), (1, 0, 0));
    }

    #[test]
    fn test_custom_rule_reference_clone() {
        // A clone refers to the same definition and caches.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let reference = registration.reference();
        let clone = reference.clone();
        assert!(Arc::ptr_eq(&clone.definition, &reference.definition));
        assert!(Arc::ptr_eq(&clone.caches.upgrade().unwrap(), &registration.caches));
    }

    #[test]
    fn test_custom_rule_reference_debug() {
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        assert_eq!(
            format!("{:?}", registration.reference()),
            "CustomRuleReference(CustomRuleDefinition { name: \"cube\", has_jvp: false, derives_jvp_from_primal: \
             false, has_vjp: false, .. })",
        );
    }

    #[test]
    fn test_custom_rule_reference_eq_and_hash() {
        // References compare and hash by the identity of the definition, so the references of equally named
        // definitions differ, whether or not their caches are alive.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let reference = registration.reference();
        let other = CustomRuleRegistration::new(TestDefinition::new("cube")).reference();
        assert_eq!(reference, reference);
        assert_eq!(reference, registration.reference());
        assert_ne!(reference, other);
        assert_eq!(hash_of(&reference), hash_of(&registration.reference()));
        let references = HashSet::from([reference.clone(), other.clone()]);
        drop(registration);
        assert!(references.contains(&reference));
        assert!(references.contains(&other));
        assert_eq!(references.len(), 2);
    }

    #[test]
    fn test_weak_custom_rule_reference_upgrade() {
        // A weak reference upgrades while the definition is alive (e.g., while a reference keeps it alive after the
        // registration is dropped), and fails once the definition was dropped.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let weak = registration.downgrade();
        assert_eq!(weak.upgrade().unwrap(), registration.reference());
        let reference = registration.reference();
        drop(registration);
        let upgraded = weak.upgrade().unwrap();
        assert_eq!(upgraded, reference);
        assert!(!upgraded.is_cached());
        drop((reference, upgraded));
        assert!(matches!(
            weak.upgrade(),
            Err(ProgramError::InvalidArgument { message }) if message == "the `composite` definition was dropped",
        ));
    }

    #[test]
    fn test_weak_custom_rule_reference_clone() {
        // A clone refers to the same definition and caches.
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        let weak = registration.downgrade();
        let clone = weak.clone();
        assert!(Weak::ptr_eq(&clone.definition, &weak.definition));
        assert!(Weak::ptr_eq(&clone.caches, &weak.caches));
        assert_eq!(clone.upgrade().unwrap(), registration.reference());
    }

    #[test]
    fn test_weak_custom_rule_reference_debug() {
        let registration = CustomRuleRegistration::new(TestDefinition::new("cube"));
        assert_eq!(format!("{:?}", registration.downgrade()), "WeakCustomRuleReference { .. }");
    }
}
