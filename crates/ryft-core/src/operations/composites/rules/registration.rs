//! The [`CustomRuleRegistration`] handle that owns the specialization caches of a [`CustomRuleDefinition`], together
//! with the [`CustomRuleReference`] and [`WeakCustomRuleRegistration`] handles through which calls and rules refer to a
//! registered definition without forming reference cycles.

use std::any::{Any, TypeId};
use std::cell::Cell;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, Weak};

use lru::LruCache;

use crate::differentiation::DifferentiationError;
use crate::operations::composites::operations::COMPOSITE_OPERATION_NAME;
use crate::operations::composites::rules::definition::CustomRuleDefinition;
use crate::operations::composites::rules::specialization::specialization_error;
use crate::operations::composites::rules::{
    CUSTOM_RULE_SPECIALIZATION_CAPACITY, CustomRuleBackwardSpecializationKey, CustomRuleBatchingRuleKey,
    CustomRuleDerivationKey, CustomRuleProgram, CustomRuleSpecialization, CustomRuleSpecializationKey,
    CustomRuleSpecializationRequest,
};
use crate::parameters::Parameter;
use crate::programs::{ProgramError, Typed};
use crate::specialization::SpecializationCache;

/// Bounded specialization caches of one registered [`CustomRuleDefinition`], keyed by static information only and
/// owned by its [`CustomRuleRegistration`] handle (refer to the Ownership section of [`CustomRuleDefinition`]).
pub(in crate::operations::composites) struct CustomRuleCaches<V: Typed + Parameter, O> {
    /// Specializations of the forward mode rule.
    pub(in crate::operations::composites) jvp_specializations:
        SpecializationCache<CustomRuleSpecializationKey<V::Type>, CustomRuleProgram<V, O>>,

    /// Specializations of the reverse mode forward rule.
    pub(in crate::operations::composites) forward_specializations:
        SpecializationCache<CustomRuleSpecializationKey<V::Type>, CustomRuleProgram<V, O>>,

    /// Destination-specialized transposes of the reverse mode backward rule.
    pub(in crate::operations::composites) backward_specializations:
        SpecializationCache<CustomRuleBackwardSpecializationKey<V::Type>, CustomRuleProgram<V, O>>,

    /// Batched primal programs produced by the custom batching rule.
    pub(super) batching_rule_specializations:
        SpecializationCache<CustomRuleBatchingRuleKey<V::Type>, CustomRuleProgram<V, O>>,

    /// Registrations of the definitions derived from this definition (refer to [`CustomRuleDerivation`]), which these
    /// caches own so that they live at most as long as the specializations of this definition. Like the
    /// specializations, they are bounded, and an evicted registration frees the specializations of its definition.
    pub(super) derived_registrations: Mutex<LruCache<CustomRuleDerivationKey, CustomRuleRegistration<V, O>>>,

    /// Specializations of this definition converted into the families that contain its family (refer to
    /// [`LiftedCustomRules`]), keyed by the [`TypeId`] of the containing family and by the request of the
    /// specialization from which each one was converted. Each one is a [`CustomRuleProgram`] of that family.
    converted_specializations:
        SpecializationCache<(TypeId, CustomRuleSpecializationRequest<V::Type>), Arc<dyn Any + Send + Sync>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleCaches<V, O> {
    /// Creates empty [`CustomRuleCaches`] that retain up to [`CUSTOM_RULE_SPECIALIZATION_CAPACITY`] specializations of
    /// each rule and as many derived registrations.
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
/// [`WeakCustomRuleRegistration`] passed to [`Self::new_cyclic`]), never a handle, because a rule that owned the caches
/// of a definition whose programs reach the rule would form a reference cycle. Handles are deliberately not [`Sync`],
/// so the compiler rejects a rule, which must be [`Send`] and [`Sync`], that captures one. References to other
/// definitions cannot form a cycle either, because a definition can only capture definitions that were registered
/// before it, except for itself through [`Self::new_cyclic`]. For example, two mutually recursive definitions are
/// registered by building the second one inside the [`Self::new_cyclic`] closure of the first one, capturing the weak
/// handle of the first one in the rules of the second one.
pub struct CustomRuleRegistration<V: Typed + Parameter, O> {
    /// Registered definition.
    definition: Arc<CustomRuleDefinition<V, O>>,

    /// Specialization caches of the definition, which only handles hold strongly.
    caches: Arc<CustomRuleCaches<V, O>>,

    /// Marker that keeps this handle from being [`Sync`] (refer to the documentation of [`CustomRuleRegistration`]).
    marker: PhantomData<Cell<()>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleRegistration<V, O> {
    /// Registers the provided definition with empty specialization caches.
    pub fn new(definition: CustomRuleDefinition<V, O>) -> Self {
        Self::from_definition(Arc::new(definition))
    }

    /// Registers the provided shared definition with empty specialization caches, which is how a derived definition
    /// that outlived its previous registration is registered again without changing its identity.
    pub(super) fn from_definition(definition: Arc<CustomRuleDefinition<V, O>>) -> Self {
        Self { definition, caches: Arc::new(CustomRuleCaches::new()), marker: PhantomData }
    }

    /// Registers the definition returned by `build_fn`, which receives a weak handle to that same registration so that
    /// the definition's rules can call it (i.e., recursively) without keeping it alive.
    pub fn new_cyclic<F: FnOnce(&WeakCustomRuleRegistration<V, O>) -> CustomRuleDefinition<V, O>>(build_fn: F) -> Self {
        let caches = Arc::new(CustomRuleCaches::new());
        let definition = Arc::new_cyclic(|definition| {
            build_fn(&WeakCustomRuleRegistration { definition: definition.clone(), caches: Arc::downgrade(&caches) })
        });
        Self { definition, caches, marker: PhantomData }
    }

    /// Returns a [`CustomRuleReference`] to this registration, with which calls are staged and rules refer to it.
    #[inline]
    pub fn reference(&self) -> CustomRuleReference<V, O> {
        CustomRuleReference { definition: self.definition.clone(), caches: Arc::downgrade(&self.caches) }
    }
}

#[cfg(test)]
impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleRegistration<V, O> {
    /// Returns the registered definition, which tests use to observe its ownership.
    pub(in crate::operations::composites) fn definition(&self) -> &Arc<CustomRuleDefinition<V, O>> {
        &self.definition
    }

    /// Returns the specialization caches, which tests use to observe lazy tracing and cache ownership.
    pub(in crate::operations::composites) fn caches(&self) -> &Arc<CustomRuleCaches<V, O>> {
        &self.caches
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for CustomRuleRegistration<V, O> {
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone(), marker: PhantomData }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for CustomRuleRegistration<V, O> {
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
    pub(super) caches: Weak<CustomRuleCaches<V, O>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> CustomRuleReference<V, O> {
    /// Returns the referenced definition.
    #[inline]
    pub fn definition(&self) -> &CustomRuleDefinition<V, O> {
        &self.definition
    }

    /// Returns the specialization for `request`, produced by `produce_fn` on a miss. The specialization is retained in
    /// the registration's caches while they are alive, and produced without caching otherwise.
    pub(super) fn specialize<F: FnOnce() -> Result<CustomRuleProgram<V, O>, DifferentiationError>>(
        &self,
        request: CustomRuleSpecializationRequest<V::Type>,
        produce_fn: F,
    ) -> Result<CustomRuleProgram<V, O>, DifferentiationError> {
        let Some(caches) = self.caches.upgrade() else {
            return self.definition.produce_uncached(request, produce_fn);
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
        .map_err(|error| specialization_error(self.definition.name(), error))
    }

    /// Returns the specialization for `request` converted into the family `(Vc, Oc)`, which contains this reference's
    /// family, produced by `produce_fn` on a miss. Like the specializations themselves, it is retained in the
    /// registration's caches while they are alive, and produced without caching otherwise.
    pub(super) fn converted_specialization<Vc, Oc, F>(
        &self,
        request: CustomRuleSpecializationRequest<V::Type>,
        produce_fn: F,
    ) -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>
    where
        Vc: 'static + Typed + Parameter,
        Oc: 'static,
        CustomRuleSpecialization<Vc, Oc>: Send + Sync,
        F: FnOnce() -> Result<CustomRuleProgram<Vc, Oc>, DifferentiationError>,
    {
        let Some(caches) = self.caches.upgrade() else {
            return produce_fn();
        };
        let specialization = caches
            .converted_specializations
            .get_or_try_insert_with((TypeId::of::<(Vc, Oc)>(), request), || {
                Ok::<_, DifferentiationError>(produce_fn()? as Arc<dyn Any + Send + Sync>)
            })
            .map_err(|error| specialization_error(self.definition.name(), error))?;

        // The key names the family, so the retained specialization always belongs to it.
        Ok(specialization.downcast::<CustomRuleSpecialization<Vc, Oc>>().unwrap())
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
        self.definition.id == other.definition.id
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Eq for CustomRuleReference<V, O> {}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Hash for CustomRuleReference<V, O> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.definition.id.hash(state);
    }
}

/// Weak handle to a registered [`CustomRuleDefinition`], which keeps neither the definition nor its caches alive. It is
/// how a definition's rules refer to that same definition (refer to [`CustomRuleRegistration::new_cyclic`]).
pub struct WeakCustomRuleRegistration<V: Typed + Parameter, O> {
    /// Registered definition.
    definition: Weak<CustomRuleDefinition<V, O>>,

    /// Specialization caches of the definition.
    caches: Weak<CustomRuleCaches<V, O>>,
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> WeakCustomRuleRegistration<V, O> {
    /// Returns a [`CustomRuleReference`] to the registration, with which the rules of that same definition stage its
    /// calls. A rule is only invoked while its definition is alive, so this fails only when the reference is used
    /// outside a rule after the definition was dropped.
    pub fn reference(&self) -> Result<CustomRuleReference<V, O>, ProgramError> {
        let definition = self.definition.upgrade().ok_or_else(|| ProgramError::InvalidArgument {
            message: format!("the `{COMPOSITE_OPERATION_NAME}` definition was dropped"),
        })?;
        Ok(CustomRuleReference { definition, caches: self.caches.clone() })
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Clone for WeakCustomRuleRegistration<V, O> {
    fn clone(&self) -> Self {
        Self { definition: self.definition.clone(), caches: self.caches.clone() }
    }
}

impl<V: Typed<Type: Eq + Hash> + Parameter, O> Debug for WeakCustomRuleRegistration<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("WeakCustomRuleRegistration").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayType, DataType};
    use crate::contexts::Context;
    use crate::operations::composites::operations::CompositeOperation;
    use crate::operations::composites::rules::specialization::CustomRuleSpecializer;
    use crate::operations::composites::tests::{
        RuleCounters, TestDefinition, cube_definition, cube_program, custom_rule_program,
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
        assert!(Arc::ptr_eq(clone.definition(), registration.definition()));
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
            (Arc::downgrade(&inner.definition()), Arc::downgrade(&inner.caches()));
        let inner_reference = inner.reference();
        let outer = CustomRuleRegistration::new(TestDefinition::new("outer").with_jvp(move |primals, tangents| {
            let r#type = primals[0].r#type().into_owned();
            let operation = CompositeOperation::new(inner_reference.clone());
            let outputs = primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
            Ok((outputs, vec![tangents[0].clone()]))
        }));
        let (retained_outer, retained_outer_caches) =
            (Arc::downgrade(&outer.definition()), Arc::downgrade(&outer.caches()));
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
                let key = CustomRuleSpecializationKey {
                    input_types: vec![r#type.clone()],
                    output_types: vec![r#type],
                    kept_outputs: None,
                    non_differentiated_count: 0,
                    tangent_activity: vec![true],
                    levels: Vec::new(),
                    discharged: false,
                };
                this.reference()?
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
        let retained = Arc::downgrade(&definition.definition());
        drop(definition);
        for _ in 0..2 {
            assert!(matches!(
                program.jvp(),
                Err(DifferentiationError::Program(ProgramError::InvalidArgument { message })) if message == reentrant,
            ));
            assert!(!retained.upgrade().unwrap().has_uncached_in_flight());
        }
    }

    #[test]
    fn test_custom_rule_registration_new_cyclic() {
        // The build function receives a weak handle to the registration that it builds, which refers to that same
        // registration and its caches once it is built, but keeps neither of them alive.
        let mut weak = None;
        let registration = CustomRuleRegistration::new_cyclic(|this| {
            weak = Some(this.clone());
            TestDefinition::new("cyclic")
        });
        let weak = weak.unwrap();
        let reference = weak.reference().unwrap();
        assert_eq!(reference, registration.reference());
        assert!(Arc::ptr_eq(&reference.caches.upgrade().unwrap(), registration.caches()));
        drop(reference);
        let (definition, caches) = (Arc::downgrade(registration.definition()), Arc::downgrade(registration.caches()));
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
                let operation = CompositeOperation::new(this.reference()?);
                let outputs =
                    primals[0].context().bind(operation, vec![cube_program(&r#type)], &[primals[0].clone()])?;
                Ok((outputs, vec![primals[0].clone() * primals[0].clone() * tangents[0].clone()]))
            })
        });
        let (retained_definition, retained_caches) =
            (Arc::downgrade(&definition.definition()), Arc::downgrade(&definition.caches()));
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
                    let operation = CompositeOperation::new(a.reference()?);
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
            (Arc::downgrade(&a.definition()), Arc::downgrade(&a.caches())),
            (Arc::downgrade(&b.definition()), Arc::downgrade(&b.caches())),
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
    fn test_weak_custom_rule_registration_reference() {
        // A weak handle yields a reference while the definition is alive (e.g., while a reference keeps it alive after
        // the registration is dropped), and fails once the definition was dropped.
        let mut weak = None;
        let registration = CustomRuleRegistration::new_cyclic(|this| {
            weak = Some(this.clone());
            TestDefinition::new("cyclic")
        });
        let weak = weak.unwrap();
        assert_eq!(format!("{weak:?}"), "WeakCustomRuleRegistration { .. }");
        assert_eq!(weak.reference().unwrap(), registration.reference());
        let reference = registration.reference();
        drop(registration);
        let upgraded = weak.reference().unwrap();
        assert_eq!(upgraded, reference);
        assert!(upgraded.caches.upgrade().is_none());
        drop((reference, upgraded));
        assert!(matches!(
            weak.reference(),
            Err(ProgramError::InvalidArgument { message }) if message == "the `composite` definition was dropped",
        ));
    }
}
