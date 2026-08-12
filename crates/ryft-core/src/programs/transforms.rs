//! Extensible caching for structural transforms of sealed [`Region`]s.
//!
//! A callee [`Program`] interned into many programs is linearized, transposed, or otherwise structurally transformed
//! once rather than once per program that interned it, because the derived result is retained on the sealed region
//! itself and every program sharing that region shares the derivation. This is shown in the following example:
//!
//! ```
//! # use std::convert::Infallible;
//! # use std::sync::Arc;
//! # use ryft_core::{
//! #     Array, ArrayOperation, ArrayType, DataType, Operation, Placeholder, ProgramBuilder, Region, Transform,
//! #     TransformArtifact, Value,
//! # };
//!
//! // A zero-sized marker names one retained derivation family and owns its own per-region namespace.
//! struct IdentityCopy;
//!
//! impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for IdentityCopy {
//!     type Arguments = usize;
//!     type Artifact = TransformArtifact<V, O, usize>;
//!
//!     const DEFAULT_CACHE_CAPACITY: usize = 2;
//! }
//!
//! # let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
//! # let input = builder.add_input(ArrayType::scalar(DataType::F64));
//! # let program = builder
//! #     .build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder])
//! #     .unwrap();
//!
//! let region = program.entry_region_ref();
//! let artifact = region
//!     .transform::<IdentityCopy, _, Infallible>(0, |region, arguments| {
//!         Ok(TransformArtifact::new(vec![Arc::new(region.to_program())], *arguments))
//!     })
//!     .unwrap();
//! assert_eq!(artifact.metadata(), &0);
//!
//! // The second request for the same marker and arguments serves the retained artifact instead of re-deriving it.
//! let repeated = region
//!     .transform::<IdentityCopy, _, Infallible>(0, |region, arguments| {
//!         Ok(TransformArtifact::new(vec![Arc::new(region.to_program())], *arguments))
//!     })
//!     .unwrap();
//! assert!(Arc::ptr_eq(&artifact.programs()[0], &repeated.programs()[0]));
//! ```
//!
//! [`Transform`] describes one retained specialization family within a type-level source universe. It selects
//! the complete [`Transform::Arguments`] key, the cheaply cloned [`Transform::Artifact`], and a default capacity.
//! [`TransformCache`] is a zero-cost alias for the corresponding [`SpecializationCache`]. Production remains with
//! the owner.
//!
//! Readers new to regions, arenas, and sealing should start with the documentation of the
//! [`regions`](crate::programs::regions) module, because every rule stated here is phrased in terms of a
//! sealed region's complete reachable contents. Two terms recur throughout. A _marker_ is the type that implements
//! [`Transform`] and names one derivation family. A _namespace_ is the per-region bounded cache owned by one marker
//! type, where the marker's [`TypeId`] identifies the namespace and the transform's arguments identify entries
//! within it.
//!
//! # Using Region Transforms
//!
//! [`RegionRef::transform`] is the safe public extension point for region-owned structural transforms. The transform
//! marker namespaces an independently bounded cache attached to the sealed region, and its arguments select one entry
//! in that namespace. The returned [`TransformArtifact`] contains concrete programs over the region's value and
//! operation universe plus static metadata. Programs may therefore retain non-`'static` backend lifetimes even though
//! the erased arguments and metadata must be `'static`.
//!
//! ## Region Reuse Contract
//!
//! A region derivation must be a deterministic structural function of the region's complete reachable contents and its
//! arguments. Arguments must contain every semantic input not represented by that graph. Metadata must not retain the
//! source region, program, transform-cache state, live context, runtime values, or invocation-specific residuals. The
//! derivation callback is the uncached kernel and must have stable semantics for every use of one marker type.
//!
//! A transform whose result depends on live [`Context`](crate::Context) state has no complete key here and must not be
//! cached against a region alone. _Partial evaluation_ is the canonical exclusion: its partition carries known outputs
//! that are values of the live parent context (i.e., concrete constants under an eager parent and tracers staged into
//! the parent's trace under a staging one), so one region together with one known-ness mask does not determine one
//! artifact. That is what keeps the `condition`, `scan`, and `while` known-ness splits, for example, out of every
//! region namespace.
//!
//! The cache is shared by content-preserving copies of a sealed region and replaced whenever construction or rewriting
//! changes the complete reachable contents. External transforms never adopt or invalidate cache provenance directly.
//! On same-thread recursive production of one marker and argument value, the callback runs without publishing so
//! recursive structural transformation never waits or deadlocks. Errors and panics retain nothing and retry later.
//!
//! # Design And Soundness
//!
//! Retained Just-In-Time (JIT) compilation dispatch and region transformation share this cache vocabulary without
//! pretending to share a source, artifact representation, validation policy, or reentrancy behavior. [`Transform`]'s
//! own documentation carries the diagram relating the two owners to the descriptor, the typed alias, and their
//! artifacts.
//!
//! ## Cache Provenance
//!
//! The [`regions`](crate::programs::regions) module owns provenance, and two rules keep a cache from outliving the
//! complete reachable contents it was derived from:
//!
//!   - every construction of a region with rewritten contents mints a fresh cache (i.e., type-identity renaming,
//!     operation mapping, value un-projection, boundary rebuilds, and program simplification all reach it), and every
//!     in-place rewrite of a region's contents detaches the cache derived from its previous contents; and
//!   - sealing a region into an arena mints a fresh cache whenever that region attaches at least one descendant,
//!     because an attached [`RegionId`](crate::RegionId) means nothing until an arena files a body under it. A region
//!     carried into a different arena would otherwise keep transforms derived from whatever descendants its previous
//!     arena happened to hold at those identifiers, which is a wrong derived program rather than a missed reuse
//!     opportunity. A region that attaches nothing has no such dependency and keeps its cache, which is what preserves
//!     the common leaf-callee sharing.
//!
//! The second rule is deliberately conservative, so the internal paths that re-seal a region while provably preserving
//! its complete reachable closure opt out of it. Those are the closure-copying imports (i.e.,
//! [`ProgramBuilder::import_region`](crate::ProgramBuilder::import_region) and the callee interning built on it), the
//! faithful whole-arena rebuilds in
//! [`ClosedProgram::without_unused_captures`](crate::ClosedProgram::without_unused_captures) and
//! [`ClosedProgram::to_program_with_lifted_captures`](crate::ClosedProgram::to_program_with_lifted_captures), the
//! entry-boundary projections [`Program::filtered`] and [`Program::into_filtered`], which carry the descendant closure
//! over verbatim, and program simplification, which additionally re-adopts each source cache in the one case where the
//! rebuild is provably the identity on the region's contents. [`RegionRef::to_program`] re-adopts the promoted entry's
//! cache the same way. Renumbering attached identifiers is always tolerated as long as the renumbering preserves the
//! complete reachable graph's topology, which is why importing and compaction keep retained transforms valid.
//!
//! ## Namespace Storage And Thread Safety
//!
//! Each region allocates only one small shared cache handle. The marker registry itself is created lazily on the first
//! transform request. Every marker receives its own bounded namespace, so keys and eviction in one transform cannot
//! affect another. Registry and namespace locks protect only bookkeeping (i.e., derivation, rechecking, formatting,
//! and caller-owned destruction run after those locks are released).
//!
//! Thread safety remains structural. A typed [`TransformCache`] is `Send` and `Sync` exactly when its arguments and
//! artifacts are. A region registry preserves the same rule for its concrete program universe, while requiring erased
//! arguments and metadata to be `'static + Send + Sync`. Different threads may derive one cold specialization
//! concurrently under [`SpecializationCache`]'s deliberate last-writer-wins policy.
//!
//! ## Ownership-Cycle Prevention
//!
//! Content-preserving program materialization can copy the source region's cache handle. Before a derived artifact is
//! published, [`RegionRef::transform`] detaches every returned region whose cache is pointer-identical to the source
//! root. This prevents a region-owned artifact from retaining its own cache state. Unrelated descendant cache handles
//! remain shared. Opaque metadata cannot be inspected, so it must never retain the source region, source program, or
//! cache state.
//!
//! Detaching only the self-identical caches is sufficient under the region reuse contract stated above, and only under
//! it. A contract-abiding derivation is a structural function of the source region's complete reachable contents, so
//! its programs can reach nothing but strict descendants of the source region, copies of the source region itself
//! (which the preceding paragraph's sanitization detaches), and freshly built regions (which carry fresh caches). "Is
//! a strict descendant of" is acyclic over a sealed arena, so a descendant's cache can never already hold an artifact
//! whose programs carry the source root, which rules out the remaining shape of the hazard: a two-region cycle in
//! which one region's retained artifact holds the other region's cache and that artifact in turn holds the first
//! region's cache.
//!
//! That acyclicity argument is contingent on the contract rather than enforced by the type system. A contract-violating
//! derivation that embeds programs built from regions unrelated to its source can construct exactly that cycle. Two
//! markers whose derivations each embed the other source region's [`RegionRef::to_program`] result, for example, build
//! a strong [`Arc`] cycle that sanitization cannot observe, because sanitization detaches only caches that are
//! pointer-identical to the source root. The consequence of such a violation is unreclaimable retained state, that is,
//! a leak, never a wrong derived program.
//!
//! ## Debug-Assertion Recheck
//!
//! With debug assertions enabled, a hit is re-derived and compared with the retained program rendering and metadata.
//! This detects only semantics represented by [`Value`] rendering, [`Operation::render`], and metadata equality.
//! Because a fresh derivation error is propagated, a hit that succeeds in an optimized build can fail in a
//! debug-assertion build. The complete re-derivation and comparison path is compiled out when debug assertions are
//! disabled.

use std::any::{Any, TypeId, type_name};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use dyn_eq::DynEq;
use dyn_hash::DynHash;

use crate::parameters::Parameter;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::{Region, RegionRef};
use crate::programs::types::Typed;
use crate::programs::values::Value;
use crate::specialization::{ReentrantSpecializationError, SpecializationCache, SpecializationCacheEntry};

/// Describes a family of retained specializations within `Source`. `Source` is a type-level coherence and
/// documentation parameter. It is not stored or invoked by a cache. For example, structural region markers implement
/// `Transform<Region<V, O>>`, while retained Just-In-Time (JIT) compilation specialization uses a private source token
/// describing its domain and function signature. Also, equal [`Arguments`](Self::Arguments) must make retained
/// artifacts interchangeable within one owning cache. Any fixed semantics absent from the arguments, such as a retained
/// closure or immutable options, must be fixed by that cache's owner.
///
/// # Type Relationships
///
/// ```mermaid
/// %%{init: {"themeCSS": ".nodeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart TB
///   descriptor["&lt;code&gt;Transform&lt;/code&gt; Descriptor"]
///   arguments["&lt;code&gt;Arguments&lt;/code&gt;"]
///   artifact["&lt;code&gt;Artifact&lt;/code&gt;"]
///   capacity["&lt;code&gt;DEFAULT_CACHE_CAPACITY&lt;/code&gt;"]
///   typed_cache["&lt;code&gt;TransformCache&lt;/code&gt; Alias"]
///   specialization["&lt;code&gt;SpecializationCache&lt;/code&gt;"]
///   region_owner["&lt;code&gt;RegionRef&lt;/code&gt;"]
///   registry["Per-Region Namespace Registry"]
///   region_artifact["&lt;code&gt;Program&lt;/code&gt; Bundle and Metadata"]
///   dispatcher["&lt;code&gt;CompiledFunctionDispatcher&lt;/code&gt;"]
///   executable["&lt;code&gt;ExecutableFunction&lt;/code&gt;"]
///   backend["&lt;code&gt;CompilationContext&lt;/code&gt; on a Miss"]
///   descriptor --> arguments
///   descriptor --> artifact
///   descriptor --> capacity
///   typed_cache --> specialization
///   region_owner --> registry
///   registry --> specialization
///   registry --> region_artifact
///   dispatcher --> typed_cache
///   typed_cache --> executable
///   dispatcher --> backend
/// ```
#[cfg_attr(doc, aquamarine::aquamarine)]
pub trait Transform<Source> {
    /// Complete equality and hash identity of one specialization within the owning transform cache.
    type Arguments: Clone + Eq + Hash;

    /// Cheaply cloneable artifact retained for one [`Self::Arguments`] instance.
    type Artifact: Clone;

    /// Default number of [`Self::Artifact`]s retained by one transform owner or region namespace.
    const DEFAULT_CACHE_CAPACITY: usize;
}

/// Typed [`SpecializationCache`] selected by [`Transform`] `T` for `Source`. This is an alias rather than a forwarding
/// wrapper, so that it exposes the exact [`SpecializationCache`] entry, statistics, invalidation, synchronization,
/// panic, and destructor behavior. Construct it with `SpecializationCache::new(T::DEFAULT_CACHE_CAPACITY)` when the
/// descriptor default applies, or with an explicit owner-configured capacity.
///
/// ```
/// # use std::convert::Infallible;
/// # use ryft_core::programs::transforms::{Transform, TransformCache};
///
/// struct Source;
/// struct Square;
///
/// impl Transform<Source> for Square {
///     type Arguments = u32;
///     type Artifact = u32;
///
///     const DEFAULT_CACHE_CAPACITY: usize = 4;
/// }
///
/// let cache = TransformCache::<Square, Source>::new(<Square as Transform<Source>>::DEFAULT_CACHE_CAPACITY);
/// assert_eq!(cache.get_or_try_insert_with(3, || Ok::<_, Infallible>(9)).unwrap(), 9);
/// assert_eq!(cache.get_or_try_insert_with(3, || Ok::<_, Infallible>(99)).unwrap(), 9);
/// ```
pub type TransformCache<T, Source> =
    SpecializationCache<<T as Transform<Source>>::Arguments, <T as Transform<Source>>::Artifact>;

// TODO(eaplatanios): Review from here onwards.

/// Structural programs and static metadata retained for one region-transform specialization.
///
/// Program order is transform-defined semantic data. Prefer a transform-specific wrapper when one exists rather than
/// depending directly on a built-in marker's raw layout. `Metadata` must be small, owned, and independent of runtime
/// invocation state; it must not retain the source region, source program, cache state, live context, backend buffer,
/// or invocation-specific residual value.
pub struct TransformArtifact<V: Typed + Parameter, O, Metadata> {
    /// Transformed programs in transform-defined semantic order.
    programs: Vec<Arc<Program<V, O, Vec<V>, Vec<V>>>>,

    /// Small static metadata interpreting `programs`.
    metadata: Metadata,
}

impl<V: Typed + Parameter, O, Metadata> TransformArtifact<V, O, Metadata> {
    /// Creates a transform artifact from programs in transform-defined semantic order and their metadata.
    ///
    /// # Parameters
    ///   - `programs`: Transformed programs in the order documented by the transform marker.
    ///   - `metadata`: Small static metadata needed to interpret the programs.
    #[inline]
    pub fn new(programs: Vec<Arc<Program<V, O, Vec<V>, Vec<V>>>>, metadata: Metadata) -> Self {
        Self { programs, metadata }
    }

    /// Returns the transformed programs in transform-defined semantic order.
    #[inline]
    pub fn programs(&self) -> &[Arc<Program<V, O, Vec<V>, Vec<V>>>] {
        &self.programs
    }

    /// Returns the transform-specific metadata.
    #[inline]
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Consumes this artifact and returns its transformed programs and metadata.
    #[inline]
    pub fn into_parts(self) -> (Vec<Arc<Program<V, O, Vec<V>, Vec<V>>>>, Metadata) {
        (self.programs, self.metadata)
    }
}

impl<V: Typed + Parameter, O, Metadata: Clone> Clone for TransformArtifact<V, O, Metadata> {
    #[inline]
    fn clone(&self) -> Self {
        Self { programs: self.programs.clone(), metadata: self.metadata.clone() }
    }
}

impl<V: Typed + Parameter, O, Metadata: Debug> Debug for TransformArtifact<V, O, Metadata> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TransformArtifact")
            .field("program_count", &self.programs.len())
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Object-safe equality/hash/debug contract for erased region-transform arguments.
trait ErasedTransformArgumentsValue: Debug + Send + Sync + DynEq + DynHash {}

impl<T: Debug + Eq + Hash + Send + Sync + 'static> ErasedTransformArgumentsValue for T {}

dyn_eq::eq_trait_object!(ErasedTransformArgumentsValue);
dyn_hash::hash_trait_object!(ErasedTransformArgumentsValue);

/// Cloneable erased key used inside one transform marker's namespace.
#[derive(Clone)]
pub(crate) struct ErasedTransformArguments(Arc<dyn ErasedTransformArgumentsValue>);

impl ErasedTransformArguments {
    /// Erases one complete typed argument value.
    #[inline]
    fn new(arguments: impl ErasedTransformArgumentsValue + 'static) -> Self {
        Self(Arc::new(arguments))
    }
}

impl Debug for ErasedTransformArguments {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

impl PartialEq for ErasedTransformArguments {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl Eq for ErasedTransformArguments {}

impl Hash for ErasedTransformArguments {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Homogeneous artifact stored by every namespace in one concrete region universe.
pub(crate) struct ErasedTransformArtifact<V: Typed + Parameter, O> {
    /// Concrete transformed programs, preserving any non-static `V`/`O` lifetimes.
    programs: Vec<Arc<Program<V, O, Vec<V>, Vec<V>>>>,

    /// Type-erased static metadata.
    metadata: Arc<dyn Any + Send + Sync>,

    /// Concrete metadata type name used only by the non-semantic debug summary and invariant diagnostics.
    metadata_type_name: &'static str,
}

impl<V: Typed + Parameter, O> ErasedTransformArtifact<V, O> {
    /// Erases one sanitized public transform artifact.
    fn new<Metadata: Send + Sync + 'static>(artifact: TransformArtifact<V, O, Metadata>) -> Self {
        let (programs, metadata) = artifact.into_parts();
        Self { programs, metadata: Arc::new(metadata), metadata_type_name: type_name::<Metadata>() }
    }

    /// Reconstructs a typed public artifact, panicking only if one marker violated its coherent artifact layout.
    fn typed<Metadata: Clone + Send + Sync + 'static>(&self) -> TransformArtifact<V, O, Metadata> {
        let metadata = self.metadata.downcast_ref::<Metadata>().unwrap_or_else(|| {
            panic!(
                "region transform cache metadata type mismatch: retained `{}` but requested `{}`",
                self.metadata_type_name,
                type_name::<Metadata>(),
            )
        });
        TransformArtifact::new(self.programs.clone(), metadata.clone())
    }
}

impl<V: Typed + Parameter, O> Clone for ErasedTransformArtifact<V, O> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            programs: self.programs.clone(),
            metadata: self.metadata.clone(),
            metadata_type_name: self.metadata_type_name,
        }
    }
}

impl<V: Typed + Parameter, O> Debug for ErasedTransformArtifact<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ErasedTransformArtifact")
            .field("program_count", &self.programs.len())
            .field("metadata_type", &self.metadata_type_name)
            .finish()
    }
}

/// Per-marker region namespace cache after argument and metadata erasure.
pub(crate) type RegionTransformNamespace<V, O> =
    SpecializationCache<ErasedTransformArguments, ErasedTransformArtifact<V, O>>;

/// Lazily allocated shared state of one sealed region's transform cache.
pub(crate) struct RegionTransformCacheState<V: Typed + Parameter, O> {
    /// Marker namespaces. The map and mutex are allocated only when the first transform is requested.
    pub(crate) registry: OnceLock<Mutex<HashMap<TypeId, Arc<RegionTransformNamespace<V, O>>>>>,
}

/// Region-owned registry of independently bounded structural transform namespaces.
///
/// Copies of one content-identical sealed region share this handle. Construction and rewriting code in
/// [`super::regions`] mints or preserves it according to complete reachable-content identity; external transforms can
/// only request typed artifacts through [`RegionRef::transform`].
pub(crate) struct RegionTransformCache<V: Typed + Parameter, O> {
    /// Shared state carried by content-preserving region copies.
    pub(crate) state: Arc<RegionTransformCacheState<V, O>>,
}

impl<V: Typed + Parameter, O> RegionTransformCache<V, O> {
    /// Creates an empty cache without allocating its namespace registry.
    pub(crate) fn new() -> Self {
        Self { state: Arc::new(RegionTransformCacheState { registry: OnceLock::new() }) }
    }

    /// Returns the namespace for `T`, creating it with `capacity` when first requested.
    fn namespace<T: 'static>(&self, capacity: usize) -> Arc<RegionTransformNamespace<V, O>> {
        let registry = self.state.registry.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(namespace) = registry
            .lock()
            .expect("region transform registry mutex is poisoned")
            .get(&TypeId::of::<T>())
            .cloned()
        {
            return namespace;
        }

        // A losing candidate is always a freshly built empty cache, so dropping it can never run caller-defined
        // metadata destructors. The explicit drop placed after the guard is released is still worth keeping, for two
        // reasons: it holds the uniform discipline that no cache structure is ever dropped while the registry lock is
        // held, and it keeps candidate construction and destruction visibly outside the critical section so that a
        // later `entry().or_insert_with(...)` rewrite, which would construct and drop under the lock, is not mistaken
        // for an equivalent simplification.
        let mut candidate = Some(Arc::new(SpecializationCache::new(capacity)));
        let namespace = {
            let mut registry = registry.lock().expect("region transform registry mutex is poisoned");
            if let Some(namespace) = registry.get(&TypeId::of::<T>()) {
                Arc::clone(namespace)
            } else {
                let namespace = candidate.take().unwrap();
                registry.insert(TypeId::of::<T>(), Arc::clone(&namespace));
                namespace
            }
        };
        drop(candidate);
        namespace
    }

    /// Returns whether this cache and `other` share the same region-content identity.
    #[inline]
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
}

impl<V: Typed + Parameter, O> Clone for RegionTransformCache<V, O> {
    #[inline]
    fn clone(&self) -> Self {
        Self { state: self.state.clone() }
    }
}

impl<V: Typed + Parameter, O> Debug for RegionTransformCache<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (namespace_count, artifact_count) = self.state.registry.get().map_or((0, 0), |registry| {
            let registry = registry.lock().expect("region transform registry mutex is poisoned");
            (registry.len(), registry.values().map(|namespace| namespace.len()).sum())
        });
        formatter
            .debug_struct("RegionTransformCache")
            .field("namespace_count", &namespace_count)
            .field("artifact_count", &artifact_count)
            .finish()
    }
}

impl<'r, V: Value, O: Operation<Type = V::Type>> RegionRef<'r, V, O> {
    /// Returns the artifact retained for one structural transform, deriving and retaining it on a miss.
    ///
    /// The marker `T` owns an independent bounded namespace on this sealed region. `arguments` must contain every
    /// semantic derivation input not represented by the complete reachable region graph. `derive` is the uncached
    /// deterministic kernel and must have stable semantics for every use of `T`. It must not recursively request the
    /// same marker and arguments itself; if recursion occurs indirectly, this method derives that nested request
    /// uncached and publishes only the outer artifact.
    ///
    /// Failed and panicking derivations retain nothing. A successful artifact is sanitized before the exact same
    /// instance is published and returned. With debug assertions enabled, a hit re-runs `derive`, sanitizes the fresh
    /// artifact, and compares it with the retained programs and metadata; optimized builds compile that work out.
    ///
    /// ```mermaid
    /// flowchart LR
    ///   request["RegionRef and Arguments"] --> namespace["Marker Namespace"]
    ///   namespace -->|"hit"| retained["Retained TransformArtifact"]
    ///   namespace -->|"miss"| derive["Uncached Derivation"]
    ///   derive --> sanitize["Detach Source Cache Identity"]
    ///   sanitize --> publish["Publish and Return TransformArtifact"]
    /// ```
    ///
    /// # Parameters
    ///   - `arguments`: Complete transform-specific specialization key.
    ///   - `derive`: Uncached deterministic structural derivation callback. Returning programs derived from regions
    ///     unrelated to the source region violates the region reuse contract and can create unreclaimable reference
    ///     cycles between region transform caches.
    #[cfg_attr(doc, aquamarine::aquamarine)]
    pub fn transform<T, Metadata, Error>(
        self,
        arguments: T::Arguments,
        derive: impl FnOnce(Self, &T::Arguments) -> Result<TransformArtifact<V, O, Metadata>, Error>,
    ) -> Result<TransformArtifact<V, O, Metadata>, Error>
    where
        T: Transform<Region<V, O>, Artifact = TransformArtifact<V, O, Metadata>> + 'static,
        T::Arguments: Debug + Send + Sync + 'static,
        Metadata: Clone + Debug + PartialEq + Send + Sync + 'static,
    {
        let namespace = self.transform_cache().namespace::<T>(T::DEFAULT_CACHE_CAPACITY);
        let erased_arguments = ErasedTransformArguments::new(arguments.clone());
        match namespace.try_entry(erased_arguments) {
            Ok(SpecializationCacheEntry::Occupied(cached)) => {
                let cached = cached.typed::<Metadata>();
                #[cfg(debug_assertions)]
                {
                    let fresh = self.sanitize_transform_artifact(derive(self, &arguments)?);
                    assert_transform_artifacts_match::<T, _, _, _>(&arguments, &cached, &fresh);
                }
                Ok(cached)
            }
            Ok(SpecializationCacheEntry::Vacant(producer)) => {
                let artifact = self.sanitize_transform_artifact(derive(self, &arguments)?);
                Ok(producer.insert(ErasedTransformArtifact::new(artifact)).typed::<Metadata>())
            }
            Err(ReentrantSpecializationError) => Ok(self.sanitize_transform_artifact(derive(self, &arguments)?)),
        }
    }

    /// Detaches this source cache identity from every program in `artifact` before it can be retained or returned.
    fn sanitize_transform_artifact<Metadata>(
        self,
        mut artifact: TransformArtifact<V, O, Metadata>,
    ) -> TransformArtifact<V, O, Metadata> {
        for program in &mut artifact.programs {
            Arc::make_mut(program).detach_transform_cache(self.transform_cache());
        }
        artifact
    }
}

/// Diagnoses a nondeterministic structural transform before a cached artifact is served.
#[cfg(debug_assertions)]
fn assert_transform_artifacts_match<T, V, O, Metadata>(
    arguments: &T::Arguments,
    cached: &TransformArtifact<V, O, Metadata>,
    fresh: &TransformArtifact<V, O, Metadata>,
) where
    V: Value,
    O: Operation<Type = V::Type>,
    T: Transform<Region<V, O>, Artifact = TransformArtifact<V, O, Metadata>>,
    T::Arguments: Debug,
    Metadata: Debug + PartialEq,
{
    let cached_programs = cached.programs.iter().map(ToString::to_string).collect::<Vec<_>>();
    let fresh_programs = fresh.programs.iter().map(ToString::to_string).collect::<Vec<_>>();
    if cached_programs == fresh_programs && cached.metadata == fresh.metadata {
        return;
    }

    // Pair the two renderings position by position as indexed blocks of real lines. Formatting the two `Vec<String>`s
    // instead would escape every newline in every program and leave the diagnostic unreadable.
    let mut programs = String::new();
    for index in 0..cached_programs.len().max(fresh_programs.len()) {
        for (label, renderings) in [("cached", &cached_programs), ("fresh", &fresh_programs)] {
            let rendering = renderings.get(index).map_or("<absent>", String::as_str);
            programs.push_str(&format!("--- {label} program {index} ---\n{rendering}\n"));
        }
    }
    panic!(
        "nondeterministic transform rule detected for `{}` with arguments {:?}: re-derivation produced a different \
         artifact than the region cache retained, but region transforms must be deterministic structural functions of \
         their complete reachable contents and arguments\n\ncached metadata: {:?}\nderived metadata: {:?}\n\n{}",
        type_name::<T>(),
        arguments,
        cached.metadata,
        fresh.metadata,
        programs,
    );
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::convert::Infallible;
    use std::fmt::{Debug, Formatter};
    use std::hash::{Hash, Hasher};
    use std::sync::{Arc, Barrier};

    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::parameters::Placeholder;
    use crate::programs::{Operation, ProgramBuilder, Region, RegionId, RegionRef, Value};
    use crate::tests::{IdentityTransform, TestRegionOperation};

    use super::*;

    /// Argument key whose constant hash proves equality, rather than hash alone, selects artifacts.
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct CollisionKey(usize);

    impl Hash for CollisionKey {
        fn hash<H: Hasher>(&self, state: &mut H) {
            0usize.hash(state);
        }
    }

    /// Two-entry transform namespace used by the generic cache tests.
    struct TestTransform;

    impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for TestTransform {
        type Arguments = CollisionKey;
        type Artifact = TransformArtifact<V, O, usize>;

        const DEFAULT_CACHE_CAPACITY: usize = 2;
    }

    /// Independently bounded one-entry namespace with the same argument and artifact layouts.
    struct OtherTransform;

    impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for OtherTransform {
        type Arguments = CollisionKey;
        type Artifact = TransformArtifact<V, O, usize>;

        const DEFAULT_CACHE_CAPACITY: usize = 1;
    }

    /// Key whose formatter panics, used to ensure cache summaries never inspect caller payloads.
    #[derive(Clone, PartialEq, Eq, Hash)]
    struct PanickingDebugKey;

    impl Debug for PanickingDebugKey {
        fn fmt(&self, _formatter: &mut Formatter<'_>) -> std::fmt::Result {
            panic!("argument formatting must not run")
        }
    }

    /// Namespace whose key and metadata formatters are deliberately unusable.
    struct PanickingDebugTransform;

    impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for PanickingDebugTransform {
        type Arguments = PanickingDebugKey;
        type Artifact = TransformArtifact<V, O, PanickingDebugKey>;

        const DEFAULT_CACHE_CAPACITY: usize = 1;
    }

    /// Builds a flat scalar identity program with no operation-specific behavior.
    fn identity_program() -> Program<Array, TestRegionOperation, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Derives the canonical test artifact for `key`.
    fn derive_test_artifact(
        region: RegionRef<'_, Array, TestRegionOperation>,
        key: &CollisionKey,
    ) -> Result<TransformArtifact<Array, TestRegionOperation, usize>, Infallible> {
        Ok(TransformArtifact::new(vec![Arc::new(region.to_program())], key.0))
    }

    #[test]
    fn test_transform_artifact() {
        let program = Arc::new(identity_program());
        let artifact = TransformArtifact::new(vec![program.clone()], 7usize);
        assert_eq!(artifact.programs().len(), 1);
        assert!(Arc::ptr_eq(&artifact.programs()[0], &program));
        assert_eq!(artifact.metadata(), &7);
        assert_eq!(format!("{artifact:?}"), "TransformArtifact { program_count: 1, metadata: 7 }",);

        let cloned = artifact.clone();
        let (programs, metadata) = artifact.into_parts();
        assert_eq!(metadata, 7);
        assert!(Arc::ptr_eq(&programs[0], &cloned.programs()[0]));
    }

    #[test]
    fn test_region_transform_namespaces_are_lazy_collision_safe_and_independent() {
        let program = identity_program();
        assert!(!program.entry_region().transform_cache().is_initialized());

        let first = program
            .entry_region_ref()
            .transform::<TestTransform, _, Infallible>(CollisionKey(0), derive_test_artifact)
            .unwrap();
        assert!(program.entry_region().transform_cache().is_initialized());
        let repeated = program
            .entry_region_ref()
            .transform::<TestTransform, _, Infallible>(CollisionKey(0), derive_test_artifact)
            .unwrap();
        assert!(Arc::ptr_eq(&first.programs()[0], &repeated.programs()[0]));

        let collision = program
            .entry_region_ref()
            .transform::<TestTransform, _, Infallible>(CollisionKey(1), derive_test_artifact)
            .unwrap();
        assert!(!Arc::ptr_eq(&first.programs()[0], &collision.programs()[0]));
        assert_eq!(collision.metadata(), &1);

        let other = program
            .entry_region_ref()
            .transform::<OtherTransform, _, Infallible>(CollisionKey(0), derive_test_artifact)
            .unwrap();
        assert!(!Arc::ptr_eq(&first.programs()[0], &other.programs()[0]));

        let evicting = program
            .entry_region_ref()
            .transform::<TestTransform, _, Infallible>(CollisionKey(2), derive_test_artifact)
            .unwrap();
        assert_eq!(evicting.metadata(), &2);
        let first_after_eviction = program
            .entry_region_ref()
            .transform::<TestTransform, _, Infallible>(CollisionKey(0), derive_test_artifact)
            .unwrap();
        assert!(!Arc::ptr_eq(&first.programs()[0], &first_after_eviction.programs()[0]));
        let other_repeated = program
            .entry_region_ref()
            .transform::<OtherTransform, _, Infallible>(CollisionKey(0), derive_test_artifact)
            .unwrap();
        assert!(Arc::ptr_eq(&other.programs()[0], &other_repeated.programs()[0]));

        let statistics = program.entry_region_ref().transform_statistics::<TestTransform>().unwrap();
        assert_eq!((statistics.productions, statistics.hits, statistics.evictions), (4, 1, 2));
        let statistics = program.entry_region_ref().transform_statistics::<OtherTransform>().unwrap();
        assert_eq!((statistics.productions, statistics.hits, statistics.evictions), (1, 1, 0));
    }

    #[test]
    fn test_region_transform_errors_panics_and_reentrancy_retry_cleanly() {
        let program = identity_program();
        let region = program.entry_region_ref();

        assert!(matches!(
            region.transform::<TestTransform, _, _>(CollisionKey(10), |_, _| {
                Err::<TransformArtifact<Array, TestRegionOperation, usize>, _>("derivation failed")
            }),
            Err("derivation failed"),
        ));
        let recovered =
            region.transform::<TestTransform, _, Infallible>(CollisionKey(10), derive_test_artifact).unwrap();
        assert_eq!(recovered.metadata(), &10);

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            region.transform::<TestTransform, _, Infallible>(CollisionKey(11), |_, _| panic!("derivation panicked"))
        }));
        assert!(panicked.is_err());
        let recovered =
            region.transform::<TestTransform, _, Infallible>(CollisionKey(11), derive_test_artifact).unwrap();
        assert_eq!(recovered.metadata(), &11);

        let derivations = Cell::new(0);
        let recursive = region
            .transform::<TestTransform, _, Infallible>(CollisionKey(12), |region, arguments| {
                derivations.set(derivations.get() + 1);
                let nested =
                    region.transform::<TestTransform, _, Infallible>(arguments.clone(), |region, arguments| {
                        derivations.set(derivations.get() + 1);
                        derive_test_artifact(region, arguments)
                    })?;
                assert_eq!(nested.metadata(), &12);
                derive_test_artifact(region, arguments)
            })
            .unwrap();
        assert_eq!(recursive.metadata(), &12);
        assert_eq!(derivations.get(), 2);

        let different_argument = region
            .transform::<TestTransform, _, Infallible>(CollisionKey(20), |region, arguments| {
                let nested =
                    region.transform::<TestTransform, _, Infallible>(CollisionKey(21), derive_test_artifact)?;
                assert_eq!(nested.metadata(), &21);
                derive_test_artifact(region, arguments)
            })
            .unwrap();
        assert_eq!(different_argument.metadata(), &20);

        let statistics = region.transform_statistics::<TestTransform>().unwrap();
        assert_eq!(statistics.abandoned_productions, 2);
    }

    #[test]
    fn test_region_transform_namespace_initialization_is_concurrent_and_cache_debug_is_payload_free() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RegionTransformCache<Array, TestRegionOperation>>();

        let program = Arc::new(identity_program());
        let barrier = Arc::new(Barrier::new(2));
        let handles = (0..2)
            .map(|index| {
                let program = program.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    program
                        .entry_region_ref()
                        .transform::<TestTransform, _, Infallible>(CollisionKey(index), derive_test_artifact)
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            assert_eq!(handle.join().unwrap().programs().len(), 1);
        }
        let statistics = program.entry_region_ref().transform_statistics::<TestTransform>().unwrap();
        assert_eq!(statistics.productions, 2);

        program
            .entry_region_ref()
            .transform::<PanickingDebugTransform, _, Infallible>(PanickingDebugKey, |region, _| {
                Ok(TransformArtifact::new(vec![Arc::new(region.to_program())], PanickingDebugKey))
            })
            .unwrap();
        let summary = format!("{:?}", program.entry_region().transform_cache());
        assert_eq!(summary, "RegionTransformCache { namespace_count: 2, artifact_count: 3 }");
    }

    #[test]
    fn test_region_transform_sanitization_prevents_source_cycles_and_preserves_descendant_caches() {
        let source = identity_program();
        let source_cache = source.entry_region().transform_cache().downgrade();
        let artifact = source.entry_region_ref().retained_identity_transform();
        assert!(!source.entry_region().transform_cache().ptr_eq(&artifact.entry_region().transform_cache));
        drop(source);
        assert!(!source_cache.is_alive());

        let leaf = identity_program();
        let leaf_retained = leaf.entry_region_ref().retained_identity_transform();
        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let leaf = builder.import_program(leaf);
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(
                TestRegionOperation::WithRegions(const { &[crate::RegionSlot::computation("body")] }),
                vec![leaf],
                vec![input],
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let transformed = program.entry_region_ref().retained_identity_transform();
        assert!(!program.entry_region().transform_cache().ptr_eq(&transformed.entry_region().transform_cache));
        let transformed_leaf = transformed.region_ref(RegionId::new(0)).unwrap().retained_identity_transform();
        assert!(Arc::ptr_eq(&leaf_retained, &transformed_leaf));
    }

    #[test]
    fn test_region_transform_does_not_require_static_value_or_operation_families() {
        fn transform_borrowed_universe<'r, V: Value, O: Operation<Type = V::Type>>(
            region: RegionRef<'r, V, O>,
        ) -> TransformArtifact<V, O, ()> {
            region
                .transform::<IdentityTransform, _, Infallible>((), |region, _| {
                    Ok(TransformArtifact::new(vec![Arc::new(region.to_program())], ()))
                })
                .unwrap()
        }

        let program = identity_program();
        assert_eq!(transform_borrowed_universe(program.entry_region_ref()).programs().len(), 1);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_region_transform_debug_recheck_detects_program_count_changes() {
        let program = identity_program();
        program.entry_region_ref().insert_transform_artifact_for_testing::<TestTransform, _>(
            CollisionKey(6),
            TransformArtifact::new(Vec::new(), 6),
        );
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            program
                .entry_region_ref()
                .transform::<TestTransform, _, Infallible>(CollisionKey(6), derive_test_artifact)
        }))
        .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        assert!(message.starts_with("nondeterministic transform rule detected for `"), "{message}");
        assert!(message.contains("TestTransform"), "{message}");
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_region_transform_debug_recheck_detects_program_rendering_changes() {
        let program = identity_program();
        let mut different_builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        different_builder.add_input(ArrayType::scalar(DataType::F64));
        let constant = different_builder.add_constant(Array::scalar(1.0));
        let different = different_builder.build(vec![constant], vec![Placeholder], vec![Placeholder]).unwrap();
        program.entry_region_ref().insert_transform_artifact_for_testing::<TestTransform, _>(
            CollisionKey(7),
            TransformArtifact::new(vec![Arc::new(different)], 7),
        );
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            program
                .entry_region_ref()
                .transform::<TestTransform, _, Infallible>(CollisionKey(7), derive_test_artifact)
        }))
        .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        assert!(message.starts_with("nondeterministic transform rule detected for `"), "{message}");
        assert!(message.contains("TestTransform"), "{message}");
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_region_transform_debug_recheck_detects_metadata_changes() {
        let program = identity_program();
        program.entry_region_ref().insert_transform_artifact_for_testing::<TestTransform, _>(
            CollisionKey(8),
            TransformArtifact::new(vec![Arc::new(program.entry_region_ref().to_program())], 9),
        );
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            program
                .entry_region_ref()
                .transform::<TestTransform, _, Infallible>(CollisionKey(8), derive_test_artifact)
        }))
        .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        assert!(message.starts_with("nondeterministic transform rule detected for `"), "{message}");
        assert!(message.contains("TestTransform"), "{message}");
    }
}
