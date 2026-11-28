use std::any::{Any, TypeId};
use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use ryft_core::{ArrayType, CompositeKey, LogicalMesh, Provenance, RegionId, ShardMapBoundary, Sharding};
use ryft_mlir::dialects::func;
use ryft_mlir::{AttributeRef, BlockRef, Context as MlirContext, Location, LocationRef, ValueRef};

use crate::lowering::sharding::{OpenDimensions, ShardingExtension};
use crate::lowering::{LoweringError, LoweringRule, LoweringRules, RaggedDotLoweringStrategy};
use crate::{Error, RetainedArray, RuntimeRequirement, normalize_runtime_requirements};

/// Context for lowering a program in the current scope of its MLIR module. Each instruction lowerer holds
/// a clone of this handle, sharing the module resources and the current scope's bindings through one [`Rc`].
/// [`enter_manual_region`](Self::enter_manual_region) derives a context for the body of an `sdy.manual_computation`.
/// Its manual axes and reachable session constants belong to that scope. The child inherits the module's target and
/// lowering policy, requests the same private functions, and continues to allocate channels and record module
/// requirements alongside the enclosing context.
#[derive(Clone)]
pub(crate) struct ProgramLoweringContext {
    /// Underlying [`ProgramLoweringContextState`].
    state: Rc<ProgramLoweringContextState>,
}

impl ProgramLoweringContext {
    /// Creates a new [`ProgramLoweringContext`] for one program, outside any manual region and without target
    /// information.
    #[inline]
    pub(crate) fn new() -> Self {
        Self {
            state: Rc::new(ProgramLoweringContextState {
                target_platform: None,
                ragged_dot_lowering_strategy: RaggedDotLoweringStrategy::default(),
                lowering_rules: LoweringRules::default(),
                module: Rc::new(ModuleResources {
                    next_channel_id: Cell::new(1),
                    functions: RefCell::default(),
                    mesh: OnceCell::new(),
                    has_provenance: Cell::new(false),
                    requires_assertion_handler: Cell::new(false),
                    runtime_requirements: RefCell::new(Vec::new()),
                }),
                manual_regions: Vec::new(),
                bound_manual_axes: Vec::new(),
                shares_functions: true,
                session_constant_indices: Rc::default(),
            }),
        }
    }

    /// Returns this context carrying the PJRT platform name of the compilation target.
    #[inline]
    pub(crate) fn with_target_platform(mut self, target_platform: Option<&str>) -> Self {
        Rc::make_mut(&mut self.state).target_platform = target_platform.map(Rc::from);
        self
    }

    /// Returns this context carrying the [`RaggedDotLoweringStrategy`] of the module.
    #[inline]
    pub(crate) fn with_ragged_dot_lowering_strategy(mut self, strategy: RaggedDotLoweringStrategy) -> Self {
        Rc::make_mut(&mut self.state).ragged_dot_lowering_strategy = strategy;
        self
    }

    /// Returns this context carrying the [`LoweringRule`]s of the module.
    #[inline]
    pub(crate) fn with_lowering_rules(mut self, lowering_rules: LoweringRules) -> Self {
        Rc::make_mut(&mut self.state).lowering_rules = lowering_rules;
        self
    }

    /// Returns this context for a scope that inlines its calls.
    /// Refer to [`Self::shared_function`] for more information.
    #[inline]
    pub(crate) fn without_shared_functions(mut self) -> Self {
        Rc::make_mut(&mut self.state).shares_functions = false;
        self
    }

    /// Binds `session_constants` in the order their values appear at the end of the capture table. Hidden entry inputs
    /// come first, followed by placeholders for arrays passed to callees that never read them.
    #[inline]
    pub(crate) fn with_session_constants<'a>(
        mut self,
        session_constants: impl IntoIterator<Item = &'a RetainedArray>,
    ) -> Self {
        let indices = session_constants
            .into_iter()
            .enumerate()
            .map(|(index, constant)| (constant.clone(), index))
            .collect();
        Rc::make_mut(&mut self.state).session_constant_indices = Rc::new(indices);
        self
    }

    /// Returns the PJRT platform name of the compilation target, or `None` when the lowering has no target
    /// information.
    #[inline]
    pub(crate) fn target_platform(&self) -> Option<&str> {
        self.state.target_platform.as_deref()
    }

    /// Returns the ragged dot lowering strategy for this module.
    #[inline]
    pub(crate) fn ragged_dot_lowering_strategy(&self) -> RaggedDotLoweringStrategy {
        self.state.ragged_dot_lowering_strategy
    }

    /// Returns the rule that lowers the instructions of the operation named `operation_name` in this module, in
    /// place of the lowering of their operation family, if the module has one.
    #[inline]
    pub(crate) fn lowering_rule(&self, operation_name: &str) -> Option<&Rc<dyn LoweringRule>> {
        self.state.lowering_rules.get(operation_name)
    }

    /// Returns the private function at position `index` among those that this module holds so far, in the order in
    /// which they were registered, or [`None`] when it holds fewer. The module emits them in that order, and it asks
    /// by position because lowering one function can request another.
    #[inline]
    pub(crate) fn function(&self, index: usize) -> Option<PrivateFunction> {
        self.state.module.functions.borrow().functions.get(index).cloned()
    }

    /// Returns the mesh that lowering has recorded for this module so far, which is the mesh that the module declares,
    /// or [`None`] when nothing that lowering has emitted refers to a mesh.
    #[inline]
    pub(super) fn mesh(&self) -> Option<LogicalMesh> {
        self.state.module.mesh.get().cloned()
    }

    /// Returns whether any scope lowered instruction provenance into an MLIR location.
    #[inline]
    pub(super) fn has_provenance(&self) -> bool {
        self.state.module.has_provenance.get()
    }

    /// Returns whether this module requires the runtime assertion handler.
    #[inline]
    pub(super) fn requires_assertion_handler(&self) -> bool {
        self.state.module.requires_assertion_handler.get()
    }

    /// Returns the runtime requirements that lowering has recorded for this module so far.
    #[inline]
    pub(super) fn runtime_requirements(&self) -> Vec<RuntimeRequirement> {
        self.state.module.runtime_requirements.borrow().clone()
    }

    /// Returns the innermost enclosing manual region's [`ShardMapBoundary`], or `None` outside manual regions.
    #[inline]
    pub(crate) fn manual_shard_map(&self) -> Option<&ShardMapBoundary> {
        self.state.manual_regions.last().map(Rc::as_ref)
    }

    /// Returns the nearest enclosing region binding `axis_name`.
    #[inline]
    pub(crate) fn manual_axis_region(&self, axis_name: &str) -> Option<&ShardMapBoundary> {
        self.state
            .manual_regions
            .iter()
            .rev()
            .find(|region| region.manual_axes().iter().any(|axis| axis == axis_name))
            .map(Rc::as_ref)
    }

    /// Returns the axes bound by any enclosing manual region, excluded from shard-local placement constraints.
    #[inline]
    pub(crate) fn bound_manual_axes(&self) -> &[String] {
        &self.state.bound_manual_axes
    }

    /// Returns the suffix of `captured_values` that holds the values of the session constants reachable in this scope.
    /// A scope whose `captured_values` table cannot hold them (e.g., a shared private function) has no such values.
    #[inline]
    pub(crate) fn session_constant_values<'o, V>(&self, captured_values: &'o [V]) -> &'o [V] {
        captured_values
            .len()
            .checked_sub(self.state.session_constant_indices.len())
            .map_or(&[], |start| &captured_values[start..])
    }

    /// Returns the private function that this module holds for `key`, and registers it when the module holds none yet,
    /// so that everything that requests a function with an equal key shares one function. Keys are compared within
    /// their own type and within the target platform of this context. The first request for a key calls `body`, which
    /// returns what the function computes and must not lower anything itself, and every later request returns the same
    /// function without calling it. The symbol of the function is `name`, with a numeric suffix when the module already
    /// holds a function of that name.
    ///
    /// Whatever emits a reference to a private function (e.g., a `func.call`, or the decomposition of a
    /// `stablehlo.composite`) requests it here, so that the module holds a function exactly when something that
    /// lowering emitted or planned refers to it.
    ///
    /// # Errors
    ///
    /// Returns the error of `body` when it fails, in which case nothing is registered for `key`.
    pub(crate) fn require_function<K, F>(&self, key: K, name: &str, body: F) -> Result<PrivateFunction, Error>
    where
        K: Eq + Hash + 'static,
        F: FnOnce() -> Result<PrivateFunctionBody, Error>,
    {
        if let Some(function) = self.registered_function(&key) {
            return Ok(function);
        }

        // The registry is not borrowed while `body` runs, so that tracing the body can use this context.
        let body = body()?;
        let mut functions = self.state.module.functions.borrow_mut();
        Ok(functions.register(self.state.target_platform.clone(), Box::new(key), name, body))
    }

    /// Makes requests for `alias` return the private function that this module holds for `key`, so that a caller that
    /// knows a function by a cheaper key (e.g., by the identity and boundary of a named composite call rather than by
    /// its body) finds it without computing `key`. Nothing happens when the module holds no function for `key`.
    pub(crate) fn alias_function<A: Eq + Hash + 'static, K: Eq + Hash + 'static>(&self, alias: A, key: &K) {
        let mut functions = self.state.module.functions.borrow_mut();
        let indices = functions.indices.entry(self.state.target_platform.clone()).or_default();
        if let Some(&index) = indices.get(key as &dyn ErasedKey) {
            indices.insert(Box::new(alias), index);
        }
    }

    /// Returns the private function that this module holds for `key`, for a caller that calls the function when the
    /// module holds one and inlines what it computes otherwise (e.g., a `call`, whose callee is shared only when the
    /// module plans a function for it). Returns [`None`] in a scope that inlines its calls (i.e., one for which
    /// [`Self::without_shared_functions`] was called), which includes the bodies of manual regions, because a
    /// private function is lowered outside any manual region.
    #[inline]
    pub(crate) fn shared_function<K: Eq + Hash + 'static>(&self, key: &K) -> Option<PrivateFunction> {
        self.state.shares_functions.then(|| self.registered_function(key)).flatten()
    }

    /// Returns the private function that this module holds for `key` on the target platform of this context.
    fn registered_function<K: Eq + Hash + 'static>(&self, key: &K) -> Option<PrivateFunction> {
        let functions = self.state.module.functions.borrow();
        let index = *functions.indices.get(&self.state.target_platform)?.get(key as &dyn ErasedKey)?;
        Some(functions.functions[index].clone())
    }

    /// Records `mesh` as the mesh that this module declares, which everything that lowering emits over a mesh (i.e.,
    /// the sharding of a signature value, of a constraint, or of the boundary of a manual computation) refers to by
    /// one symbol. Whatever lowers a sharding records its mesh here, so the module declares a mesh exactly when
    /// something that lowering emitted refers to one, rather than when an analysis of the program predicts that.
    /// The module keeps the first mesh that is recorded, including its axis types, because the emitted `sdy.mesh`
    /// records only axis names and sizes.
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::MeshTopologyMismatch`] when the axis names or sizes of `mesh` differ from those
    /// of the recorded mesh, including in order.
    pub(crate) fn require_mesh(&self, mesh: &LogicalMesh) -> Result<(), Error> {
        let recorded = self.state.module.mesh.get_or_init(|| mesh.clone());
        if recorded.has_same_topology(mesh) {
            Ok(())
        } else {
            Err(Error::Lowering(LoweringError::MeshTopologyMismatch {
                expected: recorded.clone(),
                actual: mesh.clone(),
            }))
        }
    }

    /// Records that this module requires the runtime assertion handler.
    #[inline]
    pub(super) fn require_assertion_handler(&self) {
        self.state.module.requires_assertion_handler.set(true);
    }

    /// Records `requirement` as a runtime requirement of this module.
    ///
    /// # Errors
    ///
    /// Returns the errors of `normalize_runtime_requirements` when `requirement` conflicts with a requirement
    /// that is already recorded.
    #[inline]
    pub(super) fn require_runtime_requirement(&self, requirement: RuntimeRequirement) -> Result<(), Error> {
        let mut requirements = self.state.module.runtime_requirements.borrow_mut();
        let mut updated = requirements.clone();
        updated.push(requirement);
        *requirements = normalize_runtime_requirements(updated)?;
        Ok(())
    }

    /// Returns a fresh module-unique channel ID for one channeled collective.
    pub(crate) fn allocate_channel_id(&self) -> usize {
        let channel_id = self.state.module.next_channel_id.get();
        self.state.module.next_channel_id.set(channel_id + 1);
        channel_id
    }

    /// Derives a child context whose innermost manual region binds the axes of `boundary`. The body is isolated from
    /// above, so its operations cannot use the enclosing scope's session constant bindings, and it inlines its calls.
    pub(crate) fn enter_manual_region(&self, boundary: ShardMapBoundary) -> Self {
        let mut child = self.clone();
        let state = Rc::make_mut(&mut child.state);
        for axis in boundary.manual_axes() {
            if !state.bound_manual_axes.contains(axis) {
                state.bound_manual_axes.push(axis.clone());
            }
        }
        state.manual_regions.push(Rc::new(boundary));
        state.shares_functions = false;
        state.session_constant_indices = Rc::default();
        child
    }

    /// Lowers a tensor placement constraint of `input_value` to `sharding` in this scope and returns the constrained
    /// value. The constraint is that of [`ShardingExtension::lower_sharding_constraint`] without the mesh axes that
    /// the enclosing manual regions bind, and the mesh of `sharding` is recorded as the mesh of the module.
    pub(crate) fn lower_sharding_constraint<'b, 'c: 'b, 't: 'c>(
        &self,
        sharding: &Sharding,
        input_value: ValueRef<'b, 'c, 't>,
        open_dimensions: OpenDimensions<'_>,
        block: &mut BlockRef<'b, 'c, 't>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        self.require_mesh(sharding.mesh())?;
        sharding.lower_sharding_constraint(input_value, open_dimensions, self.bound_manual_axes(), block, location)
    }

    /// Lowers a tensor placement constraint that replicates `input_value`, whose tracked placement
    /// is `sharding`, along `axes` in this scope and returns the constrained value. The constraint is that of
    /// [`ShardingExtension::lower_replication_constraint`] without the mesh axes that the enclosing manual regions
    /// bind, and the mesh of `sharding` is recorded as the mesh of the module.
    pub(crate) fn lower_replication_constraint<'b, 'c: 'b, 't: 'c>(
        &self,
        sharding: &Sharding,
        input_value: ValueRef<'b, 'c, 't>,
        axes: &[usize],
        block: &mut BlockRef<'b, 'c, 't>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        self.require_mesh(sharding.mesh())?;
        sharding.lower_replication_constraint(input_value, axes, self.bound_manual_axes(), block, location)
    }

    /// Constrains `input_value` to a `sharding` that declares or completes a pending reduction. Every dimension is
    /// closed, including dimensions split along auto axes. Unconstrained dimensions become replicated. Leaving a
    /// dimension open lets Shardy reduce the operands before the operation and then recreate a pending reduction on
    /// its result. For example, doubling contributions of 1 and 2 can then yield a global sum of 12 instead of 6.
    /// [Inline lowering](crate::lowering::programs::ProgramInlineLoweringExtension::lower_inline) retains known
    /// splits along auto axes for intermediate values, so closing these dimensions preserves their placement.
    /// As with [`Self::lower_sharding_constraint`], the module records the mesh and excludes enclosing manual axes.
    pub(super) fn lower_pending_reduction_constraint<'b, 'c: 'b, 't: 'c>(
        &self,
        sharding: &Sharding,
        input_value: ValueRef<'b, 'c, 't>,
        block: &mut BlockRef<'b, 'c, 't>,
        location: LocationRef<'c, 't>,
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        let closed = sharding.with_unconstrained_dimensions_replicated();
        self.lower_sharding_constraint(&closed, input_value, OpenDimensions::None, block, location)
    }

    /// Returns the value of `constant` in a scope whose captured values are `captured_values`.
    pub(crate) fn lower_session_constant<'b, 'c: 'b, 't: 'c>(
        &self,
        constant: &RetainedArray,
        captured_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        let values = self.session_constant_values(captured_values);
        self.state
            .session_constant_indices
            .get(constant)
            .copied()
            .filter(|_| values.len() == self.state.session_constant_indices.len())
            .map(|index| values[index])
            .ok_or_else(|| Error::Lowering(LoweringError::UnboundSessionConstant { constant: constant.to_string() }))
    }

    /// Lowers one instruction's [`Provenance`] onto MLIR locations above the provided base location and records
    /// module-wide whether any instruction-specific location was constructed. Unknown provenance uses the base location
    /// unchanged, each scope level becomes one [`NameLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc) above
    /// its lowered origin, and fused provenance becomes one metadata-free
    /// [`FusedLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc) over its recursively lowered origins. An
    /// existing file/line base location is thereby preserved as the innermost child of the named scopes rather than
    /// being replaced.
    pub(super) fn lower_provenance<'c, 't: 'c>(
        &self,
        mlir_context: &'c MlirContext<'t>,
        provenance: &Provenance,
        base: LocationRef<'c, 't>,
    ) -> LocationRef<'c, 't> {
        /// Recursively lowers `provenance` above `base` without touching the module-wide flag.
        fn lower<'c, 't: 'c>(
            mlir_context: &'c MlirContext<'t>,
            provenance: &Provenance,
            base: LocationRef<'c, 't>,
        ) -> LocationRef<'c, 't> {
            if let Some((scope, origin)) = provenance.as_scope() {
                let child = lower(mlir_context, origin, base);
                mlir_context.named_location(scope.name(), Some(child)).as_ref()
            } else if let Some(origins) = provenance.as_fused() {
                let locations = origins.iter().map(|origin| lower(mlir_context, origin, base)).collect::<Vec<_>>();
                mlir_context.fused_location::<_, AttributeRef>(locations.as_slice(), None).as_ref()
            } else {
                base
            }
        }

        if provenance.is_unknown() {
            return base;
        }

        self.state.module.has_provenance.set(true);

        lower(mlir_context, provenance, base)
    }
}

/// State held by a [`ProgramLoweringContext`]. Deriving a scope copies its bindings and shares the
/// [`ModuleResources`] that all scopes update while emitting the module.
#[derive(Clone)]
struct ProgramLoweringContextState {
    /// PJRT platform name of the compilation target (e.g., `"cuda"` or `"cpu"`), or `None` when the lowering has no
    /// target information. Lowerings that depend on the platform, such as fused attention, consult this and fall back
    /// to their portable form when it is absent.
    target_platform: Option<Rc<str>>,

    /// Backend representation used to lower ragged dots in this module.
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,

    /// Rules that lower the instructions of some operations in this module, in place of the lowering of their
    /// operation family.
    lowering_rules: LoweringRules,

    /// Resources of the module that every scope of the program shares.
    module: Rc<ModuleResources>,

    /// Enclosing manual regions, ordered from outermost to innermost for named-axis resolution.
    manual_regions: Vec<Rc<ShardMapBoundary>>,

    /// Axes bound by any enclosing manual region, excluded from shard-local placement constraints.
    bound_manual_axes: Vec<String>,

    /// Whether calls in this scope may call the private functions that the module plans for their callees. Calls in
    /// a scope that clears this are inlined.
    shares_functions: bool,

    /// Position of each retained array within the session constant suffix of `captured_values`. The suffix holds hidden
    /// entry inputs followed by bufferless placeholders for arrays that the program only passes to callees that never
    /// read them. Entering an `sdy.manual_computation` clears these bindings because its body is isolated.
    session_constant_indices: Rc<HashMap<RetainedArray, usize>>,
}

/// Resources of one module that every lowering scope of its program reads and updates while the module is emitted.
/// A [`ProgramLoweringContextState`] holds them behind one [`Rc`], so a context that is derived for a nested scope
/// shares all of them with the context that it was derived from. This type is deliberately not [`Clone`], so that
/// no scope can end up with resources of its own.
struct ModuleResources {
    /// Next unused collective channel ID in this module.
    next_channel_id: Cell<usize>,

    /// Private functions that the emitted operations of every nested lowering scope refer to, which the module emits.
    functions: RefCell<PrivateFunctionRegistry>,

    /// First mesh recorded for a sharding in this module. Subsequent shardings must use the same topology.
    mesh: OnceCell<LogicalMesh>,

    /// Flag of the module recording whether any recursive lowering path constructed a location specific to an
    /// instruction (i.e., not the base location) from a known [`Provenance`]. It selects the module serialization
    /// mode after lowering completes. Only a module that actually carries provenance locations is printed with debug
    /// information, so modules without provenance keep their existing StableHLO text and cache keys byte for byte.
    has_provenance: Cell<bool>,

    /// Whether any emitted instruction requires the runtime assertion handler.
    requires_assertion_handler: Cell<bool>,

    /// Runtime contracts declared by emitted operations across every nested lowering scope.
    runtime_requirements: RefCell<Vec<RuntimeRequirement>>,
}

/// Identity of one generated decomposition of a named composition at a concrete typed boundary. The lowering of an
/// operation can request the private function that holds such a decomposition with this key (refer to
/// [`Lowerer::require_function`](crate::Lowerer::require_function)), and requests with equal keys share one
/// function.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NamedCompositionKey {
    /// Semantic identity of the composition.
    composite: CompositeKey,

    /// Logical input types of the composition boundary.
    input_types: Vec<ArrayType>,

    /// Logical output types of the composition boundary.
    output_types: Vec<ArrayType>,
}

impl NamedCompositionKey {
    /// Creates the key of the decomposition of `composite` at the boundary with the provided logical types.
    #[inline]
    pub fn new(composite: CompositeKey, input_types: Vec<ArrayType>, output_types: Vec<ArrayType>) -> Self {
        Self { composite, input_types, output_types }
    }
}

/// Module-owned registry of the private functions that lowering requests. The registry is independent of what requests
/// a function and of the operation family of its body, and the module emits the functions that were registered. Refer
/// to [`ProgramLoweringContext::require_function`] for more information.
#[derive(Default)]
struct PrivateFunctionRegistry {
    /// Position in [`Self::functions`] of the function of each key, for each target platform. A module that is lowered
    /// for several platforms lowers a function once for each platform that requests it, because the platform can change
    /// what the function lowers to.
    indices: HashMap<Option<Rc<str>>, HashMap<Box<dyn ErasedKey>, usize>>,

    /// Functions in the stable order in which they were registered, which is the order in which the module emits them,
    /// so that the layout of the module is deterministic.
    functions: Vec<PrivateFunction>,

    /// Number of functions that were registered under each name, which makes their symbols unique.
    name_counts: HashMap<String, usize>,
}

impl PrivateFunctionRegistry {
    /// Registers a function that computes `body` for `key` on `target_platform` and returns it. Its symbol is `name`,
    /// suffixed with the number of functions that were registered under that name before it, if any.
    fn register(
        &mut self,
        target_platform: Option<Rc<str>>,
        key: Box<dyn ErasedKey>,
        name: &str,
        body: PrivateFunctionBody,
    ) -> PrivateFunction {
        let count = self.name_counts.entry(name.to_string()).or_default();
        let symbol = match *count {
            0 => name.to_string(),
            count => format!("{name}_{count}"),
        };
        *count += 1;
        let function = PrivateFunction { symbol: symbol.into(), body };
        self.indices.entry(target_platform).or_default().insert(key, self.functions.len());
        self.functions.push(function.clone());
        function
    }
}

/// Private function that a module holds, which [`ProgramLoweringContext::require_function`] registers.
#[derive(Clone)]
pub(crate) struct PrivateFunction {
    /// Module-unique symbol of the function.
    symbol: Rc<str>,

    /// What the function computes.
    body: PrivateFunctionBody,
}

impl PrivateFunction {
    /// Returns the module-unique symbol of the function.
    #[inline]
    pub(crate) fn symbol(&self) -> &Rc<str> {
        &self.symbol
    }

    /// Returns what the function computes.
    #[inline]
    pub(super) fn body(&self) -> &PrivateFunctionBody {
        &self.body
    }
}

/// What a [`PrivateFunction`] computes.
#[derive(Clone)]
pub(crate) enum PrivateFunctionBody {
    /// Region of the program that the module is lowered from, whose inputs and outputs are those of the function.
    /// The module resolves the region when it emits the function, so the registry does not depend on the operation
    /// family of that program.
    Region {
        /// Region that the function computes.
        region: RegionId,

        /// Number of leading inputs of the region that are its lifted captures, which the captured constants of the
        /// region refer to (i.e., the capture count of the `call`s that call the function).
        capture_count: usize,
    },

    /// Program that was built for this function, whose inputs and outputs are those of the function.
    /// It has no captures.
    Program(Rc<dyn PrivateFunctionProgram>),
}

/// Program that a module can emit as the body of a [`PrivateFunction`], whatever its operation family is.
pub(crate) trait PrivateFunctionProgram {
    /// Returns the private `func.func` named `symbol` whose body is the lowering of this program in `lowering_context`.
    /// The function takes the inputs of this program, followed by one input for each session constant that the program
    /// reads.
    fn lower_private_function<'c, 't: 'c>(
        &self,
        symbol: &str,
        lowering_context: &ProgramLoweringContext,
        mlir_context: &'c MlirContext<'t>,
        location: LocationRef<'c, 't>,
    ) -> Result<func::DetachedFuncOperation<'c, 't>, Error>;
}

/// Key of a [`PrivateFunction`] of any type that can key a hash map, which the registry compares and hashes without
/// knowing that type. Keys of different types are never equal.
trait ErasedKey: Any {
    /// Returns this key as [`Any`], for comparing it with a key of a known type.
    fn as_any(&self) -> &dyn Any;

    /// Returns whether `other` has the type of this key and is equal to it.
    fn erased_eq(&self, other: &dyn ErasedKey) -> bool;

    /// Feeds the type of this key and then the key itself into `state`.
    fn erased_hash(&self, state: &mut dyn Hasher);
}

impl<K: Eq + Hash + 'static> ErasedKey for K {
    #[inline]
    fn as_any(&self) -> &dyn Any {
        self
    }

    #[inline]
    fn erased_eq(&self, other: &dyn ErasedKey) -> bool {
        other.as_any().downcast_ref::<K>().is_some_and(|other| self == other)
    }

    #[inline]
    fn erased_hash(&self, mut state: &mut dyn Hasher) {
        TypeId::of::<K>().hash(&mut state);
        self.hash(&mut state);
    }
}

impl PartialEq for dyn ErasedKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.erased_eq(other)
    }
}

impl Eq for dyn ErasedKey {}

impl Hash for dyn ErasedKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.erased_hash(state);
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use ryft_core::{
        DataType, Dimension, DomainTracingContext, MeshAxis, MeshAxisType, ParallelReduce, ProgramError,
        ProvenanceScope, ReductionKind, Shape, ShardMapOptions, ShardingDimension, shard_map,
    };
    use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue};

    use crate::lowering::sharding::tests::{test_input_type, test_mesh};
    use crate::lowering::tests::{DefaultLoweringRule, test_matrix_type};
    use crate::lowering::types::ArrayTypeLoweringExtension;
    use crate::sessions::tests::{domain_mesh, f32_scalar};
    use crate::tests::cpu_client;
    use crate::{
        ArrayTracer, ProgramLoweringExtension, ProgramLoweringOptions, RuntimeError, Session, TracedProgram, Tracer,
        Type, trace,
    };

    use super::*;

    #[test]
    fn test_program_lowering_context_new() {
        // A new context has no target, is outside any manual region, and has recorded nothing about its module.
        let lowering_context = ProgramLoweringContext::new();
        assert_eq!(lowering_context.target_platform(), None);
        assert_eq!(lowering_context.ragged_dot_lowering_strategy(), RaggedDotLoweringStrategy::Decomposition);
        assert!(lowering_context.function(0).is_none());
        assert_eq!(lowering_context.mesh(), None);
        assert!(!lowering_context.has_provenance());
        assert!(!lowering_context.requires_assertion_handler());
        assert_eq!(lowering_context.runtime_requirements(), Vec::new());
        assert_eq!(lowering_context.manual_shard_map(), None);
        assert!(lowering_context.bound_manual_axes().is_empty());
        assert!(lowering_context.session_constant_values(&[0, 1]).is_empty());
    }

    #[test]
    fn test_program_lowering_context_with_target_platform() {
        // The target is a binding of the context that receives it, and the context that it was derived from keeps its
        // own.
        let lowering_context = ProgramLoweringContext::new();
        let cuda_context = lowering_context.clone().with_target_platform(Some("cuda"));
        assert_eq!(cuda_context.target_platform(), Some("cuda"));
        assert_eq!(cuda_context.with_target_platform(None).target_platform(), None);
        assert_eq!(lowering_context.target_platform(), None);
    }

    #[test]
    fn test_program_lowering_context_with_ragged_dot_lowering_strategy() {
        let lowering_context = ProgramLoweringContext::new();
        let instruction_context =
            lowering_context.clone().with_ragged_dot_lowering_strategy(RaggedDotLoweringStrategy::Instruction);
        assert_eq!(instruction_context.ragged_dot_lowering_strategy(), RaggedDotLoweringStrategy::Instruction);
        assert_eq!(lowering_context.ragged_dot_lowering_strategy(), RaggedDotLoweringStrategy::Decomposition);
    }

    #[test]
    fn test_program_lowering_context_with_lowering_rules() {
        // The rules belong to the module, so the body of a manual region has them as well.
        let lowering_context = ProgramLoweringContext::new();
        assert!(lowering_context.lowering_rule("add").is_none());
        let rules = LoweringRules::default().with_rule("add".to_string(), Rc::new(DefaultLoweringRule));
        let lowering_context = lowering_context.with_lowering_rules(rules);
        assert!(lowering_context.lowering_rule("add").is_some());
        assert!(lowering_context.lowering_rule("mul").is_none());
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        assert!(lowering_context.enter_manual_region(boundary).lowering_rule("add").is_some());
    }

    #[test]
    fn test_program_lowering_context_with_session_constants() {
        // Session constants are bound in the order provided, and binding them again replaces the earlier bindings.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let first = RetainedArray::new(&f32_scalar(&session, &mesh, 1.0));
        let second = RetainedArray::new(&f32_scalar(&session, &mesh, 2.0));
        let captures = [0, 1, 2];
        let lowering_context = ProgramLoweringContext::new().with_session_constants([&first, &second]);
        assert_eq!(lowering_context.session_constant_values(&captures), &[1, 2]);
        let lowering_context = lowering_context.with_session_constants([&second]);
        assert_eq!(lowering_context.session_constant_values(&captures), &[2]);
    }

    #[test]
    fn test_program_lowering_context_session_constant_values() {
        // The values of the session constants are the suffix of the capture table, and a table that is too short to
        // hold them (e.g., that of a shared private function) holds none.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let first = RetainedArray::new(&f32_scalar(&session, &mesh, 1.0));
        let second = RetainedArray::new(&f32_scalar(&session, &mesh, 2.0));
        let lowering_context = ProgramLoweringContext::new().with_session_constants([&first, &second]);
        assert_eq!(lowering_context.session_constant_values(&[0, 1, 2]), &[1, 2]);
        assert_eq!(lowering_context.session_constant_values(&[1, 2]), &[1, 2]);
        assert!(lowering_context.session_constant_values(&[2]).is_empty());
    }

    #[test]
    fn test_program_lowering_context_require_function() {
        // The first request for a key registers its function, which is named as requested.
        let body = || {
            let input_types = vec![Type::from(ArrayType::scalar(DataType::F32))];
            let (_, program) = DomainTracingContext::<Session>::trace(|inputs: Vec<Tracer>| Ok(inputs), input_types)?;
            Ok(PrivateFunctionBody::Program(Rc::new(program.into_simplified()?)))
        };
        let symbol = |function: Result<PrivateFunction, Error>| function.map(|function| function.symbol().to_string());
        let lowering_context = ProgramLoweringContext::new();
        assert!(lowering_context.function(0).is_none());
        assert_eq!(symbol(lowering_context.require_function("first", "f", body)), Ok("f".to_string()));

        // Every lowering scope of a program registers into the same module, including the body of a manual region, and
        // the body of a function that is already registered is not requested again. Another key that asks for a name
        // that is taken receives a suffixed symbol.
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        let region_context = lowering_context.enter_manual_region(boundary);
        assert_eq!(symbol(region_context.require_function("second", "f", body)), Ok("f_1".to_string()));
        assert_eq!(symbol(region_context.require_function("first", "g", || unreachable!())), Ok("f".to_string()));
        let symbols = (0..3)
            .map(|index| lowering_context.function(index).map(|function| function.symbol().to_string()))
            .collect::<Vec<_>>();
        assert_eq!(symbols, vec![Some("f".to_string()), Some("f_1".to_string()), None]);

        // Keys of different types never share a function, even when they hold equal values.
        assert_eq!(symbol(lowering_context.require_function(1_u8, "h", body)), Ok("h".to_string()));
        assert_eq!(symbol(lowering_context.require_function(1_u16, "h", body)), Ok("h_1".to_string()));

        // Keys are compared within the target platform of the requesting context.
        let other_platform = lowering_context.clone().with_target_platform(Some("cuda"));
        assert_eq!(symbol(other_platform.require_function("first", "f", body)), Ok("f_2".to_string()));

        // A function whose body cannot be built is not registered.
        let error = ProgramError::MalformedProgram("no body".to_string());
        assert_eq!(
            symbol(lowering_context.require_function("third", "f", || Err(error.clone().into()))),
            Err(Error::from(error)),
        );
        assert!(lowering_context.function(5).is_none());
    }

    #[test]
    fn test_program_lowering_context_alias_function() {
        let lowering_context = ProgramLoweringContext::new();
        let body = || Ok(PrivateFunctionBody::Region { region: RegionId::new(0), capture_count: 0 });
        lowering_context.require_function("key", "f", body).unwrap();

        // Requests for an alias return the function of the key that it was given for.
        lowering_context.alias_function(7_usize, &"key");
        let aliased = lowering_context.shared_function(&7_usize).unwrap();
        assert_eq!(aliased.symbol().as_ref(), "f");
        assert!(lowering_context.function(1).is_none());

        // An alias for a key without a function finds nothing.
        lowering_context.alias_function(8_usize, &"missing");
        assert!(lowering_context.shared_function(&8_usize).is_none());
    }

    #[test]
    fn test_program_lowering_context_shared_function() {
        let lowering_context = ProgramLoweringContext::new();
        assert!(lowering_context.shared_function(&"key").is_none());
        let body = || Ok(PrivateFunctionBody::Region { region: RegionId::new(0), capture_count: 0 });
        lowering_context.require_function("key", "f", body).unwrap();
        assert_eq!(lowering_context.shared_function(&"key").unwrap().symbol().as_ref(), "f");

        // A scope that inlines its calls finds no shared function, which includes the body of a manual region.
        assert!(lowering_context.clone().without_shared_functions().shared_function(&"key").is_none());
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        assert!(lowering_context.enter_manual_region(boundary).shared_function(&"key").is_none());
    }

    #[test]
    fn test_program_lowering_context_require_mesh() {
        let mesh = |axes: [(&str, usize); 2], r#type: MeshAxisType| {
            LogicalMesh::new(axes.map(|(name, size)| MeshAxis::new(name, size, r#type).unwrap()).to_vec()).unwrap()
        };
        let explicit = mesh([("x", 2), ("y", 2)], MeshAxisType::Explicit);
        let lowering_context = ProgramLoweringContext::new();
        assert_eq!(lowering_context.mesh(), None);
        assert_eq!(lowering_context.require_mesh(&explicit), Ok(()));
        assert_eq!(lowering_context.mesh(), Some(explicit.clone()));

        // A mesh that differs only in axis types shares the module mesh, which keeps the axis types of its first mesh.
        let manual = mesh([("x", 2), ("y", 2)], MeshAxisType::Manual);
        assert_eq!(lowering_context.require_mesh(&manual), Ok(()));
        assert_eq!(lowering_context.mesh(), Some(explicit.clone()));

        // Device ids follow the order of the axes of the module mesh, so a reordered mesh is another topology, as is a
        // mesh with other axis names or sizes. The module mesh stays the recorded one.
        for other in [
            mesh([("y", 2), ("x", 2)], MeshAxisType::Explicit),
            mesh([("x", 2), ("z", 2)], MeshAxisType::Explicit),
            mesh([("x", 2), ("y", 4)], MeshAxisType::Explicit),
        ] {
            assert_eq!(
                lowering_context.require_mesh(&other),
                Err(Error::Lowering(LoweringError::MeshTopologyMismatch { expected: explicit.clone(), actual: other })),
            );
        }
        assert_eq!(lowering_context.mesh(), Some(explicit));

        // Every lowering scope of a program records into the same module, including the body of a manual region.
        let lowering_context = ProgramLoweringContext::new();
        let boundary = ShardMapBoundary::new(manual.clone(), Vec::new(), Vec::new(), vec!["x".to_string()]).unwrap();
        assert_eq!(lowering_context.enter_manual_region(boundary).require_mesh(&manual), Ok(()));
        assert_eq!(lowering_context.mesh(), Some(manual));
    }

    #[test]
    fn test_program_lowering_context_require_assertion_handler() {
        // The requirement is recorded for the module, so every clone of the context sees it.
        let lowering_context = ProgramLoweringContext::new();
        let clone = lowering_context.clone();
        clone.require_assertion_handler();
        assert!(lowering_context.requires_assertion_handler());
    }

    #[test]
    fn test_program_lowering_context_require_runtime_requirement() {
        // Requirements are recorded in canonical order (i.e., sorted by identifier), and one that is recorded again is
        // recorded once.
        let lowering_context = ProgramLoweringContext::new();
        let second = RuntimeRequirement::new("ryft.tests.second", 1).unwrap();
        let first = RuntimeRequirement::new("ryft.tests.first", 2).unwrap();
        assert_eq!(lowering_context.require_runtime_requirement(second.clone()), Ok(()));
        assert_eq!(lowering_context.require_runtime_requirement(first.clone()), Ok(()));
        assert_eq!(lowering_context.require_runtime_requirement(second.clone()), Ok(()));
        assert_eq!(lowering_context.runtime_requirements(), vec![first.clone(), second.clone()]);

        // A requirement that conflicts with a recorded one (i.e., another ABI version of the same identifier) is
        // rejected, and the recorded requirements stay as they were.
        let conflicting = RuntimeRequirement::new("ryft.tests.first", 3).unwrap();
        assert_eq!(
            lowering_context.require_runtime_requirement(conflicting),
            Err(Error::from(RuntimeError::ConflictingRequirements { identifier: "ryft.tests.first".to_string() })),
        );
        assert_eq!(lowering_context.runtime_requirements(), vec![first, second]);
    }

    #[test]
    fn test_program_lowering_context_allocate_channel_id() {
        // Channel IDs start at one and are allocated for the module, so every clone of the context continues the same
        // sequence.
        let lowering_context = ProgramLoweringContext::new();
        let clone = lowering_context.clone();
        assert_eq!(lowering_context.allocate_channel_id(), 1);
        assert_eq!(clone.allocate_channel_id(), 2);
        assert_eq!(lowering_context.allocate_channel_id(), 3);
    }

    #[test]
    fn test_program_lowering_context_allocate_channel_id_gives_each_collective_of_a_module_its_own_channel() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // Two manual regions in one module each emit a channeled `all_reduce`. The channel allocator of the module
        // hands out distinct handles (i.e., 1 and 2), which XLA requires across a module.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ArrayTracer| {
                    let first = shard_map(
                        |local_x: ArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x"),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap();
                    shard_map(
                        |local_x: ArrayTracer| local_x.parallel_reduce(ReductionKind::Sum, "x"),
                        first,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                        ShardMapOptions::new(),
                    )
                    .unwrap()
                }
            },
            global_input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_rendered_mlir("main", &ProgramLoweringOptions::new()).unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "#},
        );
    }

    #[test]
    fn test_program_lowering_context_enter_manual_region() {
        let mesh = test_mesh().with_axis_types(["x", "y"], MeshAxisType::Manual).unwrap();
        let boundary =
            |axis: &str| ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), vec![axis.to_string()]).unwrap();
        let root = ProgramLoweringContext::new();
        let outer = root.enter_manual_region(boundary("m"));
        let inner = outer.enter_manual_region(boundary("x"));
        let sibling = outer.enter_manual_region(boundary("y"));

        // Each child sees its enclosing boundary and its own bindings.
        assert_eq!(root.manual_shard_map(), None);
        assert!(root.bound_manual_axes().is_empty());
        assert_eq!(outer.bound_manual_axes(), &["m"]);
        assert_eq!(outer.manual_axis_region("x"), None);
        assert_eq!(inner.bound_manual_axes(), &["m", "x"]);
        assert_eq!(inner.manual_axis_region("m"), Some(&boundary("m")));
        assert_eq!(inner.manual_shard_map(), Some(&boundary("x")));
        assert_eq!(inner.manual_axis_region("y"), None);
        assert_eq!(sibling.bound_manual_axes(), &["m", "y"]);
        assert_eq!(sibling.manual_axis_region("x"), None);

        // The same allocator and module records remain visible across both levels of nesting.
        assert_eq!(inner.allocate_channel_id(), 1);
        assert_eq!(sibling.allocate_channel_id(), 2);
        assert_eq!(root.allocate_channel_id(), 3);
        inner.require_mesh(&mesh).unwrap();
        assert_eq!(root.mesh(), Some(mesh));
        let requirement = RuntimeRequirement::new("ryft.tests.nested", 1).unwrap();
        sibling.require_runtime_requirement(requirement.clone()).unwrap();
        assert_eq!(inner.runtime_requirements(), vec![requirement]);
        inner.require_assertion_handler();
        assert!(root.requires_assertion_handler());
        let context = MlirContext::new();
        let provenance = Provenance::scope(ProvenanceScope::new("nested"), Provenance::unknown());
        inner.lower_provenance(&context, &provenance, context.unknown_location().as_ref());
        assert!(sibling.has_provenance());
    }

    #[test]
    fn test_program_lowering_context_enter_manual_region_isolates_session_constants() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let constant = RetainedArray::new(&f32_scalar(&session, &domain_mesh(client, "x", 1), 2.0));
        let root = ProgramLoweringContext::new().with_session_constants(&[constant]);
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        let child = root.enter_manual_region(boundary);
        let captures = [0, 1];
        assert!(child.session_constant_values(&captures).is_empty());
        assert_eq!(root.session_constant_values(&captures), &[1]);
    }

    #[test]
    fn test_program_lowering_context_lower_sharding_constraint() {
        // The constraints are rendered outside any function, so their detached inputs print as unknown SSA values.
        let input_type = test_input_type();
        let sharding = input_type.sharding().unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = input_type.lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut block_ref = block.as_ref();
        let location = location.as_ref();
        let lowering_context = ProgramLoweringContext::new();
        let output = lowering_context
            .lower_sharding_constraint(sharding, input, OpenDimensions::None, &mut block_ref, location)
            .unwrap();
        assert_eq!(lowering_context.mesh(), Some(test_mesh()));

        // An axis that an enclosing manual region binds is left out of the constraint, whose mesh is still recorded.
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        let lowering_context = ProgramLoweringContext::new().enter_manual_region(boundary);
        lowering_context
            .lower_sharding_constraint(sharding, input, OpenDimensions::None, &mut block_ref, location)
            .unwrap();
        assert_eq!(lowering_context.mesh(), Some(test_mesh()));

        // A sharding over a mesh of another topology cannot be lowered into the module, and nothing is emitted for it.
        let other = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_sharding = Sharding::replicated(other.clone(), 2);
        assert_eq!(
            lowering_context.lower_sharding_constraint(
                &other_sharding,
                input,
                OpenDimensions::None,
                &mut block_ref,
                location,
            ),
            Err(Error::Lowering(LoweringError::MeshTopologyMismatch { expected: test_mesh(), actual: other })),
        );
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations[0].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{"x"}, {}], replicated={"m"}> : tensor<4x2xf32>"#,
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{"x"}, {}]> : tensor<4x2xf32>"#,
            ],
        );
    }

    #[test]
    fn test_program_lowering_context_lower_replication_constraint() {
        // The constraints are rendered outside any function, so their detached inputs print as unknown SSA values.
        let input_type = test_input_type();
        let sharding = input_type.sharding().unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = input_type.lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut block_ref = block.as_ref();
        let location = location.as_ref();
        let lowering_context = ProgramLoweringContext::new();
        let output = lowering_context
            .lower_replication_constraint(sharding, input, &[0], &mut block_ref, location)
            .unwrap();
        assert_eq!(lowering_context.mesh(), Some(test_mesh()));

        // An axis that an enclosing manual region binds is left out of the constraint, whose mesh is still recorded.
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        let lowering_context = ProgramLoweringContext::new().enter_manual_region(boundary);
        lowering_context
            .lower_replication_constraint(sharding, input, &[0], &mut block_ref, location)
            .unwrap();
        assert_eq!(lowering_context.mesh(), Some(test_mesh()));
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations[0].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{}, {?}], replicated={"m", "x"}> : tensor<4x2xf32>"#,
                r#"%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{}, {?}], replicated={"x"}> : tensor<4x2xf32>"#,
            ],
        );
    }

    #[test]
    fn test_program_lowering_context_lower_pending_reduction_constraint() {
        // The constraints are rendered outside any function, so their detached inputs print as unknown SSA values.
        let sharding =
            Sharding::new(test_mesh(), vec![ShardingDimension::unconstrained(), ShardingDimension::replicated()])
                .unwrap()
                .with_unreduced_axes(["x"])
                .unwrap();
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = test_matrix_type(4, 2).lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let mut block_ref = block.as_ref();
        let location = location.as_ref();

        // Every dimension is closed, even over a mesh with the auto axis `y`, and an unconstrained one is replicated.
        let lowering_context = ProgramLoweringContext::new();
        let output = lowering_context
            .lower_pending_reduction_constraint(&sharding, input, &mut block_ref, location)
            .unwrap();

        // An axis that an enclosing manual region binds is left out of the constraint.
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        let lowering_context = lowering_context.enter_manual_region(boundary);
        lowering_context
            .lower_pending_reduction_constraint(&sharding, input, &mut block_ref, location)
            .unwrap();
        let operations = block.operations().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(output, operations[0].result(0).unwrap().as_ref());
        assert_eq!(
            operations.iter().map(ToString::to_string).collect::<Vec<_>>(),
            vec![
                "%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{}, {}], replicated={\"m\"}, \
                 unreduced={\"x\"}> : tensor<4x2xf32>",
                "%0 = sdy.sharding_constraint <<UNKNOWN SSA VALUE>> <@mesh, [{}, {}], unreduced={\"x\"}> : \
                 tensor<4x2xf32>",
            ],
        );
    }

    #[test]
    fn test_program_lowering_context_lower_session_constant() {
        // A session constant lowers to the value at its position in the suffix of the capture table.
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let first = RetainedArray::new(&f32_scalar(&session, &mesh, 1.0));
        let second = RetainedArray::new(&f32_scalar(&session, &mesh, 2.0));
        let context = MlirContext::new();
        let location = context.unknown_location();
        let tensor_type = ArrayType::scalar(DataType::F32).lower(&context, location).unwrap();
        let block = context.block(&[(tensor_type, location); 3]);
        let captured_values = [0, 1, 2].map(|index| block.argument(index).unwrap().as_ref());
        let lowering_context = ProgramLoweringContext::new().with_session_constants([&first, &second]);
        assert_eq!(lowering_context.lower_session_constant(&first, &captured_values), Ok(captured_values[1]));
        assert_eq!(lowering_context.lower_session_constant(&second, &captured_values), Ok(captured_values[2]));

        // A constant that the scope does not bind is unreachable, and so is every constant in a scope whose capture
        // table cannot hold the session constants (e.g., a shared private function).
        let unbound = RetainedArray::new(&f32_scalar(&session, &mesh, 3.0));
        assert_eq!(
            lowering_context.lower_session_constant(&unbound, &captured_values),
            Err(Error::Lowering(LoweringError::UnboundSessionConstant { constant: unbound.to_string() })),
        );
        assert_eq!(
            lowering_context.lower_session_constant(&first, &captured_values[..1]),
            Err(Error::Lowering(LoweringError::UnboundSessionConstant { constant: first.to_string() })),
        );
    }

    #[test]
    fn test_program_lowering_context_lower_provenance() {
        // Unknown provenance keeps the base location and records nothing for the module.
        let context = MlirContext::new();
        let base = context.unknown_location().as_ref();
        let lowering_context = ProgramLoweringContext::new();
        let location = lowering_context.lower_provenance(&context, &Provenance::unknown(), base);
        assert_eq!(location.to_string(), "loc(unknown)");
        assert!(!lowering_context.has_provenance());

        // Each scope becomes one named location above its origin, and fused provenance becomes one fused location
        // over its origins. Either one is recorded for the module.
        let scope = |name: &str, origin: Provenance| Provenance::scope(ProvenanceScope::new(name), origin);
        let nested = scope("inner", scope("outer", Provenance::unknown()));
        let location = lowering_context.lower_provenance(&context, &nested, base);
        assert_eq!(location.to_string(), "loc(\"inner\"(\"outer\"))");
        assert!(lowering_context.has_provenance());
        let fused = Provenance::fused([scope("left", Provenance::unknown()), scope("right", Provenance::unknown())]);
        let location = lowering_context.lower_provenance(&context, &fused, base);
        assert_eq!(location.to_string(), "loc(fused[\"left\", \"right\"])");
    }
}
