use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use ryft_core::{FlatProgram, LogicalMesh, Provenance, ShardMapBoundary, Sharding};
use ryft_mlir::{AttributeRef, BlockRef, Context as MlirContext, Location, LocationRef, ValueRef};

use crate::lowering::sharding::{OpenDimensions, ShardingExtension};
use crate::lowering::{LoweringError, RaggedDotLoweringStrategy};
use crate::operations::composites::{NamedCompositionFunction, NamedCompositionFunctionMap, NamedCompositionKey};
use crate::{Error, RetainedArray, RuntimeRequirement, Session, normalize_runtime_requirements};

/// Context for lowering a program in the current scope of its MLIR module. Each instruction lowerer holds
/// a clone of this handle, sharing the module resources and the current scope's bindings through one [`Rc`].
/// [`enter_manual_region`](Self::enter_manual_region) derives a context for the body of an `sdy.manual_computation`.
/// Its manual axes and reachable session constants belong to that scope. The child inherits the module's target and
/// lowering policy, looks up the same named compositions, and continues to allocate channels and record module
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
                next_channel_id: Rc::new(Cell::new(1)),
                named_compositions: Rc::default(),
                mesh: Rc::default(),
                has_provenance: Rc::new(Cell::new(false)),
                requires_assertion_handler: Rc::new(Cell::new(false)),
                runtime_requirements: Rc::new(RefCell::new(Vec::new())),
                manual_regions: Vec::new(),
                bound_manual_axes: Vec::new(),
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

    /// Returns the decomposition at position `index` among those that lowering has requested for this module so far,
    /// in the order of their first requests, or [`None`] when fewer were requested. The module emits them after its
    /// other functions, and it asks by position because lowering one decomposition can request another.
    #[inline]
    pub(super) fn named_composition(&self, index: usize) -> Option<NamedCompositionFunction> {
        self.state.named_compositions.borrow().functions().nth(index).cloned()
    }

    /// Returns the mesh that lowering has recorded for this module so far, which is the mesh that the module declares,
    /// or [`None`] when nothing that lowering has emitted refers to a mesh.
    #[inline]
    pub(super) fn mesh(&self) -> Option<LogicalMesh> {
        self.state.mesh.get().cloned()
    }

    /// Returns whether any scope lowered instruction provenance into an MLIR location.
    #[inline]
    pub(super) fn has_provenance(&self) -> bool {
        self.state.has_provenance.get()
    }

    /// Returns whether this module requires the runtime assertion handler.
    #[inline]
    pub(super) fn requires_assertion_handler(&self) -> bool {
        self.state.requires_assertion_handler.get()
    }

    /// Returns the runtime requirements that lowering has recorded for this module so far.
    #[inline]
    pub(super) fn runtime_requirements(&self) -> Vec<RuntimeRequirement> {
        self.state.runtime_requirements.borrow().clone()
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

    /// Returns the symbol of the private function of this module that holds the decomposition of the named
    /// composition `key`, and records that the module needs that function. The first request for a key traces
    /// `program`, which must not lower anything itself, and every later request shares its function. Whatever lowers
    /// a named composition requests its decomposition here, so the module holds a decomposition exactly when something
    /// that lowering emitted refers to it, rather than when an analysis of the program predicts that.
    ///
    /// # Errors
    ///
    /// Returns the error of `program` when the decomposition cannot be traced.
    pub(crate) fn require_named_composition<F: FnOnce() -> Result<FlatProgram<Session>, Error>>(
        &self,
        key: NamedCompositionKey,
        program: F,
    ) -> Result<Rc<str>, Error> {
        let mut named_compositions = self.state.named_compositions.borrow_mut();
        Ok(named_compositions.require(key, program)?.symbol().into())
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
    /// Returns [`LoweringError::MismatchedMeshTopology`] when the axis names or sizes of `mesh` differ from those
    /// of the recorded mesh, including in order.
    pub(crate) fn require_mesh(&self, mesh: &LogicalMesh) -> Result<(), Error> {
        let recorded = self.state.mesh.get_or_init(|| mesh.clone());
        if recorded.has_same_topology(mesh) {
            Ok(())
        } else {
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology {
                expected: recorded.clone(),
                actual: mesh.clone(),
            }))
        }
    }

    /// Records that this module requires the runtime assertion handler.
    #[inline]
    pub(super) fn require_assertion_handler(&self) {
        self.state.requires_assertion_handler.set(true);
    }

    /// Records `requirement` as a runtime requirement of this module.
    ///
    /// # Errors
    ///
    /// Returns the errors of `normalize_runtime_requirements` when `requirement` conflicts with a requirement
    /// that is already recorded.
    #[inline]
    pub(super) fn require_runtime_requirement(&self, requirement: RuntimeRequirement) -> Result<(), Error> {
        let mut requirements = self.state.runtime_requirements.borrow_mut();
        let mut updated = requirements.clone();
        updated.push(requirement);
        *requirements = normalize_runtime_requirements(updated)?;
        Ok(())
    }

    /// Returns a fresh module-unique channel ID for one channeled collective.
    pub(crate) fn allocate_channel_id(&self) -> usize {
        let channel_id = self.state.next_channel_id.get();
        self.state.next_channel_id.set(channel_id + 1);
        channel_id
    }

    /// Derives a child context whose innermost manual region binds the axes of `boundary`. The body is isolated from
    /// above, so its operations cannot use the enclosing scope's session constant bindings.
    pub(crate) fn enter_manual_region(&self, boundary: ShardMapBoundary) -> Self {
        let mut child = self.clone();
        let state = Rc::make_mut(&mut child.state);
        for axis in boundary.manual_axes() {
            if !state.bound_manual_axes.contains(axis) {
                state.bound_manual_axes.push(axis.clone());
            }
        }
        state.manual_regions.push(Rc::new(boundary));
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
            .ok_or_else(|| {
                Error::Lowering(LoweringError::UnreachableSessionConstant { constant: constant.to_string() })
            })
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
        context: &'c MlirContext<'t>,
        provenance: &Provenance,
        base: LocationRef<'c, 't>,
    ) -> LocationRef<'c, 't> {
        /// Recursively lowers `provenance` above `base` without touching the module-wide flag.
        fn lower<'c, 't: 'c>(
            context: &'c MlirContext<'t>,
            provenance: &Provenance,
            base: LocationRef<'c, 't>,
        ) -> LocationRef<'c, 't> {
            if let Some((scope, origin)) = provenance.as_scope() {
                let child = lower(context, origin, base);
                context.named_location(scope.name(), Some(child)).as_ref()
            } else if let Some(origins) = provenance.as_fused() {
                let locations = origins.iter().map(|origin| lower(context, origin, base)).collect::<Vec<_>>();
                context.fused_location::<_, AttributeRef>(locations.as_slice(), None).as_ref()
            } else {
                base
            }
        }

        if provenance.is_unknown() {
            return base;
        }

        self.state.has_provenance.set(true);

        lower(context, provenance, base)
    }
}

/// State held by a [`ProgramLoweringContext`]. Deriving a scope copies its bindings and retains the [`Rc`]s for
/// resources that all scopes update while emitting the module.
#[derive(Clone)]
struct ProgramLoweringContextState {
    /// PJRT platform name of the compilation target (e.g., `"cuda"` or `"cpu"`), or `None` when the lowering has no
    /// target information. Lowerings that depend on the platform, such as fused attention, consult this and fall back
    /// to their portable form when it is absent.
    target_platform: Option<Rc<str>>,

    /// Backend representation used to lower ragged dots in this module.
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,

    /// Next unused collective channel ID in this module.
    next_channel_id: Rc<Cell<usize>>,

    /// Private decompositions of the named StableHLO composites that the emitted operations of every nested lowering
    /// scope refer to, which the module emits.
    named_compositions: Rc<RefCell<NamedCompositionFunctionMap>>,

    /// First mesh recorded for a sharding in this module. Subsequent shardings must use the same topology.
    mesh: Rc<OnceCell<LogicalMesh>>,

    /// Flag of the module recording whether any recursive lowering path constructed a location specific to an
    /// instruction (i.e., not the base location) from a known [`Provenance`]. It selects the module serialization
    /// mode after lowering completes. Only a module that actually carries provenance locations is printed with debug
    /// information, so modules without provenance keep their existing StableHLO text and cache keys byte for byte.
    has_provenance: Rc<Cell<bool>>,

    /// Whether any emitted instruction requires the runtime assertion handler.
    requires_assertion_handler: Rc<Cell<bool>>,

    /// Runtime contracts declared by emitted operations across every nested lowering scope.
    runtime_requirements: Rc<RefCell<Vec<RuntimeRequirement>>>,

    /// Enclosing manual regions, ordered from outermost to innermost for named-axis resolution.
    manual_regions: Vec<Rc<ShardMapBoundary>>,

    /// Axes bound by any enclosing manual region, excluded from shard-local placement constraints.
    bound_manual_axes: Vec<String>,

    /// Position of each retained array within the session constant suffix of `captured_values`. The suffix holds hidden
    /// entry inputs followed by bufferless placeholders for arrays that the program only passes to callees that never
    /// read them. Entering an `sdy.manual_computation` clears these bindings because its body is isolated.
    session_constant_indices: Rc<HashMap<RetainedArray, usize>>,
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayType, CompositeKey, DataType, Dimension, DomainTracingContext, MeshAxis, MeshAxisType, ParallelReduce,
        ProgramError, ProvenanceScope, ReductionKind, Shape, ShardMapOptions, ShardingDimension, shard_map,
    };
    use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue};

    use crate::lowering::sharding::tests::{test_input_type, test_mesh};
    use crate::lowering::types::ArrayTypeLoweringExtension;
    use crate::sessions::tests::{domain_mesh, f32_scalar};
    use crate::tests::cpu_client;
    use crate::{
        ArrayTracer, ProgramLoweringExtension, ProgramLoweringOptions, Session, TracedProgram, Tracer, Type, trace,
    };

    use super::*;

    #[test]
    fn test_program_lowering_context_require_mesh() {
        let mesh = |axes: [(&str, usize); 2], r#type: MeshAxisType| {
            LogicalMesh::new(axes.map(|(name, size)| MeshAxis::new(name, size, r#type).unwrap()).to_vec()).unwrap()
        };
        let explicit = mesh([("x", 2), ("y", 2)], MeshAxisType::Explicit);
        let lowering_context = ProgramLoweringContext::new();
        assert_eq!(lowering_context.mesh(), None);
        lowering_context.require_mesh(&explicit).unwrap();
        assert_eq!(lowering_context.mesh(), Some(explicit.clone()));

        // A mesh that differs only in axis types shares the module mesh, which keeps the axis types of its first mesh.
        let manual = mesh([("x", 2), ("y", 2)], MeshAxisType::Manual);
        lowering_context.require_mesh(&manual).unwrap();
        assert_eq!(lowering_context.mesh(), Some(explicit.clone()));

        // Device ids follow the order of the axes of the module mesh, so a reordered mesh is another topology, as is a
        // mesh with other axis names or sizes. The module mesh stays the recorded one.
        for other in [
            mesh([("y", 2), ("x", 2)], MeshAxisType::Explicit),
            mesh([("x", 2), ("z", 2)], MeshAxisType::Explicit),
            mesh([("x", 2), ("y", 4)], MeshAxisType::Explicit),
        ] {
            assert!(matches!(
                lowering_context.require_mesh(&other),
                Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                    if expected == explicit && actual == other,
            ));
        }
        assert_eq!(lowering_context.mesh(), Some(explicit));

        // Every lowering scope of a program records into the same module, including the body of a manual region.
        let lowering_context = ProgramLoweringContext::new();
        let boundary = ShardMapBoundary::new(manual.clone(), Vec::new(), Vec::new(), vec!["x".to_string()]).unwrap();
        lowering_context.enter_manual_region(boundary).require_mesh(&manual).unwrap();
        assert_eq!(lowering_context.mesh(), Some(manual));
    }

    #[test]
    fn test_program_lowering_context_require_named_composition() {
        let boundary_type = ArrayType::scalar(DataType::F32);
        let trace = || {
            let input_types = vec![Type::from(boundary_type.clone())];
            let (_, program) = DomainTracingContext::<Session>::trace(|inputs: Vec<Tracer>| Ok(inputs), input_types)?;
            Ok(program.into_simplified()?)
        };
        let key = |name: &'static str| {
            let boundary_types = vec![boundary_type.clone()];
            NamedCompositionKey::new(CompositeKey::new(name, 1), boundary_types.clone(), boundary_types)
        };
        let context = ProgramLoweringContext::new();
        assert!(context.named_composition(0).is_none());
        assert_eq!(context.require_named_composition(key("first"), trace).unwrap().as_ref(), "first");

        // Every lowering scope of a program records into the same module, including the body of a manual region, and a
        // decomposition that was already requested is not traced again.
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        let scope = context.enter_manual_region(boundary);
        assert_eq!(scope.require_named_composition(key("second"), trace).unwrap().as_ref(), "second");
        assert_eq!(scope.require_named_composition(key("first"), || unreachable!()).unwrap().as_ref(), "first");
        let symbols = (0..3)
            .map(|index| context.named_composition(index).map(|function| function.symbol().to_string()))
            .collect::<Vec<_>>();
        assert_eq!(symbols, vec![Some("first".to_string()), Some("second".to_string()), None]);

        // A decomposition that cannot be traced is not recorded.
        let error = ProgramError::MalformedProgram("no decomposition".to_string());
        assert_eq!(
            context.require_named_composition(key("third"), || Err(error.clone().into())),
            Err(Error::from(error)),
        );
        assert!(context.named_composition(2).is_none());
    }

    #[test]
    fn test_program_lowering_context_allocate_channel_id() {
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
        assert!(matches!(
            lowering_context.lower_sharding_constraint(&other_sharding, input, OpenDimensions::None, &mut block_ref, location),
            Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                if expected == test_mesh() && actual == other,
        ));
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
}
