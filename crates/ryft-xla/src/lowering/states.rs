//! State that the lowering of one program shares across the nested lowering scopes of its module.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use ryft_core::{LogicalMesh, Provenance, ShardMapBoundary, Sharding};
use ryft_mlir::{AttributeRef, BlockRef, Context as MlirContext, Location, LocationRef, ValueRef};

use crate::lowering::sharding::{OpenDimensions, ShardingExtension};
use crate::lowering::{LoweringError, RaggedDotLoweringStrategy};
use crate::operations::composites::NamedCompositionFunctionMap;
use crate::{Error, RetainedArray, RuntimeRequirement, normalize_runtime_requirements};

// TODO(eaplatanios): Review this module.

/// Lowering state of one program, which is lowered to one module, shared across the nested lowering scopes of that
/// module. It holds the state of the whole program (i.e., the allocator of the channels of collectives, what lowering
/// records about the module, the registry of named compositions, and the target and representation policy that other
/// lowerings consume) together with the state of the lowering scope that it is used in (i.e., the enclosing
/// `sdy.manual_computation` regions, whose innermost [`ShardMapBoundary`] StableHLO collectives use, and the session
/// constants that the scope can reach).
///
/// The state is created once per lowered module and cloned per instruction lowerer. Every clone shares the state of
/// the program, whose mutable members and registries are shared through [`Rc`], while its immutable target and
/// representation policies are copied or cloned. [`enter_manual_region`](Self::enter_manual_region) keeps the state of
/// the program and derives the state of the nested scope, in which its [`ShardMapBoundary`] is the innermost region.
#[derive(Clone)]
pub(crate) struct ProgramLoweringState {
    /// State of the whole program, which every lowering scope of the program shares.
    shared: SharedLoweringState,

    /// State of the lowering scope that this state is used in.
    scope: LoweringScopeState,
}

impl ProgramLoweringState {
    /// Creates the lowering state for one module, outside any manual region and without target information.
    pub(crate) fn new() -> Self {
        Self {
            shared: SharedLoweringState {
                channel_ids: Rc::new(Cell::new(1)),
                target_platform: None,
                ragged_dot_lowering_strategy: RaggedDotLoweringStrategy::default(),
                named_compositions: None,
                has_provenance: Rc::new(Cell::new(false)),
                has_assertions: Rc::new(Cell::new(false)),
                runtime_requirements: Rc::new(RefCell::new(Vec::new())),
                mesh: Rc::new(RefCell::new(None)),
            },
            scope: LoweringScopeState::default(),
        }
    }

    /// Returns a copy of this state whose scope reaches the hidden entry inputs carrying `session_constants`.
    pub(crate) fn with_session_constants(mut self, session_constants: &[RetainedArray]) -> Self {
        let indices = session_constants.iter().enumerate().map(|(index, constant)| (constant.clone(), index)).collect();
        self.scope.session_constants = Rc::new(indices);
        self
    }

    /// Returns a copy of this state carrying the PJRT platform name of the compilation target.
    pub(crate) fn with_target_platform(mut self, target_platform: Option<&str>) -> Self {
        self.shared.target_platform = target_platform.map(Rc::from);
        self
    }

    /// Returns a copy of this state carrying the [`RaggedDotLoweringStrategy`] of the module.
    pub(crate) fn with_ragged_dot_lowering_strategy(mut self, strategy: RaggedDotLoweringStrategy) -> Self {
        self.shared.ragged_dot_lowering_strategy = strategy;
        self
    }

    /// Returns a copy of this state carrying the registry of the private decompositions of the named compositions of
    /// the module.
    pub(super) fn with_named_compositions(mut self, named_compositions: Rc<NamedCompositionFunctionMap>) -> Self {
        self.shared.named_compositions = Some(named_compositions);
        self
    }

    /// Returns the PJRT platform name of the compilation target, or `None` when the lowering has no target
    /// information.
    pub(crate) fn target_platform(&self) -> Option<&str> {
        self.shared.target_platform.as_deref()
    }

    /// Returns the ragged dot lowering strategy for this module.
    pub(crate) fn ragged_dot_lowering_strategy(&self) -> RaggedDotLoweringStrategy {
        self.shared.ragged_dot_lowering_strategy
    }

    /// Returns the suffix of `captured_values` that holds the values of the session constants reachable in this scope.
    /// A scope whose `captured_values` table cannot hold them (e.g., a shared private function) has no such values.
    pub(crate) fn session_values<'o, V>(&self, captured_values: &'o [V]) -> &'o [V] {
        captured_values
            .len()
            .checked_sub(self.scope.session_constants.len())
            .map_or(&[], |start| &captured_values[start..])
    }

    /// Returns the innermost enclosing manual region's [`ShardMapBoundary`], or `None` outside manual regions.
    pub(crate) fn manual_shard_map(&self) -> Option<&ShardMapBoundary> {
        self.scope.manual_regions.last().map(Rc::as_ref)
    }

    /// Returns the nearest enclosing region binding `axis_name`.
    pub(crate) fn manual_axis_region(&self, axis_name: &str) -> Option<&ShardMapBoundary> {
        self.scope
            .manual_regions
            .iter()
            .rev()
            .find(|region| region.manual_axes().iter().any(|axis| axis == axis_name))
            .map(Rc::as_ref)
    }

    /// Returns `true` if any recursive lowering path constructed a location specific to an instruction from a known
    /// [`Provenance`].
    pub(crate) fn has_provenance(&self) -> bool {
        self.shared.has_provenance.get()
    }

    /// Returns `true` if lowering has emitted an assertion in this module.
    #[inline]
    pub(super) fn has_assertions(&self) -> bool {
        self.shared.has_assertions.get()
    }

    /// Returns the runtime requirements that lowering has recorded for this module so far.
    #[inline]
    pub(super) fn runtime_requirements(&self) -> Vec<RuntimeRequirement> {
        self.shared.runtime_requirements.borrow().clone()
    }

    /// Returns the mesh that lowering has recorded for this module so far (refer to
    /// [`require_mesh`](Self::require_mesh)), which is the mesh that the module declares, or [`None`] when nothing
    /// that lowering has emitted refers to a mesh.
    #[inline]
    pub(super) fn mesh(&self) -> Option<LogicalMesh> {
        self.shared.mesh.borrow().clone()
    }

    /// Records that lowering has emitted an assertion in this module.
    #[inline]
    pub(super) fn record_assertion(&self) {
        self.shared.has_assertions.set(true);
    }

    /// Records `requirement` as a runtime requirement of this module.
    ///
    /// # Errors
    ///
    /// Returns the errors of `normalize_runtime_requirements` when `requirement` conflicts with a requirement that is
    /// already recorded.
    pub(super) fn require_runtime(&self, requirement: RuntimeRequirement) -> Result<(), Error> {
        let mut requirements = self.shared.runtime_requirements.borrow_mut();
        let mut updated = requirements.clone();
        updated.push(requirement);
        *requirements = normalize_runtime_requirements(updated)?;
        Ok(())
    }

    /// Records `mesh` as the mesh that this module declares, which everything that lowering emits over a mesh (i.e.,
    /// the sharding of a signature value, of a constraint, or of the boundary of a manual computation) refers to by
    /// one symbol. Whatever lowers a sharding records its mesh here, so the module declares a mesh exactly when
    /// something that lowering emitted refers to one, rather than when an analysis of the program predicts that. The
    /// module keeps the first mesh that is recorded, including its axis types, because the emitted `sdy.mesh` records
    /// only axis names and sizes.
    ///
    /// # Errors
    ///
    /// Returns [`LoweringError::MismatchedMeshTopology`] when the axis names or sizes of `mesh` differ from those of
    /// the recorded mesh, including in order.
    pub(crate) fn require_mesh(&self, mesh: &LogicalMesh) -> Result<(), Error> {
        let mut recorded = self.shared.mesh.borrow_mut();
        match recorded.as_ref() {
            Some(recorded) if !recorded.has_same_topology(mesh) => {
                Err(Error::Lowering(LoweringError::MismatchedMeshTopology {
                    expected: recorded.clone(),
                    actual: mesh.clone(),
                }))
            }
            Some(_) => Ok(()),
            None => {
                *recorded = Some(mesh.clone());
                Ok(())
            }
        }
    }

    /// Returns the axes bound by every enclosing manual region, excluded from shard-local placement constraints.
    #[inline]
    pub(crate) fn bound_manual_axes(&self) -> &[String] {
        &self.scope.bound_manual_axes
    }

    /// Returns the module-owned private decompositions used by named StableHLO composites.
    #[inline]
    pub(crate) fn named_compositions(&self) -> Option<&Rc<NamedCompositionFunctionMap>> {
        self.shared.named_compositions.as_ref()
    }

    /// Returns a fresh module-unique channel id for one channeled collective.
    pub(crate) fn next_channel_id(&self) -> usize {
        let channel_id = self.shared.channel_ids.get();
        self.shared.channel_ids.set(channel_id + 1);
        channel_id
    }

    /// Derives the state used inside the body of one `sdy.manual_computation` region. The state of the program (i.e.,
    /// the target platform, ragged dot lowering strategy, named compositions, provenance state, and module channel
    /// allocator, among others) is inherited, while the provided [`ShardMapBoundary`] becomes the innermost manual
    /// region of the scope, which cannot reach the session constants of the enclosing scope.
    pub(crate) fn enter_manual_region(&self, boundary: ShardMapBoundary) -> Self {
        Self { shared: self.shared.clone(), scope: self.scope.entering_manual_region(boundary) }
    }

    /// Lowers a tensor placement constraint of `input_value` to `sharding` in this scope and returns the constrained
    /// value. The constraint is that of [`ShardingExtension::lower_sharding_constraint`] without the mesh axes that
    /// the enclosing manual regions bind, and the mesh of `sharding` is recorded as the mesh of the module (refer to
    /// [`require_mesh`](Self::require_mesh)).
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

    /// Lowers a tensor placement constraint that replicates `input_value`, whose tracked placement is `sharding`,
    /// along `axes` in this scope and returns the constrained value. The constraint is that of
    /// [`ShardingExtension::lower_replication_constraint`] without the mesh axes that the enclosing manual regions
    /// bind, and the mesh of `sharding` is recorded as the mesh of the module (refer to
    /// [`require_mesh`](Self::require_mesh)).
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
        let values = self.session_values(captured_values);
        self.scope
            .session_constants
            .get(constant)
            .copied()
            .filter(|_| values.len() == self.scope.session_constants.len())
            .map(|index| values[index])
            .ok_or_else(|| {
                Error::Lowering(LoweringError::UnreachableSessionConstant { constant: constant.to_string() })
            })
    }

    /// Lowers one instruction's [`Provenance`] onto MLIR locations above the provided base location and records
    /// module-wide whether any instruction-specific location was constructed. Unknown provenance uses the base
    /// location unchanged, each scope level becomes one
    /// [`NameLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc) above its lowered origin, and fused
    /// provenance becomes one metadata-free
    /// [`FusedLoc`](https://mlir.llvm.org/docs/Dialects/Builtin/#fusedloc) over its recursively lowered origins. An
    /// existing file/line base location is thereby preserved as the innermost child of the named scopes rather than
    /// being replaced.
    pub(crate) fn instruction_location<'c, 't: 'c>(
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
        self.shared.has_provenance.set(true);
        lower(context, provenance, base)
    }
}

/// State of a whole program in a [`ProgramLoweringState`], which every lowering scope of the program shares. Lowering
/// mutates some of its members while it emits the module, and those are shared through [`Rc`], so that every clone of
/// this state allocates from and records into the same program.
#[derive(Clone)]
struct SharedLoweringState {
    /// Module-scoped counter producing the next collective channel id.
    channel_ids: Rc<Cell<usize>>,

    /// PJRT platform name of the compilation target (e.g., `"cuda"` or `"cpu"`), or `None` when the lowering has no
    /// target information. Lowerings that depend on the platform, such as fused attention, consult this and fall back
    /// to their portable form when it is absent.
    target_platform: Option<Rc<str>>,

    /// Backend representation used to lower ragged dots in this module.
    ragged_dot_lowering_strategy: RaggedDotLoweringStrategy,

    /// Module-owned private decompositions used by named StableHLO composites.
    named_compositions: Option<Rc<NamedCompositionFunctionMap>>,

    /// Flag of the module recording whether any recursive lowering path constructed a location specific to an
    /// instruction (i.e., not the base location) from a known [`Provenance`]. It selects the module serialization mode
    /// after lowering completes. Only a module that actually carries provenance locations is printed with debug
    /// information, so modules without provenance keep their existing StableHLO text and cache keys byte for byte.
    has_provenance: Rc<Cell<bool>>,

    /// Whether any emitted instruction requires the runtime assertion handler.
    has_assertions: Rc<Cell<bool>>,

    /// Runtime contracts declared by emitted operations across every nested lowering scope.
    runtime_requirements: Rc<RefCell<Vec<RuntimeRequirement>>>,

    /// Mesh that the emitted shardings of every nested lowering scope refer to, which the module declares, or [`None`]
    /// while nothing that lowering has emitted refers to a mesh.
    mesh: Rc<RefCell<Option<LogicalMesh>>>,
}

/// State of one lowering scope in a [`ProgramLoweringState`] (e.g., the entry function of the module, or the body of an
/// `sdy.manual_computation` region), which the scopes nested in it derive their own state from.
#[derive(Clone, Default)]
struct LoweringScopeState {
    /// Enclosing manual regions, ordered from outermost to innermost for named-axis resolution.
    manual_regions: Vec<Rc<ShardMapBoundary>>,

    /// Axes bound by every enclosing manual region, excluded from shard-local placement constraints.
    bound_manual_axes: Vec<String>,

    /// Distinct [`RetainedArray`]s of the module, in the order of the hidden entry inputs that carry them (refer to
    /// `collect_session_constants`). Their values form the suffix of every `captured_values` table that can reach the
    /// entry function's inputs. Scopes that cannot reach them (i.e., `sdy.manual_computation` bodies) carry no session
    /// constants. Each one maps to its position among those inputs, so that looking one up costs one hash.
    session_constants: Rc<HashMap<RetainedArray, usize>>,
}

impl LoweringScopeState {
    /// Returns the state of the scope inside the body of an `sdy.manual_computation` region that this scope encloses,
    /// in which `boundary` is the innermost manual region and its manual axes are bound as well.
    fn entering_manual_region(&self, boundary: ShardMapBoundary) -> Self {
        let mut bound_manual_axes = self.bound_manual_axes.clone();
        for axis in boundary.manual_axes() {
            if !bound_manual_axes.contains(axis) {
                bound_manual_axes.push(axis.clone());
            }
        }
        let mut manual_regions = self.manual_regions.clone();
        manual_regions.push(Rc::new(boundary));
        Self {
            manual_regions,
            bound_manual_axes,
            // `sdy.manual_computation` bodies are isolated from above, so they cannot use entry function arguments.
            session_constants: Rc::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayType, DataType, Dimension, MeshAxis, MeshAxisType, ParallelReduce, ReductionKind, Shape, ShardMapOptions,
        ShardingDimension, shard_map,
    };
    use ryft_mlir::{Block, Operation as MlirOperation, Value as MlirValue};

    use crate::lowering::sharding::tests::{test_input_type, test_mesh};
    use crate::lowering::types::ArrayTypeLoweringExtension;
    use crate::{ArrayTracer, ProgramLoweringExtension, ProgramLoweringOptions, TracedProgram, trace};

    use super::*;

    #[test]
    fn test_program_lowering_state_require_mesh() {
        let mesh = |axes: [(&str, usize); 2], r#type: MeshAxisType| {
            LogicalMesh::new(axes.map(|(name, size)| MeshAxis::new(name, size, r#type).unwrap()).to_vec()).unwrap()
        };
        let explicit = mesh([("x", 2), ("y", 2)], MeshAxisType::Explicit);
        let state = ProgramLoweringState::new();
        assert_eq!(state.mesh(), None);
        state.require_mesh(&explicit).unwrap();
        assert_eq!(state.mesh(), Some(explicit.clone()));

        // A mesh that differs only in axis types shares the module mesh, which keeps the axis types of its first mesh.
        let manual = mesh([("x", 2), ("y", 2)], MeshAxisType::Manual);
        state.require_mesh(&manual).unwrap();
        assert_eq!(state.mesh(), Some(explicit.clone()));

        // Device ids follow the order of the axes of the module mesh, so a reordered mesh is another topology, as is a
        // mesh with other axis names or sizes. The module mesh stays the recorded one.
        for other in [
            mesh([("y", 2), ("x", 2)], MeshAxisType::Explicit),
            mesh([("x", 2), ("z", 2)], MeshAxisType::Explicit),
            mesh([("x", 2), ("y", 4)], MeshAxisType::Explicit),
        ] {
            assert!(matches!(
                state.require_mesh(&other),
                Err(Error::Lowering(LoweringError::MismatchedMeshTopology { expected, actual }))
                    if expected == explicit && actual == other,
            ));
        }
        assert_eq!(state.mesh(), Some(explicit));

        // Every lowering scope of a program records into the same module, including the body of a manual region.
        let state = ProgramLoweringState::new();
        let boundary = ShardMapBoundary::new(manual.clone(), Vec::new(), Vec::new(), vec!["x".to_string()]).unwrap();
        state.enter_manual_region(boundary).require_mesh(&manual).unwrap();
        assert_eq!(state.mesh(), Some(manual));
    }

    #[test]
    fn test_program_lowering_state_next_channel_id() {
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
    fn test_program_lowering_state_lower_sharding_constraint() {
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
        let state = ProgramLoweringState::new();
        let output = state
            .lower_sharding_constraint(sharding, input, OpenDimensions::None, &mut block_ref, location)
            .unwrap();
        assert_eq!(state.mesh(), Some(test_mesh()));

        // An axis that an enclosing manual region binds is left out of the constraint, whose mesh is still recorded.
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        let state = ProgramLoweringState::new().enter_manual_region(boundary);
        state
            .lower_sharding_constraint(sharding, input, OpenDimensions::None, &mut block_ref, location)
            .unwrap();
        assert_eq!(state.mesh(), Some(test_mesh()));

        // A sharding over a mesh of another topology cannot be lowered into the module, and nothing is emitted for it.
        let other = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_sharding = Sharding::replicated(other.clone(), 2);
        assert!(matches!(
            state.lower_sharding_constraint(&other_sharding, input, OpenDimensions::None, &mut block_ref, location),
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
    fn test_program_lowering_state_lower_replication_constraint() {
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
        let state = ProgramLoweringState::new();
        let output = state.lower_replication_constraint(sharding, input, &[0], &mut block_ref, location).unwrap();
        assert_eq!(state.mesh(), Some(test_mesh()));

        // An axis that an enclosing manual region binds is left out of the constraint, whose mesh is still recorded.
        let boundary = ShardMapBoundary::new(test_mesh(), Vec::new(), Vec::new(), vec!["m".to_string()]).unwrap();
        let state = ProgramLoweringState::new().enter_manual_region(boundary);
        state.lower_replication_constraint(sharding, input, &[0], &mut block_ref, location).unwrap();
        assert_eq!(state.mesh(), Some(test_mesh()));
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
