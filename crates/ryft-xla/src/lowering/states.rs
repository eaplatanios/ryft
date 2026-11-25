//! Module-scoped state that the lowering of one module shares across its nested regions.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use ryft_core::{
    CollectiveAxes, LogicalMesh, ProgramError, Provenance, SHARD_MAP_OPERATION_NAME, ShardMapBoundary, TypeError,
};
use ryft_mlir::{AttributeRef, Context as MlirContext, Location, LocationRef, ValueRef};

use crate::lowering::{LoweringError, RaggedDotLoweringStrategy};
use crate::operations::composites::NamedCompositionFunctionMap;
use crate::{Error, RetainedArray, RuntimeRequirement, normalize_runtime_requirements};

// TODO(eaplatanios): Review this module.

/// Module-scoped lowering state shared across nested regions. StableHLO collectives use the channel counter and the
/// innermost enclosing `sdy.manual_computation` region's [`ShardMapBoundary`], while other lowerings consume immutable
/// target and representation policy stored alongside that shared state.
///
/// The state is created once per lowered module and cloned per instruction lowerer. Module-owned mutable state and
/// registries are shared through [`Rc`], while immutable target and representation policies are copied or cloned.
/// [`enter_manual_region`](Self::enter_manual_region) preserves that module state and policy while replacing the
/// innermost region's [`ShardMapBoundary`].
#[derive(Clone)]
pub(crate) struct CollectiveLoweringState {
    /// Module-scoped counter producing the next collective channel id.
    channel_ids: Rc<Cell<usize>>,

    /// Enclosing manual regions, ordered from outermost to innermost for named-axis resolution.
    manual_regions: Vec<Rc<ShardMapBoundary>>,

    /// Axes bound by every enclosing manual region, excluded from shard-local placement constraints.
    bound_manual_axes: Vec<String>,

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

    /// Distinct [`RetainedArray`]s of the module, in the order of the hidden entry arguments that carry them (refer
    /// to `collect_session_constants`). Their values form the suffix of every `captured_values` table that can reach
    /// the entry function's arguments. Scopes that cannot reach them (i.e., `sdy.manual_computation` bodies) carry no
    /// session constants. Each one maps to its position among those arguments, so that looking one up costs one hash.
    session_constants: Rc<HashMap<RetainedArray, usize>>,
}

impl CollectiveLoweringState {
    /// Creates the lowering state for one module, outside any manual region and without target information.
    pub(crate) fn new() -> Self {
        Self {
            channel_ids: Rc::new(Cell::new(1)),
            manual_regions: Vec::new(),
            bound_manual_axes: Vec::new(),
            target_platform: None,
            ragged_dot_lowering_strategy: RaggedDotLoweringStrategy::default(),
            named_compositions: None,
            has_provenance: Rc::new(Cell::new(false)),
            has_assertions: Rc::new(Cell::new(false)),
            runtime_requirements: Rc::new(RefCell::new(Vec::new())),
            session_constants: Rc::default(),
        }
    }

    /// Returns a copy of this state whose scope reaches the hidden entry arguments carrying `session_constants`.
    pub(crate) fn with_session_constants(mut self, session_constants: &[RetainedArray]) -> Self {
        let indices = session_constants.iter().enumerate().map(|(index, constant)| (constant.clone(), index)).collect();
        self.session_constants = Rc::new(indices);
        self
    }

    /// Returns a copy of this state carrying the PJRT platform name of the compilation target.
    pub(crate) fn with_target_platform(mut self, target_platform: Option<&str>) -> Self {
        self.target_platform = target_platform.map(Rc::from);
        self
    }

    /// Returns a copy of this state carrying the [`RaggedDotLoweringStrategy`] of the module.
    pub(crate) fn with_ragged_dot_lowering_strategy(mut self, strategy: RaggedDotLoweringStrategy) -> Self {
        self.ragged_dot_lowering_strategy = strategy;
        self
    }

    /// Returns a copy of this state carrying the registry of the private decompositions of the named compositions of
    /// the module.
    pub(super) fn with_named_compositions(mut self, named_compositions: Rc<NamedCompositionFunctionMap>) -> Self {
        self.named_compositions = Some(named_compositions);
        self
    }

    /// Returns the PJRT platform name of the compilation target, or `None` when the lowering has no target
    /// information.
    pub(crate) fn target_platform(&self) -> Option<&str> {
        self.target_platform.as_deref()
    }

    /// Returns the ragged dot lowering strategy for this module.
    pub(crate) fn ragged_dot_lowering_strategy(&self) -> RaggedDotLoweringStrategy {
        self.ragged_dot_lowering_strategy
    }

    /// Returns the suffix of `captured_values` that holds the values of the session constants reachable in this scope.
    /// A scope whose `captured_values` table cannot hold them (e.g., a shared private function) has no such values.
    pub(crate) fn session_values<'o, V>(&self, captured_values: &'o [V]) -> &'o [V] {
        captured_values
            .len()
            .checked_sub(self.session_constants.len())
            .map_or(&[], |start| &captured_values[start..])
    }

    /// Returns the innermost enclosing manual region's [`ShardMapBoundary`], or `None` outside manual regions.
    pub(crate) fn manual_shard_map(&self) -> Option<&ShardMapBoundary> {
        self.manual_regions.last().map(Rc::as_ref)
    }

    /// Returns the nearest enclosing region binding `axis_name`.
    pub(crate) fn manual_axis_region(&self, axis_name: &str) -> Option<&ShardMapBoundary> {
        self.manual_regions
            .iter()
            .rev()
            .find(|region| region.manual_axes().iter().any(|axis| axis == axis_name))
            .map(Rc::as_ref)
    }

    /// Returns `true` if any recursive lowering path constructed a location specific to an instruction from a known
    /// [`Provenance`].
    pub(crate) fn has_provenance(&self) -> bool {
        self.has_provenance.get()
    }

    /// Returns `true` if lowering has emitted an assertion in this module.
    #[inline]
    pub(super) fn has_assertions(&self) -> bool {
        self.has_assertions.get()
    }

    /// Returns the runtime requirements that lowering has recorded for this module so far.
    #[inline]
    pub(super) fn runtime_requirements(&self) -> Vec<RuntimeRequirement> {
        self.runtime_requirements.borrow().clone()
    }

    /// Records that lowering has emitted an assertion in this module.
    #[inline]
    pub(super) fn record_assertion(&self) {
        self.has_assertions.set(true);
    }

    /// Records `requirement` as a runtime requirement of this module.
    ///
    /// # Errors
    ///
    /// Returns the errors of `normalize_runtime_requirements` when `requirement` conflicts with a requirement that is
    /// already recorded.
    pub(super) fn require_runtime(&self, requirement: RuntimeRequirement) -> Result<(), Error> {
        let mut requirements = self.runtime_requirements.borrow_mut();
        let mut updated = requirements.clone();
        updated.push(requirement);
        *requirements = normalize_runtime_requirements(updated)?;
        Ok(())
    }

    /// Returns the axes bound by every enclosing manual region, excluded from shard-local placement constraints.
    #[inline]
    pub(crate) fn bound_manual_axes(&self) -> &[String] {
        &self.bound_manual_axes
    }

    /// Returns the module-owned private decompositions used by named StableHLO composites.
    #[inline]
    pub(crate) fn named_compositions(&self) -> Option<&Rc<NamedCompositionFunctionMap>> {
        self.named_compositions.as_ref()
    }

    /// Returns a fresh module-unique channel id for one channeled collective.
    pub(crate) fn next_channel_id(&self) -> usize {
        let channel_id = self.channel_ids.get();
        self.channel_ids.set(channel_id + 1);
        channel_id
    }

    /// Derives the state used inside the body of one `sdy.manual_computation` region. The target platform, ragged dot
    /// lowering strategy, named compositions, provenance state, and module channel allocator are inherited, while the
    /// provided [`ShardMapBoundary`] becomes the innermost manual region.
    pub(crate) fn enter_manual_region(&self, boundary: ShardMapBoundary) -> Self {
        let mut bound_manual_axes = self.bound_manual_axes.clone();
        for axis in boundary.manual_axes() {
            if !bound_manual_axes.contains(axis) {
                bound_manual_axes.push(axis.clone());
            }
        }
        let mut manual_regions = self.manual_regions.clone();
        manual_regions.push(Rc::new(boundary));
        Self {
            channel_ids: self.channel_ids.clone(),
            manual_regions,
            bound_manual_axes,
            target_platform: self.target_platform.clone(),
            ragged_dot_lowering_strategy: self.ragged_dot_lowering_strategy,
            named_compositions: self.named_compositions.clone(),
            has_provenance: self.has_provenance.clone(),
            has_assertions: self.has_assertions.clone(),
            runtime_requirements: self.runtime_requirements.clone(),
            // `sdy.manual_computation` bodies are isolated from above, so they cannot use entry function arguments.
            session_constants: Rc::default(),
        }
    }

    /// Returns the value of `constant` in a scope whose captured values are `captured_values`.
    pub(crate) fn lower_session_constant<'b, 'c: 'b, 't: 'c>(
        &self,
        constant: &RetainedArray,
        captured_values: &[ValueRef<'b, 'c, 't>],
    ) -> Result<ValueRef<'b, 'c, 't>, Error> {
        let values = self.session_values(captured_values);
        self.session_constants
            .get(constant)
            .copied()
            .filter(|_| values.len() == self.session_constants.len())
            .map(|index| values[index])
            .ok_or_else(|| {
                Error::Lowering(LoweringError::UnreachableSessionConstant { constant: constant.to_string() })
            })
    }

    /// Resolves a collective's actual manual binder and checks any payload mesh against it before communication, and
    /// returns the mesh over which the collective's types are expressed. A payload mesh is accepted when it has the
    /// binder mesh's topology and makes the same axes manual (i.e., when [`LogicalMesh::has_same_manual_axes`] holds),
    /// because `auto_axes` and `explicit_axes` retype only the free axes of a manual region, and it is then the
    /// returned mesh. A payload without a mesh yields the binder mesh and is valid only when the owning operation's
    /// inference accepts that binder. Both meshes share their topology, so either one determines axis sizes and replica
    /// groups.
    pub(crate) fn checked_collective_mesh<'o, A: Into<CollectiveAxes>>(
        &'o self,
        operation_name: &str,
        axis_name: A,
        axis_size: usize,
        payload_mesh: Option<&'o LogicalMesh>,
    ) -> Result<&'o LogicalMesh, Error> {
        // Resolve the binding and recorded size first, preserving the replica-group diagnostics and the nearest
        // enclosing binder when another manual region shadows the axis name. That resolution also requires every named
        // axis to be bound on one mesh.
        let axis_name = axis_name.into();
        self.replica_groups(&axis_name, axis_size, None)?;
        let mesh = self.manual_axis_region(&axis_name.names()[0]).unwrap().mesh();
        match payload_mesh {
            Some(payload_mesh) if payload_mesh.has_same_manual_axes(mesh) => Ok(payload_mesh),
            Some(_) => Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` operation mesh does not match the enclosing manual axis `{axis_name}` mesh",
            ))
            .into()),
            None => Ok(mesh),
        }
    }

    /// Revalidates a collective against the policy of its actual enclosing binder before it emits cross-device
    /// communication. A checked binder passes the mesh that [`Self::checked_collective_mesh`] returns to
    /// `infer_output_types`. That is the payload mesh when the payload records one, because it may retype the free axes
    /// of the binder mesh (e.g., inside `auto_axes`) and the recorded `output_types` are expressed over it. Otherwise
    /// it is the binder mesh, so a mesh-free payload cannot bypass the binder's variation and pending sum contract. An
    /// unchecked binder requires an ordinary mesh-free payload and passes `None`, preserving ordinary collective
    /// inference and adjoints. In either mode, inference must reproduce the recorded `output_types`, and the physical
    /// mesh and participant count remain checked.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if no enclosing manual region binds `axis_name`, if the recorded axis size disagrees with
    /// that region, if a payload mesh differs from that region's mesh in its topology or in the axes it makes manual,
    /// if a tracked payload is used in an unchecked binding, if re-inference fails, or if it produces different output
    /// types.
    pub(crate) fn validate_manual_collective<T: PartialEq, A: Into<CollectiveAxes>>(
        &self,
        operation_name: &str,
        axis_name: A,
        axis_size: usize,
        payload_mesh: Option<&LogicalMesh>,
        output_types: &[T],
        infer_output_types: impl FnOnce(Option<&LogicalMesh>) -> Result<Vec<T>, TypeError>,
    ) -> Result<(), Error> {
        let axis_name = axis_name.into();
        let mesh = self.checked_collective_mesh(operation_name, &axis_name, axis_size, payload_mesh)?;
        let checked = self.manual_axis_region(&axis_name.names()[0]).unwrap().checks_manual_variation();
        if axis_name
            .names()
            .iter()
            .any(|axis_name| self.manual_axis_region(axis_name).unwrap().checks_manual_variation() != checked)
        {
            return Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` axes `{axis_name}` must agree on whether their bindings check manual variation",
            ))
            .into());
        }
        if !checked && payload_mesh.is_some() {
            return Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` cannot use tracked mesh metadata inside an unchecked manual binding",
            ))
            .into());
        }
        if infer_output_types(checked.then_some(mesh)).map_err(ProgramError::from)? != output_types {
            return Err(ProgramError::MalformedProgram(format!(
                "`{operation_name}` output types do not match the enclosing manual mesh contract",
            ))
            .into());
        }
        Ok(())
    }

    /// Resolves the replica groups of the named mesh axes `axis_name` inside the enclosing `shard_map` manual regions,
    /// returning the groups together with the combined axis size. Devices are linearized row-major over the mesh axes,
    /// so the devices whose coordinates differ only along the named axes form one replica group. The group bases
    /// enumerate the coordinate combinations of the other axes, and the group members are ordered by their positions
    /// along the combined axis, which linearizes the named axes in the given order with the first name major (refer to
    /// [`CollectiveAxes`]). Returns an error outside manual regions, for axes that no enclosing region binds as manual
    /// mesh axes, and for axes whose binding regions have meshes with different topologies. The groups depend only on
    /// the names and sizes of the mesh axes in order, so regions whose meshes differ only in axis types (e.g., a nested
    /// region inside `auto_axes` or `explicit_axes`, which retype a free axis) bind axes of one collective.
    pub(crate) fn mesh_axis_replica_groups<A: Into<CollectiveAxes>>(
        &self,
        axis_name: A,
    ) -> Result<(Vec<Vec<usize>>, usize), Error> {
        let axis_names = axis_name.into();
        if axis_names.names().is_empty() {
            return Err(ProgramError::MalformedProgram("collective requires at least one axis name".to_string()).into());
        }
        if self.manual_shard_map().is_none() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "collective over axis `{axis_names}` can only be lowered inside a `{SHARD_MAP_OPERATION_NAME}` \
                     manual region",
                ),
            }
            .into());
        }
        let mut mesh = None;
        for axis_name in axis_names.names() {
            let Some(axis_region) = self.manual_axis_region(axis_name) else {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "collective over axis `{axis_name}` cannot lower inside this `{SHARD_MAP_OPERATION_NAME}` \
                         manual region because the region does not bind that axis as a manual mesh axis",
                    ),
                }
                .into());
            };
            match mesh {
                None => mesh = Some(axis_region.mesh()),
                Some(mesh) if mesh.has_same_topology(axis_region.mesh()) => {}
                Some(_) => {
                    return Err(ProgramError::MalformedProgram(format!(
                        "collective over axes `{axis_names}` spans manual regions whose meshes have different \
                         topologies",
                    ))
                    .into());
                }
            }
        }
        let mesh = mesh.unwrap();
        let axis_sizes: Vec<usize> = mesh.axes().iter().map(|axis| axis.size()).collect();
        let strides = (0..axis_sizes.len()).map(|axis| axis_sizes[axis + 1..].iter().product()).collect::<Vec<usize>>();
        let named_axes = axis_names.names().iter().map(|name| mesh.axis_index(name).unwrap()).collect::<Vec<_>>();
        let axis_size: usize = named_axes.iter().map(|axis| axis_sizes[*axis]).product();
        let device_count: usize = axis_sizes.iter().product();
        let mut replica_groups: Vec<Vec<usize>> = Vec::with_capacity(device_count / axis_size);
        for device in 0..device_count {
            // A device is a group base exactly when its coordinates along the named axes are zero.
            if named_axes.iter().all(|axis| (device / strides[*axis]) % axis_sizes[*axis] == 0) {
                replica_groups.push(
                    (0..axis_size)
                        .map(|mut position| {
                            let mut member = device;
                            for axis in named_axes.iter().rev() {
                                member += (position % axis_sizes[*axis]) * strides[*axis];
                                position /= axis_sizes[*axis];
                            }
                            member
                        })
                        .collect(),
                );
            }
        }
        Ok((replica_groups, axis_size))
    }

    /// Resolves the physical replica groups for one collective, expanding each logical subgroup within every fixed
    /// coordinate of the enclosing mesh's other axes.
    pub(crate) fn replica_groups<A: Into<CollectiveAxes>>(
        &self,
        axis_name: A,
        axis_size: usize,
        axis_index_groups: Option<&[Vec<usize>]>,
    ) -> Result<(Vec<Vec<usize>>, usize), Error> {
        let axis_name = axis_name.into();
        let (mesh_groups, mesh_axis_size) = self.mesh_axis_replica_groups(&axis_name)?;
        if axis_size != mesh_axis_size {
            return Err(ProgramError::MalformedProgram(format!(
                "collective over axis `{axis_name}` records size {axis_size}, but the enclosing mesh axis has size \
                 {mesh_axis_size}",
            ))
            .into());
        }
        let Some(axis_index_groups) = axis_index_groups else {
            return Ok((mesh_groups, mesh_axis_size));
        };
        let group_size = axis_index_groups.first().map(Vec::len).unwrap_or(0);
        let mut replica_groups = Vec::with_capacity(mesh_groups.len() * axis_index_groups.len());
        for mesh_group in mesh_groups {
            for axis_index_group in axis_index_groups {
                replica_groups.push(
                    axis_index_group
                        .iter()
                        .map(|&index| {
                            mesh_group.get(index).copied().ok_or_else(|| {
                                ProgramError::MalformedProgram(format!(
                                    "collective over axis `{axis_name}` has group member {index} outside axis size \
                                     {mesh_axis_size}",
                                ))
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                );
            }
        }
        Ok((replica_groups, group_size))
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
        self.has_provenance.set(true);
        lower(context, provenance, base)
    }
}

#[cfg(test)]
mod tests {

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        ArrayType, DataType, Dimension, Error as CoreError, LogicalMesh, MeshAxis, MeshAxisType, ParallelReduce,
        ProgramError, ReductionKind, Shape, ShardMapBoundary, ShardMapOptions, Sharding, ShardingDimension, shard_map,
    };

    use crate::{ArrayTracer, Error, ProgramExtension, TracedProgram, trace};

    use crate::lowering::tests::test_manual_mesh;

    use super::*;

    #[test]
    fn test_collective_lowering_state_next_channel_id() {
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
            traced.to_mlir_module("main").unwrap(),
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
    fn test_collective_lowering_state_checked_collective_mesh() {
        let outer_mesh = test_manual_mesh("x", 2);
        let inner_mesh = test_manual_mesh("x", 3);
        let outer = ShardMapBoundary::new(outer_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let inner = ShardMapBoundary::new(inner_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let state = CollectiveLoweringState::new().enter_manual_region(outer).enter_manual_region(inner);
        assert_eq!(state.checked_collective_mesh("parallel_permute", "x", 3, None).unwrap(), &inner_mesh);
        assert!(matches!(
            state.checked_collective_mesh("parallel_permute", "x", 3, Some(&outer_mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_permute` operation mesh does not match the enclosing manual axis `x` mesh",
            ),
        ));
        assert!(matches!(
            state.checked_collective_mesh("parallel_permute", "x", 2, Some(&outer_mesh)),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "collective over axis `x` records size 2, but the enclosing mesh axis has size 3",
            ),
        ));

        // A payload mesh that retypes only a free axis of the binder mesh (e.g., inside `auto_axes`) is accepted and
        // returned, while a payload mesh that retypes a manual axis or has another topology is still rejected.
        let binder_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let binder = ShardMapBoundary::new(binder_mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let state = CollectiveLoweringState::new().enter_manual_region(binder);
        let free_axis_retyped_mesh = binder_mesh.with_axis_types(["y"], MeshAxisType::Auto).unwrap();
        let manual_axis_retyped_mesh = binder_mesh.with_axis_types(["x"], MeshAxisType::Explicit).unwrap();
        let other_topology_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        assert_eq!(state.checked_collective_mesh("parallel_permute", "x", 2, None).unwrap(), &binder_mesh);
        assert_eq!(
            state.checked_collective_mesh("parallel_permute", "x", 2, Some(&free_axis_retyped_mesh)).unwrap(),
            &free_axis_retyped_mesh,
        );
        for payload_mesh in [&manual_axis_retyped_mesh, &other_topology_mesh] {
            assert!(matches!(
                state.checked_collective_mesh("parallel_permute", "x", 2, Some(payload_mesh)),
                Err(Error::Core(core_error)) if matches!(
                    core_error.as_ref(),
                    CoreError::Program(ProgramError::MalformedProgram(message))
                        if *message == "`parallel_permute` operation mesh does not match the enclosing manual axis `x` mesh",
                ),
            ));
        }
    }

    #[test]
    fn test_collective_lowering_state_validate_manual_collective_uses_owner_policy() {
        let mesh = test_manual_mesh("x", 2);
        let boundary = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new()).unwrap();
        let checked = CollectiveLoweringState::new().enter_manual_region(boundary.clone());
        checked
            .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, Some(&mesh));
                Ok(vec![1])
            })
            .unwrap();
        let unchecked = CollectiveLoweringState::new().enter_manual_region(boundary.with_check_manual_variation(false));
        unchecked
            .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, None);
                Ok(vec![1])
            })
            .unwrap();
    }

    #[test]
    fn test_collective_lowering_state_validate_manual_collective_uses_each_axis_owner() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let outer = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), vec!["x".to_string()]).unwrap();
        let inner = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), vec!["y".to_string()]).unwrap();
        let state = CollectiveLoweringState::new()
            .enter_manual_region(outer.clone())
            .enter_manual_region(inner.clone().with_check_manual_variation(false));
        state
            .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, Some(&mesh));
                Ok(vec![1])
            })
            .unwrap();
        state
            .validate_manual_collective("parallel_permute", "y", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, None);
                Ok(vec![1])
            })
            .unwrap();

        let state = CollectiveLoweringState::new()
            .enter_manual_region(outer.with_check_manual_variation(false))
            .enter_manual_region(inner);
        state
            .validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, None);
                Ok(vec![1])
            })
            .unwrap();
        state
            .validate_manual_collective("parallel_permute", "y", 2, None, &[1usize], |inference_mesh| {
                assert_eq!(inference_mesh, Some(&mesh));
                Ok(vec![1])
            })
            .unwrap();
    }

    #[test]
    fn test_collective_lowering_state_validate_manual_collective_rejects_tracked_unchecked_payload() {
        let mesh = test_manual_mesh("x", 2);
        let boundary = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new())
            .unwrap()
            .with_check_manual_variation(false);
        let state = CollectiveLoweringState::new().enter_manual_region(boundary);
        let result = state.validate_manual_collective("parallel_permute", "x", 2, Some(&mesh), &[1usize], |_| {
            panic!("tracked payload must be rejected before ordinary inference");
        });
        assert!(matches!(
            result,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_permute` cannot use tracked mesh metadata inside an unchecked manual binding",
            ),
        ));
    }

    #[test]
    fn test_collective_lowering_state_validate_manual_collective_retains_unchecked_validation() {
        let mesh = test_manual_mesh("x", 2);
        let boundary = ShardMapBoundary::new(mesh.clone(), Vec::new(), Vec::new(), Vec::new())
            .unwrap()
            .with_check_manual_variation(false);
        let state = CollectiveLoweringState::new().enter_manual_region(boundary);
        let wrong_size = state.validate_manual_collective("parallel_permute", "x", 3, None, &[1usize], |_| {
            panic!("axis size must be validated before ordinary inference");
        });
        assert!(matches!(
            wrong_size,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "collective over axis `x` records size 3, but the enclosing mesh axis has size 2",
            ),
        ));

        let other_mesh = test_manual_mesh("x", 3);
        let wrong_mesh =
            state.validate_manual_collective("parallel_permute", "x", 2, Some(&other_mesh), &[1usize], |_| {
                panic!("payload mesh must be validated before ordinary inference");
            });
        assert!(matches!(
            wrong_mesh,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_permute` operation mesh does not match the enclosing manual axis `x` mesh",
            ),
        ));

        let wrong_outputs =
            state.validate_manual_collective("parallel_permute", "x", 2, None, &[1usize], |_| Ok(vec![2usize]));
        assert!(matches!(
            wrong_outputs,
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if *message == "`parallel_permute` output types do not match the enclosing manual mesh contract",
            ),
        ));
    }

    #[test]
    fn test_collective_lowering_state_mesh_axis_replica_groups() {
        // Devices are linearized row-major, so device `3 * x + y` has coordinates `(x, y)`.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let state = CollectiveLoweringState::new()
            .enter_manual_region(ShardMapBoundary::new(mesh, Vec::new(), Vec::new(), Vec::new()).unwrap());
        assert_eq!(state.mesh_axis_replica_groups("x").unwrap(), (vec![vec![0, 3], vec![1, 4], vec![2, 5]], 2));
        assert_eq!(state.mesh_axis_replica_groups("y").unwrap(), (vec![vec![0, 1, 2], vec![3, 4, 5]], 3));

        // The first name of a combined axis is major.
        assert_eq!(state.mesh_axis_replica_groups(["x", "y"]).unwrap(), (vec![vec![0, 1, 2, 3, 4, 5]], 6));
        assert_eq!(state.mesh_axis_replica_groups(["y", "x"]).unwrap(), (vec![vec![0, 3, 1, 4, 2, 5]], 6));
    }

    #[test]
    fn test_collective_lowering_state_mesh_axis_replica_groups_rejects_unresolvable_axes() {
        let outer_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let inner_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let state = CollectiveLoweringState::new()
            .enter_manual_region(ShardMapBoundary::new(outer_mesh, Vec::new(), Vec::new(), Vec::new()).unwrap())
            .enter_manual_region(ShardMapBoundary::new(inner_mesh, Vec::new(), Vec::new(), Vec::new()).unwrap());
        assert!(matches!(
            state.mesh_axis_replica_groups(Vec::<String>::new()),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if message == "collective requires at least one axis name",
            ),
        ));
        assert!(matches!(
            state.mesh_axis_replica_groups(["x", "y"]),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if message
                        == "collective over axes `(x, y)` spans manual regions whose meshes have different topologies",
            ),
        ));
        assert!(matches!(
            state.mesh_axis_replica_groups("z"),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::UnsupportedOperation { message })
                    if message
                        == "collective over axis `z` cannot lower inside this `shard_map` manual region because the \
                            region does not bind that axis as a manual mesh axis",
            ),
        ));
        assert!(matches!(
            CollectiveLoweringState::new().mesh_axis_replica_groups("x"),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::UnsupportedOperation { message })
                    if message == "collective over axis `x` can only be lowered inside a `shard_map` manual region",
            ),
        ));
    }

    #[test]
    fn test_collective_lowering_state_replica_groups() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 4, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let state = CollectiveLoweringState::new()
            .enter_manual_region(ShardMapBoundary::new(mesh, Vec::new(), Vec::new(), Vec::new()).unwrap());
        assert_eq!(state.replica_groups("y", 4, None).unwrap(), (vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]], 4),);

        // Logical participant groups expand within every fixed coordinate of the other axes, in order.
        assert_eq!(
            state.replica_groups("y", 4, Some(&[vec![0, 2], vec![3, 1]])).unwrap(),
            (vec![vec![0, 2], vec![3, 1], vec![4, 6], vec![7, 5]], 2),
        );
        assert!(matches!(
            state.replica_groups("y", 2, None),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if message == "collective over axis `y` records size 2, but the enclosing mesh axis has size 4",
            ),
        ));
        assert!(matches!(
            state.replica_groups("y", 4, Some(&[vec![0, 4]])),
            Err(Error::Core(core_error)) if matches!(
                core_error.as_ref(),
                CoreError::Program(ProgramError::MalformedProgram(message))
                    if message == "collective over axis `y` has group member 4 outside axis size 4",
            ),
        ));
    }
}
