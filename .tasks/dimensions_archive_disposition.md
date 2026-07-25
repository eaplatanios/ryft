# Symbolic dimensions archive disposition

This is the exhaustive disposition of the 142 paths changed by immutable archive
`770e77d001547c72150a44843c170ea6417ab41e` relative to its parent. A path appears exactly once.

Disposition codes:

- `Extracted`: an independent cleanup has already landed or is explicitly scheduled independently.
- `Behavior Pn`: retain tests, diagnostics, semantics, or implementation knowledge as input to phase `Pn`; do not
  assume the archived structure survives.
- `Superseded Pn`: the archived implementation shape is rejected and replaced in phase `Pn`.
- `Delete`: deliberately omit unrelated or obsolete work.

Several rows contain both `Behavior` and `Superseded`: their behavior is an acceptance source while their architecture
must not be copied.

| Archived path | Disposition |
|---|---|
| `.tasks/plan_symbolic_dimensions_architecture_cleanup.md` | Extracted S0; this reviewed plan and the P0 evidence documents supersede the archived revision |
| `AGENTS.md` | Extracted S0 conventions; no archive replay |
| `Cargo.lock` | Behavior P7 only for dependencies required by the accepted XLA implementation; regenerate rather than copy |
| `crates/ryft-core/src/axes.rs` | Extracted S1 surface cleanup; Behavior P5 for batching-axis cases |
| `crates/ryft-core/src/backends/array_primitives.rs` | Behavior P2/P8 family inventory; Superseded P2 by generic projection/lift generation |
| `crates/ryft-core/src/backends/array_programs/batching.rs` | Behavior P5; Superseded P2/P3/P5 special-purpose context and recovery adapter |
| `crates/ryft-core/src/backends/array_programs/differentiation.rs` | Behavior P6; Superseded P2/P3/P6 special-purpose context and witness adapter |
| `crates/ryft-core/src/backends/array_programs/mod.rs` | Behavior P2/P3/P4; Superseded P2 composite projection/view architecture |
| `crates/ryft-core/src/backends/arrays.rs` | Behavior P2/P3 for reference eager values and operation rules; Superseded P3 dual contracts |
| `crates/ryft-core/src/backends/dimensions.rs` | Behavior P2/P4 for host dimension arithmetic, diagnostics, and prover; Superseded P2 hand-written outer adapters |
| `crates/ryft-core/src/backends/mod.rs` | Behavior P2/P9 exports; rewrite to final module ownership |
| `crates/ryft-core/src/backends/runtime_dimensions.rs` | Behavior P7 for materialization tiers; Superseded P3/P7 witness/reconstruction API |
| `crates/ryft-core/src/backends/scalars.rs` | Behavior P2/P4 gateway and scalar cases; avoid archive-wide generic propagation |
| `crates/ryft-core/src/batching.rs` | Behavior P5 generic transform contracts; Superseded P5 dimension-specific generic hooks |
| `crates/ryft-core/src/broadcasting.rs` | Behavior P3 broadcasting contract; centralize explicit extent operands |
| `crates/ryft-core/src/captures.rs` | Behavior P1/P2 capture identity cases; Superseded by generic structural identity/member projection |
| `crates/ryft-core/src/compilation/contexts.rs` | Behavior P2/P7 context integration; remove composite-specific bounds |
| `crates/ryft-core/src/compilation/function.rs` | Behavior P1/P7 canonical-signature/cache cases; Superseded expression-era cache machinery |
| `crates/ryft-core/src/contexts.rs` | Behavior P2 generic projected binding; Superseded member-specific context APIs |
| `crates/ryft-core/src/differentiation/elementwise.rs` | Behavior P6 array rules; no dimension-specific generic contract |
| `crates/ryft-core/src/differentiation/forward.rs` | Behavior P6 structural-value traversal; Superseded composite special cases |
| `crates/ryft-core/src/differentiation/hessian.rs` | Behavior P6 public transform compatibility; no archive replay |
| `crates/ryft-core/src/differentiation/jacobian.rs` | Behavior P6 public transform compatibility; no archive replay |
| `crates/ryft-core/src/differentiation/mod.rs` | Behavior P6 exports/contracts; rewrite around structural dimensions |
| `crates/ryft-core/src/differentiation/reverse.rs` | Behavior P6 transpose/residual cases; Superseded witness recovery |
| `crates/ryft-core/src/differentiation/types.rs` | Behavior P6 tangent typing; dimension values remain structural/nondifferentiable |
| `crates/ryft-core/src/interpretation.rs` | Behavior P2/P4 eager/PE diagnostics; Superseded symbolic replay/reconstruction paths |
| `crates/ryft-core/src/lib.rs` | Behavior P9 public exports; rebuild from final ownership |
| `crates/ryft-core/src/macros.rs` | Behavior P8 test helpers; Superseded concrete array-program dispatch helpers |
| `crates/ryft-core/src/operations/attention.rs` | Behavior P2/P8 homogeneous rules and tests |
| `crates/ryft-core/src/operations/collectives.rs` | Behavior P3/P5/P6/P7 shape-changing collective contracts |
| `crates/ryft-core/src/operations/compare.rs` | Behavior P2/P8 homogeneous compare and mixed dimension-compare tests |
| `crates/ryft-core/src/operations/complex.rs` | Behavior P2/P8 homogeneous rules |
| `crates/ryft-core/src/operations/constants/constant.rs` | Behavior P2/P8 homogeneous constant |
| `crates/ryft-core/src/operations/constants/fill.rs` | Behavior P3 static/dynamic constructor split; Superseded overlapping composite implementation |
| `crates/ryft-core/src/operations/constants/iota.rs` | Behavior P3 static/dynamic constructor split; Superseded overlapping composite implementation |
| `crates/ryft-core/src/operations/constants/mod.rs` | Behavior P3 constructor tests; Superseded `runtime_dimension_variables` helper |
| `crates/ryft-core/src/operations/constants/one.rs` | Behavior P3 structural/one-like rules; Superseded overlapping composite implementation |
| `crates/ryft-core/src/operations/constants/zero.rs` | Behavior P3 structural/zero-like rules; Superseded overlapping composite implementation |
| `crates/ryft-core/src/operations/control_flow/condition.rs` | Behavior P1/P4/P5/P6/P7 mixed-region carries and closure |
| `crates/ryft-core/src/operations/control_flow/mod.rs` | Behavior P4/P8 region operation exports |
| `crates/ryft-core/src/operations/control_flow/scan.rs` | Behavior P1/P4/P5/P6/P7 mixed carries and stacked outputs |
| `crates/ryft-core/src/operations/control_flow/select.rs` | Behavior P8 benign homogeneous parameterization |
| `crates/ryft-core/src/operations/control_flow/while.rs` | Behavior P1/P4/P5/P6/P7 mixed loop carries |
| `crates/ryft-core/src/operations/custom_call.rs` | Behavior P3/P7 dynamic output schema; Superseded dual contract and collector |
| `crates/ryft-core/src/operations/differentiation/coordinate_basis.rs` | Behavior P6 deletion rationale; delete obsolete materialized coordinate-basis operation |
| `crates/ryft-core/src/operations/differentiation/mod.rs` | Behavior P6 exports after coordinate-basis deletion |
| `crates/ryft-core/src/operations/differentiation/stop_gradient.rs` | Behavior P8 benign homogeneous parameterization |
| `crates/ryft-core/src/operations/manipulation/broadcasting.rs` | Behavior P3/P5/P6/P7; Superseded dual contract, collector, and recovery |
| `crates/ryft-core/src/operations/manipulation/concatenation.rs` | Behavior P3/P5/P6/P7; Superseded dual contract and extent recovery |
| `crates/ryft-core/src/operations/manipulation/conversion.rs` | Behavior P2/P7 explicit data/dimension gateways |
| `crates/ryft-core/src/operations/manipulation/dimension_size.rs` | Behavior P2/P3/P7 array-to-dimension operation; Superseded dual result contract |
| `crates/ryft-core/src/operations/manipulation/gathering.rs` | Behavior P3/P5/P6/P7; Superseded dual contract and collector |
| `crates/ryft-core/src/operations/manipulation/mod.rs` | Behavior P2/P3/P9 exports |
| `crates/ryft-core/src/operations/manipulation/padding.rs` | Behavior P3/P5/P6/P7; Superseded dual contract and collector |
| `crates/ryft-core/src/operations/manipulation/reshaping.rs` | Behavior P3/P5/P6/P7; Superseded dual contract, collector, and `transpose_dimension_variables` |
| `crates/ryft-core/src/operations/manipulation/scattering.rs` | Behavior P2/P6/P8 homogeneous scatter/update rules |
| `crates/ryft-core/src/operations/manipulation/slicing.rs` | Behavior P3/P5/P6/P7; Superseded slice-family dual contracts and collectors |
| `crates/ryft-core/src/operations/manipulation/transposition.rs` | Behavior P2/P6/P8 homogeneous transpose |
| `crates/ryft-core/src/operations/math/abs.rs` | Behavior P2/P8 homogeneous rule |
| `crates/ryft-core/src/operations/math/add.rs` | Behavior P2/P8 homogeneous rule |
| `crates/ryft-core/src/operations/math/div.rs` | Behavior P2/P8 homogeneous rule |
| `crates/ryft-core/src/operations/math/dot.rs` | Behavior P2/P5/P6/P8 homogeneous rule |
| `crates/ryft-core/src/operations/math/mod.rs` | Behavior P2/P8 operation exports |
| `crates/ryft-core/src/operations/math/mul.rs` | Behavior P2/P8 homogeneous rule |
| `crates/ryft-core/src/operations/math/neg.rs` | Behavior P2/P8 homogeneous rule |
| `crates/ryft-core/src/operations/math/reduce.rs` | Behavior P3/P5/P6/P7; Superseded dual contract, collector, and recovery |
| `crates/ryft-core/src/operations/math/sign.rs` | Behavior P2/P8 homogeneous rule |
| `crates/ryft-core/src/operations/math/sub.rs` | Behavior P2/P8 homogeneous rule |
| `crates/ryft-core/src/operations/memory.rs` | Behavior P2/P8 homogeneous memory operation |
| `crates/ryft-core/src/operations/mod.rs` | Behavior P8 outer operation dispatch and exports |
| `crates/ryft-core/src/operations/random.rs` | Behavior P3/P5/P7 RNG shape contract; Superseded dual contract and collector |
| `crates/ryft-core/src/operations/sharding.rs` | Behavior P2/P5/P8 homogeneous sharding rules |
| `crates/ryft-core/src/operations/sort.rs` | Behavior P2/P8 homogeneous rule |
| `crates/ryft-core/src/programs/builders.rs` | Behavior P1/P2 structural closure and projected binding; Superseded member-specific hooks |
| `crates/ryft-core/src/programs/effects.rs` | Behavior P2/P4 ordered assertion semantics and ordering |
| `crates/ryft-core/src/programs/identities.rs` | Behavior P1 structural identity algorithms; Superseded operation-owned identity roles |
| `crates/ryft-core/src/programs/mod.rs` | Behavior P1/P2/P9 program exports |
| `crates/ryft-core/src/programs/operations.rs` | Behavior P1/P8 operation identity and dispatch; Superseded `OutputIdentityRole` |
| `crates/ryft-core/src/programs/programs.rs` | Behavior P1/P2/P4 import, closure, PE, and canonical signatures |
| `crates/ryft-core/src/programs/regions.rs` | Behavior P1 mixed-region closure and identity diagnostics |
| `crates/ryft-core/src/programs/types.rs` | Behavior P1 minimal `Type::Identity`/`Refinements`; Superseded broader dimension-specific hooks |
| `crates/ryft-core/src/programs/values.rs` | Behavior P2 storage-boundary projection; Superseded member extraction side channels |
| `crates/ryft-core/src/sharding/mod.rs` | Behavior P5/P7 dynamic sharding type and lowering cases |
| `crates/ryft-core/src/tracing.rs` | Behavior P1/P2 trace identity/member projection |
| `crates/ryft-core/src/tracing_v2/benchmarking.rs` | Extracted P0 one-line compatibility correction; Behavior P10 graph-size benchmark |
| `crates/ryft-core/src/tracing_v2/custom_derivatives.rs` | Extracted earlier module move; Behavior P6 mixed-region custom derivative cases |
| `crates/ryft-core/src/tracing_v2/rematerialization.rs` | Behavior P6 region-polymorphic rematerialization |
| `crates/ryft-core/src/types/array_types.rs` | Extracted S5a rename to `types/arrays.rs`; Behavior P1 leaf identity/bounds semantics |
| `crates/ryft-core/src/types/data_types.rs` | Extracted S5a rename to `types/data.rs`; no dimension architecture replay |
| `crates/ryft-core/src/types/dimensions.rs` | Behavior P1 leaf identity/bounds/refinements; Superseded expression/witness-era surface |
| `crates/ryft-core/src/types/mod.rs` | Behavior P1/P9 canonical type exports |
| `crates/ryft-core/tests/dimension_allocation_counts.rs` | Behavior P0/P1/P2/P10 allocation baseline and acceptance tests |
| `crates/ryft-core/tests/first_class_broadcast_gather.rs` | Behavior P3/P5/P7 end-to-end broadcast/gather acceptance |
| `crates/ryft-macros-tests/tests/operations/error_missing_type.stderr` | Behavior P8 exact derive diagnostic |
| `crates/ryft-macros-tests/tests/operations/error_type_attribute.stderr` | Behavior P8 exact derive diagnostic |
| `crates/ryft-macros-tests/tests/parameters/error_structs.stderr` | Behavior P8 exact derive diagnostic |
| `crates/ryft-macros-tests/tests/test_operations.rs` | Behavior P1/P8 derive integration coverage |
| `crates/ryft-macros/src/operations.rs` | Behavior P1/P2/P8 generated identity/projection/lift dispatch; Superseded concrete-family branching |
| `crates/ryft-pjrt/src/extensions/ffi/futures.rs` | Delete unrelated PJRT async FFI work from this migration |
| `crates/ryft-pjrt/src/extensions/ffi/handlers.rs` | Delete unrelated PJRT handler FFI work from this migration |
| `crates/ryft-xla/Cargo.toml` | Behavior P7/P10 only for accepted lowering/assertion/benchmark dependencies |
| `crates/ryft-xla/src/arrays.rs` | Behavior P7 dynamic-shape value/ABI integration |
| `crates/ryft-xla/src/arrays_v0/array.rs` | Behavior P7 bounded dynamic array metadata |
| `crates/ryft-xla/src/arrays_v0/compiled_reshard.rs` | Behavior P7 dynamic-shape reshard compatibility |
| `crates/ryft-xla/src/arrays_v0/device_put.rs` | Behavior P7 bounded-input materialization |
| `crates/ryft-xla/src/arrays_v0/execution.rs` | Behavior P7 hidden-extent ABI and result materialization |
| `crates/ryft-xla/src/arrays_v0/host.rs` | Behavior P7 host transfer of bounded dynamic arrays |
| `crates/ryft-xla/src/arrays_v0/tests.rs` | Behavior P7/P10 end-to-end dynamic execution tests |
| `crates/ryft-xla/src/bin/compilation_benchmark.rs` | Behavior P0/P10 compilation/runtime baseline harness |
| `crates/ryft-xla/src/eager.rs` | Behavior P2/P7 host dimension eager math and array execution; Superseded projected view |
| `crates/ryft-xla/src/experimental/assertions.rs` | Behavior P4/P7 ordered assertion diagnostics and observed values |
| `crates/ryft-xla/src/experimental/benchmark_support.rs` | Behavior P0/P10 graph and compilation benchmark cases |
| `crates/ryft-xla/src/experimental/domains.rs` | Behavior P2/P7 domain integration; Superseded `ArrayContextView::bind_replayed` |
| `crates/ryft-xla/src/experimental/lowering.rs` | Behavior P7 from deleted monolith only; do not restore this file |
| `crates/ryft-xla/src/experimental/lowering/composite.rs` | Behavior P7 composite storage-boundary lowering; Superseded projected-view assumptions |
| `crates/ryft-xla/src/experimental/lowering/dispatch.rs` | Behavior P7 operation dispatch; align with P8 generated families |
| `crates/ryft-xla/src/experimental/lowering/mod.rs` | Behavior P7 accepted lowering module ownership |
| `crates/ryft-xla/src/experimental/lowering/operations/attention.rs` | Behavior P7 attention lowering |
| `crates/ryft-xla/src/experimental/lowering/operations/collectives.rs` | Behavior P7 explicit dynamic collective operands |
| `crates/ryft-xla/src/experimental/lowering/operations/constants.rs` | Behavior P7 static/dynamic constructor lowering |
| `crates/ryft-xla/src/experimental/lowering/operations/control_flow.rs` | Behavior P7 mixed region/carry lowering |
| `crates/ryft-xla/src/experimental/lowering/operations/elementwise.rs` | Behavior P7 homogeneous lowering |
| `crates/ryft-xla/src/experimental/lowering/operations/jit_call.rs` | Behavior P7 dynamic-signature call lowering |
| `crates/ryft-xla/src/experimental/lowering/operations/manipulation.rs` | Behavior P7 explicit shape-operand lowering |
| `crates/ryft-xla/src/experimental/lowering/operations/mod.rs` | Behavior P7 lowering exports |
| `crates/ryft-xla/src/experimental/lowering/operations/random.rs` | Behavior P7 dynamic RNG lowering |
| `crates/ryft-xla/src/experimental/lowering/operations/shard_map.rs` | Behavior P7 dimension-bearing shard-map lowering |
| `crates/ryft-xla/src/experimental/lowering/runtime_dimensions.rs` | Behavior P7 materialization knowledge; Superseded environment reconstruction |
| `crates/ryft-xla/src/experimental/lowering/types.rs` | Behavior P7 array/dimension StableHLO type conversion |
| `crates/ryft-xla/src/experimental/mod.rs` | Behavior P7/P9 experimental backend exports |
| `crates/ryft-xla/src/experimental/operations/reshape.rs` | Behavior P3/P7 canonical dynamic reshape contract |
| `crates/ryft-xla/src/experimental/operations/shard_map.rs` | Behavior P5/P7 mixed region signatures |
| `crates/ryft-xla/src/experimental/ops.rs` | Behavior P7 operation-family assembly; align with P8 |
| `crates/ryft-xla/src/experimental/shard_map.rs` | Behavior P5/P7 shard-map batching/closure |
| `crates/ryft-xla/src/jit.rs` | Behavior P1/P7 cache signatures and bounded-input ABI |
| `crates/ryft-xla/src/lib.rs` | Behavior P7/P9 exports |
| `crates/ryft-xla/src/telemetry.rs` | Behavior P10 compile/runtime measurement fields |
| `python/src/ryft/jax/dynamic_shape_parity.py` | Behavior P0/P3/P7 JAX behavioral parity programs |
| `python/src/ryft/jax/reshape_parity.py` | Behavior P0/P3/P7 reshape parity; retarget expression spellings to behavior |
| `python/tests/test_dynamic_shape_parity.py` | Behavior P3/P7/P10 multi-size acceptance |
| `python/tests/test_jax_tools.py` | Behavior P0/P10 parity harness coverage |

## Completion rule

P11 closes this ledger only after a script compares this path column with the archive manifest and proves a one-to-one
match. Each later phase updates the cleanup ledger with the paths it consumed; no phase may bulk-copy archive files
solely because they are classified as behavioral sources.
