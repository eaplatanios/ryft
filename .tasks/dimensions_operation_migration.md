# Symbolic dimensions operation migration matrix

This matrix is the P0 destination for every operation affected by the array/dimension storage sum. It is intentionally
about semantic contracts, not the transitional Rust enum layout.

Legend:

- `H`: homogeneous member rule through the generic P2 projected context.
- `M`: canonical mixed rule with a P3 typed dimension-operand schema.
- `S`: structural/nondifferentiable dimension value; no tangent or cotangent.
- `R`: ordinary transform residual, represented by SSA values rather than operation-payload witnesses.
- `—`: the transform is semantically inapplicable.
- Phase identifiers name the increment that owns implementation and deletion.

Unless a row says otherwise, tracing and PE preserve the canonical operand list exactly, batching rejects mapped shape
authority, differentiation treats dimensions as structural, region import uses generic identity closure, lowering
consumes operands directly, and tests cover static and dynamic eager/traced execution plus wrong-kind/count
diagnostics.

## Mixed and shape-dependent operations

| Operation | Canonical signature | Eager | Trace | PE | Batch | JVP | VJP | Transpose | Regions | Lowering | Tests | Old-code deletion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero | Static: homogeneous `() -> array`; dynamic: `dims* -> array` shaped wrapper | M host extents | M | P4 | P5 dims replicated | zero tangent | zero cotangent | self/structural | none | P7 dynamic shape operand | static/dynamic/allocation | P3 overlapping composite impl and recovery |
| One | Static: homogeneous `() -> array`; dynamic: `dims* -> array` shaped wrapper | M host extents | M | P4 | P5 dims replicated | zero tangent | zero cotangent | structural | none | P7 dynamic shape operand | static/dynamic/allocation | P3 overlapping composite impl and recovery |
| Fill | Static: homogeneous payload; dynamic: `dims* -> array` shaped wrapper | M host extents | M | P4 | P5 dims replicated | value tangent broadcast | value cotangent reduce | R output geometry | none | P7 dynamic shape operand | static/dynamic/value AD | P3 overlapping composite impl and recovery |
| Iota | Static: homogeneous `() -> array`; dynamic: `dims* -> array` shaped wrapper | M host extents | M | P4 | P5 axis policy | zero tangent | zero cotangent | structural | none | P7 dynamic iota | static/dynamic/axis | P3 overlapping composite impl and recovery |
| Broadcast | `(array, output_dims*) -> array` | M | M | P4 | P5 inserts/moves batch axis | broadcast tangent | reduce cotangent | R input extents | none | P7 dynamic broadcast | static/direct/derived/batch/AD | P3 dual contract, collectors, view recovery |
| Scalar to dimension | `rank_0_integer_array -> dimension` | checked gateway | M | known values fold | P5 mapped authority rejected | S | S | — | none | P7 host/checked extraction | bounds/kind/authority | P2 bespoke gateway projection |
| Dimension to scalar | `dimension -> rank_0_integer_array` | host scalar | M | known values fold | replicated result | S | S | — | none | P7 scalar tensor | round trip/data use | P2 bespoke gateway projection |
| Checked scalar to dimension | `(rank_0_integer_array, requirements) -> dimension` | checked gateway | M | P4 proof/fold | P5 mapped authority rejected | S | S | — | none | P7 checked extraction/assert | bounds/requirements | P2 bespoke authority path |
| Dimension size | `(array, axis) -> dimension` | reads metadata | M | known shape folds | P5 rejects mapped queried extent | S | S | — | none | P7 dimension-size scalar | static/dynamic/batch | P3 homogeneous result contract |
| Dimension compare | `(dimension, dimension) -> rank_0_bool_array` | host compare | M | P4 proof/fold | replicated | S | S | — | none | P7 scalar compare | all predicates | P2 hand-written projection |
| Reshape | `(array, output_dims*) -> array` | M | M | P4 | P5 preserves element-count authority | reshape tangent | reshape cotangent | R input extents | none | P7 `dynamic_reshape` operand | static/direct/derived/batch/AD | P3 dual contract and collectors; P6 `transpose_dimension_variables` |
| Slice | `(array, starts*, sizes*) -> array` using typed schema | M | M | P4 | P5 adjusts mapped axis | slice tangent | scatter cotangent | R input geometry/starts/sizes | none | P7 dynamic slice/static slice | static/dynamic/batch/AD | P3 dual contract and collectors |
| Slice scatter | `(update, output_geometry*, starts*, sizes*) -> array` using typed schema | M | M | P4 | P5 adjusts mapped axis | scatter tangent | slice cotangent | R geometry/starts/sizes | none | P7 scatter/update lowering | static/dynamic/batch/AD | P3 dual contract and collectors |
| Dynamic slice | `(array, start_indices, sizes*) -> array` | M | M | P4 | P5 mapped-axis policy | slice tangent | dynamic-update cotangent | R source geometry/starts/sizes | none | P7 dynamic slice | dynamic/batch/AD | P3 dual contract and collectors |
| Pad | `(array, pad_value, output_dims*) -> array` | M | M | P4 | P5 mapped-axis padding | pad tangent | crop/reduce cotangent | R operand/output extents | none | P7 dynamic/static pad | edge/interior/dynamic/batch/AD | P3 dual contract and collectors |
| Gather | `(operand, indices, slice_sizes*) -> array` | M | M | P4 | P5 index/batch mapping | gather tangent | scatter-add cotangent | R operand shape/slice sizes | none | P7 gather | dynamic sizes/batch/AD | P3 dual contract and collectors |
| Concatenate | `(arrays+, concat_extents*) -> array` | M | M | P4 | P5 axis placement | concat tangents | split cotangent | R input concat-axis extents | none | P7 concatenate | static/dynamic/batch/AD | P3 dual contract, collectors, context search |
| Reduce/mean | `(array, init, reduced_extents*) -> array` | M | M | P4 | P5 batch-axis policy | reduced tangent | broadcast/scale cotangent | R reduced extents | optional reducer region remains typed | P7 reduce | sum/mean/dynamic/batch/AD | P3 dual contract and collectors |
| RNG bit generation | `(state, output_dims*) -> (state, array)` | M | M | P4 | P5 scan/vectorization policy | nondifferentiable | nondifferentiable | — | none | P7 dynamic RNG shape | static/dynamic/batch | P3 dual contract and collectors |
| All-gather | `(array, result_dims*) -> array` | M | M | P4 | P5 collective axis policy | collective tangent | inverse collective | R inverse geometry | none | P7 collective | sharding/dynamic/batch/AD | P3 copied shape validation; P6 recovery |
| Psum-scatter | `(array, result_dims*) -> array` | M | M | P4 | P5 collective axis policy | collective tangent | inverse collective | R inverse geometry | none | P7 collective | sharding/dynamic/batch/AD | P3 copied shape validation; P6 recovery |
| All-to-all | `(array, result_dims*) -> array` | M | M | P4 | P5 collective axis policy | collective tangent | inverse collective | R inverse geometry | none | P7 collective | sharding/dynamic/batch/AD | P3 copied shape validation; P6 recovery |
| Custom call | `(arrays*, output_dims*) -> arrays*` | backend eager path | M | residual unless backend folds | P5 explicit policy/unsupported | declared rule only | declared rule only | declared residuals | none | P7 flattened output shape operands | multi-output/dynamic/errors | P3 dual contract and flattened collector |

## Region-polymorphic operations

Region operations use the outer storage type only because their regions may carry arrays and dimensions. Their
operation-specific code must not project storage variants manually; P2/P8 generated typed region schemas own that
work.

| Operation | Canonical signature | Eager | Trace | PE | Batch | JVP | VJP | Transpose | Regions | Lowering | Tests | Old-code deletion |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Condition | `(predicate, operands...) -> results...` | selected branch | preserves carries | folds known predicate | P5 structural dimension carries | P6 branch JVP | P6 branch VJP | R branch primals | both branches same typed signature | P7 `if` regions | mixed carries/closure/AD | P4/P5/P6 manual variant dispatch |
| While | `(carries...) -> carries...` | loop | preserves carries | P4 bounded known folding | P5 structural dimension carries | P6 loop JVP | P6 loop VJP policy | R loop history as required | condition/body closure | P7 `while` regions | mixed carries/closure/batch/AD | P4/P5/P6 manual variant dispatch |
| Scan | `(carries..., xs...) -> (carries..., ys...)` | loop | preserves carries | P4 known-side folding | P5 mapped inputs and structural dims | P6 scan JVP | P6 scan VJP | R carries/history | body closure and stacked outputs | P7 scan lowering | mixed carries/stacked dims/batch/AD | P4/P5/P6 manual variant dispatch |
| Custom JVP | `(operands...) -> results...` | primal region | preserves declaration | residual | P5 transforms both regions | declared JVP | derived from declaration if supported | R declared primals | primal/JVP signature checked structurally | P7 regions | mixed signature/closure | P6/P8 hand-written projections |
| Custom VJP | `(operands...) -> results...` | primal region | preserves declaration | residual | P5 transforms regions | derived if supported | declared VJP | R declared residuals | primal/forward/backward signatures checked | P7 regions | mixed signature/closure | P6/P8 hand-written projections |
| Custom-VJP tangent | `(residuals..., cotangents...) -> input_cotangents...` | backward region | preserves declaration | residual | P5 structural dims | — | declared rule | R declared residuals | backward signature checked | P7 region call | mixed residuals | P6/P8 concrete-type contract |
| Rematerialize | `(operands...) -> results...` | body region | preserves body | policy-controlled | P5 transforms body | P6 recompute JVP | P6 recompute VJP | R only irreducible values | body signature checked | P7 call/region | mixed carries/AD | P6/P8 manual variant dispatch |

## Homogeneous member families

The following operations remain homogeneous and use the same `Operation<ArrayType>` or `Operation<DimensionType>`
contract in eager execution, tracing, PE, batching, differentiation, rendering, import, and lowering. P2 supplies
projection/lift; P8 generates dispatcher shells. None receives an `Operation<ArrayProgramType>` implementation.

| Family | Operations | Transform policy | Tests and deletion |
|---|---|---|---|
| Array constants and elementwise | Zero-like, one-like, constant, abs, neg, add, sub, mul, div, sin, cos, atan2, exp, log, sqrt, rsqrt, tanh, logistic, erf, pow, sign, floor, ceil, round, maximum, minimum, remainder, not, and, or, xor, complex, conjugate, real, imaginary | H; existing array batching/AD rules | P2 vertical prototype and P8 dispatch fixtures; delete adapter matches |
| Array linear algebra and attention | Dot, scaled dot, dot-product attention, dot-product-attention backward | H; existing batch/JVP/VJP rules | Existing operation tests plus P8 compile tests |
| Array ordering/collective | Sort, generic collective, ppermute | H; existing transform rules | Existing tests; delete outer projection boilerplate |
| Array axes/manipulation | Axis index, transpose, scatter, update slice, dynamic update slice, compare, select, convert element type | H; transpose residuals remain ordinary array/SSA inputs | Existing tests plus P2 projection tests |
| Array placement/metadata | Transfer to memory, reshard, sharding constraint, stop gradient, tag, print | H; operation-specific effects/policies retained | Existing tests; P8 generated dispatch |
| Dimension arithmetic | Constant, add, subtract, clamped subtract, multiply, floor divide, remainder, minimum, maximum | H; replicated in batching and S in AD | P2/P4 eager, trace, PE, proof, overflow tests |
| Dimension predicates/requirements | Compare; require equal, less-than-or-equal, divisible-by, and nonzero | H; requirements carry ordered assertion effect until proven | P4 exact diagnostics and proof-inventory tests |

Select and stop-gradient stay generic over a homogeneous member type. The test-only nullary operation stays generic
for derive coverage. Their Rust parameterization is not evidence that a concrete payload may infer unrelated type
families.

## Deletion completion rules

An operation row is complete only when:

1. Its canonical signature is the only production type contract for that concrete payload.
2. Eager, trace, PE, batching, differentiation, regions, and lowering consume the same ordered operands.
3. No context-side array or dimension vector reconstructs a dependency.
4. No operation payload stores `transpose_dimension_variables` or another expression/witness substitute.
5. No ad hoc `runtime_dimension_variables` collector or copied validation loop remains.
6. Static and dynamic tests exercise the same canonical mixed rule, with an empty dynamic segment for static shapes.
7. Targeted searches classify every residual old-family, view, collector, and witness occurrence.
