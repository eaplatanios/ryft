//! Backend independent kernels, which express a computation over blocks of arrays as an ordinary Ryft program that
//! runs once per point of a logical grid.
//!
//! This crate defines what a kernel means, checks that a kernel is well formed, and interprets kernels on the host. It
//! compiles nothing itself. Compiler adapters (e.g., `ryft-mosaic`, `ryft-triton`, and cuTile in `ryft-cuda`)
//! implement [`KernelCompiler`], and execution integrations (e.g., the `kernels` module of `ryft-xla`) embed the
//! outputs of those adapters into executable programs. Using this crate therefore requires no XLA, CUDA, PJRT, or
//! Python installation. Adapter APIs and target specific extensions remain experimental, and this documentation
//! promises neither a stable upstream ABI nor that every kernel lowers on every target.
//!
//! # Kernels
//!
//! A kernel call is a [`KernelCallOperation`], which consists of a [`Grid`] and a list of [`KernelParameter`]s. Each
//! point of the grid runs one program of the kernel. Each parameter is a full array with a declared
//! [`KernelParameterAccess`] (i.e., read-only, write-only, or read-write) and a [`BlockMapping`], which maps the grid
//! coordinates of a program to the window of that array that the program accesses. The [`BoundaryPolicy`] of a mapping
//! states whether every window must lie within its array or whether windows may extend past it, in which case the
//! program must mask the invalid elements. A call may also take scalar prefetch values, which are rank-0 integer
//! arrays that its mappings receive as dimensions after the grid coordinates. At its outer boundary, a call takes the
//! read-only and read-write arrays followed by the scalar prefetch values, and it returns the write-only and read-write
//! arrays.
//!
//! The body of a kernel is an ordinary Ryft program over [`KernelOperation`], which extends the canonical array,
//! dimension, reference, and control flow operations of `ryft-core` with kernel specific operations. The body receives
//! one block reference per parameter, the grid coordinates of its program, and the scalar prefetch values, and it
//! returns nothing, because its stores into the block references of the write-only and read-write parameters produce
//! the results of the call.
//!
//! # Lifecycle
//!
//! A kernel is authored either with the [`kernel`] attribute macro or with [`KernelDefinition::trace`], which traces a
//! closure over the body inputs of an explicitly constructed [`KernelCallOperation`]. Both paths validate the traced
//! body into a [`KernelDefinition`], which [`validate_kernel_body`] checks against the declared parameter accesses.
//! [`VerifiedKernel::new`] then verifies that the definition initializes every result completely, accesses only valid
//! windows, and has no conflicting writes. A definition can be interpreted on the host through
//! [`KernelDefinition::interpret`], which performs the same verification first, or compiled by passing its
//! [`VerifiedKernel`] to [`VerifiedKernel::compile`] with a [`KernelCompiler`]. Compiler options never rewrite the
//! definition, so different compilers can compile the same verified definition.
//!
//! ```
//! use ryft_core::{Array, ArrayType, DataType};
//! use ryft_kernels::{kernel, VerifiedKernel};
//!
//! // This example depends on `ryft-kernels` directly rather than through the `ryft` facade.
//! #[kernel(crate = "ryft_kernels", requires = left.shape()[0] == right.shape()[0])]
//! fn add(
//!     #[input(data_type = F32, rank = 1)] left: &Array,
//!     #[input(data_type = F32, rank = 1)] right: &Array,
//!     #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
//! ) {
//!     output.store(left.load() + right.load());
//! }
//!
//! let left = Array::vector(vec![1.0f32, 2.0]).unwrap();
//! let right = Array::vector(vec![3.0f32, 5.0]).unwrap();
//! assert_eq!(add(&left, &right).unwrap(), Array::vector(vec![4.0f32, 7.0]).unwrap());
//! let vector = ArrayType::new_static(DataType::F32, [2]);
//! let definition = add::definition(&vector, &vector).unwrap();
//! let verified = VerifiedKernel::new(&definition, 1).unwrap();
//! assert_eq!(verified.definition().semantic_key().unwrap(), definition.semantic_key().unwrap());
//! ```
//!
//! The [`kernel`] macro replaces the annotated function with a callable of the same name and a module that contains a
//! `definition` function. The callable takes values of any [`KernelCall`] type. It interprets concrete host
//! [`Array`](ryft_core::Array)s through [`KernelDefinition::interpret`] with
//! [`DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS`] as its qualification bound, and it binds the call together with
//! its body into the context of projected values whose operations are [`KernelOperation`]s (e.g., values traced by a
//! [`TracingContext`](ryft_core::TracingContext) over [`KernelOperation`]). The `definition` function takes the
//! logical types of the inputs, checks them against the input annotations and the `requires` constraint, and returns
//! the inspectable [`KernelDefinition`] that the body traces into.
//!
//! Values staged by `ryft-xla` do not implement [`KernelCall`], so a kernel enters an XLA program through its
//! definition instead. The `stage_kernel` function of the `kernels` module of `ryft-xla` binds the call of a
//! [`KernelDefinition`] together with its body into the enclosing XLA program, and the compilation options of the XLA
//! session select the compiler adapter that later compiles it. Staging invokes no compiler. The
//! `stage_kernel_with_jvp`, `stage_kernel_with_vjp`, `stage_kernel_with_rules`, and `stage_kernel_with_fallback`
//! functions of the same module also attach explicit derivative rules or a pure fallback program, because kernel calls
//! have no implicit differentiation rule.
//!
//! The code that [`kernel`] generates names this crate through the `ryft` facade path `ryft::kernels` by default, so
//! `#[ryft::kernels::kernel]` and `#[ryft::kernel]` need no arguments. Crates that depend on `ryft-kernels` directly,
//! as the example above does, pass `crate = "ryft_kernels"`, and other paths are passed the same way, as in
//! `#[kernels_alias::kernel(crate = "kernels_alias")]` for a renamed dependency. The macro cannot infer this path
//! because a re-exported procedural attribute cannot see the path through which it was invoked. The override always
//! names this kernel namespace and never `ryft-core`, because generated code reaches core items through a hidden
//! macro support re-export of this crate.
//!
//! # Authoring
//!
//! The macro admits a deliberate subset of Rust, so an expression that Rust accepts outside a kernel is not
//! automatically admitted inside one. Input annotations constrain the data type and rank of each input, and the
//! output annotation specifies the shape of the output and optionally a tile shape and boundary policy, which make
//! the body run once per output tile on a parallel grid. In a tiled kernel, the statement
//! `let [row, column] = output.tile_index();` outside loops and conditions binds the coordinates of the current tile.
//! A tile view is declared as a `let` initializer such as `let tiles = input.tiles([width]).pad(0.0);` and loaded with
//! `tiles.load([index])`, where the index counts tiles. Such a load stages a [`TileLoadOperation`] whose elements
//! outside the input take the padding value. `output.store(value)` writes the window of the current program. [`zeros`]
//! and [`dot`] stage canonical array operations, and bounded `for` loops and `if` statements stage canonical control
//! flow rather than host callbacks. The macro rejects syntax outside its subset with diagnostics at the offending
//! tokens, whereas some other mistakes (e.g., a `tile_index` pattern whose binding count differs from the rank of the
//! grid) are reported when the definition is built.
//!
//! The following kernel adds two vectors of 5 elements in tiles of 2 elements, so its grid has 3 programs and the tile
//! of the last program extends past the end of both vectors. Its loads fill the missing element with the padding value,
//! and the masked output publishes only the element that lies inside the output.
//!
//! ```
//! use ryft_core::{Array, ArrayType, DataType, Dimension};
//! use ryft_kernels::kernel;
//!
//! #[kernel(crate = "ryft_kernels", requires = left.shape()[0] == right.shape()[0])]
//! fn tiled_add(
//!     #[input(data_type = F32, rank = 1)] left: &Array,
//!     #[input(data_type = F32, rank = 1)] right: &Array,
//!     #[output(data_type = F32, shape = [left.shape()[0]], tile = [2], boundary = masked)] output: &mut Array,
//! ) {
//!     let [block] = output.tile_index();
//!     let left_tiles = left.tiles([2]).pad(0.0);
//!     let right_tiles = right.tiles([2]).pad(0.0);
//!     output.store(left_tiles.load([block]) + right_tiles.load([block]));
//! }
//!
//! let left = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let right = Array::vector(vec![10.0f32, 20.0, 30.0, 40.0, 50.0]).unwrap();
//! assert_eq!(tiled_add(&left, &right), Ok(Array::vector(vec![11.0f32, 22.0, 33.0, 44.0, 55.0]).unwrap()));
//! let vector = ArrayType::new_static(DataType::F32, [5]);
//! let definition = tiled_add::definition(&vector, &vector).unwrap();
//! assert_eq!(definition.operation().grid().dimensions()[0].extent(), &Dimension::Static(3));
//! ```
//!
//! A `grid(...)` argument declares the grid explicitly instead, with one named axis per entry, each of which is
//! parallel unless it is declared as `sequential(extent)`. Each parameter then declares its window with `block` and
//! either `index`, whose entries are block indices over the grid axes, or `start`, whose entries are element starts,
//! and optionally with `boundary = masked`. Inside the body, the axis names are the grid coordinates of the current
//! program. The programs along a sequential axis run in order and may carry a write-only output window from one step
//! to the next, so a body that initializes its window when the coordinate of that axis is zero (e.g., `if k == 0`)
//! and accumulates into it at every step computes a sequentially carried reduction. A masked window is loaded with
//! `load_or(fill)`, which fills its padding elements, and `accumulate` publishes only its valid elements. The
//! following kernel multiplies a 3 by 3 matrix with a 3 by 2 matrix in blocks of 2 by 2 elements, so both the output
//! blocks and the blocks of the contraction axis are partial at the edges.
//!
//! ```
//! use ryft_core::{Array, ArrayType, DataType};
//! use ryft_kernels::{GridExecution, kernel};
//!
//! #[kernel(
//!     crate = "ryft_kernels",
//!     grid(
//!         i = left.shape()[0].div_ceil(2),
//!         j = right.shape()[1].div_ceil(2),
//!         k = sequential(left.shape()[1].div_ceil(2)),
//!     ),
//!     requires = left.shape()[1] == right.shape()[0],
//! )]
//! fn matmul(
//!     #[input(data_type = F32, rank = 2, block = [2, 2], index = [i, k], boundary = masked)] left: &Array,
//!     #[input(data_type = F32, rank = 2, block = [2, 2], index = [k, j], boundary = masked)] right: &Array,
//!     #[output(
//!         data_type = F32,
//!         shape = [left.shape()[0], right.shape()[1]],
//!         block = [2, 2],
//!         index = [i, j],
//!         boundary = masked,
//!     )]
//!     output: &mut Array,
//! ) {
//!     if k == 0 {
//!         output.store(zeros::<f32>([2, 2]));
//!     }
//!     output.accumulate(left.load_or(0.0).dot(right.load_or(0.0)));
//! }
//!
//! let left = Array::matrix(3, 3, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
//! let right = Array::matrix(3, 2, vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
//! assert_eq!(matmul(&left, &right), Ok(Array::matrix(3, 2, vec![4.0f32, 5.0, 10.0, 11.0, 16.0, 17.0]).unwrap()));
//! let definition = matmul::definition(
//!     &ArrayType::new_static(DataType::F32, [3, 3]),
//!     &ArrayType::new_static(DataType::F32, [3, 2]),
//! )
//! .unwrap();
//! let executions = definition.operation().grid().dimensions().iter().map(|axis| axis.execution()).collect::<Vec<_>>();
//! assert_eq!(executions, [GridExecution::Parallel, GridExecution::Parallel, GridExecution::Sequential]);
//! ```
//!
//! [`KernelDefinition::trace`] gives explicit control over the [`Grid`], the [`KernelParameter`]s, and their
//! [`BlockMapping`]s, and the functions of [`authoring`] stage the same operations that the macro uses.
//! [`mapped_call`] constructs a call from a grid and one [`ParameterWindow`] per parameter, whose block indices or
//! element starts are [`MappingExpression`]s over the grid coordinates and the scalar prefetch values, so explicit
//! builders need not construct mapping programs themselves. Within a body, [`WindowMaskOperation`] returns which
//! elements of a window are valid, and a [`MaskedStoreOperation`] under that mask initializes a masked output window
//! completely. The [`memory`] module also provides scratch storage, masked loads and swaps, and asynchronous copies,
//! which [`masked_load`], [`masked_store`], [`masked_swap`], [`window_mask`], [`accumulate`], [`scratch`],
//! [`async_copy`], and [`wait`] stage. A [`KernelExtension`] adds target specific operations whose semantics an adapter
//! owns, including the [`KernelCompletionProtocol`] that the tokens of its asynchronous operations follow.
//!
//! # Qualification and Execution Bounds
//!
//! Verification (i.e., [`VerifiedKernel::new`], [`validate_kernel_initialization`], and the qualification that every
//! host interpretation performs first) takes a `maximum_programs` argument, which bounds the work of enumerating grid
//! points and is not part of the identity of a kernel. A parameter is qualified without enumerating the grid only when
//! [`BlockMapping::tiling_axes`] recognizes its mapping and that recognition names every grid axis at most once. Every
//! other parameter is qualified by evaluating its mapping at every grid point, which fails with
//! [`KernelInitializationError::QualificationLimit`] when the grid has more than `maximum_programs` points. A
//! recognized mapping is therefore not always qualified without enumeration. The diagonal windows of a mapping whose
//! element starts are `(i, i)` with blocks of one element are recognized, so `tiling_axes` returns
//! `Some(vec![Some(0), Some(0)])`, but they repeat the grid axis `i`, so that parameter is enumerated. When no
//! parameter is enumerated, `maximum_programs` limits nothing, and a grid of any size is qualified.
//!
//! Execution has separate bounds. [`KernelDefinition::interpret`], which the [`kernel`] callable uses for host arrays
//! and which ordinary program interpretation of a `kernel_call` uses with
//! [`DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS`], replays every grid point without a step limit, because kernel
//! bodies admit only bounded loops. [`KernelDefinition::interpret_with_limits`] and
//! [`KernelDefinition::interpret_with_trace`] cap the number of grid points before qualification, whether or not a
//! parameter is enumerated, and they cap the replay steps (i.e., operation bindings and region entries) of the whole
//! invocation. The schedule explorer [`KernelDefinition::interpret_interleavings`] applies the same cap on grid points,
//! computes the exact replay steps of one interleaving and checks them against the step cap before any interleaving
//! runs, and bounds the number of interleavings with its own `maximum_interleavings` argument.
//!
//! # Transforms, Identity, and Debugging
//!
//!   - [`KernelDefinition::batched`] adds a parallel grid dimension for a batch and maps each batched parameter along
//!     it. Unmapped read-only inputs remain shared, and every writable parameter must be mapped.
//!   - [`KernelDefinition::specialize_grid`] replaces dynamic grid extents with static ones, and
//!     [`KernelDefinition::specialize_prefetch`] binds scalar prefetch values to constants. Transforms never read
//!     device buffers back to the host, so prefetch values must be specialized explicitly (e.g., before batching).
//!   - [`KernelDefinition::semantic_key`] returns a versioned structural key of a definition, which excludes compiler
//!     options and schedules, so caches can combine it with the configuration key of a compiler.
//!   - [`KernelDefinition::to_source`] and [`KernelDefinition::from_source`] convert a definition to and from a
//!     versioned [`KernelSource`] document, and decoding validates the reconstructed definition again.
//!   - [`KernelDefinition::interpret_with_trace`] interprets a definition under explicit [`KernelDebugOptions`] and
//!     records the operations and reference accesses of every step. [`KernelDefinition::interpret_interleavings`]
//!     interprets it under every interleaving of the instructions of its programs that preserves the order within
//!     each program, up to explicit bounds.
//!
//! A successful interpretation proves only the interpreted case, and it implies neither target support nor native
//! performance. A kernel call that the `kernels` module of `ryft-xla` stages keeps its complete call through batching,
//! control flow, and rematerialization.

pub use ryft_macros::kernel;

pub mod authoring;
pub mod calls;
pub mod compilation;
pub mod grids;
pub mod indexing;
pub mod initialization;
pub mod interpretation;
pub mod mappings;
pub mod memory;
pub mod operations;
pub mod scheduling;
pub mod serialization;
pub mod transforms;
pub mod validation;

pub use authoring::{
    KernelCall, MappingExpression, ParameterWindow, accumulate, async_copy, compare, compare_dimension, dot, for_loop,
    full, iota, mapped_call, masked_load, masked_load_or, masked_store, masked_swap, ones, reference_slice, scratch,
    select, tile_load, tile_store, wait, whole_array_parameter, window_mask, zeros,
};
pub use calls::{
    KERNEL_CALL_OPERATION_NAME, KERNEL_KEY_VERSION, KernelCallOperation, KernelDefinition, KernelError, KernelParameter,
};
pub use compilation::{KernelCompilationError, KernelCompiler, KernelSchedule, VerifiedKernel};
pub use grids::{Grid, GridDimension, GridError, GridExecution, GridPoints};
pub use indexing::{TILE_LOAD_OPERATION_NAME, TileLoadOperation};
pub use initialization::{KernelInitializationError, validate_kernel_initialization};
pub use interpretation::{
    DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_PROGRAMS, DEFAULT_KERNEL_INTERPRETATION_MAXIMUM_STEPS, KernelDebugOptions,
    KernelInterpretationError, KernelTraceAccess, KernelTraceEntry,
};
pub use mappings::{BlockMapping, BlockMappingError, BlockWindow, BoundaryPolicy};
pub use memory::{
    ASYNC_COPY_OPERATION_NAME, AsyncCopyOperation, KernelMemoryError, MASKED_LOAD_OPERATION_NAME,
    MASKED_STORE_OPERATION_NAME, MASKED_SWAP_OPERATION_NAME, MaskedLoadOperation, MaskedStoreOperation,
    MaskedSwapOperation, SCRATCH_OPERATION_NAME, ScratchOperation, WAIT_OPERATION_NAME, WINDOW_MASK_OPERATION_NAME,
    WaitOperation, WindowMaskOperation,
};
pub use operations::{
    KernelCompletionProtocol, KernelExtension, KernelExtensionMemory, KernelOperation, NoKernelExtension,
};
pub use scheduling::{KernelInterleaving, KernelSchedulingError};
pub use serialization::{KernelSerializationError, KernelSource};
pub use transforms::KernelTransformError;
pub use validation::{
    KernelBoundaryContract, KernelParameterAccess, KernelParameterSummary, KernelReferenceOperation,
    KernelReferenceSummary, KernelSwapLowering, KernelValidationError, validate_kernel_body,
};

/// Macro support re-exports used by the code that [`kernel`] generates, which are not public API. The generated code
/// needs the core crate at a path derived from the kernel namespace, because a re-exported procedural attribute cannot
/// see which path invoked it. Generated `definition` functions also call the checked shape metadata helpers (i.e.,
/// [`static_extent`](authoring::static_extent) and [`shape_div_ceil`](authoring::shape_div_ceil)) and the tiled kernel
/// call constructor [`tiled_call`](authoring::tiled_call) re-exported here, which exist only for that generated code.
#[doc(hidden)]
pub mod __private {
    pub use ryft_core as core;

    pub use crate::authoring::{shape_div_ceil, static_extent, tiled_call};
}
