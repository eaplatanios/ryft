use proc_macro::TokenStream;

mod capabilities;
mod diagrams;
mod helpers;
mod kernels;
mod operations;
mod parameters;

use capabilities::CapabilityCodeGenerator;
use diagrams::DiagramRenderer;
use kernels::KernelCodeGenerator;
use operations::OperationCodeGenerator;
use parameters::ParameterizedCodeGenerator;

/// Generates a [`Parameter`] implementation for the container this macro is applied on.
#[proc_macro_derive(Parameter)]
pub fn derive_parameter(input: TokenStream) -> TokenStream {
    ParameterizedCodeGenerator::generate_parameter_impl(input)
}

/// Generates a [`Parameterized`] implementation for the container this macro is applied on.
#[proc_macro_derive(Parameterized, attributes(ryft))]
pub fn derive_parameterized(input: TokenStream) -> TokenStream {
    ParameterizedCodeGenerator::generate_parameterized_impl(input)
}

/// Generates an [`Operation`] implementation for the enum this macro is applied on, together with the implementations
/// through which the enum dispatches every other operation contract to the payload of its active variant.
///
/// Every variant of the enum must wrap exactly one operation payload, which may be stored directly or boxed, and
/// the enum then behaves like whichever payload it holds. The macro always generates the `Display`, `Operation`,
/// `InterpretableOperation`, and `OperationPayloadProjection` implementations, one `OperationProjection`
/// implementation per projected member variant, and the owned `From<Payload>` and borrowed `TryFrom<&Enum>` payload
/// conversions. Unless a variant stores a native `GuardOperation`, it also generates an `OperationProvider`
/// implementation that rejects guard requests, and when a variant declares `#[ryft(collective)]` or
/// `#[ryft(lockstep)]`, it generates a `CollectiveOperation` or `LockstepOperation` implementation that delegates to
/// the payloads of the variants that declare it and reports no rule for every other variant. The payload-delegating
/// `PartialEq`, `Eq`, and `Hash` implementations, as well as the reference discharge, partial evaluation, batching,
/// differentiation, and transposition dispatchers, are generated only when they are selected through
/// `#[ryft(dispatch(...))]`.
///
/// The macro supports the following optional `#[ryft(...)]` attributes:
///
/// ```text
/// // Enum-level attributes:
/// #[ryft(crate = "...")]                    // Path through which generated code names `ryft` (by default, `ryft`).
/// #[ryft(type(T), constant(C))]             // Operation type and stored constant type, which are otherwise inferred.
/// #[ryft(members(U [, structural(S)]...))]  // Member universes of the operation family.
/// #[ryft(dispatch(identity, discharge, partial_evaluation, batching, differentiation, transposition))]
///
/// // Variant-level attributes:
/// #[ryft(projected(U [, structural]))]      // Every input and output of the instruction belongs to `U`.
/// #[ryft(mixed(U [, structural]))]          // The instruction crosses member universes, and `U` may be defaulted.
/// #[ryft(family)]                           // The payload is a nested operation family of the same type.
/// #[ryft(collective)]                       // `CollectiveOperation` delegates to the payload of the variant.
/// #[ryft(lockstep)]                         // `LockstepOperation` delegates to the payload of the variant.
/// #[ryft(skip_from)]                        // Skips only the owned `From<Payload>` conversion of the variant.
/// ```
///
/// For example, the following enum derives an `Operation<Type = ArrayType>` implementation, along with the other
/// implementations listed above, for both of its variants:
///
/// ```rust,ignore
/// #[derive(Clone, Debug, Operation)]
/// enum BackendOperation<V: Value<Type = ArrayType>> {
///     Zero(ZeroOperation<ArrayType>),
///     Constant(ConstantOperation<V>),
/// }
/// ```
///
/// Refer to the documentation of the `Operation` trait in `ryft-core` for the canonical reference of this macro, which
/// covers the enum-level attributes, how the operation and constant types are inferred, the grammar of the variant
/// classes along with their boundary shapes and transform roles, the generated implementations and their contracts,
/// and the requirements and limitations of the macro.
#[proc_macro_derive(Operation, attributes(ryft))]
pub fn derive_operation(input: TokenStream) -> TokenStream {
    OperationCodeGenerator::generate_operation_impl(input)
}

/// Validates the conventions of the capability trait this attribute is applied on and, for every
/// `projection(Family => Member)` argument, generates the implementation through which the values
/// of the `Family` universe apply the capability by projecting onto their `Member` values.
///
/// The attribute supports the following optional arguments:
///
/// ```text
/// #[capability(crate = "...")]                 // Path through which generated code names `ryft` (by default, `ryft`).
/// #[capability(projection(Family => Member))]  // Implements the capability for `Family` values (may be repeated).
/// ```
///
/// For example, `#[capability(projection(ArrayIrType => ArrayType))]` implements a capability for array IR values by
/// projecting them onto their array members. The annotated trait is always emitted unchanged, including when the
/// attribute reports errors. Refer to the documentation of the `Capability` trait in `ryft-core` for the canonical
/// reference of this attribute, which covers the checked conventions, the generated projection implementations, and
/// the supported function signatures.
#[proc_macro_attribute]
pub fn capability(attributes: TokenStream, input: TokenStream) -> TokenStream {
    CapabilityCodeGenerator::generate_capability_impl(attributes.into(), input.into()).into()
}

// TODO(eaplatanios): Review this.
/// Defines an experimental portable kernel from the function that it is applied on, whose parameters are annotated
/// arrays and whose body uses a subset of Rust that is lowered into staged operations, so that no host Rust code runs
/// while the kernel is traced.
///
/// The attribute replaces the function `name` with a callable of the same name and a module `name` that contains a
/// `definition` function. The callable takes the kernel arguments (i.e., its `input`, `inout`, and `prefetch`
/// parameters, in declaration order) as `&V` for any `V: KernelCall` (e.g., concrete `Array`s, which are interpreted,
/// or projected staged values, which bind the kernel call into their context) and returns the kernel results (i.e., its
/// `output` and `inout` parameters, in declaration order), as a tuple when there are several of them. The
/// `definition` function takes the `ArrayType` of each argument, validates them against the parameter annotations and
/// the `requires` constraint, and returns the `KernelDefinition` that the body traces into, which can be inspected,
/// verified, and compiled independently of the callable.
///
/// ```rust,ignore
/// # use ryft::kernel;
///
/// #[kernel(requires = left.shape()[0] == right.shape()[0])]
/// fn add(
///     #[input(data_type = F32, rank = 1)] left: &Array,
///     #[input(data_type = F32, rank = 1)] right: &Array,
///     #[output(data_type = F32, shape = [left.shape()[0]])] output: &mut Array,
/// ) {
///     output.store(left.load() + right.load());
/// }
/// ```
///
/// Kernels may also run on a grid of programs, each of which accesses one block of each parameter. The following kernel
/// multiplies two matrices of any shape in blocks of 32 by 32 elements. Every program `(i, j, k)` accumulates one block
/// product into the output block `(i, j)`, which it zeroes at the first step of the sequential axis `k`, and the edge
/// blocks are masked, so their padding is loaded as zero and never stored:
///
/// ```rust,ignore
/// #[kernel(
///     grid(i = left.shape()[0].div_ceil(32), j = right.shape()[1].div_ceil(32),
///          k = sequential(left.shape()[1].div_ceil(32))),
///     requires = left.shape()[1] == right.shape()[0],
/// )]
/// fn matmul(
///     #[input(data_type = F32, rank = 2, block = [32, 32], index = [i, k], boundary = masked)] left: &Array,
///     #[input(data_type = F32, rank = 2, block = [32, 32], index = [k, j], boundary = masked)] right: &Array,
///     #[output(data_type = F32, shape = [left.shape()[0], right.shape()[1]],
///              block = [32, 32], index = [i, j], boundary = masked)]
///     output: &mut Array,
/// ) {
///     if k == 0 {
///         output.store(zeros::<f32>([32, 32]));
///     }
///     output.accumulate(left.load_or(0.0).dot(right.load_or(0.0)));
/// }
/// ```
///
/// # Signatures
///
/// Kernel functions must be safe, synchronous, and non-generic, and they return nothing, because their results are
/// written through their parameters. The attribute takes the following arguments, all of which are optional:
///
///   - `grid(name = extent, ...)` declares the axes of the grid in order. `extent` is shape metadata, and it is
///     wrapped as `sequential(extent)` for an axis whose programs run in order and may carry state from one step to
///     the next, or optionally as `parallel(extent)` for an axis whose programs are independent, which is the default.
///     Axis names must be unique and differ from the parameter names. Each axis name is bound in the body to the
///     coordinate of the current program along that axis. Without `grid(...)`, the kernel runs a single program.
///   - `requires = ...` is shape metadata that the argument types must satisfy.
///   - `crate = "path"` names the kernel namespace, as described below.
///
/// Each parameter has exactly one annotation, where `D` is a `DataType` variant (e.g., `F32`), `R` is an integer
/// literal, and `S` is shape metadata:
///
///   - `#[input(data_type = D, rank = R, ...)]` on `&Array` declares a read-only argument.
///   - `#[output(data_type = D, shape = S, ...)]` on `&mut Array` declares a write-only result, every element of which
///     the kernel must write.
///   - `#[inout(data_type = D, rank = R, ...)]` on `&mut Array` declares a read-write argument, which is also a result
///     that holds its updated value.
///   - `#[prefetch(data_type = D)]` on `&Array` declares a rank-0 integer argument whose value is available to the
///     window mappings and, as a staged value, to the body. Prefetch arguments follow the other arguments in the call.
///
/// The `input`, `output`, and `inout` annotations may also declare the window of their parameter that each program
/// accesses, which is the whole array by default:
///
///   - `block = [...]` is the static shape metadata of the block, together with either `index = [...]`, whose entries
///     are block indices (i.e., the window of each axis starts at the index times the block extent), or
///     `start = [...]`, whose entries are element offsets. Both have one entry per block axis, and each entry is
///     dimension arithmetic over grid axis names, prefetch parameter names, integer literals, and shape metadata, with
///     `+`, `-`, `*`, `/`, `%`, `min`, `max`, and `saturating_sub`. Block indices that are grid axes let the kernel
///     verifier prove the coverage of the windows without enumerating the grid.
///   - `boundary = masked` lets blocks extend past their array, in which case only their in-bounds elements are valid,
///     and `boundary = in_bounds`, the default, requires every block to lie within its array.
///
/// An output annotation may instead tile the output with `tile = T` and `boundary = B`, where `T` is the shape metadata
/// of each tile. This is the shorthand for a parallel grid with one program per output tile, whose inputs are whole
/// arrays, and it cannot be combined with `grid(...)`, other results, `prefetch` parameters, or input windows. The body
/// binds the coordinates of the current tile with `let [row, column] = output.tile_index();` and its stores into the
/// output write that tile.
///
/// Shape metadata (i.e., the grid extents, the `shape`, `tile`, and `block` annotations, the `requires` constraint,
/// loop bounds, slice ranges, and the shapes of constructed values) consists of integer literals, arrays, argument
/// extents written as `argument.shape()[axis]` with a literal axis below the rank of the argument,
/// `extent.div_ceil(divisor)`, comparisons, and `&&` and `||`. It is evaluated from the argument types when the
/// definition is built.
///
/// # Bodies
///
/// Kernel bodies support the following statements:
///
///   - `let` bindings, which may be mutable but cannot shadow other bindings (including the parameters and the grid
///     axis names), and assignments, including `+=`, `-=`, `*=`, and `/=`,
///   - `for index in start..end` loops over shape metadata bounds, which stage `while` loops with signed 64-bit scalar
///     indices, and `if` statements over staged Boolean predicates, which stage conditions, where both carry every
///     visible binding and the mutable bindings take their staged results afterwards. A first step predicate such as
///     `if k == 0 { ... }` on a sequential grid axis lets the body initialize state that later steps accumulate into.
///   - `let [row, column] = output.tile_index();` in tiled kernels and outside loops and conditions, and
///   - tile views declared as `let tiles = input.tiles(shape).pad(literal);` and loaded with `tiles.load([indices])`,
///     where the indices count tiles and the out-of-bounds elements of the loaded tiles take the value of the literal.
///
/// Expressions are lowered as follows, where the function names are the names of the corresponding `ryft-core`
/// capability functions and fill values are scalar literals that take the element type of the accessed parameter:
///
///   | Syntax                                       | Lowering                                              |
///   |----------------------------------------------|-------------------------------------------------------|
///   | `param.load()`                               | read (rejected on a `masked` window)                  |
///   | `param.load_or(fill)`                        | masked load of the window, whose padding is `fill`    |
///   | `param.load_where(mask, fill)`               | masked load under a Boolean `mask`                    |
///   | `param.store(value)`                         | write, which publishes only the valid elements        |
///   | `param.store_where(value, mask)`             | masked store                                          |
///   | `param.swap_where(value, mask, fill)`        | masked swap, which returns the previous elements      |
///   | `param.window_mask()`                        | Boolean mask of the valid elements of the window      |
///   | `param.accumulate(value)`                    | addition into the window, masked on a `masked` window |
///   | `param.atomic_accumulate(value)`             | atomic addition into the window                       |
///   | `param.slice([a..b, ..])`                    | static sub-window (see below)                         |
///   | `scratch::<T>(shape)`                        | uninitialized scratch storage aligned for `T`         |
///   | `scratch_aligned::<T>(shape, alignment)`     | uninitialized scratch storage aligned to `alignment`  |
///   | `source.copy_to(target)`, `token.wait()`     | asynchronous copy and its completion                  |
///   | `left.dot(right)`                            | matrix multiplication of rank-2 values                |
///   | `+`, `-`, `*`, `/`, `%`, and unary `-`       | elementwise arithmetic                                |
///   | `&`, `\|`, `^`, and `!`                      | elementwise logic (`&&` and `\|\|` are rejected)      |
///   | `==`, `!=`, `<`, `<=`, `>`, and `>=`         | comparison, of dimensions for a grid axis and literal |
///   | `value as T`                                 | element type conversion to the Rust scalar type `T`   |
///   | `select(mask, on_true, on_false)`            | elementwise selection                                 |
///   | `value.exp()` and other math functions       | elementwise math                                      |
///   | `value.reduce_max([0])` and other reductions | reduction over literal axes                           |
///   | `zeros::<T>(shape)`, `ones::<T>(shape)`      | constants                                             |
///   | `full(shape, literal)`                       | constant whose elements all equal `literal`           |
///   | `iota::<T>(extent)`                          | vector that counts up from zero                       |
///
/// The unary math functions are `abs`, `ceil`, `cos`, `erf`, `exp`, `exp_m1`, `floor`, `ln_1p`, `log`, `logistic`,
/// `round`, `rsqrt`, `sign`, `sin`, `sqrt`, `tan`, and `tanh`, and the binary math functions are `atan2`,
/// `log_add_exp`, `max`, `min`, and `pow`. The reductions are `reduce_sum` (also written `sum`), `reduce_product`,
/// `reduce_mean`, `reduce_log_sum_exp`, `reduce_max`, `reduce_min`, `reduce_any`, and `reduce_all`. Parameters,
/// scratch storage, and slices are references, and every other value is staged. A slice, written inline or declared as
/// `let tail = param.slice([1..4, ..]);`, selects one static range per axis, where `..` selects a whole axis, and
/// supports only `load`, `store`, `accumulate`, and `atomic_accumulate`. Slices cannot be sliced again. A declared
/// slice and a tile view stay attached to the reference that their source names at the declaration, so assigning
/// another reference to a mutable local afterwards leaves a slice of that local unchanged, while writes to the
/// original reference remain visible through the slice.
///
/// Unsuffixed literals take the default Rust types (i.e., `i32` and `f64`) outside of fill values, so arithmetic with
/// arrays of other data types requires suffixed literals (e.g., `2.0f32`). Every method and function name is matched
/// against the closed set above together with its number of arguments, and anything else, including attributes,
/// macros, closures, and unbounded loops, is rejected with a diagnostic at the offending tokens. A kernel with errors
/// expands to its first error only.
///
/// # Kernel namespace
///
/// The generated code names the kernel API through a single namespace path and reaches `ryft-core` items through that
/// namespace's hidden `__private::core` macro-support re-export. Like the derive macros' `ryft` default, the namespace
/// defaults to the `ryft` facade path `ryft::kernels`, so callers that depend on the facade pass nothing. Callers that
/// reach `ryft-kernels` through another path pass that path as `crate = "path"`:
///
/// ```rust,ignore
/// #[ryft::kernels::kernel]                                           // `ryft` facade (default namespace).
/// #[ryft::kernel]                                                    // `ryft` facade's top-level re-export.
/// #[ryft_kernels::kernel(crate = "ryft_kernels")]                    // Direct `ryft-kernels` dependency.
/// #[kernels_alias::kernel(crate = "kernels_alias")]                  // Renamed `ryft-kernels` dependency.
/// #[facade_alias::kernels::kernel(crate = "facade_alias::kernels")]  // Renamed `ryft` facade dependency.
/// ```
///
/// The macro cannot infer the namespace from the attribute path because a re-exported procedural attribute cannot see
/// the path through which it was invoked. Relative namespace paths resolve in the module that contains the kernel
/// function, like the other paths in the kernel (e.g., the element type in `zeros::<T>`). The path always names the
/// kernel namespace. A stale override that names the core crate (e.g., `crate = "ryft_core"`) is not diagnosed by the
/// macro and instead fails with unresolved-path errors inside the generated code, the first of which is that
/// `__private` cannot be found in `ryft_core`.
#[proc_macro_attribute]
pub fn kernel(attributes: TokenStream, input: TokenStream) -> TokenStream {
    KernelCodeGenerator::generate_kernel_impl(attributes.into(), input.into()).into()
}

/// Renders the [Mermaid](https://mermaid.js.org/) diagrams in the documentation of the item that it is applied on.
///
/// The attribute rewrites every ```` ```mermaid ```` code fence (or `~~~mermaid` fence) in the outer documentation
/// attributes of the item, which include its `///` comments, into a `<pre class="mermaid">` HTML block, and inserts a
/// small module script before the first diagram of the item that loads Mermaid and renders the diagrams of the page in
/// the browser. The script renders each page once even when several of its items carry diagrams. It selects the dark
/// Mermaid theme when the page uses a dark rustdoc theme (i.e., `dark` or `ayu`), or when the page uses no other
/// built-in rustdoc theme and the browser prefers a dark color scheme. The theme is selected when the diagrams are
/// rendered, so diagrams pick up a rustdoc theme change only after the page is reloaded. All other documentation lines,
/// including other code fences, are preserved verbatim and in order, and items without diagrams are returned unchanged.
/// The attribute takes no arguments, and it rejects unterminated diagrams, code fences nested inside diagrams, and
/// attributes other than literal documentation attributes placed between the lines of a diagram. When it reports
/// errors, the item is still emitted unchanged, so that its uses elsewhere do not report additional errors about
/// a missing item.
///
/// Diagram text reaches Mermaid as written, except that HTML character references in it are decoded first, so that
/// labels can contain HTML elements spelled with entities (e.g., `&lt;code&gt;` for a `code` element).
///
/// The attribute is meant to be applied only when building documentation, so that ordinary builds never run it:
///
/// ```rust,ignore
/// /// Graph with a single edge.
/// ///
/// /// ```mermaid
/// /// flowchart LR
/// ///   source --> target
/// /// ```
/// #[cfg_attr(doc, mermaid)]
/// pub struct Graph;
/// ```
///
/// # Mermaid Bundle
///
/// When it expands an item with diagrams and the rustdoc output directory `<target>/doc` already exists, the attribute
/// writes a vendored Mermaid bundle to `<target>/doc/static.files.mermaid/mermaid.min.js` (unless that file already
/// exists with the same size), where `<target>` is `$CARGO_TARGET_DIR` or, when that variable is unset, `./target`
/// relative to the working directory of the compiler. Failing to write the bundle does not fail the build. The loader
/// script first requests this bundle relative to the rustdoc root path, which also works for documentation opened
/// directly from the file system, and then falls back to `https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js`.
/// If neither loads, each diagram is replaced by a note saying that it could not be rendered.
///
/// Because the output directory is derived only from `CARGO_TARGET_DIR` and `./target`, documentation written
/// elsewhere (e.g., by `cargo doc --target <triple>`, which writes to `<target>/<triple>/doc`, or into a target
/// directory configured through `--target-dir` or `build.target-dir`) does not receive the bundle and renders its
/// diagrams through the CDN. The same holds when the bundle is deleted while the documentation of the crate is fresh,
/// because rustdoc only expands the attribute when it documents the crate again.
#[proc_macro_attribute]
pub fn mermaid(attributes: TokenStream, input: TokenStream) -> TokenStream {
    DiagramRenderer::render_diagrams(attributes.into(), input.into()).into()
}
