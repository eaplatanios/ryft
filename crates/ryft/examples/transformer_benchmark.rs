//! Benchmarks the end-to-end latency of a forward pass of a GPT-style decoder-only Transformer on Ryft's XLA backend,
//! either eagerly (i.e., op-by-op, without `jit` or staging, which is the default) or as one function that `jit`
//! compiles (with `--jit`).
//!
//! The model is a pre-LayerNorm GPT in `f32` that is written with primitive operations only, in the same order as the
//! JAX-eager and PyTorch scripts that it is compared against, so that every framework dispatches the same sequence of
//! primitives. `--fused-attention` replaces the score, softmax, and weighted-sum primitives of every attention block
//! with Ryft's `dot_product_attention` composite, which the XLA backend may lower to a fused kernel (e.g., cuDNN's
//! fused attention on CUDA). `--primitive-softmax` replaces Ryft's accuracy-preserving `softmax` (which dispatches
//! about 20 eager primitives) with the `max`, `sub`, `exp`, `sum`, `div` composition of `jax.nn.softmax`.
//!
//! The forward pass is written once over [`ArrayOperations`](ryft::ArrayOperations) values, so the eager mode runs it
//! on XLA arrays and `--jit` traces the same code on compilation tracers. The compiled function takes all parameters,
//! the tokens, the mask, and the scalar constants as runtime inputs and returns the sum of all logits.
//!
//! ```sh
//! cargo run --release -p ryft --features xla --example transformer_benchmark -- --small
//! cargo run --release -p ryft --features xla --example transformer_benchmark -- --small --jit
//! cargo run --release -p ryft --no-default-features --features cuda-13 --example transformer_benchmark
//! ```
//!
//! The benchmark prints progress to stderr and a single JSON line with its configuration and timings to stdout. Each
//! timed forward pass ends by copying the scalar sum of all logits to the host, which is its synchronization point.
//! `--profile` additionally reports a coarse per-category breakdown of the eager operations of a forward pass.

#[cfg(feature = "xla")]
mod benchmark {
    use std::cell::RefCell;
    use std::error::Error;
    use std::time::{Duration, Instant};

    use ryft::pjrt::{ClientOptions, GpuClientOptions, GpuMemoryAllocator, GpuPlatform};
    use ryft::xla::{Array, CompileTracer, CompiledFunctionDispatcher, Error as XlaError, Platform, Session, jit};
    use ryft::{
        Add, ArrayMemberType, ArrayOperations, ArrayType, AttentionConfiguration, AttentionInputs, DataType,
        DeviceMesh, Div, DotDimensionNumbers, GatherDimensionNumbers, GatherOptions, MeshAxis, MeshAxisType, Mul,
        Parameter, Parameterized, ProgramError, ProjectedContext, ReductionKind, Sub, dot_product_attention, softmax,
    };

    /// Result type of this benchmark.
    type BenchmarkResult<T> = Result<T, Box<dyn Error>>;

    /// Execution domain of the `dot_product_attention` composite over eager XLA arrays.
    type AttentionDomain = ProjectedContext<Session, ArrayType>;

    /// Composite functions that replace primitive operations of the attention blocks, if any.
    struct Functions<'a, V: Parameter> {
        /// Attention function that replaces the primitive score, softmax, and weighted-sum operations.
        attention: Option<&'a dyn Fn(AttentionInputs<V>) -> Result<V, ProgramError>>,

        /// Softmax function that replaces the `softmax` call over the attention scores.
        softmax: Option<&'a dyn Fn(V) -> Result<V, ProgramError>>,
    }

    impl<V: Parameter> Clone for Functions<'_, V> {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<V: Parameter> Copy for Functions<'_, V> {}

    /// Number of extra forward passes of each `--profile` measurement.
    const PROFILE_PASSES: usize = 3;

    /// GPU memory allocator that `--allocator` selects.
    #[derive(Copy, Clone, Debug)]
    enum Allocator {
        /// `cudaMallocAsync` memory pool that grows on demand.
        CudaAsync,

        /// Best-fit with coalescing allocator that preallocates 75% of the GPU memory, like JAX's default allocator.
        Bfc,
    }

    impl Allocator {
        /// Returns the command-line name of this allocator.
        fn name(self) -> &'static str {
            match self {
                Self::CudaAsync => "cuda-async",
                Self::Bfc => "bfc",
            }
        }

        /// Returns the PJRT allocator configuration of this allocator.
        fn to_pjrt(self) -> GpuMemoryAllocator {
            match self {
                Self::CudaAsync => GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
                // Growing on demand keeps this safe on machines whose GPU shares memory with the host.
                Self::Bfc => GpuMemoryAllocator::BFC { memory_fraction: 0.75, preallocate: false },
            }
        }
    }

    /// Model shape and benchmark settings, all of which are reported in the JSON output.
    #[derive(Clone, Debug)]
    struct Configuration {
        layers: usize,
        d_model: usize,
        heads: usize,
        d_ff: usize,
        vocab: usize,
        max_seq: usize,
        batch: usize,
        seq: usize,
        warmup: usize,
        iterations: usize,
        fused_attention: bool,
        primitive_softmax: bool,
        composite_softmax: bool,
        jit: bool,
        profile: bool,
        allocator: Option<Allocator>,
    }

    impl Configuration {
        /// Returns the default (full-size) configuration.
        fn full() -> Self {
            Self {
                layers: 20,
                d_model: 2048,
                heads: 16,
                d_ff: 8192,
                vocab: 32000,
                max_seq: 512,
                batch: 1,
                seq: 128,
                warmup: 3,
                iterations: 20,
                fused_attention: false,
                primitive_softmax: false,
                composite_softmax: false,
                jit: false,
                profile: false,
                allocator: None,
            }
        }

        /// Returns the small configuration selected by `--small`, which is meant for local CPU testing.
        fn small() -> Self {
            Self {
                layers: 2,
                d_model: 64,
                heads: 4,
                d_ff: 256,
                vocab: 1000,
                max_seq: 32,
                batch: 2,
                seq: 16,
                warmup: 1,
                iterations: 3,
                ..Self::full()
            }
        }

        /// Returns the size of each attention head.
        fn head_dimension(&self) -> usize {
            self.d_model / self.heads
        }

        /// Returns the number of trainable parameters of the model.
        fn parameter_count(&self) -> usize {
            let (d, f) = (self.d_model, self.d_ff);
            let layer = 4 * d + (d * 3 * d + 3 * d) + (d * d + d) + (d * f + f) + (f * d + d);
            self.vocab * d + self.max_seq * d + self.layers * layer + 2 * d + d * self.vocab
        }

        /// Renders this configuration as a JSON object.
        fn to_json(&self) -> String {
            format!(
                "{{\"layers\":{},\"d_model\":{},\"heads\":{},\"d_ff\":{},\"vocab\":{},\"max_seq\":{},\"batch\":{},\
                 \"seq\":{},\"warmup\":{},\"iterations\":{},\"fused_attention\":{},\"primitive_softmax\":{},\
                 \"composite_softmax\":{},\"jit\":{},\"profile\":{},\"allocator\":{}}}",
                self.layers,
                self.d_model,
                self.heads,
                self.d_ff,
                self.vocab,
                self.max_seq,
                self.batch,
                self.seq,
                self.warmup,
                self.iterations,
                self.fused_attention,
                self.primitive_softmax,
                self.composite_softmax,
                self.jit,
                self.profile,
                self.allocator.map_or("null".to_string(), |allocator| json_string(allocator.name())),
            )
        }
    }

    /// Returns the usage message of this benchmark.
    fn usage() -> &'static str {
        "Usage: transformer_benchmark [OPTIONS]\n\
         \n\
         Options:\n\
           --layers N          Number of decoder layers (default: 20)\n\
           --d-model N         Model width (default: 2048)\n\
           --heads N           Number of attention heads (default: 16)\n\
           --d-ff N            Hidden width of the MLP blocks (default: 8192)\n\
           --vocab N           Vocabulary size (default: 32000)\n\
           --max-seq N         Rows of the position embedding (default: 512)\n\
           --batch N           Batch size (default: 1)\n\
           --seq N             Sequence length (default: 128)\n\
           --warmup N          Untimed forward passes (default: 3)\n\
           --iterations N      Timed forward passes (default: 20)\n\
           --small             Use a small configuration for local testing (explicit options still apply)\n\
           --fused-attention   Use the `dot_product_attention` composite in every attention block\n\
           --primitive-softmax Compute softmax as `exp(x - max(x)) / sum(exp(x - max(x)))`\n\
           --composite-softmax Call Ryft's `softmax` as one held composite instead of its primitives\n\
           --jit               Compile the whole forward pass once with `jit` and time calls of the compiled function\n\
           --allocator NAME    GPU memory allocator of a CUDA build, `cuda-async` or `bfc` (default: that of\n\
                               `Session::new`)\n\
           --profile           Report a per-category breakdown of eager operations after the timed passes\n\
           -h, --help          Print this help"
    }

    /// Parses the command-line arguments, returning [`None`] when only the usage message was requested.
    fn parse_arguments() -> Result<Option<Configuration>, String> {
        let arguments = std::env::args().skip(1).collect::<Vec<_>>();
        let mut configuration = if arguments.iter().any(|argument| argument == "--small") {
            Configuration::small()
        } else {
            Configuration::full()
        };
        let mut values = arguments.iter();
        while let Some(argument) = values.next() {
            let mut value = |name: &str| -> Result<usize, String> {
                values
                    .next()
                    .ok_or(format!("expected a value after `{name}`"))?
                    .parse()
                    .map_err(|error| format!("invalid `{name}` value: {error}"))
            };
            match argument.as_str() {
                "-h" | "--help" => {
                    eprintln!("{}", usage());
                    return Ok(None);
                }
                "--layers" => configuration.layers = value("--layers")?,
                "--d-model" => configuration.d_model = value("--d-model")?,
                "--heads" => configuration.heads = value("--heads")?,
                "--d-ff" => configuration.d_ff = value("--d-ff")?,
                "--vocab" => configuration.vocab = value("--vocab")?,
                "--max-seq" => configuration.max_seq = value("--max-seq")?,
                "--batch" => configuration.batch = value("--batch")?,
                "--seq" => configuration.seq = value("--seq")?,
                "--warmup" => configuration.warmup = value("--warmup")?,
                "--iterations" => configuration.iterations = value("--iterations")?,
                "--small" => {}
                "--fused-attention" => configuration.fused_attention = true,
                "--primitive-softmax" => configuration.primitive_softmax = true,
                "--composite-softmax" => configuration.composite_softmax = true,
                "--jit" => configuration.jit = true,
                "--profile" => configuration.profile = true,
                "--allocator" => {
                    configuration.allocator = Some(match values.next().map(String::as_str) {
                        Some("cuda-async") => Allocator::CudaAsync,
                        Some("bfc") => Allocator::Bfc,
                        Some(other) => {
                            return Err(format!(
                                "invalid `--allocator` value `{other}`; expected `cuda-async` or `bfc`"
                            ));
                        }
                        None => return Err("expected a value after `--allocator`".into()),
                    })
                }
                other => return Err(format!("unknown argument `{other}`\n\n{}", usage())),
            }
        }
        if configuration.heads == 0 || configuration.d_model % configuration.heads != 0 {
            return Err("`--heads` must be positive and divide `--d-model`".into());
        }
        if configuration.seq == 0 || configuration.seq > configuration.max_seq {
            return Err("`--seq` must be positive and at most `--max-seq`".into());
        }
        if configuration.iterations == 0 {
            return Err("`--iterations` must be positive".into());
        }
        if configuration.jit && configuration.profile {
            return Err("`--profile` measures eager operations and cannot be combined with `--jit`".into());
        }
        if configuration.allocator.is_some() && !Platform::Cuda.is_available() {
            return Err(
                "`--allocator` requires a CUDA build of this benchmark (e.g., with `--features cuda-13`)".into()
            );
        }
        Ok(Some(configuration))
    }

    /// Returns the next output of the SplitMix64 generator, which is used to derive independent stream seeds.
    fn split_mix_64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = *state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Returns native-endian `f32` bytes of `count` deterministic values drawn uniformly from `[-0.02, 0.02]` with
    /// per-chunk xorshift64 streams derived from `seed`. Chunks are filled in parallel, directly in the byte buffer.
    fn uniform_bytes(count: usize, seed: u64) -> Vec<u8> {
        const CHUNK_SIZE: usize = 1 << 20;
        let mut bytes = vec![0u8; count * size_of::<f32>()];
        let threads = std::thread::available_parallelism().map_or(1, |threads| threads.get());
        let chunks = bytes.chunks_mut(CHUNK_SIZE * size_of::<f32>()).enumerate().collect::<Vec<_>>();
        let chunks_per_thread = chunks.len().div_ceil(threads).max(1);
        let mut chunks = chunks.into_iter();
        std::thread::scope(|scope| {
            loop {
                let group = chunks.by_ref().take(chunks_per_thread).collect::<Vec<_>>();
                if group.is_empty() {
                    break;
                }
                scope.spawn(move || {
                    for (index, chunk) in group {
                        let mut seed_state = seed ^ (index as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
                        let mut state = split_mix_64(&mut seed_state) | 1;
                        for element in chunk.chunks_exact_mut(size_of::<f32>()) {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            let unit = (state >> 40) as f32 * (1.0 / (1u64 << 24) as f32);
                            element.copy_from_slice(&((unit * 2.0 - 1.0) * 0.02).to_ne_bytes());
                        }
                    }
                });
            }
        });
        bytes
    }

    /// Returns native-endian bytes of `count` copies of `value`.
    fn constant_bytes(count: usize, value: f32) -> Vec<u8> {
        value.to_ne_bytes().repeat(count)
    }

    /// Uploads dense row-major host bytes as a replicated array on the session's device.
    struct Uploader<'s> {
        session: &'s Session,
        mesh: DeviceMesh,
        next_seed: u64,
    }

    impl Uploader<'_> {
        /// Uploads `bytes` as a replicated array of type `data_type` and shape `dimensions`.
        fn upload(&self, data_type: DataType, dimensions: &[usize], bytes: &[u8]) -> BenchmarkResult<Array> {
            let r#type = ArrayType::new_static(data_type, dimensions.to_vec()).replicated(&self.mesh)?;
            Ok(Array::from_host_buffer(self.session, r#type, self.mesh.clone(), None, bytes)?)
        }

        /// Uploads a deterministic uniform `f32` parameter of shape `dimensions`.
        fn uniform(&mut self, dimensions: &[usize]) -> BenchmarkResult<Array> {
            self.next_seed += 1;
            let bytes = uniform_bytes(dimensions.iter().product(), self.next_seed);
            self.upload(DataType::F32, dimensions, &bytes)
        }

        /// Uploads an `f32` array of shape `dimensions` whose elements all equal `value`.
        fn constant(&self, dimensions: &[usize], value: f32) -> BenchmarkResult<Array> {
            self.upload(DataType::F32, dimensions, &constant_bytes(dimensions.iter().product(), value))
        }

        /// Uploads an `f32` scalar.
        fn scalar(&self, value: f32) -> BenchmarkResult<Array> {
            self.constant(&[], value)
        }
    }

    /// Parameters of one pre-LayerNorm decoder layer.
    #[derive(Clone, Debug, PartialEq, Eq, Hash, Parameterized)]
    struct Layer<P: Parameter> {
        ln1_scale: P,
        ln1_shift: P,
        attention_qkv_weight: P,
        attention_qkv_bias: P,
        attention_output_weight: P,
        attention_output_bias: P,
        ln2_scale: P,
        ln2_shift: P,
        mlp_weight1: P,
        mlp_bias1: P,
        mlp_weight2: P,
        mlp_bias2: P,
    }

    /// Parameters of the whole model.
    #[derive(Clone, Debug, PartialEq, Eq, Hash, Parameterized)]
    struct Model<P: Parameter> {
        token_embedding: P,
        position_embedding: P,
        layers: Vec<Layer<P>>,
        final_scale: P,
        final_shift: P,
        unembedding: P,
    }

    /// Inputs and scalar constants of the forward pass, all of which are uploaded once before timing and all of which
    /// are runtime inputs of the compiled function in `--jit` mode.
    #[derive(Clone, Debug, PartialEq, Eq, Hash, Parameterized)]
    struct Inputs<P: Parameter> {
        /// Token identifiers of shape `[batch, seq, 1]`, whose trailing unit axis is the gather index vector.
        tokens: P,

        /// Additive causal mask of shape `[seq, seq]`.
        mask: P,

        /// `d_model` as an `f32` scalar.
        d_model: P,

        /// LayerNorm epsilon (i.e., `1e-5`).
        epsilon: P,

        /// Attention score scale `1 / sqrt(d_model / heads)`.
        attention_scale: P,

        /// `sqrt(2)`, by which GELU divides its input.
        sqrt_2: P,

        /// One.
        one: P,

        /// One half.
        half: P,
    }

    /// Dot and gather dimension numbers of the forward pass.
    struct Plan {
        /// `[b, s, k] x [k, n] -> [b, s, n]`.
        projection: DotDimensionNumbers,

        /// `[b, h, s, e] x [b, h, t, e] -> [b, h, s, t]`, which contracts the head dimensions of the queries and keys
        /// directly instead of transposing the keys.
        scores: DotDimensionNumbers,

        /// `[b, h, s, t] x [b, h, t, e] -> [b, h, s, e]`.
        weighted_sum: DotDimensionNumbers,

        /// Embedding gather dimension numbers, which select rows of the `[vocab, d]` table with `[b, s, 1]` indices.
        embedding: GatherDimensionNumbers,
    }

    /// Category of the operations that the forward pass calls, by which `--profile` attributes time.
    #[derive(Copy, Clone, Debug)]
    enum Category {
        Dot,
        Add,
        Sub,
        Mul,
        Div,
        Reduce,
        Reshape,
        Transpose,
        Slice,
        Gather,
        Rsqrt,
        Erf,
        Exp,
        Softmax,
        Attention,
        HostReadback,
    }

    impl Category {
        /// Every category, in the order of their discriminants.
        const ALL: [Self; 16] = [
            Self::Dot,
            Self::Add,
            Self::Sub,
            Self::Mul,
            Self::Div,
            Self::Reduce,
            Self::Reshape,
            Self::Transpose,
            Self::Slice,
            Self::Gather,
            Self::Rsqrt,
            Self::Erf,
            Self::Exp,
            Self::Softmax,
            Self::Attention,
            Self::HostReadback,
        ];

        /// Returns the JSON name of this category.
        fn name(self) -> &'static str {
            match self {
                Self::Dot => "dot",
                Self::Add => "add",
                Self::Sub => "sub",
                Self::Mul => "mul",
                Self::Div => "div",
                Self::Reduce => "reduce",
                Self::Reshape => "reshape",
                Self::Transpose => "transpose",
                Self::Slice => "slice",
                Self::Gather => "gather",
                Self::Rsqrt => "rsqrt",
                Self::Erf => "erf",
                Self::Exp => "exp",
                Self::Softmax => "softmax",
                Self::Attention => "attention_composite",
                Self::HostReadback => "host_readback",
            }
        }
    }

    /// Observer of the operations of a forward pass over values of type `V`.
    trait Recorder<V> {
        /// Runs `operation`, which belongs to `category`, and returns its output.
        fn record(
            &self,
            category: Category,
            operation: impl FnOnce() -> Result<V, ProgramError>,
        ) -> Result<V, XlaError>;

        /// Runs the host-side `operation`, which belongs to `category`, and returns its output.
        fn record_host<T>(
            &self,
            _category: Category,
            operation: impl FnOnce() -> Result<T, XlaError>,
        ) -> Result<T, XlaError> {
            operation()
        }
    }

    /// [`Recorder`] that only runs the operations, which the timed passes and the `jit` trace use.
    struct Unrecorded;

    impl<V> Recorder<V> for Unrecorded {
        #[inline(always)]
        fn record(
            &self,
            _category: Category,
            operation: impl FnOnce() -> Result<V, ProgramError>,
        ) -> Result<V, XlaError> {
            Ok(operation()?)
        }
    }

    /// [`Recorder`] that accumulates the call count and the wall time of the eager operations of each category.
    struct Profiler {
        /// Whether every operation waits for its output with [`Array::block_until_ready`] before it returns, so that
        /// device time is attributed to the operation that caused it.
        synchronize: bool,

        /// Call count and total wall time of each category, indexed by its discriminant.
        totals: RefCell<[(u64, Duration); Category::ALL.len()]>,
    }

    impl Profiler {
        /// Returns a profiler without any recorded operations.
        fn new(synchronize: bool) -> Self {
            Self { synchronize, totals: RefCell::new([(0, Duration::ZERO); Category::ALL.len()]) }
        }

        /// Runs `operation`, waits for its output with `wait` if this profiler synchronizes, and records the elapsed
        /// time under `category`.
        fn measure<T>(
            &self,
            category: Category,
            operation: impl FnOnce() -> Result<T, XlaError>,
            wait: impl FnOnce(&T) -> Result<(), XlaError>,
        ) -> Result<T, XlaError> {
            let start = Instant::now();
            let output = operation()?;
            if self.synchronize {
                wait(&output)?;
            }
            let elapsed = start.elapsed();
            let total = &mut self.totals.borrow_mut()[category as usize];
            total.0 += 1;
            total.1 += elapsed;
            Ok(output)
        }
    }

    impl Recorder<Array> for Profiler {
        fn record(
            &self,
            category: Category,
            operation: impl FnOnce() -> Result<Array, ProgramError>,
        ) -> Result<Array, XlaError> {
            self.measure(category, || Ok(operation()?), Array::block_until_ready)
        }

        fn record_host<T>(
            &self,
            category: Category,
            operation: impl FnOnce() -> Result<T, XlaError>,
        ) -> Result<T, XlaError> {
            self.measure(category, operation, |_| Ok(()))
        }
    }

    /// Creates and uploads the model parameters.
    fn initialize_model(configuration: &Configuration, uploader: &mut Uploader<'_>) -> BenchmarkResult<Model<Array>> {
        let (d, f) = (configuration.d_model, configuration.d_ff);
        let token_embedding = uploader.uniform(&[configuration.vocab, d])?;
        let position_embedding = uploader.uniform(&[configuration.max_seq, d])?;
        let layers = (0..configuration.layers)
            .map(|layer| {
                eprintln!("initializing layer {}/{}", layer + 1, configuration.layers);
                Ok(Layer {
                    ln1_scale: uploader.constant(&[d], 1.0)?,
                    ln1_shift: uploader.constant(&[d], 0.0)?,
                    attention_qkv_weight: uploader.uniform(&[d, 3 * d])?,
                    attention_qkv_bias: uploader.constant(&[3 * d], 0.0)?,
                    attention_output_weight: uploader.uniform(&[d, d])?,
                    attention_output_bias: uploader.constant(&[d], 0.0)?,
                    ln2_scale: uploader.constant(&[d], 1.0)?,
                    ln2_shift: uploader.constant(&[d], 0.0)?,
                    mlp_weight1: uploader.uniform(&[d, f])?,
                    mlp_bias1: uploader.constant(&[f], 0.0)?,
                    mlp_weight2: uploader.uniform(&[f, d])?,
                    mlp_bias2: uploader.constant(&[d], 0.0)?,
                })
            })
            .collect::<BenchmarkResult<Vec<_>>>()?;
        let final_scale = uploader.constant(&[d], 1.0)?;
        let final_shift = uploader.constant(&[d], 0.0)?;
        let unembedding = uploader.uniform(&[d, configuration.vocab])?;
        Ok(Model { token_embedding, position_embedding, layers, final_scale, final_shift, unembedding })
    }

    /// Creates and uploads the inputs and scalar constants of the forward pass.
    fn initialize_inputs(configuration: &Configuration, uploader: &Uploader<'_>) -> BenchmarkResult<Inputs<Array>> {
        let (batch, seq) = (configuration.batch, configuration.seq);
        let tokens = (0..batch * seq)
            .flat_map(|index| ((index * 7919 % configuration.vocab) as i32).to_ne_bytes())
            .collect::<Vec<_>>();
        let mask = (0..seq * seq)
            .flat_map(|index| (if index % seq <= index / seq { 0.0f32 } else { -1e9f32 }).to_ne_bytes())
            .collect::<Vec<_>>();
        Ok(Inputs {
            tokens: uploader.upload(DataType::I32, &[batch, seq, 1], &tokens)?,
            mask: uploader.upload(DataType::F32, &[seq, seq], &mask)?,
            d_model: uploader.scalar(configuration.d_model as f32)?,
            epsilon: uploader.scalar(1e-5)?,
            attention_scale: uploader.scalar((1.0 / (configuration.head_dimension() as f64).sqrt()) as f32)?,
            sqrt_2: uploader.scalar(std::f32::consts::SQRT_2)?,
            one: uploader.scalar(1.0)?,
            half: uploader.scalar(0.5)?,
        })
    }

    /// Computes `(x - mean) * rsqrt(var + epsilon) * scale + shift` over the last axis of the rank-3 array `x`.
    fn layer_norm<V: ArrayOperations, R: Recorder<V>>(
        x: &V,
        scale: &V,
        shift: &V,
        inputs: &Inputs<V>,
        configuration: &Configuration,
        recorder: &R,
    ) -> Result<V, XlaError> {
        let keep_dims = [configuration.batch, configuration.seq, 1];
        let sum = recorder.record(Category::Reduce, || x.reduce(&[2], ReductionKind::Sum))?;
        let sum = recorder.record(Category::Reshape, || sum.reshape(keep_dims))?;
        let mean = recorder.record(Category::Div, || Div::div(&sum, &inputs.d_model))?;
        let centered = recorder.record(Category::Sub, || Sub::sub(x, &mean))?;
        let squared = recorder.record(Category::Mul, || Mul::mul(&centered, &centered))?;
        let sum = recorder.record(Category::Reduce, || squared.reduce(&[2], ReductionKind::Sum))?;
        let sum = recorder.record(Category::Reshape, || sum.reshape(keep_dims))?;
        let variance = recorder.record(Category::Div, || Div::div(&sum, &inputs.d_model))?;
        let variance = recorder.record(Category::Add, || Add::add(&variance, &inputs.epsilon))?;
        let inverse_deviation = recorder.record(Category::Rsqrt, || variance.rsqrt())?;
        let normalized = recorder.record(Category::Mul, || Mul::mul(&centered, &inverse_deviation))?;
        let scaled = recorder.record(Category::Mul, || Mul::mul(&normalized, scale))?;
        recorder.record(Category::Add, || Add::add(&scaled, shift))
    }

    /// Computes `x @ weight + bias` for a rank-3 `x`.
    fn linear<V: ArrayOperations, R: Recorder<V>>(
        x: &V,
        weight: &V,
        bias: &V,
        plan: &Plan,
        recorder: &R,
    ) -> Result<V, XlaError> {
        let projected = recorder.record(Category::Dot, || x.dot(weight, &plan.projection))?;
        recorder.record(Category::Add, || Add::add(&projected, bias))
    }

    /// Applies the causal self-attention block to the normalized activations `h` and returns its output projection.
    fn attention<V: ArrayOperations, R: Recorder<V>>(
        h: &V,
        layer: &Layer<V>,
        inputs: &Inputs<V>,
        plan: &Plan,
        attention_function: Functions<'_, V>,
        configuration: &Configuration,
        recorder: &R,
    ) -> Result<V, XlaError>
    where
        V::Type: ArrayMemberType,
    {
        let (b, s, d, n, e) = (
            configuration.batch,
            configuration.seq,
            configuration.d_model,
            configuration.heads,
            configuration.head_dimension(),
        );
        let qkv = linear(h, &layer.attention_qkv_weight, &layer.attention_qkv_bias, plan, recorder)?;
        let split = |start: usize| -> Result<V, XlaError> {
            let part = recorder.record(Category::Slice, || qkv.slice_axis(2, start, start + d, 1))?;
            recorder.record(Category::Reshape, || part.reshape([b, s, n, e]))
        };
        let (q, k, v) = (split(0)?, split(d)?, split(2 * d)?);
        let a = match attention_function.attention {
            // The composite takes `[b, s, heads, head_dimension]` inputs and returns its output in the same layout.
            Some(attention_function) => {
                recorder.record(Category::Attention, || attention_function(AttentionInputs::new(q, k, v)))?
            }
            None => {
                let to_heads = |x: &V| recorder.record(Category::Transpose, || x.transpose([0, 2, 1, 3]));
                let (q, k, v) = (to_heads(&q)?, to_heads(&k)?, to_heads(&v)?);
                let scores = recorder.record(Category::Dot, || q.dot(&k, &plan.scores))?;
                let scores = recorder.record(Category::Mul, || Mul::mul(&scores, &inputs.attention_scale))?;
                let scores = recorder.record(Category::Add, || Add::add(&scores, &inputs.mask))?;
                let p = if configuration.primitive_softmax {
                    let keep_dims = [b, n, s, 1];
                    let max = recorder.record(Category::Reduce, || scores.reduce(&[3], ReductionKind::Max))?;
                    let max = recorder.record(Category::Reshape, || max.reshape(keep_dims))?;
                    let shifted = recorder.record(Category::Sub, || Sub::sub(&scores, &max))?;
                    let unnormalized = recorder.record(Category::Exp, || shifted.exp())?;
                    let sum = recorder.record(Category::Reduce, || unnormalized.reduce(&[3], ReductionKind::Sum))?;
                    let sum = recorder.record(Category::Reshape, || sum.reshape(keep_dims))?;
                    recorder.record(Category::Div, || Div::div(&unnormalized, &sum))?
                } else if let Some(softmax_function) = attention_function.softmax {
                    recorder.record(Category::Softmax, || softmax_function(scores.clone()))?
                } else {
                    recorder.record(Category::Softmax, || scores.softmax(3))?
                };
                let weighted = recorder.record(Category::Dot, || p.dot(&v, &plan.weighted_sum))?;
                to_heads(&weighted)?
            }
        };
        let a = recorder.record(Category::Reshape, || a.reshape([b, s, d]))?;
        linear(&a, &layer.attention_output_weight, &layer.attention_output_bias, plan, recorder)
    }

    /// Applies the GELU MLP block (with the exact `erf` form of GELU) to the normalized activations `h`.
    fn mlp<V: ArrayOperations, R: Recorder<V>>(
        h: &V,
        layer: &Layer<V>,
        inputs: &Inputs<V>,
        plan: &Plan,
        recorder: &R,
    ) -> Result<V, XlaError> {
        let m = linear(h, &layer.mlp_weight1, &layer.mlp_bias1, plan, recorder)?;
        let gate = recorder.record(Category::Div, || Div::div(&m, &inputs.sqrt_2))?;
        let gate = recorder.record(Category::Erf, || gate.erf())?;
        let gate = recorder.record(Category::Add, || Add::add(&gate, &inputs.one))?;
        let m = recorder.record(Category::Mul, || Mul::mul(&m, &inputs.half))?;
        let m = recorder.record(Category::Mul, || Mul::mul(&m, &gate))?;
        linear(&m, &layer.mlp_weight2, &layer.mlp_bias2, plan, recorder)
    }

    /// Runs one forward pass over arrays or tracers and returns the scalar sum of all logits.
    fn forward<V: ArrayOperations, R: Recorder<V>>(
        model: &Model<V>,
        inputs: &Inputs<V>,
        plan: &Plan,
        attention_function: Functions<'_, V>,
        configuration: &Configuration,
        recorder: &R,
    ) -> Result<V, XlaError>
    where
        V::Type: ArrayMemberType,
    {
        let embedding_slice_sizes = [1, configuration.d_model];
        let mut x = recorder.record(Category::Gather, || {
            model
                .token_embedding
                .gather(&inputs.tokens, &plan.embedding, &embedding_slice_sizes, &GatherOptions::new())
        })?;
        let positions =
            recorder.record(Category::Slice, || model.position_embedding.slice_axis(0, 0, configuration.seq, 1))?;
        x = recorder.record(Category::Add, || Add::add(&x, &positions))?;
        for layer in &model.layers {
            let h = layer_norm(&x, &layer.ln1_scale, &layer.ln1_shift, inputs, configuration, recorder)?;
            let a = attention(&h, layer, inputs, plan, attention_function, configuration, recorder)?;
            x = recorder.record(Category::Add, || Add::add(&x, &a))?;
            let h = layer_norm(&x, &layer.ln2_scale, &layer.ln2_shift, inputs, configuration, recorder)?;
            let m = mlp(&h, layer, inputs, plan, recorder)?;
            x = recorder.record(Category::Add, || Add::add(&x, &m))?;
        }
        let x = layer_norm(&x, &model.final_scale, &model.final_shift, inputs, configuration, recorder)?;
        let logits = recorder.record(Category::Dot, || x.dot(&model.unembedding, &plan.projection))?;
        recorder.record(Category::Reduce, || logits.reduce(&[0, 1, 2], ReductionKind::Sum))
    }

    /// Copies the `f32` scalar `sum` to the host.
    fn read_scalar<R: Recorder<Array>>(sum: &Array, recorder: &R) -> BenchmarkResult<f32> {
        let bytes = recorder.record_host(Category::HostReadback, || sum.copy_to_host())?;
        let bytes: [u8; 4] = bytes.as_slice().try_into().map_err(|_| "the logit sum is not a single `f32`")?;
        Ok(f32::from_ne_bytes(bytes))
    }

    /// Runs one eager forward pass and returns the sum of all logits, copied to the host.
    fn eager_forward<R: Recorder<Array>>(
        model: &Model<Array>,
        inputs: &Inputs<Array>,
        plan: &Plan,
        attention_function: Functions<'_, Array>,
        configuration: &Configuration,
        recorder: &R,
    ) -> BenchmarkResult<f32> {
        let sum = forward(model, inputs, plan, attention_function, configuration, recorder)?;
        read_scalar(&sum, recorder)
    }

    /// Renders `value` as a JSON number, or as `null` if it is not finite.
    fn json_number(value: f64) -> String {
        if value.is_finite() { format!("{value}") } else { "null".to_string() }
    }

    /// Renders `value` as a JSON string.
    fn json_string(value: &str) -> String {
        let mut rendered = String::with_capacity(value.len() + 2);
        rendered.push('"');
        for character in value.chars() {
            match character {
                '"' => rendered.push_str("\\\""),
                '\\' => rendered.push_str("\\\\"),
                character if character.is_control() => rendered.push_str(&format!("\\u{:04x}", character as u32)),
                character => rendered.push(character),
            }
        }
        rendered.push('"');
        rendered
    }

    /// Returns the total number of eager dispatch cache lookups of `session`, which is one per dispatched operation.
    fn eager_dispatches(session: &Session) -> u64 {
        let statistics = session.eager_dispatch_statistics();
        statistics.hits + statistics.misses
    }

    /// Runs [`PROFILE_PASSES`] eager forward passes twice, first without and then with synchronization after every
    /// operation, and renders the per-forward call count and wall time of every category as a JSON object.
    fn profile(
        model: &Model<Array>,
        inputs: &Inputs<Array>,
        plan: &Plan,
        attention_function: Functions<'_, Array>,
        configuration: &Configuration,
    ) -> BenchmarkResult<String> {
        let run = |synchronize: bool| -> BenchmarkResult<Profiler> {
            let profiler = Profiler::new(synchronize);
            for _ in 0..PROFILE_PASSES {
                eager_forward(model, inputs, plan, attention_function, configuration, &profiler)?;
            }
            Ok(profiler)
        };
        let dispatch = run(false)?.totals.into_inner();
        let synchronized = run(true)?.totals.into_inner();
        let per_forward = |duration: Duration| duration.as_secs_f64() * 1e3 / PROFILE_PASSES as f64;
        let categories = Category::ALL
            .iter()
            .filter(|category| dispatch[**category as usize].0 > 0)
            .map(|category| {
                let (count, dispatch) = dispatch[*category as usize];
                let synchronized = synchronized[*category as usize].1;
                format!(
                    "{}:{{\"count\":{},\"dispatch_ms\":{},\"synchronized_ms\":{}}}",
                    json_string(category.name()),
                    count / PROFILE_PASSES as u64,
                    json_number(per_forward(dispatch)),
                    json_number(per_forward(synchronized)),
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let total = |totals: &[(u64, Duration)]| per_forward(totals.iter().map(|(_, duration)| *duration).sum());
        Ok(format!(
            "{{\"passes\":{PROFILE_PASSES},\"synchronization\":\"Array::block_until_ready\",\
             \"categories\":{{{categories}}},\"dispatch_total_ms\":{},\"synchronized_total_ms\":{}}}",
            json_number(total(&dispatch)),
            json_number(total(&synchronized)),
        ))
    }

    /// Runs the benchmark and prints its JSON report.
    pub(super) fn run() -> BenchmarkResult<()> {
        let Some(configuration) = parse_arguments().map_err(std::io::Error::other)? else {
            return Ok(());
        };

        // Without `--allocator`, the session creates its own client on the best available platform, which is a CUDA
        // GPU when a CUDA feature is enabled and a GPU is present, and the CPU otherwise.
        let session = match configuration.allocator {
            None => Session::new()?,
            Some(allocator) => Session::builder()
                .platform(Platform::Cuda)
                .client_options(ClientOptions::GPU(GpuClientOptions {
                    platform: Some(GpuPlatform::CUDA),
                    allocator: allocator.to_pjrt(),
                    ..Default::default()
                }))
                .build()?,
        };
        let platform = session.target()?.platform_name().to_string();
        eprintln!("XLA platform: {platform}");
        let mesh = session.mesh([MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
        let mut uploader = Uploader { session: &session, mesh: mesh.clone(), next_seed: 0 };

        let parameter_count = configuration.parameter_count();
        eprintln!("initializing {parameter_count} parameters");
        let start = Instant::now();
        let model = initialize_model(&configuration, &mut uploader)?;
        let inputs = initialize_inputs(&configuration, &uploader)?;
        model.unembedding.block_until_ready()?;
        inputs.half.block_until_ready()?;
        eprintln!("initialized and uploaded parameters in {:.3} s", start.elapsed().as_secs_f64());

        let plan = Plan {
            projection: DotDimensionNumbers::new(vec![2], vec![0], Vec::new(), Vec::new()),
            scores: DotDimensionNumbers::new(vec![3], vec![3], vec![0, 1], vec![0, 1]),
            weighted_sum: DotDimensionNumbers::new(vec![3], vec![2], vec![0, 1], vec![0, 1]),
            embedding: GatherDimensionNumbers::new(vec![2], vec![0], vec![0]),
        };
        let dot_product_attention = configuration.fused_attention.then(|| {
            dot_product_attention::<AttentionDomain>(
                AttentionConfiguration::new()
                    .with_scale(1.0 / (configuration.head_dimension() as f64).sqrt())
                    .with_causal(true),
            )
        });
        let eager_attention = dot_product_attention
            .as_ref()
            .map(|attention| move |inputs: AttentionInputs<Array>| attention.call(inputs));
        let softmax_composite = configuration.composite_softmax.then(|| softmax::<AttentionDomain>(3));
        let eager_softmax = softmax_composite.as_ref().map(|softmax| move |scores: Array| softmax.call(scores));
        let eager_attention: Functions<'_, Array> = Functions {
            attention: eager_attention.as_ref().map(|attention| attention as &dyn Fn(_) -> _),
            softmax: eager_softmax.as_ref().map(|softmax| softmax as &dyn Fn(_) -> _),
        };

        // Every pass returns the logit sum on the host together with its timed duration.
        let mut compile_seconds = None;
        let mut pass: Box<dyn FnMut() -> BenchmarkResult<(f32, f64)> + '_> = if configuration.jit {
            let dispatcher: CompiledFunctionDispatcher<_, (), (Model<ArrayType>, Inputs<ArrayType>), ArrayType> = jit(
                |(), (model, inputs): (Model<CompileTracer>, Inputs<CompileTracer>)| {
                    let attention = dot_product_attention
                        .as_ref()
                        .map(|attention| move |inputs: AttentionInputs<CompileTracer>| attention.call(inputs));
                    let softmax =
                        softmax_composite.as_ref().map(|softmax| move |scores: CompileTracer| softmax.call(scores));
                    let attention: Functions<'_, CompileTracer> = Functions {
                        attention: attention.as_ref().map(|attention| attention as &dyn Fn(_) -> _),
                        softmax: softmax.as_ref().map(|softmax| softmax as &dyn Fn(_) -> _),
                    };
                    forward(&model, &inputs, &plan, attention, &configuration, &Unrecorded)
                },
                &session,
                mesh.clone(),
            );
            let (model, inputs) = (&model, &inputs);
            let call = move || -> BenchmarkResult<(f32, f64)> {
                let arguments = (model.clone(), inputs.clone());
                let start = Instant::now();
                let result = read_scalar(&dispatcher.execute((), arguments)?, &Unrecorded)?;
                Ok((result, start.elapsed().as_secs_f64()))
            };
            let (result, seconds) = call()?;
            eprintln!(
                "compile_seconds = {seconds:.3} s is the duration of the first call of the `jit` dispatcher, which \
                 stages, lowers, compiles, and executes the forward pass and copies its result to the host \
                 (result = {result})"
            );
            compile_seconds = Some(seconds);
            Box::new(call)
        } else {
            Box::new(|| {
                let start = Instant::now();
                let result = eager_forward(&model, &inputs, &plan, eager_attention, &configuration, &Unrecorded)?;
                Ok((result, start.elapsed().as_secs_f64()))
            })
        };

        let start = Instant::now();
        for iteration in 0..configuration.warmup {
            let (result, _) = pass()?;
            eprintln!("warmup {}/{}: result = {result}", iteration + 1, configuration.warmup);
        }
        let warmup_seconds = start.elapsed().as_secs_f64();
        eprintln!("warmup took {warmup_seconds:.3} s");

        let mut times = Vec::with_capacity(configuration.iterations);
        let mut result = f32::NAN;
        let mut eager_operations_per_forward = None;
        for iteration in 0..configuration.iterations {
            let dispatches = eager_dispatches(&session);
            let (pass_result, seconds) = pass()?;
            result = pass_result;
            let elapsed = seconds * 1e3;
            eager_operations_per_forward.get_or_insert(eager_dispatches(&session) - dispatches);
            eprintln!("iteration {}/{}: {elapsed:.3} ms (result = {result})", iteration + 1, configuration.iterations);
            times.push(elapsed);
        }
        drop(pass);

        let profile = if configuration.profile {
            profile(&model, &inputs, &plan, eager_attention, &configuration)?
        } else {
            "null".to_string()
        };

        let mut sorted = times.clone();
        sorted.sort_by(f64::total_cmp);
        let middle = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 { (sorted[middle - 1] + sorted[middle]) / 2.0 } else { sorted[middle] };
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let times = times.iter().map(|time| json_number(*time)).collect::<Vec<_>>().join(",");
        let framework = if configuration.jit { "ryft-xla-jit" } else { "ryft-xla-eager" };
        println!(
            "{{\"framework\":\"{framework}\",\"platform\":{},\"config\":{},\"parameter_count\":{parameter_count},\
             \"compile_seconds\":{},\"warmup_seconds\":{},\"times_ms\":[{times}],\"median_ms\":{},\"min_ms\":{},\
             \"mean_ms\":{},\"result\":{},\"eager_operations_per_forward\":{},\"profile\":{profile}}}",
            json_string(&platform),
            configuration.to_json(),
            compile_seconds.map_or("null".to_string(), json_number),
            json_number(warmup_seconds),
            json_number(median),
            json_number(sorted[0]),
            json_number(mean),
            json_number(f64::from(result)),
            eager_operations_per_forward.map_or("null".to_string(), |count| count.to_string()),
        );
        Ok(())
    }
}

/// Runs the benchmark, which requires the `xla` feature.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "xla")]
    {
        benchmark::run()
    }
    #[cfg(not(feature = "xla"))]
    {
        Err("this benchmark requires building the `ryft` crate with the `xla` feature".into())
    }
}
