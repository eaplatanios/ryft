//! Trains a small multi-layer perceptron with reverse-mode automatic differentiation.
//!
//! The model, its loss, and its training loop are written once over [`ArrayOperations`], which both homogeneous array
//! values and composite array IR values implement. The default runner uses the transparent `ryft-core` reference
//! array backend, and the `core-ir` runner trains the same model over composite array IR values with reference array
//! members:
//!
//! ```sh
//! cargo run -p ryft --no-default-features --example mlp
//! cargo run -p ryft --no-default-features --example mlp -- core-ir
//! ```
//!
//! Enabling `xla` adds XLA runners over XLA arrays (`xla`) and over composite array IR values with XLA array members
//! (`xla-ir`):
//!
//! ```sh
//! cargo run -p ryft --features xla --example mlp -- xla
//! cargo run -p ryft --features xla --example mlp -- xla-ir
//! ```
//!
//! If the crate is built with `cuda-12` or `cuda-13`, the XLA runner tries the corresponding CUDA PJRT plugin before
//! falling back to the built-in XLA CPU plugin:
//!
//! ```sh
//! cargo run -p ryft --no-default-features --features cuda-13 --example mlp -- xla
//! ```
//!
//! Replace `cuda-13` with `cuda-12` to use the CUDA 12 plugin. All runners optimize the same two-layer MLP and XOR
//! dataset. The differentiated closure takes only the model as its active argument; the dataset and loss scale are
//! supplied separately as nondifferentiated runtime captures. The `parity` argument runs every available runner and
//! checks that each composite runner matches its homogeneous counterpart at every step:
//!
//! ```sh
//! cargo run -p ryft --features xla --example mlp -- parity
//! ```

use ryft::*;

#[derive(Clone, Parameterized)]
struct Linear<P: Parameter> {
    weights: P,
    bias: Option<P>,
}

impl<P: Parameter> Linear<P> {
    fn new(weights: P, bias: Option<P>) -> Self {
        Self { weights, bias }
    }

    fn forward(&self, inputs: &P) -> Result<P, ProgramError>
    where
        P: ArrayOperations,
    {
        let outputs = inputs.dot(&self.weights, &DotDimensionNumbers::matmul())?;
        Ok(match &self.bias {
            Some(bias) => outputs + bias.clone(),
            None => outputs,
        })
    }
}

#[derive(Clone, Parameterized)]
struct Mlp<P: Parameter> {
    layers: Vec<Linear<P>>,
}

impl<P: Parameter> Mlp<P> {
    /// Applies each hidden layer followed by a hyperbolic tangent, then applies the final linear output layer.
    fn forward(&self, inputs: &P) -> Result<P, ProgramError>
    where
        P: ArrayOperations,
    {
        let (output_layer, hidden_layers) = self.layers.split_last().ok_or_else(|| ProgramError::InvalidArgument {
            message: "an MLP must contain at least one layer".to_string(),
        })?;
        let hidden = hidden_layers
            .iter()
            .try_fold(inputs.clone(), |activations, layer| layer.forward(&activations)?.tanh())?;
        output_layer.forward(&hidden)
    }
}

/// Number of full-batch gradient-descent steps.
const STEP_COUNT: usize = 300;

/// Result type used by backend adapters in this example.
type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Computes the mean squared error of the MLP predictions.
fn loss<A: ArrayOperations>(model: &Mlp<A>, inputs: &A, targets: &A, mean_scale: &A) -> Result<A, ProgramError> {
    let residuals = model.forward(inputs)? - targets.clone();
    Ok((residuals.clone() * residuals).reduce(&[0, 1], ReductionKind::Sum)? * mean_scale.clone())
}

/// Applies one gradient-descent update to all trainable arrays.
fn gradient_descent_step<A: ArrayOperations>(
    model: impl Into<Parameterwise<A, Mlp<A>>>,
    gradients: impl Into<Parameterwise<A, Mlp<A>>>,
    learning_rate: &A,
) -> Mlp<A> {
    (model.into() - gradients.into() * learning_rate.clone()).into_inner()
}

/// Execution context in which [`train`] computes the loss value and the gradient of an MLP over `A`.
trait TrainingDomain<A: Value>:
    ReverseModeDifferentiate<
        Operation: OperationProvider<
            Self::Type,
            ReferenceNewOperation<<Self::Type as ReferenceMemberType>::Referent, Self::Type>,
            Operation = Self::Operation,
        > + OperationProvider<
            Self::Type,
            ReferenceFreezeOperation<<Self::Type as ReferenceMemberType>::Referent, Self::Type>,
            Operation = Self::Operation,
        > + OperationProvider<Self::Type, OneOperation<Self::Type>, Operation = Self::Operation>,
    > + Zero<A>
{
}

impl<A: Value, C> TrainingDomain<A> for C where
    C: ReverseModeDifferentiate<
            Operation: OperationProvider<
                C::Type,
                ReferenceNewOperation<<C::Type as ReferenceMemberType>::Referent, C::Type>,
                Operation = C::Operation,
            > + OperationProvider<
                C::Type,
                ReferenceFreezeOperation<<C::Type as ReferenceMemberType>::Referent, C::Type>,
                Operation = C::Operation,
            > + OperationProvider<C::Type, OneOperation<C::Type>, Operation = C::Operation>,
        > + Zero<A>
{
}

/// Full-precision results of one training run, which the `parity` mode compares across runners.
struct Training {
    /// Loss of the model before each gradient-descent step.
    losses: Vec<f64>,

    /// Predictions of the trained model for every training example.
    predictions: Vec<f64>,
}

/// Trains an MLP in `context` using a backend adapter only for host value materialization, and checks that every loss
/// and prediction is finite and that the final loss is below `1e-3` and below 10% of the initial loss. The context is
/// explicit because composite values whose array members need session state (e.g., XLA arrays) cannot recover a
/// session-backed execution domain from every leaf (e.g., from a first-class dimension), and so free transforms do not
/// serve them.
fn train<A, C, ReadValues>(
    backend: &str,
    context: &C,
    mut model: Mlp<A>,
    inputs: A,
    targets: A,
    learning_rate: A,
    mean_scale: A,
    mut read_values: ReadValues,
) -> ExampleResult<Training>
where
    A: ArrayOperations<Type: DifferentiableType + ReferenceMemberType>,
    C: Context<Type = A::Type, Value = A> + TrainingDomain<A>,
    LinearizationTracer<C>: ArrayOperations,
    ReadValues: FnMut(&A) -> ExampleResult<Vec<f64>>,
{
    let mut losses = Vec::with_capacity(STEP_COUNT);
    for step in 0..STEP_COUNT {
        let (step_loss, gradients) = differentiate_at(model.clone())
            .in_context(context)
            .with_captures((inputs.clone(), targets.clone(), mean_scale.clone()))
            .value_and_gradient(|model, (inputs, targets, mean_scale)| loss(&model, &inputs, &targets, &mean_scale))?;
        let step_loss =
            read_values(&step_loss)?.first().copied().ok_or_else(|| format!("{backend} loss has no values"))?;
        if step % 50 == 0 || step + 1 == STEP_COUNT {
            println!("{backend} step {step:>3}: loss = {step_loss:.6}");
        }
        losses.push(step_loss);
        model = gradient_descent_step(model, gradients, &learning_rate);
    }

    let predictions = read_values(&model.forward(&inputs)?)?;
    let (initial_loss, final_loss) = (losses[0], losses[STEP_COUNT - 1]);
    if !losses.iter().chain(&predictions).all(|value| value.is_finite())
        || final_loss >= 1e-3
        || final_loss >= initial_loss * 0.1
    {
        return Err(
            format!("{backend} training did not converge: loss changed from {initial_loss} to {final_loss}").into()
        );
    }
    println!("{backend} predictions: {predictions:?}");
    Ok(Training { losses, predictions })
}

/// Checks that a composite runner matches its homogeneous counterpart at every step and on every prediction, using
/// `|a - b| <= atol + rtol * max(|a|, |b|)` with `atol = 1e-6` and `rtol = 1e-5` for losses and `atol = 1e-4` and
/// `rtol = 1e-4` for predictions.
fn check_parity(
    composite_backend: &str,
    composite: &Training,
    homogeneous_backend: &str,
    homogeneous: &Training,
) -> ExampleResult<()> {
    let pairs = [
        ("loss", &composite.losses, &homogeneous.losses, 1e-6, 1e-5),
        ("prediction", &composite.predictions, &homogeneous.predictions, 1e-4, 1e-4),
    ];
    for (kind, composite_values, homogeneous_values, absolute_tolerance, relative_tolerance) in pairs {
        if composite_values.len() != homogeneous_values.len() {
            return Err(
                format!("`{composite_backend}` and `{homogeneous_backend}` have different {kind} counts").into()
            );
        }
        for (index, (&left, &right)) in composite_values.iter().zip(homogeneous_values).enumerate() {
            if (left - right).abs() > absolute_tolerance + relative_tolerance * left.abs().max(right.abs()) {
                return Err(format!(
                    "{kind} {index} of `{composite_backend}` ({left}) differs from `{homogeneous_backend}` ({right})"
                )
                .into());
            }
        }
    }
    println!("`{composite_backend}` matches `{homogeneous_backend}`");
    Ok(())
}

/// Returns deterministic layer dimensions, weights, and optional biases shared by all runners.
fn initial_layer_values() -> [(usize, usize, Vec<f32>, Option<Vec<f32>>); 2] {
    [
        (2, 4, vec![0.5, -0.4, 0.3, 0.2, -0.3, 0.6, 0.2, -0.5], Some(vec![0.1, -0.1, 0.05, 0.0])),
        (4, 1, vec![0.4, -0.5, 0.3, 0.2], Some(vec![0.0])),
    ]
}

/// Returns the four XOR input examples in row-major order.
fn input_values() -> Vec<f32> {
    vec![-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
}

/// Returns the XOR targets in the output range centered around zero.
fn target_values() -> Vec<f32> {
    vec![-1.0, 1.0, 1.0, -1.0]
}

/// Runs MLP training with the `ryft-core` reference CPU array backend in `context`, over arrays lifted into the value
/// family `A` by `lift` (e.g., composite array IR values) and read back to the host by `read_values`.
fn run_core<A, C, ReadValues>(
    backend: &str,
    context: &C,
    lift: impl Fn(Array) -> A,
    read_values: ReadValues,
) -> ExampleResult<Training>
where
    A: ArrayOperations<Type: DifferentiableType + ReferenceMemberType>,
    C: Context<Type = A::Type, Value = A> + TrainingDomain<A>,
    LinearizationTracer<C>: ArrayOperations,
    ReadValues: FnMut(&A) -> ExampleResult<Vec<f64>>,
{
    let model = Mlp {
        layers: initial_layer_values()
            .into_iter()
            .map(|(input_size, output_size, weights, bias)| {
                let weights = lift(Array::matrix(input_size, output_size, weights)?);
                let bias = bias.map(Array::vector).transpose()?.map(&lift);
                Ok(Linear::new(weights, bias))
            })
            .collect::<Result<Vec<_>, ProgramError>>()?,
    };
    let inputs = lift(Array::matrix(4, 2, input_values())?);
    let targets = lift(Array::matrix(4, 1, target_values())?);
    let learning_rate = lift(Array::scalar(0.1f32)?);
    let mean_scale = lift(Array::scalar(0.25f32)?);
    train(backend, context, model, inputs, targets, learning_rate, mean_scale, read_values)
}

/// Runs MLP training with the `ryft-core` reference CPU array backend over homogeneous arrays.
fn run_core_arrays() -> ExampleResult<Training> {
    run_core("core", &EagerContext::<Array, ArrayOperation<Array>>::new(), |array| array, |array| Ok(array.to_f64s()))
}

/// Runs MLP training with the `ryft-core` reference CPU array backend over composite array IR values.
fn run_core_ir() -> ExampleResult<Training> {
    let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
    run_core("core-ir", &context, ArrayIrValue::Array, |value| match value {
        ArrayIrValue::Array(array) => Ok(array.to_f64s()),
        _ => Err("core-ir result is not an array".into()),
    })
}

#[cfg(feature = "xla")]
mod xla_backend {
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use ryft::pjrt::{ClientOptions, CpuClientOptions, Error as PjrtError, Plugin, load_cpu_plugin};
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    use ryft::pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform};
    use ryft::xla::{Array, FromPjrt, XlaDomain, XlaSession};
    use ryft::{
        ArrayIrValue, ArrayType, DataType, Device, DeviceMesh, Dimension, LogicalMesh, MeshAxis, MeshAxisType,
        Parameterized, ProjectedContext, Shape, Sharding,
    };

    use super::{ExampleResult, Linear, Mlp, Training, initial_layer_values, input_values, target_values};

    /// Converts a slice of `f32` values into native-endian host bytes for PJRT transfer.
    fn values_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    /// Constructs a replicated `f32` array on the selected XLA device.
    fn array<'c>(
        domain: &XlaDomain<'c>,
        mesh: &DeviceMesh,
        dimensions: &[usize],
        values: &[f32],
    ) -> ExampleResult<Array<'c>> {
        let shape = Shape::new(dimensions.iter().copied().map(Dimension::Static).collect());
        let r#type = ArrayType::new(DataType::F32, shape)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), dimensions.len()))?;
        Ok(Array::from_host_buffer(domain, r#type, mesh.clone(), values_to_bytes(values))?)
    }

    /// Copies a replicated `f32` XLA array back to the host.
    fn read_f32s(array: &Array<'_>) -> ExampleResult<Vec<f32>> {
        let shard =
            array.addressable_shards().next().ok_or_else(|| "xla result has no addressable shard".to_string())?;
        let bytes = shard
            .buffer()
            .ok_or_else(|| "xla result shard has no materialized buffer".to_string())?
            .copy_to_host(None)?
            .r#await()?;
        Ok(bytes
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect())
    }

    /// Attempts to load and initialize one CUDA plugin candidate. Plugin initialization can panic inside the PJRT
    /// wrapper, so this optional-probe boundary converts that failure into the same CPU fallback used for ordinary
    /// loading and client-creation errors.
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    fn try_cuda_plugin<L>(label: &str, load: L) -> Option<(Plugin, ClientOptions)>
    where
        L: FnOnce() -> Result<Plugin, PjrtError>,
    {
        let options = ClientOptions::GPU(GpuClientOptions {
            platform: Some(GpuPlatform::CUDA),
            allocator: GpuMemoryAllocator::CudaAsync { memory_fraction_to_preallocate: None },
            ..GpuClientOptions::default()
        });
        let candidate = catch_unwind(AssertUnwindSafe(|| -> Result<_, PjrtError> {
            let plugin = load()?;
            if plugin.client(options.clone())?.addressable_devices()?.is_empty() {
                Ok(None)
            } else {
                Ok(Some((plugin, options)))
            }
        }));
        match candidate {
            Ok(Ok(Some(candidate))) => Some(candidate),
            Ok(Ok(None)) => {
                eprintln!("{label} plugin has no addressable GPU; trying the next XLA platform");
                None
            }
            Ok(Err(error)) => {
                eprintln!("{label} plugin loading or initialization failed ({error}); trying the next XLA platform");
                None
            }
            Err(_) => {
                eprintln!("{label} plugin initialization panicked; trying the next XLA platform");
                None
            }
        }
    }

    /// Chooses a usable CUDA plugin when enabled and otherwise returns the built-in CPU plugin.
    fn load_xla_plugin() -> Result<(Plugin, ClientOptions), PjrtError> {
        #[cfg(feature = "cuda-13")]
        if let Some(candidate) = try_cuda_plugin("CUDA 13", ryft::pjrt::load_cuda_13_plugin) {
            return Ok(candidate);
        }

        #[cfg(feature = "cuda-12")]
        if let Some(candidate) = try_cuda_plugin("CUDA 12", ryft::pjrt::load_cuda_12_plugin) {
            return Ok(candidate);
        }

        Ok((load_cpu_plugin()?, ClientOptions::CPU(CpuClientOptions { device_count: Some(1), ..Default::default() })))
    }

    /// Runs MLP training through XLA on the selected PJRT platform, over XLA arrays or, if `composite` is set, over
    /// composite array IR values with XLA array members.
    pub(super) fn run(composite: bool) -> ExampleResult<Training> {
        let (plugin, client_options) = load_xla_plugin()?;
        let client = plugin.client(client_options)?;
        println!("XLA platform: {}", client.platform_name()?);
        let device = client
            .addressable_devices()?
            .into_iter()
            .next()
            .ok_or_else(|| "xla client has no addressable device".to_string())?;
        let device = Device::from_pjrt(device)?;
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
        let mesh = DeviceMesh::new(logical_mesh, vec![device])?;
        let domain = XlaSession::new(&client).domain();

        let model = Mlp {
            layers: initial_layer_values()
                .into_iter()
                .map(|(input_size, output_size, weights, bias)| -> ExampleResult<_> {
                    Ok(Linear::new(
                        array(&domain, &mesh, &[input_size, output_size], &weights)?,
                        bias.map(|bias| array(&domain, &mesh, &[output_size], &bias)).transpose()?,
                    ))
                })
                .collect::<Result<_, _>>()?,
        };
        let inputs = array(&domain, &mesh, &[4, 2], &input_values())?;
        let targets = array(&domain, &mesh, &[4, 1], &target_values())?;
        let learning_rate = array(&domain, &mesh, &[], &[0.1])?;
        let mean_scale = array(&domain, &mesh, &[], &[0.25])?;
        if !composite {
            return super::train(
                "xla",
                &ProjectedContext::new(domain.clone()),
                model,
                inputs,
                targets,
                learning_rate,
                mean_scale,
                |array| Ok(read_f32s(array)?.into_iter().map(f64::from).collect()),
            );
        }

        // Composite values over XLA arrays train in the session-backed composite domain itself.
        super::train(
            "xla-ir",
            &domain,
            model.map_parameters(ArrayIrValue::Array)?,
            ArrayIrValue::Array(inputs),
            ArrayIrValue::Array(targets),
            ArrayIrValue::Array(learning_rate),
            ArrayIrValue::Array(mean_scale),
            |value| match value {
                ArrayIrValue::Array(array) => Ok(read_f32s(array)?.into_iter().map(f64::from).collect()),
                _ => Err("xla-ir result is not an array".into()),
            },
        )
    }
}

/// Selects and runs the requested backend, or every available backend for `parity`.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    match std::env::args().nth(1).as_deref().unwrap_or("core") {
        "core" => run_core_arrays().map(|_| ()),
        "core-ir" => run_core_ir().map(|_| ()),
        #[cfg(feature = "xla")]
        "xla" => xla_backend::run(false).map(|_| ()),
        #[cfg(feature = "xla")]
        "xla-ir" => xla_backend::run(true).map(|_| ()),
        #[cfg(not(feature = "xla"))]
        "xla" | "xla-ir" => Err("the XLA backends require building this example with the `xla` feature".into()),
        "parity" => {
            let core = run_core_arrays()?;
            let core_ir = run_core_ir()?;
            check_parity("core-ir", &core_ir, "core", &core)?;
            #[cfg(feature = "xla")]
            {
                let xla = xla_backend::run(false)?;
                let xla_ir = xla_backend::run(true)?;
                check_parity("xla-ir", &xla_ir, "xla", &xla)?;
            }
            Ok(())
        }
        backend => {
            Err(format!("unknown backend `{backend}`; expected `core`, `core-ir`, `xla`, `xla-ir`, or `parity`").into())
        }
    }
}
