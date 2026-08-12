//! Trains a small multi-layer perceptron with reverse-mode automatic differentiation.
//!
//! The default runner uses the transparent `ryft-core` reference array backend:
//!
//! ```sh
//! cargo run -p ryft --no-default-features --example mlp
//! ```
//!
//! Enabling `xla` adds an XLA runner selected by the `xla` argument:
//!
//! ```sh
//! cargo run -p ryft --features xla --example mlp -- xla
//! ```
//!
//! If the crate is built with `cuda-12` or `cuda-13`, the XLA runner tries the corresponding CUDA PJRT plugin before
//! falling back to the built-in XLA CPU plugin:
//!
//! ```sh
//! cargo run -p ryft --no-default-features --features cuda-13 --example mlp -- xla
//! ```
//!
//! Replace `cuda-13` with `cuda-12` to use the CUDA 12 plugin. Both runners optimize the same two-layer MLP and XOR
//! dataset. As in JAX's usual `value_and_grad(lambda model: loss(model, data))` pattern, the differentiated closure
//! takes only the model; the dataset and loss scale are closed over as constant runtime values.

use std::ops::{Add, Mul, Sub};

use ryft::{
    Array, Context, DifferentiableType, DifferentiationDual, DifferentiationTracer, Domain, Dot, DotDimensionNumbers,
    LinearizationTracer, OneOperation, Parameter, Parameterized, PartialEvaluationValue, PartialTracer,
    PartiallyEvaluatableOperation, ProgramError, Reduce, ReductionKind, ReverseModeDifferentiate, Tanh, TracingContext,
    Value, Zero, value_and_gradient,
};

/// Trainable affine layer of an [`Mlp`].
#[derive(Clone, Parameterized)]
struct Linear<P: Parameter> {
    /// Weights mapping input features to output features.
    weights: P,

    /// Optional bias added to the output features.
    bias: Option<P>,
}

impl<P: Parameter> Linear<P> {
    /// Creates a linear layer.
    ///
    /// # Parameters
    ///
    ///   - `weights`: Weights mapping input features to output features.
    ///   - `bias`: Optional bias added to the output features.
    fn new(weights: P, bias: Option<P>) -> Self {
        Self { weights, bias }
    }

    /// Applies this layer's affine transformation to `inputs`.
    fn forward(&self, inputs: &P) -> P
    where
        P: Clone + Add<Output = P> + Dot,
    {
        let outputs = inputs.dot(&self.weights, &DotDimensionNumbers::matmul());
        match &self.bias {
            Some(bias) => outputs + bias.clone(),
            None => outputs,
        }
    }
}

/// Multi-layer perceptron represented as an ordered sequence of [`Linear`] layers.
#[derive(Clone, Parameterized)]
struct Mlp<P: Parameter> {
    /// Layers ordered from the input projection through the output projection.
    layers: Vec<Linear<P>>,
}

impl<P: Parameter> Mlp<P> {
    /// Applies each hidden layer followed by a hyperbolic tangent, then applies the final linear output layer.
    fn forward(&self, inputs: &P) -> Result<P, ProgramError>
    where
        P: Clone + Add<Output = P> + Dot + Tanh,
    {
        let (output_layer, hidden_layers) = self.layers.split_last().ok_or_else(|| ProgramError::InvalidArgument {
            message: "an MLP must contain at least one layer".to_string(),
        })?;
        let hidden = hidden_layers
            .iter()
            .try_fold(inputs.clone(), |activations, layer| layer.forward(&activations).tanh())?;
        Ok(output_layer.forward(&hidden))
    }

    /// Returns the first parameter, used to recover the execution domain for backend arrays.
    fn first_parameter(&self) -> Result<&P, ProgramError> {
        self.layers.first().map(|layer| &layer.weights).ok_or_else(|| ProgramError::InvalidArgument {
            message: "an MLP must contain at least one layer".to_string(),
        })
    }
}

/// Adapts closed-over data and scaling arrays into constant linearization values before computing [`loss`].
fn loss_with_captured_arguments<C: Context>(
    model: &Mlp<LinearizationTracer<C>>,
    inputs: C::Value,
    targets: C::Value,
    mean_scale: C::Value,
) -> Result<LinearizationTracer<C>, ProgramError>
where
    C::Type: DifferentiableType,
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    LinearizationTracer<C>: Clone
        + Parameter
        + Add<Output = LinearizationTracer<C>>
        + Sub<Output = LinearizationTracer<C>>
        + Mul<Output = LinearizationTracer<C>>
        + Dot
        + Reduce
        + Tanh,
{
    let context = model.first_parameter()?.context().clone();
    let constant = |value| {
        let primal = PartialTracer::new(context.parent().clone(), PartialEvaluationValue::known_input(value));
        DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(primal), context.clone())
    };
    loss(model, &constant(inputs), &constant(targets), &constant(mean_scale))
}

/// Number of full-batch gradient-descent steps.
const STEP_COUNT: usize = 300;

/// Result type used by backend adapters in this example.
type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

/// Computes the mean squared error of the MLP predictions.
fn loss<A>(model: &Mlp<A>, inputs: &A, targets: &A, mean_scale: &A) -> Result<A, ProgramError>
where
    A: Clone + Parameter + Add<Output = A> + Sub<Output = A> + Mul<Output = A> + Dot + Reduce + Tanh,
{
    let residuals = model.forward(inputs)? - targets.clone();
    Ok((residuals.clone() * residuals).reduce(&[0, 1], ReductionKind::Sum) * mean_scale.clone())
}

/// Applies one gradient-descent update to all trainable arrays.
fn gradient_descent_step<A>(model: Mlp<A>, gradients: Mlp<A>, learning_rate: &A) -> Result<Mlp<A>, ProgramError>
where
    A: Clone + Parameter + Mul<Output = A> + Sub<Output = A>,
{
    let structure = model.parameter_structure();
    let gradients = Mlp::from_named_parameters(structure.clone(), gradients.into_named_parameters())?;
    let updates = gradients.map_parameters(|gradient| gradient * learning_rate.clone())?;
    Ok(Mlp::from_parameters(
        structure,
        model.into_parameters().zip(updates.into_parameters()).map(|(parameter, update)| parameter - update),
    )?)
}

/// Trains an MLP using a backend adapter only for host value materialization.
fn train<A, ReadValues>(
    backend: &str,
    mut model: Mlp<A>,
    inputs: A,
    targets: A,
    learning_rate: A,
    mean_scale: A,
    mut read_values: ReadValues,
) -> ExampleResult<()>
where
    A: Clone
        + Parameter
        + Value<Type: DifferentiableType, ExecutionDomain: ReverseModeDifferentiate + Zero<A>>
        + Add<Output = A>
        + Sub<Output = A>
        + Mul<Output = A>
        + Dot
        + Reduce
        + Tanh,
    <A::ExecutionDomain as Domain>::Operation: From<OneOperation<A::Type>>,
    LinearizationTracer<A::ExecutionDomain>: Clone
        + Parameter
        + Add<Output = LinearizationTracer<A::ExecutionDomain>>
        + Sub<Output = LinearizationTracer<A::ExecutionDomain>>
        + Mul<Output = LinearizationTracer<A::ExecutionDomain>>
        + Dot
        + Reduce
        + Tanh,
    ReadValues: FnMut(&A) -> ExampleResult<Vec<f64>>,
{
    let mut initial_loss = None;
    let mut final_loss = 0.0;

    for step in 0..STEP_COUNT {
        let (step_loss, gradients) = value_and_gradient(
            |model| loss_with_captured_arguments(&model, inputs.clone(), targets.clone(), mean_scale.clone()),
            model.clone(),
        )?;
        final_loss =
            read_values(&step_loss)?.first().copied().ok_or_else(|| format!("{backend} loss has no values"))?;
        initial_loss.get_or_insert(final_loss);
        if step % 50 == 0 || step + 1 == STEP_COUNT {
            println!("{backend} step {step:>3}: loss = {final_loss:.6}");
        }
        model = gradient_descent_step(model, gradients, &learning_rate)?;
    }

    let initial_loss = initial_loss.unwrap();
    if !initial_loss.is_finite() || !final_loss.is_finite() || final_loss >= initial_loss * 0.1 {
        return Err(
            format!("{backend} training did not converge: loss changed from {initial_loss} to {final_loss}").into()
        );
    }
    println!("{backend} predictions: {:?}", read_values(&model.forward(&inputs)?)?);
    Ok(())
}

/// Returns deterministic layer dimensions, weights, and optional biases shared by both backends.
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

/// Runs MLP training with the `ryft-core` reference CPU array backend.
fn run_core() -> ExampleResult<()> {
    let model = Mlp {
        layers: initial_layer_values()
            .into_iter()
            .map(|(input_size, output_size, weights, bias)| {
                Linear::new(Array::matrix(input_size, output_size, weights), bias.map(Array::vector))
            })
            .collect(),
    };
    let inputs = Array::matrix(4, 2, input_values());
    let targets = Array::matrix(4, 1, target_values());
    let learning_rate = Array::scalar(0.1_f32);
    let mean_scale = Array::scalar(0.25_f32);
    train("core", model, inputs, targets, learning_rate, mean_scale, |array| Ok(array.to_f64s()))
}

#[cfg(feature = "xla")]
mod xla_backend {
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use ryft::pjrt::{Client, ClientOptions, CpuClientOptions, Error as PjrtError, Plugin, load_cpu_plugin};
    #[cfg(any(feature = "cuda-12", feature = "cuda-13"))]
    use ryft::pjrt::{GpuClientOptions, GpuMemoryAllocator, GpuPlatform};
    use ryft::xla::{Array, FromPjrt};
    use ryft::{
        ArrayType, DataType, Device, DeviceMesh, Dimension, LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding,
    };

    use super::{ExampleResult, Linear, Mlp, initial_layer_values, input_values, target_values};

    /// Converts a slice of `f32` values into native-endian host bytes for PJRT transfer.
    fn values_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    /// Constructs a replicated `f32` array on the selected XLA device.
    fn array<'c>(
        client: &'c Client<'c>,
        mesh: &DeviceMesh,
        dimensions: &[usize],
        values: &[f32],
    ) -> ExampleResult<Array<'c>> {
        let shape = Shape::new(dimensions.iter().copied().map(Dimension::Static).collect());
        let r#type = ArrayType::new(DataType::F32, shape)
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), dimensions.len()))?;
        Ok(Array::from_host_buffer(client, r#type, mesh.clone(), values_to_bytes(values))?)
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

        Ok((load_cpu_plugin()?, ClientOptions::CPU(CpuClientOptions { device_count: Some(1) })))
    }

    /// Runs MLP training through XLA on the selected PJRT platform.
    pub(super) fn run() -> ExampleResult<()> {
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

        let model = Mlp {
            layers: initial_layer_values()
                .into_iter()
                .map(|(input_size, output_size, weights, bias)| -> ExampleResult<_> {
                    Ok(Linear::new(
                        array(&client, &mesh, &[input_size, output_size], &weights)?,
                        bias.map(|bias| array(&client, &mesh, &[output_size], &bias)).transpose()?,
                    ))
                })
                .collect::<Result<_, _>>()?,
        };
        let inputs = array(&client, &mesh, &[4, 2], &input_values())?;
        let targets = array(&client, &mesh, &[4, 1], &target_values())?;
        let learning_rate = array(&client, &mesh, &[], &[0.1])?;
        let mean_scale = array(&client, &mesh, &[], &[0.25])?;
        super::train("xla", model, inputs, targets, learning_rate, mean_scale, |array| {
            Ok(read_f32s(array)?.into_iter().map(f64::from).collect())
        })
    }
}

/// Selects and runs the requested backend.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    match std::env::args().nth(1).as_deref().unwrap_or("core") {
        "core" => run_core(),
        #[cfg(feature = "xla")]
        "xla" => xla_backend::run(),
        #[cfg(not(feature = "xla"))]
        "xla" => Err("the XLA backend requires building this example with the `xla` feature".into()),
        backend => Err(format!("unknown backend `{backend}`; expected `core` or `xla`").into()),
    }
}
